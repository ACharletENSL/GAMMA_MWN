# Replot ONLY the figures whose shape-class label went from MC to MFC (64fd56b, 1d3c825).
# Everything is read back from the sweep caches and the tables already on disk -- nothing
# is recomputed. REPLOT_KEY selects the run (default the fiducial).
#   PYTHONPATH=. python3 replot_mfc.py
import os, re, glob, time
import matplotlib; matplotlib.use('Agg')

KEY = os.environ.get('REPLOT_KEY', 'cooling_g100')
from environment import figdir
import sweep_gammacm as swp
import spectrum_shape as ss
import slope_validation as sv


def _trim(paths):
  swp.trim_pngs([p for p in paths if os.path.exists(p)])


def _step(name):
  print(f'\n=== {name} ({time.strftime("%H:%M:%S")}) ===', flush=True)


def main():
  print(f'=== replotting the MC -> MFC figures for key={KEY!r} ===', flush=True)

  # 1. per sweep: spectrum_evolution legend (article), break_panels + break_ratio_evolution
  #    (the dashed-style key), the gs02_fits_table png
  for m in ('data_rarcut', 'data'):
    for z in (4, 1):
      d = swp.method_outdir(m, KEY, z)
      res = swp.load_sweep(d) if os.path.isdir(d) else None
      if not res:
        print(f'-- {d}: no cached sweep, skipped'); continue
      _step(f'{m} z={z}: {len(res)} points')
      barT_f = swp.exit_onset_barT(KEY, z=z)
      barT_off = swp.rarefaction_off_barT(KEY, z=z)
      swp.plot_spectra_per_regime(res, barT_f, outdir=d)
      # the same swap bound as sweep_gammacm.main, so the tracks are the ones drawn there
      tracks = swp.route_break_tracks(res, KEY, m, z, barT_off=barT_off, barT_f=barT_f)
      fits = [swp.fit_break_evolution(r, tr, barT_f, barT_off) for r, tr in zip(res, tracks)]
      swp.plot_break_ratio(res, tracks, barT_f, barT_off=barT_off, outdir=d)
      swp.plot_break_panels(res, tracks, fits, barT_f, barT_off=barT_off, outdir=d)
      dets = [swp.detect_rise_peak_tail(r['Tb'], r['nub'], r['nuFnu']) for r in res]
      swp.build_gs02_table(res, dets, outdir=d)
      _trim(glob.glob(os.path.join(d, 'spectrum_evolution_logr=*.png'))
            + [os.path.join(d, n) for n in ('break_panels.png', 'break_ratio_evolution.png',
                                            'gs02_fits_table.png')])
      swp.copy_article_figures(d)

  # 2. spectrum_shape: both table pngs, redrawn from their own csvs
  d = swp.method_outdir(ss.DEFAULT_METHOD, KEY, ss.Z_RS)
  _step(f'spectrum_shape tables in {d}')
  try:
    _trim(ss.replot_tables(d))
  except (FileNotFoundError, AssertionError) as e:
    print(f'-- skipped: {e}')

  # 3. slope_validation: the worst-bin panels, titled by class
  out = figdir(sv.OUTDIR_NAME, KEY)
  if os.path.isdir(out):
    _step(f'slope_validation prescription_worst in {out}')
    sides = [sv.load_side(z=z, key=KEY) for z in (sv.Z_RS, sv.Z_FS)]
    summary = sv.prescription_summary(sides, verbose=False)
    _trim([sv.plot_prescription(sides, summary, outdir=out)])
  else:
    print(f'-- {out}: absent, skipped')

  # 4. ravasio_2sbpl examples (subtitle names the class). Its folder is not keyed: the
  #    fiducial only, and only the examples already there
  if KEY == 'cooling_g100':
    import ravasio_2sbpl as rv
    for f in sorted(glob.glob(os.path.join(rv.OUTDIR, 'example_*.png'))):
      mt = re.match(r'example_(\w+)_z=(\d)_logr=([+-]\d+)_logt=([+-][\d.]+)\.png',
                    os.path.basename(f))
      if not mt:
        continue
      _step(os.path.basename(f))
      cfg, z, logr, logt = mt.group(1), int(mt.group(2)), float(mt.group(3)), float(mt.group(4))
      rv.plot_example(key=KEY, z=z, logr=logr, logt=logt, config=cfg, outdir=rv.OUTDIR)
      _trim([f])
  print('\n=== done ===', flush=True)


if __name__ == '__main__':
  main()
