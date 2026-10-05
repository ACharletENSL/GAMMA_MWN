# Redraw ONLY the break figures (break_evolution, break_ratio_evolution, break_panels,
# break_frequencies, break_evolution_table, and the spectrum_evolution series that carries
# the same segment identification) from the sweep caches, with the route-gated tracker
# (sweep_gammacm.track_breaks_route, 73d01b0). The segment-route tracks are cached per point
# (segment_route.load_side); a cold side costs ~7 min on 3 workers at the fiducial. No sweep
# point is recomputed. REPLOT_KEY selects the run (default the fiducial).
#   PYTHONPATH=. python3 replot_breaks.py [data_rarcut:4 data:4 ...]
import os, sys, time
import matplotlib; matplotlib.use('Agg')

KEY = os.environ.get('REPLOT_KEY', 'cooling_g100')
import sweep_gammacm as swp

SIDES = (('data_rarcut', 4), ('data', 4), ('data_rarcut', 1), ('data', 1))
NAMES = ('break_evolution.png', 'break_ratio_evolution.png', 'break_panels.png',
         'break_evolution_table.png', 'break_frequencies.png')


def replot_side(method, z, nproc=None):
  d = swp.method_outdir(method, KEY, z)
  res = swp.load_sweep(d) if os.path.isdir(d) else None
  if not res:
    print(f'-- {d}: no cached sweep, skipped', flush=True)
    return
  t = time.time()
  res = sorted(res, key=lambda q: q['log10ratio'])
  barT_f = swp.exit_onset_barT(KEY, z=z)
  barT_off = swp.rarefaction_off_barT(KEY, z=z)
  tracks = swp.route_break_tracks(res, KEY, method, z, barT_off=barT_off, barT_f=barT_f,
                                  nproc=nproc)
  fits = [swp.fit_break_evolution(r, tr, barT_f, barT_off) for r, tr in zip(res, tracks)]
  c25 = swp.c25_num_curve(KEY, z, res[0]['Tb'])
  swp.plot_break_evolution(res, tracks, fits, barT_f, barT_off=barT_off, outdir=d)
  swp.plot_break_ratio(res, tracks, barT_f, barT_off=barT_off, outdir=d)
  swp.plot_break_frequencies(res, tracks, barT_f, barT_off=barT_off, outdir=d)
  # the spectra carry the same identification past crossing (four segments where they exist)
  swp.plot_spectra_per_regime(res, barT_f, outdir=d,
                              three_from=swp.three_break_from(method, barT_f),
                              ref=swp.post_rf_reference(method, KEY, z))
  swp.plot_break_panels(res, tracks, fits, barT_f, barT_off=barT_off, outdir=d)
  swp.build_break_evolution_table(res, fits, outdir=d, tracks=tracks, c25=c25)
  # trim the PATHS, then mirror: the mirror copies what is on disk
  pngs = [os.path.join(d, n) for n in NAMES] \
         + [os.path.join(d, f'spectrum_evolution_logr={r["log10ratio"]:+.1f}.png') for r in res]
  swp.trim_pngs([p for p in pngs if os.path.exists(p)])
  swp.copy_article_figures(d)
  print(f'=== {method} z={z} -> {d} ({time.time() - t:.0f} s)', flush=True)


def main(sides=SIDES, nproc=None):
  print(f'=== break figures for key={KEY!r} ===', flush=True)
  for m, z in sides:
    replot_side(m, z, nproc=nproc)
  print('=== done ===', flush=True)


if __name__ == '__main__':      # the guard is required: the route pool uses forkserver
  args = [(a.split(':')[0], int(a.split(':')[1])) for a in sys.argv[1:]]
  main(args or SIDES)
