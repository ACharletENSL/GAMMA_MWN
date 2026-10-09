# Replot the C-sweep article figures after the colour-map change (jet -> SWEEP_CMAP, plasma
# cut at 0.88) and the label/layout edits of the same commit (Fig. 6 T_rise/T_fall, the
# lightcurve_shape_cmp difference label and hspace). Everything is read back from the sweep
# caches and the tables already on disk; the only fits redone are the GS02 break tracks
# behind break_panels (route caches reused, ~30 s a point). REPLOT_KEY selects the run
# (default the fiducial), REPLOT_SHELLS the shells (default both).
#   PYTHONPATH=. python3 replot_colours.py
import os, glob, time
import matplotlib; matplotlib.use('Agg')

KEY = os.environ.get('REPLOT_KEY', 'cooling_g100')
SHELLS = tuple(int(z) for z in os.environ.get('REPLOT_SHELLS', '4,1').split(','))
from environment import figdir
import sweep_gammacm as swp
import sweep_compare as cmp
import sweep_rarcut as rc
import spectrum_shape as ss
import lightcurve_shape as ls
import mid_slope_evolution as mse

METHOD = 'data_rarcut'     # the article's Sect. 4 sweep (sweep_gammacm.DEFAULT_METHOD)


def _trim(paths):
  swp.trim_pngs([p for p in paths if os.path.exists(p)])


def _step(name):
  print(f'\n=== {name} ({time.strftime("%H:%M:%S")}) ===', flush=True)


def main():
  print(f'=== replotting the C-sweep figures for key={KEY!r}, shells {SHELLS} ===', flush=True)
  for z in SHELLS:
    d = swp.method_outdir(METHOD, KEY, z)
    res = swp.load_sweep(d)
    if not res:
      print(f'-- {d}: no cached sweep, skipped'); continue
    barT_f = swp.exit_onset_barT(KEY, z=z)
    barT_off = swp.rarefaction_off_barT(KEY, z=z)

    # Fig. 5: lightcurve_shape_nu=*; Fig. 8: spectrum_evolution_logr=*; Fig. 9: break_panels
    _step(f'{METHOD} z={z}: lightcurves, spectral evolution, break panels')
    swp.plot_lightcurve_shape(res, barT_f, barT_off=barT_off, outdir=d)
    swp.plot_spectra_per_regime(res, barT_f, outdir=d,
                                three_from=swp.three_break_from(METHOD, barT_f),
                                ref=swp.post_rf_reference(METHOD, KEY, z))
    tracks = swp.route_break_tracks(res, KEY, METHOD, z, barT_off=barT_off, barT_f=barT_f)
    fits = [swp.fit_break_evolution(r, tr, barT_f, barT_off) for r, tr in zip(res, tracks)]
    swp.plot_break_panels(res, tracks, fits, barT_f, barT_off=barT_off, outdir=d)
    _trim(glob.glob(os.path.join(d, 'lightcurve_shape_nu=*.png'))
          + glob.glob(os.path.join(d, 'spectrum_evolution_logr=*.png'))
          + [os.path.join(d, 'break_panels.png')])

    # Fig. 6: pulse_characteristics_vs_nu (+ the lightcurve_shape tables), from its csvs
    _step(f'{METHOD} z={z}: pulse characteristics')
    ls.replot(key=KEY, method=METHOD, z=z, outdir=d)
    _trim(glob.glob(os.path.join(d, 'pulse_characteristics_vs_nu*.png')))

    # Fig. B.1: mid_slope_evolution, from mid_slopes.csv
    _step(f'{METHOD} z={z}: mid-slope evolution')
    mse.main(key=KEY, method=METHOD, z=z, outdir=d, use_cache=True)
    swp.copy_article_figures(d)

    # Sect. 5 figures (rarcut_compare): lightcurve_shape_cmp_nu=*, postrf_spectra_all,
    # fluence_spectra_slope_norm-eff
    out = figdir(rc.OUTDIR_NAME if z == swp.Z_SHELL else f'{rc.OUTDIR_NAME}_z={z}', KEY)
    _step(f'{out}: post-rarefaction comparison')
    pairs = cmp.load_pairs(swp.method_outdir(rc.METHOD_A, KEY, z),
                           swp.method_outdir(rc.METHOD_B, KEY, z))
    cmp.plot_postrf_spectra(pairs, barT_f, outdir=out)
    cmp.plot_lightcurve_shape_compare(pairs, barT_f, barT_off=barT_off, outdir=out,
                                      labels=rc.LABELS, norm_side=rc.NORM_SIDE)
    series = cmp.fluence_series(pairs)
    cmp.plot_fluence_with_slope(pairs, series, outdir=out, labels=rc.LABELS,
                                norm_side=rc.NORM_SIDE)
    _trim(glob.glob(os.path.join(out, 'lightcurve_shape_cmp_nu=*.png'))
          + glob.glob(os.path.join(out, 'postrf_spectra*.png'))
          + glob.glob(os.path.join(out, 'fluence_spectra_slope*.png')))
    swp.copy_article_figures(out)

  # Fig. 7: spectrum_shape_spectra_ratios_RS (measures both shells, ~20 s)
  _step('spectrum_shape')
  ss.main(key=KEY, method=METHOD)
  print('\n=== done ===')


if __name__ == '__main__':
  main()
