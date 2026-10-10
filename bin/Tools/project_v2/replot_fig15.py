# Fig. 15 only (rarcut_compare fluence_spectra_slope_norm-eff), redrawn from the sweep caches in
# ~5 s -- the label-only route (no regen). REPLOT_KEY selects the run. Mirrors into article_choice.
#   PYTHONPATH=. python3 replot_fig15.py
import os, glob, matplotlib; matplotlib.use('Agg')
KEY = os.environ.get('REPLOT_KEY', 'cooling_g100')
from environment import figdir
import sweep_gammacm as swp, sweep_compare as cmp, sweep_rarcut as rc
z = swp.Z_SHELL
out = figdir(rc.OUTDIR_NAME, KEY)
pairs = cmp.load_pairs(swp.method_outdir(rc.METHOD_A, KEY, z), swp.method_outdir(rc.METHOD_B, KEY, z))
series = cmp.fluence_series(pairs)
cmp.plot_fluence_with_slope(pairs, series, outdir=out, labels=rc.LABELS, norm_side=rc.NORM_SIDE)
p = os.path.join(out, 'fluence_spectra_slope_norm-eff.png')
swp.trim_pngs([p])
swp.copy_article_figures(out)
print("wrote", p)
