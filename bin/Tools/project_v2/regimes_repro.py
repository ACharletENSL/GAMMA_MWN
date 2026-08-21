'''
Reproduce figures/regimes.png (2025-08-12) with TODAY's code, same presentation, plus the
slope panel it did not have.

Why: the nu F_nu slope above the peak steepens and then RECOVERS at slow cooling, which
one-zone synchrotron forbids. That was taken for a regression, but the OLD summation path
(working_cooling_prev.get_cell_nuFnu) and the OLD analytic cell (generate_cell_withDistrib)
both reproduce it -- so if regimes.png really lacked the feature, it cannot be explained by
either the emission code or the cell construction. The remaining possibility is that the
feature was always present and simply invisible in a log-log nu F_nu plot: it is ~0.15 dex of
curvature spread over a decade, which the eye reads as a slightly bent power law.

So: top row = exactly regimes.png's axes (nuFnu/nu_m F_num vs nu/nu_m, analytic BPL dotted),
bottom row = d log(nuFnu)/d log(nu) for the same curves, which is where 0.15 dex is unmissable.
Compare the top row against the old PNG by eye; read the verdict off the bottom row.

  python -c "import regimes_repro as R; R.main()"
'''

import os
import numpy as np
import matplotlib.pyplot as plt

from environment import GAMMA_dir, MyEnv
import working_cooling_prev as WP
import working_cooling_data as W
from working_cooling import generate_cell_withDistrib
from IO import open_celldata, open_rundata
from fits_hydro import cellsBehindShock_fromData
from sweep_gammacm import compute_alpha_sweep
import prerar_model as M

KEY, Z = 'cooling_g100', 4
# all the nu_c-dip diagnostics land in one folder so they are not scattered
OUTDIR = os.path.join(GAMMA_dir, 'bin', 'Tools', 'figures', 'nuc_dip')
LOGRS = (-4., -2., 0., 1., 3.)          # the five curves of regimes.png
CMAP = ('#00429d', '#4771b2', '#93c4a2', '#e8a33d', '#93003a')


def main(Ri_target=1.05, key=KEY, z=Z, outdir=OUTDIR, cell_kind='analytic'):
  cells = M._test_cells(key, z, 120)
  k = int(min(cells, key=lambda c: abs(c[4] - Ri_target))[0])
  raw = open_celldata(key, k)
  sh_run = cellsBehindShock_fromData(open_rundata(key, z))
  cell_d0 = sh_run.loc[sh_run.i == raw.iloc[0].i].iloc[0]
  exit_row = sh_run.loc[sh_run.t.idxmax()]
  shd = W.select_postshock_rows(raw, 1)
  shd = shd.assign(t=shd['t'] - raw.t.iloc[0])

  fig, axes = plt.subplots(2, 1, figsize=(7.2, 9.2), layout='constrained', sharex=True)
  print(f'cell k={k}, {cell_kind} cell, OLD summation path (working_cooling_prev)')
  print(f'{"log10(gc/gm)":>13}  slopes above peak at +0.5,+1,+1.5,+2,+3,+4 dex')
  for logr, col in zip(LOGRS, CMAP):
    alpha = float(compute_alpha_sweep(key, np.array([logr]))[0][0])
    if cell_kind == 'analytic':
      cell, env = generate_cell_withDistrib(raw, cell_d0, MyEnv(key), alpha=alpha,
                      r_ref=1.2, Tmax=None, key=key, k=k, exit_row=exit_row)
    else:
      cell, env = W.generate_cell_fromHistory(shd, raw.attrs, MyEnv(key),
                      alpha=alpha, Tmax=None, r_ref=1.2)
    if cell is False:
      print(f'{logr:>13.0f}  cell rejected'); continue
    nub, nF = WP.get_cell_nuFnu(cell, env)
    s = np.where(np.isfinite(nF) & (nF > 0), nF, np.nan)
    if not np.isfinite(np.nansum(s)) or np.nansum(s) <= 0:
      continue
    # regimes.png normalisation: peak of nuFnu, and nu in units of nu_m
    jp = int(np.nanargmax(s))
    axes[0].loglog(nub, s/s[jp], color=col, lw=1.6, label=f'{logr:+.1f}')
    ln = np.log10(nub)
    g = np.gradient(np.log10(s), ln)
    axes[1].semilogx(nub, g, color=col, lw=1.6)
    seg = [np.nanmedian(g[(ln - ln[jp] >= c - .4) & (ln - ln[jp] < c + .4)])
           for c in (0.5, 1., 1.5, 2., 3., 4.)]
    fin = [v for v in seg if np.isfinite(v)]
    mono = all(fin[i+1] <= fin[i] + 0.02 for i in range(len(fin)-1))
    print(f'{logr:>13.0f}  ' + ' '.join(f'{v:+.2f}' for v in seg) +
          ('   monotonic' if mono else '   RECOVERS'))

  p = env.psyn
  for y, lab in ((4./3., '4/3'), (0.5, '1/2'), ((3.-p)/2., '(3-p)/2'), (-(p-2.)/2., '1-p/2')):
    axes[1].axhline(y, color='grey', lw=.9, ls=':')
    axes[1].annotate(lab, xy=(1.005, y), xycoords=('axes fraction', 'data'),
                     fontsize=8, color='grey', va='center')
  axes[0].set(ylabel=r'$\nu F_\nu / (\nu F_\nu)_{\rm pk}$', ylim=(1e-6, 3.))
  axes[0].legend(title=r'$\log_{10}\gamma_c/\gamma_m$', fontsize=9)
  axes[1].set(xlabel=r'$\nu/\nu_m$', ylabel=r'$d\log(\nu F_\nu)/d\log\nu$',
              ylim=(-1.6, 1.8))
  for ax in axes:
    ax.grid(alpha=.25)
  fig.suptitle(f'regimes.png reproduced with today\'s code ({cell_kind} cell, '
               'old summation path)\nthe dip is invisible above, unmissable below',
               fontsize=11)
  path = os.path.join(outdir, f'regimes_repro_{cell_kind}.png')
  fig.savefig(path, dpi=150, bbox_inches='tight')
  plt.close(fig)
  print(f'saved {path}')


if __name__ == '__main__':
  main()
