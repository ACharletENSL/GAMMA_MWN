'''
ONE cooling sub-step, over its FULL frequency range, in fast and slow cooling.

A single sub-step is one electron distribution radiating once. Its nu F_nu must be

    4/3                      below nu(gamma_min)
    (3-p)/2                  between nu(gamma_min) and nu(gamma_max)
    monotonic steepening     through the cutoff

and NOTHING else -- no intermediate break. Any extra feature here is upstream of every
summation, every EATS weighting and every hydro choice, so this is the smallest unit the
slow-cooling spectral defect can be localised to.

Context: the summed spectrum overshoots at nu_c in slow cooling (see regimes_repro.py,
substep_slowcool.py). The kernel, the summation, the cell construction and both code versions
are all excluded. The user's reading of the sub-step decomposition is that individual sub-steps
already show an apparent break above their peak in slow cooling, which -- if true -- is the
root of it.

NB the segment is much wider than one expects at slow cooling: gamma_max/gamma_min reaches
2.6e5 there (5.4 decades in gamma = 10.8 in nu) against ~600 (2.8 decades) at fast cooling, so
a window of a few dex around the peak does NOT show the whole shape. Plot the lot.

  python -c "import onestep_shape as O; O.main()"
'''

import os
import numpy as np
import matplotlib.pyplot as plt

from environment import GAMMA_dir, MyEnv
from working_cooling import (generate_cell_withDistrib, precompute_step_cols, step_view,
                             norm_plaw_distrib, _midpoint_cell)
from radiation_cooling import get_Fnu_step, _sval
from IO import open_celldata, open_rundata
from fits_hydro import cellsBehindShock_fromData
from sweep_gammacm import compute_alpha_sweep
import prerar_model as M

KEY, Z = 'cooling_g100', 4
OUTDIR = os.path.join(GAMMA_dir, 'bin', 'Tools', 'figures')


def main(Ri_target=1.05, logrs=(-4., 3.), steps=(0, 3, 10), key=KEY, z=Z, outdir=OUTDIR):
  cells = M._test_cells(key, z, 120)
  k = int(min(cells, key=lambda c: abs(c[4] - Ri_target))[0])
  raw = open_celldata(key, k)
  shr = cellsBehindShock_fromData(open_rundata(key, z))
  d0 = shr.loc[shr.i == raw.iloc[0].i].iloc[0]
  ex = shr.loc[shr.t.idxmax()]
  nu = np.logspace(-10, 16, 3000)

  fig, axes = plt.subplots(2, len(logrs), figsize=(7.0*len(logrs), 8.6),
                           layout='constrained')
  axes = np.atleast_2d(axes)
  for c, logr in enumerate(logrs):
    alpha = float(compute_alpha_sweep(key, np.array([float(logr)]))[0][0])
    cell, env = generate_cell_withDistrib(raw, d0, MyEnv(key), alpha=alpha, r_ref=1.2,
                    Tmax=None, key=key, k=k, exit_row=ex)
    p = env.psyn
    c0 = cell.iloc[0]
    K0 = norm_plaw_distrib(c0.gmin, c0.gmax, p)
    cm = _midpoint_cell(cell)
    cols = precompute_step_cols(cm, env)
    gmn = cm.gmin.to_numpy(float); gmx = cm.gmax.to_numpy(float)
    nupB = cols['nup_B']
    print(f'=== log10(gc/gm)={logr:+.0f}, p={p} ===')
    for j, col in zip(steps, ('#0072B2', '#009E73', '#D55E00')):
      st = step_view(cols, j)
      s = np.asarray(get_Fnu_step(nu*env.nu0, float(env.Ts), st, K0, env, 20, True, 1.1),
                     float)*nu
      s = np.where(np.isfinite(s) & (s > 0), s, np.nan)
      if not np.isfinite(np.nansum(s)) or np.nansum(s) <= 0:
        continue
      lab = rf'step {j}, $\gamma_{{\max}}/\gamma_{{\min}}$={gmx[j]/gmn[j]:.3g}'
      axes[0, c].loglog(nu, s/np.nanmax(s), color=col, lw=1.5, label=lab)
      g = np.gradient(np.log10(s), np.log10(nu))
      axes[1, c].semilogx(nu, g, color=col, lw=1.5)
      # the two frequencies the distribution's own edges predict
      for gm, ls in ((gmn[j], ':'), (gmx[j], '--')):
        axes[1, c].axvline(gm**2*nupB[j]/env.nu0, color=col, ls=ls, lw=.8, alpha=.6)
      ln = np.log10(nu); jp = int(np.nanargmax(s))
      lad = [np.nanmedian(g[(ln-ln[jp] >= x-.25) & (ln-ln[jp] < x+.25)])
             for x in (-8., -6., -4., -2., -1., 0.5, 1., 2.)]
      print(f'  step {j:2d}  slope at -8,-6,-4,-2,-1 | +0.5,+1,+2 dex from peak: ' +
            ' '.join(f'{v:+.2f}' for v in lad))
    for y, t in ((4./3., '4/3'), ((3.-p)/2., '(3-p)/2')):
      axes[1, c].axhline(y, color='grey', lw=.9, ls=':')
      axes[1, c].annotate(t, xy=(1.005, y), xycoords=('axes fraction', 'data'),
                          fontsize=8, color='grey', va='center')
    axes[0, c].set(title=rf'$\log_{{10}}(\gamma_c/\gamma_m)$ = {logr:+.0f}',
                   ylabel=r'$\nu F_\nu$ (per sub-step, peak-normalised)', ylim=(1e-8, 3.))
    axes[0, c].legend(fontsize=8)
    axes[1, c].set(xlabel=r'$\nu/\nu_m$', ylabel=r'$d\log(\nu F_\nu)/d\log\nu$',
                   ylim=(-2.2, 1.8))
    for ax in (axes[0, c], axes[1, c]):
      ax.grid(alpha=.25)
  fig.suptitle(f'ONE sub-step, full range, cell k={k}  '
               r'(dotted/dashed = $\nu(\gamma_{\min})$ / $\nu(\gamma_{\max})$ of that step)'
               '\na single distribution may only show 4/3, (3-p)/2, then a cutoff',
               fontsize=11)
  path = os.path.join(outdir, 'onestep_shape.png')
  fig.savefig(path, dpi=150, bbox_inches='tight')
  plt.close(fig)
  print(f'saved {path}')


if __name__ == '__main__':
  main()
