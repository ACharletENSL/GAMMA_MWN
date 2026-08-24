'''
The `test_cooling_1hydro.png` decomposition, redone in SLOW cooling: every cooling sub-step
drawn separately against their sum, plus the slope of the sum.

WHY: in slow cooling the summed nu F_nu slope goes 4/3 -> +(3-p)/2 -> OVERSHOOTS to ~-0.43 ->
settles at -(p-2)/2, and the overshoot sits exactly at nu_c = (gamma_c/gamma_m)^2 nu_m. Both
GS02 segments come out right; only the break between them is wrong, and a smooth broken power
law (GS02 eq. 1, s ~ 1.0 there) cannot overshoot. The kernel, the summation and the cell
construction are all excluded -- see regimes_repro.py and the commit that added it.

So the question this figure answers is: WHICH sub-steps sit under the overshoot, and what is
different about them? The original fast-cooling version of this plot
(figures/test_cooling_1hydro.png) shows the sub-step spectra marching down in frequency and
summing to a clean broken power law. Here the same construction is drawn where it fails.

Read it as: if the overshoot is a GAP in coverage, there will be a frequency band around nu_c
where no sub-step peaks and the envelope sags between neighbours. If it is a WEIGHTING problem,
the sub-steps will cover the band but the ones there will be too faint.

  python -c "import substep_slowcool as S; S.main()"           # log10(gc/gm) = +3
  python -c "import substep_slowcool as S; S.main(logr=1.)"
  python -c "import substep_slowcool as S; S.main(logr=-4.)"   # the clean fast-cooling control
'''

import os
import numpy as np
import matplotlib.pyplot as plt

from environment import GAMMA_dir, MyEnv
import working_cooling_data as W
from working_cooling import (generate_cell_withDistrib, precompute_step_cols, step_view,
                             norm_plaw_distrib, _midpoint_cell)
from radiation_cooling import get_Fnu_step, _sval
from IO import open_celldata, open_rundata
from fits_hydro import cellsBehindShock_fromData
from sweep_gammacm import compute_alpha_sweep
import prerar_model as M

KEY, Z = 'cooling_g100', 4
# all the nu_c-dip diagnostics land in one folder so they are not scattered
OUTDIR = os.path.join(GAMMA_dir, 'bin', 'Tools', 'figures', 'nuc_dip')
NG = 200      # converged electron quadrature, as onestep_shape.py uses. The flux path still
              # defaults to Ng=20, which is 12.9% (fast) / 48.9% (slow) off and roughens the
              # slope at the turnover by 2.8-3.9x -- noise this figure should not inherit.


def main(Ri_target=1.05, logr=3., key=KEY, z=Z, outdir=OUTDIR, Nnu=900,
    lognu=(-6., 12.), nmax_draw=60):
  cells = M._test_cells(key, z, 120)
  k = int(min(cells, key=lambda c: abs(c[4] - Ri_target))[0])
  raw = open_celldata(key, k)
  sh_run = cellsBehindShock_fromData(open_rundata(key, z))
  cell_d0 = sh_run.loc[sh_run.i == raw.iloc[0].i].iloc[0]
  exit_row = sh_run.loc[sh_run.t.idxmax()]
  alpha = float(compute_alpha_sweep(key, np.array([float(logr)]))[0][0])
  cell, env = generate_cell_withDistrib(raw, cell_d0, MyEnv(key), alpha=alpha,
                  r_ref=1.2, Tmax=None, key=key, k=k, exit_row=exit_row)
  if cell is False:
    print('cell rejected'); return None
  p = env.psyn
  c0 = cell.iloc[0]
  K0 = norm_plaw_distrib(c0.gmin, c0.gmax, p)
  cm = _midpoint_cell(cell)
  cols = precompute_step_cols(cm, env)
  N = len(cell)
  nu = np.logspace(*lognu, Nnu)
  nuobs = nu*env.nu0
  # ONE observer time, but it must be one where the sub-steps have ARRIVED. env.Ts (what the
  # old path used) is NOT: measured tT there spans 0.27..0.94, i.e. before every step's own
  # on-axis arrival, and get_Fnu_step now correctly returns zero for tT<1. Use max(Ton).
  Tobs = float(cols['obsT'][0].max())

  # per sub-step spectra, exactly as get_cell_nuFnu sums them
  S = np.zeros((N, Nnu))
  for j in range(N):
    S[j] = np.asarray(get_Fnu_step(nuobs, float(Tobs), step_view(cols, j), K0, env,
                                   NG, True, 1.1), float)*nu
  S = np.where(np.isfinite(S), S, 0.)
  tot = S.sum(axis=0)
  live = np.flatnonzero(S.max(axis=1) > tot.max()*1e-12)
  gmn = cm.gmin.to_numpy(float); gmx = cm.gmax.to_numpy(float)

  # where is nu_c? gamma_c/gamma_m = 10**logr, so nu_c/nu_m = (gc/gm)^2
  nu_c = 10.**(2.*logr)

  fig, axes = plt.subplots(2, 1, figsize=(8.2, 9.6), layout='constrained', sharex=True)
  cmap = plt.cm.turbo(np.linspace(0, 1, max(len(live), 1)))
  step = max(1, len(live)//nmax_draw)
  for c, j in zip(cmap[::step], live[::step]):
    m = S[j] > 0
    if m.sum() > 2:
      axes[0].loglog(nu[m], S[j][m]/tot.max(), color=c, lw=.8, alpha=.85)
  m = tot > 0
  axes[0].loglog(nu[m], tot[m]/tot.max(), color='k', lw=2., label='sum over sub-steps')
  axes[0].axvline(nu_c, color='crimson', ls='--', lw=1.2,
                  label=rf'$\nu_c/\nu_m=(\gamma_c/\gamma_m)^2$')
  axes[0].axvline(1., color='grey', ls=':', lw=.9)
  axes[0].set(ylabel=r'$\nu F_\nu$ (normalised to the sum peak)', ylim=(1e-7, 3.))
  axes[0].legend(fontsize=9)

  ln = np.log10(nu)
  g = np.gradient(np.log10(np.where(m, tot, np.nan)), ln)
  axes[1].semilogx(nu, g, color='k', lw=1.8)
  for y, lab in ((4./3., '4/3'), (0.5, '1/2'), ((3.-p)/2., '(3-p)/2'), (-(p-2.)/2., '1-p/2')):
    axes[1].axhline(y, color='grey', lw=.9, ls=':')
    axes[1].annotate(lab, xy=(1.005, y), xycoords=('axes fraction', 'data'),
                     fontsize=8, color='grey', va='center')
  axes[1].axvline(nu_c, color='crimson', ls='--', lw=1.2)
  axes[1].axvline(1., color='grey', ls=':', lw=.9)
  axes[1].set(xlabel=r'$\nu/\nu_m$', ylabel=r'$d\log(\nu F_\nu)/d\log\nu$', ylim=(-1.6, 1.8))
  for ax in axes:
    ax.grid(alpha=.25)
  fig.suptitle(f'sub-step decomposition, cell k={k}, '
               rf'$\log_{{10}}(\gamma_c/\gamma_m)$={logr:+.0f}, p={p}'
               f'\n{len(live)} contributing sub-steps at $T_{{\\rm obs}}=T_s$',
               fontsize=11)
  path = os.path.join(outdir, f'substep_slowcool_logr{logr:+.0f}.png')
  fig.savefig(path, dpi=150, bbox_inches='tight')
  plt.close(fig)
  print(f'saved {path}')

  # where each sub-step peaks, against where the overshoot is
  jp_tot = int(np.nanargmax(tot))
  print(f'\ncell k={k}, log10(gc/gm)={logr:+.0f}, p={p}, {N} sub-steps ({len(live)} live)')
  print(f'  sum peaks at nu/nu_m = {nu[jp_tot]:.3g};  nu_c/nu_m = {nu_c:.3g}')
  pk = np.array([nu[int(np.argmax(S[j]))] if S[j].max() > 0 else np.nan for j in range(N)])
  print(f'  sub-step peak frequencies span {np.nanmin(pk):.3g} .. {np.nanmax(pk):.3g}')
  # coverage: how many sub-steps peak per decade, around nu_c
  edges = 10.**np.arange(np.floor(np.log10(nu_c))-3, np.floor(np.log10(nu_c))+4)
  print(f'  {"decade":>26}  {"n sub-steps peaking":>19}  {"summed peak flux":>17}')
  for lo, hi in zip(edges[:-1], edges[1:]):
    sel = (pk >= lo) & (pk < hi)
    fl = S[sel].max(axis=1).sum() if sel.any() else 0.
    mark = '   <- nu_c' if (lo <= nu_c < hi) else ''
    print(f'  {lo:11.3g} .. {hi:9.3g}  {int(sel.sum()):>19}  {fl/tot.max():17.4e}{mark}')
  return dict(nu=nu, S=S, tot=tot, cell=cell, env=env)


if __name__ == '__main__':
  main()
