'''
ONE cooling sub-step followed through EVERY stage of the emission chain, fast vs slow cooling.

Four rows, one per stage, so the stage at which the shape first goes wrong is visible at a
glance. Each panel carries its quantity (solid, left axis) AND its logarithmic slope (dashed,
right axis) against the analytic values that quantity is allowed to take:

  N(gamma)      the electron distribution, compensated by gamma^p     flat, slope 0
  e'_nu'        comoving emissivity, from get_epnu                    1/3 -> -(p-1)/2 -> cutoff
  L'_nu'        comoving luminosity, = e'_nu' * rsc * V3p/Ttz         SAME shape as e'_nu'
  nu F_nu       observer frame, after D, tT^-2 and the EATS shift     4/3 -> (3-p)/2 -> cutoff

L'_nu' differs from e'_nu' by a per-step CONSTANT, so any change of shape between rows 2 and 3
is a bug by construction; likewise rows 3 -> 4 may only shift and rescale, never re-break.

Context: the summed spectrum overshoots at nu_c in slow cooling and then recovers to -(p-2)/2,
which a smooth broken power law cannot do (GS02 eq. 1). Excluded by direct test so far: the
kernel, the summation (the old working_cooling_prev path reproduces it), the cell construction
(analytic reproduces it), midpoint, bsyn, the band cut, Ng, width_tol, r_ref, R(x) continuity,
the L'_nu' interpolation, and per-flash energy conservation.

NB the slow-cooling segment is much wider than one expects: gamma_max/gamma_min reaches 2.6e5
(5.4 decades in gamma = 10.8 in nu) against ~600 at fast cooling, so a window of a few dex
around the peak does NOT show the whole shape.

  python -c "import onestep_shape as O; O.main()"
'''

import os
import numpy as np
import matplotlib.pyplot as plt

from environment import GAMMA_dir, MyEnv
from working_cooling import (generate_cell_withDistrib, precompute_step_cols, step_view,
                             norm_plaw_distrib, _midpoint_cell)
from radiation_cooling import (get_Fnu_step, get_epnu, get_Lnu_comov, _sval, cooled_tt_eff)
from cooling_distribution import distrib_plaw_cooled
from IO import open_celldata, open_rundata
from fits_hydro import cellsBehindShock_fromData
from sweep_gammacm import compute_alpha_sweep
import prerar_model as M

KEY, Z = 'cooling_g100', 4
# all the nu_c-dip diagnostics land in one folder so they are not scattered
OUTDIR = os.path.join(GAMMA_dir, 'bin', 'Tools', 'figures', 'nuc_dip')
NG = 200          # converged electron quadrature; the flux path's default 20 is 12.9% (fast)
                  # / 48.9% (slow) off and would show as noise on the shape being judged
SLOPE_C = '#999999'


def _slope(x, y):
  s = np.where(np.isfinite(y) & (y > 0), y, np.nan)
  return np.gradient(np.log10(s), np.log10(x))


def _panel(ax, x, y, col, lab, guides, ylab, xlab, slope_ylim, first, ylim=(1e-8, 3.),
    with_slope=True):
  '''quantity on ax (solid), and optionally its slope on a twin axis (dashed).

  with_slope=False for the distribution row: the interesting thing there is the SHAPE of
  N(gamma) itself over a narrow dynamic range, which a slope axis only crowds.
  '''
  m = np.isfinite(y) & (y > 0)
  if m.any():
    ax.loglog(x[m], y[m]/np.nanmax(y[m]), color=col, lw=1.6, label=lab)
  if with_slope:
    tw = getattr(ax, '_slope_twin', None) or ax.twinx()
    ax._slope_twin = tw
    tw.semilogx(x, _slope(x, y), color=col, lw=1.1, ls='--', alpha=.85)
  if first:
    if with_slope:
      for gy, gl in guides:
        tw.axhline(gy, color=SLOPE_C, lw=.9, ls=':')
        tw.annotate(gl, xy=(1.002, gy), xycoords=('axes fraction', 'data'),
                    fontsize=7, color=SLOPE_C, va='center')
      tw.set_ylim(*slope_ylim)
      tw.set_ylabel('slope (dashed)', fontsize=8, color=SLOPE_C)
      tw.tick_params(axis='y', labelsize=7, colors=SLOPE_C)
    ax.set(ylabel=ylab, xlabel=xlab, ylim=ylim)
    ax.grid(alpha=.22)


def main(Ri_target=1.05, logrs=(-4., 3.), steps=(0, 3, 10), key=KEY, z=Z, outdir=OUTDIR):
  cells = M._test_cells(key, z, 120)
  k = int(min(cells, key=lambda c: abs(c[4] - Ri_target))[0])
  raw = open_celldata(key, k)
  shr = cellsBehindShock_fromData(open_rundata(key, z))
  d0 = shr.loc[shr.i == raw.iloc[0].i].iloc[0]
  ex = shr.loc[shr.t.idxmax()]

  fig, axes = plt.subplots(4, len(logrs), figsize=(7.8*len(logrs), 15.8),
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
    print(f'=== log10(gc/gm)={logr:+.0f}, p={p} ===')
    for i, (j, col) in enumerate(zip(steps, ('#0072B2', '#009E73', '#D55E00'))):
      st = step_view(cols, j)
      bsyn = float(_sval(st, 'bsyn', env))
      nupB = float(_sval(st, 'nup_B', env))
      tt = cooled_tt_eff(gmx[j], bsyn)
      lab = rf'step {j}, $\gamma_{{\max}}/\gamma_{{\min}}$={gmx[j]/gmn[j]:.3g}'

      # 1. distribution, compensated by gamma^p (flat = pure power law)
      G = np.geomspace(max(gmn[j], 1.), gmx[j], 3000)
      Nc = distrib_plaw_cooled(G, p, tt)*G**p
      # no slope axis here, and a TIGHT y-range: gamma^p N only runs 1 -> ~0.38, so the
      # shape is invisible on the decades the other rows need
      _panel(axes[0, c], G/gmn[j], Nc, col, lab + rf',  $b_{{\rm syn}}$={bsyn:.3f}',
             (), r'$\gamma^{p}N(\gamma)$, norm.', r'$\gamma/\gamma_{\rm min}$',
             (-3., 1.), i == 0, ylim=(0.2, 1.15), with_slope=False)

      # 2. e'_nu'  and  3. L'_nu' -- same tnu axis, must have the SAME shape
      tnu = np.geomspace(1e-2, 1e4*gmx[j]**2, 4000)
      e = np.asarray(get_epnu(tnu, st, K0, env, Ng=NG), float)
      L = np.asarray(get_Lnu_comov(tnu*nupB, st, K0, env, NG, True, 1.1), float)
      gu = ((1./3., '1/3'), (-(p-1.)/2., '-(p-1)/2'))
      elo = 1e-16 if logr > 0. else 1e-10       # slow cooling needs many more decades
      _panel(axes[1, c], tnu, e, col, lab, gu, r"$e'_{\nu'}$", r"$\nu'/\nu'_B$",
             (-2.6, 1.), i == 0, ylim=(elo, 3.))
      _panel(axes[2, c], tnu, L, col, lab, gu, r"$L'_{\nu'}$", r"$\nu'/\nu'_B$",
             (-2.6, 1.), i == 0, ylim=(elo, 3.))

      # 4. nu F_nu, observer frame
      nu = np.geomspace(1e-7, 3e2*gmx[j]**2, 3000)
      # evaluate at THIS step's own arrival (tT=1). env.Ts is before every step's onset, and
      # get_Fnu_step now correctly returns zero there.
      Ton_j, Tth_j, Tej_j = _sval(st, 'obsT', env)
      F = np.asarray(get_Fnu_step(nu*env.nu0, float(Ton_j), st, K0, env, NG, True, 1.1),
                     float)*nu
      _panel(axes[3, c], nu, F, col, lab, ((4./3., '4/3'), ((3.-p)/2., '(3-p)/2')),
             r'$\nu F_\nu$', r'$\nu/\nu_{\mathrm{m},0}$', (-2.2, 1.8), i == 0)

      # rows 2 -> 3 may only RESCALE: report the worst shape departure
      mm = (e > 0) & (L > 0) & np.isfinite(e) & np.isfinite(L)
      if mm.any():
        r = L[mm]/e[mm]; r = r/np.median(r)
        print(f"  step {j:2d}  L'/e' shape ratio  min {r.min():.6f}  max {r.max():.6f}"
              f"   (1.000000 = pure rescaling)")
    axes[0, c].set_title(rf'$\log_{{10}}\mathcal{{C}}$ = {logr:+.0f}', fontsize=12)
    for r_ in range(4):
      axes[r_, c].legend(fontsize=7, loc='lower left')
  fig.suptitle(f'ONE sub-step through the whole chain, cell k={k}   '
               f'(solid = quantity, dashed = its slope on the right axis; Ng={NG})',
               fontsize=12)
  path = os.path.join(outdir, 'onestep_shape.png')
  fig.savefig(path, dpi=140, bbox_inches='tight')
  plt.close(fig)
  print(f'saved {path}')


if __name__ == '__main__':
  main()
