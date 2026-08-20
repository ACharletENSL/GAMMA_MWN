'''
CONTROLLED TEST OF THE EMISSION KERNEL: one cell with FROZEN hydrodynamics.

rho, p and Gamma are held at their post-shock values along the whole worldline, and dx is set
to keep the COMOVING VOLUME constant (dx ~ 1/r^2), so:

  - B is constant                      -> every cooling step has the SAME nu_B
  - V3p is constant                    -> constant ELECTRON NUMBER (mass conserved), no
                                          adiabatic losses; electrons cool by synchrotron only
  - Gamma is constant                  -> one Doppler factor, one (1-beta) lag for every step

Holding dx fixed instead is the obvious thing to do and it is WRONG: V3p ~ r^2 dx then grows
1e6 over 3 decades, the electron number with it, and the cell brightens with time. That is the
test misbehaving, not the kernel -- it accounted for about half the apparent dip depth.

Under those conditions the sum of per-step synchrotron kernels MUST reproduce ordinary
synchrotron theory: a smooth broken power law whose local slope, in nu F_nu, is

    +4/3        below nu_m
    +(3-p)/2    nu_m .. nu_c
    -(p-2)/2    above nu_c, flat until the cutoff
    steepening  through the cutoff, MONOTONICALLY -- it may never come back up

There is no hydro evolution left to blame, so any departure is the kernel's: either the
per-step normalisation, the cutoff shape, or the rule that every step contributes to every
later observer time (get_Fnu_cell_evolving accumulates Fnu[iT0:] += ...).

What this is testing, concretely: a real cell shows the nu F_nu slope steepen to ~-0.40 about
a decade above the peak and then RECOVER to -0.26, which one zone forbids. See
prerar_kink_figure.py for that measurement on the physical cell.

  python -c "import frozen_cell_test as F; F.main()"
  python -c "import frozen_cell_test as F; F.main(logr=-3.)"      # fast-cooling end
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from environment import GAMMA_dir
import working_cooling_data as W
from working_cooling import get_nuFnu, get_Fnu_cell_evolving
import prerar_model as M
from sweep_gammacm import compute_alpha_sweep
from IO import open_celldata

KEY, Z = 'cooling_g100', 4
OUTDIR = os.path.join(GAMMA_dir, 'bin', 'Tools', 'figures', 'prerar_model')


def frozen_history(shocked, n_rows=4000, span_dex=3.0):
  '''
  A worldline with the input's FIRST-ROW hydro held constant: rho, p, vx, dx frozen, the
  radius advancing ballistically at that fixed velocity, t consistent with it.

  n_rows is deliberately large and log-spaced in (x - x0): the cooling-step binning is
  geometric in gamma_max and interpolates the hydro between rows, so the rows only need to
  be dense enough not to be the limiting resolution. Frozen hydro interpolates exactly.
  '''
  r0 = shocked.iloc[0]
  x0, v0 = float(r0.x), float(r0.vx)
  t0 = float(r0.t)
  x = x0*np.logspace(0., span_dex, n_rows)
  t = t0 + (x - x0)/v0                       # constant velocity, exact
  out = pd.DataFrame({c: np.full(n_rows, r0[c]) for c in shocked.columns})
  out['x'], out['t'] = x, t
  out['rho'] = float(r0.rho)
  out['p'] = float(r0.p)
  out['vx'] = v0
  # dx ~ 1/r^2 so the COMOVING VOLUME V3p ~ r^2 dx is constant. Holding dx fixed instead
  # lets V3p grow as r^2 -- a factor 1e6 over 3 decades -- which silently multiplies the
  # ELECTRON NUMBER by the same factor (mass is not conserved) and makes the cell brighten
  # with time. That is a property of the test, not of the kernel, and it cost me a wrong
  # bug report: the per-flash fluence/energy ratio is constant to 2.5% once V3p is carried
  # on both sides.
  out['dx'] = float(r0.dx)*(x0/x)**2
  if 'Sd' in out:
    out['Sd'] = 0.
  out.index = np.arange(n_rows)
  out.attrs = dict(shocked.attrs)
  return out


def main(Ri_target=1.05, logr=1., key=KEY, z=Z, outdir=OUTDIR, Nnu=700,
    targets=(0.05, 1.0, 20.), n_rows=4000, span_dex=3.0):
  os.makedirs(outdir, exist_ok=True)
  cells = M._test_cells(key, z, 120)
  k = int(min(cells, key=lambda c: abs(c[4] - Ri_target))[0])
  alpha = float(compute_alpha_sweep(key, np.array([float(logr)]))[0][0])

  cd_raw = open_celldata(key, k)
  shocked = W.select_postshock_rows(cd_raw, 1)
  shocked = shocked.assign(t=shocked['t'] - cd_raw.t.iloc[0])
  froz = frozen_history(shocked, n_rows=n_rows, span_dex=span_dex)

  from environment import MyEnv
  env0 = MyEnv(key)
  cell, env = W.generate_cell_fromHistory(froz, cd_raw.attrs, env0, alpha=alpha,
                                          Tmax=None, r_ref=1.05)
  if cell is False:
    print('frozen cell rejected by generate_cell_fromHistory')
    return None
  p = env.psyn
  # observer grid, same conventions as the cell driver
  NT = 300
  Tobs = env.Ts + (np.logspace(np.log10(1.), np.log10(1000.), NT) - 1.)*env.T0
  nuobs = np.logspace(-6, 8, Nnu)*env.nu0
  S = np.asarray(get_nuFnu(get_Fnu_cell_evolving, nuobs, Tobs, cell, env), float)
  nu, Tb = nuobs/env.nu0, Tobs/env.T0 - 1.

  gmn, gmx = cell.gmin.to_numpy(float), cell.gmax.to_numpy(float)
  print(f'FROZEN cell k={k}  log10(gc/gm)={logr:+.0f}  p={p}  steps={len(cell)}')
  print(f'  rho, p, Gamma, dx constant to: '
        f'{cell.rho.std()/cell.rho.mean():.2e}, {cell.p.std()/cell.p.mean():.2e}, '
        f'{cell.lfac.std()/cell.lfac.mean():.2e}, {cell.dx.std()/cell.dx.mean():.2e} (rel. std)')
  print(f'  gmin {gmn[0]:.4g}->{gmn[-1]:.4g}   gmax {gmx[0]:.4g}->{gmx[-1]:.4g}')
  print(f'\n  one zone allows above the peak: {-(p-2.)/2.:+.3f}, then a MONOTONIC cutoff')
  idx = [int(np.argmin(np.abs(Tb - t))) for t in targets]
  ln = np.log10(nu)
  for it in idx:
    s = np.where(np.isfinite(S[it]) & (S[it] > 0), S[it], np.nan)
    if not np.isfinite(np.nansum(s)) or np.nansum(s) <= 0:
      continue
    g = np.gradient(np.log10(s), ln)
    jp = int(np.nanargmax(s))
    seg = [np.nanmedian(g[(ln - ln[jp] >= c - .4) & (ln - ln[jp] < c + .4)])
           for c in (0.5, 1., 1.5, 2., 3., 4.)]
    below = np.nanmedian(g[(ln - ln[jp] >= -3.5) & (ln - ln[jp] < -2.5)])
    dip = np.nanmin([v for v in seg if np.isfinite(v)] or [np.nan])
    tail = seg[-1]
    print(f'  barT={Tb[it]:7.2f}  below pk {below:+.2f} | above pk ' +
          ' '.join(f'{v:+.2f}' for v in seg) +
          f'   |  dip {dip:+.3f} vs tail {tail:+.3f} -> '
          f'{"RECOVERS (unphysical)" if tail > dip + 0.02 else "monotonic (ok)"}')

  fig, axes = plt.subplots(2, len(idx), figsize=(4.2*len(idx), 7.2),
                           layout='constrained', sharex=True)
  axes = np.atleast_2d(axes)
  for j, it in enumerate(idx):
    s = S[it]
    m = np.isfinite(s) & (s > 0)
    axes[0, j].loglog(nu[m], s[m], color='#0072B2', lw=1.6)
    gg = np.gradient(np.log10(np.where(m, s, np.nan)), ln)
    axes[1, j].semilogx(nu, gg, color='#0072B2', lw=1.6)
    for y in (4./3., (3.-p)/2., -(p-2.)/2.):
      axes[1, j].axhline(y, color='#009E73', lw=1., ls=(0, (5, 2)))
    axes[0, j].set_title(rf'$\bar T$ = {Tb[it]:.2f}', fontsize=10)
    axes[1, j].set(xlabel=r'$\nu/\nu_m$', ylim=(-1.6, 1.8))
    for ax in (axes[0, j], axes[1, j]):
      ax.grid(alpha=.25)
      ax.axvline(1., color='grey', ls=':', lw=.9)
    if j == 0:
      axes[0, j].set_ylabel(r'$\nu F_\nu$  (frozen-hydro cell)')
      axes[1, j].set_ylabel(r'$d\log(\nu F_\nu)/d\log\nu$')
  fig.suptitle(f'FROZEN hydro (rho, p, $\\Gamma$ and comoving volume constant), '
               f'cell k={k}, p={p} -- the kernel alone', fontsize=11)
  path = os.path.join(outdir, f'frozen_cell_k{k}_logr{logr:+.0f}.png')
  fig.savefig(path, dpi=150, bbox_inches='tight')
  plt.close(fig)
  print(f'\nsaved {path}')
  return dict(nu=nu, Tb=Tb, S=S, cell=cell, env=env)


if __name__ == '__main__':
  main()
