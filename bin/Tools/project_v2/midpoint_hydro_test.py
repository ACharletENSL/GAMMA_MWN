'''
Cost of reading the EMISSION PREFACTOR at each cooling step's left edge, and the
validation of midpointing it (radiation_cooling.precompute_step_cols, midpoint_hydro).

The electron bounds have been midpointed for a long time (_midpoint_cell) and the energy
budget weights them exactly per electron (step_radiated_energy). What was left at the left
edge was everything the step's emission is PROPORTIONAL to -- Aad*Pmax*V3p*nu'_B -- while a
step is long enough for the hydro to move under it (up to dlnrho_max = 0.075 in ln rho).
That is a left-rectangle quadrature: first order, and single-signed because the hydro drifts
monotonically along a worldline.

WHAT IS COMPARED. Three evaluations of the SAME cell, all through the production kernels:

  reference    the production path re-derived at dlnrho_max/64, i.e. 10-25x more steps
  left edge    midpoint_hydro=False -- what the code did before
  midpoint     midpoint_hydro=True  -- what it does now

The reference is a step-size limit of the same scheme, not an independent solution, so read
these as convergence, not as absolute error. The decisive column is not the size of the
left-edge error but that the midpointed value is FLAT in dlnrho_max where the left-edge one
still crawls (--ladder): that is what identifies this as the dominant step-size error rather
than one term among several.

NOT covered, and visible in the peak column: the arrival-time discretisation. Each step is a
flash at its own Ton with an HLE tail, and the kinematics (t, x, vx -> obsT, Dop) stay at the
left edge on purpose -- T_obs = t - beta*x is a cancellation of order x/2Gamma^2, so shifting
x by half a step is ~200x the quantity itself and costs -98% on the peak. So the lightcurve
peak stays several % high in slow cooling whatever this switch does.

  python -c "import midpoint_hydro_test as M; M.main()"
  python -c "import midpoint_hydro_test as M; M.main(ladder=True)"   # the convergence check
'''

import numpy as np

from environment import MyEnv
from working_cooling import (generate_cell_withDistrib, precompute_step_cols, step_view,
                             norm_plaw_distrib, _midpoint_cell, DLNRHO_MAX)
from radiation_cooling import get_Fnu_step, step_radiated_energy, NG_FLUX
from IO import open_celldata, open_rundata
from fits_hydro import cellsBehindShock_fromData
from sweep_gammacm import compute_alpha_sweep
import prerar_model as M

KEY, Z = 'cooling_g100', 4
LOGRS = (-4., -2., 0., 1., 3.)
REF_DIV = 64                      # reference = production dlnrho_max / this
TRAPZ = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz


def _K0(cell, env):
  return norm_plaw_distrib(cell.iloc[0].gmin, cell.iloc[0].gmax, env.psyn)


def cell_E_rad(cell, env, midpoint_hydro=True, Ng=120, width_tol=1.01):
  '''cell_radiated_energy with the hydro switch exposed'''
  K0 = _K0(cell, env)
  cl = precompute_step_cols(cell, env, keys=('nup_B', 'V3p', 'Pmax'),
                            midpoint_hydro=midpoint_hydro)
  return sum(step_radiated_energy(step_view(cl, j), K0, env, Ng, width_tol)
             * cl['nup_B'][j] * cl['V3p'][j] for j in range(len(cell)))


def cell_Fnu(nuobs, Tarr, cell, env, midpoint_hydro=True, Ng=NG_FLUX):
  '''get_Fnu_cell_evolving with the hydro switch exposed (bounds always midpointed,
  band cut dropped so the two sides differ only in where the prefactor is read)'''
  K0 = _K0(cell, env)
  cl = precompute_step_cols(_midpoint_cell(cell), env, midpoint_hydro=midpoint_hydro)
  Ton = cl['obsT'][0]
  F = np.zeros((Tarr.size, np.size(nuobs)))
  for j in range(len(cell)):
    i0 = np.searchsorted(Tarr, Ton[j])
    if i0 < Tarr.size:
      F[i0:] += get_Fnu_step(nuobs, Tarr[i0:], step_view(cl, j), K0, env, Ng)
  return F


def _build(raw, d0, ex, k, alpha, dl, r_ref, key=KEY):
  return generate_cell_withDistrib(raw, d0, MyEnv(key), alpha=alpha, r_ref=r_ref,
             Tmax=None, key=key, k=k, exit_row=ex, dlnrho_max=dl)


def main(logrs=LOGRS, Ri_target=1.05, r_ref=1.1, prod_dl=DLNRHO_MAX, ref_div=REF_DIV,
    key=KEY, z=Z, NT=110, Nnu=80, ladder=False):
  cells = M._test_cells(key, z, 120)
  k = int(min(cells, key=lambda c: abs(c[4] - Ri_target))[0])
  raw = open_celldata(key, k)
  shr = cellsBehindShock_fromData(open_rundata(key, z))
  d0 = shr.loc[shr.i == raw.iloc[0].i].iloc[0]
  ex = shr.loc[shr.t.idxmax()]
  nut = np.logspace(-4., 6., Nnu)
  print(f'cell k={k}, r_ref={r_ref}, production dlnrho_max={prod_dl}')

  if ladder:
    # does midpointing COLLAPSE the step-size ladder? that is the real claim
    print('\nE_rad vs its own coarsest value, at each dlnrho_max:')
    print(f'{"logr":>5} {"dlnrho":>9} {"N":>5} {"left edge":>10} {"midpoint":>10}')
    for logr in logrs:
      alpha = float(compute_alpha_sweep(key, np.array([float(logr)]))[0][0])
      E0 = Em0 = None
      for f in (1, 2, 4, 8):
        cell, env = _build(raw, d0, ex, k, alpha, prod_dl/f, r_ref)
        E, Em = cell_E_rad(cell, env, False), cell_E_rad(cell, env, True)
        if E0 is None:
          E0 = Em0 = E
        print(f'{logr:>5.0f} {prod_dl/f:>9.6f} {len(cell):>5}'
              f' {100.*(E/E0 - 1.):>9.3f}% {100.*(Em/Em0 - 1.):>9.3f}%', flush=True)
      print()
    return

  print(f'reference = same path at dlnrho_max/{ref_div}; all deltas vs it\n')
  print(f'{"logr":>5} {"variant":>11} {"N":>5} | {"E_rad":>8} | {"fluence":>8}'
        f' {"peak Fnu":>9} | {"max|dS|":>8} {"med|dS|":>8}')
  for logr in logrs:
    alpha = float(compute_alpha_sweep(key, np.array([float(logr)]))[0][0])
    cref, env = _build(raw, d0, ex, k, alpha, prod_dl/ref_div, r_ref)
    cp, env = _build(raw, d0, ex, k, alpha, prod_dl, r_ref)
    Ton = precompute_step_cols(cref, env)['obsT'][0]
    Tarr = np.geomspace(Ton.min(), Ton.max()*5., NT)
    nu_p = nut*env.nu0
    rows = [('reference', cref, len(cref), True), ('left edge', cp, len(cp), False),
            ('midpoint', cp, len(cp), True)]
    res = [(lab, n, cell_E_rad(c, env, mh), nut*cell_Fnu(nu_p, Tarr, c, env, mh))
           for lab, c, n, mh in rows]
    E0, S0 = res[0][2], res[0][3]
    f0 = TRAPZ(np.nan_to_num(S0), Tarr, axis=0)
    m = f0 > f0.max()*1e-4
    lc0 = np.nanmax(np.nansum(np.nan_to_num(S0), axis=1))
    for lab, n, E, S in res:
      f = TRAPZ(np.nan_to_num(S), Tarr, axis=0)
      r = f[m]/f0[m] - 1.
      print(f'{logr:>5.0f} {lab:>11} {n:>5} | {100.*(E/E0 - 1.):>7.3f}% |'
            f' {100.*(TRAPZ(f, nut)/TRAPZ(f0, nut) - 1.):>7.3f}%'
            f' {100.*(np.nanmax(np.nansum(np.nan_to_num(S), axis=1))/lc0 - 1.):>8.3f}%'
            f' | {100.*np.max(np.abs(r)):>7.3f}% {100.*np.median(np.abs(r)):>7.3f}%',
            flush=True)
    print()


if __name__ == '__main__':
  main()
