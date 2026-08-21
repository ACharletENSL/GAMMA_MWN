# -*- coding: utf-8 -*-
# @Author: acharlet

'''
Produce emission from hydro data + electron distribution
Contains:
  - syn_emiss_exact
  - func_R
  - get_Fnu_cell
  - get_Fnu_hydrostep
  - get_Fnu_step
  - get_Lnu_comov
  - get_epnu
  - Pnu_instant / cooled_tt_eff (cooled-distribution cutoff, see cooled_tt_eff)
  - Pnu_instant_fit (FM26 analytic fit alternative to Pnu_instant)
  - get_Fnu_cell_analytic (optimal time binning for constant hydro)
  - get_cell_nuFnu_analytic (wrapper for analytic case)
'''

import numpy as np
from numba import njit
from scipy.integrate import simpson
from cooling_distribution import norm_plaw_distrib, distrib_plaw_cooled, \
  gamma_synCooled, split_hydrostep
from syn_fitting_FM26 import Jpl_FM26, pnu_fm26_scalars, _pnu_fm26_kernel
from IO import get_variable
from obs_functions import *
from phys_functions import (func_R, _func_R_exact, syn_cutoff_R,
    R_LOW_COEF, coeffs_bel1, coeffs_abv1,
    _R_logR, _R_LOGXMIN, _R_INV_DLOG, _R_XMIN, _R_XMAX)
from phys_constants import *

from types import SimpleNamespace


class _StepView(SimpleNamespace):
  '''Lightweight carrier of one cooling step's precomputed scalars, so the
  cooling-step hot loops avoid re-materializing cell.iloc[j] and re-deriving
  get_variable(step,...) every step. Downstream kernels read fields via _sval,
  which falls back to the pandas-row path for any non-view step (bit-identical).'''
  pass


_STEP_BASE_COLS = ('x', 'trac', 'gmin', 'gmax', 'dtp')
# base columns that older / analytic cell frames may not carry, with their default
# (bsyn=0 reproduces the historical tt=1/gmax cooled shape exactly; dtt=0 drops the
# step-integration weight of step_radiated_energy, i.e. the instantaneous-rate limit)
_STEP_OPT_COLS = {'bsyn': 0., 'dtt': 0.}


def _sval(step, key, env):
  '''Per-step scalar: precomputed attribute on a _StepView, else derived from a
  pandas row exactly as before (base column by item access, derived via
  get_variable) -> bit-identical for non-view callers. Keys in _STEP_OPT_COLS fall
  back to their default when the cell frame predates them (analytic cells), so
  those paths keep their historical values.'''
  if isinstance(step, _StepView):
    return getattr(step, key, _STEP_OPT_COLS[key]) if key in _STEP_OPT_COLS \
           else getattr(step, key)
  if key in _STEP_OPT_COLS:
    return step[key] if key in step else _STEP_OPT_COLS[key]
  if key in _STEP_BASE_COLS:
    return step[key]
  return get_variable(step, key, env)


# Electron-integral resolution for the FLUX path. Was 20, which is 12.9% off in fast
# cooling and 48.9% in SLOW (gmax/gmin reaching 2.6e5 = 3.7 points per decade in gamma),
# and roughened the slope at the 1/3 -> -(p-1)/2 turnover by 2.8-3.9x. 120 matches
# cell_radiated_energy, so the energy and flux paths now agree, and costs 1.47x
# (measured 0.120 -> 0.176 s per cell spectrum) for 0.39% / 1.44% residual error.
NG_FLUX = 120


def precompute_step_cols(cell, env, keys=('Dop', 'nup_B', 'Tth', 'V3p', 'Pmax', 'obsT')):
  '''Vectorize a whole cell's per-step quantities once (arrays), to feed
  _StepView in the cooling-step loops. Derived keys via get_variable(cell,...)
  (same formula as the per-row call); base columns via .to_numpy(); obsT is the
  (Ton, Tth, Tej) triple.'''
  cols = {}
  for key in keys:
    if key == 'obsT':
      Ton, Tth, Tej = get_variable(cell, 'obsT', env)
      cols['obsT'] = (np.asarray(Ton, float), np.asarray(Tth, float), np.asarray(Tej, float))
    else:
      cols[key] = np.asarray(get_variable(cell, key, env), float)
  for key in _STEP_BASE_COLS:
    cols[key] = cell[key].to_numpy()
  for key, default in _STEP_OPT_COLS.items():
    cols[key] = cell[key].to_numpy() if key in cell \
                else np.full(len(cell), default, dtype=float)
  return cols


def step_view(cols, j):
  '''Build the _StepView for step j from the precomputed arrays.'''
  kw = {key: ((v[0][j], v[1][j], v[2][j]) if key == 'obsT' else v[j])
        for key, v in cols.items()}
  return _StepView(**kw)


def syn_emiss_exact(gma, tnu):
  '''
  "exact" synchrotron emission function from Crusius & Schlikeiser (1986)
  for an electron with LF gma at tnu = nu'/nu'_B, nu'_B=e*B/(2*pi_*me_*c_)
  Vectorized: gma and tnu can be scalars or broadcastable arrays.

  UNITS: this returns R(x) itself, x = 2tnu/(3 gma**2). R is the physical shape,
  and its own normalisation norm_R_ = int R(x) dx = 1.0751412668 is NOT divided out
  here -- it is a single scalar and belongs where an absolute emissivity is actually
  formed, not on every element of every kernel evaluation. get_epnu applies it once,
  in the same multiply as Pmax (see there); step_radiated_energy consumes it inside
  its analytic frequency integral. So
    int R(x) dtnu = norm_R_ * (3/2) gma**2,
  and Pmax * nup_B * that / norm_R_ is the true single-electron synchrotron power
  (4/3) sigT c gma**2 U_B -- checked to 4e-7.
  '''
  gma = np.asarray(gma, dtype=float)
  tnu = np.asarray(tnu, dtype=float)
  # underflows in np.exp will be treated as 0.
  with np.errstate(under='ignore'):
    x = (2.*tnu)/(3.*gma**2)
    out = np.where(tnu < 1., 0., func_R(x))
  return out if out.ndim else float(out)

# R(x), its tabulated fast path and the cut-off shape it defines now live in
# phys_functions (granot_sari_syn needs them and cannot import this module:
# radiation_cooling imports phys_functions, not the reverse). Imported above
# and re-exported here, so `from radiation_cooling import func_R` still works.



# Contribution from a cell
def get_Fnu_cell(nuobs, Tobs, data, env,
  Ng=NG_FLUX, norm=True, width_tol=1.1, r_ref=1.2, Nmin=2, Nmax=20):
  '''
  F_nu(T) from a cell (dataframe with the cell history)
  array of size nuobs array
  Ng is the number of log bins in [gmin, gmax] for calculation
  '''

  Fnu_out = np.zeros(nuobs.shape)

  cell_0 = data.iloc[0]
  K0 = norm_plaw_distrib(cell_0.gmin, cell_0.gmax, env.psyn)
  nu_B0 = get_variable(cell_0, 'nu_B', env)

  # no need to calculate contributions if nu_M outside of observed freqs 
  gmax_cut = max(1., np.sqrt(nuobs.min()/nu_B0))
  gmax_arr = data.gmax.to_numpy()
  N = gmax_arr.size
  # np.searchsorted works in ascending order
  jcut = N - np.searchsorted(gmax_arr[::-1], gmax_cut, side = "right")
  print(f'Calculating over {jcut} sim iterations')
  jcut = max(jcut, 1)
  jcut = min(jcut, N-1)
  for j in range(jcut):
    if (j%100==0):
      print(f'Iteration {j} of {jcut}')
    hydro = data.iloc[j]
    hydro_next = data.iloc[j+1]
    Fnu = get_Fnu_hydrostep(nuobs, Tobs, hydro, hydro_next, K0, env, norm, width_tol, r_ref, Nmin, Nmax)
    Fnu_out += Fnu
  return Fnu_out

def get_Fnu_hydrostep(nuobs, Tobs, cell, cell_next, K0, env,
  Ng=NG_FLUX, norm=True, width_tol=1.1, r_ref=1.5, Nmin=2, Nmax=20):
  '''
  F_\nu (T) from a hydro step
  '''

  Fnu_out = np.zeros(nuobs.shape)
  steps = split_hydrostep(cell, cell_next, env, r_ref, Nmin, Nmax)
  Nj = len(steps)
  for j in range(Nj):
    step = steps.iloc[j]
    Fnu = get_Fnu_step(nuobs, Tobs, step, K0, env, Ng, norm, width_tol)
    Fnu_out += Fnu
  return Fnu_out

# Contribution from a single time bin
def get_Fnu_step(nuobs, Tobs, step, K0, env, Ng=NG_FLUX, norm=True, width_tol=1.1):
  '''
  F_\nu (T) from a (cooling) step
  '''
  
  D = _sval(step, "Dop", env)
  Ton, Tth, Tej = _sval(step, 'obsT', env)
  tT = (Tobs - Tej) / Tth                          # Tobs_to_tildeT inlined
  nuobs = np.asarray(nuobs, dtype=float)
  # A step emits NOTHING before its own on-axis arrival. tT=1 is Tobs=Ton (theta=0, the
  # first photon); tT<1 means Tobs<Ton, and photons from larger theta arrive LATER still,
  # so there is no line of sight that has delivered anything yet. Only tT<=0 was excluded
  # before, which let steps radiate ahead of their own onset with a tT^-2 > 1 boost and
  # their spectrum shifted down in frequency. get_Fnu_cell_evolving never saw this (it
  # slices at iT0 = searchsorted(Tarr, Ton), so tT>=1 by construction), but every
  # fixed-Tobs caller did -- working_cooling_prev.get_cell_nuFnu evaluates the whole cell
  # at Tobs=env.Ts, where measured tT spans 0.27..0.94 and NOT ONE step has arrived.
  live = tT >= 1.
  if np.ndim(Tobs) > 0:
    tT = np.asarray(tT)
    # outer-product fast path: nup[i,j] = tT[i] * (nuobs/D)[j], tT^-2 folded in
    with np.errstate(divide='ignore'):
      w = np.where(live, np.where(tT > 0., tT, 1.)**-2, 0.)
    Fnu = get_Lnu_outer(nuobs/D, tT, step, K0, env, Ng, norm, width_tol,
                        row_weight=w)
  elif not live:
    Fnu = np.zeros_like(nuobs)
  else:
    nup = nuobs * tT / D
    Lnu = get_Lnu_interp(nup, step, K0, env, Ng, norm, width_tol)
    Fnu = tT**-2 * Lnu  # * 2 * lfac
  if norm:
    Fnu /= (2*env.lfac0/3)    # /3 because F0 = zdL * 2 lfac0 * L0 / 3
  else:
    Fnu *= env.zdl
  return Fnu

def get_Lnu_comov(nup, step, K0, env, Ng=NG_FLUX, norm=True, width_tol=1.1):
  '''
  L'_\nu' from a (cooling) step
    as a function of \nu'/\nu'_m
  '''

  nup_B = _sval(step, 'nup_B', env)
  rsc = 1.
  if env.geometry == 'cartesian':
    # assume geometrical scaling nu' \propto R^(-1)
    rsc = _sval(step, 'x', env) * c_/env.R0
  nup_B /= rsc
  tnu = nup/nup_B

  Ttz = _sval(step, "Tth", env)/(1+env.z)
  V3p = _sval(step, "V3p", env)
  Lnu = get_epnu(tnu, step, K0, env, Ng, width_tol)
  Lnu *= rsc * V3p/Ttz

  if norm:
    Lnu /= (env.L0p if _sval(step, 'trac', env) < 1.5 else env.L0pFS)
  return Lnu

def get_Lnu_interp(nup, step, K0, env, Ng=NG_FLUX, norm=True, width_tol=1.1, n_grid=300):
  '''
  get_Lnu_comov evaluated over an arbitrary-shape nup grid (e.g. the 2D (T, nu)
  EATS-shifted grid from get_Fnu_step), via a 1D log-grid in nup + log-log
  interpolation. Lnu is a smooth 1D function of nup, so this is exact within each
  power-law segment and avoids the full (T x nu) kernel evaluation.
  1D inputs (scalar Tobs) are evaluated directly so those results are unchanged.
  '''
  nup = np.asarray(nup, dtype=float)
  if nup.ndim < 2 or nup.size <= n_grid:
    return get_Lnu_comov(nup, step, K0, env, Ng, norm, width_tol)
  out = np.zeros(nup.shape)
  # nup <= 0 comes from Tobs < Tej (tT < 0): unphysical, no emission there.
  # Build the interpolation grid from the positive values only, else the
  # geomspace bounds go negative and the whole grid silently turns to NaN.
  pos_in = nup > 0.
  if not pos_in.any():
    return out
  nup_grid = np.geomspace(nup[pos_in].min(), nup[pos_in].max(), n_grid)
  Lnu_grid = get_Lnu_comov(nup_grid, step, K0, env, Ng, norm, width_tol)
  pos = Lnu_grid > 0.
  if pos.sum() >= 2:
    # log-log interp; below the lowest positive node (sub-nu'_B) Lnu -> 0
    logL = np.interp(np.log(np.where(pos_in, nup, 1.)).ravel(),
                     np.log(nup_grid[pos]), np.log(Lnu_grid[pos]),
                     left=-np.inf, right=-np.inf)
    out = np.where(pos_in, np.exp(logL).reshape(nup.shape), 0.)
  return out

@njit(cache=True)
def _outer_loglog_blend(logL_grid, u, s, w):
  '''
  out[i, j] = w[i] * exp(linear blend of logL_grid at fractional index u[j]+s[i])
  Fused gather/blend/exp/scale kernel of get_Lnu_outer (indices in range by
  construction there).
  '''
  n_grid = logL_grid.size
  out = np.empty((s.size, u.size))
  for i in range(s.size):
    si, wi = s[i], w[i]
    for j in range(u.size):
      idx = si + u[j]
      i0 = int(idx)
      if i0 > n_grid - 2:
        i0 = n_grid - 2
      f = idx - i0
      out[i, j] = wi * np.exp((1.-f)*logL_grid[i0] + f*logL_grid[i0+1])
  return out

def get_Lnu_outer(nub, tT, step, K0, env, Ng=NG_FLUX, norm=True, width_tol=1.1,
    n_grid=300, row_weight=None):
  '''
  Lnu evaluated at nup[i, j] = tT[i] * nub[j] (the EATS-shifted grid of
  get_Fnu_step), exploiting the outer-product structure: log-log interpolation
  on the same uniform-in-log grid as get_Lnu_interp, but the interp positions
  are an outer sum of two 1D logs - no full-grid log, multiply or binary
  search. Rows with tT <= 0 (Tobs < Tej) get 0. Same semantics/nodes as
  get_Lnu_interp to float precision.
  row_weight: optional per-row factor folded into the kernel (e.g. tT^-2).
  '''
  nub = np.asarray(nub, dtype=float)
  tT = np.atleast_1d(np.asarray(tT, dtype=float))
  out = np.zeros((tT.size, nub.size))
  pos_t = tT > 0.
  if not pos_t.any():
    return out
  lo = tT[pos_t].min() * nub.min()
  hi = tT[pos_t].max() * nub.max()
  nup_grid = np.geomspace(lo, hi, n_grid)
  Lnu_grid = get_Lnu_comov(nup_grid, step, K0, env, Ng, norm, width_tol)
  pos = Lnu_grid > 0.
  if pos.sum() < 2:
    return out
  # -800: exp() underflows to exactly 0, avoids -inf arithmetic in the blend
  logL_grid = np.full(n_grid, -800.)
  logL_grid[pos] = np.log(Lnu_grid[pos])
  dlog = np.log(hi/lo)/(n_grid - 1)
  u = np.log(nub/lo)/dlog                       # (nu,) in [0, n_grid-1]
  s = np.log(tT[pos_t])/dlog                    # (T+,)
  if row_weight is None:
    w = np.ones(s.size)
  else:
    w = np.asarray(row_weight, dtype=float)[pos_t]
  out[pos_t] = _outer_loglog_blend(logL_grid, u, s, w)
  return out

# module switch: route the plain power-law path of get_epnu through the
# FM26 analytic fit (Pnu_instant_fit) instead of the Ng-point trapezoid.
# Toggle from a notebook with radiation_cooling.PNU_USE_FM26 = True
PNU_USE_FM26 = False
# below this gmax/gmin the fit degrades (30-80% near the cutoff for eta ~ 1.1-3,
# a limitation of the FM26 fitting form itself) while the trapezoid over the
# narrow gamma range is accurate: keep the numerical path there.
FIT_ETA_MIN = 4.

def get_epnu(tnu_arr, step, K0, env, Ng=NG_FLUX, width_tol=1.1,
    func_cooling=gamma_synCooled, func_distrib=distrib_plaw_cooled, func_emiss=syn_emiss_exact,
    use_fit=None):
  '''
  Delta e'_nu' of a step (assuming constant P'_nu' in the bin)
    as a function of nu'/nu'_B
  K0 as variable because it depends on initial gma_min/max of the distrib
  Accepts tnu_arr of any shape (e.g. 2D (T, nu) for the EATS-shifted grid): the
  kernels operate per-tnu, so we flatten, compute, then reshape back.
  use_fit=True replaces the Pnu_instant trapezoid by the FM26 analytic fitting
  function (defaults to the module switch PNU_USE_FM26); it is only valid for the
  bsyn=0 shape, so it is skipped whenever the step carries a burn-off factor.
  '''
  tnu_arr = np.asarray(tnu_arr, dtype=float)
  shape_in = tnu_arr.shape
  tnu_flat = tnu_arr.ravel()
  gmin, gmax = _sval(step, 'gmin', env), _sval(step, 'gmax', env)
  if gmax <= 1.:
    return np.zeros(shape_in)
  bsyn = _sval(step, 'bsyn', env)
  Pmax = _sval(step, "Pmax", env)
  K = K0
  if gmin < 1.:
    gmin = 1.               # see step_radiated_energy: bound truncation, no rescaling
  if gmax/gmin <= width_tol:
    K = 1
    gma_mn = np.sqrt(gmin*gmax)
    #gma_mn = gmin
    enu = np.asarray(func_emiss(gma_mn, tnu_flat))
  else:
    if use_fit is None:
      use_fit = PNU_USE_FM26
    if use_fit and gmax/gmin >= FIT_ETA_MIN and bsyn == 0.:
      enu = Pnu_instant_fit(tnu_flat, gmin, gmax, env)
    else:
      enu = Pnu_instant(tnu_flat, gmin, gmax, env, Ng, func_distrib, func_emiss,
                        bsyn=bsyn)
  # 1/norm_R_ is R(x)'s normalisation (syn_emiss_exact returns R itself): applied
  # ONCE here, as a scalar in the prefactor, for both the trapezoid and the FM26
  # branch -- never inside the per-element kernels.
  enu *= K*_sval(step, 'dtp', env)*Pmax/norm_R_
  return enu.reshape(shape_in)


def cooled_tt_eff(gmax, bsyn):
  '''
  Normalized time to pass to distrib_plaw_cooled so that the cooled shape has its
  cutoff at the RIGHT place. The exact solution of an injected power law is
    N(gma,tt) = K0 gma^-p (1-gma*tt)^(p-2)  on  [gmin(tt), gmax(tt)],
  and at the physical cutoff 1-gmax*tt = gmax/gmax0 != 0: a sharp injected edge
  stays sharp, the distribution is truncated by its SUPPORT, not by the shape
  decaying to zero. Writing u = gma/gmax and b = the cumulative SYNCHROTRON-only
  burn-off factor of the top edge (gmax_syn-only/gmax0),
    1 - gma*tt = 1 - u*(1-b)   =>   tt_eff = (1-b)/gmax
  b=0 recovers the old tt=1/gmax, i.e. the gmax0 -> infinity solution, which
  vanishes at the cutoff and under-counts the emission there (23% at injection,
  ~2.5% of the fast-cooling energy budget). b=gmax/gmax0 (pure synchrotron) gives
  the true elapsed tt = 1/gmax - 1/gmax0; b->1 (adiabatic-dominated, which does
  NOT distort the power law) gives tt_eff -> 0, a pristine power law.
  '''
  return (1. - bsyn)/gmax

def Pnu_instant(tnu_arr, gmin, gmax, env, Ng=NG_FLUX,
      func_distrib=distrib_plaw_cooled, func_emiss=syn_emiss_exact, bsyn=0.):
  '''
  Energy per unit freq per unit volume and time at normalized time tt
    as a function of nu'/nu'_B, in units P'_e,max
  Separates the range gamma_min/max into log bins for calculation
  bsyn: synchrotron-only burn-off factor of the step (see cooled_tt_eff);
    0 reproduces the historical tt=1/gmax shape.
  Carries func_emiss's units, i.e. R(x)'s own normalisation is still in: get_epnu
  divides by norm_R_ once (see syn_emiss_exact).
  '''
  p = env.psyn
  gmas_arr = np.geomspace(gmin, gmax, Ng)
  # vectorized over (gma, tnu)
  dP = func_distrib(gmas_arr[:, None], p, cooled_tt_eff(gmax, bsyn)) \
      * func_emiss(gmas_arr[:, None], tnu_arr[None, :])
  return np.trapezoid(dP, gmas_arr, axis=0)


def step_radiated_energy(step, K0, env, Ng=120, width_tol=1.01,
      func_distrib=distrib_plaw_cooled):
  '''
  Comoving radiated energy per unit (nu'_B * V3p) of one cooling step, obtained by
  integrating the single-electron synchrotron power ANALYTICALLY in frequency:
    int syn_emiss_exact(gma, tnu) dtnu / norm_R_ = (3/2) gma**2   (exact, since
    syn_emiss_exact = func_R(x), x=2tnu/(3gma**2), int R(x)dx = norm_R_;
    the tnu<1 cutoff drops only O(gma**-8/3)).
  This replaces get_epnu's per-tnu emissivity + trapezoid over tnu by the analytic
  weight (3/2)gma**2, leaving only the smooth electron integral over gma (Simpson).
  Mirrors get_epnu's normalization branch-for-branch (Pmax, K, dtp), so the absolute
  scale is identical -- only the integration is unbiased/cheaper. Multiply the return
  by nup_B * V3p (as in cell_radiated_energy) to get an energy.

  norm_R_ does NOT appear below, and must not: doing the frequency integral in closed
  form is exactly where R(x)'s normalisation gets consumed, so it is already inside
  the constant (3/2). get_epnu, which integrates over tnu numerically instead, carries
  the 1/norm_R_ explicitly in its prefactor -- the two agree.

  gma_min cooled below 1: those electrons are non-relativistic and stop emitting, so
  the integral is truncated at gma = 1 and NOTHING else is done. K0 is a NUMBER
  normalisation on the injection bounds (norm_plaw_distrib) and the cooled shape
  conserves number exactly, so the surviving electrons already carry their own weight;
  the energy the cooled ones lost is in the radiation, not handed to the survivors.
  The former xi_N = derive_xiDN factor here applied the deep-Newtonian (energy
  re-spreading) renormalisation on top of that truncation, counting it twice. See
  phys_functions.derive_xiDN_cooled for the exact factors and when they do apply.

  The gma**2 weight is the INSTANTANEOUS rate, which over a finite step is a
  first-order overestimate: the step advances every electron by 1/gma_R = 1/gma_L
  + dtt, so what it actually radiates is
    me c^2 (gma_L - gma_R) = me c^2 gma_L gma_R dtt = me c^2 dtt gma_L**2/(1+gma_L dtt),
  i.e. the geometric mean squared, not gma_L**2. With the prefactor already
  carrying dtt (through dtp*Pmax), the exact step energy is obtained by weighting
  the LEFT-EDGE distribution with
    (3/2) gma**2 / (1 + gma*dtt)
  which is what is integrated below. This is exact per electron, so the result no
  longer drifts with the step size (measured eps_rad = 1.00032/1.00031/1.00027 at
  r_ref = 1.2/1.1/1.05, against 1.0029/1.0010/1.0005 for the former midpoint-bounds
  evaluation). dtt = 0 (frames predating the column) recovers that former limit.
  '''
  p = env.psyn
  gmin, gmax = _sval(step, 'gmin', env), _sval(step, 'gmax', env)
  if gmax <= 1.:
    return 0.
  bsyn = _sval(step, 'bsyn', env)
  dtt = _sval(step, 'dtt', env)
  Pmax = _sval(step, "Pmax", env)
  K = K0
  if gmin < 1.:
    gmin = 1.               # bound truncation only (see docstring)
  if gmax/gmin <= width_tol:                       # delta-function shortcut (as get_epnu)
    K = 1.
    gma_mn = np.sqrt(gmin*gmax)
    I = 1.5*gma_mn**2/(1. + gma_mn*dtt)
  else:
    gmas = np.geomspace(gmin, gmax, Ng)
    I = simpson(func_distrib(gmas, p, cooled_tt_eff(gmax, bsyn))
                * 1.5*gmas**2/(1. + gmas*dtt), x=gmas)
  return K*_sval(step, 'dtp', env)*Pmax * I

def Pnu_instant_fit(tnu_arr, gmin, gmax, env):
  '''
  Same quantity as Pnu_instant (distrib_plaw_cooled x syn_emiss_exact integrated
  over gamma), via the Ferguson & Margalit 2026 analytic fitting function J_pl
  instead of numerical integration:
    Pnu = (1/2) ((2/3) tnu)^{(1-p)/2} J_pl(p; x1, xinf)
  with x1 = (2/3) tnu/gmin^2, xinf = (2/3) tnu/gmax^2, eta = gmax/gmin, and the
  same tnu < 1 cutoff as syn_emiss_exact. Like Pnu_instant this is in R(x)'s own
  units (J_pl is built on the raw func_R), so get_epnu's single 1/norm_R_ covers
  both branches identically. Uses the tabulated func_R as the
  pitch-angle-averaged kernel Ftilde (same function, see syn_fitting_FM26).
  Valid for 2 < p <~ 5 and the synchrotron-only cooled power law; worst-case
  fit error ~50% near eta ~ 1.05-1.2 (see module docstring).
  NB: J_pl takes only (p, x1, xinf, eta), so it encodes the bsyn=0 shape (cutoff
  AT gmax) and cannot represent the bsyn>0 family (cooled_tt_eff). get_epnu
  therefore skips this path whenever bsyn != 0; using it there would need J_pl
  refitted against (1 - u(1-b))^(p-2). See syn_fitting_FM26.
  '''
  p = env.psyn
  eta = gmax/gmin
  tnu_arr = np.asarray(tnu_arr, dtype=float)
  if eta - 1. < 1e-3:
    # near-delta distribution: vectorized path (Eq. 21 valid at all freqs);
    # rare in practice, get_epnu's width_tol shortcut intercepts it upstream
    nut = (2./3.)*tnu_arr
    J = Jpl_FM26(p, nut/gmin**2, nut/gmax**2, eta, func_F=func_R)
    # tnu <= 0 gives inf/nan in the (discarded) second branch of the where
    with np.errstate(under='ignore', divide='ignore', invalid='ignore'):
      return np.where(tnu_arr < 1., 0., 0.5 * nut**((1.-p)/2.) * J)
  # fused numba fast path: per-step scalars once, single pass over tnu
  pm1h, om_pref, A1, a1, a2, a3, a4, log_psi_pref, c1, al2, al4 = \
      pnu_fm26_scalars(p, eta)
  pfac = 0.5                 # norm_R_ is applied once by get_epnu, not here
  inv_g2sq = 1./gmax**2
  g2fac = inv_g2sq**pm1h    # = xf^pm1h * nut^-pm1h, folds the output prefactor
  tf = np.ascontiguousarray(np.atleast_1d(tnu_arr).ravel())
  out = _pnu_fm26_kernel(tf, inv_g2sq, pfac*om_pref*g2fac, pfac*A1*g2fac,
                         log_psi_pref + np.log(pfac) + pm1h*np.log(inv_g2sq),
                         a1, a2, a3, a4, pm1h, c1*eta**(2.*al2), al2, al4,
                         _R_logR, _R_LOGXMIN, _R_INV_DLOG, _R_XMIN, _R_XMAX)
  return float(out[0]) if tnu_arr.ndim == 0 else out.reshape(tnu_arr.shape)


# Analytic constant-hydro case
def get_Fnu_cell_analytic(nuobs, Tobs, gmin0, gmax0, hydro_const, env,
    Nt=200, r_ref=1.2, Ng=NG_FLUX, width_tol=1.1):
  '''
  F_nu(T) for analytic case with constant hydrodynamics (cartesian geometry)
  Uses logarithmic time binning optimized for cooling

  Parameters:
  - nuobs: observed frequency array
  - Tobs: observation time
  - gmin0, gmax0: initial electron distribution bounds
  - hydro_const: dict with constant values {'rho', 'vx', 'p', 't0', 'x0'}
  - env: environment object
  - Nt: number of logarithmic time bins
  - r_ref: reference ratio for cooling sub-bins (gmax_i / gmax_{i+1})
  - Ng: number of gamma bins for distribution integration
  - width_tol: tolerance for narrow distribution approximation
  '''

  from phys_functions import derive_Lorentz, derive_Pmax

  # Extract constant hydro quantities
  rho = hydro_const['rho']
  vx = hydro_const['vx']
  p = hydro_const['p']
  t0 = hydro_const['t0']
  x0 = hydro_const['x0']

  # Compute derived constant quantities
  lfac = derive_Lorentz(vx)
  Pmax = derive_Pmax(rho, p, env.rhoscale, env.eps_B, env.xi_e)

  # Cooling time
  from variables import var2func
  syn_func, syn_inlist, syn_envlist = var2func['syn']
  # syn depends on rho, p, so we can compute it with constant values
  tc1 = 1.0 / syn_func(rho, p, env.rhoscale, env.eps_B)

  # Total evolution time in normalized units
  # Go from tt=0 until gmax cools to 1 (non-relativistic electrons)
  # From gamma_synCooled: gamma(tt) = gamma0 / (1 + gamma0*tt)
  # So: tt = 1/gamma - 1/gamma0
  # At t=0: tt_0 = 0 corresponds to gamma = gamma0
  # At t_final: gamma = 1 gives tt_final = 1/1 - 1/gamma0 ≈ 1
  gmax_final = 1.0
  tt_0 = 0.0  # Start at injection time
  tt_final = 1.0/gmax_final - 1.0/gmax0

  # Create logarithmic time bins in normalized time tt
  # Need to avoid tt=0, so start from a small value
  tt_min = max(1e-8, 1.0/(10*gmax0))  # Small but non-zero
  tt_bins = np.geomspace(tt_min, tt_final, Nt+1)
  dtt_arr = np.diff(tt_bins)
  tt_arr = tt_bins[:-1]  # Use left edge of each bin

  # Convert to proper time
  dtp_arr = dtt_arr * tc1
  tp_arr = (tt_arr - tt_bins[0]) * tc1

  # Lab frame time (for cartesian: t = tp * lfac)
  t_arr = t0 + tp_arr * lfac

  # Position evolution (for cartesian: x = x0 + vx * (t - t0))
  x_arr = x0 + vx * (t_arr - t0)

  # Lab frame time intervals (for computing dx)
  dt_arr = dtp_arr * lfac

  # Spatial intervals (for cartesian: dx = vx * dt)
  dx_arr = vx * dt_arr

  # Evolve gamma distribution
  gmin_arr = gamma_synCooled(tt_arr, gmin0)
  gmax_arr = gamma_synCooled(tt_arr, gmax0)

  # Initialize output
  Fnu_out = np.zeros(nuobs.shape)

  # Normalization for power-law distribution
  K0 = norm_plaw_distrib(gmin0, gmax0, env.psyn)

  # Loop over time bins
  for i in range(Nt):
    if gmax_arr[i] <= 1.0:
      break

    # Create a pseudo-step dictionary with the necessary fields
    step = {
      'gmin': gmin_arr[i],
      'gmax': gmax_arr[i],
      'dtp': dtp_arr[i],
      'rho': rho,
      'vx': vx,
      'p': p,
      't': t_arr[i],
      'x': x_arr[i],
      'dx': dx_arr[i],
      'dt': dt_arr[i]
    }

    # Convert to pandas Series for compatibility with get_Fnu_step
    import pandas as pd
    step_series = pd.Series(step)

    # Calculate flux contribution from this time bin
    Fnu = get_Fnu_step(nuobs, Tobs, step_series, K0, env, Ng, norm=True, width_tol=width_tol)
    Fnu_out += Fnu

  return Fnu_out


def get_cell_nuFnu_analytic(dist, env, Nt=200, Nnu=500, width_tol=1.1):
  '''
  Wrapper to compute spectrum for analytic constant-hydro case
  Uses optimal logarithmic time binning instead of simulation timesteps

  Parameters:
  - dist: dataframe from generate_cellDistrib (only first row is used for initial conditions)
  - env: environment object
  - Nt: number of logarithmic time bins for Riemann integration
  - Nnu: number of frequency points
  - width_tol: tolerance for narrow distribution approximation

  Returns:
  - nub: normalized frequency array (nu/nu0)
  - nF: normalized flux nu*F_nu
  '''

  # Get initial conditions from first timestep
  cell0 = dist.iloc[0]
  gmin0 = cell0.gmin
  gmax0 = cell0.gmax

  # Extract constant hydro values
  hydro_const = {
    'rho': cell0.rho,
    'vx': cell0.vx,
    'p': cell0.p,
    't0': cell0.t,
    'x0': cell0.x
  }

  # Create frequency array
  log_nuM = 2*np.log10(gmax0/gmin0)
  nub = np.logspace(-8, log_nuM+1, Nnu)
  nuobs = nub * env.nu0

  # Observation time
  Tobs = env.Ts

  # Compute spectrum with optimal binning
  Fnu = get_Fnu_cell_analytic(nuobs, Tobs, gmin0, gmax0, hydro_const, env,
                               Nt=Nt, width_tol=width_tol)

  # Return normalized values
  nF = nub * Fnu

  return nub, nF