# -*- coding: utf-8 -*-
# @Author: acharlet

'''
Per-cell cooling Lorentz factor gamma_c(t; t_s) and cooling frequency nu_c(t), the
ingredients of a semi-analytical model for the cooling break of the FULL-SHELL
spectrum -- the counterpart of the semi-analytical nu_m (peak_modeling.C25_peak_model).

The physics is the exact solution of the electron cooling equation
    dgamma/dt' = -A(t') gamma - a(t') gamma^2,
    a = K_syn B'^2 (synchrotron),  A = -(1/3) dln n'/dt' (adiabatic),
in the limit gamma_0 -> infinity, i.e. the largest Lorentz factor that has had time to
cool since the cell was shocked at t_s:
    gamma_c(t;t_s) = [n'(t)/n'(t_s)]^(1/3) / ( K_syn int_{t_s}^t B'^2 [n'/n'(t_s)]^(1/3) dt' ).

This module does NOT re-integrate that. The emission pipeline already advances electrons
with the operator split (working_cooling.evolve_gma_bounds_edges)
    f_syn = 1/(1 + g_j dtt_j) ;  g_{j+1} = g_j f_syn (rho_{j+1}/rho_j)^(1/3)
over the cooling fluence tt = int K_syn B'^2 dt' (the frame's `tt`/`dtt` columns, with
K_syn = phys_constants.alpha_). In u = 1/g that split is LINEAR,
    u_{j+1} = (u_j + dtt_j) / (rho_{j+1}/rho_j)^(1/3),
so starting from u_0 = 0 telescopes to the closed form
    gamma_c[n] = rho[n]^(1/3) / sum_{j<n} dtt_j rho_j^(1/3),
which is the integral above discretised (the n'(t_s) normalisations cancel). gamma_c is
therefore a cumulative sum over columns every cell frame already carries -- no new ODE,
and by construction consistent with the electron evolution the emission actually uses.

Two consequences worth keeping in mind:

  gamma_c IS the cutoff of what the cell radiates. cooled_tt_eff(gmax, bsyn) =
  (1-bsyn)/gmax is the effective fluence handed to distrib_plaw_cooled, and in the
  gmax0 -> infinity limit 1/cooled_tt_eff is exactly the gamma_c above. So a single cell
  emits ONE cooled power law with a cutoff at gamma_c, not a broken spectrum. This is
  checked directly (nuc_validation.check_against_cooled_cutoff).

  The shell's cooling BREAK is a superposition effect. Cells shocked at different t_s
  carry different gamma_c, and the break in the summed spectrum sits where the
  longest-cooled still-bright population runs out -- which is why the effective nu_c is
  expected to follow a low quantile of {nu_c,i}, not their mean. Choosing among those
  estimators is what nuc_validation measures.

Frequency convention: nu = SYN_FAC * Dop * nu'_B * gamma^2 with SYN_FAC = 3/2, since the
note's C_syn = 3 q_e/(4 pi m_e c) is (3/2) x nu'_B/B'. The env scalars do NOT carry that
3/2 (env.nu0 = 2 lfac0 gma_m^2 nuBp/(1+z) is a bare gamma^2 nu_B), so `normed=True`
output is in units of env.nu0 and a cell at injection lands at SYN_FAC, not 1. Same
convention as working_cooling.get_critfreqs (which divides by 1.5*env.nu0p).

Pipeline-agnostic: only the shared cell-frame columns (tt, dtt, rho, t, x, dx, vx, p) are
used, so frames from working_cooling.generate_cell_withDistrib and
working_cooling_data.generate_cell_fromHistory are both accepted.
'''

import numpy as np
import pandas as pd

from IO import get_variable, open_celldata
from analysis_hydro import extract_data_cells
from obs_functions import obs_arrays
from environment import rescale_proper_velocities, rescale_hydro
from working_cooling import (check_extracted_cells, load_shell_rarefaction, rar_map_lookup,
    compute_subcell_edges, DLNRHO_MAX)
from working_cooling_data import (select_postshock_rows, generate_cell_fromHistory,
    load_shockfront_states, _prepend_shocked_row, _interp_state, _subcell_history,
    DLNSYN_MAX)

# 3/2 from the argument of R(x) in the full synchrotron computation (phys_functions.func_R
# / syn_emiss_exact): the characteristic frequency of an electron at gamma is
# (3/2) gamma^2 nu'_B, while env.nu0 and env.nu0p are built from gamma^2 nu_B alone.
# Set to 1. to work in the bare env.nu0 convention instead.
SYN_FAC = 1.5


def cooling_lfac_edges(tt_edges, rho_edges):
  '''
  gamma_c at every step edge: the gamma_0 -> infinity limit of
  working_cooling.evolve_gma_bounds_edges, in closed form.

  Writing u = 1/g, that routine's operator split is u_{j+1} = (u_j + dtt_j)/adiab_j with
  adiab_j = (rho_{j+1}/rho_j)^(1/3); from u_0 = 0 this telescopes to
    u_n = sum_{j<n} dtt_j (rho_j/rho_n)^(1/3),
  i.e. the cooling integral of the note normalised at the injection density (which
  cancels). Same argument convention as evolve_gma_bounds_edges: len(rho_edges) values
  from len(rho_edges)-1 steps, so the two can be compared edge by edge.

  Returns an array of len(rho_edges); the first entry is +inf (nothing has cooled yet).
  '''
  tt_edges = np.asarray(tt_edges, float)
  r13 = np.asarray(rho_edges, float)**(1./3.)
  dtt = np.diff(tt_edges)
  # S_n = sum_{j<n} dtt_j rho_j^(1/3): left-edge rho weighting, as the split applies the
  # step's synchrotron burn BEFORE its adiabatic factor
  S = np.concatenate(([0.], np.cumsum(dtt*r13[:-1])))
  with np.errstate(divide='ignore', invalid='ignore'):
    return r13/S


def cell_cooling_lfac(cell):
  '''
  gamma_c(t) along a cell frame, one value per stored step (its LEFT edge), from the
  frame's own tt/dtt/rho columns.

  The frame stores left-edge values only, so the right-edge rho of the last step is not
  available -- but gamma_c at a left edge never needs it (S_n runs over j < n), hence the
  placeholder closing value below, whose only effect would be on the dropped edge N.

  cell[0] is the injection state: gamma_c there is +inf by construction.
  '''
  tt = cell['tt'].to_numpy(dtype=float)
  dtt = cell['dtt'].to_numpy(dtype=float)
  rho = cell['rho'].to_numpy(dtype=float)
  n = len(tt)
  tt_e = np.append(tt, tt[-1] + dtt[-1])
  rho_e = np.append(rho, rho[-1])            # placeholder, see docstring
  return cooling_lfac_edges(tt_e, rho_e)[:n]


def cell_cooled_cutoff_lfac(cell):
  '''
  The SAME quantity read off the electron bounds the pipeline already evolved:
  1/cooled_tt_eff(gmax, bsyn). Exact only as gmax0 -> infinity, so it is the
  finite-injection version of cell_cooling_lfac and sits slightly above it -- they
  converge when gamma_c << gmax0 = gma_M. Used as an independent check, not in models.
  '''
  gmax = cell['gmax'].to_numpy(dtype=float)
  bsyn = cell['bsyn'].to_numpy(dtype=float)
  with np.errstate(divide='ignore', invalid='ignore'):
    return gmax/(1. - bsyn)


def nu_from_lfac(gma, cell, env, normed=True, syn_fac=SYN_FAC):
  '''
  Observed characteristic synchrotron frequency of electrons at Lorentz factor gma,
  for every step of a cell frame:  nu = syn_fac * Dop * nu'_B * gma^2.

  normed=True divides by env.nu0, i.e. the x = nuobs/env.nu0 axis on which
  sweep_gammacm.track_breaks_gs02 reports the fitted breaks (nu_over_num). Dop already
  carries 1/(1+z) and env.nu0 the matching 2*lfac0/(1+z), so the redshift cancels.
  '''
  Dop = get_variable(cell, 'Dop', env)
  nup_B = get_variable(cell, 'nup_B', env)
  nu = syn_fac * Dop * nup_B * np.asarray(gma, dtype=float)**2
  return nu/env.nu0 if normed else nu


def cell_cooling_freq(cell, env, normed=True, syn_fac=SYN_FAC):
  '''
  (barT, nu_c) along a cell frame: the line-of-sight arrival time of each step in the
  standard normalisation barT = (Ton - env.Ts)/env.T0 -- the axis of
  track_breaks_gs02 -- and the cell's cooling frequency there.

  barT is the ONSET of the step's arrival; its emission actually spreads over
  [Ton, Ton+Tth] on the equal-arrival-time surface (see cell_cooling_table, which carries
  Tth for estimators that need to weight over that spread).
  '''
  barT = (get_variable(cell, 'Ton', env) - env.Ts)/env.T0
  return np.asarray(barT, dtype=float), nu_from_lfac(cell_cooling_lfac(cell), cell, env,
                                                     normed=normed, syn_fac=syn_fac)


GMA_FLOOR = 1.     # electrons stop cooling when they become non-relativistic


def cell_cooling_table(cell, env, normed=True, syn_fac=SYN_FAC, gma_floor=GMA_FLOOR):
  '''
  Per-step diagnostic table for one cell: the three characteristic frequencies alongside
  the raw ingredients an emission weight needs, so the estimator layer can build weights
  without re-deriving anything.

  Columns
    barT, Tth_b : arrival onset and angular spread, both in units of env.T0
    gma_c       : cell_cooling_lfac (the model quantity)
    gma_cut     : cell_cooled_cutoff_lfac (finite-gmax0 twin, for the identity check)
    gma_m, gma_M: the frame's evolved bounds gmin, gmax
    nu_c, nu_m, nu_M : the corresponding frequencies (nu_from_lfac; env.nu0 units if normed)
    Pmax, V3p, nup_B, dtp, Dop, Kad : emission-weight ingredients (see
                  cell_radiated_energy, whose comoving per-step energy goes as
                  Kad*Pmax*V3p*nup_B*dtp, with Kad = A^(p-1) the adiabatic
                  renormalisation of the electron count)
    fresh       : cooling has not yet reached the injected cutoff (no cooling break)
    spent       : the raw gamma_c fell below gma_floor, i.e. the cell has cooled out

  gma_floor clips gamma_c from below: the synchrotron loss rate carries a factor
  (gamma^2 - 1) (see cooling_distribution.coolingFunc_ODE), so cooling stops once
  electrons turn non-relativistic and gamma_c cannot fall past 1 -- the same gamma = 1
  floor that sweep_gammacm.track_breaks_gs02 reports as nu_B = 1/gma_m^2. The closed form
  has no such floor (it is the solution of the ultra-relativistic equation), so deeply
  cooled cells run below it and must be clipped here. Pass gma_floor=None for the raw
  quantity, which is what check_identity compares.
  '''
  gma_c = cell_cooling_lfac(cell)
  spent = gma_c < (gma_floor if gma_floor is not None else -np.inf)
  if gma_floor is not None:
    gma_c = np.maximum(gma_c, gma_floor)
  gmin = cell['gmin'].to_numpy(dtype=float)
  gmax = cell['gmax'].to_numpy(dtype=float)
  out = pd.DataFrame({
      'barT':   (get_variable(cell, 'Ton', env) - env.Ts)/env.T0,
      'Tth_b':  get_variable(cell, 'Tth', env)/env.T0,
      'gma_c':  gma_c,
      'gma_cut': cell_cooled_cutoff_lfac(cell),
      'gma_m':  gmin,
      'gma_M':  gmax,
      'nu_c':   nu_from_lfac(gma_c, cell, env, normed=normed, syn_fac=syn_fac),
      'nu_m':   nu_from_lfac(gmin, cell, env, normed=normed, syn_fac=syn_fac),
      'nu_M':   nu_from_lfac(gmax, cell, env, normed=normed, syn_fac=syn_fac),
      'Pmax':   get_variable(cell, 'Pmax', env),
      'V3p':    get_variable(cell, 'V3p', env),
      'nup_B':  get_variable(cell, 'nup_B', env),
      'dtp':    cell['dtp'].to_numpy(dtype=float),
      'Dop':    get_variable(cell, 'Dop', env),
      # adiabatic renormalisation of the electron count (radiation_cooling.get_epnu);
      # 1 for frames predating the column, which is its no-expansion value
      'Kad':    (cell['Aad'].to_numpy(dtype=float)**(env.psyn - 1.)
                 if 'Aad' in cell else np.ones(len(cell))),
      })
  # gamma_c is 1/cooled_tt_eff, so it lies just ABOVE the evolved gmax at EVERY step
  # (identity 2, exact): comparing the two says nothing. Cooling has bitten only once
  # gamma_c has dropped below the cutoff the cell was INJECTED with -- before that no
  # electron has cooled and the cell has no cooling break at all.
  out['fresh'] = out.gma_c > float(gmax[0])
  out['spent'] = spent
  out.attrs = dict(i=int(cell.iloc[0].i), trac=float(cell.iloc[0].trac),
                   normed=bool(normed), syn_fac=float(syn_fac),
                   gma_floor=gma_floor)
  return out


# ---------------------------------------------------------------------------
# shell-level harvest: the population {nu_c,i(barT)} without the emission kernel

def iter_shell_cells(key, z=4, u_scale=1., alpha=1., zeta=1., klist=None, r_ref=1.1,
    Tmax=100, NT=250, Tb_min=1e-4, Tb_lin=(0.5, 9., 200), dlnrho_max=DLNRHO_MAX,
    dlnsyn_max=DLNSYN_MAX, n_settle=1, subcell_dlogT=0.02, subcell_max=32,
    early_ana='shockfit', early_frac=0.1, rar_cut='model', verbose=True):
  '''
  Yield (k, cell, cell_env) for every cell and sub-cell of shell z, exactly as
  working_cooling_data.get_shell_nuFnu_fromData builds them -- same
  select_postshock_rows / _prepend_shocked_row / compute_subcell_edges /
  rar_map_lookup / generate_cell_fromHistory calls, same defaults as
  sweep_gammacm._compute_point passes for method='data_rarcut' -- but stopping short of
  get_nuFnu. Cost is the cell construction only, so a whole shell is seconds rather than
  the minutes an emission pass takes.

  The defaults MUST track sweep_gammacm's (EARLY_ANA, SUBCELL_DLOGT, SUBCELL_MAX, R_REF,
  TB_MIN, TB_LIN, TMAX, NT) or the harvested population no longer corresponds to the
  cached spectra it is compared against.
  '''
  nub, T, env = obs_arrays(key, normed=True, Tmax=Tmax, NT=NT, Tb_min=Tb_min, Tb_lin=Tb_lin)
  env_rs = rescale_proper_velocities(u_scale, env) if u_scale != 1. else env
  if alpha != 1. or zeta != 1.:
    env_rs = rescale_hydro(alpha, zeta, env_rs)

  k4 = env.Next
  kCD = k4 + env.Nsh4
  k1 = kCD + env.Nsh1
  kmin, kmax = (k4, kCD) if (z == 4) else (kCD, k1)
  if klist is None:
    klist = np.arange(kmin, kmax)
    if z == 4: klist = np.flip(klist)
  done = check_extracted_cells(key)
  todo = [k for k in klist if k not in done]
  if todo:
    extract_data_cells(key, todo, noOut=True)

  if rar_cut is None:
    rar_map = None
  elif rar_cut == 'model':
    rar_map = load_shell_rarefaction(key, z, env, n_shell=len(klist))
  else:
    raise ValueError(f"rar_cut must be None or 'model', got {rar_cut!r}")

  sh_data = load_shockfront_states(key, z, env, source=early_ana) \
            if early_ana is not None else None

  # first pass: post-shock histories + injection rows, in onset order (see the driver)
  hists, attrs_l, inj_rows = [], [], []
  for k in klist:
    cell_data = open_celldata(key, k)
    if cell_data is False:
      hists.append(None); attrs_l.append(None); inj_rows.append(None)
      continue
    t_off = cell_data.t.iloc[0] if (len(cell_data) and cell_data.index[0] == 0) else 0.
    shocked = select_postshock_rows(cell_data, n_settle)
    if len(shocked) < 2:
      hists.append(None); attrs_l.append(None); inj_rows.append(None)
      continue
    shocked['t'] = shocked['t'] - t_off
    if sh_data is not None:
      sel = sh_data.loc[sh_data.i == k]
      if len(sel):
        shocked = _prepend_shocked_row(shocked, sel.iloc[0], env, early_frac)
    hists.append(shocked); attrs_l.append(cell_data.attrs); inj_rows.append(shocked.iloc[0])

  barT_grid = T - 1.
  floor = barT_grid[barT_grid > 0.].min()
  sub_edges = [None]*len(klist)
  if subcell_dlogT is not None:
    barT_on = np.array([((get_variable(r, 'Ton', env) - env.Ts)/env.T0) if r is not None else np.nan
                        for r in inj_rows])
    finite = np.flatnonzero(np.isfinite(barT_on))
    if len(finite):
      barT_on[finite[0]] = 0.
    sub_edges = compute_subcell_edges(barT_on, floor, subcell_dlogT, subcell_max)

  kw = dict(u_scale=u_scale, alpha=alpha, zeta=zeta, r_ref=r_ref, Tmax=Tmax,
            dlnrho_max=dlnrho_max, dlnsyn_max=dlnsyn_max)
  skipped = []
  for idx, k in enumerate(klist):
    if hists[idx] is None:
      skipped.append(k); continue
    if sub_edges[idx] is None:
      cell, cell_env = generate_cell_fromHistory(hists[idx], attrs_l[idx], env,
          rar_ratio=(rar_map_lookup(rar_map, k) if rar_map is not None else None), **kw)
      if cell is False:
        skipped.append(k); continue
      yield k, cell, cell_env
    else:
      a, b, edges = sub_edges[idx]
      inj_par = inj_rows[idx]
      for j in range(len(edges)-1):
        e0, e1 = edges[j], edges[j+1]
        onset_c = np.sqrt(e0*e1)
        f = float(np.clip((onset_c - a)/(b - a), 0., 1.))
        dx_w = inj_par.dx * (e1 - e0)/(b - a)
        sub = _interp_state(inj_par, inj_rows[idx+1], f, dx_w)
        barT_sub = (get_variable(sub, 'Ton', env) - env.Ts)/env.T0
        beta = sub['vx']
        delta = (onset_c - barT_sub)*env.T0/((1. + env.z)*(1. - beta))
        sub['t'] += delta
        sub['x'] += beta*delta
        hist = _subcell_history(hists[idx], sub)
        cell, cell_env = generate_cell_fromHistory(hist, attrs_l[idx], env,
            rar_ratio=(rar_map_lookup(rar_map, sub['i']) if rar_map is not None else None), **kw)
        if cell is False:
          continue
        yield k, cell, cell_env
  if skipped and verbose:
    print(f'iter_shell_cells on {key} z={z}: skipped {len(skipped)} cells '
          f'(no data or no usable post-shock history)')


def harvest_shell_cooling(key, z=4, alpha=1., normed=True, syn_fac=SYN_FAC, **kwargs):
  '''
  The population {nu_c,i(barT)} of a whole shell: one cell_cooling_table per cell/sub-cell
  yielded by iter_shell_cells, concatenated with a `cell` column identifying the emitter.

  Frequencies are in units of env.nu0 of the RESCALED env when normed=True -- the same
  RS-normalised axis the drivers build their observer grids on for either shell, so a z=1
  harvest is directly comparable with a z=1 cached sweep (see sweep_shells).

  Returns (env_rs, DataFrame).
  '''
  frames, env_rs = [], None
  for n, (k, cell, cell_env) in enumerate(iter_shell_cells(key, z=z, alpha=alpha, **kwargs)):
    env_rs = cell_env
    t = cell_cooling_table(cell, cell_env, normed=normed, syn_fac=syn_fac)
    t['cell'] = k
    t['emitter'] = n                     # cell OR sub-cell: the independent emitting unit
    frames.append(t)
  if not frames:
    raise RuntimeError(f'no usable cells for {key} z={z}')
  return env_rs, pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# from the population {nu_c,i} to ONE effective nu_c: the estimator candidates

DOP_EXP = 3.       # Doppler exponent of the per-step observed weight (see step_weights)
BRIGHT_FRAC = 1e-2  # `min_bright` ignores steps below this fraction of the brightest


def step_weights(H, dop_exp=DOP_EXP):
  '''
  Per-step observed emission weight, as a flux-like density in observer time.

  The comoving energy a step radiates goes as Kad*Pmax*V3p*nu'_B*dt' (the prefactor of
  radiation_cooling.step_radiated_energy, summed by cell_radiated_energy); it is boosted
  by Dop**dop_exp and arrives spread over the step's angular window Tth, so the density
  in observer time is that energy divided by Tth.

  Only RATIOS between steps at the same observer time matter to every estimator here, so
  the absolute normalisation and the exact dop_exp are second order -- which is testable,
  and is why dop_exp is a parameter rather than a constant. Kad = A^(p-1) does NOT drop
  out of those ratios, though: A falls along a worldline, so it de-weights late steps
  relative to early ones. It is the same factor step_radiated_energy applies, and
  omitting it here would leave the weights inconsistent with the energy they mirror.
  '''
  return (H.Kad*H.Pmax*H.V3p*H.nup_B*H.dtp*H.Dop**dop_exp/H.Tth_b).to_numpy(dtype=float)


def _weighted_quantile(x, w, q):
  '''weighted quantile of x (already finite, w >= 0), q in [0,1]'''
  o = np.argsort(x)
  x, w = x[o], w[o]
  c = np.cumsum(w)
  if c[-1] <= 0.:
    return np.nan
  return float(np.interp(q*c[-1], c - 0.5*w, x))


def effective_nu_c(H, barT_grid, estimators=None, dop_exp=DOP_EXP,
    bright_frac=BRIGHT_FRAC, min_steps=4, min_emitters=3):
  '''
  Collapse the population {nu_c,i} onto one nu_c,eff(barT) per estimator.

  At observer time barT a step is ACTIVE if barT lies in its arrival window
  [barT_on, barT_on + Tth] -- the equal-arrival-time box of a thin shell. Each estimator
  is then a different summary of the active steps' nu_c, weighted by step_weights:

    first_cell : nu_c of the earliest-shocked active emitter (representative-cell model)
    min_bright : minimum nu_c over active steps carrying more than bright_frac of the
                 peak weight -- the superposition expectation, since the break sits where
                 the longest-cooled still-bright population runs out
    q10 / q25 / q50 : weighted quantiles of nu_c
    wlogmean   : weighted mean of log nu_c

  Steps whose cooling has not yet reached the injected cutoff ("fresh", gamma_c > gamma_M)
  carry no cooling break and are excluded from every estimator.

  min_emitters guards the very end of the shell's life, where all but a handful of cells
  have gone dark behind the rarefaction: a quantile over one or two surviving emitters is
  not a property of the population and shows up as a spurious late-time spike.

  Returns dict name -> array over barT_grid (NaN where too few steps/emitters are active).
  '''
  if estimators is None:
    estimators = ('first_cell', 'min_bright', 'q10', 'q25', 'q50', 'wlogmean')
  on = H.barT.to_numpy(dtype=float)
  off = on + H.Tth_b.to_numpy(dtype=float)
  nu = H.nu_c.to_numpy(dtype=float)
  w = step_weights(H, dop_exp)
  emit = H.emitter.to_numpy() if 'emitter' in H else np.zeros(len(H), dtype=int)
  ok = np.isfinite(nu) & (nu > 0.) & np.isfinite(w) & (w > 0.) & ~H.fresh.to_numpy(dtype=bool)
  on, off, nu, w, emit = on[ok], off[ok], nu[ok], w[ok], emit[ok]
  ln = np.log(nu)

  out = {n: np.full(len(barT_grid), np.nan) for n in estimators}
  for it, bT in enumerate(barT_grid):
    a = (on <= bT) & (bT <= off)
    if a.sum() < min_steps or len(np.unique(emit[a])) < min_emitters:
      continue
    nu_a, w_a, ln_a, on_a = nu[a], w[a], ln[a], on[a]
    for n in estimators:
      if n == 'first_cell':
        out[n][it] = nu_a[np.argmin(on_a)]
      elif n == 'min_bright':
        b = w_a >= bright_frac*w_a.max()
        out[n][it] = nu_a[b].min() if b.any() else np.nan
      elif n == 'wlogmean':
        out[n][it] = np.exp(np.sum(w_a*ln_a)/np.sum(w_a))
      elif n.startswith('q'):
        out[n][it] = np.exp(_weighted_quantile(ln_a, w_a, float(n[1:])/100.))
      else:
        raise ValueError(f'unknown estimator {n!r}')
  return out


def compare_to_track(nu_eff, tr, barT_lo=1e-2, barT_hi=None):
  '''
  The constant C of the note's Eq. (21): fitted nu_c = C * nu_c,estimator.

  Measured as the median ratio over the bins the TRACKER itself calls measured
  (`valid` -- which already drops the VFC bins where nu_c is not observed, the FC/SC
  crossing where it is not identified, and anything off-window), restricted to
  barT >= barT_lo because below that the sub-cell reconstruction dominates the
  population and the fit has barely any dynamic range.

  Returns (C, spread, n) with spread the 16-84 percentile half-range in dex.
  '''
  b, v = tr['barT'], tr['valid']
  m = v & np.isfinite(tr['nu_c']) & (tr['nu_c'] > 0.) & np.isfinite(nu_eff) & (nu_eff > 0.) \
      & (b >= barT_lo)
  if barT_hi is not None:
    m &= (b <= barT_hi)
  if m.sum() < 4:
    return np.nan, np.nan, int(m.sum())
  rat = tr['nu_c'][m]/nu_eff[m]
  lo, hi = np.percentile(np.log10(rat), [16, 84])
  return float(np.median(rat)), float(0.5*(hi - lo)), int(m.sum())
