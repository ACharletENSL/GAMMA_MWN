# -*- coding: utf-8 -*-
# @Author: acharlet

'''
File to write new functions and test them in a notebook
(easier to modify/debug in file than in notebook)
'''

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from environment import MyEnv, rescale_proper_velocities, rescale_hydro
from analysis_hydro import extract_data_cells, extract_data_thinshell
from analysis_thinshell import *
from fits_hydro import *
from phys_functions import *
from phys_constants import *
from plotting_functions import *
from IO import *
from cooling_distribution import *
from radiation_cooling import *
from obs_functions import *

# default cap on ln(rho) spanned by a single cooling step (see worldline_from_cooling
# / _refine_tt_on_rho): the tt bins measure synchrotron fluence, which under-resolves
# the adiabatic (expansion-dominated) evolution of the inner-shell cells in the
# slow-cooling regime (they survive to R_rar ~ 2.5 R0, total dln(rho) ~ 1.2). Value
# set by a full-shell convergence study at log10(gma_c/gma_m)=+2: shell nuFnu vs the
# finest cap is off by ~15% at None, ~7-9% at 0.15, ~3-4% at 0.075. Fast cooling is
# edge-dominated (small expansion) and unchanged by the cap. None disables refinement.
DLNRHO_MAX = 0.075



def get_cellvals(name, data, env, cell_d0=None):
  '''
  Return the values of a cell, normalized by the values in cell_d0
  cell_d0 can be the expectations downstream of the shock
  '''
  val = get_variable(data, name, env)
  if cell_d0 is not None:
    norm = get_variable(cell_d0, name, env)
    val /= norm
    #ax.axhline(1., c='k', ls=':', lw=.9)
  y = val
  return y

def plot_data(name, data, env, cell_d0=None, xvar='x',
    imax=None, ax_in=None, logx=False, logy=False, logslope=False, **kwargs):
  '''
  Function for raw data plotting
  Normalization is computed from reconstructed downstream data sh_data
  '''

  if ax_in is None:
    fig, ax = plt.subplots()
  else:
    ax = ax_in
  if xvar == 'x':
    x = data.x.to_numpy() * c_ /env.R0
    xlabel = '$R/R_0$'
  elif xvar == 'it':
    x = data.index
    xlabel = 'it'
  elif xvar == 't':
    x = data.t.to_numpy() / env.t0
    xlabel = '$t/t_0$'
  if imax is not None:
    index = data.index
    xmax = x[index<=imax][-1]
  else:
    xmax = x[-1]
  
  y = get_cellvals(name, data, env, cell_d0=cell_d0)
  if logslope:
    y = logslope_arr(x, y)
  ax.plot(x[x<=xmax], y[x<=xmax], **kwargs)
  if logx: ax.set_xscale('log')
  if logy: ax.set_yscale('log')
  ax.set_xlabel(xlabel)
  if ax_in is None:
    ax.set_title(name)

def truncate_at_rarefaction(shocked_data, slope_thresh=-8., dlnx_window=0.02, minpts=20):
  '''
  Truncate post-shock cell data at the rarefaction arrival.
  The smooth downstream decay has d(ln p)/d(ln x) ~ -2 to -4; the back-edge
  rarefaction makes p crash with log-slopes <~ -15. Truncate at the first
  point where the look-ahead log-slope of p drops below slope_thresh.
  Keeps at least minpts points (else returns the data untruncated).

  THE LOOK-AHEAD SPANS A FIXED Delta ln x (fixed 2026-09-07). The slope being tested is
  d(ln p)/d(ln x), the crash is a feature in RADIUS, and the fit lives in x/x0, so radius
  is the only unit in which this window means the same thing on every run. It used to be
  a fixed number of ROWS -- equivalently of iterations, which is the same bookkeeping unit
  under another name -- and neither survives a change of resolution or of dump cadence:

  - Rows, at cooling_g100_hires: 15 rows near injection span Delta ln x ~ 1.5e-3 against
    the fiducial's ~2.2e-2, short enough that the window fires on the post-shock SETTLING
    TRANSIENT. Cell 2954's fit window was cut to 20 rows over x/x0 = 1.0450..1.0466, a
    0.16% radius range; the BPL fitted to that reaches Gamma = 8e-26 at x/x0 = 10 against
    a measured 126, _lag_profile kills the cell at Gamma <= 1, and the head never crosses
    it. 407 of 10000 RS cells, with survivors carrying fits just as wrong (cell 2955:
    Gamma = 5.15), which is where R_rar = 25-61 and the barT_off reversals came from.
  - Cell crossings (11.6 iterations): WORSE, 407 -> 2696 non-finite. A cell's crash sits
    at R/R_inj ~ 1-4, which for a late-shocked cell falls in the cadence-50 or -500 phase,
    so a 10-crossing look-ahead spans ~2 rows there against ~58 near injection. The
    detector then never fires at all and every fit swallows its own crash.

  dlnx_window = 0.02 is what the fiducial's 15-row window actually spanned (median 2.2e-2
  over its shell, range 3.9e-3 to 3.7e-2), so the run whose map is known good keeps its
  behaviour and every other resolution gets the same physical window.
  '''
  x = shocked_data.x.to_numpy()
  p = shocked_data.p.to_numpy()
  if len(x) < minpts + 2:
    return shocked_data
  lnx, lnp = np.log(x), np.log(p)
  # look-ahead partner of each row: the first row at least dlnx_window further out.
  # x is monotonic along a worldline, so searchsorted is exact.
  j = np.searchsorted(lnx, lnx + dlnx_window, side='left')
  ok = j < len(lnx)
  if not ok.any():
    return shocked_data
  i = np.flatnonzero(ok)
  j = j[ok]
  with np.errstate(invalid='ignore', divide='ignore'):
    slope = (lnp[j] - lnp[i])/(lnx[j] - lnx[i])
  steep = i[np.flatnonzero(slope < slope_thresh)]
  if len(steep) and steep[0] >= minpts:
    return shocked_data.iloc[:steep[0]]
  return shocked_data


def shock_sd_block(sd, u, n_plateau=5):
  """
  The contiguous Sd run that actually straddles the shock, as (i0, i1), or None.

  Sd fires on more than the shock. The external interface launches a rarefaction into the
  shell's rear face at t=0, and the detector flags it too: on cooling_g100 z=4 cell k=20
  carries two runs -- rows 1-168 with |u_up - u_dn| = 0.2 (the rarefaction, no velocity
  jump at all) and rows 1471-1475 with 86.6 (the reverse shock). Both consumers of Sd used
  to take the FIRST run, which for that cell meant an injection at t = 23 s instead of
  4.04e4 (a factor 48 early) and a "post-shock" history starting at t = 1784 that is mostly
  unshocked, rarefying material.

  Selecting by velocity contrast is a no-op wherever there is one run, and also for the
  CD-adjacent cell k=519, whose two runs are 72.3 then 1.0 -- there the real shock IS the
  first. It changed exactly one cell of 500 on this run.
  """
  nz = np.flatnonzero(sd != 0)
  if not len(nz):
    return None
  runs = []
  s0 = p0 = int(nz[0])
  for aa in nz[1:]:
    if int(aa) == p0 + 1:
      p0 = int(aa)
    else:
      runs.append((s0, p0)); s0 = p0 = int(aa)
  runs.append((s0, p0))
  best, best_jump = runs[0], -1.
  for (rs, re) in runs:
    rlo, rhi = max(0, rs - n_plateau), min(len(u) - 1, re + n_plateau)
    a_up = float(np.median(u[rlo:rs])) if rs > rlo else float(u[rs])
    a_dn = float(np.median(u[re+1:rhi+1])) if rhi > re else float(u[re])
    jump = abs(a_up - a_dn)
    if np.isfinite(jump) and jump > best_jump:
      best, best_jump = (rs, re), jump
  return best


def _proper_velocity(vx):
  return vx/np.sqrt(np.clip(1. - vx*vx, 1e-300, None))


def postshock_start(sd, vx, n_settle=1):
  """
  First row of a cell's post-shock history: one past the Sd run that straddles the shock,
  plus n_settle. ONE definition, so a caller working from plain columns
  (IO.open_cellcolumns) and select_postshock_rows' DataFrame path cannot drift apart.
  Returns len(sd) where no run qualifies, i.e. an empty history.
  """
  blk = shock_sd_block(sd, _proper_velocity(np.asarray(vx, dtype=float)))
  return min(blk[1] + 1 + n_settle, len(sd)) if blk is not None else len(sd)


def fit_celldata(cell_data, vars, norms, env, x0=None, cleanData=False, beta=None,
    r_max=None):
  '''
  Fit the variable vars, normed by norms, of cell_data
  x0 is the anchor radius (where fitfunc==1); defaults to the first cell radius.
  Each fit is renormalized so fitfunc(x0)=1, i.e. the reconstructed profile passes
  exactly through norms at x0. With x0=cell_d0.x and norms=cell_d0 values, the
  reconstruction therefore starts exactly at the cell_d0 (downstream) state.
  The fit window is truncated at the rarefaction arrival (truncate_at_rarefaction):
  the BPL describes the smooth downstream decay, not the rarefaction cliff.

  r_max: hard cap on the fit window, in x/x0 (None = no cap). truncate_at_rarefaction
  is a HEURISTIC -- a look-ahead log-slope of p over a fixed number of ROWS -- and on a
  very long run most of a cell's history is post-crash: cooling_g100_hires carries 153224
  rows per cell out to R/R_inj ~ 700, against ~3500 rows to ~155 in the fiducial. Where
  the heuristic misses, the BPL is asked to describe the crash and the coasting tail as
  well as the smooth decay, and the fit that comes out is useless exactly where it is
  needed. A radius cap cannot miss, and the consumers only ever need the first factor of
  a few in R/R_inj. Both are applied; the tighter one wins.
  '''
  if x0 is None:
    x0 = cell_data.x.to_numpy()[0]
  popts = []
  # select data from the moment the cell is shocked: cell files hold the full
  # history from it=0, and pre-shock rows also have Sd == 0. Keep rows after
  # the first Sd != 0 (shock crossing) block, + 1 step for numerical settling.
  sd = cell_data.Sd.to_numpy()
  ish = np.flatnonzero(sd != 0)
  if len(ish):
    # the SAME start the emission uses (postshock_start): the run that straddles the
    # shock, not the first to fire. This used to be an inlined third copy taking ish[0],
    # so on a cell whose detector also flags the rarefaction -- k=20 of cooling_g100 z=4 --
    # the fit was built from ~1300 rows of unshocked, rarefying material while the
    # emission used the real post-shock history. The fits feed the rarefaction head, so
    # the two must agree on where the cell was shocked.
    start = postshock_start(sd, cell_data.vx.to_numpy(dtype=float), 1)
    shocked_data = cell_data.iloc[start:].copy()
  else:
    # no crossing recorded (pre-trimmed data): original selection
    shocked_data = cell_data.loc[(cell_data.Sd == 0)].copy().iloc[1:]
  if not len(shocked_data):
    raise RuntimeError('no post-shock data to fit')
  # fit only the smooth downstream decay, not the rarefaction cliff
  shocked_data = truncate_at_rarefaction(shocked_data)
  if r_max is not None:
    # hard radius cap, in x/x0. searchsorted is exact (x is monotonic along a worldline);
    # keep the minpts floor truncate_at_rarefaction uses so a cell caught almost at once
    # still has something to fit.
    xs = shocked_data.x.to_numpy(dtype=float)
    n_cap = int(np.searchsorted(xs, float(r_max)*float(x0), side='right'))
    if n_cap >= 20:
      shocked_data = shocked_data.iloc[:n_cap]
  # subsample long histories: ~200 points constrain the smooth profile equally
  # well and cut the curve_fit cost (numeric jacobian scales with N points)
  if len(shocked_data) > 200:
    idx = np.unique(np.linspace(0, len(shocked_data)-1, 200).astype(int))
    shocked_data = shocked_data.iloc[idx]
  for name, norm in zip(vars, norms):
    x_ = shocked_data.x.to_numpy() / x0
    val = get_variable(shocked_data, name, env)/norm
    popt = get_fitting_smoothBPL_new(x_, val, cleanData=cleanData, beta=beta)
    popt = np.asarray(popt, dtype=float)
    if beta is not None:
      popt = np.insert(popt, -1, beta)
    popt[0] /= smooth_bpl_apy(1., *popt)   # anchor: fitfunc(x0) = 1 (free amplitude A)
    popts.append(popt)
  return popts

# bump when the fitting code/conventions change, to invalidate stale caches.
# The window parameters are NOT part of the cache key -- they are code, not call
# arguments -- so this counter is the only thing that invalidates a cache when they
# change. Bump it whenever truncate_at_rarefaction or fit_celldata's selection changes.
# 2 (2026-09-07): crossing-time window, reverted the same day; 3: back to the row
# window; 4: look-ahead over a fixed Delta ln x.
_FIT_CACHE_VERSION = 5   # 5: fits start at postshock_start (the run that
                         # straddles the shock), not the first Sd run

def load_or_fit_celldata(cell_data, vars, norms, env, x0, cleanData=False,
    key=None, k=None, r_max=None):
  '''
  Disk-cached wrapper around fit_celldata: curve_fit (x3 per cell) dominates the
  cost of get_cell/shell_nuFnu, and the fit shape depends only on the cell history
  and the anchor x0 (norms and env cancel in the fitfunc(x0)=1 renormalization, and
  u_scale/alpha/zeta rescale *after* fitting), so popts are safe to cache per
  (key, k, cleanData, r_max). Cache lives next to the cell data at
  results/{key}/cells/{k:04d}_fit.npz. key/k None => no caching (just fit).

  r_max is part of the cache identity: a cache written with a different window describes a
  different curve. Files predating it have no 'r_max' key, so the read raises and they are
  refitted -- which is what a run whose fits were made over the full history needs.

  cell_data may be a ZERO-ARGUMENT CALLABLE returning the history, and then it is called
  only when a fit is actually needed. On a cache hit the history is never read at all --
  which is the whole cost of a warm shell: open_celldata is 3.6 s on a hi-res cell, so
  compute_shell_rarefaction_head was reading 10000 histories, ~10 h of pandas, to answer
  every one of them from a cache file it had not looked at yet.
  '''
  resolve = (lambda: cell_data()) if callable(cell_data) else (lambda: cell_data)
  if key is None or k is None:
    return fit_celldata(resolve(), vars, norms, env, x0=x0, cleanData=cleanData,
                        r_max=r_max)
  path = get_dirpath(key) + f'cells/{k:04d}_fit.npz'
  if os.path.isfile(path):
    try:
      d = np.load(path)
      cached_rmax = float(d['r_max'])          # KeyError on a pre-r_max cache -> refit
      same_rmax = (np.isnan(cached_rmax) if r_max is None
                   else np.isclose(cached_rmax, float(r_max)))
      if int(d['version']) == _FIT_CACHE_VERSION and bool(d['cleanData']) == bool(cleanData) \
          and np.isclose(float(d['x0']), float(x0)) and same_rmax:
        return [d['popt_rho'], d['popt_lfac'], d['popt_p']]
    except Exception:
      pass   # unreadable/old cache => refit and overwrite
  popts = fit_celldata(resolve(), vars, norms, env, x0=x0, cleanData=cleanData,
                       r_max=r_max)
  try:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, popt_rho=popts[0], popt_lfac=popts[1], popt_p=popts[2],
             x0=float(x0), cleanData=bool(cleanData), version=_FIT_CACHE_VERSION,
             r_max=(np.nan if r_max is None else float(r_max)))
  except Exception:
    pass   # caching is best-effort; never fail the pipeline on a write error
  return popts

def reconstruct_cell(R_vals, env, cell_d0, popts, fitfunc=smooth_bpl_apy):
  '''
  Reconstruct cell data over array of positions R_vals using fitting parameters
  '''

  # anchor radius is cell_d0 (downstream state), consistent with fit_celldata
  x0 = cell_d0.x
  R0 = x0 * c_
  x_vals = R_vals / R0
  vars = ['rho', 'lfac', 'p']
  rho0, lfac0, p0 = [get_variable(cell_d0, name, env) for name in vars]
  popt_rho, popt_lfac, popt_p = popts

  # reconstruct
  rho = rho0 * fitfunc(x_vals, *popt_rho)
  lfac = lfac0 * fitfunc(x_vals, *popt_lfac)
  vx = np.sqrt(np.maximum(lfac**2 - 1., 0.))/lfac   # guard Gamma<1 fit overshoot
  p = p0 * fitfunc(x_vals, *popt_p)
  dx = cell_d0.dx * rho0 * lfac0 / (x_vals**2 * rho * lfac)
  i = np.full(R_vals.shape, cell_d0.i)
  trac = np.full(R_vals.shape, cell_d0.trac)

  # normalize to code units
  x = R_vals/c_

  return i, x, dx, rho, vx, lfac, p, trac

def worldline_from_fit(tp_edges, R0, lfac0, popt_lfac, fitfunc=smooth_bpl_apy):
  '''
  Worldline parametrized by proper time t':
    dR/dt' = c sqrt(Gamma^2 - 1)   (= c beta Gamma)
    dt/dt' = Gamma                (time dilation dt = Gamma dt')
  with Gamma(R) = lfac0 * fitfunc(R/R0).
  tp_edges must start at 0 (R = R0); returns lab-frame times elapsed since then.
  '''
  def derivs(tp, y):
    R = y[0]
    Gamma = lfac0 * fitfunc(R/R0, *popt_lfac)
    Gamma2 = Gamma**2
    return [c_*np.sqrt(max(Gamma2 - 1., 0.)), Gamma]  # [dR/dt', dt/dt']
  sol = solve_ivp(derivs, (0., tp_edges[-1]), [R0, 0.],
                  method='DOP853', dense_output=True)
  R_vals, t_vals = sol.sol(tp_edges)
  return t_vals, R_vals

def _refine_tt_on_rho(tt_edges, sol, R0, rho0, popt_rho, fitfunc, dlnrho_max):
  '''
  Insert extra tt edges so no interval spans more than dlnrho_max in ln(rho),
  using the dense worldline solution sol (no re-integration). rho is monotonic
  along the worldline, so a single linear-in-tt subdivision per interval,
  n = ceil(Dln rho / dlnrho_max) sub-steps, keeps every sub-step under the cap.
  Refining tt only adds resolution: the synchrotron law 1/g = 1/g0 + tt and the
  adiabatic product both telescope, so results are unchanged for coarse grids
  and converge as the grid is refined.
  '''
  def lnrho(tt):
    R = np.atleast_1d(sol.sol(tt))[0]        # sol state = [R, t', t]; take R
    return np.log(rho0 * fitfunc(R/R0, *popt_rho))
  out = [tt_edges[0]]
  for a, b in zip(tt_edges[:-1], tt_edges[1:]):
    n = max(1, int(np.ceil(abs(lnrho(b) - lnrho(a)) / dlnrho_max)))
    out.extend(np.linspace(a, b, n + 1)[1:])
  return np.asarray(out)

def worldline_from_cooling(tt_edges, cell_d0, env, popts, R0=None, Tobs_max=None,
    R_rar=None, dlnrho_max=None, fitfunc=smooth_bpl_apy):
  '''
  Convert the dimensionless cooling fluence tt into proper time t', lab time t and
  radius R, consistently with a decreasing comoving B (evolving synchrotron rate).

  tt is the integrated cooling rate, dtt/dt' = 1/t_c1(R) = syn(R), so the analytic
  law gamma = gma0/(1 + gma0*tt) stays exact for any B(t') history; only the
  tt <-> t' mapping becomes nonlinear. Augmented worldline integrated with tt as
  the independent variable (t_c1 > 0, so tt monotonic in t'):
    dt'/dtt = t_c1(R)
    dR /dtt = c sqrt(Gamma^2 - 1) * t_c1(R)
    dt /dtt = Gamma * t_c1(R)
  with Gamma, rho, p from the fits (popts) normalized to cell_d0. syn(R) uses
  derive_syn_cooling, identical to the 'syn'/'tc1' variables elsewhere.

  R0 is the fit anchor radius (the radius at which fitfunc=1, by construction in
  fit_celldata): the fits are normalized to cell_d0.x, so x=R/R0 MUST use that same
  radius, and the worldline starts there at the cell_d0 state. Defaulting to
  cell_d0.x*c_ is therefore the consistent choice and matches reconstruct_cell.
  tt_edges must start at 0 (R = R0).

  Tobs_max caps the integration: as the shell expands the comoving cooling time
  1/syn(R) diverges, so mapping tt (which only spans ~1) to R is a runaway that
  sends R to absurd radii (R/R0 ~ 1e120) where the extrapolated fits give
  unphysical Gamma (->0 => sqrt(Gamma^2-1) NaN and the ODE stalls, or ->inf =>
  beta rounds to 1). Emission from those radii arrives at onset time Ton beyond
  the observer window and is discarded by get_Fnu_cell_evolving anyway, so we stop
  the worldline (terminal event) as soon as Ton = (1+z)(t + t0 - R/c) reaches
  Tobs_max. Ton is monotonic in tt, so this is a single lossless crossing; None
  disables the cap. tt_edges is truncated to the integrated range and returned.

  R_rar (if finite) is a second terminal event: the radius at which the rarefaction
  wave catches the cell (compute_R_rar), past which it no longer emits. The worldline
  stops at whichever of the two events (Tobs_max or R_rar) is reached first in tt.

  dlnrho_max (if set) refines the surviving tt grid so no step spans more than that
  in ln(rho), guaranteeing a fixed adiabatic sampling resolution even when little
  synchrotron fluence tt accrues over a large radius range (slow/very-slow cooling);
  None keeps the raw cooling-fluence bins. See _refine_tt_on_rho.

  Returns t_edges, tp_edges, R_edges, tt_edges (all elapsed since the cell_d0 state).
  '''
  popt_rho, popt_lfac, popt_p = popts
  if R0 is None:
    R0 = cell_d0.x * c_
  lfac0 = get_variable(cell_d0, 'lfac', env)
  rho0  = get_variable(cell_d0, 'rho', env)
  p0    = get_variable(cell_d0, 'p', env)

  def derivs(tt, y):
    R = y[0]
    x = R/R0
    Gamma = lfac0 * fitfunc(x, *popt_lfac)
    rho   = rho0  * fitfunc(x, *popt_rho)
    p     = p0    * fitfunc(x, *popt_p)
    tc1   = 1.0 / derive_syn_cooling(rho, p, env.rhoscale, env.eps_B)
    return [c_*np.sqrt(max(Gamma**2 - 1., 0.))*tc1,  # dR/dtt (guard Gamma<1)
            tc1,                            # dt'/dtt
            Gamma*tc1]                      # dt/dtt
  events = []
  if Tobs_max is not None:
    t0_cell = cell_d0.t
    def reach_window(tt, y):
      # onset observer time of emission at (R=y[0], elapsed lab time y[2]);
      # matches derive_obsTimes: Ton = (1+z)*(t_sim + t0 - R/c), r in units c
      return (1. + env.z)*(t0_cell + y[2] + env.t0 - y[0]/c_) - Tobs_max
    reach_window.terminal = True
    reach_window.direction = 1.
    events.append(reach_window)
  if R_rar is not None and np.isfinite(R_rar):
    def reach_rar(tt, y):
      return y[0] - R_rar          # R crosses the rarefaction catch-up radius
    reach_rar.terminal = True
    reach_rar.direction = 1.
    events.append(reach_rar)
  events = events or None

  # a few cells have a rho fit decaying to ~0 at large R, so the comoving cooling
  # time tc1 = 1/syn(R) diverges and derivs hits 0*inf / inf, briefly feeding a NaN
  # radius through the fit's log during intermediate steps. Benign (a terminal event
  # stops the worldline at a finite R and the returned edges are finite), so silence
  # those known warnings here.
  with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
    sol = solve_ivp(derivs, (0., tt_edges[-1]), [R0, 0., 0.],
                    method='DOP853', dense_output=True, events=events)
  # truncate tt to the integrated range if a terminal event fired (whichever of
  # Tobs_max / R_rar came first in tt). tt_edges[0]=0 survives, so >= 2 edges remain.
  fired = events is not None and any(len(te) for te in sol.t_events)
  if fired and sol.t[-1] > 0.:
    tt_edges = np.append(tt_edges[tt_edges < sol.t[-1]], sol.t[-1])
  # refine on ln(rho) so the adiabatic (expansion-dominated) evolution is
  # sampled at a fixed resolution even when little synchrotron fluence tt
  # accrues over a large radius range (slow / very-slow cooling)
  if dlnrho_max is not None:
    tt_edges = _refine_tt_on_rho(tt_edges, sol, R0, rho0, popt_rho, fitfunc, dlnrho_max)
  R_edges, tp_edges, t_edges = sol.sol(tt_edges)
  return t_edges, tp_edges, R_edges, tt_edges

def _one_minus_beta_over_beta(G):
  '''
  (1 - beta)/beta from the Lorentz factor, WITHOUT the catastrophic cancellation of
  computing beta = sqrt(G^2-1)/G first and then 1/beta - 1: at G ~ 100 that subtracts
  two numbers agreeing to 1e-5, and at G ~ 1e3 to 1e-7. Exact identity:
    1 - beta = (G - sqrt(G^2-1))/G = 1/(G*(G + sqrt(G^2-1)))
    => (1 - beta)/beta = 1/(sqrt(G^2-1)*(G + sqrt(G^2-1)))    ( -> 1/(2 G^2) for G >> 1)
  This is the integrand of the observer lag ds/dR = (1-beta)/(c*beta) (see
  compute_shell_rarefaction_head), i.e. the ONLY quantity that distinguishes two
  nearly-luminal worldlines, so it must not be built by subtraction.
  G <= 1 (fit artefact) returns inf; callers clip it.
  '''
  G = np.asarray(G, float)
  q = np.sqrt(np.clip(G*G - 1., 0., None))
  with np.errstate(divide='ignore', invalid='ignore'):
    out = 1./(q*(G + q))
  return np.where(q > 0., out, np.inf)


def _dsdR_head(G, cs, z_fwd):
  '''
  Lag integrand ds/dR = (1 - beta_h)/(c*beta_h) of the rarefaction head moving through
  fluid of Lorentz factor G and sound speed cs, with
    beta_h = (beta +/- cs)/(1 +/- beta*cs)     (+ region 3 / z=4, - region 2)
  Built from the factorised identity rather than from beta_h itself, for the same reason
  as _one_minus_beta_over_beta -- beta_h rounds to 1 well before (1 - beta_h) stops
  mattering:
    1 - beta_h = (1 - beta)(1 -/+ cs)/(1 +/- beta*cs)
    => (1 - beta_h)/beta_h = (1 - beta)*(1 -/+ cs)/(beta +/- cs)
  with 1 - beta = 1/(G*(G + sqrt(G^2-1))) exact. The denominator beta +/- cs is O(1) here
  (beta ~ 1, cs <= 1/sqrt(3)), so nothing cancels. A non-positive denominator would mean a
  head running inward; returns inf there, which stops the lag profile (_lag_profile).
  '''
  G, cs = np.asarray(G, float), np.asarray(cs, float)
  q = np.sqrt(np.clip(G*G - 1., 0., None))
  b = np.where(q > 0., q/G, 0.)
  omb = np.where(q > 0., 1./(G*(G + q)), 1.)                   # 1 - beta, exact
  num, den = ((1. - cs), (b + cs)) if z_fwd else ((1. + cs), (b - cs))
  with np.errstate(divide='ignore', invalid='ignore'):
    out = omb*num/(den*c_)
  return np.where(den > 1e-6, out, np.inf)


def _lag_profile(Rg, dsdR, R_anchor, s_anchor):
  '''
  Observer lag s(R) = t(R) + t0 - R/c of a worldline sampled on the radius grid Rg,
  from its integrand dsdR = (1-beta)/(c*beta) (already evaluated on Rg) and the anchor
  (R_anchor, s_anchor) where the worldline starts:
    s(R) = s_anchor + int_{R_anchor}^{R} (1-beta)/(c beta) dR'
  Trapezoidal on Rg, which is log-spaced (the integrand is a smooth power law of R).

  FORWARD ONLY from the anchor, and stopping at the first non-finite integrand node:
  NaN outside that range. A worldline does not exist below the radius at which its cell
  was shocked, and the extrapolated fits do go bad (a few last-shocked cells have so few
  snapshots that their lfac fit crosses Gamma = 1, where the integrand is infinite). The
  integral is a cumsum, so a single inf anywhere would otherwise poison every node after
  it and silently drop the cell from the map.
  The anchor need not be a grid node: it is prepended so the integral starts exactly
  there (a missing sub-step is ~0.8% of the shell's lag spread, enough to add visible
  cell-to-cell jitter).
  '''
  f = np.asarray(dsdR, float)
  out = np.full(len(Rg), np.nan)
  i0 = int(np.searchsorted(Rg, R_anchor, side='left'))   # first node at/after the anchor
  if i0 >= len(Rg):
    return out
  bad = np.flatnonzero(~np.isfinite(f[i0:]))
  i1 = i0 + (int(bad[0]) if len(bad) else len(Rg) - i0)
  if i1 <= i0:
    return out
  R, fv = Rg[i0:i1], f[i0:i1]
  prepend = R[0] > R_anchor
  if prepend:
    f0 = (np.interp(R_anchor, Rg[i0-1:i1], f[i0-1:i1])
          if (i0 > 0 and np.isfinite(f[i0-1])) else fv[0])
    R, fv = np.concatenate(([R_anchor], R)), np.concatenate(([f0], fv))
  F = np.concatenate(([0.], np.cumsum(0.5*(fv[1:] + fv[:-1])*np.diff(R))))
  out[i0:i1] = s_anchor + (F[1:] if prepend else F)
  return out


def compute_R_rar(cell_d0, exit_row, env, popts, R_fac=50., n_R=2000, fitfunc=smooth_bpl_apy):
  '''
  Radius at which the rarefaction wave catches the cell, after which it stops
  emitting; np.inf if it is never caught within the horizon (falls back to the
  Tobs_max cap). Local per-cell model, spherically correct via the reconstructed
  fits: the head is launched from the shell-exit interface (exit_row: the last-
  shocked/outer-edge cell = actual RS/FS crossing, NOT the planar env.RfRS/RfFS)
  and propagated through THIS cell's hydro at the local sound-speed characteristic
    beta_RF = (beta_fl +/- cs)/(1 +/- beta_fl*cs)   (+ region 3 / z=4, - region 2)
  with beta_fl from lfac(R) and cs = derive_cs(rho(R), p(R)) from the fits (cs drops
  as the gas expands - the spherical effect the planar betaRFp3/betaRFm2 miss).

  R_rar is the intersection of the head worldline with the cell worldline, solved in
  the OBSERVER LAG s(R) = t(R) + t0 - R/c on a log-spaced radius grid, NOT as
  R_head(t) - R_cell(t) on a lab-time grid. Both worldlines are ultrarelativistic and
  sit within ~1e-6 of each other in RADIUS while their lags differ by O(1) s, so the
  lab-time form resolves the crossing to nothing (see compute_shell_rarefaction_head
  for the measured failure) while the lag form is well conditioned:
    ds/dR = (1 - beta)/(c*beta)   [_one_minus_beta_over_beta, no cancellation]
  and the crossing is the root of D(R) = s_head(R) - s_cell(R).
  '''
  popt_rho, popt_lfac, popt_p = popts
  z_fwd = bool(cell_d0.trac < 1.5)                 # region 3 (forward RF) vs region 2
  R0    = cell_d0.x * c_
  lfac0 = get_variable(cell_d0, 'lfac', env)
  rho0  = get_variable(cell_d0, 'rho', env)
  p0    = get_variable(cell_d0, 'p', env)
  s_sh  = cell_d0.t + env.t0 - R0/c_                # cell lag at its shocking event
  s_L   = exit_row.t + env.t0 - exit_row.x          # lag of the launch event (shell exit);
  R_L   = exit_row.x * c_                           # exit_row.x is already R/c

  def lfac(R):
    return lfac0 * fitfunc(np.asarray(R, float)/R0, *popt_lfac)
  def cs_fl(R):
    x = np.asarray(R, float)/R0
    return np.asarray(derive_cs(rho0*fitfunc(x, *popt_rho), p0*fitfunc(x, *popt_p)), float)

  # log-spaced radius grid over the horizon; the lag integrands are smooth in ln R
  # (power-law fits), so this resolves the launch region as well as the far field.
  Rg = np.geomspace(min(R0, R_L), R_fac*R0, n_R)
  with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
    s_c = _lag_profile(Rg, _one_minus_beta_over_beta(lfac(Rg))/c_,   R0,  s_sh)
    s_h = _lag_profile(Rg, _dsdR_head(lfac(Rg), cs_fl(Rg), z_fwd), R_L, s_L)
  # D = 0 on the exit cell (the head is launched on it) and D > 0 on every other cell of
  # the shell, in BOTH regions: the exit cell is shocked last and so carries the largest
  # lag of the shell (that is what makes barT_f the shell-crossing time). It falls to 0
  # as the head reaches the cell -- in region 3 because the head outruns the fluid
  # (+cs), in region 2 because the cells further from the exit decelerate harder and
  # their lag grows faster than the backward head's. The signbit scan below only assumes
  # a sign CHANGE, not which way.
  D = s_h - s_c
  ok = np.isfinite(D) & (Rg >= max(R0, R_L)*(1. - 1e-12))
  if ok.sum() < 2:
    return np.inf
  D, Rv = D[ok], Rg[ok]
  if abs(D[0]) < 1e-9*env.T0:                       # exit cell: the head starts on it
    return max(R0*(1.+1e-6), Rv[0])
  cross = np.flatnonzero(np.diff(np.signbit(D)))    # first head<->cell crossing
  if not len(cross):
    return np.inf
  i = cross[0]
  frac = D[i]/(D[i] - D[i+1]) if (D[i+1] != D[i]) else 0.
  R_rar = Rv[i]*(Rv[i+1]/Rv[i])**frac               # log-linear within the interval
  if not np.isfinite(R_rar) or R_rar <= R0:
    return np.inf
  return max(R_rar, R0*(1.+1e-6))


_RAR_CACHE_VERSION = 4   # v2: caches carry n_shell (coverage); v3: + per-cell barT_off;
                         # v4: crossings solved in the observer lag on a log-radius grid
                         # (the lab-time solver resolved the whole shell in ~4 steps)
_RAR_HEAD_MEM = {}    # (key, z) -> ({cell_i: R_rar/R_inj}, {cell_i: barT_off}), in-process reuse


_HEAD_CTX = {}


def _head_init(key, env, varlist, sh):
  """Per-worker context for the rarefaction head's per-cell fits."""
  _HEAD_CTX.update(key=key, env=env, varlist=varlist, sh=sh)


def _head_chunk(span):
  """
  The per-cell fit block of compute_shell_rarefaction_head, for rows [j0, j1) of the
  shock-front table. Module-level so it is picklable; returns [(j, payload or None)],
  keyed by position so the parent rebuilds the ORIGINAL row order whatever order the
  chunks come back in.

  This is what made the sweep's prologue the longest serial block in the pipeline: at
  hi-res it opens 1e4 cell files (18.4 MB each) and fits three broken power laws to each,
  in the parent, before any pool exists -- measured at 1.5-2.8 h against ~25 min for the
  sweep point it is preparing. The fits are cached per cell ({k:04d}_fit.npz), and each
  cell is touched by exactly one worker, so the writes stay disjoint.
  """
  j0, j1 = span
  c = _HEAD_CTX
  key, env, varlist, sh = c['key'], c['env'], c['varlist'], c['sh']
  out = []
  for j in range(j0, j1):
    row = sh.iloc[j]
    kk = int(row.i)
    # EXISTENCE, not the history: open_celldata costs 3.6 s on a hi-res cell and the fit
    # below answers from {k:04d}_fit.npz whenever that is warm, so the read is deferred
    # behind a callable and never happens on a hit. get_cellfile's flag is exactly what
    # open_celldata returns False on, so a missing cell still yields (j, None).
    if not get_cellfile(key, kk)[1]:
      out.append((j, None))
      continue
    norms = [get_variable(row, nm, env) for nm in varlist]
    try:
      popts = load_or_fit_celldata(lambda kk=kk: open_celldata(key, kk), varlist, norms,
                                   env, row.x, key=key, k=kk, r_max=HEAD_FIT_RMAX)
    except RuntimeError:
      out.append((j, None))
      continue
    out.append((j, dict(i=kk, R0=float(row.x*c_),
                        s0=float(row.t) + env.t0 - float(row.x),
                        lfac0=float(get_variable(row, 'lfac', env)),
                        rho0=float(get_variable(row, 'rho', env)),
                        p0=float(get_variable(row, 'p', env)),
                        popts=popts)))
  return out


def compute_shell_rarefaction_head(key, z, env, R_fac=50., n_R=2000,
    fitfunc=smooth_bpl_apy, nproc=1):
  '''
  Rarefaction catch-up radius R_rar for EVERY cell of shell z, from a SINGLE head
  trajectory integrated once through the assembled shell hydro -- instead of the
  per-cell compute_R_rar that traces a separate head through one cell's own fit
  extrapolated everywhere. Each shell cell contributes its self-similar worldline
  and (beta_k, cs_k) from its cached fit; the head is launched from the shell-exit
  interface and, at each radius, sees the LOCAL cell's fluid
  (beta +/- cs)/(1 +/- beta*cs) (+ region 3 trac<1.5, - region 2). R_rar for a
  cell = first crossing of the head with that cell's worldline. Returns
  ({cell_index: R_rar/R_inj}, {cell_index: barT_off}), where barT_off = (Ton - Ts)/T0 is
  the observer time at which the cell stops emitting.

  EVERYTHING IS SOLVED IN THE OBSERVER LAG, PARAMETRISED BY RADIUS:
    s(R) = t(R) + t0 - R/c        Ton = (1+z)*s,   barT = ((1+z)*s - Ts)/T0
    ds/dR = (1 - beta(R))/(c*beta(R))              [_one_minus_beta_over_beta]
  and NOT as R_head(t) - R_cell(t) on a lab-time grid, which is what this function used
  to do and which does not work at all here. The shocked layer is ultrarelativistic, so
  at the launch time the whole 500-cell shell spans ~0.3 light-seconds out of R_L/c =
  65305 -- a 1e-6 relative spread in RADIUS -- while spanning O(1) s in LAG. Resolving
  the crossings in radius therefore needs a lab-time step ~1e-6 of the horizon; the old
  uniform grid (2000 steps over (R_fac-1)*R_L/(0.3c), a horizon sized by a
  non-relativistic beta >= 0.3 estimate) gave dt = 5336 s while the rarefaction crosses
  the entire shell in 21000 s, i.e. FOUR steps for the whole shell. The 81 cells nearest
  the exit all collapsed onto the launch node (interpolated crossing fraction exactly
  0.0), which handed them R_rar = their own radius at t_L and a barT_off BELOW the
  shell-crossing time barT_f -- impossible, since d(t - R/c)/dt = 1 - beta_head > 0 makes
  every crossing later in observer time than the launch event. In the lag variable that
  ordering holds by construction: s_head only grows from s_L, and the crossing value
  s_k = s_head >= s_L.

  The lag also fixes the local-fluid lookup: at fixed R the cells are ordered by lag
  (small s = outer/front, large s = inner/back), so beta and cs seen by the head are
  interpolated over the cells' lags at the head's own lag, instead of over radii that
  agree to 1e-6.

  NB the radius map is normalised to R_inj = each cell's OWN radius when it was shocked
  (row.x*c_ -- the local variable is called R0 below, do not read it as env.R0, the
  shell normalisation radius). So the values start at 1 by construction and run to ~3
  across the shell: 1 at the outer edge, shocked last with the head already on it, up to
  ~3 at the contact discontinuity, shocked first and radiating to ~3x its injection
  radius. Consumers must rescale by the cell's own radius, as generate_cell_withDistrib
  does: R_rar = ror*(cell_d0.x*c_).

  Both maps are dimensionless, hence alpha/zeta-invariant (Granot rescale leaves R/R_inj,
  beta, cs invariant, and scales t, t0, R, Ts, T0 alike) and reused across the whole
  alpha sweep. They do assume the baseline env: u_scale != 1 changes beta and would need
  a recompute.

  n_R: nodes of the log-spaced radius grid, split between the pre-launch stretch (where
  the cells accumulate the lag separation that orders them) and the post-launch horizon
  up to R_fac*R_L. Converged: 500 -> 8000 moves the R_rar map by <0.1%.
  '''
  sh = cellsBehindShock_fromData(open_rundata(key, z))
  exit_row = sh.loc[sh.t.idxmax()]
  R_L = float(exit_row.x*c_)
  s_L = float(exit_row.t) + env.t0 - float(exit_row.x)   # lag of the launch event
  z_fwd = bool(exit_row.trac < 1.5)   # region 3 (forward RF) vs region 2; one shell,
                                      # one side of the CD, so one sign for all cells
  varlist = ['rho', 'lfac', 'p']
  n_rows = len(sh)
  spans = [(j, min(j + _HEAD_CELLS_PER_CHUNK, n_rows))
           for j in range(0, n_rows, _HEAD_CELLS_PER_CHUNK)]
  found = [None]*n_rows
  if nproc > 1 and len(spans) > 1:
    import cell_pool
    npr = cell_pool.resolve_nproc(nproc, cap=len(spans))
    print(f'rarefaction head: fitting {n_rows} cells on {npr} workers')
    with cell_pool.cell_executor(npr, initializer=_head_init,
                                 initargs=(key, env, varlist, sh)) as ex:
      for out in ex.map(_head_chunk, spans):
        for j, payload in out:
          found[j] = payload
  else:
    _head_init(key, env, varlist, sh)
    for span in spans:
      for j, payload in _head_chunk(span):
        found[j] = payload
  # drop the cells with no history / no usable fit, keeping the front table's own order
  cells = [pl for pl in found if pl is not None]
  if not cells:
    return {}, {}

  # radius grid, log-spaced and split at R_L so the launch radius is EXACTLY a node
  # (the head starts there and every cell lag is compared against s_L at that node).
  R_min = min(ci['R0'] for ci in cells)
  lo = np.log(max(R_L/R_min, 1. + 1e-12)), np.log(R_fac)
  n_lo = int(np.clip(round(n_R*lo[0]/(lo[0] + lo[1])), 50, n_R - 50))
  Rg = np.concatenate((np.geomspace(R_min, R_L, n_lo, endpoint=False),
                       np.geomspace(R_L, R_fac*R_L, n_R - n_lo)))
  i_L = n_lo                                  # index of R_L in Rg

  # per-cell Lorentz factor, sound speed and lag on the grid. Gamma_k(R) and cs_k(R) are
  # ANALYTIC in the fits, so no per-cell ODE is needed: the worldline is the quadrature
  # s_k(R) = s_k(R0_k) + int (1-beta)/(c beta) dR (_lag_profile).
  nc = len(cells)
  Gk = np.full((nc, len(Rg)), np.nan)
  Ck = np.full((nc, len(Rg)), np.nan)
  Sk = np.full((nc, len(Rg)), np.nan)
  with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
    for m, ci in enumerate(cells):
      popt_rho, popt_lfac, popt_p = ci['popts']
      x = Rg/ci['R0']
      G = ci['lfac0']*fitfunc(x, *popt_lfac)
      # _lag_profile already runs forward from R0 and stops where the fit goes bad
      s = _lag_profile(Rg, _one_minus_beta_over_beta(G)/c_, ci['R0'], ci['s0'])
      live = np.isfinite(s) & np.isfinite(G) & (G > 1.)
      Gk[m, live] = G[live]
      Ck[m, live] = np.asarray(derive_cs(ci['rho0']*fitfunc(x[live], *popt_rho),
                                         ci['p0']*fitfunc(x[live], *popt_p)), float)
      Sk[m, live] = s[live]

  # march ONE head from (R_L, s_L) outwards, Heun on the log-spaced grid. At each node
  # the local fluid is read at the head's LAG: sort the live cells by s_k(R) and
  # interpolate Gamma, cs there (np.interp clamps once the head leaves the shell, i.e.
  # it then keeps coasting with the edge cell's state -- by which point every cell has
  # already been crossed).
  def dsdR(i, s):
    live = np.isfinite(Sk[:, i])
    if not live.any():
      return np.nan
    o = np.argsort(Sk[live, i])
    sv, gv, cv = Sk[live, i][o], Gk[live, i][o], Ck[live, i][o]
    return float(_dsdR_head(np.interp(s, sv, gv), np.interp(s, sv, cv), z_fwd))

  Sh = np.full(len(Rg), np.nan)
  Sh[i_L] = s_L
  for i in range(i_L, len(Rg) - 1):
    dR = Rg[i+1] - Rg[i]
    f1 = dsdR(i, Sh[i])
    if not np.isfinite(f1):
      break
    f2 = dsdR(i+1, Sh[i] + f1*dR)
    Sh[i+1] = Sh[i] + 0.5*(f1 + (f2 if np.isfinite(f2) else f1))*dR

  # per-cell R_rar/R_inj (R0 here = this cell's own shocked radius, NOT env.R0) = first
  # crossing of the head with that cell's worldline, i.e. the root of D = s_head - s_cell,
  # plus the observer time barT_off = (Ton - Ts)/T0 there = when the cell stops emitting.
  # Same Ton convention as the reach_window event in worldline_from_cooling and as
  # get_variable(.., 'Ton', ..): Ton = (1+z)*(t_sim + t0 - R/c) = (1+z)*s.
  # D starts at 0 on the exit cell (the head is launched on it) and at > 0 on all the
  # others, in both regions -- see compute_R_rar for why.
  def _barT_off(s):
    return ((1. + env.z)*s - env.Ts)/env.T0
  rrar, boff = {}, {}
  for m, ci in enumerate(cells):
    R0 = ci['R0']
    D = Sh - Sk[m]
    ok = np.isfinite(D)
    ok[:i_L] = False                                      # head does not exist before R_L
    if ok.sum() < 2:
      rrar[ci['i']] = np.inf; boff[ci['i']] = np.inf; continue
    Dv, Rv, sv = D[ok], Rg[ok], Sk[m][ok]
    if abs(Dv[0]) < 1e-9*env.T0:                          # exit cell: head starts on it
      rrar[ci['i']] = max(R0*(1.+1e-6), Rv[0])/R0
      boff[ci['i']] = _barT_off(sv[0]); continue
    cross = np.flatnonzero(np.diff(np.signbit(Dv)))
    if not len(cross):
      rrar[ci['i']] = np.inf; boff[ci['i']] = np.inf; continue
    a = cross[0]
    frac = Dv[a]/(Dv[a] - Dv[a+1]) if Dv[a+1] != Dv[a] else 0.
    R_rar = Rv[a]*(Rv[a+1]/Rv[a])**frac                   # log-linear within the interval
    s_rar = sv[a] + frac*(sv[a+1] - sv[a])
    good = np.isfinite(R_rar) and R_rar > R0
    rrar[ci['i']] = max(R_rar, R0*(1.+1e-6))/R0 if good else np.inf
    boff[ci['i']] = _barT_off(s_rar) if good else np.inf
  _patch_isolated_gaps(rrar, boff, key, z)
  _check_rarefaction_maps(rrar, boff, _barT_off(s_L), key, z)
  return rrar, boff


def _check_rarefaction_maps(rrar, boff, barT_f, key, z, tol=1e-6, frac_tol=0.05):
  '''
  Guard on the invariant the old lab-time solver silently violated: the rarefaction is
  launched at the shell-exit event, and the observer lag grows monotonically along the
  head worldline (d(t - R/c)/dt = 1 - beta_head > 0), so NO cell can be cut off before
  the shell-crossing time barT_f -- the earliest cut-off is the exit cell itself, at
  exactly barT_f. Warns (does not raise) if that fails.

  Also checks that barT_off runs monotonically across the shell, but only flags
  reversals larger than frac_tol of the shell's barT_off span: the head is traced
  through 500 INDEPENDENTLY fitted cell profiles, whose cell-to-cell scatter puts a few
  ~1% wiggles in the sequence (6/499 steps, worst 2%, on cooling_fid_raref z=4) that are
  fit noise, not a solver failure. An under-resolved grid reverses whole blocks of cells
  instead (the lab-time solver ramped 81 cells DOWN by 14% of the span), which this
  catches.
  '''
  b = np.array([v for v in boff.values()], float)
  b = b[np.isfinite(b)]
  if not b.size:
    return
  if b.min() < barT_f - tol:
    print(f'compute_shell_rarefaction_head: WARNING ({key}, z={z}) min(barT_off)='
          f'{b.min():.6f} < barT_f={barT_f:.6f} by {barT_f - b.min():.2e} -- the head '
          f'crossings are under-resolved (raise n_R)')
  ks = np.array(sorted(rrar.keys()))
  seq = np.array([boff[k] for k in ks], float)
  seq = seq[np.isfinite(seq)]
  span = seq.max() - seq.min()
  if len(seq) > 2 and span > 0.:
    d = np.diff(seq)*(1. if np.nanmean(np.diff(seq)) > 0 else -1.)
    worst = -d.min()/span
    if worst > frac_tol:
      print(f'compute_shell_rarefaction_head: WARNING ({key}, z={z}) barT_off reverses '
            f'by {worst:.1%} of its span across the shell ({(d < 0).sum()}/{len(d)} '
            f'steps) -- the head crossings are under-resolved (raise n_R)')


HEAD_FIT_RMAX = 10.          # radius cap (x/x0) on the per-cell fits the rarefaction head
                             # is traced through. The head only ever needs each cell's
                             # worldline out to where it is crossed, and that is R/R_inj =
                             # 1.000-3.905 across the fiducial shell -- so 10 is ~2.5x
                             # margin on the physics while cutting a hi-res cell's fit
                             # window from R/R_inj ~ 700 to 10. Most of a long run's cell
                             # data is POST-crash, which is not what the BPL describes;
                             # see fit_celldata's r_max.


_HEAD_CELLS_PER_CHUNK = 32   # cells per task in the head's fit pass: the payload is a
                             # handful of fit parameters, so the only thing to balance is
                             # dispatch overhead against the ~0.1-0.3 s per cell


def fit_cache_stamp(key, cells):
  """
  Fingerprint of the per-cell hydro fits a rarefaction head is built from: the size and
  mtime of each {k:04d}_fit.npz, hashed. Cheap -- stat only, no reads.

  THE HEAD IS A FUNCTION OF THOSE FITS AND ITS VERSION NUMBER DOES NOT SAY SO. The
  fiducial heads on disk were dated 9 August while the fits under them had been refitted
  on 7 September, and because _RAR_CACHE_VERSION had not moved, load_shell_rarefaction
  served the stale head to every sweep: rebuilding shifted R_rar/R_inj on essentially
  every cell (by up to 0.3%). This is the same silent-staleness trap the derived tables
  carry a sweep_stamp against.

  Copying a run between machines rewrites mtimes unless cp/scp preserves them, so the
  stamp can miss and force a rebuild that was not needed. That is the safe direction, and
  a rebuild is cheap once the fits are warm (the head defers the history read entirely).
  """
  import hashlib
  h = hashlib.sha1()
  for k in sorted(int(c) for c in cells):
    f = get_dirpath(key) + f'cells/{k:04d}_fit.npz'
    try:
      st = os.stat(f)
      h.update(f'{k}:{st.st_size}:{st.st_mtime_ns};'.encode())
    except OSError:
      h.update(f'{k}:missing;'.encode())
  return h.hexdigest()


def _patch_isolated_gaps(rrar, boff, key, z):
  """
  Repair an ISOLATED cell with no head crossing, in place, from its two neighbours.

  R_rar = inf means the head never caught the cell inside the horizon, which is a real
  outcome for a cell near the end of the shell -- but not for one sitting between two
  neighbours that are both caught, since R_rar varies smoothly along the shell. There the
  cause is upstream: the head is propagated on the cells' FITTED worldlines, and a fit that
  railed against its bounds gives a cell a trajectory the head cannot meet.

  Measured on cooling_g100 z=4 cell k=45, the only railed fit in 1000 cells (popt_rho,
  popt_lfac and popt_p all pinned at -28., 0.05): its D = s_head - s_cell ends at +0.746
  where k=44 and k=46 reach -53.98 and -1523.72, so it alone returned inf where its
  neighbours give 1.0520 and 1.0580. Left alone that cell emits with NO rarefaction cut.

  ONLY isolated gaps are filled: a run of consecutive non-finite cells is left as it is,
  because that is what "the head never reaches this part of the shell" looks like and it
  is not this function's business to invent one. Never silent -- every patched cell is
  printed, with a pointer to the fit that caused it.
  """
  ks = sorted(rrar)
  patched = []
  for a, b, c in zip(ks, ks[1:], ks[2:]):
    if np.isfinite(rrar[b]):
      continue
    if not (np.isfinite(rrar[a]) and np.isfinite(rrar[c])):
      continue                                  # part of a run, not an isolated gap
    rrar[b] = 0.5*(rrar[a] + rrar[c])
    if np.isfinite(boff[a]) and np.isfinite(boff[c]):
      boff[b] = 0.5*(boff[a] + boff[c])
    patched.append(b)
  if patched:
    print(f'compute_shell_rarefaction_head ({key}, z={z}): {len(patched)} isolated '
          f'cell(s) with no head crossing, filled from their neighbours: {patched}. '
          'Check those cells\' hydro fits -- a railed fit is the usual cause.')
  return patched


def _load_shell_rarefaction_maps(key, z, env, R_fac=50., n_shell=None, nproc=1):
  '''(key, z)-cached rarefaction maps ({cell_index: R_rar/R_inj}, {cell_index: barT_off})
  from the single shared head (compute_shell_rarefaction_head -- see it for what R_inj is
  and why the radii are per-cell ratios). Both dimensionless and alpha/zeta-invariant ->
  built once, reused across the whole sweep; memoized in-process and on disk
  (results/{key}/rarefaction_head_{z}.npz).
  n_shell: number of cells the shell actually has. compute_shell_rarefaction_head only
  sees the cells already extracted to disk, so a head built before extraction covers a
  subset of the shell -- and a cell missing from the map emits with NO rarefaction cut-off.
  Passing n_shell stores the coverage in the cache and rejects (rebuilds) any cached map
  that covers fewer cells than the shell has.'''
  memo = _RAR_HEAD_MEM.get((key, z))
  if memo is not None:
    return memo
  path = get_dirpath(key) + f'rarefaction_head_{z}.npz'
  if os.path.isfile(path):
    try:
      d = np.load(path)
      covered = (n_shell is None) or (len(d['cell_i']) >= int(n_shell))
      stamp = str(d['fit_stamp']) if 'fit_stamp' in d.files else ''
      fresh = (stamp == fit_cache_stamp(key, d['cell_i']))
      if not fresh:
        print(f'load_shell_rarefaction: cached head for ({key}, {z}) was built from '
              'different per-cell fits -- rebuilding')
      if int(d['version']) == _RAR_CACHE_VERSION and np.isclose(float(d['R_fac']), R_fac) \
          and covered and fresh:
        memo = ({int(i): float(r) for i, r in zip(d['cell_i'], d['rrar_over_R0'])},
                {int(i): float(b) for i, b in zip(d['cell_i'], d['barT_off'])})
        _RAR_HEAD_MEM[(key, z)] = memo
        return memo
      if not covered:
        print(f'load_shell_rarefaction: cached head for ({key}, {z}) covers '
              f'{len(d["cell_i"])}/{int(n_shell)} shell cells -- rebuilding')
    except Exception:
      pass
  rrar, boff = compute_shell_rarefaction_head(key, z, env, R_fac=R_fac, nproc=nproc)
  if n_shell is not None and len(rrar) < int(n_shell):
    print(f'load_shell_rarefaction: WARNING head for ({key}, {z}) built from '
          f'{len(rrar)}/{int(n_shell)} shell cells (missing cell data); the rest will '
          f'get an interpolated R_rar')
  try:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ci = np.array(sorted(rrar.keys()), dtype=int)
    np.savez(path, cell_i=ci, rrar_over_R0=np.array([rrar[i] for i in ci], float),
             barT_off=np.array([boff[i] for i in ci], float),
             R_fac=float(R_fac), version=_RAR_CACHE_VERSION,
             n_shell=int(n_shell) if n_shell is not None else len(ci),
             fit_stamp=fit_cache_stamp(key, ci))
  except Exception:
    pass
  _RAR_HEAD_MEM[(key, z)] = (rrar, boff)
  return rrar, boff


def load_shell_rarefaction(key, z, env, R_fac=50., n_shell=None, nproc=1):
  '''{cell_index: R_rar/R_inj} for shell z -- the radius at which the rarefaction wave
  catches each cell and its emission stops, as a ratio to THAT cell's own radius when it
  was shocked (not to env.R0); scale by the cell's radius to use it, as
  generate_cell_withDistrib does. Runs 1 (outer edge) to ~3 (contact discontinuity).
  See compute_shell_rarefaction_head / _load_shell_rarefaction_maps.'''
  return _load_shell_rarefaction_maps(key, z, env, R_fac=R_fac, n_shell=n_shell,
                                      nproc=nproc)[0]


def load_shell_rarefaction_offT(key, z, env, R_fac=50., n_shell=None):
  '''{cell_index: barT_off} for shell z, barT_off = (Ton - Ts)/T0 the observer time at
  which the rarefaction cuts that cell's emission off. max over cells = the observer time
  past which the shell emits nothing at all and the lightcurve is pure high-latitude decay.
  See _load_shell_rarefaction_maps.'''
  return _load_shell_rarefaction_maps(key, z, env, R_fac=R_fac, n_shell=n_shell)[1]


def rar_map_lookup(rar_map, i):
  '''R_rar/R_inj for cell index i from the shell rarefaction map. Cells absent from the map
  (head built before they were extracted) are LINEARLY INTERPOLATED over the map's own
  indices -- R_rar/R_inj is smooth and monotonic across the shell (1 -> ~3), so this is far
  closer to the truth than the alternative of no cut-off at all, which lets the cell
  radiate forever and inflates its slow-cooling energy budget by tens of percent.
  A non-integer i (sub-cells carry an index interpolated between their parent's and the
  next cell's) is TRUNCATED to int first, so a sub-cell takes its parent's value rather
  than one interpolated between the two. Harmless where sub-cells are actually created --
  the CD-adjacent cells, over which the map steps by <0.3% per cell (median 0.24%, 90th
  percentile 0.27% across the shell; the largest single step is 1.6% near k=40, where the
  head is still sweeping the exit cells fastest). The old lab-time solver put a spurious
  7% jump at k=100, the boundary of its under-resolved block -- see
  compute_shell_rarefaction_head.'''
  i = int(i)
  if i in rar_map:
    return rar_map[i]
  ks = np.array(sorted(rar_map.keys()), dtype=float)
  if not len(ks):
    return np.inf
  vs = np.array([rar_map[int(k)] for k in ks], dtype=float)
  ok = np.isfinite(vs)
  if ok.sum() < 2:
    return float(vs[ok][0]) if ok.any() else np.inf
  return float(np.interp(float(i), ks[ok], vs[ok]))


def evolve_gma_bounds_edges(tt_edges, rho_edges, gmin0, gmax0):
  '''
  Core of evolve_gma_bounds, operating on precomputed arrays: per step j -> j+1,
  operator-split synchrotron (with cooling fluence dtt_j) then adiabatic
  correction (rho_{j+1}/rho_j)^(1/3). Source of rho_edges is irrelevant (fits or
  actual hydro data), which is what makes this shared between the fit-based and
  data-driven pipelines.
  Returns gmin_edges, gmax_edges, bsyn_edges, Aad_edges (see evolve_gma_bounds
  docstring).
  '''
  dtt = np.diff(tt_edges)
  adiab = (rho_edges[1:] / rho_edges[:-1])**(1./3.)   # (V'_{j+1}/V'_j)^(-1/3) per step

  def evolve(g0):
    g = np.empty(len(rho_edges))
    bs = np.empty(len(rho_edges))
    g[0], bs[0] = g0, 1.
    for j in range(len(dtt)):
      f_syn  = 1. / (1. + g[j] * dtt[j])    # synchrotron burn-off factor of the step
      bs[j+1] = bs[j] * f_syn               # cumulative, synchrotron only
      g[j+1] = g[j] * f_syn * adiab[j]      # actual trajectory: syn, then adiabatic
    return g, bs
  gmin_edges, _ = evolve(gmin0)
  gmax_edges, bsyn_edges = evolve(gmax0)
  # cumulative adiabatic factor A = prod(adiab) = (rho/rho_0)^(1/3); it telescopes,
  # but is built from the SAME adiab as the recursion so the two cannot drift apart.
  Aad_edges = np.concatenate(([1.], np.cumprod(adiab)))
  return gmin_edges, gmax_edges, bsyn_edges, Aad_edges

def evolve_gma_bounds(R_edges, tt_edges, cell_d0, env, popt_rho, R0=None, fitfunc=smooth_bpl_apy):
  '''
  Evolve the distribution bounds (gmin, gmax) over the cooling steps defined by the
  worldline edges (R_edges, tp_edges). Valid when adiabatic cooling is weak compared
  to synchrotron: per step j -> j+1, operator-split as
    1. synchrotron with the cooling time dtt_j:
         1/g_syn = 1/g_j + dtt_j
    2. adiabatic correction from the comoving expansion:
         g_{j+1} = g_syn * (V'_{j+1}/V'_j)^(-1/3) = g_syn * (rho_{j+1}/rho_j)^(1/3)
  (code rho is the comoving density, so V' ~ 1/rho along the worldline.)
  rho evaluated from the fits (popt_rho) normalized to cell_d0.
  R0 is the fit anchor radius (= cell_d0.x*c_); must match worldline_from_cooling
  and reconstruct_cell so the fit is sampled consistently.
  Also returns bsyn_edges: the cumulative SYNCHROTRON-only burn-off factor of the
  top edge, accumulated along the actual (syn+adiabatic) trajectory. Only the
  synchrotron part distorts the power-law shape -- adiabatic cooling is a uniform
  rescaling that preserves it -- so bsyn, not gmax/gmax0, is what sets the cooled
  distribution's cutoff (see cooled_tt_eff in radiation_cooling).
  Also returns Aad_edges, the cumulative adiabatic factor A = (rho/rho_0)^(1/3).
  A rescales the distribution's NORMALISATION, K = K0 * A^(p-1): the cooled shape
  conserves number under synchrotron alone, but adiabatic cooling compresses the
  gamma axis, so the shape at fixed K integrates to A^(1-p) instead of 1. The
  emission path applies A^(p-1) per step (get_epnu, step_radiated_energy).
  Returns gmin_edges, gmax_edges, bsyn_edges, Aad_edges.
  '''

  if R0 is None:
    R0 = cell_d0.x * c_
  rho0 = get_variable(cell_d0, 'rho', env)

  x   = R_edges / R0
  rho = rho0 * fitfunc(x, *popt_rho)

  gmin0 = get_variable(cell_d0, 'gma_m', env)
  gmax0 = get_variable(cell_d0, 'gma_M', env)
  return evolve_gma_bounds_edges(tt_edges, rho, gmin0, gmax0)

def generate_timebins(cell_d0, env, popt_lfac, end_val, end_cond='tt',
    r_ref=1.2, Nmin=2, Nmax=None, func_cooling=gamma_synCooled,
    fitfunc=smooth_bpl_apy):
  '''
  Generate normalized time array binned w.r.t cooling: gamma_max_j+1 = gamma_max_j/r_ref
  end_cond = 'tt' or 'gmax'
    tt: comoving time normalized to initial cooling time
    gmax: gamma_max
  popt_lfac: fitted Gamma(R/R0) profile, used to convert comoving -> lab
    time with a variable Lorentz factor (consistent with reconstruct_cell)
  '''

  lfac0 = get_variable(cell_d0, 'lfac', env)
  gmax0 = get_variable(cell_d0, 'gma_M', env)
  R0 = cell_d0.x * c_
  t0, tp0, tt0 = cell_d0.t, 0., 0.

  if end_cond == 'tt':
    gmax_end = func_cooling(end_val, gmax0)
  elif end_cond == 'gmax':
    gmax_end = end_val
  else:
    print("'end_cond' must be 'tt' or 'gmax'")
    return 0.
  ratio = gmax0/gmax_end
  N = int(np.log10(ratio)/np.log10(r_ref))
  N = max(N, Nmin)
  if Nmax is not None:
    N = min(N, Nmax)

  # bin edges in expected gmax, then the corresponding time edges (length N+1)
  gmax_bins = np.geomspace(gmax0, gmax_end, N+1, endpoint=True)
  tt_edges = gmax_bins**-1 - gmax0**-1       # tt_edges[0] = 0
  # we return tt_edges to have consistent arrays
  return tt_edges

def rescale_shocked_data(data, env_in, env):
  '''
  Rescale shocked-cells data (a single row/Series or a full shock-front
  dataframe) from env_in to env = rescale_proper_velocities(s, env_in).
  rho, p, dx are invariants (code units), but env.rhoscale ~ s^-6.
  t, x ~ s^2 only in the ultra-relativistic limit; the observer onset
  Ton - Ts = (1+z)[t - (x - R0/c)] relies on a cancellation at the
  R/2cGamma^2 level, far below the relative accuracy of s^2 at low u1.
  So use exact env ratios: radii keep R/R0, and t is mapped through the
  cell's effective speed beta_eff = dr/t, scaling its proper velocity.
  Fluid proper velocity (vx, lfac) scales by the shocked-fluid ratio
  env.u/env_in.u, the upstream vx_u by the exact shell ratio (= s).
  '''
  out = data.copy()
  r_ratio = env.R0 / env_in.R0
  u_ratio = env.u / env_in.u
  s = env.u1 / env_in.u1

  t = np.atleast_1d(np.asarray(data['t'], dtype=float))
  x = np.atleast_1d(np.asarray(data['x'], dtype=float))
  dr = x - env_in.R0/c_
  dr_new = dr * r_ratio
  x_new = env.R0/c_ + dr_new
  beta_eff = np.divide(dr, t, out=np.zeros_like(dr), where=(t > 0.))
  valid = (beta_eff > 0.) & (beta_eff < 1.)
  b = np.where(valid, beta_eff, 0.5)
  ueff = u_ratio * b/np.sqrt(1. - b**2)
  t_new = np.where(valid, dr_new*np.sqrt(1. + ueff**2)/ueff,
                   t * r_ratio)   # degenerate: shocked at collision (t ~ 0)

  def boost_proper(vx, fac):
    vx = np.atleast_1d(np.asarray(vx, dtype=float))
    u_new = fac * vx/np.sqrt(1. - vx**2)
    lfac_new = np.sqrt(1. + u_new**2)
    return u_new/lfac_new, lfac_new

  vx_new, lfac_new = boost_proper(data['vx'], u_ratio)
  news = {'t':t_new, 'x':x_new, 'vx':vx_new, 'lfac':lfac_new}
  if 'vx_u' in data.keys():
    news['vx_u'] = boost_proper(data['vx_u'], s)[0]
  is_series = isinstance(data, pd.Series)
  for name, val in news.items():
    out[name] = val.item() if is_series else val
  return out

def rescale_hydro_data(data, alpha):
  '''
  Granot length-time rescale of code-unit hydro data: t, x, dx x alpha
  (rho, p, vx invariant). Handles a Series (cell_d0) or a DataFrame.
  '''
  out = data.copy()
  for col in ('t', 'x', 'dx'):
    if col in data:
      out[col] = data[col] * alpha
  return out

def generate_cell_withDistrib(cell_data, cell_init, env_in,
    u_scale=1., alpha=1., zeta=1., cleanData=False, r_ref=1.2, Tmax=None,
    key=None, k=None, exit_row=None, dlnrho_max=DLNRHO_MAX, popts=None, rar_map=None):
  '''
  Reconstruct cell hydrodynamics from fit, adds electron distribution
  u_scale rescales (after fitting) as if simulation was run with u1 x u_scale
  alpha, zeta: Granot (2012) hydro unit-rescaling (lengths/times x alpha,
    energy/mass x zeta => rho, p x zeta*alpha^-3); composes on top of u_scale
  popts: precomputed (popt_rho, popt_lfac, popt_p) hydro fit; when given the fit
    is skipped and cell_data is unused (for interpolated sub-cells that reuse a
    neighbour's self-similar profile, see get_shell_nuFnu subcell refinement)
  r_ref: cooling-bin ratio (gmax_j/gmax_j+1 per step); larger = fewer steps
  dlnrho_max: cap on ln(rho) per cooling step, refining the tt grid so the
    adiabatic evolution stays resolved in slow/very-slow cooling (see
    worldline_from_cooling / _refine_tt_on_rho); None keeps the raw tt bins
  Tmax: if set, caps the worldline at the observer window Tobs_max = Ts + Tmax*T0
    (obs_arrays uses T in [1, Tmax+1]); avoids the tt->R runaway to unphysical
    radii. None keeps the full (uncapped) worldline. See worldline_from_cooling.
  key, k: if given, the (expensive) hydro fit is disk-cached per cell; see
    load_or_fit_celldata. None fits every call (no cache).
  exit_row: the shell-exit interface (last-shocked / outer-edge cell of the shell,
    from sh_data). If given, the worldline is capped at the rarefaction catch-up
    radius R_rar = compute_R_rar(...), past which the cell no longer emits. None
    disables the rarefaction cap (worldline bounded by Tobs_max only).
  '''

  # add copy and rescaling to env
  cell_d0 = cell_init.copy()
  vars = ['rho', 'lfac', 'p']
  norms = [get_variable(cell_d0, name, env_in) for name in vars]

  # fit hydrodynamics, anchored at the cell_d0 (downstream) state so the
  # reconstruction starts exactly at cell_d0 (disk-cached per cell when key/k set).
  # A precomputed popts (e.g. from a parent cell) skips the fit entirely.
  if popts is None:
    try:
      popts = load_or_fit_celldata(cell_data, vars, norms, env_in, cell_d0.x,
                                   cleanData=cleanData, key=key, k=k)
    except RuntimeError:
      print('Fit failed on cell ', cell_d0.i)
      return False, env_in
  popt_rho, popt_lfac, popt_p = popts

  # rescale cell_d0 and env (s = u_scale, holding a_u, f, chi fixed),
  # see rescale_shocked_data for the conventions. The shell-exit interface (exit_row)
  # is rescaled the same way so R_rar is computed in the reconstruction frame.
  env = rescale_proper_velocities(u_scale, env_in)
  if u_scale != 1.:
    cell_d0 = rescale_shocked_data(cell_d0, env_in, env)
    if exit_row is not None:
      exit_row = rescale_shocked_data(exit_row, env_in, env)

  # Granot hydro unit-rescaling on top: env carries rho,p x zeta*alpha^-3
  # (via rhoscale) and lengths/times x alpha; the anchor carries t,x,dx x alpha
  if alpha != 1. or zeta != 1.:
    env = rescale_hydro(alpha, zeta, env)
    if alpha != 1.:
      cell_d0 = rescale_hydro_data(cell_d0, alpha)
      if exit_row is not None:
        exit_row = rescale_hydro_data(exit_row, alpha)

  # rarefaction catch-up radius: past R_rar the rarefaction wave has caught the cell
  # and it stops emitting. Replaces the old rising-Gamma rejection - a cell whose
  # extrapolated fit would diverge is instead physically truncated at R_rar (which is
  # a modest radius), so it contributes finite flux instead of being dropped.
  # rar_map (from the single shared shell head, load_shell_rarefaction) gives the
  # dimensionless R_rar/R_inj per cell, normalised to the cell's OWN shocked radius (not
  # env.R0) -> scale by this cell's (rescaled) injection radius; falls back
  # to the per-cell compute_R_rar when no map is supplied. A cell absent from the map is
  # interpolated over the map's indices (rar_map_lookup), never left uncapped.
  if rar_map is not None:
    ror = rar_map_lookup(rar_map, cell_d0.i)
    R_rar = ror * (cell_d0.x*c_) if np.isfinite(ror) else np.inf
  elif exit_row is not None:
    R_rar = compute_R_rar(cell_d0, exit_row, env, popts)
  else:
    R_rar = None

  # generate time bins
  # the comoving times tp and tt start at 0 when cell is shocked
  tt_edges = generate_timebins(cell_d0, env, popt_lfac, 1., end_cond='gmax', r_ref=r_ref)

  # compute worldline, starting from the cell_d0 radius and lab time. Capped at the
  # observer window (Tobs_max) so tt->R does not run away to unphysical radii, and at
  # the rarefaction catch-up radius (R_rar); tt_edges comes back truncated to the
  # integrated range (whichever cap is reached first).
  R0 = cell_d0.x * c_
  t0 = cell_d0.t
  Tobs_max = (env.Ts + Tmax*env.T0) if Tmax is not None else None
  t_edges, tp_edges, R_edges, tt_edges = worldline_from_cooling(
      tt_edges, cell_d0, env, popts, R0=R0, Tobs_max=Tobs_max, R_rar=R_rar,
      dlnrho_max=dlnrho_max)
  t_arr = t_edges[:-1] + t0
  dt_arr = np.diff(t_edges)
  tp_arr = tp_edges[:-1]
  dtp_arr = np.diff(tp_edges)
  tt_arr = tt_edges[:-1]
  dtt_arr = np.diff(tt_edges)
  R_arr = R_edges[:-1]

  # reconstruct hydro
  i, x, dx, rho, vx, lfac, p, trac = reconstruct_cell(R_arr, env, cell_d0, popts)
  gmin_edges, gmax_edges, bsyn_edges, Aad_edges = evolve_gma_bounds(
      R_edges, tt_edges, cell_d0, env, popt_rho, R0=R0)
  gmin, gmax, bsyn = gmin_edges[:-1], gmax_edges[:-1], bsyn_edges[:-1]
  # left edge, to stay consistent with rho/Pmax (which are also left-edge): the pair
  # (n(rho), A^(p-1)) is what makes the step's electron count physical.
  Aad = Aad_edges[:-1]

  # create dataframe
  keys = ['t', 'dt', 'tp', 'dtp', 'tt', 'dtt', 'i', 'x', 'dx', 'rho', 'vx', 'lfac', 'p', 'trac', 'gmin', 'gmax', 'bsyn', 'Aad']
  vals = [t_arr, dt_arr, tp_arr, dtp_arr, tt_arr, dtt_arr, i, x, dx, rho, vx, lfac, p, trac, gmin, gmax, bsyn, Aad]
  # carry the (constant) upstream velocity so thin-shell functions (get_Fnu_vFC ->
  # nu_m2 via derive_nu_m_new) work on the reconstructed cell
  if 'vx_u' in cell_d0:
    keys.append('vx_u'); vals.append(np.full(x.shape, cell_d0.vx_u))
  dic = {key:val for key, val in zip(keys, vals)}
  out = pd.DataFrame.from_dict(dic)
  attrs = cell_d0.attrs
  if len(attrs) > 0:
    for key in attrs.keys():
      out.attrs[key] = attrs[key]
  return out, env

def generate_cell_constLfac(cell, env):
  '''
  Creates cell with analytical scalings (rho ~ R^-2, Gamma cst, B ~ R^-1)
  starting from init conditions of 'cell', output from generate_cell_withDistrib
  '''
  cell0 = cell.iloc[0]
  R0    = cell0.x * c_
  lfac0 = cell0.lfac
  rho0  = cell0.rho
  p0    = cell0.p
  vx0   = cell0.vx
  dx0   = cell0.dx
  u0   = np.sqrt(lfac0**2 - 1.)               # beta*Gamma
  syn0  = get_variable(cell0, 'syn', env)      # 1/t_c1,0 (~B0^2), from the base hydro
  gmin0 = cell0.gmin
  gmax0 = cell0.gmax

  # cooling-time bin edges, identical to the evolving cell (tt_arr=edges[:-1], dtt=diff)
  tt_arr0 = cell.tt.to_numpy()
  dtt0    = cell.dtt.to_numpy()
  tt_edges = np.append(tt_arr0, tt_arr0[-1] + dtt0[-1])

  # analytic worldline + tt -> R inversion
  tt_sat = syn0 * R0 / (c_ * u0)
  if tt_edges[-1] >= tt_sat:
    print(f"generate_cell_constLfac: B~R^-1 cooling saturates at tt={tt_sat:.3g} < "
          f"target tt={tt_edges[-1]:.3g} (slow cooling); clipping edges, gmax stays > 1.")
    tt_edges = tt_edges[tt_edges < tt_sat]
  R_edges  = R0 / (1. - tt_edges/tt_sat)
  tp_edges = (R_edges - R0)/(c_*u0)           # proper time (dR/dt' = c u0)
  t_edges  = (R_edges - R0)/(c_*vx0)           # elapsed lab time (dR/dt = c beta)

  # bin starts + widths
  t_arr  = t_edges[:-1] + cell0.t;    dt_arr  = np.diff(t_edges)
  tp_arr = tp_edges[:-1];             dtp_arr = np.diff(tp_edges)
  tt_arr = tt_edges[:-1];             dtt_arr = np.diff(tt_edges)
  R_arr  = R_edges[:-1]
  x_arr  = R_arr/c_

  # analytic hydro: constant lfac, rho & p ~ R^-2  => B ~ R^-1
  xr   = R_arr/R0
  rho  = rho0 * xr**-2
  p    = p0   * xr**-2
  lfac = np.full(R_arr.shape, lfac0)
  vx   = np.full(R_arr.shape, vx0)
  dx   = np.full(R_arr.shape, dx0)             # mass conservation => dx = const here
  i    = np.full(R_arr.shape, cell0.i)
  trac = np.full(R_arr.shape, cell0.trac)

  # distribution bounds: synchrotron over dtt (B folded into tt) then adiabatic
  rho_edges = rho0 * (R_edges/R0)**-2
  adiab = (rho_edges[1:]/rho_edges[:-1])**(1./3.)   # = (R_j/R_{j+1})^(2/3)
  def evolve(g0):
    g = np.empty(len(R_edges)); g[0] = g0
    for j in range(len(dtt_arr)):
      g_syn  = g[j] / (1. + g[j]*dtt_arr[j])
      g[j+1] = g_syn * adiab[j]
    return g
  gmin, gmax = evolve(gmin0)[:-1], evolve(gmax0)[:-1]

  keys = ['t', 'dt', 'tp', 'dtp', 'tt', 'dtt', 'i', 'x', 'dx', 'rho', 'vx', 'lfac', 'p', 'trac', 'gmin', 'gmax']
  vals = [t_arr, dt_arr, tp_arr, dtp_arr, tt_arr, dtt_arr, i, x_arr, dx, rho, vx, lfac, p, trac, gmin, gmax]
  out = pd.DataFrame.from_dict({k:v for k, v in zip(keys, vals)})
  for key in cell.attrs:
    out.attrs[key] = cell.attrs[key]
  return out

def get_Fnu_array_cell_evolving(nuobs, Tobs, cell, env, Ng=NG_FLUX, norm=True, width_tol=1.1,
    midpoint=True):
  '''
  Same as get_Fnu_cell_evolving but returns array of Fnus per step
  '''
  cell0 = cell.iloc[0]
  if midpoint:
    cell = _midpoint_cell(cell)
  K0 = norm_plaw_distrib(cell0.gmin, cell0.gmax, env.psyn)
  Tarr = np.atleast_1d(np.asarray(Tobs, dtype=float))
  Fnu = np.zeros((len(cell), Tarr.size, np.size(nuobs)))
  cols = precompute_step_cols(cell, env, midpoint_hydro=midpoint)
  Ton_arr = cols['obsT'][0]                                # onset (tT=1)
  for j in range(len(cell)):
    # only Tarr >= Ton receives flux: slice instead of computing + masking
    iT0 = np.searchsorted(Tarr, Ton_arr[j])
    if iT0 >= Tarr.size:
      continue                                             # onset after window
    Fnu[j][iT0:] += get_Fnu_step(nuobs, Tarr[iT0:], step_view(cols, j), K0, env,
                                 Ng, norm, width_tol)
  return Fnu if np.ndim(Tobs) > 0 else Fnu[:,0,:]

def get_Fnu_cell_evolving(nuobs, Tobs, cell, env, Ng=NG_FLUX, norm=True, width_tol=1.1,
    midpoint=True):
  '''
  F_nu(Tobs) of an evolving cell, summed over its cooling steps.
  'cell' is the dataframe from generate_cell_withDistrib (already binned in
  cooling time). K0 is fixed by the initial (injection) distribution bounds.

  midpoint: evaluate each step at its geometric mean rather than at its left edge,
  in BOTH of the places a step has a state -- the electron bounds (gmin, gmax, bsyn,
  via _midpoint_cell) and the emission prefactor (V3p, nu'_B, Pmax, Aad, via
  precompute_step_cols(midpoint_hydro=...)). A step emits over a finite dtt, during
  which every electron cools from gma_L to gma_R AND the hydro moves under them, so
  the left-edge state is a first-order overshoot on both counts: ~4.7% at r_ref=1.1
  (8.9% at 1.2, 2.4% at 1.05) from the bounds, and up to 3.8% on the per-frequency
  fluence in slow cooling from the prefactor. The geometric mean is the representative
  state for both (gma_L*gma_R is the exact finite-step weight, see _step_midpoint_gma)
  and makes the flux second order in the step.
  The kinematics (obsT, Dop) stay at the left edge either way -- see
  precompute_step_cols for why moving x is fatal, and for what error that leaves.
  False restores the historical left-edge evaluation.

  Returns shape (len(Tobs), len(nuobs)) for an array Tobs, (len(nuobs),) for a scalar.
  '''
  cell0 = cell.iloc[0]
  K0 = norm_plaw_distrib(cell0.gmin, cell0.gmax, env.psyn)
  if midpoint:
    cell = _midpoint_cell(cell)
  Tarr = np.atleast_1d(np.asarray(Tobs, dtype=float))
  Fnu = np.zeros((Tarr.size, np.size(nuobs)))
  # frequency-band cut (same criterion as get_Fnu_cell): 
  # drop steps that will not contribute to observed flux
  # margin of 10 because we use exact syn instead of sharp cut at gmax
  _BAND_MARGIN = 10.
  nu_B0 = get_variable(cell0, 'nu_B', env)
  gmax_cut = max(1., np.sqrt(np.min(nuobs)/(nu_B0*_BAND_MARGIN)))
  gmax_arr = cell.gmax.to_numpy()
  N = gmax_arr.size
  jcut = min(max(N - np.searchsorted(gmax_arr[::-1], gmax_cut, side='right'), 1), N)
  cols = precompute_step_cols(cell, env, midpoint_hydro=midpoint)   # per-step scalars once
  Ton_arr = cols['obsT'][0]                                # onset (tT=1)
  for j in range(jcut):
    # only Tarr >= Ton receives flux: slice instead of computing + masking
    iT0 = np.searchsorted(Tarr, Ton_arr[j])
    if iT0 >= Tarr.size:
      continue                                             # onset after window
    Fnu[iT0:] += get_Fnu_step(nuobs, Tarr[iT0:], step_view(cols, j), K0, env, Ng, norm, width_tol)
  return Fnu if np.ndim(Tobs) > 0 else Fnu[0]

def _shared_step_prefix(cols_a, cols_b):
  '''
  Number of leading cooling steps on which two precomputed column dicts
  (precompute_step_cols) are EXACTLY equal -- the steps whose emission is therefore
  the same number in both, and need be evaluated only once.

  Compared on the precomputed columns rather than on the raw cell frames because
  those are what get_Fnu_step actually consumes, and because it makes the one-row
  stencil of _midpoint_cell fall out automatically (a midpoint column already
  carries the j/j+1 mixing, so an index that survives here is genuinely shared).

  Exact `==`, never a tolerance: the point of the pairing is that the two sides stay
  bit-identical, so a near-match must NOT be treated as a match. NaN compares false,
  which truncates the prefix early -- conservative, i.e. less sharing, never wrong.
  '''
  n = min(len(cols_a['gmax']), len(cols_b['gmax']))
  if n == 0:
    return 0
  eq = np.ones(n, dtype=bool)
  for key, va in cols_a.items():
    vb = cols_b[key]
    if key == 'obsT':                      # (Ton, Tth, Tej) triple of arrays
      for ta, tb in zip(va, vb):
        eq &= (np.asarray(ta)[:n] == np.asarray(tb)[:n])
    else:
      eq &= (np.asarray(va)[:n] == np.asarray(vb)[:n])
    if not eq.any():
      return 0
  bad = np.flatnonzero(~eq)
  return int(bad[0]) if bad.size else n


def get_Fnu_cell_evolving_pair(nuobs, Tobs, cell_full, cell_cut, env, Ng=NG_FLUX, norm=True,
    width_tol=1.1, midpoint=True):
  '''
  F_nu(Tobs) of ONE cell under both rarefaction treatments at once: `cell_full`
  followed to its last snapshot (rar_cut=None) and `cell_cut` truncated at R_rar
  (rar_cut='model'). Returns (Fnu_full, Fnu_cut), each BIT-IDENTICAL to the
  corresponding separate get_Fnu_cell_evolving call.

  Why this is worth doing: the cut history is a truncation of the full one, so the
  two cooling-step tables agree exactly over a long leading run and the emission of
  those steps is computed twice for nothing. Evaluating them once is not a small
  saving, because get_Fnu_step is called on Tarr[iT0:] -- the EARLY steps span
  almost the whole observer grid and are the expensive ones, and they are precisely
  the ones the cut keeps.

  The shared length is MEASURED (_shared_step_prefix), never assumed. The tables do
  not simply agree up to the cut: _densify_nodes places nodes from the local
  variation of syn/lfac, so near the truncation its stencil sees different
  neighbours and the step binning shifts for the last ~20 rows (measured on
  cooling_g100: cut tables of 137/171/96 steps sharing 134/149/73). Assuming the
  prefix ran to the cut would corrupt those rows.

  Bit-identity holds because floating-point addition is order-dependent and the
  order is preserved: each side accumulates steps 0,1,2,... into the same array
  elements in the same sequence as the single-cell function does.
  '''
  # The cut truncates the history, it never moves its start, so the injection row --
  # which fixes K0 and the band cut -- is shared. Assert rather than assume.
  c0f, c0c = cell_full.iloc[0], cell_cut.iloc[0]
  if not (float(c0f.gmin) == float(c0c.gmin) and float(c0f.gmax) == float(c0c.gmax)):
    raise ValueError('get_Fnu_cell_evolving_pair: the two cells do not share an '
                     'injection state, so they are not the same cell')
  K0 = norm_plaw_distrib(c0f.gmin, c0f.gmax, env.psyn)
  if midpoint:
    cell_full, cell_cut = _midpoint_cell(cell_full), _midpoint_cell(cell_cut)
  Tarr = np.atleast_1d(np.asarray(Tobs, dtype=float))
  _BAND_MARGIN = 10.                       # as get_Fnu_cell_evolving
  nu_B0 = get_variable(c0f, 'nu_B', env)
  gmax_cut = max(1., np.sqrt(np.min(nuobs)/(nu_B0*_BAND_MARGIN)))

  def _jcut(cell):
    g = cell.gmax.to_numpy()
    N = g.size
    return min(max(N - np.searchsorted(g[::-1], gmax_cut, side='right'), 1), N)

  jcut_f, jcut_c = _jcut(cell_full), _jcut(cell_cut)
  cols_f = precompute_step_cols(cell_full, env, midpoint_hydro=midpoint)
  cols_c = precompute_step_cols(cell_cut, env, midpoint_hydro=midpoint)
  n_shared = min(_shared_step_prefix(cols_f, cols_c), jcut_f, jcut_c)

  def _add_steps(Fnu, cols, j0, j1):
    Ton_arr = cols['obsT'][0]
    for j in range(j0, j1):
      iT0 = np.searchsorted(Tarr, Ton_arr[j])
      if iT0 >= Tarr.size:
        continue                           # onset after the window
      Fnu[iT0:] += get_Fnu_step(nuobs, Tarr[iT0:], step_view(cols, j), K0, env,
                                Ng, norm, width_tol)
    return Fnu

  shared = _add_steps(np.zeros((Tarr.size, np.size(nuobs))), cols_f, 0, n_shared)
  out = [_add_steps(shared.copy(), cols, n_shared, jc)
         for cols, jc in ((cols_f, jcut_f), (cols_c, jcut_c))]
  if np.ndim(Tobs) > 0:
    return out[0], out[1]
  return out[0][0], out[1][0]


def get_Fnu_cell_instant(nuobs, Tobs, cell, env, Ng=NG_FLUX, norm=True, width_tol=1.1,
    midpoint=True):
  '''
  F_nu(Tobs) of evolving cell, but all contributions are summed and attributed 
  to initial lab-frame time to artificially generate vFC
  Sums fluence (F_nu(T_on) x Tth) at each step then multiply by tail tT^-2/Tth0
  same radiated energy as get_Fnu_cell_instant

  Returns shape (len(Tobs), len(nuobs)) for an array Tobs, (len(nuobs),) for a scalar.
  '''
  nuobs   = np.atleast_1d(nuobs)
  scalarT = (np.ndim(Tobs) == 0)
  Tobs    = np.atleast_1d(Tobs)
  cell0 = cell.iloc[0]
  K0 = norm_plaw_distrib(cell0.gmin, cell0.gmax, env.psyn)
  if midpoint:
    cell = _midpoint_cell(cell)

  # fluence spectrum: each step's peak amplitude (tT=1, i.e. Tobs=Ton) x its width
  Phi = np.zeros(nuobs.shape)
  for j in range(len(cell)):
    step = cell.iloc[j]
    Ton, Tth, Tej = get_variable(step, 'obsT', env)
    A_j = get_Fnu_step(nuobs, Ton, step, K0, env, Ng, norm, width_tol)  # tT=1
    Phi += A_j * Tth

  # re-emit the whole fluence with the injection-radius high-latitude profile
  Ton0, Tth0, Tej0 = get_variable(cell0, 'obsT', env)
  tT0 = (Tobs - Tej0)/Tth0
  prof = np.where(tT0 >= 1., tT0**-2, 0.) / Tth0   # int prof dTobs = 1
  out = prof[:, None] * Phi[None, :]
  return out[0] if scalarT else out

def get_nuFnu(func_Fnu, nuobs, Tobs, cell, env, norm=True, **kwargs):
  '''
  Compute nu F_nu, shape (len(Tobs), len(nuobs)) for an array Tobs, (len(nuobs),)
  for a scalar. func_Fnu must follow the same convention (get_Fnu_cell_evolving
  or get_Fnu_cell_instant).
  '''

  nu0 = env.nu0FS if (cell.iloc[0].trac > 1.5) else env.nu0
  nub = nuobs/nu0 if norm else nuobs
  if np.ndim(Tobs) > 0:
    nub = nub[np.newaxis, :]
  nF = nub * func_Fnu(nuobs, Tobs, cell, env, norm=norm, **kwargs)
  return nF

def _step_midpoint_gma(cell, col, rate_col=None):
  '''
  Geometric mean of each cooling step's (left, right) value, from the stored left
  edges: right edge of step j = left edge of step j+1, and the last step is closed
  with the synchrotron law (factor 1/(1+g*dtt)). rate_col names the column whose
  gamma sets that factor -- itself for gmin/gmax, but gmax for bsyn, which is the
  cumulative burn-off of the TOP edge.
  Used by the FLUX kernels (_midpoint_cell): a step's emission integrated over its
  duration is that of the geometric-mean state, since 1/gma_R = 1/gma_L + dtt gives
  gma_L*gma_R = gma_L**2/(1+gma_L*dtt) -- exactly the finite-step weight. Evaluating
  the spectrum there is second order in the step, against first order (a ~4.7%
  overshoot at r_ref=1.1) at the left edge. cell_radiated_energy does NOT use this:
  it applies the same weight per electron and exactly, inside step_radiated_energy.
  '''
  v = cell[col].to_numpy()
  g = cell[rate_col or col].to_numpy()
  dtt = cell['dtt'].to_numpy()
  v_r = np.empty_like(v)
  v_r[:-1] = v[1:]
  v_r[-1] = v[-1]/(1. + g[-1]*dtt[-1])
  return np.sqrt(v*v_r)

def _midpoint_cell(cell):
  '''
  Cell frame with gmin/gmax/bsyn replaced by their per-step geometric means
  (_step_midpoint_gma), i.e. the state whose emission represents the whole step.
  Returned unchanged when the frame carries no dtt column (analytic cells).
  '''
  if 'dtt' not in cell:
    return cell
  mids = dict(gmin=_step_midpoint_gma(cell, 'gmin'),
              gmax=_step_midpoint_gma(cell, 'gmax'))
  if 'bsyn' in cell:
    mids['bsyn'] = _step_midpoint_gma(cell, 'bsyn', rate_col='gmax')
  return cell.assign(**mids)

def cell_radiated_energy(cell, env, Ng=120, width_tol=1.01):
  '''
  Frame-independent total comoving radiated energy of a cell, summed over its
  cooling steps:
    E' = sum_j  nu'_B,j * V3p_j * int emiss_j(tnu) dtnu
  The frequency integral is done analytically per step (step_radiated_energy:
  int syn_emiss dtnu = (3/2)gma**2), so only the smooth electron integral over gma
  remains -- unbiased and ~1000x cheaper than the former tnu-grid trapezoid, and
  free of the log-grid overshoot that biased the fast-cooling efficiency high.
  The two halves of a step's state sit in different places, and both are deliberate:

    ELECTRONS at the step's LEFT edge. step_radiated_energy applies the exact
    finite-step weight (3/2)gma**2/(1+gma*dtt) per electron, which is what the
    distribution actually radiates over the step (me c^2 (gma_L - gma_R)). That is
    exact, so the budget does not drift with the cooling-step size -- it replaces the
    former midpoint-bounds substitution, which was the right idea applied to the bounds
    rather than per electron and left a first-order residue (eps_rad 1.0029/1.0010/1.0005
    at r_ref=1.2/1.1/1.05, then 1.00032/1.00031/1.00027).

    PREFACTOR (Aad*Pmax*V3p*nu'_B) at the step's MIDPOINT, via precompute_step_cols'
    midpoint_hydro. Nothing makes that factor exact over a finite step, so it is an
    ordinary quadrature and the midpoint rule is the cheap second-order choice. Left-edge
    was worth +2.0/+2.6/+3.1% on E_rad at log10(gc/gm) = 0/+1/+3 -- i.e. eps_rad read
    that much high in slow cooling -- against <=0.07% here. See precompute_step_cols.
  '''
  cell0 = cell.iloc[0]
  K0 = norm_plaw_distrib(cell0.gmin, cell0.gmax, env.psyn)   # injection state
  cols = precompute_step_cols(cell, env, keys=('nup_B', 'V3p', 'Pmax'))
  nupB_arr, V3p_arr = cols['nup_B'], cols['V3p']
  E = 0.
  for j in range(len(cell)):
    E += step_radiated_energy(step_view(cols, j), K0, env, Ng, width_tol) * nupB_arr[j] * V3p_arr[j]
  return E

def cell_injected_energy(cell, env):
  '''
  Comoving energy deposited in the accelerated electrons of a cell, i.e. the
  budget cell_radiated_energy draws on: the power law K0*gma**-p between the
  injection bounds (gmin0, gmax0), minus the rest mass the electrons keep.
    E'_inj = xi_e n' V'_3 m_e c^2 [ K0 (gmin0**(2-p) - gmax0**(2-p))/(p-2) - 1 ]
  This is NOT eps_e * e'_int: derive_gma_m fixes gma_m from the gma_M -> inf
  limit (Gp=(p-2)/(p-1)), so the truncated distribution actually holds only
  derive_xiE(gmin0, gmax0, p) of eps_e*e'_int -- 0.94 deep in fast cooling,
  where the alpha rescale shrinks gma_M ~ alpha**(3/4) at fixed gma_m. The two
  forms are algebraically identical; eps_e*ei*V3p*derive_xiE is the cheap
  cross-check that derive_gma_m's internal e'_int matches derive_Eint_comoving.
  '''
  p = env.psyn
  cell0 = cell.iloc[0]
  gmin0, gmax0 = cell0.gmin, cell0.gmax
  if gmax0 <= gmin0:
    return 0.
  K0 = norm_plaw_distrib(gmin0, gmax0, p)
  Ne = env.xi_e * derive_n(get_variable(cell0, 'rho', env), env.rhoscale) \
       * get_variable(cell0, 'V3p', env)
  gma_mean = K0*(gmin0**(2-p) - gmax0**(2-p))/(p-2)
  return Ne * me_*c_**2 * (gma_mean - 1.)

def total_radiated_energy(nF, nub, Tb, env=None):       # nF = nu*F_nu, shape (T, nu)
    Enu = np.trapezoid(nF, np.log(nub), axis=1)   # ∫νFν dlnν = ∫Fν dν  per T
    E = np.trapezoid(Enu, Tb)
    if env is not None:
      E *= env.nu0F0*env.T0
    return E

def get_critlfacs(cell_rad, env):
  '''
  Returns gma_m, gma_c, gma_M associated with a cell
  '''
  cell0 = cell_rad.iloc[0]
  gma_m = get_variable(cell0, 'gma_m', env)
  gma_M = get_variable(cell0, 'gma_M', env)
  tc1 = get_variable(cell0, 'tc1', env)
  tdyn = cell_rad.iloc[-1].tp - cell0.tp
  gma_c = tc1/tdyn
  return gma_m, gma_c, gma_M

def get_critfreqs(cell_rad, env, normed=False):
  '''
  Returns nu'_m, nu'_c, nu'_M associated with a cell
  '''
  cell0 = cell_rad.iloc[0]
  nup_B = get_variable(cell0, 'nup_B', env)
  critlfacs = get_critlfacs(cell_rad, env)
  nup_m, nup_c, nup_M = [nup_B*gma**2/(1.5*env.nu0p if normed else 1.) for gma in critlfacs]
  return nup_m, nup_c, nup_M

def get_shell_critlfacs(sh_data, env):
  '''
  Shell-wide gma_m, gma_c, gma_M, the shell analog of get_critlfacs. The injection
  state (gma_m, gma_M, tc1) is taken at the first-shocked cell of the shell (the
  earliest shocking time in sh_data), as cell0 is the shocking state for a cell.
  The shell dynamical time replaces the cell's worldline span: tdyn = comoving shell
  emission span = (lab span of the shocking times)/lfac0, so gma_c = tc1/tdyn.
  sh_data: the shock-front dataframe (cellsBehindShock_fromData) for the shell.
  '''
  cell0 = sh_data.loc[sh_data.t.idxmin()]                 # first-shocked cell
  gma_m = get_variable(cell0, 'gma_m', env)
  gma_M = get_variable(cell0, 'gma_M', env)
  tc1 = get_variable(cell0, 'tc1', env)
  tdyn = (sh_data.t.max() - sh_data.t.min()) / env.lfac0  # comoving shell duration
  gma_c = tc1/tdyn
  return gma_m, gma_c, gma_M

def get_shell_critfreqs(sh_data, env, normed=False):
  '''
  Shell-wide nu'_m, nu'_c, nu'_M, the shell analog of get_critfreqs (same
  nu' = nup_B*gma**2 convention, normalized by 1.5*env.nu0p if normed). nup_B and the
  critical Lorentz factors are taken at/for the first-shocked cell of the shell; see
  get_shell_critlfacs for the shell-wide gma_c (comoving shell emission span).
  '''
  cell0 = sh_data.loc[sh_data.t.idxmin()]
  nup_B = get_variable(cell0, 'nup_B', env)
  critlfacs = get_shell_critlfacs(sh_data, env)
  nup_m, nup_c, nup_M = [nup_B*gma**2/(1.5*env.nu0p if normed else 1.) for gma in critlfacs]
  return nup_m, nup_c, nup_M

def get_cell_nuFnu(key, k, func_Fnu=get_Fnu_cell_evolving,
    u_scale=1., alpha=1., zeta=1., cleanData=False, r_ref=1.2,
    Tmax=5, NT=500, lognu_min=-2.5, lognu_max=2.5, Nnu=400,
    return_cell=False, dlnrho_max=DLNRHO_MAX, **kwargs):
  '''
  The whole chain to obtain nu Fnu.
  With u_scale != 1 the env is rescaled (as if u1 -> u1*u_scale); the observer
  arrays nuobs, Tobs are built from that rescaled env (its nu0, Ts, T0), so the
  whole pipeline stays self-consistent in observer space.
  With return_cell=True the generated cell (cell_dist, incl. the exit_row
  rarefaction cutoff) is appended to the output, so callers can reuse the exact
  same cell (e.g. for get_cell_stepspectra) without re-deriving exit_row.
  '''
  # dimensionless observer grids (env-independent); scaled below with the
  # (possibly rescaled) env returned by generate_cell_withDistrib
  nub, T, env = obs_arrays(key, normed=True, Tmax=Tmax, NT=NT,
      lognu_min=lognu_min, lognu_max=lognu_max, Nnu=Nnu)
  z = 4 if (k <= env.Next + env.Nsh4) else 1
  sh_data = open_rundata(key, z)
  sh_data = cellsBehindShock_fromData(sh_data)
  cell_data = open_celldata(key, k)
  if cell_data is False:
    extract_data_cells(key, [k], noOut=True)
    cell_data = open_celldata(key, k)
  cell_d0 = sh_data.loc[sh_data.i==cell_data.iloc[0].i].iloc[0]
  # shell-exit interface (last-shocked cell) = rarefaction launch point (from data)
  exit_row = sh_data.loc[sh_data.t.idxmax()]
  cell_dist, env = generate_cell_withDistrib(cell_data, cell_d0, env,
                  u_scale=u_scale, alpha=alpha, zeta=zeta, cleanData=cleanData,
                  r_ref=r_ref, Tmax=Tmax, key=key, k=k, exit_row=exit_row,
                  dlnrho_max=dlnrho_max)
  # scale the observer grids with the rescaled env (matches obs_arrays)
  nuobs = nub * env.nu0
  Tobs = env.Ts + (T - 1) * env.T0
  if cell_dist is False:
    return (nuobs, Tobs, env, None, cell_dist) if return_cell else (nuobs, Tobs, env, None)
  nuFnu = get_nuFnu(func_Fnu, nuobs, Tobs, cell_dist, env, **kwargs)
  if return_cell:
    return nuobs, Tobs, env, nuFnu, cell_dist
  return nuobs, Tobs, env, nuFnu

def _interp_state(a, b, f, dx):
  '''
  Linear interp of the primitive columns between onset-adjacent states a, b at
  fraction f; region labels (trac, i) kept from a; dx set explicitly (the
  flux-conserving onset-interval weight). Shared by the subcell_dlogT
  refinement of get_shell_nuFnu (fit path) and get_shell_nuFnu_fromData
  (data path, working_cooling_data).
  '''
  out = a.copy()
  for col in ('x', 't', 'vx', 'rho', 'p', 'vx_u'):
    if col in a.index and col in b.index:
      out[col] = a[col]*(1.-f) + b[col]*f
  out['dx'] = dx
  return out

def compute_subcell_edges(barT_on, floor, subcell_dlogT, subcell_max, subcell_min=2,
    anchor_first=False):
  '''
  Adaptive sub-cell onset edges for the early-lightcurve staircase smoothing.
  The cell onsets bar{T}_on are ~linearly spaced, so the earliest cells (near the
  CD, onsets ~0..few*dbarT) span many decades of the log-time axis: their onset
  intervals are split so the early rise is resolved down to the grid floor.
  barT_on: per-cell onset (klist order, NaN where unknown); floor: earliest
  resolved bar{T} on the obs grid. Returns a list (len(barT_on)) of
  (a, b, edges) tuples or None (never the last index).

  Each parent's interval [a,b] is split into n_k = clip(ceil(gap/subcell_dlogT),
  subcell_min, subcell_max) equal log-intervals. Sub-cell weights follow the interval
  widths (dx * (e1-e0)/(b-a) in the emitters), so the split is flux-conserving.

  anchor_first extends the FIRST usable parent's sub-cells down to the grid floor,
  reconstructing the early rise the snapshot cadence could not resolve (the data path
  wants this; the fit path's worldline already reaches bar{T}=0, so its first onset is
  0 and the two are the same thing there). It moves the parent's EDGES only: (a,b) --
  which sets both the dx per unit bar{T} the window carries and the endpoints of the
  state interpolation -- stays the cells' own onsets.
  THE CALLERS USED TO DO THIS BY SETTING barT_on[first] = 0, and that is the bug this
  keyword exists to close. Widening (a,b) widens the interval one cell's dx is spread
  over, i.e. it lowers the EMISSION RATE. That was a no-op under early_ana='shockfit',
  whose onsets are the cells' leading edges (first onset exactly 0); under 'measured'
  the event is the cell CENTRE -- on cooling_g100 z=4 the first onset is 0.507 of the
  ladder spacing -- so the first window ran 1.507x too wide on one cell's mass. The
  resulting 34% deficit in emitter density healed the instant the next cell landed,
  putting a step of +0.54 in d ln(nuFnu)/d ln(bar{T}) at bar{T} = barT_on[1]. It is
  resolution-placed, not resolution-cured: the ladder spacing is bar{T}_f/N_sh, so the
  step sat at bar{T}/bar{T}_f = 3.0e-3 on the 500-cell shell and 1.5e-4 on the
  10000-cell one, the second merely left of the plotted window.

  subcell_min = 2 IS WHAT KEEPS THE EMITTER DENSITY SMOOTH, and it is not free (it
  splits every parent, +51% emitters on cooling_g100 z=4: 872 -> 1316). n_k is an
  integer per parent, so the realised onset SPACING steps wherever ceil() does; the
  worst step is the last one, n_k = 2 -> 1, where refinement switches OFF and the
  spacing doubles in one cell. A floor of 2 removes that step by never reaching n=1 --
  past the refined region the spacing is just gap/2, which varies as smoothly as the
  gaps do. The remaining steps (3->2, 4->3, ...) are 1.5x, 1.33x, ... and shrink as they
  move inward. Measured on the real onsets, the spacing through the old crossover is
  then even to max/min = 1.2 per bin, against 3.9 with the floor at 1.

  THE n_k = 2 -> 1 STEP WAS OBSERVABLE, contrary to what this docstring used to say.
  That claim was measured with subcell_dlogT = 0.05, where the boundary sat at cell
  k=497 -- onsets are ~linearly spaced, so the log gap falls as 0.434/rank and the
  boundary MOVES with the setting. At 0.008 it moved to cell k=465 (bar{T} = 0.1421,
  bar{T}/bar{T}_f = 0.1085) and showed up in the FAST-cooling lightcurves as a step in
  the local temporal index: detrending d ln(nuFnu)/d ln(bar{T}) against a smooth decline
  over bar{T}/bar{T}_f = 0.02..0.4 put the largest departure at 0.107-0.117 for every
  fast-cooling point at every plotted frequency, +0.056 to +0.068 in the index (~0.2-0.6%
  in flux, which is why the flux panels never showed it). Slow cooling saw nothing:
  there each emitter contributes a long smooth decay rather than a spike, so halving the
  emitter density does not alias.
  A SINGLE GLOBAL LADDER WAS TRIED FOR THIS AND REJECTED -- do not re-derive it. Onsets
  at bar{T} = 10^(j*subcell_dlogT) clipped to each parent have a continuous density by
  construction, but a parent whose edge falls near a ladder point is then split into a
  sliver and a near-full piece: the spacing goes uneven by max/min = 50-140 past the
  crossover, against 1.2 here, and on a recomputed logr=-4 point it removed only part of
  the feature (index residual 0.068 -> 0.030 at nu = 0.01 nu_pk, 0.068 -> 0.042 at
  nu_pk, and 0.044 -> 0.045, i.e. nothing, at 0.1 nu_pk). Merging the slivers back trades
  that for unevenness of 5-7 in the refined region. Uniform-within-parent is the property
  worth keeping; the floor is the cheap way to keep it AND lose the step.
  What the old measurement DID establish stands, and is a different feature: the
  fast-cooling kink at bar{T}/bar{T}_f = 0.044 is the early_ana='shockfit' prepend
  boundary, and no change to this scheme moves it.
  '''
  first = None
  if anchor_first:
    fin = np.flatnonzero(np.isfinite(barT_on))
    first = int(fin[0]) if len(fin) else None
  sub_edges = [None]*len(barT_on)
  for idx in range(len(barT_on)-1):
    a, b = barT_on[idx], barT_on[idx+1]
    if not (np.isfinite(a) and np.isfinite(b)) or b <= 0.:
      continue
    # ANCHOR THE EDGES, NOT THE ONSET: the anchored parent's sub-cells reach down to the
    # grid floor, but the interval (a, b) it is spread over stays its OWN. See the
    # anchor_first note above for what conflating the two cost.
    lo = floor if idx == first else max(a, floor)
    if b <= lo:
      continue
    nk = int(np.clip(np.ceil((np.log10(b)-np.log10(lo))/subcell_dlogT),
                     subcell_min, subcell_max))
    if nk > 1:
      sub_edges[idx] = (a, b, np.geomspace(lo, b, nk+1))
  return sub_edges

def check_extracted_cells(key):
  '''
  Return a sorted array of cell ids k already extracted with extract_data_cells
  (i.e. cells with a saved results/{key}/cells/{k:04d}.{ext} file, ext in IO.CELL_EXTS)

  Both storage formats count, and a cell present in both counts once: old runs are on
  CSV, new ones on npz (IO.CELL_FMT), and a partially re-extracted run is legitimate.

  Only all-digit stems are cells. The same directory holds the per-cell fit caches
  ({k:04d}_fit.npz, load_or_fit_celldata), which are npz too and would otherwise be
  picked up as cells the moment npz became a cell format.
  '''
  cells_dir = get_dirpath(key) + 'cells/'
  if not os.path.isdir(cells_dir):
    return np.array([], dtype=int)
  stems = (os.path.splitext(os.path.basename(f))[0]
           for ext in CELL_EXTS for f in glob.glob(cells_dir + f'*.{ext}'))
  ks = {int(s) for s in stems if s.isdigit()}
  return np.array(sorted(ks), dtype=int)

def get_shell_nuFnu(key, z, u_scale=1., alpha=1., zeta=1., klist=None,
    VFC=False, analytic=False, cleanData=False, r_ref=1.2,
    Tmax=5, NT=500, lognu_min=-2.5, lognu_max=2.5, Nnu=400,
    dlnrho_max=DLNRHO_MAX, Tb_min=None, Tb_lin=None, subcell_dlogT=None, subcell_max=32,
    return_energies=False, energies_only=False, **kwargs):
  '''
  Computes total nu F_nu from shell z of simulation key, summed over its cells.
  With u_scale != 1 the env is rescaled (as if u1 -> u1*u_scale); the observer
  grids nuobs, Tobs are built from that rescaled env (its nu0, Ts, T0), so the
  whole pipeline stays self-consistent in observer space (as get_cell_nuFnu).
  klist restricts the sum to a subset of cell ids (defaults to all of shell z).
  VFC forces all contributions from a cell to be emitted at same lab frame time
  analytic uses the 'analytic' cell for calculation (Gma = cst, rho ~ R^-2, B ~ R^-1)
  subcell_dlogT: if set, the discrete cell-sum staircase in the early lightcurve
    is smoothed by splitting cells into interpolated sub-cells wherever consecutive
    cell onsets are more than subcell_dlogT apart in log10(bar{T}_onset) (i.e. the
    CD-adjacent early cells). Each parent is split into n_k = clip(ceil(dlog/
    subcell_dlogT), 1, subcell_max) sub-cells that reuse its fitted profile and
    carry dx/n_k (flux-conserving). None keeps the raw one-cell-per-cell sum.
  return_energies: if True, also accumulate and return three comoving energies for
    the radiative-efficiency budget: E_rad = sum over cells of cell_radiated_energy
    (total comoving radiated energy), E_int = sum over cells of e'*V3p at the
    shocking state (comoving internal energy right after each cell is shocked),
    and E_inj = sum over cells of cell_injected_energy (energy actually deposited
    in the truncated electron power law, i.e. the budget E_rad draws on; this is
    eps_e*E_int only in the gma_M -> inf limit, see cell_injected_energy).
    eps_rad = E_rad/E_inj. Returns (nuobs, Tobs, env_rs, nuFnu_shell, E_rad,
    E_int, E_inj) instead of the 4-tuple.
  energies_only: skip the flux kernel and return the energy budget alone, with
    None in the nuFnu slot (implies return_energies). Same contract as
    get_shell_nuFnu_fromData's flag -- see its docstring for what still has to
    match the run being compared against.
  NB: cells shocked at R/R0 > ~Tmax arrive after the Tobs window and contribute
  nothing there; take Tmax above env.RfRS0 (or RfFS0) to capture the full shell.
  '''
  return_energies = return_energies or energies_only

  nub, T, env = obs_arrays(key, normed=True, Tmax=Tmax, NT=NT,
      lognu_min=lognu_min, lognu_max=lognu_max, Nnu=Nnu, Tb_min=Tb_min, Tb_lin=Tb_lin)
  env_rs = rescale_proper_velocities(u_scale, env) if u_scale != 1. else env
  if alpha != 1. or zeta != 1.:
    env_rs = rescale_hydro(alpha, zeta, env_rs)
  nuobs = nub * env_rs.nu0
  Tobs = env_rs.Ts + (T - 1) * env_rs.T0
  nuFnu_shell = None if energies_only else np.zeros((len(Tobs), len(nuobs)))
  sh_data = open_rundata(key, z)
  sh_data = cellsBehindShock_fromData(sh_data)
  # shell-exit interface (last-shocked cell) = rarefaction launch point, once per shell
  exit_row = sh_data.loc[sh_data.t.idxmax()]
  func_Fnu = get_Fnu_cell_instant if VFC else get_Fnu_cell_evolving

  # check cells that have not been extracted
  k4 = env.Next
  kCD = k4 + env.Nsh4
  k1 = kCD + env.Nsh1
  kmin, kmax = (k4, kCD) if (z==4) else (kCD, k1)
  if klist is None:
    klist = np.arange(kmin, kmax)
    if z==4: klist = np.flip(klist)
  done = check_extracted_cells(key)
  todo = [k for k in klist if k not in done]
  if todo:
    extract_data_cells(key, todo, noOut=True)

  # single shared rarefaction head -> {cell: R_rar/R_inj} for the whole shell, built once
  # per (key, z) and reused across every cell and alpha (alpha-invariant). AFTER the
  # extraction above: compute_shell_rarefaction_head skips cells with no data on disk,
  # so building it first would freeze (and cache) a head made of whatever subset of the
  # shell happened to be extracted at the time.
  rar_map = load_shell_rarefaction(key, z, env, n_shell=len(sh_data))

  # per-cell shocked state (in klist=onset order) and adaptive sub-cell edges
  # (compute_subcell_edges: geometric split of the CD-adjacent onset intervals)
  rows = [(sh_data.loc[sh_data.i==k].iloc[0] if len(sh_data.loc[sh_data.i==k]) else None)
          for k in klist]
  barT_grid = T - 1.
  floor = barT_grid[barT_grid > 0.].min()   # earliest resolved bar{T} on the obs grid
  sub_edges = [None]*len(klist)
  if subcell_dlogT is not None:
    barT_on = np.array([((get_variable(r, 'Ton', env) - env.Ts)/env.T0) if r is not None else np.nan
                        for r in rows])
    sub_edges = compute_subcell_edges(barT_on, floor, subcell_dlogT, subcell_max)

  # loop over cells
  skipped = []
  n_refined = 0
  E_rad = 0.    # total comoving radiated energy, summed over cells (return_energies)
  E_int = 0.    # sum of comoving internal energy right after each cell is shocked
  E_inj = 0.    # sum of comoving energy deposited in the electron power law
  def _accum_energy(cell, cell_env):
    nonlocal E_rad, E_int, E_inj
    E_rad += cell_radiated_energy(cell, cell_env)
    E_inj += cell_injected_energy(cell, cell_env)
    c0 = cell.iloc[0]
    E_int += get_variable(c0, 'ei', cell_env) * get_variable(c0, 'V3p', cell_env)
  for idx, k in enumerate(klist):
    # open data, generate distribution
    cell_data = open_celldata(key, k)
    if cell_data is False:
      skipped.append(k)
      continue
    sel = sh_data.loc[sh_data.i==cell_data.iloc[0].i]
    if not len(sel):
      skipped.append(k)
      continue
    if sub_edges[idx] is None:
      cell, cell_env = generate_cell_withDistrib(cell_data, sel.iloc[0], env,
                      u_scale=u_scale, alpha=alpha, zeta=zeta, cleanData=cleanData,
                      r_ref=r_ref, Tmax=Tmax, key=key, k=k, exit_row=exit_row,
                      dlnrho_max=dlnrho_max, rar_map=rar_map)
      if cell is False:
        skipped.append(k)
        continue
      if analytic: cell = generate_cell_constLfac(cell, cell_env)
      if not energies_only:
        nuFnu_shell += get_nuFnu(func_Fnu, nuobs, Tobs, cell, cell_env, **kwargs)
      if return_energies: _accum_energy(cell, cell_env)
    else:
      # split the parent's onset interval into flux-conserving sub-cells at
      # geometrically-spaced onsets, reusing its (self-similar) fitted profile
      a, b, edges = sub_edges[idx]
      cd0 = rows[idx]
      norms = [get_variable(cd0, v, env) for v in ('rho', 'lfac', 'p')]
      try:
        popts = load_or_fit_celldata(cell_data, ['rho', 'lfac', 'p'], norms, env,
                                     cd0.x, cleanData=cleanData, key=key, k=k)
      except RuntimeError:
        skipped.append(k)
        continue
      for j in range(len(edges)-1):
        e0, e1 = edges[j], edges[j+1]
        onset_c = np.sqrt(e0*e1)                       # geometric centre
        f = float(np.clip((onset_c - a)/(b - a), 0., 1.))
        dx_w = cd0.dx * (e1 - e0)/(b - a)              # onset-width weight => flux conserved
        sub = _interp_state(cd0, rows[idx+1], f, dx_w)
        cell, cell_env = generate_cell_withDistrib(None, sub, env,
                        u_scale=u_scale, alpha=alpha, zeta=zeta, cleanData=cleanData,
                        r_ref=r_ref, Tmax=Tmax, key=None, k=None, exit_row=exit_row,
                        dlnrho_max=dlnrho_max, popts=popts, rar_map=rar_map)
        if cell is False:
          continue
        if analytic: cell = generate_cell_constLfac(cell, cell_env)
        if not energies_only:
          nuFnu_shell += get_nuFnu(func_Fnu, nuobs, Tobs, cell, cell_env, **kwargs)
        if return_energies: _accum_energy(cell, cell_env)
      n_refined += len(edges) - 2
  if skipped:
    print(f'get_shell_nuFnu on sim {key}: skipped {len(skipped)} cells (no data or failed fit): {skipped}')
  if subcell_dlogT is not None:
    kr = [int(klist[i]) for i in range(len(klist)) if sub_edges[i] is not None]
    print(f'get_shell_nuFnu subcell refinement: +{n_refined} sub-cells over '
          f'{len(kr)} cells' + (f' (k={min(kr)}..{max(kr)})' if kr else ''))

  if return_energies:
    return nuobs, Tobs, env_rs, nuFnu_shell, E_rad, E_int, E_inj
  return nuobs, Tobs, env_rs, nuFnu_shell

def get_thinshell_nuFnu(key, front='RS', u_scale=1., alpha=1., zeta=1., norm=True, cutoff=False,
    Tmax=5, NT=500, lognu_min=-2.5, lognu_max=2.5, Nnu=400):
  '''
  nu F_nu from a shock front in the thinshell approximation (very fast cooling,
  run_nuFnu_vFC), with the same u_scale conventions as get_cell/shell_nuFnu:
  the env is rescaled (as if u1 -> u1*u_scale), the shock-front data is rescaled
  with rescale_shocked_data, and the observer grids nuobs, Tobs are built from
  the rescaled env (its nu0/nu0FS, Ts, T0/T0FS, matching get_radiation_vFC).
  front = 'RS' (shell 4) or 'FS' (shell 1).
  '''
  nub, T, env_in = obs_arrays(key, normed=True, Tmax=Tmax, NT=NT,
      lognu_min=lognu_min, lognu_max=lognu_max, Nnu=Nnu)
  env = rescale_proper_velocities(u_scale, env_in) if u_scale != 1. else env_in
  if alpha != 1. or zeta != 1.:
    env = rescale_hydro(alpha, zeta, env)
  z, nu0, T0 = (4, env.nu0, env.T0) if (front == 'RS') else (1, env.nu0FS, env.T0FS)
  nuobs = nub * nu0
  Tobs = env.Ts + (T - 1) * T0
  data = open_rundata(key, z)
  if type(data) == bool:
    extract_data_thinshell(key, cells=[z], savefile=True, noOut=True)
    data = open_rundata(key, z)
  data = cellsBehindShock_fromData(data)
  if u_scale != 1.:
    data = rescale_shocked_data(data, env_in, env)
  if alpha != 1.:
    data = rescale_hydro_data(data, alpha)
  nF = run_nuFnu_vFC(nuobs, Tobs, data, env, norm=norm, cutoff=cutoff)
  return nuobs, Tobs, env, nF

def get_cell_thinshell_nuFnu(key, k, u_scale=1., alpha=1., zeta=1., norm=True, cutoff=False, nu_resc=1.,
    Tmax=5, NT=500, lognu_min=-2.5, lognu_max=2.5, Nnu=400):
  '''
  nu F_nu from a single cell k in the thinshell approximation (very fast
  cooling), consistent with the other get_XX_nuFnu functions: same u_scale
  conventions (env + shocked state rescaled, observer grids built from the
  rescaled env) and same per-cell convention as run_nuFnu_vFC (nF = tnu * F
  at the moment the cell is shocked), so summing over a shell's cells
  reproduces get_thinshell_nuFnu. Only needs the shock-front data (no cell
  history file, no fits). Returns nuobs, Tobs, env, nF (None if no valid data).
  '''
  nub, T, env_in = obs_arrays(key, normed=True, Tmax=Tmax, NT=NT,
      lognu_min=lognu_min, lognu_max=lognu_max, Nnu=Nnu)
  env = rescale_proper_velocities(u_scale, env_in) if u_scale != 1. else env_in
  if alpha != 1. or zeta != 1.:
    env = rescale_hydro(alpha, zeta, env)
  z = 4 if (k <= env.Next + env.Nsh4) else 1   # same boundary as get_cell_nuFnu
  nu0, T0 = (env.nu0, env.T0) if (z == 4) else (env.nu0FS, env.T0FS)
  nuobs = nub * nu0
  Tobs = env.Ts + (T - 1) * T0
  data = open_rundata(key, z)
  if type(data) == bool:
    extract_data_thinshell(key, cells=[z], savefile=True, noOut=True)
    data = open_rundata(key, z)
  data = cellsBehindShock_fromData(data)
  # state at the moment the cell is shocked (run_nuFnu_vFC keeps 'first')
  sel = data.loc[data.i == k].drop_duplicates(subset='i', keep='first')
  if (not len(sel)) or (sel.iloc[0].x <= 0.):
    return nuobs, Tobs, env, None
  cell = sel.iloc[0]
  if u_scale != 1.:
    cell = rescale_shocked_data(cell, env_in, env)
  if alpha != 1.:
    cell = rescale_hydro_data(cell, alpha)
  tnu = nuobs/(nu0 if norm else 1.)
  nF = tnu * get_Fnu_vFC(nuobs, Tobs, cell, env, norm, cutoff, nu_resc)
  return nuobs, Tobs, env, nF


def set_slopes_ticks(ax, p):
  #ax.yaxis.labelpad = 0
  slopes = [4/3, (3-p)/2, 1-p/2, 1/2]
  snames = ['$\\frac{4}{3}$', '$\\frac{3-p}{2}$', '$1-\\frac{p}{2}$', '$\\frac{1}{2}$']
  for val in slopes:
    ax.axhline(val, c='k', ls=':')
  ax.tick_params(axis='both', which='both', length=0, labelright=False)
  for val, name in zip(slopes, snames):
    ax.text(1.01, val, name, ha='left', va='center', transform=transx(ax))
  

def plot_spectrum(logT, Tb, nub, nF, slopes=False, p=2.5, ax_in=None, **kwargs):
  iT = np.searchsorted(Tb, 10**logT)
  sp = nF[iT]
  if ax_in is None:
    fig, ax = plt.subplots()
    ax.set_xlabel(nu_label)
    ax.set_ylabel(nF_label)
  else:
    ax = ax_in
  ax.loglog(nub, sp, **kwargs)
  if slopes:
    s = logslope_arr(nub, sp)
    ax1 = ax.twinx()
    ax1.semilogx(nub, s, c='k')
    set_slopes_ticks(ax1, p)
  
def plot_lightcurve(lognu, Tb, nub, nF, ax_in=None, **kwargs):
  inu = np.searchsorted(nub, 10**lognu)
  lc = nF[:,inu]
  if ax_in is None:
    fig, ax = plt.subplots()
    ax.set_xlabel(T_label)
    ax.set_ylabel(nF_label)
  else:
    ax = ax_in
  ax.plot(Tb+1, lc, **kwargs)

def compare_spectra(logT, Tb, nub, nF_arr, slopes=False,
    colors=None, linestyles=None, labels=None):
  fig, ax = plt.subplots()
  n = len(nF_arr)
  colors     = colors     or [None]*n
  linestyles = linestyles or [None]*n
  labels     = labels     or [None]*n
  for nF, c, ls, lab in zip(nF_arr, colors, linestyles, labels):
    plot_spectrum(logT, Tb, nub, nF, slopes=slopes, ax_in=ax,
                  color=c, linestyle=ls, label=lab)
  

def compare_lightcurves(lognu, Tb, nub, nF_arr,
    colors=None, linestyles=None, labels=None):
  fig, ax = plt.subplots()
  n = len(nF_arr)
  colors     = colors     or [None]*n
  linestyles = linestyles or [None]*n
  labels     = labels     or [None]*n
  for nF, c, ls, lab in zip(nF_arr, colors, linestyles, labels):
    plot_lightcurve(lognu, Tb, nub, nF, ax_in=ax,
                  color=c, linestyle=ls, label=lab)

def get_cell_stepspectra(logT, nuobs, Tobs, cell, env, norm=True):
  '''
  Cooling-step-resolved nu F_nu of a single cell at observer time 10**logT
  (logT in normalized units, Tb = (Tobs - Ts)/T0).
  Returns (nub, nF_steps, nF_tot) with nF_steps shape (Nsteps, len(nuobs)) and
  nF_tot shape (len(nuobs),). The frequency normalization matches get_nuFnu
  (nu0FS for FS cells, nu0 for RS), so the total equals get_cell_nuFnu at that T.
  '''
  Tb = (Tobs - env.Ts)/env.T0
  iT = min(np.searchsorted(Tb, 10**logT), Tb.size - 1)   # clamp to last bin
  nu0 = env.nu0FS if (cell.iloc[0].trac > 1.5) else env.nu0
  nub = nuobs/nu0 if norm else nuobs
  Fnu_arr = get_Fnu_array_cell_evolving(nuobs, Tobs, cell, env, norm=norm)[:, iT, :]
  nF_steps = nub[np.newaxis, :] * Fnu_arr
  nF_tot   = nub * Fnu_arr.sum(axis=0)
  return nub, nF_steps, nF_tot

def plot_cellspectrum_withsteps(logT, nuobs, Tobs, cell, env, ax_in=None,
    ls='-', cmap='jet', label=None, norm=True, **tot_kw):
  '''
  Spectrum of an evolving cell at observer time 10**logT, decomposed into its
  cooling steps: the summed spectrum is drawn in black (ls, **tot_kw), each
  step is coloured along 'cmap'. Same convention as plot_spectrum.
  '''
  nub, nF_steps, nF_tot = get_cell_stepspectra(logT, nuobs, Tobs, cell, env, norm=norm)
  N = len(cell)
  colors = plt.get_cmap(cmap)(np.linspace(0, 1, N))
  if ax_in is None:
    fig, ax = plt.subplots()
    ax.set_xlabel(nu_label)
    ax.set_ylabel(nF_label)
  else:
    ax = ax_in
    fig = ax.figure
  ax.loglog(nub, nF_tot, c='k', ls=ls, lw=1., zorder=-1, label=label, **tot_kw)
  for j in range(N):
    ax.loglog(nub, nF_steps[j], c=colors[j], ls=ls)
  fig.tight_layout()
  return ax

def compare_cellspectra_withsteps(logT, cells, nuobs_arr, Tobs_arr, envs,
    linestyles=None, cmaps=None, labels=None, ax_in=None, norm=True):
  '''
  Overlay the cooling-step-resolved spectra of several cells at observer time
  10**logT. Intended to compare the same hydro cell rescaled differently (see
  generate_cell_withDistrib / get_cell_nuFnu with u_scale, alpha, zeta), so each
  cell carries its own (nuobs, Tobs, env) and stays self-consistent in observer
  space. Cells are distinguished by linestyle, their cooling steps by colormap.

  cells, nuobs_arr, Tobs_arr, envs are parallel lists (as compare_spectra).
  '''
  n = len(cells)
  linestyles = linestyles or ['-', '--', ':', '-.'][:n]
  cmaps      = cmaps      or ['jet', 'viridis', 'plasma', 'cividis'][:n]
  labels     = labels     or [None]*n
  if ax_in is None:
    fig, ax = plt.subplots()
    ax.set_xlabel(nu_label)
    ax.set_ylabel(nF_label)
  else:
    ax = ax_in
  for cell, nuobs, Tobs, env, ls, cmap, lab in zip(
      cells, nuobs_arr, Tobs_arr, envs, linestyles, cmaps, labels):
    plot_cellspectrum_withsteps(logT, nuobs, Tobs, cell, env, ax_in=ax,
        ls=ls, cmap=cmap, label=lab, norm=norm)
  if any(l is not None for l in labels):
    ax.legend()
  return ax