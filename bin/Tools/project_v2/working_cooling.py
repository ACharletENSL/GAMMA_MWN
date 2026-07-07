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

def truncate_at_rarefaction(shocked_data, slope_thresh=-8., window=15, minpts=20):
  '''
  Truncate post-shock cell data at the rarefaction arrival.
  The smooth downstream decay has d(ln p)/d(ln x) ~ -2 to -4; the back-edge
  rarefaction makes p crash with log-slopes <~ -15. Truncate at the first
  point where the look-ahead log-slope of p drops below slope_thresh.
  Keeps at least minpts points (else returns the data untruncated).
  '''
  x = shocked_data.x.to_numpy()
  p = shocked_data.p.to_numpy()
  if len(x) < minpts + window:
    return shocked_data
  lnx, lnp = np.log(x), np.log(p)
  slope = (lnp[window:] - lnp[:-window])/(lnx[window:] - lnx[:-window])
  steep = np.flatnonzero(slope < slope_thresh)
  if len(steep) and steep[0] >= minpts:
    return shocked_data.iloc[:steep[0]]
  return shocked_data

def fit_celldata(cell_data, vars, norms, env, x0=None, cleanData=False):
  '''
  Fit the variable vars, normed by norms, of cell_data
  x0 is the anchor radius (where fitfunc==1); defaults to the first cell radius.
  Each fit is renormalized so fitfunc(x0)=1, i.e. the reconstructed profile passes
  exactly through norms at x0. With x0=cell_d0.x and norms=cell_d0 values, the
  reconstruction therefore starts exactly at the cell_d0 (downstream) state.
  The fit window is truncated at the rarefaction arrival (truncate_at_rarefaction):
  the BPL describes the smooth downstream decay, not the rarefaction cliff.
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
    after = np.flatnonzero(sd[ish[0]:] == 0)
    start = ish[0] + after[0] + 1 if len(after) else len(sd)
    shocked_data = cell_data.iloc[start:].copy()
  else:
    # no crossing recorded (pre-trimmed data): original selection
    shocked_data = cell_data.loc[(cell_data.Sd == 0)].copy().iloc[1:]
  if not len(shocked_data):
    raise RuntimeError('no post-shock data to fit')
  # fit only the smooth downstream decay, not the rarefaction cliff
  shocked_data = truncate_at_rarefaction(shocked_data)
  # subsample long histories: ~200 points constrain the smooth profile equally
  # well and cut the curve_fit cost (numeric jacobian scales with N points)
  if len(shocked_data) > 200:
    idx = np.unique(np.linspace(0, len(shocked_data)-1, 200).astype(int))
    shocked_data = shocked_data.iloc[idx]
  for name, norm in zip(vars, norms):
    x_ = shocked_data.x.to_numpy() / x0
    val = get_variable(shocked_data, name, env)/norm
    popt = get_fitting_smoothBPL_new(x_, val, cleanData=cleanData, beta=None)
    popt = np.asarray(popt, dtype=float)
    popt[0] /= smooth_bpl_apy(1., *popt)   # anchor: fitfunc(x0) = 1 (free amplitude A)
    popts.append(popt)
  return popts

# bump when the fitting code/conventions change, to invalidate stale caches
_FIT_CACHE_VERSION = 1

def load_or_fit_celldata(cell_data, vars, norms, env, x0, cleanData=False,
    key=None, k=None):
  '''
  Disk-cached wrapper around fit_celldata: curve_fit (x3 per cell) dominates the
  cost of get_cell/shell_nuFnu, and the fit shape depends only on the cell history
  and the anchor x0 (norms and env cancel in the fitfunc(x0)=1 renormalization, and
  u_scale/alpha/zeta rescale *after* fitting), so popts are safe to cache per
  (key, k, cleanData). Cache lives next to the cell data at
  results/{key}/cells/{k:04d}_fit.npz. key/k None => no caching (just fit).
  '''
  if key is None or k is None:
    return fit_celldata(cell_data, vars, norms, env, x0=x0, cleanData=cleanData)
  path = get_dirpath(key) + f'cells/{k:04d}_fit.npz'
  if os.path.isfile(path):
    try:
      d = np.load(path)
      if int(d['version']) == _FIT_CACHE_VERSION and bool(d['cleanData']) == bool(cleanData) \
          and np.isclose(float(d['x0']), float(x0)):
        return [d['popt_rho'], d['popt_lfac'], d['popt_p']]
    except Exception:
      pass   # unreadable/old cache => refit and overwrite
  popts = fit_celldata(cell_data, vars, norms, env, x0=x0, cleanData=cleanData)
  try:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, popt_rho=popts[0], popt_lfac=popts[1], popt_p=popts[2],
             x0=float(x0), cleanData=bool(cleanData), version=_FIT_CACHE_VERSION)
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

def worldline_from_cooling(tt_edges, cell_d0, env, popts, R0=None, Tobs_max=None,
    fitfunc=smooth_bpl_apy):
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
  events = None
  if Tobs_max is not None:
    t0_cell = cell_d0.t
    def reach_window(tt, y):
      # onset observer time of emission at (R=y[0], elapsed lab time y[2]);
      # matches derive_obsTimes: Ton = (1+z)*(t_sim + t0 - R/c), r in units c
      return (1. + env.z)*(t0_cell + y[2] + env.t0 - y[0]/c_) - Tobs_max
    reach_window.terminal = True
    reach_window.direction = 1.
    events = reach_window

  # a few (non-skipped) cells have a rho fit decaying to ~0 at large R, so the
  # comoving cooling time tc1 = 1/syn(R) diverges and derivs hits 0*inf / inf,
  # briefly feeding a NaN radius through the fit's log during intermediate steps.
  # This is benign (the window event terminates the worldline at a finite R and
  # the returned edges are finite), so silence those known warnings here. Proper
  # handling of the outer edge cells needs the rarefaction-wave physics (TODO).
  with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
    sol = solve_ivp(derivs, (0., tt_edges[-1]), [R0, 0., 0.],
                    method='DOP853', dense_output=True, events=events)
  # truncate tt to the integrated range if the observer window was reached
  # (tt_edges[0] = 0 always survives, so >= 2 edges remain)
  if events is not None and len(sol.t_events[0]) and sol.t[-1] > 0.:
    tt_edges = np.append(tt_edges[tt_edges < sol.t[-1]], sol.t[-1])
  R_edges, tp_edges, t_edges = sol.sol(tt_edges)
  return t_edges, tp_edges, R_edges, tt_edges

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
  and reconstruct_cell so the fit is sampled consistently. Returns gmin_edges, gmax_edges.
  '''

  if R0 is None:
    R0 = cell_d0.x * c_
  rho0 = get_variable(cell_d0, 'rho', env)

  x   = R_edges / R0
  rho = rho0 * fitfunc(x, *popt_rho)
  dtt = np.diff(tt_edges)
  adiab = (rho[1:] / rho[:-1])**(1./3.)   # (V'_{j+1}/V'_j)^(-1/3) per step

  gmin0 = get_variable(cell_d0, 'gma_m', env)
  gmax0 = get_variable(cell_d0, 'gma_M', env)

  def evolve(g0):
    g = np.empty(len(R_edges))
    g[0] = g0
    for j in range(len(dtt)):
      g_syn  = g[j] / (1. + g[j] * dtt[j])  # synchrotron
      g[j+1] = g_syn * adiab[j]             # then adiabatic
    return g
  gmin_edges = evolve(gmin0)
  gmax_edges = evolve(gmax0)
  return gmin_edges, gmax_edges

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
    key=None, k=None):
  '''
  Reconstruct cell hydrodynamics from fit, adds electron distribution
  u_scale rescales (after fitting) as if simulation was run with u1 x u_scale
  alpha, zeta: Granot (2012) hydro unit-rescaling (lengths/times x alpha,
    energy/mass x zeta => rho, p x zeta*alpha^-3); composes on top of u_scale
  r_ref: cooling-bin ratio (gmax_j/gmax_j+1 per step); larger = fewer steps
  Tmax: if set, caps the worldline at the observer window Tobs_max = Ts + Tmax*T0
    (obs_arrays uses T in [1, Tmax+1]); avoids the tt->R runaway to unphysical
    radii. None keeps the full (uncapped) worldline. See worldline_from_cooling.
  key, k: if given, the (expensive) hydro fit is disk-cached per cell; see
    load_or_fit_celldata. None fits every call (no cache).
  '''

  # add copy and rescaling to env
  cell_d0 = cell_init.copy()
  vars = ['rho', 'lfac', 'p']
  norms = [get_variable(cell_d0, name, env_in) for name in vars]

  # fit hydrodynamics, anchored at the cell_d0 (downstream) state so the
  # reconstruction starts exactly at cell_d0 (disk-cached per cell when key/k set)
  try:
    popts = load_or_fit_celldata(cell_data, vars, norms, env_in, cell_d0.x,
                                 cleanData=cleanData, key=key, k=k)
  except RuntimeError:
    print('Fit failed on cell ', cell_d0.i)
    return False, env_in
  popt_rho, popt_lfac, popt_p = popts

  # reject cells whose fitted Lorentz profile is non-physical (Gamma rising with
  # radius): a coasting/decelerating shell has non-increasing Gamma, so a rising
  # fit (outer edge cells with short/noisy post-shock histories) is spurious. Such
  # fits send lfac->inf / vx->1 at large R, evade the worldline cap (their onset
  # time saturates, so Ton never reaches the window) and inject huge spurious flux.
  # Drop them like a failed fit. Checked on the scale-free normalized fit shape
  # (=1 at the anchor x0), so it is independent of u_scale/alpha/zeta rescaling.
  x_chk = np.logspace(0., 3., 40)
  lfac_shape = smooth_bpl_apy(x_chk, *popt_lfac)
  if (not np.all(np.isfinite(lfac_shape))) or np.any(lfac_shape <= 0.) \
      or lfac_shape[-1] > 1.05:
    print(f'Non-physical Gamma(R) fit (rising Lorentz factor) on cell '
          f'{int(cell_d0.i)}; skipping')
    return False, env_in

  # rescale cell_d0 and env (s = u_scale, holding a_u, f, chi fixed),
  # see rescale_shocked_data for the conventions
  env = rescale_proper_velocities(u_scale, env_in)
  if u_scale != 1.:
    cell_d0 = rescale_shocked_data(cell_d0, env_in, env)

  # Granot hydro unit-rescaling on top: env carries rho,p x zeta*alpha^-3
  # (via rhoscale) and lengths/times x alpha; the anchor carries t,x,dx x alpha
  if alpha != 1. or zeta != 1.:
    env = rescale_hydro(alpha, zeta, env)
    if alpha != 1.:
      cell_d0 = rescale_hydro_data(cell_d0, alpha)

  # generate time bins
  # the comoving times tp and tt start at 0 when cell is shocked
  tt_edges = generate_timebins(cell_d0, env, popt_lfac, 1., end_cond='gmax', r_ref=r_ref)

  # compute worldline, starting from the cell_d0 radius and lab time.
  # cap at the observer window (Tobs_max) so tt->R does not run away to
  # unphysical radii; tt_edges comes back truncated to the integrated range.
  R0 = cell_d0.x * c_
  t0 = cell_d0.t
  Tobs_max = (env.Ts + Tmax*env.T0) if Tmax is not None else None
  t_edges, tp_edges, R_edges, tt_edges = worldline_from_cooling(
      tt_edges, cell_d0, env, popts, R0=R0, Tobs_max=Tobs_max)
  t_arr = t_edges[:-1] + t0
  dt_arr = np.diff(t_edges)
  tp_arr = tp_edges[:-1]
  dtp_arr = np.diff(tp_edges)
  tt_arr = tt_edges[:-1]
  dtt_arr = np.diff(tt_edges)
  R_arr = R_edges[:-1]

  # reconstruct hydro
  i, x, dx, rho, vx, lfac, p, trac = reconstruct_cell(R_arr, env, cell_d0, popts)
  gmin_edges, gmax_edges = evolve_gma_bounds(R_edges, tt_edges, cell_d0, env, popt_rho, R0=R0)
  gmin, gmax = gmin_edges[:-1],gmax_edges[:-1]

  # create dataframe
  keys = ['t', 'dt', 'tp', 'dtp', 'tt', 'dtt', 'i', 'x', 'dx', 'rho', 'vx', 'lfac', 'p', 'trac', 'gmin', 'gmax']
  vals = [t_arr, dt_arr, tp_arr, dtp_arr, tt_arr, dtt_arr, i, x, dx, rho, vx, lfac, p, trac, gmin, gmax]
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

def get_Fnu_array_cell_evolving(nuobs, Tobs, cell, env, Ng=20, norm=True, width_tol=1.1):
  '''
  Same as get_Fnu_cell_evolving but returns array of Fnus per step
  '''
  cell0 = cell.iloc[0]
  K0 = norm_plaw_distrib(cell0.gmin, cell0.gmax, env.psyn)
  Tarr = np.atleast_1d(np.asarray(Tobs, dtype=float))
  Fnu = np.zeros((len(cell), Tarr.size, np.size(nuobs)))
  for j in range(len(cell)):
    step = cell.iloc[j]
    Ton = get_variable(step, 'obsT', env)[0]               # onset (tT=1)
    # only Tarr >= Ton receives flux: slice instead of computing + masking
    iT0 = np.searchsorted(Tarr, Ton)
    if iT0 >= Tarr.size:
      continue                                             # onset after window
    Fnu[j][iT0:] += get_Fnu_step(nuobs, Tarr[iT0:], step, K0, env, Ng, norm, width_tol)
  return Fnu if np.ndim(Tobs) > 0 else Fnu[:,0,:]

def get_Fnu_cell_evolving(nuobs, Tobs, cell, env, Ng=20, norm=True, width_tol=1.1):
  '''
  F_nu(Tobs) of an evolving cell, summed over its cooling steps.
  'cell' is the dataframe from generate_cell_withDistrib (already binned in
  cooling time). K0 is fixed by the initial (injection) distribution bounds.

  Returns shape (len(Tobs), len(nuobs)) for an array Tobs, (len(nuobs),) for a scalar.
  '''
  cell0 = cell.iloc[0]
  K0 = norm_plaw_distrib(cell0.gmin, cell0.gmax, env.psyn)
  Tarr = np.atleast_1d(np.asarray(Tobs, dtype=float))
  Fnu = np.zeros((Tarr.size, np.size(nuobs)))
  for j in range(len(cell)):
    step = cell.iloc[j]
    Ton = get_variable(step, 'obsT', env)[0]               # onset (tT=1)
    # only Tarr >= Ton receives flux: slice instead of computing + masking
    iT0 = np.searchsorted(Tarr, Ton)
    if iT0 >= Tarr.size:
      continue                                             # onset after window
    Fnu[iT0:] += get_Fnu_step(nuobs, Tarr[iT0:], step, K0, env, Ng, norm, width_tol)
  return Fnu if np.ndim(Tobs) > 0 else Fnu[0]

def get_Fnu_cell_instant(nuobs, Tobs, cell, env, Ng=20, norm=True, width_tol=1.1):
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

def cell_radiated_energy(cell, env, Ng=20, width_tol=1.1, tnu=None):
  '''
  Frame-independent total comoving radiated energy of a cell, summed over its
  cooling steps:
    E' = sum_j  nu'_B,j * V3p_j * int get_epnu_j(tnu) dtnu
  '''
  if tnu is None:
    gmax0 = cell.iloc[0].gmax
    tnu = np.logspace(0., 2.*np.log10(gmax0) + 1., 1000)   # nu'/nu'_B, covers up to ~gmax^2
  cell0 = cell.iloc[0]
  K0 = norm_plaw_distrib(cell0.gmin, cell0.gmax, env.psyn)
  E = 0.
  for j in range(len(cell)):
    step = cell.iloc[j]
    nupB = get_variable(step, 'nup_B', env)
    V3p  = get_variable(step, 'V3p', env)
    epnu = get_epnu(tnu, step, K0, env, Ng, width_tol)
    E += np.trapezoid(epnu, tnu) * nupB * V3p
  return E

def total_radiated_energy(nF, nub, Tb, env=None):       # nF = nu*F_nu, shape (T, nu)
    Enu = np.trapezoid(nF, np.log(nub), axis=1)   # ∫νFν dlnν = ∫Fν dν  per T
    E = np.trapezoid(Enu, Tb)
    if env is not None:
      E *= env.nu0F0*env.T0
    return E

def get_cell_nuFnu(key, k, func_Fnu=get_Fnu_cell_evolving,
    u_scale=1., alpha=1., zeta=1., cleanData=False, r_ref=1.2,
    Tmax=5, NT=500, lognu_min=-2.5, lognu_max=2.5, Nnu=400, **kwargs):
  '''
  The whole chain to obtain nu Fnu.
  With u_scale != 1 the env is rescaled (as if u1 -> u1*u_scale); the observer
  arrays nuobs, Tobs are built from that rescaled env (its nu0, Ts, T0), so the
  whole pipeline stays self-consistent in observer space.
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
  cell_dist, env = generate_cell_withDistrib(cell_data, cell_d0, env,
                  u_scale=u_scale, alpha=alpha, zeta=zeta, cleanData=cleanData,
                  r_ref=r_ref, Tmax=Tmax, key=key, k=k)
  # scale the observer grids with the rescaled env (matches obs_arrays)
  nuobs = nub * env.nu0
  Tobs = env.Ts + (T - 1) * env.T0
  if cell_dist is False:
    return nuobs, Tobs, env, None
  nuFnu = get_nuFnu(func_Fnu, nuobs, Tobs, cell_dist, env, **kwargs)
  return nuobs, Tobs, env, nuFnu

def check_extracted_cells(key):
  '''
  Return a sorted array of cell ids k already extracted with extract_data_cells
  (i.e. cells with a saved results/{key}/cells/{k:04d}.csv file)
  '''
  cells_dir = get_dirpath(key) + 'cells/'
  if not os.path.isdir(cells_dir):
    return np.array([], dtype=int)
  ks = [int(os.path.splitext(os.path.basename(f))[0])
        for f in glob.glob(cells_dir + '*.csv')]
  return np.array(sorted(ks), dtype=int)

def get_shell_nuFnu(key, z, u_scale=1., alpha=1., zeta=1., klist=None,
    VFC=False, analytic=False, cleanData=False, r_ref=1.2,
    Tmax=5, NT=500, lognu_min=-2.5, lognu_max=2.5, Nnu=400, **kwargs):
  '''
  Computes total nu F_nu from shell z of simulation key, summed over its cells.
  With u_scale != 1 the env is rescaled (as if u1 -> u1*u_scale); the observer
  grids nuobs, Tobs are built from that rescaled env (its nu0, Ts, T0), so the
  whole pipeline stays self-consistent in observer space (as get_cell_nuFnu).
  klist restricts the sum to a subset of cell ids (defaults to all of shell z).
  VFC forces all contributions from a cell to be emitted at same lab frame time
  analytic uses the 'analytic' cell for calculation (Gma = cst, rho ~ R^-2, B ~ R^-1)
  NB: cells shocked at R/R0 > ~Tmax arrive after the Tobs window and contribute
  nothing there; take Tmax above env.RfRS0 (or RfFS0) to capture the full shell.
  '''

  nub, T, env = obs_arrays(key, normed=True, Tmax=Tmax, NT=NT,
      lognu_min=lognu_min, lognu_max=lognu_max, Nnu=Nnu)
  env_rs = rescale_proper_velocities(u_scale, env) if u_scale != 1. else env
  if alpha != 1. or zeta != 1.:
    env_rs = rescale_hydro(alpha, zeta, env_rs)
  nuobs = nub * env_rs.nu0
  Tobs = env_rs.Ts + (T - 1) * env_rs.T0
  nuFnu_shell = np.zeros((len(Tobs), len(nuobs)))
  sh_data = open_rundata(key, z)
  sh_data = cellsBehindShock_fromData(sh_data)
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

  # loop over cells
  skipped = []
  for k in klist:
    # open data, generate distribution
    cell_data = open_celldata(key, k)
    if cell_data is False:
      skipped.append(k)
      continue
    sel = sh_data.loc[sh_data.i==cell_data.iloc[0].i]
    if not len(sel):
      skipped.append(k)
      continue
    cell, cell_env = generate_cell_withDistrib(cell_data, sel.iloc[0], env,
                    u_scale=u_scale, alpha=alpha, zeta=zeta, cleanData=cleanData,
                    r_ref=r_ref, Tmax=Tmax, key=key, k=k)
    if cell is False:
      skipped.append(k)
      continue
    if analytic: cell = generate_cell_constLfac(cell, cell_env)
    nuFnu_shell += get_nuFnu(func_Fnu, nuobs, Tobs, cell, cell_env, **kwargs)
  if skipped:
    print(f'get_shell_nuFnu: skipped {len(skipped)} cells (no data or failed fit): {skipped}')

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


def plot_spectrum(logT, Tb, nub, nF, ax_in=None, **kwargs):
  iT = np.searchsorted(Tb, 10**logT)
  sp = nF[iT]
  if ax_in is None:
    fig, ax = plt.subplots()
    ax.set_xlabel(nu_label)
    ax.set_ylabel(nF_label)
  else:
    ax = ax_in
  ax.loglog(nub, sp, **kwargs)
  
def plot_lightcurve(lognu, Tb, nub, nF, ax_in=None, **kwargs):
  inu = np.searchsorted(nub, 10**lognu)
  lc = nF[:,inu]
  if ax_in is None:
    fig, ax = plt.subplots()
    ax.set_xlabel(T_label)
    ax.set_ylabel(nF_label)
  else:
    ax = ax_in
  ax.semilogy(Tb+1, lc, **kwargs)

def compare_spectra(logT, Tb, nub, nF_arr,
    colors=None, linestyles=None, labels=None):
  fig, ax = plt.subplots()
  n = len(nF_arr)
  colors     = colors     or [None]*n
  linestyles = linestyles or [None]*n
  labels     = labels     or [None]*n
  for nF, c, ls, lab in zip(nF_arr, colors, linestyles, labels):
    plot_spectrum(logT, Tb, nub, nF, ax_in=ax,
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