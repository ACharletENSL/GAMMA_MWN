# -*- coding: utf-8 -*-
# @Author: acharlet

'''
Fitting procedure and associated functions for the hydro quantities
  main objective: parameters for Gamma_d and Gamma_ud - 1

TODO: delete replace_early_withfit by adding some condition in replace_withfit
'''

import numpy as np
from scipy.optimize import curve_fit, brentq
from scipy.interpolate import CubicSpline
from scipy.integrate import quad, solve_ivp
from scipy.signal import savgol_filter
import pandas as pd

from phys_constants import c_
from phys_functions import *
from IO import *
from environment import MyEnv


def fits_from_au(au, front='RS', l=4):
  '''
  Returns parameters for lfac, ShSt, nu'_m*R, L_bol from any value of a_u
  l is the length of the fitting parameters
  '''

  log_au = np.log10(au-1)
  df = open_sweep(front)
  x = df['log_aum'].to_numpy()

  groups = {
    'lfac':['A', 'x_b', 'alpha', 's'],
    'ShSt':['A', 'x_b', 'alpha', 's'],
    'nu':['A', 'x_b', 'alpha', 's'],
    'L':['A', 'x_b', 'alpha', 's'],
    'xi':['k', 'sat', 's']
  }
  popts = []
  for prefix, suffixes in groups.items():
    vals = [np.interp(log_au, x, df[f'{prefix}_{s}'].to_numpy()) for s in suffixes]
    popts.append(np.array(vals))
  return popts



def cellsBehindShock_fromData(data, N_in=None):
  '''
  Create a dataframe with the expected state of the cells behind the shock
    right when the shock reaches them, assuming instantaneous establishing of downstream values
    N_in controls the shell division in cells
  '''

  key = data.attrs['key']
  fastshell = (data.iloc[0].trac < 1.5)
  t_max = data.iloc[-1].t

  # get fit
  popt_lfac, popt_ShSt, _, _, _ = get_hydrofits_shell_new(data)
  attrs = {key:data.attrs[key] for key in data.attrs.keys()}
  df = cellsBehindShock_fromFit(key, popt_lfac, popt_ShSt, t_max,
    N_in=N_in, fastshell=fastshell, attrs=attrs)
  return df



def cellsBehindShock_fromFit(key, popt_lfac, popt_ShSt, t_max,
  N_in=None, fastshell=True, attrs={}):
  '''
  same as _fromData but using the extracted fitting values
  attrs is a dictionnary of attributes
  '''
  env = MyEnv(key)

  t_hit, cells_i, R, dx, rho, vx, lfac, p, trac = reconstruct_data(t_max, env, fastshell, popt_lfac, popt_ShSt, N_in)
  vx_u = np.full(t_hit.shape, (env.beta4 if fastshell else env.beta1))

  # create dataframe
  keys = ['t', 'i', 'x', 'dx', 'rho', 'vx', 'lfac', 'p', 'vx_u', 'trac']
  vals = [t_hit, cells_i, R, dx, rho, vx, lfac, p, vx_u, trac]
  dic = {key:val for key, val in zip(keys, vals)}
  out = pd.DataFrame.from_dict(dic)
  if len(attrs) > 0:
    for key in attrs.keys():
      out.attrs[key] = attrs[key]
  return out
  

def reconstruct_data(t_max, env, fastshell, popt_lfac, popt_ShSt, N_in=None):
  '''
  Reconstruct data from environment and fits
  '''
  # initial values from analytics
  D0, N, beta, g = (-env.D04, env.Nsh4, env.beta4, env.gRS) if fastshell \
    else (env.D01, env.Nsh1, env.beta1, env.gFS)
  if N_in is not None:
    N = N_in
  lfac_sh0 = env.lfac0/g
  dx0 = D0/(N*env.R0) # < 0 for fast shell
  init_rads = env.R0 * np.array([1. + k*dx0 for k in range(N)])

  # get the (t, R_sh) worldline
  sol = propagation_from_fit(t_max, env.R0, lfac_sh0, popt_lfac)
  t_vals = np.linspace(0, t_max, 10*N)    # N*10 for good sampling
  R_vals = sol.sol(t_vals)[0]

  # find the interception of the cells by the shock
  G_vals = R_vals - beta * c_ * t_vals
  if np.all(np.diff(G_vals) < 0):
      # reverse order for interpolation
      G_sorted = G_vals[::-1]
      t_sorted = t_vals[::-1]
  elif np.all(np.diff(G_vals) > 0):
      G_sorted = G_vals
      t_sorted = t_vals
  else:
      raise RuntimeError("G(t) not monotonic")
  t_of_G = CubicSpline(G_sorted, t_sorted)
  t_hit = t_of_G(init_rads)
  t_hit[0] = 0.
  R_hit = init_rads + t_hit * beta * c_

  # generate hydro from fit
  x = R_hit/env.R0
  lfac = env.lfac0 * smooth_bpl0_apy(x, *popt_lfac)
  Gamma2 = lfac**2
  vx = np.sqrt(Gamma2 - 1.) / lfac 
  rho0, ShSt0, lfac_i0 = (env.rho4, env.lfac34-1., env.lfac4) if fastshell \
      else (env.rho1, env.lfac21-1., env.lfac1)
  Gma_ud = 1. + ShSt0 * smooth_bpl0_apy(x, *popt_ShSt)
  rho = 4 * Gma_ud * rho0 / (x * x)
  x_k0 = init_rads/env.R0
  dx = np.abs(dx0) * x_k0**2 * (lfac_i0/lfac) / (4*Gma_ud)
  ei = (Gma_ud - 1) * rho * c_*c_
  adb = (4*Gma_ud + 1)/(3*Gma_ud)
  p = (adb - 1) * ei
  # rho *= c_*c_
  # p = np.sqrt(rho*rho + ei*ei +6*ei*rho) + ei - rho

  # normalize to code units
  R = R_hit/c_
  dx *= env.R0/c_  # dx0 was in units R0
  rho /= env.rhoscale
  p /= env.rhoscale * c_**2

  # add the passive tracer and cell index (useful in some calcs)
  trac = np.ones(t_hit.shape)
  cells_i = np.arange(N) + env.Next
  if not fastshell:   # shell 1
    trac *= 2.
    cells_i += env.Nsh4

  return t_hit, cells_i, R, dx, rho, vx, lfac, p, trac


def extrapolate_early(data_in, env, nsamp=10, it_s=500, dfindex=True):
  '''
  Extrapolate early data for shock-crossing region

  Physical context: During shock crossing, the cell takes several timesteps
  to settle to downstream values. We extrapolate the settled downstream behavior
  backwards to reconstruct the physical state at earlier times.

  Method: Fit power laws to settled data (it >= it_s) and extrapolate backwards.
  IMPORTANT: Lab-frame time (t) is NOT modified - only physical state variables.
  This preserves causality while correcting for numerical shock-crossing artifacts.

  dfindex: if True, it_s is dataframe index; if False, it_s is array position
  '''

  data = data_in.copy()
  data['lfac'] = get_variable(data, 'lfac', env)

  if dfindex:
    indices = data.index[data.index >= it_s].astype(int).to_list()[:nsamp]
    toreplace = data.loc[data.index < it_s]
    sample = data.loc[indices]
  else:
    toreplace = data.iloc[:it_s]
    sample = data.iloc[it_s:it_s+nsamp]

  # Use the boundary point (first settled point) as reference
  ref = sample.iloc[0]
  x_ref = ref.x

  # Fit power laws to settled data (using spatial coordinate x)
  x_sample = sample.x.to_numpy()

  # Extrapolate each physical variable
  for name in ['rho', 'lfac', 'p']:
    val = sample[name].to_numpy()
    # Fit in log-log space: log(val) vs log(x)
    log_x = np.log(x_sample / x_ref)
    log_val = np.log(val / val[0])
    slope = np.polyfit(log_x, log_val, 1)[0]

    # Extrapolate backwards using spatial scaling
    for i, row in toreplace.iterrows():
      x_ratio = row.x / x_ref
      data.loc[i, name] = ref[name] * (x_ratio ** slope)

  # Recalculate vx from lfac for consistency
  # Use proper relativistic formula: vx = sqrt(1 - 1/lfac^2)
  for i, row in toreplace.iterrows():
    lfac = data.loc[i, 'lfac']
    data.loc[i, 'vx'] = np.sqrt(1. - lfac**-2) if lfac > 1 else 0.

  return data

def find_breakpoint(x, y, window=7, polyorder=3, offset=5):
  """
  Detect the transition between fast-changing and slow power-law phase.
  """
  window = max(window, polyorder + 2)  # savgol constraint
  if window % 2 == 0:
      window += 1  # savgol requires odd window
  
  smoothed = savgol_filter(y, window, polyorder)
  dy = np.gradient(smoothed, x)
  
  # First zero crossing of derivative = extremum
  sign_changes = np.where(np.diff(np.sign(dy)))[0]
  if len(sign_changes) == 0:
      return 0  # monotone, no break found
  
  # offset because np.diff reduces size by 1 + settling time
  breaks = [sign_changes[0] + offset, sign_changes[2] + offset]
  return breaks

def get_hydrofits_shell_new(data):
  '''
  New version, accomodating for the new fitting function
  '''
  env = MyEnv(data.attrs['key'])
  trac = data.iloc[0].trac
  ShSt0, nu0, L0 = (
    (env.lfac34 - 1, env.nu0p, env.Lbolp) if (trac < 1.5)
    else (env.lfac21 - 1, env.nu0pFS, env.LbolpFS)
  )
  norms = [env.lfac0, ShSt0, env.beta, nu0, L0]
  names = ['lfac', 'ShSt', 'vx', 'nup_m2', 'Lbol']
  x = data.x.to_numpy() * c_/env.R0
  popt_out = []
  for name, norm in zip(names, norms):
    val = get_variable(data, name, env)/norm
    noBeta = True #if (name == 'lfac') else False
    if 'nu' in name:
      val *= x
    popt = get_fitting_smoothBPL_v2(x, val, noBeta=noBeta)
    popt_out.append(popt)
  return popt_out
  
def get_fitting_smoothBPL_v2(x, y, i_s=15, i_end=20,
    noBeta=False, initBreak=True,
    fitfuncs=(smooth_bpl_apy, smooth_bpl0_apy)):
  '''
  Fit data with a smoothly joined broken power law
  i_s: number of points for slope sampling
  i_end: number of last points to exclude from fit
  if noBeta=True, second segment is a constant
  if initBreak = True, the initial peak/dip is detected and taken out
  '''

  x_, y_ = x, y
  if initBreak:
    i_b = np.searchsorted(x, 1.3)
    # breaks = find_breakpoint(x, y)
    # i_b = breaks[1] if (breaks[1] <= i_s) else breaks[0]
    x_, y_ = x[i_b:], y[i_b:]
  
  # some smoothing before fitting
  window = 7
  polyorder = 3
  window = max(window, polyorder + 2)  # savgol constraint
  if window % 2 == 0:
    window += 1  # savgol requires odd window
  y_ = savgol_filter(y_, window, polyorder)
  
  if noBeta:
    func = fitfuncs[1]
  else:
    func = fitfuncs[0]
  
  p0, bounds = get_p0_bounds(x_, y_, i_s=i_s, noBeta=noBeta)
  x_, y_ = x_[:-i_end], y_[:-i_end]
  try:
    popt, _ = curve_fit(func, x_, y_, p0=p0, bounds=bounds)
    return popt
  except ValueError as e:
    print(f"Error occurred: {e}")
    print("p0: ", p0)
    print("lower bound: ", bounds[0])
    print("upper bounds: ", bounds[1])
    #print('ValueError with init. guesses: ', func(1.,*p0))
    return p0

def get_p0_bounds(x_, y_, i_s=15, noBeta=False):
  '''
  New version accomodating for the astropy form
  x_ and y_ start when data is settled, x_[0]~1.3
  '''
  ### priors
  # s ~ transition size in decades around x_b
  s = 0.5
  # transition should be close to 1 by construction
  # x_b0 = x_[0]
  # x_bs = x_b0**(-1/s)
  x_b0, x_bs = 1., 1.
  # slopes at x = 1 and at the end of dataset (typically x <= 10) 
  a_eff = logslope(x_[0], y_[0], x_[i_s], y_[i_s])
  y_end = y_[-20:-5].mean()
  i_b0 = np.searchsorted(x_, 2.*x_[-1]/3.)
  beta0 = 0. if noBeta else logslope(x_[i_b0], y_[i_b0], x_[-10], y_end)
  # slope at x << 1
  alpha0 = - a_eff * (1 + x_bs) - beta0 * x_bs
  # amplitude
  A0 = A_implied(x_[0], y_[0], x_b0, alpha0, beta0, s)
  # final
  p0 = [A0, x_b0, alpha0, beta0, s]

  ### bounds
  x_low, x_up = 1e-2, 100*x_[-1]
  s_low, s_up = 1e-3, 3
  a_low, a_up = alpha0 - 2., alpha0 + 2.
  b_low, b_up = beta0 - 2., beta0 + 2.
  # for amplitude, take the range of y as a factor on prior
  y_eff = x_[0]**(-a_eff) * y_[0]  # value at x = 1
  range = max(y_eff/y_end, y_end/y_eff)
  A_low, A_up = A0/range, A0*range
  bounds = ([A_low, x_low, a_low, b_low, s_low], 
              [A_up, x_up, a_up, b_up, s_up])
  
  if noBeta:
    for arr in [p0, *bounds]:
      del arr[-2]
  
  return p0, bounds

def A_implied(x_val, y_val, x_b, alpha, beta, s):
  '''
  Amplitude for smooth_bpl_apy given a datapoint 'x_val, y_val'
    and values of the other parameters 
  '''
  return y_val / smooth_bpl_apy(x_val, 1., x_b, alpha, beta, s)

def get_p0_bounds_old(x_, y_, i_s=15, noBeta=False):
  # initial guess and bounds
  a0 = logslope(x_[0], y_[0], x_[i_s], y_[i_s])
  k0 = y_[0] * x_[0]**(-a0)
  k_low, k_up = (1., 1.25*k0) if k0 > 1. else (0.75*k0, 1.)
  # a0 = a0 if abs(a0) > 1e-3 else np.sign(a0)*1e-2
  y_end = y_[-20:-5].mean()
  if a0 < 0.:
    x_b0, x_low, x_up = y_end, 0.01, 1.
  elif a0 > 0.:
    x_b0, x_low, x_up = y_end, 1., 100
  else:
    x_b0, x_low, x_up = y_end, 1.e-2, 100
  #a_low, a_up = min(0.1*a0, 10*a0), max(0.1*a0, 10*a0)
  a_low, a_up = a0 - 2., a0 + 2.

  if noBeta:
    p0 = [k0, x_b0, a0, 2.]
    bounds = ([k_low, x_low, a_low, 1.], [k_up, x_up, a_up, 200.])
  else:
    i_b0 = np.searchsorted(x_, x_[-1]/2.)
    b0 = logslope(x_[i_b0], y_[i_b0], x_[-10], y_end)
    b_low, b_up = b0 - 2., b0 + 2.
    p0 = [k0, x_b0, a0, b0, 2.]
    bounds = ([k_low, x_low, a_low, b_low, 1.],
      [k_up, x_up, a_up, b_up, 200.])
  
  return p0, bounds


def get_hydrofits_shell(data):
  '''
  Returns fitting parameters of downstream Lorentz, shock strength,
    downstream velocity, frequency, and luminosity.
  Automatically detects the break between numerical settling and hydro evolution
  '''

  env = MyEnv(data.attrs['key'])

  trac = data.iloc[0].trac
  ShSt0, nu0, L0 = (
    (env.lfac34 - 1, env.nu0p, env.Lbolp) if (trac < 1.5)
    else (env.lfac21 - 1, env.nu0pFS, env.LbolpFS)
  )
  norms = [env.lfac0, ShSt0, env.beta, nu0, L0]
  names = ['lfac', 'ShSt', 'vx', 'nup_m2', 'Lbol']
  x = data.x.to_numpy() * c_/env.R0
  popt_out = []
  for name, norm in zip(names, norms):
    val = get_variable(data, name, env)/norm
    if 'nu' in name:
      val *= x
    # select data once downstream has settled
    i_b = find_breakpoint(x, val)[0]
    x_, val_ = x[i_b:]/x[i_b], val[i_b:]
    a0 = logslope(x_[0], val_[0], x_[10], val_[10])
    a0 = a0 if abs(a0) > 1e-3 else np.sign(a0)*1e-2
    a_low, a_up = min(0.1*a0, 10*a0), max(0.1*a0, 10*a0)
    if a0 < 0.:
      p0 = [np.clip(val_[-1], 1e-3, 0.999), a0, 2.]
      bounds = ([1e-3, a_low, 1.], [0.999, a_up, 200])
    else:
      p0 = [max(val_[-1], 1.001), a0, 2.]
      bounds = ([1.001, a_low, 1.], [20., a_up, 200])
    noBeta = True
    # this doesn't work, need to rewrite method
    # if name != 'lfac':
    #   noBeta = False
    #   p0.insert(-1, 0.)
    #   bounds[0].insert(-1, -10.)
    #   bounds[1].insert(-1, 10.)
    popt = get_fitting_smoothBPL_old(x_, val_, noBeta=noBeta, p0=p0, bounds=bounds)
    popt_out.append(popt)
  return popt_out


def get_fitting_smoothBPL_old(x, array, noBeta=True, **kwargs):
  '''
  Return parameters that fit a smoothed bpl to values in an array
  x is the normalized radius R/R_0, array(1) = 1
  kwargs are arguments to pass to curve_fit (p0, bounds, etc.)
  '''

  if noBeta:
    def fitfunc(x, X_sph, alpha, s):
      return smooth_bpl0(x, X_sph, alpha, s)
  else:
    def fitfunc(x, X_sph, alpha, beta, s):
      return smooth_bpl(x, X_sph, alpha, beta, s)
    
  popt, pcov = curve_fit(fitfunc, x, array, **kwargs)
  return popt


# def get_hydrofits_shell(data, i_set=1000, dfindex=True):
#   '''
#   Returns fitting parameters for the Lorentz factor and the shock strength
#   data is a dataframe with the history of hydrodynamics downstream
#   i_set code iterations if dfindex = True or number of steps if False
#   '''

#   env = MyEnv(data.attrs['key'])
#   trac = data.iloc[0].trac
#   lfac_ud0 = env.lfac34 if (trac < 1.5) else env.lfac21
#   ShSt0 = lfac_ud0 - 1.
#   # get variables and normalize
#   index = data.index.to_numpy(copy=True)
#   if dfindex:
#     i_ = np.where(index>=i_set)[0][0]
#     i_set = i_

#   r, lfac, ShSt, vx = get_vars(data, ['x', 'lfac', 'ShSt', 'vx'], env)
#   r *= c_/env.R0
#   lfac /= env.lfac0
#   ShSt /= ShSt0
#   vx /= env.beta
#   lfac2 = lfac*lfac
#   popt_out = []
#   r_ = r[i_set:]
#   for val in [lfac2, ShSt, vx]:
#     a0 = logslope(1., 1., r[i_set], val[i_set])
#     # guard against near-zero slope (degenerate bounds width)
#     a0 = a0 if abs(a0) > 1e-3 else -1e-2
#     val_ = val[i_set:]
#     if a0 < 0.:
#       p0 = [np.clip(val_[-1], 1e-3, 0.999), a0, 2.]
#       bounds = ([1e-3, min(0.1*a0, 10*a0), 1.], [0.999, max(0.1*a0, 10*a0), 200])
#     else:
#       p0 = [max(val_[-1], 1.001), a0, 2.]
#       bounds = ([1.001, min(0.1*a0, 10*a0), 1.], [20., max(0.1*a0, 10*a0), 200])
#     popt = get_fitting_smoothBPL(r_, val_, p0=p0, bounds=bounds)
#     popt_out.append(popt)
  
#   return popt_out

# def get_radfits_shell(data):
#   '''
#   Fits comoving peak frequency and bolometric luminosity
#   data should be already reconstructed after hydro fit
#   '''

#   env = MyEnv(data.attrs['key'])
#   trac = data.iloc[0].trac
#   nu0, L0 = (env.nu0p, env.Lbolp) if (trac < 1.5) else (env.nu0pFS, env.LbolpFS)
#   x = data.x.to_numpy() * c_ /env.R0
#   popt_out = []
#   for var, norm in zip(['nup_m2', 'Lbol'], [nu0, L0]):
#     val = get_variable(data, var, env)/norm
#     if 'nu' in var:
#       val *= x
#     i_b = find_breakpoint(x, val)
#     # last 2 data points taken out to avoid
#     # a strong drop in data due to method
#     x_, val_ = x[i_b:-2]/x[i_b], val[i_b:-2]
#     a0 = logslope(x_[0], val_[0], x_[10], val_[10])
#     a0 = a0 if abs(a0) > 1e-3 else np.sign(a0)*1e-2
#     a_low, a_up = min(0.1*a0, 10*a0), max(0.1*a0, 10*a0)
#     if a0 < 0.:
#       p0 = [np.clip(val[-1], 1e-3, 0.999), a0, 2.]
#       bounds = ([1e-3, a_low, 1.], [0.999, a_up, 200])
#     else:
#       p0 = [max(val[-1], 1.001), a0, 2.]
#       bounds = ([1.001, a_low, 1.], [20., a_up, 200])
#     popt = get_fitting_smoothBPL(x_, val_, p0=p0, bounds=bounds)
#     popt_out.append(popt)
#   return popt_out



def propagation_normalized_fromfit(tau_max, popt_beta):
  '''
  Construct the normalized (tau, x) graph
    tau = (c beta_0 / R0) t
    x = R/R0
  '''

  def func_beta(x):
    return smooth_bpl0(x, *popt_beta)
  tau_span = (0., tau_max)
  sol = solve_ivp(func_beta, tau_span, [1.], method='DOP853', dense_output=True)
  return sol
  

def propagation_from_fit(t_max, R0, lfac0, popt_lfac):
  '''
  Constructs the (t, R) graph knowing \Gamma^2(R) over time t_max
    t_max is counted starting from t0
  '''

  def func_beta(R):
    x = R/R0
    Gamma = lfac0 * smooth_bpl0_apy(x, *popt_lfac)
    Gamma2 = Gamma**2
    return np.sqrt(Gamma2 - 1.) / np.sqrt(Gamma2)  # stable: avoids 1 - 1/Gamma^2 cancellation
  
  def dRdt(t, R):
    return c_ * func_beta(R)
  
  t_span = (0., t_max)
  
  sol = solve_ivp(dRdt, t_span, [R0], method='DOP853', dense_output=True)
  return sol


