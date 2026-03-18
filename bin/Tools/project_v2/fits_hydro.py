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
import pandas as pd

from phys_constants import c_
from phys_functions import smooth_bpl, smooth_bpl0, init_slope, logslope, derive_velocity
from IO import get_variable, get_vars, get_fitsfile, get_runatts
from environment import MyEnv


def cellsBehindShock_fromData(data, N_in=None, i_set=1000, dfindex=True):
  '''
  Create a dataframe with the expected state of the cells behind the shock
    right when the shock reaches them, assuming instantaneous establishing of downstream values
    N_in controls the shell division in cells
  '''

  # get fit
  popt_lfac2, popt_ShSt, _ = get_hydrofits_shell(data, i_set, dfindex)

  env = MyEnv(data.attrs['key'])
  fastshell = (data.iloc[0].trac < 1.5)
  t_max = data.iloc[-1].t

  t_hit, cells_i, R, dx, rho, vx, lfac, p, trac = reconstruct_data(t_max, env, fastshell, popt_lfac2, popt_ShSt, N_in)

  # create dataframe
  keys = ['t', 'i', 'x', 'dx', 'rho', 'vx', 'lfac', 'p', 'trac']
  vals = [t_hit, cells_i, R, dx, rho, vx, lfac, p, trac]
  dic = {key:val for key, val in zip(keys, vals)}
  out = pd.DataFrame.from_dict(dic)
  for key in data.attrs.keys():
    out.attrs[key] = data.attrs[key]
  return out

def cellsBehindShock_fromFit(key, front, N_in=None):
  '''
  same as _fromData but using the extracted fitting values
  '''
  # get parameters
  env = MyEnv(key)
  z, fastshell = (1, False) if front == 'FS' else (4, True)
  fname = get_fitsfile(key, z)
  Tf, t_max, *rest = np.loadtxt(fname)
  popts = [np.array(rest[i:i+3]) for i in range(0, 15, 3)]
  popt_lfac2, popt_ShSt, popt_nu, popt_L, popt_xi = popts
  (Tf, t_max,
    lfac2_inf, lfac2_alph, lfac2_s,
    ShSt_inf, ShSt_alph, ShSt_s,
    nu_inf, nu_alph, nu_s,
    L_inf, L_alph, L_s,
    k, xi_sat, xi_s) = np.loadtxt(fname)
  popt_lfac2 = [lfac2_inf, lfac2_alph, lfac2_s]
  popt_ShSt = [ShSt_inf, ShSt_alph, ShSt_s]

  t_hit, cells_i, R, dx, rho, vx, lfac, p, trac = reconstruct_data(t_max, env, fastshell, popt_lfac2, popt_ShSt, N_in)

  # create dataframe
  keys = ['t', 'i', 'x', 'dx', 'rho', 'vx', 'lfac', 'p', 'trac']
  vals = [t_hit, cells_i, R, dx, rho, vx, lfac, p, trac]
  dic = {key:val for key, val in zip(keys, vals)}
  out = pd.DataFrame.from_dict(dic)
  # add attributes
  mode, runname, rhoNorm, geometry = get_runatts(key)
  out.attrs['key'] = key
  out.attrs['mode']    = mode
  out.attrs['runname'] = runname 
  out.attrs['rhoNorm'] = rhoNorm
  out.attrs['geometry'] = geometry
  return out
  

def reconstruct_data(t_max, env, fastshell, popt_lfac2, popt_ShSt, N_in=None):
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
  sol = propagation_from_fit(t_max, env.R0, lfac_sh0, popt_lfac2)
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
  Gamma2 = env.lfac0**2 * smooth_bpl0(x, *popt_lfac2)
  lfac = np.sqrt(Gamma2)
  vx = np.sqrt(Gamma2 - 1.) / lfac 
  rho0, ShSt0, lfac_i0 = (env.rho4, env.lfac34-1., env.lfac4) if fastshell \
      else (env.rho1, env.lfac21-1., env.lfac1)
  Gma_ud = 1. + ShSt0 * smooth_bpl0(x, *popt_ShSt)
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

def replace_early_withfit(data, i_set=1000):
  '''
  Replace early time data with fitted function
  '''

  out = data.copy()
  # values at R0
  env = MyEnv(out.attrs['key'])
  trac = out.iloc[0].trac
  rho0, e0 = (env.rho4, env.eint3p) if (trac < 1.5) else (env.rho1, env.eint2p)
  lfac0, p0 = env.lfac0, env.p_sh
  ShSt0 = env.lfac34 - 1 if (trac < 1.5) else env.lfac21 -1

  # fits for Gamma_d^2 and Gamma_ud - 1
  popt_lfac2, popt_ShSt, _ = get_hydrofits_shell(out, i_set)
  x_lf, m, s_lf = popt_lfac2
  x_sh, n, s_sh = popt_ShSt
  x = out.loc[:i_set].x.to_numpy(copy=True) * c_ / env.R0
  lfac2 = lfac0**2 * smooth_bpl(x, x_lf, m, 0, s_lf)
  Gma_ud = 1. + ShSt0 * smooth_bpl(x, x_sh, n, 0, s_sh)


  # hydro from fits
  rho = 4 * Gma_ud * rho0 / (x * x)
  ei = (Gma_ud - 1) * rho * c_*c_
  adb = (4*Gma_ud + 1)/(3*Gma_ud)
  p = (adb - 1) * ei

  # eventual discontinuity at i_set
  j = out.index.get_loc(i_set)
  step_set = out.loc[i_set]
  rho *= step_set.rho / rho[j]
  p *= step_set.p / p[j]
  lfac2 *= get_variable(step_set, 'lfac', env)**2 / lfac2[j]

  lfac_arr = np.sqrt(lfac2)
  v = np.sqrt(lfac2 - 1.) / lfac_arr  # stable; store lfac separately to avoid recover-from-beta loss

  # replace
  out.loc[:i_set, 'rho'] = rho
  out.loc[:i_set, 'vx'] = v
  out.loc[:i_set, 'lfac'] = lfac_arr
  out.loc[:i_set, 'p'] = p
  return out

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

def get_hydrofits_shell(data, i_set=1000, dfindex=True):
  '''
  Returns fitting parameters for the Lorentz factor and the shock strength
  data is a dataframe with the history of hydrodynamics downstream
  i_set code iterations if dfindex = True or number of steps if False
  '''

  env = MyEnv(data.attrs['key'])
  trac = data.iloc[0].trac
  lfac_ud0 = env.lfac34 if (trac < 1.5) else env.lfac21
  ShSt0 = lfac_ud0 - 1.
  b_ud0 = np.sqrt(1. - 1/lfac_ud0**2)
  # get variables and normalize
  index = data.index.to_numpy(copy=True)
  if dfindex:
    i_ = np.where(index>=i_set)[0][0]
    i_set = i_

  r, lfac, ShSt, b_ud = get_vars(data, ['x', 'lfac', 'ShSt', 'vx'], env)
  r *= c_/env.R0
  lfac /= env.lfac0
  ShSt /= ShSt0
  b_ud /= env.beta
  lfac2 = lfac*lfac
  popt_out = []
  r_ = r[i_set:]
  for val in [lfac2, ShSt, b_ud]:
    a0 = logslope(1., 1., r[i_set], val[i_set])
    # guard against near-zero slope (degenerate bounds width)
    a0 = a0 if abs(a0) > 1e-3 else -1e-2
    val_ = val[i_set:]
    if a0 < 0.:
      p0 = [np.clip(val_[-1], 1e-3, 0.999), a0, 2.]
      bounds = ([1e-3, min(0.1*a0, 10*a0), 1.], [0.999, max(0.1*a0, 10*a0), 200])
    else:
      p0 = [max(val_[-1], 1.001), a0, 2.]
      bounds = ([1.001, min(0.1*a0, 10*a0), 1.], [20., max(0.1*a0, 10*a0), 200])
    popt = get_fitting_smoothBPL(r_, val_, p0=p0, bounds=bounds)
    popt_out.append(popt)
  
  return popt_out

def get_radfits_shell(data, i_set=100):
  '''
  Fits comoving peak frequency and bolometric luminosity
  data should be already reconstructed after hydro fit
  '''

  env = MyEnv(data.attrs['key'])
  trac = data.iloc[0].trac
  nu0, L0 = (env.nu0p, env.Lbolp) if (trac < 1.5) else (env.nu0pFS, env.LbolpFS)
  r = data.x.to_numpy() * c_ /env.R0
  popt_out = []
  for var, norm in zip(['nup_m2', 'Lbol'], [nu0, L0]):
    val = get_variable(data, var, env)/norm
    if 'nu' in var:
      val *= r
    a0 = logslope(1., 1., r[i_set], val[i_set])
    a0 = a0 if abs(a0) > 1e-3 else -1e-2
    if a0 < 0.:
      p0 = [np.clip(val[-1], 1e-3, 0.999), a0, 2.]
      bounds = ([1e-3, min(0.1*a0, 10*a0), 1.], [0.999, max(0.1*a0, 10*a0), 200])
    else:
      p0 = [max(val[-1], 1.001), a0, 2.]
      bounds = ([1.001, min(0.1*a0, 10*a0), 1.], [20., max(0.1*a0, 10*a0), 200])
    popt = get_fitting_smoothBPL(r, val, p0=p0, bounds=bounds)
    popt_out.append(popt)
  return popt_out


def get_fitting_smoothBPL(x, array, beta=None, **kwargs):
  '''
  Return parameters that fit a smoothed bpl to values in an array
  x is the normalized radius R/R_0, array(1) = 1
  kwargs are arguments to pass to curve_fit (p0, bounds, etc.)
  '''

  if beta is not None:
    def fitfunc(x, X_sph, alpha, s):
      return smooth_bpl(x, X_sph, alpha, beta, s)
  else:
    def fitfunc(x, X_sph, alpha, s):
      return smooth_bpl0(x, X_sph, alpha, s)
    
  popt, pcov = curve_fit(fitfunc, x, array, **kwargs)
  return popt

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
  

def propagation_from_fit(t_max, R0, lfac0, popt_lfac2):
  '''
  Constructs the (t, R) graph knowing \Gamma^2(R) over time t_max
    t_max is counted starting from t0
  '''

  def func_beta(R):
    x = R/R0
    Gamma2 = lfac0**2 * smooth_bpl0(x, *popt_lfac2)
    return np.sqrt(Gamma2 - 1.) / np.sqrt(Gamma2)  # stable: avoids 1 - 1/Gamma^2 cancellation
  
  def dRdt(t, R):
    return c_ * func_beta(R)
  
  t_span = (0., t_max)
  
  sol = solve_ivp(dRdt, t_span, [R0], method='DOP853', dense_output=True)
  return sol


