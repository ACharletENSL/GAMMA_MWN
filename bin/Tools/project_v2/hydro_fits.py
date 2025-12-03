# -*- coding: utf-8 -*-
# @Author: acharlet

'''
Fits hydro quantities
'''

import numpy as np
from scipy.optimize import curve_fit, brentq
from scipy.integrate import quad

from phys_constants import c_
from phys_functions import smooth_bpl, init_slope, logslope
from IO import get_variable, get_vars
from environment import MyEnv

def replace_withfit(data, i_set=400):
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
  popt_lfac2, popt_ShSt = get_hydrofits_shell(out)
  x_lf, m, s_lf = popt_lfac2
  x_sh, n, s_sh = popt_ShSt
  x = out.loc[:i_set].x.to_numpy(copy=True) * c_ / env.R0
  lfac2 = lfac0**2 * smooth_bpl(x, x_lf, m, 0, s_lf)
  Gma_ud = 1 + ShSt0 * smooth_bpl(x, x_sh, n, 0, s_sh)

  # hydro from fits
  rho = 4 * Gma_ud * rho0 / (x * x)
  v = 1 - lfac2**-2
  ei = (Gma_ud - 1) * rho * c_*c_
  adb = (4*Gma_ud + 1)/(3*Gma_ud)
  p = (adb - 1) * ei

  # replace
  out.loc[:i_set, 'rho'] = rho/env.rhoscale
  out.loc[:i_set, 'v'] = v
  out.loc[:i_set, 'p'] = p/(env.rhoscale*c_*c_)
  return out

def extrapolate_early(data_in, env, nsamp=10, it_s=300):
  '''
  Old method of replacing early data
  '''

  data = data_in.copy()
  data['lfac'] = get_variable(data, 'lfac', env)
  indices = data.index[data.index >= it_s].astype(int).to_list()[:nsamp]
  toreplace = data.loc[data.index < it_s]
  sample = data.loc[indices]
  ref = sample.iloc[0]

  # replace rho, v, p
  x = sample.x.to_numpy()
  for name in ['rho', 'lfac', 'p']:
    val = sample[name].to_numpy()
    # get plaw with r
    a = logslope(x[0], val[0], x[-1], val[-1])
    # replace values assuming this plaw
    for i, row in toreplace.iterrows():
      dr = row.x/ref.x
      data.loc[i,name] = ref[name] * (dr**a)
  for i, row in toreplace.iterrows():
    data.loc[i, 'vx'] = 1. - (data.loc[i].lfac)**-2

  return data

def get_hydrofits_shell(data, i_set=400):
  '''
  Returns fitting parameters for the Lorentz factor and the shock strength
  data is a dataframe with the history of hydrodynamics downstream
  i_set is a number of code iterations for hydro to settle to downstream
  '''

  env = MyEnv(data.attrs['key'])
  trac = data.iloc[0].trac
  ShSt0 = env.lfac34 - 1 if (trac < 1.5) else env.lfac21 - 1
  # get variables and normalize
  index = data.index.to_numpy(copy=True)
  r, lfac, ShSt = get_vars(data, ['x', 'lfac', 'ShSt'], env)
  r *= c_/env.R0
  lfac /= env.lfac0
  ShSt /= ShSt0
  lfac2 = lfac*lfac
  m0 = logslope(1., 1., r[i_set], lfac2[i_set])
  n0 = logslope(1., 1., r[i_set], ShSt[i_set])

  # select established downstream values
  r_, lfac2_, ShSt_ = r[i_set:], lfac2[i_set:], ShSt[i_set:]

  # fitting
  # general bounds for a decreasing function that converges to a constant
  # popt = [X_sph, alpha, s]
  bounds_lfac = ([1., min(0.5*m0, 2*m0), 2.], [2., max(0.5*m0, 2*m0), 200])
  bounds_ShSt = ([0.5, 5.*n0, 1.], [1., .5*n0, 20])
  popt_lfac2 = get_fitting_smoothBPL(r_, lfac2_, beta=0., bounds=bounds_lfac)
  popt_ShSt = get_fitting_smoothBPL(r_, ShSt_, beta=0., bounds=bounds_ShSt)
  
  return popt_lfac2, popt_ShSt


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
    def fitfunc(x, X_sph, alpha, beta, s):
      return smooth_bpl(x, X_sph, alpha, beta, s)
    
  popt, pcov = curve_fit(fitfunc, x, array, **kwargs)
  return popt



# Find the final crossing radius, knowing the function for \Gamma(R)
def crossing_radius_fromfit(D0, beta_i, lfac0, m, lfac2_sph, g, s=2):
    '''
    Computes the final crossing radius for a shock front, in unit R0
      D0:         initial shell width, in units R0
      beta_i:     velocity of the shell
      lfac0:      shocked region Gamma at collision
      m:          power-law index for Gamma^2
      lfac2_sph:  shocked region Gamma^2 at large radius
      g:          Gamma_sh/Gamma_d
      s:          smoothing parameter (default 2.0)
    '''

    # inputs from fit are given for \Gamma_d^2
    lfac_sh0, a, lfac_sph = lfac0/g, -m/2., np.sqrt(lfac2_sph)/g

    # RS or FS
    beta_sh0 = np.sqrt(1 - lfac_sh0**-2)
    reverse = (beta_sh0 < beta_i)

    def func_Gamma(x):
      '''
      Lorentz factor of the shock front with x=R/R0
      '''
      Gamma_sh = lfac_sh0 * smooth_bpl(x, lfac_sph, a, 0, s)
      return Gamma_sh

    def integrand(x):
      '''
      Function to integrate
      '''
      Gamma_sh = func_Gamma(x)
      beta_sh = np.sqrt(1. - Gamma_sh**(-2))
      if reverse:   # RS
        return 1.0 - beta_sh / beta_i
      else:         # FS
        return 1.0 - beta_i / beta_sh
    
    def cumulative(x):
      '''
      Integral
      '''
      if x <= 1: return 0.
      val, _  = quad(integrand, 1., x,)
      return val
    
    def func_root(x):
      return cumulative - D0
    
    # find an upper limit to bracket root finding
    x_up = 3.
    while func_root(x_up) <= 0.:
      x_up *= 1.5
    
    x_f = brentq(func_root, 1, x_up)
    return x_f



    # R0 = env.R0
    # shell, front = ('4', 'RS') if reverse else ('1', 'FS')
    # D0, beta_i = [getattr(env, name+shell) for name in ['D0', 'beta']]
    # frontParams1, frontParams2 = ['g', 'Rf', 'nu0'], ['Fs']
    # g, Rf_th, nu0 = [getattr(env, name+front) for name in frontParams1]
    # Fs = getattr(env, 'Fs_'+front)
    # # find an upper limit
    # R_up = Rf_th
    # while func_root(R_up, D0, R0, beta_i, a, lfac0, lfac_sph, g, s, reverse) <= 0.:
    #     R_up *= 1.5
    
    # Rf = brentq(func_root, R0, R_up, args=(D0, R0, beta_i, a, lfac0, lfac_sph, g, s, reverse))
    # return Rf/env.R0

# def func_Gamma(r, m, lfac0, lfac_sph, g, s):
#     '''Gamma_sh as a function of r=R/R0'''
#     Gamma_d = lfac0*smooth_bpl(r, lfac_sph, -0.5*m, 0.0, s)
#     return Gamma_d/g

# def integrand(r, beta_i, m, lfac0, lfac_sph, g, s, reverse):
#     Gamma_sh = func_Gamma(r, m, lfac0, lfac_sph, g, s)
#     beta_sh = np.sqrt(1. - 1/(Gamma_sh*Gamma_sh))
#     if reverse:     # RS
#         integrand = 1.0 - beta_sh / beta_i
#     else:           # FS
#         integrand = 1.0 - beta_i / beta_sh
#     return integrand

# def cumulative(R, R0, beta_i, m, lfac0, lfac_sph, g, s, reverse):
#     if R <= R0:
#         return 0.
#     val, _ = quad(integrand, R0, R, args=(beta_i, m, lfac0, lfac_sph, g, s, reverse))
#     return val

# def func_root(R, D0, R0, beta_i, m, lfac0, lfac_sph, g, s, reverse):
#     return cumulative(R, R0, beta_i, m, lfac0, lfac_sph, g, s, reverse) - D0

