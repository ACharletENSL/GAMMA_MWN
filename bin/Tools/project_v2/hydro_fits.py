# -*- coding: utf-8 -*-
# @Author: acharlet

'''
Fits hydro quantities
'''

import numpy as np
from scipy.optimize import curve_fit

from phys_constants import c_
from phys_functions import smooth_bpl, init_slope, logslope
from IO import get_vars
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
