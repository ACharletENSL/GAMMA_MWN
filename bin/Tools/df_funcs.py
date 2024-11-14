# -*- coding: utf-8 -*-
# @Author: acharlet

'''
This file contains functions that are transformations on a dataframe
'''

import numpy as np
import pandas as pd
from phys_functions import *
from phys_constants import *
from phys_edistrib import *
from environment import MyEnv

# get variables
cool_vars = ["V4", "dt", "inj", "fc", "gm", "gb", "gM",
  "num", "nub", "nuM", "Lp", "L", "V4r", "tc"]
var2func = {
  "T":(derive_temperature, ['rho', 'p'], []),
  "h":(derive_enthalpy, ['rho', 'p'], []),
  "ad":(derive_adiab, ['rho', 'p'], []),
  "lfac":(derive_Lorentz, ['vx'], []),
  "u":(derive_proper, ['vx'], []),
  "Ek":(derive_Ekin, ['rho', 'vx'], ['rhoscale']),
  "Ei":(derive_Eint, ['rho', 'vx', 'p'], ['rhoscale']),
  "ei":(derive_Eint_comoving, ['rho', 'p'], ['rhoscale']),
  "B":(derive_B_comoving, ['rho', 'p'], ['rhoscale', 'eps_B']),
  "res":(derive_resolution, ['x', 'dx'], []),
  "numax":(derive_nu, ['rho', 'p', 'gmax'], ['rhoscale', 'eps_B']),
  "numin":(derive_nu, ['rho', 'p', 'gmin'], ['rhoscale', 'eps_B']),
  "num":(derive_nu, ['rho', 'p', 'gm'], ['rhoscale', 'eps_B']),
  "nub":(derive_nu, ['rho', 'p', 'gb'], ['rhoscale', 'eps_B']),
  "nuM":(derive_nu, ['rho', 'p', 'gM'], ['rhoscale', 'eps_B']),
  "Pmax":(derive_max_emissivity, ['rho', 'p', 'gmin', 'gmax'], ['rhoscale', 'eps_B', 'psyn', 'xi_e']),
  "Tth":(derive_normTime, ['x', 'vx'], []),
  "Lp_num":(derive_localLum, ['rho', 'vx', 'p', 'x', 'dx', 'dt', 'gmin', 'gmax', 'gb'],
    ['rhoscale', 'eps_B', 'psyn', 'xi_e', 'R0', 'geometry']),
  "L":(derive_obsLum, ['rho', 'vx', 'p', 'x', 'dx', 'dt', 'gmin', 'gmax', 'gb'], 
    ['rhoscale', 'eps_B', 'psyn', 'xi_e', 'R0', 'geometry']),
  "Lum":(derive_Lum, ['x', 'dx', 'rho', 'vx', 'p', 'gmin'],
    ['R0', 'rhoscale', 'psyn', 'eps_B', 'eps_e', 'geometry']),
  "Lth":(derive_Lum_thinshell, ['x', 'dx', 'rho', 'vx', 'p'],
    ['R0', 'rhoscale', 'psyn', 'eps_B', 'eps_e', 'xi_e', 'geometry']),
  "Lth2":(derive_L_thsh_corr, ['t', 'x', 'dx', 'rho', 'vx', 'p'],
    ['R0', 'rhoscale', 'psyn', 'eps_B', 'eps_e', 'xi_e', 'geometry']),
  "Lbol":(derive_Lbol_comov, ['x', 'rho', 'p'], ['R0', 'rhoscale', 'eps_e','geometry']),
  "Lp": (derive_Lp_nupm, ['x', 'rho', 'p'], ['R0', 'rhoscale', 'psyn', 'eps_B', 'eps_e', 'xi_e', 'geometry']),
  "EpnuFC":(derive_Epnu_vFC, ['x', 'dx', 'rho', 'vx', 'p', 'gmin'],
    ['R0', 'rhoscale', 'psyn', 'eps_B', 'eps_e', 'geometry']),
  "Fpk_th":(derive_Fpeak_analytic, ['x', 'dx', 'rho', 'vx', 'p', 'gmin'],
    ['R0', 'rhoscale', 'psyn', 'eps_B', 'eps_e', 'zdl', 'geometry']),
  "Fpk_n":(derive_Fpeak_numeric, ['x', 'dx', 'rho', 'vx', 'p', 'gmin'],
    ['R0', 'rhoscale', 'psyn', 'eps_B', 'eps_e', 'zdl', 'geometry']), 
  "n":(derive_n, ['rho'], ['rhoscale']),
  "Pfac":(derive_emiss_prefac, [], ['psyn']),
  "obsT":(derive_obsTimes, ['t', 'x', 'vx'], ['t0', 'z']),
  "Tth":(derive_obsAngTime, ['t', 'x', 'vx'], ['t0', 'z']),
  "Tej":(derive_obsEjTime, ['t', 'x', 'vx'], ['t0', 'z']),
  "Ton":(derive_obsOnTime, ['t', 'x', 'vx'], ['t0', 'z']),
  "V3":(derive_3volume, ['x', 'dx'], ['R0', 'geometry']),
  "V3p":(derive_3vol_comoving, ['x', 'dx', 'vx'], ['R0', 'geometry']),
  "V4":(derive_4volume, ['x', 'dx', 'dt'], ['R0', 'geometry']),
  "V4r":(derive_rad4volume, ['x', 'dx', 'rho', 'vx', 'p', 'dt', 'gb'],
    ['rhoscale', 'eps_B', 'R0', 'geometry']),
  "tc":(derive_tcool, ['rho', 'vx', 'p', 'gmin'], ['rhoscale', 'eps_B']),
  "tc1":(derive_tc1, ['rho', 'vx', 'p'], ['rhoscale', 'eps_B']),
  "gma_m":(derive_gma_m, ['rho', 'p'], ['rhoscale', 'psyn', 'eps_e', 'xi_e']),
  "gma_c":(derive_gma_c, ['t', 'rho', 'vx', 'p'], ['rhoscale', 'eps_B']),
  "nup_B":(derive_cyclotron_comoving, ['rho', 'p'], ['rhoscale', 'eps_B']),
  "nup_m":(derive_nup_m_from_gmin, ['rho', 'p', 'gmin'], ['rhoscale', 'eps_B']),
  "nu_m":(derive_nu_m_from_gmin, ['rho', 'vx', 'p', 'gmin'], ['rhoscale', 'eps_B', 'z']),
  "nup_m2":(derive_nup_m, ['rho', 'p'], ['rhoscale', 'psyn', 'eps_B', 'eps_e', 'xi_e']),
  "nu_m2":(derive_nu_m, ['rho', 'vx', 'p'], ['rhoscale', 'psyn', 'eps_B', 'eps_e', 'xi_e', 'z']),
  "nup_c":(derive_nup_c, ['t', 'rho', 'vx', 'p'], ['rhoscale', 'eps_B']),
  "gmM":(derive_edistrib, ['rho', 'p'], ['rhoscale', 'psyn', 'eps_B', 'eps_e', 'xi_e']),
}

def get_vars(df, varlist):
  '''
  Return array of variables
  '''
  if type(df) == pd.core.series.Series:
    out = np.zeros((len(varlist)))
  else:
    out = np.zeros((len(varlist), len(df)))
  for i, var in enumerate(varlist):
    out[i] = get_variable(df, var)
  return out

def get_variable(df, var):
  '''
  Returns chosen variable
  '''
  if var in df.keys():
    try:
      res = df[var].to_numpy(copy=True)
    except AttributeError:
      res = df[var]
  else:
    func, inlist, envlist = var2func[var]
    res = df_get_var(df, func, inlist, envlist)
  return res

def df_get_var(df, func, inlist, envlist):
  params, envParams = get_params(df, inlist, envlist)
  res = func(*params, *envParams)
  return res

def get_params(df, inlist, envlist):
  if (type(df) == pd.core.series.Series):
    params = np.empty(len(inlist))
  else:
    params = np.empty((len(inlist), len(df)))
  for i, var in enumerate(inlist):
    if var in df.attrs:
      params[i] = df.attrs[var].copy()
    else:
      try:
        params[i] = df[var].to_numpy(copy=True)
      except AttributeError:
        params[i] = df[var]
  envParams = []
  if len(envlist) > 0.:
    env = MyEnv(df.attrs['key'])
    envParams = [getattr(env, name) for name in envlist]
  return params, envParams

def df_get_dt(df):
  '''
  Get time step for each cell from the Riemannian waves
  '''
  
  # create copies of dataframe with one ghost cell on left and right respectively
  # dfL : add cell with wind on the left
  dfL = df.copy(deep=True)
  row0 = dfL.iloc[0]
  dfL.loc[-1] = row0  # let's test with a simple copy
  dfL.index = dfL.index + 1
  dfL = dfL.sort_index()

  #dfR : add cell of ejecta on the right
  dfR = df.copy(deep=True)
  rowLast = pd.DataFrame(dfR.iloc[-1]).transpose() # let's test with a simple copy
  rowLast.index = rowLast.index + 1
  dfR = pd.concat([dfR, rowLast])

  # apply function
  N = len(df)+1
  dt = np.zeros(N-1)
  waves = np.zeros((N, 3))
  for i in range(N):
    SL = dfL.loc[i]
    SR = dfR.loc[i]
    lL, lR, lS = waveSpeedEstimates(SL, SR)
    waves[i] = [lL, lR, lS]
  
  for i in range(N-1):
    a = df.iloc[i]['dx']

    # wave from left side
    l = waves[i][1]
    v = waves[i][2]
    dt_candL = a / (l - v) if (l > v) else 1e15

    # wave from right side
    l = waves[i+1][0]
    v = waves[i+1][2]
    dt_candR = a / (v - l) if (l < v) else 1e15

    dt_cand = min(dt_candL, dt_candR)
    dt[i] = dt_cand

  return dt
