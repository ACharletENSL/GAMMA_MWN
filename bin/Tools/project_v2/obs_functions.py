# -*- coding: utf-8 -*-
# @Author: acharlet

'''
Functions relevant to observed quantities
'''

import numpy as np
from environment import MyEnv
from IO import get_variable

##### Standard obs frequency and time arrays
def obs_arrays(key, normed=False,
    Tmax=5, NT=450, lognu_min=-3, lognu_max=2, Nnu=200):
  '''
  Returns arrays of observed times and frequencies
  '''
  env = MyEnv(key)
  nub = np.logspace(lognu_min, lognu_max, Nnu)
  T = np.geomspace(1, Tmax+1, NT)
  if normed:
    return nub, T, env
  else:
    nuobs = nub * env.nu0
    Tobs = env.Ts + (T - 1) * env.T0
    return nuobs, Tobs, env
  
def obs_arrays_peakcentred(key, normed=False,
    Tmax=5, NT=450, Nnu=200):
  '''
  like obs_arrays but around spectral peaks
  '''
  env = MyEnv(key)
  nuRS, nuFS = env.nu0, env.nu0FS
  lognu_FS = np.log10(env.nu0FS/env.nu0)
  lognu_min = min(lognu_FS, 1) - 2.
  lognu_max = max(lognu_FS, 1) + 0.5

  nub = np.logspace(lognu_min, lognu_max, Nnu)
  T = np.geomspace(1, Tmax+1, NT)
  if normed:
    return nub, T, env
  else:
    nuobs = nub * env.nu0
    Tobs = env.Ts + (T - 1) * env.T0
    return nuobs, Tobs, env


##### Normalizations
def Tobs_to_tildeT(Tobs, cell, env):
  '''
  Normalizes observed time
  (Why did I never write this before?)
  '''
  Ton, Tth, Tej = get_variable(cell, 'obsT', env)
  tT = ((Tobs - Tej)/Tth)
  return tT

def nuobs_to_nucomov_normalized(nuobs, cell, env, norm='syn'):
  '''
  From observer to comoving frame
  norm = syn normalizes to cyclotron freq, other to nu'_0 in env
  '''

  D = get_variable(cell, 'Dop', env)
  nup = nuobs/D
  if norm == 'syn':
    nuB = get_variable(cell, 'nup_B', env)
    nu = nup/nuB
  else:
    nu = nup/env.nu0p
  return nu