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
    Tmax=5, NT=500,
    lognu_min=-3, lognu_max=2, Nnu=250, Tb_min=None, Tb_lin=None):
  '''
  Returns arrays of observed times and frequencies.
  Tb_min: if set, the time grid is geometric in bar{T} = (Tobs-Ts)/T0 = T-1
    over [Tb_min, Tmax] (log-uniform, resolving the early rise); None keeps the
    default T = geomspace(1, Tmax+1) (bar{T} only ~linear near 0).
  Tb_lin: (lo, hi, n) -- merge n extra samples spaced LINEARLY in bar{T} over
    [lo, hi] into the grid above. The geometric grid resolves the log axis, so it
    leaves the peak region thin in linear terms (on the fiducial sweep only ~12 of
    250 points fall in bar{T}/bar{T}_f = 1..2, where the lightcurve peaks) and
    linear-scale lightcurves come out jagged. Every geometric point is kept -- the
    result is a strict superset, so the early rise Tb_min buys is untouched -- and
    NT then sizes only the geometric part: len(T) == NT + n.
  '''
  env = MyEnv(key)
  nub = np.logspace(lognu_min, lognu_max, Nnu)
  if Tb_min is None:
    T = np.geomspace(1, Tmax+1, NT)
  else:
    T = 1. + np.geomspace(Tb_min, Tmax, NT)
  if Tb_lin is not None:
    lo, hi, n = Tb_lin
    T = 1. + np.unique(np.concatenate([T - 1., np.linspace(lo, min(hi, Tmax), n)]))
  if normed:
    return nub, T, env
  else:
    nuobs = nub * env.nu0
    Tobs = env.Ts + (T - 1) * env.T0
    return nuobs, Tobs, env
  
def obs_arrays_peakcentred(key, normed=False,
    Tmax=5, NT=500, Nnu=300):
  '''
  like obs_arrays but around spectral peaks
  '''
  env = MyEnv(key)
  nuRS, nuFS = env.nu0, env.nu0FS
  lognu_FS = np.log10(env.nu0FS/env.nu0)
  lognu_min = min(lognu_FS, 1) - (np.log10(Tmax)+1)
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
def normalize_time(Tobs, z, env):
  T0 = env.T0FS if z==1 else env.T0
  T = (Tobs - env.Ts)/T0 + 1
  return T

def normalize_freq(nuobs, z, env):
  nu0 = env.nu0FS if z==1 else env.nu0
  nu = nuobs/nu0
  return nu

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