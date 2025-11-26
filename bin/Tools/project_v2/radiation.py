# -*- coding: utf-8 -*-
# @Author: acharlet

'''
Produce emission
'''

from IO import *
from phys_functions import Band_func

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


##### Peak frequency and flux (normalized data)
def get_peaks_from_data(key, front='RS'):
  '''
  Obtain peak frequency and flux from simulated data, in normalized units
  from front ('RS' or 'FS')
  
  '''

  env = MyEnv(key)
  NT = env.Nsh4 if (front == 'RS') else env.Nsh1
  nu, T, nF = get_radiation_vFC(key, front=front, norm=True, NT=NT, Nnu=500)
  NT = len(T)
  nu_pk, nF_pk = np.zeros((2, NT))
  for j in range(NT):
    nF_t = nF[j]
    i = np.argmax(nF_t)
    nu_pk[j] += nu[i]
    nF_pk[j] += nF_t[i]
  return nu_pk, nF_pk


##### From extracted time data
def get_radiation_vFC(key, front='RS', norm=True,
  Tmax=5, NT=450, lognu_min=-3, lognu_max=2, Nnu=500):
  '''
  \nu F_\nu (\nu, T) from front ('RS' or 'FS')
  Create file with normalized observed frequency, time and flux
  '''

  fpath = get_radfile_thinshell(key, front)
  exists = os.path.isfile(fpath)
  samesize = True
  if exists:
    obs = np.load(fpath)
    nu, T, nF = obs['nu'], obs['T'], obs['nF']
    if (len(nu) != Nnu) or (len(T) != NT):
      samesize = False

  if (not exists) or (not samesize):
    nu, T, env = obs_arrays(key, False, Tmax, NT, lognu_min, lognu_max, Nnu)
    z, nu0, T0 = (4, env.nu0, env.T0) if (front == 'RS') else (1, env.nu0FS, env.T0FS)
    data = open_rundata(key, z)
    nF = run_nuFnu_vFC(data, nu, T, env, norm)
    if norm:
      nu /= nu0
      T = (T - env.Ts)/T0
      np.savez(fpath, nu=nu, T=T, nF=nF)
  return nu, T, nF



def run_nuFnu_vFC(data, nuobs, Tobs, env, norm=True):
  '''
  Derive \nu F_\nu from a shock front across its history
    in the very fast cooling regime (thinshell approximation)
    with Band function spectral shape
    nuobs: ndarray, Tobs: ndarray
  '''

  nF = np.zeros((len(Tobs), len(nuobs)))
  if norm:
    trac = data.trac.to_numpy()[0]
    nu0 = (env.nu0 if (trac < 1.5) else env.nu0FS)
  else:
    nu0 = 1.
  # take only contributions from the first time step where downstream cell is shocked 
  data = data.drop_duplicates(subset='i', keep='first')
  for j, cell in data.iterrows():
    if cell.x > 0.:
      F = get_Fnu_vFC(cell, nuobs, Tobs, env, norm)
      nF_step = (nuobs/nu0) * F
    nF += nF_step
  return nF 


#### From a single time step
def get_Fnu_vFC(cell, nuobs, Tobs, env, norm=True):
  '''
  Derive Fnu for a cell (downstream of shock)
    in the very fast cooling regime (thinshell approximation)
    with Band function spectral shape
  nuobs: ndarray, Tobs: float or ndarray
  '''

  # normalizations
  Ton, Tth, Tej = get_variable(cell, 'obsT', env)
  T = (Tobs - Tej)/Tth   # \tilde{T}
  nub = nuobs/get_variable(cell, "nu_m2", env)
  F = get_variable(cell, 'Lth', env)
  if norm:
    F /= (env.L0 if (cell.trac < 1.5) else env.L0FS) / 3
  else:
    F *= env.zdl

  # spectral shape
  def S(x):
    return Band_func(x, -0.5, -0.5*env.psyn)

  if (type(T) == np.ndarray):
    Fnu = np.zeros((T.size, nub.size))
    T = T[:, np.newaxis]
    nub = nub[np.newaxis, :]
  else:
    Fnu = np.zeros(nub.shape)
  Fnu += np.where(T>=1, F * S(nub*T) / (T*T), 0.)
  return Fnu

def get_Fnu_distrib(cell, nuobs, Tobs, env, norm=True):
  '''
  Derive Fnu for a cell with an electron distribution
  nuobs: ndarray, Tobs: float
  '''