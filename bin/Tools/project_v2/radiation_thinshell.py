# -*- coding: utf-8 -*-
# @Author: acharlet

'''
Produce emission from hydro data in the thinshell regime
Contains:
  - run_nuFnu_vFC
  - get_Fnu_vFC
'''

from IO import *
from phys_constants import *
from phys_functions import Band_func, smooth_bpl0



##### From extracted time data
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
  tnu = nuobs/nu0
  data = data.drop_duplicates(subset='i', keep='first')
  for j, cell in data.iterrows():
    if cell.x > 0.:
      F = get_Fnu_vFC(cell, nuobs, Tobs, env, norm)
      nF_step = tnu * F
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
  num = get_variable(cell, "nu_m2", env)
  F = get_variable(cell, 'Lth', env)
  if env.geometry == 'cartesian':
    # add geometrical scalings to planar simulation
    rfac = cell.x * c_ / env.R0
    num /= rfac
    F   *= rfac 
  nub = nuobs/num

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

