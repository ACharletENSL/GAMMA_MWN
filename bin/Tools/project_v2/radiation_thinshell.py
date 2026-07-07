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
def run_nuFnu_vFC(nuobs, Tobs, data, env, norm=True, cutoff=False, nu_resc=1., Tf=None):
  '''
  Derive \nu F_\nu from a shock front across its history
    in the very fast cooling regime (thinshell approximation)
    with Band function spectral shape
    nuobs: ndarray, Tobs: ndarray
  Signature consistent with the cooling functions (nuobs, Tobs, data, env, ...)
  Tf: float or None
    If set, artificially stop including contributions from cells whose on-axis
    (onset) normalized time \tilde{T} exceeds Tf. This mimics the source being
    switched off at \tilde{T} = Tf, forcing the transition to high-latitude
    (curvature) emission: for \tilde{T} > Tf the surviving flux is only the
    1/\tilde{T}^2 tail of the last cells that emitted before Tf.
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
      if Tf is not None:
        # normalized on-axis onset time of this cell, in the same \tilde{T} units
        # as the light curve (cf. working_peaks.get_peaks_data); skip cells whose
        # emission only starts after the artificial cutoff Tf
        Ton, Tth, Tej = get_variable(cell, 'obsT', env)
        T0 = env.T0FS if (cell.trac >= 1.5) else env.T0
        tT_on = 1 + (Ton - env.Ts)/T0
        if tT_on > Tf:
          continue
      F = get_Fnu_vFC(nuobs, Tobs, cell, env, norm, cutoff, nu_resc)
      nF_step = tnu * F
    nF += nF_step
  return nF


#### From a single time step
def get_Fnu_vFC(nuobs, Tobs, cell, env, norm=True, cutoff=False, nu_resc=1.):
  '''
  Derive Fnu for a cell (downstream of shock)
    in the very fast cooling regime (thinshell approximation)
    with Band function spectral shape
  nuobs: ndarray, Tobs: float or ndarray
  nu_resc: frequency shift
  Signature consistent with the cooling functions (nuobs, Tobs, cell, env, ...)
  '''

  # normalizations
  Ton, Tth, Tej = get_variable(cell, 'obsT', env)
  T = (Tobs - Tej)/Tth   # \tilde{T}
  num = get_variable(cell, "nu_m2", env)
  num *= nu_resc
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
  b1, b2 = -0.5, -0.5*env.psyn
  if cutoff:
    # high-frequency cutoff above nu_M, consistent with the single-electron kernel
    # syn_emiss_exact (func_R ~ exp(-x), x = 2 tnu/(3 gma^2), tnu = nu'/nu'_B).
    # Band argument x = nub*T = nu'/nu'_m, and nu'_B = nu'_m/gma_m^2, so the gma_M
    # electrons give x_R = (2/3) x (gma_m/gma_M)^2; exp(-x_R) hits exp(-2/3) at nu_M.
    gma_m = get_variable(cell, 'gma_m', env)
    gma_M = get_variable(cell, 'gma_M', env)
    ratio_Mm = (gma_M/gma_m)**2            # nu'_M/nu'_m
    def S(x):
      return Band_func(x, b1, b2) * np.exp(-(2./3.)*x/ratio_Mm)
  else:
    def S(x):
      return Band_func(x, b1, b2)

  if (type(T) == np.ndarray):
    Fnu = np.zeros((T.size, nub.size))
    T = T[:, np.newaxis]
    nub = nub[np.newaxis, :]
  else:
    Fnu = np.zeros(nub.shape)
  Fnu += np.where(T>=1, F * S(nub*T) / (T*T), 0.)
  return Fnu

