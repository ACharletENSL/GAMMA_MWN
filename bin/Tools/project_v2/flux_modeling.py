# -*- coding: utf-8 -*-
# @Author: acharlet

'''
Modeling peak frequency - peak flux correlation from internal shock collision
following Rahaman et al. 2024a,b and Charlet et al. 2025
Contains:
  - flux_model_C25
  - flux_model_C25_normed
  - flux_model_R24
'''

import numpy as np
from peak_modeling import *
from phys_functions import Band_func

def flux_model_C25_binned(Tobs, nuobs, Ts, T0, nu0, F0, au, dRRS, dRFS, alpha=0.5, beta=-1.25):
  '''
  Same as flux_model_C25 but taking into account the binning in time and energy
  '''
  print('Not implemented yet')
  return 0.

def timeIntegrated_flux_model_C25(T_min, T_max, nu_obs,
    Ts, T0, nu0, F0,
    au, dRRS, dRFS, alpha=-0.5, beta=-1.25):
  '''
  Time integrated flux over a given interval
  '''
  Tb_min = (T_min - Ts)/T0
  Tb_max = (T_max - Ts)/T0
  nu_b = nuobs/nu0
  F = timeIntegrated_flux_model_C25_normed(Tb_min, Tb_max, nu_b,
    au, dRRS, dRFS, alpha, beta)
  F *= F0
  return F


def timeIntegrated_flux_model_C25_normed(Tb_min, Tb_max, nu_b,
    au, dRRS, dRFS, alpha=-0.5, beta=-1.25):
  '''
  Time integrated flux over a given interval, all quantities normalized
  '''

  # discontinuous derivative at Tf
  popt_lfacRS = fits_from_au(au, 'RS')[0]
  TfRS = Tf_from_dR(dRRS, popt_lfac) - 1
  popt_lfacFS = fits_from_au(au, 'RS')[0]
  TfFS = Tf_from_dR(dRFS, popt_lfac) - 1
  ratio_T, _, _ = ratios_RSvFS_from_au(au)
  TtFS *= ratio_T
  Tf1, Tf2 = min(TfRS, TfFS), max(TfRS, TfFS)
  
  # define time bins
  if Tb_max <= Tf1:
    T_bins = np.linspace(Tb_min, Tb_max, 20)
  elif (Tb_max <= Tf2):
    if Tb_min < Tf1:
      bins_1 = np.linspace(Tb_min, Tf1, 10)
      bins_2 = np.linspace(Tf1, Tb_max, 10)
      T_bins = np.concatenate((bins_1, bins_2))
    else:
      T_bins = np.linspace(Tb_min, Tb_max, 20)
  else:  # Tb_max > Tf2
    if Tb_min < Tf1:
      bins_1 = np.linspace(Tb_min, Tf1, 10)
      bins_2 = np.linspace(Tf1, Tf2, 10)
      bins_3 = np.geomspace(Tf2, Tb_max, 10)
      T_bins = np.concatenate((bins_1, bins_2, bins_3))
    elif Tb_min < Tf2:
      bins_1 = np.linspace(Tb_min, Tf2, 10)
      bins_2 = np.geomspace(Tf2, Tb_max, 10)
      T_bins = np.concatenate((bins_1, bins_2))
    else:
      T_bins = np.geomspace(Tb_min, Tb_max, 20)
  
  # calculate flux
  F_arr = flux_model_C25_normed(T_bins, nu_b, au, dRRS, dRFS, alpha, beta)
  F = np.trapezoid(F_arr, T_bins)
  return F


def flux_model_C25(Tobs, nuobs, Ts, T0, nu0, F0, au, dRRS, dRFS, alpha=0.5, beta=-1.25):
  '''
  Return flux using the built model
    Tobs in s 
    nuobs in Hz
    flux in erg.cm^-2
  '''

  T_b = (Tobs - Ts)/T0
  nu_b = nuobs/nu0
  F = flux_model_C25_normed(T_b, nu_b, au, dRRS, dRFS, alpha, beta)
  F *= F0
  return F

def flux_model_C25_normed(T_b, nu_b, au, dRRS, dRFS, alpha=-0.5, beta=-1.25):
  '''
  Return flux using the built model with time
    time T_b is normalized to angular time of RS at R0
    frequency nu_b is normalized to peak frequency of the RS
    resulting flux is normalized relative to the peak RS flux
  '''

  # RS
  tT = 1 + T_b
  nupks_RS, nFpks_RS = peaks_model_C25(tT, au, dRRS, reverse=True)
  F_RS = np.zeros((len(T_b), len(nu_b)))
  for i, (nupk, nFpk) in enumerate(zip(nupks_RS, nFpks_RS)):
    Fpk = nFpk/nupk
    nu = nu_b/nupk
    F = Fpk * Band_func(nu, alpha, beta)
    F_RS[i] = F
  
  # FS
  ratio_T, ratio_nu, ratio_F = ratios_RSvFS_from_au(au)
  tT_FS = 1 + T_b * ratio_T
  nupks_FS, nFpks_FS = peaks_model_C25(tT_FS, au, dRFS, reverse=False)
  nupks_FS /= ratio_nu
  nFpks_FS /= (ratio_F*ratio_nu)
  F_FS = np.zeros((len(T_b), len(nu_b)))
  for i, (nupk, nFpk) in enumerate(zip(nupks_FS, nFpks_FS)):
    Fpk = nFpk/nupk
    nu = nu_b/nupk
    F = Fpk * Band_func(nu, alpha, beta)
    F_FS[i] = F
  
  F_tot = F_RS + F_FS
  return F_tot



def flux_model_R24(au, dR_RS, dR_FS,
    T_max=5, lognu_min=-3, lognu_max=2,
    NT=100, Nnu=500,
    sl_RS=[-.5, -1.25], sl_FS=[-.5, -1.25]):
  '''
  nu F_nu(T) for RS and FS, normalized to RS, following prescription in Rahaman+ 24
  '''

  # choice of normalization: RS
  tT_RS = np.geomspace(1, 1+T_max, NT)
  nub = np.logspace(lognu_min, lognu_max, Nnu)

  # ratios of downstream to shock LF, squared
  g2FS, g2RS = g2_from_au(au)

  # peak frequencies and flux
  nu_RS, nF_RS = peaks_model_R24(tT_RS, au, dR_RS, g2RS)

  nuFnu_RS = np.zeros((NT, Nnu))
  for i, tT, nup, nFp in zip(range(NT), tT_RS, nu_RS, nF_RS):
    x = nub/nup
    nuFnu_RS[i] += nFp * x * Band_func(x, *sl_RS)

  # for FS
  G21, G34 = relLfac_from_au(au)
  b21, b34 = np.sqrt(1. - G21**-2) , np.sqrt(1. - G34**-2)

  # fac_T = T_0,RS/T_0,FS ~ (Gamma_FS/Gamma_RS)**2 = (gRS/gFS)**2
  # R24b eqn (E16), beta_RS/beta_FS ~ 1.
  fac_T = g2RS/g2FS
  tT_FS = 1. + (tT_RS - 1.)*fac_T

  # fac_nu = nu_0,RS / nu_0,FS ; R24b eqn (C4)
  fac_nu = np.sqrt(((G21+1)/(G34+1)) * (G34/G21)) * ((G34-1)/(G21-1))**2

  # fac_nF = L'_bol,RS / L'_bol,FS ; R24b eqn (D9)
  # eps_rad,RS = eps_rad,FS = 1 ; can be added as parameters
  fac_nF = ((G21+1)/(G34+1)) * (G34/G21) * (b34/b21)
  fac_F = fac_nF / fac_nu

  # peak frequencies and flux, in units nu_0,FS and F_0,FS
  nu_FS, nF_FS = peaks_model_R24(tT_FS, au, dR_FS, g2FS)
  nu_FS /= fac_nu
  nF_FS /= fac_nF

  nuFnu_FS = np.zeros((NT, Nnu))
  for i, tT, nup, nFp in zip(range(NT), tT_FS, nu_FS, nF_FS):
    x = nub/nup
    nuFnu_FS[i] += nFp * x * Band_func(x, *sl_FS)
  
  return nuFnu_RS, nuFnu_FS


def flux_model_1shock_R24(au, dR, g2, tT_arr, nu_arr, slopes):
  '''
  Contribution from a single shock front
    all quantities are normalized accordingly
  '''

  NT = len(tT_arr)
  nuFnu = np.zeros((NT, len(nu_arr)))
  nu_pk, nF_pk = peaks_model_R24(tT_arr, au, dR, g2)

  for i, tT, nup, nFp in zip(range(NT), tT_arr, nu_pk, nF_pk):
    x = nu_arr/nup
    nuFnu[i] += nFp * x * Band_func(x, *slopes)
  
  return nuFnu
