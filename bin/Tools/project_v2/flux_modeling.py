# -*- coding: utf-8 -*-
# @Author: acharlet

'''
Modeling peak frequency - peak flux correlation from internal shock collision
following Rahaman et al. 2024a,b and Charlet et al. 2025
Contains:
  - flux_model_R24
'''

import numpy as np
from peak_modeling import relLfac_from_au, g2_from_au, peaks_model_R24
from phys_functions import Band_func

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
