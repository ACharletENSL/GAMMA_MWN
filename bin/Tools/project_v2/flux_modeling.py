# -*- coding: utf-8 -*-
# @Author: acharlet

'''
Modeling observed flux from the collision of two ultrarelativistic cold shells
following Rahaman et al. 2024a,b and Charlet et al. 2025
All shock-crossing inputs are tau = t_on/t_off (RS uses t_on4, FS uses t_on1),
matching the tau-based peak_modeling API.
Contains:
  - flux_model_C25, flux_model_C25_normed
  - flux_model_R24
  - flux_model_1shock_R24
  - flux_model_1shock_C25_fromTf
'''

import numpy as np
from peak_modeling import *
from phys_functions import Band_func


def flux_model_C25(Tobs, nuobs, Ts, T0, nu0, F0, au, tau4, tau1, alpha=-0.5, beta=-1.25):
  '''
  Observed flux F_nu(T) from RS+FS, in physical units
    Tobs in s, nuobs in Hz, flux in erg.cm^-2
  '''
  T_b = (Tobs - Ts)/T0
  nu_b = nuobs/nu0
  F = flux_model_C25_normed(T_b, nu_b, au, tau4, tau1, alpha, beta)
  F *= F0
  return F


def flux_model_C25_normed(T_b, nu_b, au, tau4, tau1, alpha=-0.5, beta=-1.25):
  '''
  Return flux using the built model with time
    time T_b is normalized to angular time of RS at R0
    frequency nu_b is normalized to peak frequency of the RS
    resulting flux is normalized relative to the peak RS flux
  '''

  # RS
  tT = 1 + T_b
  nupks_RS, nFpks_RS = C25_peak_model(tT, au, tau4, reverse=True)
  F_RS = np.zeros((len(T_b), len(nu_b)))
  for i, (nupk, nFpk) in enumerate(zip(nupks_RS, nFpks_RS)):
    Fpk = nFpk/nupk
    nu = nu_b/nupk
    F_RS[i] = Fpk * Band_func(nu, alpha, beta)

  # FS, normalized to RS quantities
  ratio_T, ratio_nu, ratio_F = ratios_RSvFS_from_au(au)
  tT_FS = 1 + T_b * ratio_T
  nupks_FS, nFpks_FS = C25_peak_model(tT_FS, au, tau1, reverse=False)
  nupks_FS /= ratio_nu
  nFpks_FS /= (ratio_F*ratio_nu)
  F_FS = np.zeros((len(T_b), len(nu_b)))
  for i, (nupk, nFpk) in enumerate(zip(nupks_FS, nFpks_FS)):
    Fpk = nFpk/nupk
    nu = nu_b/nupk
    F_FS[i] = Fpk * Band_func(nu, alpha, beta)

  return F_RS + F_FS


def flux_model_R24(au, tau4, tau1,
    T_max=5, lognu_min=-3, lognu_max=2,
    NT=100, Nnu=500,
    sl_RS=[-.5, -1.25], sl_FS=[-.5, -1.25]):
  '''
  nu F_nu(T) for RS and FS, normalized to RS, following prescription in Rahaman+ 24
    (constant-LF model)
  '''

  # choice of normalization: RS
  tT_RS = np.geomspace(1, 1+T_max, NT)
  nub = np.logspace(lognu_min, lognu_max, Nnu)

  # ratios of downstream to shock LF, squared
  g2RS, g2FS = g2_from_au(au)

  # peak frequencies and flux
  nu_RS, nF_RS = R24_peak_model(tT_RS, au, tau4, reverse=True)
  nuFnu_RS = np.zeros((NT, Nnu))
  for i, nup, nFp in zip(range(NT), nu_RS, nF_RS):
    x = nub/nup
    nuFnu_RS[i] = nFp * x * Band_func(x, *sl_RS)

  # for FS
  G34, G21 = relLfac_from_au(au)
  b21, b34 = np.sqrt(1. - G21**-2), np.sqrt(1. - G34**-2)

  # fac_T = T_0,RS/T_0,FS ~ (Gamma_FS/Gamma_RS)**2 = (gRS/gFS)**2
  # R24b eqn (E16), beta_RS/beta_FS ~ 1.
  fac_T = g2RS/g2FS
  tT_FS = 1. + (tT_RS - 1.)*fac_T

  # fac_nu = nu_0,RS / nu_0,FS ; R24b eqn (C4)
  fac_nu = np.sqrt(((G21+1)/(G34+1)) * (G34/G21)) * ((G34-1)/(G21-1))**2

  # fac_nF = L'_bol,RS / L'_bol,FS ; R24b eqn (D9)
  # eps_rad,RS = eps_rad,FS = 1 ; can be added as parameters
  fac_nF = ((G21+1)/(G34+1)) * (G34/G21) * (b34/b21)

  # peak frequencies and flux, in units nu_0,FS and F_0,FS
  nu_FS, nF_FS = R24_peak_model(tT_FS, au, tau1, reverse=False)
  nu_FS /= fac_nu
  nF_FS /= fac_nF
  nuFnu_FS = np.zeros((NT, Nnu))
  for i, nup, nFp in zip(range(NT), nu_FS, nF_FS):
    x = nub/nup
    nuFnu_FS[i] = nFp * x * Band_func(x, *sl_FS)

  return nuFnu_RS, nuFnu_FS


def flux_model_1shock_R24(au, tau, tT_arr, nu_arr, slopes, reverse=True):
  '''
  Contribution from a single shock front (constant-LF model)
    all quantities are normalized accordingly
  '''

  NT = len(tT_arr)
  nuFnu = np.zeros((NT, len(nu_arr)))
  nu_pk, nF_pk = R24_peak_model(tT_arr, au, tau, reverse=reverse)
  for i, nup, nFp in zip(range(NT), nu_pk, nF_pk):
    x = nu_arr/nup
    nuFnu[i] = nFp * x * Band_func(x, *slopes)

  return nuFnu


def flux_model_1shock_C25_fromTf(tT, nu_b, au, Tf, reverse=True, slopes=(-0.5, -1.25)):
  '''
  nu F_nu(T) from a single C25 (variable-LF) shock front, parametrized by the
    crossing time tilde{T}_f so time-integrated studies can sweep Tf directly.
  Returns array of shape (len(tT), len(nu_b)), normalized to that front.
  '''

  popts = fitparams_from_au(au, reverse)
  g2 = g2_from_au(au)[0 if reverse else 1]
  func_peaks = build_peaks_functions(Tf, g2, popts)
  nupk, nFpk = func_peaks(tT)
  nuFnu = np.zeros((len(tT), len(nu_b)))
  for i, (nup, nFp) in enumerate(zip(nupk, nFpk)):
    x = nu_b/nup
    nuFnu[i] = nFp * x * Band_func(x, *slopes)
  return nuFnu

def flux_model_1shock_C25_fromtau(tT, nu_b, au, tau, reverse=True, slopes=(-0.5, -1.25)):
  '''
  Same as _fromTf but starting from tau = ton/toff
  '''
  popts = fitparams_from_au(au, reverse)
  g2 = g2_from_au(au)[0 if reverse else 1]
  Tf, xf = compute_Tf(tau, au, popts[0], reverse)
  func_peaks = build_peaks_functions(Tf, g2, popts)
  nupk, nFpk = func_peaks(tT)
  nuFnu = np.zeros((len(tT), len(nu_b)))
  for i, (nup, nFp) in enumerate(zip(nupk, nFpk)):
    x = nu_b/nup
    nuFnu[i] = nFp * x * Band_func(x, *slopes)
  return nuFnu