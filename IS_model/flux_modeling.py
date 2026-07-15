# -*- coding: utf-8 -*-
# @Author: acharlet

'''
Modeling spectral flux from internal shock collision
'''

import numpy as np

# RS-vs-FS normalization ratios (shared analytical helper)
from analytical_funcs import ratios_RSvFS_from_au

def flux_model(nuobs, Tobs, peak_model, au, tau1, tau4, T0RS, nu0RS, F0RS,
      alphaRS, betaRS, alphaFS=None, betaFS=None, epsrad_FS=1):
  '''
  Spectral model F_{nu_obs}(T_obs), assuming Band function shape for both shocks
    nuobs:    nu_{obs}, observed frequency. Must be numpy array given the definition of S(x)
    Tobs:     T_{obs}, observed time, T = 0 at the beginning of the pulse
    peak_model:
              model, returns nu_pk, (nu F_nu)_pk for tilde{T}, au, tau_i, reverse
              use function peak_model defined in peak_modeling_C25(or R24).py
    au:       a_u, proper velocity contrast
    taui:     tau_i=t_on,i/t_off, source activity time during emission of shell i
                  over the time between the emission of the two shells
    T0RS:     T_{0,RS}, angular time corresponding to the RS at collision radius
    nu0RS:    nu_{0,RS}, peak frequency of the RS contribution, calculated at R0
    F0RS:     F_{0,RS}, flux normalization of the RS
    alpha:    spectral slope (in F_nu) below the peak of nu F_nu
    beta:     spectral slope (in F_nu) above the peak of nu F_nu
  '''

  # Set FS spectral slopes equals to RS by default
  if alphaFS is None: alphaFS = alphaRS
  if betaFS is None: betaFS = betaRS

  # check if Tobs is an array or a single value
  T_isarr = False
  F_obs = np.zeros(len(nuobs))
  if (type(Tobs) == np.ndarray):
    T_isarr = True
    F_obs = np.zeros((len(Tobs), len(nuobs)))

  # spectral shape (Band function)
  def S(x, alpha, beta):
    b  = alpha-beta
    xb = b/(1+alpha)
    def S_inf(x):
      return np.exp(1+alpha) * x**alpha * np.exp(-x*(1+alpha))
    def S_sup(x):
      return np.exp(1+alpha) * x**beta * xb**b * np.exp(-b)
    return np.where(x<=xb, S_inf(x), S_sup(x))

  # FS normalizations
  ratio_T, ratio_nu, ratio_F = ratios_RSvFS_from_au(au)
  T0FS  = T0RS / ratio_T
  nu0FS = nu0RS / ratio_nu
  F0FS  = epsrad_FS * F0RS / ratio_F

  # RS contribution
  TRS = 1 + Tobs/T0RS
  nupks_RS, nFpks_RS = peak_model(TRS, au, tau4, reverse=True)
  F_RS = np.zeros(F_obs.shape)
  nu_b = nuobs / nu0RS
  for i, (nupk, nFpk) in enumerate(zip(nupks_RS, nFpks_RS)):
    # nupks and nFpks are arrays even if TRS is a float
    Fpk = nFpk/nupk
    x = nu_b/nupk
    F = F0RS * Fpk * S(x, alphaRS, betaRS)
    if T_isarr:
      F_RS[i] = F
    else:
      F_RS = F

  # FS contribution
  TFS = 1 + Tobs/T0FS
  nupks_FS, nFpks_FS = peak_model(TFS, au, tau1, reverse=False)
  F_FS = np.zeros(F_obs.shape)
  nu_b = nuobs / nu0FS
  for i, (nupk, nFpk) in enumerate(zip(nupks_FS, nFpks_FS)):
    Fpk = nFpk/nupk
    x = nu_b/nupk
    F = F0FS * Fpk * S(x, alphaFS, betaFS)
    if T_isarr:
      F_FS[i] = F
    else:
      F_FS = F

  # Total
  F_obs = F_RS + F_FS
  return F_obs
