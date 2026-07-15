# -*- coding: utf-8 -*-
# @Author: acharlet

'''
Modeling peak frequency - peak flux correlation from internal shock collision
This model assumes constant shock (and downstream) LF, and constant dissipation rate
Contains:
  - peak_model
  - correlation_model 
  (not useful but it is fully analytic, worth keeping for reference)
'''

import numpy as np
from analytical_funcs import *

def peak_model(T, au, tau, reverse=True, renormFlux=True):
  '''
  Peak frequency and flux at this frequency with observed time T
  T:        tilde{T}, time in observers frame normalized to angular time
  au:       a_u, proper velocity contrast
  tau:      tau_i=t_on,i/t_off, source activity time during emission of shell i
  reverse:  boolean, True to model contribution from the RS, False for the FS
  '''

  i = 0 if reverse else 1
  # downstream L.F. to shock front L.F. ratio
  g2 = g2_from_au(au)[i]

  # break time
  K = tau_to_dRR0(au)[i]
  dR = K * tau
  Tf = 1 + dR
  T1f = (1. - g2) + g2 * Tf

  # model functions
  ### rising part
  def nu_rise(x):
    return 1./x
  def nF_rise(x):
    return 1. - 1/x**3
  ### HLE
  nu_f = nu_rise(Tf)
  nF_f = nF_rise(T1f)
  def nu_HLE(x):
    return nu_f/x
  def nF_HLE(x):
    return nF_f/x**3
  
  rise = (T <= Tf)
  T1 = (1. - g2) + g2 * T
  T2 = (1. - g2) + g2 * T / Tf
  nu_pk = np.where(rise, nu_rise(T), nu_HLE(T2))
  nF_pk = np.where(rise, nF_rise(T1), nF_HLE(T2))
  if renormFlux:
    # max over the track is nF at the break (= nF_f), where rise meets HLE
    nF_pk = nF_pk / nF_f
  return nu_pk, nF_pk

def correlation_model(nu_pks, au, tau, reverse=True, renormFlux=True):
  '''
  Peak flux vs peak frequency, normalized, from the collision of UR shells
  parameters:
    nu_pks:     normalized peak observed frequency
      type np.ndarray, replace "np.where" by if/else if you prefer a float as input
    au:         a_u, proper velocity contrast
    tau:        tau=t_on,i/t_off, source activity time during emission of shell i 
                  over the time between the emission of the two shells
    reverse:    boolean, True to model contribution from the RS, False for the FS
  '''

  i = 0 if reverse else 1
  # downstream L.F. to shock front L.F. ratio
  g2 = g2_from_au(au)[i]

  # break frequency
  K = tau_to_dRR0(au)[i]
  dR = K * tau
  Tf = 1 + dR
  nu_bk = 1/Tf

  # model functions
  rise = (nu_pks >= nu_bk)
  def flux_rise(nu):
    return 1 - ((1-g2) + g2/nu)**-3
  def flux_HLE(nu):
    nF_bk = flux_rise(nu_bk)
    return nF_bk * (nu/nu_bk)**3
  nF = np.where(rise, flux_rise(nu_pks), flux_HLE(nu_pks))
  if renormFlux:
    # max over the track is reached at the break (rise increases, HLE decays)
    nF /= flux_rise(nu_bk)
  return nF