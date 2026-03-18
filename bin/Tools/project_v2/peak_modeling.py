# -*- coding: utf-8 -*-
# @Author: acharlet

'''
Modeling peak frequency and peak flux from internal shock collision
following Rahaman et al. 2024a,b and Charlet et al. 2025
Contains:
  - g2_from_au
  - relLfac_from_au
  - tontoff_to_dRR0_R24
  - peaks_model_R24
  - peaks_model_plaws (not needed, to remove)
  - peaks_model_C25
  - freqs_model_C25
  - effective_angle
  - peaks_rise
  - peaks_HLE
'''

import numpy as np
from phys_functions import smooth_bpl, smooth_bpl0
from scipy.optimize import brentq, least_squares

def g2_from_au(au):
  '''
  Ratios of downstream L.F. to shock front L.F., squared
  '''

  # R24b (I1), ratio between \Gamma and \Gamma_1
  K = np.sqrt(2)*au / np.sqrt(1 + au*au)

  G21, G34 = relLfac_from_au(au)

  # plug this in R24a eqns (C2) and (C5)
  # use UR approx beta = 1 - 1/(2*Gamma**2)
  qF = 1/(4*G21*K)
  gFS2 = (K*K*qF - 1)/(qF - 1)
  qR = 4*G34*K/au
  gRS2 = (qR - K*K/(au*au))/(qR - 1)
  return gFS2, gRS2

def relLfac_from_au(au):
  '''
  Relative Lorentz factors at each shock front
  '''
  # R24b (I1), ratio between \Gamma and \Gamma_1
  K = np.sqrt(2)*au / np.sqrt(1 + au*au)

  # relative LF in UR regime (Sari & Piran 95, Eq 3)
  G21 = .5 * (1./K + K)
  G34 = .5 * (au/K + K/au)
  return G21, G34


def tontoff_to_dRR0_R24(t_ratio, au, reverse=True):
  '''
  Converts the crossed radius by the shock in units R0
    into the ratio of activity time over off time for the source
  R24b eqn (I1)
  '''
  au2 = au*au
  g2 = g2_from_au(au)[1 if reverse else 0]
  dR = 2 * ((au2 - 1)/((au2 + 1)*g2 - 2*au))
  return dR


def peaks_model_R24(T, au, dR, g2):
  '''
  Peak frequency and flux with normalized observed time \tilde{T}
  collision of ultrarelativistic cold shells, R24b eqns (9) - (10)
  parameters:
    T:          \tilde[T} = (T - T_ej,eff)/T_R,, normalized observed time
      type np.ndarray, replace "np.where" by if/else if you prefer a float as input
    au:         a_u, proper velocity contrast
    dR:         \Delta R_f/R_0, distance crossed by the shock front, units R_0
    g2:         downstream L.F. to shock front L.F. ratio
  '''

  # normalized times 
  # T1/2: T_eff,1/2
  Tf = 1. + dR
  T1 = (1. - g2) + g2 * T
  T1f = (1. - g2) + g2 * Tf
  T2 = (1. - g2) + g2 * T / Tf

  # model functions - rising part
  def nu_rise(T):
    return 1./T
  def peaks_rise(T):
    return 1. - (1./(T*T*T))

  # model functions - HLE
  nu_f = nu_rise(T1f)
  nF_f = peaks_rise(T1f)
  def nu_HLE(T):
    return nu_f/T
  def peaks_HLE(T):
    return nF_f/(T*T*T)
  
  rise = (T <= Tf)
  nu = np.where(rise, nu_rise(T1), nu_HLE(T2))
  nF = np.where(rise, peaks_rise(T1), peaks_HLE(T2))

  return nu, nF

##### From fitted functions
# end goal is flux_fromFit(au, DeltaR/R0)
def peaks_fromfits(T, Tf, g2, popt_lfac2, popt_nu, popt_L, popt_xi):
  '''
  Produce F_\nu (T) from the fitting parameters
  T is normalized time tilde(T)
  '''
  k, xi_sat, s = popt_xi
  func_peaks = peaks_function_fromFit(Tf, g2, popt_lfac2, popt_nu, popt_L)
  nupk, nFpk = func_peaks(T, *popt_xi)
  return nupk, nFpk


def peaks_function_fromFit(Tf, g2, popt_lfac2, popt_nu, popt_L):

  # intermediate functions for flux calc
  def func_Gma(x):
    Gma2 = smooth_bpl0(x, *popt_lfac2)
    Gma = np.sqrt(Gma2)
    return Gma
  def func_nu(x):
    nuR = smooth_bpl0(x, *popt_nu)
    return nuR/x
  def func_Lbol(x):
    L = smooth_bpl0(x, *popt_L)
    return L
  
  def func_rise(T, k, xi_sat, s):
    x_L = T
    t = g2*(T-1)
    xi_max = (t**(-s) + xi_sat**(-s))**(-1/s)
    xi_eff = k*xi_max
    x_eff = x_L/(1+xi_eff/g2)
    Dop = func_Gma(x_eff)/(1+xi_eff)
    nupk = Dop * func_nu(x_eff)
    nFpk = 3*xi_max*Dop**4*func_Lbol(x_eff)/func_Gma(x_L)**2
    return nupk, nFpk
  
  def func_HLE(T, k, xi_sat, s, nupk_f, nFpk_f):
    '''T is T/Tf'''
    Teff = (1-g2) + g2*T
    nupk = nupk_f / Teff
    nFpk = nFpk_f / Teff**3
    return nupk, nFpk
  

  def func_peaks(T, k, xi_sat, s):
    mask1 = (T <= Tf)
    mask2 = ~mask1
    nupk, nFpk = np.empty_like(T), np.empty_like(T)
    nupk[mask1], nFpk[mask1] = func_rise(T[mask1], k, xi_sat, s)
    rise_f = np.array(func_rise(Tf, k, xi_sat, s))
    nupk_f, nFpk_f = rise_f[0], rise_f[1]
    HLE_f = np.array([func_HLE(Tf, k, xi_sat, s, nupk_f, nFpk_f)])
    nupk[mask2], nFpk[mask2] = func_HLE(T[mask2]/Tf, k, xi_sat, s, nupk_f, nFpk_f)
      #+ rise_f - HLE_f[:, np.newaxis]
    return nupk, nFpk

  return func_peaks

def peaks_model_C25(T, au, dR,
    m, lfac2_sph, n, Sh_sph,
    k=0.5, xi_sat=2., s=2.):
  '''
  Peak frequency and flux with normalized observed time \tilde{T}
  based on the fitting method in Charlet et al. 2025
  parameters:
    T:          \tilde[T} = (T - T_ej,eff)/T_R, normalized observed time
      ! be careful with normalization, radial and angular time are different now ! 
      type np.ndarray, replace "np.where" by if/else if you prefer a float as input
    au:         a_u, proper velocity contrast
    dR:         distance crossed by the shock front, in units R0
  additional parameters (will be replaced by a lookup table)
    m:          power-law index, \Gamma^2 \propto R^-m
    lfac2_sph:  large R/R0 value of (\Gamma/\Gamma_0)^2
    n:          power-law index, L'_bol \propto R^-n
    Sh_sph:     large R/R0 value of \Gamma_{ud-1} (normalized to value at R0)
  optional parameters:
    k:          effective angle xi_eff over maximal contributing angle of EATS xi_max
    xi_sat:     max value of xi_max (order unity)
    s:          smoothing parameter
  '''

  ########################## upcoming  ########################## 
  ### lookup table (or empirical law) for m, n, and the _sph vars
  ############################################################### 

  # shock front L.F. to downstream L.F. ratio
  g2 = g2_from_au(au)[1]
  g2m = g2/(m+1)

  # Crossing time
  Tf = (1. + dR)

  # Rise and HLE with T as function of the input parameters
  def func_rise(T):
    return peaks_rise(T, g2, m, n, lfac2_sph, Sh_sph, k, xi_sat, s)
  _, _, y_eff_f =  effective_angle(Tf, g2, m, k, xi_sat, s)
  nu_pkf, Fnu_pkf = func_rise(Tf)
  def func_HLE(T):
    return peaks_HLE(T, Tf, g2, y_eff_f, nu_pkf, Fnu_pkf)

  rise = (T <= Tf)
  nu_pk, Fnu_pk = np.where(rise, func_rise(T), func_HLE(T))

  return nu_pk, Fnu_pk

def freqs_model_C25(au, dR,
    m, lfac2_sph, n, Sh_sph,
    k=0.145, xi_sat=5., s=2.):
  '''
  Get the \nu_bk / \nu_{1/2} as a function of \Delta R / R0
  '''
  g2 = g2_from_au(au)[1]
  Tf = (1 + dR)

  def func_nuFnu(T):
    nu_pk, Fnu_pk = peaks_rise(T, g2, m, n, lfac2_sph, Sh_sph, k, xi_sat, s)
    return nu_pk * Fnu_pk

  nu_bk, Fnu_bk = peaks_rise(Tf, g2, m, n, lfac2_sph, Sh_sph, k, xi_sat, s)
  nF_bk = nu_bk*Fnu_bk

  def func_root(T):
    return func_nuFnu(T) - 0.5*nF_bk
  T_hf = brentq(func_root, 1, 1.5*Tf)

  nu_hf, F_hf = peaks_rise(T_hf, g2, m, n, lfac2_sph, Sh_sph, k, xi_sat, s)
  return nu_bk, nu_hf



#### Intermediate quantities for peak frequency and flux
def effective_angle(T, g2, m, k, xi_sat, s):
  '''EATS size, contributing angle, effective angle and radius'''
  g2m = g2/(m+1)
  EATS = g2m * (T - 1.0)
  xi_max = (EATS**(-s) + xi_sat**(-s))**(-1.0/s)
  xi_eff = k*xi_max
  y_eff  = (1. + xi_eff/g2m)**(-1/(m+1))
  return xi_max, xi_eff, y_eff

def peaks_rise(T, g2, m, n, lfac2_sph, Sh_sph, k, xi_sat, s):
  '''Rising part of the observed flux'''

  # power-law indices and (normalized) large R values
  #    for frequency and luminosity
  a =  1. + 1.5*n
  d = -(1. + 2.5*n)
  lfac_sph = np.sqrt(lfac2_sph)
  nu_sph = Sh_sph**(1.5)
  L_sph = 1/np.sqrt(Sh_sph)
  
  xi_max, xi_eff, y_eff = effective_angle(T, g2, m, k, xi_sat, s)
  r_eff = y_eff * T

  # LF, comoving frequency, and luminosity at effective radius, normalized
  Gamma_d = smooth_bpl0(r_eff, lfac_sph, -0.5*m, s)
  Gamma_d_RL = smooth_bpl0(T, lfac_sph, -0.5*m, s)
  nu_prime = smooth_bpl(r_eff, nu_sph, d, -1.0, s)
  L_prime = smooth_bpl(r_eff, L_sph, a, 1.0, s)

  # peak frequency and flux (up to crossing time)
  Dop = Gamma_d / (1 + xi_eff)
  nu_pk = Dop * nu_prime
  # factor 3 absorbed in normalization
  Fnu_pk = g2 * xi_max * (Dop**3) * L_prime / (Gamma_d_RL**2)
  return nu_pk, Fnu_pk
  
def peaks_HLE(T, Tf, g2, y_eff_f, nu_pkf, Fnu_pkf):
  '''High latitude emission'''
  r_f = y_eff_f * Tf
  T_hle = (1 - g2) + g2 * T/Tf
  nu_pk = nu_pkf / T_hle
  Fnu_pk = Fnu_pkf / (T_hle**2)
  return nu_pk, Fnu_pk


