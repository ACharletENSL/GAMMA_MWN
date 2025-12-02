# -*- coding: utf-8 -*-
# @Author: acharlet

'''
Modeling peak frequency - peak flux correlation from internal shock collision
following Rahaman et al. 2024a,b and Charlet et al. 2025
'''

import numpy as np
from phys_functions import smooth_bpl

def g2_from_au(au):
  '''
  Ratios of downstream L.F. to shock front L.F., squared
  '''

  # R24b (I1), ratio between \Gamma and \Gamma_1
  K = np.sqrt(2)*au / np.sqrt(1 + au*au)

  # relative LF in UR regime (Sari & Piran 95, Eq 3)
  G21 = .5 * (1./K + K)
  G34 = .5 * (au/K + K/au)

  # plug this in R24a eqns (C2) and (C5)
  # use UR approx beta = 1 - 1/(2*Gamma**2)
  qF = 1/(4*G21*K)
  gFS2 = (K*K*qF - 1)/(qF - 1)
  qR = 4*G34*K/au
  gRS2 = (qR - K*K/(au*au))/(qR - 1)
  return gFS2, gRS2
  

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


def peaks_model_R24(T, au, dR, reverse=True):
  '''
  Peak frequency and flux with normalized observed time \tilde{T}
  collision of ultrarelativistic cold shells, R24b eqns (9) - (10)
  parameters:
    T:          \tilde[T} = (T - T_ej,eff)/T_R,, normalized observed time
      type np.ndarray, replace "np.where" by if/else if you prefer a float as input
    au:         a_u, proper velocity contrast
    dR:         \Delta R_f/R_0, distance crossed by the shock front, units R_0
    reverse:    boolean, True if we model a reverse shock
  '''

  # downstream L.F. to shock front L.F. ratio
  g2 = g2_from_au(au)[1 if reverse else 0]

  # normalized times 
  # T1/2: T_eff,1/2
  Tf = 1. + dR
  T1 = (1. - g2) + g2 * T
  T1f = (1. - g2) + g2 * Tf
  T2 = (1. - g2) + g2 * T / Tf

  # model functions - rising part
  def nu_rise(T):
    return 1./T
  def nuFnu_rise(T):
    return 1. - (1./(T*T*T))

  # model functions - HLE
  nu_f = nu_rise(T1f)
  nF_f = nuFnu_rise(T1f)
  def nu_HLE(T):
    return nu_f/T
  def nuFnu_HLE(T):
    return nF_f/(T*T*T)
  
  rise = (T <= Tf)
  nu = np.where(rise, nu_rise(T1), nu_HLE(T2))
  nF = np.where(rise, nuFnu_rise(T1), nuFnu_HLE(T2))

  return nu, nF

def peaks_model_plaws(T, au, dR, m, n, k=0.5, reverse=True):
  '''
  Peak frequency and flux with normalized observed time \tilde{T}
  collision of ultrarelativistic cold shells
  Generalized version with variable LF and shock strength
  parameters:
    T:          \tilde[T} = (T - T_ej,eff)/T_R, normalized observed time
      ! be careful with normalization, radial and angular time are different now ! 
      type np.ndarray, replace "np.where" by if/else if you prefer a float as input
    au:         a_u, proper velocity contrast
    dR:         \Delta R_f/R_0, distance crossed by the shock front, units R_0
    m:          power-law index, \Gamma^2 \propto R^-m
    n:          power-law index, L'_bol \propto R^-n
    k:          effective angle xi_eff over maximal contributing angle of EATS xi_max
  
  for now, without convergence to geometrical scalings (m=0, n=0 after a given radius)
  '''

  # power-law indices for frequency and luminosity
  a =  1. + 1.5*n
  d = -(1. + 2.5*n)

  # shock front L.F. to downstream L.F. ratio
  g2 = g2_from_au(au)[1 if reverse else 0]
  g2m = g2/(m+1)

  # effective angle
  Tsat = 1 + 1/(k*g2m)
  xi_eff = np.where(T<=Tsat, k*g2m*(T-1), 1.)
  y_eff = np.where(T<=Tsat, 1+k*(T-1), 1+1/g2m)**(-1./(m+1))

  # crossing time
  Tf = (1 + dR) #**(m+1)

  # model functions - rising part
  def nu_rise(T):
    return (T**d) * y_eff**(d-0.5*m) / (1+xi_eff)
  def Fnu_rise(T):
    ym1 = y_eff**(m+1)
    term1 = (3*g2*xi_eff/(k*(1+xi_eff)**2))
    term2 = T**(a-m/(2*(m+1))) * y_eff**(a-1-0.5*m)
    term3 = (1+m*ym1)/(g2+(m+1-g2)*ym1)
    return term1*term2*term3
  def nuFnu_rise(T):
    return nu_rise(T) * Fnu_rise(T)

  # model functions - HLE
  nu_f = nu_rise(Tf)
  nF_f = nuFnu_rise(Tf)
  def nu_HLE(T):
    return nu_f/T
  def nuFnu_HLE(T):
    return nF_f/(T*T*T)
  
  rise = (T <= Tf)
  nu = np.where(rise, nu_rise(T), nu_HLE(T/Tf))
  nF = np.where(rise, nuFnu_rise(T), nuFnu_HLE(T/Tf))

  return nu, nF

def peaks_model_C25(T, au, dR,
    m, lfac2_sph, n, Sh_sph,
    k=0.5, xi_effmax=2., s=2., reverse=True):
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
    xi_effmax:  max value of k*xi_max (order unity)
    s:          smoothing parameter
  '''

  ########################## upcoming  ########################## 
  ### lookup table (or empirical law) for m, n, and the _sph vars
  ############################################################### 

  # shock front L.F. to downstream L.F. ratio
  g2 = g2_from_au(au)[1 if reverse else 0]
  g2m = g2/(m+1)

  # power-law indices and (normalized) large R values
  #    for frequency and luminosity
  a =  1. + 1.5*n
  d = -(1. + 2.5*n)
  lfac_sph = np.sqrt(lfac2_sph)
  nu_sph = Sh_sph**(1.5)
  L_sph = 1/np.sqrt(Sh_sph)


  # Crossing time
  Tf = (1. + dR)

  def effective_angle(T):
    # EATS size, contributing angle, effective angle and radius
    EATS = g2m * (T - 1.0)
    xi_sat = xi_effmax/k
    xi_max = (EATS**(-s) + xi_sat**(-s))**(-1.0/s)
    xi_eff = k*xi_max
    y_eff  = (1. + xi_eff/g2m)**(-1/(m+1))
    return xi_max, xi_eff, y_eff

  # Rising part of the observed flux 
  def nuFnu_rise(T):
    xi_max, xi_eff, y_eff = effective_angle(T)
    r_eff = y_eff*T

    # LF, comoving frequency, and luminosity at effective radius, normalized
    Gamma_d = smooth_bpl(r_eff, lfac_sph, -0.5*m, 0.0, s)
    Gamma_d_RL = smooth_bpl(T, lfac_sph, -0.5*m, 0.0, s)
    nu_prime = smooth_bpl(r_eff, nu_sph, d, -1.0, s)
    L_prime = smooth_bpl(r_eff, L_sph, a, 1.0, s)

    # peak frequency and flux (up to crossing time)
    Dop = Gamma_d / (1 + xi_eff)
    nu_pk = Dop * nu_prime
    Fnu_pk = g2 * xi_max * (Dop**3) * L_prime / (Gamma_d_RL**2)
    return nu_pk, Fnu_pk
  
  # High latitude emission
  def nuFnu_HLE(T):
    nu_pkf, Fnu_pkf = nuFnu_rise(Tf)
    _, _, y_eff_f =  effective_angle(Tf)
    r_f = y_eff_f * Tf
    Gamma_f = smooth_bpl(r_f, lfac_sph, -0.5*m, 0.0, s)
    T_hle  = 1. + (T - 1.) * (Gamma_f**2)
    T_hle0 = 1. + (Tf - 1.) * (Gamma_f**2)
    T_hle /= T_hle0
    #T_hle = T/Tf
    nu_pk = nu_pkf / T_hle
    Fnu_pk = Fnu_pkf / (T_hle**2)
    return nu_pk, Fnu_pk

  nu_pk, Fnu_pk = np.where(T<=Tf, nuFnu_rise(T), nuFnu_HLE(T))
  return nu_pk, Fnu_pk
