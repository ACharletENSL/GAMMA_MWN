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
import os
import pandas as pd
from fits_hydro import fits_from_au
from scipy.interpolate import interp1d
from scipy.optimize import brentq, least_squares
from scipy.integrate import quad

from phys_functions import *


def ratios_RSvFS_from_au(au):
  G21, G34 = relLfac_from_au(au)
  b21 = derive_velocity(G21)
  b34 = derive_velocity(G34)
  gFS2, gRS2 = g2_from_au(au)

  ratio_T = (gRS2/gFS2)
  fac1 = np.sqrt((G34/G21)*((G21+1)/(G34+1)))
  fac2 = ((G34-1)/(G21-1))**2
  ratio_nu = fac1 * fac2
  ratio_F = fac1 * (b34/b21) / fac2
  return ratio_T, ratio_nu, ratio_F

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
def peaks_model_C25_fromTf(T, au, Tf, reverse=True):
  '''
  Same as peaks_model_C25 but takes Tf instead of dR
  '''
  front, i_g = ('RS', 1) if reverse else ('FS', 0)
  g2 = g2_from_au(au)[i_g]
  popt_lfac, _, popt_nu, popt_L, popt_xi = fits_from_au(au, front)
  nupk, nFpk = peaks_fromfits(T, Tf, g2, popt_lfac, popt_nu, popt_L, popt_xi)
  return nupk, nFpk

def peaks_model_C25(T, au, dR, reverse=True):
  '''
  Return normalized peak energy and flux at peak,
    using a table of functions built with a_u 
  '''
  front, i_g = ('RS', 1) if reverse else ('FS', 0)
  g2 = g2_from_au(au)[i_g]
  popt_lfac, _, popt_nu, popt_L, popt_xi = fits_from_au(au, front)
  #Tf = 1 + dR
  Tf = Tf_from_dR(dR, popt_lfac)

  nupk, nFpk = peaks_fromfits(T, Tf, g2, popt_lfac, popt_nu, popt_L, popt_xi)
  return nupk, nFpk

def Tf_from_dR(dR, popt_lfac):
  '''
  Final observed time (normalized) from the crossed radius (normalized)
  '''
  integrand = lambda x: 1. / smooth_bpl0_apy(x, *popt_lfac)**2
  result, _ = quad(integrand, 1., 1. + dR)
  return 1. + result

def peaks_fromfits(T, Tf, g2, popt_lfac, popt_nu, popt_L, popt_xi):
  '''
  Produce F_\nu (T) from the fitting parameters
  T is normalized time tilde(T)
  '''
  func_peaks = peaks_function_fromFit(Tf, g2, popt_lfac, popt_nu, popt_L)
  nupk, nFpk = func_peaks(T, popt_xi)
  return nupk, nFpk


def peaks_function_fromFit(Tf, g2, popt_lfac, popt_nu, popt_L):

  # intermediate functions for flux calc
  def func_Gma(x):
    Gma = smooth_bpl0_apy(x, *popt_lfac)
    return Gma
  def func_nu(x):
    nuR = smooth_bpl0_apy(x, *popt_nu)
    return nuR/x
  def func_Lbol(x):
    L = smooth_bpl0_apy(x, *popt_L)
    return L
  
  def func_rise(T, k, xi_sat=10, s_eats=10):
    x_L = T
    t = np.maximum(g2*(T-1), 1e-10)
    xi_max = (t**(-s_eats) + xi_sat**(-s_eats))**(-1/s_eats)
    xi_eff = k*xi_max
    x_eff = x_L/(1+xi_eff/g2)
    Dop = func_Gma(x_eff)/(1+xi_eff)
    nupk = Dop * func_nu(x_eff)
    nFpk = 3*xi_max*Dop**4*func_Lbol(x_eff)/func_Gma(x_L)**2
    return nupk, nFpk
  
  def func_HLE(T, k, xi_sat, s_eats, nupk_f, nFpk_f):
    '''T is T/Tf'''
    Teff = (1-g2) + g2*T
    nupk = nupk_f / Teff
    nFpk = nFpk_f / Teff**3
    return nupk, nFpk
  
  def func_peaks(T, popt_xi):
    mask1 = (T <= Tf)
    mask2 = ~mask1
    nupk, nFpk = np.empty_like(T), np.empty_like(T)
    nupk[mask1], nFpk[mask1] = func_rise(T[mask1], *popt_xi)
    rise_f = np.array(func_rise(Tf, *popt_xi))
    nupk_f, nFpk_f = rise_f[0], rise_f[1]
    HLE_f = np.array([func_HLE(Tf, *popt_xi, nupk_f, nFpk_f)])
    nupk[mask2], nFpk[mask2] = func_HLE(T[mask2]/Tf, *popt_xi, nupk_f, nFpk_f)
    return nupk, nFpk
  return func_peaks


