# -*- coding: utf-8 -*-
# @Author: acharlet

'''
This file contains functions to derive physical variables
/!\ IN NATURAL/CODE UNITS /!\ 
'''

# Imports
# --------------------------------------------------------------------------------------------------
import numpy as np
import math
from phys_constants import *
#from numba import njit

# Functions
# --------------------------------------------------------------------------------------------------
def find_closest_old(A, target):
  #A must be sorted
  idx = np.searchsorted(A, target)
  idx = np.clip(idx, 1, len(A)-1)
  left = A[idx-1]
  right = A[idx]
  idx -= target - left < right - target
  return idx

#@njit
def find_closest(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])):
        return (idx - 1)
    else:
        return (idx)

def derive_reldiff(a, b):
  '''
  Relative difference
  '''
  return (a-b)/b

def derive_resolution(x, dx):
  '''
  Resolution
  '''
  return dx/x

def prim2cons(rho, u, p):
  '''
  Primitive to conservative
  /!\ in code units, and u is proper velocity
  '''
  lfac = derive_Lorentz_from_proper(u)
  D = lfac*rho
  h = derive_enthalpy(rho, p)
  tau = D*h*lfac-p-D
  s = D*h*u

  return D, s, tau

# pdV work
def pdV_rate(v, p, A):
  '''
  Rate of pdV work across a surface A
  '''
  return p*A*v

# EoS and related
def derive_temperature(rho, p):
  '''
  Relativistic temperature from density and pressure
  '''
  return p/rho

def derive_enthalpy(rho, p, EoS='TM'):
  '''
  Enthalpy from density and pressure
  '''
  T = derive_temperature(rho, p)

  if EoS == 'TM':
    gma = derive_adiab_fromT_TM(T)
  elif EoS == 'Ryu':
    gma = derive_adiab_fromT_Ryu(T)
  elif EoS == 'Synge':
    gma = derive_adiab_fromT_Synge(T)
  else:
    gma = 4./3.
  return 1. + T*gma/(gma-1.)

def derive_adiab_fromT_Ryu(T):
  '''
  Adiabatic index from temperature, following Ryu et al 2006 EoS
  '''
  a = 3*T + 1
  return (4*a+1)/(3*a)

def derive_adiab_fromT_TM(T):
  '''
  Adiabatic index from temperature, following Taub Matthews EoS
  '''
  gamma_eff = (1./6.)*(8. - 3.*T + np.sqrt(4. + 9.*T*T))
  return gamma_eff

def derive_adiab_fromT_Synge(T):
  '''
  Adiabatic index from temperature, following Ryu et al 2006 EoS
  '''
  gma = 5./3.
  a = T/(gma-1.)
  e_ratio = a + np.sqrt(a*a+1.)
  gma_eff = gma - (gma-1.)/2. * (1.-1./(e_ratio**2))
  return gma_eff

def derive_enthalpy_fromT_TM(T):
  return 2.5*T + np.sqrt(1. + 2.25*T**2)

def derive_cs2_fromT_TM(T):
  h = derive_enthalpy_fromT_TM(T)
  num = T*(5.*h-8.*T)
  denom = 3.*h*(h-T)
  return num/denom

def derive_cs(rho, p, EoS='TM'):
  '''
  Sound speed
  '''
  T = derive_temperature(rho, p)
  c2 = T*(3*T+2)*(18*T**2+24*T+5) / (3*(6*T**2+4*T+1)*(9*T**2+12*T+2))
  return np.sqrt(c2)

def derive_cs2_fromT(T, EoS='TM'):
  '''
  Sound speed from relativistic temperature
  Expressions from Ryu et al. 2006 
  '''

  if EoS == 'TM':
    cs2 = derive_cs2_fromT_TM(T)
  elif EoS == 'RC':
    num = T*(3.*T+2)*(18.*T**2+24.*T+5)
    denom = 3.*(6.*T**2+4.*T+1.)*(9.*T**2+12.*T+2.)
    cs2 = num/denom
  else:
    print('Implement this EoS')
    cs2 = 1./3.
  return cs2

def derive_cs_fromT(T, EoS='TM'):
  cs2 = derive_cs2_fromT(T, EoS)
  return np.sqrt(cs2)

def derive_Eint(rho, v, p):
  '''
  Internal energy density in lab frame
  '''
  T = derive_temperature(rho, p)
  lfac = derive_Lorentz(v)
  gma  = derive_adiab_fromT_TM(T)
  eint = p/(gma-1.)  # in comoving frame
  return eint*lfac**2*(1+v**2*(gma-1.))

def derive_Eint_comoving(rho, p, rhoscale):
  '''
  Internal energy density in comoving frame
  '''
  rho = rho*rhoscale*c_**2
  p = p*rhoscale*c_**2
  T = derive_temperature(rho, p)
  gma  = derive_adiab_fromT_TM(T)
  return p/(gma-1.)

def derive_B_comoving(rho, p, rhoscale, eps_B=1/3.):
  e = derive_Eint_comoving(rho, p, rhoscale)
  return np.sqrt(8*pi_*eps_B*e)

# Velocity and related
def derive_Lorentz(v):
  '''
  Lorentz factor from velocity (beta)
  '''
  return 1./np.sqrt(1 - v**2)

def derive_velocity(lfac):
  '''
  Velocity (beta) from Lorentz factor
  '''
  return np.sqrt(1. - lfac**-2)

def derive_proper(v):
  '''
  Proper velocity from velocity (beta)
  '''
  return v/np.sqrt(1 - v**2)

def derive_Lorentz_from_proper(u):
  '''
  Lorentz factor from proper velocity
  '''
  return np.sqrt(1+u**2)

def derive_proper_from_Lorentz(lfac):
  '''
  Proper velocity from Lorentz factor
  '''
  return np.sqrt(lfac**2 - 1)

def derive_velocity_from_proper(u):
  '''
  Velocity in rest frame from proper velocity
  '''
  return u/np.sqrt(1+u**2)

def derive_Ekin(rho, v):
  '''
  Kinetic energy in lab frame from density and velocity
  '''
  lfac = derive_Lorentz(v)
  return (lfac-1)*lfac*rho

def derive_Ekin_fromproper(rho, u):
  '''
  Kinetic energy in lab frame from density and proper velocity
  '''
  lfac = derive_Lorentz_from_proper(u)
  return rho*(lfac-1)*lfac

def lfac2nu(lfac, B):
  '''
  Synchrotron frequency of an electron with Lorentz factor lfac in magn field B (comoving)
  '''

  fac = 3*e_/(4.*pi_*me_*c_)
  return fac * lfac**2 * B

def nu2lfac(nu, B):
  '''
  Lorentz factor of an electron emitting photon of frequency nu in magn field B
  '''

  fac = 3*e_/(4.*pi_*me_*c_)
  return (fac * B)**-0.5

def Hz2eV(nu):
  return h_eV_*nu

def derive_nu(rho, p, lfac, rhoscale, eps_B=1/3.):
  '''
  Synchrotron frequency emitted by electron at Lorentz factor lfac
  '''
  B = derive_B_comoving(rho, p, rhoscale, eps_B)
  nu = lfac2nu(lfac, B)
  return nu

def derive_normTime(r, beta, mu=1):
  '''
  Normalization time Ttheta
  '''
  return (1-beta*mu)*r

def derive_max_emissivity(rho, p, gmin, gmax, rhoscale, eps_B, psyn, xi_e):
  '''
  Returns maximal emissivity used for normalisation, from Ayache et al. 2022
  '''
  
  B = derive_B_comoving(rho, p, rhoscale, eps_B)
  n = xi_e*derive_n(rho, rhoscale)*derive_xiDN(gmin, gmax, psyn)
  fac = (4./3.) * (4*(psyn-1)/(3*psyn-1)) * (16*me_*c_**2*sigT_/(18*pi_*e_))
  return B*n*fac

def derive_emiss_prefac(psyn):
  return (4*(psyn-1)/(3*psyn - 1)) * sigT_ * (4/3) * (8*me_*c_/(9*pi_*e_))

def derive_xiDN(gmin, gmax, p):
  '''
  Fraction of emitting electrons
  '''
  def xiDN(gmin, gmax, p):
    return ((gmax**(2-p) - gmin**(2-p))/(gmax**(2-p)-1))*((gmax**(1-p)-1.)/(gmax**(1-p)-gmin**(1-p)))
  
  return np.where(gmax<1., 0., np.where(gmin<1., xiDN(gmin, gmax, p), 1.))


def derive_n(rho, rhoscale):
  '''
  Comoving number density
  '''
  return rho*rhoscale/mp_

def derive_Psyn_electron(lfac, B):
  '''
  Synchrotron power emitted by an electron of Lorentz factor lfac in a field B
  '''
  u = derive_proper_from_Lorentz(lfac)
  return (sigT_*c_/(4.*pi_)) * u**2 * B**2

def derive_3volume(x, dx, R0, geometry='cartesian'):
  '''
  3-volume of a cell
  '''
  r = x*c_
  dV = 2*dx*c_
  if geometry == 'cartesian':
    dV *= 2*pi_*R0**2
  else:
    dV *= 2*pi_*r**2
  return dV

def derive_4volume(x, dx, dt, R0, geometry='cartesian'):
  '''
  4-volume of a cell
  '''
  dV = derive_3volume(x, dx, R0, geometry)
  V4 = dV * dt
  return V4

def derive_rad4volume(x, dx, rho, vx, p, dt, gma, rhoscale, eps_B, R0, geometry='cartesian'):
  '''
  4 volume of a cell considering cooling time at the peak comoving frequency
  '''
  dV = derive_3volume(x, dx, R0, geometry)
  tcool = derive_tcool(rho, vx, p, gma, rhoscale, eps_B)
  delt = np.where(tcool<dt, tcool, dt)
  return dV * delt

def derive_tcool(rho, vx, p, gma, rhoscale, eps_B):
  '''
  Synchrotron cooling time of an electron with lfac gma in source frame
  '''
  tpcool = derive_tcool_comoving(rho, p, gma, rhoscale, eps_B)
  lfac = derive_Lorentz(vx)
  return lfac*tpcool

def derive_tcool_comoving(rho, p, gma, rhoscale, eps_B):
  '''
  Synchrotron cooling time of an electron with lfac gma in comoving frame
  '''
  eint = derive_Eint_comoving(rho, p, rhoscale)
  return 3*me_*c_/(4*sigT_*eint*eps_B*gma)

def derive_localLum(rho, vx, p, x, dx, dt, gmin, gmax, gma, rhoscale, eps_B, psyn, xi_e, R0, geometry='cartesian'):
  '''
  Local peak luminosity
  '''
  V4 = derive_rad4volume(x, dx, rho, vx, p, dt, gma, rhoscale, eps_B, R0, geometry)
  P = derive_max_emissivity(rho, p, gmin, gmax, rhoscale, eps_B, psyn, xi_e)
  lfac = derive_Lorentz(vx)
  Lp = lfac * V4 * P / x
  return Lp

def derive_obsLum(rho, vx, p, x, dx, dt, gmin, gmax, gma, rhoscale, eps_B, psyn, xi_e, R0, geometry='cartesian'):
  '''
  Local peak luminosity in observer frame
  '''
  Lp = derive_localLum(rho, vx, p, x, dx, dt, gmin, gmax, gma, rhoscale, eps_B, psyn, xi_e, R0, geometry='cartesian')
  lfac = derive_Lorentz(vx)
  return 2*lfac*Lp

def derive_obsEjTime(t, r, beta, t0, z):
  Tej, T0 = derive_obsTimes(t, r, beta, t0, z)
  return Tej

def derive_obsNormTime(t, r, beta, t0, z):
  Tej, T0 = derive_obsTimes(t, r, beta, t0, z)
  return T0

def derive_obsPkTime(t, r, beta, t0, z):
  '''
  Returns observer time at which the emission from this cell will peak
  '''
  Tej, T0 = derive_obsTimes(t, r, beta, t0, z)
  return 2*T0 + Tej

def derive_obsTimes(t, r, beta, t0, z):
  '''
  Derive the relevant times in observer frame to calculate flux
  Tej : effective ejection time of the cell
  T0  : onset to peak time
  /!\ r in units c /!\ (function to be used by data_IO:get_variable)
  '''
  t = t + t0
  tej = t - r/beta
  T0 = (1+z)*((1-beta)/beta)*r
  Tej = (1+z)*tej
  return Tej, T0


# wave speed at interfaces
def waveSpeedEstimates(SL, SR):
  '''
  Estimates wave speeds (Riemannian fan) at an interface between two cells of fluid states S (SL on left, SR on right)
  '''
  gma = 4./3. # function uses the default GAMMA and doesn't rederive the "real" value even in the case of a variable adiab. index
  gma1 = gma/(gma-1.)
  vL = SL['vx']
  vR = SR['vx']
  lfacL = derive_Lorentz(vL)
  lfacR = derive_Lorentz(vR)
  rhoL = SL['rho']
  rhoR = SR['rho']
  pL = SL['p']
  pR = SR['p']
  mmL = SL['sx']
  mmR = SR['sx']
  EL = SL['tau'] + SL['D']
  ER = SR['tau'] + SR['D']

  # sound velocity and related sigma parameter (Mignone 2006 eqn. 4 & 22-23)
  cSL = np.sqrt(gma * pL / (rhoL + pL*gma1))
  cSR = np.sqrt(gma * pR / (rhoR + pR*gma1))
  sgSL = cSL * cSL / (lfacL * lfacL * (1. - cSL * cSL))
  sgSR = cSR * cSR / (lfacR * lfacR * (1. - cSR * cSR))

  # L and R speeds
  l1 = (vR - np.sqrt(sgSR * (1. - vR * vR + sgSR))) / (1. + sgSR)
  l2 = (vL - np.sqrt(sgSL * (1. - vL * vL + sgSL))) / (1. + sgSL)
  try:
    lL = min(l1,l2)
  except ValueError:
    print(l1, l2)

  l1 = (vR + np.sqrt(sgSR * (1. - vR * vR + sgSR))) / (1. + sgSR)
  l2 = (vL + np.sqrt(sgSL * (1. - vL * vL + sgSL))) / (1. + sgSL)
  lR = max(l1,l2)

  # temporary variables (Mignone 2006 eqn 17)
  AL = lL * EL - mmL
  AR = lR * ER - mmR
  BL = mmL * (lL - vL) - pL
  BR = mmR * (lR - vR) - pR

  FhllE = (lL * AR - lR * AL) / (lR - lL)
  Ehll  = (AR - AL) / (lR - lL)
  Fhllm = (lL * BR - lR * BL) / (lR - lL)
  mhll  = (BR - BL) / (lR - lL)

  if np.abs(FhllE) == 0:
    lS = mhll / (Ehll + Fhllm)
  else:
    delta = (Ehll + Fhllm)**2 - 4. * FhllE * mhll
    lS = ((Ehll + Fhllm) - np.sqrt(delta)) / (2. * FhllE)

  return lL, lR, lS

# Shock detection
def detect_shocks(fs1, fs2, reverse=False):
  '''
  Given two neighboring fluid states fs1, fs2 and detect if there is a shock
  fs1 and fs2 are rows of a pandas dataframe
  From Rezzola&Zanotti2013, implemented in GAMMA: radiation_sph.cpp
  '''
  # 1S1R Rezzolla&Zanotti2013 eq. 4.211 (p238)
  # ------------------------------------------
  # this equation involves numerically solving an integral over pressure
  n_evals = 10    # number of points in the integral
  chi = 0         # param for strength of shocks to detect (0:weak, 1:strong)

  p1 = fs1['p']
  p2 = fs2['p']

  if (np.abs((p1-p2)/p1)<1.e-10): return(-1)  # no shock possible
  #if (p1 < p2): return (-1)                  # shock increases pressure, commented because it would rule out FS

  delta_p = p2 - p1
  dp = delta_p / n_evals

  rho1 = fs1['rho']
  rho2 = fs2['rho']
  gma1 = derive_adiab_fromT_Ryu(p1/rho1)
  gma2 = derive_adiab_fromT_Ryu(p2/rho2)
  h1 = derive_enthalpy(rho1, p1)
  h2 = derive_enthalpy(rho2, p2)

  vx1 = fs1['vx']
  vx2 = fs2['vx']
  lfac1 = derive_Lorentz(vx1)
  lfac2 = derive_Lorentz(vx2)
  if reverse:
    vx1 *= -1
    vx2 *= -1
  ut1 = 0
  # check 2d implementation here
  v12 = (vx1 - vx2) / (1 - vx1*vx2)

  A1 = h1*ut1
  I = 0
  # we get I = 0 if p1=p2, so no need for integration:
  if delta_p !=0:
    for ip in range(n_evals):
      p = p1 + (ip+.5)*dp

      # h is computed from s (entropy). Since rarefaction waves are isentropic,
      # we can set s = s1, so rho = rho1(p/p1)^(1/GAMMA_) (Laplace)
      rho = rho1  * (p/p1)**(1./gma1)
      h = derive_enthalpy(rho, p)
      cs = derive_cs(rho, p)

      dI = np.sqrt(h*h + A1*A1 * (1.-cs*cs)) / ((h*h + A1*A1) * rho * cs) * dp
      I += dI
  
  vSR = np.tanh(I)

  # 2S Rezzolla&Zanotti2013 eq. 4.206 (p238)
  # ----------------------------------------
  rho22 = rho2*rho2
  lfac22 = lfac2*lfac2
  vx22 = vx2*vx2
  g = gma2
  gm1 = (g-1.)
  gm12 = gm1*gm1

  Da = 4. * g * p1 * ((g-1)*p2 + p1) / (gm12 * (p1-p2) * (p1-p2))
  Db = h2 * (p2-p1) / rho2 - h2*h2
  D = 1. - Da*Db

  h3 = (np.sqrt(D)-1) * gm1 * (p1-p2) / (2 * (gm1*p2 + p1))

  J232a = - g/gm1 * (p1-p2)
  J232b = h3*(h3-1)/p1 - h2*(h2-1)/p2
  J232 = J232a/J232b
  J23 = np.sqrt(J232)

  Vsa = rho22*lfac22*vx2 + np.abs(J23) * np.sqrt(J232 + rho22*lfac22*(1.-vx22))
  Vsb = rho22*lfac22 + J232
  Vs = Vsa / Vsb

  v2Sa = (p1 - p2) * (1. - vx2*Vs)
  v2Sb = (Vs - vx2) * (h2*rho2*lfac22*(1.-vx22) + p1 - p2)
  v2S = v2Sa / v2Sb

  # Shock detection threshold
  vlim = vSR + chi*(v2S - vSR)
  Sd = v12 - vlim

  return Sd


