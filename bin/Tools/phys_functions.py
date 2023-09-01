# -*- coding: utf-8 -*-
# @Author: acharlet

'''
This file contains functions to derive physical variables
/!\ IN NATURAL/CODE UNITS /!\ 
'''

# Imports
# --------------------------------------------------------------------------------------------------
import numpy as np

# Functions
# --------------------------------------------------------------------------------------------------
# EoS and related
def derive_temperature(rho, p):
  '''
  Relativistic temperature from density and pressure
  '''
  return p/rho

def derive_enthalpy(rho, p):
  '''
  Enthalpy from density and pressure
  '''
  T = derive_temperature(rho, p)
  gma = derive_adiab_fromT_Ryu(T)
  return 1. + T*gma/(gma-1.)

def derive_adiab_fromT_Ryu(T):
  '''
  Adiabatic index from temperature, following Ryu et al 2006 EoS
  '''
  a = 3*T + 1
  return (4*a+1)/(3*a)

def derive_adiab_from_T_Synge(T):
  '''
  Adiabatic index from temperature, following Ryu et al 2006 EoS
  '''

def derive_cs(rho, p):
  '''
  Sound speed
  '''
  T = derive_temperature(rho, p)
  c2 = T*(3*T+2)*(18*T**2+24*T+5) / (3*(6*T**2+4*T+1)*(9*T**2+12*T+2))
  return np.sqrt(c2)


def derive_Eint(rho, p):
  '''
  Internal energy from density and pressure
  '''
  T = derive_temperature(rho, p)
  gma = derive_adiab_fromT_Ryu(T)
  return p/(gma-1.)

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
  return 1. - lfac**-2

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
  Kinetic energy from density and velocity
  '''
  lfac = derive_Lorentz(v)
  return rho*(lfac-1)

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

