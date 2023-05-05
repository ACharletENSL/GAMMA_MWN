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
  Lorentz factor from velocity
  '''
  return 1./np.sqrt(1 - v**2)

def derive_velocity(lfac):
  '''
  Velocity from Lorentz factor
  '''
  return 1. - lfac**-2

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