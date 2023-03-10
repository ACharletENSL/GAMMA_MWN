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