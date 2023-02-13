# -*- coding: utf-8 -*-
# @Author: acharlet

'''
This file contains (semi-)analytical models for the evolution of MWN
'''

# Imports
# --------------------------------------------------------------------------------------------------
import numpy as np

# Constants
# --------------------------------------------------------------------------------------------------
c = 2.99792458e10
mp = 1.67262192e-24
Msun = 1.9891e33
pi = np.pi

# Parameters
# to add: consistent reading of parameters 
# --------------------------------------------------------------------------------------------------
L0 = 5e45         # erg/s, wind kinetic luminosity
t0 = 2e4          # s, spindown time
n = 3             # braking index
m = (n+1)/(n-1)   # spindown luminosity index
E_sn = 1.e51      # erg, SNR kinetic energy
M_ej = 3.*Msun    # g, ejecta mass
delt = 0          # ejecta core density index
om = 10           # ejecta envelope density index
fac1 = (5-delt)/(11-2*delt)

# intermediate values
# --------------------------------------------------------------------------------------------------
vt = np.sqrt(2*(5-delt)*(om-5) / ((3-delt)*(om-3))) * np.sqrt(E_sn/M_ej)
A = (5-delt)*(om-5)/(2*pi*(om-delt))


# MWN bubble functions
# --------------------------------------------------------------------------------------------------
# 0th order: L(t)=L0
def R_0th(t):
  '''
  Analytical solution of the MWN bubble radius in 0th order expansion of L(t) 
  '''
  fac = (3-delt)*(5-delt)**2 * fac1 / ((9-2*delt))
  param = L0*t**(6-delt)/(4*pi*A*vt**delt)
  return (fac*param)**(1/(5-delt))

def Q_0th(t):
  '''
  Analytical solution of the quantity Q = 4 pi p R^4 in the 0th order expansion of L(t)
  '''
  R0 = R_0th(t)
  return fac1 * L0 * R0 * t

def p_0th(t):
  '''
  Analytical solution of the MWN bubble pressure in 0th order expansion of L(t)
  '''
  R0 = R_0th(t)
  return (fac1/(4.*pi)) * L0 * t /R0**3

# 1st order: L(t) = L0*(1 - m t/t0)
def R_1st(t):
  '''
  Analytical solution of the MWN bubble radius in 1st order expansion of L(t)
  '''
  R0 = R_0th(t)
  fac = (1. - m * (t / t0) / (fac1 * (49 - 9*delt)) )
  return R0 * fac

def Q_1st(t):
  '''
  Analytical solution of the quantity Q = 4 pi p R^4 in the Ast order expansion of L(t)
  '''
  Q0 = Q_0th(t)
  fac = (1. - m * (t / t0) * (16 - 3*delt) / (fac1 * (49 - 9*delt)) )
  return Q0 * fac

def p_1st(t):
  '''
  Analytical solution of the MWN bubble pressure in 1st order expansion of L(t)
  '''
  p0 = p_0th(t)
  fac = (1. - m * (t / t0) * 3 * (4 - delt) / (fac1 * (49 - 9*delt)) )
  return p0 * fac

