# -*- coding: utf-8 -*-
# @Author: acharlet

'''
This file contains (semi-)analytical models for the evolution of MWN
'''

# Imports
# --------------------------------------------------------------------------------------------------
import numpy as np
from environment import *

# MWN bubble functions for t<<t0, from Bandiera et al. 2023
# --------------------------------------------------------------------------------------------------
# 0th order: L(t)=L0
def R_0th(t):
  '''
  Analytical solution of the MWN bubble radius in 0th order expansion of L(t) 
  '''
  fac = (3-delta)*(5-delta)**2 * fac1 / ((9-2*delta))
  param = L0*t**(6-delta)/(4*pi*D*vt**delta)
  return (fac*param)**(1/(5-delta))

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
  fac = (1. - m * (t / t0) / (fac1 * (49 - 9*delta)) )
  return R0 * fac

def Q_1st(t):
  '''
  Analytical solution of the quantity Q = 4 pi p R^4 in the Ast order expansion of L(t)
  '''
  Q0 = Q_0th(t)
  fac = (1. - m * (t / t0) * (16 - 3*delta) / (fac1 * (49 - 9*delta)) )
  return Q0 * fac

def p_1st(t):
  '''
  Analytical solution of the MWN bubble pressure in 1st order expansion of L(t)
  '''
  p0 = p_0th(t)
  fac = (1. - m * (t / t0) * 3 * (4 - delta) / (fac1 * (49 - 9*delta)) )
  return p0 * fac

