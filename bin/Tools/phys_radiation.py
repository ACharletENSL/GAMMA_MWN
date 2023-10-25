# -*- coding: utf-8 -*-
# @Author: acharlet

'''
This file contains the functions necessary to compute emitted radiation
'''
import numpy as np
from phys_constants import *

def lfac2nu(lfac, B):
  '''
  Synchrotron frequency of an electron with Lorentz factor lfac in magn field B (comoving)
  '''

  fac = 3*e_/(4.*pi_*me_*c_)
  return fac * lfac**2 * B

def Hz2eV(nu):
  return h_eV_*nu

def derive_normPnu_GS99(nu, nu_m, nu_c, nu_sa, nu_ac):
  '''
  Returns emissivity normalized at its max following Granot & Sari 1999
  '''


def derive_normPnu_Colle12(nu, nu_m, nu_c, p):
  '''
  Returns emissivity normalized at its max following Colle et al 2012
  '''
  def isin_ab(x, a, b):
    return bool((x>=min(a,b) and x<=max(a,b)))

  x   = nu/nu_m
  p_m =( 1-p)/2.
  y   = nu/nu_c
  p_c = -1/2.

  try:
    Pnu = np.zeros(nu.shape)
    nu_min = np.where(nu_m<=nu_c, nu_m, nu_c)
    nu_max = np.where(nu_m>=nu_c, nu_m, nu_c)
    Pnu = np.where(nu<=nu_min, (nu/nu_min)**(1./3.), 
      np.where(nu>nu_min & nu<=nu_max,
        np.where(nu_m<=nu_c, x**p_m, y**p_c), 
        np.where(nu>nu_max, x**p_m*y**p_c)))
    return Pnu

  except AttributeError:
    nu_min = min(nu_m, nu_c)
    if nu <= nu_min:
      return (nu/nu_min)**(1./3.)
    elif isin_ab(nu, nu_m, nu_c):
      pow = p_m if nu_m<nu_c else p_c
      return (nu/nu_min)**pow
    else:
      return x**p_m * y**p_c




def derive_max_emissivity(ne, eint, p, epsB):
  '''
  Returns maximal emissivity used for normalisation
  from comoving electron density ne, comoving internal energy density
  '''

  fac  = 0.88*(512*np.sqrt(2*pi_)/27.) * (e_**3/rme_)
  pfac = (p-1.)/(3*p-1.)
  eM   = epsB * eint
  return fac * pfac * np.sqrt(eM) * ne

def derive_nu_m(ne, eint, p, epse, epsB):
  '''
  Derive synchrotron min frequency
  '''
  
  fac  = (3.*np.sqrt(2.*pi_)/8.) * (e_**3/rme_)
  pfac = ((p-2.)/(p-1.))**2 
  eM   = epsB * eint
  eE   = epse * eint
  return fac * pfac * np.sqrt(eM) * eE**2 / ne**2

def derive_nuc(t, lfac, eint, epsB):
  '''
  Derive cooling break frequency depending on t time elapsed since shock heating,
  local Lorenz factor, comoving internal energy eint
  '''

  fac = (27.*np.sqrt(2*pi_)/128.) * e_*me_*c_ / sigT_**2
  eM  = epsB*eint
  return fac * eM**(-3/2.) * (lfac/t)**2