# -*- coding: utf-8 -*-
# @Author: acharlet

'''
This file contains the functions necessary to compute emitted radiation
'''
import numpy as np
from phys_constants import *
from phys_functions import *
from phys_edistrib import *
import warnings
#from numba import njit

def cell_flux(cell, nuobs, Tobs, dTobs, env, func='Band'):
  '''
  Flux of a single (spherical) cell
  '''
  if func not in ['Band', 'plaw', 'analytic']:
    print("func parameter must be 'Band', 'plaw', or 'analytic'")
    return 0.
  
  out = np.zeros((len(Tobs), len(nuobs)))
  if cell['gmax'] <1.:
    return out
  
  t = cell['t']
  dt = cell['dt']
  r = cell['x']
  dr = cell['dx']
  beta = cell['vx']
  trac = cell['trac']
  isIn1 = (trac > 1.5)

  Tejk, T0k = derive_obsTimes(t, r, beta, env.t0, env.z)
  tT  = (Tobs-Tejk)/T0k
  if func=='analytic':
    params = ['rho', 'vx', 'p', 'x', 'dx', 'dt', 'gmin', 'gmax', 'gb']
    envparams = ['rhoscale', 'eps_B', 'psyn', 'xi_e', 'R0', 'geometry']
    envvals = [getattr(env, name) for name in envparams]
    nuk = env.nu0FS if isIn1 else env.nu0
    gma = env.gma_mFS if isIn1 else env.gma_m
    eps_rad = env.eps_radFS if isIn1 else env.eps_rad
    #rho = (env.rho2 if isIn1 else env.rho3)/env.rhoscale
    #prs = env.p_sh/(env.rhoscale*c_**2)
    # derive_localLum takes cooling time into account
    #L0 = derive_obsLum(rho, env.beta, prs, r, dr, dt, cell['gmin'], cell['gmax'], gma,
    #  env.rhoscale, env.eps_B, env.psyn, env.xi_e, env.R0, env.geometry)
    L0 = derive_obsLum(*cell[params].to_list(), *envvals)
    nFk = L0 * env.zdl * nuk * eps_rad
    if (isIn1 and (nFk>env.nu0F0)):
      print(f"Cell {cell.name} peaks at {nFk/env.nu0F0} nu0F0")
    nu = nuobs[np.newaxis, :]
    x = nu/nuk
    tT[tT < 0.] = 0.1 # avoid warnings from applying power to negative values
    tT = tT[:, np.newaxis]
    out += np.where(tT<1, 0., nFk * tT**(-2) * x * Band_func(x*tT))
  else:
    lfac = cell['D']/cell['rho']
    P = cell_derive_max_emissivity(cell, env) 
    nupk = lfac2nu(cell['gb'], B)
    nuk = nupk * (2*lfac)/(1+env.z)
    if (nuk > nuobs.max()):
      print(f'Cell {cell.name} radiates outside of observed bins, log nup/nu0 = {np.log10(nupk/env.nu0p):.1f}')
    for i_obs, T in enumerate(Tobs):
      if tT[i_obs] < 1:
        continue
      try:
        dT = dTobs[i_obs]
      except (IndexError, TypeError):
        dT = dTobs
      t += env.t0
      mu = (c_/r)*(t - T)
      dmu = c_*dTobs/r
      dV4 = V3_fac*dmu*dt
      d = 1/(lfac * (1-beta*mu))
      nup = (1+env.z) * nuobs / d
      dnFk = env.zdl * (dV4/dT) * d**3 * nupk * P / (1+env.z)
      x = nup/nupk
      if func == 'Band':
        fac = (cell['gb']/cell['gm'])**(1./3.)
        # formula for P is at gmin <=> gm, peak is at gb after a 1/3 plaw 
        Q = Band_func(x) * fac 
      elif func == 'plaw':
        nu_m = lfac2nu(cell['gm'], B)
        nu_b = lfac2nu(cell['gb'], B)
        nu_M = lfac2nu(cell['gM'], B)
        a1 = env.psyn
        a2 = env.psyn + 1
        if cell['fc']:
          a1 = 2.
        b1 = (a1-1)/2
        b2 = (a2-1)/2
        Q = broken_plaw_with_a0(x, 1., nu_b/nu_m, nu_M/nu_m, 1/3., b1, b2)
      out[i_obs] += dnFk * x * Q

  return out

def cell_derive_obsTimes(cell, env):
  t, r, beta = cell[['t', 'x', 'vx']]
  t = t + env.t0
  z = env.z
  Tej = (1+z)*(t - ((1+beta)/beta)*r)
  Tth = (1+z)*(1-beta)*r
  Ton = (1+z)*(t - 2*r)
  return Tth, Ton, Tej


def cell_derive_radNumDty(cell, env):
  '''
  Return radiative comoving electron number density
  '''
  rho = cell['rho'] * env.rhoscale
  ne = env.xi_e * rho / mp_
  return ne


def cell_derive_B(cell, env):
  '''
  Return comoving magnetic field (in G) from code values
  '''
  rho  = cell['rho']*env.rhoscale*c_**2
  p    = cell['p']*env.rhoscale*c_**2
  eint = derive_Eint_comoving(rho, p)
  return np.sqrt(env.eps_B * eint * 8 * pi_)

def cell_derive_max_emissivity(cell, env):
  '''
  Returns maximal emissivity used for normalisation, from Ayache et al. 2022
  '''
  
  B = cell_derive_B(cell, env)
  n = cell_derive_radNumDty(cell, env) * cell_derive_xiDN(cell, env)
  eps_rad = 1. #env.eps_radFS if cell['trac'] > 1.5 else env.eps_rad
  fac = (4./3.) * (4*(env.psyn-1)/(3*env.psyn-1)) * (16*me_*c_**2*sigT_/(18*pi_*e_))
  return fac * B * n * eps_rad

def cell_derive_xiDN(cell, env):
  xi_e_DN = 1.
  gmax = cell['gmax']
  gmin = cell['gmin']
  p = env.psyn
  try:
    if gmin < 1.:
      xi_e_DN = ((gmax**(2-p) - gmin**(2-p))/(gmax**(2-p)-1))*((gmax**(1-p)-1.)/(gmax**(1-p)-gmin**(1-p)))
  except ValueError:
    xi_e_DN = np.where(gmin < 1., ((gmax**(2-p) - gmin**(2-p))/(gmax**(2-p)-1))*((gmax**(1-p)-1.)/(gmax**(1-p)-gmin**(1-p))), 1.)
  return xi_e_DN

def cell_derive_localLum(cell, env):
  '''
  Local luminosity
  '''
  r = cell['x']*c_
  dV = 2*cell['dx']*c_
  if env.geometry == 'cartesian':
    dV *= 2*pi_*env.R0**2
  else:
    dV *= 2*pi_*r**2
  V4 = dV * cell['dt']
  P = cell_derive_max_emissivity(cell, env) * cell_derive_xiDN(cell, env)
  lfac = cell['D']/cell['rho']
  L = lfac * (c_/r) * V4 * P
  return L

def cell_derive_edistrib(cell, env):
  '''
  Return theoretical gamma_min and gamma_max of accelerated electron distribution
  '''
  Nmp_ = mp_ / (env.rhoscale*c_**3)
  Nme_ = me_ / (env.rhoscale*c_**3)
  rho = cell['rho']
  p   = cell['p']
  gma = 4./3. # à corriger (ajouter cell_derive_adiab)
  h   = 1+p*gma/(gma-1.)/rho
  eps = rho*(h-1)/gma
  eB  = eps * env.eps_B
  B   = np.sqrt(8.*pi_*eB)
  gmax2 = 3*e_/(sigT_*B)
  gmax = 1 + np.sqrt(gmax2)
  ee  = eps * env.eps_e
  ne  = env.xi_e * rho / Nmp_
  lfac_av = ee / (ne * Nme_)
  gmin = 1 + ((env.psyn-2.)/(env.psyn-1.) * lfac_av)
  return gmin, gmax


def broken_plaw_with_a0(x, g1, g2, g3, a0, a1, a2):
  '''
  Powerlaw with index a0 to g1, -a1 between g1 and g2, -a2 between g2 and g3
  put exponential cutoff after g3?
  '''
  try:
    n = np.where(x < g1, (x/g1)**a0, 
      np.where(x < g2, (x/g1)**(-a1),
        np.where(x < g3, ((g2/g1)**(-a1))*(x/g2)**(-a2), 0.)))
          #((g2/g1)**(-a1))*(g3/g2)**(-a2)*(x/g3)**(-a3))))
    return n
  except TypeError:
    if x < g1: return (x/g1)**a0
    elif x < g2: return (x/g1)**(-a1)
    elif x < g3: return ((g2/g1)**(-a1))*(x/g2)**(-a2)
    else: return 0.#((g2/g1)**(-a1))*(g3/g2)**(-a2)*(x/g3)**(-a3)


# Maths functions
# --------------------------------------------------------------------------------------------------
def Band_func(x, b1=-0.25, b2=-1.25, Bnorm=1.):
  '''
  Band function
  '''
  b  = (b1 - b2)
  xb = b/(1+b1)
  try:
    return Bnorm * np.exp(1+b1) * np.where(x<=xb, x**b1 * np.exp(-x*(1+b1)), x**b2 * xb**b * np.exp(-b))
  except TypeError:
    if x <= xb:
      return Bnorm * np.exp(1+b1) * x**b1 * np.exp(-x*(1+b1))
    else:
      return Bnorm * np.exp(1+b1) * x**b2 * xb**b * np.exp(-b)
