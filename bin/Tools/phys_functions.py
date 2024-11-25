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
from numba import jit, prange, vectorize

# General functions
# --------------------------------------------------------------------------------------------------
def logslope(x1, y1, x2, y2):
  return (np.log10(y2)-np.log10(y1))/(np.log10(x2)-np.log10(x1))

def find_closest_old(A, target):
  #A must be sorted
  idx = np.searchsorted(A, target)
  idx = np.clip(idx, 1, len(A)-1)
  left = A[idx-1]
  right = A[idx]
  idx -= target - left < right - target
  return idx

def intersect(a, b):
  '''
  check if segments [aL, aR] and [bL, bR] intersect
  '''
  aL, aR = a
  bL, bR = b
  return bool(aL in range(bL, bR+1) or aR in range(bL, bR+1))


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

@jit(nopython=True)
def broken_plaw_simple(x, b1=-0.5, b2=-1.25):
  '''
  Simple broken power-law, normalized at break
  '''
  return np.piecewise(x, [x<=1., x>1.], [lambda x: x**b1, lambda x: x**b2])

@jit(nopython=True)
def Band_func_v2(x_arr, b1=-0.5, b2=-1.25):
  '''
  Rewritten to be accelerated by numba
  '''

  b1 = np.float64(b1)
  b2 = np.float64(b2)
  b  = b1-b2
  xb = b/(1+b1)
  xdim = len(x_arr.shape)
  if xdim == 1:
    N = len(x_arr)
    y_arr = np.zeros(N)
    for i in prange(N):
      x = x_arr[i]
      if x<=xb:
        y = np.exp(1+b1) * x**b1 * np.exp(-x*(1+b1))
      else:
        y = np.exp(1+b1) * x**b2 * xb**b * np.exp(-b)
      y_arr[i] = y + y_arr[i]
  elif xdim == 2:
    N1, N2 = x_arr.shape
    y_arr = np.zeros((N1, N2))
    for i in prange(N1):
      for j in prange(N2):
        x = x_arr[i,j]
        if x<=xb:
          y = np.exp(1+b1) * x**b1 * np.exp(-x*(1+b1))
        else:
          y = np.exp(1+b1) * x**b2 * xb**b * np.exp(-b)
        y_arr[i,j] = y + y_arr[i]

  return y_arr


def Band_func_basic(x, b1=-0.5, b2=-1.25):
  b  = b1-b2
  xb = b/(1+b1)
  if x<= xb:
    return np.exp(1+b1) * x**b1 * np.exp(-x*(1+b1))
  else:
    return np.exp(1+b1) * x**b2 * xb**b * np.exp(-b)

Band_func = vectorize(Band_func_basic)

def Band_func_old(x, b1=-0.5, b2=-1.25):
  '''
  Band function, as seen in Genet & Granot 2009, eqn (1)
  '''
  xb = (b1-b2)/(1+b1)
  return np.piecewise(x, [x<=xb, x>xb],
    [lambda x: inf_Bandfunc(x, b1), lambda x: sup_Bandfunc(x, b1, b2)])


@jit(nopython=True)
def inf_Bandfunc(x, b1):
  return np.exp(1+b1) * x**b1 * np.exp(-x*(1+b1))

@jit(nopython=True)
def sup_Bandfunc(x, b1, b2):
  b  = (b1-b2)
  xb = b/(1+b1)
  return np.exp(1+b1) * x**b2 * xb**b * np.exp(-b)

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

def derive_adiab(rho, p):
  '''
  Adiabatic index, following Taub-Matthews EoS
  '''
  T = p/rho
  gma = (1./6.)*(8. - 3.*T + np.sqrt(4. + 9.*T*T))
  return gma

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

def derive_Eint(rho, v, p, rhoscale):
  '''
  Internal energy density in lab frame
  '''
  lfac = derive_Lorentz(v)
  gma  = derive_adiab(rho, p)
  eint = derive_Eint_comoving(rho, p, rhoscale)
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

def derive_epint(rho, p):
  '''
  Internal energy in comoving frame in code units
  '''
  gma = derive_adiab(rho, p)
  h   = 1+p*gma/(gma-1.)/rho
  eps = rho*(h-1)/gma
  return eps


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

def derive_Ekin(rho, v, rhoscale):
  '''
  Kinetic energy in lab frame from density and velocity
  '''
  lfac = derive_Lorentz(v)
  return (lfac-1)*lfac*rho*rhoscale*c_**2

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
  ne = xi_e*derive_n(rho, rhoscale)*derive_xiDN(gmin, gmax, psyn)
  fac = (4./3.) * (4*(psyn-1)/(3*psyn-1)) * (16*me_*c_**2*sigT_/(18*pi_*e_))
  return B*ne*fac

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
  dr = dx*c_
  S = 1.
  if geometry == 'cartesian':
    S *= 4*pi_*R0**2
  else:
    S *= 4*pi_*r**2
  return dr*S

def derive_3vol_comoving(x, dx, vx, R0, geometry):
  '''
  Comoving 3-volume of a cell
  '''
  lfac = derive_Lorentz(vx)
  V3 = derive_3volume(x, dx, R0, geometry)
  return lfac*V3

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

def derive_tc1(rho, vx, p, rhoscale, eps_B):
  '''
  Synchrotron cooling time of an electron at non-relativistic energy, in source frame
  '''
  return derive_tcool(rho, vx, p, 1., rhoscale, eps_B)

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

def derive_Ppmax_noprefac(rho, p, rhoscale, eps_B):
  B = derive_B_comoving(rho, p, rhoscale, eps_B)
  ne = xi_e*derive_n(rho, rhoscale)
  fac = sigT_ * me_ * c_**2 / e_
  return fac * ne * B

def derive_Epnu_FC(x, dx, dt, rho, p, R0, rhoscale, eps_B, geometry):
  V4 = derive_4volume(x, dx, dt, R0, geometry)
  Pmax = ((psyn-1)/(2*psyn))*derive_Ppmax_noprefac(rho, p, rhoscale, eps_B)
  return V4*Pmax

def derive_Epnu_SC(x, dx, dt, rho, p, R0, rhoscale, eps_B, geometry):
  V4 = derive_4volume(x, dx, dt, R0, geometry)
  Pmax = (4*(psyn-1)/(3*(3*psyn-1)))*derive_Ppmax_noprefac(rho, p, rhoscale, eps_B)
  return V4*Pmax

def derive_Epnu_vFC(x, dx, rho, vx, p, gmin,
    R0, rhoscale, psyn, eps_B, eps_e, geometry):
  '''
  Total emitted energy per unit frequency of a 4D cell in very fast cooling
  '''
  Wp = 2*((psyn-1)/(psyn-2))
  lfac = derive_Lorentz(vx)
  V3 = derive_3volume(x, dx, R0, geometry)
  nup_m = derive_nup_m_from_gmin(rho, p, gmin, rhoscale, eps_B)
  epe = eps_e * derive_Eint_comoving(rho, p, rhoscale)
  Epnu = lfac * V3 * epe / (Wp * nup_m)
  return Epnu

def derive_Epnu_thsh(x, dx, rho, vx, p,
    R0, rhoscale, psyn, eps_B, eps_e, xi_e, geometry):
  '''
  Total emitted energy per unit frequency of a 4D cell in very fast cooling
  '''
  Wp = 2*((psyn-1)/(psyn-2))
  lfac = derive_Lorentz(vx)
  V3 = derive_3volume(x, dx, R0, geometry)
  nup_m = derive_nup_m(rho, p, rhoscale, psyn, eps_B, eps_e, xi_e)
  epe = eps_e * derive_Eint_comoving(rho, p, rhoscale)
  Epnu = lfac * V3 * epe / (Wp * nup_m)
  return Epnu

def derive_Lum(r, dr, rho, vx, p, gmin, R0, rhoscale, psyn, eps_B, eps_e, geometry):
  '''
  Luminosity normalization for flux calculation in thin shell & very fast cooling regime
  '''
  Epnu = derive_Epnu_vFC(r, dr, rho, vx, p, gmin, R0, rhoscale, psyn, eps_B, eps_e, geometry)
  lfac = derive_Lorentz(vx)
  L0 = (2*vx*lfac**2/r) * Epnu
  return L0

def derive_Lum_thinshell(r, dr, rho, vx, p, R0, rhoscale, psyn, eps_B, eps_e, xi_e, geometry):
  '''
  Luminosity normalization for flux calculation in thin shell & very fast cooling regime
  '''
  Epnu = derive_Epnu_thsh(r, dr, rho, vx, p, R0, rhoscale, psyn, eps_B, eps_e, xi_e, geometry)
  lfac = derive_Lorentz(vx)
  L0 = (2*vx*lfac**2/r) * Epnu
  return L0

def derive_L_thsh_corr(t, r, dr, rho, vx, p, R0, rhoscale, psyn, eps_B, eps_e, xi_e, geometry):
  '''
  lum but corrected to be peak lum of nuLnu instead of peak lnu
  '''
  Epnu = derive_Epnu_thsh(r, dr, rho, vx, p, R0, rhoscale, psyn, eps_B, eps_e, xi_e, geometry)
  lfac = derive_Lorentz(vx)
  gma_m = derive_gma_m(rho, p, rhoscale, psyn, eps_e, xi_e)
  gma_c = derive_gma_c(t, rho, vx, p, rhoscale, eps_B)
  L0 = (2*vx*lfac**2/r) * Epnu
  return L0 * (gma_c/gma_m)

def derive_Fpeak_analytic(x, dx, rho, vx, p, gmin,
    R0, rhoscale, psyn, eps_B, eps_e, zdl, geometry):
  Epnu = derive_Epnu_vFC(x, dx, rho, vx, p, gmin, R0, rhoscale, psyn, eps_B, eps_e, geometry)
  lfac = derive_Lorentz(vx)
  L = (2*vx*lfac**2/x) * Epnu
  return zdl * L

def derive_Fpeak_numeric(x, dx, rho, vx, p, gmin,
    R0, rhoscale, psyn, eps_B, eps_e, zdl, geometry):
  Epnu = derive_Epnu_vFC(x, dx, rho, vx, p, gmin, R0, rhoscale, psyn, eps_B, eps_e, geometry)
  lfac = derive_Lorentz(vx)
  d2 = (lfac*(1-vx))**-2
  L = (d2/(2*x)) * Epnu
  return zdl * L

def derive_Lbol_comov(x, rho, p, R0, rhoscale, eps_e, geometry):
  ''' Bolometric luminosity in comoving frame'''
  eint = derive_Eint_comoving(rho, p, rhoscale)
  r = R0 if geometry == 'cartesian' else x*c_
  S = (4./3.)*pi_*r**2
  return eps_e*eint*S*c_

def derive_Lp_nupm(x, rho, p, R0, rhoscale, psyn, eps_B, eps_e, xi_e, geometry):
  Lbol = derive_Lbol_comov(x, rho, p, R0, rhoscale, eps_e, geometry)
  nup_m = derive_nup_m(rho, p, rhoscale, psyn, eps_B, eps_e, xi_e)
  Wp = 2*(psyn-1)/(psyn-2)
  return Lbol/(Wp*nup_m)

def derive_localLum(rho, vx, p, x, dx, dt, gmin, gmax, gma, rhoscale, eps_B, psyn, xi_e, R0, geometry='cartesian'):
  '''
  Local peak luminosity
  '''
  V4 = derive_rad4volume(x, dx, rho, vx, p, dt, gma, rhoscale, eps_B, R0, geometry)
  #V4 = derive_4volume(x, dx, dt, R0, geometry)
  P = derive_max_emissivity(rho, p, gmin, gmax, rhoscale, eps_B, psyn, xi_e)
  lfac = derive_Lorentz(vx)
  Lp = lfac * V4 * P / x
  return Lp

def derive_obsLum(rho, vx, p, x, dx, dt, gmin, gmax, gma, rhoscale, eps_B, psyn, xi_e, R0, geometry='cartesian'):
  '''
  Local peak luminosity in observer frame
  '''
  Lp = derive_localLum(rho, vx, p, x, dx, dt, gmin, gmax, gma, rhoscale, eps_B, psyn, xi_e, R0, geometry)
  lfac = derive_Lorentz(vx)
  return 2*lfac*Lp

def derive_obsTimes(tsim, r, beta, t0, z):
  '''
  Derive the relevant times in observer frame to calculate flux
  Ton : onset time
  Tth : angular time (comes from Doppler beaming)
  Tej : effective ejection time
  /!\ r in units c /!\ (function to be used by data_IO:get_variable)
  '''
  t = tsim + t0 
  #lfac = derive_Lorentz(beta)
  Ton = (1+z)*(t - r)
  Tth = (1+z)*((1-beta)/beta)*r
  Tej = Ton - Tth
  return Ton, Tth, Tej
  
def derive_obsEjTime(t, r, beta, t0, z):
  '''
  Ejection time in observer's frame
  '''
  Ton, Tr, Tej = derive_obsTimes(t, r, beta, t0, z)
  return Tej

def derive_obsOnTime(t, r, beta, t0, z):
  '''
  Onset time in observer's frame
  '''
  Ton, Tth, Tej = derive_obsTimes(t, r, beta, t0, z)
  return Ton

def derive_obsAngTime(t, r, beta, t0, z):
  '''
  Angular time in observer's frame
  '''
  Ton, Tth, Tej = derive_obsTimes(t, r, beta, t0, z)
  return Tth

def derive_cyclotron_comoving(rho, p, rhoscale, eps_B):
  '''
  Comoving cyclotron frequency
  '''
  B = derive_B_comoving(rho, p, rhoscale, eps_B)
  nup_B = e_*B/(2*pi_*me_*c_)
  return nup_B

def derive_DopplerRed_los(vx, z):
  '''
  Derive Doppler + redshift factor along the line of sight
  '''
  lfac = derive_Lorentz(vx)
  D = 1/(lfac*(1-vx))
  return D/(1+z)

def derive_nu_m(rho, vx, p, rhoscale, psyn, eps_B, eps_e, xi_e, z):
  '''
  Peak frequency of the electron distribution, in observer frame
  '''
  D = derive_DopplerRed_los(vx, z)
  nup_m = derive_nup_m(rho, p, rhoscale, psyn, eps_B, eps_e, xi_e)
  nu_m = nup_m*D
  return nu_m

def derive_nu_m_from_gmin(rho, vx, p, gmin, rhoscale, eps_B, z):
  '''
  Derive peak frequency, with Lorentz factor pre-determined
  '''
  D = derive_DopplerRed_los(vx, z)
  nup_m = derive_nup_m_from_gmin(rho, p, gmin, rhoscale, eps_B)
  return D*nup_m

def derive_nup_m_from_gmin(rho, p, gmin, rhoscale, eps_B):
  '''
  Derive comoving peak frequency, with Lorentz factor pre-determined
  '''
  nup_B = derive_cyclotron_comoving(rho, p, rhoscale, eps_B)
  nup_m = nup_B*gmin**2
  return nup_m

def derive_nup_m(rho, p, rhoscale, psyn, eps_B, eps_e, xi_e):
  '''
  Peak frequency of the electron distribution, in comoving frame
  '''
  nup_B = derive_cyclotron_comoving(rho, p, rhoscale, eps_B)
  gma_m = derive_gma_m(rho, p, rhoscale, psyn, eps_e, xi_e)
  nup_m = nup_B*gma_m**2
  return nup_m

def derive_nup_c(t, rho, vx, p, rhoscale, eps_B):
  '''
  Comoving cooling frequency
  '''
  nup_B = derive_cyclotron_comoving(rho, p, rhoscale, eps_B)
  gma_c = derive_gma_c(t, rho, vx, p, rhoscale, eps_B)
  nup_c = nup_B*gma_c**2
  return nup_c

def derive_gma_c(t, rho, vx, p, rhoscale, eps_B):
  '''
  Typical Lorentz factor of cooled electrons
  '''
  lfac = derive_Lorentz(vx)
  tdyn = t/lfac
  e = derive_Eint_comoving(rho, p, rhoscale)
  return 3*me_*c_/(4*sigT_*eps_B*e*tdyn)


def derive_gma_m(rho, p, rhoscale, psyn, eps_e, xi_e):
  '''
  Return minimum Lorentz factor of accelerated electron distribution
  '''
  mp  = mp_ / (rhoscale*c_**3)
  me  = me_ / (rhoscale*c_**3)
  ne  = xi_e*rho/mp
  gma = derive_adiab(rho, p)
  h   = 1+p*gma/(gma-1.)/rho
  ei  = rho*(h-1)/gma
  ee  = eps_e*ei
  Gp  = (psyn-2.)/(psyn-1.)
  return Gp*ee/(ne*me)

def derive_edistrib(rho, p, rhoscale, psyn, eps_B, eps_e, xi_e):
  '''
  Return theoretical gamma_min and gamma_max of accelerated electron distribution
  '''
  Nmp_ = mp_ / (rhoscale*c_**3)
  Nme_ = me_ / (rhoscale*c_**3)
  gma = derive_adiab(rho, p)
  h   = 1+p*gma/(gma-1.)/rho
  eps = rho*(h-1)/gma
  eB  = eps * eps_B
  B   = np.sqrt(8.*pi_*eB)
  gmax2 = 3*e_/(sigT_*B)
  gmax = 1 + np.sqrt(gmax2)
  ee  = eps * eps_e
  ne  = xi_e * rho / Nmp_
  lfac_av = ee / (ne * Nme_)
  gmin = 1 + ((psyn-2.)/(psyn-1.) * lfac_av)
  return gmin, gmax

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


