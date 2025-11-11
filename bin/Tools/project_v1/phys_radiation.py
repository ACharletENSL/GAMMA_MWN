# -*- coding: utf-8 -*-
# @Author: acharlet

'''
This file contains the functions necessary to compute emitted radiation
'''
import numpy as np
import scipy.integrate as spi
import pandas as pd
from phys_constants import *
from phys_functions import *
from phys_edistrib import *
from df_funcs import *
import warnings
#from numba import njit

def cell_lightcurve(cell, nuobs, Tobs, env, method='analytic', func='Band'):
  '''
  nuFnu of a single (spherical) cell vs time
  '''

  params = ['x', 'dx', 'dt', 'rho', 'vx', 'p', 'gmin', 'gmax']
  Ton, Tth, Tej = get_variable(cell, 'obsT')
  tau = (Tobs-Tej)/Tth
  Tb = tau - 1.
  Fnu = np.zeros(Tobs.shape)
  if method == 'analytic':
    Fnu[tau>=1.] = derive_flux_analytic(nuobs, tau[tau>=1.], *cell[params].to_list(), env, func)
  elif method == 'numeric':
    mu = (cell['t']+env.t0-Tobs/(1+env.z))/cell['x']
    Fnu[mu<=1.] = derive_flux_numeric(nuobs, mu[mu<=1.], *cell[params].to_list(), env, func)
  nuFnu = nuobs * Fnu
  return Tb, nuFnu

def cell_nuFnu(cell, nuobs, Tobs, env, func='Band', hybrid=False, method='analytic'):
  '''
  nuFnu of a single (spherical) cell
  '''

  params = ['x', 'dx', 'dt', 'rho', 'vx', 'p', 'gmin', 'gmax']
  Ton, Tth, Tej = get_variable(cell, 'obsT')
  tau  = (Tobs-Tej)/Tth
  Fnu = np.zeros((len(tau), len(nuobs)))
  if method == 'analytic':
    Fnu[tau>=1.] = derive_flux_analytic(nuobs, tau[tau>=1.], *cell[params].to_list(), env, func, hybrid)
  elif method == 'numeric':
    mu = (cell['t']+env.t0-Tobs/(1+env.z))/cell['x']
    Fnu[mu<=1.] = derive_flux_numeric(nuobs, mu[mu<=1.], *cell[params].to_list(), env, func)
  nuFnu = nuobs[np.newaxis, :] * Fnu
  return nuFnu

def derive_flux_analytic(nuobs, tau, r, dr, dt,
    rho, vx, p, gmin, gmax, env, func='Band', hybrid=False):
  '''
  Returns flux Fnu of a cell, using analytical formulas
  for a spherical infinitely thin shell.
  tau is the normalized time \tilde{T}, the array is supposed to start at 1
  '''
  # move env variables outside, as input params?
  # to have consistent form with the data_IO:get_variable

  if (type(tau)==np.ndarray) and (type(nuobs)==np.ndarray):
    out = np.zeros((len(tau), len(nuobs)))
  elif (type(tau)==np.ndarray):
    out = np.zeros(tau.shape)
  elif (type(nuobs)==np.ndarray):
    out = np.zeros(nuobs.shape)
  hybrid = True if env.geometry == 'cartesian' else False

  if func not in ['Band', 'plaw']:
    print("'func' input must be 'Band' or 'plaw'")
    return out

  # we're in thin shell, very fast cooling regime
  nu_m = derive_nu_m_from_gmin(rho, vx, p, gmin, env.rhoscale, env.eps_B, env.z)
  L0 = derive_Lum_thinshell(r, dr, rho, vx, p, 
    env.R0, env.rhoscale, env.psyn, env.eps_B, env.eps_e, env.xi_e, env.geometry)
  if hybrid:
    a, d = 1, -1
    nu_m *= (r*c_/env.R0)**d
    L0 *= (r*c_/env.R0)**a
  Fk = env.zdl * L0
  F = Fk * tau**-2
  nu_b = nuobs/nu_m
  if (type(tau)==np.ndarray) and (type(nuobs)==np.ndarray):
    nu_b  = nu_b[np.newaxis, :]
    tau = tau[:, np.newaxis]
    F  = F[:, np.newaxis]

  x = nu_b*tau
  if func == 'Band':
    S = Band_func(x)
  elif func == 'plaw':
    S = broken_plaw_simple(x)
  out = F * S
  return out

def derive_flux_numeric(nuobs, mu, r, dr, dt,
    rho, vx, p, gmin, gmax, env, func='Band'):
  '''
  Returns flux Fnu of a spherical cell, subdivising it in subcells of delta mu
  Takes array of mu contributing to the emission as input
  '''
  if (type(mu)==np.ndarray) and (type(nuobs)==np.ndarray):
    out = np.zeros((len(mu), len(nuobs)))
  elif (type(mu)==np.ndarray):
    out = np.zeros(mu.shape)
  elif (type(nuobs)==np.ndarray):
    out = np.zeros(nuobs.shape)

  if func not in ['Band', 'plaw']:
    print("'func' input must be 'Band' or 'plaw'")
    return out
  
  lfac = derive_Lorentz(vx)
  nup_m = derive_nup_m_from_gmin(rho, p, gmin, env.rhoscale, env.eps_B)
  
  # check cooling regime
  # very fc
  Epnu = derive_Epnu_vFC(r, dr, rho, vx, p, gmin,
    env.R0, env.rhoscale, env.psyn, env.eps_B, env.eps_e, env.geometry)
  # FC or SC, same inputs just change func name
  # Epnu = derive_Epnu_FC(x, dx, dt, rho, p, R0, rhoscale, eps_B, geometry)
  
  d = 1/(lfac * (1-vx*mu))
  if (type(mu)==np.ndarray) and (type(nuobs)==np.ndarray):
    d = d[:, np.newaxis]
  nup = (1+env.z) * nuobs / d
  x = nup/nup_m
  dFnu = env.zdl * 1/(2*r) * d**2 * Epnu
  if func == 'Band':
    S = Band_func(x)
    out = dFnu * S
  elif func == 'plaw':
    S = broken_plaw_simple(x)
  return out


# Fully analytical methods
# --------------------------------------------------------------------------------------------
def get_nuFnu(nuobs, Tobs, env, nuFnu_func):
  Tb = (Tobs-env.Ts)/env.T0
  nub = nuobs/env.nu0
  TbFS = env.fac_T * Tb
  nubFS = nub * env.fac_nu
  gRS, gFS = env.gRS, env.gFS
  nuFnu_RS = nub[:, np.newaxis] * nuFnu_func(nub, Tb, gRS, env.dRRS/env.R0)
  nuFnu_FS = nub[:, np.newaxis] * env.fac_F**-1 * nuFnu_func(nubFS, TbFS, gFS, env.dRFS/env.R0)
  return np.array([nuFnu_RS, nuFnu_FS])


def nuFnu_theoric(nuobs, Tobs, env, g1=None, hybrid=False, func='Band'):
  '''
  Returns nuFnu for both shock fronts using the analytical formulas
  from cases m=0; a,d = 1,-1 and a,d = 0,0
  '''
  Tb = (Tobs-env.Ts)/env.T0
  nub = nuobs/env.nu0
  TbFS = env.fac_T * Tb
  nubFS = nub * env.fac_nu
  gRS, gFS = (1., 1.) if g1 else (env.gRS, env.gFS)
  if (env.geometry == 'cartesian') and (not hybrid):
    print('Calculating in planar geometry')
    nuFnu_RS = nub[:, np.newaxis] * flux_shockfront_planar(nub, Tb, gRS, env.dRRS/env.R0, func=func)
    nuFnu_FS = nub[:, np.newaxis] * env.fac_F**-1 * flux_shockfront_planar(nubFS, TbFS, gFS, env.dRFS/env.R0, func=func)
  else:
    nuFnu_RS = nub[:, np.newaxis] * flux_shockfront_hybrid(nub, Tb, gRS, env.dRRS/env.R0, func=func)
    nuFnu_FS = nub[:, np.newaxis] * env.fac_F**-1 * flux_shockfront_hybrid(nubFS, TbFS, gFS, env.dRFS/env.R0, func=func)
  return np.array([nuFnu_RS, nuFnu_FS])

def nuFnu_general(nuobs, Tobs, env, m, a, d, g1=None, func='Band'):
  '''
  Returns nuFnu for both shock fronts by integrating the general formula
  '''
  Tb = (Tobs-env.Ts)/env.T0
  nub = nuobs/env.nu0
  TbFS = env.fac_T * Tb
  nubFS = nub * env.fac_nu
  gRS, gFS = (1., 1.) if g1 else (env.gRS, env.gFS)
  b1, b2 = -0.5, -env.psyn/2

  nuFnu_RS = nub * flux_shockfront_general(nub, 1.+Tb, gRS, env.dRRS/env.R0, a, d, m, b1, b2)
  nuFnu_FS = nub * env.fac_F**-1 * flux_shockfront_general(nubFS, 1.+TbFS, gFS, env.dRFS/env.R0, a, d, m, b1, b2)
  return nuFnu_RS, nuFnu_FS

def flux_shockfront_planar(nu_bar, T_bar, g, DeltaR, b1=-0.5, b2=-1.25, func='Band'):
  '''
  Returns Fnu of a given shock front in the fully planar case (a=0, d=0)
  T_bar must already be scaled to the shock front considered
  '''
  if (type(nu_bar)==np.ndarray) and (type(T_bar)==np.ndarray):
    case = 0
  elif (type(nu_bar)==np.ndarray):
    case = 1
    T_bar = np.array([T_bar])
  elif (type(T_bar)==np.ndarray):
    case = 2
    nu_bar = np.array([nu_bar])
  
  F_nu = np.zeros((len(nu_bar), len(T_bar)))

  if func == 'Band':
    def f(x):
      return x**-3*Band_func(x, b1=b1, b2=b2)
  elif func == 'plaw':
    def f(x):
      return x**-3*broken_plaw_simple(x, b1=b1, b2=b2)
  else:
    print("func must be 'Band' of 'plaw'!")

  if g == 1.: # Avoid crash for g = 1
    g += 1e-4

  tilde_T = 1 + T_bar
  for i in range(len(T_bar)):
    for j in range(len(nu_bar)):
      if(tilde_T[i]<1):
        y_min = 1
      else:
        y_min = 1/tilde_T[i]

      if(tilde_T[i]<(1+DeltaR)):
        y_max = 1
      else:
        y_max = (1+DeltaR)/tilde_T[i]
      
      if(tilde_T[i]<1):
        quant = 0
      else:
        x_u = nu_bar[j]*(1 + (g**2)*( (1/y_min) -1.0 ))
        x_l = nu_bar[j]*(1 + (g**2)*( (1/y_max) -1.0 ))
        quant = spi.quad(f, x_l, x_u)[0]
        F_nu[j,i] = 3*(quant)*(nu_bar[j]**2)
  if case == 1:
    out = F_nu[:,0]
  elif case == 2:
    out = F_nu[0]
  else:
    out = F_nu
  return out

# def flux_shockfront_planar_old(nu_bar, T_bar, g, DeltaR, b1=-0.5, b2=-1.25, func='Band'):
#   '''
#   Returns Fnu of a given shock front in the fully planar case (a=0, d=0)
#   T_bar must already be scaled to the shock front considered
#   '''
#   if (type(nu_bar)==np.ndarray) and (type(T_bar)==np.ndarray):
#     case = 0
#   elif (type(nu_bar)==np.ndarray):
#     case = 1
#     T_bar = np.array([T_bar])
#   elif (type(T_bar)==np.ndarray):
#     case = 2
#     nu_bar = np.array([nu_bar])
#   F_nu = np.zeros((len(nu_bar), len(T_bar)))

#   if g == 1.: # Avoid crash for g = 1
#     g += 1e-4
#   x_b = (b1-b2)/(1+b1)
#   tilde_T = 1 + T_bar

#   for i in range(len(T_bar)):
#     for j in range(len(nu_bar)):
#       if(tilde_T[i]<1):
#         y_min = 1
#       else:
#         y_min = 1/tilde_T[i]

#       if(tilde_T[i]<(1+DeltaR)):
#         y_max = 1
#       else:
#         y_max = (1+DeltaR)/tilde_T[i]
      
#       if(tilde_T[i]<1):
#         quant = 0
#       else:
#         # y_max > y_min => x(y_max) < x(y_min)
#         x_u = nu_bar[j]*(1 + (g**2)*( (1/y_min) -1.0 ))
#         x_l = nu_bar[j]*(1 + (g**2)*( (1/y_max) -1.0 ))
        
#         if(x_u<x_b):
#           I1 = spi.quad(integrand1_simplified, x_l, x_u, args=(b1))[0]
#           quant1 = (np.exp(1+b1))*I1
#           quant2 = 0
#         elif(x_l>x_b):
#           I2 = integral2_simplified(x_l, x_u, b2)
#           quant1 = 0
#           quant2 = (x_b**(b1-b2))*(np.exp(1+b2))*I2
#         else:
#           I1 = spi.quad(integrand1_simplified, x_l, x_b, args=(b1))[0]
#           I2 = integral2_simplified(x_b, x_u, b2)
#           quant1 = (np.exp(1+b1))*I1
#           quant2 = (x_b**(b1-b2))*(np.exp(1+b2))*I2
#         quant = quant1+quant2
#         F_nu[j,i] = 3*(quant)*(nu_bar[j]**2)
#   if case == 1:
#     out = F_nu[:,0]
#   elif case == 2:
#     out = F_nu[0]
#   else:
#     out = F_nu
#   return out

def flux_shockfront_planar_yInt(nu_bar, T_bar, g, DeltaR, b1=-0.5, b2=-1.25, func='Band'):
  '''
  Returns Fnu of a given shock front in the fully planar case (a=0, d=0)
  Here we use the integral form without changing variables
  T_bar must already be scaled to the shock front considered
  '''
  if (type(nu_bar)==np.ndarray) and (type(T_bar)==np.ndarray):
    case = 0
  elif (type(nu_bar)==np.ndarray):
    case = 1
    T_bar = np.array([T_bar])
  elif (type(T_bar)==np.ndarray):
    case = 2
    nu_bar = np.array([nu_bar])
  
  F_nu = np.zeros((len(nu_bar), len(T_bar)))

  if func == 'Band':
    def f(y, nu_bar):
      gy = 1 + g**2 * (y**-1 - 1)
      return gy**-3 * y**-2 * Band_func(gy*nu_bar, b1=b1, b2=b2)
  elif func == 'plaw':
    def f(y, nu_bar):
      gy = 1 + g**2 * (y**-1 - 1)
      return gy**-3 * y**-2 * broken_plaw_simple(gy*nu_bar, b1=b1, b2=b2)
  else:
    print("func must be 'Band' of 'plaw'!")

  if g == 1.: # Avoid crash for g = 1
    g += 1e-4

  tilde_T = 1 + T_bar
  for i in range(len(T_bar)):
    for j in range(len(nu_bar)):
      if(tilde_T[i]<1):
        y_min = 1
      else:
        y_min = 1/tilde_T[i]

      if(tilde_T[i]<(1+DeltaR)):
        y_max = 1
      else:
        y_max = (1+DeltaR)/tilde_T[i]
      
      if(tilde_T[i]<1):
        quant = 0
      else:
        quant = spi.quad(f, y_min, y_max, args=(nu_bar[j]))[0]
        F_nu[j,i] = 3*g**2*quant

  if case == 1:
    out = F_nu[:,0]
  elif case == 2:
    out = F_nu[0]
  else:
    out = F_nu
  return out

def flux_shockfront_hybrid(nu_bar, T_bar, g, DeltaR, b1=-0.5, b2=-1.25, func='Band'):
  '''
  Returns Fnu of a given shock front in case (a=1, d=-1, m=0)
  T_bar must already be scaled to the shock front considered
  '''
  if (type(nu_bar)==np.ndarray) and (type(T_bar)==np.ndarray):
    case = 0
  elif (type(nu_bar)==np.ndarray):
    case = 1
    T_bar = np.array([T_bar])
  elif (type(T_bar)==np.ndarray):
    case = 2
    nu_bar = np.array([nu_bar])
  F_nu = np.zeros((len(nu_bar), len(T_bar)))


  if g == 1.: # Avoid crash for g = 1
    g += 1e-4
  x_b = (b1-b2)/(1+b1)
  tilde_T = 1 + T_bar

  for i in range(len(T_bar)):
    for j in range(len(nu_bar)):
      if(tilde_T[i]<1):
        y_min = 1
      else:
        y_min = 1/tilde_T[i]

      if(tilde_T[i]<(1+DeltaR)):
        y_max = 1
      else:
        y_max = (1+DeltaR)/tilde_T[i]
      
      if(tilde_T[i]<1):
        quant = 0
      else:
        A = nu_bar[j]*tilde_T[i] 
        A_inv = 1/A
        x_min = A*y_min*(1 + (g**2)*( (1/y_min) -1.0 ))
        x_max = A*y_max*(1 + (g**2)*( (1/y_max) -1.0 ))
        
        if func == 'Band':
          if(x_max<x_b):
            I1 = spi.quad(integrand1, x_min, x_max, args=(b1,A_inv,g))[0]
            quant1 = (np.exp(1+b1))*I1
            quant2 = 0
          elif(x_min>=x_b):
            I2 = spi.quad(integrand2, x_min, x_max, args=(b2,A_inv,g))[0]
            quant1 = 0
            quant2 = (x_b**(b1-b2))*(np.exp(1+b2))*I2
          else:
            I1 = spi.quad(integrand1, x_min, x_b, args=(b1,A_inv,g))[0]
            I2 = spi.quad(integrand2, x_b, x_max, args=(b1,A_inv,g))[0]
            quant1 = (np.exp(1+b1))*I1
            quant2 = (x_b**(b1-b2))*(np.exp(1+b2))*I2
          quant = quant1+quant2
        elif func == 'plaw':
          def integrand(x):
            term1 = (A_inv*x - g**2)**2
            return term1 * x**-3 * broken_plaw_simple(x, b1, b2)
          # if (x_max < 1.):
          #   quant1 = spi.quad(integrand_plaw, x_min, x_max, args=(b1,A_inv,g))[0]
          #   quant2 = 0
          # elif (x_min >= 1.):
          #   quant1 = 0
          #   quant2 = spi.quad(integrand_plaw, x_min, x_max, args=(b2,A_inv,g))[0]
          # else:
          #   quant1 = spi.quad(integrand_plaw, x_min, 1., args=(b1,A_inv,g))[0]
          #   quant1 = spi.quad(integrand_plaw, 1., x_max, args=(b2,A_inv,g))[0]
          quant = spi.quad(integrand, x_min, x_max)[0]
        F_nu[j,i] = 3*(quant)*(A**2)*(g**2/(1-g**2)**3)*tilde_T[i]
  if case == 1:
    out = F_nu[:,0]
  elif case == 2:
    out = F_nu[0]
  else:
    out = F_nu
  return out

def flux_shockfront_general(nu_bar, T_bar, g, DeltaR,
      a=1, d=-1, m=0, b1=-0.5, b2=-1.25, func='Band'):
  '''
  Generate the observed flux for a shockfront crossing a region of normalized width DeltaR.
  The shock propagates with Gamma = Gamma0 * (r/R0)**(-m/2), with downstream lfac = g * shock lfac.
  The luminosity follows L = L0 * (r/R0)**a * func(nu'/nu'_pk, **kwargs), and nu'_pk = nu'_0 * (r/R0)**d
  b1 and b2 are the low- and high frequency indices
  nu_bar = nuobs/nu_pk(r=R0), T_tilde = (Tobs-Tej(r=R0, Gamma=Gamma0))/Ttheta(r=R0, Gamma=Gamma0)
  Return a value normalized to F0 = (1+z)*2*Gamma0*L'0/(4*pi_*dL**2)
  '''

  if (type(nu_bar)==np.ndarray) and (type(T_bar)==np.ndarray):
    case = 0
  elif (type(nu_bar)==np.ndarray):
    case = 1
    T_bar = np.array([T_bar])
  elif (type(T_bar)==np.ndarray):
    case = 2
    nu_bar = np.array([nu_bar])

  F_nu = np.zeros((len(nu_bar), len(T_bar)))

  if func == 'Band':
    def f(y, nub, tT):
      m1 = m+1
      gy = 1 + g**2 * (y**-m1 - 1) / m1
      term = (1+m*y**m1)/(g**2 + (m1-g**2)*y**m1)
      x = tT**-d * y**(m/2-d) * gy * nub
      return gy**-2 * y**(a-1-m/2) * term * Band_func(x, b1=b1, b2=b2)
  elif func == 'plaw':
    def f(y, nub, tT):
      m1 = m+1
      gy = 1 + g**2 * (y**-m1 - 1) / m1
      term = (1+m*y**m1)/(g**2 + (m1-g**2)*y**m1)
      x = tT**-d * y**(m/2-d) * gy * nub
      return gy**-2 * y**(a-1-m/2) * term * broken_plaw_simple(x, b1=b1, b2=b2)
  else:
    print("func must be 'Band' of 'plaw'!")

  if g == 1.: # Avoid crash for g = 1
    g += 1e-4

  tilde_T = 1. + T_bar
  for i in range(len(T_bar)):
    for j in range(len(nu_bar)):
      if(tilde_T[i]<1):
        y_min = 1
      else:
        y_min = 1/tilde_T[i]

      if(tilde_T[i]<(1+DeltaR)):
        y_max = 1
      else:
        y_max = (1+DeltaR)/tilde_T[i]
      
      if(tilde_T[i]<1):
        quant = 0
      else:
        quant = spi.quad(f, y_min, y_max, args=(nu_bar[j], tilde_T[i]))[0]
        F_nu[j,i] = 3*g**2*tilde_T[i]**(a-m/(2*(m+1)))*quant

  # for j, nub in enumerate(nu_bar):
  #   for i, tT in enumerate(T_tilde):
  #     if (tT >= 1.):
  #       y_min = tT**-1
  #       if (tT < (1+DeltaR)):
  #         y_max = 1
  #       else:
  #         y_max = (1+DeltaR)/tT
  #       quant = spi.quad(f, y_min, y_max, args=(nub, tT))[0]
  #       F_nu[j,i] = 3 * g**2 * tT**(a-m/(2*(m+1))) * quant
      
  if case == 1:
    out = F_nu[:,0]
  elif case == 2:
    out = F_nu[0]
  else:
    out = F_nu
  return out

# Approximate functions found in R24 by fitting the obtained flux
def approx_bhvbrk(dRoR, key='cart_fid', sh='RS'):
  '''
  Get the break frequency of the spectral behavior as a function of dR/R_0
  '''
  nuobs, Tobs, env = get_radEnv(key)
  g, Tth = env.gFS, env.T0FS if sh == 'FS' else env.gRS, env.T0
  tT = 1. + (Tobs - env.Ts)/Tth
  tTf = 1. + dRoR

def approx_bhv_hybrid(tTi, tTfi, gi, d1=1, d2=1, p1=3, p2=0.5):
  '''
  Returns nu_pk, nu_pk F_{nu_pk} from hybrid approach, based on fits from R24
  d1, d2: exponents for nu_pk before and after break
  p1, p2: exponents for (nu*F_nu)_pk before and after break
  '''
  nu = approx_nupk_hybrid(tTi, tTfi, gi, d1=d1, d2=d2)
  nuFnu = approx_nupkFnupk_hybrid(tTi, tTfi, gi, p1=p1, p2=p2)
  return nu, nuFnu

def nupk_approx(tT, tTf, d1, d2):
  return np.piecewise(tT, [tT<=tTf, tT>tTf],
      [lambda x: x**-d1, lambda x: tTf**-d1*(x/tTf)**-d2])

def nupkFnupk_approx(tT, tTf, p1, p2):
  return np.piecewise(tT, [tT<=tTf, tT>tTf],
  [lambda x: 1 - x**-p1, lambda x: (1 - tTf**-p1)*(x/tTf)**-p2])
   

def approx_nupkFnupk_hybrid(tTi, tTfi, gi, p1=3, p2=0.5):
  '''
  Peak flux (peak of nu F_nu) of a shock front with respect to its normalized time
  tTfi = R_{f,i}/R_0
  p1: exponent before break (default=3)
  p2: exponent after break (default=0.5 for increasing F_pk with decreasing nu_pk)
  '''

  def f(x):
    tTeff = tT_eff1(x, gi)
    return 1 - tTeff**-p1
  return np.piecewise(tTi, [tTi<=tTfi, tTi>tTfi], [lambda x: f(x), lambda x: f(tTfi)*tT_eff2(x, tTfi, gi)**-p2])

def approx_nupk_hybrid(tTi, tTfi, gi, d1=1, d2=1):
  '''
  Peak frequency of a shock front with respect to its normalized time 
  tTfi = R_{f,i}/R_0
  '''

  return np.piecewise(tTi, [tTi<=tTfi, tTi>tTfi], [lambda x: x**-d1, lambda x: tTfi**-d1*tT_eff2(x, tTfi, gi)**-d2])

def tT_eff1(tTi, gi):
  '''
  Normalized time 1 for fitted curves of R24b
  '''
  gi2 = gi**2
  return (1-gi2)+gi2*tTi

def tT_eff2(tTi, tTfi, gi):
  '''
  Normalized time 2 for fitted curves of R24b
  '''
  gi2 = gi**2
  return (1-gi2) + gi2*tTi/tTfi


# Maths functions
# --------------------------------------------------------------------------------------------------
# integrands for the a=1, d=-1 case
def integrand1(x, b1, A_inv, g):
  term1 = (np.exp(-(1+b1)*x))*(x**(-3+b1))
  term2 = A_inv*x - g**2
  term_integ1 = term1*(term2**2)
  return term_integ1

def integrand2(x, b2, A_inv, g):
  term1 = x**(-3+b2)
  term2 = A_inv*x - g**2
  term_integ2 = term1*(term2**2)
  return term_integ2

def integrand_plaw(x, b, A_inv, g):
  return (A_inv*x - g**2)*x**(b-3)

def integrand_planar(x, b1, b2, A_inv, g, func='Band'):
  return np.zeros(x.shape)

def integrand_hybrid(x, b1, b2, A_inv, g, func='Band'):
  g2 = g**2
  if func == 'Band':
    S = Band_func(x, b1, b2)
  elif func == 'plaw':
    S = broken_plaw_simple(x, b1, b2)
  gterm = 3/(1-g2)**3
  xterm = x**-3 * (A_inv*x - g2)**2
  return gterm*xterm*S

# integrand1_Band and integral 2 for the a=0, d=0 case
def integrand1_simplified(x, b1):
  return (np.exp(-(1+b1)*x))*(x**(b1-3))

def integral2_simplified(xl, xu, b2):
  a = b2-2
  return (xu**a - xl**a)/a

def integrand_hybrid_y(y, nub, tT, g, b1=0.5, b2=-1.25, func='Band'):
  gy = 1 + g**2*(y**-1-1.)
  x = y*tT*nub*gy
  if func == 'Band':
    S = Band_func(x, b1, b2)
  elif func == 'plaw':
    S = broken_plaw_simple(x, b1, b2)
  return tT * y**-1 * gy**-3 * S


# integrands in the general case for unspecified a and d
def integrand_general(y, nub, tT, g, a, d, m, b1=0.5, b2=-1.25, func='Band'):
  m1  = m+1
  g2  = g**2
  ym1 = y**m1
  term1 = (1 + g2*(ym1**-1-1.)/m1)**-2
  term2 = (1+m*ym1)/(g2+(m1-g2)*ym1)
  tTy = tT*y
  gy = 1. + g2*(y**-1 - 1.)
  x = tTy**(-d) * gy * nub
  if func == 'Band':
    return y**(-1-m/2.) * term1 * term2 * Band_func(x, b1=b1, b2=b2)
  elif func == 'plaw':
    return y**(-1-m/2.) * term1 * term2 * broken_plaw_simple(x, b1=b1, b2=b2)
  else:
    print("func must be 'Band' of 'plaw'!")
    return 0.

