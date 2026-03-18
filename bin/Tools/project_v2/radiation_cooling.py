# -*- coding: utf-8 -*-
# @Author: acharlet

'''
Produce emission from hydro data + electron distribution
Contains:
  - syn_emiss_exact
  - func_R
  - get_Fnu_cell
  - get_Fnu_hydrostep
  - get_Fnu_step
  - get_Lnu_comov
  - get_epnu
  - Pnu_instant_bins
  - get_Fnu_cell_analytic (optimal time binning for constant hydro)
  - get_cell_nuFnu_analytic (wrapper for analytic case)
'''

import numpy as np
from cooling_distribution import norm_plaw_distrib, distrib_plaw_cooled, \
  gamma_synCooled, split_hydrostep
from IO import get_variable
from obs_functions import *
from phys_functions import derive_xiDN
from phys_constants import *

###### Emission function
# coeffs from table 1 in Finke, Dermer & Böttcher 08
coeffs_bel1 = [-0.35775237, -0.83695385, -1.1449608,
    -0.68137283, -0.22754737, -0.031967334]
coeffs_abv1 = [-0.35842494, -0.79652041, -1.6113032,
    0.26055213, -1.6979017, 0.032955035]

def syn_emiss_exact(gma, tnu):
  '''
  "exact" synchrotron emission function from Crusius & Schlikeiser (1986)
  for an electron with LF gma at tnu = nu'/nu'_B, nu'_B=e*B/(2*pi_*me_*c_)
  
  '''
  # underflows in np.exp will be treated as 0.
  np.seterr(under='ignore')
  if tnu < 1.:
    return 0.
  else:
    x = (2*tnu)/(3*gma**2)
    return func_R(x)

def func_R(x):
  '''
  R function see eqns (18) to (20) in Finke, Dermer & Böttcher 2008
  '''
  y = np.log10(x)
  if x < 0.01:
    R = 1.80842*x**(1./3.)
    return R
  elif (x>=0.01) and (x<1):
    A0, A1, A2, A3, A4, A5 = coeffs_bel1
    logR = A0 + A1*y + A2*y**2 + A3*y**3 + A4*y**4 + A5*y**5
    return 10**(logR)
  elif (x>=1) and (x<10):
    A0, A1, A2, A3, A4, A5 = coeffs_abv1
    logR = A0 + A1*y + A2*y**2 + A3*y**3 + A4*y**4 + A5*y**5
    return 10**(logR)
  else:
    R = .5*pi_*(1-99/(162*x))*np.exp(-x)
    return R


# Contribution from a cell
def get_Fnu_cell(nuobs, Tobs, data, env,
  Ng=20, norm=True, width_tol=1.1, r_ref=1.2, Nmin=2, Nmax=20):
  '''
  F_nu(T) from a cell (dataframe with the cell history)
  array of size nuobs array
  Ng is the number of log bins in [gmin, gmax] for calculation
  '''

  Fnu_out = np.zeros(nuobs.shape)

  cell_0 = data.iloc[0]
  K0 = norm_plaw_distrib(cell_0.gmin, cell_0.gmax, env.psyn)
  nu_B0 = get_variable(cell_0, 'nu_B', env)

  # no need to calculate contributions if nu_M outside of observed freqs 
  gmax_cut = max(1., np.sqrt(nuobs.min()/nu_B0))
  gmax_arr = data.gmax.to_numpy()
  N = gmax_arr.size
  # np.searchsorted works in ascending order
  jcut = N - np.searchsorted(gmax_arr[::-1], gmax_cut, side = "right")
  print(f'Calculating over {jcut} sim iterations')
  jcut = max(jcut, 1)
  jcut = min(jcut, N-1)
  for j in range(jcut):
    if (j%100==0):
      print(f'Iteration {j} of {jcut}')
    hydro = data.iloc[j]
    hydro_next = data.iloc[j+1]
    Fnu = get_Fnu_hydrostep(nuobs, Tobs, hydro, hydro_next, K0, env, norm, width_tol, r_ref, Nmin, Nmax)
    Fnu_out += Fnu
  return Fnu_out

def get_Fnu_hydrostep(nuobs, Tobs, cell, cell_next, K0, env,
  Ng=20, norm=True, width_tol=1.1, r_ref=1.5, Nmin=2, Nmax=20):
  '''
  F_\nu (T) from a hydro step
  '''

  Fnu_out = np.zeros(nuobs.shape)
  steps = split_hydrostep(cell, cell_next, env, r_ref, Nmin, Nmax)
  Nj = len(steps)
  for j in range(Nj):
    step = steps.iloc[j]
    Fnu = get_Fnu_step(nuobs, Tobs, step, K0, env, Ng, norm, width_tol)
    Fnu_out += Fnu
  return Fnu_out

# Contribution from a single time bin
def get_Fnu_step(nuobs, Tobs, step, K0, env, Ng=20, norm=True, width_tol=1.1):
  '''
  F_\nu (T) from a (cooling) step
  '''
  
  D = get_variable(step, "Dop", env)
  nup = nuobs/D
  lfac = get_variable(step, 'lfac', env)
  tT = Tobs_to_tildeT(Tobs, step, env)
  Fnu = get_Lnu_comov(nup, step, K0, env, Ng, norm, width_tol)
  Fnu *= 2*lfac * tT**-2
  if norm:
    Fnu /= 2*env.lfac0
  else:
    Fnu *= env.zdl
  return Fnu

def get_Lnu_comov(nup, step, K0, env, Ng=20, norm=True, width_tol=1.1):
  '''
  L'_\nu' from a (cooling) step
    as a function of \nu'/\nu'_m
  '''

  nup_B = get_variable(step, 'nup_B', env)
  rsc = 1.
  if env.geometry == 'cartesian':
    # assume geometrical scaling nu' \propto R^(-1)
    rsc = step.x * c_/env.R0
  nup_B /= rsc
  tnu = nup/nup_B

  Ttz = get_variable(step, "Tth", env)/(1+env.z)
  V3p = get_variable(step, "V3p", env)
  Lnu = get_epnu(tnu, step, K0, env, Ng, width_tol)
  Lnu *= rsc * V3p/Ttz

  if norm:
    Lnu /= env.L0p_Rshape
  return Lnu

def get_epnu(tnu_arr, step, K0, env, Ng=20, width_tol=1.1,
    func_cooling=gamma_synCooled, func_distrib=distrib_plaw_cooled, func_emiss=syn_emiss_exact):
  '''
  Delta e'_\nu' of a step (assuming constant P'_\nu' in the bin)
    as a function of \nu'/\nu'_B
  K0 as variable because it depends on initial gma_min/max of the distrib
  '''
  p   = env.psyn
  gmin, gmax = step.gmin, step.gmax
  if gmax <= 1.:
    return np.zeros(tnu_arr.shape)
  gma_keys = [key for key in step.keys() if 'gm' in key]
  Pmax = get_variable(step, "Pmax", env)
  K = K0
  xi_N = 1.
  if gmin < 1.:
    xi_N = derive_xiDN(gmin, gmax, p)
    gmin = 1.
  if gmax/gmin <= width_tol:
    K = 1
    gma_mn = np.sqrt(gmin*gmax)
    #gma_mn = gmin
    enu = np.array([func_emiss(gma_mn, tnu) for tnu in tnu_arr])
  else:
    if len(gma_keys) > 2:
      gmas_arr = step[gma_keys].to_numpy(copy=True)
      enu = Pnu_instant_bins(tnu_arr, gmas_arr, env, func_distrib, func_emiss)
    else:
      enu = Pnu_instant(tnu_arr, gmin, gmax, env, Ng, func_distrib, func_emiss)
  enu *= xi_N*K*step['dtp']*Pmax
  return enu


def Pnu_instant_bins(tnu_arr, gmas_arr, env,
      func_distrib=distrib_plaw_cooled, func_emiss=syn_emiss_exact):
  '''
  Energy per unit freq per unit volume and time at normalized time tt
    as a function of \nu'/\nu'_B, in units P'_e,max
  from an array of logarithmic bins
  ''' 
  p = env.psyn
  gmax = gmas_arr[-1]
  def func(gma, tnu):
    return func_distrib(gma, p, 1/gmax)*func_emiss(gma, tnu)
  
  Pnu = np.zeros(tnu_arr.shape)
  # center values in bins for calculation
  gmas_ctr = np.sqrt(gmas_arr[:-1] * gmas_arr[1:])
  dgmas = np.diff(gmas_arr)
  for i, tnu in enumerate(tnu_arr):
    for gma, dgma in zip(gmas_ctr, dgmas):
      Pnu[i] += func(gma, tnu)*dgma
  return Pnu

def Pnu_instant(tnu_arr, gmin, gmax, env, Ng=20,
      func_distrib=distrib_plaw_cooled, func_emiss=syn_emiss_exact):
  '''
  Energy per unit freq per unit volume and time at normalized time tt
    as a function of \nu'/\nu'_B, in units P'_e,max
  Separates the range \gamma_min/max into log bins for calculation
  '''
  p = env.psyn
  def func(gma, tnu):
    return func_distrib(gma, p, 1/gmax)*func_emiss(gma, tnu)

  Pnu = np.zeros(tnu_arr.shape)
  gmas_arr = np.geomspace(gmin, gmax, Ng)

  for i, tnu in enumerate(tnu_arr):
    dP_arr = [func(gma, tnu) for gma in gmas_arr]
    Pnu[i] = np.trapezoid(dP_arr, gmas_arr)
  return Pnu


# Analytic constant-hydro case
def get_Fnu_cell_analytic(nuobs, Tobs, gmin0, gmax0, hydro_const, env,
    Nt=200, r_ref=1.2, Ng=20, width_tol=1.1):
  '''
  F_nu(T) for analytic case with constant hydrodynamics (cartesian geometry)
  Uses logarithmic time binning optimized for cooling

  Parameters:
  - nuobs: observed frequency array
  - Tobs: observation time
  - gmin0, gmax0: initial electron distribution bounds
  - hydro_const: dict with constant values {'rho', 'vx', 'p', 't0', 'x0'}
  - env: environment object
  - Nt: number of logarithmic time bins
  - r_ref: reference ratio for cooling sub-bins (gmax_i / gmax_{i+1})
  - Ng: number of gamma bins for distribution integration
  - width_tol: tolerance for narrow distribution approximation
  '''

  from phys_functions import derive_Lorentz, derive_Pmax

  # Extract constant hydro quantities
  rho = hydro_const['rho']
  vx = hydro_const['vx']
  p = hydro_const['p']
  t0 = hydro_const['t0']
  x0 = hydro_const['x0']

  # Compute derived constant quantities
  lfac = derive_Lorentz(vx)
  Pmax = derive_Pmax(rho, p, env.rhoscale, env.eps_B, env.xi_e)

  # Cooling time
  from variables import var2func
  syn_func, syn_inlist, syn_envlist = var2func['syn']
  # syn depends on rho, p, so we can compute it with constant values
  tc1 = 1.0 / syn_func(rho, p, env.rhoscale, env.eps_B)

  # Total evolution time in normalized units
  # Go from tt=0 until gmax cools to 1 (non-relativistic electrons)
  # From gamma_synCooled: gamma(tt) = gamma0 / (1 + gamma0*tt)
  # So: tt = 1/gamma - 1/gamma0
  # At t=0: tt_0 = 0 corresponds to gamma = gamma0
  # At t_final: gamma = 1 gives tt_final = 1/1 - 1/gamma0 ≈ 1
  gmax_final = 1.0
  tt_0 = 0.0  # Start at injection time
  tt_final = 1.0/gmax_final - 1.0/gmax0

  # Create logarithmic time bins in normalized time tt
  # Need to avoid tt=0, so start from a small value
  tt_min = max(1e-8, 1.0/(10*gmax0))  # Small but non-zero
  tt_bins = np.geomspace(tt_min, tt_final, Nt+1)
  dtt_arr = np.diff(tt_bins)
  tt_arr = tt_bins[:-1]  # Use left edge of each bin

  # Convert to proper time
  dtp_arr = dtt_arr * tc1
  tp_arr = (tt_arr - tt_bins[0]) * tc1

  # Lab frame time (for cartesian: t = tp * lfac)
  t_arr = t0 + tp_arr * lfac

  # Position evolution (for cartesian: x = x0 + vx * (t - t0))
  x_arr = x0 + vx * (t_arr - t0)

  # Lab frame time intervals (for computing dx)
  dt_arr = dtp_arr * lfac

  # Spatial intervals (for cartesian: dx = vx * dt)
  dx_arr = vx * dt_arr

  # Evolve gamma distribution
  gmin_arr = gamma_synCooled(tt_arr, gmin0)
  gmax_arr = gamma_synCooled(tt_arr, gmax0)

  # Initialize output
  Fnu_out = np.zeros(nuobs.shape)

  # Normalization for power-law distribution
  K0 = norm_plaw_distrib(gmin0, gmax0, env.psyn)

  # Loop over time bins
  for i in range(Nt):
    if gmax_arr[i] <= 1.0:
      break

    # Create a pseudo-step dictionary with the necessary fields
    step = {
      'gmin': gmin_arr[i],
      'gmax': gmax_arr[i],
      'dtp': dtp_arr[i],
      'rho': rho,
      'vx': vx,
      'p': p,
      't': t_arr[i],
      'x': x_arr[i],
      'dx': dx_arr[i],
      'dt': dt_arr[i]
    }

    # Convert to pandas Series for compatibility with get_Fnu_step
    import pandas as pd
    step_series = pd.Series(step)

    # Calculate flux contribution from this time bin
    Fnu = get_Fnu_step(nuobs, Tobs, step_series, K0, env, Ng, norm=True, width_tol=width_tol)
    Fnu_out += Fnu

  return Fnu_out


def get_cell_nuFnu_analytic(dist, env, Nt=200, Nnu=500, width_tol=1.1):
  '''
  Wrapper to compute spectrum for analytic constant-hydro case
  Uses optimal logarithmic time binning instead of simulation timesteps

  Parameters:
  - dist: dataframe from generate_cellDistrib (only first row is used for initial conditions)
  - env: environment object
  - Nt: number of logarithmic time bins for Riemann integration
  - Nnu: number of frequency points
  - width_tol: tolerance for narrow distribution approximation

  Returns:
  - nub: normalized frequency array (nu/nu0)
  - nF: normalized flux nu*F_nu
  '''

  # Get initial conditions from first timestep
  cell0 = dist.iloc[0]
  gmin0 = cell0.gmin
  gmax0 = cell0.gmax

  # Extract constant hydro values
  hydro_const = {
    'rho': cell0.rho,
    'vx': cell0.vx,
    'p': cell0.p,
    't0': cell0.t,
    'x0': cell0.x
  }

  # Create frequency array
  log_nuM = 2*np.log10(gmax0/gmin0)
  nub = np.logspace(-8, log_nuM+1, Nnu)
  nuobs = nub * env.nu0

  # Observation time
  Tobs = env.Ts

  # Compute spectrum with optimal binning
  Fnu = get_Fnu_cell_analytic(nuobs, Tobs, gmin0, gmax0, hydro_const, env,
                               Nt=Nt, width_tol=width_tol)

  # Return normalized values
  nF = nub * Fnu

  return nub, nF