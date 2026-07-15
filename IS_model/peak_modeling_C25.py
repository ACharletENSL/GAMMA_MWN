# -*- coding: utf-8 -*-
# @Author: acharlet

'''
Modeling observed flux from the collision of two ultrarelativistic cold shells
This model includes spherical effects
Contains:
  - peak_model:           (nu_pk, (nu F_nu)_pk)(tilde{T}, au, tau)
and intermediate functions:
  - _load_fittable
  - fitparams_from_au
  - smooth_bpl0
  - invGma2_cumint
  - compute_Tf
  - build_peaks_functions
'''

import os
import numpy as np
import pandas as pd
from functools import lru_cache
from scipy.optimize import brentq
from scipy.integrate import cumulative_trapezoid
from analytical_funcs import *

# directory holding this module (and the fit tables), so imports work from any CWD
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

##### fit tables (C25-specific)
@lru_cache(maxsize=None)
def _load_fittable(front):
  '''
  Fit table parsed once per front, columns as numpy arrays
  '''
  df = pd.read_csv(os.path.join(_MODULE_DIR, f'fullsweep_au_{front}.csv'), sep='\t')
  return {c: df[c].to_numpy() for c in df.columns}

def fitparams_from_au(au, reverse=True):
  '''
  Returns parameters obtain by fitting at any given a_u,
    piecewise linear interpolation in log space
  '''
  log_au = np.log10(au-1)
  front = 'RS' if reverse else 'FS'
  cols = _load_fittable(front)
  x = cols['log_aum']
  groups = {
    'lfac':['A', 'x_b', 'alpha', 's'],
    'nu':['A', 'x_b', 'alpha', 's'],
    'L':['A', 'x_b', 'alpha', 's'],
    'xi':['k', 'sat', 's']
  }
  popts = []
  for prefix, suffixes in groups.items():
    vals = [np.interp(log_au, x, cols[f'{prefix}_{s}']) for s in suffixes]
    popts.append(np.array(vals))
  return popts

##### fit-dependant
def smooth_bpl0(x, A, x_b, alpha, s, a_tol=1e-4, s_tol=1e-3):
  '''
  Smoothly joined broken power law, with 2nd index = 0
  floors a_tol and s_tol to avoid extreme behaviors
  '''
  if np.abs(alpha) <= a_tol:
    # very weak index -> constant value
    return A
  s = max(np.abs(s), s_tol)
  q = alpha*s
  y = x/x_b
  low = y**(-alpha)
  high = (.5 * (1. + y**(1/s)))**q
  return A * low * high

def invGma2_cumint(popt_lfac, x_max, N=1000):
  '''
  Table of I(x) = int_1^x dx'/Gamma(x')^2, Gamma(x) = smooth_bpl0(x, *popt_lfac)
  The integrand is smooth so a cumulative trapezoid on a fine grid replaces
    repeated adaptive quad calls
  '''
  x = np.linspace(1., x_max, N)
  # broadcast: smooth_bpl0 returns scalar A in its small-alpha branch
  Gma = np.broadcast_to(smooth_bpl0(x, *popt_lfac), x.shape)
  I = cumulative_trapezoid(1./Gma**2, x, initial=0.)
  return x, I

def compute_Tf(tau, au, popt_lfac, reverse=True):
  '''
  Normalized shock crossing time in observer frame from tau = t_on/t_off
  varying Lorentz factor, defined by f(x) = smooth_bpl0(x, *popt_lfac)
  Returns (Tf, xf): crossing time and crossing radius
  '''

  au2 = au*au
  gRS2, gFS2 = g2_from_au(au)
  KRS, KFS = tau_to_dRR0(au)
  g2, K = (gRS2, KRS) if reverse else (gFS2, KFS)
  k = 2 / ((au2 + 1)*g2)
  D = k * (au2 - 1)

  # bracket size from constant Gamma
  xf_cst = K*tau
  x_grid, I_grid = invGma2_cumint(popt_lfac, 1. + 3*xf_cst)
  if reverse:
    res_grid = I_grid - k*(x_grid-1) - D*tau
  else:
    res_grid = k*au2*(x_grid-1) - I_grid - D*tau
  residual = lambda x: np.interp(x, x_grid, res_grid)

  xf = brentq(residual, 1., 1. + 3*xf_cst)
  return 1. + np.interp(xf, x_grid, I_grid), xf

def build_peaks_functions(Tf, g2, popts, N=1000, xf=None, renormFlux=True):
  '''
  Builds tilde{nu}(tilde{T}) and tilde{nu F_nu}(tilde{T}) from
    g2:         g_sh^2, downstream region to shock LF
    Tf:         tilde{T}_f, tansition from rise to HLE
    popts:      parameters for func_Gma, func_nu, func_Lbol, xi_eff
    N:          number of points for the x_L(tilde{T}) table
    xf:         crossing radius, so that the table spans exactly T in [1, Tf]
    renormFlux: renormalize nu F_nu to its maximal value over the track
  '''
  popt_lfac, popt_nu, popt_L, popt_xi = popts  # ShSt unused for the peaks
  # Gamma_d, nu'_m, L'_bol with R/R_0
  # remove small numerical artifacts (~1%) at x=1
  def func_Gma(x):
    Gma = smooth_bpl0(x, *popt_lfac)/smooth_bpl0(1, *popt_lfac)
    return Gma
  def func_nu(x):
    nuR = smooth_bpl0(x, *popt_nu)/smooth_bpl0(1, *popt_nu)
    return nuR/x
  def func_Lbol(x):
    L = smooth_bpl0(x, *popt_L)/smooth_bpl0(1, *popt_L)
    return L

  # build x_L(tilde{T})
  x_max = xf if xf is not None else Tf+1e-2
  x_grid, I_grid = invGma2_cumint(popt_lfac, x_max, N)
  T_grid = 1. + I_grid
  xL_of_T = lambda T: np.interp(T, T_grid, x_grid)

  # rising flux
  def func_rise(T, k, xi_sat, s_eats):
    x_L = xL_of_T(T)
    t = np.maximum(g2*(T-1), 1e-10)
    xi_max = (t**(-s_eats) + xi_sat**(-s_eats))**(-1/s_eats)
    xi_eff = k*xi_max
    x_eff = x_L/(1+xi_eff/g2)
    Dop = func_Gma(x_eff)/(1+xi_eff)
    nupk = Dop * func_nu(x_eff)
    nFpk = 3*xi_max*Dop**4*func_Lbol(x_eff)/func_Gma(x_L)**2
    return nupk, nFpk

  # high latitude emission
  def func_HLE(T, k, xi_sat, s_eats, nupk_f, nFpk_f):
    # T = tilde{T}/tilde{T}_f here
    Teff = (1-g2) + g2*T
    nupk = nupk_f / Teff
    nFpk = nFpk_f / Teff**3
    return nupk, nFpk
  
  # normalization: max of nu F_nu over the track, reached during the rise
  # (HLE only decays from the rise endpoint), computed once on the rise grid
  nF_norm = 1.
  if renormFlux:
    nF_norm = func_rise(T_grid, *popt_xi)[1].max()

  # rise + HLE
  def func_peaks(T):
    mask1 = (T <= Tf)
    mask2 = ~mask1
    nupk, nFpk = np.empty_like(T), np.empty_like(T)
    nupk[mask1], nFpk[mask1] = func_rise(T[mask1], *popt_xi)
    rise_f = np.array(func_rise(Tf, *popt_xi))
    nupk_f, nFpk_f = rise_f[0], rise_f[1]
    nupk[mask2], nFpk[mask2] = func_HLE(T[mask2]/Tf, *popt_xi, nupk_f, nFpk_f)
    nFpk /= nF_norm
    return nupk, nFpk

  return func_peaks

def peak_model(T, au, tau, reverse=True, renormFlux=True):
  '''
  tilde{nu}_pk, tilde{nu F_nu}_pk with tilde{T} for a given front
  renormFlux renormalizes nu F_nu to its max value
  '''

  popts = fitparams_from_au(au, reverse)
  g2 = g2_from_au(au)[0 if reverse else 1]
  Tf, xf = compute_Tf(tau, au, popts[0], reverse)
  func_peaks = build_peaks_functions(Tf, g2, popts, xf=xf, renormFlux=renormFlux)
  nupk, nFpk = func_peaks(T)

  return nupk, nFpk



