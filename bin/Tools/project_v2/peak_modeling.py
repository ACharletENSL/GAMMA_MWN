# -*- coding: utf-8 -*-
# @Author: acharlet


import numpy as np
import pandas as pd
from functools import lru_cache
from scipy.optimize import brentq
from scipy.integrate import cumulative_trapezoid
from IO import GAMMA_dir

##### a_u dependent quantities
def relLfac_from_au(au):
  '''
  Relative Lorentz factors at each shock front
  '''

  K = np.sqrt(2 / (1 + au*au))
  G21 = .5 * (K*au + 1/(K*au))
  G34 = .5 * (K + 1/K)
  return G34, G21

def g2_from_au(au):
  '''
  Ratios of downstream L.F. to shock front L.F., squared
  '''

  K = np.sqrt(2 / (au**2+1))
  G34, G21 = relLfac_from_au(au)
  gFS2 = (K*au - 4*G21)/(1/(K*au) - 4*G21)
  gRS2 = (4*G34 - K)/(4*G34 - 1/K)
  return gRS2, gFS2

def tau_to_dRR0_cst(au):
  '''
  Proportionality factor between the crossed radius by the shock in units R0
    and the ratio of activity time over off time for the source,
    for constant velocity shock fronts
  '''

  au2 = au*au
  gRS2, gFS2 = g2_from_au(au)
  KFS = 2 * (au2 - 1) / (2*au2 - (au2 + 1)*gFS2)
  KRS = 2 * (au2 - 1) / ((au2 + 1)*gRS2 - 2)
  return KRS, KFS

def ratios_RSvFS_from_au(au):
  '''
  Ratios between normalizations of RS and FS
  '''
  G34, G21 = relLfac_from_au(au)
  b21 = np.sqrt(1. - G21**-2)
  b34 = np.sqrt(1. - G34**-2)
  gRS2, gFS2 = g2_from_au(au)

  ratio_T = (gRS2/gFS2)
  fac1 = np.sqrt((G34/G21)*((G21+1)/(G34+1)))
  fac2 = ((G34-1)/(G21-1))**2
  ratio_nu = fac1 * fac2
  ratio_F = fac1 * (b34/b21) / fac2
  return ratio_T, ratio_nu, ratio_F

##### fitted parameters and intermediate functions for C25 model
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

@lru_cache(maxsize=None)
def _load_fittable(front):
  '''
  Fit table parsed once per front, columns as numpy arrays
  '''
  folder = GAMMA_dir + '/extracted_data/'
  filename = folder + f'fullsweep_au_{front}.csv'
  df = pd.read_csv(filename, sep='\t')
  todrop = ['Tf', 't_max'] + [f'ShSt_{s}' for s in ['A', 'x_b', 'alpha', 's']]
  df = df.drop(columns=todrop)
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
  KRS, KFS = tau_to_dRR0_cst(au)
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

##### R24 model: constant Lorentz factor
def R24_peak_model(T, au, tau, reverse=True):
  '''
  Peak frequency and flux at this frequency with observed time T
  T:        tilde{T}, time in observers frame normalized to angular time
  au:       a_u, proper velocity contrast
  tau:      tau_i=t_on,i/t_off, source activity time during emission of shell i
  reverse:  boolean, True to model contribution from the RS, False for the FS
  '''

  i = 0 if reverse else 1
  # downstream L.F. to shock front L.F. ratio
  g2 = g2_from_au(au)[i]

  # break time
  K = tau_to_dRR0_cst(au)[i]
  dR = K * tau
  Tf = 1 + dR
  T1f = (1. - g2) + g2 * Tf

  # model functions
  ### rising part
  def nu_rise(x):
    return 1./x
  def nF_rise(x):
    return 1. - 1/x**3
  ### HLE
  nu_f = nu_rise(Tf)
  nF_f = nF_rise(T1f)
  def nu_HLE(x):
    return nu_f/x
  def nF_HLE(x):
    return nF_f/x**3
  
  rise = (T <= Tf)
  T1 = (1. - g2) + g2 * T
  T2 = (1. - g2) + g2 * T / Tf
  nu_pk = np.where(rise, nu_rise(T), nu_HLE(T2))
  nF_pk = np.where(rise, nF_rise(T1), nF_HLE(T2))
  return nu_pk, nF_pk

def R24_correlation_model(nu_pks, au, tau, reverse=True):
  '''
  Peak flux vs peak frequency, normalized, from the collision of UR shells
  parameters:
    nu_pks:     normalized peak observed frequency
      type np.ndarray, replace "np.where" by if/else if you prefer a float as input
    au:         a_u, proper velocity contrast
    tau:        tau=t_on,i/t_off, source activity time during emission of shell i 
                  over the time between the emission of the two shells
    reverse:    boolean, True to model contribution from the RS, False for the FS
  '''

  i = 0 if reverse else 1
  # downstream L.F. to shock front L.F. ratio
  g2 = g2_from_au(au)[i]

  # break frequency
  K = tau_to_dRR0_cst(au)[i]
  dR = K * tau
  Tf = 1 + dR
  nu_bk = 1/Tf

  # model functions
  rise = (nu_pks >= nu_bk)
  def flux_rise(nu):
    return 1 - ((1-g2) + g2/nu)**-3
  def flux_HLE(nu):
    nF_bk = flux_rise(nu_bk)
    return nF_bk * (nu/nu_bk)**3
  nF = np.where(rise, flux_rise(nu_pks), flux_HLE(nu_pks))
  return nF

##### C25 model: variable Lorentz factor, with table
def build_peaks_functions_xi(Tf, g2, popt_lfac, popt_nu, popt_L, N=1000, xf=None):
  '''
  Builds tilde{nu}(tilde{T}), tilde{nu F_nu}(tilde{T}) with the EATS angular
    parameters popt_xi = (k, xi_sat, s_eats) left free in the signature, i.e.
    returns func_peaks(T, popt_xi). The x_L(tilde{T}) table depends only on
    popt_lfac, so it is built once here and reused for every popt_xi, which makes
    this the function to use when fitting popt_xi.
    g2:        g_sh^2, downstream region to shock LF
    Tf:        tilde{T}_f, transition from rise to HLE
    popt_lfac, popt_nu, popt_L: parameters for func_Gma, func_nu, func_Lbol
    N:         number of points for the x_L(tilde{T}) table
    xf:        crossing radius, so that the table spans exactly T in [1, Tf]
  '''
  # Gamma_d, nu'_m, L'_bol with R/R_0
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
  def func_HLE(T, nupk_f, nFpk_f):
    # T = tilde{T}/tilde{T}_f here
    Teff = (1-g2) + g2*T
    nupk = nupk_f / Teff
    nFpk = nFpk_f / Teff**3
    return nupk, nFpk

  # rise + HLE, with popt_xi = (k, xi_sat, s_eats) free
  def func_peaks(T, popt_xi):
    T = np.atleast_1d(np.asarray(T, dtype=float))
    mask1 = (T <= Tf)
    mask2 = ~mask1
    nupk, nFpk = np.empty_like(T), np.empty_like(T)
    nupk[mask1], nFpk[mask1] = func_rise(T[mask1], *popt_xi)
    rise_f = np.array(func_rise(Tf, *popt_xi))
    nupk_f, nFpk_f = rise_f[0], rise_f[1]
    nupk[mask2], nFpk[mask2] = func_HLE(T[mask2]/Tf, nupk_f, nFpk_f)
    return nupk, nFpk

  return func_peaks

def build_peaks_functions(Tf, g2, popts, N=1000, xf=None):
  '''
  Builds tilde{nu}(tilde{T}) and tilde{nu F_nu}(tilde{T}) from
    g2:     g_sh^2, downstream region to shock LF
    Tf:     tilde{T}_f, tansition from rise to HLE
    popts:  parameters for func_Gma, func_nu, func_Lbol, xi_eff
    N:      number of points for the x_L(tilde{T}) table
    xf:     crossing radius, so that the table spans exactly T in [1, Tf]
  Wraps build_peaks_functions_xi with popt_xi = popts[3] baked in, so the
    returned function has signature func_peaks(T).
  '''
  popt_lfac, popt_nu, popt_L, popt_xi = popts  # ShSt unused for the peaks
  func_peaks_xi = build_peaks_functions_xi(Tf, g2, popt_lfac, popt_nu, popt_L, N=N, xf=xf)
  def func_peaks(T):
    return func_peaks_xi(T, popt_xi)

  return func_peaks

def C25_peak_model(T, au, tau, reverse=True):
  '''
  tilde{nu}_pk, tilde{nu F_nu}_pk with tilde{T} for a given front
  '''

  popts = fitparams_from_au(au, reverse)
  g2 = g2_from_au(au)[0 if reverse else 1]
  Tf, xf = compute_Tf(tau, au, popts[0], reverse)
  func_peaks = build_peaks_functions(Tf, g2, popts, xf=xf)

  nupk, nFpk = func_peaks(T)
  return nupk, nFpk

def C25_correlation_model(nupk, au, tau, reverse=True, N=200, Tmax=5.):
  '''
  (nu F_nu)_pk/nu_0F_0 as a function of nu_pk/nu_0 and model parameters
  Computes nu_pk and (nu F_nu)_pk over N log spaced points from 1 to 1+Tmax,
    interpolation gives the (nu F_nu)_pk (nu_pk) track
  '''

  T = np.geomspace(1, 1+Tmax, N)
  nu_track, nF_track = C25_peak_model(T, au, tau, reverse)
  # sanity check if nu_pk is monotonic
  dnu = np.diff(nu_track)
  if not (np.all(dnu <= 0.) or np.all(dnu >= 0.)):
    raise ValueError(
      "nu_pk is not monotonic along the track (it loops): the mapping "
      "nu_pk -> nF_pk is multivalued. Use peak_model() and select a branch.")
  order = np.argsort(nu_track)
  return np.interp(nupk, nu_track[order], nF_track[order], left=np.nan, right=np.nan)

#### Time-binning 
def correlation_onebin(T1, T2, au, tau, reverse=True,
  peak_model=C25_peak_model, Nbins=20, renormFlux=True):
  '''
  Returns values for peak freq and flux between normalized times T1 and T2
    this is equivalent (to <~ % for small T2-T1) to integrating the flux
    between T1 and T2 and returning peak frequency and flux over this bin
  '''

  Tlow, Tup = min(T1, T2), max(T1, T2)
  T_bins = np.geomspace(Tlow, Tup, Nbins)
  nu_bins, nF_bins = peak_model(T_bins, au, tau, reverse, renormFlux)
  F_bins = nF_bins/nu_bins
  DT = Tup - Tlow
  int_F  = np.trapezoid(F_bins, T_bins)
  int_nF = np.trapezoid(nF_bins, T_bins)
  ave_nF = int_nF/DT    # /DT or not /DT? That is the question
  ave_nu = int_nF/int_F
  return ave_nu, ave_nF

def correlation_binned(au, tau, Tarr, peak_model, reverse=True,
  N=1000, renormFlux=True):
  '''
  Computes couples (nu_pk/nu0 - nuFnu_pk/nuFnu_max) over the given time bins
  Tarr contains the edges of the time bins, normalized (T=1 at pulse start), sorted.
  In each bin, nu_pk and nuFnu_pk are estimated as means over this bin (flux-weighted for nu)
  Returns (ave_nu, ave_nF), each of length len(Tarr)-1.
    au, tau:     model parameters (proper velocity contrast, t_on/t_off)
    peak_model:  the chosen peak function, peak_modeling_C25/R24.peak_model
    N:           total number of points to evaluate peak_model once over the full range
  '''

  edges = np.sort(np.asarray(Tarr, dtype=float))
  # single peak_model evaluation on an N-point grid spanning the full range,
  # with the bin edges forced in as nodes for exact per-bin integration
  T_grid = np.union1d(np.geomspace(edges[0], edges[-1], N), edges)
  nu, nF = peak_model(T_grid, au, tau, reverse, renormFlux)
  F = nF/nu
  # per-bin integrals as differences of the cumulative integral at the edges
  cum_nF = cumulative_trapezoid(nF, T_grid, initial=0.)
  cum_F  = cumulative_trapezoid(F,  T_grid, initial=0.)
  idx = np.searchsorted(T_grid, edges)
  int_nF = cum_nF[idx[1:]] - cum_nF[idx[:-1]]
  int_F  = cum_F[idx[1:]]  - cum_F[idx[:-1]]
  DT = edges[1:] - edges[:-1]
  ave_nF = int_nF/DT       # mean nu F_nu over the bin
  ave_nu = int_nF/int_F    # flux-weighted mean peak frequency
  return ave_nu, ave_nF

def correlation_binned_full(au, tau, Tarr, peak_model, reverse=True,
  N=1000, Nnu=200, alpha=-0.5, beta=-1.25, renormFlux=True):
  '''
  Full-spectrum equivalent of correlation_binned: in each bin the per-instant
    spectra (Band shape S around each (nu_pk, nuFnu_pk)) are superposed and
    integrated in time, then (nu_pk, nuFnu_pk) is read off as the peak of the
    resulting bin-integrated nu F_nu spectrum.
  Tarr contains the edges of the time bins, normalized (T=1 at pulse start), sorted.
  Returns (ave_nu, ave_nF), each of length len(Tarr)-1.
    au, tau:      model parameters (proper velocity contrast, t_on/t_off)
    peak_model:   the chosen peak function, peak_modeling_C25/R24.peak_model
    N:            total number of points to evaluate peak_model once over the full range
    Nnu:          number of frequency points for the spectral grid
    alpha, beta:  Band-function slopes below/above the peak
  '''

  edges = np.sort(np.asarray(Tarr, dtype=float))
  # single peak_model evaluation on an N-point grid spanning the full range,
  # with the bin edges forced in as nodes so each bin is an exact sub-slice
  T_grid = np.union1d(np.geomspace(edges[0], edges[-1], N), edges)
  nu, nF = peak_model(T_grid, au, tau, reverse, renormFlux)
  idx = np.searchsorted(T_grid, edges)
  # each bin spans its own frequency range, so the spectral grid is per-bin
  Nbin = len(edges) - 1
  ave_nu, ave_nF = np.empty(Nbin), np.empty(Nbin)
  for k in range(Nbin):
    sl = slice(idx[k], idx[k+1] + 1)
    T_bins, nu_bins, nF_bins = T_grid[sl], nu[sl], nF[sl]
    nub = np.logspace(np.log10(nu_bins.min()) - 0.5, np.log10(nu_bins.max()) + 0.5, Nnu)
    x = nub[None, :] / nu_bins[:, None]                  # (Nsub, Nnu)
    nF_full = nF_bins[:, None] * x * S(x, alpha, beta)   # (Nsub, Nnu)
    # nF_int is integrated in time; divide by the bin width to get the
    # time-averaged peak flux, consistent with correlation_binned (int_nF/DT)
    nF_int = np.trapezoid(nF_full, T_bins, axis=0) / (edges[k+1] - edges[k])
    imax = np.argmax(nF_int)
    ave_nu[k], ave_nF[k] = nub[imax], nF_int[imax]
  return ave_nu, ave_nF

def fluence_halftime(au, tau, Tarr, peak_model, reverse=True, N=1000, renormFlux=True):
  '''
  For each bin defined by the edges in Tarr, returns T_half, the time at which half of
    the bin's fluence (time-integrated peak flux nuFnu_pk) has been emitted:
      int_{T_k}^{T_half} nuFnu_pk dT = 0.5 * int_{T_k}^{T_{k+1}} nuFnu_pk dT
  Returns (T_half, nu_half, nF_half), each of length len(Tarr)-1:
    T_half           : half-fluence (median) time in each bin
    nu_half, nF_half : peak_model on the track at T_half (for plotting on the curve)
  Note: T_half is invariant under flux renormalization (cancels in the ratio).
  '''
  edges = np.sort(np.asarray(Tarr, dtype=float))
  # one peak_model evaluation over the whole range, edges forced in as nodes
  T_grid = np.union1d(np.geomspace(edges[0], edges[-1], N), edges)
  nu, nF = peak_model(T_grid, au, tau, reverse, renormFlux)
  # cumulative fluence; half-target per bin, inverted by interpolation
  cum = cumulative_trapezoid(nF, T_grid, initial=0.)
  idx = np.searchsorted(T_grid, edges)
  half = 0.5 * (cum[idx[:-1]] + cum[idx[1:]])
  T_half = np.interp(half, cum, T_grid)
  # values on the curve at T_half, from the same fine grid (single peak_model call)
  nu_half = np.interp(T_half, T_grid, nu)
  nF_half = np.interp(T_half, T_grid, nF)
  return T_half, nu_half, nF_half

def photon_halftime(au, tau, Tarr, peak_model, reverse=True, N=1000, renormFlux=True):
  '''
  Photon-flux analogue of fluence_halftime: for each bin in Tarr, returns T_half, the time
    at which half the bin's PHOTON fluence has been emitted. Band-free proxy: the photon rate
    is taken proportional to nuFnu_pk / nu_pk (energy flux per characteristic photon energy):
      int_{T_k}^{T_half} (nuFnu_pk/nu_pk) dT = 0.5 * int_{T_k}^{T_{k+1}} (nuFnu_pk/nu_pk) dT
  Returns (T_half, nu_half, nF_half), each of length len(Tarr)-1 (peak_model on the track at
    T_half, for plotting on the nu_pk / nuFnu_pk curves). T_half is invariant under renormFlux.
  '''
  edges = np.sort(np.asarray(Tarr, dtype=float))
  # one peak_model evaluation over the whole range, edges forced in as nodes
  T_grid = np.union1d(np.geomspace(edges[0], edges[-1], N), edges)
  nu, nF = peak_model(T_grid, au, tau, reverse, renormFlux)
  # cumulative photon fluence; half-target per bin, inverted by interpolation
  Nph = nF/nu                                  # photon-rate proxy (= F_nu at the peak)
  cum = cumulative_trapezoid(Nph, T_grid, initial=0.)
  idx = np.searchsorted(T_grid, edges)
  half = 0.5 * (cum[idx[:-1]] + cum[idx[1:]])
  T_half = np.interp(half, cum, T_grid)
  # values on the curve at T_half, from the same fine grid
  nu_half = np.interp(T_half, T_grid, nu)
  nF_half = np.interp(T_half, T_grid, nF)
  return T_half, nu_half, nF_half
