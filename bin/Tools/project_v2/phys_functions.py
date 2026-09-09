# -*- coding: utf-8 -*-
# @Author: acharlet

'''
This file contains functions to derive physical variables
!!! IN NATURAL/CODE UNITS !!!
'''

# Imports
# --------------------------------------------------------------------------------------------------
import numpy as np
import math
from phys_constants import *
from numba import jit, njit, prange, vectorize
from scipy.signal import savgol_filter
from scipy.special import hyp2f1

# General functions
# --------------------------------------------------------------------------------------------------
def logslope(x1, y1, x2, y2):
  return (np.log10(y2)-np.log10(y1))/(np.log10(x2)-np.log10(x1))

def logslope_arr(x, y):
  '''
  Slope of np.arrays
  '''
  logx = np.log(x)
  logy = np.log(y)
  s = np.gradient(logy, logx)
  return s

def savgol_smooth(array, window=7, polyorder=3):
  window = max(window, polyorder + 2)  # savgol constraint
  if window % 2 == 0:
    window += 1  # savgol requires odd window
  out = savgol_filter(array, window, polyorder)
  return out

def init_slope(x, y, i_smp=20):
  return logslope(x[0], y[0], x[i_smp], y[i_smp])

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

def smooth_bpl(x, x_b, alpha, beta, s=1.):
  '''
  Smoothly joined broken power-law with smoothing factor s
  '''
  s = np.sign(beta-alpha)*np.abs(s)
  x_a = (1 - x_b**s)**(1/s)
  term_a = (x_a * x**alpha)**s
  term_b = (x_b * x**beta)**s
  return (term_a + term_b)**(1./s)

def smooth_bpl0(x, x_b, alpha, s=1.):
  '''
  Same as smooth_bpl but beta=0
  '''
  return smooth_bpl(x, x_b, alpha, 0., s)

def smooth_bpl_apy(x, A, x_b, alpha, beta, s):
  '''
  as defined in astropy, evaluated in log-space so the smoothing term never
  overflows (large |ln y|/s): equivalent to A * y**-alpha * (.5(1+y**(1/s)))**((alpha-beta)s).
  s > 0
  '''
  if abs(alpha - beta) < 1e-12:
    return A * (x/x_b)**(-alpha)  # = A when alpha = 0
  s = max(np.abs(s), 1e-3)
  q = (alpha - beta)*s
  ln_y = np.log(x/x_b)
  # log(1 + y**(1/s)) = logaddexp(0, ln_y/s), stable at both extremes
  return A * np.exp(-alpha * ln_y + q * (np.log(0.5) + np.logaddexp(0., ln_y / s)))

def smooth_bpl0_apy(x, A, x_b, alpha, s):
  return smooth_bpl_apy(x, A, x_b, alpha, 0., s)


###### Synchrotron emission function R(x)
# coeffs from table 1 in Finke, Dermer & Böttcher 08
coeffs_bel1 = [-0.35775237, -0.83695385, -1.1449608,
    -0.68137283, -0.22754737, -0.031967334]
coeffs_abv1 = [-0.35842494, -0.79652041, -1.6113032,
    0.26055213, -1.6979017, 0.032955035]

def _func_R_exact(x):
  '''
  R function see eqns (18) to (20) in Finke, Dermer & Böttcher 2008
  Vectorized: accepts scalar or ndarray. Piecewise reference implementation;
  func_R below is a tabulated fast path built from this.
  '''
  x = np.asarray(x, dtype=float)
  xs = np.where(x > 0., x, 1.)   # safe placeholder, branches masked by select
  y = np.log10(xs)
  with np.errstate(under='ignore', over='ignore'):
    R = np.select(
        [x < 0.01, x < 1., x < 10.],
        [1.80842*xs**(1./3.),
         10**np.polyval(coeffs_bel1[::-1], y),
         10**np.polyval(coeffs_abv1[::-1], y)],
        default=.5*pi_*(1.-99./(162.*xs))*np.exp(-xs))
  return R if R.ndim else float(R)

# Precompute func_R once on a dense log grid and evaluate by log-log interpolation
# in a fused numba kernel, replacing the per-call np.select + 2 polyval + log10.
# Outside the grid we use analytic asymptotes
_R_XMIN, _R_XMAX, _R_NGRID = 1e-6, 60., 8192
_R_xgrid = np.geomspace(_R_XMIN, _R_XMAX, _R_NGRID)
_R_logR  = np.ascontiguousarray(np.log(_func_R_exact(_R_xgrid)))
_R_LOGXMIN = np.log(_R_XMIN)
_R_INV_DLOG = (_R_NGRID - 1) / np.log(_R_XMAX / _R_XMIN)   # uniform log spacing

@njit(cache=True)
def _func_R_kernel(xf, logxmin, inv_dlog, logR, xmin, xmax):
  '''
  R(x) on a flat array: uniform-log-grid direct-index log-log blend, with the
  analytic low-x power law / high-x exp tail outside [xmin, xmax]. x<=0 -> 0.
  '''
  ng = logR.size
  out = np.zeros(xf.size)
  for i in range(xf.size):
    xi = xf[i]
    if xi <= 0.:
      continue
    elif xi < xmin:
      out[i] = 1.80842 * xi**(1./3.)
    elif xi > xmax:
      out[i] = 0.5*np.pi*(1. - 99./(162.*xi))*np.exp(-xi)
    else:
      u = (np.log(xi) - logxmin) * inv_dlog
      i0 = int(u)
      if i0 > ng - 2:
        i0 = ng - 2
      f = u - i0
      out[i] = np.exp((1.-f)*logR[i0] + f*logR[i0+1])
  return out

def func_R(x):
  '''
  R function (Finke, Dermer & Böttcher 2008 eqns 18-20), tabulated log-log fast
  path over _func_R_exact. Vectorized: accepts scalar or ndarray of any shape.
  '''
  x = np.asarray(x, dtype=float)
  xf = np.ascontiguousarray(np.atleast_1d(x).ravel())
  out = _func_R_kernel(xf, _R_LOGXMIN, _R_INV_DLOG, _R_logR, _R_XMIN, _R_XMAX)
  return float(out[0]) if x.ndim == 0 else out.reshape(x.shape)


R_LOW_COEF = 1.80842      # R(x) -> R_LOW_COEF * x**(1/3) as x -> 0 (_func_R_exact)

def syn_cutoff_R(u):
  '''
  Shape of the high-frequency cut-off of a synchrotron spectrum, as a multiplicative
  factor to apply to a power-law spectrum, for u = nu/nu_M with nu_M = gma_M**2 nu_B
  the burnoff frequency.

  Above nu_M the emission comes from the top of the electron distribution, so the
  spectrum rolls over with the shape of the single-electron emissivity P'_nu' ~ R(x),
  x = 2 nu'/(3 gma**2 nu'_B) = (2/3) u. Dividing R by its own low-x asymptote
  R_LOW_COEF*x**(1/3) makes the factor -> 1 well below the cut-off, so it leaves the
  power-law body untouched and only supplies the rolloff.

  Much shallower than exp(-u): R spreads its turnover over ~3 decades in x (at u=10 it
  leaves 5.4e-4 against exp's 4.5e-5). Measured on the sweep spectra, swapping exp for
  this cuts the rise-phase rms of the Granot & Sari fits from 0.133 to 0.029 dex.
  '''
  x = (2./3.)*np.asarray(u, dtype=float)
  return func_R(x)/(R_LOW_COEF*x**(1./3.))


# Quadrature for the lognormal average in syn_cutoff_R_smeared: a UNIFORM grid in s (dex),
# truncated at +-SMEAR_NSIG sigma, with the Gaussian weights renormalised to sum to 1.
# Deliberately not Gauss-Hermite: func_R is a table interpolation, so the integrand is only
# piecewise smooth and polynomial quadrature converges badly on it (measured non-monotonic,
# still ~4e-3 at 31 Gauss-Hermite nodes). A uniform grid converges cleanly instead -- see the
# ladder in syn_cutoff_R_smeared's docstring. Cost is one vectorised func_R call per node.
# 61, lowered from 121 on 2026-09-10 after measuring what the node count actually buys.
# The tabulation is the dominant cost of the segment route, and halving the nodes halves it
# (2.09x end to end, against 2.22x at 31 nodes -- i.e. 61 already captures nearly all of the
# available speedup, and below it the cost is the scan rather than the kernel). What it
# costs, measured over 120 bins of both shells spanning logr -5..+2, against 121 nodes:
#   nu_M     UNCHANGED, exactly, in every bin (it is picked off a 400-point grid)
#   sigma    UNCHANGED, exactly, in every bin (picked off SMEAR_SIGMAS' 0.01 dex grid)
#   s1, s2   median |rel| 1e-7 / 1e-6, worst 1e-4 -- against exponents quoted to 2 decimals
#   rms      median +3e-8 dex, worst 1.5e-6, against fit rms of 0.02-0.04 dex
#   s_ok     no bin changed status
# The shape error itself goes 3.8e-4 -> 9.0e-4 (the ladder below), still an order below the
# rms of every fit that uses it. Raise it back to 121 if the cut-off shape is ever the
# quantity under study rather than a nuisance divided out.
SMEAR_NODES = 61
SMEAR_NSIG = 4.0


def syn_cutoff_R_smeared(u, sigma):
  '''
  syn_cutoff_R for an emitting region that is NOT one zone: the cut-off shape when the
  contributing cells carry a spread of nu_M rather than a single value.

  WHY. syn_cutoff_R(u) is the single-electron rolloff and assumes ONE burnoff frequency. The
  observed spectrum sums cells, and each cell contributes the same rolloff SHAPE displaced in
  log-frequency by its own nu_M. Summing them is therefore a convolution in log nu, not a
  change of shape parameter:

      C_sigma(u) = INT R(u / 10**s) N(s; 0, sigma) ds,     sigma in DEX

  with the spread taken lognormal about the median nu_M. sigma -> 0 returns syn_cutoff_R
  exactly, and C_sigma -> 1 well below nu_M for any sigma (the weights are normalised), so
  this keeps the contract the unsmeared shape has: it leaves the power-law body untouched and
  supplies only the rolloff.

  WHAT IT FIXES. Fitting a single-zone R to a superposed rolloff biases nu_M HIGH, because the
  fit widens the only way it can -- by pushing the cut-off up. On a synthetic superposition of
  known sigma = 0.12 dex, a one-parameter R fit returns nu_M 1.299x the true median at an rms
  of 0.1007 dex, while this shape returns sigma = 0.120 and nu_M 0.971x at rms 0.0072. On the
  computed sweep spectra it improves the rolloff rms in every case (1.5-3x) and brings the
  fitted nu_M to within ~5% of the independent Granot & Sari whole-spectrum refit, against a
  systematic 10-12% disagreement with the unsmeared shape.

  sigma IS A MEASUREMENT, not a shape knob -- it is the dex spread of nu_M over the cells that
  are contributing. Fitted on the sweep it runs 0.02-0.10 dex and is systematically larger at
  the pulse peak (0.06-0.10) than on the rise (0.02-0.07), which is the expected ordering since
  the most cells contribute simultaneously at peak. It must be FITTED, not tabulated: holding a
  single global sigma = 0.07 costs +0.0126 dex of rms against fitting it per spectrum, ten
  times what holding the Granot & Sari smoothing costs (+0.0013), i.e. unlike s1/s2 this
  parameter is well constrained by the data.

  NB an alternative that looks cheaper -- stretching R in log frequency, R(u**(1/k)) -- was
  tested and is NOT adequate: on the same synthetic it reaches only rms 0.0357 and distorts
  nu_M by -24%. The convolution is doing real work.

  QUADRATURE. Uniform grid in s over +-SMEAR_NSIG sigma, Gaussian weights renormalised to 1
  (see SMEAR_NODES). Convergence against a 481-node reference, max |dC| over nu/nu_M in
  1e-3..30, at sigma = 0.05 / 0.10 / 0.20:
      31 nodes  1.9e-03 / 2.0e-03 / 2.0e-03
      61        9.0e-04 / 9.1e-04 / 9.1e-04
      121       3.8e-04 / 4.3e-04 / 4.5e-04
      241       1.2e-04 / 1.5e-04 / 1.6e-04
  It converges like 1/N, as expected for a piecewise-smooth integrand truncated at +-4 sigma.
  121 puts the shape error near 4e-4, i.e. ~2e-3 dex where the rolloff has fallen to C ~ 0.1 --
  more than an order below the ~0.02-0.04 dex rms of the fits that use it.

  COST. Callers scanning nu_M should NOT call this per trial value: C_sigma depends on u only,
  and changing nu_M is a pure shift in log u, so tabulate it once per sigma on a log-u grid and
  interpolate (measure_cutoff_nuM does exactly that). Evaluated naively inside a 400 x 21 scan
  this is ~3e8 kernel calls; tabulated it is 21.

  Vectorised over u; sigma is scalar.
  '''
  sigma = float(sigma)
  if sigma <= 1e-6:
    return syn_cutoff_R(u)
  u = np.asarray(u, dtype=float)
  s = np.linspace(-SMEAR_NSIG*sigma, SMEAR_NSIG*sigma, SMEAR_NODES)
  w = np.exp(-0.5*(s/sigma)**2)
  w /= w.sum()
  with np.errstate(divide='ignore', invalid='ignore'):
    vals = syn_cutoff_R(u[..., None]/10**s)
  return np.sum(vals*w, axis=-1)


def granot_sari_syn(nu, num, nuc, psyn, s1=1.3, s2=2.0, nuM=None, F_ext=1.,
    nuFnu=False, cutoff='R', beta_mid=None, beta_lo_single=-0.5):
  '''
  Synchrotron spectrum with both breaks joined the Granot & Sari (2002) way, plus a
  physical high-frequency cut-off at nuM. GS02 write each break as
      F_nu = F_b,ext [ (nu/nu_b)**(-s*beta1) + (nu/nu_b)**(-s*beta2) ]**(-1/s)
  with beta1, beta2 the F_nu indices either side and F_b,ext the flux where the two
  power laws extrapolate to cross. For two breaks the lower one takes that form and the
  upper one enters as the multiplicative correction [1 + (nu/nu_b2)**(s2*dbeta)]**(-1/s2).

  num, nuc: the two breaks, in ANY order and in the same units as nu -- the ordering
  alone sets the middle slope, beta_mid = -1/2 if nuc < num (fast cooling) or
  -(p-1)/2 if nuc > num (slow), so callers need not say which regime they are in. The
  F_nu indices are 1/3, beta_mid, -p/2 from low to high frequency.

  nuc=None is the SINGLE-BREAK spectrum: one break at num joining beta_lo_single straight to
  -p/2, with smoothing s2 (s1 and beta_mid are unused). Do NOT emulate it by passing nuc=num,
  which keeps the intermediate segment. Two cases use it, differing only in the lower index:
    beta_lo_single = -1/2 (DEFAULT)  very fast cooling -- gamma_c is small enough that the
        whole distribution has cooled, so nu_c AND the 1/3 segment beneath it both sit below
        the observed band, leaving nu_m as the only break.
    beta_lo_single = 1/3             the MARGINAL regime -- nu_c and nu_m are too close for
        any intermediate power law to exist in the data, so the spectrum runs from the 1/3
        asymptote straight to -p/2 through one broad break (see spectral_breaks.
        fit_single_break). Writing it this way rather than as a generic smoothly-broken power
        law is what keeps s in the SAME convention as s1, s2 everywhere else: larger is
        sharper.
  nuM:   the burnoff frequency, where the spectrum cuts off. Not part of GS02; nuM > num,
         nuc always, so it is the top of the spectrum. None -> no cutoff.
  cutoff: the SHAPE of that cut-off, as a factor of u = nu/nuM.
         'R'      -> syn_cutoff_R(u) (DEFAULT), the true single-electron synchrotron
                     emissivity R(x) normalised by its own low-x asymptote. This is the
                     physical rolloff: above nuM the emission comes from the top of the
                     electron distribution, so the spectrum turns over like P'_nu'.
         'exp'    -> exp(-u), the usual crude stand-in. Kept for comparison only: it
                     turns over far too abruptly and costs a factor ~4 in rise-phase
                     residual against the computed spectra (0.133 vs 0.029 dex).
         callable -> any cutoff(u);  None -> no cut-off.
  F_ext: GS02's F_b,ext at the LOWER break.
  nuFnu: return nu*F_nu instead of F_nu.
  beta_mid: override the middle F_nu slope, which is otherwise set by the ordering of the
         two breaks (-1/2 fast, -(p-1)/2 slow). Only useful in the MARGINAL regime: with
         gamma_c = gamma_m the shell-integrated mid segment lands genuinely between the two
         asymptotes (measured -0.67 / -0.56 there), and the fast/slow labels become
         meaningless -- neither break is cleanly nu_c or nu_m. Every other regime pins it
         at an asymptote when left free, so the ordering rule is right there.

  CAREFUL with the smoothing convention: here (as in GS02) a LARGER s is a SHARPER
  break, the opposite of sweep_gammacm._slope_step / paired_syn_bpl, where larger s is
  smoother. s1 applies to the lower break, s2 to the upper.

  Evaluated in log space: nu/nu_b spans >10 decades in these spectra and the naive
  powers overflow (same reason smooth_bpl_apy is written with logaddexp).
  '''
  nu = np.asarray(nu, float)
  beta_hi = -psyn/2.
  if nuc is None:
    # single break at num, beta_lo_single -> -p/2, in the same GS02 two-term form
    ln2 = np.log(nu/num)
    ln_F = np.log(F_ext) - np.logaddexp(-s2*beta_lo_single*ln2, -s2*beta_hi*ln2)/s2
  else:
    b_lo, b_hi = min(num, nuc), max(num, nuc)
    beta_lo = 1/3.
    if beta_mid is None:
      beta_mid = -0.5 if nuc < num else -(psyn-1.)/2.

    ln1 = np.log(nu/b_lo)
    # [y**(-s1 beta_lo) + y**(-s1 beta_mid)]**(-1/s1)
    ln_t1 = -np.logaddexp(-s1*beta_lo*ln1, -s1*beta_mid*ln1)/s1
    ln2 = np.log(nu/b_hi)
    # [1 + (nu/b_hi)**(s2 (beta_mid - beta_hi))]**(-1/s2)
    ln_t2 = -np.logaddexp(0., s2*(beta_mid - beta_hi)*ln2)/s2
    ln_F = np.log(F_ext) + ln_t1 + ln_t2
  if nuFnu:
    ln_F = ln_F + np.log(nu)
  F = np.exp(ln_F)
  if nuM is not None and cutoff is not None:
    if cutoff == 'R':
      F = F*syn_cutoff_R(nu/nuM)
    elif cutoff == 'exp':
      F = F*np.exp(-nu/nuM)
    elif callable(cutoff):
      F = F*cutoff(nu/nuM)
    else:
      raise ValueError(f"cutoff must be 'R', 'exp', None or a callable, got {cutoff!r}")
  return F


# Ravasio et al. (2018) 2SBPL smoothing values. n1 is the LOW-ENERGY break, n2 the peak;
# both are fixed in that paper, n2 = 2.69 to match the GBM catalogue's SBPL curvature
# (Lambda = 0.3) and n1 = 5.38 to the mean of its own free time-resolved fits. The break is
# therefore SHARPER than the peak there (n1 = 2 n2), the opposite ordering to this project's
# GS02 defaults (s1 = 1.3 < s2 = 2.0) -- see two_sbpl.
RAVASIO_N1, RAVASIO_N2 = 5.38, 2.69


def two_sbpl_Ej(E_peak, alpha2, beta, n2=RAVASIO_N2):
  '''
  Ravasio et al. (2018) eq. (4): the CROSSING energy E_j of the mid and high power laws, from
  the peak of the E^2 N_E (= nu F_nu) spectrum. The two are not the same energy -- smoothing
  pushes the peak away from the crossing -- and the 2SBPL is parameterised by the peak
  because that is what a GRB spectrum is quoted by, while the shape needs the crossing.

  alpha2, beta are PHOTON indices (N_E ~ E^alpha), so the nuFnu peak exists only if
  alpha2 > -2 > beta; otherwise there is no turnover and this returns NaN.

  Inverse of two_sbpl_Epeak. At the synchrotron values (alpha2 = -3/2, beta = -p/2 - 1 with
  p = 2.5, n2 = 2.69) it gives E_j = 0.709 E_peak.
  '''
  num, den = -(alpha2 + 2.), (beta + 2.)
  if not (np.isfinite(num) and np.isfinite(den)) or den == 0. or num/den <= 0.:
    return np.nan
  return E_peak*(num/den)**(1./((beta - alpha2)*n2))


def two_sbpl_Epeak(E_j, alpha2, beta, n2=RAVASIO_N2):
  '''E_peak of the nuFnu spectrum from the crossing energy E_j -- the inverse of
  two_sbpl_Ej, i.e. Ravasio et al. (2018) eq. (4) solved the other way.'''
  num, den = -(alpha2 + 2.), (beta + 2.)
  if not (np.isfinite(num) and np.isfinite(den)) or den == 0. or num/den <= 0.:
    return np.nan
  return E_j*(num/den)**(-1./((beta - alpha2)*n2))


def two_sbpl(E, E_break, E_peak, alpha1=-2/3., alpha2=-1.5, beta=-2.5,
    n1=RAVASIO_N1, n2=RAVASIO_N2, A=1., nuFnu=False, E_j=None, Ecut=None, cutoff=None):
  '''
  The double smoothly broken power law of Ravasio et al. (2018), A&A 613, A16, eq. (3):
  three power laws joined by two smooth breaks, introduced there to fit GRB 160625B's prompt
  spectrum, which no single-break model (Band, SBPL) could describe. Written in the PHOTON
  spectrum N_E they use,

      N_E = A E_break**a1 { [ (E/E_br)**(-a1 n1) + (E/E_br)**(-a2 n1) ]**(n2/n1)
                            + (E/E_j)**(-b n2) [ (E_j/E_br)**(-a1 n1)
                                                 + (E_j/E_br)**(-a2 n1) ]**(n2/n1) }**(-1/n2)

  with E_j given by eq. (4) (two_sbpl_Ej) so that the E^2 N_E peak sits at E_peak.

  alpha1, alpha2, beta are PHOTON indices, as in the paper: alpha1 below E_break, alpha2
  between E_break and the peak, beta above it. The F_nu index is alpha + 1 and the nuFnu
  index alpha + 2, so the synchrotron values the paper recovers -- alpha1 = -2/3,
  alpha2 = -3/2 -- are our 1/3 and -1/2 in F_nu, i.e. the SAME asymptotes granot_sari_syn
  carries, and beta = -psyn/2 - 1 is our -p/2. The defaults here are the paper's fast-cooling
  synchrotron values, NOT anything measured on this project's spectra.

  n1 (break) and n2 (peak) are smoothing exponents in the SAME convention as GS02's s1, s2:
  LARGER IS SHARPER. The paper holds them at RAVASIO_N1, RAVASIO_N2.

  WHAT IT IS, ALGEBRAICALLY, AND HOW IT DIFFERS FROM granot_sari_syn. Write the lower
  two-segment SBPL as L(E) = [(E/E_br)**(-a1 n1) + (E/E_br)**(-a2 n1)]**(-1/n1) and the high
  power law anchored on it as P(E) = L(E_j) (E/E_j)**b. Then eq. (3) is exactly

      N_E = A E_br**a1 [ L**(-n2) + P**(-n2) ]**(-1/n2),

  the smooth MINIMUM of the lower SBPL and the high power law -- a NESTED smoothly broken
  power law. granot_sari_syn instead writes the upper break as a multiplicative correction,
  L(E) x [1 + (E/b_hi)**(s2 (a2 - b))]**(-1/s2). The two agree exactly in the limit where L
  has reached its a2 asymptote by E_j, since there L/P = (E/E_j)**(a2-b); they differ only
  through the CURVATURE OF THE LOWER BREAK still present at E_j, i.e. only when the two
  breaks are close. Measured on this project's asymptotes (a1 = -2/3, a2 = -3/2,
  b = -2.25) with the breaks, slopes and smoothings all matched (b_lo = E_break,
  b_hi = E_j, s1 = n1, s2 = n2), the maximum |log10| difference between the two shapes over
  the whole spectrum, at the two smoothing pairs of interest, is

      log10(b_hi/b_lo)     0.5      1.0      1.5      2.0      3.0
      s = (1.3, 2.0)     0.084    0.026    0.008    0.002    0.0002
      s = (5.38, 2.69)   0.005    0.0005   0.0001   0.000    0.000

  i.e. they are the SAME three-segment family, differently parameterised, everywhere the
  breaks are more than ~1.5 decades apart -- below the fit rms of either. The difference
  only becomes comparable to that rms in the marginal regime, and it is larger for a SMOOTH
  lower break (small n1), which is exactly when L still carries curvature at E_j.
  See ravasio_2sbpl.shape_difference for the scan.

  E_break and E_j are both CROSSING energies (the two terms of their bracket are equal
  there), so they map one-to-one onto granot_sari_syn's b_lo and b_hi. E_peak does not:
  it is the nuFnu turnover, which smoothing puts above E_j.

  E_j: pass it directly to bypass eq. (4) -- then E_peak is ignored. Useful when comparing
  against a fit parameterised on the crossing, and required when alpha2 <= -2 (no turnover).
  Ecut, cutoff: a high-frequency roll-off, as in granot_sari_syn -- Ecut is where it sits and
  cutoff its shape, 'R' (the single-electron synchrotron emissivity), 'exp', a callable, or
  None. The paper's own spectra needed an exponential cut-off at ~50 MeV only once LAT data
  were included (their Sect. 3); the DEFAULT here is no cut-off at all, because the natural
  use in this project is on cut-off-flattened spectra.
  nuFnu: return E^2 N_E (= nu F_nu) instead of the photon spectrum N_E.

  Evaluated in log space for the same reason granot_sari_syn is: E/E_break spans >10 decades
  on these spectra and the naive powers overflow.
  '''
  E = np.asarray(E, float)
  if E_j is None:
    E_j = two_sbpl_Ej(E_peak, alpha2, beta, n2)
  if not np.isfinite(E_j) or E_j <= 0.:
    return np.full(E.shape, np.nan)
  u = np.log(E/E_break)                 # ln(E/E_break)
  w = np.log(E_j/E_break)               # ... at the crossing, for the second term's anchor
  # [ (E/E_br)**(-a1 n1) + (E/E_br)**(-a2 n1) ]**(n2/n1), in log
  ln_L = (n2/n1)*np.logaddexp(-alpha1*n1*u, -alpha2*n1*u)
  ln_P = -beta*n2*np.log(E/E_j) + (n2/n1)*np.logaddexp(-alpha1*n1*w, -alpha2*n1*w)
  ln_N = np.log(A) + alpha1*np.log(E_break) - np.logaddexp(ln_L, ln_P)/n2
  if nuFnu:
    ln_N = ln_N + 2.*np.log(E)
  N = np.exp(ln_N)
  if Ecut is not None and cutoff is not None:
    if cutoff == 'R':
      N = N*syn_cutoff_R(E/Ecut)
    elif cutoff == 'exp':
      N = N*np.exp(-E/Ecut)
    elif callable(cutoff):
      N = N*cutoff(E/Ecut)
    else:
      raise ValueError(f"cutoff must be 'R', 'exp', None or a callable, got {cutoff!r}")
  return N


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

def broken_plaw_basic(x, b1=-0.5, b2=-1.25):
  if x<= 1:
    return x**b1
  else:
    return x**b2

broken_plaw = vectorize(broken_plaw_basic)

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
  if x <= 0.:
    return 0.
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
  !!! in code units, and u is proper velocity
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

def derive_polytropic(rho, p):
  adb = derive_adiab(rho, p)
  cons = p / rho**adb
  return cons

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
  adb = derive_adiab_TM(rho, p)
  return adb

def derive_adiab_TM(rho, p):
  '''
  Adiabatic index, following Taub-Matthews EoS
  '''
  T = p/rho
  gma = (1./6.)*(8. - 3.*T + np.sqrt(4. + 9.*T*T))
  return gma

def derive_adiab_Ryu(rho, p):
  '''
  Adiabatic index, following Ryu et al 2006 EoS
  '''
  T = p/rho
  a = 3*T + 1
  return (4*a+1)/(3*a)

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

def derive_Eint_lab(x, dx, rho, vx, p, R0, rhoscale, geometry):
  '''
  Total internal energy of a cell
  '''
  ei = derive_Eint(rho, vx, p, rhoscale)
  V3 = derive_3volume(x, dx, R0, geometry)
  return ei*V3

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

def derive_Eint_comoving_R2(x, rho, p, R0, rhoscale):
  '''
  Internal energy density in comoving frame, rescaled by (R/R0)^2
  '''
  R = x * c_ / R0
  rho = rho*rhoscale*c_**2
  p = p*rhoscale*c_**2
  T = derive_temperature(rho, p)
  gma  = derive_adiab_fromT_TM(T)
  return p*R**2/(gma-1.)

def derive_epint(rho, p):
  '''
  Internal energy in comoving frame in code units
  '''
  gma = derive_adiab(rho, p)
  h   = 1+p*gma/(gma-1.)/rho
  eps = rho*(h-1)/gma
  return eps

def derive_shockStrength(rho, p, rhoscale):
  '''
  Shock strength Gamma_ud - 1 = e'_int / rho_d c^2 
  '''
  ei = derive_Eint_comoving(rho, p, rhoscale)
  ShSt = ei/(rho*rhoscale*c_**2)
  return ShSt

def derive_relatvel_ud(rho, p, rhoscale):
  '''
  Relative velocity between up and donwstream
  '''
  
  lfac_ud = derive_shockStrength(rho, p, rhoscale) + 1
  beta_ud = derive_velocity(lfac_ud)
  return beta_ud

def derive_lfac_ud(vx, vx_u):
  '''
  Relative Lorentz factor between downstream and upstream
  cell is taken downstream, must have vx_u saved
  '''
  lfac_d = derive_Lorentz(vx)
  lfac_u = derive_Lorentz(vx_u)
  return lfac_u * lfac_d * (1 - vx_u * vx)

def derive_lfac_ud_minus1(vx, vx_u):
  '''
  Same as lfac_ud but - 1
  '''
  lfac_d = derive_Lorentz(vx)
  lfac_u = derive_Lorentz(vx_u)
  return lfac_u * lfac_d * (1 - vx_u * vx) - 1.

def derive_B_comoving(rho, p, rhoscale, eps_B=1/3.):
  B2 = derive_B2_comoving(rho, p, rhoscale, eps_B)
  return np.sqrt(B2)

def derive_B2_comoving(rho, p, rhoscale, eps_B):
  e = derive_Eint_comoving(rho, p, rhoscale)
  return 8*pi_*eps_B*e

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

def derive_relatLfac(lfac1, lfac2):
  '''
  Relative Lorentz factor
  '''
  beta1 = derive_velocity(lfac1)
  beta2 = derive_velocity(lfac2)
  lfac12 = lfac1*lfac2*(1-beta1*beta2)
  return lfac12

def derive_Ekin(rho, v, rhoscale):
  '''
  Kinetic energy density in lab frame from density and velocity
  '''
  lfac = derive_Lorentz(v)
  return (lfac-1)*lfac*rho*rhoscale*c_**2

def derive_Ekin_lab(x, dx, rho, vx, p, R0, rhoscale, geometry):
  '''
  Total kinetic energy of a cell
  '''
  ek = derive_Ekin(rho, vx, rhoscale)
  V3 = derive_3volume(x, dx, R0, geometry)
  return ek*V3

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

def derive_Pnum(x, dx, rho, vx, p, rhoscale, R0, eps_B, xi_e, psyn, geometry):
  '''
  Normalization of emitted power by a p-law distribution in slow cooling
  '''
  Pmax_e = derive_Pmax(rho, p, rhoscale, eps_B, xi_e)
  V3p = derive_3vol_comoving(x, dx, vx, R0, geometry)
  return ((psyn-1)/(3*psyn-1))*V3p*Pmax_e

def derive_Pemax(rho, p, rhoscale, eps_B):
  '''
  Max emissivity of a single electron,
    assumes P_nu' ~ (nu'/nu'_syn)^(1/3) for nu'<=nu'_syn, then 0
  '''
  B = derive_B_comoving(rho, p, rhoscale, eps_B)
  fac = (2./3) * sigT_*me_*c_**2/(3*e_)
  return fac*B

def derive_Pmax(rho, p, rhoscale, eps_B, xi_e):
  '''
  Max emissivity of electrons in a cell
  '''
  Pe = derive_Pemax(rho, p, rhoscale, eps_B)
  ne = xi_e*derive_n(rho, rhoscale)
  return ne*Pe

def derive_Lmax(rho, vx, p, dtp, rhoscale, eps_B, xi_e):
  '''
  Maximal luminosity in source frame
  '''
  lfac = derive_Lorentz(vx)
  Pmax = derive_Pmax(rho, p, rhoscale, eps_B, xi_e)
  Lmax = 2*lfac*Pmax*dtp
  return Lmax

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

def _cooled_energy_antideriv(s, p, tt):
  '''
  Antiderivative of s**(p-2)/(s+tt) with G(0) = 0, i.e. the energy integral of the
  cooled shape written in the un-cooling variable s (see derive_xiDN_cooled):
    G(S) = S**(p-1)/((p-1)*tt) * 2F1(1, p-1; p; -S/tt)
  '''
  s = np.asarray(s, dtype=float)
  pos = s > 0.
  sp = np.where(pos, s, 1.)                  # 2F1 is evaluated on every branch
  return np.where(pos, sp**(p-1)/((p-1)*tt) * hyp2f1(1., p-1., p, -sp/tt), 0.)

def derive_xiDN_cooled(gmin, gmax, p, tt, parts=False):
  '''
  derive_xiDN for the COOLED shape N ~ gma**-p (1-gma*tt)**(p-2) (cooling_distribution.
  distrib_plaw_cooled) rather than the pristine power law, with
  tt = radiation_cooling.cooled_tt_eff(gmax, bsyn) = (1-bsyn)/gmax.

  Cooling only relabels electrons, so the change of variable
    s(gma) = 1/gma - tt = 1/gma0      (gma0 = the Lorentz factor it was INJECTED with)
  sends every moment of the cooled shape back to a power-law integral:
    N(a,b) = int_a^b gma**-p     (1-gma*tt)**(p-2) dgma = (s_a**(p-1)-s_b**(p-1))/(p-1)
    E(a,b) = int_a^b gma**(1-p)  (1-gma*tt)**(p-2) dgma = int_{s_b}^{s_a} s**(p-2)/(s+tt) ds
                                                        = G(s_a) - G(s_b)
  so xi keeps derive_xiDN's two-factor structure,
    xi_N = [E(gmin,gmax)/E(1,gmax)] * [N(1,gmax)/N(gmin,gmax)] = F1 * F2,
  evaluated at s(gmax) = bsyn/gmax, s(1) = 1-tt, s(gmin) = 1/gmin - tt. F2 reads off
  directly: the electrons below gma = 1 today are exactly those injected below
  gma0 = 1/(1-tt), so it is the INJECTED power law's number fraction above that
  pulled-back threshold. tt -> 0 reproduces derive_xiDN to machine precision, and
  p = 2 is covered as well (2F1(1,1;2;z) is the log the energy integral degenerates
  to; checked against it to 1e-12).

  WHICH factors to use is a prescription choice, and they are not interchangeable:
    - electrons that COOLED below 1 stop emitting and their energy is already in the
      radiation, so nothing is re-spread: the emission integral is simply truncated
      at gma = 1 and no factor applies. That is what radiation_cooling.get_epnu and
      step_radiated_energy do -- they do NOT call this function;
    - deep-Newtonian INJECTION (derive_gma_m returning gma_m < 1, the Ayache et al.
      2022 case derive_xiDN was written for) does re-spread that energy over the
      surviving electrons. Against a NUMBER normalisation (norm_plaw_distrib on the
      injection bounds) that is F1 alone; F1*F2 is the emitting-number fraction, i.e.
      what multiplies a distribution renormalised to unit number on [1, gmax].
  Applying F1*F2 on top of a K0-normalised integral already truncated at 1 counts the
  renormalisation twice (F2 = 0.35 at gmin=0.5, gmax=1e3, p=2.5).

  parts=True returns (xi_N, F1, F2) instead of xi_N.

  The cooled shape itself barely matters: [gmin, 1] lies far below the cutoff
  (gma*tt <= tt <= 1/gmax there), so (1-gma*tt)**(p-2) ~ 1 exactly where the ratio is
  decided and derive_xiDN is already within 0.16% (median over p in [2.2, 3], gmin in
  [1e-3, 1], gmax in [1.05, 1e6], bsyn in [0, 1]). The residual is the O(gmax**(2-p))
  top-edge curl in F1's denominator, reaching ~10% only for gmax <~ 3.
  '''
  gmin, gmax, tt = (np.asarray(v, dtype=float) for v in (gmin, gmax, tt))
  # np.where evaluates every branch: hold the masked-out ones on finite placeholders
  edge = (gmax <= 1.) | (gmin >= 1.)
  gmn  = np.where(edge, .5, gmin)
  gmx  = np.where(edge, 2., gmax)
  cool = (tt > 0.) & ~edge
  t    = np.where(cool, tt, .5/gmx)
  sm, s1, sM = 1./gmn - t, 1. - t, 1./gmx - t

  F2 = (s1**(p-1) - sM**(p-1))/(sm**(p-1) - sM**(p-1))
  F1 = (_cooled_energy_antideriv(sm, p, t) - _cooled_energy_antideriv(sM, p, t)) \
       / (_cooled_energy_antideriv(s1, p, t) - _cooled_energy_antideriv(sM, p, t))
  # bsyn = 1 (purely adiabatic step, tt = 0): the shape is the pristine power law
  F2 = np.where(cool, F2, (gmx**(1-p) - 1.)/(gmx**(1-p) - gmn**(1-p)))
  F1 = np.where(cool, F1, (gmx**(2-p) - gmn**(2-p))/(gmx**(2-p) - 1.))
  # gma_max cooled below 1: nothing emits. gma_min still above 1: nothing is missing.
  F1 = np.where(edge, 1., F1)
  F2 = np.where(gmax <= 1., 0., np.where(gmin >= 1., 1., F2))
  return (F1*F2, F1, F2) if parts else F1*F2

def derive_xiE(gmin, gmax, p):
  '''
  Fraction of eps_e*e'_int actually carried by the truncated power law between
  gmin and gmax: derive_gma_m sets gma_m from the gma_M -> inf limit
  (Gp = (p-2)/(p-1)), so a distribution cut at gma_M holds only
    (1 - x**(2-p))/(1 - x**(1-p)),  x = gmax/gmin
  of that energy. -> 1 as x -> inf, and drops below 1 for narrow distributions
  (0.94 at x=281, reached deep in fast cooling by the alpha rescale).
  '''
  if p == 2.:
    # degenerate: Gp = 0 so derive_gma_m (and hence the reference energy) is undefined
    return np.nan
  x = np.asarray(gmax, dtype=float)/np.asarray(gmin, dtype=float)
  xs = np.where(x<=1., 2., x)          # placeholder, masked out below
  return np.where(x<=1., 0., (1.-xs**(2-p))/(1.-xs**(1-p)))

def derive_Ne(x, dx, vx, rho, R0, rhoscale, geometry):
  '''
  Number of electrons
  '''
  n = derive_n(rho, rhoscale)
  V3p = derive_3vol_comoving(x, dx, vx, R0, geometry)
  return n*V3p

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

def derive_tc1(rho, p, rhoscale, eps_B):
  '''
  Synchrotron cooling time of an electron at non-relativistic energy, in comoving frame
  '''
  tc1 = 1/derive_syn_cooling(rho, p, rhoscale, eps_B)
  return tc1

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

def derive_Ppmax_noprefac(rho, p, rhoscale, eps_B, xi_e):
  B = derive_B_comoving(rho, p, rhoscale, eps_B)
  ne = xi_e*derive_n(rho, rhoscale)
  fac = sigT_ * me_ * c_**2 / e_
  return fac * ne * B

def derive_Epnu_FC(x, dx, dt, rho, p, R0, rhoscale, eps_B, xi_e, geometry):
  V4 = derive_4volume(x, dx, dt, R0, geometry)
  Pmax = ((psyn-1)/(2*psyn))*derive_Ppmax_noprefac(rho, p, rhoscale, eps_B, xi_e)
  return V4*Pmax

def derive_Epnu_SC(x, dx, dt, rho, p, R0, rhoscale, psyn, eps_B, xi_e, geometry):
  V4 = derive_4volume(x, dx, dt, R0, geometry)
  Pmax = (4*(psyn-1)/(3*(3*psyn-1)))*derive_Ppmax_noprefac(rho, p, rhoscale, eps_B, xi_e)
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
  V3p = derive_3vol_comoving(x, dx, vx, R0, geometry)
  ep_nu_m = derive_ep_nu_m_VFC(rho, p, rhoscale, psyn, eps_B, eps_e, xi_e)
  Epnu = V3p * ep_nu_m
  return Epnu

def derive_ep_nu_m_VFC(rho, p, rhoscale, psyn, eps_B, eps_e, xi_e):
  '''
  Energy per unit volume emitted by accelerated electrons in very fast cooling regime
  '''
  Wp = 2*((psyn-1)/(psyn-2))
  nup_m = derive_nup_m(rho, p, rhoscale, psyn, eps_B, eps_e, xi_e)
  epe = eps_e * derive_Eint_comoving(rho, p, rhoscale)
  ep_nu_m =  epe / (Wp * nup_m)
  return ep_nu_m

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

def derive_Lbol_comov_new(x, rho, vx, p, vx_u, R0, rhoscale, eps_e, geometry):
  ''' Bolometric luminosity in comoving frame'''
  # eint = derive_Eint_comoving(rho, p, rhoscale)
  # lfac_ud = derive_lfac_ud(vx, vx_u)
  # beta_ud = derive_velocity(lfac_ud)
  # r = R0 if geometry == 'cartesian' else x*c_
  # S = (4./3.)*pi_*r**2
  # return eps_e*eint*beta_ud*S*c_
  lfac_ud = derive_lfac_ud(vx, vx_u)
  beta_ud = derive_velocity(lfac_ud)
  r = R0 if geometry == 'cartesian' else x*c_
  S = 4.*pi_*r**2
  dMdt = (rho * rhoscale) * (beta_ud/3.) * c_ * S
  Lbol = eps_e * (lfac_ud - 1) * dMdt * c_**2
  return Lbol


def derive_Lp_nupm_new(x, rho, vx, p, vx_u, R0, rhoscale, psyn, eps_B, eps_e, xi_e, geometry):
  Lbol = derive_Lbol_comov_new(x, rho, vx, p, vx_u, R0, rhoscale, eps_e, geometry)
  nup_m = derive_nup_m_new(rho, vx, p, vx_u, rhoscale, psyn, eps_B, eps_e, xi_e)
  Wp = 2*(psyn-1)/(psyn-2)
  return Lbol/(Wp*nup_m)

def derive_Lbol_comov(x, rho, p, R0, rhoscale, eps_e, geometry):
  ''' Bolometric luminosity in comoving frame'''
  eint = derive_Eint_comoving(rho, p, rhoscale)
  lfac_ud = eint/(rho*rhoscale*c_**2) + 1
  beta_ud = derive_velocity(lfac_ud)
  r = R0 if geometry == 'cartesian' else x*c_
  S = (4./3.)*pi_*r**2
  return eps_e*eint*beta_ud*S*c_

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
  returns Ton, Tth, Tej
  Ton : onset time
  Tth : angular time (comes from Doppler beaming)
  Tej : effective ejection time
  !!! r in units c !!! (function to be used by data_IO:get_variable)
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
  Ton, Tth, Tej = derive_obsTimes(t, r, beta, t0, z)
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

def derive_cyclotron(rho, vx, p, rhoscale, eps_B, z):
  '''
  Cyclotron frequency
  '''
  nup_B = derive_cyclotron_comoving(rho, p, rhoscale, eps_B)
  D = derive_DopplerRed_los(vx, z)
  nu_B = nup_B*D
  return nu_B


def derive_DopplerRed_los(vx, z):
  '''
  Derive Doppler + redshift factor along the line of sight
  '''
  lfac = derive_Lorentz(vx)
  D = 1/(lfac*(1-vx))
  return D/(1+z)

def derive_gma_c(rho, p, R0, lfac0, rhoscale, eps_B):
  '''
  (generalized) cooling Lorentz factor gma_c
  '''
  tc = 1/derive_syn_cooling(rho, p, rhoscale, eps_B)
  tdyn = R0/(lfac0*c_)
  return tc/tdyn

def derive_nup_c(rho, p, R0, lfac0, rhoscale, eps_B):
  '''
  Comoving cooling freq
  '''

  gma_c = derive_gma_c(rho, p, R0, lfac0, rhoscale, eps_B)
  nup_c = derive_nup_from_gma(rho, p, gma_c, rhoscale, eps_B)
  return nup_c

def derive_nu_c(rho, vx, p, R0, lfac0, rhoscale, eps_B, z):
  '''
  Cooling freq
  '''
  gma_c = derive_gma_c(rho, p, R0, lfac0, rhoscale, eps_B)
  nu_c = derive_nu_from_gma(rho, vx, p, gma_c, rhoscale, eps_B, z)
  return nu_c


def derive_nup_from_gma(rho, p, gma, rhoscale, eps_B):
  '''
  Derive comoving frequency from e- Lorentz factor
  '''
  nup_B = derive_cyclotron_comoving(rho, p, rhoscale, eps_B)
  nup = nup_B*gma**2
  return nup

def derive_nu_from_gma(rho, vx, p, gma, rhoscale, eps_B, z):
  '''
  Derive frequency from e- Lorentz factor
  '''
  D = derive_DopplerRed_los(vx, z)
  nup = derive_nup_from_gma(rho, p, gma, rhoscale, eps_B)
  return D*nup

def derive_nu_m_new(rho, vx, p, vx_u, rhoscale, psyn, eps_B, eps_e, xi_e, z):
  '''
  Peak frequency of the electron distribution, in observer frame
  '''
  D = derive_DopplerRed_los(vx, z)
  nup_m = derive_nup_m_new(rho, vx, p, vx_u, rhoscale, psyn, eps_B, eps_e, xi_e)
  nu_m = nup_m*D
  return nu_m

def derive_nu_m(rho, vx, p, rhoscale, psyn, eps_B, eps_e, xi_e, z):
  '''
  Peak frequency of the electron distribution, in observer frame
  '''
  D = derive_DopplerRed_los(vx, z)
  nup_m = derive_nup_m(rho, p, rhoscale, psyn, eps_B, eps_e, xi_e)
  nu_m = nup_m*D
  return nu_m

def derive_nup_m_new(rho, vx, p, vx_u, rhoscale, psyn, eps_B, eps_e, xi_e):
  '''
  Peak frequency of the electron distribution, in comoving frame
  '''
  nup_B = derive_cyclotron_comoving(rho, p, rhoscale, eps_B)
  gma_m = derive_gma_m_new(vx, vx_u, psyn, eps_e, xi_e)
  nup_m = nup_B*gma_m**2
  return nup_m

def derive_nup_m_sc(x, rho, vx, p, vx_u, R0, rhoscale, psyn, eps_B, eps_e, xi_e):
  '''
  nu'_m x (R/R0)
  '''
  sc = x * c_ / R0 
  nup_m = derive_nup_m_new(rho, vx, p, vx_u, rhoscale, psyn, eps_B, eps_e, xi_e)
  nup_m_sc = nup_m * sc
  return nup_m_sc

def derive_nup_m(rho, p, rhoscale, psyn, eps_B, eps_e, xi_e):
  '''
  Peak frequency of the electron distribution, in comoving frame
  '''
  nup_B = derive_cyclotron_comoving(rho, p, rhoscale, eps_B)
  gma_m = derive_gma_m(rho, p, rhoscale, psyn, eps_e, xi_e)
  nup_m = nup_B*gma_m**2
  return nup_m

def derive_nu_M(rho, vx, p, rhoscale, eps_B, z):
  '''
  Peak frequency of the electron distribution, in observer frame
  '''
  D = derive_DopplerRed_los(vx, z)
  nup_M = derive_nup_M(rho, p, rhoscale, eps_B)
  nu_M = nup_M*D
  return nu_M


def derive_nup_M(rho, p, rhoscale, eps_B):
  '''
  Peak frequency of the electron distribution, in comoving frame
  '''
  nup_B = derive_cyclotron_comoving(rho, p, rhoscale, eps_B)
  gma_M = derive_gma_M(rho, p, rhoscale, eps_B)
  nup_M = nup_B*gma_M**2
  return nup_M

# def derive_nup_c(t, rho, vx, p, rhoscale, eps_B):
#   '''
#   Comoving cooling frequency
#   '''
#   nup_B = derive_cyclotron_comoving(rho, p, rhoscale, eps_B)
#   gma_c = derive_gma_c(t, rho, vx, p, rhoscale, eps_B)
#   nup_c = nup_B*gma_c**2
#   return nup_c

# def derive_gma_c(t, rho, vx, p, t0, rhoscale, eps_B):
#   '''
#   Typical Lorentz factor of cooled electrons
#   '''
#   lfac = derive_Lorentz(vx)
#   tdyn = (t + t0)/lfac
#   e = derive_Eint_comoving(rho, p, rhoscale)
#   return 3*me_*c_/(4*sigT_*eps_B*e*tdyn)

def derive_syn_cooling(rho, p, rhoscale, eps_B):
  '''
  Synchrotron cooling
  '''
  ei = derive_Eint_comoving(rho, p, rhoscale)
  B2 = 8*pi_*eps_B*ei
  return alpha_*B2

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

def derive_gma_m_new(vx, vx_u, psyn, eps_e, xi_e):
  '''
  Minimum Lorentz factor of accelerated electron distribution
  '''
  lfac_ud = derive_lfac_ud(vx, vx_u)
  Gp = (psyn-2.)/(psyn-1.)
  return (mp_/me_) * Gp * (eps_e/xi_e) * (lfac_ud - 1.)


def derive_gma_M(rho, p, rhoscale, eps_B):
  '''
  Return max Lorentz factor of an accelerated electron distribution
  '''

  B = derive_B_comoving(rho, p, rhoscale, eps_B)
  gmaM2 = 6*pi_*e_/(sigT_*B)
  gmaM = np.sqrt(gmaM2)
  return gmaM

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


