# -*- coding: utf-8 -*-
# @Author: acharlet

'''
Analytic fitting function for the synchrotron emissivity of a cooled power-law
electron distribution, from Ferguson & Margalit 2026 (arXiv:2607.13130), section 3.1.

Jpl_FM26 fits the dimensionless integral (their Eq. 18)
    J_pl(p; x1, xinf) = int_{xinf}^{x1} dx F~(x) x^{(p-3)/2} (1 - sqrt(xinf/x))^{p-2}
i.e. the emissivity of dn/dgma ~ gma^-p (1-gma/gma_inf)^{p-2}
(cooling_distribution.distrib_plaw_cooled with gma_inf = gmax), with F~ the
pitch-angle-averaged synchrotron kernel. F~ is identical to the CS86/FDB08 R(x)
tabulated in radiation_cooling.func_R: both -> 1.80842 x^{1/3} at low x and
(pi/2) e^{-x} at high x.

Mapping to the code's variables (tnu = nu'/nu'_B, nu0 = (3/2) nu_B):
    x1 = (2/3) tnu / gmin^2,  xinf = (2/3) tnu / gmax^2,  eta = gmax/gmin
    Pnu_instant(tnu) = (1/2) ((2/3) tnu)^{(1-p)/2} J_pl   (tnu >= 1)
  in R(x)'s own units; radiation_cooling.get_epnu applies the single 1/norm_R_
  (norm_R_ = int R dx) once, in its Pmax prefactor.

Validity: 2 < p <~ 5, synchrotron-only cooling with gma_2 = gma_inf (injected
gmax0 >> gmin0) -- the assumptions of gamma_synCooled/distrib_plaw_cooled.
NOT valid for ODE-evolved distributions with adiabatic cooling unless the paper's
F, G functions are supplied.

NOTE on accuracy (validated against dense quadrature of the exact integrand):
- the printed sigmoid constant alpha4 (their Eq. 39) collapses to ~1.6 for
  eta >> 1, which makes >= 53% error at x1 ~ 1 unavoidable -- presumably a typo.
  alpha1, alpha4 were refitted here at eta >> 1 and blended back to the printed
  values (which do reproduce the paper's quoted errors) at eta <~ 2.
- with the refit: eta >= 4 -> max error ~5-10% for p <= 2.5 (~15-30% for p >= 3),
  median ~1%; eta ~ 1.1-3 -> up to ~30-80% near the spectral cutoff (the paper's
  own worst regime, 58% quoted at eta = 1.14). radiation_cooling.get_epnu
  therefore only routes eta >= FIT_ETA_MIN through the fit, and its width_tol
  single-gamma shortcut covers eta <= 1.1.
'''

import numpy as np
from numba import njit
from scipy.special import gamma as Gamma, hyp2f1


###### pitch-angle-averaged synchrotron kernel F~(x)
# default standalone kernel, Eq. B13 (Aharonian, Kelner & Prosekin 2010, eq. D7);
# radiation_cooling passes its tabulated func_R instead (same function).
def Ftilde_B13(x):
  x = np.asarray(x, dtype=float)
  with np.errstate(under='ignore'):
    x23 = x**(2./3.)
    out = 1.808 * x**(1./3.) / np.sqrt(1. + 3.4*x23) \
        * (1. + 2.21*x23 + 0.347*x23**2) / (1. + 1.353*x23 + 0.217*x23**2) \
        * np.exp(-x)
  return out


###### Table 1: coefficients a_i(p) = sum_j aleph_j p^j (ascending)
_TAB1 = {
  'a1': [ 0.502, -0.287,  0.057, -0.004,  0.    ],
  'a2': [ 1.115,  0.441, -0.006,  0.,     0.    ],
  'a3': [-3.27,   3.17,  -0.718,  0.072, -0.0022],
  'a4': [-0.221,  0.721, -0.352,  0.065, -0.004 ],
}

def _poly(coeffs, p):
  return sum(c * p**j for j, c in enumerate(coeffs))


def A1_FM26(p):
  '''Intermediate-frequency plateau A1(p), Eq. 22'''
  return np.sqrt(np.pi) * 2**((p+3)/2.) / ((p+1.)*(p+3.)) \
      * Gamma(p/4.+5./4.) * Gamma(p/4.+19./12.) * Gamma(p/4.-1./12.) \
      / Gamma(p/4.+3./4.)


def Omega_FM26(p, eta, xinf, func_F=Ftilde_B13):
  '''
  Low-frequency (x1 << 1) exact limit Omega_p(eta, xinf), Eq. 19,
  with the eta-1 << 1 expansion (Eq. 21) where the hypergeometric
  form fails numerically. Valid at all frequencies for eta -> 1.
  '''
  if eta - 1. < 1e-3:
    x1 = xinf * eta**2
    return 2./(p-1.) * x1**((p-1.)/2.) * func_F(x1) * max(eta-1., 0.)**(p-1.)
  brace = eta**(p-1./3.) * hyp2f1(2.-p, 1./3.-p, 4./3.-p, 1./eta) \
      - Gamma(p-1.)*Gamma(4./3.-p)/Gamma(-2./3.)
  return 6./(3.*p-1.) * xinf**((p-1.)/2.) * func_F(xinf) * brace


def Psi_FM26(p, xinf):
  '''
  Fit fusing the intermediate plateau A1(p) and the high-frequency
  steepest-descent tail psi_p((p-3)/2, xinf), Eqs. 29-32.
  The psi*delta2 product is evaluated in log space: psi ~ xinf^{(1-p)/2}
  diverges as xinf -> 0 while delta2 -> 0, and the naive product can
  overflow to inf*0 = nan for tiny xinf.
  '''
  a1 = _poly(_TAB1['a1'], p)
  a2 = _poly(_TAB1['a2'], p)
  a3 = _poly(_TAB1['a3'], p)
  a4 = _poly(_TAB1['a4'], p)
  with np.errstate(under='ignore', divide='ignore', invalid='ignore'):
    delta1 = np.exp(-a1*xinf**2 - a2*xinf**(2./3.))
    # log[psi_p((p-3)/2, xinf) * delta2] ; q - p + 2 = (1-p)/2 for q=(p-3)/2
    log_psi_d2 = np.log(np.pi*Gamma(p-1.)/2**(p-1.)) \
        + (1.-p)/2.*np.log(xinf) - xinf \
        + a3*np.log(-np.expm1(-a4*xinf))
    tail = np.where(xinf > 0., np.exp(log_psi_d2), 0.)
  return A1_FM26(p)*delta1 + tail


def Jpl_FM26(p, x1, xinf, eta, func_F=Ftilde_B13):
  '''
  Full fitting function J_pl(p; x1, xinf), Eq. 33: low-frequency limit Omega_p
  and intermediate/high-frequency fit Psi_p blended by the sigmoids S1, S2
  (Eqs. 34-39). p, eta scalars; x1, xinf arrays (x1 = eta^2 * xinf).
  '''
  # Gamma(4/3-p) and hyp2f1(c=4/3-p) have poles at p = 7/3, 10/3, 13/3
  for pc in (7./3., 10./3., 13./3.):
    if abs(p - pc) < 3e-6:
      p = pc + 3e-6
      break
  x1 = np.asarray(x1, dtype=float)
  xinf = np.asarray(xinf, dtype=float)

  Om = Omega_FM26(p, eta, xinf, func_F)
  if eta - 1. < 1e-3:
    # Omega (Eq. 21) is the correct limit at all frequencies; S1 -> 1, S2 -> 0
    return Om

  # Sigmoid constants: the printed Eq. 39 collapses to al4 ~ 1.6 for eta >> 1,
  # which provably cannot reach the paper's quoted ~9% accuracy there (the
  # blend floor at x1 ~ 1 is >= 53%): presumably a typo. We therefore use the
  # printed alpha_i in the eta ~ 1 regime (where they reproduce the paper's
  # quoted errors, e.g. 58% at eta = 1.14) and constants refitted against exact
  # numerical Jpl for eta >> 1, blended by the paper's own switch
  # w = exp(-0.01 (eta^2-1.5)^2). Refit accuracy at eta = inf: max error
  # ~5-14% for 2 < p <= 3.5 (26% at p = 5), median ~1-3%.
  al1_inf = 0.0298 + 1.0078*p - 0.2656*p**2 + 0.02104*p**3
  al4_inf = 0.4886 + 1.3920*p - 0.1968*p**2 + 0.10323*p**3
  al1_prn = -0.03*p**3 + 0.45*p**2 - 2.29*p + 4.8
  al4_one = 0.77 + 0.538*p    # printed Eq. 39 in its w -> 1 limit
  al2 = 0.622 + 0.347*p**(2./3.) - 0.017*p**(4./3.)
  with np.errstate(under='ignore'):
    w = np.exp(-0.01*(eta**2 - 1.5)**2)
    al1 = al1_inf + (al1_prn - al1_inf)*w
    al4 = al4_inf + (al4_one - al4_inf)*w
    al3 = 1. + (0.1*p - 0.71)*np.exp(-(eta**2 - 1.1)**2)
    S1 = np.exp(-al1 * x1**al2 * np.exp(-al3/(eta**2 - 1.)))
    S2 = (1. - S1)**al4
  return Om*S1 + Psi_FM26(p, xinf)*S2


###### fused numba fast path
# All (p, eta)-dependent scalars are computed once per step in pnu_fm26_scalars,
# then _pnu_fm26_kernel does a single pass over tnu with the tabulated
# log F~ grid inlined (same table/asymptotes as radiation_cooling.func_R).
# This removes the ~25 numpy ufunc dispatches (a ~70 us N-independent floor)
# of the vectorized Jpl_FM26 path.

_PCACHE = {}

def _p_consts(p):
  '''p-only constants (special functions, Table 1 / sigmoid polynomials),
  cached: p is fixed for a whole run while eta changes per step.'''
  c = _PCACHE.get(p)
  if c is None:
    pn = p
    for pc in (7./3., 10./3., 13./3.):
      if abs(p - pc) < 3e-6:
        pn = pc + 3e-6
        break
    c = dict(
      p=pn, pm1h=(pn-1.)/2.,
      gterm=Gamma(pn-1.)*Gamma(4./3.-pn)/Gamma(-2./3.),
      A1=A1_FM26(pn),
      a1=_poly(_TAB1['a1'], pn), a2=_poly(_TAB1['a2'], pn),
      a3=_poly(_TAB1['a3'], pn), a4=_poly(_TAB1['a4'], pn),
      log_psi_pref=np.log(np.pi*Gamma(pn-1.)/2**(pn-1.)),
      al1_inf=0.0298 + 1.0078*pn - 0.2656*pn**2 + 0.02104*pn**3,
      al4_inf=0.4886 + 1.3920*pn - 0.1968*pn**2 + 0.10323*pn**3,
      al1_prn=-0.03*pn**3 + 0.45*pn**2 - 2.29*pn + 4.8,
      al4_one=0.77 + 0.538*pn,
      al2=0.622 + 0.347*pn**(2./3.) - 0.017*pn**(4./3.),
    )
    _PCACHE[p] = c
  return c

def pnu_fm26_scalars(p, eta):
  '''
  Per-step scalar setup for _pnu_fm26_kernel: requires eta - 1 >= 1e-3
  (the near-delta branch stays on the vectorized Jpl_FM26 / Eq. 21 path).
  Returns (pm1h, om_pref, A1, a1, a2, a3, a4, log_psi_pref, c1, al2, al4).
  '''
  c = _p_consts(p)
  p = c['p']
  # low-frequency Omega prefactor (Eq. 19 brace, hyp2f1 once per step)
  brace = eta**(p-1./3.) * hyp2f1(2.-p, 1./3.-p, 4./3.-p, 1./eta) - c['gterm']
  om_pref = 6./(3.*p-1.) * brace
  # sigmoid constants (refitted large-eta values blended to printed, cf. Jpl_FM26)
  with np.errstate(under='ignore'):
    w = np.exp(-0.01*(eta**2 - 1.5)**2)
    al3 = 1. + (0.1*p - 0.71)*np.exp(-(eta**2 - 1.1)**2)
    c1 = (c['al1_inf'] + (c['al1_prn'] - c['al1_inf'])*w) \
        * np.exp(-al3/(eta**2 - 1.))
  al4 = c['al4_inf'] + (c['al4_one'] - c['al4_inf'])*w
  return (c['pm1h'], om_pref, c['A1'], c['a1'], c['a2'], c['a3'], c['a4'],
          c['log_psi_pref'], c1, c['al2'], al4)


@njit(cache=True, fastmath=True)
def _pnu_fm26_kernel(tnu, inv_g2sq, cOm, cPl, lp2, a1, a2, a3, a4,
                     pm1h, c1p, al2, al4,
                     logR, logxmin, inv_dlog, xmin, xmax):
  '''
  Pnu(tnu) = pfac * nut^{-pm1h} Jpl(x1, xinf) on a flat tnu array, with
  F~(x) from the uniform-log grid logR (asymptotes outside [xmin, xmax]) and
  the tnu < 1 cutoff of syn_emiss_exact. Everything is expressed in
  xf = (2/3) tnu inv_g2sq and its log, with the nut^{-pm1h} prefactor folded
  into the per-step constants (the Omega term's powers cancel exactly:
  xf^pm1h nut^{-pm1h} = inv_g2sq^pm1h):
    cOm = pfac * om_pref * inv_g2sq^pm1h        (Omega + prefactor)
    cPl = pfac * A1 * inv_g2sq^pm1h             (Psi plateau + prefactor)
    lp2 = log_psi_pref + log(pfac) + pm1h*log(inv_g2sq)   (Psi tail, log space)
    c1p = c1 * eta^{2 al2}                      (S1 argument, x1 = xf eta^2)
  with pfac = 0.5 (R(x)'s own units: radiation_cooling.get_epnu supplies the single
  1/norm_R_ downstream, so it never enters this per-element loop).
  '''
  ng = logR.size
  out = np.zeros(tnu.size)
  for i in range(tnu.size):
    ti = tnu[i]
    if ti < 1.:
      continue
    xf = (2./3.)*ti*inv_g2sq
    lx = np.log(xf)
    # F~(xf): tabulated log-log blend with analytic asymptotes
    if xf < xmin:
      F = 1.80842 * np.exp(lx/3.)
    elif xf > xmax:
      F = 0.5*np.pi*(1. - 99./(162.*xf))*np.exp(-xf)
    else:
      u = (lx - logxmin) * inv_dlog
      i0 = int(u)
      if i0 > ng - 2:
        i0 = ng - 2
      f = u - i0
      F = np.exp((1.-f)*logR[i0] + f*logR[i0+1])
    # sigmoids
    S1 = np.exp(-c1p*np.exp(al2*lx))
    S2 = (1. - S1)**al4
    res = cOm*F*S1
    if S2 > 0.:
      # Psi_p (x prefactor): plateau + steepest-descent tail in log space
      Psi = cPl*np.exp(-a1*xf*xf - a2*np.exp((2./3.)*lx) - pm1h*lx)
      lt = lp2 - 2.*pm1h*lx - xf + a3*np.log(-np.expm1(-a4*xf))
      if lt > -700.:
        Psi += np.exp(lt)
      res += Psi*S2
    out[i] = res
  return out
