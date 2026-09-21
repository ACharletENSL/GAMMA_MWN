# -*- coding: utf-8 -*-
# @Author: acharlet

'''
The time-integrated electron distribution WITH adiabatic cooling, over 10^3 t_dyn.

APPENDIX ILLUSTRATION, on the same footing as cooling_shape_figure: the comoving cooling
time t'_c is held CONSTANT, so the only thing added to the synchrotron-only picture is
the adiabatic drag. Not a measurement of this run, not a main result.

WHY THAT IS THE RIGHT FOOTING, and where the field went. cooling_shape_figure and
cooling_integrated_figure never mention B' at all: they are parameterised entirely in

    tt = int dt'/t_{c,1},

and that integral ABSORBS any B'(t') whatsoever -- dgma/dtt = -gma^2 is exact for any
field history. Those two figures are therefore agnostic, not "constant B'". This one
cannot be: the adiabatic drag A = (rho/rho_0)^(1/3) is a function of RADIUS, so it needs
the tt <-> t' <-> R map that the synchrotron figures were free to ignore, and that map is
exactly where B' re-enters. Something has to be assumed. Holding t'_c fixed is the
minimal choice and the one that isolates the adiabatic term: let B' decay as well and two
things move at once.

  CORRECTION, recorded because the earlier draft of this module got it wrong. Its
  docstring announced B' ~ R^-1 while the code did, and always did, the constant-t'_c
  case -- R/R_0 = 1 + tau was written with tau = tt/tt_dyn, which is only the radius when
  tt is proportional to t'. The two are NOT interchangeable, and not by a small amount.
  With B' ~ R^-q the map is dtt = tt_dyn (1+sigma)^-2q dsigma (sigma = t'/t'_dyn), so

      q = 0:  tt = tt_dyn*sigma                 unbounded
      q = 1:  tt = tt_dyn*sigma/(1+sigma)  ->  tt_dyn        SATURATES

  i.e. with B' ~ R^-1 an electron can only ever spend a finite normalised time cooling,
  and the synchrotron burn S saturates at tt_dyn/(2q-1-alpha) = 0.600 tt_dyn instead of
  reaching 27.0 tt_dyn as it does at q = 0. Past ~one t_dyn all further cooling is then
  adiabatic -- and that does NOT leave the tail index alone. Measured, same a_rho, same
  10^3 t_dyn, only q moved:

      q = 0    e = +0.333    tail index  -1.95     (S unbounded)
      q = 0.5  e = -0.667    tail index  -1.10     (S saturates)
      q = 1    e = -1.667    tail index  +0.42     (S saturates)

  so the index is a strong function of the FIELD history, not just of the density one.
  The q=1 value is positive because once S is frozen every trajectory collapses onto
  gma = A/S_inf: the population slides down as one delta function, and with
  dtt = tt_dyn (1+sigma)^-2 dsigma the normalised time it spends at low gma goes to
  nothing -- asymptotically N ~ gma^(-1-1/alpha) = gma^(+1/2) for coasting, which is the
  +0.42 above still short of its limit at sigma = 1e3. (Note tt IS the right weight for
  a fluence: dt' * P ~ dt' * B'^2 ~ dtt, so the B' that cancels out of the trajectory
  cancels out of the emission too.) NB the gma^-1 guessed in an earlier turn is wrong at
  BOTH ends -- -1.95 at q=0, +0.42 at q=1; it is not a limit this system takes.
  q_B is a parameter (default 0) so this is computed rather than argued about; main()
  reports the scan.

THE EQUATION IS LINEAR IN u = 1/gma. With the adiabatic term the cooling equation is

    dgma/dtt = (dlnA/dtt) gma - gma^2,     A = (rho/rho_0)^(1/3),

which in u = 1/gma is LINEAR -- du/dtt + (dlnA/dtt) u = 1 -- the same structure
working_cooling.evolve_gma_bounds_edges telescopes over. With A as integrating factor:

    u(tt) = [u_0 + S]/A,    S = int_0^tt A dtt',    => gma = A gma_0/(1 + gma_0 S).

S is the EFFECTIVE synchrotron burn: expansion drags gma down through A AND throttles the
losses that follow, and S is what is left of them. A = 1, S = tt recovers gamma_synCooled.
Number conservation (dgma_0/dgma = A/(A - S*gma)^2) then gives

    N(gma,tt) = K0 A gma^-p (A - S*gma)^(p-2)   on [gma_m(tt), gma_M(tt)],

the synchrotron form with 1 -> A and tt -> S. Both edges follow the same trajectory.

THE HYDRODYNAMICS: simple coasting. R/R_0 = 1 + sigma with sigma = t'/t'_dyn (t_dyn = the
comoving radius-doubling time R/(Gamma c), cooling_distribution.get_tdbl_cell), and a
shell of fixed comoving width expanding spherically has rho' ~ R^-2, i.e.

    a_rho = dln rho/dln R = -2,    alpha = a_rho/3 = -2/3.

With B' ~ R^-q and e = alpha - 2q + 1,  S(sigma) = tt_dyn [(1+sigma)^e - 1]/e, so
(1 + e*S/tt_dyn) = (1+sigma)^e and A can be written as a function of S alone,

    A(S) = (1 + e*S/tt_dyn)^(alpha/e),

which is what the code uses -- it is valid for BOTH signs of e, where a (1+tau)^b
parameterisation is not. a_rho is free too; the run's own measured value is -1.205
(prerar_cell_evolution alpha_D = -0.795 through dlnrho/dlnR = -2 - alpha_D, the shell
being spread rather than of fixed width). At q = 0, e > 0 iff a_rho > -3; at e <= 0 the
burn saturates and synchrotron cooling freezes out at a finite total.

THE TIME INTEGRAL. Since dS = A dtt, the integral collapses to one smooth quadrature in S:

    N(gma;tt_end) = K0 gma^-p INT_{S_a}^{S_b} f(S)^(p-2) dS,   f(S) = A(S) - S*gma,

with f strictly DECREASING (both terms fall), so the support -- gma_0 = gma/f in
[gma_m0, gma_M0] -- is one interval bracketed by two monotone root-finds: f(S_a) =
gma/gma_m0 (or S_a = 0) and f(S_b) = gma/gma_M0 (or S_b = S_end). The window is SOLVED
for, never scanned, which is what keeps the very narrow ones at high gma exact -- handed
the full range instead, an adaptive rule misses them and silently returns zero.

WHAT ADIABATIC COOLING CHANGES AT CONSTANT t'_c -- AND THE gma^-1 THAT DOES NOT HAPPEN
HERE. The synchrotron-only gma^-2 segment is a dwell time, dtt = dgma/gma^2. With
expansion the loss rate is gma^2 + a*gma, and the tempting reading -- that the tail
flattens toward gma^-1 wherever a > gma -- is WRONG at q = 0. It treats a as fixed while
gma falls; they fall TOGETHER. At q = 0, e = 1 + alpha, so alpha - e = -1 identically and
the bottom edge gma_m ~ A/S ~ sigma^(alpha-e) = 1/sigma decays at exactly the rate
a ~ 1/sigma does. Hence a/gma -> |alpha|/e, CONSTANT (2.0 for coasting), the loss rate is
gma^2 (1 + |alpha|/e) and the dwell time is e*dgma/gma^2 -- gma^-2 again, index untouched,
only the normalisation moved. Checked over a_rho = -0.5 .. -2.9: index -2.00 .. -1.87,
never near -1 (check_index_is_robust). THIS ARGUMENT IS SPECIFIC TO q = 0: the identity
alpha - e = -1 needs e = 1 + alpha, and at q > 0 it fails.

So at constant t'_c the effect is a renormalisation, not a new segment:
  - the tail keeps index ~ -2 (-1.95 just above the edge, drifting to -1.84 five decades
    up, because high gma was passed EARLY, when A was still ~1);
  - its amplitude is suppressed toward e = 1 + a_rho/3 = 1/3 deep down, rising back to 1
    near gma_m (check_deep_tail_ratio);
  - the bottom edge moves DOWN, x0.370 here (edge_drop_factor), tending to e only as
    sigma -> inf: the exact factor is A*e*sigma/((1+sigma)^e - 1) and that -1 is worth
    11% at sigma = 1e3 with e = 1/3.
The one genuinely new thing is that over 10^3 t_dyn the sharp cut-off at min(gma_c, gma_m)
is GONE, replaced by that long gma^-2 tail: electrons keep sliding instead of stalling.

  TRAP, and this check was wrong once: the bottom edge has TWO limits, not one.
  gma_m = A gma_m0/(1 + gma_m0 S) tends to A/S only when gma_m0*S >> 1 (cooled), and to
  A*gma_m0 -- pure adiabatic drag, no burn at all -- when gma_m0*S << 1. At log10 C = +3
  the population has barely cooled (gma_m0*S = 0.027), so scoring it against A/S reads
  97% off while nothing is wrong. check_bottom_edge gates each regime into the limit it
  is actually in.

VALIDITY, and it BINDS here in a way it did not before. These are ultra-relativistic
trajectories; below gma = 1 the real loss rate (gma^2 - 1, cooling_distribution.
coolingFunc_ODE) collapses and electrons stall near gma ~ 1 instead of continuing down,
which is why the emission code floors its gamma integral there. At 10^3 t_dyn every
regime with C <= 1 has its bottom edge BELOW that floor (C = -3 reaches gma = 3.7e-4), so
the shaded band is not decoration: inside it these curves are the model continued past
where it is true, drawn to show where the population is heading, not where it is.

Run:  python cooling_integrated_adiabatic.py
'''

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.integrate import quad

from cooling_distribution import norm_plaw_distrib, gamma_synCooled
from cooling_integrated_figure import (P_SYN, GM0, GMA_M0, LOGC_SAMPLES, OUTDIR,
    INK, MUTED, FIGSIZE, FS_LAB, FS_TICK, FS_ANN, FS_LEG, C_LABEL, GMA_LABEL,
    tt_dyn_of_C, log_slope, N_integrated)

# --- defaults -------------------------------------------------------------------------
A_RHO = -2.0                # dln rho/dln R: simple coasting, rho' ~ R^-2. The run's own
                            # measured value is -1.205 (prerar_cell_evolution alpha_D
                            # = -0.795); nothing qualitative moves, see check_index_is_robust
Q_B = 0.0                   # B' ~ R^-q. 0 = constant B'/t'_c, the footing cooling_shape
                            # is on. q=1 (B' ~ R^-1) makes the burn SATURATE -- see the
                            # CORRECTION block in the docstring
SIGMA_END = 1e3             # integrate to 10^3 t_dyn.  sigma = t'/t'_dyn (PHYSICAL)
NG = 700                    # points per curve (each costs 2 root-finds + 1 quadrature)
GMA_FLOOR = 1.              # below this the ultra-relativistic trajectory is not physical
FNAME = 'cooling_integrated_adiabatic.png'


# --- the expansion ---------------------------------------------------------------------
def _exps(a_rho=A_RHO, q=Q_B):
  '''
  (alpha, e) = (a_rho/3, alpha - 2q + 1). e sets how the synchrotron burn S grows with
  radius: e > 0 unbounded, e <= 0 SATURATES (the field decays faster than the electrons
  can radiate). Both signs are supported -- that is the point of working in S.
  '''
  alpha = a_rho/3.
  e = alpha - 2.*q + 1.
  if abs(e) < 1e-9:
    raise ValueError(f'a_rho={a_rho}, q={q} gives e=0 (S ~ log sigma); not supported.')
  return alpha, e


def A_of_sigma(sigma, a_rho=A_RHO):
  "Adiabatic factor A = (rho/rho_0)^(1/3) at sigma = t'/t'_dyn, coasting R/R_0 = 1+sigma."
  return (1. + np.asarray(sigma, dtype=float))**(a_rho/3.)


def S_of_sigma(sigma, ttd, a_rho=A_RHO, q=Q_B):
  'Effective synchrotron burn S = int_0^tt A dtt, as a function of PHYSICAL time.'
  _, e = _exps(a_rho, q)
  return ttd*((1. + np.asarray(sigma, dtype=float))**e - 1.)/e


def tt_of_sigma(sigma, ttd, q=Q_B):
  '''
  The normalised time itself, tt = int dt'/t_{c,1}, with t_c1 ~ R^2q. At q=0 this is just
  ttd*sigma; at q=1 it saturates at ttd, which is the whole content of the correction.
  '''
  sigma = np.asarray(sigma, dtype=float)
  k = 1. - 2.*q
  return ttd*np.log1p(sigma) if abs(k) < 1e-9 else ttd*((1. + sigma)**k - 1.)/k


def A_of_S(S, ttd, a_rho=A_RHO, q=Q_B):
  '''
  A expressed through S alone: (1 + e*S/ttd) = (1+sigma)^e, so A = (1+e*S/ttd)^(alpha/e).
  This is the form the quadrature needs and it is valid for either sign of e.
  '''
  alpha, e = _exps(a_rho, q)
  return (1. + e*np.asarray(S, dtype=float)/ttd)**(alpha/e)


def gamma_cooled(sigma, gma0, ttd, a_rho=A_RHO, q=Q_B):
  'gma(sigma) = A gma_0/(1 + gma_0 S) -- synchrotron AND adiabatic.'
  return (A_of_sigma(sigma, a_rho)*gma0
          /(1. + gma0*S_of_sigma(sigma, ttd, a_rho, q)))


# --- the distribution -------------------------------------------------------------------
def N_instant(gma, sigma, ttd, p=P_SYN, gm0=GM0, gM0=GMA_M0, a_rho=A_RHO, q=Q_B):
  'K0 A gma^-p (A - S gma)^(p-2), zero off the support.'
  gma = np.asarray(gma, dtype=float)
  A, S = A_of_sigma(sigma, a_rho), S_of_sigma(sigma, ttd, a_rho, q)
  fb = A - S*gma                                   # = gma/gma_0
  ok = (fb > gma/gM0) & (fb <= gma/gm0)            # gma_0 inside [gma_m0, gma_M0]
  return np.where(ok, norm_plaw_distrib(gm0, gM0, p)*A*gma**-p*np.abs(fb)**(p-2.), 0.)


def N_integrated_adiab(gma, ttd, sigma_end=SIGMA_END, p=P_SYN, gm0=GM0, gM0=GMA_M0,
    a_rho=A_RHO, q=Q_B):
  '''
  int_0^{t'(sigma_end)} N dtt at one gma, by the S quadrature of the module docstring.
  The support is solved for, never scanned.
  '''
  S_end = float(S_of_sigma(sigma_end, ttd, a_rho, q))
  fb = lambda S: float(A_of_S(S, ttd, a_rho, q)) - gma*S    # decreasing, fb(0) = 1

  def _root(target):
    if fb(0.) <= target:
      return 0.                                    # already past it at sigma = 0
    if fb(S_end) >= target:
      return None                                  # never reached within the window
    return brentq(lambda S: fb(S) - target, 0., S_end, xtol=1e-300, rtol=1e-14)

  S_a = _root(gma/gm0)
  if S_a is None:
    return 0.                                      # gma never reached in time
  S_b = _root(gma/gM0)
  S_b = S_end if S_b is None else S_b
  if S_b <= S_a:
    return 0.
  I = quad(lambda S: max(fb(S), 0.)**(p-2.), S_a, S_b, limit=400)[0]
  return norm_plaw_distrib(gm0, gM0, p)*gma**-p*I


def integrated_distrib_adiab(logC, sigma_end=SIGMA_END, p=P_SYN, gm0=GM0, gM0=GMA_M0,
    a_rho=A_RHO, q=Q_B, Ng=NG):
  '''
  (gma, N(gma; sigma_end)) over the support: from the lowest gma the bottom edge reaches,
  gma_m(sigma_end), up to the injected ceiling.
  '''
  ttd = tt_dyn_of_C(logC, gm0)
  gma = np.geomspace(gamma_cooled(sigma_end, gm0, ttd, a_rho, q), gM0, Ng)
  N = np.array([N_integrated_adiab(g, ttd, sigma_end, p, gm0, gM0, a_rho, q) for g in gma])
  return gma, N


# --- validation -------------------------------------------------------------------------
def check_reduces_to_syn(logC_arr=LOGC_SAMPLES, p=P_SYN, gm0=GM0, gM0=GMA_M0, n=7):
  '''
  THE tie to the validated module: at a_rho = 0 (no expansion) A = 1, S = tt, and this
  must reproduce cooling_integrated_figure.N_integrated, whose own closed form was checked
  against quadrature and a sum rule. Integrating to sigma_end = 1 makes the two the same
  quantity. Returns the largest relative deviation and the number of points compared.
  '''
  dev, npts = 0., 0
  for logC in logC_arr:
    ttd = tt_dyn_of_C(logC, gm0)
    lo = gamma_synCooled(ttd, gm0)
    for gma in np.geomspace(lo, gM0, n+2)[1:-1]:
      ref = float(N_integrated(gma, ttd, p, gm0, gM0))
      got = N_integrated_adiab(gma, ttd, 1., p, gm0, gM0, a_rho=1e-9, q=0.)
      if ref > 0.:
        dev = max(dev, abs(got/ref - 1.)); npts += 1
  return dev, npts


def check_trajectory(logC=0., sigma_end=SIGMA_END, gm0=GM0, a_rho=A_RHO, q=Q_B, n=6):
  '''
  gma(tau) = A gma_0/(1 + gma_0 S) against a direct ODE solve of
  dgma/dtau = tt_dyn*[(dlnA/dtau)gma/tt_dyn - gma^2], i.e. the equation before it was
  integrated. Returns the largest relative deviation over a spread of gma_0.
  '''
  from scipy.integrate import solve_ivp
  ttd = tt_dyn_of_C(logC, gm0)
  alpha, _ = _exps(a_rho, q)
  # dgma/dtau = alpha*gma/(1+tau) - ttd*gma^2   (dtt = ttd dtau)
  rhs = lambda sg, y: alpha*y/(1.+sg) - ttd*(1.+sg)**(-2.*q)*y*y
  dev = 0.
  for gma0 in np.geomspace(gm0, GMA_M0, n):
    sol = solve_ivp(rhs, (0., sigma_end), [gma0], rtol=1e-11, atol=1e-30, method='Radau')
    got = gamma_cooled(sigma_end, gma0, ttd, a_rho, q)
    dev = max(dev, abs(sol.y[0, -1]/got - 1.))
  return dev


def check_sum_rule(logC_arr=LOGC_SAMPLES, sigma_end=SIGMA_END, p=P_SYN, gm0=GM0, gM0=GMA_M0,
    a_rho=A_RHO, q=Q_B, Ng=1200):
  '''
  Number is conserved at every tt, so int N(gma;tt_end) dgma = tt_end for every regime.
  Returns (dev at Ng, dev at 2*Ng). Each point costs a root-find pair and a quadrature,
  so the grid here is thousands of points, not the synchrotron module's 4e5, and the
  residual is dominated by the TRAPEZOID over the support's hard edges -- first order,
  so the pair should roughly HALVE. That is the check: a ratio near 2 says the residual
  is the grid, a ratio near 1 would say it is the model.
  '''
  out = []
  for n in (Ng, 2*Ng):
    dev = 0.
    for logC in logC_arr:
      ttd = tt_dyn_of_C(logC, gm0)
      gma, N = integrated_distrib_adiab(logC, sigma_end, p, gm0, gM0, a_rho, q, n)
      tt_end = float(tt_of_sigma(sigma_end, ttd, q))   # NOT ttd*sigma_end unless q = 0
      dev = max(dev, abs(np.trapezoid(N, gma)/tt_end - 1.))
    out.append(dev)
  return tuple(out)


def check_bottom_edge(logC_arr=LOGC_SAMPLES, sigma_end=SIGMA_END, gm0=GM0, a_rho=A_RHO,
    q=Q_B, gate=30.):
  '''
  The bottom edge has TWO limits and the single asymptote b*gma_c/sigma_end is only one of
  them -- applying it everywhere is a 97% error at log10 C = +3, which is how this check
  was got wrong the first time. gma_m = A gma_m0/(1 + gma_m0 S) interpolates between

      gma_m0*S >> 1 (fast, cooled):  gma_m -> A/S
      gma_m0*S << 1 (slow, uncooled):  gma_m -> A*gma_m0   (pure adiabatic drag)

  Returns (dev_fast, n_fast, dev_slow, n_slow), each regime tested only against the limit
  it is actually in (gma_m0*S above `gate` or below 1/gate; the middle regimes are in neither limit).
  '''
  A, S1 = float(A_of_sigma(sigma_end, a_rho)), float(S_of_sigma(sigma_end, 1., a_rho, q))
  df, ds, nf, ns = 0., 0., 0, 0
  for logC in logC_arr:
    ttd = tt_dyn_of_C(logC, gm0)
    meas, gS = gamma_cooled(sigma_end, gm0, ttd, a_rho, q), gm0*S1*ttd
    if gS > gate:
      df = max(df, abs(meas/(A/(S1*ttd)) - 1.)); nf += 1
    elif gS < 1./gate:
      ds = max(ds, abs(meas/(A*gm0) - 1.)); ns += 1
  return df, nf, ds, ns


def edge_drop_factor(sigma_end=SIGMA_END, a_rho=A_RHO, q=Q_B):
  '''
  How far below the synchrotron-only bottom edge (~gma_c/sigma_end) the adiabatic one sits:
  (A/S)/(1/tt_end) = A*b*sigma_end/((1+sigma_end)^b - 1). It tends to b as sigma_end -> inf, but
  the (1+tau)^b - 1 is NOT negligible at tau = 1e3 with a small b -- 0.370 against b =
  0.333 for coasting. Returns (measured factor, b).
  '''
  _, e = _exps(a_rho, q)
  A, S1 = float(A_of_sigma(sigma_end, a_rho)), float(S_of_sigma(sigma_end, 1., a_rho, q))
  tt_end = float(tt_of_sigma(sigma_end, 1., q))
  return (A/S1)*tt_end, e


def check_deep_tail_ratio(logC=-3., sigma_end=SIGMA_END, p=P_SYN, gm0=GM0, gM0=GMA_M0,
    a_rho=A_RHO, q=Q_B, n=6):
  '''
  Deep in the tail the loss rate is gma^2/b, so the dwell time -- and with every electron
  already past, the whole integral -- is b times the synchrotron-only one at the same gma
  and the same sigma_end. Returns [(gma, ratio)] walking up from the synchrotron bottom
  edge; the ratio should leave b and climb back toward 1 near gma_m, where the electrons
  passed early and A was still ~1.
  '''
  ttd = tt_dyn_of_C(logC, gm0)
  tt_end = float(tt_of_sigma(sigma_end, ttd, q))  # compare at the SAME normalised time
  lo = gamma_synCooled(tt_end, gm0)               # syn support starts here, adiab is lower
  out = []
  for gma in np.geomspace(1.2*lo, gm0, n):
    ns = float(N_integrated(gma, tt_end, p, gm0, gM0))
    na = N_integrated_adiab(gma, ttd, sigma_end, p, gm0, gM0, a_rho, q)
    out.append((gma, na/ns))
  return out


def check_index_is_robust(a_rhos=(-0.5, -1.205, -2.0, -2.5, -2.9), logC=-3.,
    sigma_end=SIGMA_END, p=P_SYN, gm0=GM0, gM0=GMA_M0, q=Q_B):
  '''
  The claim the appendix rests on: the tail index is ~ -2 for ANY expansion law, because
  alpha - b = -1 pins a/gma whatever a_rho is. Only the amplitude (-> b) moves. Returns
  [(a_rho, b, index)].
  '''
  out = []
  for a_rho in a_rhos:
    _, e = _exps(a_rho, q)
    out.append((a_rho, e, measure_low_slope(logC, sigma_end, p, gm0, gM0, a_rho, q)[1]))
  return out


def measure_low_slope(logC, sigma_end=SIGMA_END, p=P_SYN, gm0=GM0, gM0=GMA_M0, a_rho=A_RHO,
    q=Q_B, frac=(3., 30.)):
  '''
  The index of the new low-energy tail, measured a decade or so above the bottom edge so
  neither the edge itself nor the gma_c break contaminates it. Returns (gma, slope).
  '''
  ttd = tt_dyn_of_C(logC, gm0)
  lo = gamma_cooled(sigma_end, gm0, ttd, a_rho, q)
  g1, g2 = lo*frac[0], lo*frac[1]
  n1, n2 = (N_integrated_adiab(g, ttd, sigma_end, p, gm0, gM0, a_rho, q) for g in (g1, g2))
  return np.sqrt(g1*g2), np.log(n2/n1)/np.log(g2/g1)


# --- the figure ---------------------------------------------------------------------------
def plot_integrated_adiab(logC=LOGC_SAMPLES, sigma_end=SIGMA_END, p=P_SYN, gm0=GM0,
    gM0=GMA_M0, a_rho=A_RHO, q=Q_B, outdir=OUTDIR, fname=FNAME, syn_ref=True, show=False):
  '''
  Same two-panel design as the synchrotron-only figure. The synchrotron-only result over
  the SAME 10^3 t_dyn is drawn underneath as a thin ghost of each curve, so the
  difference read off the figure is adiabatic cooling and not the integration limit.
  '''
  logC = np.asarray(logC, dtype=float)
  norm = plt.Normalize(vmin=logC.min(), vmax=logC.max())
  colors = plt.cm.jet(norm(logC))
  sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=norm)

  fig, (axN, axS) = plt.subplots(2, 1, figsize=FIGSIZE, sharex=True,
      gridspec_kw=dict(height_ratios=[1.9, 1.], hspace=.08))

  order = np.argsort(logC)[::-1]       # slowest cooling first, fast curves on top
  breaks = []
  for i in order:
    ttd = tt_dyn_of_C(logC[i], gm0)
    if syn_ref:   # the same integral without the expansion term, same upper limit
      tt_end = float(tt_of_sigma(sigma_end, ttd, q))   # same NORMALISED time, not sigma
      g_s = np.geomspace(gamma_synCooled(tt_end, gm0), gM0, 600)
      N_s = N_integrated(g_s, tt_end, p, gm0, gM0)
      axN.loglog(g_s/gm0, N_s, color=colors[i], lw=.55, alpha=.5, zorder=2)
      # the ghost belongs in the slope panel too: that the two lie on top of each other
      # over the whole plateau IS the result -- adiabatic cooling moves the cut-off and
      # the amplitude, not the index -- and it can only be read where slopes are drawn
      axS.semilogx(g_s/gm0, log_slope(g_s, N_s), color=colors[i], lw=.55, alpha=.5,
                   zorder=2)
    gma, N = integrated_distrib_adiab(logC[i], sigma_end, p, gm0, gM0, a_rho, q)
    axN.loglog(gma/gm0, N, color=colors[i], lw=1.2, solid_capstyle='round', zorder=3)
    axS.semilogx(gma/gm0, log_slope(gma, N), color=colors[i], lw=1.1, zorder=3)
    gma_c = 10.**logC[i]*gm0
    breaks.append((gma_c/gm0,
                   N_integrated_adiab(gma_c, ttd, sigma_end, p, gm0, gM0, a_rho, q)))
  breaks = np.array(breaks)
  axN.scatter(breaks[:, 0], breaks[:, 1], s=11, facecolors=colors[order],
              edgecolors='w', linewidths=.5, zorder=6)

  axN.set_ylabel('$N(\\gamma;10^{3}\\tilde{t}_{\\rm dyn})/N_{\\rm e}$', fontsize=FS_LAB)
  # the deep tail climbs to ~b/gma^2, so the top has to clear it; the bottom is the
  # gma_M rollover, the same span the synchrotron-only figure carries
  axN.set_ylim(1e-25, 1e3)

  # the expected indices are reference VALUES, so they belong on an axis: the guides stay
  # inside, their labels go on the right-hand spine as ticks. Inside the panel they had to
  # dodge the curves -- -2 and -p are only half an index apart and were hung on opposite
  # sides of their own lines -- and on the spine they simply line up.
  levels = ((-2., '$-2$'), (-p, '$-p$'), (-(p+1.), '$-(p+1)$'))
  for lev, _ in levels:
    axS.axhline(lev, color=MUTED, ls='--', lw=.7, zorder=1)
  axS.set_ylim(-(p+2.6), .4)
  axS.set_xlabel(GMA_LABEL, fontsize=FS_LAB)
  axS.set_ylabel('$\\mathrm{d}\\ln N/\\mathrm{d}\\ln\\gamma$', fontsize=FS_LAB)
  axR = axS.twinx()                       # right-hand spine carries the expected indices
  axR.set_ylim(axS.get_ylim())
  axR.set_yticks([lev for lev, _ in levels])
  axR.set_yticklabels([lab for _, lab in levels])
  axR.tick_params(axis='y', labelsize=FS_ANN, length=2.5, pad=1.5, colors=INK)
  axR.grid(False)

  for ax in (axN, axS):
    ax.axvspan(1e-12, GMA_FLOOR/gm0, color='crimson', alpha=.07, lw=0, zorder=0)
    ax.axvline(GMA_FLOOR/gm0, color='crimson', ls=':', lw=.9, zorder=1)
    ax.axvline(1., color=INK, ls=':', lw=.8, zorder=1)
    ax.axvline(gM0/gm0, color=INK, ls=':', lw=.8, zorder=1)
    ax.grid(alpha=.25, lw=.4)
    ax.tick_params(which='both', labelsize=FS_TICK)
  # ONE decade below the gma = 1 floor, not four: past that these curves are the model
  # continued past where it is true, and a panel should not be mostly that
  axN.set_xlim(.1/gm0, 4.*gM0/gm0)
  for v, lab in ((1., '$\\gamma_\\mathrm{m}$'), (gM0/gm0, '$\\gamma_\\mathrm{M}$')):
    axN.annotate(lab, (v, .985), xycoords=('data', 'axes fraction'), color=INK,
                 fontsize=FS_ANN, ha='center', va='top',
                 bbox=dict(fc='w', ec='none', alpha=.85, pad=1.))
  axS.annotate('$\\gamma=1$', (GMA_FLOOR/gm0, .10), xycoords=('data', 'axes fraction'),
               textcoords='offset points', xytext=(3, 0), color='crimson',
               fontsize=FS_ANN, ha='left', va='bottom',
               bbox=dict(fc='w', ec='none', alpha=.85, pad=1.))
  axN.scatter([], [], s=11, facecolors='none', edgecolors=INK, linewidths=.7,
              label='$\\gamma_\\mathrm{c}$')
  if syn_ref:
    axN.plot([], [], color=INK, lw=.55, alpha=.6, label='no adiab.')
  # right of top centre: the curves all run upper-left to lower-right, so this is the
  # one patch of the panel no line crosses, and it clears the gma_m / gma_M top labels
  # (which sit at x ~ 0.39-0.42 and ~ 0.93) because the box spans about 0.55-0.85
  axN.legend(fontsize=FS_LEG, loc='upper center', bbox_to_anchor=(.70, .95),
             framealpha=.9, handletextpad=.4, handlelength=1.4, labelspacing=.3,
             borderpad=.4)

  # the bar moves UP beside the top panel only: the slope panel's right-hand side now
  # carries the expected-index ticks. An EXPLICIT cax, not ax=axN -- stealing space from
  # one of two stacked shared-x panels leaves them different widths and breaks the
  # alignment the pair is read on.
  fig.subplots_adjust(right=.85)
  pos = axN.get_position()
  cax = fig.add_axes([.875, pos.y0, .032, pos.height])
  cb = fig.colorbar(sm, cax=cax)
  cb.set_label(C_LABEL, fontsize=FS_LAB)
  cax.tick_params(labelsize=FS_TICK)

  os.makedirs(outdir, exist_ok=True)
  path = os.path.join(outdir, fname)
  fig.savefig(path, dpi=300, bbox_inches='tight')
  print(f'saved {path}')
  if show:
    plt.show()
  return fig, (axN, axS)


def main(show=False):
  alpha, e = _exps(A_RHO, Q_B)
  print(f"hydro : simple coasting, rho' ~ R^{A_RHO:g};  field: B' ~ R^-{Q_B:g}"
        f"{'  (constant B'+chr(39)+'/t'+chr(39)+'_c -- the cooling_shape footing)' if Q_B == 0 else ''}")
  print(f'        alpha={alpha:+.5f}  e=alpha-2q+1={e:+.5f}  -> burn '
        f'{"grows without bound" if e > 0 else f"SATURATES at S={-1./e:.3f} tt_dyn"}')
  A = float(A_of_sigma(SIGMA_END, A_RHO))
  print(f'over sigma={SIGMA_END:g} t_dyn: A={A:.4g} (adiabatic drag x{1./A:.1f}), '
        f'tt_end/tt_dyn={float(tt_of_sigma(SIGMA_END, 1.)):.4g}, '
        f'S/tt_dyn={float(S_of_sigma(SIGMA_END, 1.)):.4g}')

  # what the OTHER field index would do -- the prescription this module used to claim
  for qq in (0., 1.):
    al, ee = _exps(A_RHO, qq)
    print(f'   q={qq:g}: tt_end/tt_dyn={float(tt_of_sigma(SIGMA_END, 1., qq)):9.4g}   '
          f'S/tt_dyn={float(S_of_sigma(SIGMA_END, 1., A_RHO, qq)):9.4g}   '
          f'{"unbounded" if ee > 0 else "SATURATED"}')

  dev, n = check_reduces_to_syn()
  print(f'reduces to syn : max rel dev over {n} points at a_rho->0, q=0 = {dev:.2e}')
  print(f'trajectory ODE : max rel dev at sigma={SIGMA_END:g} = {check_trajectory():.2e}')
  s1, s2 = check_sum_rule()
  print(f'sum rule       : {s1:.2e} at Ng, {s2:.2e} at 2Ng (ratio {s1/s2:.2f}: '
        'first order, so it is the trapezoid over the support edges, not the model)')
  df, nf, ds, ns = check_bottom_edge()
  drop, ee = edge_drop_factor()
  print(f'bottom edge    : fast limit A/S  dev {df:.2e} ({nf} regimes); '
        f'slow limit A*gma_m0 dev {ds:.2e} ({ns})')
  print(f'               : edge drops x{drop:.3f} below the synchrotron-only one at the '
        f'same tt (-> e = {ee:.3f} as sigma -> inf)')
  print(f'deep-tail ratio N_adiab/N_syn (-> e = {ee:.4f} deep, -> 1 near gma_m):')
  for g, r in check_deep_tail_ratio():
    print(f'    gma={g:10.3e}: {r:.4f}')
  print('index is ~ -2 for ANY a_rho AT q=0 (only the amplitude, e, moves):')
  for a_rho, ee2, idx in check_index_is_robust():
    print(f'    a_rho={a_rho:+.3f}  e={ee2:.4f}  ->  index {idx:+.4f}')
  print("and what B' ~ R^-1 (q=1) does to that same index, since the burn saturates:")
  for qq in (0., .5, 1.):
    try:
      idx = measure_low_slope(-3., q=qq)[1]
      _, ee3 = _exps(A_RHO, qq)
      print(f'    q={qq:.1f}  e={ee3:+.4f}  ->  index {idx:+.4f}')
    except ValueError as exc:
      print(f'    q={qq:.1f}  skipped: {exc}')
  print('bottom edge and the low-energy tail per regime:')
  for lc in LOGC_SAMPLES:
    ttd = tt_dyn_of_C(lc, GM0)
    lo = gamma_cooled(SIGMA_END, GM0, ttd, A_RHO, Q_B)
    g, sl = measure_low_slope(lc)
    print(f'  log10 C = {lc:+.0f}: gma_m(10^3 t_dyn) = {lo:10.3e}'
          f'{"  (BELOW the gma=1 floor)" if lo < GMA_FLOOR else "":26s}'
          f'  tail index at gma={g:9.3e}: {sl:+.3f}')
  plot_integrated_adiab(show=show)


if __name__ == '__main__':
  main()
