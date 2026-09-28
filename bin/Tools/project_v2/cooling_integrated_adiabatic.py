# -*- coding: utf-8 -*-
# @Author: acharlet

'''
The time-integrated electron distribution WITH adiabatic cooling, over t_dyn.

APPENDIX ILLUSTRATION on a FREELY EXPANDING SHELL at constant eps_B: rho' ~ R^-2, so
a_rho = -2, and the field follows from it rather than being chosen -- see THE
HYDRODYNAMICS below. Earlier drafts held t'_c constant (q = 0) to isolate the adiabatic
drag; the field decays too, and that is not a cosmetic difference -- see THE BURN
SATURATES.

THE EXPONENTS ARE DEFINED POSITIVE, the signs carried by the definitions. With
tau = t'/t'_i,

    A = tau^-d          d = -a_rho/3 = 2/3          the adiabatic drag
    t'_c = t'_c,i tau^q   q = -a_rho*gma_ad = 10/3   the clock stretching
    s = d + q - 1 = 3                                the one combination that matters

and s > 0 is exactly the saturating case: S = (t'_i/t'_c,i)(1 - tau^-s)/s, bounded by
S_inf = (t'_i/t'_c,i)/s. Nothing in the module needs |.| or a sign test any more.

sigma_end is a PARAMETER and the figure is meant to be re-run at several values; the
numbers below are at the current default, 100 t_dyn, with gma_ad = 5/3:

    A(100) = 0.0461 (drag x21.7)      S/tt_dyn = 0.3333 (SATURATED, = 1/s)
    tt_end/tt_dyn = 0.4286            edge drops x0.059 below synchrotron-only
    deep-tail ratio 0.571             1 of 7 regimes under gma = 1 (C = 1e-3, to 0.138)
    low-energy index +2.50            against -1.45 when integrating to ONE t_dyn

THE LOW-ENERGY INDEX IS POSITIVE HERE, at +5/2, and that is the headline rather than a
detail. Once the burn has frozen every trajectory collapses onto gma ~ A, so essentially
the whole population passes a given low gma together and the integral is just its dwell
time. With gma ~ tau^-d and dtt = (t'_i/t'_c,i) tau^-q dtau,

    dtt/dgma ~ gma^((q-1)/d - 1)   =>   index = (q-1)/d - 1 = +5/2

at a_rho = -2, gma_ad = 5/3. Measured +2.499 on the plateau, and the prediction tracks
a_rho: -2 -> +5/2, -2.5 -> +2.8, -2.9 -> +2.97 (check_index_is_robust prints them).
NOTE the singular law: s = 0 at a_rho = -3/(1+3*gma_ad), which is -0.5 at gma_ad = 5/3,
where the burn is logarithmic and there is no power form at all.

So the time-integrated distribution PEAKS at gma_c and falls away below it as gma^+5/2,
instead of holding a gma^-2 plateau down to a sharp cut-off. tt is the right weight for a
fluence (dt'*P ~ dt'*B'^2 ~ dtt), so this is a statement about the emitted spectrum and
not only about the electrons. The gma_c markers sit on those peaks. Integrating to one
t_dyn instead returns the old picture, which is why sigma_end stays a parameter.

  MEASUREMENT TRAP, and it cost a wrong number in two earlier versions of this file:
  measure_low_slope's window used to be 3x-30x above the cut-off, which STRADDLES the
  turnover toward gma_c and averages the plateau with the decline beyond it. The plateau
  runs from the cut-off to about 3x above it, so the window is 1.2x-3x now. The same
  mistake sized the slope panel, whose top was set at the plateau's own value and so cut
  it off and made it look like an edge spike.

THE EQUATION IS LINEAR IN u = 1/gma. With the adiabatic term the cooling equation is

    dgma/dtt = (dlnA/dtt) gma - gma^2,     A = (rho'/rho'_i)^(1/3),

which in u = 1/gma is LINEAR -- du/dtt + (dlnA/dtt) u = 1 -- the same structure
working_cooling.evolve_gma_bounds_edges telescopes over. With A as integrating factor:

    u(tt) = [u_0 + S]/A,    S = int_0^tt A dtt',    => gma = A gma_0/(1 + gma_0 S).

S is the EFFECTIVE synchrotron burn: expansion drags gma down through A AND throttles the
losses that follow, and S is what is left of them. A = 1, S = tt recovers gamma_synCooled.
Number conservation (dgma_0/dgma = A/(A - S*gma)^2) then gives

    N(gma,tt) = K0 A gma^-p (A - S*gma)^(p-2)   on [gma_m(tt), gma_M(tt)],

the synchrotron form with 1 -> A and tt -> S. Both edges follow the same trajectory.

THE HYDRODYNAMICS: coasting, freely expanding shell. R/R_0 = 1 + sigma with
sigma = t'/t'_dyn (t_dyn = the comoving radius-doubling time R/(Gamma c),
cooling_distribution.get_tdbl_cell), and

    a_rho = dln rho'/dln R = -2,    d = -a_rho/3 = 2/3    (fixed comoving width).

B' IS NOT FREE. A constant fraction of the energy density goes to the field,
B'^2/8pi = eps_B e', and e' ~ rho'^gma_ad along an adiabat, so B' ~ rho'^(gma_ad/2) and
t'_c ~ B'^-2 gives q = -a_rho*gma_ad = 10/3 at gma_ad = 5/3. It is DERIVED, not chosen,
and a_rho and gma_ad are the only two dials.

With s = d + q - 1,  S(sigma) = tt_dyn (1 - (1+sigma)^-s)/s, so (1 - s*S/tt_dyn) =
(1+sigma)^-s and A can be written as a function of S alone,

    A(S) = (1 - s*S/tt_dyn)^(d/s),

which is what the code uses -- valid for BOTH signs of s, where a (1+tau)^b
parameterisation is not; at s > 0 the bracket runs from 1 down to 0, hitting 0 exactly
at S_inf. a_rho is free too; the run's own measured value is -1.205
(prerar_cell_evolution alpha_D = -0.795 through dlnrho/dlnR = -2 - alpha_D, the shell
being spread rather than of fixed width), and at gma_ad = 5/3 that gives s = 1.41.
At q = 0, s = d - 1 > 0 iff a_rho < -3; wherever s > 0 the burn saturates and
synchrotron cooling freezes out at a finite total.

THE TIME INTEGRAL. Since dS = A dtt, the integral collapses to one smooth quadrature in S:

    N(gma;tt_end) = K0 gma^-p INT_{S_a}^{S_b} f(S)^(p-2) dS,   f(S) = A(S) - S*gma,

with f strictly DECREASING (both terms fall), so the support -- gma_0 = gma/f in
[gma_m0, gma_M0] -- is one interval bracketed by two monotone root-finds: f(S_a) =
gma/gma_m0 (or S_a = 0) and f(S_b) = gma/gma_M0 (or S_b = S_end). The window is SOLVED
for, never scanned, which is what keeps the very narrow ones at high gma exact -- handed
the full range instead, an adaptive rule misses them and silently returns zero.

GMA_C CARRIES THE CHANGED RATE. 1/tt_dyn is the cooling Lorentz factor only while the
rate is constant. With the field decaying, the electron that has just cooled by sigma_end
is the one with gma_0 S ~ 1, and what survives the drag is

    gma_c = A/S = 1/tt_eff(sigma_end),

which reduces to 1/tt_end when A = 1 and S = tt. The difference is not cosmetic: at the
shell values A/S = 0.138/tt_dyn against 1/tt_dyn, a factor 7 apart. A/S lands on the
cut-off -- 0.1383 against a measured 0.1379 at log10 C = -3 -- which is where the break
really is, because gma_m -> A/S once gma_m0 S >> 1.

THE BURN SATURATES, and it decides which regimes can cool at all. With s = d + q - 1 = 3
the integral in S converges: instead of growing without bound,

    S -> S_inf = tt_dyn/s = 0.333 tt_dyn,     and tt itself -> tt_dyn/(q-1) = 0.429 tt_dyn.

An electron can therefore only ever accumulate a FINITE synchrotron burn, however long
you wait. The bottom edge cools only if that finite budget covers it, S_inf > 1/gma_m0,
which is a condition on the regime alone:

    C < 1/s = 0.333     (log10 C < -0.477)

Of the seven regimes drawn, only log10 C = -3, -2 and -1 ever reach fast cooling; the
other four stay slow-cooling FOREVER, not merely within the integration window. At q = 0
and a_rho > -3 that statement does not exist -- S grows without bound and every regime
gets there eventually. It is the single sharpest consequence of the field decaying.

It also makes the tt_eff clock crawl (cooling_shape_adiabatic's abscissa): tt_eff grows
only as sigma^d, so tt_eff = 1 needs sigma = 1.6e5 at C = 1 and 5.2e9 at C = 1e3. Late
tt_eff on that panel is not a time this system reaches.

WHAT ADIABATIC COOLING CHANGES -- AND THE gma^-1 THAT DOES NOT HAPPEN HERE. The synchrotron-only gma^-2 segment is a dwell time, dtt = dgma/gma^2. With
expansion the loss rate is gma^2 + a*gma, and the tempting reading -- that the tail
flattens toward gma^-1 wherever a > gma -- is WRONG at q = 0. It treats a as fixed while
gma falls; they fall TOGETHER. At q = 0, s = d - 1, so s - d = -1 identically and the
bottom edge gma_m ~ A/S ~ sigma^(s-d) = 1/sigma decays at exactly the rate
a ~ 1/sigma does. Hence a/gma -> d/|s|, CONSTANT (2.0 for coasting), the loss rate is
gma^2 (1 + d/|s|) and the dwell time is dgma/[gma^2 (1 + d/|s|)] -- gma^-2 again, index
untouched, only the normalisation moved. main()'s q-scan row shows it: at q = 0 the index
is -1.94 at sigma_end = 100 and -1.97 at 1e3, never near -1. THIS ARGUMENT IS SPECIFIC TO
q = 0: the identity s - d = -1 needs s = d - 1, and at q > 0 it fails.

So at constant t'_c the effect is a renormalisation, not a new segment:
  - the tail keeps index ~ -2: -1.932 at one t_dyn, and -1.98 .. -1.90 across
    a_rho = -0.5 .. -2.9 (check_index_is_robust);
  - its amplitude is suppressed toward e = 1 + a_rho/3 = 1/3, but only ASYMPTOTICALLY --
    over one t_dyn it reaches just 0.754 at the deepest gma where the synchrotron curve
    still exists to divide by, climbing back to 1 near gma_m (check_deep_tail_ratio);
  - the bottom edge moves DOWN by x0.808 (edge_drop_factor), far short of its e = 1/3
    limit: the exact factor is A*e*sigma/((1+sigma)^e - 1), and at sigma = 1 the -1 in
    that denominator dominates it.
So over one t_dyn the effect is MODEST and the cut-off SURVIVES, merely pushed down by
that x0.808. Only over many dynamical times does it wash out into a long gma^-2 tail,
electrons sliding on instead of stalling -- x0.350 and a tail reaching gma = 3.5e-5 at
sigma_end = 1e4. That contrast is why sigma_end stays a parameter.

  TRAP, and this check was wrong once: the bottom edge has TWO limits, not one.
  gma_m = A gma_m0/(1 + gma_m0 S) tends to A/S only when gma_m0*S >> 1 (cooled), and to
  A*gma_m0 -- pure adiabatic drag, no burn at all -- when gma_m0*S << 1. At log10 C = +3
  the population has barely cooled at short sigma_end (gma_m0*S = 7.8e-4 at one t_dyn),
  so scoring it against A/S reads wildly off while nothing is wrong. check_bottom_edge
  gates each regime into the limit it is actually in. HOW MANY fall in each moves with
  sigma_end -- 2 fast / 2 slow at the default one t_dyn, 4 fast / 0 slow at 10^4, where
  the burn has cooled even the slowest regime -- which is why it is gated and reported,
  not hard-coded. A zero count is a statement about the scan point, not a failure.

VALIDITY, and it BINDS here in a way it did not before. These are ultra-relativistic
trajectories; below gma = 1 the real loss rate (gma^2 - 1, cooling_distribution.
coolingFunc_ODE) collapses and electrons stall near gma ~ 1 instead of continuing down,
which is why the emission code floors its gamma integral there. At the default one
t_dyn this barely bites: only the fastest regime crosses, and barely -- log10 C = -3 ends
at gma = 0.807 -- so the shaded band is a sliver at the left edge rather than half the
panel. It still means what it says for that one curve. The caveat grows with sigma_end,
and by 10^4 t_dyn FIVE of the seven regimes end below the floor, down to gma = 3.5e-5,
with only log10 C = +2 and +3 entirely above it; there the shaded half of the panel is
the ultra-relativistic trajectory continued far past where it holds, and should be read
as where the population is HEADING, not where it is.

Run:  python cooling_integrated_adiabatic.py
'''

import os
import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction
from scipy.optimize import brentq
from scipy.integrate import quad

from cooling_distribution import norm_plaw_distrib, gamma_synCooled
from cooling_integrated_figure import (P_SYN, GM0, GMA_M0, LOGC_SAMPLES, OUTDIR,
    INK, MUTED, FIGSIZE, FS_LAB, FS_TICK, FS_ANN, FS_LEG, C_LABEL, GMA_LABEL,
    tt_dyn_of_C, log_slope, N_integrated)

# --- defaults -------------------------------------------------------------------------
A_RHO = -2.0                # dln rho'/dln R: a FREELY EXPANDING shell, rho' ~ R^-2
GMA_AD = 5./3.              # adiabatic index of the shocked gas. This run measures
                            # gma_ad ~ 1.64 (prerar_cell_evolution), so 5/3 rather than
                            # the relativistic 4/3. It enters only through Q_C, below
# THE EXPONENTS ARE DEFINED POSITIVE, with the signs carried by the definitions, so that
# an expanding shell with a decaying field has d, q, s all > 0. With tau = t'/t'_i,
#     A = tau^-d        d = -a_rho/3            (the adiabatic drag)
#     t'_c = t'_c,i tau^q                       (the clock stretching)
# B' IS NOT FREE. A constant fraction of the energy density goes to the field,
# B'^2/8pi = eps_B e', and e' ~ rho'^gma_ad along an adiabat, so B' ~ rho'^(gma_ad/2),
# and t'_c ~ B'^-2 then gives   q = -a_rho*gma_ad.   At the shell values q = 10/3.
Q_C = -A_RHO*GMA_AD
SIGMA_END = 100.            # integrate to 100 t_dyn.  sigma = t'/t'_dyn (PHYSICAL time).
                            # A parameter the user scans: at 1 it matches
                            # cooling_integrated_figure exactly, at 1e3 the cut-off has
                            # washed out entirely. The panel limits follow it (below)
NG = 700                    # points per curve (each costs 2 root-finds + 1 quadrature)
GMA_FLOOR = 1.              # below this the ultra-relativistic trajectory is not physical
FNAME = 'cooling_integrated_adiabatic.png'


def _pow10(x):
  'Render an integration limit for a label: 1 -> \'\', 100 -> 10^{2}, 2.5 -> 2.5.'
  l = np.log10(x)
  return ('' if x == 1. else
          f'10^{{{l:.0f}}}' if abs(l - round(l)) < 1e-9 else f'{x:g}')


# --- the expansion ---------------------------------------------------------------------
def _exps(a_rho=A_RHO, q=Q_C):
  '''
  (d, s) = (-a_rho/3, d + q - 1), both POSITIVE for an expanding shell with a decaying
  field. With tau = t'/t'_i, A = tau^-d and t'_c = t'_c,i tau^q, so the burn integrand
  A/t'_c goes as tau^-(d+q) and

      S(tau) = (t'_i/t'_c,i) (1 - tau^-s)/s .

  s > 0 therefore SATURATES the burn at S_inf = (t'_i/t'_c,i)/s: the field decays faster
  than the electrons can radiate. s < 0 leaves it unbounded. Both signs are supported --
  that is the point of working in S.
  '''
  d = -a_rho/3.
  s = d + q - 1.
  if abs(s) < 1e-9:
    raise ValueError(f'a_rho={a_rho}, q={q} gives s=0 (S ~ log sigma); not supported.')
  return d, s


def A_of_sigma(sigma, a_rho=A_RHO):
  "A = (rho'/rho'_i)^(1/3) = tau^-d at sigma = t'/t'_dyn, coasting tau = 1 + sigma."
  return (1. + np.asarray(sigma, dtype=float))**(a_rho/3.)


def S_of_sigma(sigma, ttd, a_rho=A_RHO, q=Q_C):
  'Effective synchrotron burn S = int_0^tt A dtt, as a function of PHYSICAL time.'
  _, s_ = _exps(a_rho, q)
  return ttd*(1. - (1. + np.asarray(sigma, dtype=float))**(-s_))/s_


def tt_of_sigma(sigma, ttd, q=Q_C):
  '''
  The normalised time itself, tt = int dt'/t_{c,1}, with t'_c ~ tau^q. At q = 0 this is
  just ttd*sigma; at q = 2 it saturates at ttd, which is the whole content of the
  correction.
  '''
  sigma = np.asarray(sigma, dtype=float)
  k = 1. - q
  return ttd*np.log1p(sigma) if abs(k) < 1e-9 else ttd*((1. + sigma)**k - 1.)/k


def A_of_S(S, ttd, a_rho=A_RHO, q=Q_C):
  '''
  A expressed through S alone: (1 - s*S/ttd) = tau^-s, so A = (1 - s*S/ttd)^(d/s). This
  is the form the quadrature needs and it is valid for either sign of s; at s > 0 the
  bracket runs from 1 down to 0, reaching 0 exactly at the saturated S_inf = ttd/s.
  '''
  d, s_ = _exps(a_rho, q)
  return (1. - s_*np.asarray(S, dtype=float)/ttd)**(d/s_)


def gamma_cooled(sigma, gma0, ttd, a_rho=A_RHO, q=Q_C):
  'gma(sigma) = A gma_0/(1 + gma_0 S) -- synchrotron AND adiabatic.'
  return (A_of_sigma(sigma, a_rho)*gma0
          /(1. + gma0*S_of_sigma(sigma, ttd, a_rho, q)))


# --- the distribution -------------------------------------------------------------------
def N_instant(gma, sigma, ttd, p=P_SYN, gm0=GM0, gM0=GMA_M0, a_rho=A_RHO, q=Q_C):
  'K0 A gma^-p (A - S gma)^(p-2), zero off the support.'
  gma = np.asarray(gma, dtype=float)
  A, S = A_of_sigma(sigma, a_rho), S_of_sigma(sigma, ttd, a_rho, q)
  fb = A - S*gma                                   # = gma/gma_0
  ok = (fb > gma/gM0) & (fb <= gma/gm0)            # gma_0 inside [gma_m0, gma_M0]
  return np.where(ok, norm_plaw_distrib(gm0, gM0, p)*A*gma**-p*np.abs(fb)**(p-2.), 0.)


def N_integrated_adiab(gma, ttd, sigma_end=SIGMA_END, p=P_SYN, gm0=GM0, gM0=GMA_M0,
    a_rho=A_RHO, q=Q_C):
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
    a_rho=A_RHO, q=Q_C, Ng=NG):
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


def check_trajectory(logC=0., sigma_end=SIGMA_END, gm0=GM0, a_rho=A_RHO, q=Q_C, n=6):
  '''
  gma(tau) = A gma_0/(1 + gma_0 S) against a direct ODE solve of
  dgma/dtau = tt_dyn*[(dlnA/dtau)gma/tt_dyn - gma^2], i.e. the equation before it was
  integrated. Returns the largest relative deviation over a spread of gma_0.
  '''
  from scipy.integrate import solve_ivp
  ttd = tt_dyn_of_C(logC, gm0)
  d_, _ = _exps(a_rho, q)
  # dgma/dtau = -d*gma/(1+tau) - ttd*(1+tau)^-q*gma^2   (dtt = ttd (1+tau)^-q dtau)
  rhs = lambda sg, y: -d_*y/(1.+sg) - ttd*(1.+sg)**(-q)*y*y
  dev = 0.
  for gma0 in np.geomspace(gm0, GMA_M0, n):
    sol = solve_ivp(rhs, (0., sigma_end), [gma0], rtol=1e-11, atol=1e-30, method='Radau')
    got = gamma_cooled(sigma_end, gma0, ttd, a_rho, q)
    dev = max(dev, abs(sol.y[0, -1]/got - 1.))
  return dev


def check_sum_rule(logC_arr=LOGC_SAMPLES, sigma_end=SIGMA_END, p=P_SYN, gm0=GM0, gM0=GMA_M0,
    a_rho=A_RHO, q=Q_C, Ng=1200):
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
    q=Q_C, gate=30.):
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


def edge_drop_factor(sigma_end=SIGMA_END, a_rho=A_RHO, q=Q_C):
  '''
  How far below the synchrotron-only bottom edge (~gma_c/sigma_end) the adiabatic one sits:
  (A/S)/(1/tt_end) = A*b*sigma_end/((1+sigma_end)^b - 1). It tends to b as sigma_end -> inf, but
  the (1+tau)^b - 1 is NOT negligible at tau = 1e3 with a small b -- 0.370 against b =
  0.333 for coasting. Returns (measured factor, b).
  '''
  _, s_ = _exps(a_rho, q)
  A, S1 = float(A_of_sigma(sigma_end, a_rho)), float(S_of_sigma(sigma_end, 1., a_rho, q))
  tt_end = float(tt_of_sigma(sigma_end, 1., q))
  # the -s limit is a q = 0 result: it needs d + s = 1. Once the burn saturates (s > 0)
  # S stops growing while A keeps falling, so the drop has NO finite limit -- it goes to
  # zero. Reporting -s there would assert a negative ratio.
  return (A/S1)*tt_end, (-s_ if s_ < 0. else None)


def check_deep_tail_ratio(logC=-3., sigma_end=SIGMA_END, p=P_SYN, gm0=GM0, gM0=GMA_M0,
    a_rho=A_RHO, q=Q_C, n=6):
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
    sigma_end=SIGMA_END, p=P_SYN, gm0=GM0, gM0=GMA_M0, gma_ad=GMA_AD):
  '''
  The claim the appendix rests on: the tail index is ~ -2 for ANY expansion law, because
  d + s = 1 pins a/gma whatever a_rho is. Only the amplitude moves. Returns
  [(a_rho, s, index)].

  SKIPS the singular law. s = 0 at a_rho = -3/(1+3*gma_ad) -- -0.5 at gma_ad = 5/3, which
  was in the default list -- where the burn is logarithmic and S(tau) has no power form.
  That is one law out of a continuum, not a failure of the claim, so it is stepped over
  rather than raised.
  '''
  out = []
  for a_rho in a_rhos:
    q = -a_rho*gma_ad             # q is TIED to a_rho; scanning at fixed q would leave
    if abs(-a_rho/3. + q - 1.) < 1e-9:        # the B'-rho relation behind
      continue
    _, s_ = _exps(a_rho, q)
    out.append((a_rho, s_, measure_low_slope(logC, sigma_end, p, gm0, gM0, a_rho, q)[1]))
  return out


def measure_low_slope(logC, sigma_end=SIGMA_END, p=P_SYN, gm0=GM0, gM0=GMA_M0, a_rho=A_RHO,
    q=Q_C, frac=(1.2, 3.)):
  '''
  The index of the low-energy tail, measured on the PLATEAU: 1.2x to 3x above the bottom
  edge. A 3x-30x window straddled the turnover toward gma_c and averaged the plateau with
  the decline, reporting +0.33 where the plateau is +1.50. Returns (gma, slope).
  '''
  ttd = tt_dyn_of_C(logC, gm0)
  lo = gamma_cooled(sigma_end, gm0, ttd, a_rho, q)
  g1, g2 = lo*frac[0], lo*frac[1]
  n1, n2 = (N_integrated_adiab(g, ttd, sigma_end, p, gm0, gM0, a_rho, q) for g in (g1, g2))
  return np.sqrt(g1*g2), np.log(n2/n1)/np.log(g2/g1)


# --- the figure ---------------------------------------------------------------------------
def plot_integrated_adiab(logC=LOGC_SAMPLES, sigma_end=SIGMA_END, p=P_SYN, gm0=GM0,
    gM0=GMA_M0, a_rho=A_RHO, q=Q_C, outdir=OUTDIR, fname=FNAME, syn_ref=True, show=False):
  '''
  Same two-panel design as the synchrotron-only figure. The synchrotron-only result over
  the SAME sigma_end is drawn underneath as a thin ghost of each curve (at sigma_end = 1
  those ghosts ARE cooling_integrated_figure's curves), so the difference read off the
  figure is adiabatic cooling and not the integration limit.
  '''
  logC = np.asarray(logC, dtype=float)
  norm = plt.Normalize(vmin=logC.min(), vmax=logC.max())
  colors = plt.cm.jet(norm(logC))
  sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=norm)

  fig, (axN, axS) = plt.subplots(2, 1, figsize=FIGSIZE, sharex=True,
      gridspec_kw=dict(height_ratios=[1.9, 1.], hspace=.08))

  order = np.argsort(logC)[::-1]       # slowest cooling first, fast curves on top
  breaks, lo_edge, hi_N = [], np.inf, 0.
  for i in order:
    ttd = tt_dyn_of_C(logC[i], gm0)
    if syn_ref:   # the same integral without the expansion term, same upper limit
      tt_end = float(tt_of_sigma(sigma_end, ttd, q))   # same NORMALISED time, not sigma
      g_s = np.geomspace(gamma_synCooled(tt_end, gm0), gM0, 600)
      N_s = N_integrated(g_s, tt_end, p, gm0, gM0)
      axN.loglog(g_s, N_s, color=colors[i], lw=.55, alpha=.5, zorder=2)
      # the ghost belongs in the slope panel too: that the two lie on top of each other
      # over the whole plateau IS the result -- adiabatic cooling moves the cut-off and
      # the amplitude, not the index -- and it can only be read where slopes are drawn
      axS.semilogx(g_s, log_slope(g_s, N_s), color=colors[i], lw=.55, alpha=.5,
                   zorder=2)
    gma, N = integrated_distrib_adiab(logC[i], sigma_end, p, gm0, gM0, a_rho, q)
    lo_edge, hi_N = min(lo_edge, gma[0]), max(hi_N, float(N.max()))
    axN.loglog(gma, N, color=colors[i], lw=1.2, solid_capstyle='round', zorder=3)
    axS.semilogx(gma, log_slope(gma, N), color=colors[i], lw=1.1, zorder=3)
    # gma_c FROM THE ACTUAL BURN, not from tt_dyn. 1/tt_dyn is the cooling Lorentz
    # factor only if the rate is constant; with the field decaying the electron that has
    # just cooled by sigma_end is the one with gma_0 S ~ 1, and what survives the drag is
    #     gma_c = A/S = 1/tt_eff(sigma_end),
    # which reduces to 1/tt_end when A = 1, S = tt. It matters: at 1/tt_dyn the marker
    # fell BELOW the support in every fast-cooling regime -- gma_c = 1 against a cut-off
    # at 1.705 -- so no dot was drawn there at all. A/S lands on the cut-off, which is
    # where the break actually is, because gma_m -> A/S once gma_m0 S >> 1.
    # evaluated AT ONE t_dyn, not at sigma_end: gma_c is the cooling Lorentz factor at
    # the dynamical time, a property of the system, and it should not slide when the
    # integration window is lengthened. It still lands inside the support, since the
    # bottom edge only moves further down as sigma_end grows.
    gma_c = float(A_of_sigma(1., a_rho))/float(S_of_sigma(1., ttd, a_rho, q))
    breaks.append((gma_c,
                   N_integrated_adiab(gma_c, ttd, sigma_end, p, gm0, gM0, a_rho, q)))
  breaks = np.array(breaks)
  axN.scatter(breaks[:, 0], breaks[:, 1], s=11, facecolors=colors[order],
              edgecolors='w', linewidths=.5, zorder=6)

  axN.set_ylabel("$N_{\\rm e}^{-1}\\,{\\rm d}N_{\\rm e}/{\\rm d}\\gamma_{\\rm e}"
                 + f'\\;({_pow10(sigma_end)}\\tilde{{t}}_{{\\rm dyn}})$',
                 fontsize=FS_LAB)
  # the limits FOLLOW the support: sigma_end is scanned, and hand-set bounds either
  # crop the fastest cut-offs off the panel or leave decades of blank. The floor stays
  # fixed -- it is the gma_M rollover, which sigma_end does not move.
  axN.set_ylim(1e-25, 10.*hi_N)

  # the expected indices are reference VALUES, so they belong on an axis: the guides stay
  # inside, their labels go on the right-hand spine as ticks. Inside the panel they had to
  # dodge the curves -- -2 and -p are only half an index apart and were hung on opposite
  # sides of their own lines -- and on the spine they simply line up.
  # the low-energy plateau, (q-1)/d - 1, joins the three cooled-segment levels. DERIVED
  # from the hydro rather than written in, so it follows a_rho and q.
  d_, _ = _exps(a_rho, q)
  fr = Fraction((q - 1.)/d_ - 1.).limit_denominator(100)
  lo_lab = (f'${fr.numerator:+d}$' if fr.denominator == 1
            else f'${fr.numerator:+d}/{fr.denominator}$')
  levels = ((float(fr), lo_lab), (-2., '$-2$'), (-p, '$-p$'), (-(p+1.), '$-(p+1)$'))
  for lev, _ in levels:
    axS.axhline(lev, color=MUTED, ls='--', lw=.7, zorder=1)
  # the low-energy plateau sits at (q-1)/d - 1, which is +5/2 at gma_ad = 5/3. A top
  # set AT the plateau cuts it off exactly and makes it read as an edge spike -- that
  # is how it was got wrong twice -- so the top clears it by a full index. The much
  # larger spikes right at the cut-off are the edge singularity and do clip.
  # the top is DERIVED from the plateau, never hard-coded: set at its own value it
  # clips it into what looks like an edge spike, which has happened twice
  axS.set_ylim(-(p+2.6), float(fr) + 1.)
  axS.set_xlabel(GMA_LABEL, fontsize=FS_LAB)
  axS.set_ylabel("${\\rm d}\\ln({\\rm d}N_{\\rm e}/{\\rm d}\\gamma_{\\rm e})"
                 "/{\\rm d}\\ln\\gamma_{\\rm e}$", fontsize=FS_LAB)
  axR = axS.twinx()                       # right-hand spine carries the expected indices
  axR.set_ylim(axS.get_ylim())
  axR.set_yticks([lev for lev, _ in levels])
  axR.set_yticklabels([lab for _, lab in levels])
  axR.tick_params(axis='y', labelsize=FS_ANN, length=2.5, pad=1.5, colors=INK)
  axR.grid(False)

  for ax in (axN, axS):
    ax.axvspan(1e-12, GMA_FLOOR, color='crimson', alpha=.07, lw=0, zorder=0)
    ax.axvline(GMA_FLOOR, color='crimson', ls=':', lw=.9, zorder=1)
    ax.axvline(gm0, color=INK, ls=':', lw=.8, zorder=1)
    ax.axvline(gM0, color=INK, ls=':', lw=.8, zorder=1)
    ax.grid(alpha=.25, lw=.4)
    ax.tick_params(which='both', labelsize=FS_TICK)
  axN.set_xlim(.3*lo_edge, 4.*gM0)
  for v, lab in ((gm0, '$\\gamma_{\\mathrm{m},\\!\\mathrm{i}}$'),
                 (gM0, '$\\gamma_{\\mathrm{M},\\!\\mathrm{i}}$')):
    axN.annotate(lab, (v, .985), xycoords=('data', 'axes fraction'), color=INK,
                 fontsize=FS_ANN, ha='center', va='top',
                 bbox=dict(fc='w', ec='none', alpha=.85, pad=1.))
  axS.annotate('$\\gamma=1$', (GMA_FLOOR, .10), xycoords=('data', 'axes fraction'),
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
  d, s = _exps(A_RHO, Q_C)
  print(f"hydro : simple coasting, rho' ~ R^{A_RHO:g};  clock: t'_c ~ R^{Q_C:g}"
        f"{'  (constant B'+chr(39)+'/t'+chr(39)+'_c -- the cooling_shape footing)' if Q_C == 0 else ''}")
  print(f'        d=-a_rho/3={d:+.5f}  s=d+q-1={s:+.5f}  -> burn '
        f'{f"SATURATES at S={1./s:.3f} tt_dyn" if s > 0 else "grows without bound"}')
  A = float(A_of_sigma(SIGMA_END, A_RHO))
  print(f'over sigma={SIGMA_END:g} t_dyn: A={A:.4g} (adiabatic drag x{1./A:.1f}), '
        f'tt_end/tt_dyn={float(tt_of_sigma(SIGMA_END, 1.)):.4g}, '
        f'S/tt_dyn={float(S_of_sigma(SIGMA_END, 1.)):.4g}')

  # what the OTHER field index would do -- the prescription this module used to claim
  for qq in (0., 2.):
    al, ee = _exps(A_RHO, qq)
    print(f'   q={qq:g}: tt_end/tt_dyn={float(tt_of_sigma(SIGMA_END, 1., qq)):9.4g}   '
          f'S/tt_dyn={float(S_of_sigma(SIGMA_END, 1., A_RHO, qq)):9.4g}   '
          f'{"SATURATED" if ee > 0 else "unbounded"}')

  dev, n = check_reduces_to_syn()
  print(f'reduces to syn : max rel dev over {n} points at a_rho->0, q=0 = {dev:.2e}')
  print(f'trajectory ODE : max rel dev at sigma={SIGMA_END:g} = {check_trajectory():.2e}')
  s1, s2 = check_sum_rule()
  print(f'sum rule       : {s1:.2e} at Ng, {s2:.2e} at 2Ng (ratio {s1/s2:.2f}: '
        'first order, so it is the trapezoid over the support edges, not the model)')
  df, nf, ds, ns = check_bottom_edge()
  drop, ee = edge_drop_factor()
  lim = f'-> {ee:.3f} as sigma -> inf' if ee is not None else \
        '-> 0 as sigma -> inf: the burn saturates, so there is no finite limit'
  print(f'bottom edge    : fast limit A/S  dev {df:.2e} ({nf} regimes); '
        f'slow limit A*gma_m0 dev {ds:.2e} ({ns})')
  print(f'               : edge drops x{drop:.3f} below the synchrotron-only one at the '
        f'same tt ({lim})')
  print(f'deep-tail ratio N_adiab/N_syn (-> 1 near gma_m'
        + (f', -> {ee:.4f} deep' if ee is not None else '') + '):')
  for g, r in check_deep_tail_ratio():
    print(f'    gma={g:10.3e}: {r:.4f}')
  print(f'index is ~ -2 for ANY a_rho at this q ({Q_C:g}):')
  for a_rho, ee2, idx in check_index_is_robust():
    print(f'    a_rho={a_rho:+.3f}  s={ee2:.4f}  ->  index {idx:+.4f}')
  print("the field index q matters only once the burn has had TIME to saturate:")
  for se in sorted({1., 1e3, SIGMA_END}):   # always show the short/long contrast
    row = '   '.join(f'q={qq:.1f} {measure_low_slope(-3., se, q=qq)[1]:+.3f}'
                     for qq in (0., 1., 2.))
    print(f'    sigma_end={se:>7g} t_dyn:  {row}')
  print('bottom edge, and the local index just above it, per regime:')
  for lc in LOGC_SAMPLES:
    ttd = tt_dyn_of_C(lc, GM0)
    lo = gamma_cooled(SIGMA_END, GM0, ttd, A_RHO, Q_C)
    g, sl = measure_low_slope(lc)
    print(f'  log10 C = {lc:+.0f}: gma_m({SIGMA_END:g} t_dyn) = {lo:10.3e}'
          f'{"  (BELOW the gma=1 floor)" if lo < GMA_FLOOR else "":26s}'
          f'  index 1.2-3x above it: {sl:+.3f}')
  plot_integrated_adiab(show=show)


if __name__ == '__main__':
  main()
