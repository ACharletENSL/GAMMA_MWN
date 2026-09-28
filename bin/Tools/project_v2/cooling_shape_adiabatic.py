# -*- coding: utf-8 -*-
# @Author: acharlet

'''
The instantaneous distribution under synchrotron AND adiabatic cooling: the physics,
not a figure.

This module used to draw cooling_shape_adiabatic.png. That figure is retired --
cooling_shape_panels now draws the whole family, three panels per quantity -- so what is
left here is the adiabatic half of the shape problem and its checks, imported by
cooling_shape_panels. The hydrodynamic constants and the trajectory live one level down,
in cooling_integrated_adiabatic (A_RHO, GMA_AD, Q_C, _exps, A_of_sigma, S_of_sigma,
gamma_cooled), so nothing is restated here and the two cannot drift apart.

THE SOLUTION. The cooling equation dgma/dtt = (dlnA/dtt) gma - gma^2 is linear in
u = 1/gma and integrates with A = (rho'/rho'_i)^(1/3) as the integrating factor:

    gma(sigma) = A gma_0/(1 + gma_0 S),    S = int_0^tt A dtt',    sigma = t'/t'_dyn,

and number conservation turns the injected power law into

    dN_e/dgma_e = K0 A gma^-p (A - S gma)^(p-2)   on [gma_m(sigma), gma_M(sigma)].

THE EXPONENTS ARE POSITIVE, the signs carried by the definitions (_exps). With
tau = 1 + sigma,

    A = tau^-d    d = -a_rho/3 = 2/3      t'_c = t'_c,i tau^q    q = -a_rho*gma_ad = 10/3
    s = d + q - 1 = 3                     S = tt_dyn (1 - tau^-s)/s

so s > 0 saturates the burn at S_inf = tt_dyn/s. Three consequences, each checked:

  1. WHICH REGIMES CAN COOL. The bottom edge needs S > 1/gma_m,i, so only C < 1/s = 1/3
     reaches fast cooling. C = 1 and above never do, however long you wait.
  2. THE FINAL WIDTH is closed-form. gma_M/gma_m depends on S alone (the A cancels), so
     once S freezes so does the width, at

         gma_M/gma_m -> 1 + s C

     -- check_width_law holds it to 2e-16. Mono-energetic needs s*C << 1, the SAME
     threshold as (1). At C = 1e-2 it is 1.03; at C = 1e2 it stalls at 301.
  3. THE CLOCK CRAWLS. tt itself saturates at tt_dyn/(q-1), and tt_eff = S/A grows only
     as tau^d, so late tt_eff is asymptotic behaviour, not a time a shell reaches.

THE GENERALISED NORMALISED TIME. cooling_shape_figure plots against tt = int dt'/t'_c,
in which the synchrotron solution is gma = gma_0/(1 + gma_0 tt) and 1/tt is the burn-off
asymptote. Plotting the adiabatic problem against tt (or against sigma) loses that. The
variable that keeps it is

    tt_eff = (1/A) int_{t'_i}^{t'} A/t'_c ds  =  S/A,

because u = (u_0 + S)/A can be rewritten 1/gma = 1/(A gma_0) + tt_eff: an electron
injected with gma_0 -> inf therefore sits at gma = 1/tt_eff, so 1/tt_eff IS the asymptote
the top edge slides down, exactly as 1/tt is without expansion. It is strictly increasing
(S grows while A falls), so it is a valid abscissa, and tt_eff -> tt as A -> 1.

cooling_shape_panels does NOT use it: its abscissa is the physical t'/t'_c,i, in which
gma_M is C-dependent and each regime needs its own panel. tt_eff is kept here because it
is what makes the adiabatic and the synchrotron problem the same problem.

ADIABATIC COOLING ADDS NO NEW SHAPE. Substituting x = gma/A collapses the distribution to

    N_adiab(gma, sigma) = (1/A) * N_syn(gma/A, S),

an exact SIMILARITY transform of the synchrotron-only solution -- slide the whole
distribution down the gamma axis by A, up in amplitude by 1/A (so int N dgma is
conserved), and relabel the clock tt -> S. Nothing bends, nothing breaks, no segment is
created or destroyed; check_similarity verifies it to ~1e-9. Two things follow:

  - the WIDTH evolution is untouched, gma_M/gma_m = (gma_M,i/gma_m,i)(1 + gma_m,i S)
    /(1 + gma_M,i S), which is the synchrotron collapse law with tt -> S. Since S grows
    more slowly than tt, expansion DELAYS the collapse when it is clocked in real time;
  - and the time-integrated distribution cannot acquire a new index from adiabatic
    cooling either, which is why cooling_integrated_adiabatic's change is a new
    LOW-ENERGY segment -- the dwell time once the burn freezes -- and not a new tail.

Run:  python cooling_shape_adiabatic.py     -- prints the checks, draws nothing
'''

import os
import numpy as np
import matplotlib.pyplot as plt

from cooling_distribution import norm_plaw_distrib, distrib_plaw_cooled
from scipy.optimize import brentq
from cooling_shape_figure import P_SYN, GM0, GMA_M0, NG, cooled_distrib
from cooling_integrated_figure import tt_dyn_of_C
from cooling_integrated_adiabatic import (A_RHO, Q_C, _exps, A_of_sigma, S_of_sigma,
    gamma_cooled)

# --- defaults -------------------------------------------------------------------------
LOGC = -3.                  # the reference C for the tt <-> sigma map and the checks
                            # below. NOT the marginal case: with the burn saturating the
                            # bottom edge cools only for C < 1/s, so C = 1 never
                            # reaches fast cooling however long you wait.
# the sampled log10 t'/t'_c,i the checks are run at -- the same values cooling_shape_
# panels draws, so the numbers main() prints are the ones on the figures
LOGT_SAMPLES = (-8., -7., -6., -5., -4., -3., -2., -1., 0.)


def lab_C(lc):
  'Render a regime label: 0 -> 1, otherwise 10^n.'
  return '1' if lc == 0. else f'10^{{{lc:.0f}}}'


# --- the generalised normalised time ----------------------------------------------------
def tt_eff_of_sigma(sigma, ttd, a_rho=A_RHO, q=Q_C):
  '''
  The GENERALISED normalised time of the module docstring,

      tt_eff = (1/A(t')) int_{t'_i}^{t'} A(s)/t'_c(s) ds  =  S/A,

  the variable in which the adiabatic problem carries the synchrotron figure's 1/tt
  asymptote. Strictly increasing (S grows while A falls), so it is a valid abscissa.
  '''
  return S_of_sigma(sigma, ttd, a_rho, q)/A_of_sigma(sigma, a_rho)


def sigma_of_tt_eff(tt_eff, ttd, a_rho=A_RHO, q=Q_C, br=(-16., 40.)):
  '''
  Inverse of tt_eff_of_sigma -- monotone, so one bracketed root in log10 sigma.
  The bracket has to be WIDE. At q > 1/2 the clock crawls: tt_eff grows only as
  sigma^|alpha|, so reaching tt_eff = 1e2 on the slowest-cooling C needs sigma
  ~ 1e20. A bracket of 1e9, enough at q = 0, raises 'f(a) and f(b) must have
  different signs' there.
  '''
  g = lambda ls: float(tt_eff_of_sigma(10.**ls, ttd, a_rho, q)) - tt_eff
  return 10.**brentq(g, br[0], br[1], xtol=1e-13, rtol=8.9e-16)


# --- the distribution -----------------------------------------------------------------
def cooled_distrib_adiab(sigma, ttd, p=P_SYN, gm0=GM0, gM0=GMA_M0, a_rho=A_RHO, q=Q_C,
    Ng=NG):
  '''
  (gma, N(gma,sigma)) sampled over the SUPPORT [gma_m(sigma), gma_M(sigma)] only --
  as in cooling_shape_figure, the expression stays positive above gma_M but those
  electrons do not exist, nothing having been injected above gma_M0.
  '''
  A, S = float(A_of_sigma(sigma, a_rho)), float(S_of_sigma(sigma, ttd, a_rho, q))
  gm, gM = (gamma_cooled(sigma, g, ttd, a_rho, q) for g in (gm0, gM0))
  gma = np.geomspace(gm, gM, Ng)
  return gma, norm_plaw_distrib(gm0, gM0, p)*A*gma**-p*np.abs(A - S*gma)**(p-2.)


def width(sigma, ttd, gm0=GM0, gM0=GMA_M0, a_rho=A_RHO, q=Q_C):
  'gma_M/gma_m at sigma -- the A cancels, so this is the synchrotron law clocked in S.'
  return (gamma_cooled(sigma, gM0, ttd, a_rho, q)
          /gamma_cooled(sigma, gm0, ttd, a_rho, q))


# the sigma values LOGT_SAMPLES maps to at the module defaults: what the figure draws,
# and what the checks below are run at. Nine bracketed root-finds, done once at import.
SIG_SAMPLES = tuple(sigma_of_tt_eff(10.**l, tt_dyn_of_C(LOGC, GM0))
                    for l in LOGT_SAMPLES)


# --- validation -------------------------------------------------------------------------
def check_number_conservation(sigmas=SIG_SAMPLES, logC=LOGC, p=P_SYN, gm0=GM0,
    gM0=GMA_M0, a_rho=A_RHO, q=Q_C, Ng=40000):
  '''
  int N dgma must stay 1 at every sigma: cooling moves electrons, adiabatic or not.
  Returns the largest relative deviation.
  '''
  ttd = tt_dyn_of_C(logC, gm0)
  dev = 0.
  for sigma in sigmas:
    gma, N = cooled_distrib_adiab(sigma, ttd, p, gm0, gM0, a_rho, q, Ng)
    dev = max(dev, abs(np.trapezoid(N, gma) - 1.))
  return dev


def check_similarity(sigmas=SIG_SAMPLES, logC=LOGC, p=P_SYN, gm0=GM0, gM0=GMA_M0,
    a_rho=A_RHO, q=Q_C, n=40):
  '''
  THE claim of this module: N_adiab(gma,sigma) = (1/A) N_syn(gma/A, S) exactly. Evaluated
  against the pipeline's own distrib_plaw_cooled on the rescaled axis. Returns the
  largest relative deviation over the sampled times and the support.
  '''
  ttd = tt_dyn_of_C(logC, gm0)
  K0 = norm_plaw_distrib(gm0, gM0, p)
  dev = 0.
  for sigma in sigmas:
    A, S = float(A_of_sigma(sigma, a_rho)), float(S_of_sigma(sigma, ttd, a_rho, q))
    gma, N = cooled_distrib_adiab(sigma, ttd, p, gm0, gM0, a_rho, q, n)
    ref = K0*distrib_plaw_cooled(gma/A, p, S)/A          # the synchrotron shape, slid
    ok = ref > 0.
    dev = max(dev, float(np.max(np.abs(N[ok]/ref[ok] - 1.))))
  return dev


def check_width_law(sigmas=SIG_SAMPLES, logC=LOGC, gm0=GM0, gM0=GMA_M0, a_rho=A_RHO,
    q=Q_C):
  '''
  The width must obey cooling_shape_figure's collapse law with tt -> S, the A having
  cancelled. Returns the largest relative deviation.
  '''
  ttd = tt_dyn_of_C(logC, gm0)
  dev = 0.
  for sigma in sigmas:
    S = float(S_of_sigma(sigma, ttd, a_rho, q))
    pred = (gM0/gm0)*(1. + gm0*S)/(1. + gM0*S)
    dev = max(dev, abs(width(sigma, ttd, gm0, gM0, a_rho, q)/pred - 1.))
  return dev


def check_reduces_to_shape(sigmas=SIG_SAMPLES, logC=LOGC, p=P_SYN, gm0=GM0,
    gM0=GMA_M0, n=40):
  '''
  At a_rho -> 0 (no expansion) A = 1 and S = tt, so this must reproduce
  cooling_shape_figure.cooled_distrib. The tie to the already-validated module.
  '''
  ttd = tt_dyn_of_C(logC, gm0)
  dev = 0.
  for sigma in sigmas:
    g_a, N_a = cooled_distrib_adiab(sigma, ttd, p, gm0, gM0, a_rho=1e-12, q=0., Ng=n)
    _, N_s = cooled_distrib(ttd*sigma, p, gm0, gM0, Ng=n)
    dev = max(dev, float(np.max(np.abs(N_a/N_s - 1.))))
  return dev


def main():
  d, s = _exps(A_RHO, Q_C)
  ttd = tt_dyn_of_C(LOGC, GM0)
  print(f"hydro: coasting rho' ~ R^{A_RHO:g}, t'_c ~ R^{Q_C:g};  "
        f'd={d:+.4f}  s={s:+.4f}')
  print(f'clock: log10 C = {LOGC:.0f} -> tt_dyn = {ttd:.3e}, so sigma = tt/tt_dyn')
  print(f'similarity      : max |A*N_adiab(A x) / N_syn(x,S) - 1| = '
        f'{check_similarity():.2e}')
  print(f'reduces to shape: max rel dev at a_rho->0 = {check_reduces_to_shape():.2e}')
  print(f'number conserved: max |int N dgma - 1| = {check_number_conservation():.2e}')
  print(f'width law (S)   : max rel dev = {check_width_law():.2e}')
  print('per sampled time:')
  for ls, sigma in zip(LOGT_SAMPLES, SIG_SAMPLES):
    A = float(A_of_sigma(sigma, A_RHO))
    S = float(S_of_sigma(sigma, ttd, A_RHO, Q_C))
    gm = float(gamma_cooled(sigma, GM0, ttd, A_RHO, Q_C))
    gM = float(gamma_cooled(sigma, GMA_M0, ttd, A_RHO, Q_C))
    print(f'  log10 tt_eff = {ls:+.0f}: sigma={sigma:9.3e}  A={A:9.3e}  '
          f'S={S:9.3e}  '
          f'gma_m={gm:9.3e}  gma_M={gM:9.3e}  width={gM/gm:8.4f}'
          f'{"   (under the gma=1 floor)" if gM < 1. else ""}')


if __name__ == '__main__':
  main()
