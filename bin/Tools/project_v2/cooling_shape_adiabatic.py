# -*- coding: utf-8 -*-
# @Author: acharlet

'''
How a power-law electron distribution is reshaped by synchrotron AND adiabatic cooling.

APPENDIX ILLUSTRATION, the fourth corner of the family that shares figures/
cooling_distributions:

                          synchrotron only          + adiabatic
    instantaneous N     cooling_shape_figure      THIS MODULE
    time-integrated     cooling_integrated_figure cooling_integrated_adiabatic

Same hydrodynamics as cooling_integrated_adiabatic: the SHELL FIDUCIAL, rho' ~ R^-1.2
and B' ~ R^-1. The field decays too now, so unlike cooling_shape_figure this is not a
constant-t'_c picture, and the consequences are large -- see THE BURN SATURATES there. The physics functions are imported from
that module rather than restated, so the two adiabatic figures cannot drift apart.

THE SCALINGS, IN ONE PLACE. Three are put in, one combination controls everything.

  PUT IN, all against radius, with R/R_0 = 1 + sigma and sigma = t'/t'_dyn (coasting):
      rho' ~ R^-1.2      a_rho = -1.2   ->  A = (rho/rho_0)^(1/3) = (1+sigma)^alpha,
                                            alpha = a_rho/3 = -0.4
      B'   ~ R^-1        q = 1          ->  t'_c ~ 1/B'^2 ~ R^2q, so the CLOCK stretches:
                                            dtt = dt'/t'_c ~ (1+sigma)^-2q dsigma

  THE ONE COMBINATION. S = int A dtt has integrand (1+sigma)^(alpha-2q), so everything
  turns on

      e = alpha - 2q + 1 = -1.4,     S(sigma) = tt_dyn [(1+sigma)^e - 1]/e.

  e > 0 and the burn grows without bound; e < 0 and it SATURATES at S_inf = tt_dyn/|e|.
  At the shell fiducial it saturates. Three consequences, each measured:

    1. WHICH REGIMES CAN COOL. The bottom edge needs S > 1/gma_m0, so only
       C < 1/|e| = 0.714 ever reaches fast cooling. C = 1 and above never do, however
       long you wait -- which is why LOGC is -3 and not the marginal case.
    2. THE FINAL WIDTH is closed-form. gma_M/gma_m depends on S alone (the A cancels),
       so once S freezes so does the width, at
           gma_M/gma_m -> 1 + |e| C
       -- exact to six figures over C = 1e-3 .. 10. Mono-energetic needs |e| C << 1,
       the SAME threshold as (1). At C = 1e-3 it is 1.0014; at C = 1 it stalls at 2.40.
    3. THE CLOCK CRAWLS. tt itself saturates at tt_dyn, and tt_eff = S/A grows only as
       (1+sigma)^|alpha|, so tt_eff = 1e2 is sigma ~ 2e5 at the reference C. The top
       panel's right-hand end is asymptotic behaviour, not a time a shell reaches.

  THE TWO ASYMPTOTES on the top panel follow from the same algebra. gma_M -> 1/tt_eff
  always. For the adiabatic track, substitute S and A into tt_eff = S/A:

      tt_eff = (ttd/e) [ (1+sigma)^(1-2q) - (1+sigma)^|alpha| ]

  -- two competing powers, and the larger wins. Write m for it. Since e > 0 is the same
  statement as 1-2q > |alpha|, m = max(1-2q, |alpha|) is not a trick: it is just asking
  WHICH MECHANISM is advancing the clock.

      e > 0, burn-driven.  S still grows, tt_eff rides it, m = 1-2q, and inverting gives
                           A ~ tt_eff^(alpha/(1-2q)) -- at q = 0 that is alpha = a_rho/3.
      e < 0, drag-driven.  S has frozen at S_inf, so tt_eff = S_inf/A and therefore
                           A = S_inf/tt_eff IDENTICALLY. Slope -1, for any a_rho, and
                           not as an asymptote -- it is the definition of tt_eff once S
                           is constant. Verified: A*tt_eff/S_inf = 0.9984, 0.999997, 1.0
                           at sigma = 1e2, 1e4, 1e6.

  The shell fiducial is the second case, so both guides are tt_eff^-1 and they are
  labelled by mechanism rather than by index. They differ only in normalisation --
  1/tt_eff for gma_M against S_inf/tt_eff for the drag.

THE SOLUTION. The cooling equation dgma/dtt = (dlnA/dtt) gma - gma^2 is linear in
u = 1/gma and integrates with A = (rho/rho_0)^(1/3) as the integrating factor:

    gma(sigma) = A gma_0/(1 + gma_0 S),    S = int_0^tt A dtt',   sigma = t'/t'_dyn,

and number conservation turns the injected power law into

    N(gma,sigma) = K0 A gma^-p (A - S gma)^(p-2)   on [gma_m(sigma), gma_M(sigma)].

THE TIME AXIS IS A GENERALISED NORMALISED TIME. cooling_shape_figure plots against
tt = int dt'/t'_c, in which the synchrotron solution is gma = gma_0/(1 + gma_0 tt) and
1/tt is the burn-off asymptote. Plotting the adiabatic problem against tt (or against
sigma) loses that. The variable that keeps it is

    tt_eff = (1/A(t')) int_{t'_i}^{t'} A(s)/t'_c(s) ds  =  S/A,

because u = (u_0 + S)/A can be rewritten 1/gma = 1/(A gma_0) + tt_eff: an electron
injected with gma_0 -> inf therefore sits at gma = 1/tt_eff, so 1/tt_eff IS the asymptote
the top edge slides down, exactly as 1/tt is in the synchrotron figure. Measured,
gma_M * tt_eff = 1.000 at every sample once the edge has burnt.

The ADIABATIC track has an asymptote of its own on the same axis, and the panel draws
it: the same identity tt_eff = (ttd/e)[(1+sigma) - (1+sigma)^|alpha|] leaves
(1+sigma) -> e*tt_eff/ttd for any a_rho > -3, so

    A -> (|e| tt_eff/ttd)^(alpha/m),   m = max(1-2q, |alpha|),

which is slope alpha = a_rho/3 at q = 0 but alpha/|alpha| = -1 once q > 1/2 -- the SAME
slope as the synchrotron asymptote, differing only in normalisation. At the shell
fiducial (q = 1) that is the case, which is why the panel carries two parallel guides
labelled 'syn.' and 'adiab.' rather than two different indices. Measured local slopes converge on alpha for every law tried
(a_rho = -0.5, -1.205, -2, -2.9 give -0.167, -0.402, -0.665, -0.914 by tt_eff = 1e4
against -0.167, -0.402, -0.667, -0.967 predicted). TWO WAYS it is unlike tt_eff^-1:
it converges far more slowly, because the dropped term dies only as (1+sigma)^(|alpha|-1)
-- over the drawn range the coasting track reads about -0.63, not -0.667 -- and it is
C-DEPENDENT, its normalisation carrying tt_dyn, so it is a guide for this panel's
reference C and not a universal line the way tt_eff^-1 is.

The guide is normalised on gma_m0 and the SLOWEST-cooling C in LOGC_M, because that is
the curve pure adiabatic drag describes: gma_m = A gma_m0/(1 + gma_m0 A tt_eff), and at
large C the burn term stays small, so gma_m ~ A gma_m0. Measured against log10 C = +3 the
guide is within 20% over tt_eff = 1e-4 .. 1 and sits on the curve to the eye. It leaves
at both ends -- early the curve has not entered the asymptotic regime (A is still ~1 and
gma_m is still gma_m0), late the burn term grows as tt_eff^e and turns the curve over
toward tt_eff^-1. So the -2/3 stretch is a TRANSIENT, not the end state.

Three properties make the pair directly comparable:
  - tt_eff is strictly increasing (S grows while A falls), so it is a valid abscissa;
  - tt_eff -> tt as A -> 1, agreeing to 3e-8 at sigma = 1e-8, so the two figures share
    one clock at early times and the panels can be laid side by side;
  - the samples are therefore the SAME log10 values cooling_shape_figure uses, and the
    last of them, tt_eff = 1, lands exactly on gma_M = 1 -- the same end-of-validity
    coincidence the synchrotron figure has, and for the same reason.
The knees keep the sibling's names: tt_M and tt_m are where S reaches 1/gma_M0 and
1/gma_m0. In tt_eff they sit a little RIGHT of 1/gma_M0 and 1/gma_m0, by whatever 1/A
has grown to by then -- tt_m at 1.8e-3 rather than 1e-3.

THE POINT OF THE FIGURE: ADIABATIC COOLING ADDS NO NEW SHAPE. Substituting x = gma/A
collapses the expression above to

    N_adiab(gma, sigma) = (1/A) * N_syn(gma/A, S),

an exact SIMILARITY transform of the synchrotron-only solution -- slide the whole
distribution down the gamma axis by A, up in amplitude by 1/A (so int N dgma stays 1),
and relabel the clock tt -> S. Nothing bends, nothing breaks, no segment is created or
destroyed. check_similarity verifies it to ~1e-14.

Three consequences worth reading off the panels:

  - The WIDTH evolution is untouched. A cancels out of the ratio,
    gma_M/gma_m = (gma_M0/gma_m0)(1 + gma_m0 S)/(1 + gma_M0 S), which is exactly
    cooling_shape_figure's collapse law with tt -> S. So the population still ends up
    MONO-ENERGETIC, on the same schedule measured in S -- just at a lower gamma.
    check_width_law.
  - The two knees still sit where the edges burn, but in S, not in tt: the top edge
    starts burning at S = 1/gma_M0 and the power law is gone by S = 1/gma_m0. Since
    S grows more slowly than tt (S/tt_dyn = 27.0 after 1e3 t_dyn, against tt/tt_dyn =
    1e3), expansion DELAYS the collapse when it is clocked in real time.
  - And it explains the companion figure: if the instantaneous shape is only ever slid,
    the time-integrated one cannot acquire a new index from adiabatic cooling either.
    That is why cooling_integrated_adiabatic finds the tail still at ~ -2.

WHY THIS FIGURE NEEDS A PARAMETER THE SYNCHROTRON ONE DID NOT. cooling_shape_figure is
a one-parameter family: everything is a function of tt alone, because tt = int dt'/t_c1
absorbs the field history. Here A is a function of RADIUS, so the tt <-> sigma map has to
be fixed, and that map carries tt_dyn = 1/gma_c. The figure therefore has to choose a
cooling regime; LOGC = 0 (the marginal case gma_c = gma_m) is the default and is stated
inside panel (a). Other regimes only stretch the sigma axis, they do not change the
shapes -- which is the similarity result again.

THE COOLING REGIMES ARE NOT DRAWN HERE any more. They are defined by synchrotron
cooling, and on this abscissa they would be C-dependent: the saturating burn moves t_m,
and for C above 1/|e| there is no t_m at all. They live on cooling_shape_figure's
pure-synchrotron panel instead, where t_M = 1/gma_M,0 and t_m = 1/gma_m,0 need no
reference C.

THE ABSCISSA IS t'/t'_c,i -- the comoving time in units of the cooling time at injection,
which is just sigma*tt_dyn (sigma = t'/t'_dyn, tt_dyn = t'_dyn/t'_c,i). It is NOT
tt = int dt'/t'_c, which lags it once the field decays, and not the tt_eff = S/A this
figure used before. Both columns share one t'/t'_c,i range, so they can be read across.
The cost, and the reason for the 2x2: gma_M is C-independent in tt_eff but not here, so
each regime needs both of its edges and they cannot share a panel.

VALIDITY. As everywhere in this family, gma_synCooled's ultra-relativistic trajectory is
not physical below gma = 1 (marked in crimson): the real loss rate goes as gma^2 - 1 and
electrons stall near 1 instead of continuing down. With the adiabatic drag the population
gets there SOONER. The last sampled time, tt_eff = 1, puts gma_M exactly on the floor
(and gma_m just under it at 0.95), so as in the synchrotron figure that sample IS the end
of the model's validity, not a time it can be pushed past.

Run:  python cooling_shape_adiabatic.py
'''

import os
import numpy as np
import matplotlib.pyplot as plt

from cooling_distribution import norm_plaw_distrib, distrib_plaw_cooled
from scipy.optimize import brentq
from cooling_shape_figure import (P_SYN, GM0, GMA_M0, NG, OUTDIR, INK, MUTED, FIGSIZE,
    FS_LAB, FS_TICK, FS_ANN, FS_LEG, LOGTT_SAMPLES, cooled_distrib)
from cooling_integrated_figure import tt_dyn_of_C
from cooling_integrated_adiabatic import (A_RHO, Q_B, _exps, A_of_sigma, S_of_sigma,
    gamma_cooled)

# --- defaults -------------------------------------------------------------------------
LOGC = -3.                  # gma_c/gma_m for the tt <-> sigma map, the bands, the knees
                            # and panel (b). NOT the marginal case any more: with the
                            # burn saturating, the bottom edge cools only for
                            # C < 1/|e| = 0.714, so at C = 1 there is no tt_m and the
                            # panel showed VSC + SC alone. This is an ILLUSTRATIVE
                            # figure, so the reference sits on a regime that does cool --
                            # and on one of the two curves LOGC_M draws, so the tt_m
                            # vertical passes through a knee that is on the plot.
# sampled log10 tt_eff, out to 1e2 so the fast regime is seen to cool right through and
# collapse. NOTE the physical cost: with B' ~ R^-1 the clock crawls (tt_eff grows only
# as sigma^|alpha|), so the right-hand end is sigma ~ 7e12 dynamical times at the
# reference C. The late panel is asymptotic behaviour, not a time the shell reaches.
LOGT_SAMPLES = (-8., -7., -6., -5., -4., -3., -2., -1., 0.)
T_LIM = (1e-11, 1e2)        # t'/t'_c,i span of the track panels
TTE_LIM = (1e-11, 1e2)      # the TOP panel runs two decades further than the samples:
                            # the tracks' asymptotes only declare themselves past
                            # tt_eff ~ 1, while the distributions are already a
                            # collapsed spike by then and add nothing below
LOGC_M = (-3., 3.)          # log10(bar{gma}_c/gma_m): ONE COLUMN PER VALUE. Both edges
                            # are drawn per regime now -- gma_M is C-independent only in
                            # tt_eff, not on this abscissa -- so they cannot share panels
FNAME = 'cooling_shape_adiabatic.png'


def lab_C(lc):
  'Render a regime label: 0 -> 1, otherwise 10^n.'
  return '1' if lc == 0. else f'10^{{{lc:.0f}}}'


# --- the generalised normalised time ----------------------------------------------------
def tt_eff_of_sigma(sigma, ttd, a_rho=A_RHO, q=Q_B):
  '''
  The GENERALISED normalised time of the module docstring,

      tt_eff = (1/A(t')) int_{t'_i}^{t'} A(s)/t'_c(s) ds  =  S/A,

  the variable in which the adiabatic problem carries the synchrotron figure's 1/tt
  asymptote. Strictly increasing (S grows while A falls), so it is a valid abscissa.
  '''
  return S_of_sigma(sigma, ttd, a_rho, q)/A_of_sigma(sigma, a_rho)


def sigma_of_tt_eff(tt_eff, ttd, a_rho=A_RHO, q=Q_B, br=(-16., 40.)):
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
def cooled_distrib_adiab(sigma, ttd, p=P_SYN, gm0=GM0, gM0=GMA_M0, a_rho=A_RHO, q=Q_B,
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


def width(sigma, ttd, gm0=GM0, gM0=GMA_M0, a_rho=A_RHO, q=Q_B):
  'gma_M/gma_m at sigma -- the A cancels, so this is the synchrotron law clocked in S.'
  return (gamma_cooled(sigma, gM0, ttd, a_rho, q)
          /gamma_cooled(sigma, gm0, ttd, a_rho, q))


# the sigma values LOGT_SAMPLES maps to at the module defaults: what the figure draws,
# and what the checks below are run at. Nine bracketed root-finds, done once at import.
SIG_SAMPLES = tuple(sigma_of_tt_eff(10.**l, tt_dyn_of_C(LOGC, GM0))
                    for l in LOGT_SAMPLES)


# --- validation -------------------------------------------------------------------------
def check_number_conservation(sigmas=SIG_SAMPLES, logC=LOGC, p=P_SYN, gm0=GM0,
    gM0=GMA_M0, a_rho=A_RHO, q=Q_B, Ng=40000):
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
    a_rho=A_RHO, q=Q_B, n=40):
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
    q=Q_B):
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


# --- the figure ---------------------------------------------------------------------------
def plot_cooling_shape_adiab(logC=LOGC, p=P_SYN, gm0=GM0, gM0=GMA_M0, a_rho=A_RHO,
    q=Q_B, logtte=LOGT_SAMPLES, logC_m=LOGC_M, outdir=OUTDIR, fname=FNAME, show=False):
  """
  A 2x2 grid: ONE COLUMN PER REGIME, tracks above and distributions below. Overlaying the
  two regimes on shared panels was unreadable once both edges had to be drawn per regime
  -- which is forced by the abscissa (below) -- so they are separated instead.

  THE ABSCISSA IS t'/t'_c,i, the physical comoving time in units of the cooling time at
  injection. That is simply sigma*tt_dyn: sigma = t'/t'_dyn and tt_dyn = t'_dyn/t'_c,i.
  It is NOT tt = int dt'/t'_c, which lags it once the field decays, and it is not the
  tt_eff = S/A this figure used before. The cost, and the reason for the split: gma_M is
  C-independent in tt_eff but NOT here, so each regime needs both its edges.
  """
  tte_arr = 10.**np.asarray(logtte, dtype=float)
  colors = plt.cm.viridis(np.linspace(0., .85, len(tte_arr)))
  K0 = norm_plaw_distrib(gm0, gM0, p)
  nC = len(logC_m)

  fig, axs = plt.subplots(2, nC, figsize=(3.2*nC, 5.4), squeeze=False,
      gridspec_kw=dict(height_ratios=[1., 1.35], hspace=.30, wspace=.28))

  for k, lc in enumerate(logC_m):
    axT, axD = axs[0, k], axs[1, k]
    ttd_c = tt_dyn_of_C(lc, gm0)
    x = np.geomspace(*T_LIM, 900)                  # t'/t'_c,i, SHARED by both columns
    sg = x/ttd_c                                   # sigma = (t'/t'_c,i)/tt_dyn
    # both edges now, the abscissa having cost gma_M its C-independence
    axT.loglog(x, gamma_cooled(sg, gM0, ttd_c, a_rho, q), color='k', lw=1.4,
               label='$\\gamma_\\mathrm{M}$')
    axT.loglog(x, gamma_cooled(sg, gm0, ttd_c, a_rho, q), color='k', lw=1.1, ls='--',
               label='$\\gamma_\\mathrm{m}$')
    axT.axhspan(1e-30, 1., color='crimson', alpha=.07, lw=0, zorder=0)
    axT.axhline(1., color='crimson', ls=':', lw=.9, zorder=1)
    axT.set_xlim(x[0], x[-1])
    axT.set_ylim(.3*float(gamma_cooled(sg[-1], gm0, ttd_c, a_rho, q)), 30.*gM0)
    axT.set_yticks(10.**np.arange(-2., np.log10(gM0) + 1., 2.))
    axT.set_xlabel("$t'/t'_{\\rm c,i}$", fontsize=FS_LAB, labelpad=1.)
    axT.tick_params(axis='x', pad=1.5)
    axT.annotate(f'$\\bar{{\\gamma}}_{{\\rm c}}/\\gamma_{{\\rm m}}={lab_C(lc)}$',
                 (.5, 1.02), xycoords='axes fraction', color=INK, fontsize=FS_LAB,
                 ha='center', va='bottom')

    # (b) the distributions for THIS regime
    axD.loglog(*cooled_distrib_adiab(0., ttd_c, p, gm0, gM0, a_rho, q), color='k',
               lw=1.4, zorder=2)
    edges, lo_D = [], np.inf
    for ls, c in zip(logtte, colors):
      sg_b = 10.**ls/ttd_c
      gma, N = cooled_distrib_adiab(sg_b, ttd_c, p, gm0, gM0, a_rho, q)
      axD.loglog(gma, N, color=c, lw=1.2, solid_capstyle='round',
                 label=(f'{ls:.0f}' if k == 0 else None), zorder=3)
      lo_D = min(lo_D, float(gma[0])); edges.append((gma[-1], N[-1]))
    ed = np.array(edges)
    axD.plot(ed[:, 0], ed[:, 1], color=MUTED, lw=.7, zorder=4)
    gg = np.geomspace(gm0, gM0, 3)
    axD.loglog(gg, 12.*K0*gg**-p, color=MUTED, ls=':', lw=.9)
    for v, lab in ((gm0, '$\\gamma_{\\mathrm{m},\\!0}$'),
                   (gM0, '$\\gamma_{\\mathrm{M},\\!0}$')):
      axD.axvline(v, color=INK, ls=':', lw=.8, zorder=1)
      axD.annotate(lab, (v, .985), xycoords=('data', 'axes fraction'), color=INK,
                   fontsize=FS_ANN, ha='center', va='top',
                   bbox=dict(fc='w', ec='none', alpha=.85, pad=1.))
    axD.axvspan(1e-30, 1., color='crimson', alpha=.07, lw=0, zorder=0)
    axD.axvline(1., color='crimson', ls=':', lw=.9, zorder=1)
    axD.set_xlim(.5*lo_D, 2.*gM0)
    axD.set_ylim(1e-16, 1e4)
    axD.set_xlabel('$\\gamma$', fontsize=FS_LAB)
    if k == 0:
      axT.set_ylabel('$\\gamma$', fontsize=FS_LAB)
      axD.set_ylabel("$N(\\gamma,t')/N_{\\rm e}$", fontsize=FS_LAB)
      leg = axD.legend(fontsize=FS_LEG, ncol=2, loc='lower left', framealpha=.9,
                       title="$\\log_{10}(t'/t'_{\\rm c,i})$", handlelength=1.1,
                       labelspacing=.25, columnspacing=.9, handletextpad=.5, borderpad=.4)
      leg.get_title().set_fontsize(FS_LEG)
      axT.legend(fontsize=FS_LEG, loc='lower left', framealpha=.9, handlelength=1.4,
                 labelspacing=.25, handletextpad=.5, borderpad=.4)
    for ax in (axT, axD):
      ax.grid(alpha=.25, lw=.4)
      ax.tick_params(which='both', labelsize=FS_TICK)

  os.makedirs(outdir, exist_ok=True)
  path = os.path.join(outdir, fname)
  fig.savefig(path, dpi=300, bbox_inches='tight')
  print(f'saved {path}')
  if show:
    plt.show()
  return fig, axs


def main(show=False):
  alpha, e = _exps(A_RHO, Q_B)
  ttd = tt_dyn_of_C(LOGC, GM0)
  print(f"hydro: coasting rho' ~ R^{A_RHO:g}, B' ~ R^-{Q_B:g} (constant t'_c);  "
        f'alpha={alpha:+.4f}  e={e:+.4f}')
  print(f'clock: log10 C = {LOGC:.0f} -> tt_dyn = {ttd:.3e}, so sigma = tt/tt_dyn')
  print(f'similarity      : max |A*N_adiab(A x) / N_syn(x,S) - 1| = '
        f'{check_similarity():.2e}')
  print(f'reduces to shape: max rel dev at a_rho->0 = {check_reduces_to_shape():.2e}')
  print(f'number conserved: max |int N dgma - 1| = {check_number_conservation():.2e}')
  print(f'width law (S)   : max rel dev = {check_width_law():.2e}')
  print('per sampled time:')
  for ls, sigma in zip(LOGT_SAMPLES, SIG_SAMPLES):
    A = float(A_of_sigma(sigma, A_RHO))
    S = float(S_of_sigma(sigma, ttd, A_RHO, Q_B))
    gm = float(gamma_cooled(sigma, GM0, ttd, A_RHO, Q_B))
    gM = float(gamma_cooled(sigma, GMA_M0, ttd, A_RHO, Q_B))
    print(f'  log10 tt_eff = {ls:+.0f}: sigma={sigma:9.3e}  A={A:9.3e}  '
          f'S={S:9.3e}  '
          f'gma_m={gm:9.3e}  gma_M={gM:9.3e}  width={gM/gm:8.4f}'
          f'{"   (under the gma=1 floor)" if gM < 1. else ""}')
  plot_cooling_shape_adiab(show=show)


if __name__ == '__main__':
  main()
