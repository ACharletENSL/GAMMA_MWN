# -*- coding: utf-8 -*-
# @Author: acharlet

'''
How a power-law electron distribution is reshaped by synchrotron AND adiabatic cooling.

APPENDIX ILLUSTRATION, the fourth corner of the family that shares figures/
cooling_distributions:

                          synchrotron only          + adiabatic
    instantaneous N     cooling_shape_figure      THIS MODULE
    time-integrated     cooling_integrated_figure cooling_integrated_adiabatic

Same hydrodynamics and the same footing as cooling_integrated_adiabatic: simple coasting
(rho' ~ R^-2) with the comoving cooling time t'_c held CONSTANT, so the only thing added
to cooling_shape_figure is the adiabatic drag. The physics functions are imported from
that module rather than restated, so the two adiabatic figures cannot drift apart.

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

    A -> (e*tt_eff/ttd)^alpha,     slope alpha = a_rho/3  =  -2/3 for coasting,

against -1 for synchrotron. Measured local slopes converge on alpha for every law tried
(a_rho = -0.5, -1.205, -2, -2.9 give -0.167, -0.402, -0.665, -0.914 by tt_eff = 1e4
against -0.167, -0.402, -0.667, -0.967 predicted). TWO WAYS it is unlike tt_eff^-1:
it converges far more slowly, because the dropped term dies only as (1+sigma)^(|alpha|-1)
-- over the drawn range the coasting track reads about -0.63, not -0.667 -- and it is
C-DEPENDENT, its normalisation carrying tt_dyn, so it is a guide for this panel's
reference C and not a universal line the way tt_eff^-1 is.

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

THE COOLING REGIMES, as bands on the top panel. The two knees are the boundaries: tt_M
and tt_m are where the injected edges burn (S = 1/gma_M0 and S = 1/gma_m0), so below tt_M
none of the population has cooled and above tt_m all of it has.

    VSC   tt_eff < tt_M                     nothing has cooled yet
    SC    tt_M < tt_eff < tt_m/MC_FAC       the top edge is burning, gma_m untouched
    MC    tt_m/MC_FAC .. tt_m*MC_FAC        a NEIGHBOURHOOD of tt_m, not a boundary
    FC    tt_m*MC_FAC < tt_eff < 1          the whole population has cooled
    VFC   tt_eff > 1

MC is the one band that is a choice: marginal cooling is a neighbourhood of tt_m rather
than a point, taken here as a factor MC_FAC = 3 either side. The others are set by the
knees. Colours are the house shape-class palette (sweep_gammacm's per-spectrum table),
RdBu from VSC red to VFC blue; 'marginal' is #f7f7f7 there, invisible as a tint, so MC
gets a grey instead.

VFC IS EXACTLY WHERE THE MODEL STOPS, and not by coincidence of these bounds. gma_M ->
1/tt_eff once the top edge has burnt (that is what this abscissa is for), so tt_eff > 1
means gma_M < 1: the whole population, its most energetic electron included, is
sub-relativistic. That holds for any gma_M0 >> 1, so the VFC band is ALWAYS past the
ultra-relativistic trajectory's validity. The tracks are drawn through it anyway -- the
integrated figures have always carried their curves into the sub-relativistic region, and
cutting them here made the one band they were added to populate look like blank axis. The
crimson gma = 1 line is the caveat: below it the real loss rate dies as gma^2 - 1 and the
electrons stall near 1 instead of reaching the 1e-2 the tracks show at the right edge.
Read that stretch as where the model says the population is HEADING, not where it is.

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
import matplotlib.colors as mcolors
from fractions import Fraction

from cooling_distribution import gamma_synCooled, norm_plaw_distrib, distrib_plaw_cooled
from scipy.optimize import brentq
from cooling_shape_figure import (P_SYN, GM0, GMA_M0, NG, OUTDIR, INK, MUTED, FIGSIZE,
    FS_LAB, FS_TICK, FS_ANN, FS_LEG, LOGTT_SAMPLES, cooled_distrib)
from cooling_integrated_figure import tt_dyn_of_C
from cooling_integrated_adiabatic import (A_RHO, Q_B, _exps, A_of_sigma, S_of_sigma,
    tt_of_sigma, gamma_cooled)

# --- defaults -------------------------------------------------------------------------
LOGC = 0.                   # gma_c/gma_m for the tt <-> sigma map; marginal cooling
# sampled at the SAME log10 values cooling_shape_figure samples log10 tt at -- the point
# of tt_eff being that the two figures then share one clock, slice for slice
LOGTTE_SAMPLES = LOGTT_SAMPLES
TTE_LIM = (1e-11, 1e2)      # tt_eff range of the top panel: past 1 so VFC is a band
MC_FAC = 3.                 # MC is taken as tt_m/MC_FAC .. tt_m*MC_FAC
BAND_ALPHA = .13            # tint of the regime bands
LOGC_M = (2., 0., -2.)      # log10(bar{gma}_c/gma_m) drawn for the gma_m track. gma_M
                            # is C-INDEPENDENT in tt_eff (measured identical to 4-5
                            # significant figures over logC = -3..+3), so only the
                            # bottom edge is worth repeating. LOGC stays the REFERENCE:
                            # it sets the bands, the knees and panel (b).
FNAME = 'cooling_shape_adiabatic.png'


# --- the generalised normalised time ----------------------------------------------------
def tt_eff_of_sigma(sigma, ttd, a_rho=A_RHO, q=Q_B):
  '''
  The GENERALISED normalised time of the module docstring,

      tt_eff = (1/A(t')) int_{t'_i}^{t'} A(s)/t'_c(s) ds  =  S/A,

  the variable in which the adiabatic problem carries the synchrotron figure's 1/tt
  asymptote. Strictly increasing (S grows while A falls), so it is a valid abscissa.
  '''
  return S_of_sigma(sigma, ttd, a_rho, q)/A_of_sigma(sigma, a_rho)


def sigma_of_tt_eff(tt_eff, ttd, a_rho=A_RHO, q=Q_B, br=(-16., 9.)):
  'Inverse of tt_eff_of_sigma -- monotone, so one bracketed root in log10 sigma.'
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


# the sigma values LOGTTE_SAMPLES maps to at the module defaults: what the figure draws,
# and what the checks below are run at. Nine bracketed root-finds, done once at import.
SIG_SAMPLES = tuple(sigma_of_tt_eff(10.**l, tt_dyn_of_C(LOGC, GM0))
                    for l in LOGTTE_SAMPLES)


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


def _band_bg(col, alpha=BAND_ALPHA):
  '''
  The OPAQUE colour a band tint blends to over white. A label box painted in it is
  invisible against its own band but still masks whatever runs underneath -- which is
  what the MC label needs, sitting as it does exactly on the tt_m vertical.
  '''
  return tuple(alpha*c + (1. - alpha) for c in mcolors.to_rgb(col))


# --- the figure ---------------------------------------------------------------------------
def plot_cooling_shape_adiab(logC=LOGC, p=P_SYN, gm0=GM0, gM0=GMA_M0, a_rho=A_RHO, q=Q_B,
    logtte=LOGTTE_SAMPLES, logC_m=LOGC_M, outdir=OUTDIR, fname=FNAME, show=False):
  '''
  Two-panel view, laid out exactly as cooling_shape_figure so the pair can be read across:
  the edge tracks on top, the distributions they sample below.
  '''
  ttd = tt_dyn_of_C(logC, gm0)
  tte_arr = 10.**np.asarray(logtte, dtype=float)
  sig_arr = np.array([sigma_of_tt_eff(t, ttd, a_rho, q) for t in tte_arr])
  colors = plt.cm.viridis(np.linspace(0., .85, len(tte_arr)))
  K0 = norm_plaw_distrib(gm0, gM0, p)

  fig, (axT, axD) = plt.subplots(2, 1, figsize=FIGSIZE,
      # hspace .24 left panel (a)'s xlabel only 3.2 px clear of panel (b) -- it bit.
      # .32 puts it 15.1 px clear while its own tick labels stay 1.4 px above it, so
      # it groups with the panel it names. The FIGURE HEIGHT is unchanged by this:
      # hspace redistributes space between panels, it does not add any.
      gridspec_kw=dict(height_ratios=[1., 1.45], hspace=.32))

  # (a) edge tracks vs sigma -------------------------------------------------------------
  sg = np.geomspace(*(sigma_of_tt_eff(t, ttd, a_rho, q) for t in TTE_LIM), 900)
  x = tt_eff_of_sigma(sg, ttd, a_rho, q)             # the abscissa: S/A, not sigma
  # tracks continue BELOW gma = 1 rather than stopping there, matching the integrated
  # figures, which have always drawn their curves into the shaded sub-relativistic
  # region. The gma = 1 line still marks where the ultra-relativistic trajectory
  # stops being the physical one; it is a caveat on the curve, not a reason to hide
  # where the model says the population is heading.
  # the cooling REGIMES, as bands on the same axis the knees are marked on. tt_M and
  # tt_m are where the two injected edges burn (S = 1/gma_M0, 1/gma_m0), so they are the
  # regime boundaries: above tt_m the whole population has cooled, below tt_M none of it
  # has. MC is not a boundary but a NEIGHBOURHOOD of tt_m, taken here as a factor
  # MC_FAC either side. Colours are the house shape-class palette (sweep_gammacm's
  # per-spectrum table), RdBu from VSC red to VFC blue; 'marginal' is #f7f7f7 there,
  # invisible as a tint, so MC gets a grey instead.
  S_sg = S_of_sigma(sg, ttd, a_rho, q)
  t_M, t_m = (np.interp(1./g, S_sg, x) for g in (gM0, gm0))
  bands = (('VSC', x[0],        t_M,          '#b2182b'),
           ('SC',  t_M,         t_m/MC_FAC,   '#ef8a62'),
           ('MC',  t_m/MC_FAC,  t_m*MC_FAC,   '0.6'),
           ('FC',  t_m*MC_FAC,  1.,           '#67a9cf'),
           ('VFC', 1.,          x[-1],        '#2166ac'))
  # GUARD: tt_m is found by interpolating S, and at large C the bottom edge never burns
  # inside the plotted window -- S tops out below 1/gma_m0 -- so the interp saturates at
  # the panel edge and FC comes out as a REVERSED span lying over VFC. Clip every band to
  # the axis and drop the ones that collapse; a missing band is the honest rendering of
  # a regime this window does not reach.
  for lab, a_, b_, col in bands:
    a_, b_ = max(a_, x[0]), min(b_, x[-1])
    if not a_ < b_:
      continue
    axT.axvspan(a_, b_, color=col, alpha=BAND_ALPHA, lw=0, zorder=0)
    # y = 0.96, not 0.985: with a box the label needs its pad to stay INSIDE the axes,
    # or the box paints over the top spine and breaks it into segments
    axT.annotate(lab, (np.sqrt(a_*b_), .96), xycoords=('data', 'axes fraction'),
                 color=INK, fontsize=FS_ANN, ha='center', va='top',
                 bbox=dict(fc=_band_bg(col), ec='none', pad=1.5))
  # 1/tt_eff is the burn-off asymptote, EXACTLY as 1/tt is in cooling_shape_figure --
  # that is what this abscissa buys, and the top edge slides down along it
  axT.loglog(x, 1./x, color=MUTED, ls='-.', lw=.9,
             label='$\\tilde{t}_{\\rm eff}^{-1}$')
  # the ADIABATIC asymptote, companion to that one and drawn the same way -- exactly,
  # not offset, so each guide converges onto the track it describes. With e = 1 + alpha
  # the identity tt_eff = (ttd/e)[(1+sigma) - (1+sigma)^|alpha|] leaves (1+sigma) ->
  # e*tt_eff/ttd whenever a_rho > -3, so A -> (e*tt_eff/ttd)^alpha: SLOPE alpha = a_rho/3,
  # i.e. -2/3 for coasting, against -1 for synchrotron. Two caveats it does not share
  # with tt_eff^-1: it converges far more slowly (the dropped term dies only as
  # (1+sigma)^(|alpha|-1), so the drawn range shows ~-0.63 rather than -0.667), and it is
  # C-DEPENDENT -- the normalisation carries tt_dyn -- so it is a guide for this panel's
  # reference C, not a universal line.
  alpha_a, e_a = _exps(a_rho, q)
  fr = Fraction(alpha_a).limit_denominator(100)
  exp_lab = f'{fr.numerator}' if fr.denominator == 1 else f'{fr.numerator}/{fr.denominator}'
  axT.loglog(x, gM0*(e_a*x/ttd)**alpha_a, color=MUTED, lw=.9,
             ls=(0, (4, 1.2, 1, 1.2, 1, 1.2)),
             label=f'$\\tilde{{t}}_{{\\rm eff}}^{{{exp_lab}}}$')
  # the two ghosts separate the causes: what synchrotron alone would do at the same time,
  # and what expansion alone would do. The real track is below both.
  axT.loglog(x, gamma_synCooled(tt_of_sigma(sg, ttd, q), gM0), color='0.68', lw=.8,
             ls='-', label='syn.')
  axT.loglog(x, A_of_sigma(sg, a_rho)*gM0, color='0.68', lw=.8, ls=(0, (4, 1.5)),
             label='adiab.')
  axT.loglog(x, gamma_cooled(sg, gM0, ttd, a_rho, q), color='k', lw=1.4,
             label='$\\gamma_\\mathrm{M}$')
  # gma_m ONCE PER C. Each regime has its own tt_eff <-> sigma map, so each gets its own
  # sigma grid; the abscissa is the same physical variable for all of them, which is why
  # they can share an axis at all -- and why gma_M, drawn once above, lands on every one
  # of their tracks.
  colors_m = plt.cm.jet(plt.Normalize(-3., 3.)(np.asarray(logC_m, dtype=float)))
  h_m, lo_m = [], []
  for lc, col in zip(logC_m, colors_m):
    ttd_c = tt_dyn_of_C(lc, gm0)
    sg_c = np.geomspace(*(sigma_of_tt_eff(t, ttd_c, a_rho, q) for t in TTE_LIM), 900)
    x_c = tt_eff_of_sigma(sg_c, ttd_c, a_rho, q)
    g_c = gamma_cooled(sg_c, gm0, ttd_c, a_rho, q)
    ln, = axT.loglog(x_c, g_c, color=col, lw=1.1, ls='--', zorder=3)
    h_m.append(ln); lo_m.append(float(g_c[-1]))
  # a black PROXY so gma_m appears in the top row beside gma_M, as the quantity it is;
  # which C each coloured track belongs to is the in-panel key's job, not this one's
  axT.plot([], [], color='k', lw=1.1, ls='--', label='$\\gamma_\\mathrm{m}$')
  axT.axhline(1., color='crimson', ls=':', lw=.9, zorder=1)
  # the knees are where the edges burn, i.e. where S reaches 1/gma_edge. In tt_eff they
  # land a little RIGHT of 1/gma_edge, by the factor 1/A they have picked up by then.
  for k, lab in ((t_M, '$\\tilde{t}_M$'), (t_m, '$\\tilde{t}_m$')):
    axT.axvline(k, color=INK, ls=':', lw=.8, zorder=1)
    axT.annotate(lab, (k, .015), xycoords=('data', 'axes fraction'), color=INK,
                 fontsize=FS_ANN, ha='center', va='bottom',
                 bbox=dict(fc='w', ec='none', alpha=.85, pad=1.))
  axT.set_xlim(x[0], x[-1])
  # the floor follows the tracks: they now run on below gma = 1, and at the right-hand
  # edge both edges have converged to ~1e-2, so a fixed 0.15 would cut them off partway
  # through VFC -- the one band they were added to populate. A decade of empty top for
  # the band labels.
  # TWO decades of empty top, not one: the band labels now carry boxes, and a box at
  # y = 0.96 reaches down to ~0.86 in axes units -- with only one decade the gma_M
  # plateau sits at 0.877 and the VSC box paints over it.
  axT.set_ylim(.3*min(lo_m), 300.*gM0)
  # pin the ticks to EVEN powers so gma = 1 carries one: with the extra headroom the
  # default locator lands on 10^9, 10^7, ... and the crimson gma = 1 line ends up
  # between two labelled ticks, which is the one value a reader looks for here
  axT.set_yticks(10.**np.arange(-2., np.log10(gM0) + 1., 2.))
  # this label sits in the GAP between the panels, so it is kept tight against the
  # axis it names -- at the default pads it drifts closer to the panel below and
  # reads as belonging to that one
  axT.set_xlabel('$\\tilde{t}_{\\rm eff}$', fontsize=FS_LAB, labelpad=1.)
  axT.tick_params(axis='x', pad=1.5)
  axT.set_ylabel('$\\gamma$', fontsize=FS_LAB)
  # TWO legends, split by what depends on C. The C-INDEPENDENT curves go in one
  # horizontal row above the panel (the interior has no box-sized gap: the tracks sweep
  # through the bottom centre and the bands claim the top strip). The gma_m family gets
  # its own key at lower left, the one corner all three leave empty.
  # ORDER MATTERS. The legend that sits OUTSIDE the axes has to be the one left in
  # ax.legend_, because bbox_inches='tight' walks that and not artists re-parented with
  # add_artist -- built the other way round, the top row is cropped off the page.
  # the C key is a VERTICAL box inside the panel, centre left. It sits over the gma_m
  # plateaux, which is accepted here: they are flat and identical there, so the box hides
  # nothing a reader needs, and it keeps the row above the panel to one line.
  lab_c = lambda lc: '$1$' if lc == 0. else f'$10^{{{lc:.0f}}}$'
  leg_m = axT.legend(h_m, [lab_c(lc) for lc in logC_m], fontsize=FS_LEG,
                     loc='center left',
                     title='$\\bar{\\gamma}_{\\rm c}/\\gamma_{\\rm m}$',
                     framealpha=.9, handlelength=1.4, labelspacing=.22,
                     handletextpad=.4, borderpad=.35)
  leg_m.get_title().set_fontsize(FS_LEG)
  axT.add_artist(leg_m)
  h_ref = [ln for ln in axT.get_lines() if not ln.get_label().startswith('_')]
  # a FIGURE legend, placed over panel (a) in figure coordinates. As an AXES legend
  # anchored outside its own axes this was silently dropped by savefig's tight-bbox pass
  # and cropped off the page -- get_tightbbox() reported it at dpi=100 but the saved
  # figure came out 0.25 in shorter, exactly the legend's height, and reordering the two
  # legends did not help. Figure legends are always walked.
  pT = axT.get_position()
  leg = fig.legend(h_ref, [ln.get_label() for ln in h_ref], fontsize=FS_LEG,
                   loc='lower center', bbox_to_anchor=(pT.x0 + .5*pT.width, pT.y1 + .008),
                   bbox_transform=fig.transFigure, ncol=len(h_ref), framealpha=1.,
                   handlelength=1.2, handletextpad=.4, borderpad=.3, columnspacing=.9)
  axT.grid(alpha=.25, lw=.4)

  # (b) the distributions ------------------------------------------------------------------
  gma0, N0 = cooled_distrib_adiab(0., ttd, p, gm0, gM0, a_rho, q)
  axD.loglog(gma0, N0, color='k', lw=1.4, zorder=2)
  edges = []
  for ls, sigma, c in zip(logtte, sig_arr, colors):
    gma, N = cooled_distrib_adiab(sigma, ttd, p, gm0, gM0, a_rho, q)
    axD.loglog(gma, N, color=c, lw=1.2, solid_capstyle='round', label=f'{ls:.0f}',
               zorder=3)
    edges.append((gma[-1], N[-1]))
  edges = np.array(edges)
  axD.plot(edges[:, 0], edges[:, 1], color=MUTED, lw=.7, zorder=4)
  axD.scatter(edges[:, 0], edges[:, 1], s=9, facecolors=colors, edgecolors='w',
              linewidths=.5, zorder=6)

  gg = np.geomspace(gm0, gM0, 3)
  axD.loglog(gg, 12.*K0*gg**-p, color=MUTED, ls=':', lw=.9)
  axD.annotate('$\\propto\\gamma^{-p}$', (gg[1], 12.*K0*gg[1]**-p),
               textcoords='offset points', xytext=(3, 3), color=MUTED, fontsize=FS_ANN)
  for v, lab in ((gm0, '$\\gamma_{\\mathrm{m},\\!0}$'), (gM0, '$\\gamma_{\\mathrm{M},\\!0}$')):
    axD.axvline(v, color=INK, ls=':', lw=.8, zorder=1)
    axD.annotate(lab, (v, .985), xycoords=('data', 'axes fraction'), color=INK,
                 fontsize=FS_ANN, ha='center', va='top',
                 bbox=dict(fc='w', ec='none', alpha=.85, pad=1.))
  axD.axvline(1., color='crimson', ls=':', lw=.9, zorder=1)
  axD.set_xlim(.5*float(gamma_cooled(sig_arr[-1], gm0, ttd, a_rho, q)), 2.*gM0)
  axD.set_ylim(1e-16, 1e4)
  axD.set_xlabel('$\\gamma$', fontsize=FS_LAB)
  axD.set_ylabel('$N(\\gamma,\\tilde{t}_{\\rm eff})/N_{\\rm e}$', fontsize=FS_LAB)
  leg = axD.legend(fontsize=FS_LEG, ncol=2, loc='lower left', framealpha=.9,
                   title='$\\log_{10}\\tilde{t}_{\\rm eff}$', handlelength=1.1,
                   labelspacing=.25,
                   columnspacing=.9, handletextpad=.5, borderpad=.4)
  leg.get_title().set_fontsize(FS_LEG)
  axD.grid(alpha=.25, lw=.4)

  for ax in (axT, axD):
    ax.tick_params(which='both', labelsize=FS_TICK)

  os.makedirs(outdir, exist_ok=True)
  path = os.path.join(outdir, fname)
  # NO bbox_extra_artists here. Passing it REPLACES the default extras for every axes
  # (Figure.get_tightbbox hands the same list down to each Axes.get_tightbbox), which
  # dropped the legend above panel (a) and cropped it off the page. The default walk
  # already covers figure legends.
  fig.savefig(path, dpi=300, bbox_inches='tight')
  print(f'saved {path}')
  if show:
    plt.show()
  return fig, (axT, axD)


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
  for ls, sigma in zip(LOGTTE_SAMPLES, SIG_SAMPLES):
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
