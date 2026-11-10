# -*- coding: utf-8 -*-
# @Author: acharlet

'''
The cooling-shape family, regrouped BY QUANTITY instead of by physics.

cooling_shape_figure and cooling_shape_adiabatic each draw one physics case with two
stacked panels (edge tracks above, distributions below). This module transposes that:
one figure per quantity, three panels side by side, one per physics case. The panel
ORDER is the same in both figures, so panel (b) of one is panel (b) of the other.

    cooling_tracks.png    gma_m(t') and gma_M(t') against t'/t'_c,i
    cooling_shapes.png    dN_e/dgma_e at sampled log10(t'/t'_c,i)

THE THREE PANELS, and what each assumes.

  (a) SYNCHROTRON ONLY, constant t'_c. No expansion, so the field has no reason to
      change and tt = int dt'/t'_c IS t'/t'_c,i exactly. This is the panel that carries
      the COOLING REGIMES: they are defined by the two injected edges burning, at
      t'_M = 1/gma_M,i and t'_m = 1/gma_m,i, and in these units they need no reference
      C at all. Physics from cooling_shape_figure.

      In (b) and (c) the knees are also the CLASSICAL t'_x = 1/gma_X,i (article Eq.
      eqn:t_x, 2026-10-23): the exact half-energy times are a line width later, and in
      the slow-cooling panel t'_m is drawn although gma_m never loses half its energy.

  (b) SYNCHROTRON + ADIABATIC, fast cooling, gma_m,i I_dyn = 1e+2.
  (c) the same, slow cooling, gma_m,i I_dyn = 1e-2.

      The control parameter C (LOGC_PANELS = log10 C) is the old dynamical ratio,
      1/(gma_m,i I_dyn) with I_dyn = I(2t'_i) = (1 - 2^-s)/(s gbar_c,i) (article Eq.
      eqn:I_dyn, 2026-10-23), and gbar_c,i = t'_c,i/t'_i = 1/tt_dyn the article's
      cooling Lorentz factor. At s = 3 the two differ by 3.43, so tt_dyn = 3.43/(C gma_m,i)
      (tt_dyn_of_gbar_c) -- the same C as Figs. 4-6, which is the point.

      Both on the FREELY EXPANDING SHELL at constant eps_B, which is one choice, not
      two: rho' ~ R^-2 (a shell of fixed comoving width, coasting R/R_0 = 1 + sigma),
      and then B'^2/8pi = eps_B e' with e' ~ rho'^gma_ad ties the field to it,
      B' ~ rho'^(gma_ad/2), so q = -a_rho*gma_ad = 10/3 is DERIVED (q is the
      exponent of t'_c, not of B'). With d = -a_rho/3 = 2/3 that gives
      s = d + q - 1 = 3 > 0: the synchrotron burn saturates at S_inf = tt_dyn/s, and
      gma_m,i burns only for C < 1/(1 - 2^-s) = 8/7. Each carries the scale its regime
      is set by, as a dash-dotted vertical: t'_dyn on the tracks, I_dyn^-1 on the
      shapes, I_dyn^-1 = s/[(1 - 2^-s) tt_dyn]. The constants live in
      cooling_integrated_adiabatic (A_RHO, Q_C)
      and the trajectory in gamma_cooled / cooled_distrib_adiab, so nothing is
      restated here.

WHY gma_M MOVES BETWEEN (a), (b) AND (c). The abscissa is the physical comoving time
t'/t'_c,i in all three, not the generalised tt_eff = S/A. In tt_eff the top edge is
C-independent and slides down 1/tt_eff in every panel; in t'/t'_c,i it is not, which is
exactly why the two regimes need panels of their own instead of one shared pair of axes.
For (b) and (c), sigma = (t'/t'_c,i)/tt_dyn does the conversion.

EVERY VERTICAL IS NAMED ON ITS OWN LINE rather than in the legend, which leaves the
legend for the curves alone. The tracks put their names along the FLOOR, in the gutter
the gma_e = 1 cut leaves empty; the shapes put theirs along the top, where the curves
have already fallen away. A&A: no titles and no grid -- the case each panel shows is
written INSIDE it (top right on the tracks, bottom left on the shapes), on an opaque box.
Subscripts follow the article: upright c, m, M and italic i (t'_{c,i} is {\rm c},i).

THE TRACKS ARE CUT AT gma_e = 1. Below it the ultra-relativistic trajectory is not the
physical one, and continuing it only asserts where a model that no longer holds would
point. The shapes still draw into the shaded region, because their curves END there
rather than passing through.

The two figures share: the sampled times (cooling_shape_figure.LOGTT_SAMPLES, the same
log10 t'/t'_c,i in every panel), the viridis ramp ordered by time, the injected bounds
as dotted verticals, and the gma < 1 shading -- below it the ultra-relativistic
trajectory is not the physical one, so the curves are drawn but caveated.

Run:  python cooling_shape_panels.py
'''

import os
from fractions import Fraction
import numpy as np
import matplotlib.pyplot as plt

from cooling_distribution import gamma_synCooled, norm_plaw_distrib
from cooling_shape_figure import (P_SYN, GM0, GMA_M0, OUTDIR, INK, MUTED, MFC_FAC,
    named_ticks_top,
    FS_LAB, FS_TICK, FS_ANN, FS_LEG, LOGTT_SAMPLES, BAND_ALPHA, _band_bg, cooled_distrib)
from cooling_integrated_figure import tt_dyn_of_C
from cooling_integrated_adiabatic import (A_RHO, Q_C, _exps, A_of_sigma,
    gamma_cooled)
from cooling_shape_adiabatic import cooled_distrib_adiab, lab_C

# --- defaults -------------------------------------------------------------------------
LOGC_PANELS = (-2., 2.)     # the two adiabatic columns, C = bar{gma}_c,i/gma_m,i:
                            # either side of the C < 1/(1-2^-s) threshold
T_LIM = (1e-3/GMA_M0, 1e2)  # t'/t'_c,i span of the TRACK panels, shared by all three:
                            # starts 3 decades before t'_M = 1/gma_M,i, ends 2 decades
                            # into VFC, whose edge tt = 1 does not move with the bounds
NX = 900                    # points per track
# ONE y range for the three track panels, so the drag in (b)/(c) is read against (a)
# rather than against a rescaled axis. The tracks are CUT at gma_e = 1, so the floor no
# longer has to clear them: it leaves 1.3 decades of empty gutter below the cut, so the
# vertical labels sit clear of the gma_e = 1 line (at .15 that line ran through them),
# and the ceiling leaves the decade the regime-band labels of panel (a) sit in.
GMA_LIM = (.05, 30.*GMA_M0)
FLOOR_GAP = .085            # axes fraction the track verticals stop above the floor: the
                            # names sit in that gap, so no line crosses them and they need
                            # no box (user, 2026-10-20: the boxes stood out on the bands)
N_LIM = (1e-16, 1e4)        # N_e^-1 dN_e/dgma_e range of the SHAPE panels
# a row of three, sized for a two-column article's full width. The shape panels get the
# taller box: twenty decades of N against ten of gamma.
GUIDE_LW = 1.               # guides lie ON the track, so they are drawn over it: dotted
                            # and dash-dot at this width read as grey marks while the
                            # black shows through the gaps. Wider buries the track.
GUIDE_PAD = 2.              # how far a guide runs past the knee it takes over at, and
                            # past the handover: enough to see it arrive and leave, not
                            # enough to leave the track. Starting at the panel edge
                            # instead sent the 1/tt line up through (a)'s VSC label.
GUIDE_FRAC = .2             # the HANDOVER, as a fraction of S_inf: synchrotron guide up
                            # to it, frozen-burn guide after. Each asymptote is exact
                            # only in its own limit, and .2 is where the two errors
                            # balance best -- measured at s = 3, 1/tt is 8-9% low there
                            # and sigma^-d 16-18% high, against 13-14% and 29-30% at .3 --
                            # so the pair covers the whole cooled track without either
                            # visibly leaving it.
FIGSIZE_TRACKS, FIGSIZE_SHAPES = (7.1, 2.75), (7.1, 3.15)
FN_TRACKS, FN_SHAPES = 'cooling_tracks.png', 'cooling_shapes.png'
SAVE_DPI = 450
# A&A: no background grid, and text on an OPAQUE borderless white box so no line crosses
# it (user, 2026-10-16; afterglow_lightcurves.TEXT_BOX)
TEXT_BOX = dict(boxstyle='square,pad=0.15', fc='w', ec='none')
# what each panel is, said on the panel rather than left to the caption
# panel (a) names its CLOCK as well: the grey reference tracks in (b) and (c) are also
# synchrotron only, but on the panel's own decaying t'_c, so 'syn. only' alone would not
# separate the two
PANEL_LABS = ("syn. only\n$t'_{\\rm c}=t'_{{\\rm c},\\!i}$",
              'syn. + adiab. (FC)\n$\\gamma_{{\\rm m},\\!i}I_{\\rm dyn}=10^{2}$',
              'syn. + adiab. (SC)\n$\\gamma_{{\\rm m},\\!i}I_{\\rm dyn}=10^{-2}$')
# the control parameter is gma_m,i I_dyn = 1/C (author 2026-10-23): C, the ratio the runs
# are set by, is the old dynamical bar{gma}_c,i/gma_m,i, now written 1/(gma_m,i I_dyn)

# the shapes put the case bottom left, where the (c) box ran into the gbar_c,i line at
# 1e5 with the regime word on its first line: the ratio already says FC or SC
SHAPE_LABS = (PANEL_LABS[0],) + tuple(l.replace(' (FC)', '').replace(' (SC)', '')
                                      for l in PANEL_LABS[1:])


def _frac_tex(v, sign=True):
  'Render v as a signed (or unsigned) TeX fraction: -2/3 rather than -0.667.'
  fr = Fraction(v).limit_denominator(100)
  n = f'{fr.numerator:+d}' if sign else f'{fr.numerator:d}'
  return n if fr.denominator == 1 else f'{n}/{fr.denominator}'


def _gbar_fac(a_rho=A_RHO, q=Q_C):
  "bar{gma}_c,i/gbar^syn = s/(1 - 2^-s): one dynamical time of a burn that saturates."
  s = _exps(a_rho, q)[1]
  return s/(1. - 2.**-s)


def tt_dyn_of_gbar_c(logC, gm0=GM0, a_rho=A_RHO, q=Q_C):
  '''
  tt_dyn = t'_i/t'_c,i from the ARTICLE's C = bar{gma}_c,i/gma_m,i, with
  bar{gma}_c,i = 1/I(2t'_i) (Eq. eqn:bargamma_c). cooling_integrated_figure's
  tt_dyn_of_C is the s = -1 case (no drag, constant field), where the two coincide.
  '''
  return _gbar_fac(a_rho, q)*tt_dyn_of_C(logC, gm0)


def gbar_c_i(ttd, a_rho=A_RHO, q=Q_C):
  "bar{gma}_c,i = 1/I(2t'_i) of a panel with tt_dyn = ttd: the inverse of tt_dyn_of_gbar_c."
  return _gbar_fac(a_rho, q)/ttd


def panel_specs(gm0=GM0, logC_m=LOGC_PANELS, labs=PANEL_LABS, a_rho=A_RHO, q=Q_C):
  '''
  (kind, tt_dyn, label) for the three panels, in the order the figures use. tt_dyn is
  None for the first panel, which has no C to set one. `labs` names them; the no-drag
  figure passes its own.
  '''
  return ([('syn', None, labs[0])]
          + [('adiab', tt_dyn_of_gbar_c(lc, gm0, a_rho, q), lab)
             for lc, lab in zip(logC_m, labs[1:])])


def _row(figsize, n=3, wspace=.10):
  'A row of n panels sharing one y axis, plus the helper that puts a legend above them.'
  fig, axs = plt.subplots(1, n, figsize=figsize, sharey=True, squeeze=False,
                          gridspec_kw=dict(wspace=wspace))
  return fig, axs[0]


def _legend_above(fig, axs, order=(), **kw):
  '''
  ONE legend row above the panels. In-panel placement has nowhere to go here: the
  panels are a third as wide as the two-panel figures' were, and the injected bounds,
  the band labels and the curve bundle already claim top, bottom and middle. A FIGURE
  legend, because bbox_inches='tight' walks those -- and no bbox_extra_artists, which
  would REPLACE every axes' own extras and crop it.

  Handles are merged across ALL panels and deduped by label, because the entries are
  not all drawn in the same one: the 1/tt guide belongs to (a) and t'_dyn to (b)/(c).
  `order` then fixes the reading order, which plot order cannot -- the guide has to be
  drawn first to sit under the tracks.
  '''
  seen = {}
  for ax in axs:
    for h, l in zip(*ax.get_legend_handles_labels()):
      seen.setdefault(l, h)
  keys = [l for l in order if l in seen] + [l for l in seen if l not in order]
  # above everything on top of the axes -- the named ticks of the shapes included
  fig.canvas.draw()
  r = fig.canvas.get_renderer()
  inv = fig.transFigure.inverted()
  top = max(inv.transform(ax.get_tightbbox(r))[1, 1] for ax in axs)
  p0, p1 = axs[0].get_position(), axs[-1].get_position()
  fig.legend([seen[l] for l in keys], keys, fontsize=FS_LEG, loc='lower center',
             bbox_to_anchor=(.5*(p0.x0 + p1.x1), top + .012),
             bbox_transform=fig.transFigure, framealpha=1., handlelength=1.4,
             handletextpad=.5, borderpad=.3, columnspacing=.9, **kw)


def tt_knee(gma0, ttd, a_rho=A_RHO, q=Q_C):
  '''
  t'/t'_c,i at which the edge injected at gma0 starts to burn -- where the effective
  burn S reaches 1/gma0. S(sigma) inverts in closed form, so this needs no root-find:

      S = ttd[(1+sigma)^e - 1]/e = 1/gma0   =>   1 + sigma = [1 + e/(gma0 ttd)]^(1/e)

  None when that bracket is <= 0. With s > 0 the burn saturates at S_inf = ttd/s, so
  an edge with gma0*ttd < s NEVER burns and the knee does not exist -- the C < 1/s
  threshold seen from the other side, and the reason panel (c) has a t'_M but no t'_m.
  At gma0*ttd >> s the bracket -> 1 and the knee returns to its synchrotron value
  1/gma0, which is why the two tt_M verticals line up across the three panels.
  '''
  _, s = _exps(a_rho, q)
  b = 1. - s/(gma0*ttd)
  return None if b <= 0. else (b**(-1./s) - 1.)*ttd


def _mark_gma1(ax, axis, span=True):
  '''
  gma_e = 1, below which the ultra-relativistic trajectory is not the physical one:
  a y mark on the track panels, an x mark on the shape panels. `span` shades the
  region as well as drawing the line -- the tracks CUT there instead, so they only
  want the line and keep the strip below it as a gutter for the vertical labels.
  '''
  line = ax.axhline if axis == 'y' else ax.axvline
  if span:
    (ax.axhspan if axis == 'y' else ax.axvspan)(1e-30, 1., color='crimson', alpha=.07,
                                                lw=0, zorder=0)
  line(1., color='crimson', ls=':', lw=.9, zorder=1)


def _cut1(y):
  "Tracks stop at gma_e = 1 rather than running on into a region the model cannot state."
  y = np.asarray(y, dtype=float)
  return np.where(y >= 1., y, np.nan)


def _panel_tag(ax, lab, xy, ha, va, box=True):
  '''
  The case a panel shows, written inside it (A&A: no titles), on an opaque box where a
  line crosses it; box=False where none does (panel (a) of the tracks).
  '''
  return ax.annotate(lab, xy, xycoords='axes fraction', ha=ha, va=va, color=INK,
                     fontsize=FS_ANN, multialignment=ha, bbox=TEXT_BOX if box else None,
                     zorder=7)


def _vline(ax, v, lab=None, ls='-', lw=.8, color=INK, ha='center', ymax=1.):
  '''
  A marked vertical. With a name (the tracks' times), the name sits on the FLOOR, in the
  gutter below gma_e = 1, and the line STOPS above it (FLOOR_GAP): nothing crosses the
  text, so it needs no box. Without one (the shapes' Lorentz factors), the name is a
  named tick on the top axis instead (named_ticks_top).
  '''
  ax.axvline(v, ymin=FLOOR_GAP if lab else 0., ymax=ymax, color=color, ls=ls, lw=lw,
             zorder=1)
  if lab:
    ax.annotate(lab, (v, .015), xycoords=('data', 'axes fraction'), color=color,
                fontsize=FS_ANN, ha=ha, va='bottom')


# --- (1) the edge tracks ----------------------------------------------------------------
LAB_GM, LAB_GMM = '$\\gamma_\\mathrm{m}$', '$\\gamma_\\mathrm{M}$'
TCI = "t'_{{\\rm c},\\!i}"          # the article's t'_{\\c,i}: upright c, italic i
LAB_GUIDE, LAB_TDYN = f"$(\\Delta t'_i/{TCI})^{{-1}}$", "$t'_{\\rm dyn}$"
# the knees are named for the PHYSICAL time, not for tt: the abscissa is t'/t'_c,i, and
# outside panel (a) that is not tt at all -- tt lags it once the field decays
LAB_TMM, LAB_TM = "$t'_\\mathrm{M}$", "$t'_\\mathrm{m}$"
LAB_TC1 = "$t'_{\\rm c,1}$"      # time to cool down to gamma_e = 1 (VFC from there)
LAB_SYN = 'syn. only'       # the same population with the expansion switched off
LAB_GC = '$I_{\\rm dyn}^{-1}$'   # 1/I(2t'_i), where the line is drawn (article 2026-10-23)
SYN_REF = '0.62'            # lighter than the tracks, and solid/dashed rather than the
                            # guides' dash-dot and dotted, so it reads as another CASE



def lab_adrift(a_rho=A_RHO):
  '''
  Legend entry for the frozen-burn guide, as a POWER of the abscissa. Once the burn has
  frozen gma ~ A = (1+sigma)^-d, and past a few t'_dyn that is sigma^-d, i.e. a
  straight line of index -d = a_rho/3 in t'/t'_c,i. Rendered as a fraction, so -2/3
  reads as -2/3 and not as -0.667.
  '''
  if abs(a_rho) < 1e-12:      # no drag: once the burn freezes gma_e simply STOPS
    return 'const.'
  return f"$(\\Delta t'_i/{TCI})^{{" + _frac_tex(a_rho/3.) + "}$"
# mathtext puts no space after the comma, so the \! keeps the two indices from touching
LAB_GMI = '$\\gamma_{\\mathrm{m},\\!i}$'
LAB_GMMI = '$\\gamma_{\\mathrm{M},\\!i}$'


def freeze_x(ttd, a_rho=A_RHO, q=Q_C, frac=.5):
  '''
  t'/t'_c,i at which the burn S has reached `frac` of S_inf -- the bend between the two
  asymptotes. With e < 0, S/S_inf = 1 - (1+sigma)^-s inverts exactly. None when e >= 0,
  where the burn never freezes and there is no bend.
  '''
  _, s = _exps(a_rho, q)
  return None if s <= 0. else ((1. - frac)**(-1./s) - 1.)*ttd


def _frozen_guide(ax, x, ttd, a_rho=A_RHO, q=Q_C, frac=GUIDE_FRAC):
  '''
  The frozen-burn asymptote, drawn as the PURE POWER LAW it becomes.

  Once the burn has frozen, the denominator of gma = A gma_0/(1 + gma_0 S) is a constant
  per electron, so EVERY trajectory becomes gma ~ A(t'): cooling is purely adiabatic from
  there on. An edge that burnt (gma_0 S_inf >> 1) forgets gma_0 and lands on A/S_inf
  itself; one that never burnt -- panel (c)'s gma_m -- runs PARALLEL to it at A gma_0,
  the offset between them being the 1 + (1-2^-s) C the width law stalls at.

  A = (1+sigma)^-d, so the line drawn is s/tt_dyn * sigma^-d -- the large-sigma form, a
  straight s-normalised power of the abscissa rather than the exact A/S_inf
  (the two agree to 0.07% by the right-hand edge; main() prints it). It runs from the
  GUIDE_FRAC handover to the right-hand edge, taking over from the 1/tt guide exactly
  where that one gives out, so the two together span the whole cooled track.
  '''
  x0 = freeze_x(ttd, a_rho, q, frac)
  if x0 is None:
    return                                     # unbounded burn: nothing ever freezes
  xg = x[x >= x0/GUIDE_PAD]                    # overlapping the 1/tt guide at the bend
  ax.loglog(xg, _cut1(_exps(a_rho, q)[1]/ttd*(xg/ttd)**(a_rho/3.)), color=MUTED,
            ls=':', lw=GUIDE_LW, zorder=5, label=lab_adrift(a_rho))


def plot_cooling_tracks(p=P_SYN, gm0=GM0, gM0=GMA_M0, a_rho=A_RHO, q=Q_C,
    logC_m=LOGC_PANELS, labs=PANEL_LABS, ref_lab=None, ref_q=None, outdir=OUTDIR,
    fname=FN_TRACKS, show=False):
  '''
  gma_m and gma_M against t'/t'_c,i, one panel per physics case (see module docstring).
  The regime bands are drawn on panel (a) only: that is where they are defined. The
  adiabatic panels instead mark t'_dyn, the scale their C is measured against. The two
  knees echo the edges they belong to -- solid for M, dashed for m -- thin, so they read
  as marks rather than as a third and fourth track. The adiabatic panels also carry the
  frozen-burn asymptote (see _frozen_guide).
  '''
  fig, axs = _row(FIGSIZE_TRACKS)
  x = np.geomspace(*T_LIM, NX)
  # THE COOLING REGIMES. t_M and t_m are the cooling times of the two injected edges;
  # MFC is a NEIGHBOURHOOD of t_m, a factor MFC_FAC either side. Colours are the house
  # shape-class palette (sweep_gammacm's per-spectrum table), RdBu from VSC red to VFC
  # blue; its 'marginal' #f7f7f7 is invisible as a tint, so MFC gets a grey.
  t_M, t_m = 1./gM0, 1./gm0
  bands = (('VSC', x[0],       t_M,        '#b2182b'),
           ('SC',  t_M,        t_m/MFC_FAC, '#ef8a62'),
           ('MFC', t_m/MFC_FAC, t_m*MFC_FAC, '0.6'),
           ('FC',  t_m*MFC_FAC, 1.,         '#67a9cf'),
           ('VFC', 1.,         x[-1],      '#2166ac'))

  ref_lab = ref_lab or LAB_SYN
  # THE REFERENCE TRACK is synchrotron only -- a_rho = 0, so A = 1 and S = tt -- but on
  # the SAME clock as the panel by default, q. That isolates the adiabatic drag: the gap
  # between black and grey is A, nothing else. Passing ref_q = 0 instead gives the
  # constant-t'_c synchrotron, which is what the no-drag figure wants for its reference.
  # gamma_cooled at a_rho = 0, q = 0 reduces to gamma_synCooled exactly.
  ref_q = q if ref_q is None else ref_q
  for ax, (kind, ttd, plab) in zip(axs, panel_specs(gm0, logC_m, labs, a_rho, q)):
    if kind == 'syn':
      for lab, a_, b_, col in bands:
        a_, b_ = max(a_, x[0]), min(b_, x[-1])
        if not a_ < b_:
          continue
        ax.axvspan(a_, b_, color=col, alpha=BAND_ALPHA, lw=0, zorder=0)
        # the last band is the narrowest on the axis: its name is set flush with the
        # right spine instead of centred, where its box ran over the frame
        last = b_ >= x[-1]
        ax.annotate(lab, (.975 if last else np.sqrt(a_*b_), .985),
                    xycoords=('axes fraction' if last else 'data', 'axes fraction'),
                    color=INK, fontsize=FS_ANN, ha='right' if last else 'center',
                    va='top', bbox=dict(fc=_band_bg(col), ec='none', pad=1.5))
      gM, gm = gamma_synCooled(x, gM0), gamma_synCooled(x, gm0)
      x0 = None                                    # nothing freezes: 1/tt runs on
    else:
      sg = x/ttd                                   # sigma = (t'/t'_c,i)/tt_dyn
      gM = gamma_cooled(sg, gM0, ttd, a_rho, q)
      gm = gamma_cooled(sg, gm0, ttd, a_rho, q)
      # the SAME population with the adiabatic drag switched off, drawn thin
      # underneath, so the drag is read off the gap rather than by looking across the
      # figure. The 1/tt guide is kept alongside and is the CONSTANT-t'_c case, which
      # both tracks follow until the clock starts to stretch.
      for g0, ls in ((gM0, '-'), (gm0, '--')):
        ax.loglog(x, _cut1(gamma_cooled(sg, g0, ttd, 0., ref_q)), color=SYN_REF,
                  lw=.9, ls=ls, zorder=2, label=(ref_lab if g0 == gM0 else None))
      # the second asymptote. Before the burn freezes A is still ~1 and S is still
      # ~t'/t'_c,i, so the top edge runs down the same 1/tt it does in (a); after, it
      # turns onto A. The two hand over at GUIDE_FRAC, overlapping by GUIDE_PAD.
      x0 = freeze_x(ttd, a_rho, q, GUIDE_FRAC)
      _frozen_guide(ax, x, ttd, a_rho, q, GUIDE_FRAC)

    # THE KNEES ARE THE ARTICLE'S t'_x (Eq. eq:t_X-gamma_ci), where the integrated burn
    # reaches 1/gma_X,i: t'_c,i/gma_X,i exactly in (a), tt_knee in (b)/(c). There they
    # are later than the naive 1/gma_X,i by 1 + (1+s)/(2 gma_X,i tt_dyn) -- 0.002% for
    # t'_M, 0.6% for t'_m in (b), a line width -- but tt_knee also knows that an edge
    # below (1-2^-s) gbar_c,i NEVER burns: t'_m does not exist in (c), and is not drawn there
    # (it used to be, at the naive value, contradicting Sect. 2; fixed 2026-10-20).
    # 2026-10-23: the article's t'_x is now the CLASSICAL cooling time t'_c,i/gma_x,i
    # (Eq. eqn:t_x), in every panel; in (c) t'_m is drawn although gma_m never burns.
    knees = (t_M, t_m)

    # 1/tt is the asymptote the top edge slides down, never a place it bends. It
    # starts a little before the knee where the edge begins to burn -- there is nothing
    # for it to describe while the edge still sits at gma_M,i -- and runs on past the
    # handover to 2 t'_dyn. Cut AT the handover it was indistinguishable from gma_M
    # everywhere it was drawn; two dynamical times leaves it about a factor 3 clear of
    # the track, enough to read the departure without a long tail of dead line. Panel
    # (a) has no t'_dyn and nothing to depart from, so there it runs the full range.
    xs = x if knees[0] is None else x[x >= knees[0]/GUIDE_PAD]
    if ttd is not None:
      xs = xs[xs <= 2.*ttd]
    ax.loglog(xs, _cut1(1./xs), color=MUTED, ls='-.', lw=GUIDE_LW, zorder=5,
              label=LAB_GUIDE)

    ax.loglog(x, _cut1(gM), color='k', lw=1.4, label=LAB_GMM)
    ax.loglog(x, _cut1(gm), color='k', lw=1.1, ls='--', label=LAB_GM)
    _mark_gma1(ax, 'y', span=False)
    # the verticals, each named ON its line along the floor. The knees echo the edges
    # they belong to, solid for M and dashed for m; None where the edge never burns,
    # which is the slow-cooling panel's missing t'_m.
    # in (c) the classical t'_m lies only 1.5 decades after t'_dyn: their names, centred,
    # ran into each other, so they lean apart, t'_dyn to the left and t'_m to the right
    late_m = ttd is not None and knees[1] > ttd
    for v, ls, lab in zip(knees, ('-', '--'), (LAB_TMM, LAB_TM)):
      if v is not None:
        _vline(ax, v, lab, ls=ls, ha='left' if (late_m and lab == LAB_TM) else 'center')
    if ttd is not None:
      _vline(ax, ttd, LAB_TDYN, ls='-.', lw=.9,   # t'_dyn IS tt_dyn on this abscissa
             ha='right' if late_m else 'center')
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(*GMA_LIM)
    ax.set_yticks(10.**np.arange(0., np.log10(gM0) + 1., 2.))
    # the clock starts at injection, Delta t' = t'-t'_i: sigma = x/tt_dyn is Delta t'/t'_i, R/R_0 = 1 + sigma
    ax.set_xlabel(f"$\\Delta t'_i/{TCI}$", fontsize=FS_LAB, labelpad=1.)
    # the case INSIDE the panel (A&A: no titles), top right where every track has
    # already fallen away; (a) drops it below its regime-band row
    tag = _panel_tag(ax, plab, (.97, .86 if kind == 'syn' else .97), 'right', 'top',
                     box=kind != 'syn')      # (a): no line crosses it
    if kind == 'syn':
      # t'_c,1 = t'_c,i: the time to cool down to gamma_e = 1, where VFC starts (meeting
      # 2026-11-10); on this abscissa it is x = 1, the FC/VFC boundary. Its name is centred
      # on the line, and the line stops below the panel tag (and the VFC band name above
      # it), so that it crosses no text.
      ax.set_xlim(x[0], x[-1]); ax.set_ylim(*GMA_LIM)
      bb = tag.get_window_extent(fig.canvas.get_renderer())
      (x0t, y0t), (x1t, _) = ax.transAxes.inverted().transform(bb.get_points())
      xl = ax.transAxes.inverted().transform(ax.transData.transform((1., 1.)))[0]
      _vline(ax, 1., LAB_TC1, ls=':', lw=.9, ha='center',
             ymax=y0t - .02 if x0t - .01 <= xl <= x1t + .01 else 1.)
    ax.tick_params(axis='x', pad=1.5)
    ax.tick_params(which='both', labelsize=FS_TICK)
  axs[0].set_ylabel('$\\gamma_{\\rm e}$', fontsize=FS_LAB)

  # the scalings last: the tracks and the times that mark them first, then what they
  # tend to -- 1/tt while synchrotron still bites, A(t') once the burn has frozen
  _legend_above(fig, axs, ncol=5,
                order=(LAB_GMM, LAB_GM, ref_lab, LAB_GUIDE, lab_adrift(a_rho)))
  return _save(fig, axs, outdir, fname, show)


# --- (2) the distribution shapes ---------------------------------------------------------
def plot_cooling_shapes(p=P_SYN, gm0=GM0, gM0=GMA_M0, a_rho=A_RHO, q=Q_C,
    logtt=LOGTT_SAMPLES, logC_m=LOGC_PANELS, outdir=OUTDIR, fname=FN_SHAPES, show=False,
    regime_bands=True):
  '''
  N(gma,t') at the sampled log10(t'/t'_c,i), one panel per physics case.

  regime_bands (meeting 2026-11-10, 'all information on regimes visible by eye'): the gamma_e
  axis of panel (a) only is shaded with the regime bands of Fig. 1 (a), in the same colours, read for a cooling
  Lorentz factor: VFC below 1, FC up to gma_m,i, MFC within MFC_FAC of it, SC up to gma_M,i,
  VSC above. In (a) the band holding each dot (gma_M = t'_c,i/Delta t'_i) is the regime that
  Delta t'_i would give as a dynamical time. Same panel
  order as plot_cooling_tracks, and the SAME sampled times in every panel.

  Every vertical is named ON its line along the top strip: the two injected bounds in
  all three panels, and bar{gma}_c,i = 1/I(2t'_i) on the adiabatic two, marked with the
  dash-dot the tracks give t'_dyn.
  '''
  fig, axs = _row(FIGSIZE_SHAPES)
  colors = plt.cm.viridis(np.linspace(0., .85, len(logtt)))
  K0 = norm_plaw_distrib(gm0, gM0, p)
  lo = np.inf

  for ax, (kind, ttd, plab) in zip(axs, panel_specs(gm0, logC_m, SHAPE_LABS, a_rho, q)):
    # the injected law goes UNDER the cooled ones: the earliest of them tracks it
    # almost exactly, and on top the black would hide that curve entirely
    ax.loglog(*(cooled_distrib(0., p, gm0, gM0) if kind == 'syn' else
                cooled_distrib_adiab(0., ttd, p, gm0, gM0, a_rho, q)),
              color='k', lw=1.4, zorder=2)
    edges = []
    for lt, c in zip(logtt, colors):
      gma, N = (cooled_distrib(10.**lt, p, gm0, gM0) if kind == 'syn' else
                cooled_distrib_adiab(10.**lt/ttd, ttd, p, gm0, gM0, a_rho, q))
      ax.loglog(gma, N, color=c, lw=1.2, solid_capstyle='round', zorder=3,
                label=(f'{lt:.0f}' if kind == 'syn' else None))
      lo = min(lo, float(gma[0]))
      edges.append((gma[-1], N[-1]))
    # the burn-off front: locus of the top edge as it slides down
    ed = np.array(edges)
    ax.plot(ed[:, 0], ed[:, 1], color=MUTED, lw=.7, zorder=4)
    ax.scatter(ed[:, 0], ed[:, 1], s=7, facecolors=colors, edgecolors='w',
               linewidths=.4, zorder=6)
    # gma^-p guide, offset above the injected curve so it does not hide it
    gg = np.geomspace(gm0, gM0, 3)
    ax.loglog(gg, 12.*K0*gg**-p, color=MUTED, ls=':', lw=.9)
    if kind == 'syn':
      ax.annotate('$\\propto\\gamma_{\\rm e}^{-p}$', (gg[1], 12.*K0*gg[1]**-p),
                  textcoords='offset points', xytext=(3, 3), color=MUTED,
                  fontsize=FS_ANN)
    else:
      _vline(ax, gbar_c_i(ttd, a_rho, q), ls='-.', lw=.9)  # bar{gma}_c,i = 1/I(2t'_i)
    for v in (gm0, gM0):                                 # the injected bounds
      _vline(ax, v, ls=':')
    # their names are TICKS on the top axis, which leaves the panel to the curves
    named_ticks_top(ax, sorted([(gm0, LAB_GMI), (gM0, LAB_GMMI)]
                               + ([] if kind == 'syn' else
                                  [(gbar_c_i(ttd, a_rho, q), LAB_GC)])))
    _mark_gma1(ax, 'x')
    ax.set_ylim(*N_LIM)
    ax.set_xlabel('$\\gamma_{\\rm e}$', fontsize=FS_LAB)
    # bottom left: the cooled curves never reach below N ~ 1e-9 at gma_e < 1e2
    _panel_tag(ax, plab, (.03, .03), 'left', 'bottom')
    ax.tick_params(which='both', labelsize=FS_TICK)
  for k, ax in enumerate(axs):
    ax.set_xlim(.3 if regime_bands else .5*lo, 2.*gM0)
    if regime_bands and k == 0:          # author 2026-11-10: bands in the synchrotron-only panel only
      x0, x1 = ax.get_xlim()
      bands = (('VFC', x0, 1., '#2166ac'), ('FC', 1., gm0/MFC_FAC, '#67a9cf'),
               ('MFC', gm0/MFC_FAC, gm0*MFC_FAC, '0.6'), ('SC', gm0*MFC_FAC, gM0, '#ef8a62'),
               ('VSC', gM0, x1, '#b2182b'))
      for lab, a_, b_, col in bands:
        ax.axvspan(a_, b_, color=col, alpha=BAND_ALPHA, lw=0, zorder=0)
        if lab in ('VFC', 'VSC'):            # narrow bands: name set inside, flush with the spine
          ax.annotate(lab, (.015 if lab == 'VFC' else .985, .985), xycoords='axes fraction',
                      color=INK, fontsize=FS_ANN - 1, ha='left' if lab == 'VFC' else 'right',
                      va='top', rotation=90, bbox=dict(fc=_band_bg(col), ec='none', pad=1.))
        else:
          ax.annotate(lab, (np.sqrt(a_*b_), .985), xycoords=('data', 'axes fraction'), color=INK,
                      fontsize=FS_ANN, ha='center', va='top', bbox=dict(fc=_band_bg(col), ec='none', pad=1.5))
  axs[0].set_ylabel("$N_{\\rm e}^{-1}\\,{\\rm d}N_{\\rm e}/{\\rm d}\\gamma_{\\rm e}$", fontsize=FS_LAB)

  _legend_above(fig, axs, ncol=len(logtt),
                title=f"$\\log_{{10}}(\\Delta t'_i/{TCI})$")
  fig.legends[-1].get_title().set_fontsize(FS_LEG)
  return _save(fig, axs, outdir, fname, show)


def _save(fig, axs, outdir, fname, show):
  os.makedirs(outdir, exist_ok=True)
  path = os.path.join(outdir, fname)
  # 450, not 300: these are full-width (figure*) and at 300 the tracks printed at ~277 dpi
  # at A&A's 17 cm, the lowest of the article; 450 gives ~415
  fig.savefig(path, dpi=SAVE_DPI, bbox_inches='tight')
  print(f'saved {path}')
  if show:
    plt.show()
  return fig, axs


def main(show=False):
  # the floors the shared y range has to clear, printed so GMA_LIM can be checked
  # against the physics rather than against a remembered render
  x_end = T_LIM[1]
  print(f"lowest edge at t'/t'_c,i = {x_end:g}:")
  print(f'  (a) synchrotron   gma_m = {gamma_synCooled(x_end, GM0):.3e}')
  for lc in LOGC_PANELS:
    ttd = tt_dyn_of_gbar_c(lc, GM0)
    gm = float(gamma_cooled(x_end/ttd, GM0, ttd, A_RHO, Q_C))
    kM, km = (tt_knee(g, ttd) for g in (GMA_M0, GM0))
    print(f'  C = 10^{lc:+.0f}      gma_m = {gm:.3e}   (tt_dyn = {ttd:.3e})'
          f'   t_M = {kM:.3e}'
          f"   t_m = {'never burns' if km is None else f'{km:.3e}'}")
    # the frozen-burn guide is an ASYMPTOTE, so say how close the edges are to it by
    # the end of the panel rather than trusting the overlay to the eye. gma_M burnt,
    # so it tends to A/S_inf itself; gma_m only does where it burnt too.
    s_inf = ttd/abs(_exps(A_RHO, Q_C)[1])
    gA = float(A_of_sigma(x_end/ttd, A_RHO))/s_inf
    gM = float(gamma_cooled(x_end/ttd, GMA_M0, ttd, A_RHO, Q_C))
    # and how good the PURE POWER the legend advertises is against the exact A/S_inf
    pw = abs(_exps(A_RHO, Q_C)[1])/ttd*(x_end/ttd)**(A_RHO/3.)
    print(f'                   gma_M/(A/S_inf) = {gM/gA:.4f}'
          f'   power/exact = {pw/gA:.4f}'
          f'   gma_m/(A gma_m,i) = {gm/(float(A_of_sigma(x_end/ttd, A_RHO))*GM0):.4f}'
          f'   gma_M/gma_m = {gM/gm:8.3f}  (1+(1-2^-s)C = {1. + (1. - 2.**-_exps(A_RHO, Q_C)[1])*10.**lc:8.3f})')
  # the no-drag case is no longer a figure of its own -- it IS the grey reference in
  # the adiabatic panels. s collapses to q - 1 there, and with nothing to carry gma_e
  # down after the burn freezes each edge simply stops at 1/S_inf, which is why those
  # tracks run flat while the black ones keep falling as A(t')
  s_nd = _exps(0., Q_C)[1]
  print(f'no drag (a_rho = 0): s = q-1 = {s_nd:.4f}, threshold C < 1/s = {1./s_nd:.4f}')
  print("naive knees t'_X = 1/gma_X,i against the integrated ones (S = 1/gma_X,i):")
  for lc in LOGC_PANELS:
    ttd = tt_dyn_of_gbar_c(lc, GM0)
    for g0, nm in ((GMA_M0, "t'_M"), (GM0, "t'_m")):
      ex = tt_knee(g0, ttd, A_RHO, Q_C)
      r = f'{ex/(1./g0):.4f}' if ex is not None else 'NEVER BURNS'
      print(f"  C = 10^{lc:+.0f}  {nm}: naive {1./g0:.3e}   exact/naive {r:>11s}"
            f"   (gma_0 tt_dyn/s = {g0*ttd/_exps(A_RHO, Q_C)[1]:.3g})")
  for lc in LOGC_PANELS:
    ttd = tt_dyn_of_gbar_c(lc, GM0)
    s_inf = ttd/s_nd
    print(f'  C = 10^{lc:+.0f}   1/S_inf = {1./s_inf:.3e}   '
          f'gma_m,i*S_inf = {GM0*s_inf:9.3g} -> '
          f'{"cools" if GM0*s_inf > 1. else "never burns"}')
  plot_cooling_tracks(show=show)
  plot_cooling_shapes(show=show)


if __name__ == '__main__':
  main()
