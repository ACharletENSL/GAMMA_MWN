# -*- coding: utf-8 -*-
# @Author: acharlet

'''
How a power-law electron distribution is reshaped by synchrotron cooling.

Base module of the four that share figures/cooling_distributions (it defines OUTDIR for
all of them): {cooling_shape_figure, cooling_integrated_figure} x {synchrotron only,
+ adiabatic}, the adiabatic halves being cooling_shape_adiabatic and
cooling_integrated_adiabatic. This one is the instantaneous, synchrotron-only corner.

Purely analytic figure -- no simulation data. It draws the exact solution that the
whole cooling pipeline is built on, using the very functions the pipeline calls
(cooling_distribution.gamma_synCooled / .distrib_plaw_cooled), so the picture is a
statement about the implemented model, not a redrawing of it.

The model. An electron injected with Lorentz factor gma0 and cooling by synchrotron
alone follows gma(tt) = gma0/(1 + gma0*tt), with the NORMALIZED TIME

    tt = tilde{t} = int dt'/t_{c,1}

(t_{c,1} = comoving cooling time of a gma = 1 electron). Number conservation,
N(gma,tt) dgma = N0(gma0) dgma0, then turns the injected power law
N0 = K0 gma0^-p on [gma_m0, gma_M0] into

    N(gma,tt) = K0 gma^-p (1 - gma*tt)^(p-2)   on  [gma_m(tt), gma_M(tt)]

with both edges cooling by the same law. The two panels split the story in time:

(a) The edges and the break versus tt, i.e. the time axis panel (b) samples. The two
    knees are at tilde{t}_M = 1/gma_M0 (the top edge starts burning) and
    tilde{t}_m = 1/gma_m0 (the bottom edge follows, the power law is gone) -- each the
    cooling time of the edge it takes down.

(b) The support burns down from the top. gma_M(tt) -> 1/tt for tt >> 1/gma_M0, so the
    cooling break 1/tt is an ASYMPTOTE the top edge slides down along, never a place
    where the distribution bends: the injected edge stays a sharp edge, only curled
    just below it by the factor (1 - gma*tt)^(p-2). Below the edge the gma^-p segment
    is untouched until tt reaches 1/gma_m0. Past that the bottom edge follows the top
    one down and the width collapses --

        gma_M/gma_m = (gma_M0/gma_m0) (1 + gma_m0 tt)/(1 + gma_M0 tt)  ->  1,

    so the distribution ends up MONO-ENERGETIC at gma ~ 1/tt, whatever it was injected
    as. That is the last curves of the figure: on the fiducial bounds (gma_m0 = 1e3,
    gma_M0 = 1e8) five injected decades are down to a 10%-wide spike at gma = 100 by
    tt = 1e-2, and 0.1% wide by tt = 1.

Number is conserved exactly at every tt (checked by check_number_conservation) -- the
distribution loses energy, not electrons.

Validity. gma_synCooled is the ultra-relativistic solution and drives gma -> 0; below
gma = 1 (marked in red in both panels) it is no longer the physical trajectory, which
is why the emission code truncates its gamma integral there. The TRACKS are still drawn
below it, as the integrated figures draw their curves into the same region -- the red
line is a caveat on the curve, not a reason to hide where the model points
(radiation_cooling.get_epnu:
number is conserved, so the clipped bound is the whole correction). The last
sampled time, tt = 1, lands exactly on that line: gma_M = 1/(1 + 1/gma_M0) -> 1, so on
these bounds tt = 1 IS the end of the model's validity, not a time it can be pushed
past.

Run:  python cooling_shape_figure.py
'''

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from environment import GAMMA_dir
from cooling_distribution import gamma_synCooled, norm_plaw_distrib, distrib_plaw_cooled

# --- defaults -------------------------------------------------------------------------
P_SYN = 2.5                 # phys_input.ini psyn
GM0, GMA_M0 = 1e3, 1e8      # injected bounds gma_m0, gma_M0 (fiducial case)
# sampled normalized times as log10(tt): from the top edge just starting to burn
# (tt = 1/gma_M0 = 1e-8) through the mono-energetic limit (tt >> 1/gma_m0 = 1e-3) and
# on to tt = 1, where the population has cooled all the way onto gma = 1
LOGTT_SAMPLES = (-8., -7., -6., -5., -4., -3., -2., -1., 0.)
NG = 800                    # points per curve
# ONE folder for the whole analytic-distribution family: the instantaneous shape and the
# time-integrated distribution, each with and without adiabatic cooling. Defined here, in
# the base module, and imported by the other three so the four cannot drift apart.
OUTDIR = os.path.join(GAMMA_dir, 'bin', 'Tools', 'figures', 'cooling_distributions')

MC_FAC = 3.                 # MC is taken as t_m/MC_FAC .. t_m*MC_FAC
BAND_ALPHA = .13            # tint of the regime bands

# recessive ink for every non-data mark (text never wears a series colour)
INK, MUTED = '0.25', '0.55'

# sized for ONE column of a two-column article: panels stacked, ~3.4 in wide, so
# every mark and every font is set for that final printed size, not rescaled after
FIGSIZE = (3.4, 5.0)
FS_LAB, FS_TICK, FS_ANN, FS_LEG = 9., 8., 8., 7.


def _band_bg(col, alpha=BAND_ALPHA):
  'Opaque colour a band tint blends to over white, so a label box can match its band.'
  return tuple(alpha*c + (1. - alpha) for c in mcolors.to_rgb(col))


# --- the distribution -----------------------------------------------------------------
def cooled_distrib(tt, p=P_SYN, gm0=GM0, gM0=GMA_M0, Ng=NG):
  '''
  (gma, N(gma,tt)) sampled over the SUPPORT [gma_m(tt), gma_M(tt)] only.
  distrib_plaw_cooled stays positive all the way up to 1/tt, but the electrons
  between gma_M(tt) and 1/tt do not exist -- nothing was injected above gma_M0.
  '''
  K0 = norm_plaw_distrib(gm0, gM0, p)
  gm, gM = gamma_synCooled(tt, gm0), gamma_synCooled(tt, gM0)
  gma = np.geomspace(gm, gM, Ng)
  return gma, K0*distrib_plaw_cooled(gma, p, tt)


def check_number_conservation(tt_arr, p=P_SYN, gm0=GM0, gM0=GMA_M0, Ng=20000):
  '''
  int N dgma must stay 1 for every tt: cooling moves electrons, it does not remove
  them. Returns the largest relative deviation over tt_arr.
  '''
  dev = 0.
  for tt in tt_arr:
    gma, N = cooled_distrib(tt, p, gm0, gM0, Ng)
    dev = max(dev, abs(np.trapezoid(N, gma) - 1.))
  return dev


# --- the figure -----------------------------------------------------------------------
def plot_cooling_shape(p=P_SYN, gm0=GM0, gM0=GMA_M0, logtt=LOGTT_SAMPLES,
    outdir=OUTDIR, fname='cooling_shape.png', show=False):
  '''
  Two-panel view of the reshaping (see module docstring). The injected power law is
  the black anchor; the cooled states run along a sequential ramp ordered by tt.
  '''
  tt_arr = 10.**np.asarray(logtt, dtype=float)
  colors = plt.cm.viridis(np.linspace(0., .85, len(tt_arr)))
  K0 = norm_plaw_distrib(gm0, gM0, p)

  # the tracks go on top, the distributions they sample below; the height ratio follows
  # the content, so the five decades of panel (b) keep the taller box
  fig, (axT, axD) = plt.subplots(2, 1, figsize=FIGSIZE,
      # hspace .24 left panel (a)'s xlabel only 3.2 px clear of panel (b) -- it bit.
      # .32 puts it 15.1 px clear while its own tick labels stay 1.4 px above it, so
      # it groups with the panel it names. The FIGURE HEIGHT is unchanged by this:
      # hspace redistributes space between panels, it does not add any.
      gridspec_kw=dict(height_ratios=[1., 1.45], hspace=.32))

  # (a) edges and break vs tt ------------------------------------------------------------
  # NOTE ON UNITS. This figure has no expansion, so the field has no reason to change
  # and t'_c is constant: tt = int dt'/t'_c IS t'/t'_c,i, exactly. The axis is labelled
  # that way because it is the physical reading; cooling_shape_adiabatic, which does
  # expand, has to convert between the two.
  tt = np.geomspace(1e-3/gM0, 1e2, 900)
  # THE COOLING REGIMES live here, on the pure-synchrotron panel, because that is what
  # defines them: t_M = 1/gma_M0 and t_m = 1/gma_m0 are where the two injected edges
  # burn, and in these units they need no reference C at all. MC is a NEIGHBOURHOOD of
  # t_m, taken as a factor MC_FAC either side. Colours are the house shape-class palette
  # (sweep_gammacm's per-spectrum table), RdBu from VSC red to VFC blue; its 'marginal'
  # #f7f7f7 is invisible as a tint, so MC gets a grey.
  t_M, t_m = 1./gM0, 1./gm0
  for lab, a_, b_, col in (('VSC', tt[0],       t_M,         '#b2182b'),
                           ('SC',  t_M,         t_m/MC_FAC,  '#ef8a62'),
                           ('MC',  t_m/MC_FAC,  t_m*MC_FAC,  '0.6'),
                           ('FC',  t_m*MC_FAC,  1.,          '#67a9cf'),
                           ('VFC', 1.,          tt[-1],      '#2166ac')):
    a_, b_ = max(a_, tt[0]), min(b_, tt[-1])
    if not a_ < b_:
      continue
    axT.axvspan(a_, b_, color=col, alpha=BAND_ALPHA, lw=0, zorder=0)
    axT.annotate(lab, (np.sqrt(a_*b_), .96), xycoords=('data', 'axes fraction'),
                 color=INK, fontsize=FS_ANN, ha='center', va='top',
                 bbox=dict(fc=_band_bg(col), ec='none', pad=1.5))
  # tracks continue BELOW gma = 1 rather than stopping there, matching the integrated
  # figures, which have always drawn their curves into the shaded sub-relativistic
  # region. The gma = 1 line still marks where the ultra-relativistic trajectory
  # stops being the physical one; it is a caveat on the curve, not a reason to hide
  # where the model says the population is heading.
  axT.loglog(tt, 1./tt, color=MUTED, ls='-.', lw=.9,
             label="$(t'/t'_{\\rm c,i})^{-1}$")
  axT.loglog(tt, gamma_synCooled(tt, gM0), color='k', lw=1.4, label='$\\gamma_\\mathrm{M}$')
  axT.loglog(tt, gamma_synCooled(tt, gm0), color='k', lw=1.1, ls='--',
             label='$\\gamma_\\mathrm{m}$')
  axT.axhline(1., color='crimson', ls=':', lw=.9, zorder=1)
  # the two knees, labelled along the bottom where nothing else runs; each is the
  # cooling time of the edge it burns, tilde{t}_M = 1/gma_M0 and tilde{t}_m = 1/gma_m0
  for v, lab in ((t_M, '$\\tilde{t}_M$'), (t_m, '$\\tilde{t}_m$')):
    axT.axvline(v, color=INK, ls=':', lw=.8, zorder=1)
    axT.annotate(lab, (v, .015), xycoords=('data', 'axes fraction'), color=INK,
                 fontsize=FS_ANN, ha='center', va='bottom',
                 bbox=dict(fc='w', ec='none', alpha=.85, pad=1.))
  axT.set_xlim(tt[0], tt[-1])
  axT.set_ylim(.15, 300.*gM0)    # two decades of empty top for the band labels
  axT.set_yticks(10.**np.arange(-2., np.log10(gM0) + 1., 2.))
  # this label sits in the GAP between the panels, so it is kept tight against the
  # axis it names -- at the default pads it drifts closer to the panel below and
  # reads as belonging to that one
  axT.set_xlabel("$t'/t'_{\\rm c,i}$", fontsize=FS_LAB, labelpad=1.)
  axT.tick_params(axis='x', pad=1.5)
  axT.set_ylabel('$\\gamma$', fontsize=FS_LAB)
  # ONE row ABOVE the panel. With the bands claiming the top strip and the knee labels
  # the floor, every in-panel placement now sits on one or the other -- lower centre put
  # the box straight over the tt_m label. A FIGURE legend, because bbox_inches='tight'
  # walks those; and no bbox_extra_artists, which would replace every axes' defaults.
  pT = axT.get_position()
  fig.legend(*axT.get_legend_handles_labels(), fontsize=FS_LEG, loc='lower center',
             bbox_to_anchor=(pT.x0 + .5*pT.width, pT.y1 + .008),
             bbox_transform=fig.transFigure, ncol=3, framealpha=1., handlelength=1.4,
             handletextpad=.5, borderpad=.3, columnspacing=.9)
  axT.grid(alpha=.25, lw=.4)

  # (b) the distributions ----------------------------------------------------------------
  # the injected law goes UNDER the cooled ones: the earliest of them tracks it almost
  # exactly, and on top the black would hide that curve entirely
  gma0, N0 = cooled_distrib(0., p, gm0, gM0)
  axD.loglog(gma0, N0, color='k', lw=1.4, zorder=2)
  edges = []
  for lt, tt_s, c in zip(logtt, tt_arr, colors):
    gma, N = cooled_distrib(tt_s, p, gm0, gM0)
    axD.loglog(gma, N, color=c, lw=1.2, solid_capstyle='round', label=f'{lt:.1f}',
               zorder=3)
    edges.append((gma[-1], N[-1]))
  # the burn-off front: locus of the top edge gma_M(tt), which slides down along 1/tt
  edges = np.array(edges)
  axD.plot(edges[:, 0], edges[:, 1], color=MUTED, lw=.7, ls='-', zorder=4)
  axD.scatter(edges[:, 0], edges[:, 1], s=9, facecolors=colors, edgecolors='w',
              linewidths=.5, zorder=6)

  # gma^-p guide, offset above the injected curve so it does not hide it
  gg = np.geomspace(gm0, gM0, 3)
  axD.loglog(gg, 12.*K0*gg**-p, color=MUTED, ls=':', lw=.9)
  axD.annotate('$\\propto\\gamma^{-p}$', (gg[1], 12.*K0*gg[1]**-p),
               textcoords='offset points', xytext=(3, 3), color=MUTED, fontsize=FS_ANN)
  # the front is labelled where it runs, no leader line
  axD.annotate("$\\propto(t'/t'_{\\rm c,i})^{-1}$", xy=(edges[3, 0], edges[3, 1]),
               xycoords='data',
               textcoords='offset points', xytext=(-4, -5), color=INK, fontsize=FS_ANN,
               ha='right', va='top')
  # the injected bounds, marked as their inverses are in panel (a); along the TOP here,
  # the bottom of this panel belongs to the legend
  for v, lab in ((gm0, '$\\gamma_{m,0}$'), (gM0, '$\\gamma_{M,0}$')):
    axD.axvline(v, color=INK, ls=':', lw=.8, zorder=1)
    axD.annotate(lab, (v, .985), xycoords=('data', 'axes fraction'), color=INK,
                 fontsize=FS_ANN, ha='center', va='top',
                 bbox=dict(fc='w', ec='none', alpha=.85, pad=1.))
  # gma = 1 marked here as in panel (a): the last sampled time lands right on it
  axD.axvline(1., color='crimson', ls=':', lw=.9, zorder=1)
  axD.set_xlim(.5*gamma_synCooled(tt_arr[-1], gm0), 2.*gM0)
  axD.set_ylim(1e-16, 1e4)
  axD.set_xlabel('$\\gamma$', fontsize=FS_LAB)
  axD.set_ylabel("$N(\\gamma,t')/N_{\\rm e}$", fontsize=FS_LAB)
  leg = axD.legend(fontsize=FS_LEG, ncol=2, loc='lower left', framealpha=.9,
                   title="$\\log_{10}(t'/t'_{\\rm c,i})$", handlelength=1.1,
                   labelspacing=.25,
                   columnspacing=.9, handletextpad=.5, borderpad=.4)
  leg.get_title().set_fontsize(FS_LEG)
  axD.grid(alpha=.25, lw=.4)

  for ax in (axT, axD):
    ax.tick_params(which='both', labelsize=FS_TICK)

  os.makedirs(outdir, exist_ok=True)
  path = os.path.join(outdir, fname)
  fig.savefig(path, dpi=300, bbox_inches='tight')
  print(f'saved {path}')
  if show:
    plt.show()
  return fig, (axT, axD)


def main(show=False):
  tt_arr = np.r_[0., 10.**np.asarray(LOGTT_SAMPLES)]
  print(f'number conservation: max |int N dgma - 1| = '
        f'{check_number_conservation(tt_arr):.2e}')
  gm, gM = gamma_synCooled(tt_arr[-1], GM0), gamma_synCooled(tt_arr[-1], GMA_M0)
  print(f'last sample tt=10^{LOGTT_SAMPLES[-1]:g}: '
        f'gma_m={gm:.3f}, gma_M={gM:.3f}, width gma_M/gma_m={gM/gm:.4f}')
  plot_cooling_shape(show=show)


if __name__ == '__main__':
  main()
