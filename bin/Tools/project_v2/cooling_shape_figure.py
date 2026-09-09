# -*- coding: utf-8 -*-
# @Author: acharlet

'''
How a power-law electron distribution is reshaped by synchrotron cooling.

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
is why the emission code truncates its gamma integral there (radiation_cooling.get_epnu:
number is conserved, so the clipped bound is the whole correction). The last
sampled time, tt = 1, lands exactly on that line: gma_M = 1/(1 + 1/gma_M0) -> 1, so on
these bounds tt = 1 IS the end of the model's validity, not a time it can be pushed
past.

Run:  python cooling_shape_figure.py
'''

import os
import numpy as np
import matplotlib.pyplot as plt

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
OUTDIR = os.path.join(GAMMA_dir, 'bin', 'Tools', 'figures', 'cooling_shape')

# recessive ink for every non-data mark (text never wears a series colour)
INK, MUTED = '0.25', '0.55'

# sized for ONE column of a two-column article: panels stacked, ~3.4 in wide, so
# every mark and every font is set for that final printed size, not rescaled after
FIGSIZE = (3.4, 6.4)
FS_LAB, FS_TICK, FS_ANN, FS_LEG = 9., 8., 8., 7.


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
      gridspec_kw=dict(height_ratios=[1., 1.5], hspace=.32))

  # (a) edges and break vs tt ------------------------------------------------------------
  tt = np.geomspace(1e-3/gM0, 1.5*tt_arr[-1], 800)
  # every track stops AT gma = 1: past it gamma_synCooled is no longer the physical
  # trajectory, so drawing it there would assert an evolution the model does not have
  cut = lambda y: np.where(y >= 1., y, np.nan)
  axT.loglog(tt, 1./tt, color=MUTED, ls='-.', lw=.9, label='$1/\\tilde{t}$')
  axT.loglog(tt, cut(gamma_synCooled(tt, gM0)), color='k', lw=1.4, label='$\\gamma_\\mathrm{M}$')
  axT.loglog(tt, cut(gamma_synCooled(tt, gm0)), color='k', lw=1.1, ls='--',
             label='$\\gamma_\\mathrm{m}$')
  axT.axhline(1., color='crimson', ls=':', lw=.9, zorder=1)
  # the two knees, labelled along the bottom where nothing else runs; each is the
  # cooling time of the edge it burns, tilde{t}_M = 1/gma_M0 and tilde{t}_m = 1/gma_m0
  for v, lab in ((1./gM0, '$\\tilde{t}_M$'), (1./gm0, '$\\tilde{t}_m$')):
    axT.axvline(v, color=INK, ls=':', lw=.8, zorder=1)
    axT.annotate(lab, (v, .015), xycoords=('data', 'axes fraction'), color=INK,
                 fontsize=FS_ANN, ha='center', va='bottom',
                 bbox=dict(fc='w', ec='none', alpha=.85, pad=1.))
  axT.set_xlim(tt[0], tt[-1])
  axT.set_ylim(.15, 3.*gM0)      # headroom under gma=1 for the knee labels
  axT.set_xlabel('$\\tilde{t}$', fontsize=FS_LAB)
  axT.set_ylabel('$\\gamma$', fontsize=FS_LAB)
  axT.legend(fontsize=FS_LEG, loc='upper right', framealpha=.9, handlelength=1.4,
             labelspacing=.25, handletextpad=.5, borderpad=.4)
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
  axD.annotate('$\\propto1/\\tilde{t}$', xy=(edges[3, 0], edges[3, 1]), xycoords='data',
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
  axD.set_ylabel('$N(\\gamma,\\tilde{t})/N_{\\rm e}$', fontsize=FS_LAB)
  leg = axD.legend(fontsize=FS_LEG, ncol=2, loc='lower left', framealpha=.9,
                   title='$\\log_{10}\\tilde{t}$', handlelength=1.1, labelspacing=.25,
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
