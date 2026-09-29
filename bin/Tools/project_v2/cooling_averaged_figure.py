# -*- coding: utf-8 -*-
# @Author: acharlet

'''
The TIME-AVERAGED electron distribution, pure synchrotron at constant t'_c.

The fifth figure of the family in figures/cooling_distributions, and the one that plots
a NORMALISED quantity:

    dNN_e/dgma_e (t') = 1/(t'-t'_0) INT_{t'_0}^{t'} dN_e/dgma_e (that; t'_0) dthat,

with t'_0 the initial collision time. At constant t'_c the clock is linear,

    tt = (t'-t'_0)/t'_c,0,

so dthat = t'_c,0 dtt and t'-t'_0 = t'_c,0 tt: the two factors cancel and the average is
simply the time INTEGRAL over the elapsed clock,

    dNN_e/dgma_e = N(gma_e; tt)/tt.

Everything in cooling_integrated_figure therefore carries over with a 1/tt prefactor:

    dNN_e/dgma_e = K0/[(p-1) tt] gma_e^-(p+1) [A^(p-1) - B^(p-1)],
    A = min(1, gma_e/gma_m,0),   B = max(1 - gma_e tt, gma_e/gma_M,0).

EVERY SLOPE IS UNCHANGED -- dividing by a gma-independent number leaves dln/dln gma
alone -- so the segments, the breaks and the hard edges are exactly the integrated
figure's. What the averaging changes is the amplitude of each, and that is the point.

WHY AVERAGE AT ALL. Two things follow that the integrated form does not have.

  1. IT IS A DISTRIBUTION. The sum rule INT N dgma = N_e tt becomes

         INT dNN_e/dgma_e dgma_e = N_e     at EVERY t',

     checked over eleven decades in tt (check_normalisation). The integrated form's
     norm grows with time; this one does not, so it is the quantity to compare against
     a measured distribution, and the time-averaged spectrum is this convolved with the
     single-electron kernel -- at constant B' the kernel carries no t' of its own.
  2. THE REGIME LABEL STOPS BEING AMBIGUOUS. The window ends at the current time, so
     gma_c = 1/tt is at once the break AND the label. cooling_integrated's
     sigma_end/C subtlety -- where C = 1e2 still gives a fast-cooled shape because the
     integral runs past t_dyn -- simply does not arise. The branch is gma_c vs gma_m,0.

THE TWO MIDDLES COME OUT CLEAN.

    slow, gma_m,0 < gma_e < gma_c :  K0 gma_e^-p          the INJECTED law, exactly
    fast, gma_c < gma_e < gma_m,0 :  N_e gma_c gma_e^-2   gma_c = 1/tt, bounds gone

  The first is a real statement, not a triviality: while nothing has cooled the time
  average IS the injected distribution, with no t' dependence at all, so the average
  carries no information about elapsed time until gma_c drops through gma_m,0. Measured
  ratio 0.9995 well below gma_c, drifting to 0.975 by gma_e/gma_c = 0.1 through the
  usual (p-2)/2 gma_e/gma_c tilt (check_slow_limit).

  The second holds to eight digits (check_fast_limit) and has no trace of gma_m,0 or
  gma_M,0 in it: it is the dwell time dtt = dgma/gma^2 times the WHOLE population, and
  gma_c enters only as the amplitude.

THE EVOLUTION, which is what the figure draws. The average starts as the injected power
law, grows a gma_e^-2 foot as gma_c descends through gma_M,0, and ends as a narrowing,
GROWING spike at gma_c: measured peak at gma_e = 1/tt and height N_e tt = N_e/gma_c,
both to three decimals over tt = 10 .. 1e3 (check_peak). Height x width ~ N_e tt x 1/tt
is the normalisation showing through -- the whole population piling into a shrinking
range of gma_e. That is the mono-energetic collapse, seen in a normalised quantity.

THE MARKERS are the two running edges gma_m(tt) and gma_M(tt), the same landmarks the
segments table uses. gma_M is the break between the cooled middle and the gma^-(p+1)
tail and tends to gma_c = 1/tt once the top edge has burnt; gma_m is the cut-off, where
the curve ends.

VALIDITY. Below gma_e = 1 the ultra-relativistic trajectory is not the physical one
(shaded in both panels). tt = 1 puts gma_c there exactly, which is why the samples stop
at tt = 1.

SUBSCRIPTS. Elsewhere in the family `i` is the injection epoch OF A GIVEN FLUID
ELEMENT -- gma_m,i, t'_c,i. The average here is a SHELL-level quantity, so its initial
values carry 0 instead: t'_0, t'_c,0, gma_m,0, gma_M,0, and tt measured from t'_0. The
integrated figures keep `i`; the two are the same functions of different reference
epochs, not different physics.

Run:  python cooling_averaged_figure.py
'''

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from cooling_distribution import gamma_synCooled, norm_plaw_distrib
from cooling_integrated_figure import (P_SYN, GM0, GMA_M0, OUTDIR, INK, MUTED, FIGSIZE,
    FS_LAB, FS_TICK, FS_ANN, FS_LEG, GMA_LABEL, N_integrated, log_slope)
from cooling_shape_figure import LOGTT_SAMPLES

# --- defaults -------------------------------------------------------------------------
# the SAME sampled times the shape figure uses, so the two are read together: gma_c runs
# from gma_M,0 (nothing has cooled) down to 1 (the end of the model's validity)
LOGTT = LOGTT_SAMPLES
NG = 3000                   # points per curve, log-spaced over the support
N_LO = 1e-14                # floor of the distribution panel; the gma_m ticks sit on it
FNAME = 'cooling_averaged.png'
TT_LABEL = '$\\log_{10}\\tilde{t}$'


def N_averaged(gma, tt, p=P_SYN, gm0=GM0, gM0=GMA_M0):
  '''
  The time-averaged distribution: the integrated one over the elapsed clock. Not a
  second implementation -- the closed form is cooling_integrated_figure's.
  '''
  return N_integrated(gma, tt, p, gm0, gM0)/tt


def averaged_distrib(tt, p=P_SYN, gm0=GM0, gM0=GMA_M0, Ng=NG):
  '(gma, dNN/dgma) over the support [gma_m(tt), gma_M,0].'
  gma = np.geomspace(gamma_synCooled(tt, gm0), gM0, Ng)
  return gma, N_averaged(gma, tt, p, gm0, gM0)


# --- validation -----------------------------------------------------------------------
def check_normalisation(tts=(1e-9, 1e-6, 1e-3, 1e-1, 1., 1e2), p=P_SYN, gm0=GM0,
    gM0=GMA_M0, Ng=2000001):
  'max |int (N/tt) dgma - 1| over tts. Unlike the integral, the average is normalised.'
  dev = 0.
  for tt in tts:
    g = np.geomspace(gamma_synCooled(tt, gm0)*(1. - 1e-12), gM0, Ng)
    dev = max(dev, abs(np.trapezoid(N_averaged(g, tt, p, gm0, gM0), g) - 1.))
  return dev


def check_slow_limit(tt=1e-6, p=P_SYN, gm0=GM0, gM0=GMA_M0, hi=.03):
  '''
  Slow cooling: the average IS the injected law. Returns the worst ratio over
  gma_m,0 .. hi*gma_c, staying clear of the (p-2)/2 gma/gma_c tilt near the break.
  '''
  K0 = norm_plaw_distrib(gm0, gM0, p)
  g = np.geomspace(2.*gm0, hi/tt, 40)
  return float(np.max(np.abs(N_averaged(g, tt, p, gm0, gM0)/(K0*g**-p) - 1.)))


def check_fast_limit(tt=1e2, p=P_SYN, gm0=GM0, gM0=GMA_M0):
  'Fast cooling: the middle is N_e gma_c gma^-2 exactly, with the injected bounds gone.'
  g = np.geomspace(3./tt, gm0/3., 40)
  return float(np.max(np.abs(N_averaged(g, tt, p, gm0, gM0)/(g**-2./tt) - 1.)))


def check_peak(tts=(1e1, 1e2, 1e3), p=P_SYN, gm0=GM0, gM0=GMA_M0, Ng=400001):
  '''
  Deep fast cooling: the peak sits at gma_c = 1/tt and reaches N_e/gma_c. Returns
  [(tt, gma_peak*tt, height/tt)], both -> 1.
  '''
  out = []
  for tt in tts:
    g = np.geomspace(gamma_synCooled(tt, gm0)*(1. + 1e-7), gM0, Ng)
    a = N_averaged(g, tt, p, gm0, gM0)
    k = int(np.argmax(a))
    out.append((tt, float(g[k]*tt), float(a[k]/tt)))
  return out


# --- the figure -------------------------------------------------------------------------
def plot_averaged(p=P_SYN, gm0=GM0, gM0=GMA_M0, logtt=LOGTT, outdir=OUTDIR,
    fname=FNAME, show=False):
  '''
  Two-panel view, the family's layout for a distribution against gma_e: the averaged
  distributions over the sampled times, and the local slope underneath where the
  gma^-p / gma^-2 / gma^-(p+1) segments can be read off. The injected law is the black
  anchor -- the early curves lie ON it, which is the slow-cooling statement.
  '''
  logtt = np.asarray(logtt, dtype=float)
  norm = plt.Normalize(vmin=logtt.min(), vmax=logtt.max())
  # the ramp is TRUNCATED at .85 to keep clear of viridis's brightest yellow, and the
  # bar is built from the same truncated map -- taking the lines from one and the bar
  # from the other leaves a colour key that does not match the curves
  cmap = mcolors.LinearSegmentedColormap.from_list(
      'viridis85', plt.cm.viridis(np.linspace(0., .85, 256)))
  colors = cmap(norm(logtt))
  sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
  K0 = norm_plaw_distrib(gm0, gM0, p)

  fig, (axN, axS) = plt.subplots(2, 1, figsize=FIGSIZE, sharex=True,
      gridspec_kw=dict(height_ratios=[1.9, 1.], hspace=.08))

  # the injected law UNDER everything: at the earliest times the curves sit on it, and
  # drawn on top the black would hide exactly the agreement the panel is making
  # WIDER than the curves, so where one lies on it the black still shows either side.
  # At lw 1.4 the earliest curve covered it exactly and the agreement was invisible.
  gg = np.geomspace(gm0, gM0, 400)
  axN.loglog(gg, K0*gg**-p, color='k', lw=2.6, zorder=1, solid_capstyle='butt',
             label='injected')
  lo_g, hi_N = np.inf, 0.
  for lt, c in zip(logtt, colors):
    tt = 10.**lt
    g, a = averaged_distrib(tt, p, gm0, gM0)
    axN.loglog(g, a, color=c, lw=1.1, solid_capstyle='round', zorder=3)
    axS.semilogx(g, log_slope(g, a), color=c, lw=1.1, zorder=3)
    lo_g, hi_N = min(lo_g, float(g[0])), max(hi_N, float(np.max(a)))
    # THE TWO RUNNING EDGES, which are the table's landmarks. They are marked
    # differently because they sit differently on the curve: gma_M(tt) is the BREAK
    # between the cooled middle and the gma^-(p+1) tail, an interior point with a finite
    # value, so it goes on the curve; gma_m(tt) is where the curve ENDS and NN vanishes
    # there, so a point on the curve would be at zero -- it is ticked on the floor
    # instead. gma_M -> 1/tt = gma_c once the top edge has burnt, which is what the
    # single gma_c marker used to show.
    gM_t = 1./(tt + 1./gM0)
    axN.scatter([gM_t], [N_averaged(gM_t, tt, p, gm0, gM0)], s=11, facecolors='none',
                edgecolors=INK, linewidths=.7, zorder=6)
    # an INK triangle on the floor, not a coloured tick: the curve already plunges
    # to the floor at gma_m, so a mark in the curve's own colour is invisible on it
    axN.scatter([g[0]], [N_LO*2.2], marker='^', s=13, facecolors='none',
                edgecolors=INK, linewidths=.7, zorder=7)

  # the expected indices, on the right-hand spine rather than as in-panel labels
  levels = ((-2., '$-2$'), (-p, '$-p$'), (-(p+1.), '$-(p+1)$'))
  for lev, _ in levels:
    axS.axhline(lev, color=MUTED, ls='--', lw=.7, zorder=1)
  axS.set_ylim(-(p + 2.6), .4)

  for ax in (axN, axS):
    ax.axvspan(1e-30, 1., color='crimson', alpha=.07, lw=0, zorder=0)
    for v in (gm0, gM0):
      ax.axvline(v, color=INK, ls=':', lw=.8, zorder=1)
    ax.axvline(1., color='crimson', ls=':', lw=.9, zorder=1)
    ax.grid(alpha=.25, lw=.4)
    ax.tick_params(which='both', labelsize=FS_TICK)
  axN.set_xlim(.3*lo_g, 4.*gM0)
  axN.set_ylim(N_LO, 10.*hi_N)
  axN.set_ylabel('${\\rm d}\\mathcal{N}_{\\rm e}/{\\rm d}\\gamma_{\\rm e}$',
                 fontsize=FS_LAB)
  axS.set_xlabel(GMA_LABEL, fontsize=FS_LAB)
  axS.set_ylabel("${\\rm d}\\ln({\\rm d}\\mathcal{N}_{\\rm e}/{\\rm d}\\gamma_{\\rm e})"
                 "/{\\rm d}\\ln\\gamma_{\\rm e}$", fontsize=FS_LAB)
  axR = axS.twinx()
  axR.set_ylim(axS.get_ylim())
  axR.set_yticks([lev for lev, _ in levels])
  axR.set_yticklabels([lab for _, lab in levels])
  axR.tick_params(axis='y', labelsize=FS_ANN, length=2.5, pad=1.5, colors=INK)
  axR.grid(False)

  for v, lab in ((gm0, '$\\gamma_{\\mathrm{m},\\!0}$'),
                 (gM0, '$\\gamma_{\\mathrm{M},\\!0}$')):
    axN.annotate(lab, (v, .985), xycoords=('data', 'axes fraction'), color=INK,
                 fontsize=FS_ANN, ha='center', va='top',
                 bbox=dict(fc='w', ec='none', alpha=.85, pad=1.))
  axS.annotate('$\\gamma_{\\rm e}=1$', (1., .10), xycoords=('data', 'axes fraction'),
               textcoords='offset points', xytext=(3, 0), color='crimson',
               fontsize=FS_ANN, ha='left', va='bottom',
               bbox=dict(fc='w', ec='none', alpha=.85, pad=1.))
  axN.scatter([], [], s=11, facecolors='none', edgecolors=INK, linewidths=.7,
              label='$\\gamma_\\mathrm{M}(\\tilde{t}\\,)$')
  axN.scatter([], [], marker='^', s=13, facecolors='none', edgecolors=INK,
              linewidths=.7, label='$\\gamma_\\mathrm{m}(\\tilde{t}\\,)$')
  axN.legend(fontsize=FS_LEG, loc='lower left', framealpha=.9, handletextpad=.4,
             borderpad=.4, labelspacing=.3)

  # the colour bar beside the TOP panel only, on an explicit cax: stealing space from
  # one of two stacked shared-x panels leaves them different widths
  pN = axN.get_position()
  cax = fig.add_axes([pN.x1 + .015, pN.y0, .022, pN.height])
  cb = fig.colorbar(sm, cax=cax)
  cb.set_label(TT_LABEL, fontsize=FS_LAB)
  cb.ax.tick_params(labelsize=FS_TICK)

  os.makedirs(outdir, exist_ok=True)
  path = os.path.join(outdir, fname)
  fig.savefig(path, dpi=300, bbox_inches='tight')
  print(f'saved {path}')
  if show:
    plt.show()
  return fig, (axN, axS)


def main(show=False):
  print(f'normalisation  : max |int (N/tt) dgma_e - 1| = {check_normalisation():.2e}'
        '   (the integral itself grows as tt; the average does not)')
  print(f'slow limit     : max |avg/(K0 gma^-p) - 1| = {check_slow_limit():.2e}'
        '   -- the average IS the injected law')
  print(f'fast limit     : max |avg/(N_e gma_c gma^-2) - 1| = {check_fast_limit():.2e}')
  print('peak, deep fast cooling (both ratios -> 1):')
  for tt, xr, yr in check_peak():
    print(f'    tt={tt:7.0e}:  gma_peak*tt = {xr:.4f}   height/(N_e tt) = {yr:.4f}')
  plot_averaged(show=show)


if __name__ == '__main__':
  main()
