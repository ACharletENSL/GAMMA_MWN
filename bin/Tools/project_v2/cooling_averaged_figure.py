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

THE MARKER is gma_M(tt), the running top edge, which is the break between the cooled
middle and the gma^-(p+1) tail -- the same landmark the segments table uses. It is drawn
on the distribution panel only, where the amplitude turns over, and tends to
gma_c = 1/tt once the top edge has burnt. The other landmark, gma_m(tt), needs no marker:
the curve ends there.

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
from fractions import Fraction
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from cooling_distribution import gamma_synCooled, norm_plaw_distrib
from cooling_integrated_figure import (P_SYN, GM0, GMA_M0, OUTDIR, INK, MUTED, FIGSIZE,
    FS_LAB, FS_TICK, FS_ANN, FS_LEG, GMA_LABEL, N_integrated, log_slope, tt_dyn_of_C)
from cooling_integrated_adiabatic import Q_C, _exps, S_of_sigma
from cooling_shape_figure import LOGTT_SAMPLES

# --- defaults -------------------------------------------------------------------------
# the SAME sampled times the shape figure uses, so the two are read together: gma_c runs
# from gma_M,0 (nothing has cooled) down to 1 (the end of the model's validity)
LOGTT = LOGTT_SAMPLES
NG = 3000                   # points per curve, log-spaced over the support
N_LO = 1e-14                # floor of the distribution panel; the gma_m ticks sit on it
FNAME = 'cooling_averaged.png'           # the three cases together
FN_STEADY = 'cooling_averaged_steady.png'   # the steady case on its own
# the decaying-field panels name the clock they run on; q is rendered from Q_C so the
# titles follow the constant rather than being written in
_QF = Fraction(Q_C).limit_denominator(100)
_QT = (f'{_QF.numerator}' if _QF.denominator == 1
       else f'{_QF.numerator}/{_QF.denominator}')
# (title, log10 C, q). (a) and (b) share kappa and differ ONLY in the clock; (b) and
# (c) share the clock and differ only in the regime
CASES = (f"$t'_{{\\rm c}}=\\,$cst", -2., 0.), \
        (f"$t'_{{\\rm c}}\\propto t'^{{{_QT}}}$ \u2013 FC", -2., Q_C), \
        (f"$t'_{{\\rm c}}\\propto t'^{{{_QT}}}$ \u2013 SC", 2., Q_C)
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


# --- the steady case on its own -------------------------------------------------------------------------
def plot_averaged_steady(p=P_SYN, gm0=GM0, gM0=GMA_M0, logtt=LOGTT, outdir=OUTDIR,
    fname=FN_STEADY, show=False):
  '''
  The STEADY case on its own, which is the form the main text derives: with t'_c
  constant the result depends on tt alone, so this figure needs no kappa and is sampled
  in tt directly rather than in sigma. Two panels, the averaged distributions over the
  sampled times, and the local slope underneath where the
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
    sl = log_slope(g, a)
    axN.loglog(g, a, color=c, lw=1.1, solid_capstyle='round', zorder=3)
    axS.semilogx(g, sl, color=c, lw=1.1, zorder=3)
    lo_g, hi_N = min(lo_g, float(g[0])), max(hi_N, float(np.max(a)))
    # gma_M(tt), the BREAK between the cooled middle and the gma^-(p+1) tail, where
    # the amplitude turns over. It tends to gma_c = 1/tt once the top edge has burnt.
    # Marked on the DISTRIBUTION panel only: the slope panel already shows the break as
    # a step, and a marker on a step lands wherever the numerical derivative smears it.
    # gma_m(tt) is not marked either -- the curve already ENDS there, visibly.
    gM_t = 1./(tt + 1./gM0)
    axN.scatter([gM_t], [N_averaged(gM_t, tt, p, gm0, gM0)], s=11, facecolors='none',
                edgecolors=INK, linewidths=.7, zorder=6)

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
  axN.legend(fontsize=FS_LEG, loc='lower left', framealpha=.9, handletextpad=.4,
             borderpad=.4, labelspacing=.3)

  # the colour bar beside the TOP panel only, on an explicit cax: stealing space from
  # one of two stacked shared-x panels leaves them different widths
  pN = axN.get_position()
  cax = fig.add_axes([pN.x1 + .015, pN.y0, .022, pN.height])
  cb = fig.colorbar(sm, cax=cax)
  cb.set_label(TT_LABEL, fontsize=FS_LAB)
  cb.ax.tick_params(labelsize=FS_TICK)

  return _save_fig(fig, (axN, axS), outdir, fname, show)


# --- the decaying field: t'_c ~ tau^q ----------------------------------------------------
LOGSIG = (-3., -2., -1., 0., 1., 2., 3.)   # sampled log10 sigma, sigma = t'/t'_0 - 1
LOGC_FS = (-2., 2.)                        # the fast- and slow-cooling columns
NG_D = 500                                 # points per curve
DEC_SPAN = 1e-21                           # how far below the tallest peak the panels
                                           # reach. The three span 20-29 decades of
                                           # their own, so a fixed floor cut the
                                           # slow-cooling column off half way across
_GL_X, _GL_W = np.polynomial.legendre.leggauss(64)


def u_of_sigma(sigma, kap, q=Q_C):
  '''
  The normalised clock u = int dt'/t'_c with t'_c = t'_c,0 tau^q, tau = 1 + sigma. With
  no adiabatic drag A = 1, so the burn IS the clock and S_of_sigma at a_rho = 0 gives it:
  u = (kappa/s)(1 - tau^-s), s = q - 1. It SATURATES at u_inf = kappa/s.
  '''
  return S_of_sigma(sigma, kap, 0., q)


def sigma_of_u(u, kap, q=Q_C):
  'Inverse of u_of_sigma, in closed form: tau = (1 - s u/kappa)^(-1/s).'
  _, s = _exps(0., q)
  return (1. - s*np.asarray(u, dtype=float)/kap)**(-1./s) - 1.


def N_instant(gma, u, p=P_SYN, gm0=GM0, gM0=GMA_M0):
  "K0 gma^-p (1 - gma u)^(p-2) on [gma_m(u), gma_M(u)], zero off it."
  gma = np.asarray(gma, dtype=float)
  gm, gM = gm0/(1. + gm0*u), gM0/(1. + gM0*u)
  return np.where((gma >= gm) & (gma <= gM),
                  norm_plaw_distrib(gm0, gM0, p)*gma**-p*np.abs(1. - gma*u)**(p - 2.), 0.)


def N_frozen(gma, kap, p=P_SYN, gm0=GM0, gM0=GMA_M0, q=Q_C):
  'The state the distribution freezes into, N_instant at the saturated clock u_inf.'
  return N_instant(gma, kap/_exps(0., q)[1], p, gm0, gM0)


def N_averaged_decay(gma, sigma, kap, p=P_SYN, gm0=GM0, gM0=GMA_M0, q=Q_C):
  '''
  The time average AS DEFINED, over physical time:

      dNN_e/dgma_e = 1/(t'-t'_0) INT dN_e/dgma_e dthat = 1/sigma INT_0^sigma ... dshat,

  since dthat = t'_0 dshat. With a decaying field this is NOT N/u: dthat = t'_c,0 tau^q du,
  so the average over t' is no longer the average over the clock, and the weight tau^q
  diverges as u -> u_inf. There is no closed form (two binomials in the integrand give a
  2F1), so it is quadrature -- but over the SOLVED window, never the full range.

  At fixed gma_e the population sweeps past once, so the integrand is supported on
  u in [max(0, 1/gma - 1/gma_m,0), min(u(sigma), 1/gma - 1/gma_M,0)], which maps to a
  sigma window through sigma_of_u. Handed [0, sigma] instead, an adaptive rule misses
  that window whenever it is a small fraction of the range and silently returns 0 --
  the same trap cooling_integrated_figure documents. Inside the window the integrand is
  smooth and bounded (it does not vanish at either end), so fixed-order Gauss-Legendre
  is exact to ~5e-6 against scipy.quad on the same window.
  '''
  gma = np.atleast_1d(np.asarray(gma, dtype=float))
  out = np.zeros_like(gma)
  u_end = float(u_of_sigma(sigma, kap, q))
  ua = np.maximum(0., 1./gma - 1./gm0)
  ub = np.minimum(u_end, 1./gma - 1./gM0)
  ok = ua < ub
  if not ok.any():
    return out
  a, b = sigma_of_u(ua[ok], kap, q), sigma_of_u(ub[ok], kap, q)
  sh = .5*(b - a)[:, None]*_GL_X[None, :] + .5*(a + b)[:, None]
  g = gma[ok][:, None]
  integ = (norm_plaw_distrib(gm0, gM0, p)*g**-p
           * np.abs(1. - g*u_of_sigma(sh, kap, q))**(p - 2.))
  out[ok] = .5*(b - a)*np.sum(_GL_W[None, :]*integ, axis=1)/sigma
  return out


def check_decay_norm(kap, sigmas=(1e-3, 1e-1, 1e1, 1e3), p=P_SYN, gm0=GM0, gM0=GMA_M0,
    q=Q_C, Ng=400001):
  '''
  int (dNN/dgma) dgma = N_e still, at every t': the average is over a set of instantaneous
  distributions each carrying N_e, so exchanging the integrations gives N_e whatever the
  weight. Returns the worst deviation.
  '''
  dev = 0.
  for sg in sigmas:
    lo = gm0/(1. + gm0*float(u_of_sigma(sg, kap, q)))
    g = np.geomspace(lo*(1. - 1e-12), gM0, Ng)
    dev = max(dev, abs(np.trapezoid(N_averaged_decay(g, sg, kap, p, gm0, gM0, q), g) - 1.))
  return dev


def check_decay_freeze(kap, sigmas=(1e1, 1e2, 1e3), p=P_SYN, gm0=GM0, gM0=GMA_M0, q=Q_C):
  '''
  Inside the frozen support the average tends to the FROZEN distribution; the residual
  falls as 1/sigma. Returns [(sigma, worst |avg/N_frozen - 1|)].
  '''
  _, s = _exps(0., q)
  u_i = kap/s
  lo, hi = 1./(u_i + 1./gm0), 1./(u_i + 1./gM0)
  g = np.geomspace(lo*1.02, hi*.98, 9)
  ref = N_frozen(g, kap, p, gm0, gM0, q)
  return [(sg, float(np.max(np.abs(N_averaged_decay(g, sg, kap, p, gm0, gM0, q)/ref - 1.))))
          for sg in sigmas]


def plot_averaged(p=P_SYN, gm0=GM0, gM0=GMA_M0, logsig=LOGSIG, cases=None,
    outdir=OUTDIR, fname=FNAME, show=False):
  '''
  The whole time-averaged family in one figure: three cases across, distributions above
  and their local slope below, on one log10 sigma colour axis.

  Column (a) is the constant-t'_c case and (b) the SAME system with the field decaying,
  so the pair isolates the CLOCK -- same kappa, same sampled times, nothing else
  different. (b) against (c) then isolates the regime. Panels (b) and (c) carry the
  frozen state as a dashed line, which (a) has none of: with a constant field the clock
  never saturates and the average never settles.
  '''
  cases = CASES if cases is None else cases
  logsig = np.asarray(logsig, dtype=float)
  norm = plt.Normalize(vmin=logsig.min(), vmax=logsig.max())
  cmap = mcolors.LinearSegmentedColormap.from_list(
      'viridis85', plt.cm.viridis(np.linspace(0., .85, 256)))
  colors, sm = cmap(norm(logsig)), plt.cm.ScalarMappable(cmap=cmap, norm=norm)
  K0 = norm_plaw_distrib(gm0, gM0, p)

  fig, axs = plt.subplots(2, len(cases), figsize=(7.1, 4.8), sharex=True, sharey='row',
                          squeeze=False,
                          gridspec_kw=dict(height_ratios=[1.9, 1.], hspace=.08,
                                           wspace=.06))
  lo_g, hi_N = np.inf, 0.
  for k, (lab, lc, q) in enumerate(cases):
    axN, axS = axs[0, k], axs[1, k]
    kap = tt_dyn_of_C(lc, gm0)
    _, s = _exps(0., q)
    gg = np.geomspace(gm0, gM0, 400)
    axN.loglog(gg, K0*gg**-p, color='k', lw=2.6, zorder=1, solid_capstyle='butt')
    for ls, c in zip(logsig, colors):
      sg = 10.**ls
      u = float(u_of_sigma(sg, kap, q))
      g = np.geomspace(gm0/(1. + gm0*u), gM0, NG_D)
      a = N_averaged_decay(g, sg, kap, p, gm0, gM0, q)
      m = a > 0.
      if not m.any():
        continue
      axN.loglog(g[m], a[m], color=c, lw=1.1, zorder=3)
      axS.semilogx(g[m], log_slope(g[m], a[m]), color=c, lw=1.1, zorder=3)
      lo_g, hi_N = min(lo_g, float(g[m][0])), max(hi_N, float(np.max(a)))
      # gma_M(u), the break between the cooled middle and the gma^-(p+1) tail
      gM_u = 1./(u + 1./gM0)
      axN.scatter([gM_u], N_averaged_decay(gM_u, sg, kap, p, gm0, gM0, q), s=11,
                  facecolors='none', edgecolors=INK, linewidths=.7, zorder=6)
    if s > 0.:        # the burn saturates: draw the state everything is heading for
      u_i = kap/s
      gf = np.geomspace(1./(u_i + 1./gm0), 1./(u_i + 1./gM0), 600)
      axN.loglog(gf, N_instant(gf, u_i, p, gm0, gM0), color='crimson', ls='--', lw=1.1,
                 zorder=5)
    axN.set_title(lab, fontsize=FS_LAB, pad=3.)
    for ax in (axN, axS):
      ax.axvspan(1e-30, 1., color='crimson', alpha=.07, lw=0, zorder=0)
      for v in (gm0, gM0):
        ax.axvline(v, color=INK, ls=':', lw=.8, zorder=1)
      ax.axvline(1., color='crimson', ls=':', lw=.9, zorder=1)
      ax.grid(alpha=.25, lw=.4)
      ax.tick_params(which='both', labelsize=FS_TICK)
    axS.set_xlabel(GMA_LABEL, fontsize=FS_LAB)
    for lev in (-2., -p, -(p+1.)):
      axS.axhline(lev, color=MUTED, ls='--', lw=.7, zorder=1)
    axS.set_ylim(-(p + 2.6), .4)
  axR = axs[1, -1].twinx()
  axR.set_ylim(axs[1, -1].get_ylim())
  axR.set_yticks([-2., -p, -(p+1.)])
  axR.set_yticklabels(['$-2$', '$-p$', '$-(p+1)$'])
  axR.tick_params(axis='y', labelsize=FS_ANN, length=2.5, pad=1.5, colors=INK)
  axR.grid(False)
  for k in range(len(cases)):
    axs[0, k].set_xlim(.3*lo_g, 4.*gM0)
    axs[0, k].set_ylim(10.*hi_N*DEC_SPAN, 10.*hi_N)
  axs[0, 0].set_ylabel('${\\rm d}\\mathcal{N}_{\\rm e}/{\\rm d}\\gamma_{\\rm e}$',
                       fontsize=FS_LAB)
  axs[1, 0].set_ylabel("${\\rm d}\\ln({\\rm d}\\mathcal{N}_{\\rm e}/{\\rm d}\\gamma_"
                       "{\\rm e})/{\\rm d}\\ln\\gamma_{\\rm e}$", fontsize=FS_LAB)
  # In the LAST panel: the slow-cooling curves only start at gma_m,0, so its upper left
  # is the one large empty patch in the figure. Proxy handles, because the entries are
  # spread over panels -- the frozen line is not in every one.
  axs[0, -1].legend(handles=[
      plt.Line2D([], [], color='k', lw=2.6, label='injected'),
      plt.Line2D([], [], color='crimson', ls='--', lw=1.1, label='frozen'),
      plt.Line2D([], [], color=INK, lw=0, marker='o', mfc='none', ms=3.5,
                 label='$\\gamma_\\mathrm{M}$')],
      fontsize=FS_LEG, loc='upper left', framealpha=.9, handletextpad=.3,
      borderpad=.3, ncol=3, columnspacing=.9, handlelength=1.3)
  p1 = axs[0, -1].get_position()
  cax = fig.add_axes([p1.x1 + .014, p1.y0, .016, p1.height])
  cb = fig.colorbar(sm, cax=cax)
  cb.set_label("$\\log_{10}[(t'-t'_0)/t'_0]$", fontsize=FS_LAB)
  cb.ax.tick_params(labelsize=FS_TICK)
  return _save_fig(fig, axs, outdir, fname, show)


# --- the steady window, integrated rather than averaged -----------------------------------
LOGSIG_S = (-3., -2.5, -2., -1.5, -1., -.5, 0.)   # t' in [t'_0, 2t'_0], so sigma <= 1
FN_INT_STEADY = 'cooling_integrated_steady.png'
STEADY_CASES = ((f"$t'_{{\\rm c}}=\\,$cst – FC", -2.),
                (f"$t'_{{\\rm c}}=\\,$cst – SC", 2.))


def plot_integrated_steady(p=P_SYN, gm0=GM0, gM0=GMA_M0, logsig=LOGSIG_S,
    cases=None, outdir=OUTDIR, fname=FN_INT_STEADY, show=False):
  '''
  The time-INTEGRATED distribution over the steady window, t' in [t'_0, 2t'_0].

  Same system and same sampling as the averaged figure, without the 1/(t'-t'_0): the
  curves are tt x dNN_e/dgma_e, which is cooling_integrated_figure's N(gma_e;tt) with
  tt = kappa*sigma. Dropping the normalisation costs the property that made the average
  a distribution -- int dgma_e = N_e tt grows with time rather than staying at N_e -- so
  the curves FAN instead of converging, each slow-cooling one lying on tt K0 gma_e^-p.
  The shapes are identical to the average's at the same tt; only the amplitudes move.

  sigma = (t'-t'_0)/t'_0 needs a kappa = t'_0/t'_c,0 to become tt, and kappa IS the
  regime: C = 1/(kappa gma_m,0). Hence two columns rather than one, fast and slow
  cooling, the same pair the rest of the family uses.
  '''
  cases = STEADY_CASES if cases is None else cases
  logsig = np.asarray(logsig, dtype=float)
  norm = plt.Normalize(vmin=logsig.min(), vmax=logsig.max())
  cmap = mcolors.LinearSegmentedColormap.from_list(
      'viridis85', plt.cm.viridis(np.linspace(0., .85, 256)))
  colors, sm = cmap(norm(logsig)), plt.cm.ScalarMappable(cmap=cmap, norm=norm)
  K0 = norm_plaw_distrib(gm0, gM0, p)

  fig, axs = plt.subplots(2, len(cases), figsize=(6.4, 4.8), sharex=True, sharey='row',
                          squeeze=False,
                          gridspec_kw=dict(height_ratios=[1.9, 1.], hspace=.08,
                                           wspace=.06))
  lo_g, hi_N, lo_N = np.inf, 0., np.inf
  for k, (lab, lc) in enumerate(cases):
    axN, axS = axs[0, k], axs[1, k]
    kap = tt_dyn_of_C(lc, gm0)
    for ls, c in zip(logsig, colors):
      tt = kap*10.**ls                       # tt = kappa sigma, the integration limit
      g, a = averaged_distrib(tt, p, gm0, gM0)
      a = a*tt                               # average -> integral
      m = a > 0.
      axN.loglog(g[m], a[m], color=c, lw=1.1, zorder=3)
      axS.semilogx(g[m], log_slope(g[m], a[m]), color=c, lw=1.1, zorder=3)
      lo_g, hi_N = min(lo_g, float(g[m][0])), max(hi_N, float(np.max(a)))
      lo_N = min(lo_N, float(np.min(a[m])))
      gM_t = 1./(tt + 1./gM0)
      axN.scatter([gM_t], [N_averaged(gM_t, tt, p, gm0, gM0)*tt], s=11,
                  facecolors='none', edgecolors=INK, linewidths=.7, zorder=6)
    # a SHAPE reference, offset so it claims no amplitude: the slow-cooling curves are
    # tt K0 gma^-p and so are parallel to it, each at its own height
    gg = np.geomspace(gm0, gM0, 3)
    axN.loglog(gg, 12.*K0*gg**-p*kap*10.**logsig.max(), color=MUTED, ls=':', lw=.9,
               zorder=2)
    axN.annotate('$\\propto\\gamma_{\\rm e}^{-p}$',
                 (gg[1], 12.*K0*gg[1]**-p*kap*10.**logsig.max()),
                 textcoords='offset points', xytext=(3, 3), color=MUTED, fontsize=FS_ANN)
    axN.set_title(lab, fontsize=FS_LAB, pad=3.)
    for ax in (axN, axS):
      ax.axvspan(1e-30, 1., color='crimson', alpha=.07, lw=0, zorder=0)
      for v in (gm0, gM0):
        ax.axvline(v, color=INK, ls=':', lw=.8, zorder=1)
      ax.axvline(1., color='crimson', ls=':', lw=.9, zorder=1)
      ax.grid(alpha=.25, lw=.4)
      ax.tick_params(which='both', labelsize=FS_TICK)
    axS.set_xlabel(GMA_LABEL, fontsize=FS_LAB)
    for lev in (-2., -p, -(p+1.)):
      axS.axhline(lev, color=MUTED, ls='--', lw=.7, zorder=1)
    axS.set_ylim(-(p + 2.6), .4)
  axR = axs[1, -1].twinx()
  axR.set_ylim(axs[1, -1].get_ylim())
  axR.set_yticks([-2., -p, -(p+1.)])
  axR.set_yticklabels(['$-2$', '$-p$', '$-(p+1)$'])
  axR.tick_params(axis='y', labelsize=FS_ANN, length=2.5, pad=1.5, colors=INK)
  axR.grid(False)
  for k in range(len(cases)):
    axs[0, k].set_xlim(.3*lo_g, 4.*gM0)
    # the floor follows the DATA here: the tails run to zero at gma_M,0, so a fixed
    # window clipped the last four decades of them
    axs[0, k].set_ylim(lo_N/3., 10.*hi_N)
  axs[0, 0].set_ylabel("$N_{\\rm e}^{-1}\\,\\tilde{t}\\;{\\rm d}\\mathcal{N}_{\\rm e}"
                       "/{\\rm d}\\gamma_{\\rm e}$", fontsize=FS_LAB)
  axs[1, 0].set_ylabel("${\\rm d}\\ln(\\tilde{t}\\,{\\rm d}\\mathcal{N}_{\\rm e}"
                       "/{\\rm d}\\gamma_{\\rm e})/{\\rm d}\\ln\\gamma_{\\rm e}$",
                       fontsize=FS_LAB)
  axs[0, -1].legend(handles=[
      plt.Line2D([], [], color=INK, lw=0, marker='o', mfc='none', ms=3.5,
                 label='$\\gamma_\\mathrm{M}(\\tilde{t}\\,)$')],
      fontsize=FS_LEG, loc='upper left', framealpha=.9, handletextpad=.3, borderpad=.3)
  p1 = axs[0, -1].get_position()
  cax = fig.add_axes([p1.x1 + .016, p1.y0, .019, p1.height])
  cb = fig.colorbar(sm, cax=cax)
  cb.set_label("$\\log_{10}[(t'-t'_0)/t'_0]$", fontsize=FS_LAB)
  cb.ax.tick_params(labelsize=FS_TICK)
  return _save_fig(fig, axs, outdir, fname, show)


def _save_fig(fig, axs, outdir, fname, show):
  os.makedirs(outdir, exist_ok=True)
  path = os.path.join(outdir, fname)
  fig.savefig(path, dpi=300, bbox_inches='tight')
  print(f'saved {path}')
  if show:
    plt.show()
  return fig, axs


def check_reduces_to_closed(logC=-2., sigmas=(1e-3, 1e-1, 1e1, 1e3), p=P_SYN,
    gm0=GM0, gM0=GMA_M0):
  '''
  At q = 0 the clock is linear and the average has the CLOSED FORM N(gma;tt)/tt of
  cooling_integrated_figure. The quadrature must reproduce it -- that is what keeps the
  three columns one implementation rather than two. Returns the worst deviation.
  '''
  kap = tt_dyn_of_C(logC, gm0)
  dev = 0.
  for sg in sigmas:
    u = float(u_of_sigma(sg, kap, 0.))
    g = np.geomspace(1.02*gm0/(1. + gm0*u), .9*gM0, 60)
    dev = max(dev, float(np.max(np.abs(
        N_averaged_decay(g, sg, kap, p, gm0, gM0, 0.)/N_averaged(g, u, p, gm0, gM0)
        - 1.))))
  return dev


def main(show=False):
  print(f'q = 0 vs the closed form N/tt : max rel dev = '
        f'{check_reduces_to_closed():.2e}   (one implementation, not two)')
  print(f'normalisation, q = 0         : max |int - 1| = {check_normalisation():.2e}')
  print(f'slow limit  -> injected law  : {check_slow_limit():.2e}')
  print(f'fast limit  -> N_e gma_c/gma^2: {check_fast_limit():.2e}')
  print('peak, deep fast cooling (both ratios -> 1):')
  for tt, xr, yr in check_peak():
    print(f'    tt={tt:7.0e}:  gma_peak*tt = {xr:.4f}   height/(N_e tt) = {yr:.4f}')
  print(f"\nDECAYING FIELD, t'_c ~ tau^q, q = {Q_C:.4f} (s = q-1 = {Q_C-1.:.4f}):")
  for lab, lc, q in CASES:
    if q == 0.:
      continue
    kap = tt_dyn_of_C(lc, GM0)
    u_i = kap/_exps(0., q)[1]
    tag = 'FC' if lc < 0 else 'SC'
    print(f'  {tag} (C = 10^{lc:+.0f}): kappa={kap:.3e}  u_inf={u_i:.4e}  '
          f'frozen support [{1./(u_i+1./GM0):.4g}, {1./(u_i+1./GMA_M0):.4g}]')
    print(f'     normalisation : max |int dgma - 1| = {check_decay_norm(kap):.2e}')
    print('     -> frozen state: ' + '  '.join(
        f'sigma={s:.0e}: {d:.3e}' for s, d in check_decay_freeze(kap)))
  plot_averaged_steady(show=show)
  plot_integrated_steady(show=show)
  plot_averaged(show=show)


if __name__ == '__main__':
  main()
