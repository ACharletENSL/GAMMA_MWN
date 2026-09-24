# -*- coding: utf-8 -*-
# @Author: acharlet

'''
The cooling-shape family, regrouped BY QUANTITY instead of by physics.

cooling_shape_figure and cooling_shape_adiabatic each draw one physics case with two
stacked panels (edge tracks above, distributions below). This module transposes that:
one figure per quantity, three panels side by side, one per physics case. The panel
ORDER is the same in both figures, so panel (b) of one is panel (b) of the other.

    cooling_tracks.png    gma_m(t') and gma_M(t') against t'/t'_c,i
    cooling_shapes.png    N(gma,t') at sampled log10(t'/t'_c,i)

THE THREE PANELS, and what each assumes.

  (a) SYNCHROTRON ONLY, constant t'_c. No expansion, so the field has no reason to
      change and tt = int dt'/t'_c IS t'/t'_c,i exactly. This is the panel that carries
      the COOLING REGIMES: they are defined by the two injected edges burning, at
      tilde{t}_M = 1/gma_M0 and tilde{t}_m = 1/gma_m0, and in these units they need no
      reference C at all. Physics from cooling_shape_figure.

  (b) SYNCHROTRON + ADIABATIC, fast cooling, bar{gma}_c/gma_m = 1e-3.
  (c) the same, slow cooling, bar{gma}_c/gma_m = 1e+3.

      Both on the FREELY EXPANDING SHELL at constant eps_B, which is one choice, not
      two: rho' ~ R^-2 (a shell of fixed comoving width, coasting R/R_0 = 1 + sigma),
      and then B'^2/8pi = eps_B e' with e' ~ rho'^gma_ad ties the field to it,
      B' ~ rho'^(gma_ad/2), so q = -a_rho*gma_ad/2 = 4/3 is DERIVED. With
      alpha = a_rho/3 = -2/3 that gives e = alpha - 2q + 1 = -7/3 < 0: the synchrotron
      burn saturates at S_inf = tt_dyn/|e|, and only C < 1/|e| reaches fast cooling --
      which is why (b) and (c) sit either side of that threshold rather than either
      side of C = 1. The constants live in cooling_integrated_adiabatic (A_RHO, Q_B)
      and the trajectory in gamma_cooled / cooled_distrib_adiab, so nothing is
      restated here.

WHY gma_M MOVES BETWEEN (a), (b) AND (c). The abscissa is the physical comoving time
t'/t'_c,i in all three, not the generalised tt_eff = S/A. In tt_eff the top edge is
C-independent and slides down 1/tt_eff in every panel; in t'/t'_c,i it is not, which is
exactly why the two regimes need panels of their own instead of one shared pair of axes.
For (b) and (c), sigma = (t'/t'_c,i)/tt_dyn does the conversion.

The two figures share: the sampled times (cooling_shape_figure.LOGTT_SAMPLES, the same
log10 t'/t'_c,i in every panel), the viridis ramp ordered by time, the injected bounds
as dotted verticals, and the gma < 1 shading -- below it the ultra-relativistic
trajectory is not the physical one, so the curves are drawn but caveated.

Run:  python cooling_shape_panels.py
'''

import os
import numpy as np
import matplotlib.pyplot as plt

from cooling_distribution import gamma_synCooled, norm_plaw_distrib
from cooling_shape_figure import (P_SYN, GM0, GMA_M0, OUTDIR, INK, MUTED, MC_FAC,
    FS_LAB, FS_TICK, FS_ANN, FS_LEG, LOGTT_SAMPLES, BAND_ALPHA, _band_bg, cooled_distrib)
from cooling_integrated_figure import tt_dyn_of_C
from cooling_integrated_adiabatic import A_RHO, Q_B, gamma_cooled
from cooling_shape_adiabatic import cooled_distrib_adiab, lab_C

# --- defaults -------------------------------------------------------------------------
LOGC_PANELS = (-3., 3.)     # the two adiabatic columns: either side of the C < 1/|e|
                            # cooling threshold, not either side of C = 1
T_LIM = (1e-11, 1e2)        # t'/t'_c,i span of the TRACK panels, shared by all three
NX = 900                    # points per track
# ONE y range for the three track panels, so the drag in (b)/(c) is read against (a)
# rather than against a rescaled axis. The floor clears the lowest edge any panel
# reaches (4.6e-3, the slow-cooling adiabatic gma_m); the ceiling leaves the decade the
# regime-band labels of panel (a) sit in.
GMA_LIM = (1e-3, 30.*GMA_M0)
N_LIM = (1e-16, 1e4)        # N/N_e range of the SHAPE panels
# a row of three, sized for a two-column article's full width. The shape panels get the
# taller box: twenty decades of N against ten of gamma.
FIGSIZE_TRACKS, FIGSIZE_SHAPES = (7.1, 2.75), (7.1, 3.15)
FN_TRACKS, FN_SHAPES = 'cooling_tracks.png', 'cooling_shapes.png'


def panel_specs(gm0=GM0, logC_m=LOGC_PANELS):
  '''
  (kind, tt_dyn, title) for the three panels, in the order BOTH figures use.
  tt_dyn is None for the synchrotron panel, which has no C to set one.
  '''
  out = [('syn', None, '(a) synchrotron only')]
  for k, lc in zip('bc', logC_m):
    out.append(('adiab', tt_dyn_of_C(lc, gm0),
                f'({k}) $+$ adiabatic, '
                f'$\\bar{{\\gamma}}_{{\\rm c}}/\\gamma_{{\\rm m}}={lab_C(lc)}$'))
  return out


def _row(figsize, n=3, wspace=.10):
  'A row of n panels sharing one y axis, plus the helper that puts a legend above them.'
  fig, axs = plt.subplots(1, n, figsize=figsize, sharey=True, squeeze=False,
                          gridspec_kw=dict(wspace=wspace))
  return fig, axs[0]


def _legend_above(fig, ax_src, axs, **kw):
  '''
  ONE legend row above the panels. In-panel placement has nowhere to go here: the
  panels are a third as wide as the two-panel figures' were, and the band labels,
  the injected-bound labels and the curve bundle already claim top, bottom and middle.
  A FIGURE legend, because bbox_inches='tight' walks those -- and no
  bbox_extra_artists, which would REPLACE every axes' own extras and crop it.

  Anchored above the TITLES, measured rather than guessed: axes.y1 leaves the row
  sitting straight on top of them, and the clearance a title needs is its font size,
  which is not a fixed fraction of a figure whose height changes between the two.
  '''
  fig.canvas.draw()                       # titles have no extent until they are laid out
  inv = fig.transFigure.inverted()
  top = max(inv.transform(ax.title.get_window_extent())[1, 1] for ax in axs)
  p0, p1 = axs[0].get_position(), axs[-1].get_position()
  fig.legend(*ax_src.get_legend_handles_labels(), fontsize=FS_LEG, loc='lower center',
             bbox_to_anchor=(.5*(p0.x0 + p1.x1), top + .015),
             bbox_transform=fig.transFigure, framealpha=1., handlelength=1.4,
             handletextpad=.5, borderpad=.3, columnspacing=.9, **kw)


def _mark_gma1(ax, axis):
  "gma = 1: a y band on the track panels, an x band on the shape panels."
  span = ax.axhspan if axis == 'y' else ax.axvspan
  line = ax.axhline if axis == 'y' else ax.axvline
  span(1e-30, 1., color='crimson', alpha=.07, lw=0, zorder=0)
  line(1., color='crimson', ls=':', lw=.9, zorder=1)


# --- (1) the edge tracks ----------------------------------------------------------------
def plot_cooling_tracks(p=P_SYN, gm0=GM0, gM0=GMA_M0, a_rho=A_RHO, q=Q_B,
    logC_m=LOGC_PANELS, outdir=OUTDIR, fname=FN_TRACKS, show=False):
  '''
  gma_m and gma_M against t'/t'_c,i, one panel per physics case (see module docstring).
  The regime bands are drawn on panel (a) only: that is where they are defined.
  '''
  fig, axs = _row(FIGSIZE_TRACKS)
  x = np.geomspace(*T_LIM, NX)

  for ax, (kind, ttd, title) in zip(axs, panel_specs(gm0, logC_m)):
    if kind == 'syn':
      # THE COOLING REGIMES. t_M and t_m are the cooling times of the two injected
      # edges; MC is a NEIGHBOURHOOD of t_m, a factor MC_FAC either side. Colours are
      # the house shape-class palette (sweep_gammacm's per-spectrum table), RdBu from
      # VSC red to VFC blue; its 'marginal' #f7f7f7 is invisible as a tint, so MC
      # gets a grey.
      t_M, t_m = 1./gM0, 1./gm0
      for lab, a_, b_, col in (('VSC', x[0],       t_M,        '#b2182b'),
                               ('SC',  t_M,        t_m/MC_FAC, '#ef8a62'),
                               ('MC',  t_m/MC_FAC, t_m*MC_FAC, '0.6'),
                               ('FC',  t_m*MC_FAC, 1.,         '#67a9cf'),
                               ('VFC', 1.,         x[-1],      '#2166ac')):
        a_, b_ = max(a_, x[0]), min(b_, x[-1])
        if not a_ < b_:
          continue
        ax.axvspan(a_, b_, color=col, alpha=BAND_ALPHA, lw=0, zorder=0)
        ax.annotate(lab, (np.sqrt(a_*b_), .985), xycoords=('data', 'axes fraction'),
                    color=INK, fontsize=FS_ANN, ha='center', va='top',
                    bbox=dict(fc=_band_bg(col), ec='none', pad=1.5))
      # 1/tt is the asymptote the top edge slides down, never a place it bends
      ax.loglog(x, 1./x, color=MUTED, ls='-.', lw=.9, label="$(t'/t'_{\\rm c,i})^{-1}$")
      gM, gm = gamma_synCooled(x, gM0), gamma_synCooled(x, gm0)
    else:
      sg = x/ttd                                   # sigma = (t'/t'_c,i)/tt_dyn
      gM = gamma_cooled(sg, gM0, ttd, a_rho, q)
      gm = gamma_cooled(sg, gm0, ttd, a_rho, q)

    ax.loglog(x, gM, color='k', lw=1.4, label='$\\gamma_\\mathrm{M}$')
    ax.loglog(x, gm, color='k', lw=1.1, ls='--', label='$\\gamma_\\mathrm{m}$')
    _mark_gma1(ax, 'y')
    if kind == 'syn':
      # the two knees, along the floor where nothing else runs
      for v, lab in ((1./gM0, '$\\tilde{t}_M$'), (1./gm0, '$\\tilde{t}_m$')):
        ax.axvline(v, color=INK, ls=':', lw=.8, zorder=1)
        ax.annotate(lab, (v, .015), xycoords=('data', 'axes fraction'), color=INK,
                    fontsize=FS_ANN, ha='center', va='bottom',
                    bbox=dict(fc='w', ec='none', alpha=.85, pad=1.))
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(*GMA_LIM)
    ax.set_yticks(10.**np.arange(-2., np.log10(gM0) + 1., 2.))
    ax.set_xlabel("$t'/t'_{\\rm c,i}$", fontsize=FS_LAB, labelpad=1.)
    ax.set_title(title, fontsize=FS_LAB, pad=3.)
    ax.tick_params(axis='x', pad=1.5)
    ax.tick_params(which='both', labelsize=FS_TICK)
    ax.grid(alpha=.25, lw=.4)
  axs[0].set_ylabel('$\\gamma$', fontsize=FS_LAB)

  _legend_above(fig, axs[0], axs, ncol=3)
  return _save(fig, axs, outdir, fname, show)


# --- (2) the distribution shapes ---------------------------------------------------------
def plot_cooling_shapes(p=P_SYN, gm0=GM0, gM0=GMA_M0, a_rho=A_RHO, q=Q_B,
    logtt=LOGTT_SAMPLES, logC_m=LOGC_PANELS, outdir=OUTDIR, fname=FN_SHAPES, show=False):
  '''
  N(gma,t') at the sampled log10(t'/t'_c,i), one panel per physics case. Same panel
  order as plot_cooling_tracks, and the SAME sampled times in every panel.
  '''
  fig, axs = _row(FIGSIZE_SHAPES)
  colors = plt.cm.viridis(np.linspace(0., .85, len(logtt)))
  K0 = norm_plaw_distrib(gm0, gM0, p)
  lo = np.inf

  for ax, (kind, ttd, title) in zip(axs, panel_specs(gm0, logC_m)):
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
      ax.annotate('$\\propto\\gamma^{-p}$', (gg[1], 12.*K0*gg[1]**-p),
                  textcoords='offset points', xytext=(3, 3), color=MUTED,
                  fontsize=FS_ANN)
    # the injected bounds, along the TOP
    for v, lab in ((gm0, '$\\gamma_{\\mathrm{m},\\!0}$'),
                   (gM0, '$\\gamma_{\\mathrm{M},\\!0}$')):
      ax.axvline(v, color=INK, ls=':', lw=.8, zorder=1)
      ax.annotate(lab, (v, .985), xycoords=('data', 'axes fraction'), color=INK,
                  fontsize=FS_ANN, ha='center', va='top',
                  bbox=dict(fc='w', ec='none', alpha=.85, pad=1.))
    _mark_gma1(ax, 'x')
    ax.set_ylim(*N_LIM)
    ax.set_xlabel('$\\gamma$', fontsize=FS_LAB)
    ax.set_title(title, fontsize=FS_LAB, pad=3.)
    ax.tick_params(which='both', labelsize=FS_TICK)
    ax.grid(alpha=.25, lw=.4)
  for ax in axs:
    ax.set_xlim(.5*lo, 2.*gM0)
  axs[0].set_ylabel("$N(\\gamma,t')/N_{\\rm e}$", fontsize=FS_LAB)

  _legend_above(fig, axs[0], axs, ncol=len(logtt),
                title="$\\log_{10}(t'/t'_{\\rm c,i})$")
  fig.legends[-1].get_title().set_fontsize(FS_LEG)
  return _save(fig, axs, outdir, fname, show)


def _save(fig, axs, outdir, fname, show):
  os.makedirs(outdir, exist_ok=True)
  path = os.path.join(outdir, fname)
  fig.savefig(path, dpi=300, bbox_inches='tight')
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
    ttd = tt_dyn_of_C(lc, GM0)
    gm = float(gamma_cooled(x_end/ttd, GM0, ttd, A_RHO, Q_B))
    print(f'  C = 10^{lc:+.0f}      gma_m = {gm:.3e}   (tt_dyn = {ttd:.3e})')
  plot_cooling_tracks(show=show)
  plot_cooling_shapes(show=show)


if __name__ == '__main__':
  main()
