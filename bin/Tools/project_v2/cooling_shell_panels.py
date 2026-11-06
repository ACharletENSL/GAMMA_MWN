# -*- coding: utf-8 -*-
# @Author: acharlet

'''
Paper I, Figs. 4 (fig:intNe_gen) and 5 (fig:shell_spectrum), redesigned 2026-10-26, reworked
after the 2026-11-03 meeting: the shell-integrated distribution of the HOMOGENEOUS coasting
shell (cooling_shell_integrated) and its comoving spectrum (cooling_shell_spectrum), in ONE
flat 2 x 2 layout, and their comparisons to the broken power law, Figs. E.1 (fig:shell_bpl) and
F.1 (fig:spectrum_bpl), in one row of four (2026-11-06).

    columns  continuous injection | finite injection (Delta t'_inj = t'_0): the same shell
             up to the end of the injection (grey), then its old electrons only
    rows     fast cooling at injection (gma_m,0 I_dyn,0 = 1e3) | slow cooling (1e-2)
    each     the curves, and their local slope below (Figs. 4-5) or the ratio of the broken
             power law to them (Figs. E.1, F.1)

THE NORMALIZATION divides out what only rescales the curves:
  Fig. 4  f_e = dN_e/(N_e dln gma_e), the quantity of Eqs. bpl_FC - bpl_finite;
  Fig. 5  L'_nu'/(p'_syn(t') N_e) = int f_e k(gma_nu'^2/gma^2) dln gma (Eq. Lnu_shell), at
          u = nu'/nu'_syn(gma_e = 1, t') = gma_nu'^2, so each spectral break sits at the
          square of its Fig. 4 abscissa. The field decay and the growth of N_e are out.

THE CURVES: elapsed times Delta t'/t'_0 = 0.3 (the steady state), 3, 15 and 100 (after t'_tr
in fast cooling), one colour each (key above the panels); the end-of-injection curve
(= continuous, Delta t' = t'_0) in grey in the finite column. In the continuous column, the
curve at Delta t' = T_SPLIT = 3 t'_0 is split into its old and young electrons at the switch of
Eq. eta_zeta, t' - t'_i = t'_i/|s|, i.e. t'_i = 3t'/4 here. Breaks marked on one curve per panel
(NAMED), at the positions of the broken power law. Figs. E.1, F.1 draw it, dashed, on every
curve it describes (not at 0.3, nor after t'_tr); Fig. F.1 adds the step emissivity (dotted)
on the named curve.

THE PARAMETERS: reference fluid element (lam_n = 2, lam_B = 5/3, s = 3), lam_inj = lam_m = 0,
p = 2.5. gma_m,0 = 1e5, gma_M,0 = 1e10 (an exact shift by 10 from the 1e4, 1e9 of the
family) keeps the oldest electrons at gma_e >= 4 at Delta t' = 100 t'_0 with the fast
cooling row at gma_m,0 I_dyn,0 = 1e3 (t'_tr = 33 t'_0 from Eq. t_tr, in gbar_c,0; 19.3 t'_0
where gma_m,i I_dyn = 1, LAWS.R_tr). With Delta t'_inj = t'_0 the old fast cooling tail
spans 2^3 = 8 and the adiabatic tail 2^(2/3) = 1.6: too short for a segment of their own,
which is the point of that column (author, 2026-11-03).

RESULTS (main(), ~15 s): see report().
'''

import os
import numpy as np
import matplotlib.pyplot as plt

import cooling_shell_integrated as csi
import shell_bpl_check as sbc
import cooling_shell_spectrum as css
from cooling_shape_figure import OUTDIR

P = csi.P_SYN
LAWS = csi.COASTING
GM0, GMM0 = 1e5, 1e10
LC = {'FC': -3., 'SC': 2.}                      # log10 C = -log10(gma_m,0 I_dyn,0)
DINJ = 1.                                       # Delta t'_inj/t'_0 (coasting: = R/R_0)

ALL_T = (0.3, 3., 15., 100.)
TIMES = {('FC', 0): ALL_T, ('SC', 0): (0.3, 3., 15.),
         ('FC', 1): ALL_T, ('SC', 1): ALL_T}
COL = dict(zip(ALL_T, plt.cm.Blues(np.linspace(.45, 1., len(ALL_T)))))
NAMED = {('FC', 0): 3., ('SC', 0): 3., ('FC', 1): 100., ('SC', 1): 100.}
T_SPLIT = 3.                                    # the curve split into old and young electrons
NO_BPL = {('FC', 0): (0.3, 100.), ('SC', 0): (0.3,), ('FC', 1): (0.3,), ('SC', 1): (0.3,)}

INK, MUTED, GREY = '0.15', '0.45', '0.75'
OLD_LS = dict(color='#D55E00', ls=(0, (4, 1.5)), lw=1.3, zorder=4)
YOUNG_LS = dict(color='#009E73', ls=(0, (1.2, 1.2)), lw=1.5, zorder=4)
FS_LAB, FS_TICK, FS_ANN = 9., 8., 7.
# A&A: no grid; the only text in a panel is the regime header (on an opaque white box): segments
# and times are read from the slope panels' guides and the keys above the panels (author,
# 2026-10-26; the break symbols moved from the panels to the key, 2026-11-06)
BOX = dict(boxstyle='square,pad=0.1', fc='w', ec='none')
BPL_LS = dict(color=INK, ls=(0, (3, 2)), lw=.8, zorder=5)
STEP_LS = dict(ls=':', lw=1.1)
FINITE_TITLE = r"Finite injection, $\Delta t'_{\rm inj}=t'_0$"


# --- curves ---------------------------------------------------------------------------
def comov(x, regime):
  return sbc.comoving(LAWS, 0., x, LC[regime], P, gm0=GM0)


def bpl_pieces(x, regime, finite):
  '''Eqs. bpl_FC, bpl_SC or (finite) the frozen / adiabatic tail and Eq. bpl_finite,
  as pl_spectrum pieces in f_e, and the breaks'''
  return css.table_pieces(comov(x, regime), regime, 1. + DINJ if finite else None, GMM0)


def eval_pieces(g, pieces):
  f = np.zeros_like(g)
  for lo, hi, f_lo, al in pieces:
    m = (g >= lo) & (g <= hi)
    f[m] += f_lo*(g[m]/lo)**(2.*al)
  return f


def f_exact(x, regime, finite, per_dec=120):
  '''(gma, f_e) on the support of the numerical shell integral'''
  dinj = DINJ if finite else np.inf
  lo, hi = csi.support(x, LC[regime], dinj, LAWS, GM0, GMM0)
  lo, hi = lo*(1. + 1e-9), hi*(1. - 1e-9)
  g = np.geomspace(lo, hi, int(np.log10(hi/lo)*per_dec) + 2)
  N = csi.shell_integrated(g, x, LC[regime], dinj, LAWS, P, GM0, GMM0)
  return g, g*N/csi.injected_number(x, dinj, LAWS)


def old_young(x, regime):
  '''(gma, f_e, f_e of the old electrons, f_e of the young ones) of the continuous injection
  at x, split at the switch of Eq. eta_zeta, t' - t'_i = t'_i/|s|; both in units of N_e(t')'''
  g, f = f_exact(x, regime, False)
  y = x/(1. + 1./abs(LAWS.Lam))
  N = csi.shell_integrated(g, x, LC[regime], y - 1., LAWS, P, GM0, GMM0)
  fo = g*N/csi.injected_number(x, np.inf, LAWS)
  return g, f, fo, np.maximum(f - fo, 0.)


def spectrum(g, f, u, full=True):
  '''L'_nu'/(p'_syn N_e) at u = gma_nu'^2: full kernel, or step emissivity (exact on the
  piecewise power-law interpolant of f)'''
  gnu = np.sqrt(u)
  if not full:
    return css.pl_spectrum(gnu, css.grid_pieces(g, f))
  lg = np.log(g)
  L = np.empty_like(u)
  for k in range(0, u.size, 64):
    y = (gnu[k:k + 64, None]/g[None, :])**2
    L[k:k + 64] = np.trapezoid(f[None, :]*css.kernel_full(y), lg, axis=1)
  return L


def slope(xv, yv):
  with np.errstate(divide='ignore', invalid='ignore'):
    return np.gradient(np.log(yv), np.log(xv))


# --- shared layout --------------------------------------------------------------------
# flat, for the length of the article (author, 2026-11-06), on the A&A text width: Figs. 4-5
# in 2 x 2 (panels 7.6 cm wide), Figs. E.1, F.1 in one row of four (3.4 cm)
WIDTH = 7.24
REGIME = {'FC': r'$\gamma_{{\rm m},0}I_{{\rm dyn},0}=10^3$',
          'SC': r'$\gamma_{{\rm m},0}I_{{\rm dyn},0}=10^{-2}$'}
SLOPE_LAB = 'local slope'                       # the panels are too low for the fraction
KW_KEY = dict(loc='lower center', fontsize=FS_ANN, frameon=False, handlelength=1.6,
              columnspacing=1.1, handletextpad=.5, borderaxespad=0, borderpad=0)


def _guides(axS_row, levs, nudge):
  '''the row's indices as dashed guides in both slope panels, ticked on the right spine
  and named beside their ticks; nudge = {label: dy in points} where two would overlap'''
  levs = sorted(levs, reverse=True)
  for ax in axS_row:
    for v, _ in levs:
      ax.axhline(v, color=MUTED, ls='--', lw=.6, zorder=1)
  axR = axS_row[1].twinx()
  axR.set_ylim(axS_row[1].get_ylim())
  axR.set_yticks([v for v, _ in levs])
  axR.set_yticklabels([])
  axR.tick_params(axis='y', length=2.5, colors=INK)
  for v, t in levs:
    axR.annotate(t, (1., v), xycoords=('axes fraction', 'data'), textcoords='offset points',
                 xytext=(4., nudge.get(t, 0.)), ha='left', va='center', fontsize=FS_ANN,
                 color=INK, annotation_clip=False)


def _key(fig, xy, times, styles, dy=0., breaks=None):
  '''above the panels, one line for the times and the line styles and, under it, one for
  the symbols of the breaks (none in Figs. E.1, F.1)'''
  h = [plt.Line2D([], [], color='none', label=r"$\Delta t'/t'_0$:")]
  h += [plt.Line2D([], [], color=COL[t], lw=1.8, label=f'{t:g}') for t in times]
  fig.legend(handles=h + styles, ncol=len(h) + len(styles), bbox_to_anchor=(xy[0], xy[1] + dy),
             **KW_KEY)
  if breaks:
    h = [plt.Line2D([], [], marker=MARK[q], label=breaks[q], **MARK_KW) for q in KEY_BREAKS]
    fig.legend(handles=h, ncol=len(h), bbox_to_anchor=xy, **dict(KW_KEY, handlelength=1.,
                                                                 columnspacing=1.4))


def _axes(axN, axS, A, col, sl, every):
  '''labels and ticks of one (curves, slope or ratio) pair of panels; A = AX_G or AX_U'''
  axN.set_xlim(*A['xlim'])
  axS.set_xticks(10.**np.arange(*A['ticks'], every*A['step']))
  for ax in (axN, axS):
    ax.tick_params(which='both', labelsize=FS_TICK)
  if col == 0:
    _ylab_col0(axN, axS, A['ylab'], sl)
  else:
    axN.tick_params(labelleft=False)
    axS.tick_params(labelleft=False, which='both')


def _flat(draw, A, breaks, slopes, nudge, slope_ticks, outdir, fname, show):
  '''Figs. 4-5: fast cooling above slow cooling, continuous | finite injection'''
  fig, axs = plt.subplots(5, 2, figsize=(WIDTH, 4.55), sharex=True,
      gridspec_kw=dict(height_ratios=[2.1, .85, .22, 2.1, .85], hspace=.07, wspace=.05))
  fig.subplots_adjust(left=.075, right=.905, top=.86, bottom=.09)
  for ax in axs[2]:
    ax.set_visible(False)
  for reg, (axNs, axSs) in (('FC', axs[:2]), ('SC', axs[3:])):
    for col in (0, 1):
      draw(reg, col, axNs[col], axSs[col])
      axNs[col].annotate({'FC': 'fast cooling\n', 'SC': 'slow cooling\n'}[reg] + REGIME[reg],
                         (.98, .96), xycoords='axes fraction', ha='right', va='top',
                         multialignment='right', fontsize=FS_ANN, color=INK, bbox=BOX, zorder=8)
      axSs[col].set_yticks(slope_ticks)
      _axes(axNs[col], axSs[col], A, col, SLOPE_LAB, 1)
    _guides(axSs, slopes[reg], nudge[reg])
  for ax in axs[4]:
    ax.set_xlabel(A['xlab'], fontsize=FS_LAB)
  axs[0, 0].set_title('Continuous injection', fontsize=FS_LAB, pad=3)
  axs[0, 1].set_title(FINITE_TITLE, fontsize=FS_LAB, pad=3)
  p, q = axs[0, 0].get_position(), axs[0, 1].get_position()
  _key(fig, (.5*(p.x0 + q.x1), p.y1 + .05), ALL_T, KEY_SPLIT, .04, breaks)
  return _save(fig, outdir, fname, show)


def _row(draw, A, sl, key, outdir, fname, show):
  '''Figs. E.1, F.1: fast cooling, then slow cooling, each continuous | finite injection'''
  fig = plt.figure(figsize=(WIDTH, 2.75))
  gs = fig.add_gridspec(2, 5, height_ratios=[2.2, 1.], width_ratios=[1, 1, .34, 1, 1],
                        hspace=.07, wspace=.05, left=.075, right=.925, top=.83, bottom=.14)
  head = {}
  for i, (reg, col) in zip((0, 1, 3, 4), (('FC', 0), ('FC', 1), ('SC', 0), ('SC', 1))):
    axN = fig.add_subplot(gs[0, i])
    axS = fig.add_subplot(gs[1, i], sharex=axN)
    draw(reg, col, axN, axS)
    _axes(axN, axS, A, col, sl, 2)
    axN.tick_params(labelbottom=False)
    axS.set_xlabel(A['xlab'], fontsize=FS_LAB - A['xlab_small'])
    axN.text(.5, 1.03, ('continuous', r"finite, $\Delta t'_{\rm inj}=t'_0$")[col], ha='center',
             va='bottom', transform=axN.transAxes, fontsize=FS_ANN, color=INK)
    head.setdefault(reg, []).append(axN.get_position())
  for reg, name in (('FC', 'Fast cooling, '), ('SC', 'Slow cooling, ')):
    p, q = head[reg]
    fig.text(.5*(p.x0 + q.x1), p.y1 + .065, name + REGIME[reg], ha='center', va='bottom',
             fontsize=FS_LAB - 1, color=INK)
  p, q = head['FC'][0], head['SC'][1]
  _key(fig, (.5*(p.x0 + q.x1), p.y1 + .135), BPL_T, key)
  return _save(fig, outdir, fname, show)


# one marker per break, the same for a quantity in every panel and in both figures
MARK = {'c,ad': 'o', 'm,ad': 's', 'cFC': 'D', 'cSC': 'D', 'm,i': '^', 'c,l': 'v',
        'm,l': 'p'}
MARK_KW = dict(ms=5., mfc='w', mec=INK, mew=1., ls='none')
KEY_BREAKS = ('c,ad', 'm,ad', 'c,l', 'm,l', 'cFC', 'm,i')


def _mark_breaks(ax, at, value, br, quantities):
  '''a marker on the exact curve at each break'''
  for k, q in quantities.items():
    xb = at(br[k])
    ax.plot([xb], [value(xb)], marker=MARK[q], zorder=6, **MARK_KW)


def _interp_log(xs, ys):
  ly = np.log(np.maximum(ys, 1e-300))           # the support edges hold zeros
  return lambda v: float(np.exp(np.interp(np.log(v), np.log(xs), ly)))


# --- Fig. 4 ---------------------------------------------------------------------------
G_BREAKS = {  # table break -> quantity (MARK)
  ('FC', 0): {'low': 'c,ad', 'fr': 'cFC', 'mi': 'm,i'},
  ('SC', 0): {'low': 'm,ad', 'mi': 'm,i', 'b': 'cSC'},
  ('FC', 1): {'low': 'c,ad', 'hi': 'c,l'},
  ('SC', 1): {'low': 'm,ad', 'adf': 'm,l', 'hi': 'c,ad', 'top': 'c,l'}}
G_LAB = {'c,ad': r'$\gamma_{\rm c,ad}$', 'm,ad': r'$\gamma_{\rm m,ad}$',
         'cFC': r'$\tilde{\gamma}_{\rm c}$', 'cSC': r'$\tilde{\gamma}_{\rm c}$',
         'm,i': r'$\gamma_{\rm m}$', 'c,l': r'$\gamma_{\rm c,l}$',
         'm,l': r'$\gamma_{\rm m,l}$'}
G_NUDGE = {'FC': {r'$-2/3$': 0., r'$-2$': 2., r'$-p$': -1.5, r'$-(p+1)$': -2.},
           'SC': {r'$-p$': 1.5, r'$-(p+1)$': -2.}}
G_SLOPES = {'FC': ((-2./3., r'$-2/3$'), (-2., r'$-2$'), (-P, r'$-p$'),
                   (-(P + 1.), r'$-(p+1)$')),
            'SC': ((.5, r'$+1/2$'), (-P, r'$-p$'), (-(P + 1.), r'$-(p+1)$'))}
XLIM_G = (2., 3e9)
YLIM_F = {'FC': (1e-5, 8.), 'SC': (2e-7, 8.)}
SL_LIM_G = (-4.3, 1.3)
F_LAB = r'$f_{\rm e}$'
AX_G = dict(xlim=XLIM_G, xlab=r'$\gamma_{\rm e}$', xlab_small=0, ylab=F_LAB, ticks=(1, 10), step=2)
KEY_SPLIT = [plt.Line2D([], [], color=GREY, lw=1.6, label='end of injection'),
             plt.Line2D([], [], label='old electrons', **OLD_LS),
             plt.Line2D([], [], label='young electrons', **YOUNG_LS)]


def _ylab_col0(axN, axS, yl, sl):
  if yl == F_LAB:                               # one symbol: read horizontally
    axN.set_ylabel(yl, fontsize=FS_LAB, rotation=0, ha='right', va='center', labelpad=5.)
  else:
    axN.set_ylabel(yl, fontsize=FS_LAB)
  axS.set_ylabel(sl, fontsize=FS_ANN + .5, labelpad=2.)


def _named_g(br, reg, col, g, f):
  '''the last elements pile up at their frozen Lorentz factor: mark gma_c,l at the peak of the
  exact curve, where it reads, rather than at the table's value, which falls on the sharp
  drop just above it (author, 2026-10-26)'''
  if G_BREAKS[(reg, col)].get('hi') == 'c,l':
    return dict(br, hi=float(g[np.argmax(f)]))
  return br


def _draw_distribution(reg, col, axN, axS):
  finite = col == 1
  if finite:
    g, f = f_exact(1. + DINJ, reg, False)
    axN.loglog(g, f, color=GREY, lw=1.4, zorder=2)
  for t in TIMES[(reg, col)]:
    x = 1. + t
    g, f = f_exact(x, reg, finite)
    axN.loglog(g, f, color=COL[t], lw=1.5, zorder=3)
    s = slope(g, f/g)                           # of dN/dgma, as the text quotes them
    s[~np.isfinite(s) | (s < SL_LIM_G[0]) | (s > SL_LIM_G[1])] = np.nan
    axS.semilogx(g, s, color=COL[t], lw=1.2, zorder=3)
    if not finite and t == T_SPLIT:
      ge, _, fo, fy = old_young(x, reg)
      axN.loglog(ge, np.where(fo > 0, fo, np.nan), **OLD_LS)
      axN.loglog(ge, np.where(fy > 0, fy, np.nan), **YOUNG_LS)
    if t == NAMED[(reg, col)]:
      _, br = bpl_pieces(x, reg, finite and t > DINJ)
      _mark_breaks(axN, lambda v: np.clip(v, g[0]*1.03, g[-1]/1.03), _interp_log(g, f),
                   _named_g(br, reg, col, g, f), G_BREAKS[(reg, col)])
  axN.set_ylim(*YLIM_F[reg])
  axS.set_ylim(*SL_LIM_G)


def fig_distribution(outdir=OUTDIR, fname='cooling_shell_integrated.png', show=False):
  return _flat(_draw_distribution, AX_G, G_LAB, G_SLOPES, G_NUDGE, (-4, -2, 0), outdir, fname, show)


# --- Fig. 5 ---------------------------------------------------------------------------
U_LAB = {'c,ad': r"$\nu'_{\rm c,ad}$", 'm,ad': r"$\nu'_{\rm m,ad}$", 'cFC': r"$\nu'_{\rm c}$",
         'cSC': r"$\nu'_{\rm c}$", 'm,i': r"$\nu'_{\rm m}$", 'c,l': r"$\nu'_{\rm c,l}$",
         'm,l': r"$\nu'_{\rm m,l}$"}
U_NUDGE = {'FC': {r'$1/3$': 3., r'$1/6$': -3., r'$-1/2$': 2., r'$-(p-1)/2$': -1.}, 'SC': {}}
U_SLOPES = {'FC': ((1./3., r'$1/3$'), (1./6., r'$1/6$'), (-.5, r'$-1/2$'),
                   (-(P - 1.)/2., r'$-(p-1)/2$'), (-P/2., r'$-p/2$')),
            'SC': ((1./3., r'$1/3$'), (-(P - 1.)/2., r'$-(p-1)/2$'), (-P/2., r'$-p/2$'))}
XLIM_U = (1e-1, 3e19)
U = np.geomspace(*XLIM_U, int(np.log10(XLIM_U[1]/XLIM_U[0])*40))
YLIM_L = {'FC': (1e-5, 8.), 'SC': (1e-6, 8.)}
SL_LIM_U = (-1.55, .55)
L_LAB = r"$L'_{\nu'}/(p'_{\rm syn}\mathcal{N}_{\rm e})$"
U_XLAB = r"$\nu'/\nu'_{\rm syn}(\gamma_{\rm e}=1)$"
AX_U = dict(xlim=XLIM_U, xlab=U_XLAB, xlab_small=1, ylab=L_LAB, ticks=(0, 20), step=4)


def _draw_spectrum(reg, col, axN, axS):
  finite = col == 1
  if finite:
    g, f = f_exact(1. + DINJ, reg, False)
    axN.loglog(U, spectrum(g, f, U), color=GREY, lw=1.4, zorder=2)
  for t in TIMES[(reg, col)]:
    x = 1. + t
    g, f = f_exact(x, reg, finite)
    L = spectrum(g, f, U)
    axN.loglog(U, L, color=COL[t], lw=1.5, zorder=3)
    sl = slope(U, L)
    sl[~np.isfinite(sl) | (sl < SL_LIM_U[0]) | (sl > SL_LIM_U[1])] = np.nan
    axS.semilogx(U, sl, color=COL[t], lw=1.2, zorder=3)
    if not finite and t == T_SPLIT:
      ge, _, fo, fy = old_young(x, reg)
      axN.loglog(U, spectrum(ge, fo, U), **OLD_LS)
      axN.loglog(U, spectrum(ge, fy, U), **YOUNG_LS)
    if t == NAMED[(reg, col)]:
      _, br = bpl_pieces(x, reg, finite and t > DINJ)
      _mark_breaks(axN, lambda v: np.clip(v, *XLIM_U), _interp_log(U, L),
                   {k: v**2 for k, v in _named_g(br, reg, col, g, f).items()},
                   G_BREAKS[(reg, col)])
  axN.set_ylim(*YLIM_L[reg])
  axS.set_ylim(*SL_LIM_U)


def fig_spectrum(outdir=OUTDIR, fname='cooling_shell_spectrum.png', show=False):
  return _flat(_draw_spectrum, AX_U, U_LAB, U_SLOPES, U_NUDGE, (-1, 0), outdir, fname, show)


# --- the comparisons to the broken power law, Figs. E.1 and F.1 -----------------------
BPL_TIMES = {('FC', 0): (3., 15.), ('SC', 0): (3., 15.),
             ('FC', 1): (3., 15., 100.), ('SC', 1): (3., 15., 100.)}
BPL_T = (3., 15., 100.)
RATIO_LIM = (.25, 4.)
KEY_BPL = [plt.Line2D([], [], label='broken power law', **BPL_LS)]


def _ratio_axes(ax):
  ax.set_yscale('log')
  ax.set_ylim(*RATIO_LIM)
  ax.axhline(1., color=MUTED, ls='--', lw=.6, zorder=1)
  ax.set_yticks([.5, 1., 2.])
  ax.set_yticklabels(['0.5', '1', '2'])
  ax.yaxis.set_minor_formatter(plt.NullFormatter())


def _draw_distribution_bpl(reg, col, axN, axS):
  finite = col == 1
  for t in BPL_TIMES[(reg, col)]:
    x = 1. + t
    g, f = f_exact(x, reg, finite)
    axN.loglog(g, f, color=COL[t], lw=1.5, zorder=3)
    pieces, _ = bpl_pieces(x, reg, finite and t > DINJ)
    gb = np.geomspace(min(q[0] for q in pieces), max(q[1] for q in pieces), 800)
    fb = eval_pieces(gb, pieces)
    axN.loglog(gb, np.where(fb > 0, fb, np.nan), **BPL_LS)
    r = eval_pieces(g, pieces)/np.where(f > 0, f, np.nan)
    # the lowest Lorentz factors differ by ~1 %: not a segment of the distribution
    r[(g < 1.05*g[0]) | (g > g[-1]/1.05)] = np.nan
    axS.semilogx(g, np.where(r > 0, r, np.nan), color=COL[t], lw=1.2, zorder=3)
  axN.set_ylim(*YLIM_F[reg])
  _ratio_axes(axS)


def fig_distribution_bpl(outdir=OUTDIR, fname='cooling_shell_bpl.png', show=False):
  return _row(_draw_distribution_bpl, AX_G, r'$f_{\rm e}^{\rm BPL}/f_{\rm e}^{\rm num}$', KEY_BPL,
              outdir, fname, show)


def _draw_spectrum_bpl(reg, col, axN, axS):
  finite = col == 1
  for t in BPL_TIMES[(reg, col)]:
    x = 1. + t
    g, f = f_exact(x, reg, finite)
    L = spectrum(g, f, U)
    axN.loglog(U, L, color=COL[t], lw=1.5, zorder=3)
    pieces, _ = bpl_pieces(x, reg, finite and t > DINJ)
    Lb = css.pl_spectrum(np.sqrt(U), pieces)
    ok = (Lb > 0) & (L > YLIM_L[reg][0])
    axN.loglog(U, np.where(Lb > 0, Lb, np.nan), **BPL_LS)
    with np.errstate(divide='ignore', invalid='ignore'):
      axS.semilogx(U, np.where(ok, Lb/L, np.nan), color=COL[t], lw=1.2, zorder=3)
    if t == NAMED[(reg, col)]:
      Ls = spectrum(g, f, U, full=False)
      axN.loglog(U, np.where(Ls > 0, Ls, np.nan), color=COL[t], zorder=4, **STEP_LS)
      with np.errstate(divide='ignore', invalid='ignore'):
        axS.semilogx(U, np.where(ok & (Ls > 0), Ls/L, np.nan), color=COL[t], zorder=3, **STEP_LS)
  axN.set_ylim(*YLIM_L[reg])
  _ratio_axes(axS)


def fig_spectrum_bpl(outdir=OUTDIR, fname='cooling_spectrum_bpl.png', show=False):
  key = KEY_BPL + [plt.Line2D([], [], color=MUTED, label='step emissivity', **STEP_LS)]
  return _row(_draw_spectrum_bpl, AX_U, r"$L'_{\nu'}/L'^{\,{\rm full}}_{\nu'}$", key,
              outdir, fname, show)


def _save(fig, outdir, fname, show):
  os.makedirs(outdir, exist_ok=True)
  path = os.path.join(outdir, fname)
  fig.savefig(path, dpi=300, bbox_inches='tight')
  print(f'saved {path}')
  if show:
    plt.show()
  return fig


# --- the numbers quoted in the text ---------------------------------------------------
def report():
  '''the numbers quoted in the text: tails of the short finite injection, index of its
  spectrum, the old and young electrons of the split curve, accuracy of the broken power law'''
  c = comov(1. + DINJ, 'FC')
  te = 1. + DINJ
  print(f"t'_tr (gma_m,i I_dyn = 1) = {LAWS.R_tr(LC['FC']):.3g} t'_0; last element: gma_m,i I_dyn = "
        f"{GM0/(c.s*c.gci(te))*(1. - 2.**-c.s):.3g}; old FC tail spans {te**c.eta:.4g}, adiabatic "
        f"tail {te**c.zeta:.3g}")
  u = np.geomspace(1e-3, 1e20, 2400)
  print('-- spectral index between the lowest and the top break')
  for reg, col, t in (('FC', 0, 3.), ('FC', 0, 15.), ('FC', 1, 3.), ('FC', 1, 15.), ('FC', 1, 100.),
                      ('SC', 1, 3.), ('SC', 1, 15.), ('SC', 1, 100.)):
    finite = col == 1
    x = 1. + t
    g, f = f_exact(x, reg, finite)
    _, br = bpl_pieces(x, reg, finite and t > DINJ)
    if reg == 'FC':
      lo, top = br['low'], br['hi'] if finite and t > DINJ else br['fr']
    else:
      lo, top = br['low'], br['adf']
    sl = slope(u, spectrum(g, f, u))
    samp = {d: float(np.interp(np.log(lo**2*10.**d), np.log(u), sl)) for d in (.5, 1., 1.5)}
    print(f"  {reg} {'finite' if finite else 'contin'} dt'={t:g}: nu_top/nu_low = 10^"
          f"{np.log10(top**2/lo**2):.2f}; index at +0.5, 1, 1.5 dec: "
          + ', '.join(f'{v:+.3f}' for v in samp.values()))
  print('-- old and young electrons of the split curve')
  for reg in ('FC', 'SC'):
    x = 1. + T_SPLIT
    g, f, fo, fy = old_young(x, reg)
    Lg = np.log(g)
    uu = np.geomspace(1e-1, 3e19, 800)
    L, Lo = spectrum(g, f, uu), spectrum(g, fo, uu)
    _, br = bpl_pieces(x, reg, False)
    k = 'fr' if reg == 'FC' else 'b'
    cr = g[np.argmin(np.abs(np.log(np.maximum(fo, 1e-300)/np.maximum(fy, 1e-300))))]
    print(f"  {reg}: number fraction of old {np.trapezoid(fo, Lg)/np.trapezoid(f, Lg):.3f}; "
          f"old = young at gma = {cr:.3g}, gamma~_c = {br[k]:.3g}; top of old {g[fo > 0].max():.3g}; "
          f"old/all at the peak of L' {Lo[np.argmax(L)]/L.max():.3f}")
  print('-- broken power law / numerical')
  for reg, col, t in (('FC', 0, 3.), ('FC', 0, 15.), ('SC', 0, 3.), ('SC', 0, 15.),
                      ('FC', 1, 3.), ('FC', 1, 15.), ('FC', 1, 100.),
                      ('SC', 1, 3.), ('SC', 1, 15.), ('SC', 1, 100.)):
    finite = col == 1
    x = 1. + t
    g, f = f_exact(x, reg, finite)
    pieces, _ = bpl_pieces(x, reg, finite and t > DINJ)
    fb = eval_pieces(g, pieces)
    m = (f > 1e-6*f.max()) & (fb > 0)
    r = fb[m]/f[m]
    q = np.quantile(r, [.05, .5, .95])
    print(f"  {reg} {'finite' if finite else 'contin'} dt'={t:g}: {r.min():.2f}-{r.max():.2f}, "
          f"5-50-95%: {q[0]:.2f} {q[1]:.2f} {q[2]:.2f}")


def main(show=False):
  report()
  fig_distribution(show=show)
  fig_spectrum(show=show)
  fig_distribution_bpl(show=show)
  fig_spectrum_bpl(show=show)


if __name__ == '__main__':
  main()
