# -*- coding: utf-8 -*-
# @Author: acharlet

'''
Paper I, Figs. 4 (fig:intNe_gen) and 5 (fig:shell_spectrum), redesigned 2026-10-26: the
shell-integrated distribution of the HOMOGENEOUS coasting shell (cooling_shell_integrated)
and its comoving spectrum (cooling_shell_spectrum), in ONE layout.

    columns  continuous injection | finite injection (Delta t'_inj = 5 t'_0), after its end
    rows     fast cooling at injection (gma_m,0 I_dyn,0 = 1e3) | slow cooling (1e-2)
    each     the curves, and their local slope below

THE NORMALIZATION divides out what only rescales the curves:
  Fig. 4  f_e = dN_e/(N_e dln gma_e), the quantity of Eqs. bpl_FC - bpl_finite;
  Fig. 5  L'_nu'/(p'_syn(t') N_e) = int f_e k(gma_nu'^2/gma^2) dln gma (Eq. Lnu_shell), at
          u = nu'/nu'_syn(gma_e = 1, t') = gma_nu'^2, so each spectral break sits at the
          square of its Fig. 4 abscissa. The field decay and the growth of N_e are out.

THE CURVES: elapsed times Delta t'/t'_0 = 0.3 (before freezing: the steady state), 3, 15
and 100 (after t'_tr in fast cooling), one colour each, labelled on Fig. 4; the
end-of-injection curve (= continuous, Delta t' = 5 t'_0) in grey in the finite column.
The broken power law of Eqs. bpl_FC, bpl_SC, bpl_finite (shell_bpl_check.table, with
cooling_shell_spectrum.table_pieces for Eq. Lnu_bpl) dashed on every curve it describes
(not at 0.3, nor after t'_tr), its breaks named on one curve per panel (NAMED); on that
curve Fig. 5 also draws the step emissivity, dotted.

THE PARAMETERS: reference fluid element (lam_n = 2, lam_B = 5/3, s = 3), lam_inj = lam_m = 0,
p = 2.5. gma_m,0 = 1e5, gma_M,0 = 1e10 (an exact shift by 10 from the 1e4, 1e9 of the
family) keeps the oldest electrons at gma_e >= 4 at Delta t' = 100 t'_0 with the fast
cooling row at gma_m,0 I_dyn,0 = 1e3 (t'_tr = 19.3 t'_0). That row is deep enough for the
last elements of the finite injection (t'_i = 6 t'_0, gma_m,i I_dyn = 15) to freeze as in
Eq. mapping: at 1e2 (t'_tr = 12 t'_0) and Delta t'_inj = 10 t'_0 they did not, and the
table overshot the top of the frozen tail x3. With Delta t'_inj = 5 t'_0 the frozen tail
spans 6^3 = 216 and the adiabatic tail 6^(2/3) = 3.3.

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
DINJ = 5.                                       # Delta t'_inj/t'_0 (coasting: = R/R_0)

ALL_T = (0.3, 3., 15., 100.)
TIMES = {('FC', 0): ALL_T, ('SC', 0): (0.3, 3., 15.),
         ('FC', 1): (15., 100.), ('SC', 1): (15., 100.)}
COL = dict(zip(ALL_T, plt.cm.Blues(np.linspace(.45, 1., len(ALL_T)))))
NAMED = {('FC', 0): 3., ('SC', 0): 3., ('FC', 1): 100., ('SC', 1): 100.}
NO_BPL = {('FC', 0): (0.3, 100.), ('SC', 0): (0.3,)}

INK, MUTED, GREY = '0.15', '0.45', '0.75'
FS_LAB, FS_TICK, FS_ANN = 9., 8., 7.
# A&A: no grid, text on an opaque borderless white box
BOX = dict(boxstyle='square,pad=0.1', fc='w', ec='none')
BPL_LS = dict(color=INK, ls=(0, (3, 2)), lw=.8, zorder=5)
STEP_LS = dict(ls=':', lw=1.1)
FINITE_TITLE = r"Finite injection ($\Delta t'_{\rm inj}=5\,t'_0$), after its end"


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
def _layout(figsize, slope_h):
  fig, axs = plt.subplots(5, 2, figsize=figsize, sharex=True,
      gridspec_kw=dict(height_ratios=[2.1, slope_h, .2, 2.1, slope_h], hspace=.07,
                       wspace=.05))
  for ax in axs[2]:
    ax.set_visible(False)
  return fig, {'FC': (axs[0], axs[1]), 'SC': (axs[3], axs[4])}, axs


def _guides(axS_row, levs):
  '''the row's indices as dashed guides in both slope panels, named on the right spine'''
  levs = sorted(levs, reverse=True)
  for ax in axS_row:
    for v, _ in levs:
      ax.axhline(v, color=MUTED, ls='--', lw=.6, zorder=1)
  axR = axS_row[1].twinx()
  axR.set_ylim(axS_row[1].get_ylim())
  axR.set_yticks([v for v, _ in levs])
  axR.set_yticklabels([t for _, t in levs])
  tl = axR.get_yticklabels()
  for k in range(len(levs) - 1):
    if levs[k][0] - levs[k + 1][0] < .6:
      tl[k].set_va('bottom')
      tl[k + 1].set_va('top')
  axR.tick_params(axis='y', labelsize=FS_ANN, length=2.5, pad=1.5, colors=INK)


def _headers(rows):
  rows['FC'][0][0].set_title('Continuous injection', fontsize=FS_LAB, pad=4)
  rows['FC'][0][1].set_title(FINITE_TITLE, fontsize=FS_LAB, pad=4)
  for reg, txt in (('FC', 'fast cooling\n' + r'$\gamma_{{\rm m},0}I_{{\rm dyn},0}=10^3$'),
                   ('SC', 'slow cooling\n' + r'$\gamma_{{\rm m},0}I_{{\rm dyn},0}=10^{-2}$')):
    for ax in rows[reg][0]:
      ax.annotate(txt, (.98, .96), xycoords='axes fraction', ha='right', va='top',
                  multialignment='right', fontsize=FS_ANN, color=INK, bbox=BOX, zorder=8)


def _key(fig, axs, extra):
  h = [plt.Line2D([], [], color=COL[t], lw=1.8, label=f'{t:g}') for t in ALL_T] + extra
  top = axs[0, 0].get_position().y1
  fig.legend(handles=h, loc='lower center', ncol=len(h), bbox_to_anchor=(.5, top + .035),
             fontsize=FS_ANN, frameon=False, handlelength=1.6, columnspacing=1.1,
             handletextpad=.5)
  fig.text(.5, top + .075, r"elapsed time $\Delta t'/t'_0$ (colors)", ha='center',
           fontsize=FS_ANN, color=INK)


def _name_breaks(ax, at, value, br, names, offs):
  '''a short mark on the exact curve at each break, and its name at offs (points)'''
  for k, lab in names.items():
    xb = at(br[k])
    yb = value(xb)
    ax.plot([xb], [yb], marker='|', ms=8, mew=1., color=INK, zorder=6)
    dx, dy = offs[k]
    ax.annotate(lab, (xb, yb), textcoords='offset points', xytext=(dx, dy),
                ha='left' if dx > 0 else ('right' if dx < 0 else 'center'),
                va='bottom' if dy >= 0 else 'top', fontsize=FS_ANN, color=INK, bbox=BOX,
                zorder=7)


def _panel_text(ax, spots, italic):
  for xv, yv, txt in spots:
    ax.text(xv, yv, txt, fontsize=FS_ANN + (0. if italic else .5), color=MUTED,
            style='italic' if italic else 'normal', ha='center', va='center', zorder=7,
            bbox=BOX)


def _interp_log(xs, ys):
  ly = np.log(np.maximum(ys, 1e-300))           # the support edges hold zeros
  return lambda v: float(np.exp(np.interp(np.log(v), np.log(xs), ly)))


# --- Fig. 4 ---------------------------------------------------------------------------
G_NAMES = {
  ('FC', 0): {'low': r'$\gamma_{\rm c,ad}$', 'fr': r'$\kappa_{\rm FC}\gamma_{\rm c}$',
              'mi': r'$\gamma_{{\rm m},i}$'},
  ('SC', 0): {'low': r'$\gamma_{\rm m,ad}$', 'mi': r'$\gamma_{{\rm m},i}$',
              'b': r'$\kappa_{\rm SC}\gamma_{\rm c}$'},
  ('FC', 1): {'low': r'$\gamma_{\rm c,ad}$', 'hi': r'$\gamma_{\rm c,ad,f}$'},
  ('SC', 1): {'low': r'$\gamma_{\rm m,ad}$', 'adf': r'$\gamma_{\rm m,ad,f}$',
              'hi': r'$\gamma_{\rm c,ad}$'}}
G_OFFS = {
  ('FC', 0): {'low': (2, 7), 'fr': (0, 8), 'mi': (7, -3)},
  ('SC', 0): {'low': (-5, 2), 'mi': (5, 4), 'b': (-4, -5)},
  ('FC', 1): {'low': (2, 7), 'hi': (5, -4)},
  ('SC', 1): {'low': (-5, 2), 'adf': (0, 8), 'hi': (4, 4)}}
G_SEGS = {   # (gma_e, f_e, segment name), by eye on the render
  ('FC', 0): ((1.5e2, 1.2, 'frozen tail'), (5e3, 1.8e-3, 'cooling tail'),
              (2e7, 1.2e-3, 'cooled')),
  ('SC', 0): ((2.6e4, 3e-3, 'adiabatic\ntail'), (2e6, 1e-1, 'injected'),
              (6e8, 2e-4, 'cooled')),
  ('FC', 1): ((2e2, 1.2, 'frozen tail'),),
  ('SC', 1): ((1.2e3, 1.5e-1, 'adiabatic\ntail'), (4e5, 1.6, 'injected'))}
G_TIMES = {  # (gma_e, f_e, ha) of each time's label, beside the curve's lowest gma_e
  ('FC', 0): {0.3: (1.6e2, 3e-2, 'left'), 3.: (30., 3e-2, 'right'),
              15.: (12., 3e-2, 'right'), 100.: (3.5, 3e-3, 'right')},
  ('SC', 0): {0.3: (9e4, 2e-5, 'left'), 3.: (3.5e4, 2e-5, 'right'),
              15.: (1.4e4, 2e-5, 'right')},
  ('FC', 1): {15.: (16., 3e-3, 'left'), 100.: (3.5, 3e-3, 'right')},
  ('SC', 1): {15.: (1.7e4, 1e-5, 'left'), 100.: (4e3, 1e-5, 'right')}}
G_SLOPES = {'FC': ((-2./3., r'$-2/3$'), (-2., r'$-2$'), (-P, r'$-p$'),
                   (-(P + 1.), r'$-(p+1)$')),
            'SC': ((.5, r'$+1/2$'), (-P, r'$-p$'), (-(P + 1.), r'$-(p+1)$'))}
XLIM_G = (2., 3e9)
YLIM_F = {'FC': (1e-5, 8.), 'SC': (1e-6, 8.)}
SL_LIM_G = (-4.3, 1.3)


def fig_distribution(outdir=OUTDIR, fname='cooling_shell_integrated.png', show=False):
  fig, rows, axs = _layout((7.0, 6.9), .8)
  for reg in ('FC', 'SC'):
    axNs, axSs = rows[reg]
    for col in (0, 1):
      axN, axS = axNs[col], axSs[col]
      finite = col == 1
      if finite:
        g, f = f_exact(1. + DINJ, reg, False)
        axN.loglog(g, f, color=GREY, lw=1.4, zorder=2)
      for t in TIMES[(reg, col)]:
        x = 1. + t
        g, f = f_exact(x, reg, finite)
        axN.loglog(g, f, color=COL[t], lw=1.5, zorder=3)
        s = slope(g, f/g)                      # of dN/dgma, as the text quotes them
        s[~np.isfinite(s) | (s < SL_LIM_G[0]) | (s > SL_LIM_G[1])] = np.nan
        axS.semilogx(g, s, color=COL[t], lw=1.2, zorder=3)
        if t in NO_BPL.get((reg, col), ()):
          continue
        pieces, br = bpl_pieces(x, reg, finite)
        gb = np.geomspace(min(q[0] for q in pieces), max(q[1] for q in pieces), 800)
        fb = eval_pieces(gb, pieces)
        axN.loglog(gb, np.where(fb > 0, fb, np.nan), **BPL_LS)
        if t == NAMED[(reg, col)]:
          names = G_NAMES[(reg, col)]
          _name_breaks(axN, lambda v: np.clip(v, g[0]*1.03, g[-1]/1.03),
                       _interp_log(g, f), {k: br[k] for k in names}, names,
                       G_OFFS[(reg, col)])
      _panel_text(axN, G_SEGS[(reg, col)], italic=True)
      for t, (xv, yv, ha) in G_TIMES[(reg, col)].items():
        axN.text(xv, yv, f'{t:g}', ha=ha, va='center', fontsize=FS_ANN, color=COL[t],
                 fontweight='bold', bbox=BOX, zorder=8)
      axN.set_ylim(*YLIM_F[reg])
      axS.set_ylim(*SL_LIM_G)
      for ax in (axN, axS):
        ax.tick_params(which='both', labelsize=FS_TICK)
      if col == 0:
        axN.set_ylabel(r'$f_{\rm e}={\rm d}\mathcal{N}_{\rm e}/(\mathcal{N}_{\rm e}\,'
                       r'{\rm d}\ln\gamma_{\rm e})$', fontsize=FS_LAB)
        axS.set_ylabel('local slope of\n' + r'${\rm d}\mathcal{N}_{\rm e}/{\rm d}\gamma_{\rm e}$',
                       fontsize=FS_LAB - 1)
      else:
        axN.tick_params(labelleft=False)
        axS.tick_params(labelleft=False)
    _guides(axSs, G_SLOPES[reg])
  rows['FC'][0][0].annotate(r"after $t'_{\rm tr}$", (4e4, 3.5e-1), xytext=(3e6, 1.5e-1),
                            fontsize=FS_ANN, color=COL[100.], ha='center', va='center',
                            bbox=BOX, arrowprops=dict(arrowstyle='-', color=COL[100.],
                                                      lw=.6))
  for ax in axs[4]:
    ax.set_xlabel(r'$\gamma_{\rm e}$', fontsize=FS_LAB)
  axs[0, 0].set_xlim(*XLIM_G)
  axs[0, 0].set_xticks(10.**np.arange(1, 10, 2))
  _headers(rows)
  _key(fig, axs, [plt.Line2D([], [], color=GREY, lw=1.6, label='end of injection'),
                  plt.Line2D([], [], label='broken power law', **BPL_LS)])
  return _save(fig, outdir, fname, show)


# --- Fig. 5 ---------------------------------------------------------------------------
U_NAMES = {
  ('FC', 0): {'low': r"$\nu'_{\rm c,ad}$", 'fr': r"$\nu'_{\rm c}$", 'mi': r"$\nu'_{\rm m}$"},
  ('SC', 0): {'low': r"$\nu'_{\rm m,ad}$", 'mi': r"$\nu'_{\rm m}$", 'b': r"$\nu'_{\rm c}$"},
  ('FC', 1): {'low': r"$\nu'_{\rm c,ad}$", 'hi': r"$\nu'_{\rm c,ad,f}$"},
  ('SC', 1): {'low': r"$\nu'_{\rm m,ad}$", 'adf': r"$\nu'_{\rm m,ad,f}$",
              'hi': r"$\nu'_{\rm c,ad}$"}}
U_OFFS = {
  ('FC', 0): {'low': (-3, 6), 'fr': (0, 8), 'mi': (5, 3)},
  ('SC', 0): {'low': (-4, 5), 'mi': (4, 5), 'b': (-5, 3)},
  ('FC', 1): {'low': (-3, 6), 'hi': (5, 3)},
  ('SC', 1): {'low': (-6, 3), 'adf': (4, 9), 'hi': (5, 4)}}
U_SEGS = {   # (u, L, spectral index of the segment)
  ('FC', 0): ((1e1, 3e-2, r"$\nu'^{1/3}$"), (3e4, 6e-1, r"$\nu'^{\,\alpha_{\rm fr}}$"),
              (2e8, 1.5e-2, r"$\nu'^{-1/2}$"), (1e13, 2e-4, r"$\nu'^{-p/2}$")),
  ('SC', 0): ((1e4, 3e-2, r"$\nu'^{1/3}$"), (1e13, 1.5e-1, r"$\nu'^{-(p-1)/2}$"),
              (1e18, 3e-5, r"$\nu'^{-p/2}$")),
  ('FC', 1): ((1e1, 3e-2, r"$\nu'^{1/3}$"), (1e4, 8e-1, r"$\nu'^{\,\alpha_{\rm fr}}$")),
  ('SC', 1): ((1e3, 3e-2, r"$\nu'^{1/3}$"), (1e13, 1e-1, r"$\nu'^{-(p-1)/2}$"))}
U_SLOPES = {'FC': ((1./3., r'$1/3$'), (1./6., r'$1/6$'), (-.5, r'$-1/2$'),
                   (-(P - 1.)/2., r'$-(p-1)/2$'), (-P/2., r'$-p/2$')),
            'SC': ((1./3., r'$1/3$'), (-(P - 1.)/2., r'$-(p-1)/2$'), (-P/2., r'$-p/2$'))}
XLIM_U = (1e-1, 3e19)
YLIM_L = {'FC': (1e-5, 8.), 'SC': (1e-6, 8.)}
SL_LIM_U = (-1.55, .55)


def fig_spectrum(outdir=OUTDIR, fname='cooling_shell_spectrum.png', show=False):
  u = np.geomspace(*XLIM_U, int(np.log10(XLIM_U[1]/XLIM_U[0])*40))
  fig, rows, axs = _layout((7.0, 7.3), 1.05)
  for reg in ('FC', 'SC'):
    axNs, axSs = rows[reg]
    for col in (0, 1):
      axN, axS = axNs[col], axSs[col]
      finite = col == 1
      if finite:
        g, f = f_exact(1. + DINJ, reg, False)
        axN.loglog(u, spectrum(g, f, u), color=GREY, lw=1.4, zorder=2)
      for t in TIMES[(reg, col)]:
        x = 1. + t
        g, f = f_exact(x, reg, finite)
        L = spectrum(g, f, u)
        axN.loglog(u, L, color=COL[t], lw=1.5, zorder=3)
        sl = slope(u, L)
        sl[~np.isfinite(sl) | (sl < SL_LIM_U[0]) | (sl > SL_LIM_U[1])] = np.nan
        axS.semilogx(u, sl, color=COL[t], lw=1.2, zorder=3)
        if t in NO_BPL.get((reg, col), ()):
          continue
        pieces, br = bpl_pieces(x, reg, finite)
        Lb = css.pl_spectrum(np.sqrt(u), pieces)
        if finite:                    # Eq. Lnu_bpl drops to 0 at the top of the finite BPL
          Lb[u > .3*br['hi']**2] = 0.
        axN.loglog(u, np.where(Lb > 0, Lb, np.nan), **BPL_LS)
        if t == NAMED[(reg, col)]:
          Ls = spectrum(g, f, u, full=False)
          axN.loglog(u, np.where(Ls > 0, Ls, np.nan), color=COL[t], zorder=4, **STEP_LS)
          names = U_NAMES[(reg, col)]
          _name_breaks(axN, lambda v: np.clip(v, *XLIM_U), _interp_log(u, L),
                       {k: br[k]**2 for k in names}, names, U_OFFS[(reg, col)])
      _panel_text(axN, U_SEGS[(reg, col)], italic=False)
      axN.set_ylim(*YLIM_L[reg])
      axS.set_ylim(*SL_LIM_U)
      for ax in (axN, axS):
        ax.tick_params(which='both', labelsize=FS_TICK)
      if col == 0:
        axN.set_ylabel(r"$L'_{\nu'}/(p'_{\rm syn}\mathcal{N}_{\rm e})$", fontsize=FS_LAB)
        axS.set_ylabel(r"${\rm d}\ln L'_{\nu'}/{\rm d}\ln\nu'$", fontsize=FS_LAB - 1)
      else:
        axN.tick_params(labelleft=False)
        axS.tick_params(labelleft=False)
    _guides(axSs, U_SLOPES[reg])
  for ax in axs[4]:
    ax.set_xlabel(r"$\nu'/\nu'_{\rm syn}(\gamma_{\rm e}=1)$", fontsize=FS_LAB)
  axs[0, 0].set_xlim(*XLIM_U)
  axs[0, 0].set_xticks(10.**np.arange(0, 20, 4))
  _headers(rows)
  _key(fig, axs, [plt.Line2D([], [], color=GREY, lw=1.6, label='end of injection'),
                  plt.Line2D([], [], label='broken power law', **BPL_LS),
                  plt.Line2D([], [], color=MUTED, label='step emissivity', **STEP_LS)])
  return _save(fig, outdir, fname, show)


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
  '''t'_tr; the finite fast cooling frozen tail (table/exact, residual width W_e - 1 of
  the last element); the spectral index in the frozen-tail segments'''
  print(f"t'_tr = {LAWS.R_tr(LC['FC']):.3g} t'_0 (fast cooling row)")
  c = comov(1. + DINJ, 'FC')
  te = 1. + DINJ
  s_gci = c.s*c.gci(te)
  print(f"last element: gma_m,i I_dyn = {GM0/(c.s*c.gci(te))*(1. - 2.**-c.s):.3g}, "
        f"W_e - 1 = s gbar_c,i/gma_m,i = {s_gci/GM0:.3g}; frozen tail spans "
        f"{te**c.eta:.4g}, adiabatic tail {te**c.zeta:.3g}")
  u = np.geomspace(1e-3, 1e20, 2400)
  for col, t in ((0, 3.), (0, 15.), (1, 15.), (1, 100.)):
    finite = col == 1
    g, f = f_exact(1. + t, 'FC', finite)
    pieces, br = bpl_pieces(1. + t, 'FC', finite)
    top = br['hi'] if finite else br['fr']
    fb = eval_pieces(g, pieces)
    m = (g > 1.1*br['low']) & (g < top/1.5) & (fb > 0.)
    r = fb[m]/f[m]
    sl = slope(u, spectrum(g, f, u))
    ok = (u > br['low']**2) & (u < top**2)
    near = ok & (np.abs(sl - 1./6.) < .05)
    mid = np.sqrt(br['low']*top)**2
    print(f"FC {'finite' if finite else 'continuous'} Delta t' = {t:g}: table/exact in the "
          f"frozen tail {r.min():.3f}-{r.max():.3f}; nu'_top/nu'_low = "
          f"10^{np.log10(top**2/br['low']**2):.1f}; index within 0.05 of 1/6 up to "
          f"{np.log10(top**2/u[near].max()):.1f} dec below nu'_top; index "
          f"{np.interp(np.log(mid), np.log(u), sl):.3f} mid-segment, range in the segment "
          f"{np.nanmax(sl[ok & (u > 3.*br['low']**2) & (u < top**2/30.)]):.3f} to "
          f"{np.nanmin(sl[ok & (u > 3.*br['low']**2) & (u < top**2/30.)]):.3f}")


def main(show=False):
  report()
  fig_distribution(show=show)
  fig_spectrum(show=show)


if __name__ == '__main__':
  main()
