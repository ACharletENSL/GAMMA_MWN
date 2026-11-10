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
from matplotlib.transforms import Bbox

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
NAMED = {('FC', 0): (3.,), ('SC', 0): (3.,), ('FC', 1): (0.3, 100.), ('SC', 1): (0.3, 100.)}
# meeting 2026-11-10: 'add a couple broken power laws and some missing breaks, especially before the
# end of the injection': breaks on the curve during the injection in the impulsive column too, and
# the broken power law of the text over one curve per panel
OVER_BPL = {('FC', 0): (), ('SC', 0): (), ('FC', 1): (), ('SC', 1): ()}   # superseded by GUIDES
# author 2026-11-10: power-law GUIDES (as in the old steady-state figure) rather than broken power laws:
# each segment of these curves gets a dotted line parallel to it, offset above, with its slope written
GUIDES = {('FC', 0): (3.,), ('SC', 0): (3.,), ('FC', 1): (0.3, 100.), ('SC', 1): (0.3, 100.)}
GUIDES_U = {('FC', 0): (3.,), ('SC', 0): (3.,), ('FC', 1): (0.3, 100.), ('SC', 1): (0.3, 100.)}   # spectra
GUIDE_LS = dict(color='0.35', ls=(0, (1, 1.3)), lw=.9, zorder=5)
GUIDE_FAC = 4.
P_FC, P_SC = 2./3., -.5                          # reference fluid element, kappa_N = kappa_m = 0
F_EXP = {1. - P_FC: r'1-p_{\rm FC}', -1.: r'-1', -P: r'-p', 1. - P: r'1-p', 1. - P_SC: r'1-p_{\rm SC}'}
U_EXP = {1./3.: r'1/3', (1. - P_FC)/2.: r'\alpha_{\rm FC}', -.5: r'-1/2', -(P - 1.)/2.: r'-(p-1)/2', -P/2.: r'-p/2'}


def _exp_label(v, table):
  for k, t in table.items():
    if abs(v - k) < 1e-6:
      return t
  return None


def _draw_guides(ax, segs, curve, table, var, min_dec, ylim, line=None, prio=0, t=None):
  '''queue the guides of one curve (placed by _place_guides once the panel is complete); prio 0 for the
  curves chosen in GUIDES, 1 for the others, which take a slope only where the chosen curves have no room'''
  ax.__dict__.setdefault('_guide_req', []).append((segs, curve, table, var, min_dec, ylim, line, prio, t))


GUIDE_CORE = (.25, .75)      # the guide covers the central half of its segment (in log), away from its breaks
GUIDE_FACS = (2.5, 3.5, 5., 7., 10., 15., 22., 33., 50., 75., 110.)


def _other_logy(ln, xs):
  '''log10 y of another line at xs (nan outside it); markers-only lines give their points'''
  x, y = (np.asarray(v, float) for v in ln.get_data())
  ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
  x, y = x[ok], y[ok]
  if x.size < 2 or ln.get_linestyle() in ('None', 'none', '', ' '):
    return np.full_like(xs, np.nan)
  o = np.argsort(x)
  lx, ly = np.log10(x[o]), np.log10(y[o])
  out = np.interp(np.log10(xs), lx, ly)
  out[(np.log10(xs) < lx[0]) | (np.log10(xs) > lx[-1])] = np.nan
  return out


def _place_guides(ax):
  '''place the queued guides: for each exponent, the first curve (chosen curves first, then the longest
  segments) where an offset (either side, closest first) leaves no other line between the curve and the
  guide, the label (set beyond the guide) crosses no line, marker or other label, and both stay inside the
  panel; each exponent once per panel'''
  fig = ax.figure
  fig.canvas.draw()
  rnd = fig.canvas.get_renderer()
  cands = []
  done = set()
  for spec in ax.__dict__.get('_guide_manual', []):   # guides placed by hand (author 2026-11-10)
    for segs, curve, table, var, min_dec, ylim, target, prio, t in ax.__dict__.get('_guide_req', []):
      if t != spec['t']:
        continue
      for lo, hi, e in segs:
        if abs(e - spec['e']) < 1e-6 and _force_guide(ax, rnd, _exp_label(e, table), lo, hi, e, curve, var, spec, target):
          done.add(_exp_label(e, table))
          break
  for segs, curve, table, var, min_dec, ylim, target, prio, t in ax.__dict__.get('_guide_req', []):
    for lo, hi, e in segs:
      lab = _exp_label(e, table)
      if lab is None:
        continue
      xs = np.geomspace(lo, hi, 200)
      cv = np.array([curve(v) for v in xs])
      vis = xs[(cv > 3.*ylim[0]) & (xs > ax.get_xlim()[0]) & (xs < ax.get_xlim()[1])]
      if vis.size < 2 or np.log10(vis[-1]/vis[0]) < min_dec:
        continue
      cands.append((prio, -np.log10(vis[-1]/vis[0]), lab, vis[0], vis[-1], e, curve, var, ylim, target))
  cands.sort(key=lambda c: (c[0], c[1]))
  for prio, _, lab, lo, hi, e, curve, var, ylim, target in cands:
    if lab in done:
      continue
    if _try_guide(ax, rnd, lab, lo, hi, e, curve, var, ylim, target):
      done.add(lab)


GUIDE_WINDOWS = ((.25, .75), (.1, .6), (.4, .9), (.1, .45), (.55, .9), (.3, .65))   # fractions of the segment (log)
GUIDE_LABEL_AT = (.5, 0., 1.)                                                       # along the guide (log)


def _is_marks(ln):
  return ln.get_linestyle() in ('None', 'none', '', ' ')


def _line_points(ax, ln):
  '''display coordinates of a line, densified along its segments'''
  x, y = (np.asarray(v, float) for v in ln.get_data())
  ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
  if not ok.any():
    return np.zeros((0, 2))
  P = ax.transData.transform(np.c_[x[ok], y[ok]])
  if _is_marks(ln) or P.shape[0] < 2:
    return P
  seg = np.hypot(*np.diff(P, axis=0).T)
  n = np.maximum((seg/1.5).astype(int), 1)
  return np.vstack([P[k] + np.outer(np.linspace(0., 1., n[k], endpoint=False), P[k + 1] - P[k])
                    for k in range(len(n))] + [P[-1:]])


def _stretch(lo, hi, e, curve, tol=None, ax=None, room=30.):
  '''the longest stretch of [lo, hi] where the curve follows the slope e within tol (GUIDE_TOL) and, with ax,
  stays inside the panel with room (a factor) above its lower limit'''
  xd = np.geomspace(lo, hi, 300)
  ld = np.log10([curve(v) for v in xd])
  okd = np.abs(np.gradient(ld, np.log10(xd)) - e) < (tol or GUIDE_TOL)
  if ax is not None:
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    okd &= (xd > x0*1.5) & (xd < x1/1.5) & (ld > np.log10(y0*room)) & (ld < np.log10(y1/3.))
  best, run = (0, 0), None
  for k, v in enumerate(np.r_[okd, False]):
    if v and run is None:
      run = k
    elif not v and run is not None:
      if k - run > best[1] - best[0]:
        best = (run, k)
      run = None
  return (xd[best[0]], xd[best[1] - 1]) if best[1] - best[0] >= 3 else None


GUIDE_CLEAR = 6.          # px: least distance between a guide and any other line


def _force_guide(ax, rnd, lab, lo, hi, e, curve, var, spec, target=None):
  '''a guide placed by hand: spec = dict(t, e, win=(w0, w1) within the stretch that follows the slope,
  side=+1 above / -1 below the curve, fac=least offset factor, text='beyond' (default) | 'above' | 'below',
  dx=+1 right / -1 left, at=position along the guide). The offset grows until the guide is GUIDE_CLEAR from
  every other line, and the label takes the first position (preferred first) that crosses no line, symbol
  or other label and stays inside the panel.'''
  st = _stretch(lo, hi, e, curve, spec.get('tol', .3), ax, spec['fac']*2. if spec['side'] < 0 else 3.)
  if st is None:
    return False
  lo, hi = st
  a, b = lo*(hi/lo)**spec['win'][0], lo*(hi/lo)**spec['win'][1]
  xx = np.geomspace(a, b, 40)
  xc = np.sqrt(a*b)
  lines = [ln for ln in ax.get_lines()]
  pts = {id(ln): _line_points(ax, ln) for ln in lines}
  y0l, y1l = np.log10(ax.get_ylim())
  lg = None
  for f in spec['fac']*np.array([1., 1.4, 2., 2.8, 4.]):
    cand = np.log10(curve(xc)) + e*np.log10(xx/xc) + spec['side']*np.log10(f)
    if cand.min() < y0l + .15 or cand.max() > y1l - .15:
      break
    G = ax.transData.transform(np.c_[xx, 10.**cand])
    dmin = min((np.min(np.hypot(*(pts[id(ln)][:, None, :] - G[None, :, :]).transpose(2, 0, 1)))
                for ln in lines if ln is not target and pts[id(ln)].size), default=99.)
    lg = cand
    if dmin >= GUIDE_CLEAR:
      break
  if lg is None:
    return False
  gl, = ax.plot(xx, 10.**lg, **GUIDE_LS)
  G = ax.transData.transform(np.c_[xx, 10.**lg])
  axbb = ax.get_window_extent(rnd)
  texts = ax.__dict__.get('_guide_texts', [])
  beyond = spec['side'] > 0
  up0 = {'beyond': beyond, 'above': True, 'below': False}[spec.get('text', 'beyond')]
  dx0 = spec.get('dx', 1. if (e < 0) == up0 else -1.)
  opts = []
  for f_at in (spec.get('at', .5), .5, 0., 1.):
    for up in (up0, not up0):
      for dx in (dx0, -dx0):
        opts.append((f_at, up, dx))
  txt = None
  for f_at, up, dx in opts:
    xm = a*(b/a)**f_at
    yv = 10.**np.interp(np.log10(xm), np.log10(xx), lg)
    t = ax.annotate(rf'$\propto{var}^{{{lab}}}$', (xm, yv), textcoords='offset points',
                    xytext=(4.*dx, 3. if up else -3.), ha='left' if dx > 0 else 'right',
                    va='bottom' if up else 'top', fontsize=FS_ANN - .5, color='0.3', zorder=7)
    bb = t.get_window_extent(rnd).expanded(1.08, 1.2)
    ok = axbb.x0 <= bb.x0 and bb.x1 <= axbb.x1 and axbb.y0 <= bb.y0 and bb.y1 <= axbb.y1
    if ok:
      for P in [pts[id(ln)] for ln in lines] + [G]:
        if np.any((P[:, 0] > bb.x0) & (P[:, 0] < bb.x1) & (P[:, 1] > bb.y0) & (P[:, 1] < bb.y1)):
          ok = False; break
    if ok and any(bb.overlaps(o.get_window_extent(rnd)) for o in texts):
      ok = False
    if ok:
      txt = t; break
    if txt is None and (f_at, up, dx) == opts[0]:
      first = (f_at, up, dx)
    t.remove()
  if txt is None:                                   # nothing clear: the preferred position
    f_at, up, dx = opts[0]
    xm = a*(b/a)**f_at; yv = 10.**np.interp(np.log10(xm), np.log10(xx), lg)
    txt = ax.annotate(rf'$\propto{var}^{{{lab}}}$', (xm, yv), textcoords='offset points',
                      xytext=(4.*dx, 3. if up else -3.), ha='left' if dx > 0 else 'right',
                      va='bottom' if up else 'top', fontsize=FS_ANN - .5, color='0.3', zorder=7)
  ax.__dict__.setdefault('_guide_lines', []).append(gl)
  ax.__dict__.setdefault('_guide_texts', []).append(txt)
  return True


GUIDE_TOL = .12          # the curve follows the guide's slope within this, over the guide


def _try_guide(ax, rnd, lab, lo, hi, e, curve, var, ylim, target):
  '''one guide on one curve segment: a window of the segment away from its ends, an offset on either side
  (closest first) with no other line or symbol between the curve and the guide, and a label beyond the guide
  that crosses nothing; True if placed'''
  ylo, yhi = np.log10(ylim[0]), np.log10(ylim[1])
  # restrict the segment to the longest stretch where the curve follows the slope (within GUIDE_TOL)
  xd = np.geomspace(lo, hi, 300)
  ld = np.log10([curve(v) for v in xd])
  okd = np.abs(np.gradient(ld, np.log10(xd)) - e) < GUIDE_TOL
  best_run, run = (0, 0), None
  for k, v in enumerate(np.r_[okd, False]):
    if v and run is None:
      run = k
    elif not v and run is not None:
      if k - run > best_run[1] - best_run[0]:
        best_run = (run, k)
      run = None
  if best_run[1] - best_run[0] < 3:
    return False
  lo, hi = xd[best_run[0]], xd[best_run[1] - 1]
  lines = ax.get_lines()
  others = [ln for ln in lines if ln is not target]
  pts = {id(ln): _line_points(ax, ln) for ln in lines}
  axbb = ax.get_window_extent(rnd)
  texts = ax.__dict__.get('_guide_texts', [])
  for w0, w1 in GUIDE_WINDOWS:
    a, b = lo*(hi/lo)**w0, lo*(hi/lo)**w1
    if np.log10(b/a) < .35:
      continue
    xx = np.geomspace(a, b, 40)
    lc = np.log10([curve(v) for v in xx])
    xc = np.sqrt(a*b)                           # the guide is a straight power law through the curve at xc
    lp = np.log10(curve(xc)) + e*np.log10(xx/xc)
    oth = [_other_logy(ln, xx) for ln in others if not _is_marks(ln)]
    marks = [ln for ln in others if _is_marks(ln)]
    for fac in GUIDE_FACS:
      for side in (1., -1.):
        lg = lp + side*np.log10(fac)
        if lg.min() < ylo + .25 or lg.max() > yhi - .25 or np.any(side*(lg - lc) < .2):   # on its side, clear
          continue
        lo_b, hi_b = np.minimum(lc, lg), np.maximum(lc, lg)
        if any(np.any(np.isfinite(ly) & (ly > lo_b + .03) & (ly < hi_b - .03)) for ly in oth):
          continue
        blocked = False
        for mk in marks:
          for px, py in zip(*(np.asarray(v, float) for v in mk.get_data())):
            if a <= px <= b:
              lyc = np.interp(np.log10(px), np.log10(xx), lc); lyg = np.interp(np.log10(px), np.log10(xx), lg)
              if min(lyc, lyg) - .1 < np.log10(py) < max(lyc, lyg) + .1:
                blocked = True
        if blocked:
          continue
        # the guide itself must not touch another line
        G = ax.transData.transform(np.c_[xx, 10.**lg])
        gbb = Bbox.from_extents(G[:, 0].min() - 2, G[:, 1].min() - 2, G[:, 0].max() + 2, G[:, 1].max() + 2)
        near = False
        for ln in lines:
          P = pts[id(ln)]
          m = (P[:, 0] > gbb.x0) & (P[:, 0] < gbb.x1) & (P[:, 1] > gbb.y0) & (P[:, 1] < gbb.y1)
          if m.any() and np.min(np.hypot(*(P[m][:, None, :] - G[None, :, :]).transpose(2, 0, 1)), initial=99.) < 3.:
            near = True; break
        if near:
          continue
        up = side > 0
        for f_at in GUIDE_LABEL_AT:
          xm = a*(b/a)**f_at
          y0 = 10.**np.interp(np.log10(xm), np.log10(xx), lg)
          sidex = (1. if (e < 0) == up else -1.) if f_at == .5 else (-1. if f_at == 0. else 1.)
          txt = ax.annotate(rf'$\propto{var}^{{{lab}}}$', (xm, y0), textcoords='offset points',
                            xytext=(4.*sidex, 3. if up else -3.), ha='left' if sidex > 0 else 'right',
                            va='bottom' if up else 'top', fontsize=FS_ANN - .5, color='0.3', zorder=7)
          bb = txt.get_window_extent(rnd).expanded(1.08, 1.2)
          ok = axbb.x0 <= bb.x0 and bb.x1 <= axbb.x1 and axbb.y0 <= bb.y0 and bb.y1 <= axbb.y1
          if ok:
            for ln in lines:
              P = pts[id(ln)]
              if np.any((P[:, 0] > bb.x0) & (P[:, 0] < bb.x1) & (P[:, 1] > bb.y0) & (P[:, 1] < bb.y1)):
                ok = False; break
          if ok and any(bb.overlaps(t.get_window_extent(rnd)) for t in texts):
            ok = False
          if not ok:
            txt.remove(); continue
          gl, = ax.plot(xx, 10.**lg, **GUIDE_LS)
          ax.__dict__.setdefault('_guide_lines', []).append(gl)
          ax.__dict__.setdefault('_guide_texts', []).append(txt)
          return True
  return False


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
FINITE_TITLE = r"Impulsive injection, $\Delta t'_{\rm inj}=t'_0$"


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
# {,}: no space after the comma of a subscript (mathtext)
REGIME = {'FC': r'$\gamma_{{\rm m}{,}0}I_{{\rm dyn}{,}0}=10^3$',
          'SC': r'$\gamma_{{\rm m}{,}0}I_{{\rm dyn}{,}0}=10^{-2}$'}
SLOPE_LAB = 'slope'                             # the panels are too low for the fraction
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
    axN.text(.5, 1.03, ('continuous', r"impulsive, $\Delta t'_{\rm inj}=t'_0$")[col], ha='center',
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
G_LAB = {'c,ad': r'$\gamma_{\rm c{,}ad}$', 'm,ad': r'$\gamma_{\rm m{,}ad}$',
         'cFC': r'$\tilde{\gamma}_{\rm c}$', 'cSC': r'$\tilde{\gamma}_{\rm c}$',
         'm,i': r'$\gamma_{\rm m}$', 'c,l': r'$\gamma_{\rm c{,}inj}$',
         'm,l': r'$\gamma_{\rm m{,}inj}$'}
G_NUDGE = {'FC': {r'$-p_{\rm FC}$': 1., r'$-2$': 2., r'$-p$': -1.5, r'$-(p+1)$': -2.},
           'SC': {r'$-p$': 1.5, r'$-(p+1)$': -2.}}
G_SLOPES = {'FC': ((-2./3., r'$-p_{\rm FC}$'), (-2., r'$-2$'), (-P, r'$-p$'),
                   (-(P + 1.), r'$-(p+1)$')),
            'SC': ((.5, r'$-p_{\rm SC}$'), (-P, r'$-p$'), (-(P + 1.), r'$-(p+1)$'))}
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


def _draw_distribution(reg, col, axN, axS, ylim=None):
  ylim = ylim or YLIM_F[reg]
  finite = col == 1
  if finite:
    g, f = f_exact(1. + DINJ, reg, False)
    axN.loglog(g, f, color=GREY, lw=1.4, zorder=2)
  for t in TIMES[(reg, col)]:
    x = 1. + t
    g, f = f_exact(x, reg, finite)
    cur, = axN.loglog(g, f, color=COL[t], lw=1.5, zorder=3)
    s = slope(g, f/g)                           # of dN/dgma, as the text quotes them
    s[~np.isfinite(s) | (s < SL_LIM_G[0]) | (s > SL_LIM_G[1])] = np.nan
    axS.semilogx(g, s, color=COL[t], lw=1.2, zorder=3)
    if not finite and t == T_SPLIT:
      ge, _, fo, fy = old_young(x, reg)
      axN.loglog(ge, np.where(fo > 0, fo, np.nan), **OLD_LS)
      axN.loglog(ge, np.where(fy > 0, fy, np.nan), **YOUNG_LS)
    after = finite and t > DINJ
    if t in OVER_BPL[(reg, col)]:
      pieces, _ = bpl_pieces(x, reg, after)
      gb = np.geomspace(min(q[0] for q in pieces), max(q[1] for q in pieces), 800)
      fb = eval_pieces(gb, pieces)
      axN.loglog(gb, np.where(fb > 0, fb, np.nan), **BPL_LS)
    pieces, _ = bpl_pieces(x, reg, after)
    _draw_guides(axN, [(lo, min(hi, g[-1]), 2.*al) for lo, hi, _, al in pieces], _interp_log(g, f),
                 F_EXP, r'\gamma_{\rm e}', .5, ylim, cur, 0 if t in GUIDES[(reg, col)] else 1, t)
    if t in NAMED[(reg, col)]:
      _, br = bpl_pieces(x, reg, after)
      cb = col if after else 0
      _mark_breaks(axN, lambda v: np.clip(v, g[0]*1.03, g[-1]/1.03), _interp_log(g, f),
                   _named_g(br, reg, cb, g, f), G_BREAKS[(reg, cb)])
  axN.set_ylim(*ylim)
  axS.set_ylim(*SL_LIM_G)


def fig_distribution(outdir=OUTDIR, fname='cooling_shell_integrated.png', show=False):
  return _flat(_draw_distribution, AX_G, G_LAB, G_SLOPES, G_NUDGE, (-4, -2, 0), outdir, fname, show)


# --- Fig. 5 ---------------------------------------------------------------------------
U_LAB = {'c,ad': r"$\nu'_{\rm c{,}ad}$", 'm,ad': r"$\nu'_{\rm m{,}ad}$", 'cFC': r"$\nu'_{\rm c}$",
         'cSC': r"$\nu'_{\rm c}$", 'm,i': r"$\nu'_{\rm m}$", 'c,l': r"$\nu'_{\rm c{,}inj}$",
         'm,l': r"$\nu'_{\rm m{,}inj}$"}
U_NUDGE = {'FC': {r'$1/3$': 3.5, r'$\alpha_{\rm FC}$': -1.5, r'$-1/2$': 2., r'$-(p-1)/2$': -1.}, 'SC': {}}
U_SLOPES = {'FC': ((1./3., r'$1/3$'), (1./6., r'$\alpha_{\rm FC}$'), (-.5, r'$-1/2$'),
                   (-(P - 1.)/2., r'$-(p-1)/2$'), (-P/2., r'$-p/2$')),
            'SC': ((1./3., r'$1/3$'), (-(P - 1.)/2., r'$-(p-1)/2$'), (-P/2., r'$-p/2$'))}
XLIM_U = (1e-1, 3e19)
U = np.geomspace(*XLIM_U, int(np.log10(XLIM_U[1]/XLIM_U[0])*40))
YLIM_L = {'FC': (1e-5, 8.), 'SC': (1e-6, 8.)}
SL_LIM_U = (-1.55, .55)
L_LAB = r"$L'_{\nu'}/(p'_{\rm syn}\mathcal{N}_{\rm e})$"
U_XLAB = r"$\nu'/\nu'_{\rm syn}(\gamma_{\rm e}=1)$"
AX_U = dict(xlim=XLIM_U, xlab=U_XLAB, xlab_small=1, ylab=L_LAB, ticks=(0, 20), step=4)


def _draw_spectrum(reg, col, axN, axS, ylim=None):
  ylim = ylim or YLIM_L[reg]
  finite = col == 1
  if finite:
    g, f = f_exact(1. + DINJ, reg, False)
    axN.loglog(U, spectrum(g, f, U), color=GREY, lw=1.4, zorder=2)
  for t in TIMES[(reg, col)]:
    x = 1. + t
    g, f = f_exact(x, reg, finite)
    L = spectrum(g, f, U)
    cur, = axN.loglog(U, L, color=COL[t], lw=1.5, zorder=3)
    sl = slope(U, L)
    sl[~np.isfinite(sl) | (sl < SL_LIM_U[0]) | (sl > SL_LIM_U[1])] = np.nan
    axS.semilogx(U, sl, color=COL[t], lw=1.2, zorder=3)
    if not finite and t == T_SPLIT:
      ge, _, fo, fy = old_young(x, reg)
      axN.loglog(U, spectrum(ge, fo, U), **OLD_LS)
      axN.loglog(U, spectrum(ge, fy, U), **YOUNG_LS)
    after = finite and t > DINJ
    if t in OVER_BPL[(reg, col)]:
      pieces, _ = bpl_pieces(x, reg, after)
      Lb = css.pl_spectrum(np.sqrt(U), pieces)
      axN.loglog(U, np.where(Lb > 0, Lb, np.nan), **BPL_LS)
    pieces, _ = bpl_pieces(x, reg, after)
    lo0 = min(q[0] for q in pieces)
    segs = [(max(XLIM_U[0]*3., lo0**2/1e4), lo0**2, 1./3.)]
    segs += [(lo**2, min(hi, g[-1])**2, al) for lo, hi, _, al in pieces if al < 1./3.]
    _draw_guides(axN, segs, _interp_log(U, L), U_EXP, r"\nu'", 1.5, ylim, cur,
                 0 if t in GUIDES_U[(reg, col)] else 1, t)
    if t in NAMED[(reg, col)]:
      _, br = bpl_pieces(x, reg, after)
      cb = col if after else 0
      _mark_breaks(axN, lambda v: np.clip(v, *XLIM_U), _interp_log(U, L),
                   {k: v**2 for k, v in _named_g(br, reg, cb, g, f).items()},
                   G_BREAKS[(reg, cb)])
  axN.set_ylim(*ylim)
  axS.set_ylim(*SL_LIM_U)


def fig_spectrum(outdir=OUTDIR, fname='cooling_shell_spectrum.png', show=False):
  return _flat(_draw_spectrum, AX_U, U_LAB, U_SLOPES, U_NUDGE, (-1, 0), outdir, fname, show)


# --- Figs. 4-5 merged per regime (meeting 2026-11-10): distribution above spectrum ----
FIG_REGIME = {'FC': 'cooling_shell_FC.png', 'SC': 'cooling_shell_SC.png'}
YLIM_REG = {'FC': ((1e-5, 8.), (1e-6, 8.)), 'SC': ((1e-8, 8.), (1e-8, 8.))}   # (f_e, L'), author 2026-11-10
# guides placed by hand (author 2026-11-10): (regime, 0 distribution | 1 spectrum, column) -> specs
P2 = -P/2.
GUIDE_MANUAL = {
  ('FC', 0, 0): [dict(t=.3, e=-P, win=(.05, .95), side=-1., fac=2.5)],
  ('FC', 1, 0): [dict(t=.3, e=P2, win=(.05, .95), side=-1., fac=2.5),
                 dict(t=.3, e=-.5, win=(.3, .8), side=-1., fac=3., text='below', dx=1.)],
  ('FC', 1, 1): [dict(t=.3, e=1./3., win=(.1, .45), side=-1., fac=2.5),
                 dict(t=.3, e=-.5, win=(.25, .75), side=1., fac=2.5),
                 dict(t=.3, e=P2, win=(.05, .95), side=1., fac=2.5)],
  ('SC', 0, 0): [dict(t=.3, e=-P, win=(.0, .45), side=-1., fac=2.5)],
  ('SC', 0, 1): [dict(t=.3, e=-P, win=(.2, .7), side=1., fac=3.)],
  ('SC', 1, 0): [dict(t=.3, e=P2, win=(.0, .45), side=-1., fac=2.5)],
  ('SC', 1, 1): [dict(t=.3, e=P2, win=(.2, .7), side=1., fac=3.),
                 dict(t=.3, e=1./3., win=(.1, .4), side=-1., fac=3.)],
}


def fig_regime(reg, outdir=OUTDIR, fname=None, show=False):
  '''one figure per cooling regime: distribution (f_e, slope of dN_e/dgma_e) above the comoving
  spectrum (L'_nu', index), continuous | impulsive injection'''
  fig, axs = plt.subplots(5, 2, figsize=(WIDTH, 5.0),
      gridspec_kw=dict(height_ratios=[2.1, .85, .72, 2.1, .85], hspace=.07, wspace=.05))
  fig.subplots_adjust(left=.075, right=.905, top=.83, bottom=.08)
  for ax in axs[2]:
    ax.set_visible(False)
  for col in (0, 1):
    axs[1, col].sharex(axs[0, col]); axs[4, col].sharex(axs[3, col])
    _draw_distribution(reg, col, axs[0, col], axs[1, col], YLIM_REG[reg][0])
    _draw_spectrum(reg, col, axs[3, col], axs[4, col], YLIM_REG[reg][1])
    axs[0, col]._guide_manual = GUIDE_MANUAL.get((reg, 0, col), [])
    axs[3, col]._guide_manual = GUIDE_MANUAL.get((reg, 1, col), [])
    axs[1, col].set_yticks((-4, -2, 0)); axs[4, col].set_yticks((-1, 0))
    _axes(axs[0, col], axs[1, col], AX_G, col, SLOPE_LAB, 1)
    _axes(axs[3, col], axs[4, col], AX_U, col, 'index', 1)
    axs[0, col].tick_params(labelbottom=False); axs[3, col].tick_params(labelbottom=False)
    axs[1, col].set_xlabel(AX_G['xlab'], fontsize=FS_LAB, labelpad=1.)
    axs[4, col].set_xlabel(AX_U['xlab'], fontsize=FS_LAB - AX_U['xlab_small'], labelpad=1.)
  _guides(axs[1], G_SLOPES[reg], G_NUDGE[reg])
  _guides(axs[4], U_SLOPES[reg], U_NUDGE[reg])
  for ax in (axs[0, 0], axs[0, 1], axs[3, 0], axs[3, 1]):
    _place_guides(ax)
  axs[0, 0].set_title('Continuous injection', fontsize=FS_LAB, pad=3)
  axs[0, 1].set_title(FINITE_TITLE, fontsize=FS_LAB, pad=3)
  p, q = axs[0, 0].get_position(), axs[0, 1].get_position()
  fig.text(.5*(p.x0 + q.x1), p.y1 + .118, {'FC': 'Fast cooling, ', 'SC': 'Slow cooling, '}[reg] + REGIME[reg],
           ha='center', va='bottom', fontsize=FS_LAB, color=INK)
  _key(fig, (.5*(p.x0 + q.x1), p.y1 + .035), ALL_T, KEY_SPLIT, .04, G_LAB)
  return _save(fig, outdir, fname or FIG_REGIME[reg], show)


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
  return _row(_draw_distribution_bpl, AX_G, r'$f_{\rm e}^{\,{\rm BPL}}/f_{\rm e}^{\,{\rm num}}$', KEY_BPL,
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
    if t == NAMED[(reg, col)][-1]:
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
  fig_regime('FC', show=show)
  fig_regime('SC', show=show)
  fig_distribution(show=show)
  fig_spectrum(show=show)
  fig_distribution_bpl(show=show)
  fig_spectrum_bpl(show=show)


if __name__ == '__main__':
  main()
