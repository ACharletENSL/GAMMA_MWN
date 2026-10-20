# -*- coding: utf-8 -*-
# @Author: acharlet

'''
The SHELL-INTEGRATED electron distribution (Paper I, Sect. 3): every population present
when the shell is at radius R, summed over its injection radius R_i,

    dNcal/dgma(R) = int_{R_0}^{R_f} Ndot_inj(R_i) dN/dgma(R; R_i) dR_i,
    R_f = min(R, R_0 + Delta R_inj),

with dN/dgma(R; R_i) the NORMALISED distribution of the electrons injected at R_i (Eq.
eqn:dndgma_full) and Ndot_inj = dN_inj/dR_i. Radii are in R_0: x = R/R_0, y = R_i/R_0,
r = R/R_i = x/y.

THE RADIUS CLOCK (Sect. 2.1, Eq. eqn:plaws). After injection every fluid element evolves
self-similarly,

    n' ~ r^-lam_n,   B' ~ r^-lam_B,   Gamma ~ r^-g,

so A = r^(-lam_n/3) and, with dt' = dR/(c Gamma),

    I(R; R_i) = (1 - r^-Lam)/(Lam gbar_c,i),     Lam = lam_n/3 + 2 lam_B - 1 - g,
    gma(R) = A gma_i/(1 + gma_i I)               (ln r/gbar_c,i at Lam = 0).

All indices and gbar_c,i here are RADIAL. The article now works in comoving time (Sect. 2):
its indices are these over (1+g), its s = Lam/(1+g) and its gbar_c,i = (1+g) times this one
(Eq. eqn:plaws_R), so s gbar_c,i = Lam gbar_c,i and I are unchanged. shell_bpl_check maps
them onto Table tab:shell_bpl.

THE SHOCK FRONT sets what each element starts with: gbar_c,i ~ R_i^lam_c, gma_m,i ~
R_i^-lam_m, gma_M,i ~ R_i^lam_M and Ndot_inj ~ R_i^lam_N. A HOMOGENEOUS shell (all elements
see the same B' at a given R) has lam_c = 2 lam_B - 1 - g, which is the default. These
indices, and nothing else, enter the distribution: g^sh only acts through lam_c and lam_m.

THE REFERENCE CASE (the article's figure, COASTING): coasting, constant comoving width,
gma_ad = 5/3 at constant eps_B, i.e. lam_n = 2, lam_B = 5/3, g = 0, Lam = 3, lam_c = 7/3,
constant gma_m,i, gma_M,i and Ndot. There t'/t'_0 = R/R_0, so this reproduces the previous
t'-clock version of this file (d = 2/3, q = 10/3, s = 3) to rounding.
bm_forward_shock(k) gives the Blandford-McKee forward-shock laws (Table tab:indices and
Sect. 4).

THE REGIME LABEL is the article's: C = gbar_c(R_0)/gma_m,0, with gbar_c = 1/I(2R_i; R_i)
= Lam gbar_c,i/(1 - 2^-Lam) the dynamical value of the FIRST element (Eq. eqn:bargamma_c),
so C = 1 is exactly R_m = 2R_0 for that element. The article's dynamical value is now taken
at t' = 2t'_i (Eq. eqn:bargamma_c), i.e. R = 2^(1/(1+g)) R_i: the same for the coasting
figure, not for the BM laws (whose logC here only feeds algebra checks).

THE TWO SEGMENTS that need a spread of injection radii (Sect. 3.2):
  frozen fast cooling   gma ~ R_i^(lam_n/3 + lam_c)  =>  (1 + lam_N)/(lam_n/3 + lam_c) - 1
  adiabatic smearing    gma ~ R_i^(lam_n/3 - lam_m)  =>  (1 + lam_N)/(lam_n/3 - lam_m) - 1
-2/3 and +1/2 in the reference case, (2-7k)/(16+k) and (14-7k)/(4+k) for the BM forward
shock.

NUMERICS. At fixed (x, gma) both edges increase with y in every case of interest (A grows,
I falls; lam_m < lam_n/3 for gma_m), so the support is ONE interval [y_lo, y_hi] with
gma_M(x;y_lo) = gma and gma_m(x;y_hi) = gma. Both are found by vectorised bisection in ln y,
and the integrand, smooth inside, by Gauss-Legendre after y = y_lo + (y_hi - y_lo) u^2.
_check_monotone refuses a set of laws where that assumption fails, rather than returning a
wrong integral. Checked by the SUM RULE int Ncal dgma = int_1^y_f y^lam_N dy (in Ndot_0 R_0).

Two injection cases, the two columns of the figure:
  internal shocks  Delta R_inj = R_0  (shock crosses at ~ the doubling radius)
  afterglow        Delta R_inj -> inf (injection never stops)
and two regimes, the rows: C = 10^-2 (fast) and C = 10^2 (slow).
'''

import os
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

from cooling_distribution import norm_plaw_distrib
from cooling_shape_figure import OUTDIR, named_ticks_top, marks_bottom   # the shared family folder

P_SYN = 2.5
GM0, GMA_M0 = 1e3, 1e8        # injected bounds at R_0, as the rest of the family
LAM_EPS = 1e-12               # |Lam| below this is the logarithmic case
LOGC_ROWS = (-2., 2.)
X_SAMPLES = (1.1, 1.5, 2., 3., 10., 30., 100.)
CASES = (('internal shocks', '$\\Delta t\'_{\\rm inj}=t\'_0$', 1.),
         ('afterglow', '$\\Delta t\'_{\\rm inj}\\gg t\'_0$', np.inf))
# the figure's injected law is the same for every population, so its bounds carry the
# article's collision subscript ,0 (Sect. 2.3) rather than the per-population ,i
NG, NQ, NBIS = 900, 96, 80

INK, MUTED = '0.25', '0.55'
# A&A: no background grid, text on an OPAQUE borderless white box (afterglow_lightcurves)
TEXT_BOX = dict(boxstyle='square,pad=0.15', fc='w', ec='none')
# THE NAMED BREAKS (Table tab:shell_bpl), on ONE reference curve per panel, with the
# table's broken power law for that curve dashed in ink (shell_bpl_check.table). The
# continuous FC panel takes x = 3, before R_tr = 7.2 R_0 where fresh injection turns slow
# and the FC rows stop applying; the continuous SC one x = 10, where gma_b is still
# below gma_M,0 (at x = 100 it is not). The finite ones the last curve.
REF_X = {(0, 0): 100., (0, 1): 3., (1, 0): 100., (1, 1): 10.}    # (row, column) -> x
# finite-injection curves drawn DASHED over the continuous FC panel: the same oldest
# elements, so they lie on its low end (author's note on the figure)
OVERLAY_X = (3., 30.)
BREAK_LABS = {'low': '$\\gamma_{\\rm low}$', 'fr': '$\\gamma_{\\rm fr}$',
              'mi': '$\\gamma_{\\mathrm{m},\\!i}$', 'b': '$\\gamma_{\\rm b}$'}
FINITE_LABS = {'FC': {'low': "$\\gamma_\\infty(t';t'_0)$",
                      'hi': "$\\gamma_\\infty(t';t'_{\\rm end})$"},
               'SC': {'low': "$\\gamma_\\mathrm{m}(t';t'_{\\rm end})$",
                      'hi': "$\\gamma_\\infty(t';t'_0)$"}}
# where each break's name goes, per (row, column, break), set by eye on the render so that
# no box sits on a curve: ('off', dx, dy) in points from the mark, or ('at', gma_e,
# axes-y) in free space with a thin leader line back to the mark. gma_m,i has no entry:
# with lam_m = 0 it IS gma_m,0, and the dotted line there is named for both
BREAK_AT = {(0, 0, 'low'): ('at', .5, .925), (0, 0, 'hi'): ('at', 80., .925),
            (0, 1, 'low'): ('at', .9, .925), (0, 1, 'fr'): ('at', 40., .925),
            (1, 0, 'low'): ('off', -6, -6), (1, 0, 'hi'): ('at', 5e5, .66),
            (1, 1, 'low'): ('off', -6, -6), (1, 1, 'b'): ('at', 4e7, .45)}
TOP_ROOM = (300., 8.)          # ylim top over the curves' max, per row: the FC row's names
                              # sit in a band above its curves
FIGSIZE = (7.0, 8.4)
FS_LAB, FS_TICK, FS_ANN, FS_LEG = 9., 8., 7.5, 7.
GMA_LABEL = '$\\gamma_{\\rm e}$'


# --- the power laws -------------------------------------------------------------------
@dataclass(frozen=True)
class ShellLaws:
  '''
  Element evolution (lam_n, lam_B, g) and shock-front laws (lam_c, lam_m, lam_M, lam_N),
  all as defined in the article. lam_c = None is the homogeneous shell, 2 lam_B - 1 - g.
  '''
  lam_n: float = 2.
  lam_B: float = 5./3.
  g: float = 0.
  lam_c: float = None
  lam_m: float = 0.
  lam_M: float = 0.
  lam_N: float = 0.
  label: str = 'coasting'

  def __post_init__(self):
    if self.lam_c is None:
      object.__setattr__(self, 'lam_c', 2.*self.lam_B - 1. - self.g)

  @property
  def Lam(self):
    '''the freezing index, Eq. eqn:I_plaw'''
    return self.lam_n/3. + 2.*self.lam_B - 1. - self.g

  @property
  def frozen_index(self):
    '''frozen fast-cooling segment, gma ~ R_i^(lam_n/3 + lam_c)'''
    return (1. + self.lam_N)/(self.lam_n/3. + self.lam_c) - 1.

  @property
  def smeared_index(self):
    '''adiabatically smeared segment below gma_m,i, gma ~ R_i^(lam_n/3 - lam_m)'''
    return (1. + self.lam_N)/(self.lam_n/3. - self.lam_m) - 1.

  def R_tr(self, logC):
    '''R_tr/R_0 where fresh injection turns from fast to slow cooling'''
    return 10.**(-logC/(self.lam_c + self.lam_m))


COASTING = ShellLaws()


def bm_forward_shock(k=0., lam_M=.75):
  '''
  Blandford-McKee forward shock in n_ext ~ R^-k: elements follow the BM row of Table
  tab:indices, injection follows the shock (Sect. 4), g^sh = (3-k)/2, so
  lam_c = k + g^sh - 1, lam_m = g^sh and lam_N = 2 - k (Ndot = 4 pi R^2 n_ext).
  lam_M = 3/4 is the Bohm-limited gma_M,i ~ B_i'^-1/2 with B_i' ~ Gamma_i n_ext^1/2; it is
  a choice of acceleration model, not a consequence of the hydro.
  '''
  gsh = (3. - k)/2.
  return ShellLaws(lam_n=(13. - 2.*k)/2., lam_B=(13. - 2.*k)/3., g=(7. - 2.*k)/2.,
                   lam_c=k + gsh - 1., lam_m=gsh, lam_M=lam_M, lam_N=2. - k,
                   label=f'BM forward shock, k={k:g}')


# --- one population -------------------------------------------------------------------
def gbar_ci0_of_C(logC, laws=COASTING, gm0=GM0):
  '''gbar_c,i of the first element from C = gbar_c(R_0)/gma_m,0, Eq. eqn:bargamma_c.'''
  L = laws.Lam
  f = np.log(2.) if abs(L) < LAM_EPS else (1. - 2.**(-L))/L     # = gbar_c,i I(2R_i)
  return 10.**logC*gm0*f


def A_of(x, y, laws=COASTING):
  '''adiabatic shift of the element injected at y, seen at x'''
  return (y/x)**(laws.lam_n/3.)


def I_of(x, y, gci0, laws=COASTING):
  '''I(R; R_i), Eq. eqn:I_plaw, with gbar_c,i = gci0 y^lam_c'''
  r = x/y
  gci = gci0*y**laws.lam_c
  L = laws.Lam
  if abs(L) < LAM_EPS:
    return np.log(r)/gci
  return (1. - r**(-L))/(L*gci)


def inj_bounds(y, laws=COASTING, gm0=GM0, gM0=GMA_M0):
  '''gma_m,i and gma_M,i of the element injected at y'''
  return gm0*y**(-laws.lam_m), gM0*y**laws.lam_M


def edge(x, y, gci0, which, laws=COASTING, gm0=GM0, gM0=GMA_M0):
  '''gma_m (which='m') or gma_M ('M') at x of the element injected at y, Eq. eqn:gma_t.'''
  gi = inj_bounds(y, laws, gm0, gM0)[0 if which == 'm' else 1]
  return A_of(x, y, laws)*gi/(1. + gi*I_of(x, y, gci0, laws))


def dNdg_pop(gma, x, y, gci0, laws=COASTING, p=P_SYN, gm0=GM0, gM0=GMA_M0):
  '''One population, normalised to one electron, zero off its support.'''
  gmi, gMi = inj_bounds(y, laws, gm0, gM0)
  K = norm_plaw_distrib(gmi, gMi, p)
  A = A_of(x, y, laws)
  I = I_of(x, y, gci0, laws)
  inside = (gma >= A*gmi/(1. + gmi*I)) & (gma <= A*gMi/(1. + gMi*I))
  core = np.clip(1. - gma*I/A, 0., None)
  return np.where(inside, K*A**(p - 1.)*gma**(-p)*core**(p - 2.), 0.)


# --- the shell integral ---------------------------------------------------------------
def _bisect_y(fun, target, lo, hi, n=NBIS):
  '''y in [lo, hi] with fun(y) = target, fun increasing; vectorised, in ln y.'''
  a, b = np.log(lo)*np.ones_like(target), np.log(hi)*np.ones_like(target)
  for _ in range(n):
    m = .5*(a + b)
    up = fun(np.exp(m)) < target
    a, b = np.where(up, m, a), np.where(up, b, m)
  return np.exp(.5*(a + b))


def _check_monotone(x, yf, gci0, laws, gm0, gM0, n=257):
  '''both edges must increase with y on [1, yf], or the support is not one interval'''
  if yf <= 1.:
    return
  ys = np.geomspace(1., yf, n)
  for which in ('m', 'M'):
    e = edge(x, ys, gci0, which, laws, gm0, gM0)
    if np.any(np.diff(e) < -1e-12*e[:-1]):
      raise ValueError(f'gma_{which}(x={x:g}; y) is not increasing in y for {laws}: '
                       'the support is not a single interval in R_i')


def shell_integrated(gma, x, logC, dinj, laws=COASTING, p=P_SYN, gm0=GM0, gM0=GMA_M0,
    nq=NQ):
  '''
  Ncal(gma; x) / (Ndot_0 R_0), injection over y in [1, min(x, 1 + dinj)], dinj in R_0.
  '''
  gma = np.asarray(gma, dtype=float)
  gci0 = gbar_ci0_of_C(logC, laws, gm0)
  yf = min(x, 1. + dinj)
  _check_monotone(x, yf, gci0, laws, gm0, gM0)
  e_m = lambda y: edge(x, y, gci0, 'm', laws, gm0, gM0)
  e_M = lambda y: edge(x, y, gci0, 'M', laws, gm0, gM0)
  y_lo = _bisect_y(e_M, gma, 1., yf)
  y_hi = _bisect_y(e_m, gma, 1., yf)
  # outside the range of either edge the root saturates at a bound: test directly
  y_lo = np.where(e_M(1.) >= gma, 1., y_lo)
  y_hi = np.where(e_m(yf) <= gma, yf, y_hi)
  ok = (y_hi > y_lo) & (gma <= e_M(yf)) & (gma >= e_m(1.))
  u, w = np.polynomial.legendre.leggauss(nq)
  u, w = .5*(u + 1.), .5*w
  dy = (y_hi - y_lo)[:, None]
  yy = y_lo[:, None] + dy*u[None, :]**2
  f = yy**laws.lam_N*dNdg_pop(gma[:, None], x, yy, gci0, laws, p, gm0, gM0)
  N = np.sum(f*2.*u[None, :]*w[None, :], axis=1)*dy[:, 0]
  return np.where(ok, N, 0.)


def injected_number(x, dinj, laws=COASTING):
  '''int_1^y_f y^lam_N dy: what the shell integral must sum to, in Ndot_0 R_0'''
  yf = min(x, 1. + dinj)
  a = 1. + laws.lam_N
  return np.log(yf) if abs(a) < LAM_EPS else (yf**a - 1.)/a


def support(x, logC, dinj, laws=COASTING, gm0=GM0, gM0=GMA_M0):
  '''lowest and highest gma present in the shell at x.'''
  gci0 = gbar_ci0_of_C(logC, laws, gm0)
  yf = min(x, 1. + dinj)
  ys = np.geomspace(1., yf, 400) if yf > 1. else np.array([1.])
  return (float(edge(x, ys, gci0, 'm', laws, gm0, gM0).min()),
          float(edge(x, ys, gci0, 'M', laws, gm0, gM0).max()))


def shell_curve(x, logC, dinj, ng=NG, laws=COASTING, **kw):
  lo, hi = support(x, logC, dinj, laws)
  gma = np.geomspace(lo*(1. + 1e-9), hi*(1. - 1e-9), ng)
  return gma, shell_integrated(gma, x, logC, dinj, laws, **kw)


def log_slope(gma, N):
  with np.errstate(divide='ignore', invalid='ignore'):
    return np.gradient(np.log(N), np.log(gma))


# --- checks ---------------------------------------------------------------------------
def check_sum_rule(laws=COASTING, ng=6000, xs=X_SAMPLES):
  '''int Ncal dgma / injected_number -> 1: number conservation, every (case, C, x).'''
  worst = 0.
  for _, _, dinj in CASES:
    for lc in LOGC_ROWS:
      for x in xs:
        g, N = shell_curve(x, lc, dinj, ng=ng, laws=laws)
        tot = np.trapezoid(N, g)
        worst = max(worst, abs(tot/injected_number(x, dinj, laws) - 1.))
  return worst


def _slope_at(g, x, lc, dinj, laws):
  gg = g*np.array([.97, 1.03])
  N = shell_integrated(gg, x, lc, dinj, laws)
  return float((np.diff(np.log(N))/np.diff(np.log(gg)))[0])


def check_slopes(laws=COASTING):
  '''the asymptotic indices of the text, measured on the afterglow (reference case).'''
  out = []
  gF = gbar_ci0_of_C(-2., laws)
  g0 = laws.Lam*gF*100.**(-laws.lam_n/3.)     # oldest (frozen) edge of the fast afterglow
  for lab, lc, x, g, exp in (
      ('FC fresh -2', -2., 3., 500., -2.),
      ('FC frozen', -2., 100., 3.*g0, laws.frozen_index),
      ('SC smeared', 2., 100., GM0*100.**(-laws.lam_n/6.), laws.smeared_index),
      ('SC injected -p', 2., 100., 3.*GM0, -P_SYN),
      ('SC cooled -(p+1)', 2., 1.5, 3e-2*GMA_M0, -(P_SYN + 1.))):
    out.append((f'{lab} (x={x:g})', g, _slope_at(g, x, lc, np.inf, laws), exp))
  return out


def check_bm_slopes(k, x=300., logC_fast=-3., logC_slow=2.):
  '''
  The two spread-of-injection segments on the BM forward shock, continuous injection.
  Frozen: 10x above the oldest relic's edge, with R_tr = C^-1/2 R_0 ~ 30 R_0 so that the
  relics span y = 1..~15. Smeared: in slow cooling, at the geometric middle of
  [gma_m(x;1), gma_m,i(x)], where the R_i = R_0 lower limit is negligible.
  At x = 300 these sit below gma = 1 (the BM drag A is steep): a check of the algebra,
  which does not care, not of a physical spectrum.
  '''
  laws = bm_forward_shock(k)
  gF = gbar_ci0_of_C(logC_fast, laws)
  g_frz = 10.*edge(x, 1., gF, 'M', laws)
  lo, hi = edge(x, 1., gbar_ci0_of_C(logC_slow, laws), 'm', laws), GM0*x**(-laws.lam_m)
  g_sm = np.sqrt(lo*hi)
  return laws, ((f'FC frozen (x={x:g})', g_frz,
                 _slope_at(g_frz, x, logC_fast, np.inf, laws), laws.frozen_index),
                (f'SC smeared (x={x:g})', g_sm,
                 _slope_at(g_sm, x, logC_slow, np.inf, laws), laws.smeared_index))


# --- the figure -----------------------------------------------------------------------
def _reference_bpl(axN, row, col, x, lc, dinj, laws, p, regime):
  '''
  The reference curve's broken power law (Table tab:shell_bpl, via shell_bpl_check),
  dashed in ink, and its breaks named on the EXACT curve with a short vertical mark.
  Returns nothing; draws on axN. The table's dN/(N dln gma) times N/gma is the plotted
  (Ndot_0 t'_0)^-1 dNcal/dgma (coasting: R_0 units are t'_0 units).
  '''
  import shell_bpl_check as sbc       # it imports this module: not at module level
  c = sbc.comoving(laws, 0., x, lc, p, gm0=GM0)
  tau_end = None if np.isinf(dinj) else 1. + dinj
  lo, hi = support(x, lc, dinj, laws)
  g = np.geomspace(lo, hi/3., 600)
  f, _, br, _ = sbc.table(g, c, regime, tau_end)
  ntot = injected_number(x, dinj, laws)
  m = f > 0.
  axN.loglog(g[m], ntot*f[m]/g[m], color=INK, ls=(0, (3, 2)), lw=.9, zorder=5)
  labs = BREAK_LABS if tau_end is None else FINITE_LABS[regime]
  for key in ('low', 'fr', 'mi', 'b', 'hi'):
    if key not in br:                 # FC has gma_fr, SC gma_b; finite only low, hi
      continue
    gb = br[key]
    # on the exact curve; an edge sits where the curve stops, so step just inside it
    gq = np.clip(gb, lo*1.15, hi/1.15)
    yb = float(shell_integrated(np.array([gq]), x, lc, dinj, laws, p)[0])
    axN.plot([gq], [yb], marker='|', ms=9, mew=1.1, color=INK, zorder=6)
    where = BREAK_AT.get((row, col, key))
    if where is None:
      continue
    if where[0] == 'off':
      dx, dy = where[1:]
      axN.annotate(labs[key], (gq, yb), textcoords='offset points', xytext=(dx, dy),
                   ha='left' if dx >= 0 else 'right', va='bottom' if dy >= 0 else 'top',
                   color=INK, fontsize=FS_ANN, bbox=TEXT_BOX, zorder=7)
    else:
      axN.annotate(labs[key], (gq, yb), xytext=where[1:],
                   textcoords=('data', 'axes fraction'), ha='center', va='center',
                   color=INK, fontsize=FS_ANN, bbox=TEXT_BOX, zorder=7,
                   arrowprops=dict(arrowstyle='-', color=INK, lw=.5, shrinkA=0,
                                   shrinkB=5))


def plot_shell_integrated(outdir=OUTDIR, fname='cooling_shell_integrated.png', show=False,
    p=P_SYN, laws=COASTING):
  '''
  Per regime (fast on top, slow below): the distributions, and their local slope
  underneath, with the indices the text derives as guides. Columns are the two
  injection cases. A blank spacer row separates the two regimes.

  On each distribution panel ONE reference curve (REF_X) carries the breaks of Table
  tab:shell_bpl, named, and the table's broken power law dashed in ink. The continuous
  fast-cooling panel also carries two finite-injection curves (OVERLAY_X), dashed in
  their time colour: they are its oldest elements, and lie on its low end.
  A&A style: no grid, no titles (each case is written inside its panel), opaque boxes.
  '''
  lx = np.log10(X_SAMPLES)
  norm = plt.Normalize(vmin=lx.min(), vmax=lx.max())
  cmap = plt.cm.Blues
  col = lambda x: cmap(.35 + .65*norm(np.log10(x)))
  # expected indices per regime row: the steady-state ones and the two that need a
  # spread of injection radii (frozen relics in fast, smearing in slow cooling)
  levels = {LOGC_ROWS[0]: ((laws.frozen_index, '$-2/3$'), (-2., '$-2$'),
                           (-(p + 1.), '$-(p+1)$')),
            LOGC_ROWS[1]: ((laws.smeared_index, '$+1/2$'), (-p, '$-p$'),
                           (-(p + 1.), '$-(p+1)$'))}
  # continuous injection turns slow after R_tr, so its fast-cooling panel ends on -p
  extra = {(LOGC_ROWS[0], 1): ((-p, '$-p$'),)}
  SL_LIM = (-(p + 2.4), 1.6)
  fig, axs = plt.subplots(5, 2, figsize=FIGSIZE, sharex=True,
      gridspec_kw=dict(height_ratios=[1.9, 1., .13, 1.9, 1.], hspace=.08, wspace=.06))
  for ax in axs[2]:
    ax.set_visible(False)
  rows = ((axs[0], axs[1]), (axs[3], axs[4]))
  for i, lc in enumerate(LOGC_ROWS):
    axN_row, axS_row = rows[i]
    regime = 'FC' if lc < 0. else 'SC'
    top = 0.
    for j, (name, lab, dinj) in enumerate(CASES):
      axN, axS = axN_row[j], axS_row[j]
      for x in X_SAMPLES:
        g, N = shell_curve(x, lc, dinj, laws=laws)
        axN.loglog(g, N, color=col(x), lw=1.2, zorder=3)
        sl = log_slope(g, N)
        # the support edges are hard, so the slope runs off to +-inf there: blank what
        # leaves the panel instead of drawing the jump as a vertical line
        sl[~np.isfinite(sl) | (sl > SL_LIM[1]) | (sl < SL_LIM[0])] = np.nan
        axS.semilogx(g, sl, color=col(x), lw=1.1, zorder=3)
        top = max(top, N.max())
      if i == 0 and np.isinf(dinj):
        for x in OVERLAY_X:
          g, N = shell_curve(x, lc, CASES[0][2], laws=laws)
          axN.loglog(g, N, color=col(x), lw=1.2, ls=(0, (2.5, 1.5)), zorder=4)
      _reference_bpl(axN, i, j, REF_X[(i, j)], lc, dinj, laws, p, regime)
      lev_ij = tuple(sorted(levels[lc] + extra.get((lc, j), ()), reverse=True))
      for lev, _ in lev_ij:
        axS.axhline(lev, color=MUTED, ls='--', lw=.7, zorder=1)
      axS.set_ylim(*SL_LIM)
      for ax in (axN, axS):
        ax.axvspan(1e-30, 1., color='crimson', alpha=.07, lw=0, zorder=0)
        ax.axvline(1., color='crimson', ls=':', lw=.9, zorder=1)
        for v in (GM0, GMA_M0):
          ax.axvline(v, color=INK, ls=':', lw=.8, zorder=1)
        ax.tick_params(which='both', labelsize=FS_TICK)
      # the case, INSIDE the panel (no titles), top right where the curves have fallen
      # away; C is the DYNAMICAL value of the first element, Eq. eqn:bargamma_c
      axN.annotate(f'{lab}\n$\\bar{{\\gamma}}_{{\\rm c}}(t\'_0)=10^{{{lc:+.0f}}}'
                   f'\\gamma_{{\\mathrm{{m}},\\!0}}$', (.98, .97), xycoords='axes fraction',
                   fontsize=FS_ANN, color=INK, ha='right', va='top', multialignment='right',
                   bbox=TEXT_BOX, zorder=7)
      if j == 0:
        axN.set_ylabel('$(\\dot N_0 t\'_0)^{-1}\\,{\\rm d}\\mathcal{N}_{\\rm e}'
                       '/{\\rm d}\\gamma_{\\rm e}$', fontsize=FS_LAB)
        axS.set_ylabel('${\\rm d}\\ln\\mathcal{N}_{\\rm e}/{\\rm d}\\ln\\gamma_{\\rm e}$',
                       fontsize=FS_LAB)
      else:
        axN.tick_params(labelleft=False)
        axS.tick_params(labelleft=False)
        # the expected indices are reference VALUES: label them on the right spine
        axR = axS.twinx()
        axR.set_ylim(axS.get_ylim())
        axR.set_yticks([lev for lev, _ in lev_ij])
        axR.set_yticklabels([t for _, t in lev_ij])
        # -2 and -p are half an index apart: lift the upper label, drop the lower one
        tl = axR.get_yticklabels()
        for k in range(len(lev_ij) - 1):
          if lev_ij[k][0] - lev_ij[k + 1][0] < .8:
            tl[k].set_va('bottom')
            tl[k + 1].set_va('top')
        axR.tick_params(axis='y', labelsize=FS_ANN, length=2.5, pad=1.5, colors=INK)
    for ax in axN_row:
      ax.set_ylim(top*1e-12, top*TOP_ROOM[i])
  for ax in axs[4]:
    ax.set_xlabel(GMA_LABEL, fontsize=FS_LAB)
  axs[0, 0].set_xlim(.05, 3.*GMA_M0)
  # the injected bounds are NAMED TICKS on the top axis of each regime block (user,
  # 2026-10-20), which leaves the panels to the curves and the break names. The
  # continuous column names gma_m,0 for gma_m,i(t') too: with lam_m = 0 they coincide
  # named ONCE, on the slow-cooling row's top axis in the thin gap between the two
  # regime blocks, with bare marks pointing down from the block above (user, 2026-10-20)
  for c, lm in ((0, '$\\gamma_{\\mathrm{m},\\!0}$'),
                (1, '$\\gamma_{\\mathrm{m},\\!i}=\\gamma_{\\mathrm{m},\\!0}$')):
    named_ticks_top(axs[3, c], ((GM0, lm), (GMA_M0, '$\\gamma_{\\mathrm{M},\\!0}$')))
    marks_bottom(axs[1, c], (GM0, GMA_M0))
  fig.subplots_adjust(right=.84)
  # line-style key ABOVE the panels, where it covers nothing; the colour is the time
  h = [plt.Line2D([], [], color=MUTED, lw=1.2, ls=(0, (2.5, 1.5)),
                  label="$\\Delta t'_{\\rm inj}=t'_0$, over the continuous case"),
       plt.Line2D([], [], color=INK, lw=.9, ls=(0, (3, 2)),
                  label='broken power law (one curve)'),
       plt.Line2D([], [], color=INK, lw=0, marker='|', ms=7, mew=1.1, label='its breaks')]
  pa, pb = axs[0, 0].get_position(), axs[0, 1].get_position()
  fig.canvas.draw()                      # above the named ticks, which need a layout
  r_ = fig.canvas.get_renderer()
  top = max(fig.transFigure.inverted().transform(ax.get_tightbbox(r_))[1, 1]
            for ax in axs[0])
  fig.legend(handles=h, loc='lower center', ncol=3, bbox_to_anchor=(.5*(pa.x0 + pb.x1),
             top + .004), bbox_transform=fig.transFigure, fontsize=FS_LEG,
             frameon=False, handlelength=2., columnspacing=1.2)
  p0, p1 = axs[0, 1].get_position(), axs[4, 1].get_position()
  cax = fig.add_axes([.935, p1.y0, .018, p0.y1 - p1.y0])
  sm = plt.cm.ScalarMappable(cmap=plt.cm.colors.ListedColormap(
      cmap(np.linspace(.35, 1., 256))), norm=norm)
  cb = fig.colorbar(sm, cax=cax)
  # coasting: t'/t'_0 = R/R_0, which is what X_SAMPLES are
  cb.set_label("$\\log_{10}(t'/t'_0)$", fontsize=FS_LAB)
  cax.tick_params(labelsize=FS_TICK)

  os.makedirs(outdir, exist_ok=True)
  path = os.path.join(outdir, fname)
  fig.savefig(path, dpi=300, bbox_inches='tight')
  print(f'saved {path}')
  if show:
    plt.show()
  return fig, axs


def main(show=False):
  L = COASTING
  print(f'{L.label}: Lam = {L.Lam:g}, lam_c = {L.lam_c:.4g}; gbar_c,i(R_0): '
        f'C=1e-2 -> {gbar_ci0_of_C(-2.):.4g}, C=1e2 -> {gbar_ci0_of_C(2.):.4g}; '
        f'I(2R_i) gbar_c,i = {(1. - 2.**(-L.Lam))/L.Lam:.4f}')
  print(f'sum rule: max |int Ncal dgma/injected - 1| = {check_sum_rule():.2e}')
  for lab, g, meas, exp in check_slopes():
    print(f'  {lab:26s} at gma={g:9.3e}: {meas:+.4f} (expected {exp:+.4f})')
  for k in (0., 2.):
    laws, rows = check_bm_slopes(k)
    print(f'{laws.label}: Lam = {laws.Lam:.4g}, lam_c = {laws.lam_c:g}, '
          f'R_tr(C=1e-3) = {laws.R_tr(-3.):.3g} R_0')
    for lab, g, meas, exp in rows:
      print(f'  {lab:26s} at gma={g:9.3e}: {meas:+.4f} (expected {exp:+.4f})')
  plot_shell_integrated(show=show)


if __name__ == '__main__':
  main()
