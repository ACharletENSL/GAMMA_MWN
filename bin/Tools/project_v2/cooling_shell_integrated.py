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
from cooling_shape_figure import OUTDIR   # the shared family folder

P_SYN = 2.5
GM0, GMA_M0 = 1e4, 1e9        # injected bounds at R_0, as the rest of the family (1e3,
                              # 1e8 before 2026-10-21, when the oldest frozen edges fell
                              # below gamma_e = 1)
LAM_EPS = 1e-12               # |Lam| below this is the logarithmic case
LOGC_ROWS = (-2., 2.)
# t'/t'_0 = 1 + Delta t'/t'_0 at round elapsed times Delta t'/t'_0 = 0.1 ... 100 (article 2026-10-23:
# elapsed times since the start of the injection, as in Sect. 2 and Fig. 3)
X_SAMPLES = (1.1, 1.5, 2., 3., 11., 31., 101.)
CASES = (('internal shocks', '$\\Delta t\'_{\\rm inj}=t\'_0$', 1.),
         ('afterglow', '$\\Delta t\'_{\\rm inj}\\gg t\'_0$', np.inf))
# the figure's injected law is the same for every population, so its bounds carry the
# article's collision subscript ,0 (Sect. 2.3) rather than the per-population ,i
NG, NQ, NBIS = 900, 96, 80

INK, MUTED = '0.25', '0.55'
# A&A: no background grid, text on an OPAQUE borderless white box (afterglow_lightcurves)
TEXT_BOX = dict(boxstyle='square,pad=0.15', fc='w', ec='none')
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
def plot_shell_integrated(outdir=OUTDIR, fname='cooling_shell_integrated.png', show=False):
  '''Fig. 4 (fig:intNe_gen), redesigned 2026-10-26: cooling_shell_panels, shared with
  Fig. 5 (layout, cases, times, normalization f_e).'''
  import cooling_shell_panels as csp      # it imports this module: not at module level
  return csp.fig_distribution(outdir, fname, show)


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
