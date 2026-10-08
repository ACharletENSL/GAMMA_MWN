# -*- coding: utf-8 -*-
# @Author: acharlet

'''
The SHELL-INTEGRATED electron distribution: every population present at one comoving
time t', summed over its injection time t'_i,

    dNcal/dgma(t') = int_{t'_0}^{t'_f} dN/dgma(t'; t'_i) dt'_i,
    t'_f = min(t', t'_0 + Delta t'_inj),

for a HOMOGENEOUS shell: every fluid element sees the same B'(t') and the same
expansion, so the integral over position is an integral over t'_i alone.

THE POWER LAWS (cooling_integrated_adiabatic's, positive exponents). With tau = t'/t'_i,

    A = tau^-d                    the adiabatic factor, d = 2/3     (rho ~ t'^-2)
    t'_c = t'_c,i tau^q           the synchrotron clock, q = 10/3   (B'^2 ~ rho^5/3)
    s = d + q - 1 = 3

and the f-weighted clock of the appendix integrates in closed form,

    A tt_eff = S(t'; t'_i) = (t'_i/t'_c,i) (1 - tau^-s)/s,

bounded since s > 0: every population's burn FREEZES after a few t'_i and from then on
it only slides down as A. Homogeneity ties the populations together through
t'_c,i = t'_c,0 (t'_i/t'_0)^q, i.e. t'_i/t'_c,i = kappa y^(1-q), with y = t'_i/t'_0,
x = t'/t'_0 and kappa = t'_0/t'_c,0.

ONE population (appendix Eqn. app_dndgma_full, per unit injection time, constant rate
Ndot and constant injected law):

    dN/dgma = Ndot K A^(p-1) gma^-p (1 - gma S/A)^(p-2),
    gma_m,M(x;y) = A gma_{m,M},i / (1 + gma_{m,M},i S).

THE REGIME LABEL is the article's: gma_c,0 = 1/I over the first t'_0 after collision,
synchrotron AND adiabatic, i.e. kappa (1 - 2^-s)/s = 0.2917 kappa = 1/(C gma_m,i), so
C = 1 is exactly t'_m = t'_dyn for the first fluid element. The 0.2917 is <A B'^2>/B'_0^2
over [t'_0, 2t'_0]. (The previous label, 1/tt, used the A-free 0.3435 = <B'^2>/B'_0^2:
the same C now means 1.178x more cooling.)

NUMERICS. At fixed (x, gma) both edges are increasing in y (A grows, S falls), so the
support is ONE interval [y_lo, y_hi] with gma_M(x;y_lo) = gma and gma_m(x;y_hi) = gma.
Both are found by vectorised bisection in ln y, and the integrand -- smooth inside, a
(1 - gma S/A)^(p-2) square-root zero at y_lo -- by Gauss-Legendre after y = y_lo +
(y_hi - y_lo) u^2. Checked by the SUM RULE int Ncal dgma = Ndot (y_f - 1) t'_0.

Two cases, the two columns of the figure:
  internal shocks  Delta t'_inj = t'_0  (shock crosses at ~ the doubling radius)
  afterglow        Delta t'_inj -> inf  (injection never stops)
and two regimes, the rows: C = 10^-2 (fast) and C = 10^2 (slow).
'''

import os
import numpy as np
import matplotlib.pyplot as plt

from cooling_distribution import norm_plaw_distrib
from cooling_shape_figure import OUTDIR          # the shared family folder

P_SYN = 2.5
GM0, GMA_M0 = 1e3, 1e8        # injected bounds, as the rest of the family
D_AD, Q_C = 2./3., 10./3.     # A = tau^-d, t'_c ~ tau^q
S_EXP = D_AD + Q_C - 1.
LOGC_ROWS = (-2., 2.)
X_SAMPLES = (1.1, 1.5, 2., 3., 10., 30., 100.)
CASES = (('internal shocks', '$\\Delta t\'_{\\rm inj}=t\'_0$', 1.),
         ('afterglow', '$\\Delta t\'_{\\rm inj}\\gg t\'_0$', np.inf))
# the injected law is the same for every population, so its bounds carry the article's
# collision subscript ,0 (Sect. 2.3) rather than the per-population ,i
NG, NQ, NBIS = 900, 96, 80

INK, MUTED = '0.25', '0.55'
FIGSIZE = (7.0, 8.4)
FS_LAB, FS_TICK, FS_ANN, FS_LEG = 9., 8., 7.5, 7.
GMA_LABEL = '$\\gamma_{\\rm e}$'


# --- one population -------------------------------------------------------------------
def kappa_of_C(logC, q=Q_C, gm0=GM0):
  '''kappa = t'_0/t'_c,0 from C: gma_c,0 = 1/I(2t'_0; t'_0) = C gma_m,i.'''
  s = D_AD + q - 1.
  return s/((1. - 2.**(-s))*10.**logC*gm0)


def S_of(x, y, kap, d=D_AD, q=Q_C):
  '''A tt_eff for the population injected at y, seen at x (both in t'_0).'''
  s = d + q - 1.
  return kap*y**(1. - q)*(1. - (y/x)**s)/s


def edges(x, y, kap, gi, d=D_AD, q=Q_C):
  '''gma(x) of an electron injected at y with gma_i (gi), Eqn. app_gma_exact.'''
  A = (y/x)**d
  return A*gi/(1. + gi*S_of(x, y, kap, d, q))


def dNdg_pop(gma, x, y, kap, p=P_SYN, gm0=GM0, gM0=GMA_M0, d=D_AD, q=Q_C):
  '''One population per unit injection time (Ndot = 1), zero off its support.'''
  K = norm_plaw_distrib(gm0, gM0, p)
  A = (y/x)**d
  S = S_of(x, y, kap, d, q)
  inside = (gma >= edges(x, y, kap, gm0, d, q)) & (gma <= edges(x, y, kap, gM0, d, q))
  core = np.clip(1. - gma*S/A, 0., None)
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


def shell_integrated(gma, x, logC, dinj, p=P_SYN, gm0=GM0, gM0=GMA_M0, d=D_AD, q=Q_C,
    nq=NQ):
  '''
  Ncal(gma; x) / (Ndot t'_0), injection over y in [1, min(x, 1 + dinj)].
  '''
  gma = np.asarray(gma, dtype=float)
  kap = kappa_of_C(logC, q, gm0)
  yf = min(x, 1. + dinj)
  y_lo = _bisect_y(lambda y: edges(x, y, kap, gM0, d, q), gma, 1., yf)
  y_hi = _bisect_y(lambda y: edges(x, y, kap, gm0, d, q), gma, 1., yf)
  # outside the range of either edge the root saturates at a bound: test directly
  y_lo = np.where(edges(x, 1., kap, gM0, d, q) >= gma, 1., y_lo)
  y_hi = np.where(edges(x, yf, kap, gm0, d, q) <= gma, yf, y_hi)
  ok = (y_hi > y_lo) & (gma <= edges(x, yf, kap, gM0, d, q)) \
       & (gma >= edges(x, 1., kap, gm0, d, q))
  u, w = np.polynomial.legendre.leggauss(nq)
  u, w = .5*(u + 1.), .5*w
  dy = (y_hi - y_lo)[:, None]
  yy = y_lo[:, None] + dy*u[None, :]**2
  f = dNdg_pop(gma[:, None], x, yy, kap, p, gm0, gM0, d, q)
  N = np.sum(f*2.*u[None, :]*w[None, :], axis=1)*dy[:, 0]
  return np.where(ok, N, 0.)


def support(x, logC, dinj, gm0=GM0, gM0=GMA_M0, d=D_AD, q=Q_C):
  '''lowest and highest gma present in the shell at x.'''
  kap = kappa_of_C(logC, q, gm0)
  yf = min(x, 1. + dinj)
  ys = np.geomspace(1., yf, 400) if yf > 1. else np.array([1.])
  return (float(edges(x, ys, kap, gm0, d, q).min()),
          float(edges(x, ys, kap, gM0, d, q).max()))


def shell_curve(x, logC, dinj, ng=NG, **kw):
  lo, hi = support(x, logC, dinj)
  gma = np.geomspace(lo*(1. + 1e-9), hi*(1. - 1e-9), ng)
  return gma, shell_integrated(gma, x, logC, dinj, **kw)


def log_slope(gma, N):
  with np.errstate(divide='ignore', invalid='ignore'):
    return np.gradient(np.log(N), np.log(gma))


# --- checks ---------------------------------------------------------------------------
def check_sum_rule(ng=6000):
  '''int Ncal dgma / (y_f - 1) -> 1: number conservation, every (case, C, x).'''
  worst = 0.
  for _, _, dinj in CASES:
    for lc in LOGC_ROWS:
      for x in X_SAMPLES:
        g, N = shell_curve(x, lc, dinj, ng=ng)
        tot = np.trapezoid(N, g)
        worst = max(worst, abs(tot/(min(x, 1. + dinj) - 1.) - 1.))
  return worst


def check_slopes():
  '''the asymptotic indices of the text, measured on the afterglow at x = 100.'''
  out = []
  kapF = kappa_of_C(-2.)
  g0 = S_EXP/kapF*100.**(-D_AD)          # oldest (frozen) edge of the fast afterglow
  for lab, lc, x, g, exp in (
      ('FC fresh -2', -2., 3., 500., -2.),
      ('FC relics 1/s-1', -2., 100., 3.*g0, 1./S_EXP - 1.),
      ('SC smeared 1/d-1', 2., 100., GM0*100.**(-D_AD/2.), 1./D_AD - 1.),
      ('SC injected -p', 2., 100., 3.*GM0, -P_SYN),
      ('SC cooled -(p+1)', 2., 1.5, 3e-2*GMA_M0, -(P_SYN + 1.))):
    gg = g*np.array([.97, 1.03])
    N = shell_integrated(gg, x, lc, np.inf)
    out.append((f'{lab} (x={x:g})', g,
                float((np.diff(np.log(N))/np.diff(np.log(gg)))[0]), exp))
  return out


# --- the figure -----------------------------------------------------------------------
def plot_shell_integrated(outdir=OUTDIR, fname='cooling_shell_integrated.png', show=False,
    p=P_SYN):
  '''
  Per regime (fast on top, slow below): the distributions, and their local slope
  underneath, with the indices the text derives as guides. Columns are the two
  injection cases. A blank spacer row separates the two regimes.
  '''
  lx = np.log10(X_SAMPLES)
  norm = plt.Normalize(vmin=lx.min(), vmax=lx.max())
  cmap = plt.cm.Blues
  col = lambda x: cmap(.35 + .65*norm(np.log10(x)))
  # expected indices per regime row: the steady-state ones and the two that need a
  # spread of injection times (relics 1/s-1 in fast, smearing 1/d-1 in slow cooling)
  levels = {LOGC_ROWS[0]: ((1./S_EXP - 1., '$-2/3$'), (-2., '$-2$'),
                           (-(p + 1.), '$-(p+1)$')),
            LOGC_ROWS[1]: ((1./D_AD - 1., '$+1/2$'), (-p, '$-p$'),
                           (-(p + 1.), '$-(p+1)$'))}
  # continuous injection turns slow after t'_tr, so its fast-cooling panel ends on -p
  extra = {(LOGC_ROWS[0], 1): ((-p, '$-p$'),)}
  SL_LIM = (-(p + 2.4), 1.6)
  fig, axs = plt.subplots(5, 2, figsize=FIGSIZE, sharex=True,
      gridspec_kw=dict(height_ratios=[1.9, 1., .22, 1.9, 1.], hspace=.08, wspace=.06))
  for ax in axs[2]:
    ax.set_visible(False)
  rows = ((axs[0], axs[1]), (axs[3], axs[4]))
  for i, lc in enumerate(LOGC_ROWS):
    axN_row, axS_row = rows[i]
    top = 0.
    for j, (name, lab, dinj) in enumerate(CASES):
      axN, axS = axN_row[j], axS_row[j]
      for x in X_SAMPLES:
        g, N = shell_curve(x, lc, dinj)
        axN.loglog(g, N, color=col(x), lw=1.2, zorder=3)
        sl = log_slope(g, N)
        # the support edges are hard, so the slope runs off to +-inf there: blank what
        # leaves the panel instead of drawing the jump as a vertical line
        sl[~np.isfinite(sl) | (sl > SL_LIM[1]) | (sl < SL_LIM[0])] = np.nan
        axS.semilogx(g, sl, color=col(x), lw=1.1, zorder=3)
        top = max(top, N.max())
      lev_ij = tuple(sorted(levels[lc] + extra.get((lc, j), ()), reverse=True))
      for lev, _ in lev_ij:
        axS.axhline(lev, color=MUTED, ls='--', lw=.7, zorder=1)
      axS.set_ylim(*SL_LIM)
      for ax in (axN, axS):
        ax.axvspan(1e-30, 1., color='crimson', alpha=.07, lw=0, zorder=0)
        ax.axvline(1., color='crimson', ls=':', lw=.9, zorder=1)
        for v in (GM0, GMA_M0):
          ax.axvline(v, color=INK, ls=':', lw=.8, zorder=1)
        ax.grid(alpha=.25, lw=.4)
        ax.tick_params(which='both', labelsize=FS_TICK)
      axN.annotate(f'$\\bar{{\\gamma}}_{{\\rm c,0}}/\\gamma_{{\\rm m,0}}=10^{{{lc:+.0f}}}$',
                   (.03, .05), xycoords='axes fraction', fontsize=FS_ANN, color=INK,
                   ha='left', va='bottom',
                   bbox=dict(fc='w', ec='none', alpha=.85, pad=1.))
      if i == 0:
        axN.annotate(lab, (.5, 1.02), xycoords='axes fraction',
                     fontsize=FS_LAB, color=INK, ha='center', va='bottom')
      if j == 0:
        axN.set_ylabel('$(\\dot N t\'_0)^{-1}\\,{\\rm d}\\mathcal{N}_{\\rm e}'
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
        axR.grid(False)
    for ax in axN_row:
      ax.set_ylim(top*1e-12, top*8.)
  for ax in axs[4]:
    ax.set_xlabel(GMA_LABEL, fontsize=FS_LAB)
  axs[0, 0].set_xlim(.05, 3.*GMA_M0)
  for v, lab in ((GM0, '$\\gamma_{\\mathrm{m},\\!0}$'),
                 (GMA_M0, '$\\gamma_{\\mathrm{M},\\!0}$')):
    axs[0, 1].annotate(lab, (v, .985), xycoords=('data', 'axes fraction'), color=INK,
                       fontsize=FS_ANN, ha='center', va='top',
                       bbox=dict(fc='w', ec='none', alpha=.85, pad=1.))
  fig.subplots_adjust(right=.84)
  p0, p1 = axs[0, 1].get_position(), axs[4, 1].get_position()
  cax = fig.add_axes([.935, p1.y0, .018, p0.y1 - p1.y0])
  sm = plt.cm.ScalarMappable(cmap=plt.cm.colors.ListedColormap(
      cmap(np.linspace(.35, 1., 256))), norm=norm)
  cb = fig.colorbar(sm, cax=cax)
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
  print(f'kappa: C=1e-2 -> {kappa_of_C(-2.):.4g}, C=1e2 -> {kappa_of_C(2.):.4g}; '
        f'<A B^2>/B0^2 over [1,2] = {(1. - 2.**(-S_EXP))/S_EXP:.4f}')
  print(f'sum rule: max |int Ncal dgma/(y_f-1) - 1| = {check_sum_rule():.2e}')
  for lab, g, meas, exp in check_slopes():
    print(f'  {lab:26s} at gma={g:9.3e}: {meas:+.4f} (expected {exp:+.4f})')
  plot_shell_integrated(show=show)


if __name__ == '__main__':
  main()
