# -*- coding: utf-8 -*-
# @Author: acharlet

'''
How a power-law electron distribution is reshaped by synchrotron AND adiabatic cooling.

APPENDIX ILLUSTRATION, the fourth corner of the family that shares figures/
cooling_distributions:

                          synchrotron only          + adiabatic
    instantaneous N     cooling_shape_figure      THIS MODULE
    time-integrated     cooling_integrated_figure cooling_integrated_adiabatic

Same hydrodynamics and the same footing as cooling_integrated_adiabatic: simple coasting
(rho' ~ R^-2) with the comoving cooling time t'_c held CONSTANT, so the only thing added
to cooling_shape_figure is the adiabatic drag. The physics functions are imported from
that module rather than restated, so the two adiabatic figures cannot drift apart.

THE SOLUTION. The cooling equation dgma/dtt = (dlnA/dtt) gma - gma^2 is linear in
u = 1/gma and integrates with A = (rho/rho_0)^(1/3) as the integrating factor:

    gma(sigma) = A gma_0/(1 + gma_0 S),    S = int_0^tt A dtt',   sigma = t'/t'_dyn,

and number conservation turns the injected power law into

    N(gma,sigma) = K0 A gma^-p (A - S gma)^(p-2)   on [gma_m(sigma), gma_M(sigma)].

THE POINT OF THE FIGURE: ADIABATIC COOLING ADDS NO NEW SHAPE. Substituting x = gma/A
collapses the expression above to

    N_adiab(gma, sigma) = (1/A) * N_syn(gma/A, S),

an exact SIMILARITY transform of the synchrotron-only solution -- slide the whole
distribution down the gamma axis by A, up in amplitude by 1/A (so int N dgma stays 1),
and relabel the clock tt -> S. Nothing bends, nothing breaks, no segment is created or
destroyed. check_similarity verifies it to ~1e-14.

Three consequences worth reading off the panels:

  - The WIDTH evolution is untouched. A cancels out of the ratio,
    gma_M/gma_m = (gma_M0/gma_m0)(1 + gma_m0 S)/(1 + gma_M0 S), which is exactly
    cooling_shape_figure's collapse law with tt -> S. So the population still ends up
    MONO-ENERGETIC, on the same schedule measured in S -- just at a lower gamma.
    check_width_law.
  - The two knees still sit where the edges burn, but in S, not in tt: the top edge
    starts burning at S = 1/gma_M0 and the power law is gone by S = 1/gma_m0. Since
    S grows more slowly than tt (S/tt_dyn = 27.0 after 1e3 t_dyn, against tt/tt_dyn =
    1e3), expansion DELAYS the collapse when it is clocked in real time.
  - And it explains the companion figure: if the instantaneous shape is only ever slid,
    the time-integrated one cannot acquire a new index from adiabatic cooling either.
    That is why cooling_integrated_adiabatic finds the tail still at ~ -2.

WHY THIS FIGURE NEEDS A PARAMETER THE SYNCHROTRON ONE DID NOT. cooling_shape_figure is
a one-parameter family: everything is a function of tt alone, because tt = int dt'/t_c1
absorbs the field history. Here A is a function of RADIUS, so the tt <-> sigma map has to
be fixed, and that map carries tt_dyn = 1/gma_c. The figure therefore has to choose a
cooling regime; LOGC = 0 (the marginal case gma_c = gma_m) is the default and is stated
inside panel (a). Other regimes only stretch the sigma axis, they do not change the
shapes -- which is the similarity result again.

VALIDITY. As everywhere in this family, gma_synCooled's ultra-relativistic trajectory is
not physical below gma = 1 (marked in crimson): the real loss rate goes as gma^2 - 1 and
electrons stall near 1 instead of continuing down. With the adiabatic drag the population
gets there SOONER -- at log10 C = 0 the whole distribution is under the floor by
sigma ~ 1e2 -- so the last sampled times are drawn to show where it is heading, not
where it is.

Run:  python cooling_shape_adiabatic.py
'''

import os
import numpy as np
import matplotlib.pyplot as plt

from cooling_distribution import gamma_synCooled, norm_plaw_distrib, distrib_plaw_cooled
from cooling_shape_figure import (P_SYN, GM0, GMA_M0, NG, OUTDIR, INK, MUTED, FIGSIZE,
    FS_LAB, FS_TICK, FS_ANN, FS_LEG, cooled_distrib)
from cooling_integrated_figure import tt_dyn_of_C
from cooling_integrated_adiabatic import (A_RHO, Q_B, _exps, A_of_sigma, S_of_sigma,
    tt_of_sigma, gamma_cooled)

# --- defaults -------------------------------------------------------------------------
LOGC = 0.                   # gma_c/gma_m for the tt <-> sigma map; marginal cooling
# sampled sigma = t'/t'_dyn as log10. At LOGC = 0 (tt_dyn = 1e-3) this is exactly
# cooling_shape_figure's tt window, 1e-8 .. 1, expressed in dynamical times.
LOGSIG_SAMPLES = (-5., -4., -3., -2., -1., 0., 1., 2., 3.)
FNAME = 'cooling_shape_adiabatic.png'


# --- the distribution -----------------------------------------------------------------
def cooled_distrib_adiab(sigma, ttd, p=P_SYN, gm0=GM0, gM0=GMA_M0, a_rho=A_RHO, q=Q_B,
    Ng=NG):
  '''
  (gma, N(gma,sigma)) sampled over the SUPPORT [gma_m(sigma), gma_M(sigma)] only --
  as in cooling_shape_figure, the expression stays positive above gma_M but those
  electrons do not exist, nothing having been injected above gma_M0.
  '''
  A, S = float(A_of_sigma(sigma, a_rho)), float(S_of_sigma(sigma, ttd, a_rho, q))
  gm, gM = (gamma_cooled(sigma, g, ttd, a_rho, q) for g in (gm0, gM0))
  gma = np.geomspace(gm, gM, Ng)
  return gma, norm_plaw_distrib(gm0, gM0, p)*A*gma**-p*np.abs(A - S*gma)**(p-2.)


def width(sigma, ttd, gm0=GM0, gM0=GMA_M0, a_rho=A_RHO, q=Q_B):
  'gma_M/gma_m at sigma -- the A cancels, so this is the synchrotron law clocked in S.'
  return (gamma_cooled(sigma, gM0, ttd, a_rho, q)
          /gamma_cooled(sigma, gm0, ttd, a_rho, q))


# --- validation -------------------------------------------------------------------------
def check_number_conservation(sig_arr=LOGSIG_SAMPLES, logC=LOGC, p=P_SYN, gm0=GM0,
    gM0=GMA_M0, a_rho=A_RHO, q=Q_B, Ng=40000):
  '''
  int N dgma must stay 1 at every sigma: cooling moves electrons, adiabatic or not.
  Returns the largest relative deviation.
  '''
  ttd = tt_dyn_of_C(logC, gm0)
  dev = 0.
  for ls in sig_arr:
    gma, N = cooled_distrib_adiab(10.**ls, ttd, p, gm0, gM0, a_rho, q, Ng)
    dev = max(dev, abs(np.trapezoid(N, gma) - 1.))
  return dev


def check_similarity(sig_arr=LOGSIG_SAMPLES, logC=LOGC, p=P_SYN, gm0=GM0, gM0=GMA_M0,
    a_rho=A_RHO, q=Q_B, n=40):
  '''
  THE claim of this module: N_adiab(gma,sigma) = (1/A) N_syn(gma/A, S) exactly. Evaluated
  against the pipeline's own distrib_plaw_cooled on the rescaled axis. Returns the
  largest relative deviation over the sampled times and the support.
  '''
  ttd = tt_dyn_of_C(logC, gm0)
  K0 = norm_plaw_distrib(gm0, gM0, p)
  dev = 0.
  for ls in sig_arr:
    sigma = 10.**ls
    A, S = float(A_of_sigma(sigma, a_rho)), float(S_of_sigma(sigma, ttd, a_rho, q))
    gma, N = cooled_distrib_adiab(sigma, ttd, p, gm0, gM0, a_rho, q, n)
    ref = K0*distrib_plaw_cooled(gma/A, p, S)/A          # the synchrotron shape, slid
    ok = ref > 0.
    dev = max(dev, float(np.max(np.abs(N[ok]/ref[ok] - 1.))))
  return dev


def check_width_law(sig_arr=LOGSIG_SAMPLES, logC=LOGC, gm0=GM0, gM0=GMA_M0, a_rho=A_RHO,
    q=Q_B):
  '''
  The width must obey cooling_shape_figure's collapse law with tt -> S, the A having
  cancelled. Returns the largest relative deviation.
  '''
  ttd = tt_dyn_of_C(logC, gm0)
  dev = 0.
  for ls in sig_arr:
    sigma = 10.**ls
    S = float(S_of_sigma(sigma, ttd, a_rho, q))
    pred = (gM0/gm0)*(1. + gm0*S)/(1. + gM0*S)
    dev = max(dev, abs(width(sigma, ttd, gm0, gM0, a_rho, q)/pred - 1.))
  return dev


def check_reduces_to_shape(sig_arr=LOGSIG_SAMPLES, logC=LOGC, p=P_SYN, gm0=GM0,
    gM0=GMA_M0, n=40):
  '''
  At a_rho -> 0 (no expansion) A = 1 and S = tt, so this must reproduce
  cooling_shape_figure.cooled_distrib. The tie to the already-validated module.
  '''
  ttd = tt_dyn_of_C(logC, gm0)
  dev = 0.
  for ls in sig_arr:
    sigma = 10.**ls
    g_a, N_a = cooled_distrib_adiab(sigma, ttd, p, gm0, gM0, a_rho=1e-12, q=0., Ng=n)
    _, N_s = cooled_distrib(ttd*sigma, p, gm0, gM0, Ng=n)
    dev = max(dev, float(np.max(np.abs(N_a/N_s - 1.))))
  return dev


# --- the figure ---------------------------------------------------------------------------
def plot_cooling_shape_adiab(logC=LOGC, p=P_SYN, gm0=GM0, gM0=GMA_M0, a_rho=A_RHO, q=Q_B,
    logsig=LOGSIG_SAMPLES, outdir=OUTDIR, fname=FNAME, show=False):
  '''
  Two-panel view, laid out exactly as cooling_shape_figure so the pair can be read across:
  the edge tracks on top, the distributions they sample below.
  '''
  ttd = tt_dyn_of_C(logC, gm0)
  sig_arr = 10.**np.asarray(logsig, dtype=float)
  colors = plt.cm.viridis(np.linspace(0., .85, len(sig_arr)))
  K0 = norm_plaw_distrib(gm0, gM0, p)

  fig, (axT, axD) = plt.subplots(2, 1, figsize=FIGSIZE,
      gridspec_kw=dict(height_ratios=[1., 1.5], hspace=.32))

  # (a) edge tracks vs sigma -------------------------------------------------------------
  sg = np.geomspace(1e-3*sig_arr[0], 1.5*sig_arr[-1], 900)
  cut = lambda y: np.where(y >= 1., y, np.nan)   # never draw below the validity floor
  # the two ghosts separate the causes: what synchrotron alone would do at the same time,
  # and what expansion alone would do. The real track is below both.
  axT.loglog(sg, cut(gamma_synCooled(tt_of_sigma(sg, ttd, q), gM0)), color=MUTED, lw=.8,
             ls='-', label='syn. only')
  axT.loglog(sg, cut(A_of_sigma(sg, a_rho)*gM0), color=MUTED, lw=.8, ls='-.',
             label='adiab. only')
  axT.loglog(sg, cut(gamma_cooled(sg, gM0, ttd, a_rho, q)), color='k', lw=1.4,
             label='$\\gamma_\\mathrm{M}$')
  axT.loglog(sg, cut(gamma_cooled(sg, gm0, ttd, a_rho, q)), color='k', lw=1.1, ls='--',
             label='$\\gamma_\\mathrm{m}$')
  axT.axhline(1., color='crimson', ls=':', lw=.9, zorder=1)
  # the knees are where the edges burn, and they are set by S, not by sigma
  for gedge, lab in ((gM0, '$\\sigma_M$'), (gm0, '$\\sigma_m$')):
    tgt = 1./gedge                                    # S at which that edge has burnt
    k = np.interp(tgt, S_of_sigma(sg, ttd, a_rho, q), sg)
    axT.axvline(k, color=INK, ls=':', lw=.8, zorder=1)
    axT.annotate(lab, (k, .015), xycoords=('data', 'axes fraction'), color=INK,
                 fontsize=FS_ANN, ha='center', va='bottom',
                 bbox=dict(fc='w', ec='none', alpha=.85, pad=1.))
  axT.set_xlim(sg[0], sg[-1])
  axT.set_ylim(.15, 3.*gM0)
  axT.set_xlabel("$\\sigma = t'/t'_{\\rm dyn}$", fontsize=FS_LAB)
  axT.set_ylabel('$\\gamma$', fontsize=FS_LAB)
  # the regime this figure had to pick, stated inside the panel (article convention)
  axT.annotate(f'$\\log_{{10}}\\mathcal{{C}}={logC:.0f}$', (.03, .22),
               xycoords='axes fraction', color=INK, fontsize=FS_ANN, ha='left',
               va='bottom')      # clear of gma=1 (~0.09) and of the gma_m track (~0.41)
  axT.legend(fontsize=FS_LEG, loc='upper right', framealpha=.9, handlelength=1.4,
             labelspacing=.25, handletextpad=.5, borderpad=.4)
  axT.grid(alpha=.25, lw=.4)

  # (b) the distributions ------------------------------------------------------------------
  gma0, N0 = cooled_distrib_adiab(0., ttd, p, gm0, gM0, a_rho, q)
  axD.loglog(gma0, N0, color='k', lw=1.4, zorder=2)
  edges = []
  for ls, sigma, c in zip(logsig, sig_arr, colors):
    gma, N = cooled_distrib_adiab(sigma, ttd, p, gm0, gM0, a_rho, q)
    axD.loglog(gma, N, color=c, lw=1.2, solid_capstyle='round', label=f'{ls:.0f}',
               zorder=3)
    edges.append((gma[-1], N[-1]))
  edges = np.array(edges)
  axD.plot(edges[:, 0], edges[:, 1], color=MUTED, lw=.7, zorder=4)
  axD.scatter(edges[:, 0], edges[:, 1], s=9, facecolors=colors, edgecolors='w',
              linewidths=.5, zorder=6)

  gg = np.geomspace(gm0, gM0, 3)
  axD.loglog(gg, 12.*K0*gg**-p, color=MUTED, ls=':', lw=.9)
  axD.annotate('$\\propto\\gamma^{-p}$', (gg[1], 12.*K0*gg[1]**-p),
               textcoords='offset points', xytext=(3, 3), color=MUTED, fontsize=FS_ANN)
  for v, lab in ((gm0, '$\\gamma_{\\mathrm{m},\\!0}$'), (gM0, '$\\gamma_{\\mathrm{M},\\!0}$')):
    axD.axvline(v, color=INK, ls=':', lw=.8, zorder=1)
    axD.annotate(lab, (v, .985), xycoords=('data', 'axes fraction'), color=INK,
                 fontsize=FS_ANN, ha='center', va='top',
                 bbox=dict(fc='w', ec='none', alpha=.85, pad=1.))
  axD.axvline(1., color='crimson', ls=':', lw=.9, zorder=1)
  axD.set_xlim(.5*float(gamma_cooled(sig_arr[-1], gm0, ttd, a_rho, q)), 2.*gM0)
  axD.set_ylim(1e-16, 1e4)
  axD.set_xlabel('$\\gamma$', fontsize=FS_LAB)
  axD.set_ylabel('$N(\\gamma,\\sigma)/N_{\\rm e}$', fontsize=FS_LAB)
  leg = axD.legend(fontsize=FS_LEG, ncol=2, loc='lower left', framealpha=.9,
                   title='$\\log_{10}\\sigma$', handlelength=1.1, labelspacing=.25,
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
  alpha, e = _exps(A_RHO, Q_B)
  ttd = tt_dyn_of_C(LOGC, GM0)
  print(f"hydro: coasting rho' ~ R^{A_RHO:g}, B' ~ R^-{Q_B:g} (constant t'_c);  "
        f'alpha={alpha:+.4f}  e={e:+.4f}')
  print(f'clock: log10 C = {LOGC:.0f} -> tt_dyn = {ttd:.3e}, so sigma = tt/tt_dyn')
  print(f'similarity      : max |A*N_adiab(A x) / N_syn(x,S) - 1| = '
        f'{check_similarity():.2e}')
  print(f'reduces to shape: max rel dev at a_rho->0 = {check_reduces_to_shape():.2e}')
  print(f'number conserved: max |int N dgma - 1| = {check_number_conservation():.2e}')
  print(f'width law (S)   : max rel dev = {check_width_law():.2e}')
  print('per sampled time:')
  for ls in LOGSIG_SAMPLES:
    sigma = 10.**ls
    A = float(A_of_sigma(sigma, A_RHO))
    S = float(S_of_sigma(sigma, ttd, A_RHO, Q_B))
    gm = float(gamma_cooled(sigma, GM0, ttd, A_RHO, Q_B))
    gM = float(gamma_cooled(sigma, GMA_M0, ttd, A_RHO, Q_B))
    print(f'  log10 sigma = {ls:+.0f}: A={A:9.3e}  S={S:9.3e}  '
          f'gma_m={gm:9.3e}  gma_M={gM:9.3e}  width={gM/gm:8.4f}'
          f'{"   (under the gma=1 floor)" if gM < 1. else ""}')
  plot_cooling_shape_adiab(show=show)


if __name__ == '__main__':
  main()
