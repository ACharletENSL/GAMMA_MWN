# -*- coding: utf-8 -*-
# @Author: acharlet

'''
Paper I, Sect. 4.1: the comoving spectral luminosity L'_nu' of the HOMOGENEOUS coasting
shell, i.e. the spectrum of Fig. 4 (cooling_shell_integrated), with the step emissivity of
Eq. eqn:p_nu_step and with the full kernel of Appendix A.1.

THE STEP KERNEL (Eq. eqn:p_nu_step): p'_nu' = p'_syn (nu'/nu'_syn)^(1/3) below nu'_syn
= (3/2) x_syn gma^2 nu'_B, 0 above. It keeps the low-x asymptote R_LOW_COEF x^(1/3) of the
kernel R(x), x = 2 nu'/(3 gma^2 nu'_B), and radiates the synchrotron power, (3/4) p'_syn
nu'_syn = int P'_nu' dnu', so x_syn = [4 I_S/(3 R_LOW_COEF)]^(3/4) = 0.8399, I_S = int R dx
= 16 pi/(27 sqrt 3). THE FULL KERNEL is func_R normalised to the same asymptote,
p'_nu' = p'_syn R(x)/(R_LOW_COEF x_syn^(1/3)), x = x_syn nu'/nu'_syn.

THE SHELL (Eq. eqn:Lnu_shell). Every element sees B'(t') = B'_0 tau^-lam_B, so with
u = nu'/nu'_syn,0, nu'_syn,0 the synchrotron frequency of gma = 1 at t'_0,

    L'_nu'/(p'_syn,0 Ndot_0 t'_0) = tau^-lam_B int (gma Ncal) k(u tau^lam_B/gma^2) dln gma,

Ncal = shell_integrated (in Ndot_0 t'_0 per unit gma: coasting, x = R/R_0 = tau) and k the
kernel in units of p'_syn. The step kernel is integrated EXACTLY on the piecewise power-law
interpolant of the sampled distribution (pl_spectrum, which is Eq. eqn:Lnu_bpl itself), the
full one by trapezoid in ln gma. bpl_spectrum is Eq. eqn:Lnu_bpl on the segments of Table
tab:shell_bpl (shell_bpl_check.table), its top segment extended to gma_M,i; the slow-cooling
adiabatic segment is a sum of two power laws, each its own term.

RESULTS (2026-10-20; main() ~2 min, the figures alone ~10 s; p = 2.5 reference case and
the shell_bpl_check grid p = 2.2-3, lam_inj = -1/2-1, lam_m = 0-0.3, tau = 3-1000):
  * Kernel: x_syn = 0.8399, nu'_syn = 1.2599 gma^2 nu'_B, p'_syn = 0.3528 sigma_T m_e c^2 B'/e.
    r(kappa) closed form = 0.870 (1/6), 0.831 (1/12), 0.803 (0), 0.798 (-1/2), 0.822 (-0.6),
    0.873 (-3/4), 1 (-1), 1.068 (-1.1), 1.192 (-5/4), 1.469 (-3/2); func_R quadrature agrees
    to <= 0.4% (its int R dx is 1.07514, 0.03% above 16 pi/27 sqrt3). Measured full/step
    mid-segment on B1 tau=16: 1.190 in -p/2 (r = 1.192); 0.83 and 0.72 in the frozen and -1/2
    segments, below r because those segments are too short for the asymptote.
    Eq. eqn:Lnu_bpl = the table integrated as a fine interpolant to 7e-6.
  * BPL/exact, step kernel (grid, FC at C = 1e-5; reference case in brackets):
    nu'_fr: 1.27-1.71 two decades below [1.29 at tau=6], 1.69-2.11 at it [1.75 at tau=3],
    1.04-1.11 two decades above [1.05]; max within a decade 1.6-2.3 (lam_inj < 1: 1.6-2.0)
    [1.65-1.89]. nu'_b: 1.18-1.50 below [1.33-1.34], 1.75-2.18 at it [1.99-2.00], 1.08-1.16
    above [1.11], max within a decade 1.8-2.3 [2.09-2.10]. nu'_m,i: 1.00-1.06 on both sides
    (SC), 1.00-1.04 at it; FC two decades below it 1.0-1.7, but there it is nu'_fr's tail.
    The harmonic joining, carried through Eq. eqn:Lnu_shell, cuts the max within a decade
    to 1-51% at nu'_fr (1-23% for lam_inj < 1; reference 5-18%) and 2-30% at nu'_b
    (reference 17-18%). Checked by brute-force trapezoid (1.332/1.990/1.110 at SC tau=10).
  * Frozen segment, B1 (since 2026-10-21: gma_m,0 = 1e4, C = 1e-3, tau = 10, 13, 16 <
    t'_tr = 19.3, nu'_c/nu'_inf(t';t'_0) = 1.0e6, 4.8e6, 1.7e7; lowest gma_e 1.9, 1.6, 1.4).
    No clean nu'^(1/6) segment yet: the local index rises to a maximum 0.143/0.152/0.158
    (full; step 0.136-0.150, BPL 0.149-0.155, frozen segment alone 0.144-0.153) and is within
    0.05 of 1/6 from just above nu'_inf(t';t'_0) up to 4.67/4.63/4.55 decades below nu'_c
    (full), 4.45/4.37/4.25 (step), 3.27/3.30/3.37 (Eq. eqn:Lnu_bpl) and 3.82 (frozen
    segment alone, the text's formula), i.e. over 0.85, 1.6, 2.2 decades (full). The -1/2
    electrons above gma_c add a nu'^(1/3) term (the BPL converges sooner) and the exact
    distribution is below the table at gma_c (it converges later). B2 (lam_inj = 2): 1/3
    down to nu'_inf. (Before: C = 1e-4, tau = 10-50, gma_m,0 = 1e3: 4.4-4.8 decades
    (full), max 0.162, but the frozen range at gma_e = 0.006-0.019.)
  * Kernel smoothing, |full - step local index| > 0.05: 0.65-1.05 decade below nu'_low
    (sharp edge, step width 0.05), 0.9-1.0 decade around nu'_m,i in FC ([0.42, 3.9] nu'_m,i),
    0.68 in SC ([0.70, 3.4]), 0.75 above nu'_b ([2, 11.5] nu'_b, the r(kappa) change), and
    nothing at nu'_fr, already smooth with the step kernel. Transition widths (index > 0.05
    from both asymptotes), step -> full: nu'_m,i FC 0.67 -> 1.28 decades, SC (tau=10) 1.22 ->
    2.55; nu'_b 3.5 -> 3.6; nu'_fr: kappa_fr never reached within halfway to nu'_low, the
    -1/2 side reached at 8 (step) / 8 (full) nu'_fr at tau = 3.
'''

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma as Gfun

import cooling_shell_integrated as csi
import shell_bpl_check as sbc
from cooling_shell_integrated import (COASTING, ShellLaws, X_SAMPLES, CASES, LOGC_ROWS,
    P_SYN, GM0, GMA_M0, INK, MUTED, TEXT_BOX, FIGSIZE, FS_LAB, FS_TICK, FS_ANN, FS_LEG)
from cooling_shape_figure import OUTDIR, named_ticks_top, marks_bottom
from phys_functions import func_R, R_LOW_COEF

I_S = 16.*np.pi/(27.*np.sqrt(3.))              # int R dx
X_SYN = (4.*I_S/(3.*R_LOW_COEF))**.75           # 0.8399
NU_SYN_COEF = 1.5*X_SYN                         # nu'_syn = 1.2599 gma^2 nu'_B
P_SYN_COEF = 8./(27.*X_SYN)                     # p'_syn in sigma_T m_e c^2 B'/e, 0.3528
G_PER_DEC = 100                                 # distribution samples per decade of gma
U_PER_DEC = 30                                  # spectrum samples per decade of nu'
# frequencies scale as gma^2: the abscissae below were set at gma_m,0 = 1e3 and move with it
_U = (GM0/1e3)**2
U_LIM = (1e-6*_U, 1e18*_U)                      # Fig. A: below the lowest nu'_inf(t';t'_0)
                                                # (finite FC, tau=100: 7.6e-3) to past
                                                # nu'_syn(gma_M,0)
SLOPE_TOL = .05

# same reference curve per panel as Fig. 4 (row, column) -> tau
REF_X = csi.REF_X
FC_B = -3.                                      # Fig. B: t'_tr = 19.3 t'_0 (was -4, 52 t'_0,
X_B = (10., 13., 16.)                           # and tau = 10, 30, 50 before 2026-10-21)
LAWS_B2 = ShellLaws(lam_N=2., label='coasting, lam_inj = 2')


# --- kernels --------------------------------------------------------------------------
def kernel_full(y):
  '''p'_nu'/p'_syn of the full kernel at y = nu'/nu'_syn(gma), -> y^(1/3) at low y'''
  return func_R(X_SYN*y)/(R_LOW_COEF*X_SYN**(1./3.))


def r_closed(kappa):
  '''
  full/step normalisation of a segment f ~ gma^(2 kappa), kappa < 1/3:
  (1/3 - kappa) int x^mu R dx/[R_LOW_COEF x_syn^(1/3 - kappa)], mu = -kappa - 1, with
  int x^mu R dx = the Mellin transform of F (Rybicki & Lightman) x int_0^pi/2 sin^(3+mu)
  '''
  mu = -kappa - 1.
  M = (2.**(mu + 1.)/(mu + 2.)*Gfun(mu/2. + 7./3.)*Gfun(mu/2. + 2./3.)
       *.5*np.sqrt(np.pi)*Gfun(2. + mu/2.)/Gfun(2.5 + mu/2.))
  return (1./3. - kappa)*M/(R_LOW_COEF*X_SYN**(1./3. - kappa))


def r_numeric(kappa, n=20001):
  '''the same with func_R by quadrature (checks r_closed and func_R's tabulation)'''
  x = np.geomspace(1e-14, 80., n)
  return ((1./3. - kappa)*np.trapezoid(x**(-kappa)*func_R(x), np.log(x))
          /(R_LOW_COEF*X_SYN**(1./3. - kappa)))


# --- Eq. eqn:Lnu_bpl ------------------------------------------------------------------
def _phi(z):
  '''expm1(z)/z, 1 at z = 0'''
  z = np.asarray(z, float)
  out = np.ones_like(z)
  m = np.abs(z) > 1e-10
  out[m] = np.expm1(z[m])/z[m]
  return out


def pl_spectrum(gnu, pieces):
  '''
  Eq. eqn:Lnu_bpl: int_{gnu} (gnu/gma)^(2/3) f dln gma for f a sum of power laws on
  [g_lo, g_hi], pieces = [(g_lo, g_hi, f_lo, kappa)] with f = f_lo (gma/g_lo)^(2 kappa).
  Written as f(a)(gnu/a)^(2/3) ln(b/a) phi(2(kappa - 1/3) ln(b/a)), a = max(g_lo, gnu),
  b = g_hi, which is the equation's ratio and stays finite at kappa = 1/3.
  '''
  gnu = np.asarray(gnu, float)[:, None]
  g_lo, g_hi, f_lo, kap = (np.asarray(v, float)[None, :] for v in zip(*pieces))
  a = np.minimum(np.maximum(g_lo, gnu), g_hi)
  ok = g_hi > a
  L = np.log(np.where(ok, g_hi/a, 1.))
  fa = f_lo*(a/g_lo)**(2.*kap)
  val = fa*(gnu/a)**(2./3.)*L*_phi(2.*(kap - 1./3.)*L)
  return np.sum(np.where(ok, val, 0.), axis=1)


def grid_pieces(g, f):
  '''the piecewise power-law interpolant of a sampled f >= 0 (cells with a zero dropped)'''
  ok = (f[:-1] > 0.) & (f[1:] > 0.)
  kap = np.zeros(g.size - 1)
  kap[ok] = .5*np.log(f[1:][ok]/f[:-1][ok])/np.log(g[1:][ok]/g[:-1][ok])
  return [(g[k], g[k + 1], f[k], kap[k]) for k in np.flatnonzero(ok)]


def table_pieces(c, regime, tau_end=None, gM=GMA_M0):
  '''
  The segments of Table tab:shell_bpl (shell_bpl_check.table) as pl_spectrum pieces, in
  dN/(N dln gma), and the breaks. Continuous injection: the top segment runs to gma_M,i.
  '''
  p, a, eta, zeta, P = c.p, c.a, c.eta, c.zeta, c.P
  _, _, br, _ = sbc.table(np.array([1.]), c, regime, tau_end)
  pw = lambda A, g0, k: lambda g: A*(g/g0)**(2.*k)       # A at g0, f ~ gma^(2k)
  if tau_end is None:
    t, F = c.tau, c.F(c.tau)
    sgc, gc, gm = c.s*c.gci(t), c.gci(t), c.gmi(t)
    gMi = gM*t**0.                                        # lam_M = 0 here
    if regime == 'FC':
      segs = [(br['low'], br['fr'], [(pw(a*F/eta, sgc, a/(2.*eta)), a/(2.*eta))]),
              (br['fr'], gm, [(pw(a*F*gc/gm, gm, -.5), -.5)]),
              (gm, gMi, [(pw(a*F*gc/gm, gm, -p/2.), -p/2.)])]
    else:
      C = (p - 1.)*a*F/P
      segs = [(br['low'], gm, [(pw(C, gm, a/(2.*zeta)), a/(2.*zeta)),
                               (pw(-C*t**(-P), gm, (1. - p)/2.), (1. - p)/2.)]),
              (gm, br['b'], [(pw(C*(1. - t**(-P)), gm, (1. - p)/2.), (1. - p)/2.)]),
              (br['b'], gMi, [(pw(a*F*gc/gm, gm, -p/2.), -p/2.)])]
  else:
    F = c.F(tau_end)
    if regime == 'FC':
      hi = br['hi']
      segs = [(br['low'], hi, [(pw(a*F/eta, hi, a/(2.*eta)), a/(2.*eta))])]
    else:
      gme = c.gm_of(tau_end)
      segs = [(br['low'], br['hi'],
               [(pw((p - 1.)*a*F/P*(1. - tau_end**(-P)), gme, (1. - p)/2.), (1. - p)/2.)])]
  pieces = [(lo, hi, fn(lo), k) for lo, hi, terms in segs if hi > lo for fn, k in terms]
  return pieces, br


# --- the spectra ----------------------------------------------------------------------
def tau_of(x, laws=COASTING):
  '''t'/t'_0 of the shell at R = x R_0: coasting only (g = m = 0), where the two coincide'''
  assert laws.g == 0.
  return x


def distribution(x, logC, dinj, laws=COASTING, p=P_SYN, gM0=GMA_M0, extra=()):
  '''(gma, gma Ncal) on the support, G_PER_DEC per decade, with the points `extra` added'''
  lo, hi = csi.support(x, logC, dinj, laws, gM0=gM0)
  lo, hi = lo*(1. + 1e-9), hi*(1. - 1e-9)
  g = np.geomspace(lo, hi, int(np.log10(hi/lo)*G_PER_DEC) + 2)
  ex = np.asarray([v for v in extra if lo < v < hi])
  g = np.unique(np.concatenate([g, ex])) if ex.size else g
  return g, g*csi.shell_integrated(g, x, logC, dinj, laws, p, gM0=gM0)


def spectra(u, x, g, fN, laws=COASTING, full=True):
  '''
  L'_nu'/(p'_syn,0 Ndot_0 t'_0) at u = nu'/nu'_syn,0 of the distribution fN = gma Ncal:
  (step, full). tau^-lam_B is kept, as in the figure.
  '''
  t = float(tau_of(x, laws))
  sc = t**(-laws.lam_B)
  gnu = np.sqrt(u*t**laws.lam_B)
  Ls = sc*pl_spectrum(gnu, grid_pieces(g, fN))
  if not full:
    return Ls, None
  lg = np.log(g)
  Lf = np.empty_like(u)
  for k in range(0, u.size, 64):
    y = (gnu[k:k + 64, None]/g[None, :])**2
    Lf[k:k + 64] = np.trapezoid(fN[None, :]*kernel_full(y), lg, axis=1)
  return Ls, sc*Lf


def bpl(u, x, logC, dinj, regime, laws=COASTING, p=P_SYN, gM0=GMA_M0):
  '''Eq. eqn:Lnu_bpl on Table tab:shell_bpl, same units as spectra; and the breaks in u'''
  c = sbc.comoving(laws, 0., x, logC, p, gm0=GM0)
  tau_end = None if np.isinf(dinj) else 1. + dinj
  pieces, br = table_pieces(c, regime, tau_end, gM0)
  t = float(tau_of(x, laws))
  N = csi.injected_number(x, dinj, laws)
  L = t**(-laws.lam_B)*N*pl_spectrum(np.sqrt(u*t**laws.lam_B), pieces)
  return L, {k: v**2*t**(-laws.lam_B) for k, v in br.items()}, pieces, c


def frozen_alone(u, x, pieces, laws=COASTING, dinj=np.inf):
  '''
  the spectrum of the frozen fast-cooling segment of Table tab:shell_bpl alone (its first
  piece, [gma_low, gma_fr]), in the units of spectra: nu'^(1/3) below nu'_low, then the
  local index kappa_fr_local up to nu'_fr, and nothing above
  '''
  t = float(tau_of(x, laws))
  return (t**(-laws.lam_B)*csi.injected_number(x, dinj, laws)
          *pl_spectrum(np.sqrt(u*t**laws.lam_B), pieces[:1]))


def nu_of_gma(g, x, laws=COASTING):
  return g**2*float(tau_of(x, laws))**(-laws.lam_B)


def local_slope(u, L):
  with np.errstate(divide='ignore', invalid='ignore'):
    return np.gradient(np.log(L), np.log(u))


def kappa_fr_local(u, u_fr, kfr):
  '''the text's local index in the frozen segment alone, kfr - (1/3-kfr)/[(u_fr/u)^(1/3-kfr) - 1]'''
  d = 1./3. - kfr
  return kfr - d/((u_fr/u)**d - 1.)


def ugrid(lo, hi, n=U_PER_DEC):
  return np.geomspace(lo, hi, int(np.log10(hi/lo)*n) + 1)


# --- checks (the numbers of the report) -----------------------------------------------
def check_kernel():
  '''x_syn, p'_syn and r(kappa), closed form vs quadrature of func_R'''
  out = dict(x_syn=X_SYN, nu_coef=NU_SYN_COEF, p_coef=P_SYN_COEF,
             I_S_func_R=np.trapezoid(func_R(np.geomspace(1e-14, 80., 20001))
                                     *np.geomspace(1e-14, 80., 20001),
                                     np.log(np.geomspace(1e-14, 80., 20001))))
  ks = (1./6., 1./12., 0., -.5, -.6, -.75, -1., -1.1, -1.25, -1.5)
  out['r'] = [(k, r_closed(k), r_numeric(k)) for k in ks]
  return out


def _at(u, v, uq):
  '''v at uq by log interpolation (nan outside)'''
  lv = np.interp(np.log(uq), np.log(u), np.log(v), left=np.nan, right=np.nan)
  return np.exp(lv)


def bpl_metrics(laws, x, logC, regime, p, dinj=np.inf, gM0=1e14):
  '''
  Item 2: BPL (Eq. eqn:Lnu_bpl) over the step spectrum of the exact distribution at
  u_X 10^(+-2) for X = fr, b, mi (nan when that point is within a decade of another break
  or of the top segment's end: then it measures that one); the max within a decade of
  fr, b; the same for the harmonic joining (shell_bpl_check.joined) carried through
  Eq. eqn:Lnu_shell. gma_M,0 = 1e14 as in shell_bpl_check, so the cut-off stays away.
  '''
  c = sbc.comoving(laws, 0., x, logC, p, gm0=GM0)
  pieces, br = table_pieces(c, regime, None, gM0)
  g, fN = distribution(x, logC, dinj, laws, p, gM0, extra=br.values())
  N = csi.injected_number(x, dinj, laws)
  gnu = np.geomspace(g[0]/10., g[-1]/30., int(np.log10(300.*g[-1]/g[0])*80))
  Le = pl_spectrum(gnu, grid_pieces(g, fN/N))
  Lb = pl_spectrum(gnu, pieces)
  _, seg, br2, join = sbc.table(g, c, regime)
  Lj = pl_spectrum(gnu, grid_pieces(g, sbc.joined(g, seg, br2, join)))
  u = gnu**2                   # nu'/nu'_syn(gma=1, t'): ratios only
  rb, rj = Lb/Le, Lj/Le
  ub = {k: v**2 for k, v in br.items()}
  ends = list(ub.values()) + [gM0**2]
  clear = lambda uq, own: all(abs(np.log10(uq/v)) >= 1. for v in ends if v != own)
  out = dict(tau=c.tau)
  for k in ('fr', 'b', 'mi'):
    if k not in br:
      continue
    ux = ub[k]
    out[k] = tuple(float(_at(u, rb, uq)) if clear(uq, ux) else np.nan
                   for uq in (ux/100., ux, ux*100.))
    if k in ('fr', 'b'):
      m = np.abs(np.log10(u/ux)) <= 1.
      # keep 0.6 decade off the sharp edges (gma_low and gma_m,i), as shell_bpl_check
      for e in ('low', 'mi'):
        m &= np.abs(np.log10(u/ub[e])) > .6
      out[k + '_over'] = float(np.nanmax(rb[m])) if m.any() else np.nan
      out[k + '_harm'] = float(np.nanmax(np.abs(rj[m] - 1.))) if m.any() else np.nan
  return out


GRID_REF = ((2.5, 0., 0., 'FC', -2., (3., 6.)), (2.5, 0., 0., 'SC', 2., (10., 100.)))


def bpl_grid():
  '''
  the shell_bpl_check coasting grid (homogeneous shells only; FC at C = 1e-5 so that
  fr and mi are apart), tau up to 1000, then the reference case of the figures
  '''
  rows = []
  for p in (2.2, 2.5, 3.):
    for lN, lm in ((0., 0.), (1., 0.), (-.5, 0.), (0., .3), (1., .3)):
      L = sbc.coasting(lam_N=lN, lam_m=lm)
      for x in (3., 10., 30., 100.):
        if x < .8*L.R_tr(-5.):                 # continuous FC rows hold before t'_tr
          rows.append((p, lN, lm, 'FC', x, bpl_metrics(L, x, -5., 'FC', p), False))
      for x in (3., 10., 100., 1000.):
        rows.append((p, lN, lm, 'SC', x, bpl_metrics(L, x, 2., 'SC', p), False))
  for p, lN, lm, reg, lc, xs in GRID_REF:
    for x in xs:
      rows.append((p, lN, lm, reg, x, bpl_metrics(COASTING, x, lc, reg, p), True))
  return rows


def grid_ranges(rows):
  '''min-max over the grid rows (not the reference ones) of each item-2 number'''
  out = {}
  for _, lN, _, reg, _, o, ref in rows:
    if ref:
      continue
    for k in ('fr', 'b', 'mi'):
      if k not in o:
        continue
      for side, v in zip(('below', 'at', 'above'), o[k]):
        out.setdefault(f'{reg} {k} {side}', []).append(v)
      for t in ('over', 'harm'):
        if f'{k}_{t}' in o:
          out.setdefault(f'{reg} {k} {t}', []).append(o[f'{k}_{t}'])
          tag = 'linj=1' if lN == 1. else 'linj<1'
          out.setdefault(f'{reg} {k} {t} ({tag})', []).append(o[f'{k}_{t}'])
  return {k: (np.nanmin(v), np.nanmax(v)) for k, v in out.items()
          if np.any(np.isfinite(v))}


def kernel_smoothing(u, sf, ss, ux, u_lo, u_hi, tol=SLOPE_TOL):
  '''
  Item 4: the contiguous interval around u_X, inside [u_lo, u_hi], where the full-kernel
  local index departs from the step one by more than tol. (width, lo/u_X, hi/u_X)
  '''
  ok = np.isfinite(sf) & np.isfinite(ss) & (u >= u_lo) & (u <= u_hi)
  far = ok & (np.abs(sf - ss) > tol)
  if not far.any():
    return 0., np.nan, np.nan
  i0 = np.argmin(np.abs(np.log(u/ux)))
  # grow from the far point nearest u_X while the departure lasts
  idx = np.flatnonzero(far)
  k = idx[np.argmin(np.abs(idx - i0))]
  a = b = k
  while a > 0 and far[a - 1]:
    a -= 1
  while b < u.size - 1 and far[b + 1]:
    b += 1
  return np.log10(u[b]/u[a]), u[a]/ux, u[b]/ux


def transition_width(u, s, ux, k_lo, k_hi, u_lo, u_hi, tol=SLOPE_TOL):
  '''
  decades around u_X over which the local index s is more than tol from both asymptotes
  k_lo (below) and k_hi (above), searched in [u_lo, u_hi]; an unreached end is nan
  '''
  ok = np.isfinite(s) & (u >= u_lo) & (u <= u_hi)
  below = ok & (u <= ux) & (np.abs(s - k_lo) <= tol)
  above = ok & (u >= ux) & (np.abs(s - k_hi) <= tol)
  lo = u[below].max() if below.any() else np.nan
  hi = u[above].min() if above.any() else np.nan
  return np.log10(hi/lo), lo/ux, hi/ux


def smoothing(cases):
  '''per break: the full-vs-step departure, and the step and full transition widths'''
  out = []
  for name, laws, x, lc, dinj, regime, brk in cases:
    _, br, _, _ = bpl(np.array([1.]), x, lc, dinj, regime, laws)
    g, fN = distribution(x, lc, dinj, laws)
    br['M'] = nu_of_gma(g[-1], x, laws)     # the cut-off bounds the top break's window
    u = ugrid(nu_of_gma(g[0], x, laws)/1e3, br['M']*10., 60)
    Ls, Lf = spectra(u, x, g, fN, laws)
    ss, sf = local_slope(u, Ls), local_slope(u, Lf)
    for key, k_lo, k_hi, nb_lo, nb_hi in brk:
      ux = br[key]
      # bounded halfway (in log) to the neighbouring breaks
      lo = np.sqrt(ux*br[nb_lo]) if nb_lo else u[0]
      hi = np.sqrt(ux*br[nb_hi]) if nb_hi else u[-1]
      out.append((name, key, kernel_smoothing(u, sf, ss, ux, lo, hi),
                  transition_width(u, ss, ux, k_lo, k_hi, lo, hi),
                  transition_width(u, sf, ux, k_lo, k_hi, lo, hi)))
  return out


def smoothing_cases(p=P_SYN, laws=COASTING):
  kfr = (1. + laws.frozen_index)/2.         # (1 + lam_N)/(2 eta)
  th = 1./3.
  return (('B1 FC tau=10', laws, 10., FC_B, np.inf, 'FC',
           (('low', th, kfr, None, 'fr'), ('fr', kfr, -.5, 'low', 'mi'),
            ('mi', -.5, -p/2., 'fr', 'M'))),
          ('cont FC tau=3', laws, 3., -2., np.inf, 'FC',
           (('low', th, kfr, None, 'fr'), ('fr', kfr, -.5, 'low', 'mi'),
            ('mi', -.5, -p/2., 'fr', 'M'))),
          ('cont SC tau=10', laws, 10., 2., np.inf, 'SC',
           (('mi', th, -(p - 1.)/2., None, 'b'), ('b', -(p - 1.)/2., -p/2., 'mi', 'M'))),
          ('cont SC tau=100', laws, 100., 2., np.inf, 'SC',
           (('mi', th, -(p - 1.)/2., None, 'b'),)),
          ('fin FC tau=100', laws, 100., -2., 1., 'FC',
           (('low', th, kfr, None, 'hi'),)),
          ('fin SC tau=100', laws, 100., 2., 1., 'SC',
           (('low', th, -(p - 1.)/2., None, 'hi'),)))


def frozen_convergence(laws=COASTING, xs=X_B, tol=SLOPE_TOL):
  '''
  Item 3 on B1: the highest u/u_fr below which the local index stays within tol of
  kappa_fr (step and full), against the frozen-segment-alone formula and Eq. eqn:Lnu_bpl.
  '''
  kfr = (1. + laws.frozen_index)/2.
  d = 1./3. - kfr
  y_text = ((d/tol) + 1.)**(-1./d)          # the text's formula
  out = dict(kfr=kfr, text=y_text)
  for x in xs:
    Lb, br, _, _ = bpl(np.array([1.]), x, FC_B, np.inf, 'FC', laws)
    g, fN = distribution(x, FC_B, np.inf, laws, extra=())
    u = ugrid(br['low']*3., br['fr'], 60)
    Ls, Lf = spectra(u, x, g, fN, laws)
    Lb, _, _, _ = bpl(u, x, FC_B, np.inf, 'FC', laws)
    res = []
    for L in (Ls, Lf, Lb):
      s = local_slope(u, L)
      bad = np.abs(s - kfr) > tol
      # the first u above which the index leaves the band for good, away from nu'_low
      inside = ~bad & (u > 10.*br['low'])
      top = u[inside].max()/br['fr'] if inside.any() else np.nan
      res.append((top, float(s[(u > 10.*br['low'])].max()), float(s[(u > 10.*br['low'])].min())))
    out[x] = dict(range=br['fr']/br['low'], tr=laws.R_tr(FC_B), step=res[0], full=res[1],
                  bpl=res[2])
  return out


def full_over_step(p=P_SYN, laws=COASTING):
  '''full/step in mid-segment on B1 at its last time, against r(kappa)'''
  x = X_B[-1]
  _, br, _, _ = bpl(np.array([1.]), x, FC_B, np.inf, 'FC', laws)
  g, fN = distribution(x, FC_B, np.inf, laws)
  pts = dict(frozen=np.sqrt(br['low']*br['fr']), cooling=np.sqrt(br['fr']*br['mi']),
             cooled=br['mi']*1e4)
  u = np.array(list(pts.values()))
  Ls, Lf = spectra(u, x, g, fN, laws)
  kaps = ((1. + laws.frozen_index)/2., -.5, -p/2.)
  return {k: (Lf[i]/Ls[i], r_closed(kap)) for i, (k, kap) in enumerate(zip(pts, kaps))}


# --- Figure A -------------------------------------------------------------------------
# the article's break names (2026-10-20), as Fig. 4: the lowest edge written out,
# nu'_inf(t';t'_0) in fast and nu'_m(t';t'_0) in slow cooling; the cooling break nu'_c in
# both regimes (the table's keys 'fr' and 'b')
BREAK_LABS = {'FC': {'low': "$\\nu'_\\mathrm{M}(t';t'_0)$", 'fr': "$\\nu'_{\\rm c}$",
                     'mi': "$\\nu'_{\\mathrm{m},\\!i}$"},
              'SC': {'low': "$\\nu'_\\mathrm{m}(t';t'_0)$", 'b': "$\\nu'_{\\rm c}$",
                     'mi': "$\\nu'_{\\mathrm{m},\\!i}$"}}
FINITE_LABS = {'FC': {'low': "$\\nu'_\\mathrm{M}(t';t'_0)$",
                      'hi': "$\\nu'_\\mathrm{M}(t';t'_0+\\Delta t'_{\\rm inj})$"},
               'SC': {'low': "$\\nu'_\\mathrm{m}(t';t'_0+\\Delta t'_{\\rm inj})$",
                      'hi': "$\\nu'_\\mathrm{M}(t';t'_0)$"}}
# ('off', dx, dy) in points from the mark, or ('at', u, axes-y) with a leader line; set by
# eye on the render so that no box sits on a curve
BREAK_AT = {(0, 0, 'low'): ('at', 4e-4*_U, .86), (0, 0, 'hi'): ('at', 1e1*_U, .93),
            (0, 1, 'low'): ('at', 3e-3*_U, .93), (0, 1, 'fr'): ('at', 1e2*_U, .93),
            (0, 1, 'mi'): ('at', 1e6*_U, .93),
            (1, 0, 'low'): ('at', 1e-2*_U, .25), (1, 0, 'hi'): ('at', 1e2*_U, .08),
            (1, 1, 'low'): ('at', 1e-2*_U, .93), (1, 1, 'mi'): ('at', 2e3*_U, .93),
            (1, 1, 'b'): ('at', 1e9*_U, .12)}
TOP_ROOM = (60., 60.)
SL_LIM = (-2.1, .7)
Y_DEC = 9.                     # decades shown below each row's maximum


def _mark_breaks(ax, row, col, u, Lf, br, regime, finite):
  labs = (FINITE_LABS if finite else BREAK_LABS)[regime]
  for key in ('low', 'fr', 'mi', 'b', 'hi'):
    if key not in br:
      continue
    ub = br[key]
    yb = float(_at(u, Lf, ub))
    ax.plot([ub], [yb], marker='|', ms=9, mew=1.1, color=INK, zorder=6)
    where = BREAK_AT.get((row, col, key))
    if where is None:
      continue
    if where[0] == 'off':
      dx, dy = where[1:]
      ax.annotate(labs[key], (ub, yb), textcoords='offset points', xytext=(dx, dy),
                  ha='left' if dx >= 0 else 'right', va='bottom' if dy >= 0 else 'top',
                  color=INK, fontsize=FS_ANN, bbox=TEXT_BOX, zorder=7)
    else:
      ax.annotate(labs[key], (ub, yb), xytext=where[1:],
                  textcoords=('data', 'axes fraction'), ha='center', va='center',
                  color=INK, fontsize=FS_ANN, bbox=TEXT_BOX, zorder=7,
                  arrowprops=dict(arrowstyle='-', color=INK, lw=.5, shrinkA=0, shrinkB=5))


def _guides(p):
  '''the reference indices of the slope panels, by name'''
  return {'1/3': (1./3., '$1/3$'), '1/6': (1./6., '$1/6$'), '-1/2': (-.5, '$-1/2$'),
          '-(p-1)/2': (-(p - 1.)/2., '$-(p-1)/2$'), '-p/2': (-p/2., '$-p/2$')}


# which guides each slope panel draws: only those its curves follow over a segment (more
# than half a decade within 0.03, measured 2026-10-20), so a panel carries no idle lines.
# 1/6: the continuous FC panel only (1.3 dec; finite FC only crosses it); -(p-1)/2 there
# after t'_tr (0.5 dec); -1/2 in neither SC panel (<= 0.1 dec); B2 has no 1/6 (kappa = 1/3)
GUIDES_A = {(0, 0): ('1/3', '-1/2', '-p/2'),
            (0, 1): ('1/3', '1/6', '-1/2', '-(p-1)/2', '-p/2'),
            (1, 0): ('1/3', '-(p-1)/2', '-p/2'),
            (1, 1): ('1/3', '-(p-1)/2', '-p/2')}
GUIDES_B = (('1/3', '1/6', '-1/2', '-p/2'), ('1/3', '-1/2', '-p/2'))


def _split_va(labels, levs):
  '''labels of neighbouring levels closer than 0.4: lift the upper one, drop the lower'''
  for k in range(len(levs) - 1):
    if levs[k][0] - levs[k + 1][0] < .4:
      labels[k].set_va('bottom')
      labels[k + 1].set_va('top')


def _levels_right(axS, levs):
  '''the reference indices as named ticks on the right spine (as Fig. 4)'''
  levs = sorted(levs, reverse=True)
  axR = axS.twinx()
  axR.set_ylim(axS.get_ylim())
  axR.set_yticks([v for v, _ in levs])
  axR.set_yticklabels([t for _, t in levs])
  _split_va(axR.get_yticklabels(), levs)
  axR.tick_params(axis='y', labelsize=FS_ANN, length=2.5, pad=1.5, colors=INK)


def _levels_left(axS, levs):
  '''
  indices drawn in a left panel only: named ticks on its own (left) axis, among the
  numeric ones, since its right spine faces the next column
  '''
  if not levs:
    return
  num = [v for v in axS.get_yticks() if axS.get_ylim()[0] <= v <= axS.get_ylim()[1]
         and all(abs(v - lv) > .3 for lv, _ in levs)]
  ticks = sorted([(v, f'${v:g}$') for v in num] + list(levs), reverse=True)
  axS.set_yticks([v for v, _ in ticks])
  axS.set_yticklabels([t for _, t in ticks])


def _colour():
  lx = np.log10(np.array(X_SAMPLES) - 1.)          # log10(Delta t'/t'_0), as Fig. 4
  norm = plt.Normalize(vmin=lx.min(), vmax=lx.max())
  cmap = plt.cm.Blues
  return norm, cmap, (lambda x: cmap(.35 + .65*norm(np.log10(x - 1.))))


def _colorbar(fig, axs_top, axs_bot, norm, cmap):
  p0, p1 = axs_top.get_position(), axs_bot.get_position()
  cax = fig.add_axes([.935, p1.y0, .018, p0.y1 - p1.y0])
  sm = plt.cm.ScalarMappable(cmap=plt.cm.colors.ListedColormap(
      cmap(np.linspace(.35, 1., 256))), norm=norm)
  cb = fig.colorbar(sm, cax=cax)
  cb.set_label("$\\log_{10}(\\Delta t'/t'_0)$", fontsize=FS_LAB)
  cax.tick_params(labelsize=FS_TICK)


XLABEL = "$\\nu'/\\nu'_{{\\rm syn},0}$"
YLABEL = "$L'_{\\nu'}/(p'_{{\\rm syn},0}\\,\\dot N_0 t'_0)$"
SLABEL = "${\\rm d}\\ln L'_{\\nu'}/{\\rm d}\\ln\\nu'$"


def plot_shell_spectrum(outdir=OUTDIR, fname='cooling_shell_spectrum.png', show=False,
    p=P_SYN, laws=COASTING):
  '''
  Fig. A: the layout, cases and colours of Fig. 4. Full-kernel spectra coloured by time;
  on the Fig. 4 reference curve the step-kernel spectrum dotted, Eq. eqn:Lnu_bpl dashed in
  ink and its breaks as short marks. Slope panels with the indices of Table tab:shell_spectrum.
  '''
  norm, cmap, col = _colour()
  u = ugrid(*U_LIM)
  # the spacer row only holds the named ticks of the slow-cooling row, which then sit
  # between the two regime blocks and read for both
  fig, axs = plt.subplots(5, 2, figsize=FIGSIZE, sharex=True,
      gridspec_kw=dict(height_ratios=[1.9, 1., .13, 1.9, 1.], hspace=.08, wspace=.06))
  for ax in axs[2]:
    ax.set_visible(False)
  rows = ((axs[0], axs[1]), (axs[3], axs[4]))
  G = _guides(p)
  for i, lc in enumerate(LOGC_ROWS):
    axN_row, axS_row = rows[i]
    regime = 'FC' if lc < 0. else 'SC'
    top = 0.
    for j, (name, lab, dinj) in enumerate(CASES):
      axN, axS = axN_row[j], axS_row[j]
      xref = REF_X[(i, j)]
      for x in X_SAMPLES:
        g, fN = distribution(x, lc, dinj, laws, p)
        Ls, Lf = spectra(u, x, g, fN, laws)
        axN.loglog(u, Lf, color=col(x), lw=1.2, zorder=3)
        s = local_slope(u, Lf)
        s[~np.isfinite(s) | (s < SL_LIM[0]) | (s > SL_LIM[1])] = np.nan
        axS.semilogx(u, s, color=col(x), lw=1.1, zorder=3)
        top = max(top, np.nanmax(Lf))
        if x == xref:
          axN.loglog(u, np.where(Ls > 0., Ls, np.nan), color=col(x), lw=1.1, ls=':',
                     zorder=4)
          Lb, br, _, _ = bpl(u, x, lc, dinj, regime, laws, p)
          axN.loglog(u, np.where(Lb > 0., Lb, np.nan), color=INK, ls=(0, (3, 2)), lw=.9,
                     zorder=5)
          _mark_breaks(axN, i, j, u, Lf, br, regime, not np.isinf(dinj))
      for lev, _ in (G[k] for k in GUIDES_A[(i, j)]):
        axS.axhline(lev, color=MUTED, ls='--', lw=.7, zorder=1)
      axS.set_ylim(*SL_LIM)
      for ax in (axN, axS):
        ax.tick_params(which='both', labelsize=FS_TICK)
      axN.annotate(f'{lab}\n$\\gamma_{{\\mathrm{{m}},\\!0}}I_{{{{\\rm dyn}},0}}=10^{{{-lc:.0f}}}$', (.98, .97), xycoords='axes fraction',
                   fontsize=FS_ANN, color=INK, ha='right', va='top', multialignment='right',
                   bbox=TEXT_BOX, zorder=7)
      if j == 0:
        axN.set_ylabel(YLABEL, fontsize=FS_LAB)
        axS.set_ylabel(SLABEL, fontsize=FS_LAB)
      else:
        axN.tick_params(labelleft=False)
        axS.tick_params(labelleft=False)
        # the right column's guides on its right spine, the left column's own ones (if
        # any) on its left axis
        _levels_right(axS, [G[k] for k in GUIDES_A[(i, 1)]])
        _levels_left(axS_row[0], [G[k] for k in GUIDES_A[(i, 0)]
                                  if k not in GUIDES_A[(i, 1)]])
    for ax in axN_row:
      ax.set_ylim(top*10.**-Y_DEC, top*TOP_ROOM[i])
  for ax in axs[4]:
    ax.set_xlabel(XLABEL, fontsize=FS_LAB)
  axs[0, 0].set_xlim(*U_LIM)
  axs[0, 0].set_xticks(10.**np.arange(-4., 17., 4.)*_U)   # none on the edges: the columns touch
  # named once, between the two rows, with bare marks pointing down from the row above
  for c in (0, 1):
    named_ticks_top(axs[3, c], ((GM0**2, "$\\nu'_{\\rm syn}(\\gamma_{\\mathrm{m},\\!0})$"),
                                (GMA_M0**2, "$\\nu'_{\\rm syn}(\\gamma_{\\mathrm{M},\\!0})$")))
    marks_bottom(axs[1, c], (GM0**2, GMA_M0**2))
  fig.subplots_adjust(right=.84)
  h = [plt.Line2D([], [], color=MUTED, lw=1.2, label='full kernel'),
       plt.Line2D([], [], color=MUTED, lw=.9, ls=':', label='step kernel (one curve)'),
       plt.Line2D([], [], color=INK, lw=.9, ls=(0, (3, 2)), label='broken power law'),
       plt.Line2D([], [], color=INK, lw=0, marker='|', ms=7, mew=1.1, label='its breaks')]
  pa, pb = axs[0, 0].get_position(), axs[0, 1].get_position()
  fig.canvas.draw()
  r_ = fig.canvas.get_renderer()
  ytop = max(fig.transFigure.inverted().transform(ax.get_tightbbox(r_))[1, 1]
             for ax in axs[0])
  fig.legend(handles=h, loc='lower center', ncol=4, bbox_to_anchor=(.5*(pa.x0 + pb.x1),
             ytop + .004), bbox_transform=fig.transFigure, fontsize=FS_LEG,
             frameon=False, handlelength=2., columnspacing=1.2)
  _colorbar(fig, axs[0, 1], axs[4, 1], norm, cmap)
  os.makedirs(outdir, exist_ok=True)
  path = os.path.join(outdir, fname)
  fig.savefig(path, dpi=300, bbox_inches='tight')
  print(f'saved {path}')
  if show:
    plt.show()
  return fig, axs


# --- Figure B -------------------------------------------------------------------------
FIGSIZE_B = (3.5, 5.9)
U_LIM_B = (1e-6, 1e11)
FROZEN_LS = dict(lw=.9, ls=(0, (5, 1.5, 1, 1.5)))   # dash-dot, in the curve's colour
B_LABS = (('low', BREAK_LABS['FC']['low'], ('off', 0, -11)),
          ('fr', BREAK_LABS['FC']['fr'], ('at', 3e3, .3)))
# nu'_c (key 'fr') named in the free space below, with a leader: the frozen-alone spectrum drops
# vertically at the mark and the steep segments run just right of it


def plot_frozen(outdir=OUTDIR, fname='cooling_shell_spectrum_frozen.png', show=False,
    p=P_SYN):
  '''
  Fig. B: continuous injection in fast cooling with gbar_c,0 = 1e-3 gma_m,0, at
  tau = 10, 13, 16 < t'_tr = 19.3: B1 lam_inj = 0 (kappa_fr = 1/6), B2 lam_inj = 2 (1/3).
  Full kernel solid, step dotted, nu'_low and nu'_fr marked on each curve (named on the
  last); the frozen segment alone (frozen_alone) dash-dot in the curve's colour, and in the slope panels
  its local index (the text's formula, kappa = (1 + lam_inj)/2 eta) between nu'_low and
  nu'_fr.
  '''
  norm, cmap, _ = _colour()
  # tau = 10, 13, 16 are one shade on Fig. A's 1.1-100 ramp: the same Blues, stretched over
  # the three times (the key above the panels names them)
  col = lambda x: cmap(.45 + .55*(x - X_B[0])/(X_B[-1] - X_B[0]))
  u = ugrid(*U_LIM_B, n=40)
  fig, axs = plt.subplots(5, 1, figsize=FIGSIZE_B, sharex=True,
      gridspec_kw=dict(height_ratios=[1.6, 1., .22, 1.6, 1.], hspace=.08))
  axs[2].set_visible(False)
  for i, (laws, tag) in enumerate(((COASTING, '$\\lambda_{\\rm inj}=0$'),
                                   (LAWS_B2, '$\\lambda_{\\rm inj}=2$'))):
    axN, axS = axs[3*i], axs[3*i + 1]
    kraw = (1. + laws.lam_N)/(2.*(laws.lam_n/3. + laws.lam_c))
    top = 0.
    for x in X_B:
      g, fN = distribution(x, FC_B, np.inf, laws, p)
      Ls, Lf = spectra(u, x, g, fN, laws)
      _, br, pieces, _ = bpl(np.array([1.]), x, FC_B, np.inf, 'FC', laws, p)
      axN.loglog(u, Lf, color=col(x), lw=1.2, zorder=3)
      # the frozen segment alone: Eq. eqn:Lnu_bpl on the table's [gma_low, gma_fr] piece
      Lz = frozen_alone(u, x, pieces, laws)
      axN.loglog(u, np.where(Lz > 0., Lz, np.nan), color=col(x), zorder=5, **FROZEN_LS)
      axN.loglog(u, np.where(Ls > 0., Ls, np.nan), color=col(x), lw=1.1, ls=':', zorder=4)
      for key, lab, (how, a, b) in B_LABS:
        yb = float(_at(u, Lf, br[key]))
        axN.plot([br[key]], [yb], marker='|', ms=7, mew=1., color=INK, zorder=6)
        if x != X_B[-1] or i != 0:
          continue
        if how == 'off':
          axN.annotate(lab, (br[key], yb), textcoords='offset points', xytext=(a, b),
                       ha='center', va='top', color=INK, fontsize=FS_ANN, bbox=TEXT_BOX,
                       zorder=7)
        else:
          axN.annotate(lab, (br[key], yb), xytext=(a, b),
                       textcoords=('data', 'axes fraction'), ha='center', va='center',
                       color=INK, fontsize=FS_ANN, bbox=TEXT_BOX, zorder=7,
                       arrowprops=dict(arrowstyle='-', color=INK, lw=.5, shrinkA=0,
                                       shrinkB=4))
      for L, ls in ((Lf, '-'), (Ls, ':')):
        s = local_slope(u, L)
        s[~np.isfinite(s) | (s < SL_LIM[0]) | (s > SL_LIM[1])] = np.nan
        axS.semilogx(u, s, color=col(x), lw=1.1, ls=ls, zorder=3)
      m = (u > br['low']) & (u < br['fr'])
      axS.semilogx(u[m], kappa_fr_local(u[m], br['fr'], kraw), color=col(x), zorder=5,
                   **FROZEN_LS)
      top = max(top, Lf.max())
    axN.set_ylim(top*10.**-(5. + 2.*i), top*8.)   # room for the style key under B2
    levs = [_guides(p)[k] for k in GUIDES_B[i]]
    for lev, _ in levs:
      axS.axhline(lev, color=MUTED, ls='--', lw=.7, zorder=1)
    axS.set_ylim(-1.6, .6)
    _levels_right(axS, levs)
    # gbar_c(t'_0) is the same in both panels: it goes in the caption, not here
    axN.annotate(tag, (.03, .95), xycoords='axes fraction', fontsize=FS_ANN, color=INK,
                 ha='left', va='top', zorder=7)
    axN.set_ylabel(YLABEL, fontsize=FS_LAB)
    axS.set_ylabel(SLABEL, fontsize=FS_LAB)
    for ax in (axN, axS):
      ax.tick_params(which='both', labelsize=FS_TICK)
  axs[4].set_xlabel(XLABEL, fontsize=FS_LAB)
  axs[0].set_xlim(*U_LIM_B)
  axs[0].set_xticks(10.**np.arange(-3., 11., 3.))
  # two keys: the times (Blues, stretched over the three) above the panels, the line
  # styles in the free lower left of the second panel, under the curves
  hc = [plt.Line2D([], [], color=col(x), lw=2., label=f"$t'={x:g}\\,t'_0$") for x in X_B]
  hs = [plt.Line2D([], [], color=MUTED, lw=1.2, label='full kernel'),
        plt.Line2D([], [], color=MUTED, lw=1.1, ls=':', label='step kernel'),
        plt.Line2D([], [], color=MUTED, label='frozen segment alone', **FROZEN_LS)]
  fig.canvas.draw()
  r_ = fig.canvas.get_renderer()
  inv = fig.transFigure.inverted()
  ytop = inv.transform(axs[0].get_tightbbox(r_))[1, 1]
  pa = axs[0].get_position()
  xc = .5*(pa.x0 + pa.x1)
  kw = dict(bbox_transform=fig.transFigure, fontsize=FS_LEG, frameon=False,
            handlelength=1.8, columnspacing=1.2)
  fig.legend(handles=hc, loc='lower center', ncol=3, bbox_to_anchor=(xc, ytop + .004), **kw)
  axs[3].legend(handles=hs, loc='lower left', fontsize=FS_LEG, frameon=False,
                handlelength=2.2, borderaxespad=.4)
  os.makedirs(outdir, exist_ok=True)
  path = os.path.join(outdir, fname)
  fig.savefig(path, dpi=300, bbox_inches='tight')
  print(f'saved {path}')
  if show:
    plt.show()
  return fig, axs


# --- main -----------------------------------------------------------------------------
def report_checks():
  k = check_kernel()
  print(f"x_syn = {k['x_syn']:.4f}, nu'_syn = {k['nu_coef']:.4f} gma^2 nu'_B, p'_syn = "
        f"{k['p_coef']:.4f} sigma_T m_e c^2 B'/e; int func_R dx = {k['I_S_func_R']:.5f} "
        f"(16pi/27sqrt3 = {I_S:.5f})")
  for kap, rc, rn in k['r']:
    print(f'  r(kappa={kap:+.4f}) = {rc:.4f} closed, {rn:.4f} func_R')
  for name, (fs, rk) in full_over_step().items():
    print(f'  B1 tau={X_B[-1]:g} mid-{name}: full/step = {fs:.4f}, r = {rk:.4f}')
  # Eq. eqn:Lnu_bpl against the same table integrated as a fine piecewise interpolant
  c = sbc.comoving(COASTING, 0., 10., 2., P_SYN, gm0=GM0)
  pieces, br = table_pieces(c, 'SC', None, GMA_M0)
  g = np.unique(np.concatenate([np.geomspace(br['low'], GMA_M0, 4000), list(br.values())]))
  f, _, _, _ = sbc.table(g, c, 'SC')
  gnu = np.geomspace(br['low']/10., GMA_M0/3., 200)
  print(f"  Eq. Lnu_bpl vs interpolant of the table (SC tau=10): max |ratio-1| = "
        f"{np.max(np.abs(pl_spectrum(gnu, pieces)/pl_spectrum(gnu, grid_pieces(g, f)) - 1.)):.1e}")
  print('item 2: BPL/exact (step) at nu_X/100, nu_X, 100 nu_X; over = max within a decade;'
        ' harm = max |joined/exact - 1| within a decade')
  rows = bpl_grid()
  for p, lN, lm, reg, x, o, ref in rows:
    s = f'  {"REF " if ref else ""}p={p:g} linj={lN:+g} lm={lm:g} {reg} tau={x:6g}:'
    for kk in ('fr', 'b', 'mi'):
      if kk in o:
        s += f' {kk} ' + '/'.join(f'{v:.3f}' for v in o[kk])
      if kk + '_over' in o:
        s += f' over {o[kk + "_over"]:.2f} harm {o[kk + "_harm"]:.3f}'
    print(s, flush=True)
  for k, (lo, hi) in grid_ranges(rows).items():
    print(f'  range {k:24s} {lo:.3f} - {hi:.3f}')
  fc = frozen_convergence()
  print(f"item 3: kappa_fr = {fc['kfr']:.4f}; text formula: within {SLOPE_TOL} below "
        f"{fc['text']:.2e} nu'_fr")
  for x in X_B:
    o = fc[x]
    print(f"  tau={x:g} (t'_tr = {o['tr']:.1f}): nu'_fr/nu'_low = {o['range']:.2e}; "
          + '; '.join(f"{n}: top {o[n][0]:.2e} nu'_fr, index {o[n][2]:+.3f}..{o[n][1]:+.3f}"
                      for n in ('step', 'full', 'bpl')))
  print('item 4: per break, decades where |full - step index| > tol [ends in nu/nu_X]; '
        'then the transition width (index > tol from both asymptotes), step | full')
  for name, key, d, ws, wf in smoothing(smoothing_cases()):
    print(f'  {name:16s} {key:4s} full-step {d[0]:5.2f} [{d[1]:.2g}, {d[2]:.2g}]   '
          f'step {ws[0]:5.2f} [{ws[1]:.2g}, {ws[2]:.2g}]   full {wf[0]:5.2f} '
          f'[{wf[1]:.2g}, {wf[2]:.2g}]')


def main(show=False, checks=True):
  if checks:
    report_checks()
  plot_shell_spectrum(show=show)
  plot_frozen(show=show)


if __name__ == '__main__':
  main()
