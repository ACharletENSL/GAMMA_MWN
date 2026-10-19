# -*- coding: utf-8 -*-
# @Author: acharlet

'''
Paper I, Table tab:shell_bpl (Sect. 3.2) and its radius mapping (Sect. 3.3), against the
exact shell integral of cooling_shell_integrated.shell_integrated. Replaces the scratch
scripts of the article's notes/scripts/ (test4.py & co.), which had their own integrator.

THE TABLE is written in comoving time: tau = t'/t'_0, injection indices lam_inj, lam_m,
lam_c, element indices lam_n, s. shell_integrated works in radius, with the RADIAL indices
of ShellLaws. The mapping of Sect. 3.3, for a shock with Gamma_sh^2 ~ R^-m and elements
with Gamma ~ R^-g (a1 = 1 + m/2):

    tau = (R/R_0)^a1,   lam_X = lam_X^rad/a1 (X = c, m, n~),   1 + lam_inj = (1 + lam_N)/a1,
    s = Lam/(1 + g),    gbar_c,i = (1 + g) gci^rad     (Eq. eqn:plaws_R),

so that s gbar_c,i = Lam gci^rad, the frozen edge, is clock-independent.

THE AGEING FACTOR f = a1/(1 + g). An element injected at R_i has, at R, the age
t'_i[(R/R_i)^(1+g) - 1], while the injection clock has advanced by t'_i[(R/R_i)^a1 - 1]:
for g > m/2 elements age faster than the shell is filled, and the steady-state segments,
whose height is (injection rate) x (cooling time), are lower by f. MAPPING = 'article'
applies Sect. 3.3 as printed (f = 1), MAPPING = 'aged' multiplies every bare gbar_c,i by f
(s gbar_c,i unchanged). f = 1 in a homogeneous shell (g = m/2) and in the coasting cases.

METRICS, per (case, tau): the exact/table ratio of dN/(N dln gma), as a function of the
distance in decades to the nearest break of the table (gma_low, gma_fr or gma_b, gma_m,i),
the support's upper edge (gma_M, outside the table) excluded within 2/(p-1) decades:
  seg1, seg2   max |ratio - 1| at >= 1 and >= 2 decades from every break
  seg03        the same at >= 0.3 decade (finite injection, whose segments are short)
  over         max table/exact within 1 decade of gma_fr or gma_b (the smooth breaks)
  harm         max |ratio - 1| of the harmonic joining (f_j^-1 + f_j+1^-1)^-1 across them,
               more than 0.3 dex from gma_low and gma_m,i (sharp edges of the table)

RESULTS (2026-10-20; run main, ~5 s; p = 2.2, 2.5, 3, lam_inj = -1/2, 0, 1, lam_m = 0, 0.3,
coasting and frozen-flux elements, BM k = 0, 2 and a g = 0 < m/2 shell, tau = 3-1000):
  * Sect. 3.3 as printed misses f: every steady-state segment is off by 1/f - 1 at all
    distances from the breaks (-46% BM k=0, f = 5/9; -42% k=2, f = 3/5; +50% for g = 0,
    m = 1). With 'aged' the R-mapped cases fall back to the homogeneous accuracy.
  * The breaks are approached as power laws of the distance, so 'one decade away' is not
    one number. FC, both sides of gma_fr: 1 - ratio ~ 1.2-1.5 (gma/gma_fr)^(+-1), i.e.
    10-15% at one decade, 1-1.6% at two. SC, below gma_b: 1 - ratio ~ (gma/gma_b)^(P/eta)
    (prefactor 0.9-1.2 while P < eta; a term ~ gma/gma_b takes over at P >= eta): the
    elements whose frozen edge, ~ t_i'^eta, is already below gma, weighted x^(P-1) as in
    Eq. eqn:adiabatic_int. SC: 10-27% at one decade, 1-10% at two (P/eta = 0.5: 10%).
  * Table/exact within a decade of gma_fr, gma_b: 2.0-2.5 (FC lam_inj = 1 and frozen flux:
    3.0-3.2; R mapping, aged: 1.5-2.1). Harmonic joining: 3-24% (FC lam_inj = 1 and frozen
    flux: 34-37%).
  * Finite injection (Delta t'_inj = t'_0), more than 0.3 dex inside the row's edges: frozen
    FC 1-2% from tau = 10 (15% at tau = 3, the last elements still freezing); SC 8% (p=2.5),
    4% (2.2), 16% (3). The closed form Eq. eqn:finite_SC: 0.7/1.5/3.2% (p = 2.2/2.5/3) more
    than a decade below gma_inf(t';t'_0), overestimates by 10% at -0.2 dex, 17-32% at
    gma_inf(t';t'_0) and up to x2 (x3 at p=3) in the tail above, toward gma_inf(t';t'_end):
    the neglected (1 - gma tt_eff)^(p-2), whose deficit grows with p - 2. Neither finite row
    involves f (only s gbar_c,i enters).
'''

from dataclasses import dataclass
import numpy as np

import cooling_shell_integrated as csi
from cooling_shell_integrated import ShellLaws, bm_forward_shock, LAM_EPS

GM0, GMA_M0 = 1e3, 1e14
N_PER_DEC = 40
MAPPINGS = ('article', 'aged')


# --- the table, in the article's comoving variables -----------------------------------
@dataclass
class Comoving:
  tau: float
  lam_inj: float
  lam_m: float
  lam_c: float
  lam_n: float           # tilde lam_n in the radius mapping
  s: float
  gc0: float             # gbar_c,0 (bare, the f-scaled one when mapping = 'aged')
  gm0: float
  p: float

  @property
  def a(self):
    return 1. + self.lam_inj

  @property
  def eta(self):
    return self.lam_n/3. + self.lam_c

  @property
  def zeta(self):
    return self.lam_n/3. - self.lam_m

  @property
  def P(self):
    return self.a + (self.p - 1.)*self.zeta

  def F(self, tau):
    return 1./(1. - tau**(-self.a))

  def gci(self, tau):
    return self.gc0*tau**self.lam_c

  def gmi(self, tau):
    return self.gm0*tau**(-self.lam_m)

  def ginf(self, tau_i):
    '''frozen edge at tau of the element injected at tau_i, Eq. eqn:mapping'''
    return self.s*self.gci(tau_i)*(self.tau/tau_i)**(-self.lam_n/3.)

  def gm_of(self, tau_i):
    return self.gmi(tau_i)*(self.tau/tau_i)**(-self.lam_n/3.)


def comoving(laws, m, x, logC, p, mapping='article', gm0=GM0):
  '''the table's parameters at R = x R_0, from the radial laws and the shock index m'''
  a1 = 1. + m/2.
  g1 = 1. + laws.g
  s = laws.Lam/g1
  gc0 = g1*csi.gbar_ci0_of_C(logC, laws, gm0)
  if mapping == 'aged':
    gc0 *= a1/g1
    s *= g1/a1
  return Comoving(tau=x**a1, lam_inj=(1. + laws.lam_N)/a1 - 1., lam_m=laws.lam_m/a1,
                  lam_c=laws.lam_c/a1, lam_n=laws.lam_n/a1, s=s, gc0=gc0, gm0=gm0, p=p)


def table(gma, c, regime, tau_end=None):
  '''
  dN/(N dln gma) of Table tab:shell_bpl, its segments (each extended over all gma, for the
  harmonic joining) and its breaks. regime: 'FC'/'SC' (continuous injection, or finite
  after its end when tau_end is given).
  '''
  p, a, eta, zeta, P = c.p, c.a, c.eta, c.zeta, c.P
  if tau_end is None:
    t, F = c.tau, c.F(c.tau)
    sgc, gc, gm = c.s*c.gci(t), c.gci(t), c.gmi(t)
    if regime == 'FC':
      seg = dict(frozen=a*F/eta*(gma/sgc)**(a/eta), cooling=a*F*gc/gma,
                 cooled=a*F*gc/gm*(gma/gm)**(-p))
      br = dict(low=sgc*t**(-eta), fr=sgc*(eta/c.s)**(eta/(a + eta)), mi=gm)
      f = np.where(gma < br['fr'], seg['frozen'],
                   np.where(gma < gm, seg['cooling'], seg['cooled']))
      join = (('frozen', 'cooling', None, gm), ('cooled', None, gm, None))
    else:
      C = (p - 1.)*a*F/P
      seg = dict(adiabatic=C*((gma/gm)**(a/zeta) - t**(-P)*(gma/gm)**(1. - p)),
                 injected=C*(1. - t**(-P))*(gma/gm)**(1. - p),
                 cooled=a*F*gc/gm*(gma/gm)**(-p))
      br = dict(low=gm*t**(-zeta), mi=gm, b=P*gc/((p - 1.)*(1. - t**(-P))))
      f = np.where(gma < gm, seg['adiabatic'],
                   np.where(gma < br['b'], seg['injected'], seg['cooled']))
      join = (('adiabatic', None, None, gm), ('injected', 'cooled', gm, None))
    lo = br['low']
  else:
    F = c.F(tau_end)
    if regime == 'FC':
      hi = c.ginf(tau_end)
      seg = dict(frozen=a*F/eta*(gma/hi)**(a/eta))
      br = dict(low=c.ginf(1.), hi=hi)
      f = np.where(gma < hi, seg['frozen'], 0.)
    else:
      gme = c.gm_of(tau_end)
      seg = dict(slow=(p - 1.)*a*F/P*(1. - tau_end**(-P))*(gma/gme)**(1. - p))
      br = dict(low=gme, hi=c.ginf(1.))
      f = np.where(gma < br['hi'], seg['slow'], 0.)
    join = None
    lo = br['low']
  f = np.where(gma >= lo, f, 0.)
  return f, seg, br, join


def joined(gma, seg, br, join):
  '''the table with (f_j^-1 + f_j+1^-1)^-1 across the smooth break of each piece'''
  out = np.zeros_like(gma)
  for s1, s2, lo, hi in join:
    v = seg[s1] if s2 is None else 1./(1./seg[s1] + 1./seg[s2])
    m = np.ones(gma.shape, bool)
    if lo is not None:
      m &= gma >= lo
    if hi is not None:
      m &= gma < hi
    out = np.where(m, v, out)
  return np.where(gma >= br['low'], out, 0.)


def finite_SC(gma, c, tau_end):
  '''the closed form Eq. eqn:finite_SC, as dN/(N dln gma)'''
  p, a, eta, zeta, P = c.p, c.a, c.eta, c.zeta, c.P
  xm = np.maximum(1., (gma/c.ginf(1.))**(1./eta))
  xp = np.minimum(tau_end, (gma/c.gm_of(1.))**(1./zeta))
  N = (tau_end**a - 1.)/a                                    # in Ndot_0 t'_0
  dN = (p - 1.)*c.gm_of(1.)**(p - 1.)*gma**(1. - p)*(xp**P - xm**P)/P
  return np.where(xp > xm, dN/N, 0.)


# --- the comparison -------------------------------------------------------------------
def exact(gma, x, logC, laws, p, dinj=np.inf, gm0=GM0, gM0=GMA_M0):
  '''shell_integrated as dN/(N dln gma)'''
  N = csi.shell_integrated(gma, x, logC, dinj, laws, p, gm0, gM0)
  return gma*N/csi.injected_number(x, dinj, laws)


def compare(laws, m, x, logC, p, regime, mapping='article', dinj=np.inf, gm0=GM0,
    gM0=GMA_M0):
  '''metrics of the docstring at R = x R_0; dinj in R_0 (finite: after the end only)'''
  lo, hi = csi.support(x, logC, dinj, laws, gm0, gM0)
  gma = np.geomspace(lo*(1. + 1e-9), hi*(1. - 1e-9),
                     int(np.log10(hi/lo)*N_PER_DEC) + 2)
  ex = exact(gma, x, logC, laws, p, dinj, gm0, gM0)
  c = comoving(laws, m, x, logC, p, mapping, gm0)
  tau_end = None if np.isinf(dinj) else (1. + dinj)**(1. + m/2.)
  tb, seg, br, join = table(gma, c, regime, tau_end)
  smooth = [br[k] for k in ('fr', 'b') if k in br]
  sharp = [br[k] for k in ('low', 'mi', 'hi') if k in br]
  d = lambda v: np.abs(np.log10(gma/v))
  dist = np.min([d(v) for v in smooth + sharp], axis=0)
  # gma_M,i is outside the table: its cut-off eats (gma/gma_M)^(p-1) of the -(p+1) segment,
  # so keep 2/(p-1) decades off (1% there). After a finite injection every element is frozen
  # and the support's top is the table's own last edge, gma_inf(t'; t'_end)
  ok = ((d(hi) > max(1., 2./(p - 1.))) | (tau_end is not None)) & (tb > 0.) & (ex > 0.)
  r = np.where(ok, ex/np.where(tb > 0., tb, 1.), np.nan)
  out = dict(tau=c.tau, breaks=br, seg1=_mx(np.abs(r - 1.), ok & (dist >= 1.)),
             seg2=_mx(np.abs(r - 1.), ok & (dist >= 2.)),
             seg03=_mx(np.abs(r - 1.), ok & (dist >= .3)))
  if smooth:
    # at the earliest times gma_low sits within a decade of gma_fr: its cut-off is not
    # what the smooth-break numbers measure
    far_sharp = np.min([d(v) for v in sharp], axis=0) > .3
    near = ok & far_sharp & (np.min([d(v) for v in smooth], axis=0) <= 1.)
    out['over'] = _mx(1./r, near)
    jn = joined(gma, seg, br, join)
    okj = ok & (jn > 0.) & far_sharp
    out['harm'] = _mx(np.abs(ex/np.where(jn > 0., jn, 1.) - 1.), okj)
  if tau_end is not None and regime == 'SC':
    cf = finite_SC(gma, c, tau_end)
    okc = (ex > 0.) & (cf > 0.)
    rc = np.where(okc, ex/np.where(cf > 0., cf, 1.), np.nan)
    out['closed'] = (gma, rc, br)
  return out


def _mx(v, m):
  return float(np.nanmax(v[m])) if np.any(m) else np.nan


# --- the cases ------------------------------------------------------------------------
def coasting(lam_n=2., lam_B=5./3., lam_N=0., lam_m=0., label='coasting'):
  return ShellLaws(lam_n=lam_n, lam_B=lam_B, lam_N=lam_N, lam_m=lam_m, label=label)


def cases():
  '''(name, laws, m, logC, regime, taus as x = R/R_0)'''
  out = []
  for p in (2.2, 2.5, 3.):
    for lN, lm in ((0., 0.), (1., 0.), (-.5, 0.), (0., .3), (1., .3)):
      L = coasting(lam_N=lN, lam_m=lm)
      tag = f'coast p={p:g} linj={lN:g} lm={lm:g}'
      out.append((tag, L, 0., -5., 'FC', p, (3., 10., 30.)))
      out.append((tag, L, 0., 2., 'SC', p, (3., 10., 100.)))
  ff = coasting(lam_B=1., label='frozen flux')
  out.append(('frozen flux p=2.5', ff, 0., 2., 'SC', 2.5, (3., 10., 100.)))
  out.append(('frozen flux p=2.5', ff, 0., -4., 'FC', 2.5, (3., 10., 30.)))
  # radius mapping: BM forward shock (g = 7/2 - k > m/2) and a coasting spreading shell
  # behind a decelerating shock (g = 0 < m/2), lam_N and lam_m in radius
  for k in (0., 2.):
    L = bm_forward_shock(k)
    out.append((f'BM k={k:g}', L, 3. - k, -3., 'FC', 2.5, (1.5, 3., 10.)))
    out.append((f'BM k={k:g}', L, 3. - k, 2., 'SC', 2.5, (1.5, 3., 10.)))
  sp = ShellLaws(lam_n=3., lam_B=2.5, g=0., lam_c=1., lam_m=.5, lam_N=1., label='spread')
  out.append(('spreading, m=1', sp, 1., -4., 'FC', 2.5, (3., 10.)))
  out.append(('spreading, m=1', sp, 1., 2., 'SC', 2.5, (3., 10., 100.)))
  return out


def run_continuous(mappings=MAPPINGS):
  rows = []
  for name, L, m, lc, reg, p, xs in cases():
    for mp in (mappings if m != 2.*L.g else ('article',)):
      for x in xs:
        o = compare(L, m, x, lc, p, reg, mp)
        rows.append((name, reg, mp, x, o))
        print(f'{name:30s} {reg} {mp:7s} tau={o["tau"]:8.3g}  seg1 {o["seg1"]:6.3f}  '
              f'seg2 {o["seg2"]:6.3f}  over {o.get("over", np.nan):5.2f}  '
              f'harm {o.get("harm", np.nan):6.3f}', flush=True)
  return rows


def finite_cases():
  '''(name, laws, m, logC, regime, p, dinj in R_0, xs): injection over one doubling'''
  ref = coasting()
  out = [('coasting', ref, 0., -2., 'FC', 2.5, 1., (3., 10., 100.)),
         ('coasting', ref, 0., 2., 'SC', 2.5, 1., (3., 10., 100.))]
  for p in (2.2, 3.):
    out.append((f'coasting p={p:g}', ref, 0., 2., 'SC', p, 1., (10.,)))
  out.append(('coasting linj=1 lm=0.3', coasting(lam_N=1., lam_m=.3), 0., 2., 'SC', 2.5,
              1., (10.,)))
  for k in (0., 2.):
    out.append((f'BM k={k:g}', bm_forward_shock(k), 3. - k, 2., 'SC', 2.5, 1., (3., 10.)))
  return out


PROFILE = (-1., -.5, -.2, -.1, 0., .2, .4, .6)


def closed_form_profile(gma, rc, br, edges=PROFILE):
  '''
  exact / Eq. eqn:finite_SC at log10(gma/gma_inf(t';t'_0)) = edges (the closed form keeps
  the elements' frozen edges, up to gma_inf(t';t'_end)), and its worst deviation in the
  middle, > 3 gma_low and < gma_inf(t';t'_0)/10
  '''
  lg = np.log10(gma/br['hi'])
  ok = np.isfinite(rc)
  mid = ok & (gma > 3.*br['low']) & (lg < -1.)
  prof = [float(np.interp(e, lg[ok], rc[ok])) for e in edges]
  return float(np.nanmax(np.abs(rc[mid] - 1.))) if np.any(mid) else np.nan, prof


def run_finite(mappings=MAPPINGS):
  rows = []
  for name, L, m, lc, reg, p, dinj, xs in finite_cases():
    for mp in (mappings if m != 2.*L.g else ('article',)):
      for x in xs:
        o = compare(L, m, x, lc, p, reg, mp, dinj=dinj)
        rows.append((name, reg, mp, x, o))
        # the finite segments span ~1-2 decades: only the 0.3 dex interior is meaningful
        line = f'{name:24s} {reg} {mp:7s} tau={o["tau"]:7.3g}  seg03 {o["seg03"]:6.3f}'
        if 'closed' in o:
          mid, prof = closed_form_profile(*o['closed'])
          line += (f'  closed form: mid {mid:.3f}, at log(g/g_inf,0) = '
                   + ','.join(f'{e:g}' for e in PROFILE) + ': '
                   + ' '.join(f'{v:.3f}' for v in prof))
        print(line, flush=True)
  return rows


def main():
  run_continuous()
  run_finite()


if __name__ == '__main__':
  main()
