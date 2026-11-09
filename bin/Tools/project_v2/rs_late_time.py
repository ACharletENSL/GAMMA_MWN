# -*- coding: utf-8 -*-
# @Author: acharlet

'''
Paper I, Sect. 5 (reworked after the 2026-11-03 meeting): late-time emission of the
reverse-shocked ejecta, for three physical cases applied to k = 0 (uniform medium) and k = 2
(wind), with the full synchrotron kernel and one-zone self-absorption, compared with the
post-crossing models of the literature.

THE THREE CASES (Granot 2012, Sect. 2; Nakamura & Shigeyama 2006):
  thin shell   xi = Gamma4^2 Delta0/R_dec << 1: the shell spreads, Delta4 = Delta0 + R/Gamma4^2,
               Newtonian reverse shock (limit of the jump conditions), Gamma3 = Gamma4,
               Gamma34 - 1 = phi^2/2, dN3/N_ej = (4 phi/3) dx/(xi + x)          -> FINITE injection
  thick shell  xi >> 1: no spreading, relativistic reverse shock, Gamma34 = (phi/2)^1/2,
               Gamma3 = Gamma4 (2 phi)^-1/2, dN3/N_ej = 2 phi dx/(xi + x)      -> FINITE injection
               with phi^2 = (3-k)(xi + x) x^(2-k), x = R/R_dec
  stratified   free ejecta rho_e ~ t^-3 Gamma^-n, N(>Gamma) ~ Gamma^-kej, kej = n + 1, between
               Gamma_max and Gamma_min; NS06 self-similar jump: gamma_e = (m+1)^1/2 Gamma_RS at the
               reverse shock, m = (6-2k)/(n+2), x = gamma_2/Gamma_RS root of NS06 Eq. 14 (hat
               gamma = 5/3 in the shocked ejecta), Gamma3 = x Gamma_RS, Gamma34 = (y + 1/y)/2,
               y = (m+1)^1/2/x, CONSTANT; radius from energy conservation (one zone,
               E(>gamma_e) = 4 pi/(3-k) rho R^3 c^2 Gamma3^2)               -> CONTINUOUS injection
  after the crossing: thin = cold shocked shell spreading at its sound speed (hat gamma 5/3,
  Gamma ~ R^-g, g = 2 for k = 0, Kobayashi & Sari 2000; g = 1 for k = 2, Zou+2005);
  thick and stratified = Blandford-McKee element of the same k.

THE ELECTRONS: one zone, common pressure e'_3 = 4 Gamma3^2 rho c^2 during the crossing (the
shocked medium's), each element compressed adiabatically, B'^2 = 8 pi eps_B e'; injection with
gamma_m,i = eps_e (Gamma34 - 1)(m_p/m_e)(p-2)/(p-1), deep-Newtonian floor sqrt 2 with the
accelerated fraction gamma_m,formal/sqrt 2 (Rahaman+2026), radiating electrons gamma_e >= sqrt 2.
Exact element-by-element distribution (Sect. 2 of Paper I), summed (Eq. intNe_def_gen).

THE EMISSION: comoving L'_nu' with the full kernel (phys_functions.func_R, App. A), one-zone
SSA with the exact kernel (Eq. tau_nu, second form), observed flux of the shell as one
element over its equal arrival time surface (Eq. F_single), Gamma = Gamma3.

THE LITERATURE MODEL (dashed): what Kobayashi (2000), Kobayashi & Zhang (2003), Zou+2005 and
Sadeh+2026 compute: N3 electrons in the SPN98 steady state of the current gamma_m, gamma_c, B'
during the crossing; after it, that distribution cut at gamma_c(R_RS) and shifted
adiabatically (slow cooling: power law gamma_m-gamma_c; fast cooling: all N3 at gamma_c, i.e.
mono-energetic), sharp SPN98 spectrum, line of sight only, no self-absorption.
check_literature() verifies that its post-crossing indices are Kobayashi 2000 Eq. (11) with
his hydrodynamics.

Usage: python3 rs_late_time.py [fig]   (report + figure ../figures/afterglow/rs_late_time.png)
'''

import sys, os, time
import numpy as np

from phys_constants import c_, mp_, me_, e_, sigT_, pi_, norm_R_
from phys_functions import func_R

SQ2 = 2.**0.5
MJY = 1e-26


# --- NS06 jump ----------------------------------------------------------------------------
def ns06_jump(k, kej, hat_gamma=5./3.):
  '''m, x = gamma_2/Gamma_RS (NS06 Eq. 14), Gamma34 at the reverse shock'''
  n = kej - 1.
  m = (6. - 2.*k)/(n + 2.)
  al = hat_gamma/(hat_gamma - 1.)
  s = np.sqrt(m + 1.)
  r = np.roots([al, (al - 2.)*s, -(al - 2.), -al*s])
  x = [z.real for z in r if abs(z.imag) < 1e-10 and 1. - 1e-12 <= z.real <= s + 1e-12][0]
  y = s/x
  return m, x, 0.5*(y + 1./y)


class RSShell:
  def __init__(self, kind, k, E=1e53, Gam4=100., Dt0=1., kej=2.1, Gmax=300., Gmin=30., n0=1.,
               Astar=1., epse=0.3, epsB=1e-2, p=2.5, z=1., dL=2.0e28, nel=400, g_thin=None):
    self.__dict__.update(dict(kind=kind, k=k, E=E, epse=epse, epsB=epsB, p=p, z=z, dL=dL, nel=nel))
    self.rhoR = 5e11*Astar if k == 2 else n0*mp_*(1e17)**k
    self.rho = lambda R: self.rhoR*np.asarray(R, float)**-k
    if kind in ('thin', 'thick'):
      self.Gam4 = Gam4
      self.Rdec = ((3 - k)*E/(4*pi_*Gam4**2*c_**2*self.rhoR))**(1/(3 - k))
      self.xi = Gam4**2*c_*Dt0/(1 + z)/self.Rdec           # Dt0: observed duration (1+z) Delta0/c
      self.Ntot = E/(Gam4*mp_*c_**2)
      self._crossing_shell()
      self.hat = 5./3. if kind == 'thin' else 4./3.
    else:
      self.kej, self.Gmax, self.Gmin = kej, Gmax, Gmin
      self._crossing_strat()
      self.hat = 5./3.
    # after the crossing
    if kind == 'thin':
      self.g = (2. if k == 0 else 1.) if g_thin is None else g_thin
      self.ln_post, self.le_post, self.hat_post = 3*(3 + self.g)/4, 5*(3 + self.g)/4, 5./3.
    else:
      self.g = (7 - 2*k)/2
      self.ln_post, self.le_post, self.hat_post = (13 - 2*k)/2, 2*(13 - 2*k)/3, 4./3.
    self.ex = 4*self.Gx**2*self.rho(self.Rx)*c_**2

  # --- crossing ---------------------------------------------------------------------------
  def _crossing_shell(self):
    k, xi = self.k, self.xi
    x = np.geomspace(1e-4*max(1., xi**0.5), 1e3*max(1., xi**0.5), 20000)
    phi = np.sqrt((3 - k)*(xi + x)*x**(2 - k))
    if self.kind == 'thin':
      G34 = 1. + phi**2/2; Gr = np.ones_like(x); h = 4*phi/3
    else:
      G34 = np.maximum(np.sqrt(phi/2), 1.); Gr = np.minimum((2*phi)**-0.5, 1.); h = 2*phi
    integ = h/(xi + x)
    N = np.concatenate([[0.], np.cumsum(0.5*(integ[1:] + integ[:-1])*np.diff(x))]) + integ[0]*x[0]/1.5
    ix = np.searchsorted(N, 1.)
    self.Rx = np.interp(1., N, x)*self.Rdec
    self.tab = dict(R=x[:ix + 1]*self.Rdec, G=Gr[:ix + 1]*self.Gam4, G34=G34[:ix + 1], N=N[:ix + 1])
    self.R0 = self.tab['R'][0]
    self.Gx = np.interp(self.Rx, self.tab['R'], self.tab['G'])
    self.G34x = np.interp(self.Rx, self.tab['R'], self.tab['G34'])
    self.phix = np.sqrt((3 - k)*(xi + self.Rx/self.Rdec)*(self.Rx/self.Rdec)**(2 - k))

  def _crossing_strat(self):
    k, kej, Gmin, Gmax, E = self.k, self.kej, self.Gmin, self.Gmax, self.E
    self.m, self.xq, G34 = ns06_jump(k, kej)
    rG = self.xq/np.sqrt(self.m + 1.)                      # Gamma3/gamma_e
    gcr = (3 - k)/(1 + kej)
    # E Gmin^(kej-1) ge^(1-kej) = 4 pi/(3-k) rhoR R^(3-k) c^2 (rG ge)^2
    C = E*Gmin**(kej - 1)*(3 - k)/(4*pi_*self.rhoR*c_**2*rG**2)
    self.Rx = (C/Gmin**(1 + kej))**(1/(3 - k))
    self.R0 = self.Rx*(Gmax/Gmin)**(-1/gcr)
    self.Rdec = ((3 - k)*E/(4*pi_*Gmin**2*c_**2*self.rhoR))**(1/(3 - k))
    R = np.geomspace(self.R0, self.Rx, 4000)
    ge = Gmin*(R/self.Rx)**-gcr
    f = lambda G: G**-kej - Gmax**-kej
    self.Ntot = (kej - 1)*E/(kej*Gmin*mp_*c_**2)*(1 - (Gmax/Gmin)**-kej)/(1 - (Gmax/Gmin)**(1 - kej))
    self.tab = dict(R=R, G=rG*ge, G34=np.full_like(R, G34), N=f(ge)/f(Gmin))
    self.Gx, self.G34x, self.gcr, self.xi, self.phix = rG*Gmin, G34, gcr, np.nan, np.nan

  # --- hydrodynamics ------------------------------------------------------------------------
  def Gam(self, R):
    R = np.asarray(R, float)
    return np.where(R < self.Rx, np.interp(R, self.tab['R'], self.tab['G']),
                    self.Gx*(np.maximum(R, self.Rx)/self.Rx)**-self.g)

  def eint(self, R):
    R = np.asarray(R, float)
    return np.where(R < self.Rx, 4*self.Gam(R)**2*self.rho(R)*c_**2,
                    self.ex*(np.maximum(R, self.Rx)/self.Rx)**-self.le_post)

  def B(self, R):
    return np.sqrt(8*pi_*self.epsB*self.eint(R))

  def N3(self, R):
    return self.Ntot*(np.interp(R, self.tab['R'], self.tab['N']) if R < self.Rx else 1.)

  def gm_formal(self, R):
    G34 = np.interp(np.minimum(R, self.Rx), self.tab['R'], self.tab['G34'])
    return self.epse*(G34 - 1)*(mp_/me_)*(self.p - 2)/(self.p - 1)

  def Tr(self, R):
    '''radial time (1+z) int dR/(2 c Gamma^2), the shell starting at R0'''
    Rg = np.geomspace(self.R0, max(np.max(R), self.Rx)*1.0001, 8000)
    integ = 1/(2*c_*self.Gam(Rg)**2)
    T = self.R0*integ[0]/(1 + 2*max(self.gcr if self.kind == 'strat' else 0., 0.)) \
        + np.concatenate([[0.], np.cumsum(0.5*(integ[1:] + integ[:-1])*np.diff(Rg))])
    return (1 + self.z)*np.interp(R, Rg, T)

  def setup(self, Rmax_fac=300., nR=6000):
    Rg = np.geomspace(self.R0, Rmax_fac*self.Rx, nR)
    e_ = self.eint(Rg)
    lna = np.where(Rg < self.Rx, np.log(e_)/(3*self.hat),
                   np.log(self.ex)/(3*self.hat) + (np.log(e_) - np.log(self.ex))/(3*self.hat_post))
    a = np.exp(lna - lna[0])                                 # A(R; R_i) = a(R)/a(R_i)
    tc = 6*pi_*me_*c_/(sigT_*8*pi_*self.epsB*e_)
    integ = a/(tc*c_*self.Gam(Rg))
    self.Rg, self.a = Rg, a
    self.J = np.concatenate([[0.], np.cumsum(0.5*(integ[1:] + integ[:-1])*np.diff(Rg))])
    t0 = self.R0/(c_*self.Gam(self.R0))
    self.tp = t0 + np.concatenate([[0.], np.cumsum(0.5*(1/self.Gam(Rg[1:]) + 1/self.Gam(Rg[:-1]))*np.diff(Rg)/c_)])
    Ri = np.geomspace(self.R0, self.Rx, self.nel + 1)
    self.Ric = np.sqrt(Ri[1:]*Ri[:-1])
    Nf = lambda R: self.Ntot*np.interp(R, self.tab['R'], self.tab['N'])
    dN = Nf(Ri[1:]) - Nf(Ri[:-1])
    gmf = self.gm_formal(self.Ric)
    self.gmi = np.maximum(gmf, SQ2)
    self.dNacc = dN*np.minimum(1., np.maximum(gmf, 0.)/SQ2)
    self.ai = np.exp(np.interp(np.log(self.Ric), np.log(Rg), np.log(a)))
    self.Ji = np.interp(np.log(self.Ric), np.log(Rg), self.J)
    return self

  def gc_now(self, R):
    '''usual gamma_c = t'_c/t' at R'''
    return 6*pi_*me_*c_/(sigT_*self.B(R)**2*np.interp(R, self.Rg, self.tp))

  def A_b(self, R):
    return np.exp(np.interp(np.log(R), np.log(self.Rg), np.log(self.a)))/ \
           np.exp(np.interp(np.log(self.Rx), np.log(self.Rg), np.log(self.a)))

  # --- distributions ---------------------------------------------------------------------
  def dist(self, R, gam):
    '''dN/dgamma of the shell at R (exact element sum), gamma_e >= sqrt 2'''
    aR = np.exp(np.interp(np.log(R), np.log(self.Rg), np.log(self.a)))
    JR = np.interp(np.log(R), np.log(self.Rg), self.J)
    on = self.Ric < R
    A = (aR/self.ai)[on][:, None]
    I = ((JR - self.Ji)/self.ai)[on][:, None]
    den = A - gam[None, :]*I
    ok = den > 0
    d = np.where(ok, den, 1.)
    gi = gam[None, :]/d
    gm = self.gmi[on][:, None]
    n = np.where(ok & (gi >= gm), self.dNacc[on][:, None]*(self.p - 1)*gm**(self.p - 1)*gi**(-self.p)*A/d**2, 0.)
    return np.where(gam >= SQ2, n.sum(0), 0.)

  def lit_state(self, R):
    '''literature model: N, gamma_m, gamma_c (or cut), shift, regime and post-crossing flag'''
    if R <= self.Rx:
      return dict(N=self.N3(R), gm=max(self.gm_formal(R), SQ2), gc=self.gc_now(R), post=False)
    Rx = self.Rx*(1 - 1e-9)
    A = self.A_b(R)
    return dict(N=self.Ntot, gm=max(self.gm_formal(Rx), SQ2)*A, gc=self.gc_now(Rx)*A, post=True)

  # --- emission --------------------------------------------------------------------------
  def dist_usual(self, R, gam):
    '''usual distribution (literature model) as dN/dgamma: SPN98 steady state of N3 during the crossing;
    after it, cut at gamma_c(R_RS) and shifted (fast cooling: all at gamma_c, as a 2%-wide line)'''
    s = self.lit_state(R)
    N, gm, gc, p = s['N'], s['gm'], s['gc'], self.p
    if s['post']:
      if s['gc'] < SQ2*self.A_b(R)*(1 + 1e-9) and s['gc'] < s['gm']:
        return np.zeros_like(gam)
      if gc <= gm:
        w = 0.02
        return N*np.exp(-0.5*(np.log(gam/gc)/w)**2)/(np.sqrt(2*pi_)*w*gam)
      return np.where((gam >= gm) & (gam <= gc), N*(p - 1)*gm**(p - 1)*gam**(-p), 0.)
    if gc < gm:
      return np.where(gam < gc, 0., np.where(gam < gm, N*gc/gam**2, N*gc*gm**(p - 1)*gam**(-p - 1)))
    return np.where(gam < gm, 0., np.where(gam < gc, N*(p - 1)*gm**(p - 1)*gam**(-p),
                                           N*(p - 1)*gm**(p - 1)*gc*gam**(-p - 1)))

  def spectrum(self, R, nup, gam, usual=False):
    '''optically thin L'_nu' (full kernel) and one-zone SSA optical depth'''
    n = self.dist_usual(R, gam) if usual else self.dist(R, gam)
    Bp = float(self.B(R))
    nuB = e_*Bp/(2*pi_*me_*c_)
    pre = sigT_*c_*Bp**2/(9*pi_*nuB*norm_R_)                  # P'_nu' = pre R(x)
    lg = np.log(gam)
    x = (2./3.)*nup[:, None]/(gam[None, :]**2*nuB)
    Rx = func_R(x)
    dlnR = (np.log(func_R(x*1.01) + 1e-300) - np.log(func_R(x/1.01) + 1e-300))/(2*np.log(1.01))
    w = n*gam                                                  # dN/dln gamma
    L = pre*np.trapezoid(w[None, :]*Rx, lg, axis=1)
    # tau = 1/(8 pi m_e nu'^2 4 pi R^2) int (n/gam^2) d[gam^2 p']/dgam dgam,
    # d[gam^2 p']/dgam = 2 gam pre R (1 - dlnR/dlnx)
    integ = (n/gam**2)*2*gam*pre*Rx*(1 - dlnR)*gam
    tau = np.trapezoid(integ, lg, axis=1)/(8*pi_*me_*nup**2*4*pi_*R**2)
    return L, tau, n

  def flux_grid(self, nR=160, nnu=220, nu_lo=1e5, nu_hi=1e20, Rmax_fac=200.):
    self.RE = np.geomspace(self.R0*1.02, Rmax_fac*self.Rx, nR)
    self.nup = np.geomspace(nu_lo, nu_hi, nnu)
    gam = np.geomspace(1.0, 1e9, 2500)
    self.Lt, self.Le, self.Ll, self.Lu, self.Lue = (np.zeros((nR, nnu)) for _ in range(5))
    for i, R in enumerate(self.RE):
      L, tau, _ = self.spectrum(R, self.nup, gam)
      self.Lt[i] = L
      self.Le[i] = L*np.where(tau > 1e-6, -np.expm1(-tau)/np.maximum(tau, 1e-300), 1.)
      self.Ll[i] = self.L_lit(R, self.nup)
      L, tau, _ = self.spectrum(R, self.nup, gam, usual=True)
      self.Lu[i] = L
      self.Lue[i] = L*np.where(tau > 1e-6, -np.expm1(-tau)/np.maximum(tau, 1e-300), 1.)
    self.TrE = self.Tr(self.RE)
    return self

  def L_lit(self, R, nup):
    '''literature comoving spectrum: sharp SPN98 broken power law of the literature state'''
    s = self.lit_state(R)
    Bp = float(self.B(R))
    nuB = e_*Bp/(2*pi_*me_*c_)
    Pmax = me_*c_**2*sigT_*Bp/(3*e_)
    num, nuc = s['gm']**2*nuB, s['gc']**2*nuB
    p = self.p
    L = np.zeros_like(nup)
    if s['post'] and s['gc'] < SQ2*self.A_b(R)*(1 + 1e-9) and s['gc'] < s['gm']:
      return np.zeros_like(nup)       # gamma_c(R_RS) < sqrt 2: no radiating electron left (undefined there)
    if s['post']:
      if nuc <= num:                  # fast cooling at the crossing: mono-energetic at gamma_c
        L = np.where(nup < nuc, (nup/nuc)**(1/3), 0.)
      else:
        L = np.where(nup < num, (nup/num)**(1/3), np.where(nup < nuc, (nup/num)**(-(p - 1)/2), 0.))
    elif nuc < num:
      L = np.where(nup < nuc, (nup/nuc)**(1/3), np.where(nup < num, (nup/nuc)**-0.5,
                   (num/nuc)**-0.5*(nup/num)**(-p/2)))
    else:
      L = np.where(nup < num, (nup/num)**(1/3), np.where(nup < nuc, (nup/num)**(-(p - 1)/2),
                   (nuc/num)**(-(p - 1)/2)*(nup/nuc)**(-p/2)))
    return s['N']*Pmax*L

  def Fnu(self, T, nu, which='ssa'):
    '''observed flux (mJy). 'ssa', 'thin': shell as one element over its EATS (Eq. F_single);
    'lit': literature, line of sight only; 'litE': literature spectrum over the EATS; 'us', 'us_ssa': usual
    distribution with the same kernel, SSA and EATS as ours (attributes the differences to the old electrons).'''
    nu = np.atleast_1d(nu)
    lnu = np.log(self.nup)
    if which == 'lit':
      R = np.exp(np.interp(T, self.TrE, np.log(self.RE)))
      G = float(self.Gam(R))
      Lr = np.array([np.exp(np.interp(np.log(v*(1 + self.z)/(2*G)), lnu, np.log(self.L_lit(R, self.nup) + 1e-300),
                                      left=-690, right=-690)) for v in nu])
      return (1 + self.z)*G*Lr/(4*pi_*self.dL**2)/MJY
    L = {'ssa': self.Le, 'thin': self.Lt, 'litE': self.Ll, 'us': self.Lu, 'us_ssa': self.Lue}[which]
    R, G = self.RE, self.Gam(self.RE)
    Tth = (1 + self.z)*R/(2*c_*G**2)
    Tt = 1 + (T - self.TrE)/Tth
    ok = Tt >= 1
    dl = 2*G/np.where(ok, Tt, 1.)
    lL = np.log(L + 1e-300)
    out = np.zeros(len(nu))
    for j, v in enumerate(nu):
      nupv = np.log(v*(1 + self.z)/dl)
      Lv = np.array([np.exp(np.interp(nupv[i], lnu, lL[i], left=-690, right=-690)) for i in range(len(R))])
      out[j] = np.trapezoid(np.where(ok, dl**2/(G*R)*Lv, 0.), R)
    return (1 + self.z)/(8*pi_*self.dL**2)*out/MJY


# --- cases --------------------------------------------------------------------------------
# Media of the observed radio reverse shocks: n ~ 5e-5 - 1e-3 cm^-3 (GRBs 160625B, 160509A, 161219B; Alexander+17,
# Laskar+16, +18a) and A_* ~ 3e-3 (GRB 130427A; Laskar+13, Perley+14).
# eps_B ~ 0.1: fitted eps_B = 0.2 for GRB 130427A (Laskar+13), R_B = eps_B,RS/eps_B,FS ~ 1-100 (Laskar+16, +18a, +19).
COMMON = dict(E=1e53, p=2.5, epse=0.3, epsB=0.1, z=1., n0=1e-3, Astar=3e-3)
EJECTA = {'thin': dict(Gam4=300., Dt0=2.), 'thick': dict(Gam4=300., Dt0=4000.),   # same ejecta, thin vs thick by the burst duration
          'strat': dict(kej=2.1, Gmax=300., Gmin=30.)}
KINDS = ('thin', 'thick', 'strat')
TITLES = {'thin': 'Thin shell', 'thick': 'Thick shell', 'strat': 'Stratified ejecta'}
DAY = 86400.
T_OBS = np.array([0.1, 1., 10.])*DAY   # times of the observed radio-mm reverse shocks (0.1-10 d)


def build(kind, k, **kw):
  return RSShell(kind, k, **{**COMMON, **EJECTA[kind], **kw}).setup()


def times(m):
  TRS = float(m.Tr(m.Rx))
  return TRS, T_OBS


def gamma_a(m, R=None):
  '''one-zone tau = 1 at R (default: the crossing), full kernel: gamma_a = (nu'_a/1.26 nu'_B)^1/2 and
  the observed nu_a = 2 Gamma3 nu'_a/(1+z)'''
  R = m.Rx*(1 - 1e-9) if R is None else R
  nuB = e_*float(m.B(R))/(2*pi_*me_*c_)
  nup = np.geomspace(1.3*nuB, 1e12*nuB, 800)
  _, tau, _ = m.spectrum(R, nup, np.geomspace(1.0, 1e9, 2500))
  i = np.where(tau >= 1)[0]
  nua = float(np.exp(np.interp(0., -np.log(tau[i[-1]:i[-1] + 2]), np.log(nup[i[-1]:i[-1] + 2])))) if len(i) else np.nan
  return np.sqrt(nua/(1.26*nuB)), 2*float(m.Gam(R))*nua/(1 + m.z)


def check_literature():
  '''post-crossing indices of the literature model vs Kobayashi 2000 Eq. (11) (thick, k=0,
  BM element, hat gamma 4/3): nu_m ~ t^-73/48, F_max ~ t^-47/48 on the line of sight.'''
  m = build('thick', 0.)
  R = m.Rx*np.array([3., 6.])
  T = m.Tr(R)
  num = [m.lit_state(r)['gm']**2*float(m.B(r))*float(m.Gam(r)) for r in R]
  Fm = [m.Ntot*float(m.B(r))*float(m.Gam(r)) for r in R]
  sl = lambda y: np.log(y[1]/y[0])/np.log(T[1]/T[0])
  print(f'literature model, thick k=0: dln nu_m/dln T = {sl(num):.4f} (K00: {-73/48:.4f}), '
        f'dln F_max/dln T = {sl(Fm):.4f} (K00: {-47/48:.4f})')


def report(models=None):
  print('NS06 jump (hat gamma = 5/3): m, x = gamma_2/Gamma_RS, Gamma34 - 1')
  for k in (0., 2.):
    for kej in (2.1, 3., 3.6):
      mm, xq, G34 = ns06_jump(k, kej)
      print(f'  k={k:.0f} kej={kej}: m={mm:.3f} x={xq:.4f} Gamma34-1={G34 - 1:.4f}')
  check_literature()
  models = models or {(kd, k): build(kd, k) for k in (0., 2.) for kd in KINDS}
  print('case        xi      R_RS/Rdec  R_RS[cm]   T_RS[s]  Gam3     G34-1   gm_i   gc     regime  gamma_a  nu_a,obs[Hz]  R_dec[cm]')
  for (kd, k), m in models.items():
    TRS = float(m.Tr(m.Rx))
    gm, gc = max(m.gm_formal(m.Rx*(1 - 1e-9)), SQ2), m.gc_now(m.Rx*(1 - 1e-9))
    print(f'{kd:6s} k={k:.0f} {m.xi:8.3g} {m.Rx/m.Rdec:8.3f} {m.Rx:9.2e} {TRS:8.1f} {m.Gx:7.1f} '
          f'{m.G34x - 1:7.3f} {gm:7.1f} {gc:8.3g}  {"FC" if gc < gm else "SC"}  {gamma_a(m)[0]:6.1f}  {gamma_a(m)[1]:9.2g}  {m.Rdec:9.2g}')
  return models


BANDS = (1.4e9, 6e9, 1.5e10, 1e11, 2.3e11)


def numbers(m):
  '''per evaluation time: peak (nu, F) of the three curves, fluxes in BANDS, decades over which the
  optically thin index lies in [0, 0.3] (broad maximum), and the 100 GHz light-curve slopes'''
  if not hasattr(m, 'Le'):
    m.flux_grid()
  TRS, Ts = times(m)
  nu = np.geomspace(1e7, 1e18, 440)
  out = []
  for T in Ts:
    F, Ft, Fl, FlE, Fu, Fus = (m.Fnu(T, nu, w) for w in ('ssa', 'thin', 'lit', 'litE', 'us', 'us_ssa'))
    a = np.gradient(np.log(Ft + 1e-300), np.log(nu))
    flat = (a >= 0.) & (a <= 0.3) & (Ft > 1e-6*Ft.max())
    def ix(A, f1, f2):
      ip = A.argmax()
      return float(np.log(np.interp(np.log(nu[ip]*f2), np.log(nu), A)/np.interp(np.log(nu[ip]*f1), np.log(nu), A))/np.log(f2/f1))
    band = lambda A: [float(np.exp(np.interp(np.log(v), np.log(nu), np.log(A + 1e-300)))) for v in BANDS]
    out.append(dict(T=T, TT=T/TRS, pk=(nu[F.argmax()], F.max()), pkt=(nu[Ft.argmax()], Ft.max()),
                    a_lo=ix(F, 0.1, 0.3), a_hi=ix(F, 3., 10.), u_lo=ix(Fus, 0.1, 0.3), u_hi=ix(Fus, 3., 10.),
                    pku=(nu[Fus.argmax()], Fus.max()), pkut=(nu[Fu.argmax()], Fu.max()), Fus=band(Fus), Fu=band(Fu), pkl=(nu[Fl.argmax()], Fl.max()), pklE=(nu[FlE.argmax()], FlE.max()), FlE=band(FlE), F=band(F), Ft=band(Ft), Fl=band(Fl),
                    flat=flat.sum()*np.log10(nu[1]/nu[0]), nu_flat=(nu[flat].min() if flat.any() else np.nan,
                                                                   nu[flat].max() if flat.any() else np.nan)))
  sl = lambda w, v, T1, T2: np.log(max(m.Fnu(T2, [v], w)[0], 1e-300)/max(m.Fnu(T1, [v], w)[0], 1e-300))/np.log(T2/T1)
  slopes = {(w, v, i): sl(w, v, Ts[i], Ts[i + 1]) for w in ('ssa', 'thin', 'lit', 'us_ssa') for v in (6e9, 1e11) for i in (0, 1)}
  return out, slopes


def figure(models, out):
  '''optically thin flux solid (main result), with self-absorption dotted, usual model dashed;
  local index of the optically thin flux below'''
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  from matplotlib.lines import Line2D
  cols = ['#2a78d6', '#eb6834', '#3a9b5c']
  fig, axes = plt.subplots(4, 3, figsize=(7.2, 6.6), sharex=True,
                           gridspec_kw=dict(height_ratios=[2.2, 1, 2.2, 1], hspace=0.08, wspace=0.06))
  nu = np.geomspace(1e7, 1e17, 200)
  for row, k in enumerate((0., 2.)):
    top = 0.
    for col, kd in enumerate(KINDS):
      m = models[(kd, k)]
      if not hasattr(m, 'Le'):
        m.flux_grid()
      TRS, Ts = times(m)
      ax, axi = axes[2*row, col], axes[2*row + 1, col]
      for c, T in zip(cols, Ts):
        F = m.Fnu(T, nu, 'ssa'); Ft = m.Fnu(T, nu, 'thin'); Fl = m.Fnu(T, nu, 'lit')
        top = max(top, Ft.max())
        ax.loglog(nu, Ft, color=c, lw=1.4, label=f'{T/DAY:g} d')
        ax.loglog(nu, F, color=c, lw=1.1, ls=':')
        if Fl.max() > 0:
          ax.loglog(nu, np.where(Fl > 0, Fl, np.nan), color=c, lw=0.9, ls='--')
        ok = Ft > 1e-8*Ft.max()
        axi.semilogx(nu[ok], np.gradient(np.log(Ft[ok]), np.log(nu[ok])), color=c, lw=1.)
      for a in (ax, axi):
        a.axvspan(1e9, 3e11, color='#e9e8e4', zorder=0, lw=0)
      axi.axhline(1/3, color='#52514e', lw=0.6, ls=':')
      axi.set_ylim(-1.6, 0.7)
      if row == 0:
        ax.set_title(TITLES[kd], fontsize=9)
      ax.text(0.03, 0.95, f'$k={k:.0f}$', transform=ax.transAxes, ha='left', va='top', fontsize=8)
      if row == 0 and col == 0:
        ax.legend(fontsize=6.5, loc='upper right', frameon=False, handlelength=1.5)
      if col == 0:
        ax.set_ylabel(r'$F_\nu$ [mJy]', fontsize=8); axi.set_ylabel('index', fontsize=8)
      else:
        ax.set_yticklabels([]); axi.set_yticklabels([])
    for col in range(3):
      axes[2*row, col].set_ylim(top*1e-5, top*5)
  key = [Line2D([], [], color='k', lw=1.4), Line2D([], [], color='k', lw=1.1, ls=':'),
         Line2D([], [], color='k', lw=0.9, ls='--')]
  fig.legend(key, ['optically thin', 'self-absorbed', 'usual model'], loc='upper center', ncol=3,
             fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, 0.955))
  for ax in axes[-1]:
    ax.set_xlabel(r'$\nu$ [Hz]', fontsize=8)
  for ax in axes.ravel():
    ax.tick_params(labelsize=7)
  fig.subplots_adjust(top=0.88)
  fig.savefig(out, dpi=200, bbox_inches='tight')


def print_numbers(models):
  for (kd, k), m in models.items():
    rows, sl = numbers(m)
    print(f'=== {kd} k={k:.0f}  light-curve slopes 0.1-1 d | 1-10 d: ' + '; '.join(
      f'{v/1e9:g} GHz {w} {sl[(w, v, 0)]:.2f} | {sl[(w, v, 1)]:.2f}' for v in (6e9, 1e11) for w in ('ssa', 'thin', 'lit', 'us_ssa')))
    for r in rows:
      print(f'  T={r["T"]/DAY:5.2g} d T/T_RS={r["TT"]:7.3g}  peak ssa ({r["pk"][0]:.2g} Hz, {r["pk"][1]:.3g} mJy) thin ({r["pkt"][0]:.2g}, {r["pkt"][1]:.3g}) '
            f'[SSA peak index below {r["a_lo"]:.2f}, above {r["a_hi"]:.2f}; usual+SSA peak ({r["pku"][0]:.2g}, {r["pku"][1]:.3g}) index {r["u_lo"]:.2f}, {r["u_hi"]:.2f}; usual thin peak ({r["pkut"][0]:.2g}, {r["pkut"][1]:.3g})] lit ({r["pkl"][0]:.2g}, {r["pkl"][1]:.3g}) litE ({r["pklE"][0]:.2g}, {r["pklE"][1]:.3g});  index 0-0.3 over {r["flat"]:.2f} dec [{r["nu_flat"][0]:.2g}, {r["nu_flat"][1]:.2g}] Hz')
      print('     F(1.4, 6, 15, 100, 230 GHz) ssa ' + ', '.join(f'{x:.2g}' for x in r['F']) + ' | thin ' +
            ', '.join(f'{x:.2g}' for x in r['Ft']) + ' | lit ' + ', '.join(f'{x:.2g}' for x in r['Fl']) +
            ' | litE ' + ', '.join(f'{x:.2g}' for x in r['FlE']) + ' | usual ' + ', '.join(f'{x:.2g}' for x in r['Fu']) +
            ' | usual+SSA ' + ', '.join(f'{x:.2g}' for x in r['Fus']))


if __name__ == '__main__':
  t0 = time.time()
  models = report()
  if len(sys.argv) > 1 and sys.argv[1] == 'num':
    print_numbers(models)
  if len(sys.argv) > 1 and sys.argv[1] == 'fig':
    out = os.path.join(os.path.dirname(__file__), '..', 'figures', 'afterglow', 'rs_late_time.png')
    figure(models, out)
    print('figure', out)
    print_numbers(models)
  print(f'[{time.time() - t0:.0f} s]')
