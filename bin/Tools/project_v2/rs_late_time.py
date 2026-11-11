# -*- coding: utf-8 -*-
# @Author: acharlet

'''
Paper I, Sect. 5 (reworked after the 2026-11-03 meeting): late-time emission of the
reverse-shocked ejecta, for three physical cases applied to k = 0 (uniform medium) and k = 2
(wind), with the full synchrotron kernel and one-zone self-absorption, compared with the
post-crossing models of the literature.

THE THREE CASES, one jump for all (author 2026-11-11): the exact planar reverse-shock solution (planar(): pressure
balance with the Kumar & Granot 2003 adiabatic index on both sides, (Gamma34^2 - 1) n4' = (Gamma3^2 - 1) n1, compression
4 Gamma34, reverse-shock speed from particle flux conservation), with the local Gamma4 and n4'/n1:
  thin shell   xi = Gamma4^2 Delta0/R_dec << 1: the shell spreads, Delta4 = Delta0 + R/Gamma4^2  -> IMPULSIVE injection
  thick shell  xi >> 1: no spreading                                                       -> IMPULSIVE injection
               (n4'/n1 = (Gamma4/phi)^2, phi^2 = (3-k)(xi + x) x^(2-k), x = R/R_dec; Granot 2012's Newtonian and
               relativistic limits are those of planar())
  stratified   free ejecta N(>Gamma) ~ Gamma^-kej between Gamma_max and Gamma_min, R = beta c t; the reverse-shock
               speed sets the ejecta it reaches; Gamma34 tends to NS06's self-similar value   -> CONTINUOUS injection
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



def planar(G4, f):
  """Exact planar reverse-shock solution for a cold shell of proper density n4' = f n1 running into a cold medium
  n1 (2026-11-11, one ingredient for all cases): pressure balance at the contact with the adiabatic index of
  Kumar & Granot (2003), hat gamma = (4 Gamma + 1)/(3 Gamma), for which e' = 4 Gamma (Gamma - 1) n and
  p = (4/3)(Gamma^2 - 1) n on both sides, so (Gamma34^2 - 1) n4' = (Gamma3^2 - 1) n1 (Newtonian limit
  Gamma34 - 1 = Gamma4^2/(2f), Granot 2012 thin shell; relativistic Gamma34 = (Gamma4^2/4f)^1/4, thick shell).
  The compression is n3'/n4' = 4 Gamma34, and particle flux conservation across the reverse shock gives its speed:
  h = Gamma4^2 (beta4 - beta_RS) = Gamma4^2 r Gamma3 (beta4 - beta3)/(r Gamma3 - Gamma4) (4 phi/3 and 2 phi in the two
  limits, with phi^2 = Gamma4^2/f). Returns Gamma3, Gamma34 - 1, h (arrays, G4 scalar or array)."""
  from scipy.optimize import brentq
  G4 = np.broadcast_to(np.asarray(G4, float), np.shape(f)).ravel()
  f = np.asarray(f, float).ravel()
  def g3m1(g4, d):
    '''Gamma3 - 1 for Gamma34 = 1 + d (region 3 slower than region 4), without cancellation'''
    g34 = 1 + d
    return (g4 - g34)**2/((g4*g34 - 1) + np.sqrt((g4**2 - 1)*d*(d + 2)))
  G3 = np.empty_like(f); d34 = np.empty_like(f); dG = np.empty_like(f)
  for j in range(len(f)):
    g4, fj = G4[j], f[j]
    def res(ld):                                                            # unknown: ln(Gamma34 - 1)
      d = np.exp(ld); x = g3m1(g4, d)
      return np.log(d*(d + 2)*fj) - np.log(x*(x + 2))
    ld = brentq(res, np.log(1e-250), np.log((g4 - 1)*(1 - 1e-9)), xtol=1e-13)
    d = np.exp(ld)
    d34[j], G3[j] = d, 1 + g3m1(g4, d)
    dG[j] = np.sqrt((g4**2 - 1)*d*(d + 2)) - g4*d                            # Gamma4 - Gamma3
  b3, b4 = np.sqrt(1 - 1/G3**2), np.sqrt(1 - 1/G4**2)
  db = dG*(G4 + G3)/(G3**2*G4**2)/(b4 + b3)                                 # beta4 - beta3
  r = 4*(1 + d34)
  h = G4**2*r*G3*db/(r*G3 - G4)
  return G3, d34, h


class RSShell:
  def __init__(self, kind, k, E=1e53, Gam4=100., Dt0=1., kej=2.1, Gmax=300., Gmin=30., n0=1.,
               Astar=1., epse=0.3, epsB=1e-2, p=2.5, z=1., dL=2.0e28, nel=400, g_thin=None, field='toroidal',
               sigma4=1e-4, epsB_max=1/3):
    '''field: 'toroidal' (default since 2026-11-11, J. Granot): a toroidal field frozen in the ejecta,
    B' ~ 1/(R Delta') ~ n' R, compressed by the shock with the density, so that B'/(n'R) = K is the same in
    all the elements at all times; B'_i = K n'_i R_i with n'_i = e'_3/[(Gamma34 - 1) m_p c^2], and K set
    by the magnetization sigma4 of the ejecta shocked at R_RS: eps_B,i(R_RS) = 2 sigma4 Gamma34/(Gamma34 - 1);
    'epsB' = B'^2 = 8 pi eps_B e' in every element at all times (common field, homogeneous shell);
    'frozen' = B'^2 = 8 pi eps_B e' at injection only, then frozen in a tangled field, B'^2 ~ n'^4/3'''
    self.__dict__.update(dict(kind=kind, k=k, E=E, epse=epse, epsB=epsB, p=p, z=z, dL=dL, nel=nel, field=field,
                              sigma4=sigma4, epsB_max=epsB_max))
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
    if field == 'toroidal':
      self.epsBx = 2*sigma4*self.G34x/(self.G34x - 1)
      self.K = np.sqrt(8*pi_*self.epsBx*self.ex)/(self.n_inj(self.Rx)*self.Rx)

  # --- crossing ---------------------------------------------------------------------------
  def _crossing_shell(self):
    k, xi = self.k, self.xi
    x = np.geomspace(1e-4*max(1., xi**0.5), 1e3*max(1., xi**0.5), 20000)
    phi = np.sqrt((3 - k)*(xi + x)*x**(2 - k))
    # exact planar solution (one ingredient for all cases, author 2026-11-11), tabulated in phi
    pg = np.geomspace(phi.min(), phi.max(), 500)
    G3g, d34g, hg = planar(self.Gam4, (self.Gam4/pg)**2)
    lphi = np.log(phi)
    G3 = 1 + np.exp(np.interp(lphi, np.log(pg), np.log(G3g - 1)))
    G34 = 1 + np.exp(np.interp(lphi, np.log(pg), np.log(d34g)))
    h = np.exp(np.interp(lphi, np.log(pg), np.log(hg)))
    Gr = G3/self.Gam4
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
    """Stratified ejecta with the same planar solution as the shells (author 2026-11-11): the reverse shock moves
    through freely expanding ejecta, R = beta c t for the ejecta of Lorentz factor Gamma (N(>Gamma) ~ Gamma^-kej
    between Gmin and Gmax), whose proper density at R is n4' = Gamma^2 beta^2 (dN/dGamma)/(4 pi R^3). At each R
    the planar solution with the local Gamma4 and n4'/n1 gives Gamma3, Gamma34 and the reverse-shock speed, which
    sets the ejecta it reaches: d ln Gamma4/d ln R = -Gamma4^2 beta4^2 (beta4 - beta_RS)/beta_RS. Crossing at
    Gamma4 = Gmin. Gamma34 tends to NS06's self-similar value (ns06_jump, kept as a check) to 1-3%: their crossing is
    set by the same density ratio at the shock. (The former radius from E(>Gamma4) = shocked-medium energy put all
    the energy in the shocked medium; a one-zone energy closure with both regions at Gamma3 gives Gamma34 6-10 times
    NS06's: the shocked ejecta are not uniform in the self-similar flow.)"""
    from scipy.integrate import solve_ivp
    k, kej, Gmin, Gmax, E = self.k, self.kej, self.Gmin, self.Gmax, self.E
    self.m, self.xq, self.G34_ns06 = ns06_jump(k, kej)
    fG = lambda G: G**-kej - Gmax**-kej
    self.Ntot = (kej - 1)*E/(kej*Gmin*mp_*c_**2)*(1 - (Gmax/Gmin)**-kej)/(1 - (Gmax/Gmin)**(1 - kej))
    dNdG = lambda G: self.Ntot*kej*G**(-kej - 1)/fG(Gmin)
    def ff(lR, G4):
      R = np.exp(lR)
      return G4**2*(1 - 1/G4**2)*dNdG(G4)/(4*pi_*R**3)/(self.rho(R)/mp_)
    def rhs(lR, y):
      G4 = np.exp(y[0])
      _, _, h = planar(G4, np.array([ff(lR, G4)]))
      b4 = np.sqrt(1 - 1/G4**2); dbs = h[0]/G4**2
      return [-G4**2*b4**2*dbs/(b4 - dbs)]
    Rdec_min = ((3 - k)*E/(4*pi_*Gmax**2*c_**2*self.rhoR))**(1/(3 - k))
    lR0 = np.log(1e-4*Rdec_min)
    while planar(Gmax*(1 - 1e-9), np.array([ff(lR0, Gmax*(1 - 1e-9))]))[2][0] > 1e-4:
      lR0 -= np.log(10.)
    ev = lambda lR, y: y[0] - np.log(Gmin)
    ev.terminal = True
    sol = solve_ivp(rhs, (lR0, lR0 + np.log(1e14)), [np.log(Gmax*(1 - 1e-9))], events=ev, rtol=1e-6, atol=1e-9,
                    dense_output=True, max_step=0.05, first_step=1e-3)
    lRx = sol.t_events[0][0]
    lR = np.linspace(lR0, lRx, 4000)
    G4 = np.minimum(np.exp(sol.sol(lR)[0]), Gmax*(1 - 1e-9))
    G3, d34, _ = planar(G4, np.array([ff(l, g) for l, g in zip(lR, G4)]))
    R = np.exp(lR)
    self.Rx, self.R0 = R[-1], R[0]
    self.Rdec = ((3 - k)*E/(4*pi_*Gmin**2*c_**2*self.rhoR))**(1/(3 - k))
    self.tab = dict(R=R, G=G3, G34=1 + d34, N=fG(G4)/fG(Gmin), G4=G4)
    self.Gx, self.G34x, self.gcr, self.xi, self.phix = G3[-1], 1 + d34[-1], 0., np.nan, np.nan

  # --- hydrodynamics ------------------------------------------------------------------------
  def Gam(self, R):
    R = np.asarray(R, float)
    return np.where(R < self.Rx, np.interp(R, self.tab['R'], self.tab['G']),
                    self.Gx*(np.maximum(R, self.Rx)/self.Rx)**-self.g)

  def eint(self, R):
    R = np.asarray(R, float)
    return np.where(R < self.Rx, 4*self.Gam(R)**2*self.rho(R)*c_**2,
                    self.ex*(np.maximum(R, self.Rx)/self.Rx)**-self.le_post)

  def G34(self, R):
    return np.interp(np.minimum(R, self.Rx), self.tab['R'], self.tab['G34'])

  def n_inj(self, R):
    '''proper density of the ejecta shocked at R (energy per particle (Gamma34 - 1) m_p c^2)'''
    R = np.minimum(np.asarray(R, float), self.Rx)
    return 4*self.Gam(R)**2*self.rho(R)*c_**2/(np.maximum(self.G34(R) - 1, 1e-6)*mp_*c_**2)   # floor: no shock yet

  def Bc(self, R):
    '''field at injection: eps_B of the common pressure, or the compressed toroidal field K n'_i R'''
    if self.field == 'toroidal':
      return self.K*self.n_inj(R)*np.minimum(np.asarray(R, float), self.Rx)
    return np.sqrt(8*pi_*self.epsB*self.eint(R))

  def epsB_inj(self, R):
    return self.Bc(R)**2/(8*pi_*self.eint(np.minimum(R, self.Rx)))

  def _a(self, R):
    return np.exp(np.interp(np.log(R), np.log(self.Rg), np.log(self.a)))

  def B(self, R):
    '''field of the latest element (all elements for field = 'epsB')'''
    if self.field == 'epsB':
      return self.Bc(R)
    R = np.asarray(R, float)
    Rl = np.minimum(R, self.Rx*(1 - 1e-12))
    if self.field == 'toroidal':
      return self.Bc(Rl)*(self._a(R)/self._a(Rl))**3*R/Rl
    return self.Bc(Rl)*(self._a(R)/self._a(Rl))**2

  def b_el(self, R):
    '''B'(R; R_i)/B(R) of the elements injected before R'''
    if self.field == 'epsB':
      return np.ones_like(self.Ric)
    if self.field == 'toroidal':
      return self.Bi*(self._a(R)/self.ai)**3*(R/self.Ric)/float(self.B(R))
    return self.Bi*(self._a(R)/self.ai)**2/float(self.B(R))

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
    if self.field == 'epsB':
      tc = 6*pi_*me_*c_/(sigT_*8*pi_*self.epsB*e_)
      integ = a/(tc*c_*self.Gam(Rg))
    elif self.field == 'toroidal':                           # B'(R; R_i) = Bc(R_i) (a/a_i)^3 R/R_i
      integ = a**7*Rg**2*sigT_/(6*pi_*me_*c_)/(c_*self.Gam(Rg))
    else:                                                    # B'^2(R; R_i) = Bc^2(R_i) (a/a_i)^4
      integ = a**5*sigT_/(6*pi_*me_*c_)/(c_*self.Gam(Rg))
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
    if self.field == 'toroidal':
      # validity of the hydrodynamic framework (author 2026-11-11): the magnetic over thermal pressure behind the
      # reverse shock is 1.5 eps_B,i; the hydrodynamic pressure balance (Gamma34, N3, R_RS) holds while it is <~ 1/2,
      # eps_B,i <= 1/3. Where it is larger, the field of the ejecta supports the contact discontinuity and only a weak
      # reverse shock forms (Granot 2012, Sect. 4): those electrons are not accelerated
      self.cut = float(np.sum(dN*np.minimum(1., np.maximum(gmf, 0.)/SQ2)*(self.epsB_inj(self.Ric) > self.epsB_max)))
      self.dNacc = np.where(self.epsB_inj(self.Ric) <= self.epsB_max, self.dNacc, 0.)
      self.cut /= max(float(np.sum(dN*np.minimum(1., np.maximum(gmf, 0.)/SQ2))), 1e-300)
    self.Nacc_c = np.concatenate([[0.], np.cumsum(self.dNacc)])   # accelerated electrons injected up to Ri
    self.ai = np.exp(np.interp(np.log(self.Ric), np.log(Rg), np.log(a)))
    self.Ji = np.interp(np.log(self.Ric), np.log(Rg), self.J)
    self.Bi = self.Bc(self.Ric)
    self.ci = (1/self.ai if self.field == 'epsB' else                       # I = (J(R) - J_i) c_i
               (self.Bi/(self.ai**3*self.Ric))**2/self.ai if self.field == 'toroidal' else self.Bi**2/self.ai**5)
    return self

  def gc_now(self, R):
    '''usual gamma_c = t'_c/t' at R'''
    return 6*pi_*me_*c_/(sigT_*self.B(R)**2*np.interp(R, self.Rg, self.tp))

  def A_b(self, R):
    return np.exp(np.interp(np.log(R), np.log(self.Rg), np.log(self.a)))/ \
           np.exp(np.interp(np.log(self.Rx), np.log(self.Rg), np.log(self.a)))

  # --- distributions ---------------------------------------------------------------------
  def dist(self, R, gam, wpow=None):
    '''dN/dgamma of the shell at R (exact element sum), gamma_e >= sqrt 2. With elements in their own
    fields b B(R): an electron gamma in b B radiates as gamma b^1/2 in B with b times the power, and absorbs
    with b^3/2 times the weight; dist returns sum_i b_i^wpow n_i(gam/b_i^1/2)/b_i^1/2 (wpow = 1 for the
    emission, 3/2 for the absorption); wpow = None: the true distribution'''
    aR = np.exp(np.interp(np.log(R), np.log(self.Rg), np.log(self.a)))
    JR = np.interp(np.log(R), np.log(self.Rg), self.J)
    on = self.Ric < R
    A = (aR/self.ai)[on][:, None]
    I = ((JR - self.Ji)*self.ci)[on][:, None]
    b = self.b_el(R)[on][:, None] if wpow is not None else np.ones((on.sum(), 1))
    wpow = 0.5 if wpow is None else wpow
    gt = gam[None, :]/np.sqrt(b)                              # true Lorentz factor
    den = A - gt*I
    ok = den > 0
    d = np.where(ok, den, 1.)
    gi = gt/d
    gm = self.gmi[on][:, None]
    n = np.where(ok & (gi >= gm) & (gt >= SQ2),
                 self.dNacc[on][:, None]*(self.p - 1)*gm**(self.p - 1)*gi**(-self.p)*A/d**2*b**(wpow - 0.5), 0.)
    return n.sum(0)

  def Nacc(self, R):
    '''accelerated electrons injected up to R (the electrons of the usual model, for a like-for-like comparison)'''
    Ri = np.geomspace(self.R0, self.Rx, self.nel + 1)
    return float(np.interp(min(R, self.Rx), Ri, self.Nacc_c))

  def lit_state(self, R):
    '''literature model: N, gamma_m, gamma_c (or cut), shift, regime and post-crossing flag'''
    if R <= self.Rx:
      return dict(N=self.Nacc(R), gm=max(self.gm_formal(R), SQ2), gc=self.gc_now(R), post=False)
    Rx = self.Rx*(1 - 1e-9)
    A = self.A_b(R)
    return dict(N=self.Nacc(self.Rx), gm=max(self.gm_formal(Rx), SQ2)*A, gc=self.gc_now(Rx)*A, post=True)

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
    if usual:
      n = na = self.dist_usual(R, gam)
    else:
      n, na = self.dist(R, gam, 1.), self.dist(R, gam, 1.5)
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
    integ = (na/gam**2)*2*gam*pre*Rx*(1 - dlnR)*gam
    tau = np.trapezoid(integ, lg, axis=1)/(8*pi_*me_*nup**2*4*pi_*R**2)
    return L, tau, n

  def flux_grid(self, nR=240, nnu=220, nu_lo=1e5, nu_hi=1e20, Rmax_fac=200.):
    # denser around the crossing, where the spectrum changes fastest (smooth light curves near T_RS)
    self.RE = np.unique(np.concatenate([np.geomspace(self.R0*1.02, Rmax_fac*self.Rx, nR),
                                        np.geomspace(0.3*self.Rx, 3*self.Rx, 80)]))
    nR = len(self.RE)
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
    # EATS integral on a fine radius grid that ends exactly at R_l (T_r(R_l) = T), with ln L interpolated bilinearly
    # in (ln R, ln nu'). Integrating on the coarse table grid with the cut Tt >= 1 made the light curves jump each
    # time R_l crossed a grid point.
    lR, lT = np.log(self.RE), np.log(self.TrE)
    if T <= self.TrE[0]:
      return np.zeros(len(nu))
    Rl = np.exp(min(np.interp(np.log(T), lT, lR), lR[-1]))
    Rf = np.unique(np.concatenate([np.geomspace(self.RE[0], Rl, 700),
                                   Rl*(1 - np.geomspace(1e-5, 1 - self.RE[0]/Rl, 300))]))
    Rf = Rf[(Rf >= self.RE[0]) & (Rf <= Rl)]
    lRf = np.log(Rf)
    G = self.Gam(Rf)
    Trf = np.exp(np.interp(lRf, lR, lT))
    Tth = (1 + self.z)*Rf/(2*c_*G**2)
    dl = 2*G/np.maximum(1 + (T - Trf)/Tth, 1.)
    lL = np.log(L + 1e-300)
    iR = np.clip(np.searchsorted(lR, lRf) - 1, 0, len(lR) - 2)
    wR = (lRf - lR[iR])/(lR[iR + 1] - lR[iR])
    dlnu = lnu[1] - lnu[0]
    out = np.zeros(len(nu))
    for j, v in enumerate(nu):
      x = (np.log(v*(1 + self.z)/dl) - lnu[0])/dlnu
      inside = (x >= 0) & (x <= len(lnu) - 1)
      iN = np.clip(np.floor(x).astype(int), 0, len(lnu) - 2)
      wN = np.clip(x - iN, 0., 1.)
      lv = ((1 - wR)*((1 - wN)*lL[iR, iN] + wN*lL[iR, iN + 1]) +
            wR*((1 - wN)*lL[iR + 1, iN] + wN*lL[iR + 1, iN + 1]))
      out[j] = np.trapezoid(dl**2/(G*Rf)*np.where(inside, np.exp(lv), 0.), Rf)
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
  '''post-crossing indices of the literature model (line of sight) vs the published ones:
  thick k=0: Kobayashi (2000) Eq. 11, nu_m ~ T^-73/48, F_max ~ T^-47/48 (BM element);
  thick k=2: Kobayashi & Zhang (2003), Zou+ (2005) Table 1, nu_m ~ T^-15/8, F_max ~ T^-9/8 (BM element);
  thin k=0 and k=2: Kobayashi & Sari (2000), Zou+ (2005) Table 1, cold shell spreading at its sound
  speed with hat gamma 4/3: nu_m ~ T^-3(8+5g)/7(1+2g), F_max ~ T^-(12+11g)/7(1+2g); the run itself uses
  hat gamma 5/3 for the thin shell after the crossing (printed too).'''
  def slopes(m):
    R = m.Rx*np.array([50., 100.])
    T = m.Tr(R)
    num = [m.lit_state(r)['gm']**2*float(m.B(r))*float(m.Gam(r)) for r in R]
    Fm = [m.Ntot*float(m.B(r))*float(m.Gam(r)) for r in R]
    sl = lambda y: np.log(y[1]/y[0])/np.log(T[1]/T[0])
    return sl(num), sl(Fm)
  for kd, k, ref, pub in (('thick', 0., 'K00 Eq. 11', (-73/48, -47/48)),
                          ('thick', 2., 'KZ03, Zou05', (-15/8, -9/8))):
    a, b = slopes(build(kd, k))
    print(f'literature model, {kd} k={k:.0f}: nu_m T^{a:.4f} ({ref}: {pub[0]:.4f}), F_max T^{b:.4f} ({pub[1]:.4f})')
  for k in (0., 2.):
    m = build('thin', k)
    g = m.g
    a5, b5 = slopes(m)
    m.ln_post, m.le_post, m.hat_post = 6*(3 + g)/7, 8*(3 + g)/7, 4./3.
    m.setup()
    a, b = slopes(m)
    print(f'literature model, thin k={k:.0f} (g={g:g}), hat gamma 4/3: nu_m T^{a:.4f} (KS00, Zou05: '
          f'{-3*(8 + 5*g)/(7*(1 + 2*g)):.4f}), F_max T^{b:.4f} ({-(12 + 11*g)/(7*(1 + 2*g)):.4f}); '
          f'with 5/3 (run): nu_m T^{a5:.4f}, F_max T^{b5:.4f}')


def report(models=None):
  print('NS06 jump (hat gamma = 5/3): m, x = gamma_2/Gamma_RS, Gamma34 - 1')
  for k in (0., 2.):
    for kej in (2.1, 3., 3.6):
      mm, xq, G34 = ns06_jump(k, kej)
      print(f'  k={k:.0f} kej={kej}: m={mm:.3f} x={xq:.4f} Gamma34-1={G34 - 1:.4f}')
  check_literature()
  models = models or {(kd, k): build(kd, k) for k in (0., 2.) for kd in KINDS}
  print('case        xi      R_RS/Rdec  R_RS[cm]   T_RS[s]  Gam3     G34-1   gm_i   gc     regime  gamma_a  nu_a,obs[Hz]  R_dec[cm]  epsB(R_RS) epsB(N=10%) cut')
  for (kd, k), m in models.items():
    TRS = float(m.Tr(m.Rx))
    gm, gc = max(m.gm_formal(m.Rx*(1 - 1e-9)), SQ2), m.gc_now(m.Rx*(1 - 1e-9))
    print(f'{kd:6s} k={k:.0f} {m.xi:8.3g} {m.Rx/m.Rdec:8.3f} {m.Rx:9.2e} {TRS:8.1f} {m.Gx:7.1f} '
          f'{m.G34x - 1:7.3f} {gm:7.1f} {gc:8.3g}  {"FC" if gc < gm else "SC"}  {gamma_a(m)[0]:6.1f}  {gamma_a(m)[1]:9.2g}  {m.Rdec:9.2g}'
          f'  {float(m.epsB_inj(m.Rx*(1 - 1e-9))):9.3g} {float(m.epsB_inj(np.interp(.1, m.tab["N"], m.tab["R"]))):9.3g} {getattr(m, "cut", 0.):6.3f}')
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
                           gridspec_kw=dict(height_ratios=[2.2, 1, 2.2, 1], hspace=0.08, wspace=0.27))
  nu = np.geomspace(1e7, 1e17, 200)
  for row, k in enumerate((0., 2.)):
    for col, kd in enumerate(KINDS):
      m = models[(kd, k)]
      if not hasattr(m, 'Le'):
        m.flux_grid()
      TRS, Ts = times(m)
      ax, axi = axes[2*row, col], axes[2*row + 1, col]
      curves = [(c, m.Fnu(T, nu, 'thin'), m.Fnu(T, nu, 'ssa'), m.Fnu(T, nu, 'lit')) for c, T in zip(cols, Ts)]
      # per-panel range: from 3x the highest curve down to 30x below the lowest self-absorbed peak,
      # so that every peak (optically thin, self-absorbed, usual) and the cutoffs show
      hi = 3*max(max(Ft.max(), Fl.max()) for _, Ft, _, Fl in curves)
      lo = max(min(F.max() for _, _, F, _ in curves)/30, hi*1e-9)
      for c, Ft, F, Fl in curves:
        ax.loglog(nu, Ft, color=c, lw=1.4)
        ax.loglog(nu, F, color=c, lw=1.1, ls=':')
        if Fl.max() > 0:
          ax.loglog(nu, np.where(Fl > 0, Fl, np.nan), color=c, lw=0.9, ls='--')
        ok = Ft > 1e-6*Ft.max()           # index down to 6 decades below the peak (numerical noise beyond)
        idx = np.gradient(np.log(Ft[ok]), np.log(nu[ok]))
        axi.semilogx(nu[ok], idx, color=c, lw=1.)
      ax.set_ylim(lo, hi)
      for a in (ax, axi):
        a.axvspan(1e9, 3e11, color='#e9e8e4', zorder=0, lw=0)
      for v, lab in ((1/3, r'$1/3$'), (-(m.p - 1)/2, r'$-(p-1)/2$'), (-m.p/2, r'$-p/2$')):
        axi.axhline(v, color='#52514e', lw=0.6, ls=':', zorder=1)
        if col == 2:
          axi.text(1.02, v, lab, transform=axi.get_yaxis_transform(), ha='left', va='center', fontsize=6.5,
                   color='#52514e')
      axi.set_ylim(-1.85, 0.5)
      axi.set_yticks([-1.5, -1, -0.5, 0])
      axi.set_yticklabels(['', '$-1$', '', '$0$'])
      if row == 0:
        ax.set_title(TITLES[kd], fontsize=9)
      ax.text(0.97, 0.95, f'$k={k:.0f}$', transform=ax.transAxes, ha='right', va='top', fontsize=8)
      if col == 0:
        ax.set_ylabel(r'$F_\nu$ [mJy]', fontsize=8); axi.set_ylabel('index', fontsize=8)
      else:
        axi.set_yticklabels([])
  key = [Line2D([], [], color='k', lw=1.4), Line2D([], [], color='k', lw=1.1, ls=':'),
         Line2D([], [], color='k', lw=0.9, ls='--')] + [Line2D([], [], color=c, lw=1.4) for c in cols]
  labs = ['optically thin', 'self-absorbed', 'usual model'] + [f'$T={t/DAY:g}$ d' for t in T_OBS]
  fig.legend(key, labs, loc='upper center', ncol=6, fontsize=7, frameon=False, handlelength=1.8,
             columnspacing=1.2, bbox_to_anchor=(0.5, 0.955))
  for ax in axes[-1]:
    ax.set_xlabel(r'$\nu$ [Hz]', fontsize=8)
  for ax in axes.ravel():
    ax.tick_params(labelsize=7)
  fig.subplots_adjust(top=0.88)
  fig.savefig(out, dpi=400, bbox_inches='tight')


LC_NU = (6e9, 1e11, 2.3e11)
LC_T = np.geomspace(0.01, 30., 90)*DAY


def lightcurves(models, out):
  '''F_nu(T) at LC_NU: ours with self-absorption (solid), optically thin (thin), usual distribution with the same
  kernel, SSA and EATS (dashed), literature line of sight (dotted)'''
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  from matplotlib.lines import Line2D
  cols = ['#2a78d6', '#eb6834', '#3a9b5c']
  fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.8), sharex=True, gridspec_kw=dict(hspace=0.08, wspace=0.27))
  for row, k in enumerate((0., 2.)):
    for col, kd in enumerate(KINDS):
      m = models[(kd, k)]
      if not hasattr(m, 'Le'):
        m.flux_grid()
      ax = axes[row, col]
      TRS = float(m.Tr(m.Rx))
      hi = 0.
      for c, v in zip(cols, LC_NU):
        F = np.array([m.Fnu(T, [v], 'ssa')[0] for T in LC_T])
        Fu = np.array([m.Fnu(T, [v], 'us_ssa')[0] for T in LC_T])
        Fl = np.array([m.Fnu(T, [v], 'lit')[0] for T in LC_T])
        ax.loglog(LC_T/DAY, F, color=c, lw=1.4)
        ax.loglog(LC_T/DAY, Fu, color=c, lw=0.9, ls='--')
        ax.loglog(LC_T/DAY, np.where(Fl > 0, Fl, np.nan), color=c, lw=0.8, ls=':')
        hi = max(hi, F.max(), Fu.max())
      ax.set_ylim(hi*1e-5, hi*3)
      if TRS/DAY > LC_T[0]/DAY:
        ax.axvline(TRS/DAY, color='#52514e', lw=0.6, ls='-.')
      if row == 0:
        ax.set_title(TITLES[kd], fontsize=9)
      ax.text(0.97, 0.95, f'$k={k:.0f}$', transform=ax.transAxes, ha='right', va='top', fontsize=8)
      if col == 0:
        ax.set_ylabel(r'$F_\nu$ [mJy]', fontsize=8)
      ax.tick_params(labelsize=7)
  for ax in axes[-1]:
    ax.set_xlabel(r'$T$ [d]', fontsize=8)
  key = [Line2D([], [], color='k', lw=1.4), Line2D([], [], color='k', lw=0.9, ls='--'),
         Line2D([], [], color='k', lw=0.8, ls=':'), Line2D([], [], color='#52514e', lw=0.6, ls='-.')] + \
        [Line2D([], [], color=c, lw=1.4) for c in cols]
  labs = ['this work', 'usual distribution', 'literature model', r'$T_{\rm RS}$'] + \
         [f'{v/1e9:g} GHz' for v in LC_NU]
  fig.legend(key, labs, loc='upper center', ncol=7, fontsize=7, frameon=False, handlelength=1.8,
             columnspacing=1.0, bbox_to_anchor=(0.5, 0.99))
  fig.subplots_adjust(top=0.88)
  fig.savefig(out, dpi=300, bbox_inches='tight')


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
    lightcurves(models, out.replace('rs_late_time.png', 'rs_lightcurves.png'))
    print('light curves', out.replace('rs_late_time.png', 'rs_lightcurves.png'))
    print_numbers(models)
  print(f'[{time.time() - t0:.0f} s]')
