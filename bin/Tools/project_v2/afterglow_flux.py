# -*- coding: utf-8 -*-
# @Author: acharlet

'''
Synchrotron flux of a shell defined by the power laws of Paper I (Sects. 2-4), by direct
quadrature: no hydro data, no cell frames, no MyEnv.

THE MODEL. Electrons are injected at the shock, dN_inj = Ndot(R_i) dR_i, with the injected
state set by an INJECTION object (Gamma_i, B'_i, gma_m,i, gma_M,i, gbar_c,i and the arrival
offset tau_i, all functions of R_i). Every element then follows the self-similar laws of
cooling_shell_integrated.ShellLaws (lam_n, lam_B, g only; r = R/R_i):

    Gamma = Gamma_i r^-g,   B' = B'_i r^-lam_B,   A = r^(-lam_n/3),
    I = (1 - r^-Lam)/(Lam gbar_c,i),   gma(gma_i) = A gma_i/(1 + gma_i I).

gbar_c,i here is RADIAL, 6 pi m_e c^2 Gamma_i/(sigma_T B_i'^2 R_i): the article's comoving
gbar_c,i (Eq. eqn:plaws_R) is (1+g) times it, and Lam = (1+g) s, so I is the article's to
rounding (checked against a quadrature of the physical B'(R), Gamma(R): 2e-9).

THE EMISSIVITY is integrated over the INJECTED Lorentz factor, not over gma: the electron
count is K gma_i^-p dgma_i exactly, the fast-cooled pile-up at A/I needs no resolution, and
the (1 - gma I/A)^(p-2) factor of the cooled shape (which cancels catastrophically when the
burn-off b_syn is tiny) never appears. Per electron,

    p'_nu'(gma) = sigma_T c B'^2/(9 pi nu'_B) R(x)/norm_R,   x = 2 nu'/(3 gma^2 nu'_B),

the pitch-angle-averaged synchrotron function of radiation_cooling.syn_emiss_exact, which
integrates to (4/3) sigma_T c gma^2 U_B. Electrons whose gma falls below 1 are dropped (the
bound truncation of the simulation path).

THE FLUX. Each element at R is a thin spherical shell; with T = (1+z)(t - R mu/c) the
mu-integral gives, in the ultra-relativistic limit (delta_0 = 2 Gamma, dt' = dR/(c Gamma)),

    F_nu(T) = (1+z)/(2 pi d_L^2) int dR_i Ndot int dR (Gamma/R) tT^-2 P'_nu'((1+z) nu tT/(2 Gamma)),
    tT = 1 + (T - T_on)/T_th,   T_on = (1+z) tau(R; R_i),   T_th = (1+z) R/(2 Gamma^2 c),
    tau(R; R_i) = t - R/c = tau_i + R_i/(2 Gamma_i^2 c) (r^(1+2g) - 1)/(1+2g),

with P'_nu' the comoving power per unit frequency of one element's electrons (normalised to
one electron). An element emits at T only up to R_on(T; R_i), where T_on = T: closed form.
The same approximation is used everywhere (BM itself is an ultra-relativistic solution), so
the energy identity check_energy tests holds exactly; the physical normalisation is tested
against GS02 by check_gs02.

QUADRATURE. Outer, s = ln(R_l/R_i), R_l the line-of-sight shock radius at T, on panels
refined toward s = 0. Inner, w = ln(r - 1): the fast-cooling emission above nu_c comes from
r - 1 ~ gbar_c,i/gma << 1, so the nodes must reach down to the cooling time of gma_M,i; the
piece below delta_min is added as one rectangle. gma_i: composite Gauss-Legendre in ln gma_i.

VALIDATION (main): the BM injection's shock-front indices equal bm_forward_shock(k)'s; one
element's int dT int dnu F_nu equals (1+z) E_rad/(4 pi d_L^2) to 1e-4; and against GS02's
smoothed spectra (Table 2) the flux agrees to 0.97-1.04 (k=0, slow cooling, 1 day),
0.99-1.06 (k=0, fast, 1e-3 day) and 0.96-1.09 (k=2, slow, 10 days) from 2 decades below the
first break to 2 above the second -- the size of GS02's own fit errors (0.5-7% per break).
The quadrature is converged: doubling any grid moves those ratios by < 2e-4. gs02_k0 and
gs02_k2_slow transcribe Table 2 as printed (e.g. F_9 ~ t^+1/2); a pdftotext copy drops
every minus sign, so a transcription from one must be checked against the neighbouring
break, F_b2 = F_b1 (nu_b2/nu_b1)^beta.

Not included: synchrotron self-absorption, inverse Compton, a jet, the coasting phase
(injection starts at R_0 on the self-similar solution), and the reverse shock.
'''

from dataclasses import dataclass, field
import numpy as np

from phys_constants import c_, mp_, me_, e_, sigT_, pi_, norm_R_
from phys_functions import func_R
from cooling_distribution import norm_plaw_distrib
from cooling_shell_integrated import ShellLaws, bm_forward_shock, LAM_EPS

MJY = 1e-26          # erg s^-1 cm^-2 Hz^-1
DAY = 86400.

N_GI_PAN, N_GI_NODE = 12, 6        # gma_i: panels over [gma_m,i, gma_M,i] x nodes per panel
N_IN_PAN, N_IN_NODE = 16, 6        # inner, in ln(r - 1)
N_OUT_PAN, N_OUT_NODE = 28, 6      # outer, in s = ln(R_l/R_i)
S_MIN = 1e-3                       # first outer panel edge above s = 0
DELTA_FAC = 1e-3                   # delta_min = DELTA_FAC min(1, gbar_c,i/gma_M,i)
CHUNK = 1500                       # (R_i, R) nodes per vectorised block
GM_MIN_WIDTH = 1.001               # gma_M,i >= this x gma_m,i


def _gl(n):
  u, w = np.polynomial.legendre.leggauss(n)
  return .5*(u + 1.), .5*w


def _composite(lo, hi, npan, nnode, edges_unit=None):
  '''
  Composite Gauss-Legendre nodes and weights on [lo, hi] (arrays, broadcast over a leading
  axis). edges_unit: panel edges on [0, 1], uniform by default.
  '''
  lo, hi = np.asarray(lo, float), np.asarray(hi, float)
  e = np.linspace(0., 1., npan + 1) if edges_unit is None else np.asarray(edges_unit)
  u, w = _gl(nnode)
  a, b = e[:-1], e[1:]
  un = (a[:, None] + (b - a)[:, None]*u[None, :]).ravel()          # (npan*nnode,)
  wn = ((b - a)[:, None]*w[None, :]).ravel()
  span = (hi - lo)[..., None]
  return lo[..., None] + span*un, span*wn


# --- injection: Blandford-McKee forward shock -----------------------------------------
@dataclass
class BMForwardShock:
  '''
  Relativistic forward shock in n_ext = A R^-k (cgs; k = 0: A = n_0 in cm^-3), on the
  Blandford & McKee (1976) solution from R_0 where Gamma_sh = Gamma0. Injection as in
  Paper I Sect. 4: e' = 4 Gamma_i^2 n_ext m_p c^2 with Gamma_i = Gamma_sh/sqrt(2),
  B'_i^2 = 8 pi eps_B e', gma_m,i = eps_e (p-2)/(p-1) (m_p/m_e) Gamma_i / xi_e,
  Ndot = 4 pi R^2 xi_e n_ext, and gma_M,i = (6 pi e/sigma_T B'_i)^1/2 times gM_fac: the Bohm
  limit, acceleration time = Larmor time gma m_e c/(e B') = synchrotron cooling time.
  '''
  E: float = 1e52
  A: float = 1.
  k: float = 0.
  eps_e: float = .1
  eps_B: float = .01
  p: float = 2.5
  xi_e: float = 1.
  z: float = 1.
  dL: float = 1e28
  Gamma0: float = 1e3
  gM_fac: float = 1.
  laws: ShellLaws = field(init=False)

  def __post_init__(self):
    self.laws = bm_forward_shock(self.k)

  # the shock
  def _C(self):
    '''Gamma_sh^2 R^(3-k)'''
    return (17. - 4.*self.k)*self.E/(8.*pi_*self.A*mp_*c_**2)

  def Gamma_sh(self, R):
    return np.sqrt(self._C()*R**(self.k - 3.))

  def tau_sh(self, R):
    '''t - R/c of the shock: int_0^R dR/(2 Gamma_sh^2 c)'''
    return R/(2.*(4. - self.k)*self.Gamma_sh(R)**2*c_)

  def R_los(self, T):
    '''line-of-sight shock radius seen at observed time T: (1+z) tau_sh(R) = T'''
    return (T/(1. + self.z)*2.*(4. - self.k)*c_*self._C())**(1./(4. - self.k))

  @property
  def R0(self):
    return (self._C()/self.Gamma0**2)**(1./(3. - self.k))

  # what an element starts with
  def injection(self, Ri):
    Ri = np.asarray(Ri, float)
    n = self.A*Ri**(-self.k)
    Gi = self.Gamma_sh(Ri)/np.sqrt(2.)
    Bi = np.sqrt(32.*pi_*self.eps_B*n*mp_*c_**2)*Gi
    gm = self.eps_e*(self.p - 2.)/(self.p - 1.)*(mp_/me_)*Gi/self.xi_e
    gM = self.gM_fac*np.sqrt(6.*pi_*e_/(sigT_*Bi))
    gci = 6.*pi_*me_*c_**2*Gi/(sigT_*Bi**2*Ri)             # radial: Eq. eqn:plaws_R/(1+g)
    # where the Bohm limit falls below gma_m,i (very early, strong field) the injection is
    # mono-energetic: a 0.1%-wide law, which norm_plaw_distrib still normalises exactly
    return dict(Gamma=Gi, B=Bi, gm=gm, gM=np.maximum(gM, GM_MIN_WIDTH*gm), gci=gci,
                Ndot=4.*pi_*Ri**2*self.xi_e*n, tau=self.tau_sh(Ri))


# --- one element ----------------------------------------------------------------------
def element_state(r, inj, laws):
  '''Gamma, B', A, I and tau of elements at r = R/R_i (inj entries broadcast with r).'''
  L = laws.Lam
  Gam = inj['Gamma']*r**(-laws.g)
  B = inj['B']*r**(-laws.lam_B)
  A = r**(-laws.lam_n/3.)
  I = np.log(r)/inj['gci'] if abs(L) < LAM_EPS else (1. - r**(-L))/(L*inj['gci'])
  a = 1. + 2.*laws.g
  tau = inj['tau'] + inj['Ri']/(2.*inj['Gamma']**2*c_)*(r**a - 1.)/a
  return Gam, B, A, I, tau


def r_onset(T, inj, laws, z):
  '''r at which the element's on-axis arrival time reaches T (inf-safe, >= 1)'''
  a = 1. + 2.*laws.g
  dt = np.maximum(T/(1. + z) - inj['tau'], 0.)
  return (1. + a*dt*2.*inj['Gamma']**2*c_/inj['Ri'])**(1./a)


def _gi_nodes(inj):
  '''ln gma_i nodes per element: gma_i (M, Ng) and the weight K gma_i^(1-p) dln gma_i'''
  p = inj['p']
  lg, w = _composite(np.log(inj['gm']), np.log(inj['gM']), N_GI_PAN, N_GI_NODE)
  gi = np.exp(lg)
  K = norm_plaw_distrib(inj['gm'], inj['gM'], p)
  return gi, K[..., None]*gi**(1. - p)*w


def Pnu_comoving(nup, B, A, I, gi, wgi):
  '''
  P'_nu' per electron (erg s^-1 Hz^-1) at nup (M, Nnu), for elements with B', A, I (M,) and
  injected-gma nodes gi, wgi (M, Ng).
  '''
  gam = A[:, None]*gi/(1. + gi*I[:, None])                        # (M, Ng)
  w = np.where(gam >= 1., wgi, 0.)
  nuB = e_*B/(2.*pi_*me_*c_)
  x = (2./3.)*nup[:, None, :]/(gam[:, :, None]**2*nuB[:, None, None])
  R = func_R(x)
  pre = sigT_*c_*B**2/(9.*pi_*nuB*norm_R_)
  return pre[:, None]*np.einsum('mg,mgn->mn', w, R)


def gamma2_mean(A, I, gi, wgi):
  '''<gma^2> per electron (gma >= 1), the bolometric counterpart of Pnu_comoving'''
  gam = A[:, None]*gi/(1. + gi*I[:, None])
  return np.sum(np.where(gam >= 1., wgi, 0.)*gam**2, axis=1)


# --- the flux -------------------------------------------------------------------------
def _inner_nodes(T, inj, laws, z, R_max=None):
  '''
  Inner nodes for each outer element (rows): r (M_out, N_in + 1) and the weight of dR,
  with the rectangle [1, 1 + delta_min] as the first column.
  '''
  r_on = r_onset(T, inj, laws, z)
  if R_max is not None:
    r_on = np.minimum(r_on, R_max/inj['Ri'])
  span = np.maximum(r_on - 1., 0.)
  dmin = DELTA_FAC*np.minimum(1., inj['gci']/inj['gM'])
  dmin = np.minimum(dmin, 1e-3*span)
  lo, hi = np.log(np.maximum(dmin, 1e-300)), np.log(np.maximum(span, 1e-300))
  wn, ww = _composite(lo, np.maximum(hi, lo), N_IN_PAN, N_IN_NODE)
  r = 1. + np.exp(wn)
  wR = ww*np.exp(wn)*inj['Ri'][:, None]                          # dR = R_i (r-1) dw
  r = np.concatenate([1. + .5*dmin[:, None], r], axis=1)
  wR = np.concatenate([(dmin*inj['Ri'])[:, None], wR], axis=1)
  wR[span <= 0.] = 0.
  return r, wR


def flux_from_elements(T, nu, Ri, wRi, model, R_max=None):
  '''
  F_nu(T) (erg s^-1 cm^-2 Hz^-1) at one observed time T and frequencies nu, from elements
  injected at Ri, each carrying wRi electrons (wRi = Ndot dR_i for a shell quadrature, or
  the number of electrons of a single element).
  '''
  laws, z = model.laws, model.z
  nu = np.atleast_1d(np.asarray(nu, float))
  inj = model.injection(Ri)
  inj['Ri'], inj['p'] = np.asarray(Ri, float), model.p
  r, wR = _inner_nodes(T, inj, laws, z, R_max)
  M_out, N_in = r.shape
  # flatten the (element, radius) nodes, carrying each element's injection along
  rep = lambda v: np.repeat(np.asarray(v, float), N_in)
  injf = {key: rep(v) for key, v in inj.items() if key != 'p'}
  injf['p'] = model.p
  rf, wf = r.ravel(), (wR*np.asarray(wRi, float)[:, None]).ravel()
  keep = wf > 0.
  rf, wf = rf[keep], wf[keep]
  injf = {key: (v[keep] if np.ndim(v) else v) for key, v in injf.items()}
  F = np.zeros(nu.size)
  for j0 in range(0, rf.size, CHUNK):
    sl = slice(j0, j0 + CHUNK)
    sub = {key: (v[sl] if np.ndim(v) else v) for key, v in injf.items()}
    Gam, B, A, I, tau = element_state(rf[sl], sub, laws)
    R = rf[sl]*sub['Ri']
    Ton = (1. + z)*tau
    Tth = (1. + z)*R/(2.*Gam**2*c_)
    tT = 1. + (T - Ton)/Tth
    ok = (tT >= 1.) & (T - Ton <= 2.*(1. + z)*R/c_)              # mu in [-1, 1]
    nup = (1. + z)*nu[None, :]*tT[:, None]/(2.*Gam[:, None])
    gi, wgi = _gi_nodes(sub)
    P = Pnu_comoving(nup, B, A, I, gi, wgi)
    wt = np.where(ok, wf[sl]*Gam/R*tT**-2, 0.)
    F += wt @ P
  return (1. + z)/(2.*pi_*model.dL**2)*F


def outer_nodes(T, model, R_lo=None):
  '''injection-radius nodes for observed time T: Ri and Ndot dR_i'''
  Rl = model.R_los(T)
  R_lo = model.R0 if R_lo is None else R_lo
  smax = np.log(Rl/R_lo)
  if smax <= 0.:
    return np.zeros(0), np.zeros(0)
  edges = np.concatenate([[0.], np.geomspace(min(S_MIN, smax/10.), smax, N_OUT_PAN)])/smax
  s, ws = _composite(0., smax, N_OUT_PAN, N_OUT_NODE, edges)
  Ri = Rl*np.exp(-s)
  return Ri, ws*Ri*model.injection(Ri)['Ndot']                    # dR_i = R_i ds


def shell_flux(T, nu, model):
  '''F_nu[T, nu] of the whole shell (erg s^-1 cm^-2 Hz^-1).'''
  T = np.atleast_1d(np.asarray(T, float))
  out = np.zeros((T.size, np.size(nu)))
  for i, Ti in enumerate(T):
    Ri, wRi = outer_nodes(Ti, model)
    if Ri.size:
      out[i] = flux_from_elements(Ti, nu, Ri, wRi, model)
  return out


# --- checks ---------------------------------------------------------------------------
A_WIND = 3e35        # cm^-1: n_ext = A_WIND A_* R^-2 (rho = 5e11 A_* g/cm)


def check_injection_indices(k=0., R=1e17, h=1e-3):
  '''d ln/d ln R_i of the BM injection against bm_forward_shock(k)'s shock-front indices'''
  m = BMForwardShock(k=k, A=1. if k == 0. else A_WIND)
  a, b = m.injection(R*np.exp(-h)), m.injection(R*np.exp(h))
  d = lambda key: float(np.log(b[key]/a[key])/(2.*h))
  L = m.laws
  return [('lam_c', d('gci'), L.lam_c), ('-lam_m', d('gm'), -L.lam_m),
          ('lam_M', d('gM'), L.lam_M), ('lam_N', d('Ndot'), L.lam_N),
          ('-g_sh', d('Gamma'), -(3. - k)/2.)]


def check_energy(model=None, Ri=None, r_max=30., nT=100, nnu=80):
  '''
  One element: int dT int dnu F_nu against (1+z) E_rad/(4 pi d_L^2), E_rad = N int dR
  P'_tot/c with P'_tot = (sigma_T c B'^2/6 pi) <gma^2>. Tests the arrival times, Doppler
  factors and the frequency kernel together; returns the relative difference.
  '''
  model = BMForwardShock() if model is None else model
  Ri = 10.*model.R0 if Ri is None else Ri
  inj = model.injection(np.array([Ri]))
  inj['Ri'], inj['p'] = np.array([Ri]), model.p
  laws, z = model.laws, model.z
  # radiated energy, directly in R
  wn, ww = _composite(np.log(1e-12), np.log(r_max - 1.), 60, 8)
  r = 1. + np.exp(wn); wr = ww*np.exp(wn)*Ri
  injr = {key: (np.repeat(v, r.size) if np.ndim(v) else v) for key, v in inj.items()}
  Gam, B, A, I, tau = element_state(r, injr, laws)
  gi, wgi = _gi_nodes(injr)
  Prad = sigT_*c_*B**2/(6.*pi_)*gamma2_mean(A, I, gi, wgi)
  E_rad = np.sum(wr*Prad)/c_
  # the observed fluence of the same element, emission cut at R_max
  R_max = r_max*Ri
  T0 = (1. + z)*inj['tau'][0]
  Tth_end = (1. + z)*R_max/(2.*(inj['Gamma'][0]*r_max**(-laws.g))**2*c_)
  Ton_end = (1. + z)*element_state(np.array([r_max]), inj, laws)[4][0]
  dT = np.geomspace(1e-14*(Ton_end - T0), Ton_end - T0 + 1e4*Tth_end, nT)
  nuB = e_*inj['B'][0]/(2.*pi_*me_*c_)
  nu = np.geomspace(1e-3*nuB, 1e2*inj['gM'][0]**2*nuB*inj['Gamma'][0], nnu)
  F = np.array([flux_from_elements(T0 + d, nu, np.array([Ri]), np.array([1.]), model,
                                   R_max=R_max) for d in dT])
  # power laws in both: trapezoid in the logs (the linear rule is 2-3% off at these grids)
  fl = np.trapezoid(dT*np.trapezoid(F*nu, np.log(nu), axis=1), np.log(dT))
  return fl/((1. + z)*E_rad/(4.*pi_*model.dL**2)) - 1.


def gs02_k0(model, T):
  '''GS02 Table 2, k = 0: breaks 2, 3 (slow) and 9, 11 (fast cooling), nu in Hz, F in mJy'''
  p, z = model.p, model.z
  eb, ee = model.eps_B, model.eps_e*(p - 2.)/(p - 1.)
  n0, E52, td, d28 = model.A, model.E/1e52, T/DAY, model.dL/1e28
  return {
    2: (3.73*(p - .67)*1e15*(1. + z)**.5*E52**.5*ee**2*eb**.5*td**-1.5,
        9.93*(p + .14)*(1. + z)*eb**.5*n0**.5*E52*d28**-2),
    3: (6.37*(p - .46)*1e13*np.exp(-1.16*p)*(1. + z)**-.5*eb**-1.5/n0*E52**-.5*td**-.5,
        4.68*np.exp(4.82*(p - 2.5))*1e3*(1. + z)**((p + 1.)/2.)*ee**(p - 1.)
        *eb**(p - .5)*n0**(p/2.)*E52**((p + 1.)/2.)*td**((1. - p)/2.)*d28**-2),
    9: (3.94*(p - .74)*1e15*(1. + z)**.5*ee**2*eb**.5*E52**.5*td**-1.5,
        .221*(6.27 - p)*(1. + z)**.5/ee*eb**-.5*E52**.5*td**.5*d28**-2),
    11: (5.86e12*(1. + z)**-.5*eb**-1.5/n0*E52**-.5*td**-.5,
         28.4*(1. + z)*eb**.5*n0**.5*E52*d28**-2)}


def gs02_k2_slow(model, T):
  '''GS02 Table 2, k = 2: breaks 2 and 3 (slow cooling); A in units of A_* (A_WIND)'''
  p, z = model.p, model.z
  eb, ee = model.eps_B, model.eps_e*(p - 2.)/(p - 1.)
  As, E52, td, d28 = model.A/A_WIND, model.E/1e52, T/DAY, model.dL/1e28
  return {
    2: (4.02*(p - .69)*1e15*(1. + z)**.5*E52**.5*ee**2*eb**.5*td**-1.5,
        76.9*(p + .12)*(1. + z)**1.5*eb**.5*As*E52**.5*td**-.5*d28**-2),
    3: (4.40*(3.45 - p)*1e10*np.exp(.45*p)*(1. + z)**-1.5*eb**-1.5*As**-2*E52**.5*td**.5,
        8.02*np.exp(7.02*(p - 2.5))*1e5*(1. + z)**(p + .5)*ee**(p - 1.)*eb**(p - .5)
        *As**p*E52**.5*td**(.5 - p)*d28**-2)}


def gs02_spectrum_k0(model, T, nu, regime):
  '''
  GS02's smoothed optically thin spectrum (their Eq. 1 for the first break, Eq. 4's F~_b
  factor for the next, s(p) of Table 2): 'slow' = PLS D-G-H (breaks 2, 3), 'fast' = PLS
  E-F-H (breaks 11, 9; k = 0 only). Normalised on the first break's F_b alone, as GS02
  prescribe, since Table 2's independent p-fits disagree by up to ~8% between breaks. mJy.
  No self-absorption, as in this module.
  '''
  p = model.p
  if model.k == 2.:
    if regime != 'slow':
      raise ValueError('GS02 give no k = 2 fast-cooling nu_c (PLS E, small radii)')
    b = gs02_k2_slow(model, T)
    (n1, F1), (n2, _) = b[2], b[3]
    s1, s2 = 1.76 - .38*p, .80 - .03*p
    be = (1./3., (1. - p)/2., -p/2.)
  elif regime == 'slow':
    b = gs02_k0(model, T)
    (n1, F1), (n2, _) = b[2], b[3]
    s1, s2 = 1.84 - .40*p, 1.15 - .06*p
    be = (1./3., (1. - p)/2., -p/2.)
  else:
    b = gs02_k0(model, T)
    (n1, F1), (n2, _) = b[11], b[9]
    s1, s2 = .597, 3.34 - .82*p
    be = (1./3., -.5, -p/2.)
  x1, x2 = nu/n1, nu/n2
  F = F1*(x1**(-s1*be[0]) + x1**(-s1*be[1]))**(-1./s1)
  return F*(1. + x2**(s2*(be[1] - be[2])))**(-1./s2)


def check_gs02(n_per=3):
  '''
  The spectrum against GS02's smoothed k = 0 spectrum, from 2 decades below the first break
  to 2 decades above the second: slow cooling at 1 day, fast cooling at 1e-3 day (eps_B =
  0.1, n_0 = 10). Returns (regime, nu, ours/GS02). GS02's own fits carry a maximal relative
  difference of 0.5-6% per break (their Table 2) plus that of the p-dependence fit.
  '''
  out = []
  for regime, model, T, (b1, b2) in (
      ('slow', BMForwardShock(eps_B=.01, A=1.), DAY, (2, 3)),
      ('fast', BMForwardShock(eps_B=.1, A=10.), 1e-3*DAY, (11, 9)),
      ('slow', BMForwardShock(k=2., eps_B=.01, A=A_WIND), 10.*DAY, (2, 3))):
    b = gs02_k2_slow(model, T) if model.k == 2. else gs02_k0(model, T)
    lo, hi = np.log10(b[b1][0]) - 2., np.log10(b[b2][0]) + 2.
    nu = np.logspace(lo, hi, int(round((hi - lo)*n_per)) + 1)
    F = shell_flux(T, nu, model)[0]/MJY
    out += [(f'{regime} k={model.k:g}', n, r)
            for n, r in zip(nu, F/gs02_spectrum_k0(model, T, nu, regime))]
  return out


def main():
  for k in (0., 2.):
    print(f'BM injection indices, k={k:g} (measured, expected):')
    for lab, meas, exp in check_injection_indices(k):
      print(f'  {lab:7s} {meas:+.5f} {exp:+.5f}')
  print(f'energy identity, one element: {check_energy():+.2e}')
  print('GS02 smoothed spectrum (ours/GS02):')
  for regime, nu, ratio in check_gs02():
    print(f'  {regime:10s} nu={nu:9.3e} Hz: {ratio:.4f}')


if __name__ == '__main__':
  main()
