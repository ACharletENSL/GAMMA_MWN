"""Paper I, Sect. 5 (Round 8 v17, 2026-11-13): reverse-shock spectra as BROKEN POWER LAWS built from the article's
segment results only (author's choice: no quadrature, no exponential cutoff, no kernel smoothing).

Inputs: the power-law indices of tab:indices (rs_analytic_spectra.rows, constant magnetization of the ejecta) and the
cooling regime of the newest elements, Q = gamma_c/gamma_m, from fiducial afterglow parameters (FID).
Comoving units: frequency in nu'_m of the newest element, luminosity in the peak of the usual model (SPN98 broken power
law of the newest conditions: F_max at min(nu'_m, nu'_c)). Observed spectra along the line of sight: shifted with
nu_m(T) and F_max(T) of the usual model (Eq. nu_obs_T during the injection, after-crossing rows of tab:indices after).

Shell, newest elements in slow cooling (Q > 1):
  elements older than t'_tr = t'_inj Q^(-1/(eta - zeta)) are in fast cooling (gamma_m/gamma_M ~ t_i^(zeta - eta));
  high line F nu^-(p-1)/2 from nu'_m (F: brightness factor over the slow cooling elements, finite tau_inj);
  above nu'_m the shell is F times the usual model, its -p/2 cooled segment above nu'_c = Q^2 included;
  below nu'_m: adiabatic tail clip(alpha_SC, -(p-1)/2, 1/3) down to nu'_tr (or nu'_m,ad); if hidden (alpha_SC > 1/3 or
  2 zeta + lambda_B - kappa_B <= 0), a 1/3 line at the low-frequency factor F_lo meeting the high line;
  relic tail clip(alpha_FC, -1/2, 1/3) from nu'_c,ad up to nu'_tr; 1/3 below the lowest break.
Shell, newest elements in fast cooling (Q < 1): cooled line Q nu^-p/2 above nu'_m, -1/2 down to nu'_c = Q^2,
  relic tail clip(alpha_FC, -1/2, 1/3) down to nu'_c,ad, 1/3 below.
Electrons below gamma_e = sqrt(2) do not radiate: each tail stops at the element whose minimum (slow cooling) or maximum
(fast cooling) is sqrt(2), which moves up after the crossing as all the electrons cool adiabatically.
After the crossing: no young electrons; the shape at the crossing ends at the last element's break (Q^2) and shifts
rigidly. Usage: python3 rs_bpl_spectra.py  -> report + figures/cooling_distributions/rs_observed_spectra.png
"""
import os
import numpy as np
from rs_analytic_spectra import rows, G, AFTER, inj_indices, KEJ, P, TITLES, T_OBS

ME, MP, SIGT, C = 9.1094e-28, 1.6726e-24, 6.6524e-25, 2.9979e10
FID = dict(E=1e53, G4=300., eps_e=0.3, sigma4=1e-4, n=1., Astar=0.1, xi=10.)
NS06 = {0.: 0.106, 2.: 0.025}       # Gamma_34 - 1 of the self-similar solution for k_ej = 1.5 (planar runs, 2026-11-12)
TAU_DEC_STRAT_THICK = 1e4            # t'_inj/t'_0 at T_dec, i.e. R_0 ~ 1e-3 to 1e-4 R_dec
T_RS = {'thin': 1., 'thick': 10., 'strat': np.inf}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures', 'cooling_distributions',
                   'rs_observed_spectra.png')


def gm_from(G34m1):
  return FID['eps_e']*G34m1*MP/ME*(P - 2)/(P - 1)


def fiducial(kd, k):
  """gamma_m,i and Q = gamma_c/gamma_m of the newest element at T_dec, and tau_inj at T_dec"""
  E, G4, xi = FID['E'], FID['G4'], FID['xi']
  if k == 0.:
    rho = FID['n']*MP
    R = (3*E/(4*np.pi*rho*C**2*G4**2))**(1/3)
  else:
    A = 5e11*FID['Astar']
    R = E/(4*np.pi*A*G4**2*C**2)
    rho = A/R**2
  g = G[kd](k)
  if kd == 'thin':
    G34m1, G3 = 0.15, G4
  elif kd == 'thick':
    phi = np.sqrt((3 - k)*xi)
    G34 = np.sqrt(phi/2)
    G34m1, G3 = G34 - 1, G4/(G34 + np.sqrt(G34**2 - 1))
  else:
    G34m1, G3 = NS06[k], G4
  gm = gm_from(G34m1)
  epsB = 2*FID['sigma4']*(1 + G34m1)/G34m1
  gc = (1 + g)*3*ME/(16*SIGT*epsB*G3*rho*R)
  tau = (gm/np.sqrt(2))**(1/(3 - k)) if kd == 'thin' else TAU_DEC_STRAT_THICK
  return dict(gm=gm, gc=gc, Q=gc/gm, epsB=epsB, G34m1=G34m1, tau=tau, R=R)


def indices(kd, k):
  r = rows(k)[kd]
  zeta = r['ln']/3 - r['km']
  D = r['kB'] - r['lB']
  s = r['ln']/3 + 2*r['lB'] - 1
  eta = r['ln']/3 + 2*r['kB'] - 1 if s > 0 else 2*D
  kN = r['kN']
  xm = 2*zeta - D                               # frequency index of the old minima in t'_i
  xc = 2*eta - D                                # frequency index of the old maxima in t'_i

  def alpha_old(pX):
    den = 2*(1 + kN) - (1 - pX)*D
    return np.inf if abs(den) < 1e-12 else (1 - pX)*(1 + kN - D)/den
  aSC = alpha_old(1 - (1 + kN)/zeta)
  aFC = alpha_old(1 - (1 + kN)/eta)
  ehi = 1 + kN - D + xm*(P - 1)/2
  elo = 1 + kN - D - xm/3
  return dict(zeta=zeta, D=D, s=s, eta=eta, kN=kN, xm=xm, xc=xc, aSC=aSC, aFC=aFC, ehi=ehi, elo=elo,
              qexp=2*r['kB'] + r['km'] - 1)


def _int(e, ua):
  """int_ua^1 u^e dln u"""
  return -np.log(ua) if abs(e) < 1e-12 else (1 - ua**e)/e


def after_ln_g(kd, k):
  if kd == 'thin':
    g = 2. if k == 0 else 1.
    return 3*(3 + g)/(4*(1 + g)), g
  return (13 - 2*k)/(9 - 2*k), (7 - 2*k)/2


def state(kd, k, T):
  """tau_inj, Q of the newest element, whether the injection goes on, and gamma_m of the newest element (absolute,
  adiabatically shifted after the crossing), at T (units of T_dec)"""
  f, ix = fiducial(kd, k), indices(kd, k)
  if kd == 'thin':
    tau, Q, Tc = f['tau'], f['Q'], 1.
    gm = f['gm']
  else:
    Tc = min(T, T_RS[kd])
    c = inj_indices(kd, k)[2]                   # d ln tau / d ln T
    tau = f['tau']*Tc**c
    Q = f['Q']*(tau/f['tau'])**ix['qexp']
    gm = f['gm']*(tau/f['tau'])**(-rows(k)[kd]['km'])
  if T > Tc:                                    # after the crossing: A = (t'/t'_RS)^(-lambda_n,b/3), T ~ R^(1+2g_b)
    lnb, gb = after_ln_g(kd, k)
    gm *= (T/Tc)**(-(1 + gb)*lnb/(3*(1 + 2*gb)))
  return tau, Q, T <= Tc if kd != 'thin' else False, gm


def lines_shell(kd, k, T):
  """list of (nu_lo, nu_hi, index) with an anchor (nu_a, L_a): comoving broken power law of the shell"""
  ix = indices(kd, k)
  tau, Q, during, gm = state(kd, k, T)
  u0 = 1/tau
  r2 = np.sqrt(2)
  u_sc = (r2/gm)**(1/ix['zeta']) if gm > r2 else 1.          # slow cooling elements: minimum = sqrt(2)
  u_fc = (r2/(Q*gm))**(1/ix['eta']) if Q*gm > r2 else 1.     # fast cooling elements: maximum = sqrt(2)
  lo13, hi13 = -(P - 1)/2, 1/3
  info = dict(tau=tau, Q=Q, during=during)
  if Q >= 1:                                    # newest elements in slow cooling
    utr = Q**(-1/(ix['eta'] - ix['zeta']))      # (t'_tr/t'_inj): older elements in fast cooling
    ua = max(u0, utr, u_sc)
    Ntot = _int(1 + ix['kN'], u0)
    F = _int(ix['ehi'], ua)/Ntot
    Flo = _int(ix['elo'], ua)/Ntot
    info.update(utr=utr, F=F, Flo=Flo, mixed=utr > max(u0, u_fc), u_sc=u_sc, u_fc=u_fc, gm=gm)
    segs = []
    # top
    nu_end = Q**2
    if during:                                  # F times the usual model above nu'_m, cooling break included
      segs += [(1., nu_end, lo13), (nu_end, np.inf, -P/2)]
    else:
      segs += [(1., nu_end, lo13)]
    anchor = (1., F)
    # below nu'_m: one contiguous chain from the top segment down to zero
    a = ix['aSC']
    nu_a = ua**ix['xm'] if ix['xm'] > 0 else 1.
    below = []
    if ix['xm'] > 1e-6 and lo13 < a < hi13:     # visible adiabatic tail
      below.append((nu_a, 1., a))
      cursor = nu_a
    elif ix['xm'] > 1e-6 and a <= lo13:         # tail steeper than -(p-1)/2: injected slope extends down
      below.append((nu_a, 1., lo13))
      cursor = nu_a
    else:                                       # hidden tail: 1/3 at F_lo meeting the high line at nux
      nux = (F/Flo)**(1/(1/3 + (P - 1)/2))
      if nux > 1.:
        segs = [(1., nux, hi13)] + [(max(s_[0], nux), s_[1], s_[2]) for s_ in segs if s_[1] > nux]
        cursor = 1.
      else:
        below.append((nux, 1., lo13))
        cursor = nux
      anchor = (nux, Flo*nux**(1/3))
    if info['mixed']:                           # relic tail of the fast cooling elements below nu'_tr
      nu_ca = Q**2*max(u0, u_fc)**ix['xc']
      nu_tr = min(utr**ix['xm'] if ix['xm'] > 0 else 1., cursor)
      if nu_tr < cursor:                        # 1/3 line of the hidden tail down to nu'_tr
        below.append((nu_tr, cursor, hi13))
      af = min(max(ix['aFC'], -0.5), hi13)
      if nu_ca < nu_tr:
        below.append((nu_ca, nu_tr, af))
        cursor = nu_ca
      else:
        cursor = nu_tr
    below.append((0., cursor, hi13))
    return segs + below, anchor, info
  # newest elements in fast cooling
  nu_c = Q**2
  nu_ca = Q**2*max(u0, u_fc)**ix['xc']
  af = min(max(ix['aFC'], -0.5), hi13)
  info.update(utr=1., F=np.nan, Flo=np.nan, mixed=True, u_sc=u_sc, u_fc=u_fc, gm=gm)
  if during:
    segs = [(1., np.inf, -P/2), (nu_c, 1., -0.5), (nu_ca, nu_c, af), (0., nu_ca, hi13)]
    anchor = (1., Q)
  else:
    segs = [(nu_ca, nu_c, af), (0., nu_ca, hi13)]
    anchor = (nu_c, 1.)
  return segs, anchor, info


def lines_usual(kd, k, T):
  tau, Q, during, gm = state(kd, k, T)
  if Q >= 1:
    segs = [(0., 1., 1/3), (1., Q**2, -(P - 1)/2)] + ([(Q**2, np.inf, -P/2)] if during else [])
    return segs, (1., 1.)
  segs = [(0., Q**2, 1/3)] + ([(Q**2, 1., -0.5), (1., np.inf, -P/2)] if during else [])
  return segs, (Q**2, 1.)


def evaluate(segs, anchor, nu):
  """continuous broken power law through the anchor; zero outside the segments"""
  segs = sorted([s_ for s_ in segs if s_[1] > s_[0]], key=lambda s_: s_[0])
  # levels at segment starts by continuity from the anchor
  na, La = anchor
  lev = {}
  ia = [i for i, s_ in enumerate(segs) if s_[0] <= na <= s_[1]][0]
  lo, hi, a = segs[ia]
  lev[ia] = (na, La)
  for i in range(ia + 1, len(segs)):            # upward
    pn, pL = lev[i - 1]
    x0 = segs[i][0]
    lev[i] = (x0, pL*(x0/pn)**segs[i - 1][2])
  for i in range(ia - 1, -1, -1):               # downward
    nn, nL = lev[i + 1]
    x1 = segs[i][1]
    xr = segs[i + 1][0]
    L1 = nL*(xr/nn)**segs[i + 1][2]
    lev[i] = (x1, L1)
  out = np.zeros_like(nu)
  for i, (lo, hi, a) in enumerate(segs):
    m = (nu >= lo) & (nu < hi)
    rn, rL = lev[i]
    out[m] = rL*(nu[m]/rn)**a
  return out


def observed(kd, k, T, nu):
  """F_nu / F_max^usual(T_dec) at nu / nu_m^usual(T_dec): (shell, usual)"""
  if kd == 'thin':
    a, b = AFTER[(kd, k)]
    lnu, lF = a*np.log(T), b*np.log(T)
  else:
    ai, bi, ci = inj_indices(kd, k)
    Tc = min(T, T_RS[kd])
    lnu, lF = ai*np.log(Tc), bi*np.log(Tc)
    if T > T_RS[kd]:
      a, b = AFTER[(kd, k)]
      lnu, lF = lnu + a*np.log(T/Tc), lF + b*np.log(T/Tc)
  x = nu/np.exp(lnu)
  s, an, info = lines_shell(kd, k, T)
  u, au = lines_usual(kd, k, T)
  return np.exp(lF)*evaluate(s, an, x), np.exp(lF)*evaluate(u, au, x), info


def report():
  for k in (0., 2.):
    for kd in ('thin', 'thick', 'strat'):
      f, ix = fiducial(kd, k), indices(kd, k)
      print(f"{kd:5s} k={k:.0f}: gamma_m={f['gm']:.1f} eps_B={f['epsB']:.2e} gamma_c={f['gc']:.3g} Q={f['Q']:.3g} "
            f"tau_dec={f['tau']:.3g} | s={ix['s']:.3f} zeta={ix['zeta']:.3f} eta={ix['eta']:.3f} "
            f"x_m={ix['xm']:.3f} x_c={ix['xc']:.3f} aSC={ix['aSC']:.3f} aFC={ix['aFC']:.3f} qexp={ix['qexp']:.3f}")
      for T in T_OBS:
        s, an, info = lines_shell(kd, k, T)
        print(f"     T={T:5g}: tau={info['tau']:.3g} Q={info['Q']:.3g} during={info['during']} gm={info['gm']:.3g} "
              f"t_tr/t_inj={info['utr']:.3g} mixed={info['mixed']} u_sqrt2: SC {info['u_sc']:.3g} FC {info['u_fc']:.3g} "
              f"(u0={1/info['tau']:.3g}) F={info['F']:.3g} Flo={info['Flo']:.3g}")
      if kd != 'thin':                          # switch of the newest element during the injection
        c = inj_indices(kd, k)[2]
        if ix['qexp'] != 0:
          Tsw = (1/f['Q'])**(1/(ix['qexp']*c))
          print(f"     newest element switches regime at T/T_dec = {Tsw:.3g} (injection until {T_RS[kd]})")


GUIDE_COL = {'SC': '#7b3fa0', 'FC': '#8c5a2b'}


def tails_shown(kd, k):
  """(alpha, colour) of the adiabatic and relic tails that appear as segments of the shell at one of the times"""
  ix = indices(kd, k)
  out = []
  for name, a in (('SC', ix['aSC']), ('FC', ix['aFC'])):
    if not -(P - 1)/2 < a < 1/3:
      continue
    if any(abs(sg[2] - a) < 1e-9 and sg[1] > sg[0] for T in T_OBS for sg in lines_shell(kd, k, T)[0]):
      out.append((a, GUIDE_COL[name]))
  return out


def figure(out=OUT):
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  from matplotlib.lines import Line2D
  cols = ['#2a78d6', '#eb6834', '#3a9b5c']
  nu = np.geomspace(1e-9, 1e6, 3000)
  fig, axes = plt.subplots(4, 3, figsize=(7.2, 6.4), sharex=True,
                           gridspec_kw=dict(height_ratios=[2.2, 1, 2.2, 1], hspace=0.08, wspace=0.27))
  for row, k in enumerate((0., 2.)):
    for col, kd in enumerate(('thin', 'thick', 'strat')):
      ax, axi = axes[2*row, col], axes[2*row + 1, col]
      hi = 0.
      for c, T in zip(cols, T_OBS):
        F, Fu, _ = observed(kd, k, T, nu)
        for y, ls, lw in ((F, '-', 1.4), (Fu, '--', 0.9)):
          ok = y > 0
          ax.loglog(nu[ok], y[ok], color=c, lw=lw, ls=ls)
        ok = F > 0
        idx = np.gradient(np.log(F[ok]), np.log(nu[ok]))
        axi.semilogx(nu[ok], idx, color=c, lw=1.)
        hi = max(hi, F.max(), Fu.max())
      ax.set_ylim(hi*1e-6, hi*3)
      for v in (1/3, -(P - 1)/2, -P/2):
        axi.axhline(v, color='#52514e', lw=0.6, ls=':', zorder=1)
      for v, cg in tails_shown(kd, k):            # alpha_SC, alpha_FC of tab:indices where the tail shows
        axi.axhline(v, color=cg, lw=0.9, ls='-.', zorder=1.5)
      if col == 2:
        for v, lab in ((1/3, r'$1/3$'), (-(P - 1)/2, r'$-(p-1)/2$'), (-P/2, r'$-p/2$')):
          axi.text(1.02, v, lab, transform=axi.get_yaxis_transform(), ha='left', va='center', fontsize=6.5,
                   color='#52514e')
        ax.text(1.04, 0.5, f'$k={k:.0f}$', transform=ax.transAxes, ha='left', va='center', fontsize=9, rotation=90)
      axi.set_ylim(-1.5, 0.6)
      axi.set_yticks([-1, -0.5, 0, 0.5])
      axi.set_yticklabels(['$-1$', '', '$0$', ''])
      if row == 0:
        ax.set_title(TITLES[kd], fontsize=9)
      if col == 0:
        ax.set_ylabel(r'$F_\nu/F^{\,\rm usual}_{\nu,\max}(T_{\rm dec})$', fontsize=8)
        axi.set_ylabel('index', fontsize=8)
      else:
        axi.set_yticklabels([])
  key = [Line2D([], [], color='k', lw=1.4), Line2D([], [], color='k', lw=0.9, ls='--')] + \
        [Line2D([], [], color=c, lw=1.4) for c in cols] + \
        [Line2D([], [], color=GUIDE_COL[n], lw=0.9, ls='-.') for n in ('SC', 'FC')]
  labs = ['shell', 'usual model'] + [rf'$T/T_{{\rm dec}}={a:g}$' for a in T_OBS] + \
         [r'$\alpha_{\rm SC}$', r'$\alpha_{\rm FC}$']
  fig.legend(key, labs, loc='upper center', ncol=7, fontsize=7, frameon=False, handlelength=1.8,
             columnspacing=1.2, bbox_to_anchor=(0.5, 0.955))
  for ax in axes[-1]:
    ax.set_xlabel(r'$\nu/\nu^{\rm usual}_\mathrm{m}(T_{\rm dec})$', fontsize=8)
  for ax in axes.ravel():
    ax.tick_params(labelsize=7)
  fig.subplots_adjust(top=0.89)
  os.makedirs(os.path.dirname(out), exist_ok=True)
  fig.savefig(out, dpi=300, bbox_inches='tight')
  return out


if __name__ == '__main__':
  report()
  print(figure())
