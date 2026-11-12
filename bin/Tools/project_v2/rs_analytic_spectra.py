"""Paper I, Sect. 5 (Round 8, 2026-11-12): purely analytic comoving spectra of the reverse-shocked ejecta.

Power-law limits of tab:indices (toroidal frozen field), slow cooling, below the cooling breaks.
An element injected at t'_i = u t'_inj holds N ~ u^(1+kN) electrons per ln u, injected as a power law -p above
gamma_m(t';t'_i) ~ u^zeta, in the field B'(t';t'_i) ~ u^(lB-kB); its synchrotron frequency is ~ u^xnu,
xnu = 2 zeta + lB - kB. Each element radiates with the step emissivity of Eq. (p_nu_step):
p'_syn(B) N (p-1)/(p-1/3) S(nu'/nu'_i), S = x^(1/3) below, x^(-(p-1)/2) above.
The shell is the integral over u in [1/tau_inj, 1]; the usual model puts all the electrons at the minimum and in the
field of the latest element. Both in units of the usual model at nu'_m (latest element), nu' in units of nu'_m.
The shape is kept after the crossing (the elements share the new laws, Eq. restart), so it is the observed shape too.
Usage: python3 rs_analytic_spectra.py   ->  figures/cooling_distributions/rs_analytic_spectra.png
"""
import os
import numpy as np
from fractions import Fraction as Fr

P = 2.5
KEJ = 2.1
TAUS = (3., 30.)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures', 'cooling_distributions',
                   'rs_analytic_spectra.png')


def rows(k):
  g = (3 - k)/(1 + KEJ)
  return {'thin': dict(ln=3*k/5, kN=(1 - k)/2, km=k - 3, lB=(3*k - 5)/5, kB=2.),
          'thick': dict(ln=3*(2 + k)/(2*(6 - k)), kN=(2 - k)/(6 - k), km=-(2 - k)/(6 - k),
                        lB=(3*k - 2)/(2*(6 - k)), kB=(2 + k)/(6 - k)),
          'strat': dict(ln=3*(2*g + k)/(5*(1 + g)), kN=KEJ*g/(1 + g) - 1, km=0.,
                        lB=(6*g + 3*k - 5)/(5*(1 + g)), kB=(2*g + k - 1)/(1 + g))}


def S(x):
  return np.where(x < 1, np.cbrt(x), x**(-(P - 1)/2))


def spectrum(r, tau, x):
  zeta = r['ln']/3 - r['km']
  D = r['kB'] - r['lB']
  xnu = 2*zeta - D
  lu = np.linspace(-np.log(tau), 0., 4001)
  u = np.exp(lu)[:, None]
  w = u**(1 + r['kN'])
  L = np.trapezoid(w*u**(-D)*S(x[None, :]/u**xnu), lu, axis=0)
  return L/np.trapezoid(w[:, 0], lu)


def alpha_seg(r):
  zeta = r['ln']/3 - r['km']
  D = r['kB'] - r['lB']
  pSC = 1 - (1 + r['kN'])/zeta
  den = 2*(1 + r['kN']) - (1 - pSC)*D
  return None if abs(den) < 1e-12 else (1 - pSC)*(1 + r['kN'] - D)/den


TITLES = {'thin': 'Thin shell', 'thick': 'Thick shell', 'strat': 'Stratified ejecta'}


def figure(out=OUT):
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  from matplotlib.lines import Line2D
  cols = ['#2a78d6', '#eb6834']
  x = np.geomspace(1e-4, 1e2, 400)
  fig, axes = plt.subplots(4, 3, figsize=(7.2, 6.2), sharex=True,
                           gridspec_kw=dict(height_ratios=[2.2, 1, 2.2, 1], hspace=0.08, wspace=0.27))
  for row, k in enumerate((0., 2.)):
    for col, kd in enumerate(('thin', 'thick', 'strat')):
      r = rows(k)[kd]
      ax, axi = axes[2*row, col], axes[2*row + 1, col]
      ys = [spectrum(r, t, x) for t in TAUS]
      for c, y in zip(cols, ys):
        ax.loglog(x, y, color=c, lw=1.4)
        axi.semilogx(x, np.gradient(np.log(y), np.log(x)), color=c, lw=1.)
      us = S(x)
      ax.loglog(x, us, color='k', lw=0.9, ls='--')
      axi.semilogx(x, np.gradient(np.log(us), np.log(x)), color='k', lw=0.9, ls='--')
      hi = 3*max(max(y.max() for y in ys), 1.)
      ax.set_ylim(hi*1e-4, hi)
      a = alpha_seg(r)
      for v in (1/3, -(P - 1)/2):
        axi.axhline(v, color='#52514e', lw=0.6, ls=':', zorder=1)
      if a is not None and -(P - 1)/2 < a < 1/3:
        axi.axhline(a, color='#8a2be2', lw=0.8, ls='-.', zorder=1)
        axi.text(0.03, a, rf'$\alpha_{{\rm SC}}={Fr(a).limit_denominator(100)}$', transform=axi.get_yaxis_transform(),
                 ha='left', va='top', fontsize=6.5, color='#8a2be2')
      if col == 2:
        for v, lab in ((1/3, r'$1/3$'), (-(P - 1)/2, r'$-(p-1)/2$')):
          axi.text(1.02, v, lab, transform=axi.get_yaxis_transform(), ha='left', va='center', fontsize=6.5,
                   color='#52514e')
      axi.set_ylim(-1.3, 0.6)
      axi.set_yticks([-1, -0.5, 0, 0.5])
      axi.set_yticklabels(['$-1$', '', '$0$', ''])
      if row == 0:
        ax.set_title(TITLES[kd], fontsize=9)
      ax.text(0.97, 0.95, f'$k={k:.0f}$', transform=ax.transAxes, ha='right', va='top', fontsize=8)
      if col == 0:
        ax.set_ylabel(r"$L'_{\nu'}/L'^{\,\rm usual}_{\nu'}(\nu'_\mathrm{m})$", fontsize=8)
        axi.set_ylabel('index', fontsize=8)
      else:
        axi.set_yticklabels([])
  key = [Line2D([], [], color=c, lw=1.4) for c in cols] + [Line2D([], [], color='k', lw=0.9, ls='--')]
  labs = [rf"$t'_\mathrm{{inj}}/t'_0={t:g}$" for t in TAUS] + ['usual model']
  fig.legend(key, labs, loc='upper center', ncol=3, fontsize=7, frameon=False, handlelength=1.8,
             columnspacing=1.5, bbox_to_anchor=(0.5, 0.955))
  for ax in axes[-1]:
    ax.set_xlabel(r"$\nu'/\nu'_\mathrm{m}$", fontsize=8)
  for ax in axes.ravel():
    ax.tick_params(labelsize=7)
  fig.subplots_adjust(top=0.89)
  os.makedirs(os.path.dirname(out), exist_ok=True)
  fig.savefig(out, dpi=300, bbox_inches='tight')
  return out


def report():
  x = np.array([1e-3, 1e-2, 0.1, 1., 10.])
  for k in (0., 2.):
    for kd, r in rows(k).items():
      for t in TAUS:
        y = spectrum(r, t, x)
        print(f'k={k:.0f} {kd:5s} tau={t:4g}: L/L_usual(nu_m) at nu/nu_m = {x}: {np.round(y, 3)};'
              f' ratio to usual at 10 nu_m {y[-1]/S(x[-1:])[0]:.2f}, alpha_SC={alpha_seg(r)}')


if __name__ == '__main__':
  report()
  print(figure())
