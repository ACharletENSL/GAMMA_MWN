# -*- coding: utf-8 -*-
# @Author: acharlet

'''
Paper I, App. B: the equal arrival time surface (EATS) of a shell of fluid elements behind
the BM forward shock, at one observed time T. PROTOTYPE (2026-10-21, J. Granot's suggestion).

(a) Geometry, meridional plane: the EATS of the shock (the injection points of all the
    elements) and of a few elements, each from its injection point (high latitude) to the
    line of sight at R_max = (R_i R_l)^1/2. Lengths in R_l along the axis and in
    R_l/Gamma_sh(R_l) across it. Exact (not small-angle): 1 - mu = c (T/(1+z) - tau)/R.
(b) The same region in the integration plane of Eq. (eqn:Fnu), (R_i, R), coloured by the
    share of F_nu per dln R_i dln R at three bands, with the contours enclosing 50% and 90%
    of the flux (highest density first). Each curve of (a) is a vertical line of (b).

The integrand is that of afterglow_flux.flux_from_elements, on a regular grid instead of
the quadrature nodes; check: its sum over the grid recovers shell_flux to < 1% (printed).
'''

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from phys_constants import c_, pi_
import afterglow_flux as af
from afterglow_lightcurves import FIDUCIAL, OUTDIR, INK, MUTED, TEXT_BOX, FS_LAB, FS_TICK, FS_ANN

T_DAY = 1.                                    # slow cooling, nu_m ~ 1e12, nu_c ~ 5e15 Hz
BANDS = (('10 GHz', 1e10, r'$\nu<\nu_\mathrm{m}$'),
         ('R band', 4.6e14, r'$\nu_\mathrm{m}<\nu<\nu_\mathrm{c}$'),
         ('1 keV', 2.42e17, r'$\nu>\nu_\mathrm{c}$'))
YI_SHOW = (.9, .5, .2, .08)                   # R_i/R_l of the elements drawn (plus R_0)
# dataviz reference blue ramp, steps 100..700 (sequential); elements use 250..700 (ordinal)
BLUE = ('#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5',
        '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b')
CMAP = LinearSegmentedColormap.from_list('blue_seq', BLUE)
N_S, N_U = 240, 200                           # grid in ln(R_l/R_i) and ln(R/R_i)/ln(R_max/R_i)
DYN = 1e-3                                    # colour range below the peak density
FIGSIZE = (7.0, 4.9)                          # A&A full width


def model_T(model=None):
  model = af.BMForwardShock(**FIDUCIAL) if model is None else model
  return model, T_DAY*af.DAY


# --- (a) geometry ---------------------------------------------------------------------
def element_eats(model, T, Ri, n=400):
  '''(R cos th, R sin th) of the EATS of the element injected at Ri, from R_i to R_max'''
  inj = model.injection(np.atleast_1d(Ri))
  inj['Ri'] = np.atleast_1d(np.asarray(Ri, float))
  r_max = af.r_onset(T, inj, model.laws, model.z)[0]
  r = np.geomspace(1., r_max, n)
  sub = {k: v[0] for k, v in inj.items()}
  *_, tau = af.element_state(r, sub, model.laws)
  R = r*Ri
  mu = np.clip(1. - c_*(T/(1. + model.z) - tau)/R, -1., 1.)
  return R*mu, R*np.sqrt(1. - mu**2)


def shock_eats(model, T, R_lo, n=600):
  R = np.geomspace(R_lo, model.R_los(T), n)
  mu = np.clip(1. - c_*(T/(1. + model.z) - model.tau_sh(R))/R, -1., 1.)
  return R*mu, R*np.sqrt(1. - mu**2)


# --- (b) integrand --------------------------------------------------------------------
def integrand_grid(model, T, nu):
  '''
  dF_nu/(F_nu dln R_i dln R) on the grid s = ln(R_l/R_i) (cells), u = ln(R/R_i)/ln(R_max/R_i)
  in [0, 1]; returns s, u edges, the density [nu, s, u] and the grid sum / shell_flux.
  '''
  laws, z = model.laws, model.z
  Rl = model.R_los(T)
  smax = np.log(Rl/model.R0)
  se = np.linspace(0., smax, N_S + 1)
  ue = np.linspace(0., 1., N_U + 1)
  sc, uc = .5*(se[1:] + se[:-1]), .5*(ue[1:] + ue[:-1])
  Ri = Rl*np.exp(-sc)
  inj = model.injection(Ri)
  inj['Ri'], inj['p'] = Ri, model.p
  lnrmax = np.log(af.r_onset(T, inj, laws, z))
  lnr = lnrmax[:, None]*uc[None, :]                                # (N_S, N_U)
  rep = lambda v: np.repeat(np.asarray(v, float), N_U)
  injf = {k: rep(v) for k, v in inj.items() if k != 'p'}
  injf['p'] = model.p
  rf = np.exp(lnr).ravel()
  dens = np.zeros((len(nu), rf.size))
  for j0 in range(0, rf.size, af.CHUNK):
    sl = slice(j0, j0 + af.CHUNK)
    sub = {k: (v[sl] if np.ndim(v) else v) for k, v in injf.items()}
    Gam, B, A, I, tau = af.element_state(rf[sl], sub, laws)
    R = rf[sl]*sub['Ri']
    tT = 1. + (T - (1. + z)*tau)/((1. + z)*R/(2.*Gam**2*c_))
    nup = (1. + z)*np.asarray(nu)[None, :]*tT[:, None]/(2.*Gam[:, None])
    gi, wgi = af._gi_nodes(sub)
    P = af.Pnu_comoving(nup, B, A, I, gi, wgi)                     # (M, Nnu)
    # dF/(dln R_i dln R) = (1+z)/(2 pi d_L^2) Ndot R_i Gamma tT^-2 P'
    wt = (1. + z)/(2.*pi_*model.dL**2)*sub['Ndot']*sub['Ri']*Gam*tT**-2
    dens[:, sl] = (wt[:, None]*P).T
  dens = dens.reshape(len(nu), N_S, N_U)
  F = af.shell_flux(T, nu, model)[0]
  dens /= F[:, None, None]
  # grid sum: dln R_i = ds, dln R = lnrmax du (midpoint rule)
  tot = np.sum(dens*lnrmax[None, :, None], axis=(1, 2))*(se[1] - se[0])*(ue[1] - ue[0])
  return se, ue, sc, lnrmax, dens, tot


def hpd_levels(dens, w, fracs):
  '''density levels enclosing fracs of sum(dens w), highest density first'''
  d, ww = dens.ravel(), w.ravel()
  o = np.argsort(d)[::-1]
  cum = np.cumsum(d[o]*ww[o])/np.sum(d*ww)
  return [d[o][np.searchsorted(cum, f)] for f in fracs]


# --- figure ---------------------------------------------------------------------------
def _style(ax):
  ax.tick_params(which='both', labelsize=FS_TICK, colors=INK)
  for s in ax.spines.values():
    s.set_color(MUTED)


def plot_eats(model=None, outdir=os.path.join(OUTDIR), fname='eats_shell.png', show=False):
  model, T = model_T(model)
  Rl, R0 = model.R_los(T), model.R0
  Gl = model.Gamma_sh(Rl)
  yis = np.array(sorted(YI_SHOW, reverse=True) + [R0/Rl])
  cols = [BLUE[i] for i in np.linspace(3, len(BLUE) - 1, yis.size).round().astype(int)]

  fig = plt.figure(figsize=FIGSIZE)
  gs = fig.add_gridspec(2, 4, height_ratios=[1., 1.15], width_ratios=[1., 1., 1., .06],
                        hspace=.42, wspace=.12)
  ax = fig.add_subplot(gs[0, :3])

  # (a) the region swept by the elements, the shock, the elements
  xs, ps = shock_eats(model, T, R0)
  xo, po = element_eats(model, T, R0)
  ax.fill(np.r_[xs, xo[::-1]]/Rl, np.r_[ps, po[::-1]]*Gl/Rl, color=BLUE[0], lw=0, zorder=1)
  for yi, col in zip(yis, cols):
    x, pp = element_eats(model, T, yi*Rl)
    ax.plot(x/Rl, pp*Gl/Rl, color=col, lw=1.5, solid_capstyle='round', zorder=3)
    ax.plot(x[-1]/Rl, 0., 'o', ms=4.5, color=col, mec='w', mew=.8, zorder=5, clip_on=False)
    ax.plot(x[0]/Rl, pp[0]*Gl/Rl, 'o', ms=3.5, color=col, mec='w', mew=.6, zorder=5)
  ax.plot(xs/Rl, ps*Gl/Rl, color=INK, lw=2., solid_capstyle='round', zorder=4)
  ax.plot(1., 0., 'o', ms=4.5, color=INK, mec='w', mew=.8, zorder=6, clip_on=False)
  # labels
  i_pk = np.argmax(ps)
  ax.annotate('shock', (xs[i_pk]/Rl, ps[i_pk]*Gl/Rl), xytext=(0, 3),
              textcoords='offset points', ha='center', va='bottom', fontsize=FS_ANN,
              color=INK, bbox=TEXT_BOX, zorder=7)
  ax.annotate('$R_l$', (1., 0.), xytext=(4, 3), textcoords='offset points', ha='left',
              va='bottom', fontsize=FS_ANN, color=INK, annotation_clip=False)
  ax.annotate(r'$\to$ observer', (1., 0.), xytext=(4, 14), textcoords='offset points',
              ha='left', va='bottom', fontsize=FS_ANN, color=MUTED, annotation_clip=False)
  # each element labelled R_i/R_l at its injection point, on the shock
  for yi, col in zip(yis, cols):
    lab = f'{yi:.2g}' if yi != yis[-1] else r'$R_0/R_l$'
    if yi == .5:
      lab = r'$R_i/R_l=$' + lab
    x, pp = element_eats(model, T, yi*Rl, n=2)
    right = yi > .7 or yi == yis[-1]
    ax.annotate(lab, (x[0]/Rl, pp[0]*Gl/Rl), xytext=(5, -9) if yi == yis[-1] else
                (5 if right else -3, 4),
                textcoords='offset points', ha='left' if right else 'right', va='bottom',
                fontsize=FS_ANN, color=INK, bbox=TEXT_BOX, zorder=7)
  x5, _ = element_eats(model, T, .5*Rl, n=2)
  ax.annotate(r'$R_{\rm max}$', (x5[-1]/Rl, 0.), xytext=(-4, 3), textcoords='offset points',
              ha='right', va='bottom', fontsize=FS_ANN, color=INK, zorder=7)
  ax.set_xlim(0., 1.)
  ax.set_ylim(0., .43)
  ax.set_xlabel(r'$R\cos\theta/R_l$', fontsize=FS_LAB, color=INK, labelpad=2)
  ax.set_ylabel(r'$\Gamma_{\rm sh}(R_l)\,R\sin\theta/R_l$', fontsize=FS_LAB, color=INK)
  _style(ax)

  # (b) the integration plane
  nus = [b[1] for b in BANDS]
  se, ue, sc, lnrmax, dens, tot = integrand_grid(model, T, nus)
  print('grid sum / shell_flux:', ', '.join(f'{t:.4f}' for t in tot))
  yi_c = np.exp(-sc)
  vmax = dens.max()
  norm = LogNorm(vmax*DYN, vmax)
  axs = [fig.add_subplot(gs[1, j]) for j in range(3)]
  for j, (axb, (lab, nu, reg)) in enumerate(zip(axs, BANDS)):
    # cell corners in (R_i/R_l, R/R_i): u maps to ln(R/R_i) = u ln(R_max/R_i)
    Y = np.exp(-se)[:, None]*np.ones_like(ue)[None, :]
    lnrm_e = np.interp(se, sc, lnrmax)                   # ln(R_max/R_i) at the cell edges
    Z = np.exp(lnrm_e[:, None]*ue[None, :])
    d = np.where(dens[j] > vmax*DYN, dens[j], np.nan)
    axb.pcolormesh(Y, Z, d, cmap=CMAP, norm=norm, shading='flat', rasterized=True, zorder=1)
    w = lnrmax[:, None]*np.ones(N_U)[None, :]
    lev = hpd_levels(dens[j], w, (.9, .5))
    Yc = yi_c[:, None]*np.ones(N_U)[None, :]
    Zc = np.exp(lnrmax[:, None]*(.5*(ue[1:] + ue[:-1]))[None, :])
    axb.contour(Yc, Zc, dens[j], levels=sorted(lev), colors=INK, linewidths=(.6, 1.),
                linestyles=('dashed', 'solid'), zorder=3)
    # boundaries: line of sight (top) and shock (bottom)
    axb.plot(yi_c, np.exp(lnrmax), color=INK, lw=1., zorder=4)
    axb.axhline(1., color=INK, lw=1., zorder=4)
    for yi, col in zip(yis, cols):
      axb.plot([yi, yi], [1., np.exp(np.interp(-np.log(yi), sc, lnrmax))], color=col,
               lw=1., ls=(0, (1, 1.5)), zorder=2)
    axb.set_yscale('log')
    axb.set_yticks([1., 1.5, 2., 3., 5.])
    axb.set_yticks([], minor=True)
    axb.set_yticklabels(['1', '1.5', '2', '3', '5'])
    axb.set_xlim(0., 1.)
    axb.set_xticks([0., .5, 1.])
    axb.set_xticklabels(['0', '0.5', '1'])
    axb.set_xticks([.25, .75], minor=True)
    axb.set_ylim(1., np.exp(lnrmax.max())*1.08)
    axb.set_title(f'{lab} ({reg})', fontsize=FS_ANN + .5, color=INK, pad=3)
    axb.set_xlabel(r'$R_i/R_l$', fontsize=FS_LAB, color=INK)
    if j == 0:
      axb.set_ylabel(r'$R/R_i$', fontsize=FS_LAB, color=INK)
      axb.annotate(r'line of sight ($R=R_{\rm max}$)',
                   (.2, np.exp(np.interp(-np.log(.2), sc, lnrmax))),
                   xytext=(5, 2), textcoords='offset points', ha='left', va='bottom',
                   fontsize=FS_ANN, color=INK, bbox=TEXT_BOX, zorder=6)
    else:
      axb.tick_params(labelleft=False)
    _style(axb)
  cax = fig.add_subplot(gs[1, 3])
  cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax)
  cb.set_label(r'$\mathrm{d}F_\nu/(F_\nu\,\mathrm{d}\ln R_i\,\mathrm{d}\ln R)$',
               fontsize=FS_LAB, color=INK)
  _style(cax)

  os.makedirs(outdir, exist_ok=True)
  path = os.path.join(outdir, fname)
  fig.savefig(path, dpi=300, bbox_inches='tight')
  print(f'saved {path}')
  if show:
    plt.show()
  return fig, path


if __name__ == '__main__':
  plot_eats()


def flux_shares(model=None):
  '''where the flux comes from, per band: printed shares of F_nu'''
  model, T = model_T(model)
  Rl = model.R_los(T)
  se, ue, sc, lnrmax, dens, tot = integrand_grid(model, T, [b[1] for b in BANDS])
  w = lnrmax[:, None]*np.ones(N_U)[None, :]*(se[1] - se[0])*(ue[1] - ue[0])
  lnr = lnrmax[:, None]*(.5*(ue[1:] + ue[:-1]))[None, :]
  yi = np.exp(-sc)[:, None]*np.ones(N_U)[None, :]
  # tT on the grid (ultra-relativistic, BM: Eq. eqn:Ttilde_R with R_max/R = exp(lnrmax - lnr))
  a = 1. + 2.*model.laws.g
  tT = (a - 1. + np.exp(a*(lnrmax[:, None] - lnr)))/a
  for j, (lab, nu, _) in enumerate(BANDS):
    d = dens[j]*w
    lev50, lev90 = hpd_levels(dens[j], w/w.max(), (.5, .9))
    print(f'{lab:7s}: R_i/R_l>0.5 {d[yi > .5].sum():.2f}, tT<2 {d[tT < 2.].sum():.2f}, '
          f'R/R_i-1<0.05 {d[np.exp(lnr) - 1. < .05].sum():.2f}, '
          f'min R_i/R_l in 90% region {yi[dens[j] >= lev90].min():.2f}')
