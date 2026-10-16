# -*- coding: utf-8 -*-
# @Author: acharlet

'''
Lightcurves and figures for Paper I Sect. 4, from afterglow_flux.

fs_lightcurves: F_nu(t_obs) of the BM forward shock at a few bands, and GS02's smoothed
spectrum at the same (t_obs, nu) for comparison: their slow-cooling spectrum (breaks 2, 3)
once nu_m < nu_c, their fast-cooling one (breaks 11, 9) before. GS02 give no k = 2 fast-
cooling nu_c, so for k = 2 the comparison starts at the transition.

THE COMPARISON IS NOT VALID NEAR THE TRANSITION. GS02's spectra are fitted with their breaks
well apart, and at k = 0 nu_c/nu_m ~ t_obs, so after nu_m = nu_c the breaks stay within two
decades of each other for two decades in time. There the GS02 formula and this module differ
by up to 64% (1.64 at 2.2e-4 d, right after the transition; the reference's limitation, its
Eq. 4 smoothing assumes separated breaks), so GS02 is drawn only where |log10(nu_c/nu_m)| >= GS02_MIN_SEP = 2.
Measured at the fiducial, over all four bands, ours/GS02 is 0.96-1.11 with the breaks >= 2
dex apart, 0.96-1.04 at >= 3 dex and 0.96-1.03 at >= 3.5 dex: the excess shrinks as the
breaks separate, toward the accuracy of the spectral check (check_gs02).

Cheap (~0.3 s per observed time at 4 bands): no cache.
'''

import os
import numpy as np
import matplotlib.pyplot as plt

from environment import GAMMA_dir
import afterglow_flux as af

OUTDIR = os.path.join(GAMMA_dir, 'bin', 'Tools', 'figures', 'afterglow')

# observing bands: (label, nu in Hz). No self-absorption in the model: nu_sa ~ 2.5 GHz for
# the fiducial ISM (GS02 break 1), so the lowest band stays above it.
BANDS = (('10 GHz', 1e10), ('100 GHz', 1e11), ('R band', 4.6e14), ('1 keV', 2.42e17))
# categorical slots 1-4 of the dataviz reference palette, validated for 4 adjacent series
# (CVD dE >= 9.1, normal >= 22.9); two sit below 3:1 on white, so the legend names every band
# (no direct labels: the user prefers the legend alone, 2026-10-16)
COLORS = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100')
INK, MUTED = '0.2', '0.5'
# A&A figures: no background grid, and text sits on an opaque white box so no line
# crosses it (user, 2026-10-16)
TEXT_BOX = dict(boxstyle='square,pad=0.15', fc='w', ec='none')
FS_LAB, FS_TICK, FS_ANN = 9., 8., 7.5
FIGSIZE = (3.6, 4.4)                  # A&A single column

# the fiducial forward shock of the figure: ISM, z = 1 at d_L = 6.8 Gpc (Planck 2018)
FIDUCIAL = dict(E=1e52, A=1., k=0., eps_e=.1, eps_B=.01, p=2.5, z=1., dL=2.1e28)
# Gamma_l = 212 .. 2.8 at the fiducial. EARLY TIMES DEPEND ON Gamma0 (injection starts at R_0,
# the coasting phase is not modelled): Gamma0 = 1e5 instead of 1e3 raises F_nu by up to 22% at
# 1e-4 d (R_l/R_0 = 2.8), 3% at 1e-3 d, < 0.2% from 2e-2 d, where the GS02 ratio starts
T_DAYS = (1e-4, 10.)
GS02_MIN_SEP = 2.                     # decades between GS02's breaks for the comparison



def gs02_flux(model, T, nu, min_sep=0.):
  '''
  GS02's smoothed spectrum at each (T, nu), mJy; NaN where no GS02 spectrum applies or
  where its two breaks are less than min_sep decades apart.
  '''
  out = np.full((np.size(T), np.size(nu)), np.nan)
  for i, Ti in enumerate(np.atleast_1d(T)):
    b = af.gs02_k2_slow(model, Ti) if model.k == 2. else af.gs02_k0(model, Ti)
    sep = np.log10(b[3][0]/b[2][0])                 # slow-cooling nu_c/nu_m
    if abs(sep) < min_sep or (model.k == 2. and sep < 0.):
      continue
    out[i] = af.gs02_spectrum_k0(model, Ti, nu, 'slow' if sep > 0. else 'fast')
  return out


def transition_time(model, lo=1e-8*af.DAY, hi=1e4*af.DAY, n=200):
  '''t_obs where GS02's slow-cooling nu_m and nu_c meet (bisection in ln t)'''
  breaks = af.gs02_k2_slow if model.k == 2. else af.gs02_k0
  f = lambda t: np.log(breaks(model, t)[2][0]/breaks(model, t)[3][0])
  a, b = np.log(lo), np.log(hi)
  for _ in range(n):
    m = .5*(a + b)
    a, b = (m, b) if f(np.exp(m)) > 0. else (a, m)
  return np.exp(.5*(a + b))


def fs_lightcurves(model, T, bands=BANDS, min_sep=GS02_MIN_SEP):
  '''F_nu[T, band] in mJy, and GS02's at the same points'''
  nu = np.array([b[1] for b in bands])
  return af.shell_flux(T, nu, model)/af.MJY, gs02_flux(model, T, nu, min_sep)


def plot_fs_lightcurves(model=None, n_per_dec=12, outdir=OUTDIR, fname=None, show=False):
  '''
  Top: F_nu(t_obs) per band, this module solid, GS02 dashed where their breaks are at
  least GS02_MIN_SEP decades apart. Bottom: their ratio. The vertical hairline is GS02's
  nu_m = nu_c.
  '''
  model = af.BMForwardShock(**FIDUCIAL) if model is None else model
  fname = fname or f'fs_lightcurves_k{model.k:g}.png'
  lo, hi = np.log10(T_DAYS)
  td = np.logspace(lo, hi, int(round((hi - lo)*n_per_dec)) + 1)
  F, G = fs_lightcurves(model, td*af.DAY)
  t0 = transition_time(model)/af.DAY

  fig, (ax, axr) = plt.subplots(2, 1, figsize=FIGSIZE, sharex=True,
      gridspec_kw=dict(height_ratios=[3., 1.], hspace=.06))
  for j, ((lab, nu), col) in enumerate(zip(BANDS, COLORS)):
    ax.loglog(td, F[:, j], color=col, lw=2., solid_capstyle='round', zorder=3,
              label=lab)
    ax.loglog(td, G[:, j], color=col, lw=1., ls=(0, (4, 2)), zorder=2)
    axr.semilogx(td, F[:, j]/G[:, j], color=col, lw=1.5, zorder=3)
  # line-style key, in ink: identity of the bands is carried by the legend
  h_ours, = ax.plot([], [], color=MUTED, lw=2., label='this work')
  h_gs, = ax.plot([], [], color=MUTED, lw=1., ls=(0, (4, 2)), label='GS02')
  leg = ax.legend(loc='lower left', fontsize=FS_ANN, frameon=True, ncol=2,
                  handlelength=2.2, columnspacing=1., fancybox=False, framealpha=1.,
                  facecolor='w', edgecolor='none', borderpad=.3)
  leg.set_zorder(6)
  for t in leg.get_texts():
    t.set_color(INK)
  for a in (ax, axr):
    a.axvline(t0, color=MUTED, lw=.7, zorder=1)
    a.tick_params(which='both', labelsize=FS_TICK, colors=INK)
    for s in a.spines.values():
      s.set_color(MUTED)
  ytop = np.nanmax(F)*12.
  ax.set_ylim(ytop*1e-8, ytop)
  ax.annotate('$\\nu_\\mathrm{m}=\\nu_\\mathrm{c}$', (t0, ytop), xytext=(3, -3),
              textcoords='offset points', ha='left', va='top', fontsize=FS_ANN, color=INK,
              bbox=TEXT_BOX, zorder=5)
  par = (f'$E=10^{{{np.log10(model.E):.0f}}}$ erg, $n_0={model.A:g}$ cm$^{{-3}}$\n'
         f'$\\epsilon_\\mathrm{{e}}={model.eps_e:g}$, $\\epsilon_B={model.eps_B:g}$, '
         f'$p={model.p:g}$, $z={model.z:g}$')
  ax.annotate(par, (.04, .26), xycoords='axes fraction', ha='left', va='bottom',
              fontsize=FS_ANN, color=INK, bbox=TEXT_BOX, zorder=5)
  ax.set_ylabel('$F_\\nu$ (mJy)', fontsize=FS_LAB, color=INK)
  axr.axhline(1., color=MUTED, lw=.7, zorder=1)
  i_ok = np.flatnonzero(np.isfinite(G[:, 0]))
  if i_ok.size and i_ok[0] > 0:
    axr.annotate(f'GS02 breaks\n< {GS02_MIN_SEP:g} dex apart',
                 (np.sqrt(td[0]*td[i_ok[0]]), .87), ha='center', va='bottom',
                 fontsize=FS_ANN, color=MUTED, bbox=TEXT_BOX, zorder=5)
  axr.set_ylim(.85, 1.15)
  axr.set_ylabel('$F_\\nu/F_\\nu^{\\rm GS02}$', fontsize=FS_LAB, color=INK)
  axr.set_xlabel('$t_{\\rm obs}$ (days)', fontsize=FS_LAB, color=INK)
  axr.set_xlim(td[0], td[-1])

  os.makedirs(outdir, exist_ok=True)
  path = os.path.join(outdir, fname)
  fig.savefig(path, dpi=300, bbox_inches='tight')
  print(f'saved {path}')
  if show:
    plt.show()
  return fig, (ax, axr), (td, F, G, t0)


def main(show=False):
  fig, axs, (td, F, G, t0) = plot_fs_lightcurves(show=show)
  r = F/G
  print(f'GS02 nu_m = nu_c at t_obs = {t0:.3g} d')
  for j, (lab, _) in enumerate(BANDS):
    print(f'  {lab:8s} ours/GS02 where its breaks are >= {GS02_MIN_SEP:g} dex apart: '
          f'{np.nanmin(r[:, j]):.3f} .. {np.nanmax(r[:, j]):.3f}')


if __name__ == '__main__':
  main()
