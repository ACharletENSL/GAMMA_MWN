'''
Band (1993) low-energy index alpha fitted to the synthetic spectra over the window
[nu_pk/30, 30 nu_pk] around the nuFnu peak, i.e. the way a catalogued GBM spectrum is
fitted (PB's comment on Sect. 5.2: report the Band alpha as a function of C rather than
an index measured decades below the peak).

Cache-only: the two shell sweeps (sweep_shells.load_shell_pairs, RS and the RS+FS sum on
the RS flux scale), with (data) and without (data_rarcut) the post-rarefaction emission.
Both the time-integrated spectrum and the spectrum at the bolometric peak are fitted.

The fit is in log10(nuFnu) with uniform weight per sample; the obs grid is uniform in
log nu, so that is uniform weight per decade (no detector, no Poisson weighting -- this is
the shape a Band function gives the noiseless spectrum, not what a GBM fit would return).

  PYTHONPATH=. python3 band_alpha.py               # fiducial cooling_g100
  BAND_KEY=cooling_g100_hires PYTHONPATH=. python3 band_alpha.py
  BAND_WINDOWS=1 PYTHONPATH=. python3 band_alpha.py  # lower edge nu_pk/30..1000 compared
'''
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

import sweep_gammacm as swp
import sweep_shells as ssh
from environment import figdir
from plotting_functions import COL_RS, COL_TOT

WIN = 30.                 # the window is [nu_pk/WIN, WIN*nu_pk]
WIN_LO_CMP = (30., 100., 300., 1000.)   # lower edges compared by main_windows (upper fixed)
OUTDIR_NAME = 'band_alpha'
METHODS = {'full': 'data', 'cut': 'data_rarcut'}
LS = {'full': '-', 'cut': '--'}
LAB = {'full': 'full', 'cut': 'rf cut'}
SHELLS = {'rs': ('nuFnu_rs', COL_RS, 'RS'), 'tot': ('nuFnu_tot', COL_TOT, 'RS+FS')}


def band_nufnu(lx, logA, alpha, beta, lxp):
  '''log10 nuFnu of a Band function, x in any unit; peak of nuFnu at 10**lxp'''
  x, xp = 10.**lx, 10.**lxp
  x0 = xp/(2. + alpha)                       # e-folding energy
  xb = (alpha - beta)*x0                     # break
  lo = alpha*np.log10(x/xp) - (x - xp)/x0/np.log(10.)
  hi = (beta*np.log10(x/xp) + (alpha - beta)*np.log10(xb/xp)
        - (xb - xp)/x0/np.log(10.))
  return logA + 2.*np.log10(x/xp) + np.where(x < xb, lo, hi)


def fit_band(x, sp, win=WIN, win_lo=None):
  '''
  Band fit of one nuFnu spectrum over [x_pk/win_lo, win*x_pk], x_pk its own maximum
  (win_lo defaults to win).
  Returns alpha, beta, the fitted peak over the data peak, rms (dex) and the
  number of samples.
  '''
  g = (sp > 0) & np.isfinite(sp)
  x, sp = x[g], sp[g]
  ipk = int(np.argmax(sp))
  lxp = np.log10(x[ipk])
  dlo, dhi = np.log10(win_lo or win), np.log10(win)
  m = (np.log10(x) >= lxp - dlo) & (np.log10(x) <= lxp + dhi)
  lx, ly = np.log10(x[m]), np.log10(sp[m])
  # the window must be whole: a truncated one would quietly fit a narrower band
  whole = lx.min() <= lxp - dlo + 0.05 and lx.max() >= lxp + dhi - 0.05
  p0 = (ly.max(), -1., -2.5, lxp)
  bnds = ([ly.max() - 3, -1.99, -10., lxp - 1.5], [ly.max() + 3, 3., -2.01, lxp + 1.5])
  best = None
  for a0 in (-1.5, -1., -0.5):               # a few starts: the alpha-beta-Ep valley is long
    r = least_squares(lambda p: band_nufnu(lx, *p) - ly, (p0[0], a0, p0[2], p0[3]),
                      bounds=bnds, x_scale=(1., .3, .3, .3))
    if best is None or r.cost < best.cost:
      best = r
  res = best.fun
  return dict(alpha=best.x[1], beta=best.x[2], xp_ratio=10.**(best.x[3] - lxp),
              rms=float(np.sqrt(np.mean(res**2))), n=int(m.sum()), whole=bool(whole),
              lxp=lxp)


def measure(key=ssh.KEY, win=WIN, win_los=None):
  '''One row per (C, method, shell, epoch, lower edge): the Band fit of that spectrum.'''
  win_los = win_los or (win,)
  rows = []
  for mk, meth in METHODS.items():
    for p in ssh.load_shell_pairs(key=key, method=meth):
      x, psyn = p['x'], float(p['env'].psyn)
      ok = x >= swp.phys_x_floor(dict(env=p['env'], nub=p['nub']))
      for sk, (col, _, _) in SHELLS.items():
        nf = p[col]
        ipk = swp.bolometric_peak_index(nf, p['nub'])
        for ep, sp in (('integrated', swp.compute_fluence_spectrum(p['Tb'], nf)),
                       ('peak', nf[ipk])):
          for wl in win_los:
            f = fit_band(x[ok], sp[ok], win, wl)
            rows.append(dict(logC=round(float(p['log10ratio']), 2), method=mk, shell=sk,
                             epoch=ep, win_lo=wl, p=psyn, **f))
  return pd.DataFrame(rows)


LEG = dict(fontsize=8, frameon=True, facecolor='w', edgecolor='none', framealpha=1.)


def _legend(ax, **kw):
  '''opaque white legend, drawn over the guide lines that run through it'''
  ax.legend(**LEG, **kw).set_zorder(5)


def _guides(ax, p):
  '''one-zone photon indices: -2/3 (below nu_c, nu_m), -3/2 (fast cooling), -(p+1)/2 (slow
  cooling above nu_m). Labels sit off the curves: -2/3 under its line, the SC one at the right.'''
  tr = ax.get_yaxis_transform()
  for a, lab, xt, dy, ha, va in ((-2/3, r'$-2/3$', 0.02, -0.02, 'left', 'top'),
                                 (-3/2, r'$-3/2$', 0.02, 0.02, 'left', 'bottom'),
                                 (-(p + 1)/2, rf'$-(p+1)/2$', 0.98, 0.02, 'right', 'bottom')):
    ax.axhline(a, c='0.5', ls=':', lw=0.9, zorder=0)
    ax.text(xt, a + dy, lab, color='0.4', fontsize=8, ha=ha, va=va, transform=tr)


def plot(df, outdir, fname='band_alpha.png', win=WIN):
  fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True, layout='constrained')
  for ax, ep, title in zip(axs, ('integrated', 'peak'), ('time-integrated', 'peak')):
    _guides(ax, df.p.iloc[0])
    for sk, (_, c, slab) in SHELLS.items():
      # at the peak the cut has removed nothing yet (it only drops late emission): one line
      for mk in (METHODS if ep == 'integrated' else ('full',)):
        d = df[(df.epoch == ep) & (df.shell == sk) & (df.method == mk)].sort_values('logC')
        ax.plot(d.logC, d.alpha, c=c, ls=LS[mk], lw=1.2, marker='o', ms=3.5,
                mfc=c if mk == 'full' else 'w', label=f'{slab}, {LAB[mk]}' if ep == 'integrated' else
                f'{slab} (full = rf cut)')
        bad = d[~d.whole]
        ax.plot(bad.logC, bad.alpha, ls='', marker='x', c='0.3', ms=6)
    ax.set_xlabel(r'$\log_{10}\mathcal{C}$')
    ax.text(0.03, 0.80, title, transform=ax.transAxes, fontsize=10)
    _legend(ax, loc='upper right')
  axs[0].set_ylabel(rf'Band $\alpha$  ($\nu_{{\rm pk}}/{win:g}$ to ${win:g}\,\nu_{{\rm pk}}$)')
  os.makedirs(outdir, exist_ok=True)
  path = os.path.join(outdir, fname)
  fig.savefig(path, dpi=300)
  plt.close(fig)
  return path


def plot_windows(df, outdir, fname='band_alpha_windows.png', win=WIN):
  '''
  alpha against C for each lower edge of the window (colour), upper edge fixed at win.
  Rows: RS, RS+FS; columns: time-integrated, peak. Solid full, dashed rf cut
  (integrated only -- at the peak the two are identical).
  '''
  wls = sorted(df.win_lo.unique())
  cols = swp.SWEEP_CMAP(np.linspace(0., 1., len(wls)))
  fig, axs = plt.subplots(2, 2, figsize=(9.6, 6.6), sharex=True, sharey=True,
                          layout='constrained')
  for j, (ep, etitle) in enumerate((('integrated', 'time-integrated'), ('peak', 'peak'))):
    for i, (sk, (_, _, slab)) in enumerate(SHELLS.items()):
      ax = axs[i, j]
      _guides(ax, df.p.iloc[0])
      for wl, c in zip(wls, cols):
        for mk in (METHODS if ep == 'integrated' else ('full',)):
          d = df[(df.epoch == ep) & (df.shell == sk) & (df.method == mk)
                 & (df.win_lo == wl)].sort_values('logC')
          ax.plot(d.logC, d.alpha, c=c, ls=LS[mk], lw=1.2, marker='o', ms=3.5,
                  mfc=c if mk == 'full' else 'w',
                  label=rf'$\nu_{{\rm pk}}/{wl:g}$' if mk == 'full' else None)
          bad = d[~d.whole]
          ax.plot(bad.logC, bad.alpha, ls='', marker='x', c='0.3', ms=6)
      ax.text(0.03, 0.80, f'{slab}, {etitle}', transform=ax.transAxes, fontsize=10)
      if i == 1:
        ax.set_xlabel(r'$\log_{10}\mathcal{C}$')
      if j == 0:
        ax.set_ylabel(rf'Band $\alpha$  (to ${win:g}\,\nu_{{\rm pk}}$)')
  _legend(axs[0, 0], loc='upper right', title='lower edge', title_fontsize=8)
  from matplotlib.lines import Line2D
  _legend(axs[0, 1], handles=[Line2D([], [], c='k', ls=LS[mk], label=LAB[mk]) for mk in METHODS],
          loc='upper right')
  os.makedirs(outdir, exist_ok=True)
  path = os.path.join(outdir, fname)
  fig.savefig(path, dpi=300)
  plt.close(fig)
  return path


def main_windows(key=ssh.KEY, win=WIN, win_los=WIN_LO_CMP):
  '''The Band alpha for several lower window edges, upper edge fixed at win*nu_pk.'''
  out = figdir(OUTDIR_NAME, key)
  df = measure(key, win, win_los)
  csv = os.path.join(out, 'band_alpha_windows.csv')
  os.makedirs(out, exist_ok=True)
  df.to_csv(csv, index=False, float_format='%.5g')
  piv = df[df.method == 'full'].pivot_table(index='logC', columns=['epoch', 'shell', 'win_lo'],
                                            values='alpha')
  with pd.option_context('display.width', 250, 'display.precision', 2):
    print(piv)
    print(f"rms max {df.groupby('win_lo').rms.max().round(3).to_dict()} dex; incomplete "
          f"windows: {df[~df.whole][['logC', 'method', 'shell', 'epoch', 'win_lo']].values.tolist()}")
  print('wrote', plot_windows(df, out, win=win), 'and', csv)
  return df


def main(key=ssh.KEY, win=WIN):
  out = figdir(OUTDIR_NAME, key)
  df = measure(key, win)
  csv = os.path.join(out, 'band_alpha.csv')
  os.makedirs(out, exist_ok=True)
  df.to_csv(csv, index=False, float_format='%.5g')
  piv = df.pivot_table(index='logC', columns=['epoch', 'shell', 'method'], values='alpha')
  with pd.option_context('display.width', 200, 'display.precision', 2):
    print(piv)
    print(f"rms max {df.rms.max():.3f} dex; incomplete windows: "
          f"{df[~df.whole][['logC', 'method', 'shell', 'epoch']].values.tolist()}")
  print('wrote', plot(df, out, win=win), 'and', csv)
  return df


if __name__ == '__main__':
  key = os.environ.get('BAND_KEY', ssh.KEY)
  main_windows(key) if os.environ.get('BAND_WINDOWS') else main(key)
