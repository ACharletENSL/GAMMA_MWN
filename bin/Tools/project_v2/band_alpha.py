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


def fit_band(x, sp, win=WIN):
  '''
  Band fit of one nuFnu spectrum over [x_pk/win, win*x_pk], x_pk its own maximum.
  Returns alpha, beta, the fitted peak over the data peak, rms (dex) and the
  number of samples.
  '''
  g = (sp > 0) & np.isfinite(sp)
  x, sp = x[g], sp[g]
  ipk = int(np.argmax(sp))
  lxp = np.log10(x[ipk])
  m = np.abs(np.log10(x) - lxp) <= np.log10(win)
  lx, ly = np.log10(x[m]), np.log10(sp[m])
  # the window must be whole: a truncated one would quietly fit a narrower band
  whole = lx.min() <= lxp - np.log10(win) + 0.05 and lx.max() >= lxp + np.log10(win) - 0.05
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


def measure(key=ssh.KEY, win=WIN):
  '''One row per (C, method, shell, epoch): the Band fit of that spectrum.'''
  rows = []
  for mk, meth in METHODS.items():
    for p in ssh.load_shell_pairs(key=key, method=meth):
      x = p['x']
      ok = x >= swp.phys_x_floor(dict(env=p['env'], nub=p['nub']))
      for sk, (col, _, _) in SHELLS.items():
        nf = p[col]
        ipk = swp.bolometric_peak_index(nf, p['nub'])
        for ep, sp in (('integrated', swp.compute_fluence_spectrum(p['Tb'], nf)),
                       ('peak', nf[ipk])):
          f = fit_band(x[ok], sp[ok], win)
          rows.append(dict(logC=round(float(p['log10ratio']), 2), method=mk, shell=sk,
                           epoch=ep, **f))
  return pd.DataFrame(rows)


def plot(df, outdir, fname='band_alpha.png', win=WIN):
  fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True, layout='constrained')
  for ax, ep, title in zip(axs, ('integrated', 'peak'), ('time-integrated', 'peak')):
    for a, lab in ((-2/3, r'$-2/3$'), (-3/2, r'$-3/2$')):
      ax.axhline(a, c='0.5', ls=':', lw=0.9, zorder=0)
      ax.text(0.02, a + 0.02, lab, color='0.4', fontsize=8,
              transform=ax.get_yaxis_transform())
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
    ax.legend(frameon=False, fontsize=8, loc='upper right')
  axs[0].set_ylabel(rf'Band $\alpha$  ($\nu_{{\rm pk}}/{win:g}$ to ${win:g}\,\nu_{{\rm pk}}$)')
  os.makedirs(outdir, exist_ok=True)
  path = os.path.join(outdir, fname)
  fig.savefig(path, dpi=300)
  plt.close(fig)
  return path


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
  main(key=os.environ.get('BAND_KEY', ssh.KEY))
