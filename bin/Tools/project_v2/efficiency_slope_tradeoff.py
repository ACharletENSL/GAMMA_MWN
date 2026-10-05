'''
Radiative efficiency against low-energy photon index, one point per cooling
parameter C, with (full) and without (rf cut) the post-rarefaction emission.

Cache-only: reads the eps_rad table written by sweep_efficiency and the
time-integrated slope table written by sweep_rarcut (RS, z=4). Nothing is
recomputed.

Two slope estimates are drawn:
  - asymptotic: the free-fitted index far below min(nu_m, nu_c)
    (spectral_breaks.fluence_low_slope), i.e. what the spectrum tends to;
  - in band: the nuFnu slope 2 decades below the nuFnu peak, closer to what
    a Band fit over a GBM-like window around the peak would return.
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm, colors

FIGROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures', 'fiducial')
EFF_CSV = os.path.join(FIGROOT, 'efficiency_sweep', 'efficiency_table.csv')
SLOPE_CSV = os.path.join(FIGROOT, 'rarcut_compare', 'fluence_low_slopes.csv')
OUTDIR = os.path.join(FIGROOT, 'rarcut_compare')

METHODS = {'full': 'data', 'cut': 'data_rarcut'}
LS = {'full': '-', 'cut': '--'}
LAB = {'full': 'full', 'cut': 'rf cut'}


def load_tradeoff(z=4, band_dex=2):
  '''DataFrame indexed by log10 C: eps_rad and photon indices for both treatments.'''
  eff = pd.read_csv(EFF_CSV)
  eff = eff[eff.z == z].copy()
  eff['logr'] = eff.log10ratio_target.round(2)
  sl = pd.read_csv(SLOPE_CSV)
  sl['logr'] = sl.logr.round(2)
  out = sl[['logr']].copy()
  for key, meth in METHODS.items():
    e = eff[eff.method == meth].set_index('logr').eps_rad
    out[f'eps_{key}'] = out.logr.map(e)
    out[f'alpha_asym_{key}'] = sl[f'a_{key}'] - 2.
    out[f'alpha_band_{key}'] = sl[f'aband{band_dex}_{key}'] - 2.
  return out.set_index('logr')


def plot_tradeoff(z=4, band_dex=2, outdir=OUTDIR, fname='efficiency_slope_tradeoff.png'):
  df = load_tradeoff(z, band_dex)
  norm = colors.Normalize(vmin=df.index.min(), vmax=df.index.max())
  cmap = cm.jet
  fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True, layout='constrained')
  panels = [('asym', r'asymptotic'),
            ('band', rf'${band_dex}$ dex below $\nu_{{\rm pk}}$')]
  for ax, (kind, txt) in zip(axs, panels):
    # reference values: one-zone fast cooling, line of death, GBM means
    ax.axhspan(-1.1, -0.8, color='0.85', zorder=0)
    ax.axhline(-2/3, c='0.4', ls=':', lw=1)
    ax.axhline(-3/2, c='0.4', ls=':', lw=1)
    ax.text(1.1e-2, -2/3 + 0.02, r'$-2/3$', color='0.4', fontsize=9)
    ax.text(1.1e-2, -3/2 + 0.02, r'$-3/2$', color='0.4', fontsize=9)
    ax.text(1.1e-2, -0.95, 'GBM', color='0.3', fontsize=9, va='center')
    for key in METHODS:
      x, y = df[f'eps_{key}'], df[f'alpha_{kind}_{key}']
      ax.plot(x, y, c='k', ls=LS[key], lw=1, zorder=1, label=LAB[key])
      ax.scatter(x, y, c=df.index, cmap=cmap, norm=norm, s=36, zorder=2,
                 edgecolors='k', linewidths=0.5,
                 marker='o' if key == 'full' else 's')
    ax.set_xscale('log')
    ax.set_xlim(1e-2, 1.5)
    ax.set_xlabel(r'$\epsilon_{\rm rad}$')
    ax.text(0.04, 0.04, txt, transform=ax.transAxes, va="bottom")
  axs[0].set_ylabel(r'low-energy photon index $\alpha$')
  axs[0].set_ylim(-1.85, -0.55)
  axs[0].legend(loc='lower right', frameon=False)
  fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs,
               label=r'$\log_{10}\mathcal{C}$')
  os.makedirs(outdir, exist_ok=True)
  path = os.path.join(outdir, fname)
  fig.savefig(path, dpi=300)
  df.round(4).to_csv(os.path.join(outdir, fname.replace('.png', '.csv')))
  return path, df


if __name__ == '__main__':
  path, df = plot_tradeoff()
  print(df.round(3).to_string())
  print(path)
