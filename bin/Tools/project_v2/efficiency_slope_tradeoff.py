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
from matplotlib.lines import Line2D

FIGROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figures', 'fiducial')
EFF_CSV = os.path.join(FIGROOT, 'efficiency_sweep', 'efficiency_table.csv')
SLOPE_CSV = os.path.join(FIGROOT, 'rarcut_compare', 'fluence_low_slopes.csv')
OUTDIR = os.path.join(FIGROOT, 'rarcut_compare')

METHODS = {'full': 'data', 'cut': 'data_rarcut'}
LS = {'full': '-', 'cut': '--'}
LAB = {'full': 'full', 'cut': 'rf cut'}
# line style carries the rarefaction treatment, marker the slope estimator
MK = {'asym': 'o', 'band': 'D'}


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
  est = {'asym': 'asymptotic', 'band': rf'${band_dex}$ dex below $\nu_{{\rm pk}}$'}
  fig, ax = plt.subplots(figsize=(6, 4.5), layout='constrained')
  # reference values: line of death, one-zone fast cooling
  for a, lab in ((-2/3, r'$-2/3$'), (-3/2, r'$-3/2$')):
    ax.axhline(a, c='0.4', ls=':', lw=1)
    ax.text(1.1e-2, a + 0.02, lab, color='0.4', fontsize=9)
  for kind in est:
    for key in METHODS:
      x, y = df[f'eps_{key}'], df[f'alpha_{kind}_{key}']
      ax.plot(x, y, c='k', ls=LS[key], lw=1, zorder=1)
      ax.scatter(x, y, c=df.index, cmap=cmap, norm=norm, s=36, zorder=2,
                 edgecolors='k', linewidths=0.5, marker=MK[kind])
  ax.set_xscale('log')
  ax.set_xlim(1e-2, 1.5)
  ax.set_ylim(-1.85, -0.55)
  ax.set_xlabel(r'$\epsilon_{\rm rad}$')
  ax.set_ylabel(r'low-energy photon index $\alpha$')
  handles = [Line2D([], [], c='k', ls=LS[key], lw=1, label=LAB[key]) for key in METHODS]
  handles += [Line2D([], [], ls='', marker=MK[kind], mfc='0.7', mec='k', mew=0.5,
                     ms=6, label=est[kind]) for kind in est]
  ax.legend(handles=handles, loc='center left', frameon=False)
  fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
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
