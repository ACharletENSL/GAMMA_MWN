# -*- coding: utf-8 -*-
# @Author: acharlet

'''
Snapshots of the whole collision (rho, Gamma, p) at one time per hydrodynamical phase: the
film strip that goes with the per-cell phases of prerar_model.plot_article (post-shock /
rarefaction / post-rarefaction / trailing shock).

One ROW per time, read down the page; columns rho, Gamma, p. The abscissa is the distance to
the CD in units of the initial width of shell 4, so the two shells keep a readable size while
the system moves out by three decades in R. Shell 4 is red and shell 1 blue (COL_RS/COL_FS),
the external medium grey; cells flagged by the shock detector (Sd) are black dots.

Times are lab time since the collision (the dumps' t). On the fiducial (cooling_g100): RS
leaves shell 4 at ~4.0e4 s, FS leaves shell 1 at ~4.4e4 s, both rarefaction heads reach the CD
at ~5.6-6e4 s, the trailing compression steepens into a shock at ~4.5e5 s and reaches the CD
at ~3.4e6 s (see prerar_model.plot_trailing_shock_strip).

  python hydro_phases.py [key]          ~20 s, single core
'''

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

from environment import MyEnv, GAMMA_dir
from phys_constants import c_
from plotting_functions import COL_RS, COL_FS
from shell_cells import shell_cell_range
from prerar_model import _snap_index, _sci
import prerar_cell_evolution as P

OUTDIR = os.path.join(GAMMA_dir, 'bin', 'Tools', 'figures', 'hydro_phases')

PHASE_SNAPS = (
  (0.,     'initial state'),
  (2.0e4,  'RS and FS propagate'),
  (4.2e4,  'RS out, FS still in S1'),
  (5.0e4,  'rarefactions in both shells'),
  (1.0e5,  'post-rarefaction'),
  (3.0e5,  'trailing compression'),
  (1.2e6,  'trailing shock in S4'),
  (5.6e6,  'shock enters S1'),
)
COL_EXT = '0.6'
X_PAD = 0.15     # fraction of the two shells' extent added either side (external medium)


def load_snapshot(key, t, its=None, ts=None):
  '''Whole-domain dump nearest to t, with Gamma, R/R0, the region label and xi = (R-R_CD)/Delta0.'''
  env = MyEnv(key)
  if its is None:
    its, ts = _snap_index(key)
  n = int(np.argmin(np.abs(ts - t)))
  d = pd.read_csv(os.path.join(GAMMA_dir, 'results', key, f'phys{its[n]:010d}.out'), sep=' ',
                  usecols=['t', 'i', 'x', 'dx', 'rho', 'vx', 'p', 'Sd']).set_index('i')
  k4, _, kCD4 = shell_cell_range(key, 4)
  _, k1, _ = shell_cell_range(key, 1)
  d['G'] = 1./np.sqrt(1. - d.vx**2)
  d['zone'] = np.where(d.index < k4, 'ext', np.where(d.index <= kCD4, 'S4',
                       np.where(d.index < k1, 'S1', 'ext')))
  Rcd = d.x.loc[kCD4] + 0.5*d.dx.loc[kCD4]
  D0 = env.D04/c_                       # initial width of shell 4, code units (light-s)
  d['xi'] = (d.x - Rcd)/D0
  d.attrs.update(t=float(d.t.iloc[0]), LR=float(np.log10(Rcd*c_/env.R0)),
                 xlo=float((d.x.loc[k4] - 0.5*d.dx.loc[k4] - Rcd)/D0),
                 xhi=float((d.x.loc[k1 - 1] + 0.5*d.dx.loc[k1 - 1] - Rcd)/D0))
  return d


def plot_phase_snapshots(key='cooling_g100', snaps=PHASE_SNAPS, outdir=OUTDIR):
  os.makedirs(outdir, exist_ok=True)
  its, ts = _snap_index(key)
  nrow = len(snaps)
  fig, axes = plt.subplots(nrow, 3, figsize=(11., 1.55*nrow + 0.5), sharex='row',
                           layout='constrained')
  cols = {'S4': COL_RS, 'S1': COL_FS}
  for r, (t, tag) in enumerate(snaps):
    d = load_snapshot(key, t, its, ts)
    a0, a1 = d.attrs['xlo'], d.attrs['xhi']
    pad = X_PAD*(a1 - a0)
    view = d[(d.xi > a0 - pad) & (d.xi < a1 + pad)]
    for c, q in enumerate(('rho', 'G', 'p')):
      ax = axes[r, c]
      # external medium drawn per side so the line does not jump across the shells
      for side in (view[view.index < view[view.zone == 'S4'].index.min()],
                   view[view.index > view[view.zone == 'S1'].index.max()]):
        ax.plot(side.xi, side[q], color=COL_EXT, lw=1.0)
      for z in ('S4', 'S1'):
        s = view[view.zone == z]
        ax.plot(s.xi, s[q], color=cols[z], lw=1.5)
      sh = view[view.Sd > 0]
      ax.plot(sh.xi, sh[q], 'o', ms=2.5, color='k', zorder=4)
      ax.axvline(0., color='0.6', lw=0.8, ls=(0, (4, 2)), zorder=1)
      if q != 'G':
        ax.set_yscale('log')
      ax.set_xlim(a0 - pad, a1 + pad)
      ax.tick_params(labelsize=8)
      P._style(ax)
    lab = 't = 0' if d.attrs['t'] == 0 else rf't = {_sci(d.attrs["t"])}\,{{\rm s}}'
    LR = d.attrs['LR'] if abs(d.attrs['LR']) > 5e-3 else 0.
    axes[r, 0].set_title(rf'$\bf{{{chr(97 + r)}}}$   {tag}:   ${lab}$,  '
                         rf'$\log_{{10}}(R_{{\rm CD}}/R_0) = {LR:.2f}$',
                         loc='left', fontsize=9)
    for c, ylab in enumerate((r'$\rho$ [code]', r'$\Gamma$', r'$p$ [code]')):
      axes[r, c].set_ylabel(ylab, fontsize=9)
  for ax in axes[:, (0, 2)].ravel():
    # label a subset of the minor ticks when the panel spans about a decade or less
    ax.yaxis.set_minor_formatter(
        matplotlib.ticker.LogFormatterSciNotation(minor_thresholds=(1.1, 0.4)))
  for c in range(3):
    axes[-1, c].set_xlabel(r'$(R - R_{\rm CD})/\Delta_{4,0}$')
  path = os.path.join(outdir, f'phase_snapshots_{key}.png')
  fig.savefig(path, dpi=200, bbox_inches='tight')
  plt.close(fig)
  print(f'saved {path}')
  return path


if __name__ == '__main__':
  plot_phase_snapshots(*sys.argv[1:2])
