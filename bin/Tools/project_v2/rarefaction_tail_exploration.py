# -*- coding: utf-8 -*-
# @Author: acharlet

'''
Explore the raw post-rarefaction-catch-up hydro of shocked shell cells, using a run whose
stop condition was pushed well past the shell-wide rarefaction convergence (the fiducial
results/cooling_g100 runs to bar{T} ~ 650, ~400x past the last cell's R_rar, so even the
last-crashed, outermost cells get real cushion past their own).
Purpose: decide how to model a cell's hydro/emission after the rarefaction wave catches it
(currently a hard cutoff at R_rar, see working_cooling.generate_cell_withDistrib) -- is the
post-crash decline universal across the shell (supports one shared decay law) or position
dependent (needs a per-cell/position-dependent one)?
Example use:
  python -c "import rarefaction_tail_exploration as X; X.main()"
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from environment import MyEnv, GAMMA_dir
from phys_constants import c_
from phys_functions import derive_Lorentz
from working_cooling import (cellsBehindShock_fromData, open_rundata, open_celldata,
    load_shell_rarefaction, truncate_at_rarefaction)

KEY = 'cooling_g100'
Z_SHELL = 4
OUTDIR = GAMMA_dir + '/bin/Tools/figures/rarefaction_tail_exploration/'
# spread of cells across the shell, weighted toward the outer (last-crashed, previously
# uncovered) end -- Next=20 offsets the shell's own index range [20, 519]
CELLS = [25, 50, 100, 150, 200, 250, 300, 350, 400, 440, 460, 480, 495, 505, 512, 519]
SLOPE_WINDOW = 15   # same look-ahead window as truncate_at_rarefaction, for consistency


def shocked_tail(key, k):
  '''
  Full post-shock history of cell k (from shock onset to end of run), same onset
  selection as fit_celldata (working_cooling.py) but WITHOUT truncating at the
  rarefaction: we want to see past it here.
  '''
  cd = open_celldata(key, k)
  if cd is False:
    return None
  sd = cd.Sd.to_numpy()
  ish = np.flatnonzero(sd != 0)
  if len(ish):
    after = np.flatnonzero(sd[ish[0]:] == 0)
    start = ish[0] + after[0] + 1 if len(after) else len(sd)
    return cd.iloc[start:].copy()
  return cd.loc[(cd.Sd == 0)].copy().iloc[1:]


def loglog_slope(x, y, window=SLOPE_WINDOW):
  '''Local look-ahead log-log slope of y(x), same convention as truncate_at_rarefaction.'''
  lnx, lny = np.log(x), np.log(y)
  n = len(x)
  if n <= window:
    return np.full(n, np.nan)
  s = (lny[window:] - lny[:-window]) / (lnx[window:] - lnx[:-window])
  return np.concatenate([np.full(window, np.nan), s])


def explore(key=KEY, z=Z_SHELL, cells=CELLS, outdir=OUTDIR):
  import os
  os.makedirs(outdir, exist_ok=True)
  env = MyEnv(key)
  sh = cellsBehindShock_fromData(open_rundata(key, z))
  rar_map = load_shell_rarefaction(key, z, env, n_shell=len(sh))

  cmap = plt.cm.viridis
  norm = plt.Normalize(vmin=min(cells), vmax=max(cells))

  fig, (ax_rho, ax_p, ax_lfac, ax_slope) = plt.subplots(4, 1, figsize=(7, 12), sharex=True)
  rows = []
  for k in cells:
    if k not in rar_map or not np.isfinite(rar_map[k]):
      continue
    data = shocked_tail(key, k)
    if data is None or len(data) < 2:
      continue
    x0 = data.x.iloc[0]
    R_over_R0 = data.x.to_numpy() / x0
    R_over_Rrar = R_over_R0 / rar_map[k]
    rho = data.rho.to_numpy()
    p = data.p.to_numpy()
    lfac = derive_Lorentz(data.vx.to_numpy())
    col = cmap(norm(k))

    ax_rho.loglog(R_over_Rrar, rho / rho[0], color=col, lw=1.1)
    ax_p.loglog(R_over_Rrar, p / p[0], color=col, lw=1.1)
    ax_lfac.loglog(R_over_Rrar, lfac / lfac[0], color=col, lw=1.1)
    s_p = loglog_slope(R_over_R0, p)
    ax_slope.plot(R_over_Rrar, s_p, color=col, lw=1.1, label=f'k={k}')

    # characterise the post-crash regime: mean p log-slope over R/R_rar in [2.0, 3.0] --
    # far enough past the crash transient to probe the asymptote rather than the still-
    # relaxing region right after it (or as much of that window as the cushion covers)
    post = (R_over_Rrar >= 2.0) & (R_over_Rrar <= 3.0) & np.isfinite(s_p)
    max_cushion = R_over_R0[-1] / rar_map[k] - 1.
    rows.append(dict(k=k, R_rar_over_R0=rar_map[k], max_cushion=max_cushion,
                      n_post=int(post.sum()),
                      slope_post_mean=float(np.mean(s_p[post])) if post.any() else np.nan))

  for ax, ylab in ((ax_rho, r"$\rho/\rho(R_{\rm rar})$"), (ax_p, r"$p/p(R_{\rm rar})$"),
                   (ax_lfac, r"$\Gamma/\Gamma(R_{\rm rar})$")):
    ax.axvline(1., color='grey', ls=':', lw=.9)
    ax.set_ylabel(ylab)
  ax_slope.axvline(1., color='grey', ls=':', lw=.9)
  ax_slope.axhline(-2., color='grey', ls='--', lw=.6)
  ax_slope.axhline(-8., color='grey', ls='--', lw=.6)
  ax_slope.set_ylabel(r'$d\ln p/d\ln R$')
  ax_slope.set_xlabel(r'$R/R_{\rm rar}$')
  ax_slope.set_ylim(-15, 2)
  ax_slope.legend(fontsize=6, ncol=4, loc='lower left')
  sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
  fig.colorbar(sm, ax=[ax_rho, ax_p, ax_lfac, ax_slope], label='cell index k', pad=0.02)
  fig.suptitle(f'{key}: raw post-shock hydro vs $R/R_{{\\rm rar}}$ (grey dashed: slope $-2,-8$)')
  fig.savefig(outdir + 'post_crash_profiles.png', dpi=200, bbox_inches='tight')
  plt.close(fig)

  df = pd.DataFrame(rows).sort_values('k')
  print(df.to_string(index=False))
  df.to_csv(outdir + 'post_crash_slopes.csv', index=False)
  print(f'\nfigure -> {outdir}post_crash_profiles.png')
  print(f'table  -> {outdir}post_crash_slopes.csv')
  return df


def main():
  explore()


if __name__ == '__main__':
  main()
