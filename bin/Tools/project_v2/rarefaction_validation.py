# -*- coding: utf-8 -*-
# @Author: acharlet

'''
Validation of the rarefaction-wave terminal event used to truncate the emitting
cell worldlines (compute_R_rar / worldline_from_cooling in working_cooling.py),
against the GAMMA simulation it post-processes. Two independent checks:

  validate_Rrar    - compares the model catch-up radius R_rar (compute_R_rar,
                     kinematic head-vs-cell worldlines on the fitted profiles) to
                     the true rarefaction arrival read off each cell's simulated
                     pressure crash; also reports the fitted-vs-simulation Lorentz
                     mismatch (post-shock settling steps) that biases the model.

  profile_collapse - tests the approximate self-similarity of the shocked layer
                     that the R_rar head-propagation proxy relies on, by
                     normalising each cell's raw profile to its own just-shocked
                     state and measuring the spread across cells at fixed R/R_i.
                     Includes the sound speed c_s (the quantity that actually
                     enters the head speed beta_h = (beta +/- c_s)/(1 +/- beta c_s)).

Run `python rarefaction_validation.py` to produce both sets of figures in
GAMMA/bin/Tools/figures/rar_validation, or import and call the functions with
custom cell lists / cuts.
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from environment import MyEnv
from phys_constants import c_
from phys_functions import derive_cs
from IO import get_variable, GAMMA_dir
from working_cooling import (open_rundata, open_celldata, cellsBehindShock_fromData,
    load_or_fit_celldata, compute_R_rar, check_extracted_cells,
    truncate_at_rarefaction, smooth_bpl_apy)

OUTDIR = os.path.join(GAMMA_dir, 'bin', 'Tools', 'figures', 'rar_validation')
FIT_VARS = ['rho', 'lfac', 'p']


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def shocked_after_crossing(cell_data):
  '''
  Post-shock portion of a cell's history: rows after the first Sd != 0 block
  (+1 for numerical settling), matching the selection in fit_celldata.
  '''
  sd = cell_data.Sd.to_numpy()
  ish = np.flatnonzero(sd != 0)
  if len(ish):
    after = np.flatnonzero(sd[ish[0]:] == 0)
    start = ish[0] + after[0] + 1 if len(after) else len(sd)
    return cell_data.iloc[start:].copy()
  return cell_data.loc[cell_data.Sd == 0].copy().iloc[1:]


def sim_rarefaction_radius(shocked, slope_thresh=-8., window=15, minpts=20):
  '''
  True (simulation) rarefaction-arrival radius: the first steep pressure crash
  in the post-shock history, same criterion as truncate_at_rarefaction.
  Returns (R_rar_cm, seen); seen=False if no crash is resolved (rarefaction not
  reached within the simulated duration -> censored).
  '''
  x = shocked.x.to_numpy(); p = shocked.p.to_numpy()
  if len(x) < minpts + window:
    return np.nan, False
  lnx, lnp = np.log(x), np.log(p)
  slope = (lnp[window:] - lnp[:-window]) / (lnx[window:] - lnx[:-window])
  steep = np.flatnonzero(slope < slope_thresh)
  if len(steep) and steep[0] >= minpts:
    return x[steep[0]] * c_, True
  return np.nan, False


def _rs_cell_list(key, z, ncells, ks):
  '''Default: ncells cell ids spread over shell z, restricted to extracted ones.'''
  if ks is not None:
    return list(ks)
  env = MyEnv(key)
  k4, kCD, k1 = env.Next, env.Next + env.Nsh4, env.Next + env.Nsh4 + env.Nsh1
  lo, hi = (k4, kCD) if z == 4 else (kCD, k1)
  extracted = set(check_extracted_cells(key).tolist())
  return [k for k in np.unique(np.linspace(lo, hi - 1, ncells).astype(int)) if k in extracted]


# ---------------------------------------------------------------------------
# 1. R_rar model vs simulation
# ---------------------------------------------------------------------------
def validate_Rrar(key='cooling_g100', z=4, ncells=80, ks=None, plot=True, outdir=OUTDIR):
  '''
  Compare compute_R_rar (model) to the simulated pressure-crash radius per cell,
  and report the fitted-vs-simulation Lorentz-factor mismatch over the
  pre-rarefaction range. Returns a DataFrame with one row per cell:
    k, Ri, Rsim, seen, Rpred (all radii in units of env.R0), dG_med, dG_max.
  Saves Rrar_pred_vs_sim.png and Rrar_residual_vs_position.png when plot=True.
  '''
  env = MyEnv(key)
  sh_data = cellsBehindShock_fromData(open_rundata(key, z))
  exit_row = sh_data.loc[sh_data.t.idxmax()]
  R0env = env.R0

  rows = []
  for k in _rs_cell_list(key, z, ncells, ks):
    cell_data = open_celldata(key, k)
    if cell_data is False:
      continue
    sel = sh_data.loc[sh_data.i == cell_data.iloc[0].i]
    if not len(sel):
      continue
    cell_d0 = sel.iloc[0]
    shocked = shocked_after_crossing(cell_data)
    if len(shocked) < 40:
      continue
    R_sim, seen = sim_rarefaction_radius(shocked)

    norms = [get_variable(cell_d0, n, env) for n in FIT_VARS]
    try:
      popts = load_or_fit_celldata(cell_data, FIT_VARS, norms, env, cell_d0.x, key=key, k=k)
    except Exception:
      continue
    R_pred = compute_R_rar(cell_d0, exit_row, env, popts)

    # fitted vs simulated Lorentz factor over the pre-rarefaction shocked range
    xcut = R_sim / c_ if seen else shocked.x.to_numpy()[-1]
    pre = shocked[shocked.x <= xcut]
    if len(pre) < 5:
      pre = shocked
    G_fit = get_variable(cell_d0, 'lfac', env) * smooth_bpl_apy(pre.x.to_numpy() / cell_d0.x, *popts[1])
    G_sim = get_variable(pre, 'lfac', env)
    dG = np.abs(G_fit - G_sim) / G_sim
    rows.append(dict(k=int(k), Ri=cell_d0.x * c_ / R0env,
                     Rsim=R_sim / R0env if seen else np.nan, seen=seen,
                     Rpred=(R_pred / R0env if np.isfinite(R_pred) else np.inf),
                     dG_med=float(np.median(dG)), dG_max=float(dG.max())))

  df = pd.DataFrame(rows)
  val = df[df.seen & np.isfinite(df.Rpred)]
  ratio = (val.Rpred / val.Rsim).to_numpy()
  print(f'[validate_Rrar] {len(val)}/{len(df)} cells with a resolved sim crash & finite prediction')
  if len(val):
    print(f'  R_pred/R_sim: median={np.median(ratio):.3f}  '
          f'16-84%=[{np.percentile(ratio,16):.3f}, {np.percentile(ratio,84):.3f}]')
  print(f'  fit-vs-sim Gamma mismatch (median over cells): {df.dG_med.median():.4f}')
  print(f'  censored (no crash in sim): {int((~df.seen).sum())}/{len(df)}')

  if plot and len(val):
    os.makedirs(outdir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    sc = ax.scatter(val.Rsim, val.Rpred, c=val.k, cmap='viridis', s=28, zorder=3)
    lim = [0.9, max(val.Rsim.max(), val.Rpred.max()) * 1.1]
    ax.plot(lim, lim, 'k--', lw=1, label='$y=x$')
    ax.set(xlim=lim, ylim=lim, xlabel='$R_{\\rm rar}^{\\rm sim}/R_0$ (pressure crash)',
           ylabel='$R_{\\rm rar}^{\\rm pred}/R_0$ (compute_R_rar)',
           title='Rarefaction catch-up radius: model vs simulation')
    ax.legend(); fig.colorbar(sc, ax=ax, label='cell index k (edge$\\to$CD)')
    fig.tight_layout(); fig.savefig(f'{outdir}/Rrar_pred_vs_sim.png', dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axhline(0, color='grey', lw=.8)
    ax.plot(val.k, ratio - 1, 'o-', color='C0', label='$R_{\\rm pred}/R_{\\rm sim}-1$')
    ax.set_xlabel('cell index k (outer edge $\\to$ CD)')
    ax.set_ylabel('relative residual', color='C0')
    ax2 = ax.twinx()
    ax2.plot(df.k, df.dG_med, 's--', color='C3', ms=4)
    ax2.set_ylabel('$|\\Gamma_{\\rm fit}-\\Gamma_{\\rm sim}|/\\Gamma_{\\rm sim}$', color='C3')
    ax.set_title('Residual and fitted-vs-simulation $\\Gamma$ mismatch across the shell')
    fig.tight_layout(); fig.savefig(f'{outdir}/Rrar_residual_vs_position.png', dpi=150); plt.close(fig)
    print(f'  figures -> {outdir}')
  return df


# ---------------------------------------------------------------------------
# 2. self-similarity / profile collapse (incl. sound speed)
# ---------------------------------------------------------------------------
def profile_collapse(key='cooling_g100', z=4, ncells=50, ks=None,
    xr_range=(1., 1.5), min_overlap=5, plot=True, outdir=OUTDIR):
  '''
  Test the approximate self-similarity of the shocked layer: normalise each
  cell's raw (pre-rarefaction) profile to its own just-shocked value and measure
  the spread across cells at fixed R/R_i. rho, lfac, p are shown as value/value_i;
  the sound speed c_s (which enters the head speed as an absolute velocity) is
  shown as c_s/c and its spread reported both fractionally and absolutely.
  Returns a dict {var: (median_frac_spread, max_frac_spread)} (plus 'cs_abs':
  std of c_s/c). Saves profile_collapse.png when plot=True.
  '''
  env = MyEnv(key)
  ks = _rs_cell_list(key, z, ncells, ks)

  prof = {v: [] for v in ('rho', 'lfac', 'p', 'cs')}
  kept = []
  for k in ks:
    cd = open_celldata(key, k)
    if cd is False:
      continue
    sh = truncate_at_rarefaction(shocked_after_crossing(cd))
    if len(sh) < 20:
      continue
    xr = sh.x.to_numpy() / sh.x.to_numpy()[0]                  # R / R_i
    rho = get_variable(sh, 'rho', env); p = get_variable(sh, 'p', env)
    cs = derive_cs(rho, p)                                      # absolute, in units of c
    kept.append(int(k))
    prof['rho'].append((xr, rho / rho[0]))
    prof['lfac'].append((xr, get_variable(sh, 'lfac', env) / get_variable(sh.iloc[[0]], 'lfac', env)[0]))
    prof['p'].append((xr, p / p[0]))
    prof['cs'].append((xr, cs))                                # NOT normalised: absolute c_s/c
  kept = np.array(kept)
  print(f'[profile_collapse] {len(kept)} cells, k in [{kept.min()}, {kept.max()}]')

  xg = np.geomspace(xr_range[0], xr_range[1], 40)
  def spread(v, normed=True):
    stack = []
    for xr, yr in prof[v]:
      m = (xg >= xr.min()) & (xg <= xr.max())
      yi = np.full_like(xg, np.nan)
      yi[m] = np.exp(np.interp(np.log(xg[m]), np.log(xr), np.log(yr)))
      stack.append(yi)
    Y = np.array(stack)
    n = np.sum(np.isfinite(Y), axis=0)
    frac = np.nanstd(Y, axis=0) / np.nanmedian(Y, axis=0)
    absol = np.nanstd(Y, axis=0)
    ok = n >= min_overlap
    return frac, absol, ok

  out = {}
  print('  fractional spread across cells (std/median), R/R_i in '
        f'[{xr_range[0]}, {xr_range[1]}]:')
  for v in ('rho', 'lfac', 'p', 'cs'):
    frac, absol, ok = spread(v)
    out[v] = (float(np.nanmedian(frac[ok])), float(np.nanmax(frac[ok])))
    extra = f'   (|abs c_s/c| std: median={np.nanmedian(absol[ok]):.3f})' if v == 'cs' else ''
    print(f'    {v:5s}: median={out[v][0]:.3f}  max={out[v][1]:.3f}{extra}')
    if v == 'cs':
      out['cs_abs'] = float(np.nanmedian(absol[ok]))

  if plot:
    os.makedirs(outdir, exist_ok=True)
    cmap = plt.cm.viridis; norm = plt.Normalize(kept.min(), kept.max())
    labels = {'rho': r'$\rho/\rho_i$', 'lfac': r'$\Gamma/\Gamma_i$',
              'p': r'$p/p_i$', 'cs': r'$c_s/c$ (absolute)'}
    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    for ax, v in zip(axs.ravel(), ('rho', 'lfac', 'p', 'cs')):
      for k, (xr, yr) in zip(kept, prof[v]):
        ax.loglog(xr, yr, color=cmap(norm(k)), lw=0.8, alpha=0.7)
      ax.set(xlabel='$R/R_i$', ylabel=labels[v], xlim=(1, 2.6))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, ax=axs, label='cell index k (outer edge $\\to$ CD)', pad=0.02)
    fig.suptitle('Shocked-layer profile collapse (raw simulation, pre-rarefaction)')
    fig.savefig(f'{outdir}/profile_collapse.png', dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f'  figure -> {outdir}/profile_collapse.png')
  return out


def main(key='cooling_g100', z=4):
  validate_Rrar(key, z)
  profile_collapse(key, z)


if __name__ == '__main__':
  main()
