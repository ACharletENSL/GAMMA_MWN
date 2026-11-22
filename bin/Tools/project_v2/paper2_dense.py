'''
Paper II, Figs. peak_tinteg_bkshift (Fig. 3) and tint_slope (Fig. 5) on the DENSE fiducial
grid (2026-11-22).

The 0.2-dex sweep of the rarefaction-cut spectra (hpc/sweep_dense.sh) lives in its own
folders, method_outdir('data_rarcut', key, z) + '_dense', together with copies of the
decade and half-decade points, so the per-regime figures that loop over the article caches
(Figs. 1, 2 and the like) keep their nine panels. This module reads those folders and draws:

  plot_fig3  peak and time-integrated spectra of the RS for a few log C (SHOW), and the
             peak/time-integrated ratios of the fitted breaks and of the nuFnu maximum over
             ALL points, as labelled curves;
  plot_fig5  eps_rad against log C, the time-integrated spectra of the RS and of the two
             shells for SHOW, and the photon indices against eps_rad over ALL points, as
             labelled curves.

Curves, not markers: at 0.2 dex the points are denser than a readable symbol spacing (the
argument sweep_efficiency.plot_efficiency_curve makes), and each curve carries its name
beside it instead of a legend. The spectra carry their log C at their peak.
'''
import os
import numpy as np
import matplotlib.pyplot as plt

import sweep_gammacm as swp
import sweep_shells as ssh
import spectrum_shape as ss
from environment import figdir

KEY = 'cooling_g100'
METHOD = 'data_rarcut'
SUFFIX = '_dense'
GRID_STEP = 0.2                   # the dense sweep's spacing (hpc/sweep_dense.sh)
SHOW = (-4., -2., 0., 2.)         # the spectra drawn (author, 2026-11-22)
CRANGE = (-5., 3.)                # colour scale of the whole sweep, as in Figs. 1-2
OUTDIR_NAME = 'paper2_dense'
LBL_FIXED = (0.19, -1.18, 'right')   # (eps_rad, alpha, ha) of the index curves' names in
LBL_BAND = (0.5, -1.48, 'center')    # Fig. 5, placed by eye in the gaps between curves


def dense_dir(z, key=KEY, method=METHOD):
  return swp.method_outdir(method, key, z).rstrip('/') + SUFFIX


def _color(logr, show=SHOW):
  # scaled on the drawn values, not on the whole sweep: four curves need the full contrast
  # of the map (on -5..3, log C = -4 and -2 came out as two near-identical purples)
  return swp.SWEEP_CMAP((logr - min(show))/(max(show) - min(show)))


def _clabel(logr):
  return f'${logr:+.0f}$'.replace('+0', '0')


def _label_cutoffs(ax, curves, yref, dx=1.8):
  '''log C written beside each curve's high-frequency cutoff, where it crosses yref, in
  the curve's colour. Not at the peaks: the fast-cooling curves peak at the same place.'''
  for x, y, logr in curves:
    j = np.nonzero(y >= yref)[0]
    if j.size:
      ax.text(x[j[-1]]*dx, yref, _clabel(logr), color=_color(logr), ha='left',
              va='center', fontsize=9)


def _on_curve(ax, x, y, x0, text, color, dy=0., ha='left', va='bottom', **kw):
  '''A curve's name, placed at x0 on the curve (data coordinates), offset by dy.'''
  ok = np.isfinite(y)
  y0 = np.interp(x0, np.asarray(x)[ok], np.asarray(y)[ok])
  ax.text(x0, y0 + dy, text, color=color, ha=ha, va=va, fontsize=9, **kw)


# ---------------------------------------------------------------------------------------
# Fig. 3
# ---------------------------------------------------------------------------------------
def measure_rs(key=KEY):
  '''spectrum_shape rows (peak and fluence) of every dense RS point, and the results.'''
  res = swp.load_sweep(dense_dir(4, key))
  return ss.measure_sweep(res, 4), res


def plot_fig3(rows=None, results=None, outdir=None, key=KEY, show=SHOW,
    fname='spectrum_shape_spectra_ratios_RS.png'):
  if rows is None:
    rows, results = measure_rs(key)
  outdir = figdir(OUTDIR_NAME, key) if outdir is None else outdir
  os.makedirs(outdir, exist_ok=True)
  sel = [r for r in results if any(abs(r['log10ratio'] - s) < 1e-6 for s in show)]
  fig, axs = plt.subplots(1, 3, figsize=(14.2, 4.3))
  peak_getter = swp._peak_getter(results)
  for ax, get_spec, sym in ((axs[0], peak_getter, '\\nu F_\\nu'),
                            (axs[1], swp._fluence_getter, '\\nu \\mathcal{F}_\\nu')):
    curves, ypk = [], []
    for r in sel:
      x = swp.nu_over_num(r)
      sp = get_spec(r)
      keep = x >= swp.phys_x_floor(r)
      x, sp = x[keep], sp[keep]
      y = sp/sp.max()*swp.compute_efficiency(r)
      curves.append((x, y, float(r['log10ratio'])))
      ypk.append(y.max())
    for x, y, lr in curves[::-1]:
      ax.loglog(x, y, color=_color(lr), lw=1.3)
    ylo = min(ypk)/10.**3.5
    ax.set_ylim(ylo, max(ypk)*3.)
    _label_cutoffs(ax, curves, ylo*10.**0.6)
    xhi = max(x[y > ylo].max() for x, y, _ in curves)
    ax.set_xlim(min(x[0] for x, _, _ in curves), 30.*xhi)
    ax.set_xlabel(swp.NU_M_LABEL)
    ax.set_ylabel(f'$\\varepsilon_{{\\rm rad}}\\,{sym}/({sym})_{{\\rm max}}$')

  # right panel: ratios over every point
  ax = axs[2]
  by = {}
  for m in rows:
    by.setdefault(m['kind'], {})[round(m['logr'], 4)] = m
  pk, fl = by['peak'], by['fluence']
  lr = np.array(sorted(set(pk) & set(fl)), float)
  mfc = [q for q in lr if pk[q]['cls'] == 'MC' or fl[q]['cls'] == 'MC']
  # half a grid step on each side; the gap tolerance bridges the half-decade points, which
  # sit between two 0.2-dex points and would otherwise halve the step and split the band
  ss._held_mid_band(ax, mfc, half=GRID_STEP/2., gap=1.5*GRID_STEP)
  def ratio(key):
    return np.array([pk[q][key]/fl[q][key] if np.isfinite(pk[q][key])
                     and np.isfinite(fl[q][key]) and fl[q][key] > 0. else np.nan
                     for q in lr], float)
  in_mfc = np.isin(lr, mfc)
  hi = np.where(in_mfc, np.nan, ratio('nu_hi_fit'))
  lo = np.where(in_mfc, np.nan, ratio('nu_lo_fit'))
  pkr = ratio('x_pk')
  ax.plot(lr, lo, color=ss._QCOL[1], lw=1.5)
  ax.plot(lr, hi, color=ss._QCOL[0], lw=1.5)
  ax.plot(lr, pkr, color='0.35', lw=1.2, ls='--')
  ax.set_xlabel(ss._CLABEL)
  ax.yaxis.tick_right()
  ax.yaxis.set_label_position('right')
  ax.set_ylabel('peak / time-integrated')
  ax.set_xlim(CRANGE[0] - 0.2, CRANGE[1] + 0.2)
  ax.grid(alpha=0.25)
  ax.set_ylim(1., max(np.nanmax(lo), np.nanmax(hi))*1.08)
  fig.tight_layout(w_pad=1.2)
  # names on the curves, once the limits are fixed
  _on_curve(ax, lr, lo, -3.6, 'lower break', ss._QCOL[1], dy=-0.15, va='top')
  _on_curve(ax, lr, hi, 1.1, 'upper break', ss._QCOL[0], dy=-0.07, va='top')
  _on_curve(ax, lr, pkr, -4.6, '$\\nu F_\\nu$ peak', '0.35', dy=-0.07, va='top')
  f = os.path.join(outdir, fname)
  fig.savefig(f, dpi=300, bbox_inches='tight')
  plt.close(fig)
  print(f'Fig. 3 -> {f}')
  return f


# ---------------------------------------------------------------------------------------
# Fig. 5
# ---------------------------------------------------------------------------------------
def shell_pairs(key=KEY):
  '''(RS, RS+FS) per dense point, as sweep_shells.plot_fluence_slope_shells builds them,
  with each shell's own eps_rad kept for the efficiency panel.'''
  od = ssh.method_outdir
  ssh.method_outdir = lambda m, k=None, z=4: od(m, k, z).rstrip('/') + SUFFIX
  try:
    P = list(ssh.load_shell_pairs(key, METHOD))
  finally:
    ssh.method_outdir = od
  pairs, eps = [], []
  for p in sorted(P, key=lambda p: p['log10ratio']):
    base = dict(log10ratio=p['log10ratio'], alpha=p['alpha'], env=p['env'], Tb=p['Tb'],
                nub=p['nub'])
    rs = dict(base, nuFnu=p['nuFnu_rs'], E_rad=p['E_rad_rs'], E_inj=p['E_inj_rs'])
    tot = dict(base, nuFnu=p['nuFnu_tot'], E_rad=p['E_rad_rs'] + p['E_rad_fs'],
               E_inj=p['E_inj_rs'] + p['E_inj_fs'])
    pairs.append((rs, tot))
    eps.append((p['E_rad_rs']/p['E_inj_rs'], p['E_rad_fs']/p['E_inj_fs'],
                tot['E_rad']/tot['E_inj']))
  return pairs, np.array(eps)


def index_table(pairs):
  '''Fixed-window and Band indices of the RS ('cut') and RS+FS ('full') time-integrated
  spectra, as in sweep_shells.plot_fluence_slope_shells.'''
  import sweep_compare as cmp
  import band_alpha as ba
  series = cmp.fluence_series(pairs, nu_break=cmp._total_nu_break)
  tab = cmp.fluence_slope_table(series, labels=('RS', 'RS+FS'), band_ref=2., verbose=False)
  fits = {k: ([], []) for k in ('band', 'a1', 'a2')}
  for (ra, _), e in zip(pairs, series):
    g = e['x'] >= swp.phys_x_floor(ra)
    for side, sp in enumerate((e['sp_a'], e['sp_b'])):
      fits['band'][side].append(ba.fit_band(e['x'][g], sp[g], 10., 100.)['alpha'])
      f2 = ba.fit_2sbpl(e['x'][g], sp[g], 10., 100.)     # CSV only, not drawn
      fits['a1'][side].append(f2['a1'])
      fits['a2'][side].append(f2['a2'])
  for k, (va, vb) in fits.items():
    tab[f'{k}_cut'], tab[f'{k}_full'] = va, vb
  return tab, series


def plot_fig5(pairs=None, eps=None, tab=None, outdir=None, key=KEY, show=SHOW,
    fname='fluence_spectra_slope_shells.png', p=2.5):
  if pairs is None:
    pairs, eps = shell_pairs(key)
  if tab is None:
    tab, _ = index_table(pairs)
  outdir = figdir(OUTDIR_NAME, key) if outdir is None else outdir
  os.makedirs(outdir, exist_ok=True)
  lr = np.array([a['log10ratio'] for a, _ in pairs], float)
  fig, (ax_e, ax_s, ax_i) = plt.subplots(1, 3, figsize=(15.6, 4.6),
                                         gridspec_kw={'width_ratios': [0.85, 1.1, 1.]})

  # (a) eps_rad against log C
  # the shells' colours and line styles of Fig. RS+FS (sweep_shells.STY)
  EPS_STY = ((0, 'RS'), (1, 'FS'), (2, 'tot'))
  for k, tag in EPS_STY:
    ax_e.semilogy(lr, eps[:, k], **{q: v for q, v in ssh.STY[tag].items() if q != 'label'})
  ax_e.axhline(1., color='grey', ls=':', lw=.8)
  ax_e.set_xlabel(ss._CLABEL)
  ax_e.set_ylabel('$\\varepsilon_{\\rm rad}$')
  ax_e.set_xlim(CRANGE[0], CRANGE[1] + 1.3)   # room for the curve names at the right end
  ax_e.set_xticks(np.arange(CRANGE[0] + 1., CRANGE[1] + 0.1, 2.))
  ax_e.grid(alpha=.25)

  # (b) time-integrated spectra for `show`
  sym = '\\nu \\mathcal{F}_\\nu'
  get = lambda r: swp.compute_fluence_spectrum(r['Tb'], r['nuFnu'])
  curves, ypk = [], []
  for (ra, rb) in pairs:
    if not any(abs(ra['log10ratio'] - s) < 1e-6 for s in show):
      continue
    x = swp.nu_over_num(ra)
    keep = x >= swp.phys_x_floor(ra)
    sa, sb = get(ra)[keep], get(rb)[keep]
    norm = sb.max()/swp.compute_efficiency(rb)
    curves.append((x[keep], sa/norm, sb/norm, float(ra['log10ratio'])))
    ypk.append((sb/norm).max())
  for x, ya, yb, l in curves[::-1]:
    ax_s.loglog(x, ya, color=_color(l), lw=1.2, ls='--')
    ax_s.loglog(x, yb, color=_color(l), lw=1.3)
  ylo = min(ypk)/10.**3.5
  ax_s.set_ylim(ylo, max(ypk)*3.)
  _label_cutoffs(ax_s, [(x, yb, l) for x, _, yb, l in curves], ylo*10.**0.6)
  ax_s.set_xlim(min(x[0] for x, *_ in curves),
                30.*max(x[yb > ylo].max() for x, _, yb, _ in curves))
  ax_s.set_xlabel(swp.NU_M_LABEL)
  ax_s.set_ylabel(f'$\\varepsilon_{{\\rm rad}}\\,{sym}/({sym})_{{\\rm max}}$')

  # (c) photon indices against eps_rad, every point
  cw, cb = 'tab:purple', 'tab:orange'
  for side, ls, k in (('cut', '--', 0), ('full', '-', 2)):
    ax_i.plot(eps[:, k], tab[f'aband2_{side}'] - 2., color=cw, ls=ls, lw=1.4)
    ax_i.plot(eps[:, k], tab[f'band_{side}'], color=cb, ls=ls, lw=1.4)
  for v, lab in ((-2./3., '$-2/3$'), (-1.5, '$-3/2$'), (-(p + 1.)/2., '$-(p+1)/2$')):
    ax_i.axhline(v, color='grey', ls=':', lw=.8)
    ax_i.text(0.011, v + 0.012, lab, color='grey', fontsize=8, va='bottom')
  ax_i.set_xscale('log')
  ax_i.set_xlim(1e-2, 1.6)
  ax_i.set_ylim(-1.9, -0.55)
  ax_i.set_xlabel('$\\varepsilon_{\\rm rad}$')
  ax_i.set_ylabel('$\\alpha$')
  from matplotlib.lines import Line2D
  ax_i.legend(handles=[Line2D([], [], color='k', ls='--', lw=1.2, label='RS'),
                       Line2D([], [], color='k', ls='-', lw=1.2, label='RS+FS')],
              fontsize=9, loc='upper left', bbox_to_anchor=(0., 0.93), frameon=False)
  fig.tight_layout(w_pad=1.0)
  # names on the curves
  for k, tag in EPS_STY:
    ax_e.text(CRANGE[1] + 0.1, eps[-1, k], 'RS+FS' if tag == 'tot' else tag,
              color=ssh.STY[tag]['color'], ha='left', va='center', fontsize=9)
  for (x, y, ha), text, c in ((LBL_FIXED, '$10^{-3}$–$10^{-2}\\,\\nu_{\\rm pk}$', cw),
                              (LBL_BAND, 'Band', cb)):
    ax_i.text(x, y, text, color=c, ha=ha, va='center', fontsize=9)
  f = os.path.join(outdir, fname)
  fig.savefig(f, dpi=300, bbox_inches='tight')
  plt.close(fig)
  tab = tab.copy()
  tab['eps_rs'], tab['eps_fs'], tab['eps_tot'] = eps[:, 0], eps[:, 1], eps[:, 2]
  tab.to_csv(f.replace('.png', '.csv'), index=False, float_format='%.5g')
  print(f'Fig. 5 -> {f}')
  return f, tab


def main(key=KEY):
  rows, res = measure_rs(key)
  f3 = plot_fig3(rows, res, key=key)
  pairs, eps = shell_pairs(key)
  f5, tab = plot_fig5(pairs, eps, key=key)
  return f3, f5, tab


if __name__ == '__main__':
  main()
