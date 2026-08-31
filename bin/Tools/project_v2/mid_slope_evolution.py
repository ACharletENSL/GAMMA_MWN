# -*- coding: utf-8 -*-
# @Author: acharlet

'''
Where does the MID segment of the spectrum actually sit, and when?

sweep_gammacm.identify_segments holds each candidate slope at its one-zone asymptote
(1/2 fast-cooling, (3-p)/2 slow) because that is what makes a segment identifiable, and
reports how far the spectrum then departs from it (dep = a_core - a). That departure is
a RESULT, not a fitting residual: the shell-integrated fast-cooling segment is hardened
by the cell-to-cell spread in nu_c, by up to +0.098 as gamma_c climbs toward gamma_m,
while slow cooling softens by at most -0.034 (see SEG_FC_TOL_HI). This module tracks it
over the whole pulse, one measurement per sampled observer time, so the hardening can be
read against the hydro clock rather than at three phases.

Two ways of measuring the same segment, and the figure distinguishes them:
  SOLID   a_core -- the slope where the segment SETTLES (flat_core), i.e. the value it
          holds over its flat interior. Defined only where such an interior exists.
  HOLLOW  a_win -- a free straight-line fit over the whole identified window, used where
          no settled core exists. The slope is sweeping through the window there, so the
          number is an average over a knee and must not be read as a segment slope.

Read the time axis as the shell's, not the spectrum's: bar{T}/bar{T}_f = 1 is the shell
crossing and the grey band is the rarefaction switching the shell off, its right edge
bar{T}_rf = where emission stops everywhere (rarefaction_off_barT).

The measurement is on the CACHED sweep points, not a recomputation of the emission, so
it costs one identify_segments per sampled step and nothing else. Rows are written to
mid_slopes.csv beside the figure and reused unless use_cache=False -- the sampling is
dense enough (2800 rows on the fiducial) that re-measuring for a colour change is waste.

Example use in command line:
  python -c "import mid_slope_evolution as M; M.main()"
  python -c "import mid_slope_evolution as M; M.main(method='data')"
  python -c "import mid_slope_evolution as M; M.main(use_cache=False)"
'''

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from sweep_gammacm import (DEFAULT_KEY, Z_SHELL, load_sweep, method_outdir,
    exit_onset_barT, rarefaction_off_barT, identify_segments, trim_pngs,
    copy_article_figures)

METHOD = 'data_rarcut'        # the sweep this figure is read from: the cut is the
                              # prescription the article quotes (see sweep_rarcut)
CSV_NAME = 'mid_slopes.csv'
FIG_NAME = 'mid_slope_evolution.png'
STEP_FINE, STEP_COARSE = 2, 10     # sample every Nth observer step, fine early where the
X_FINE = 0.05                      # excursion lives (x < X_FINE) and coarser afterwards:
                                   # the late slope drifts by <0.002 per decade, so a
                                   # denser late sampling buys nothing but runtime
FLUX_FLOOR = 1e-10                 # skip steps whose peak flux is this far below the
                                   # brightest: the spectrum there is numerical noise
# Diverging assignment on logr: blue = fast cooling, neutral grey at logr=0 (gamma_c=gamma_m,
# the marginal point), red = slow. One global map, so logr=-3 is the same colour in both
# panels -- colour follows the entity, not its rank within a panel. NB this is NOT the jet
# colorbar the sweep figures use; there is no colorbar here, the curves are labelled.
COL = {-5: '#08306b', -4: '#1f6cb0', -3: '#4393c3', -2: '#7fb8d8',
       -1: '#a8cfe3', 0: '#737373', 1: '#ef6548', 2: '#a50f15'}
INK, MUTED, GRID = '#1a1a1a', '#6b6b6b', '#d9d9d9'
FIELDS = ('logr', 'barT', 'x', 'step', 'branch', 'a_th', 'a_core', 'dep', 'core', 'dex',
          'a_win', 'regime')


def measure(key=DEFAULT_KEY, method=METHOD, z=Z_SHELL, verbose=True):
  '''
  The mid segment of every sampled spectrum of every sweep point: its held asymptote
  a_th, the settled slope a_core and their difference dep, the free-window slope a_win,
  and the shape class the whole spectrum falls in. One row per (sweep point, step,
  branch), where branch is 'fc' or 'sc' -- the two mid candidates. A spectrum has at
  most one of them (identify_segments drops the narrower where both linger), so the two
  branches never describe the same step.
  '''
  outdir = method_outdir(method, key, z)
  results = load_sweep(outdir)
  if not results:
    raise RuntimeError(f'no cached sweep in {outdir}; run sweep_gammacm.run_sweep first')
  barT_f = exit_onset_barT(key, z=z)
  rows = []
  for r in results:
    nub, nuFnu, p = r['nub'], r['nuFnu'], r['env'].psyn
    barT = np.asarray(r['Tb'], float) - 1.
    x = barT/barT_f
    Fpk = np.nanmax(nuFnu, axis=1)
    bright = Fpk > FLUX_FLOOR*np.nanmax(Fpk)
    step = np.where(x < X_FINE, STEP_FINE, STEP_COARSE)
    for i in range(len(nuFnu)):
      if not bright[i] or i % int(step[i]):
        continue
      d = identify_segments(nub, nuFnu[i], p)
      if not d:
        continue
      for br in ('fc', 'sc'):
        if br not in d['segs']:
          continue
        g = d['segs'][br]
        # the free fit over the identified window, for the steps with no settled core.
        # identify_segments already computes it (a_fit, the line the mid segments are DRAWN
        # with); refitting it here was one extra segment_slopes call per row for a
        # bit-identical number.
        a_win = g['a_fit']
        rows.append(dict(logr=r['log10ratio'], barT=barT[i], x=x[i], step=i, branch=br,
                         a_th=g['a'], a_core=g['a_core'], dep=g['dep'], core=g['core'],
                         dex=g['dex'], a_win=a_win, regime=d['regime'] or 'None'))
    if verbose:
      print(f"  logr={r['log10ratio']:+.1f} done ({len(rows)} rows)", flush=True)
  return rows


def write_rows(rows, outdir, fname=CSV_NAME):
  with open(os.path.join(outdir, fname), 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(FIELDS))
    w.writeheader(); w.writerows(rows)
  return os.path.join(outdir, fname)


def read_rows(outdir, fname=CSV_NAME):
  '''Rows back from the csv, or None if it is not there.'''
  path = os.path.join(outdir, fname)
  if not os.path.isfile(path):
    return None
  rows = []
  for r in csv.DictReader(open(path)):
    for k in FIELDS:
      if k not in ('branch', 'regime'):
        r[k] = float(r[k])
    rows.append(r)
  return rows


def plot(rows, outdir, barT_f, barT_off=None, fname=FIG_NAME):
  '''
  Two panels, one per branch, on a shared log time axis. Each holds its asymptote as a
  dashed guide and the sweep points as one curve per log10(gma_c/gma_m). Curves are
  labelled at their right end where they are far enough apart to be told apart, and in
  the legend always.
  barT_off shades the rarefaction band; no line is drawn at its right edge (the band's
  own edge IS bar{T}_rf -- see sweep_gammacm.plot_lightcurve_shape).
  '''
  xoff = tuple(b/barT_f for b in barT_off) if (barT_off and barT_f > 0.) else None
  fig, axes = plt.subplots(2, 1, figsize=(8.4, 7.6), sharex=True)
  panels = (('fc', 0.5, 'Fast-cooling branch', '$a_{\\rm th}=1/2$', (0.478, 0.615)),
            ('sc', 0.25, 'Slow-cooling branch', '$a_{\\rm th}=(3-p)/2$', (0.205, 0.272)))
  for ax, (br, a_th, title, a_lab, ylim) in zip(axes, panels):
    sub = [r for r in rows if r['branch'] == br]
    if xoff is not None:
      ax.axvspan(xoff[0], xoff[1], color='grey', alpha=0.15, lw=0, zorder=0)
    ax.axvline(1., color='grey', ls=':', lw=0.9, zorder=1)
    ax.axhline(a_th, color=MUTED, lw=1.2, ls='--', zorder=1)
    ax.annotate(a_lab, xy=(1., a_th), xycoords=('axes fraction', 'data'),
                xytext=(-4, 5), textcoords='offset points', ha='right', va='bottom',
                fontsize=9, color=MUTED)
    for lr in sorted({r['logr'] for r in sub}):
      d = sorted([r for r in sub if r['logr'] == lr], key=lambda r: r['x'])
      c = COL[int(lr)]
      settled = [r for r in d if np.isfinite(r['a_core'])]
      sweeping = [r for r in d if not np.isfinite(r['a_core'])]
      if settled:
        ax.plot([r['x'] for r in settled], [r['a_core'] for r in settled], '-', color=c,
                lw=1.8, solid_capstyle='round', zorder=3, label=f'{lr:+.0f}')
      if sweeping:   # no settled core: the slope sweeps. Free window fit, hollow.
        ax.plot([r['x'] for r in sweeping], [r['a_win'] for r in sweeping], 'o', mfc='none',
                mec=c, ms=3.6, mew=1.0, ls='none', zorder=2,
                label=None if settled else f'{lr:+.0f}')
    ax.set_xscale('log'); ax.set_ylim(*ylim)
    ax.grid(True, which='major', color=GRID, lw=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
      ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
      ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_ylabel('measured mid slope', color=INK, fontsize=10)
    ax.set_title(title, color=INK, fontsize=11, loc='left', pad=6)
    leg = ax.legend(title='$\\log_{10}(\\gamma_c/\\gamma_m)$', fontsize=8.5,
                    title_fontsize=8.5, ncol=2, loc='upper left', frameon=True,
                    framealpha=0.92, edgecolor=GRID)
    leg.get_title().set_color(MUTED)
    for t in leg.get_texts():
      t.set_color(INK)
  # end-of-curve labels, but only where the ends are resolved: two curves ending within
  # 5% of the panel height of each other get neither, since the label could not be
  # attributed. Short tracks (<25 settled samples) are skipped -- they end mid-panel.
  for ax, (br, _a, _t, _l, ylim) in zip(axes, panels):
    sub = [r for r in rows if r['branch'] == br]
    ends = []
    for lr in sorted({r['logr'] for r in sub}):
      d = sorted([r for r in sub if r['logr'] == lr and np.isfinite(r['a_core'])],
                 key=lambda r: r['x'])
      if len(d) >= 25:
        ends.append((lr, d[-1]['x'], d[-1]['a_core']))
    span = ylim[1] - ylim[0]
    for lr, bx, by in ends:
      if any(abs(by - o[2]) < 0.05*span for o in ends if o[0] != lr):
        continue
      ax.annotate(f'{lr:+.0f}', xy=(bx, by), xytext=(5, 0), textcoords='offset points',
                  va='center', ha='left', fontsize=8.5, color=COL[int(lr)],
                  fontweight='bold', clip_on=False)
  axes[1].annotate('crossing', xy=(1., 0.), xycoords=('data', 'axes fraction'),
                   xytext=(-4, 6), textcoords='offset points', ha='right', va='bottom',
                   fontsize=8.5, color=MUTED)
  axes[1].set_xlabel('$\\bar{T}/\\bar{T}_f$', color=INK, fontsize=10)
  fig.suptitle('Measured mid-segment slope vs cooling regime and time',
               color=INK, fontsize=12.5, x=0.055, ha='left', y=0.985)
  fig.text(0.055, 0.938, 'Solid: slope where the segment settles (flat_core).  '
           'Hollow: no settled core - free fit over the identified window.',
           color=MUTED, fontsize=9, ha='left')
  fig.tight_layout(rect=[0, 0, 1, 0.925])
  path = os.path.join(outdir, fname)
  fig.savefig(path, dpi=200, facecolor='white')
  plt.close(fig)
  return path


def main(key=DEFAULT_KEY, method=METHOD, z=Z_SHELL, outdir=None, use_cache=True):
  '''
  Measure (or reload) the mid-segment slopes of a cached sweep and draw the figure into
  that sweep's own directory, beside the spectra it is measured from.
  '''
  outdir = method_outdir(method, key, z) if outdir is None else outdir
  os.makedirs(outdir, exist_ok=True)
  rows = read_rows(outdir) if use_cache else None
  if rows:
    print(f'{len(rows)} rows reloaded from {os.path.join(outdir, CSV_NAME)}')
  else:
    print(f'--- measuring the mid segment on {key}, {method}, z={z} ---')
    rows = measure(key, method, z)
    print(f'{len(rows)} rows -> {write_rows(rows, outdir)}')
  barT_f = exit_onset_barT(key, z=z)
  path = plot(rows, outdir, barT_f, rarefaction_off_barT(key, z=z))
  trim_pngs(outdir)
  copy_article_figures(outdir)
  print(f'-> {path}')
  return rows


if __name__ == '__main__':
  main()
