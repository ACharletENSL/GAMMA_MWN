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

The measured quantity is a_mid: the slope the mid segment of the SHELL-INTEGRATED
spectrum actually shows. a_th is the one-zone asymptote it is held at while the segment
is being identified, and dep = a_mid - a_th is the departure described above. The two
names are not interchangeable -- a_th is an input, a_mid the measurement.

WHAT IS PLOTTED is what the spectral fit actually uses: a_mid from
spectral_breaks.breaks_from_identified, the mid line re-measured over a window centred on
its own slope, which is the beta_mid fit_smoothing_held then holds. The two window-bounded
estimators it replaced are still written to the csv beside it -- a_core (the settled slope
inside the identified window) and a_win (a free line across all of it) -- so the change of
estimator can be read off the same rows. Both inherit the identified window's asymmetry and
return +0.036 (fc) / -0.035 (sc) on synthetics whose mid slope IS the asymptote; a_mid does
not, and what remains of its own bias is calibrated in segment_route.mid_bias_calibration.
  SOLID   the window re-centred successfully -- the value the fit stands on.
  HOLLOW  it did not, and the fit fell back to the line over the identified window.

The MARGINAL class is measured (under both the shape and the physical bounds on a_mid) and
written to the csv, but is NOT drawn. Bounded to what a fused fast- or slow-cooling knee can
produce, [(3-p)/2 - 0.10, 1/2 + 0.10], its fitted mid slope pins at the ceiling in 83% of
bins and at the floor in none -- so there is no track to plot. That pinning is a statement
about which shape those spectra want, not a slope, and belongs in the text.

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
from matplotlib.lines import Line2D

from sweep_gammacm import (DEFAULT_KEY, Z_SHELL, load_sweep, method_outdir,
    exit_onset_barT, rarefaction_off_barT, trim_pngs, copy_article_figures)
import spectral_breaks as sb
import cell_pool

METHOD = 'data_rarcut'        # the sweep this figure is read from: the cut is the
                              # prescription the article quotes (see sweep_rarcut)
CSV_NAME = 'mid_slopes.csv'
FIG_NAME = 'mid_slope_evolution.png'
FIG_NAME_RAW = 'mid_slope_evolution_raw.png'   # the same tracks, bias NOT subtracted
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
       -1: '#a8cfe3', 0: '#737373', 1: '#ef6548', 2: '#a50f15', 3: '#67000d'}


def col(lr):
  '''
  COL for one log10ratio, falling back to the darkest end of the map instead of raising.
  The dict is written out so a regime keeps its colour across figures, but it was indexed
  raw -- so extending sweep_gammacm.LOG10RATIO_ARR to +3 made this module die with
  KeyError: 3 after 17 min of work, and took mid_slope_evolution.png (an ARTICLE_SERIES
  glob) out of BOTH runs' figure sets without anything noticing. A missing colour must
  never cost a figure.
  '''
  i = int(lr)
  if i in COL:
    return COL[i]
  lo, hi = min(COL), max(COL)
  return COL[lo if i < lo else hi]
INK, MUTED, GRID = '#1a1a1a', '#6b6b6b', '#d9d9d9'
FIELDS = ('logr', 'barT', 'x', 'step', 'branch', 'a_th', 'a_mid', 'dep', 'mid_from',
          'dex_mid', 'a_core', 'dep_core', 'core', 'dex', 'a_win', 'regime',
          'a_mid_phys', 'rms', 'rms_phys', 'phys_at_bound', 'sep', 's1', 's2', 'depth')
NPROC = 4                          # the route costs ~0.8 s a step (the smeared cut-off scan
                                   # dominates), so the points are run in parallel -- but on
                                   # a laptop, so this is deliberately not ncpu-1


def _measure_point(args):
  '''
  One sweep point, in a worker: the mid slope THE ROUTE ACTUALLY USES on every sampled
  spectrum. That is spectral_breaks.breaks_from_identified's a_mid -- the line re-measured
  over a window centred on its own slope, which is the beta_mid fit_smoothing_held then
  holds. The identification (and so the branch and the class) is unchanged; what is recorded
  is the slope the spectral fit stands on, not the one the window was selected around.
  '''
  key, method, z, logr, barT_f = args
  results = load_sweep(method_outdir(method, key, z))
  r = [q for q in results if abs(q['log10ratio'] - logr) < 1e-9][0]
  nub, nuFnu, p = r['nub'], r['nuFnu'], r['env'].psyn
  barT = np.asarray(r['Tb'], float) - 1.
  x = barT/barT_f
  Fpk = np.nanmax(nuFnu, axis=1)
  bright = Fpk > FLUX_FLOOR*np.nanmax(Fpk)
  step = np.where(x < X_FINE, STEP_FINE, STEP_COARSE)
  rows = []
  for i in range(len(nuFnu)):
    if not bright[i] or i % int(step[i]):
      continue
    br_ = sb.breaks_from_identified(nub, nuFnu[i], p)
    d = br_['det']
    if not d:
      continue
    nan = float('nan')
    if d['regime'] == 'MC':
      # no mid segment is displayed, so there is no branch and no asymptote to depart from.
      # The slope the fit uses there is FITTED inside it (free_bmid='mc'), so it has to come
      # from the fit itself rather than from the geometry -- passing br_ back in means the
      # cut-off scan and the identification are not repeated.
      # BOTH bounds, on the same geometry, so the comparison costs one extra least-squares
      # and no extra cut-off scan: the shape limits (what a three-segment spectrum can carry)
      # and the physical ones (what a fused FC or SC knee can produce, BMID_DEP).
      gm = sb.smoothing_from_identified(nub, nuFnu[i], p, br=br_)
      gp = sb.smoothing_from_identified(nub, nuFnu[i], p, br=br_, bmid_physical=True)
      if np.isfinite(gm['a_mid']) or np.isfinite(gp['a_mid']):
        rows.append(dict(logr=r['log10ratio'], barT=barT[i], x=x[i], step=i, branch='mc',
                         a_th=nan, a_mid=(gm['a_mid'] if gm['s_ok'] else nan), dep=nan,
                         mid_from=gm['mid_from'] or 'none', dex_mid=gm['dex_mid'],
                         a_core=nan, dep_core=nan, core=nan, dex=nan, a_win=nan,
                         regime='MC', a_mid_phys=gp['a_mid'], rms=gm['rms'],
                         rms_phys=gp['rms'], sep=nan, s1=gm['s1'], s2=gm['s2'],
                         depth=nan,
                         phys_at_bound=float(bool(gp['bmid_at_bound']))))
      continue
    for br in ('fc', 'sc'):
      if br not in d['segs']:
        continue
      g = d['segs'][br]
      # a_mid: the re-centred measurement, i.e. what the spectral fit holds. a_core and
      # a_win are kept alongside as the two window-bounded estimators they superseded --
      # a_core the settled slope inside the identified window, a_win a free line across
      # all of it. Both inherit that window's asymmetry; a_mid does not.
      a_mid = br_['a_mid']
      # the smoothing fit is run here too, for the two numbers the bias correction is
      # indexed by: the break separation and s1. The bias is driven by s1 (a smoother lower
      # break bleeds further into the mid window), and s1 varies along a track, so a
      # correction by epoch would step the curve where the physics does not.
      gf = sb.smoothing_from_identified(nub, nuFnu[i], p, br=br_)
      sep = (np.log10(br_['b_hi']/br_['b_lo'])
             if np.isfinite(br_['b_hi']) and np.isfinite(br_['b_lo'])
             and br_['b_lo'] > 0 else nan)
      # for a SINGLE-break spectrum (VFC, FC*) there is no separation to index a bias by;
      # what sets the tilt there is the upper break's smoothing and how many decades of the
      # 1/2 segment sit in band beneath it -- see segment_route.vfc_bias_grid
      nub_lo = float(np.nanmin(nub[nub > 0.])) if np.any(nub > 0.) else nan
      depth = (np.log10(br_['b_hi']/nub_lo)
               if np.isfinite(br_['b_hi']) and np.isfinite(nub_lo) and nub_lo > 0 else nan)
      rows.append(dict(sep=sep, s1=gf['s1'], s2=gf['s2'], depth=depth,
                       logr=r['log10ratio'], barT=barT[i], x=x[i], step=i, branch=br,
                       a_th=g['a'], a_mid=a_mid, dep=a_mid - g['a'],
                       mid_from=br_['mid_from'] or 'none', dex_mid=br_['dex_mid'],
                       a_core=g['a_core'], dep_core=g['dep'], core=g['core'],
                       dex=g['dex'], a_win=g['a_fit'], regime=d['regime'] or 'None',
                       a_mid_phys=nan, rms=gf['rms'], rms_phys=nan, phys_at_bound=nan))
  return rows


def measure(key=DEFAULT_KEY, method=METHOD, z=Z_SHELL, verbose=True, nproc=NPROC):
  '''
  The mid segment of every sampled spectrum of every sweep point, measured the way the
  spectral fit measures it: a_mid is breaks_from_identified's re-centred slope, the one
  fit_smoothing_held holds, and dep = a_mid - a_th its departure from the one-zone
  asymptote. a_core and a_win are still recorded so the change of estimator can be read
  off the same rows.

  One row per (sweep point, step, branch), branch being 'fc' or 'sc'. A spectrum has at
  most one of them (identify_segments drops the narrower where both linger), so the two
  branches never describe the same step. Steps whose class is MC carry no branch and are
  therefore absent: their mid slope is a fitted transition slope, not a segment.
  '''
  outdir = method_outdir(method, key, z)
  results = load_sweep(outdir)
  if not results:
    raise RuntimeError(f'no cached sweep in {outdir}; run sweep_gammacm.run_sweep first')
  barT_f = exit_onset_barT(key, z=z)
  jobs = [(key, method, z, float(r['log10ratio']), barT_f) for r in results]
  np_ = cell_pool.resolve_nproc(nproc, cap=len(jobs))
  if np_ > 1:
    with cell_pool.pool_context().Pool(np_) as pool:
      out = pool.map(_measure_point, jobs)
  else:
    out = [_measure_point(j) for j in jobs]
  rows = [w for chunk in out for w in chunk]
  if verbose:
    for j, chunk in zip(jobs, out):
      print(f'  logr={j[3]:+.1f} done ({len(chunk)} rows)', flush=True)
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
      if k not in ('branch', 'regime', 'mid_from'):
        r[k] = float(r[k])
    rows.append(r)
  return rows


BIAS_CSV = 'bias_grid.csv'      # written by segment_route.bias_grid, in ITS outdir
VFC_BIAS_CSV = 'vfc_bias_grid.csv'   # ... and its single-break counterpart
BIAS_MAX = 0.15               # grid cells beyond this are estimator breakdown, not a
                              # correction: the sc column at s1 = 0.4 flips sign and reaches
                              # +0.30 while its neighbours sit at -0.02. Interpolating across
                              # that would subtract more than the whole physical range of the
                              # slope, so those cells are dropped and any bin landing among
                              # them is left uncorrected and drawn as such.


def _vfc_bias_interp():
  '''
  bias(depth, s2) for the SINGLE-break spectra -- VFC and FC*, which carry the 1/2 segment
  with no lower break under it and so have no separation to index bias_grid by. Their tilt is
  the opposite sign to the two-break case (-0.003 to -0.036): with nothing bleeding up from
  below, only the upper break curving down into the window acts on the fitted line.
  '''
  from scipy.interpolate import LinearNDInterpolator
  from segment_route import OUTDIR as SR_OUT
  path = os.path.join(SR_OUT, VFC_BIAS_CSV)
  if not os.path.isfile(path):
    return None
  pts, val = [], []
  for r in csv.DictReader(open(path)):
    if r['bias'] in ('', 'nan') or r['regime'] not in ('VFC', 'FC*'):
      continue
    b = float(r['bias'])
    if abs(b) > BIAS_MAX:
      continue
    pts.append((float(r['depth']), float(r['s2']))); val.append(b)
  if len(pts) < 4:
    return None
  return LinearNDInterpolator(np.array(pts), np.array(val))


def _bias_interp(branch):
  '''
  bias(sep, s1) for one branch, from segment_route's grid: what the estimator returns on a
  synthetic whose mid slope IS the asymptote, at that break separation and lower-break
  smoothing. Linear inside the sampled hull, None outside -- a bin the grid does not cover is
  reported uncorrected rather than extrapolated.
  '''
  from scipy.interpolate import LinearNDInterpolator
  from segment_route import OUTDIR as SR_OUT
  path = os.path.join(SR_OUT, BIAS_CSV)
  if not os.path.isfile(path):
    return None
  pts, val = [], []
  for r in csv.DictReader(open(path)):
    if r['branch'] != branch or r['bias'] in ('', 'nan'):
      continue
    b = float(r['bias'])
    if abs(b) > BIAS_MAX:
      continue
    pts.append((float(r['sep']), float(r['s1']))); val.append(b)
  if len(pts) < 4:
    return None
  return LinearNDInterpolator(np.array(pts), np.array(val))


def _ylim(vals, pad=0.06, top=0.0):
  '''Limits from the data being drawn, with `top` extra headroom in units of its range --
  the panel carrying the legend needs a band above the tracks, the other does not.'''
  v = np.asarray([q for q in vals if np.isfinite(q)], float)
  if not v.size:
    return None
  lo, hi = float(v.min()), float(v.max())
  r = max(hi - lo, 1e-3)
  return lo - pad*r, hi + (pad + top)*r


def plot(rows, outdir, barT_f, barT_off=None, fname=FIG_NAME, corrected=True):
  '''
  Two panels, one per branch, on a shared log time axis. Each holds its asymptote as a
  dashed guide and the sweep points as one curve per log10(gma_c/gma_m). ONE horizontal
  legend, along the empty top of the slow-cooling panel, covers the union of the two
  branches' regimes -- the colour map is global (COL), so a legend built per panel would
  say the same thing twice and split the shared entries across two keys.
  barT_off shades the rarefaction band; no line is drawn at its right edge (the band's
  own edge IS bar{T}_rf -- see sweep_gammacm.plot_lightcurve_shape).
  '''
  xoff = tuple(b/barT_f for b in barT_off) if (barT_off and barT_f > 0.) else None
  fig, axes = plt.subplots(2, 1, figsize=(8.4, 7.6), sharex=True)
  # SC on top, FC below, the order the spectrum passes through them as gamma_c falls. The
  # MARGINAL class is measured (both bounds, see _measure_point) and written to the csv, but
  # NOT drawn: bounded to what a fused fc or sc knee can produce, its fitted mid slope pins
  # at the ceiling in 83% of bins, so there is no track to plot -- that pinning is a result
  # about which shape those spectra want, and belongs in the text.
  panels = (('sc', 0.25, 'Slow-cooling branch', (0.205, 0.266)),
            ('fc', 0.5, 'Fast-cooling branch', (0.478, 0.615)))
            # the sc top is set by the legend, not by the data: no track goes above
            # a_th = 0.25, so what is left above it is exactly the legend's band
  for ax, (br, a_th, title, ylim) in zip(axes, panels):
    sub = [r for r in rows if r['branch'] == br]
    # subtract the estimator's own tilt, bin by bin, at that bin's own parameters -- the
    # (separation, s1) grid for a two-break spectrum, the (band depth, s2) one for a single
    # break. corrected=False leaves every track raw, for the companion figure.
    itp = _bias_interp(br) if corrected else None
    itp1 = _vfc_bias_interp() if corrected else None
    n_un = 0
    for r in sub:
      b = np.nan
      if itp is not None and np.isfinite(r.get('sep', np.nan)) \
         and np.isfinite(r.get('s1', np.nan)):
        b = float(itp(r['sep'], r['s1']))
      elif itp1 is not None and np.isfinite(r.get('depth', np.nan)) \
           and np.isfinite(r.get('s2', np.nan)):
        b = float(itp1(r['depth'], r['s2']))   # single break: no lower break to index by
      r['bias'] = b
      r['a_corr'] = r['a_mid'] - b if np.isfinite(b) else np.nan
      n_un += int(not np.isfinite(b))
    if corrected and n_un:
      print(f'  {br}: {n_un}/{len(sub)} bins outside the bias grid, left uncorrected')
    if xoff is not None:
      ax.axvspan(xoff[0], xoff[1], color='grey', alpha=0.15, lw=0, zorder=0)
    ax.axvline(1., color='grey', ls=':', lw=0.9, zorder=1)
    # the one-zone asymptote, unlabelled: with each figure scaled to its own data the line
    # is the only fixed reference, but a label on it invites the departure to be eyeballed
    # off whichever version is to hand.
    ax.axhline(a_th, color=MUTED, lw=1.2, ls='--', zorder=1)
    val = (lambda r: r['a_corr']) if corrected else (lambda r: r['a_mid'])
    drawn = [val(r) for r in sub]
    for lr in sorted({r['logr'] for r in sub}):
      d = sorted([r for r in sub if r['logr'] == lr], key=lambda r: r['x'])
      c = col(lr)
      # SOLID where the measurement window was re-centred on its own slope -- the value
      # the spectral fit holds. HOLLOW where that re-centring found no window and the fit
      # fell back to the line over the identified (asymmetric) window, which carries a
      # known offset of about +0.02 fast / -0.02 slow.
      # the split is by HOW the slope was measured, not by whether a correction existed:
      # re-centred windows draw as a line, the fell-back ones (VFC/FC*, which have no 4/3
      # window to re-centre against) as open circles. In the raw figure every bin has a
      # value, so nothing is faint there.
      rec = [r for r in d if r['mid_from'] == 'plateau_recentred' and np.isfinite(val(r))]
      fell = [r for r in d if r['mid_from'] != 'plateau_recentred' and np.isfinite(val(r))]
      unc = ([r for r in d if not np.isfinite(r['a_corr']) and np.isfinite(r['a_mid'])]
             if corrected else [])
      if rec:
        ax.plot([r['x'] for r in rec], [val(r) for r in rec], '-', color=c,
                lw=1.8, solid_capstyle='round', zorder=3)
      if fell:
        ax.plot([r['x'] for r in fell], [val(r) for r in fell], 'o', mfc='none',
                mec=c, ms=3.6, mew=1.0, ls='none', zorder=2)
      if unc:   # corrected figure only: no correction available, so plotted raw
        ax.plot([r['x'] for r in unc], [r['a_mid'] for r in unc], '.', color=c,
                ms=3.0, alpha=0.7, ls='none', zorder=2)
    ax.set_xscale('log')
    # limits from what is actually drawn, so a correction that shifts the distribution
    # cannot push points off the panel; the sc axes carry the legend and keep headroom
    # the sc axes carry the legend, so they keep a band above the tracks -- as a fraction of
    # the drawn range, which is what makes it a constant slice of the panel whatever range
    # the data happens to span. 0.30 leaves the legend room without the empty strip a larger
    # value opened up on the raw figure, whose sc tracks span twice the corrected ones.
    lim = _ylim(drawn, top=(0.30 if br == 'sc' else 0.0))
    ax.set_ylim(*(lim if lim else ylim))
    ax.grid(True, which='major', color=GRID, lw=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
      ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
      ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_ylabel('$a_{\\rm mid}$', color=INK, fontsize=11)
    ax.set_title(title, color=INK, fontsize=11, loc='left', pad=6)
  axes[-1].set_xlabel('$\\bar{T}/\\bar{T}_f$', color=INK, fontsize=10)
  # ONE legend for both panels, in a single row along the top of the slow-cooling panel:
  # that band is empty (every sc track sits at or below a_th = 0.25) and the gap between
  # the panels then costs nothing. The handles are built by hand rather than harvested
  # from either axes -- the union of the two branches' regimes is what has to appear, and
  # neither panel carries all of it.
  lrs = sorted({int(r['logr']) for r in rows})
  handles = [Line2D([], [], color=col(lr), lw=1.8) for lr in lrs]
  leg = axes[0].legend(handles, [f'{lr:+d}' for lr in lrs],
                       title='$\\log_{10}\\mathcal{C}$', fontsize=8.5,
                       title_fontsize=8.5, ncol=len(lrs), loc='upper center',
                       frameon=True, framealpha=0.92, edgecolor=GRID,
                       columnspacing=1.2, handlelength=1.5, handletextpad=0.4,
                       borderaxespad=0.4)
  leg.get_title().set_color(MUTED)
  for t in leg.get_texts():
    t.set_color(INK)
  fig.tight_layout()
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
  off = rarefaction_off_barT(key, z=z)
  path = plot(rows, outdir, barT_f, off)
  # the same tracks with the estimator's tilt left in, for comparison. Its y range is set
  # from its own data, so the two figures are NOT on a shared scale -- read the guides.
  raw = plot(rows, outdir, barT_f, off, fname=FIG_NAME_RAW, corrected=False)
  trim_pngs([path, raw])
  copy_article_figures(outdir)
  print(f'-> {path}\n-> {raw}')
  return rows


if __name__ == '__main__':
  main()
