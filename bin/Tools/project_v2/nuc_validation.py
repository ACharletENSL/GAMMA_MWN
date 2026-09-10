# -*- coding: utf-8 -*-
# @Author: acharlet

'''
Validation of the per-cell cooling frequency (cooling_frequency.py) against the cooling
break actually fitted to the full-shell spectra, on the cached gamma_c/gamma_m sweeps.

The question this answers is the one left open by the semi-analytical strategy: the model
gives a cooling frequency PER CELL, nu_c,i(t), but the shell spectrum shows a SINGLE
cooling break. Which summary of the population {nu_c,i} is that break?

  check_identity      - gamma_c from the closed-form cumulative sum vs 1/cooled_tt_eff
                        read off the electron bounds the emission pipeline already
                        evolved. These are the same quantity; this is the check that the
                        model is consistent with what the code actually radiates.

  calibrate_num       - the frequency-normalisation control. Pushes nu_m,i through the
                        identical path and compares with the fitted nu_m, which measures
                        the offset between the single-electron characteristic frequency
                        (3/2) gamma^2 nu_B and the break PARAMETER of the Granot-Sari
                        shape. Everything else inherits that offset, so it is measured
                        first and never tuned.

  estimator_scan      - the central diagnostic: C = nu_c,fit / nu_c,estimator per
                        estimator, per cooling regime, per shell. A model exists only if
                        some estimator makes C constant.

  plot_population     - {nu_c,i} as a weighted percentile band with the fitted break over
                        it, one panel per regime: the picture behind the numbers.

  plot_collapse       - every curve divided by the nominal (gma_c/gma_m)^2. The alpha
                        rescaling makes that ratio the ONLY regime dependence, so if the
                        dimensionless function is universal the eight curves land on one.

Ground truth is track_breaks_gs02 on the cached sweeps (no spectrum is recomputed); the
population comes from cooling_frequency.harvest_shell_cooling, which rebuilds the cell
frames only. Both shells of KEY are covered, the same configuration sweep_shells.py uses.

Colours map the ordered sweep parameter log10(gamma_c/gamma_m) through the house
_sweep_colors helper, but on a perceptually uniform ramp rather than the default jet:
these panels are read for WHERE curves fall on a magnitude scale, which a rainbow
misrepresents.

WHERE THIS STANDS (last full run: both shells of the RETIRED cooling_fid_raref_ext, with
METHOD='data_rarcut'. KEY/METHOD below now point at cooling_g100 / 'data', so every number
in this block is pending a re-run before it can be quoted again.)

  The model is cooling_frequency: per-cell gamma_c from the closed-form cumsum, nu_c,i from
  it, and the shell value taken as the emission-weighted 5th percentile of {nu_c,i} over the
  steps whose arrival window is open. The quantile is the ONLY calibrated number; it sets C.

  Against the segment reference, on the regimes both trackers resolve:
      RS  C = 1.03, regime drift 1.54, in-regime shape 0.030 dex   (that track's A drift 2.00)
      FS  C = 1.00, regime drift 1.43, in-regime shape 0.040 dex   (that track's A drift 1.94)
  Against the GS02 reference the same numbers are 1.05/3.08/0.086 and 1.00/2.53/0.101, i.e.
  changing only how nu_c is READ halves the drift and cuts the scatter ~3x.

  But epoch_scan shows the aggregate hides real structure: C ~ 1 (0.91-1.07) through fast
  cooling and the marginal band, while in SLOW cooling the model over-predicts nu_c by up to
  2x during the rise and recovers by the tail. A fixed quantile is the wrong object -- the
  cooled fraction of the shell varies by orders of magnitude across the sweep. The next move
  is a regime-aware rule in cooling_frequency.effective_nu_c (a threshold on the cooled
  fraction, or on gamma_c relative to the injection gamma_M), re-tested with epoch_scan.

  Not done: the closed-form f_c(tau) is not tabulated, so this is still a computation
  (~16 s/point) rather than a law; spectral_breaks.smoothing_from_deficit is written but
  unexercised; only one hydro setup has been tested.

Example use in command line:
  python -c "import nuc_validation as V; V.main()"
  python -c "import nuc_validation as V; V.check_identity()"
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from environment import MyEnv, GAMMA_dir, figdir
from IO import open_celldata
from working_cooling_data import generate_cell_fromData
import cooling_frequency as cf
import spectral_breaks as sb
import sweep_gammacm as swp

# The two independent ways of reading nu_c(t) off the same cached spectra. They share the
# track_breaks_gs02 output contract, so everything downstream takes either.
#   gs02     - fit the whole smoothed Granot-Sari template, smoothing held at (1.3, 2.0).
#   segments - intersect the asymptotic power-law segments; smoothing never enters, and the
#              measurement is declined where the segments are not resolved.
# Running both is the point: their difference is the systematic that a single reference
# hides, and it is what decides whether a residual drift in C belongs to the model.
TRACKERS = {'gs02': swp.track_breaks_gs02, 'segments': sb.track_breaks_segments}

OUTDIR_NAME = 'nuc_model'          # figdir(OUTDIR_NAME, key) puts it under the
OUTDIR = figdir(OUTDIR_NAME)    # run's own folder; this is the fiducial's
KEY = 'cooling_g100'
METHOD = swp.DEFAULT_METHOD  # the reference computation; imported, not hardcoded, so this
                         # module cannot drift from sweep_gammacm's default
Z_RS, Z_FS = 4, 1
CMAP = plt.cm.viridis          # sequential: log10(gma_c/gma_m) is an ordered magnitude
ESTIMATORS = ('min_bright', 'q05', 'q10', 'q15', 'q20', 'q25', 'q50', 'wlogmean')
BART_LO = 1e-2                 # below this the sub-cell reconstruction dominates the
                               # population and the fit has little dynamic range


def _ensure_outdir():
  os.makedirs(OUTDIR, exist_ok=True)


def load_side(z=Z_RS, key=KEY, method=METHOD, trackers=('gs02', 'segments')):
  '''
  Cached sweep points of one shell, with EVERY requested break track and the harvested
  {nu_c,i} population. The population is the expensive part (~16 s a point), so it is
  harvested once and shared by all trackers rather than once per tracker.
  Returns a list of dicts ordered by log10(gamma_c/gamma_m); `tracks` holds the tracks by
  name and `tr` aliases the first for callers that want a single one.
  '''
  outdir = swp.method_outdir(method, key, z)
  res = swp.load_sweep(outdir)
  if not res:
    raise FileNotFoundError(f'no cached sweep in {outdir} -- run sweep_shells.main first')
  out = []
  for r in sorted(res, key=lambda x: x['log10ratio']):
    trk = {name: TRACKERS[name](r) for name in trackers}
    env, H = cf.harvest_shell_cooling(key, z=z, alpha=r['alpha'], verbose=False)
    out.append(dict(r=r, tracks=trk, tr=trk[trackers[0]], env=env, H=H,
                    logr=r['log10ratio'], z=z))
    ns = ', '.join(f"{n}:{int(t['valid'].sum())}" for n, t in trk.items())
    print(f"  log10ratio={r['log10ratio']:+.1f} z={z}: {H.emitter.nunique()} emitters, "
          f"{len(H)} steps | valid bins {ns}", flush=True)
  return out


# ---------------------------------------------------------------------------
# checks

def check_identity(key=KEY, z=Z_RS, klist=None, verbose=True):
  '''
  gamma_c (cooling_frequency.cell_cooling_lfac, the closed-form cumsum) against
  1/cooled_tt_eff(gmax, bsyn) (cell_cooled_cutoff_lfac, read off the bounds the pipeline
  evolved). They are the same quantity, so this must hold to round-off; if it ever does
  not, the model has drifted from the emission and nothing downstream is meaningful.
  '''
  env = MyEnv(key)
  if klist is None:
    kCD = env.Next + env.Nsh4
    klist = [env.Next + 2, env.Next + 40, kCD - 3, kCD - 1]
  rows = []
  for k in klist:
    cd = open_celldata(key, k)
    if cd is False:
      continue
    cell, cenv = generate_cell_fromData(cd, env)
    if cell is False:
      continue
    gc = cf.cell_cooling_lfac(cell)
    gcut = cf.cell_cooled_cutoff_lfac(cell)
    d = np.abs(gc[1:]/gcut[1:] - 1.)
    rows.append(dict(cell=k, steps=len(cell), max_reldiff=float(np.nanmax(d)),
                     gma_c_end=float(gc[-1]), gma_M0=float(cell.iloc[0].gmax)))
  df = pd.DataFrame(rows)
  if verbose:
    print('\n=== check_identity: gamma_c (cumsum) vs 1/cooled_tt_eff (evolved bounds) ===')
    print(df.to_string(index=False))
    print(f'worst relative difference over all cells: {df.max_reldiff.max():.3e}')
  return df


def calibrate_num(sides, barT_lo=BART_LO, barT_hi=1., verbose=True):
  '''
  Control measurement: fitted nu_m / model nu_m,injection.

  The model side is the nu_m of each emitter at its FIRST step, as a function of that
  emitter's onset -- i.e. the injection frequency of whatever is being shocked at that
  observer time, which is what sets the shell's nu_m.

  The ratio is the offset between the single-electron characteristic frequency and the
  Granot-Sari break parameter. It is a property of the SHAPE, not of the cooling, so it
  is expected to be constant; carrying it explicitly keeps it out of the nu_c constant C.
  '''
  rows = []
  for s in sides:
    H, tr = s['H'], s['tr']
    inj = H.sort_values(['emitter', 'barT']).groupby('emitter').first().sort_values('barT')
    b = tr['barT']
    m = tr.get('valid_m', tr['valid']) & np.isfinite(tr['nu_m']) & (tr['nu_m'] > 0.) \
        & (b >= barT_lo) & (b <= barT_hi)
    if m.sum() < 4:
      continue
    pred = np.interp(b[m], inj.barT.to_numpy(float), inj.nu_m.to_numpy(float))
    rat = tr['nu_m'][m]/pred
    lo, hi = np.percentile(np.log10(rat), [16, 84])
    rows.append(dict(z=s['z'], logr=s['logr'], n=int(m.sum()),
                     ratio=float(np.median(rat)), spread_dex=float(0.5*(hi - lo))))
  df = pd.DataFrame(rows)
  if verbose and len(df):
    print('\n=== calibrate_num: fitted nu_m / model nu_m(injection) ===')
    print(df.to_string(index=False))
    print(f'median over all points: {df.ratio.median():.3f}  '
          f'=> the GS02 break sits at {cf.SYN_FAC*df.ratio.median():.3f} x gamma^2 nu_B '
          f'(SYN_FAC = {cf.SYN_FAC} is the single-electron convention)')
  return df


def ground_truth_drift(sides, key=KEY, verbose=True, tracker='gs02'):
  '''
  How much the GROUND TRUTH itself moves with the cooling regime, so that the model's own
  regime dependence can be judged against something.

  fit_break_evolution already reduces each fitted track to A = median(nu_c bar{T}^2) /
  (gma_c/gma_m)^2 -- the collapse normalisation that would be constant if the fitted nu_c
  followed the nominal cooling ratio exactly. It does not: A drifts monotonically from
  fast to slow cooling. Any constant C measured against these tracks inherits that drift,
  so A's max/min is the yardstick for the estimator scan, not 1.
  '''
  z = sides[0]['z']
  barT_f = swp.exit_onset_barT(key, z)
  barT_off = swp.rarefaction_off_barT(key, z)
  rows = []
  for s in sides:
    f = swp.fit_break_evolution(s['r'], s['tracks'][tracker], barT_f, barT_off)
    rows.append(dict(z=z, tracker=tracker, logr=s['logr'], A=f['A'], nominal=f['nominal'],
                     ratio_pk=f['ratio_pk'], sep_min=f['sep_min'], flag=f['flag']))
  df = pd.DataFrame(rows)
  if verbose:
    print(f'\n=== ground_truth_drift z={z} [{tracker}]: the track against its own nominal ===')
    print(df.to_string(index=False, float_format=lambda v: f'{v:.3g}'))
    a = df.A.to_numpy(float); a = a[np.isfinite(a)]
    if len(a) > 1:
      print(f'collapse constant A: median {np.median(a):.2f}, max/min {a.max()/a.min():.2f} '
            '<- the regime drift already present in the ground truth')
  return df


def estimator_scan(sides, estimators=ESTIMATORS, barT_lo=BART_LO, verbose=True, gt=None,
    trackers=('gs02', 'segments')):
  '''
  C = nu_c,fit / nu_c,estimator for every estimator and every cooling regime.

  The deliverable is the LAST block printed: how constant C is across the eight regimes.
  An estimator whose C moves with the regime is not a model of the break, however small
  its scatter within any one point -- but the comparison is against the ground truth's own
  drift (ground_truth_drift), not against perfect constancy, since the fitted tracks do
  not themselves follow the nominal cooling ratio.
  '''
  rows = []
  for s in sides:
    ne = cf.effective_nu_c(s['H'], s['tr']['barT'], estimators=estimators)
    for tk in trackers:
      for e in estimators:
        C, spread, n = cf.compare_to_track(ne[e], s['tracks'][tk], barT_lo=barT_lo)
        rows.append(dict(z=s['z'], tracker=tk, logr=s['logr'], estimator=e,
                         C=C, spread_dex=spread, n=n))
  df = pd.DataFrame(rows)
  if verbose:
    for (z, tk), d in df.groupby(['z', 'tracker']):
      print(f'\n=== estimator_scan z={z} [{tk}]: C = measured nu_c / estimator ===')
      print(d.pivot(index='logr', columns='estimator', values='C').to_string(
          float_format=lambda v: f'{v:.3g}'))
    print('\n=== constancy of C across regimes (the deliverable) ===')
    # Compare trackers ONLY on the regimes both could measure: the segment method declines
    # where the segments are unresolved, and crediting it with a tighter C partly won on a
    # different subset would not be a comparison.
    summ = []
    for (z, e), d in df.groupby(['z', 'estimator']):
      piv = d.pivot(index='logr', columns='tracker', values='C')
      common = piv.dropna()
      if len(common) < 2:
        continue
      for tk in piv.columns:
        v = common[tk].to_numpy(float)
        v = v[np.isfinite(v) & (v > 0.)]
        if len(v) < 2:
          continue
        sp = d[(d.tracker == tk)].spread_dex
        summ.append(dict(z=z, estimator=e, tracker=tk, npts=len(v), C=np.median(v),
                         max_over_min=v.max()/v.min(),
                         spread_dex=float(np.nanmedian(sp))))
    if summ:
      summ = pd.DataFrame(summ).sort_values(['z', 'estimator', 'tracker'])
      print(summ.to_string(index=False, float_format=lambda v: f'{v:.3g}'))
    if gt is not None:
      a = gt.A.to_numpy(float); a = a[np.isfinite(a)]
      if len(a) > 1:
        print(f"\nyardstick: that track's own collapse constant A drifts by "
              f'{a.max()/a.min():.2f}x over the same regimes. An estimator whose C drifts '
              'by about that much is as constant as the track can show.')
  return df


# ---------------------------------------------------------------------------
# figures

EPOCHS = (('rise', 1e-3, 0.1), ('cross', 0.1, 0.5), ('late', 0.5, 2.), ('tail', 2., 1e3))


def epoch_scan(sides, estimator='q05', tracker='segments', epochs=EPOCHS, verbose=True):
  '''
  C split by epoch of the pulse, which is what the aggregate estimator_scan hides.

  A single median C per sweep point averages over the burst; doing that made the model look
  better than it is. Resolved in time against the segment reference, two coherent residuals
  appear: C ~ 1 throughout fast cooling and the marginal band, but in SLOW cooling the model
  over-predicts nu_c by up to 2x during the rise and recovers toward 1 by the tail, so C
  moves ~2x across the pulse within a single point.

  Read this before trusting a constant C, and re-run it after any change to
  cooling_frequency.effective_nu_c -- it is the diagnostic that says whether a new estimator
  actually fixed the slow-cooling limit or just moved the average.
  '''
  rows = []
  for s in sides:
    tr = s['tracks'][tracker]
    if tr['valid'].sum() < 8:
      continue
    ne = cf.effective_nu_c(s['H'], tr['barT'], estimators=(estimator,))[estimator]
    b = tr['barT']
    m0 = tr['valid'] & np.isfinite(ne) & (ne > 0.)
    d = dict(z=s['z'], logr=s['logr'])
    for lab, lo, hi in epochs:
      m = m0 & (b >= lo) & (b < hi)
      d[lab] = float(np.median(tr['nu_c'][m]/ne[m])) if m.sum() > 3 else np.nan
      d['n_' + lab] = int(m.sum())
    rows.append(d)
  df = pd.DataFrame(rows)
  if verbose and len(df):
    print(f'\n=== epoch_scan [{tracker}, {estimator}]: C = measured nu_c / model, by epoch ===')
    print(df.to_string(index=False, float_format=lambda v: f'{v:.3g}'))
    for lab, _, _ in epochs:
      v = df[lab].to_numpy(float); v = v[np.isfinite(v)]
      if len(v):
        print(f'  {lab:<6} median C = {np.median(v):.3f}  (n={len(v)})')
  return df


def plot_population(sides, outdir=OUTDIR, estimator='q10', tag=''):
  '''
  The population behind the numbers: weighted 10-50-90 percentile band of {nu_c,i(barT)}
  with the fitted break (solid) and the chosen estimator (dashed) over it, one panel per
  cooling regime. Where the tracker declares no measurement there is a gap, not a line.
  '''
  outdir = figdir(OUTDIR_NAME, key) if outdir is None else outdir
  _ensure_outdir(outdir)
  n = len(sides)
  ncol = min(4, n)
  nrow = int(np.ceil(n/ncol))
  fig, axs = plt.subplots(nrow, ncol, figsize=(3.6*ncol, 3.1*nrow), sharex=True, sharey=False)
  axs = np.atleast_1d(axs).ravel()
  for ax, s in zip(axs, sides):
    tr, H = s['tr'], s['H']
    b = tr['barT']
    qs = cf.effective_nu_c(H, b, estimators=('q10', 'q50', 'q90', estimator))
    ax.fill_between(b, qs['q10'], qs['q90'], color='0.75', lw=0, alpha=.7,
                    label='{$\\nu_{c,i}$} 10-90%')
    ax.loglog(b, qs['q50'], color='0.45', lw=1., ls='-', label='median')
    ax.loglog(b, qs[estimator], color='#1f77b4', lw=1.4, ls='--', label=f'{estimator}')
    for name, tk, col, lw in (('gs02', s['tracks'].get('gs02'), 'k', 1.6),
                              ('segments', s['tracks'].get('segments'), '#d62728', 1.6)):
      if tk is None:
        continue
      ax.loglog(b, np.where(tk['valid'], tk['nu_c'], np.nan), color=col, lw=lw,
                label=f'$\\nu_\\mathrm{{c}}$ {name}')
    ax.set_title(f"log$_{{10}}\\mathcal{{C}}$ = {s['logr']:+.0f}", fontsize=9)
    ax.grid(alpha=.2, lw=.5)
  for ax in axs[len(sides):]:
    ax.axis('off')
  axs[0].legend(fontsize=7, framealpha=.9)
  for ax in axs[-ncol:]:
    ax.set_xlabel('$\\bar{T}$')
  for i in range(0, len(axs), ncol):
    axs[i].set_ylabel('$\\nu/\\nu_{\\mathrm{m},0}$')
  z = sides[0]['z']
  fig.suptitle(f'cell cooling-frequency population vs the fitted break  (z={z})', fontsize=10)
  fig.tight_layout()
  p = os.path.join(outdir, f'nuc_population_z{z}{tag}.png')
  fig.savefig(p, dpi=200, bbox_inches='tight')
  plt.close(fig)
  print(f'-> {p}')
  return p


def plot_collapse(sides, outdir=OUTDIR, estimator='q10', tag='', ref='segments'):
  '''
  Universality test. Under the Granot rescaling the ONLY regime dependence of nu_c/nu_0
  is the prefactor (gma_c/gma_m)^2, so dividing it out must land every regime on one
  curve -- for the model (top) and for the fitted break (bottom) alike. Divergence
  between the two panels is where the single-break description of the shell breaks down.
  '''
  _ensure_outdir()
  fig, axs = plt.subplots(2, 1, figsize=(6.4, 7.2), sharex=True)
  logr = np.array([s['logr'] for s in sides], float)
  norm = plt.Normalize(vmin=logr.min(), vmax=logr.max())
  sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
  for s in reversed(sides):   # slowest cooling first, fast-cooling curves drawn on top
    c = CMAP(norm(s['logr']))
    tr, env = s['tr'], s['env']
    nominal = (env.gma_c/env.gma_m)**2
    b = tr['barT']
    est = cf.effective_nu_c(s['H'], b, estimators=(estimator,))[estimator]
    axs[0].loglog(b, est/nominal, color=c, lw=1.4)
    tk = s['tracks'].get(ref, tr)
    axs[1].loglog(b, np.where(tk['valid'], tk['nu_c'], np.nan)/nominal, color=c, lw=1.4)
  for ax, lab in zip(axs, [f'model ({estimator})', f'measured ({ref})']):
    ax.grid(alpha=.2, lw=.5)
    ax.set_ylabel(f'$\\nu_\\mathrm{{c}}$ {lab} $/[\\nu_{{\\mathrm{{m}},0}}\\mathcal{{C}}^2]$')
  axs[-1].set_xlabel('$\\bar{T}$')
  z = sides[0]['z']
  axs[0].set_title(f'collapse on the nominal cooling ratio  (z={z})', fontsize=10)
  fig.colorbar(sm, ax=axs, label='log$_{10}\\mathcal{C}$')
  p = os.path.join(outdir, f'nuc_collapse_z{z}_{ref}{tag}.png')
  fig.savefig(p, dpi=200, bbox_inches='tight')
  plt.close(fig)
  print(f'-> {p}')
  return p


def write_tables(cal, est, outdir=OUTDIR, gt=None, ep=None):
  '''calibration + estimator scan (+ ground-truth drift, epoch scan) to csv'''
  _ensure_outdir()
  cal.to_csv(os.path.join(outdir, 'num_calibration.csv'), index=False)
  est.to_csv(os.path.join(outdir, 'estimator_scan.csv'), index=False)
  names = 'num_calibration.csv, estimator_scan.csv'
  for df, fn in ((gt, 'ground_truth_drift.csv'), (ep, 'epoch_scan.csv')):
    if df is not None:
      df.to_csv(os.path.join(outdir, fn), index=False)
      names += f', {fn}'
  print(f'-> {outdir}/{names}')


def main(key=KEY, method=METHOD, zlist=(Z_RS, Z_FS), estimator='q05', outdir=None,
    trackers=('gs02', 'segments')):
  '''
  Full diagnostic on the cached sweeps of both shells, against EVERY requested break
  tracker. No shell spectrum is recomputed: the references come from the caches, the
  population from cell frames only.
  '''
  _ensure_outdir()
  check_identity(key=key, z=zlist[0])
  all_sides, cals, ests, gts, eps = [], [], [], [], []
  for z in zlist:
    print(f'\n--- harvesting shell z={z} ---', flush=True)
    sides = load_side(z=z, key=key, method=method, trackers=trackers)
    all_sides.append(sides)
    cals.append(calibrate_num(sides))
    gt = pd.concat([ground_truth_drift(sides, key=key, tracker=tk) for tk in trackers],
                   ignore_index=True)
    gts.append(gt)
    ests.append(estimator_scan(sides, gt=gt[gt.tracker == trackers[-1]], trackers=trackers))
    eps.append(epoch_scan(sides, estimator=estimator, tracker=trackers[-1]))
    plot_population(sides, outdir=outdir, estimator=estimator)
    for tk in trackers:
      plot_collapse(sides, outdir=outdir, estimator=estimator, ref=tk)
  cal = pd.concat(cals, ignore_index=True)
  est = pd.concat(ests, ignore_index=True)
  write_tables(cal, est, outdir=outdir, gt=pd.concat(gts, ignore_index=True),
               ep=pd.concat(eps, ignore_index=True) if eps else None)
  swp.trim_pngs(outdir)
  print(f'\nnu_c validation figures and tables saved to {outdir}')
  return all_sides, cal, est


if __name__ == '__main__':
  main()
