# -*- coding: utf-8 -*-
# @Author: acharlet

'''
Visual comparison of the two emission computations over the gamma_c/gamma_m sweep:
  'fit'  -- get_shell_nuFnu, per-cell hydro reconstructed from smooth-BPL fits,
            worldline ODE, rarefaction cut at R_rar (working_cooling.py)
  'data' -- get_shell_nuFnu_fromData, actual per-cell snapshot histories, cooling
            fluence by quadrature, rarefaction taken from the data
            (working_cooling_data.py), with early_ana='shockfit' so both methods
            share their injection states and the comparison isolates the
            post-injection hydro treatment.

Both sweeps use the same observer grids per point (_nu_window / TB_MIN / NT depend
only on env and alpha), so every comparison here is pointwise -- no interpolation.
Run the sweeps first (sweep_gammacm.run_sweep(key, arr, method='fit'|'data'), each
cached in its own directory), then main() builds the figures.
Convention on every figure: DASHED = side A, SOLID = side B, colour = the sweep point.
Which side's flux NORMALISES the panels is a separate choice (norm_side), independent of
the linestyles.
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from environment import GAMMA_dir
import spectral_breaks as sb
import sweep_gammacm as swp
from plotting_functions import slope_label, transx
from sweep_gammacm import (load_sweep, method_outdir, _sweep_colors, _draw_order, nu_over_num,
    compute_fluence_spectrum, detect_rise_peak_tail, compute_efficiency,
    exit_onset_barT, rarefaction_off_barT, data_end_barT, run_sweep, trim_pngs,
    local_index, _hle_index, _index_panel, _spectra_series, _series_colors,
    LOG10RATIO_ARR, NU_TARGETS, NU_M_LABEL, NU_REF, Z_SHELL, DEFAULT_KEY,
    SPEC_LOGT, SPEC_SERIES_YSPAN, XLIM_LIN, XLIM_LOG, SPEC_MODES, _MODE_TITLE)

OUTDIR = os.path.join(GAMMA_dir, 'bin', 'Tools', 'figures', 'gammacm_sweep_compare')
YCLIP_DEC = 3.5           # decades below the highest curve shown on the spectral panels
RATIO_SPAN = (0.5, 2.)    # default y-range of the ratio panels (rescaled if exceeded)
YSPAN_LINLOG = 4          # decades of flux shown on the linear-time / log-flux variant,
                          # below the highest curve inside the time window
                          # XLIM_LIN (the linear-time bar{T}/bar{T}_f window) is imported from
                          # sweep_gammacm: one window for the whole suite, so these A/B
                          # variants read at the same scale as the single-method figures.
METHODS = ('data_rarcut', 'data')   # main's default pair: DATA-ONLY. The 'fit' method is no
                          # longer plotted anywhere, so nothing on the default path can
                          # trigger a fit sweep; pass `methods` to compare any other two.
LABELS = ('rf cut', 'full')
                          # (side A = dashed, side B = solid); every figure takes `labels`
                          # so the same machinery compares two METHODS (the default) or two
                          # runs of one method (see sweep_duration.py).
                          # SIDE B IS THE REFERENCE, in every use of this module: the full
                          # run vs the cut one (the default, and sweep_rarcut), the long run
                          # vs the short one (sweep_duration). Ratios therefore read B/A.
                          # WHICH SIDE NORMALISES the flux panels is a SEPARATE choice --
                          # `norm_side` ('A'|'B') on each plotting function, and never tied
                          # to the linestyles. Both curves of a point are always divided by
                          # the SAME number, so their vertical offset is the B/A ratio
                          # either way; norm_side only picks which curve passes through 1,
                          # i.e. what the figure is anchored on.
                          #   'B' anchors on the reference: the figure never rescales
                          #       itself when the side being tested changes (the function
                          #       default, and what a run-vs-run comparison wants).
                          #   'A' anchors on the prescription being tested. This is what
                          #       the cut-vs-full figures use (main below, sweep_rarcut):
                          #       the cut IS the model one would quote, so anchoring on it
                          #       makes the full run read directly as the extra emission
                          #       the cut leaves out.


def _regrid_onto(rd, rf):
  '''
  Put a data point on the fit point's (nub, Tb) grid: log-log interpolation of
  nuFnu along nu, linear-in-log along Tb. Only needed when the two sweeps'
  Nnu differ by the rounding of _nu_window's points-per-decade (the windows
  themselves are identical), never for a physical mismatch -- load_pairs checks
  alpha first, which is what actually pins the point's physics.
  '''
  out = dict(rd)
  y = np.log(np.maximum(rd['nuFnu'], 1e-300))
  if rd['nub'].shape != rf['nub'].shape or not np.allclose(rd['nub'], rf['nub']):
    y = np.array([np.interp(np.log(rf['nub']), np.log(rd['nub']), row) for row in y])
  if rd['Tb'].shape != rf['Tb'].shape or not np.allclose(rd['Tb'], rf['Tb']):
    y = np.array([np.interp(np.log(rf['Tb']), np.log(rd['Tb']), col) for col in y.T]).T
  out['nuFnu'] = np.exp(y)
  out['nub'], out['Tb'] = rf['nub'], rf['Tb']
  return out


def load_pairs(outdir_fit=None, outdir_data=None, alpha_rtol=1e-6):
  '''
  Load both cached sweeps and match their points on log10ratio.
  Returns a list of (r_fit, r_data) sorted by log10ratio; points present in only
  one sweep are reported and dropped.
  The physical invariant checked here is ALPHA, not the grid: alpha fixes the
  rescaled environment, so two points at the same log10ratio but different alpha
  come from different baseline envs (e.g. one sweep cached before a change to
  the setup) and must not be compared -- those points are dropped with a loud
  message telling you to recompute. Grids are then reconciled by _regrid_onto,
  which only ever corrects a +-1 rounding of Nnu.
  '''
  res_f = load_sweep(outdir_fit or method_outdir('fit'))
  res_d = load_sweep(outdir_data or method_outdir('data'))
  if not res_f or not res_d:
    raise RuntimeError('missing sweep cache: run sweep_gammacm.run_sweep(key, arr, '
                       'method=...) for both sides first (see sweep_compare.METHODS)')
  # provenance: alpha (checked below) pins the rescaled env, but two runs of the SAME
  # setup differing only in duration share it exactly -- only (key, method) tells those
  # apart, so an accidentally self-compared cache would otherwise look perfect.
  prov = lambda res: {(r.get('key', ''), r.get('method', '')) for r in res}
  pf, pd = prov(res_f), prov(res_d)
  if pf == pd and pf != {('', '')}:
    print(f'WARNING: both sides have the same provenance {sorted(pf)} -- you are '
          'comparing a cache against itself')
  if ('', '') in pf | pd:
    print("note: a cache predates key tagging (key=''); its simulation is assumed to be "
          'the one you asked for -- recompute it if unsure')
  by_logr = {round(r['log10ratio'], 6): r for r in res_d}
  pairs, unmatched, stale = [], [], []
  for rf in res_f:
    rd = by_logr.pop(round(rf['log10ratio'], 6), None)
    if rd is None:
      unmatched.append(rf['log10ratio'])
      continue
    if abs(rd['alpha'] - rf['alpha']) > alpha_rtol*abs(rf['alpha']):
      stale.append((rf['log10ratio'], rf['alpha'], rd['alpha']))
      continue
    if (rf['nub'].shape != rd['nub'].shape or rf['Tb'].shape != rd['Tb'].shape
        or not (np.allclose(rf['nub'], rd['nub']) and np.allclose(rf['Tb'], rd['Tb']))):
      rd = _regrid_onto(rd, rf)
    pairs.append((rf, rd))
  if unmatched or by_logr:
    print(f'unmatched sweep points: fit-only {unmatched}, data-only {sorted(by_logr)}')
  if stale:
    print('DIFFERENT alpha at the same target -> the two sweeps come from different '
          'baseline environments; recompute both with the current setup:')
    for lr, af, ad in stale:
      print(f'  log10ratio={lr:+.1f}: alpha fit={af:.6f} data={ad:.6f}')
  print(f'{len(pairs)} matched sweep points')
  return pairs


def _lc_at(r, nu_t):
  '''Lightcurve at the nub grid point closest to nu_t (fraction of nu_pk).'''
  inu = min(np.searchsorted(r['nub'], nu_t), len(r['nub']) - 1)
  return r['nuFnu'][:, inu]

def _peak_spectrum(r):
  '''Spectrum at this point's own lightcurve peak (per method, not a shared index).'''
  i_peak = detect_rise_peak_tail(r['Tb'], r['nub'], r['nuFnu'])[3].get('i_peak', 0)
  return r['nuFnu'][i_peak, :]

def _norm_label(labels, norm_side):
  """
  Name of the side whose flux normalises a comparison panel (and the norm_side guard).
  Kept separate from `labels`' own order so a figure can say which curve sits at 1
  without the reader having to know which side is dashed.
  """
  if norm_side not in ('A', 'B'):
    raise ValueError(f"norm_side must be 'A' or 'B', got {norm_side!r}")
  return labels[0] if norm_side == 'A' else labels[1]


def _ratio_ylim(ax, ratios):
  '''Symmetric-in-log ratio range, widened past RATIO_SPAN only if the data needs it.'''
  vals = np.concatenate([np.asarray(v)[np.isfinite(v) & (np.asarray(v) > 0.)] for v in ratios]) \
         if ratios else np.array([1.])
  if not vals.size:
    return
  lo = min(RATIO_SPAN[0], np.percentile(vals, 1)*0.9)
  hi = max(RATIO_SPAN[1], np.percentile(vals, 99)*1.1)
  ax.set_ylim(lo, hi)


def plot_efficiency_compare(pairs, outdir=OUTDIR, labels=LABELS):
  '''
  Radiative efficiency eps_rad = E_rad/E_inj vs the cooling regime, both sides,
  with the B/A ratio underneath. The energy budget is a pure sum over cells
  (no observer projection), so this panel isolates how much the two treatments
  differ in what the electrons actually radiate.
  '''
  la, lb = labels
  logr = np.array([rf['log10ratio'] for rf, _ in pairs], float)
  eff_f = np.array([compute_efficiency(rf) for rf, _ in pairs], float)
  eff_d = np.array([compute_efficiency(rd) for _, rd in pairs], float)
  print(f"\n{'log10(gc/gm)':>12} {'eps_rad '+la:>12} {'eps_rad '+lb:>13} {'ratio':>8} "
        f"{'E_rad B/A':>10} {'E_inj B/A':>10}")
  for (rf, rd), ef, ed in zip(pairs, eff_f, eff_d):
    print(f"{rf['log10ratio']:+12.0f} {ef:12.4f} {ed:13.4f} {ed/ef:8.4f} "
          f"{rd.get('E_rad', np.nan)/rf.get('E_rad', np.nan):10.4f} "
          f"{rd.get('E_inj', np.nan)/rf.get('E_inj', np.nan):10.4f}")

  fig, axs = plt.subplots(2, 1, figsize=(6.5, 6.), sharex=True,
                          gridspec_kw={'height_ratios': [2.2, 1]})
  axs[0].plot(logr, eff_f, 'o--', color='C0', label=la)
  axs[0].plot(logr, eff_d, 's-', color='C1', label=lb)
  axs[0].axhline(1., color='grey', ls=':', lw=.9)
  axs[0].set_ylabel('$\\varepsilon_{\\rm rad}=E_{\\rm rad}/E_{\\rm inj}$')
  axs[0].legend()
  axs[1].plot(logr, eff_d/eff_f, 'k.-')
  axs[1].axhline(1., color='grey', ls=':', lw=.9)
  axs[1].set_ylabel(f'{lb} / {la}')
  axs[1].set_xlabel('$\\log_{10}\\mathcal{C}$')
  fig.tight_layout()
  fig.savefig(os.path.join(outdir, 'radiative_efficiency_cmp.png'), dpi=300)
  plt.close(fig)


def plot_spectra_compare(pairs, kind='peak', mode='nu_m', outdir=OUTDIR, labels=LABELS,
    ratio=True, norm_side='B'):
  '''
  Spectra of every sweep point, both sides overlaid (dashed A / solid B), vs nu/nu_m.
  kind: 'peak' (spectrum at each curve's own lightcurve peak) | 'fluence'
    (time-integrated). mode: any of sweep_gammacm.SPEC_MODES -- 'nu_m' (normalised at
    nu_m), 'max' (peak-normalised) or 'eff' (peak-normalised, then multiplied by the
    point's radiative efficiency, which restores the energetics the two shape
    normalisations divide out and stacks the regimes by how much they actually radiate)
    -- same conventions as sweep_gammacm._plot_spectra_all, so the shapes are
    directly readable against the single-method figures. Under 'eff' the y-floor hangs
    off the FAINTEST point rather than the brightest, so every point still shows
    YCLIP_DEC decades of its own shape; the pair is scaled by the NORMALISING side's
    eps_rad, the same side its norm comes from.
  Normalisation is taken from ONE side's curve (norm_side, 'A' or 'B') and applied to
  both, so the vertical offset between a point's two curves IS the B/A ratio whichever
  side is picked; norm_side only sets which curve passes through 1 (see LABELS).
  ratio: draw the B/A ratio in a lower panel. It is redundant with that offset -- it
    only re-plots it on its own axis -- so it is worth its half of the figure ONLY when
    the two sides are close enough that the offset is hard to judge by eye. Off for the
    fluence spectra (see sweep_rarcut), where the curves separate visibly and the
    spectral shape is what the figure is for.
  '''
  if mode not in SPEC_MODES:
    raise ValueError(f'mode must be one of {SPEC_MODES}, got {mode!r}')
  la, lb = labels
  ln = _norm_label(labels, norm_side)
  get_spec = _peak_spectrum if kind == 'peak' else \
             (lambda r: compute_fluence_spectrum(r['Tb'], r['nuFnu']))
  colors, sm = _sweep_colors([rf for rf, _ in pairs])
  if ratio:
    fig, axs = plt.subplots(2, 1, figsize=(7.5, 7.), sharex=True,
                            gridspec_kw={'height_ratios': [2.4, 1]})
    ax_s, ax_r, cb_ax = axs[0], axs[1], axs
  else:
    fig, ax_s = plt.subplots(figsize=(7.5, 5.))
    ax_r, cb_ax = None, ax_s
  ymax, ypks, ratios, curves = 0., [], [], []
  for (rf, rd), c in _draw_order(zip(pairs, colors)):
    x = nu_over_num(rf)
    sf, sd = get_spec(rf), get_spec(rd)
    sn = sf if norm_side == 'A' else sd
    norm = sn[int(np.argmin(np.abs(x - 1.)))] if mode == 'nu_m' else sn.max()
    if mode == 'eff':
      # scaled by the NORMALISING side's eps_rad, the side the norm itself comes from:
      # the two differ by a few percent at most, and taking one from each would fold
      # that difference into the stacking on top of the offset it already sets
      eff = compute_efficiency(rf if norm_side == 'A' else rd)
      if not np.isfinite(eff) or eff <= 0.:
        continue              # no energy budget in this point's cache: cannot scale it
      norm /= eff
    if norm <= 0.:
      continue
    yf, yd = sf/norm, sd/norm
    curves.append((x, yf, yd, c))
    ypks.append(max(np.nanmax(yf), np.nanmax(yd)))
    ymax = max(ymax, ypks[-1])
    if ax_r is not None:
      with np.errstate(divide='ignore', invalid='ignore'):
        ratios.append(np.where(sf > 0., sd/sf, np.nan))
      ax_r.plot(x, ratios[-1], color=c, lw=.9)
  for x, yf, yd, c in curves:
    ax_s.loglog(x, yf, color=c, lw=1.1, ls='--')
    ax_s.loglog(x, yd, color=c, lw=1.1)
  if not curves:
    print(f'{kind}_spectra_cmp_norm-{mode}: nothing to plot'
          + (' (energies absent from the caches; re-run the sweeps with use_cache=False)'
             if mode == 'eff' else ''))
    plt.close(fig)
    return
  # 'eff' hangs the floor off the FAINTEST point, so each keeps YCLIP_DEC of its own shape
  # however far the efficiency scaling has pushed it down (as _plot_spectra_all does)
  ylo = (min(ypks) if mode == 'eff' else ymax)/10.**YCLIP_DEC if ymax > 0. else None
  if ylo is not None:
    # only a little headroom above the peak: the legend sits upper-LEFT, over the
    # low-frequency end where every curve is near the floor, so it needs no room here
    ax_s.set_ylim(ylo, ymax*1.5)
    xhi = max(x[np.nanmax([yf, yd], axis=0) > ylo].max() for x, yf, yd, _ in curves)
    ax_s.set_xlim(min(x[0] for x, _, _, _ in curves), 2.*xhi)
  sym = '\\nu F_\\nu' if kind == 'peak' else '\\nu \\mathcal{F}_\\nu'
  sub = f'({sym})_{{\\nu_\\mathrm{{m}}}}' if mode == 'nu_m' else f'({sym})_{{\\rm max}}'
  pre = '\\varepsilon_{\\rm rad}\\,' if mode == 'eff' else ''
  ax_s.set_ylabel(f'${pre}{sym}/{sub}$  ({ln} norm.)')
  # no nu = nu_m guide on either panel -- see sweep_gammacm._plot_spectra_all for why it
  # went from every spectrum figure. ax_r keeps its HORIZONTAL unity line: that one marks
  # the two sides agreeing, which is the whole point of the ratio panel.
  ax_s.plot([], [], 'k--', label=la); ax_s.plot([], [], 'k-', label=lb)
  ax_s.legend(loc='upper left', fontsize=9)
  if ax_r is not None:
    ax_r.axhline(1., color='grey', ls=':', lw=.9)
    ax_r.set_xscale('log'); ax_r.set_yscale('log')
    _ratio_ylim(ax_r, ratios)
    ax_r.set_ylabel(f'{lb} / {la}')
    ax_r.set_xlabel(NU_M_LABEL)
  else:
    ax_s.set_xlabel(NU_M_LABEL)
  ax_s.set_title(f'{"Peak" if kind=="peak" else "Time-integrated"} spectra, '
                 f'{la} vs {lb} ({_MODE_TITLE[mode]})')
  fig.colorbar(sm, ax=cb_ax, label='log$_{10}\\mathcal{C}$')
  fig.savefig(os.path.join(outdir, f'{kind}_spectra_cmp_norm-{mode}.png'), dpi=300)
  plt.close(fig)


def plot_spectral_evolution_compare(pairs, barT_f, outdir=OUTDIR, labels=LABELS,
    yspan=SPEC_SERIES_YSPAN, logt=SPEC_LOGT, norm_side='B'):
  '''
  Spectral evolution, both sides overlaid: one figure per sweep point, showing the
  instantaneous spectra of a series of observed times vs nu/nu_m on a SINGLE panel.
  Colour = time bin (the SPEC_LOGT grid and colours of
  sweep_gammacm.plot_spectra_per_regime), linestyle = side (dashed A / solid B), so the
  single-method figures stay readable against these.

  Both sides are sampled at the SAME observed times -- SPEC_LOGT, logarithmically spaced
  bins of bar{T}/bar{T}_f -- so a colour pair is one instant seen twice and the offset
  between its two curves is a spectral difference and nothing else. That is the change from
  the rise/peak/tail sampling this replaced: there each side was taken at ITS OWN
  peak-relative phases (detect_rise_peak_tail per side), so a phase that had moved folded a
  timing difference into the same offset, and the legend had to print both times to admit
  it. Flux is normalised to the brightest bin of the norm_side side (see LABELS), so the
  vertical offset between a bin's two curves IS the B/A ratio, read off the same axis as
  the shapes.

  No ratio sub-panel: the offset between the dashed and solid curve of one colour already
  carries it, and dropping it gives the spectra the full figure height -- these run many
  decades in nu and the shape comparison is what the figure is for. plot_summary_ratios
  and summary_ratios are where the numbers live.

  Dropped vs the single-method version: the paired-BPL guides and the measured-regime
  annotations. They belong to reading one spectrum's shape; here the question is what
  changes between the two sides, and doubling the dashed guides would collide with the
  dashed side-A curves.
  '''
  la, lb = labels
  ln = _norm_label(labels, norm_side)
  ylo = 10.**(-yspan)
  cols = _series_colors(logt)
  for rf, rd in pairs:
    # each side is sampled on its OWN observer grid (_spectra_series drops a bin the grid
    # does not reach), and a bin is drawn only where BOTH sides have one: a lone curve in
    # a bin colour would read as a comparison when there is nothing to compare it against
    sa = {k: i for k, _, i in _spectra_series(rf, barT_f, logt)}
    sb_ = {k: i for k, _, i in _spectra_series(rd, barT_f, logt)}
    ks = [k for k in sorted(sa) if k in sb_]
    if not ks:
      continue
    x = nu_over_num(rf)
    # one side normalises (norm_side, see LABELS), at ITS OWN rows
    rn, sn = (rf, sa) if norm_side == 'A' else (rd, sb_)
    norm = max(np.nanmax(rn['nuFnu'][sn[k], :]) for k in ks)
    if not (norm > 0.):
      continue

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    sps, handles, labs = [], [], []
    for k in ks:
      c = cols[k]
      sf, sd = rf['nuFnu'][sa[k], :]/norm, rd['nuFnu'][sb_[k], :]/norm
      ax.loglog(x, sf, color=c, lw=1.2, ls='--')
      ax.loglog(x, sd, color=c, lw=1.2)
      sps += [sf, sd]
      # the time key is a SOLID swatch whatever the linestyles and whichever side
      # normalises: it names the COLOUR only, and the two black keys carry the
      # dashed/solid convention. Taking it from one of the drawn lines instead made the
      # time legend read as if that side were the time.
      (h,) = ax.plot([], [], color=c, lw=1.2)
      handles.append(h)
      labs.append(f'{logt[k]:+.0f}')
    ax.set_ylim(ylo, 3.)
    vis = np.any(np.array(sps) > ylo, axis=0)      # clip x to the visible spectra
    if vis.any():
      ax.set_xlim(x[vis].min()/3., x[vis].max()*3.)
    (ha,) = ax.plot([], [], 'k--')                # side keys, colour-neutral
    (hb,) = ax.plot([], [], 'k-')
    # two legends, because one box cannot carry two keys under a single title: the time
    # bins go under their axis name in two columns along the bottom centre (the panel
    # floor between the rising and falling bundles, as in plot_spectra_per_regime), the
    # sides in the upper right -- these curves peak near nu_m and run out to their nu_M
    # cutoff many decades higher, so that corner is empty too
    ax.add_artist(ax.legend(handles, labs, title='$\\log_{10}(\\bar{T}/\\bar{T}_f)$',
                            ncol=2, fontsize=9, title_fontsize=9, loc='lower center'))
    ax.legend([ha, hb], [la, lb], fontsize=9, loc='upper right')
    ax.set_ylabel(f'$\\nu F_\\nu/(\\nu F_\\nu)_{{\\rm pk}}$  ({ln} norm.)')
    ax.set_xlabel(NU_M_LABEL)
    ax.set_title(f'Spectral evolution, {la} vs {lb}, '
                 f'$\\log_{{10}}\\mathcal{{C}}={rf["log10ratio"]:+.0f}$')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir,
        f'spectrum_evolution_cmp_logr={rf["log10ratio"]:+.1f}.png'), dpi=300)
    plt.close(fig)


def plot_lightcurve_compare(pairs, barT_f, barT_off=None, nu_targets=NU_TARGETS,
    outdir=OUTDIR, labels=LABELS, barT_end=None, scale='log', xlim_lin=XLIM_LIN,
    norm_side='B', slope=True):
  '''
  Lightcurves at a fixed fraction nu_t of each point's own peak frequency, both
  sides overlaid, time normalised to the shell-crossing bar{T}_f, flux to the peak of
  the norm_side side of that point (see LABELS), so that side peaks at 1 and the other
  reads directly as a fraction of it. The B/A ratio panel is unaffected by the choice:
  it is a ratio of the two curves, and both are divided by the same number.
  One figure per nu_t: lightcurves on top, then (slope=True) the local temporal index
  of both sides, then the B/A ratio -- the panel where the early rise (cadence /
  injection) and the rarefaction-tail treatment show up.
  slope: draw the middle panel, d ln(nuFnu)/d ln(bar{T}) at every grid point
  (sweep_gammacm.local_index), same linestyles as the flux panel. It is the SHAPE
  counterpart of the ratio panel below it: the ratio says how much emission a
  prescription costs, the index says what it does to the decay -- where the cut
  steepens away from the reference and whether it comes back to the same
  high-latitude asymptote -(2+p/2) (the dash-dotted guide). Invariant under the flux
  normalisation, so norm_side does not move it.
  barT_off: rarefaction cut-off band (rarefaction_off_barT); a pure hydro/geometry
  quantity of the simulation, so the same band annotates both sides.
  barT_end: optional ((first,last)_A, (first,last)_B) from data_end_barT -- where
  each side's cells run out of snapshots. For a run-vs-run comparison this, not
  barT_off, is what sets the divergence: drawn as one vertical marker per side
  (dashed A / solid B) at that side's LAST cell.
  scale: which axes the top panel uses, each written to its own file so the
  variants coexist.
    'log'    log-log (default, the wide view: XLIM_LOG = 1e-3..1e3 in bar{T}/bar{T}_f,
             i.e. out to where the runs themselves end)
    'linlog' linear time, log flux -- a linear clock on the decay, where the whole
             cut-vs-full difference lives; the readable one for these comparisons
    'lin'    both axes linear -- the pulse shape (rise/peak/early decay). The late
             divergence sits at 1e-2..1e-5 of peak and is flat against zero here,
             so read it off the ratio panel, not the top one.
  xlim_lin: time range of the two linear variants, in bar{T}/bar{T}_f.
  '''
  if scale not in ('log', 'linlog', 'lin'):
    raise ValueError(f"scale must be 'log', 'linlog' or 'lin', got {scale!r}")
  la, lb = labels
  ln = _norm_label(labels, norm_side)
  logx, logy = (scale == 'log'), (scale != 'lin')
  suff = '' if scale == 'log' else f'_{scale}'
  colors, sm = _sweep_colors([rf for rf, _ in pairs])
  xoff = tuple(b/barT_f for b in barT_off) if (barT_off and barT_f > 0.) else None
  xend = [(b[1]/barT_f if b else None) for b in barT_end] if barT_end else None
  a_hle = _hle_index([rf for rf, _ in pairs])
  hr = [2.4, 1.1, 1] if slope else [2.4, 1]
  for nu_t in nu_targets:
    fig, axs = plt.subplots(len(hr), 1, figsize=(7.5, 8.8 if slope else 7.), sharex=True,
                            gridspec_kw={'height_ratios': hr})
    ax_f, ax_s, ax_r = axs[0], (axs[1] if slope else None), axs[-1]
    ratios, ymax_win = [], 0.
    for (rf, rd), c in _draw_order(zip(pairs, colors)):
      x = (rf['Tb'] - 1.)/barT_f
      lf, ld = _lc_at(rf, nu_t), _lc_at(rd, nu_t)
      pk_b = ld.max()        # B's peak: the scale the ratio mask stands on, so that the
                             # ratio panel does not move when norm_side does
      pk = lf.max() if norm_side == 'A' else pk_b   # the anchor (see LABELS)
      if pk <= 0. or pk_b <= 0. or barT_f <= 0.:
        continue
      ax_f.plot(x, lf/pk, color=c, lw=1.1, ls='--')
      ax_f.plot(x, ld/pk, color=c, lw=1.1)
      if ax_s is not None:
        ax_s.plot(x, local_index(x, lf), color=c, lw=.9, ls='--')
        ax_s.plot(x, local_index(x, ld), color=c, lw=.9)
      if scale != 'log':      # peak inside the linear window, over BOTH sides. The
        win = (x >= xlim_lin[0]) & (x <= xlim_lin[1])   # normalising side peaks at 1 on the
        if win.any():                                   # full grid, but its peak can fall
                                                        # outside this window
          ymax_win = max(ymax_win, float(np.max(lf[win]/pk)), float(np.max(ld[win]/pk)))
      with np.errstate(divide='ignore', invalid='ignore'):
        rr = np.where(lf > 1e-6*pk_b, ld/lf, np.nan)  # ratio only where side A has flux
      ratios.append(rr)
      ax_r.plot(x, rr, color=c, lw=.9)
    if ax_s is not None:
      _index_panel(ax_s, a_hle)
    for ax in axs:
      ax.axvline(1., color='grey', ls=':', lw=.7)
      if xoff is not None:
        # the band alone, as in sweep_gammacm.plot_lightcurve_shape: its right edge is
        # bar{T}_rf. The black dashed line that used to be drawn there is gone, so where a
        # side stops exactly AT the cut-off (sweep_rarcut passes barT_off as side A's end)
        # that x now carries only the crimson end-of-data line below.
        ax.axvspan(xoff[0], xoff[1], color='grey', alpha=0.15, lw=0, zorder=0)
      if xend is not None:
        for xe, ls in zip(xend, ('--', '-')):
          if xe is not None:
            ax.axvline(xe, color='crimson', ls=ls, lw=.9, alpha=.8)
    ax_f.set_xscale('log' if logx else 'linear')
    ax_f.set_yscale('log' if logy else 'linear')
    if scale == 'log':
      ax_f.set_ylim(ymin=1e-8)
      ax_f.set_xlim(*XLIM_LOG)
    else:
      ax_f.set_xlim(*xlim_lin)
      # scale the flux axis to what is actually inside the linear time window, not to
      # the full grid: the rise starts ~8 decades down at bar{T} -> 0 and a fixed floor
      # would spend most of the panel on empty space (linlog) or clip side B's peak (lin).
      ymax_win = ymax_win or 1.
      ax_f.set_ylim(ymax_win*10.**-YSPAN_LINLOG, ymax_win*1.6) if logy \
          else ax_f.set_ylim(0., ymax_win*1.05)
    ax_f.set_ylabel(f'$\\nu F_\\nu/(\\nu F_\\nu)_{{\\rm max}}$ ({ln} norm.)')
    ax_f.plot([], [], 'k--', label=la); ax_f.plot([], [], 'k-', label=lb)
    ax_f.legend(loc='upper right', fontsize=9)   # the decay tail leaves this corner free
    ax_r.axhline(1., color='grey', ls=':', lw=.9)
    ax_r.set_yscale('log' if logy else 'linear')
    _ratio_ylim(ax_r, ratios)
    ax_r.set_ylabel(f'{lb} / {la}')
    ax_r.set_xlabel('$\\bar{T}/\\bar{T}_f$')
    ax_f.set_title(f'Lightcurve at $\\nu={nu_t:g}\\,\\nu_{{\\rm pk}}$, {la} vs {lb}')
    fig.colorbar(sm, ax=axs, label='log$_{10}\\mathcal{C}$')
    fig.savefig(os.path.join(outdir, f'lightcurve_cmp{suff}_nu={nu_t:g}.png'), dpi=300)
    plt.close(fig)


def summary_ratios(pairs, nu_ref=NU_REF):
  '''
  Per sweep point, the B/A ratio of the observables one would quote:
  peak flux and peak time at nu_ref, lightcurve fluence at nu_ref, total
  (spectrum-integrated) fluence, and eps_rad. Returns a dict of arrays.
  '''
  out = {k: [] for k in ('logr', 'peak_flux', 'peak_time', 'fluence_nu', 'fluence_tot', 'eps_rad')}
  for rf, rd in pairs:
    lf, ld = _lc_at(rf, nu_ref), _lc_at(rd, nu_ref)
    Tb = rf['Tb']
    out['logr'].append(rf['log10ratio'])
    out['peak_flux'].append(ld.max()/lf.max())
    out['peak_time'].append((Tb[np.argmax(ld)] - 1.)/(Tb[np.argmax(lf)] - 1.))
    out['fluence_nu'].append(np.trapezoid(ld, Tb)/np.trapezoid(lf, Tb))
    ff = compute_fluence_spectrum(rf['Tb'], rf['nuFnu'])
    fd = compute_fluence_spectrum(rd['Tb'], rd['nuFnu'])
    lnx = np.log(rf['nub'])
    out['fluence_tot'].append(np.trapezoid(fd, lnx)/np.trapezoid(ff, lnx))
    out['eps_rad'].append(compute_efficiency(rd)/compute_efficiency(rf))
  return {k: np.array(v, float) for k, v in out.items()}


def plot_summary_ratios(pairs, outdir=OUTDIR, flag_tol=0.15, labels=LABELS):
  '''
  One-glance answer to 'where do the two sides diverge': every B/A observable
  ratio vs the cooling regime. Points beyond flag_tol are printed.
  '''
  la, lb = labels
  s = summary_ratios(pairs)
  labels = {'peak_flux': 'peak flux', 'peak_time': 'peak time',
            'fluence_nu': f'fluence at $\\nu_{{\\rm pk}}$', 'fluence_tot': 'total fluence',
            'eps_rad': '$\\varepsilon_{\\rm rad}$'}
  print(f"\n{'log10(gc/gm)':>12}" + ''.join(f'{k:>14}' for k in labels))
  for i, lr in enumerate(s['logr']):
    print(f'{lr:+12.0f}' + ''.join(f'{s[k][i]:14.4f}' for k in labels))
  flagged = [(k, s['logr'][i], s[k][i]) for k in labels
             for i in range(len(s['logr'])) if abs(s[k][i] - 1.) > flag_tol]
  if flagged:
    print(f'\ndivergences beyond {flag_tol:.0%}:')
    for k, lr, v in flagged:
      print(f'  {labels[k]:>22} at log10(gc/gm)={lr:+.0f}: {v:.3f}')
  else:
    print(f'\nno {lb}/{la} ratio deviates by more than {flag_tol:.0%}')

  fig, ax = plt.subplots(figsize=(7., 4.5))
  for (k, lab), m in zip(labels.items(), 'osd^v'):
    ax.plot(s['logr'], s[k], m + '-', label=lab, ms=5)
  ax.axhline(1., color='grey', ls=':', lw=.9)
  ax.axhspan(1.-flag_tol, 1.+flag_tol, color='grey', alpha=0.12, lw=0, zorder=0)
  ax.set_xlabel('$\\log_{10}\\mathcal{C}$')
  ax.set_ylabel(f'{lb} / {la}')
  ax.set_title(f'{lb} vs {la}, over the cooling regimes')
  ax.legend(fontsize=9, ncol=2)
  fig.tight_layout()
  fig.savefig(os.path.join(outdir, 'summary_ratios.png'), dpi=300)
  plt.close(fig)
  return s


def _fluence_before_after(r, barT_cut):
  '''
  Frequency- and time-integrated fluence of one sweep point, split at bar{T}_cut:
  (before, after). Trapezoid in bar{T} then in ln(nu) -- the same conventions as
  compute_fluence_spectrum and summary_ratios' fluence_tot, so the two agree by
  construction (before + after = fluence_tot). The split point is inserted by
  interpolation, so it does not have to fall on the observer grid.
  '''
  Tb, lnx = r['Tb'], np.log(r['nub'])
  F = np.trapezoid(r['nuFnu'], lnx, axis=1)          # nu-integrated lightcurve
  tot = np.trapezoid(F, Tb)
  bt = Tb - 1.
  if barT_cut is None or barT_cut <= bt[0]:
    return 0., tot
  if barT_cut >= bt[-1]:
    return tot, 0.
  i = int(np.searchsorted(bt, barT_cut))
  x = np.concatenate([bt[:i], [barT_cut]])
  y = np.concatenate([F[:i], [np.interp(barT_cut, bt, F)]])
  before = float(np.trapezoid(y, x))
  return before, float(tot - before)


def fluence_split(pairs, barT_cut, outdir=OUTDIR, labels=LABELS, cut_label='cut',
    cut_math=None):
  '''
  How much of each side's total fluence is emitted BEFORE vs AFTER bar{T}_cut.
  Post-hoc on the cached (Tb, nuFnu) -- nothing is recomputed. Built for the
  run-vs-run comparison, where barT_cut is the rarefaction cut-off (the last
  bar{T} at which the modelled wave darkens a cell, rarefaction_off_barT[1]):
  the 'after' column is then the emission the fit method discards by construction
  and that only a long-enough simulation can show. Reports the numbers and the
  stacked composition; whether that late emission is physical is a separate
  question (the post-crash decay is position-dependent, see
  rarefaction_tail_exploration.py).
  cut_label names the cut in the printed table, cut_math on the figure (defaults to
  cut_label; pass the mathtext form when the two should differ).
  Returns a dict of arrays.
  '''
  la, lb = labels
  cut_math = cut_label if cut_math is None else cut_math
  out = {k: [] for k in ('logr', 'a_before', 'a_after', 'b_before', 'b_after')}
  for rf, rd in pairs:
    ab, aa = _fluence_before_after(rf, barT_cut)
    bb, ba = _fluence_before_after(rd, barT_cut)
    out['logr'].append(rf['log10ratio'])
    out['a_before'].append(ab); out['a_after'].append(aa)
    out['b_before'].append(bb); out['b_after'].append(ba)
  s = {k: np.array(v, float) for k, v in out.items()}

  print(f'\nfluence split at bar_T = {barT_cut:.4f} ({cut_label})')
  print(f"{'log10(gc/gm)':>12} {'%after '+la:>12} {'%after '+lb:>12} "
        f"{'tot '+lb+'/'+la:>12} {'before B/A':>12}")
  for i, lr in enumerate(s['logr']):
    ta = s['a_before'][i] + s['a_after'][i]
    tb = s['b_before'][i] + s['b_after'][i]
    print(f'{lr:+12.0f} {100*s["a_after"][i]/ta:12.2f} {100*s["b_after"][i]/tb:12.2f} '
          f'{tb/ta:12.4f} {s["b_before"][i]/s["a_before"][i]:12.4f}')

  fig, axs = plt.subplots(2, 1, figsize=(7., 6.5), sharex=True)
  w = 0.36
  for ax, (bef, aft), lab in zip(axs, [(s['a_before'], s['a_after']),
                                       (s['b_before'], s['b_after'])], labels):
    tot = bef + aft
    ax.bar(s['logr'], bef/tot, width=2*w, color='C0', label=f'$\\bar T<$ {cut_math}')
    ax.bar(s['logr'], aft/tot, width=2*w, bottom=bef/tot, color='C3',
           label=f'$\\bar T>$ {cut_math}')
    ax.set_ylabel(f'fluence fraction\n({lab})')
    ax.set_ylim(0., 1.)
  axs[0].legend(fontsize=9, loc='lower left')
  axs[1].set_xlabel('$\\log_{10}\\mathcal{C}$')
  axs[0].set_title(f'Where the fluence is emitted, relative to $\\bar T$ = {cut_math}')
  fig.tight_layout()
  fig.savefig(os.path.join(outdir, 'fluence_split.png'), dpi=300)
  plt.close(fig)
  return s


# --- low-energy index of the time-integrated spectra ----------------------------------
# The integrated measures above (eps_rad, total fluence, peak flux/time, the fluence split)
# all come out within a few percent, which reads as "the cut barely matters". They measure
# the AREA. The fluence spectra figure shows the cut changing the SHAPE -- its curves peel
# away from the full ones toward low frequency -- and that is what these three functions put
# a number on. See spectral_breaks.fluence_low_slope for the measurement and why neither
# breaks_from_segments nor free_slopes can be used on a time-integrated spectrum.
A_LO_ASYMP = 4./3.        # nuFnu index below both breaks, the instantaneous-spectrum value
                          # (= F_nu ~ nu^(1/3), Band alpha = -2/3, the synchrotron
                          # "line of death"); the cut side lands on it, the full side does not


def _shell_nu_break(env, x=None):
  '''
  Lower spectral break min(nu_m, nu_c) on the nu/nu_m axis of nu_over_num, i.e. the
  frequency below which the low-energy asymptote lives. Fast cooling puts it at
  (gma_c/gma_m)**2, slow cooling at nu_m itself.
  '''
  return float(min(1., (env.gma_c/env.gma_m)**2))


def _total_nu_break(env):
  '''
  Same, for the RS+FS SUM on the shared (RS-normalised) axis: the sum only reaches its
  low-energy asymptote below BOTH shells' breaks, so the lower of the two governs. The FS
  cache carries the RS env, with its own nu_m a factor fac_nu below the RS one -- on this
  run that puts the FS break ~15x lower, so it is the FS that sets the bound.
  '''
  b_rs = _shell_nu_break(env)
  if not all(hasattr(env, k) for k in ('gma_cFS', 'gma_mFS', 'fac_nu')):
    return b_rs
  return float(min(b_rs, min(1., (env.gma_cFS/env.gma_mFS)**2)/env.fac_nu))


def fluence_series(pairs, spec_key='nuFnu', x_key=None, nu_break=_shell_nu_break, **kw):
  '''
  The time-integrated spectrum of both sides of every pair, plus its measured low-energy
  indices. One entry per regime, in sweep order.

  spec_key/x_key let the same path serve the per-shell pairs from load_pairs (dicts with
  'nuFnu', frequency axis rebuilt by nu_over_num) and the RS+FS totals from
  sweep_shells.load_shell_pairs (dicts with 'nuFnu_tot' and a precomputed 'x').
  nu_break is called on the point's env to bound the asymptote scan (see
  spectral_breaks.fluence_low_slope); pass _total_nu_break for the sum.

  Returns a list of dicts: logr, x, sp_a, sp_b, nu_break, fa, fb (the two
  fluence_low_slope results).
  '''
  out = []
  for ra, rb in pairs:
    x = np.asarray(ra[x_key], float) if x_key else nu_over_num(ra)
    sp_a = compute_fluence_spectrum(ra['Tb'], ra[spec_key])
    sp_b = compute_fluence_spectrum(rb['Tb'], rb[spec_key])
    nb = nu_break(ra['env'])
    out.append(dict(logr=float(ra['log10ratio']), x=x, sp_a=sp_a, sp_b=sp_b, nu_break=nb,
                    fa=sb.fluence_low_slope(x, sp_a, nu_break=nb, **kw),
                    fb=sb.fluence_low_slope(x, sp_b, nu_break=nb, **kw)))
  return out


def fluence_slope_table(series, labels=LABELS, band_ref=3., verbose=True):
  '''
  The low-energy indices of the time-integrated spectra, both sides, one row per regime.

  The nuFnu index a is primary (4/3 below both breaks); beta = a - 1 is the F_nu index and
  Band alpha = a - 2, both derived columns since a difference in a is the same difference
  in either. `resolved` = the asymptote was both in band and converged -- read a_inf only
  where it is True; elsewhere the value describes whatever segment the grid floor covers
  (the fast-cooling nu^(1/2) one, in deep fast cooling) and is a lower bound at best.

  Returns a pandas DataFrame; prints it when verbose.
  '''
  la, lb = labels
  rows = []
  for e in series:
    row = {'logr': e['logr'], 'nu_break': e['nu_break']}
    for f, tag in ((e['fb'], 'full'), (e['fa'], 'cut')):     # B = reference = full
      res = bool(f['converged'] and f['in_band'])
      row.update({f'a_{tag}': f['a_inf'], f'resolved_{tag}': res,
                  f'inband_{tag}': bool(f['in_band']), f'drift_{tag}': f['conv_diff'],
                  f'nulo_{tag}': f['nu_lo_inf'], f'nupk_{tag}': f['x_pk']})
      for d, v in f['a_band'].items():
        row[f'aband{d:g}_{tag}'] = v
        # is that window actually ON the low-energy segment? Its top sits d decades below
        # the peak, which in slow cooling is still far ABOVE nu_m -- there the peak-anchored
        # index measures the mid segment instead, and must not be read as a low-energy one.
        row[f'bandlow{d:g}_{tag}'] = bool(f['x_pk']/10.**d
                                          <= e['nu_break']/sb.FREE_DFAC)
    row['d_a'] = row['a_cut'] - row['a_full']
    row['d_aband'] = row[f'aband{band_ref:g}_cut'] - row[f'aband{band_ref:g}_full']
    row['resolved'] = row['resolved_full'] and row['resolved_cut']
    row['bandlow'] = (row[f'bandlow{band_ref:g}_full']
                      and row[f'bandlow{band_ref:g}_cut'])
    for tag in ('full', 'cut'):        # same number, three conventions
      row[f'beta_{tag}'] = row[f'a_{tag}'] - 1.
      row[f'alpha_{tag}'] = row[f'a_{tag}'] - 2.
    rows.append(row)
  tab = pd.DataFrame(rows).sort_values('logr').reset_index(drop=True)
  if verbose:
    print(f'\nlow-energy index of the time-integrated spectra ({la} vs {lb})')
    print(f'  a = d log(nuFnu)/d log(nu);  F_nu index beta = a-1;  Band alpha = a-2')
    print(f"{'log10(gc/gm)':>12} {'a '+lb:>9} {'a '+la:>9} {'delta':>8} {'res?':>5} "
          f"{'drift '+lb:>10} {'drift '+la:>10} {'a_band3 '+lb:>10} {'a_band3 '+la:>10}")
    for _, r in tab.iterrows():
      print(f"{r['logr']:+12.0f} {r['a_full']:9.3f} {r['a_cut']:9.3f} {r['d_a']:+8.3f} "
            f"{'yes' if r['resolved'] else 'NO':>5} {r['drift_full']:10.3f} "
            f"{r['drift_cut']:10.3f} {r[f'aband{band_ref:g}_full']:10.3f} "
            f"{r[f'aband{band_ref:g}_cut']:10.3f}")
    res = tab[tab['resolved']]
    if len(res):
      print(f"  where resolved: {lb} a = {res['a_full'].min():.3f}..{res['a_full'].max():.3f} "
            f"(Band alpha {res['alpha_full'].min():+.3f}..{res['alpha_full'].max():+.3f}), "
            f"{la} a = {res['a_cut'].min():.3f}..{res['a_cut'].max():.3f} "
            f"(alpha {res['alpha_cut'].min():+.3f}..{res['alpha_cut'].max():+.3f})")
    fmt = lambda v: ', '.join(f'{x:+.0f}' for x in v)
    off = tab[~(tab['inband_full'] & tab['inband_cut'])]['logr'].tolist()
    drift = tab[(tab['inband_full'] & tab['inband_cut'])
                & ~tab['resolved']]['logr'].tolist()
    if off:
      print(f'  asymptote OFF-GRID at log10(gc/gm) = {fmt(off)} -- the break sits at or '
            'below the grid floor, so no window can stand off below it; the value quoted '
            'is the segment that IS in band')
    if drift:
      print(f'  asymptote IN BAND but still drifting at log10(gc/gm) = {fmt(drift)} -- too '
            'few decades below the break to converge; the value is a lower bound')
  return tab


def plot_fluence_slope_profile(series, outdir=OUTDIR, labels=LABELS, fname=None,
    title_extra=''):
  '''
  The local log-log slope of the time-integrated spectrum vs nu/nu_m, one curve per regime,
  both sides overlaid (dashed A / solid B) -- the whole picture behind the single number in
  fluence_slope_table. Guides at 4/3 (below both breaks), 1/2 (fast-cooling segment) and 1
  (where the full runs actually saturate). Each point's lower break is marked on the axis;
  the slope must be read to the LEFT of it.
  '''
  la, lb = labels
  colors, sm = _sweep_colors([{'log10ratio': e['logr']} for e in series])
  fig, ax = plt.subplots(figsize=(7.5, 5.))
  for e, c in _draw_order(zip(series, colors)):
    for f, ls in ((e['fa'], '--'), (e['fb'], '-')):
      ax.plot(10.**f['prof_lx'], f['prof_slope'], color=c, lw=1.1, ls=ls)
    ax.plot([e['nu_break']], [A_LO_ASYMP], marker='v', color=c, ms=4, clip_on=False)
  for y, lab in ((A_LO_ASYMP, '4/3'), (1., '1'), (.5, '1/2')):
    ax.axhline(y, color='grey', ls=':', lw=.8)
    ax.annotate(lab, xy=(0.997, y), xycoords=transx(ax), fontsize=8, color='grey',
                ha='right', va='bottom')
  ax.set_xscale('log')
  ax.set_xlim(min(10.**e['fb']['prof_lx'][0] for e in series), 2.)
  ax.set_ylim(-0.2, 1.6)
  ax.set_xlabel(NU_M_LABEL)
  ax.set_ylabel(slope_label('$\\nu \\mathcal{F}_\\nu$'))
  ax.plot([], [], 'k--', label=la); ax.plot([], [], 'k-', label=lb)
  ax.plot([], [], 'kv', ms=4, ls='none', label='$\\min(\\nu_\\mathrm{m},\\nu_\\mathrm{c})$')
  ax.legend(loc='lower left', fontsize=9)
  ax.set_title(f'Low-energy slope of the time-integrated spectra, {la} vs {lb}{title_extra}')
  fig.colorbar(sm, ax=ax, label='log$_{10}\\mathcal{C}$')
  fig.savefig(os.path.join(outdir, fname or 'fluence_slope_profile.png'), dpi=300)
  plt.close(fig)


def plot_fluence_slopes_vs_regime(tables, outdir=OUTDIR, labels=LABELS, kind='asymptote',
    band_ref=3., fname=None):
  """
  The headline: the low-energy index vs the cooling regime, both sides, for each entry of
  `tables` = {series name: (DataFrame, colour)}. Colour = series, linestyle = side, and a
  Band-alpha twin axis on the right.

  ONE panel. The cut - full difference used to sit under it, but it is the vertical gap
  between a series' two curves, already on the axis above and against the 4/3 guide that
  gives it its meaning; on its own axis it was the same information a second time, at the
  cost of half the figure. The number itself lives in fluence_slope_table's `d_a` /
  `d_aband` columns and in the CSV.

  kind='asymptote' plots a_inf, with filled markers where it is resolved and open where it
  is not -- an open marker is a lower bound on whatever segment is in band, not a measured
  asymptote. kind='band' plots the peak-anchored index instead, band_ref decades below the
  nuFnu peak. The two get SEPARATE figures on purpose: they measure different segments (in
  slow cooling the peak-anchored window sits on the mid segment and drops to ~0.25), so
  overlaying them puts two unrelated quantities on one axis and wrecks its scale.
  """
  from matplotlib.patches import Patch
  la, lb = labels
  asym = kind == 'asymptote'
  col_of = (lambda tag: f'a_{tag}') if asym else (lambda tag: f'aband{band_ref:g}_{tag}')
  ok_of = (lambda tag: f'resolved_{tag}') if asym else (lambda tag: f'bandlow{band_ref:g}_{tag}')
  fig, ax = plt.subplots(figsize=(7.5, 5.))
  for name, (tab, col) in tables.items():
    for tag, ls in (('cut', '--'), ('full', '-')):
      x, y, r = tab['logr'].values, tab[col_of(tag)].values, tab[ok_of(tag)].values
      ax.plot(x, y, ls, color=col, lw=1.2, zorder=2)
      ax.plot(x[r], y[r], 'o', color=col, ms=5.5, zorder=3)
      ax.plot(x[~r], y[~r], 'o', mfc='none', mec=col, ms=5.5, zorder=3)
  for y in (A_LO_ASYMP, 1.):
    ax.axhline(y, color='grey', ls=':', lw=.8)
  ax.annotate('4/3  (single-particle asymptote, Band $\\alpha=-2/3$)',
              xy=(0.985, A_LO_ASYMP + .012), xycoords=transx(ax), fontsize=8,
              color='grey', ha='right')
  ax.annotate('1  (Band $\\alpha=-1$)', xy=(0.985, 1. + .012),
              xycoords=transx(ax), fontsize=8, color='grey', ha='right')
  ax.set_ylabel('$a = $ d$\\log(\\nu\\mathcal{F}_\\nu)/$d$\\log\\nu$')
  ax.set_xlabel('$\\log_{10}\\mathcal{C}$')
  # series keys are colour SWATCHES, not lines: a coloured line would collide with the
  # dashed/solid side keys, and a filled marker with the resolved/not-resolved one
  keys = [Patch(color=col, label=name) for name, (_, col) in tables.items()]
  keys.append(plt.Line2D([], [], color='k', ls='--', label=la))
  keys.append(plt.Line2D([], [], color='k', ls='-', label=lb))
  keys.append(plt.Line2D([], [], color='k', marker='o', mfc='none', ls='none', ms=5.5,
              label='asymptote not resolved' if asym else 'window above the break'))
  # headroom for the legend: with the difference panel gone the curves fill the whole
  # figure, and upper-left is where they are lowest in BOTH kinds (fast cooling)
  lo, hi = ax.get_ylim()
  ax.set_ylim(lo, hi + .3*(hi - lo))
  ax.legend(handles=keys, fontsize=9, loc='upper left', ncol=2)
  what = ('asymptotic' if asym else f'{band_ref:g} decades below the peak')
  ax.set_title(f'Low-energy index of the time-integrated spectra ({what})')
  fig.tight_layout()
  # the Band-alpha twin must be built AFTER tight_layout, or it takes the pre-layout limits
  ax2 = ax.twinx()
  ax2.set_ylim(*(np.array(ax.get_ylim()) - 2.))
  ax2.set_ylabel('Band $\\alpha = a - 2$')
  fig.savefig(os.path.join(outdir, fname or f'fluence_slopes_{kind}_vs_regime.png'),
              dpi=300, bbox_inches='tight')
  plt.close(fig)


def main(key=DEFAULT_KEY, log10ratio_arr=LOG10RATIO_ARR, outdir=OUTDIR,
    use_cache=True, nproc=None, methods=METHODS, labels=LABELS, norm_side='A'):
  '''
  Ensure both sweeps exist (computing whichever is missing), then build every
  comparison figure. With the caches warm this is plotting-only.
  methods: (side A, side B = the REFERENCE) sweep methods, defaulting to the data-only
  METHODS pair -- the 'fit' method is not plotted any more, so the default path never
  computes a fit sweep. Any pair method_outdir knows still works; pass `labels` to match.
  norm_side: which side's flux anchors the panels. 'A' here, not the 'B' the plotting
  functions default to, because the default `methods` pair is cut-vs-full: the cut is the
  prescription being tested, so anchoring on it makes the full run read as the extra
  emission the cut leaves out (see LABELS). Pass 'B' for a run-vs-run comparison.
  '''
  os.makedirs(outdir, exist_ok=True)
  for m in methods:
    d = method_outdir(m, key)
    os.makedirs(d, exist_ok=True)
    if not (use_cache and load_sweep(d)):
      print(f'--- running the {m} sweep ---')
      # skip_cached MUST be forwarded (see sweep_gammacm.main): run_sweep defaults it
      # to True, so use_cache=False would otherwise recompute nothing.
      run_sweep(key, log10ratio_arr, method=m, nproc=nproc,
                skip_cached=use_cache)
  pairs = load_pairs(*(method_outdir(m, key) for m in methods))

  barT_f = exit_onset_barT(key, z=Z_SHELL)
  barT_off = rarefaction_off_barT(key, z=Z_SHELL)
  print(f'crossing bar_T_f = {barT_f:.4f}' +
        (f', rarefaction cut-off {barT_off[0]:.4f}..{barT_off[1]:.4f}' if barT_off else ''))

  plot_efficiency_compare(pairs, outdir=outdir, labels=labels)
  for kind in ('peak', 'fluence'):
    for mode in SPEC_MODES:
      plot_spectra_compare(pairs, kind=kind, mode=mode, outdir=outdir, labels=labels,
                           norm_side=norm_side)
  # the lin-lin variant ONLY (see plot_lightcurve_compare's `scale`): the log and linlog
  # views of the same three curves were three more files per frequency to keep straight,
  # and the ratio panel already carries the late divergence they were kept for
  plot_lightcurve_compare(pairs, barT_f, barT_off=barT_off, outdir=outdir, labels=labels,
                          norm_side=norm_side, scale='lin')
  s = plot_summary_ratios(pairs, outdir=outdir, labels=labels)

  trim_pngs(outdir)
  print(f'Comparison figures saved to {outdir}')
  return pairs, s


if __name__ == '__main__':
  main()
