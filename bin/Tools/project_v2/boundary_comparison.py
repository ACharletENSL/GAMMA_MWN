# -*- coding: utf-8 -*-
# @Author: acharlet

'''
Semi-infinite shells: what the rarefaction wave actually costs.

Everything we know about the rarefaction's effect on the emission comes from a
POST-PROCESSING prescription -- rar_cut on/off in working_cooling_data, with the cut
radius supplied by the kinematic model compute_shell_rarefaction_head, which has no
ambient term at all and therefore assumes vacuum breakout. This module compares that
against a simulation in which the rarefaction genuinely does not exist.

The trick is GAMMA's ghost cells. Grid::updateGhosts (1d.cpp:232-243) copies the whole
outermost active cell into the ghosts, so the boundary imposes dp/dr = 0, not p -> 0.
Removing the external-medium buffer (Next = 0) puts those outflow ghosts directly on the
shell edges: when the shock reaches the boundary it exits, the shocked layer is held at
its edge pressure instead of being released, and no rarefaction is launched. The run is
therefore the shocked layer we already build, evolving adiabatically with no release --
the closest simulated analogue of a model that ignores rarefactions.

Note this is the OPPOSITE lever from lowering rhoContr: the buffer is what supplies
back-pressure against the release wave, so thinning it makes the rarefaction stronger.
It is already effectively vacuum anyway (p_ext/p_shocked ~ 2.5e-5).

Two runs, identical but for Next:
  REF_KEY   Next=20, the fiducial cooling_g100
  SEMI_KEY  Next=0,  same physics, no rarefaction

Contents:
  causality_gate    - the correctness test. The two runs are causally identical until a
                      boundary signal reaches a cell, so |dln p| must sit at the t=0
                      edge-erosion floor and only then rise. Early divergence is a bug.
  arrival_radii     - the divergence radius per cell IS the rarefaction arrival, measured
                      with no fit and no threshold on a single run's own profile.
                      Compared against the cached model map rarefaction_head_{z}.npz.
  crash_detector    - cross-check: truncate_at_rarefaction / sim_rarefaction_radius must
                      fire in the reference run and never in the semi run.

Run `python boundary_comparison.py` for all three, figures in
GAMMA/bin/Tools/figures/boundary_comparison.
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from environment import MyEnv
from phys_constants import c_
from IO import get_variable, GAMMA_dir, open_celldata, openData, dataList
from working_cooling import (check_extracted_cells, truncate_at_rarefaction,
    load_shell_rarefaction)
from rarefaction_validation import shocked_after_crossing, sim_rarefaction_radius

REF_KEY = 'cooling_g100'
SEMI_KEY = 'cooling_g100_semi'
OUTDIR = os.path.join(GAMMA_dir, 'bin', 'Tools', 'figures', 'boundary_comparison')

# divergence threshold on |dln p| between the two runs. Chosen well above the numerical
# floor but far below the rarefaction's own crash (which takes p down by decades); the
# measured floor is reported by causality_gate so this can be re-tuned against it.
DLNP_THRESH = 0.02
# the causality floor is measured over R < QUIET_FRAC * R_div, i.e. strictly inside the
# region no boundary signal has reached yet. 0.8 leaves a margin for the finite width of
# the divergence onset (it is a ramp, not a step).
QUIET_FRAC = 0.8
# cells with R_rar/R_inj below this are the head's launch cells (shell edge): no
# pre-arrival region exists, so the causality floor is undefined there rather than failed.
LAUNCH_CELL_ROR = 1.05
# onset_ladder: |ratio - 1| above this means the crash detector did not resolve that
# cell's onset at all (it fired on later steepening), rather than that the onset moved.
# Far above any plausible physical shift, which is a sub-percent effect.
UNRESOLVED_DEV = 0.5
# crash_radius_uniform: the common log-radius grid. step sets the interpolation grid,
# window the differentiation baseline (in dex), min_dex the smooth-decay lead-in required
# before a crash may be reported. window_dex ~ 0.02 is close to what 15 rows spanned in
# the fiducial around the onset, so the detector keeps its established sensitivity.
CRASH_STEP_DEX = 0.0002
CRASH_WINDOW_DEX = 0.02
CRASH_MIN_DEX = 0.01
# step_dex trades measurement RESOLUTION against a residual cadence sensitivity, and the
# ladder differences are small enough that both matter:
#   0.002  -> exactly cadence-independent (0.0000 spread under 6x subsampling) but the
#            onset is quantised in steps of 0.46%, i.e. AT the size of the effect. The
#            first run at this step returned ratios that were all multiples of 0.0046 --
#            a resolution floor masquerading as a measurement.
#   0.0002 -> resolution 0.046%, residual cadence spread 0.046%. Both an order of
#            magnitude below the ~1% edge-treatment differences being resolved.
# Quote no onset difference smaller than ~0.05% from this detector.


# ---------------------------------------------------------------------------
# cell pairing
# ---------------------------------------------------------------------------
def shell_offsets(env, z):
  '''(first cell index, n cells) of shell z on this run's grid.

  The grid is [Next | Nsh4 | Nsh1 | Next], so the SAME physical cell carries different
  indices in the two runs (offset by Next). Everything here is keyed on the position j
  within the shell, never on the raw index.
  '''
  k4 = int(env.Next)
  kCD = k4 + int(env.Nsh4)
  return (k4, int(env.Nsh4)) if z == 4 else (kCD, int(env.Nsh1))


def paired_cells(z, ref_key=REF_KEY, semi_key=SEMI_KEY):
  '''[(j, k_ref, k_semi)] for the cells of shell z extracted in BOTH runs.

  j is the position within the shell: j=0 is the outer edge of shell 4 / the CD-side
  edge of shell 1, following the grid layout. Pairing on j is what makes the two runs
  comparable at all, since Next differs.
  '''
  env_r, env_s = MyEnv(ref_key), MyEnv(semi_key)
  k4_r, n_r = shell_offsets(env_r, z)
  k4_s, n_s = shell_offsets(env_s, z)
  if n_r != n_s:
    raise ValueError(f'shell {z} has {n_r} cells in {ref_key} but {n_s} in {semi_key}; '
                     'the runs are not comparable cell-by-cell')
  have_r = set(check_extracted_cells(ref_key).tolist())
  have_s = set(check_extracted_cells(semi_key).tolist())
  return [(j, k4_r + j, k4_s + j) for j in range(n_r)
          if (k4_r + j) in have_r and (k4_s + j) in have_s]


def final_time(key):
  '''Lab time of a run's last snapshot.'''
  its = dataList(key)
  return float(openData(key, its[-1]).t.iloc[0])


def it_at_time(key, t_target, verbose=False):
  '''First iteration of `key` whose lab time reaches t_target, or None if it never does.'''
  for it in dataList(key):
    if float(openData(key, it).t.iloc[0]) >= t_target:
      if verbose:
        print(f'  {key}: t>={t_target:.6e} first at it={it}')
      return it
  return None


def matched_window(key_a=SEMI_KEY, key_b=REF_KEY, verbose=True):
  '''
  Truncate whichever run is LONGER so both cover the same physical window.

  Equal iteration count is not equal physics. The timestep is CFL-limited by the
  steepest feature in the box, and the rarefaction is what lets the shell spread: with
  it, cells stretch and dt grows (fiducial dt/dit reaches ~62); with the edges confined
  by outflow boundaries, cells stay compact and dt stagnates (semi peaks at 9.5, then
  FALLS to 2.6). So the two runs cover very different spans at the same itmax, and the
  data method follows every cell to its last snapshot -- any surplus lands straight in
  the integrated emission and reads as a boundary effect.

  Which run is longer is NOT fixed: the semi run gains early (it outran the fiducial
  2.5x by it=100k) and loses late (it stopped at t=4.14e6 against the fiducial's
  2.53e7, a factor 6.1 short, when ITMAX_ fired before TSTOP_). Hence both directions.

  Returns (t_common, itmax_a, itmax_b), each itmax being the value to pass to
  extract_data_cells for that run (None = no truncation needed for that side).
  '''
  ta, tb = final_time(key_a), final_time(key_b)
  t_common = min(ta, tb)
  ita = it_at_time(key_a, t_common) if ta > t_common else None
  itb = it_at_time(key_b, t_common) if tb > t_common else None
  if verbose:
    longer = key_a if ta > tb else key_b
    print(f'{key_a} ends at t={ta:.6e}; {key_b} ends at t={tb:.6e}  '
          f'(ratio {max(ta,tb)/min(ta,tb):.2f})')
    print(f'common window t <= {t_common:.6e}; truncating the longer run ({longer}): '
          f'itmax_a={ita}, itmax_b={itb}')
  return t_common, ita, itb


def _post_shock(key, k):
  '''Post-shock history of cell k, same selection as fit_celldata.'''
  data = open_celldata(key, k)
  if data is False:
    return None
  sh = shocked_after_crossing(data)
  return sh if len(sh) else None


def dlnp_profile(k_ref, k_semi, ref_key=REF_KEY, semi_key=SEMI_KEY, npts=400):
  '''dln p between the two runs for one paired cell, on a shared R/R_inj grid.

  Both histories are interpolated in log R onto the overlap of their radial ranges,
  normalised by each run's own R_inj (that cell's radius when it was shocked). Returns
  (ror, dlnp) with ror = R/R_inj, or (None, None) if the cell is unusable.
  '''
  sh_r, sh_s = _post_shock(ref_key, k_ref), _post_shock(semi_key, k_semi)
  if sh_r is None or sh_s is None:
    return None, None
  x_r, x_s = sh_r.x.to_numpy(), sh_s.x.to_numpy()
  p_r, p_s = sh_r.p.to_numpy(), sh_s.p.to_numpy()
  ok_r, ok_s = (p_r > 0.), (p_s > 0.)
  if ok_r.sum() < 4 or ok_s.sum() < 4:
    return None, None
  x_r, p_r, x_s, p_s = x_r[ok_r], p_r[ok_r], x_s[ok_s], p_s[ok_s]
  # each run's own injection radius: the cell's radius on its first post-shock row
  r_r, r_s = x_r/x_r[0], x_s/x_s[0]
  lo, hi = max(r_r[0], r_s[0]), min(r_r[-1], r_s[-1])
  if not (hi > lo):
    return None, None
  ror = np.logspace(np.log10(lo), np.log10(hi), npts)
  ln_r = np.interp(np.log(ror), np.log(r_r), np.log(p_r))
  ln_s = np.interp(np.log(ror), np.log(r_s), np.log(p_s))
  return ror, ln_s - ln_r


def divergence_radius(ror, dlnp, thresh=DLNP_THRESH, floor_frac=0.05):
  '''First R/R_inj where the two runs part company by more than thresh in ln p.

  floor_frac guards the very start of the history, where post-shock settling differs by
  a step or two between runs at different dump cadences: the first floor_frac of the
  log-radial range is excluded from the search. Returns (ror_div, seen).
  '''
  if ror is None:
    return np.nan, False
  lnr = np.log(ror)
  start = np.searchsorted(lnr, lnr[0] + floor_frac*(lnr[-1] - lnr[0]))
  over = np.flatnonzero(np.abs(dlnp[start:]) > thresh)
  if len(over):
    return ror[start + over[0]], True
  return np.nan, False


# ---------------------------------------------------------------------------
# 1. causality gate
# ---------------------------------------------------------------------------
def causality_gate(z=4, ncells=40, plot=True, outdir=OUTDIR):
  '''
  The correctness test. A cell cannot know about the boundary until a signal from it
  arrives, so |dln p| between the two runs must sit at the numerical + t=0-edge-erosion
  floor out to the model arrival radius, then rise. Divergence EARLIER than the model
  head is either a bug or evidence the model head is late.

  Reports, per cell, the pre-arrival floor (max |dln p| inside the model R_rar) and the
  divergence radius, and prints the worst offenders.
  '''
  env = MyEnv(REF_KEY)
  pairs = paired_cells(z)
  if not pairs:
    raise RuntimeError(f'no cells of shell {z} extracted in both runs')
  sel = [pairs[i] for i in np.unique(np.linspace(0, len(pairs)-1, ncells).astype(int))]
  rar_map = load_shell_rarefaction(REF_KEY, z, env, n_shell=len(pairs))

  rows = []
  curves = []
  for j, k_ref, k_semi in sel:
    ror, dlnp = dlnp_profile(k_ref, k_semi)
    if ror is None:
      continue
    r_model = rar_map.get(k_ref, np.nan)
    r_div, seen = divergence_radius(ror, dlnp)
    # The floor must be measured STRICTLY BEFORE the wave arrives, and the arrival to
    # use is the MEASURED one (r_div), not the model's. Bounding by r_model instead
    # sweeps up the real post-arrival divergence wherever r_div < r_model -- which is
    # exactly what happens at the CD end of the RS (r_div/r_model ~ 0.81) -- and
    # re-reports the signal as if it were noise.
    # Cells whose modelled arrival IS their injection radius are the head's launch cells
    # (the shell edge, shocked last, r_model = 1 by construction). They are born into the
    # rarefaction, so no causally-quiet interval exists and the gate is undefined -- not
    # failed. The same degeneracy censors sim_rarefaction_radius there.
    floor = np.nan
    if seen and np.isfinite(r_model) and r_model > LAUNCH_CELL_ROR:
      quiet = ror < QUIET_FRAC*r_div
      if quiet.sum() > 2:
        floor = np.abs(dlnp[quiet]).max()
    rows.append(dict(j=j, k_ref=k_ref, k_semi=k_semi, r_model=r_model,
                     r_div=r_div, seen=seen, floor=floor,
                     ratio=r_div/r_model if np.isfinite(r_model) else np.nan))
    curves.append((j, ror, dlnp, r_model))

  df = pd.DataFrame(rows)
  fin = df[np.isfinite(df.floor)]
  print(f'\n--- causality gate, shell {z} ({len(df)} cells) ---')
  if len(fin):
    print(f'quiet-zone |dln p| floor (R < {QUIET_FRAC}*R_div): '
          f'median {fin.floor.median():.2e}, '
          f'90th pct {fin.floor.quantile(0.9):.2e}, max {fin.floor.max():.2e}')
    bad = fin[fin.floor > DLNP_THRESH]
    if len(bad):
      print(f'  !! {len(bad)}/{len(fin)} cells already differ before their own '
            f'divergence (threshold {DLNP_THRESH}). Worst:')
      print(bad.nlargest(5, 'floor')[['j', 'k_ref', 'floor', 'r_div', 'r_model']]
            .to_string(index=False))
    else:
      print(f'  all {len(fin)} cells stay below {DLNP_THRESH} until the model head: gate PASSED')
  seen = df[df.seen]
  if len(seen):
    print(f'divergence radius / model R_rar: median {seen.ratio.median():.3f}, '
          f'16-84% [{seen.ratio.quantile(0.16):.3f}, {seen.ratio.quantile(0.84):.3f}]')
  print(f'{(~df.seen).sum()}/{len(df)} cells never diverge (censored)')

  if plot:
    os.makedirs(outdir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = plt.get_cmap('viridis')
    n = max(len(curves)-1, 1)
    for i, (j, ror, dlnp, r_model) in enumerate(curves):
      col = cmap(i/n)
      ax.plot(ror, np.abs(dlnp), color=col, lw=0.8, alpha=0.8)
      if np.isfinite(r_model):
        ax.axvline(r_model, color=col, lw=0.5, ls=':', alpha=0.4)
    ax.axhline(DLNP_THRESH, color='k', ls='--', lw=1, label=f'threshold {DLNP_THRESH}')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('$R/R_{\\rm inj}$')
    ax.set_ylabel(r'$|\Delta \ln p|$  (semi $-$ ref)')
    ax.set_title(f'Causality gate, shell {z}\n'
                 'dotted: model $R_{\\rm rar}$ of the same cell (colour-matched)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f'causality_gate_z{z}.png'), dpi=150)
    plt.close(fig)
  return df


# ---------------------------------------------------------------------------
# 2. measured arrival radius vs the model map
# ---------------------------------------------------------------------------
def arrival_radii(z=4, ncells=80, plot=True, outdir=OUTDIR):
  '''
  Where the decompression arrives, measured three ways.

  IMPORTANT -- the divergence radius here is NOT the independent measurement it was
  designed to be, and must not be quoted as one. The premise was that the semi run has
  no rarefaction, so the runs would part company when the fiducial's wave arrived. That
  premise is false: BOTH runs decompress at the same radius (measured cell by cell,
  semi/fid crash ratio = 1.00 median on both shells). The onset is set by the shock
  running out of shell to sweep, not by what lies outside the edge -- which is why the
  R_rar model, carrying no ambient term at all, predicts it to ~1%.

  What r_div therefore detects is the ~0.3% MISALIGNMENT of two nearly identical
  crashes: inside a near-vertical pressure drop, a small radial offset gives a large
  |dln p|. Hence its 1% agreement with the crash detector -- same quantity, not
  confirmation. Kept because the three-way agreement (model / crash / divergence) is
  still a useful consistency check on the model.

  The boundary's actual effect is on the DEPTH of the decompression, not its timing:
  see pressure_retention.
  '''
  env = MyEnv(REF_KEY)
  pairs = paired_cells(z)
  sel = [pairs[i] for i in np.unique(np.linspace(0, len(pairs)-1, ncells).astype(int))]
  rar_map = load_shell_rarefaction(REF_KEY, z, env, n_shell=len(pairs))

  rows = []
  for j, k_ref, k_semi in sel:
    ror, dlnp = dlnp_profile(k_ref, k_semi)
    r_div, seen = divergence_radius(ror, dlnp)
    sh_r = _post_shock(REF_KEY, k_ref)
    r_crash = np.nan
    if sh_r is not None:
      R_cm, crash_seen = sim_rarefaction_radius(sh_r)
      if crash_seen:
        r_crash = R_cm/(sh_r.x.to_numpy()[0]*c_)
    rows.append(dict(j=j, k_ref=k_ref, r_model=rar_map.get(k_ref, np.nan),
                     r_div=r_div, seen=seen, r_crash=r_crash))
  df = pd.DataFrame(rows)

  ok = df[df.seen & np.isfinite(df.r_model)]
  print(f'\n--- arrival radius, shell {z} ({len(ok)}/{len(df)} cells measured) ---')
  if len(ok):
    ratio = ok.r_div/ok.r_model
    print(f'measured / model : median {ratio.median():.3f}, '
          f'16-84% [{ratio.quantile(0.16):.3f}, {ratio.quantile(0.84):.3f}]')
    both = ok[np.isfinite(ok.r_crash)]
    if len(both):
      rc = both.r_div/both.r_crash
      print(f'measured / crash-detector : median {rc.median():.3f}, '
            f'16-84% [{rc.quantile(0.16):.3f}, {rc.quantile(0.84):.3f}]  '
            f'({len(both)} cells)')

  if plot and len(df):
    os.makedirs(outdir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df.j, df.r_model, 'k-', lw=1.5, label='model (rarefaction_head)')
    m = df.seen
    ax.plot(df.j[m], df.r_div[m], 'o', ms=4, color='tab:red',
            label='measured (ref vs semi divergence)')
    mc = np.isfinite(df.r_crash)
    ax.plot(df.j[mc], df.r_crash[mc], 's', ms=3, mfc='none', color='tab:blue',
            label=r'crash detector ($d\ln p/d\ln R < -8$)')
    # the two shells run in OPPOSITE directions: the CD is at shell 4's top end and at
    # shell 1's bottom end, so j=0 is the outer edge for z=4 but the CD-adjacent cell
    # for z=1 (as shell_offsets and sweep_rarcut.shell_cell_range document). Label it
    # explicitly -- read the wrong way round, the R_rar profile looks inverted.
    ax.set_xlabel(f'cell position in shell {z}  '
                  + ('(0 = outer edge -> CD)' if z == 4 else '(0 = CD -> outer edge)'))
    ax.set_ylabel('$R_{\\rm rar}/R_{\\rm inj}$')
    ax.set_title(f'Rarefaction arrival, shell {z}')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f'arrival_radii_z{z}.png'), dpi=150)
    plt.close(fig)
  return df


# ---------------------------------------------------------------------------
# 2b. pressure retention -- what the boundary ACTUALLY changes
# ---------------------------------------------------------------------------
def pressure_retention(z=4, ncells=13, targets=(1.5, 3., 10., 30.), plot=True,
    outdir=OUTDIR):
  '''
  How much pressure the confined layer keeps that the released one loses.

  Both runs decompress at the same radius (see arrival_radii), so the boundary does not
  set the TIMING of the wave. It sets its DEPTH: with a low-pressure edge the shocked
  layer drains, with dp/dr = 0 it cannot. Measured as p_semi/p_fid for the same physical
  cell at fixed R/R_inj -- 1.000 means the two runs are still identical there.

  The radial structure is the real causality signature: at R/R_inj = 1.5 the ratio is
  1.000 for mid-shell cells and > 1 only near the edges, i.e. the difference propagates
  INWARD from the boundaries, as it must.
  '''
  pairs = paired_cells(z)
  sel = [pairs[i] for i in np.unique(np.linspace(0, len(pairs)-1, ncells).astype(int))]
  rows = []
  for j, k_ref, k_semi in sel:
    ror, dlnp = dlnp_profile(k_ref, k_semi)
    if ror is None:
      continue
    r = dict(j=j, k_ref=k_ref)
    for t in targets:
      r[f'r{t:g}'] = (np.exp(np.interp(t, ror, dlnp))
                      if ror[0] <= t <= ror[-1] else np.nan)
    rows.append(r)
  df = pd.DataFrame(rows)
  print(f'\n--- pressure retention (p_semi/p_fid), shell {z} ---')
  print(df.to_string(index=False, float_format=lambda v: f'{v:.3f}'))

  if plot and len(df):
    os.makedirs(outdir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    for t in targets:
      ax.plot(df.j, df[f'r{t:g}'], 'o-', ms=3, lw=1, label=f'$R/R_{{\\rm inj}}={t:g}$')
    ax.axhline(1., color='k', lw=0.8, ls='--')
    ax.set_yscale('log')
    ax.set_xlabel(f'cell position in shell {z}  '
                  + ('(0 = outer edge -> CD)' if z == 4 else '(0 = CD -> outer edge)'))
    ax.set_ylabel('$p_{\\rm semi}/p_{\\rm fid}$')
    ax.set_title(f'Pressure retained by the confined layer, shell {z}')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f'pressure_retention_z{z}.png'), dpi=150)
    plt.close(fig)
  return df


# ---------------------------------------------------------------------------
# 2c. onset ladder -- is the decompression onset independent of the edge?
# ---------------------------------------------------------------------------
# Four edge treatments of the SAME collision, spanning a rigid dp/dr = 0 clamp and four
# decades of ambient density. If R_crash/R_inj agrees across all of them, the onset is a
# property of the shell's own geometry -- the shock running out of shell to sweep -- and
# not of anything at the boundary.
LADDER = (
  ('cooling_g100_semi',       'Next=0, dp/dr=0 clamp'),
  ('cooling_g100',            r'coupled, $\rho_{\rm ext}/\rho_{\rm sh}=5\times10^{-2}$'),
  ('cooling_g100_c4_coupled', r'coupled, $5\times10^{-4}$ (short)'),
  ('cooling_g100_p4',         r'p-matched, $5\times10^{-4}$'),
  ('cooling_g100_p6',         r'p-matched, $5\times10^{-6}$'),
)


def crash_radius_uniform(sh, step_dex=CRASH_STEP_DEX, window_dex=CRASH_WINDOW_DEX,
    slope_thresh=-8., min_dex=CRASH_MIN_DEX):
  '''
  R_crash/R_inj measured on a COMMON log-radius grid -- cadence-independent.

  sim_rarefaction_radius takes its log-slope over a fixed number of ROWS (window=15), so
  the radial baseline it differentiates over depends on how densely that run was dumped.
  Measured on cooling_g100_p4, subsampling the history 1/2/3/6x moves R_crash by 0.4% at
  mid-shell and 3.4% at the CD -- the same order as, or larger than, the inter-run
  differences this ladder is trying to resolve. The runs here have very different
  cadences (p4 ~3.2k dumps to t=1.7e5, p6 ~20k to t=1.1e5), so comparing their raw
  detections would confuse dump cadence with edge physics.

  Here ln p is interpolated onto a uniform log10(R/R_inj) grid of fixed step, and the
  slope is taken over a fixed window in DEX. Every run is then differentiated over the
  same radial baseline no matter how it was dumped. Returns (ror, seen).
  '''
  x = sh.x.to_numpy(float); p = sh.p.to_numpy(float)
  ok = np.isfinite(x) & np.isfinite(p) & (p > 0.) & (x > 0.)
  if ok.sum() < 8:
    return np.nan, False
  x, p = x[ok], p[ok]
  lr = np.log10(x/x[0]); lp = np.log10(p)
  if lr[-1] - lr[0] < min_dex:
    return np.nan, False
  grid = np.arange(lr[0], lr[-1], step_dex)
  if len(grid) < 4:
    return np.nan, False
  lpg = np.interp(grid, lr, lp)
  w = max(int(round(window_dex/step_dex)), 1)
  if len(grid) <= w:
    return np.nan, False
  slope = (lpg[w:] - lpg[:-w])/(grid[w:] - grid[:-w])
  steep = np.flatnonzero(slope < slope_thresh)
  # require a little smooth decay first, the dex analogue of sim_rarefaction_radius's
  # minpts guard: without it a cell born into the wave reports its first grid point
  lead = np.searchsorted(grid, grid[0] + min_dex)
  steep = steep[steep >= lead]
  if len(steep):
    return float(10**grid[steep[0]]), True
  return np.nan, False


def onset_ladder(keys=None, z=4, ncells=60, plot=True, outdir=OUTDIR, ref=REF_KEY,
    uniform=True, step_dex=None):
  '''
  R_crash/R_inj per cell for every available edge treatment, on one axis.

  Cells are matched by POSITION WITHIN THE SHELL, never by raw index: cooling_g100_semi
  has Next=0 and every other run Next=20, so the same physical cell carries indices 20
  apart (shell_offsets). Getting this wrong silently compares different cells.

  Keys that are not on disk yet are skipped with a note, so this is runnable while the
  ladder is still being filled in.

  Reports, per cell, the spread across keys relative to `ref`. The claim under test is
  invariance at the ~1% level already measured between cooling_g100 and _semi -- not
  exact equality, since sim_rarefaction_radius carries its own threshold and the dump
  cadence differs between runs.
  '''
  keys = list(keys) if keys is not None else [k for k, _ in LADDER]
  labels = dict(LADDER)
  have = []
  for k in keys:
    if len(check_extracted_cells(k)):
      have.append(k)
    else:
      print(f'  (skipping {k}: no extracted cells yet)')
  if ref not in have:
    raise RuntimeError(f'reference {ref} has no extracted cells')

  # positions common to every available run
  pos = None
  offs = {}
  for k in have:
    k0, n = shell_offsets(MyEnv(k), z)
    offs[k] = k0
    ks = set(check_extracted_cells(k).tolist())
    p = {j for j in range(n) if (k0 + j) in ks}
    pos = p if pos is None else (pos & p)
  pos = sorted(pos)
  sel = [pos[i] for i in np.unique(np.linspace(0, len(pos)-1, ncells).astype(int))]

  # Launch cells (model R_rar/R_inj ~ 1, the shell edge) are excluded from the statistics:
  # they are born into the rarefaction, so there is no smooth-decay-then-crash for
  # sim_rarefaction_radius to find. When it fires there it locks onto unrelated later
  # steepening -- that is the source of the 30x outlier on the FS outer edge, and of the
  # 2.36 one on the RS, neither of which says anything about the onset. Same degeneracy
  # the causality gate handles with LAUNCH_CELL_ROR.
  env_ref = MyEnv(ref)
  rar_map = load_shell_rarefaction(ref, z, env_ref, n_shell=shell_offsets(env_ref, z)[1])
  k0_ref = offs[ref]

  rows = []
  for j in sel:
    r = dict(j=j, r_model=rar_map.get(k0_ref + j, np.nan))
    r['launch'] = not (r['r_model'] > LAUNCH_CELL_ROR)
    for k in have:
      sh = _post_shock(k, offs[k] + j)
      if sh is None:
        r[k] = np.nan
        continue
      if uniform:
        ror, seen = crash_radius_uniform(
            sh, **({} if step_dex is None else dict(step_dex=step_dex)))
        # ABSOLUTE crash radius, not R/R_inj. R_inj is that run's first post-shock
        # snapshot, so it moves with the dump cadence: under a modulus thinning of p4 the
        # ratio shifted by up to 1.46% while the absolute radius moved 0.02%. Every run
        # here has its own cadence, so the ratio carries a per-run systematic as large as
        # the edge effect being measured. The absolute radius has no such normalisation,
        # and is directly comparable because all runs share the same initial geometry.
        r[k] = ror*sh.x.to_numpy()[0]*c_ if seen else np.nan
        r[f'{k}__ror'] = ror if seen else np.nan
      else:
        R, seen = sim_rarefaction_radius(sh)
        r[k] = R/(sh.x.to_numpy()[0]*c_) if seen else np.nan
    rows.append(r)
  df = pd.DataFrame(rows)
  n_launch = int(df.launch.sum())
  if n_launch:
    print(f'  ({n_launch} launch cells excluded from statistics: R_rar/R_inj <= '
          f'{LAUNCH_CELL_ROR}, no resolvable onset)')
  stat = df[~df.launch]

  print(f'\n--- onset ladder, shell {z} ({len(have)} runs, {len(df)} cells) ---')
  base = stat[ref].to_numpy(float)
  for k in have:
    v = stat[k].to_numpy(float)
    g = np.isfinite(v) & np.isfinite(base)
    if k == ref:
      print(f'  {k:22s}: reference, {np.isfinite(v).sum()}/{len(v)} cells resolve a crash')
      continue
    ratio = v[g]/base[g]
    # Cells where the detector never resolves the onset and instead locks onto later
    # steepening. Reported, NOT dropped: excluding points because they disagree with the
    # model would be circular. Expected where confinement makes the pressure drop
    # shallowest -- near the CD, where p_semi/p_fid is largest (pressure_retention) and
    # the drop can fail to cross the dln p/dln R < -8 threshold at all.
    unres = np.abs(ratio - 1.) > UNRESOLVED_DEV
    core = ratio[~unres]
    if not len(core):
      # a short run can resolve no cell at all in a given shell (its history ends before
      # those cells decompress). Say so rather than dying on an empty quantile.
      print(f'  {k:22s}: 0/{len(v)} cells resolve in this shell -- run too short here')
      continue
    print(f'  {k:22s}: {np.isfinite(v).sum()}/{len(v)} resolve; ratio to ref '
          f'median {np.median(core):.4f}, 16-84% [{np.quantile(core,.16):.4f}, '
          f'{np.quantile(core,.84):.4f}], max |dev| {np.max(np.abs(core-1)):.4f}')
    if unres.any():
      js = stat.j.to_numpy()[g][unres]
      print(f'  {"":22s}  + {unres.sum()} cell(s) with NO resolvable onset '
            f'(j={list(js)}, ratio up to {np.max(ratio[unres]):.1f}) -- detector limit, '
            'not an onset shift; see the docstring')
  # spread across all keys, per cell
  M = stat[have].to_numpy(float)
  # keep only cells that resolve a crash in EVERY run: a spread taken over a partly
  # censored row compares different subsets of the ladder cell to cell. (Rows censored
  # everywhere would also make nanmax/nanmin warn on an all-NaN slice.)
  full = M[np.isfinite(M).all(axis=1)]
  spread = (full.max(axis=1)/full.min(axis=1) - 1.) if len(full) else np.array([])
  fin = spread[np.isfinite(spread)]
  if len(fin):
    print(f'  per-cell spread across all {len(have)} runs '
          f'({len(fin)}/{len(df)} cells resolve in every run): median {np.median(fin):.4f}, '
          f'90th pct {np.quantile(fin,.9):.4f}, max {np.max(fin):.4f}')

  if plot and len(df):
    os.makedirs(outdir, exist_ok=True)
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True,
                                  gridspec_kw=dict(height_ratios=[2, 1]))
    env = MyEnv(ref)
    rar_map = load_shell_rarefaction(ref, z, env, n_shell=shell_offsets(env, z)[1])
    k0 = offs[ref]
    # the model map is in R_rar/R_inj, so put it on the absolute axis using the
    # reference run's own R_inj for each cell
    model_abs = []
    for j in df.j:
      ror_ref = df.loc[df.j == j, f'{ref}__ror'].to_numpy(float)
      Rc = df.loc[df.j == j, ref].to_numpy(float)
      m = rar_map.get(k0+j, np.nan)
      model_abs.append(m*Rc[0]/ror_ref[0] if (len(Rc) and np.isfinite(ror_ref[0])
                                              and ror_ref[0]) else np.nan)
    ax.plot(df.j, model_abs, 'k-', lw=1.5, label='model (rarefaction_head)', zorder=1)
    for k in have:
      ax.plot(df.j, df[k], 'o', ms=3, alpha=0.8, label=labels.get(k, k))
      if k != ref:
        # base is the launch-cell-filtered array used for the statistics; the panel
        # plots every sampled cell, so take the reference over the full frame here
        ax2.plot(df.j, df[k].to_numpy(float)/df[ref].to_numpy(float), 'o', ms=3, alpha=0.8)
    ax2.axhline(1., color='k', lw=0.8, ls='--')
    ax.set_ylabel('$R_{\\rm crash}$  [cm]')
    ax2.set_ylabel(f'ratio to {ref}')
    ax2.set_xlabel(f'cell position in shell {z}  '
                   + ('(0 = outer edge -> CD)' if z == 4 else '(0 = CD -> outer edge)'))
    ax.set_title(f'Decompression onset vs edge treatment, shell {z}')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f'onset_ladder_z{z}.png'), dpi=150)
    plt.close(fig)
  return df


# ---------------------------------------------------------------------------
# 3. crash-detector cross-check
# ---------------------------------------------------------------------------
def crash_detector(z=4, ncells=60):
  '''
  Designed as: truncate_at_rarefaction must fire in the reference and never in the semi
  run. MEASURED: it fires in the semi run almost as often (35/40 RS, 38/40 FS) and at
  the SAME radius (crash ratio 1.00 median). That is the result, not a bug -- the
  decompression is launched by the shock reaching the end of the shell, so removing the
  ambient and clamping dp/dr = 0 at the edge does not remove it. What the boundary
  changes is how deep the pressure falls afterwards (pressure_retention).

  The count is kept as a regression check: if a future setup ever DOES suppress the
  crash, this is where it shows up.
  '''
  pairs = paired_cells(z)
  sel = [pairs[i] for i in np.unique(np.linspace(0, len(pairs)-1, ncells).astype(int))]
  n_ref = n_semi = n = 0
  for j, k_ref, k_semi in sel:
    sh_r, sh_s = _post_shock(REF_KEY, k_ref), _post_shock(SEMI_KEY, k_semi)
    if sh_r is None or sh_s is None:
      continue
    n += 1
    n_ref += len(truncate_at_rarefaction(sh_r)) < len(sh_r)
    n_semi += len(truncate_at_rarefaction(sh_s)) < len(sh_s)
  print(f'\n--- crash detector, shell {z} ({n} cells) ---')
  print(f'  fires in {REF_KEY:>20s}: {n_ref}/{n}  (expected: most)')
  print(f'  fires in {SEMI_KEY:>20s}: {n_semi}/{n}  (expected: 0)')
  return n_ref, n_semi, n


# ---------------------------------------------------------------------------
# 4. emission: what the rarefaction actually costs
# ---------------------------------------------------------------------------
# Both sides use the data method with NO rarefaction machinery (rar_cut=None): in the
# reference the wave is in the simulation, in the semi run it does not exist. So the
# difference is the rarefaction itself, not a prescription for it. Do NOT ask for
# rar_cut='model' on the semi key -- load_shell_rarefaction would build a head from a
# breakout that produces no release.
EMISSION_METHOD = 'data'
EMISSION_LABELS = ('no rf', 'full')      # A = semi (no rarefaction), B = reference
EMISSION_OUTDIR = os.path.join(GAMMA_dir, 'bin', 'Tools', 'figures', 'boundary_compare')

# env attributes that must agree for the comparison to be pointwise (no interpolation):
# the two runs differ only in Next, which is an index offset and must not touch physics.
_ENV_PHYS = ('R0', 'D01', 'D04', 'rho1', 'rho4', 'u1', 'u4', 'p1', 'p4',
             'Theta0', 'eps_e', 'eps_B', 'xi_e', 'psyn', 'nu0', 'T0', 'Ts', 'lfac')


def check_env_match(key_a=SEMI_KEY, key_b=REF_KEY, rtol=1e-10, verbose=True):
  '''
  Precondition for the emission comparison: the two runs must share every physical
  quantity that enters the flux, so the alpha sweep, frequency windows and observer
  grids coincide and the pairs line up pointwise.

  A plain common-span check cannot be used here for two reasons: the runs are meant to
  diverge (that is the measurement, see causality_gate), and a raw cell index k is a
  different physical cell in each run because Next differs.
  '''
  ea, eb = MyEnv(key_a), MyEnv(key_b)
  ok, rows = True, []
  for a in _ENV_PHYS:
    va, vb = getattr(ea, a, None), getattr(eb, a, None)
    if va is None or vb is None:
      rows.append((a, va, vb, np.nan, va is vb))
      continue
    try:
      dev = abs(float(vb) - float(va))/(abs(float(va)) if float(va) else 1.)
    except (TypeError, ValueError):
      continue
    ok &= dev <= rtol
    rows.append((a, float(va), float(vb), dev, dev <= rtol))
  if verbose:
    print(f'\n--- env match: {key_a} vs {key_b} ---')
    for a, va, vb, dev, good in rows:
      if not good or (isinstance(dev, float) and dev > 0):
        print(f'  {a:>8s}: {va!r} vs {vb!r}  (rel dev {dev:.2e})')
    print(f'  Next: {int(ea.Next)} vs {int(eb.Next)}   (expected to differ -- index offset only)')
    print('  -> physics identical, comparison is pointwise' if ok else
          '  -> MISMATCH: the runs differ by more than the boundary; comparison is void')
  return bool(ok)


def common_Tmax(z=4, margin=0.99):
  '''
  Largest observer time both runs cover, for the matched emission window.

  The runs stop at very different bar{T}: the fiducial's cells end at ~646-650 (RS),
  the semi run's at ~99, because ITMAX_ fired before TSTOP_ (the confined layer keeps
  a small CFL timestep -- see matched_window). Integrating each to its own end would
  compare "confined AND 6.5x shorter", and a comparable 5x duration change was measured
  to move the low-frequency late lightcurve by up to 2.1x.

  Tmax is the HISTORY window in get_shell_nuFnu_fromData, so passing this to BOTH
  sweeps matches the windows without re-extracting or truncating either run. The semi
  run's cells simply run out of data at its own end; the fiducial's are cut there.
  '''
  from sweep_gammacm import data_end_barT
  ends = [data_end_barT(k, z=z) for k in (SEMI_KEY, REF_KEY)]
  return margin*min(e[0] for e in ends if e)


def emission_compare(z=4, log10ratio_arr=None, nproc=None, use_cache=True,
    outdir=None, labels=EMISSION_LABELS, Tmax=None):
  '''
  The rarefaction's cost in the observables, measured rather than prescribed.

  Runs the data-method sweep over cooling regime on both simulations (each in its own
  key-tagged cache, sweep_gammacm.method_outdir) and drives the whole sweep_compare
  machinery with the semi run as side A, so every ratio panel reads full/no-rf: > 1 is
  emission the rarefaction removes.

  Companion to sweep_rarcut, which compares rar_cut='model' against the same reference
  on the SAME simulation. Read the two together: sweep_rarcut is the cost of the sharp
  cut-off PRESCRIPTION, this is the cost of the physical wave, and the gap between them
  is how good the prescription is.
  '''
  import sweep_compare as cmp
  from sweep_gammacm import (run_sweep, load_sweep, method_outdir, data_end_barT,
      exit_onset_barT, rarefaction_off_barT, LOG10RATIO_ARR, trim_pngs)

  if log10ratio_arr is None:
    log10ratio_arr = LOG10RATIO_ARR
  outdir = (EMISSION_OUTDIR if z == 4 else f'{EMISSION_OUTDIR}_z={z}') if outdir is None else outdir
  os.makedirs(outdir, exist_ok=True)

  if not check_env_match():
    raise RuntimeError('env mismatch between the runs; see check_env_match output')

  if Tmax is None:
    Tmax = common_Tmax(z)
  print(f'matched observer window: Tmax = {Tmax:.3f} (both runs)')

  # The sweep cache directory is keyed on (method, key, z) but NOT on Tmax, so reusing
  # method_outdir here would silently overwrite the fiducial's existing TMAX=1000 cache
  # -- the reference every other module (sweep_rarcut, sweep_shells, ...) compares
  # against. Tag the directory with Tmax so the two coexist.
  sdir = lambda key: f'{method_outdir(EMISSION_METHOD, key, z)}_Tmax{Tmax:.0f}'
  for key in (SEMI_KEY, REF_KEY):
    d = sdir(key)
    os.makedirs(d, exist_ok=True)
    if not (use_cache and load_sweep(d)):
      print(f'--- running the {EMISSION_METHOD} sweep on {key}, shell z={z}, '
            f'Tmax={Tmax:.3f} ---')
      # skip_cached MUST be forwarded (see sweep_gammacm.main): run_sweep defaults it
      # to True, so use_cache=False would otherwise recompute nothing.
      run_sweep(key, log10ratio_arr, z=z, method=EMISSION_METHOD, nproc=nproc,
                Tmax=Tmax, outdir=d, skip_cached=use_cache)
  pairs = cmp.load_pairs(sdir(SEMI_KEY), sdir(REF_KEY))

  barT_f = exit_onset_barT(REF_KEY, z=z)
  barT_off = rarefaction_off_barT(REF_KEY, z=z)
  barT_end = (data_end_barT(SEMI_KEY, z=z), data_end_barT(REF_KEY, z=z))
  print(f'crossing bar_T_f = {barT_f:.4f}' +
        (f', modelled rarefaction cut-off {barT_off[0]:.4f}..{barT_off[1]:.4f}' if barT_off else ''))
  for key, b, lab in zip((SEMI_KEY, REF_KEY), barT_end, labels):
    if b:
      print(f'end of data ({lab}): bar_T = {b[0]:.4f}..{b[1]:.4f}')

  cmp.plot_efficiency_compare(pairs, outdir=outdir, labels=labels)
  for kind in ('peak', 'fluence'):
    for mode in ('nu_m', 'max'):
      cmp.plot_spectra_compare(pairs, kind=kind, mode=mode, outdir=outdir, labels=labels,
                               ratio=(kind != 'fluence'))
  cmp.plot_spectral_evolution_compare(pairs, barT_f, outdir=outdir, labels=labels)
  for sc in ('log', 'linlog', 'lin'):
    cmp.plot_lightcurve_compare(pairs, barT_f, barT_off=barT_off, outdir=outdir,
        labels=labels, barT_end=barT_end, scale=sc)
  s = cmp.plot_summary_ratios(pairs, outdir=outdir, labels=labels)
  fs = cmp.fluence_split(pairs, barT_off[1] if barT_off else None, outdir=outdir,
      labels=labels, cut_label='rarefaction', cut_math='rarefaction')
  # the shape question: the modelled cut pins the time-integrated low-energy index at
  # 4/3 while the uncut reference softens toward 0.5 (see rarcut_lowenergy_slope). The
  # semi run has no rarefaction at all, so it should track the REFERENCE here, not the
  # cut. If it pins at 4/3 instead, the cut is not doing what we think it is.
  series = cmp.fluence_series(pairs)
  tab = cmp.fluence_slope_table(series, labels=labels)
  tab.to_csv(os.path.join(outdir, 'fluence_low_slopes.csv'), index=False)
  cmp.plot_fluence_slope_profile(series, outdir=outdir, labels=labels,
                                 title_extra=f'  ({"RS" if z == 4 else "FS"})')
  trim_pngs(outdir)
  print(f'Boundary-comparison figures saved to {outdir}')
  return pairs, s, fs, tab


def main(z_list=(4, 1), emission=False, nproc=None):
  '''Hydro diagnostics for both shells; emission=True adds the sweep (much slower).'''
  for z in z_list:
    causality_gate(z)
    arrival_radii(z)
    pressure_retention(z)
    crash_detector(z)
  if emission:
    for z in z_list:
      emission_compare(z, nproc=nproc)


if __name__ == '__main__':
  main()
