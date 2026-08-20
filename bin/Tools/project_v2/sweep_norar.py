# -*- coding: utf-8 -*-
# @Author: acharlet

'''
What does the rarefaction wave itself cost the emission?

Its two companions each answer a different question, and neither answers this one:

  sweep_rarcut.py      'data' vs 'data_rarcut' -- the cost of the sharp R_rar CUT-OFF
                       PRESCRIPTION. Same hydro, but the cut side stops at R_rar and the
                       reference runs to bar{T} ~ 650, so "the tail was removed" and "the
                       tail is weak because the hydro crashed" are conflated.
  boundary_comparison  cooling_g100 vs cooling_g100_semi -- the cost of the PHYSICAL wave,
  .emission_compare    measured rather than modelled, but across two SIMULATIONS, which
                       forces a common window (common_Tmax -> Tmax ~ 99) and leaves the
                       two runs' grids and durations to be matched by hand.

This module holds the worldline EXTENT fixed and changes only whether the crash happens,
on ONE run over the full bar{T} ~ 650 window:

    'data'        cells followed to their last snapshot          REFERENCE
    'data_norar'  the same cells, same final radius, but the
                  rarefaction crash replaced by the smooth
                  shocked-layer decay from the handover onward

Ratios are reported full/no-rf, i.e. reference/counterfactual, so <= 1 is emission the
rarefaction destroys.

CONSTRUCTION (working_cooling_data.norar_history / _norar_rows). Real snapshot rows are
kept out to the pressure crash -- detected with the same look-ahead d ln p/d ln x rule the
fit path uses (truncate_at_rarefaction), so the handover coincides with the end of that
cell's fit window -- and a synthetic tail continues the decay to that cell's own last
radius. The target radius is read off the history itself, so the endpoints match EXACTLY
(measured ratio 1.000000 on every cell tested), which a {cell: R_end/R_inj} ratio map
could not do: with early_ana='shockfit' the injection row sits 0.8-5% below the first
measured row.

THE EXTENSION LAW IS NOT FITTED, AND IS NOT CALIBRATED ON ANYTHING. Every ingredient
follows from what "no rarefaction" means kinematically, plus mass conservation and the EoS:

    Gamma = Gamma_h, dr = dr_h    no wave arrived => no acceleration, no spreading
    rho ~ R^-2                    mass conservation (rho*Gamma*R^2*dr = const)
    dlnp/dlnrho = gma_ad(T)       Taub-Matthews, INTEGRATED as the expansion cools the gas
                                  (T = p/rho = 0.031..0.051 here, so gma_ad = 1.642..1.652
                                  rising toward 5/3; _adiabat_integrated)

cooling_g100_semi (Next=0, rigid dp/dr=0 clamp) is NOT a control for this and must not be
used to tune it, however tempting. Its decompression fires at the SAME radius as the
fiducial's (R_h/R_inj = 1.001..2.35): the onset is set by the shock exhausting the shell
and is boundary-independent to ~1%, the edge setting only the DEPTH. On top of that its
rigid clamp CONFINES the layer, holding p ABOVE free adiabatic coasting by up to 1.28x at
R/R_inj = 30 (measured by splicing semi's own cells at their own handover and comparing
with their own later profile). Two contaminations in opposite directions, neither
calibrated. See the memory note rarefaction-onset-boundary-independent.

law_validation() is therefore a COARSE SANITY CHECK, not a go/no-go: it can reject a law
that is orders of magnitude wrong, and it does exactly that for the obvious alternative --
extrapolating each cell's smooth-BPL fit (law='bpl'), which overshoots by 18-90x at
R/R_inj = 30 and 87-153x at 100, against semi's own ~1.3x bias. Those fits are constrained
over R/R_inj in [1, 1.06..3.15], i.e. 0.03 to 0.5 decades entirely inside the
shock-crossing transient, and asymptote to rho ~ R^-1.2..-1.7, p ~ R^-2.0..-2.4; several
cells' Gamma fit also decays, and the extra observer lag then pushes Ton past Tmax so the
counterfactual stops SHORT of the reference, destroying the endpoint match the whole
design exists for. What law_validation CANNOT do is arbitrate the tens-of-percent choices
(frozen vs integrated gma_ad, dr const vs dr ~ R^-0.044); those are settled on physics.

MEASURED on cooling_g100, RS z=4, Bernoulli closure (2026-08-15):

    logr   peak_flux  fluence_nu  fluence_tot  eps_rad        <- ratios are full / no-rf
      -5      1.0000      1.0000       1.0000   1.0000
      -3      1.0000      1.0000       0.9984   0.9985
      -1      1.0000      0.9743       0.9386   0.9453
      +0      1.0000      0.8695       0.8980   0.9064
      +2      1.0000      0.8788       0.8762   0.8846

so the wave costs ~11.5% of the radiated energy and ~12% of the fluence in slow cooling,
and nothing below logr = -3 (the electrons burn gamma_max -> 1 before the handover).

THREE INTERNAL CONSISTENCY CHECKS, all of which a correct construction must pass:
  peak_flux = 1.0000 EXACTLY in every regime -- the peak is emitted before the handover,
    so both sides are the same cell there (sweep_rarcut finds the same for its own pair);
  the lightcurve ratio first departs from 1 at bar{T}/bar{T}_f = 1.036..1.067, i.e. AT the
    rarefaction band (1.00..1.55), not before it;
  max ratio = 1.0000 everywhere -- removing the crash only ever ADDS emission.
An earlier run violated all three (peak 1.048, divergence at 0.068 bar{T}_f, ratio > 1);
that was the shockfit-prepend detector bug, see NORAR_PERSIST.

Per-cell: E_rad(full)/E_rad(no rf) rises monotonically from the outer edge to the CD --
the wave arrives earliest at the outer edge, which is shocked LAST, so the effect lands
near and after bar{T}_f. Post-arrival fluence share: 35.9% (no rf) vs 28.4% (full) at
logr=+2. The time-integrated low-energy index moves a = 0.906 -> 1.224.

SYSTEMATICS of the extension law, each measured on the OBSERVABLE rather than argued:
  Bernoulli vs Gamma=const ('adiab', cached in gammacm_sweep_data_norar_adiab)
                                     <= 1.15 pts on eps_rad, 0.73 on fluence
  dr_slope 0 -> -0.05 (a plausible residual contraction)   <= 1.2% on E_rad
  frozen vs integrated gma_ad                              <= 0.21% on E_rad
  extension length: capped at R/R_inj = 30 vs uncapped      <= 0.01 pts (D4)
Total law freedom ~1 pt against an ~11.5 pt effect: the conclusion is robust to every one
of these, and they belong in the paper as a bounded systematic rather than a debate.

The one surviving ASSUMPTION is dr = const. It is known to be imperfect: at the handover
the cell is still contracting at dln(dr)/dlnR = -0.59..-0.82 (the tail of the shock-
crossing transient), and the closure switches to 0 there. That is what the dr_slope
systematic above bounds.

GEOMETRY: R_h/R_inj = 1.063..3.145, R_end/R_inj = 401..950, so the full variant
extrapolates ~2.5 decades and the capped one (r_cap = R_CAP = 30, applied to BOTH sides so
the endpoints still match) ~1.0-1.5. D4 shows the difference is <0.01 pts, i.e. the long
extrapolation is not load-bearing.

KNOWN OUTLIER k=20, the outermost cell: a spurious early Sd block makes
select_postshock_rows start it 2.4x too early in radius (R_end/R_inj = 891 against ~401
for its neighbours), so its injection state is misidentified. Pre-existing, and the
REFERENCE sees exactly the same wrong state, so handover_table flags it (by a break in the
R_end/R_inj ladder) rather than dropping it; it carries eps_rad ~ 0.007 against 0.25-0.66
elsewhere, i.e. nothing.

Example use in command line:
  python -c "import sweep_norar as N; N.diagnostics()"
  python -c "import sweep_norar as N; N.main(nproc=7)"
  python -c "import sweep_norar as N; N.main_capped(nproc=7)"
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from environment import MyEnv, GAMMA_dir
from IO import open_celldata
from working_cooling import cell_radiated_energy, cell_injected_energy
from working_cooling_data import (generate_cell_fromData, select_postshock_rows,
    norar_history, _norar_fits, load_shockfront_states, _prepend_shocked_row,
    NORAR_LAW, NORAR_PTS_PER_DEC, NORAR_PRECURSOR_TOL)
import sweep_compare as cmp
from sweep_gammacm import (run_sweep, load_sweep, method_outdir, data_end_barT,
    cap_end_barT, exit_onset_barT, rarefaction_off_barT, compute_alpha_sweep, trim_pngs,
    copy_article_figures, LOG10RATIO_ARR, Z_SHELL, R_REF, TMAX, R_CAP, EARLY_ANA)
import boundary_comparison as bc

KEY = 'cooling_g100'
# A = the counterfactual, B = the REFERENCE. sweep_compare's convention: dashed = A,
# solid = B, ratios read B/A, and B sets the flux normalisation.
METHOD_A, METHOD_B = 'data_norar', 'data'
LABELS = ('no rf', 'full')          # matches boundary_comparison.EMISSION_LABELS, so the
                                    # two studies' figures read side by side
OUTDIR = os.path.join(GAMMA_dir, 'bin', 'Tools', 'figures', 'norar_compare')

N_PROFILE = 42                      # cells in the per-cell profiles
LOGR_PROFILE = (-3., 0., 2.)        # cooling regimes shown per cell
# law_validation: R/R_inj at which the spliced hydro is compared with the semi run. All
# inside its coverage (64..155 across the shell).
LAW_TARGETS = (3., 10., 30., 100.)
LAW_PASS_DEX = 0.5                  # |ln(spliced/semi)| on p below this = the law tracks
# a cell's R_end/R_inj deviating from its LOCAL neighbourhood by more than this is
# flagged as a misidentified injection row (see handover_table). Across cooling_g100's RS
# the ladder steps by ~0.13% per cell, so 0.5 is ~400x above the honest local variation
# while staying far below the k=20 defect (2.4x).
R_END_OUTLIER = 0.5
R_END_WINDOW = 5
SHELL_NAME = {4: 'RS', 1: 'FS'}


# ---------------------------------------------------------------------------
# cell bookkeeping
# ---------------------------------------------------------------------------
def shell_cell_range(key=KEY, z=Z_SHELL):
  '''(first, last, CD-side) cell indices of shell z; grid is [Next | Nsh4 | Nsh1 | Next].'''
  env = MyEnv(key)
  k4 = int(env.Next)
  kCD = k4 + int(env.Nsh4)
  k1 = kCD + int(env.Nsh1)
  return (k4, kCD - 1, kCD - 1) if z == 4 else (kCD, k1 - 1, kCD)


def profile_cells(key=KEY, z=Z_SHELL, n=N_PROFILE):
  '''n cell indices spread over shell z (endpoints included).'''
  kmin, kmax, _ = shell_cell_range(key, z)
  return np.unique(np.linspace(kmin, kmax, n).astype(int))


def _col(frame, v):
  '''Column v of a history frame as float64. lfac is absent from the raw cell CSVs (and
  therefore from the synthetic rows, which carry the parent's columns) -- everything
  downstream derives it from vx, so do the same here.'''
  if v in frame:
    return frame[v].to_numpy(dtype=float)
  if v == 'lfac':
    vx = frame['vx'].to_numpy(dtype=float)
    return 1./np.sqrt(1. - vx**2)
  raise KeyError(v)


_SH_MEM = {}

def _history(key, k, n_settle=1, z=Z_SHELL, early_ana=EARLY_ANA, early_frac=0.):
  '''
  Post-shock history EXACTLY as get_shell_nuFnu_fromData's first pass builds it, shockfit
  prepend included.

  The prepend is not optional bookkeeping: it inserts the shock-front state at a smaller
  radius and a much higher pressure than the first measured row, and the crash detector
  reads slopes. Diagnostics that skip it validate a history production never sees -- which
  is how a detector artifact (a single-row jump read as a crash, dragging 30+ cells from
  R_h/R_inj ~ 2.7 to ~1.0) survived a full round of per-cell checks here. Keep this the
  one place histories are built for the diagnostics.
  '''
  d = open_celldata(key, k)
  if d is False:
    return None
  s = select_postshock_rows(d, n_settle)
  if len(s) < 2:
    return None
  s = s.copy()
  s['t'] = s['t'] - (d.t.iloc[0] if d.index[0] == 0 else 0.)
  if early_ana is not None:
    if (key, z, early_ana) not in _SH_MEM:
      _SH_MEM[(key, z, early_ana)] = load_shockfront_states(key, z, MyEnv(key),
                                                            source=early_ana)
    sh = _SH_MEM[(key, z, early_ana)]
    sel = sh.loc[sh.i == k]
    if len(sel):
      s = _prepend_shocked_row(s, sel.iloc[0], MyEnv(key), early_frac)
  return s


# ---------------------------------------------------------------------------
# D1: where does the handover land, and is there anything to extend?
# ---------------------------------------------------------------------------
def handover_table(key=KEY, z=Z_SHELL, cells=None, verbose=True):
  '''
  Per cell: the crash-detector outcome, the handover radius and the target radius.
  status 'no_crash' means the counterfactual IS the reference for that cell -- correct,
  but if it is common the detector is not seeing the wave.
  '''
  if cells is None:
    kmin, kmax, _ = shell_cell_range(key, z)
    cells = np.arange(kmin, kmax + 1)
  cells = np.asarray(cells)
  rows = []
  for k in cells:
    s = _history(key, k)
    if s is None:
      continue
    out, info = norar_history(s)
    x = s.x.to_numpy(dtype=float)
    rows.append(dict(k=int(k), status=info['status'], h=info['h'], n_syn=info['n_syn'],
                     n_rows=len(s), R_h=info['x_h']/x[0], R_end=x[-1]/x[0]))
  df = pd.DataFrame(rows)
  if not len(df):
    return df
  # a cell that BREAKS THE LADDER in R_end/R_inj has a MISIDENTIFIED INJECTION ROW, not
  # an unusual worldline: R_end is common to the whole shell (one simulation, one final
  # time), so the ratio varies only through R_inj -- smoothly and monotonically, since
  # cells are shocked in order. The test is therefore against the LOCAL neighbourhood,
  # not against the shell median: the honest spread is already a factor 2.5 (376..943
  # here), so a median test cannot see a single cell sitting at its far end.
  # On cooling_g100 this flags k=20, whose spurious early Sd block makes
  # select_postshock_rows start it 2.4x too early in radius. Pre-existing, and the
  # REFERENCE sees the same wrong state, so it is reported, never silently dropped; it
  # carries eps_rad ~ 0.007 against 0.25-0.66 elsewhere, i.e. nothing.
  df = df.sort_values('k').reset_index(drop=True)
  loc = df.R_end.rolling(R_END_WINDOW, center=True, min_periods=2).median()
  # rolling median includes the cell itself; with a window of 5 one bad cell cannot move
  # it, and the edges fall back to whatever neighbours exist
  df['R_end_outlier'] = (df.R_end/loc - 1.).abs() > R_END_OUTLIER
  if verbose:
    n = len(df)
    print(f'handover_table {key} {SHELL_NAME.get(z, z)} z={z}: {n} cells')
    for st, g in df.groupby('status'):
      print(f'  {st:9s} {len(g):4d}  ({100.*len(g)/n:5.1f}%)')
    ok = df[df.status == 'ok']
    if len(ok):
      print(f'  R_h/R_inj   {ok.R_h.min():7.3f} .. {ok.R_h.max():7.3f}')
      print(f'  R_end/R_inj {ok.R_end.min():7.1f} .. {ok.R_end.max():7.1f}')
      print(f'  synthetic rows per cell {ok.n_syn.min():d} .. {ok.n_syn.max():d} '
            f'(on {ok.n_rows.min():d}..{ok.n_rows.max():d} real rows)')
    bad = df[df.R_end_outlier]
    if len(bad):
      print(f'  *** {len(bad)} cell(s) with a misidentified injection row: R_end/R_inj '
            f'breaks the local ladder by >{100*R_END_OUTLIER:.0f}% -- '
            + ', '.join(f'k={int(r.k)} ({r.R_end:.0f} vs {loc[i]:.0f} nearby)'
                        for i, r in bad.iterrows()))
  return df


# ---------------------------------------------------------------------------
# D0: does the extension law track a simulation that really has no rarefaction?
# ---------------------------------------------------------------------------
def law_validation(z=Z_SHELL, targets=LAW_TARGETS, laws=('adiab', 'bpl'),
    ncells=40, outdir=OUTDIR, plot=True, verbose=True,
    precursor_tol=NORAR_PRECURSOR_TOL, dr_slope=None):
  '''
  THE GO/NO-GO. Splice each fiducial cell with each law and divide the result by
  cooling_g100_semi's ACTUAL hydro at the same R/R_inj -- the semi run (Next=0, outflow
  ghosts) being a simulation of the very counterfactual this module constructs.

  Cells are paired by POSITION WITHIN THE SHELL (boundary_comparison.paired_cells), never
  by raw index: Next differs between the two runs, so the same physical cell carries
  indices 20 apart and pairing on the index silently compares different cells.

  Both sides are normalised by their OWN injection radius, as boundary_comparison.
  dlnp_profile does, so the comparison is of profiles, not of absolute placement.
  '''
  pairs = bc.paired_cells(z)
  if ncells is not None and len(pairs) > ncells:
    pairs = [pairs[i] for i in np.unique(np.linspace(0, len(pairs)-1, ncells).astype(int))]
  env_r = MyEnv(bc.REF_KEY)
  fits = _norar_fits(bc.REF_KEY, z, env_r, [kr for _, kr, _ in pairs]) \
         if 'bpl' in laws else {}
  rows = []
  for j, k_ref, k_semi in pairs:
    s = _history(bc.REF_KEY, k_ref)
    ss = _history(bc.SEMI_KEY, k_semi)
    if s is None or ss is None:
      continue
    xs = ss.x.to_numpy(dtype=float)
    rs = xs/xs[0]
    for law in laws:
      try:
        kwd = {} if dr_slope is None else dict(dr_slope=dr_slope)
        out, info = norar_history(s, law=law, fit=fits.get(int(k_ref)),
                                  precursor_tol=precursor_tol, **kwd)
      except ValueError:
        continue
      if info['status'] != 'ok':
        continue
      xo = out.x.to_numpy(dtype=float)
      ro = xo/xo[0]
      for tgt in targets:
        if tgt > rs[-1] or tgt > ro[-1] or tgt < ro[0]:
          continue                       # outside one run's coverage: not a failure
        r = dict(j=int(j), k_ref=int(k_ref), law=law, target=float(tgt))
        for v in ('p', 'rho', 'lfac'):
          a = np.interp(tgt, ro, _col(out, v))
          b = np.interp(tgt, rs, _col(ss, v))
          r[v] = a/b
        rows.append(r)
  df = pd.DataFrame(rows)
  if verbose and len(df):
    print(f'law_validation {SHELL_NAME.get(z, z)} z={z}: spliced / {bc.SEMI_KEY}, '
          f'{df.k_ref.nunique()} paired cells')
    for law, g in df.groupby('law'):
      print(f'  law={law}')
      for tgt, gg in g.groupby('target'):
        print(f'    R/R_inj={tgt:6.1f}  p {gg.p.min():8.3f}..{gg.p.max():8.3f}   '
              f'rho {gg.rho.min():6.3f}..{gg.rho.max():6.3f}   '
              f'Gamma {gg.lfac.min():6.3f}..{gg.lfac.max():6.3f}')
      inb = g[g.target <= R_CAP]
      frac = float((np.abs(np.log(inb.p)) < LAW_PASS_DEX).mean()) if len(inb) else np.nan
      print(f'    -> |ln(p ratio)| < {LAW_PASS_DEX} on {100*frac:.1f}% of points at '
            f'R/R_inj <= {R_CAP:g}  [{"PASS" if frac >= 0.9 else "FAIL"}]')
  if plot and len(df):
    os.makedirs(outdir, exist_ok=True)
    # sharey per ROW: the two laws must be read on the SAME scale, otherwise matplotlib
    # autoscales each column to its own range and a 90x failure looks like a 1.4x one
    fig, axes = plt.subplots(3, len(laws), figsize=(5.2*len(laws), 8.4), sharex=True,
                             sharey='row', squeeze=False)
    cols = plt.cm.viridis(np.linspace(0., .9, len(targets)))
    vlab = {'p': '$p$', 'rho': r'$\rho$', 'lfac': r'$\Gamma$'}
    for ic, law in enumerate(laws):
      g = df[df.law == law]
      for ir, v in enumerate(('p', 'rho', 'lfac')):
        ax = axes[ir][ic]
        for it, tgt in enumerate(targets):
          gg = g[g.target == tgt].sort_values('j')
          if len(gg):
            ax.plot(gg.j, gg[v], color=cols[it], lw=1.2,
                    label=f'$R/R_{{inj}}={tgt:g}$')
        ax.axhline(1., color='k', lw=.8, ls=':')
        ax.set_yscale('log')
        ax.set_ylabel(f'{vlab[v]} spliced / semi')
        if ir == 0:
          ax.set_title(f"law = '{law}'")
        if ir == 0 and ic == 0:
          ax.legend(fontsize=7, ncol=2)
      axes[-1][ic].set_xlabel('cell position in shell $j$')
    fig.suptitle(f'extension law vs a simulation with no rarefaction '
                 f'({bc.SEMI_KEY}), {SHELL_NAME.get(z, z)}')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f'law_validation_z{z}.png'), dpi=150)
    plt.close(fig)
    df.to_csv(os.path.join(outdir, f'law_validation_z{z}.csv'), index=False)
  return df


# ---------------------------------------------------------------------------
# D2: does the counterfactual worldline actually reach the matched radius?
# ---------------------------------------------------------------------------
def _reach(cell):
  '''R_final/R_injection of a built cell frame. The frame stores the LEFT edge of each
  cooling step, so x.iloc[-1] alone under-reports the endpoint by one step (~0.5%).'''
  return float((cell.x.iloc[-1] + cell.vx.iloc[-1]*cell.dt.iloc[-1])/cell.x.iloc[0])


def endpoint_table(key=KEY, z=Z_SHELL, logr_list=(-5., -3., 0., 2.), cells=None,
    cap=None, law=NORAR_LAW, r_ref=R_REF, Tmax=TMAX, verbose=True):
  '''
  The check the whole design rests on: per (cell, regime), does the counterfactual stop at
  the SAME radius as the reference? Three bounds sit between the history and the emitting
  worldline (generate_cell_fromHistory): the end of the history, the gamma_max -> 1
  cooling budget (tt_geo), and the Tmax observer window.

  bound='gmax' is LEGITIMATE, not a failure: both sides exhaust the electron budget before
  the history ends, the counterfactual slightly sooner because no crash quenches syn. The
  endpoints then do not match because neither cell survives to them, and the emission is
  identical anyway (the burn-out precedes the handover).

  bound='Tmax' on the counterfactual but not the reference IS a failure: the extension's
  own worldline ran past the observer window, which is the law='bpl' Gamma-decay mode.
  Reported with a loud warning, never silently.
  '''
  cells = profile_cells(key, z) if cells is None else np.asarray(cells)
  alphas, _ = compute_alpha_sweep(key, np.asarray(logr_list, dtype=float))
  env0 = MyEnv(key)
  fits = _norar_fits(key, z, env0, cells) if law == 'bpl' else {}
  rows = []
  for logr, alpha in zip(logr_list, alphas):
    for k in cells:
      d = open_celldata(key, k)
      if d is False:
        continue
      kw = dict(alpha=float(alpha), r_ref=r_ref, Tmax=Tmax, r_cap=cap)
      a, ea = generate_cell_fromData(d, env0, **kw)
      b, eb = generate_cell_fromData(d, env0, norar=law,
                                     norar_fit=fits.get(int(k)), **kw)
      if a is False or b is False:
        continue
      ra, rb = _reach(a), _reach(b)
      # which bound stopped each side: tt_geo[-1] ~ 1 is the gamma_max -> 1 budget
      tt_a, tt_b = float(a.tt.iloc[-1] + a.dtt.iloc[-1]), float(b.tt.iloc[-1] + b.dtt.iloc[-1])
      Ton_b = (1. + eb.z)*(b.t.iloc[-1] + b.dt.iloc[-1] + eb.t0 - b.x.iloc[-1])
      bound = 'data'
      if abs(np.log(rb/ra)) > 1e-6:
        bound = 'Tmax' if Ton_b >= (eb.Ts + Tmax*eb.T0)*(1. - 1e-9) else 'gmax'
      Erad_a, Erad_b = cell_radiated_energy(a, ea), cell_radiated_energy(b, eb)
      rows.append(dict(logr=float(logr), k=int(k), bound=bound,
                       R_ref=ra, R_cf=rb, dlnR=float(np.log(rb/ra)),
                       tt_ref=tt_a, tt_cf=tt_b,
                       eps_ref=Erad_a/cell_injected_energy(a, ea),
                       eps_cf=Erad_b/cell_injected_energy(b, eb),
                       Erad_ratio=Erad_a/Erad_b))
  df = pd.DataFrame(rows)
  if verbose and len(df):
    print(f'endpoint_table {key} {SHELL_NAME.get(z, z)} z={z}, law={law}, '
          f'cap={cap}, {df.k.nunique()} cells')
    for logr, g in df.groupby('logr'):
      cnt = g.bound.value_counts().to_dict()
      print(f'  logr={logr:+.1f}  bounds={cnt}  '
            f'max|dlnR|={g.dlnR.abs().max():.2e}  '
            f'E_rad full/no-rf {g.Erad_ratio.min():.4f}..{g.Erad_ratio.max():.4f}')
    bad = df[df.bound == 'Tmax']
    if len(bad):
      print(f'  *** WARNING: {len(bad)} (cell, regime) points stopped on the OBSERVER '
            f'WINDOW, not on the data: the endpoint match FAILED there. '
            f'cells={sorted(bad.k.unique().tolist())}')
    mism = df[(df.bound == 'data') & (df.dlnR.abs() > 1e-6)]
    if len(mism):
      print(f'  *** WARNING: {len(mism)} points claim bound=data yet differ in reach')
    print(f'  eps_rad max: reference {df.eps_ref.max():.5f}, '
          f'counterfactual {df.eps_cf.max():.5f}  (must stay <= 1.002)')
  return df


# ---------------------------------------------------------------------------
# D3: sampling convergence of the synthetic tail
# ---------------------------------------------------------------------------
def ppd_convergence(key=KEY, z=Z_SHELL, ppds=(32, 64, 128), logr_list=(-3., 0., 2.),
    cells=None, law=NORAR_LAW, r_ref=R_REF, Tmax=TMAX, verbose=True):
  '''E_rad of the counterfactual vs the sampling of the synthetic tail (points per decade
  of R). The reference is untouched by this, so only the counterfactual is scanned.'''
  cells = profile_cells(key, z, 6) if cells is None else np.asarray(cells)
  alphas, _ = compute_alpha_sweep(key, np.asarray(logr_list, dtype=float))
  env0 = MyEnv(key)
  rows = []
  for logr, alpha in zip(logr_list, alphas):
    for k in cells:
      d = open_celldata(key, k)
      if d is False:
        continue
      e = {}
      for ppd in ppds:
        c, ce = generate_cell_fromData(d, env0, alpha=float(alpha), r_ref=r_ref,
                                       Tmax=Tmax, norar=law, norar_ppd=ppd)
        e[ppd] = cell_radiated_energy(c, ce) if c is not False else np.nan
      ref = e[ppds[-1]]
      rows.append(dict(logr=float(logr), k=int(k),
                       **{f'rel_{p}': e[p]/ref - 1. for p in ppds}))
  df = pd.DataFrame(rows)
  if verbose and len(df):
    print(f'ppd_convergence: E_rad relative to ppd={ppds[-1]}')
    for p in ppds[:-1]:
      print(f'  ppd={p:4d}  max |dE_rad/E_rad| = {df[f"rel_{p}"].abs().max():.2e}')
    print(f'  (production uses NORAR_PTS_PER_DEC = {NORAR_PTS_PER_DEC})')
  return df


# ---------------------------------------------------------------------------
# per-cell energy profile
# ---------------------------------------------------------------------------
def plot_norar_energy_profile(key=KEY, z=Z_SHELL, logr_list=LOGR_PROFILE,
    n=N_PROFILE, cap=None, law=NORAR_LAW, r_ref=R_REF, Tmax=TMAX, outdir=OUTDIR,
    use_cache=True):
  '''
  Per-cell comoving E_rad(full)/E_rad(no-rf) across the shell, one curve per cooling
  regime, with the handover and target radii beneath. <= 1 is emission the rarefaction
  destroyed; the effect should be largest at the OUTER EDGE, where the wave arrives
  earliest (R_h/R_inj -> 1) and which is shocked last.
  '''
  os.makedirs(outdir, exist_ok=True)
  cells = profile_cells(key, z, n)
  suff = f'_cap={cap:g}' if cap else ''
  csv = os.path.join(outdir, f'norar_energy_profile_z{z}{suff}.csv')
  # the table costs ~2 cell builds per (cell, regime); re-plotting must not pay that again
  df = pd.read_csv(csv) if (use_cache and os.path.isfile(csv)) else \
       endpoint_table(key, z, logr_list=logr_list, cells=cells, cap=cap, law=law,
                      r_ref=r_ref, Tmax=Tmax, verbose=False)
  ht = handover_table(key, z, cells=cells, verbose=False)
  # cells whose injection row is misidentified (k=20 here) invert the ratio and would set
  # the y-scale for the whole figure, hiding the physical 0.5-1.0 signal. Plotted as
  # isolated markers and EXCLUDED from the limits, never dropped -- see handover_table.
  bad = set(ht.k[ht.R_end_outlier].tolist()) if len(ht) else set()
  # drop first: the CSV this may have been reloaded from already carries the column, and
  # merging onto it would silently produce R_end_outlier_x / _y instead of failing
  df = df.drop(columns=['R_end_outlier'], errors='ignore') \
         .merge(ht[['k', 'R_end_outlier']], on='k', how='left')

  fig, (ax, axr) = plt.subplots(2, 1, figsize=(7., 6.4), sharex=True,
                                gridspec_kw=dict(height_ratios=[2.2, 1.]))
  cols = plt.cm.jet(np.linspace(0., 1., len(logr_list)))
  lo, hi = 1., 1.
  for ic, logr in enumerate(logr_list):
    g = df[df.logr == logr].sort_values('k')
    if not len(g):
      continue
    ok, out = g[~g.R_end_outlier.astype(bool)], g[g.R_end_outlier.astype(bool)]
    ax.plot(ok.k, ok.Erad_ratio, color=cols[ic], lw=1.3,
            label=r'$\log_{10}(\gamma_c/\gamma_m) = %+.0f$' % logr)
    ax.plot(out.k, out.Erad_ratio, 'o', color=cols[ic], ms=4, mfc='none')
    lo, hi = min(lo, ok.Erad_ratio.min()), max(hi, ok.Erad_ratio.max())
  ax.axhline(1., color='k', lw=.8, ls=':')
  pad = 0.05*(hi - lo)
  ax.set_ylim(lo - pad, hi + pad)
  ax.set_ylabel(r'$E_{rad}$ full / no rf')
  if bad:
    ax.annotate(f'k={",".join(str(int(b)) for b in sorted(bad))}: injection row '
                'misidentified\n(off scale; see handover_table)',
                xy=(0.34, 0.04), xycoords='axes fraction', fontsize=6.5, color='0.35')
  ax.legend(fontsize=8)
  ax.set_title(f'{key}, {SHELL_NAME.get(z, z)} (z={z}), law={law}'
               + (f', cap={cap:g}' if cap else ''))
  if len(ht):
    axr.plot(ht.k, ht.R_h, color='k', lw=1.2, label=r'$R_h/R_{inj}$ (handover)')
    axr.plot(ht.k, ht.R_end, color='k', lw=1.2, ls='--',
             label=r'$R_{end}/R_{inj}$ (target)')
  if cap:
    axr.axhline(cap, color='C3', lw=1., ls='-.', label=r'$r_{cap}$')
  axr.set_yscale('log')
  axr.set_ylabel(r'$R/R_{inj}$')
  axr.set_xlabel('cell index $k$')
  axr.legend(fontsize=8)
  fig.tight_layout()
  fig.savefig(os.path.join(outdir, f'norar_energy_profile_z{z}{suff}.png'), dpi=150)
  plt.close(fig)
  df.to_csv(csv, index=False)
  return df


def diagnostics(key=KEY, z=Z_SHELL, cap=None, law=NORAR_LAW, outdir=None):
  '''D0 -> D3 in the order they must be run, before any sweep. D0 is the go/no-go.'''
  outdir = (OUTDIR if outdir is None else outdir)
  os.makedirs(outdir, exist_ok=True)
  print('=== D0  law_validation (go/no-go) ===')
  d0 = law_validation(z=z, outdir=outdir)
  print('=== D1  handover_table ===')
  d1 = handover_table(key, z)
  print('=== D2  endpoint_table ===')
  d2 = endpoint_table(key, z, cap=cap, law=law)
  print('=== D3  ppd_convergence ===')
  d3 = ppd_convergence(key, z, law=law)
  print('=== per-cell energy profile ===')
  d4 = plot_norar_energy_profile(key, z, cap=cap, law=law, outdir=outdir)
  return d0, d1, d2, d3, d4


# ---------------------------------------------------------------------------
# the sweep comparison
# ---------------------------------------------------------------------------
def main(key=KEY, log10ratio_arr=LOG10RATIO_ARR, outdir=None, use_cache=True,
    nproc=None, labels=LABELS, z=Z_SHELL, cap=None):
  '''
  Full log10(gamma_c/gamma_m) sweep of the counterfactual against the reference, then the
  sweep_compare figure battery.

  cap=None runs the counterfactual ALONE against the already-cached 'data' reference: a
  paired run would rewrite figures/gammacm_sweep_data/, which sweep_rarcut, sweep_shells,
  sweep_efficiency and boundary_comparison all compare against. With cap set both sides
  are new, so they are computed together (norar='both', sharing their common leading
  cooling steps).
  '''
  method_a = 'data_norar' + ('_cap' if cap else '')
  method_b = 'data' + ('_cap' if cap else '')
  suff = (f'_cap={cap:g}' if cap else '') + (f'_z={z}' if z != Z_SHELL else '')
  outdir = (OUTDIR + suff) if outdir is None else outdir
  os.makedirs(outdir, exist_ok=True)

  # run_sweep resumes from the per-point cache itself (skip_cached), so it is called
  # unconditionally and computes only what is missing -- a partially finished sweep is
  # completed rather than skipped, which a truthiness test on load_sweep would get wrong.
  if cap:
    print(f'--- capped pair on {key}, shell z={z} ---')
    run_sweep(key, log10ratio_arr, z=z, method='cap+norar_cap', nproc=nproc,
              skip_cached=use_cache)
  else:
    print(f'--- {method_a} sweep on {key}, shell z={z} ---')
    run_sweep(key, log10ratio_arr, z=z, method=method_a, nproc=nproc,
              skip_cached=use_cache)
    # the reference is the shared cached computation; only build it if it is missing
    # entirely, and never as part of a paired run that would rewrite it
    if not load_sweep(method_outdir(method_b, key, z)):
      print(f'--- {method_b} reference sweep on {key}, shell z={z} ---')
      run_sweep(key, log10ratio_arr, z=z, method=method_b, nproc=nproc)

  pairs = cmp.load_pairs(method_outdir(method_a, key, z),
                         method_outdir(method_b, key, z))
  barT_f = exit_onset_barT(key, z=z)
  barT_off = rarefaction_off_barT(key, z=z)
  # BOTH sides stop at the same radius, hence at the same observer time to within the
  # extension's few-percent lag difference -- unlike sweep_rarcut, where the two sides
  # differ by ~2.5 decades in bar{T}. That identity IS the design.
  end = cap_end_barT(key, z=z, cap=cap) if cap else data_end_barT(key, z=z)
  barT_end = (end, end)
  print(f'crossing bar_T_f = {barT_f:.4f}')
  print(f'rarefaction reaches the cells at bar_T = {barT_off[0]:.4f}..{barT_off[1]:.4f}')
  if end:
    print(f'both sides stop at bar_T = {end[0]:.4f}..{end[1]:.4f}')

  cmp.plot_efficiency_compare(pairs, outdir=outdir, labels=labels)
  for kind in ('peak', 'fluence'):
    for mode in ('nu_m', 'max'):
      # as in sweep_rarcut: the peak phase is emitted BEFORE the handover, so the two
      # sides sit on top of each other there and the ratio panel is the only way to see
      # the difference; on the fluence spectra the difference is a visible separation
      cmp.plot_spectra_compare(pairs, kind=kind, mode=mode, outdir=outdir, labels=labels,
                               ratio=(kind != 'fluence'))
  cmp.plot_spectral_evolution_compare(pairs, outdir=outdir, labels=labels)
  for sc in ('log', 'linlog', 'lin'):
    cmp.plot_lightcurve_compare(pairs, barT_f, barT_off=barT_off, outdir=outdir,
        labels=labels, barT_end=barT_end, scale=sc)
  s = cmp.plot_summary_ratios(pairs, outdir=outdir, labels=labels)
  fs = cmp.fluence_split(pairs, barT_off[1] if barT_off else None, outdir=outdir,
      labels=labels, cut_label='rarefaction arrival',
      cut_math='rarefaction arrival')
  series = cmp.fluence_series(pairs)
  tab = cmp.fluence_slope_table(series, labels=labels)
  tab.to_csv(os.path.join(outdir, 'fluence_low_slopes.csv'), index=False)
  cmp.plot_fluence_slope_profile(series, outdir=outdir, labels=labels,
                                 title_extra=f'  ({SHELL_NAME.get(z, f"z={z}")})')
  trim_pngs(outdir)
  copy_article_figures(outdir)
  print(f'No-rarefaction comparison figures saved to {outdir}')
  return pairs, s, fs, tab


def main_capped(cap=R_CAP, **kw):
  '''The capped variant: both sides truncated at R/R_inj = cap, so the endpoints still
  match but the extension only has to carry ~1-1.5 decades instead of ~2.5. Compare its
  ratios with main()'s -- that is the convergence check on the extrapolation length.'''
  return main(cap=cap, **kw)


if __name__ == '__main__':
  diagnostics()
