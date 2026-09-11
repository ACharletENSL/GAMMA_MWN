# -*- coding: utf-8 -*-
# @Author: acharlet

'''
Data-driven emission computation: same physics as working_cooling (cooling-binned
cell histories fed to the radiation_cooling kernels) but rooted in the ACTUAL
hydrodynamical data instead of the smooth-BPL fit reconstruction.

Differences with the fit-based pipeline (working_cooling.generate_cell_withDistrib):
  - the cell worldline (t', t, R) and hydro (rho, p, Gamma) come straight from the
    extracted cell history (open_celldata / extract_data_cells): the cooling
    fluence tt = int dt'/t_c1 is a quadrature over the measured syn/lfac instead
    of the worldline ODE over fitted profiles;
  - the injection state is the cell's own first settled post-shock row (Sd block),
    not a fitted cellsBehindShock state;
  - no rarefaction machinery by default: the rarefaction wave is IN the data (pressure
    crash), so cells are simply followed to the end of the available snapshots. The
    Delta ln(rho) refinement resolves the crash itself (little tt accrues there).
    Two opt-in treatments exist to MEASURE the wave rather than to model it, and both
    leave the default path untouched: `rar_ratio`/`rar_cut` (the fit path's sharp cut at
    R_rar -- stops the cell early, so it prices the cut-off prescription, sweep_rarcut),
    and `norar` (keep the real rows to the crash, then continue the smooth decay to the
    SAME final radius -- holds the worldline extent fixed and removes only the crash,
    which is what isolates the wave itself, sweep_prerar);
  - intended for long-term simulations and setups where the BPL fit ansatz breaks.

Kept identical (imported, not duplicated): the geometric-in-gamma_max time binning
(generate_timebins), the electron-bound operator split (evolve_gma_bounds_edges),
the emission kernels (get_Fnu_cell_evolving / get_nuFnu -> radiation_cooling), the
energy budget (cell_radiated_energy / cell_injected_energy) and the u_scale /
alpha / zeta rescaling conventions. The generated cell DataFrame has the same
schema, so everything downstream of the builder is shared.
'''

import collections
from types import SimpleNamespace

from working_cooling import *
from working_cooling import _interp_state, _one_minus_beta_over_beta
from phys_functions_shells import derive_betaRS, derive_betaFS
# (underscore names are skipped by import *)
import cell_pool

# cap on the variation of ln(syn/lfac) across one node interval of the tt/tp
# quadrature (see _densify_nodes): the snapshot grid is refined until the
# cooling rate is resolved at this level, which keeps the fluence dtt consistent
# with the local rate the emission kernel evaluates. The residual energy bias is
# ~dlnsyn_max/2, so 1e-3 holds it below 5e-4 (it was up to 2e-2 unrefined, which
# pushed eps_rad = E_rad/E_inj above 1 in fast cooling). None disables refinement.
DLNSYN_MAX = 1e-3
NSUB_MAX = 256           # max sub-intervals per snapshot interval

# --- counterfactual histories: the shocked layer WITHOUT the rarefaction crash --------
# The data method follows every cell to its last snapshot, so the rarefaction is IN the
# reference. To measure what the wave costs, the worldline EXTENT must be held fixed and
# only the crash removed: keep the real rows out to the pressure crash, then continue the
# smooth shocked-layer decay synthetically to the SAME final radius the cell reached in
# the data. Sharp-cut comparisons (rar_ratio / sweep_rarcut) cannot do this -- they stop
# the cell early, conflating "the tail was removed" with "the tail is weak".
NORAR_SLOPE_THRESH = -8.  # same look-ahead d ln p/d ln x crash rule as the fit path's
NORAR_SLOPE_WINDOW = 15   # working_cooling.truncate_at_rarefaction, so the handover
                          # coincides with the end of that cell's BPL fit window...
NORAR_MINPTS = 2          # ...but NOT its minpts=20 guard. On cooling_g100 z=4 that guard
                          # suppresses the crash on exactly 20 of 500 cells (k=21..40, all
                          # at the outer edge, detected at slope index 0..19) -- and those
                          # are shocked LAST, so they dominate the peak and the late
                          # lightcurve. 0 cells fail to fire outright. 2 = the minimum the
                          # tt/tp quadrature needs.
NORAR_PTS_PER_DEC = 64    # base sampling of the synthetic tail, points per decade of R.
                          # Gives d ln(rho) = 0.072 per interval, just under DLNRHO_MAX
                          # (0.075), and a log-linear interpolation error of 2.5e-4 in
                          # ln(syn), under DLNSYN_MAX (1e-3) -- _densify_nodes interpolates
                          # log-linearly in t and is a deliberate no-op for the hydro, so
                          # the base grid's error is a floor the refinement cannot remove.
NORAR_PERSIST = 3         # consecutive steep slope entries required to call it a
                          # crash. Guards the detector against SINGLE-ROW pressure
                          # jumps, which a look-ahead slope cannot tell from a
                          # collapse -- above all the early_ana='shockfit'
                          # prepended injection row. See rarefaction_handover.
NORAR_PRECURSOR_TOL = 0.005
                          # The crash detector uses a LOOK-AHEAD slope, so it fires before
                          # the collapse -- but not before the wave. A rarefaction is a FAN:
                          # its head arrives at the sound speed and p already sags below the
                          # smooth decay before the steep part. Measured on cooling_g100 RS,
                          # that sag at the detected handover runs 0.0006 (k=60) to 0.0195
                          # (k=519) in ln p, growing toward the CD (which the wave reaches
                          # last, so its precursor is best resolved). Rows the wave has
                          # already touched must not seed a NO-wave counterfactual, so the
                          # anchor is backed off to the last row whose p is within this
                          # tolerance of the smooth decay extrapolated from a clean interior
                          # window. 0.005 = 0.5% in p, 0.25% in B; below the smallest
                          # measured sag, so cells with no resolved precursor keep the
                          # detector's own index and nothing moves for them.
NORAR_Z = 4               # shell the 'prerar' alpha table is read for when the caller does
                          # not say; every shell-level entry point passes its own z, so this
                          # only backstops a direct single-cell call. 4 = reverse shock.
NORAR_LAW = 'prerar'      # the only surviving extension law (prerar_model). The
                          # free-coasting family ('bernoulli', 'adiab', 'adiab_frozen',
                          # 'bpl') was retired: it closed the tail on conservation laws
                          # alone, i.e. rho -> R^-2, which treats a shocked cell as a FREE
                          # fluid element. It is not -- it sits in a causally-connected
                          # shocked layer still compressing it toward the CD, so rho ~
                          # R^-1.2 (see prerar_model). Over the ~2.5 decades this
                          # counterfactual extrapolates that is a factor ~100 in density,
                          # and it roughly DOUBLED the inferred cost of the wave
                          # (full/no-rf 0.885 vs 0.761 at logr=+2, RS).


def _precursor_backoff(lnx, lnp, c, window, minpts, tol):
  '''
  Walk the handover back from the detected crash index c to the last row whose pressure
  is still within `tol` (in ln p) of the smooth decay -- i.e. before the rarefaction FAN's
  head, not merely before its steep part.

  The smooth decay is characterised by a straight fit to ln p vs ln R over a clean
  interior window ending well short of c, then extrapolated forward; the first row that
  falls `tol` BELOW that extrapolation is where the wave has demonstrably arrived.
  Returns c when there is no room for a clean window (the outer-edge cells, whose wave
  is on them at injection) or when no row departs, so this can only ever move the anchor
  EARLIER, never later.
  '''
  b = max(minpts, c - 2*window)          # end of the clean window: clear of the crash
  a = max(0, b - 4*window)               # ...and long enough to fix a slope
  if b - a < 5 or c <= b:
    return c
  sl, ic = np.polyfit(lnx[a:b], lnp[a:b], 1)
  sag = (sl*lnx[b:c+1] + ic) - lnp[b:c+1]     # > 0 where the data sits BELOW the decay
  hit = np.flatnonzero(sag > tol)
  return c if not hit.size else max(int(b + hit[0]), minpts)


def rarefaction_handover(shocked, slope_thresh=NORAR_SLOPE_THRESH,
    window=NORAR_SLOPE_WINDOW, minpts=NORAR_MINPTS, precursor_tol=NORAR_PRECURSOR_TOL,
    persist=NORAR_PERSIST):
  '''
  Index of the LAST row still on the smooth downstream decay, i.e. before the rarefaction
  reaches this cell; None if this history never crashes (nothing to replace).

  precursor_tol: back the anchor off the detected crash to before the fan's head
    (_precursor_backoff); None keeps the raw detector index. The counterfactual is seeded
    from this row, so a row the wave has already depressed would bias the whole synthetic
    tail low.

  Same detector as working_cooling.truncate_at_rarefaction (look-ahead log-slope of p
  over `window` rows, crash = slope < slope_thresh), so the handover lands exactly where
  that cell's BPL fit window ends -- but returning an index instead of a frame, and with
  the minpts guard relaxed (see NORAR_MINPTS). truncate_at_rarefaction keeps iloc[:steep[0]],
  so the last kept row is steep[0]-1.

  DO NOT reimplement this by calling truncate_at_rarefaction and taking len(): its
  minpts=20 silently returns the history untruncated for the outer-edge cells, which
  would make them look crash-free and hand back the reference unchanged.

  Rows with p <= 0 or a non-monotonic x (restart overlaps) are treated as a crash at that
  row: the log-slope is undefined there, and the smooth decay demonstrably ended.
  '''
  x = shocked.x.to_numpy(dtype=float)
  p = shocked.p.to_numpy(dtype=float)
  if len(x) < window + minpts + 1:
    return None
  # first row where the logs stop being defined; the slope array is built on [:n] only
  bad = np.flatnonzero(~(p > 0.) | np.insert(np.diff(x) <= 0., 0, False))
  if bad.size and bad[0] <= minpts:
    return None                      # degenerate from the start: nothing trustworthy to fit
  n = int(bad[0]) if bad.size else len(x)
  if n < window + minpts + 1:
    return max(n - 1, minpts - 1)
  lnx, lnp = np.log(x[:n]), np.log(p[:n])
  slope = (lnp[window:] - lnp[:-window])/(lnx[window:] - lnx[:-window])
  steep = np.flatnonzero(slope < slope_thresh)
  if persist > 1 and steep.size:
    # Require the steep slope to PERSIST. A single-row pressure jump makes exactly ONE
    # slope entry steep (the look-ahead window [j, j+window] contains that pair only for
    # j = the row before it), whereas the rarefaction collapse steepens a long run of
    # consecutive entries. Without this, early_ana='shockfit' breaks the detector: the
    # prepended shock-front row sits at a smaller radius and a much higher pressure than
    # the first measured row, so slope[0] plunges and the crash is "found" at index 0 --
    # measured on cooling_g100 RS, that dragged 30 of the 41 cells in k=440..480 from
    # R_h/R_inj ~ 2.7 down to ~1.0004, i.e. it replaced almost their entire history with
    # the synthetic tail. minpts alone cannot separate the two cases: the outer-edge cells
    # really are caught at injection (see NORAR_MINPTS), they just decline for many rows.
    runs = np.split(steep, np.flatnonzero(np.diff(steep) != 1) + 1)
    runs = [r for r in runs if r.size >= persist]
    steep = runs[0] if runs else np.array([], dtype=int)
  if not steep.size:
    # no steep slope, but the history may still have gone degenerate further out
    return (n - 1) if n < len(x) else None
  c = max(int(steep[0]) - 1, minpts - 1)
  if precursor_tol is not None:
    c = min(c, _precursor_backoff(lnx, lnp, c, window, minpts, precursor_tol))
  return c


def _adiabat_frozen(rho, rho_h, p_h):
  '''p along an adiabat with gma_ad FROZEN at the handover state: p = p_h (rho/rho_h)^gma.
  Kept for comparison -- see _adiabat_integrated for why it is not the default.'''
  return p_h * (rho/rho_h)**derive_adiab(rho_h, p_h)


def _adiabat_integrated(rho, rho_h, p_h):
  '''
  p along the Taub-Matthews adiabat with gma_ad EVOLVING as the gas cools:
      d ln p / d ln rho = gma_ad(T),   T = p/rho
  integrated from (rho_h, p_h) with a midpoint (RK2) step on the given rho grid.

  Why this and not a frozen index: the shocked gas here is only mildly relativistic
  (T = p/rho = 0.031..0.051 across cooling_g100's RS), so gma_ad(h) = 1.642..1.652, and it
  RISES toward 5/3 as the expansion cools the gas -- p therefore falls slightly faster
  than any single power of rho. The no-rarefaction simulation shows an effective index
  a_p/a_rho = 3.25/1.95 = 1.667, i.e. above every frozen value in the shell, so freezing
  systematically under-steepens the tail.

  RK2 suffices by a wide margin: the grid gives d ln rho = 0.072 per step
  (NORAR_PTS_PER_DEC) and gma_ad varies by <0.03 over the whole extension, so the local
  truncation error is ~1e-5 in ln p.
  '''
  lnr = np.log(np.asarray(rho, dtype=float)/rho_h)     # 0 at the handover, decreasing
  out = np.empty(lnr.shape)
  lp, lr = np.log(p_h), 0.
  for j in range(lnr.size):
    d = lnr[j] - lr
    g1 = derive_adiab(rho_h*np.exp(lr), np.exp(lp))
    g2 = derive_adiab(rho_h*np.exp(lr + 0.5*d), np.exp(lp + 0.5*d*g1))   # midpoint
    lp += d*g2
    lr = lnr[j]
    out[j] = lp
  return np.exp(out)








def _truncate_at_ratio(shocked, ratio):
  '''
  Keep the rows out to R = ratio*R_injection. x is monotonic along a worldline, so
  searchsorted is exact; a ratio the wave/window catches almost at once keeps the 2-row
  minimum the tt/tp quadrature needs. Factored out of generate_cell_fromHistory so the
  rarefaction cut (rar_ratio) and the analysis window (r_cap) land on the SAME snapshot
  row, not merely the same nominal radius.
  '''
  n = int(np.searchsorted(shocked.x.to_numpy(dtype=float),
                          ratio*float(shocked.x.iloc[0]), side='right'))
  return shocked.iloc[:max(n, 2)]


def data_method_name(law=None, cap=None):
  '''
  Sweep-method / cache-directory name of a data-path variant: 'data', 'data_norar',
  'data_norar_bpl', 'data_cap', 'data_norar_cap'. Single source of truth, imported by
  sweep_gammacm so method_outdir and _compute_point cannot drift apart.
  law: None (reference hydro) | 'prerar'; cap: None | R/R_inj window.
  'prerar' is the measured-reconstruction law (prerar_model).
  '''
  name = 'data'
  if law is not None:
    # the law is ALWAYS in the name, never abbreviated away for the current default.
    # Abbreviating it means flipping NORAR_LAW silently re-points 'data_norar' at a cache
    # holding a different physics -- the numbers change, the directory does not, and
    # nothing warns. Cost of being explicit: one longer directory name.
    name += f'_norar_{law}'
  if cap is not None:
    name += '_cap'
  return name




def shock_sd_block(sd, u, n_plateau=5):
  """
  The contiguous Sd run that actually straddles the shock, as (i0, i1), or None.

  Sd fires on more than the shock. The external interface launches a rarefaction into the
  shell's rear face at t=0, and the detector flags it too: on cooling_g100 z=4 cell k=20
  carries two runs -- rows 1-168 with |u_up - u_dn| = 0.2 (the rarefaction, no velocity
  jump at all) and rows 1471-1475 with 86.6 (the reverse shock). Both consumers of Sd used
  to take the FIRST run, which for that cell meant an injection at t = 23 s instead of
  4.04e4 (a factor 48 early) and a "post-shock" history starting at t = 1784 that is mostly
  unshocked, rarefying material.

  Selecting by velocity contrast is a no-op wherever there is one run, and also for the
  CD-adjacent cell k=519, whose two runs are 72.3 then 1.0 -- there the real shock IS the
  first. It changed exactly one cell of 500 on this run.
  """
  nz = np.flatnonzero(sd != 0)
  if not len(nz):
    return None
  runs = []
  s0 = p0 = int(nz[0])
  for aa in nz[1:]:
    if int(aa) == p0 + 1:
      p0 = int(aa)
    else:
      runs.append((s0, p0)); s0 = p0 = int(aa)
  runs.append((s0, p0))
  best, best_jump = runs[0], -1.
  for (rs, re) in runs:
    rlo, rhi = max(0, rs - n_plateau), min(len(u) - 1, re + n_plateau)
    a_up = float(np.median(u[rlo:rs])) if rs > rlo else float(u[rs])
    a_dn = float(np.median(u[re+1:rhi+1])) if rhi > re else float(u[re])
    jump = abs(a_up - a_dn)
    if np.isfinite(jump) and jump > best_jump:
      best, best_jump = (rs, re), jump
  return best


def _proper_velocity(vx):
  return vx/np.sqrt(np.clip(1. - vx*vx, 1e-300, None))


def postshock_start(sd, vx, n_settle=1):
  """
  First row of a cell's post-shock history: one past the Sd run that straddles the shock,
  plus n_settle. ONE definition, so a caller working from plain columns
  (IO.open_cellcolumns) and select_postshock_rows' DataFrame path cannot drift apart.
  Returns len(sd) where no run qualifies, i.e. an empty history.
  """
  blk = shock_sd_block(sd, _proper_velocity(np.asarray(vx, dtype=float)))
  return min(blk[1] + 1 + n_settle, len(sd)) if blk is not None else len(sd)


def select_postshock_rows(cell_data, n_settle=1):
  '''
  Rows of a cell history from the moment the cell is shocked (same selection as
  fit_celldata): cell files hold the full history from it=0 and pre-shock rows
  also have Sd == 0, so keep rows after the first Sd != 0 (shock crossing) block,
  + n_settle steps for numerical settling.
  Returns a (possibly empty) DataFrame; empty = cell never shocked in the data
  (legitimate for interrupted long runs), the caller skips it.
  '''
  sd = cell_data.Sd.to_numpy()
  ish = np.flatnonzero(sd != 0)
  if len(ish):
    start = postshock_start(sd, cell_data.vx.to_numpy(dtype=float), n_settle)
    return cell_data.iloc[start:].copy()
  # no crossing recorded: a raw full history (starts at it=0) means the shock
  # never reached this cell -> empty; otherwise assume pre-trimmed data
  if len(cell_data) and cell_data.index[0] == 0:
    return cell_data.iloc[0:0].copy()
  return cell_data.loc[(cell_data.Sd == 0)].copy().iloc[1:]

def _cumtrapz0(f, t):
  '''Cumulative trapezoid of f(t) with out[0] = 0.'''
  return np.insert(np.cumsum(0.5*(f[1:] + f[:-1])*np.diff(t)), 0, 0.)

def _densify_nodes(t, cols_log, cols_lin, rate, dlnsyn_max, nsub_max=NSUB_MAX):
  '''
  Refine the snapshot grid so the tt/tp quadrature resolves the intra-interval
  variation of the cooling rate, splitting each interval into
    n = clip(ceil(|Dln rate| / dlnsyn_max), 1, nsub_max)   sub-intervals.

  WHY. tt(t) is built by cumulative trapezoid over the nodes and inverted with
  np.interp, so it is piecewise LINEAR between snapshots: its slope inside an
  interval is the interval-AVERAGE of syn/lfac. The sub-step hydro, however, is
  interpolated log-linearly and the emission kernel evaluates Pmax ~ syn LOCALLY
  at each sub-step edge before multiplying by dtp. When many cooling sub-steps
  fall inside one snapshot interval (fast cooling: the whole burn-off can span
  ~1 interval), every step then pairs a local rate with an interval-averaged
  fluence -- a first-order bias of ~half the intra-interval variation of syn,
  with the sign of a decreasing syn, i.e. an OVER-count (it drove eps_rad =
  E_rad/E_inj above 1, up to 1.02 per cell at log10(gma_c/gma_m) = -5).
  The fit-based path is immune by construction: worldline_from_cooling
  integrates dt'/dtt = t_c1(R) and reads the hydro from the same fits at the
  same R, so its dtt and its local rate always agree.

  Interpolating each column with the SAME rule the sub-step evaluation uses
  (log-linear for cols_log, linear for cols_lin) makes densification a no-op for
  the hydro at any sub-step edge: it only refines the tt/tp quadrature, which is
  exactly the intent. `rate` (syn/lfac) sets the refinement; the caller
  recomputes syn from the interpolated rho, p rather than interpolating it.
  Returns (t_new, cols_log_new, cols_lin_new).
  '''
  with np.errstate(divide='ignore', invalid='ignore'):
    dln = np.abs(np.diff(np.log(np.where(rate > 0., rate, np.nan))))
  dln = np.nan_to_num(dln, nan=0., posinf=0.)          # syn -> 0 at the rarefaction crash
  n = np.clip(np.ceil(dln/dlnsyn_max), 1, nsub_max).astype(int)
  if (n == 1).all():
    return t, cols_log, cols_lin
  t_new = np.unique(np.concatenate(
      [np.linspace(t[i], t[i+1], n[i] + 1) for i in range(len(t) - 1)]))
  out_log = [np.exp(np.interp(t_new, t, np.log(v))) for v in cols_log]
  out_lin = [np.interp(t_new, t, v) for v in cols_lin]
  return t_new, out_log, out_lin

def _refine_tt_on_rho_data(tt_edges, tt_nodes, lnrho_nodes, dlnrho_max):
  '''
  Data analog of working_cooling._refine_tt_on_rho: insert extra tt edges so no
  interval spans more than dlnrho_max in ln(rho). Two differences from the fit
  version, both because raw data need not be monotonic within a coarse tt bin:
    - the interval budget is the TOTAL VARIATION of ln(rho) over the snapshot
      nodes it spans (the rarefaction cliff + post-cliff bounce can net out at
      the endpoints while spanning a large excursion); on smooth stretches TV
      reduces to |Delta ln rho|, so the DLNRHO_MAX convergence study carries over;
    - sub-edges are placed at EQUAL TV increments (mapped back to tt through the
      nodes) instead of uniformly in tt: through the cliff almost no tt accrues
      (syn ~ p -> 0), so uniform-in-tt sub-edges would all miss the crash.
  Refining tt only adds resolution (the synchrotron law and adiabatic product
  telescope), results converge as the grid is refined.
  '''
  TV_nodes = np.insert(np.cumsum(np.abs(np.diff(lnrho_nodes))), 0, 0.)
  TV_edges = np.interp(tt_edges, tt_nodes, TV_nodes)
  out = [tt_edges[0]]
  for j in range(len(tt_edges) - 1):
    a, b = tt_edges[j], tt_edges[j+1]
    n = max(1, int(np.ceil((TV_edges[j+1] - TV_edges[j]) / dlnrho_max)))
    if n > 1:
      tv_sub = np.linspace(TV_edges[j], TV_edges[j+1], n + 1)[1:-1]
      # flat-TV runs make the inverse map non-unique; clip into the interval,
      # duplicates are dropped below
      out.extend(np.clip(np.interp(tv_sub, TV_nodes, tt_nodes), a, b))
    out.append(b)
  out = np.asarray(out)
  return out[np.insert(np.diff(out) > 0., 0, True)]   # strictly increasing

def generate_cell_fromData(cell_data, env_in, u_scale=1., alpha=1., zeta=1.,
    r_ref=1.2, Tmax=None, dlnrho_max=DLNRHO_MAX, dlnsyn_max=DLNSYN_MAX,
    n_settle=1, sh_row=None, early_frac=0., rar_ratio=None, r_cap=None, norar=None,
    norar_fit=None, norar_ppd=NORAR_PTS_PER_DEC, norar_z=NORAR_Z):
  '''
  Build the cooling-binned cell DataFrame (same schema as
  generate_cell_withDistrib: one row per cooling sub-step, columns t, dt, tp,
  dtp, tt, dtt, i, x, dx, rho, vx, lfac, p, trac, gmin, gmax, bsyn) directly
  from the cell's extracted snapshot history, without any hydro fitting.
  Selection + collision-time offset, optional early-datapoint reconstruction
  (sh_row: this cell's shock-front state from load_shockfront_states, prepended
  by _prepend_shocked_row when the measured onset is late by more than
  early_frac in relative terms), then generate_cell_fromHistory (see its
  docstring for the method and the meaning of the shared keywords, rar_ratio /
  r_cap / norar / norar_fit / norar_ppd included).

  Returns (cell, env) or (False, env_in) when the cell has no usable post-shock
  history (never shocked, shocked at the last snapshot, or onset past Tobs_max).
  '''
  # lab times counted from collision = t at it=0 (openData_withtime convention);
  # raw cell files carry the full history so the offset is their first row
  t_off = cell_data.t.iloc[0] if (len(cell_data) and cell_data.index[0] == 0) else 0.

  shocked = select_postshock_rows(cell_data, n_settle)
  if len(shocked) < 2:
    return False, env_in
  shocked['t'] = shocked['t'] - t_off
  if sh_row is not None:
    shocked = _prepend_shocked_row(shocked, sh_row, env_in, early_frac)
  return generate_cell_fromHistory(shocked, cell_data.attrs, env_in,
      u_scale=u_scale, alpha=alpha, zeta=zeta, r_ref=r_ref, Tmax=Tmax,
      dlnrho_max=dlnrho_max, dlnsyn_max=dlnsyn_max, rar_ratio=rar_ratio,
      r_cap=r_cap, norar=norar, norar_fit=norar_fit, norar_ppd=norar_ppd,
      norar_z=norar_z)

def generate_cell_fromHistory(shocked, attrs, env_in, u_scale=1., alpha=1.,
    zeta=1., r_ref=1.2, Tmax=None, dlnrho_max=DLNRHO_MAX, dlnsyn_max=DLNSYN_MAX,
    rar_ratio=None, r_cap=None, norar=None, norar_fit=None,
    norar_ppd=NORAR_PTS_PER_DEC, norar_z=NORAR_Z):
  '''
  Core of generate_cell_fromData, on an already-selected post-shock history
  (first row = injection state, t counted from collision). Also consumed
  directly by the subcell_dlogT refinement of get_shell_nuFnu_fromData with
  synthetic sub-cell histories (_subcell_history).

  u_scale, alpha, zeta: same rescaling conventions (and composition order) as
    generate_cell_withDistrib; applied to the raw post-shock history via
    rescale_shocked_data / rescale_hydro_data. NB rescale_shocked_data scales
    every row's proper velocity by the single shocked-fluid ratio env.u/env_in.u
    -- the same approximation the fit path makes (it rescales cell_d0 and keeps
    the fitted shape), so the two methods stay comparable across a sweep.
  r_ref: cooling-bin ratio (gmax_j/gmax_j+1 per step) of the global gamma_max-
    geometric grid (generate_timebins), independent of the snapshot cadence:
    slow cooling merges many snapshots per step, fast cooling sub-divides them.
  Tmax: if set, stops the history once the onset time Ton reaches the observer
    window Tobs_max = Ts + Tmax*T0 (optional here -- data is finite, unlike the
    fit extrapolation -- it only avoids computing steps whose flux
    get_Fnu_cell_evolving would discard).
  dlnrho_max: cap on the ln(rho) total variation per cooling step
    (_refine_tt_on_rho_data); resolves the adiabatic evolution in slow cooling
    AND the rarefaction crash. None keeps the raw cooling-fluence bins.
  dlnsyn_max: cap on the variation of ln(syn/lfac) per node interval of the
    tt/tp quadrature (_densify_nodes). Keeps the fluence dtt consistent with the
    local cooling rate the emission kernel uses; without it the energy is
    over-counted in fast cooling (eps_rad > 1). None reproduces that behaviour.
  rar_ratio: opt-in SHARP rarefaction cut-off, R_rar/R_injection for this cell
    (the dimensionless per-cell ratio load_shell_rarefaction returns, consumed
    the way the fit path does at generate_cell_withDistrib:
    R_rar = ror*(cell_d0.x*c_)). The history is truncated at that radius, which
    caps tt_nodes[-1] and hence every downstream grid. None (default) is the
    method's normal behaviour: no cut, the cell is followed to its last
    snapshot. Being a ratio of radii it is alpha/zeta-invariant, so the cut is
    applied to the RAW history before any rescaling, and it carries over to the
    sub-cells of _subcell_history unchanged.
  r_cap: analysis window, R/R_injection, truncating the history the same way rar_ratio
    does. Kept SEPARATE from rar_ratio although mechanically identical: they mean
    different things (a modelled rarefaction cut-off vs a window both sides of a
    comparison share), and only r_cap belongs in a cache-directory name. When both are
    given the tighter one wins.
  norar: build the COUNTERFACTUAL history instead -- real rows to the rarefaction crash,
    then the smooth shocked-layer decay continued to the same final radius
    (prerar_model.prerar_history).
    None (default) is the method's normal behaviour and leaves this code path untouched.
    'prerar' is the only extension law; norar_fit is its alpha table (prerar_model.
    with_settle). norar_ppd is its sampling in points per decade of R.
    Applied AFTER the truncation above, so the counterfactual's target radius is whatever
    the reference's own endpoint is under the same rar_ratio/r_cap.

  Sub-step hydro is interpolated from the snapshots (log-linear in t for rho, p,
  dx, lfac -- exact at the nodes, positive through the rarefaction crash; linear
  in t for x, consistent with the Ton cap), completing what
  cooling_distribution.generate_cellDistrib_interpolated / split_hydrostep left
  open (they froze the hydro across sub-steps).

  Returns (cell, env) or (False, env_in).
  '''
  if len(shocked) < 2:
    return False, env_in
  # sharp rarefaction cut-off / analysis window, BEFORE any rescaling (both are radius
  # ratios, hence alpha/zeta-invariant): keep only the rows out to ratio*x_injection.
  # Cells the wave catches almost at once (outer edge, rar_ratio ~ 1.006) keep the 2-row
  # minimum the tt/tp quadrature needs; anything degenerate falls through the guard below.
  ratios = [r for r in (rar_ratio, r_cap) if r is not None and np.isfinite(r)]
  if ratios:
    shocked = _truncate_at_ratio(shocked, min(ratios))
  # counterfactual hydro, AFTER the truncation so the extension targets whatever endpoint
  # the reference has under the same window (prerar_history takes it from the frame)
  if norar is not None:
    # 'prerar' is the only law: follow the real rows out to the edge of the
    # rarefaction-free window, then prolong on the measured alpha tables -> the derived
    # causal-contact asymptote. Imported lazily so the module stays importable and the
    # forkserver workers stay cheap when no counterfactual is asked for.
    from prerar_model import prerar_history
    shocked, _ = prerar_history(shocked, norar_fit, z=norar_z, pts_per_dec=norar_ppd)
  # rescale env + data (s = u_scale holding a_u fixed, then Granot alpha/zeta),
  # same composition order as generate_cell_withDistrib
  env = rescale_proper_velocities(u_scale, env_in)
  if u_scale != 1.:
    shocked = rescale_shocked_data(shocked, env_in, env)
  if alpha != 1. or zeta != 1.:
    env = rescale_hydro(alpha, zeta, env)
    if alpha != 1.:
      shocked = rescale_hydro_data(shocked, alpha)

  # hoist to numpy once; no pandas below this point
  t    = shocked.t.to_numpy(dtype=float)
  x    = shocked.x.to_numpy(dtype=float)
  dx   = shocked.dx.to_numpy(dtype=float)
  rho  = shocked.rho.to_numpy(dtype=float)
  vx   = shocked.vx.to_numpy(dtype=float)
  p    = shocked.p.to_numpy(dtype=float)
  lfac = shocked.lfac.to_numpy(dtype=float) if 'lfac' in shocked \
         else 1./np.sqrt(1. - vx**2)
  # keep strictly increasing t (restart overlaps can duplicate/reorder rows)
  keep = np.insert(np.diff(t) > 0., 0, True)
  if not keep.all():
    t, x, dx, rho, vx, p, lfac = (a[keep] for a in (t, x, dx, rho, vx, p, lfac))
  if len(t) < 2:
    return False, env_in

  # refine the node grid until the cooling rate is resolved, so the tt/tp
  # quadrature below stays consistent with the LOCAL rate the emission kernel
  # evaluates at each sub-step (see _densify_nodes; no effect on the hydro
  # itself, only on the quadrature)
  syn = derive_syn_cooling(rho, p, env.rhoscale, env.eps_B)
  if dlnsyn_max is not None:
    t, (rho, p, dx, lfac), (x,) = _densify_nodes(
        t, (rho, p, dx, lfac), (x,), syn/lfac, dlnsyn_max)
    syn = derive_syn_cooling(rho, p, env.rhoscale, env.eps_B)   # from the interpolated hydro
    vx = np.sqrt(lfac**2 - 1.)/lfac

  # worldline cumulatives at the (refined) nodes (replaces worldline_from_cooling):
  # comoving time t' and cooling fluence tt = int dt'/t_c1, quadrature over the
  # measured hydro (generate_cellDistrib pattern, trapezoid, zero at injection)
  tp_nodes = _cumtrapz0(1./lfac, t)
  tt_nodes = _cumtrapz0(syn/lfac, t)

  # injection state: the cell's own first settled post-shock row
  inj = shocked.iloc[0]
  gmin0 = get_variable(inj, 'gma_m', env)
  gmax0 = get_variable(inj, 'gma_M', env)

  # global gamma_max-geometric grid, shared logic with the fit path
  tt_geo = generate_timebins(inj, env, None, 1., end_cond='gmax', r_ref=r_ref)

  # termination: end of data / gmax -> 1 / (optional) observer window, on nodes
  tt_end = min(tt_nodes[-1], tt_geo[-1])
  if Tmax is not None:
    Tobs_max = env.Ts + Tmax*env.T0
    Ton_nodes = (1. + env.z)*(t + env.t0 - x)     # x in light-seconds (c=1 code units)
    if Ton_nodes[0] >= Tobs_max:
      return False, env_in                        # whole history past the window
    if Ton_nodes[-1] > Tobs_max:                  # Ton monotonic: single crossing
      tt_end = min(tt_end, np.interp(Tobs_max, Ton_nodes, tt_nodes))
  if tt_end <= 0.:
    return False, env_in
  tt_edges = np.append(tt_geo[tt_geo < tt_end], tt_end)

  # refine on the ln(rho) total variation (adiabatic resolution + rarefaction crash)
  if dlnrho_max is not None:
    tt_edges = _refine_tt_on_rho_data(tt_edges, tt_nodes, np.log(rho), dlnrho_max)

  # tt -> t: exact inverse of the trapezoid model (tt piecewise-linear in t with
  # breakpoints at the nodes); drop plateau duplicates (syn ~ 0 stretches)
  t_edges = np.interp(tt_edges, tt_nodes, t)
  keep = np.insert(np.diff(t_edges) > 0., 0, True)
  t_edges, tt_edges = t_edges[keep], tt_edges[keep]
  if len(t_edges) < 2:
    return False, env_in
  tp_edges = np.interp(t_edges, t, tp_nodes)

  # hydro at the sub-step edges: log-linear in t (exact at nodes, positive
  # through the rarefaction crash); x linear in t (consistent with the Ton cap)
  def _loginterp(v):
    return np.exp(np.interp(t_edges, t, np.log(v)))
  rho_e, p_e, dx_e, lfac_e = (_loginterp(v) for v in (rho, p, dx, lfac))
  x_e  = np.interp(t_edges, t, x)
  vx_e = np.sqrt(lfac_e**2 - 1.)/lfac_e

  # electron distribution bounds along the actual (interpolated) rho history
  gmin_edges, gmax_edges, bsyn_edges, Aad_edges = evolve_gma_bounds_edges(
      tt_edges, rho_e, gmin0, gmax0)

  # assemble (left edges + diffs, as generate_cell_withDistrib; t is already
  # the since-collision lab time, no t0 to add)
  n1 = len(t_edges) - 1
  dic = {'t': t_edges[:-1], 'dt': np.diff(t_edges),
         'tp': tp_edges[:-1], 'dtp': np.diff(tp_edges),
         'tt': tt_edges[:-1], 'dtt': np.diff(tt_edges),
         'i': np.full(n1, inj.i), 'x': x_e[:-1], 'dx': dx_e[:-1],
         'rho': rho_e[:-1], 'vx': vx_e[:-1], 'lfac': lfac_e[:-1], 'p': p_e[:-1],
         'trac': np.full(n1, inj.trac),
         'gmin': gmin_edges[:-1], 'gmax': gmax_edges[:-1], 'bsyn': bsyn_edges[:-1],
         'Aad': Aad_edges[:-1]}
  out = pd.DataFrame.from_dict(dic)
  for key in attrs:
    out.attrs[key] = attrs[key]
  return out, env

def cell_crossing_time(u_up, u_dn, w, fastshell):
  """
  How long the shock takes to cross ONE cell, from that cell's own plateaus and width.

  Everything here is per-cell and measured: u_up and u_dn are the proper velocities either
  side of the cell's own Sd block, w is its width just before it is shocked, and the shock's
  lab velocity follows from the JUMP CONDITIONS on those two plateaus
  (derive_betaRS/derive_betaFS, fed the relative proper velocity of the two states). No
  neighbouring cell and no fitted worldline enters, which is the point: the alternatives
  either assume the crossing equals the spacing between neighbouring events, or take the
  shock speed from a fit that is known to drift against the cells.

    T = w / |beta_up - beta_sh|          (w in lt-s -> T in seconds)

  The closing speed is a near-cancellation -- |beta_up - beta_sh| ~ 2.7e-5 here -- so it is
  computed from the jump conditions rather than by differencing two velocities near 1.
  Returns nan if the plateaus do not admit a shock.
  """
  if not (np.isfinite(u_up) and np.isfinite(u_dn) and np.isfinite(w)) or w <= 0.:
    return float('nan')
  lfac_up = derive_Lorentz_from_proper(u_up)
  lfac_dn = derive_Lorentz_from_proper(u_dn)
  lfac_rel = derive_relatLfac(lfac_up, lfac_dn)
  if not np.isfinite(lfac_rel) or lfac_rel < 1.:
    return float('nan')
  u_rel = derive_proper_from_Lorentz(lfac_rel)
  beta_sh = (derive_betaRS(u_up, u_rel, u_dn) if fastshell
             else derive_betaFS(u_up, u_rel, u_dn))
  dv = abs(u_up/lfac_up - beta_sh)
  if not np.isfinite(dv) or dv <= 0.:
    return float('nan')
  return w/dv


def measured_injection_event(cell_data, env, z, estimator='midpoint', n_plateau=5,
    at='edge'):
  """
  (t_inj, x_inj) at which the shock front crosses THIS cell, from the cell's own history
  and at SUB-CADENCE resolution. Returns (nan, nan) if the cell never gets shocked.

  WHY NOT THE Sd FLAG, AND WHY NOT A FITTED WORLDLINE.
  - The shock detector flags a CONTIGUOUS BLOCK OF 3-4 CELLS that straddles the front
    (measured on cooling_g100_hires z=4, constant from it=2000 to 4e5: 3-4 cells, radial
    span 0.9-1.7e-4 lt-s, proper velocity running 199 -> 127 across it). A cell's FIRST
    Sd firing is therefore 3-4 cell crossings before the front reaches it, and
    select_postshock_rows' first row is that much after -- 102-136 iterations at the
    measured 34 iterations per cell crossing. Either edge biases every cell.
  - The fitted worldline puts the cells on a CONSTANT crossing rate; measured, the rate
    varies 26% across the shell, so fitted and measured onsets cross and
    _prepend_shocked_row's guard flips at each crossing.

  THE ASYMPTOTES ARE READ PER CELL, NOT FROM env. The shocked proper velocity is NOT
  constant: it drifts 126.7 -> 133.6 along the propagation (5.4%, and the same 126-133.5
  the field-average work measured along a worldline). Anchoring on env.u would be right at
  the CD and ~5.6% low by the back of the shell -- a MONOTONIC error across the shell, so
  it would tilt the whole onset ladder rather than scatter it, which is precisely the class
  of defect this function exists to avoid. u_up and u_dn are therefore the plateaus either
  side of the cell's own Sd block.

  estimator: 'midpoint' -- u crosses (u_up + u_dn)/2. For a sharp front sweeping a
    finite-volume cell, the cell average is halfway across when the front is at the cell
    CENTRE, so this estimates 'front at this cell's position'. Its weakness is that the
    numerical shock is 3-4 cells wide, so the transition is smeared and its SHAPE enters.

  THE MIDPOINT IS NOT THE CELL'S ONSET, so `at` decides what comes back. Everything
  downstream wants the LEADING edge -- when the cell starts being shocked -- and at='edge'
  (the default) returns it, by backing off half of THIS cell's own crossing time
  (cell_crossing_time) and re-reading the cell's position from its own worldline there.
  at='centre' returns the raw midpoint crossing, for inspection.

  WHY PER-CELL AND NOT FROM THE NEIGHBOURS. The boundary between two cells can also be had
  as the midpoint of two adjacent centre crossings, and that is smooth (it averages two
  measurements rather than differencing them). But it mixes the two cells' RADII, so the
  observer time it implies carries a geometric -dr/c term belonging to neither cell: the
  same event then reads at a different fraction of its crossing in lab time (0.31) and in
  bar{T} (0.51), with no way to say which is meant. Backing off half of the cell's own
  crossing keeps everything on one worldline, and the fraction is 1/2 by the estimator's
  own definition -- the front is at the cell centre when the cell average is halfway
  between the plateaus.
  estimator: 'inflection' -- the extremum of du/dt, refined parabolically. The centre of
    the transition profile, independent of where the asymptotes sit.
  """
  # a DataFrame (any caller) or the plain-array mapping IO.open_cellcolumns returns. The
  # array form exists because building the DataFrame costs 213x the read at hi-res and
  # every column but these five is then discarded -- see open_cellcolumns.
  if hasattr(cell_data, 'columns'):
    col = lambda c: cell_data[c].to_numpy(dtype=float)
  else:
    col = lambda c: np.asarray(cell_data[c], dtype=float)
  t, x, vx, dxa, sd = (col(c) for c in ('t', 'x', 'vx', 'dx', 'Sd'))
  u = vx/np.sqrt(np.clip(1. - vx*vx, 1e-300, None))
  nz = np.flatnonzero(sd != 0)
  if not len(nz) or nz[0] == 0:
    return float('nan'), float('nan')
  # the run that straddles the shock, not simply the first to fire (shock_sd_block)
  blk = shock_sd_block(sd, u, n_plateau)
  if blk is None:
    return float('nan'), float('nan')
  i0, i1 = blk
  lo = max(0, i0 - n_plateau)
  hi = min(len(u) - 1, i1 + n_plateau)
  if hi - lo < 3:
    return float('nan'), float('nan')
  u_up = float(np.median(u[lo:i0])) if i0 > lo else float(u[i0])
  u_dn = float(np.median(u[i1+1:hi+1])) if hi > i1 else float(u[i1])
  if not (np.isfinite(u_up) and np.isfinite(u_dn)) or abs(u_up - u_dn) < 1e-9:
    return float('nan'), float('nan')

  def _interp(i, j, frac):
    f = float(np.clip(frac, 0., 1.))
    return t[i] + f*(t[j] - t[i]), x[i] + f*(x[j] - x[i])

  if estimator == 'inflection':
    seg = slice(lo, hi + 1)
    du = np.gradient(u[seg], t[seg])
    m = int(np.argmax(np.abs(du)))
    # parabolic refinement on |du/dt| about its extremum
    frac = 0.
    if 0 < m < len(du) - 1:
      y0, y1, y2 = np.abs(du[m-1]), np.abs(du[m]), np.abs(du[m+1])
      den = y0 - 2.*y1 + y2
      if abs(den) > 1e-30:
        frac = float(np.clip(0.5*(y0 - y2)/den, -0.5, 0.5))
    i = lo + m
    j = i + 1 if frac >= 0. else i - 1
    j = int(np.clip(j, 0, len(t) - 1))
    return _interp(i, j, abs(frac))

  if estimator == 'midpoint':
    u_mid = 0.5*(u_up + u_dn)
    crossed = (u <= u_mid) if u_up > u_dn else (u >= u_mid)
    jj = np.flatnonzero(crossed[lo:hi+1])
    if not len(jj) or (lo + int(jj[0])) == 0:
      return float('nan'), float('nan')
    j = lo + int(jj[0]); i = j - 1
    du = u[i] - u[j]
    t_c, x_c = _interp(i, j, 0.5 if abs(du) < 1e-12 else (u[i] - u_mid)/du)
  else:
    raise ValueError(f"unknown estimator {estimator!r}")

  if at == 'centre':
    return t_c, x_c
  # LEADING EDGE: back off half of THIS cell's own crossing, then re-read the cell's
  # position from its own worldline. Both halves are per-cell, so no neighbour's radius
  # leaks into the event -- which is what makes the observer time unambiguous (see the
  # docstring). w is taken just ahead of the Sd block, where the cell is still unshocked.
  w = float(np.median(dxa[lo:i0])) if i0 > lo else float(dxa[i0])
  T = cell_crossing_time(u_up, u_dn, w, fastshell=(z == 4))
  if not np.isfinite(T):
    return t_c, x_c
  t_e = t_c - 0.5*T
  if t_e <= t[0]:
    return float(t[0]), float(x[0])       # the CD-adjacent cell: shocked from collision
  return float(t_e), float(np.interp(t_e, t, x))


def load_shockfront_states(key, z, env, source='shockfit', t_max_fac=3., nproc=1):
  '''
  Per-cell shocked-state table for the early-datapoint reconstruction
  (columns [t, i, x, dx, rho, vx, lfac, p, vx_u, trac], code units, t since
  collision -- the reconstruct_data schema). The states follow the
  radius-dependent shock evolution (spherical: the shock Lorentz factor and
  strength change past R0), which the planar Riemann solution misses.
    source='shockfit': cellsBehindShock_fromData on this run's shock-front
      history (run_data_{z}.csv) -- most accurate, and exactly the states the
      fit-based pipeline anchors on.
    source='au': fits_from_au(env.a_u) -- lfac(x), ShSt(x) profiles
      interpolated in log10(a_u-1) from the peak-modeling sweep tables
      (extracted_data/fullsweep_au_{RS,FS}.csv), no fit of this run needed.
      Table popts are used un-normalized (reconstruct_data convention).
      t_max = t_max_fac * planar crossing time (must cover the actual,
      slower spherical crossing; only early cells are consumed anyway).
  '''
  fastshell = (z == 4)
  if source == 'shockfit':
    data = open_rundata(key, z)
    if data is False or not len(data):
      raise FileNotFoundError(
          f'run_data_{z}.csv missing/empty for {key}: extract it first with '
          'analysis_hydro.extract_data_thinshell')
    return cellsBehindShock_fromData(data)
  elif source == 'au':
    log_aum = np.log10(env.a_u - 1.)
    if not (-1. <= log_aum <= 1.5):
      print(f'load_shockfront_states: log10(a_u-1)={log_aum:.2f} outside the '
            'calibrated range [-1, 1.5], table edge values used')
    popt_lfac, popt_ShSt = fits_from_au(env.a_u, 'RS' if fastshell else 'FS')[:2]
    t_max = t_max_fac * (env.tRS if fastshell else env.tFS)
    return cellsBehindShock_fromFit(key, popt_lfac, popt_ShSt, t_max,
                                    fastshell=fastshell)
  elif source == 'measured':
    return measured_shockfront_states(key, z, env, nproc=nproc)
  raise ValueError(
      f"early_ana source must be 'shockfit', 'au' or 'measured', got {source!r}")


def _repair_front_order(ev, key, z):
  """
  Reject measured injection events that break the front's monotonicity along the cell
  ordering, and replace them by interpolation from the neighbours that survive.

  The shock reaches spatially ordered cells in order, so t must increase along `ev`.
  Where it does not, the velocity-jump measurement has latched onto something that is not
  the shock -- on cooling_g100 z=4 that is the shell's REARMOST cell (k=20), which borders
  the external medium and starts expanding at t=0: it reads 0.908 in bar{T} where its
  neighbours k=21,22 sit at 1.353, 1.348. That one cell IS the centre ladder's worst local
  gap (ratio 3.754 at bar{T}=0.907); it predates the leading-edge construction and was
  simply carried through, self-consistently misplaced.

  The events are per-cell, so a bad one corrupts only itself -- but it still lands out of
  order along the front, and the ladder must stay monotone.

  A rejected event is replaced by a linear interpolation in cell index between the nearest
  good neighbours on each side, or -- past the last good one -- extrapolated with the
  median local spacing. Never silent: the count and the cells are printed.
  """
  t = np.array([e[1] for e in ev], dtype=float)
  x = np.array([e[2] for e in ev], dtype=float)
  good = np.ones(len(ev), dtype=bool)
  run = -np.inf
  for j in range(len(ev)):
    if t[j] <= run:
      good[j] = False
    else:
      run = t[j]
  if good.all():
    return ev
  j_ok = np.flatnonzero(good)
  bad = np.flatnonzero(~good)
  # interpolate in cell index; np.interp clamps past the ends, so fix those up with the
  # median spacing of the good run rather than repeating its last value
  t_new, x_new = np.interp(bad, j_ok, t[j_ok]), np.interp(bad, j_ok, x[j_ok])
  dt, dx_ = np.median(np.diff(t[j_ok])), np.median(np.diff(x[j_ok]))
  for m, j in enumerate(bad):
    if j > j_ok[-1]:
      t_new[m], x_new[m] = t[j_ok[-1]] + (j - j_ok[-1])*dt, x[j_ok[-1]] + (j - j_ok[-1])*dx_
    elif j < j_ok[0]:
      t_new[m], x_new[m] = t[j_ok[0]] - (j_ok[0] - j)*dt, x[j_ok[0]] - (j_ok[0] - j)*dx_
  print(f'measured_shockfront_states on {key} z={z}: {len(bad)} injection event(s) out of '
        f'order along the front, replaced by neighbour interpolation (cells '
        f'{[ev[j][0] for j in bad]})')
  out = list(ev)
  for m, j in enumerate(bad):
    k, _, _, init_r = ev[j]
    out[j] = (k, float(t_new[m]), float(x_new[m]), init_r)
  return out


def _shockfront_cache_path(key, z):
  return get_dirpath(key) + f'shockfront_measured_z={z}.npz'


def _read_shockfront_cache(path, key):
  """The cached table, or None if it is not the current version."""
  try:
    with np.load(path, allow_pickle=False) as d:
      if int(d['version']) != _SHOCKFRONT_CACHE_VERSION:
        print(f'shock-front cache {os.path.basename(path)} is version {int(d["version"])}, '
              f'not {_SHOCKFRONT_CACHE_VERSION}; rebuilding')
        return None
      cols = [str(c) for c in d['cols']]
      out = pd.DataFrame({c: d[c] for c in cols}, columns=cols)
  except Exception as e:
    print(f'shock-front cache {os.path.basename(path)} unreadable ({e}); rebuilding')
    return None
  out.attrs['key'] = key
  return out


def _write_shockfront_cache(path, out):
  np.savez(path, version=_SHOCKFRONT_CACHE_VERSION,
           cols=np.array([str(c) for c in out.columns]),
           **{c: out[c].to_numpy() for c in out.columns})


_EVENT_COLS = ('t', 'x', 'dx', 'vx', 'Sd')
_SHOCKFRONT_CACHE_VERSION = 1


def _event_chunk(ks):
  '''Injection events of a run of cells; module-level so it is picklable.'''
  key, z, env = _EVENT_CTX.key, _EVENT_CTX.z, _EVENT_CTX.env
  out = []
  for k in ks:
    cols = open_cellcolumns(key, int(k), _EVENT_COLS)
    if cols is None or not len(cols['t']):
      continue
    t_inj, x_inj = measured_injection_event(cols, env, z)
    if not (np.isfinite(t_inj) and np.isfinite(x_inj)):
      continue
    out.append((int(k), float(t_inj), float(x_inj), float(cols['x'][0])*c_))
  return out


_EVENT_CTX = None

def _event_init(ctx):
  global _EVENT_CTX
  _EVENT_CTX = ctx


def _measure_events(key, z, env, ks, nproc=1):
  '''
  (k, t, x, init_r) for every cell of the shell, in shock order.

  Reads the FIVE columns the estimator needs rather than the whole history: building the
  DataFrame, not the I/O, is the cost, and at hi-res it is 213x the read (open_cellcolumns).
  That alone takes a 10000-cell shell from ~10 h to ~3 min; the pool is on top of it.
  '''
  ks = [int(k) for k in ks]
  if nproc is not None and nproc > 1 and len(ks) > 1:
    import concurrent.futures as cf
    nchunk = min(len(ks), max(1, int(nproc)*4))
    bounds = np.array_split(np.asarray(ks), nchunk)
    ctx = SimpleNamespace(key=key, z=z, env=env)
    with cf.ProcessPoolExecutor(max_workers=int(nproc), mp_context=cell_pool.pool_context(),
                                initializer=_event_init, initargs=(ctx,)) as ex:
      parts = list(ex.map(_event_chunk, [b for b in bounds if len(b)]))
    return [e for part in parts for e in part]
  _event_init(SimpleNamespace(key=key, z=z, env=env))
  return _event_chunk(ks)


def measured_shockfront_states(key, z, env, nproc=1, use_cache=True):
  """
  Per-cell shocked-state table with the injection EVENT measured and the injection STATE
  from the fitted hydro -- the two halves the other sources conflate.

  event: measured_injection_event, i.e. where each cell's own velocity jump puts the
    front, sub-cadence, read from the same snapshots as the cell histories.
  state: fits_hydro.state_at_radius, the fitted lfac(x) and shock-strength(x) profiles
    evaluated at THAT radius. Those fits are smooth functions of R/R0 and that is the one
    job they do well; what they must not also supply is the worldline.

  Why this exists: 'shockfit' derives both from one fitted worldline laid against cells at
  uniform initial radii, which fixes the cell-crossing RATE. Measured, the rate varies 26%
  across a hi-res shell, so the fitted and measured onsets cross, _prepend_shocked_row's
  `sh_row.t < first.t` guard flips at each crossing, and the shell splits into corrected
  and uncorrected blocks with a step between them. Each ladder is smooth ON ITS OWN
  (checked on cooling_g100_hires z=4: worst local gap 1.00 fitted, 1.00-1.02 measured);
  the defect is only ever in the MIXTURE. One convention throughout removes it, and the
  measured one is the convention that cannot drift against the cell data.

  Same columns and code units as cellsBehindShock_fromData, so it drops into
  load_shockfront_states unchanged.
  """
  from fits_hydro import get_hydrofits_shell_new, state_at_radius
  data = open_rundata(key, z)
  if data is False or not len(data):
    raise FileNotFoundError(
        f'run_data_{z}.csv missing/empty for {key}: extract it first with '
        'analysis_hydro.extract_data_thinshell')
  popt_lfac, popt_ShSt = get_hydrofits_shell_new(data)[:2]
  fastshell = (z == 4)
  path = _shockfront_cache_path(key, z)
  if use_cache and os.path.isfile(path):
    out = _read_shockfront_cache(path, key)
    if out is not None:
      return out
  kmin = env.Next + (0 if fastshell else env.Nsh4)
  kmax = kmin + (env.Nsh4 if fastshell else env.Nsh1)
  # cells in the order the shock reaches them: outward from the contact discontinuity,
  # which for the fast shell is DESCENDING k (the driver's klist does the same flip).
  # _repair_front_order checks monotonicity along the front, so it must run in this order.
  ks = list(range(int(kmin), int(kmax)))
  if fastshell:
    ks = ks[::-1]
  ev = _measure_events(key, z, env, ks, nproc)
  if not ev:
    raise RuntimeError(f'no measurable injection event in shell z={z} of {key}')
  ev = _repair_front_order(ev, key, z)

  rows = []
  for j, (k, t_e, x_e, init_r) in enumerate(ev):
    R_hit = x_e*c_                        # lt-s -> cm, the state_at_radius convention
    x, rho, vx, lfac, p, dx = state_at_radius(R_hit, init_r, env, fastshell,
                                              popt_lfac, popt_ShSt)
    rows.append(dict(t=t_e, i=k, x=x_e, dx=dx*env.R0/c_,
                     rho=rho/env.rhoscale, vx=vx, lfac=lfac,
                     p=p/(env.rhoscale*c_**2),
                     vx_u=(env.beta4 if fastshell else env.beta1),
                     trac=(1. if fastshell else 2.)))
  out = pd.DataFrame(rows)
  out.attrs['key'] = key
  if use_cache:
    _write_shockfront_cache(path, out)
  return out

def _prepend_shocked_row(shocked, sh_row, env, early_frac):
  '''
  Prepend the shock-front state (sh_row, from load_shockfront_states) to a
  cell's selected post-shock history: the first settled snapshot row lags the true
  crossing by the settle + cadence delay, which dominates the early lightcurve. The
  prepended row becomes the injection state (gma_m/gma_M/K0 anchored exactly as the
  fit pipeline); the log-linear-in-t interpolation of generate_cell_fromHistory
  bridges the gap to the first measured row. The only guard is t_sh < t_meas.

  early_frac skips the prepend where the relative onset error
  (barT_meas - barT_sh) <= early_frac*barT_sh. It DEFAULTS TO 0, i.e. correct every
  cell, because any threshold puts a treatment boundary in the middle of the shell
  and that boundary is visible in the lightcurve. Measured on cooling_g100 z=4 at
  log10(gc/gm) = -5: the historical early_frac=0.1 corrected only the first 20 cells
  and left a kink at bar{T}/bar{T}_f = 0.044 (|d slope| 10.755 against a background
  of 1.450, ratio 7.4). At the threshold the skipped correction is still ~10% of
  barT, so it switches off as a finite jump, not a fade. early_frac = 0 (or anything
  <= 0.02, which is identical) gives ratio 2.6-2.7 AND improves the pre-peak wiggle
  by 21% (0.192% -> 0.151% of peak). Dropping the prepend altogether is worse than
  both (ratio 3.7), so the cure is to apply it uniformly, not to remove it.
  NB the earlier suspects for that kink -- compute_subcell_edges' ceil() spacing
  step, which lands at the same bar{T} by coincidence -- were tested and cleared.
  '''
  first = shocked.iloc[0]
  if sh_row.t >= first.t:
    return shocked
  barT_meas = (get_variable(first, 'Ton', env) - env.Ts)/env.T0
  barT_sh = (get_variable(sh_row, 'Ton', env) - env.Ts)/env.T0
  if (barT_meas - barT_sh) <= early_frac * max(barT_sh, 0.):
    return shocked
  row = first.copy()
  for col in ('t', 'x', 'dx', 'rho', 'vx', 'p'):
    row[col] = sh_row[col]
  if 'vx_u' in sh_row.index and 'vx_u' in row.index:
    row['vx_u'] = sh_row['vx_u']
  if 'Sd' in row.index:
    row['Sd'] = 0.
  return pd.concat([row.to_frame().T, shocked])

def _subcell_history(shocked_par, sub):
  '''
  Synthetic post-shock history of an interpolated sub-cell, for the
  subcell_dlogT early-lightcurve refinement: the parent cell's actual history
  normalized to its injection state (row 0) and rescaled to the sub-cell's
  interpolated injection state `sub` (from _interp_state: t, x, vx, rho, p
  interpolated between onset-adjacent cells, dx the flux-conserving weight).
  Data analog of the fit path reusing the parent's fitted profile for its
  sub-cells: the history SHAPE is the parent's, only the anchor state moves.
    q(tau)   = q_sub0 * q_par(tau)/q_par0        (rho, p, lfac, dx)
    t(tau)   = t_sub0 + (t_par - t_par0)         (worldline shift)
    x(tau)   = x_sub0 + (x_par - x_par0)
  vx recomputed from the scaled lfac (consistency); trac, i, Sd kept from the
  parent (as _interp_state keeps the labels of its first argument). The
  rarefaction-cliff timing is inherited from the parent, shifted to the
  sub-onset (mirrors the fit path using the parent's R_rar/R_inj).
  '''
  par0 = shocked_par.iloc[0]
  out = shocked_par.copy()
  for col in ('rho', 'p', 'dx'):
    out[col] = shocked_par[col] * (sub[col]/par0[col])
  lfac_par = shocked_par['lfac'].to_numpy() if 'lfac' in shocked_par \
             else 1./np.sqrt(1. - shocked_par['vx'].to_numpy()**2)
  lfac_sub0 = 1./np.sqrt(1. - sub['vx']**2)
  lfac = lfac_par * (lfac_sub0/lfac_par[0])
  out['lfac'] = lfac
  out['vx'] = np.sqrt(lfac**2 - 1.)/lfac
  out['t'] = sub['t'] + (shocked_par['t'] - par0['t'])
  out['x'] = sub['x'] + (shocked_par['x'] - par0['x'])
  return out

def get_cell_nuFnu_fromData(key, k, func_Fnu=get_Fnu_cell_evolving,
    u_scale=1., alpha=1., zeta=1., r_ref=1.2,
    Tmax=5, NT=500, lognu_min=-2.5, lognu_max=2.5, Nnu=400,
    return_cell=False, dlnrho_max=DLNRHO_MAX, dlnsyn_max=DLNSYN_MAX, n_settle=1,
    early_ana=None, early_frac=0., rar_cut=None, norar=None, r_cap=None,
    norar_ppd=NORAR_PTS_PER_DEC, **kwargs):
  '''
  The whole chain to obtain nu Fnu of one cell from its actual snapshot history
  (data-driven counterpart of get_cell_nuFnu; same observer-grid and rescaling
  conventions, same return signature).
  early_ana: None | 'shockfit' | 'au' -- reconstruct the cadence-missed early
    datapoint from the shock-front states (load_shockfront_states /
    _prepend_shocked_row) when the measured onset is late by more than
    early_frac (relative).
  '''
  nub, T, env = obs_arrays(key, normed=True, Tmax=Tmax, NT=NT,
      lognu_min=lognu_min, lognu_max=lognu_max, Nnu=Nnu)
  cell_data = open_celldata(key, k)
  if cell_data is False:
    extract_data_cells(key, [k], noOut=True)
    cell_data = open_celldata(key, k)
  z = 4 if (k < env.Next + env.Nsh4) else 1
  sh_row = None
  if early_ana is not None:
    sh_data = load_shockfront_states(key, z, env, source=early_ana)
    sel = sh_data.loc[sh_data.i == k]
    if len(sel):
      sh_row = sel.iloc[0]
  # rar_cut / norar / r_cap: see get_shell_nuFnu_fromData (same treatments, one cell)
  rar_ratio = None
  if rar_cut == 'model':
    rar_ratio = rar_map_lookup(load_shell_rarefaction(key, z, env), k)
  elif rar_cut is not None:
    raise ValueError(f"rar_cut must be None or 'model', got {rar_cut!r}")
  if norar is not None and rar_cut is not None:
    raise ValueError('norar and rar_cut are alternative treatments of the same wave; '
                     'set at most one')
  norar_fit = None
  if norar == 'prerar':
    from prerar_model import load_table, with_settle, TABLE_KEY, NCELLS
    norar_fit = with_settle(load_table(TABLE_KEY, z), key, z, NCELLS)
  cell_dist, env = generate_cell_fromData(cell_data, env,
      u_scale=u_scale, alpha=alpha, zeta=zeta, r_ref=r_ref, Tmax=Tmax,
      dlnrho_max=dlnrho_max, dlnsyn_max=dlnsyn_max, n_settle=n_settle,
      sh_row=sh_row, early_frac=early_frac, rar_ratio=rar_ratio,
      r_cap=r_cap, norar=norar, norar_fit=norar_fit, norar_z=z, norar_ppd=norar_ppd)
  # scale the observer grids with the (possibly rescaled) env (matches obs_arrays)
  nuobs = nub * env.nu0
  Tobs = env.Ts + (T - 1) * env.T0
  if cell_dist is False:
    return (nuobs, Tobs, env, None, cell_dist) if return_cell else (nuobs, Tobs, env, None)
  nuFnu = get_nuFnu(func_Fnu, nuobs, Tobs, cell_dist, env, **kwargs)
  if return_cell:
    return nuobs, Tobs, env, nuFnu, cell_dist
  return nuobs, Tobs, env, nuFnu

# =======================================================================================
# Cell-level parallelism of the shell pass
# =======================================================================================
# get_shell_nuFnu_fromData sums independent per-cell contributions into one (T, nu) grid,
# so the shell loop is a map-reduce -- but it used to be written as one serial loop over a
# list holding EVERY cell's history at once. Two things broke at 20x resolution:
#   - memory: 1e4 cells x ~9e4 rows of history in the parent is ~200 GB. The sweep OOMs
#     before it is slow.
#   - cores: the only parallelism was across the 8 sweep points (sweep_gammacm.run_sweep),
#     so an HPC allocation past 8 cores did nothing.
# The fix is the same for both: workers own a contiguous chunk of the flat EMITTER list
# (cells and sub-cells), read the cells it touches themselves, accumulate locally, and hand
# back one (T, nu) array. The parent holds only a bounded window of partial sums, and the
# only whole-shell arrays it keeps are the injection rows (scan_shell_cells).
#
# DETERMINISM. Chunk boundaries are a function of the problem alone (EMITTERS_PER_CHUNK and
# the per-emitter cost weights), never of the worker count, and the parent reduces chunks in
# INDEX order. So 8 workers and 128 workers give bit-identical answers -- an HPC result can
# be reproduced on a laptop. Against the historical serial reduction the result differs
# only by float associativity (measured 1.7e-15 relative on a production point);
# ncell_proc=1 walks the emitters in the historical order and is bit-identical to it.

EMITTERS_PER_CHUNK = 16
'''Target emitters per pool task. Sets the number of chunks, hence the granularity of
load balancing and the size of the reduction. It is deliberately NOT derived from the
worker count: that is what makes the answer core-count independent.

A chunk is a contiguous run of the flat EMITTER list, not of cells -- one cell's
sub-cells may be split across chunks. That distinction is the whole point: sub-refinement
is concentrated on the few CD-adjacent cells (54 of 500 carry all 372 sub-cells at
SUBCELL_DLOGT=0.008, up to SUBCELL_MAX=400 each), so with cells as the atom a single cell
outweighs the mean chunk 5-18x and caps the shell pass no matter how many cores are
thrown at it -- measured makespan/ideal 1.74 at 7 workers and 31.9 at 128. Splitting
inside a cell costs only a re-read of that cell's history in the second chunk.'''

_SCAN_CACHE_VERSION = 2   # 2: early_ana='measured' onsets are the cells' LEADING EDGES
                          # (measured_shockfront_states, half a measured crossing back), not the
                          # velocity-jump centres. The cache path carries the early_ana
                          # NAME but not its semantics, so a v1 'measured' scan would be
                          # reused unchanged under the new convention -- bump, don't trust.
_SCAN_CELLS_PER_CHUNK = 64      # scan is I/O bound, so keep chunks small enough to fill a
                                # large pool (10k cells -> ~157 tasks)


def _load_cell_history(key, k, n_settle, env, sh_data=None, early_frac=0.):
  '''
  One cell's post-shock history, with the early reconstruction applied: exactly the
  per-cell body of what used to be get_shell_nuFnu_fromData's first pass.
  Returns (hist, attrs, n_prepended), hist=None where the cell has no usable history.

  ONE definition, used by both scan_shell_cells and the emission workers, so the
  injection row the sub-cell edges are built from cannot drift from the injection row
  the emission actually uses.
  '''
  cell_data = open_celldata(key, k)
  if cell_data is False:
    return None, None, 0
  t_off = cell_data.t.iloc[0] if (len(cell_data) and cell_data.index[0] == 0) else 0.
  shocked = select_postshock_rows(cell_data, n_settle)
  if len(shocked) < 2:
    return None, None, 0
  shocked['t'] = shocked['t'] - t_off
  n_prepended = 0
  if sh_data is not None:
    sel = sh_data.loc[sh_data.i == k]
    if len(sel):
      n0 = len(shocked)
      shocked = _prepend_shocked_row(shocked, sel.iloc[0], env, early_frac)
      n_prepended = len(shocked) - n0
  return shocked, cell_data.attrs, n_prepended


def _scan_cache_path(key, z, n_settle, early_ana, early_frac):
  return (get_dirpath(key) +
          f'cellscan_z={z}_ns={n_settle}_ea={early_ana}_ef={early_frac:g}.npz')


_SCAN_CTX = None

def _scan_init(ctx):
  global _SCAN_CTX
  _SCAN_CTX = ctx

def _scan_chunk(spec):
  '''Scan one contiguous run of cells; module-level so it is picklable.'''
  cid, i0, i1 = spec
  c = _SCAN_CTX
  out = []
  for idx in range(i0, i1):
    k = int(c.klist[idx])
    hist, _, npre = _load_cell_history(k=k, key=c.key, n_settle=c.n_settle, env=c.env,
                                       sh_data=c.sh_data, early_frac=c.early_frac)
    if hist is None:
      out.append((idx, False, 0, None, np.nan, 0))
      continue
    inj = hist.iloc[0]
    barT = (get_variable(inj, 'Ton', c.env) - c.env.Ts)/c.env.T0
    out.append((idx, True, len(hist), inj, barT, npre))
  return cid, out


def scan_shell_cells(key, z, klist, env, n_settle=1, early_ana=None, early_frac=0.,
    sh_data=None, nproc=1, use_cache=True):
  '''
  Per-cell injection rows, usability, onset bar{T} and cost weights for a whole shell --
  everything the shell pass needs from the cells BEFORE any flux is computed, and nothing
  that scales with history length.

  This is alpha-independent: the injection row comes from the raw history, and bar{T}_on
  is built with the UNRESCALED env (the dimensionless convention compute_subcell_edges
  works in). All 8 sweep points therefore share one scan, where the old first pass redid
  it per point -- and per point it also held every history in memory, which is the thing
  that does not fit at hi-res.

  Cached to results/{key}/cellscan_z=...npz, versioned and coverage-checked like the
  rarefaction head. Callers MUST have run extract_data_cells first: scanning a partially
  extracted run would freeze a cache made of whatever subset happened to be on disk (the
  same trap documented for load_shell_rarefaction).

  use_cache: read AND write the shared cache. Callers pass False for a NON-default
    klist: the cache path is keyed on (key, z) alone, so a subset scan would overwrite
    the whole shell's cache with a partial one. The stored cell list is checked on read
    too, so a stale cache is rebuilt rather than trusted.

  Returns dict(k, usable, nrows, barT_on, inj, n_prepended).
  '''
  klist = np.asarray(klist)
  path = _scan_cache_path(key, z, n_settle, early_ana, early_frac)
  if use_cache and os.path.isfile(path):
    with np.load(path, allow_pickle=False) as d:
      if int(d['version']) == _SCAN_CACHE_VERSION and np.array_equal(d['k'], klist):
        cols = [str(c) for c in d['inj_cols']]
        return dict(k=d['k'], usable=d['usable'], nrows=d['nrows'],
                    barT_on=d['barT_on'], n_prepended=int(d['n_prepended']),
                    inj=_inj_frame(d['inj'], cols, key))
      print(f'cell scan cache {os.path.basename(path)} does not match this shell '
            '(version or cell list); rebuilding')

  n = len(klist)
  bounds = _chunk_bounds(np.ones(n), max(1, int(np.ceil(n/_SCAN_CELLS_PER_CHUNK))))
  specs = [(c, i0, i1) for c, (i0, i1) in enumerate(bounds)]
  ctx = SimpleNamespace(key=key, klist=klist, n_settle=n_settle, env=env,
                        sh_data=sh_data, early_frac=early_frac)
  rows = [None]*n
  if nproc > 1 and len(specs) > 1:
    npr = cell_pool.resolve_nproc(nproc, cap=len(specs))
    with cell_pool.cell_executor(npr, initializer=_scan_init, initargs=(ctx,)) as ex:
      for _, out in ex.map(_scan_chunk, specs):
        for r in out:
          rows[r[0]] = r
  else:
    _scan_init(ctx)
    for spec in specs:
      for r in _scan_chunk(spec)[1]:
        rows[r[0]] = r

  usable = np.array([r[1] for r in rows], dtype=bool)
  nrows = np.array([r[2] for r in rows], dtype=np.int64)
  barT_on = np.array([r[4] for r in rows], dtype=float)
  n_prepended = int(sum(r[5] for r in rows))
  first = next((r[3] for r in rows if r[3] is not None), None)
  if first is None:
    raise RuntimeError(f'no usable cell history in shell z={z} of {key}: nothing to scan')
  cols = [str(c) for c in first.index]
  inj = np.full((n, len(cols)), np.nan)
  for i, r in enumerate(rows):
    if r[3] is not None:
      inj[i] = r[3].to_numpy(dtype=float)
  if use_cache:
    np.savez(path, version=_SCAN_CACHE_VERSION, k=klist, usable=usable, nrows=nrows,
             barT_on=barT_on, n_prepended=n_prepended, inj=inj,
             inj_cols=np.array(cols))
  return dict(k=klist, usable=usable, nrows=nrows, barT_on=barT_on,
              n_prepended=n_prepended, inj=_inj_frame(inj, cols, key))


def _inj_frame(values, cols, key):
  '''The scan's injection rows as a frame carrying the run attributes open_celldata puts
  on a cell history, so a row taken out of it is interchangeable with the shocked.iloc[0]
  the shell pass used to carry around.'''
  df = pd.DataFrame(values, columns=cols)
  mode, runname, rhoNorm, geometry = get_runatts(key)
  df.attrs.update(key=key, mode=mode, runname=runname, rhoNorm=rhoNorm,
                  geometry=geometry)
  return df


def _chunk_bounds(weights, nchunks):
  '''
  Split a weighted sequence into <= nchunks CONTIGUOUS [i0, i1) runs of roughly equal
  cumulative weight. Depends only on (weights, nchunks) -- never on the worker count,
  which is what keeps the reduction reproducible across machines.
  '''
  w = np.asarray(weights, dtype=float)
  n = w.size
  nchunks = max(1, min(int(nchunks), n))
  if nchunks == 1:
    return [(0, n)]
  cw = np.cumsum(w)
  tot = cw[-1]
  if tot <= 0.:
    edges = np.linspace(0, n, nchunks + 1).astype(int)
  else:
    cuts = np.searchsorted(cw, tot*np.arange(1, nchunks)/nchunks, side='left') + 1
    edges = np.concatenate([[0], cuts, [n]])
  edges = np.unique(np.clip(edges, 0, n))
  return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def _new_acc(ctx):
  '''One accumulator set per variant, in `variants` order (None where the flux is not
  being computed at all, so it cannot be mistaken for a zero lightcurve).'''
  nv = len(ctx.variants)
  return SimpleNamespace(
      nuFnu=[None]*nv if ctx.energies_only
            else [np.zeros((len(ctx.Tobs), len(ctx.nuobs))) for _ in range(nv)],
      E_rad=[0.]*nv, E_int=[0.]*nv, E_inj=[0.]*nv, skipped=[])


def _merge_acc(dst, src):
  '''Reduce one chunk's partial sums into the running total. Called in chunk-index
  order, which is what makes the total independent of completion order.'''
  for iv in range(len(dst.E_rad)):
    if dst.nuFnu[iv] is not None:
      dst.nuFnu[iv] += src.nuFnu[iv]
    dst.E_rad[iv] += src.E_rad[iv]
    dst.E_int[iv] += src.E_int[iv]
    dst.E_inj[iv] += src.E_inj[iv]
  dst.skipped.extend(src.skipped)


def _accum_energy(ctx, acc, iv, cell, cell_env):
  acc.E_rad[iv] += cell_radiated_energy(cell, cell_env)
  acc.E_inj[iv] += cell_injected_energy(cell, cell_env)
  c0 = cell.iloc[0]
  acc.E_int[iv] += get_variable(c0, 'ei', cell_env) * get_variable(c0, 'V3p', cell_env)


def _make_cell(ctx, hist, attrs, kk):
  '''Build one cell per variant from a shared history; None where it is unusable.'''
  out = []
  for _, rmap, kw in ctx.variants:
    c, ce = generate_cell_fromHistory(hist, attrs, ctx.env,
        u_scale=ctx.u_scale, alpha=ctx.alpha, zeta=ctx.zeta, r_ref=ctx.r_ref,
        Tmax=ctx.Tmax, dlnrho_max=ctx.dlnrho_max, dlnsyn_max=ctx.dlnsyn_max,
        rar_ratio=(rar_map_lookup(rmap, kk) if rmap is not None else None), **kw)
    out.append((c, ce))
  return out


def _emit(ctx, acc, cells):
  '''Accumulate one cell's flux (+ energies) into every variant. With two variants
  the pair evaluator computes their shared leading steps ONCE -- the whole point of
  rar_cut='both' -- and is bit-identical to evaluating them separately.'''
  if ctx.energies_only:
    ok = False
    for iv, (cell, cell_env) in enumerate(cells):
      if cell is False:
        continue
      _accum_energy(ctx, acc, iv, cell, cell_env)
      ok = True
    return ok
  if ctx.paired and cells[0][0] is not False and cells[1][0] is not False \
      and ctx.func_Fnu is get_Fnu_cell_evolving:
    (cf, ef), (cc, _) = cells
    Ff, Fc = get_Fnu_cell_evolving_pair(ctx.nuobs, ctx.Tobs, cf, cc, ef, **ctx.kwargs)
    # nu F_nu exactly as get_nuFnu does it (same nu0 branch, same broadcast), so the
    # paired path differs from the separate one in nothing but evaluation order
    norm = ctx.kwargs.get('norm', True)
    for iv, F in enumerate((Ff, Fc)):
      cell_iv, env_iv = cells[iv]
      nu0 = env_iv.nu0FS if (cell_iv.iloc[0].trac > 1.5) else env_iv.nu0
      nub_iv = (ctx.nuobs/nu0 if norm else ctx.nuobs)[np.newaxis, :]
      acc.nuFnu[iv] += nub_iv * F
      if ctx.return_energies: _accum_energy(ctx, acc, iv, cell_iv, env_iv)
    return True
  ok = False
  for iv, (cell, cell_env) in enumerate(cells):
    if cell is False:
      continue
    acc.nuFnu[iv] += get_nuFnu(ctx.func_Fnu, ctx.nuobs, ctx.Tobs, cell, cell_env,
                               **ctx.kwargs)
    if ctx.return_energies: _accum_energy(ctx, acc, iv, cell, cell_env)
    ok = True
  return ok


def _process_subcells(ctx, acc, idx, hist, attrs, j0, j1):
  '''
  Sub-cells [j0, j1) of cell idx: its onset interval split into flux-conserving
  sub-cells at geometrically-spaced onsets, reusing its actual history shape.

  The range exists so a chunk boundary can fall INSIDE a heavily refined cell -- each
  sub-cell is built from (inj_par, inj_next, edges) alone, so any sub-range is
  computable on its own.
  '''
  a, b, edges = ctx.sub_edges[idx]
  inj_par = ctx.inj.iloc[idx]
  inj_next = ctx.inj.iloc[idx+1]
  for j in range(j0, j1):
    e0, e1 = edges[j], edges[j+1]
    onset_c = np.sqrt(e0*e1)                       # geometric centre
    f = float(np.clip((onset_c - a)/(b - a), 0., 1.))
    dx_w = inj_par.dx * (e1 - e0)/(b - a)          # onset-width weight => flux conserved
    sub_st = _interp_state(inj_par, inj_next, f, dx_w)
    # place the sub-cell onset exactly at onset_c: shift the anchor state
    # along its worldline (dt, dx = beta*dt => dbarT = (1+z)(1-beta)dt/T0).
    # Backward shifts (onset_c below the measured onsets) extrapolate the
    # parent's injection state to where the snapshot cadence could not see --
    # the data analog of the fit path's sub-states interpolated on the
    # reconstructed shock front (which reaches barT = 0 at collision).
    barT_sub = (get_variable(sub_st, 'Ton', ctx.env) - ctx.env.Ts)/ctx.env.T0
    beta = sub_st['vx']
    delta = (onset_c - barT_sub)*ctx.env.T0/((1. + ctx.env.z)*(1. - beta))
    sub_st['t'] += delta
    sub_st['x'] += beta*delta
    hist_sub = _subcell_history(hist, sub_st)
    # sub-cells look the map up on their own INTERPOLATED index, as the fit path
    # does -- R_rar/R_injection varies smoothly across the shell
    _emit(ctx, acc, _make_cell(ctx, hist_sub, attrs, sub_st['i']))


_EMIT_CTX = None

def _emit_init(ctx):
  global _EMIT_CTX
  _EMIT_CTX = ctx

def _emit_chunk(spec):
  '''
  One contiguous run [e0, e1) of the flat emitter list: read the cells it touches, emit
  them, return the partial sums. Module-level so it is picklable; the read-only shell
  context arrives once per worker through the pool initializer, so the per-task payload
  is three ints.

  Emitters of the same cell are adjacent in the list, so walking them in runs reads each
  cell's history exactly once per chunk -- a chunk boundary landing inside a refined cell
  costs one extra read of that cell, nothing more.
  '''
  cid, e0, e1 = spec
  ctx = _EMIT_CTX
  acc = _new_acc(ctx)
  em = ctx.emitters
  i = e0
  while i < e1:
    idx = em[i][0]
    j = i
    while j < e1 and em[j][0] == idx:      # the run of this cell's emitters in the chunk
      j += 1
    hist, attrs, _ = _load_cell_history(key=ctx.key, k=int(ctx.klist[idx]),
                                        n_settle=ctx.n_settle, env=ctx.env,
                                        sh_data=ctx.sh_data, early_frac=ctx.early_frac)
    if hist is None:
      acc.skipped.append(ctx.klist[idx])
    elif em[i][1] < 0:                     # unsplit cell: one emitter, the cell itself
      if not _emit(ctx, acc, _make_cell(ctx, hist, attrs, ctx.klist[idx])):
        acc.skipped.append(ctx.klist[idx])
    else:
      _process_subcells(ctx, acc, idx, hist, attrs, em[i][1], em[j-1][1] + 1)
    i = j
  return cid, acc


def _run_cell_loop(ctx, weights, ncell_proc):
  '''
  Map the shell's cells over chunks and reduce them in index order.

  Serial (ncell_proc None/1) walks every cell in klist order into one accumulator --
  the historical evaluation order, bit for bit.

  Parallel keeps at most 2*nproc partial (T, nu) arrays alive in the parent by
  submitting through a bounded window and consuming the OLDEST future each time: the
  reduction stays in chunk order (reproducible) without buffering every chunk (a
  (2000, 650) float64 partial is ~10 MB, and there are hundreds of chunks).
  '''
  ne = len(ctx.emitters)
  acc = _new_acc(ctx)
  if ncell_proc in (None, 1):
    _emit_init(ctx)
    _merge_acc(acc, _emit_chunk((0, 0, ne))[1])
    return acc

  nchunks = max(1, int(np.ceil(ne / EMITTERS_PER_CHUNK)))
  specs = [(c, i0, i1) for c, (i0, i1) in enumerate(_chunk_bounds(weights, nchunks))]
  npr = cell_pool.resolve_nproc(ncell_proc, cap=len(specs))
  if npr == 1:
    _emit_init(ctx)
    for spec in specs:
      _merge_acc(acc, _emit_chunk(spec)[1])
    return acc
  print(f'  shell pass: {len(ctx.klist)} cells / {ne} emitters -> {len(specs)} chunks '
        f'on {npr} workers')
  window = 2*npr
  with cell_pool.cell_executor(npr, initializer=_emit_init, initargs=(ctx,)) as ex:
    pending, nxt = collections.deque(), 0
    for out in range(len(specs)):
      while nxt < len(specs) and len(pending) < window:
        pending.append(ex.submit(_emit_chunk, specs[nxt])); nxt += 1
      cid, part = pending.popleft().result()
      assert cid == out, f'chunk {cid} out of order (expected {out})'
      _merge_acc(acc, part)
      del part
  return acc


def get_shell_nuFnu_fromData(key, z, u_scale=1., alpha=1., zeta=1., klist=None,
    VFC=False, r_ref=1.2, Tmax=5, NT=500, lognu_min=-2.5, lognu_max=2.5, Nnu=400,
    dlnrho_max=DLNRHO_MAX, dlnsyn_max=DLNSYN_MAX, Tb_min=None, Tb_lin=None, n_settle=1,
    subcell_dlogT=None, subcell_max=32, early_ana=None, early_frac=0.,
    rar_cut=None, norar=None, r_cap=None, ncell_proc=None,
    return_energies=False, energies_only=False, **kwargs):
  '''
  Total nu F_nu from shell z of simulation key, summed over its cells, from the
  actual snapshot histories (data-driven counterpart of get_shell_nuFnu; same
  observer-grid, rescaling, klist and return conventions).
  Dropped vs the fit version: cleanData/analytic (fit-bound).
  rar_cut: None (default) -- the method's normal behaviour, no rarefaction
    machinery: the wave is in the data and every cell is followed to its last
    snapshot. 'both' computes None AND 'model' in a single pass, sharing the
    cooling steps the two have in common (get_Fnu_cell_evolving_pair) instead of
    walking the shell twice; the return is then a DICT keyed by sweep-method name
    ('data', 'data_rarcut') in place of the single nuFnu / energy triple, and the
    values are bit-identical to the two separate calls. 'model' opts IN to the fit
    path's sharp cut-off alone, truncating each
    cell at the modelled catch-up radius from the shared shell head
    (load_shell_rarefaction -> {cell: R_rar/R_injection}, looked up per cell and
    per sub-cell with rar_map_lookup, exactly as get_shell_nuFnu does). Use it to
    separate the two things that differ between the methods -- the hydro
    treatment and the post-rarefaction treatment -- rather than to model
    anything: with rar_cut='model' the remaining fit/data gap is hydro alone.
  norar: build the COUNTERFACTUAL shell instead -- every cell followed to the same final
    radius as the reference, but with the rarefaction crash replaced by the smooth
    shocked-layer decay past the handover (prerar_model.prerar_history). Mutually
    exclusive with rar_cut: both are treatments of the same wave, and where rar_cut stops
    the cell early, this one holds the worldline EXTENT fixed and removes only the crash
    -- which is what isolates the wave's effect on the emission. None (default) leaves
    this code path untouched. 'prerar' is the only law; 'both' computes the
    reference AND the 'prerar' counterfactual in one pass, sharing their common leading
    cooling steps (get_Fnu_cell_evolving_pair) and returning a DICT keyed by sweep-method
    name, exactly as rar_cut='both' does.
    NB the shared prefix is SMALLER here than for rar_cut='both' (measured 86 of 433 steps
    on k=250 vs almost everything): the two sides differ from the handover onward, not
    from the cut onward. Do not quote that 1.28x speed-up for this path.
  r_cap: analysis window in R/R_injection applied to BOTH sides (the same truncation
    rar_ratio performs, hence the same snapshot row, not merely the same nominal radius).
    Used to bound how far the counterfactual's extension has to reach: at r_cap the
    reference is truncated too, so the endpoints still match exactly. None follows every
    cell to its last snapshot, as the method normally does. It participates in the cache
    name (data_method_name) because it changes the numbers.
  early_ana: None | 'shockfit' | 'au' -- reconstruct the cadence-missed early
    datapoints: each early cell's history is prepended with its shock-front
    state (load_shockfront_states: per-run fit 'shockfit', or a_u sweep-table
    profiles 'au') when its measured onset is late by more than early_frac
    (relative). Fixes both the onset ladder (the settle + cadence delay,
    ~2-3 snapshots) and the early injection states, so early-lightcurve
    comparisons with the fit method reflect the hydro treatment, not the
    snapshot cadence.
  subcell_dlogT: same early-lightcurve staircase smoothing as the fit driver
    (shared compute_subcell_edges/_interp_state): CD-adjacent cells whose onset
    interval spans more than subcell_dlogT in log10(bar{T}) are split into
    flux-conserving sub-cells at geometric onsets; each sub-cell reuses the
    PARENT's actual history shape rescaled to its interpolated injection state
    (_subcell_history -- the data analog of reusing the parent's fitted
    profile). None keeps the raw one-cell-per-cell sum.
  ncell_proc: workers for the CELL loop (None/1 = serial). The shell sum is a
    map-reduce over cells, so this is the parallelism that scales -- the sweep drivers'
    own pool is capped at the number of sweep points (8), which wastes an HPC
    allocation. Chunking is keyed on EMITTERS_PER_CHUNK, never on the worker count, so
    any two worker counts give BIT-IDENTICAL results; against the historical serial
    reduction the difference is float associativity alone (~1e-15 relative), and
    ncell_proc=1 reproduces the old order exactly. Never combine with a point-level
    pool: nested pools multiply the forkserver cost and the peak memory (see cell_pool).
  return_energies: as get_shell_nuFnu -- returns (nuobs, Tobs, env_rs,
    nuFnu_shell, E_rad, E_int, E_inj) with eps_rad = E_rad/E_inj.
  energies_only: skip the flux kernel entirely and return the energy budget
    alone, with None in the nuFnu slot (implies return_energies). The budget is
    comoving and per-cell -- cell_radiated_energy/cell_injected_energy never see
    nuobs/Tobs -- so the energies are bit-identical to the full run and the
    frequency grid is unused (callers may pass a token Nnu). Everything that
    DOES set the budget must still match the run it is compared with: Tmax
    (history window), r_ref, dlnrho_max/dlnsyn_max (step resolution), rar_cut,
    and NT/Tb_min/Tb_lin + subcell_* (the obs grid still sets the sub-cell
    floor). Used by sweep_efficiency, which needs eps_rad on a grid far finer
    than a flux sweep can afford (~3.5x cheaper per point here).
  '''
  return_energies = return_energies or energies_only
  nub, T, env = obs_arrays(key, normed=True, Tmax=Tmax, NT=NT,
      lognu_min=lognu_min, lognu_max=lognu_max, Nnu=Nnu, Tb_min=Tb_min, Tb_lin=Tb_lin)
  env_rs = rescale_proper_velocities(u_scale, env) if u_scale != 1. else env
  if alpha != 1. or zeta != 1.:
    env_rs = rescale_hydro(alpha, zeta, env_rs)
  nuobs = nub * env_rs.nu0
  Tobs = env_rs.Ts + (T - 1) * env_rs.T0
  func_Fnu = get_Fnu_cell_instant if VFC else get_Fnu_cell_evolving

  # cell id range of shell z; extract cells missing from disk
  k4 = env.Next
  kCD = k4 + env.Nsh4
  k1 = kCD + env.Nsh1
  kmin, kmax = (k4, kCD) if (z==4) else (kCD, k1)
  klist_default = klist is None       # only the whole shell may use the shared scan cache
  if klist is None:
    klist = np.arange(kmin, kmax)
    if z==4: klist = np.flip(klist)
  done = check_extracted_cells(key)
  todo = [k for k in klist if k not in done]
  if todo:
    extract_data_cells(key, todo, noOut=True)

  # opt-in sharp cut-off: single shared rarefaction head -> {cell: R_rar/R_injection}
  # for the whole shell, alpha-invariant so it is built once per (key, z). AFTER the
  # extraction above, for the same reason the fit driver does it there: the head skips
  # cells with no data on disk, so building it first would freeze (and cache) a head
  # made of whatever subset happened to be extracted.
  # `variants` drives the whole accumulation below: one (name, rar_map, kw) triple per
  # treatment being computed, kw going straight to generate_cell_fromHistory. A scalar
  # rar_cut/norar gives exactly one, and every return value keeps its historical shape;
  # 'both' gives two and returns them keyed by sweep-method name (see the docstring).
  if norar is not None and rar_cut is not None:
    raise ValueError('norar and rar_cut are alternative treatments of the same wave; '
                     'set at most one')
  if norar is not None:
    laws = [None, NORAR_LAW] if norar == 'both' else [norar]
    if any(l not in (None, 'prerar') for l in laws):
      raise ValueError(f"norar must be None, 'prerar' or 'both', got {norar!r}")
    # 'prerar' needs the alpha table, which is SHARED by every cell and carries this run's
    # own settling correction.
    fits = None
    if 'prerar' in laws:
      from prerar_model import load_table, with_settle, TABLE_KEY, NCELLS
      fits = with_settle(load_table(TABLE_KEY, z), key, z, NCELLS)
    variants = [(data_method_name(l, r_cap), None,
                 dict(r_cap=r_cap, norar=l, norar_fit=fits, norar_z=z)) for l in laws]
  elif rar_cut is None:
    variants = [(data_method_name(None, r_cap), None, dict(r_cap=r_cap))]
  elif rar_cut == 'model':
    variants = [('data_rarcut',
                 load_shell_rarefaction(key, z, env, n_shell=len(klist),
                                        nproc=(ncell_proc or 1)), {})]
  elif rar_cut == 'both':
    variants = [('data', None, {}),
                ('data_rarcut',
                 load_shell_rarefaction(key, z, env, n_shell=len(klist),
                                        nproc=(ncell_proc or 1)), {})]
  else:
    raise ValueError(f"rar_cut must be None, 'model' or 'both', got {rar_cut!r}")
  paired = len(variants) == 2

  # shock-front state table for the early-datapoint reconstruction, once per shell
  sh_data = load_shockfront_states(key, z, env, source=early_ana, nproc=(ncell_proc or 1)) \
            if early_ana is not None else None

  # first pass: injection rows + usability + cost weights (klist = onset order), so the
  # sub-cell edges can be computed from the cells' own onsets. The early reconstruction
  # (prepend) happens here, BEFORE rescaling, so the onset ladder and injection states
  # already carry it. The histories themselves are NOT kept: they are re-read per chunk
  # in the cell loop below, which is what bounds the parent's memory at hi-res.
  scan = scan_shell_cells(key, z, klist, env, n_settle=n_settle, early_ana=early_ana,
      early_frac=early_frac, sh_data=sh_data, nproc=(ncell_proc or 1),
      use_cache=klist_default)
  if early_ana is not None:
    print(f'get_shell_nuFnu_fromData early reconstruction ({early_ana}): '
          f"prepended shock-front states to {scan['n_prepended']} cells")

  # adaptive sub-cell onset edges (shared with the fit driver); onsets from the
  # cells' own injection rows, unrescaled env (dimensionless bar{T} convention)
  barT_grid = T - 1.
  floor = barT_grid[barT_grid > 0.].min()   # earliest resolved bar{T} on the obs grid
  sub_edges = [None]*len(klist)
  if subcell_dlogT is not None:
    # measured onsets are late by the settle + snapshot-cadence delay (the first
    # settled row is 1+ snapshots after the crossing), so they cannot reach below
    # the cadence; the CD-adjacent first cell is physically shocked at collision
    # (barT ~ 0, as the fit's reconstructed sh_data states). anchor_first spans its
    # sub-cells down to the grid floor, reconstructing the cadence-missed early rise
    # (their onsets are then placed exactly by the worldline shift in the sub-cell
    # loop below). It anchors the EDGES and leaves the onset interval alone -- this
    # used to set barT_on[first] = 0, which also stretched the window that cell's dx
    # is spread over and put a step in the temporal index at barT_on[1] under
    # early_ana='measured'. See compute_subcell_edges.
    sub_edges = compute_subcell_edges(scan['barT_on'], floor, subcell_dlogT,
                                      subcell_max, anchor_first=True)

  # flat emitter list, in the historical evaluation order: one entry per unsplit cell
  # (j = -1), else one per sub-cell. This is the unit of work AND the unit of chunking --
  # see EMITTERS_PER_CHUNK for why cells are too coarse an atom. The weight of an emitter
  # is its cell's post-shock row count, the best cheap proxy for its cooling-step count
  # (every sub-cell re-walks the parent's history shape, so they weigh the same).
  # Unusable cells keep an entry so the worker still reports them as skipped.
  emitters, weights = [], []
  for idx in range(len(klist)):
    w = float(max(int(scan['nrows'][idx]), 1))
    s = sub_edges[idx] if scan['usable'][idx] else None
    if s is None:
      emitters.append((idx, -1)); weights.append(w)
    else:
      for j in range(len(s[2]) - 1):
        emitters.append((idx, j)); weights.append(w)
  ctx = SimpleNamespace(
      key=key, klist=np.asarray(klist), env=env, nuobs=nuobs, Tobs=Tobs,
      variants=variants, paired=paired, func_Fnu=func_Fnu,
      u_scale=u_scale, alpha=alpha, zeta=zeta, r_ref=r_ref, Tmax=Tmax,
      dlnrho_max=dlnrho_max, dlnsyn_max=dlnsyn_max, n_settle=n_settle,
      early_frac=early_frac, sh_data=sh_data, energies_only=energies_only,
      return_energies=return_energies, kwargs=kwargs, sub_edges=sub_edges,
      inj=scan['inj'], emitters=emitters)
  acc = _run_cell_loop(ctx, np.asarray(weights), ncell_proc)
  nuFnu_v, E_rad_v, E_int_v, E_inj_v = acc.nuFnu, acc.E_rad, acc.E_int, acc.E_inj

  if acc.skipped:
    # a cell can only be reported once: its emitters are adjacent, and only the run that
    # fails to load (or the single unsplit emitter) records it
    print(f'get_shell_nuFnu_fromData on sim {key}: skipped {len(acc.skipped)} cells '
          f'(no data or no usable post-shock history): {acc.skipped}')
  if subcell_dlogT is not None:
    kr = [int(klist[i]) for i in range(len(klist)) if sub_edges[i] is not None]
    # from sub_edges directly, not accumulated: a refined cell may be split across
    # chunks, and this is a property of the ladder rather than of the walk
    n_refined = sum(len(s[2]) - 2 for s in sub_edges if s is not None)
    print(f'get_shell_nuFnu_fromData subcell refinement: +{n_refined} sub-cells over '
          f'{len(kr)} cells' + (f' (k={min(kr)}..{max(kr)})' if kr else ''))

  if paired:
    # {method name: nuFnu} or {method name: (nuFnu, E_rad, E_int, E_inj)} -- the caller
    # asked for two treatments, so it cannot get the flat single-variant tuple
    out = {nm: ((nuFnu_v[iv], E_rad_v[iv], E_int_v[iv], E_inj_v[iv]) if return_energies
                else nuFnu_v[iv]) for iv, (nm, _, _) in enumerate(variants)}
    return nuobs, Tobs, env_rs, out
  if return_energies:
    return nuobs, Tobs, env_rs, nuFnu_v[0], E_rad_v[0], E_int_v[0], E_inj_v[0]
  return nuobs, Tobs, env_rs, nuFnu_v[0]
