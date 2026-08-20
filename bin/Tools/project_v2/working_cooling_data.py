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
    which is what isolates the wave itself, sweep_norar);
  - intended for long-term simulations and setups where the BPL fit ansatz breaks.

Kept identical (imported, not duplicated): the geometric-in-gamma_max time binning
(generate_timebins), the electron-bound operator split (evolve_gma_bounds_edges),
the emission kernels (get_Fnu_cell_evolving / get_nuFnu -> radiation_cooling), the
energy budget (cell_radiated_energy / cell_injected_energy) and the u_scale /
alpha / zeta rescaling conventions. The generated cell DataFrame has the same
schema, so everything downstream of the builder is shared.
'''

from working_cooling import *
from working_cooling import _interp_state, _one_minus_beta_over_beta
# (underscore names are skipped by import *)

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
NORAR_DR_SLOPE = 0.       # d ln(dr)/d ln R of the synthetic tail. 0 = strict
                          # no-spreading, which is what 'no rarefaction' means
                          # kinematically and the only value that needs no
                          # calibration. See _norar_rows.
NORAR_Z = 4               # shell the 'prerar' alpha table is read for when the caller does
                          # not say; every shell-level entry point passes its own z, so this
                          # only backstops a direct single-cell call. 4 = reverse shock.
NORAR_LAW = 'bernoulli'   # see _norar_rows. Closes the tail on THREE conservation laws
                          # (mass, TM adiabat, Bernoulli) with no free parameter; the only
                          # remaining assumption is dr = const, worth <=1.2% on E_rad.
                          # 'adiab' replaces Bernoulli with Gamma = const, which discards
                          # the real 7-15% acceleration and moves E_rad by up to 3.4%.


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


def _bernoulli_state(x_ext, xh, rho_h, p_h, lf_h, dx, dx_h, niter=4):
  '''
  (rho, p, lfac) of the synthetic tail closed with the relativistic BERNOULLI invariant
  instead of an assumed Gamma:

      rho*Gamma*R^2*dr = const     mass conservation (exact)
      dlnp/dlnrho = gma_ad(T)      Taub-Matthews adiabat
      Gamma*h(T)  = const          Bernoulli, h = 2.5T + sqrt(1+2.25T^2), T = p/rho

  Bernoulli is the first integral of the momentum equation for steady adiabatic flow, so
  it supplies exactly the relation the other two lack -- and it is not an assumption here:
  measured along the coasting phase of cooling_g100_semi, Gamma*h is conserved to 0.7-1.3%
  while h falls 3-9% and Gamma RISES 2.0-7.7% to compensate. That acceleration is real
  physics (internal energy converting to bulk motion as the gas cools) which a Gamma=const
  closure discards. In the shock-crossing transient BEFORE the handover it is violated by
  up to 8% -- correctly, that flow is not steady -- which is another reason the extension
  must start at the handover and not at injection.

  Solved by marching along x with a fixed-point iteration per step: mass conservation
  fixes K = rho*Gamma, the adiabat gives p from rho, Bernoulli gives Gamma from T, and
  rho <- K/Gamma closes the loop. Gamma moves by <10% over the whole tail, so the
  iteration contracts hard; niter=4 is far beyond what is needed.
  '''
  hh = derive_enthalpy_fromT_TM(p_h/rho_h)
  B = lf_h*hh                                   # the invariant
  K = rho_h*lf_h*xh**2*dx_h/(x_ext**2*dx)       # = rho*Gamma at each radius
  rho = np.empty(x_ext.shape); p = np.empty(x_ext.shape); lf = np.empty(x_ext.shape)
  lnp_prev, lnr_prev = np.log(p_h), np.log(rho_h)
  for j in range(x_ext.size):
    r_j = K[j]/lf_h if j == 0 else rho[j-1]     # first guess: previous step's density
    for _ in range(niter):
      d = np.log(r_j) - lnr_prev                # adiabat, midpoint step in ln rho
      g1 = derive_adiab(np.exp(lnr_prev), np.exp(lnp_prev))
      g2 = derive_adiab(np.exp(lnr_prev + 0.5*d), np.exp(lnp_prev + 0.5*d*g1))
      lnp_j = lnp_prev + d*g2
      T = np.exp(lnp_j)/r_j
      g = B/derive_enthalpy_fromT_TM(T)         # Bernoulli -> Gamma
      r_j = K[j]/max(g, 1. + 1e-12)             # mass conservation -> rho
    rho[j], p[j], lf[j] = r_j, np.exp(lnp_j), max(g, 1. + 1e-12)
    lnr_prev, lnp_prev = np.log(r_j), lnp_j
  return rho, p, lf


def _norar_rows(row_h, x_ext, law=NORAR_LAW, fit=None, dr_slope=NORAR_DR_SLOPE):
  '''
  Hydro of the synthetic no-rarefaction tail at radii x_ext (light-seconds, all > x_h),
  continuing the state of the handover row row_h. Returns a dict of column arrays
  {rho, p, lfac, vx, dx, t}.

  law='adiab' (default): coasting, no lab-frame spreading, adiabatic. No rarefaction IS
    no spreading, so Gamma = Gamma_h and dr = dr_h; mass conservation then gives
    rho ~ R^-2, and p follows the Taub-Matthews adiabat with gma_ad EVOLVING as the gas
    cools (_adiabat_integrated). Self-anchored on row_h, so it needs no fit, no cache,
    and is continuous by construction.
  law='adiab_frozen': the same, with gma_ad held at its handover value. Under-steepens
    the tail (see _adiabat_integrated); kept only for comparison.
    Every ingredient is first-principles: nothing here is fitted or calibrated.
    NOT VALIDATED against cooling_g100_semi, despite the temptation -- Next=0 is NOT a
    no-rarefaction control. Its decompression fires at the SAME radius as the fiducial's
    (R_h/R_inj = 1.001..2.35, onset being boundary-independent to ~1%; the edge sets only
    the DEPTH), and its rigid dp/dr=0 clamp then CONFINES the layer, holding p ABOVE free
    adiabatic coasting by up to 1.28x at R/R_inj = 30. Two contaminations in opposite
    directions, neither calibrated against a real control, so that run can reject a law
    that is orders of magnitude wrong (it does, for 'bpl') but cannot arbitrate anything
    at the tens-of-percent level -- and must never be used to tune this closure.
  law='bpl': the cell's smooth-BPL fit shape, continuity-corrected at the handover so the
    fit normalisations cancel (only popts and the anchor x0 are needed, never rho0/p0).
    fit = (popts, x0) with popts = (popt_rho, popt_lfac, popt_p) from load_or_fit_celldata.
    RETAINED AS A BOUND, NOT RECOMMENDED: those fits are constrained over R/R_inj in
    [1, 1.05..3.14] -- 0.02 to 0.5 decades, entirely inside the shock-crossing transient
    -- and asymptote to rho ~ R^-1.2..-1.7, p ~ R^-2.0..-2.4, far shallower than the
    measured -1.95/-3.25. Extrapolated to R/R_inj = 30 that over-predicts p by 18-90x
    (B by 4.5-9.5x); several cells' Gamma fit also decays, whose extra observer lag pushes
    Ton past Tmax and breaks the endpoint match the whole construction exists for.
  '''
  # float() everywhere: the early_ana='shockfit' prepend builds its frame with
  # pd.concat([row.to_frame().T, shocked]), which can come back object-dtype
  xh   = float(row_h.x)
  rho_h = float(row_h.rho)
  p_h   = float(row_h.p)
  dx_h  = float(row_h.dx)
  lf_h  = float(row_h.lfac) if 'lfac' in row_h.index \
          else 1./np.sqrt(1. - float(row_h.vx)**2)
  x_ext = np.asarray(x_ext, dtype=float)

  if law == 'bernoulli':
    dx   = dx_h * (x_ext/xh)**dr_slope
    rho, p, lfac = _bernoulli_state(x_ext, xh, rho_h, p_h, lf_h, dx, dx_h)
  elif law in ('adiab', 'adiab_frozen'):
    # Gamma = Gamma_h and dr = dr_h (dr_slope = 0) => rho ~ R^-2 by mass conservation
    # (rho*Gamma*R^2*dr = const, the invariant reconstruct_cell uses). dr_slope != 0
    # relaxes the strict no-spreading assumption; see NORAR_DR_SLOPE.
    dx   = dx_h * (x_ext/xh)**dr_slope
    rho  = rho_h * (xh/x_ext)**2 * (dx_h/dx)
    p    = (_adiabat_frozen(rho, rho_h, p_h) if law == 'adiab_frozen'
            else _adiabat_integrated(rho, rho_h, p_h))
    lfac = np.full(x_ext.shape, lf_h)
  elif law == 'bpl':
    if fit is None:
      raise ValueError("law='bpl' needs fit=(popts, x0)")
    (popt_rho, popt_lfac, popt_p), x0 = fit
    def shape(popt, xx):
      return smooth_bpl_apy(np.asarray(xx, dtype=float)/float(x0), *popt)
    rho  = rho_h * shape(popt_rho, x_ext)/shape(popt_rho, xh)
    p    = p_h   * shape(popt_p,   x_ext)/shape(popt_p,   xh)
    # the Gamma fit crosses 1 on some cells; reconstruct_cell clips that to beta=0, which
    # here would make the 1/beta of the lab-time integral diverge
    lfac = np.maximum(lf_h * shape(popt_lfac, x_ext)/shape(popt_lfac, xh), 1. + 1e-12)
    dx   = dx_h * (rho_h*lf_h*xh**2)/(rho*lfac*x_ext**2)   # mass conservation
  else:
    raise ValueError(f"unknown extension law {law!r}")

  vx = np.sqrt(lfac**2 - 1.)/lfac
  # lab time: t = t_h + int dx/beta = t_h + (x - x_h) + int (1-beta)/beta dx.
  # NEVER as int dx/beta directly: the whole Ton = (1+z)(t + t0 - x) observable lives in
  # that second term (~700 light-seconds out of x ~ 1e7 here), so building it by
  # subtraction destroys it -- exactly what _one_minus_beta_over_beta exists to prevent.
  xa  = np.concatenate(([xh], x_ext))
  fa  = _one_minus_beta_over_beta(np.concatenate(([lf_h], lfac)))
  lag = np.concatenate(([0.], np.cumsum(0.5*(fa[1:] + fa[:-1])*np.diff(xa))))[1:]
  t   = float(row_h.t) + (x_ext - xh) + lag
  return dict(rho=rho, p=p, lfac=lfac, vx=vx, dx=dx, t=t, x=x_ext)


def norar_history(shocked, law=NORAR_LAW, fit=None, pts_per_dec=NORAR_PTS_PER_DEC,
    slope_thresh=NORAR_SLOPE_THRESH, window=NORAR_SLOPE_WINDOW, minpts=NORAR_MINPTS,
    precursor_tol=NORAR_PRECURSOR_TOL, dr_slope=NORAR_DR_SLOPE,
    persist=NORAR_PERSIST):
  '''
  Counterfactual post-shock history: the real rows out to the rarefaction crash, then a
  synthetic tail (see _norar_rows) continuing the smooth decay to the SAME final radius
  the input history reached. Returns (history, info).

  The target radius is taken from the history itself (shocked.x.iloc[-1]) rather than from
  a {cell: R_end/R_inj} map, so the endpoint matches the reference EXACTLY whatever
  produced it -- an r_cap window, a rar_ratio cut, the early_ana='shockfit' prepend (whose
  injection row sits 0.8-5% below the first measured row, which a ratio map would get
  wrong), or a sub-cell shift. It is also free, and keeps a 500-CSV rescan out of the
  forkserver workers.

  info: status ('ok' | 'no_crash' | 'at_end'), h (handover index), n_syn, x_h, x_end.
  status != 'ok' means the counterfactual IS the reference -- correct, not a failure:
  'no_crash' = this history never crashes, 'at_end' = the crash is at or past the target
  (e.g. an r_cap window shorter than the handover radius), so there is nothing to replace.
  '''
  x = shocked.x.to_numpy(dtype=float)
  info = dict(status='no_crash', h=None, n_syn=0,
              x_h=float(x[-1]) if len(x) else np.nan,
              x_end=float(x[-1]) if len(x) else np.nan)
  if len(shocked) < 2:
    return shocked, info
  h = rarefaction_handover(shocked, slope_thresh=slope_thresh, window=window,
                           minpts=minpts, precursor_tol=precursor_tol, persist=persist)
  if h is None:
    return shocked, info
  info.update(h=int(h), x_h=float(x[h]), status='at_end')
  if h >= len(x) - 1 or x[-1] <= x[h]*(1. + 1e-9):
    info['x_end'] = float(x[h])            # the returned frame really does stop there
    return shocked.iloc[:h+1], info        # crash at/past the target: nothing to replace

  n_syn = max(2, int(np.ceil(np.log10(x[-1]/x[h])*pts_per_dec)) + 1)
  x_ext = np.geomspace(x[h], x[-1], n_syn)[1:]   # ends EXACTLY on the target radius
  row_h = shocked.iloc[h]
  cols = _norar_rows(row_h, x_ext, law=law, fit=fit, dr_slope=dr_slope)
  # carry every column of the parent frame (i, trac, vx_u, ...) as _prepend_shocked_row
  # does, so get_variable and the sub-cell path keep working on the result
  ext = pd.DataFrame({c: np.full(len(x_ext), row_h[c]) for c in shocked.columns})
  for c, v in cols.items():
    if c in ext:
      ext[c] = v
  if 'Sd' in ext:
    ext['Sd'] = 0.                               # synthetic rows are never shocked
  ext.index = shocked.index[-1] + 1 + np.arange(len(x_ext))
  out = pd.concat([shocked.iloc[:h+1], ext])
  out.attrs = shocked.attrs
  info.update(status='ok', n_syn=len(x_ext))
  return out, info


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
  law: None (reference hydro) | 'adiab' | 'bernoulli' | 'bpl' | 'prerar'; cap: None |
  R/R_inj window. 'prerar' is the measured-reconstruction law (prerar_model).
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


def _norar_fits(key, z, env, klist):
  '''
  {cell: (popts, x0)} for the 'bpl' extension law, on the cells of klist.

  INVARIANT, do not break it: load_or_fit_celldata's disk cache
  (results/{key}/cells/{k:04d}_fit.npz) is keyed on the anchor x0 and OVERWRITES on
  mismatch, so key/k may be passed only when x0 is that cell's cellsBehindShock_fromData
  radius -- the anchor generate_cell_withDistrib and compute_shell_rarefaction_head use.
  Anchoring anywhere else (e.g. on the cell's own first measured row, which sits 0.8-5%
  higher) would silently refit and rewrite the cache the fit path and the rarefaction head
  both depend on. Using that anchor makes every call here a cache HIT, never a write.
  The continuity factors in _norar_rows divide the fit shape by its value at the handover,
  so the normalisations cancel: `norms` only has to be consistent, never correct.
  '''
  sh_data = cellsBehindShock_fromData(open_rundata(key, z))
  out = {}
  for k in klist:
    cell_data = open_celldata(key, k)
    if cell_data is False:
      continue
    sel = sh_data.loc[sh_data.i == k]
    if not len(sel):
      continue
    row = sel.iloc[0]
    norms = [get_variable(row, nm, env) for nm in ('rho', 'lfac', 'p')]
    try:
      popts = load_or_fit_celldata(cell_data, ['rho', 'lfac', 'p'], norms, env,
                                   row.x, key=key, k=int(k))
    except RuntimeError:
      continue                     # unfittable cell: falls back to no extension
    out[int(k)] = (popts, float(row.x))
  return out


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
    after = np.flatnonzero(sd[ish[0]:] == 0)
    start = ish[0] + after[0] + n_settle if len(after) else len(sd)
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
    then the smooth shocked-layer decay continued to the same final radius (norar_history).
    None (default) is the method's normal behaviour and leaves this code path untouched.
    'adiab' | 'bpl' select the extension law (see _norar_rows); norar_fit = (popts, x0) is
    required by 'bpl' only. norar_ppd is its sampling in points per decade of R.
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
  # the reference has under the same window (norar_history takes it from the frame)
  if norar is not None:
    if norar == 'prerar':
      # The reconstruction law (prerar_model): follows the real rows out to the edge of the
      # rarefaction-free window, then prolongs on the measured alpha tables -> the derived
      # causal-contact asymptote. Imported lazily so the module stays importable, and the
      # forkserver workers stay cheap, for every other law.
      from prerar_model import prerar_history
      shocked, _ = prerar_history(shocked, norar_fit, z=norar_z,
                                  pts_per_dec=norar_ppd)
    else:
      shocked, _ = norar_history(shocked, law=norar, fit=norar_fit, pts_per_dec=norar_ppd)
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
  gmin_edges, gmax_edges, bsyn_edges = evolve_gma_bounds_edges(
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
         'gmin': gmin_edges[:-1], 'gmax': gmax_edges[:-1], 'bsyn': bsyn_edges[:-1]}
  out = pd.DataFrame.from_dict(dic)
  for key in attrs:
    out.attrs[key] = attrs[key]
  return out, env

def load_shockfront_states(key, z, env, source='shockfit', t_max_fac=3.):
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
  raise ValueError(f"early_ana source must be 'shockfit' or 'au', got {source!r}")

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
  norar_fit = _norar_fits(key, z, env, [k]).get(int(k)) if norar == 'bpl' else None
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

def get_shell_nuFnu_fromData(key, z, u_scale=1., alpha=1., zeta=1., klist=None,
    VFC=False, r_ref=1.2, Tmax=5, NT=500, lognu_min=-2.5, lognu_max=2.5, Nnu=400,
    dlnrho_max=DLNRHO_MAX, dlnsyn_max=DLNSYN_MAX, Tb_min=None, Tb_lin=None, n_settle=1,
    subcell_dlogT=None, subcell_max=32, early_ana=None, early_frac=0.,
    rar_cut=None, norar=None, r_cap=None,
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
    shocked-layer decay past the handover (norar_history / _norar_rows). Mutually
    exclusive with rar_cut: both are treatments of the same wave, and where rar_cut stops
    the cell early, this one holds the worldline EXTENT fixed and removes only the crash
    -- which is what isolates the wave's effect on the emission. None (default) leaves
    this code path untouched. 'adiab' | 'bpl' select the law; 'both' computes the
    reference AND the 'adiab' counterfactual in one pass, sharing their common leading
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
    if any(l not in (None, 'adiab', 'adiab_frozen', 'bernoulli', 'bpl', 'prerar')
           for l in laws):
      raise ValueError(f"norar must be None, 'adiab', 'adiab_frozen', 'bernoulli', 'bpl', "
                       f"'prerar' or 'both', got {norar!r}")
    # the 'bpl' law needs each cell's fitted profile; anchored on its
    # cellsBehindShock_fromData radius, which is the anchor load_or_fit_celldata's disk
    # cache is keyed on -- any other x0 would silently overwrite it (see _norar_fits)
    fits = _norar_fits(key, z, env, klist) if 'bpl' in laws else None
    # 'prerar' instead needs the alpha table, which is SHARED by every cell (not narrowed
    # per-cell below, unlike the bpl fits) and carries this run's own settling correction.
    if 'prerar' in laws:
      from prerar_model import load_table, with_settle, TABLE_KEY, NCELLS
      fits = with_settle(load_table(TABLE_KEY, z), key, z, NCELLS)
    variants = [(data_method_name(l, r_cap), None,
                 dict(r_cap=r_cap, norar=l, norar_fit=fits, norar_z=z)) for l in laws]
  elif rar_cut is None:
    variants = [(data_method_name(None, r_cap), None, dict(r_cap=r_cap))]
  elif rar_cut == 'model':
    variants = [('data_rarcut',
                 load_shell_rarefaction(key, z, env, n_shell=len(klist)), {})]
  elif rar_cut == 'both':
    variants = [('data', None, {}),
                ('data_rarcut',
                 load_shell_rarefaction(key, z, env, n_shell=len(klist)), {})]
  else:
    raise ValueError(f"rar_cut must be None, 'model' or 'both', got {rar_cut!r}")
  paired = len(variants) == 2

  # shock-front state table for the early-datapoint reconstruction, once per shell
  sh_data = load_shockfront_states(key, z, env, source=early_ana) \
            if early_ana is not None else None

  # first pass: post-shock histories + injection rows (klist = onset order),
  # so the sub-cell edges can be computed from the cells' own onsets. The
  # early reconstruction (prepend) happens here, BEFORE rescaling, so the
  # onset ladder and injection states already carry it.
  hists, attrs_l, inj_rows = [], [], []
  n_prepended = 0
  for k in klist:
    cell_data = open_celldata(key, k)
    if cell_data is False:
      hists.append(None); attrs_l.append(None); inj_rows.append(None)
      continue
    t_off = cell_data.t.iloc[0] if (len(cell_data) and cell_data.index[0] == 0) else 0.
    shocked = select_postshock_rows(cell_data, n_settle)
    if len(shocked) < 2:
      hists.append(None); attrs_l.append(None); inj_rows.append(None)
      continue
    shocked['t'] = shocked['t'] - t_off
    if sh_data is not None:
      sel = sh_data.loc[sh_data.i == k]
      if len(sel):
        n0 = len(shocked)
        shocked = _prepend_shocked_row(shocked, sel.iloc[0], env, early_frac)
        n_prepended += len(shocked) - n0
    hists.append(shocked); attrs_l.append(cell_data.attrs); inj_rows.append(shocked.iloc[0])
  if early_ana is not None:
    print(f'get_shell_nuFnu_fromData early reconstruction ({early_ana}): '
          f'prepended shock-front states to {n_prepended} cells')

  # adaptive sub-cell onset edges (shared with the fit driver); onsets from the
  # cells' own injection rows, unrescaled env (dimensionless bar{T} convention)
  barT_grid = T - 1.
  floor = barT_grid[barT_grid > 0.].min()   # earliest resolved bar{T} on the obs grid
  sub_edges = [None]*len(klist)
  if subcell_dlogT is not None:
    barT_on = np.array([((get_variable(r, 'Ton', env) - env.Ts)/env.T0) if r is not None else np.nan
                        for r in inj_rows])
    # measured onsets are late by the settle + snapshot-cadence delay (the first
    # settled row is 1+ snapshots after the crossing), so they cannot reach below
    # the cadence; the CD-adjacent first cell is physically shocked at collision
    # (barT ~ 0, as the fit's reconstructed sh_data states). Anchor its interval
    # at 0 so its sub-cells span down to the grid floor, reconstructing the
    # cadence-missed early rise (their onsets are then placed exactly by the
    # worldline shift in the sub-cell loop below).
    finite = np.flatnonzero(np.isfinite(barT_on))
    if len(finite):
      barT_on[finite[0]] = 0.
    sub_edges = compute_subcell_edges(barT_on, floor, subcell_dlogT, subcell_max)

  skipped = []
  n_refined = 0
  # one accumulator set per variant, in `variants` order (None where the flux is
  # not being computed at all, so it cannot be mistaken for a zero lightcurve)
  nuFnu_v = [None]*len(variants) if energies_only \
            else [np.zeros((len(Tobs), len(nuobs))) for _ in variants]
  E_rad_v, E_int_v, E_inj_v = ([0.]*len(variants) for _ in range(3))
  def _accum_energy(iv, cell, cell_env):
    E_rad_v[iv] += cell_radiated_energy(cell, cell_env)
    E_inj_v[iv] += cell_injected_energy(cell, cell_env)
    c0 = cell.iloc[0]
    E_int_v[iv] += get_variable(c0, 'ei', cell_env) * get_variable(c0, 'V3p', cell_env)

  def _make_cell(hist, attrs, kk):
    '''Build one cell per variant from a shared history; None where it is unusable.'''
    out = []
    for _, rmap, kw in variants:
      # the 'bpl' fit table is per cell (and per sub-cell on its truncated index, the
      # rar_map_lookup convention): resolve it here, not in the variant spec
      if kw.get('norar') == 'bpl':
        kw = dict(kw, norar_fit=(kw['norar_fit'] or {}).get(int(kk)))
      c, ce = generate_cell_fromHistory(hist, attrs, env,
          u_scale=u_scale, alpha=alpha, zeta=zeta, r_ref=r_ref, Tmax=Tmax,
          dlnrho_max=dlnrho_max, dlnsyn_max=dlnsyn_max,
          rar_ratio=(rar_map_lookup(rmap, kk) if rmap is not None else None), **kw)
      out.append((c, ce))
    return out

  def _emit(cells):
    '''Accumulate one cell's flux (+ energies) into every variant. With two variants
    the pair evaluator computes their shared leading steps ONCE -- the whole point of
    rar_cut='both' -- and is bit-identical to evaluating them separately.'''
    if energies_only:
      ok = False
      for iv, (cell, cell_env) in enumerate(cells):
        if cell is False:
          continue
        _accum_energy(iv, cell, cell_env)
        ok = True
      return ok
    if paired and cells[0][0] is not False and cells[1][0] is not False \
        and func_Fnu is get_Fnu_cell_evolving:
      (cf, ef), (cc, _) = cells
      Ff, Fc = get_Fnu_cell_evolving_pair(nuobs, Tobs, cf, cc, ef, **kwargs)
      # nu F_nu exactly as get_nuFnu does it (same nu0 branch, same broadcast), so the
      # paired path differs from the separate one in nothing but evaluation order
      norm = kwargs.get('norm', True)
      for iv, F in enumerate((Ff, Fc)):
        cell_iv, env_iv = cells[iv]
        nu0 = env_iv.nu0FS if (cell_iv.iloc[0].trac > 1.5) else env_iv.nu0
        nub_iv = (nuobs/nu0 if norm else nuobs)[np.newaxis, :]
        nuFnu_v[iv] += nub_iv * F
        if return_energies: _accum_energy(iv, cell_iv, env_iv)
      return True
    ok = False
    for iv, (cell, cell_env) in enumerate(cells):
      if cell is False:
        continue
      nuFnu_v[iv] += get_nuFnu(func_Fnu, nuobs, Tobs, cell, cell_env, **kwargs)
      if return_energies: _accum_energy(iv, cell, cell_env)
      ok = True
    return ok

  for idx, k in enumerate(klist):
    if hists[idx] is None:
      skipped.append(k)
      continue
    if sub_edges[idx] is None:
      cells = _make_cell(hists[idx], attrs_l[idx], k)
      if not _emit(cells):
        skipped.append(k)
        continue
    else:
      # split the parent's onset interval into flux-conserving sub-cells at
      # geometrically-spaced onsets, reusing its actual history shape
      a, b, edges = sub_edges[idx]
      inj_par = inj_rows[idx]
      for j in range(len(edges)-1):
        e0, e1 = edges[j], edges[j+1]
        onset_c = np.sqrt(e0*e1)                       # geometric centre
        f = float(np.clip((onset_c - a)/(b - a), 0., 1.))
        dx_w = inj_par.dx * (e1 - e0)/(b - a)          # onset-width weight => flux conserved
        sub = _interp_state(inj_par, inj_rows[idx+1], f, dx_w)
        # place the sub-cell onset exactly at onset_c: shift the anchor state
        # along its worldline (dt, dx = beta*dt => dbarT = (1+z)(1-beta)dt/T0).
        # Backward shifts (onset_c below the measured onsets) extrapolate the
        # parent's injection state to where the snapshot cadence could not see --
        # the data analog of the fit path's sub-states interpolated on the
        # reconstructed shock front (which reaches barT = 0 at collision).
        barT_sub = (get_variable(sub, 'Ton', env) - env.Ts)/env.T0
        beta = sub['vx']
        delta = (onset_c - barT_sub)*env.T0/((1. + env.z)*(1. - beta))
        sub['t'] += delta
        sub['x'] += beta*delta
        hist = _subcell_history(hists[idx], sub)
        # sub-cells look the map up on their own INTERPOLATED index, as the fit path
        # does -- R_rar/R_injection varies smoothly across the shell
        _emit(_make_cell(hist, attrs_l[idx], sub['i']))
      n_refined += len(edges) - 2
  if skipped:
    print(f'get_shell_nuFnu_fromData on sim {key}: skipped {len(skipped)} cells '
          f'(no data or no usable post-shock history): {skipped}')
  if subcell_dlogT is not None:
    kr = [int(klist[i]) for i in range(len(klist)) if sub_edges[i] is not None]
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
