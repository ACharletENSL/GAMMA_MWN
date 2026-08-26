# -*- coding: utf-8 -*-
# @Author: acharlet

'''
Sweep of log10(gamma_c/gamma_m) via the Granot (2012) 'alpha' hydro
rescaling, applied to a single hydro simulation (default: cooling_g100), to
study how the cooling regime reshapes the full-shell (all shocked cells
summed) lightcurves and spectra of one shell (default: the reverse shock,
z = Z_SHELL = 4; z=1 is the forward shock, on the same RS-normalised grids).
Plot set:
  - lightcurve SHAPE: peak-normalised in bar{T}=(Tobs-Ts)/T0 and flux (every
    curve through (1,1)), linear + log, one 2-panel fig per frequency
  - spectral EVOLUTION: rise/peak/tail spectra, one fig per gamma_c/gamma_m
  - peak & fluence spectra, all regimes together on a nu/nu_m axis, normalised
    to the flux at nu_m, to the peak flux, or peak-normalised and then scaled by
    each point's radiative efficiency (SPEC_MODES)
Example use in command line:
  python -c "import sweep_gammacm as S; S.main(use_cache=False, nproc=4)"
  python -c "import sweep_gammacm as S; S.main(z=1, method='data_rarcut', nproc=7)"
'''

import os
import glob
from types import SimpleNamespace
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import least_squares

from environment import MyEnv, rescale_hydro, GAMMA_dir
from phys_functions import granot_sari_syn, syn_cutoff_R
from spectral_breaks import (segment_slopes, measure_cutoff_nuM, _widest_run, edge_slope,
    SLOPE_SMOOTH, SLOPE_TOL, MIN_PTS, MIN_DEX, FIT_DEC, CUT_FAC, EDGE_VFC_TOL)
from working_cooling import (get_shell_nuFnu, open_rundata, cellsBehindShock_fromData,
    load_shell_rarefaction_offT, check_extracted_cells, open_celldata)
from working_cooling_data import (get_shell_nuFnu_fromData, data_method_name,
    select_postshock_rows, NORAR_LAW)
from IO import get_variable
from plotting_functions import nF_label, sci_notation

# env scalars kept per sweep point (enough for nu_over_num + the regime analysis)
_ENV_KEYS = ('nu0', 'nuc', 'T0', 'Ts', 'nu0F0', 'gma_c', 'gma_m', 'gma_max', 'psyn')
# forward-shock counterparts, needed only by the both-shells figures (sweep_shells):
# fac_nu/fac_F carry the FS->RS flux-unit conversion, the gma_*FS the FS regime.
# Kept SEPARATE from _ENV_KEYS and read back guarded, so caches written before they
# existed still load.
_ENV_KEYS_OPT = ('nu0FS', 'T0FS', 'fac_nu', 'fac_F', 'gma_mFS', 'gma_cFS', 'gma_maxFS')

DEFAULT_KEY = 'cooling_g100'    # fiducial simulation the sweep runs on unless told otherwise;
                                # it gets the unsuffixed cache dirs (method_outdir)
DEFAULT_METHOD = 'data'   # reference flux computation: the data-driven driver with NO
                          # rarefaction machinery (rar_cut=None), i.e. the rarefaction wave
                          # as the simulation itself resolves it, every cell followed to its
                          # last snapshot. 'data_rarcut' (the modelled sharp R_rar cut-off)
                          # is the prescription measured AGAINST this, see sweep_rarcut.py
R_CAP = 30.               # analysis window in R/R_injection for the '_cap' methods, applied
                          # to BOTH sides of the no-rarefaction comparison so their endpoints
                          # still match while the counterfactual's extension only has to
                          # carry ~1-1.5 decades instead of ~2.5 (sweep_prerar.main_capped).
                          # 30 sits inside cooling_g100_semi's coverage (R/R_inj reaches
                          # 64..155) for every cell, so the capped variant is the one that
                          # can be checked against a simulation that has no rarefaction.
OUTDIR = os.path.join(GAMMA_dir, 'bin', 'Tools', 'figures', 'gammacm_sweep')
OUTDIR_DATA = OUTDIR + '_data'   # data-driven method (working_cooling_data), see method_outdir
EARLY_ANA = 'shockfit'    # data method: reconstruct the cadence-missed early datapoints from
                          # this run's shock-front states, so the two methods share their
                          # injection states and the comparison isolates the post-injection
                          # hydro treatment (see working_cooling_data.load_shockfront_states)

LOG10RATIO_ARR = np.arange(-5, 3)          # -5..+2
Z_SHELL = 4                                 # reverse-shock shell
TMAX, NT = 1000, 250      # sized on the REFERENCE method, which has no cut-off: with
                          # rar_cut=None every cell is followed to its last snapshot, and on
                          # cooling_g100 that is bar{T} = 646..650 (the run is 82x longer than
                          # the old cooling_fid_raref_ext fiducial, whose cells ended at
                          # 8.2..10.8). Tobs_max at 100 would have clipped the reference itself.
                          # Costs nothing: get_Fnu_cell_evolving walks every cooling step
                          # whatever the window, and NT is what the kernel is linear in.
                          # The obs grid is geometric in bar{T} (obs_arrays, Tb_min set), so the
                          # decay is log-sampled; NT keeps ~36 points/decade over 7 decades, and
                          # TB_LIN adds 200 linear samples across the peak regardless.
                          # With rar_cut='model' all cells are instead dark by bar{T} = 2.03
                          # (RS) / 1.70 (FS) and the window is pure tT^-2 tail past that.
LOGNU_MIN = -6            # fixed low end of the frequency window, in log10(nu/nu_m) and the
                          # SAME for every sweep point, so all spectra start at the same x
                          # (their left ends land on the panel's left spine, not inside it)
LOGNU_ABOVE_NUM = 1.5     # high end, in decades above each point's OWN nu_M: the window top is
                          # log10(nu_M/nu_m) + this, so every spectrum shows its exponential
                          # cutoff and has fallen >3.5 decades below its peak (the plot floor)
                          # before the grid ends. nu_M/nu_m = (gma_max/gma_m)**2 grows as
                          # alpha**(3/2) over the sweep (1e4.9 at logr=-5 to 1e10.1 at +2), so
                          # a single fixed top cannot capture it. Measured: the logr=-5 spectrum
                          # crosses the floor 0.68 decade above nu_M, so 1.5 clears it.
NNU_PER_DEC = 33          # frequency sampling; the window span now varies with the point
                          # (12.4 decades at logr=-5, 17.6 at +2), so Nnu is set from the span
NNU_MIN, NNU_MAX = 400, 650
NU_TARGETS = [1e-2, 0.1, 1.0]                # lightcurve panel freqs, as fractions of the
                                             # peak freq nu_pk=max(nu_m,nu_c) (= the nub axis)
XLIM_LIN = (0., 4.)                          # bar{T}/bar{T}_f range of every LINEAR-time
                                             # lightcurve panel. ONE window across the whole
                                             # suite -- this module's shape figures,
                                             # sweep_compare's A/B variants and sweep_shells'
                                             # per-regime panels all import it, so the pulse
                                             # is read at the same scale everywhere. It clears
                                             # the crossing (x=1) and the rarefaction band
                                             # (T_rf ~ 1.55) with room for the decay; the
                                             # full/cut divergence sweep_compare tracks is
                                             # still creeping up at the right edge (it tops
                                             # out near x~5-6), which is the price of the
                                             # shared scale. The log panels need no such
                                             # constant: they all run from xmin=1e-3.
SPEC_YSPAN = 3.55                            # decades of flux shown on the spectral-evolution
                                             # plots (same fixed range on all, for comparison)
NU_REF = 1.0                                 # reference freq for rise/peak/tail detection
FRAC_RISE, FRAC_TAIL = 0.1, 0.1              # rise/tail spectra are taken where the nu_ref
                                             # lightcurve is at this fraction of its peak. 0.2 is
                                             # the knee of the evolution-vs-flux-spread trade:
                                             # above it, 0.1 dex of extra rise->tail break motion
                                             # costs ~0.15 dec of spread between the curves, below
                                             # it ~0.3 dec, and the faint curves sink into the
                                             # SPEC_YSPAN floor (max spread 0.75 dec at 0.2, 1.05
                                             # at 0.1, 1.34 at 0.05). The TAIL is held higher
                                             # because measure_regime's nu_m/nu_c assignment only
                                             # holds while the mid-segment slope stays on its side
                                             # of SMID_FCSC: in the slow-cooling points the
                                             # high-latitude tail softens past it and the two
                                             # breaks SWAP (log10(gc/gm)=+1 at F/F_pk=0.181,
                                             # +2 at 0.076, measured on the raref_ext data cache,
                                             # whose tails are not rarefaction-cut). 0.2 sampled
                                             # +1 two grid points from that flip; 0.3 keeps a
                                             # 1.6x margin for 7% less break evolution.
                                             # FRAC_RISE has a CEILING instead, ~0.6: above
                                             # F/F_pk~0.73 the log10(gc/gm)=0 rise has already
                                             # left its early uncooled SC state and the figure
                                             # loses the SC-rise -> FC-peak transition.
TB_MIN = 1e-4                                # early-time floor: obs grid geometric in
                                             # bar{T}=(Tobs-Ts)/T0 over [TB_MIN, Tmax]
TB_LIN = (0.5, 9., 200)                      # extra samples spaced LINEARLY in bar{T},
                                             # merged into that grid (obs_arrays). The
                                             # geometric grid resolves the log axis and
                                             # leaves only 12 of 250 points in
                                             # bar{T}/bar{T}_f = 1..2, where the lightcurve
                                             # peaks -> linear-scale lightcurves come out
                                             # jagged. This covers the window they show
                                             # (XLIM_LIN, with margin) at a uniform 0.043
                                             # in bar{T}, and keeps every geometric point,
                                             # so the early rise TB_MIN buys is untouched.
SUBCELL_DLOGT = 0.008                         # smooth the early-time cell-sum staircase:
                                             # split CD-adjacent cells whose onsets are
                                             # >this in log10(bar{T}) apart (None=off).
                                             # Was 0.05, whose 0.0486 dex realised onset
                                             # spacing sat at ~2x the early grid spacing
                                             # (0.0241 dex) and ALIASED into a point-to-
                                             # point zigzag on the fast-cooling early rise
                                             # (lag-1 autocorrelation -0.76, period 0.048
                                             # dex; ln-flux roughness 0.072 at
                                             # log10(gc/gm)=-5 vs 0.011 slow, 0.0004 late).
                                             # Fast cooling aliases and slow does not
                                             # because there each sub-cell contributes a
                                             # sharp spike (plus its tT^-2 tail) rather
                                             # than a long smooth decay. Measured at
                                             # log10(gc/gm)=-5: 0.05 -> 0.02 halves the
                                             # residual (rms of ln F about a local
                                             # quadratic 0.0203 -> 0.0106, lag-1 ac
                                             # -0.76 -> -0.29). It does NOT eliminate it:
                                             # amplitude ~ onset spacing^0.8, so this is a
                                             # linear cost/smoothness trade with no
                                             # threshold. 0.02 -> 0.008 takes it 0.0079 ->
                                             # 0.0053 (lag-1 ac -0.53 -> -0.36) for ~+57%
                                             # per point; 0.004 gives 0.0036 for +223%,
                                             # which is where the trade stops being worth it.
SUBCELL_MAX = 400                            # cap on sub-cells per parent. Must rise with
                                             # SUBCELL_DLOGT: the first cell spans TB_MIN
                                             # up to the next onset, so at 0.02 the old
                                             # default 32 clipped it back to 0.044 dex and
                                             # undid the refinement. Raising it ALONE is a
                                             # no-op (nothing was capped at 0.05). 400 for
                                             # SUBCELL_DLOGT=0.008; verified nothing capped.
# ---------------------------------------------------------------------------
# WHAT THE SUB-CELL LADDER CANNOT FIX (measured on cooling_g100, z=4, 2026-08-10)
# ---------------------------------------------------------------------------
# The lightcurves carry a residual wiggle just BEFORE the peak, over
# bar{T}/bar{T}_f = 0.5..1: 0.19% rms of the peak flux (0.54% peak-to-peak) at
# log10(gc/gm) = -5, 0.17% at -4, fading to 0.08% by 0. Visible on the LINEAR panel
# in fast cooling only. It is a RESOLUTION FLOOR of the 500-cell shell, not a defect:
# the ~275 cells switching on across that window sit at a natural onset spacing of
# 0.00134 dex, and in fast cooling each one burns off almost instantly, so its
# emission is a spike NARROWER than that spacing and the shell sum is a comb.
#
# Do not attack it with SUBCELL_DLOGT. Sub-cells reuse the parent's history shape, so
# they add more spikes without widening any, and the return is dismal -- measured:
#   dlogT  0.02 -> 0.1916%   (+129 sub-cells,   63 s)
#          0.004 -> 0.1915%  (+821,  130 s)  -- no effect, it never reaches 0.00134 dex
#          0.002 -> 0.1566%  (+1803, 228 s)
#          0.001 -> 0.1385%  (+3896, 456 s)
# i.e. ~12% per doubling of emitters (amplitude ~ spacing^0.19); an invisible 0.05%
# would need ~500x the cost. Also note the metric matters: a second difference in
# INDEX space is dominated by the curve's own curvature on this uneven grid (it
# returned an identical 0.01999 for r_ref = 1.05..1.4) and will mislead. Score it as
# the residual about a local quadratic fitted in bar{T} DISTANCE.
# ---------------------------------------------------------------------------
R_REF = 1.1                                   # cooling-step ratio (gma_max drops by R_REF per
                                             # step). cell_radiated_energy evaluates each step
                                             # at its geometric-mean gamma, so the budget is
                                             # second-order in R_REF (the old left-edge sum
                                             # overshot by ~3.5% at 1.1, ~5% at 1.2)
NU_M_LABEL = '$\\nu/\\nu_m$'


def compute_alpha_sweep(key, log10ratio_arr):
  '''
  Closed-form alpha (Granot length/time rescale) needed to reach each target
  log10(gma_c/gma_m), derived from the live baseline env: gma_m is invariant
  under alpha, gma_c ~ alpha**2, so alpha = 10**((target-log10ratio0)/2).
  '''
  env0 = MyEnv(key)
  log10ratio0 = np.log10(env0.gma_c / env0.gma_m)
  log10ratio_arr = np.asarray(log10ratio_arr, dtype=float)
  alpha_arr = 10.**((log10ratio_arr - log10ratio0) / 2.)
  return alpha_arr, log10ratio0


def _pool_context():
  '''
  Start method for the sweep's process pool: 'forkserver', explicitly, on every Python
  version. Both alternatives are broken here, in ways that cost hours because neither
  raises -- the pool just stops.

  NOT 'fork'. The parent is multi-threaded by the time the pool is built: numpy's BLAS
  spins up one thread per core on the serial warm-up point. A forked child gets a copy
  of the parent's memory but only the forking thread, so any mutex a BLAS thread held at
  fork time is locked forever in the child. Measured here: sweep 1 finished, sweep 2's
  four workers each sat in futex_wait_queue at 25 MB RSS -- a bare interpreter, blocked
  before it could load a single cell -- for six hours with no error. Python 3.14
  deprecated fork in multi-threaded processes for exactly this, and made forkserver the
  Linux default; do not "restore" fork to get the old behaviour back.

  NOT the platform default either, which is still fork on <=3.13 and would silently
  reintroduce the above wherever this runs.

  The forkserver cost is that workers do NOT inherit the in-process state the warm-up
  point builds (_RAR_HEAD_MEM, _DATA_END_MEM, loaded cell fits, numba's JIT) and re-read
  it from disk instead. That is a slowdown, not a correctness problem -- every one of
  those caches is backed by a file, which is what the warm-up point is really for: it
  populates them serially so parallel workers only ever READ those shared paths.

  Callers must be import-safe: a forkserver child re-imports the main module (as
  __mp_main__), so a driver SCRIPT needs its work behind `if __name__ == '__main__':`.
  Without it the driver's top level re-runs inside the forkserver, which then sits in a
  second sweep and never serves a worker -- the same silent stall, different cause.
  Driving via `python -c` has no main path and is unaffected.

  Falls back to the default context where forkserver is unavailable (Windows: spawn).
  '''
  import multiprocessing as mp
  try:
    return mp.get_context('forkserver')
  except ValueError:
    return mp.get_context()


def _resolve_nproc(nproc, npoints):
  '''Resolve the worker count: explicit arg > env GAMMACM_NPROC > cpu_count()-1
  (leave one core free). Clamped to [1, npoints].'''
  if nproc is None:
    env_np = os.environ.get('GAMMACM_NPROC')
    nproc = int(env_np) if env_np else max(1, (os.cpu_count() or 1) - 1)
  return max(1, min(int(nproc), npoints))


def _nu_window(key, alpha, lognu_min=LOGNU_MIN, lognu_above=LOGNU_ABOVE_NUM):
  '''
  Frequency window of one sweep point, in log10(nu/nu_m) (the units get_shell_nuFnu's
  lognu_min/lognu_max are in, since it builds nuobs = nub*env.nu0 with nu0 = nu_m).
  The low end is the fixed shared LOGNU_MIN; the top is anchored on the point's own
  cutoff nu_M/nu_m = (gma_max/gma_m)**2, taken from the alpha-rescaled env (the same
  env get_shell_nuFnu builds internally at u_scale=zeta=1). Returns (lo, hi, Nnu),
  Nnu set from the span at ~NNU_PER_DEC points per decade.
  '''
  env0 = MyEnv(key)
  env_a = rescale_hydro(alpha, 1., env0) if alpha != 1. else env0
  hi = 2.*np.log10(env_a.gma_max/env_a.gma_m) + lognu_above
  Nnu = int(np.clip(round((hi - lognu_min)*NNU_PER_DEC), NNU_MIN, NNU_MAX))
  return lognu_min, hi, Nnu


def _data_method_spec(method):
  '''
  Parse a data-path sweep method into (extension law, r_cap, canonical name).

  Grammar: 'data' [ '_norar' [ '_<law>' ] ] [ '_cap' ], e.g.
    data             the reference: real histories to their last snapshot
    data_norar       the no-rarefaction counterfactual, default law (NORAR_LAW)
    data_cap         the reference truncated at R/R_inj = R_CAP
    data_norar_cap   the counterfactual over that same window (endpoints still match)

  ONE parser for the whole grammar, so method_outdir and _compute_point cannot drift
  apart; working_cooling_data.data_method_name builds the same strings from the other
  direction and is checked against this here.
  '''
  cap = None
  base = method
  if base.endswith('_cap'):
    base, cap = base[:-4], R_CAP
  if base == 'data':
    law = None
  elif base == 'data_norar':
    law = NORAR_LAW             # shorthand on INPUT only; canonicalised to the explicit
  elif base.startswith('data_norar_'):   # form below, so caches never collide across laws
    law = base[len('data_norar_'):]
    if law not in ('prerar',):
      raise ValueError(f"unknown extension law {law!r} in method {method!r}")
  else:
    raise ValueError(f"unknown method {method!r}")
  name = data_method_name(law, cap)
  if name != method and method not in ('data_norar', 'data_norar_cap'):
    raise ValueError(f"method {method!r} is not canonical (expected {name!r})")
  return law, cap, name


def method_outdir(method=DEFAULT_METHOD, key=None, z=Z_SHELL):
  '''
  Output/cache directory of a sweep method: 'data' (get_shell_nuFnu_fromData, actual
  per-cell snapshot histories, cells followed to their last snapshot -- the REFERENCE,
  carrying the rarefaction wave as the simulation resolves it), 'data_rarcut' (the
  same, but with the fit path's sharp modelled R_rar cut-off opted in, rar_cut='model')
  or 'fit' (get_shell_nuFnu, hydro reconstructed from smooth-BPL fits). Separate
  directories because the point cache is keyed on log10ratio alone -- a shared one
  would overwrite.
  key: the sweep's simulation. DEFAULT_KEY (or None) keeps the historical
  unsuffixed paths; any other key gets its own '_{key}' directory. The data
  method reads the cell histories to their last snapshot, so two runs of the
  SAME setup with different durations give different results and must not share
  a cache -- and they cannot be told apart by alpha (identical env), which is
  what sweep_compare.load_pairs checks.
  z: the emitting shell. Z_SHELL (the reverse shock) keeps the historical
  unsuffixed paths; the forward shock (z=1) gets its own '_z={z}' directory, for
  the same reason as key -- the two shells' points share the log10ratio cache
  filename and would otherwise overwrite each other (see sweep_shells.py).
  '''
  if method == 'fit':
    d = OUTDIR
  elif method == 'data':
    d = OUTDIR_DATA
  elif method == 'data_rarcut':
    d = OUTDIR_DATA + '_rarcut'
  elif method == 'data+rarcut':
    # not a cache of its own: it fills the 'data' and 'data_rarcut' directories (see
    # _compute_point). Report the reference one so run_sweep's reload finds points.
    d = OUTDIR_DATA
  elif method == 'cap+norar_cap':
    # likewise, fills 'data_cap' and 'data_norar_cap'; report the reference one
    d = OUTDIR_DATA + f'_cap={R_CAP:g}'
  else:
    law, cap, _ = _data_method_spec(method)     # raises on anything unknown
    d = OUTDIR_DATA
    if law is not None:
      d += f'_norar_{law}'      # law ALWAYS explicit; see data_method_name
    if cap is not None:
      # R_CAP goes in NUMERICALLY: it changes the numbers, so a different window must
      # never silently reuse a cache written with the old one
      d += f'_cap={cap:g}'
  if key not in (None, DEFAULT_KEY):
    d = f'{d}_{key}'
  return d if z == Z_SHELL else f'{d}_z={z}'


def _compute_point(key, z, logr, alpha, Tmax, NT, lognu_min, lognu_above, outdir,
    method='fit'):
  '''
  Compute + cache one sweep point (module-level so it is picklable for the
  process pool). Returns a small picklable summary dict (no env / arrays).
  method: 'fit' | 'data' | 'data_rarcut' | 'data_norar[_bpl]' | any of those data
  variants with a '_cap' suffix | the paired 'data+rarcut' and 'cap+norar_cap' -- which
  nuFnu driver computes the point; they take the same kwargs here, and the observer grids
  (_nu_window, Tb_min, NT) are identical, so the points are directly comparable.

  'data+rarcut' computes the reference AND the modelled cut together
  (get_shell_nuFnu_fromData(rar_cut='both')), which is much cheaper than the two
  sweeps separately because the treatments share their leading cooling steps. It
  writes ONE point into EACH of the two methods' own cache directories, tagged with
  that method's name, so the cache layout is exactly what running them separately
  produces and every downstream consumer is untouched. `outdir` is then ignored --
  the two real destinations are derived from (key, z).
  '''
  lo, hi, Nnu = _nu_window(key, alpha, lognu_min, lognu_above)
  kw = dict(alpha=alpha, Tmax=Tmax, NT=NT, Nnu=Nnu, lognu_min=lo, lognu_max=hi,
            Tb_min=TB_MIN, Tb_lin=TB_LIN, subcell_dlogT=SUBCELL_DLOGT,
            subcell_max=SUBCELL_MAX, r_ref=R_REF, return_energies=True)
  if method == 'data+rarcut':
    nuobs, Tobs, env, res = get_shell_nuFnu_fromData(
        key, z, early_ana=EARLY_ANA, rar_cut='both', **kw)
    variants = [(m, outdir_m) + res[m] for m, outdir_m
                in (('data', method_outdir('data', key, z)),
                    ('data_rarcut', method_outdir('data_rarcut', key, z)))]
  elif method == 'cap+norar_cap':
    # the capped reference AND the capped counterfactual in one pass, sharing the cooling
    # steps they have in common (get_Fnu_cell_evolving_pair). Both sides are new here, so
    # unlike the uncapped counterfactual this cannot clobber a cache others compare
    # against. NB the shared prefix is smaller than for 'data+rarcut' -- the two sides
    # diverge at the HANDOVER, not at the cut.
    nuobs, Tobs, env, res = get_shell_nuFnu_fromData(
        key, z, early_ana=EARLY_ANA, norar='both', r_cap=R_CAP, **kw)
    variants = [(m, method_outdir(m, key, z)) + res[m]
                for m in _paired_members('cap+norar_cap')]
  elif method == 'data_rarcut':
    nuobs, Tobs, env, nuFnu, E_rad, E_int, E_inj = get_shell_nuFnu_fromData(
        key, z, early_ana=EARLY_ANA, rar_cut='model', **kw)
    variants = [(method, outdir, nuFnu, E_rad, E_int, E_inj)]
  elif method.startswith('data'):
    # 'data' (the reference) and the no-rarefaction counterfactuals, capped or not
    law, cap, _ = _data_method_spec(method)
    nuobs, Tobs, env, nuFnu, E_rad, E_int, E_inj = get_shell_nuFnu_fromData(
        key, z, early_ana=EARLY_ANA, norar=law, r_cap=cap, **kw)
    variants = [(method, outdir, nuFnu, E_rad, E_int, E_inj)]
  elif method == 'fit':
    nuobs, Tobs, env, nuFnu, E_rad, E_int, E_inj = get_shell_nuFnu(key, z, **kw)
    variants = [(method, outdir, nuFnu, E_rad, E_int, E_inj)]
  else:
    raise ValueError(f"unknown method {method!r}")
  # NB: RS-normalised for BOTH shells -- the drivers build nuobs/Tobs from env.nu0
  # and env.T0 whatever z is, so a z=1 and a z=4 point of the same alpha land on the
  # SAME observer grid. sweep_shells relies on that to sum them (the FLUX units do
  # differ per shell; see _ENV_KEYS_OPT).
  Tb = 1 + (Tobs - env.Ts) / env.T0
  nub = nuobs / max(env.nu0, env.nuc)
  eff = np.nan
  for m, odir, nuFnu, E_rad, E_int, E_inj in variants:
    os.makedirs(odir, exist_ok=True)
    r = dict(log10ratio=logr, alpha=alpha, env=env, Tb=Tb, nub=nub, nuFnu=nuFnu,
             E_rad=E_rad, E_int=E_int, E_inj=E_inj, eps_e=env.eps_e, method=m,
             key=key, z=z)
    _save_point(odir, r)
    if np.isnan(eff):                       # report the reference variant's efficiency
      eff = E_rad / E_inj if E_inj > 0. else np.nan
  return dict(logr=logr, alpha=alpha, ratio=env.gma_c/env.gma_m, gma_m=env.gma_m,
              RfRS0=env.RfRS0, T0_over_alpha=env.T0/alpha, eff=eff)


def _print_point(s):
  print(f"target={s['logr']:+.1f}  alpha={s['alpha']:10.5f}  "
        f"log10(gma_c/gma_m)={np.log10(s['ratio']):+.6f}  "
        f"RfRS0={s['RfRS0']:.6f}  gma_m={s['gma_m']:.6f}  "
        f"T0/alpha={s['T0_over_alpha']:.6f}  eff={s['eff']:.4f}")


def _paired_members(method):
  '''The two methods a paired method fills, or [] if it is not paired. One place, so the
  cache bookkeeping cannot disagree with what _compute_point actually writes.'''
  return {'data+rarcut': ['data', 'data_rarcut'],
          'cap+norar_cap': [data_method_name(None, R_CAP),
                            data_method_name(NORAR_LAW, R_CAP)]}.get(method, [])


def cached_targets(outdir):
  '''Targets already cached in outdir, as the rounded keys run_sweep's point list uses.'''
  res = load_sweep(outdir)
  return set() if not res else {round(float(r['log10ratio']), 4) for r in res}


def run_sweep(key, log10ratio_arr, z=Z_SHELL, Tmax=TMAX, NT=NT,
    lognu_min=LOGNU_MIN, lognu_above=LOGNU_ABOVE_NUM, outdir=None, nproc=None,
    method=DEFAULT_METHOD, skip_cached=True):
  '''
  Run the shell nuFnu computation once per target log10(gma_c/gma_m), via the alpha
  lever (zeta=1, u_scale=1). Each point gets its own frequency window (_nu_window:
  shared low end, top anchored on that point's nu_M). The 8 points are independent;
  nproc>1 runs them across a process pool (nproc: explicit > env GAMMACM_NPROC >
  cpu_count()-1).
  method: 'data' (the reference; get_shell_nuFnu_fromData with rar_cut=None) |
  'data_rarcut' (same with the modelled cut) | 'data_norar[_prerar]' and their '_cap'
  variants (the no-rarefaction counterfactual, sweep_prerar) | the paired 'data+rarcut'
  and 'cap+norar_cap' | 'fit' (get_shell_nuFnu); outdir
  defaults to that (method, key, z) triple's directory (method_outdir) so the caches coexist.
  The first point is always computed serially to warm the (key,k) cell/fit caches (and
  the rarefaction head) so parallel workers never race on those shared-path writes.
  skip_cached: leave points already in the cache alone, so an interrupted sweep resumes
  where it stopped instead of recomputing hours of finished work (delete the point files,
  or pass False, to force a recompute). The per-point cache exists precisely for this.
  Returns the results list (rebuilt from the per-point cache).
  '''
  outdir = method_outdir(method, key, z) if outdir is None else outdir
  alpha_arr, log10ratio0 = compute_alpha_sweep(key, log10ratio_arr)
  print(f'baseline (alpha=1): log10(gma_c/gma_m) = {log10ratio0:.6f}')
  pts = list(zip(np.asarray(log10ratio_arr, float), alpha_arr))
  if skip_cached:
    # points already on disk are left alone, so an interrupted sweep resumes instead of
    # recomputing hours of finished work (sweep_efficiency.run_sweep does the same).
    # A PAIRED method writes into two directories at once, so a point counts as done only
    # when BOTH sides of the pair are there -- otherwise the second cache stays half full.
    dirs = ([method_outdir(m, key, z) for m in _paired_members(method)]
            or [outdir])
    have = set.intersection(*[cached_targets(d) for d in dirs])
    n0 = len(pts)
    pts = [p for p in pts if round(float(p[0]), 4) not in have]
    if n0 - len(pts):
      print(f'{n0 - len(pts)} of {n0} points already cached, {len(pts)} to compute')
  if not pts:
    return load_sweep(outdir)
  npr = _resolve_nproc(nproc, len(pts))
  args = lambda logr, alpha: (key, z, float(logr), float(alpha), Tmax, NT,
                              lognu_min, lognu_above, outdir, method)
  if npr == 1:
    for logr, alpha in pts:
      _print_point(_compute_point(*args(logr, alpha)))
  else:
    print(f'sweep on {npr} workers (1 serial warm-up point, then pool)')
    _print_point(_compute_point(*args(*pts[0])))          # serial: warms caches + numba
    import concurrent.futures as cf
    with cf.ProcessPoolExecutor(max_workers=npr, mp_context=_pool_context()) as ex:
      futs = {ex.submit(_compute_point, *args(logr, alpha)): logr for logr, alpha in pts[1:]}
      for fut in cf.as_completed(futs):
        _print_point(fut.result())
  return load_sweep(outdir)


def _save_point(outdir, r):
  '''Cache one sweep point (arrays + env scalars) so plotting/analysis can be
  re-run without recomputing the ~15-min sweep.'''
  cdir = os.path.join(outdir, 'cache'); os.makedirs(cdir, exist_ok=True)
  env = r['env']
  np.savez(os.path.join(cdir, f'point_logr={r["log10ratio"]:+.1f}.npz'),
      log10ratio=r['log10ratio'], alpha=r['alpha'], Tb=r['Tb'], nub=r['nub'],
      nuFnu=r['nuFnu'], E_rad=r['E_rad'], E_int=r['E_int'], E_inj=r['E_inj'],
      eps_e=r['eps_e'], method=r.get('method', 'fit'), key=r.get('key', ''),
      z=r.get('z', Z_SHELL),
      **{k: getattr(env, k) for k in _ENV_KEYS},
      **{k: getattr(env, k) for k in _ENV_KEYS_OPT if hasattr(env, k)})


def load_sweep(outdir=OUTDIR):
  '''Rebuild the results list from the cache (env replaced by a scalar namespace);
  returns None if no cache is present, or if it predates the nu_M-anchored frequency
  window (no gma_max stored): those spectra stop before their cutoff, so they are
  recomputed rather than silently re-plotted.'''
  files = sorted(glob.glob(os.path.join(outdir, 'cache', 'point_logr=*.npz')),
                 key=lambda f: float(f.split('logr=')[1].rstrip('.npz')))
  if not files:
    return None
  results = []
  for f in files:
    d = np.load(f)
    if 'gma_max' not in d.files:
      print(f'cache in {outdir}/cache predates the nu_M-anchored frequency window '
            '(no gma_max): recomputing the sweep')
      return None
    env = SimpleNamespace(**{k: float(d[k]) for k in _ENV_KEYS},
                          **{k: float(d[k]) for k in _ENV_KEYS_OPT if k in d.files})
    r = dict(log10ratio=float(d['log10ratio']), alpha=float(d['alpha']),
        env=env, Tb=d['Tb'], nub=d['nub'], nuFnu=d['nuFnu'])
    for ke in ('E_rad', 'E_int', 'E_inj', 'eps_e'):   # radiative-efficiency budget (newer caches)
      if ke in d.files:
        r[ke] = float(d[ke])
    r['method'] = str(d['method']) if 'method' in d.files else 'fit'   # provenance
    r['key'] = str(d['key']) if 'key' in d.files else ''               # '' = predates key tagging
    r['z'] = int(d['z']) if 'z' in d.files else Z_SHELL                # emitting shell
    results.append(r)
  print(f'loaded {len(results)} cached sweep points from {outdir}/cache')
  return results


def nu_over_num(r):
  '''
  Frequency axis in units of the injection frequency nu_m (= env.nu0), recovered
  from the stored nub. run_sweep builds nub = nuobs/max(nu0, nuc); since
  nuc/nu0 = (gma_c/gma_m)**2, nu/nu_m = nuobs/nu0 = nub * max(1, (gma_c/gma_m)**2).
  '''
  env = r['env']
  return r['nub'] * max(1., (env.gma_c / env.gma_m)**2)


def nu_M_over_num(r):
  '''Cutoff frequency nu_M/nu_m = (gma_M/gma_m)**2 of a sweep point, on the same
  nu/nu_m axis as nu_over_num. The frequency window is built to extend
  LOGNU_ABOVE_NUM decades past it (_nu_window).'''
  env = r['env']
  return (env.gma_max / env.gma_m)**2


# ---------------------------------------------------------------------------
# synchrotron broken-power-law pairing + apparent cooling-regime classification
# ---------------------------------------------------------------------------
def _slope_step(x, xb, dslope, s=0.4):
  '''smooth multiplicative factor: local log-log slope 0 below xb, +dslope above.'''
  ln = np.log(np.asarray(x, float) / xb)
  return np.exp(dslope * s * (np.log(0.5) + np.logaddexp(0., ln / s)))


def paired_syn_bpl(x, num, nuc, p, peak=1., xM=None, s=0.4):
  '''
  The synchrotron nuFnu broken power law that pairs with a spectrum, on
  x=nu/nu_m_collision, with BOTH breaks free: nu_m at `num` and nu_c at `nuc`
  (both in collision-nu_m units, time-dependent). Asymptotic slopes 4/3 below the
  lower break; 1/2 (nuc<num, fast cooling) or (3-p)/2 (nuc>num, slow cooling)
  between the breaks; 1-p/2 above the upper break; optional exp cutoff at xM.
  Scaled so its peak matches `peak`.
  '''
  x = np.asarray(x, float)
  b_lo, b_hi = min(num, nuc), max(num, nuc)
  mid = 0.5 if nuc < num else (3. - p)/2.
  hi = 1. - p/2.
  y = x**(4/3.) * _slope_step(x, b_lo, mid - 4/3., s) * _slope_step(x, b_hi, hi - mid, s)
  if xM is not None:
    y = y * np.exp(-x/xM)
  return y * (peak / np.nanmax(y))

SEG_MIN_MID_DEX = 1.7     # decades of MID segment required before it counts as identified.
                          # The low and high segments are ASYMPTOTES -- a spectrum can only
                          # approach them -- so MIN_DEX (0.25) is enough there; the mid slope
                          # is a value the turnover PASSES THROUGH on its way from one
                          # asymptote to the other, and a knee crossing it registers as a
                          # short spurious window. Measured on the sweep, those spurious runs
                          # reach 0.82 dex (the 1/2 window of a slow-cooling spectrum, and the
                          # (3-p)/2 window of a fast one), while every real mid segment is
                          # >= 2.00 dex and the marginal ones stop at 1.43 -- so the threshold
                          # sits inside a measured factor-1.4 gap and the classification is
                          # unchanged anywhere in 1.5-2.0. Below ~1.0 the four spectra with
                          # 1.18-1.43 dex mid windows would be named FC/SC instead of MC.
SEG_EXT = 0.5             # decades each identified segment is drawn past its own window OR
                          # past where it meets the neighbouring segment, whichever is
                          # further, so that every adjacent pair crosses visibly and a short
                          # window is still legible on a 15-decade axis. Cosmetic only: the
                          # window is what the identification and the regime rest on, and the
                          # crossing is read off the drawn lines, never computed into one.


def _seg_cross(g1, g2):
  '''log10 of the frequency where two identified segments' lines meet. For DRAWING only --
  it decides how far each line is extended so the pair crosses inside the panel; nothing in
  the identification or the regime uses it.'''
  return (g1['c'] - g2['c'])/(g2['a'] - g1['a'])


def identify_segments(x, sp, psyn, slope_tol=SLOPE_TOL, min_dex=MIN_DEX,
    min_mid_dex=SEG_MIN_MID_DEX, smooth=SLOPE_SMOOTH, min_pts=MIN_PTS):
  '''
  Which synchrotron power-law segments one nuFnu spectrum sp(x) actually shows, and the
  cooling regime that follows from the answer.

  Four candidates, each a slope the shape theory fixes -- nu^(4/3) below both breaks,
  nu^(1/2) between them when fast-cooling, nu^((3-p)/2) between them when slow, and
  nu^(1-p/2) above both. A candidate is IDENTIFIED where the local log-log slope stays
  within slope_tol of it over a wide enough run (spectral_breaks._widest_run): the slope is
  held, only the intercept is fitted, so what comes back is the segment itself -- position,
  extent and normalisation -- with nothing said about how the spectrum turns from one into
  the next. No break, no crossing and no smoothing is involved anywhere.

  The identified SET is the regime, which is the whole point of measuring it this way:

      4/3, 1/2, 1-p/2        FC    both breaks in band, fast cooling
      4/3, (3-p)/2, 1-p/2    SC    both breaks in band, slow cooling
      4/3, 1-p/2 only        MC    marginal: no mid segment survives between them, the
                                   spectrum goes from one asymptote to the other
      1/2, 1-p/2 (no 4/3)    VFC   nothing below the 1/2 segment: nu_c is off the band
      4/3, (3-p)/2 (no 1-p/2)  VSC nothing above the (3-p)/2 segment: nothing has cooled

  These are SHAPE classes -- statements about what the spectrum displays, not about
  gamma_c. measure_regime's labels answer the other question (a bin on the break ratio,
  plus gamma_c against GMA_VFC / gma_M) and build_regime_table still tabulates those; the
  two disagree where a break sits at the edge of the band, which is exactly where a break
  ratio is a guess and a missing segment is a measurement.

  VFC is the one verdict resting on a segment being ABSENT, which is a measurement only if
  the band bottom was reached, so it alone is gated on the LOW-END slope
  (spectral_breaks.edge_slope, over the lowest decade of usable band): the claim that the
  nu^(4/3) segment lies below the band requires the lowest in-band slope to still BE the 1/2
  one. A spectrum bending up toward 4/3 without having got there has its cooling break AT the
  band edge, supports no verdict, and is returned as regime None rather than guessed.
  Measured on the rarcut sweep, five of the seven VFC candidates sit at 0.50-0.60 and are
  clean; the two declined sit at 0.69 (logr=-5 rise) and 0.82 (-3 tail), the latter being the
  population edge_slope was written for. The other classes are NOT gated this way -- their
  4/3 window is identified, which is the evidence itself, and the same test would decline
  logr=-3 peak at 1.14, where a real 0.33-dex 4/3 window is only just reached and the edge
  decade still averages in the knee.

  Everything is done on the spectrum AS PLOTTED, with no cutoff division: measure_cutoff_nuM
  is called only to locate nu_M, above which the 1-p/2 candidate is not searched (the
  rolloff is not a power law). Flattening the cutoff instead -- what spectral_breaks does --
  widens the high window but leaves the identified low and mid segments bit-identical here,
  and its flattened spectrum turns back UP past nu_M, which is what made the high segment
  of every high-latitude tail unfindable.

  Returns dict(regime, segs, nuM, a_edge), or None if the spectrum is unusable. segs maps
  the name ('lo', 'fc', 'sc', 'hi') to dict(a, c, x0, x1, dex): slope, intercept in log10,
  the window it was identified over, and its width in decades. regime is None when the
  identified set matches no case above, or when the low end supports no verdict.
  '''
  x = np.asarray(x, float); sp = np.asarray(sp, float)
  cut = measure_cutoff_nuM(x, sp, psyn, flatten=False)
  if not cut['ok']:
    return None
  lx, ly, s = segment_slopes(x, sp, smooth)
  if len(lx) < 8 or not np.any(np.isfinite(s)):
    return None
  i_pk = int(np.argmax(ly))
  idx = np.arange(len(lx))
  keep = (ly > ly.max() - FIT_DEC) & (lx < np.log10(cut['nuM']/CUT_FAC))
  segs = {}
  for name, a, below, mdex in (('lo', 4./3., True, min_dex),
                               ('fc', 0.5, True, min_mid_dex),
                               ('sc', (3. - psyn)/2., True, min_mid_dex),
                               ('hi', 1. - psyn/2., False, min_dex)):
    m = keep & np.isfinite(s) & (np.abs(s - a) < slope_tol)
    m &= (idx < i_pk) if below else (idx > i_pk)
    w = _widest_run(m, lx, min_pts, mdex)
    if w is None:
      continue
    i, j = w
    segs[name] = dict(a=a, c=float(np.mean(ly[i:j+1] - a*lx[i:j+1])),
                      x0=float(10**lx[i]), x1=float(10**lx[j]), dex=float(lx[j] - lx[i]))
  # a spectrum has ONE mid segment: if the slope lingers at both candidates, the wider run
  # is the segment and the other is a knee (never fires on this sweep -- the two windows are
  # 0.24-0.46 and 0.27-0.36 dex where they coexist, i.e. both are rejected anyway)
  if 'fc' in segs and 'sc' in segs:
    segs.pop('sc' if segs['fc']['dex'] >= segs['sc']['dex'] else 'fc')
  # VFC is the one verdict that rests on a segment being ABSENT, so it is gated on the band
  # bottom actually being converged to the 1/2 it claims. The other classes have their 4/3
  # window identified -- that IS the evidence -- and must NOT be gated the same way: the same
  # test applied to them declines logr=-3 peak, whose edge slope is 1.14 because its 0.33-dex
  # 4/3 window is only just reached, over a decade that still averages in the knee.
  a_edge = edge_slope(x, sp, cut['nuM'])
  vfc_ok = bool(np.isfinite(a_edge) and abs(a_edge - 0.5) <= EDGE_VFC_TOL)
  has = set(segs)
  if   {'lo', 'fc', 'hi'} <= has:      regime = 'FC'
  elif {'lo', 'sc', 'hi'} <= has:      regime = 'SC'
  elif {'lo', 'hi'} <= has:            regime = 'MC'
  elif {'fc', 'hi'} <= has and vfc_ok: regime = 'VFC'
  elif {'lo', 'sc'} <= has:            regime = 'VSC'
  else:                                regime = None
  return dict(regime=regime, segs=segs, nuM=float(cut['nuM']), a_edge=float(a_edge))


# ---------------------------------------------------------------------------
# Granot & Sari (2002) spectral shape: fit to the computed instantaneous spectra
# ---------------------------------------------------------------------------
# GS02 tabulate FIXED smoothing exponents per break rather than fitting them, and that
# practice transfers here: over the 24 rise/peak/tail spectra of the sweep, the best single
# pair costs only +0.0013 dex in mean rms against fitting s1, s2 freely on every spectrum,
# and the rms landscape is flat (0.1111-0.1266 over s1 in [0.7,2.4], s2 in [1,4]) -- s is
# weakly constrained, the same degeneracy that makes free-smoothing break fits unreliable
# (see SEP_UNRESOLVED). Values below are the grid minimum, RE-DERIVED with GS02_CUTOFF: the
# earlier (1.0, 1.8) was calibrated against an exp cutoff, whose error the smoothing was
# partly absorbing. NB: GS02's convention is INVERTED relative to _slope_step -- here a
# larger s is a SHARPER break.
GS02_S1, GS02_S2 = 1.3, 2.0
# Cutoff shape, passed straight to granot_sari_syn: 'R' is the true single-electron
# synchrotron emissivity (the nuFnu spectrum rolls over like P'_nu') and is its default;
# 'exp' is the cruder exp(-nu/nu_M), kept as the reference the table compares against.
GS02_CUTOFF = 'R'
GS02_FIT_DEC = 5.0        # fit the top this many decades of the spectrum: below that the
                          # flux is off the plotted range and dominated by the far tail
GS02_BOUND_TOL = 1e-3     # a fitted break this close (in dex) to its bound is not measured
GS02_BODY_FAC = 10.       # rms is also reported over the spectral BODY alone, nu < nu_M/this.
                          # The split matters: the GS02 broken-power-law part fits the body to
                          # ~0.02 dex (5% in flux) while the total rms is ~0.13, i.e. the
                          # residual is dominated by the exp(-nu/nu_M) cutoff, which is NOT
                          # part of GS02 (it is our addition) and is a cruder stand-in for the
                          # synchrotron kernel's true rolloff.


GS02_BMID_TOL = 0.02      # a free mid-slope this close to an asymptote counts as that regime
GS02_S_VFC = 2.0          # smoothing of the VERY-fast-cooling single break (grid minimum over
                          # the 10 out-of-band spectra of the sweep; 0.0383 at 2.0, flat to
                          # 0.039 over 1.7-2.6). Sharper than the two-break values because it
                          # joins -1/2 straight to -p/2, a smaller slope change.


def fit_gs02_spectrum(x, sp, psyn, nuM, s=(GS02_S1, GS02_S2), free_s=False,
    free_nuM=True, fit_dec=GS02_FIT_DEC, cutoff=GS02_CUTOFF, free_bmid=False):
  '''
  Fit the Granot & Sari (2002) shape (phys_functions.granot_sari_syn) to ONE nuFnu
  spectrum sp(x), x = nu/nu_m_collision. Free parameters: the two breaks and F_ext,
  plus s1, s2 if free_s and nuM if free_nuM.

  The breaks are parameterised as (b_lo, b_hi = b_lo*delta) with delta >= 1, so their
  ordering is guaranteed and the FIT ITSELF chooses the cooling regime: the run is done
  twice, once assigning (nu_m, nu_c) = (b_lo, b_hi) (slow cooling, middle F_nu slope
  -(p-1)/2) and once (b_hi, b_lo) (fast, -1/2), keeping whichever has the lower rms. That
  makes the regime an OUTPUT of the shape fit rather than a threshold on a measured slope
  (measure_regime's s_mid flips spuriously; see track_breaks).

  nuM is FITTED by default, and the `nuM` argument is only the starting value / bound
  centre. The burnoff frequency is B-independent and identical for every sweep point --
  but only in the COMOVING frame: what reaches the observer is Doppler-slid, so the
  spectrum at bar{T} cuts off at nu_M(t), not at its collision value (nu_M(t) is flat
  while the shell emits, then goes as bar{T}^-1 in the high-latitude tail; see
  track_breaks/nu_Mt). Holding it fixed costs a factor ~6 in the tail (rms 0.22 vs
  0.037) and nothing on the rise, where the fitted value comes back at 0.99-1.02 of the
  known one -- an independent recovery of the burnoff limit to a couple of percent.
  The fitted nuM/nominal ratio is therefore a measurement of the Doppler slide.
  Pass free_nuM=False to hold it.

  Returns a dict: rms (in log10 flux), rms_alt (the rms of the REJECTED ordering, so a
  caller can tell a real preference from a tie), num, nuc, b_lo, b_hi, regime, F_ext,
  s1, s2, nuM, npts, and at_bound (a break ran into the frequency window's edge, i.e. it is outside the
  data and unconstrained -- expected deep in fast cooling, where nu_c is off-window).
  '''
  x = np.asarray(x, float); sp = np.asarray(sp, float)
  good = np.isfinite(sp) & (sp > 0.) & np.isfinite(x) & (x > 0.)
  xg, yg = x[good], np.log10(sp[good])
  keep = yg > yg.max() - fit_dec
  xg, yg = xg[keep], yg[keep]
  if len(xg) < 12:
    return None
  y0 = yg.max(); yg = yg - y0                     # fit the shape; F_ext carries the scale

  lo_b, hi_b = np.log10(xg.min()) - 0.5, np.log10(xg.max()) + 0.5
  span = hi_b - lo_b
  x_pk = xg[int(np.argmax(yg))]

  def unpack(q):
    lb, ld, lA = q[0], q[1], q[2]
    s1, s2 = (10**q[3], 10**q[4]) if free_s else s
    nM = 10**q[-1] if free_nuM else nuM
    return 10**lb, 10**(lb + ld), 10**lA, s1, s2, nM

  # free_bmid: one fit with the middle F_nu slope free between the two asymptotes, instead
  # of two fits at the asymptotes. A free beta_mid pins itself at exactly -1/2 (fast) or
  # -(p-1)/2 (slow) in 23 of the 24 rise/peak/tail spectra -- so the binary regime choice is
  # right, not a modelling limitation. It lands BETWEEN them only around the moment nu_c(t)
  # crosses nu_m(t), whichever point is crossing then (log10(gc/gm)=-1 at rise, -0.611; the
  # =0 point late in its tail, -0.67/-0.56), and there neither break is cleanly nu_c or nu_m
  # -- reported as regime 'MC'. NB the merged single-break spectrum is NOT the right shape
  # for that moment: it is 1.6-12x worse in rms even at the smallest separations the sweep
  # reaches (~3), because the two breaks never actually coincide.
  bm_fc, bm_sc = -0.5, -(psyn-1.)/2.
  if free_bmid:
    def resid_bm(q):
      b_lo, b_hi, A, s1, s2, nM = unpack(q[:-1])
      mod = granot_sari_syn(xg, b_lo, b_hi, psyn, s1=s1, s2=s2, nuM=nM, F_ext=A,
                            nuFnu=True, cutoff=cutoff, beta_mid=q[-1])
      return np.log10(np.maximum(mod, 1e-300)) - yg
    q0 = [np.log10(x_pk) - 1., 1., 0.] + ([np.log10(s[0]), np.log10(s[1])] if free_s else []) \
         + ([np.log10(nuM)] if free_nuM else []) + [0.5*(bm_fc + bm_sc)]
    blo = [lo_b, 0., -8.] + ([-1., -1.] if free_s else []) + ([lo_b] if free_nuM else []) + [bm_sc]
    bhi = [hi_b, span, 8.] + ([1.3, 1.3] if free_s else []) + ([hi_b + 2.] if free_nuM else []) + [bm_fc]
    r = least_squares(resid_bm, q0, bounds=(blo, bhi))
    rms = float(np.sqrt(np.mean(r.fun**2)))
    b_lo, b_hi, A, s1, s2, nM = unpack(r.x[:-1])
    bmid = float(r.x[-1])
    if abs(bmid - bm_fc) < GS02_BMID_TOL:   regime = 'FC'
    elif abs(bmid - bm_sc) < GS02_BMID_TOL: regime = 'SC'
    else:                                   regime = 'MC'
    # ambiguous mid slope -> neither break is cleanly nu_c or nu_m; say so rather than
    # forcing a name on them (that is what produces the jump in nu_c(t) at the crossing)
    num = nuc = np.nan
    if regime == 'FC':   num, nuc = b_hi, b_lo
    elif regime == 'SC': num, nuc = b_lo, b_hi
    best_bm = (rms, np.nan, regime, b_lo, b_hi, A, s1, s2, nM, bmid, num, nuc,
               bool(min(abs(r.x[0] - lo_b), abs(r.x[0] - hi_b)) < GS02_BOUND_TOL))
  else:
    best_bm = None

  best = None; tried = {}
  for regime in ([] if free_bmid else ('SC', 'FC')):
    def resid(q):
      b_lo, b_hi, A, s1, s2, nM = unpack(q)
      num, nuc = (b_lo, b_hi) if regime == 'SC' else (b_hi, b_lo)
      mod = granot_sari_syn(xg, num, nuc, psyn, s1=s1, s2=s2, nuM=nM, F_ext=A,
                            nuFnu=True, cutoff=cutoff)
      return np.log10(np.maximum(mod, 1e-300)) - yg
    q0 = [np.log10(x_pk) - 1., 1., 0.]
    blo, bhi = [lo_b, 0., -8.], [hi_b, span, 8.]
    if free_s:
      q0 += [np.log10(s[0]), np.log10(s[1])]; blo += [-1., -1.]; bhi += [1.3, 1.3]
    if free_nuM:
      q0 += [np.log10(nuM)]; blo += [lo_b]; bhi += [hi_b + 2.]
    try:
      r = least_squares(resid, q0, bounds=(blo, bhi))
    except ValueError:
      continue
    rms = float(np.sqrt(np.mean(r.fun**2)))
    tried[regime] = rms
    if best is None or rms < best[0]:
      best = (rms, r.x, regime)
  if best is None and best_bm is None:
    return None

  if best_bm is not None:
    (rms, rms_alt, regime, b_lo, b_hi, A, s1, s2, nM, bmid, num, nuc, at_bound) = best_bm
  else:
    rms, q, regime = best
    rms_alt = tried.get('FC' if regime == 'SC' else 'SC', np.nan)

  # VERY fast cooling: the fit put nu_c BELOW the observed frequencies, so no curvature at
  # the low end was ever measured -- the "break" is a free shape knob outside the data and
  # the FC label it implies is unsupported. Physically this is gamma_c small enough that the
  # whole distribution has cooled: nu_c (and the 4/3 segment under it) sit below the band,
  # leaving a SINGLE observable break, -1/2 straight to -p/2. Refit that shape and report
  # regime 'VFC' with nu_c only as an upper bound. Judged on where the break lands, not on
  # rms: the two-break form can score better here precisely BY bending the in-band spectrum
  # with an unobserved break (0.025 vs 0.037 at log10(gc/gm)=-5 rise), which is overfitting.
  if min(b_lo, b_hi) < xg.min() if best_bm is not None else \
     min(unpack(q)[0], unpack(q)[1]) < xg.min():
    def resid_vfc(qv):
      mod = granot_sari_syn(xg, 10**qv[0], None, psyn, s2=GS02_S_VFC, nuM=10**qv[2],
                            F_ext=10**qv[1], nuFnu=True, cutoff=cutoff)
      return np.log10(np.maximum(mod, 1e-300)) - yg
    rv = least_squares(resid_vfc, [np.log10(x_pk), 0., np.log10(nuM)],
                       bounds=([lo_b, -8., lo_b], [hi_b, 8., hi_b + 2.]))
    rms_v = float(np.sqrt(np.mean(rv.fun**2)))
    nb, Av, nMv = 10**rv.x[0], 10**rv.x[1], 10**rv.x[2]
    d = rv.fun; body = xg < nMv/GS02_BODY_FAC
    return dict(rms=rms_v, rms_alt=rms, rms_body=(float(np.sqrt(np.mean(d[body]**2)))
                                                 if body.sum() >= 5 else np.nan),
                num=nb, nuc=np.nan, nuc_max=float(xg.min()), b_lo=nb, b_hi=nb,
                regime='VFC', beta_mid=-0.5, F_ext=Av*10**y0, s1=GS02_S_VFC, s2=GS02_S_VFC,
                nuM=nMv, npts=int(len(xg)),
                at_bound=bool(min(abs(rv.x[0] - lo_b), abs(rv.x[0] - hi_b)) < GS02_BOUND_TOL),
                cutoff=cutoff)
  if best_bm is None:
    b_lo, b_hi, A, s1, s2, nM = unpack(q)
    num, nuc = (b_lo, b_hi) if regime == 'SC' else (b_hi, b_lo)
    bmid = bm_sc if regime == 'SC' else bm_fc
    at_bound = bool(min(abs(q[0] - lo_b), abs(q[0] - hi_b)) < GS02_BOUND_TOL
                    or abs(q[0] + q[1] - hi_b) < GS02_BOUND_TOL)
  # rms over the body alone, i.e. excluding the cutoff region: separates how well GS02's
  # broken-power-law form does from how well our exp(-nu/nu_M) stands in for the rolloff
  mod = granot_sari_syn(xg, b_lo, b_hi, psyn, s1=s1, s2=s2, nuM=nM, F_ext=A,
                        nuFnu=True, cutoff=cutoff, beta_mid=bmid)
  body = xg < nM/GS02_BODY_FAC
  d = np.log10(np.maximum(mod, 1e-300)) - yg
  rms_body = float(np.sqrt(np.mean(d[body]**2))) if body.sum() >= 5 else np.nan
  return dict(rms=rms, rms_alt=rms_alt, rms_body=rms_body, num=num, nuc=nuc, b_lo=b_lo,
              b_hi=b_hi, regime=regime, beta_mid=bmid, F_ext=A*10**y0, s1=s1, s2=s2, nuM=nM,
              npts=int(len(xg)), at_bound=at_bound, cutoff=cutoff)


def gs02_model(x, fit, psyn):
  '''
  The fitted GS02 nuFnu on an arbitrary x grid, from a fit_gs02_spectrum result. A VFC fit
  is rebuilt with nuc=None -- passing its nuc (NaN) straight through would silently give a
  two-break spectrum with a 1/3 low-frequency segment, i.e. NOT the shape that was fitted.
  '''
  nuc = None if fit['regime'] == 'VFC' else fit['nuc']
  return granot_sari_syn(x, fit['num'], nuc, psyn, s1=fit['s1'], s2=fit['s2'],
                         nuM=fit['nuM'], F_ext=fit['F_ext'], nuFnu=True,
                         cutoff=fit.get('cutoff', GS02_CUTOFF),
                         beta_mid=fit.get('beta_mid'))


# FC (mid-segment slope 1/2) vs SC ((3-p)/2) divide, empirically shifted below the
# theoretical midpoint because shell integration softens the mid segment.
SMID_FCSC = 0.30

GMA_VFC = 1     # VFC boundary: gamma_c << 1, read strictly as one decade below the
                  # gamma_c = 1 floor (electrons cannot cool past gamma = 1)


def measure_regime(x, sp, p, env):
  '''
  Measure the cooling regime of one nuFnu spectrum sp(x), x=nu/nu_m_collision, from
  its TWO breaks -- the peak (= max(nu_m,nu_c) at the emission time) and the lower
  knee to the nu^4/3 segment (= min(nu_m,nu_c)). Which knee is nu_m is set by the
  mid-segment slope (~1/2 FC, ~(3-p)/2 SC). The regime is nu_c(t)/nu_m(t) = the
  break RATIO -- independent of the static collision nu_m, so it tracks the physical
  nu_m as it drops with time (hard-to-soft). FC/marginal/SC are bins on that ratio,
  but the two EXTREME regimes are physical statements about gamma_c (hence env, which
  supplies gma_m, gma_c and the injection gma_M): VFC is gamma_c < GMA_VFC, VSC is
  gamma_c > gma_M. All returned frequencies are in units of the collision nu_m.
  Returns num_t (nu_m(t)), nuc_t (nu_c(t)), ratio=nuc_t/num_t, gc_t (gamma_c(t)),
  off (a break fell outside the window), s_mid, s_above, x_pk, regime.
  '''
  x = np.asarray(x, float); sp = np.asarray(sp, float)
  good = np.isfinite(sp) & (sp > 0.) & np.isfinite(x) & (x > 0.)
  x, sp = x[good], sp[good]
  lx, ly = np.log10(x), np.log10(sp)
  s = np.gradient(ly, lx)                                  # local log-log slope
  i_pk = int(np.argmax(sp)); x_pk = float(x[i_pk])

  def slope_win(xa, xb):
    xa, xb = max(xa, x.min()), min(xb, x.max())
    m = (x >= xa) & (x <= xb)
    if m.sum() >= 3:
      return float(np.polyfit(lx[m], ly[m], 1)[0])
    xc = np.sqrt(max(xa, x.min()) * min(xb, x.max()))     # fallback: local slope at centre
    return float(s[int(np.argmin(np.abs(x - xc)))])
  s_above = slope_win(x_pk*2., x_pk*10.)                   # 1-p/2 segment (check)

  # lower break: scan the slope down from the peak to where it steepens to ~4/3
  KNEE = 0.5*(0.5 + 4/3.)                                  # ~0.92, between mid (0.25-0.5) and 4/3
  below = x < x_pk/1.1
  x_lo, off = None, False
  if below.sum() >= 2:
    xb, sb = x[below][::-1], s[below][::-1]                # scan downward in x
    cross = np.where(sb >= KNEE)[0]
    if len(cross):
      x_lo = float(xb[cross[0]])
  if x_lo is None:
    x_lo, off = float(x.min()), True                      # other break below the window

  # mid-segment slope (between the two breaks) -> FC (~1/2) vs SC (~(3-p)/2)
  s_mid = slope_win(x_lo*1.5, x_pk/1.5)
  fast = s_mid >= SMID_FCSC
  if not off and x_pk/x_lo < 2.:                           # breaks merged -> marginal
    num_t = nuc_t = float(np.sqrt(x_pk*x_lo))
  elif fast:                                               # FC: nu_m = peak, nu_c = lower break
    num_t, nuc_t = x_pk, x_lo
  else:                                                    # SC: nu_c = peak, nu_m = lower break
    num_t, nuc_t = x_lo, x_pk
  ratio = nuc_t/num_t                                      # = nu_c(t)/nu_m(t)

  # gamma_c(t) = gma_m*sqrt(ratio): the break RATIO (not nuc_t alone) cancels the B*D
  # drift of the static collision-nu_m units, since ratio = (gamma_c(t)/gamma_m)**2.
  # With the lower break off-grid only a bound was measured -> use the exact shell-wide
  # env.gma_c. VSC compares with the INJECTION gma_M: gma_M(t) burns down as ~1/t and
  # converges to gamma_c(t), so a time-dependent threshold would be vacuous -- VSC means
  # nothing cooled over the shell's dynamical time.
  gc_t = env.gma_c if off else env.gma_m*np.sqrt(ratio)
  lr = np.log10(ratio)                      # marginal band at |log10(nu_c/nu_m)| = 0.5
  if gc_t > env.gma_max: regime = 'VSC'
  elif gc_t < GMA_VFC:   regime = 'VFC'
  elif lr < -0.5:        regime = 'FC'
  elif lr <= 0.5:        regime = 'marginal'
  else:                  regime = 'SC'
  return dict(x_pk=x_pk, num_t=num_t, nuc_t=nuc_t, ratio=ratio, gc_t=gc_t, off=off,
              s_mid=s_mid, s_above=s_above, regime=regime)


# ---------------------------------------------------------------------------
# time-resolved break tracking: nu_c(t), nu_m(t) over the WHOLE lightcurve
# ---------------------------------------------------------------------------
TRACK_FLUX_FLOOR = 1e-10   # time bins whose brightest nuFnu is below this fraction of the
                           # lightcurve peak carry no measurable spectrum (the earliest bins
                           # are identically zero: nothing has been shocked yet)
EDGE_FAC = 2.              # a break closer than this factor to a frequency-window edge is a
                           # clamp, not a measurement -> flagged invalid
EDGE_FAC_M = 10.           # margin on the CUTOFF side, wider because the failure there is not a
                           # clamp but a misread: while nu_c > nu_M nothing has cooled and the
                           # spectrum has NO cooling break, so measure_regime's upper break is
                           # the exponential cutoff itself. Measured on the slow-cooling points,
                           # that misread saturates at ~nu_M/3.5, so 10 clears it
SWAP_RISE = 2.             # the nu_c/nu_m role swap is taken at the MINIMUM of the break
                           # separation, and only if the separation then re-widens by this
                           # factor (see track_breaks)
SEP_UNRESOLVED = 12.       # below this DETECTED separation the two breaks are not resolved and
                           # neither is measured (only their blend). measure_regime's knee scan
                           # has a floor that grows with how smooth the break is, calibrated by
                           # feeding it paired_syn_bpl spectra of KNOWN separation: a true ratio
                           # of 1 comes back as 5.4 at s=0.6, 9.4 at s=0.8, 15.3 at s=1.0, and
                           # true ratios up to 3 are indistinguishable from 1. The shell-
                           # integrated spectra sit at s ~ 0.8-1.1 near the crossing, so a
                           # detected ~9 there means "unresolved", NOT two breaks a decade apart.
                           # 12 is the floor at s=0.8; detected separations just above it (the
                           # log10(gc/gm)=0 point plateaus at 16) are weak measurements, closer
                           # to lower bounds than to values.
CUTOFF_SLOPE = -1.5        # local log-log slope defining the point measured on the spectral
                           # rolloff (see _measure_cutoff); anything well below the asymptotic
                           # 1-p/2 works, the estimator divides the calibration back out
RISE_WIN = (3e-3, 0.3)     # bar{T} window of the constant-hydro cooling law: starts where the
                           # shell is bright enough for both breaks to be measured, ends where
                           # the cells have expanded enough to soften the decay (fit_break_evolution)


def _measure_cutoff(x, sp, psyn, slope_cut=CUTOFF_SLOPE, max_dec=6.):
  '''
  Position of the spectral cutoff nu_M(t) of one spectrum, from the shape of the rolloff.
  Above the upper break the spectrum is x^(1-p/2) times the burnoff tail ~exp(-x/nu_M), so
  its local slope is (1-p/2) - x/nu_M: the frequency at which that slope reaches slope_cut
  sits at a FIXED multiple k = (1-p/2) - slope_cut of nu_M, whatever the flux level there.
  Scans up from the peak to the first crossing; NaN if the rolloff is not on the grid or is
  more than max_dec decades below the peak (numerical dust). The absolute scale carries the
  ~25% error of treating the synchrotron kernel's tail as a pure exponential -- the TIME
  EVOLUTION, which is what this is for, does not.
  '''
  g = np.isfinite(sp) & (sp > 0.)
  if g.sum() < 20:
    return np.nan
  lx, ly = np.log10(x[g]), np.log10(sp[g])
  s = np.gradient(ly, lx)
  i_pk = int(np.argmax(ly))
  hit = np.flatnonzero((np.arange(len(lx)) > i_pk) & (s <= slope_cut))
  if not hit.size or ly[hit[0]] < ly[i_pk] - max_dec:
    return np.nan
  return float(10**lx[hit[0]] / ((1. - psyn/2.) - slope_cut))


def track_breaks(r, flux_floor=TRACK_FLUX_FLOOR):
  '''
  Run measure_regime on EVERY time bin of a sweep point and follow the two spectral
  breaks through the burst. All frequencies in units of the collision nu_m (the
  nu_over_num axis). Returns arrays over bar{T} = Tb-1: nu_lo, nu_hi (the two breaks,
  sorted), nu_c, nu_m, sep = nu_hi/nu_lo, ratio = nu_c/nu_m, s_mid, off, valid (the
  mask to fit and plot on), and nu_Mt, the cutoff MEASURED per time bin (_measure_cutoff)
  -- as opposed to the scalar nu_M, its nominal collision value (gma_M/gma_m)^2. Plus
  i_swap and nu_B (the gamma=1 floor).

  measure_regime's OWN nu_c/nu_m assignment cannot be used bin by bin: it labels the
  breaks from the mid-segment slope (s_mid vs SMID_FCSC), which flips spuriously
  wherever that slope drifts across the threshold -- at log10(gc/gm)=0 the softening
  tail reaches s_mid=0.33 and the labels swap, putting a factor-30 vertical jump in
  nu_c(t) around bar{T}~0.5 where nothing physical happens. Here the roles are assigned
  by CONTINUITY: every regime starts slow-cooling (nothing has cooled yet, so nu_c is
  the UPPER break), and nu_c hands over to the lower break exactly once, at the FC/SC
  crossing -- located as the MINIMUM of the detected separation, and taken only if the
  breaks then re-widen by SWAP_RISE (at log10(gc/gm) >= 0 the separation decreases
  monotonically to its frozen high-latitude value and no crossing is detected).

  Around that minimum the two breaks are NOT resolved. The detected separation bottoms
  out at ~9 in every crossing point, which is the floor of measure_regime's knee scan at
  the smoothing these spectra have, not a real separation: fed a synthetic spectrum whose
  two breaks sit at the SAME frequency, it returns 9.4 at s=0.8 (see SEP_UNRESOLVED).
  Bins below that floor are therefore reported as one blended break,
  nu_c = nu_m = sqrt(nu_lo*nu_hi), and dropped from `valid` -- what the crossing does to
  the true breaks (merge, or stay a factor of a few apart) is below what this method can
  see, so it is flagged rather than asserted.
  '''
  x = nu_over_num(r); env = r['env']; barT = r['Tb'] - 1.
  nuFnu = r['nuFnu']; n = len(barT)
  Fpk = np.nanmax(nuFnu, axis=1)
  nu_lo = np.full(n, np.nan); nu_hi = np.full(n, np.nan); s_mid = np.full(n, np.nan)
  nu_Mt = np.full(n, np.nan)
  off = np.ones(n, bool)
  # measure_regime needs a few positive points to differentiate: the pre-onset bins are empty
  bright = (np.isfinite(Fpk) & (Fpk > flux_floor*np.nanmax(Fpk))
            & ((np.isfinite(nuFnu) & (nuFnu > 0.)).sum(axis=1) >= 5))
  for i in np.flatnonzero(bright):
    m = measure_regime(x, nuFnu[i, :], env.psyn, env)
    nu_lo[i], nu_hi[i] = sorted((m['num_t'], m['nuc_t']))
    s_mid[i] = m['s_mid']; off[i] = m['off']
    nu_Mt[i] = _measure_cutoff(x, nuFnu[i, :], env.psyn)
  meas = bright & ~off                                    # both breaks actually measured
  sep = nu_hi/nu_lo

  fast = np.zeros(n, bool); i_swap = None
  idx = np.flatnonzero(meas & np.isfinite(sep))
  if idx.size > 3:
    j = idx[int(np.argmin(sep[idx]))]
    after = idx[idx > j]
    if after.size and np.nanmax(sep[after]) > SWAP_RISE*sep[j]:
      i_swap = int(j); fast[j:] = True
  nu_c = np.where(fast, nu_lo, nu_hi)
  nu_m = np.where(fast, nu_hi, nu_lo)
  # unresolved stretch: report the blend, not two breaks
  unres = meas & np.isfinite(sep) & (sep < SEP_UNRESOLVED)
  blend = np.sqrt(nu_lo*nu_hi)
  nu_c = np.where(unres, blend, nu_c)
  nu_m = np.where(unres, blend, nu_m)

  nu_M = nu_M_over_num(r)
  valid = (meas & ~unres & np.isfinite(nu_c) & (nu_c > EDGE_FAC*x.min())
           & (nu_c < x.max()/EDGE_FAC) & (nu_c < nu_M/EDGE_FAC_M))
  return dict(barT=barT, nu_lo=nu_lo, nu_hi=nu_hi, nu_c=nu_c, nu_m=nu_m, sep=sep,
              ratio=nu_c/nu_m, s_mid=s_mid, off=off, unres=unres, valid=valid,
              valid_m=valid, is_vfc=np.zeros(n, bool), sep_unres=SEP_UNRESOLVED,
              i_swap=i_swap, nu_M=nu_M, nu_Mt=nu_Mt, nu_B=1./env.gma_m**2,
              nu_win=(float(x.min()), float(x.max())), Fpk=Fpk)


GS02_TRACK_RMSMAX = 0.15   # a time bin whose GS02 fit is worse than this is not a measurement
GS02_TRACK_SEPMIN = 1.5    # below this fitted break separation the two are effectively one
GS02_REGIME_MARGIN = 0.02  # the FC/SC ordering is only taken as decided when the rejected
                           # ordering fits this much worse in relative rms; below it the two
                           # are a tie and the branch assignment is held (see track_breaks_gs02)


def track_breaks_gs02(r, flux_floor=TRACK_FLUX_FLOOR, rms_max=GS02_TRACK_RMSMAX,
    barT_swap_max=None):
  '''
  Follow nu_m, nu_c and nu_M through the burst by FITTING the Granot & Sari shape to
  every instantaneous spectrum (fit_gs02_spectrum), rather than reading knees off it.
  Same output contract as track_breaks, so the break-evolution plots and
  fit_break_evolution take either.

  Better than the knee scan on every count that mattered:
    - the breaks are fitted, so they do not carry its position bias (measured on
      synthetic spectra: 0.73 low-side, 1.5-2.7 high-side). Here the early-time nu_m
      comes back at 1.00-1.02 of the collision value, as it should;
    - the FC/SC assignment is whichever of the two orderings fits better over the whole
      spectrum, instead of a threshold on a local slope. It is still held monotone in
      time and only acted on where the data decides (GS02_REGIME_MARGIN), because the
      two orderings go degenerate once the tail softens the mid segment;
    - the separation is a real measurement, not a resolution floor: it bottoms out at
      ~5 at log10(gc/gm)=-2 rather than the knee scan's universal ~9 (see
      SEP_UNRESOLVED). Caveat: s1, s2 are held fixed, so any mismatch in the true
      smoothing is absorbed by the separation;
    - nu_M comes out of the same fit (nu_Mt) instead of a slope-threshold estimator.
  '''
  x = nu_over_num(r); env = r['env']; barT = r['Tb'] - 1.
  nuFnu = r['nuFnu']; n = len(barT)
  nuM_nom = nu_M_over_num(r)
  Fpk = np.nanmax(nuFnu, axis=1)

  b_lo = np.full(n, np.nan); b_hi = np.full(n, np.nan); nu_Mt = np.full(n, np.nan)
  sep = np.full(n, np.nan); rms = np.full(n, np.nan); pref = np.full(n, np.nan)
  fast_fit = np.zeros(n, bool); ok = np.zeros(n, bool); is_vfc = np.zeros(n, bool)
  ambig = np.zeros(n, bool); bmid = np.full(n, np.nan)
  bright = (np.isfinite(Fpk) & (Fpk > flux_floor*np.nanmax(Fpk))
            & ((np.isfinite(nuFnu) & (nuFnu > 0.)).sum(axis=1) >= 12))
  for i in np.flatnonzero(bright):
    f = fit_gs02_spectrum(x, nuFnu[i, :], env.psyn, nuM_nom, free_bmid=True)
    if f is None:
      continue
    b_lo[i], b_hi[i], nu_Mt[i] = f['b_lo'], f['b_hi'], f['nuM']
    rms[i] = f['rms']; bmid[i] = f['beta_mid']
    is_vfc[i] = (f['regime'] == 'VFC')
    if not is_vfc[i]:
      sep[i] = f['b_hi']/f['b_lo']
      fast_fit[i] = (f['regime'] == 'FC')
      ambig[i] = (f['regime'] == 'MC')
    ok[i] = (not f['at_bound']) and f['rms'] <= rms_max

  # Which break is nu_c: the fit's own choice, but ONLY where the data actually prefers one
  # ordering. SC and FC differ just in the mid-segment slope ((3-p)/2 vs 1/2), and once the
  # tail softens that segment the two become degenerate -- at log10(gc/gm)=0 the rms of the
  # two orderings agree to 4 digits, and taking the per-bin argmin there flips the labels in
  # the middle of the high-latitude tail, swapping nu_c and nu_m and scrambling their fitted
  # slopes. So: hold the branch through ties, switch at most once, SC -> FC (nu_c only ever
  # falls through nu_m, never back).
  # The FC/SC identification now comes from the FITTED mid slope, which moves continuously
  # from -(p-1)/2 to -1/2 through the crossing instead of flipping between them. Bins where
  # it sits between the asymptotes ('MC') are the crossing itself: there neither break is
  # cleanly nu_c or nu_m, so neither is named. Forcing a name on them is what used to put an
  # unphysical factor-of-several jump into nu_c(t) -- the fitted branches b_lo, b_hi are
  # smooth through the crossing, only the labels were not.
  # A bin is only UNNAMEABLE if it sits in the transition itself, i.e. between the last
  # spectrum the fit calls slow-cooling and the first it calls fast-cooling. An intermediate
  # beta_mid on its own does not make the identification ambiguous: in the high-latitude tail
  # the mid segment softens away from its asymptote while the two breaks stay a decade apart
  # and their ordering never comes into question (log10(gc/gm)=0 does this for its whole
  # tail), and there the assignment follows by continuity.
  in_win = (barT <= barT_swap_max) if barT_swap_max is not None else np.ones(n, bool)
  fc_bins = np.flatnonzero(ok & fast_fit & in_win)
  fast = np.zeros(n, bool); ambig_ident = np.zeros(n, bool)
  if fc_bins.size:
    i_fc = int(fc_bins[0])
    fast[i_fc:] = True
    sc_before = np.flatnonzero(ok & (~fast_fit) & (~ambig) & (np.arange(n) < i_fc))
    i_sc = int(sc_before[-1]) if sc_before.size else 0
    ambig_ident[i_sc+1:i_fc] = True          # the crossing: named neither way
  ambig = ambig & ambig_ident
  nu_c = np.where(fast, b_lo, b_hi)
  nu_m = np.where(fast, b_hi, b_lo)
  # crossing: branches measured, identification not
  nu_c = np.where(ambig, np.nan, nu_c)
  nu_m = np.where(ambig, np.nan, nu_m)
  # VFC: only one break is observed and it is nu_m; nu_c is below the band, bounded not
  # measured. Keep nu_m (it is a real measurement) and drop nu_c.
  nu_c = np.where(is_vfc, np.nan, nu_c)
  nu_m = np.where(is_vfc, b_lo, nu_m)

  # a break the fit places outside the observed window is an extrapolation of the shape,
  # not a measurement -- however well the fit does over the data. Deep in fast cooling nu_c
  # sits below the window and must be dropped, exactly as in the knee-scan tracker.
  with np.errstate(invalid='ignore'):
    inwin = (np.minimum(nu_c, nu_m) > EDGE_FAC*x.min()) \
            & (np.maximum(nu_c, nu_m) < x.max()/EDGE_FAC)
  unres = ok & (ambig | (np.isfinite(sep) & (sep < GS02_TRACK_SEPMIN)))
  # `valid` gates the two-break quantities (nu_c, the ratio); `valid_m` gates nu_m alone,
  # which is still measured in the VFC bins where nu_c is not
  valid = ok & inwin & ~unres & np.isfinite(nu_c) & np.isfinite(nu_m)
  valid_m = ok & np.isfinite(nu_m) & (nu_m > EDGE_FAC*x.min()) & (nu_m < x.max()/EDGE_FAC) \
            & (valid | is_vfc)
  sw = np.flatnonzero(fast)
  i_swap = int(sw[0]) if sw.size else None
  nu_lo, nu_hi = b_lo, b_hi
  return dict(barT=barT, nu_lo=nu_lo, nu_hi=nu_hi, nu_c=nu_c, nu_m=nu_m, sep=sep,
              ratio=nu_c/nu_m, s_mid=rms, off=~ok, unres=unres, valid=valid,
              valid_m=valid_m, is_vfc=is_vfc, ambig=ambig, beta_mid=bmid,
              sep_unres=GS02_TRACK_SEPMIN,
              i_swap=i_swap, nu_M=nuM_nom, nu_Mt=nu_Mt, nu_B=1./env.gma_m**2,
              nu_win=(float(x.min()), float(x.max())), Fpk=Fpk)


def _fit_slope(barT, y, valid, lo, hi, nmin=4):
  '''log-log slope of y(bar{T}) over [lo, hi], on the valid bins only. NaN if the
  window holds fewer than nmin points -- the correct answer deep in fast cooling,
  where nu_c has left the frequency window long before the tail.'''
  m = valid & (barT >= lo) & (barT <= hi) & np.isfinite(y) & (y > 0.)
  if m.sum() < nmin:
    return np.nan, int(m.sum())
  return float(np.polyfit(np.log10(barT[m]), np.log10(y[m]), 1)[0]), int(m.sum())


def fit_break_evolution(r, tr, barT_f, barT_off=None, rise_win=RISE_WIN):
  '''
  Power-law exponents of nu_c(t) and nu_m(t) over the three phases of the pulse:
    rise      RISE_WIN, the shock still crossing at ~constant hydro
    crossing  [rise_win[1], bar{T}_f], the cells expanding (B and Gamma dropping)
    HLE       past the rarefaction cut-off, where the shell is dark and the observed
              spectrum is the frozen high-latitude one, sliding as the Doppler factor
  Also returns the collapse normalisation A = nu_c*bar{T}^2 / (gma_c/gma_m)^2 (median
  over the rise window; the same for every regime iff the law is regime-independent),
  the OBSERVED nu_c/nu_m at the lightcurve peak against the nominal (gma_c/gma_m)^2,
  the avoided-crossing separation minimum, and a flag naming what truncates the track.
  '''
  barT, v = tr['barT'], tr['valid']
  hle_lo = max(2., 1.2*barT_off[1]) if barT_off is not None else 2.
  out = {}
  v_m = tr.get('valid_m', v)
  for name, y, vv in (('c', tr['nu_c'], v), ('m', tr['nu_m'], v_m)):
    out[f'{name}_rise'], out[f'n_{name}_rise'] = _fit_slope(barT, y, vv, *rise_win)
    out[f'{name}_cross'], out[f'n_{name}_cross'] = _fit_slope(barT, y, vv, rise_win[1], barT_f)
    out[f'{name}_hle'], out[f'n_{name}_hle'] = _fit_slope(barT, y, vv, hle_lo, barT.max())

  env = r['env']
  mr = v & (barT >= rise_win[0]) & (barT <= rise_win[1])
  out['A'] = (float(np.nanmedian(tr['nu_c'][mr]*barT[mr]**2)/(env.gma_c/env.gma_m)**2)
              if mr.sum() >= 4 else np.nan)
  ipk = int(np.nanargmax(tr['Fpk']))
  out['ratio_pk'] = float(tr['ratio'][ipk]) if v[ipk] else np.nan   # NaN: break off-band at peak
  out['nominal'] = (env.gma_c/env.gma_m)**2
  # smallest separation still ABOVE the resolution floor, plus the bar{T} span over which
  # the breaks are unresolved -- the honest statement about how close they get
  out['sep_min'] = float(np.nanmin(tr['sep'][v])) if v.any() else np.nan
  u = tr['unres']
  out['unres_span'] = ((float(barT[u].min()), float(barT[u].max())) if u.any() else None)
  out['barT_swap'] = float(barT[tr['i_swap']]) if tr['i_swap'] is not None else np.nan
  # what ends the measurable track. The physical stop is the gamma=1 floor -- nu_c cannot
  # fall below nu_B, the electrons having stopped cooling -- but it only shows up if nu_B
  # is inside the frequency window: whenever the window bottom sits ABOVE nu_B (it does
  # throughout this sweep, LOGNU_MIN=-6 vs nu_B/nu_m ~ 2e-7) the window is what cuts the
  # track, and saying 'floor' would credit the physics for a plotting choice.
  nu_lo_edge = EDGE_FAC*tr['nu_win'][0]
  if not v.any():
    flag = 'unmeasured'
  elif barT[v].max() >= 0.9*barT.max():
    flag = '-'
  elif tr['nu_c'][v][-1] <= 3.*nu_lo_edge:
    flag = 'gamma=1 floor' if tr['nu_B'] >= nu_lo_edge else 'below window'
  elif tr['nu_c'][v][-1] >= tr['nu_M']/(3.*EDGE_FAC_M):
    flag = 'above nu_M'
  else:
    flag = 'faded'
  out['flag'] = flag
  return out


def detect_rise_peak_tail(Tb, nub, nuFnu, nu_ref=NU_REF,
    frac_rise=FRAC_RISE, frac_tail=FRAC_TAIL):
  '''
  Auto-detect rise/peak/high-latitude-tail times from the lightcurve at
  nub=nu_ref, by walking away from the peak to the first/last points at or
  below frac_rise/frac_tail of the peak flux. Returns (T_rise, T_peak,
  T_tail, info); info carries indices/flags for edge cases (rise before the
  grid start, tail not decayed within Tmax).
  '''
  inu = min(np.searchsorted(nub, nu_ref), len(nub) - 1)
  lc = nuFnu[:, inu]
  i_peak = int(np.argmax(lc))
  F_peak = lc[i_peak]
  if not (F_peak > 0):
    return np.nan, np.nan, np.nan, {'warning': 'no positive flux at nu_ref'}
  T_peak = Tb[i_peak]

  info = {}
  pre = lc[:i_peak + 1]
  idx = np.where(pre <= frac_rise * F_peak)[0]
  if idx.size:
    i_rise = idx[-1]
  else:
    i_rise = 0
    info['warn_rise'] = 'rise before grid start (onset faster than resolvable); clamped to Tb[0]'
  T_rise = Tb[i_rise]

  post = lc[i_peak:]
  idx = np.where(post <= frac_tail * F_peak)[0]
  if idx.size:
    i_tail = i_peak + idx[0]
  else:
    i_tail = len(lc) - 1
    info['warn_tail'] = f'did not decay below {frac_tail:.0%} of peak within Tmax; clamped to Tb[-1]'
  T_tail = Tb[i_tail]

  info.update(i_rise=i_rise, i_peak=i_peak, i_tail=i_tail, F_peak=F_peak)
  return T_rise, T_peak, T_tail, info


def compute_fluence_spectrum(Tb, nuFnu):
  '''
  Time-integrated spectrum: integral of nu F_nu over Tb (own normalized
  time), same trapz-over-T convention as working_peaks.py's time-integrated
  spectra (np.trapezoid(nuFnu, Tb, axis=0)).
  '''
  return np.trapezoid(nuFnu, Tb, axis=0)


def exit_onset_barT(key, z=Z_SHELL):
  '''
  bar{T}_f = (Ton - Ts)/T0 for the last-shocked cell (the shell-exit / outer-edge
  cell reached last by the shock) = the shell-crossing observer time. A pure
  hydro/geometry quantity (~ RfRS0 - 1), alpha- and cooling-independent, so it is
  a single value for the whole sweep; used to normalise the lightcurve time axis.
  '''
  env0 = MyEnv(key)
  sh = cellsBehindShock_fromData(open_rundata(key, z))
  exit_row = sh.loc[sh.t.idxmax()]
  return float((get_variable(exit_row, 'Ton', env0) - env0.Ts) / env0.T0)


def rarefaction_off_barT(key, z=Z_SHELL):
  '''
  (first, last) bar{T} = (Ton - Ts)/T0 at which the rarefaction wave cuts a cell's
  emission off, over all cells of shell z (load_shell_rarefaction_offT). `last` is the
  observer time past which the shell emits NOTHING: beyond it the lightcurve is purely
  the tT^-2 high-latitude tails of already-emitted steps. Like bar{T}_f a pure
  hydro/geometry quantity, alpha- and cooling-independent -> one pair for the sweep.
  '''
  env0 = MyEnv(key)
  sh = cellsBehindShock_fromData(open_rundata(key, z))
  b = np.array(list(load_shell_rarefaction_offT(key, z, env0, n_shell=len(sh)).values()), float)
  b = b[np.isfinite(b)]
  if not b.size:
    return None
  return float(b.min()), float(b.max())


_DATA_END_MEM = {}      # (key, z) -> (first, last); the CSV scan is ~0.5 GB for a long run

def data_end_barT(key, z=Z_SHELL):
  '''
  (first, last) bar{T} = (Ton - Ts)/T0 at which the cells of shell z run out of
  SNAPSHOTS, i.e. where the data method stops following them (tt_end is capped by
  tt_nodes[-1], working_cooling_data.generate_cell_fromHistory). The data-method
  counterpart of rarefaction_off_barT: past `last` the shell contributes only the
  tT^-2 tails of steps already emitted, not because the wave cut it off but because
  the simulation ended. Like the other two a pure geometry quantity (t and x both
  scale with alpha), so one pair for the whole sweep.
  '''
  if (key, z) in _DATA_END_MEM:
    return _DATA_END_MEM[(key, z)]
  env0 = MyEnv(key)
  kmin = env0.Next + (0 if z == 4 else env0.Nsh4)
  kmax = kmin + (env0.Nsh4 if z == 4 else env0.Nsh1)
  b = []
  for k in check_extracted_cells(key):
    if not (kmin <= k < kmax):
      continue                                  # stray files outside the shell's id range
    d = open_celldata(key, k)
    if d is False or not len(d):
      continue
    b.append((get_variable(d.iloc[-1], 'Ton', env0) - env0.Ts)/env0.T0)
  b = np.array(b, float)
  b = b[np.isfinite(b)]
  out = (float(b.min()), float(b.max())) if b.size else None
  _DATA_END_MEM[(key, z)] = out
  return out


_CAP_END_MEM = {}       # (key, z, cap) -> (first, last)

def cap_end_barT(key, z=Z_SHELL, cap=R_CAP):
  '''
  (first, last) bar{T} = (Ton - Ts)/T0 at which the CAPPED histories stop, i.e. where the
  cells of shell z cross R = cap*R_injection. The capped counterpart of data_end_barT,
  for the figure annotations of sweep_prerar.main_capped -- there BOTH sides stop there,
  reference and counterfactual alike, which is the point of the window.

  Interpolated in Ton at the crossing radius rather than read off the truncated row, so
  the annotation does not inherit the snapshot cadence. Like the other barT helpers a
  pure geometry quantity (t and x both scale with alpha), so one pair per sweep.
  '''
  if (key, z, cap) in _CAP_END_MEM:
    return _CAP_END_MEM[(key, z, cap)]
  env0 = MyEnv(key)
  kmin = env0.Next + (0 if z == 4 else env0.Nsh4)
  kmax = kmin + (env0.Nsh4 if z == 4 else env0.Nsh1)
  b = []
  for k in check_extracted_cells(key):
    if not (kmin <= k < kmax):
      continue
    d = open_celldata(key, k)
    if d is False or not len(d):
      continue
    s = select_postshock_rows(d)
    if len(s) < 2:
      continue
    x = s.x.to_numpy(dtype=float)
    Ton = np.asarray(get_variable(s, 'Ton', env0), dtype=float)
    b.append((np.interp(min(cap*x[0], x[-1]), x, Ton) - env0.Ts)/env0.T0)
  b = np.array(b, float)
  b = b[np.isfinite(b)]
  out = (float(b.min()), float(b.max())) if b.size else None
  _CAP_END_MEM[(key, z, cap)] = out
  return out


def _sweep_colors(results, cmap=plt.cm.jet):
  logr = np.array([r['log10ratio'] for r in results], dtype=float)
  norm = plt.Normalize(vmin=logr.min(), vmax=logr.max())
  colors = cmap(norm(logr))
  sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
  return colors, sm


def _draw_order(pairs):
  '''
  Stacking order of the sweep families: slowest cooling first, so the fast-cooling
  curves end up drawn on top. load_sweep stays ASCENDING in log10(gma_c/gma_m) -- the
  x=logr analyses rely on that (sweep_efficiency's np.interp over the regime axis needs
  an increasing abscissa) -- so only the plotting loops are reversed, here. Colours are
  value-indexed (_sweep_colors), so the curve colours and the colorbar are untouched.
  Takes the already-zipped tuples so every zipped list stays in lockstep.
  '''
  return list(pairs)[::-1]


def plot_lightcurve_shape(results, barT_f, barT_off=None, nu_targets=NU_TARGETS,
    outdir=OUTDIR, annotate=True):
  '''
  Shape-normalised lightcurves at a fixed fraction nu_t of each curve's own peak
  frequency nu_pk = max(nu_m, nu_c) (= nu_m for fast cooling, nu_c for slow; this
  is the stored nub axis) -- so every curve is sampled at the same RELATIVE
  spectral position, not at nu_m for all. For each nu_t, one 2-panel figure
  (linear left, log-log right): flux normalised to each curve's own peak, time to
  bar{T}_f = the onset of the last-shocked (shell-exit) cell = the shell-crossing
  time (a common hydro timescale, same for every sweep point). So x=1 marks
  crossing (grey guide) and each curve's flux peak sits at x_pk<1 on the y=1 line.
  barT_off: (first, last) rarefaction cut-off times from rarefaction_off_barT. Shaded
  as the band over which the rarefaction progressively switches the shell off, with a
  line at `last` = when emission stops everywhere; right of it the decay is pure
  high-latitude. Deep in slow cooling that line, not the cooling time, ends the pulse.

  annotate=False drops that whole annotation -- band, bar{T}_rf line and its legend -- and
  writes the figures as a SEPARATE '_plain' series, leaving the curves and the two grey
  guides. Same data, nothing marked on it, for use where the rarefaction cut-off is not the
  point being made. Both series are produced by main.
  '''
  colors, sm = _sweep_colors(results)
  tag = '' if annotate else '_plain'
  xoff = tuple(b/barT_f for b in barT_off) if (annotate and barT_off and barT_f > 0.) else None
  for nu_t in nu_targets:
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    for r, c in _draw_order(zip(results, colors)):
      barT = r['Tb'] - 1.
      inu = min(np.searchsorted(r['nub'], nu_t), len(r['nub']) - 1)
      lc = r['nuFnu'][:, inu]
      ipk = int(np.argmax(lc))
      if lc[ipk] <= 0. or barT_f <= 0.:
        continue
      x, y = barT / barT_f, lc / lc[ipk]
      axs[0].plot(x, y, color=c)
      axs[1].loglog(x, y, color=c)
    for ax, sc in zip(axs, ('linear', 'log')):
      ax.axhline(1., color='grey', ls=':', lw=.7); ax.axvline(1., color='grey', ls=':', lw=.7)
      if xoff is not None:
        ax.axvspan(xoff[0], xoff[1], color='grey', alpha=0.15, lw=0, zorder=0)
        ax.axvline(xoff[1], color='k', ls='--', lw=.9,
                   label='$\\bar{T}_{\\rm rf}$')
      ax.set_xlabel('$\\bar{T}/\\bar{T}_f$')
      ax.set_ylabel('$\\nu F_\\nu/(\\nu F_\\nu)_{\\rm max}$')
      ax.set_title(sc)
    axs[0].set_xlim(*XLIM_LIN)
    axs[1].set_xlim(xmin=1e-3)
    axs[1].set_ylim(ymin=1e-8)
    if xoff is not None:
      axs[0].legend(fontsize=10, loc='upper right')
    fig.colorbar(sm, ax=axs, label='log$_{10}(\\gamma_c/\\gamma_m)$')
    fig.suptitle(f'Lightcurve shape at $\\nu={nu_t:g}\\,\\nu_{{\\rm pk}}$ ')
    fig.savefig(os.path.join(outdir, f'lightcurve_shape{tag}_nu={nu_t:g}.png'), dpi=300)
    plt.close(fig)


def plot_spectra_per_regime(results, detections, outdir=OUTDIR, segments=True):
  '''
  One figure per gamma_c/gamma_m, overlaying the instantaneous spectra at rise,
  peak and tail vs nu/nu_m. Flux is normalised to the plot's peak nuFnu value (the
  brightest of the three spectra, at its spectral peak -- nu_m for fast cooling,
  nu_c for slow), so the peak-phase curve tops out at 1 and rise/tail show their
  brightness evolution below it. Over each spectrum are drawn the synchrotron power-law
  SEGMENTS it actually shows (dash-dotted; identify_segments), each on its own fitted
  intercept over the window it was identified on, run out until it meets its neighbours (and
  to the end of the band for the 1-p/2 one) so that every adjacent pair is seen to cross.
  Nothing is anchored on the spectral peak or on a break: no shape is fitted, the crossings
  are read off the drawn lines rather than computed into a break, and the spectrum's own
  turnovers are left undescribed, which is the honest statement about them. The grey dotted line at
  x=1 is the collision nu_m (the fixed x-axis unit). All plots share the same fixed y-range
  (SPEC_YSPAN decades below the peak); x is clipped to the visible spectra.

  The legend names the regime the identified SET implies -- FC and SC when a mid segment
  survives between the two asymptotes, MC when none does, VFC when there is nothing below
  the nu^(1/2) segment, VSC when there is nothing above the nu^((3-p)/2) one -- and '?'
  where the bottom of the band has converged to neither asymptote, so that no statement
  about a MISSING segment is supportable (see identify_segments). Those are shape classes;
  measure_regime's labels (a bin on the break ratio, tabulated by build_regime_table)
  answer a different question and need not agree.

  segments=False draws the spectra alone -- no segment lines, no regime in the legend, only
  the phase -- and writes them as a SEPARATE '_plain' series. Nothing is measured in that
  pass, so it is also the figure to read when the question is what the spectra do rather
  than what can be said about them. Both series are produced by main.
  '''
  styles = {'rise': 'C0', 'peak': 'C1', 'tail': 'C3'}
  tag = '' if segments else '_plain'
  ylo = 10.**(-SPEC_YSPAN)
  for r, det in zip(results, detections):
    info = det[3]
    x = nu_over_num(r)
    p = r['env'].psyn
    phases = [(w, r['nuFnu'][info[f'i_{w}'], :])
              for w in ('rise', 'peak', 'tail') if info.get(f'i_{w}') is not None]
    if not phases:
      continue
    pkmax = max(np.nanmax(sp) for _, sp in phases)         # plot peak nuFnu (= peak-phase peak)
    fig, ax = plt.subplots()
    handles = []; labels = []; sps = []
    for which, sp in phases:
      col = styles[which]
      (h,) = ax.loglog(x, sp/pkmax, color=col, lw=1.6)
      sps.append(sp/pkmax)
      ident = identify_segments(x, sp, p) if segments else None
      # steepest first = left to right, so consecutive entries are the adjacent pairs
      segs = sorted(ident['segs'].items(), key=lambda kv: -kv[1]['a']) if ident else []
      for k, (name, sg) in enumerate(segs):
        # the segment on its OWN intercept and slope, run out far enough to meet its
        # neighbours: down to the crossing with the previous one, up to the crossing with the
        # next, each by SEG_EXT further, so the pair is seen to cross. The 1-p/2 segment has
        # no neighbour above it and runs to the end of the band.
        l0, l1 = np.log10(sg['x0']), np.log10(sg['x1'])
        if k:
          l0 = min(l0, _seg_cross(segs[k-1][1], sg))
        if k + 1 < len(segs):
          l1 = max(l1, _seg_cross(sg, segs[k+1][1]))
        lxs = np.array([l0 - SEG_EXT,
                        np.log10(x.max()) if name == 'hi' else l1 + SEG_EXT])
        ax.loglog(10**lxs, 10**(sg['c'] + sg['a']*lxs)/pkmax,
                  color=col, ls='-.', lw=0.9, alpha=0.8)
      handles.append(h)
      labels.append(which if not segments else
                    f"{which}: {(ident['regime'] or '?') if ident else '?'}")
    ax.axvline(1., color='grey', ls=':', lw=.9)                # nu_m at collision (x-axis unit)
    ax.set_ylim(ylo, 3.)
    # clip x to where the (y-clipped) spectra are actually visible, +half a decade
    vis = np.any(np.array(sps) > ylo, axis=0)
    if vis.any():
      xv = x[vis]
      ax.set_xlim(xv.min()/3., xv.max()*3.)
    ax.set_xlabel(NU_M_LABEL)
    ax.set_ylabel('$\\nu F_\\nu/(\\nu F_\\nu)_{\\rm pk}$')
    ax.set_title(f'Spectral evolution, $\\log_{{10}}(\\gamma_c/\\gamma_m)={r["log10ratio"]:+.0f}$')
    ax.legend(handles, labels, fontsize=10, loc='lower center')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir,
        f'spectrum_evolution{tag}_logr={r["log10ratio"]:+.1f}.png'), dpi=300)
    plt.close(fig)


def build_regime_table(results, detections, outdir=OUTDIR):
  '''
  For every (log10ratio x rise/peak/tail) spectrum, measure the time-dependent
  regime from its two breaks nu_m(t), nu_c(t) (in collision-nu_m units), and
  tabulate the break ratio nu_c(t)/nu_m(t) vs the nominal nu_c/nu_m=(gc/gm)^2,
  with the gamma_c(t) the ratio implies (the VFC/VSC classifier, see measure_regime).
  Prints the table, writes regime_table.csv and renders regime_table.png.
  '''
  p_slope = results[0]['env'].psyn
  cols = ['log10(gc/gm)', 'phase', 'nom nu_c/nu_m', 'nu_m(t)/nu_m0',
          'nu_c(t)/nu_m(t)', 'gamma_c(t)', 's_mid', 's_above', 'regime']
  rows = []        # plain text (print + CSV)
  rows_disp = []   # sci_notation freq columns (PNG render)
  for r, det in zip(results, detections):
    info = det[3]; x = nu_over_num(r); logr = r['log10ratio']
    for which in ('rise', 'peak', 'tail'):
      iT = info.get(f'i_{which}')
      if iT is None:
        continue
      m = measure_regime(x, r['nuFnu'][iT, :], r['env'].psyn, r['env'])
      nom = 10.**(2*logr)
      pre = ('<' if m['ratio'] < 1. else '>') if m['off'] else ''
      rat = f"{pre}{m['ratio']:.2e}"
      rat_d = ((('$<$' if m['ratio'] < 1. else '$>$') if m['off'] else '') + sci_notation(m['ratio']))
      rows.append([f'{logr:+.0f}', which, f'{nom:.0e}', f"{m['num_t']:.2e}",
                   rat, f"{m['gc_t']:.2e}", f"{m['s_mid']:+.2f}", f"{m['s_above']:+.2f}",
                   m['regime']])
      rows_disp.append([f'{logr:+.0f}', which, sci_notation(nom), sci_notation(m['num_t']),
                        rat_d, sci_notation(m['gc_t']), f"{m['s_mid']:+.2f}",
                        f"{m['s_above']:+.2f}", m['regime']])

  # theoretical slope references + the two physical regime boundaries, for the header note
  note = (f'p={p_slope:g}: slopes  4/3={4/3.:.2f}  (3-p)/2={(3-p_slope)/2:.2f}  '
          f'1/2=0.50  1-p/2={1-p_slope/2:.2f}   |   '
          f'VFC: gamma_c<{GMA_VFC:g}   VSC: gamma_c>gamma_M')
  widths = [max(len(cols[i]), max(len(row[i]) for row in rows)) for i in range(len(cols))]
  def fmt(row): return '  '.join(v.ljust(widths[i]) for i, v in enumerate(row))
  print('\n' + note)
  print(fmt(cols)); print('  '.join('-'*w for w in widths))
  for row in rows:
    print(fmt(row))

  import csv
  with open(os.path.join(outdir, 'regime_table.csv'), 'w', newline='') as f:
    w = csv.writer(f); w.writerow(cols); w.writerows(rows)

  fig, ax = plt.subplots(figsize=(11, 0.32*len(rows)+1.1)); ax.axis('off')
  tbl = ax.table(cellText=rows_disp, colLabels=cols, loc='center', cellLoc='center')
  tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.25)
  cmap = {'VFC':'#2166ac','FC':'#67a9cf','marginal':'#f7f7f7','SC':'#ef8a62','VSC':'#b2182b'}
  for i, row in enumerate(rows):
    tbl[(i+1, len(cols)-1)].set_facecolor(cmap.get(row[-1], 'w'))
  ax.set_title('Apparent cooling regime per spectrum  (' + note + ')', fontsize=9)
  fig.savefig(os.path.join(outdir, 'regime_table.png'), dpi=200, bbox_inches='tight')
  plt.close(fig)
  print(f'\nregime table -> {outdir}/regime_table.csv, regime_table.png')
  return rows


def c25_num_curve(key, z, Tb):
  '''
  nu_m(t) of the C25 thin-shell peak model on the sweep's own time grid, in the same
  collision-nu_m units as the fitted breaks -- the analytic estimate the shape fits can be
  checked against. Tb IS the model's tilde{T} (1 at collision), so it is passed through.

  The model's parameters come from the base simulation (working_peaks.get_model_params ->
  a_u, tau = t_on/t_off) and are invariant under the alpha rescaling, since t{z} and toff
  scale together: ONE curve therefore serves every sweep point. C25_peak_model returns
  (nu_pk, nuFnu_pk) and its nu_pk IS nu_m -- the model carries no cooling break.

  Returns None if the setup has no fitted a_u family or the env lacks a_u / t{z} / toff, so
  sweeps of other simulations lose the reference rather than failing.
  '''
  try:
    from working_peaks import get_model_params
    from peak_modeling import C25_peak_model
    au, tau, reverse = get_model_params(key, z)
    num, _ = C25_peak_model(np.asarray(Tb, float), au, tau, reverse)
    num = np.asarray(num, float)
    return num if np.isfinite(num).any() else None
  except Exception as e:
    print(f'c25_num_curve: no thin-shell reference for key={key!r} z={z} ({e})')
    return None


def _gap(y, mask):
  '''y with the unmasked samples blanked to NaN, keeping the ORIGINAL length. Plotting
  y[mask] against barT[mask] instead would compress the arrays and let matplotlib draw a
  straight segment across every gap in the mask -- a line where nothing was measured.'''
  return np.where(mask, y, np.nan)


def _mark_hydro_times(ax, barT_f, barT_off=None):
  '''bar{T}_f (shell crossing) and the rarefaction band, the common hydro times of
  every sweep point -- same guides as plot_lightcurve_shape, on a bar{T} axis.'''
  ax.axvline(barT_f, color='grey', ls=':', lw=.9)
  if barT_off is not None:
    ax.axvspan(barT_off[0], barT_off[1], color='grey', alpha=0.15, lw=0, zorder=0)
    ax.axvline(barT_off[1], color='k', ls='--', lw=.9)


def _guide(ax, x, slope, y_at, **kw):
  '''power-law guide line of the given slope through (x[0], y_at).'''
  x = np.asarray(x, float)
  ax.plot(x, y_at*(x/x[0])**slope, color='k', lw=.8, alpha=.6, **kw)


def plot_break_evolution(results, tracks, fits, barT_f, barT_off=None, outdir=OUTDIR,
    c25=None):
  '''
  Time evolution of the three spectral frequencies, one colour per gamma_c/gamma_m.
  Top: nu_c(t) (solid), nu_m(t) (dashed) and the cutoff nu_M(t) (dash-dot) in
  collision-nu_m units vs bar{T}; the stretches where a break is not measurable
  (off-window, or within EDGE_FAC of an edge / of the cutoff nu_M) are drawn faint, and
  the gamma=1 floor nu_B -- below which nu_c cannot go, the electrons having stopped
  cooling -- is the dotted horizontal. Guides of slope -2 (synchrotron cooling at
  ~constant hydro) and -1 (high-latitude Doppler slide) frame the two asymptotic laws.
  The three carry different physics: nu_M is the synchrotron BURNOFF limit, whose
  gma_M ~ B^-1/2 makes nu'_M ~ B*gma_M^2 independent of B -- a constant of nature
  (3e^2/2 pi m_e c sigma_T = 25 MeV), so it moves ONLY with the Doppler factor and is
  flat until the shell goes dark. nu_m is injection, drifting down as the shock weakens.
  Only nu_c carries cooling, and only it has the -2 law.
  Middle: the same nu_c divided by the nominal (gma_c/gma_m)^2 -- if the law is
  regime-independent, every curve falls on one, with the A*bar{T}^-2 reference.
  Bottom (only when c25 is given): the fitted nu_m against the C25 thin-shell estimate
  (c25_num_curve). nu_m is injection, so a shell calculation that resolves the shock the
  same way should reproduce the thin-shell value -- this is the panel that says whether it
  does. The crossing gaps stay gaps: where the fit cannot name nu_m there is no ratio.
  '''
  colors, sm = _sweep_colors(results)
  npan = 3 if c25 is not None else 2
  fig, axs = plt.subplots(npan, 1, figsize=(7.5, 8.5 + 2.6*(npan - 2)),
                          sharex=True, gridspec_kw=dict(height_ratios=[3, 3, 2][:npan]))
  A = np.nanmedian([f['A'] for f in fits])
  for r, tr, f, c in _draw_order(zip(results, tracks, fits, colors)):
    barT, v = tr['barT'], tr['valid']
    if not v.any():
      continue
    # only MEASURED points are drawn: a finite fitted nu_c that failed a validity cut is a
    # break the fit placed outside the observed band (the transition into VFC), and showing
    # it even faintly asserts a cooling break where the fit found none
    axs[0].loglog(barT, _gap(tr['nu_c'], v), color=c, lw=1.5)
    v_m = tr.get('valid_m', v)
    axs[0].loglog(barT, _gap(tr['nu_m'], v_m), color=c, lw=1.1, ls='--')
    axs[0].loglog(barT, _gap(tr['nu_Mt'], np.isfinite(tr['nu_Mt'])), color=c, lw=1.1, ls='-.')
    # the blended stretch: one break, drawn as such, so the gap in the solid curves is
    # visibly "not measured" rather than "not there"
    u = tr['unres']
    if u.any():
      axs[0].loglog(barT, _gap(tr['nu_c'], u), color=c, lw=1.2, ls=':')
    axs[1].loglog(barT, _gap(tr['nu_c']/f['nominal'], v), color=c, lw=1.5)
  # y-range from the MEASURED points only: the faint stretches are extrapolations of the
  # shape beyond the frequency window and can run many decades off, which would otherwise
  # squash every real curve into the middle of the panel
  vals = np.concatenate([np.concatenate([t['nu_c'][t['valid']],
                                         t['nu_m'][t.get('valid_m', t['valid'])]]
                                        + [t['nu_Mt'][np.isfinite(t['nu_Mt'])]])
                         for t in tracks if t['valid'].any()])
  vals = vals[np.isfinite(vals) & (vals > 0.)]
  if vals.size:
    axs[0].set_ylim(min(vals.min(), np.nanmin([t['nu_B'] for t in tracks]))/3., vals.max()*3.)
  # guides: free-floating references, anchored on the panel's own range (the curves
  # themselves span ~12 decades, so no single curve is a sensible anchor)
  y0, y1 = axs[0].get_ylim()
  _guide(axs[0], np.array(RISE_WIN), -2., y1/30., ls='-')
  xh = np.array([max(2., barT_off[1] if barT_off else 2.), results[0]['Tb'].max()-1.])
  _guide(axs[0], xh, -1., np.sqrt(y0*y1), ls='-.')
  _guide(axs[1], np.array([RISE_WIN[0], 1.]), -2., A*RISE_WIN[0]**-2, ls='-')
  # nu_B = nu_m/gma_m^2 with gma_m the same for every point (the alpha rescaling leaves the
  # injection Lorentz factor alone), so the floor is one shared line
  nu_B = np.unique(np.round([t['nu_B'] for t in tracks], 12))
  for nb in nu_B:
    axs[0].axhline(nb, color='k', ls=':', lw=.8)
  if c25 is not None:
    for r, tr, c in _draw_order(zip(results, tracks, colors)):
      v_m = tr.get('valid_m', tr['valid'])
      if v_m.any():
        axs[2].semilogx(tr['barT'], _gap(tr['nu_m']/c25, v_m), color=c, lw=1.3)
    axs[2].axhline(1., color='grey', ls=':', lw=.9)
    axs[2].set_ylim(0.3, 3.)
    axs[2].set_ylabel('$\\nu_m$ fit / C25')
    axs[0].loglog(results[0]['Tb'] - 1., c25, color='k', ls='--', lw=1.2, alpha=.8)
  for ax in axs:
    _mark_hydro_times(ax, barT_f, barT_off)
  axs[0].set_ylabel('$\\nu_c$ (solid), $\\nu_m$ (dashed), $\\nu_M$ (dash-dot)  $/\\nu_{m,0}$')
  axs[0].set_title('Break evolution: guides of slope $-2$ (cooling) and $-1$ (high latitude);'
                   '\ndotted horizontal = the $\\gamma=1$ floor $\\nu_B$'
                   + ('; black dashed = C25 $\\nu_m$' if c25 is not None else ''))
  axs[1].set_ylabel('$\\nu_c(t)/[\\nu_{m,0}(\\gamma_c/\\gamma_m)^2]$')
  axs[-1].set_xlabel('$\\bar{T}=(T_{\\rm obs}-T_s)/T_0$')
  axs[1].set_title(f'collapsed on the nominal ratio; reference $A\\,\\bar{{T}}^{{-2}}$, $A={A:.2f}$')
  fig.colorbar(sm, ax=axs, label='log$_{10}(\\gamma_c/\\gamma_m)$')
  fig.savefig(os.path.join(outdir, 'break_evolution.png'), dpi=300)
  plt.close(fig)


def plot_break_ratio(results, tracks, barT_f, barT_off=None, outdir=OUTDIR):
  '''
  nu_c(t)/nu_m(t) vs bar{T}, one curve per regime, with the FC/SC line at 1. nu_c falls
  as bar{T}^-2 while nu_m is nearly flat, so EVERY regime starts slow-cooling and hardens
  toward fast cooling; past the rarefaction cut-off both breaks slide as the same Doppler
  factor and the ratio -- hence the whole spectral shape -- freezes. The crossing itself is
  a GAP, not a curve: while the ratio is within the detector's resolution floor the two
  breaks are one blended feature (shaded band, see SEP_UNRESOLVED), so how closely they
  approach is not measured here.
  '''
  colors, sm = _sweep_colors(results)
  fig, ax = plt.subplots(figsize=(7.5, 5))
  # band width comes from whichever tracker produced these tracks: the knee scan cannot
  # separate breaks closer than ~SEP_UNRESOLVED, the GS02 shape fit gets down to ~1.5
  sep_u = max(tr.get('sep_unres', SEP_UNRESOLVED) for tr in tracks)
  ax.axhspan(1./sep_u, sep_u, color='grey', alpha=0.12, lw=0, zorder=0)
  for r, tr, c in _draw_order(zip(results, tracks, colors)):
    v = tr['valid']
    if not v.any():
      continue
    ax.loglog(tr['barT'], _gap(tr['ratio'], v), color=c, lw=1.4)
  ax.axhline(1., color='grey', ls=':', lw=.9)
  _mark_hydro_times(ax, barT_f, barT_off)
  ax.set_xlabel('$\\bar{T}=(T_{\\rm obs}-T_s)/T_0$')
  ax.set_ylabel('$\\nu_c(t)/\\nu_m(t)$')
  ax.set_title('Observed cooling regime vs time\n(shaded: breaks unresolved, one blended feature)')
  fig.colorbar(sm, ax=ax, label='log$_{10}(\\gamma_c/\\gamma_m)$')
  fig.savefig(os.path.join(outdir, 'break_ratio_evolution.png'), dpi=300)
  plt.close(fig)


def build_break_evolution_table(results, fits, outdir=OUTDIR, tracks=None, c25=None):
  '''
  One row per gamma_c/gamma_m: the fitted exponents of nu_c(t) and nu_m(t) in the three
  phases, the collapse normalisation A, the avoided-crossing separation minimum and the
  bar{T} at which the roles swap, and the OBSERVED nu_c/nu_m at the lightcurve peak
  against the nominal (gma_c/gma_m)^2. Prints, writes break_evolution.csv and .png.
  '''
  cols = ['log10(gc/gm)', 'nu_c rise', 'nu_c cross', 'nu_c HLE', 'nu_m rise', 'nu_m HLE',
          'A', 'sep min', 'unresolved bar_T', 'nu_m/C25', 'nu_c/nu_m pk', 'nominal', 'ends on']

  def c25_ratio(tr):
    '''median fitted nu_m / C25 over the measured bins, with the half-range spread'''
    if c25 is None or tr is None:
      return '--'
    v = tr.get('valid_m', tr['valid']) & (tr['barT'] >= 5e-3)
    rat = (tr['nu_m']/c25)[v]
    rat = rat[np.isfinite(rat) & (rat > 0.)]
    if rat.size < 4:
      return '--'
    return f'{np.median(rat):.2f} ±{np.sqrt(rat.max()/rat.min())-1.:.0%}'
  def g(v, fmt='{:+.2f}'):
    return '--' if not np.isfinite(v) else fmt.format(v)
  def span(s):
    return '--' if s is None else f'{s[0]:.3g}-{s[1]:.3g}'
  rows = [[f"{r['log10ratio']:+.0f}", g(f['c_rise']), g(f['c_cross']), g(f['c_hle']),
           g(f['m_rise']), g(f['m_hle']), g(f['A'], '{:.2f}'), g(f['sep_min'], '{:.1f}'),
           span(f['unres_span']), c25_ratio(tr), g(f['ratio_pk'], '{:.2e}'),
           g(f['nominal'], '{:.0e}'), f['flag']]
          for r, f, tr in zip(results, fits,
                              tracks if tracks is not None else [None]*len(results))]

  note = ('slopes are d ln(nu)/d ln(bar_T);  rise = '
          f'bar_T in [{RISE_WIN[0]:g}, {RISE_WIN[1]:g}] (cooling at ~constant hydro, expect -2),'
          '  HLE = past the rarefaction cut-off (Doppler slide, expect -1).\n'
          '"sep min" is the closest RESOLVED break separation; over "unresolved bar_T" the '
          'two breaks are too close for the tracker to separate, and neither is measured.'
          '\n"nu_m/C25" compares the fitted nu_m with the thin-shell peak model '
          '(peak_modeling.C25_peak_model at the run\'s a_u, tau); the spread is over the '
          'measured bins, so a wide one means the FC/SC crossing falls inside the window')
  widths = [max(len(cols[i]), max(len(row[i]) for row in rows)) for i in range(len(cols))]
  def fmt(row): return '  '.join(v.ljust(widths[i]) for i, v in enumerate(row))
  print('\n' + note)
  print(fmt(cols)); print('  '.join('-'*w for w in widths))
  for row in rows:
    print(fmt(row))

  import csv
  with open(os.path.join(outdir, 'break_evolution_table.csv'), 'w', newline='') as fcsv:
    w = csv.writer(fcsv); w.writerow(cols); w.writerows(rows)

  fig, ax = plt.subplots(figsize=(13, 0.32*len(rows)+1.1)); ax.axis('off')
  tbl = ax.table(cellText=rows, colLabels=cols, loc='center', cellLoc='center')
  tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.25)
  ax.set_title('Break evolution  (' + note + ')', fontsize=8)
  fig.savefig(os.path.join(outdir, 'break_evolution_table.png'), dpi=200, bbox_inches='tight')
  plt.close(fig)
  print(f'\nbreak-evolution table -> {outdir}/break_evolution_table.csv, '
        'break_evolution_table.png')
  return rows


GS02_NT_RMS = 120         # time samples for the rms(bar{T}) curve. A GS02 fit is ~12 ms, so
                          # the full 450-bin grid would be ~43 s per method; 120 log-spaced
                          # samples resolve every feature of the curve in ~11 s.


def plot_gs02_fits(results, detections, outdir=OUTDIR):
  '''
  One figure per gamma_c/gamma_m: the rise/peak/tail instantaneous spectra with the fitted
  Granot & Sari (2002) shape overlaid (dashed), and a residual panel below in log10 flux.
  Tests whether GS02's smoothly-joined form describes the spectra the full cooling
  calculation produces -- unlike paired_syn_bpl, the fit CHOOSES the regime (the label in
  the legend) and its breaks are fitted, not read off a knee scan.
  '''
  styles = {'rise': 'C0', 'peak': 'C1', 'tail': 'C3'}
  for r, det in zip(results, detections):
    info = det[3]; x = nu_over_num(r); p = r['env'].psyn; nuM = nu_M_over_num(r)
    fig, axs = plt.subplots(2, 1, figsize=(7.5, 7), sharex=True,
                            gridspec_kw=dict(height_ratios=[3, 1]))
    pkmax = None; drawn = False
    for which in ('rise', 'peak', 'tail'):
      iT = info.get(f'i_{which}')
      if iT is None:
        continue
      sp = r['nuFnu'][iT, :]
      f = fit_gs02_spectrum(x, sp, p, nuM)
      if f is None:
        continue
      if pkmax is None:
        pkmax = np.nanmax(r['nuFnu'][info.get('i_peak', iT), :])
      mod = gs02_model(x, f, p)
      c = styles[which]
      axs[0].loglog(x, sp/pkmax, color=c, lw=1.5)
      axs[0].loglog(x, mod/pkmax, color=c, lw=0.9, ls='--')
      axs[0].axvline(f['num'], color=c, ls=':', lw=.7)      # fitted nu_m
      axs[0].axvline(f['nuc'], color=c, ls='--', lw=.7)     # fitted nu_c
      ok = np.isfinite(sp) & (sp > 0.) & (sp/pkmax > 10.**(-SPEC_YSPAN))
      axs[1].semilogx(x[ok], np.log10(sp[ok]) - np.log10(mod[ok]), color=c, lw=1.,
                      label=f"{which}: {f['regime']}, rms={f['rms']:.3f} "
                            f"(body {f['rms_body']:.3f})"
                            + ('  break at bound' if f['at_bound'] else ''))
      drawn = True
    if not drawn:
      plt.close(fig); continue
    axs[0].set_ylim(10.**(-SPEC_YSPAN), 3.)
    axs[0].axvline(1., color='grey', ls=':', lw=.9)
    axs[0].set_ylabel('$\\nu F_\\nu/(\\nu F_\\nu)_{\\rm pk}$')
    axs[0].set_title('Granot & Sari (2002) shape vs computed spectra, '
                     f"$\\log_{{10}}(\\gamma_c/\\gamma_m)={r['log10ratio']:+.0f}$\n"
                     f'dashed = fit ($s_1={GS02_S1:g}$, $s_2={GS02_S2:g}$ fixed, '
                     'cutoff $\\propto R(x)$); '
                     'dotted/dashed verticals = fitted $\\nu_m$/$\\nu_c$')
    axs[1].axhline(0., color='grey', ls=':', lw=.9)
    axs[1].set_ylim(-0.6, 0.6)
    axs[1].set_ylabel('$\\log_{10}$(data/fit)')
    axs[1].set_xlabel(NU_M_LABEL)
    axs[1].legend(fontsize=8, loc='lower left')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"gs02_fit_logr={r['log10ratio']:+.1f}.png"), dpi=300)
    plt.close(fig)


def plot_gs02_rms(results, barT_f, barT_off=None, outdir=OUTDIR, n_times=GS02_NT_RMS):
  '''
  Goodness of the GS02 shape versus time: rms of log10(data/fit) per instantaneous
  spectrum, one curve per regime, over n_times log-spaced bar{T} samples. Shows where the
  ansatz holds (around the peak) and where it degrades (the high-latitude tail, whose
  spectrum is a Doppler-smeared superposition rather than a broken power law).
  '''
  colors, sm = _sweep_colors(results)
  fig, ax = plt.subplots(figsize=(7.5, 5))
  for r, c in _draw_order(zip(results, colors)):
    x = nu_over_num(r); p = r['env'].psyn; nuM = nu_M_over_num(r)
    barT = r['Tb'] - 1.
    Fpk = np.nanmax(r['nuFnu'], axis=1)
    live = np.flatnonzero(np.isfinite(Fpk) & (Fpk > 1e-10*np.nanmax(Fpk)))
    if live.size < 5:
      continue
    idx = np.unique(np.geomspace(live[0] + 1, live[-1] + 1, min(n_times, live.size)).astype(int) - 1)
    bb, rr = [], []
    for i in idx:
      f = fit_gs02_spectrum(x, r['nuFnu'][i, :], p, nuM)
      if f is not None:
        bb.append(barT[i]); rr.append(f['rms'])
    if bb:
      ax.loglog(bb, rr, color=c, lw=1.3)
  _mark_hydro_times(ax, barT_f, barT_off)
  ax.set_xlabel('$\\bar{T}=(T_{\\rm obs}-T_s)/T_0$')
  ax.set_ylabel('rms of $\\log_{10}$(data/fit)')
  ax.set_title('Quality of the Granot & Sari (2002) shape vs time')
  fig.colorbar(sm, ax=ax, label='log$_{10}(\\gamma_c/\\gamma_m)$')
  fig.savefig(os.path.join(outdir, 'gs02_rms.png'), dpi=300)
  plt.close(fig)


def build_gs02_table(results, detections, outdir=OUTDIR):
  '''
  Per regime x phase: the fitted GS02 breaks, the regime the fit picked, its rms, and the
  same fit with s1, s2 FREE -- the gap between the two columns is what GS02's practice of
  tabulating fixed smoothing exponents costs here. paired_syn_bpl fitted the same way is
  the baseline. Prints, writes gs02_fits_table.csv and .png.
  '''
  cols = ['log10(gc/gm)', 'phase', 'fit nu_m', 'fit nu_c', 'regime', 'nu_M fit/coll',
          'rms (R cutoff)', 'rms body', 'b_mid free', 'rms (exp cutoff)',
          'rms paired_bpl', 'note']
  rows = []
  fixed, freed, paired, bodies, expcut = [], [], [], [], []
  for r, det in zip(results, detections):
    info = det[3]; x = nu_over_num(r); p = r['env'].psyn; nuM = nu_M_over_num(r)
    for which in ('rise', 'peak', 'tail'):
      iT = info.get(f'i_{which}')
      if iT is None:
        continue
      sp = r['nuFnu'][iT, :]
      f = fit_gs02_spectrum(x, sp, p, nuM)
      ff = fit_gs02_spectrum(x, sp, p, nuM, free_s=True)
      if f is None:
        continue
      fe = fit_gs02_spectrum(x, sp, p, nuM, cutoff='exp')      # cruder cutoff, for reference
      fb = fit_gs02_spectrum(x, sp, p, nuM, free_bmid=True)    # marginality diagnostic
      rp = _fit_paired_syn_bpl(x, sp, p, nuM)
      fixed.append(f['rms']); paired.append(rp); bodies.append(f['rms_body'])
      expcut.append(fe['rms'] if fe else np.nan)
      if ff is not None:
        freed.append(ff['rms'])
      nuc_txt = (f"{f['nuc']:.2e}" if np.isfinite(f['nuc'])
                 else f"<{f.get('nuc_max', np.nan):.0e}")   # VFC: bounded, not measured
      rows.append([f"{r['log10ratio']:+.0f}", which, f"{f['num']:.2e}", nuc_txt,
                   f['regime'], f"{f['nuM']/nuM:.2f}", f"{f['rms']:.4f}", f"{f['rms_body']:.4f}",
                   (f"{fb['beta_mid']:+.3f}" + ('*' if fb['regime'] == 'MC' else '')) if fb else '--',
                   f"{fe['rms']:.4f}" if fe else '--', f'{rp:.4f}',
                   'break at bound' if f['at_bound'] else '-'])

  note = (f'GS02 shape, s1={GS02_S1:g} s2={GS02_S2:g} fixed, '
          'cutoff shaped by the single-electron emissivity R(x) (radiation_cooling.syn_cutoff_R),'
          '\nnu_M FITTED -- "nu_M fit/coll" is its ratio to the collision value, i.e. the '
          'Doppler slide (~1 while emitting, ~0.4 in the tail).'
          f'\nrms is of log10(data/fit) over the top {GS02_FIT_DEC:g} decades; "rms body" '
          f'excludes the cutoff (nu < nu_M/{GS02_BODY_FAC:g}).'
          f'\nmean: {np.mean(fixed):.4f} with the R cutoff vs {np.nanmean(expcut):.4f} with '
          f'exp(-nu/nu_M); free-s {np.mean(freed):.4f} (cost {np.mean(fixed)-np.mean(freed):+.4f}); '
          f'paired_syn_bpl {np.mean(paired):.4f}; GS02 better in '
          f'{sum(a < b for a, b in zip(fixed, paired))}/{len(fixed)} cases.  '
          f'BODY ALONE: {np.nanmean(bodies):.4f}.'
          '\n"b_mid free" is the middle F_nu slope when left free between its asymptotes '
          f'({-0.5:+.2f} fast, {-(results[0]["env"].psyn-1)/2:+.2f} slow); * marks a value '
          'between them, i.e. a spectrum for which neither break is cleanly nu_c or nu_m')
  widths = [max(len(cols[i]), max(len(row[i]) for row in rows)) for i in range(len(cols))]
  def fmt(row): return '  '.join(v.ljust(widths[i]) for i, v in enumerate(row))
  print('\n' + note)
  print(fmt(cols)); print('  '.join('-'*w for w in widths))
  for row in rows:
    print(fmt(row))

  import csv
  with open(os.path.join(outdir, 'gs02_fits_table.csv'), 'w', newline='') as fcsv:
    w = csv.writer(fcsv); w.writerow(cols); w.writerows(rows)

  fig, ax = plt.subplots(figsize=(12, 0.32*len(rows)+1.4)); ax.axis('off')
  tbl = ax.table(cellText=rows, colLabels=cols, loc='center', cellLoc='center')
  tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.25)
  ax.set_title('Granot & Sari (2002) shape vs computed spectra\n' + note, fontsize=8)
  fig.savefig(os.path.join(outdir, 'gs02_fits_table.png'), dpi=200, bbox_inches='tight')
  plt.close(fig)
  print(f'\nGS02 table -> {outdir}/gs02_fits_table.csv, gs02_fits_table.png')
  return rows


def _fit_paired_syn_bpl(x, sp, psyn, nuM, fit_dec=GS02_FIT_DEC):
  '''paired_syn_bpl fitted to a spectrum the same way as fit_gs02_spectrum (free breaks,
  free smoothing, free scale), so the two shapes are compared on equal terms. Returns the
  rms of log10(data/fit).'''
  good = np.isfinite(sp) & (sp > 0.)
  xg, yg = x[good], np.log10(sp[good])
  keep = yg > yg.max() - fit_dec
  xg, yg = xg[keep], yg[keep] - yg[keep].max()
  if len(xg) < 12:
    return np.nan
  x_pk = xg[int(np.argmax(yg))]
  def resid(q):
    mod = paired_syn_bpl(xg, 10**q[0], 10**q[1], psyn, peak=10**q[3], xM=nuM,
                         s=max(10**q[2], 0.02))
    return np.log10(np.maximum(mod, 1e-300)) - yg
  lo_b, hi_b = np.log10(xg.min()) - 0.5, np.log10(xg.max()) + 0.5
  r = least_squares(resid, [np.log10(x_pk), np.log10(x_pk), np.log10(0.4), 0.],
                    bounds=([lo_b, lo_b, -1.7, -8.], [hi_b, hi_b, 0.7, 8.]))
  return float(np.sqrt(np.mean(r.fun**2)))


SPEC_MODES = ('nu_m', 'max', 'eff')   # normalisations of the all-regimes spectra figures
_MODE_TITLE = {'nu_m': 'normalised at $\\nu_m$',
               'max': 'peak-normalised',
               'eff': 'peak-normalised $\\times\\,\\varepsilon_{\\rm rad}$'}


def _plot_spectra_all(results, get_spec, mode, title, fname, outdir, yclip_dec=3.5,
    sym='\\nu F_\\nu'):
  '''
  Overlay one spectrum per sweep point vs nu/nu_m, normalised to the value at nu_m
  (mode='nu_m'), to the peak value (mode='max'), or to the peak value and then
  multiplied by that point's radiative efficiency eps_rad=E_rad/E_inj (mode='eff'),
  which restores the ENERGETICS the two shape normalisations divide out: the curves
  keep their shapes but are stacked by how much of the injected electron energy each
  regime actually radiates (~1 in fast cooling down to a few % in slow cooling), so a
  faint slow-cooling regime no longer looks as bright as a fast-cooling one. Points
  whose cache holds no energy budget (compute_efficiency -> nan) are dropped from the
  'eff' figure rather than drawn unscaled. get_spec(r) -> spectrum.
  sym is the plotted quantity's LaTeX symbol (nuFnu for a flux spectrum, nu*fluence
  for a time-integrated one). The y-axis is clipped to yclip_dec decades below the
  highest curve -- a gentler clip than the spectral-evolution plots so all the
  spectral shapes stay visible. Under mode='eff' the floor hangs off the LOWEST
  curve's peak instead, so every curve still shows yclip_dec decades of its own
  shape however far the efficiency scaling has pushed it down (the eps_rad spread is
  ~1.6 decades across the sweep); the panel is that much taller, and nothing else
  about the figure changes.
  No curve may END inside the panel: each point's window runs LOGNU_ABOVE_NUM decades
  past its own nu_M (_nu_window), so every spectrum has rolled over and dropped below
  the y-floor before its grid ends (it exits through the bottom), and they all share
  LOGNU_MIN so their low-frequency ends sit on the left spine. The axes limits are set
  explicitly -- x from that shared low end to just past the last curve still above the
  floor, which also trims the dead space above the highest nu_M.
  '''
  colors, sm = _sweep_colors(results)
  fig, ax = plt.subplots()
  ylab = {'nu_m': f'${sym}/({sym})_{{\\nu_m}}$',
          'max': f'${sym}/({sym})_{{\\rm max}}$',
          'eff': f'$\\varepsilon_{{\\rm rad}}\\,{sym}/({sym})_{{\\rm max}}$'}[mode]
  curves, ypks = [], []
  for r, c in zip(results, colors):
    x = nu_over_num(r)
    sp = get_spec(r)
    norm = sp[int(np.argmin(np.abs(x - 1.)))] if mode == 'nu_m' else sp.max()
    if mode == 'eff':
      eff = compute_efficiency(r)
      if not np.isfinite(eff) or eff <= 0.:
        continue                # no energy budget in this point's cache: cannot scale it
      norm /= eff
    if norm <= 0.:
      continue
    y = sp / norm
    curves.append((x, y, c))
    ypks.append(float(np.nanmax(y)))
  if not curves:
    print(f'{fname}: nothing to plot'
          + (' (energies absent from the cache; re-run with use_cache=False)'
             if mode == 'eff' else ''))
    plt.close(fig)
    return
  ymax = max(ypks)
  # 'eff' hangs the floor off the FAINTEST curve so each keeps yclip_dec of its own shape
  ylo = (min(ypks) if mode == 'eff' else ymax)/10.**yclip_dec if ymax > 0. else None
  for x, y, c in _draw_order(curves):
    ax.loglog(x, y, color=c)
  ax.axvline(1., color='grey', ls=':', lw=.7)
  if ylo is not None:
    ax.set_ylim(ylo, ymax*3.)
    xhi = max(x[y > ylo].max() for x, y, _ in curves if np.any(y > ylo))
    ax.set_xlim(min(x[0] for x, _, _ in curves), 2.*xhi)
  fig.colorbar(sm, ax=ax, label='log$_{10}(\\gamma_c/\\gamma_m)$')
  ax.set_xlabel(NU_M_LABEL)
  ax.set_ylabel(ylab)
  ax.set_title(title)
  fig.tight_layout()
  fig.savefig(os.path.join(outdir, fname), dpi=300)
  plt.close(fig)


def plot_peak_spectra_all(results, detections, mode, outdir=OUTDIR):
  '''Peak-time spectra of all sweep points together, nu/nu_m axis (mode: SPEC_MODES).'''
  ipeak = {id(r): det[3].get('i_peak') for r, det in zip(results, detections)}
  def get_peak(r):
    return r['nuFnu'][ipeak[id(r)], :]
  _plot_spectra_all(results, get_peak, mode, f'Peak spectra ({_MODE_TITLE[mode]})',
      f'peak_spectra_norm-{mode}.png', outdir)


def plot_fluence_all(results, mode, outdir=OUTDIR):
  '''Time-integrated (fluence) spectra of all sweep points, nu/nu_m axis (mode: SPEC_MODES).'''
  def get_fluence(r):
    return compute_fluence_spectrum(r['Tb'], r['nuFnu'])
  _plot_spectra_all(results, get_fluence, mode,
      f'Time-integrated spectra ({_MODE_TITLE[mode]})',
      f'fluence_spectra_norm-{mode}.png', outdir, sym='\\nu \\mathcal{F}_\\nu')


def compute_efficiency(r):
  '''Radiative efficiency eps_rad = E_rad / E_inj of one sweep point (comoving
  energy budget), E_inj being the energy actually deposited in the truncated
  electron power law. Caches written before E_inj existed fall back to the old
  E_rad/(eps_e*E_int), which overestimates the budget by 1/derive_xiE (up to 6%
  deep in fast cooling, where gma_M/gma_m is smallest). Returns nan if the
  energies are not in the cache.'''
  if 'E_rad' not in r:
    return float('nan')
  if r.get('E_inj', 0.) > 0.:
    return r['E_rad'] / r['E_inj']
  if all(k in r for k in ('E_int', 'eps_e')) and r['E_int'] > 0.:
    return r['E_rad'] / (r['eps_e'] * r['E_int'])
  return float('nan')


def plot_radiative_efficiency(results, outdir=OUTDIR):
  '''
  Radiative efficiency eps_rad = E_rad/E_inj vs the cooling regime
  log10(gamma_c/gamma_m). E_rad = total comoving radiated energy summed over shocked
  cells; E_inj = energy actually given to the accelerated electrons, i.e. the
  truncated power law between gma_m and gma_M. The printed 'xi_E' column is
  E_inj/(eps_e*sum E'_int) < 1, the fraction of eps_e*e'_int that a distribution
  cut at gma_M can hold (the old denominator); it drops deep in fast cooling
  because the alpha rescale shrinks gma_M ~ alpha**(3/4) at fixed gma_m.
  Fast cooling -> eps_rad ~ 1; slow cooling -> eps_rad < 1.
  '''
  logr = np.array([r['log10ratio'] for r in results], float)
  eff = np.array([compute_efficiency(r) for r in results], float)
  if not np.any(np.isfinite(eff)):
    print('radiative efficiency: energies absent from cache (re-run with use_cache=False)')
    return
  print(f"\n{'log10(gc/gm)':>12} {'E_rad':>12} {'E_inj':>12} {'eps_e*E_int':>12} "
        f"{'xi_E':>7} {'eps_rad':>9}")
  for r in results:
    ei = r.get('eps_e', np.nan) * r.get('E_int', np.nan)
    einj = r.get('E_inj', np.nan)
    print(f"{r['log10ratio']:+12.0f} {r.get('E_rad', np.nan):12.4e} {einj:12.4e} "
          f"{ei:12.4e} {einj/ei:7.4f} {compute_efficiency(r):9.4f}")
  fig, ax = plt.subplots()
  ax.plot(logr, eff, 'o-', color='C2')
  ax.axhline(1., color='grey', ls=':', lw=.9)
  ax.set_xlabel('$\\log_{10}(\\gamma_c/\\gamma_m)$')
  ax.set_ylabel('$\\varepsilon_{\\rm rad}=E_{\\rm rad}/E_{\\rm inj}$')
  fig.tight_layout()
  fig.savefig(os.path.join(outdir, 'radiative_efficiency.png'), dpi=300)
  plt.close(fig)


def main(key=DEFAULT_KEY, log10ratio_arr=LOG10RATIO_ARR, outdir=None, use_cache=True,
    nproc=None, method=DEFAULT_METHOD, z=Z_SHELL):
  '''
  Full figure set of one (key, method, z) sweep. z selects the emitting shell:
  Z_SHELL=4 (reverse shock) or 1 (forward shock); both are computed on the
  REVERSE-shock observer grids (see _compute_point), so the bar{T} axis and the
  hydro-time marks below are in RS units whatever z is.
  '''
  outdir = method_outdir(method, key, z) if outdir is None else outdir
  os.makedirs(outdir, exist_ok=True)
  results = load_sweep(outdir) if use_cache else None
  if results is None:
    # skip_cached MUST be forwarded: run_sweep defaults it to True, so without this
    # use_cache=False would bypass load_sweep and then silently skip every point that is
    # already on disk -- recomputing nothing and re-plotting the stale cache. (That is
    # exactly what it did after the A^(p-1) normalisation landed.) sweep_efficiency.main
    # has always forwarded it; these two now behave the same.
    results = run_sweep(key, log10ratio_arr, z=z, outdir=outdir, nproc=nproc,
                        method=method, skip_cached=use_cache)

  detections = []
  for r in results:
    det = detect_rise_peak_tail(r['Tb'], r['nub'], r['nuFnu'])
    detections.append(det)
    T_rise, T_peak, T_tail, info = det
    warnings = {k: v for k, v in info.items() if 'warn' in k}
    print(f"target={r['log10ratio']:+.1f}  T_rise={T_rise:.4g}  "
          f"T_peak={T_peak:.4g}  T_tail={T_tail:.4g}  {warnings}")

  # nu_m must be in-grid for the nu_m-normalised spectra
  for r in results:
    x = nu_over_num(r)
    assert x.min() < 1. < x.max(), \
        f"nu_m off-grid for log10ratio={r['log10ratio']:+.1f} (nu/nu_m in [{x.min():.1e},{x.max():.1e}])"

  barT_f = exit_onset_barT(key, z=z)
  print(f'lightcurve time normalisation: crossing bar_T_f = {barT_f:.4f}')
  barT_off = rarefaction_off_barT(key, z=z)
  if barT_off is not None:
    print(f'rarefaction cut-off: first cell at bar_T = {barT_off[0]:.4f}, last (emission '
          f'stops) at {barT_off[1]:.4f}  -> bar_T/bar_T_f in '
          f'[{barT_off[0]/barT_f:.3f}, {barT_off[1]/barT_f:.3f}]')
  plot_lightcurve_shape(results, barT_f, barT_off=barT_off, outdir=outdir)
  plot_spectra_per_regime(results, detections, outdir=outdir)
  # the same two figures unannotated, as '_plain' series (see each function's docstring).
  # Named so that no existing glob picks them up -- ARTICLE_SERIES matches on
  # 'lightcurve_shape_nu=*', which '_plain' breaks by construction
  plot_lightcurve_shape(results, barT_f, barT_off=barT_off, outdir=outdir, annotate=False)
  plot_spectra_per_regime(results, detections, outdir=outdir, segments=False)
  # breaks from the Granot & Sari shape fit rather than the knee scan (track_breaks):
  # unbiased break positions, a fitted nu_M, and a regime label the fit chooses
  # barT_off[1] bounds where the SC -> FC swap may be DETECTED, deliberately the same
  # bound for both methods: it is where the shell's on-axis emission ends under the
  # modelled cut, and for the reference method (which keeps emitting on-axis to
  # bar{T} ~ 650) it is still well past the peak, so the transition is inside it either
  # way. Sharing the bound keeps the two methods' tracks directly comparable.
  tracks = [track_breaks_gs02(r, barT_swap_max=(barT_off[1] if barT_off else barT_f))
            for r in results]
  fits = [fit_break_evolution(r, tr, barT_f, barT_off) for r, tr in zip(results, tracks)]
  # thin-shell reference for nu_m: one curve for the whole sweep (a_u, tau are invariant
  # under the alpha rescaling), computed once and handed to both consumers
  c25 = c25_num_curve(key, z, results[0]['Tb'])
  plot_break_evolution(results, tracks, fits, barT_f, barT_off=barT_off, outdir=outdir,
                       c25=c25)
  plot_break_ratio(results, tracks, barT_f, barT_off=barT_off, outdir=outdir)
  build_break_evolution_table(results, fits, outdir=outdir, tracks=tracks, c25=c25)
  plot_gs02_fits(results, detections, outdir=outdir)
  plot_gs02_rms(results, barT_f, barT_off=barT_off, outdir=outdir)
  build_gs02_table(results, detections, outdir=outdir)
  for mode in SPEC_MODES:
    plot_peak_spectra_all(results, detections, mode, outdir=outdir)
    plot_fluence_all(results, mode, outdir=outdir)
  # build_regime_table(results, detections, outdir=outdir)
  plot_radiative_efficiency(results, outdir=outdir)

  trim_pngs(outdir)
  copy_article_figures(outdir)
  print(f'Figures saved to {outdir}')
  return results, detections


def trim_pngs(outdir=OUTDIR):
  '''Auto-trim surrounding whitespace from the saved figures, i.e. run
  'mogrify -trim *.png' (ImageMagick) on outdir. No-op if mogrify is absent.'''
  import subprocess, shutil
  if shutil.which('mogrify') is None:
    print('trim_pngs: mogrify (ImageMagick) not found; skipping whitespace trim')
    return
  pngs = glob.glob(os.path.join(outdir, '*.png'))
  if pngs:
    subprocess.run(['mogrify', '-trim'] + pngs, check=False)
    print(f'trimmed {len(pngs)} figures with mogrify -trim')


ARTICLE_DIR = os.path.join(GAMMA_dir, 'bin', 'Tools', 'figures', 'article_choice')
ARTICLE_SERIES = {        # {source figure dir: globs of the series picked for the article}
  'gammacm_sweep_data': ('lightcurve_shape_nu=*.png', 'peak_spectra_norm-*.png'),
  'rarcut_compare': ('lightcurve_cmp_lin_nu=*.png',   # the LINEAR series only: a bare
                                                      # 'lightcurve_cmp_lin*' would also
                                                      # sweep up the linlog variants
                     'fluence_spectra_cmp_norm-*.png',
                     'spectrum_evolution_cmp_logr=*.png'),
}


def copy_article_figures(outdir, article_dir=ARTICLE_DIR, series=ARTICLE_SERIES):
  '''
  Mirror the figures picked for the article into one flat folder. Called by every main
  that writes a source directory named in ARTICLE_SERIES, right after trim_pngs, so the
  selection is refreshed whenever those figures are rebuilt and the article folder can
  never hold a stale copy of a figure that has since changed.
  Keyed on the EXACT source directory name, so the '_z=1' and other suffixed variants are
  deliberately not mirrored; add an entry to ARTICLE_SERIES to include a new series.
  '''
  import shutil
  globs = series.get(os.path.basename(os.path.normpath(outdir)))
  if not globs:
    return []
  os.makedirs(article_dir, exist_ok=True)
  copied = []
  for g in globs:
    for f in sorted(glob.glob(os.path.join(outdir, g))):
      shutil.copy2(f, os.path.join(article_dir, os.path.basename(f)))
      copied.append(os.path.basename(f))
  print(f'copied {len(copied)} figures to {article_dir}' if copied else
        f'copy_article_figures: nothing matched in {outdir}')
  return copied


if __name__ == '__main__':
  main()
