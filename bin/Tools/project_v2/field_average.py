# -*- coding: utf-8 -*-
# @Author: acharlet

'''
The sweep parameter gamma_c/gamma_m is built on a magnetic field the shell only has at R0.
This module measures what that costs, compares the correction with the break ratio the
time-integrated spectra actually show, and -- via measure_field_correction -- lets a run
adopt the corrected quantity as its sweep parameter.

WHAT THE LABEL ASSUMES. phys_functions_shells.shells_add_radNorm writes

    gamma_c = 6 pi m_e c Gamma_RS / (sigma_T B'^2 t_RS),     B' = sqrt(8 pi eps_B e'_3),

with e'_3 the immediate post-shock internal energy of shell 4 AT R0 and t_RS/Gamma_RS the
shock's proper crossing time. Read as physics that is "an electron injected at R0 cools for
the crossing time, at the rate it had when it was injected" -- a constant-rate cooling in a
frozen field. The field is not frozen, and the honest statement is

    1/gamma_c = (sigma_T/6 pi m_e c) int B'^2 dt'      over the shock propagation,

    dt' = dt/Gamma(t),      both B' and Gamma taken ALONG THE SHOCK'S WORLDLINE.

THE Gamma_RS IN THAT FORMULA IS RECENT (it was Gamma_0). The crossing is a pair of events,
"shock at R0" and "shock at R_f", and they lie on the SHOCK's worldline: Gamma_0 is the
shocked fluid's Lorentz factor and converts durations for a comoving observer who is not
present at both events. Switching to lfacRS/lfacFS moved the label by -0.063 dex (RS) and
+0.033 dex (FS) and moved exactly that much OUT of C -- the corrected gamma_c is unchanged
(5.31 on the RS either way), and so is every ratio measured against it below. Cached sweeps
predate it; fluence_break_rows re-derives each point's label from the live env and the
point's own alpha so old caches are still compared against the definition in force now.

GAMMA_M IS NOT CORRECTED, deliberately. gamma_c is a cooling estimator and what it is
measured against is the INJECTION characteristic Lorentz factor, so gamma_m stays at
gamma_m,0 and the correction to gamma_c/gamma_m IS the correction to gamma_c. Two
temptations, both declined:
  - letting gamma_m fall adiabatically as (rho/rho_0)^(1/3). Its prefactor then cancels
    against the same factor in the exact cooled cutoff and leaves a (rho/rho_0)^(1/3)
    WEIGHT inside the cooling integral, raising the correction from 2.43 to ~2.9 on the
    RS. That is a statement about the evolved distribution, not about the injection
    scale the sweep is labelled by.
  - giving each cell its own gamma_m,0 (it falls to 0.62 x nominal by the end of the RS
    crossing, since the shock weakens). That is a real spread in the shell, but it is a
    spread in the INJECTION scale, not in the cooling, and folding it in would inflate
    the per-cell correction by up to 1.6x.
Both were measured before being dropped; the numbers above are what they would add.

IT IS A RIGID SHIFT OF THE WHOLE SWEEP AXIS. Under the Granot alpha rescaling that
generates the sweep (sweep_gammacm.compute_alpha_sweep) lengths and times go as alpha and
rho, p as alpha^-3, so B'^2 dt' -- and therefore everything below -- is alpha^-2, exactly
as gamma_c itself is (checked numerically on B'^2 t_RS/Gamma_RS). The dimensionless decay
B'^2/B'^2(R0) against t/t_cr is the SAME curve at every sweep point. One number corrects
all eight.

WHICH CLOCK, AND WHAT GAMMA ACTUALLY DOES. t'_cr is measured on the shock's own worldline
as the invariant interval sum, sqrt(dt^2 - dx^2) over the front's trajectory -- not on the
worldline of any one cell. Three clocks are measured and all three are reported, because
the answer is not what one would guess:

                                    RS (z=4)              FS (z=1)
    Gamma_RS (the label, analytic)   109.50                136.56
    Gamma_0 (what it used to use)    126.45                126.45
    shock front's own trajectory     112.57 -> 111.95      136.16 -> 135.39
    fluid just behind the front      130.00 -> 129.28      126.08 -> 125.36
    first cell, along its worldline  130.00 -> 127.76      126.08 -> 127.82
    t'_cr / (t_cr/Gamma_RS), shock    0.9741                1.0047
    ... same, fluid clock             0.8435                1.0851

  GAMMA BARELY EVOLVES ANYWHERE. Over the crossing every clock moves by 0.55%, and along a
  cell's whole worldline -- rarefaction included -- the fluid Gamma only wanders 126-133.5
  (max 133.5, reached at t ~ 500-700 t_cr). There is no substantial increase to capture,
  on the shock's worldline or any other. What was substantial was the shock trajectory's
  OFFSET from Gamma_0 (-11% RS, +7.8% FS), and that now lives in the LABEL -- it is the
  Gamma_RS fix above. What is left here is only the 2.5% between the analytic lfacRS and
  the trajectory the simulation actually follows (109.50 against 112.2).
  The clock changes only the TOTAL t'_cr, never the shape of the average: <B'^2>/B'^2(R0)
  comes out at 0.3455 (RS) and 0.3635 (FS) under both, to four decimals, because a Gamma
  that flat divides out of a time average.
  Caveat worth stating once: the cooling equation runs on the FLUID's proper time, since
  B' is the comoving field, so the fluid clock is what an electron's losses integrate
  against, while the shock clock times the crossing as a process. Both are carried
  (`clock='shock'|'fluid'`); the correction is

    RS   2.811 (shock)   3.247 (fluid)        FS   2.748 (shock)   2.545 (fluid)

  i.e. the choice is worth +15% on the RS and -7% on the FS, and it is the largest single
  uncertainty in the number. `clock='shock'` is the default, matching the label's own
  Gamma_RS; nothing downstream assumes it, and C_avg (below) is free of it entirely.

MEASURED, cooling_g100, injection states from the shock fit (EARLY_ANA), clock='shock'.

                                          RS (z=4)      FS (z=1)
    <B'^2>/B'^2(R0) over the propagation    0.3455        0.3635  -> B'_rms = 0.6 B'(R0)
    C_avg = 1/<B'^2>_norm (the definition)  2.894         2.751     +0.462 / +0.440 dex
    t'_nom/t'_cr   (the clock)              0.9741        1.0047
    B'^2_ana/B'^2(R0) (analytic vs sim)     0.9461        1.0038
    C = their product                       2.811         2.748
    injection field, first -> last cell   1.057->0.114  0.996->0.129
    C_i along a cell's worldline: crossing / end / shell q05 / median / q95
                                2.84/2.27/2.58/6.58/26.6  2.62/2.28/2.58/6.82/46.2

  So gamma_c/gamma_m is understated by ~2.8 on both shells -- +0.45 dex -- for the material
  the label describes. In the quantity a spectrum shows, the break RATIO, that is x7.9 and
  x7.6. C_avg is doing nearly all of the work; the clock and the analytic-vs-simulated
  field are now only 0.4-5% each, both having had their large parts absorbed into the
  label's own Gamma_RS.
  THE COOLING SATURATES past t ~ 2-3 t_cr (B'^2 has fallen ~50x, the rarefaction has
  crossed), so the per-cell C_i integrated to the END of each worldline is a converged
  number and not a window choice -- it is the total cooling those electrons will ever
  suffer, which is the endpoint a TIME-INTEGRATED spectrum sees. The first cell's
  C_i(end) = 2.27 against C_i(crossing) = 2.84: stopping at the crossing or following the
  cooling to exhaustion differ by 25%, which bounds how much the endpoint can matter.

AGAINST THE SPECTRA. sweep_gammacm.compute_fluence_spectrum on each cached point, then
spectral_breaks.smoothing_from_identified (the segment route, segment_route.py) for the two
breaks; their ratio is (gamma_c/gamma_m)^2 for whatever material the spectrum is
effectively showing, so measured/nominal is the offset the correction has to explain.
`nominal` is the shell's own (gamma_c/gamma_m)^2 -- on the FS that is 10^(2 logr) x 15.78,
since the sweep axis labels the RS and the FS sits 0.599 dex above it. data_rarcut (the article's sweep), the points whose
class puts both breaks in band and measures the mid slope on an identified segment:

sqrt(nu_c/nu_m) is then a MEASURED gamma_c/gamma_m, to be read against the label:

    shell logr class sqrt(nu_c/nu_m) gma_c/gma_m  ratio    C  ratio/C C_avg ratio/C_avg pct
     RS    -2   FC       0.02726        0.008659   3.15  2.81   1.12   2.89    1.09     14%
     RS    +1   SC      86.67           8.659     10.01  2.81   3.56   2.89    3.46     68%
     RS    +2   SC     710.6           86.59       8.21  2.81   2.92   2.89    2.84     60%
     FS    +0   SC      41.71           3.439     12.13  2.75   4.41   2.75    4.41     71%
     FS    +1   SC     287.7           34.39       8.37  2.75   3.04   2.75    3.04     58%
     FS    +2   SC    2601            343.9        7.56  2.75   2.75   2.75    2.75     54%

  (`gma_c/gma_m` here is the CURRENT definition's label, re-derived per point; ratio/C and
  the percentiles are invariant under the Gamma_RS change, since it moved label and
  correction by the same factor.)

  FAST COOLING: THE CORRECTION ACCOUNTS FOR IT OUTRIGHT. The one FC point comes out at
  ratio/C = 1.12 against full C and ratio/C_avg = 1.09 against the field average alone --
  the offset is explained to within 12% either way, with no free parameter. Equivalently the factor the spectrum asks for sits at the
  14th percentile of the shell's own C_i, among the earliest-shocked cells in the strongest
  field, which are the least-corrected ones. That is where it belongs: in fast cooling
  nu_c is set by the material that has cooled the MOST, and C is C_i for that material.
  Which of the two to quote is not decidable from one point: the 18% between them is the
  clock and the analytic-vs-simulated field, both ~10% effects (see WHICH CLOCK), and one
  measurement cannot separate them. C_avg is the more robust of the two in that it is
  clock-free.
  SLOW COOLING: SHORT BY 2.7-4.4x, whichever version is used, and consistently so on both
  shells (against C: RS 2.9-3.6, FS 2.8-4.4; against C_avg: RS 2.8-3.5, FS 2.8-4.4),
  falling monotonically as the regime gets more slowly-cooling. The leftover is not noise
  and not a normalisation -- it survives stripping C down to the field average, which is
  what stripping it to its clock-free core does. It is that the break is set further along
  the shell, at 54-71% of the C_i distribution rather than at its low end: with gamma_c far
  above gamma_m no cell has cooled much, no population dominates, and the break lands in
  the body of the distribution.
  So a single scalar CANNOT correct the whole sweep to better than the width of C_i --
  about half a decade in the ratio between the fast- and slow-cooling ends, and the
  regime-dependent part of the answer is not in C at all. What the correction DOES buy is
  that the offset stops being unexplained: every measured point lands inside a distribution
  computed from the hydro alone, at the percentile its regime predicts.

  The MC points (RS -1, 0; FS -2, -1) are excluded, not missing: their mid slope is fitted
  free inside the shape fit (no mid segment exists to measure), so their crossings carry
  the spread segment_route.py warns about -- a_mid comes back at 0.67, 1.11, 0.54, 1.24
  against asymptotes of 1/2 and 1/4, and the offsets (1.2, 96, 5.3, 0.18) straddle
  everything. The VFC/FC* points (-5..-3) have no nu^(4/3) segment in band at all, so no
  lower break and no ratio; that is a band limit, not a failure.

CAVEATS.
  The nominal B'(R0) is analytic; the simulation's own first-shocked state carries B'^2
  1.057x that (RS), 0.996x (FS). C is normalised on the ANALYTIC value, because what is
  being corrected is the label -- the `b2_ana_over_sim` factor separates the two.
  t_RS is the planar analytic crossing time; the spherical run takes longer, so 33 of 500
  RS cells (7 of 500 FS) are shocked after it. They are kept -- C_i(end) does not use t_RS
  as an endpoint -- but C_i(crossing) is meaningless for them and comes back NaN.
  One outer-edge cell per shell (k = Next) never gets a clean shocked state (its injection
  field is 1e-4 of nominal, it sits in the shell's own edge rarefaction); it is dropped by
  INJ_FLOOR rather than allowed to set a percentile.
  Parent cells only, as radiative_length: a sub-cell is a slice of its parent's history,
  not an independent emitter, and counting them would weight the early cells by their
  sub-cell count in every quantile here.

ADOPTING IT AS THE SWEEP PARAMETER (what to do for a new run).

    python -c "import field_average as F; F.measure_field_correction('<key>')"

writes results/<key>/field_correction.json, after which MyEnv('<key>').gma_c carries C_avg,
sweep_gammacm's alpha lever targets the CORRECTED log10(gamma_c/gamma_m), and every sweep
cache for that run lands in a '_fc' directory so the two definitions can never be confused
on reload. Run it once, after analysis_hydro.extract_data_thinshell (it needs only the
shock-front table, not the cell histories) and BEFORE the sweep -- compute_alpha_sweep picks
its alphas from the definition in force at that moment.

Nothing is retroactive and nothing is global: a run with no sidecar is bit-identical to
before. cooling_g100 deliberately has none, so the local sweeps stay as they are; the
correction is for the hi-res run, whose own C_avg must be measured on its own hydro (the
number is alpha-invariant but not run-invariant).

Example use:
  python -c "import field_average as F; F.main()"
  python -c "import field_average as F; F.main(use_cache=False)"
  python -c "import field_average as F; F.field_average(z=4, clock='fluid')"
'''

import os
import csv
import json
import numpy as np
import matplotlib.pyplot as plt

from environment import (MyEnv, field_correction, field_correction_path,
    FIELD_CORR_VERSION, FIELD_CORR_TAG)
from IO import get_variable
from phys_constants import pi_, c_
from phys_functions import derive_Eint_comoving, derive_Lorentz
from working_cooling import check_extracted_cells
from working_cooling_data import _load_cell_history, load_shockfront_states
import spectral_breaks as sb
import sweep_gammacm as swp
from sweep_gammacm import (DEFAULT_KEY, EARLY_ANA, method_outdir, trim_pngs)
from plotting_functions import COL_RS, COL_FS
from mid_slope_evolution import INK, MUTED, GRID

KEY = DEFAULT_KEY
METHOD = 'data_rarcut'        # the article's sweep (sweep_gammacm.ARTICLE_SERIES); the
                              # correction itself is method-independent (it is hydro), only
                              # the measured break ratios are read from a method
Z_RS, Z_FS = 4, 1
CLOCK = 'shock'               # 'shock': proper time along the shock front's own trajectory,
                              # as the invariant interval. 'fluid': dt/Gamma of the material
                              # just behind it -- the clock the cooling equation actually
                              # runs on. See the header: they differ by 12% (RS), 7% (FS),
                              # essentially all of it a constant offset, not evolution.
N_SETTLE = 1                  # as the emission pipeline: the injection row is the shock-fit
                              # state prepended by _load_cell_history, so the correction and
                              # the flux see the SAME first step
INJ_FLOOR = 0.1               # drop a cell whose injection gamma_m is below this fraction of
                              # the shell's nominal one: not a weak shock but no shock at all
                              # (the outer-edge cell, see CAVEATS)
T_FRACS = (0.25, 0.5, 1., 1.5, 2., 3., 5., 10.)   # t/t_cr sampling of the running average
QS = (0.05, 0.25, 0.5, 0.75, 0.95)                # quantiles reported over the shell
CELLS_CSV = 'field_average_cells.csv'
BREAKS_CSV = 'field_average_breaks.csv'
FIG = 'field_average.png'
CELL_FIELDS = ('z', 'k', 'ts_frac', 'b2_0', 'gm_0', 'C_cr', 'C_end')
BREAK_FIELDS = ('z', 'logr', 'regime', 'b_lo', 'b_hi', 'a_mid', 'ratio', 'nominal',
                'offset')
# shape classes whose mid slope is MEASURED on an identified segment, hence whose two
# crossings can be read as nu_c and nu_m. MC's mid is fitted free (segment_route.py), VFC
# and FC* have no lower break in band.
TRUSTED = ('FC', 'SC')


# ---------------------------------------------------------------------------
# the hydro side

def _cumtrapz0(y, x):
  '''Cumulative trapezoid with out[0] = 0 (working_cooling_data._cumtrapz0's convention).'''
  return np.concatenate(([0.], np.cumsum(0.5*(y[1:] + y[:-1])*np.diff(x))))


def comoving_field(rho, p, env):
  '''B'^2 = 8 pi eps_B e'_int, the field derive_syn_cooling builds the cooling rate from.'''
  return 8.*pi_*env.eps_B*derive_Eint_comoving(np.asarray(rho, float),
                                               np.asarray(p, float), env.rhoscale)


def shell_klist(key, z, env):
  '''Parent cells of shell z, in shocking order, restricted to what is on disk.'''
  k4 = env.Next
  kCD = k4 + env.Nsh4
  klist = np.flip(np.arange(k4, kCD)) if z == 4 else np.arange(kCD, kCD + env.Nsh1)
  done = set(check_extracted_cells(key).tolist())
  return np.array([k for k in klist if k in done])


def shell_nominal(env, z):
  '''
  (B'^2, gamma_m, gamma_c, t_cross, Gamma_cross) of shell z as the label defines them.

  Gamma_cross is the SHOCK's analytic Lorentz factor (lfacRS / lfacFS), which is the clock
  shells_add_radNorm now divides the crossing time by -- the crossing's two events sit on
  the shock's worldline, not on a fluid element's. Everything here normalises on
  t_cross/Gamma_cross for that reason, so the correction reported is what is left AFTER
  the label already uses the right clock analytically.
  '''
  if z == 4:
    return env.Bp**2, env.gma_m, env.gma_c, env.tRS, env.lfacRS
  return env.BpFS**2, env.gma_mFS, env.gma_cFS, env.tFS, env.lfacFS


def cooling_integrals(hist, env):
  '''
  One cell's worldline, in cgs: its comoving clock and its accumulated cooling.

    tp    comoving time since the cell was shocked, int dt/Gamma over the snapshots
    I_B   int B'^2 dt'  -- so 1/(alpha_ I_B) is the gamma_c that cell has actually
          reached, synchrotron losses only, which is what gamma_c is defined by

  No adiabatic weight: see GAMMA_M IS NOT CORRECTED in the header. With gamma_m held at
  injection the consistent partner is the purely radiative gamma_c, and the weight
  cooling_frequency.cell_cooling_lfac carries belongs to the evolved distribution.
  '''
  t = hist.t.to_numpy(dtype=float)
  lfac = derive_Lorentz(hist.vx.to_numpy(dtype=float))
  B2 = comoving_field(hist.rho.to_numpy(dtype=float), hist.p.to_numpy(dtype=float), env)
  tp = np.concatenate(([0.], np.cumsum(np.diff(t)/(0.5*(lfac[1:] + lfac[:-1])))))
  return dict(t=t, tp=tp, B2=B2, lfac=lfac, x=hist.x.to_numpy(dtype=float),
              I_B=_cumtrapz0(B2, tp))


def comoving_crossing(q, tcr):
  '''
  Comoving time a CELL has had when the shock finishes crossing, int_0^{t_cr} dt/Gamma
  along its own worldline. Used for the per-cell C_i(crossing) and to set the figure's
  axis; the headline correction takes its clock from the shock instead (shock_worldline).
  Returns NaN for a cell shocked after t_cr (np.interp would silently clamp to tp[0]).
  '''
  return np.interp(tcr, q['t'], q['tp']) if q['t'][0] < tcr else np.nan


def shock_worldline(key=KEY, z=Z_RS, env=None, clock=CLOCK, source=None):
  '''
  The shock's own propagation: the freshly shocked state at the front, cell by cell, and
  the correction that follows from averaging over it.

  THE CLOCK. 'shock' takes the proper time along the front's trajectory as the invariant
  interval, dtau = sqrt(dt^2 - dx^2) with x in light-seconds -- differencing the trajectory
  rather than differentiating it, which matters at Gamma ~ 112 where beta - 1 ~ 4e-5 and
  np.gradient would be differencing the same nearly-equal numbers anyway (the two agree to
  0.01% here). 'fluid' takes dt/Gamma of the material just behind the front, which is the
  clock the cooling equation runs on since B' is the comoving field.

  Returns the front arrays plus, for the chosen clock, the crossing proper time, the
  time-averaged field, and the three factors the correction decomposes into.
  '''
  env = MyEnv(key) if env is None else env
  B2ana, _, gc_nom, tcr, lfcr = shell_nominal(env, z)
  # source: which injection-state convention the cells actually start from. EARLY_ANA
  # ('shockfit') is the historical one; 'measured' takes the event from each cell's own
  # velocity jump and the state from the fitted profiles at THAT radius
  # (working_cooling_data.measured_injection_event). b2_ana_over_sim compares the analytic
  # B'_0 against sel's FIRST state, so it means "analytic vs the state the cells are given"
  # -- and that state changes with the source, which is why this is a parameter.
  sh = load_shockfront_states(key, z, env,
                              source=(EARLY_ANA if source is None else source)).sort_values('t')
  sel = sh.loc[sh.t <= tcr]
  t = sel.t.to_numpy(dtype=float)
  x = sel.x.to_numpy(dtype=float)
  B2 = comoving_field(sel.rho.to_numpy(dtype=float), sel.p.to_numpy(dtype=float), env)
  lf_fluid = derive_Lorentz(sel.vx.to_numpy(dtype=float))
  # invariant interval along the front's trajectory, and the Gamma it implies per step
  dt, dx = np.diff(t), np.diff(x)
  dtau = np.sqrt(np.clip(dt*dt - dx*dx, 0., None))
  lf_shock = np.divide(dt, dtau, out=np.full_like(dt, np.nan), where=dtau > 0.)
  if clock == 'shock':
    dtp = dtau
  elif clock == 'fluid':
    dtp = dt/(0.5*(lf_fluid[1:] + lf_fluid[:-1]))
  else:
    raise ValueError(f"clock must be 'shock' or 'fluid', got {clock!r}")
  tpcr = float(dtp.sum())
  I_B = float((0.5*(B2[1:] + B2[:-1])*dtp).sum())      # int B'^2 dt' along the front
  tp_nom = tcr/lfcr                                    # the label's own proper crossing
  return dict(z=z, clock=clock, t=t, x=x, B2=B2, lfac_fluid=lf_fluid, lfac_shock=lf_shock,
              tpcr=tpcr, tp_nom=tp_nom, I_B=I_B,
              mean_B2=I_B/(tpcr*B2[0]),                # <B'^2>/B'^2(R0) over the crossing
              b2_ana_over_sim=B2ana/B2[0],
              clock_fac=tp_nom/tpcr,
              C=B2ana*tp_nom/I_B,                      # the correction to the label
              gc_nom=gc_nom, b2_first=B2[0]/B2ana, b2_last=B2[-1]/B2ana)


def measure_field_correction(key=KEY, zlist=(Z_RS, Z_FS), env=None, write=True,
    verbose=True, quantity='C', clock='fluid', source='measured'):
  '''
  MEASURE THE RUN'S FIELD CORRECTION AND MAKE IT THE SWEEP'S DEFINITION.

  C_avg = 1/<B'^2>_norm over the shock propagation, per shell, written to the run's
  field_correction.json sidecar. From then on MyEnv(key) reports a gamma_c corrected by it
  (environment.field_correction -> shells_add_radNorm), so sweep_gammacm's alpha lever
  targets the CORRECTED log10(gamma_c/gamma_m) and every figure axis means the corrected
  quantity. sweep caches written afterwards land in a '_fc'-suffixed directory, so a
  corrected sweep can never be reloaded as an uncorrected one or vice versa.

  WHY C_avg AND NOT THE FULL C. C_avg is the only one of C's three factors that is
  clock-free -- a Gamma this flat divides out of a time average, so it is identical under
  clock='shock' and 'fluid' while C differs by 13% -- and it is the one that is a property
  of the FIELD rather than of a comparison with the analytic setup. A definition should not
  inherit a 10% choice that one measurement cannot settle. The other two factors (the
  analytic-vs-simulated B'_0, and the clock) stay in field_average's reporting, where they
  can be quoted as a systematic.

  CHEAP AND EARLY: this reads the shock-front table (run_data_{z}.csv via
  load_shockfront_states), not the cell histories, so it can be run as soon as
  analysis_hydro.extract_data_thinshell has been -- before any cell extraction, and long
  before a sweep. That is the point: the correction has to exist before compute_alpha_sweep
  picks its alphas, or the sweep targets the old definition.

  RUN IT ONCE PER SIMULATION. The number is alpha-invariant, so it is a property of the
  hydro alone; re-running it on the same run rewrites the same value. Running it on a
  DIFFERENT run gives a different value, which is why it lives beside the run and not in a
  module constant.

  write=False measures without touching the sidecar (what to use to look before leaping).
  '''
  env = MyEnv(key) if env is None else env
  out, rows = {}, []
  for z in zlist:
    fr = shock_worldline(key, z, env, clock=clock, source=source)
    # quantity='C': ALL THREE measured factors -- the field average over the propagation,
    # the analytic-vs-simulated B'_0, and the comoving crossing time -- so gamma_c reflects
    # the simulation rather than the analytic setup. 'C_avg' is the historical field
    # average alone, kept so the old definition can still be reproduced.
    out[z] = float(fr['C'] if quantity == 'C' else 1./fr['mean_B2'])
    rows.append((z, fr['mean_B2'], 1./fr['mean_B2'], fr['C']))
  if verbose:
    print(f'--- field correction for {key} ---')
    print(f'{"shell":>6} {"<B^2>/B0^2":>11} {"C_avg":>8} {"dex":>7}   (full C, for reference)')
    for z, m, c, C in rows:
      print(f'{("RS" if z == 4 else "FS"):>6} {m:11.4f} {c:8.4f} {np.log10(c):+7.4f}   '
            f'{C:.4f}')
  if write:
    path = field_correction_path(key)
    if path is None:
      raise ValueError(f'{key!r} is not a run directory: nowhere to write the sidecar')
    with open(path, 'w') as fh:
      json.dump(dict(version=FIELD_CORR_VERSION, key=key, clock=clock,
                     quantity=('C = (B0 ana/sim) x (clock) x C_avg, all measured'
                               if quantity == 'C' else
                               "C_avg = 1/<B'^2> over the shock propagation"),
                     injection_source=source,
                     written_by='field_average.measure_field_correction',
                     factor={str(z): float(c) for z, c in out.items()}), fh, indent=2)
    print(f'-> {path}   (MyEnv({key!r}).gma_c is now corrected; sweep caches get '
          f'{FIELD_CORR_TAG!r})')
  return out


def field_average(key=KEY, z=Z_RS, env=None, clock=CLOCK, fracs=T_FRACS, verbose=True):
  '''
  The headline correction and the numbers behind it.

  The correction proper comes from shock_worldline -- the field and the clock both averaged
  over the SHOCK's propagation, which is what the label is a statement about. The running
  table alongside it follows the FIRST-SHOCKED CELL instead, whose own field decay is what
  the electrons that cool the most actually experience; it is the same story told on a
  worldline, and it is what the shell distribution (shell_rows) generalises.
  '''
  env = MyEnv(key) if env is None else env
  B2ana, _, gc_nom, tcr, lfcr = shell_nominal(env, z)
  fr = shock_worldline(key, z, env, clock=clock)
  sh = load_shockfront_states(key, z, env, source=EARLY_ANA)
  k0 = int(shell_klist(key, z, env)[0])
  hist, _, _ = _load_cell_history(key, k0, N_SETTLE, env, sh_data=sh, early_frac=0.)
  q = cooling_integrals(hist, env)
  tp_cr = comoving_crossing(q, tcr)
  out = dict(front=fr, z=z, k0=k0, C=fr['C'], tp_cr_cell=tp_cr,
             C_cell_cr=B2ana*fr['tp_nom']/np.interp(tp_cr, q['tp'], q['I_B']),
             C_cell_end=B2ana*fr['tp_nom']/q['I_B'][-1], running=[])
  for f in fracs:
    tt = f*tp_cr
    if tt > q['tp'][-1]:
      continue
    ib = np.interp(tt, q['tp'], q['I_B'])
    out['running'].append(dict(frac=f, R_over_R0=np.interp(tt, q['tp'], q['x'])*c_/env.R0,
                               mean_B2=ib/(q['B2'][0]*tt), C=B2ana*fr['tp_nom']/ib))
  if verbose:
    lfs, lff = fr['lfac_shock'], fr['lfac_fluid']
    print(f'--- z={z}, over the shock propagation (clock={clock!r}) ---')
    print(f'  Gamma_shock (label) = {lfcr:.2f}, Gamma_0 = {env.lfac0:.2f} | '
          f'measured trajectory {lfs[0]:.2f} -> {lfs[-1]:.2f}'
          f' | fluid at the front {lff[0]:.2f} -> {lff[-1]:.2f}')
    print(f"  t'_cr = {fr['tpcr']:.5g} s against the label's t_cr/Gamma_shock = "
          f"{fr['tp_nom']:.5g} s  ({1./fr['clock_fac']:.4f}x)")
    print(f"  <B'^2>/B'^2(R0) = {fr['mean_B2']:.4f}   (injection field falls "
          f"{fr['b2_first']:.3f} -> {fr['b2_last']:.3f} x nominal)")
    print(f'  C = {fr["b2_ana_over_sim"]:.4f} (analytic vs simulated B\'_0) x '
          f'{fr["clock_fac"]:.4f} (clock) x {1./fr["mean_B2"]:.3f} (field average) '
          f'= {fr["C"]:.3f}')
    print(f'  nominal gamma_c = {gc_nom:.4g} -> corrected {gc_nom*fr["C"]:.4g}; '
          f'break ratio x {fr["C"]**2:.2f}')
    print(f'  ... along the first-shocked cell k={k0} instead '
          f'(C at the crossing / to the end): {out["C_cell_cr"]:.3f} / '
          f'{out["C_cell_end"]:.3f}')
    print(f'{"t/t_cr":>7} {"R/R0":>8} {"<B^2>/B0^2":>11} {"C_i":>8}')
    for r in out['running']:
      print(f'{r["frac"]:7.2f} {r["R_over_R0"]:8.3f} {r["mean_B2"]:11.4f} {r["C"]:8.3f}')
  return out


def shell_rows(key=KEY, z=Z_RS, env=None, verbose=True):
  '''
  Per-cell correction factors over a whole shell: each cell's own field decay, on its own
  worldline, against the one label the sweep gives all of them.

  C_i is normalised on the nominal 1/(alpha_ B'^2_ana t_cr/Gamma_0), so a row reads
  directly as "this cell's gamma_c is C_i times the sweep label's", and is reported both
  at the end of the crossing and at the end of the cell's history (where the cooling has
  saturated). The cell's own gamma_m,0 is recorded but NOT folded in -- see the header.
  '''
  env = MyEnv(key) if env is None else env
  B2ana, gm_nom, _, tcr, lfcr = shell_nominal(env, z)
  norm = B2ana*tcr/lfcr              # 1/(alpha_ * norm) is the nominal gamma_c
  sh = load_shockfront_states(key, z, env, source=EARLY_ANA)
  klist = shell_klist(key, z, env)
  rows, skipped = [], 0
  for k in klist:
    hist, _, _ = _load_cell_history(key, int(k), N_SETTLE, env, sh_data=sh, early_frac=0.)
    if hist is None or len(hist) < 3:
      skipped += 1
      continue
    gm0 = float(get_variable(hist.iloc[0], 'gma_m', env))
    if not np.isfinite(gm0) or gm0 < INJ_FLOOR*gm_nom:
      skipped += 1
      continue
    q = cooling_integrals(hist, env)
    tend = comoving_crossing(q, tcr)
    ib_c = np.interp(tend, q['tp'], q['I_B']) if np.isfinite(tend) else np.nan
    rows.append(dict(z=z, k=int(k), ts_frac=q['t'][0]/tcr, b2_0=q['B2'][0]/B2ana,
                     gm_0=gm0/gm_nom, C_cr=norm/ib_c, C_end=norm/q['I_B'][-1]))
  if verbose:
    print(f'  z={z}: {len(rows)} cells ({skipped} skipped), '
          f'{int(np.sum([r["ts_frac"] < 1. for r in rows]))} shocked before t_cr')
  return rows


# ---------------------------------------------------------------------------
# the spectral side: the break ratio of the time-integrated spectra

def fluence_break_rows(key=KEY, method=METHOD, z=Z_RS, verbose=True):
  '''
  The two breaks of every cached point's TIME-INTEGRATED spectrum, and the offset of their
  ratio from the sweep label.

  compute_fluence_spectrum integrates nu F_nu over the observer grid; the breaks come from
  the segment route (smoothing_from_identified), i.e. the crossings of the segments the
  spectrum actually shows. Which crossing is nu_c follows from the class, not from the
  label: fast cooling puts nu_c below nu_m, slow cooling above.

  `nominal` is the shell's OWN (gamma_c/gamma_m)^2 -- on the FS that is not 10^(2 logr),
  since the sweep axis labels the RS and the FS sits ~0.5 dex above it.

  IT IS RE-DERIVED FROM THE RUN'S LIVE env AND THE POINT'S ALPHA, not read off the point's
  cached env scalars. gamma_c ~ alpha^2 and gamma_m is alpha-invariant, so the live env
  reproduces any point exactly -- and doing it this way means a cache written under an
  OLDER gamma_c definition (before the t_cr/Gamma_shock fix, or before a field correction
  was installed) is still compared against the definition in force NOW. Otherwise `ratio`
  and C would have different denominators and their quotient would mean nothing.
  '''
  res = sorted(swp.load_sweep(method_outdir(method, key, z)),
               key=lambda r: r['log10ratio'])
  if not res:
    raise FileNotFoundError(f'no cached sweep for {method} z={z}: run sweep_gammacm.main')
  env0 = MyEnv(key)
  gc0, gm0 = ((env0.gma_c, env0.gma_m) if z == 4 else (env0.gma_cFS, env0.gma_mFS))
  rows = []
  for r in res:
    x = swp.nu_over_num(r)
    sp = swp.compute_fluence_spectrum(r['Tb'], r['nuFnu'])
    f = sb.smoothing_from_identified(x, sp, r['env'].psyn)
    logr = float(r['log10ratio'])
    nominal = (gc0*float(r['alpha'])**2/gm0)**2
    blo, bhi = f.get('b_lo'), f.get('b_hi')
    blo = np.nan if blo is None else float(blo)
    bhi = np.nan if bhi is None else float(bhi)
    ratio = np.nan
    if np.isfinite(blo) and np.isfinite(bhi) and blo > 0. and bhi > 0.:
      ratio = blo/bhi if nominal < 1. else bhi/blo
    rows.append(dict(z=z, logr=logr, regime=f['regime'], b_lo=blo, b_hi=bhi,
                     a_mid=float(f.get('a_mid', np.nan)), ratio=ratio, nominal=nominal,
                     offset=ratio/nominal))
  if verbose:
    print(f'--- time-integrated breaks, {method}, z={z} ---')
    print(f'{"logr":>5} {"class":>5} {"use":>4} {"nu_lo":>10} {"nu_hi":>10} {"a_mid":>7} '
          f'{"ratio":>10} {"nominal":>10} {"offset":>9} {"sqrt":>7}')
    for r in rows:
      use = 'yes' if r['regime'] in TRUSTED else '--'
      print(f'{r["logr"]:+5.1f} {str(r["regime"]):>5} {use:>4} {r["b_lo"]:10.3e} '
            f'{r["b_hi"]:10.3e} {r["a_mid"]:7.3f} {r["ratio"]:10.3e} '
            f'{r["nominal"]:10.3e} {r["offset"]:9.2f} '
            f'{np.sqrt(r["offset"]) if r["offset"] > 0 else np.nan:7.2f}')
    print("  use = '--': class whose crossings are not a nu_c/nu_m pair (see the header)")
  return rows


# ---------------------------------------------------------------------------
# comparison, csv, figure

def _quantiles(rows, key, qs=QS):
  v = np.array([r[key] for r in rows], dtype=float)
  v = v[np.isfinite(v)]
  return v, {q: float(np.quantile(v, q)) for q in qs}


def _percentile_of(v, value):
  '''Where a measured sqrt(offset) sits in the shell's own distribution, in per cent.'''
  v = np.sort(v[np.isfinite(v)])
  return 100.*np.searchsorted(v, value)/len(v)


def compare(cells, breaks, z=Z_RS, C=None, C_avg=None, verbose=True):
  '''
  THE COMPARISON THE MODULE EXISTS FOR, one row per usable spectral point.

    gc_gm_meas = sqrt(nu_c/nu_m) read off the TIME-INTEGRATED spectrum. A break ratio is
                 (gamma_c/gamma_m)^2 for whatever material the spectrum is showing and
                 nothing else, so its square root IS a measured gamma_c/gamma_m.
    gc_gm_nom  = the shell's own label (10^logr on the RS; 0.503 dex above it on the FS)
    ratio      = the first over the second: the factor the spectrum asks for
    C          = the factor the hydro supplies (shock_worldline), one number per shell
    C_avg      = 1/<B'^2>_norm, the FIELD AVERAGE ALONE -- C with its other two factors
                 (analytic-vs-simulated B'_0, and the clock) stripped out. Worth carrying
                 beside C because it is the only one of the three that is clock-free:
                 a Gamma as flat as this one divides out of a time average, so C_avg is
                 identical under clock='shock' and 'fluid' while C is not.
    ratio/C    = what is left over. 1 would mean the correction accounts for the offset
                 outright.
    pct        = where `ratio` sits in the shell's own distribution of per-cell C_i --
                 0% the first-shocked cell, 100% the last. This is the leftover restated:
                 C is C_i for the material the label describes, and a point with
                 ratio/C > 1 is a point whose break is set further along the shell.
  '''
  cz = [r for r in cells if int(r['z']) == z]
  v_end, q_end = _quantiles(cz, 'C_end')
  out = []
  for r in breaks:
    if int(r['z']) != z or r['regime'] not in TRUSTED:
      continue
    if not np.isfinite(r['offset']) or r['offset'] <= 0.:
      continue
    s = np.sqrt(r['offset'])
    nom = np.sqrt(r['nominal'])
    out.append(dict(z=z, logr=r['logr'], regime=r['regime'], offset=r['offset'],
                    gc_gm_meas=np.sqrt(r['ratio']), gc_gm_nom=nom, sqrt=s, C=C,
                    C_avg=C_avg, over_C=(s/C if C else np.nan),
                    over_C_avg=(s/C_avg if C_avg else np.nan),
                    pct=_percentile_of(v_end, s)))
  if verbose:
    print(f'\n--- measured vs labelled gamma_c/gamma_m, z={z} ---')
    print(f'{"logr":>5} {"class":>6} {"sqrt(nu_c/nu_m)":>16} {"gma_c/gma_m":>12} '
          f'{"ratio":>7} {"C":>6} {"ratio/C":>8} {"C_avg":>6} {"ratio/C_avg":>12} '
          f'{"pct of C_i":>11}')
    for r in out:
      print(f'{r["logr"]:+5.1f} {r["regime"]:>6} {r["gc_gm_meas"]:16.4g} '
            f'{r["gc_gm_nom"]:12.4g} {r["sqrt"]:7.2f} {r["C"]:6.2f} '
            f'{r["over_C"]:8.2f} {r["C_avg"]:6.2f} {r["over_C_avg"]:12.2f} '
            f'{r["pct"]:10.0f}%')
    print('  shell C_i(end): ' + '  '.join(f'q{int(100*q):02d} {v:.2f}'
                                           for q, v in q_end.items()))
  return out, q_end


def _write(rows, path, fields):
  with open(path, 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(fields), extrasaction='ignore')
    w.writeheader()
    w.writerows([{k: r.get(k, '') for k in fields} for r in rows])
  return path


def _read(path, fields, strkeys=()):
  if not os.path.isfile(path):
    return None
  out = []
  for r in csv.DictReader(open(path)):
    if set(fields) - set(r):
      return None                     # a cache written by an older field list
    out.append({k: (r[k] if k in strkeys else (float(r[k]) if r[k] not in ('', 'nan')
                                               else np.nan)) for k in fields})
  return out


def plot_correction(cells, breaks, outdir, key=KEY, clock=CLOCK, fname=FIG):
  '''
  Left: the two fields the label replaces with one number -- B'^2 along the first-shocked
  cell's worldline (with its running time average, whose distance from 1 at the crossing IS
  the correction) and B'^2 at the shock FRONT against the shocking time, i.e. the field
  later cells are injected into. Right: the correction cell by cell across the shell (C_i
  is monotone in shocking time, so the curve doubles as the shell's cumulative
  distribution), with a marker where each measured break ratio's factor meets it.
  '''
  env = MyEnv(key)
  fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.5))
  ax = axes[0]
  for z, col, lab in ((Z_RS, COL_RS, 'RS'), (Z_FS, COL_FS, 'FS')):
    tcr = shell_nominal(env, z)[3]
    sh = load_shockfront_states(key, z, env, source=EARLY_ANA)
    k0 = int(shell_klist(key, z, env)[0])
    hist, _, _ = _load_cell_history(key, k0, N_SETTLE, env, sh_data=sh, early_frac=0.)
    q = cooling_integrals(hist, env)
    u = q['t'][1:]/tcr
    ax.plot(u, q['B2'][1:]/q['B2'][0], color=col, lw=1.6,
            label=f"{lab}: $B'^2$ along one worldline")
    ax.plot(u, q['I_B'][1:]/(q['B2'][0]*q['tp'][1:]), color=col, lw=1.6, ls='--',
            label=f"{lab}: $\\langle B'^2\\rangle$ over it")
    # the dotted curve is against SHOCKING time, not time since shocked: it is the field
    # later cells are injected into, and it lands almost on top of the worldline decay
    fr = shock_worldline(key, z, env, clock=clock)
    ax.plot(fr['t'][1:]/tcr, fr['B2'][1:]/fr['B2'][0], color=col, lw=1.2, ls=':',
            label=f"{lab}: $B'^2$ at the front, vs $t_s$")
  ax.axvline(1., color=MUTED, lw=0.9, ls=':')
  ax.text(0.93, 0.02, 'shock crossing', color=MUTED, fontsize=8, rotation=90,
          ha='right', va='bottom', transform=ax.get_xaxis_transform())
  ax.set(xscale='log', yscale='log', xlim=(2e-2, 20.), ylim=(1e-2, 2.),
         xlabel='$t/t_{\\rm cr}$  (since shocked, or shocking time $t_s$ for the dotted)',
         ylabel='comoving field, normalised at injection')
  ax.grid(color=GRID, lw=0.5)
  ax.legend(fontsize=7.5, frameon=False, loc='lower left')
  ax.set_title('the field the sweep label holds constant', fontsize=10, color=INK)

  ax = axes[1]
  # the two shells' markers land on top of each other in slow cooling, so their labels go
  # on opposite sides: RS below its marker, FS above
  for z, col, lab, mk, dy in ((Z_RS, COL_RS, 'RS', 'o', -13), (Z_FS, COL_FS, 'FS', 's', 9)):
    cz = sorted((r for r in cells if int(r['z']) == z), key=lambda r: r['ts_frac'])
    if not cz:
      continue
    ts = np.array([r['ts_frac'] for r in cz])
    C = np.array([r['C_end'] for r in cz])
    ax.plot(ts, C, color=col, lw=1.6, label=f'{lab}: $C_i$')
    o = np.argsort(C)
    for r in sorted((b for b in breaks
                     if int(b['z']) == z and b['regime'] in TRUSTED and b['offset'] > 0.),
                    key=lambda b: b['offset']):
      s = np.sqrt(r['offset'])
      ax.plot(np.interp(s, C[o], ts[o]), s, mk, color=col, ms=6, mec=INK, mew=0.7,
              zorder=5)
      ax.annotate(f'${r["logr"]:+.0f}$ ({r["regime"]})',
                  (np.interp(s, C[o], ts[o]), s), textcoords='offset points',
                  xytext=(0, dy), ha='center', va='top' if dy < 0 else 'bottom',
                  fontsize=7.5, color=col)
  ax.set(yscale='log', xlim=(-0.02, 1.08),
         xlabel='shocking time $t_s/t_{\\rm cr}$ (position along the shell)',
         ylabel='correction to $\\gamma_c/\\gamma_m$')
  ax.grid(color=GRID, lw=0.5)
  ax.legend(fontsize=8, frameon=False, loc='upper left')
  ax.set_title('...and where each fluence spectrum lands on it', fontsize=10, color=INK)
  fig.tight_layout()
  path = os.path.join(outdir, fname)
  fig.savefig(path, dpi=300)
  plt.close(fig)
  return path


def main(key=KEY, method=METHOD, outdir=None, clock=CLOCK, use_cache=True, verbose=True):
  '''
  Measure (or reload) the field average on both shells, read the time-integrated break
  ratios off the cached sweep, and draw the comparison into that sweep's directory.
  '''
  outdir = method_outdir(method, key, Z_RS) if outdir is None else outdir
  os.makedirs(outdir, exist_ok=True)
  cpath = os.path.join(outdir, CELLS_CSV)
  cells = _read(cpath, CELL_FIELDS) if use_cache else None
  if cells:
    print(f'{len(cells)} cell rows reloaded from {cpath}')
  else:
    print(f"--- averaging B'^2 over the shock propagation, {key} ---")
    cells = shell_rows(key, Z_RS) + shell_rows(key, Z_FS)
    _write(cells, cpath, CELL_FIELDS)
    print(f'{len(cells)} cell rows -> {CELLS_CSV}')
  fa = {z: field_average(key, z, clock=clock, verbose=verbose) for z in (Z_RS, Z_FS)}
  breaks = (fluence_break_rows(key, method, Z_RS, verbose=verbose)
            + fluence_break_rows(key, method, Z_FS, verbose=verbose))
  _write(breaks, os.path.join(outdir, BREAKS_CSV), BREAK_FIELDS)
  cmp_rows = []
  for z in (Z_RS, Z_FS):
    cmp_rows += compare(cells, breaks, z=z, C=fa[z]['C'],
                        C_avg=1./fa[z]['front']['mean_B2'], verbose=verbose)[0]
  png = plot_correction(cells, breaks, outdir, key=key, clock=clock)
  trim_pngs([png])
  print(f'-> {png}')
  return fa, cells, breaks, cmp_rows


if __name__ == '__main__':
  main()
