# -*- coding: utf-8 -*-
# @Author: acharlet

'''
The sweep parameter gamma_c/gamma_m is built on a magnetic field the shell only has at
R0, held for a comoving crossing time built on a Lorentz factor it only has at R0 either.
This module measures what that costs, and compares the correction with the break ratio
the time-integrated spectra actually show.

WHAT THE LABEL ASSUMES. phys_functions_shells.shells_add_radNorm writes

    gamma_c = 6 pi m_e c Gamma_0 / (sigma_T B'^2 t_RS),      B' = sqrt(8 pi eps_B e'_3),

with e'_3 the immediate post-shock internal energy of shell 4 AT R0, t_RS the lab crossing
time and Gamma_0 the analytic Lorentz factor at R0. Read as physics that is "an electron
injected at R0 cools for the crossing time, at the rate it had when it was injected" -- a
constant-rate cooling. Two things in it are not constant, and the honest statement is

    1/gamma_c = (sigma_T/6 pi m_e c) int B'^2 dt'      over the shock propagation,

    dt' = dt/Gamma(t),      both B' and Gamma taken ALONG THE SHOCK'S WORLDLINE.

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
as gamma_c itself is (checked numerically on B'^2 t_RS/Gamma_0). The dimensionless decay
B'^2/B'^2(R0) against t/t_cr is the SAME curve at every sweep point. One number corrects
all eight.

WHICH CLOCK, AND WHAT GAMMA ACTUALLY DOES. t'_cr is measured on the shock's own worldline
as the invariant interval sum, sqrt(dt^2 - dx^2) over the front's trajectory -- not on the
worldline of any one cell. Three clocks are measured and all three are reported, because
the answer is not what one would guess:

                                    RS (z=4)              FS (z=1)
    Gamma_0 (the label)              126.45                126.45
    shock front's own trajectory     112.57 -> 111.95      136.16 -> 135.39
    fluid just behind the front      130.00 -> 129.28      126.08 -> 125.36
    first cell, along its worldline  130.00 -> 127.76      126.08 -> 127.82
    t'_cr / (t_cr/Gamma_0), shock     1.1249                0.9303
    ... same, fluid clock             0.9741                1.0047

  GAMMA BARELY EVOLVES ANYWHERE. Over the crossing every clock moves by 0.55%, and along a
  cell's whole worldline -- rarefaction included -- the fluid Gamma only wanders 126-133.5
  (max 133.5, reached at t ~ 500-700 t_cr). There is no substantial increase to capture,
  on the shock's worldline or any other. What IS substantial is the shock trajectory's
  OFFSET from Gamma_0: -11% on the RS, +7.8% on the FS, because Gamma_0 is the shocked
  FLUID's Lorentz factor and the shock surface does not move with the fluid. That offset
  is worth 12% on the RS correction and 7% on the FS one, in opposite directions -- an
  order of magnitude more than the evolution, and the reason to take the clock from the
  shock rather than from Gamma_0.
  The clock changes only the TOTAL t'_cr, never the shape of the average: <B'^2>/B'^2(R0)
  comes out at 0.3455 (RS) and 0.3635 (FS) under both, to four decimals, because a Gamma
  that flat divides out of a time average.
  Caveat worth stating once: the cooling equation runs on the FLUID's proper time, since
  B' is the comoving field, so the fluid clock is the physically consistent one and the
  shock clock is a property of the surface, not of anything that cools. Both are carried
  (`clock='shock'|'fluid'`); the correction is

    RS   2.434 (shock)   2.811 (fluid)        FS   2.968 (shock)   2.748 (fluid)

  i.e. the choice is worth -13% on the RS and +8% on the FS, and it is the largest single
  uncertainty in the number. `clock='shock'` is the default because it is what was asked
  for; nothing downstream assumes it.

MEASURED, cooling_g100, injection states from the shock fit (EARLY_ANA), clock='shock'.

                                          RS (z=4)      FS (z=1)
    <B'^2>/B'^2(R0) over the propagation    0.3455        0.3635  -> B'_rms = 0.6 B'(R0)
    1/<B'^2>_norm  (pure averaging)         2.894         2.751
    t'_nom/t'_cr   (the clock)              0.8890        1.0750
    B'^2_ana/B'^2(R0) (analytic vs sim)     0.9461        1.0038
    C = their product                       2.434         2.968
    injection field, first -> last cell   1.057->0.114  0.996->0.129
    C_i along a cell's worldline: first / q05 / median / q95
                                      1.97/2.24/5.70/23.0   2.47/2.78/7.36/49.9

  So gamma_c/gamma_m is understated by 2.4 on the RS -- +0.39 dex -- and 3.0 on the FS,
  for the material the label describes. In the quantity a spectrum shows, the break RATIO,
  that is x5.9 and x8.8. The field average is doing nearly all of the work (x2.75-2.89);
  the clock and the analytic-vs-simulated field are both ~10% corrections on top of it,
  and on the RS they happen to cancel.
  THE COOLING SATURATES past t ~ 2-3 t_cr (B'^2 has fallen ~50x, the rarefaction has
  crossed), so the per-cell C_i integrated to the END of each worldline is a converged
  number and not a window choice -- it is the total cooling those electrons will ever
  suffer, which is the endpoint a TIME-INTEGRATED spectrum sees. The first cell's
  C_i(end) = 1.97 against C_i(crossing) = 2.46: stopping at the crossing or following the
  cooling to exhaustion differ by 25%, which bounds how much the endpoint can matter.

AGAINST THE SPECTRA. sweep_gammacm.compute_fluence_spectrum on each cached point, then
spectral_breaks.smoothing_from_identified (the segment route, segment_route.py) for the two
breaks; their ratio is (gamma_c/gamma_m)^2 for whatever material the spectrum is
effectively showing, so measured/nominal is the offset the correction has to explain.
`nominal` is the shell's own (gamma_c/gamma_m)^2 -- on the FS that is 10^(2 logr) x 10.14,
since the sweep axis labels the RS. data_rarcut (the article's sweep), the points whose
class puts both breaks in band and measures the mid slope on an identified segment:

    shell  log10(gc/gm)  class   nu_c/nu_m   nominal   offset  sqrt   percentile of C_i
     RS         -2        FC     7.43e-04     1e-04      7.4    2.73         14%
     RS         +1        SC     7.51e+03     1e+02     75.1    8.67         68%
     RS         +2        SC     5.05e+05     1e+04     50.5    7.11         60%
     FS         +0        SC     1.74e+03    1.01e+01  171.6   13.10         71%
     FS         +1        SC     8.28e+04    1.01e+03   81.6    9.03         58%
     FS         +2        SC     6.76e+06    1.01e+05   66.7    8.17         54%

  FAST COOLING SITS LOW IN THE SHELL'S DISTRIBUTION. sqrt(7.4) = 2.73 against 1.97 for the
  first-shocked cell and q05 = 2.24: the 14th percentile, i.e. the earliest-shocked cells,
  in the strongest field, which are the least-corrected ones. That is where it belongs --
  in fast cooling nu_c is set by the material that has cooled the MOST.
  SLOW COOLING SITS AT OR ABOVE THE MEDIAN, on both shells independently (54-71%). Same
  physics, different sample: with gamma_c far above gamma_m no cell has cooled much, no
  population dominates, and the break lands in the body of the distribution.
  So a single scalar CANNOT correct the whole sweep to better than the width of C_i --
  about half a decade in the ratio between the fast- and slow-cooling ends. What the
  correction DOES buy is that the offset stops being unexplained: every measured point
  lands inside a distribution computed from the hydro alone, at the percentile its regime
  predicts.

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

Example use:
  python -c "import field_average as F; F.main()"
  python -c "import field_average as F; F.main(use_cache=False)"
  python -c "import field_average as F; F.field_average(z=4, clock='fluid')"
'''

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from environment import MyEnv
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
  '''(B'^2, gamma_m, gamma_c, t_cross) of shell z as the sweep label defines them.'''
  if z == 4:
    return env.Bp**2, env.gma_m, env.gma_c, env.tRS
  return env.BpFS**2, env.gma_mFS, env.gma_cFS, env.tFS


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


def shock_worldline(key=KEY, z=Z_RS, env=None, clock=CLOCK):
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
  B2ana, _, gc_nom, tcr = shell_nominal(env, z)
  sh = load_shockfront_states(key, z, env, source=EARLY_ANA).sort_values('t')
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
  tp_nom = tcr/env.lfac0                               # the label's own comoving crossing
  return dict(z=z, clock=clock, t=t, x=x, B2=B2, lfac_fluid=lf_fluid, lfac_shock=lf_shock,
              tpcr=tpcr, tp_nom=tp_nom, I_B=I_B,
              mean_B2=I_B/(tpcr*B2[0]),                # <B'^2>/B'^2(R0) over the crossing
              b2_ana_over_sim=B2ana/B2[0],
              clock_fac=tp_nom/tpcr,
              C=B2ana*tp_nom/I_B,                      # the correction to the label
              gc_nom=gc_nom, b2_first=B2[0]/B2ana, b2_last=B2[-1]/B2ana)


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
  B2ana, _, gc_nom, tcr = shell_nominal(env, z)
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
    print(f'  Gamma_0 = {env.lfac0:.2f} | shock trajectory {lfs[0]:.2f} -> {lfs[-1]:.2f}'
          f' | fluid at the front {lff[0]:.2f} -> {lff[-1]:.2f}')
    print(f"  t'_cr = {fr['tpcr']:.5g} s against the label's t_cr/Gamma_0 = "
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
  B2ana, gm_nom, _, tcr = shell_nominal(env, z)
  norm = B2ana*tcr/env.lfac0         # 1/(alpha_ * norm) is the nominal gamma_c
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
  since the sweep axis labels the RS and the FS sits 0.503 dex above it.
  '''
  res = sorted(swp.load_sweep(method_outdir(method, key, z)),
               key=lambda r: r['log10ratio'])
  if not res:
    raise FileNotFoundError(f'no cached sweep for {method} z={z}: run sweep_gammacm.main')
  rows = []
  for r in res:
    x = swp.nu_over_num(r)
    sp = swp.compute_fluence_spectrum(r['Tb'], r['nuFnu'])
    f = sb.smoothing_from_identified(x, sp, r['env'].psyn)
    logr = float(r['log10ratio'])
    env_r = r['env']
    gc, gm = ((env_r.gma_c, env_r.gma_m) if z == 4 else
              (getattr(env_r, 'gma_cFS', np.nan), getattr(env_r, 'gma_mFS', np.nan)))
    nominal = (gc/gm)**2
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


def compare(cells, breaks, z=Z_RS, verbose=True):
  '''
  Each trusted spectral point against its own shell's distribution: sqrt(offset) is the
  factor the break ratio needs on gamma_c/gamma_m, and C_i is the factor the hydro
  supplies. `pct` is where the first lands in the second -- 0% would be the first-shocked
  cell, 100% the last.
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
    out.append(dict(z=z, logr=r['logr'], regime=r['regime'], offset=r['offset'], sqrt=s,
                    pct=_percentile_of(v_end, s)))
  if verbose:
    print(f'\n--- what the correction has to explain, z={z} ---')
    print('  shell C_i(end): ' + '  '.join(f'q{int(100*q):02d} {v:.2f}'
                                           for q, v in q_end.items()))
    print(f'{"logr":>5} {"class":>6} {"offset":>9} {"sqrt":>7} {"percentile of C_i":>19}')
    for r in out:
      print(f'{r["logr"]:+5.1f} {r["regime"]:>6} {r["offset"]:9.2f} {r["sqrt"]:7.2f} '
            f'{r["pct"]:18.0f}%')
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
    cmp_rows += compare(cells, breaks, z=z, verbose=verbose)[0]
  png = plot_correction(cells, breaks, outdir, key=key, clock=clock)
  trim_pngs([png])
  print(f'-> {png}')
  return fa, cells, breaks, cmp_rows


if __name__ == '__main__':
  main()
