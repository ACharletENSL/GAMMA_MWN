# -*- coding: utf-8 -*-
# @Author: acharlet

'''
The sweep parameter gamma_c/gamma_m is built on a magnetic field the shell only has at
R0. This module measures what that costs, and compares the correction with the break
ratio the time-integrated spectra actually show.

WHAT THE LABEL ASSUMES. phys_functions_shells.shells_add_radNorm writes

    gamma_c = 6 pi m_e c Gamma_0 / (sigma_T B'^2 t_RS),      B' = sqrt(8 pi eps_B e'_3),

with e'_3 the immediate post-shock internal energy of shell 4 AT R0 and t_RS the whole
lab crossing time. Read as physics that is "an electron injected at R0 cools for the
crossing time in the field it was injected into" -- a constant-rate cooling. The rate is
not constant: the cell expands from the moment it is shocked, so its B' decays through
the very interval the formula integrates over. The honest statement of gamma_c is

    1/gamma_c = (sigma_T/6 pi m_e c) int B'^2(t') dt'   over the shock crossing,

so the label is wrong by the ratio of B'^2(R0) to the field's TIME AVERAGE over the
propagation. That ratio is what this module measures.

THE OTHER CONSTANT THE LABEL HOLDS is Gamma. It converts the lab crossing time into the
comoving one as t_cr/Gamma_0 with the analytic Gamma_0, and Gamma is not constant either.
The comoving time a cell has actually had when the shock finishes crossing is
int_0^{t_cr} dt/Gamma(t) along its own worldline (comoving_crossing), and every integral
here stops THERE, not at t_cr/Gamma_0. Measured, this one is small: Gamma runs 130.0 ->
127.8 across the RS crossing, a harmonic mean of 127.36 against Gamma_0 = 126.45, so the
cell has had 0.72% less comoving time than the label credits it with (0.76% on the FS).
It enters the correction as a factor t'_nom/t'_true = 1.0072 (1.0077), i.e. +0.3% on C_B
once the shorter window is also integrated over -- worth taking properly, not worth
quoting. C_B then factorises exactly into the three things the label got wrong:

    C_B = (B'^2_ana/B'^2_0) x (t'_nom/t'_true) x (1/<B'^2>_norm)
    RS:   1/1.0569        x   1.0072          x   2.584          = 2.462
    FS:   1/0.9962        x   1.0077          x   2.800          = 2.832

  -- the analytic-vs-simulated injection field, the Gamma evolution, and the field
  average, in ascending order of how much they matter.

IT IS A RIGID SHIFT OF THE WHOLE SWEEP AXIS. Under the Granot alpha rescaling that
generates the sweep (sweep_gammacm.compute_alpha_sweep) lengths and times go as alpha and
rho, p as alpha^-3, so B'^2 dt' -- and therefore every quantity below -- is alpha^-2,
exactly as gamma_c itself is. The dimensionless decay B'^2(t')/B'^2(0) against t'/t'_RS is
the SAME curve at every sweep point. One number corrects all eight.

THREE VERSIONS OF THE CORRECTION, and which to quote for what.

  C_B  = B'^2_ana t'_nom / int B'^2 dt'
         THE CORRECTION TO gamma_c. Synchrotron losses alone, which is what gamma_c is
         defined by, so this is the number to quote for "the label mis-states gamma_c by".

  Q    = B'^2_ana t'_nom / int B'^2 (rho/rho_0)^(1/3) dt'
         THE CORRECTION TO THE RATIO gamma_c/gamma_m -- the quantity the sweep is labelled
         by and the only one a spectrum can show, since a break ratio is
         nu_c/nu_m = (gamma_c/gamma_m)^2 and nothing else. It is NOT a different physical
         quantity from C_B; it is the same integral with the adiabatic weight restored,
         and it is what you get by writing gamma_c/gamma_m out honestly:

           electrons also lose energy by expanding, so along a worldline
             gamma_c(t) = (rho/rho_0)^(1/3) / (K int B'^2 (rho/rho_0)^(1/3) dt')
                                              ... cooling_frequency's closed form, exact
             gamma_m(t) = gamma_m,0 (rho/rho_0)^(1/3)
                                              ... the same adiabatic factor, no losses
           so the (rho/rho_0)^(1/3) PREFACTOR CANCELS in the ratio and only the weight
           inside the integral survives:
             gamma_c/gamma_m = 1 / (gamma_m,0 K int B'^2 (rho/rho_0)^(1/3) dt').

         So Q is the honest correction and C_B is Q with the adiabatic weight dropped.
         Q > C_B always, because (rho/rho_0)^(1/3) < 1 de-rates exactly the late, expanded
         steps -- an electron that has already expanded is cooling in a weaker field AND
         is worth less to the integral, and the label counts both at their R0 value.
         RS: Q = 2.97 against C_B = 2.46, so the adiabatic weight is a fifth of the
         correction and the field average the other four fifths.

  Q_i  the same Q for every cell of the shell, each with its OWN injection field and its
         own gamma_m,0 -- the shock weakens as it propagates, so the last cell is injected
         into a field ~9x weaker (RS) and its label is wrong by much more. The shell does
         not have A correction factor, it has a distribution, and which part of that
         distribution a spectrum shows is the regime-dependent part of the answer below.

MEASURED, cooling_g100, parent cells, injection states from the shock fit (EARLY_ANA).

                                          RS (z=4)      FS (z=1)
    <B'^2>/B'^2_0 over the crossing         0.3870        0.3571  -> B'_rms = 0.62 B'_0
    1/<B'^2>_norm  (pure averaging)         2.584         2.800
    comoving crossing / t_cr/Gamma_0        0.9928        0.9924  (Gamma harm. 127.4)
    C_B at the crossing                     2.462         2.832
    C_B integrated to the end               1.969         2.467   (the cooling SATURATES)
    Q   at the crossing                     2.974         3.781
    Q   integrated to the end               2.521         3.427
    shell Q(end):  q05 / median / q95    2.7/8.8/48    3.4/9.1/69
    shock-front <B'^2>/B'^2(R0)             0.3655        0.3625
    ... injection field, first -> last    1.057->0.114  0.996->0.129

  So for the cell the label describes, gamma_c/gamma_m is understated by a factor 2.5-3.0
  on the RS -- +0.40 to +0.47 dex -- and 3.4-3.8 on the FS; for the shell's median cell,
  by ~9 on both. In the quantity a spectrum shows, the break RATIO, that is x6.4-8.8 and
  x77-83.
  THE COOLING SATURATES: past t' ~ 2-3 t'_cr the integral stops growing (B'^2 has fallen
  ~50x, the rarefaction has crossed), so "integrated to the end" is a converged number and
  not a window choice. It is the total cooling those electrons will ever suffer, which is
  the right endpoint for a TIME-INTEGRATED spectrum; the crossing is the one that matches
  the label's own definition. The two differ by only ~18%, which is the useful part: the
  answer does not hang on where the average is stopped.

AGAINST THE SPECTRA. sweep_gammacm.compute_fluence_spectrum on each cached point, then
spectral_breaks.smoothing_from_identified (the segment route, segment_route.py) for the two
breaks; their ratio is (gamma_c/gamma_m)^2 for whatever zone the spectrum is effectively
showing, so measured/nominal is the offset the correction has to explain. `nominal` is the
shell's own (gamma_c/gamma_m)^2 -- on the FS that is 10^(2 logr) x 10.14, since the sweep
axis labels the RS. data_rarcut (the article's sweep), the points whose class puts both
breaks in band and measures the mid slope on an identified segment:

    shell  log10(gc/gm)  class   nu_c/nu_m   nominal   offset  sqrt   percentile of Q(end)
     RS         -2        FC     7.43e-04     1e-04      7.4    2.73          5%
     RS         +1        SC     7.51e+03     1e+02     75.1    8.67         50%
     RS         +2        SC     5.05e+05     1e+04     50.5    7.11         42%
     FS         +0        SC     1.74e+03    1.01e+01  171.6   13.10         64%
     FS         +1        SC     8.28e+04    1.01e+03   81.6    9.03         50%
     FS         +2        SC     6.76e+06    1.01e+05   66.7    8.17         45%

  FAST COOLING IS THE FIRST-SHOCKED CELL. sqrt(7.4) = 2.73 against Q(end) = 2.52 for that
  cell and q05 = 2.72 over the shell: the correction accounts for the offset outright, no
  free factor. That is the expected place for it -- in fast cooling nu_c is set by the
  material that has cooled the MOST, i.e. the earliest-shocked cells in the strongest
  field, which are the least-corrected ones.
  SLOW COOLING IS THE MEDIAN CELL, on both shells independently: 42-50% of the RS
  distribution, 45-64% of the FS one. Same physics, different sample -- with gamma_c far
  above gamma_m no cell has cooled much, no population dominates, and the break lands in
  the middle. The FS is a genuine second test, not a re-reading of the first: different
  shock, different crossing time, its own nominal ratio and its own Q distribution.
  So a single scalar CANNOT correct the whole sweep to better than the width of Q_i --
  half a decade in the ratio between the fast- and slow-cooling ends. What the correction
  DOES buy is that the offset stops being unexplained: every measured point lands inside a
  distribution computed from the hydro alone, at the percentile its regime predicts.

  The MC points (RS -1, 0; FS -2, -1) are excluded, not missing: their mid slope is fitted
  free inside the shape fit (no mid segment exists to measure), so their crossings carry
  the spread segment_route.py warns about -- a_mid comes back at 0.67, 1.11, 0.54, 1.24
  against asymptotes of 1/2 and 1/4, and the offsets (1.2, 96, 5.3, 0.18) straddle
  everything. The VFC/FC* points (-5..-3) have no nu^(4/3) segment in band at all, so no
  lower break and no ratio; that is a band limit, not a failure.

CAVEATS.
  The nominal B'(R0) is analytic; the simulation's own first-shocked cell carries B'^2
  1.057x that (RS), 0.996x (FS). C_B and Q are normalised on the ANALYTIC value, because
  what is being corrected is the label -- so 5.7% of the RS factor is normalisation rather
  than averaging, and `b2_0_over_ana` plus the `mean_B2` column separate the two.
  t_RS is likewise the planar analytic crossing time; the spherical run takes longer, so
  33 of 500 RS cells (7 of 500 FS) are shocked after it. They are kept -- Q(end) does not
  use t_RS as an endpoint -- but Q(t_cr) is meaningless for them and comes back NaN.
  One outer-edge cell per shell (k = Next) never gets a clean shocked state (its
  injection field is 1e-4 of nominal, it sits in the shell's own edge rarefaction); it is
  dropped by INJ_FLOOR rather than allowed to set a percentile.
  Parent cells only, as radiative_length: a sub-cell is a slice of its parent's history,
  not an independent emitter, and counting them would weight the early cells by their
  sub-cell count in every quantile here.

Example use:
  python -c "import field_average as F; F.main()"
  python -c "import field_average as F; F.main(use_cache=False)"
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
from sweep_gammacm import (DEFAULT_KEY, EARLY_ANA, LOG10RATIO_ARR, method_outdir,
    trim_pngs)
from plotting_functions import COL_RS, COL_FS
from mid_slope_evolution import INK, MUTED, GRID

KEY = DEFAULT_KEY
METHOD = 'data_rarcut'        # the article's sweep (sweep_gammacm.ARTICLE_SERIES); the
                              # correction itself is method-independent (it is hydro), only
                              # the measured break ratios are read from a method
Z_RS, Z_FS = 4, 1
N_SETTLE = 1                  # as the emission pipeline: the injection row is the shock-fit
                              # state prepended by _load_cell_history, so the correction and
                              # the flux see the SAME first step
INJ_FLOOR = 0.1               # drop a cell whose injection gamma_m is below this fraction of
                              # the shell's nominal one: not a weak shock but no shock at all
                              # (the outer-edge cell, see CAVEATS)
T_FRACS = (0.25, 0.5, 1., 1.5, 2., 3., 5., 10.)   # t'/t'_cr sampling of the running average
QS = (0.05, 0.25, 0.5, 0.75, 0.95)                # quantiles reported over the shell
CELLS_CSV = 'field_average_cells.csv'
BREAKS_CSV = 'field_average_breaks.csv'
FIG = 'field_average.png'
CELL_FIELDS = ('z', 'k', 'ts_frac', 'b2_0', 'gm_0', 'C_B_cr', 'C_B_end', 'Q_cr', 'Q_end')
BREAK_FIELDS = ('z', 'logr', 'regime', 'b_lo', 'b_hi', 'a_mid', 'ratio', 'nominal',
                'offset')
# shape classes whose mid slope is MEASURED on an identified segment, hence whose two
# crossings can be read as nu_c and nu_m. MC's mid is fitted free (segment_route.py), VFC
# and FC* have no lower break in band.
TRUSTED = ('FC', 'SC')


# ---------------------------------------------------------------------------
# the hydro side: <B'^2> along a cell's worldline

def _cumtrapz0(y, x):
  '''Cumulative trapezoid with out[0] = 0 (working_cooling_data._cumtrapz0's convention).'''
  return np.concatenate(([0.], np.cumsum(0.5*(y[1:] + y[:-1])*np.diff(x))))


def cooling_integrals(hist, env):
  '''
  The two cooling integrals along one cell history, in cgs.

    tp     comoving time since the cell was shocked, int dt/Gamma over the snapshots
    I_B    int B'^2 dt'                          -- the field average alone
    I_ad   int B'^2 (rho/rho_0)^(1/3) dt'        -- with the adiabatic weight the electron
                                                   split carries (see the header)

  B'^2 = 8 pi eps_B e'_int is the same field derive_syn_cooling builds the cooling rate
  from, so 1/(alpha_ I_B) IS the gamma_c the pipeline would reach with no adiabatic term
  and rho^(1/3)/(alpha_ I_ad) is cooling_frequency.cell_cooling_lfac's closed form.
  '''
  t = hist.t.to_numpy(dtype=float)
  rho = hist.rho.to_numpy(dtype=float)
  p = hist.p.to_numpy(dtype=float)
  lfac = derive_Lorentz(hist.vx.to_numpy(dtype=float))
  B2 = 8.*pi_*env.eps_B*derive_Eint_comoving(rho, p, env.rhoscale)
  tp = np.concatenate(([0.], np.cumsum(np.diff(t)/(0.5*(lfac[1:] + lfac[:-1])))))
  r13 = (rho/rho[0])**(1./3.)
  return dict(t=t, tp=tp, B2=B2, r13=r13, x=hist.x.to_numpy(dtype=float),
              I_B=_cumtrapz0(B2, tp), I_ad=_cumtrapz0(B2*r13, tp))


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


def comoving_crossing(q, tcr):
  '''
  How much COMOVING time a cell has actually had when the shock finishes crossing, i.e.
  int_0^{t_cr} dt/Gamma(t) along its own worldline -- the endpoint every integral here
  stops at.

  The label uses t_cr/Gamma_0 with the analytic Gamma_0 held constant, and Gamma is not
  constant: on the first-shocked RS cell it runs 130.0 -> 127.8 across the crossing, a
  harmonic mean of 127.36 against Gamma_0 = 126.45, so the cell has had 0.72% LESS
  comoving time than the label credits it with (FS: 0.76% less). Sub-percent, and in the
  direction of slightly less cooling, hence a slightly larger correction -- but it is the
  difference between stopping the average at the shock crossing and stopping it at a
  proxy for it, so it is taken properly rather than assumed small.

  Returns NaN for a cell shocked after t_cr (np.interp would silently clamp to tp[0]).
  '''
  return np.interp(tcr, q['t'], q['tp']) if q['t'][0] < tcr else np.nan


def shell_rows(key=KEY, z=Z_RS, env=None, verbose=True):
  '''
  Per-cell correction factors over a whole shell.

  Each row carries the cell's own injection state (b2_0, gm_0, both normalised on the
  shell's NOMINAL values) and its four correction factors: C_B and Q, each evaluated at
  the end of the crossing and at the end of the cell's history. All four are normalised
  on the nominal 1/(alpha_ B'^2_ana t'_cr), so a row reads directly as "this cell's
  gamma_c is X times the sweep label's".

  Q folds in gamma_m,0: it is the correction to the RATIO gamma_c/gamma_m, and a later
  cell is injected with a smaller gamma_m as well as a weaker field.
  '''
  env = MyEnv(key) if env is None else env
  B2ana, gm_nom, _, tcr = shell_nominal(env, z)
  tpcr = tcr/env.lfac0
  norm = B2ana*tpcr                  # 1/(alpha_ * norm) is the nominal gamma_c
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
    t, tp = q['t'], q['tp']
    tend = comoving_crossing(q, tcr)
    crossed = np.isfinite(tend)
    ib_c = np.interp(tend, tp, q['I_B']) if crossed else np.nan
    ia_c = np.interp(tend, tp, q['I_ad']) if crossed else np.nan
    rows.append(dict(z=z, k=int(k), ts_frac=t[0]/tcr, b2_0=q['B2'][0]/B2ana,
                     gm_0=gm0/gm_nom,
                     C_B_cr=norm/ib_c, C_B_end=norm/q['I_B'][-1],
                     Q_cr=(gm_nom/gm0)*norm/ia_c,
                     Q_end=(gm_nom/gm0)*norm/q['I_ad'][-1]))
  if verbose:
    print(f'  z={z}: {len(rows)} cells ({skipped} skipped), '
          f'{int(np.sum([r["ts_frac"] < 1. for r in rows]))} shocked before t_cr')
  return rows


def field_average(key=KEY, z=Z_RS, env=None, fracs=T_FRACS, verbose=True):
  '''
  The headline correction, from the FIRST-shocked cell: the running time-average of B'^2
  over its worldline, and the factors that follow from it.

  The first-shocked cell is what the label describes -- injected at R0, cooling for the
  whole crossing -- so this is the correction to the DEFINITION, with no shell statistics
  in it. shell_rows carries the rest of the shell.

  `fracs` samples the running average in units of the cell's TRUE comoving crossing time
  (comoving_crossing), so frac = 1 is the shock crossing itself and not the label's
  t_cr/Gamma_0 proxy for it. C_B and Q keep the label's t_cr/Gamma_0 in their NUMERATOR,
  because what they correct is the label; the Gamma evolution therefore shows up inside
  them, as the t'_nom/t'_true factor the module header decomposes them into.
  '''
  env = MyEnv(key) if env is None else env
  B2ana, gm_nom, gc_nom, tcr = shell_nominal(env, z)
  tpcr = tcr/env.lfac0
  norm = B2ana*tpcr
  sh = load_shockfront_states(key, z, env, source=EARLY_ANA)
  k0 = int(shell_klist(key, z, env)[0])
  hist, _, _ = _load_cell_history(key, k0, N_SETTLE, env, sh_data=sh, early_frac=0.)
  q = cooling_integrals(hist, env)
  gm0 = float(get_variable(hist.iloc[0], 'gma_m', env))
  # the endpoint is the shock crossing, in LAB time; the comoving time the cell has had by
  # then is int dt/Gamma along its worldline, NOT t_cr/Gamma_0 (comoving_crossing)
  tp_cr = comoving_crossing(q, tcr)
  out = dict(z=z, k0=k0, b2_0_over_ana=q['B2'][0]/B2ana, gm_0_over_nom=gm0/gm_nom,
             tpcr_nom=tpcr, tp_cr=tp_cr, tp_cr_over_nom=tp_cr/tpcr,
             lfac_harm=tcr/tp_cr, C_B_end=norm/q['I_B'][-1],
             Q_end=(gm_nom/gm0)*norm/q['I_ad'][-1], running=[])
  for f in fracs:
    tt = f*tp_cr
    if tt > q['tp'][-1]:
      continue
    ib = np.interp(tt, q['tp'], q['I_B'])
    ia = np.interp(tt, q['tp'], q['I_ad'])
    out['running'].append(dict(frac=f, R_over_R0=np.interp(tt, q['tp'], q['x'])*c_/env.R0,
                               mean_B2=ib/(q['B2'][0]*tt), C_B=norm/ib,
                               Q=(gm_nom/gm0)*norm/ia))
  one = [r for r in out['running'] if r['frac'] == 1.]
  out['mean_B2_cr'] = one[0]['mean_B2'] if one else np.nan
  out['C_B_cr'] = one[0]['C_B'] if one else np.nan
  out['Q_cr'] = one[0]['Q'] if one else np.nan
  if verbose:
    print(f'--- z={z}, first-shocked cell k={k0} '
          f'(B\'^2 = {out["b2_0_over_ana"]:.4f} x analytic) ---')
    print(f"  comoving crossing time: int dt/Gamma = {tp_cr:.5g} s against the label's "
          f"t_cr/Gamma_0 = {tpcr:.5g} s ({out['tp_cr_over_nom']:.4f}x; Gamma runs "
          f"{out['lfac_harm']:.2f} harmonic vs Gamma_0 = {env.lfac0:.2f})")
    print(f'{"t/t_cr":>7} {"R/R0":>8} {"<B^2>/B0^2":>11} {"C_B":>8} {"Q":>8}')
    for r in out['running']:
      print(f'{r["frac"]:7.2f} {r["R_over_R0"]:8.3f} {r["mean_B2"]:11.4f} '
            f'{r["C_B"]:8.3f} {r["Q"]:8.3f}')
    print(f'  integrated to the end of the history: C_B = {out["C_B_end"]:.3f}, '
          f'Q = {out["Q_end"]:.3f}   (the cooling saturates)')
    print(f'  C_B at the crossing decomposes as '
          f'1/{out["b2_0_over_ana"]:.4f} (analytic vs simulated B\'_0) x '
          f'{1./out["tp_cr_over_nom"]:.4f} (Gamma evolution) x '
          f'{1./out["mean_B2_cr"]:.3f} (field average) = {out["C_B_cr"]:.3f}')
    print(f'  nominal gamma_c = {gc_nom:.4g} -> corrected {gc_nom*out["C_B_cr"]:.4g} '
          f'(synchrotron only, stopped at the crossing); gamma_c/gamma_m x '
          f'{out["Q_cr"]:.3f} -> break ratio x {out["Q_cr"]**2:.2f}')
  return out


def front_average(key=KEY, z=Z_RS, env=None):
  '''
  The OTHER average the phrase can mean: B'^2 immediately behind the shock, averaged over
  the propagation instead of along one cell. It measures how much weaker the field a later
  cell is INJECTED into is, which is a different statement from how a given cell's field
  decays -- both are in the label, and shell_rows' Q_i carries them together.
  '''
  env = MyEnv(key) if env is None else env
  B2ana, _, _, tcr = shell_nominal(env, z)
  sh = load_shockfront_states(key, z, env, source=EARLY_ANA)
  sel = sh.loc[sh.t < tcr].sort_values('t')
  t = sel.t.to_numpy(dtype=float)
  B2 = 8.*pi_*env.eps_B*derive_Eint_comoving(sel.rho.to_numpy(dtype=float),
                                             sel.p.to_numpy(dtype=float), env.rhoscale)
  return dict(z=z, mean=np.trapezoid(B2, t)/((t[-1] - t[0])*B2ana),
              first=B2[0]/B2ana, last=B2[-1]/B2ana)


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
  factor the break ratio needs on gamma_c/gamma_m, and Q_i is the factor the hydro
  supplies. `pct` is where the first lands in the second -- 0% would be the first-shocked
  cell, 100% the last.
  '''
  cz = [r for r in cells if int(r['z']) == z]
  v_end, q_end = _quantiles(cz, 'Q_end')
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
    print('  shell Q(end): ' + '  '.join(f'q{int(100*q):02d} {v:.2f}'
                                         for q, v in q_end.items()))
    print(f'{"logr":>5} {"class":>6} {"offset":>9} {"sqrt":>7} {"percentile of Q(end)":>22}')
    for r in out:
      print(f'{r["logr"]:+5.1f} {r["regime"]:>6} {r["offset"]:9.2f} {r["sqrt"]:7.2f} '
            f'{r["pct"]:21.0f}%')
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
    out.append({k: (r[k] if k in strkeys else (float(r[k]) if r[k] not in ('', 'nan')
                                               else np.nan)) for k in fields})
  return out


def plot_correction(cells, breaks, outdir, key=KEY, fname=FIG):
  '''
  Left: the field a cell cools in, against its own worldline -- B'^2(t')/B'^2(0) and its
  running time average, both shells. The gap between the two at t' = t'_cr IS the
  correction, and the average FLATTENING past ~2 t'_cr is the cooling saturating.
  Right: the same correction cell by cell across the shell (Q_i is monotone in shocking
  time, so the curve doubles as the shell's cumulative distribution), with a marker where
  each measured break ratio's factor meets it -- i.e. which part of the shell that
  spectrum is effectively showing.
  '''
  env = MyEnv(key)
  fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.5))
  ax = axes[0]
  for z, col, lab in ((Z_RS, COL_RS, 'RS'), (Z_FS, COL_FS, 'FS')):
    sh = load_shockfront_states(key, z, env, source=EARLY_ANA)
    k0 = int(shell_klist(key, z, env)[0])
    hist, _, _ = _load_cell_history(key, k0, N_SETTLE, env, sh_data=sh, early_frac=0.)
    q = cooling_integrals(hist, env)
    u = q['tp'][1:]/comoving_crossing(q, shell_nominal(env, z)[3])
    ax.plot(u, q['B2'][1:]/q['B2'][0], color=col, lw=1.6, label=f"{lab}: $B'^2/B'^2_0$")
    ax.plot(u, q['I_B'][1:]/(q['B2'][0]*q['tp'][1:]), color=col, lw=1.6, ls='--',
            label=f"{lab}: $\\langle B'^2\\rangle/B'^2_0$")
  ax.axvline(1., color=MUTED, lw=0.9, ls=':')
  ax.text(0.93, 0.02, 'shock crossing', color=MUTED, fontsize=8, rotation=90,
          ha='right', va='bottom', transform=ax.get_xaxis_transform())
  ax.set(xscale='log', yscale='log', xlim=(2e-2, 20.), ylim=(1e-2, 2.),
         xlabel="$t'/t'_{\\rm cr}$ since the cell is shocked",
         ylabel='comoving field, normalised at injection')
  ax.grid(color=GRID, lw=0.5)
  ax.legend(fontsize=8, frameon=False, loc='lower left')
  ax.set_title('the field the sweep label holds constant', fontsize=10, color=INK)

  ax = axes[1]
  # the two shells' markers land on top of each other in slow cooling, so their labels go
  # on opposite sides: RS below its marker, FS above
  for z, col, lab, mk, dy in ((Z_RS, COL_RS, 'RS', 'o', -13), (Z_FS, COL_FS, 'FS', 's', 9)):
    cz = sorted((r for r in cells if int(r['z']) == z), key=lambda r: r['ts_frac'])
    if not cz:
      continue
    ts = np.array([r['ts_frac'] for r in cz])
    Q = np.array([r['Q_end'] for r in cz])
    ax.plot(ts, Q, color=col, lw=1.6, label=f'{lab}: $Q_i$')
    o = np.argsort(Q)
    for r in sorted((b for b in breaks
                     if int(b['z']) == z and b['regime'] in TRUSTED and b['offset'] > 0.),
                    key=lambda b: b['offset']):
      s = np.sqrt(r['offset'])
      ax.plot(np.interp(s, Q[o], ts[o]), s, mk, color=col, ms=6, mec=INK, mew=0.7,
              zorder=5)
      ax.annotate(f'${r["logr"]:+.0f}$ ({r["regime"]})',
                  (np.interp(s, Q[o], ts[o]), s), textcoords='offset points',
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


def main(key=KEY, method=METHOD, outdir=None, use_cache=True, verbose=True):
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
    print(f'--- averaging B\'^2 over the shock propagation, {key} ---')
    cells = shell_rows(key, Z_RS) + shell_rows(key, Z_FS)
    _write(cells, cpath, CELL_FIELDS)
    print(f'{len(cells)} cell rows -> {CELLS_CSV}')
  fa = {}
  for z in (Z_RS, Z_FS):
    fa[z] = field_average(key, z, verbose=verbose)
    fr = front_average(key, z)
    fa[z]['front'] = fr
    print(f"  shock-front average: <B'^2>/B'^2(R0) = {fr['mean']:.4f} "
          f'(the injection field itself falls {fr["first"]:.3f} -> {fr["last"]:.3f} '
          'x nominal across the crossing)')
  breaks = (fluence_break_rows(key, method, Z_RS, verbose=verbose)
            + fluence_break_rows(key, method, Z_FS, verbose=verbose))
  _write(breaks, os.path.join(outdir, BREAKS_CSV), BREAK_FIELDS)
  cmp_rows = []
  for z in (Z_RS, Z_FS):
    cmp_rows += compare(cells, breaks, z=z, verbose=verbose)[0]
  png = plot_correction(cells, breaks, outdir, key=key)
  trim_pngs([png])
  print(f'-> {png}')
  return fa, cells, breaks, cmp_rows


if __name__ == '__main__':
  main()
