# -*- coding: utf-8 -*-
# @Author: acharlet

'''
How does a shocked cell evolve BEFORE the rarefaction reaches it?

The post-handover half of that worldline is already a physical law with no free parameter
(working_cooling_data.norar_history: mass conservation + the Taub-Matthews adiabat +
Bernoulli). The pre-handover half is not: production describes it either with three
smooth-BPL curve_fits per cell (working_cooling.fit_celldata, constrained over 0.03-0.50
decades and the dominant end-to-end cost) or with the raw snapshot rows. Neither is a model.

This module measures the one quantity a model would need. With the cell's proper width
Delta' = Gamma*dr and

    alpha_D  = dln(Delta')/dlnR,        dlnrho/dlnR = -2 - alpha_D,
    alpha_G  = dln(Gamma)/dlnR,         p  from the TM adiabat p(rho, s_sh),

a prescription is complete once alpha_D is known, so the question is whether alpha_D obeys
a simple law. It does. See RESULTS.

TWO THINGS THIS MODULE DELIBERATELY DOES NOT CLAIM

  dlnrho/dlnR = -2 - alpha_D is an IDENTITY here, not a result. rho*Gamma*R^2*dr = const is
  the code's own invariant (working_cooling.reconstruct_cell derives dr FROM rho with it),
  and analysis_hydro.smooth_dump_density rescales dr specifically to preserve it. It is
  checked (mass_conservation_audit) as an audit on the data -- it localises remap/restart
  damage -- and never reported as evidence for the model.

  The Gamma_sh dependence is untestable on existing data. RS vs FS gives a 4.4x lever on
  Theta_sh for free, but Gamma_sh is 126-130 on both shells and in every other extracted
  run: the sweep_log_au=* runs kept no snapshot dumps, fid_au=2_u1=100 has hydro identical
  to cooling_g100 (it differs only in eps_B), and the gammacm sweep points are unit
  rescalings, under which Theta = p/rho and Gamma are invariant by construction. Nothing
  below constrains f(Gamma_sh); that needs new runs.

RESULTS (cooling_g100, both shells, 40 cells each, slope half-width 0.01 dex)

  alpha_D IS NOT A PLATEAU. IT IS A CONVERGENCE. At a fixed small R/R_inj the cells are
  spread over ~0.28 in alpha_D and ordered monotonically by k -- shallow at the contact
  discontinuity, steep at the outer edge -- and that spread NARROWS to ~0.08 by 0.35 dex as
  they converge onto a common value. Reading the early spread as scatter, or the late
  agreement as a plateau, each misses half of it.

  THE ASYMPTOTE, AND A CORRECTION TO IT FROM THE WIDER-SHELL RUNS (2026-08-18).

  asymptote() on the FIDUCIAL returns -0.7614 +- 0.0198 (RS) / -0.7576 +- 0.0101 (FS),
  agreeing to 0.0038 across a 3.62x lever in Theta_sh. THAT NUMBER IS BIASED SHALLOW BY
  ~0.03 AND SHOULD NOT BE QUOTED AS THE CONVERGED VALUE. The fiducial's longest window is
  0.50 dex, and alpha_D does not stop moving until ~0.60. Measured on cooling_g100_w2
  (W=2, windows to 0.734 dex), median over 0.55-0.65 dex on the deepest cells:

      converged alpha_D    RS  -0.797 +- 0.006    FS  -0.793 +- 0.004    (W=5, 25/21 cells)
      => dlnrho/dlnR           -1.203               -1.207
      => p ~ R^-1.97 (RS) / R^-2.00 (FS)   via the TM adiabat

  measured with converged_alpha() on cooling_g100_w5 over R_i/R_0 in [1.00, 2.46] (RS) and
  [1.00, 2.38] (FS), i.e. covering the whole fiducial R_i range. USE converged_alpha(), NOT
  asymptote(): on the same runs the latter returns -0.760 (fiducial), -0.808 (W=2), -0.840
  (W=5) for what is one and the same quantity, because its fixed [0.25,0.35] window averages
  in the large-R_i cells that are still mid-transient there, and it drifts steeper the wider
  the shell. It now warns when a run's windows extend past its window.

  AND THE CONVERGED VALUE IS ESSENTIALLY A CONSTANT -- no R_i axis needed for it. Across
  R_i/R_0 = 1.0 -> 2.5 it spans only 0.031 (RS) / 0.016 (FS):
      RS  [1.0,1.5) -0.788   [1.5,2.0) -0.798   [2.0,3.0) -0.800
      FS  [1.0,1.5) -0.790   [1.5,2.0) -0.795   [2.0,3.0) -0.792   (corr with R_i = -0.10)
  and the two shells agree to 0.004. So the strong R_i dependence seen at [0.25,0.35] dex
  (corr -0.945, span 0.12) is NOT a property of the converged state: at a fixed 0.30 dex,
  cells of different R_i are simply at different STAGES of the same relaxation. The R_i axis
  of the table is needed for the TRANSIENT; the asymptote is one number, -0.795 +- 0.005.

  The p slope is unchanged at ~R^-2 and still sits inside the BPL fits' independently
  measured p ~ R^-2.0..-2.4 (the retired free-coasting laws), which remains the one
  external check available.

  WHY THE FIDUCIAL GOT IT WRONG, and why the pooled curve hides it. Per cell, alpha_D
  steepens MONOTONICALLY through the window -- on the deepest W=2 cells, -0.660 at 0.10 dex
  -> -0.761 at 0.40 -> -0.787 at 0.60, then flat (drift 0.0000 over the last 0.08 dex). The
  fiducial simply stops at 0.50, before the flattening. Worse, the POOLED median over all
  cells runs the OTHER WAY on the RS (-0.899 at 0.10 dex -> -0.782 at 0.65) because the cell
  population changes with window position -- cells at larger R_i have steeper alpha_D and
  drop out first. Pooled trend and per-cell trend have OPPOSITE SIGNS, so any statement about
  "convergence" must be made on a FIXED cell set. asymptote()'s selected-subsample caveat was
  the right worry; this is its size.
  Exclude the last ~0.03 dex of each window before reading any of this: within one slope
  window of the handover the estimator runs off the fitted range and returns -0.17..-0.63.

  ALPHA_D DEPENDS ON R_i DURING THE TRANSIENT, which is what the table's R_i axis is for. Over
  [0.25, 0.35] dex on the W=2 RS it runs -0.835..-0.715 with corr(alpha_D, R_i) = -0.945. That
  is a difference in RELAXATION STAGE, not in endpoint (see the converged numbers above), but
  it is real and it is where most of a cell's radial range sits, so the 2-D table in
  (R_i/R_0, R/R_i) remains the right object -- and table_closure shows it works.

VALIDATION AGAINST WIDER AND COARSER RUNS (compare_runs, window_law; 2026-08-18)

  Two W=2 runs were made for this: cooling_g100_w2 (Nsh1=1000, matched dr and mass/cell to the
  fiducial) and cooling_g100_w2_lores (Nsh1=500, 2x coarser), both t1=t4=2 and stop=tstop at
  1.5e5 s, i.e. paired on physical duration and reaching R/R_0 = 6.62.

  RESOLUTION: alpha_D IS CONVERGED. Across the 2x change the converged value moves by 0.0034
  (RS) / 0.0022 (FS), and over the whole (R_i, R/R_i) domain the signed bias is -0.0008 /
  +0.0006 -- zero to three decimals. The pointwise scatter (median 0.006/0.009, 84th
  0.014/0.021) is flat across the window and its >0.05 tail is 1-2% of samples spread over
  most cells, i.e. per-sample estimator noise from two independent discretisations, not a
  resolution dependence. (The FS 84th of 0.0213 marginally exceeds the 0.02 tolerance set in
  advance; that tolerance was designed to catch a systematic, and the systematic is absent.)
  Consequence: wider runs may be made COARSE. The 2x-coarse run cost 0.150 h against 0.71 h.

  SHELL WIDTH: THE TABLE TRANSFERS, over a 5x range. Median |d alpha_D| over each pair's
  common (R_i/R_0, R/R_i) domain, with the signed bias in brackets:

      fiducial vs W=2   RS 0.0044 (-0.0007)    FS 0.0054 (-0.0002)
      fiducial vs W=5   RS 0.0047 (-0.0013)    FS 0.0059 (-0.0000)
      W=2      vs W=5   RS 0.0056 (-0.0009)    FS 0.0084 (+0.0006)

  Bias <= 0.0013 everywhere. A wider shell is the SAME problem in these variables, which is
  what licenses measuring the table on a wide run and applying it to the fiducial's cells --
  the whole basis of the approach.

  THE SIZING LAW SCALES ~LINEARLY, checked at W=2 and W=5. Per unit W, a goes 1.835 -> 1.931
  -> 1.968 (RS) and b goes 1.227 -> 1.378 -> 1.451, i.e. a mild systematic drift of +7% (a)
  and +18% (b) from W=1 to W=5 rather than exact linearity. The prediction that matters is
  unaffected: the deepest-cell window lands at 1.069 dex against 1.08 predicted (1%), and the
  0.35-dex yield goes 31% -> 48% -> 56% (RS). Size from these three points.

  COST, measured (2 OMP procs):
      W=2 matched  Nsh1=1000  2040 cells  160500 it  0.71 h  2.5 GB
      W=2 coarse   Nsh1= 500  1040 cells   59500 it  0.150 h 0.57 GB
      W=5 coarse   Nsh1=1250  2540 cells  179500 it  0.84 h  4.2 GB
  So a W=5 coarse basis run costs about ONE fiducial run (2.28 h) and a third of its disk. NB
  the W=2 matched iteration count came in ~45% ABOVE an estimate extrapolated from the
  fiducial's cost-per-unit-radius -- a wider shell keeps cells pre-rarefaction, and so
  contracting, for longer, which holds the CFL step down. Size from these measured points.

  A WORKFLOW TRAP, since every run writes to the shared results/Last: only the cells passed to
  extract_data_cells exist afterwards, and the analysis default ncells=40 picks a DIFFERENT
  linspace from the 120 extracted here -- it silently finds ~6 cells instead of ~110. Pass
  ncells=120 for cooling_g100_w2, _w2_lores and _w5. Likewise a Monitor watching results/Last
  re-targets itself onto the next run if it misses the gap between them.

  ALPHA_D IS NOT A FUNCTION OF THETA_SH, and this is the document's ansatz refuted
  (shockstate_test). Within one shell alpha_D correlates with Theta_sh at |r| > 0.99, which
  is worth nothing: Theta_sh is itself monotonic in k, so position and shock state are
  collinear and either reads as the other. Only the cross-shell test discriminates, and it
  fails outright -- the two shells' d(alpha_D)/d(ln Theta_sh) are +0.513 and +1.121 (2.2x
  apart), and the RS law extrapolated to the FS's Theta_sh predicts -1.389 against -0.737
  measured (off by 0.652; the reverse is off by 1.372). The gradient is a POSITION effect --
  cells at different stages of the same relaxation -- not a shock-state law.

  THE WINDOW-MEAN alpha_D IS NEITHER, AND MUST NOT BE QUOTED AS EITHER. It runs -0.929 to
  -0.690 (RS), which looks like strong cell-to-cell variation and is not: it tracks the
  WINDOW LENGTH, not the position. The window spans 0.006-0.50 dex, so a short-window cell
  averages only the steep early transient. alpha_table therefore reports mean, plateau and
  the asymptote in separate columns, and alpha_table's own "plateau" (which ends at each
  cell's dex, collinear with k) is superseded by asymptote() for anything quantitative.

  The transient is physical, not the estimator (window_convergence). Its length is 0.100-
  0.130 dex (RS) and 0.14-0.25 (FS) and does NOT track the slope window over an 8x range of
  half-widths (0.005-0.040 dex), while the fitted value drifts by only 0.0117 (RS) / 0.0062
  (FS) over every half-width and cell stride tried. Note the FS transient is LONGER than
  PLATEAU_FROM = 0.12, so the FS entry in alpha_table's plateau column carries ~0.007 of
  residual transient; asymptote() is free of it.

  NO ABSCISSA CLEANLY WINS, and only one is cleanly excluded (collapse_test):

      abscissa                            RS rel spread   FS rel spread
      proper sound-crossings since t_sh       0.042           0.016
      log10(R/R_inj)                          0.053           0.024
      log10(R/R_0)  (shell-wide)              0.054           0.009
      (R-R_inj)/(R_h-R_inj)                   0.095           0.019

  The top three are within a factor 2 and swap order between shells, which is expected
  rather than disappointing: R_inj varies by only ~3x across a shell, so log(R/R_inj) and
  log(R/R_0) differ by a per-cell constant of at most ~0.5 dex and are very nearly the same
  variable. Do not read a winner out of this table.
  In particular do not promote tau_s on the strength of the RS column. The cell is
  Delta'/R ~ 2e-6 thick, so it is sound-crossed 826-1348 times over its own pre-rarefaction
  window and one crossing is 0.07-0.12% of it: the cell's internal sound clock is three
  orders of magnitude too fast to set a transient that takes 0.1 dex to relax. tau_s scores
  well because it is a monotonic reparametrisation of time, not because it is the physics.
  Nor does log(R/R_0) collapse anything on the FS despite the smallest number in the table --
  its pooled median SWEEPS 0.18 across the range (see alpha_abscissae_z1.png) while the band
  around it stays narrow, i.e. small spread at fixed abscissa with a strongly
  abscissa-dependent value. Small spread is necessary for a collapse, not sufficient.
  What it DOES establish is negative and solid: the wave does not clock the transient. The
  R_h-normalised abscissa is forced to agree at both endpoints AND pools the most cells per
  bin (median 33-36 against 19-22 for the others, since every cell spans [0,1] by
  construction), and it is still the worst or next-to-worst on both shells.

CLOSURES, both of which a prescription needs

  THE ADIABAT HOLDS, ON THE MEASURED ANCHOR. p(rho) along the real pre-rarefaction worldline
  tracks _adiabat_integrated to a median |dln p| of 0.009 (max 0.039) on the RS and 0.003
  (max 0.012) on the FS. The gas is adiabatic to ~1% in p and TM is the right EoS.

  It does NOT hold on the prepended shockfit anchor -- median 0.075, max 0.428 (RS); median
  0.010, max 0.617 (FS), i.e. 8-60x worse -- and that is a statement about the ANCHOR, not
  the adiabat. The prepend is a MODEL state (cellsBehindShock_fromData) spliced in at a
  smaller radius and much higher pressure than the first measured row, so the entropy it
  carries is not the cell's. Running both anchors is the only way to see this, which is why
  adiabat_check does. Consequence for any prescription: anchor the adiabat on the first
  MEASURED row, NOT on the injection row the emission pipeline anchors gma_m/gma_M to.

  The measured index sits slightly BELOW the local TM value on both shells and at every k --
  a_p/a_rho = 1.625 +- 0.002 against gma_ad(Theta) = 1.637-1.647 (RS), and 1.654 +- 0.001
  against 1.658-1.661 (FS). The sign is consistent with a little numerical entropy
  generation (p falling slower than strictly adiabatic); the size, 0.004-0.022 in the index,
  is well inside what the alpha_D spread already costs.
  This is NOT new and NOT specific to the pre-rarefaction phase: the no-rarefaction study
  independently found that with rho ~ R^-2 imposed, the index reproducing the measured p in
  cooling_g100_semi's COASTING phase is 3.25/2 = 1.626, likewise below the frozen
  1.642-1.652 (see the norar-counterfactual note). Two different runs, two different epochs,
  1.626 and 1.625 -- so the deficit is a persistent property of the scheme, not something
  this epoch does.

  BERNOULLI FAILS ON THE RS, as _bernoulli_state's docstring warned it would: h*Gamma drifts
  by a median 3.5% and up to 11.4%, against the 0.7-1.3% it holds to along the post-handover
  coasting phase. That is correct physics -- the shock-crossing transient is not steady flow
  -- and it means Bernoulli cannot close Gamma before the handover.
  On the FS it very nearly survives (median 0.5%, max 2.8%), which is a shell asymmetry, not
  a reprieve: a closure that works on one shell and not the other is not a closure.

  IT DOES NOT NEED TO. alpha_G = -0.029 +- 0.022 (RS) and +0.024 +- 0.009 (FS), with
  |alpha_G| <= 0.062 on every gated cell -- Gamma is constant to a few percent over the whole
  pre-rarefaction phase. Across the shell it is a smooth monotonic ramp, -0.062 at the RS
  outer edge through zero at k ~ 470 to +0.039 at the FS outer edge, CONTINUOUS across the
  contact discontinuity: the outer RS decelerates slightly and the outer FS accelerates
  slightly. Propagated into rho through mass conservation, the full alpha_G spread moves rho
  by 1.3% (RS) / 0.5% (FS) at the handover, far below what Bernoulli's 11% would have cost.

  And Gamma's closure is DECOUPLED from rho's, which is the real argument for keying the
  model on Delta' = Gamma*dr rather than on dr: given alpha_D, mass conservation fixes rho
  and the adiabat fixes p with no reference to Gamma at all. Gamma then enters only the
  observables (Doppler, tt).

  So the prescription the document asked for, every ingredient measured:

      Delta'/Delta'_sh = (R/R_sh)^alpha_D    alpha_D = -0.759  (both shells)
      rho                                    mass conservation, dlnrho/dlnR = -1.241
      p                                      TM adiabat, anchored on the first MEASURED row
      Gamma/Gamma_sh   = (R/R_sh)^alpha_G    alpha_G = -0.03 (RS) .. +0.02 (FS), smooth in k

  It is a constant-index law with no shock-state dependence, valid from the ~0.1-dex
  transient to R_h, where norar_history's law takes over. Whether it is good enough is an
  OBSERVABLE question (eps_rad, fluence) and is NOT settled here.

DOES A TABLE CLOSE THE HISTORY? (table_closure, entropy_anchor)

  Leave-one-out over alpha(R_i/R_0, R/R_i): build the table from every OTHER cell, predict the
  held-out one, integrate. 34 of 36 RS / 31 of 33 FS cells are predictable (the two R_i
  extremes can only be extrapolated, so they are declined).

      quantity                                   median (RS/FS)     worst
      alpha_D pointwise LOO residual, rms         0.020 / 0.040      0.112
      integrated Delta'  at the handover          0.16% / 0.23%      0.56%
      integrated rho                              0.16% / 0.23%      0.56%
      integrated p                                0.26% / 0.38%      0.92%
      Gamma via an alpha_G table -> observer time  0.04% / 0.05%      0.24%
      Gamma = const instead      -> observer time  3.20% / 2.46%      4.01%

  THE POINTWISE RESIDUAL IS ~100x THE INTEGRATED ONE, and that is the useful fact: it is
  oscillatory, not biased, so integration averages it out. Quote the integrated number as the
  table's accuracy and the pointwise one as its noise. Neither substitutes for the other.

  GAMMA NEEDS ITS OWN TABLE. alpha_D gives rho and p with no reference to Gamma (mass
  conservation is rho*R^2*Delta' = const), so Gamma is genuinely unconstrained by it, and
  Bernoulli cannot supply it here (above). |alpha_G| <= 0.062 makes Gamma = const look
  harmless, but Ton = t - R/c with (1-beta) ~ 1/(2 Gamma^2) AMPLIFIES a Gamma error x2: that
  closure costs 2.5-3.2% of the observer time (4.0% worst), a lightcurve timing shift, against
  0.04% for a second table. Take the second table.

  Two independent routes to rho agree to <0.001% (a tabulated alpha_rho, versus mass
  conservation applied to the tabulated alpha_D). p is obtained by running _adiabat_integrated
  over the predicted and true rho TRACKS, not by a frozen-index multiply; its implied index
  comes out at 1.650 / 1.661, slightly ABOVE gma_ad at the shocked state (1.641 / 1.658),
  which is correct -- gma_ad rises toward 5/3 as the expansion cools the gas.

  THE ONE GAP IS THE ENTROPY ANCHOR, and it is a gap only for a PREDICTIVE model. The adiabat
  works anchored on the first measured row and fails on the shock-jump state (above), and a
  predictive model has only the latter. The offset ln(p_meas/p_adiab-from-jump) runs to +0.376
  (RS) / +0.599 (FS), i.e. 46% / 82% in p. A 1-D table in R_i absorbs it -- it interpolates to
  0.37% / 0.52% in p for most cells, and once a constant offset is removed the adiabat tracks
  to a median 1.5% / 0.5% (worst 5.2% / 1.8%) across the window. But 1 cell per shell misses by
  23-36% at this cell stride, so it needs DENSER R_i sampling near the outer edge, not a
  smoother fit.
  The offset is POSITIVE: the gas gains entropy between the jump state and the first measured
  row. Whether that is physical post-shock settling or an inaccuracy of the shockfit state is
  NOT settled, and it matters -- a numerical offset would not transfer to a run at different
  resolution. That is one of the things a resolution probe should look at.

CAVEATS, in order of how much they could change the above

  THE ASYMPTOTE IS MEASURED ON A SELECTED SUBSAMPLE and the selection is not removable. Only
  cells whose window reaches 0.35 dex contribute -- the CD-side third of each shell -- because
  the outer-edge cells are caught by the wave first. Those cells all approach the asymptote
  from ABOVE (shallow to steep). The outer-edge cells approach from BELOW and are still
  rising (-0.82 at 0.2 dex) when their window ends, so that they would converge to the same
  value is consistent with the data but NOT tested by it. This is the single largest
  reservation about the prescription above.

  Every slope uses a FIXED Delta-log-R window, never a fixed row count: the dump cadence
  varies along a history, so a fixed row window spans a varying dlogR (the trap in the
  crash-radius-measurement-traps note).

  Cells are DECLINED rather than fitted, by one shared predicate (declined()): 3 RS and 7 FS
  for a window under DEX_MIN, all at the outer edge, plus k=20 for an UNSHOCKED anchor. That
  last is the pre-existing defect handover_table flags -- its anchor row carries Theta_sh =
  4.5e-5 = Theta0 and Gamma_sh = 199.8 = u4, i.e. cold coasting shell material -- and left in
  it alone returns alpha_D_mean = +1.117 and inflates the plateau spread from 0.045 to 0.291.

  Spreads are median/MAD and inter-quantile, never std. At log10(R/R_inj) = 0.185 a couple of
  cells put the std at 0.22 (RS) while the median does not move at all; a std-based spread
  would report a feature that is not there.

  The mass identity holds at the float32 dump precision floor, 7.50e-08, on every RS cell.
  One FS cell is off it: k=520, the CD-adjacent cell, at 3.8e-05 (identity residual 7.5e-04).
  That is the contact discontinuity, where the moving mesh does the most work, and the audit
  flagging it is the audit working -- it is 500x the floor and still 4 orders below anything
  that would matter.

Example use:
  python -c "import prerar_cell_evolution as P; P.main()"
  python -c "import prerar_cell_evolution as P; print(P.alpha_table('cooling_g100', 4))"
  python -c "import prerar_cell_evolution as P; P.collapse_test('cooling_g100', 4)"
  python -c "import prerar_cell_evolution as P; P.asymptote(); P.shockstate_test()"
  python -c "import prerar_cell_evolution as P; P.table_closure('cooling_g100', 4)"
  python -c "import prerar_cell_evolution as P; P.entropy_anchor('cooling_g100', 4)"
'''

import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from environment import MyEnv, GAMMA_dir
from phys_constants import c_
from phys_functions import derive_cs, derive_enthalpy, derive_adiab_fromT_TM
from working_cooling_data import rarefaction_handover, _adiabat_integrated, _cumtrapz0
from shell_cells import _history, profile_cells, SHELL_NAME

KEY = 'cooling_g100'
Z_LIST = (4, 1)
NCELLS = 40
OUTDIR = os.path.join(GAMMA_dir, 'bin', 'Tools', 'figures', 'prerar_cell')

# --- slope estimator -------------------------------------------------------------------
# Half-width of the least-squares window, in DECADES of R/R_inj. Fixed in dex and never in
# rows: the dump cadence varies by ~2x along a single history and by more across cells, so
# a fixed row count spans a varying dlogR and biases every cross-cell comparison (see the
# crash-radius-measurement-traps note). 0.01 dex is the smallest width whose plateau is
# still converged -- window_convergence puts the drift over 0.005..0.04 dex at <0.006 in
# alpha_D, i.e. well under the 0.013 cross-cell spread it is used to measure.
SLOPE_HALFWIDTH = 0.01
SLOPE_MINPTS = 5         # rows required in the window; below this a slope is not fitted
GRID_STEP = 0.005        # spacing of the common log10(R/R_inj) grid
# Upper end of the common grid, in decades of R/R_inj. Must exceed the LONGEST window in any
# run analysed, or that run's most valuable range is silently discarded: the fiducial's
# deepest cell reaches 0.50 dex, but a shell W times wider reaches log10(1 + 2.25W), i.e.
# 0.73 at W=2 and 1.09 at W=5. 1.2 covers through W=5; grid points past every cell's window
# are NaN and gated out by MIN_OVERLAP, so the only cost of a generous value is a slightly
# larger slope loop.
GRID_MAX = 1.2

# --- gates -----------------------------------------------------------------------------
# A cell whose whole pre-rarefaction window is shorter than this carries no measurable
# slope and is DECLINED, not fitted. The window runs 0.008-0.50 dex across the shell (the
# outer-edge cells are caught by the wave at injection, h <= 1), so this is a real cut, not
# a formality: 0.06 dex is 6x the slope window, the minimum that fixes a slope at all.
# k=1000 shows what the gate is for -- over its 0.010-dex window it returns alpha_G =
# -0.247 against |alpha_G| <= 0.062 everywhere else.
DEX_MIN = 0.06
# A cell whose first post-shock row is not actually SHOCKED carries no measurement at all:
# its alpha_D describes cold coasting shell material, not the shocked layer. The shock
# raises Theta by ~3 orders of magnitude over the shells' initial Theta0, so requiring
# Theta_sh > THETA_SH_FAC*Theta0 separates the two cleanly with nothing in between (a factor
# 32 of margin here).
# This fires on exactly one cell, k=20, whose spurious early Sd block makes
# select_postshock_rows start it 2.4x too early in radius -- the pre-existing defect
# the retired handover_table flagged by a break in the R_end/R_inj ladder. Its anchor row has
# Theta_sh = 5e-5 = Theta0 and Gamma_sh = 199.8 = u4, i.e. unshocked shell-4 material. Left
# in, it alone returns alpha_D_mean = +1.117 against -0.93..-0.69 elsewhere and inflates the
# plateau spread from 0.013 to 0.291. handover_table is right to keep it (the emission
# reference sees the same wrong state, so dropping it there would hide a real bias); here
# the quantity being measured does not exist for it.
THETA_SH_FAC = 10.
# The plateau is measured over a FIXED abscissa range common to every cell. Averaging each
# cell over its own window instead would reintroduce exactly the window-length artifact
# that makes the window-mean useless (see the module docstring). PLATEAU_FROM is where the
# post-shock transient has relaxed (measured, ~0.10-0.12 dex); a cell must cover at least
# PLATEAU_MIN_SPAN of the range past it to get a plateau value at all.
PLATEAU_FROM = 0.12
PLATEAU_MIN_SPAN = 0.03
# First grid point free of the PREPEND STRADDLE. The shockfit row is a model state spliced
# in at a smaller radius and a much higher pressure than the first measured row, so a slope
# window covering both measures that discontinuity rather than the flow: the pooled median
# reads -0.68 over the first 0.010 dex against -0.83 immediately after, and the cell count
# there is 18-29 of 36 (the window is one-sided at g=0, so it is also a different sample).
# Anything quoted "at injection" is measured here instead.
INJ_SAFE = 0.015
# The ASYMPTOTE range. alpha_D does not sit on a plateau: the cells fan out by ~0.28 at
# 0.12 dex and converge to within ~0.08 by 0.35, so the model-relevant number is the value
# they converge ON. [0.25, 0.35] is the range that maximises cells reaching it (11 RS / 9 FS)
# while the two shells still agree to 0.004; pushing to [0.35, 0.45] leaves 4/3 cells and
# the agreement degrades to 0.010, i.e. it is measuring the sample, not the asymptote.
ASYM_FROM, ASYM_TO = 0.25, 0.35
MIN_OVERLAP = 5          # cells required at a grid point before its spread is reported
BAD_MAD = 3.0            # |x - median| beyond this many MAD counts as a bad bin entry
RELAX_TOL = 0.02         # |median/plateau - 1| counting as relaxed onto the plateau
RELAX_PERSIST = 5        # consecutive covered bins it must hold for (5*GRID_STEP = 0.025 dex)
# Stretch at the END of each cell's window that must be discarded before any slope is read.
# Within one slope window of the handover the least-squares fit runs off the end of the
# fitted range and returns nonsense -- measured -0.17 to -0.63 against a true ~-0.79. Costs
# nothing (the handover is where the model stops applying anyway) and silently corrupts the
# converged value if omitted.
EDGE_EXCL = 0.03

COL = {4: '#D55E00', 1: '#0072B2'}      # RS red / FS blue (Charlet et al. 2025)
COL_INK, COL_MUTED = '#1a1a1a', '#8a8a8a'


# ---------------------------------------------------------------------------
# per-cell extraction
# ---------------------------------------------------------------------------
def _cols(s):
  '''(x, dx, rho, p, lfac, t) of a history frame as float64. lfac is absent from the raw
  cell CSVs, so derive it from vx exactly as everything downstream does.'''
  x = s.x.to_numpy(dtype=float)
  vx = s.vx.to_numpy(dtype=float)
  return (x, s.dx.to_numpy(dtype=float), s.rho.to_numpy(dtype=float),
          s.p.to_numpy(dtype=float), 1./np.sqrt(1. - vx*vx), s.t.to_numpy(dtype=float))


def _theta_sh(s):
  '''Theta = p/rho of the anchor row, the quantity the unshocked-anchor gate tests.'''
  return float(s.p.iloc[0])/float(s.rho.iloc[0])


def prerar_window(s):
  '''
  (i_h, dex) for a history: the last row before the rarefaction, and the extent of the
  pre-rarefaction window in decades of R/R_inj.

  i_h comes from rarefaction_handover, which is the crash detector WITH the fan-head
  backoff (_precursor_backoff) -- so the window ends before the wave has touched the cell,
  not merely before its pressure collapses. i_h is None (no crash in this history, so the
  window is the whole thing) is mapped to the last row.
  '''
  h = rarefaction_handover(s)
  if h is None:
    h = len(s) - 1
  x = s.x.to_numpy(dtype=float)
  if h < 1:
    return h, 0.
  return h, float(np.log10(x[h]/x[0]))


def cell_slopes(s, grid, halfwidth=SLOPE_HALFWIDTH, minpts=SLOPE_MINPTS):
  '''
  Local log-slopes of a cell's pre-rarefaction history, on the common grid of
  L = log10(R/R_inj). Returns a dict of arrays aligned with `grid` (NaN where the window
  holds fewer than `minpts` rows), plus the scalars of the window itself.

    a_D   dln(Delta')/dlnR,  Delta' = Gamma*dr      the quantity a model needs
    a_G   dln(Gamma)/dlnR
    a_rho dln(rho)/dlnR                             for the identity audit only

  Slopes are always taken against lnR, whatever abscissa the caller later pools them on:
  alpha_D is DEFINED as a derivative with respect to radius, and re-parametrising the
  derivative as well as the pooling variable would compare different quantities.
  '''
  h, dex = prerar_window(s)
  x, dx, rho, p, lfac, t = (a[:h+1] for a in _cols(s))
  out = dict(h=h, dex=dex, n_rows=h + 1,
             Theta_sh=float(p[0]/rho[0]), lfac_sh=float(lfac[0]),
             rho_sh=float(rho[0]), p_sh=float(p[0]),
             x_inj=float(x[0]), R_h_ratio=float(x[h]/x[0]) if h >= 1 else 1.)
  L = np.log10(x/x[0])
  # a_p is measured for the reconstruction model (prerar_model): the gas is not exactly
  # adiabatic (measured a_p/a_rho = 1.625 against gma_ad = 1.641), so a tabulated a_p
  # captures what the TM adiabat cannot -- see prerar_model.predict_history adiabat='table'.
  series = dict(a_D=np.log10(lfac*dx), a_G=np.log10(lfac), a_rho=np.log10(rho),
                a_p=np.log10(p), a_T=np.log10(p/rho))
  for name, y in series.items():
    sl = np.full(grid.size, np.nan)
    ok = np.isfinite(y)
    for j, g in enumerate(grid):
      m = ok & (np.abs(L - g) <= halfwidth)
      if int(m.sum()) >= minpts and np.ptp(L[m]) > 0.:
        sl[j] = np.polyfit(L[m], y[m], 1)[0]
    out[name] = sl
  # window-mean slopes: one straight line over the whole window. Reported so the
  # window-length artifact is visible in the table rather than hidden in it.
  if h >= 2:
    lnx = np.log(x)
    out['a_D_mean'] = float(np.polyfit(lnx, np.log(lfac*dx), 1)[0])
    out['a_G_mean'] = float(np.polyfit(lnx, np.log(lfac), 1)[0])
  else:
    out['a_D_mean'] = out['a_G_mean'] = np.nan
  # mass-conservation residual over the window: the identity, as an audit on the data
  m = rho*lfac*x*x*dx
  out['mass_res'] = float(np.max(np.abs(np.log(m/m[0])))) if h >= 1 else 0.
  return out


def _grid():
  return np.arange(0., GRID_MAX + 0.5*GRID_STEP, GRID_STEP)


_ENV_MEM = {}

def _env(key):
  if key not in _ENV_MEM:
    _ENV_MEM[key] = MyEnv(key)
  return _ENV_MEM[key]


def declined(Theta_sh, dex, key=KEY):
  '''
  Why this cell carries no alpha_D measurement, or None if it does. One predicate, used by
  every entry point here so a cell can never be measured by one table and declined by the
  next.
    'unshocked'  the anchor row is not shocked (see THETA_SH_FAC)
    'short'      the pre-rarefaction window is shorter than DEX_MIN
  '''
  if not (Theta_sh > THETA_SH_FAC*float(_env(key).Theta0)):
    return 'unshocked'
  if not (dex >= DEX_MIN):
    return 'short'
  return None


def _plateau(grid, a, dex):
  '''Median local slope over the fixed range [PLATEAU_FROM, dex], NaN unless the cell
  covers at least PLATEAU_MIN_SPAN of it. Fixed range, not a per-cell fraction: see
  PLATEAU_FROM.'''
  if not np.isfinite(dex) or dex < PLATEAU_FROM + PLATEAU_MIN_SPAN:
    return np.nan
  m = (grid >= PLATEAU_FROM) & (grid <= dex) & np.isfinite(a)
  return float(np.median(a[m])) if m.sum() >= 3 else np.nan


def _collect(key, z, ncells=NCELLS, ks=None, halfwidth=SLOPE_HALFWIDTH, verbose=False):
  '''
  Per-cell slope dicts for one shell, keyed by k. Histories are built by
  shell_cells._history, i.e. WITH the shockfit prepend, so this measures the history
  production actually sees; skipping the prepend is how a detector artifact once survived
  a full round of per-cell checks.
  '''
  grid = _grid()
  ks = profile_cells(key, z, ncells) if ks is None else np.asarray(ks)
  out = {}
  for k in ks:
    s = _history(key, int(k), z=z)
    if s is None or len(s) < 3:
      continue
    d = cell_slopes(s, grid, halfwidth=halfwidth)
    d['k'] = int(k)
    d['declined'] = declined(d['Theta_sh'], d['dex'], key)
    d['measurable'] = d['declined'] is None
    out[int(k)] = d
  if verbose:
    print(f'  {len(out)} histories, {sum(d["measurable"] for d in out.values())} measurable')
  return grid, out


# ---------------------------------------------------------------------------
# S1: the alpha table
# ---------------------------------------------------------------------------
def alpha_table(key=KEY, z=4, ncells=NCELLS, ks=None, verbose=True):
  '''
  Per cell: the pre-rarefaction window, the shock state, and the slopes.

  a_D_plateau / a_G_plateau are the model-relevant numbers. a_D_mean / a_G_mean are the
  single-line fits over the whole window and are reported ONLY so the window-length
  artifact stays visible: they run -0.93..-0.69 in a_D and track `dex`, not `k`. Never
  quote the mean as alpha_D.
  '''
  grid, cells = _collect(key, z, ncells, ks)
  rows = []
  for k in sorted(cells):
    d = cells[k]
    rows.append(dict(k=k, dex=d['dex'], R_h=d['R_h_ratio'], n_rows=d['n_rows'],
                     Theta_sh=d['Theta_sh'], lfac_sh=d['lfac_sh'],
                     a_D_inj=d['a_D'][int(round(INJ_SAFE/GRID_STEP))],
                     a_D_plateau=_plateau(grid, d['a_D'], d['dex']),
                     a_D_mean=d['a_D_mean'],
                     a_G_plateau=_plateau(grid, d['a_G'], d['dex']),
                     a_G_mean=d['a_G_mean'],
                     mass_res=d['mass_res'], measurable=d['measurable'],
                     declined=d['declined']))
  df = pd.DataFrame(rows)
  if verbose and len(df):
    ok = df[df.measurable]
    print(f'\nalpha_table {key} {SHELL_NAME.get(z, z)}: {len(df)} cells, '
          f'{len(ok)} measurable (dex >= {DEX_MIN})')
    # ranges over the MEASURABLE cells: an unshocked anchor carries neither a window nor a
    # shock state, so including it would misreport both
    print(f'  window        dex {ok.dex.min():.3f} .. {ok.dex.max():.3f}   '
          f'R_h/R_inj {ok.R_h.min():.3f} .. {ok.R_h.max():.3f}')
    print(f'  shock state   Theta_sh {ok.Theta_sh.min():.4f} .. {ok.Theta_sh.max():.4f}   '
          f'Gamma_sh {ok.lfac_sh.min():.1f} .. {ok.lfac_sh.max():.1f}')
    pl = ok.a_D_plateau.dropna()
    if len(pl):
      # half inter-quantile, not std: see _robust
      sd = 0.5*float(np.subtract(*np.percentile(pl, [84, 16])))
      sub = ok.dropna(subset=['a_D_plateau'])
      print(f'  a_D plateau   {pl.median():+.3f} +- {sd:.3f}  '
            f'({len(pl)} cells)   => dlnrho/dlnR = {-2. - pl.median():+.3f}')
      print(f'                range {pl.min():+.3f} .. {pl.max():+.3f}, '
            f'corr with k = {np.corrcoef(sub.a_D_plateau, sub.k)[0, 1]:+.3f}  '
            '<- a resolved GRADIENT, not scatter')
    gl = ok.a_G_plateau.dropna()
    if len(gl):
      sdg = 0.5*float(np.subtract(*np.percentile(gl, [84, 16])))
      print(f'  a_G plateau   {gl.median():+.4f} +- {sdg:.4f}   '
            f'max |a_G| = {gl.abs().max():.4f}')
    print(f'  a_D window-mean {ok.a_D_mean.min():+.3f} .. {ok.a_D_mean.max():+.3f}  '
          '<- tracks `dex`, NOT position; not the plateau')
    dec = df[~df.measurable]
    if len(dec):
      for why, g in dec.groupby('declined'):
        print(f'  DECLINED {len(g)} {why}: ' + ', '.join(
            f'k={int(r.k)} (dex {r.dex:.3f}, Theta_sh {r.Theta_sh:.2g})'
            for _, r in g.iterrows()))
  return df


def mass_conservation_audit(key=KEY, z=4, ncells=NCELLS, ks=None, verbose=True):
  '''
  max |dln(rho*Gamma*R^2*dr)| along each pre-rarefaction worldline, and the consistency of
  the two routes to dlnrho/dlnR.

  This is an AUDIT, not a result: the product is the code's own invariant and
  smooth_dump_density preserves it by construction, so the expected value is the float32
  dump precision floor and nothing else. Its use is that a cell departing from that floor
  has taken remap or restart damage, and that `id_res` catches an inconsistent slope
  estimator (a_rho and -2-a_D are the same measurement; if they disagree, the estimator,
  not the physics, is at fault).
  '''
  grid, cells = _collect(key, z, ncells, ks)
  rows = []
  for k in sorted(cells):
    d = cells[k]
    if not d['measurable']:
      continue
    m = np.isfinite(d['a_rho']) & np.isfinite(d['a_D']) & (grid <= d['dex'])
    idr = np.max(np.abs(d['a_rho'][m] - (-2. - d['a_D'][m]))) if m.any() else np.nan
    rows.append(dict(k=k, dex=d['dex'], mass_res=d['mass_res'], id_res=idr))
  df = pd.DataFrame(rows)
  if verbose and len(df):
    print(f'\nmass_conservation_audit {key} {SHELL_NAME.get(z, z)}: {len(df)} cells')
    print(f'  max |dln(rho.Gamma.R^2.dr)|   {df.mass_res.min():.2e} .. '
          f'{df.mass_res.max():.2e}   (median {df.mass_res.median():.2e})')
    print(f'  |a_rho - (-2 - a_D)|         {df.id_res.min():.2e} .. '
          f'{df.id_res.max():.2e}   (median {df.id_res.median():.2e})')
    hi = df[df.mass_res > 10.*df.mass_res.median()]
    if len(hi):
      print('  *** off the precision floor (remap/restart damage?): ' +
            ', '.join(f'k={int(r.k)} ({r.mass_res:.1e})' for _, r in hi.iterrows()))
  return df


# ---------------------------------------------------------------------------
# S1: does it collapse, and on what abscissa
# ---------------------------------------------------------------------------
def _abscissa(s, name, h):
  '''
  Value of a candidate pooling abscissa at every row of the window [0, h].
    R_Rinj  log10(R/R_inj)              the cell's own expansion factor
    frac    (R-R_inj)/(R_h-R_inj)       clocked by the arriving wave
    tau_s   proper sound-crossings of the cell since t_sh, tau/(Delta'_sh/c_s,sh)
    R_R0    log10(R/R_0)                a shell-wide function of radius
  '''
  x, dx, rho, p, lfac, t = (a[:h+1] for a in _cols(s))
  if name == 'R_Rinj':
    return np.log10(x/x[0])
  if name == 'frac':
    return (x - x[0])/(x[h] - x[0]) if x[h] > x[0] else np.full(x.shape, np.nan)
  if name == 'tau_s':
    # code units have c = 1 (R = x*c_), so Delta' = Gamma*dr is already a time and the
    # cell's proper sound-crossing time is Delta'_sh/c_s,sh. Proper time elapsed is
    # integral dt/Gamma, which is the frame the sound crossing happens in.
    ts = lfac[0]*dx[0]/derive_cs(rho[0], p[0])
    return _cumtrapz0(1./lfac, t)/ts
  if name == 'R_R0':
    return np.log10(x*c_/_env(_ABS_KEY[0]).R0)
  raise ValueError(f'unknown abscissa {name!r}')


_ABS_KEY = [KEY]        # R_R0 needs the run's R0; set by collapse_test before use
ABSCISSAE = ('R_Rinj', 'frac', 'tau_s', 'R_R0')
ABS_LABEL = {'R_Rinj': r'$\log_{10}(R/R_{\rm inj})$',
             'frac': r'$(R-R_{\rm inj})/(R_h-R_{\rm inj})$',
             'tau_s': r'proper sound-crossings since $t_{\rm sh}$',
             'R_R0': r'$\log_{10}(R/R_0)$'}


def _robust(Y):
  '''(median, MAD-sigma, half inter-quantile, n, bad fraction) down axis 0, NaN-safe.
  Median/MAD and quantiles, never std: a couple of bad cells put the std at 0.22 where the
  median does not move at all, and a std-based spread would report that as structure.'''
  # grid points past every cell's coverage are all-NaN BY DESIGN (each cell's window ends at
  # its own handover), so the empty-slice warnings are expected and the n array below is what
  # the caller gates on
  with np.errstate(invalid='ignore'), warnings.catch_warnings():
    warnings.simplefilter('ignore', RuntimeWarning)
    med = np.nanmedian(Y, axis=0)
    mad = 1.4826*np.nanmedian(np.abs(Y - med), axis=0)
    q16, q84 = np.nanpercentile(Y, [16., 84.], axis=0)
  n = np.sum(np.isfinite(Y), axis=0)
  with np.errstate(invalid='ignore', divide='ignore'):
    bad = np.sum(np.abs(Y - med) > BAD_MAD*mad, axis=0)/np.maximum(n, 1)
  return med, mad, 0.5*(q84 - q16), n, bad


def collapse_test(key=KEY, z=4, ncells=NCELLS, ks=None, abscissae=ABSCISSAE,
    halfwidth=SLOPE_HALFWIDTH, verbose=True):
  '''
  Cross-cell spread of the LOCAL alpha_D at fixed abscissa, for each candidate pooling
  variable. The winner is the one with the smallest relative spread: that is the variable
  the pre-rarefaction transient is a function of.

  The slope itself is always dln(Delta')/dlnR (see cell_slopes); only the pooling changes.
  Returns (summary DataFrame, {abscissa: (grid, stacked alpha_D)}).
  '''
  _ABS_KEY[0] = key
  grid = _grid()
  ks = profile_cells(key, z, ncells) if ks is None else np.asarray(ks)
  # per cell: the local alpha_D on the R/R_inj grid, plus each abscissa evaluated there
  per = []
  for k in ks:
    s = _history(key, int(k), z=z)
    if s is None or len(s) < 3:
      continue
    d = cell_slopes(s, grid, halfwidth=halfwidth)
    if declined(d['Theta_sh'], d['dex'], key) is not None:
      continue
    h = d['h']
    L = _abscissa(s, 'R_Rinj', h)
    cell = dict(k=int(k), a_D=d['a_D'], dex=d['dex'])
    for nm in abscissae:
      v = _abscissa(s, nm, h)
      # the grid lives in log10(R/R_inj); map it onto each abscissa through the cell's own
      # rows, so a grid point carries the same PHYSICAL state on every abscissa
      cell[nm] = np.interp(grid, L, v, left=np.nan, right=np.nan)
    per.append(cell)

  stacks, rows = {}, []
  for nm in abscissae:
    lo = np.nanmin([np.nanmin(c[nm]) for c in per])
    hi = np.nanmax([np.nanmax(c[nm]) for c in per])
    xg = np.linspace(lo, hi, 60)
    Y = []
    for c in per:
      m = np.isfinite(c[nm]) & np.isfinite(c['a_D'])
      if m.sum() < 2:
        continue
      v, a = c[nm][m], c['a_D'][m]
      o = np.argsort(v)
      row = np.interp(xg, v[o], a[o], left=np.nan, right=np.nan)
      Y.append(row)
    Y = np.array(Y)
    med, mad, iqr, n, bad = _robust(Y)
    ok = n >= MIN_OVERLAP
    with np.errstate(invalid='ignore', divide='ignore'):
      rel = iqr/np.abs(med)
    stacks[nm] = (xg, Y, med, iqr, n)
    rows.append(dict(abscissa=nm, n_cells=len(Y), n_bins=int(ok.sum()),
                     rel_spread=float(np.nanmedian(rel[ok])),
                     abs_spread=float(np.nanmedian(iqr[ok])),
                     bad_frac=float(np.nanmedian(bad[ok]))))
  df = pd.DataFrame(rows).sort_values('rel_spread').reset_index(drop=True)
  if verbose:
    print(f'\ncollapse_test {key} {SHELL_NAME.get(z, z)}: {len(per)} measurable cells')
    print('  abscissa   n_bins  rel spread  abs spread  bad frac')
    for _, r in df.iterrows():
      print(f'  {r.abscissa:9s} {r.n_bins:6d}  {r.rel_spread:10.3f}  {r.abs_spread:10.3f}'
            f'  {r.bad_frac:8.3f}')
    print(f'  -> collapses on {df.abscissa.iloc[0]!r}')
  return df, stacks


def plateau_common(key=KEY, z=4, ncells=NCELLS, ks=None,
    lo=PLATEAU_FROM, hi=PLATEAU_FROM + PLATEAU_MIN_SPAN):
  '''
  alpha_D over a range [lo, hi] common to EVERY cell, with the per-cell position, shock
  state and window length alongside.

  alpha_table's plateau ends at each cell's own `dex`, which is collinear with k (cells
  nearer the CD are shocked first and have longer windows), so a gradient measured there
  cannot be told from an averaging effect. Fixing both ends removes `dex` from the
  measurement and is what shows the gradient to be real.
  '''
  grid = _grid()
  _, cells = _collect(key, z, ncells, ks)
  rows = []
  for k in sorted(cells):
    d = cells[k]
    if not d['measurable'] or d['dex'] < hi:
      continue
    m = (grid >= lo) & (grid <= hi) & np.isfinite(d['a_D'])
    if m.sum() < 3:
      continue
    rows.append(dict(k=k, Ri=float(d['x_inj']*c_/_env(key).R0),
                     a_D=float(np.median(d['a_D'][m])), Theta_sh=d['Theta_sh'],
                     lfac_sh=d['lfac_sh'], dex=d['dex'], R_h=d['R_h_ratio']))
  return pd.DataFrame(rows)


def converged_alpha(key=KEY, z=4, ncells=NCELLS, lo=0.55, hi=0.65, edge=EDGE_EXCL,
    verbose=True):
  '''
  The CONVERGED alpha_D per cell, against that cell's R_i/R_0 -- the deliverable a table
  needs, and what a run wider than the fiducial exists to measure.

  alpha_D steepens monotonically through each cell's window and flattens only around 0.60
  dex (see the module docstring), so [lo, hi] = [0.55, 0.65] by default and ONLY cells whose
  window reaches `hi + edge` contribute. On the fiducial that is nobody -- its longest window
  is 0.50 dex, which is exactly why its asymptote() came out ~0.03 too shallow.

  `edge` drops the last stretch of each window: within one slope window of the handover the
  estimator runs off the fitted range and returns nonsense (-0.17..-0.63 measured).
  '''
  grid = _grid()
  _, cells, Ri = _cells_by_Ri(key, z, ncells)
  rows = []
  for d, ri in zip(cells, Ri):
    if d['dex'] < hi + edge:
      continue
    m = (grid >= lo) & (grid <= min(hi, d['dex'] - edge)) & np.isfinite(d['a_D'])
    if m.sum() >= 3:
      rows.append(dict(k=d['k'], Ri=float(ri), dex=d['dex'],
                       a_D=float(np.median(d['a_D'][m])), Theta_sh=d['Theta_sh']))
  df = pd.DataFrame(rows)
  if verbose:
    print(f'\nconverged_alpha {key} {SHELL_NAME.get(z, z)}  '
          f'(median over [{lo}, {hi}] dex, edge {edge})')
    if not len(df):
      print(f'  NO cells reach {hi + edge:.2f} dex -- this run cannot measure the converged '
            'value (the fiducial cannot; that is the whole point of a wider run)')
      return df
    sd = 0.5*float(np.subtract(*np.percentile(df.a_D, [84, 16]))) if len(df) > 2 else np.nan
    print(f'  {len(df)} cells, R_i/R_0 in [{df.Ri.min():.2f}, {df.Ri.max():.2f}]')
    print(f'  alpha_D = {df.a_D.median():+.4f} +- {sd:.4f}   range {df.a_D.min():+.4f} .. '
          f'{df.a_D.max():+.4f}')
    if len(df) > 3:
      r = float(np.corrcoef(df.a_D, df.Ri)[0, 1])
      span = float(df.a_D.max() - df.a_D.min())
      # judge on the SPAN, not the correlation: with a tight scatter even a trivial trend
      # correlates strongly, and it is the span that decides whether a constant will do
      verdict = ('spans only %.3f -- a CONSTANT is adequate' % span if span <= 0.05
                 else 'spans %.3f -- too much for a constant, needs the R_i axis' % span)
      print(f'  corr(alpha_D, R_i) = {r:+.3f}, {verdict}')
      for a, b in ((1.0, 1.5), (1.5, 2.0), (2.0, 3.0)):
        s = df[(df.Ri >= a) & (df.Ri < b)]
        if len(s):
          print(f'    R_i/R_0 [{a:.1f},{b:.1f}): {s.a_D.median():+.4f}  (n={len(s)})')
  return df


def asymptote(key=KEY, z_list=Z_LIST, ncells=NCELLS, lo=ASYM_FROM, hi=ASYM_TO,
    verbose=True):
  '''
  The value alpha_D CONVERGES on, per shell, and whether the shells agree.

  This is the model-relevant number, and it is not the same thing as alpha_table's plateau.
  The cells do not sit on a plateau: at 0.12 dex they are spread over ~0.28 in alpha_D and
  ordered by k, and they converge to within ~0.08 by 0.35 dex. So the quantity a prescription
  should carry is the common asymptote, measured over [lo, hi].

  IT IS MEASURED ON A SELECTED SUBSAMPLE and that limitation is not removable: only cells
  whose pre-rarefaction window reaches `hi` contribute, which is the CD-side third of each
  shell (the outer-edge cells are caught by the wave first -- see DEX_MIN). Whether an
  outer-edge cell would converge to the same value cannot be tested on this data, because
  its window ends before it gets there. What CAN be said is that the cells which do reach it
  agree across two shells whose Theta_sh differs by 3.3x.
  '''
  grid = _grid()
  out = {}
  for z in z_list:
    _, cells = _collect(key, z, ncells)
    vals, ks = [], []
    for k in sorted(cells):
      d = cells[k]
      if not d['measurable'] or d['dex'] < hi:
        continue
      m = (grid >= lo) & (grid <= hi) & np.isfinite(d['a_D'])
      if m.sum() >= 3:
        vals.append(float(np.median(d['a_D'][m])))
        ks.append(k)
    v = np.array(vals)
    out[z] = dict(n=v.size, n_total=sum(d['measurable'] for d in cells.values()),
                  dex_max=max((d['dex'] for d in cells.values() if d['measurable']),
                              default=0.),
                  a_D=float(np.median(v)) if v.size else np.nan,
                  spread=float(0.5*np.subtract(*np.percentile(v, [84, 16])))
                  if v.size > 2 else np.nan, ks=ks)
  if verbose:
    print(f'\nasymptote {key}, alpha_D over log10(R/R_inj) in [{lo}, {hi}]')
    for z, d in out.items():
      print(f'  {SHELL_NAME.get(z, z):3s}  {d["a_D"]:+.4f} +- {d["spread"]:.4f}   '
            f'({d["n"]} of {d["n_total"]} measurable cells reach it, k = '
            f'{min(d["ks"]) if d["ks"] else "-"}..{max(d["ks"]) if d["ks"] else "-"})')
    # On a run whose windows run far past [lo, hi] this estimator is MISLEADING, and
    # increasingly so with width: its fixed window catches the large-R_i cells mid-transient
    # and averages them in. Measured on the same quantity: fiducial -0.760, W=2 -0.808,
    # W=5 -0.840, while the actual converged value is ~-0.795 on all three. Use
    # converged_alpha for anything wider than the fiducial.
    if any(d.get('dex_max', 0.) > hi + 0.15 for d in out.values()):
      print(f'  *** WINDOWS EXTEND WELL PAST {hi} dex: this fixed-window estimator mixes in '
            'cells still\n      mid-transient and drifts steeper with shell width. '
            'Use converged_alpha() instead.')
    zs = [z for z in out if np.isfinite(out[z]['a_D'])]
    if len(zs) >= 2:
      a, b = out[zs[0]]['a_D'], out[zs[1]]['a_D']
      print(f'  cross-shell |{SHELL_NAME.get(zs[0], zs[0])} - '
            f'{SHELL_NAME.get(zs[1], zs[1])}| = {abs(a - b):.4f}   '
            f'=> alpha_D = {0.5*(a + b):+.3f},  dlnrho/dlnR = {-2. - 0.5*(a + b):+.3f}')
  return out


def shockstate_test(key=KEY, z_list=Z_LIST, ncells=NCELLS, verbose=True):
  '''
  Is alpha_D a function of the shock state, as f(Theta_sh, Gamma_sh, ...) supposes?

  Within one shell alpha_D correlates with Theta_sh at |r| > 0.99 -- and that means nothing,
  because Theta_sh is itself monotonic in k, so the two are collinear and any position
  dependence reads as a Theta dependence. The test that discriminates is whether ONE law
  fits BOTH shells: the RS spans Theta_sh = 0.047-0.071 and the FS 0.017-0.018, so a genuine
  Theta law fitted on one must predict the other.

  Reports each shell's d(alpha_D)/d(ln Theta_sh) and the cross-shell prediction error. It
  cannot test Gamma_sh at all -- 125-130 everywhere, see the module docstring.
  '''
  fits = {}
  for z in z_list:
    df = plateau_common(key, z, ncells)
    if len(df) < 4:
      continue
    sl = np.polyfit(np.log(df.Theta_sh.to_numpy()), df.a_D.to_numpy(), 1)
    fits[z] = (df, sl)
    if verbose:
      print(f'\nshockstate_test {key} {SHELL_NAME.get(z, z)}: {len(df)} cells')
      print(f'  Theta_sh {df.Theta_sh.min():.4f}..{df.Theta_sh.max():.4f} '
            f'(lever {df.Theta_sh.max()/df.Theta_sh.min():.2f}x)   '
            f'Gamma_sh {df.lfac_sh.min():.1f}..{df.lfac_sh.max():.1f}')
      print(f'  alpha_D  {df.a_D.min():+.3f}..{df.a_D.max():+.3f}   '
            f'corr(alpha_D, k) = {np.corrcoef(df.a_D, df.k)[0, 1]:+.3f}   '
            f'corr(alpha_D, Theta_sh) = {np.corrcoef(df.a_D, df.Theta_sh)[0, 1]:+.3f}')
      print(f'  d(alpha_D)/d(ln Theta_sh) = {sl[0]:+.3f}   '
            '<- meaningless alone: Theta_sh is collinear with k')
  if verbose and len(fits) >= 2:
    (za, (da, sa)), (zb, (db, sb)) = list(fits.items())[:2]
    print(f'\n  CROSS-SHELL, the only test that discriminates:')
    print(f'    slopes  {SHELL_NAME.get(za, za)} {sa[0]:+.3f}   '
          f'{SHELL_NAME.get(zb, zb)} {sb[0]:+.3f}   '
          f'(disagree by {max(sa[0], sb[0])/min(sa[0], sb[0]):.1f}x)')
    for (z1, s1), (z2, d2) in (((za, sa), (zb, db)), ((zb, sb), (za, da))):
      th = float(d2.Theta_sh.median())
      pred, meas = float(np.polyval(s1, np.log(th))), float(d2.a_D.median())
      print(f'    {SHELL_NAME.get(z1, z1)} law at {SHELL_NAME.get(z2, z2)} '
            f'Theta_sh={th:.4f}: predicts {pred:+.3f}, measured {meas:+.3f} '
            f'(off by {abs(pred - meas):.3f})')
    print('    => alpha_D is NOT a function of Theta_sh. The within-shell correlation is '
          'collinearity\n       with position, and the gradient is a POSITION effect.')
  return fits


def window_convergence(key=KEY, z=4, ncells=NCELLS,
    halfwidths=(0.005, 0.01, 0.02, 0.04), strides=(1, 2, 4), verbose=True):
  '''
  The plateau against the slope half-width and against the cell sample.

  Without this the measured ~0.1-dex relaxation length cannot be told from the smoothing
  scale of the estimator: a window of w dex smears any transient over w, so a transient
  whose length TRACKS w is an artifact of the window and one that does not is physical.
  '''
  grid = _grid()
  rows = []
  for hw in halfwidths:
    for st in strides:
      ks = profile_cells(key, z, ncells)[::st]
      _, cells = _collect(key, z, ks=ks, halfwidth=hw)
      pl = np.array([_plateau(grid, d['a_D'], d['dex'])
                     for d in cells.values() if d['measurable']])
      pl = pl[np.isfinite(pl)]
      # Relaxation length: the first covered grid point where the pooled median comes within
      # RELAX_TOL of the plateau AND STAYS there for RELAX_PERSIST consecutive covered bins.
      # Persistence rather than "for the whole tail": past ~0.4 dex only the few longest-
      # window cells remain, so the pooled median there is a different sample and wanders,
      # which made a whole-tail rule return NaN on runs whose transient is perfectly clean.
      A = np.array([d['a_D'] for d in cells.values() if d['measurable']])
      med = np.nanmedian(A, axis=0)
      n = np.sum(np.isfinite(A), axis=0)
      tgt = np.median(pl) if pl.size else np.nan
      okb = (n >= MIN_OVERLAP) & np.isfinite(med)
      cov = np.flatnonzero(okb)
      near = np.abs(med/tgt - 1.) <= RELAX_TOL
      relax = np.nan
      for i, j in enumerate(cov):
        run = cov[i:i + RELAX_PERSIST]
        if run.size == RELAX_PERSIST and np.all(near[run]):
          relax = float(grid[j])
          break
      rows.append(dict(halfwidth=hw, stride=st, n_cells=len(pl),
                       a_D_plateau=float(np.median(pl)) if pl.size else np.nan,
                       spread=float(0.5*np.subtract(*np.percentile(pl, [84, 16])))
                       if pl.size > 2 else np.nan,
                       relax_dex=relax))
  df = pd.DataFrame(rows)
  if verbose:
    print(f'\nwindow_convergence {key} {SHELL_NAME.get(z, z)}')
    print('  halfwidth  stride  n   a_D plateau  spread  relax(dex)')
    for _, r in df.iterrows():
      print(f'  {r.halfwidth:9.3f}  {int(r.stride):6d} {int(r.n_cells):3d}  '
            f'{r.a_D_plateau:+11.4f}  {r.spread:6.4f}  {r.relax_dex:10.3f}')
    p = df.a_D_plateau.dropna()
    print(f'  plateau drift over all settings: {p.max() - p.min():.4f}  '
          f'(against a cross-cell spread of ~{df.spread.median():.3f})')
  return df


# ---------------------------------------------------------------------------
# S2: the two closures
# ---------------------------------------------------------------------------
def adiabat_check(key=KEY, z=4, ncells=NCELLS, ks=None, verbose=True):
  '''
  Is the pre-rarefaction gas on the Taub-Matthews adiabat through its shocked state?

  p_data(R) against _adiabat_integrated(rho_data, rho_a, p_a) along the real worldline, for
  BOTH candidate entropy anchors:

    anchor 'inj'   row 0 = the prepended shockfit state, which is what the emission
                   pipeline anchors gma_m/gma_M on. It is a MODEL state
                   (cellsBehindShock_fromData) at a smaller radius and much higher pressure
                   than the first measured row, so the entropy it carries need not be the
                   cell's.
    anchor 'meas'  the first MEASURED row.

  Running both is the only way to separate 'the adiabat fails' from 'the anchor is wrong',
  and on cooling_g100 that distinction is the whole result. Also reports the implied index
  a_p/a_rho, to be compared with gma_ad(Theta) from the EoS directly.
  '''
  ks = profile_cells(key, z, ncells) if ks is None else np.asarray(ks)
  rows = []
  for k in ks:
    s = _history(key, int(k), z=z)
    if s is None or len(s) < 3:
      continue
    h, dex = prerar_window(s)
    if declined(_theta_sh(s), dex, key) is not None:
      continue
    # how many rows the prepend added, so 'meas' is the first row that was measured
    s_raw = _history(key, int(k), z=z, early_ana=None)
    n_pre = len(s) - len(s_raw) if s_raw is not None else 0
    x, dx, rho, p, lfac, t = (a[:h+1] for a in _cols(s))
    rec = dict(k=int(k), dex=dex, n_pre=int(n_pre), Theta_sh=float(p[0]/rho[0]),
               gma_ad=float(derive_adiab_fromT_TM(p/rho).mean()))
    for tag, i0 in (('inj', 0), ('meas', int(n_pre))):
      if i0 >= h:
        rec[f'res_{tag}'] = rec[f'resmax_{tag}'] = np.nan
        continue
      pm = _adiabat_integrated(rho[i0:], rho[i0], p[i0])
      d = np.abs(np.log(pm/p[i0:]))
      rec[f'res_{tag}'] = float(np.median(d))
      rec[f'resmax_{tag}'] = float(np.max(d))
    # measured effective index, on the measured stretch only
    i0 = int(n_pre)
    rec['a_p_a_rho'] = float(np.polyfit(np.log(rho[i0:]), np.log(p[i0:]), 1)[0]) \
        if h - i0 >= 2 else np.nan
    rows.append(rec)
  df = pd.DataFrame(rows)
  if verbose and len(df):
    print(f'\nadiabat_check {key} {SHELL_NAME.get(z, z)}: {len(df)} cells '
          f'({int((df.n_pre > 0).sum())} with a prepended row)')
    for tag, lab in (('meas', 'first MEASURED row'), ('inj', 'prepended shockfit row')):
      med, mx = df[f'res_{tag}'].median(), df[f'resmax_{tag}'].max()
      print(f'  anchor {tag:4s} ({lab:22s})  |dln p| median {med:.3f}  max {mx:.3f}')
    print(f'  a_p/a_rho   {df.a_p_a_rho.median():.3f} +- {df.a_p_a_rho.std():.3f}   '
          f'vs gma_ad(Theta) = {df.gma_ad.min():.3f} .. {df.gma_ad.max():.3f}')
  return df


def bernoulli_check(key=KEY, z=4, ncells=NCELLS, ks=None, verbose=True):
  '''
  Can Bernoulli close Gamma before the handover? Drift of h*Gamma along the real
  pre-rarefaction worldline, h from derive_enthalpy (bit-identical to the TM
  h = 2.5T + sqrt(1+2.25T^2), verified to 2.2e-16).

  _bernoulli_state uses h*Gamma = const to close the POST-handover tail, where it is
  measured to hold to 0.7-1.3%, and its docstring already warns that the shock-crossing
  transient violates it by up to 8% -- correctly, since that flow is not steady. This
  quantifies that, and prices the alternative: alpha_G is small enough that a measured
  power law closes Gamma far more tightly than Bernoulli would. The sensitivity that
  matters is what the alpha_G spread does to rho THROUGH mass conservation, so that is
  reported too.
  '''
  ks = profile_cells(key, z, ncells) if ks is None else np.asarray(ks)
  rows = []
  for k in ks:
    s = _history(key, int(k), z=z)
    if s is None or len(s) < 3:
      continue
    h, dex = prerar_window(s)
    if declined(_theta_sh(s), dex, key) is not None:
      continue
    x, dx, rho, p, lfac, t = (a[:h+1] for a in _cols(s))
    hb = derive_enthalpy(rho, p)*lfac
    d = np.abs(np.log(hb/hb[0]))
    rows.append(dict(k=int(k), dex=dex,
                     bern_med=float(np.median(d)), bern_max=float(np.max(d)),
                     bern_end=float(np.abs(np.log(hb[-1]/hb[0]))),
                     h_drift=float(np.log(derive_enthalpy(rho[-1], p[-1])
                                          / derive_enthalpy(rho[0], p[0]))),
                     a_G=float(np.polyfit(np.log(x), np.log(lfac), 1)[0]),
                     lfac_drift=float(np.log(lfac[-1]/lfac[0]))))
  df = pd.DataFrame(rows)
  if verbose and len(df):
    print(f'\nbernoulli_check {key} {SHELL_NAME.get(z, z)}: {len(df)} cells')
    print(f'  |dln(h.Gamma)|   median {df.bern_med.median():.4f}   '
          f'max {df.bern_max.max():.4f}   at the handover {df.bern_end.median():.4f}')
    print('    (post-handover coasting holds it to 0.007-0.013 -- see _bernoulli_state)')
    print(f'  a_G              {df.a_G.median():+.4f} +- {df.a_G.std():.4f}   '
          f'max |a_G| {df.a_G.abs().max():.4f}')
    # the alpha_G freedom, propagated into rho through rho ~ R^(-2-a_G-a_dr)
    dex_med = df.dex.median()
    drho = np.expm1(np.log(10.)*dex_med*df.a_G.std())
    print(f'  a_G spread propagated into rho at the handover (dex {dex_med:.2f}): '
          f'{100.*abs(drho):.1f}%')
  return df


# ---------------------------------------------------------------------------
# S3: does a TABLE of alpha close the pre-rarefaction history?
# ---------------------------------------------------------------------------
def _cells_by_Ri(key, z, ncells, ks=None, halfwidth=SLOPE_HALFWIDTH):
  '''Measurable cells and their R_i/R_0, both sorted by R_i. Sorting by R_i and not by k is
  required: the table's first argument IS R_i, and on the RS k runs the other way (the
  CD-side cells are shocked first, at the smallest radius).'''
  grid, cells = _collect(key, z, ncells, ks, halfwidth=halfwidth)
  ok = [d for d in cells.values() if d['measurable']]
  Ri = np.array([d['x_inj']*c_/_env(key).R0 for d in ok])
  o = np.argsort(Ri)
  return grid, [ok[i] for i in o], Ri[o]


def interp_alpha(cells, Ri, Ri_at, grid, series, exclude=None):
  '''
  alpha(grid) for a cell at R_i = Ri_at, by interpolating the table cells in R_i at each grid
  point -- i.e. exactly what a tabulated alpha(R_i/R_0, R/R_i) returns.

  exclude: index of a table cell to leave out. Pass the test cell's own index for a
  leave-one-out prediction; pass None for an ordinary table lookup. One code path serves both
  so the two can never diverge.

  NaN wherever Ri_at falls outside the (remaining) cells' R_i range -- the table DECLINES
  rather than extrapolates. Extrapolating the R_i axis would report the table as worse than
  it is for a reason that has nothing to do with the table.
  '''
  keep = [i for i in range(len(cells)) if i != exclude]
  pred = np.full(grid.size, np.nan)
  for g in range(grid.size):
    xs, ys = [], []
    for i in keep:
      v = cells[i][series][g]
      if np.isfinite(v):
        xs.append(Ri[i])
        ys.append(v)
    if len(xs) >= 3 and min(xs) <= Ri_at <= max(xs):
      pred[g] = np.interp(Ri_at, xs, ys)
  return pred


def _loo_predict(cells, Ri, j, grid, series):
  '''Leave-one-out shorthand: interp_alpha for cell j with cell j excluded.'''
  return interp_alpha(cells, Ri, Ri[j], grid, series, exclude=j)


def _integrate(alpha, dex_grid):
  '''cumulative integral of alpha d ln R over a log10 grid -> ln of the ratio.'''
  L = dex_grid*np.log(10.)
  return np.concatenate([[0.], np.cumsum(0.5*(alpha[1:] + alpha[:-1])*np.diff(L))])


def table_closure(key=KEY, z=4, ncells=NCELLS, ks=None, verbose=True):
  '''
  Would a tabulated alpha(R_i/R_0, R/R_i) reproduce a cell's pre-rarefaction history?

  Leave-one-out: build the table from every other cell (_loo_predict), integrate the
  prediction and the truth into ln(Delta'), and propagate to rho and p:

      Delta'   integral of alpha_D d lnR
      rho      mass conservation, d ln rho = -d ln Delta'   (the R^2 carries no error)
      p        TM adiabat, d ln p = gma_ad * d ln rho

  Reported for both alpha_D (-> Delta', rho, p) and alpha_G (-> Gamma, and hence the observer
  time). This is the gate on any prescription built from the table.

  rho and p are each obtained by an INDEPENDENT route, so that the agreement is a real check
  and not a restatement: rho also via a leave-one-out table of alpha_rho measured directly
  (`d_rho`, to be compared with -d_D from mass conservation), and p by running
  _adiabat_integrated over the predicted and the true rho TRACKS rather than multiplying by a
  frozen gma_ad. The latter matters because gma_ad depends on T = p/rho along the way.

  TWO THINGS THE NUMBERS MEAN, both of which are easy to get backwards:

  The POINTWISE residual is ~100x the INTEGRATED one, because it is oscillatory rather than
  biased -- integration averages it out. Quote the integrated number as the table's accuracy
  and the pointwise one as its noise; neither is a substitute for the other.

  GAMMA IS AMPLIFIED x2 into the observable. Ton = t - R/c and (1-beta) ~ 1/(2 Gamma^2), so a
  fractional error in Gamma doubles into the observer time -- which is why alpha_G needs its
  OWN table rather than a Gamma = const closure, even though |alpha_G| <= 0.062. The two are
  reported side by side so the choice is priced rather than argued.

  Correctness check with teeth: d_rho (measured alpha_rho, tabulated) must equal -d_D (mass
  conservation applied to the tabulated alpha_D). Those are two different measurements routed
  through two different relations, so agreement is evidence and disagreement localises a bug.
  '''
  grid, cells, Ri = _cells_by_Ri(key, z, ncells, ks)
  rows = []
  for j, d in enumerate(cells):
    out = dict(k=d['k'], Ri=Ri[j], dex=d['dex'])
    for series, tag in (('a_D', 'D'), ('a_G', 'G'), ('a_rho', 'rho_dir')):
      pred = _loo_predict(cells, Ri, j, grid, series)
      m = (np.isfinite(pred) & np.isfinite(d[series])
           & (grid >= INJ_SAFE) & (grid <= d['dex']))
      if m.sum() < 5:
        out[f'n_{tag}'] = 0
        continue
      resid = pred[m] - d[series][m]
      dln = _integrate(pred[m], grid[m]) - _integrate(d[series][m], grid[m])
      out[f'n_{tag}'] = int(m.sum())
      out[f'rms_{tag}'] = float(np.sqrt(np.mean(resid**2)))
      out[f'max_{tag}'] = float(np.abs(resid).max())
      out[f'dln_{tag}'] = float(dln[-1])
      if tag == 'G':
        # Gamma = const, the alternative closure: its error IS the real drift in ln Gamma
        out['dln_G_const'] = float(_integrate(d[series][m], grid[m])[-1])
    if 'dln_D' in out:
      # rho from mass conservation on the tabulated alpha_D, and p by integrating the TM
      # adiabat over the two rho TRACKS (not a frozen-index multiply): ln(rho/rho_i) =
      # -2 ln(R/R_i) - integral alpha_D dlnR.
      pred = _loo_predict(cells, Ri, j, grid, 'a_D')
      m = (np.isfinite(pred) & np.isfinite(d['a_D'])
           & (grid >= INJ_SAFE) & (grid <= d['dex']))
      L = grid[m]*np.log(10.)
      geo = -2.*(L - L[0])
      lr_p = geo - (_integrate(pred[m], grid[m]) - _integrate(pred[m], grid[m])[0])
      lr_t = geo - (_integrate(d['a_D'][m], grid[m]) - _integrate(d['a_D'][m], grid[m])[0])
      rho_p, rho_t = d['rho_sh']*np.exp(lr_p), d['rho_sh']*np.exp(lr_t)
      pp = _adiabat_integrated(rho_p, d['rho_sh'], d['p_sh'])
      pt = _adiabat_integrated(rho_t, d['rho_sh'], d['p_sh'])
      out['gma_ad'] = float(derive_adiab_fromT_TM(d['Theta_sh']))
      out['dln_rho'] = float(lr_p[-1] - lr_t[-1])
      out['dln_p'] = float(np.log(pp[-1]/pt[-1]))
    rows.append(out)
  df = pd.DataFrame(rows)
  if verbose and len(df):
    ok = df.dropna(subset=['dln_D'])
    print(f'\ntable_closure {key} {SHELL_NAME.get(z, z)}: {len(ok)} of {len(df)} cells '
          f'predictable (the 2 R_i extremes cannot be interpolated)')
    print(f'  alpha_D pointwise LOO residual   rms median {ok.rms_D.median():.4f}   '
          f'worst cell {ok.rms_D.max():.4f}   worst point {ok.max_D.max():.4f}')
    print('  INTEGRATED error at the handover:')
    for col, lab in (('dln_D', "Delta'"), ('dln_rho', 'rho'), ('dln_p', 'p')):
      v = ok[col].abs()
      print(f'    {lab:7s} median {100*v.median():5.2f}%   worst {100*v.max():5.2f}%')
    okg = df.dropna(subset=['dln_G'])
    if len(okg):
      print('  GAMMA, and its x2 amplification into the observer time:')
      for col, lab in (('dln_G', 'alpha_G table'), ('dln_G_const', 'Gamma = const')):
        v = okg[col].abs()
        print(f'    {lab:14s} ln(Gamma) err median {100*v.median():6.3f}%  '
              f'worst {100*v.max():6.3f}%   -> Ton median {200*v.median():5.2f}%  '
              f'worst {200*v.max():5.2f}%')
    # independent-route check: alpha_rho tabulated directly vs mass conservation on alpha_D
    ind = ok.dropna(subset=['dln_rho_dir'])
    if len(ind):
      dev = (ind.dln_rho_dir - ind.dln_rho).abs()
      print(f'  INDEPENDENT ROUTE: tabulated alpha_rho vs mass conservation on alpha_D -- '
            f'differ by median {100*dev.median():.3f}%, worst {100*dev.max():.3f}%')
    eff = (ok.dln_p/ok.dln_rho).median()
    print(f'  implied index of the p route: {eff:.3f}  '
          f'(gma_ad at the shocked state = {ok.gma_ad.median():.3f})')
  return df


def entropy_anchor(key=KEY, z=4, ncells=NCELLS, ks=None, verbose=True):
  '''
  The gap between a PREDICTIVE model and the adiabat that works.

  adiabat_check shows p(rho) tracking the TM adiabat to ~1% when anchored on the first
  MEASURED row, and failing by 8-60x when anchored on the prepended shockfit row. A
  predictive model has only the shock-jump state -- i.e. only the anchor that fails. So the
  question is whether the offset between them is a smooth function of R_i, in which case one
  extra 1-D table absorbs it and the closure survives.

  Reports, per cell:
    K       ln(p_measured / p_adiabat-from-jump) at the first measured row -- the offset
    smooth  leave-one-out linear interpolation residual of K in R_i
    drift   what is LEFT of the adiabat mismatch once a constant K is removed, i.e. does the
            adiabat then actually track across the window

  The offset is POSITIVE: the gas gains entropy between the jump state and the first measured
  row. Whether that is physical post-shock settling or an inaccuracy of the shockfit state is
  NOT settled here, and it matters -- a numerical offset would not transfer to a run at a
  different resolution, so this is one of the things the resolution probe should look at.
  '''
  ks = profile_cells(key, z, ncells) if ks is None else np.asarray(ks)
  rows = []
  for k in ks:
    s = _history(key, int(k), z=z)
    if s is None or len(s) < 3:
      continue
    h, dex = prerar_window(s)
    if declined(_theta_sh(s), dex, key) is not None:
      continue
    s_raw = _history(key, int(k), z=z, early_ana=None)
    n_pre = len(s) - len(s_raw) if s_raw is not None else 0
    if n_pre < 1 or n_pre >= h:
      continue                     # no prepended row => no jump-vs-measured distinction
    x, dx, rho, p, lfac, t = (a[:h+1] for a in _cols(s))
    pm = _adiabat_integrated(rho[n_pre:], rho[0], p[0])     # adiabat from the JUMP state
    rows.append(dict(k=int(k), Ri=float(x[0]*c_/_env(key).R0), dex=dex,
                     K=float(np.log(p[n_pre]/pm[0])),
                     drift=float(np.log(pm[-1]/p[-1]) - np.log(pm[0]/p[n_pre]))))
  df = pd.DataFrame(rows).sort_values('Ri').reset_index(drop=True)
  if verbose and len(df):
    res = np.array([abs(np.interp(df.Ri[j], df.drop(j).Ri, df.drop(j).K) - df.K[j])
                    for j in range(1, len(df) - 1)])
    print(f'\nentropy_anchor {key} {SHELL_NAME.get(z, z)}: {len(df)} cells')
    print(f'  offset ln(p_meas/p_adiab-from-jump)  {df.K.min():+.4f} .. {df.K.max():+.4f}'
          f'   (up to {100*np.expm1(df.K.abs().max()):.0f}% in p)')
    print(f'  smooth in R_i?  LOO interp residual median {np.median(res):.5f} '
          f'({100*np.expm1(np.median(res)):.2f}% in p)   worst {res.max():.5f} '
          f'({100*np.expm1(res.max()):.1f}%)')
    print(f'  residual drift once a constant offset is removed: median '
          f'{df.drift.abs().median():.4f}   worst {df.drift.abs().max():.4f} in ln p')
    if res.max() > 0.05:
      print(f'  *** {int((res > 0.05).sum())} cell(s) miss by >5% in p: the offset needs '
            'DENSER R_i sampling than this cell stride, not a smoother fit')
  return df


# ---------------------------------------------------------------------------
# S4: comparing two runs (resolution convergence, and does the table transfer?)
# ---------------------------------------------------------------------------
def compare_runs(key_a, key_b, z=4, ncells=NCELLS, tol=0.02, verbose=True):
  '''
  alpha_D of two runs compared at fixed (R_i/R_0, R/R_i). Two uses, same machinery:

    resolution convergence   key_a/key_b same shell width, different Nsh1. Does alpha_D
                             depend on resolution? If it does, the fiducial's own alpha_D is
                             not converged and the table inherits that.
    does the table transfer  key_a = fiducial, key_b = a wider-shell run. Over the COMMON
                             (R_i/R_0, R/R_i) domain the two must agree, or a wider shell is
                             a different problem and a table measured on it does not apply.

  CELLS ARE PAIRED BY POSITION (R_i/R_0), NEVER BY INDEX. Two runs with different Nsh1 have
  different index->position maps, and two runs with different shell width have different
  index->R_i maps even at equal Nsh1; pairing on k would compare unrelated cells. This is the
  same trap boundary_comparison.paired_cells exists for.

  Comparison is made by interpolating run B's cells in R_i onto run A's R_i at each grid
  point, over the overlap of both runs' R_i ranges and both cells' covered R/R_i.
  Returns (per-grid-point DataFrame, summary dict).
  '''
  grid, ca, Ria = _cells_by_Ri(key_a, z, ncells)
  _, cb, Rib = _cells_by_Ri(key_b, z, ncells)
  lo, hi = max(Ria.min(), Rib.min()), min(Ria.max(), Rib.max())
  rows = []
  for j, d in enumerate(ca):
    if not (lo <= Ria[j] <= hi):
      continue
    for g in range(grid.size):
      if grid[g] < INJ_SAFE or grid[g] > d['dex'] or not np.isfinite(d['a_D'][g]):
        continue
      xs = [(Rib[i], cb[i]['a_D'][g]) for i in range(len(cb))
            if np.isfinite(cb[i]['a_D'][g])]
      if len(xs) < 3:
        continue
      xv = np.array([p[0] for p in xs])
      yv = np.array([p[1] for p in xs])
      if not (xv.min() <= Ria[j] <= xv.max()):
        continue
      o = np.argsort(xv)
      rows.append(dict(k_a=d['k'], Ri=Ria[j], dex_grid=grid[g],
                       a_A=d['a_D'][g], a_B=float(np.interp(Ria[j], xv[o], yv[o]))))
  df = pd.DataFrame(rows)
  out = {}
  if len(df):
    df['diff'] = df.a_B - df.a_A
    ad = df['diff'].abs()
    out = dict(n=len(df), n_cells=int(df.k_a.nunique()),
               Ri_overlap=(float(lo), float(hi)),
               med=float(ad.median()), q84=float(np.percentile(ad, 84)),
               worst=float(ad.max()), bias=float(df['diff'].median()),
               passed=bool(np.percentile(ad, 84) <= tol))
  if verbose:
    print(f'\ncompare_runs {SHELL_NAME.get(z, z)}: {key_a}  vs  {key_b}')
    if not out:
      print('  no overlapping (R_i, R/R_i) samples -- nothing to compare')
      return df, out
    print(f'  overlap R_i/R_0 in [{lo:.3f}, {hi:.3f}]: {out["n_cells"]} cells of A, '
          f'{out["n"]} grid samples')
    print(f'  |alpha_D(B) - alpha_D(A)|   median {out["med"]:.4f}   84th {out["q84"]:.4f}   '
          f'worst {out["worst"]:.4f}')
    print(f'  signed bias (B - A)         {out["bias"]:+.4f}   '
          '<- a systematic offset, not scatter, if it dominates')
    print(f'  tol = {tol} (the fiducial RS cross-cell spread is 0.045, FS 0.008)  -> '
          f'{"PASS" if out["passed"] else "FAIL"} on the 84th percentile')
    # the asymptote is the number a prescription would carry, so compare it directly too
    aa = asymptote(key_a, (z,), ncells, verbose=False)[z]
    bb = asymptote(key_b, (z,), ncells, verbose=False)[z]
    if np.isfinite(aa['a_D']) and np.isfinite(bb['a_D']):
      print(f'  asymptote: A {aa["a_D"]:+.4f} ({aa["n"]} cells)   '
            f'B {bb["a_D"]:+.4f} ({bb["n"]} cells)   '
            f'difference {abs(aa["a_D"] - bb["a_D"]):.4f}')
    else:
      print(f'  asymptote: A {aa["n"]} cells, B {bb["n"]} cells reach '
            f'{ASYM_TO} dex -- not both measurable')
  return df, out


def window_law(key=KEY, z=4, ncells=NCELLS, verbose=True):
  '''
  The rarefaction-window geometry, i.e. the law the wider-shell sizing rests on:

      R_cross/R_0                     where the shock exhausts the shell
      (R_h - R_i)/R_0 = a*phi + c     phi = remaining shell depth (1 at the CD, 0 last-shocked)

  Fitted on the fiducial this gives a = 1.847, c = 0.405, R_cross/R_0 = 2.235 (corr 0.998)
  and predicts the observed 30% asymptote yield to within a point. Both a and R_cross-1 should
  scale LINEARLY with shell width, which is the extrapolation a wider run tests -- run this on
  the wide run and check that they do.
  '''
  grid, cells, Ri = _cells_by_Ri(key, z, ncells)
  Rh = np.array([d['R_h_ratio'] for d in cells])*Ri
  Rcross = Ri.max()
  phi = (Rcross - Ri)/(Rcross - 1.) if Rcross > 1. else np.zeros_like(Ri)
  win = Rh - Ri
  a, c = np.polyfit(phi, win, 1)
  f = 10.**ASYM_TO - 1.
  yield_meas = float(np.mean(Rh/Ri >= 1. + f))
  out = dict(R_cross=float(Rcross), b=float(Rcross - 1.), a=float(a), c=float(c),
             corr=float(np.corrcoef(phi, win)[0, 1]), yield_meas=yield_meas,
             n=len(Ri), dex_max=float(np.log10((Rh/Ri).max())))
  if verbose:
    print(f'\nwindow_law {key} {SHELL_NAME.get(z, z)}: {out["n"]} cells')
    print(f'  R_cross/R_0 = {out["R_cross"]:.3f}   (b = {out["b"]:.3f})')
    print(f'  (R_h - R_i)/R_0 = {a:.3f}*phi {c:+.3f}   corr = {out["corr"]:.4f}')
    print(f'  deepest cell window = {out["dex_max"]:.3f} dex;'
          f'  asymptote yield = {100*yield_meas:.0f}%')
  return out


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def _style(ax):
  ax.grid(True, which='major', color='0.92', lw=0.6, zorder=0)
  ax.set_axisbelow(True)
  for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)


def plot_alpha_collapse(key=KEY, z_list=Z_LIST, ncells=NCELLS, outdir=OUTDIR):
  '''alpha_D(R/R_inj), one line per cell coloured by k, one panel per shell, with the
  pooled median, the plateau band and the gate drawn.'''
  os.makedirs(outdir, exist_ok=True)
  grid = _grid()
  fig, axes = plt.subplots(1, len(z_list), figsize=(11, 4.4), sharey=True,
                           layout='constrained')
  axes = np.atleast_1d(axes)
  for ax, z in zip(axes, z_list):
    _, cells = _collect(key, z, ncells)
    ok = [d for d in cells.values() if d['measurable']]
    if not ok:
      continue
    kk = np.array([d['k'] for d in ok])
    cmap, norm = plt.cm.viridis, plt.Normalize(kk.min(), kk.max())
    for d in ok:
      m = np.isfinite(d['a_D']) & (grid <= d['dex']) & (grid >= INJ_SAFE)
      ax.plot(grid[m], d['a_D'][m], color=cmap(norm(d['k'])), lw=0.8, alpha=0.75)
    A = np.array([d['a_D'] for d in ok])
    med, mad, iqr, n, _ = _robust(A)
    m = (n >= MIN_OVERLAP) & (grid >= INJ_SAFE)
    ax.plot(grid[m], med[m], color=COL_INK, lw=2.2, zorder=5, label='pooled median')
    asy = asymptote(key, (z,), ncells, verbose=False)[z]
    if np.isfinite(asy['a_D']):
      v, sd = asy['a_D'], asy['spread']
      ax.axhspan(v - sd, v + sd, color=COL[z], alpha=0.18, zorder=1,
                 label=rf'asymptote ${v:+.3f}\pm{sd:.3f}$')
      ax.axhline(v, color=COL[z], lw=1.1, ls='--', zorder=4)
    # the range the asymptote is measured over, and the prepend-contaminated head
    ax.axvspan(ASYM_FROM, ASYM_TO, color=COL_MUTED, alpha=0.12, zorder=1)
    ax.axvspan(0., INJ_SAFE, color='0.55', alpha=0.35, zorder=6)
    ax.set_xlabel(ABS_LABEL['R_Rinj'])
    ax.set_title(f'{SHELL_NAME.get(z, z)}  '
                 rf'($\Theta_{{\rm sh}}$ {min(d["Theta_sh"] for d in ok):.3f}'
                 rf'--{max(d["Theta_sh"] for d in ok):.3f})', fontsize=10)
    ax.set_xlim(0., GRID_MAX)
    ax.legend(fontsize=8, frameon=False, loc='lower right')
    _style(ax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, ax=ax, label='cell index $k$', pad=0.02)
  axes[0].set_ylabel(r"$\alpha_\Delta = d\ln\Delta'/d\ln R$")
  axes[0].set_ylim(-1.05, -0.55)
  fig.suptitle(f'{key}: pre-rarefaction proper expansion  '
               '(cells converge on a common asymptote)', fontsize=11)
  path = os.path.join(outdir, 'alpha_delta_collapse.png')
  fig.savefig(path, dpi=160, bbox_inches='tight')
  plt.close(fig)
  print(f'saved {path}')
  return path


def plot_abscissa_comparison(key=KEY, z=4, ncells=NCELLS, outdir=OUTDIR):
  '''The same alpha_D pooled on each candidate abscissa: which one collapses.'''
  os.makedirs(outdir, exist_ok=True)
  df, stacks = collapse_test(key, z, ncells, verbose=False)
  fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True, layout='constrained')
  for ax, nm in zip(axes.ravel(), ABSCISSAE):
    xg, Y, med, iqr, n = stacks[nm]
    for row in Y:
      ax.plot(xg, row, color=COL[z], lw=0.7, alpha=0.35)
    m = n >= MIN_OVERLAP
    ax.plot(xg[m], med[m], color=COL_INK, lw=2.)
    ax.fill_between(xg[m], (med - iqr)[m], (med + iqr)[m], color=COL_INK, alpha=0.18)
    rel = float(df.loc[df.abscissa == nm, 'rel_spread'].iloc[0])
    ax.set_title(f'{ABS_LABEL[nm]}   rel spread {rel:.3f}', fontsize=9)
    ax.set_xlabel(ABS_LABEL[nm], fontsize=8)
    _style(ax)
  for ax in axes[:, 0]:
    ax.set_ylabel(r'$\alpha_\Delta$')
  axes[0, 0].set_ylim(-1.15, -0.45)
  fig.suptitle(f'{key} {SHELL_NAME.get(z, z)}: what is $\\alpha_\\Delta$ a function of?',
               fontsize=11)
  path = os.path.join(outdir, f'alpha_abscissae_z{z}.png')
  fig.savefig(path, dpi=160, bbox_inches='tight')
  plt.close(fig)
  print(f'saved {path}')
  return path


def plot_convergence(keys=('cooling_g100', 'cooling_g100_w2', 'cooling_g100_w5'),
    z=4, ncells=120, outdir=OUTDIR, lo=0.55, hi=0.65):
  '''
  The result a wider-shell run exists to produce, in two panels:

    left   alpha_D vs log10(R/R_inj), one line per cell of the WIDEST run, coloured by R_i,
           with each run's maximum reach marked. Shows directly that the fiducial stops
           before alpha_D flattens, which is why its asymptote() came out ~0.03 shallow.
    right  the CONVERGED alpha_D against R_i, per run -- i.e. how far across the shell the
           converged value can actually be measured, and its R_i dependence.

  Per-cell curves, never the pooled median: pooled and per-cell trends have opposite signs
  (see the module docstring), so a pooled curve here would show the reverse of the truth.
  '''
  os.makedirs(outdir, exist_ok=True)
  grid = _grid()
  avail = []
  for k in keys:
    try:
      _, cells, Ri = _cells_by_Ri(k, z, ncells)
      if len(cells):
        avail.append((k, cells, Ri))
    except Exception:
      continue
  if not avail:
    print('plot_convergence: no runs available')
    return None
  fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')

  # --- left: per-cell curves of the widest run
  kw, cw, Riw = max(avail, key=lambda a: max(d['dex'] for d in a[1]))
  cmap = plt.cm.viridis
  norm = plt.Normalize(Riw.min(), Riw.max())
  for d, ri in zip(cw, Riw):
    m = (np.isfinite(d['a_D']) & (grid >= INJ_SAFE) & (grid <= d['dex'] - EDGE_EXCL))
    if m.sum() > 3:
      axes[0].plot(grid[m], d['a_D'][m], color=cmap(norm(ri)), lw=0.7, alpha=0.7)
  for k, cells, _ in avail:
    dm = max(d['dex'] for d in cells)
    axes[0].axvline(dm, color=COL_MUTED, lw=1., ls=':')
    _m = re.search(r'_w(\d+)', k)
    axes[0].annotate(f"W={_m.group(1)}" if _m else 'fiducial',
                     xy=(dm, 0.02), xycoords=('data', 'axes fraction'), rotation=90,
                     fontsize=7, color=COL_MUTED, ha='right', va='bottom')
  axes[0].axvspan(lo, hi, color=COL[z], alpha=0.15)
  axes[0].set(xlabel=ABS_LABEL['R_Rinj'], ylabel=r"$\alpha_\Delta$", ylim=(-0.95, -0.55))
  axes[0].set_title(f'{kw}: per-cell (shaded = where the converged value is read)',
                    fontsize=9)
  fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=axes[0],
               label=r'$R_i/R_0$', pad=0.02)

  # --- right: converged alpha_D vs R_i, per run
  for k, _, _ in avail:
    df = converged_alpha(k, z, ncells, lo=lo, hi=hi, verbose=False)
    m = re.search(r'_w(\d+)', k)
    lab = f'W={m.group(1)}' if m else 'fiducial (W=1)'
    if not len(df):
      print(f'  {lab}: no cells reach {hi + EDGE_EXCL:.2f} dex, not plotted')
      continue
    axes[1].plot(df.Ri, df.a_D, 'o-', ms=3.5, lw=1.2, label=f'{lab}  (n={len(df)})')
  axes[1].axvline(2.235, color=COL_MUTED, lw=1., ls='--')
  axes[1].annotate("fiducial's $R_i$ range ends", xy=(2.235, 0.5),
                   xycoords=('data', 'axes fraction'), rotation=90, fontsize=7,
                   color=COL_MUTED, ha='right', va='center')
  axes[1].set(xlabel=r'$R_i/R_0$', ylabel=r'converged $\alpha_\Delta$')
  axes[1].set_title(f'converged over [{lo}, {hi}] dex -- the table\'s asymptotic column',
                    fontsize=9)
  axes[1].legend(fontsize=8, frameon=False)
  for ax in axes:
    _style(ax)
  fig.suptitle(f'Pre-rarefaction $\\alpha_\\Delta$: convergence and its $R_i$ dependence '
               f'({SHELL_NAME.get(z, z)})', fontsize=11)
  path = os.path.join(outdir, f'alpha_convergence_z{z}.png')
  fig.savefig(path, dpi=160, bbox_inches='tight')
  plt.close(fig)
  print(f'saved {path}')
  return path


def plot_closures(key=KEY, z_list=Z_LIST, ncells=NCELLS, outdir=OUTDIR):
  '''The two closures across the shell: the adiabat on both anchors, and h*Gamma / alpha_G.'''
  os.makedirs(outdir, exist_ok=True)
  fig, axes = plt.subplots(2, 2, figsize=(10.5, 7), layout='constrained')
  for z in z_list:
    ad, be = adiabat_check(key, z, ncells, verbose=False), \
        bernoulli_check(key, z, ncells, verbose=False)
    c, lab = COL[z], SHELL_NAME.get(z, z)
    axes[0, 0].plot(ad.k, ad.res_meas, 'o-', color=c, ms=3, lw=1., label=f'{lab} measured')
    axes[0, 0].plot(ad.k, ad.res_inj, 's--', color=c, ms=3, lw=1., alpha=0.5,
                    label=f'{lab} prepended')
    axes[0, 1].plot(ad.k, ad.a_p_a_rho, 'o-', color=c, ms=3, lw=1., label=lab)
    axes[0, 1].plot(ad.k, ad.gma_ad, ':', color=c, lw=1.4)
    axes[1, 0].plot(be.k, be.bern_max, 'o-', color=c, ms=3, lw=1., label=lab)
    axes[1, 1].plot(be.k, be.a_G, 'o-', color=c, ms=3, lw=1., label=lab)
  axes[0, 0].set(ylabel=r'median $|\Delta\ln p|$ vs TM adiabat', yscale='log')
  axes[0, 0].set_title('adiabat: anchor matters, the EoS does not fail', fontsize=9)
  axes[0, 1].set(ylabel=r'$a_p/a_\rho$')
  axes[0, 1].set_title(r'measured index vs $\hat\gamma_{\rm TM}(\Theta)$ (dotted)',
                       fontsize=9)
  axes[1, 0].axhspan(0.007, 0.013, color=COL_MUTED, alpha=0.25)
  axes[1, 0].set(ylabel=r'max $|\Delta\ln(h\Gamma)|$')
  axes[1, 0].set_title('Bernoulli fails here (band = post-handover coasting)', fontsize=9)
  axes[1, 1].axhline(0., color=COL_MUTED, lw=0.8)
  axes[1, 1].set(ylabel=r'$\alpha_\Gamma$')
  axes[1, 1].set_title(r'$\Gamma$ is constant to a few percent', fontsize=9)
  for ax in axes.ravel():
    ax.set_xlabel('cell index $k$')
    ax.legend(fontsize=7, frameon=False)
    _style(ax)
  fig.suptitle(f'{key}: closures for a pre-rarefaction prescription', fontsize=11)
  path = os.path.join(outdir, 'prerar_closures.png')
  fig.savefig(path, dpi=160, bbox_inches='tight')
  plt.close(fig)
  print(f'saved {path}')
  return path


# ---------------------------------------------------------------------------
def main(key=KEY, z_list=Z_LIST, ncells=NCELLS, outdir=OUTDIR):
  '''Every table and figure, both shells.'''
  out = {}
  for z in z_list:
    out[z] = dict(alpha=alpha_table(key, z, ncells),
                  mass=mass_conservation_audit(key, z, ncells),
                  collapse=collapse_test(key, z, ncells)[0],
                  window=window_convergence(key, z, ncells),
                  adiabat=adiabat_check(key, z, ncells),
                  bernoulli=bernoulli_check(key, z, ncells),
                  table=table_closure(key, z, ncells),
                  anchor=entropy_anchor(key, z, ncells))
  # the two cross-shell results, which need both shells and so cannot live in the loop
  out['asymptote'] = asymptote(key, z_list, ncells)
  out['shockstate'] = shockstate_test(key, z_list, ncells)
  plot_alpha_collapse(key, z_list, ncells, outdir)
  plot_closures(key, z_list, ncells, outdir)
  for z in z_list:
    plot_abscissa_comparison(key, z, ncells, outdir)
  return out


if __name__ == '__main__':
  main()
