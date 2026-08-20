# -*- coding: utf-8 -*-
# @Author: acharlet

'''
Reconstruct a shocked cell's hydro from just after the shock out to large radius, AS IF THE
RAREFACTION NEVER ARRIVED.

This is the pre-handover mirror of working_cooling_data.norar_history, which starts at the
handover and can there assume Gamma = const and dr = const. Before the handover neither holds
(the cell is in the shock-crossing transient), so this closes the state on the measured
expansion law instead:

    settle jump -> g0        three 1-D tables in R_i          measured, not derivable
    Delta'(R)   = Delta'(g0) * exp(int alpha_D dlnR)          2-D table -> 2/gma - 2
    rho(R)                     mass conservation rho R^2 Delta' = const        EXACT
    p(R)        = p(g0) * exp(int alpha_p dlnR)               2-D table -> -2  (NOT the adiabat)
    Gamma(R)    = Gamma(g0) * exp(int alpha_G dlnR)           2-D table
    dx, vx                     Delta'/Gamma, beta(Gamma)                       EXACT
    t(R)        = t + (x-x0) + int (1-beta)/beta dx                            EXACT

DESIGN PRINCIPLE, which settles every trade-off below: this is a PHYSICS-BASED semi-analytic
model that uses the NUMERICS for the initial evolution and CONVERGES TO THE PHYSICS. A small,
quantified departure from the simulation is acceptable where the physics is unambiguous; the
reverse -- reproducing the numerics at the cost of an unphysical asymptote -- is not.

  numerics govern    the settling stretch (settle tables) and the shock-crossing transient
                     (the 2-D alpha tables), i.e. everything the physics cannot derive
  physics governs    the asymptote (ALPHA_INF_DERIVED): alpha_D = 2/gma - 2 = -0.800,
                     alpha_p = -2, alpha_G = 0, with gma = 5/3

  THE ACCEPTED DEVIATION, measured so it is never a surprise: the converged alpha_D measured on
  the widest run is -0.790, against the derived -0.800. The gap of 0.010 is exactly
  2*(1/1.653 - 3/5), i.e. the gas has NOT fully cooled over the rarefaction-free range -- Theta
  falls 0.061 -> 0.010 across the longest window and gma_ad(TM) climbs 1.638 -> 1.662, short of
  5/3. So the fluid's effective index there is 1.653, not 5/3.
  The price is bounded and applies ONLY past the table: preferring 5/3 over the measured 1.653
  costs 2.3% per decade in Delta' and rho where the derived value is in force, 5.9% over the
  2.5 decades norar_history spans. Accepted deliberately -- 5/3 is where the fluid is going,
  -0.790 is only where it happens to be inside a window the rarefaction cuts short.
  It does NOT explain any ramp in rho R^1.2 INSIDE the table: that was tested by swapping the
  asymptote over -0.800/-0.795/-0.789/-0.785 and the tracks came back bit-identical, which
  falsifies the attribution. The ramp that used to be there came from the table's own alpha_D
  sitting 0.0076 below the cell's, cured by TABLE_EDGE (see build_table). Past the table
  rho R^1.2 and p R^2 now hold flat to machine precision -- verify with plot_article, not by
  assumption, since two separate bugs have hidden behind a plausible-looking slope here.

THE ASYMPTOTE IS DERIVED, NOT FITTED, and the derivation is the reason the whole thing is well
behaved -- see ALPHA_INF_DERIVED. In brief: the shocked layer is in CAUSAL CONTACT, so pressure
equilibrates across it and p ~ R^-2 is carried from the shock to the CD; the CD fluid is
adiabatic, so rho ~ R^(-2/gma); mass conservation then gives alpha_D = 2/gma - 2 = -0.800 with
gma = 5/3, the FULLY COOLED limit (see GMA_REF). Measured: -0.797 (RS) / -0.793 (FS). The
cell is NOT freely coasting -- it is being compressed toward the CD by the layer it sits in --
which is why alpha_D plateaus near -0.79 rather than relaxing to 0, and why rho ~ R^-1.2 rather
than R^-2 is the CORRECT behaviour here.

rho ~ R^-2 IS NOT AN ASYMPTOTE THIS MODEL EVER RELAXES TO, at any radius. That law describes a
fluid element evolving WITHOUT INTERACTION; these cells sit inside a shocked shell and interact
with it for as long as they are shocked, which is the whole domain of this counterfactual. So
rho ~ R^-1.2 and p ~ R^-2 are the correct scalings THROUGHOUT the prolongation -- there is no
handover radius past which the law is meant to change, and none should be built. Free coasting
becomes right only once the rarefaction has passed and the layer has decompressed, i.e. in the
history norar_history describes -- the case this model exists to counterfactual AWAY.

GAMMA: TABLE vs THE THIN-SHELL MODEL'S SMOOTH BROKEN POWER LAW. gamma_law='bpl' fits each cell's
Gamma with get_fitting_smoothBPL_new(beta=0) -- i.e. smooth_bpl0_apy, the same form
get_hydrofits_shell_new uses -- and interpolates its 4 parameters in R_i. beta=0 forces the
second segment flat, so Gamma -> const, the coasting asymptote, is built in. Measured
(median/84th % of |dln Gamma|):

    closure                fiducial        W=5 LOO       -> Ton
    alpha_G table        0.046 / 0.086   0.046 / 0.178   0.09% / 0.09%    <- default
    smooth BPL (beta=0)  0.060 / 0.113   0.110 / 0.208   0.12% / 0.22%
    Gamma = const        1.109 / 1.944   2.454 / 4.380   2.22% / 4.91%

  The BPL is a legitimate closure -- 18-22x better than Gamma = const, and both it and the table
  are far inside the gate. It also has two structural advantages: 4 numbers per cell against 241
  grid points, and a guaranteed asymptote, so it needs none of the MIN_TABLE_CELLS /
  ALPHA_BLEND_DEX machinery the tables need where their coverage collapses.
  It is not the default only because it is 1.3x worse on the fiducial and 2.4x worse at W=5 LOO.
  The fitted parameters are otherwise well behaved: x_b spans 1.07-2.18 with corr(R_i) = -0.83
  and alpha spans -0.016..+0.343 with corr(R_i) = +0.95, both smoothly interpolable. The weak
  link is the SMOOTHING parameter s, which spans its whole allowed range 0.05-3.0 with
  corr(R_i) = +0.07 -- so s is what does not interpolate, and that is why the BPL loses more at
  leave-one-out (where interpolating in R_i is the whole test) than on the fiducial.

  TWO THINGS TO GET RIGHT WHEN FITTING HERE, both of which I got wrong first time:
    * PASS cleanData=False. get_fitting_smoothBPL_new's cleanData path runs find_fitting_
      boundaries, which is meant to strip the settling artifact -- but the data handed to it here
      has ALREADY been trimmed at INJ_SAFE / EDGE_EXCL, so it double-trims and misfires for the
      same reason settle_end needed guards. With cleanData=True only 65 of 112 cells fit at all,
      and alpha comes back scattered (range -1.28..+0.39, corr(R_i) = -0.14), which looks exactly
      like fit degeneracy and is not: it is the trimming.
    * SAVGOL PRE-SMOOTHING MAKES NO DIFFERENCE (residual 0.1562% raw vs 0.1569% smoothed, and
      identical parameter interpolability). The thin-shell path smooths because it fits few, noisy
      shock-front samples; here curve_fit already least-squares over several hundred points per
      cell, which averages the same dump-cadence jitter. Do not add it expecting a gain.

WHY p IS TABULATED AND NOT CLOSED BY THE EoS. 'integrated' (_adiabat_integrated) already does
what the EoS can do: it recomputes gma_ad from the CURRENT (rho, p) at every RK2 step, so the
adiabatic index does evolve as the gas cools -- that is not the limitation. The limitation is
that the gas is not adiabatic at all. Its measured index is a_p/a_rho = 1.625, BELOW gma_ad =
1.641 at the anchor, and gma_ad only RISES toward 5/3 as T falls, i.e. away from the data. So no
TM adiabat, frozen or integrated, can match this phase: p error 0.11%/0.37% tabulated against
0.95%/2.65% integrated and 0.80%/1.34% frozen. (Frozen beating integrated is an ACCIDENTAL
CANCELLATION, the same one on record for the post-handover law in the norar-counterfactual note,
where it cancelled a dr = const error. It is not a reason to prefer frozen.)

  adiabat='temp' TABULATES THE RELATIVISTIC TEMPERATURE T = p/rho INSTEAD, and gives NUMERICALLY
  IDENTICAL results (0.11%/0.37%) -- as it must: rho is exact from mass conservation, so
  alpha_T = alpha_p - alpha_rho and the two are algebraically the same closure. Prefer 'temp' if
  anything downstream needs gma_ad(T) or c_s(T) consistently with the reconstructed state; the
  numbers will not change.

  The price of either: p is no longer tied to rho by an equation of state, so the implied entropy
  drifts by design. That is correct -- the gas really does gain entropy -- but the reconstruction
  is not thermodynamically closed. 'integrated' remains available if that matters more than a
  factor 7 in p.

rho comes from mass conservation and NEVER from its own table, so only Delta' carries the
integration error; and alpha_D needs no reference to Gamma, which is the whole reason to key
the model on Delta' = Gamma*dr rather than on dr. predict_history returns the same dict
contract as _norar_rows ({rho, p, lfac, vx, dx, t, x}) so it can drop into the same call sites.

WHY THERE IS A SETTLING TABLE AT ALL. A predictive model has only the shock-JUMP state
(load_shockfront_states, or fits_from_au(a_u) with no run at all -- this is where a_u enters).
That state is not the settled post-shock state: the prepended row sits at a smaller radius and
a much higher pressure than the first measured row, and prerar_cell_evolution.INJ_SAFE exists
because a slope window straddling the two measures the discontinuity rather than the flow.
The first 0.015 dex is therefore not modelled but TABULATED, as log-ratios in R_i:

    measured (cooling_g100_w5, jump row -> 0.015 dex, median / worst / LOO-interp)
      Delta'   -0.007 / +0.237 (27%) / 0.009      <- the accuracy floor, see CAVEATS
      p        -0.109 / -0.179 (20%) / 0.005
      Gamma    +0.002 / +0.007 ( 1%) / 0.001      <- tiny on W=5, NOT on other runs: see below
    rho follows from mass conservation and is not tabulated.

CAVEATS

  VALIDATED ONLY OUT TO EACH CELL'S HANDOVER -- 1.069 dex (11.7x in radius) for W=5's deepest
  cells. Past that no run has rarefaction-free data, so the extension is untested BY
  CONSTRUCTION, not by omission; that regime is what norar_history's law covers. "Large radius"
  here means as large as any simulation can witness without the wave.

  THE SETTLING TABLE'S RESOLUTION-INDEPENDENCE IS UNTESTED. prerar_cell_evolution's probe
  established that alpha_D is resolution-converged (bias <= 0.0008 across 2x), and said nothing
  about this correction, which is the more likely of the two to be a numerical shock-structure
  artifact. The cooling_g100_w2 / _w2_lores pair is on disk and can test it directly. Since the
  correction is worth 18x on Delta' and the run-to-run transfer already failed once (Gamma,
  above), treat this as the main open risk.

  The widest windows' last ~0.05 dex is noisy (one bin to ~13%) on the handful of cells that
  reach it, even after EDGE_EXCL. Do not read the extreme tail as a model failure.

THE OBJECTIVE, and what it fixes: the model must FOLLOW the cell where the simulation exists
and PROLONG it after. So where data exists the settling stretch is TAKEN, not predicted --
validate_model and plot_article default to anchor='measured', starting the reconstruction from
the cell's own post-settling state. Measured effect (median |d ln|, %):

    anchor      fiducial D / p / G       W=2 D / p / G
    jump        0.126 / 0.126 / 0.037    0.080 / 0.105 / 0.029
    measured    0.091 / 0.141 / 0.035    0.073 / 0.115 / 0.028

  Delta' improves 28% and the anchor residual becomes exactly 0 by construction, which also
  removes the settling table's interpolation error from the reconstruction path. p gets 12%
  WORSE, and that is the honest direction: with the jump anchor the settling correction was
  partly CANCELLING the alpha_p integration error. Same accidental-cancellation pattern as
  frozen gma_ad beating integrated -- do not read it as the jump anchor being better.
  anchor='jump' remains for the case with NO simulation at all (the jump state from
  fits_from_au), which is the whole point of a predictive model.

  IT DOES NOT change the ramps (unchanged to three decimals): the settling anchor was already
  good to 0.014%, so all remaining error is the alpha integration, not the anchor.

RESULTS (table from cooling_g100_w5, RS; median / 84th percentile of |d ln|, in %)

                              Delta' = rho        p          Gamma      Ton (=2 dlnG)
    W=5, leave-one-out         0.37 / 2.30   0.40 / 2.56   0.04 / 0.18   0.08 / 0.36
    fiducial, out-of-sample    0.13 / 0.26   0.15 / 0.35   0.04 / 0.08   0.08 / 0.16
    W=2, out-of-sample         0.09 / 0.22   0.12 / 0.34   0.03 / 0.07   0.06 / 0.14

  So the reconstruction is SUB-PERCENT on every quantity out-of-sample, and 0.4% at
  leave-one-out over the full 1.069 dex -- comfortably inside the few-percent gate. Error is
  NOT monotonic in radius: it peaks around 0.25-0.50 dex (mid-transient, where alpha_D varies
  fastest) and falls to 0.16-0.24% past 0.5 dex once the cells have converged. The last 0.05
  dex of the widest windows is noisier (up to ~13% in one bin) on the few cells that reach it.

  The out-of-sample tests beat leave-one-out, which is not a paradox: LOO removes a cell from a
  table whose R_i spacing widens toward R_i ~ 8, and it spans the full 1.069 dex, whereas the
  fiducial and W=2 only probe R_i <= 2.2 / 3.8 where the table is dense.

  THE INTERPOLATION ARTIFACT WAS AN UNDER-SAMPLING PROBLEM, AND MORE POINTS FIXED IT. alpha_p
  used to step by ~0.008 in ONE 0.005-dex grid interval near 0.70 dex, where the table's cell
  count fell 29 -> 24 (np.interp uses only the two BRACKETING cells, so a dropout jumps
  discretely). Only ~0.02% in p, invisible in every residual statistic, and visible ONLY as a
  kink on the near-flat p R^2 against its horizontal reference line.

  Three ways of FILTERING it were measured and all cost more than the artifact:
      raise MIN_TABLE_CELLS to ~25   reaches it, but LOO p 0.43 -> 0.52
      widen ALPHA_SMOOTH_PTS to 15+  removes it, but LOO p 0.43 -> 0.60 (1.42 at 41 pts)
      LOESS on the R_i axis (_loess) 3.3x smaller, fiducial better, but W2 p 0.13 -> 0.19 and
                                     LOO 0.43 -> 0.64
  Each buys smoothness by blurring real structure, on one axis or the other.

  The actual cause was that table_cells did not exist yet: the table was sampled UNIFORMLY in k,
  while every cell carrying large R/R_i sits bunched at the CD. All 23 sampled cells with a
  window >= 0.70 dex lay in k = 1038..1269, a span holding 232 cells -- 10x under-sampled exactly
  where coverage was thinnest. Sampling that band at full density (see table_cells) took the
  count at 0.70 dex from 24 to 240 and the step from -0.0078 to +0.0007, a 10.6x reduction
  matching the 1/N expected from denser bracketing, and IMPROVED leave-one-out rather than
  trading against it: Delta' 0.43 -> 0.38, 84th 2.69 -> 2.20, worst 13.7 -> 12.5. The residual
  worst steps moved out to 1.035-1.065 dex, past where the fiducial (0.50) or W=2 (0.75) reach.
  Densifying costs extraction time and nothing else. Filtering costs accuracy. Prefer points.

  Densification alone did NOT improve out-of-sample p (0.11 -> 0.16% on the fiducial), and it is
  neither the threshold (identical from MIN_TABLE_CELLS 15 to 100 now that coverage exceeds 100
  everywhere) nor the settling table (identical from either cell list). It is a bias-variance
  trade: sparse sampling SMOOTHED by interpolating between distant cells, while dense sampling
  follows each cell's own measurement noise. The fix is the second half of the pair -- a local
  fit on the R_i axis (_loess), which is harmful on a sparse table and helpful on a dense one,
  and recovers it to 0.15%. DENSIFY FIRST, SMOOTH SECOND.

  A NEW DENSITY DISCONTINUITY now sits at R_i ~ 2.40, where the dense band ends and the R_i
  spacing jumps 0.006 -> 0.062. It is not biting (the largest steps are all at small R_i), but it
  is the same defect one level out: extend TABLE_NDENSE if it ever does.

WHY THE FIDUCIAL, A WIDER RUN AND THE RECONSTRUCTION DIFFER -- and it is ALMOST ALL THE ANCHOR.
Decomposed on Gamma, with every curve normalised at INJ_SAFE (median ln-difference over
R/R_i > 0.1 dex, and the fraction of cells sharing the sign):

    shell width only  (W=1 vs W=2 matched)   -0.000%   48% -> pure noise, no width dependence
    resolution only   (W=2 fine vs coarse)   +0.021%   74% -> real but tiny; coarser cells
                                                              decelerate slightly less
    both              (W=1 vs W=5)           +0.030%   73%
    THE SETTLING ANCHOR (jump -> settled)     ~1.8%          -> an ORDER OF MAGNITUDE larger

  So the visible separation between the curves is the ANCHOR, not the evolution. Gamma's
  jump->settled ratio is +0.0201 on the fiducial against +0.0020 on W=5, and that difference is
  what with_settle exists to keep local. The resolution effect in alpha_G is concentrated at
  ~0.02 dex (0.0019 there, <=0.0005 beyond), i.e. right where the shock is freshly crossed and
  least resolved, and contributes ~0.02%.

  MEASUREMENT TRAP, which cost me a wrong conclusion: normalise BOTH curves at the SAME radius.
  Normalising the fiducial at INJ_SAFE and the comparison run at its row 0 (the jump row) folds
  the 1.8% settling difference into the comparison and manufactures an apparent -0.24% offset,
  100% sign-consistent, which reads exactly like a resolution systematic and is not.

  This also explains why the reconstruction tracks the FIDUCIAL to 0.05% while its alpha_G table
  is built from W=5: there is no systematic slope difference to inherit. The table's alpha_G sits
  within 0.0008 of the fiducial's -- CLOSER than any individual W=5 cell (scatter +-0.002),
  because the local fit averages the per-cell noise away. It takes the SLOPES from W=5, which are
  run-independent, and the ANCHOR from the run being reconstructed, which is not.

R_i/R_0 IS THE RIGHT TABLE KEY, by a factor 26. The obvious alternative is the shell-crossing
phase phi (at matched R_i the fiducial has crossed 40% of its shell where W=5 has crossed 7%, so
they sit at very different stages with very different amounts of shocked material behind them).
Measured cross-run median |d alpha|, keyed on R_i vs on phi:

    a_D   0.0044 vs 0.1156      a_G   0.0012 vs 0.0329      a_p   0.0070 vs 0.1842

  So the cell's own injection radius organises these, not its position in the crossing. Do not
  re-key the table on phi.

WHAT EACH INGREDIENT IS WORTH (ablation, on the fiducial / on W=5 LOO, median %)

    variant                     Delta'         p            Ton
    full model                0.12 / 0.44   0.11 / 0.37   0.08 / 0.08
    no settling correction    2.17 / 1.30   8.99 / 8.61   3.04 / 0.59   <- ESSENTIAL
    Gamma = const             0.12 / 0.44   0.11 / 0.37   2.22 / 4.91   <- alpha_G earns its keep
    p: TM adiabat, integrated 0.12 / 0.44   0.95 / 2.65   0.08 / 0.08
    p: TM adiabat, frozen     0.12 / 0.44   0.80 / 1.34   0.08 / 0.08
    p: tabulated alpha_T      0.12 / 0.44   0.11 / 0.37   0.08 / 0.08   <- identical to alpha_p
    no converged constant     0.12 / 0.44   0.11 / 0.37   0.08 / 0.08   <- not load-bearing here

  WHERE SETTLING ENDS IS DETECTED PER CELL, NOT ASSUMED -- but the fixed cut still wins. The
  project already has a settling detector (fits_hydro.find_fitting_boundaries, whose i0 exists
  because "artifact at beginning corresponds to numerical settling"); settle_end wraps it with
  guards, since on a cell whose p is already declining at row 0 it grabs an unrelated late
  extremum (measured: 0.72 dex, row 3930). Detected settling ends at 0.017 dex median on the
  fiducial and 0.026 on W=5, and its length correlates with R_i at +0.98, so a single fixed value
  is inside settling for 59% of fiducial cells and past it for 41%. Nonetheless SETTLE_MODE
  defaults to 'fixed', because that is what the model error says (see the table there): 'detect'
  wins on same-run leave-one-out and LOSES on both out-of-sample tests. The reason is a mismatch
  rather than a bad detector -- prerar_cell_evolution measures alpha on a grid starting at
  INJ_SAFE, so anchoring at a per-cell g0 != INJ_SAFE integrates the table across a stretch it
  was not measured over. Doing this properly means making INJ_SAFE per-cell in cell_slopes too,
  which would change every number in that module. Worth doing; not done here.

  The settling correction is worth 18x on Delta' and 8x on p, and reproduces entropy_anchor's
  finding from the other direction. Gamma = const costs 2.2-4.9% of observer time, matching the
  2.5-3.2% table_closure predicted from alpha_G alone -- so the second table was necessary, as
  argued there. The converged constant makes no difference on THESE tests because almost every
  sample lies inside the table's coverage; it would matter for a cell extending past it.

THE SETTLING TABLE MUST BE LOCAL TO THE RUN BEING RECONSTRUCTED (with_settle). Importing W=5's
into the fiducial costs 1.56% in Gamma -> 3.4% in Ton, i.e. as much as the Gamma = const
closure; using the fiducial's own drops it to 0.04%. Gamma's jump->settled ratio is +0.0020 on
W=5 and +0.0201 on the fiducial, a factor 10, because it is partly a property of each run's own
shockfit reconstruction rather than of the physics. This costs nothing -- the settling table is
1-D and needs only the run's own first 0.015 dex, which every run has. The DIVISION OF LABOUR:
2-D alpha tables from the wide basis run (what a wide run is FOR), 1-D settling locally.

Example use:
  python -c "import prerar_model as M; M.build_table('cooling_g100_w5', 4)"
  python -c "import prerar_model as M; M.validate_model('cooling_g100_w5','cooling_g100',4)"
  python -c "import prerar_model as M; M.main()"
'''

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from environment import MyEnv, GAMMA_dir
from phys_constants import c_
from working_cooling import _one_minus_beta_over_beta
from working_cooling_data import _adiabat_integrated, _adiabat_frozen
from fits_hydro import find_fitting_boundaries
from phys_functions import savgol_smooth, smooth_bpl0_apy
from fits_hydro import get_fitting_smoothBPL_new
from scipy.signal import savgol_filter
from sweep_norar import _history, SHELL_NAME, shell_cell_range
import prerar_cell_evolution as P

TABLE_KEY = 'cooling_g100_w5'      # the basis run: widest windows (1.069 dex), coarse, cheap
NCELLS = 120
TABLE_NDENSE = 240        # cells sampled at FULL density at the CD end; see table_cells                       # MUST match what was extracted for these runs; the default
                                   # 40 picks a different linspace and finds ~6 cells
OUTDIR = os.path.join(GAMMA_dir, 'bin', 'Tools', 'figures', 'prerar_model')
TABLE_DIR = os.path.join(GAMMA_dir, 'extracted_data')
_TABLE_VERSION = 4                 # bump to invalidate cached tables

COL = {4: '#D55E00', 1: '#0072B2'}
COL_INK, COL_MUTED = '#1a1a1a', '#8a8a8a'


# ---------------------------------------------------------------------------
# the table
# ---------------------------------------------------------------------------
# How the end of numerical settling is located. MEASURED COMPARISON, model error in % (median,
# Delta'/p), which is the only thing that decides this:
#
#                              fiducial      W=5 LOO      W=2
#     'fixed'  (g0 = INJ_SAFE)  0.12/0.11   0.44/0.37   0.08/0.10
#     'detect' (per cell)       0.11/0.27   0.38/0.25   0.14/0.21
#
# 'detect' wins on the SAME-RUN leave-one-out and loses on both out-of-sample tests, which are
# the actual use case -- so 'fixed' is the default. The reason is a mismatch, not a failure of
# the detector: prerar_cell_evolution measures alpha_D on a grid that starts at INJ_SAFE, so
# anchoring the model at a per-cell g0 != INJ_SAFE integrates the alpha table across a stretch
# it was not measured over. Using detection properly would mean re-measuring the alpha tables on
# per-cell settling windows too (i.e. making INJ_SAFE per-cell in prerar_cell_evolution.
# cell_slopes) -- worth doing, not done here, and it would change every number in that module.
SETTLE_MODE = 'fixed'

# Minimum number of table cells that must have coverage at a grid point before the interpolated
# alpha is trusted there. The table's coverage COLLAPSES with radius (measured on
# cooling_g100_w5: 107 cells at 0.10 dex, 72 at 0.30, 45 at 0.50, 24 at 0.70, 7 at 0.95, 3 at
# 1.04), because only long-window cells reach large R/R_i. Each time a cell leaves the set,
# np.interp switches to a different pair of neighbours and the answer JUMPS -- measured
# |d alpha| up to 2.14 (a_D) and 3.54 (a_p) between ADJACENT grid points, at 0.74 and 0.88-0.90
# dex. Integrated, those become visible kinks in the reconstructed Delta' and p at large radius,
# which look like physics and are not.
# Past this threshold the DERIVED asymptote is used instead -- which is where alpha has gone
# anyway by ~0.5-0.6 dex, so this costs nothing physical and removes the artifact at its source.
# 15 rather than 8: scanned against both the artifact and the model error. At 8 a residual step
# survives where the count falls 29 -> 24 (measured -0.0079 in alpha_p in ONE 0.005-dex step at
# 0.70 dex, against ~0.0002 typical) -- only ~0.02% in p, invisible on a declining curve but
# plainly visible as a kink on the near-flat p R^2. Raising to 15 removes it AND improves the
# fiducial (p 0.165 -> 0.113%, Delta' 0.143 -> 0.129%) at no cost at leave-one-out (0.42 -> 0.43).
# Going further degrades LOO (0.48 at 25, 0.50/0.61 at 30) by discarding table information that
# is still real at moderate radii, with no further gain. Do not raise it past ~20.
MIN_TABLE_CELLS = 15
LOESS_NNB = 12            # neighbours in the locally weighted R_i fit; see _loess
# How far BELOW/ABOVE the table's R_i range a lookup may still be answered, in units of the
# local R_i spacing. Needed because flag_offtrend removes the CD-adjacent cell, which IS the
# R_i = 1.000 entry -- so without a margin the very cell the rejection exists to rescue falls
# outside the table and is DECLINED instead of reconstructed. _loess extrapolates linearly from
# the surviving neighbours, which is exactly "use the neighbouring cells' trend".
# Kept to a few spacings: interpolation-with-a-margin, not open-ended extrapolation.
EXTRAP_SPACINGS = 3.
TABLE_EDGE = 0.10         # dex of each cell's window EXCLUDED when building the table. Without
                          # it the table's large-R/R_i columns are built from cells sitting in
                          # their own rarefaction crash: at 1.00 dex EVERY contributing cell is
                          # within 0.15 dex of its handover, and the crash drives alpha_D toward
                          # -0.65 against a true -0.79. That was the bump-then-drop seen as a dip
                          # in the prolonged reconstruction. The precursor is measurable to
                          # ~0.15 dex before the handover (slope -0.003, vs -0.183 AT it); 0.10
                          # removes the damaging part while keeping most of the reach.
MIN_SIDE = 3              # covered cells required on EACH side of Ri_at (within the margin)
                          # before the table is used at that grid point. Without this the fit
                          # EXTRAPOLATES in R_i wherever coverage thins -- at large R/R_i every
                          # long-window cell sits at small R_i, so a lookup at R_i = 1.86 was
                          # being answered by cells at R_i <= 1.70 and diverged (raw alpha_D of
                          # +0.55 against a true ~-0.79), producing a bump-then-drop into the
                          # asymptote that showed as a dip in the prolonged reconstruction.
                          # With the requirement the table simply STOPS where it can only
                          # extrapolate, and the blend carries the curve smoothly onto the
                          # derived asymptote -- i.e. the extension continues as it was.
# ...and alpha is smoothed along the grid over this many points (odd), since alpha(R) is a
# physical relaxation and cannot jump. Removes the residual dropout steps inside the
# well-covered region, where the threshold above does not bite.
ALPHA_SMOOTH_PTS = 9
# Span, in dex, over which the interpolated alpha is BLENDED into the converged constant as the
# coverage thins. A hard switch leaves a step -- measured |d alpha| of 0.66 (a_D) and 1.09 (a_p)
# in one 0.005-dex grid interval at the seam for R_i = 1.5, i.e. 0.76% and 1.25% of the
# reconstructed Delta' and p spent in a single step, which is exactly the unphysical kink at large
# R/R_i. Blending removes the seam instead of smoothing over it.
ALPHA_BLEND_DEX = 0.10

# --- the asymptotic law is DERIVED, not fitted ---------------------------------------------
# The shocked layer is in CAUSAL CONTACT, so its pressure equilibrates. With ~constant shock
# strength the immediate post-shock density and pressure dilute as R^-2, and pressure continuity
# carries that to the contact discontinuity: p_CD ~ p_d ~ R^-2. The CD fluid is not being shocked,
# so it is adiabatic there, p_CD ~ rho_CD^gma -- which fixes the density index, and then mass
# conservation (rho R^2 Delta' = const) fixes the width:
#
#     dlnp/dlnR = -2            dlnrho/dlnR = -2/gma            alpha_D = 2/gma - 2
#
# NOT the free-coasting rho ~ R^-2 / Delta' = const: a coasting cell has no pressure support
# holding it, whereas this one is being compressed toward the CD by the layer it sits in. That
# is why alpha_D plateaus at ~-0.79 instead of relaxing to 0, and it is correct rather than an
# unconverged transient. (Free coasting IS the right law once the rarefaction has passed and the
# layer has decompressed -- which is exactly where norar_history takes over.)
#
# Measured against it, with gma evaluated at the COOLED temperature (T ~ p/rho falls ~3x over the
# window, so gma climbs toward 5/3):
#     RS  alpha_D -0.797 measured   vs  -0.794 derived
#     FS          -0.793            vs  -0.797
#     dlnrho/dlnR -1.205            vs  -1.206          dlnp/dlnR -1.98 vs -2 exactly
#
# It also explains the near-universality found empirically: 2/gma - 2 spans only 0.019 over the
# whole measured Theta_sh range, so alpha_D cannot depend much on the shock state. And it is what
# GENERALISES -- a relativistically hot shell (gma -> 4/3) would give alpha_D -> -0.5, which no
# constant fitted here would predict.
ALPHA_INF_DERIVED = True   # use 2/gma(T) - 2 past the table's coverage rather than the tabulated
                           # constant; they agree to 0.003 here, but only this one transfers.
# Adiabatic index for the DERIVED asymptote. 5/3 -- the FULLY COOLED (non-relativistic) limit,
# not the local value at injection. The asymptote is what the fluid tends to as it expands and
# cools, and gma_ad(T) -> 5/3 as T -> 0; the measured 1.64-1.66 here is the gas on its way there.
# This is not a cosmetic change: it puts the derived alpha_D at 2/gma - 2 = -0.800 exactly, which
# the MEASUREMENT confirms far better than 1.65 did (converged_alpha gives -0.797 RS / -0.793 FS,
# so 0.003-0.007 off 5/3's prediction against 0.009-0.005 off 1.65's -0.788, and on the correct
# side). It also makes the rho scaling in plot_three_way exactly R^1.2.
# The hot limit gma_ad = 4/3 would give rho ~ r^-1.5; the cooled limit gives r^-1.2, which is
# where these cells sit.
GMA_REF = 5./3.

SETTLE_MAX_DEX = 0.06     # largest credible settling length, in decades of R/R_inj. The
                          # detector below takes the FIRST dp/dx sign change, which is the
                          # settling peak when there is one -- but on a cell whose p is already
                          # declining at row 0 it grabs an unrelated late extremum instead
                          # (measured: up to 0.72 dex, row index 3930, on cooling_g100_w5).
                          # Detections beyond this are rejected and the run's median used.
SETTLE_MAX_ROW = 60       # ...and the same guard on the row index, for short-cadence cells.


def settle_end(s, h, fallback=None):
  '''
  Where numerical settling ends for this cell, as log10(R/R_inj), detected per cell rather
  than assumed.

  Uses fits_hydro.find_fitting_boundaries -- the project's existing settling detector, whose
  whole purpose is that "artifact at beginning corresponds to numerical settling" -- on the
  savgol-smoothed pressure, which is the most sensitive channel. This replaces a fixed
  P.INJ_SAFE cut, which is the wrong thing per cell: the detected length correlates with R_i at
  +0.98 on the fiducial, so a single value sits INSIDE settling for 59% of cells and past it for
  the other 41%.

  Returns (dex, ok). ok=False means the detection was implausible (see the guards) and `dex` is
  the fallback -- never silently a bad value.
  '''
  x = s.x.to_numpy(dtype=float)[:h+1]
  p = s.p.to_numpy(dtype=float)[:h+1]
  fb = P.INJ_SAFE if fallback is None else fallback
  if len(x) < 30:
    return fb, False
  try:
    i0, _ = find_fitting_boundaries(x, savgol_smooth(p))
  except Exception:
    return fb, False
  if not (0 < i0 < min(len(x), SETTLE_MAX_ROW)):
    return fb, False
  d = float(np.log10(x[i0]/x[0]))
  if not (0. < d <= SETTLE_MAX_DEX):
    return fb, False
  return d, True


def table_cells(key, z, ncells=NCELLS, ndense=TABLE_NDENSE):
  """
  Cell list for BUILDING the table: every cell over the `ndense` nearest the contact
  discontinuity, plus the usual uniform sample elsewhere.

  Uniform sampling in k is the wrong choice here, and it is what produced the interpolation
  artifact. The cells that carry LARGE R/R_i are exactly the long-window ones, and those are all
  bunched at the CD -- measured on cooling_g100_w5, all 23 sampled cells with a window >= 0.70
  dex lay in k = 1038..1269, a span containing 232 cells of which uniform sampling took 23. So
  the table was 10x under-sampled precisely where its coverage was thinnest, which is why
  np.interp's bracketing pair kept changing and alpha jumped.

  Denser sampling is the fix that costs nothing but extraction time, unlike smoothing or a
  wider coverage threshold, both of which buy smoothness by blurring real structure (see the
  verdict in _loess).

  NB the CD end is HIGH k on the RS (shell 4) and LOW k on the FS (shell 1).
  """
  kmin, kmax, _ = shell_cell_range(key, z)
  dense = (np.arange(kmax - ndense + 1, kmax + 1) if z == 4
           else np.arange(kmin, kmin + ndense))
  ks = np.unique(np.concatenate([dense, P.profile_cells(key, z, ncells)]))
  return ks[(ks >= kmin) & (ks <= kmax)]


def _settle_row(key, k, z, mode=None):
  '''
  log-ratios (Delta', p, Gamma) from the shock-jump row to the first clean grid point
  (P.INJ_SAFE), plus R_i/R_0. None when this cell has no prepended jump row (nothing to
  correct from) or does not reach g0.
  '''
  s = _history(key, int(k), z=z)
  if s is None or len(s) < 3:
    return None
  h, dex = P.prerar_window(s)
  if P.declined(P._theta_sh(s), dex, key) is not None:
    return None
  s_raw = _history(key, int(k), z=z, early_ana=None)
  n_pre = len(s) - len(s_raw) if s_raw is not None else 0
  if n_pre < 1:
    return None
  x, dx, rho, p, lfac, t = (a[:h+1] for a in P._cols(s))
  mode = SETTLE_MODE if mode is None else mode
  g0, ok = (settle_end(s, h) if mode == 'detect' else (P.INJ_SAFE, True))
  tgt = x[0]*10.**g0
  if x[-1] < tgt:
    return None
  j = min(int(np.searchsorted(x, tgt)), len(x) - 1)
  D = lfac*dx
  return dict(k=int(k), Ri=float(x[0]*c_/P._env(key).R0), g0=float(g0), detected=bool(ok),
              sD=float(np.log(D[j]/D[0])), sp=float(np.log(p[j]/p[0])),
              sG=float(np.log(lfac[j]/lfac[0])))



_AVAIL_MEM = {}

def available_cells(key, z):
  """
  EVERY extracted cell of shell z, read off the results directory rather than sub-sampled.

  The settling table should use all of them. It is 1-D and its interpolation error is inherited
  as an ANCHOR OFFSET by every reconstructed curve -- measured: a cell at R_i = 1.86 sits outside
  the table_cells dense band on the fiducial (which covers R_i <~ 1.55) and picks up a 0.25%
  offset in p and 0.10% in Gamma from the moment it starts, which then persists across its whole
  window. Sub-sampling here buys nothing: the fiducial has all 2001 cells on disk.
  """
  if (key, z) not in _AVAIL_MEM:
    import glob
    d = os.path.join(GAMMA_dir, 'results', key, 'cells')
    ks = []
    for f in glob.glob(os.path.join(d, '*.csv')):
      b = os.path.basename(f).split('.')[0]
      if b.isdigit():
        ks.append(int(b))
    kmin, kmax, _ = shell_cell_range(key, z)
    _AVAIL_MEM[(key, z)] = np.array(sorted(k for k in ks if kmin <= k <= kmax))
  return _AVAIL_MEM[(key, z)]


_SETTLE_MEM = {}

def _settle_table(key, z, ncells):
  """
  The settling table for one run, KEEPING ONLY CELLS WHOSE SETTLING END WAS DETECTED.

  Cells where the detector fails are dropped rather than kept with a fallback g0, so that
  neighbouring detections interpolate across the gap instead of a constant flattening the R_i
  trend (the settling length follows R_i at corr +0.98 on the fiducial).

  This only matters when settle_mode='detect', which is NOT the default -- see SETTLE_MODE for
  the measured reason why.
  """
  if (key, z) in _SETTLE_MEM:
    return _SETTLE_MEM[(key, z)]
  ks = available_cells(key, z)
  if not len(ks):
    ks = table_cells(key, z, ncells)
  rows = [r for r in (_settle_row(key, int(k), z) for k in ks) if r is not None]
  if not rows:
    raise RuntimeError(f'{key}: no cell yields a settling row (no prepended jump state?)')
  df = pd.DataFrame(rows)
  ok = df[df.detected]
  if len(ok) >= max(4, int(0.25*len(df))):
    df = ok
  else:
    print(f'  settle_table {key}: only {len(ok)}/{len(df)} detections, keeping all '
          '(the fallback trend is then flat -- treat the settling correction as suspect)')
  out = df.sort_values('Ri').reset_index(drop=True)
  # DO NOT SCREEN THE SETTLING TABLE, even though the CD-adjacent cell is a 86-97 sigma outlier
  # in it (Delta' jump->settled +0.237 against -0.017 for its neighbours: the mesh STRETCHES the
  # cell next to a contact discontinuity by 27% while every other cell contracts ~2%).
  # That deviation is REAL and specific to that cell, not noise, and screening it was measured to
  #   help nobody   -- neighbours' reconstructions are bit-identical either way, because _settle
  #                    interpolates between BRACKETING entries and a dense table brackets the
  #                    outlier out of everyone else's way
  #   hurt badly    -- the CD cell's own anchor residual goes 0.014% -> 31%, since substituting
  #                    the neighbour trend for its settling is precisely a 29% error
  # The alpha table IS screened (flag_offtrend), because there the CD cell's deviation is modest
  # (3-6 sigma) and does propagate. Different tables, opposite treatment, both measured.
  _SETTLE_MEM[(key, z)] = out          # rebuilt per session only: ~2 CSV reads per cell
  return out

OUTLIER_NSIG = 4.0        # cells this many robust sigma off the local R_i trend are DROPPED
OUTLIER_SPAN = 8          # neighbours each side used to define that trend (excluding the cell)
OUTLIER_DEX = (0.10, 0.20, 0.30)   # grid points the test is evaluated at


def flag_offtrend(cells, Ri, series='a_D', nsig=OUTLIER_NSIG, span=OUTLIER_SPAN):
  """
  Boolean mask of cells whose alpha sits off the local trend in R_i, and so must not enter the
  table.

  The motivating case is the CONTACT-DISCONTINUITY-ADJACENT cell, which is anomalous in every
  run: measured at 0.20 dex it sits +0.0184 (fiducial, k=519) and +0.0117 (W=5, k=1269) off the
  trend set by its neighbours, which are themselves within +-0.006. The CD is where the moving
  mesh does the most work -- the same place whose FS counterpart carries a mass-conservation
  residual 500x the float32 floor -- so its cell is not measuring the same thing the others are.

  Left in, it does double damage: it contaminates the table at R_i ~ 1.0, AND it is the cell a
  "longest window" selection always picks first (it is shocked first), so it lands in figures.
  Its reconstruction then carries a 0.77% rho ramp against 0.02-0.16% for its neighbours.

  Dropping it lets the neighbours' trend supply alpha at that R_i by interpolation, which is the
  physically consistent value; the reconstruction there is then as good as anywhere else.

  Judged on the deviation at OUTLIER_DEX, against a robust (MAD) spread over all cells, so the
  test adapts to the run rather than to a hand-set tolerance.
  """
  grid = _grid_of(cells)
  n = len(cells)
  dev = np.full(n, np.nan)
  for i in range(n):
    ds = []
    for gg in OUTLIER_DEX:
      j = int(round(gg/(grid[1] - grid[0])))
      if j >= grid.size or not np.isfinite(cells[i][series][j]):
        continue
      lo, hi = max(0, i - span), min(n, i + span + 1)
      nb = [(Ri[q], cells[q][series][j]) for q in range(lo, hi)
            if q != i and np.isfinite(cells[q][series][j])]
      if len(nb) < 5:
        continue
      x = np.array([q[0] for q in nb]); y = np.array([q[1] for q in nb])
      ds.append(cells[i][series][j] - np.polyval(np.polyfit(x, y, 1), Ri[i]))
    if ds:
      dev[i] = np.median(ds)
  # Judge each deviation against the LOCAL scatter, not a global one. The R_i spacing varies
  # 10x across the table (0.006 in the dense CD band, 0.062 in the sparse tail), so a local
  # linear fit is intrinsically noisier where cells are sparse. A single global MAD is then set
  # by the sparse tail and the genuinely bad CD cell no longer stands out: with a global scale
  # this dropped 15 sparse-region cells and MISSED k=1269, which is the cell it exists for.
  bad = np.zeros(n, dtype=bool)
  ok = np.isfinite(dev)
  if ok.sum() < 10:
    return bad
  for i in np.flatnonzero(ok):
    lo, hi = max(0, i - 3*span), min(n, i + 3*span + 1)
    loc = dev[lo:hi][np.isfinite(dev[lo:hi])]
    loc = loc[np.arange(len(loc)) != min(i - lo, len(loc) - 1)]
    if loc.size < 6:
      continue
    med = np.median(loc)
    mad = 1.4826*np.median(np.abs(loc - med))
    bad[i] = abs(dev[i] - med) > nsig*max(mad, 1e-4)
  return bad


def _grid_of(cells):
  return _grid_cache.setdefault('g', P._grid())


_grid_cache = {}

def table_path(src_key, z):
  return os.path.join(TABLE_DIR, f'prerar_table_{src_key}_z{z}.npz')


def build_table(src_key=TABLE_KEY, z=4, ncells=NCELLS, save=True, verbose=True):
  '''
  The reconstruction table, from one run. Contents:
    Ri, grid, aD, aG   the 2-D expansion tables (NaN outside each cell's window)
    dex                each source cell's window length
    aD_inf, aG_inf     the CONVERGED constants, used beyond a cell's tabulated coverage
    sD, sp, sG         the 1-D settling log-ratios (see the module docstring)
    R0                 the source run's R_0, so a lookup can refuse a run with a different one

  aD_inf / aP_inf are stored as MEASURED (converged_alpha, never asymptote() -- for one and the
  same quantity the latter returns -0.760 / -0.808 / -0.840 at W = 1 / 2 / 5, because its fixed
  [0.25,0.35] window averages in cells still mid-transient and drifts steeper with shell width).
  But predict_history uses the DERIVED causal-contact values by default (ALPHA_INF_DERIVED),
  because only those generalise. The two agree here to 0.009 in alpha_D (-0.797 measured vs
  -0.788 derived) and 0.015 in alpha_p (-1.985 vs -2.000), and give identical model accuracy --
  so the stored values serve as a running check on the derivation rather than as the law.
  '''
  ks_tab = table_cells(src_key, z, ncells)
  grid, cells, Ri = P._cells_by_Ri(src_key, z, ncells, ks=ks_tab)
  # drop cells sitting off the local R_i trend before they can contaminate the table
  bad = flag_offtrend(cells, Ri)
  if bad.any():
    dropped = [int(cells[i]['k']) for i in np.flatnonzero(bad)]
    cells = [c for c, b in zip(cells, bad) if not b]
    Ri = Ri[~bad]
    if verbose:
      print(f'  off-trend cells DROPPED ({bad.sum()}): k = ' +
            ', '.join(str(k) for k in dropped[:12])
            + (' ...' if len(dropped) > 12 else ''))
  def _clip_edge(key):
    # blank each cell's last TABLE_EDGE dex: those samples are inside its own rarefaction
    # precursor/crash and are not measuring the pre-rarefaction law (see TABLE_EDGE)
    A = np.array([d[key] for d in cells])
    for i, d in enumerate(cells):
      A[i, grid > d['dex'] - TABLE_EDGE] = np.nan
    return A
  aD, aG, aP, aT = (_clip_edge(k) for k in ('a_D', 'a_G', 'a_p', 'a_T'))
  dex = np.array([d['dex'] for d in cells])

  conv = P.converged_alpha(src_key, z, ncells, verbose=False)
  if not len(conv):
    raise RuntimeError(f'{src_key} has no cells reaching the converged range; it cannot be a '
                       'basis run (the fiducial cannot -- that is why a wide run exists)')
  aD_inf = float(conv.a_D.median())
  # alpha_G's converged value: same range, same cells, read off the aG table
  m_inf = (grid >= 0.55) & (grid <= 0.65)
  aG_inf = float(np.nanmedian(aG[dex >= 0.68][:, m_inf])) if (dex >= 0.68).any() \
      else float(np.nanmedian(aG[:, m_inf]))

  st = _settle_table(src_key, z, ncells)

  # Gamma also fitted with the thin-shell model's smooth broken power law (beta = 0, so the
  # second segment is constant -- i.e. Gamma -> const, the coasting asymptote). Stored so
  # gamma_law='bpl' can be priced against the alpha_G table; see that option in predict_history.
  gB = np.full((len(Ri), 4), np.nan)
  for i, d in enumerate(cells):
    sh = _history(src_key, int(d['k']), z=z)
    if sh is None:
      continue
    hh, dx_cell = P.prerar_window(sh)        # NOT `dex`: that name holds the per-cell array
    x, dxc, rho, pp, lf, tt = (a[:hh+1] for a in P._cols(sh))
    L = np.log10(x/x[0])
    m = (L >= P.INJ_SAFE) & (L <= dx_cell - P.EDGE_EXCL)
    if m.sum() < 40:
      continue
    try:
      gB[i] = get_fitting_smoothBPL_new(x[m]/x[0], lf[m]/lf[m][0], beta=0., cleanData=False)
    except Exception:
      pass

  aP_inf = float(np.nanmedian(aP[dex >= 0.68][:, m_inf])) if (dex >= 0.68).any() \
      else float(np.nanmedian(aP[:, m_inf]))
  aT_inf = float(np.nanmedian(aT[dex >= 0.68][:, m_inf])) if (dex >= 0.68).any() \
      else float(np.nanmedian(aT[:, m_inf]))
  tab = dict(Ri=Ri, grid=grid, aD=aD, aG=aG, aP=aP, aT=aT, gB=gB, dex=dex,
             aD_inf=aD_inf, aG_inf=aG_inf, aP_inf=aP_inf, aT_inf=aT_inf,
             sRi=st.Ri.to_numpy(), sD=st.sD.to_numpy(), sp=st.sp.to_numpy(),
             sG=st.sG.to_numpy(), sg0=st.g0.to_numpy(),
             R0=float(P._env(src_key).R0), Rcross=float(Ri.max()),
             meta=json.dumps(dict(src_key=src_key, z=int(z), ncells=int(ncells),
                                  version=_TABLE_VERSION, n_cells=int(len(Ri)),
                                  n_settle=int(len(st)))))
  if save:
    os.makedirs(TABLE_DIR, exist_ok=True)
    np.savez(table_path(src_key, z), **tab)
  if verbose:
    print(f'\nbuild_table {src_key} {SHELL_NAME.get(z, z)}: {len(Ri)} cells, '
          f'R_i/R_0 in [{Ri.min():.3f}, {Ri.max():.3f}], windows to {dex.max():.3f} dex')
    print(f'  converged  alpha_D_inf = {aD_inf:+.4f}   alpha_G_inf = {aG_inf:+.4f}')
    print(f'  settling   {len(st)} cells:  Delta\' {st.sD.median():+.4f}  '
          f'p {st.sp.median():+.4f}  Gamma {st.sG.median():+.4f}  (median log-ratios)')
    if save:
      print(f'  -> {table_path(src_key, z)}')
  return tab


def load_table(src_key=TABLE_KEY, z=4, ncells=NCELLS, rebuild=False):
  '''Cached table; rebuilds when missing, stale or rebuild=True.'''
  path = table_path(src_key, z)
  if not rebuild and os.path.isfile(path):
    d = np.load(path, allow_pickle=False)
    meta = json.loads(str(d['meta']))
    if meta.get('version') == _TABLE_VERSION and meta.get('ncells') == ncells:
      return {k: d[k] for k in d.files}
  return build_table(src_key, z, ncells, verbose=False)


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------
def with_settle(tab, key, z=4, ncells=NCELLS):
  '''
  A copy of `tab` whose 1-D SETTLING tables are rebuilt from `key` instead of from the run the
  2-D alpha tables came from.

  THIS IS REQUIRED WHENEVER THE TEST RUN IS NOT THE TABLE RUN, and it is not a tuning knob.
  The settling ratio is partly a property of each run's own shockfit reconstruction rather than
  of the physics, so it does not transfer: Gamma's jump->settled ratio is +0.0020 on
  cooling_g100_w5 but +0.0201 on the fiducial, a factor 10, and importing the wrong one costs
  1.56% in Gamma -> 3.4% in observer time, i.e. as much as the Gamma = const closure the
  alpha_G table exists to beat. With the local one it drops to ~0.1%.

  This costs nothing: the settling table is 1-D in R_i and needs only the run's OWN first
  0.015 dex, which every run has, the fiducial included. Only the 2-D alpha tables need a wide
  basis run. So the division of labour is:
      alpha_D, alpha_G (2-D)   from the wide basis run   -- what a wide run is FOR
      settling (1-D)           from the run being reconstructed
  '''
  out = dict(tab)
  st = _settle_table(key, z, ncells)
  # R_cross is local too: whether a cell is past shock crossing depends on ITS OWN shell
  _, _, Ri_loc = P._cells_by_Ri(key, z, ncells, ks=table_cells(key, z, ncells))
  out.update(sRi=st.Ri.to_numpy(), sD=st.sD.to_numpy(), sp=st.sp.to_numpy(),
             sG=st.sG.to_numpy(), sg0=st.g0.to_numpy(), Rcross=float(Ri_loc.max()))
  return out


def _loess(xs, ys, x0, nnb=LOESS_NNB):
  """
  Locally weighted linear fit of ys(xs) evaluated at x0, tricube kernel over the nnb nearest
  points. ADOPTED, but only once the table was made dense -- see the verdict below, which
  REVERSED.

  It replaces np.interp on the R_i axis for a structural reason.
  np.interp uses only the two BRACKETING cells, so when the table's coverage thins with radius
  and one of them drops out, the answer jumps discretely -- measured -0.0079 in alpha_p in a
  single 0.005-dex grid step where the count fell 29 -> 24, which is a visible kink on the
  near-flat p R^2 even though it is only ~0.02% in p. A weighted local fit degrades gracefully
  instead: a departing cell's influence is already ~0 if it is far, so nothing jumps.

  VERDICT, AND IT DEPENDS ENTIRELY ON TABLE DENSITY -- which is the interesting part.

    on the SPARSE table (112 cells): NOT adopted. Fiducial improved but W2 p 0.13 -> 0.19% and
      W=5 LOO 0.43 -> 0.64% got worse, because smoothing along R_i blurred real R_i structure.
    on the DENSE table (329 cells): ADOPTED, nnb = 12. Every out-of-sample number improves --
      fiducial p 0.161 -> 0.149%, W2 p 0.122 -> 0.118%, LOO Delta' 0.378 -> 0.366% -- with LOO p
      flat (0.397 -> 0.400).

  The reversal is the bias-variance trade made visible. Sparse sampling SMOOTHS by construction,
  since interpolating between distant cells averages them, so adding LOESS on top over-smooths
  and destroys structure. Dense sampling instead follows each cell's own measurement noise
  faithfully, and THEN a local fit has noise to suppress and structure to spare. Densify first,
  smooth second; smoothing a sparse table is strictly harmful.

  Nor do the alternatives work: raising MIN_TABLE_CELLS to reach 0.70 dex needs ~25 and costs
  LOO accuracy (p 0.43 -> 0.52); widening ALPHA_SMOOTH_PTS to 15+ removes the step but blurs
  real structure (LOO p 0.43 -> 0.60, 1.42 by 41 pts).

  CONCLUSION: the residual ~0.0024-0.0079 step in alpha_p near 0.70 dex is at the level of the
  TABLE'S OWN SAMPLING NOISE, and every way of removing it costs more real accuracy than the
  artifact costs. It is ~0.02% in p -- invisible in any residual statistic, visible only as a
  kink on the near-flat p R^2 against its horizontal reference line. Left in, documented, and
  not chased further.
  """
  d = np.abs(xs - x0)
  n = len(xs)
  if n < 3:
    return float(np.interp(x0, xs, ys))
  k = min(nnb, n) - 1
  h = np.partition(d, k)[k]
  if not (h > 0):
    h = float(d.max()) or 1.0
  w = np.clip(1. - (d/h)**3, 0., None)**3
  if np.count_nonzero(w) < 3:
    return float(np.interp(x0, xs, ys))
  # weighted linear least squares, centred on x0 so the intercept IS the estimate
  dx = xs - x0
  sw = w.sum()
  sx = (w*dx).sum()
  sxx = (w*dx*dx).sum()
  sy = (w*ys).sum()
  sxy = (w*dx*ys).sum()
  det = sw*sxx - sx*sx
  if abs(det) < 1e-30*max(sw*sxx, 1e-30):
    return float(sy/sw)
  est = (sxx*sy - sx*sxy)/det
  # CLAMP to the range of the points that actually carry weight. A local LINEAR fit evaluated
  # outside its data extrapolates, and where the table's coverage thins the contributing cells
  # are all on ONE side of Ri_at, so it diverges: measured raw alpha_D of +0.55 against a true
  # ~-0.79 just inside the coverage limit, which the blend then dragged into a visible dip in
  # the prolonged reconstruction. A local fit cannot legitimately leave the local data range.
  wm = w > 0.
  if wm.any():
    lo, hi = float(np.min(ys[wm])), float(np.max(ys[wm]))
    est = min(max(est, lo), hi)
  return float(est)


def GRID_STEP_LOCAL(grid):
  '''Grid spacing in dex, read off the grid itself rather than assumed.'''
  return float(grid[1] - grid[0]) if grid.size > 1 else 1.


def _profile(tab, Ri_at, series, exclude=None):
  '''
  alpha(grid) for a cell at Ri_at: interpolated in R_i where the table has coverage, and the
  CONVERGED CONSTANT beyond it. Returns None if Ri_at is outside the table's R_i range -- the
  table declines rather than extrapolating that axis.
  '''
  key = {'a_D': ('aD', 'aD_inf'), 'a_G': ('aG', 'aG_inf'),
         'a_p': ('aP', 'aP_inf'), 'a_T': ('aT', 'aT_inf')}[series]
  A, inf = tab[key[0]], float(tab[key[1]])
  Ri, grid = tab['Ri'], tab['grid']
  sp = float(np.median(np.diff(np.sort(Ri)))) if Ri.size > 3 else 0.
  marg = EXTRAP_SPACINGS*sp
  if not (Ri.min() - marg <= Ri_at <= Ri.max() + marg):
    return None
  out = np.full(grid.size, np.nan)
  nsamp = np.zeros(grid.size, dtype=int)
  for g in range(grid.size):
    col = A[:, g]
    m = np.isfinite(col)
    if exclude is not None and exclude < len(col):
      m = m.copy()
      m[exclude] = False
    if m.sum() >= 3:
      Rm = Ri[m]
      # the table is used only where Ri_at is BRACKETED by covered cells (a margin at each end
      # keeps the CD-adjacent cell, which sits just below the table's minimum, usable)
      n_lo = int((Rm <= Ri_at + marg).sum())
      n_hi = int((Rm >= Ri_at - marg).sum())
      if n_lo >= MIN_SIDE and n_hi >= MIN_SIDE:
        out[g] = _loess(Rm, col[m], Ri_at)
        nsamp[g] = int(m.sum())
  # where the contributing sample is too thin to interpolate stably, use the converged
  # constant rather than whatever two surviving neighbours happen to bracket Ri_at
  if ALPHA_INF_DERIVED and series in ('a_D', 'a_p', 'a_G'):
    # causal-contact law: dlnp/dlnR = -2 and alpha_D = 2/gma - 2 (see ALPHA_INF_DERIVED).
    # alpha_G -> 0: nothing is pushing the cell once the layer stops driving it, so Gamma is
    # constant. The MEASURED aG_inf (-0.0054) is a fixed-[0.55,0.65]-dex median that averages in
    # cells still in transient -- the same bias that made asymptote() wrong for alpha_D -- and
    # applying it past the table's coverage makes Gamma keep declining, 0.25% over 0.2 dex,
    # which showed up as a spurious downturn at the end of the widest reconstructions.
    # The data agrees with 0: on the deepest cells alpha_G is -0.0027 at 0.5 dex, -0.0007 at
    # 0.7 and +0.0012 at 1.0, i.e. zero to within noise.
    inf = {'a_p': -2., 'a_D': 2./GMA_REF - 2., 'a_G': 0.}[series]
  good = np.flatnonzero((nsamp >= MIN_TABLE_CELLS) & np.isfinite(out))
  if not good.size:
    return None
  jl = int(good[-1])                       # last grid point with a trustworthy sample
  # bridge any interior gap, then BLEND into the constant over ALPHA_BLEND_DEX before the seam
  # and hold it after -- a hard switch leaves a step of ~0.7-1.1 in alpha (see ALPHA_BLEND_DEX)
  out[:jl+1] = np.interp(grid[:jl+1], grid[good], out[good])
  # SMOOTH BEFORE BLENDING, not after. The blend leaves a kink at the seam (the ramp meets the
  # constant), and a quadratic savgol run across that kink RINGS: measured spikes of +0.028
  # (R_i=1.86, at 0.820 dex) and +0.040 (R_i=1.05, at 1.060) in alpha_D right at the coverage
  # limit, which integrate into visible dips in the prolonged reconstruction. Smoothing the
  # table region first and blending afterwards leaves the ramp untouched.
  i0 = int(np.searchsorted(grid, P.INJ_SAFE))
  ws = min(ALPHA_SMOOTH_PTS, ((jl + 1 - i0) // 2)*2 - 1)
  if ws >= 5:
    out[i0:jl+1] = savgol_filter(out[i0:jl+1], ws, 2)
  nb = max(1, int(round(ALPHA_BLEND_DEX/GRID_STEP_LOCAL(grid))))
  j0 = max(0, jl - nb)
  w = np.clip((grid[j0:jl+1] - grid[j0])/max(grid[jl] - grid[j0], 1e-12), 0., 1.)
  out[j0:jl+1] = (1. - w)*out[j0:jl+1] + w*inf
  out[jl+1:] = inf
  # alpha(R) is a relaxation and cannot jump: smooth along the grid to remove the residual
  # cell-dropout steps (see MIN_TABLE_CELLS).
  # SMOOTH ONLY THE USED REGION (grid >= INJ_SAFE). Below that the profile holds the
  # prepend-straddle garbage -- a slope window spanning the model shockfit row and the first
  # measured row measures that discontinuity, which is why INJ_SAFE exists at all. Running the
  # filter across the whole array lets those points bleed ~half a window (0.02 dex) upward, and
  # they were doing exactly that: the reconstructed rho residual climbed 0.01% -> 0.26% over
  # 0.015-0.034 dex, an 8x faster rate than the sustained ramp beyond it, which read as a
  # spurious jump at the start of every residual panel.

  return out


def _settle(tab, Ri_at, exclude_Ri=None):
  '''
  (sD, sp, sG) log-ratios at Ri_at, interpolated in R_i; edges clamp to the end values.

  exclude_Ri: drop the settling entry at this R_i before interpolating. Needed for a genuine
  leave-one-out -- otherwise the test cell supplies its own settling correction and that part
  of the prediction is not held out at all.
  '''
  sRi = tab['sRi']
  m = np.ones(sRi.size, dtype=bool)
  if exclude_Ri is not None:
    j = int(np.argmin(np.abs(sRi - exclude_Ri)))
    if abs(sRi[j] - exclude_Ri) <= 1e-9:
      m[j] = False
  if m.sum() < 2:
    m = np.ones(sRi.size, dtype=bool)
  return tuple(float(np.interp(Ri_at, sRi[m], tab[c][m]))
               for c in ('sD', 'sp', 'sG', 'sg0'))


def _integ_at(L, grid, I, a_inf):
  """
  Evaluate a cumulative alpha-integral at log-radii L, CONTINUING ANALYTICALLY past the end of
  the grid instead of clamping.

  np.interp clamps outside its range, which silently freezes the integral: beyond GRID_MAX the
  reconstruction then held Delta' and p CONSTANT, so rho ~ R^-2 (rho R^1.2 decaying as R^-0.8)
  and p R^2 growing as R^2 -- visible the moment the figure was extended past 1.2 dex. Past the
  grid alpha is the derived asymptote by construction, so the integral continues as
  a_inf * ln(10) * (L - L_end); this is exact, not an approximation.
  """
  out = np.interp(L, grid, I)
  over = L > grid[-1]
  if np.any(over):
    out[over] = I[-1] + a_inf*np.log(10.)*(L[over] - grid[-1])
  return out


def predict_history(tab, inj_row, x_ext, z=4, settle=True, gamma_law='table',
    adiabat='table', alpha_inf=True, exclude=None, anchor_row=None):
  '''
  Reconstructed hydro at radii x_ext (light-seconds, ascending, >= inj_row.x), from the
  shock-jump state inj_row. Returns the _norar_rows dict {rho, p, lfac, vx, dx, t, x}, or None
  if the table declines this cell's R_i.

  Options exist to be PRICED by ablation(), not to be tuned:
    settle=False           skip the settling correction (anchor on the jump state)
    gamma_law='const'      Gamma = Gamma_i instead of the alpha_G table
    adiabat='integrated'   p from the TM adiabat with gma_ad evolving, instead of tabulated
    adiabat='frozen'       p from the TM adiabat with gma_ad held at the anchor
    alpha_inf=False        hold the table's last value instead of the converged constant

  adiabat DEFAULTS TO 'table' -- p from its own measured log-slope -- because the gas is NOT
  exactly adiabatic and the EoS closures cannot express that: measured p error 0.11% (fiducial)
  / 0.37% (W=5 LOO) against 0.95% / 2.65% integrated and 0.80% / 1.34% frozen. Note that frozen
  beating integrated here is an ACCIDENTAL CANCELLATION, not a reason to prefer it: the measured
  index is ~1.626, below the frozen 1.641, and integrating raises gma_ad toward 5/3, i.e. away
  from it. The same accident is on record for the post-handover law (see the norar-counterfactual
  note), where it cancelled a dr = const error instead.
  The price of 'table' is that p is no longer tied to rho by an equation of state, so the
  reconstruction is not thermodynamically closed -- correct, since the gas is not adiabatic, but
  it means the implied entropy drifts by design. Use 'integrated' if a closed EoS matters more
  than a factor 7 in p.
  '''
  x_i = float(inj_row.x)
  Ri_at = x_i*c_/float(tab['R0'])
  aD = _profile(tab, Ri_at, 'a_D', exclude)
  if aD is None:
    return None
  aG = _profile(tab, Ri_at, 'a_G', exclude)
  if aG is None:
    return None
  if not alpha_inf:
    # hold the last tabulated value rather than the converged constant
    for a in (aD, aG):
      fin = np.flatnonzero(np.isfinite(a))
      a[fin[-1]+1:] = a[fin[-1]]

  grid = tab['grid']
  rho_i, p_i = float(inj_row.rho), float(inj_row.p)
  lf_i = (float(inj_row.lfac) if 'lfac' in inj_row.index
          else 1./np.sqrt(1. - float(inj_row.vx)**2))
  D_i = lf_i*float(inj_row.dx)

  # g0 is the END OF NUMERICAL SETTLING for this cell, detected per cell (settle_end) rather
  # than assumed: it correlates with R_i at +0.98 on the fiducial, so one fixed value sits
  # inside settling for most cells and past it for the rest.
  if settle:
    sD, sp, sG, g0 = _settle(tab, Ri_at, Ri_at if exclude is not None else None)
  else:
    sD, sp, sG, g0 = 0., 0., 0., P.INJ_SAFE
  # state at g0, i.e. after the un-modellable settling stretch
  x_0 = x_i*10.**g0
  D_0, p_0, lf_0 = D_i*np.exp(sD), p_i*np.exp(sp), lf_i*np.exp(sG)
  rho_0 = rho_i*(x_i/x_0)**2*(D_i/D_0)          # mass conservation across the stretch


  # ANCHOR ON A MEASURED POST-SETTLING STATE, when one is supplied.
  # The objective is to FOLLOW the cell where the simulation exists and PROLONG it after, so
  # where data exists the settling stretch should be taken, not predicted. That removes the
  # settling table from the reconstruction path entirely, and with it its interpolation error
  # (measured anchor offsets of -0.19% in p, -0.0075% in Gamma).
  # It also handles the CD-adjacent cell correctly BY CONSTRUCTION: its Delta' grows 27% through
  # settling against -2% for its neighbours (86-97 sigma), which is real behaviour of the cell
  # next to a contact discontinuity. Predicting it from the neighbour trend is a 29% error;
  # taking it from the data costs nothing.
  # anchor_row=None keeps the fully predictive path, which is what a run with NO simulation
  # needs (the jump state from fits_from_au) -- both are supported deliberately.
  if anchor_row is not None:
    x_a = float(anchor_row.x)
    g0 = float(np.log10(x_a/x_i))
    lf_0 = (float(anchor_row.lfac) if 'lfac' in anchor_row.index
            else 1./np.sqrt(1. - float(anchor_row.vx)**2))
    D_0 = lf_0*float(anchor_row.dx)
    rho_0, p_0 = float(anchor_row.rho), float(anchor_row.p)
    t_i, x_i_eff = float(anchor_row.t), x_a
  else:
    t_i, x_i_eff = float(inj_row.t), x_i

  # integrate the expansion laws on the table grid, zeroed at g0
  ln10 = np.log(10.)
  def integ(a):
    c = np.concatenate(([0.], np.cumsum(0.5*(a[1:] + a[:-1])*np.diff(grid)*ln10)))
    return c - np.interp(g0, grid, c)
  ID, IG = integ(aD), integ(aG)

  x_ext = np.asarray(x_ext, dtype=float)
  L = np.log10(x_ext/x_i)
  # below g0 the model has nothing but the two tabulated endpoints: interpolate between them
  aD_inf = 2./GMA_REF - 2. if ALPHA_INF_DERIVED else float(tab['aD_inf'])
  aG_inf = 0. if ALPHA_INF_DERIVED else float(tab['aG_inf'])
  wD = np.where(L >= g0, _integ_at(L, grid, ID, aD_inf), (L/g0)*sD + 0.*L)
  wG = np.where(L >= g0, _integ_at(L, grid, IG, aG_inf), (L/g0)*sG + 0.*L)
  D = np.where(L >= g0, D_0*np.exp(wD), D_i*np.exp(wD))
  lfac = np.where(L >= g0, lf_0*np.exp(wG), lf_i*np.exp(wG))
  if gamma_law == 'bpl' and 'gB' in tab:
    # thin-shell-style smooth BPL with beta = 0, its 4 parameters interpolated in R_i
    B = tab['gB']
    ok = np.isfinite(B).all(axis=1)
    if ok.sum() >= 3 and tab['Ri'][ok].min() <= Ri_at <= tab['Ri'][ok].max():
      par = [np.interp(Ri_at, tab['Ri'][ok], B[ok, q]) for q in range(4)]
      f = smooth_bpl0_apy(10.**L, *par)
      f0 = smooth_bpl0_apy(10.**g0, *par)
      lfac = lf_0*f/f0
  if gamma_law == 'const':
    lfac = np.full(x_ext.shape, lf_i)
    D = D  # Delta' is unchanged: it is tabulated directly, not via Gamma
  lfac = np.maximum(lfac, 1. + 1e-12)

  # mass conservation from the ANCHOR state (identical to the jump state when anchor_row=None,
  # since rho_0 = rho_i (x_i/x_0)^2 (D_i/D_0) by construction there)
  rho = rho_0*(np.float64(10.)**(g0)*x_i/x_ext)**2*(D_0/D)
  if adiabat == 'temp':
    # p = rho*T with T tabulated. Algebraically equivalent to tabulating p, since rho is exact
    # from mass conservation and alpha_T = alpha_p - alpha_rho -- but T is the thermodynamic
    # state variable, so this is the form to use if anything downstream needs gma_ad(T) or c_s
    # consistently. Measured: see the ablation.
    aT = _profile(tab, Ri_at, 'a_T', exclude)
    if aT is None:
      return None
    IT = integ(aT)
    # d ln T = d ln p - d ln rho across the settling stretch; d ln rho there is implied by
    # mass conservation, -2 d lnR - d ln Delta'
    sT = sp - (-2.*g0*np.log(10.) - sD)
    wT = np.where(L >= g0, _integ_at(L, grid, IT, float(tab['aT_inf'])),
                  (L/g0)*sT + 0.*L)
    T0 = (p_0/rho_0)
    T = np.where(L >= g0, T0*np.exp(wT), (p_i/rho_i)*np.exp(wT))
    p = rho*T
  elif adiabat == 'table':
    # p from its OWN measured log-slope rather than from the EoS. The gas is not exactly
    # adiabatic (a_p/a_rho = 1.625 vs gma_ad = 1.641), and over ~1 dex that deficit integrates
    # to a few percent that neither the frozen nor the integrated adiabat can express.
    aP = _profile(tab, Ri_at, 'a_p', exclude)
    if aP is None:
      return None
    IP = integ(aP)
    wP = np.where(L >= g0, _integ_at(L, grid, IP, -2. if ALPHA_INF_DERIVED
                                     else float(tab['aP_inf'])), (L/g0)*sp + 0.*L)
    p = np.where(L >= g0, p_0*np.exp(wP), p_i*np.exp(wP))
  elif adiabat == 'frozen':
    p = _adiabat_frozen(rho, rho_0, p_0)
  else:
    p = _adiabat_integrated(rho, rho_0, p_0)
  dx = D/lfac
  vx = np.sqrt(lfac**2 - 1.)/lfac
  # lab time: t_i + (x - x_i) + int (1-beta)/beta dx. NEVER int dx/beta -- the entire
  # Ton observable lives in that second term (~700 light-s out of x ~ 1e7), so building it
  # by subtraction destroys it. See _one_minus_beta_over_beta.
  xa = np.concatenate(([x_i_eff], x_ext))
  fa = _one_minus_beta_over_beta(np.concatenate(([lf_0 if anchor_row is not None else lf_i],
                                                 lfac)))
  lag = np.concatenate(([0.], np.cumsum(0.5*(fa[1:] + fa[:-1])*np.diff(xa))))[1:]
  t = t_i + (x_ext - x_i_eff) + lag
  return dict(rho=rho, p=p, lfac=lfac, vx=vx, dx=dx, t=t, x=x_ext)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
def _test_cells(test_key, z, ncells, ks=None):
  '''(k, history, i_h, dex, R_i) for every measurable cell of the test run.'''
  out = []
  for k in (P.profile_cells(test_key, z, ncells) if ks is None else ks):
    s = _history(test_key, int(k), z=z)
    if s is None or len(s) < 3:
      continue
    h, dex = P.prerar_window(s)
    if P.declined(P._theta_sh(s), dex, test_key) is not None:
      continue
    x = s.x.to_numpy(dtype=float)
    out.append((int(k), s, h, dex, float(x[0]*c_/P._env(test_key).R0)))
  return out


def validate_model(table_key=TABLE_KEY, test_key='cooling_g100', z=4, ncells=NCELLS,
    loo=False, local_settle=True, anchor='measured', verbose=True, **kw):
  '''
  Predicted vs measured hydro, per cell, over each test cell's RAREFACTION-FREE window
  (injection to its handover) -- the only domain where "without the rarefaction" and "there is
  data" both hold.

  loo=True excludes each test cell from the table before predicting it. REQUIRED when
  test_key == table_key, or the test is circular; harmless otherwise.

  Errors are |d ln| on Delta', rho, p, Gamma, plus the observer-time proxy 2*|d ln Gamma|
  (Ton = t - R/c with 1-beta ~ 1/2Gamma^2, so a Gamma error doubles into it). Binned by
  R/R_i decade, because a model good to 0.5 dex and poor at 1.0 is a useful result and one
  number would hide it.
  '''
  tab = load_table(table_key, z, ncells)
  # the settling table must be LOCAL to the run being reconstructed -- see with_settle
  if local_settle and test_key != table_key:
    tab = with_settle(tab, test_key, z, ncells)
  tRi = tab['Ri']
  rows = []
  n_declined = 0
  for k, s, h, dex, Ri in _test_cells(test_key, z, ncells):
    j_excl = int(np.argmin(np.abs(tRi - Ri))) if loo else None
    if loo and abs(tRi[j_excl] - Ri) > 1e-9:
      j_excl = None                       # not the same cell after all; nothing to exclude
    x, dx, rho, p, lfac, t = (a[:h+1] for a in P._cols(s))
    L = np.log10(x/x[0])
    m = (L >= P.INJ_SAFE) & (L <= dex - P.EDGE_EXCL)
    if m.sum() < 5:
      continue
    # anchor='measured': start the reconstruction from the cell's own post-settling state, so
    # the model FOLLOWS the cell where data exists and PROLONGS it after -- the stated
    # objective. 'jump' keeps the fully predictive path (no simulation needed).
    a_row = s.iloc[int(np.flatnonzero(m)[0])] if anchor == 'measured' else None
    pr = predict_history(tab, s.iloc[0], x[m], z=z, exclude=j_excl, anchor_row=a_row, **kw)
    if pr is None:
      n_declined += 1
      continue
    D_m, D_p = (lfac*dx)[m], pr['lfac']*pr['dx']
    for i, LL in enumerate(L[m]):
      rows.append(dict(k=k, Ri=Ri, dexpos=LL,
                       eD=abs(np.log(D_p[i]/D_m[i])),
                       erho=abs(np.log(pr['rho'][i]/rho[m][i])),
                       ep=abs(np.log(pr['p'][i]/p[m][i])),
                       eG=abs(np.log(pr['lfac'][i]/lfac[m][i]))))
  df = pd.DataFrame(rows)
  if verbose:
    tag = ('LOO' if loo else 'out-of-sample') + ('' if local_settle else ', IMPORTED settling')
    print(f'\nvalidate_model  table={table_key}  test={test_key}  '
          f'{SHELL_NAME.get(z, z)}  [{tag}]')
    if not len(df):
      print(f'  nothing to compare ({n_declined} cells declined by the table)')
      return df
    print(f'  {df.k.nunique()} cells, {len(df)} samples, R_i/R_0 in '
          f'[{df.Ri.min():.2f}, {df.Ri.max():.2f}]'
          + (f'   ({n_declined} declined: R_i outside the table)' if n_declined else ''))
    print('   R/R_i decade      Delta\'          rho             p               Gamma'
          '           2*dlnG (Ton)')
    edges = [(P.INJ_SAFE, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 1.10)]
    for a, b in edges:
      g = df[(df.dexpos >= a) & (df.dexpos < b)]
      if not len(g):
        continue
      cells = []
      for c in ('eD', 'erho', 'ep', 'eG'):
        cells.append(f'{100*g[c].median():5.2f}/{100*np.percentile(g[c], 84):5.2f}')
      print(f'   [{a:.3f},{b:.2f})  n={len(g):5d}  ' + '  '.join(cells)
            + f'  {200*g.eG.median():5.2f}/{200*np.percentile(g.eG, 84):5.2f}')
    print('   (median/84th %, per quantity)')
    for c, lab in (('eD', "Delta'"), ('erho', 'rho'), ('ep', 'p'), ('eG', 'Gamma')):
      print(f'   overall {lab:7s} median {100*df[c].median():6.2f}%   '
            f'84th {100*np.percentile(df[c], 84):6.2f}%   worst {100*df[c].max():7.2f}%')
  return df


def ablation(test_key='cooling_g100', z=4, ncells=NCELLS, table_key=TABLE_KEY, loo=None):
  '''
  What each ingredient of the closure is worth, on the observable-relevant quantities.
  Prices the design instead of arguing it.
  '''
  if loo is None:
    loo = (test_key == table_key)
  variants = (('full model', {}),
              ('no settling correction', dict(settle=False)),
              ('Gamma = const', dict(gamma_law='const')),
              ('p: TM adiabat, integrated', dict(adiabat='integrated')),
              ('p: TM adiabat, frozen', dict(adiabat='frozen')),
              ('p: tabulated alpha_T', dict(adiabat='temp')),
              ('no converged constant', dict(alpha_inf=False)))
  print(f'\nablation  table={table_key}  test={test_key}  {SHELL_NAME.get(z, z)}'
        + ('  [LOO]' if loo else ''))
  print(f'   {"variant":24s}  {"Delta\'":>14s}  {"rho":>14s}  {"p":>14s}  {"Ton":>14s}')
  out = {}
  for lab, kw in variants:
    df = validate_model(table_key, test_key, z, ncells, loo=loo, verbose=False, **kw)
    if not len(df):
      print(f'   {lab:24s}  (no samples)')
      continue
    out[lab] = df
    f = lambda c, s=1.: (f'{100*s*df[c].median():6.2f}/'
                         f'{100*s*np.percentile(df[c], 84):<6.2f}')
    print(f'   {lab:24s}  {f("eD"):>14s}  {f("erho"):>14s}  {f("ep"):>14s}  '
          f'{f("eG", 2.):>14s}')
  print('   (median/84th %, over the whole rarefaction-free window)')
  return out


# ---------------------------------------------------------------------------
def plot_model_validation(table_key=TABLE_KEY, test_key='cooling_g100', z=4, ncells=NCELLS,
    outdir=OUTDIR, ncurve=4):
  '''Predicted vs measured tracks for a few cells spread over R_i, and error vs R/R_i.'''
  os.makedirs(outdir, exist_ok=True)
  tab = load_table(table_key, z, ncells)
  loo = (test_key == table_key)
  tRi = tab['Ri']
  cells = _test_cells(test_key, z, ncells)
  cells = [cells[i] for i in np.unique(np.linspace(0, len(cells)-1, ncurve).astype(int))]
  fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), layout='constrained')
  cmap = plt.cm.viridis
  norm = plt.Normalize(min(c[4] for c in cells), max(c[4] for c in cells))
  for k, s, h, dex, Ri in cells:
    x, dx, rho, p, lfac, t = (a[:h+1] for a in P._cols(s))
    L = np.log10(x/x[0])
    m = (L >= P.INJ_SAFE) & (L <= dex - P.EDGE_EXCL)
    j = int(np.argmin(np.abs(tRi - Ri))) if loo else None
    pr = predict_history(tab, s.iloc[0], x[m], z=z, exclude=j)
    if pr is None:
      continue
    col = cmap(norm(Ri))
    for ax, ym, yp in ((axes[0, 0], (lfac*dx)[m]/(lfac*dx)[m][0],
                        pr['lfac']*pr['dx']/(pr['lfac'][0]*pr['dx'][0])),
                       (axes[0, 1], rho[m]/rho[m][0], pr['rho']/pr['rho'][0]),
                       (axes[1, 0], p[m]/p[m][0], pr['p']/pr['p'][0])):
      ax.plot(L[m], ym, color=col, lw=1.6, alpha=0.85)
      ax.plot(L[m], yp, color=col, lw=1.2, ls='--')
  for ax, lab in ((axes[0, 0], r"$\Delta'/\Delta'_0$"), (axes[0, 1], r'$\rho/\rho_0$'),
                  (axes[1, 0], r'$p/p_0$')):
    ax.set(xlabel=P.ABS_LABEL['R_Rinj'], ylabel=lab, yscale='log')
    P._style(ax)
  axes[0, 0].plot([], [], color=COL_INK, lw=1.6, label='measured')
  axes[0, 0].plot([], [], color=COL_INK, lw=1.2, ls='--', label='reconstructed')
  axes[0, 0].legend(fontsize=8, frameon=False)
  df = validate_model(table_key, test_key, z, ncells, loo=loo, verbose=False)
  if len(df):
    b = np.arange(P.INJ_SAFE, df.dexpos.max() + 0.05, 0.05)
    mid = 0.5*(b[1:] + b[:-1])
    for c, lab, ls in (('eD', r"$\Delta'$", '-'), ('erho', r'$\rho$', '--'),
                       ('ep', '$p$', '-.'), ('eG', r'$\Gamma$', ':')):
      med = [100*df[(df.dexpos >= b[i]) & (df.dexpos < b[i+1])][c].median()
             for i in range(len(b)-1)]
      axes[1, 1].plot(mid, med, ls, lw=1.6, label=lab)
    axes[1, 1].set(xlabel=P.ABS_LABEL['R_Rinj'], ylabel='median error (%)', yscale='log')
    axes[1, 1].legend(fontsize=8, frameon=False, ncol=2)
    P._style(axes[1, 1])
  fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=axes[0, :],
               label=r'$R_i/R_0$', pad=0.02)
  fig.suptitle(f'Pre-rarefaction reconstruction: table {table_key} -> test {test_key} '
               f'({SHELL_NAME.get(z, z)})', fontsize=11)
  path = os.path.join(outdir, f'model_validation_{test_key}_z{z}.png')
  fig.savefig(path, dpi=160, bbox_inches='tight')
  plt.close(fig)
  print(f'saved {path}')
  return path


def plot_three_way(table_key=TABLE_KEY, z=4, ncells=NCELLS, outdir=OUTDIR, ncurve=3,
    Ri_targets=(1.05, 1.5, 2.1)):
  """
  The three-way comparison: for cells at matched R_i/R_0,

     fiducial MEASURED      what a real run gives, over its short window
     RECONSTRUCTED          the model, from the fiducial cell's own jump state
     W=5 MEASURED           the same physical cell in a 5x wider shell, which stays
                            rarefaction-free ~4x further in R/R_i

  EACH QUANTITY IS SCALED BY ITS OWN DERIVED INDEX, so every panel is FLAT if the
  causal-contact law holds and the reference line in each is a level datum:

      rho * R^(2/gma)   the law says rho ~ R^(-2/gma)   (= R^1.2 exactly, gma = 5/3)
      p   * R^2         pressure continuity, p ~ R^-2
      Gamma             unscaled: coasting says Gamma -> const

  This is the sharpest form of the figure -- a residual trend in any panel is a departure from
  the derived law, read off against a horizontal line rather than inferred from a slope. Delta'
  is not shown: it carries the same information as rho, being its exact reciprocal through
  mass conservation (rho R^2 Delta' = const).

  Cells are matched on R_i/R_0, never on index: the runs differ in Nsh1 and in crossing extent.
  """
  os.makedirs(outdir, exist_ok=True)
  tab = load_table(table_key, z, ncells)
  fid = _test_cells('cooling_g100', z, ncells)
  # the W=5 comparison cell is picked from the DENSE list, not the sparse one. Near R_i ~ 1.5
  # the slope of rho R^1.2 CROSSES ZERO (measured: -0.0122 at R_i=1.455, -0.0007 at 1.518,
  # +0.0098 at 1.575), so a mismatch of only ~0.017 in R_i puts the two runs on opposite sides
  # of the crossover and the comparison curve appears flat where the fiducial decreases. The
  # sparse list gave exactly that; the dense list matches to ~0.003.
  wide = _test_cells(table_key, z, ncells, ks=table_cells(table_key, z, ncells))
  tab_f = with_settle(tab, 'cooling_g100', z, ncells)

  gi = 2./GMA_REF
  cols = (('rho', r'$\rho\,(R/R_{\rm inj})^{1.2}$', gi, 'log'),
          ('p',   r'$p\,(R/R_{\rm inj})^{2}$',                  2., 'log'),
          ('G',   r'$\Gamma$',                                   0., 'linear'))
  fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.0), layout='constrained',
                           gridspec_kw=dict(height_ratios=[2.1, 1]))
  styles = dict(fid=dict(color=COL_INK, lw=3.4, alpha=0.30, solid_capstyle='round'),
                rec=dict(color='#D55E00', lw=1.7, ls='--', dash_capstyle='round'),
                w5=dict(color='#0072B2', lw=1.1, alpha=0.95))

  for Rt in Ri_targets[:ncurve]:
    kf, sf, hf, dexf, Rif = min(fid, key=lambda c: abs(c[4] - Rt))
    kw, sw, hw, dexw, Riw = min(wide, key=lambda c: abs(c[4] - Rif))
    cf = [a2[:hf+1] for a2 in P._cols(sf)]
    cw = [a2[:hw+1] for a2 in P._cols(sw)]
    Lf, Lw = np.log10(cf[0]/cf[0][0]), np.log10(cw[0]/cw[0][0])
    mf = (Lf >= P.INJ_SAFE) & (Lf <= dexf - P.EDGE_EXCL)
    mw = (Lw >= P.INJ_SAFE) & (Lw <= dexw - P.EDGE_EXCL)
    if mf.sum() < 5 or mw.sum() < 5:
      continue
    xg = np.geomspace(cf[0][0]*10.**P.INJ_SAFE, cf[0][0]*10.**(dexw - P.EDGE_EXCL), 400)
    pr = predict_history(tab_f, sf.iloc[0], xg, z=z)
    if pr is None:
      continue
    Lr = np.log10(xg/cf[0][0])
    raw = dict(fid={'rho': cf[2], 'p': cf[3], 'G': cf[4]},
               w5={'rho': cw[2], 'p': cw[3], 'G': cw[4]},
               rec={'rho': pr['rho'], 'p': pr['p'], 'G': pr['lfac']})
    ann = None
    for j, (key, lab, pw, _) in enumerate(cols):
      yf = raw['fid'][key][mf]*10.**(pw*Lf[mf])
      yw = raw['w5'][key][mw]*10.**(pw*Lw[mw])
      yr = raw['rec'][key]*10.**(pw*Lr)
      yf, yw, yr = yf/yf[0], yw/yw[0], yr/yr[0]
      axes[0, j].plot(Lf[mf], yf, **styles['fid'])
      axes[0, j].plot(Lr, yr, **styles['rec'])
      axes[0, j].plot(Lw[mw], yw, **styles['w5'])
      axes[0, j].plot([Lf[mf][-1]], [yf[-1]], 'o', ms=5, color=COL_INK, zorder=6)
      # residuals: the scale factor is common to model and data and cancels exactly
      axes[1, j].plot(Lf[mf], 100*(np.interp(Lf[mf], Lr, yr)/yf - 1.),
                      color=COL_INK, lw=1.8, alpha=0.55)
      axes[1, j].plot(Lw[mw], 100*(np.interp(Lw[mw], Lr, yr)/yw - 1.),
                      color='#0072B2', lw=1.1)
      if j == 0:
        ann = (Lw[mw][-1], yw[-1])
    if ann:
      axes[0, 0].annotate(f'$R_i/R_0$={Rif:.3f}' + (f' (W5 {Riw:.3f})'
                          if abs(Riw - Rif) > 0.005 else ''), xy=ann, xytext=(-4, -9),
                          textcoords='offset points', fontsize=7, color=COL_MUTED,
                          ha='right', va='top')

  for j, (key, lab, pw, sc) in enumerate(cols):
    # every panel's derived asymptote is FLAT by construction of the scaling
    axes[0, j].axhline(1., color='#009E73', lw=1.3, ls=(0, (5, 2)), zorder=1)
    axes[0, j].set(ylabel=f'{lab}  (normalised at injection)', yscale=sc)
    axes[1, j].set(xlabel=P.ABS_LABEL['R_Rinj'],
                   ylabel='reconstruction / measured $-1$ (%)')
    axes[1, j].axhline(0., color=COL_MUTED, lw=0.8)
    for ax in (axes[0, j], axes[1, j]):
      ax.axvline(0.50, color=COL_MUTED, lw=1., ls=':')
      P._style(ax)
  axes[0, 0].annotate("fiducial's longest reach", xy=(0.50, 0.06),
                      xycoords=('data', 'axes fraction'), rotation=90, fontsize=7,
                      color=COL_MUTED, ha='right', va='bottom')
  hnd = [plt.Line2D([], [], **styles['fid'], label='fiducial, measured (dot = window end)'),
         plt.Line2D([], [], **styles['rec'], label='reconstructed, from the fiducial jump state'),
         plt.Line2D([], [], **styles['w5'], label='W=5, measured'),
         plt.Line2D([], [], color='#009E73', lw=1.3, ls=(0, (5, 2)),
                    label=r'derived asymptote (flat by construction)')]
  axes[0, 0].legend(handles=hnd, fontsize=8, frameon=False, loc='lower left')
  fig.suptitle('Pre-rarefaction hydro, each quantity scaled by its DERIVED index so the '
               f'asymptote is flat  ({SHELL_NAME.get(z, z)})', fontsize=11)
  path = os.path.join(outdir, f'three_way_z{z}.png')
  fig.savefig(path, dpi=160, bbox_inches='tight')
  plt.close(fig)
  print(f'saved {path}')
  return path


# --- follow-and-prolong ----------------------------------------------------------------------
PRERAR_PTS_PER_DEC = 64   # sampling of the synthetic tail, points per decade of R. Matches
                          # working_cooling_data.NORAR_PTS_PER_DEC so the two halves of a
                          # worldline are sampled the same way.


def prerar_history(shocked, tab, z=4, x_end=None, pts_per_dec=PRERAR_PTS_PER_DEC,
    edge=None, **kw):
  """
  A cell's worldline that FOLLOWS the simulation where it exists and PROLONGS it after: the
  real rows out to the end of the rarefaction-free window, then synthetic rows continuing on
  the reconstruction law to x_end.

  This is the pre-rarefaction mirror of working_cooling_data.norar_history, and it exists
  because a table built from neighbouring cells CANNOT also follow a cell whose behaviour
  differs from its neighbours -- the CD-adjacent cell being the case in point, off-trend by
  86-97 sigma in its settling and 3-6 sigma in alpha_D. Following it exactly requires its own
  data; only the extension needs a law. So:

      real rows                   every cell followed EXACTLY, idiosyncratic ones included
      synthetic tail              predict_history, anchored on the LAST REAL ROW

  Returns (history, info) with info: status ('ok' | 'no_extension' | 'declined'), h (last real
  row), n_syn, x_h, x_end -- the same contract as norar_history, so the two can be composed.

  WHICH LAW THE TAIL USES: predict_history continues the pre-rarefaction law (alpha tables ->
  the causal-contact asymptote, rho ~ R^-1.2, p ~ R^-2) for the WHOLE prolongation, and that is
  correct at every radius here -- it is not a near-field approximation to be handed over out of.
  The cell is inside a shocked shell and interacting with it, so it never becomes the
  non-interacting fluid that rho ~ R^-2 describes. Do NOT compose this tail with norar_history's
  coasting law at some R_cross: that law belongs to the decompressed post-rarefaction shell,
  which is exactly the history this counterfactual removes.
  """
  edge = P.EDGE_EXCL if edge is None else edge
  x = shocked.x.to_numpy(dtype=float)
  h, dex = P.prerar_window(shocked)
  info = dict(status='no_extension', h=int(h), n_syn=0,
              x_h=float(x[h]) if h < len(x) else float(x[-1]),
              x_end=float(x[-1]) if len(x) else np.nan)
  if len(shocked) < 3 or h < 2:
    return shocked, info
  # last real row inside the rarefaction-free window (edge-excluded)
  L = np.log10(x/x[0])
  keep = np.flatnonzero((L <= dex - edge) & (np.arange(len(x)) <= h))
  if keep.size < 3:
    return shocked, info
  j = int(keep[-1])
  info.update(h=j, x_h=float(x[j]))
  x_end = float(x[-1]) if x_end is None else float(x_end)
  if x_end <= x[j]*(1. + 1e-9):
    return shocked.iloc[:j+1], info          # nothing to prolong
  n = max(2, int(np.ceil(np.log10(x_end/x[j])*pts_per_dec)) + 1)
  x_ext = np.geomspace(x[j], x_end, n)[1:]   # ends EXACTLY on the target radius
  cols = predict_history(tab, shocked.iloc[0], x_ext, z=z, anchor_row=shocked.iloc[j], **kw)
  if cols is None:
    info['status'] = 'declined'
    return shocked.iloc[:j+1], info
  # carry every parent column, as norar_history does, so get_variable and the sub-cell path
  # keep working on the result
  ext = pd.DataFrame({c: np.full(len(x_ext), shocked.iloc[j][c]) for c in shocked.columns})
  for c, v in cols.items():
    if c in ext:
      ext[c] = v
  if 'Sd' in ext:
    ext['Sd'] = 0.                           # synthetic rows are never shocked
  ext.index = shocked.index[-1] + 1 + np.arange(len(x_ext))
  out = pd.concat([shocked.iloc[:j+1], ext])
  out.attrs = shocked.attrs
  info.update(status='ok', n_syn=len(x_ext), x_end=x_end)
  return out, info


# --- article figure -------------------------------------------------------------------------
ARTICLE_CELLS = (1.05, 1.86)   # R_i/R_0 of the representative FIDUCIAL cells. Chosen to sit on
                               # OPPOSITE sides of the causal-contact asymptote (rho R^1.2 slope
                               # -0.150 and +0.046), and DELIBERATELY NOT ~1.5: that is where the
                               # slope crosses zero, so a cell there is maximally sensitive to
                               # R_i and its signal is closest to the per-cell noise (MAD
                               # ~0.010-0.015). It misled twice during development.
                               # NOR the CD-adjacent cell at R_i = 1.000: it is off-trend by
                               # 86-97 sigma in its settling and 3-6 sigma in alpha_D, and
                               # reconstructs with a 1.2% ramp against 0.07-0.20% for its
                               # neighbours. "Longest window" selection picks it first (it is
                               # shocked first), so it must be excluded explicitly.

def plot_article(table_key=TABLE_KEY, z=4, ncells=NCELLS, outdir=OUTDIR,
    Ri_targets=ARTICLE_CELLS, test_key='cooling_g100', dex_max=None):
  """
  Article figure: the hydro evolution of a couple of representative cells, simulation versus
  semi-analytic reconstruction.

  Shown on the FIDUCIAL run and reconstructed OUT-OF-SAMPLE -- the alpha tables come from
  cooling_g100_w5, a different simulation with 5x wider shells and 2x coarser cells, and only
  the 1-D settling correction is taken from the fiducial itself (with_settle). So the dashed
  curves are predictions for a run the table was not built from.

  EACH QUANTITY IS SCALED BY ITS DERIVED INDEX, so the causal-contact law is a HORIZONTAL line
  in every panel and any departure from it is read against a level datum rather than inferred
  from a slope:

      rho * R^(2/gma) = rho * R^1.2   (gma = 5/3, the fully cooled limit)
      p   * R^2                       (pressure continuity across the causally connected layer)
      Gamma                           (unscaled; coasting says Gamma -> const)

  Delta' is omitted: it is rho's exact reciprocal through mass conservation (rho R^2 Delta' =
  const), so it would add a panel without adding information.
  """
  os.makedirs(outdir, exist_ok=True)
  tab = with_settle(load_table(table_key, z, ncells), test_key, z, ncells)
  cells = _test_cells(test_key, z, ncells)

  gi = 2./GMA_REF
  # Linear y throughout. Log-y resolves the crash (4-6 decades) but that is not the point of
  # the figure -- it also drags the eye onto floor-limited, mesh-artifact tails. Linear keeps
  # the contrast the figure is about: the crash as a cliff against a reconstruction that holds.
  cols = (('rho', r'$\rho\,(R/R_{\rm inj})^{1.2}$', gi, 'linear'),
          ('p',   r'$p\,(R/R_{\rm inj})^{2}$',        2., 'linear'),
          ('G',   r'$\Gamma/\Gamma_{\rm inj}$',      0., 'linear'))
  ccol = ('#0072B2', '#D55E00')          # Okabe-Ito blue / vermillion, CVD-safe
  fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.6), layout='constrained')
  axes = np.atleast_2d(axes)

  for ci, Rt in enumerate(Ri_targets):
    k, s_, h, dex, Ri = min(cells, key=lambda c: abs(c[4] - Rt))
    # FULL measured history -- past the handover too, so the rarefaction CRASH is shown. That
    # crash is the whole point of the counterfactual: it is what the reconstruction is being
    # asked to replace.
    cf = P._cols(s_)
    Lf = np.log10(cf[0]/cf[0][0])
    mf = (Lf >= P.INJ_SAFE) if dex_max is None else ((Lf >= P.INJ_SAFE) & (Lf <= dex_max))
    # rarefaction-free window: where the comparison is meaningful
    m = (Lf >= P.INJ_SAFE) & (Lf <= dex - P.EDGE_EXCL)
    if m.sum() < 10 or mf.sum() < 10:
      continue
    # reconstruction PROLONGED well past the handover, on the same radial grid as the data
    xg = cf[0][mf]
    pr = predict_history(tab, s_.iloc[0], xg, z=z,
                         anchor_row=s_.iloc[int(np.flatnonzero(m)[0])])
    if pr is None:
      continue
    meas = {'rho': cf[2][mf], 'p': cf[3][mf], 'G': cf[4][mf]}
    rec = {'rho': pr['rho'], 'p': pr['p'], 'G': pr['lfac']}
    Lr = Lf[mf]
    nres = int(m.sum())          # residuals only over the rarefaction-free stretch
    col = ccol[ci % len(ccol)]
    for jj, (key, lab, pw, sc) in enumerate(cols):
      ym = meas[key]*10.**(pw*Lr)
      yr = rec[key]*10.**(pw*Lr)
      nm = ym[0]
      axes[0, jj].plot(Lr, ym/nm, color=col, lw=3.2, alpha=0.35, solid_capstyle='round',
                       zorder=2)
      axes[0, jj].plot(Lr, yr/nm, color=col, lw=1.5, ls='--', dash_capstyle='round', zorder=3)
      # mark where the rarefaction reaches this cell
      jh = int(np.argmin(np.abs(Lr - (dex - P.EDGE_EXCL))))
      axes[0, jj].plot([Lr[jh]], [(ym/nm)[jh]], 'o', ms=5.5, mfc='white', mec=col, mew=1.4,
                       zorder=5)
      if jj == 0:
        # Label at the END of the reconstruction. On these LINEAR axes the crashed tails all
        # pile onto zero, so the asymptote is the only place the two cells are separated --
        # the reverse of the log-y case, where only the crash separated them.
        axes[0, 0].annotate(rf'$R_i/R_0={Ri:.2f}$', xy=(Lr[-1], (yr/nm)[-1]),
                            xytext=(-3, 5), textcoords='offset points', fontsize=8,
                            color=col, ha='right', va='bottom')

  for jj, (key, lab, pw, sc) in enumerate(cols):
    axes[0, jj].set(ylabel=lab, yscale=sc,
                    xlabel=r'$\log_{10}(R/R_{\rm inj})$')
    P._style(axes[0, jj])
  hnd = [plt.Line2D([], [], color=COL_INK, lw=3.2, alpha=0.35,
                    label='simulation (crashes when the rarefaction arrives)'),
         plt.Line2D([], [], color=COL_INK, lw=1.5, ls='--',
                    label='reconstruction, prolonged (no rarefaction)'),
         plt.Line2D([], [], marker='o', ls='none', mfc='white', mec=COL_INK, mew=1.4,
                    label='rarefaction reaches the cell')]
  fig.legend(handles=hnd, fontsize=9, frameon=False, ncol=3, loc='outside lower center')
  path = os.path.join(outdir, f'article_cells_z{z}.png')
  fig.savefig(path, dpi=200, bbox_inches='tight')
  plt.close(fig)
  print(f'saved {path}')
  return path


def main(table_key=TABLE_KEY, z_list=(4, 1), ncells=NCELLS):
  '''Build the table, then validate LOO on the basis run and out-of-sample on the others.'''
  out = {}
  for z in z_list:
    build_table(table_key, z, ncells)
    out[z] = dict(
        loo=validate_model(table_key, table_key, z, ncells, loo=True),
        fid=validate_model(table_key, 'cooling_g100', z, ncells),
        w2=validate_model(table_key, 'cooling_g100_w2', z, ncells))
    ablation('cooling_g100', z, ncells, table_key)
  for z in z_list:
    plot_three_way(table_key, z, ncells)
    plot_article(table_key, z, ncells)
    for tk in ('cooling_g100', table_key):
      plot_model_validation(table_key, tk, z, ncells)
  return out


if __name__ == '__main__':
  main()
