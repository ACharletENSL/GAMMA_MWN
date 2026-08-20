# -*- coding: utf-8 -*-
# @Author: acharlet

'''
Do the computed shell spectra actually carry the synchrotron asymptotes that every
measurement in this project HOLDS?

phys_functions.granot_sari_syn hardcodes the F_nu indices 1/3 and -p/2; spectral_breaks
.fit_segments holds the matching nuFnu slopes 4/3 and 1-p/2 and selects their windows by
thresholding the local slope around those very values. So the whole break-measurement chain
assumes the asymptotes and none of it tests them -- a systematic in the slopes would be
absorbed silently into the break positions, which are what every nu_c(T) result rests on.

This module answers that with spectral_breaks.free_slopes: windows placed GEOMETRICALLY a
factor FREE_DFAC away from breaks located by an independent GS02 fit, a free line fitted in
each, no slope value entering the selection. Run over every time bin of every cached sweep
point, on both shells.

  slope_table          - per sweep point: the three slopes over all measured bins, and the
                         electron index p the high segment implies vs the p that went in.
  epoch_table          - the same split into rise / crossing / high-latitude tail: how the
                         slopes EVOLVE, which is where the shell integration shows up.
  regime_table         - everything pooled by the regime DETECTED in each bin (VFC/FC/MC/SC,
                         from the fitted mid slope); the two smoothing exponents from
                         spectral_breaks.fit_smoothing_held (granot_sari_syn refitted with
                         the crossings and all three slopes held); and the merged-break
                         alternative fit_single_break scored against the two-break form,
                         split on-axis vs high-latitude for the MC bins.
  prescription_check   - the test the table itself needs: freeze s at the regime median and
                         refit, so the residual against the free fit IS the cost of tabulating.
  prescription_summary - the recommended prescription assembled from that, split by epoch
                         where a single median is shown to be inadequate.
  convergence_check    - a_lo against the window standoff, against a synthetic
                         granot_sari_syn control whose a_lo is 4/3 by construction. This is
                         what separates residual break curvature from a real deviation.
  plot_slope_evolution - the three slopes vs bar{T}/bar{T}_f, one colour per regime, both
                         shells side by side.
  hle_analysis         - separate entry point (doubles the tracker cost): what high-latitude
                         emission does to the shape, by running the frozen-parameter test
                         against time on BOTH methods. Only with the rarefaction cut is the
                         late flux purely high-latitude, which is what makes the separation
                         possible.

Ground truth for the window placement is track_breaks_gs02 on the cached sweeps; no
spectrum is recomputed.

WHAT IT SHOWS (last full run: cooling_g100, METHOD='data', p=2.5, both shells, all 450 time
bins of all 8 sweep points, dfac=10)

  a_hi is the strong result and the only true end-to-end one. Bin-weighted over every
  measured bin it comes back at -0.2529 (RS) and -0.2513 (FS) against the expected -0.2500,
  i.e. the electron index reads back as p = 2.506 / 2.503 against the 2.5 that went in. The
  per-point scatter is 0.003-0.011 and there is no trend with cooling regime, epoch or
  shell: the panel is flat within +-0.01 over four decades of bar{T}. Nothing about this is
  circular -- p enters the emission path at the electron distribution and is recovered from
  the integrated, Doppler-summed spectrum.

  a_mid sits on its asymptote for as long as the shell is emitting on-axis, at 0.50-0.53
  fast-cooling and 0.19-0.25 slow-cooling (asymptotes 0.5 and (3-p)/2 = 0.25), and moves
  only in the high-latitude tail. Two DIFFERENT things move it there and the table reports
  their sum: genuine softening (the fast-cooling points, where it drifts by -0.005 to -0.16),
  and, in the slow-cooling points, the fact that nu_c has fallen through nu_m before the
  tail so the expected asymptote itself changes from (3-p)/2 to 1/2 (logr=+1 RS moves
  +0.21, logr=0 FS +0.38, which is mostly that). Read the per-epoch a_mid against the
  regime, not as one number.

  a_mid also agrees with the GS02 template's OWN fitted beta_mid to a mean 0.004 (RS) /
  0.000 (FS) on the rise and 0.009 / 0.002 through the crossing -- two independent
  measurements of the same quantity, one of which never touches the template. They part
  company only in the HLE (mean +0.030 / +0.036, worst 0.20 / 0.25), where the mid segment
  is genuinely curved and a straight line and a template asymptote stop being the same
  object.

  a_lo is recovered wherever it is measurable at all, at +1.3220 (RS) / +1.3200 (FS) against
  4/3. The residual ~0.012 is the window's own break curvature, not the spectrum: it matches
  the -0.0104 / -0.0142 measured independently on the peak spectra at the same standoff, and
  walking the window out drives it to 1.3333 exactly by dfac=300, at the same rate as a
  synthetic granot_sari_syn whose a_lo IS 4/3 by construction.

  The one thing that looked like a real slope change is not one. Left ungated, a_lo falls
  steadily through the high-latitude tail -- to ~1.06 by bar{T}/bar{T}_f ~ 7 on the FS, and
  it correlates with the fitted window width at r = +0.96. It is a band limit: the HLE
  superposition smears the lower break far below any single-zone smoothing, so the nu^(4/3)
  asymptote is pushed off the bottom of the frequency window. The synthetic control with s1
  dropped to ~0.4-0.5 and the band bottom 2.4 decades below the break reproduces the
  measured values. Those bins are now DECLINED rather than reported (lo_converged, see
  spectral_breaks.FREE_CONV_TOL): 524 of 1853 RS bins and 812 of 2064 FS bins. The FS loses
  more, and earlier, because its shell goes dark sooner.

  Corollary worth keeping: three independent routes now say the lower break is smoother
  than the s1 = 1.3 held in fit_gs02_spectrum -- the free-s fits (which want 0.53-0.76 in
  the tails), this module's convergence rate (tracking the s1 = 1.0 control rather than the
  1.3 one), and the HLE band-limit reconstruction above.

BY DETECTED REGIME (regime_table; RS / FS, bins pooled across sweep points)

  a_hi is flat over every regime -- -0.2544/-0.2523 fast, -0.2469/-0.2490 slow,
  -0.2559/-0.2548 very-fast, -0.2539/-0.2466 at the crossing, i.e. p = 2.49-2.51 everywhere
  against 2.5 in. a_lo likewise, 1.312-1.326 wherever it is measurable (never in VFC, which
  has no nu^(4/3) segment in band by construction). Neither slope carries regime information.

  a_mid does, and it lands where it should: VFC +0.498/+0.501 against the 1/2 its single
  break joins to -p/2; SC +0.234/+0.240 against (3-p)/2 = 0.25; MC +0.333/+0.325, squarely
  between the two asymptotes, which is what makes it MC. Only FC sits off, at +0.531/+0.574
  against 0.5, and with by far the widest bin-to-bin scatter (sd 0.087/0.094) -- because the
  FC bins pool the on-axis pulse with the softening high-latitude tail.

  The SMOOTHING now comes from fit_smoothing_held -- granot_sari_syn refitted with the
  crossings and all three slopes held, so nothing passes through an intercept. It replaced
  the deficit estimator of _free_smoothing, which did not converge. Measured as the relative
  spread of s over dfac = 3, 10, 30, sampled across all time bins:

      s1     new 3.3-7.1% per regime   vs   old 11.2-68.2%   -- FIXED, s1 is a measurement
      s2     new 0.9-21.8%             vs   old  7.2-24.6%   -- better where the breaks are
             well separated (VFC 0.9%, SC 8.9/9.7%), no better at the crossing (MC ~21%),
             where the two turnovers overlap and the upper crossing cannot be pinned

  s1 survives because it is insensitive to an error in the held break (a 30% displacement of
  b_lo moves it under 4%); s2 does not, being limited by the room between b_hi and nuM.
  Synthetic recovery through the full pipeline: s1 to <=3.5%, s2 to ~5-17%.

      regime    s1 (lower)              s2 (upper)              fit rms
      VFC       -- (single break)       1.72 / 1.82             0.012 / 0.014
      FC        0.73 / 0.69             1.07 / 0.79             0.027 / 0.030
      MC        0.59 / 0.25             1.16 / 0.50             0.024 / 0.051
      SC        1.05 / 1.04             1.68 / 1.66             0.011 / 0.012

  (VFC's rms improved from 0.018/0.019 once the upper crossing was taken from the held
  high line -- see below. The MC row is superseded by the merged-break shape further down.)

  The ordering is the same on both shells -- MC < FC < SC for the lower break, and the
  smooth pair (MC, FC) against the sharp pair (SC, VFC) for the upper. Slow cooling is both
  the sharpest-broken and by far the best-described regime (rms 0.011, a third of the
  others'); the FS crossing is the worst (0.051), i.e. the GS02 shape does not really
  describe an FS spectrum caught mid-crossing.

  The practical consequence: the held (1.3, 2.0) is too SHARP essentially everywhere. Every
  measured s1 is at or below 1.05 and every measured s2 at or below 1.87, with slow cooling
  the closest and the crossing not close at all. That is now a statement worth acting on for
  s1, which is converged; for s2 treat it as indicative outside VFC and SC.

  The upper crossing is now located with a_hi HELD at 1-p/2 rather than fitted (free_slopes'
  b_hi_held) -- legitimate only because this module verified that asymptote first. It makes
  the high line a one-parameter fit and removes the slope-intercept covariance that dominates
  a short window: on synthetics the upper-crossing error falls from -58/-34/-19% to
  -17/-7.5/-4.7% and s1 recovery improves from 3.5% to 1.5% median. a_lo is deliberately NOT
  held the same way -- it is the asymptote often NOT reached inside the band, and holding it
  there displaces b_lo (+18.5% error on s1 at s1 = 0.5).

THE MC SPECTRA NEED A DIFFERENT SHAPE (fit_single_break)

  The two-break form gives out in the marginal regime, and not gracefully. An MC spectrum's
  local slope runs 1.33, 1.33, 1.24, 0.83, -0.01, -0.24: from the low asymptote straight to
  the high one over ~2 decades, never plateauing. There is no mid segment to measure -- a_mid
  comes back NaN -- and below a break separation of ~100 the mid window (b_lo*dfac, b_hi/dfac)
  is empty by construction, so the model is not merely imprecise there but unusable.

  fit_single_break replaces it with ONE broad break, nu^(4/3) straight to nu^(1-p/2), both
  asymptotes held at their now-verified values and no mid slope at all: three parameters
  (position, smoothing, scale) against the two-break form's four or five. Against the GS02
  two-break fit on the MC bins:

      MC, on-axis (bar{T} <= bar{T}_f)   RS 16/16 bins, rms 0.0268 -> 0.0102 (gain 2.62)
                                         FS 21/21 bins, rms 0.0274 -> 0.0108 (gain 2.54)
      MC, high-latitude                  RS 138/251, gain 1.01 -- a tie, both poor (rms ~0.09)
                                         FS 192/192, gain 1.90, but rms still 0.058

  So the merged break is a solved case: the single broad break wins in every on-axis MC bin
  on both shells, by a factor 2.5-2.6 in rms, and its smoothing is tight enough to quote (see
  the table below). The high-latitude crossing is not solved by either shape -- the transition
  there is much broader again and neither form describes it.

  Its `s` is in the SAME convention as s1, s2: fit_single_break writes the break in the GS02
  two-term form, as granot_sari_syn's single-break branch with beta_lo_single = 1/3 instead of
  the -1/2 that gives VFC. Larger is sharper throughout. (This shape was originally written
  with smooth_bpl_apy, whose exponent runs the other way; the two are the same function under
  s = 1/[sigma (1/3 + p/2)], so the old sigma = 1.08 and 4.38 are s = 0.585 and 0.144.
  Converting changed no residual.) Comparing s ACROSS breaks is still not a comparison of
  width: this one spans a slope change of 1/3 + p/2 = 1.58 against 0.83 (FC) or 1.08 (SC) for
  the lower break of the two-break form.

DO THE TABULATED VALUES ACTUALLY FIT? (prescription_check / prescription_summary)

  Every number above came from a fit with s FREE per spectrum, which is not what a table
  claims. Freezing s at the regime median and refitting each bin with everything else
  identical -- same breaks, same beta_mid, same nu_M -- measures the cost of tabulating
  directly. Two results, one reassuring and one that changes the table.

  Pooling the two shells is FREE. The per-shell and across-shell medians give residuals equal
  to 4 decimal places in every regime (e.g. VFC 0.0120 vs 0.0121, SC 0.0116 vs 0.0116). There
  is no measurable RS/FS difference in the smoothing, so one number per case is enough and the
  RS/FS pairs quoted above are over-specified.

  Regimes must be split by EPOCH, not by shell. Frozen at a single median over all its bins,
  FC costs +0.0122 dex on the RS and +0.0209 on the FS, failing 16.5%/50.9% of bins; MC fails
  100% of them, because its bin count is dominated by the post-crossing bins whose smoothing
  (s ~ 0.14) is nothing like the on-axis value (s ~ 0.59). Splitting on bar{T} <= bar{T}_f fixes both:

      case          s1     s2   s(1brk)  rms free  rms held    cost   %bad   verdict
      VFC all       --    1.77    --      0.0132    0.0133   +0.0000   1.3%  use
      FC on-axis   1.18   1.49    --      0.0177    0.0179   +0.0002   0.0%  use
      FC HLE       0.63   0.86    --      0.0302    0.0415   +0.0113  25.4%  use with care
      MC on-axis    --     --    0.585    0.0105    0.0111   +0.0006   0.0%  use
      MC HLE        --     --    0.144    0.0656    0.1191   +0.0535 100.0%  DO NOT USE

  (the s(1brk) column is only the model where the merged shape is used, i.e. the MC rows; it
  is still computed for the others and printed, but there the two-break s1, s2 are the fit)
      SC all       1.05   1.67    --      0.0118    0.0124   +0.0007  15.1%  use with care

  So the tabulated values are usable, and essentially free, for VFC, SC, and the on-axis half
  of FC and MC -- the cost of freezing is 0.0000-0.0007 dex against fitting s per spectrum,
  and on-axis FC and MC fail no bin at all. Note that the on-axis FC values (1.18, 1.49) are
  nothing like the all-bin FC medians (0.73, 0.98) reported further up: the unsplit FC row is
  an average over two different shapes and should not be quoted.

  The high-latitude tail is where it stops. SC and FC HLE carry a real tail of poorly fitted
  bins (15-25%) even though their medians pass, and MC HLE cannot be described at all -- which
  is the same conclusion the merged-shape comparison reached from the other direction.

WHAT HIGH-LATITUDE EMISSION ACTUALLY DOES (hle_analysis)

  The post-crossing degradation above was assumed to be a high-latitude effect. It is not.
  Running the same frozen-parameter test against bar{T}/bar{T}_f on BOTH methods separates the
  two things that happen after crossing, because with rar_cut the shell is dark by
  bar{T} ~ 2 and its late flux is purely high-latitude, while the reference method has cells
  radiating out to bar{T} ~ 650:

      bar{T}/bar{T}_f      data: rms held / %bad      data_rarcut: rms held / %bad
      0-1                  0.012-0.014 /  1%          0.012-0.014 /  1%
      2-3                  0.032       / 26%          0.015       /  0%
      5-10                 0.054       / 51%          0.016       /  0%
      30-100               0.539       /100%          0.016       /  0%
      >100                 0.575       /100%          0.024       /  0%

  With the shell switched off, the on-axis smoothing keeps describing the spectra out to
  bar{T}/bar{T}_f > 100 and fails NOT ONE BIN at any epoch. The fitted s saturates there:
  fitting log s = a + k log(bar{T}/bar{T}_f) past successive floors gives k = +0.00 to +0.05
  for every regime and floor (SC s1 +0.02..+0.05, s2 -0.02..-0.04; MC +0.00..+0.01; VFC
  +0.01..+0.02), i.e. the spectral SHAPE freezes, exactly as it should for a thin shell that
  has stopped emitting -- the spectrum then slides in frequency and flux but keeps its form.

  All of the degradation belongs to the CONTINUED EMISSION instead. In the reference method s
  collapses (FC s2 k = -0.47..-0.75, SC -0.6..-0.7) and both breaks broaden together, s1 from
  1.07 to 0.25 and s2 from 1.69 to 0.52. What widens them is the spread in cooling stage and
  emission radius among cells that are still radiating at one observed time -- not the angular
  superposition, which on its own preserves the shape.

  THE COMPARISON HAD TO BE MATCHED BY REGIME FIRST. Pooled over regime, the two columns hold
  different SHAPE-CLASS populations: fit_gs02_spectrum decided VFC on where its own fitted lower
  break landed relative to the band, and in these tail spectra that break is unconstrained, so
  the class flipped on noise (spectra with edge slopes 0.77 and 0.84 -- neither consistent with
  the 1/2 a VFC claim needs -- were called VFC under 'data' and FC under 'data_rarcut'). That is
  now gated on the data itself (spectral_breaks.edge_slope); deep fast cooling is untouched and
  the two methods agree exactly there (362/362 VFC bins at log10(gc/gm) = -5, edge slope 0.501).
  Matched FC-to-FC, SC-to-SC and MC-to-MC the result stands, and is cleaner:

      regime   bar{T}/bar{T}_f     rms data / rms rarcut
      FC       0-1                 1.00   (identical bins, identical residual)
      FC       3 / 5               2.47 / 2.72
      SC       3 / 5 / 100         6.60 / 10.7 / 13.7
      MC       3 / 10 / 100        20.7 / 32.8 / 36.4

  The band limit is ruled out as the cause two ways. (i) Both s1 and s2 fall together, and s2
  does not go through the low-frequency band at all. (ii) A synthetic spectrum of constant s
  slid down the band by four decades returns s low by only 2.6-6.8%, against the ~70% collapse
  measured. An earlier third argument -- that the band shrinks as much under rar_cut while s
  stays flat -- was WITHDRAWN: it does not discriminate, since a shape-class difference predicts
  the same pattern.

  The on-axis reference values are identical between the two methods (FC 1.182/1.492,
  SC 1.058/1.686, MC 0.585, VFC 1.853), confirming the cut does nothing before crossing and the
  comparison is like-for-like.

  Consequence for the table above: its post-crossing rows are a property of the reference
  method, not of high-latitude emission. Under a rarefaction cut-off the on-axis row is valid
  at all times.

LOW-ENERGY SOFTENING IN THE TAIL (low_energy_softening)

  The clearest single symptom, and the one that connects to the paper's own result. Measured as
  a_edge, the nuFnu slope over the lowest decade of band -- used rather than the fitted low
  segment because in the tail NEITHER run reaches the nu^(4/3) asymptote in band, so anything
  requiring it is undefined exactly where the comparison is wanted:

      bar{T}/bar{T}_f    full history    rar_cut     softening
      0-1                1.333           1.333       +0.000
      2-3                1.216           1.287       +0.071
      5-10               1.027           1.333       +0.306
      10-30              0.793           1.333       +0.541
      30-100             0.723           1.327       +0.604
      >100               0.510           1.284       +0.774

  The cut-off run stays pinned at 4/3 at EVERY epoch: switching the cells off freezes the
  low-frequency end at its uncooled asymptote. The full-history run softens monotonically from
  the moment the shock finishes crossing, and by bar{T}/bar{T}_f > 100 reaches 0.51 -- i.e.
  essentially the fast-cooling 1/2. That is the physical content: with cells still radiating,
  the low-frequency end of the late spectrum is dominated by material that is still cooling,
  so it carries the cooled slope rather than the uncooled one.

  Cross-check worth keeping: the median softening past 2 bar{T}_f is +0.424, which reproduces
  the TIME-INTEGRATED difference of 0.433 measured independently in
  figures/rarcut_compare/fluence_low_slopes.csv (index 0.90 full against 1.333 cut). The
  fluence-level softening that the paper reports is therefore built up entirely after crossing,
  and this locates it in time.

  Not done: only p = 2.5 has ever been run, so the p-dependence of a_hi is untested, and
  only METHOD='data' -- 'data_rarcut' cuts the shell off early and would mostly remove the
  HLE bins this module has the most to say about. s2 at the crossing still has no converged
  estimator, and no shape yet fits the high-latitude MC spectra.

Example use in command line:
  python -c "import slope_validation as V; V.main()"
  python -c "import slope_validation as V; V.main(zlist=(4,))"
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from environment import GAMMA_dir
from phys_functions import granot_sari_syn, syn_cutoff_R
import spectral_breaks as sb
import sweep_gammacm as swp

OUTDIR = os.path.join(GAMMA_dir, 'bin', 'Tools', 'figures', 'slope_check')
KEY = 'cooling_g100'
METHOD = 'data'                # the reference computation, see sweep_gammacm.DEFAULT_METHOD
Z_RS, Z_FS = 4, 1
CMAP = plt.cm.viridis          # sequential: log10(gma_c/gma_m) is an ordered magnitude

# Epoch windows, in bar{T}, reusing the split fit_break_evolution already applies to the
# breaks so the two are read on the same phases:
#   rise      swp.RISE_WIN, the shock crossing at ~constant hydro
#   crossing  [RISE_WIN[1], bar{T}_f], the cells expanding
#   HLE       past the rarefaction cut-off: the shell is dark, the spectrum frozen and
#             sliding with the Doppler factor
EPOCHS = ('rise', 'crossing', 'HLE')
MIN_BINS = 6                   # an epoch summarised from fewer bins than this is not reported


def _ensure_outdir():
  os.makedirs(OUTDIR, exist_ok=True)


def load_side(z=Z_RS, key=KEY, method=METHOD, dfac=sb.FREE_DFAC):
  '''
  Cached sweep points of one shell, each with its GS02 break track and the free-slope
  measurement on every time bin. The track is the expensive part (~6 s a point).
  Returns a list of dicts ordered by log10(gamma_c/gamma_m).
  '''
  outdir = swp.method_outdir(method, key, z)
  res = swp.load_sweep(outdir)
  if not res:
    raise FileNotFoundError(f'no cached sweep in {outdir} -- run sweep_gammacm.main first')
  barT_f = swp.exit_onset_barT(key, z=z)
  barT_off = swp.rarefaction_off_barT(key, z=z)
  out = []
  for r in sorted(res, key=lambda x: x['log10ratio']):
    tr = swp.track_breaks_gs02(r, barT_swap_max=(barT_off[1] if barT_off else barT_f))
    fs = sb.track_free_slopes(r, tr=tr, dfac=dfac)
    out.append(dict(r=r, tr=tr, fs=fs, logr=r['log10ratio'], z=z, barT_f=barT_f,
                    barT_off=barT_off, psyn=r['env'].psyn,
                    beta_mid=tr.get('beta_mid', np.full(len(fs['barT']), np.nan))))
    n = {k: int(np.isfinite(fs[k]).sum()) for k in ('a_lo', 'a_mid', 'a_hi')}
    print(f"  log10ratio={r['log10ratio']:+.1f} z={z}: bins measured "
          f"lo:{n['a_lo']} mid:{n['a_mid']} hi:{n['a_hi']} of {len(fs['barT'])}", flush=True)
  return out


def _epoch_mask(s, epoch):
  '''bar{T} mask of one epoch, on the same split fit_break_evolution uses'''
  b = s['fs']['barT']
  lo, hi = swp.RISE_WIN
  if epoch == 'rise':
    return (b >= lo) & (b <= hi)
  if epoch == 'crossing':
    return (b > hi) & (b <= s['barT_f'])
  if epoch == 'HLE':
    off = s['barT_off']
    return b >= (max(2., 1.2*off[1]) if off is not None else 2.)
  raise ValueError(f'unknown epoch {epoch!r}')


def _summary(v, m):
  '''mean/std/N of v over the mask m, on the finite samples only'''
  g = m & np.isfinite(v)
  if g.sum() < 1:
    return np.nan, np.nan, 0
  return float(np.mean(v[g])), float(np.std(v[g])), int(g.sum())


def _robust(v, m):
  '''
  median and 16-84 percentiles of v over the mask m. Used for the smoothing exponents,
  where mean/std is the wrong summary: s = ln2/deficit diverges as the measured deficit goes
  to zero, so a handful of near-zero-deficit bins put the sample standard deviation above the
  mean itself (z=1 slow cooling: mean 2.8, sd 5.3, median 1.6). The percentile range is what
  the distribution actually is.
  Returns (median, q16, q84, N).
  '''
  g = m & np.isfinite(v)
  if g.sum() < 1:
    return np.nan, np.nan, np.nan, 0
  q16, q84 = np.percentile(v[g], [16, 84])
  return float(np.median(v[g])), float(q16), float(q84), int(g.sum())


def slope_table(sides, verbose=True):
  '''
  Per sweep point, over ALL measured bins: the three free slopes against the values held
  everywhere else, and the electron index the high segment implies. p_hi vs the input p is
  the end-to-end check -- it passes through the whole emission path and the shell
  integration before being read back off the integrated spectrum.
  '''
  rows = []
  for s in sides:
    fs = s['fs']
    all_ = np.ones(len(fs['barT']), bool)
    # a_lo only where the asymptote is actually reached in band (see FREE_CONV_TOL); the
    # rest are lower bounds, counted separately rather than averaged in
    (lo, lo_s, n_lo) = _summary(fs['a_lo'], fs['lo_converged'])
    (md, md_s, n_md) = _summary(fs['a_mid'], all_)
    (hi, hi_s, n_hi) = _summary(fs['a_hi'], all_)
    n_lo_try = int(np.isfinite(fs['a_lo']).sum())
    rows.append(dict(z=s['z'], logr=s['logr'], n_lo=n_lo, n_lo_unconv=n_lo_try - n_lo,
                     a_lo=lo, a_lo_sd=lo_s, da_lo=lo - fs['a_lo_exp'], n_mid=n_md,
                     a_mid=md, a_mid_sd=md_s, n_hi=n_hi, a_hi=hi, a_hi_sd=hi_s,
                     da_hi=hi - fs['a_hi_exp'], p_hi=2.*(1. - hi), p_in=s['psyn']))
  df = pd.DataFrame(rows)
  if verbose and len(df):
    print(f"\n{'=== free slopes, all time bins ':=<96}")
    print(f"expected  a_lo = {4/3.:+.4f}   a_hi = {1-sides[0]['psyn']/2.:+.4f}  "
          f"(p = {sides[0]['psyn']})   windows at dfac = {sb.FREE_DFAC:g}")
    print("a_lo is averaged over CONVERGED bins only; 'unc' counts bins where the 4/3 "
          "asymptote\nis not reached inside the band (high-latitude tail) and only a lower "
          "bound exists.")
    print(f"{'z':>2} {'logr':>5} | {'N':>4} {'unc':>4} {'a_lo':>8} {'sd':>6} {'da_lo':>7} |"
          f" {'N':>4} {'a_mid':>8} {'sd':>6} | {'N':>4} {'a_hi':>8} {'sd':>6} {'da_hi':>7}"
          f" {'p_hi':>6}")
    print('-'*103)
    val = lambda v, n=8: f'{v:+{n}.4f}' if np.isfinite(v) else '--'.rjust(n)
    sd = lambda v: f'{v:6.4f}' if np.isfinite(v) else '--'.rjust(6)
    for _, w in df.iterrows():
      print(f"{w.z:>2.0f} {w.logr:+5.0f} | {w.n_lo:>4.0f} {w.n_lo_unconv:>4.0f} "
            f"{val(w.a_lo)} {sd(w.a_lo_sd)} {val(w.da_lo, 7)} | "
            f"{w.n_mid:>4.0f} {val(w.a_mid)} {sd(w.a_mid_sd)} | "
            f"{w.n_hi:>4.0f} {val(w.a_hi)} {sd(w.a_hi_sd)} {val(w.da_hi, 7)} {w.p_hi:6.3f}")
    for z in df.z.unique():
      d = df[df.z == z]
      def wmean(col, ncol):
        m = d[col].notna() & (d[ncol] > 0)
        return np.average(d[col][m], weights=d[ncol][m]) if m.any() else np.nan
      alo, ahi = wmean('a_lo', 'n_lo'), wmean('a_hi', 'n_hi')
      print(f"  z={z:.0f} bin-weighted:  a_lo = {alo:+.4f} (bias {alo-4/3.:+.4f}, "
            f"{d.n_lo.sum():.0f} converged / {d.n_lo.sum()+d.n_lo_unconv.sum():.0f} bins)   "
            f"a_hi = {ahi:+.4f} (bias {ahi-(1-d.p_in.iloc[0]/2.):+.4f})   "
            f"=> p = {2*(1-ahi):.4f} vs {d.p_in.iloc[0]}")
  return df


def epoch_table(sides, verbose=True):
  '''
  The three slopes split into rise / crossing / high-latitude tail. This is where the
  evolution lives: a_lo and a_hi are epoch-independent (they are properties of the electron
  distribution's ends), while a_mid softens away from its asymptote once the emission is
  dominated by high latitudes, exactly as granot_sari_syn's beta_mid note describes.
  '''
  rows = []
  for s in sides:
    fs = s['fs']
    for ep in EPOCHS:
      m = _epoch_mask(s, ep)
      lo, lo_s, n_lo = _summary(fs['a_lo'], m & fs['lo_converged'])
      md, md_s, n_md = _summary(fs['a_mid'], m)
      hi, hi_s, n_hi = _summary(fs['a_hi'], m)
      n_unc = int((m & np.isfinite(fs['a_lo']) & ~fs['lo_converged']).sum())
      # the free-window mid slope against the TEMPLATE's fitted beta_mid: two independent
      # measurements of the same quantity, one of which never used the GS02 shape
      dmf, _, n_dmf = _summary((fs['a_mid'] - 1.) - s['beta_mid'], m)
      rows.append(dict(z=s['z'], logr=s['logr'], epoch=ep, n_lo=n_lo, n_lo_unconv=n_unc,
                       a_lo=lo, n_mid=n_md, a_mid=md, a_mid_sd=md_s, n_hi=n_hi, a_hi=hi,
                       da_hi=hi - fs['a_hi_exp'], p_hi=2.*(1. - hi),
                       d_bmid=dmf, n_bmid=n_dmf))
  df = pd.DataFrame(rows)
  if verbose and len(df):
    print(f"\n{'=== slopes by epoch ':=<96}")
    print(f"{'z':>2} {'logr':>5} {'epoch':>9} | {'N':>4} {'unc':>4} {'a_lo':>8} |"
          f" {'N':>4} {'a_mid':>8} {'sd':>6} | {'N':>4} {'a_hi':>8} {'da_hi':>7}")
    print('-'*86)
    val = lambda v, n=8: f'{v:+{n}.4f}' if np.isfinite(v) else '--'.rjust(n)
    for _, w in df.iterrows():
      sd = f'{w.a_mid_sd:6.4f}' if np.isfinite(w.a_mid_sd) else '--'.rjust(6)
      print(f"{w.z:>2.0f} {w.logr:+5.0f} {w.epoch:>9} | {w.n_lo:>4.0f} {w.n_lo_unconv:>4.0f} "
            f"{val(w.a_lo)} | {w.n_mid:>4.0f} {val(w.a_mid)} {sd} | "
            f"{w.n_hi:>4.0f} {val(w.a_hi)} {val(w.da_hi, 7)}")
    # how far the mid slope moves between crossing and tail. NB this is NOT softening alone:
    # in the slow-cooling points nu_c falls through nu_m before the tail, so the expected
    # asymptote itself changes from (3-p)/2 to 1/2 and part of the move is that regime
    # change. The beta_mid residual below is the clean statement.
    print('\n  mid-slope move, crossing -> HLE (softening AND any FC/SC regime change):')
    for z in df.z.unique():
      for lr in sorted(df.logr.unique()):
        d = df[(df.z == z) & (df.logr == lr)].set_index('epoch')
        if not {'crossing', 'HLE'} <= set(d.index):
          continue
        c, h = d.loc['crossing'], d.loc['HLE']
        if not (np.isfinite(c.a_mid) and np.isfinite(h.a_mid)
                and c.n_mid >= MIN_BINS and h.n_mid >= MIN_BINS):
          continue
        print(f"    z={z:.0f} logr={lr:+.0f}: {c.a_mid:+.4f} -> {h.a_mid:+.4f} "
              f"({h.a_mid - c.a_mid:+.4f})")
    # independent check on the template: the free-window mid slope vs the GS02 fit's own
    # beta_mid, which is what track_breaks_gs02 uses to name the cooling regime
    g = df[df.n_bmid >= MIN_BINS]
    if len(g):
      print('\n  (a_mid - 1) - beta_mid fitted by the GS02 template, per epoch:')
      for ep in EPOCHS:
        d = g[g.epoch == ep]
        if len(d) and d.d_bmid.notna().any():
          print(f"    {ep:>9}: mean {d.d_bmid.mean():+.4f}  "
                f"max|.| {d.d_bmid.abs().max():.4f}  over {len(d.d_bmid.dropna())} points")
  return df


REGIMES = ('VFC', 'FC', 'MC', 'SC')


def regime_table(sides, verbose=True):
  '''
  Everything measured, pooled by the cooling regime DETECTED in each time bin rather than by
  the sweep parameter: the three free slopes and the two smoothing exponents, per regime,
  per shell. Bins are labelled by spectral_breaks.classify_regime (the fitted mid slope), so
  a single sweep point contributes to several regimes as nu_c crosses nu_m during its pulse
  -- which is the point: the regime is a property of the spectrum at that instant, not of
  the run.

  a_lo is restricted to bins where the nu^(4/3) asymptote is reached in band
  (lo_converged); s1, s2 to bins where the two breaks are far enough apart for the deficit
  identity to apply to each alone (s_clean, see spectral_breaks.SEP_CLEAN).
  '''
  rows = []
  for s in sides:
    fs = s['fs']
    reg = fs['regime']
    for rg in REGIMES:
      m = np.array([q == rg for q in reg])
      if not m.any():
        continue
      lo, lo_s, n_lo = _summary(fs['a_lo'], m & fs['lo_converged'])
      md, md_s, n_md = _summary(fs['a_mid'], m)
      hi, hi_s, n_hi = _summary(fs['a_hi'], m)
      ms = m & fs['s_clean']
      s1, s1_lo, s1_hi, n_s1 = _robust(fs['s1'], ms)
      s2, s2_lo, s2_hi, n_s2 = _robust(fs['s2'], ms)
      # the converged estimator: granot_sari_syn refitted with breaks and slopes held
      mf = m & fs['s_fit_ok']
      f1, f1_lo, f1_hi, n_f1 = _robust(fs['s1_fit'], mf)
      f2, f2_lo, f2_hi, n_f2 = _robust(fs['s2_fit'], mf)
      # how far s moves when the windows are pushed out: the number that says whether the
      # deficit estimate is a measurement at all (it mostly is not -- see _free_smoothing)
      dr1 = np.abs(fs['s1_far'] - fs['s1'])/np.abs(fs['s1'])
      dr2 = np.abs(fs['s2_far'] - fs['s2'])/np.abs(fs['s2'])
      drift = np.nanmedian(np.fmax(dr1[ms], dr2[ms])) if ms.any() else np.nan
      sep = np.nanmedian(fs['sep_free'][m]) if np.isfinite(fs['sep_free'][m]).any() else np.nan
      rows.append(dict(z=s['z'], logr=s['logr'], regime=rg, n_bins=int(m.sum()),
                       n_lo=n_lo, a_lo=lo, a_lo_sd=lo_s, n_mid=n_md, a_mid=md,
                       a_mid_sd=md_s, n_hi=n_hi, a_hi=hi, a_hi_sd=hi_s, p_hi=2.*(1. - hi),
                       n_s1=n_s1, s1=s1, s1_q16=s1_lo, s1_q84=s1_hi,
                       n_s2=n_s2, s2=s2, s2_q16=s2_lo, s2_q84=s2_hi,
                       s_drift=drift, n_s_stable=int((m & fs['s_stable']).sum()),
                       n_fit=max(n_f1, n_f2), n_fit_s1=n_f1, n_fit_s2=n_f2,
                       s1_fit=f1, s1_fit_q16=f1_lo, s1_fit_q84=f1_hi,
                       s2_fit=f2, s2_fit_q16=f2_lo, s2_fit_q84=f2_hi,
                       fit_rms=float(np.nanmedian(fs['s_fit_rms'][mf])) if mf.any() else np.nan,
                       sep_med=sep, psyn=s['psyn']))
  df = pd.DataFrame(rows)
  if verbose and len(df):
    p = sides[0]['psyn']
    print(f"\n{'=== pooled BY DETECTED REGIME ':=<104}")
    print(f"expected  a_lo = {4/3.:+.4f}   a_mid = {0.5:+.4f} (FC) / {(3.-p)/2.:+.4f} (SC)"
          f"   a_hi = {1.-p/2.:+.4f}    held smoothing: s1 = {swp.GS02_S1:g}, "
          f"s2 = {swp.GS02_S2:g}")
    print('slopes are mean +- sd; s1, s2 are median [q16-q84] -- see _robust.')
    print('s1, s2 columns are the HELD-BREAK FIT (fit_smoothing_held): granot_sari_syn '
          'refitted with\nthe crossings and all three slopes held, so the smoothing is the '
          'only freedom left.\nAccuracy through the full pipeline, from synthetics: s1 to '
          '<=3.5%, s2 to ~5-17%.\nThe last block is the MERGED-BREAK alternative (fit_single_break): '
          'one broad 4/3 -> 1-p/2\nbreak with NO mid slope. "gain" = rms(two-break)/'
          'rms(one-break), so >1 favours the single break.')
    hdr = (f"{'z':>2} {'regime':>6} {'bins':>5} | {'N':>4} {'a_lo':>8} {'sd':>6} | "
           f"{'N':>4} {'a_mid':>8} {'sd':>6} | {'N':>4} {'a_hi':>8} {'sd':>6} {'p_hi':>6} | "
           f"{'N':>4} {'s1':>18} | {'s2':>18} | {'rms':>6} | {'gain':>6} {'1brk wins':>10} "
           f"{'s1brk':>6} | {'sep':>8}")
    print(hdr); print('-'*len(hdr))
    val = lambda v, n=8, d=4: f'{v:+{n}.{d}f}' if np.isfinite(v) else '--'.rjust(n)
    pos = lambda v, n=6, d=3: f'{v:{n}.{d}f}' if np.isfinite(v) else '--'.rjust(n)
    def band(v):
      '''median [q16-q84] of a POOLED bin sample'''
      v = v[np.isfinite(v)]
      if not v.size:
        return '--'.rjust(18), 0
      q16, q84 = np.percentile(v, [16, 84])
      return f'{np.median(v):5.2f} [{q16:4.2f}-{q84:4.2f}]'.rjust(18), v.size
    # pool the raw BINS across sweep points: aggregating per-point summaries would mix a
    # weighted mean of medians with medians of percentiles and put the reported centre
    # outside its own quoted band
    for z in sorted(df.z.unique()):
      zs = [s for s in sides if s['z'] == z]
      for rg in REGIMES:
        cat = lambda key, gate=None: np.concatenate(
            [s['fs'][key][np.array([q == rg for q in s['fs']['regime']])
                          & (gate(s['fs']) if gate else True)] for s in zs])
        nb = sum(int(np.array([q == rg for q in s['fs']['regime']]).sum()) for s in zs)
        if not nb:
          continue
        alo = cat('a_lo', lambda f: f['lo_converged'])
        amd, ahi = cat('a_mid'), cat('a_hi')
        s1v = cat('s1_fit', lambda f: f['s_fit_ok'])
        s2v = cat('s2_fit', lambda f: f['s_fit_ok'])
        rms = cat('s_fit_rms', lambda f: f['s_fit_ok'])
        dr = cat('s_drift_max', lambda f: f['s_clean'])
        sep = cat('sep_free')
        # the merged-break alternative: one broad 4/3 -> 1-p/2 break, no mid slope
        g1 = cat('rms_gain_1brk', lambda f: f['s_1brk_ok'])
        sv1 = cat('s_1brk', lambda f: f['s_1brk_ok'])
        mn = lambda v: (np.nanmean(v), np.nanstd(v), int(np.isfinite(v).sum()))
        med = lambda v: np.nanmedian(v) if np.isfinite(v).any() else np.nan
        (ml, sl, nl), (mm, sm, nm), (mh, sh, nh) = mn(alo), mn(amd), mn(ahi)
        b1, n1 = band(s1v); b2, n2 = band(s2v)
        w1 = int(np.sum(g1[np.isfinite(g1)] > 1.))
        print(f"{z:>2.0f} {rg:>6} {nb:>5} | {nl:>4} {val(ml)} {pos(sl)} | "
              f"{nm:>4} {val(mm)} {pos(sm)} | {nh:>4} {val(mh)} {pos(sh)} "
              f"{pos(2*(1-mh))} | {max(n1, n2):>4} {b1} | {b2} | "
              f"{pos(med(rms), 6, 4)} | {pos(med(g1), 6, 2)} "
              f"{w1:>4}/{int(np.isfinite(g1).sum()):<5} {pos(med(sv1), 6, 2)} | "
              f"{med(sep):8.1e}")
    # The MC row above averages two populations that need different shapes: the genuinely
    # MERGED break while the shell emits on-axis, and the high-latitude smearing. Split them.
    print('\n  MC only, on-axis vs high-latitude (the merged-break case is the one the '
          'single\n  broad break is for; in the HLE neither shape describes the spectrum '
          'well):')
    print(f"   {'z':>2} {'MC bins':>8} {'rms 2-brk':>10} {'rms 1-brk':>10} {'gain':>6} "
          f"{'1brk wins':>11} {'s (1brk)':>18}")
    for z in sorted(df.z.unique()):
      zs = [s for s in sides if s['z'] == z]
      for lab, onax in (('on-axis', True), ('HLE', False)):
        r2, r1, sv = [], [], []
        for s in zs:
          fs = s['fs']
          m = (np.array([q == 'MC' for q in fs['regime']]) & fs['s_1brk_ok']
               & (fs['barT'] <= s['barT_f'] if onax else fs['barT'] > s['barT_f']))
          if not m.any():
            continue
          r1 += list(fs['rms_1brk'][m]); sv += list(fs['s_1brk'][m])
          r2 += list(fs['rms_1brk'][m]*fs['rms_gain_1brk'][m])
        if not r1:
          continue
        r1, r2, sv = np.array(r1), np.array(r2), np.array(sv)
        q16, q84 = np.percentile(sv[np.isfinite(sv)], [16, 84])
        print(f"   {z:>2.0f} {lab:>8} {len(r1):>4} {np.nanmedian(r2):10.4f} "
              f"{np.nanmedian(r1):10.4f} {np.nanmedian(r2)/np.nanmedian(r1):6.2f} "
              f"{int((r1 < r2).sum()):>5}/{len(r1):<5} "
              f"{f'{np.nanmedian(sv):.2f} [{q16:.2f}-{q84:.2f}]':>18}")
  return df


PRESC_BAD_MAX = 0.10   # a case whose frozen values fail MORE than this fraction of its bins is
                       # split by epoch even if its MEDIAN passes. Judging adequacy on the
                       # median alone is what let SC (15.1% of bins failing) and FC on the RS
                       # (16.5%) through unsplit: the symptom of a regime pooling two
                       # populations shows up in the tail long before it moves the median.
PRESC_RMS_MAX = 0.05   # a frozen-parameter fit worse than this (dex; ~12% in flux) is not an
                       # adequate description of that spectrum. Sits between the free-fit rms
                       # actually measured (0.011-0.051) and swp.GS02_TRACK_RMSMAX = 0.15,
                       # above which a bin is already not treated as a measurement at all.


def _bin_iter(sides, regime, onaxis=None):
  '''
  Yield (side, bin index) for every bin of one regime, optionally restricted to the on-axis
  pulse (bar{T} <= bar{T}_f) or the high-latitude tail. Bins are visited raw, never through a
  per-point summary -- the same pooling regime_table uses.
  '''
  for s in sides:
    fs = s['fs']
    m = np.array([q == regime for q in fs['regime']])
    if onaxis is not None:
      m &= (fs['barT'] <= s['barT_f']) if onaxis else (fs['barT'] > s['barT_f'])
    for i in np.flatnonzero(m):
      yield s, int(i)


def _refit_bin(s, i, s_hold=None, s1brk_hold=None):
  '''
  Refit one bin's spectrum with the SAME breaks, beta_mid and nu_M the free fit used, changing
  only whether the smoothing is frozen. Returns the rms, or NaN if that bin is not fittable.
  Passing neither hold reproduces the free fit. s_hold is the two-break pair (s1, s2);
  s1brk_hold the single scalar of the merged shape -- both now in the GS02 convention.
  '''
  fs, tr, r = s['fs'], s['tr'], s['r']
  x = swp.nu_over_num(r)
  vfc = bool(tr['is_vfc'][i])
  b_hi = fs['b_hi_held'][i] if np.isfinite(fs['b_hi_held'][i]) else fs['b_hi_free'][i]
  b_lo = b_hi if vfc else fs['b_lo_free'][i]
  if s1brk_hold is not None:
    return sb.fit_single_break(x, r['nuFnu'][i, :], s['psyn'], tr['nu_Mt'][i],
                               s_hold=s1brk_hold)['rms']
  f = sb.fit_smoothing_held(x, r['nuFnu'][i, :], s['psyn'], b_lo, b_hi, tr['nu_Mt'][i],
                            fs['a_mid'][i] - 1., vfc=vfc, s_hold=s_hold)
  return f['rms']


def _medians(sides, regime, onaxis=None):
  '''median (s1, s2) of the two-break fits, and median s of the merged fit, over one
  regime -- pooled over whichever sides are passed in, so the caller controls per-shell vs
  across-shell by what it hands over.'''
  s1, s2, sg = [], [], []
  for s, i in _bin_iter(sides, regime, onaxis):
    fs = s['fs']
    if fs['s_fit_ok'][i]:
      s1.append(fs['s1_fit'][i]); s2.append(fs['s2_fit'][i])
    if fs['s_1brk_ok'][i]:
      sg.append(fs['s_1brk'][i])
  nan = lambda v: np.nanmedian(v) if len(v) and np.isfinite(v).any() else np.nan
  return nan(np.array(s1, float)), nan(np.array(s2, float)), nan(np.array(sg, float))


def prescription_check(sides_by_z, outdir=OUTDIR, verbose=True):
  '''
  Does the TABULATED median smoothing actually describe the spectra of its regime?

  Every number in regime_table came from a fit with s free per spectrum. This freezes s at the
  regime median and refits each bin with everything else identical -- same breaks, same
  beta_mid, same nu_M -- so the difference in residual is exactly the cost of tabulating a
  fixed pair rather than fitting one. Four models are compared per bin:

    free    s fitted per spectrum (the floor; already in fs['s_fit_rms'])
    shell   s frozen at that shell's regime median
    pool    s frozen at the regime median pooled over BOTH shells
    global  s frozen at (GS02_S1, GS02_S2) = (1.3, 2.0), the status quo the table replaces

  MC uses the merged single-break shape and its s. A regime is adequate if the pooled
  frozen fit stays under PRESC_RMS_MAX; where it does not, the caller should split it by epoch
  (see prescription_refine).

  Returns a DataFrame, one row per (shell, regime).
  '''
  allsides = [s for sides in sides_by_z for s in sides]
  rows = []
  for rg in REGIMES:
    p1, p2, pg = _medians(allsides, rg)                      # pooled across shells
    for sides in sides_by_z:
      z = sides[0]['z']
      h1, h2, hg = _medians(sides, rg)                       # this shell alone
      rec = {k: [] for k in ('free', 'shell', 'pool', 'glob')}
      for s, i in _bin_iter(sides, rg):
        if rg == 'MC':
          if not s['fs']['s_1brk_ok'][i]:
            continue
          rec['free'].append(s['fs']['rms_1brk'][i])
          rec['shell'].append(_refit_bin(s, i, s1brk_hold=hg))
          rec['pool'].append(_refit_bin(s, i, s1brk_hold=pg))
          rec['glob'].append(np.nan)      # no global default exists for the merged shape
        else:
          if not s['fs']['s_fit_ok'][i]:
            continue
          rec['free'].append(s['fs']['s_fit_rms'][i])
          rec['shell'].append(_refit_bin(s, i, s_hold=(h1, h2)))
          rec['pool'].append(_refit_bin(s, i, s_hold=(p1, p2)))
          rec['glob'].append(_refit_bin(s, i, s_hold=(swp.GS02_S1, swp.GS02_S2)))
      a = {k: np.array(v, float) for k, v in rec.items()}
      if not a['free'].size:
        continue
      good = np.isfinite(a['pool'])
      rows.append(dict(
          z=z, regime=rg, n=int(a['free'].size),
          s1_shell=h1, s2_shell=h2, sig_shell=hg, s1_pool=p1, s2_pool=p2, sig_pool=pg,
          rms_free=np.nanmedian(a['free']), rms_shell=np.nanmedian(a['shell']),
          rms_pool=np.nanmedian(a['pool']),
          # MC has no global default to compare against, so a['glob'] is all-NaN there
          rms_glob=(np.nanmedian(a['glob']) if np.isfinite(a['glob']).any() else np.nan),
          cost=np.nanmedian(a['pool']) - np.nanmedian(a['free']),
          rms_pool_q84=(np.nanpercentile(a['pool'][good], 84) if good.any() else np.nan),
          frac_bad=(float(np.mean(a['pool'][good] > PRESC_RMS_MAX)) if good.any() else np.nan)))
  df = pd.DataFrame(rows)
  if verbose and len(df):
    print(f"\n{'=== DO THE TABULATED PARAMETERS FIT? ':=<100}")
    print(f'Every bin refit with s FROZEN at the regime median, breaks and beta_mid unchanged.'
          f'\n"free" is the same fit with s free per spectrum -- the floor. "glob" freezes at '
          f'the\ncurrent global ({swp.GS02_S1:g}, {swp.GS02_S2:g}). Adequate means median rms '
          f'< {PRESC_RMS_MAX:g} dex.')
    hdr = (f"{'z':>2} {'regime':>6} {'N':>5} | {'s1/sig':>7} {'s2':>6} (pooled) | "
           f"{'free':>7} {'shell':>7} {'pool':>7} {'glob':>7} | {'cost':>7} {'q84':>7} "
           f"{'%>thr':>6} | verdict")
    print(hdr); print('-'*len(hdr))
    f4 = lambda v: f'{v:7.4f}' if np.isfinite(v) else '     --'
    for _, w in df.iterrows():
      ok = np.isfinite(w.rms_pool) and w.rms_pool < PRESC_RMS_MAX
      v = 'ok' if ok else 'INADEQUATE'
      sa = w.sig_pool if w.regime == 'MC' else w.s1_pool
      sb_ = np.nan if w.regime == 'MC' else w.s2_pool
      print(f"{w.z:>2.0f} {w.regime:>6} {w.n:>5} | "
            f"{sa:7.3f} {(f'{sb_:6.3f}' if np.isfinite(sb_) else '    --')} "
            f"{'':9}| {f4(w.rms_free)} {f4(w.rms_shell)} {f4(w.rms_pool)} {f4(w.rms_glob)} | "
            f"{f4(w.cost)} {f4(w.rms_pool_q84)} "
            f"{(f'{100*w.frac_bad:5.1f}%' if np.isfinite(w.frac_bad) else '    --')} | {v}")
  return df


def _presc_row(allsides, regime, onaxis=None):
  '''
  Derive the pooled median for one (regime, epoch) case and measure what freezing at it costs,
  over every bin of that case on both shells. The single building block behind both the
  refinement and the recommended table.
  '''
  p1, p2, pg = _medians(allsides, regime, onaxis=onaxis)
  free, pool = [], []
  for s, i in _bin_iter(allsides, regime, onaxis=onaxis):
    if regime == 'MC':
      if not s['fs']['s_1brk_ok'][i]:
        continue
      free.append(s['fs']['rms_1brk'][i]); pool.append(_refit_bin(s, i, s1brk_hold=pg))
    else:
      if not s['fs']['s_fit_ok'][i]:
        continue
      free.append(s['fs']['s_fit_rms'][i]); pool.append(_refit_bin(s, i, s_hold=(p1, p2)))
  free, pool = np.array(free, float), np.array(pool, float)
  if not free.size:
    return None
  g = np.isfinite(pool)
  # NB 'post-crossing' is bar{T} > bar{T}_f, i.e. everything after the shock finishes crossing.
  # That is EARLIER than the formal high-latitude window _epoch_mask uses; the two must not be
  # conflated when reading this against epoch_table.
  return dict(regime=regime, epoch=('all' if onaxis is None else
                                    ('on-axis' if onaxis else 'post-crossing')),
              n=int(free.size), s1=p1, s2=p2, sig=pg,
              rms_free=np.nanmedian(free), rms_pool=np.nanmedian(pool),
              cost=np.nanmedian(pool) - np.nanmedian(free),
              frac_bad=(float(np.mean(pool[g] > PRESC_RMS_MAX)) if g.any() else np.nan))


# The cases the recommended table is built from. FC, MC and SC are split at bar{T}_f because
# each pools two populations with genuinely different break widths; measured on its own subset,
# every one of their on-axis halves then fails NO bin at all. VFC is left whole: splitting it
# does not help, because its (small) tail of poorly fitted bins is the ON-AXIS subset, not the
# late one -- 6.8% of 235 on-axis bins against 0.0% of 1056 post-crossing ones, which is the
# 1.3% the unsplit case reports.
PRESC_CASES = (('VFC', None), ('FC', True), ('FC', False), ('MC', True), ('MC', False),
               ('SC', True), ('SC', False))


def prescription_summary(sides_by_z, cases=PRESC_CASES, verbose=True):
  '''
  The recommended prescription: one row per case actually worth tabulating, with the cost of
  using it. This is the table the appendix should carry -- prescription_check says which
  regimes need splitting, prescription_refine derives the split values, and this assembles
  the answer.
  '''
  allsides = [s for sides in sides_by_z for s in sides]
  rows = [r for r in (_presc_row(allsides, rg, ax) for rg, ax in cases) if r is not None]
  df = pd.DataFrame(rows)
  if verbose and len(df):
    print(f"\n{'=== RECOMMENDED PRESCRIPTION (both shells pooled) ':=<92}")
    print(f'"cost" is the median rms penalty against fitting s per spectrum; "%>thr" the '
          f'fraction of\nbins the frozen values fail to describe to {PRESC_RMS_MAX:g} dex.')
    hdr = (f"{'case':>17} {'N':>5} | {'s1':>6} {'s2':>6} {'s(1brk)':>8} | {'rms free':>8} "
           f"{'rms held':>8} {'cost':>8} {'%>thr':>6} | verdict")
    print(hdr); print('-'*len(hdr))
    f3 = lambda v: f'{v:6.3f}' if np.isfinite(v) else '    --'
    for _, w in df.iterrows():
      ok = np.isfinite(w.rms_pool) and w.rms_pool < PRESC_RMS_MAX
      tag = 'use' if ok and w.frac_bad < 0.1 else ('use with care' if ok else 'DO NOT USE')
      print(f"{w.regime + ' ' + w.epoch:>17} {w.n:>5} | {f3(w.s1)} {f3(w.s2)} {f3(w.sig)} | "
            f"{w.rms_free:8.4f} {w.rms_pool:8.4f} {w.cost:+8.4f} {100*w.frac_bad:5.1f}% | {tag}")
  return df


def prescription_refine(sides_by_z, regimes, outdir=OUTDIR, verbose=True):
  '''
  Re-derive and re-test the medians of the given regimes SPLIT by epoch (on-axis pulse vs
  high-latitude tail), the same split regime_table already applies to MC. Used where
  prescription_check finds a regime inadequate -- typically because its bins pool two
  populations with genuinely different break widths, which a single median cannot represent.
  '''
  allsides = [s for sides in sides_by_z for s in sides]
  rows = [r for r in (_presc_row(allsides, rg, ax)
                      for rg in regimes for ax in (True, False)) if r is not None]
  df = pd.DataFrame(rows)
  if verbose and len(df):
    print(f"\n  REFINED: medians re-derived per epoch, both shells pooled")
    print(f"  {'regime':>6} {'epoch':>8} {'N':>5} | {'s1':>6} {'s2':>6} {'s(1brk)':>8} | "
          f"{'free':>7} {'pool':>7} {'cost':>7} {'%>thr':>6} | verdict")
    for _, w in df.iterrows():
      f3 = lambda v: f'{v:6.3f}' if np.isfinite(v) else '    --'
      ok = np.isfinite(w.rms_pool) and w.rms_pool < PRESC_RMS_MAX
      print(f"  {w.regime:>6} {w.epoch:>8} {w.n:>5} | {f3(w.s1)} {f3(w.s2)} {f3(w.sig)} | "
            f"{w.rms_free:7.4f} {w.rms_pool:7.4f} {w.cost:+7.4f} "
            f"{100*w.frac_bad:5.1f}% | {'ok' if ok else 'INADEQUATE'}")
  return df


def _model_curve(s, i, s_hold=None, s1brk_hold=None):
  '''
  The fitted model of one bin, rebuilt on the observed frequency grid in nuFnu units, so it
  can be drawn over the data. Returns (x, model) or (None, None).
  '''
  fs, tr, r = s['fs'], s['tr'], s['r']
  x = swp.nu_over_num(r); p = s['psyn']; nuM = tr['nu_Mt'][i]
  with np.errstate(divide='ignore', invalid='ignore'):
    R = syn_cutoff_R(x/nuM)
  if s1brk_hold is not None or (s_hold is None and np.array([q == 'MC'
                                for q in fs['regime']])[i]):
    f = sb.fit_single_break(x, r['nuFnu'][i, :], p, nuM, s_hold=s1brk_hold)
    if not np.isfinite(f['rms']):
      return None, None
    m = granot_sari_syn(x, f['nu_b'], None, p, s2=f['s'], nuM=None, F_ext=f['A'],
                        nuFnu=True, beta_lo_single=1./3.)
  else:
    vfc = bool(tr['is_vfc'][i])
    b_hi = fs['b_hi_held'][i] if np.isfinite(fs['b_hi_held'][i]) else fs['b_hi_free'][i]
    b_lo = b_hi if vfc else fs['b_lo_free'][i]
    f = sb.fit_smoothing_held(x, r['nuFnu'][i, :], p, b_lo, b_hi, nuM,
                              fs['a_mid'][i] - 1., vfc=vfc, s_hold=s_hold)
    if not np.isfinite(f['rms']):
      return None, None
    # beta_mid MUST be passed: without it granot_sari_syn infers the middle slope from the
    # ordering of the breaks, which is not the measured value the fit actually used
    m = granot_sari_syn(x, b_lo, (None if vfc else f['b_hi_fit']), p, s1=f['s1'], s2=f['s2'],
                        nuM=None, F_ext=f['F_ext'], nuFnu=True,
                        beta_mid=fs['a_mid'][i] - 1.)
  return x, m*10**f['y0']*np.where(np.isfinite(R) & (R > 0.), R, np.nan)


def plot_prescription(sides_by_z, summary, outdir=OUTDIR, cases=PRESC_CASES):
  '''
  The WORST bin of each RECOMMENDED case under its frozen parameters, with the frozen model
  and the free-s model drawn over the data. A median residual can hide a shape that is simply
  wrong somewhere; this is the panel that shows whether "adequate" means adequate everywhere.
  The cases are the split ones actually being recommended, not the unsplit regimes -- including
  the one that fails, since a reader needs to see what "DO NOT USE" looks like.
  '''
  _ensure_outdir()
  allsides = [s for sides in sides_by_z for s in sides]
  panels = []
  for rg, onax in cases:
    lab = 'all' if onax is None else ('on-axis' if onax else 'HLE')
    row = summary[(summary.regime == rg) & (summary.epoch == lab)]
    if not len(row):
      continue
    p1, p2, pg = row.s1.iloc[0], row.s2.iloc[0], row.sig.iloc[0]
    worst, wr = None, -np.inf
    for s, i in _bin_iter(allsides, rg, onaxis=onax):
      ok = s['fs']['s_1brk_ok'][i] if rg == 'MC' else s['fs']['s_fit_ok'][i]
      if not ok:
        continue
      rms = (_refit_bin(s, i, s1brk_hold=pg) if rg == 'MC'
             else _refit_bin(s, i, s_hold=(p1, p2)))
      if np.isfinite(rms) and rms > wr:
        worst, wr = (s, i), rms
    if worst is not None:
      panels.append((f'{rg} {lab}', worst, wr, (p1, p2, pg), rg))
  if not panels:
    return None
  ncol = 3
  nrow = int(np.ceil(len(panels)/ncol))
  fig, axs0 = plt.subplots(nrow, ncol, figsize=(4.3*ncol, 3.8*nrow), squeeze=False)
  axs = [(a, pn) for a, pn in zip(axs0.ravel(), panels)]
  for a in axs0.ravel()[len(panels):]:
    a.axis('off')
  for ax, (rg, (s, i), wr, (p1, p2, pg), rg0) in axs:
    x = swp.nu_over_num(s['r']); data = s['r']['nuFnu'][i, :]
    norm = np.nanmax(data)
    ax.loglog(x, data/norm, color='k', lw=1.6, label='computed')
    for lab, kw, col, ls in (('s free', {}, '#1f77b4', '--'),
                             ('s at tabulated value',
                              (dict(s1brk_hold=pg) if rg0 == 'MC' else dict(s_hold=(p1, p2))),
                              '#d62728', '-')):
      xm, m = _model_curve(s, i, **kw)
      if xm is not None:
        ax.loglog(xm, m/norm, color=col, ls=ls, lw=1.2, label=lab)
    ax.set_ylim(10**-3.6, 3.)
    ax.set_xlim(max(x.min(), 1e-7), x.max())
    ax.grid(alpha=.2, lw=.5)
    ax.set_title(f"{rg}: worst bin, rms = {wr:.3f}\n"
                 f"z={s['z']}, log$_{{10}}(\\gamma_c/\\gamma_m)$={s['logr']:+.0f}, "
                 f"$\\bar{{T}}$={s['fs']['barT'][i]:.2f}",
                 fontsize=8, color=('#b00' if wr > PRESC_RMS_MAX else 'k'))
    ax.set_xlabel('$\\nu/\\nu_{m,0}$')
  for a in axs0[:, 0]:
    a.set_ylabel('$\\nu F_\\nu$ (peak-normalised)')
  axs0.ravel()[0].legend(fontsize=7, framealpha=.9)
  fig.suptitle('Tabulated per-case parameters vs the free fit, at each case\'s WORST bin '
               f'(red title: worse than {PRESC_RMS_MAX:g} dex)', fontsize=10)
  fig.tight_layout()
  path = os.path.join(outdir, 'prescription_worst.png')
  fig.savefig(path, dpi=200, bbox_inches='tight')
  plt.close(fig)
  print(f'-> {path}')
  return path


def convergence_check(sides, dscan=sb.FREE_DSCAN, verbose=True):
  '''
  a_lo against the window standoff dfac, at each point's lightcurve peak, beside a
  synthetic granot_sari_syn control whose a_lo IS 4/3 by construction.

  The point of the control: if the computed spectra converge to 4/3 at the same rate the
  synthetic does, the deficit at small dfac is the template's own break curvature and says
  nothing about the physics. If they converged more slowly -- or to something else -- the
  spectrum would genuinely not carry the asymptote.
  '''
  rows = []
  for s in sides:
    r, tr, fs = s['r'], s['tr'], s['fs']
    x = swp.nu_over_num(r)
    i = int(np.nanargmax(fs['Fpk']))
    if not (np.isfinite(tr['nu_lo'][i]) and np.isfinite(tr['nu_Mt'][i])):
      continue
    conv = sb.free_slope_convergence(x, r['nuFnu'][i, :], s['psyn'], tr['nu_lo'][i],
                                     tr['nu_hi'][i], tr['nu_Mt'][i], dscan=dscan,
                                     vfc=bool(tr['is_vfc'][i]))
    row = dict(z=s['z'], logr=s['logr'], kind='computed')
    row.update({f'd{d:g}': conv[d]['a_lo'] for d in dscan})
    rows.append(row)
  # synthetic controls at three smoothings, same estimator
  xs = np.logspace(-6., 6., 600)
  p = sides[0]['psyn']
  for s1 in (1.0, 1.3, 2.0):
    syn = granot_sari_syn(xs, 1., 1e3, p, s1=s1, s2=2.0, nuM=1e5, nuFnu=True)
    conv = sb.free_slope_convergence(xs, syn, p, 1., 1e3, 1e5, dscan=dscan)
    row = dict(z=np.nan, logr=np.nan, kind=f'synthetic s1={s1:g}')
    row.update({f'd{d:g}': conv[d]['a_lo'] for d in dscan})
    rows.append(row)
  df = pd.DataFrame(rows)
  if verbose and len(df):
    cols = [f'd{d:g}' for d in dscan]
    print(f"\n{'=== a_lo vs window standoff (peak spectra) ':=<96}")
    print(f"{'kind':>16} {'z':>3} {'logr':>5} | " + ' '.join(f'{c:>8}' for c in cols)
          + f"   (4/3 = {4/3.:+.4f})")
    print('-'*80)
    for _, w in df.iterrows():
      zt = f'{w.z:>3.0f}' if np.isfinite(w.z) else '  -'
      lt = f'{w.logr:+5.0f}' if np.isfinite(w.logr) else '    -'
      print(f"{w.kind:>16} {zt} {lt} | "
            + ' '.join(f'{w[c]:+8.4f}' if np.isfinite(w[c]) else '      --' for c in cols))
    comp = df[df.kind == 'computed']
    print('\n  bias vs 4/3, computed spectra:  '
          + '  '.join(f'{c}: {comp[c].mean()-4/3.:+.4f}' for c in cols
                      if comp[c].notna().any()))
  return df


def plot_slope_evolution(sides_by_z, outdir=OUTDIR):
  '''
  The three free slopes vs bar{T}/bar{T}_f, one row per segment and one column per shell,
  coloured by cooling regime. Dashed lines mark the values the rest of the code HOLDS;
  for the mid segment both asymptotes are drawn, since which one applies is the regime.
  Gaps are bins where the segment is not resolved -- not interpolated.

  In the a_lo panel the faint continuations are the bins where the nu^(4/3) asymptote is
  NOT reached inside the frequency band (lo_converged False): they are lower bounds on the
  slope, not measurements of it, and the apparent late-time decline is the band running out
  under a break the high-latitude superposition has smeared -- see FREE_CONV_TOL.
  '''
  _ensure_outdir()
  nz = len(sides_by_z)
  fig, axs = plt.subplots(3, nz, figsize=(6.2*nz, 9.0), sharex=True, squeeze=False)
  p = sides_by_z[0][0]['psyn']
  segs = (('a_lo', '$a_{\\rm low}$  ($\\nu F_\\nu \\propto \\nu^{4/3}$)', (4/3.,),
           (0.65, 1.45)),
          ('a_mid', '$a_{\\rm mid}$', (0.5, (3.-p)/2.), (-0.05, 0.95)),
          ('a_hi', f'$a_{{\\rm high}}$  ($\\nu^{{1-p/2}}$, p={p:g})', (1.-p/2.,),
           (-0.40, -0.10)))
  for jz, sides in enumerate(sides_by_z):
    logrs = np.array([s['logr'] for s in sides], float)
    norm = plt.Normalize(logrs.min(), logrs.max())
    for i, (key, lab, refs, ylim) in enumerate(segs):
      ax = axs[i][jz]
      for s in sides:
        b = s['fs']['barT']/s['barT_f']
        y = s['fs'][key]
        col = CMAP(norm(s['logr']))
        if key == 'a_lo':
          cv = s['fs']['lo_converged']
          ax.plot(b, np.where(cv, y, np.nan), lw=1.4, color=col,
                  label=f"{s['logr']:+.0f}")
          ax.plot(b, np.where(cv, np.nan, y), lw=0.8, color=col, alpha=.35)
          continue
        ax.plot(b, y, lw=1.1, color=col)
      for v in refs:
        ax.axhline(v, color='k', ls='--', lw=.9, alpha=.7)
      ax.axvline(1., color='0.4', ls=':', lw=.9)          # shell crossing
      off = sides[0]['barT_off']
      if off is not None:
        ax.axvspan(off[0]/sides[0]['barT_f'], off[1]/sides[0]['barT_f'],
                   color='0.85', alpha=.5, lw=0)          # rarefaction cut-off band
      ax.set_xscale('log'); ax.set_xlim(1e-3, None); ax.set_ylim(*ylim)
      ax.grid(alpha=.2, lw=.5)
      if jz == 0:
        ax.set_ylabel(lab, fontsize=10)
      if i == 0:
        ax.set_title(f"z = {sides[0]['z']}"
                     f"  ({'reverse' if sides[0]['z'] == 4 else 'forward'} shock)", fontsize=10)
    axs[-1][jz].set_xlabel('$\\bar{T}/\\bar{T}_f$')
  axs[0][0].legend(fontsize=7, ncol=2, framealpha=.9,
                   title='log$_{10}(\\gamma_c/\\gamma_m)$', title_fontsize=7)
  fig.suptitle('Segment slopes measured with NO value imposed, vs the values held '
               f'everywhere else (dashed)   [windows at dfac = {sb.FREE_DFAC:g}]', fontsize=11)
  fig.tight_layout()
  path = os.path.join(outdir, 'slope_evolution.png')
  fig.savefig(path, dpi=200, bbox_inches='tight')
  plt.close(fig)
  print(f'-> {path}')
  return path


def write_tables(tab, ep, conv, outdir=OUTDIR, reg=None, presc=None, refined=None,
    summary=None):
  '''the tables to csv'''
  _ensure_outdir()
  out = [(tab, 'free_slopes.csv'), (ep, 'free_slopes_by_epoch.csv'),
         (conv, 'a_lo_convergence.csv')]
  if reg is not None:
    out.append((reg, 'free_slopes_by_regime.csv'))
  if presc is not None:
    out.append((presc, 'prescription_check.csv'))
  if refined is not None and len(refined):
    out.append((refined, 'prescription_refined.csv'))
  if summary is not None and len(summary):
    out.append((summary, 'prescription_recommended.csv'))
  for df, fn in out:
    df.to_csv(os.path.join(outdir, fn), index=False)
  print('-> ' + outdir + '/' + ', '.join(fn for _, fn in out))


def main(key=KEY, method=METHOD, zlist=(Z_RS, Z_FS), outdir=OUTDIR, dfac=sb.FREE_DFAC):
  '''
  Full free-slope diagnostic on the cached sweeps of both shells, every time bin. No shell
  spectrum is recomputed -- the breaks positioning the windows come from track_breaks_gs02
  on the cache.
  '''
  _ensure_outdir()
  sides_by_z, tabs, eps, regs, convs = [], [], [], [], []
  for z in zlist:
    print(f'\n--- shell z={z} ---', flush=True)
    sides = load_side(z=z, key=key, method=method, dfac=dfac)
    sides_by_z.append(sides)
    tabs.append(slope_table(sides))
    eps.append(epoch_table(sides))
    regs.append(regime_table(sides))
    convs.append(convergence_check(sides))
  tab = pd.concat(tabs, ignore_index=True)
  ep = pd.concat(eps, ignore_index=True)
  reg = pd.concat(regs, ignore_index=True)
  conv = pd.concat(convs, ignore_index=True)
  # does the tabulated median actually describe its regime's spectra?
  presc = prescription_check(sides_by_z, outdir=outdir)
  bad = sorted({w.regime for _, w in presc.iterrows()
                if not (np.isfinite(w.rms_pool) and w.rms_pool < PRESC_RMS_MAX
                        and np.isfinite(w.frac_bad) and w.frac_bad <= PRESC_BAD_MAX)})
  refined = prescription_refine(sides_by_z, bad, outdir=outdir) if bad else None
  summary = prescription_summary(sides_by_z)
  plot_prescription(sides_by_z, summary, outdir=outdir)
  plot_slope_evolution(sides_by_z, outdir=outdir)
  write_tables(tab, ep, conv, outdir=outdir, reg=reg, presc=presc,
               refined=refined, summary=summary)
  swp.trim_pngs(outdir)
  print(f'\nfree-slope diagnostic saved to {outdir}')
  return sides_by_z, tab, ep, reg, conv, presc, summary


if __name__ == '__main__':
  main()


# ---------------------------------------------------------------------------
# High-latitude degradation: how the on-axis prescription fails in time, and what
# smoothing would fix it
# ---------------------------------------------------------------------------
HLE_TBINS = np.array([0., 0.3, 1., 2., 3., 5., 10., 30., 100., np.inf])  # barT/barT_f edges


def hle_bins(sides_by_z, method, refs, verbose=True):
  '''
  One row per time bin of every sweep point: the residual left by FREEZING the smoothing at
  its ON-AXIS value for that bin's detected regime, against the residual of the same bin fitted
  freely, plus the freely-fitted smoothing itself and the diagnostics that say whether it can
  be trusted there.

  `refs` maps regime -> (s1, s2, s_1brk) measured on the on-axis bins, so the frozen values are
  exactly the ones prescription_summary validated. The free-fit residual (fs['s_fit_rms'],
  fs['rms_1brk']) is carried alongside so the prescription's failure can be separated from the
  spectrum simply being harder to fit at late times -- those are different statements and the
  excess is the one that belongs to the prescription.
  '''
  rows = []
  for sides in sides_by_z:
    for s in sides:
      fs, tr = s['fs'], s['tr']
      for i, rg in enumerate(fs['regime']):
        if rg is None or rg not in refs:
          continue
        merged = (rg == 'MC')
        ok = fs['s_1brk_ok'][i] if merged else fs['s_fit_ok'][i]
        if not ok:
          continue
        r1, r2, rg1 = refs[rg]
        held = (_refit_bin(s, i, s1brk_hold=rg1) if merged
                else _refit_bin(s, i, s_hold=(r1, r2)))
        free = fs['rms_1brk'][i] if merged else fs['s_fit_rms'][i]
        rows.append(dict(
            method=method, z=s['z'], logr=s['logr'], regime=rg,
            barT_n=fs['barT'][i]/s['barT_f'], rms_held=held, rms_free=free,
            excess=held - free,
            s1=fs['s1_fit'][i], s2=fs['s2_fit'][i], s1brk=fs['s_1brk'][i],
            lo_conv=bool(fs['lo_converged'][i]),
            dex_lo=fs['dex_lo'][i], dex_mid=fs['dex_mid'][i],
            a_edge=fs['a_edge'][i], a_lo=fs['a_lo'][i]))
  df = pd.DataFrame(rows)
  if verbose and len(df):
    print(f"  {method}: {len(df)} bins "
          f"({int((df.barT_n <= 1).sum())} on-axis, {int((df.barT_n > 1).sum())} post-crossing)")
  return df


def hle_report(df, tbins=HLE_TBINS, verbose=True, by_regime=False):
  '''
  The degradation curve: median held / free residual and the failure fraction, in bins of
  barT/barT_f, per method. Answers "how badly does each shape stop fitting, and when".

  by_regime=True splits it by the detected regime as well, which is the ONLY form in which the
  two methods may be compared: pooled over regime their columns contain different shape-class
  populations (see spectral_breaks.edge_slope), so a difference between them can be a
  difference in what was fitted rather than in the spectra.
  '''
  keys = ['method', 'regime'] if by_regime else ['method']
  rows = []
  for gk, dm in df.groupby(keys):
    method, regime = (gk if by_regime else (gk, 'all'))
    for lo, hi in zip(tbins[:-1], tbins[1:]):
      d = dm[(dm.barT_n > lo) & (dm.barT_n <= hi)]
      if not len(d):
        continue
      rows.append(dict(method=method, regime=regime, t_lo=lo, t_hi=hi, n=len(d),
                       rms_held=d.rms_held.median(), rms_free=d.rms_free.median(),
                       excess=d.excess.median(),
                       frac_bad=float((d.rms_held > PRESC_RMS_MAX).mean())))
  out = pd.DataFrame(rows)
  if verbose and len(out) and by_regime:
    print(f"\n{'=== DEGRADATION, MATCHED BY REGIME (the only valid method comparison) ':=<86}")
    for rg, d in out.groupby('regime'):
      piv = d.pivot_table(index=['t_lo'], columns='method', values=['rms_held', 'n'])
      if len(piv) < 2:
        continue
      print(f'\n  regime {rg}')
      print(f"    {'barT/barT_f':>12} {'N data':>7} {'N rcut':>7} {'rms data':>9} "
            f"{'rms rcut':>9} {'ratio':>7}")
      for tl in piv.index:
        nd = piv[('n', 'data')].get(tl, np.nan); nr = piv[('n', 'data_rarcut')].get(tl, np.nan)
        rd = piv[('rms_held', 'data')].get(tl, np.nan)
        rr = piv[('rms_held', 'data_rarcut')].get(tl, np.nan)
        if not (np.isfinite(nd) and np.isfinite(nr)):
          continue
        print(f"    {tl:>12g} {nd:>7.0f} {nr:>7.0f} {rd:>9.4f} {rr:>9.4f} "
              f"{(rd/rr if np.isfinite(rd/rr) else np.nan):>7.2f}")
    return out
  if verbose and len(out):
    print(f"\n{'=== DEGRADATION OF THE ON-AXIS PRESCRIPTION IN TIME ':=<86}")
    print(f'smoothing frozen at each regime\'s ON-AXIS value; "free" is the same bin with s '
          f'fitted.\n"bad" = fraction of bins the frozen values miss by more than '
          f'{PRESC_RMS_MAX:g} dex.')
    for method, d in out.groupby('method'):
      print(f'\n  {method}')
      print(f"    {'barT/barT_f':>14} {'N':>5} {'rms held':>9} {'rms free':>9} "
            f"{'excess':>8} {'bad':>7}")
      for _, w in d.iterrows():
        lab = f'{w.t_lo:g}-{w.t_hi:g}' if np.isfinite(w.t_hi) else f'>{w.t_lo:g}'
        print(f"    {lab:>14} {w.n:>5.0f} {w.rms_held:9.4f} {w.rms_free:9.4f} "
              f"{w.excess:+8.4f} {100*w.frac_bad:6.1f}%")
  return out


def hle_smoothing_evolution(df, floors=(1., 2., 5., 10.), verbose=True):
  '''
  Does the freely-fitted smoothing keep falling after crossing, or does it saturate?

  A thin shell that has switched off should show a FROZEN spectral shape at high latitude --
  the spectrum slides in frequency and flux but keeps its form, so s should go to a constant.
  We therefore fit log10 s = a + k log10(barT/barT_f) past a series of floors: if the shape
  freezes, k -> 0 as the floor moves late. `k_const_gain` is how much better the sloped fit is
  than a constant (ratio of residual rms); a value near 1 means a constant describes it just as
  well and no trend should be claimed.

  s1 and s2 are reported separately ON PURPOSE. s2 does not depend on the low-frequency band,
  whereas s1 comes through b_lo, which is exactly what the band limit destroys in the tail
  (see lo_converged). A decline in s1 alone is a measurement artefact; a decline in both is a
  real broadening.
  '''
  rows = []
  for (method, rg), d in df.groupby(['method', 'regime']):
    for col in ('s1', 's2', 's1brk'):
      for fl in floors:
        m = (d.barT_n > fl) & np.isfinite(d[col]) & (d[col] > 0)
        if m.sum() < 12:
          continue
        x = np.log10(d.barT_n[m].to_numpy()); y = np.log10(d[col][m].to_numpy())
        k, a = np.polyfit(x, y, 1)
        res_slope = float(np.std(y - (a + k*x)))
        res_const = float(np.std(y - y.mean()))
        rows.append(dict(method=method, regime=rg, quantity=col, floor=fl, n=int(m.sum()),
                         k=float(k), s_at_floor=float(10**(a + k*np.log10(fl))),
                         res_slope=res_slope, res_const=res_const,
                         gain=(res_const/res_slope if res_slope > 0 else np.nan)))
  out = pd.DataFrame(rows)
  if verbose and len(out):
    print(f"\n{'=== DOES s SATURATE AFTER CROSSING? ':=<86}")
    print('fit log10 s = a + k log10(barT/barT_f) past each floor. k -> 0 with the floor means'
          '\nthe shape freezes, as a switched-off thin shell should. "gain" is how much better'
          '\nthe sloped fit is than a constant: ~1 means no trend worth claiming.')
    for (method, rg), d in out.groupby(['method', 'regime']):
      if rg == 'MC':
        cols = ('s1brk',)
      elif rg == 'VFC':
        cols = ('s2',)
      else:
        cols = ('s1', 's2')
      shown = d[d.quantity.isin(cols)]
      if not len(shown):
        continue
      print(f'\n  {method}  {rg}')
      for q, dq in shown.groupby('quantity'):
        cells = '  '.join(f'>{w.floor:g}: k={w.k:+.2f} (g={w.gain:.2f}, n={w.n:.0f})'
                          for _, w in dq.iterrows())
        print(f'    {q:>6}  {cells}')
  return out


def low_energy_softening(df, tbins=HLE_TBINS, verbose=True):
  '''
  Quantify how far the low-energy end of the INSTANTANEOUS spectra softens when the cells are
  allowed to keep radiating, against the run in which they are switched off at the rarefaction
  wave.

  The measure is `a_edge` (spectral_breaks.edge_slope): the nuFnu log-log slope over the lowest
  decade of usable band. It is used rather than the fitted low-segment slope because in the
  tail NEITHER run reaches the nu^(4/3) asymptote inside the band, so a quantity that requires
  that asymptote is undefined exactly where the comparison is wanted. a_edge is defined
  wherever there is band, and is measured identically in both runs.

  This is the instantaneous counterpart of the time-integrated result already in
  figures/rarcut_compare/fluence_low_slopes.csv (index 0.90 with the full history against
  1.333 = 4/3 with the cut-off, a difference of 0.43), and it locates in time where that
  difference is built up.
  '''
  rows = []
  for lo, hi in zip(tbins[:-1], tbins[1:]):
    d = df[(df.barT_n > lo) & (df.barT_n <= hi)]
    if not len(d):
      continue
    g = d.groupby('method').a_edge.median()
    n = d.groupby('method').a_edge.count()
    if not {'data', 'data_rarcut'} <= set(g.index):
      continue
    rows.append(dict(t_lo=lo, t_hi=hi, n_data=int(n['data']), n_cut=int(n['data_rarcut']),
                     a_data=g['data'], a_cut=g['data_rarcut'],
                     softening=g['data_rarcut'] - g['data']))
  out = pd.DataFrame(rows)
  if verbose and len(out):
    print(f"\n{'=== LOW-ENERGY SOFTENING IN THE TAIL (instantaneous spectra) ':=<86}")
    print('a_edge = nuFnu slope over the lowest decade of band, measured identically in both'
          '\nruns. "softening" = a_edge(cut) - a_edge(full): how much shallower the full-history'
          '\nspectrum is at its low-frequency end. 4/3 = 1.333 is the uncooled asymptote.')
    print(f"\n    {'barT/barT_f':>14} {'N':>6} {'a_edge full':>12} {'a_edge cut':>11} "
          f"{'softening':>10}")
    for _, w in out.iterrows():
      lab = f'{w.t_lo:g}-{w.t_hi:g}' if np.isfinite(w.t_hi) else f'>{w.t_lo:g}'
      print(f"    {lab:>14} {w.n_data:>6.0f} {w.a_data:12.3f} {w.a_cut:11.3f} "
            f"{w.softening:+10.3f}")
    pre = out[out.t_hi <= 1.]; post = out[out.t_lo >= 2.]
    if len(pre) and len(post):
      print(f"\n    before crossing: {pre.softening.median():+.3f}   "
            f"past 2 bar T_f: {post.softening.median():+.3f}")
  return out


def hle_validity(df, tbins=HLE_TBINS, verbose=True):
  '''
  Is the s trend physics or the band running out? Reports, against time, the fraction of bins
  whose nu^(4/3) asymptote is still reached in band (lo_conv) and the widths of the fitting
  windows, beside the s values. If s falls where lo_conv collapses and the windows shrink, the
  trend is a measurement artefact -- see hle_smoothing_evolution.
  '''
  rows = []
  for method, dm in df.groupby('method'):
    for lo, hi in zip(tbins[:-1], tbins[1:]):
      d = dm[(dm.barT_n > lo) & (dm.barT_n <= hi)]
      if not len(d):
        continue
      rows.append(dict(method=method, t_lo=lo, t_hi=hi, n=len(d),
                       lo_conv=float(d.lo_conv.mean()), dex_lo=d.dex_lo.median(),
                       dex_mid=d.dex_mid.median(), s1=d.s1.median(), s2=d.s2.median()))
  out = pd.DataFrame(rows)
  if verbose and len(out):
    print(f"\n{'=== IS THE TREND PHYSICS OR THE BAND RUNNING OUT? ':=<86}")
    for method, d in out.groupby('method'):
      print(f'\n  {method}')
      print(f"    {'barT/barT_f':>14} {'N':>5} {'lo_conv':>8} {'dex_lo':>7} {'dex_mid':>8} "
            f"{'s1':>7} {'s2':>7}")
      for _, w in d.iterrows():
        lab = f'{w.t_lo:g}-{w.t_hi:g}' if np.isfinite(w.t_hi) else f'>{w.t_lo:g}'
        f = lambda v: f'{v:7.2f}' if np.isfinite(v) else '     --'
        print(f"    {lab:>14} {w.n:>5.0f} {100*w.lo_conv:7.0f}% {f(w.dex_lo)} "
              f"{f(w.dex_mid):>8} {f(w.s1)} {f(w.s2)}")
  return out


def plot_hle(bins_by_method, sides_by_method, outdir=OUTDIR):
  '''
  Top row: the residual left by the on-axis prescription against barT/barT_f, with each bin's
  own free fit drawn underneath as the floor -- the gap between them is what the prescription
  costs, and where it opens up is the answer to "when does it stop working". The answer turns
  out to depend entirely on the column: it opens up where cells are still radiating, and not at
  all once they are dark.
  Bottom row: the freely-fitted smoothing over the same axis, as one continuous track per sweep
  point (regime labels are least reliable in the tail, so the tracks are not split by regime),
  with the on-axis reference drawn flat for comparison.
  One column per method: with the rarefaction cut the shell is dark past the shaded band, so
  that column is genuine high-latitude emission and the other is not.
  '''
  _ensure_outdir()
  methods = list(bins_by_method)
  fig, axs = plt.subplots(3, len(methods), figsize=(6.4*len(methods), 11.4), squeeze=False,
                          sharex=True)
  for j, meth in enumerate(methods):
    df = bins_by_method[meth]
    sides = [s for sbz in sides_by_method[meth] for s in sbz]
    logrs = np.array([s['logr'] for s in sides], float)
    norm = plt.Normalize(logrs.min(), logrs.max())

    ax = axs[0][j]
    for lab, col, c in (('free fit (floor)', 'rms_free', '0.55'),
                        ('on-axis $s$ frozen', 'rms_held', '#d62728')):
      g = df.groupby(pd.cut(df.barT_n, np.geomspace(1e-3, df.barT_n.max()*1.01, 40)),
                     observed=True)[col].median()
      x = np.array([iv.mid for iv in g.index])
      ax.loglog(x, g.to_numpy(), color=c, lw=1.7, label=lab)
    ax.axhline(PRESC_RMS_MAX, color='k', ls='--', lw=.9, alpha=.7,
               label=f'{PRESC_RMS_MAX:g} dex')
    ax.set_ylim(3e-3, 1.); ax.set_ylabel('rms of $\\log_{10}$(data/fit)')
    ax.set_title(f"{meth}", fontsize=10)
    if j == 0:
      ax.legend(fontsize=7, framealpha=.9)

    ax = axs[1][j]
    for s in sides:
      b = s['fs']['barT']/s['barT_f']
      c = CMAP(norm(s['logr']))
      for key, gate, ls in (('s1_fit', 's_fit_ok', '-'), ('s2_fit', 's_fit_ok', '--')):
        y = np.where(s['fs'][gate], s['fs'][key], np.nan)
        ax.loglog(b, y, color=c, lw=1.0, ls=ls)
    ax.set_ylabel('fitted $s$   (solid $s_1$, dashed $s_2$)')
    ax.set_ylim(0.05, 6.)

    ax = axs[2][j]
    g = df.groupby(pd.cut(df.barT_n, np.geomspace(1e-3, df.barT_n.max()*1.01, 40)),
                   observed=True).a_edge.median()
    xx = np.array([iv.mid for iv in g.index])
    ax.semilogx(xx, g.to_numpy(), color='#1f77b4', lw=1.7)
    ax.axhline(4/3., color='k', ls='--', lw=.9, alpha=.7, label='$4/3$ (uncooled)')
    ax.axhline(0.5, color='0.5', ls=':', lw=.9, label='$1/2$ (fast cooling)')
    ax.set_ylim(0.2, 1.5)
    ax.set_ylabel('$a_{\\rm edge}$: slope over lowest decade')
    ax.set_xlabel('$\\bar{T}/\\bar{T}_f$')
    if j == 0:
      ax.legend(fontsize=7, framealpha=.9)

    for ax in (axs[0][j], axs[1][j], axs[2][j]):
      ax.axvline(1., color='0.4', ls=':', lw=.9)
      off = sides[0]['barT_off']
      if off is not None:
        ax.axvspan(off[0]/sides[0]['barT_f'], off[1]/sides[0]['barT_f'],
                   color='0.85', alpha=.6, lw=0)
      ax.set_xlim(1e-3, None); ax.grid(alpha=.2, lw=.5)
  fig.suptitle('What breaks the spectral shape after crossing is CONTINUED EMISSION, not high '
               'latitude\nleft: cells keep radiating   right: emission cut at the rarefaction '
               'wave, so the late flux is purely high-latitude\n'
               '(dotted: shock crossing; shaded: rarefaction cut-off band)', fontsize=10)
  fig.tight_layout()
  path = os.path.join(outdir, 'hle_evolution.png')
  fig.savefig(path, dpi=200, bbox_inches='tight')
  plt.close(fig)
  print(f'-> {path}')
  return path


def hle_analysis(key=KEY, methods=('data', 'data_rarcut'), zlist=(Z_RS, Z_FS), outdir=OUTDIR,
    dfac=sb.FREE_DFAC):
  '''
  Quantify what high-latitude emission does to the spectral shape: how fast the on-axis
  prescription stops describing the spectra, and how the smoothing that would describe them
  evolves. Run over both methods, because with the reference method the cells never stop
  radiating (they run to barT ~ 650) so its late emission is NOT high-latitude only; with
  rar_cut the shell is dark by barT ~ 2 and it is. The difference between the two columns is
  the part that belongs to continued emission rather than to geometry.

  Deliberately kept out of main(): it doubles the tracker cost.
  '''
  _ensure_outdir()
  bins_by_method, sides_by_method, refs_by_method = {}, {}, {}
  for meth in methods:
    print(f'\n--- {meth} ---', flush=True)
    sbz = [load_side(z=z, key=key, method=meth, dfac=dfac) for z in zlist]
    sides_by_method[meth] = sbz
    allsides = [s for sides in sbz for s in sides]
    # the reference is each method's OWN on-axis median, so the frozen values are the ones that
    # method validated; they are printed to make any difference between methods visible
    refs = {}
    for rg in REGIMES:
      r = _presc_row(allsides, rg, onaxis=True)
      if r is not None:
        refs[rg] = (r['s1'], r['s2'], r['sig'])
    refs_by_method[meth] = refs
    print('  on-axis reference: ' + '  '.join(
        f"{rg}: " + (f"s={v[2]:.3f}" if rg == 'MC' else f"({v[0]:.3f}, {v[1]:.3f})")
        for rg, v in refs.items()))
    bins_by_method[meth] = hle_bins(sbz, meth, refs)

  df = pd.concat(bins_by_method.values(), ignore_index=True)
  deg = hle_report(df)
  deg_rg = hle_report(df, by_regime=True)
  sat = hle_smoothing_evolution(df)
  val = hle_validity(df)
  soft = low_energy_softening(df)
  plot_hle(bins_by_method, sides_by_method, outdir=outdir)
  for d, fn in ((df, 'hle_bins.csv'), (deg, 'hle_degradation.csv'),
                (deg_rg, 'hle_degradation_by_regime.csv'),
                (sat, 'hle_saturation.csv'), (val, 'hle_validity.csv'),
                (soft, 'low_energy_softening.csv')):
    d.to_csv(os.path.join(outdir, fn), index=False)
  print('\n-> ' + outdir + '/' + ', '.join(
      ('hle_bins.csv', 'hle_degradation.csv', 'hle_degradation_by_regime.csv',
       'hle_saturation.csv', 'hle_validity.csv', 'low_energy_softening.csv')))
  swp.trim_pngs(outdir)
  return df, deg, sat, val
