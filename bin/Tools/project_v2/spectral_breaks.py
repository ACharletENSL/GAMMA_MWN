# -*- coding: utf-8 -*-
# @Author: acharlet

'''
WHICH ROUTE THE PAPER USES -- read this before quoting any number out of this module.

The paper (Charlet et al., cooling regimes) uses ONE route and one only: the self-contained
segment route at the bottom of this file,

    sweep_gammacm.identify_segments -> breaks_from_identified -> smoothing_from_identified

driven by segment_route.py. Everything else here is SUPERSEDED for the paper and kept only
because other modules still stand on it:

  breaks_from_segments / track_breaks_segments   SUPERSEDED by breaks_from_identified. Same
      idea -- breaks as the crossing of fitted segments -- but it selects its windows with
      fit_segments (which thresholds the slope around the expected value on the FLATTENED
      spectrum) rather than with identify_segments, and it returns no shape class. Still
      referenced by nuc_validation.TRACKERS as a cross-check.
  free_slopes / track_free_slopes                NOT stale, and not a break-measurement route:
      it is the test that the computed spectra carry the 4/3 and 1-p/2 asymptotes at all
      (slope_validation), which is exactly what makes HOLDING them in the segment route
      legitimate. Keep. Its per-regime SMOOTHING table, however, is superseded -- see
      slope_validation's own banner.
  fit_smoothing_held / fit_single_break          LIVE. These are the estimators the segment
      route calls; nothing about them is stale, only the geometry that used to be fed to them.

The rest of this docstring describes breaks_from_segments and is kept for the reasoning it
records, not as a description of the current method.

---

Spectral breaks measured as the INTERSECTION OF THE ASYMPTOTIC POWER-LAW SEGMENTS,
independently of how sharply the spectrum turns over at them.

Why this exists. sweep_gammacm.fit_gs02_spectrum fits the whole smoothed Granot & Sari
template and holds the smoothing at (s1, s2) = (1.3, 2.0). That fitter is sound -- it
recovers a known nu_c to 1.000 from a synthetic GS02 spectrum at every break separation --
but the template can only absorb a MISMATCH in the true break sharpness by moving the
breaks, and the leverage grows as the breaks close. Injecting a true s = (0.9, 1.5) biases
the recovered nu_c to 0.61/0.70/0.72 at separation 10/21.5/26.7, relaxing to ~1 only by
separation >~600. On the computed shell spectra, refitting with free_s moves nu_c by up to
25x where the separation is 10-40, for an rms gain of ~1e-3 dex -- i.e. the data does not
constrain that direction at all. Since the median separation runs from ~20 to ~1e5 across
the cooling-regime sweep, this is a REGIME-DEPENDENT bias sitting in the reference against
which any nu_c(T) model is judged.

The break parameter is the asymptote crossing -- exactly. granot_sari_syn writes each break
as F = F_ext [ y**(-s b1) + y**(-s b2) ]**(-1/s) with F_ext "the flux where the two power
laws extrapolate to cross"; its two asymptotes are F_ext y**b1 and F_ext y**b2, which meet
at y = 1, i.e. AT the break, for any s. The same holds for the upper break, which enters as
[1 + y**(s2 db)]**(-1/s2). So measuring where the segments cross targets precisely the
quantity the template parameterises, while never referring to s. What the smoothing does
control is how far the spectrum sags BELOW the crossing: exactly 2**(-1/s) there, which is
what makes the smoothing recoverable afterwards (smoothing_from_deficit).

This is a third method, not a variant of an existing one:
  measure_regime (sweep_gammacm:604)  - knee scan: takes the nuFnu PEAK as the upper break
    and scans down to where the slope steepens past ~0.92. Both are smoothing-dependent
    positions, hence its documented bias (0.73 low-side, 1.5-2.7 high-side). SUPERSEDED.
  fit_gs02_spectrum                   - full template, smoothing held: the degeneracy above.
    SUPERSEDED as a measurement; still the scaffold slope_validation places its windows from.
  breaks_from_segments (here)         - asymptote intersection: smoothing-free by construction,
    and NaN rather than a biased number when the segments are too short to be seen.
    SUPERSEDED by the fourth method below, which is the one the paper uses.
  breaks_from_identified (here, bottom) - the same intersection, but of the segments
    identify_segments found, so the regime is the shape class and no window is placed where
    no segment was identified. THE PAPER ROUTE.

A fourth method is deliberately ABSENT: detecting the segments without saying what slope to
expect, i.e. finding where the spectrum is straight rather than where it matches one of the
four theory slopes. It is the obvious generalisation and it does not survive contact with
these spectra, in either of its two forms.

  Pointwise curvature (|ds/dlog nu| < tol) -- fails on the DATA. On a synthetic GS02 spectrum
    it looks decisive: plateau curvature <= 0.046 against >= 0.365 in the knees, a factor 8.
    On the computed spectra, INSIDE windows identify_segments accepted as genuine 2-6.8 dex
    segments, it reaches 0.23-0.71 -- overlapping the knee value entirely. And that is not
    numerical noise to be filtered away: the residual of those segments from a straight line
    has a lag-1 autocorrelation of 0.998, i.e. it is smooth, coherent CURVATURE. The low and
    high segments are asymptotes the spectrum only approaches (see FREE_DFAC), so they are
    genuinely curved everywhere, most of all near their ends. No amount of smoothing or extra
    frequency sampling removes it, because it is signal. A curvature threshold therefore has
    nothing clean to threshold on, while the value test survives because SLOPE_TOL = 0.15 is
    several times the in-segment slope departure (0.036-0.054) and takes no second derivative.
  Windowed flatness (rms of a free line fit < tol) -- does not fail on noise, but is not
    SCALE-FREE, so it answers a different question than the one asked. It is more sensitive
    (it finds a mid-slope run at 1.1-2.5 dex of break separation against identify_segments'
    2.75) but the runs it finds there are 0.6-1.2 dex wide with slopes 0.514-0.568 for a true
    0.5, and its threshold is non-monotonic in tol. Any smooth curve passes an rms test over a
    short enough window: on the logr=-3 declined spectra every 0.6-dex window at the band
    bottom fits to an rms of 0.003-0.005 dex while the slope drifts 0.20-0.45 across 2.1 dex.
    It therefore does not remove the width gate (SEG_MIN_MID_DEX), it relocates it -- trading
    an assumption about the slope's VALUE for one about the window's WIDTH.

What did survive from that test is edge_slope_drift, at the bottom of this module: sliding the
free fit is the right way to tell a segment from a knee, even though thresholding its residual
is not.

nuFnu log-log slopes are a = 1 + beta, with beta the F_nu index:
  a_lo = 4/3      (beta = 1/3, below both breaks)
  a_mid = 1/2     fast cooling (beta = -1/2)   |  (3-p)/2  slow cooling (beta = -(p-1)/2)
  a_hi = 1 - p/2  (beta = -p/2, above both breaks)
The low and high slopes are held at those values; the MID slope is fitted free, because the
shell-integrated mid segment is documented to land genuinely between the two asymptotes
(-0.67/-0.56 measured in the marginal regime, see granot_sari_syn's beta_mid), and imposing
an asymptote there would reintroduce a bias of the very kind this module removes.

Holding a_lo and a_hi is only legitimate if the computed spectra actually carry them, which
nothing in the chain checked until free_slopes / track_free_slopes / free_slope_convergence
were added at the bottom of this module -- see slope_validation.py for the driver and the
measured answer.
'''

import numpy as np
from scipy.optimize import least_squares

from phys_functions import granot_sari_syn, syn_cutoff_R, syn_cutoff_R_smeared

# --- defaults -------------------------------------------------------------------------
SLOPE_TOL = 0.15      # |local slope - asymptote| accepted into a fixed-slope segment window
SLOPE_SMOOTH = 5      # boxcar width (samples) on the local-slope array; the answer must not
                      # depend on it, and does not: feeding identify_segments synthetic GS02
                      # spectra of known break separation at 33 pts/dex, the separation at
                      # which the mid segment first registers is 2.80 dex for EVERY smoothing
                      # in 1, 3, 5, 7, 9, 15 (scan step 0.1 dex). The estimator is
                      # np.gradient (+-1 sample) o _boxcar(n) (+-(n-1)/2), i.e. a +-(n+1)/2
                      # sample stencil -- 0.182 dex at n=5, 33 pts/dex -- and a boxcar shrinks
                      # a plateau symmetrically without moving its centre slope, so the width
                      # gates below never see it. (An earlier version of this comment cited a
                      # check in nuc_validation that was never written.)
MIN_PTS = 5           # minimum samples in a segment window for it to be fitted. At 33 pts/dex
                      # it spans only 0.121 dex and is SLACK -- MIN_DEX binds instead, and this
                      # gate only starts to bite below ~16 pts/dex (see sweep_gammacm.
                      # NNU_PER_DEC for the resolution ladder).
MIN_DEX = 0.25        # minimum log10 width of a segment window = 10 samples at 33 pts/dex.
                      # This is the width of the IDENTIFIED window, which is narrower than the
                      # segment that produced it, because a physical break is smooth and its
                      # curvature eats into its own plateau. Measured on granot_sari_syn at
                      # s = (1.3, 2.0), p = 2.5, the TRUE width needed to clear this gate is
                      # 0.89 dex of band below the lower break for the 4/3 segment (0.33 dex
                      # if the kink were sharp) and 0.35 dex between nu_m and nu_M/CUT_FAC for
                      # the 1-p/2 one. The mid segment is gated separately and far harder --
                      # see MIN_MID_DEX and sweep_gammacm.SEG_MIN_MID_DEX.
CUT_FAC = 3.          # the high segment is fitted below nuM/CUT_FAC when the cutoff is NOT
                      # divided out (flatten=False)
FIT_DEC = 8.          # ignore everything more than this many decades below the peak
# How close the FITTED beta_mid must sit to an asymptote to name the regime. Looser than
# fit_gs02_spectrum's 0.02, which applies to a least-squares parameter that pins exactly at
# a bound; here beta_mid is a regression slope over a finite plateau and carries real
# scatter. For p = 2.5 the asymptotes are -0.5 (FC) and -0.75 (SC), so 0.08 leaves an
# unnamed band of (-0.67, -0.58) -- which is where the shell-integrated mid segment is
# actually measured to sit in the marginal regime (-0.67/-0.56, see granot_sari_syn).
BMID_TOL = 0.08
# Minimum decades of straight mid segment for a free-slope measurement to be trusted; see
# breaks_from_segments. Ignored when the mid slope is held.
# In physical terms: the identified window runs ~1.06 dex short of the true break separation
# on a GS02-smoothed spectrum (s = 1.3, 2.0; measured at 33 pts/dex), so 2.0 dex of window
# means the breaks are >~ 3.06 dex apart -- a nu_c/nu_m ratio of ~1150.
# NOT the same number as sweep_gammacm.SEG_MIN_MID_DEX (1.7), deliberately: that one gates a
# HELD-slope shape verdict, where no break position is being fitted and a short window costs
# only a classification. This one gates a FREE-slope break FIT, whose accuracy is set by the
# plateau width alone (see breaks_from_segments: within 6% once mid_dex >~ 2.5, i.e. a true
# separation >~ 3.55 dex, degrading to a factor ~2 by mid_dex ~ 0.85). 2.0 is the point below
# which that fit stops being worth reporting; 1.7 is where a shape class stops being legible.
MIN_MID_DEX = 2.0

# --- free-slope diagnostic ------------------------------------------------------------
# Everything above HOLDS the low and high slopes at 4/3 and 1-p/2, and granot_sari_syn
# hardcodes the matching F_nu indices 1/3 and -p/2, so nothing in the measurement chain ever
# checks them. fit_segments cannot be repurposed for it either: it selects those windows by
# thresholding the local slope AROUND the expected value (|s - a| < SLOPE_TOL), so it would
# only confirm that samples chosen for having slope ~4/3 have slope ~4/3.
# free_slopes imposes nothing. It places the windows GEOMETRICALLY -- a fixed factor in
# frequency away from breaks located by an independent fit -- and fits a free line in each.
FREE_DFAC = 10.        # a window starts this factor in frequency away from the nearest break.
                       # The asymptotes are approached slowly, so residual break curvature
                       # biases the recovered a_lo low: measured -0.018 at dfac=3, -0.008 at
                       # 10, -0.003 at 30, -0.0004 at 100, 0.0000 at 300, and the synthetic
                       # control (granot_sari_syn, where a_lo IS 4/3 by construction) shows
                       # the same approach at the same rate. dfac therefore trades bias
                       # against coverage: 10 is the knee, its bias sitting under the
                       # bin-to-bin scatter while the low window survives in most bins that
                       # have a low segment at all.
FREE_MIN_PTS = 6       # a window carrying fewer samples than this is not fitted
FREE_MIN_DEX = 0.4     # ... nor is one narrower than this in log10 nu
FREE_DSCAN = (3., 10., 30., 100., 300.)    # dfac ladder walked by free_slope_convergence
FREE_CONV_FAC = 3.     # a_lo is re-measured with the window pushed this much further out;
                       # if the two disagree the asymptote has not been reached inside the
                       # band and no slope is reported (lo_converged False)
S_STABLE_TOL = 0.05    # relative move in s between the dfac and FREE_CONV_FAC*dfac windows
                       # below which the deficit estimate is called stable. Most bins fail
                       # it -- see _free_smoothing; that is the finding, not a bug.
SEP_CLEAN = 100.       # break separation above which the 2**(-1/s) deficit identity applies
                       # to each break on its own; below it the two turnovers overlap, the
                       # deficits add and s comes out too large. Same threshold
                       # smoothing_from_deficit uses for its `clean` flag.
FREE_CONV_TOL = 0.015  # tolerance of that agreement. Separates the two populations cleanly:
                       # where the asymptote IS reached the dfac=10 -> 30 step moves a_lo by
                       # 0.003-0.005, where it is not (the high-latitude tail, below) it moves
                       # by 0.03-0.07.
                       # WHY IT IS NEEDED: in the HLE the lower break is smeared by the
                       # angular superposition into something far smoother than any
                       # single-zone value, and the nu^(4/3) asymptote is pushed off the
                       # bottom of the observable band. The measured a_lo then falls steadily
                       # (to ~1.06 by bar{T}/bar{T}_f ~ 7, correlating with window width at
                       # r = +0.96) WITHOUT the spectrum having lost the 4/3 segment: a
                       # synthetic granot_sari_syn whose a_lo is 4/3 by construction returns
                       # the same values once s1 is dropped to ~0.4-0.5 and the band bottom
                       # is placed 2.4 decades below the break, as it is there. Reporting the
                       # unconverged number would read as a physical slope change; it is a
                       # band limit, so the measurement is declined instead.


def _boxcar(y, n):
  '''centred moving average, edges handled by shrinking the window (no wraparound)'''
  n = int(n)
  if n <= 1:
    return np.asarray(y, float)
  y = np.asarray(y, float)
  c = np.concatenate(([0.], np.cumsum(y)))
  out = np.empty_like(y)
  for i in range(len(y)):
    a, b = max(0, i - n//2), min(len(y), i + n//2 + 1)
    out[i] = (c[b] - c[a])/(b - a)
  return out


def segment_slopes(x, sp, smooth=SLOPE_SMOOTH):
  '''
  Local log-log slope d ln(nuFnu)/d ln(nu) on the finite spectrum grid -- the
  np.gradient(ly, lx) idiom of sweep_gammacm.measure_regime, lightly smoothed because
  numerical differentiation of a finite grid is noisy and the segment windows below are
  selected by thresholding this array.

  Returns (lx, ly, slope) on the good samples only (finite, positive).
  '''
  x = np.asarray(x, float); sp = np.asarray(sp, float)
  good = np.isfinite(sp) & (sp > 0.) & np.isfinite(x) & (x > 0.)
  lx, ly = np.log10(x[good]), np.log10(sp[good])
  o = np.argsort(lx)
  lx, ly = lx[o], ly[o]
  if len(lx) < 4:
    return lx, ly, np.full(len(lx), np.nan)
  return lx, ly, _boxcar(np.gradient(ly, lx), smooth)


def _widest_run(mask, lx, min_pts=MIN_PTS, min_dex=MIN_DEX):
  '''
  Index slice of the widest contiguous True run of `mask`, in log10 x width. None if no
  run is wide enough -- which is how this module says "the segment is not resolved here"
  instead of fitting a line to a knee.
  '''
  best = None
  i = 0
  n = len(mask)
  while i < n:
    if not mask[i]:
      i += 1; continue
    j = i
    while j + 1 < n and mask[j+1]:
      j += 1
    if (j - i + 1) >= min_pts and (lx[j] - lx[i]) >= min_dex:
      if best is None or (lx[j] - lx[i]) > (lx[best[1]] - lx[best[0]]):
        best = (i, j)
    i = j + 1
  return best


# --- flat core ------------------------------------------------------------------------
# How far the free slope HOLDS across a window, as a DIAGNOSTIC. It is deliberately not a gate
# anywhere: gating segment identification on it was tried and reverted, because how flat a real
# mid segment is depends on the cooling regime. In slow cooling the core lands within 0.016 of
# the (3-p)/2 asymptote; in fast cooling the shell-integrated segment is curved and hardened by
# the smearing of nu_c across cells, settling at 0.567-0.590 against an asymptote of 0.500 while
# the breaks are 3.5-4.4 dex apart and unambiguously resolved. Any tolerance loose enough to
# admit the second is loose enough to admit a knee. See sweep_gammacm.SEG_MIN_MID_DEX for the
# criterion that does work, and why.
# What it IS good for is telling a settled slope from a sweeping one when you already know the
# separation -- reading a spectrum, not classifying it.
CORE_WIN_DEX = 0.4     # width of each sliding free-fit window, in dex
CORE_STEP_DEX = 0.2    # step between them -- half-overlapping, so two windows share half their
                       # samples and a core is located to ~CORE_STEP_DEX
CORE_TOL = 0.02        # slopes spanning no more than this over consecutive windows count as ONE
                       # settled core


def sliding_slopes(lx, ly, lo=None, hi=None, win=CORE_WIN_DEX, step=CORE_STEP_DEX,
    min_pts=MIN_PTS):
  '''
  Free straight-line slopes in half-overlapping windows across [lo, hi] of an already-sorted
  (lx, ly). The primitive behind both flat_core and edge_slope_drift: sliding a free fit is
  how this module tells a settled power law from a smooth turn, since a turn is locally
  straight at any tolerance (rms 0.003-0.005 dex inside a knee) but does not hold its slope.

  Returns dict(slopes, x0, rms, n) over the windows that carried min_pts samples.
  '''
  lx = np.asarray(lx, float); ly = np.asarray(ly, float)
  out = dict(slopes=np.array([]), x0=np.array([]), rms=np.array([]), n=0)
  if len(lx) < min_pts:
    return out
  lo = float(lx.min()) if lo is None else float(lo)
  hi = float(lx.max()) if hi is None else float(hi)
  sl, x0s, rm = [], [], []
  off = lo
  while off + win <= hi + 1e-12:
    m = (lx >= off) & (lx < off + win)
    if m.sum() >= min_pts:
      a, b = np.polyfit(lx[m], ly[m], 1)
      sl.append(float(a)); x0s.append(float(off))
      rm.append(float(np.sqrt(np.mean((ly[m] - a*lx[m] - b)**2))))
    off += step
  if not sl:
    return out
  out.update(slopes=np.array(sl), x0=np.array(x0s), rms=np.array(rm), n=len(sl))
  return out


def flat_core(lx, ly, lo=None, hi=None, tol=CORE_TOL, win=CORE_WIN_DEX, step=CORE_STEP_DEX,
    min_pts=MIN_PTS):
  '''
  The widest stretch of [lo, hi] over which the free slope actually HOLDS: the longest run of
  consecutive sliding windows whose slopes span no more than `tol`, reported as the frequency
  width it covers.

  dex is 0 when no two consecutive windows agree, which is the answer for a knee. slope is the
  mean over the core, and is a free measurement -- nothing is imposed, so it can be compared
  against a theory asymptote afterwards rather than assumed equal to one.

  Returns dict(dex, slope, n, i0), n = windows in the core.
  '''
  out = dict(dex=0., slope=np.nan, n=0, i0=-1)
  w = sliding_slopes(lx, ly, lo=lo, hi=hi, win=win, step=step, min_pts=min_pts)
  if w['n'] < 2:
    return out
  s, x0 = w['slopes'], w['x0']
  best = None
  i = 0
  while i < len(s):
    j = i
    while j + 1 < len(s) and (max(s[i:j+2]) - min(s[i:j+2])) <= tol:
      j += 1
    if j > i and (best is None or (x0[j] - x0[i]) > (x0[best[1]] - x0[best[0]])):
      best = (i, j)
    i += 1
  if best is None:
    return out
  i, j = best
  out.update(dex=float(x0[j] - x0[i] + win), slope=float(np.mean(s[i:j+1])),
             n=int(j - i + 1), i0=int(i))
  return out


# Grid for the OPT-IN smeared-cutoff fit (measure_cutoff_nuM(smear=True)). In dex of nu_M
# spread; the sweep spectra want 0.02-0.10, so 0.20 is generous headroom and 0.01 resolves the
# minimum (rms moves measurably between adjacent steps -- see syn_cutoff_R_smeared).
SMEAR_SIGMAS = np.arange(0., 0.2001, 0.01)


def measure_cutoff_nuM(x, sp, psyn, smooth=SLOPE_SMOOTH, flatten=True, smear=False,
    sigmas=SMEAR_SIGMAS):
  '''
  nu_M from the high-frequency rolloff alone, then optionally divide it out.

  The cutoff SHAPE is physically fixed -- syn_cutoff_R(u), the single-electron emissivity
  R(x) normalised by its own low-x asymptote -- so the rolloff carries exactly one free
  parameter. Holding the 1-p/2 slope of the segment beneath it, a scan over x_M minimises
  the residual of log10(sp) - [a_hi*lx + c_hi] - log10(syn_cutoff_R(x/x_M)) over the top
  of the spectrum, with c_hi profiled out analytically at each x_M (it is a pure offset).

  With flatten=True the returned spectrum is sp/syn_cutoff_R(x/x_M), on which the 1-p/2
  segment is a clean power law over its whole range instead of only below ~nuM/3 -- which
  is what lets the high asymptote be fitted where it is best determined.

  smear=True fits the SUPERPOSED rolloff instead (syn_cutoff_R_smeared), adding one parameter
  -- the dex spread of nu_M across the contributing cells -- and scanning it jointly with x_M
  over `sigmas`. It is OPT-IN and default OFF: with smear=False every number this returns is
  bit-identical to before the smeared shape existed, so no existing result moves until a caller
  asks for it. Turn it on when nu_M ITSELF is the quantity of interest; it is not needed for
  anything measured below the cut-off (see below).

  WHY THE DEFAULT IS STILL THE UNSMEARED SHAPE. The single-zone fit is biased: on a synthetic
  superposition of known sigma = 0.12 dex it returns nu_M 1.30x the true median, and the spread
  inferred from the sweep's own rolloffs (sigma ~ 0.10-0.13 at peak epochs) puts the shipped
  nu_M some +23-35% high there. But that bias barely propagates. Measured end to end by
  re-flattening with the smeared shape and refitting: b_lo moves by x1.0000 in every case,
  a_mid by +0.0000, mid_dex not at all, and only b_hi moves (median x1.019, worst x1.105).
  Two reasons -- the convolution is symmetric in log nu, so it broadens the rolloff about nu_M
  without moving its onset (the departure point shifts <= 0.07 dex at the 1% level and ~0.00 at
  the 10% and 50% levels), and the nu_M shift itself is only 0.04-0.09 dex against segments
  spanning several decades. b_hi is the exception because its intercept is fitted partly inside
  the rolloff and the crossing amplifies it: dlog b_hi = dc_hi/(a_hi - a_mid) with the
  denominator ~ -0.49. b_hi is nu_m in fast cooling and nu_c in slow, so the residual ~2%
  (worst 10%) systematic lands on nu_c on the slow-cooling side and belongs in that error
  budget rather than being assumed zero.

  Returns dict(nuM, sp_flat, ok), plus sigma and rms when smear=True.
  '''
  lx, ly, s = segment_slopes(x, sp, smooth)
  a_hi = 1. - psyn/2.
  xg = 10**lx
  out = dict(nuM=np.nan, sp_flat=np.asarray(sp, float), ok=False)
  if smear:
    out.update(sigma=np.nan, rms=np.nan)
  if len(lx) < 8:
    return out
  # work above the peak, over the top FIT_DEC decades
  i_pk = int(np.argmax(ly))
  m = (np.arange(len(lx)) > i_pk) & (ly > ly.max() - FIT_DEC)
  if m.sum() < 6:
    return out
  lxm, lym = lx[m], ly[m]
  grid = 10**np.linspace(lxm.min() - 1., lxm.max() + 3., 400)
  sig_grid = np.asarray(sigmas, float) if smear else np.array([0.])
  best = (np.inf, np.nan, np.nan)
  # C_sigma depends on u = x/xM only, so changing xM is a pure shift in log u: tabulate
  # log10 C_sigma ONCE per sigma on a log-u grid covering every trial, then interpolate.
  # Evaluating the kernel per (sigma, xM) instead would be ~3e8 calls (see the shape's
  # docstring). Outside the tabulated range the factor is 1 below and unusable above, which
  # the isfinite guard below already rejects.
  # The unsmeared branch keeps evaluating the kernel directly, so smear=False stays
  # bit-identical to before this option existed; only the smeared branch is tabulated.
  lu = np.linspace(lxm.min() - np.log10(grid.max()) - 1.,
                   lxm.max() - np.log10(grid.min()) + 1., 4000)
  for sig in sig_grid:
    if smear:
      with np.errstate(divide='ignore', invalid='ignore'):
        tab = np.log10(syn_cutoff_R_smeared(10**lu, sig))
    for xM in grid:
      if smear:
        corr = np.interp(lxm - np.log10(xM), lu, tab, left=0., right=np.nan)
      else:
        with np.errstate(divide='ignore', invalid='ignore'):
          corr = np.log10(syn_cutoff_R(10**lxm/xM))
      if not np.all(np.isfinite(corr)):
        continue
      r = lym - a_hi*lxm - corr            # = c_hi + residual
      c = float(np.mean(r))                 # profile the offset out
      rms = float(np.sqrt(np.mean((r - c)**2)))
      if rms < best[0]:
        best = (rms, xM, sig)
  if not np.isfinite(best[1]):
    return out
  xM = float(best[1])
  out['nuM'] = xM
  out['ok'] = True
  if smear:
    out['sigma'], out['rms'] = float(best[2]), float(best[0])
  if flatten:
    with np.errstate(divide='ignore', invalid='ignore'):
      f = (syn_cutoff_R(np.asarray(x, float)/xM) if not smear
           else syn_cutoff_R_smeared(np.asarray(x, float)/xM, best[2]))
    sp_f = np.asarray(sp, float)/np.where(np.isfinite(f) & (f > 0.), f, np.nan)
    out['sp_flat'] = sp_f
  return out


def _tangent_mid(lx, ly, s, interior, psyn):
  '''
  The mid line anchored where the slope REACHES a theory asymptote, for spectra that carry no
  plateau to fit one over.

  Why it exists. fit_segments' plateau needs MIN_MID_DEX of straight mid segment, and below
  that it declines -- correctly, because a line fitted to a knee is a biased line. But a
  spectrum whose slope runs monotonically from 4/3 down to 1-p/2 must PASS THROUGH the mid
  asymptote exactly once on the way, whether or not it ever settles there, and at that single
  point the tangent is the mid asymptote by construction. Anchoring there uses no plateau and
  no fit: the slope is held at the theory value and only the intercept comes from the data.

  On synthetic GS02 spectra of known breaks this recovers b_lo, b_hi to 1-3% down to a
  separation of 3.0 dex, where the plateau fit needs 4.0 and declines below it; it stays
  usable (7% on the separation) to 2.5 dex and degrades past that. It does NOT fit better than
  the plateau where a plateau exists -- the two are measured equal, 0.019-0.055 rms against
  0.021-0.054 -- so it is a fallback for coverage, never a replacement.

  Which asymptote to cross is a real choice: for p = 2.5 both 1/2 and (3-p)/2 lie inside the
  interior slope range, so both are crossed. The candidate nearest the interior MEDIAN slope
  wins -- the value the spectrum spends most of its interior near, which is the same statistic
  the plateau iteration seeds from. Anchoring instead at the flattest interior point (which
  would keep the mid slope free) was tried and is not robust: it locks onto the tail of the
  4/3 asymptote in fast cooling, returning a_mid ~ 1.18 and an rms of 0.077-0.084.

  CONSEQUENCE FOR THE REGIME LABEL, which callers must not read past. The slope is HELD at a
  candidate here, so beta_mid comes back exactly -1/2 or -(p-1)/2 and breaks_from_segments'
  classifier can only ever answer FC or SC on these spectra -- never MC, whose whole meaning is
  a mid slope sitting between the asymptotes. A tangent-anchored bin is therefore evidence
  about WHERE THE BREAKS ARE, not about whether the spectrum is marginal; the marginality
  question is answered by sweep_gammacm.identify_segments (which does return MC), or by the
  plateau fit where one exists. Mixing the two populations in a
  regime census would manufacture a spurious FC/SC excess exactly in the marginal regime, which
  is where this fallback does all its work. Check mid_from before counting.

  Returns (a_mid, c_mid, x_mid) or None, x_mid being log10 of the anchor frequency.
  '''
  if interior.sum() < MIN_PTS:
    return None
  L, S, Y = lx[interior], s[interior], ly[interior]
  o = np.argsort(L); L, S, Y = L[o], S[o], Y[o]
  a_mid = min((0.5, (3. - psyn)/2.), key=lambda a: abs(a - float(np.median(S))))
  k = np.where((S[:-1] - a_mid)*(S[1:] - a_mid) <= 0.)[0]
  if not len(k):
    return None
  k = int(k[-1])                      # the crossing closest to the upper break
  dS = S[k+1] - S[k]
  t = (a_mid - S[k])/dS if dS != 0. else 0.
  x_mid = float(L[k] + t*(L[k+1] - L[k]))
  return float(a_mid), float(np.interp(x_mid, L, Y) - a_mid*x_mid), x_mid


def fit_segments(x, sp, psyn, smooth=SLOPE_SMOOTH, slope_tol=SLOPE_TOL,
    min_pts=MIN_PTS, min_dex=MIN_DEX, mid_slope=None, min_mid_dex=MIN_MID_DEX):
  '''
  The three power-law segments of one nuFnu spectrum.

  Low and high: slopes HELD at 4/3 and 1-p/2, windows selected where the local slope is
  within slope_tol of that value, intercept = mean(ly - a*lx) over the window (the
  least-squares solution for a line of known slope).
  Mid: BOTH slope and intercept fitted, over the widest run strictly between the low and
  high windows -- see the module docstring for why this one is not imposed.

  Assumes any cutoff has already been divided out (measure_cutoff_nuM(flatten=True)),
  otherwise the high window is contaminated by the rolloff.

  Returns dict(a_lo, c_lo, a_mid, c_mid, a_hi, c_hi, win_lo, win_mid, win_hi, ok, ...).
  '''
  lx, ly, s = segment_slopes(x, sp, smooth)
  a_lo, a_hi = 4./3., 1. - psyn/2.
  out = dict(a_lo=a_lo, a_hi=a_hi, a_mid=np.nan, c_lo=np.nan, c_mid=np.nan, c_hi=np.nan,
             win_lo=None, win_mid=None, win_hi=None, ok=False, n=len(lx))
  if len(lx) < 8 or not np.any(np.isfinite(s)):
    return out
  keep = ly > ly.max() - FIT_DEC
  i_pk = int(np.argmax(ly))

  def fixed_slope_window(a, side):
    m = keep & np.isfinite(s) & (np.abs(s - a) < slope_tol)
    m &= (np.arange(len(lx)) < i_pk) if side == 'lo' else (np.arange(len(lx)) > i_pk)
    return _widest_run(m, lx, min_pts, min_dex)

  w_lo = fixed_slope_window(a_lo, 'lo')
  w_hi = fixed_slope_window(a_hi, 'hi')
  if w_lo is None or w_hi is None:
    out['win_lo'], out['win_hi'] = w_lo, w_hi
    return out
  out['c_lo'] = float(np.mean(ly[w_lo[0]:w_lo[1]+1] - a_lo*lx[w_lo[0]:w_lo[1]+1]))
  out['c_hi'] = float(np.mean(ly[w_hi[0]:w_hi[1]+1] - a_hi*lx[w_hi[0]:w_hi[1]+1]))
  out['win_lo'], out['win_hi'] = w_lo, w_hi

  # mid: strictly interior, and NOT part of either fixed-slope window
  interior = np.zeros(len(lx), bool)
  interior[w_lo[1]+1:w_hi[0]] = True
  interior &= keep & np.isfinite(s)
  # exclude the knees: keep where the slope is between the two asymptotes with a margin
  interior &= (s < a_lo - slope_tol) & (s > a_hi + slope_tol)
  if interior.sum() < min_pts:
    return out

  # The mid slope is DISCOVERED, not imposed, so its window cannot be selected by a
  # tolerance around a known value the way the low/high ones are. Iterate to a fixed
  # point instead: guess the plateau from the interior median, keep the samples within
  # slope_tol of that guess, refit, repeat. Without this the window keeps the two
  # turnovers flanking the plateau and the fitted slope is dragged off it -- which
  # showed up directly as a ~0.85 bias in the recovered nu_c on synthetic spectra.
  # mid_slope not None holds it instead (see breaks_from_segments): then only the
  # intercept is free, exactly as for the low and high segments.
  held = mid_slope is not None
  a_mid = float(mid_slope) if held else float(np.median(s[interior]))
  w_mid = None
  for _ in range(1 if held else 6):
    m = interior & (np.abs(s - a_mid) < slope_tol)
    w = _widest_run(m, lx, min_pts, min_dex)
    if w is None:
      break
    i, j = w
    if held:
      w_mid = w; break
    A = np.polyfit(lx[i:j+1], ly[i:j+1], 1)
    if w == w_mid:
      w_mid = w; a_mid = float(A[0]); break
    w_mid, a_mid = w, float(A[0])
  # No plateau wide enough to carry a free line -- fall back to the TANGENT anchor, which
  # needs no plateau at all (see _tangent_mid). Only when the slope is not held: a held mid
  # slope with no window is a caller error, not something to guess around.
  if (w_mid is None or (lx[w_mid[1]] - lx[w_mid[0]]) < min_mid_dex) and not held:
    tan = _tangent_mid(lx, ly, s, interior, psyn)
    if tan is not None:
      out['a_mid'], out['c_mid'], out['x_mid'] = tan
      out['mid_dex'] = 0.
      out['mid_from'] = 'tangent'
      out['ok'] = True
      return out
  if w_mid is None:
    return out
  i, j = w_mid
  if held:
    out['a_mid'] = a_mid
    out['c_mid'] = float(np.mean(ly[i:j+1] - a_mid*lx[i:j+1]))
  else:
    A = np.polyfit(lx[i:j+1], ly[i:j+1], 1)
    out['a_mid'], out['c_mid'] = float(A[0]), float(A[1])
  out['win_mid'] = w_mid
  out['mid_dex'] = float(lx[j] - lx[i])
  out['mid_from'] = 'plateau'
  out['ok'] = True
  return out


def breaks_from_segments(x, sp, psyn, smooth=SLOPE_SMOOTH, slope_tol=SLOPE_TOL,
    min_pts=MIN_PTS, min_dex=MIN_DEX, nuM=None, bmid_tol=BMID_TOL, mid_slope=None,
    min_mid_dex=MIN_MID_DEX):
  '''
  The two breaks of one nuFnu spectrum sp(x), as the crossings of the fitted segments:

      b_lo = 10**((c_lo - c_mid)/(a_mid - a_lo))     4/3 meets the mid line
      b_hi = 10**((c_mid - c_hi)/(a_hi - a_mid))     mid line meets 1-p/2

  Naming follows the FITTED mid slope, beta_mid = a_mid - 1: nearer -1/2 the spectrum is
  fast-cooling and nu_c is the LOWER break; nearer -(p-1)/2 it is slow-cooling and nu_c is
  the upper one; in between neither break is cleanly nu_c or nu_m and the regime is 'MC',
  with both left unnamed -- the same convention fit_gs02_spectrum uses with free_bmid.

  Returns the same keys fit_gs02_spectrum does (num, nuc, b_lo, b_hi, regime, beta_mid,
  nuM, ...) so the two are interchangeable in a tracker, plus the segment fit itself.
  Every field is NaN when the segments are not resolved: this module declines rather than
  extrapolating a line from a knee.
  '''
  x = np.asarray(x, float); sp = np.asarray(sp, float)
  cut = measure_cutoff_nuM(x, sp, psyn, smooth=smooth, flatten=True) if nuM is None else \
        dict(nuM=nuM, sp_flat=sp/np.where(syn_cutoff_R(x/nuM) > 0., syn_cutoff_R(x/nuM), np.nan),
             ok=True)
  seg = fit_segments(x, cut['sp_flat'], psyn, smooth=smooth, slope_tol=slope_tol,
                     min_pts=min_pts, min_dex=min_dex, mid_slope=mid_slope,
                     min_mid_dex=min_mid_dex)
  out = dict(num=np.nan, nuc=np.nan, b_lo=np.nan, b_hi=np.nan, regime=None,
             beta_mid=np.nan, nuM=cut['nuM'], seg=seg, ok=False, at_bound=False,
             mid_dex=seg.get('mid_dex', np.nan), mid_from=seg.get('mid_from'))
  if not seg['ok']:
    return out
  # The one thing that sets this method's accuracy: how many decades of straight mid
  # segment there are to fit. Measured on synthetic GS02 spectra, the recovered lower
  # break is within 6% of truth once mid_dex >~ 2.5 and degrades to a factor ~2 by
  # mid_dex ~ 0.85, for ANY smoothing -- so the plateau width, not s, is the controlling
  # variable. A short plateau carries no unambiguous line, and a free fit to one would be the
  # same kind of silent bias the template fit gives -- so the FREE fit is still refused below
  # min_mid_dex. What used to happen next was a flat decline; now fit_segments falls back to
  # the tangent anchor (mid_from == 'tangent'), which needs no plateau because it holds the
  # slope and takes only the intercept from the data. The decline therefore survives only
  # where the fallback ALSO fails -- there is no crossing, or no interior at all.
  if seg.get('mid_from') != 'tangent' and seg.get('mid_dex', 0.) < min_mid_dex \
     and mid_slope is None:
    return out
  a_lo, a_mid, a_hi = seg['a_lo'], seg['a_mid'], seg['a_hi']
  b_lo = 10**((seg['c_lo'] - seg['c_mid'])/(a_mid - a_lo))
  b_hi = 10**((seg['c_mid'] - seg['c_hi'])/(a_hi - a_mid))
  if not (np.isfinite(b_lo) and np.isfinite(b_hi)) or b_hi <= b_lo:
    return out
  out['b_lo'], out['b_hi'] = float(b_lo), float(b_hi)
  out['beta_mid'] = float(a_mid - 1.)
  bm_fc, bm_sc = -0.5, -(psyn - 1.)/2.
  if abs(out['beta_mid'] - bm_fc) < bmid_tol:
    out['regime'] = 'FC'; out['nuc'], out['num'] = out['b_lo'], out['b_hi']
  elif abs(out['beta_mid'] - bm_sc) < bmid_tol:
    out['regime'] = 'SC'; out['num'], out['nuc'] = out['b_lo'], out['b_hi']
  else:
    out['regime'] = 'MC'
  # a break outside the sampled range is an extrapolation of two lines, not a measurement
  out['at_bound'] = bool(b_lo < x.min() or b_hi > x.max())
  out['ok'] = True
  return out


def track_breaks_segments(r, flux_floor=1e-10, **kw):
  '''
  nu_c(t), nu_m(t) of one sweep point, measured by segment intersection on every
  instantaneous spectrum. Same output contract as sweep_gammacm.track_breaks_gs02, so
  fit_break_evolution, plot_break_evolution, build_break_evolution_table and
  cooling_frequency.compare_to_track all take either.

  The regime is decided per spectrum by the FITTED mid slope, so no cross-time
  monotonicity logic is needed to hold it: unlike the two-ordering template fit, there is
  no discrete branch here that could flip. Bins where the mid segment is unresolved are
  simply not measurements and are excluded by `valid` -- the coverage is lower than the
  template's and honestly so.
  '''
  from sweep_gammacm import nu_over_num, EDGE_FAC
  x = nu_over_num(r)
  nuFnu = r['nuFnu']
  barT = np.asarray(r['Tb'], float) - 1.
  env = r['env']
  n = len(barT)
  Fpk = np.nanmax(nuFnu, axis=1)
  nan = lambda: np.full(n, np.nan)
  b_lo, b_hi, nu_Mt, bmid, mdex = nan(), nan(), nan(), nan(), nan()
  reg = np.array([None]*n, dtype=object)
  ok = np.zeros(n, bool)
  bright = (np.isfinite(Fpk) & (Fpk > flux_floor*np.nanmax(Fpk))
            & ((np.isfinite(nuFnu) & (nuFnu > 0.)).sum(axis=1) >= 12))
  for i in np.flatnonzero(bright):
    f = breaks_from_segments(x, nuFnu[i, :], env.psyn, **kw)
    nu_Mt[i], mdex[i] = f['nuM'], f['mid_dex']
    if not f['ok']:
      continue
    b_lo[i], b_hi[i], bmid[i] = f['b_lo'], f['b_hi'], f['beta_mid']
    reg[i] = f['regime']
    ok[i] = not f['at_bound']
  fast = np.array([rg == 'FC' for rg in reg])
  slow = np.array([rg == 'SC' for rg in reg])
  ambig = np.array([rg == 'MC' for rg in reg])
  nu_c = np.where(fast, b_lo, np.where(slow, b_hi, np.nan))
  nu_m = np.where(fast, b_hi, np.where(slow, b_lo, np.nan))
  sep = b_hi/b_lo
  with np.errstate(invalid='ignore'):
    inwin = (np.minimum(nu_c, nu_m) > EDGE_FAC*x.min()) \
            & (np.maximum(nu_c, nu_m) < x.max()/EDGE_FAC)
  valid = ok & inwin & np.isfinite(nu_c) & np.isfinite(nu_m)
  nuM_nom = (env.gma_max/env.gma_m)**2
  return dict(barT=barT, nu_lo=b_lo, nu_hi=b_hi, nu_c=nu_c, nu_m=nu_m, sep=sep,
              ratio=nu_c/nu_m, s_mid=nan(), off=~ok, unres=ambig, valid=valid,
              valid_m=valid, is_vfc=np.zeros(n, bool), ambig=ambig, beta_mid=bmid,
              mid_dex=mdex, sep_unres=np.nan, i_swap=None, nu_M=nuM_nom, nu_Mt=nu_Mt,
              nu_B=1./env.gma_m**2, nu_win=(float(x.min()), float(x.max())), Fpk=Fpk)


def smoothing_from_deficit(x, sp, psyn, br=None, **kw):
  '''
  The smoothing exponents, read off the FLUX DEFICIT at each break rather than fitted.

  granot_sari_syn puts the spectrum exactly 2**(-1/s) below the asymptote crossing at its
  own break, so once the crossings are known from breaks_from_segments,

      s1 = ln2 / ln( F_cross(b_lo) / F_measured(b_lo) )

  and likewise s2 at b_hi against the extrapolated mid line. No fitting is involved.

  ONLY valid where the two breaks are well separated: when they are close, each break sits
  inside the other's turnover and the deficits add, so the deficit at b_lo is no longer
  2**(-1/s1) alone. `clean` reports whether the separation clears that -- fall back to a
  2-parameter least squares of granot_sari_syn with the positions held when it does not.

  Returns dict(s1, s2, clean, sep).
  '''
  br = breaks_from_segments(x, sp, psyn, **kw) if br is None else br
  out = dict(s1=np.nan, s2=np.nan, clean=False, sep=np.nan)
  if not br['ok']:
    return out
  seg = br['seg']
  b_lo, b_hi = br['b_lo'], br['b_hi']
  out['sep'] = b_hi/b_lo
  lxg = np.log10(np.asarray(x, float))
  # measured flux at each break, on the CUTOFF-FLATTENED spectrum the segments were fitted to
  cut = measure_cutoff_nuM(np.asarray(x, float), np.asarray(sp, float), psyn, flatten=True)
  ly = np.log10(cut['sp_flat'])
  ok = np.isfinite(ly)
  def meas(b):
    return float(np.interp(np.log10(b), lxg[ok], ly[ok]))
  cross_lo = seg['c_lo'] + seg['a_lo']*np.log10(b_lo)     # = c_mid + a_mid*log10(b_lo)
  cross_hi = seg['c_mid'] + seg['a_mid']*np.log10(b_hi)
  d1 = cross_lo - meas(b_lo)        # dex below the crossing
  d2 = cross_hi - meas(b_hi)
  with np.errstate(divide='ignore', invalid='ignore'):
    out['s1'] = float(np.log10(2.)/d1) if d1 > 0 else np.nan
    out['s2'] = float(np.log10(2.)/d2) if d2 > 0 else np.nan
  out['clean'] = bool(out['sep'] > 100.)
  return out


def _free_line(lx, ly, m, min_pts=FREE_MIN_PTS, min_dex=FREE_MIN_DEX):
  '''free-slope least-squares line over the samples m, or NaN if the window is too thin.
  Returns (slope, intercept, log10 width, n).'''
  if m.sum() < min_pts:
    return np.nan, np.nan, np.nan, int(m.sum())
  w = float(lx[m].max() - lx[m].min())
  if w < min_dex:
    return np.nan, np.nan, w, int(m.sum())
  a, c = np.polyfit(lx[m], ly[m], 1)
  return float(a), float(c), w, int(m.sum())


def free_slopes(x, sp, psyn, b_lo, b_hi, nuM, dfac=FREE_DFAC, cutfac=CUT_FAC,
    min_pts=FREE_MIN_PTS, min_dex=FREE_MIN_DEX, vfc=False, fit_dec=FIT_DEC):
  '''
  The three power-law segments of one nuFnu spectrum, measured with NO slope imposed --
  the check that the computed spectra actually carry the synchrotron asymptotes
  (nuFnu ~ nu^(4/3), the mid segment, nu^(1-p/2)) that every other measurement here holds.

  The breaks (b_lo, b_hi) and the cutoff nuM come from OUTSIDE -- typically a
  fit_gs02_spectrum / track_breaks_gs02 fit -- and are used only to POSITION the windows,
  a factor dfac away from the nearest break in frequency. No slope value enters the window
  selection, which is what separates this from fit_segments (see FREE_DFAC).

  The cutoff is divided out first, using the supplied nuM rather than a re-measured one, so
  the high segment is a clean power law rather than the rolloff; the window is still capped
  at nuM/cutfac, beyond which dividing by a small syn_cutoff_R amplifies numerical dust.

  vfc=True is the very-fast-cooling spectrum: a SINGLE break, with the -1/2 segment running
  below it and no nu^(4/3) segment in band at all. a_lo is then NaN by construction (there
  is nothing to measure, not a failed measurement) and the mid window is taken below b_lo.

  a_lo carries a convergence gate. It is measured twice, at dfac and at FREE_CONV_FAC*dfac,
  and `lo_converged` says whether the two agree to FREE_CONV_TOL. They do not in the
  high-latitude tail, where the break is smeared so broadly that the nu^(4/3) asymptote
  never arrives inside the band -- see FREE_CONV_TOL. a_lo is still returned there; callers
  wanting a MEASUREMENT rather than a lower bound should mask on lo_converged.

  Returns dict(a_lo, a_mid, a_hi, dex_lo, dex_mid, dex_hi, n_lo, n_mid, n_hi, da_lo, da_hi,
  p_hi, a_lo_far, d_alo, lo_converged), with da_* the residuals against the expected 4/3 and
  1-p/2 and p_hi = 2(1 - a_hi) the electron index the high segment implies.
  '''
  x = np.asarray(x, float); sp = np.asarray(sp, float)
  a_lo_exp, a_hi_exp = 4./3., 1. - psyn/2.
  out = dict(a_lo=np.nan, a_mid=np.nan, a_hi=np.nan, dex_lo=np.nan, dex_mid=np.nan,
             dex_hi=np.nan, n_lo=0, n_mid=0, n_hi=0, da_lo=np.nan, da_hi=np.nan,
             p_hi=np.nan, a_lo_far=np.nan, d_alo=np.nan, lo_converged=False,
             b_hi_held=np.nan)
  if not (np.isfinite(b_lo) and np.isfinite(b_hi) and np.isfinite(nuM)) or nuM <= 0.:
    return out
  g = np.isfinite(sp) & (sp > 0.) & np.isfinite(x) & (x > 0.)
  if g.sum() < 12:
    return out
  xg, spg = x[g], sp[g]
  with np.errstate(divide='ignore', invalid='ignore'):
    R = syn_cutoff_R(xg/nuM)
  flat = spg/np.where(np.isfinite(R) & (R > 0.), R, np.nan)
  ok = np.isfinite(flat) & (flat > 0.)
  if ok.sum() < 12:
    return out
  lx, ly = np.log10(xg[ok]), np.log10(flat[ok])
  o = np.argsort(lx); lx, ly = lx[o], ly[o]
  keep = ly > ly.max() - fit_dec

  def segments_at(d):
    '''the three (slope, intercept, width, n) lines with the windows standing off by d'''
    if vfc:
      m_lo = np.zeros(len(lx), bool)                     # no nu^(4/3) segment in band
      m_mid = keep & (lx < np.log10(b_lo/d))
    else:
      m_lo = keep & (lx < np.log10(b_lo/d))
      m_mid = keep & (lx > np.log10(b_lo*d)) & (lx < np.log10(b_hi/d))
    m_hi = keep & (lx > np.log10(b_hi*d)) & (lx < np.log10(nuM/cutfac))
    return {k: _free_line(lx, ly, m, min_pts, min_dex)
            for k, m in (('lo', m_lo), ('mid', m_mid), ('hi', m_hi))}

  near = segments_at(dfac)
  far = segments_at(FREE_CONV_FAC*dfac)
  # The upper crossing, redone with a_hi HELD at 1-p/2 instead of fitted. Legitimate only
  # because slope_validation has now VERIFIED that asymptote (recovered to 0.2%, in every
  # regime and epoch); holding it turns the high line into a one-parameter fit and removes
  # the slope-intercept covariance that dominates the crossing when the window is short.
  # Measured on synthetics, the upper-crossing error drops from -58/-34/-19% to -17/-7.5/
  # -4.7%, and s1 recovery improves from 3.5% to 1.5% median.
  # a_lo is deliberately NOT held the same way: it is the asymptote that is often NOT
  # reached inside the band (the whole lo_converged story), and holding it there forces a
  # wrong line and displaces b_lo -- measured +18.5% error on s1 at s1 = 0.5.
  m_hi_h = keep & (lx > np.log10(b_hi*dfac)) & (lx < np.log10(nuM/cutfac))
  c_hi_h = (float(np.mean(ly[m_hi_h] - a_hi_exp*lx[m_hi_h]))
            if m_hi_h.sum() >= min_pts
            and (lx[m_hi_h].max() - lx[m_hi_h].min()) >= min_dex else np.nan)
  for k in ('lo', 'mid', 'hi'):
    out[f'a_{k}'], out[f'c_{k}'], out[f'dex_{k}'], out[f'n_{k}'] = near[k]
  out['da_lo'] = out['a_lo'] - a_lo_exp
  out['da_hi'] = out['a_hi'] - a_hi_exp
  out['p_hi'] = 2.*(1. - out['a_hi'])
  # has a_lo actually reached the asymptote inside the band? compare against the far windows
  if not vfc and np.isfinite(out['a_lo']):
    out['a_lo_far'] = far['lo'][0]
    out['d_alo'] = out['a_lo_far'] - out['a_lo']
    out['lo_converged'] = bool(np.isfinite(out['d_alo'])
                               and abs(out['d_alo']) < FREE_CONV_TOL)
  _free_smoothing(out, lx, ly, vfc, near, far)
  # upper crossing against the HELD high line, for fit_smoothing_held to stand on
  a_md, c_md = out['a_mid'], out['c_mid']
  out['b_hi_held'] = np.nan
  if all(np.isfinite(v) for v in (a_md, c_md, c_hi_h)) and a_hi_exp != a_md:
    lb = (c_md - c_hi_h)/(a_hi_exp - a_md)
    if lx.min() < lb < lx.max():
      out['b_hi_held'] = float(10**lb)
  return out


def _free_smoothing(out, lx, ly, vfc, near, far):
  '''
  s1, s2 from the FLUX DEFICIT at the free segments' own crossings, in place, PLUS the
  standoff-stability check that says whether they mean anything.

  The identity is smoothing_from_deficit's -- granot_sari_syn sits exactly 2**(-1/s) below
  the asymptote crossing at its own break, so s = ln2/ln(F_cross/F_measured) -- but built on
  the geometrically-windowed lines of free_slopes instead of fit_segments' threshold-selected
  ones, which lifts coverage from ~1 bin in 5 to most of them.

  READ THE STABILITY FLAG BEFORE THE VALUE. Unlike a_lo, which converges on 4/3 as the
  windows are pushed out, s does NOT settle: measured on the peak spectra it drifts
  monotonically with the standoff (logr=+2 RS: s2 = 1.93, 1.66, 1.49, 1.36 at dfac = 3, 10,
  30, 100; s1 = 1.22, 1.13, 1.09, 1.08), with no plateau anywhere on the ladder. The cause is
  structural: s comes from an INTERCEPT, and the same residual break curvature that leaves a
  ~0.01 bias on a slope is levered over the decades between window and crossing into a
  displaced crossing and hence a displaced deficit. So these are standoff-dependent estimates
  of the sag depth, not measurements of the smoothing, and they should not be quoted against
  GS02's tabulated s or used to retune GS02_S1/GS02_S2. s_stable marks the minority of bins
  where the dfac -> FREE_CONV_FAC*dfac step moves s by less than S_STABLE_TOL in relative
  terms.

  Also gated on separation: the identity is for ONE break in isolation, and when the two
  close each sits inside the other's turnover so the deficits add. s_clean marks bins
  clearing SEP_CLEAN.
  '''
  out.update(s1=np.nan, s2=np.nan, b_lo_free=np.nan, b_hi_free=np.nan, sep_free=np.nan,
             s_clean=False, s1_far=np.nan, s2_far=np.nan, s_stable=False,
             s_drift_max=np.nan)

  def deficit_s(lines):
    '''(s1, b_lo), (s2, b_hi) from one set of three lines'''
    def one(A, B):
      aA, cA = A[0], A[1]
      aB, cB = B[0], B[1]
      if not all(np.isfinite(v) for v in (cA, aA, cB, aB)) or aB == aA:
        return np.nan, np.nan
      lb = (cA - cB)/(aB - aA)                     # log10 of the crossing frequency
      if not (lx.min() < lb < lx.max()):
        return np.nan, np.nan                      # crossing outside the sampled band
      d = (cA + aA*lb) - float(np.interp(lb, lx, ly))   # dex of sag below the crossing
      return (float(np.log10(2.)/d) if d > 0. else np.nan), 10**lb
    lo = (np.nan, np.nan) if vfc else one(lines['lo'], lines['mid'])
    return lo, one(lines['mid'], lines['hi'])

  (s1, b1), (s2, b2) = deficit_s(near)
  (s1f, _), (s2f, _) = deficit_s(far)
  out.update(s1=s1, s2=s2, b_lo_free=b1, b_hi_free=b2, s1_far=s1f, s2_far=s2f)
  if np.isfinite(b1) and np.isfinite(b2):
    out['sep_free'] = b2/b1
    out['s_clean'] = bool(out['sep_free'] > SEP_CLEAN)
  elif vfc and np.isfinite(b2):
    out['s_clean'] = True                          # single break: nothing to blend with
  drifts = [abs(a - b)/abs(a) for a, b in ((s1, s1f), (s2, s2f))
            if np.isfinite(a) and np.isfinite(b) and a != 0.]
  out['s_drift_max'] = float(max(drifts)) if drifts else np.nan
  out['s_stable'] = bool(drifts and max(drifts) < S_STABLE_TOL)


def track_free_slopes(r, tr=None, dfac=FREE_DFAC, flux_floor=1e-10, **kw):
  '''
  free_slopes on EVERY time bin of a sweep point: how the three measured segment slopes
  evolve through the pulse, against the values held everywhere else.

  The breaks positioning the windows come from `tr`, a track_breaks_gs02 output (computed
  here if not supplied). Bins are taken wherever that fit returned finite breaks and a
  finite nu_M -- deliberately looser than the tracker's own `valid`, which drops bins on
  grounds (a break off-window, the FC/SC identification ambiguous) that do not stop a
  SLOPE from being measurable. The tracker's own verdict is carried through as `fit_ok`
  so a caller can tighten it.

  Returns dict of arrays over bar{T}: a_lo, a_mid, a_hi, dex_*, da_lo, da_hi, p_hi,
  a_lo_far, d_alo, lo_converged, plus barT, is_vfc, fit_ok, Fpk. Mask a_lo on lo_converged
  to keep only the bins where the nu^(4/3) asymptote is actually reached in band.
  '''
  from sweep_gammacm import nu_over_num, track_breaks_gs02
  if tr is None:
    tr = track_breaks_gs02(r)
  x = nu_over_num(r); nuFnu = r['nuFnu']; env = r['env']
  barT = np.asarray(r['Tb'], float) - 1.
  n = len(barT)
  keys = ('a_lo', 'a_mid', 'a_hi', 'dex_lo', 'dex_mid', 'dex_hi', 'da_lo', 'da_hi', 'p_hi',
          'a_lo_far', 'd_alo', 's1', 's2', 's1_far', 's2_far', 'sep_free',
          's_drift_max', 's1_fit', 's2_fit', 's_fit_rms', 'b_hi_held',
          'b_lo_free', 'b_hi_free', 'nu_b1', 's_1brk', 'rms_1brk', 'rms_gain_1brk',
          'a_edge')
  out = {k: np.full(n, np.nan) for k in keys}
  conv = np.zeros(n, bool)
  sclean = np.zeros(n, bool)
  sstable = np.zeros(n, bool)
  sfit_ok = np.zeros(n, bool)
  s1brk_ok = np.zeros(n, bool)
  Fpk = np.nanmax(nuFnu, axis=1)
  bright = np.isfinite(Fpk) & (Fpk > flux_floor*np.nanmax(Fpk))
  for i in np.flatnonzero(bright):
    if not (np.isfinite(tr['nu_lo'][i]) and np.isfinite(tr['nu_hi'][i])
            and np.isfinite(tr['nu_Mt'][i])):
      continue
    vfc_i = bool(tr['is_vfc'][i])
    f = free_slopes(x, nuFnu[i, :], env.psyn, tr['nu_lo'][i], tr['nu_hi'][i],
                    tr['nu_Mt'][i], dfac=dfac, vfc=vfc_i, **kw)
    # the converged smoothing: refit the shape with the crossings and slopes held
    # VFC has a single break and free_slopes reports it as the mid/high crossing
    bh = f['b_hi_held'] if np.isfinite(f['b_hi_held']) else f['b_hi_free']
    bl = bh if vfc_i else f['b_lo_free']
    h = fit_smoothing_held(x, nuFnu[i, :], env.psyn, bl, bh,
                           tr['nu_Mt'][i], f['a_mid'] - 1., vfc=vfc_i)
    # the merged-break alternative: one broad 4/3 -> 1-p/2 break, no mid slope at all
    sb1 = fit_single_break(x, nuFnu[i, :], env.psyn, tr['nu_Mt'][i])
    a_edge_i = edge_slope(x, nuFnu[i, :], tr['nu_Mt'][i])
    f = dict(f, s1_fit=h['s1'], s2_fit=h['s2'], s_fit_rms=h['rms'], nu_b1=sb1['nu_b'],
             s_1brk=sb1['s'], rms_1brk=sb1['rms'],
             rms_gain_1brk=(tr['s_mid'][i]/sb1['rms'] if sb1['rms'] > 0 else np.nan),
             a_edge=a_edge_i)
    sfit_ok[i] = h['ok']
    s1brk_ok[i] = sb1['ok']
    for k in keys:
      out[k][i] = f[k]
    conv[i] = f['lo_converged']
    sclean[i] = f['s_clean']
    sstable[i] = f['s_stable']
  out.update(barT=barT, is_vfc=tr['is_vfc'].copy(), fit_ok=~tr['off'], Fpk=Fpk,
             lo_converged=conv, s_clean=sclean, s_stable=sstable, s_fit_ok=sfit_ok, s_1brk_ok=s1brk_ok,
             regime=classify_regime(tr, env.psyn, a_edge=out['a_edge']),
             a_lo_exp=4./3., a_hi_exp=1. - env.psyn/2., psyn=env.psyn)
  return out


EDGE_FAC_LO = 3.       # a fitted lower break closer than this to the band bottom is not
                       # constrained by the data; slightly wider than track_breaks_gs02's
                       # EDGE_FAC = 2, because the tail spectra put the break at 1.5-2.5x the
                       # bottom and that is exactly the population at issue.
EDGE_NDEC = 1.0        # decades at the bottom of the band over which the edge slope is measured
EDGE_VFC_TOL = 0.12    # how far above the fast-cooling 1/2 the edge slope may sit and still
                       # support a VFC claim. A VFC classification asserts that nu_c AND the
                       # nu^(4/3) segment lie BELOW the band, which requires the lowest in-band
                       # slope to be the -1/2 one. Measured on the tail spectra it is ~0.8 in
                       # BOTH methods, so neither supports the claim -- see edge_slope.
EDGE_WIN_DEX = 0.6     # sub-window width for edge_slope_drift, in dex (not samples, matching
                       # every other gate here). 0.6 clears MIN_DEX with margin at 33 pts/dex
EDGE_WIN_STEP = 0.5    # ... and stepping by 0.5 puts four windows in EDGE_DRIFT_NDEC.
EDGE_DRIFT_NDEC = 2.1  # band the DRIFT is measured over -- deliberately wider than EDGE_NDEC,
                       # which is not a free choice: a drift needs at least two windows, and
                       # one decade holds only one 0.6-dex window at this step. 2.1 = the four
                       # windows starting at +0.0, +0.5, +1.0, +1.5 that resolved the logr=-3
                       # decline (slopes 1.13/0.95/0.79/0.68 there). edge_slope keeps its own
                       # EDGE_NDEC = 1.0 -- the two measure different things and must not be
                       # merged: the VFC gate wants the slope AT the bottom, this wants how
                       # far it moves on the way up.


def _edge_band(x, sp, nuM, cutfac=CUT_FAC, fit_dec=FIT_DEC, min_pts=FREE_MIN_PTS):
  '''
  The lowest usable stretch of one spectrum, cutoff-flattened: (lx, ly) sorted in lx, with
  the rolloff divided out by syn_cutoff_R, everything above nuM/cutfac dropped and everything
  more than fit_dec below the peak dropped. Shared by edge_slope and edge_slope_drift so the
  two cannot come to disagree about which samples "the band bottom" means.

  Returns (None, None) when the band is unusable.
  '''
  x = np.asarray(x, float); sp = np.asarray(sp, float)
  if not np.isfinite(nuM) or nuM <= 0.:
    return None, None
  g = np.isfinite(sp) & (sp > 0.) & np.isfinite(x) & (x > 0.)
  if g.sum() < min_pts:
    return None, None
  xg, spg = x[g], sp[g]
  with np.errstate(divide='ignore', invalid='ignore'):
    R = syn_cutoff_R(xg/nuM)
  flat = spg/np.where(np.isfinite(R) & (R > 0.), R, np.nan)
  ok = np.isfinite(flat) & (flat > 0.) & (xg < nuM/cutfac)
  if ok.sum() < min_pts:
    return None, None
  lx, ly = np.log10(xg[ok]), np.log10(flat[ok])
  o = np.argsort(lx); lx, ly = lx[o], ly[o]
  keep = ly > ly.max() - fit_dec
  return lx[keep], ly[keep]


def edge_slope(x, sp, nuM, ndec=EDGE_NDEC, cutfac=CUT_FAC, fit_dec=FIT_DEC,
    min_pts=FREE_MIN_PTS):
  '''
  The nuFnu log-log slope over the lowest `ndec` decades of usable band, on the
  cutoff-flattened spectrum. This is the only thing that can decide the VFC question FROM THE
  DATA: a VFC spectrum runs -1/2 -> -p/2, so its lowest in-band nuFnu slope is 1/2, whereas a
  spectrum whose cooling break is at or just below the band edge is already steepening toward
  4/3 there.

  Needed because fit_gs02_spectrum decides VFC on whether its FITTED lower break falls below
  the fit window -- a statement about the fit, not the data. Where that break is unconstrained
  (it sits at the band bottom in the tail spectra of both methods) the test flips on noise:
  at log10(gc/gm) = -3, bar{T}/bar{T}_f = 5.7 the two methods produce near-identical spectra
  (edge slopes 0.77 and 0.84) yet land on opposite sides, 'data' being called VFC for all 86
  bins of that window while 'data_rarcut' is called FC with its break 1.5-2.5x the band bottom.
  '''
  lx, ly = _edge_band(x, sp, nuM, cutfac=cutfac, fit_dec=fit_dec, min_pts=min_pts)
  if lx is None or len(lx) < min_pts:
    return np.nan
  m = lx <= lx.min() + ndec
  if m.sum() < min_pts:
    return np.nan
  return float(np.polyfit(lx[m], ly[m], 1)[0])


def edge_slope_drift(x, sp, nuM, ndec=EDGE_DRIFT_NDEC, win=EDGE_WIN_DEX, step=EDGE_WIN_STEP,
    cutfac=CUT_FAC, fit_dec=FIT_DEC, min_pts=FREE_MIN_PTS):
  '''
  How much the free low-end slope MOVES across the band bottom -- the one thing edge_slope
  cannot say, and the difference between a segment and a knee.

  edge_slope collapses the lowest `ndec` decades into a single number, so a genuine segment
  at 0.72 and a knee running 0.89 -> 0.60 come back identical. This fits a free line in
  sliding `win`-wide sub-windows stepped by `step` across the same band (same samples, via
  _edge_band) and reports the spread of those slopes. A segment holds its slope; a knee does
  not, and a knee's individual windows are still beautifully straight -- measured on the
  logr=-3 declined run, every 0.6-dex window fits to an rms of 0.003-0.005 dex while the
  slope moves by 0.20-0.45. That is why the rms is reported but the DRIFT is the
  discriminator: straightness over a short window is not evidence of a power law.

  It separates the populations cleanly on the rarcut sweep. Median drift by verdict:
  SC 0.000, VSC 0.000, MC 0.011, VFC 0.039 -- against 0.259 for the 23 no-verdict spectra.
  The VFC comparison is the sharp one, since the declines are VFC-shaped: every ACCEPTED VFC
  has drift <= 0.104 and every DECLINED one >= 0.115, so the two do not overlap and the
  edge_slope gate is not cutting through a continuum.

  Returns dict(drift, slopes, x0, rms, n), drift = max(slopes) - min(slopes), NaN when fewer
  than two windows carry min_pts samples (one window can say nothing about drift).
  '''
  out = dict(drift=np.nan, slopes=np.array([]), x0=np.array([]), rms=np.array([]), n=0)
  lx, ly = _edge_band(x, sp, nuM, cutfac=cutfac, fit_dec=fit_dec, min_pts=min_pts)
  if lx is None or len(lx) < min_pts:
    return out
  w = sliding_slopes(lx, ly, lo=lx.min(), hi=lx.min() + ndec, win=win, step=step,
                     min_pts=min_pts)
  if w['n'] < 2:
    return out
  out.update(drift=float(w['slopes'].max() - w['slopes'].min()), **w)
  return out


def classify_regime(tr, psyn, bmid_tol=BMID_TOL, a_edge=None, edge_tol=EDGE_VFC_TOL):
  '''
  Per-bin cooling-regime label from a break track, as an object array over bar{T}:

    VFC  the fit found a single break with nu_c below the band (tr['is_vfc'])
    FC   fitted mid F_nu slope within bmid_tol of -1/2
    SC   ... of -(p-1)/2
    MC   between the two: the FC/SC crossing itself, where neither break is cleanly
         nu_c or nu_m
    None no usable fit in that bin

  The label comes from the FITTED mid slope rather than the ordering of the breaks, for the
  reason track_breaks_gs02 documents: an ordering flips discontinuously while beta_mid moves
  through the crossing continuously.
  '''
  n = len(tr['barT'])
  bm = tr.get('beta_mid', np.full(n, np.nan))
  reg = np.array([None]*n, dtype=object)
  bm_fc, bm_sc = -0.5, -(psyn - 1.)/2.
  for i in range(n):
    if tr['is_vfc'][i]:
      # a VFC label is only supportable if the lowest in-band slope IS the -1/2 one; where the
      # spectrum is already steepening there, the cooling break sits at or below the band edge
      # and is unconstrained, so the class is declined rather than guessed (see edge_slope)
      if a_edge is not None and np.isfinite(a_edge[i]) and a_edge[i] > 0.5 + edge_tol:
        reg[i] = None
      else:
        reg[i] = 'VFC'
    elif not np.isfinite(bm[i]) or tr['off'][i]:
      reg[i] = None
    elif (np.isfinite(tr['nu_lo'][i])
          and tr['nu_lo'][i] <= EDGE_FAC_LO*tr['nu_win'][0]):
      # the mirror of the VFC gate: a two-break label whose LOWER break sits on the band edge
      # rests on the same unconstrained parameter, so it is declined too. Applying only one of
      # the two gates would bias one method against the other, which is the confound that made
      # the data / data_rarcut comparison compare shape classes rather than physics.
      reg[i] = None
    elif abs(bm[i] - bm_fc) < bmid_tol:
      reg[i] = 'FC'
    elif abs(bm[i] - bm_sc) < bmid_tol:
      reg[i] = 'SC'
    else:
      reg[i] = 'MC'
  return reg


def free_slope_convergence(x, sp, psyn, b_lo, b_hi, nuM, dscan=FREE_DSCAN, **kw):
  '''
  a_lo and a_hi against the window standoff dfac -- the test that separates residual break
  curvature from a genuine deviation of the spectrum from its asymptote. Curvature dies as
  the window is pushed away from the break; a real deviation does not.

  Returns dict(dfac -> free_slopes result).
  '''
  return {d: free_slopes(x, sp, psyn, b_lo, b_hi, nuM, dfac=d, **kw) for d in dscan}


# --- low-energy index of a TIME-INTEGRATED spectrum -----------------------------------
# Everything above measures INSTANTANEOUS spectra, where the low-energy segment is the
# nu^(4/3) asymptote and the machinery may lean on that: fit_segments hardcodes a_lo = 4/3
# and selects its window by |s - 4/3| < SLOPE_TOL, and free_slopes positions its window from
# breaks that fit came from. Neither survives time integration. A fluence spectrum is a
# superposition of instantaneous ones whose breaks slide down through the band as the pulse
# decays, and its low-energy slope is NOT 4/3 -- on the full (uncut) runs it saturates near
# 1 (i.e. F_nu ~ nu^0, Band alpha ~ -1) and never enters fit_segments' tolerance band at all,
# so breaks_from_segments returns all-NaN on every regime of the rarcut sweep. The index has
# to be fitted FREE, with the window placed without reference to any expected value.
FLU_WIN_DEX = 1.0     # width of one slope window, in decades of nu
FLU_WIN_STEP = 0.5    # the scan advances by this many decades (half-window overlap)
FLU_CONV_TOL = 0.03   # |a(window) - a(window one width above)| accepted as "converged".
                      # Looser than FREE_CONV_TOL (0.015), which gates a window pushed
                      # FREE_CONV_FAC=3 further from a break; here the comparison window is
                      # only one width away, so residual curvature is larger by construction.
                      # It separates the populations on the rarcut sweep: <=0.031 where the
                      # asymptote is reached, 0.19-0.21 where it is not.
FLU_BAND_DEC = (2., 3., 4.)   # peak-anchored windows: the window ENDS this many decades in
                              # nu below the nuFnu peak. Representative value 3.


def _peak_anchored_slope(lx, ly, dec, win_dex, min_pts, min_dex):
  '''free slope over the win_dex-wide window ending `dec` decades below the nuFnu peak'''
  hi = lx[int(np.argmax(ly))] - dec
  m = (lx >= hi - win_dex) & (lx <= hi)
  return _free_line(lx, ly, m, min_pts, min_dex)[0]


def fluence_low_slope(x, sp, nu_break=None, break_fac=FREE_DFAC, win_dex=FLU_WIN_DEX,
    step_dex=FLU_WIN_STEP, conv_tol=FLU_CONV_TOL, band_dec=FLU_BAND_DEC,
    smooth=SLOPE_SMOOTH, min_pts=FREE_MIN_PTS, min_dex=FREE_MIN_DEX, fit_dec=None):
  '''
  The low-energy log-log index of one TIME-INTEGRATED (fluence) nuFnu spectrum, with
  nothing imposed. Two anchors, because they answer different questions:

  a_inf -- the ASYMPTOTIC index. Windows of width win_dex are walked upward from the low
    end of the grid in step_dex steps; each is compared with the window one full width
    above, and the LOWEST one whose pair agrees within conv_tol is taken. That is the same
    logic as free_slopes' lo_converged gate, but self-anchored: a fluence spectrum has no
    fitted break to stand off from (see the note above), so the scan finds the plateau
    instead of being placed at one.

    The scan MUST be bounded, on both sides, or it reports a plateau that is not the one
    asked for. Above: unbounded it walks past the nuFnu peak and locks onto the falling
    side (it returned -0.225 at log10(gc/gm) = -3 before this was capped). Below the peak
    but above the lower break: the fast-cooling nu^(1/2) segment is a perfectly flat, wide
    plateau and converges cleanly, so convergence ALONE cannot tell "reached the 4/3
    asymptote" from "sitting on the mid segment". Pass nu_break = min(nu_m, nu_c)/nu_m and
    the scan is confined to windows ending a factor break_fac below it, which is the only
    region where the low-energy asymptote can live. If that region is empty -- as it is
    whenever the break falls below the grid, i.e. deep fast cooling -- the grid-floor
    window is returned with converged=False and in_band=False. That is the honest answer
    there: the asymptote is off-grid (at log10(gc/gm) = -5 it would need nu ~ 1e-12 nu_m),
    and the number returned describes the segment that IS in band.

    nu_break=None leaves the scan bounded only by the peak; use it on spectra whose break
    position is not known independently (e.g. the synthetic controls).

  a_band[d] -- the PEAK-ANCHORED index over the window ending d decades below the nuFnu
    peak. The asymptote can sit 4-6 decades below the peak, far outside any observable band,
    so a_inf is not what a Band-function fit to real data would return; a_band is the
    nearer-in number. Read the two as different segments, not as estimates of one another:
    where the peak is far above nu_m (slow cooling) the peak-anchored window lands on the
    MID segment and a_band says nothing about the low-energy asymptote.

  Both are pure log-log slopes of nuFnu, so they are invariant under any rescaling of sp
  (to fitting precision, ~1e-13); beta = a - 1 is the F_nu index and Band alpha = a - 2.

  fit_dec drops samples more than that many decades below the peak. Default None keeps the
  whole grid on purpose -- the far low-frequency tail IS the signal here, and the FIT_DEC=8
  clip used for instantaneous spectra would cut a_inf away in the slow-cooling regimes.

  Returns dict(a_inf, conv_diff, converged, in_band, nu_lo_inf, n_inf, x_pk, a_band,
  prof_lx, prof_slope), with a_band a dict keyed on the entries of band_dec and prof_* the
  local slope curve (so a caller plotting the profile need not recompute it).
  '''
  lx, ly, slope = segment_slopes(x, sp, smooth)
  out = dict(a_inf=np.nan, conv_diff=np.nan, converged=False, in_band=False,
             nu_lo_inf=np.nan, n_inf=0, x_pk=np.nan,
             a_band={float(d): np.nan for d in band_dec}, prof_lx=lx, prof_slope=slope)
  if len(lx) < 2*min_pts:
    return out
  if fit_dec is not None:
    keep = ly > ly.max() - fit_dec
    lx, ly = lx[keep], ly[keep]
    if len(lx) < 2*min_pts:
      return out
  lx_pk = lx[int(np.argmax(ly))]
  out['x_pk'] = float(10.**lx_pk)

  def win(lo):
    m = (lx >= lo) & (lx <= lo + win_dex)
    return _free_line(lx, ly, m, min_pts, min_dex)

  # Top of the scan. Both windows must stay below the peak. The break standoff applies to
  # the MEASUREMENT window only, matching FREE_DFAC's semantics (a window stands off from
  # the break by break_fac); the comparison window is allowed to run up to the break, since
  # break curvature there can only inflate the drift and fail the test conservatively.
  # Demanding the standoff of both costs 3 decades of clearance and would put the RS+FS
  # total off-band at every regime -- its lower break is the FS one, a factor fac_nu ~ 15
  # below the RS break on the shared axis.
  lo_max = lx_pk - 2.*win_dex
  if nu_break is not None and np.isfinite(nu_break) and nu_break > 0.:
    lo_max = min(lo_max, np.log10(nu_break/break_fac) - win_dex)
  out['in_band'] = bool(lo_max >= lx[0] - 1e-9)

  floor, lo = None, lx[0]
  while lo <= lo_max + 1e-9:
    a, _, _, n = win(lo)
    a_up = win(lo + win_dex)[0]
    if np.isfinite(a) and floor is None:
      floor = (a, n, lo)
    if np.isfinite(a) and np.isfinite(a_up):
      d = abs(a_up - a)
      if not np.isfinite(out['conv_diff']):
        out['conv_diff'] = d          # the drift measured at the grid floor
      if d < conv_tol:
        out.update(a_inf=a, conv_diff=d, converged=True, nu_lo_inf=float(10.**lo), n_inf=n)
        break
    lo += step_dex
  if not out['converged']:
    # nothing converged in band (or the band was empty): report the grid-floor window, which
    # describes the segment actually covered, and let `converged`/`in_band` disown it.
    if floor is None:
      w0 = win(lx[0])
      floor = (w0[0], w0[3], lx[0])
    a, n, lo = floor
    out.update(a_inf=a, nu_lo_inf=float(10.**lo), n_inf=n)

  for d in band_dec:
    out['a_band'][float(d)] = _peak_anchored_slope(lx, ly, float(d), win_dex,
                                                   min_pts, min_dex)
  return out


S_FIT_BOUNDS = (0.15, 10.)   # bounds on a fitted smoothing exponent; a fit reaching one of
                             # them is reported at_bound rather than as a value


def _flatten_cutoff(x, sp, nuM, sigma=None):
  '''
  sp divided by the cut-off shape at nuM: the single-zone syn_cutoff_R, or the SMEARED
  syn_cutoff_R_smeared when a finite sigma (the dex spread of nu_M across the contributing
  cells) is given. sigma None or <= 0 is bit-identical to the unsmeared division, which is
  what keeps every existing caller where it was.
  '''
  x = np.asarray(x, float)
  with np.errstate(divide='ignore', invalid='ignore'):
    R = (syn_cutoff_R(x/nuM) if sigma is None or not np.isfinite(sigma) or sigma <= 0.
         else syn_cutoff_R_smeared(x/nuM, float(sigma)))
  return np.asarray(sp, float)/np.where(np.isfinite(R) & (R > 0.), R, np.nan)


# Bounds on a FITTED mid slope, in nuFnu index. Deliberately NOT the asymptote interval
# [(3-p)/2, 1/2] that fit_gs02_spectrum's free_bmid uses: the shell-integrated mid segment is
# measured OUTSIDE that interval on exactly the spectra this is for -- the free line across an
# MC knee returns 0.64 [0.47-0.76] -- so those bounds would pin it at 1/2 and reproduce the
# tangent by construction. The only defensible limits are the ones the shape itself needs, a
# mid slope strictly between the two outer asymptotes, with a margin so the fit cannot
# degenerate into a single break.
BMID_MARGIN = 0.05


def fit_smoothing_held(x, sp, psyn, b_lo, b_hi, nuM, beta_mid, free_bhi=True,
    cutfac=CUT_FAC, fit_dec=FIT_DEC, vfc=False, bounds=S_FIT_BOUNDS, s_hold=None,
    sigma=None, free_bmid=False, bmid_margin=BMID_MARGIN):
  '''
  s1, s2 by fitting granot_sari_syn with the BREAK POSITIONS AND ALL THREE SLOPES HELD, so
  the smoothing is the only shape freedom left. The replacement for the deficit estimator of
  _free_smoothing, which does not converge because it reads s off an INTERCEPT and the ~0.01
  residual slope curvature is levered over decades into a displaced crossing.

  Nothing here goes through an intercept: b_lo comes in already measured (the free-segment
  crossing, which IS the GS02 break parameter), beta_mid is the measured free mid slope, the
  outer slopes are the asymptotes this module has now verified, and the fit varies only
  (s1, s2, scale) -- plus b_hi if free_bhi.

  Validated on synthetic granot_sari_syn, breaks held at their TRUE values: s1, s2 recover
  exactly (0.0% at every separation and smoothing tried). Through the FULL pipeline, with
  b_lo and b_hi taken from free_slopes' measured crossings instead:

      s1   recovers to <=3.5% for true s1 in 0.5-2.0 -- a real measurement, and the point of
           this function. It survives because s1 is insensitive to an error in the held
           break: a 30% displacement of b_lo moves it by under 4%.
      s2   recovers to ~5-17% (median ~6% with free_bhi, ~9% with it held). It is limited by
           how well the UPPER crossing can be located, which in turn is limited by the room
           between b_hi and nuM: the high window is squeezed into log10(nuM/cutfac) -
           log10(b_hi*dfac), and s2 is strongly sensitive to what is left. Treat s2 as
           indicative, not measured. Widening the window past nuM does help the synthetic but
           NOT the data -- dividing by a small syn_cutoff_R drifts the measured a_hi from
           -0.256 to -0.271, so cutfac stays where it is.

  free_bhi refits the upper break rather than holding it at its crossing, which is the better
  of the two on median (above). b_lo is never freed: it is the well-measured one, and freeing
  both is the degeneracy fit_gs02_spectrum(free_s=True) already fails on.

  s_hold=(s1, s2) FREEZES both exponents instead of fitting them, leaving only the scale (and
  b_hi if free_bhi). This is what turns the per-regime medians of slope_validation.regime_table
  from a summary statistic into a testable prescription: the residual it returns, against the
  same spectrum with s free, is exactly the cost of tabulating a fixed pair. In VFC only s2 is
  used. Returned s1, s2 are then the held values, and at_bound is False by construction.

  sigma is the dex spread of nu_M across the contributing cells: given, the SMEARED cut-off
  shape is divided out instead of the single-zone one (see _flatten_cutoff). It must be the
  same shape the breaks were measured against, or the held b_hi and the flattened spectrum
  are describing two different rolloffs.

  free_bmid FITS the mid slope as a parameter of the shape instead of holding it, seeded at
  the `beta_mid` passed in and bounded by BMID_MARGIN inside the two outer asymptotes. This
  is the CONSTRAINED version of the template's free-beta_mid fit: b_lo stays held at the
  measured crossing, so one anchor survives and the fit cannot slide both breaks against the
  smoothing the way fit_gs02_spectrum(free_s=True) does. It exists for the marginal spectra,
  where no mid segment can be measured before the fit and holding an asymptote is an
  assumption rather than a measurement. The fitted value comes back as out['beta_mid'], with
  out['bmid_at_bound'] flagging a fit that ran into the margin -- which means the shape wanted
  a mid slope outside what a three-segment spectrum can carry, and the result is not a
  measurement.

  Returns dict(s1, s2, b_hi_fit, rms, npts, at_bound, ok).
  '''
  x = np.asarray(x, float); sp = np.asarray(sp, float)
  out = dict(s1=np.nan, s2=np.nan, b_hi_fit=np.nan, rms=np.nan, npts=0,
             at_bound=False, ok=False, F_ext=np.nan, y0=np.nan,
             beta_mid=(np.nan if free_bmid else beta_mid), bmid_at_bound=False)
  # VFC carries ONE break and no nu^(4/3) segment, so b_lo is that break and b_hi/beta_mid
  # are not used at all (granot_sari_syn's nuc=None branch joins -1/2 straight to -p/2)
  need = (b_lo, nuM) if vfc else (b_lo, b_hi, nuM, beta_mid)
  if not all(np.isfinite(v) for v in need) or nuM <= 0. or b_lo <= 0.:
    return out
  g = np.isfinite(sp) & (sp > 0.) & np.isfinite(x) & (x > 0.)
  if g.sum() < 12:
    return out
  xg, spg = x[g], sp[g]
  flat = _flatten_cutoff(xg, spg, nuM, sigma)
  ok = np.isfinite(flat) & (flat > 0.) & (xg < nuM/cutfac)
  if ok.sum() < 12:
    return out
  xf, yf = xg[ok], np.log10(flat[ok])
  keep = yf > yf.max() - fit_dec
  xf, yf = xf[keep], yf[keep]
  if len(xf) < 12:
    return out
  y0 = float(yf.max())
  yf = yf - y0                             # the scale is a free parameter; fit the shape
  lb, ub = np.log10(bounds[0]), np.log10(bounds[1])

  fit_bhi = bool(free_bhi) and not vfc     # VFC has no upper break to free

  # The free vector is assembled by NAME, not by fixed index: with s_hold the exponents drop
  # out of it entirely, and hard-coded slots (an earlier r.x[3]) are how that goes wrong.
  fit_bmid = bool(free_bmid) and not vfc      # VFC has no mid segment to free

  names = [] if s_hold is not None else (['s2'] if vfc else ['s1', 's2'])
  names.append('A')
  if fit_bhi:
    names.append('b_hi')
  if fit_bmid:
    names.append('bmid')
  slot = {n: i for i, n in enumerate(names)}

  def unpack(q):
    if s_hold is not None:
      s1, s2 = s_hold
    elif vfc:
      s1, s2 = np.nan, 10**q[slot['s2']]
    else:
      s1, s2 = 10**q[slot['s1']], 10**q[slot['s2']]
    bh = 10**q[slot['b_hi']] if fit_bhi else b_hi
    bm = q[slot['bmid']] if fit_bmid else beta_mid
    return s1, s2, 10**q[slot['A']], bh, bm

  def resid(q):
    s1, s2, A, bh, bm = unpack(q)
    m = granot_sari_syn(xf, b_lo, (None if vfc else bh), psyn, s1=s1, s2=s2, nuM=None,
                        F_ext=A, nuFnu=True, beta_mid=bm)
    return np.log10(np.maximum(m, 1e-300)) - yf

  init = {'s1': (np.log10(1.3), lb, ub), 's2': (np.log10(2.), lb, ub), 'A': (0., -8., 8.)}
  if fit_bhi:
    if not (b_lo < b_hi < nuM):
      return out
    init['b_hi'] = (np.log10(b_hi), np.log10(b_lo), np.log10(nuM))
  if fit_bmid:
    # in nuFnu index: a_hi + margin < a_mid < a_lo - margin, i.e. what a three-segment
    # spectrum can carry. Converted to the F_nu index granot_sari_syn takes.
    bm_lo, bm_hi = (1. - psyn/2.) + bmid_margin - 1., 4./3. - bmid_margin - 1.
    bm0 = float(np.clip(beta_mid if np.isfinite(beta_mid) else -0.5, bm_lo, bm_hi))
    init['bmid'] = (bm0, bm_lo, bm_hi)
  q0 = [init[n][0] for n in names]
  blo = [init[n][1] for n in names]
  bhi = [init[n][2] for n in names]
  try:
    r = least_squares(resid, q0, bounds=(blo, bhi))
  except ValueError:
    return out
  s1, s2, A, bh, bm = unpack(r.x)
  out['F_ext'], out['y0'] = float(A), y0
  out['s1'] = np.nan if vfc else float(s1)
  out['s2'] = float(s2)
  out['b_hi_fit'] = float(bh)
  out['beta_mid'] = float(bm) if np.isfinite(bm) else np.nan
  if fit_bmid:
    e = init['bmid']
    out['bmid_at_bound'] = bool(min(abs(bm - e[1]), abs(bm - e[2])) < 1e-3)
  out['rms'] = float(np.sqrt(np.mean(r.fun**2)))
  out['npts'] = int(len(xf))
  # only exponents actually fitted can pin at a bound: none when s_hold is given, and in VFC
  # s1 is never one of them
  ed = [abs(r.x[slot[n]] - e) for n in ('s1', 's2') if n in slot for e in (lb, ub)]
  out['at_bound'] = bool(ed and min(ed) < 1e-3)
  out['ok'] = not out['at_bound']
  return out


S1BRK_BOUNDS = (0.02, 15.)   # bounds on the single-break smoothing, in the GS02 convention
                             # (larger = sharper), the SAME one s1 and s2 use.


def fit_single_break(x, sp, psyn, nuM, cutfac=CUT_FAC, fit_dec=FIT_DEC,
    s_bounds=S1BRK_BOUNDS, s_hold=None, sigma=None):
  '''
  ONE broad break, nu^(4/3) straight to nu^(1-p/2), both asymptotes HELD at the values
  slope_validation has verified. Free: break position, smoothing, scale -- three parameters,
  and crucially NO mid slope, because in the spectra this is for there is no mid segment to
  measure.

  WHY IT EXISTS. The MC ("marginal") spectra are the ones where the two-break form gives out.
  Their local slope runs 1.33, 1.33, 1.24, 0.83, -0.01, -0.24 -- from the low asymptote
  straight to the high one over ~2 decades, never plateauing anywhere in between. free_slopes
  returns NaN for a_mid there and no crossings at all, which is honest but leaves nothing to
  fit; and below a break separation of ~100 the mid window (b_lo*dfac, b_hi/dfac) is empty by
  construction, so the two-break model is not merely imprecise there but structurally
  unusable.

  MEASURED against the two-break GS02 fit on the MC bins of both shells:

      on-axis MC (rise/crossing)   ONE break wins in 53 of 53 bins, rms 2.3-2.8x better
                                   (0.027 -> 0.010), with s = 1.07-1.18 and a q16-q84 span
                                   of only +-0.02 -- a well-determined parameter
      high-latitude MC             s ~ 4.4-4.7 (a ~4x broader transition) with a wide spread
                                   [2.5-6.3]; the one-break form still wins 189/189 on the FS
                                   but only 125/238 on the RS, where both models are poor
                                   (rms ~0.09). Neither shape describes the HLE crossing.

  So this is the right model for a genuinely merged break, and the honest answer for the
  high-latitude case is still that nothing here fits it well.

  The break is written in the GS02 two-term form, exactly as the very-fast-cooling one is
  (granot_sari_syn with nuc=None), differing only in the lower index: 1/3 here against -1/2
  there. So `s` is in the SAME convention as s1 and s2 everywhere else -- larger is sharper --
  and the three cases are one building block instantiated three ways.
  This replaced an equivalent smooth_bpl_apy parameterisation whose exponent ran the opposite
  way. The two are the same function, to a relative 5e-15, under
      s = 1 / [ sigma (beta_lo - beta_hi) ] = 1 / [ sigma (1/3 + p/2) ],
  which is 1/(1.5833 sigma) at p = 2.5; the old sigma = 1.079 / 4.377 are s = 0.585 / 0.144.
  Converting changes no residual -- it is a reparameterisation, not a refit.

  NB `s` here is a scalar, unlike fit_smoothing_held's s_hold=(s1, s2) pair: this shape has
  one break, so one exponent.

  s_hold FREEZES the smoothing instead of fitting it, leaving (nu_b, A) free -- the same
  prescription test fit_smoothing_held's s_hold provides for the two-break form.
  sigma divides out the SMEARED cut-off shape instead of the single-zone one, as in
  fit_smoothing_held.

  Returns dict(nu_b, s, rms, npts, at_bound, ok).
  '''
  x = np.asarray(x, float); sp = np.asarray(sp, float)
  out = dict(nu_b=np.nan, s=np.nan, rms=np.nan, npts=0, at_bound=False, ok=False,
             A=np.nan, y0=np.nan)
  if not np.isfinite(nuM) or nuM <= 0.:
    return out
  g = np.isfinite(sp) & (sp > 0.) & np.isfinite(x) & (x > 0.)
  if g.sum() < 12:
    return out
  xg, spg = x[g], sp[g]
  flat = _flatten_cutoff(xg, spg, nuM, sigma)
  ok = np.isfinite(flat) & (flat > 0.) & (xg < nuM/cutfac)
  if ok.sum() < 12:
    return out
  xf, yf = xg[ok], np.log10(flat[ok])
  keep = yf > yf.max() - fit_dec
  xf, yf = xf[keep], yf[keep]
  if len(xf) < 12:
    return out
  y0 = float(yf.max())
  yf = yf - y0
  lo, hi = np.log10(xf.min()), np.log10(xf.max())
  ls_lo, ls_hi = np.log10(s_bounds[0]), np.log10(s_bounds[1])

  held = s_hold is not None

  def resid(q):
    s = s_hold if held else 10**q[2]
    m = granot_sari_syn(xf, 10**q[0], None, psyn, s2=s, nuM=None, F_ext=10**q[1],
                        nuFnu=True, beta_lo_single=1./3.)
    return np.log10(np.maximum(m, 1e-300)) - yf

  q0 = [np.log10(xf[int(np.argmax(yf))]), 0.] + ([] if held else [np.log10(1.)])
  blo = [lo, -8.] + ([] if held else [ls_lo])
  bhi = [hi, 8.] + ([] if held else [ls_hi])
  try:
    r = least_squares(resid, q0, bounds=(blo, bhi))
  except ValueError:
    return out
  out['nu_b'] = float(10**r.x[0])
  out['A'], out['y0'] = float(10**r.x[1]), y0
  out['s'] = float(s_hold) if held else float(10**r.x[2])
  out['rms'] = float(np.sqrt(np.mean(r.fun**2)))
  out['npts'] = int(len(xf))
  out['at_bound'] = bool(not held
                         and min(abs(r.x[2] - ls_lo), abs(r.x[2] - ls_hi)) < 1e-3)
  out['ok'] = not out['at_bound']
  return out


# ---------------------------------------------------------------------------------------
# THE SELF-CONTAINED ROUTE: identified segments -> crossings -> smoothing
# ---------------------------------------------------------------------------------------
# One chain, rooted in the spectrum alone:
#
#   sweep_gammacm.identify_segments   which power-law segments the spectrum SHOWS, and the
#                                     shape class that follows from the set of them
#   breaks_from_identified            the breaks, as the crossings of exactly those segments
#   smoothing_from_identified         s, by refitting granot_sari_syn with those crossings
#                                     and those slopes HELD -- the only freedom left
#
# WHY IT IS NOT free_slopes + fit_smoothing_held, which fits the same shape from the same
# kind of ingredients. That route places its windows a fixed factor away from breaks taken
# from a GS02 template fit, and labels the regime from the TEMPLATE's fitted beta_mid
# (classify_regime reads tr['beta_mid']). Neither enters a fitted parameter, so the slopes
# and the smoothing it returns are not circular -- but the ROWS of its per-regime table are
# a template verdict, and its coverage is a template's coverage: a window can always be
# placed once a break has been fitted, whether or not the spectrum shows a segment there.
# This route cannot place a window the spectrum does not support. Where no mid segment is
# resolved it says MC and offers the merged shape instead of a two-break fit, so its
# coverage is lower BY CONSTRUCTION and the bins it does report are the ones a segment was
# actually measured in.
#
# WHAT IS HELD, AND WHY THAT IS CONSISTENT HERE. The outer slopes are held at 4/3 and 1-p/2
# in all three steps -- identification, crossing, smoothing -- so the smoothing fit holds
# exactly what the identification asserted, which is the sense in which the route is
# self-consistent. slope_validation is what makes that legitimate rather than circular: it
# measures both asymptotes with no slope imposed anywhere and recovers them (a_hi to 0.2%,
# a_lo to 1.3220 against 4/3 with the residual traced to window curvature).
# The MID slope is NOT held. It is taken at the free value identify_segments already fits
# over its own window (a_fit), because shell integration moves the mid segment off the
# one-zone asymptote by a measured +0.098 (fast) to -0.079 (slow) -- see SEG_FC_TOL_HI. A
# held mid line would fan away from the data across the window and drag both crossings with
# it.


def _recentred_mid(lx, ly, s, interior, a0, slope_tol=SLOPE_TOL, min_pts=MIN_PTS,
    min_dex=MIN_DEX, n_iter=6):
  '''
  The mid line re-measured over a window CENTRED on its own slope, iterated to a fixed point:
  keep the interior samples within slope_tol of the current estimate, refit, repeat.

  WHY THIS IS NOT THE IDENTIFICATION WINDOW. identify_segments selects the mid candidate with
  an ASYMMETRIC window -- widened by SEG_FC_TOL_HI above 1/2 in fast cooling and by
  SEG_SC_TOL_LO below (3-p)/2 in slow -- because the shell-integrated segment really is
  displaced that way and a symmetric window loses genuine FC and SC spectra to MC. That is the
  right window for deciding WHICH segments a spectrum shows. It is the wrong one for measuring
  a slope: an unbalanced slice of a knee tilts the line fitted through it, and on synthetic
  spectra whose mid slope IS the asymptote by construction the identified window returns
  a_mid = 0.536 (fast) and 0.215 (slow) against 0.500 and 0.250. That 0.036 then levers into
  the crossings over several decades and came out as nu_c recovered 21% low in FC and 47% high
  in SC (segment_route.validation_sweep).

  Centring the window on the estimate removes the imprint without imposing a value: the
  iteration converges to wherever the plateau actually sits, so a genuinely displaced segment
  is still measured as displaced -- which is the whole point of reporting dep. This is the
  same fixed-point fit_segments uses, and for the same reason its comment records.

  Returns (slope, intercept, width in dex) or None if no centred window survives.
  '''
  a = float(a0)
  best, prev_w = None, None
  for _ in range(n_iter):
    m = interior & np.isfinite(s) & (np.abs(s - a) < slope_tol)
    w = _widest_run(m, lx, min_pts, min_dex)
    if w is None:
      return best
    i, j = w
    A = np.polyfit(lx[i:j+1], ly[i:j+1], 1)
    best = (float(A[0]), float(A[1]), float(lx[j] - lx[i]))
    if w == prev_w:
      return best
    prev_w, a = w, best[0]
  return best


def _free_mid(lx, ly, interior, min_pts=FREE_MIN_PTS, min_dex=FREE_MIN_DEX):
  '''
  The mid line MEASURED where there is no plateau to hold it: a free straight line over the
  widest contiguous run of the interior, with no slope value entering the selection.

  This is the fallback the GS02-scaffolded route used, transplanted into the self-contained
  one. There, free_slopes placed a mid window geometrically between two fitted breaks and fit
  a free line in it; the window selection knew nothing about the expected slope, and the
  resulting a_mid evolved continuously through the FC/SC crossing rather than jumping between
  asymptotes. Here the same idea uses the identified windows to bound the interior instead of
  a template's breaks, so no fit from outside is needed.

  Returns (slope, intercept, width in dex) or None.
  '''
  w = _widest_run(interior, lx, min_pts, min_dex)
  if w is None:
    return None
  i, j = w
  A = np.polyfit(lx[i:j+1], ly[i:j+1], 1)
  return float(A[0]), float(A[1]), float(lx[j] - lx[i])


def _line_on(lx, ly, x0, x1, a=None, min_pts=MIN_PTS):
  '''
  The straight line of one identified window, re-measured on whatever (lx, ly) is passed in:
  free when a is None, otherwise the least-squares intercept of a line of known slope.
  Returns (slope, intercept), NaN where the window carries too few samples.
  '''
  m = (lx >= np.log10(x0) - 1e-12) & (lx <= np.log10(x1) + 1e-12)
  if m.sum() < min_pts:
    return np.nan, np.nan
  if a is None:
    if m.sum() < 3:
      return np.nan, np.nan
    A = np.polyfit(lx[m], ly[m], 1)
    return float(A[0]), float(A[1])
  return float(a), float(np.mean(ly[m] - a*lx[m]))


def _cross(l1, l2):
  '''log10 of the frequency where two (slope, intercept) lines meet; NaN if parallel'''
  (a1, c1), (a2, c2) = l1, l2
  if not all(np.isfinite(v) for v in (a1, c1, a2, c2)) or a1 == a2:
    return np.nan
  return (c1 - c2)/(a2 - a1)


def breaks_from_identified(x, sp, psyn, det=None, cut=None, smear=True, flatten=True,
    mid='measured', mid_fallback=True, smooth=SLOPE_SMOOTH, slope_tol=SLOPE_TOL,
    min_pts=MIN_PTS, cutfac=CUT_FAC, **kw):
  '''
  The breaks of one nuFnu spectrum as the crossings of the segments identify_segments found,
  with no template anywhere in the chain.

  The shape class decides which crossings exist, and there is no case where a crossing is
  invented for a segment that was not identified:

      FC, SC     lo x mid -> b_lo,  mid x hi -> b_hi          two breaks
      VFC, FC*   mid x hi -> the single break                 one break, no nu^(4/3) in band
      MC         lo x hi  -> the merged break                 one break, no mid segment
                 ... and, with mid_fallback, ALSO the tangent-anchored two-break geometry
      VSC        lo x mid -> b_lo                             upper break above the band
      None       nothing

  THE CUT-OFF IS MEASURED WITH THE SMEARED SHAPE (smear=True, the default here and only
  here). The observed spectrum sums cells carrying a spread of nu_M, so its rolloff is the
  single-electron R(x) convolved with that spread; fitting one zone to it biases nu_M high
  by +23-35% at peak epochs (syn_cutoff_R_smeared). That bias barely propagates into a slope,
  which is why the rest of the project leaves the option off -- but this route divides the
  cut-off out before locating a CROSSING, and the high line's intercept is fitted partly
  inside the rolloff, where dlog b_hi = dc_hi/(a_hi - a_mid) levers it. One measurement is
  made here and then used everywhere -- the window cap inside identify_segments (via `cut`),
  the flattening below, and the smoothing fit downstream (via `sigma`) -- so no two steps
  ever assume different rolloffs. Pass smear=False to reproduce the single-zone shape.

  flatten=True (default) re-measures the three lines on the CUTOFF-FLATTENED spectrum over
  the same windows identify_segments selected on the raw one. Identification and geometry
  are deliberately split that way. identify_segments works on the spectrum as plotted because
  flattening turns the high-latitude tails back UP past nuM and makes their high segment
  unfindable; but its `hi` window is capped only at nuM/CUT_FAC, where the raw spectrum is
  already sagging into the rolloff, and that sag goes straight into c_hi and is levered into
  b_hi by the same denominator. Flattening removes it without touching which windows were
  chosen.

  mid='measured' (default) takes the mid line at the FREE slope fitted over its window;
  'held' takes the one-zone asymptote instead. The default is the physical one -- see the
  section header.

  mid_fallback supplies a mid line wherever the mid slope cannot be properly measured -- i.e.
  no mid window was identified at all, which is what an MC class means. Two are available and
  they differ in what they assume:

    'tangent' (or True)  HOLDS the slope at a theory asymptote. A spectrum whose slope runs
        monotonically from 4/3 to 1-p/2 passes through the mid asymptote exactly once, and at
        that point the tangent IS that asymptote, so the line needs no plateau and no free
        fit: only the intercept comes from the data. Recovers the breaks to 1-3% on
        synthetics down to 3.0 dex of separation. Shape '2brk_tangent'.
        It carries NO evidence about marginality -- the slope is imposed, so beta_mid comes
        back exactly at an asymptote by construction -- and it is discontinuous in time,
        since which asymptote is nearest can switch from bin to bin.
    'free'               FITS the slope over the widest straight run of the interior, with no
        slope value entering the selection (_free_mid). This is what the GS02-scaffolded
        route did, and its virtue is that a_mid then evolves CONTINUOUSLY through the FC/SC
        crossing, which is the physical expectation: the mid slope of a shell-integrated
        spectrum moves smoothly as nu_c passes nu_m, it does not jump between asymptotes.
        The cost is that a line fitted across a knee is window-dependent where the knee has
        no straight part at all. Shape '2brk_free'.
    False                no fallback: MC keeps only the merged single break.

  The merged single break is measured on every MC bin whatever the fallback, so the two
  descriptions can always be compared; `mid_from` records which line the crossings came from
  ('plateau', 'tangent' or 'free'). Never pool 'tangent' bins into a regime census.

  Returns dict(regime, b_lo, b_hi, a_lo, a_mid, a_hi, c_lo, c_mid, c_hi, nuM, sigma,
  mid_name, mid_from, shape, n_breaks, dex_*, ok, det), shape being which model the
  smoothing step should use ('2brk', '2brk_tangent', '1brk_vfc', '1brk_mc', None).
  '''
  from sweep_gammacm import identify_segments
  x = np.asarray(x, float); sp = np.asarray(sp, float)
  nan = np.nan
  out = dict(regime=None, b_lo=nan, b_hi=nan, a_lo=nan, a_mid=nan, a_hi=nan, c_lo=nan,
             c_mid=nan, c_hi=nan, nuM=nan, sigma=nan, mid_name=None, mid_from=None,
             shape=None, n_breaks=0, dex_lo=nan, dex_mid=nan, dex_mid_id=nan, dex_hi=nan,
             ok=False, det=None)
  # ONE cut-off measurement for the whole chain (see the docstring): flatten=False because
  # the flattening below is done against the windows, not the peak-anchored scan's own grid
  if cut is None:
    cut = measure_cutoff_nuM(x, sp, psyn, smooth=smooth, flatten=False, smear=smear)
  if not cut['ok']:
    return out
  sigma = float(cut.get('sigma', np.nan)) if smear else np.nan
  det = identify_segments(x, sp, psyn, smooth=smooth, slope_tol=slope_tol,
                          min_pts=min_pts, cut=cut, cutfac=cutfac,
                          **kw) if det is None else det
  if det is None:
    return out
  out['det'] = det
  out['regime'] = det['regime']
  out['nuM'] = float(det['nuM'])
  out['sigma'] = sigma
  segs, nuM = det['segs'], det['nuM']
  if det['regime'] is None or not segs:
    return out

  g = np.isfinite(sp) & (sp > 0.) & np.isfinite(x) & (x > 0.)
  if g.sum() < min_pts:
    return out
  xg, spg = x[g], sp[g]
  yv = _flatten_cutoff(xg, spg, nuM, sigma) if flatten else spg
  ok = np.isfinite(yv) & (yv > 0.)
  if ok.sum() < min_pts:
    return out
  lx, ly = np.log10(xg[ok]), np.log10(yv[ok])
  o = np.argsort(lx); lx, ly = lx[o], ly[o]

  mid_name = 'fc' if 'fc' in segs else ('sc' if 'sc' in segs else None)
  out['mid_name'] = mid_name
  lines = {}
  for name in ('lo', mid_name, 'hi'):
    if name is None or name not in segs:
      continue
    sg = segs[name]
    # the mid line free (its slope is a measurement), the two asymptotes held (the
    # identification's claim about them is that the spectrum has converged there)
    a = None if (name == mid_name and mid == 'measured') else sg['a']
    lines[name] = _line_on(lx, ly, sg['x0'], sg['x1'], a=a, min_pts=min_pts)
    out[f'dex_{"mid" if name == mid_name else name}'] = float(sg['dex'])
  if mid_name is not None and mid_name in lines:
    out['mid_from'] = 'plateau'

  # the interior: between the two identified asymptote windows, with the samples still
  # hugging either asymptote dropped. Both the re-centred measurement and the fallbacks
  # below work on it.
  a_lo_th, a_hi_th = 4./3., 1. - psyn/2.
  need_interior = ({'lo', 'hi'} <= set(lines)) and (mid_fallback or mid_name is not None)
  lxs = lys = s = interior = None
  if need_interior:
    lxs, lys, s = segment_slopes(10**lx, 10**ly, smooth)
    interior = np.zeros(len(lxs), bool)
    i_lo = np.searchsorted(lxs, np.log10(segs['lo']['x1']), 'right')
    i_hi = np.searchsorted(lxs, np.log10(segs['hi']['x0']), 'left')
    interior[i_lo:i_hi] = True
    interior &= np.isfinite(s) & (lys > lys.max() - FIT_DEC)
    interior &= (s < a_lo_th - slope_tol) & (s > a_hi_th + slope_tol)

  # RE-CENTRE the measurement window on its own slope. identify_segments chose the mid
  # window with an asymmetric tolerance, which is right for deciding the class and wrong for
  # measuring a slope -- see _recentred_mid for the bias it costs and the synthetic that
  # exposed it. The identification stands; only the line is re-measured.
  if mid_name is not None and mid_name in lines and mid == 'measured' \
     and interior is not None and interior.any():
    rc = _recentred_mid(lxs, lys, s, interior, lines[mid_name][0], slope_tol=slope_tol,
                        min_pts=min_pts)
    if rc is not None:
      lines[mid_name] = (rc[0], rc[1])
      out['mid_from'] = 'plateau_recentred'
      out['dex_mid_id'], out['dex_mid'] = out['dex_mid'], rc[2]

  # no mid window at all: supply one instead of declining the geometry
  if mid_fallback and mid_name is None and interior is not None:
    if mid_fallback == 'free':
      fr = _free_mid(lxs, lys, interior)
      if fr is not None:
        lines['mid'] = (fr[0], fr[1])
        mid_name = 'mid'
        out['mid_from'] = 'free'
        out['dex_mid'] = fr[2]
    else:
      tan = _tangent_mid(lxs, lys, s, interior, psyn)
      if tan is not None:
        lines['mid'] = (tan[0], tan[1])
        mid_name = 'mid'
        out['mid_from'] = 'tangent'
  for key, name in (('lo', 'lo'), ('mid', mid_name), ('hi', 'hi')):
    if name in lines:
      out[f'a_{key}'], out[f'c_{key}'] = lines[name]

  rg = det['regime']
  fb = {'tangent': '2brk_tangent', 'free': '2brk_free'}.get(out['mid_from'], '2brk')
  if rg in ('FC', 'SC', 'MC') and mid_name is not None \
     and {'lo', mid_name, 'hi'} <= set(lines):
    lb = _cross(lines['lo'], lines[mid_name])
    hb = _cross(lines[mid_name], lines['hi'])
    if np.isfinite(lb) and np.isfinite(hb) and lb < hb:
      out.update(b_lo=float(10**lb), b_hi=float(10**hb), n_breaks=2, shape=fb)
  elif rg in ('VFC', 'FC*') and mid_name is not None and {mid_name, 'hi'} <= set(lines):
    hb = _cross(lines[mid_name], lines['hi'])
    if np.isfinite(hb):
      out.update(b_hi=float(10**hb), shape='1brk_vfc', n_breaks=1)
  elif rg == 'VSC' and mid_name is not None and {'lo', mid_name} <= set(lines):
    lb = _cross(lines['lo'], lines[mid_name])
    if np.isfinite(lb):
      out.update(b_lo=float(10**lb), n_breaks=1)      # no shape: nothing to fit s on
  if rg == 'MC' and out['shape'] is None and {'lo', 'hi'} <= set(lines):
    # the merged break, where even the tangent found no crossing
    hb = _cross(lines['lo'], lines['hi'])
    if np.isfinite(hb):
      out.update(b_lo=float(10**hb), b_hi=float(10**hb), shape='1brk_mc', n_breaks=1)

  # a crossing outside the sampled band is an extrapolation of two lines, not a measurement
  lo_b, hi_b = float(lx.min()), float(lx.max())
  inband = [lo_b <= np.log10(b) <= hi_b
            for b in (out['b_lo'], out['b_hi']) if np.isfinite(b)]
  out['ok'] = bool(out['n_breaks'] and inband and all(inband))
  return out


def smoothing_from_identified(x, sp, psyn, det=None, br=None, free_bhi=True, s_hold=None,
    s1brk_hold=None, free_bmid='mc', cutfac=CUT_FAC, **kw):
  '''
  s of every break the identified segments define, by refitting granot_sari_syn with those
  crossings and those slopes held -- the third step of the self-contained route.

  The shape follows the class, so each spectrum is fitted with the model it actually shows:

      FC, SC     two-break form, s1 and s2, beta_mid at its MEASURED value
      VFC, FC*   single break, -1/2 to -p/2: s2 only (granot_sari_syn's nuc=None branch)
      MC         single BROAD break, 4/3 to -p/2, no mid slope: fit_single_break -- and,
                 where the tangent fallback anchored a mid line, ALSO the two-break form on
                 that geometry, so the two descriptions can be compared on the same bin
      VSC, None  declined -- the upper break is out of band, so no shape is constrained

  free_bmid='mc' (the DEFAULT) frees the mid slope inside the shape fit for the MARGINAL
  class alone, holding it everywhere else. True or False force it either way.

  WHY MC AND ONLY MC. FC, SC and VFC display a mid segment (or none at all, in VFC), so their
  mid slope is measured before the fit and holding it there is a measurement, not an
  assumption. An MC spectrum displays no mid segment by definition, so whatever is held there
  is a guess -- and both ways of guessing fail: the tangent anchor pins a_mid at an asymptote
  and jumps discontinuously when the nearest one switches, while a free line across the knee
  displaces the lower break by a factor ~2. Freeing a_mid inside the fit, with b_lo still
  anchored at its crossing, does better on every count: measured over 1419 MC bins of both
  shells it gives a_mid = 0.70 [0.60-0.96] evolving CONTINUOUSLY in time (median step 0.003
  between bins, against the tangent's five hard jumps per shell), at rms 0.0086/0.0060 --
  2-3x better than the tangent and 3-4x better than the merged single break, a margin far
  beyond what its two extra parameters can buy.
  THE PRICE, which callers must not paper over: MC's s1 and s2 then come from a
  four-parameter fit and do NOT agree with the held-mid values -- s2 falls to 1.12/1.40
  against 2.28/2.16, and s1 grows a long upper tail [0.67-3.37]. Quote MC's smoothing with
  its spread, never as a tabulated pair. And a_mid ~ 0.70 is the effective slope of a
  TRANSITION, not a cooling index: it sits well above the +0.098 hardening measured where a
  genuine fast-cooling segment is resolved, and does not belong on the same axis as those.

  Bins whose fitted mid slope runs into its bound (~4%) are declined -- s_ok False -- because
  a mid slope outside what a three-segment spectrum can carry is not a measurement.

  Every fit divides out the SAME cut-off breaks_from_identified measured, smeared shape
  included (sigma is carried through): the held break positions and the flattened spectrum
  would otherwise be describing two different rolloffs.

  s_hold=(s1, s2) freezes the two-break exponents and s1brk_hold the merged one, exactly as
  in fit_smoothing_held / fit_single_break: the residual against the free fit is then the
  cost of tabulating that value.

  Returns the breaks_from_identified dict plus s1, s2, s_1brk, rms, rms_1brk, npts,
  at_bound, s_ok.
  '''
  br = breaks_from_identified(x, sp, psyn, det=det, cutfac=cutfac,
                              **kw) if br is None else br
  out = dict(br, s1=np.nan, s2=np.nan, s_1brk=np.nan, rms=np.nan, rms_1brk=np.nan,
             nu_b1=np.nan, npts=0, at_bound=False, s_ok=False, a_mid_fit=np.nan,
             a_mid_seed=br.get('a_mid', np.nan), bmid_at_bound=False, b_hi_fit=np.nan,
             mid_fitted=False)
  if not br['ok']:
    return out
  sig = br.get('sigma', np.nan)
  fb = (br['regime'] == 'MC') if free_bmid == 'mc' else bool(free_bmid)
  if br['shape'] in ('2brk', '2brk_tangent', '2brk_free'):
    f = fit_smoothing_held(x, sp, psyn, br['b_lo'], br['b_hi'], br['nuM'],
                           br['a_mid'] - 1., free_bhi=free_bhi, s_hold=s_hold, sigma=sig,
                           free_bmid=fb, cutfac=cutfac)
    # a fitted mid slope is the better estimate of the mid index, so it becomes a_mid;
    # what was held going in is kept as a_mid_seed. A bin whose fit hit the mid-slope
    # bound is declined outright -- see the docstring.
    out.update(s1=f['s1'], s2=f['s2'], rms=f['rms'], npts=f['npts'],
               at_bound=f['at_bound'], b_hi_fit=f['b_hi_fit'], mid_fitted=fb,
               a_mid_fit=f['beta_mid'] + 1., bmid_at_bound=f['bmid_at_bound'],
               s_ok=bool(f['ok'] and not f['bmid_at_bound']))
    if fb and np.isfinite(f['beta_mid']):
      out['a_mid'] = f['beta_mid'] + 1.
  elif br['shape'] == '1brk_vfc':
    # the single break is b_hi here; fit_smoothing_held's vfc branch takes it as b_lo and
    # uses neither b_hi nor beta_mid (nuc=None joins -1/2 straight to -p/2)
    f = fit_smoothing_held(x, sp, psyn, br['b_hi'], np.nan, br['nuM'], np.nan,
                           vfc=True, s_hold=s_hold, sigma=sig, cutfac=cutfac)
    out.update(s2=f['s2'], rms=f['rms'], npts=f['npts'], at_bound=f['at_bound'],
               s_ok=f['ok'])
  # the merged shape is the reference description of an MC spectrum, so it is measured on
  # EVERY MC bin -- including the ones the tangent fallback just gave a two-break geometry
  if br['regime'] == 'MC':
    f1 = fit_single_break(x, sp, psyn, br['nuM'], s_hold=s1brk_hold, sigma=sig,
                          cutfac=cutfac)
    out.update(s_1brk=f1['s'], rms_1brk=f1['rms'], nu_b1=f1['nu_b'])
    if br['shape'] == '1brk_mc':
      out.update(rms=f1['rms'], npts=f1['npts'], at_bound=f1['at_bound'], s_ok=f1['ok'])
  return out


def track_segment_route(r, flux_floor=1e-10, **kw):
  '''
  The self-contained route on EVERY time bin of one sweep point: what the spectrum shows,
  where its segments cross, and how sharp those crossings are -- with no GS02 fit anywhere,
  so nothing here needs track_breaks_gs02 and the regime column is the SHAPE CLASS rather
  than a template verdict.

  Bins where identify_segments finds no usable set are simply not measurements (regime None,
  s_ok False); the coverage is lower than track_free_slopes' and honestly so.

  Returns dict of arrays over bar{T}: regime, b_lo, b_hi, a_lo, a_mid, a_hi, dex_*, s1, s2,
  s_1brk, rms, rms_1brk, nuM, sigma, a_edge, shape, mid_from, n_breaks, s_ok, br_ok, plus
  barT, Fpk, psyn.
  '''
  from sweep_gammacm import nu_over_num
  x = nu_over_num(r)
  nuFnu = r['nuFnu']
  env = r['env']
  barT = np.asarray(r['Tb'], float) - 1.
  n = len(barT)
  fkeys = ('b_lo', 'b_hi', 'b_hi_fit', 'a_lo', 'a_mid', 'a_mid_seed', 'a_hi',
           'dex_lo', 'dex_mid', 'dex_hi', 's1', 's2', 's_1brk', 'rms', 'rms_1brk',
           'nu_b1', 'nuM', 'sigma', 'a_edge', 'a_drift')
  out = {k: np.full(n, np.nan) for k in fkeys}
  out['regime'] = np.array([None]*n, dtype=object)
  out['shape'] = np.array([None]*n, dtype=object)
  out['mid_from'] = np.array([None]*n, dtype=object)
  out['mid_name'] = np.array([None]*n, dtype=object)
  out['n_breaks'] = np.zeros(n, int)
  s_ok = np.zeros(n, bool); br_ok = np.zeros(n, bool)
  mid_fitted = np.zeros(n, bool); bmid_bound = np.zeros(n, bool)
  Fpk = np.nanmax(nuFnu, axis=1)
  bright = (np.isfinite(Fpk) & (Fpk > flux_floor*np.nanmax(Fpk))
            & ((np.isfinite(nuFnu) & (nuFnu > 0.)).sum(axis=1) >= 12))
  for i in np.flatnonzero(bright):
    f = smoothing_from_identified(x, nuFnu[i, :], env.psyn, **kw)
    for k in fkeys:
      if k in f and f[k] is not None:
        out[k][i] = f[k]
    if f['det'] is not None:
      out['a_edge'][i], out['a_drift'][i] = f['det']['a_edge'], f['det']['a_drift']
    out['regime'][i], out['shape'][i] = f['regime'], f['shape']
    out['mid_name'][i], out['n_breaks'][i] = f['mid_name'], f['n_breaks']
    out['mid_from'][i] = f['mid_from']
    br_ok[i], s_ok[i] = f['ok'], f['s_ok']
    mid_fitted[i], bmid_bound[i] = f['mid_fitted'], f['bmid_at_bound']
  out.update(mid_fitted=mid_fitted, bmid_at_bound=bmid_bound)
  out.update(barT=barT, Fpk=Fpk, s_ok=s_ok, br_ok=br_ok, psyn=env.psyn,
             a_lo_exp=4./3., a_hi_exp=1. - env.psyn/2.)
  return out
