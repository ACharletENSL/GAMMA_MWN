# -*- coding: utf-8 -*-
# @Author: acharlet

'''
How does the PULSE change across the cooling sweep, at the frequencies the
lightcurve figures actually show?

sweep_gammacm.plot_lightcurve_shape draws, for each of NU_TARGETS = 0.01, 0.1,
1 x nu_pk, the peak-normalised lightcurves of the whole log10(gma_c/gma_m) =
-5..+2 sweep on a bar{T}/bar{T}_f axis. This module puts numbers on those
curves: where the peak sits in units of the shell-crossing time bar{T}_f, and
what the pulse looks like around it (rise index, widths, asymmetry, decay
index and how fast it steepens onto the high-latitude asymptote).

Everything is measured on the SAME cached sweep the figures are drawn from
(method_outdir, default the 'data' reference), so the tables and the pngs
cannot drift apart.

THE PEAK TIME IS CHOSEN PER CURVE. Neither estimator works everywhere: the argmax
is exact on a real peak but hops between comb teeth on a flat top (these curves
hold within 0.02% of their maximum over x = 0.8-1.0 in slow cooling, far below the
sample spacing), while the centre of the top is immune to that but biased when the
top is lopsided. So x_pk measures each curve's own comb (_comb_sigma), asks whether
its top is flat on that scale (flatness phi), and smoothsteps between the argmax
and the TOP_FRAC midpoint accordingly -- exact where the peak is well defined,
stable where it is not. w_flat records which branch a row actually came from, and
x_amax/x_flat keep both raw estimates. Every quantity measured about the peak
(asym, t_rise, t_fall, asym10, bar_T_pk) and every figure uses x_pk.

The three metrics that answer 'how does the PULSE run with frequency' -- peak
time x_pk, half-maximum width FWHM and the rise/fall asymmetry asym10 -- are
also measured across the WHOLE frequency grid, not just at NU_TARGETS
(measure_frequency_scan -> lightcurve_freqscan_nu_m.csv, lightcurve_shape_vs_nu.png).
That scan is on the nu/nu_m axis alone: LOGNU_MIN is defined in log10(nu/nu_m),
so every sweep point's grid starts at the same 1e-6 there and the regimes are
directly comparable, which the stored nub = nu/nu_pk axis cannot be (see the
reference switch above). It runs on the grid columns themselves, so no
frequency interpolation is involved.

TWO ASYMMETRIES, do not mix them. asym = (x_1/2 dec - x_pk)/(x_pk - x_1/2 rise)
is measured at HALF maximum and reads fall/rise (>1 = slow decay). asym10 =
t_rise/t_fall with t_rise = x_pk - x_10 rise and t_fall = x_10 dec - x_pk is
measured at a TENTH of the peak and reads rise/fall (<1 = slow decay). They are
not reciprocals of each other -- different levels, different curve.

ON THE CUT SIDE (method='data_rarcut'), the shell is dark past bar{T} ~ 2.03, so
at low frequency the tenth-maximum DECAY crossing can land in the pure
high-latitude tail rather than on emission from live cells. That is the physics
of the cut, not a measurement failure, but it is why t_fall and asym10 of the cut
are not comparable to the reference method's without saying so.

READ THE FREQUENCY AXIS FIRST. nub = nuobs/max(nu_m, nu_c) (_compute_point), so
'nu = 0.01 nu_pk' means 0.01 nu_m in FAST cooling (logr <= 0) and 0.01 nu_c in
SLOW cooling (logr > 0) -- the reference frequency switches at logr = 0. In
nu_m units the logr = +1 and +2 panels are therefore at 100x and 10^4x the
frequency of the logr <= 0 ones, which by itself moves the peak. measure_sweep
takes unit='num' to redo every measurement at fixed nu/nu_m instead, and
plot_peaktime_reference puts the two side by side: at fixed nu/nu_pk the peak
time is NON-monotonic in log10(gma_c/gma_m) (it turns over at logr = 0), at
fixed nu/nu_m it is monotonic and saturates. The turnover is the reference
switch, not the cooling.

Time marks used throughout (cooling_g100, RS z=4; pure hydro, one value for the
whole sweep since alpha rescales lengths and times together):
    bar{T}_f  = 1.3094   shell crossing  = last-shocked cell's onset (x = 1)
    bar{T}_rf = 2.0254   last rarefaction cut-off (x = 1.547): past it the
                         MODELLED cut has the shell dark, while the reference
                         method here keeps cells alive to bar{T} ~ 646-650
                         (x ~ 494), so x_rf is a marker, not a switch-off.

Example use in command line:
  python -c "import lightcurve_shape as L; L.main()"
  python -c "import lightcurve_shape as L; L.main(method='data_rarcut')"
  python -c "import lightcurve_shape as L; L.main(z=1)"
'''

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from sweep_gammacm import (DEFAULT_KEY, DEFAULT_METHOD, Z_SHELL, NU_TARGETS, XLIM_LIN,
    load_sweep, method_outdir, exit_onset_barT, rarefaction_off_barT, run_sweep,
    LOG10RATIO_ARR, nu_over_num, nu_M_over_num, trim_pngs, copy_article_figures,
    local_index as _slope_profile)
                                   # the local temporal index (and its fit window, SLOPE_NMIN
                                   # / SLOPE_HALF0) lives in sweep_gammacm, where the
                                   # lightcurve figures draw it as a panel under every curve;
                                   # measured and plotted must stay the same function

RISE_LEVELS = (1e-2, 1e-1, 0.5)    # flux fractions of the peak at which the LOCAL rise index
                                   # is read. The rise is NOT one power law: it is broken,
                                   # steep (~2) early and shallow (~1) late, so a single
                                   # fitted index hides the shape. Three levels show it, and
                                   # RISE_BREAK below locates the break itself.
                                   # Do not add a 1e-3 level: for the deep fast-cooling points
                                   # the curve is already at 1e-3 of its peak by x ~ 4e-4, five
                                   # grid points from TB_MIN, so that column reads the start of
                                   # the observer grid rather than the pulse.
RISE_BREAK = 1.5                   # the rise break is taken where the local rise index first
                                   # falls to this, i.e. halfway between the two segments. What
                                   # moves across the sweep is the FLUX LEVEL of that break
                                   # (F_br): 0.001 of the peak deep in fast cooling, ~0.3 by
                                   # gamma_c = gamma_m, i.e. the shallow segment shrinks from
                                   # the whole visible rise to the last half-decade.
LATE_WIN = (100., 400.)            # bar{T}/bar{T}_f window for the asymptotic decay index.
                                   # Starts where MOST curves have settled onto -(2+p/2) (the
                                   # local index is within 0.02 of its x = 600 value by x ~ 100
                                   # everywhere except at nu = 0.01 nu_pk in slow cooling, which
                                   # is still steepening -- that is a result, not a fit window
                                   # problem: it shows up as a_late above -3.25 and as x(a=-3)
                                   # of 170-210). Stops well short of x ~ 494, where the cells
                                   # run out of snapshots (data_end_barT = 645.9..650.2) and the
                                   # reference method's on-axis emission ends.
TOP_FRAC = 0.99                    # x_pk = the log-midpoint of the span above this fraction of
                                   # the maximum. THE peak time of this module: the highest
                                   # SAMPLE is not a usable peak here, because near the top
                                   # these curves are flat to well under the sample spacing --
                                   # in slow cooling they hold within 0.02% of their maximum
                                   # over x = 0.8-1.0, so argmax hops between neighbours and
                                   # steps by up to 21% in ln from one frequency column to the
                                   # next while every level crossing stays smooth. Centring on
                                   # the top removes that (x_amax keeps the raw value, so the
                                   # hop stays visible in the tables).
                                   # 0.99 is where the two failure modes cross, measured over
                                   # the whole rarcut scan (3566 columns, 3077 of them with a
                                   # peak at least as sharp as parabolic):
                                   #   level   median bias vs argmax   worst jitter   spans
                                   #           on a well-defined peak  in ln(x_pk)    <3 samples
                                   #   0.95         2.20%                 0.033          0%
                                   #   0.98         1.13%                 0.039          0%
                                   #   0.99         0.57%                 0.051        5.8%
                                   #   0.995        0.47%                 0.053       13.7%
                                   #   0.997        0.42%                 0.101       42.0%
                                   # Below 0.99 the midpoint averages over the shoulder instead
                                   # of locating the peak (at FLAT_FRAC's 0.9 the span reaches
                                   # ~50% of x_pk in slow cooling, far wider than the peak looks
                                   # on the lightcurve panels). HIGHER IS NOT SAFER: the curve
                                   # is U-shaped, and its right-hand branch is the threshold
                                   # descending into the 500-CELL COMB -- the resolution floor
                                   # of the shell (artifact C of the 2026-08-10 hunt: ~275 cells
                                   # switch on across the top at 0.00134 dex spacing, each
                                   # burning off faster than the spacing, so the sum is a comb).
                                   # Measured here as the leave-one-out residual about a local
                                   # quadratic in bar{T} DISTANCE (index-space second
                                   # differences are the metric trap that hunt documents), the
                                   # comb is 0.03-0.11% rms and up to 0.53% peak-to-peak over
                                   # F > 0.98 F_pk. So the cliff is predicted where the cut
                                   # depth 1-TOP_FRAC meets ~0.3-0.5%, i.e. at 0.995-0.997, and
                                   # that is exactly where the jitter doubles. 0.99 cuts at
                                   # 1.0%, ~2x the worst-case comb: the last level clear of it.
                                   # Refining the OBSERVER grid does not help and makes it
                                   # slightly worse (the TB_LIN union test is in that hunt's
                                   # refuted table); the comb is set by the cell count.
                                   # Consequence: a top can carry two comb teeth of equal height
                                   # (near nu_M in fast cooling, e.g. 0.99986 and 1.00000 either
                                   # side of a dip to 0.99859), and there the peak is ambiguous
                                   # by ~14% whatever the threshold. The worst-case bias
                                   # therefore stays ~11% at every level from 0.98 up; only the
                                   # MEDIAN responds to this constant.
COMB_HALF = 3                      # half-width, in samples, of the leave-one-out local fit that
                                   # measures the comb (_comb_sigma): 6 neighbours per point.
FLAT_K = 3.0                       # the 'flat region' used to JUDGE flatness is where the curve
                                   # stays within FLAT_K*sigma_comb of its maximum, i.e. where
                                   # the samples are not separable from the peak given the comb.
FLAT_LO, FLAT_HI = 0.7, 1.1        # flatness phi = (width of that region)/(width a PARABOLIC top
                                   # of the same FWHM would have at the same depth). phi is the
                                   # blend variable: <= FLAT_LO the peak is sharp and x_pk is the
                                   # argmax, >= FLAT_HI it is flat and x_pk is the TOP_FRAC
                                   # midpoint, between them a smoothstep. Calibrated on the
                                   # rarcut scan, where phi has median 0.51 on well-defined peaks
                                   # and 1.10 on flat ones -- note phi < 1 is the norm, these tops
                                   # are SHARPER than parabolic (the 0.95 span is 0.28 of the
                                   # FWHM against 0.316 for a parabola).
                                   # The blend is smoothed rather than switched because a hard
                                   # switch is strictly worse: same bias, and the branch change
                                   # itself becomes the largest step in x_pk (0.118 vs 0.078).
                                   # Measured over the 3566-column rarcut scan, as
                                   # (bias vs argmax on a well-defined peak: median/p95) and
                                   # (jitter in ln(x_pk) between adjacent columns: p95/max):
                                   #   raw argmax        0      / 0      | 0.0088 / 0.2086
                                   #   TOP_FRAC only     0.0057 / 0.0585 | 0.0143 / 0.0506
                                   #   this blend        0      / 0.0156 | 0.0174 / 0.0776
                                   # i.e. exact on a well-defined peak and 3.7x better at p95,
                                   # for a cost confined to the tail of the smoothness. That
                                   # residual jitter is IRREDUCIBLE for any estimator that equals
                                   # the argmax when sharp and the midpoint when flat: sharpness
                                   # varies with frequency, the two disagree by ~10% on the
                                   # columns where it turns over, and those columns are adjacent.
FLAT_FRAC = 0.9                    # 'flat top' = the bar{T} span over which the curve stays
                                   # above this fraction of its peak, in units of x_pk
                                   # NB the shoulder that the log10(gc/gm) = 0 low-frequency
                                   # curve shows just before its peak (the shocked -> rarefied
                                   # handover, x ~ 1.2) is NOT measured by a rise statistic: the
                                   # local index dips to 0.28 and recovers by only 0.03, which
                                   # is the size of the sub-cell comb wiggle on the deep
                                   # fast-cooling curves, so no threshold separates them. It
                                   # shows up instead in flat90 and in the decay index at
                                   # bar{T}_rf, which that point alone brings to ~0.
NU_COLORS = {1e-2: 'tab:purple', 1e-1: 'tab:orange', 1.: 'tab:green'}
RISE_EDGE_N = 3                    # the tenth-maximum RISE crossing is rejected if it lands
                                   # within this many samples of the start of the observer
                                   # grid: the onset is then faster than the grid resolves and
                                   # t_rise measures TB_MIN, not the pulse (same failure the
                                   # 1e-3 rise level hits above, which is why that level is not
                                   # tabulated). Only t_rise/asym10 are dropped -- x_pk, the
                                   # FWHM and the decay side are unaffected.
NU_SCAN_MAX = 10**-0.5             # top of the frequency scan, in units of each point's own
                                   # nu_M (nu_M_over_num): half a decade BELOW the cutoff.
                                   # The grid runs LOGNU_ABOVE_NUM = 1.5 decades past nu_M so
                                   # every spectrum shows its cutoff, but there is no pulse to
                                   # measure in there: the flux falls exponentially and the
                                   # lightcurve underflows, so the peak and the crossings read
                                   # the floor. Stopping AT nu_M is not enough either -- the
                                   # exponential is already bending the spectrum below it, and
                                   # the columns that approach it are where the tops go bimodal
                                   # (see TOP_FRAC) and the worst peak-vs-argmax disagreements
                                   # sit. Half a decade of clearance keeps the scan on the
                                   # power-law part of the spectrum, where the pulse shape is
                                   # the pulse shape.


def lightcurve_at(r, nu_t, unit='pk'):
  '''
  One sweep point's lightcurve at a single frequency, log-log interpolated in
  frequency between the two bracketing grid columns rather than snapped to the
  nearest one (the grid is 33 pts/decade, so snapping misses the target by up to
  4%; the interpolation moves x_pk by <0.004, checked against the snapped values
  plot_lightcurve_shape draws).
  unit: 'pk' -> nu_t is a fraction of nu_pk = max(nu_m, nu_c), i.e. of the stored
  nub axis, exactly as in the figures; 'num' -> a fraction of nu_m for every point.
  Returns (barT, nuFnu, nu_over_num) with barT = (Tobs - Ts)/T0, or None if the
  requested frequency falls off that point's grid.
  '''
  env = r['env']
  ratio2 = max(1., (env.gma_c/env.gma_m)**2)      # nu_pk/nu_m
  nub_t = nu_t/ratio2 if unit == 'num' else nu_t
  nub = r['nub']
  if not (nub[0] <= nub_t <= nub[-1]):
    return None
  j = int(np.clip(np.searchsorted(nub, nub_t), 1, len(nub) - 1))
  w = (np.log(nub_t) - np.log(nub[j-1]))/(np.log(nub[j]) - np.log(nub[j-1]))
  a, b = r['nuFnu'][:, j-1], r['nuFnu'][:, j]
  ok = (a > 0.) & (b > 0.)
  y = np.where(ok, np.exp((1.-w)*np.log(np.where(ok, a, 1.)) + w*np.log(np.where(ok, b, 1.))),
               (1.-w)*a + w*b)
  return r['Tb'] - 1., y, nub_t*ratio2


def _comb_sigma(x, y, ipk, Fpk, half=COMB_HALF):
  '''
  The comb amplitude of one lightcurve near its top, as a fraction of F_pk: the rms
  leave-one-out residual about a local quadratic fitted in x DISTANCE over F > 0.9 F_pk.
  Each sample is compared with what its own neighbours predict, so the smooth pulse
  shape is absorbed by the local fit and what is left is the point-to-point wiggle.

  This is the 500-cell resolution comb (see TOP_FRAC), and it is what decides whether a
  top is genuinely flat or merely noisy. Fitted in DISTANCE, not in index: scoring
  wiggle as a second difference in index space is dominated by the curve's own curvature
  sampled on an uneven grid, which is the documented metric trap of this artifact hunt.
  Returns NaN if the top holds too few samples to leave one out.
  '''
  m = y > 0.9*Fpk
  if m.sum() < 2*half + 4:
    return np.nan
  xs, ys = x[m], y[m]/Fpk
  r = []
  for i in range(half, len(xs) - half):
    k = [q for q in range(i - half, i + half + 1) if q != i]
    r.append(ys[i] - np.polyval(np.polyfit(xs[k], ys[k], 2), xs[i]))
  return float(np.std(r)) if len(r) >= 3 else np.nan


def _smoothstep(t):
  t = min(max(t, 0.), 1.)
  return t*t*(3. - 2.*t)


def _level_cross(x, y, lvl, ipk, side):
  '''bar{T} at which y crosses lvl, walking outwards from the peak index ipk
  ('r' = rise, before the peak; 'd' = decay, after it); log-log interpolated.'''
  if side == 'r':
    idx = np.where(y[:ipk+1] <= lvl)[0]
    if not idx.size:
      return np.nan
    i = idx[-1]; j = i + 1
  else:
    idx = np.where(y[ipk:] <= lvl)[0]
    if not idx.size:
      return np.nan
    j = ipk + idx[0]; i = j - 1
  if j >= len(y) or y[i] <= 0. or y[j] <= 0. or y[i] == y[j]:
    return np.nan
  f = (np.log(lvl) - np.log(y[i]))/(np.log(y[j]) - np.log(y[i]))
  return float(np.exp(np.log(x[i]) + f*(np.log(x[j]) - np.log(x[i]))))


def _slope_cross(x, s, ipk, a):
  '''First bar{T} past the peak at which the local index falls to a.'''
  m = np.where((np.arange(len(x)) > ipk) & np.isfinite(s))[0]
  if m.size < 2:
    return np.nan
  xd, sd = x[m], s[m]
  k = np.where(sd <= a)[0]
  if not k.size or k[0] == 0:
    return np.nan
  k = k[0]
  f = (a - sd[k-1])/(sd[k] - sd[k-1])
  return float(np.exp(np.log(xd[k-1]) + f*(np.log(xd[k]) - np.log(xd[k-1]))))


def shape_metrics(barT, lc, barT_f, barT_rf=None, slopes=True):
  '''
  Peak time and pulse shape of one lightcurve, all times in x = bar{T}/bar{T}_f.

  peak      x_pk        THE peak time, chosen per curve by how flat its top is: the
                        argmax where the peak is well defined, the centre of the top
                        where it is not, smoothstepped between the two so the estimator
                        never jumps. Everything measured about the peak -- asym, t_rise,
                        t_fall, asym10, barT_pk -- uses this, and so does every figure.
                          x_pk = x_amax^(1-w) * x_flat^w,  w = w_flat
            x_amax      the sharp-peak branch: parabolic refinement in (log x, log F)
                        about the grid maximum. Exact when the top is a real peak, and
                        useless when it is not -- it then hops between comb teeth
                        (see TOP_FRAC), which is why it is not used alone.
            x_flat      the flat-top branch: log-midpoint of the span above TOP_FRAC of
                        the maximum, sqrt(x_top_r*x_top_d). Immune to the hopping,
                        biased on a lopsided top -- which is why it is not used alone
                        either. Falls back to x_amax if either crossing is missing.
            comb_sig    the measured comb amplitude of this curve (_comb_sigma)
            flatness    phi: the span the comb cannot resolve, over the span a parabolic
                        top of the same FWHM would have there. ~0.5 on a well-defined
                        peak, >~1.1 on a flat one (see FLAT_LO/FLAT_HI)
            w_flat      the resulting weight on x_flat: 0 = pure argmax, 1 = pure
                        midpoint. Read it before quoting x_pk on a single column.
            x_top_r/d   the TOP_FRAC crossings themselves
            top_width   (x_top_d - x_top_r)/x_flat: how wide the top is at TOP_FRAC
  widths    x_hr,x_hd   half-maximum crossings, FWHM = x_hd - x_hr
            asym        (x_hd - x_pk)/(x_pk - x_hr): 1 = symmetric about the peak
            x_10r,x_10d tenth-maximum crossings, w10 = x_10d - x_10r
            t_rise      x_pk - x_10r: the 10% -> 100% rise time
            t_fall      x_10d - x_pk: the 100% -> 10% fall time
            asym10      t_rise/t_fall, the pulse asymmetry at a tenth of the peak.
                        NB this is rise/fall, the OTHER way round from asym above,
                        and at a different level: the two are not reciprocals.
            rise_edge   True when the rise crossing sits within RISE_EDGE_N samples
                        of the grid start, i.e. t_rise (and hence asym10) is the
                        observer grid, not the pulse
            x05,x50,x95 FLUENCE quantiles (cumulative trapz of nuFnu over x, whole
                        window): x50 is the median photon arrival time and w90 the
                        duration carrying 90% of the fluence -- these see the long
                        tT^-2 tail that the half-max width cannot
  flat top  flat90      span above FLAT_FRAC of the peak, in units of x_pk
  rise      a_rise[L]   local index at each RISE_LEVELS fraction L of the peak
            x_br,F_br   where the rise index falls to RISE_BREAK, and the flux
                        fraction there: the break between the steep (~2) early
                        segment and the shallow (~1) one below the peak
  decay     a_rf        local index at bar{T}_rf (NaN if barT_rf is None)
            x_m2,x_m3   where the local index reaches -2 and -3
            a_late      index fitted over LATE_WIN (high-latitude asymptote)

  slopes=False skips every local-index quantity (the whole a_rise/x_br/F_br/a_rf/
  x_m2/x_m3/a_late block stays NaN). _slope_profile is one straight-line fit PER
  GRID POINT, ~450 of them per lightcurve, which dominates the cost when the
  measurement is repeated over the whole frequency grid; the peak, the widths and
  the asymmetries need none of it.
  '''
  x = barT/barT_f
  y = np.asarray(lc, float)
  out = {k: np.nan for k in ('x_pk', 'x_amax', 'x_flat', 'x_top_r', 'x_top_d', 'top_width',
                             'comb_sig', 'flatness', 'w_flat',
                             'x_hr', 'x_hd', 'fwhm', 'asym', 'x_10r',
                             'x_10d', 'w10', 't_rise', 't_fall', 'asym10', 'x05', 'x50',
                             'x95', 'w90', 'x_90r', 'x_90d', 'flat90', 'x_br', 'F_br',
                             'a_rf', 'x_m2', 'x_m3', 'a_late')}
  out['a_rise'] = {L: np.nan for L in RISE_LEVELS}
  out['rise_edge'] = True
  if not np.isfinite(y).any() or y.max() <= 0.:
    return out
  ipk = int(np.argmax(y)); Fpk = y[ipk]
  out['x_hr'] = _level_cross(x, y, .5*Fpk, ipk, 'r')
  out['x_hd'] = _level_cross(x, y, .5*Fpk, ipk, 'd')
  out['fwhm'] = out['x_hd'] - out['x_hr']

  # --- the peak, in three steps: the argmax, the flat-top centre, and how much of each
  out['x_amax'] = float(x[ipk])
  if 0 < ipk < len(y) - 1 and y[ipk-1] > 0. and y[ipk+1] > 0.:
    c = np.polyfit(np.log(x[ipk-1:ipk+2]), np.log(y[ipk-1:ipk+2]), 2)
    if c[0] < 0.:
      out['x_amax'] = float(np.exp(-c[1]/(2.*c[0])))
  lo, hi = (_level_cross(x, y, TOP_FRAC*Fpk, ipk, s) for s in ('r', 'd'))
  out['x_top_r'], out['x_top_d'] = lo, hi
  out['x_flat'] = float(np.sqrt(lo*hi)) if (lo > 0. and hi > 0.) else out['x_amax']
  out['top_width'] = (hi - lo)/out['x_flat']

  # is this top actually flat, or just a sharp peak sitting on the comb? Compare the span
  # the comb cannot resolve (within FLAT_K*sigma of the max) with the span a parabolic top
  # of the same FWHM would occupy at that same depth.
  sig = _comb_sigma(x, y, ipk, Fpk)
  out['comb_sig'] = sig
  phi, w = np.nan, 0.
  if np.isfinite(sig) and out['fwhm'] > 0.:
    lvl = min(max(1. - FLAT_K*sig, 0.90), 0.9999)
    a, b = (_level_cross(x, y, lvl*Fpk, ipk, s) for s in ('r', 'd'))
    if a > 0. and b > 0.:
      phi = (b - a)/(out['fwhm']*np.sqrt(2.*(1. - lvl)))
      w = _smoothstep((phi - FLAT_LO)/(FLAT_HI - FLAT_LO))
  out['flatness'], out['w_flat'] = phi, w
  # sharp -> the argmax (exact); flat -> the TOP_FRAC midpoint; smoothstep in between
  out['x_pk'] = float(np.exp((1. - w)*np.log(out['x_amax']) + w*np.log(out['x_flat'])))

  out['asym'] = (out['x_hd'] - out['x_pk'])/(out['x_pk'] - out['x_hr'])
  x1r, x1d = _level_cross(x, y, .1*Fpk, ipk, 'r'), _level_cross(x, y, .1*Fpk, ipk, 'd')
  out['x_10r'], out['x_10d'] = x1r, x1d
  out['w10'] = x1d - x1r
  out['t_rise'], out['t_fall'] = out['x_pk'] - x1r, x1d - out['x_pk']
  out['asym10'] = out['t_rise']/out['t_fall']
  # the rise is resolved only if its 10% crossing has grid points before it
  out['rise_edge'] = bool(ipk == 0 or not np.isfinite(x1r)
                          or x1r <= x[min(RISE_EDGE_N, len(x) - 1)])
  lo90, hi90 = (_level_cross(x, y, FLAT_FRAC*Fpk, ipk, s) for s in ('r', 'd'))
  out['x_90r'], out['x_90d'] = lo90, hi90
  out['flat90'] = (hi90 - lo90)/out['x_pk']

  cum = np.concatenate([[0.], np.cumsum(.5*(y[1:] + y[:-1])*np.diff(x))])
  if cum[-1] > 0.:
    q = np.interp([.05, .5, .95], cum/cum[-1], x)
    out['x05'], out['x50'], out['x95'] = (float(v) for v in q)
    out['w90'] = out['x95'] - out['x05']

  if not slopes:
    return out
  s = _slope_profile(x, y)
  rise = (np.arange(len(x)) < ipk) & np.isfinite(s)
  for L in RISE_LEVELS:                      # local index where the rise passes level L
    k = np.where(rise & (y <= L*Fpk))[0]
    if k.size:
      out['a_rise'][L] = float(s[k[-1]])
  k = np.where(rise & (y > 1e-4*Fpk) & (s <= RISE_BREAK))[0]
  if k.size:
    out['x_br'], out['F_br'] = float(x[k[0]]), float(y[k[0]]/Fpk)
  if barT_rf is not None:
    d = np.where((np.arange(len(x)) > ipk) & np.isfinite(s))[0]
    if d.size:
      out['a_rf'] = float(np.interp(barT_rf/barT_f, x[d], s[d]))
  out['x_m2'], out['x_m3'] = _slope_cross(x, s, ipk, -2.), _slope_cross(x, s, ipk, -3.)
  m = (x >= LATE_WIN[0]) & (x <= LATE_WIN[1]) & (y > 0.)
  if m.sum() >= 4:
    out['a_late'] = float(np.polyfit(np.log10(x[m]), np.log10(y[m]), 1)[0])
  return out


def measure_sweep(results, barT_f, barT_rf=None, nu_targets=NU_TARGETS, unit='pk'):
  '''shape_metrics for every (nu_target, sweep point); returns a flat list of rows.'''
  rows = []
  for nu_t in nu_targets:
    for r in results:
      lc = lightcurve_at(r, nu_t, unit=unit)
      if lc is None:
        print(f"  nu/nu_{'m' if unit == 'num' else 'pk'}={nu_t:g} off-grid for "
              f"logr={r['log10ratio']:+.0f} -- skipped")
        continue
      barT, y, nu_num = lc
      m = shape_metrics(barT, y, barT_f, barT_rf)
      m.update(nu_t=nu_t, unit=unit, logr=r['log10ratio'], nu_over_num=nu_num,
               barT_pk=m['x_pk']*barT_f)
      rows.append(m)
  return rows


def measure_frequency_scan(results, barT_f, barT_rf=None, nu_max_fac=NU_SCAN_MAX):
  '''
  The same measurement over the WHOLE frequency grid instead of NU_TARGETS: for
  every sweep point, every stored frequency column up to nu_max_fac*nu_M. Returns a
  flat list of rows, one per (sweep point, frequency), carrying nu_num = nu/nu_m.

  This runs on the grid columns themselves -- unlike lightcurve_at, which
  interpolates between them to hit a requested frequency, there is nothing to
  interpolate here. The axis is nu/nu_m (nu_over_num) and only that: the stored nub
  axis is nu/nu_pk, whose reference switches from nu_m to nu_c at gma_c = gma_m, so
  a scan on it would compare slow- and fast-cooling points at different physical
  frequencies. Local indices are skipped (slopes=False, see shape_metrics).
  '''
  rows = []
  for r in results:
    x_nu, barT = nu_over_num(r), r['Tb'] - 1.
    keep = np.where(x_nu <= nu_max_fac*nu_M_over_num(r))[0]
    n0 = 0
    for j in keep:
      lc = r['nuFnu'][:, j]
      if not (lc.max() > 0.):        # underflowed column, nothing to measure
        continue
      m = shape_metrics(barT, lc, barT_f, barT_rf, slopes=False)
      m.update(logr=r['log10ratio'], nu_num=float(x_nu[j]), nub=float(r['nub'][j]),
               barT_pk=m['x_pk']*barT_f)
      rows.append(m); n0 += 1
    print(f"  logr={r['log10ratio']:+.0f}: {n0} frequencies, "
          f"nu/nu_m = {x_nu[keep[0]]:.3g}..{x_nu[keep[-1]]:.3g} "
          f"(nu_M/nu_m = {nu_M_over_num(r):.3g})")
  return rows


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
_COLS = [('log10(gc/gm)', 'logr', '{:+.0f}'), ('nu/nu_m', 'nu_over_num', '{:.3g}'),
         ('x_pk', 'x_pk', '{:.3f}'), ('bar_T_pk', 'barT_pk', '{:.3f}'),
         ('x_amax', 'x_amax', '{:.3f}'), ('x_flat', 'x_flat', '{:.3f}'),
         ('w_flat', 'w_flat', '{:.2f}'), ('top_w', 'top_width', '{:.2f}'),
         ('x_1/2 rise', 'x_hr', '{:.3f}'), ('x_1/2 dec', 'x_hd', '{:.3f}'),
         ('FWHM', 'fwhm', '{:.3f}'), ('asym', 'asym', '{:.2f}'), ('W10', 'w10', '{:.2f}'),
         ('t_rise', 't_rise', '{:.3f}'), ('t_fall', 't_fall', '{:.2f}'),
         ('asym10', 'asym10', '{:.3f}'),
         ('x_50', 'x50', '{:.2f}'), ('W90', 'w90', '{:.1f}'),
         ('flat90', 'flat90', '{:.2f}'),
         ('a_rise 1e-2', ('a_rise', 1e-2), '{:+.2f}'),
         ('a_rise 0.1', ('a_rise', 1e-1), '{:+.2f}'),
         ('a_rise 0.5', ('a_rise', 0.5), '{:+.2f}'),
         ('x_br', 'x_br', '{:.3g}'), ('F_br', 'F_br', '{:.3g}'),
         ('a(x_rf)', 'a_rf', '{:+.2f}'), ('x(a=-2)', 'x_m2', '{:.2f}'),
         ('x(a=-3)', 'x_m3', '{:.1f}'), ('a_late', 'a_late', '{:+.3f}')]

_NOTE = ('x = bar_T/bar_T_f (crossing at x=1, last rarefaction cut-off at x_rf=1.547); '
         'a = d ln(nuFnu)/d ln(bar_T).\n'
         'x_pk blends the argmax x_amax (used where the peak is well defined) with the '
         f'centre of the span above {TOP_FRAC:g} of the maximum, x_flat (used where the '
         'top is flat on the scale of the cell comb); w_flat is the weight on x_flat, '
         '0 = pure argmax. Everything measured about the peak uses x_pk.\n'
         'FWHM/W10 are the half- and tenth-maximum widths, asym = (x_1/2 dec - x_pk)/'
         '(x_pk - x_1/2 rise); x_50 and W90 = x_95-x_05 are FLUENCE quantiles, so they '
         'include the tail.\nt_rise/t_fall are the 10%->100% and 100%->10% times and '
         'asym10 = t_rise/t_fall -- that is RISE/FALL at a tenth of the peak, the other '
         'way round from asym, which is fall/rise at half maximum: the two are not '
         'reciprocals.\nflat90 = span above 0.9 of the peak in units of x_pk; '
         f'(x_br, F_br) is where the rise index falls to {RISE_BREAK:g}, i.e. the break '
         'between its steep (~2) and shallow (~1) segments.\n'
         f'a_late is fitted over x = {LATE_WIN[0]:g}-{LATE_WIN[1]:g}; the high-latitude '
         'value is -(2+p/2) = -3.25 at p = 2.5.')


def _cell(m, key, fmt):
  v = m[key[0]][key[1]] if isinstance(key, tuple) else m[key]
  return '--' if not np.isfinite(v) else fmt.format(v)


def build_shape_table(rows, outdir, unit='pk'):
  '''One printed/csv/png table per frequency (all rows in a single csv).'''
  cols = [c[0] for c in _COLS]
  tag = 'nu_m' if unit == 'num' else 'nu_pk'
  csv_path = os.path.join(outdir, f'lightcurve_shape_table_{tag}.csv')
  with open(csv_path, 'w', newline='') as f:
    w = csv.writer(f); w.writerow([f'nu/{tag}'] + cols)
    for m in rows:
      w.writerow([f"{m['nu_t']:g}"] + [_cell(m, k, fmt) for _, k, fmt in _COLS])
  print('\n' + _NOTE)
  for nu_t in sorted({m['nu_t'] for m in rows}):
    sub = [m for m in rows if m['nu_t'] == nu_t]
    body = [[_cell(m, k, fmt) for _, k, fmt in _COLS] for m in sub]
    wd = [max(len(cols[i]), max(len(b[i]) for b in body)) for i in range(len(cols))]
    fmt_row = lambda rw: '  '.join(v.ljust(wd[i]) for i, v in enumerate(rw))
    title = f'nu = {nu_t:g} nu_{{{tag[3:]}}}'
    print(f'\n--- {title} ' + '-'*40)
    print(fmt_row(cols)); print('  '.join('-'*w for w in wd))
    for b in body:
      print(fmt_row(b))
    fig, ax = plt.subplots(figsize=(16, 0.32*len(body) + 1.6)); ax.axis('off')
    tbl = ax.table(cellText=body, colLabels=cols, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(7.5); tbl.scale(1, 1.25)
    ax.set_title(f'Lightcurve shape, {title}\n' + _NOTE, fontsize=7.5)
    fig.savefig(os.path.join(outdir, f'lightcurve_shape_table_{tag}_nu={nu_t:g}.png'),
                dpi=200, bbox_inches='tight')
    plt.close(fig)
  print(f'\nshape tables -> {outdir}/lightcurve_shape_table_{tag}*.csv,.png')
  return csv_path


_SCAN_COLS = [('log10(gc/gm)', 'logr', '{:+.0f}'), ('nu/nu_m', 'nu_num', '{:.6g}'),
              ('nu/nu_pk', 'nub', '{:.6g}'),
              ('x_pk', 'x_pk', '{:.4f}'), ('bar_T_pk', 'barT_pk', '{:.4f}'),
              ('x_amax', 'x_amax', '{:.4f}'), ('x_flat', 'x_flat', '{:.4f}'),
              ('w_flat', 'w_flat', '{:.4f}'), ('flatness', 'flatness', '{:.4f}'),
              ('comb_sig', 'comb_sig', '{:.6f}'), ('top_width', 'top_width', '{:.4f}'),
              ('flat90', 'flat90', '{:.4f}'), ('FWHM', 'fwhm', '{:.4f}'),
              ('x_10 rise', 'x_10r', '{:.4f}'), ('x_10 dec', 'x_10d', '{:.4f}'),
              ('t_rise', 't_rise', '{:.4f}'), ('t_fall', 't_fall', '{:.4f}'),
              ('asym10', 'asym10', '{:.4f}'), ('rise_edge', 'rise_edge', '{:d}')]


def build_scan_table(scan_rows, outdir):
  '''
  The frequency scan as one csv, every (sweep point, frequency) row. No printed or
  png table: this is thousands of rows, and plot_shape_vs_nu is what it is meant to
  be read as.
  '''
  csv_path = os.path.join(outdir, 'lightcurve_freqscan_nu_m.csv')
  with open(csv_path, 'w', newline='') as f:
    w = csv.writer(f); w.writerow([c[0] for c in _SCAN_COLS])
    for m in scan_rows:
      w.writerow([('{:d}'.format(int(m[k])) if k == 'rise_edge' else _cell(m, k, fmt))
                  for _, k, fmt in _SCAN_COLS])
  n_edge = sum(1 for m in scan_rows if m['rise_edge'])
  print(f'frequency scan -> {csv_path}  ({len(scan_rows)} rows, '
        f'{n_edge} with an unresolved rise)')
  return csv_path


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def _by_nu(rows, nu_t, key):
  sub = sorted([m for m in rows if m['nu_t'] == nu_t], key=lambda m: m['logr'])
  x = np.array([m['logr'] for m in sub], float)
  if isinstance(key, tuple):
    y = np.array([m[key[0]][key[1]] for m in sub], float)
  else:
    y = np.array([m[key] for m in sub], float)
  return x, y


_SCAN_UNITS = {'num': ('nu_num', '$\\nu/\\nu_m$', '', '\\nu_m'),
               'pk': ('nub', '$\\nu/\\nu_{\\rm pk}$', '_pk', '\\nu_{\\rm pk}')}


def _by_logr(scan_rows, logr, key, mask_edge=False, xkey='nu_num'):
  '''One regime's frequency scan, sorted in the chosen frequency unit: (x, value).'''
  sub = sorted([m for m in scan_rows if m['logr'] == logr], key=lambda m: m[xkey])
  if mask_edge:
    sub = [m for m in sub if not m['rise_edge']]
  return (np.array([m[xkey] for m in sub], float),
          np.array([m[key] for m in sub], float))


def plot_shape_vs_nu(scan_rows, results, outdir, x_rf=None, unit='num'):
  '''
  The three pulse observables against frequency, one curve per cooling regime:
  peak time x_pk = bar{T}_pk/bar{T}_f, half-maximum width, and the 10% rise/fall
  asymmetry. Log-log throughout -- over the ~6-16 decades between the grid floor and
  nu_M all three span decades themselves, and the peak time in particular drops from
  x_pk > 1 (the pulse still building at shell crossing) at low frequency to x_pk << 1
  (an early flash) above nu_c.

  unit: 'num' -> the axis is nu/nu_m, shared by every point (LOGNU_MIN is defined there),
  so the regimes are directly comparable; 'pk' -> nu/nu_pk = the stored nub axis, matching
  the lightcurve panels, but READ THE MODULE DOCSTRING FIRST: nu_pk switches from nu_m to
  nu_c at gma_c = gma_m, so on that axis the slow-cooling curves sit at a different
  physical frequency from the fast-cooling ones and the two halves are not comparable.
  Both sets are produced. The grey vertical guide is the unit's own reference frequency.
  Rows whose rise is unresolved by the observer grid (rise_edge) are dropped from the
  asymmetry panel rather than drawn.

  Only x_pk is drawn -- x_amax, and the per-regime nu_c/nu_M ticks, are in the csv and
  in the spectral figures respectively. Each panel's y range is clipped to its own data,
  so a horizontal guide is drawn only where it actually falls inside.
  '''
  from matplotlib.lines import Line2D
  from sweep_gammacm import _sweep_colors, _draw_order
  xkey, xlab, tag, ref = _SCAN_UNITS[unit]
  colors, sm = _sweep_colors(results)
  nu_all = np.array([m[xkey] for m in scan_rows], float)
  nu_lo, nu_hi = nu_all.min(), nu_all.max()
  # wspace above the default 0.2: each panel carries its own y label and tick labels, and
  # at the default the right panel's label lands on the middle panel's frame
  fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True,
                          gridspec_kw={'wspace': 0.28})
  seen = [[], [], []]
  for r, c in _draw_order(zip(results, colors)):
    logr = r['log10ratio']
    for i, (key, edge) in enumerate((('x_pk', False), ('fwhm', False), ('asym10', True))):
      xn, v = _by_logr(scan_rows, logr, key, mask_edge=edge, xkey=xkey)
      ok = np.isfinite(v) & (v > 0.)
      if ok.any():
        axs[i].semilogx(xn[ok], v[ok], color=c, lw=1.)
        seen[i].append(v[ok])

  ylim = []
  for s in seen:                         # clip each panel to its own data, 3% margin
    if not s:
      ylim.append(None); continue
    a = np.concatenate(s); lo, hi = float(a.min()), float(a.max())
    pad = .03*(hi - lo) or .01*abs(hi)
    ylim.append((lo - pad, hi + pad))

  inside = lambda i, v: ylim[i] is not None and ylim[i][0] <= v <= ylim[i][1]
  for i, (ttl, ylab) in enumerate((
      ('peak time', '$x_{\\rm pk}=\\bar{T}_{\\rm pk}/\\bar{T}_f$'),
      ('width at half maximum', 'FWHM $/\\bar{T}_f$'),
      ('pulse asymmetry at $0.1\\,F_{\\rm pk}$', '$t_{\\rm rise}/t_{\\rm fall}$'))):
    ax = axs[i]
    ax.axvline(1., color='grey', ls=':', lw=.8)
    if i in (0, 2) and inside(i, 1.):    # x=1 = shell crossing / symmetric pulse
      ax.axhline(1., color='grey', ls=':', lw=.8)
    ax.set_xlim(nu_lo, nu_hi)
    ax.set_xlabel(xlab); ax.set_ylabel(ylab); ax.set_title(ttl)
  if x_rf is not None and inside(0, x_rf):
    axs[0].axhline(x_rf, color='k', ls='--', lw=.8)
    axs[0].legend(handles=[Line2D([], [], color='k', ls='--', lw=.8,
                                  label='$\\bar{T}_{\\rm rf}$')],
                  fontsize=8, loc='upper right')
  for ax, yl in zip(axs, ylim):
    if yl is not None:
      ax.set_ylim(*yl)
  # pad/fraction are fractions of the COMBINED width of the three panels (as in
  # sweep_gammacm.plot_lightcurve_shape): the defaults size the gap for one panel
  fig.colorbar(sm, ax=axs, pad=0.012, fraction=0.035,
               label='log$_{10}(\\gamma_c/\\gamma_m)$')
  fig.suptitle('Pulse shape against frequency: peak time, width and rise/fall asymmetry'
               f'   (in ${ref}$)')
  fn = os.path.join(outdir, f'lightcurve_shape_vs_nu{tag}.png')
  fig.savefig(fn, dpi=200)
  plt.close(fig)
  print(f'shape-vs-frequency figure -> {fn}')


def plot_shape_metrics(rows, outdir, barT_f=None, x_rf=None, unit='pk'):
  '''
  Six panels vs log10(gma_c/gma_m), one line per plotted frequency: peak time,
  widths, asymmetry, rise index, decay index, flat top. The horizontal guides on
  the peak-time panel are the crossing (x=1) and the last rarefaction cut-off.
  '''
  sub = 'm' if unit == 'num' else 'pk'
  nus = sorted({m['nu_t'] for m in rows})
  fig, axs = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
  kw = lambda nu: dict(color=NU_COLORS.get(nu, 'k'), marker='o', ms=4,
                       label=f'$\\nu={nu:g}\\,\\nu_{{\\rm {sub}}}$')

  ax = axs[0, 0]
  for nu in nus:
    ax.plot(*_by_nu(rows, nu, 'x_pk'), **kw(nu))
  ax.axhline(1., color='grey', ls=':', lw=.8)
  if x_rf is not None:
    ax.axhline(x_rf, color='k', ls='--', lw=.8, label='$\\bar{T}_{\\rm rf}$')
  ax.set_ylabel('$x_{\\rm pk}=\\bar{T}_{\\rm pk}/\\bar{T}_f$')
  ax.set_title('peak time'); ax.legend(fontsize=8)

  ax = axs[0, 1]
  for nu in nus:
    ax.plot(*_by_nu(rows, nu, 'fwhm'), **kw(nu))
    ax.plot(*_by_nu(rows, nu, 'w10'), color=NU_COLORS.get(nu, 'k'), ls='--', marker='s', ms=3)
  ax.set_ylabel('width $/\\bar{T}_f$'); ax.set_title('FWHM (solid), $W_{10}$ (dashed)')

  ax = axs[0, 2]
  for nu in nus:
    ax.plot(*_by_nu(rows, nu, 'asym'), **kw(nu))
  ax.axhline(1., color='grey', ls=':', lw=.8)
  ax.set_ylabel('$(x_{1/2,\\rm dec}-x_{\\rm pk})/(x_{\\rm pk}-x_{1/2,\\rm rise})$')
  ax.set_title('peak asymmetry (1 = symmetric)')

  ax = axs[1, 0]
  for nu in nus:
    ax.plot(*_by_nu(rows, nu, ('a_rise', 1e-2)), **kw(nu))
    ax.plot(*_by_nu(rows, nu, ('a_rise', 1e-1)), color=NU_COLORS.get(nu, 'k'), ls='--',
            marker='s', ms=3)
  ax.set_ylabel('$d\\ln(\\nu F_\\nu)/d\\ln\\bar{T}$')
  ax.set_title('rise index at $10^{-2}$ (solid) and $0.1$ (dashed) of the peak')
  ax.set_xlabel('log$_{10}(\\gamma_c/\\gamma_m)$')

  ax = axs[1, 1]
  for nu in nus:
    ax.plot(*_by_nu(rows, nu, 'a_rf'), **kw(nu))
    ax.plot(*_by_nu(rows, nu, 'a_late'), color=NU_COLORS.get(nu, 'k'), ls='--',
            marker='s', ms=3)
  ax.axhline(-3.25, color='grey', ls=':', lw=.8)
  ax.set_ylabel('$d\\ln(\\nu F_\\nu)/d\\ln\\bar{T}$')
  ax.set_title('decay index at $\\bar{T}_{\\rm rf}$ (solid) and late (dashed, $-3.25$)')
  ax.set_xlabel('log$_{10}(\\gamma_c/\\gamma_m)$')

  ax = axs[1, 2]
  for nu in nus:
    ax.plot(*_by_nu(rows, nu, 'flat90'), **kw(nu))
    ax.plot(*_by_nu(rows, nu, 'F_br'), color=NU_COLORS.get(nu, 'k'), ls='--',
            marker='s', ms=3)
  ax.set_yscale('log')
  ax.set_ylabel('flat top  /  $F_{\\rm br}/F_{\\rm pk}$')
  ax.set_title('flat top $\\Delta x(F>0.9F_{\\rm pk})/x_{\\rm pk}$ (solid),\n'
               'flux level of the rise break (dashed)')
  ax.set_xlabel('log$_{10}(\\gamma_c/\\gamma_m)$')

  fig.suptitle('Lightcurve shape across the cooling sweep'
               + ('   (at fixed $\\nu/\\nu_m$)' if unit == 'num' else ''))
  fig.tight_layout()
  fn = os.path.join(outdir, f'lightcurve_shape_metrics{"_num" if unit == "num" else ""}.png')
  fig.savefig(fn, dpi=200)
  plt.close(fig)
  print(f'shape-metric figure -> {fn}')


def plot_peaktime_reference(rows_pk, rows_num, outdir, x_rf=None):
  '''
  The frequency-reference check: peak time vs log10(gma_c/gma_m) measured at fixed
  nu/nu_pk (left, what the sweep figures show) and at fixed nu/nu_m (right). The
  turnover at logr = 0 on the left is nu_pk switching from nu_m to nu_c, not a
  change in the pulse.
  '''
  fig, axs = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
  for ax, rows, lab in ((axs[0], rows_pk, '\\nu_{\\rm pk}'), (axs[1], rows_num, '\\nu_m')):
    for nu in sorted({m['nu_t'] for m in rows}):
      ax.plot(*_by_nu(rows, nu, 'x_pk'), color=NU_COLORS.get(nu, 'k'), marker='o', ms=4,
              label=f'$\\nu={nu:g}\\,{lab}$')
    ax.axhline(1., color='grey', ls=':', lw=.8)
    if x_rf is not None:
      ax.axhline(x_rf, color='k', ls='--', lw=.8)
    ax.axvline(0., color='grey', ls='-', lw=.6, alpha=.5)
    ax.set_xlabel('log$_{10}(\\gamma_c/\\gamma_m)$')
    ax.set_title(f'fixed $\\nu/{lab}$'); ax.legend(fontsize=9)
  axs[0].set_ylabel('$x_{\\rm pk}=\\bar{T}_{\\rm pk}/\\bar{T}_f$')
  fig.suptitle('Peak time: the turnover at $\\gamma_c=\\gamma_m$ is the reference frequency, '
               'not the pulse')
  fig.tight_layout()
  fn = os.path.join(outdir, 'lightcurve_peaktime_reference.png')
  fig.savefig(fn, dpi=200)
  plt.close(fig)
  print(f'peak-time reference figure -> {fn}')


def plot_normalised_pulses(results, rows, barT_f, outdir, x_rf=None):
  '''
  The measured shape, drawn: each frequency's curves normalised to their own peak
  AND shifted to x/x_pk, so what is left is the shape alone. Half-max markers on
  every curve; the sweep's colour scale is kept (jet in log10(gma_c/gma_m)).
  '''
  from sweep_gammacm import _sweep_colors, _draw_order
  colors, sm = _sweep_colors(results)
  nus = sorted({m['nu_t'] for m in rows})
  fig, axs = plt.subplots(1, len(nus), figsize=(4.6*len(nus), 4.2), sharey=True)
  axs = np.atleast_1d(axs)
  for ax, nu in zip(axs, nus):
    for (r, c) in _draw_order(zip(results, colors)):
      lc = lightcurve_at(r, nu)
      if lc is None:
        continue
      barT, y, _ = lc
      m = [q for q in rows if q['nu_t'] == nu and q['logr'] == r['log10ratio']]
      if not m or not np.isfinite(m[0]['x_pk']) or y.max() <= 0.:
        continue
      ax.loglog(barT/barT_f/m[0]['x_pk'], y/y.max(), color=c, lw=1.)
    ax.axvline(1., color='grey', ls=':', lw=.8); ax.axhline(.5, color='grey', ls=':', lw=.8)
    ax.set_xlim(3e-2, 3e2); ax.set_ylim(1e-4, 2.)
    ax.set_xlabel('$\\bar{T}/\\bar{T}_{\\rm pk}$')
    ax.set_title(f'$\\nu={nu:g}\\,\\nu_{{\\rm pk}}$')
  axs[0].set_ylabel('$\\nu F_\\nu/(\\nu F_\\nu)_{\\rm max}$')
  fig.colorbar(sm, ax=axs, label='log$_{10}(\\gamma_c/\\gamma_m)$')
  fig.suptitle('Pulse shape with the peak time divided out')
  fn = os.path.join(outdir, 'lightcurve_shape_collapsed.png')
  fig.savefig(fn, dpi=200)
  plt.close(fig)
  print(f'collapsed-shape figure -> {fn}')


def main(key=DEFAULT_KEY, method=DEFAULT_METHOD, z=Z_SHELL, nu_targets=NU_TARGETS,
    outdir=None, nproc=None, scan=True):
  '''
  Measure and report the lightcurve shape of a cached sweep. Runs the sweep first
  if its cache is missing (run_sweep, same (key, method, z) directory).
  scan=False drops the frequency scan (measure_frequency_scan and its csv/figure),
  leaving only the NU_TARGETS tables and figures.
  '''
  outdir = method_outdir(method, key, z) if outdir is None else outdir
  os.makedirs(outdir, exist_ok=True)
  results = load_sweep(outdir)
  if results is None:
    results = run_sweep(key, LOG10RATIO_ARR, z=z, outdir=outdir, nproc=nproc, method=method)

  barT_f = exit_onset_barT(key, z=z)
  off = rarefaction_off_barT(key, z=z)
  barT_rf = off[1] if off else None
  x_rf = barT_rf/barT_f if barT_rf else None
  print(f'bar_T_f = {barT_f:.4f}'
        + (f',  bar_T_rf = {barT_rf:.4f}  (x_rf = {x_rf:.3f})' if x_rf else ''))

  rows = measure_sweep(results, barT_f, barT_rf, nu_targets, unit='pk')
  build_shape_table(rows, outdir, unit='pk')
  plot_shape_metrics(rows, outdir, barT_f=barT_f, x_rf=x_rf, unit='pk')
  plot_normalised_pulses(results, rows, barT_f, outdir, x_rf=x_rf)

  # same measurements at fixed nu/nu_m: isolates the pulse from the nu_pk switch
  rows_num = measure_sweep(results, barT_f, barT_rf, nu_targets, unit='num')
  build_shape_table(rows_num, outdir, unit='num')
  plot_peaktime_reference(rows, rows_num, outdir, x_rf=x_rf)

  # peak time, width and asymmetry across the WHOLE frequency grid, on the nu/nu_m axis
  scan_rows = []
  if scan:
    print('\nfrequency scan:')
    scan_rows = measure_frequency_scan(results, barT_f, barT_rf)
    build_scan_table(scan_rows, outdir)
    for u in _SCAN_UNITS:                  # same measurements, both frequency references
      plot_shape_vs_nu(scan_rows, results, outdir, x_rf=x_rf, unit=u)
  trim_pngs(outdir)
  # the two vs-nu figures are in ARTICLE_SERIES, and this main is what writes them, so the
  # article folder is refreshed here too (no-op for a directory that is not selected)
  copy_article_figures(outdir)
  print(f'\nFigures saved to {outdir}')
  return rows, rows_num, scan_rows
