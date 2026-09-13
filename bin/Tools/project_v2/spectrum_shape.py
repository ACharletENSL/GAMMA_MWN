# -*- coding: utf-8 -*-
# @Author: acharlet

'''
PEAK versus TIME-INTEGRATED spectra, quantified on the same axis.

sweep_gammacm.plot_spectra_pair draws them side by side -- the spectrum at the peak of the
lightcurve, and the fluence spectrum of the whole pulse -- for the entire
log10(gamma_c/gamma_m) sweep. This module puts numbers on that pair, so the two can be
compared at the same $\\mathcal{C}$ rather than by eye:

    SLOPES        the three power-law indices, measured FREE (spectral_breaks.free_slopes),
                  plus the self-anchored low-energy asymptote (fluence_low_slope)
    BREAKS        where the segments cross (the paper route, breaks_from_identified), and
                  the shape class each spectrum displays
    TURNOVERS     where the spectrum actually breaks -- the half-slope point (primary) and
                  the curvature maximum (comparison only; it disagrees by a median 0.207
                  dex and fails outright on MC -- see turnovers_curvature)
    PEAK WIDTH    the ratio of the frequencies at half the peak nuFnu -- no fit anywhere

Each row is one (shell, sweep point, spectrum kind); the ratio table pairs the two kinds of
the SAME sweep point, which is the comparison asked for. Nothing is recomputed: the spectra
come from the cached sweep the figures are drawn from.

READ THIS BEFORE QUOTING A SLOPE. There are two a_lo and two a_hi in the table and they are
not the same quantity.
  a_lo, a_mid, a_hi (slope block)  MEASURED, nothing imposed. free_slopes places a window a
      factor FREE_DFAC in frequency from each break and fits a free line in it; no slope
      value enters the window selection. `lo_conv` says whether the low window and the one
      three times further out agree, i.e. whether the asymptote is reached INSIDE the band.
  a_lo_h, a_hi_h (break block)     HELD at 4/3 and 1-p/2. They are inputs to the crossing
      that defines the break, not measurements of the spectrum. Reporting them as slopes is
      the one easy mistake to make with this table, which is why they carry the _h suffix.
The difference between the two is the interesting column, not an inconsistency:
slope_validation is what licenses the hold, and `da_lo`/`da_hi` are where a fluence spectrum
declines to honour it.

WHY THE BREAKS COME FROM THE PAPER ROUTE AND NOT FROM A FREE FIT. identify_segments accepts
a low window only where the local slope sits within SLOPE_TOL of 4/3, and a fluence spectrum
does not always get there -- that is the documented failure mode of running the route on
time-integrated spectra. The temptation is to replace it here with a free plateau finder
(widest run of sliding windows whose slope holds to a tolerance). It was tried on these
spectra and it does not work, for the reason spectral_breaks' own header records: the low
and high segments are asymptotes the spectrum only APPROACHES, so they never settle. Over
the four decades above the upper break the free slope drifts -0.16 -> -0.26 without ever
holding to 0.02, and a flatness threshold there returns a string of 0.6-dex "plateaus" that
are the drift sampled, not segments. The value test survives exactly because it is
scale-free. So the route is kept, and the class it assigns is REPORTED per spectrum
(`class`) so that a peak/fluence disagreement is visible in the table instead of hidden
inside a slope that was measured under two different assumptions.

WHAT THE WIDTH IS FOR. It is the only quantity here that needs no segment, no class and no
fit: the peak nuFnu and the two frequencies at half of it. It therefore stays comparable
across a class flip, and it is the number to reach for when the break machinery declines a
spectrum. W = nu_+/nu_- is a RATIO OF FREQUENCIES (= of photon energies), reported directly
and as log10; W_lo = nu_pk/nu_- and W_hi = nu_+/nu_pk split it about the peak, and
asym = log(W_hi)/log(W_lo) is how lopsided the SED peak is in the log -- 1 is symmetric.

THE nuFnu PEAK IS NOT THE UPPER BREAK. x_pk is the maximum of the spectrum; b_hi is where
the mid and high asymptotes cross. A smooth break puts the maximum BELOW the crossing, and
by a factor that runs with the regime rather than a constant, so the table carries
`b_hi/x_pk` instead of a single conversion. Neither is wrong -- they are different
definitions, and the width is measured about x_pk because that is what "half the peak flux"
means.

Example use in command line:
  python -c "import spectrum_shape as S; S.main()"
  python -c "import spectrum_shape as S; S.main(shells=(4,))"
'''

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

import spectral_breaks as sb
import sweep_gammacm as swp
                                   # for _draw_spectra_all and the two
                                   # spectrum getters: the spectra panels must
                                   # be the ones plot_spectra_pair already draws
from sweep_gammacm import (DEFAULT_KEY, DEFAULT_METHOD, Z_SHELL, LOG10RATIO_ARR,
    load_sweep, method_outdir, run_sweep, nu_over_num, compute_fluence_spectrum,
    write_table_stamp, trim_pngs, NU_M_LABEL)
from plotting_functions import COL_RS, COL_FS
from lightcurve_shape import _level_cross
                                   # the log-log level crossing, shared with the PULSE
                                   # measurement on purpose: a half-maximum width means the
                                   # same construction in frequency as in time, and the two
                                   # modules must not drift into two versions of it

Z_RS, Z_FS = 4, 1
KINDS = ('peak', 'fluence')        # order is fixed: every table, ratio and figure reads
                                   # fluence AGAINST peak, never the other way round
WIDTH_LEVELS = ((0.5, 'half'), (0.1, 'tenth'))
                                   # fractions of the peak nuFnu at which a width is taken.
                                   # 0.5 is the one asked for; the tenth-maximum width comes
                                   # free from the same crossings and is what shows that the
                                   # broadening is not confined to the top of the peak.
TOP_FRAC = 0.99                    # the "flat top" span, in units of the peak flux -- the
                                   # same fraction lightcurve_shape uses, for the same reason
EDGE_N = 3                         # a level crossing landing within this many samples of the
                                   # grid edge is the WINDOW, not the spectrum


def bolometric_lightcurve(r):
  """
  L(bar{T}) = int F_nu dnu, the frequency-integrated flux at each observer time.

  Evaluated as int nuFnu dln(nu), the same integral -- F_nu dnu = (nu F_nu) dln(nu) --
  in the form the stored grid is logarithmic in. It is the inner half of the double
  integral sweep_shells.shell_shares calls the bolometric fluence, taken per time bin
  instead of on the time-integrated spectrum.

  BOLOMETRIC OVER THE COMPUTED BAND, which is what the word can mean here: the window
  runs from LOGNU_MIN, which sits at the gamma=1 synchrotron floor nu_B where there is no
  emission to miss, to LOGNU_ABOVE_NUM decades past each point's own nu_M, by which the
  spectrum has rolled over. Complete for practical purposes, but a band integral.
  """
  return np.trapezoid(r['nuFnu'], np.log(r['nub']), axis=1)


def _peak_index(r):
  """
  The observer-time row whose spectrum is "the peak spectrum": the argmax of the
  BOLOMETRIC lightcurve.

  CHANGED 2026-09-13, and it is a change of definition rather than a refinement. It was
  detect_rise_peak_tail's row -- the argmax of the lightcurve at the single frequency
  nub = NU_REF = 1, i.e. at nu_0 = max(nu_m, nu_c). That is a monochromatic peak, and
  which frequency it is asked at moves it: on this sweep, shifting the reference a decade
  down moves the selected time by up to +0.76 in bar{T} and the resulting spectrum's own
  peak by a factor 0.4. The bolometric peak has no such knob.

  WHAT IT COST. Little, which is why nothing downstream jumped: the bolometric row sits
  -0.075 to +0.096 in bar{T} from the old one (3-30 grid rows; later at every regime but
  log10(C) = 0), and the spectrum it picks differs by 0.87-1.10 in nu_pk, 0.89-0.99 in
  nu_bk and 0.97-1.06 in the half-maximum width.

  WHAT IT BREAKS, and this is the part to keep in mind: sweep_gammacm's plot_spectra_pair
  still selects on nu_0, so THIS MODULE'S peak spectrum is no longer the one that figure
  draws. plot_spectra_and_ratios builds its own getter off this index rather than
  borrowing swp._peak_getter, so the panels and the numbers here at least agree with each
  other.
  """
  L = bolometric_lightcurve(r)
  return int(np.nanargmax(L)) if np.isfinite(L).any() else None


def spectra_of(r):
  '''{kind: nuFnu spectrum} of one sweep point, on the nu_over_num axis.'''
  ipk = _peak_index(r)
  return {'peak': None if ipk is None else r['nuFnu'][ipk, :],
          'fluence': compute_fluence_spectrum(r['Tb'], r['nuFnu'])}


def peak_and_width(x, sp, levels=WIDTH_LEVELS, top_frac=TOP_FRAC, edge_n=EDGE_N):
  '''
  The SED peak of one nuFnu spectrum and how wide it is, with no fit and no segment.

  x_pk is the maximum, refined by a parabola through the three samples about the grid
  argmax in (log nu, log nuFnu). lightcurve_shape needs a far more careful peak than this
  -- an argmax/flat-top blend weighted by a measured comb amplitude -- because a lightcurve
  sampled on 500 cells has a comb on its top that the argmax hops along. A spectrum has no
  such comb: it is a sum over cells of smooth kernels on a clean log grid, and the two
  estimators agree to better than 2% (worst 1.9%, at log10(C) = 0, where the top is
  flattest) over the whole sweep, both shells and both kinds. The flat-top centre is reported anyway as
  x_flat, with top_dex = the log10 span above top_frac of the maximum, so the reader can
  see how flat the top was rather than take that agreement on trust.

  For each level L: nu_lo/nu_hi are where the spectrum passes L*F_pk either side of the
  peak, W = nu_hi/nu_lo the ratio of photon energies there, W_lo = x_pk/nu_lo and
  W_hi = nu_hi/x_pk the two halves, and asym = log(W_hi)/log(W_lo).

  `edge_lo`/`edge_hi` flag a crossing found within edge_n samples of the grid edge: the
  width is then the frequency window, not the spectrum.
  '''
  x = np.asarray(x, float); sp = np.asarray(sp, float)
  keys = ['F_pk', 'x_pk', 'x_amax', 'x_flat', 'top_dex']
  for _, tag in levels:
    keys += [f'nu_lo_{tag}', f'nu_hi_{tag}', f'W_{tag}', f'logW_{tag}',
             f'Wlo_{tag}', f'Whi_{tag}', f'asym_{tag}']
  out = {k: np.nan for k in keys}
  out.update(edge_lo=False, edge_hi=False)
  g = np.isfinite(sp) & (sp > 0.) & np.isfinite(x) & (x > 0.)
  if g.sum() < 8:
    return out
  xg, yg = x[g], sp[g]
  o = np.argsort(xg); xg, yg = xg[o], yg[o]
  ip = int(np.argmax(yg)); Fpk = float(yg[ip])
  out['F_pk'] = Fpk
  out['x_amax'] = float(xg[ip])
  if 0 < ip < len(yg) - 1:
    c = np.polyfit(np.log10(xg[ip-1:ip+2]), np.log10(yg[ip-1:ip+2]), 2)
    if c[0] < 0.:
      out['x_amax'] = float(10.**(-c[1]/(2.*c[0])))
  out['x_pk'] = out['x_amax']
  t_lo, t_hi = (_level_cross(xg, yg, top_frac*Fpk, ip, s) for s in ('r', 'd'))
  if np.isfinite(t_lo) and np.isfinite(t_hi) and t_lo > 0.:
    out['x_flat'] = float(np.sqrt(t_lo*t_hi))
    out['top_dex'] = float(np.log10(t_hi/t_lo))

  n = len(xg)
  for L, tag in levels:
    lo, hi = (_level_cross(xg, yg, L*Fpk, ip, s) for s in ('r', 'd'))
    out[f'nu_lo_{tag}'], out[f'nu_hi_{tag}'] = lo, hi
    if np.isfinite(lo) and np.isfinite(hi) and lo > 0.:
      out[f'W_{tag}'] = float(hi/lo)
      out[f'logW_{tag}'] = float(np.log10(hi/lo))
      out[f'Wlo_{tag}'] = float(out['x_pk']/lo)
      out[f'Whi_{tag}'] = float(hi/out['x_pk'])
      wl = np.log10(out[f'Wlo_{tag}'])
      out[f'asym_{tag}'] = float(np.log10(out[f'Whi_{tag}'])/wl) if wl > 0. else np.nan
    if L == 0.5:      # the edge test is about the width that was asked for
      out['edge_lo'] = bool(not np.isfinite(lo) or lo <= xg[min(edge_n, n-1)])
      out['edge_hi'] = bool(not np.isfinite(hi) or hi >= xg[max(n-1-edge_n, 0)])
  return out


def segments_and_breaks(x, sp, psyn, **kw):
  '''
  The breaks of one spectrum (the paper route) and its three slopes measured free against
  them.

  breaks_from_identified supplies the shape class, the crossings and the smeared cut-off;
  free_slopes then places its windows a factor FREE_DFAC from those crossings and fits a
  free line in each, which is the project's validated way of asking what the slopes ARE
  rather than assuming them (slope_validation). fluence_low_slope adds a second,
  self-anchored low index: it needs no break at all, walking windows up from the bottom of
  the grid until two agree, so it still answers where free_slopes' window cannot be placed.

  The single-break classes are passed to free_slopes as vfc=True with the one crossing in
  both slots. That is not a fudge: VFC/FC* mean the nu^(4/3) segment is not in band, so
  a_lo comes back NaN BY CONSTRUCTION -- nothing to measure, rather than a failed
  measurement -- and the mid window is taken below the break instead of between two.
  '''
  out = dict(cls=None, shape=None, mid_from=None, n_breaks=0, br_ok=False,
             b_lo=np.nan, b_hi=np.nan, sep=np.nan, nuM=np.nan, sigma=np.nan,
             a_lo_h=np.nan, a_hi_h=np.nan, a_mid_route=np.nan,
             a_lo=np.nan, a_mid=np.nan, a_hi=np.nan,
             dex_lo=np.nan, dex_mid=np.nan, dex_hi=np.nan,
             da_lo=np.nan, da_hi=np.nan, p_hi=np.nan, lo_conv=False,
             a_inf=np.nan, inf_conv=False, inf_band=False,
             b_lo_free=np.nan, b_hi_free=np.nan, s1_def=np.nan, s2_def=np.nan,
             c_lo_f=np.nan, c_mid_f=np.nan, c_hi_f=np.nan,
             sag_lo=np.nan, sag_hi=np.nan, s_clean=False, s_stable=False,
             nu_knee_lo=np.nan, nu_knee_hi=np.nan,
             nu_curv_lo=np.nan, nu_curv_hi=np.nan, dexc_lo=np.nan, dexc_hi=np.nan,
             curv_off_lo=np.nan, curv_off_hi=np.nan,
             nu_knee_1=np.nan, nu_curv_1=np.nan, dexc_1=np.nan, curv_off_1=np.nan,
             knee_off_lo=np.nan, knee_off_hi=np.nan,
             kneef_off_lo=np.nan, kneef_off_hi=np.nan, knee_from=None)
  b = sb.breaks_from_identified(x, sp, psyn, **kw)
  out.update(cls=b['regime'], shape=b['shape'], mid_from=b['mid_from'],
             n_breaks=int(b['n_breaks']), br_ok=bool(b['ok']),
             b_lo=b['b_lo'], b_hi=b['b_hi'], nuM=b['nuM'], sigma=b['sigma'],
             a_lo_h=b['a_lo'], a_hi_h=b['a_hi'], a_mid_route=b['a_mid'])
  if np.isfinite(b['b_lo']) and np.isfinite(b['b_hi']) and b['b_lo'] > 0.:
    out['sep'] = float(b['b_hi']/b['b_lo'])

  vfc = b['shape'] == '1brk_vfc'
  b_lo = b['b_hi'] if vfc else b['b_lo']
  if b['n_breaks'] and np.isfinite(b['b_hi']) and np.isfinite(b_lo):
    f = sb.free_slopes(x, sp, psyn, b_lo, b['b_hi'], b['nuM'], vfc=vfc)
    for k in ('a_lo', 'a_mid', 'a_hi', 'dex_lo', 'dex_mid', 'dex_hi',
              'da_lo', 'da_hi', 'p_hi', 'b_lo_free', 'b_hi_free'):
      out[k] = f[k]
    # the INTERCEPTS of the same free lines, so a figure can draw the asymptotes a knee is
    # being judged against instead of leaving the reader to guess where they run
    for k in ('lo', 'mid', 'hi'):
      out['c_' + k + '_f'] = f.get('c_' + k, np.nan)
    out['lo_conv'] = bool(f['lo_converged'])
    # the sag at those free crossings, which is what free_slopes' s1/s2 are read off
    out['s1_def'], out['s2_def'] = f['s1'], f['s2']
    out['s_clean'], out['s_stable'] = bool(f['s_clean']), bool(f['s_stable'])
    for tag, sv in (('lo', f['s1']), ('hi', f['s2'])):
      out['sag_' + tag] = float(np.log10(2.)/sv) if np.isfinite(sv) and sv > 0. else np.nan

    # ... and where the spectrum actually turns over.
    #
    # A MERGED SPECTRUM GETS ONE KNEE, NOT TWO. Where free_slopes finds no mid plateau the
    # spectrum HAS no mid segment: its index runs 4/3 -> 1-p/2 in a single turn, and that
    # is what MC means. Two knees could still be produced by borrowing the route's tangent
    # mid slope, and were until 2026-09-13, but they are not two measurements of anything:
    # the target levels are built on a held value, and the curvature estimator -- which
    # needs no level and so cannot be steered by one -- returns the SAME frequency for
    # both brackets (curv_hi/curv_lo = 0.97 at RS log10(C) = 0, 0.98 at FS +0, against
    # nominal separations of 18 and 98). A single turn admits a single knee, measured at
    # the half-slope point between the two OUTER asymptotes, and the pair is left NaN.
    a_lo_k = f['a_lo'] if np.isfinite(f['a_lo']) else b['a_lo']
    a_hi_k = f['a_hi'] if np.isfinite(f['a_hi']) else b['a_hi']
    a_mid_k = f['a_mid']
    if np.isfinite(a_mid_k):
      out['knee_from'] = 'free'
      out.update(turnovers(x, sp, b['nuM'], b['sigma'], a_lo_k, a_mid_k, a_hi_k))
      out.update(turnovers_curvature(x, sp, b['nuM'], b['sigma'], a_lo_k, a_mid_k, a_hi_k))
    else:
      # one turn: pass the HIGH asymptote where the mid one would go, so the single target
      # is (a_lo + a_hi)/2 and the single curvature bracket spans the whole transition
      out['knee_from'] = 'merged'
      out['nu_knee_1'] = turnovers(x, sp, b['nuM'], b['sigma'],
                                   a_lo_k, a_hi_k, np.nan)['nu_knee_lo']
      cv1 = turnovers_curvature(x, sp, b['nuM'], b['sigma'], a_lo_k, a_hi_k, np.nan)
      out['nu_curv_1'], out['dexc_1'] = cv1['nu_curv_lo'], cv1['dexc_lo']
      k1, c1 = out['nu_knee_1'], out['nu_curv_1']
      out['curv_off_1'] = (float(c1/k1) if np.isfinite(c1) and np.isfinite(k1) and k1 > 0.
                           else np.nan)
    # AGAINST BOTH CROSSINGS, and the second one is not optional -- see the section header.
    # b_* is the route's, whose low line is HELD at 4/3; b_*_free is the crossing of the
    # lines free_slopes actually measured. Where the spectrum's low index is not 4/3 the
    # held line is tilted and its crossing displaced, and knee/b inherits that displacement.
    for tag in ('lo', 'hi'):
      kn = out['nu_knee_' + tag]
      for pre, bb in (('knee_off_', out['b_' + tag]), ('kneef_off_', out['b_%s_free' % tag])):
        out[pre + tag] = (float(kn/bb) if np.isfinite(kn) and np.isfinite(bb) and bb > 0.
                          else np.nan)
      cv = out['nu_curv_' + tag]      # the two estimators against each other
      out['curv_off_' + tag] = (float(cv/kn) if np.isfinite(cv) and np.isfinite(kn)
                                and kn > 0. else np.nan)

  # the low index that needs no break: bounded by min(nu_m, nu_c) where the route found it,
  # so the scan cannot lock onto the fast-cooling nu^(1/2) plateau instead of the asymptote
  nu_break = b['b_lo'] if np.isfinite(b['b_lo']) else (b['b_hi'] if vfc else None)
  fl = sb.fluence_low_slope(x, sp, nu_break=nu_break)
  out.update(a_inf=fl['a_inf'], inf_conv=bool(fl['converged']),
             inf_band=bool(fl['in_band']))
  return out


# ---------------------------------------------------------------------------------------
# WHERE THE SPECTRUM ACTUALLY TURNS OVER, as against where the asymptotes cross
# ---------------------------------------------------------------------------------------
# b_lo and b_hi are CROSSINGS: extrapolate the two neighbouring segments until they meet.
# The spectrum is nowhere near that point -- it sits log10(2)/s dex below it -- so "the
# break" has two meanings and this module reports both.
#
# The turnover is taken as the HALF-SLOPE POINT: the frequency at which the local index has
# fallen half way from one segment's slope to the next's. That choice is not arbitrary. For
# granot_sari_syn, F = F_ext [y^(-s b1) + y^(-s b2)]^(-1/s), the local index at y = 1 is
#     d ln F/d ln y = [b1 y^(-s b1) + b2 y^(-s b2)] / (y^(-s b1) + y^(-s b2)) = (b1 + b2)/2
# EXACTLY, for every s. So on a GS02 break the half-slope point and the crossing are the same
# frequency whatever the smoothing, and knee/b is 1 by construction -- verified numerically
# to 1e-5 at s1 = 0.4, 0.8, 1.3, 2.0, 4.0. knee/b is therefore not a re-measurement of the
# break: it is a test of whether the break has the GS02 SHAPE, and it departs from 1 only
# when the turnover is asymmetric or when the neighbouring break overlaps it (the same
# synthetic returns knee_hi/nu_m = 1.13 once s1 = 0.4 smears the lower break into the mid
# segment, which is what SEP_CLEAN is about).
#
# READ knee/b AGAINST BOTH CROSSINGS OR NOT AT ALL. The route's b_lo is where the HELD 4/3
# line meets the mid line; b_lo_free is where the two lines free_slopes measured meet. On a
# spectrum whose low index is not 4/3 the held line is tilted, and the several decades of
# lever between window and crossing turn that tilt into a displaced b_lo -- so knee/b_lo
# picks up the displacement and reads as break asymmetry.
# That is not hypothetical, it is the whole result here. Against b_lo the time-integrated
# turnover looks far more displaced than the peak one (RS log10(C) = +3: 1.343 -> 1.537;
# +2: 1.346 -> 1.503; -2: 0.976 -> 1.317). Against b_lo_free the difference is GONE
# (1.270 -> 1.233, 1.268 -> 1.268, 0.958 -> 0.939): the two kinds agree to 2-4%, and what
# looked like a broader fluence break was the fluence spectrum's a_lo ~ 1.30 being held at
# 4/3. What DOES survive is common to both kinds -- knee/b_lo_free ~ 1.26-1.29 at every
# slow-cooling point, both shells, i.e. the turnover sits ~27% above even the free crossing.
# That is a departure from the GS02 break shape in these shell-integrated spectra, not
# anything to do with time integration.
#
# IS THE UPPER KNEE THE SPECTRAL PEAK? Only where a_mid = -a_hi, and that is a statement
# about the regime, not about the estimator. The knee is placed where the local index
# reaches t = (a_mid + a_hi)/2; the nuFnu maximum is where it reaches 0. So:
#   slow cooling   a_mid -> (3-p)/2, a_hi -> 1-p/2, hence t -> 0 and the two COINCIDE.
#                  Measured |t| <= 0.01 at log10(C) = +2, +3 both shells, and knee_hi/x_pk
#                  = 0.88-1.08 there.
#   fast cooling   a_mid -> 1/2 against a_hi -> 1-p/2, so t -> +0.13 and the knee sits
#                  BELOW the peak -- measured knee_hi/x_pk = 0.37-0.67, i.e. 0.2-0.4 dex.
#   log10(C) = +1  a_mid is still 0.17-0.21, short of the asymptote, so t goes NEGATIVE
#                  and the knee crosses to ABOVE the peak (1.19-1.52).
# The estimator itself is exact: run _slope_at_level at target 0 and it returns the nuFnu
# maximum to 0.0031 dex worst case over all 36 spectra of this sweep, against the parabolic
# argmax measured independently on the unflattened curve.
#
# WHICH MEANS: IN SLOW COOLING, QUOTE x_pk, NOT knee_hi. They are the same feature there,
# and x_pk needs no fit while knee_hi carries a_mid's error into it -- and carries it
# amplified, because the lever is 1/(ds/dlog nu) at the knee and the slope profile is
# flattest around the peak exactly when the breaks are far apart. Perturbing a_mid by
# +-0.05 moves knee_hi by 0.045-0.067 dex in fast cooling but 0.12-0.18 dex in slow, so a
# mid slope good to a few 0.01 still leaves the slow-cooling upper knee the softest number
# in the table. It is worth having as the counterpart of the lower knee and as the input to
# knee_sep; it is not worth preferring to the maximum it is trying to find.
#
# The SAG is the other half of the answer -- how far below the crossing the spectrum passes.
# free_slopes already measures it, as the deficit its s1/s2 are read off: sag = log10(2)/s
# dex. READ s_stable AND s_clean BEFORE EITHER. The deficit is taken at the FREE lines'
# crossing, which moves with the window standoff, so s drifts monotonically along the dfac
# ladder with no plateau (see _free_smoothing). These are standoff-tagged estimates of the
# sag depth, not measurements of the smoothing, and must not be quoted against GS02's
# tabulated s. The segment route's fit_smoothing_held is what measures s properly.


def _slope_at_level(lx, sl, target, lo=None, hi=None):
  """First frequency, scanning up, at which the local index falls to `target`."""
  m = np.isfinite(sl)
  if lo is not None:
    m &= lx >= lo
  if hi is not None:
    m &= lx <= hi
  if m.sum() < 2:
    return np.nan
  lxm, slm = lx[m], sl[m]
  k = np.flatnonzero(slm <= target)
  if not k.size or k[0] == 0:
    return np.nan            # never reaches it, or is already below it at the window start
  i = k[0]
  d = slm[i] - slm[i-1]
  f = 0.5 if d == 0. else (target - slm[i-1])/d
  return float(10.**(lxm[i-1] + f*(lxm[i] - lxm[i-1])))


def turnovers(x, sp, nuM, sigma, a_lo, a_mid, a_hi, cutfac=sb.CUT_FAC,
    smooth=sb.SLOPE_SMOOTH):
  """
  The half-slope point of each break, measured on the cut-off-flattened spectrum -- the
  same curve breaks_from_identified fits its lines to, so the turnover and the crossing
  refer to one spectrum and not to two versions of it.

  Needs the pair of slopes bounding each break; whichever are passed in are used, so the
  caller decides whether the mid slope is the measured one or the route's held fallback
  (and records which in `knee_from`).
  """
  out = dict(nu_knee_lo=np.nan, nu_knee_hi=np.nan)
  g = np.isfinite(sp) & (sp > 0.) & np.isfinite(x) & (x > 0.)
  if g.sum() < 12 or not np.isfinite(nuM) or nuM <= 0.:
    return out
  yv = sb._flatten_cutoff(x[g], sp[g], nuM, sigma)
  ok = np.isfinite(yv) & (yv > 0.)
  if ok.sum() < 12:
    return out
  lx, ly, sl = sb.segment_slopes(x[g][ok], yv[ok], smooth)
  cap = np.log10(nuM/cutfac)
  if np.isfinite(a_lo) and np.isfinite(a_mid):
    out['nu_knee_lo'] = _slope_at_level(lx, sl, 0.5*(a_lo + a_mid), hi=cap)
  if np.isfinite(a_mid) and np.isfinite(a_hi):
    out['nu_knee_hi'] = _slope_at_level(lx, sl, 0.5*(a_mid + a_hi), hi=cap)
  return out


CURV_SMOOTH = 9        # extra boxcar on the SECOND derivative. segment_slopes already
                       # boxcars the first over SLOPE_SMOOTH = 5 samples; differentiating
                       # that again re-injects the grid noise the first pass removed, and 9
                       # is where the curvature maximum of these spectra stops moving.
CURV_TOL = 0.02        # margin by which the bracket stands off each bounding slope
CURV_MIN_PTS = 5
CURV_MIN_DEX = 0.1


def _curv_argmax(lx, c, idx):
  '''Frequency of the largest |curvature| among samples `idx`, refined by a parabola.'''
  i = idx[int(np.argmax(np.abs(c[idx])))]
  if 0 < i < len(lx) - 1:
    q = np.abs(c[i-1:i+2])
    d = q[0] - 2.*q[1] + q[2]
    if d != 0.:
      return float(10.**(lx[i] - 0.5*(q[2] - q[0])/d*(lx[i] - lx[i-1])))
  return float(10.**lx[i])


def turnovers_curvature(x, sp, nuM, sigma, a_lo, a_mid, a_hi, cutfac=sb.CUT_FAC,
    smooth=sb.SLOPE_SMOOTH, csmooth=CURV_SMOOTH, tol=CURV_TOL):
  '''
  THE SECOND KNEE ESTIMATOR: the frequency of maximum log-log curvature, i.e. where the
  local index is changing fastest. Same null as the half-slope point, and derived the same
  way: for granot_sari_syn the index is b2 + D/(1 + e^{sDu}) in u = ln y with D = b1 - b2,
  so its derivative is -sD^2 sigma(1-sigma), maximal at sigma = 1/2, i.e. u = 0, i.e. AT
  the crossing -- for every s. Recovered to 1e-5 on synthetics at s1 = 0.8, 1.3, 2.0, 4.0.

  ITS ATTRACTION is that it needs no slope VALUE: an extremum is located by the curve
  alone, where the half-slope point has to be told which level to look for and therefore
  carries a_mid's error (0.12-0.18 dex per 0.05 of a_mid in slow cooling). That is the only
  reason it is here.

  THE BRACKET MUST BE ONE CONTIGUOUS RUN, and this is not a detail. Selecting every sample
  whose slope lies between the two bounding asymptotes gives a mask that RE-ENTERS near the
  cut-off, because flattening turns the curve back up past nuM and the slope climbs back
  into the window with enormous curvature. The argmax then lands in the rolloff rather than
  on the break: measured on this sweep the upper knee came back at nu_M/88 instead of the
  break, a factor 7.3e4 in slow cooling. Taking the widest contiguous run instead removes
  every such failure.

  READ THE COMPARISON BEFORE USING IT, because the two do NOT agree on these spectra.
  Measured over the whole sweep, both shells, both kinds (curv/knee in the table):
    median disagreement 0.207 dex -- a factor 1.6 -- at BOTH breaks;
    lower knee  curv/knee = 1.46-1.79 in SC, 0.68-1.08 in FC, i.e. biased HIGH and not
                by a constant;
    upper knee  0.29-2.33, straddling 1 without settling anywhere;
    three outright failures out of 36, all MC: FS log10(C) = -1 returns 114x the
                half-slope point on the peak spectrum and the BAND FLOOR (1.6e-16) on the
                time-integrated one, and FS +0 fluence returns 0.002x.
  Two causes, and neither is fixable by smoothing. (1) These transitions are 2.4-5.0 dex
  wide and asymmetric; a second derivative weights that asymmetry, a level crossing does
  not, so the curvature maximum drifts to the steep side. (2) In MC there is only ONE
  transition -- the index runs 4/3 -> 1-p/2 with no mid plateau -- so the two brackets
  cover parts of the same turn and the argmax returns essentially the SAME frequency for
  both: curv_hi/curv_lo = 0.97 at RS log10(C) = 0 and 0.98 at FS +0, against true
  separations of 18 and 98. The half-slope point survives MC because it asks for two
  different LEVELS on one monotonic curve, which stay distinct however merged the breaks.

  This is the same verdict spectral_breaks' header already records for curvature on these
  spectra, reached from the other side: there it was curvature as a THRESHOLD for segment
  detection, here as an EXTREMUM for break location, and the cause is the same coherent
  in-segment curvature (lag-1 autocorrelation 0.998) that makes the knee no longer special.

  `nu_knee_*` remains the primary. This is reported for comparison and nothing else.
  '''
  out = dict(nu_curv_lo=np.nan, nu_curv_hi=np.nan, dexc_lo=np.nan, dexc_hi=np.nan)
  g = np.isfinite(sp) & (sp > 0.) & np.isfinite(x) & (x > 0.)
  if g.sum() < 12 or not np.isfinite(nuM) or nuM <= 0.:
    return out
  yv = sb._flatten_cutoff(x[g], sp[g], nuM, sigma)
  ok = np.isfinite(yv) & (yv > 0.)
  if ok.sum() < 12:
    return out
  lx, ly, sl = sb.segment_slopes(x[g][ok], yv[ok], smooth)
  cap = np.log10(nuM/cutfac)
  c = sb._boxcar(np.gradient(sl, lx), csmooth)
  for tag, a_bot, a_top in (('lo', a_mid, a_lo), ('hi', a_hi, a_mid)):
    if not (np.isfinite(a_bot) and np.isfinite(a_top)):
      continue
    m = (sl > a_bot + tol) & (sl < a_top - tol) & (lx <= cap) & np.isfinite(c)
    w = sb._widest_run(m, lx, min_pts=CURV_MIN_PTS, min_dex=CURV_MIN_DEX)
    if w is None:
      continue
    idx = np.arange(w[0], w[1] + 1)
    out['dexc_' + tag] = float(lx[w[1]] - lx[w[0]])
    out['nu_curv_' + tag] = _curv_argmax(lx, c, idx)
  return out


def measure_spectrum(x, sp, psyn, **kw):
  '''Every measurement this module makes on one spectrum: width block + segment block.'''
  m = peak_and_width(x, sp)
  m.update(segments_and_breaks(x, sp, psyn, **kw))
  m['bhi_over_xpk'] = (m['b_hi']/m['x_pk']
                       if np.isfinite(m['b_hi']) and m['x_pk'] > 0. else np.nan)
  # the separation the TURNOVERS give, as against `sep` = b_hi/b_lo from the crossings
  m['knee_sep'] = (m['nu_knee_hi']/m['nu_knee_lo']
                   if np.isfinite(m['nu_knee_hi']) and np.isfinite(m['nu_knee_lo'])
                   and m['nu_knee_lo'] > 0. else np.nan)
  # THE TWO REFERENCE FREQUENCIES THE FIGURES USE.
  #   nu_bk  the low-energy break, as MEASURED: nu_knee_lo, and ONLY nu_knee_lo. It is
  #          left undefined wherever a separated lower break does not exist -- below
  #          log10(C) = -2, where nu_c falls under the synchrotron floor nu_B, and at the
  #          merged regimes, where the single turn is neither break but the pair of them
  #          run together. The merged knee is still measured and still in the table as
  #          nu_knee_1; it is simply not this quantity, and substituting it here would put
  #          a different measurement into the middle of the series. nu_pk is what carries
  #          those regimes on the figure.
  #   nu_pk  the high-energy reference: the nuFnu maximum, NOT nu_knee_hi. The maximum is
  #          fit-free (parabolic argmax, verified to 0.0031 dex against the estimator run
  #          at target slope 0) and defined at every regime, while nu_knee_hi is levered by
  #          a_mid -- 0.12-0.18 dex per 0.05 of it in slow cooling, the softest number in
  #          the table -- and undefined once the breaks merge.
  m['nu_bk'] = m['nu_knee_lo']
  m['nu_pk'] = m['x_pk']
  m['pk_over_bk'] = (m['nu_pk']/m['nu_bk']
                     if np.isfinite(m['nu_bk']) and m['nu_bk'] > 0. else np.nan)
  return m


def measure_point(r, z, **kw):
  '''Both spectrum kinds of one sweep point, as two rows.'''
  x = nu_over_num(r)
  env = r['env']
  sp = spectra_of(r)
  rows = []
  for kind in KINDS:
    if sp[kind] is None:
      continue
    m = measure_spectrum(x, sp[kind], env.psyn, **kw)
    # C is the SWEEP LABEL's ratio, which is the reverse shock's whatever z is -- the
    # observer grids are RS-normalised for both shells and every figure in this suite is
    # drawn against it. C_shell is the emitting shell's OWN gamma_c/gamma_m, and the two
    # are NOT the same number on the forward shock: at the label log10(C) = +2 the FS sits
    # at C = 312, i.e. +2.49. Read a cross-shell comparison with that in hand.
    C_shell = float(env.gma_c/env.gma_m)
    if z != Z_RS and hasattr(env, 'gma_cFS') and hasattr(env, 'gma_mFS'):
      C_shell = float(env.gma_cFS/env.gma_mFS)
    m.update(kind=kind, z=z, logr=float(r['log10ratio']), psyn=float(env.psyn),
             C=float(env.gma_c/env.gma_m), C_shell=C_shell,
             nuM_env=float((env.gma_max/env.gma_m)**2))
    rows.append(m)
  return rows


def measure_sweep(results, z, **kw):
  '''measure_point over a whole cached sweep; a flat list of rows.'''
  rows = []
  for r in results:
    rows += measure_point(r, z, **kw)
  return rows


# ---------------------------------------------------------------------------------------
# the fluence-against-peak pairing: same sweep point, same shell, the two kinds
# ---------------------------------------------------------------------------------------
# SLOPES ARE COMPARED AS DIFFERENCES, frequencies and widths as RATIOS. A log-log slope is
# already a logarithm, so the ratio of two of them is not a meaningful quantity (it diverges
# wherever the peak value passes through zero, which a_mid does between FC and SC); a break
# position and a width are scales, and their ratio is what "shifted down by x" means.
_DIFF_KEYS = ('a_lo', 'a_mid', 'a_hi', 'a_inf', 'asym_half', 'asym_tenth',
              'logW_half', 'logW_tenth')
_RATIO_KEYS = ('x_pk', 'nu_bk', 'pk_over_bk', 'b_lo', 'b_hi', 'nu_knee_lo',
               'nu_knee_hi', 'nu_knee_1', 'sep', 'knee_sep', 'nuM',
               'W_half', 'W_tenth', 'Wlo_half', 'Whi_half', 'F_pk')


def pair_rows(rows):
  '''
  One row per (shell, sweep point) holding the fluence/peak comparison: `d_*` differences
  for the slopes, `R_*` ratios for the scales, and `class_flip` where the two kinds are not
  the same shape class -- the case in which every segment-derived column below is comparing
  two different measurements and only the width block is like for like.
  '''
  by = {}
  for m in rows:
    by.setdefault((m['z'], m['logr']), {})[m['kind']] = m
  out = []
  for (z, logr), d in sorted(by.items(), key=lambda kv: (-kv[0][0], kv[0][1])):
    if not {'peak', 'fluence'} <= set(d):
      continue
    pk, fl = d['peak'], d['fluence']
    row = dict(z=z, logr=logr, C=pk['C'],
               cls_peak=pk['cls'], cls_flu=fl['cls'],
               class_flip=bool(pk['cls'] != fl['cls']),
               # a difference between two slopes is a measurement only where BOTH sides
               # measured one. Where either low index is unconverged, d_a_lo / d_a_inf is
               # the gap between two bounds and reads as a large spurious softening --
               # -0.25 at log10(C) = -3, where neither kind reaches the asymptote in band.
               lo_conv=bool(pk['lo_conv'] and fl['lo_conv']),
               inf_conv=bool(pk['inf_conv'] and fl['inf_conv']),
               # the knee target needed a held mid slope on BOTH sides or on neither;
               # the figures grey the points where it did
               knee_held=bool(pk['knee_from'] == 'merged'
                              or fl['knee_from'] == 'merged'))
    for k in _DIFF_KEYS:
      row['d_' + k] = fl[k] - pk[k]
    for k in _RATIO_KEYS:
      row['R_' + k] = (fl[k]/pk[k] if np.isfinite(fl[k]) and np.isfinite(pk[k])
                       and pk[k] != 0. else np.nan)
    out.append(row)
  return out


# ---------------------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------------------
SHAPE_CSV = 'spectrum_shape_table.csv'
RATIO_CSV = 'spectrum_shape_ratios.csv'
# NOT 'spectrum_shape_fluence_vs_peak.csv': _write_table derives its png from the csv name,
# and that collided exactly with plot_fluence_vs_peak's figure. build_tables runs first, so
# the figure silently overwrote the table png on every run until the figures were split per
# shell and the clash disappeared by accident. Renamed so it cannot come back.

_COLS = [('shell', 'shell', '{:s}'), ('log10(C)', 'logr', '{:+.0f}'),
         ('log10(C) shell', 'logC_shell', '{:+.2f}'),
         ('kind', 'kind', '{:s}'), ('class', 'cls', '{:s}'),
         ('nu_pk', 'nu_pk', '{:.4g}'), ('nu_bk', 'nu_bk', '{:.4g}'),
         ('nu_pk/nu_bk', 'pk_over_bk', '{:.4g}'),
         ('top_dex', 'top_dex', '{:.2f}'),
         ('nu_1/2 lo', 'nu_lo_half', '{:.4g}'), ('nu_1/2 hi', 'nu_hi_half', '{:.4g}'),
         ('W_1/2', 'W_half', '{:.4g}'), ('log W_1/2', 'logW_half', '{:.3f}'),
         ('W_lo', 'Wlo_half', '{:.3g}'), ('W_hi', 'Whi_half', '{:.3g}'),
         ('asym', 'asym_half', '{:.3f}'),
         ('W_1/10', 'W_tenth', '{:.4g}'), ('log W_1/10', 'logW_tenth', '{:.3f}'),
         ('b_lo', 'b_lo', '{:.4g}'), ('b_hi', 'b_hi', '{:.4g}'),
         ('b_hi/b_lo', 'sep', '{:.4g}'), ('b_hi/x_pk', 'bhi_over_xpk', '{:.3f}'),
         ('knee_hi/knee_lo', 'knee_sep', '{:.4g}'),
         ('knee_lo', 'nu_knee_lo', '{:.4g}'), ('knee_hi', 'nu_knee_hi', '{:.4g}'),
         ('knee/b lo', 'knee_off_lo', '{:.3f}'), ('knee/b hi', 'knee_off_hi', '{:.3f}'),
         ('knee/bf lo', 'kneef_off_lo', '{:.3f}'),
         ('knee/bf hi', 'kneef_off_hi', '{:.3f}'),
         ('knee_1', 'nu_knee_1', '{:.4g}'), ('curv_1', 'nu_curv_1', '{:.4g}'),
         ('curv/knee 1', 'curv_off_1', '{:.3f}'),
         ('curv_lo', 'nu_curv_lo', '{:.4g}'), ('curv_hi', 'nu_curv_hi', '{:.4g}'),
         ('curv/knee lo', 'curv_off_lo', '{:.3f}'),
         ('curv/knee hi', 'curv_off_hi', '{:.3f}'),
         ('b_lo free', 'b_lo_free', '{:.4g}'), ('b_hi free', 'b_hi_free', '{:.4g}'),
         ('knee_from', 'knee_from', '{:s}'),
         ('sag_lo', 'sag_lo', '{:.3f}'), ('sag_hi', 'sag_hi', '{:.3f}'),
         ('s_stab', 's_stable', '{:d}'), ('s_clean', 's_clean', '{:d}'),
         ('nu_M', 'nuM', '{:.4g}'), ('sigma', 'sigma', '{:.3f}'),
         ('a_lo', 'a_lo', '{:+.3f}'), ('lo_conv', 'lo_conv', '{:d}'),
         ('a_mid', 'a_mid', '{:+.3f}'), ('a_hi', 'a_hi', '{:+.4f}'),
         ('dex_lo', 'dex_lo', '{:.2f}'), ('dex_mid', 'dex_mid', '{:.2f}'),
         ('dex_hi', 'dex_hi', '{:.2f}'),
         ('a_inf', 'a_inf', '{:+.3f}'), ('inf_conv', 'inf_conv', '{:d}'),
         ('a_lo_h', 'a_lo_h', '{:+.3f}'), ('a_hi_h', 'a_hi_h', '{:+.3f}'),
         ('mid_from', 'mid_from', '{:s}'), ('ok', 'br_ok', '{:d}')]

_RCOLS = [('shell', 'shell', '{:s}'), ('log10(C)', 'logr', '{:+.0f}'),
          ('class pk', 'cls_peak', '{:s}'), ('class flu', 'cls_flu', '{:s}'),
          ('flip', 'class_flip', '{:d}'),
          ('W_1/2 flu/pk', 'R_W_half', '{:.3f}'),
          ('dlogW_1/2', 'd_logW_half', '{:+.3f}'),
          ('W_1/10 flu/pk', 'R_W_tenth', '{:.3f}'),
          ('W_lo flu/pk', 'R_Wlo_half', '{:.3f}'),
          ('W_hi flu/pk', 'R_Whi_half', '{:.3f}'),
          ('d asym', 'd_asym_half', '{:+.3f}'),
          ('nu_pk flu/pk', 'R_x_pk', '{:.3f}'),
          ('nu_bk flu/pk', 'R_nu_bk', '{:.3f}'),
          ('nu_pk/nu_bk flu/pk', 'R_pk_over_bk', '{:.3f}'),
          ('b_lo flu/pk', 'R_b_lo', '{:.3f}'), ('b_hi flu/pk', 'R_b_hi', '{:.3f}'),
          ('knee_lo flu/pk', 'R_nu_knee_lo', '{:.3f}'),
          ('knee_hi flu/pk', 'R_nu_knee_hi', '{:.3f}'),
          ('knee_held', 'knee_held', '{:d}'),
          ('sep flu/pk', 'R_sep', '{:.3f}'), ('nu_M flu/pk', 'R_nuM', '{:.3f}'),
          ('d a_lo', 'd_a_lo', '{:+.3f}'), ('lo_conv', 'lo_conv', '{:d}'),
          ('d a_mid', 'd_a_mid', '{:+.3f}'),
          ('d a_hi', 'd_a_hi', '{:+.4f}'), ('d a_inf', 'd_a_inf', '{:+.3f}'),
          ('inf_conv', 'inf_conv', '{:d}')]

_NOTE = (
  'Frequencies are nu/nu_{m,0} (the collision nu_m of MyEnv, nu_over_num), so a column can '
  'be read across the sweep.\n'
  'log10(C) is the SWEEP LABEL, which is the reverse shock ratio for both shells (the '
  'observer grids are RS-normalised, and every figure here is drawn against it). '
  '"log10(C) shell" is the emitting shell own gamma_c/gamma_m: on the FS it runs ~0.5 dex '
  'above the label, so the two shells at one x are NOT at the same cooling ratio.\n'
  'The PEAK spectrum is the observer-time row at which the BOLOMETRIC flux int F_nu dnu '
  'is maximal (bolometric_lightcurve), not the row where the lightcurve at one reference '
  'frequency peaks. sweep_gammacm.plot_spectra_pair still uses the latter, so its peak '
  'panel is a different row from this one.\n'
  'REFERENCE FREQUENCIES: nu_pk is the nuFnu maximum and nu_bk the measured low-energy '
  'break (nu_knee_lo alone -- blank below log10(C) = -2, where nu_c falls under nu_B, and '
  'at the merged regimes, where the single turn is the pair run together and is reported '
  'separately as knee_1). Those two, not the knee pair, are what the ratio figure uses.\n'
  'WIDTH (no fit, no class): x_pk is the nuFnu maximum, W_1/2 = nu_hi/nu_lo the ratio of '
  'the frequencies at half of it, W_lo = x_pk/nu_lo and W_hi = nu_hi/x_pk its two halves, '
  'asym = log(W_hi)/log(W_lo) (1 = symmetric in the log). top_dex is the span within 1% of '
  'the maximum -- how flat the top the peak was taken on is.\n'
  'BREAKS (spectral_breaks.breaks_from_identified, the paper route): b_lo/b_hi are where '
  'the identified segments CROSS, which is not where the spectrum peaks -- b_hi/x_pk says '
  'how far apart the two definitions are for that spectrum. nu_M and sigma are the smeared '
  'cut-off. class is what the spectrum displays; a peak/fluence disagreement is flagged in '
  'the ratio table.\n'
  'TURNOVER, i.e. where the spectrum actually breaks rather than where the asymptotes meet: '
  'knee_lo/knee_hi are the half-slope points, and sag_* is how far below the crossing the '
  'spectrum passes, in dex. On a Granot-Sari break the half-slope point IS the crossing for '
  'any smoothing, so knee/b = 1 is the null and a departure means the turnover is '
  'asymmetric or overlapped by its neighbour -- it is a shape test, not a second break '
  'position. knee_from=route: no mid plateau existed, so the target used the route held mid '
  'slope. sag = log10(2)/s at the FREE lines crossing, which moves with the window '
  'standoff: read s_stab (and s_clean, the separation gate) before quoting either, and '
  'never against GS02 tabulated s.\n'
  'knee/b uses the route crossing, whose low line is HELD at 4/3; knee/bf uses the crossing '
  'of the lines actually measured (b_lo free / b_hi free). QUOTE knee/bf. Where a spectrum '
  'does not carry 4/3 the held line is tilted and its crossing displaced, and knee/b '
  'inherits that: the peak-to-fluence growth visible in knee/b lo is absent from knee/bf lo.'
  '\n'
  'SLOPES: a_lo/a_mid/a_hi are MEASURED (free_slopes: a free line in a window standing off '
  'FREE_DFAC from each break, no slope value in the selection), dex_* their widths. a_lo is '
  'NaN by construction in the single-break classes VFC/FC*, which assert no nu^(4/3) '
  'segment in band. lo_conv=0 means the low window and the one 3x further out disagree, '
  'i.e. the asymptote is not reached inside the band -- a bound, not a value. a_inf is the '
  'second, self-anchored low index (fluence_low_slope), which needs no break.\n'
  'a_lo_h and a_hi_h are HELD at 4/3 and 1-p/2. They are inputs to the crossing, not '
  'measurements: compare them with a_lo/a_hi, do not quote them as slopes.')

_RNOTE = (
  'fluence AGAINST peak, same shell and same sweep point. Slopes are compared as '
  'DIFFERENCES (d_*), scales as RATIOS (flu/pk) -- the ratio of two log-log slopes is not a '
  'meaningful quantity, and a_mid passes through zero between FC and SC.\n'
  'flip=1: the two kinds were assigned different shape classes, so every segment-derived '
  'column in that row compares two different measurements. The width block is the only part '
  'that is like for like there.\n'
  'lo_conv / inf_conv = 1 only where BOTH kinds reached the low-energy asymptote inside the '
  'band. Where they are 0 the corresponding d_a_lo / d_a_inf is the gap between two bounds, '
  'not a slope difference, and it is large for that reason alone.')


def _shell_name(z):
  return {Z_RS: 'RS', Z_FS: 'FS'}.get(z, f'z={z}')


def _cell(m, key, fmt):
  v = m.get(key)
  if v is None:
    return '--'
  if fmt.endswith('d}'):
    return fmt.format(int(bool(v)))
  if fmt.endswith('s}'):
    return fmt.format(str(v))
  return '--' if not np.isfinite(v) else fmt.format(v)


def _write_table(rows, cols, path, note, outdir, title):
  '''csv + printed + png, the three forms every table in this suite comes in.'''
  body = [[_cell(m, k, fmt) for _, k, fmt in cols] for m in rows]
  head = [c[0] for c in cols]
  with open(path, 'w', newline='') as f:
    w = csv.writer(f); w.writerow(head); w.writerows(body)
  write_table_stamp(path, outdir)
  wd = [max(len(head[i]), max((len(b[i]) for b in body), default=0))
        for i in range(len(head))]
  fmt_row = lambda rw: '  '.join(v.ljust(wd[i]) for i, v in enumerate(rw))
  print(f'\n--- {title} ' + '-'*40)
  print(fmt_row(head)); print('  '.join('-'*w for w in wd))
  for b in body:
    print(fmt_row(b))
  print('\n' + note)
  fig, ax = plt.subplots(figsize=(0.13*sum(wd) + 2., 0.32*len(body) + 2.4)); ax.axis('off')
  tbl = ax.table(cellText=body, colLabels=head, loc='center', cellLoc='center')
  tbl.auto_set_font_size(False); tbl.set_fontsize(6.5); tbl.scale(1, 1.25)
  ax.set_title(note, fontsize=6.5)
  fig.savefig(path.replace('.csv', '.png'), dpi=200, bbox_inches='tight')
  plt.close(fig)
  return path


def build_tables(rows, outdir):
  '''The measurement table and the fluence-against-peak table.'''
  for m in rows:
    m['shell'] = _shell_name(m['z'])
    m['logC_shell'] = np.log10(m['C_shell']) if m.get('C_shell', 0.) > 0. else np.nan
  srt = sorted(rows, key=lambda m: (-m['z'], m['logr'], KINDS.index(m['kind'])))
  p1 = _write_table(srt, _COLS, os.path.join(outdir, SHAPE_CSV), _NOTE, outdir,
                    'Peak and time-integrated spectra')
  pairs = pair_rows(rows)
  for m in pairs:
    m['shell'] = _shell_name(m['z'])
  p2 = _write_table(pairs, _RCOLS, os.path.join(outdir, RATIO_CSV), _RNOTE, outdir,
                    'Time-integrated against peak, same log10(C)')
  print(f'\ntables -> {outdir}/{SHAPE_CSV}, {RATIO_CSV} (+ .png)')
  return p1, p2, pairs


# ---------------------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------------------
# ONE SHELL PER FIGURE. The two shells were overlaid by colour until 2026-09-12, which cost
# the colour channel: every quantity in a panel then had to be told apart by marker shape
# alone, and a panel carrying three of them ran out of shapes that read differently at 4 pt
# (nu_knee_lo and nu_knee_hi were a down- and an up-triangle, which is the pair that made
# this necessary). Split, colour is free and carries the QUANTITY, so each series is
# legible on its own; the shell is named INSIDE the panel, as the article convention wants
# a parameter carried.
#
# The kind stays the linestyle: peak dashed with an open marker, time-integrated solid and
# filled, as the fluence_vs_peak overlay already drew them.
#
# Quantity colours are lightcurve_shape's NU_COLORS family (purple/orange/green, plus brown
# for a fourth) rather than anything near the shell pair -- COL_RS/COL_FS mean RS and FS
# throughout this project, and a red curve inside a figure labelled FS would read as the
# other shell however the legend is worded. COL_RS/COL_FS are still used, for the one thing
# they should be: the shell tag itself.
_STY = {'peak': dict(ls='--', marker='o', ms=4, mfc='none'),
        'fluence': dict(ls='-', marker='o', ms=4)}
_CLABEL = '$\\log_{10}\\mathcal{C}$'
_QCOL = ('tab:purple', 'tab:orange', 'tab:green', 'tab:brown')
_QMK = ('o', 's', 'D', '^')       # circle / square / diamond / triangle: four shapes that
                                  # stay distinguishable filled or open at this size


def _held_mid_band(ax, logrs, half=0.35, gap=1.5):
  '''
  Shade the sweep points whose knee target needed a HELD mid slope. Those are the MC
  points, where no mid plateau exists to measure one on: the turnover there is not a free
  measurement, and the band says so without spending a colour, a marker or a fill on it.

  ADJACENT POINTS GET ONE SPAN, not one each. The MC points are consecutive on the sweep
  (log10(C) = -1 and 0, a single marginal-cooling stretch), and drawing them as two bands
  with a sliver of white between reads as two separate exclusions rather than the one
  regime it is. Runs are cut where the gap exceeds `gap`, so a genuinely isolated point
  still gets its own band.
  '''
  # deliberately unlabelled: the callers either build their legend from explicit handles
  # or explain the band in the figure caption, and an axvspan label picked up by a bare
  # ax.legend() lands a full-width swatch in the middle of a crowded panel
  v = sorted(set(logrs))
  if not v:
    return
  runs, lo, prev = [], v[0], v[0]
  for x in v[1:]:
    if x - prev <= gap:
      prev = x
    else:
      runs.append((lo, prev)); lo = prev = x
  runs.append((lo, prev))
  for lo, hi in runs:
    ax.axvspan(lo - half, hi + half, color='0.85', alpha=0.55, lw=0, zorder=0)


def _shell_tag(ax, z):
  '''The shell, inside the panel -- these figures carry no titles.'''
  # upper LEFT with a white box behind it: upper right is where the steeply rising
  # break and separation curves end up, and a bare tag there is drawn over by them
  ax.annotate(_shell_name(z), (0.025, 0.97), xycoords='axes fraction', fontsize=9,
              fontweight='bold', ha='left', va='top', zorder=7,
              color=COL_RS if z == Z_RS else COL_FS,
              bbox=dict(fc='w', ec='none', alpha=0.8, pad=1.5))


def _series(rows, z, kind, key):
  s = sorted([m for m in rows if m['z'] == z and m['kind'] == kind],
             key=lambda m: m['logr'])
  return (np.array([m['logr'] for m in s], float),
          np.array([m[key] for m in s], float))


def _kind_legend(ax, color='0.35', marker=None, **kw):
  '''
  The peak/time-integrated key. `marker` and `color` OVERRIDE the _STY defaults and must be
  set to whatever the panel actually drew: _STY carries a circle, so a panel that recoloured
  and re-markered its series got a legend showing circles against squares on the axes.
  '''
  h = []
  for k in KINDS:
    st = dict(_STY[k])
    st['color'] = color
    if marker is not None:
      st['marker'] = marker
    h.append(plt.Line2D([], [], **st))
  ax.legend(h, ['peak', 'time-integrated'], fontsize=7, **kw)


def _qstyle(kind, i):
  '''Series style: colour and marker from the quantity, fill and dash from the kind.'''
  st = dict(_STY[kind])
  st.update(color=_QCOL[i % len(_QCOL)], marker=_QMK[i % len(_QMK)])
  if kind == 'peak':
    st['mfc'] = 'none'
  return st


def _qlegend(ax, labels, kinds=True, **kw):
  '''Quantity legend (colour + marker), with the two kinds appended unless told not to.'''
  h = [plt.Line2D([], [], color=_QCOL[i % len(_QCOL)], marker=_QMK[i % len(_QMK)],
                  ms=4.5, ls='-') for i in range(len(labels))]
  lab = list(labels)
  if kinds:
    h += [plt.Line2D([], [], color='0.35', **_STY[k]) for k in KINDS]
    lab += ['peak', 'time-integrated']
  ax.legend(h, lab, fontsize=7, **kw)


def plot_shape_vs_regime(rows, outdir, z):
  '''
  The three answers against the cooling regime, for ONE shell: the free slopes, where the
  spectrum turns over, and the half-maximum width.

  The low-slope points that did NOT converge inside the band are drawn hollow and
  unjoined. They are bounds, not measurements (see lo_conv), and joining them to the
  converged ones draws a slope change that is a band limit.
  '''
  fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.2))
  p = float(np.median([m['psyn'] for m in rows]))

  # --- free slopes, one colour per segment
  ax = axes[0]
  for a, lab in ((4./3., '$4/3$'), (0.5, '$1/2$'), ((3.-p)/2., '$(3-p)/2$'),
                 (1. - p/2., '$1-p/2$')):
    ax.axhline(a, color='0.75', lw=0.8, zorder=0)
    # a white box behind the guide label: a_hi sits ON its own guide across the whole
    # sweep, so an unboxed label there is drawn over by the curve it labels
    ax.annotate(lab, (0.995, a), xycoords=('axes fraction', 'data'), fontsize=6.5,
                color='0.45', va='bottom', ha='right', zorder=6,
                bbox=dict(fc='w', ec='none', alpha=0.75, pad=0.6))
  for kind in KINDS:
    for i, key in enumerate(('a_lo', 'a_mid', 'a_hi')):
      st = _qstyle(kind, i)
      lr, v = _series(rows, z, kind, key)
      if key == 'a_lo':          # hollow and unjoined where the asymptote is not in band
        cv = _series(rows, z, kind, 'lo_conv')[1].astype(bool)
        ax.plot(np.where(cv, lr, np.nan), np.where(cv, v, np.nan), **st)
        ax.plot(lr[~cv], v[~cv], color=st['color'], ls='none', marker=st['marker'],
                ms=st['ms'], mfc='none', alpha=0.55)
      else:
        ax.plot(lr, v, **st)
  ax.set_ylabel('free slope  $\\mathrm{d}\\log\\nu F_\\nu/\\mathrm{d}\\log\\nu$')
  # the empty band between the mid segments (~0.2-0.63) and the low ones (~1.33) is the
  # only part of this panel no curve crosses at any regime
  _qlegend(ax, ['$a_{\\rm lo}$', '$a_{\\rm mid}$', '$a_{\\rm hi}$'], loc='center',
           bbox_to_anchor=(0.5, 0.63), ncol=2, framealpha=0.9)

  # --- the TURNOVERS and the nuFnu maximum. The knees, not the crossings: b_lo/b_hi are
  # where the extrapolated asymptotes meet, which is a point the spectrum never passes
  # through, and at the lower break the route's held 4/3 line displaces it (see the
  # turnover section header). Both crossings stay in the table.
  ax = axes[1]
  _held_mid_band(ax, [m['logr'] for m in rows
                      if m['z'] == z and m['knee_from'] == 'merged'])
  bkeys = (('nu_knee_lo', '$\\nu_{\\rm knee,lo}$'),
           ('nu_knee_hi', '$\\nu_{\\rm knee,hi}$'),
           ('x_pk', '$\\nu_{\\rm pk}$'))
  for kind in KINDS:
    for i, (key, _) in enumerate(bkeys):
      lr, v = _series(rows, z, kind, key)
      ax.semilogy(lr, v, **_qstyle(kind, i))
  ax.set_ylabel('$\\nu/\\nu_{\\mathrm{m},\\!0}$')
  _qlegend(ax, [lab for _, lab in bkeys], loc='lower right', ncol=2)

  # --- the width
  ax = axes[2]
  for kind in KINDS:
    for i, key in enumerate(('logW_half', 'logW_tenth')):
      ax.plot(*_series(rows, z, kind, key), **_qstyle(kind, i))
  ax.set_ylabel('$\\log_{10}$ peak width  $\\nu_+/\\nu_-$')
  _qlegend(ax, ['half maximum', 'tenth maximum'], loc='lower right')

  for ax in axes:
    ax.set_xlabel(_CLABEL)
    ax.grid(alpha=0.25)
    _shell_tag(ax, z)
  fig.tight_layout()
  fig.text(0.5, -0.005, 'hollow, unjoined = asymptote not reached inside the band;   '
           'grey band = no mid plateau, knee target used a held mid slope',
           fontsize=7.5, color='0.3', ha='center')
  f = os.path.join(outdir, 'spectrum_shape_vs_regime_%s.png' % _shell_name(z))
  fig.savefig(f, dpi=200, bbox_inches='tight'); plt.close(fig)
  return f


def plot_fluence_vs_peak(pairs, outdir, z):
  '''
  The comparison itself, for ONE shell: how far each quantity moves from peak to
  time-integrated.

  The two low-energy slope differences are masked on their convergence flags -- an
  unconverged pair differs by up to 0.25, which is the gap between two band limits and
  not a softening. Those points are drawn hollow and unjoined, as in the table.
  '''
  s_ = sorted([m for m in pairs if m['z'] == z], key=lambda m: m['logr'])
  lr = np.array([m['logr'] for m in s_], float)
  flip = np.array([m['class_flip'] for m in s_], bool)
  col = lambda k: np.array([m[k] for m in s_], float)

  fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.2))
  panels = [(('R_W_half', '$W_{1/2}$', None), ('R_W_tenth', '$W_{1/10}$', None),
             ('R_Wlo_half', '$W_{\\rm lo}$', None), ('R_Whi_half', '$W_{\\rm hi}$', None)),
            (('R_x_pk', '$\\nu_{\\rm pk}$', None),
             ('R_nu_knee_lo', '$\\nu_{\\rm knee,lo}$', None),
             ('R_nu_knee_hi', '$\\nu_{\\rm knee,hi}$', None),
             ('R_nuM', '$\\nu_{\\rm M}$', None)),
            (('d_a_lo', '$a_{\\rm lo}$', 'lo_conv'), ('d_a_mid', '$a_{\\rm mid}$', None),
             ('d_a_hi', '$a_{\\rm hi}$', None), ('d_a_inf', '$a_\\infty$', 'inf_conv'))]
  for ax, keys in zip(axes, panels):
    if any(k.startswith('R_nu_knee') for k, _, _ in keys):
      _held_mid_band(ax, [m['logr'] for m in s_ if m['knee_held']])
    ax.axhline(1. if keys[0][0].startswith('R_') else 0., color='0.6', lw=0.9, zorder=0)
    for i, (k, lab, gate) in enumerate(keys):
      v = col(k)
      c, mk = _QCOL[i], _QMK[i]
      if gate is None:
        ax.plot(lr, v, color=c, ls='-', lw=1.1, marker=mk, ms=4.5, label=lab)
      else:
        cv = col(gate).astype(bool)
        ax.plot(np.where(cv, lr, np.nan), np.where(cv, v, np.nan), color=c, ls='-',
                lw=1.1, marker=mk, ms=4.5, label=lab)
        ax.plot(lr[~cv], v[~cv], color=c, ls='none', marker=mk, ms=4.5, mfc='none',
                alpha=0.55)
      if flip.any():         # a class flip: the segment columns are not like for like
        ax.plot(lr[flip], v[flip], ls='none', marker='x', ms=9, color='k', zorder=5)
    ax.set_xlabel(_CLABEL)
    ax.grid(alpha=0.25)
    # 'best' does not know about annotations, and _shell_tag lives in the top-left
    # corner; the bbox keeps the legend out of the strip the tag occupies
    ax.legend(fontsize=7, ncol=2, loc='best', bbox_to_anchor=(0., 0., 1., 0.92))
    _shell_tag(ax, z)
  axes[0].set_ylabel('width, time-integrated / peak')
  axes[1].set_ylabel('frequency, time-integrated / peak')
  axes[1].set_yscale('log')
  axes[2].set_ylabel('slope, time-integrated $-$ peak')
  # SCALE THE SLOPE PANEL ON THE MEASUREMENTS ALONE. The gated points are differences
  # between two band limits, not slope differences -- they reach -0.25 while everything
  # measured sits inside +-0.04, so letting them set the axis flattens the real structure
  # into a line on zero. They are still drawn, hollow, and the count of any that fall off
  # scale is stated rather than left for the reader to notice.
  keep = []
  for k, _, gate in panels[2]:
    v = col(k)
    m = np.isfinite(v)
    if gate is not None:
      m &= col(gate).astype(bool)
    keep.append(v[m])
  keep = np.concatenate(keep) if keep else np.array([])
  if keep.size:
    lo, hi = float(keep.min()), float(keep.max())
    pad = max(0.15*(hi - lo), 0.005)
    lo, hi = min(lo - pad, -pad), max(hi + pad, pad)
    axes[2].set_ylim(lo, hi)
    off = sum(int(np.sum(np.isfinite(col(k)) & ((col(k) < lo) | (col(k) > hi))))
              for k, _, _ in panels[2])
    if off:
      axes[2].annotate(f'{off} point{"s" if off > 1 else ""} off scale (not measurements)',
                       (0.5, 0.02), xycoords='axes fraction', fontsize=6.5, color='0.45',
                       ha='center')
  fig.tight_layout()
  fig.text(0.5, -0.005, '$\\times$ = shape class differs;   hollow, unjoined = asymptote '
           'not reached inside the band;   grey band = no mid plateau, knee target used a '
           'held mid slope', fontsize=7.5, color='0.3', ha='center')
  f = os.path.join(outdir, 'spectrum_shape_fluence_vs_peak_%s.png' % _shell_name(z))
  fig.savefig(f, dpi=200, bbox_inches='tight'); plt.close(fig)
  return f


def plot_knee_ratios(rows, outdir, z):
  '''
  The two reference frequencies of this module, against the cooling regime, for ONE shell.

    nu_bk   the low-energy break AS MEASURED -- nu_knee_lo, and only where a separated
            lower break exists. It does not below log10(C) = -2, where nu_c falls under
            the synchrotron floor nu_B (nu_c/nu_B = 0.048 at -4, 4.8e-4 at -5) so there is
            no low-energy break at any resolution; nor at the merged regimes, where the
            single turn is the two breaks run together rather than the lower one. Those
            gaps are the honest answer and nu_pk is what spans them.
    nu_pk   the high-energy reference -- the nuFnu maximum, and deliberately NOT
            nu_knee_hi. The maximum needs no fit and is defined at every regime; the upper
            knee is levered by a_mid (0.12-0.18 dex per 0.05 of it in slow cooling) and is
            undefined once the breaks merge. Where both exist they differ by up to 2x, so
            this is a change of quantity, not a relabelling.

  LEFT: both references as peak over time-integrated -- nu_pk across the whole sweep, and
  nu_bk on the four regimes that have a separated lower break. >1 means the feature has
  moved DOWN in frequency under integration. Two series and not one spliced curve: where
  both exist they are a factor 2-5 apart, so a join would read as a jump in the physics.

  RIGHT: nu_pk/nu_bk, one curve per kind -- the span between the two references, which is
  the separation of the spectrum's two features measured without ever fitting the upper
  one.
  '''
  by = {}
  for m in rows:
    if m['z'] == z:
      by.setdefault(m['kind'], {})[m['logr']] = m
  held = [m['logr'] for m in rows if m['z'] == z and m['knee_from'] == 'merged']

  fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.2))
  ax = axes[0]
  _held_mid_band(ax, held)
  ax.axhline(1., color='0.6', lw=0.9, zorder=0)
  pk, fl = by.get('peak', {}), by.get('fluence', {})
  lr = np.array(sorted(set(pk) & set(fl)), float)
  for i, (key, lab) in enumerate((('nu_pk', '$\\nu_{\\rm pk}$'),
                                  ('nu_bk', '$\\nu_{\\rm bk}$'))):
    v = np.array([pk[q][key]/fl[q][key]
                  if np.isfinite(pk[q][key]) and np.isfinite(fl[q][key])
                  and fl[q][key] > 0. else np.nan for q in lr], float)
    ax.plot(lr, v, color=_QCOL[i], ls='-', lw=1.3, marker=_QMK[i], ms=6, label=lab)
  ax.set_ylabel('peak / time-integrated')
  ax.legend(fontsize=8, loc='best', bbox_to_anchor=(0., 0., 1., 0.92))

  ax = axes[1]
  _held_mid_band(ax, held)
  for kind in KINDS:
    st = dict(_STY[kind]); st.update(color=_QCOL[1], marker=_QMK[1], ms=5.5)
    ax.semilogy(*_series(rows, z, kind, 'pk_over_bk'), **st)
  ax.set_ylabel('$\\nu_{\\rm pk}/\\nu_{\\rm bk}$')
  _kind_legend(ax, color=_QCOL[1], marker=_QMK[1], loc='best',
               bbox_to_anchor=(0., 0., 1., 0.92))

  for ax in axes:
    ax.set_xlabel(_CLABEL)
    ax.grid(alpha=0.25)
  fig.tight_layout()
  f = os.path.join(outdir, 'spectrum_shape_knee_ratios_%s.png' % _shell_name(z))
  fig.savefig(f, dpi=200, bbox_inches='tight'); plt.close(fig)
  return f


# Axis limits for the spectra panels of plot_spectra_and_ratios. They are context there,
# not the subject, so they do not need the article pair's floor: that one runs deep enough
# to reach nu_B, which is ~10 decades of y on the slow-cooling points and squashes every
# shape into the top tenth of the panel. Six decades shows the peak, both breaks and the
# cut-off on every regime of this sweep.
SPEC_YLO = 1e-6        # y floor, as a fraction of the peak
SPEC_XLO = 10**-6.5    # left x edge, in nu/nu_m,0. The grid starts at LOGNU_MIN = -6.7 and
                       # every point shares it, so anything below this is blank margin on
                       # the left of every curve; -6.5 trims that at the cost of the
                       # lowest 0.2 dex of the curves themselves.


SPEC_REGIMES = (-4., -2., 0., 2.)   # the four the spectra figures have always shown: deep
                                    # fast, fast, marginal, slow


def plot_knee_estimators(rows, results, outdir, z, kind='peak'):
  '''
  THE TWO ESTIMATORS ON THE SPECTRA THEMSELVES, so the disagreement can be seen rather
  than read off a ratio.

  One column per regime, spectra on top and the local index below them -- the knees are
  defined on the index curve, and a reader shown only the spectrum cannot see why the two
  estimators land where they do.

  ONE SPECTRUM KIND per figure (`kind`, default the peak one). Both at once put two curves
  and eight vertical lines in every panel, and the point of this figure is which FEATURE
  each estimator picks, not how the two kinds differ -- that is what the ratio figures are
  for. The kind is named in the panel and in the file name.

  Vertical lines: the half-slope point (solid) and the curvature maximum (dotted), purple
  for the lower break and orange for the upper. The horizontal ticks on the lower panels
  are the half-slope TARGETS, (a_lo+a_mid)/2 and (a_mid+a_hi)/2 -- where those levels cut
  the index curve IS the solid line above them, which is the whole construction in one
  picture.
  '''
  res = {float(r['log10ratio']): r for r in results}
  regs = [g for g in SPEC_REGIMES if g in res]
  fig, axes = plt.subplots(2, len(regs), figsize=(3.5*len(regs), 6.4), squeeze=False,
                           sharex='col')
  for j, g in enumerate(regs):
    r = res[g]
    xall = nu_over_num(r)
    sps = spectra_of(r)
    axS, axA = axes[0][j], axes[1][j]
    m = next((q for q in rows if q['z'] == z and q['logr'] == g
              and q['kind'] == kind), None)
    if m is not None:
      sp = sps[kind]
      ok = np.isfinite(sp) & (sp > 0.) & (xall > 0.)
      lx, ly, sl = sb.segment_slopes(xall[ok], sp[ok], sb.SLOPE_SMOOTH)
      axS.loglog(10.**lx, 10.**(ly - ly.max()), color='0.25', lw=1.2, zorder=4)
      axA.semilogx(10.**lx, sl, color='0.25', lw=1.2, zorder=4)
      # THE ASYMPTOTES THEMSELVES. Without them a reader judges the knee against the
      # visible bend of the curve, and on a break this broad (s1 ~ 0.8-1.2, the spectrum
      # sagging 0.25-0.37 dex under its own crossing) there is no bend to judge against:
      # the transition runs 1.2-1.9 dex and the curve only settles onto the mid segment
      # 2.6x above the knee. Drawn, the crossing is a place on the figure rather than a
      # number in a table. Free lines, and only where flattening is negligible.
      for a_, c_ in ((m['a_lo'], m['c_lo_f']), (m['a_mid'], m['c_mid_f']),
                     (m['a_hi'], m['c_hi_f'])):
        if np.isfinite(a_) and np.isfinite(c_):
          axS.plot(10.**lx, 10.**(a_*lx + c_ - ly.max()), color='0.55', lw=0.8,
                   ls=(0, (4, 3)), zorder=2)
      tags = (('1',),) if m['knee_from'] == 'merged' else (('lo', 'hi'),)
      for tg in tags[0]:
        i = {'lo': 0, 'hi': 1, '1': 2}[tg]
        for key, lsv in (('nu_knee_', '-'), ('nu_curv_', ':')):
          v = m[key + tg]
          if np.isfinite(v):
            for ax in (axS, axA):
              ax.axvline(v, color=_QCOL[i], ls=lsv, lw=1.3, alpha=0.85)
      # the levels the half-slope point is looking for
      am = m['a_mid'] if np.isfinite(m['a_mid']) else m['a_mid_route']
      al = m['a_lo'] if np.isfinite(m['a_lo']) else m['a_lo_h']
      ah = m['a_hi'] if np.isfinite(m['a_hi']) else m['a_hi_h']
      levels = ([(2, 0.5*(al + ah))] if m['knee_from'] == 'merged'
                else [(0, 0.5*(al + am)), (1, 0.5*(am + ah))])
      for i, t in levels:
        if np.isfinite(t):
          axA.axhline(t, color=_QCOL[i], lw=0.7, alpha=0.4, zorder=0)
    axS.set_ylim(1e-4, 3.)
    axS.annotate(f'$\\log_{{10}}\\mathcal{{C}} = {g:+.0f}$', (0.04, 0.06),
                 xycoords='axes fraction', fontsize=8)
    # THE RATIO HAS TO BE PRINTED. The x axis spans 12-18 decades here, so a factor 1.5
    # between the two estimators is a hair's breadth on the page -- the lines look
    # coincident at every regime while the table says they are not. The number is the
    # measurement; the lines only show which feature each one picked.
    if m is not None:
      cell = lambda k: ('--' if not np.isfinite(m[k]) else f'{m[k]:.2f}')
      lab = (f"merged  {cell('curv_off_1')}" if m['knee_from'] == 'merged'
             else f"lo {cell('curv_off_lo')}   hi {cell('curv_off_hi')}")
      axA.annotate('curv/half-slope\n' + lab, (0.03, 0.05), xycoords='axes fraction',
                   fontsize=6.5, color='0.3', va='bottom', family='monospace')
    axA.set_xlabel(NU_M_LABEL)
    axA.set_ylim(-0.6, 1.6)
    for ax in (axS, axA):
      ax.grid(alpha=0.2)
    if j:
      axS.set_yticklabels([]); axA.set_yticklabels([])
  axes[0][0].set_ylabel('$\\nu F_\\nu$ / peak')
  axes[1][0].set_ylabel('$\\mathrm{d}\\log\\nu F_\\nu/\\mathrm{d}\\log\\nu$')
  _shell_tag(axes[0][0], z)
  axes[0][0].annotate({'peak': 'peak spectrum',
                       'fluence': 'time-integrated spectrum'}[kind],
                      (0.04, 0.14), xycoords='axes fraction', fontsize=8, color='0.3')
  h = [plt.Line2D([], [], color=_QCOL[0], lw=1.3), plt.Line2D([], [], color=_QCOL[1], lw=1.3),
       plt.Line2D([], [], color=_QCOL[2], lw=1.3),
       plt.Line2D([], [], color='0.35', ls='-', lw=1.3),
       plt.Line2D([], [], color='0.35', ls=':', lw=1.3),
       plt.Line2D([], [], color='0.55', ls=(0, (4, 3)), lw=0.9)]
  fig.tight_layout()
  # below the axes, not inside one: every panel here is full to the edges, and the corner
  # the legend used covered the regime label of the last column
  fig.legend(h, ['lower break', 'upper break', 'merged break', 'half-slope point',
                 'curvature maximum', 'free asymptotes'], fontsize=7.5, ncol=6,
             loc='upper center', bbox_to_anchor=(0.5, 0.035), frameon=False)
  f = os.path.join(outdir,
                   'spectrum_shape_knee_estimators_%s_%s.png' % (_shell_name(z), kind))
  fig.savefig(f, dpi=200, bbox_inches='tight'); plt.close(fig)
  return f


def plot_spectra_and_ratios(rows, results, outdir, z, mode='eff'):
  '''
  The sweep's two spectra families and what this module measures off them, in one
  figure: peak spectra, time-integrated spectra, and how far the two reference
  frequencies move between them.

  Panels 1 and 2 are drawn by sweep_gammacm._draw_spectra_all, the same function
  plot_spectra_pair uses, so they keep every convention of the existing pair -- the
  normalisation, the y-floor, the x-clip, and the panel label inside the axes. Nothing
  about a spectrum is re-derived here.

  Panel 3 is the peak/time-integrated ratio of nu_pk and nu_bk, i.e. the two references
  read off exactly the curves in panels 1 and 2. Reading them together is the point: the
  eye sees the fluence family sitting to the LEFT of the peak family, and the third panel
  says by how much, separately for the SED peak and for the low-energy break -- which do
  not move together.

  THE COLOUR BAR IS ATTACHED TO THE SPECTRA PANELS ONLY. It is the sweep colour scale and
  says nothing about panel 3, whose x axis is that same parameter; hanging it off all
  three would put a redundant scale beside an axis already labelled with it.
  '''
  colors, sm = swp._sweep_colors(results)
  # this module's own peak row (bolometric, see _peak_index), not swp._peak_getter's nu_0
  # one: borrowing that would draw a different spectrum from the one panel 3 measured.
  ipk = {id(r): _peak_index(r) for r in results}
  peak_getter = lambda r: r['nuFnu'][ipk[id(r)], :]
  # THE AXES ARE PLACED BY HAND, and the two gaps set INDIVIDUALLY, because a gridspec
  # spaces every column alike while these two gaps have different jobs: the one between
  # the spectra holds panel 2's tick labels and its y label, the one before the ratio
  # panel holds the colour bar and nothing else. Uniform spacing crushed the y label onto
  # panel 1's spine and left a hole around the bar, at the same time. Explicit margins and
  # no tight_layout, so the coordinates below stay the ones set here.
  fig = plt.figure(figsize=(14.2, 4.2))
  L, R, B, T = 0.050, 0.962, 0.135, 0.950
  GAP_SPEC = 0.062      # between the spectra: panel 2's tick labels and its y label
  GAP_CBAR = 0.050      # before the ratio panel: the bar and its tick labels, and nothing
                        # else -- the ratio panel's own labels are on its far side
  w, h = (R - L - GAP_SPEC - GAP_CBAR)/3., T - B
  axs = [fig.add_axes([L, B, w, h]), fig.add_axes([L + w + GAP_SPEC, B, w, h])]
  ax_r = fig.add_axes([L + 2.*w + GAP_SPEC + GAP_CBAR, B, w, h])
  ok = False
  for ax, (lab, get_spec, sym) in zip(axs, (
      ('peak', peak_getter, '\\nu F_\\nu'),
      ('time-integrated', swp._fluence_getter, '\\nu \\mathcal{F}_\\nu'))):
    if swp._draw_spectra_all(ax, results, get_spec, mode,
                             f'spectrum_shape_spectra_ratios_{_shell_name(z)}.png',
                             sym=sym, ylo_min=SPEC_YLO, xlo=SPEC_XLO):
      ax.text(0.97, 0.97, lab, transform=ax.transAxes, ha='right', va='top', fontsize=11)
      ok = True
  if not ok:
    plt.close(fig)
    return None

  ax = ax_r
  _held_mid_band(ax, [m['logr'] for m in rows
                      if m['z'] == z and m['knee_from'] == 'merged'])
  by = {}
  for m in rows:
    if m['z'] == z:
      by.setdefault(m['kind'], {})[m['logr']] = m
  pk, fl = by.get('peak', {}), by.get('fluence', {})
  lr = np.array(sorted(set(pk) & set(fl)), float)
  for i, (key, lab) in enumerate((('nu_pk', '$\\nu_{\\rm pk}$'),
                                  ('nu_bk', '$\\nu_{\\rm bk}$'))):
    v = np.array([pk[q][key]/fl[q][key]
                  if np.isfinite(pk[q][key]) and np.isfinite(fl[q][key])
                  and fl[q][key] > 0. else np.nan for q in lr], float)
    ax.plot(lr, v, color=_QCOL[i], ls='-', lw=1.3, marker=_QMK[i], ms=6, label=lab)
  ax.set_xlabel(_CLABEL)
  # ticks and label on the RIGHT. This is the last panel, so nothing sits beyond it, and
  # on the left they would have to clear the colour bar -- which is what was forcing a
  # panel-width gap there. Moved, the three panels sit as close as their own labels allow.
  ax.yaxis.tick_right()
  ax.yaxis.set_label_position('right')
  ax.set_ylabel('peak / time-integrated')
  ax.grid(alpha=0.25)
  ax.legend(fontsize=8, loc='best', bbox_to_anchor=(0., 0., 1., 0.92))

  # the bar sits INSIDE the gap before the ratio panel, and its label goes ABOVE it:
  # rotated beside it, the label needs a panel gap of its own.
  cax = fig.add_axes([L + 2.*w + GAP_SPEC + 0.011, B, 0.009, h])
  cb = fig.colorbar(sm, cax=cax)
  cb.ax.set_title('log$_{10}\\mathcal{C}$', fontsize=9, pad=6)
  f = os.path.join(outdir, 'spectrum_shape_spectra_ratios_%s.png' % _shell_name(z))
  fig.savefig(f, dpi=200, bbox_inches='tight'); plt.close(fig)
  return f


# ---------------------------------------------------------------------------------------
def main(key=DEFAULT_KEY, method=DEFAULT_METHOD, shells=(Z_RS, Z_FS), nproc=None):
  '''
  Measure and report the peak and time-integrated spectra of a cached sweep, both shells.
  Runs the sweep first if a cache is missing. Everything is written into the RS directory
  so the two shells land in one table.
  '''
  rows, by_shell = [], {}
  for z in shells:
    outdir = method_outdir(method, key, z)
    os.makedirs(outdir, exist_ok=True)
    results = load_sweep(outdir)
    if results is None:
      results = run_sweep(key, LOG10RATIO_ARR, z=z, outdir=outdir, nproc=nproc,
                          method=method)
    print(f'{_shell_name(z)}: {len(results)} sweep points from {outdir}')
    rows += measure_sweep(results, z)
    by_shell[z] = results
  outdir = method_outdir(method, key, shells[0])
  _, _, pairs = build_tables(rows, outdir)
  for z in shells:                     # one figure per shell -- see the figures header
    plot_shape_vs_regime(rows, outdir, z)
    plot_fluence_vs_peak(pairs, outdir, z)
    plot_knee_ratios(rows, outdir, z)
    plot_knee_estimators(rows, by_shell[z], outdir, z)
    plot_spectra_and_ratios(rows, by_shell[z], outdir, z)
  trim_pngs(outdir)
  print(f'\nfigures -> {outdir}/spectrum_shape_*.png')
  return rows, pairs
