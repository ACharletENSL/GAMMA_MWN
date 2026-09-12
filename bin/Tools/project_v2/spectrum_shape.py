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
from sweep_gammacm import (DEFAULT_KEY, DEFAULT_METHOD, Z_SHELL, LOG10RATIO_ARR,
    load_sweep, method_outdir, run_sweep, nu_over_num, compute_fluence_spectrum,
    detect_rise_peak_tail, write_table_stamp, trim_pngs, NU_M_LABEL)
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


def _peak_index(r):
  '''
  The observer-time row whose spectrum is "the peak spectrum". Taken from
  detect_rise_peak_tail exactly as sweep_gammacm._peak_getter takes it, so the spectrum
  measured here is the one plot_spectra_pair draws -- the lightcurve peak at nub = NU_REF,
  not the row whose own spectrum peaks highest.
  '''
  return detect_rise_peak_tail(r['Tb'], r['nub'], r['nuFnu'])[3].get('i_peak')


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
             sag_lo=np.nan, sag_hi=np.nan, s_clean=False, s_stable=False,
             nu_knee_lo=np.nan, nu_knee_hi=np.nan,
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
    out['lo_conv'] = bool(f['lo_converged'])
    # the sag at those free crossings, which is what free_slopes' s1/s2 are read off
    out['s1_def'], out['s2_def'] = f['s1'], f['s2']
    out['s_clean'], out['s_stable'] = bool(f['s_clean']), bool(f['s_stable'])
    for tag, sv in (('lo', f['s1']), ('hi', f['s2'])):
      out['sag_' + tag] = float(np.log10(2.)/sv) if np.isfinite(sv) and sv > 0. else np.nan

    # ... and where the spectrum actually turns over. The mid slope is the measured one
    # where there is a plateau to measure it on; where there is not (MC, whose mid line is
    # the tangent) the route's held value is used instead and knee_from says so, because a
    # target built on a held slope is not a free measurement of the turnover.
    a_mid_k, out['knee_from'] = f['a_mid'], 'free'
    if not np.isfinite(a_mid_k):
      a_mid_k, out['knee_from'] = b['a_mid'], 'route'
    a_lo_k = f['a_lo'] if np.isfinite(f['a_lo']) else b['a_lo']
    a_hi_k = f['a_hi'] if np.isfinite(f['a_hi']) else b['a_hi']
    out.update(turnovers(x, sp, b['nuM'], b['sigma'], a_lo_k, a_mid_k, a_hi_k))
    # AGAINST BOTH CROSSINGS, and the second one is not optional -- see the section header.
    # b_* is the route's, whose low line is HELD at 4/3; b_*_free is the crossing of the
    # lines free_slopes actually measured. Where the spectrum's low index is not 4/3 the
    # held line is tilted and its crossing displaced, and knee/b inherits that displacement.
    for tag in ('lo', 'hi'):
      kn = out['nu_knee_' + tag]
      for pre, bb in (('knee_off_', out['b_' + tag]), ('kneef_off_', out['b_%s_free' % tag])):
        out[pre + tag] = (float(kn/bb) if np.isfinite(kn) and np.isfinite(bb) and bb > 0.
                          else np.nan)

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


def measure_spectrum(x, sp, psyn, **kw):
  '''Every measurement this module makes on one spectrum: width block + segment block.'''
  m = peak_and_width(x, sp)
  m.update(segments_and_breaks(x, sp, psyn, **kw))
  m['bhi_over_xpk'] = (m['b_hi']/m['x_pk']
                       if np.isfinite(m['b_hi']) and m['x_pk'] > 0. else np.nan)
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
    m.update(kind=kind, z=z, logr=float(r['log10ratio']), psyn=float(env.psyn),
             C=float(env.gma_c/env.gma_m), nuM_env=float((env.gma_max/env.gma_m)**2))
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
_RATIO_KEYS = ('x_pk', 'b_lo', 'b_hi', 'nu_knee_lo', 'nu_knee_hi', 'sep', 'nuM',
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
               knee_held=bool(pk['knee_from'] == 'route' or fl['knee_from'] == 'route'))
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
RATIO_CSV = 'spectrum_shape_fluence_vs_peak.csv'

_COLS = [('shell', 'shell', '{:s}'), ('log10(C)', 'logr', '{:+.0f}'),
         ('kind', 'kind', '{:s}'), ('class', 'cls', '{:s}'),
         ('x_pk', 'x_pk', '{:.4g}'), ('top_dex', 'top_dex', '{:.2f}'),
         ('nu_1/2 lo', 'nu_lo_half', '{:.4g}'), ('nu_1/2 hi', 'nu_hi_half', '{:.4g}'),
         ('W_1/2', 'W_half', '{:.4g}'), ('log W_1/2', 'logW_half', '{:.3f}'),
         ('W_lo', 'Wlo_half', '{:.3g}'), ('W_hi', 'Whi_half', '{:.3g}'),
         ('asym', 'asym_half', '{:.3f}'),
         ('W_1/10', 'W_tenth', '{:.4g}'), ('log W_1/10', 'logW_tenth', '{:.3f}'),
         ('b_lo', 'b_lo', '{:.4g}'), ('b_hi', 'b_hi', '{:.4g}'),
         ('b_hi/b_lo', 'sep', '{:.4g}'), ('b_hi/x_pk', 'bhi_over_xpk', '{:.3f}'),
         ('knee_lo', 'nu_knee_lo', '{:.4g}'), ('knee_hi', 'nu_knee_hi', '{:.4g}'),
         ('knee/b lo', 'knee_off_lo', '{:.3f}'), ('knee/b hi', 'knee_off_hi', '{:.3f}'),
         ('knee/bf lo', 'kneef_off_lo', '{:.3f}'),
         ('knee/bf hi', 'kneef_off_hi', '{:.3f}'),
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
          ('x_pk flu/pk', 'R_x_pk', '{:.3f}'),
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
# Shells as COLOUR (RS red, FS blue -- plotting_functions.COL_RS/COL_FS, the Charlet et al.
# convention), kinds as LINESTYLE (fluence solid, peak dashed, as the existing
# fluence_vs_peak_sweep overlay draws them). No titles: the y label carries the quantity.
_STY = {'peak': dict(ls='--', marker='o', ms=4, mfc='none'),
        'fluence': dict(ls='-', marker='o', ms=4)}
_CLABEL = '$\\log_{10}\\mathcal{C}$'


def _held_mid_band(ax, logrs, half=0.35):
  """
  Shade the sweep points whose knee target needed a HELD mid slope. Those are the MC
  points, where no mid plateau exists to measure one on: the turnover there is not a free
  measurement, and the band says so without spending a colour, a marker or a fill on it.
  """
  # deliberately unlabelled: the callers either build their legend from explicit handles
  # or explain the band in the figure caption, and an axvspan label picked up by a bare
  # ax.legend() lands a full-width swatch in the middle of a crowded panel
  for lr in sorted(set(logrs)):
    ax.axvspan(lr - half, lr + half, color='0.85', alpha=0.55, lw=0, zorder=0)


def _series(rows, z, kind, key):
  s = sorted([m for m in rows if m['z'] == z and m['kind'] == kind],
             key=lambda m: m['logr'])
  return (np.array([m['logr'] for m in s], float),
          np.array([m[key] for m in s], float))


def _kind_legend(ax, shells, **kw):
  h = [plt.Line2D([], [], color='k', **_STY[k]) for k in KINDS]
  lab = ['peak', 'time-integrated']
  for z in shells:
    h.append(plt.Line2D([], [], color=COL_RS if z == Z_RS else COL_FS, lw=2))
    lab.append(_shell_name(z))
  ax.legend(h, lab, fontsize=7, **kw)


def plot_shape_vs_regime(rows, outdir, shells=(Z_RS, Z_FS)):
  """
  The three answers against the cooling regime: the free slopes, the break positions and
  the half-maximum width, peak against time-integrated.

  The low-slope points that did NOT converge inside the band are drawn hollow and
  unjoined. They are bounds, not measurements (see lo_conv), and joining them to the
  converged ones draws a slope change that is a band limit.
  """
  fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.2))
  p = float(np.median([m['psyn'] for m in rows]))

  # --- free slopes
  ax = axes[0]
  for a, lab in ((4./3., '$4/3$'), (0.5, '$1/2$'), ((3.-p)/2., '$(3-p)/2$'),
                 (1. - p/2., '$1-p/2$')):
    ax.axhline(a, color='0.75', lw=0.8, zorder=0)
    # a white box behind the guide label: a_hi sits ON its own guide across the whole
    # sweep, so an unboxed label there is drawn over by the curve it labels
    ax.annotate(lab, (0.995, a), xycoords=('axes fraction', 'data'), fontsize=6.5,
                color='0.45', va='bottom', ha='right', zorder=6,
                bbox=dict(fc='w', ec='none', alpha=0.75, pad=0.6))
  for z in shells:
    c = COL_RS if z == Z_RS else COL_FS
    for kind in KINDS:
      st = _STY[kind]
      for key in ('a_mid', 'a_hi'):
        lr, v = _series(rows, z, kind, key)
        ax.plot(lr, v, color=c, **st)
      lr, v = _series(rows, z, kind, 'a_lo')
      _, cv = _series(rows, z, kind, 'lo_conv')
      cv = cv.astype(bool)
      ax.plot(np.where(cv, lr, np.nan), np.where(cv, v, np.nan), color=c, **st)
      ax.plot(lr[~cv], v[~cv], color=c, ls='none', marker=st['marker'], ms=st['ms'],
              mfc='none', alpha=0.55)
  ax.set_ylabel('free slope  $\\mathrm{d}\\log\\nu F_\\nu/\\mathrm{d}\\log\\nu$')
  _kind_legend(ax, shells, loc='lower left', framealpha=0.9)
  ax.annotate('hollow, unjoined: asymptote not reached in band',
              (0.5, 0.73), xycoords='axes fraction', fontsize=6.5, color='0.35',
              ha='center', va='center')

  # --- the TURNOVERS and the nuFnu maximum. The knees, not the crossings: b_lo/b_hi are
  # where the extrapolated asymptotes meet, which is a point the spectrum never passes
  # through, and at the lower break the route's held 4/3 line displaces it (see the
  # turnover section header). Both crossings stay in the table.
  ax = axes[1]
  _held_mid_band(ax, [m['logr'] for m in rows if m['knee_from'] == 'route'])
  bkeys = (('nu_knee_lo', 'v', '$\\nu_{\\rm knee,lo}$'),
           ('nu_knee_hi', '^', '$\\nu_{\\rm knee,hi}$'),
           ('x_pk', 's', '$\\nu_{\\rm pk}$'))
  for z in shells:
    c = COL_RS if z == Z_RS else COL_FS
    for kind in KINDS:
      for key, mk, _ in bkeys:
        lr, v = _series(rows, z, kind, key)
        st = dict(_STY[kind]); st.update(marker=mk)
        if key == 'x_pk':
          st.update(alpha=0.5, ms=3.5)
        ax.semilogy(lr, v, color=c, **st)
  ax.set_ylabel('$\\nu/\\nu_{\\mathrm{m},0}$')
  h = [plt.Line2D([], [], color='0.35', marker=mk, ms=4, ls='-') for _, mk, _ in bkeys]
  lb = [lab for _, _, lab in bkeys]
  h.append(plt.Rectangle((0, 0), 1, 1, color='0.85'))
  lb.append('mid slope held')
  ax.legend(h, lb, fontsize=7, loc='upper left', ncol=2)

  # --- the width
  ax = axes[2]
  for z in shells:
    c = COL_RS if z == Z_RS else COL_FS
    for kind in KINDS:
      lr, v = _series(rows, z, kind, 'logW_half')
      ax.plot(lr, v, color=c, **_STY[kind])
      lr, v = _series(rows, z, kind, 'logW_tenth')
      st = dict(_STY[kind]); st.update(alpha=0.4, ms=3)
      ax.plot(lr, v, color=c, **st)
  ax.set_ylabel('$\\log_{10}$ peak width  $\\nu_+/\\nu_-$')
  ax.annotate('half maximum', (0.03, 0.30), xycoords='axes fraction', fontsize=7,
              color='0.35')
  ax.annotate('tenth maximum (faint)', (0.03, 0.90), xycoords='axes fraction',
              fontsize=7, color='0.55')
  for ax in axes:
    ax.set_xlabel(_CLABEL)
    ax.grid(alpha=0.25)
  fig.tight_layout()
  f = os.path.join(outdir, 'spectrum_shape_vs_regime.png')
  fig.savefig(f, dpi=200, bbox_inches='tight'); plt.close(fig)
  return f


def plot_fluence_vs_peak(pairs, outdir, shells=(Z_RS, Z_FS)):
  """
  The comparison itself: how far each quantity moves from peak to time-integrated.

  The two low-energy slope differences are masked on their convergence flags -- an
  unconverged pair differs by up to 0.25, which is the gap between two band limits and
  not a softening. Those points are drawn hollow and unjoined, as in the table.
  """
  fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.2))
  def ser(z, key):
    s_ = sorted([m for m in pairs if m['z'] == z], key=lambda m: m['logr'])
    return (np.array([m['logr'] for m in s_], float),
            np.array([m[key] for m in s_], float),
            np.array([m['class_flip'] for m in s_], bool))
  panels = [(('R_W_half', 'W_{1/2}', None), ('R_W_tenth', 'W_{1/10}', None),
             ('R_Wlo_half', 'W_{\\rm lo}', None), ('R_Whi_half', 'W_{\\rm hi}', None)),
            (('R_x_pk', '\\nu_{\\rm pk}', None),
             ('R_nu_knee_lo', '\\nu_{\\rm knee,lo}', None),
             ('R_nu_knee_hi', '\\nu_{\\rm knee,hi}', None),
             ('R_nuM', '\\nu_{\\rm M}', None)),
            (('d_a_lo', 'a_{\\rm lo}', 'lo_conv'), ('d_a_mid', 'a_{\\rm mid}', None),
             ('d_a_hi', 'a_{\\rm hi}', None), ('d_a_inf', 'a_\\infty', 'inf_conv'))]
  mk = ('o', 's', '^', 'v')
  for ax, keys in zip(axes, panels):
    if any(k.startswith('R_nu_knee') for k, _, _ in keys):
      _held_mid_band(ax, [m['logr'] for m in pairs if m['knee_held']])
    ax.axhline(1. if keys[0][0].startswith('R_') else 0., color='0.6', lw=0.9, zorder=0)
    for z in shells:
      c = COL_RS if z == Z_RS else COL_FS
      for (k, lab, gate), m_ in zip(keys, mk):
        lr, v, fl = ser(z, k)
        lb = f'${lab}$' if z == shells[0] else None
        if gate is None:
          ax.plot(lr, v, color=c, ls='-', lw=1.0, marker=m_, ms=4, label=lb)
        else:
          cv = ser(z, gate)[1].astype(bool)
          ax.plot(np.where(cv, lr, np.nan), np.where(cv, v, np.nan), color=c, ls='-',
                  lw=1.0, marker=m_, ms=4, label=lb)
          ax.plot(lr[~cv], v[~cv], color=c, ls='none', marker=m_, ms=4, mfc='none',
                  alpha=0.55)
        if fl.any():           # a class flip: the segment columns are not like for like
          ax.plot(lr[fl], v[fl], ls='none', marker='x', ms=9, color='k', zorder=5)
    ax.set_xlabel(_CLABEL)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2, loc='best')
  axes[0].set_ylabel('width, time-integrated / peak')
  axes[1].set_ylabel('frequency, time-integrated / peak')
  axes[1].set_yscale('log')
  axes[2].set_ylabel('slope, time-integrated $-$ peak')
  fig.tight_layout()
  fig.text(0.5, -0.005, '$\\times$ = shape class differs;   hollow, unjoined = asymptote '
           'not reached inside the band;   grey band = no mid plateau, knee target used a '
           'held mid slope', fontsize=7.5, color='0.3', ha='center')
  f = os.path.join(outdir, 'spectrum_shape_fluence_vs_peak.png')
  fig.savefig(f, dpi=200, bbox_inches='tight'); plt.close(fig)
  return f


# ---------------------------------------------------------------------------------------
def main(key=DEFAULT_KEY, method=DEFAULT_METHOD, shells=(Z_RS, Z_FS), nproc=None):
  '''
  Measure and report the peak and time-integrated spectra of a cached sweep, both shells.
  Runs the sweep first if a cache is missing. Everything is written into the RS directory
  so the two shells land in one table.
  '''
  rows = []
  for z in shells:
    outdir = method_outdir(method, key, z)
    os.makedirs(outdir, exist_ok=True)
    results = load_sweep(outdir)
    if results is None:
      results = run_sweep(key, LOG10RATIO_ARR, z=z, outdir=outdir, nproc=nproc,
                          method=method)
    print(f'{_shell_name(z)}: {len(results)} sweep points from {outdir}')
    rows += measure_sweep(results, z)
  outdir = method_outdir(method, key, shells[0])
  _, _, pairs = build_tables(rows, outdir)
  plot_shape_vs_regime(rows, outdir, shells=shells)
  plot_fluence_vs_peak(pairs, outdir, shells=shells)
  trim_pngs(outdir)
  print(f'\nfigures -> {outdir}/spectrum_shape_*.png')
  return rows, pairs
