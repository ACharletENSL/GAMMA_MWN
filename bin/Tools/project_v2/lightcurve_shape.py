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
    LOG10RATIO_ARR)

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
SLOPE_NMIN = 7                     # points in the local-slope fit. The obs grid mixes a
SLOPE_HALF0 = 0.04                 # geometric ladder with 200 LINEAR samples over
                                   # bar{T} = 0.5..9 (TB_LIN), so a fixed +-dex window holds
                                   # ~11 points near the peak but only ~3.6 past x ~ 7. The
                                   # window starts at +-SLOPE_HALF0 dex and widens until it
                                   # holds SLOPE_NMIN points, which keeps the resolution where
                                   # the grid is dense without going NaN where it is not.
NU_COLORS = {1e-2: 'tab:purple', 1e-1: 'tab:orange', 1.: 'tab:green'}


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


def _slope_profile(x, y, nmin=SLOPE_NMIN, half0=SLOPE_HALF0):
  '''Local log-log index d ln(nuFnu)/d ln(bar{T}) at every grid point, from a
  straight-line fit over a log-x window widened until it holds nmin points.'''
  lx, ly = np.log10(x), np.log10(np.maximum(y, 1e-300))
  good = (y > 0.) & np.isfinite(x) & (x > 0.)
  s = np.full(len(x), np.nan)
  for i in np.where(good)[0]:
    h, m = half0, None
    for _ in range(8):
      m = good & (np.abs(lx - lx[i]) <= h)
      if m.sum() >= nmin:
        break
      h *= 1.6
    if m.sum() >= 4:
      s[i] = np.polyfit(lx[m], ly[m], 1)[0]
  return s


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


def shape_metrics(barT, lc, barT_f, barT_rf=None):
  '''
  Peak time and pulse shape of one lightcurve, all times in x = bar{T}/bar{T}_f.

  peak      x_pk        parabolic refinement in (log x, log F) about the grid maximum
  widths    x_hr,x_hd   half-maximum crossings, FWHM = x_hd - x_hr
            asym        (x_hd - x_pk)/(x_pk - x_hr): 1 = symmetric about the peak
            w10         width at a tenth of the peak
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
  '''
  x = barT/barT_f
  y = np.asarray(lc, float)
  out = {k: np.nan for k in ('x_pk', 'x_hr', 'x_hd', 'fwhm', 'asym', 'w10', 'x05', 'x50',
                             'x95', 'w90', 'flat90', 'x_br', 'F_br', 'a_rf', 'x_m2',
                             'x_m3', 'a_late')}
  out['a_rise'] = {L: np.nan for L in RISE_LEVELS}
  if not np.isfinite(y).any() or y.max() <= 0.:
    return out
  ipk = int(np.argmax(y)); Fpk = y[ipk]
  out['x_pk'] = float(x[ipk])
  if 0 < ipk < len(y) - 1 and y[ipk-1] > 0. and y[ipk+1] > 0.:
    c = np.polyfit(np.log(x[ipk-1:ipk+2]), np.log(y[ipk-1:ipk+2]), 2)
    if c[0] < 0.:
      out['x_pk'] = float(np.exp(-c[1]/(2.*c[0])))

  out['x_hr'] = _level_cross(x, y, .5*Fpk, ipk, 'r')
  out['x_hd'] = _level_cross(x, y, .5*Fpk, ipk, 'd')
  out['fwhm'] = out['x_hd'] - out['x_hr']
  out['asym'] = (out['x_hd'] - out['x_pk'])/(out['x_pk'] - out['x_hr'])
  x1r, x1d = _level_cross(x, y, .1*Fpk, ipk, 'r'), _level_cross(x, y, .1*Fpk, ipk, 'd')
  out['w10'] = x1d - x1r
  lo90, hi90 = (_level_cross(x, y, FLAT_FRAC*Fpk, ipk, s) for s in ('r', 'd'))
  out['flat90'] = (hi90 - lo90)/out['x_pk']

  cum = np.concatenate([[0.], np.cumsum(.5*(y[1:] + y[:-1])*np.diff(x))])
  if cum[-1] > 0.:
    q = np.interp([.05, .5, .95], cum/cum[-1], x)
    out['x05'], out['x50'], out['x95'] = (float(v) for v in q)
    out['w90'] = out['x95'] - out['x05']

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


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
_COLS = [('log10(gc/gm)', 'logr', '{:+.0f}'), ('nu/nu_m', 'nu_over_num', '{:.3g}'),
         ('x_pk', 'x_pk', '{:.3f}'), ('bar_T_pk', 'barT_pk', '{:.3f}'),
         ('x_1/2 rise', 'x_hr', '{:.3f}'), ('x_1/2 dec', 'x_hd', '{:.3f}'),
         ('FWHM', 'fwhm', '{:.3f}'), ('asym', 'asym', '{:.2f}'), ('W10', 'w10', '{:.2f}'),
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
         'FWHM/W10 are the half- and tenth-maximum widths, asym = (x_1/2 dec - x_pk)/'
         '(x_pk - x_1/2 rise); x_50 and W90 = x_95-x_05 are FLUENCE quantiles, so they '
         'include the tail.\nflat90 = span above 0.9 of the peak in units of x_pk; '
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
    outdir=None, nproc=None):
  '''
  Measure and report the lightcurve shape of a cached sweep. Runs the sweep first
  if its cache is missing (run_sweep, same (key, method, z) directory).
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
  print(f'\nFigures saved to {outdir}')
  return rows, rows_num
