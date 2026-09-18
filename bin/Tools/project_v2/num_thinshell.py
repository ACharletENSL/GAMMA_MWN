# -*- coding: utf-8 -*-
# @Author: acharlet

'''
nu_m(t) MEASURED ON THE SPECTRA AGAINST THE C25 THIN-SHELL PEAK MODEL
(peak_modeling.C25_peak_model). One figure per shell, standing on its own: the sweep's
nu_m/nu_m,0 tracks, the analytic curve they are meant to reproduce, and their ratio.

WHY IT IS ITS OWN MODULE. The comparison used to be a column of
sweep_gammacm.build_break_evolution_table ('nu_m/C25') and, before that, a third panel on
break_evolution -- which is the break evolution's business only by accident. nu_m is not a
cooling quantity: the model carries no cooling break at all, its nu_pk IS nu_m, and what
the comparison tests is the thin-shell hydro + EATS integration, not the spectral shape.
Split out, it can carry its own axes, its own honest coverage statement and its own table
without loading the break figures with a reference they do not otherwise use.

WHAT IS COMPARED. The sweep's frequency axis is nu/nu_m,0 = nu_over_num (env.nu0 units)
and its time axis is bar{T} = Tb - 1, so the measured track and the model's nu_pk are
already in the same units and on the same grid: Tb IS the model's tilde{T}. The model's
parameters (a_u, tau = t_on/t_off) come from the run's own env and are invariant under the
alpha rescaling the sweep uses -- t{z} and toff scale together -- so ONE model curve serves
all nine sweep points. Any spread between the curves is therefore spread in the
MEASUREMENT, not in the model.

THE FORWARD-SHOCK CONVERSION, WHICH IS A TRAP. The sweep grids are RS-NORMALISED FOR BOTH
SHELLS (sweep_gammacm._compute_point builds nuobs/env.nu0 and (Tobs-env.Ts)/env.T0 from the
reverse shock's env whatever z is, so the two shells' points can be summed). The model, on
the other hand, is written in each front's OWN units. Feeding a z=1 track's bar{T} straight
into C25_peak_model(..., reverse=False) therefore compares two different normalisations and
misses a factor ratio_nu = 15.09 on this run -- which is exactly what
sweep_gammacm.c25_num_curve does, and why the z=1 break table reports nu_m/C25 ~ 0.05.
Here the conversion is the one flux_modeling.flux_model_C25_normed already uses:

    tilde{T}_FS = 1 + bar{T} * ratio_T           (time)
    nu_pk,FS|RS = nu_pk,FS / ratio_nu            (frequency)

with (ratio_T, ratio_nu, ratio_F) = peak_modeling.ratios_RSvFS_from_au(a_u). With it the FS
ratios land at 0.95-1.12, i.e. the same quality of agreement as the RS -- the conversion is
its own check.

TWO MEASUREMENT ROUTES, DRAWN TOGETHER, BECAUSE NEITHER ALONE IS THE ERROR BAR.

  segment   spectral_breaks.track_segment_route, via segment_route's per-point cache --
            THE PAPER ROUTE. nu_m is the break the SHAPE CLASS says is nu_m: the upper
            break (mid x hi crossing) in FC, FC* and VFC, the lower one (lo x mid) in SC
            and VSC. MC bins are left out entirely: an MC spectrum shows no mid segment,
            so neither break is named, and naming one would be the guess this route exists
            to avoid. That costs coverage exactly where the crossing is -- log10(C) = -1
            and 0 keep 402 and 652 of 2067 bins -- and the gap in the figure is the honest
            statement of it.
  gs02      sweep_gammacm.track_breaks_gs02, the Granot & Sari template fit, which is what
            the shipped break table's nu_m/C25 column is built on. It names every bin by
            CONTINUITY (`fast`: monotone, one SC -> FC switch, held through ties), so it
            covers the crossing where the segment route declines, at the price of a name
            that rests on the neighbouring bins rather than on the spectrum in hand.

Agreement between the two is the measurement's systematic; the model comparison is only
meaningful to that accuracy.

COST AND CACHING. The segment tracks are segment_route's and are reloaded, never refitted
(~0.8 s a bin cold). The GS02 tracks are ~30 s a point and are cached HERE, one npz per
point, stamped on the sweep point and on sweep_gammacm.py's mtime -- the same contract
segment_route's track cache uses, so an edit to the tracker invalidates them rather than
being silently reused. A warm run is a pure replot.

Reads the cached sweep only; no spectrum is recomputed and no sweep is run.
'''

import os
import numpy as np
import matplotlib.pyplot as plt

from environment import figdir, FIDUCIAL_KEY
import cell_pool
import sweep_gammacm as swp
import segment_route as sr
from peak_modeling import C25_peak_model, ratios_RSvFS_from_au, compute_Tf
from working_peaks import get_model_params


KEY = FIDUCIAL_KEY                 # the fiducial run; the article's figures are hi-res,
                                   # this is where the comparison is validated cheaply
METHOD = swp.DEFAULT_METHOD        # 'data_rarcut', imported rather than hardcoded so the
                                   # module follows the sweep's reference computation
Z_RS, Z_FS = 4, 1
OUTDIR_NAME = 'num_thinshell'
OUTDIR = figdir(OUTDIR_NAME)       # figdir puts it under the run's own folder; this is the
                                   # FIDUCIAL's -- another key resolves its own (_outdir)
NPROC = None                       # None -> cell_pool.resolve_nproc (GAMMACM_NPROC, else
                                   # the allocation minus one core)

ROUTES = ('segment', 'gs02')
ROUTE_STYLE = {'segment': dict(lw=1.6, ls='-'),      # the paper route, drawn solid
               'gs02': dict(lw=1.0, ls=':')}         # the template fit, as the cross-check
ROUTE_LABEL = {'segment': 'segment route', 'gs02': 'GS02 fit'}

# the shape classes whose named break is nu_m, and which break that is. MC is absent on
# purpose: it shows no mid segment, so neither of its breaks is nu_m (see the banner).
NUM_FROM_CLASS = {'FC': 'b_hi', 'FC*': 'b_hi', 'VFC': 'b_hi',
                  'SC': 'b_lo', 'VSC': 'b_lo'}

NUM_LABEL = '$\\nu_\\mathrm{m}/\\nu_{\\mathrm{m},\\!0}$'
RATIO_LABEL = '$\\nu_\\mathrm{m}/\\nu_\\mathrm{m}^\\mathrm{C25}$'
CBAR_LABEL = 'log$_{10}\\mathcal{C}$'
RATIO_YLIM = (.3, 3.)                                 # the ratio panel's range, the SAME for
RATIO_TICKS = (.3, .5, .7, 1., 1.5, 2., 3.)           # both shells so the two are comparable
                                                      # at a glance; .3 is set by the one
                                                      # GS02 outlier (log10(C)=0 on the RS)


def _outdir(key, outdir=None):
  '''Where this run's comparison lives. An explicit outdir wins; otherwise the run's own
  folder, so a non-fiducial key cannot silently write into the fiducial's (OUTDIR is
  evaluated at import with key=None).'''
  return outdir if outdir is not None else figdir(OUTDIR_NAME, key)


# ---------------------------------------------------------------------------
# the model curve
# ---------------------------------------------------------------------------
def c25_num(key, z, Tb):
  '''
  The C25 thin-shell model's nu_m on a sweep's own time grid, IN THE SWEEP'S UNITS.

  Tb is the sweep's Tb (1 at collision), which for the reverse shock IS the model's
  tilde{T}. For the FORWARD shock the sweep grid is still RS-normalised (see the banner),
  so both axes are converted with ratios_RSvFS_from_au before and after the model call --
  time by ratio_T, frequency by 1/ratio_nu. Returns an array on Tb.
  '''
  au, tau, reverse = get_model_params(key, z)
  Tb = np.asarray(Tb, float)
  if reverse:
    num, _ = C25_peak_model(Tb, au, tau, True)
    return np.asarray(num, float)
  ratio_T, ratio_nu, _ = ratios_RSvFS_from_au(au)
  num, _ = C25_peak_model(1. + (Tb - 1.)*ratio_T, au, tau, False)
  return np.asarray(num, float)/ratio_nu


def c25_barT_f(key, z):
  '''
  The model's own shell-crossing time, as bar{T} on the SWEEP's axis -- the thing to put
  beside exit_onset_barT(key, z). compute_Tf returns tilde{T}_f in the front's own units,
  so the FS value is brought back to the RS axis by the same ratio_T as the curve.
  fitparams_from_au is imported lazily: a missing fit table then costs this module its
  crossing-time line rather than its import.
  '''
  from peak_modeling import fitparams_from_au
  au, tau, reverse = get_model_params(key, z)
  popt_lfac = fitparams_from_au(au, reverse)[0]
  Tf, _ = compute_Tf(tau, au, popt_lfac, reverse)
  if reverse:
    return float(Tf - 1.)
  ratio_T, _, _ = ratios_RSvFS_from_au(au)
  return float((Tf - 1.)/ratio_T)


# ---------------------------------------------------------------------------
# the measured tracks
# ---------------------------------------------------------------------------
def _seg_num(tk):
  '''
  nu_m and its validity mask out of one segment-route track. The shape class picks which
  break is nu_m (NUM_FROM_CLASS); MC and unclassified bins have none. `br_ok` already
  carries the route's own in-band test, so nothing here re-derives it.
  '''
  reg = np.asarray([('' if q is None else str(q)) for q in tk['regime']])
  num = np.full(len(reg), np.nan)
  for cls, field in NUM_FROM_CLASS.items():
    m = (reg == cls)
    if m.any():
      num[m] = np.asarray(tk[field], float)[m]
  ok = np.asarray(tk['br_ok'], bool) & np.isfinite(num) & (num > 0.)
  return num, ok


def _gs02_cache_path(key, method, z, logr, outdir=OUTDIR):
  return os.path.join(outdir, 'cache',
                      f'gs02num_{key}_{method}_z={z}_logr={logr:+.1f}.npz')


def _gs02_stamp(key, method, z, logr):
  '''What a cached GS02 track was computed from: the sweep point and sweep_gammacm.py,
  which is where track_breaks_gs02 lives. Same contract as segment_route._track_stamp.'''
  st = os.stat(sr._sweep_point_path(key, method, z, logr))
  return np.array([st.st_size, st.st_mtime_ns,
                   os.stat(swp.__file__).st_mtime_ns], dtype=np.int64)


def _gs02_cached(key, method, z, logr, outdir):
  '''True when a GS02 track for this point is on disk AND was fitted from what is there
  now -- the same test _gs02_point makes, factored out so the pool can be sized on it.'''
  path = _gs02_cache_path(key, method, z, logr, outdir)
  if not os.path.isfile(path):
    return False
  try:
    with np.load(path, allow_pickle=False) as f:
      return '_stamp' in f and np.array_equal(f['_stamp'],
                                              _gs02_stamp(key, method, z, logr))
  except (OSError, ValueError, EOFError):
    return False


def _gs02_point(args):
  '''One point's GS02 nu_m track, in a worker: the cache, or ~30 s of per-bin fits.
  bswap is passed IN rather than derived here -- it is a hydro constant of the shell, so
  deriving it per job would have every worker repeat the same run-level read.'''
  key, method, z, logr, use_cache, outdir, bswap = args
  path = _gs02_cache_path(key, method, z, logr, outdir)
  stamp = _gs02_stamp(key, method, z, logr)
  if use_cache and os.path.isfile(path):
    try:
      with np.load(path, allow_pickle=False) as f:
        if '_stamp' in f and np.array_equal(f['_stamp'], stamp):
          return dict(logr=logr, z=z, barT=f['barT'], num=f['num'], ok=f['ok'],
                      cached=True)
    except (OSError, ValueError, EOFError):
      pass                                   # a truncated file is a miss, not a crash
  res = swp.load_sweep(swp.method_outdir(method, key, z))
  r = [q for q in res if abs(q['log10ratio'] - logr) < 1e-9][0]
  tr = swp.track_breaks_gs02(r, barT_swap_max=bswap)
  out = dict(barT=tr['barT'], num=tr['nu_m'],
             ok=tr.get('valid_m', tr['valid']) & np.isfinite(tr['nu_m']))
  os.makedirs(os.path.dirname(path), exist_ok=True)
  np.savez(path, _stamp=stamp, **out)
  out.update(logr=logr, z=z, cached=False)
  return out


def load_side(z=Z_RS, key=KEY, method=METHOD, route='segment', nproc=NPROC,
    use_cache=True, outdir=None):
  '''
  One shell's measured nu_m tracks, one entry per sweep point, ascending in log10(C) --
  the same order load_sweep returns, which the x=logr reductions rely on.
  Each entry: logr, z, barT, num (in nu_m,0 units), ok (the route's own validity mask).
  route: 'segment' (the paper route, reloaded from segment_route's cache) or 'gs02'
  (the template fit, cached here).
  '''
  outdir = _outdir(key, outdir)
  res = swp.load_sweep(swp.method_outdir(method, key, z))
  if not res:
    raise FileNotFoundError(f'no cached sweep for z={z} -- run sweep_gammacm.main first')
  logrs = [float(r['log10ratio']) for r in res]
  if route == 'segment':
    sides = sr.load_side(z=z, key=key, method=method, nproc=nproc, use_cache=use_cache)
    out = []
    for tk in sides:
      num, ok = _seg_num(tk)
      out.append(dict(logr=float(tk['logr']), z=z, barT=np.asarray(tk['barT'], float),
                      num=num, ok=ok))
    return sorted(out, key=lambda d: d['logr'])
  if route != 'gs02':
    raise ValueError(f'unknown route {route!r} (expected one of {ROUTES})')
  # the SC -> FC swap may only be detected while the shell still emits on-axis, exactly as
  # sweep_gammacm.main gates it, so the tracks here are the ones the break figures draw
  off = swp.rarefaction_off_barT(key, z=z)
  bswap = off[1] if off else swp.exit_onset_barT(key, z=z)
  jobs = [(key, method, z, lr, use_cache, outdir, bswap) for lr in logrs]
  # the pool is sized on what actually has to be FITTED, so the stamp is checked here and
  # not just the file's existence: nine stale caches would otherwise look like nine hits,
  # size the pool at one worker, and refit them one after another
  todo = [j for j in jobs if not (use_cache and _gs02_cached(*j[:4], outdir))]
  np_ = cell_pool.resolve_nproc(nproc, cap=len(todo)) if todo else 1
  if np_ > 1:
    with cell_pool.pool_context().Pool(np_) as pool:
      tracks = pool.map(_gs02_point, jobs)
  else:
    tracks = [_gs02_point(j) for j in jobs]
  nc = sum(1 for t in tracks if t.pop('cached', False))
  print(f'  z={z} gs02: {nc}/{len(tracks)} points from the track cache', flush=True)
  return sorted(tracks, key=lambda d: d['logr'])


# ---------------------------------------------------------------------------
# the figure
# ---------------------------------------------------------------------------
def plot_num_model(key=KEY, method=METHOD, z=Z_RS, routes=ROUTES, outdir=None,
    nproc=NPROC, use_cache=True, sides=None):
  '''
  THE FIGURE. Two panels sharing bar{T}/bar{T}_f, one horizontal colour bar underneath --
  the break_panels layout, for the same reason: the lower panel is the upper one divided,
  so the two have to share an abscissa to the pixel.

    top     nu_m/nu_m,0 measured, one colour per log10(C), with the C25 curve in black.
            The model is ONE curve for the whole sweep (see the banner), so the colours
            spreading around a single black line is the whole content of the panel.
    bottom  the ratio to that curve, on a log axis so a factor 1/x and x are the same
            distance from 1.

  Both panels are drawn only where the route calls the bin a measurement; the curves are
  blanked (swp._gap) rather than compressed, so a gap is a gap and not a straight line
  drawn across one. The crossing line and the rarefaction band are the usual hydro marks.

  routes: which measurements to draw, as line styles (ROUTE_STYLE); the colour stays the
  sweep point's in both, so a disagreement between routes reads as two lines of one colour.
  sides: pre-loaded {route: load_side(...)}, to draw without reloading.
  '''
  outdir = _outdir(key, outdir)
  results = swp.load_sweep(swp.method_outdir(method, key, z))
  if not results:
    raise FileNotFoundError(f'no cached sweep for z={z}')
  barT_f = swp.exit_onset_barT(key, z=z)
  barT_off = swp.rarefaction_off_barT(key, z=z)
  Tb = results[0]['Tb']
  barT = Tb - 1.
  model = c25_num(key, z, Tb)
  colors, sm = swp._sweep_colors(results)
  cmap = {float(r['log10ratio']): c for r, c in zip(results, colors)}

  if sides is None:
    sides = {rt: load_side(z=z, key=key, method=method, route=rt, nproc=nproc,
                           use_cache=use_cache, outdir=outdir) for rt in routes}

  # left is wider than the break panels' .105: the ratio label carries a superscript and
  # the ratio axis's ticks are spelled out (RATIO_TICKS), so the label lands off the canvas
  # at that margin and the figure comes back with 'nu_m/nu_m' on it
  fig, axes = plt.subplots(2, 1, figsize=(7.5, 8.2), sharex=True,
                           gridspec_kw=dict(hspace=.06, bottom=.12, top=.985,
                                            left=.135, right=.985))
  ax_n, ax_r = axes
  vals = []
  for rt in routes:
    sty = ROUTE_STYLE[rt]
    for d in swp._draw_order(sides[rt]):
      c = cmap[d['logr']]
      x = d['barT']/barT_f
      y = swp._gap(d['num'], d['ok'])
      ax_n.loglog(x, y, color=c, **sty)
      # the model is on the SWEEP's grid and every track shares it, so the ratio needs no
      # interpolation -- assert it rather than resampling silently
      assert len(d['barT']) == len(barT), 'track and sweep grids differ'
      ax_r.loglog(x, swp._gap(d['num']/model, d['ok']), color=c, **sty)
      vals.append(d['num'][d['ok']])
  ax_n.loglog(barT/barT_f, model, color='k', lw=2., ls='--', zorder=5)

  # y range from the MEASURED points and the model together: a track that runs off the
  # frequency window would otherwise squash every real curve into the middle
  vals = np.concatenate(vals) if vals else np.array([])
  vals = np.concatenate([vals, model[np.isfinite(model)]])
  vals = vals[np.isfinite(vals) & (vals > 0.)]
  if vals.size:
    ax_n.set_ylim(vals.min()/2., vals.max()*2.)
  ax_r.set_ylim(RATIO_YLIM)
  # a log axis over less than a decade labels itself '3 x 10^-1, 4 x 10^-1, ...'; the
  # numbers here are read as factors off 1, so they are spelled as factors
  ax_r.set_yticks(RATIO_TICKS)
  ax_r.set_yticklabels([f'{t:g}' for t in RATIO_TICKS])
  ax_r.yaxis.set_minor_formatter(plt.NullFormatter())
  ax_r.axhline(1., color='k', ls='--', lw=1., zorder=0)
  for ax in axes:
    swp._mark_hydro_times(ax, barT_f, barT_off, tnorm=barT_f)
  ax_n.set_ylabel(NUM_LABEL)
  ax_r.set_ylabel(RATIO_LABEL)
  ax_r.set_xlabel(swp.TNORM_LABEL)

  handles = [plt.Line2D([], [], color='k', lw=2., ls='--', label='C25')]
  handles += [plt.Line2D([], [], color='k', label=ROUTE_LABEL[rt], **ROUTE_STYLE[rt])
              for rt in routes]
  ax_n.legend(handles=handles, loc='lower left', fontsize=9, framealpha=.9)
  cax = fig.add_axes([.105, .048, .88, .016])
  fig.colorbar(sm, cax=cax, orientation='horizontal', label=CBAR_LABEL)

  os.makedirs(outdir, exist_ok=True)
  name = 'num_vs_C25.png' if z == Z_RS else f'num_vs_C25_z={z}.png'
  png = os.path.join(outdir, name)
  fig.savefig(png, dpi=300)
  plt.close(fig)
  print(f'  -> {png}')
  return png


# ---------------------------------------------------------------------------
# the table
# ---------------------------------------------------------------------------
def _stats(rat, m):
  '''median and the 16-84 range of a ratio over the bins m selects, as strings.'''
  v = rat[m & np.isfinite(rat)]
  if v.size < 3:
    return '--', '--', 0
  q16, q50, q84 = np.percentile(v, [16, 50, 84])
  return f'{q50:.3f}', f'{q16:.3f}-{q84:.3f}', int(v.size)


def build_table(key=KEY, method=METHOD, zlist=(Z_RS, Z_FS), routes=ROUTES,
    outdir=None, nproc=NPROC, use_cache=True, sides=None):
  '''
  nu_m/C25 per shell, per route and per sweep point, split at the shell crossing --
  the rise (bar{T} <= bar{T}_f, where the model's own rising branch applies) and the tail
  (past it, where the model is pure high-latitude) are different statements and pooling
  them hides which half disagrees. `n` is the number of bins the route calls a
  measurement, i.e. its coverage, which is part of the result and not a footnote.
  '''
  import csv
  outdir = _outdir(key, outdir)
  rows = []
  for z in zlist:
    results = swp.load_sweep(swp.method_outdir(method, key, z))
    if not results:
      print(f'build_table: no cached sweep for z={z}')
      continue
    barT_f = swp.exit_onset_barT(key, z=z)
    model = c25_num(key, z, results[0]['Tb'])
    for rt in routes:
      side = (sides or {}).get((z, rt)) or load_side(
          z=z, key=key, method=method, route=rt, nproc=nproc, use_cache=use_cache,
          outdir=outdir)
      for d in side:
        rat = d['num']/model
        rise = d['ok'] & (d['barT'] <= barT_f)
        tail = d['ok'] & (d['barT'] > barT_f)
        r_all = _stats(rat, d['ok']); r_ri = _stats(rat, rise); r_ta = _stats(rat, tail)
        rows.append([z, rt, f"{d['logr']:+.0f}", r_all[2], r_all[0], r_all[1],
                     r_ri[2], r_ri[0], r_ta[2], r_ta[0]])
  cols = ['z', 'route', 'log10(C)', 'n', 'nu_m/C25', '[q16-q84]',
          'n rise', 'rise', 'n tail', 'tail']
  os.makedirs(outdir, exist_ok=True)
  path = os.path.join(outdir, 'num_vs_C25_table.csv')
  with open(path, 'w', newline='') as f:
    w = csv.writer(f); w.writerow(cols); w.writerows(rows)
  print(f'  -> {path}')

  fig, ax = plt.subplots(figsize=(9.5, .28*len(rows) + 1.))
  ax.axis('off')
  tbl = ax.table(cellText=rows, colLabels=cols, loc='center', cellLoc='center')
  tbl.auto_set_font_size(False); tbl.set_fontsize(7.5); tbl.scale(1, 1.2)
  png = os.path.join(outdir, 'num_vs_C25_table.png')
  fig.savefig(png, dpi=200, bbox_inches='tight')
  plt.close(fig)
  print(f'  -> {png}')
  return rows


def report_crossing(key=KEY, zlist=(Z_RS, Z_FS)):
  '''bar{T}_f as the model computes it against the run's measured crossing, per shell --
  the one number of the comparison that is not a curve.'''
  for z in zlist:
    meas = swp.exit_onset_barT(key, z=z)
    try:
      mod = c25_barT_f(key, z)
    except Exception as e:                   # no fit table for this a_u family
      print(f'  z={z}: bar_T_f measured {meas:.4f}, model unavailable ({e})')
      continue
    print(f'  z={z}: bar_T_f model {mod:.4f} vs measured {meas:.4f} '
          f'({100*(mod/meas - 1):+.2f}%)')


def main(key=KEY, method=METHOD, zlist=(Z_RS, Z_FS), routes=ROUTES, outdir=None,
    nproc=NPROC, use_cache=True):
  '''
  The whole comparison: one figure per shell, one table over both, and the crossing-time
  line. The tracks are loaded ONCE and handed to both consumers, so a warm run pays for
  the plotting alone.
  '''
  outdir = _outdir(key, outdir)
  os.makedirs(outdir, exist_ok=True)
  print(f'nu_m vs C25: key={key} method={method} shells={list(zlist)}')
  sides = {}
  for z in zlist:
    for rt in routes:
      sides[(z, rt)] = load_side(z=z, key=key, method=method, route=rt, nproc=nproc,
                                 use_cache=use_cache, outdir=outdir)
  for z in zlist:
    plot_num_model(key=key, method=method, z=z, routes=routes, outdir=outdir,
                   sides={rt: sides[(z, rt)] for rt in routes})
  build_table(key=key, method=method, zlist=zlist, routes=routes, outdir=outdir,
              sides=sides)
  report_crossing(key=key, zlist=zlist)
  swp.trim_pngs(outdir)
  print(f'Figures saved to {outdir}')
  return sides


if __name__ == '__main__':
  # the pools start with 'forkserver' (cell_pool.pool_context), which re-imports this
  # module in every worker -- so the entry point MUST be guarded or each worker reruns it
  main()
