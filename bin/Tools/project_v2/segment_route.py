# -*- coding: utf-8 -*-
# @Author: acharlet

'''
THE PAPER'S SPECTRAL-SHAPE MEASUREMENT. Charlet et al. (cooling regimes) takes its regimes,
its break frequencies and its smoothing exponents from this module and from nothing else.
The other routes in the project -- sweep_gammacm.measure_regime, fit_gs02_spectrum /
track_breaks_gs02, spectral_breaks.breaks_from_segments, and slope_validation's smoothing
tables -- are superseded for that purpose and carry banners saying so. The one thing that is
NOT superseded is slope_validation's free-slope validation of the 4/3 and 1-p/2 asymptotes,
which is what makes holding them here legitimate.

The smoothing exponents measured by the SELF-CONTAINED route, per break and per regime.

One chain, rooted in the spectrum and nothing else:

    identify_segments             which power-law segments the spectrum SHOWS, and the shape
                                  class that follows from the set of them
    breaks_from_identified        the breaks, as the crossings of exactly those segments
    smoothing_from_identified     s, by refitting granot_sari_syn with those crossings and
                                  those slopes HELD -- the only freedom left

WHAT IS DIFFERENT FROM slope_validation, which fits the same shape with the same estimator.
That module places its segment windows a fixed factor in frequency from breaks taken from a
GS02 template fit, and labels each bin's regime from the TEMPLATE's fitted beta_mid
(classify_regime reads tr['beta_mid']). Neither enters a fitted parameter, so its slopes and
its s are not circular -- but the ROWS of its per-regime table are a template verdict, and
its coverage is a template's coverage: a window can always be placed once a break has been
fitted, whether or not the spectrum shows a segment there. Here the regime is the SHAPE
CLASS (what is identified), and a bin with no identified mid segment gets the merged shape or
the tangent fallback rather than a two-break fit. Lower coverage, and every reported bin
stands on a segment that was actually measured.

TWO THINGS THIS ROUTE DOES THAT THE OTHER DOES NOT

  The SMEARED cut-off (syn_cutoff_R_smeared) is used throughout, not the single-zone R.
  The observed rolloff is a superposition over cells carrying a spread of nu_M, and fitting
  one zone to it biases nu_M high. Elsewhere that barely matters -- it moves no slope -- but
  this route divides the cut-off out before locating a CROSSING, and the high line's
  intercept is fitted partly inside the rolloff, where dlog b_hi = dc_hi/(a_hi - a_mid)
  levers it. One cut-off measurement per spectrum serves the window cap, the flattening and
  the smoothing fit, so no two steps assume different rolloffs.

  The TANGENT FALLBACK (_tangent_mid) supplies a mid line wherever the mid slope cannot be
  properly measured -- i.e. no mid window was identified at all, which is what an MC class
  means. The slope is held at the asymptote the spectrum passes through on its way from 4/3
  to 1-p/2 and only the intercept is fitted, so it needs no plateau. Those bins are marked
  mid_from='tangent' and carry NO evidence about whether the spectrum is marginal; the merged
  single break is still measured on every MC bin so the two descriptions can be compared.

Ground truth is the cached sweeps; no spectrum is recomputed. The cut-off scan with a free
sigma costs ~0.8 s a bin, so points are run in parallel (see NPROC).

  regime_table          - s1, s2 per shape class per shell, and pooled; MC's merged-break s
                          reported separately from the two-break exponents
  epoch_table           - the same split on-axis (bar{T} <= bar{T}_f) vs post-crossing, which
                          slope_validation showed is the split that matters for FC and SC
  prescription_check    - freeze s at the pooled median and refit: the residual against the
                          free fit IS the cost of tabulating that value
'''

import os
import numpy as np
import pandas as pd

from environment import GAMMA_dir
import cell_pool
import spectral_breaks as sb
import sweep_gammacm as swp

OUTDIR = os.path.join(GAMMA_dir, 'bin', 'Tools', 'figures', 'segment_route')
KEY = 'cooling_g100'
METHOD = 'data'
Z_RS, Z_FS = 4, 1
# shape classes, in the order identify_segments' docstring lists them. MC carries the merged
# break rather than a pair, and VSC has no upper break in band at all -- both are reported,
# neither is pooled into an s1/s2 row.
CLASSES = ('VFC', 'FC*', 'FC', 'MC', 'SC', 'VSC')
# MC is a two-break class here, but not on the same footing as FC and SC: its mid slope is
# FITTED inside the shape fit rather than held (spectral_breaks.smoothing_from_identified's
# free_bmid='mc'), because an MC spectrum displays no mid segment to measure beforehand. Its
# s1 and s2 therefore come out of a four-parameter fit and carry a much wider spread -- quote
# them with it, never as a tabulated pair. The merged single break is still measured on every
# MC bin and reported alongside, as the alternative description.
TWO_BREAK = ('FC', 'SC', 'MC')    # the classes whose s1 AND s2 are both measured
ONE_BREAK = ('VFC', 'FC*')        # ... s2 only: no nu^(4/3) segment in band
PRESC_RMS_MAX = 0.05              # as in slope_validation: a frozen fit worse than this
PRESC_BAD_MAX = 0.10              # ... and a case failing more than this fraction is split
NPROC = None                      # None -> cell_pool.resolve_nproc (GAMMACM_NPROC or ncpu-1)


def _ensure_outdir():
  os.makedirs(OUTDIR, exist_ok=True)


def _run_point(args):
  '''One sweep point, in a worker: load the cache, run the route on every time bin.'''
  key, method, z, logr = args
  res = swp.load_sweep(swp.method_outdir(method, key, z))
  r = [q for q in res if abs(q['log10ratio'] - logr) < 1e-9][0]
  tk = sb.track_segment_route(r)
  tk['logr'], tk['z'] = logr, z
  return tk


def load_side(z=Z_RS, key=KEY, method=METHOD, nproc=NPROC):
  '''
  Every cached sweep point of one shell, with the segment route run on every time bin.
  Points are independent, so they are run in parallel -- the smeared cut-off scan is ~0.8 s
  a bin and a point carries ~450 of them.
  '''
  res = swp.load_sweep(swp.method_outdir(method, key, z))
  if not res:
    raise FileNotFoundError(f'no cached sweep for z={z} -- run sweep_gammacm.main first')
  logrs = sorted(float(r['log10ratio']) for r in res)
  barT_f = swp.exit_onset_barT(key, z=z)
  jobs = [(key, method, z, lr) for lr in logrs]
  np_ = cell_pool.resolve_nproc(nproc, cap=len(jobs))
  if np_ > 1:
    ctx = cell_pool.pool_context()
    with ctx.Pool(np_) as pool:
      tracks = pool.map(_run_point, jobs)
  else:
    tracks = [_run_point(j) for j in jobs]
  out = []
  for tk in tracks:
    tk['barT_f'] = barT_f
    n_ok = int(tk['s_ok'].sum())
    cl = {c: int(sum(1 for q in tk['regime'] if q == c)) for c in CLASSES}
    print(f"  log10ratio={tk['logr']:+.1f} z={z}: {n_ok} bins with s, classes "
          + ' '.join(f'{c}:{n}' for c, n in cl.items() if n), flush=True)
    out.append(tk)
  return out


def _cat(sides, key, mask_fn):
  '''one field, pooled over sides on the bins mask_fn selects'''
  v = [np.asarray(tk[key], float)[mask_fn(tk)] for tk in sides]
  return np.concatenate(v) if v else np.array([])


def _class_mask(tk, cls, onaxis=None, tangent=None):
  '''bins of one shape class, optionally split by epoch and by how the mid line was got'''
  m = np.array([q == cls for q in tk['regime']]) & tk['s_ok']
  if onaxis is not None:
    m &= (tk['barT'] <= tk['barT_f']) if onaxis else (tk['barT'] > tk['barT_f'])
  if tangent is not None:
    m &= np.array([q == 'tangent' for q in tk['mid_from']]) == bool(tangent)
  return m


def _band(v):
  '''median [q16-q84] of a pooled bin sample, and its size'''
  v = np.asarray(v, float); v = v[np.isfinite(v)]
  if not v.size:
    return '--'.rjust(18), np.nan, 0
  q16, q84 = np.percentile(v, [16, 84])
  return f'{np.median(v):5.2f} [{q16:4.2f}-{q84:4.2f}]'.rjust(18), float(np.median(v)), v.size


def regime_table(sides_by_z, verbose=True, epoch=False):
  '''
  s1 and s2 per shape class, pooled over the raw BINS (never over per-point summaries: a
  median of medians would sit outside its own quoted band).

  With epoch=True every class is additionally split on bar{T} <= bar{T}_f. The two-break
  exponents are reported only for the classes that HAVE two breaks in band; VFC and FC* carry
  s2 alone, and MC's number is the merged break's single s -- a different shape, spanning a
  slope change of 1/3 + p/2 rather than the mid-to-outer step, so it is never to be compared
  with s1 or s2 as a width.
  '''
  rows = []
  splits = ((True, 'on-axis'), (False, 'post-crossing')) if epoch else ((None, 'all'),)
  for zlab, sides in [('RS', sides_by_z[0]), ('FS', sides_by_z[1])] + \
                     [('both', sides_by_z[0] + sides_by_z[1])]:
    for cls in CLASSES:
      for onax, elab in splits:
        m = lambda tk: _class_mask(tk, cls, onax)
        n = int(sum(m(tk).sum() for tk in sides))
        if not n:
          continue
        _, s1, n1 = _band(_cat(sides, 's1', m))
        _, s2, n2 = _band(_cat(sides, 's2', m))
        _, sg, ng = _band(_cat(sides, 's_1brk', m))
        rms = _cat(sides, 'rms', m)
        tan = int(sum((_class_mask(tk, cls, onax, tangent=True)).sum() for tk in sides))
        rows.append(dict(shell=zlab, regime=cls, epoch=elab, n=n, n_s1=n1, s1=s1,
                         n_s2=n2, s2=s2, n_1brk=ng, s_1brk=sg, n_tangent=tan,
                         rms=float(np.nanmedian(rms)) if np.isfinite(rms).any() else np.nan))
  df = pd.DataFrame(rows)
  if verbose and len(df):
    ttl = 'BY SHAPE CLASS AND EPOCH' if epoch else 'BY SHAPE CLASS'
    print(f"\n{'=== ' + ttl + ' (identified segments -> crossings -> held-break s) ':=<118}")
    print('s are median [q16-q84] over pooled bins. s1/s2 are the TWO-BREAK exponents '
          '(lower/upper);\ns(1brk) is the merged 4/3 -> 1-p/2 break of the MC spectra, a '
          'different shape -- not a width to\ncompare with s1. "tan" counts bins whose mid '
          'line came from the tangent fallback, which hold\nthe mid slope and so say nothing '
          'about marginality.')
    hdr = (f"{'shell':>5} {'class':>5} {'epoch':>14} {'bins':>5} | {'N':>4} {'s1':>18} | "
           f"{'N':>4} {'s2':>18} | {'N':>4} {'s(1brk)':>18} | {'rms':>7} {'tan':>5}")
    print(hdr); print('-'*len(hdr))
    f = lambda v: f'{v:5.2f}'.rjust(18) if np.isfinite(v) else '--'.rjust(18)
    for _, w in df.iterrows():
      print(f"{w.shell:>5} {w.regime:>5} {w.epoch:>14} {w.n:>5} | {w.n_s1:>4} "
            f"{f(w.s1)} | {w.n_s2:>4} {f(w.s2)} | {w.n_1brk:>4} "
            f"{f(w.s_1brk)} | {w.rms:7.4f} {w.n_tangent:>5}")
  return df


def regime_bands(sides_by_z, epoch=True, verbose=True):
  '''
  The same numbers as regime_table, printed with their q16-q84 spread -- the form to quote,
  since a median alone hides how well the class is determined.
  '''
  rows = []
  splits = ((True, 'on-axis'), (False, 'post-crossing')) if epoch else ((None, 'all'),)
  sides = sides_by_z[0] + sides_by_z[1]
  for cls in CLASSES:
    for onax, elab in splits:
      m = lambda tk: _class_mask(tk, cls, onax)
      n = int(sum(m(tk).sum() for tk in sides))
      if not n:
        continue
      b1, s1, n1 = _band(_cat(sides, 's1', m))
      b2, s2, n2 = _band(_cat(sides, 's2', m))
      bg, sg, ng = _band(_cat(sides, 's_1brk', m))
      rms = _cat(sides, 'rms', m)
      rows.append(dict(regime=cls, epoch=elab, n=n, s1=s1, s2=s2, s_1brk=sg,
                       b1=b1, b2=b2, bg=bg, n_s1=n1, n_s2=n2, n_1brk=ng,
                       rms=float(np.nanmedian(rms)) if np.isfinite(rms).any() else np.nan))
  df = pd.DataFrame(rows)
  if verbose and len(df):
    print(f"\n{'=== BOTH SHELLS POOLED ':=<96}")
    hdr = (f"{'class':>5} {'epoch':>14} {'bins':>5} | {'s1 (lower)':>18} | "
           f"{'s2 (upper)':>18} | {'s (merged)':>18} | {'rms':>7}")
    print(hdr); print('-'*len(hdr))
    for _, w in df.iterrows():
      print(f"{w.regime:>5} {w.epoch:>14} {w.n:>5} | {w.b1} | {w.b2} | {w.bg} | "
            f"{w.rms:7.4f}")
  return df


def _refit(tk, i, s_hold=None, s1brk_hold=None, merged=False):
  '''
  Refit one bin with the SAME breaks, mid slope and cut-off the free fit used, changing only
  whether s is frozen -- so the difference in rms is the frozen smoothing and nothing else.

  The geometry is taken from the track rather than re-derived: track_segment_route already
  stores every argument the smoothing step takes (b_lo, b_hi, a_mid, nuM, sigma), so the
  identification and the smeared cut-off scan -- 0.8 s a bin, and by far the expensive part
  -- are not repeated. Re-running the whole route here would give the same numbers a
  thousand times slower.

  merged=True forces the MERGED single break whatever geometry the bin was given. An MC bin
  that the tangent fallback supplied a mid line for carries shape '2brk_tangent', but the
  value being tabulated for MC is the merged break's s -- refitting it as a two-break form
  with s free would compare the frozen prescription against nothing.
  '''
  r = tk['_r']
  x = swp.nu_over_num(r)
  sp, p, sig = r['nuFnu'][i, :], r['env'].psyn, tk['sigma'][i]
  shape = '1brk_mc' if merged else tk['shape'][i]
  if shape in ('2brk', '2brk_tangent', '2brk_free'):
    # the mid slope stays as free as it was in the bin's own fit, so the residual difference
    # is the frozen SMOOTHING and nothing else; freezing a_mid too would price two changes
    return sb.fit_smoothing_held(x, sp, p, tk['b_lo'][i], tk['b_hi'][i], tk['nuM'][i],
                                 tk['a_mid'][i] - 1., s_hold=s_hold, sigma=sig,
                                 free_bmid=bool(tk['mid_fitted'][i]))['rms']
  if shape == '1brk_vfc':
    return sb.fit_smoothing_held(x, sp, p, tk['b_hi'][i], np.nan, tk['nuM'][i], np.nan,
                                 vfc=True, s_hold=s_hold, sigma=sig)['rms']
  if shape == '1brk_mc':
    return sb.fit_single_break(x, sp, p, tk['nuM'][i], s_hold=s1brk_hold, sigma=sig)['rms']
  return np.nan


def prescription_check(sides_by_z, key=KEY, method=METHOD, epoch=True, verbose=True,
    rms_max=PRESC_RMS_MAX, bad_max=PRESC_BAD_MAX):
  '''
  Does the tabulated median actually fit? Freeze s at the pooled per-class median, refit every
  bin of that class, and compare against the same bin fitted with s free. The difference is
  the cost of tabulating; the fraction of bins the frozen values fail to describe to rms_max
  is what says whether one number can stand for the class at all.

  Needs the spectra, so the sweep cache is re-loaded here and attached to each track.
  '''
  for zi, z in enumerate((Z_RS, Z_FS)):
    res = swp.load_sweep(swp.method_outdir(method, key, z))
    for tk in sides_by_z[zi]:
      tk['_r'] = [q for q in res if abs(q['log10ratio'] - tk['logr']) < 1e-9][0]
  sides = sides_by_z[0] + sides_by_z[1]
  splits = ((True, 'on-axis'), (False, 'post-crossing')) if epoch else ((None, 'all'),)
  rows = []
  for cls in CLASSES:
    for onax, elab in splits:
      m = lambda tk: _class_mask(tk, cls, onax)
      n = int(sum(m(tk).sum() for tk in sides))
      if not n:
        continue
      # MC is now tabulated on the two-break shape with its mid slope fitted, like every
      # other two-break class; the merged break is measured too but is no longer the value
      # being tested, since it is not the shape the route reports for MC.
      merged = False
      s1 = np.nanmedian(_cat(sides, 's1', m)) if cls in TWO_BREAK else np.nan
      s2 = np.nanmedian(_cat(sides, 's2', m)) if cls in TWO_BREAK + ONE_BREAK else np.nan
      sg = np.nanmedian(_cat(sides, 's_1brk', m)) if merged else np.nan
      if not (np.isfinite(s2) or np.isfinite(sg)):
        continue                    # VSC: no shape is constrained, so nothing to freeze
      free, held = [], []
      for tk in sides:
        for i in np.flatnonzero(m(tk)):
          free.append(tk['rms_1brk'][i] if merged else tk['rms'][i])
          held.append(_refit(tk, i, s_hold=(s1, s2) if np.isfinite(s2) else None,
                             s1brk_hold=sg if np.isfinite(sg) else None, merged=merged))
      free, held = np.array(free, float), np.array(held, float)
      g = np.isfinite(free) & np.isfinite(held)
      if not g.any():
        continue
      rows.append(dict(regime=cls, epoch=elab, n=int(g.sum()), s1=s1, s2=s2, s_1brk=sg,
                       rms_free=float(np.median(free[g])), rms_held=float(np.median(held[g])),
                       cost=float(np.median(held[g]) - np.median(free[g])),
                       frac_bad=float(np.mean(held[g] > rms_max))))
  df = pd.DataFrame(rows)
  if verbose and len(df):
    print(f"\n{'=== DOES THE TABULATED VALUE FIT? ':=<96}")
    print(f'"cost" is the median rms penalty against fitting s per spectrum; "%>thr" the '
          f'fraction of bins\nthe frozen values fail to describe to {rms_max:g} dex.')
    hdr = (f"{'class':>5} {'epoch':>14} {'N':>5} | {'s1':>6} {'s2':>6} {'s(1brk)':>8} | "
           f"{'free':>8} {'held':>8} {'cost':>8} {'%>thr':>6} | verdict")
    print(hdr); print('-'*len(hdr))
    f3 = lambda v: f'{v:6.3f}' if np.isfinite(v) else '    --'
    for _, w in df.iterrows():
      ok = np.isfinite(w.rms_held) and w.rms_held < rms_max
      tag = 'use' if ok and w.frac_bad <= bad_max else ('use with care' if ok
                                                        else 'DO NOT USE')
      print(f"{w.regime:>5} {w.epoch:>14} {w.n:>5} | {f3(w.s1)} {f3(w.s2)} {f3(w.s_1brk)} | "
            f"{w.rms_free:8.4f} {w.rms_held:8.4f} {w.cost:+8.4f} {100*w.frac_bad:5.1f}% "
            f"| {tag}")
  return df


def coverage_table(sides_by_z, verbose=True):
  '''
  What the route measures and what it declines, per sweep point: the price of refusing to
  place a window where no segment was identified. Compare against slope_validation's own bin
  counts, which are a template's coverage.
  '''
  rows = []
  for zlab, sides in (('RS', sides_by_z[0]), ('FS', sides_by_z[1])):
    for tk in sides:
      cl = {c: int(sum(1 for q in tk['regime'] if q == c)) for c in CLASSES}
      rows.append(dict(shell=zlab, logr=tk['logr'], bins=len(tk['barT']),
                       identified=int(sum(cl.values())), with_s=int(tk['s_ok'].sum()),
                       tangent=int(sum(1 for q in tk['mid_from'] if q == 'tangent')),
                       **cl))
  df = pd.DataFrame(rows)
  if verbose and len(df):
    print(f"\n{'=== COVERAGE ':=<96}")
    print(df.to_string(index=False))
  return df


# ---------------------------------------------------------------------------------------
# Which fallback should supply the mid line where none is identified?
# ---------------------------------------------------------------------------------------
def _compare_point(args):
  '''
  One sweep point, in a worker: every MC bin measured BOTH ways. The cut-off scan and the
  identification are done once per bin and shared between the two fallbacks, so the second
  costs only its crossings and its smoothing fit -- the comparison is on identical
  identifications by construction, not merely on the same spectra.
  '''
  key, method, z, logr = args
  res = swp.load_sweep(swp.method_outdir(method, key, z))
  r = [q for q in res if abs(q['log10ratio'] - logr) < 1e-9][0]
  x = swp.nu_over_num(r)
  nuFnu, p = r['nuFnu'], r['env'].psyn
  barT = np.asarray(r['Tb'], float) - 1.
  Fpk = np.nanmax(nuFnu, axis=1)
  bright = (np.isfinite(Fpk) & (Fpk > 1e-10*np.nanmax(Fpk))
            & ((np.isfinite(nuFnu) & (nuFnu > 0.)).sum(axis=1) >= 12))
  rows = []
  for i in np.flatnonzero(bright):
    sp = nuFnu[i, :]
    cut = sb.measure_cutoff_nuM(x, sp, p, flatten=False, smear=True)
    if not cut['ok']:
      continue
    det = swp.identify_segments(x, sp, p, cut=cut)
    if det is None or det['regime'] != 'MC':
      continue
    row = dict(z=z, logr=logr, i=int(i), barT=float(barT[i]))
    # 'fitted' is the CONSTRAINED variant: the tangent geometry, but the mid slope freed
    # inside the shape fit rather than held, with b_lo still anchored at its crossing
    for mode, geom, fb in (('tangent', 'tangent', False), ('free', 'free', False),
                           ('fitted', 'tangent', True)):
      br = sb.breaks_from_identified(x, sp, p, det=det, cut=cut, mid_fallback=geom)
      f = sb.smoothing_from_identified(x, sp, p, br=br, free_bmid=fb)
      row.update({f'{mode}_{k}': f[k] for k in
                  ('a_mid', 'b_lo', 'b_hi', 's1', 's2', 'rms', 'dex_mid')})
      row[f'{mode}_ok'] = bool(f['s_ok'])
      row[f'{mode}_from'] = f['mid_from']
      if fb:   # the fitted mid slope, and the upper break the same fit chose
        row[f'{mode}_a_mid'] = f['a_mid_fit']
        row[f'{mode}_b_hi'] = f['b_hi_fit']
        row[f'{mode}_at_bound'] = bool(f['bmid_at_bound'])
      row['s_1brk'], row['rms_1brk'] = f['s_1brk'], f['rms_1brk']
    rows.append(row)
  return rows


def compare_mid_fallbacks(key=KEY, method=METHOD, zlist=(Z_RS, Z_FS), nproc=NPROC,
    verbose=True, outdir=OUTDIR):
  '''
  Tangent versus free mid line, on every MC bin of the sweep.

  The two answer different questions and the comparison has to keep them apart. The tangent
  HOLDS the mid slope at a theory asymptote, so it cannot be wrong about it and cannot be
  informative about it either; the free line MEASURES it, so it can report a mid slope that
  evolves through the FC/SC crossing -- the physical expectation -- but pays for it wherever
  the knee has no straight part.

  Reported per shell: how often each yields a usable geometry, what each returns for the mid
  slope and the two breaks, how well the resulting held-break fit describes the spectrum, and
  -- the point of the exercise -- how CONTINUOUS a_mid(bar{T}) is in each, measured as the
  median absolute step between consecutive MC bins. The tangent's steps include the jumps
  where the nearest asymptote switches from 1/2 to (3-p)/2, which the free line cannot make.
  '''
  jobs = [(key, method, z, float(r['log10ratio']))
          for z in zlist
          for r in swp.load_sweep(swp.method_outdir(method, key, z))]
  np_ = cell_pool.resolve_nproc(nproc, cap=len(jobs))
  if np_ > 1:
    with cell_pool.pool_context().Pool(np_) as pool:
      out = pool.map(_compare_point, jobs)
  else:
    out = [_compare_point(j) for j in jobs]
  df = pd.DataFrame([w for rows in out for w in rows])
  if not len(df):
    return df
  if verbose:
    print(f"\n{'=== MID-LINE FALLBACK: TANGENT vs FREE, on MC bins ':=<100}")
    print('Identical identifications and cut-offs; only the mid line differs. "step" is the '
          'median\n|delta a_mid| between consecutive MC bins of one sweep point -- the '
          'continuity in time.\n"switch" counts bins where the tangent changed which '
          'asymptote it holds.')
    hdr = (f"{'shell':>5} {'mode':>8} {'usable':>7} | {'a_mid':>18} | {'step':>7} "
           f"{'switch':>7} | {'s1':>18} | {'s2':>18} | {'rms':>7} | {'vs 1brk':>8}")
    print(hdr); print('-'*len(hdr))
    for z in zlist:
      d = df[df.z == z]
      for mode in ('tangent', 'free', 'fitted'):
        ok = d[d[f'{mode}_ok']]
        steps, switches = [], 0
        for lr, g in d.groupby('logr'):
          g = g.sort_values('barT')
          v = g[f'{mode}_a_mid'].to_numpy(float)
          v = v[np.isfinite(v)]
          if len(v) > 1:
            steps += list(np.abs(np.diff(v)))
            if mode == 'tangent':
              switches += int((np.abs(np.diff(v)) > 1e-9).sum())
        b_am, _, _ = _band(ok[f'{mode}_a_mid'])
        b_s1, _, _ = _band(ok[f'{mode}_s1'])
        b_s2, _, _ = _band(ok[f'{mode}_s2'])
        rms = np.nanmedian(ok[f'{mode}_rms']) if len(ok) else np.nan
        r1 = np.nanmedian(ok['rms_1brk']) if len(ok) else np.nan
        # for the fitted variant, "switch" instead counts fits that ran into the bound --
        # a mid slope outside what a three-segment spectrum can carry, i.e. not a measurement
        flag = switches if mode == 'tangent' else (
            int(ok[f'{mode}_at_bound'].sum()) if f'{mode}_at_bound' in ok else 0)
        print(f"{z:>5} {mode:>8} {len(ok):>7} | {b_am} | "
              f"{np.median(steps) if steps else np.nan:7.4f} "
              f"{flag:>7} | {b_s1} | {b_s2} | {rms:7.4f} | {rms/r1:8.2f}")
    # where both work, how far apart are the breaks they return?
    both = df[df.tangent_ok & df.free_ok]
    if len(both):
      rl = (both.free_b_lo/both.tangent_b_lo).to_numpy(float)
      rh = (both.free_b_hi/both.tangent_b_hi).to_numpy(float)
      q = lambda v: f'{np.median(v):.3f} [{np.percentile(v, 16):.3f}-{np.percentile(v, 84):.3f}]'
      print(f'\n  on the {len(both)} bins where both work, free/tangent break ratio: '
            f'nu_1 {q(rl)}, nu_2 {q(rh)}')
      w = int((both.free_rms < both.tangent_rms).sum())
      print(f'  free fits better in {w}/{len(both)} of them '
            f'(median rms {np.nanmedian(both.free_rms):.4f} vs '
            f'{np.nanmedian(both.tangent_rms):.4f})')
  if outdir:
    _ensure_outdir()
    df.to_csv(os.path.join(outdir, 'mid_fallback_comparison.csv'), index=False)
    print(f"  wrote {os.path.join(outdir, 'mid_fallback_comparison.csv')}")
  return df


def write_tables(*dfs_named, outdir=OUTDIR):
  _ensure_outdir()
  for df, name in dfs_named:
    if df is not None and len(df):
      df.to_csv(os.path.join(outdir, name), index=False)
      print(f'  wrote {os.path.join(outdir, name)}')


def main(key=KEY, method=METHOD, outdir=OUTDIR, nproc=NPROC):
  _ensure_outdir()
  sides_by_z = []
  for z in (Z_RS, Z_FS):
    print(f'\n--- shell z={z} ---', flush=True)
    sides_by_z.append(load_side(z=z, key=key, method=method, nproc=nproc))
  cov = coverage_table(sides_by_z)
  reg = regime_table(sides_by_z)
  ep = regime_table(sides_by_z, epoch=True, verbose=False)
  bands = regime_bands(sides_by_z, epoch=True)
  presc = prescription_check(sides_by_z, key=key, method=method)
  write_tables((cov, 'coverage.csv'), (reg, 'smoothing_by_class.csv'),
               (ep, 'smoothing_by_class_epoch.csv'), (bands, 'smoothing_pooled.csv'),
               (presc, 'prescription_check.csv'), outdir=outdir)
  print(f'\nsegment-route smoothing saved to {outdir}')
  return sides_by_z, cov, reg, ep, bands, presc


if __name__ == '__main__':
  main()
