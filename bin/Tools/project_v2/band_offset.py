# -*- coding: utf-8 -*-
# @Author: acharlet

'''
THE ESTIMATOR OFFSET, MEASURED ON THE REAL SPECTRA INSTEAD OF ON SYNTHETICS.

FC* and VFC cannot have their mid window re-centred, because `_recentred_mid` needs BOTH
outer windows and they have no nu^(4/3) one: LOGNU_MIN already sits at -6.7 against
nu_B/nu_m = 10^-6.678, and syn_emiss_exact zeroes everything below nu'_B, so nu_B is a hard
floor. That is the whole reason those classes need a bias treatment of their own.

Lift the floor (SYN_NO_LOWCUT=1, see radiation_cooling) and recompute the sweep on a band
reaching 10^-10 -- hpc/unstaged/calib_deepband.sh -- and the window appears. Each spectrum is
then measured TWICE: over the extended band, where it re-centres, and over its own production
sub-range, where it falls back. The difference is the offset the production bins need, taken
from the spectra themselves rather than from GS02 synthetics the computed spectra are known
not to belong to.

MEASURED, fiducial, 2646 bins over both shells:
  VFC -> FC  323 bins and FC* -> FC  274 gain the window; mid_from goes plateau ->
  plateau_recentred for exactly those 597.
  offset da = a_deep - a_prod:  FC* -0.0084 [-0.0209, +0.0024],  VFC +0.0094 [+0.0025,
  +0.0216]. OPPOSITE SIGNS -- no single synthetic grid was going to capture both.
  INERT WHERE IT SHOULD BE: FC, SC, MC and VSC (1996 bins) give da EXACTLY 0.0000, with
  |dlog b_hi| 0.0000 and nu_M ratio 1.000000. Extending the band changes nothing that
  already had its window.
  nu_M is 1.000000 even where the class CHANGES, so the extension only supplies the missing
  low end. b_hi moves only where a_mid does (FC* 0.0151 dex, VFC 0.0409), which is the
  documented dlog b_hi = dc_hi/(a_hi - a_mid) amplification and is why this table carries
  b_hi alongside a_mid: b_hi is nu_m in fast cooling and nu_c in slow.

THE TABLE IS RUN-SPECIFIC (it is measured on one run's spectra), unlike segment_route's
synthetic grids -- hence the key in its filename. A hi-res table needs its own deep sweep.

  python -c "import band_offset as B; B.main()"
  python -c "import band_offset as B; B.main(key='cooling_g100_hires')"
'''

import sys, os, time, csv, numpy as np
import sweep_gammacm as S, spectral_breaks as sb

PROD_BOTTOM = S.LOGNU_MIN          # -6.7: where the production band stops
FIELDS = ('logr', 'z', 'step', 'barT', 'x',
          'cls_deep', 'cls_prod', 'mid_deep', 'mid_prod',
          'a_deep', 'a_prod', 'da', 'bhi_deep', 'bhi_prod', 'dlog_bhi',
          'nuM_deep', 'nuM_prod',
          # geometry ON THE EXTENDED BAND, where the bin is re-centred: this is what the
          # synthetic grid must be evaluated at, because after `da` the bin sits on the
          # re-centred footing and would classify FC there.
          'sep_deep', 's1_deep', 'off_deep', 'depth_deep', 'bias_deep',
          # total = da - bias_deep: production a_mid + total lands on the asymptote. `da`
          # is estimator-to-estimator (measured on the real spectra), `bias_deep` is
          # estimator-to-truth (measured on synthetics of known slope). Each dataset is
          # used for the only thing it can measure.
          'total')


_GRID = {}


def _grid_bias(br, sep, s1, off):
  '''
  The synthetic grid's bias for the EXTENDED-band measurement, at that bin's own class --
  interpolated within class, because the surface steps where the classifier flips (see
  mid_slope_evolution._bias_interp).
  '''
  import mid_slope_evolution as M
  d = (br.get('det') or {})
  rg, branch = d.get('regime'), ('fc' if 'fc' in (d.get('segs') or {}) else 'sc')
  k = (branch, rg)
  if k not in _GRID:
    _GRID[k] = M._bias_interp(branch, rg)
  f = _GRID[k]
  if f is None or not all(np.isfinite(v) for v in (sep, s1, off)):
    return np.nan
  return float(f(sep, s1, off))


def one_point(r, z, step=14, verbose=True, key=None):
  nub = np.asarray(r['nub'], float); nuFnu = np.asarray(r['nuFnu'], float)
  p = r['env'].psyn
  barT = np.asarray(r['Tb'], float) - 1.
  bf = S.exit_onset_barT(key or S.DEFAULT_KEY, z=z)
  # THE PRODUCTION BOTTOM IN *STORED* UNITS. run_sweep stores nub = nuobs/max(nu0, nuc)
  # while LOGNU_MIN is in nu/nu_m, so for a slow-cooling point (nuc > nu0) the stored grid
  # sits 2*logr decades lower. Cutting at LOGNU_MIN directly threw away 2*logr decades of
  # real production band and, at logr=+3, left a window so short that every bin was declined.
  shift = max(0., 2.*float(r['log10ratio']))
  prod = nub >= 10**(PROD_BOTTOM - shift)
  Fpk = np.nanmax(nuFnu, axis=1); thr = 1e-10*np.nanmax(Fpk)
  out = []
  for i in range(0, len(nuFnu), step):
    if Fpk[i] <= thr:
      continue
    sp = nuFnu[i]
    try:
      bd = sb.breaks_from_identified(nub, sp, p)
      bp = sb.breaks_from_identified(nub[prod], sp[prod], p)
    except Exception:
      continue
    dd, dp = (bd.get('det') or {}), (bp.get('det') or {})
    if not np.isfinite(bd['a_mid']) or not np.isfinite(bp['a_mid']):
      continue
    lo = float(np.log10(nub[prod].min()))   # = PROD_BOTTOM - shift, the real production edge
    gs = sb.smoothing_from_identified(nub[prod], sp[prod], p, br=bp)
    gd = sb.smoothing_from_identified(nub, sp, p, br=bd)
    # geometry EXACTLY as mid_slope_evolution._measure_point derives it: '2brk_flo' has no
    # lower CROSSING, its b_lo is a band-bottom seed and the fit is what places it, so its
    # separation must come from the fitted breaks. Getting this wrong pinned off at 0.030
    # for every FC* row in the first pass.
    def _geom(br, g_, lo_):
      blo = g_['b_lo_fit'] if br['shape'] == '2brk_flo' else br['b_lo']
      bhi = g_['b_hi_fit'] if br['shape'] == '2brk_flo' else br['b_hi']
      if not (np.isfinite(bhi) and np.isfinite(blo) and blo > 0):
        return np.nan, np.nan, np.nan
      sp_ = np.log10(bhi/blo); dp_ = np.log10(bhi) - lo_
      return sp_, dp_, dp_ - sp_
    sep, depth, _o = _geom(bp, gs, lo)
    lo_d = float(np.log10(nub.min()))
    sep_d, depth_d, off_d = _geom(bd, gd, lo_d)
    bias_d = _grid_bias(bd, sep_d, gd['s1'], off_d)
    out.append(dict(logr=r['log10ratio'], z=z, step=i, barT=barT[i], x=barT[i]/bf,
        cls_deep=str(dd.get('regime')), cls_prod=str(dp.get('regime')),
        mid_deep=str(bd['mid_from']), mid_prod=str(bp['mid_from']),
        a_deep=bd['a_mid'], a_prod=bp['a_mid'], da=bd['a_mid']-bp['a_mid'],
        bhi_deep=bd['b_hi'], bhi_prod=bp['b_hi'],
        dlog_bhi=(np.log10(bd['b_hi']/bp['b_hi'])
                  if np.isfinite(bd['b_hi']) and np.isfinite(bp['b_hi']) else np.nan),
        nuM_deep=bd['nuM'], nuM_prod=bp['nuM'],
        sep_deep=sep_d, s1_deep=gd['s1'], off_deep=off_d, depth_deep=depth_d,
        bias_deep=bias_d,
        total=(bd['a_mid']-bp['a_mid']) - bias_d if np.isfinite(bias_d) else np.nan))
  if verbose:
    print(f'  logr={r["log10ratio"]:+.0f} z={z}: {len(out)} bins', flush=True)
  return out


def reindex(key=None, path=None):
  """
  Re-evaluate bias_deep (and so total) from the geometry ALREADY IN THE TABLE, against the
  current bias grid. The spectra do not change when the grid is re-indexed -- da, the
  classes and the deep geometry are all measurements -- so re-running the deep sweep to pick
  up a new grid would be an hour of recomputing numbers that cannot move. Only the two
  columns that are grid READS are rewritten.
  """
  import segment_route as R
  key = key or S.DEFAULT_KEY
  path = path or R.calib_path(f'band_offsets_{key}.csv')
  rows = list(csv.DictReader(open(path)))
  _GRID.clear()
  import mid_slope_evolution as M
  itp = {}
  def f_(cls):
    if cls not in itp:
      br = 'sc' if cls in ('SC', 'VSC') else 'fc'
      itp[cls] = M._bias_interp(br, cls)
    return itp[cls]
  n = 0
  for r in rows:
    g = f_(r['cls_deep'])
    try:
      q = [float(r[k]) for k in ('sep_deep', 's1_deep', 'off_deep', 'da')]
    except ValueError:
      q = [np.nan]*4
    b = float(g(*q[:3])) if (g is not None and all(np.isfinite(q[:3]))) else np.nan
    old = r['bias_deep']
    r['bias_deep'] = b
    r['total'] = (q[3] - b) if np.isfinite(b) and np.isfinite(q[3]) else np.nan
    n += int(np.isfinite(b) != (old not in ('', 'nan')))
  with open(path, 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(FIELDS)); w.writeheader(); w.writerows(rows)
  fin = sum(1 for r in rows if np.isfinite(float(r['total'])))
  print(f'{len(rows)} rows re-indexed -> {path}  ({fin} with a total, {n} changed coverage)')
  return rows


def main(step=14, key=None):
  '''
  The offset table for ONE run. THE TABLE IS RUN-SPECIFIC -- it is measured on that run's
  spectra -- so the key goes into the deep-sweep path, into the crossing time the x axis is
  normalised by, AND into the filename. It used to be fixed at DEFAULT_KEY throughout, which
  meant a hi-res table could not be built at all and the hi-res figure silently borrowed the
  fiducial one (mid_slope_evolution._band_total).
  '''
  key = key or S.DEFAULT_KEY
  rows = []
  for z, suf in ((4, '_deepband'), (1, '_z=1_deepband')):
    od = S.figdir('gammacm_sweep_data_rarcut_fc2' + suf, key)
    res = S.load_sweep(od)
    if not res:
      print(f'no deep sweep in {od}'); continue
    for r in sorted(res, key=lambda q: q['log10ratio']):
      rows += one_point(r, z, step=step, key=key)
  from segment_route import calib_path
  out = calib_path(f'band_offsets_{key}.csv')
  with open(out, 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(FIELDS)); w.writeheader(); w.writerows(rows)
  print(f'{len(rows)} rows -> {out}')
  return rows


if __name__ == '__main__':
  t0 = time.time(); main(); print(f'{time.time()-t0:.0f} s')
