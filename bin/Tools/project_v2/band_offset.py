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
'''

import sys, os, time, csv, numpy as np
import sweep_gammacm as S, spectral_breaks as sb

PROD_BOTTOM = S.LOGNU_MIN          # -6.7: where the production band stops
FIELDS = ('logr', 'z', 'step', 'barT', 'x',
          'cls_deep', 'cls_prod', 'mid_deep', 'mid_prod',
          'a_deep', 'a_prod', 'da', 'bhi_deep', 'bhi_prod', 'dlog_bhi',
          'nuM_deep', 'nuM_prod', 'sep', 's1', 'off', 'depth', 's2')


def one_point(r, z, step=14, verbose=True):
  nub = np.asarray(r['nub'], float); nuFnu = np.asarray(r['nuFnu'], float)
  p = r['env'].psyn
  barT = np.asarray(r['Tb'], float) - 1.
  bf = S.exit_onset_barT(S.DEFAULT_KEY, z=z)
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
    sep = (np.log10(bp['b_hi']/bp['b_lo'])
           if np.isfinite(bp['b_hi']) and np.isfinite(bp['b_lo']) and bp['b_lo'] > 0 else np.nan)
    lo = float(np.log10(nub[prod].min()))   # = PROD_BOTTOM - shift, the real production edge
    depth = np.log10(bp['b_hi']) - lo if np.isfinite(bp['b_hi']) else np.nan
    gs = sb.smoothing_from_identified(nub[prod], sp[prod], p, br=bp)
    out.append(dict(logr=r['log10ratio'], z=z, step=i, barT=barT[i], x=barT[i]/bf,
        cls_deep=str(dd.get('regime')), cls_prod=str(dp.get('regime')),
        mid_deep=str(bd['mid_from']), mid_prod=str(bp['mid_from']),
        a_deep=bd['a_mid'], a_prod=bp['a_mid'], da=bd['a_mid']-bp['a_mid'],
        bhi_deep=bd['b_hi'], bhi_prod=bp['b_hi'],
        dlog_bhi=(np.log10(bd['b_hi']/bp['b_hi'])
                  if np.isfinite(bd['b_hi']) and np.isfinite(bp['b_hi']) else np.nan),
        nuM_deep=bd['nuM'], nuM_prod=bp['nuM'],
        sep=sep, s1=gs['s1'], off=depth-sep if np.isfinite(sep) else np.nan,
        depth=depth, s2=gs['s2']))
  if verbose:
    print(f'  logr={r["log10ratio"]:+.0f} z={z}: {len(out)} bins', flush=True)
  return out


def main(step=14):
  rows = []
  for z, suf in ((4, '_deepband'), (1, '_z=1_deepband')):
    od = S.figdir('gammacm_sweep_data_rarcut_fc2' + suf)
    res = S.load_sweep(od)
    if not res:
      print(f'no deep sweep in {od}'); continue
    for r in sorted(res, key=lambda q: q['log10ratio']):
      rows += one_point(r, z, step=step)
  from segment_route import calib_path
  out = calib_path(f'band_offsets_{S.DEFAULT_KEY}.csv')
  with open(out, 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(FIELDS)); w.writeheader(); w.writerows(rows)
  print(f'{len(rows)} rows -> {out}')
  return rows


if __name__ == '__main__':
  t0 = time.time(); main(); print(f'{time.time()-t0:.0f} s')
