"""Test 8: spectral-measurement robustness on ACTUAL model spectra.

The production measurement HOLDS a_lo = 4/3 and a_hi = 1 - p/2 and selects the
windows by thresholding the local slope around those values, so it cannot check
them.  free_slopes imposes nothing: it places a window a fixed factor D in
frequency away from a break located by an independent fit and fits a free line.
Comparing the two on the computed spectra is the non-circular test.
"""
import pandas as pd, numpy as np
from _paths import GAMMA_DIR, FIG   # resolves the repo root from __file__
pd.set_option('display.width', 220)
P_SYN = 2.5
A_LO, A_HI = 4/3, 1 - P_SYN/2
A_MID_FC, A_MID_SC = 0.5, (3 - P_SYN)/2

print('='*104)
print('8a. THE HELD LOW-ENERGY ASYMPTOTE, MEASURED FREE ON THE COMPUTED SPECTRA')
print('    a_lo fitted in a window pushed a factor D away from the break (held value 4/3 = 1.3333)')
print('='*104)
rows = []
for run in ('fiducial', 'hires'):
    a = pd.read_csv(f'{FIG}/{run}/slope_check/a_lo_convergence.csv')
    a = a[(a.z == 4) | a.z.isna()]
    for _, r in a.iterrows():
        lab = ('synthetic ' + str(r['kind']).split('synthetic')[-1].strip()
               if 'synthetic' in str(r['kind']) else f"computed logC={r['logr']:+.0f}")
        rows.append(dict(run=run, case=lab,
                         **{f'D={d}': r[f'd{d}'] - A_LO for d in (3, 10, 30, 100, 300)}))
d = pd.DataFrame(rows).drop_duplicates(subset=['run', 'case'])
print(d.to_string(index=False, float_format=lambda v: f'{v:+8.4f}'), '\n(values are a_lo - 4/3)')

print()
print('='*104)
print('8b. FREE vs HELD SLOPES BY REGIME (RS shell).  Held: a_lo=4/3, a_hi=1-p/2=-0.25.')
print('='*104)
out = []
for run in ('fiducial', 'hires'):
    f = pd.read_csv(f'{FIG}/{run}/slope_check/free_slopes_by_regime.csv')
    f = f[f.z == 4]
    for _, r in f.iterrows():
        if r['regime'] in ('VFC',) and not np.isfinite(r.get('a_lo', np.nan)):
            pass
        th_mid = (A_MID_FC if r['regime'] in ('FC', 'VFC', 'FC*') else
                  A_MID_SC if r['regime'] in ('SC', 'VSC') else np.nan)
        out.append(dict(run=run, logC=r['logr'], regime=r['regime'], n=r['n_bins'],
                        d_a_lo=r['a_lo'] - A_LO if np.isfinite(r['a_lo']) else np.nan,
                        sd_lo=r['a_lo_sd'],
                        d_a_hi=r['a_hi'] - A_HI if np.isfinite(r['a_hi']) else np.nan,
                        sd_hi=r['a_hi_sd'],
                        d_a_mid=r['a_mid'] - th_mid if np.isfinite(r.get('a_mid', np.nan)) else np.nan,
                        sd_mid=r['a_mid_sd']))
O = pd.DataFrame(out)
O = O[O.n >= 30]
print(O.to_string(index=False, float_format=lambda v: f'{v:+8.4f}'))

print()
print('='*104)
print('8c. DOES THE FREE MEASUREMENT AGREE BETWEEN THE TWO RESOLUTIONS?')
print('='*104)
m = O[O.run == 'fiducial'].merge(O[O.run == 'hires'], on=['logC', 'regime'],
                                 suffixes=('_f', '_h'))
for c in ('d_a_lo', 'd_a_mid', 'd_a_hi'):
    v = (m[c + '_f'] - m[c + '_h']).abs()
    v = v[np.isfinite(v)]
    if len(v):
        print(f'  |{c}(500 cells) - {c}(1e4 cells)|:  median {v.median():.4f}   max {v.max():.4f}   n={len(v)}')
