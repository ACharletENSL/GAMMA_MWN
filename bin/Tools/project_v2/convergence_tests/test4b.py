import os, sys
from _paths import GAMMA_DIR, FIG   # resolves the repo root from __file__
import numpy as np, pandas as pd
import prerar_model as M
FINE, COARSE, Z = 'cooling_g100_w2', 'cooling_g100_w2_lores', 4
pd.set_option('display.width', 220)
_orig = M.with_settle
def force(k):
    return lambda tab, key, z=4, ncells=M.NCELLS: _orig(tab, k, z, ncells)

frames = {}
for lab, skey in (('own', FINE), ('imported', COARSE)):
    M.with_settle = force(skey)
    r = M.validate_model(table_key=M.TABLE_KEY, test_key=FINE, z=Z, ncells=M.NCELLS,
                         local_settle=True, anchor='measured', verbose=False)
    frames[lab] = r[0] if isinstance(r, tuple) else r
M.with_settle = _orig
a, b = frames['own'], frames['imported']
print('columns:', list(a.columns), '  rows:', len(a), len(b))
num = [c for c in a.columns if a[c].dtype.kind == 'f']
print()
print('Reconstruction error |d ln| on the FINE run, settling table taken from the run')
print('itself vs from its half-resolution twin.  Median over all cells and radii,')
print('and split by R/R_i decade.')
print()
rows = []
for c in num:
    if c in ('Ri', 'dex_grid', 'dex', 'k', 'k_a'):
        continue
    x, y = a[c].to_numpy(float), b[c].to_numpy(float)
    if len(x) != len(y):
        continue
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10:
        continue
    rows.append(dict(quantity=c, med_own=np.median(np.abs(x[ok])),
                     med_imported=np.median(np.abs(y[ok])),
                     med_change=np.median(np.abs(y[ok])) - np.median(np.abs(x[ok])),
                     med_abs_pointwise_diff=np.median(np.abs(y[ok] - x[ok]))))
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f'{v:11.6f}'))
