"""Test 4: is the post-shock SETTLING correction of the rarefaction-free
reconstruction resolution-dependent?  Pair: cooling_g100_w2 (Nsh1=1000) vs
cooling_g100_w2_lores (Nsh1=500), identical physics and shell width."""
import os, sys
from _paths import GAMMA_DIR, FIG   # resolves the repo root from __file__
import numpy as np, pandas as pd
import prerar_model as M
import prerar_cell_evolution as P

FINE, COARSE, Z = 'cooling_g100_w2', 'cooling_g100_w2_lores', 4
pd.set_option('display.width', 200)

print('='*90); print('4a. SETTLING TABLES AT TWO RESOLUTIONS (same shell width)'); print('='*90)
tf = M._settle_table(FINE, Z, M.NCELLS)
tc = M._settle_table(COARSE, Z, M.NCELLS)
print(f'{FINE}:   {len(tf)} cells, R_i/R_0 in [{tf.Ri.min():.3f}, {tf.Ri.max():.3f}]')
print(f'{COARSE}: {len(tc)} cells, R_i/R_0 in [{tc.Ri.min():.3f}, {tc.Ri.max():.3f}]')
print('columns:', list(tf.columns))

lo, hi = max(tf.Ri.min(), tc.Ri.min()), min(tf.Ri.max(), tc.Ri.max())
m = (tf.Ri >= lo) & (tf.Ri <= hi)
rows = []
for col, lab in (('sD', "settling ratio Delta'"), ('sp', 'settling ratio p'),
                 ('sG', 'settling ratio Gamma')):
    if col not in tf.columns:
        continue
    a = tf.loc[m, col].to_numpy()
    b = np.interp(tf.loc[m, 'Ri'].to_numpy(), tc.Ri.to_numpy(), tc[col].to_numpy())
    d = a - b                                   # these are ln-ratios
    rows.append(dict(quantity=lab, n=int(m.sum()),
                     fine_med=float(np.median(a)), coarse_med=float(np.median(b)),
                     med_absdiff=float(np.median(np.abs(d))),
                     q84_absdiff=float(np.percentile(np.abs(d), 84)),
                     signed_bias=float(np.median(d)),
                     frac_same_sign=float(np.mean(np.sign(d) == np.sign(np.median(d))))))
S = pd.DataFrame(rows)
print(); print(S.to_string(index=False, float_format=lambda v: f'{v:10.5f}'))

print()
print('='*90)
print('4b. DOES IMPORTING THE WRONG-RESOLUTION SETTLING TABLE CHANGE THE RECONSTRUCTION?')
print('    validate_model on the FINE run, with its own settling table vs the COARSE one.')
print('='*90)
out = {}
for lab, settle_key in (('own (fine)', FINE), ('imported (coarse)', COARSE)):
    tab = M.load_table(M.TABLE_KEY, Z, M.NCELLS)
    tab = M.with_settle(tab, settle_key, Z, M.NCELLS)
    try:
        res = M.validate_model(table_key=M.TABLE_KEY, test_key=FINE, z=Z,
                               ncells=M.NCELLS, local_settle=False, anchor='measured',
                               verbose=False, _tab=tab)
    except TypeError:
        res = None
    out[lab] = res
    print(f'  {lab}: {"ok" if res is not None else "validate_model has no table override"}')
if out.get('own (fine)') is None:
    print('  -> falling back to comparing the settling tables only (4a), plus alpha_D (4c).')

print()
print('='*90); print('4c. alpha_D RESOLUTION CONVERGENCE (the existing probe, for reference)')
print('='*90)
df, summ = P.compare_runs(FINE, COARSE, z=Z, ncells=P.NCELLS, verbose=True)
