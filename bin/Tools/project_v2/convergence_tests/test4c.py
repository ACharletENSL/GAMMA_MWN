"""4c: does the settling table's resolution change the PRODUCTION reconstruction
(prerar_history, anchored on the last real row)?"""
import os, sys
from _paths import GAMMA_DIR, FIG   # resolves the repo root from __file__
import numpy as np, pandas as pd
import prerar_model as M, prerar_cell_evolution as P
from working_cooling_data import select_postshock_rows
from IO import open_celldata
FINE, COARSE, Z = 'cooling_g100_w2', 'cooling_g100_w2_lores', 4
tab0 = M.load_table(M.TABLE_KEY, Z, M.NCELLS)
tabF = M.with_settle(tab0, FINE, Z, M.NCELLS)
tabC = M.with_settle(tab0, COARSE, Z, M.NCELLS)
ks = M.table_cells(FINE, Z, M.NCELLS)
ks = list(np.asarray(ks)[np.linspace(0, len(ks)-1, 12).astype(int)])
rows = []
for k in ks:
    try:
        sh = select_postshock_rows(open_celldata(FINE, int(k)), 1)
        hF, iF = M.prerar_history(sh, tabF, z=Z)
        hC, iC = M.prerar_history(sh, tabC, z=Z)
    except Exception as e:
        rows.append(dict(k=int(k), status=f'err {type(e).__name__}')); continue
    if iF['status'] != 'ok' or iC['status'] != 'ok':
        rows.append(dict(k=int(k), status=f"{iF['status']}/{iC['status']}")); continue
    n = iF['n_syn']
    d = {}
    for c in ('rho', 'p', 'vx'):
        a = hF[c].to_numpy(float)[-n:]; b = hC[c].to_numpy(float)[-n:]
        d['max|dln '+c+'|'] = float(np.max(np.abs(np.log(np.abs(a)/np.abs(b)))))
    rows.append(dict(k=int(k), status='ok', n_syn=n, **d))
df = pd.DataFrame(rows)
pd.set_option('display.width', 200)
print(df.to_string(index=False, float_format=lambda v: f'{v:.3e}'))
ok = df[df.status == 'ok']
if len(ok):
    print()
    for c in [c for c in ok.columns if c.startswith('max')]:
        print(f'  {c}: median {ok[c].median():.3e}  worst {ok[c].max():.3e}')
