"""Control: recompute one sweep point with EXACT production settings and compare
to its cached value.  Needed because running these tests added 7 per-cell fit
caches and triggered a rebuild of rarefaction_head_4.npz; if the control
reproduces the cache, every variant comparison against the cache is sound."""
import os, sys, time, json
from _paths import GAMMA_DIR, FIG   # resolves the repo root from __file__
import numpy as np
from variants import variant, baseline, OUT
import sweep_gammacm as S
if __name__ == '__main__':
    tag = 'control_logr+3_production'
    if not os.path.exists(f'{OUT}/V_{tag}.npz'):
        variant(3., tag)
    v = np.load(f'{OUT}/V_{tag}.npz', allow_pickle=True)
    b = baseline(3.)
    m = b['nuFnu'] > b['nuFnu'].max() * 1e-6
    r = dict(E_rad_rel=float(v['E_rad'] / b['E_rad'] - 1),
             nuFnu_med=float(np.median(np.abs(v['nuFnu'][m] / b['nuFnu'][m] - 1))),
             nuFnu_max=float(np.max(np.abs(v['nuFnu'][m] / b['nuFnu'][m] - 1))),
             same_grid=bool(len(v['Tb']) == len(b['Tb'])))
    print('CONTROL vs cache:', json.dumps(r, indent=1), flush=True)
    json.dump(r, open(f'{OUT}/control.json', 'w'), indent=1)
