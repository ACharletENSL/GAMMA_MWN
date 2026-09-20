"""Test 8b: the free mid-slope, split by EPOCH.

test8_slopes.py reads free_slopes_by_regime.csv, which POOLS the rise, crossing and
HLE bins of a sweep point.  free_slopes_by_epoch.csv keeps them apart, and the split
changes the reading of the fast-cooling branch completely:

  - FC: the departure from 1/2 is ~0 at logC=-5, where the two breaks are far apart,
    and grows monotonically to +0.05 at logC=-2, where they are close.  That is the
    signature of the estimator tilting on a curved knee as the mid plateau shortens,
    not of a physical hardening.  (Cf. the production estimator's own calibrated tilt,
    +0.0138 in FC on-axis -- same size as the whole effect.)
  - SC: the softening is small on the rise and grows through crossing into the HLE,
    i.e. it is a post-crossing effect, which is what the main text describes.

The free estimator has no calibrated bias of its own: at the production standoff D=10
it still leaves a_lo 0.005-0.008 short of a 4/3 that is known to be exact
(a_lo_convergence.csv).  So it CORROBORATES a sign; it does not measure a departure.
"""
import pandas as pd, numpy as np

from _paths import GAMMA_DIR, FIG   # resolves the repo root from __file__
A_MID_FC, A_MID_SC = 0.5, (3 - 2.5)/2
A_HI = 1 - 2.5/2

for run in ('fiducial', 'hires'):
    f = pd.read_csv(f'{FIG}/{run}/slope_check/free_slopes_by_epoch.csv')
    f = f[(f.z == 4) & (f.n_mid >= 30)].copy()
    f['branch'] = np.where(f.logr <= -2, 'FC', np.where(f.logr >= 0, 'SC', '--'))
    f['th'] = np.where(f.branch == 'FC', A_MID_FC, A_MID_SC)
    f['d_mid'] = f.a_mid - f.th
    f['d_hi'] = f.a_hi - A_HI
    print('=' * 78)
    print(f'{run}   RS (z=4), free mid slope by epoch   [d_mid = a_mid - asymptote]')
    print('=' * 78)
    print(f[['logr', 'branch', 'epoch', 'n_mid', 'a_mid', 'a_mid_sd', 'd_mid', 'd_hi']]
          .to_string(index=False, float_format=lambda v: f'{v:+8.4f}'))
    for br in ('FC', 'SC'):
        g = f[f.branch == br]
        if not len(g):
            continue
        print(f'\n  {br}: d_mid over epochs  min {g.d_mid.min():+.4f}  max {g.d_mid.max():+.4f}')
        for ep in ('rise', 'crossing', 'HLE'):
            h = g[g.epoch == ep]
            if len(h):
                print(f'     {ep:>8}: {h.d_mid.min():+.4f} .. {h.d_mid.max():+.4f}'
                      f'   (|d_hi| up to {h.d_hi.abs().max():.4f})')
    # the FC trend with break separation is the whole point
    g = f[(f.branch == 'FC') & (f.epoch == 'rise')].sort_values('logr')
    if len(g):
        print(f'\n  FC rise, d_mid vs logC (separation shrinks as logC rises):')
        print('     ' + '  '.join(f'{r.logr:+.0f}:{r.d_mid:+.4f}' for r in g.itertuples()))
    print()
