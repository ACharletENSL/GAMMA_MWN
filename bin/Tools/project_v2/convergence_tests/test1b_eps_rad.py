# eps_rad across resolutions, from the cached flux-sweep points alone (no recompute).
# Both keys are read HERE, on the cluster, so the two sides share a code vintage
# (the local _fc2 caches predate the leading-edge onset fix; these do not).
import os, glob, datetime
import numpy as np

from _paths import GAMMA_DIR, FIG   # resolves the repo root from __file__

# --- CLUSTER ONLY -------------------------------------------------------------
# This compares the fiducial against the hi-res caches. The LOCAL _fc2 point
# caches predate the leading-edge onset fix while the HPC ones do not, so reading
# one side locally and one side remotely -- or both sides locally -- silently
# compares two code vintages and the answer is wrong in a way nothing flags.
# Run it on openuHPC. Set CONV_ALLOW_LOCAL=1 only if you have checked the caches.
if not (os.path.realpath(GAMMA_DIR).startswith('/RG/')
        or os.environ.get('CONV_ALLOW_LOCAL')):
    raise SystemExit(
        f'REFUSING TO RUN: {os.path.basename(__file__)} must run on the cluster.\n'
        f'  GAMMA_DIR resolves to {GAMMA_DIR}, which is not the HPC tree.\n'
        '  The local _fc2 caches predate the leading-edge onset fix; comparing\n'
        '  them against the hi-res set mixes code vintages. Run this on openuHPC\n'
        '  (ssh openuHPC, then python3 <this file>), or export CONV_ALLOW_LOCAL=1\n'
        '  if you have verified both sides share a vintage.')
# ------------------------------------------------------------------------------

def eps_rad(d):
    """E_rad/E_inj, the same definition as sweep_gammacm.compute_efficiency."""
    if 'E_rad' not in d.files:
        return float('nan')
    E_rad = float(d['E_rad'])
    if 'E_inj' in d.files and float(d['E_inj']) > 0.:
        return E_rad / float(d['E_inj'])
    if 'E_int' in d.files and 'eps_e' in d.files and float(d['E_int']) > 0.:
        return E_rad / (float(d['eps_e']) * float(d['E_int']))
    return float('nan')

def read(key, sub):
    out = {}
    for f in glob.glob(os.path.join(FIG, key, sub, 'cache', 'point_logr=*.npz')):
        d = np.load(f)
        lr = round(float(d['log10ratio']), 4)
        out[lr] = dict(eps=eps_rad(d),
                       E_rad=float(d['E_rad']) if 'E_rad' in d.files else np.nan,
                       E_inj=float(d['E_inj']) if 'E_inj' in d.files else np.nan,
                       mtime=datetime.date.fromtimestamp(os.path.getmtime(f)).isoformat(),
                       key_tag=str(d['key']) if 'key' in d.files else '',
                       method=str(d['method']) if 'method' in d.files else '')
    return out

METHODS = [('data',        'gammacm_sweep_data_fc2'),
           ('data_rarcut', 'gammacm_sweep_data_rarcut_fc2')]
SHELLS  = [('RS (z=4)', ''), ('FS (z=1)', '_z=1')]

worst_all = {}
for mname, msub in METHODS:
    for sname, zsuf in SHELLS:
        sub = msub + zsuf
        lo = read('fiducial', sub)
        hi = read('hires',    sub)
        common = sorted(set(lo) & set(hi))
        print(f'\n=== {mname}  {sname}   [{sub}] ===')
        print(f'{"logC":>6} {"eps(500)":>12} {"eps(1e4)":>12} {"rel.diff":>11}   '
              f'{"tag500":>18} {"tag1e4":>22}')
        devs = []
        for lr in common:
            a, b = lo[lr], hi[lr]
            dev = abs(b['eps'] - a['eps'])/a['eps'] if a['eps'] > 0 else np.nan
            devs.append(dev)
            print(f'{lr:+6.1f} {a["eps"]:12.6f} {b["eps"]:12.6f} {dev:11.2e}   '
                  f'{a["key_tag"]:>18} {b["key_tag"]:>22}')
        if devs:
            w = np.nanmax(devs)
            worst_all[(mname, sname)] = w
            print(f'  -> max |d eps/eps| = {w:.3e}  ({100*w:.3f} %)   over {len(devs)} points')
        # cache dates, to show the two sides are the same vintage
        print(f'  cache dates: 500 -> {sorted({v["mtime"] for v in lo.values()})}')
        print(f'               1e4 -> {sorted({v["mtime"] for v in hi.values()})}')

print('\n================ SUMMARY ================')
for k, v in worst_all.items():
    print(f'{k[0]:>12}  {k[1]:>9}   max |Delta eps_rad/eps_rad| = {100*v:.3f} %')
if worst_all:
    print(f'\nOVERALL MAX = {100*max(worst_all.values()):.3f} %')
