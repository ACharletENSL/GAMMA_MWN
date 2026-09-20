# Does the dedicated fine efficiency grid give the SAME eps_rad as the flux sweep
# at the nine shared targets? If yes, the flux-sweep caches (which exist at BOTH
# resolutions) are a valid proxy for the efficiency curve's resolution error.
import os, glob
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

def eps(d):
    if 'E_rad' not in d.files: return float('nan')
    if 'E_inj' in d.files and float(d['E_inj']) > 0.: return float(d['E_rad'])/float(d['E_inj'])
    if 'E_int' in d.files and 'eps_e' in d.files: return float(d['E_rad'])/(float(d['eps_e'])*float(d['E_int']))
    return float('nan')

def read(pat):
    out = {}
    for f in glob.glob(pat):
        d = np.load(f); out[round(float(d['log10ratio']), 4)] = eps(d)
    return out

for label, fine_pat, flux_sub in [
    ('data  RS(z=4)', f'{FIG}/fiducial/efficiency_sweep/cache/z=4/point_logr=*.npz',
                      'gammacm_sweep_data_fc2'),
    ('data  FS(z=1)', f'{FIG}/fiducial/efficiency_sweep/cache/z=1/point_logr=*.npz',
                      'gammacm_sweep_data_fc2_z=1')]:
    fine = read(fine_pat)
    flux = read(f'{FIG}/fiducial/{flux_sub}/cache/point_logr=*.npz')
    common = sorted(set(fine) & set(flux))
    print(f'\n=== {label} ===   fine grid: {len(fine)} pts, flux sweep: {len(flux)} pts, shared: {len(common)}')
    devs = []
    for lr in common:
        dev = abs(fine[lr]-flux[lr])/flux[lr] if flux[lr] > 0 else np.nan
        devs.append(dev)
        print(f'  {lr:+6.1f}  fine={fine[lr]:.8f}  flux={flux[lr]:.8f}  rel={dev:.2e}')
    if devs: print(f'  -> worst {np.nanmax(devs):.2e}')
