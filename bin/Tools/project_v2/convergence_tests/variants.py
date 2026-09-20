"""Full-shell variant runs for the convergence appendix.

Baseline = the PRODUCTION cached sweep point (figures/fiducial/..._rarcut_fc2/cache),
so only the varied side has to be computed.  Each variant is written to its own npz
as soon as it finishes.  Ordered cheapest-first.
"""
import os, sys, time, json, traceback
from _paths import GAMMA_DIR, FIG   # resolves the repo root from __file__
import numpy as np
OUT = os.path.dirname(os.path.abspath(__file__)) + '/results'
os.makedirs(OUT, exist_ok=True)

import sweep_gammacm as S
from working_cooling_data import get_shell_nuFnu_fromData

KEY, Z = S.DEFAULT_KEY, S.Z_SHELL
NPROC = int(os.environ.get('VNPROC', '5'))
CACHE = S.method_outdir('data_rarcut', KEY, Z) + '/cache'

def log(*a): print(f'[{time.strftime("%H:%M:%S")}]', *a, flush=True)

def baseline(logr):
    d = np.load(f'{CACHE}/point_logr={logr:+.1f}.npz', allow_pickle=True)
    return dict(Tb=d['Tb'], nub=d['nub'], nuFnu=d['nuFnu'],
                E_rad=float(d['E_rad']), E_inj=float(d['E_inj']))

def variant(logr, tag, **over):
    alpha = S.compute_alpha_sweep(KEY, [logr])[0][0]
    lo, hi, Nnu = S._nu_window(KEY, alpha)
    kw = dict(alpha=alpha, Tmax=S.TMAX, NT=S.NT, Nnu=Nnu, lognu_min=lo, lognu_max=hi,
              Tb_min=S.TB_MIN, Tb_lin=S.TB_LIN, subcell_dlogT=S.SUBCELL_DLOGT,
              subcell_max=S.SUBCELL_MAX, r_ref=S.R_REF, return_energies=True,
              ncell_proc=NPROC, dlogT_max=S.DLOGT_MAX)
    kw.update(over)
    t0 = time.time()
    nuobs, Tobs, env, nuFnu, E_rad, E_int, E_inj = get_shell_nuFnu_fromData(
        KEY, Z, early_ana=S.EARLY_ANA, rar_cut='model', **kw)
    Tb = 1 + (Tobs - env.Ts) / env.T0
    nub = nuobs / max(env.nu0, env.nuc)
    np.savez_compressed(f'{OUT}/V_{tag}.npz', Tb=Tb, nub=nub, nuFnu=nuFnu,
                        E_rad=E_rad, E_inj=E_inj, logr=logr, dt=time.time() - t0,
                        tag=tag, over=json.dumps({k: str(v) for k, v in over.items()}))
    log(f'{tag}: {time.time()-t0:.0f}s  E_rad={E_rad:.6e}  eps={E_rad/E_inj:.5f}')
    return True

JOBS = [
    ('logr+3_cap0.02',  3.,  dict(dlogT_max=0.02)),
    ('logr+3_cap0.005', 3.,  dict(dlogT_max=0.005)),
    ('logr+1_NTx2',     1.,  dict(NT=2 * S.NT, Tb_lin=(0.5, 2.5, 534))),
    ('logr-3_NTx2',    -3.,  dict(NT=2 * S.NT, Tb_lin=(0.5, 2.5, 534))),
]

if __name__ == '__main__':
    only = sys.argv[1:] or None
    for tag, logr, over in JOBS:
        if only and tag not in only:
            continue
        if os.path.exists(f'{OUT}/V_{tag}.npz'):
            log(f'{tag}: already done, skip'); continue
        try:
            log(f'##### {tag} start (nproc={NPROC})')
            variant(logr, tag, **over)
        except Exception:
            log(f'##### {tag} FAILED'); traceback.print_exc()
    log('VARIANTS DONE')
