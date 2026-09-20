"""Test 2: observer-frame energy closure.

The pipeline implies, for one cooling step emitting as a flash at its own onset
with a tT^-2 high-latitude tail,

    int dT int dnu  Delta F_nu  =  zdl * (1/2) * delta * Delta E'   * (1 - tau_max^-2)

with zdl = (1+z)/(4 pi d_L^2), delta the step's Doppler factor and Delta E' its
COMOVING radiated energy; the last factor is the part of the tau^-3 tail that
falls inside the observer window.  Summing over every emitter and every step:

    E_obs == 0.5 * sum_j delta_j * Delta E'_j * (1 - tau_max,j^-2),
    E_obs  = (1/zdl) * int int F_nu dnu dT.

The left side comes from the production flux kernel, the right from the
independent comoving budget (step_radiated_energy), so the test closes the loop
on Doppler factors, EATS weighting, both observer grids and the window.

Run on a CONTIGUOUS mid-shell block (real neighbours, so the sub-cell path
behaves) plus, separately, the CD-adjacent block where sub-cells actually fire.
"""
import os, sys, time, json, traceback
from _paths import GAMMA_DIR, FIG   # resolves the repo root from __file__
import numpy as np
OUT = os.path.dirname(os.path.abspath(__file__)) + '/results'
os.makedirs(OUT, exist_ok=True)

import sweep_gammacm as S
import working_cooling_data as W
from working_cooling import precompute_step_cols, step_view, norm_plaw_distrib
from radiation_cooling import step_radiated_energy
from environment import MyEnv

KEY, Z = S.DEFAULT_KEY, S.Z_SHELL
TRAPZ = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
def log(*a): print(f'[{time.strftime("%H:%M:%S")}]', *a, flush=True)

_B = dict(on=False)
_orig = W._accum_energy
def _patched(ctx, acc, iv, cell, cell_env):
    _orig(ctx, acc, iv, cell, cell_env)
    if iv != 0 or not _B['on']:
        return
    c0 = cell.iloc[0]
    K0 = norm_plaw_distrib(c0.gmin, c0.gmax, cell_env.psyn)
    cols = precompute_step_cols(cell, cell_env,
                                keys=('nup_B', 'V3p', 'Pmax', 'Dop', 'obsT'))
    nupB, V3p, D = cols['nup_B'], cols['V3p'], cols['Dop']
    Ton, Tth, Tej = cols['obsT']
    Tmax_obs = _B['Tmax_obs']
    ed = ew = eb = 0.
    bad = 0
    for j in range(len(cell)):
        e = step_radiated_energy(step_view(cols, j), K0, cell_env, 120, 1.01) \
            * nupB[j] * V3p[j]
        if not np.isfinite(e):
            bad += 1
            continue
        tau_max = 1. + (Tmax_obs - Ton[j]) / Tth[j]
        w = 1. - tau_max**-2 if tau_max > 1. else 0.
        ed += D[j] * e
        ew += D[j] * e * w
        eb += e
    _B['Edop'] += ed; _B['Ewin'] += ew; _B['Ebare'] += eb
    _B['n'] += 1; _B['bad'] += bad; _B['tot'] += len(cell)
W._accum_energy = _patched


def shell(logr, klist, **over):
    alpha = S.compute_alpha_sweep(KEY, [logr])[0][0]
    lo, hi, Nnu = S._nu_window(KEY, alpha)
    kw = dict(alpha=alpha, Tmax=S.TMAX, NT=S.NT, Nnu=Nnu, lognu_min=lo, lognu_max=hi,
              Tb_min=S.TB_MIN, Tb_lin=S.TB_LIN, subcell_dlogT=None,
              subcell_max=S.SUBCELL_MAX, r_ref=S.R_REF, return_energies=True,
              ncell_proc=1, dlogT_max=S.DLOGT_MAX, klist=list(klist), norm=False)
    kw.update(over)
    return W.get_shell_nuFnu_fromData(KEY, Z, early_ana=S.EARLY_ANA,
                                      rar_cut='model', **kw)


def closure(logr, klist, label):
    e0 = MyEnv(KEY)
    _B.update(on=True, Edop=0., Ewin=0., Ebare=0., n=0, bad=0, tot=0, Tmax_obs=np.inf)
    # first pass only to learn the observer window; cheap to set from env directly
    alpha = S.compute_alpha_sweep(KEY, [logr])[0][0]
    from environment import rescale_hydro
    env_a = rescale_hydro(alpha, 1., e0) if alpha != 1. else e0
    _B['Tmax_obs'] = env_a.Ts + (S.TMAX - 1) * env_a.T0
    t0 = time.time()
    nuobs, Tobs, env, nuFnu, E_rad, E_int, E_inj = shell(logr, klist)
    _B['on'] = False
    # norm=False  =>  stored = nuobs * F_nu (physical)  =>  int F dnu = int stored dln nu
    inner = TRAPZ(nuFnu, np.log(nuobs), axis=1)
    S_flu = TRAPZ(inner, Tobs)
    E_obs = S_flu / float(env.zdl)
    out = dict(label=label, logr=logr, n_emit=_B['n'], bad_steps=_B['bad'], tot_steps=_B['tot'],
               E_obs=E_obs, E_pred_win=_B['Ewin'], E_pred_full=_B['Edop'],
               rel_win=(E_obs - _B['Ewin']) / _B['Ewin'],
               rel_full=(E_obs - _B['Edop']) / _B['Edop'],
               window_loss=1 - _B['Ewin'] / _B['Edop'],
               E_rad_driver=float(E_rad), E_bare_mine=_B['Ebare'],
               eps_rad=float(E_rad / E_inj) if E_inj else np.nan,
               dt=time.time() - t0, Tmax_obs=_B['Tmax_obs'])
    log(json.dumps(out, default=float))
    return out


if __name__ == '__main__':
    e = MyEnv(KEY)
    k4, kCD = e.Next, e.Next + e.Nsh4
    mid = list(range(k4 + 230, k4 + 270))          # 40 contiguous mid-shell cells
    cd  = list(range(kCD - 12, kCD))               # 12 cells next to the CD (sub-cells fire)
    res = []
    for logr in (0., 3., -3.):
        for lab, ks in (('mid-shell x40', mid), ('CD-adjacent x12', cd)):
            try:
                res.append(closure(logr, ks, lab))
            except Exception:
                log(f'FAILED logr={logr} {lab}'); traceback.print_exc()
            json.dump(res, open(f'{OUT}/A_closure3.json', 'w'), indent=1, default=float)
    log('CLOSURE DONE')
