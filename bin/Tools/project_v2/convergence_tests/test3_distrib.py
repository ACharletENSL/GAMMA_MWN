"""Test 3: independent validation of the cooled electron distribution.

The production code evolves the distribution by OPERATOR SPLITTING -- an exact
synchrotron step over Delta t~ followed by the adiabatic factor
(rho_{j+1}/rho_j)^{1/3} -- and then asserts the SHAPE

    dn/dgamma  propto  gamma^-p (1 - gamma t~_eff)^{p-2},
    t~_eff = (1 - b_syn)/gamma_M.

Here the same cell history is solved WITHOUT either assumption: in the code's own
normalised time t~ the cooling ODE (Eq. ODE_cooling) is unit-free,

    dgamma/dt~ = -gamma^2 + (gamma/3) dln(rho)/dt~,

so a set of characteristics gamma(t~; gamma_i) can be integrated with a stiff
solver on the cell's actual rho(t~), and the distribution follows from number
conservation along them, dn/dgamma = dn/dgamma_i * |dgamma_i/dgamma|.

Compared: the bounds gamma_m, gamma_M; the SHAPE of dn/dgamma; and the
synchrotron emissivity formed from each with the production kernel.
"""
import os, sys, json
from _paths import GAMMA_DIR, FIG   # resolves the repo root from __file__
import numpy as np
from scipy.integrate import solve_ivp
OUT = os.path.dirname(os.path.abspath(__file__)) + '/results'
os.makedirs(OUT, exist_ok=True)

from environment import MyEnv
from working_cooling import generate_cell_withDistrib
from cooling_distribution import distrib_plaw_cooled
from radiation_cooling import cooled_tt_eff, syn_emiss_exact
from IO import open_celldata, open_rundata
from fits_hydro import cellsBehindShock_fromData
from sweep_gammacm import compute_alpha_sweep
import prerar_model as M

KEY, Z = 'cooling_g100', 4
TRAPZ = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz


def build(logr, Ri_target=1.05, r_ref=1.1):
    cells = M._test_cells(KEY, Z, 120)
    k = int(min(cells, key=lambda c: abs(c[4] - Ri_target))[0])
    raw = open_celldata(KEY, k)
    shr = cellsBehindShock_fromData(open_rundata(KEY, Z))
    d0 = shr.loc[shr.i == raw.iloc[0].i].iloc[0]
    ex = shr.loc[shr.t.idxmax()]
    alpha = float(compute_alpha_sweep(KEY, np.array([float(logr)]))[0][0])
    cell, env = generate_cell_withDistrib(raw, d0, MyEnv(KEY), alpha=alpha,
                                          r_ref=r_ref, Tmax=None, key=KEY, k=k,
                                          exit_row=ex)
    return cell, env, k


def run(logr, Ng=400, rtol=1e-10):
    cell, env, k = build(logr)
    p = env.psyn
    tt = np.concatenate(([0.], np.cumsum(cell['dtt'].to_numpy(float))))[:len(cell)]
    rho = cell['rho'].to_numpy(float)
    ok = np.isfinite(tt) & np.isfinite(rho) & (rho > 0)
    tt, rho = tt[ok], rho[ok]
    # strictly increasing t~ for the interpolant
    keep = np.concatenate(([True], np.diff(tt) > 0))
    tt, rho = tt[keep], rho[keep]
    lnrho = np.log(rho)
    dlnrho_dtt = np.gradient(lnrho, tt)

    g0min = float(cell.iloc[0].gmin)
    g0max = float(cell.iloc[0].gmax)
    gi = np.geomspace(g0min, g0max, Ng)

    def rhs(t, g):
        d = np.interp(t, tt, dlnrho_dtt)
        return -g * g + g * d / 3.

    sol = solve_ivp(rhs, (tt[0], tt[-1]), gi, method='LSODA',
                    rtol=rtol, atol=1e-12, dense_output=True)
    if not sol.success:
        return dict(logr=logr, error=sol.message)

    rows = []
    idx = [int(f * (len(cell) - 1)) for f in (0.05, 0.25, 0.5, 0.75, 0.95)]
    for j in idx:
        t_j = float(np.interp(j, np.arange(len(tt)), tt)) if j < len(tt) else tt[-1]
        g = sol.sol(t_j)                        # characteristics at this time
        g = np.maximum(g, 1e-30)
        order = np.argsort(g)
        gs, gis = g[order], gi[order]
        # dn/dgamma from number conservation along characteristics
        dn_i = gis**(-p)
        dgi_dg = np.gradient(gis, gs)
        dn_num = dn_i * np.abs(dgi_dg)
        # production analytic shape at the same t~
        gmin_p, gmax_p = float(cell.iloc[j].gmin), float(cell.iloc[j].gmax)
        bsyn = float(cell.iloc[j].get('bsyn', 0.))
        tte = cooled_tt_eff(gmax_p, bsyn)
        m = (gs >= gmin_p * 1.001) & (gs <= gmax_p * 0.999) & (gs > 1.)
        if m.sum() < 20:
            continue
        dn_ana = distrib_plaw_cooled(gs[m], p, tte)
        a = dn_num[m] / TRAPZ(dn_num[m], gs[m])
        b = dn_ana / TRAPZ(dn_ana, gs[m])
        shape_dev = np.abs(np.log(a / b))
        # emissivity from each, same kernel
        tnu = np.geomspace(1e-3, 3e2, 90)
        Ea = np.array([TRAPZ(a * syn_emiss_exact(gs[m], t), gs[m]) for t in tnu])
        Eb = np.array([TRAPZ(b * syn_emiss_exact(gs[m], t), gs[m]) for t in tnu])
        good = (Ea > Ea.max() * 1e-6) & (Eb > Eb.max() * 1e-6)
        rows.append(dict(step=j, tt=t_j,
                         gmin_num=float(gs[gs > 1.].min()), gmin_prod=gmin_p,
                         gmax_num=float(gs.max()), gmax_prod=gmax_p,
                         d_gmax=float(np.log(gs.max() / gmax_p)),
                         d_gmin=float(np.log(max(gs[gs > 1.].min(), 1e-30) / gmin_p)),
                         shape_med=float(np.median(shape_dev)),
                         shape_p95=float(np.percentile(shape_dev, 95)),
                         emiss_med=float(np.median(np.abs(np.log(Ea[good] / Eb[good])))),
                         emiss_max=float(np.max(np.abs(np.log(Ea[good] / Eb[good]))))))
    return dict(logr=logr, k=k, n_steps=len(cell), rows=rows)


if __name__ == '__main__':
    res = []
    for logr in (-3., 0., 3.):
        r = run(logr)
        res.append(r)
        print(f"=== log10(C) = {logr:+.0f}   cell k={r.get('k')}  steps={r.get('n_steps')}")
        if 'error' in r:
            print('   ERROR', r['error']); continue
        print(f"{'step':>6} {'t~':>11} {'dln gmax':>10} {'dln gmin':>10} "
              f"{'shape med':>10} {'shape p95':>10} {'emiss med':>10} {'emiss max':>10}")
        for x in r['rows']:
            print(f"{x['step']:>6} {x['tt']:>11.4e} {x['d_gmax']:>10.5f} "
                  f"{x['d_gmin']:>10.5f} {x['shape_med']:>10.5f} {x['shape_p95']:>10.5f} "
                  f"{x['emiss_med']:>10.5f} {x['emiss_max']:>10.5f}")
        print()
    json.dump(res, open(f'{OUT}/T3_distrib.json', 'w'), indent=1, default=float)
