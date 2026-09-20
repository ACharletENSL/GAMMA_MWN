"""Test 7: does the moving-mesh density filter change the radiation?

The filter (running median of ln rho over 9 cells, dx rescaled so rho*dx is
preserved) is applied ONCE, at cell extraction.  Here the same cells are
re-extracted with the filter ON (production, window=9), OFF (window=None) and
with a substantially different width (window=25), WITHOUT touching the cached
cell files (savefile=False), and each variant is carried through the single-cell
emission path to a spectrum and a light curve.
"""
import os, sys, json, time
from _paths import GAMMA_DIR, FIG   # resolves the repo root from __file__
import numpy as np, pandas as pd
OUT = os.path.dirname(os.path.abspath(__file__)) + '/results'
os.makedirs(OUT, exist_ok=True)

from environment import MyEnv
from analysis_hydro import extract_data_cells, RHO_SMOOTH_WINDOW, RHO_SMOOTH_ROUGH
from working_cooling import generate_cell_withDistrib, get_nuFnu
from working_cooling import get_Fnu_cell_evolving
from IO import open_rundata
from fits_hydro import cellsBehindShock_fromData
from sweep_gammacm import compute_alpha_sweep
import prerar_model as M

KEY, Z = 'cooling_g100', 4
TRAPZ = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
def log(*a): print(f'[{time.strftime("%H:%M:%S")}]', *a, flush=True)

CELLS = [int(k) for k in (260, 300, 340, 380, 420, 460, 498)]
VARIANTS = [('prod w=9', 9), ('no filter', None), ('w=25', 25)]

def extract(window):
    t0 = time.time()
    out = extract_data_cells(KEY, CELLS, savefile=False, noOut=False,
                             rho_smooth_window=window,
                             rho_smooth_rough=RHO_SMOOTH_ROUGH, nproc=4)
    log(f'  extraction window={window}: {time.time()-t0:.0f}s, {len(out)} cells')
    return out

def main():
    raws = {}
    for lab, w in VARIANTS:
        raws[lab] = extract(w)
    shr = cellsBehindShock_fromData(open_rundata(KEY, Z))
    ex = shr.loc[shr.t.idxmax()]
    env0 = MyEnv(KEY)
    rows = []
    for logr in (-3., 0., 3.):
        alpha = float(compute_alpha_sweep(KEY, np.array([float(logr)]))[0][0])
        for ic, k in enumerate(CELLS):
            spec, lc, Er = {}, {}, {}
            for lab, _ in VARIANTS:
                raw = raws[lab][ic]
                if not isinstance(raw, pd.DataFrame):
                    raw = pd.DataFrame(raw)
                d0 = shr.loc[shr.i == raw.iloc[0].i].iloc[0]
                cell, env = generate_cell_withDistrib(raw, d0, env0, alpha=alpha,
                        r_ref=1.1, Tmax=None, key=KEY, k=k, exit_row=ex)
                nuobs = np.geomspace(1e-3, 1e4, 120) * env.nu0
                Tobs = env.Ts + (np.geomspace(1e-2, 50., 160) - 1) * env.T0
                nF = get_nuFnu(get_Fnu_cell_evolving, nuobs, Tobs, cell, env)
                spec[lab] = TRAPZ(nF, Tobs, axis=0)          # fluence spectrum
                lc[lab] = nF[:, np.argmin(np.abs(nuobs/env.nu0 - 1.))]
                Er[lab] = float(np.nansum(nF))
            ref = spec['prod w=9']
            for lab, _ in VARIANTS[1:]:
                m = (ref > ref.max()*1e-4) & (spec[lab] > 0)
                lm = (lc['prod w=9'] > lc['prod w=9'].max()*1e-2) & (lc[lab] > 0)
                rows.append(dict(logr=logr, k=k, variant=lab,
                    spec_med=float(np.median(np.abs(np.log(spec[lab][m]/ref[m])))),
                    spec_max=float(np.max(np.abs(np.log(spec[lab][m]/ref[m])))),
                    lc_med=float(np.median(np.abs(np.log(lc[lab][lm]/lc['prod w=9'][lm])))),
                    lc_max=float(np.max(np.abs(np.log(lc[lab][lm]/lc['prod w=9'][lm]))))))
        log(f'  logr={logr:+.0f} done')
    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT}/T7_filter.csv', index=False)
    pd.set_option('display.width', 200)
    print()
    print('|d ln| of the single-cell FLUENCE SPECTRUM and of the nu_0 LIGHT CURVE,')
    print('against the production filter (window = 9 cells):')
    print()
    g = df.groupby(['logr', 'variant']).agg(
        spec_med=('spec_med', 'median'), spec_max=('spec_max', 'max'),
        lc_med=('lc_med', 'median'), lc_max=('lc_max', 'max'), n=('k', 'count'))
    print(g.to_string(float_format=lambda v: f'{v:10.5f}'))

if __name__ == '__main__':
    main()
