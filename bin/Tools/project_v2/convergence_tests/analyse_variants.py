"""Compare each full-shell variant run against the PRODUCTION cached sweep point."""
import os, sys, glob, json
from _paths import GAMMA_DIR, FIG   # resolves the repo root from __file__
import numpy as np, pandas as pd
import sweep_gammacm as S
OUT = os.path.dirname(os.path.abspath(__file__)) + '/results'
CACHE = S.method_outdir('data_rarcut', S.DEFAULT_KEY, S.Z_SHELL) + '/cache'
TRAPZ = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
pd.set_option('display.width', 200)

def lc(Tb, nub, nF, nu_t):
    j = int(np.argmin(np.abs(np.log(nub / nu_t))))
    y = nF[:, j]
    i = int(np.argmax(y))
    Fpk, Tpk = y[i], Tb[i]
    def cross(side):
        h = 0.5 * Fpk
        if side == 'r':
            idx = np.flatnonzero(y[:i + 1] <= h)
            if not idx.size: return np.nan
            a = idx[-1]
            return np.interp(h, [y[a], y[a + 1]], [Tb[a], Tb[a + 1]])
        idx = np.flatnonzero(y[i:] <= h)
        if not idx.size: return np.nan
        a = i + idx[0]
        return np.interp(-h, [-y[a - 1], -y[a]], [Tb[a - 1], Tb[a]])
    return Fpk, Tpk, cross('d') - cross('r')

def flu(Tb, nub, nF):
    f = TRAPZ(nF, Tb, axis=0)
    k = int(np.argmax(f))
    return f, nub[k], f[k]

rows = []
for p in sorted(glob.glob(f'{OUT}/V_*.npz')):
    v = np.load(p, allow_pickle=True)
    tag = str(v['tag']); logr = float(v['logr'])
    b = np.load(f'{CACHE}/point_logr={logr:+.1f}.npz', allow_pickle=True)
    Tb_b, nub_b, nF_b = b['Tb'], b['nub'], b['nuFnu']
    Tb_v, nub_v, nF_v = v['Tb'], v['nub'], v['nuFnu']
    r = dict(tag=tag, logC=logr, NT_prod=len(Tb_b), NT_var=len(Tb_v),
             dt_s=float(v['dt']),
             E_rad_rel=float(b['E_rad'] / v['E_rad'] - 1),
             eps_prod=float(b['E_rad'] / b['E_inj']),
             eps_var=float(v['E_rad'] / v['E_inj']))
    for nu_t in (0.01, 0.1, 1.0):
        Fb, Tp_b, W_b = lc(Tb_b, nub_b, nF_b, nu_t)
        Fv, Tp_v, W_v = lc(Tb_v, nub_v, nF_v, nu_t)
        r[f'dF_pk({nu_t:g})'] = Fb / Fv - 1
        r[f'dT_pk({nu_t:g})'] = (Tp_b - 1) / (Tp_v - 1) - 1
        r[f'dFWHM({nu_t:g})'] = W_b / W_v - 1
    fb, npk_b, fpk_b = flu(Tb_b, nub_b, nF_b)
    fv, npk_v, fpk_v = flu(Tb_v, nub_v, nF_v)
    fvi = np.interp(np.log(nub_b), np.log(nub_v), fv)
    m = (fb > fb.max() * 1e-4) & (fvi > 0)
    r['d_nu_pk_int'] = npk_b / npk_v - 1
    r['d_F_pk_int'] = fpk_b / fpk_v - 1
    r['spec_med'] = float(np.median(np.abs(fb[m] / fvi[m] - 1)))
    r['spec_max'] = float(np.max(np.abs(fb[m] / fvi[m] - 1)))
    rows.append(r)

if not rows:
    print('no variants finished yet'); sys.exit()
D = pd.DataFrame(rows)
json.dump(rows, open(f'{OUT}/variants_summary.json', 'w'), indent=1, default=float)
print('Production MINUS variant, as a relative difference (production is the reference):')
print()
cols1 = ['tag', 'logC', 'dt_s', 'E_rad_rel', 'eps_prod', 'eps_var']
print(D[cols1].to_string(index=False, float_format=lambda v: f'{v:12.5f}'))
print()
cols2 = ['tag'] + [c for c in D.columns if c.startswith(('dF_pk', 'dT_pk', 'dFWHM'))]
print(D[cols2].to_string(index=False, float_format=lambda v: f'{v:+9.4f}'))
print()
cols3 = ['tag', 'NT_prod', 'NT_var', 'd_nu_pk_int', 'd_F_pk_int', 'spec_med', 'spec_max']
print(D[cols3].to_string(index=False, float_format=lambda v: f'{v:+10.5f}'))
