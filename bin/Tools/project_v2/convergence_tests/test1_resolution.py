"""Test 1: 500-cell vs 10^4-cell sweep, compared on the paper's observables."""
import numpy as np, pandas as pd, os
from _paths import GAMMA_DIR, FIG   # resolves the repo root from __file__
M = 'gammacm_sweep_data_rarcut_fc2'
LOGRS = [-5,-4,-3,-2,-1,0,1,2,3]
pd.set_option('display.width', 200)

def load(run, name, sub=M):
    return pd.read_csv(os.path.join(FIG, run, sub, name))

def reldiff(a, b):
    a, b = np.asarray(a,float), np.asarray(b,float)
    m = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 0)
    return np.where(m, (a-b)/np.abs(b), np.nan)

# ---------------------------------------------------------------- frequency scan
print('='*100)
print('A. PULSE METRICS ACROSS THE WHOLE FREQUENCY SCAN  (500 cells vs 10^4 cells)')
print('='*100)
f = load('fiducial','lightcurve_freqscan_nu_m.csv')
h = load('hires','lightcurve_freqscan_nu_m.csv')
rows=[]
for lr in LOGRS:
    ff = f[f['log10(gc/gm)']==lr].reset_index(drop=True)
    hh = h[h['log10(gc/gm)']==lr].reset_index(drop=True)
    # align on nu/nu_m by interpolation onto the coarser (hires) grid
    xr = hh['nu/nu_m'].to_numpy()
    r={'logC':lr,'N_f':len(ff),'N_h':len(hh)}
    for col,lab in (('x_pk','T_pk/T_f'),('FWHM','FWHM'),('asym10','t_r/t_f'),
                    ('comb_sig','comb')):
        yf = np.interp(np.log10(xr), np.log10(ff['nu/nu_m']), ff[col])
        yh = hh[col].to_numpy()
        if col=='comb_sig':
            r['comb_500']  = np.nanmedian(yf)*100
            r['comb_1e4']  = np.nanmedian(yh)*100
            continue
        d = reldiff(yf, yh)*100
        r[lab+' med%'] = np.nanmedian(np.abs(d))
        r[lab+' max%'] = np.nanmax(np.abs(d))
    rows.append(r)
A = pd.DataFrame(rows)
print(A.to_string(index=False, float_format=lambda v: f'{v:8.3f}'))

# ---------------------------------------------------------------- spectra
print()
print('='*100)
print('B. SPECTRAL SHAPE (peak and time-integrated spectra), RS shell')
print('='*100)
sf = load('fiducial','spectrum_shape_table.csv')
sh = load('hires','spectrum_shape_table.csv')
key = ['shell','log10(C)','kind']
cols = ['nu_pk','nu_bk','nu_pk/nu_bk','W_1/2','a_lo','a_mid','a_hi','nu_M','sigma']
mg = sf.merge(sh, on=key, suffixes=('_f','_h'))
mg = mg[mg.shell.astype(str).str.contains('RS|4', case=False, na=False)] if 'shell' in mg else mg
out=[]
for _,r in mg.iterrows():
    o={'logC':r['log10(C)'],'kind':r['kind'],'class_f':r.get('class_f'),'class_h':r.get('class_h')}
    for c in cols:
        a,b = r.get(c+'_f'), r.get(c+'_h')
        try: a,b = float(a), float(b)
        except (TypeError, ValueError): a,b = np.nan, np.nan
        if c.startswith('a_') or c=='sigma':
            o['d '+c] = a-b                      # slopes: absolute difference
        else:
            o[c+' %'] = 100*(a-b)/abs(b) if np.isfinite(b) and b!=0 else np.nan
    out.append(o)
B = pd.DataFrame(out)
print(B.to_string(index=False, float_format=lambda v: f'{v:7.3f}'))

# ---------------------------------------------------------------- break tracks
print()
print('='*100)
print('C. CHARACTERISTIC-FREQUENCY TRACKS (power-law indices fitted to nu_c, nu_m vs T)')
print('='*100)
bf = load('fiducial','break_evolution_table.csv')
bh = load('hires','break_evolution_table.csv')
k='log10(gc/gm)'
mb = bf.merge(bh, on=k, suffixes=('_f','_h'))
cc=[c for c in bf.columns if c!=k]
o=[]
for _,r in mb.iterrows():
    d={'logC':r[k]}
    for c in cc:
        try: a,b=float(r[c+'_f']),float(r[c+'_h'])
        except (TypeError,ValueError): continue
        d['d '+c]= a-b
    o.append(d)
print(pd.DataFrame(o).to_string(index=False, float_format=lambda v: f'{v:7.3f}'))
