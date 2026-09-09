'''
Single-cell diagnostic for the reconstruction's emission: hydro, electron bounds, and the
cell's OWN instantaneous spectra at selected observer times, reference vs reconstruction.

Built to answer "why does the reconstructed tail spectrum not look synchrotron-like", by
showing the three things that determine it, for one cell, on one radius axis:

  row 1   rho R^1.2, p R^2, Gamma   -- the reconstructed hydro (the model's output)
  row 2   gamma_min, gamma_max      -- the electron bounds that set nu_m and nu_max
  row 3   nu F_nu of THIS CELL ALONE at several observer times

TWO TRAPS this module exists to avoid, both of which produced wrong numbers before it:

  - PICK OBSERVER TIMES INSIDE THE CELL'S EMISSION SPAN. A cell emits over a Ton range that
    can be far shorter than the Tobs grid (here ~0.3-326 against a grid running to 1000);
    sampling past the end returns flux that is not this cell radiating and is meaningless to
    interpret. Times are chosen here from the cell's OWN lightcurve, by flux quantile.
  - COMPARE THE TWO SIDES AT THE SAME OBSERVER TIME. sweep_compare's phase detector picks
    the tail per side (measured T_bar 5.198 vs 3.960), so its tail panel draws the two curves
    at different times; roughly a fifth of the apparent shape difference is that alone.

  python -c "import prerar_cell_diagnostic as D; D.main()"
  python -c "import prerar_cell_diagnostic as D; D.main(logr=0.)"
'''

import os
import numpy as np
import matplotlib.pyplot as plt

from environment import GAMMA_dir
import working_cooling_data as W
import prerar_model as M
import prerar_cell_evolution as P
from sweep_gammacm import compute_alpha_sweep

KEY = 'cooling_g100'
Z = 4
OUTDIR = os.path.join(GAMMA_dir, 'bin', 'Tools', 'figures', 'prerar_model')
COL = {'full': '#D55E00', 'reconstructed': '#0072B2'}   # Okabe-Ito, CVD-safe
GMA_REF = M.GMA_REF


def cell_frames(Ri_target=1.05, logr=1., key=KEY, z=Z, Tmax=1000, NT=250,
    Nnu=400, lognu_min=-6, lognu_max=8):
  '''
  Emitting cell frames + spectra for one cell, reference and reconstruction, at the sweep
  point log10(gamma_c/gamma_m) = logr. Returns (out, k, alpha) with out[label] =
  dict(cell, nu, Tobs, nuFnu, env).
  '''
  cells = M._test_cells(key, z, 120)
  k = int(min(cells, key=lambda c: abs(c[4] - Ri_target))[0])
  alpha = float(compute_alpha_sweep(key, np.array([float(logr)]))[0][0])
  out = {}
  for label, law in (('full', None), ('reconstructed', 'prerar')):
    nuobs, Tobs, env, nuFnu, cd = W.get_cell_nuFnu_fromData(
        key, k, norar=law, alpha=alpha, Tmax=Tmax, NT=NT, Nnu=Nnu,
        lognu_min=lognu_min, lognu_max=lognu_max, return_cell=True)
    out[label] = dict(cell=cd, nu=nuobs/env.nu0, Tobs=Tobs,
                      nuFnu=np.asarray(nuFnu, float), env=env)
  return out, k, alpha


def emission_times(out, n=4, qlo=1e-5):
  '''
  n observer-time indices spanning THIS CELL's emission, shared by both sides so the
  spectra are compared at equal times. Chosen on the reference lightcurve: the window where
  it is above qlo of its peak, sampled from the peak outward in equal log steps.

  Returns (indices, Tbar) with Tbar the sweep's bar-T convention (Tobs/T0 - 1).
  '''
  ref = out['full']
  lc = np.nansum(ref['nuFnu'], axis=1)
  T0 = ref['env'].T0
  Tbar = ref['Tobs']/T0 - 1.
  # Span the WHOLE live range, both sides: the interesting divergence is late (the cell
  # emits to bar_T ~ 300 here), and a quantile cut on the peak throws exactly that away.
  live = None
  for d in out.values():
    l = np.nansum(d['nuFnu'], axis=1)
    w = np.flatnonzero(np.isfinite(l) & (l > np.nanmax(l)*qlo))
    live = w if live is None else np.union1d(live, w)
  ipk = int(np.nanargmax(lc))
  tgt = np.geomspace(max(Tbar[ipk], 1e-2), max(Tbar[int(live[-1])], 1e-1), n)
  idx = sorted({int(np.argmin(np.abs(Tbar - t))) for t in tgt})
  return idx, Tbar


def main(Ri_target=1.05, logr=1., key=KEY, z=Z, outdir=OUTDIR, n_times=4):
  os.makedirs(outdir, exist_ok=True)
  out, k, alpha = cell_frames(Ri_target, logr, key, z)
  idx, Tbar = emission_times(out, n=n_times)
  ref = out['full']
  Ri = float(M._test_cells(key, z, 120)[0][4]) if False else Ri_target

  fig, axes = plt.subplots(3, 3, figsize=(14., 11.), layout='constrained')

  # ---- row 1: hydro, scaled by the derived indices (flat = on the causal-contact law)
  hyd = (('rho', rf'$\rho\,(R/R_{{\rm inj}})^{{{2./GMA_REF:.1f}}}$', 2./GMA_REF),
         ('p',   r'$p\,(R/R_{\rm inj})^{2}$', 2.),
         ('lfac', r'$\Gamma/\Gamma_{\rm inj}$', 0.))
  for j, (col, lab, pw) in enumerate(hyd):
    ax = axes[0, j]
    for name, d in out.items():
      cd = d['cell']; x = cd.x.to_numpy(float); L = np.log10(x/x[0])
      y = cd[col].to_numpy(float)*10.**(pw*L)
      ax.plot(L, y/y[0], color=COL[name], lw=1.6,
              ls='--' if name == 'reconstructed' else '-', label=name)
    ax.set(xlabel=r'$\log_{10}(R/R_{\rm inj})$', ylabel=lab)
    ax.grid(alpha=.25)
    if j == 0:
      ax.legend(fontsize=8, frameon=False)

  # ---- row 2: electron bounds (and their ratio), the two numbers that set nu_m and nu_max
  for j, (col, lab) in enumerate((('gmin', r'$\gamma_{\rm min}$'),
                                  ('gmax', r'$\gamma_{\rm max}$'),
                                  (None,   r'$\gamma_{\rm max}/\gamma_{\rm min}$'))):
    ax = axes[1, j]
    for name, d in out.items():
      cd = d['cell']; x = cd.x.to_numpy(float); L = np.log10(x/x[0])
      y = (cd.gmax.to_numpy(float)/cd.gmin.to_numpy(float)) if col is None \
          else cd[col].to_numpy(float)
      ax.plot(L, y, color=COL[name], lw=1.6,
              ls='--' if name == 'reconstructed' else '-')
    ax.set(xlabel=r'$\log_{10}(R/R_{\rm inj})$', ylabel=lab, yscale='log')
    ax.grid(alpha=.25)

  # ---- row 3: this cell's own instantaneous spectra, SAME observer times both sides
  show = idx[:3] if len(idx) >= 3 else idx
  for j, it in enumerate(show):
    ax = axes[2, j]
    for name, d in out.items():
      s = d['nuFnu'][it]
      m = np.isfinite(s) & (s > 0.)
      ax.loglog(d['nu'][m], s[m], color=COL[name], lw=1.5,
                ls='--' if name == 'reconstructed' else '-')
    ax.axvline(1., color='grey', ls=':', lw=.9)
    ax.set(xlabel=r'$\nu/\nu_{\mathrm{m},0}$', ylabel=r'$\nu F_\nu$ (this cell)',
           title=rf'$\bar T$ = {Tbar[it]:.2f}')
    ax.grid(alpha=.25)
    smax = max(np.nanmax(d['nuFnu'][it]) for d in out.values())
    if smax > 0:
      ax.set_ylim(smax*1e-6, smax*3.)

  fig.suptitle(f'cell k={k}  (R_i/R_0 ~ {Ri_target})   '
               rf'$\log_{{10}}\mathcal{{C}}$ = {logr:+.0f},  '
               rf'$\alpha$ = {alpha:.4g}', fontsize=12)
  path = os.path.join(outdir, f'cell_diagnostic_k{k}_logr{logr:+.0f}.png')
  fig.savefig(path, dpi=150, bbox_inches='tight')
  plt.close(fig)
  print(f'saved {path}')

  # numbers to read alongside the figure
  print(f'\ncell k={k}, log10(gc/gm)={logr:+.0f}, alpha={alpha:.5g}')
  for name, d in out.items():
    cd = d['cell']; x = cd.x.to_numpy(float)
    t = cd.t.to_numpy(float); Ton = (t - x); Ton = (Ton - Ton[0])/d['env'].T0
    print(f'  {name:14s} rows={len(cd):4d}  R_end/R_inj={x[-1]/x[0]:7.1f}  '
          f'gmin {cd.gmin.iloc[0]:.4g}->{cd.gmin.iloc[-1]:.4g}  '
          f'gmax {cd.gmax.iloc[0]:.4g}->{cd.gmax.iloc[-1]:.4g}')
    print(f'  {"":14s} own emission spans bar_T = {Ton.min():.3f} .. {Ton.max():.3f}')
  print(f'  observer times plotted (bar_T): {[round(float(Tbar[i]),3) for i in show]}')
  return out, k, alpha


if __name__ == '__main__':
  main()
