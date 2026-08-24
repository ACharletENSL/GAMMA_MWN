'''
full vs rarcut vs reconstructed, in SLOW cooling: spectral evolution, time-integrated
spectra, and lightcurves, from the cached sweep points.

The three treatments of the rarefaction wave:

  data                the reference -- cells followed through their real history, carrying
                      the crash as the simulation resolves it
  data_rarcut         the wave modelled as a SHARP cut-off at R_rar (rar_cut='model')
  data_norar_prerar   the wave REMOVED -- real rows to the edge of the rarefaction-free
                      window, then prerar_model's reconstruction to the same final radius

All three are plotted at the SAME observer-time indices, taken from the reference. That
matters: sweep_compare's phase detector picks the tail per side (measured bar_T 5.198 vs
3.960 in one case), so its panels can draw the three at different epochs and part of the
apparent difference is then the epoch, not the treatment.

Read alongside the nu_c investigation: the per-cell spectra carry a ~0.15 dex excess at nu_c
in slow cooling. In `data` and `data_rarcut` the cells' nu_c are spread over ~1 dex, so it
averages away in the shell; the reconstruction imposes one law on every cell, collapsing the
spread to ~0.1 dex, so the excess stacks and IS visible in the reconstructed shell. Treat the
reconstructed spectral SHAPE accordingly; eps_rad is unaffected (separate, bolometric path).

  python -c "import three_method_compare as C; C.main()"
  python -c "import three_method_compare as C; C.main(logr=2.)"
'''

import os
import numpy as np
import matplotlib.pyplot as plt

from environment import GAMMA_dir
from sweep_gammacm import load_sweep, method_outdir, detect_rise_peak_tail

KEY, Z = 'cooling_g100', 4
OUTDIR = os.path.join(GAMMA_dir, 'bin', 'Tools', 'figures', 'nuc_dip')
METHODS = (('data',              'full',          '#000000', '-'),
           ('data_rarcut',       'rarcut',        '#0072B2', '--'),
           ('data_norar_prerar', 'reconstructed', '#D55E00', '-.'))
NU_LC = (0.01, 1., 100.)        # lightcurve frequencies, in nu/nu_m
TRAPZ = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz


def main(logr=1., key=KEY, z=Z, outdir=OUTDIR):
  os.makedirs(outdir, exist_ok=True)
  data = {}
  for meth, lab, col, ls in METHODS:
    od = method_outdir(meth, key, z)
    if not os.path.isdir(od):
      print(f'{meth}: no cache, skipped'); continue
    R = {d['log10ratio']: d for d in load_sweep(od)}
    if logr not in R:
      print(f'{meth}: log10ratio={logr} missing, skipped'); continue
    data[meth] = R[logr]
  if 'data' not in data:
    print('reference missing, cannot proceed'); return
  ref = data['data']
  nu, Tb = np.asarray(ref['nub'], float), np.asarray(ref['Tb'], float)
  barT = Tb - 1.
  # SAME phase indices for all three, taken from the reference
  info = detect_rise_peak_tail(Tb, nu, ref['nuFnu'])[3]
  phases = [(w, info.get(f'i_{w}')) for w in ('rise', 'peak', 'tail')]
  phases = [(w, i) for w, i in phases if i is not None]
  nu_c = 10.**(2.*logr)

  fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), layout='constrained')

  # ---- 1. spectral evolution, at shared epochs
  pcol = {'rise': '#56B4E9', 'peak': '#E69F00', 'tail': '#CC79A7'}
  norm = max(np.nanmax(ref['nuFnu'][i]) for _, i in phases)
  for meth, lab, col, ls in METHODS:
    if meth not in data: continue
    S = np.asarray(data[meth]['nuFnu'], float)
    for w, i in phases:
      m = np.isfinite(S[i]) & (S[i] > 0)
      axes[0].loglog(nu[m], S[i][m]/norm, color=pcol[w], ls=ls, lw=1.4)
  for w, i in phases:
    axes[0].plot([], [], color=pcol[w], lw=2.,
                 label=rf'{w}  ($\bar T$={barT[i]:.2f})')
  for meth, lab, col, ls in METHODS:
    if meth in data:
      axes[0].plot([], [], color='0.35', ls=ls, lw=1.4, label=lab)
  axes[0].set(xlabel=r'$\nu/\nu_m$', ylabel=r'$\nu F_\nu$ (reference peak = 1)',
              ylim=(1e-7, 3.), title='spectral evolution (shared epochs)')

  # ---- 2. time-integrated (fluence) spectra
  for meth, lab, col, ls in METHODS:
    if meth not in data: continue
    S = np.asarray(data[meth]['nuFnu'], float)
    flu = TRAPZ(np.nan_to_num(S), barT, axis=0)
    m = flu > 0
    axes[1].loglog(nu[m], flu[m]/np.nanmax(flu), color=col, ls=ls, lw=1.6, label=lab)
  axes[1].set(xlabel=r'$\nu/\nu_m$', ylabel=r'fluence $\nu F_\nu$, normalised',
              ylim=(1e-5, 3.), title='time-integrated spectra')

  # ---- 3. lightcurves
  for meth, lab, col, ls in METHODS:
    if meth not in data: continue
    S = np.asarray(data[meth]['nuFnu'], float)
    for nut, a in zip(NU_LC, (1., .55, .3)):
      j = int(np.argmin(np.abs(nu - nut)))
      lc = np.nan_to_num(S[:, j]); m = (lc > 0) & (barT > 0)
      axes[2].loglog(barT[m], lc[m]/np.nanmax(lc), color=col, ls=ls, lw=1.4, alpha=a)
  for nut, a in zip(NU_LC, (1., .55, .3)):
    axes[2].plot([], [], color='0.35', lw=2., alpha=a, label=rf'$\nu={nut:g}\,\nu_m$')
  axes[2].set(xlabel=r'$\bar T$', ylabel=r'$\nu F_\nu$ (each normalised)',
              ylim=(1e-5, 3.), title='lightcurves')

  for ax in axes:
    ax.grid(alpha=.25)
    ax.legend(fontsize=8, frameon=False)
  for ax in axes[:2]:
    ax.axvline(nu_c, color='crimson', ls=':', lw=1.2)
  fig.suptitle(rf'full / rarcut / reconstructed, $\log_{{10}}(\gamma_c/\gamma_m)$={logr:+.0f}'
               r'  (red dotted = $\nu_c$)', fontsize=12)
  path = os.path.join(outdir, f'three_method_compare_logr{logr:+.0f}.png')
  fig.savefig(path, dpi=140, bbox_inches='tight')
  plt.close(fig)
  print(f'saved {path}')

  # numbers alongside
  print(f'\nlog10(gc/gm)={logr:+.0f}, nu_c/nu_m={nu_c:.4g}')
  for meth, lab, col, ls in METHODS:
    if meth not in data: continue
    d = data[meth]; S = np.asarray(d['nuFnu'], float)
    flu = TRAPZ(np.nan_to_num(S), barT, axis=0)
    eps = d['E_rad']/d['E_inj']
    lc = np.nansum(np.nan_to_num(S), axis=1)
    print(f'  {lab:>14}: eps_rad {eps:.4f}   total fluence {TRAPZ(flu, nu):.5g}'
          f'   peak barT {barT[int(np.nanargmax(lc))]:.3f}')
  return data


if __name__ == '__main__':
  main()
