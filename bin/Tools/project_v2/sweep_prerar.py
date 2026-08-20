'''
log10(gamma_c/gamma_m) sweep of the PRE-RAREFACTION RECONSTRUCTION counterfactual against the
reference, and its figure battery.

WHAT IS BEING COMPARED

  'data'                the reference: every cell followed through its real history, carrying
                        the rarefaction crash as the simulation resolves it.
  'data_norar_prerar'   the same cells to the same final radius, but with the crash REPLACED
                        by prerar_model's reconstruction: real rows out to the edge of the
                        rarefaction-free window, then a tail on the measured alpha tables
                        running onto the derived causal-contact asymptote.

So the pair isolates the rarefaction wave, exactly as sweep_norar does -- what differs is the
EXTENSION LAW, and that is the whole point of this module existing alongside it:

  sweep_norar          'bernoulli' (also 'adiab', 'bpl'): the tail is closed from the handover
                       row on conservation laws alone, with rho -> R^-2 free coasting.
  sweep_prerar (here)  the tail is the MEASURED evolution of these cells -- alpha_D, alpha_G,
                       alpha_p tabulated in (R_i/R_0, R/R_i) on a wide-shell basis run, blended
                       onto the DERIVED asymptote alpha_D = 2/gma-2 = -0.800, alpha_p = -2,
                       alpha_G = 0.

WHY THE LAWS DIFFER, AND BY HOW MUCH: a shocked cell is NOT a free fluid element. It sits inside
a causally-connected shocked layer that keeps compressing it toward the contact discontinuity,
so rho ~ R^-1.2, not R^-2 -- see prerar_model's module docstring for the derivation. Over the
~2.5 decades this counterfactual extrapolates, that index difference is a factor ~100 in
density, so the two extension laws are NOT small perturbations of each other. rho ~ R^-2 is
never an asymptote this model relaxes to: the cell is interacting for the whole prolongation.

WHERE IT SHOWS UP: at the fast-cooling end of the sweep essentially all the injected electron
energy radiates whatever the hydro does, so eps_rad is nearly law-independent (measured 8e-5
relative on the fiducial cell). The law has to matter at the SLOW-cooling end, where only part
of the energy is radiated and the hydro path sets how much. That gradient across the sweep is
the result this module produces.

  python -c "import sweep_prerar as S; S.main(nproc=7)"      # sweep + figures
  python -c "import sweep_prerar as S; S.main(z=1, nproc=7)" # forward shock
  python sweep_prerar.py                                     # same as main(), guarded

MUST BE RUN UNDER A __main__ GUARD (the module-level entry point below does this). The pool
uses forkserver -- plain fork deadlocks on threaded BLAS, and forkserver without the guard
re-imports the module in every worker. Both failure modes stall silently rather than erroring.
'''

import os
import numpy as np

from environment import GAMMA_dir
import sweep_compare as cmp
from sweep_gammacm import (run_sweep, load_sweep, method_outdir, data_end_barT,
    exit_onset_barT, rarefaction_off_barT, cap_end_barT, trim_pngs,
    copy_article_figures, LOG10RATIO_ARR, Z_SHELL, R_CAP)

KEY = 'cooling_g100'
LAW = 'prerar'
METHOD_A, METHOD_B = 'data_norar_prerar', 'data'
LABELS = ('reconstructed', 'full')   # 'no rf' is sweep_norar's label for its own law; keeping
                                     # them distinct so figures from the two are never confused
OUTDIR = os.path.join(GAMMA_dir, 'bin', 'Tools', 'figures', 'prerar_compare')
SHELL_NAME = {4: 'RS', 1: 'FS'}      # z -> shell name, for figure titles


def main(key=KEY, log10ratio_arr=LOG10RATIO_ARR, outdir=None, use_cache=True,
    nproc=None, labels=LABELS, z=Z_SHELL, cap=None):
  '''
  Full sweep of the reconstruction counterfactual against the reference, then the
  sweep_compare figure battery.

  Runs the counterfactual ALONE against the already-cached 'data' reference, as
  sweep_norar.main does and for the same reason: a paired run would rewrite
  figures/gammacm_sweep_data/, which sweep_rarcut, sweep_shells, sweep_efficiency and
  boundary_comparison all compare against.
  '''
  method_a = METHOD_A + ('_cap' if cap else '')
  method_b = METHOD_B + ('_cap' if cap else '')
  suff = (f'_cap={cap:g}' if cap else '') + (f'_z={z}' if z != Z_SHELL else '')
  outdir = (OUTDIR + suff) if outdir is None else outdir
  os.makedirs(outdir, exist_ok=True)

  # run_sweep resumes from its own per-point cache (skip_cached), so it is called
  # unconditionally and computes only what is missing.
  print(f'--- {method_a} sweep on {key}, shell z={z} ---')
  run_sweep(key, log10ratio_arr, z=z, method=method_a, nproc=nproc,
            skip_cached=use_cache)
  if not load_sweep(method_outdir(method_b, key, z)):
    print(f'--- {method_b} reference sweep on {key}, shell z={z} ---')
    run_sweep(key, log10ratio_arr, z=z, method=method_b, nproc=nproc)

  pairs = cmp.load_pairs(method_outdir(method_a, key, z),
                         method_outdir(method_b, key, z))
  barT_f = exit_onset_barT(key, z=z)
  barT_off = rarefaction_off_barT(key, z=z)
  # both sides stop at the same radius, hence the same observer time to within the
  # extension's lag difference -- the same design identity sweep_norar relies on
  end = cap_end_barT(key, z=z, cap=cap) if cap else data_end_barT(key, z=z)
  barT_end = (end, end)
  print(f'crossing bar_T_f = {barT_f:.4f}')
  print(f'rarefaction reaches the cells at bar_T = {barT_off[0]:.4f}..{barT_off[1]:.4f}')
  if end:
    print(f'both sides stop at bar_T = {end[0]:.4f}..{end[1]:.4f}')

  cmp.plot_efficiency_compare(pairs, outdir=outdir, labels=labels)
  for kind in ('peak', 'fluence'):
    for mode in ('nu_m', 'max'):
      cmp.plot_spectra_compare(pairs, kind=kind, mode=mode, outdir=outdir, labels=labels,
                               ratio=(kind != 'fluence'))
  cmp.plot_spectral_evolution_compare(pairs, outdir=outdir, labels=labels)
  for sc in ('log', 'linlog', 'lin'):
    cmp.plot_lightcurve_compare(pairs, barT_f, barT_off=barT_off, outdir=outdir,
        labels=labels, barT_end=barT_end, scale=sc)
  s = cmp.plot_summary_ratios(pairs, outdir=outdir, labels=labels)
  fs = cmp.fluence_split(pairs, barT_off[1] if barT_off else None, outdir=outdir,
      labels=labels, cut_label='rarefaction arrival',
      cut_math='rarefaction arrival')
  series = cmp.fluence_series(pairs)
  tab = cmp.fluence_slope_table(series, labels=labels)
  tab.to_csv(os.path.join(outdir, 'fluence_low_slopes.csv'), index=False)
  cmp.plot_fluence_slope_profile(series, outdir=outdir, labels=labels,
                                 title_extra=f'  ({SHELL_NAME.get(z, f"z={z}")})')
  trim_pngs(outdir)
  copy_article_figures(outdir)
  print(f'Reconstruction comparison figures saved to {outdir}')
  return pairs, s, fs, tab


def main_capped(cap=R_CAP, **kw):
  '''Both sides truncated at R/R_inj = cap, so the endpoints still match but the
  reconstruction only carries ~1-1.5 decades instead of ~2.5. Comparing its ratios with
  main()'s is the convergence check on the extrapolation length.'''
  return main(cap=cap, **kw)


if __name__ == '__main__':
  main()
