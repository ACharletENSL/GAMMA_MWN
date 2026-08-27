# -*- coding: utf-8 -*-
# @Author: acharlet

'''
What does stopping the emission sharply at the rarefaction wave cost?

The two emission pipelines differ in TWO independent ways at once: the hydro
treatment (BPL fits vs actual snapshot histories) and the post-rarefaction
treatment. The fit path kills every worldline at the modelled catch-up radius
R_rar (terminal `reach_rar` event, working_cooling.worldline_from_cooling); the
data path has no rarefaction machinery at all and follows each cell to its last
snapshot -- the wave is in the simulation, so nothing has to be modelled. This
module isolates the second difference: the SAME data-driven computation on the
SAME simulation, run twice --

    'data'         cells followed to end of data   (rar_cut=None)   REFERENCE
    'data_rarcut'  cells truncated at R_rar        (rar_cut='model')

so every ratio here is the cost of the cut-off prescription alone, with the hydro
held fixed. Ratios are reported full/cut, i.e. reference/prescription, so >= 1 is
emission the sharp cut throws away.

Run on cooling_g100 (the fiducial), measured per shell:

              cells end at bar{T}   modelled cut at bar{T}   R_rar/R_injection
    RS z=4      645.85 .. 650.23        1.3094 .. 2.0254        1.000 .. 3.905
    FS z=1      526.21 .. 645.82        1.0662 .. 1.7040        1.000 .. 3.297

so the reference outlives the cut by ~2.5 decades in observer time. That is 82x
more post-cut history than the retired cooling_fid_raref_ext (cells ended at
bar{T} = 8.2..10.8), where the cut cost <=7.6% integrated -- expect MORE here.

R_rar/R_injection is 1 at the outer edge (shocked last, the wave is already on it,
so it is the launch cell by construction and its cut-off bar{T} IS bar{T}_f) and
rises to ~3-4 at the contact discontinuity (shocked first, radiates to several
times its injection radius before the wave arrives). The cut therefore bites
hardest at the outer edge.

Both shells: z=4 (reverse, default) and z=1 (forward) each get their own sweeps
and their own output directory.

Note what "sharp" does and does not mean: truncating the worldline stops the
cell's ON-AXIS emission, but get_Fnu_cell_evolving still delivers the tT^-2
high-latitude tail of every step already emitted. Photons in flight still arrive.
That is why the cut costs far less than the bar{T} numbers alone suggest.

"The simulation ended" and "the wave arrived" are distinct causes of a missing
tail, and this module only controls the second. The run-LENGTH companion that
separated them (sweep_duration.py) was removed once its paired runs were deleted:
it measured that a 5x duration change moves the low-frequency late lightcurve by
up to 2.1x, and <3% on integrated observables.

Example use in command line:
  python -c "import sweep_rarcut as R; R.main(nproc=4)"
  python -c "import sweep_rarcut as R; R.main(z=1, nproc=7)"
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from environment import MyEnv
from IO import open_celldata, get_variable
from plotting_functions import COL_RS, COL_FS, COL_TOT
from working_cooling import (load_shell_rarefaction, rar_map_lookup,
    cell_radiated_energy)
from working_cooling_data import generate_cell_fromData
import sweep_compare as cmp
from sweep_gammacm import (run_sweep, load_sweep, method_outdir, data_end_barT,
    exit_onset_barT, rarefaction_off_barT, compute_alpha_sweep, trim_pngs, _draw_order,
    copy_article_figures, GAMMA_dir, LOG10RATIO_ARR, Z_SHELL, R_REF, SPEC_MODES)

KEY = 'cooling_g100'
METHOD_A, METHOD_B = 'data_rarcut', 'data'   # A = cut (denominator), B = full = REFERENCE
LABELS = ('rf cut', 'full')               # every ratio panel is then full/cut
NORM_SIDE = 'A'                              # the flux panels are anchored on the CUT: it is
                                             # the prescription one would quote, so the full
                                             # curve reads directly as the extra emission the
                                             # cut leaves out. Linestyles are untouched by
                                             # this (dashed = cut, solid = full), and so are
                                             # the ratio panels, which stay full/cut.
OUTDIR = os.path.join(GAMMA_dir, 'bin', 'Tools', 'figures', 'rarcut_compare')
N_PROFILE = 42                               # per-cell energy profile: cells sampled per shell
LOGR_PROFILE = (-3., 0., 2.)
SHELL_NAME = {4: 'RS', 1: 'FS'}              # z -> shell name, for figure titles


def shell_cell_range(key=KEY, z=Z_SHELL):
  '''
  (k_first, k_last, k_CD) of shell z, from the env's own cell counts rather than
  hardcoded indices: the grid is [Next | Nsh4 | Nsh1 | Next], so shell 4 spans
  [Next, Next+Nsh4) with the contact discontinuity at its TOP end, and shell 1
  spans [Next+Nsh4, Next+Nsh4+Nsh1) with the CD at its BOTTOM end. k_CD is
  returned separately because the energy-profile x axis is labelled by it.
  '''
  env = MyEnv(key)
  k4 = int(env.Next)
  kCD = k4 + int(env.Nsh4)
  k1 = kCD + int(env.Nsh1)
  return ((k4, kCD, kCD - 1) if z == 4 else (kCD, k1, kCD))


def profile_cells(key=KEY, z=Z_SHELL, n=N_PROFILE):
  '''Evenly spaced sample of shell z's cells for plot_cut_energy_profile.'''
  lo, hi, _ = shell_cell_range(key, z)
  return range(lo, hi, max(1, (hi - lo)//n))


def plot_cut_energy_profile(cells=None, logr_list=LOGR_PROFILE, key=KEY,
    outdir=OUTDIR, r_ref=R_REF, z=Z_SHELL):
  '''
  Which part of the shell the cut actually costs: E_rad(full)/E_rad(cut) per cell,
  one curve per cooling regime, against the cell's own R_rar/R_injection. The two
  ends of the shell are in opposite situations -- the CD-adjacent cells are shocked
  first and radiate to ~3x their injection radius before the wave arrives, the
  outer-edge cells are shocked last and the wave is already on them (ratio ~1) --
  so the cut is expected to bite hardest at the outer edge.
  Comoving energies (cell_radiated_energy), so this is a pure per-cell budget with
  no observer projection.
  cells defaults to an even sample of shell z (profile_cells).
  '''
  os.makedirs(outdir, exist_ok=True)
  cells = profile_cells(key, z) if cells is None else cells
  _, _, kCD = shell_cell_range(key, z)
  env0 = MyEnv(key)
  rar_map = load_shell_rarefaction(key, z, env0)
  alphas, logr0 = compute_alpha_sweep(key, logr_list)
  print(f'baseline log10(gma_c/gma_m) = {logr0:.4f}')

  fig, axs = plt.subplots(2, 1, figsize=(7., 7.), sharex=True)
  # value-indexed, like the sweep colorbars elsewhere: the colour of a regime does not
  # depend on where it sits in logr_list, so the draw order below can be reversed freely
  lr = np.asarray(logr_list, float)
  colors = plt.cm.coolwarm(plt.Normalize(vmin=lr.min(), vmax=lr.max())(lr))
  for logr, alpha, c in _draw_order(zip(logr_list, alphas, colors)):
    ks, ratio, ror_l = [], [], []
    for k in cells:
      d = open_celldata(key, k)
      if d is False:
        continue
      ror = rar_map_lookup(rar_map, k)
      cc, ec = generate_cell_fromData(d, env0, alpha=alpha, r_ref=r_ref, rar_ratio=ror)
      cf, ef = generate_cell_fromData(d, env0, alpha=alpha, r_ref=r_ref)
      if cc is False or cf is False:
        continue
      Ec = cell_radiated_energy(cc, ec)
      if Ec <= 0.:
        continue
      ks.append(k); ror_l.append(ror)
      ratio.append(cell_radiated_energy(cf, ef)/Ec)
    axs[0].plot(ks, ratio, '.-', color=c, lw=1.1, ms=4,
                label=f'$\\log_{{10}}(\\gamma_c/\\gamma_m)={logr:+.0f}$')
    axs[1].plot(ks, ror_l, '.-', color='k', lw=1.1, ms=4)
  axs[0].axhline(1., color='grey', ls=':', lw=.9)
  axs[0].set_yscale('log')
  axs[0].set_ylabel('$E_{\\rm rad}$ full / cut')
  axs[0].legend(fontsize=9)
  shell = 'RS' if z == 4 else 'FS'
  axs[0].set_title(f'Radiated energy the sharp $R_{{\\rm rar}}$ cut discards, per cell ({shell})')
  axs[1].set_ylabel('$R_{\\rm rar}/R_{\\rm inj}$')
  lo, hi, _ = shell_cell_range(key, z)
  axs[1].set_xlabel(f'cell index $k$   (CD at {kCD}, outer edge at '
                    f'{lo if z == 4 else hi - 1})')
  fig.tight_layout()
  fig.savefig(os.path.join(outdir, 'cut_energy_profile.png'), dpi=300)
  plt.close(fig)
  print(f'per-cell energy profile saved to {outdir}')


def main(key=KEY, log10ratio_arr=LOG10RATIO_ARR, outdir=None, use_cache=True,
    nproc=None, labels=LABELS, profile=True, z=Z_SHELL, norm_side=NORM_SIDE):
  '''
  Ensure both sweeps of shell z exist (the 'data' side is normally already cached,
  being the reference computation), then build every comparison figure with the CUT
  as side A, so the ratio panels read full/cut = reference/prescription, and (norm_side,
  see the constant) the flux panels anchored on the cut.
  z: 4 (reverse shock, default) or 1 (forward shock); the latter gets its own
  '_z={z}' output directory, as its sweep caches do (method_outdir).
  '''
  outdir = (OUTDIR if z == Z_SHELL else f'{OUTDIR}_z={z}') if outdir is None else outdir
  os.makedirs(outdir, exist_ok=True)
  for m in (METHOD_A, METHOD_B):
    d = method_outdir(m, key, z)
    os.makedirs(d, exist_ok=True)
    if not (use_cache and load_sweep(d)):
      print(f'--- running the {m} sweep on {key}, shell z={z} ---')
      # skip_cached MUST be forwarded (see sweep_gammacm.main): run_sweep defaults it
      # to True, so use_cache=False would otherwise recompute nothing.
      run_sweep(key, log10ratio_arr, z=z, method=m, nproc=nproc,
                skip_cached=use_cache)
  pairs = cmp.load_pairs(method_outdir(METHOD_A, key, z), method_outdir(METHOD_B, key, z))

  barT_f = exit_onset_barT(key, z=z)
  barT_off = rarefaction_off_barT(key, z=z)
  barT_end = (barT_off, data_end_barT(key, z=z))   # where each side stops
  print(f'crossing bar_T_f = {barT_f:.4f}')
  for b, lab in zip(barT_end, labels):
    if b:
      print(f'emission stops ({lab}): bar_T = {b[0]:.4f}..{b[1]:.4f}')

  cmp.plot_efficiency_compare(pairs, outdir=outdir, labels=labels)
  for kind in ('peak', 'fluence'):
    for mode in SPEC_MODES:
      # no ratio panel on the fluence spectra: the cut's effect there is a visible
      # separation between the two curves, which the shared normalisation already puts
      # on the flux axis. The PEAK spectra keep theirs -- the peak phase is emitted
      # before the cut, so the two sides sit on top of each other and the only way to
      # see the (tiny) difference is on its own axis
      cmp.plot_spectra_compare(pairs, kind=kind, mode=mode, outdir=outdir, labels=labels,
                               ratio=(kind != 'fluence'), norm_side=norm_side)
  # rise/peak/tail spectra per regime: the peak phase is emitted before the cut and must
  # come out identical, so this isolates where in the spectrum the discarded tail sits
  cmp.plot_spectral_evolution_compare(pairs, outdir=outdir, labels=labels,
                                      norm_side=norm_side)
  # the lin-lin variant ONLY: the cut-vs-full difference is all in the decay, a linear
  # clock is where it reads naturally, and the log / linlog views of the same curves were
  # two more files per frequency for a divergence the ratio panel already carries
  cmp.plot_lightcurve_compare(pairs, barT_f, barT_off=barT_off, outdir=outdir,
      labels=labels, barT_end=barT_end, scale='lin', norm_side=norm_side)
  s = cmp.plot_summary_ratios(pairs, outdir=outdir, labels=labels)
  fs = cmp.fluence_split(pairs, barT_off[1] if barT_off else None, outdir=outdir,
      labels=labels, cut_label='R_rar cut-off', cut_math='$R_{\\rm rar}$ cut-off')
  # the SHAPE the cut leaves behind: everything above measures the area under the pulse
  # and comes out within a few percent, while the low-energy index of the time-integrated
  # spectrum moves by ~1/3 (see cmp.fluence_slope_table)
  series = cmp.fluence_series(pairs)
  tab = cmp.fluence_slope_table(series, labels=labels)
  tab.to_csv(os.path.join(outdir, 'fluence_low_slopes.csv'), index=False)
  cmp.plot_fluence_slope_profile(series, outdir=outdir, labels=labels,
                                 title_extra=f'  ({SHELL_NAME.get(z, f"z={z}")})')

  if profile:
    plot_cut_energy_profile(key=key, outdir=outdir, z=z)
  trim_pngs(outdir)
  copy_article_figures(outdir)
  print(f'Rarefaction-cut comparison figures saved to {outdir}')
  return pairs, s, fs, tab


def slopes_all_shells(key=KEY, log10ratio_arr=LOG10RATIO_ARR, outdir=None, verbose=True):
  '''
  The low-energy index of the time-integrated spectra vs cooling regime, cut vs full, for
  the RS (z=4), the FS (z=1) and their SUM on one figure -- the sum being what an observer
  would actually see.

  CACHE-ONLY by design: unlike main() this never triggers a sweep. It reads the four point
  caches (two methods x two shells) and does nothing but measure, so it is safe to re-run
  while a sweep is going on elsewhere. Run main(z=4) and main(z=1) first if any is missing.

  The sum is built by sweep_shells.load_shell_pairs, which puts the FS on the RS flux scale
  (x 1/(fac_nu*fac_F)) and on the RS frequency axis; its low-energy asymptote is therefore
  bounded by the LOWER of the two shells' breaks, which is the FS one -- see
  cmp._total_nu_break.

  Writes fluence_slopes_vs_regime.png and fluence_low_slopes_all.csv into `outdir`.
  Returns {series name: DataFrame}.
  '''
  import sweep_shells as ssh
  outdir = OUTDIR if outdir is None else outdir
  os.makedirs(outdir, exist_ok=True)

  tables, styled = {}, {}
  for z, name, col in ((4, 'RS', COL_RS), (1, 'FS', COL_FS)):
    pairs = cmp.load_pairs(method_outdir(METHOD_A, key, z), method_outdir(METHOD_B, key, z))
    if verbose:
      print(f'\n=== {name} (z={z}) ===')
    tables[name] = cmp.fluence_slope_table(cmp.fluence_series(pairs), labels=LABELS,
                                           verbose=verbose)
    styled[name] = (tables[name], col)

  # the sum: pair the two shell-sums regime by regime, cut against full
  tot = {m: {round(p['log10ratio'], 6): p
             for p in ssh.load_shell_pairs(key, m, log10ratio_arr)}
         for m in (METHOD_A, METHOD_B)}
  common = sorted(set(tot[METHOD_A]) & set(tot[METHOD_B]))
  tot_pairs = [(tot[METHOD_A][k], tot[METHOD_B][k]) for k in common]
  if verbose:
    print('\n=== RS + FS total ===')
  tables['total'] = cmp.fluence_slope_table(
      cmp.fluence_series(tot_pairs, spec_key='nuFnu_tot', x_key='x',
                         nu_break=cmp._total_nu_break), labels=LABELS, verbose=verbose)
  styled['total'] = (tables['total'], COL_TOT)

  for kind in ('asymptote', 'band'):
    cmp.plot_fluence_slopes_vs_regime(styled, outdir=outdir, labels=LABELS, kind=kind)
  out = pd.concat([t.assign(series=n) for n, t in tables.items()], ignore_index=True)
  out.to_csv(os.path.join(outdir, 'fluence_low_slopes_all.csv'), index=False)
  trim_pngs(outdir)
  print(f'\nlow-energy slope figure + table saved to {outdir}')
  return tables
