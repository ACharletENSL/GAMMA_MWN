# -*- coding: utf-8 -*-
# @Author: acharlet

'''
Does it matter how long the simulation follows the electron distribution?

WARNING -- INOPERABLE AS IT STANDS: both runs it compares (KEY_A, KEY_B below) were
deleted when cooling_g100 became the fiducial, and cooling_g100 is a SINGLE
duration -- there is no second run of the same setup to pair it with. Nothing else
in the module is stale; to revive it, run cooling_g100's phys_input.ini twice with
different stopping times (differing only in `runname` and the stop condition) and
point KEY_A/KEY_B at those. The numbers quoted below describe the retired runs and
are kept only as the record of what was measured on them.

The data-driven emission path (working_cooling_data.get_shell_nuFnu_fromData) has
no rarefaction machinery: each cell is followed to its LAST SNAPSHOT
(tt_end = min(tt_nodes[-1], tt_geo[-1])). That makes the duration of the hydro run
a physical input to the flux, which it was not for the fit path -- there every
worldline is cut at the modelled R_rar regardless of how long the run went.

This module compares the same emission computation on two runs of the SAME setup
(their phys_input.ini differ only in `runname`) that stop at different times:

    cooling_fid_raref      885 rows/cell, cells end at bar{T} = 1.66 .. 2.21
    cooling_fid_raref_ext 3814 rows/cell, cells end at bar{T} = 8.20 .. 10.77

For reference the modelled rarefaction darkens the shell over bar{T} = 1.28 .. 1.67,
i.e. `_raref` (stop = rarefaction) ends essentially as the wave finishes crossing,
and ALL of `_ext`'s extra coverage is post-crossing.

Expectation: in fast cooling the electrons have burnt off long before either run
ends, so the two must agree; in slow cooling the distribution is still radiating
when `_raref` stops, so truncating there loses on-axis flux and leaves only the
tT^-2 high-latitude tails of the steps already emitted. The sweep measures where
that crossover actually is.

Because the two runs share an identical MyEnv, the alpha sweep, the frequency
windows and the observer grids are identical -> the comparison is pointwise, with
no interpolation, and every figure is the sweep_compare machinery with labels
('raref', 'ext') instead of ('fit', 'data').

Caveat carried into any writeup: `_ext` ran to it=190650, past both `itmax 80000`
in its own phys_input.ini and the ITMAX_ backstop in the committed
src/Initial/Shells/Shells.cpp, and its longer tail came from a 10x EXTRA_TIME set
at build time (setup.py: EXTRA_TIME = 0.05*env.tmax). It is NOT reproducible from
the committed source as it stands. The extra emission also comes from post-crash
material whose decay slope was measured to be position-dependent (-3.6 inner to
-6.0 outer, rarefaction_tail_exploration.py); fluence_split quantifies how much of
the difference lives there, it does not settle whether it is physical.

Example use in command line:
  python -c "import sweep_duration as D; D.check_common_span()"
  python -c "import sweep_duration as D; D.main(nproc=7)"
'''

import os
import numpy as np
import matplotlib.pyplot as plt

from environment import GAMMA_dir, MyEnv
from IO import open_celldata, get_variable
from working_cooling_data import generate_cell_fromData
import sweep_compare as cmp
from sweep_gammacm import (run_sweep, load_sweep, method_outdir, data_end_barT,
    exit_onset_barT, rarefaction_off_barT, compute_alpha_sweep, trim_pngs,
    LOG10RATIO_ARR, Z_SHELL, R_REF)

KEY_A, KEY_B = 'cooling_fid_raref', 'cooling_fid_raref_ext'   # DELETED, see the module note
LABELS = ('raref', 'ext')
OUTDIR = os.path.join(GAMMA_dir, 'bin', 'Tools', 'figures', 'duration_compare')
METHOD = 'data'           # the whole point: only the data path sees the extra snapshots
CELLS = (25, 100, 300, 519)          # distribution-diagnostic cells; the RS runs inward in
                                     # index, so 519 is CD-adjacent (shocked first, bar{T}~0)
                                     # and 25 is the outer edge (shocked last)
LOGR_DIST = (-3., 0., 2.)            # and the regimes to show them in


def check_common_span(key_a=KEY_A, key_b=KEY_B, k=100, rtol=1e-10):
  '''
  The two runs are the same simulation, so their cell histories must agree on the
  overlapping iterations -- if they do not, the runs differ by something other than
  their duration and the whole comparison is void. Prints the max relative deviation
  in rho, p, vx and x over the shared `it` values. Returns True if within rtol.
  '''
  da, db = open_celldata(key_a, k), open_celldata(key_b, k)
  if da is False or db is False:
    print(f'cell {k} missing from {key_a if da is False else key_b}')
    return False
  a, b = da, db                       # open_celldata already indexes on the iteration
  shared = a.index.intersection(b.index)
  print(f'cell {k}: {len(a)} rows ({key_a}) vs {len(b)} ({key_b}), '
        f'{len(shared)} shared iterations up to it={shared.max()}, '
        f't={a.loc[shared.max(), "t"]:.6g}')
  ok = True
  for c in ('rho', 'p', 'vx', 'x'):
    xa, xb = a.loc[shared, c].to_numpy(float), b.loc[shared, c].to_numpy(float)
    den = np.where(np.abs(xa) > 0., np.abs(xa), 1.)
    dev = np.max(np.abs(xb - xa)/den)
    ok &= dev <= rtol
    print(f'  max |d{c}|/{c} = {dev:.3e}')
  print('  -> identical over the shared span' if ok else
        f'  -> DEVIATION beyond rtol={rtol:g}: the runs differ by more than duration')
  return bool(ok)


def plot_distrib_evolution(cells=CELLS, logr_list=LOGR_DIST, key_a=KEY_A, key_b=KEY_B,
    labels=LABELS, outdir=OUTDIR, r_ref=R_REF):
  '''
  The direct view of what the sweep measures: the electron bounds gmin, gmax and
  the accumulated cooling fluence tt = int dt'/t_c1 along the SAME cells' worldlines,
  followed to each run's end of data. The short run is drawn as a wide pale halo and
  the extended one as a thin line on top: the halo must track the line exactly (same
  simulation) and simply stop early -- the visible overhang is the extra evolution
  only the longer run resolves. One figure per cooling regime.
  '''
  os.makedirs(outdir, exist_ok=True)
  env0 = MyEnv(key_a)
  alphas, logr0 = compute_alpha_sweep(key_a, logr_list)
  print(f'baseline log10(gma_c/gma_m) = {logr0:.4f}')
  colors = plt.cm.viridis(np.linspace(0., .85, len(cells)))
  # (linewidth, alpha) per side: wide pale halo = short run, thin line = extended
  styles = ((3.2, .30), (1.1, 1.))

  for logr, alpha in zip(logr_list, alphas):
    fig, axs = plt.subplots(2, 1, figsize=(7., 7.5), sharex=True)
    gm = gc = None
    ttmax = 0.
    for k, c in zip(cells, colors):
      for key, (lw, al) in zip((key_a, key_b), styles):
        d = open_celldata(key, k)
        if d is False:
          continue
        cell, cenv = generate_cell_fromData(d, env0, alpha=alpha, r_ref=r_ref)
        if cell is False:
          print(f'  {key} k={k} logr={logr:+.0f}: no usable history')
          continue
        gm, gc = cenv.gma_m, cenv.gma_c
        bt = (get_variable(cell, 'Ton', cenv) - cenv.Ts)/cenv.T0
        axs[0].loglog(bt, cell.gmax, color=c, lw=lw, alpha=al, solid_capstyle='round',
                      label=(f'k={k}' if al == 1. else None))
        axs[0].loglog(bt, cell.gmin, color=c, lw=.6*lw, alpha=.6*al, ls=':')
        pos = cell.tt.to_numpy(float) > 0.        # tt starts at exactly 0 (log axis)
        axs[1].loglog(bt[pos], cell.tt[pos], color=c, lw=lw, alpha=al,
                      solid_capstyle='round')
        ttmax = max(ttmax, float(cell.tt.max()))
    if gm is not None:
      for v, lab in ((gm, '$\\gamma_m$'), (gc, '$\\gamma_c$')):
        axs[0].axhline(v, color='grey', ls='-.', lw=.8)
        axs[0].annotate(lab, (0.995, v), xycoords=('axes fraction', 'data'), fontsize=9,
                        color='grey', va='bottom', ha='right')
    for ax in axs:
      for b, ls, lab in zip((data_end_barT(key_a), data_end_barT(key_b)), ('--', '-'),
                            labels):
        if b:
          ax.axvline(b[1], color='crimson', ls=ls, lw=.9, alpha=.8)
    axs[1].set_ylim(max(ttmax*1e-5, 1e-300), ttmax*3.)   # tt spans to 0; show the plateau
    axs[0].set_ylabel('$\\gamma_{\\max}$ (solid), $\\gamma_{\\min}$ (dotted)')
    axs[0].plot([], [], 'k-', lw=styles[0][0], alpha=styles[0][1], label=labels[0])
    axs[0].plot([], [], 'k-', lw=styles[1][0], label=labels[1])
    axs[0].plot([], [], color='crimson', lw=.9, label='end of data')
    axs[0].legend(fontsize=8, ncol=2, loc='lower left', framealpha=.85)
    axs[1].set_ylabel('$\\tilde{t}=\\int \\mathrm{d}t\'/t_{c,1}$')
    axs[1].set_xlabel('$\\bar{T}=(T_{\\rm obs}-T_s)/T_0$')
    axs[0].set_title(f'Electron bounds along the worldline, '
                     f'$\\log_{{10}}(\\gamma_c/\\gamma_m)={logr:+.0f}$')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f'distrib_evolution_logr={logr:+.0f}.png'), dpi=300)
    plt.close(fig)
  print(f'distribution figures saved to {outdir}')


def main(key_a=KEY_A, key_b=KEY_B, log10ratio_arr=LOG10RATIO_ARR, outdir=OUTDIR,
    use_cache=True, nproc=None, labels=LABELS, distrib=True):
  '''
  Ensure the data-method sweep exists for both runs (computing whichever is
  missing, each in its own key-tagged cache -- see sweep_gammacm.method_outdir),
  then build every comparison figure with the short run as side A.
  '''
  os.makedirs(outdir, exist_ok=True)
  for key in (key_a, key_b):
    d = method_outdir(METHOD, key)
    os.makedirs(d, exist_ok=True)
    if not (use_cache and load_sweep(d)):
      print(f'--- running the {METHOD} sweep on {key} ---')
      run_sweep(key, log10ratio_arr, method=METHOD, nproc=nproc)
  pairs = cmp.load_pairs(method_outdir(METHOD, key_a), method_outdir(METHOD, key_b))

  barT_f = exit_onset_barT(key_a, z=Z_SHELL)
  barT_off = rarefaction_off_barT(key_a, z=Z_SHELL)
  barT_end = (data_end_barT(key_a, z=Z_SHELL), data_end_barT(key_b, z=Z_SHELL))
  print(f'crossing bar_T_f = {barT_f:.4f}' +
        (f', rarefaction cut-off {barT_off[0]:.4f}..{barT_off[1]:.4f}' if barT_off else ''))
  for key, b, lab in zip((key_a, key_b), barT_end, labels):
    if b:
      print(f'end of data ({lab}): bar_T = {b[0]:.4f}..{b[1]:.4f}')

  cmp.plot_efficiency_compare(pairs, outdir=outdir, labels=labels)
  for kind in ('peak', 'fluence'):
    for mode in ('nu_m', 'max'):
      cmp.plot_spectra_compare(pairs, kind=kind, mode=mode, outdir=outdir, labels=labels)
  cmp.plot_lightcurve_compare(pairs, barT_f, barT_off=barT_off, outdir=outdir,
      labels=labels, barT_end=barT_end)
  s = cmp.plot_summary_ratios(pairs, outdir=outdir, labels=labels)
  fs = cmp.fluence_split(pairs, barT_off[1] if barT_off else None, outdir=outdir,
      labels=labels, cut_label='R_rar cut-off', cut_math='$R_{\\rm rar}$ cut-off')

  if distrib:
    plot_distrib_evolution(key_a=key_a, key_b=key_b, labels=labels, outdir=outdir)
  trim_pngs(outdir)
  print(f'Duration-comparison figures saved to {outdir}')
  return pairs, s, fs
