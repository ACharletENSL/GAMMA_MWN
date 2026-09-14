# -*- coding: utf-8 -*-
# @Author: acharlet

'''
How much does each shell contribute, and how does the split move with the
cooling regime?

Every other sweep_*.py module computes the FAST shell alone -- the electrons
accelerated behind the reverse shock (Z_SHELL = 4). This one adds the SLOW shell
(forward-shocked, z = 1) and shows the two side by side with their sum, one
figure per cooling regime (3 curves per panel, hence the split by regime).

Computation: METHOD, which is sweep_gammacm.DEFAULT_METHOD and so has been
'data_rarcut' since 2026-09-07 -- get_shell_nuFnu_fromData with rar_cut='model',
the modelled sharp cut-off at R_rar, on the fiducial run cooling_g100. It is the
article's own prescription and the same numbers sweep_rarcut uses as its side A,
so the RS side here IS that cached sweep, reused unchanged.

WHICH DIRECTORY HOLDS WHICH METHOD moved when that default did.
`figures/<run>/shells_split` is whatever METHOD is TODAY, i.e. the RARCUT set;
any other method gets a '_{method}' suffix, so the uncut reference is
`shells_split_data` (method='data', rar_cut=None, every cell followed to its
last snapshot). The suffix is therefore relative to the CURRENT default, not to
a fixed method: a directory written before 2026-09-07 carries the opposite
mapping, because back then METHOD was 'data' and the rarcut set was the suffixed
one. One such leftover, `shells_split_data_rarcut` (11 August, the same physics
as `shells_split`), was deleted on 2026-09-14. If another appears, read the
mtimes and not the name, and regenerate rather than trust it.

Two facts make the sum well defined:

  GRIDS are already shared. Both drivers build nuobs = nub*env.nu0 and
  Tobs = env.Ts + (T-1)*env.T0 from the REVERSE-shock scalars whatever z is
  (working_cooling_data.py, get_shell_nuFnu_fromData), so a z=1 and a z=4 point
  of the same alpha land on the same observer grid, and can be added pointwise.

  FLUXES are not. get_nuFnu divides by nu0FS and get_Lnu_comov by L0pFS whenever
  the cell's own trac says forward shock, so a returned value is
  nuFnu_phys/(nu0_s*L0p_s). With nu0FS = nu0/fac_nu and L0pFS = L0p/fac_F,
  putting the FS on the RS scale is one constant:
      nuFnu_FS|RS = nuFnu_FS / (fac_nu*fac_F)          (= x0.51826 on this run)
  fac_nu and fac_F are ratios of Lorentz factors and velocities, invariant under
  the Granot length-time rescale, so it is ONE number for the whole sweep.

Both shells must share the same alpha (that is what makes them the same physical
collision), so they are NOT at the same log10(gamma_c/gamma_m): the FS sits at a
fixed offset above the RS (+0.50 dex here). Figures are indexed by the RS value
-- the sweep parameter -- with the FS value annotated.

Example use in command line:
  python -c "import sweep_shells as S; S.main(nproc=7)"                    # rarcut
  python -c "import sweep_shells as S; S.main(method='data', nproc=7)"     # uncut
'''

import os
import numpy as np
import matplotlib.pyplot as plt
from functools import lru_cache

from environment import MyEnv, GAMMA_dir, figdir
from IO import get_dirpath
from peak_modeling import offset_gcgm_from_au
from plotting_functions import nF_label, transy, COL_RS, COL_FS, COL_TOT
from spectrum_shape import peak_and_width
                                  # the suite's ONE half-maximum width construction;
                                  # see shell_widths for why the sum needs a width and
                                  # not a pair of breaks
from sweep_compare import _regrid_onto
from sweep_gammacm import (run_sweep, load_sweep, method_outdir, compute_alpha_sweep,
    exit_onset_barT, rarefaction_off_barT, nu_over_num, bolometric_peak_index,
    compute_fluence_spectrum, _plot_spectra_all, trim_pngs, LOG10RATIO_ARR,
    NU_TARGETS, NU_REF, NU_M_LABEL, SPEC_YSPAN, XLIM_LIN, DEFAULT_METHOD)

KEY = 'cooling_g100'
METHOD = DEFAULT_METHOD           # reference computation, imported from sweep_gammacm
                                  # ('data_rarcut' since 2026-09-07)
Z_RS, Z_FS = 4, 1                 # reverse (fast) shell, forward (slow) shell
OUTDIR_NAME = 'shells_split'          # figdir(OUTDIR_NAME, key) puts it under the
OUTDIR = figdir(OUTDIR_NAME)    # run's own folder; this is the fiducial's

# one style per curve, used by every figure here. Colours come from the suite-wide
# convention (plotting_functions: RS red, FS blue, sum black -- Charlet et al. 2025);
# only the line styles are local, so the shells stay separable in print.
STY = {
  'RS':  dict(color=COL_RS,  ls='--', lw=1.5, label='RS (fast shell)'),
  'FS':  dict(color=COL_FS,  ls=':',  lw=1.8, label='FS (slow shell)'),
  'tot': dict(color=COL_TOT, ls='-',  lw=1.5, label='total'),
}
YSPAN_LC = 6.      # decades of flux shown on the log-log lightcurve panels, below the
                   # total's peak
XMAX_LIN = XLIM_LIN[1]   # right edge of the LINEAR lightcurve panels, in bar{T}/bar{T}_f,RS.
                   # The suite-wide window (sweep_gammacm.XLIM_LIN), shared with the
                   # single-method and A/B lightcurves; both shells are switched off well
                   # before it


def fs_to_rs_factor(key=KEY, env=None):
  '''
  The constant that puts a forward-shock nuFnu on the reverse-shock scale,
  1/(fac_nu*fac_F) -- see the module docstring. Read off `env` when it carries
  the two factors (points cached since sweep_gammacm._ENV_KEYS_OPT), else from a
  fresh MyEnv(key); both give the same number, the factors being alpha-invariant.
  '''
  if env is None or not (hasattr(env, 'fac_nu') and hasattr(env, 'fac_F')):
    env = MyEnv(key)
  return 1./(env.fac_nu*env.fac_F)


def load_shell_pairs(key=KEY, method=METHOD, log10ratio_arr=LOG10RATIO_ARR):
  '''
  Pair the two cached shell sweeps regime by regime and build the sum. Matches on
  log10ratio (the RS target, i.e. the sweep parameter), checks that the paired
  points really are the same physical rescale (same alpha), puts both on one grid
  and the FS on the RS flux scale, then adds. Returns one dict per regime with
  Tb, nub (shared), nuFnu_rs/_fs/_tot, the two energy budgets and env.
  '''
  d_rs, d_fs = method_outdir(method, key, Z_RS), method_outdir(method, key, Z_FS)
  res_rs, res_fs = load_sweep(d_rs), load_sweep(d_fs)
  if not res_rs or not res_fs:
    missing = [d for d, r in ((d_rs, res_rs), (d_fs, res_fs)) if not r]
    raise FileNotFoundError(f'no cached sweep in {missing} -- run sweep_shells.main first')
  by_logr = {round(r['log10ratio'], 6): r for r in res_fs}
  fac = fs_to_rs_factor(key, res_fs[0]['env'])
  print(f'FS -> RS flux scale: x{fac:.6f}  (1/(fac_nu*fac_F))')

  pairs = []
  for rs in res_rs:
    logr = round(rs['log10ratio'], 6)
    if logr not in by_logr:
      print(f'no FS point at log10ratio={logr:+.1f}, skipped')
      continue
    fs = by_logr[logr]
    if not np.isclose(rs['alpha'], fs['alpha'], rtol=1e-6):
      raise ValueError(f'log10ratio={logr:+.1f}: alpha mismatch RS {rs["alpha"]:.6g} '
                       f'vs FS {fs["alpha"]:.6g} -- the two shells are not the same rescale')
    # same _nu_window and NT for both, so this is normally a no-op; it only absorbs
    # the +-1 Nnu rounding of the points-per-decade (as sweep_compare.load_pairs does)
    fs = _regrid_onto(fs, rs)
    nf_rs, nf_fs = rs['nuFnu'], fs['nuFnu']*fac
    e = rs['env']
    pairs.append(dict(log10ratio=rs['log10ratio'], alpha=rs['alpha'], env=e,
        Tb=rs['Tb'], nub=rs['nub'], x=nu_over_num(rs),
        nuFnu_rs=nf_rs, nuFnu_fs=nf_fs, nuFnu_tot=nf_rs + nf_fs,
        E_rad_rs=rs.get('E_rad', np.nan), E_inj_rs=rs.get('E_inj', np.nan),
        E_rad_fs=fs.get('E_rad', np.nan), E_inj_fs=fs.get('E_inj', np.nan)))
  print(f'paired {len(pairs)} regimes (RS {d_rs.split("/")[-1]} | FS {d_fs.split("/")[-1]})')
  return pairs


@lru_cache(maxsize=None)
def _regime_offset(key=KEY, verbose=True):
  '''log10 of (gma_cFS/gma_mFS)/(gma_c/gma_m): the two shells share alpha, so the
  FS sits at this fixed offset above the RS target, always > 0 (the FS is further
  into slow cooling). Exact, from the loaded env; cross-checked against the
  analytic a_u law offset_gcgm_from_au, which assumes the fiducial family
  Ek1 = Ek4, D01 = D04 -- a large mismatch means this setup is off that family.'''
  e0 = MyEnv(key)
  off = np.log10((e0.gma_cFS/e0.gma_mFS)/(e0.gma_c/e0.gma_m))
  off_au = np.log10(offset_gcgm_from_au(e0.a_u))
  if verbose:
    print(f'RS->FS cooling-regime offset: {off:+.4f} dex (exact), '
          f'{off_au:+.4f} dex (analytic at a_u = {e0.a_u:g})')
    if abs(off - off_au) > 0.02:
      print(f'  /!\\ {abs(off-off_au):.3f} dex mismatch: setup is off the '
            f'Ek1 = Ek4, D01 = D04 family the a_u law assumes')
  return off


def _regime_title(p, key=KEY):
  '''Both shells' cooling regimes: they share alpha, so the FS sits at a fixed
  offset above the RS target (see _regime_offset).'''
  off = _regime_offset(key)
  return (f'$\\log_{{10}}\\mathcal{{C}} = {p["log10ratio"]:+.1f}$ (RS), '
          f'${p["log10ratio"] + off:+.1f}$ (FS)')


def _curves(p):
  '''(key, nuFnu) for the three curves, in draw order (sum last, on top).'''
  return (('RS', p['nuFnu_rs']), ('FS', p['nuFnu_fs']), ('tot', p['nuFnu_tot']))


def _lc_marks(key=KEY, verbose=True):
  '''Per shell, the two hydro/geometry times drawn on every lightcurve panel:
  its shell-crossing bar{T}_f (exit_onset_barT) and its (first, last) rarefaction
  shut-off bar{T} (rarefaction_off_barT). Both alpha-independent, so one set for
  the whole sweep. All in RS units of bar{T}, the shared clock.'''
  marks = {tag: (exit_onset_barT(key, z=z), rarefaction_off_barT(key, z=z))
           for tag, z in (('RS', Z_RS), ('FS', Z_FS))}
  if verbose:
    for tag, (bf, boff) in marks.items():
      txt = f'{boff[0]:.3f}..{boff[1]:.3f}' if boff else 'n/a'
      print(f'  {tag}: crossing bar_T_f = {bf:.4f}, rarefaction off {txt}')
  return marks


def plot_lightcurves_per_regime(pairs, key=KEY, nu_targets=NU_TARGETS, outdir=OUTDIR):
  '''
  One figure per regime, one panel per observing frequency (as a fraction nu_t of
  the RS spectral peak nu_pk = max(nu_m, nu_c), i.e. the stored nub axis), each
  showing the two shells and their sum against bar{T} = (Tobs-Ts)/T0 in REVERSE-
  shock units -- the shared clock, so the FS's earlier arrival is readable
  directly. Per shell: a dotted vline at its shell-crossing bar{T}_f and a shaded
  band over its rarefaction shut-off range (past the band's right edge that shell
  emits only the tT^-2 high-latitude tails of what it already radiated).
  Log-log; see plot_lightcurves_norm_per_regime for the normalised linear view.
  '''
  os.makedirs(outdir, exist_ok=True)
  marks = _lc_marks(key)

  for p in pairs:
    barT = p['Tb'] - 1.
    fig, axs = plt.subplots(1, len(nu_targets), figsize=(4.7*len(nu_targets), 4.5),
                            sharex=True)
    axs = np.atleast_1d(axs)
    for ax, nu_t in zip(axs, nu_targets):
      inu = min(np.searchsorted(p['nub'], nu_t), len(p['nub']) - 1)
      ymax = 0.
      for tag, nF in _curves(p):
        lc = nF[:, inu]
        ax.loglog(barT, np.where(lc > 0., lc, np.nan), **STY[tag])
        ymax = max(ymax, np.nanmax(lc))
      for tag, (bf, boff) in marks.items():
        c = STY[tag]['color']
        ax.axvline(bf, color=c, ls=':', lw=.8)
        if boff:
          ax.axvspan(boff[0], boff[1], color=c, alpha=0.10, lw=0, zorder=0)
      if ymax > 0.:
        ax.set_ylim(ymax*10.**(-YSPAN_LC), ymax*3.)
      ax.set_xlim(1e-3, barT.max())
      ax.set_xlabel('$\\bar{T} = (T_{\\rm obs}-T_s)/T_0$   (RS units)')
      ax.set_title(f'$\\nu = {nu_t:g}\\,\\nu_{{\\rm pk,RS}}$', fontsize=11)
    axs[0].set_ylabel(nF_label)
    axs[0].legend(fontsize=9, loc='lower left')
    fig.suptitle('Shell contributions to the lightcurve,   ' + _regime_title(p, key))
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f'lightcurves_logr={p["log10ratio"]:+.1f}.png'), dpi=300)
    plt.close(fig)
  print(f'{len(pairs)} per-regime lightcurve figures saved to {outdir}')


def plot_lightcurves_norm_per_regime(pairs, key=KEY, nu_targets=NU_TARGETS,
    outdir=OUTDIR, xmax=XMAX_LIN):
  '''
  The shape-normalised, LINEAR counterpart of plot_lightcurves_per_regime, on the
  same normalisations as the one-shell figures (sweep_gammacm.plot_lightcurve_shape)
  but referred to the TOTAL rather than to each curve's own peak:

    y = nu F_nu / (nu F_nu)_max of the TOTAL at that frequency
    x = bar{T}/bar{T}_f  with bar{T}_f the REVERSE-shock crossing time

  So the black sum peaks at y=1 in every panel and the two shells read directly as
  the fraction of the total each supplies, instant by instant -- which peak-normalising
  each curve separately (the one-shell convention) would destroy. Both shells keep
  the same x unit for the same reason: bar{T}_f,RS is one clock, and the FS's own
  crossing is drawn as its vline (at 0.81 here) rather than used to rescale it.
  Linear axes, so this is the view of the pulse shape and of where in the pulse each
  shell dominates; the log-log version covers the tails.
  '''
  os.makedirs(outdir, exist_ok=True)
  marks = _lc_marks(key, verbose=False)
  bf_rs = marks['RS'][0]
  if not bf_rs > 0.:
    raise ValueError(f'non-positive RS crossing time bar_T_f={bf_rs}')

  for p in pairs:
    x = (p['Tb'] - 1.)/bf_rs
    fig, axs = plt.subplots(1, len(nu_targets), figsize=(4.7*len(nu_targets), 4.5),
                            sharex=True, sharey=True)
    axs = np.atleast_1d(axs)
    for ax, nu_t in zip(axs, nu_targets):
      inu = min(np.searchsorted(p['nub'], nu_t), len(p['nub']) - 1)
      pk = np.nanmax(p['nuFnu_tot'][:, inu])          # ONE normalisation per panel
      if not pk > 0.:
        continue
      for tag, nF in _curves(p):
        ax.plot(x, nF[:, inu]/pk, **STY[tag])
      for tag, (bf, boff) in marks.items():
        c = STY[tag]['color']
        ax.axvline(bf/bf_rs, color=c, ls=':', lw=.8)
        if boff:
          ax.axvspan(boff[0]/bf_rs, boff[1]/bf_rs, color=c, alpha=0.10, lw=0, zorder=0)
      ax.axhline(1., color='grey', ls=':', lw=.7)
      ax.set_xlim(0., xmax)
      ax.set_ylim(0., 1.15)
      ax.set_xlabel('$\\bar{T}/\\bar{T}_{f,\\rm RS}$')
      ax.set_title(f'$\\nu = {nu_t:g}\\,\\nu_{{\\rm pk,RS}}$', fontsize=11)
    axs[0].set_ylabel('$\\nu F_\\nu/(\\nu F_\\nu)_{\\rm max,tot}$')
    axs[0].legend(fontsize=9, loc='upper right')
    fig.suptitle('Shell contributions to the lightcurve (normalised to the total),   '
                 + _regime_title(p, key))
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f'lightcurves_norm_logr={p["log10ratio"]:+.1f}.png'),
                dpi=300)
    plt.close(fig)
  print(f'{len(pairs)} per-regime normalised linear lightcurve figures saved to {outdir}')


def _spectrum_figure(p, spectra, title, ylabel, fname, key, outdir, extra=None):
  '''Shared body of the two spectral figures: three curves on nu/nu_m,RS, the two
  shells' injection frequencies marked, y clipped to SPEC_YSPAN decades below the
  total's peak and x clipped to where anything is visible.'''
  e0 = MyEnv(key)
  x = p['x']
  ymax = max(np.nanmax(s) for _, s in spectra)
  ylo = ymax*10.**(-SPEC_YSPAN)
  fig, ax = plt.subplots(figsize=(6.6, 4.8))
  for tag, s in spectra:
    ax.loglog(x, np.where(s > 0., s, np.nan), **STY[tag])
  for nu_m, tag in ((1., 'RS'), (e0.nu0FS/e0.nu0, 'FS')):
    ax.axvline(nu_m, color=STY[tag]['color'], ls='-.', lw=.8, alpha=.6)
    ax.annotate(f'$\\nu_{{m,\\rm {tag}}}$', (nu_m, ymax*1.5), color=STY[tag]['color'],
                fontsize=9, ha='center', va='bottom')
  vis = np.any(np.array([s for _, s in spectra]) > ylo, axis=0)
  if vis.any():
    ax.set_xlim(x[vis].min()/3., x[vis].max()*3.)
  ax.set_ylim(ylo, ymax*6.)
  ax.set_xlabel(NU_M_LABEL + '   (RS)')
  ax.set_ylabel(ylabel)
  ax.set_title(title + ',\n' + _regime_title(p, key), fontsize=11)
  ax.legend(fontsize=9, loc='lower center')
  if extra:
    ax.annotate(extra, (0.02, 0.95), xycoords='axes fraction', fontsize=9, va='top')
  fig.tight_layout()
  fig.savefig(os.path.join(outdir, fname), dpi=300)
  plt.close(fig)


def plot_fluence_spectra_per_regime(pairs, key=KEY, outdir=OUTDIR):
  '''
  One figure per regime: the time-integrated spectrum (int nuFnu d bar{T}, over the
  RS clock shared by both shells) of each shell and of their sum. This is where the
  two shells separate most cleanly -- nu_m,FS sits 1.18 decades below nu_m,RS on
  this run -- so it reads as which shell owns which part of the observed band.
  '''
  os.makedirs(outdir, exist_ok=True)
  for p in pairs:
    sp = [(tag, compute_fluence_spectrum(p['Tb'], nF)) for tag, nF in _curves(p)]
    _spectrum_figure(p, sp, 'Shell contributions to the fluence spectrum',
        '$\\int (\\nu F_\\nu/\\nu_0 F_0) \\, {\\rm d}\\bar{T}$',
        f'fluence_spectrum_logr={p["log10ratio"]:+.1f}.png', key, outdir)
  print(f'{len(pairs)} per-regime fluence-spectrum figures saved to {outdir}')


def plot_total_fluence_all(pairs, mode, outdir=OUTDIR):
  '''
  Every regime's TOTAL time-integrated spectrum (both shocks summed) on one panel,
  coloured by log10(gamma_c/gamma_m) -- the two-shell counterpart of
  sweep_gammacm's time-integrated panel, built with the same _plot_spectra_all so the
  normalisation, y-clip and axis conventions match the one-shell figures exactly and
  the two can be laid side by side. mode: 'nu_m' normalises each curve at nu_m,
  'max' at its own peak.
  A `pairs` entry carries the log10ratio, env and nub that _plot_spectra_all needs,
  so it doubles as a `results` entry; only the spectrum getter differs (nuFnu_tot
  instead of nuFnu). The x axis stays nu/nu_m,RS -- the FS has its own nu_m a
  decade below, and the sum has no single injection frequency to normalise by.
  '''
  os.makedirs(outdir, exist_ok=True)
  def get_total_fluence(p):
    return compute_fluence_spectrum(p['Tb'], p['nuFnu_tot'])
  norm_txt = 'normalised at $\\nu_\\mathrm{m}$' if mode == 'nu_m' else 'peak-normalised'
  _plot_spectra_all(pairs, get_total_fluence, mode,
      f'Total time-integrated spectra, both shocks ({norm_txt})',
      f'total_fluence_spectra_norm-{mode}.png', outdir, sym='\\nu \\mathcal{F}_\\nu')
  print(f'all-regime total fluence spectra (norm {mode}) saved to {outdir}')


def plot_peak_spectra_per_regime(pairs, key=KEY, nu_ref=NU_REF, outdir=OUTDIR):
  '''
  One figure per regime: the INSTANTANEOUS spectra of both shells and their sum, all
  three taken at the same observer time -- the peak of the TOTAL lightcurve at
  peak of the TOTAL's BOLOMETRIC lightcurve (bolometric_peak_index on the sum), as every
  peak spectrum in the suite is. Taking each shell at its own peak would compare
  different instants and could not be added; here the three curves are simultaneous, so
  the sum is literally the sum of the two below it.
  '''
  os.makedirs(outdir, exist_ok=True)
  for p in pairs:
    ipk = bolometric_peak_index(p['nuFnu_tot'], p['nub'])
    if ipk is None:
      print(f'log10ratio={p["log10ratio"]:+.1f}: no peak found, skipped')
      continue
    barT_pk = p['Tb'][ipk] - 1.
    sp = [(tag, nF[ipk, :]) for tag, nF in _curves(p)]
    _spectrum_figure(p, sp, 'Shell contributions at the peak of the total lightcurve',
        nF_label, f'peak_spectrum_logr={p["log10ratio"]:+.1f}.png', key, outdir,
        extra=f'$\\bar{{T}}_{{\\rm pk}} = {barT_pk:.3f}$  (at $\\nu={nu_ref:g}\\,\\nu_{{\\rm pk,RS}}$)')
  print(f'{len(pairs)} per-regime peak-spectrum figures saved to {outdir}')


# --- the composite figures: what adding the slow shell does to the observed spectrum ----
# The per-regime figures above draw one regime per file, all nine of them. These two draw
# the cooling SEQUENCE: three regimes side by side with the sum's departure from the RS
# alone on its own axis, and then that departure reduced to one number -- the width of the
# peak it leaves behind -- for every regime of the sweep.

LOGR_PANELS = (-2., 0., 2.)   # one column per regime: deep fast cooling, marginal, deep
                              # slow cooling. The three points of LOG10RATIO_ARR that
                              # bracket the sequence, so the composite says what the nine
                              # per-regime files say without being nine files.
PANEL_YSPAN = 5.              # decades of flux shown on a composite's spectral panel, below
                              # the TOTAL's peak. Wider than SPEC_YSPAN (3.55, the per-regime
                              # figures' span) on purpose: the sum's half-maximum width alone
                              # reaches 3.3 dex in fast cooling, and the point of these
                              # panels is the low-frequency end, where the FS peaks a decade
                              # below the RS and the two curves separate.
WIDTH_LEVEL = 'half'          # which of spectrum_shape.WIDTH_LEVELS the width is quoted at.
                              # 'half' is nu_-1/2 and nu_+1/2, the frequencies at half the
                              # maximum nu F_nu either side of the peak, so
                              # W_pk = log10(nu_+1/2/nu_-1/2) is the peak's width in decades.
                              # Quoted POSITIVE, hi/lo: a width is a span, and the suite's
                              # other half-maximum widths (spectrum_shape's logW_half, the
                              # pulse width in lightcurve_shape) are all hi/lo too.
KIND_SYM = {'peak': '\\nu F_\\nu', 'fluence': '\\nu \\mathcal{F}_\\nu'}
KIND_NAME = {'peak': 'spectra at the peak of the total lightcurve',
             'fluence': 'time-integrated spectra'}


def shell_spectra(p, kind):
  '''
  {tag: spectrum} for the three curves of one regime, and the observer time it was taken at
  (None for the time-integrated kind).

  kind 'peak'    -- all three at the SAME instant, the bolometric peak of the TOTAL, as
                    plot_peak_spectra_per_regime takes them: taking each shell at its own
                    peak would compare different instants and the black curve would no
                    longer be the sum of the two below it.
       'fluence' -- integrated over the shared RS clock (compute_fluence_spectrum).
  '''
  if kind not in KIND_SYM:
    raise ValueError(f'kind must be one of {tuple(KIND_SYM)}, got {kind!r}')
  if kind == 'fluence':
    return {tag: compute_fluence_spectrum(p['Tb'], nF) for tag, nF in _curves(p)}, None
  ipk = bolometric_peak_index(p['nuFnu_tot'], p['nub'])
  if ipk is None:
    return None, None
  return {tag: nF[ipk, :] for tag, nF in _curves(p)}, float(p['Tb'][ipk] - 1.)


def shell_widths(pairs, kind='peak', level=WIDTH_LEVEL, verbose=True):
  '''
  How wide the SED peak of each of the three curves is, per regime:
  W_pk = log10(nu_+1/2/nu_-1/2), the frequencies at half the maximum nu F_nu either side of
  it. spectrum_shape.peak_and_width does the measuring -- the suite's ONE width
  construction, shared with the peak-vs-time-integrated comparison and, through
  lightcurve_shape._level_cross, with the pulse width, so a width in frequency and a width
  in time are the same construction and cannot drift into two versions of it.

  No fit and no break, which is what makes this the quantity to quote for the SUM: two
  shells whose nu_m sit a decade apart do not add up to a Granot & Sari shape, so there is
  no pair of breaks to report, but there is always a maximum and two half-maximum
  crossings.

  edge_lo/edge_hi flag a crossing found on the grid edge -- the width is then the observing
  WINDOW's and not the spectrum's. None of the fiducial's 9 regimes trips one on either
  kind; the flag is carried, and printed, so that a narrower band cannot pass one off as a
  measurement.

  Returns one row per regime: dict(logr, barT_pk, RS=, FS=, tot=) with each shell's entry
  the full peak_and_width dict.
  '''
  rows = []
  for p in pairs:
    sp, barT_pk = shell_spectra(p, kind)
    if sp is None:
      print(f'log10ratio={p["log10ratio"]:+.1f}: no peak found, skipped')
      continue
    row = dict(logr=float(p['log10ratio']), barT_pk=barT_pk)
    row.update({tag: peak_and_width(p['x'], sp[tag]) for tag in sp})
    rows.append(row)
  if verbose:
    w = lambda r, t: r[t][f'logW_{level}']
    print(f'\n{KIND_NAME[kind]}: peak width W_pk = log10(nu_+1/2/nu_-1/2) [dex]')
    print(f"{'log10(C)':>9} {'W RS':>8} {'W FS':>8} {'W RS+FS':>9} {'tot-RS':>8} "
          f"{'nu_pk RS':>10} {'nu_pk tot':>10}  edge")
    for r in rows:
      edge = ','.join(t for t in ('RS', 'FS', 'tot')
                      if r[t]['edge_lo'] or r[t]['edge_hi']) or '-'
      print(f"{r['logr']:+9.0f} {w(r,'RS'):8.3f} {w(r,'FS'):8.3f} {w(r,'tot'):9.3f} "
            f"{w(r,'tot')-w(r,'RS'):+8.3f} {r['RS']['x_pk']:10.3e} "
            f"{r['tot']['x_pk']:10.3e}  {edge}")
  return rows


def _nu_m_lines(ax, e0, labels=False):
  '''The two shells' injection frequencies on a nu/nu_m,0 axis: the RS's is 1 by
  construction (the axis IS its nu_m), the FS's sits at nu0FS/nu0 below it.'''
  for nu_m, tag in ((1., 'RS'), (e0.nu0FS/e0.nu0, 'FS')):
    ax.axvline(nu_m, color=STY[tag]['color'], ls='-.', lw=.8, alpha=.6)
    if labels:
      ax.annotate(f'$\\nu_{{\\mathrm{{m}},\\!\\mathrm{{{tag}}}}}$', (nu_m, 1.01),
                  xycoords=transy(ax),
                  color=STY[tag]['color'], fontsize=8, ha='center', va='bottom')


def plot_shell_spectra_panels(pairs, kind='peak', logr_list=LOGR_PANELS, key=KEY,
    outdir=OUTDIR, yspan=PANEL_YSPAN, level=WIDTH_LEVEL):
  '''
  One column per regime: the two shells and their sum on top, and the sum's departure from
  the RS alone -- (RS+FS)/RS -- underneath.

  Each column is normalised by its OWN total's peak, so the black curve tops out at 1 in
  all three and the columns are read as SHAPES; how much each regime actually radiates is
  the separate cross-regime figure (plot_shell_shares).

  The ratio panel is what the figure is for. It is 1 wherever the RS owns the band and
  lifts where the FS does, and where it lifts is set by nu_m,FS sitting a decade below
  nu_m,RS (both marked) rather than by the FS being the brighter shell -- which is why the
  departure is a low-frequency one in fast cooling and nearly nothing in slow cooling,
  where the two shells' peaks have run together.

  The half-maximum crossings of the RS and of the sum are drawn as bars at their own
  half-peak level, so the two widths quoted in each panel can be read off the curves rather
  than taken on trust.
  '''
  os.makedirs(outdir, exist_ok=True)
  e0 = MyEnv(key)
  by_logr = {round(p['log10ratio'], 6): p for p in pairs}
  sel = [by_logr[round(lr, 6)] for lr in logr_list if round(lr, 6) in by_logr]
  missing = [lr for lr in logr_list if round(lr, 6) not in by_logr]
  if missing:
    print(f'no sweep point at log10ratio = {missing}; dropped from the composite')
  if not sel:
    print(f'{kind} spectra composite: nothing to plot')
    return
  ylo, rmax = 10.**(-yspan), 1.

  fig, axs = plt.subplots(2, len(sel), figsize=(4.3*len(sel), 5.9), sharex='col',
      sharey='row', squeeze=False, gridspec_kw={'height_ratios': [2.4, 1]})
  for j, p in enumerate(sel):
    ax_s, ax_r = axs[0, j], axs[1, j]
    sp, barT_pk = shell_spectra(p, kind)
    if sp is None:
      continue
    x, norm = p['x'], float(np.nanmax(sp['tot']))
    for tag, _ in _curves(p):
      y = sp[tag]/norm
      ax_s.loglog(x, np.where(y > 0., y, np.nan), **STY[tag])
    m = {tag: peak_and_width(x, sp[tag]) for tag in ('RS', 'tot')}
    for tag in ('RS', 'tot'):     # the half-maximum span, on the curve it was measured on
      lo, hi = m[tag][f'nu_lo_{level}'], m[tag][f'nu_hi_{level}']
      if np.isfinite(lo) and np.isfinite(hi):
        ax_s.plot([lo, hi], [0.5*m[tag]['F_pk']/norm]*2, color=STY[tag]['color'],
                  lw=.9, marker='|', ms=5, alpha=.85, zorder=5)
    with np.errstate(divide='ignore', invalid='ignore'):
      ratio = np.where(sp['RS'] > 0., sp['tot']/sp['RS'], np.nan)
    ax_r.plot(x, ratio, color=COL_TOT, lw=1.2)
    ax_r.set_xscale('log')
    ax_r.axhline(1., color='grey', ls=':', lw=.9)
    rmax = max(rmax, float(np.nanmax(ratio[np.isfinite(ratio)])) if np.isfinite(ratio).any()
               else 1.)
    for ax in (ax_s, ax_r):
      _nu_m_lines(ax, e0, labels=(ax is ax_s))
    vis = np.any(np.array([sp[t] for t, _ in _curves(p)])/norm > ylo, axis=0)
    if vis.any():
      ax_s.set_xlim(x[vis].min()/3., x[vis].max()*3.)
    # 1.5 decades of headroom above the total's peak, so the half-maximum bars (which sit
    # at half of it) clear the corner annotations below
    ax_s.set_ylim(ylo, 10.**1.5)
    # regime upper LEFT, widths upper RIGHT: the rising 4/3 end is at the bottom of the
    # panel and the cut-off has taken every curve down by the right edge, so both corners
    # are free on all three columns whatever the regime does to the peak's position
    ax_s.text(.03, .97, f'$\\log_{{10}}\\mathcal{{C}} = {p["log10ratio"]:+.0f}$',
              transform=ax_s.transAxes, ha='left', va='top', fontsize=11)
    for i, (tag, lab) in enumerate((('RS', 'RS'), ('tot', 'RS+FS'))):
      ax_s.text(.97, .97 - .085*i, f'$W_{{\\rm pk,{lab}}} = {m[tag][f"logW_{level}"]:.2f}$',
                transform=ax_s.transAxes, ha='right', va='top', fontsize=9,
                color=STY[tag]['color'])
    ax_r.set_xlabel(NU_M_LABEL + '   (RS)')
  axs[1, 0].set_ylim(1. - 0.03*(rmax - 1.), rmax + 0.08*(rmax - 1.))
  sym = KIND_SYM[kind]
  axs[0, 0].set_ylabel(f'${sym}/({sym})_{{\\rm max,tot}}$')
  axs[1, 0].set_ylabel('(RS+FS) / RS')
  h = [plt.Line2D([], [], **{k: v for k, v in STY[t].items() if k != 'label'})
       for t, _ in _curves(sel[0])]
  fig.legend(h, [STY[t]['label'] for t, _ in _curves(sel[0])], loc='lower center',
             ncol=3, fontsize=9, frameon=False)
  fig.tight_layout(rect=(0., 0.045, 1., 1.))
  f = os.path.join(outdir, f'shell_spectra_panels_{kind}.png')
  fig.savefig(f, dpi=300)
  plt.close(fig)
  print(f'{kind} spectra composite ({len(sel)} regimes) saved to {f}')


def plot_shell_width_vs_regime(pairs, kind='peak', key=KEY, outdir=OUTDIR,
    level=WIDTH_LEVEL, rows=None):
  '''
  The composite's two widths, for the whole sweep: W_pk of the RS alone and of the sum
  against the cooling regime. Both are log10 of a frequency RATIO, i.e. decades, so the
  vertical gap between the curves IS the broadening the slow shell adds and needs no second
  panel to carry it.

  The x axis is the RS's cooling regime -- the sweep parameter; the FS shares the rescale,
  not the regime, and sits a fixed offset above it (_regime_offset, annotated).
  A point whose crossing landed on the grid edge is drawn hollow: there the width is the
  observing window's, not the spectrum's (see shell_widths).
  '''
  os.makedirs(outdir, exist_ok=True)
  rows = shell_widths(pairs, kind=kind, level=level) if rows is None else rows
  if not rows:
    print(f'{kind} width figure: nothing to plot')
    return
  lr = np.array([r['logr'] for r in rows], float)
  fig, ax = plt.subplots(figsize=(6.4, 4.4))
  for tag, lab, mk in (('RS', 'RS', 'o'), ('tot', 'RS+FS', 's')):
    v = np.array([r[tag][f'logW_{level}'] for r in rows], float)
    ok = np.array([not (r[tag]['edge_lo'] or r[tag]['edge_hi']) for r in rows], bool)
    ax.plot(np.where(ok, lr, np.nan), np.where(ok, v, np.nan), mk + '-',
            color=STY[tag]['color'], lw=1.4, ms=5, label=lab)
    ax.plot(lr[~ok], v[~ok], marker=mk, color=STY[tag]['color'], ls='none', ms=5,
            mfc='none', alpha=.6)
  ax.set_xlabel('$\\log_{10}\\mathcal{C}$   (RS)')
  ax.set_ylabel('$W_{\\rm pk} = \\log_{10}(\\nu_{+1/2}/\\nu_{-1/2})$   [dex]')
  ax.text(.03, .97, KIND_NAME[kind].capitalize(), transform=ax.transAxes,
          ha='left', va='top', fontsize=10)
  ax.text(.03, .90, f'FS regime offset ${_regime_offset(key, verbose=False):+.2f}$ dex',
          transform=ax.transAxes, ha='left', va='top', fontsize=8, color='0.4')
  ax.grid(alpha=.25)
  ax.legend(fontsize=9, loc='lower right')
  fig.tight_layout()
  f = os.path.join(outdir, f'shell_peak_width_{kind}.png')
  fig.savefig(f, dpi=300)
  plt.close(fig)
  print(f'{kind} peak-width figure saved to {f}')
  return rows


def shell_shares(pairs, nu_ref=NU_REF):
  '''
  The whole series condensed: the FS share of the emission, three ways, per regime.
    fluence   -- bolometric, int int nuFnu dln(nu) d bar{T} (observer frame)
    peak flux -- of the two shells at the total's peak time, at nu_ref*nu_pk,RS
    E_rad     -- comoving radiated energy (return_energies budget), no observer
                 projection at all: the pure per-shell energetics
  The first two are observer-weighted and so respond to the shells' different
  arrival times and spectral positions; the third does not. Returns a dict of arrays.
  '''
  out = {k: [] for k in ('logr', 'fluence', 'peak', 'E_rad', 'eps_rs', 'eps_fs')}
  for p in pairs:
    lnnu = np.log(p['nub'])
    F = {tag: np.trapezoid(compute_fluence_spectrum(p['Tb'], nF), lnnu)
         for tag, nF in _curves(p)}
    ipk = bolometric_peak_index(p['nuFnu_tot'], p['nub'])
    ipk = 0 if ipk is None else ipk
    inu = min(np.searchsorted(p['nub'], nu_ref), len(p['nub']) - 1)
    pk = {tag: nF[ipk, inu] for tag, nF in _curves(p)}
    out['logr'].append(p['log10ratio'])
    out['fluence'].append(F['FS']/F['tot'] if F['tot'] > 0. else np.nan)
    out['peak'].append(pk['FS']/pk['tot'] if pk['tot'] > 0. else np.nan)
    Etot = p['E_rad_rs'] + p['E_rad_fs']
    out['E_rad'].append(p['E_rad_fs']/Etot if Etot > 0. else np.nan)
    out['eps_rs'].append(p['E_rad_rs']/p['E_inj_rs'] if p['E_inj_rs'] > 0. else np.nan)
    out['eps_fs'].append(p['E_rad_fs']/p['E_inj_fs'] if p['E_inj_fs'] > 0. else np.nan)
  return {k: np.array(v, float) for k, v in out.items()}


def plot_shell_shares(pairs, outdir=OUTDIR, nu_ref=NU_REF):
  '''
  Single cross-regime summary: top, what fraction of the emission the SLOW shell
  supplies (0 = RS only, 0.5 = equal, 1 = FS only) by the three measures of
  shell_shares; bottom, each shell's own radiative efficiency E_rad/E_inj, which
  says whether a share is set by how much energy a shell gets or by how well it
  radiates it.
  '''
  os.makedirs(outdir, exist_ok=True)
  s = shell_shares(pairs, nu_ref=nu_ref)
  fig, axs = plt.subplots(2, 1, figsize=(6.5, 6.5), sharex=True)
  for ke, lab, m in (('fluence', 'bolometric fluence', 'o-'),
                     ('peak', f'peak flux at $\\nu={nu_ref:g}\\,\\nu_{{\\rm pk,RS}}$', 's-'),
                     ('E_rad', "$E_{\\rm rad}$ (comoving)", '^-')):
    axs[0].plot(s['logr'], s[ke], m, lw=1.3, ms=5, label=lab)
  axs[0].axhline(0.5, color='grey', ls=':', lw=.9)
  axs[0].set_ylim(0., 1.)
  axs[0].set_ylabel('FS share,  FS/(RS+FS)')
  axs[0].set_title('How much of the emission the slow shell supplies')
  axs[0].legend(fontsize=9)
  axs[1].plot(s['logr'], s['eps_rs'], 'o-', lw=1.3, ms=5, color=STY['RS']['color'],
              label=STY['RS']['label'])
  axs[1].plot(s['logr'], s['eps_fs'], 's-', lw=1.3, ms=5, color=STY['FS']['color'],
              label=STY['FS']['label'])
  axs[1].set_yscale('log')
  axs[1].set_ylabel('$\\epsilon_{\\rm rad} = E_{\\rm rad}/E_{\\rm inj}$')
  axs[1].set_xlabel('$\\log_{10}\\mathcal{C}$   (RS)')
  axs[1].legend(fontsize=9)
  fig.tight_layout()
  fig.savefig(os.path.join(outdir, 'shell_shares.png'), dpi=300)
  plt.close(fig)
  for i, lr in enumerate(s['logr']):
    print(f'  logr={lr:+.1f}  FS share: fluence {s["fluence"][i]:.3f}  '
          f'peak {s["peak"][i]:.3f}  E_rad {s["E_rad"][i]:.3f}   '
          f'eps_rad RS {s["eps_rs"][i]:.4f} FS {s["eps_fs"][i]:.4f}')
  return s


def _check_frontdata(key, z):
  '''The FS shock-front history is not extracted by default (only run_data_4.csv
  ships with these runs); fail early with the command that makes it.'''
  path = os.path.join(get_dirpath(key), f'run_data_{z}.csv')
  if not os.path.isfile(path):
    raise FileNotFoundError(
        f'{path} missing -- extract the shell-{z} shock front first:\n'
        f"  python -c \"from analysis_hydro import extract_data_thinshell; "
        f"extract_data_thinshell('{key}', cells=[{z}], noOut=True)\"")


def main(key=KEY, log10ratio_arr=LOG10RATIO_ARR, method=METHOD, outdir=None,
    use_cache=True, nproc=None):
  '''
  Ensure both shells' sweeps exist (the RS one is normally already cached from
  sweep_rarcut -- same key, same method, same z, so it is reused bit for bit),
  then build the per-regime series and the cross-regime summary.
  method: METHOD -- 'data_rarcut', the article's prescription -- writes to OUTDIR
  (`shells_split`); any other computation gets its own '_{method}' directory so the
  two sets coexist, which puts the uncut reference in `shells_split_data`. See the
  module docstring: the naming is relative to today's METHOD, not to a fixed method,
  so a directory written before the default moved can carry a misleading name.
  '''
  outdir = (figdir(OUTDIR_NAME if method == METHOD else f'{OUTDIR_NAME}_{method}', key)
            if outdir is None else outdir)
  os.makedirs(outdir, exist_ok=True)
  for z in (Z_RS, Z_FS):
    _check_frontdata(key, z)
    d = method_outdir(method, key, z)
    os.makedirs(d, exist_ok=True)
    if not (use_cache and load_sweep(d)):
      print(f'--- running the {method} sweep on {key}, shell z={z} ---')
      # skip_cached MUST be forwarded (see sweep_gammacm.main): run_sweep defaults it
      # to True, so use_cache=False would otherwise recompute nothing.
      run_sweep(key, log10ratio_arr, z=z, method=method, nproc=nproc,
                skip_cached=use_cache)

  pairs = load_shell_pairs(key, method, log10ratio_arr)
  plot_lightcurves_per_regime(pairs, key=key, outdir=outdir)
  plot_lightcurves_norm_per_regime(pairs, key=key, outdir=outdir)
  plot_fluence_spectra_per_regime(pairs, key=key, outdir=outdir)
  for mode in ('nu_m', 'max'):
    plot_total_fluence_all(pairs, mode, outdir=outdir)
  plot_peak_spectra_per_regime(pairs, key=key, outdir=outdir)
  # the cooling SEQUENCE rather than one regime per file: three regimes side by side with
  # the sum's departure from the RS underneath, and that departure as one number -- the
  # width of the peak -- for every regime. Both kinds, peak and time-integrated.
  for kind in KIND_SYM:
    plot_shell_spectra_panels(pairs, kind=kind, key=key, outdir=outdir)
    plot_shell_width_vs_regime(pairs, kind=kind, key=key, outdir=outdir)
  s = plot_shell_shares(pairs, outdir=outdir)
  trim_pngs(outdir)
  print(f'Both-shell figures saved to {outdir}')
  return pairs, s


if __name__ == '__main__':
  # REQUIRED under the forkserver start method: a worker re-imports this module as
  # __mp_main__, and without a main guard the driver's top level would re-run inside
  # the forkserver, which then sits in a second sweep and never serves a worker -- a
  # silent stall, not an error. Harmless when driven via `python -c` (no main path),
  # which is how this file was used before; needed the moment it goes in a batch script.
  main()
