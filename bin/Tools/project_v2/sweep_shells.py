# -*- coding: utf-8 -*-
# @Author: acharlet

'''
How much does each shell contribute, and how does the split move with the
cooling regime?

Every other sweep_*.py module computes the FAST shell alone -- the electrons
accelerated behind the reverse shock (Z_SHELL = 4). This one adds the SLOW shell
(forward-shocked, z = 1) and shows the two side by side with their sum, one
figure per cooling regime (3 curves per panel, hence the split by regime).

Computation: the reference data-driven pipeline (method 'data', i.e.
get_shell_nuFnu_fromData with rar_cut=None -- the rarefaction wave as the
simulation resolves it, every cell followed to its last snapshot) on the fiducial
run cooling_g100 -- the same numbers sweep_rarcut uses as its side B, so the RS
side here IS that cached sweep, reused unchanged. Pass method='data_rarcut' for
the same figures under the modelled sharp cut-off instead; they land in their own
directory.

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
  python -c "import sweep_shells as S; S.main(nproc=7)"
  python -c "import sweep_shells as S; S.main(method='data_rarcut', nproc=7)"
'''

import os
import numpy as np
import matplotlib.pyplot as plt
from functools import lru_cache

from environment import MyEnv, GAMMA_dir
from IO import get_dirpath
from peak_modeling import offset_gcgm_from_au
from plotting_functions import nF_label, COL_RS, COL_FS, COL_TOT
from sweep_compare import _regrid_onto
from sweep_gammacm import (run_sweep, load_sweep, method_outdir, compute_alpha_sweep,
    exit_onset_barT, rarefaction_off_barT, nu_over_num, detect_rise_peak_tail,
    compute_fluence_spectrum, _plot_spectra_all, trim_pngs, LOG10RATIO_ARR,
    NU_TARGETS, NU_REF, NU_M_LABEL, SPEC_YSPAN, XLIM_LIN)

KEY = 'cooling_g100'
METHOD = 'data'                   # reference: rarefaction wave taken from the simulation
Z_RS, Z_FS = 4, 1                 # reverse (fast) shell, forward (slow) shell
OUTDIR = os.path.join(GAMMA_dir, 'bin', 'Tools', 'figures', 'shells_split')

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
  return (f'$\\log_{{10}}(\\gamma_c/\\gamma_m) = {p["log10ratio"]:+.1f}$ (RS), '
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
  sweep_gammacm.plot_fluence_all, built with the same _plot_spectra_all so the
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
  norm_txt = 'normalised at $\\nu_m$' if mode == 'nu_m' else 'peak-normalised'
  _plot_spectra_all(pairs, get_total_fluence, mode,
      f'Total time-integrated spectra, both shocks ({norm_txt})',
      f'total_fluence_spectra_norm-{mode}.png', outdir, sym='\\nu \\mathcal{F}_\\nu')
  print(f'all-regime total fluence spectra (norm {mode}) saved to {outdir}')


def plot_peak_spectra_per_regime(pairs, key=KEY, nu_ref=NU_REF, outdir=OUTDIR):
  '''
  One figure per regime: the INSTANTANEOUS spectra of both shells and their sum, all
  three taken at the same observer time -- the peak of the TOTAL lightcurve at
  nu = nu_ref*nu_pk,RS (detect_rise_peak_tail on the sum). Taking each shell at its
  own peak would compare different instants and could not be added; here the three
  curves are simultaneous, so the sum is literally the sum of the two below it.
  '''
  os.makedirs(outdir, exist_ok=True)
  for p in pairs:
    _, _, _, info = detect_rise_peak_tail(p['Tb'], p['nub'], p['nuFnu_tot'], nu_ref=nu_ref)
    ipk = info.get('i_peak')
    if ipk is None:
      print(f'log10ratio={p["log10ratio"]:+.1f}: no peak found, skipped')
      continue
    barT_pk = p['Tb'][ipk] - 1.
    sp = [(tag, nF[ipk, :]) for tag, nF in _curves(p)]
    _spectrum_figure(p, sp, 'Shell contributions at the peak of the total lightcurve',
        nF_label, f'peak_spectrum_logr={p["log10ratio"]:+.1f}.png', key, outdir,
        extra=f'$\\bar{{T}}_{{\\rm pk}} = {barT_pk:.3f}$  (at $\\nu={nu_ref:g}\\,\\nu_{{\\rm pk,RS}}$)')
  print(f'{len(pairs)} per-regime peak-spectrum figures saved to {outdir}')


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
    _, _, _, info = detect_rise_peak_tail(p['Tb'], p['nub'], p['nuFnu_tot'], nu_ref=nu_ref)
    ipk = info.get('i_peak', 0)
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
  axs[1].set_xlabel('$\\log_{10}(\\gamma_c/\\gamma_m)$   (RS)')
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
  method: the reference METHOD ('data') writes to OUTDIR; any other computation
  gets its own '_{method}' directory so the two sets coexist.
  '''
  outdir = (OUTDIR if method == METHOD else f'{OUTDIR}_{method}') if outdir is None else outdir
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
  s = plot_shell_shares(pairs, outdir=outdir)
  trim_pngs(outdir)
  print(f'Both-shell figures saved to {outdir}')
  return pairs, s
