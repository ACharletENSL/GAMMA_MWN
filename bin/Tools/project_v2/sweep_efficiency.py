# -*- coding: utf-8 -*-
# @Author: acharlet

'''
Radiative efficiency eps_rad = E_rad/E_inj across the cooling regime, on a grid
FAR finer than a flux sweep can afford: log10(gamma_c/gamma_m) from -5 to +3 at
10 points per decade (81 points), for BOTH shells (reverse z=4, forward z=1) and
their combined budget.

Same lever, same simulation and same numerical settings as sweep_gammacm (the
Granot alpha hydro rescale on cooling_g100, every constant imported from there),
so the 8 coarse points already cached in figures/gammacm_sweep_data*/cache must
land exactly on these curves -- they are plotted as open circles for that check.

The only difference is that the flux is never computed: energies_only=True on
get_shell_nuFnu_fromData skips get_Fnu_cell_evolving, which is ~3.5x of a point's
cost and contributes nothing to a comoving, per-cell energy budget.

Example use in command line:
  python -c "import sweep_efficiency as S; S.main(nproc=7)"
  python -c "import sweep_efficiency as S; S.main(use_cache=False, nproc=7)"
  python -c "import sweep_efficiency as S; S.run_sweep('cooling_g100', [-5.,-1.,2.], z_list=(4,))"
'''

import os
import csv
import glob
import numpy as np
import matplotlib.pyplot as plt

from working_cooling_data import get_shell_nuFnu_fromData
from plotting_functions import COL_RS, COL_FS, COL_TOT
from sweep_gammacm import (GAMMA_dir, DEFAULT_KEY, Z_SHELL, TMAX, NT, TB_MIN, TB_LIN,
    SUBCELL_DLOGT, SUBCELL_MAX, R_REF, EARLY_ANA, compute_alpha_sweep, compute_efficiency,
    method_outdir, load_sweep, trim_pngs, _pool_context, _resolve_nproc)

LOG10RATIO_FINE = np.linspace(-5., 5., 101)  # 10 points/decade, both endpoints included
Z_LIST = (4, 1)                              # reverse shock, forward shock
METHOD = 'data'      # reference treatment: get_shell_nuFnu_fromData with rar_cut=None,
                     # i.e. the rarefaction wave as the simulation resolves it and every
                     # cell followed to its last snapshot. Fixed here (an efficiency
                     # curve is a property of the reference, not of a prescription);
                     # sweep_rarcut is what measures the modelled cut against it.
OUTDIR = os.path.join(GAMMA_dir, 'bin', 'Tools', 'figures', 'efficiency_sweep')

# Sub-cell ladder. SUBCELL_DLOGT (0.008) is what the cached flux sweep used, so it is
# the default and makes the coarse-point check EXACT (measured: 0.00e+00 relative
# deviation on the shared targets, both shells -- this path computes the very same
# cell_radiated_energy/cell_injected_energy calls the flux sweep does).
# It exists to smooth the OBSERVER-time onset staircase and splits each parent
# flux-conservingly by dx, so it barely moves a comoving budget -- confirmed by
# measure_subcell_cost() on z=4 (+372 sub-cells over 54 cells, k=466..519):
#   logr    eps(subcell)   eps(none)   rel.dev    t_sub   t_none
#   -5.00    0.998646      0.998647    4.8e-07     72 s     49 s
#   -1.00    0.816960      0.817054    1.1e-04    334 s    173 s
#   +2.00    0.059594      0.059640    7.7e-04    194 s    146 s
# i.e. dropping it is worth 0.08% at worst but only ~1.5x in time, and it breaks the
# exactness above. Kept ON. Pass subcell_dlogT=SUBCELL_FAST to run_sweep for the
# cheaper variant, but then do NOT expect check_against_flux_sweep to pass at 1e-9.
SUBCELL_FAST = None

# scalars kept per point: enough to place the point in BOTH shells' regimes and to
# rebuild every quantity the figures and the table show
_ENV_KEYS = ('gma_m', 'gma_c', 'gma_max', 'gma_mFS', 'gma_cFS', 'gma_maxFS', 'psyn',
             'eps_e', 'eps_B', 'xi_e')


def _shell_regime(env, z):
  '''log10(gamma_c/gamma_m) of shell z on env. Both shells' gamma_c scale as alpha**2
  and their gamma_m are alpha-invariant, so the FS regime is the RS one shifted by a
  constant (+0.5030 dex on cooling_g100) all along the sweep.'''
  return np.log10((env.gma_cFS/env.gma_mFS) if z == 1 else (env.gma_c/env.gma_m))


def cache_dir(outdir=OUTDIR, z=Z_SHELL):
  return os.path.join(outdir, 'cache', f'z={z}')


def _save_point(outdir, r):
  '''Cache one point. Per-point files (not one table) so an interrupted sweep
  resumes: 162 tasks at ~1 min each is a run worth not restarting.'''
  cdir = cache_dir(outdir, r['z'])
  os.makedirs(cdir, exist_ok=True)
  np.savez(os.path.join(cdir, f'point_logr={r["log10ratio"]:+.2f}.npz'),
      log10ratio=r['log10ratio'], alpha=r['alpha'], E_rad=r['E_rad'], E_int=r['E_int'],
      E_inj=r['E_inj'], key=r['key'], z=r['z'], method=r['method'],
      subcell_dlogT=(np.nan if r['subcell_dlogT'] is None else r['subcell_dlogT']),
      **{k: getattr(r['env'], k) for k in _ENV_KEYS if hasattr(r['env'], k)})


def load_efficiency_sweep(outdir=OUTDIR, z=Z_SHELL):
  '''Rebuild one shell's point list from the cache, sorted by target log10ratio.
  Returns None if the cache is empty.'''
  files = sorted(glob.glob(os.path.join(cache_dir(outdir, z), 'point_logr=*.npz')),
                 key=lambda f: float(f.split('logr=')[1].rstrip('.npz')))
  if not files:
    return None
  results = []
  for f in files:
    d = np.load(f)
    r = {k: float(d[k]) for k in ('log10ratio', 'alpha', 'E_rad', 'E_int', 'E_inj')}
    r.update({k: float(d[k]) for k in _ENV_KEYS if k in d.files})
    r['key'] = str(d['key']); r['z'] = int(d['z']); r['method'] = str(d['method'])
    r['subcell_dlogT'] = float(d['subcell_dlogT']) if 'subcell_dlogT' in d.files else np.nan
    r['regime'] = np.log10(r['gma_cFS']/r['gma_mFS']) if r['z'] == 1 \
                  else np.log10(r['gma_c']/r['gma_m'])
    results.append(r)
  return results


def _compute_point(key, z, logr, alpha, outdir, subcell_dlogT=SUBCELL_DLOGT, Tmax=TMAX):
  '''
  Compute + cache one (regime, shell) point: the comoving energy budget alone.
  Module-level so it is picklable for the process pool.

  Nnu=2 (a token frequency grid) because energies_only never touches nuobs: the
  budget comes from cell_radiated_energy / cell_injected_energy, which are per-cell
  and comoving. That also spares this sweep sweep_gammacm._nu_window, whose
  nu_M-anchored window needs an extra MyEnv + rescale_hydro per point.
  Everything that DOES set the budget is imported from sweep_gammacm, so a point
  here is directly comparable to the cached flux sweep's.
  '''
  nuobs, Tobs, env, nuFnu, E_rad, E_int, E_inj = get_shell_nuFnu_fromData(
      key, z, alpha=alpha, energies_only=True, early_ana=EARLY_ANA, rar_cut=None,
      Tmax=Tmax, NT=NT, Tb_min=TB_MIN, Tb_lin=TB_LIN, subcell_dlogT=subcell_dlogT,
      subcell_max=SUBCELL_MAX, r_ref=R_REF, Nnu=2, lognu_min=0., lognu_max=1.)
  r = dict(log10ratio=float(logr), alpha=float(alpha), env=env, E_rad=E_rad,
           E_int=E_int, E_inj=E_inj, key=key, z=z, method=METHOD,
           subcell_dlogT=subcell_dlogT)
  _save_point(outdir, r)
  return dict(logr=float(logr), z=z, alpha=float(alpha),
              regime=_shell_regime(env, z), eff=(E_rad/E_inj if E_inj > 0. else np.nan),
              E_rad=E_rad, E_inj=E_inj)


def _print_point(s):
  print(f"z={s['z']}  target={s['logr']:+.2f}  alpha={s['alpha']:11.5g}  "
        f"log10(gc/gm)={s['regime']:+.4f}  eps_rad={s['eff']:.5f}")


def cached_targets(outdir=OUTDIR, z=Z_SHELL):
  '''Targets already on disk for shell z, as the rounded keys the task list uses.'''
  res = load_efficiency_sweep(outdir, z)
  return set() if res is None else {round(r['log10ratio'], 4) for r in res}


def run_sweep(key=DEFAULT_KEY, log10ratio_arr=LOG10RATIO_FINE, z_list=Z_LIST,
    outdir=OUTDIR, nproc=None, subcell_dlogT=SUBCELL_DLOGT, Tmax=TMAX,
    skip_cached=True):
  '''
  Energy budget of every (target log10(gma_c/gma_m), shell) pair, via the alpha lever
  (zeta=1, u_scale=1) exactly as sweep_gammacm.run_sweep drives it -- the alphas come
  from the same compute_alpha_sweep, so the two sweeps' shared targets are the same
  physical points.

  The tasks are independent and run on one process pool (_pool_context: forkserver,
  see the note in sweep_gammacm -- fork deadlocks here on threaded BLAS, silently).
  ONE serial warm-up point PER SHELL first: those populate the (key, z) cell and
  shock-front disk caches, so pool workers only ever read shared paths.
  skip_cached: leave points already in outdir/cache alone, so an interrupted sweep
  resumes where it stopped (delete the point files, or pass False, to recompute).
  Returns {z: results list} rebuilt from the cache.
  '''
  alpha_arr, log10ratio0 = compute_alpha_sweep(key, log10ratio_arr)
  print(f'baseline (alpha=1): log10(gma_c/gma_m) = {log10ratio0:.6f}')
  args = lambda z, logr, alpha: (key, z, float(logr), float(alpha), outdir,
                                 subcell_dlogT, Tmax)
  tasks = []
  for z in z_list:
    have = cached_targets(outdir, z) if skip_cached else set()
    tasks += [args(z, logr, alpha) for logr, alpha in zip(log10ratio_arr, alpha_arr)
              if round(float(logr), 4) not in have]
  n_all = len(log10ratio_arr)*len(z_list)
  print(f'{len(log10ratio_arr)} targets x {len(z_list)} shells = {n_all} points, '
        f'{n_all - len(tasks)} already cached, {len(tasks)} to compute '
        f'(subcell_dlogT={subcell_dlogT})')
  npr = _resolve_nproc(nproc, max(len(tasks), 1))
  if not tasks:
    pass
  elif npr == 1:
    for t in tasks:
      _print_point(_compute_point(*t))
  else:
    # warm up once per shell present in the task list, so pool workers only READ the
    # (key, z) cell / shock-front / rarefaction disk caches
    warm = [next(t for t in tasks if t[1] == z) for z in z_list
            if any(t[1] == z for t in tasks)]
    print(f'sweep on {npr} workers ({len(warm)} serial warm-up points, then pool)')
    for t in warm:
      _print_point(_compute_point(*t))
    import concurrent.futures as cf
    with cf.ProcessPoolExecutor(max_workers=npr, mp_context=_pool_context()) as ex:
      futs = [ex.submit(_compute_point, *t) for t in tasks if t not in warm]
      for fut in cf.as_completed(futs):
        _print_point(fut.result())
  return {z: load_efficiency_sweep(outdir, z) for z in z_list}


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------

def _arrays(results):
  '''(target logr, own-shell regime, eps_rad) of one shell, as arrays.'''
  logr = np.array([r['log10ratio'] for r in results], float)
  reg = np.array([r['regime'] for r in results], float)
  eff = np.array([compute_efficiency(r) for r in results], float)
  return logr, reg, eff


def combined_efficiency(res_rs, res_fs):
  '''
  Combined budget of the two shells at each sweep target,
  (E_rad,RS + E_rad,FS)/(E_inj,RS + E_inj,FS) -- the same construction as
  sweep_shells.shell_shares. Energies are comoving and additive, so this needs no
  flux-unit conversion between the shells (unlike their fluxes, which do).
  Matched on the TARGET log10ratio, i.e. on alpha: at a given alpha the two shells
  sit at different regimes (the FS is offset by a constant +0.5 dex), so this is the
  efficiency of the whole double-shell system at one hydro rescaling, plotted below
  against the RS regime that labels the sweep.
  '''
  fs = {round(r['log10ratio'], 4): r for r in res_fs}
  logr, Erad, Einj = [], [], []
  for r in res_rs:
    o = fs.get(round(r['log10ratio'], 4))
    if o is None:
      continue
    logr.append(r['log10ratio'])
    Erad.append(r['E_rad'] + o['E_rad'])
    Einj.append(r['E_inj'] + o['E_inj'])
  Erad, Einj = np.array(Erad), np.array(Einj)
  return np.array(logr), np.where(Einj > 0., Erad/np.where(Einj > 0., Einj, 1.), np.nan)


def plot_efficiency_curve(res_rs, res_fs, outdir=OUTDIR):
  '''
  eps_rad = E_rad/E_inj vs the GLOBAL cooling regime, both shells and their combined
  budget on one x axis. That axis is the sweep's control parameter -- the single
  log10(gamma_c/gamma_m) the alpha rescaling sets for the system as a whole -- so all
  three curves are read against the same number. Each shell's OWN regime sits at a rigid
  offset from it (the FS by +0.503 dex, alpha-invariant since both gamma_c scale as
  alpha**2); that per-shell view is plot_efficiency_own_regime's job, and is deliberately
  NOT duplicated here as a secondary axis.
  Log y: eps_rad runs from ~1 to ~1e-3 over the range and the slow-cooling wing (a clean
  power law) is only legible on a log axis; a linear panel shows nothing the eye cannot
  get from this one.
  No markers -- at 10 points per decade the samples are denser than a readable symbol
  spacing, so the curves are drawn as curves. The agreement with the cached FLUX sweep is
  checked NUMERICALLY instead, by check_against_flux_sweep (it must be exact, both paths
  making the same cell_radiated_energy/cell_injected_energy calls).
  '''
  logr_rs, _, eff_rs = _arrays(res_rs)
  logr_fs, _, eff_fs = _arrays(res_fs)
  logr_c, eff_c = combined_efficiency(res_rs, res_fs)

  fig, ax = plt.subplots(figsize=(7., 4.8))
  for x, y, c, lab in ((logr_rs, eff_rs, COL_RS, 'RS ($z=4$)'),
                       (logr_fs, eff_fs, COL_FS, 'FS ($z=1$)'),
                       (logr_c, eff_c, COL_TOT, 'RS + FS')):
    ax.plot(x, y, '-', color=c, lw=1.4, label=lab)
  ax.axhline(1., color='grey', ls=':', lw=.9)
  ax.set_yscale('log')
  ax.grid(alpha=.25, lw=.5)
  ax.set_ylabel('$\\varepsilon_{\\rm rad}=E_{\\rm rad}/E_{\\rm inj}$')
  ax.set_xlabel('$\\log_{10}(\\gamma_c/\\gamma_m)$')
  ax.legend(fontsize=9)
  ax.set_title('Radiative efficiency across the cooling regime')
  fig.tight_layout()
  fig.savefig(os.path.join(outdir, 'radiative_efficiency_fine.png'), dpi=300)
  plt.close(fig)


def plot_efficiency_own_regime(res_rs, res_fs, outdir=OUTDIR):
  '''
  The same two shells, each against ITS OWN log10(gamma_c/gamma_m) rather than the
  sweep's RS control axis. If eps_rad is a function of the local cooling regime alone
  the two curves collapse; whatever separation is left is the shells' differing hydro
  (density and Lorentz-factor histories, hence adiabatic losses) at the same regime.
  The lower panel is that separation, eps_FS/eps_RS interpolated onto a common regime.
  '''
  _, reg_rs, eff_rs = _arrays(res_rs)
  _, reg_fs, eff_fs = _arrays(res_fs)
  lo, hi = max(reg_rs.min(), reg_fs.min()), min(reg_rs.max(), reg_fs.max())
  xc = np.linspace(lo, hi, 200)
  ratio = np.interp(xc, reg_fs, eff_fs)/np.interp(xc, reg_rs, eff_rs)

  fig, axs = plt.subplots(2, 1, figsize=(7., 7.), sharex=True,
                          gridspec_kw={'height_ratios': [2.2, 1]})
  axs[0].plot(reg_rs, eff_rs, '-', color=COL_RS, lw=1.4, label='RS ($z=4$)')
  axs[0].plot(reg_fs, eff_fs, '-', color=COL_FS, lw=1.4, label='FS ($z=1$)')
  axs[0].set_yscale('log')
  axs[0].axhline(1., color='grey', ls=':', lw=.9)
  axs[0].set_ylabel('$\\varepsilon_{\\rm rad}$')
  axs[0].legend(fontsize=9)
  axs[0].set_title('Is $\\varepsilon_{\\rm rad}$ a function of the local regime alone?')
  axs[1].plot(xc, ratio, '-', color='k', lw=1.2)
  axs[1].axhline(1., color='grey', ls=':', lw=.9)
  axs[1].set_ylabel('FS / RS')
  axs[1].set_xlabel("$\\log_{10}(\\gamma_c/\\gamma_m)$   (each shell's own)")
  for ax in axs:
    ax.grid(alpha=.25, lw=.5)
  fig.tight_layout()
  fig.savefig(os.path.join(outdir, 'radiative_efficiency_own_regime.png'), dpi=300)
  plt.close(fig)


def build_efficiency_table(res_by_z, outdir=OUTDIR, fname='efficiency_table.csv'):
  '''
  One row per (shell, sweep point): the alpha, the shell's own regime, the three
  energies, xi_E = E_inj/(eps_e*E_int) (the fraction of eps_e*e'_int a distribution
  truncated at gma_M can hold -- the old, too-large denominator) and eps_rad.
  '''
  path = os.path.join(outdir, fname)
  with open(path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['z', 'log10ratio_target', 'alpha', 'log10_gc_over_gm', 'E_rad', 'E_inj',
                'E_int', 'xi_E', 'eps_rad'])
    for z, results in res_by_z.items():
      for r in results:
        ei = r['eps_e']*r['E_int']
        w.writerow([z, f"{r['log10ratio']:+.2f}", f"{r['alpha']:.6e}",
                    f"{r['regime']:+.6f}", f"{r['E_rad']:.6e}", f"{r['E_inj']:.6e}",
                    f"{r['E_int']:.6e}", f"{(r['E_inj']/ei if ei > 0 else np.nan):.6f}",
                    f"{compute_efficiency(r):.6f}"])
  print(f'efficiency table written to {path}')
  return path


def check_against_flux_sweep(res_by_z, key=DEFAULT_KEY, rtol=1e-9):
  '''
  The energies here come from the same cell_radiated_energy / cell_injected_energy
  calls the flux sweep makes, so wherever the two grids share a target they must agree
  to round-off. Prints the comparison and returns the worst relative deviation (nan if
  the flux sweep has not been run).
  '''
  devs = []
  for z, results in res_by_z.items():
    coarse = load_sweep(method_outdir(METHOD, key, z))
    if not coarse or not results:
      print(f'z={z}: no cached flux sweep to check against')
      continue
    fine = {round(r['log10ratio'], 4): r for r in results}
    print(f"\nz={z}  {'logr':>7} {'eps_rad (fine)':>15} {'eps_rad (flux)':>15} {'rel. dev.':>11}")
    for c in coarse:
      f = fine.get(round(c['log10ratio'], 4))
      if f is None:
        continue
      ef, ec = compute_efficiency(f), compute_efficiency(c)
      dev = abs(ef - ec)/ec if ec > 0 else np.nan
      devs.append(dev)
      flag = '' if (np.isfinite(dev) and dev <= rtol) else '   <-- MISMATCH'
      print(f"  {c['log10ratio']:+7.2f} {ef:15.8f} {ec:15.8f} {dev:11.2e}{flag}")
  worst = max(devs) if devs else np.nan
  if np.isfinite(worst):
    print(f'\nworst relative deviation vs the flux sweep: {worst:.2e} (tol {rtol:.0e})')
  return worst


def measure_subcell_cost(key=DEFAULT_KEY, z=Z_SHELL, logr_list=(-5., -1., 2.),
    outdir=None):
  '''
  What turning the sub-cell ladder off costs the BUDGET (and saves in time). The
  ladder splits parents flux-conservingly by dx to smooth the observer-time onset
  staircase, so it should barely move a comoving energy sum -- this measures it
  rather than assuming it, and is the gate on running the production sweep with
  SUBCELL_FAST. Writes nothing to the production cache.
  '''
  import time, tempfile
  outdir = tempfile.mkdtemp(prefix='effsweep_subcell_') if outdir is None else outdir
  alphas, _ = compute_alpha_sweep(key, logr_list)
  print(f"{'logr':>7} {'eps(subcell)':>13} {'eps(none)':>13} {'rel.dev':>9} "
        f"{'t_sub':>7} {'t_none':>7}")
  worst = 0.
  for logr, alpha in zip(logr_list, alphas):
    out = []
    for sc in (SUBCELL_DLOGT, SUBCELL_FAST):
      t0 = time.time()
      s = _compute_point(key, z, logr, alpha, outdir, subcell_dlogT=sc)
      out.append((s['eff'], time.time() - t0))
    (e1, t1), (e2, t2) = out
    dev = abs(e2 - e1)/e1
    worst = max(worst, dev)
    print(f'{logr:+7.2f} {e1:13.6f} {e2:13.6f} {dev:9.2e} {t1:7.1f} {t2:7.1f}')
  print(f'\nworst relative deviation from dropping the sub-cells: {worst:.2e}')
  print(f'(scratch cache in {outdir})')
  return worst


def main(key=DEFAULT_KEY, log10ratio_arr=LOG10RATIO_FINE, z_list=Z_LIST, outdir=OUTDIR,
    use_cache=True, nproc=None, subcell_dlogT=SUBCELL_DLOGT, check=True):
  '''
  Full efficiency sweep + figures + table. use_cache reuses whatever points are
  already in outdir/cache (the run is resumable: re-run with use_cache=False to
  recompute, or just delete the points you want redone).
  '''
  os.makedirs(outdir, exist_ok=True)
  # run_sweep is itself incremental: it computes only the (z, target) points missing
  # from the cache, so this is both the first run and the resume path
  res_by_z = run_sweep(key, log10ratio_arr, z_list=z_list, outdir=outdir, nproc=nproc,
                       subcell_dlogT=subcell_dlogT, skip_cached=use_cache)
  if check:
    check_against_flux_sweep(res_by_z, key=key)
  if 4 in res_by_z and 1 in res_by_z:
    plot_efficiency_curve(res_by_z[4], res_by_z[1], outdir=outdir)
    plot_efficiency_own_regime(res_by_z[4], res_by_z[1], outdir=outdir)
  build_efficiency_table(res_by_z, outdir=outdir)
  trim_pngs(outdir)
  print(f'Figures saved to {outdir}')
  return res_by_z


if __name__ == '__main__':
  main()
