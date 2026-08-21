'''
The ENVELOPE of the sub-steps: the line joining the PEAKS of the individual sub-step spectra,
fast vs slow cooling, with its local slope.

The composite spectrum above nu_c is built out of this locus, so its shape is what sets the
tail: where the envelope is a clean power law the composite is too, and where it bends the
composite bends with it. This plots it explicitly rather than summarising it with one number
-- a single quadratic fit over the whole envelope gave +0.0006 in slow cooling against -0.0556
in fast, which is a misleading statistic because the variation is not quadratic and is
concentrated at one end.

  row 1  every sub-step (faint), the peak locus (markers + line), the composite (black)
  row 2  d log A / d log nu_peak along the envelope, i.e. how the peak flux scales with where
         that sub-step peaks

Ordering matters when reading it: EARLY sub-steps have the highest gamma_max, so they sit at
HIGH nu_peak; late ones at low nu_peak. Colour runs with step index.

  python -c "import substep_envelope as E; E.main()"
'''

import os
import numpy as np
import matplotlib.pyplot as plt

from environment import GAMMA_dir, MyEnv
from working_cooling import (generate_cell_withDistrib, precompute_step_cols, step_view,
                             norm_plaw_distrib, _midpoint_cell)
from radiation_cooling import get_Fnu_step
from IO import open_celldata, open_rundata
from fits_hydro import cellsBehindShock_fromData
from sweep_gammacm import compute_alpha_sweep
import prerar_model as M

KEY, Z = 'cooling_g100', 4
OUTDIR = os.path.join(GAMMA_dir, 'bin', 'Tools', 'figures')


def main(Ri_target=1.05, logrs=(-4., 3.), key=KEY, z=Z, outdir=OUTDIR, Nnu=2400):
  cells = M._test_cells(key, z, 120)
  k = int(min(cells, key=lambda c: abs(c[4] - Ri_target))[0])
  raw = open_celldata(key, k)
  shr = cellsBehindShock_fromData(open_rundata(key, z))
  d0 = shr.loc[shr.i == raw.iloc[0].i].iloc[0]
  ex = shr.loc[shr.t.idxmax()]
  nu = np.logspace(-8, 14, Nnu)
  ln = np.log10(nu)

  fig, axes = plt.subplots(2, len(logrs), figsize=(7.6*len(logrs), 9.4),
                           layout='constrained')
  axes = np.atleast_2d(axes)
  for c, logr in enumerate(logrs):
    alpha = float(compute_alpha_sweep(key, np.array([float(logr)]))[0][0])
    cell, env = generate_cell_withDistrib(raw, d0, MyEnv(key), alpha=alpha, r_ref=1.2,
                    Tmax=None, key=key, k=k, exit_row=ex)
    p = env.psyn
    c0 = cell.iloc[0]
    K0 = norm_plaw_distrib(c0.gmin, c0.gmax, p)
    cm = _midpoint_cell(cell)
    cols = precompute_step_cols(cm, env)
    N = len(cell)
    Ton, Tth, Tej = cols['obsT']
    T = float(Ton.max())
    S = np.zeros((N, Nnu))
    for j in range(N):
      s = np.asarray(get_Fnu_step(nu*env.nu0, T, step_view(cols, j), K0, env), float)*nu
      S[j] = np.where(np.isfinite(s) & (s > 0), s, 0.)
    tot = S.sum(axis=0)
    live = np.flatnonzero(S.max(axis=1) > tot.max()*1e-12)
    A = np.array([S[j].max() for j in live])
    npk = np.array([nu[int(np.argmax(S[j]))] for j in live])

    cmap = plt.cm.turbo(np.linspace(0, 1, len(live)))
    for col, j in zip(cmap, live):
      m = S[j] > tot.max()*1e-9
      if m.sum() > 2:
        axes[0, c].loglog(nu[m], S[j][m]/tot.max(), color=col, lw=.6, alpha=.5)
    axes[0, c].loglog(npk, A/tot.max(), 'o-', color='k', ms=3.2, lw=1.4,
                      label='envelope (locus of sub-step peaks)')
    m = tot > 0
    axes[0, c].loglog(nu[m], tot[m]/tot.max(), color='crimson', lw=2., label='composite')
    nu_c = 10.**(2.*logr)
    for ax in axes[:, c]:
      ax.axvline(nu_c, color='grey', ls='--', lw=1.1)
    axes[0, c].set(title=rf'$\log_{{10}}(\gamma_c/\gamma_m)$ = {logr:+.0f}'
                         rf'   ({len(live)} sub-steps)',
                   ylabel=r'$\nu F_\nu$ / composite peak', ylim=(1e-7, 3.))
    axes[0, c].legend(fontsize=8, loc='lower center')

    o = np.argsort(npk)
    x, y = np.log10(npk[o]), np.log10(A[o])
    sl = np.gradient(y, x)
    axes[1, c].semilogx(10**x, sl, 'o-', color='k', ms=3.2, lw=1.2)
    axes[1, c].axhline(0., color='grey', lw=.9, ls=':')
    axes[1, c].set(xlabel=r'$\nu_{\rm peak}$ of the sub-step  $/\nu_m$',
                   ylabel=r'$d\log A/d\log\nu_{\rm peak}$ along the envelope',
                   ylim=(-1.6, 1.2))
    for ax in axes[:, c]:
      ax.grid(alpha=.25)
    n = len(x); q = max(1, n//4)
    print(f'logr={logr:+.0f}: envelope over {n} sub-steps, '
          f'nu_peak {10**x.min():.3g}..{10**x.max():.3g} nu_m  (nu_c={nu_c:.3g})')
    print(f'   slope  LATE end (low nu_pk) {np.median(sl[:q]):+.3f}'
          f'   middle {np.median(sl[q:3*q]):+.3f}'
          f'   EARLY end (high nu_pk) {np.median(sl[3*q:]):+.3f}')
    print(f'   slope range along the envelope: {sl.min():+.3f} .. {sl.max():+.3f}'
          f'   (a straight envelope would be constant)')
  fig.suptitle(f'Envelope = locus of the sub-step PEAKS, cell k={k}  '
               r'(grey dashed = $\nu_c$)', fontsize=12)
  path = os.path.join(outdir, 'substep_envelope.png')
  fig.savefig(path, dpi=140, bbox_inches='tight')
  plt.close(fig)
  print(f'saved {path}')


if __name__ == '__main__':
  main()
