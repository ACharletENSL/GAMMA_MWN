'''
The spectral kink, on ONE cell: nu F_nu and its LOCAL SLOPE, reference vs reconstruction,
at several observer times.

The slope panel is the point. A kink is a few-hundredths-of-a-dex wiggle in nu F_nu and is
close to invisible on a log-log spectrum; in d log(nu F_nu) / d log(nu) it is unmissable, and
it can be read directly against the analytic synchrotron indices (drawn as guides).

For a one-zone slow-cooling spectrum with electron index p the slope must be:

    +4/3          below nu_m
    +(3-p)/2      nu_m .. nu_c        (+0.25 at p=2.5)
    -(p-2)/2      above nu_c          (-0.25 at p=2.5)   <- flat, to the cutoff
    steepening    above nu_max = (gmax/gmin)^2 nu_m, exponentially

so the ONLY feature allowed above the nu F_nu peak is the cutoff, and once it starts the
slope may never come back up. A dip-and-recover is not a synchrotron shape.

NB the cutoff arrives sooner than one expects here: gmax/gmin settles to ~23 after the
shock, so the cooled segment spans only 23^2 ~ 530 (0.7 dex) before cutting off.

  python -c "import prerar_kink_figure as K; K.main()"
  python -c "import prerar_kink_figure as K; K.main(logr=-3.)"
'''

import os
import numpy as np
import matplotlib.pyplot as plt

from environment import GAMMA_dir
import working_cooling_data as W
import prerar_model as M
from sweep_gammacm import compute_alpha_sweep

KEY, Z = 'cooling_g100', 4
OUTDIR = os.path.join(GAMMA_dir, 'bin', 'Tools', 'figures', 'prerar_model')
COL = {'full': '#D55E00', 'reconstructed': '#0072B2'}
SMOOTH = 9          # slope smoothing, in frequency samples (the grid is ~600 pts / 14 dex)


def _slope(nu, s, w=SMOOTH):
  '''Local d log(nuFnu)/d log(nu), lightly smoothed so the guides stay readable.'''
  ok = np.isfinite(s) & (s > 0.)
  ln, ls = np.log10(nu), np.full(nu.shape, np.nan)
  ls[ok] = np.log10(s[ok])
  g = np.gradient(ls, ln)
  if w > 1:                                   # running mean over finite entries only
    k = np.ones(w)/w
    fin = np.isfinite(g).astype(float)
    gg = np.where(np.isfinite(g), g, 0.)
    num, den = np.convolve(gg, k, 'same'), np.convolve(fin, k, 'same')
    g = np.where(den > 0, num/den, np.nan)
  return g


def main(Ri_target=1.05, logr=1., key=KEY, z=Z, outdir=OUTDIR,
    targets=(0.0, 1.33, 12., 187.7), Nnu=600):
  os.makedirs(outdir, exist_ok=True)
  cells = M._test_cells(key, z, 120)
  k = int(min(cells, key=lambda c: abs(c[4] - Ri_target))[0])
  alpha = float(compute_alpha_sweep(key, np.array([float(logr)]))[0][0])

  out = {}
  for label, law in (('full', None), ('reconstructed', 'prerar')):
    nuobs, Tobs, env, nuFnu, cd = W.get_cell_nuFnu_fromData(
        key, k, norar=law, alpha=alpha, Tmax=1000, NT=250, Nnu=Nnu,
        lognu_min=-6, lognu_max=8, return_cell=True)
    out[label] = dict(nu=nuobs/env.nu0, Tb=Tobs/env.T0 - 1.,
                      S=np.asarray(nuFnu, float), cell=cd, env=env)
  p = out['full']['env'].psyn
  idx = [int(np.argmin(np.abs(out['full']['Tb'] - t))) for t in targets]

  n = len(idx)
  fig, axes = plt.subplots(2, n, figsize=(3.6*n, 7.4), layout='constrained',
                           sharex=True)
  for j, it in enumerate(idx):
    Tb = out['full']['Tb'][it]
    ax0, ax1 = axes[0, j], axes[1, j]
    for name, d in out.items():
      s = d['S'][it]
      m = np.isfinite(s) & (s > 0.)
      ls = '--' if name == 'reconstructed' else '-'
      ax0.loglog(d['nu'][m], s[m], color=COL[name], lw=1.5, ls=ls, label=name)
      ax1.semilogx(d['nu'], _slope(d['nu'], s), color=COL[name], lw=1.5, ls=ls)
    # analytic guides: the only indices a one-zone synchrotron spectrum may show
    for y, lab in ((4./3., r'$4/3$'), ((3.-p)/2., r'$(3-p)/2$'), (-(p-2.)/2., r'$-(p-2)/2$')):
      ax1.axhline(y, color='#009E73', lw=1., ls=(0, (5, 2)), zorder=1)
      if j == 0:
        ax1.annotate(lab, xy=(0.01, y), xycoords=('axes fraction', 'data'),
                     fontsize=7, color='#009E73', va='bottom')
    smax = max(np.nanmax(d['S'][it]) for d in out.values())
    if smax > 0:
      ax0.set_ylim(smax*1e-7, smax*3.)
    ax0.set_title(rf'$\bar T$ = {Tb:.2f}', fontsize=10)
    ax1.set(xlabel=r'$\nu/\nu_m$', ylim=(-1.6, 1.8))
    for ax in (ax0, ax1):
      ax.axvline(1., color='grey', ls=':', lw=.9)
      ax.grid(alpha=.25)
    if j == 0:
      ax0.set_ylabel(r'$\nu F_\nu$  (this cell)')
      ax1.set_ylabel(r'$d\log(\nu F_\nu)/d\log\nu$')
      ax0.legend(fontsize=8, frameon=False)

  fig.suptitle(f'cell k={k}   '
               rf'$\log_{{10}}(\gamma_c/\gamma_m)$={logr:+.0f},  p={p},  '
               rf'$\gamma_{{\max}}/\gamma_{{\min}}\to$'
               f'{out["full"]["cell"].gmax.iloc[-1]/out["full"]["cell"].gmin.iloc[-1]:.0f}'
               ' (full) / '
               f'{out["reconstructed"]["cell"].gmax.iloc[-1]/out["reconstructed"]["cell"].gmin.iloc[-1]:.0f}'
               ' (rec)', fontsize=11)
  path = os.path.join(outdir, f'kink_k{k}_logr{logr:+.0f}.png')
  fig.savefig(path, dpi=150, bbox_inches='tight')
  plt.close(fig)
  print(f'saved {path}')

  # the same thing as numbers, above each spectrum's own peak
  print(f'\ncell k={k}  log10(gc/gm)={logr:+.0f}  p={p}: one zone allows only '
        f'{-(p-2.)/2.:+.2f} above the peak, then a cutoff that never recovers')
  for j, it in enumerate(idx):
    print(f'  barT={out["full"]["Tb"][it]:8.2f}')
    for name, d in out.items():
      s = d['S'][it]
      g = _slope(d['nu'], s)
      ok = np.isfinite(s) & (s > 0.)
      if not ok.any():
        continue
      jp = int(np.nanargmax(np.where(ok, s, np.nan)))
      ln = np.log10(d['nu'])
      seg = [np.nanmedian(g[(ln - ln[jp] >= c - .4) & (ln - ln[jp] < c + .4)])
             for c in (0.5, 1., 1.5, 2., 3., 4.)]
      print(f'    {name:14s} pk 10^{ln[jp]:+.2f}  ' +
            '  '.join(f'{v:+.2f}' for v in seg))
  return out, k


if __name__ == '__main__':
  main()
