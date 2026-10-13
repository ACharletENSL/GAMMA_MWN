'''
Band (1993) low-energy index alpha fitted to the synthetic spectra over the window
[nu_pk/30, 30 nu_pk] around the nuFnu peak, i.e. the way a catalogued GBM spectrum is
fitted (PB's comment on Sect. 5.2: report the Band alpha as a function of C rather than
an index measured decades below the peak).

Cache-only: the two shell sweeps (sweep_shells.load_shell_pairs, RS and the RS+FS sum on
the RS flux scale), with (data) and without (data_rarcut) the post-rarefaction emission.
Both the time-integrated spectrum and the spectrum at the bolometric peak are fitted.

The fit is in log10(nuFnu) with uniform weight per sample; the obs grid is uniform in
log nu, so that is uniform weight per decade (no detector, no Poisson weighting -- this is
the shape a Band function gives the noiseless spectrum, not what a GBM fit would return).

  PYTHONPATH=. python3 band_alpha.py               # fiducial cooling_g100
  BAND_KEY=cooling_g100_hires PYTHONPATH=. python3 band_alpha.py
  BAND_WINDOWS=1 PYTHONPATH=. python3 band_alpha.py  # lower edge nu_pk/30..1000 compared
  BAND_SHELLS=1 PYTHONPATH=. python3 band_alpha.py   # RS vs RS+FS, rarcut, into shells_split
  BAND_2SBPL=1 PYTHONPATH=. python3 band_alpha.py    # same with Ravasio+2018's 2SBPL
'''
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

import sweep_gammacm as swp
import sweep_shells as ssh
from environment import figdir
from plotting_functions import COL_RS, COL_TOT
from phys_functions import two_sbpl, RAVASIO_N1, RAVASIO_N2

WIN = 30.                 # the window is [nu_pk/WIN, WIN*nu_pk]
WIN_LO_CMP = (30., 100., 300., 1000.)   # lower edges compared by main_windows (upper fixed)
OUTDIR_NAME = 'band_alpha'
METHODS = {'full': 'data', 'cut': 'data_rarcut'}
LS = {'full': '-', 'cut': '--'}
LAB = {'full': 'full', 'cut': 'rf cut'}
SHELLS = {'rs': ('nuFnu_rs', COL_RS, 'RS'), 'tot': ('nuFnu_tot', COL_TOT, 'RS+FS')}


def band_nufnu(lx, logA, alpha, beta, lxp):
  '''log10 nuFnu of a Band function, x in any unit; peak of nuFnu at 10**lxp'''
  x, xp = 10.**lx, 10.**lxp
  x0 = xp/(2. + alpha)                       # e-folding energy
  xb = (alpha - beta)*x0                     # break
  lo = alpha*np.log10(x/xp) - (x - xp)/x0/np.log(10.)
  hi = (beta*np.log10(x/xp) + (alpha - beta)*np.log10(xb/xp)
        - (xb - xp)/x0/np.log(10.))
  return logA + 2.*np.log10(x/xp) + np.where(x < xb, lo, hi)


def fit_band(x, sp, win=WIN, win_lo=None):
  '''
  Band fit of one nuFnu spectrum over [x_pk/win_lo, win*x_pk], x_pk its own maximum
  (win_lo defaults to win).
  Returns alpha, beta, the fitted peak over the data peak, rms (dex) and the
  number of samples.
  '''
  g = (sp > 0) & np.isfinite(sp)
  x, sp = x[g], sp[g]
  ipk = int(np.argmax(sp))
  lxp = np.log10(x[ipk])
  dlo, dhi = np.log10(win_lo or win), np.log10(win)
  m = (np.log10(x) >= lxp - dlo) & (np.log10(x) <= lxp + dhi)
  lx, ly = np.log10(x[m]), np.log10(sp[m])
  # the window must be whole: a truncated one would quietly fit a narrower band
  whole = lx.min() <= lxp - dlo + 0.05 and lx.max() >= lxp + dhi - 0.05
  p0 = (ly.max(), -1., -2.5, lxp)
  bnds = ([ly.max() - 3, -1.99, -10., lxp - 1.5], [ly.max() + 3, 3., -2.01, lxp + 1.5])
  best = None
  for a0 in (-1.5, -1., -0.5):               # a few starts: the alpha-beta-Ep valley is long
    r = least_squares(lambda p: band_nufnu(lx, *p) - ly, (p0[0], a0, p0[2], p0[3]),
                      bounds=bnds, x_scale=(1., .3, .3, .3))
    if best is None or r.cost < best.cost:
      best = r
  res = best.fun
  return dict(alpha=best.x[1], beta=best.x[2], xp_ratio=10.**(best.x[3] - lxp),
              rms=float(np.sqrt(np.mean(res**2))), n=int(m.sum()), whole=bool(whole),
              lxp=lxp)


BRK_EDGE = 0.1     # dex: a 2SBPL break this close to the window bottom or to the peak is at bound


def fit_2sbpl(x, sp, win=WIN, win_lo=None, n1=RAVASIO_N1, n2=RAVASIO_N2):
  '''
  Ravasio et al. (2018) 2SBPL fit of one nuFnu spectrum over the same window as fit_band:
  A, alpha1, E_break, alpha2, E_peak, beta free, n1 and n2 HELD at their values (their
  Sect. 2.4). Residual in log10 nuFnu, uniform per sample. The break is confined to the
  window, below the peak; at_bound flags a break within BRK_EDGE of either end, where
  alpha1 (bottom) or alpha2 (top) rests on too little of the window to mean much.

  Fitted as alpha1 = alpha2 + delta, delta >= 0. The lower SBPL of eq. (3),
  [(E/E_br)**(-a1 n1) + (E/E_br)**(-a2 n1)]**(-1/n1), is SYMMETRIC in a1 <-> a2 (the
  asymmetric prefactor E_br**a1 is absorbed by A): the HARDER index always governs below
  the break, whichever is called alpha1. Left free, the fit returned both labellings at
  identical rms (0.00269 dex, log C = -3, nu_pk/1000), with alpha1 and alpha2 swapped and
  the break moved. delta >= 0 is exactly the set of shapes the model can make.
  '''
  g = (sp > 0) & np.isfinite(sp)
  x, sp = x[g], sp[g]
  lxp = np.log10(x[int(np.argmax(sp))])
  dlo, dhi = np.log10(win_lo or win), np.log10(win)
  m = (np.log10(x) >= lxp - dlo) & (np.log10(x) <= lxp + dhi)
  lx, ly = np.log10(x[m]), np.log10(sp[m])
  E = 10.**lx

  def model(q):           # q = logA, delta = a1 - a2, d = log(E_pk/E_br), a2, logE_pk, beta
    return np.log10(np.maximum(two_sbpl(E, 10.**(q[4] - q[2]), 10.**q[4], q[3] + q[1], q[3],
                                        q[5],
                                        n1=n1, n2=n2, A=10.**q[0], nuFnu=True), 1e-300))

  ymax = ly.max()
  lo = [ymax - 40., 0., 0., -1.99, lxp - .5, -8.]
  hi = [ymax + 40., 3., dlo, -0.2, lxp + .5, -2.01]
  best = None
  for d0 in np.linspace(0.15, dlo - 0.15, 4):      # the break position is the long valley
    for dl0 in (0.2, 0.8):
      q0 = [ymax, dl0, d0, -1.7, lxp, -2.3]
      # A is set by E_break**a1: re-centre it on the data at the start so the first steps
      # are not spent on the normalisation
      q0[0] += ymax - model(q0).max()
      q0 = np.clip(q0, np.array(lo) + 1e-6, np.array(hi) - 1e-6)
      try:
        r = least_squares(lambda q: model(q) - ly, q0, bounds=(lo, hi),
                          x_scale=(1., .3, .3, .3, .3, .3))
      except ValueError:
        continue
      if best is None or r.cost < best.cost:
        best = r
  q = best.x
  return dict(a1=q[3] + q[1], a2=q[3], beta=q[5], brk_dex=q[2], xp_ratio=10.**(q[4] - lxp),
              rms=float(np.sqrt(np.mean(best.fun**2))), n=int(m.sum()),
              at_bound=bool(q[2] < BRK_EDGE or q[2] > dlo - BRK_EDGE), lxp=lxp)


def measure(key=ssh.KEY, win=WIN, win_los=None):
  '''One row per (C, method, shell, epoch, lower edge): the Band fit of that spectrum.'''
  win_los = win_los or (win,)
  rows = []
  for mk, meth in METHODS.items():
    for p in ssh.load_shell_pairs(key=key, method=meth):
      x, psyn = p['x'], float(p['env'].psyn)
      ok = x >= swp.phys_x_floor(dict(env=p['env'], nub=p['nub']))
      for sk, (col, _, _) in SHELLS.items():
        nf = p[col]
        ipk = swp.bolometric_peak_index(nf, p['nub'])
        for ep, sp in (('integrated', swp.compute_fluence_spectrum(p['Tb'], nf)),
                       ('peak', nf[ipk])):
          for wl in win_los:
            f = fit_band(x[ok], sp[ok], win, wl)
            rows.append(dict(logC=round(float(p['log10ratio']), 2), method=mk, shell=sk,
                             epoch=ep, win_lo=wl, p=psyn, **f))
  return pd.DataFrame(rows)


LEG = dict(fontsize=8, frameon=True, facecolor='w', edgecolor='none', framealpha=1.)


def _legend(ax, **kw):
  '''opaque white legend, drawn over the guide lines that run through it'''
  ax.legend(**LEG, **kw).set_zorder(5)


def _guides(ax, p):
  '''one-zone photon indices: -2/3 (below nu_c, nu_m), -3/2 (fast cooling), -(p+1)/2 (slow
  cooling above nu_m). Labels sit off the curves: -2/3 under its line, the SC one at the right.'''
  tr = ax.get_yaxis_transform()
  for a, lab, xt, dy, ha, va in ((-2/3, r'$-2/3$', 0.02, -0.02, 'left', 'top'),
                                 (-3/2, r'$-3/2$', 0.02, 0.02, 'left', 'bottom'),
                                 (-(p + 1)/2, rf'$-(p+1)/2$', 0.98, 0.02, 'right', 'bottom')):
    ax.axhline(a, c='0.5', ls=':', lw=0.9, zorder=0)
    ax.text(xt, a + dy, lab, color='0.4', fontsize=8, ha=ha, va=va, transform=tr)


def plot(df, outdir, fname='band_alpha.png', win=WIN):
  fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True, layout='constrained')
  for ax, ep, title in zip(axs, ('integrated', 'peak'), ('time-integrated', 'peak')):
    _guides(ax, df.p.iloc[0])
    for sk, (_, c, slab) in SHELLS.items():
      # at the peak the cut has removed nothing yet (it only drops late emission): one line
      for mk in (METHODS if ep == 'integrated' else ('full',)):
        d = df[(df.epoch == ep) & (df.shell == sk) & (df.method == mk)].sort_values('logC')
        ax.plot(d.logC, d.alpha, c=c, ls=LS[mk], lw=1.2, marker='o', ms=3.5,
                mfc=c if mk == 'full' else 'w', label=f'{slab}, {LAB[mk]}' if ep == 'integrated' else
                f'{slab} (full = rf cut)')
        bad = d[~d.whole]
        ax.plot(bad.logC, bad.alpha, ls='', marker='x', c='0.3', ms=6)
    ax.set_xlabel(r'$\log_{10}\mathcal{C}$')
    ax.text(0.03, 0.80, title, transform=ax.transAxes, fontsize=10)
    _legend(ax, loc='upper right')
  axs[0].set_ylabel(rf'Band $\alpha$  ($\nu_{{\rm pk}}/{win:g}$ to ${win:g}\,\nu_{{\rm pk}}$)')
  os.makedirs(outdir, exist_ok=True)
  path = os.path.join(outdir, fname)
  fig.savefig(path, dpi=300)
  plt.close(fig)
  return path


def plot_windows(df, outdir, fname='band_alpha_windows.png', win=WIN):
  '''
  alpha against C for each lower edge of the window (colour), upper edge fixed at win.
  Rows: RS, RS+FS; columns: time-integrated, peak. Solid full, dashed rf cut
  (integrated only -- at the peak the two are identical).
  '''
  wls = sorted(df.win_lo.unique())
  cols = swp.SWEEP_CMAP(np.linspace(0., 1., len(wls)))
  fig, axs = plt.subplots(2, 2, figsize=(9.6, 6.6), sharex=True, sharey=True,
                          layout='constrained')
  for j, (ep, etitle) in enumerate((('integrated', 'time-integrated'), ('peak', 'peak'))):
    for i, (sk, (_, _, slab)) in enumerate(SHELLS.items()):
      ax = axs[i, j]
      _guides(ax, df.p.iloc[0])
      for wl, c in zip(wls, cols):
        for mk in (METHODS if ep == 'integrated' else ('full',)):
          d = df[(df.epoch == ep) & (df.shell == sk) & (df.method == mk)
                 & (df.win_lo == wl)].sort_values('logC')
          ax.plot(d.logC, d.alpha, c=c, ls=LS[mk], lw=1.2, marker='o', ms=3.5,
                  mfc=c if mk == 'full' else 'w',
                  label=rf'$\nu_{{\rm pk}}/{wl:g}$' if mk == 'full' else None)
          bad = d[~d.whole]
          ax.plot(bad.logC, bad.alpha, ls='', marker='x', c='0.3', ms=6)
      ax.text(0.03, 0.80, f'{slab}, {etitle}', transform=ax.transAxes, fontsize=10)
      if i == 1:
        ax.set_xlabel(r'$\log_{10}\mathcal{C}$')
      if j == 0:
        ax.set_ylabel(rf'Band $\alpha$  (to ${win:g}\,\nu_{{\rm pk}}$)')
  _legend(axs[0, 0], loc='upper right', title='lower edge', title_fontsize=8)
  from matplotlib.lines import Line2D
  _legend(axs[0, 1], handles=[Line2D([], [], c='k', ls=LS[mk], label=LAB[mk]) for mk in METHODS],
          loc='upper right')
  os.makedirs(outdir, exist_ok=True)
  path = os.path.join(outdir, fname)
  fig.savefig(path, dpi=300)
  plt.close(fig)
  return path


def measure_shells(key=ssh.KEY, method='data_rarcut', win=WIN, win_los=WIN_LO_CMP):
  '''
  Band fits of the RS and of the RS+FS sum, one method, every lower edge. The spectra are
  sweep_shells.shell_spectra's, so at the peak both are taken at the SAME instant (the
  bolometric peak of the total), as in sweep_shells.plot_shell_slopes -- not each at its
  own peak, as measure() does.
  '''
  rows = []
  for p in ssh.load_shell_pairs(key=key, method=method):
    x = p['x']
    ok = x >= 1./p['env'].gma_m**2
    for kind in ('fluence', 'peak'):
      sp, _ = ssh.shell_spectra(p, kind)
      for tag in ('RS', 'tot'):
        for wl in win_los:
          f = fit_band(x[ok], sp[tag][ok], win, wl)
          rows.append(dict(logC=round(float(p['log10ratio']), 2), kind=kind, shell=tag,
                           win_lo=wl, p=float(p['env'].psyn), **f))
  return pd.DataFrame(rows)


# Fixed TOTAL width, slid below the peak: Ravasio et al. (2018, 2019) fit GBM's 8 keV-40 MeV,
# which with E_break ~ 100 keV and E_pk/E_break ~ 5-35 puts the peak ~2-2.5 dex above the
# bottom and ~1-2 below the top. The break needs that room below the peak to be measured;
# the symmetric (30, 30) window gives it 1.5 dex. (lower, upper) factors around nu_pk.
WINDOWS_3DEX = ((30., 30.), (100., 10.), (300., 3.))


def _wlab(lo, hi):
  hi_s = '' if hi == 1. else f'{hi:g}\\,'
  return rf'$\nu_{{\rm pk}}/{lo:g}$ to ${hi_s}\nu_{{\rm pk}}$'


def measure_shells_2sbpl(key=ssh.KEY, method='data_rarcut', windows=WINDOWS_3DEX):
  '''
  measure_shells with the 2SBPL, the Band fit carried alongside (alpha and beta), for
  each (lower, upper) window in `windows`.
  '''
  rows = []
  for p in ssh.load_shell_pairs(key=key, method=method):
    x = p['x']
    ok = x >= 1./p['env'].gma_m**2
    for kind in ('fluence', 'peak'):
      sp, _ = ssh.shell_spectra(p, kind)
      for tag in ('RS', 'tot'):
        for wl, wh in windows:
          f = fit_2sbpl(x[ok], sp[tag][ok], wh, wl)
          b = fit_band(x[ok], sp[tag][ok], wh, wl)
          f.update(alpha_band=b['alpha'], beta_band=b['beta'], rms_band=b['rms'])
          rows.append(dict(logC=round(float(p['log10ratio']), 2), kind=kind, shell=tag,
                           win_lo=wl, win_hi=wh, p=float(p['env'].psyn), **f))
  return pd.DataFrame(rows)


def _windows(df):
  return list(dict.fromkeys(zip(df.win_lo, df.win_hi)))


def _sel(df, kind, tag, w):
  return df[(df.kind == kind) & (df.shell == tag) & (df.win_lo == w[0])
            & (df.win_hi == w[1])].sort_values('logC')


def _beta_guide(ax, p):
  '''the one-zone index above the peak, -(p/2 + 1), labelled under the line at the right,
  where no curve passes (they all flatten towards C ~ 1-10 and stay above it in SC)'''
  ax.axhline(-(p/2. + 1.), c='0.5', ls=':', lw=0.9, zorder=0)
  ax.text(0.98, -(p/2. + 1.) - 0.01, r'$-(p/2+1)$', color='0.4', fontsize=8, ha='right',
          va='top', transform=ax.get_yaxis_transform())


def plot_shells_2sbpl(df, outdir, fname='twosbpl_shells_3dex.png'):
  '''
  The 2SBPL counterpart of plot_shells: alpha1, alpha2 and beta (rows), RS (dashed) vs
  RS+FS (solid), time-integrated | peak; colour = window, x = break at a window bound.
  '''
  ws = _windows(df)
  cols = swp.SWEEP_CMAP(np.linspace(0., 1., len(ws)))
  fig, axs = plt.subplots(3, 2, figsize=(9.6, 9.6), sharex=True, sharey='row',
                          layout='constrained', gridspec_kw={'height_ratios': [1., 1., .6]})
  for i, (q, qlab) in enumerate((('a1', r'$\alpha_1$'), ('a2', r'$\alpha_2$'),
                                 ('beta', r'$\beta$'))):
    for j, (kind, title) in enumerate((('fluence', 'time-integrated'), ('peak', 'peak'))):
      ax = axs[i, j]
      _beta_guide(ax, df.p.iloc[0]) if q == 'beta' else _guides(ax, df.p.iloc[0])
      for w, c in zip(ws, cols):
        for tag, ls in (('RS', '--'), ('tot', '-')):
          d = _sel(df, kind, tag, w)
          ax.plot(d.logC, d[q], c=c, ls=ls, lw=1.2, marker='o', ms=3.5,
                  mfc=c if tag == 'tot' else 'w')
          if q != 'beta':
            bad = d[d.at_bound]
            ax.plot(bad.logC, bad[q], ls='', marker='x', c='0.3', ms=6)
      ax.text(0.03, 0.80 if q != 'beta' else 0.85, f'{title}, {qlab}',
              transform=ax.transAxes, fontsize=10)
      if i == 2:
        ax.set_xlabel(r'$\log_{10}\mathcal{C}$')
    axs[i, 0].set_ylabel(f'2SBPL {qlab}')
  _legend(axs[0, 0], handles=[plt.Line2D([], [], c=c, marker='o', ms=3.5, label=_wlab(*w))
                              for w, c in zip(ws, cols)],
          loc='upper right', title='window', title_fontsize=8)
  _legend(axs[0, 1], handles=[plt.Line2D([], [], c='k', ls='--', marker='o', ms=3.5, mfc='w',
                                         label='RS'),
                              plt.Line2D([], [], c='k', ls='-', marker='o', ms=3.5,
                                         label='RS+FS'),
                              plt.Line2D([], [], c='0.3', ls='', marker='x', ms=6,
                                         label='break at window bound')],
          loc='upper left', bbox_to_anchor=(0., .78))
  os.makedirs(outdir, exist_ok=True)
  path = os.path.join(outdir, fname)
  fig.savefig(path, dpi=300)
  plt.close(fig)
  return path


def plot_band_vs_2sbpl(df, outdir, w=(30., 30.), fname=None):
  '''
  One window: Band alpha against the 2SBPL alpha1 and alpha2 (top), and the two betas
  (bottom); RS (red) and RS+FS (black), time-integrated | peak.
  '''
  fig, axs = plt.subplots(2, 2, figsize=(9.6, 6.4), sharex=True, sharey='row',
                          layout='constrained', gridspec_kw={'height_ratios': [1., .6]})
  est = (('alpha_band', 'Band $\\alpha$', '-', 'o'), ('a1', '2SBPL $\\alpha_1$', '--', 'v'),
         ('a2', '2SBPL $\\alpha_2$', ':', 's'))
  est_b = (('beta_band', 'Band $\\beta$', '-', 'o'), ('beta', '2SBPL $\\beta$', '--', 'v'))
  for j, (kind, title) in enumerate((('fluence', 'time-integrated'), ('peak', 'peak'))):
    ax, axb = axs[0, j], axs[1, j]
    _guides(ax, df.p.iloc[0])
    _beta_guide(axb, df.p.iloc[0])
    for tag, c in (('RS', COL_RS), ('tot', COL_TOT)):
      d = _sel(df, kind, tag, w)
      for q, _, ls, mk in est:
        ax.plot(d.logC, d[q], c=c, ls=ls, lw=1.1, marker=mk, ms=3.5)
      bad = d[d.at_bound]
      for q in ('a1', 'a2'):
        ax.plot(bad.logC, bad[q], ls='', marker='x', c='0.3', ms=7)
      for q, _, ls, mk in est_b:
        axb.plot(d.logC, d[q], c=c, ls=ls, lw=1.1, marker=mk, ms=3.5)
    axb.set_xlabel(r'$\log_{10}\mathcal{C}$')
    ax.text(0.03, 0.80, f'{title}, {_wlab(*w)}', transform=ax.transAxes, fontsize=10)
  axs[0, 0].set_ylabel('index below the peak')
  axs[1, 0].set_ylabel('index above the peak')
  _legend(axs[0, 0], handles=[plt.Line2D([], [], c='k', ls=ls, marker=mk, ms=3.5, label=lab)
                              for _, lab, ls, mk in est]
                             + [plt.Line2D([], [], c='0.3', ls='', marker='x', ms=6,
                                           label='break at window bound')],
          loc='upper right')
  _legend(axs[0, 1], handles=[plt.Line2D([], [], c=COL_RS, label='RS'),
                              plt.Line2D([], [], c=COL_TOT, label='RS+FS')], loc='upper right')
  _legend(axs[1, 1], handles=[plt.Line2D([], [], c='k', ls=ls, marker=mk, ms=3.5, label=lab)
                              for _, lab, ls, mk in est_b], loc='upper left', ncol=2)
  os.makedirs(outdir, exist_ok=True)
  path = os.path.join(outdir, fname or f'band_vs_2sbpl_shells_{w[0]:g}-{w[1]:g}.png')
  fig.savefig(path, dpi=300)
  plt.close(fig)
  return path


def main_shells_2sbpl(key=ssh.KEY, method='data_rarcut', windows=WINDOWS_3DEX):
  '''RS vs RS+FS 2SBPL and Band indices for one method, over 3-decade windows.'''
  out = figdir(ssh.OUTDIR_NAME if method == ssh.METHOD else f'{ssh.OUTDIR_NAME}_{method}', key)
  df = measure_shells_2sbpl(key, method, windows)
  csv = os.path.join(out, 'twosbpl_shells_3dex.csv')
  os.makedirs(out, exist_ok=True)
  df.to_csv(csv, index=False, float_format='%.5g')
  with pd.option_context('display.width', 250, 'display.precision', 2):
    for q in ('alpha_band', 'a1', 'a2', 'brk_dex', 'beta_band', 'beta'):
      print(f'--- {q}')
      print(df.pivot_table(index='logC', columns=['kind', 'shell', 'win_lo'], values=q))
    g = df.groupby(['win_lo', 'win_hi'])
    print(f"rms max 2SBPL {g.rms.max().round(3).to_dict()}, Band {g.rms_band.max().round(3).to_dict()}"
          f" dex; break at bound: {g.at_bound.sum().to_dict()} of {len(df)//len(windows)}")
  paths = [plot_shells_2sbpl(df, out)] + [plot_band_vs_2sbpl(df, out, w) for w in windows]
  print('wrote', *paths, csv)
  return df


def plot_shells(df, outdir, fname='band_alpha_shells.png', win=WIN):
  '''
  The Band-alpha counterpart of sweep_shells.plot_shell_slopes: RS (dashed) against RS+FS
  (solid), time-integrated | peak, alpha against C; colour = lower edge of the window.
  '''
  wls = sorted(df.win_lo.unique())
  cols = swp.SWEEP_CMAP(np.linspace(0., 1., len(wls)))
  fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True, layout='constrained')
  for ax, kind, title in zip(axs, ('fluence', 'peak'), ('time-integrated', 'peak')):
    _guides(ax, df.p.iloc[0])
    for wl, c in zip(wls, cols):
      for tag, ls in (('RS', '--'), ('tot', '-')):
        d = df[(df.kind == kind) & (df.shell == tag) & (df.win_lo == wl)].sort_values('logC')
        ax.plot(d.logC, d.alpha, c=c, ls=ls, lw=1.2, marker='o', ms=3.5,
                mfc=c if tag == 'tot' else 'w')
        bad = d[~d.whole]
        ax.plot(bad.logC, bad.alpha, ls='', marker='x', c='0.3', ms=6)
    ax.set_xlabel(r'$\log_{10}\mathcal{C}$')
    ax.text(0.03, 0.80, title, transform=ax.transAxes, fontsize=10)
  axs[0].set_ylabel(rf'Band $\alpha$  (to ${win:g}\,\nu_{{\rm pk}}$)')
  _legend(axs[0], handles=[plt.Line2D([], [], c=c, marker='o', ms=3.5,
                                      label=rf'$\nu_{{\rm pk}}/{wl:g}$')
                           for wl, c in zip(wls, cols)],
          loc='upper right', title='lower edge', title_fontsize=8)
  _legend(axs[1], handles=[plt.Line2D([], [], c='k', ls='--', marker='o', ms=3.5, mfc='w',
                                      label='RS'),
                           plt.Line2D([], [], c='k', ls='-', marker='o', ms=3.5,
                                      label='RS+FS')],
          loc='upper right')
  os.makedirs(outdir, exist_ok=True)
  path = os.path.join(outdir, fname)
  fig.savefig(path, dpi=300)
  plt.close(fig)
  return path


def main_shells(key=ssh.KEY, method='data_rarcut', win=WIN, win_los=WIN_LO_CMP):
  '''RS vs RS+FS Band alpha for one method, written next to plot_shell_slopes' figure.'''
  out = figdir(ssh.OUTDIR_NAME if method == ssh.METHOD else f'{ssh.OUTDIR_NAME}_{method}', key)
  df = measure_shells(key, method, win, win_los)
  csv = os.path.join(out, 'band_alpha_shells.csv')
  os.makedirs(out, exist_ok=True)
  df.to_csv(csv, index=False, float_format='%.5g')
  with pd.option_context('display.width', 250, 'display.precision', 2):
    print(df.pivot_table(index='logC', columns=['kind', 'shell', 'win_lo'], values='alpha'))
    print(f"rms max {df.groupby('win_lo').rms.max().round(3).to_dict()} dex; incomplete "
          f"windows: {df[~df.whole][['logC', 'kind', 'shell', 'win_lo']].values.tolist()}")
  print('wrote', plot_shells(df, out, win=win), 'and', csv)
  return df


def main_windows(key=ssh.KEY, win=WIN, win_los=WIN_LO_CMP):
  '''The Band alpha for several lower window edges, upper edge fixed at win*nu_pk.'''
  out = figdir(OUTDIR_NAME, key)
  df = measure(key, win, win_los)
  csv = os.path.join(out, 'band_alpha_windows.csv')
  os.makedirs(out, exist_ok=True)
  df.to_csv(csv, index=False, float_format='%.5g')
  piv = df[df.method == 'full'].pivot_table(index='logC', columns=['epoch', 'shell', 'win_lo'],
                                            values='alpha')
  with pd.option_context('display.width', 250, 'display.precision', 2):
    print(piv)
    print(f"rms max {df.groupby('win_lo').rms.max().round(3).to_dict()} dex; incomplete "
          f"windows: {df[~df.whole][['logC', 'method', 'shell', 'epoch', 'win_lo']].values.tolist()}")
  print('wrote', plot_windows(df, out, win=win), 'and', csv)
  return df


def main(key=ssh.KEY, win=WIN):
  out = figdir(OUTDIR_NAME, key)
  df = measure(key, win)
  csv = os.path.join(out, 'band_alpha.csv')
  os.makedirs(out, exist_ok=True)
  df.to_csv(csv, index=False, float_format='%.5g')
  piv = df.pivot_table(index='logC', columns=['epoch', 'shell', 'method'], values='alpha')
  with pd.option_context('display.width', 200, 'display.precision', 2):
    print(piv)
    print(f"rms max {df.rms.max():.3f} dex; incomplete windows: "
          f"{df[~df.whole][['logC', 'method', 'shell', 'epoch']].values.tolist()}")
  print('wrote', plot(df, out, win=win), 'and', csv)
  return df


if __name__ == '__main__':
  key = os.environ.get('BAND_KEY', ssh.KEY)
  if os.environ.get('BAND_2SBPL'):
    main_shells_2sbpl(key)
  elif os.environ.get('BAND_SHELLS'):
    main_shells(key)
  elif os.environ.get('BAND_WINDOWS'):
    main_windows(key)
  else:
    main(key)
