'''
Refine the cooling steps on the HYDRO AS THE EMISSION SEES IT (V3p and nu'_B), and test
whether that removes the slow-cooling dip above nu_c.

READ THE VERDICT BELOW BEFORE THE MOTIVATION: this file's original premise was half right,
and the half that was right got fixed somewhere else.

ORIGINAL MOTIVATION (2026-08-21). The per-step radiated energy departed from the exact
electron budget by up to 18% in the late steps of a slow-cooling cell, correlating with step
duration (corr 0.97 with dtp) and with hydro drift across the step (corr 0.82 with
|dV3p/V3p|), and was EXACTLY 1.0000 in a frozen-hydro cell. The binning had no guard for it:
r_ref is geometric in gamma_max and dlnrho_max refines on rho, and neither bounds V3p or
nu'_B, which are what set the emission.

VERDICT, in two parts.

  ON THE DIP: NEGATIVE, and this still stands. 8.4x more sub-steps (N 41 -> 346) with the
  drift capped at 0.5% moved the GS02 rms by 0.0003 dex (0.0520 -> 0.0517) and the slope at
  +1 dex by 0.001; fast cooling was untouched at 0.0132. Within-step hydro drift is NOT the
  cause of the nu_c feature. The sign said so in advance: a deficit in the late steps would
  make the fall above nu_c shallower, not steeper. What the figure does show is that the
  residual is a clean +0.15 dex EXCESS peaking at nu_c flanked by a -0.10 dex trough -- a
  bump, not a dip, and the "slope dip" is its derivative.

  ON THE ENERGY ERROR: REAL, and now FIXED -- but by a second-order quadrature rule, not by
  refinement. Reading the emission prefactor (Aad*Pmax*V3p*nu'_B) at each step's LEFT edge
  is a left-rectangle rule, so the cure is to read it at the step MIDPOINT, which costs
  nothing and needs no extra steps. Shipped in commit fcf92a1 as
  precompute_step_cols(midpoint_hydro=True); see midpoint_hydro_test.py for the measurement.
  It was worth +2.0/+2.6/+3.2% on a cell's E_rad at log10(gc/gm) = 0/+1/+3 and 1.9-2.4% on
  the SHELL eps_rad, always one sign, ~0 in fast cooling.

So this test asked the right question of the wrong lever. Refining the steps and midpointing
the prefactor converge to the same answer; midpointing gets there at the production step
count, which is why the fix went there and this file stayed a diagnostic.

WHY IT COULD NOT SEE ITS OWN POINT: it judged on the GS02 fit rms, which is a SHAPE statistic
and is blind to a smooth few-percent normalisation -- exactly the error that was present. It
never looked at E_rad. Anything measuring a normalisation must say so explicitly; a fit
residual will absorb it.

NUMBERS ABOVE PREDATE two changes -- the Aad renormalisation (5d2294d) and the midpoint
(fcf92a1) -- so the baseline this script now draws as 'as-is' is the midpointed one, and the
refinement it applies on top should move things LESS than it used to. Re-run before quoting
any figure from it.

  python -c "import hydro_refine_test as H; H.main()"
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from environment import GAMMA_dir, MyEnv
from working_cooling import (generate_cell_withDistrib, precompute_step_cols, step_view,
                             norm_plaw_distrib, _midpoint_cell, step_radiated_energy)
from radiation_cooling import get_Fnu_step, cooled_tt_eff
from cooling_distribution import distrib_plaw_cooled
from IO import open_celldata, open_rundata
from fits_hydro import cellsBehindShock_fromData
from sweep_gammacm import compute_alpha_sweep, fit_gs02_spectrum, gs02_model
import prerar_model as M

KEY, Z = 'cooling_g100', 4
# all the nu_c-dip diagnostics land in one folder so they are not scattered
OUTDIR = os.path.join(GAMMA_dir, 'bin', 'Tools', 'figures', 'nuc_dip')
LOGC = ('rho', 'p', 'gmin', 'gmax', 'bsyn', 'Aad', 'lfac', 'dx')   # interpolated in log
TRAPZ = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz


def refine_on_hydro(cell, env, dmax=0.01, Mmax=64):
  '''
  Split every step into as many sub-steps as it takes for the EMISSION-RELEVANT hydro
  (V3p, nu'_B) to move by at most dmax across each. dt/dtp/dtt are divided among them, so
  the total radiated energy is unchanged by construction; the states are interpolated.
  '''
  cols = precompute_step_cols(cell, env)
  V, nB = cols['V3p'], cols['nup_B']
  n = len(cell)
  dV = np.abs(np.diff(V)/V[:-1]); dB = np.abs(np.diff(nB)/nB[:-1])
  drive = np.concatenate([np.maximum(dV, dB), [0.]])
  A = cell.reset_index(drop=True)
  out = {c: [] for c in cell.columns}
  for j in range(n):
    Mj = int(min(Mmax, max(1, np.ceil(drive[j]/dmax))))
    a = A.iloc[j]; b = A.iloc[min(j+1, n-1)]
    for m in range(Mj):
      f = m/Mj
      for c in cell.columns:
        va, vb = float(a[c]), float(b[c])
        if c in ('dt', 'dtp', 'dtt'):
          out[c].append(va/Mj)
        elif c in LOGC and va > 0 and vb > 0:
          out[c].append(np.exp((1-f)*np.log(va) + f*np.log(vb)))
        else:
          out[c].append((1-f)*va + f*vb)
  return pd.DataFrame(out)


def _spectrum(cell, env, K0, nu):
  cm = _midpoint_cell(cell)
  cols = precompute_step_cols(cm, env)
  Ton, Tth, Tej = cols['obsT']
  T = float(Ton.max())
  tot = np.zeros_like(nu)
  for j in range(len(cell)):
    tot += np.asarray(get_Fnu_step(nu*env.nu0, T, step_view(cols, j), K0, env), float)*nu
  return np.where(np.isfinite(tot) & (tot > 0), tot, 0.), cols


def _energy_error(cell, env, K0, p):
  cl = precompute_step_cols(cell, env)
  V, bs, dt = cl['V3p'], cl['bsyn'], cl['dtt']
  gm = cell.gmin.to_numpy(float); gx = cell.gmax.to_numpy(float)
  Ec, Eb = [], []
  for j in range(len(cell)):
    if gx[j] <= 1.05 or gm[j] >= gx[j]:
      continue
    Ec.append(float(step_radiated_energy(step_view(cl, j), K0, env))*V[j])
    g = np.geomspace(max(gm[j], 1.), gx[j], 3000)
    Nj = distrib_plaw_cooled(g, p, cooled_tt_eff(gx[j], bs[j]))
    Eb.append(V[j]*TRAPZ(Nj*g*g*dt[j]/(1. + g*dt[j]), g))
  r = np.array(Ec)/np.array(Eb)
  return r/np.median(r)


def main(Ri_target=1.05, logrs=(-4., 3.), dmaxes=(None, 0.02, 0.005), key=KEY, z=Z,
    outdir=OUTDIR):
  cells = M._test_cells(key, z, 120)
  k = int(min(cells, key=lambda c: abs(c[4] - Ri_target))[0])
  raw = open_celldata(key, k)
  shr = cellsBehindShock_fromData(open_rundata(key, z))
  d0 = shr.loc[shr.i == raw.iloc[0].i].iloc[0]
  ex = shr.loc[shr.t.idxmax()]
  nu = np.logspace(-4, 10, 1100)
  ln = np.log10(nu)

  fig, axes = plt.subplots(2, len(logrs), figsize=(7.8*len(logrs), 9.2),
                           layout='constrained')
  axes = np.atleast_2d(axes)
  cols_ = ('#999999', '#0072B2', '#D55E00')
  print(f'{"logr":>5} {"dmax":>7} {"N":>5} {"GS02 rms":>9} {"max res":>8}'
        f' {"late E err":>11}   slopes above peak')
  for c, logr in enumerate(logrs):
    alpha = float(compute_alpha_sweep(key, np.array([float(logr)]))[0][0])
    base, env = generate_cell_withDistrib(raw, d0, MyEnv(key), alpha=alpha, r_ref=1.2,
                    Tmax=None, key=key, k=k, exit_row=ex)
    p = env.psyn
    c0 = base.iloc[0]
    K0 = norm_plaw_distrib(c0.gmin, c0.gmax, p)
    nu_c = 10.**(2.*logr)
    for col, dmax in zip(cols_, dmaxes):
      cell = base if dmax is None else refine_on_hydro(base, env, dmax)
      sp, cc = _spectrum(cell, env, K0, nu)
      g = np.gradient(np.log10(np.where(sp > 0, sp, np.nan)), ln)
      jp = int(np.nanargmax(sp))
      seg = [np.nanmedian(g[(ln-ln[jp] >= x-.4) & (ln-ln[jp] < x+.4)])
             for x in (0.5, 1., 1.5, 2., 3.)]
      nuM0 = 1.5*cell.gmax.to_numpy(float)[0]**2*cc['nup_B'][0]*cc['Dop'][0]/env.nu0
      fit = fit_gs02_spectrum(nu, sp, p, nuM0)
      rms = mx = np.nan
      if fit is not None:
        mod = gs02_model(nu, fit, p)
        m = (sp > sp.max()*1e-6) & (mod > 0)
        res = np.log10(sp[m]/mod[m])
        rms, mx = np.sqrt(np.mean(res**2)), np.max(np.abs(res))
        axes[1, c].semilogx(nu[m], res, color=col, lw=1.3)
      err = _energy_error(cell, env, K0, p)
      q = max(1, len(err)//4)
      lab = 'as-is' if dmax is None else rf'refined, $\delta_{{\max}}$={dmax}'
      axes[0, c].semilogx(nu, g, color=col, lw=1.5,
                          label=f'{lab}  (N={len(cell)}, rms={rms:.4f})')
      print(f'{logr:>5.0f} {str(dmax):>7} {len(cell):>5} {rms:9.4f} {mx:8.4f}'
            f' {np.median(err[3*q:]):11.4f}   ' + ' '.join(f'{v:+.3f}' for v in seg))
    for y, t in ((4./3., '4/3'), ((3.-p)/2., '(3-p)/2'), (-(p-2.)/2., '1-p/2')):
      axes[0, c].axhline(y, color='grey', lw=.8, ls=':')
      axes[0, c].annotate(t, xy=(1.003, y), xycoords=('axes fraction', 'data'),
                          fontsize=7, color='grey', va='center')
    for ax in axes[:, c]:
      ax.axvline(nu_c, color='crimson', ls='--', lw=1.1)
      ax.grid(alpha=.25)
    axes[0, c].set(title=rf'$\log_{{10}}(\gamma_c/\gamma_m)$ = {logr:+.0f}'
                         r'   (red dashed = $\nu_c$)',
                   ylabel=r'$d\log(\nu F_\nu)/d\log\nu$', ylim=(-1.2, 1.6))
    axes[0, c].legend(fontsize=8, loc='lower left')
    axes[1, c].axhline(0., color='grey', lw=.8, ls=':')
    axes[1, c].set(xlabel=r'$\nu/\nu_m$', ylabel='log10( spectrum / GS02 fit )',
                   ylim=(-0.25, 0.25))
  fig.suptitle(f'Refining the steps on the emission-relevant hydro (V3p, '
               r"$\nu'_B$)" f'   cell k={k}', fontsize=12)
  path = os.path.join(outdir, 'hydro_refine_test.png')
  fig.savefig(path, dpi=140, bbox_inches='tight')
  plt.close(fig)
  print(f'saved {path}')


if __name__ == '__main__':
  main()
