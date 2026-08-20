# -*- coding: utf-8 -*-
# @Author: acharlet

'''
One cell's hydrodynamics, reference vs no-rarefaction counterfactual.

The illustration figure for the construction in working_cooling_data.norar_history: real
snapshot rows up to the rarefaction handover, then a synthetic tail closed on mass
conservation + the Taub-Matthews adiabat + Bernoulli (see _bernoulli_state), carried to
the SAME final radius the cell reached in the data.

The shared pre-handover stretch is drawn ONCE, in ink rather than in a series colour,
because the two treatments are bit-identical there by construction -- plotting it twice
would invite the reader to look for a difference that cannot exist. Only the two branches
past the handover carry series colour.

  python -c "import norar_cell_figure as F; F.main()"
  python -c "import norar_cell_figure as F; F.main(k=100)"
'''

import os
import numpy as np
import matplotlib.pyplot as plt

from environment import MyEnv, GAMMA_dir
from working_cooling_data import rarefaction_handover, norar_history
import sweep_norar as N

KEY, Z = 'cooling_g100', 4
OUTDIR = os.path.join(GAMMA_dir, 'bin', 'Tools', 'figures', 'norar_compare')

# Okabe-Ito blue / vermillion: validated (CVD dE 21.9 protan, 30.9 tritan; normal 31.2;
# contrast >= 3:1 on a light surface). Identity is carried by linestyle AS WELL as hue,
# so the pair never relies on colour alone.
COL_FULL, COL_NORAR = '#0072B2', '#D55E00'
COL_INK, COL_MUTED = '#1a1a1a', '#8a8a8a'
LW = 2.0


def cell_profiles(key=KEY, k=300, z=Z):
  '''(R/R_inj, reference, counterfactual, handover index) for one cell, both treatments
  built exactly as the emission pipeline builds them.'''
  s = N._history(key, k, z=z)
  if s is None:
    raise RuntimeError(f'cell {k} has no usable post-shock history')
  h = rarefaction_handover(s)
  ext, info = norar_history(s)
  return s, ext, (h if h is not None else len(s) - 1), info


def main(key=KEY, k=300, z=Z, outdir=OUTDIR, fname=None):
  os.makedirs(outdir, exist_ok=True)
  s, ext, h, info = cell_profiles(key, k, z)

  def cols(f):
    x = f.x.to_numpy(float)
    g = 1./np.sqrt(1. - f.vx.to_numpy(float)**2)
    return x, f.rho.to_numpy(float), g, f.p.to_numpy(float)

  x0 = float(s.x.iloc[0])
  xf, rf, gf, pf = cols(s)          # reference: real data, crash included
  xe, re, ge, pe = cols(ext)        # counterfactual: real to h, then synthetic
  r_h = xf[h]/x0

  fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.4), sharex=True, layout='constrained',
                           gridspec_kw=dict(hspace=0.06))
  panels = ((r'$\rho\,/\,\rho_{\rm inj}$', rf/rf[0], re/re[0], 'log'),
            (r'$\Gamma\,/\,\Gamma_{\rm inj}$', gf/gf[0], ge/ge[0], 'linear'),
            (r'$p\,/\,p_{\rm inj}$', pf/pf[0], pe/pe[0], 'log'))

  for ax, (lab, yf, ye, yscale) in zip(axes, panels):
    # shared trunk: identical in both treatments, so drawn once and in ink
    ax.plot(xf[:h+1]/x0, yf[:h+1], color=COL_INK, lw=LW, solid_capstyle='round',
            zorder=3)
    # the two branches
    ax.plot(xf[h:]/x0, yf[h:], color=COL_FULL, lw=LW, solid_capstyle='round',
            zorder=2)
    ax.plot(xe[h:]/x0, ye[h:], color=COL_NORAR, lw=LW, ls='--', dash_capstyle='round',
            zorder=4)
    ax.axvline(r_h, color=COL_MUTED, lw=1., ls=':', zorder=1)
    ax.set_xscale('log')
    ax.set_yscale(yscale)
    ax.set_ylabel(lab)
    ax.grid(True, which='major', color='0.92', lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
      ax.spines[sp].set_visible(False)

  # the handover label goes BELOW the curves in the log-rho panel, where both branches
  # have already fallen away, so it collides with neither
  axL = axes[0]
  axL.annotate('rarefaction reaches this cell', xy=(r_h, 1.), xycoords=('data', 'axes fraction'),
               xytext=(r_h*1.6, 0.08), textcoords=('data', 'axes fraction'),
               fontsize=8, color=COL_MUTED, ha='left', va='center',
               arrowprops=dict(arrowstyle='-', color=COL_MUTED, lw=0.8,
                               shrinkA=0, shrinkB=2))
  axes[-1].set_xlabel(r'$R\,/\,R_{\rm inj}$')

  hnd = [plt.Line2D([], [], color=COL_INK, lw=LW, label='shared (real data)'),
         plt.Line2D([], [], color=COL_FULL, lw=LW, label='full (crash in the data)'),
         plt.Line2D([], [], color=COL_NORAR, lw=LW, ls='--',
                    label='no rf (Bernoulli extension)')]
  axL.legend(handles=hnd, fontsize=8, frameon=False, loc='upper right')
  env = MyEnv(key)
  axL.set_title(f'{key}, cell k={k} ({"RS" if z == 4 else "FS"})   '
                rf'$R_h/R_{{\rm inj}}={r_h:.2f}$,  '
                rf'$R_{{\rm end}}/R_{{\rm inj}}={xf[-1]/x0:.0f}$', fontsize=10)
  fname = fname or f'cell_hydro_norar_k{k:04d}.png'
  path = os.path.join(outdir, fname)
  fig.savefig(path, dpi=160, bbox_inches='tight')
  plt.close(fig)
  print(f'k={k}: handover row {h} at R/R_inj={r_h:.3f}, {info["n_syn"]} synthetic rows, '
        f'target R/R_inj={xf[-1]/x0:.1f}')
  print(f'saved {path}')
  return path


if __name__ == '__main__':
  main()
