# -*- coding: utf-8 -*-
# @Author: acharlet

'''
How far does a cell propagate before it has radiated its energy away, and what does that
distance do to the fast-cooling mid slope?

THE QUANTITY. Follow one cell from the moment it is shocked and accumulate the comoving
energy it radiates, step by cooling step -- the very sum working_cooling.cell_radiated_energy
returns as a single number, kept as a running total instead (cell_radiated_profile, whose
sum reproduces that number to round-off). The radius at which the running total first
reaches a fraction f of the energy injected into the cell's electrons (cell_injected_energy)
is R_f, and

    ell_f = R_f/R_inj - 1        the RADIATIVE LENGTH, in units of the injection radius
    dT_f  = barT_f - barT_inj    the same thing on the observer clock, in units of env.T0

are what this module measures. f = 0.9 is the working value: "all of its energy" cannot be
taken literally (the last percent is radiated by electrons that have cooled to gamma ~ 1,
over a distance that diverges), and 0.5 is carried alongside to show the answer is not an
artefact of where the threshold sits -- ell_0.9/ell_0.5 is 14.2, 14.5, 14.8, 17.2 across
the four points, i.e. a constant of the geometry and not a regime dependence.

WHAT IT COMES OUT AT (cooling_g100 RS, data_rarcut, 500 parent cells per point):

    log10(gc/gm)      -5        -4        -3        -2
    ell_50/R       7.7e-06   7.4e-05   7.2e-04   7.1e-03
    ell_90/R       1.1e-04   1.1e-03   1.1e-02   9.8e-02
    ell_90/(gc/gm)   11.0      10.8      10.7       9.8
    dT_90/barT_f   1.1e-04   8.2e-04   6.4e-03   7.8e-02
    cells that never reach f=0.9        0%        0%        0%       21%

i.e. ell ~ 10 (gamma_c/gamma_m) R over three decades, which is the closed-form expectation
and not a fit: the electrons carrying the energy sit at gamma_m, their cooling time is
t_cool(gamma_m) = (gamma_c/gamma_m) t_dyn by the definition of gamma_c, and a cell covers
R per t_dyn. The scaling is exact in a stronger sense than that table shows -- the RIGHT
panel of the figure divides ell by gamma_c/gamma_m and the four regimes fall on ONE curve,
3.2 at the first-shocked cell, 10.8 at mid-shell, ~23 at the last, so the whole regime
dependence of ell is gamma_c/gamma_m and the rest is the shell's own history (B' decays as
it expands, so every later cell cools in a weaker field). The distance is MEASURED here,
but it is also predictable; the value of measuring it is that it is the same number the
spectrum sees.

THE LINK TO a_mid. The shell-integrated fast-cooling mid slope is not the one-zone 1/2
(see sweep_gammacm.SEG_FC_TOL_HI): cells shocked at different times carry different
gamma_c, and summing their one-zone spectra hardens the segment between the breaks. What
sets the width of that smearing is how long a cell stays in the act of cooling -- which is
exactly ell. A cell whose radiative length is short compared with the shell's own scale has
finished before its neighbour is shocked, so at any instant only a thin freshly-shocked skin
is emitting, every cell in it at the same stage, and the sum is one-zone. As ell grows the
emitting population spans more and more cooling stages at once and the segment hardens.

The module measures that composition directly rather than assuming it: at each observer
time it takes the steps whose arrival window is open, weights them by the energy they
actually radiate (times Dop^3/Tth, the same observed weight cooling_frequency.step_weights
builds), and reports

    f_act    the fraction of the arriving light emitted by cells that have NOT yet passed
             f = 0.9, i.e. that are still radiating their energy away
    spread   the weighted 10-90 percentile width of log10 nu_c over those steps, in dex
    sep      log10(nu_m/nu_c) over the same population -- the width of the segment the
             spread is smeared across

and puts them beside dep = a_mid - 1/2 read from mid_slope_evolution's cached rows.

FINDINGS. Regime for regime the correspondence is clean:

    log10(gc/gm)      -5        -4        -3        -2
    ell_90/R       1.1e-04   1.1e-03   1.1e-02   9.8e-02
    a_mid - 1/2     +0.004    +0.011    +0.053    no settled core, anywhere
    widest mid window (dex)   5.6       5.3       3.8       --
    (dep median over the settled fc cores at barT > BART_LO; the widest identified window)

a_mid is the one-zone 1/2 while a cell finishes inside a thousandth of its radius, departs
by +0.05 once it needs a per cent of it, and by a tenth of a radius the segment does not
settle at all -- 0 of the 112 sampled spectra at logr=-2 have a flat core. ell also grows
ALONG the shell, by x7 (logr=-5) to x11 (-3) between the first-shocked cell and the last,
because B' decays as the shell expands and each later cell cools in a weaker field.

ell alone does NOT predict dep, and the time-resolved figure is what shows it. At logr=-3
the two rise together through the pulse (dep +0.033 -> +0.098 at barT ~ 1.2, the emission-
weighted <ell> 3.3e-3 -> 1.1e-2, correlation +0.59 in log ell). At -4 and -5 they run the
other way: dep peaks early (+0.105 at barT = 0.019, +0.017 at 0.088) and is back at zero by
the crossing while <ell> only grows, so the correlation is NEGATIVE (-0.69 and -0.88). That
is not a contradiction -- ell is the width of the smearing, and what dep responds to is that
width relative to the width of the SEGMENT being smeared, which narrows through the pulse
(median mid window 5.1 dex at logr=-5 against 3.3 at -3). ell carries the first factor and
nothing about the second, so it is necessary and not sufficient.

TRIED, DOES NOT WORK -- do not re-derive. The obvious closure is to predict the slope from
the population directly. One cell is 4/3 below its own nu_c and 1/2 above it, so
    a(nu) = <a_i(nu)>_w = 1/2 + (5/6) W(nu),   W = weight fraction with nu_c,i > nu
and W is exactly "the light still coming from cells in the act of radiating". Evaluated at
the geometric centre of the mid window identify_segments actually used, it returns W =
0.98-0.99 and a = 1.30 against a measured 0.50-0.59, at every point and every time. The
failure is in the WEIGHT, not in the idea: w is the step's BOLOMETRIC output, while a step
whose nu_c is far above nu is on its own nu^(4/3) rise there and delivers a tiny fraction of
that output at that frequency. A weight valid at one frequency is the per-cell spectrum,
i.e. the emission kernel itself, so this is not a shortcut to it. Closing the link needs a
frequency-resolved decomposition of the shell sum, not another summary of {nu_c,i}.

SCOPE. Parent cells only (subcell_dlogT=None). A sub-cell is not an independent emitter --
it is a slice of its parent's history introduced to smooth the early-time arrival staircase
-- so counting sub-cells would weight the first-shocked cells by their sub-cell count in
every statistic here. The price is that the composition is under-resolved at barT < 1e-2,
where the 500 parents are still arriving one by one; BART_LO drops that range, as
nuc_validation does for the same reason. Everything else (r_ref, Tmax, TB_*, EARLY_ANA,
rar_cut) tracks sweep_gammacm so the cells are the ones the cached spectra were computed
from.

Example use in command line:
  python -c "import radiative_length as R; R.main()"
  python -c "import radiative_length as R; R.main(use_cache=False)"
  python -c "import radiative_length as R; R.main(log10ratio_arr=[-5.,-4.,-3.])"
'''

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from IO import get_variable
from cooling_distribution import norm_plaw_distrib
from radiation_cooling import precompute_step_cols, step_view, step_radiated_energy
from working_cooling import cell_injected_energy
import cooling_frequency as cf
from sweep_gammacm import (DEFAULT_KEY, Z_SHELL, TMAX, TB_MIN, TB_LIN, R_REF, EARLY_ANA,
    compute_alpha_sweep, exit_onset_barT, rarefaction_off_barT, method_outdir,
    trim_pngs, copy_article_figures)
from mid_slope_evolution import COL, INK, MUTED, GRID, read_rows, METHOD

KEY = DEFAULT_KEY
Z = Z_SHELL
LOG10RATIO_ARR = (-5., -4., -3., -2.)   # the fast-cooling side of the sweep. Above -2 the
                          # mid segment is the SLOW-cooling one and ell is not defined the
                          # same way (the cell never radiates its budget: eps_rad < 1), so
                          # the link this module makes has no slow-cooling counterpart.
FRACS = (0.5, 0.9)        # radiated-energy thresholds. 0.9 is the working value (F_MAIN);
F_MAIN = 0.9              # 0.5 is the control -- if the regime scaling were an artefact of
                          # the threshold the two would not be a constant factor apart.
NT_HARVEST = 250          # observer grid handed to iter_shell_cells. It fixes the arrival
                          # bookkeeping only (Ton/Tth per step), never the worldline, so it
                          # need not be the sweep's NT=1800; the composition is evaluated on
                          # BART_GRID below in any case.
BART_GRID = np.geomspace(1e-3, 900., 200)    # composition sampling, log in barT: the
                          # a_mid rows run from 3e-4 to 9e2 and nothing here has structure
                          # finer than ~10 points/decade
BART_LO = 1e-2            # below this the parent-cell population is still arriving one cell
                          # at a time (see SCOPE); composition rows survive but are excluded
                          # from every correlation, exactly as nuc_validation.BART_LO does
DOP_EXP = cf.DOP_EXP      # one definition of the observed per-step weight, shared
CELLS_CSV = 'radiative_length_cells.csv'
TIME_CSV = 'radiative_length_time.csv'
FIG_LEN = 'radiative_length.png'
FIG_LINK = 'radiative_length_amid.png'
CELL_FIELDS = ('logr', 'cell', 'barT0', 'R0', 'eps', 'nstep') \
              + tuple(f'{p}{f:g}' for f in FRACS for p in ('ell', 'dT'))
TIME_FIELDS = ('logr', 'barT', 'nact', 'f_act', 'frac_w', 'age_w', 'ell_w',
               'nuc_q10', 'nuc_q50', 'nuc_q90', 'spread', 'num_q50', 'sep')


# ---------------------------------------------------------------------------
# one cell: the radiated energy along its worldline

def cell_radiated_profile(cell, env, Ng=120, width_tol=1.01):
  '''
  Comoving energy radiated by each cooling step of a cell, as an array.

  This is working_cooling.cell_radiated_energy with the sum left un-taken: same
  precompute_step_cols (so the same midpointed prefactor), same K0 on the injection
  bounds, same step_radiated_energy per step. cell_radiated_profile(...).sum() therefore
  reproduces cell_radiated_energy(...) to round-off, which is what makes a cumulative
  fraction built from it commensurate with the eps_rad the efficiency sweep quotes.
  '''
  cols = precompute_step_cols(cell, env, keys=('nup_B', 'V3p', 'Pmax'))
  nupB, V3p = cols['nup_B'], cols['V3p']
  c0 = cell.iloc[0]
  K0 = norm_plaw_distrib(c0.gmin, c0.gmax, env.psyn)
  return np.array([step_radiated_energy(step_view(cols, j), K0, env, Ng, width_tol)
                   * nupB[j] * V3p[j] for j in range(len(cell))])


def radiative_length(cell, env, dE=None, fracs=FRACS):
  '''
  Radiative length of one cell: how far it propagates, and how long it takes on the
  observer clock, before its running radiated energy reaches each fraction of the energy
  injected into its electrons.

  The threshold is crossed inside a cooling step, so R_f is interpolated within that step
  -- geometrically in R (the worldline is a power law over a step, and the steps are
  geometric in gamma_max), linearly in barT (the arrival time is not). A cell whose
  worldline ends before the threshold is reached returns NaN for that fraction rather
  than its last radius: those are CENSORED, not long, and averaging their last radius in
  would bias the answer short exactly where the true value is longest.

  Returns dict(R0, barT0, eps, nstep, ell<f>, dT<f>) -- ell in units of R_inj, dT in
  units of env.T0, eps = the cell's own E_rad/E_inj.
  '''
  if dE is None:
    dE = cell_radiated_profile(cell, env)
  E_inj = cell_injected_energy(cell, env)
  x = cell['x'].to_numpy(dtype=float)
  barT = np.asarray((get_variable(cell, 'Ton', env) - env.Ts)/env.T0, dtype=float)
  cum = np.cumsum(dE)/E_inj if E_inj > 0. else np.full(len(dE), np.nan)
  out = dict(R0=float(x[0]), barT0=float(barT[0]), eps=float(cum[-1]), nstep=len(cell))
  for f in fracs:
    j = int(np.searchsorted(cum, f))
    if j >= len(cum):
      out[f'ell{f:g}'], out[f'dT{f:g}'] = np.nan, np.nan
      continue
    lo = cum[j-1] if j else 0.
    w = (f - lo)/(cum[j] - lo) if cum[j] > lo else 0.
    if j:
      Rf = x[j-1]*(x[j]/x[j-1])**w
      Tf = barT[j-1] + w*(barT[j] - barT[j-1])
    else:
      Rf, Tf = x[0], barT[0]
    out[f'ell{f:g}'] = float(Rf/x[0] - 1.)
    out[f'dT{f:g}'] = float(Tf - barT[0])
  return out


# ---------------------------------------------------------------------------
# one sweep point: every cell of the shell

def _harvest_kwargs():
  '''
  iter_shell_cells settings that make the harvested cells the ones the cached spectra
  were computed from. Only subcell_dlogT departs from sweep_gammacm (see SCOPE), and
  early_frac is pinned to the driver's default 0. -- cooling_frequency's own default is
  0.1, which would prepend a different injection row.
  '''
  return dict(r_ref=R_REF, Tmax=TMAX, NT=NT_HARVEST, Tb_min=TB_MIN, Tb_lin=TB_LIN,
              subcell_dlogT=None, early_ana=EARLY_ANA, early_frac=0., rar_cut='model',
              verbose=False)


def harvest_point(logr, key=KEY, z=Z, fracs=FRACS, verbose=True):
  '''
  One sweep point: the per-cell radiative lengths and the per-step table the observer-time
  composition is built from.

  The step weight is dE_j * Dop^3 / Tth -- cooling_frequency.step_weights with the step's
  ACTUAL radiated energy in place of its prefactor. The prefactor version is a good proxy
  when only ratios at one instant are wanted; here the electron integral is already
  computed for the profile, so the exact energy costs nothing and keeps the weights
  consistent with the cumulative fraction they are used alongside.

  Returns (cell rows, dict of per-step arrays).
  '''
  alpha = float(compute_alpha_sweep(key, [logr])[0][0])
  rows, cols = [], []
  for k, cell, env in cf.iter_shell_cells(key, z=z, alpha=alpha, **_harvest_kwargs()):
    dE = cell_radiated_profile(cell, env)
    r = radiative_length(cell, env, dE=dE, fracs=fracs)
    E_inj = cell_injected_energy(cell, env)
    tab = cf.cell_cooling_table(cell, env)
    x = cell['x'].to_numpy(dtype=float)
    n = len(cell)
    cols.append(dict(
        barT=tab.barT.to_numpy(dtype=float), Tth=tab.Tth_b.to_numpy(dtype=float),
        w=dE*tab.Dop.to_numpy(dtype=float)**DOP_EXP/tab.Tth_b.to_numpy(dtype=float),
        frac=np.cumsum(dE)/E_inj, nu_c=tab.nu_c.to_numpy(dtype=float),
        nu_m=tab.nu_m.to_numpy(dtype=float), fresh=tab.fresh.to_numpy(dtype=bool),
        barT0=np.full(n, r['barT0']), ell=x/x[0] - 1.,
        ell90=np.full(n, r[f'ell{F_MAIN:g}']), cell=np.full(n, k)))
    r.update(logr=logr, cell=k)
    rows.append(r)
  if not cols:
    raise RuntimeError(f'no usable cells at logr={logr} (key={key}, z={z})')
  H = {c: np.concatenate([d[c] for d in cols]) for c in cols[0]}
  if verbose:
    ell = np.array([r[f'ell{F_MAIN:g}'] for r in rows], dtype=float)
    print(f'  logr={logr:+.1f}: {len(rows)} cells, {len(H["barT"])} steps, '
          f'ell{F_MAIN:g} median {np.nanmedian(ell):.3e} '
          f'({np.isnan(ell).mean()*100:.0f}% censored)', flush=True)
  return rows, H


def _wq(v, w, q):
  '''weighted quantile of v (finite, w >= 0)'''
  o = np.argsort(v)
  v, w = v[o], w[o]
  c = np.cumsum(w)
  return float(np.interp(q*c[-1], c - 0.5*w, v)) if c[-1] > 0. else np.nan


def composition(H, barT_grid=BART_GRID, f_spent=F_MAIN, min_steps=4):
  '''
  What the shell is made of, at each observer time.

  A step is ACTIVE at barT when barT lies in its arrival window [barT_on, barT_on + Tth],
  the same equal-arrival-time box cooling_frequency.effective_nu_c uses. Over the active
  steps, weighted by w:
    f_act   fraction of the weight carried by steps whose cell has not yet passed f_spent
            of its injected energy -- the light still coming from cells in the act of
            radiating it away
    frac_w  weighted mean of the cumulative radiated fraction itself (the continuous
            version of f_act, with no threshold in it at all)
    age_w   weighted mean age barT - barT_inj of the emitting cells
    ell_w   weighted mean of the emitting cells' own radiative length
    spread  weighted 10-90 percentile width of log10 nu_c, in dex
    sep     log10(nu_m/nu_c) taken as num_q50 - nuc_q10, i.e. the width of the segment
            over which that spread is smeared. nu_c is read at q10 rather than the median
            because the shell break sits at a LOW quantile of the population (this is the
            nuc_validation result, and it is the same population)
  Steps with no cooling break yet (`fresh`) are excluded from the frequency statistics,
  as they are from every estimator in cooling_frequency.
  '''
  on, Tth, w = H['barT'], H['Tth'], H['w']
  ok = np.isfinite(w) & (w > 0.)
  on, Tth, w = on[ok], Tth[ok], w[ok]
  off = on + Tth
  fr, b0 = H['frac'][ok], H['barT0'][ok]
  ell, fresh = H['ell90'][ok], H['fresh'][ok]
  lc, lm = np.log10(H['nu_c'][ok]), np.log10(H['nu_m'][ok])
  order = np.argsort(on)
  on_sorted = on[order]
  rows = []
  for T in barT_grid:
    a = np.zeros(len(on), dtype=bool)
    a[order[:np.searchsorted(on_sorted, T, 'right')]] = True
    a &= (off >= T)
    if a.sum() < min_steps:
      continue
    ww = w[a]
    W = ww.sum()
    # censored cells carry no ell, so they are dropped from ell_w and the weight is
    # renormalised over the rest -- averaging their (unknown, longest) length in as zero
    # would bias it short exactly where it is longest
    e = ell[a]
    fe = np.isfinite(e)
    d = dict(barT=float(T), nact=int(a.sum()),
             f_act=float(ww[fr[a] < f_spent].sum()/W),
             frac_w=float((ww*np.minimum(fr[a], 1.)).sum()/W),
             age_w=float((ww*(T - b0[a])).sum()/W),
             ell_w=float((ww[fe]*e[fe]).sum()/ww[fe].sum()) if fe.any() else np.nan)
    m = np.isfinite(lc[a]) & ~fresh[a]
    if m.sum() >= min_steps:
      lw, lv = ww[m], lc[a][m]
      q10, q50, q90 = (_wq(lv, lw, q) for q in (.1, .5, .9))
      d.update(nuc_q10=q10, nuc_q50=q50, nuc_q90=q90, spread=q90 - q10,
               num_q50=_wq(lm[a][m], lw, .5))
      d['sep'] = d['num_q50'] - q10
    rows.append(d)
  return rows


# ---------------------------------------------------------------------------
# csv i/o

def _write(rows, path, fields):
  with open(path, 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(fields), extrasaction='ignore')
    w.writeheader()
    w.writerows([{k: r.get(k, '') for k in fields} for r in rows])
  return path


def _read(path, fields, strkeys=()):
  if not os.path.isfile(path):
    return None
  out = []
  for r in csv.DictReader(open(path)):
    out.append({k: (r[k] if k in strkeys else (float(r[k]) if r[k] != '' else np.nan))
                for k in fields})
  return out


def measure(key=KEY, z=Z, log10ratio_arr=LOG10RATIO_ARR, outdir=None, verbose=True):
  '''Every sweep point: cell rows and composition rows, written beside the spectra.'''
  outdir = method_outdir(METHOD, key, z) if outdir is None else outdir
  cells, times = [], []
  for logr in log10ratio_arr:
    rows, H = harvest_point(logr, key=key, z=z, verbose=verbose)
    cells += rows
    for d in composition(H):
      d['logr'] = logr
      times.append(d)
  _write(cells, os.path.join(outdir, CELLS_CSV), CELL_FIELDS)
  _write(times, os.path.join(outdir, TIME_CSV), TIME_FIELDS)
  return cells, times


# ---------------------------------------------------------------------------
# figures

def _furniture(ax, barT_f, xoff=None, crossing=True):
  '''The shell's clock, drawn exactly as mid_slope_evolution draws it: the grey band is
  the rarefaction switching cells off, the dotted line is the shell crossing.'''
  if xoff is not None:
    ax.axvspan(xoff[0], xoff[1], color='grey', alpha=0.15, lw=0, zorder=0)
  if crossing:
    ax.axvline(1., color='grey', ls=':', lw=0.9, zorder=1)
  ax.grid(True, which='major', color=GRID, lw=0.6, alpha=0.9)
  ax.set_axisbelow(True)
  for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
  for sp in ('left', 'bottom'):
    ax.spines[sp].set_color(MUTED)
  ax.tick_params(colors=MUTED, labelsize=9)


def _by_logr(rows, key):
  '''{logr: (x, y)} sorted in x, NaNs kept out.'''
  out = {}
  for lr in sorted({r['logr'] for r in rows}):
    d = sorted([r for r in rows if r['logr'] == lr], key=lambda r: r[key[0]])
    x = np.array([r[key[0]] for r in d], float)
    y = np.array([r[key[1]] for r in d], float)
    m = np.isfinite(x) & np.isfinite(y)
    out[lr] = (x[m], y[m])
  return out


def plot_lengths(cells, outdir, barT_f, fname=FIG_LEN):
  '''
  The radiative length itself. LEFT: ell_f of every cell against the observer time at
  which that cell was shocked, so the x axis is a position along the shell (0 = first
  shocked, 1 = shell crossing) and not the emission clock. RIGHT: the same divided by
  gamma_c/gamma_m = 10**logr, which is the closed-form scaling -- four curves collapsing
  onto one is the statement that ell IS the cooling length of the gamma_m electrons.
  Censored cells (never reach f) are simply absent; the count is annotated.
  '''
  fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3))
  for ax in axes:
    _furniture(ax, barT_f)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('$\\bar{T}_{\\rm inj}/\\bar{T}_f$ (position along the shell)',
                  color=INK, fontsize=10)
  for f, ls, lw in ((F_MAIN, '-', 1.8), (min(FRACS), ':', 1.3)):
    for lr, (x, y) in _by_logr(cells, ('barT0', f'ell{f:g}')).items():
      m = x > 0.
      axes[0].plot(x[m]/barT_f, y[m], ls, color=COL[int(lr)], lw=lw, zorder=3,
                   label=f'{lr:+.0f}' if f == F_MAIN else None)
      axes[1].plot(x[m]/barT_f, y[m]/10**lr, ls, color=COL[int(lr)], lw=lw, zorder=3)
  axes[0].set_ylabel('$\\ell_f = R_f/R_{\\rm inj} - 1$', color=INK, fontsize=11)
  axes[1].set_ylabel('$\\ell_f\\,/\\,(\\gamma_c/\\gamma_m)$', color=INK, fontsize=11)
  axes[0].set_title(f'Radiative length, $f={F_MAIN:g}$ (solid) and $f={min(FRACS):g}$ '
                    '(dotted)', color=INK, fontsize=11, loc='left', pad=6)
  axes[1].set_title('divided by the one-zone scaling: one curve, not four',
                    color=INK, fontsize=11, loc='left', pad=6)
  cens = {lr: np.mean([not np.isfinite(r[f'ell{F_MAIN:g}'])
                       for r in cells if r['logr'] == lr])
          for lr in sorted({r['logr'] for r in cells})}
  axes[1].annotate('censored at $f=%g$: ' % F_MAIN
                   + ', '.join(f'{lr:+.0f}: {100*c:.0f}%' for lr, c in cens.items()),
                   xy=(0.02, 0.97), xycoords='axes fraction', ha='left', va='top',
                   fontsize=8, color=MUTED)
  leg = axes[0].legend(title='$\\log_{10}(\\gamma_c/\\gamma_m)$', fontsize=8.5,
                       title_fontsize=8.5, loc='upper left', frameon=True,
                       framealpha=0.92, edgecolor=GRID)
  leg.get_title().set_color(MUTED)
  for t in leg.get_texts():
    t.set_color(INK)
  fig.suptitle('How far a cell propagates before it has radiated its energy away',
               color=INK, fontsize=12.5, x=0.045, ha='left', y=0.99)
  fig.tight_layout(rect=[0, 0, 1, 0.94])
  path = os.path.join(outdir, fname)
  fig.savefig(path, dpi=200, facecolor='white')
  plt.close(fig)
  return path


def plot_link(times, mid_rows, outdir, barT_f, barT_off=None, fname=FIG_LINK):
  '''
  The two curves side by side on the shell's clock: the departure of the measured mid
  slope from its one-zone value (top, from mid_slope_evolution's cached rows -- settled
  cores only, which is the only estimator of a_mid worth a slope) and the emission-weighted
  radiative length of whatever is emitting at that time (bottom).

  Read the pair, not either alone. They rise together at log10(gc/gm) = -3, where ell is a
  per-cent of the radius; at -4 and -5 dep peaks early and is gone by the crossing while ell
  goes on growing, which is the whole content of "ell is necessary but not sufficient" in
  the FINDINGS block -- the smearing has to be read against the WIDTH of the segment it is
  smearing, and that width is not in ell.
  '''
  xoff = tuple(b/barT_f for b in barT_off) if (barT_off and barT_f > 0.) else None
  fig, axes = plt.subplots(2, 1, figsize=(8.4, 6.6), sharex=True)
  fc = [r for r in mid_rows if r['branch'] == 'fc' and np.isfinite(r['a_core'])]
  for ax in axes:
    _furniture(ax, barT_f, xoff)
    ax.set_xscale('log')
  axes[0].axhline(0., color=MUTED, lw=1.2, ls='--', zorder=1)
  for lr in sorted({r['logr'] for r in fc}):
    d = sorted([r for r in fc if r['logr'] == lr], key=lambda r: r['x'])
    if len(d) < 5:
      continue
    axes[0].plot([r['x'] for r in d], [r['a_core'] - r['a_th'] for r in d], '-',
                 color=COL[int(lr)], lw=1.8, zorder=3, label=f'{lr:+.0f}')
  for lr, (x, y) in _by_logr(times, ('barT', 'ell_w')).items():
    m = x > 0.
    axes[1].plot(x[m]/barT_f, y[m], '-', color=COL[int(lr)], lw=1.8, zorder=3,
                 label=f'{lr:+.0f}')
  axes[1].set_yscale('log')
  # everything left of BART_LO is under-resolved by construction (see SCOPE) and is
  # excluded from every number quoted in the docstring. Shaded last, so the span is drawn
  # over the data limits rather than setting them
  for ax in axes:
    x0 = ax.get_xlim()[0]
    ax.axvspan(x0, BART_LO/barT_f, color='grey', alpha=0.09, lw=0, zorder=0)
    ax.set_xlim(left=x0)
  axes[1].annotate('under-resolved', xy=(BART_LO/barT_f, 0.45),
                   xycoords=('data', 'axes fraction'), xytext=(-4, 0),
                   textcoords='offset points', ha='right', va='center',
                   fontsize=8, color=MUTED)
  axes[0].set_ylabel('$a_{\\rm mid} - 1/2$', color=INK, fontsize=11)
  axes[1].set_ylabel('$\\langle\\ell_{0.9}\\rangle_{\\rm emission}$', color=INK, fontsize=11)
  axes[0].set_title('departure of the measured mid slope from the one-zone value',
                    color=INK, fontsize=11, loc='left', pad=6)
  axes[1].set_title('radiative length of the cells emitting at that time',
                    color=INK, fontsize=11, loc='left', pad=6)
  axes[1].set_xlabel('$\\bar{T}/\\bar{T}_f$', color=INK, fontsize=10)
  for ax in axes:
    leg = ax.legend(title='$\\log_{10}(\\gamma_c/\\gamma_m)$', fontsize=8.5,
                    title_fontsize=8.5, ncol=2, loc='upper left', frameon=True,
                    framealpha=0.92, edgecolor=GRID)
    leg.get_title().set_color(MUTED)
    for t in leg.get_texts():
      t.set_color(INK)
  axes[1].annotate('crossing', xy=(1., 0.), xycoords=('data', 'axes fraction'),
                   xytext=(-4, 6), textcoords='offset points', ha='right', va='bottom',
                   fontsize=8.5, color=MUTED)
  fig.suptitle('Radiative length against the fast-cooling mid slope',
               color=INK, fontsize=12.5, x=0.055, ha='left', y=0.985)
  fig.tight_layout(rect=[0, 0, 1, 0.95])
  path = os.path.join(outdir, fname)
  fig.savefig(path, dpi=200, facecolor='white')
  plt.close(fig)
  return path


def summarise(cells, times, mid_rows, verbose=True):
  '''
  The table quoted in the module docstring: per sweep point, the median radiative length
  (radial and observer-time), its ratio to the one-zone scaling gamma_c/gamma_m, the
  censored fraction, and the a_mid departure the same point shows. Returns the rows.
  '''
  out = []
  for lr in sorted({r['logr'] for r in cells}):
    c = [r for r in cells if r['logr'] == lr]
    e9 = np.array([r[f'ell{F_MAIN:g}'] for r in c], float)
    e5 = np.array([r[f'ell{min(FRACS):g}'] for r in c], float)
    t9 = np.array([r[f'dT{F_MAIN:g}'] for r in c], float)
    dep = [r['a_core'] - r['a_th'] for r in mid_rows
           if r['branch'] == 'fc' and r['logr'] == lr and np.isfinite(r['a_core'])
           and r['x'] > BART_LO]
    out.append(dict(logr=lr, n=len(c), ell50=np.nanmedian(e5), ell90=np.nanmedian(e9),
                    ratio=np.nanmedian(e9)/10.**lr, dT90=np.nanmedian(t9),
                    cens=float(np.isnan(e9).mean()),
                    dep_med=float(np.median(dep)) if dep else np.nan,
                    dep_max=float(np.max(dep)) if dep else np.nan, n_dep=len(dep)))
  if verbose:
    print(f'{"logr":>5} {"ell_0.5":>10} {"ell_0.9":>10} {"/(gc/gm)":>9} {"dT_0.9":>10} '
          f'{"cens":>6} {"dep med":>8} {"dep max":>8}')
    for r in out:
      print(f'{r["logr"]:+5.0f} {r["ell50"]:10.2e} {r["ell90"]:10.2e} {r["ratio"]:9.1f} '
            f'{r["dT90"]:10.2e} {100*r["cens"]:5.0f}% '
            + (f'{r["dep_med"]:+8.3f} {r["dep_max"]:+8.3f}' if r['n_dep']
               else f'{"--":>8} {"--":>8}'))
  return out


def main(key=KEY, z=Z, log10ratio_arr=LOG10RATIO_ARR, outdir=None, use_cache=True):
  '''
  Measure (or reload) the radiative lengths of a cached sweep and draw both figures into
  that sweep's own directory, beside the spectra and beside mid_slope_evolution's rows.
  '''
  outdir = method_outdir(METHOD, key, z) if outdir is None else outdir
  os.makedirs(outdir, exist_ok=True)
  cells = _read(os.path.join(outdir, CELLS_CSV), CELL_FIELDS) if use_cache else None
  times = _read(os.path.join(outdir, TIME_CSV), TIME_FIELDS) if use_cache else None
  if cells and times:
    print(f'{len(cells)} cells + {len(times)} composition rows reloaded from {outdir}')
  else:
    print(f'--- measuring the radiative length on {key}, {METHOD}, z={z} ---')
    cells, times = measure(key, z, log10ratio_arr, outdir=outdir)
    print(f'{len(cells)} cells -> {CELLS_CSV}, {len(times)} rows -> {TIME_CSV}')
  mid_rows = read_rows(outdir)
  if mid_rows is None:
    raise RuntimeError(f'no {outdir}/mid_slopes.csv; run mid_slope_evolution.main first')
  barT_f = exit_onset_barT(key, z=z)
  p1 = plot_lengths(cells, outdir, barT_f)
  p2 = plot_link(times, mid_rows, outdir, barT_f, rarefaction_off_barT(key, z=z))
  summarise(cells, times, mid_rows)
  trim_pngs(outdir)
  copy_article_figures(outdir)
  print(f'-> {p1}\n-> {p2}')
  return cells, times


if __name__ == '__main__':
  main()
