# -*- coding: utf-8 -*-
# @Author: acharlet

'''
The TIME-INTEGRATED electron distribution, across cooling regimes.

Companion to cooling_shape_figure.py. That one draws the instantaneous N(gma,tt) as
the injected power law burns down from the top; this one draws what is left after the
whole dynamical time has been integrated over,

    N(gma; tilde{t}_dyn) = int_0^{tilde{t}_dyn} N(gma,tilde{t}) d tilde{t},

for a family of cooling regimes. Purely analytic -- no simulation data -- and built on
the same pipeline functions (cooling_distribution.gamma_synCooled /
.norm_plaw_distrib / .distrib_plaw_cooled), so it is a statement about the implemented
model rather than a redrawing of it.

WHY THIS IS THE QUANTITY THAT MATTERS. The time-integrated emission (the fluence) is
int dtt int dgma N(gma,tt) P(nu,gma); since the single-electron power P does not depend
on tt at fixed gma, the tt integral passes straight through onto N. So N(gma;tt_dyn) IS
the electron distribution the time-integrated spectrum sees -- its breaks are the
fluence's breaks. The instantaneous distribution never has the shape the integrated
spectrum reports, which is why reading one off the other fails.

THE REGIME AXIS. tt is normalized comoving time, tt = int dt'/t_{c,1}, so the cooling
time of a gma electron is 1/gma and the cooling Lorentz factor is simply

    gma_c = 1/tilde{t}_dyn

(exactly cooling_distribution.check_coolRegime_cell's gma_c = tc1/tdyn). One curve per
value of C = bar{gma}_c/gma_m (the article's `\\mathcal{C}`, spelled out in this
series' labels): fixing C fixes the integration
limit, tt_dyn = 1/(C*gma_m0). The colours are the sweep's, jet indexed on log10 C.

THE CLOSED FORM. Two routes, and they agree.

  (i) Straight integration. N(gma,tt) = K0 gma^-p (1-gma*tt)^(p-2) on the support
      [gma_m(tt), gma_M(tt)]; at fixed gma the support is a WINDOW IN TIME -- the
      population sweeps down past gma once -- running from when the bottom edge arrives,
      tt = 1/gma - 1/gma_m0, to when the top edge leaves, tt = 1/gma - 1/gma_M0, both
      clipped to [0, tt_dyn]. (1-gma*tt) is just gma/gma_0, so the integral is

        N(gma;tt_dyn) = K0 gma^-(p+1)/(p-1) * [A(gma)^(p-1) - B(gma)^(p-1)],
        A = min(1, gma/gma_m0),   B = max(1 - gma/gma_c, gma/gma_M0).

  (ii) Time-of-flight. dgma/dtt = -gma^2, so an electron passing through gma spends
      dtt = dgma/gma^2 there whatever it was injected as. Hence N(gma;tt_dyn) =
      gma^-2 * (number of electrons that reach gma within tt_dyn) -- and integrating
      K0 gma_0^-p over the injected range that qualifies returns exactly (i). That 1/gma^2
      is the whole origin of the fast-cooling gma^-2 segment: it is a dwell time, not a
      distribution.

THE SHAPE, and it is symmetric in the two regimes. The low edge sits at min(gma_c,gma_m),
the break at max(gma_c,gma_m), and above it every regime shares one cooled tail:

           gma < min      |   min .. max        |   gma > max
    FAST   nothing        |   gma^-2            |   gma^-(p+1)
    SLOW   nothing        |   gma^-p            |   gma^-(p+1)

so C only decides WHICH of gma_c, gma_m is the cutoff and which is the break, and what
the one segment between them does. The high tail is regime-independent. Note the slow
branch's middle is the INJECTED law: for C >> 1 the integral is just tt_dyn * N_0, the
population never cools within the dynamical time.

Two edges are not power laws and are drawn as they are, not idealized:
  - the TOP, gma -> gma_M0, where N -> 0 like [1 - (gma/gma_M0)^(p-1)]: the injected
    ceiling is occupied at the single instant tt = 0, so it carries no time weight.
  - the BOTTOM, a hard edge at gma_m(tt_dyn) = gma_m0/(1 + gma_m0*tt_dyn), which in fast
    cooling is ~gma_c and in slow cooling is just under gma_m0.

INTEGRATION LIMIT. SIGMA_END sets how many t_dyn the integral runs over; the default is
100, so the label reads N(gma; 10^2 tt_dyn). gma_c stays 1/tt_dyn -- the cooling Lorentz
factor AT the dynamical time, a property of the system rather than of the window -- so
its markers sit ON the curves rather than at their cut-offs once sigma_end > 1. The
panel's peak and its x floor both follow sigma_end; a fixed top clipped the
fastest-cooling curve at 100 t_dyn, whose peak is ~1e4.

VALIDATION, all four printed by main().

  - SUM RULE: int N(gma;tt_dyn) dgma = tt_dyn exactly, for every C -- cooling moves
    electrons, it never removes them, so the number integral is 1 at every instant and
    the time integral of that is the elapsed time. Holds to 3e-7 (trapezoid error).
  - AGAINST QUADRATURE: the closed form against direct quadrature of the pipeline's
    distrib_plaw_cooled, 7e-6 over 49 (C,gma) pairs. The quadrature must be handed the
    support WINDOW: over the full [0, tt_dyn] an adaptive rule misses the spike entirely
    and silently returns 0, because at gma >> gma_m0 the window is many decades narrower
    than tt_dyn. That failure returns a clean zero, not a warning.
  - TAIL UNIVERSALITY: N at gma = 1e5 is bit-identical across the five regimes whose
    break lies below it (spread 0.00e0), which is the tt_dyn-independence of the tail.
  - SEGMENT SLOPES, measured clear of the breaks. Two are exact and two are not, and
    the two residuals are physical, not numerical:
        fast gma_c..gma_m   -2.00000   dev  9e-16  -- exact: it is a dwell time
        tail (any C)        -3.50005   dev -5e-5   -- the gma_M rollover, 1.5*(gma/gma_M0)^(p-1)
        slow gma_m..gma_c   -2.50807   dev -8e-3   -- the approach to the asymptote
    The last one matters for reading real spectra: the slow-cooling middle is the
    injected -p only well below gma_c, since 1-(1-gma/gma_c)^(p-1) tilts the slope by
    ~(p-2)/2 * gma/gma_c. Measured at gma/gma_c = 0.0316 it is already 0.008 steep.

LOSSES: SYNCHROTRON ONLY. gma_synCooled solves dgma/dtt = -gma^2 and nothing else
(its own docstring: "only synchrotron, no adiabatic"). There is no inverse Compton term
here and none anywhere in project_v2, so that omission is the pipeline's, not this
figure's. The ADIABATIC term is the real difference from production: radiation_cooling
evolves the bounds along the actual syn+adiabatic trajectory and carries the
renormalisation K0 -> K0*A^(p-1) in its Aad column, worth -16% on eps_rad at slow
cooling. Two consequences, opposite in size:

  - The field is NOT assumed constant. tt = int dt'/t_{c,1} is an integral, and
    generate_cellDistrib recomputes t_{c,1} per step, so gma = gma0/(1+gma0*tt) is exact
    for ANY B'(t') as long as synchrotron is the only loss. Nothing here needs a
    steady field.
  - What it changes is the CUT-OFF, not the slope. With expansion the loss rate is
    -dgma/dtt = gma^2 + a*gma (a = -(1/3) dln rho/dtt), and the tempting reading -- that
    the dwell time dtt = dgma/(gma^2 + a*gma) flattens the tail toward gma^-1 below
    gma ~ a -- IS WRONG, as cooling_integrated_adiabatic computes: a and the cooled
    population fall at the same rate (1/tau), so a/gma is pinned at a constant and the
    index stays ~ -2 for every expansion law tested (-2.0 .. -1.87 over
    a_rho = -0.5 .. -2.9). The -2 drawn here is therefore NOT an upper bound on the
    steepness. Adiabatic cooling instead (i) suppresses the tail's amplitude toward
    b = 1 + a_rho/3 and (ii) removes the sharp low-energy cut-off altogether once the
    integration runs past ~t_dyn, because electrons keep sliding instead of stalling.
    The break positions and the -p, -(p+1) segments are untouched either way, since
    adiabatic cooling rescales gma uniformly and maps a power law to the same one.

VALIDITY. gma_synCooled is the ultra-relativistic solution and drives gma -> 0; below
gma = 1 (marked in red) it is not the physical trajectory, which is why the emission code
truncates its gamma integral there. On the fiducial bounds that floor is reached at
log10 C = -3 exactly -- gma_c = 1 -- so the fastest-cooling curve shown is the last one
the model can state, not an arbitrary stopping point.

Run:  python cooling_integrated_figure.py
'''

import os
import numpy as np
import matplotlib.pyplot as plt

from cooling_distribution import gamma_synCooled, norm_plaw_distrib, distrib_plaw_cooled

# --- defaults -------------------------------------------------------------------------
P_SYN = 2.5                 # phys_input.ini psyn
GM0, GMA_M0 = 1e3, 1e8      # injected bounds gma_m0, gma_M0 (fiducial, as cooling_shape)
# one curve per cooling regime, log10 C with C = gma_c/gma_m. Symmetric about the
# marginal case C = 1, and the fast end stops at -3 because that is where gma_c reaches
# the gma = 1 floor on these bounds (see VALIDITY above).
LOGC_SAMPLES = (-3., -2., -1., 0., 1., 2., 3.)
NG = 3000                   # points per curve (log-spaced over the support)
SIGMA_END = 100.            # integrate to this many t_dyn. gma_c stays 1/tt_dyn -- it is
                            # the cooling Lorentz factor AT the dynamical time, a property
                            # of the system, not of how long one chooses to integrate
from cooling_shape_figure import OUTDIR          # the shared family folder

# recessive ink for every non-data mark (text never wears a series colour)
INK, MUTED = '0.25', '0.55'

# sized for ONE column of a two-column article: panels stacked, ~3.4 in wide, so
# every mark and every font is set for that final printed size, not rescaled after
FIGSIZE = (3.4, 5.6)
FS_LAB, FS_TICK, FS_ANN, FS_LEG = 9., 8., 7.5, 7.

# the ratio is SPELLED OUT in this series, not abbreviated to the article's
# \mathcal{C} -- parenthesised because it sits inside the log
C_LABEL = '$\\log_{10}(\\bar{\\gamma}_{\\rm c}/\\gamma_{\\rm m,i})$'
# plain gamma, not gamma/gamma_m0: the shape figures' distribution panels are already
# in gamma, so this puts the whole cooling_distributions family on one abscissa
GMA_LABEL = '$\\gamma_{\\rm e}$'


# --- the distribution -----------------------------------------------------------------
def tt_dyn_of_C(logC, gm0=GM0):
  '''
  tt_dyn from the regime label: gma_c = C*gma_m0 and gma_c = 1/tt_dyn.
  '''
  return 1./(10.**logC*gm0)


def N_integrated(gma, ttd, p=P_SYN, gm0=GM0, gM0=GMA_M0):
  '''
  The closed form derived in the module docstring, at any gma (scalar or array).
  '''
  gma = np.asarray(gma, dtype=float)
  K0 = norm_plaw_distrib(gm0, gM0, p)
  A = np.minimum(1., gma/gm0)              # bottom edge has arrived (or was never above)
  B = np.maximum(1. - gma*ttd, gma/gM0)    # tt_dyn reached, or top edge has left
  return np.where(A > B, K0*gma**(-(p+1.))/(p-1.)*(A**(p-1.) - B**(p-1.)), 0.)


def integrated_distrib(logC, p=P_SYN, gm0=GM0, gM0=GMA_M0, Ng=NG, sigma_end=SIGMA_END):
  '''
  (gma, N(gma; sigma_end*tt_dyn)) over the support. Sampled from the lowest gma the
  population ever reaches up to the injected ceiling gma_M0; N vanishes at both ends by
  construction. N_integrated's second argument IS the integration limit, so the only
  change from integrating to one t_dyn is passing sigma_end*ttd.
  '''
  tt_end = sigma_end*tt_dyn_of_C(logC, gm0)
  gma = np.geomspace(gamma_synCooled(tt_end, gm0), gM0, Ng)
  return gma, N_integrated(gma, tt_end, p, gm0, gM0)


def log_slope(gma, N):
  '''
  d ln N / d ln gma, exact on any power-law segment. Masked where N = 0 (the two edges).
  '''
  ok = N > 0.
  s = np.full(gma.shape, np.nan)
  s[ok] = np.gradient(np.log(N[ok]), np.log(gma[ok]))
  return s


# --- validation -----------------------------------------------------------------------
def check_sum_rule(logC_arr, p=P_SYN, gm0=GM0, gM0=GMA_M0, Ng=400001):
  '''
  int N(gma;tt_dyn) dgma must be tt_dyn for every regime. Returns the largest relative
  deviation over logC_arr.
  '''
  dev = 0.
  for logC in logC_arr:
    gma, N = integrated_distrib(logC, p, gm0, gM0, Ng)
    ttd = tt_dyn_of_C(logC, gm0)
    dev = max(dev, abs(np.trapezoid(N, gma)/ttd - 1.))
  return dev


def check_against_quad(logC_arr, p=P_SYN, gm0=GM0, gM0=GMA_M0, n_gma=9):
  '''
  The closed form against direct quadrature of the pipeline's distrib_plaw_cooled,
  integrated over the TRUE support window in tt (see the docstring: handed [0, tt_dyn]
  instead, an adaptive rule misses the window and returns 0). Returns the largest
  relative deviation, and how many (C, gma) pairs were compared.
  '''
  from scipy.integrate import quad
  K0 = norm_plaw_distrib(gm0, gM0, p)
  dev, n = 0., 0
  for logC in logC_arr:
    ttd = tt_dyn_of_C(logC, gm0)
    for gma in np.geomspace(gamma_synCooled(ttd, gm0), gM0, n_gma)[1:-1]:
      ta = max(0., 1./gma - 1./gm0)          # bottom edge arrives
      tb = min(ttd, 1./gma - 1./gM0)         # top edge leaves, or time runs out
      if tb <= ta:
        continue
      num = quad(lambda t: K0*distrib_plaw_cooled(gma, p, t), ta, tb, limit=800)[0]
      dev = max(dev, abs(num/float(N_integrated(gma, ttd, p, gm0, gM0)) - 1.))
      n += 1
  return dev, n


def check_asymptotic_slopes(p=P_SYN, gm0=GM0, gM0=GMA_M0, h=.05):
  '''
  The docstring's segment table, MEASURED rather than asserted. Each slope is read at a
  point kept clear of both bracketing breaks -- a break contaminates its neighbourhood,
  and the gma_M rollover contaminates the top of the tail -- so the tail is sampled on
  the fast-cooling curves, where it has five clean decades. Returns
  [(label, gma, measured, expected)].
  '''
  def slope_at(gma, ttd):
    f = lambda g: np.log(float(N_integrated(g, ttd, p, gm0, gM0)))
    return (f(gma*10.**h) - f(gma*10.**-h))/(2.*h*np.log(10.))

  out = []
  ttd_f = tt_dyn_of_C(-3., gm0)                    # gma_c = 1, three decades below gma_m
  out.append(('fast   gma_c..gma_m', np.sqrt(gm0/ttd_f),
              slope_at(np.sqrt(gm0/ttd_f), ttd_f), -2.))
  ttd_s = tt_dyn_of_C(3., gm0)                     # gma_c = 1e6, three decades above
  out.append(('slow   gma_m..gma_c', np.sqrt(gm0/ttd_s),
              slope_at(np.sqrt(gm0/ttd_s), ttd_s), -p))
  for lc in (-3., -1.):                            # the shared tail, two decades clear
    ttd = tt_dyn_of_C(lc, gm0)
    out.append((f'tail   C=1e{lc:+.0f}     ', 1e2*gm0, slope_at(1e2*gm0, ttd), -(p+1.)))
  return out


def check_tail_universality(logC_arr=None, gma=1e5, p=P_SYN, gm0=GM0, gM0=GMA_M0):
  '''
  Above max(gma_c, gma_m) the closed form loses every trace of tt_dyn, so the cooled tail
  should be the SAME curve in every regime. Returns the relative spread of N at `gma`
  over the regimes whose break sits below it, and how many qualified.
  '''
  logC_arr = LOGC_SAMPLES if logC_arr is None else logC_arr
  vals = [float(N_integrated(gma, tt_dyn_of_C(lc, gm0), p, gm0, gM0))
          for lc in logC_arr if 10.**lc*gm0 < gma]
  return (max(vals)/min(vals) - 1.), len(vals)


# --- the figure -----------------------------------------------------------------------
def plot_integrated(p=P_SYN, gm0=GM0, gM0=GMA_M0, logC=LOGC_SAMPLES, sigma_end=SIGMA_END,
    outdir=OUTDIR, fname='cooling_integrated.png', show=False):
  '''
  Two-panel view: the time-integrated distributions over the regime family, and the
  local slope underneath, which is where the gma^-2 / gma^-p / gma^-(p+1) segments and
  the break positions can actually be read off.
  '''
  logC = np.asarray(logC, dtype=float)
  norm = plt.Normalize(vmin=logC.min(), vmax=logC.max())
  colors = plt.cm.jet(norm(logC))
  sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=norm)

  fig, (axN, axS) = plt.subplots(2, 1, figsize=FIGSIZE, sharex=True,
      gridspec_kw=dict(height_ratios=[1.9, 1.], hspace=.08))

  # slowest cooling first, so the fast-cooling curves end up on top (sweep _draw_order).
  # The fast branch is DEGENERATE -- every C < 1 shares one envelope and differs only in
  # where it is cut off -- so the overdraw hides nothing that is not identical.
  order = np.argsort(logC)[::-1]
  breaks, hi_N = [], 0.   # hi_N follows the peak; a fixed top clipped it at 100 t_dyn
  for i in order:
    gma, N = integrated_distrib(logC[i], p, gm0, gM0, sigma_end=sigma_end)
    hi_N = max(hi_N, float(N.max()))
    axN.loglog(gma, N, color=colors[i], lw=1.2, solid_capstyle='round', zorder=3)
    axS.semilogx(gma, log_slope(gma, N), color=colors[i], lw=1.1, zorder=3)
    # gma_c on its own curve, evaluated exactly rather than read off the sampling: the
    # low CUT-OFF when C < 1, the BREAK when C > 1 -- one symbol carrying both roles
    gma_c = 10.**logC[i]*gm0
    breaks.append((gma_c, float(N_integrated(gma_c, sigma_end/gma_c, p, gm0, gM0))))
  breaks = np.array(breaks)
  axN.scatter(breaks[:, 0], breaks[:, 1], s=11, facecolors=colors[order],
              edgecolors='w', linewidths=.5, zorder=6)

  # (a) the distributions ------------------------------------------------------------
  # no power-law guides here: the slope panel states the three indices quantitatively,
  # and dotted guides over these curves only collide with them
  _t = '' if sigma_end == 1. else f'10^{{{np.log10(sigma_end):.0f}}}'
  axN.set_ylabel("$N_{\\rm e}^{-1}\\,{\\rm d}N/{\\rm d}\\gamma_{\\rm e}"
                 + f'\\;({_t}\\tilde{{t}}_{{\\rm dyn}})$',
                 fontsize=FS_LAB)
  axN.set_ylim(1e-25, 10.*hi_N)   # follows the peak: at 100 t_dyn a fixed top
                                  # clipped the fastest-cooling curve

  # (b) the slopes ---------------------------------------------------------------------
  # the expected indices are reference VALUES, so they belong on an axis: the guides stay
  # inside, their labels go on the right-hand spine as ticks. Inside the panel they had to
  # dodge the curves -- -2 and -p are only half an index apart and were hung on opposite
  # sides of their own lines -- and on the spine they simply line up.
  levels = ((-2., '$-2$'), (-p, '$-p$'), (-(p+1.), '$-(p+1)$'))
  for lev, _ in levels:
    axS.axhline(lev, color=MUTED, ls='--', lw=.7, zorder=1)
  # no positive plateau here: synchrotron alone cuts off sharply, so the only thing
  # above zero is the edge spike and the panel is clipped back to it
  axS.set_ylim(-(p+2.6), .4)
  axS.set_xlabel(GMA_LABEL, fontsize=FS_LAB)
  axS.set_ylabel("${\\rm d}\\ln({\\rm d}N/{\\rm d}\\gamma_{\\rm e})"
                 "/{\\rm d}\\ln\\gamma_{\\rm e}$", fontsize=FS_LAB)
  axR = axS.twinx()                       # right-hand spine carries the expected indices
  axR.set_ylim(axS.get_ylim())
  axR.set_yticks([lev for lev, _ in levels])
  axR.set_yticklabels([lab for _, lab in levels])
  axR.tick_params(axis='y', labelsize=FS_ANN, length=2.5, pad=1.5, colors=INK)
  axR.grid(False)

  # marks shared by both panels: the injected bottom edge sits at x = 1 by construction,
  # the injected ceiling closes every curve, and gma = 1 is where the model stops
  for ax in (axN, axS):
    ax.axvspan(1e-30, 1., color='crimson', alpha=.07, lw=0, zorder=0)        # gma < 1: the model's floor, shaded in every
    ax.axvline(gm0, color=INK, ls=':', lw=.8, zorder=1)
    ax.axvline(gM0, color=INK, ls=':', lw=.8, zorder=1)
    ax.axvline(1., color='crimson', ls=':', lw=.9, zorder=1)
    ax.grid(alpha=.25, lw=.4)
    ax.tick_params(which='both', labelsize=FS_TICK)
  axN.set_xlim(.3*float(gamma_synCooled(sigma_end*tt_dyn_of_C(min(logC), gm0), gm0)),
               4.*gM0)
  # the injected bounds are labelled along the top of (a); gma = 1 cannot go there --
  # the fastest-cooling curve peaks in that corner -- so it is labelled in (b) instead,
  # where the bottom left is empty
  for v, lab in ((gm0, '$\\gamma_\\mathrm{m}$'), (gM0, '$\\gamma_\\mathrm{M}$')):
    axN.annotate(lab, (v, .985), xycoords=('data', 'axes fraction'), color=INK,
                 fontsize=FS_ANN, ha='center', va='top',
                 bbox=dict(fc='w', ec='none', alpha=.85, pad=1.))
  axS.annotate('$\\gamma=1$', (1., .10), xycoords=('data', 'axes fraction'),
               textcoords='offset points', xytext=(3, 0), color='crimson',
               fontsize=FS_ANN, ha='left', va='bottom',
               bbox=dict(fc='w', ec='none', alpha=.85, pad=1.))
  # the marker legend goes in the panel, the regime axis is the colour bar
  axN.scatter([], [], s=11, facecolors='none', edgecolors=INK, linewidths=.7,
              label='$\\gamma_\\mathrm{c}$')
  # right of top centre: the curves all run upper-left to lower-right, so this is the
  # one patch of the panel no line crosses, and it clears the gma_m / gma_M top labels
  # (which sit at x ~ 0.39-0.42 and ~ 0.93) because the box spans about 0.55-0.85
  axN.legend(fontsize=FS_LEG, loc='upper center', bbox_to_anchor=(.70, .95),
             framealpha=.9, handletextpad=.2, borderpad=.4)

  # the bar moves UP beside the top panel only: the slope panel's right-hand side now
  # carries the expected-index ticks. An EXPLICIT cax, not ax=axN -- stealing space from
  # one of two stacked shared-x panels leaves them different widths and breaks the
  # alignment the pair is read on.
  fig.subplots_adjust(right=.85)
  pos = axN.get_position()
  cax = fig.add_axes([.875, pos.y0, .032, pos.height])
  cb = fig.colorbar(sm, cax=cax)
  cb.set_label(C_LABEL, fontsize=FS_LAB)
  cax.tick_params(labelsize=FS_TICK)

  os.makedirs(outdir, exist_ok=True)
  path = os.path.join(outdir, fname)
  fig.savefig(path, dpi=300, bbox_inches='tight')
  print(f'saved {path}')
  if show:
    plt.show()
  return fig, (axN, axS)


def main(show=False):
  print(f'sum rule      : max |int N dgma / tt_dyn - 1| = '
        f'{check_sum_rule(LOGC_SAMPLES):.2e}')
  dev, n = check_against_quad(LOGC_SAMPLES)
  print(f'vs quadrature : max rel dev over {n} (C,gma) pairs = {dev:.2e}')
  spread, nc = check_tail_universality()
  print(f'tail universal: spread of N at gma=1e5 over {nc} regimes = {spread:.2e}')
  print('segment slopes (measured clear of the breaks):')
  for lab, gma, meas, exp in check_asymptotic_slopes():
    print(f'  {lab} at gma={gma:9.3e}: {meas:+.5f}  (expected {exp:+.3f}, '
          f'dev {meas-exp:+.1e})')
  for lc in (LOGC_SAMPLES[0], 0., LOGC_SAMPLES[-1]):
    ttd = tt_dyn_of_C(lc)
    print(f'  log10 C = {lc:+.0f}: tt_dyn = {ttd:.3e}, gma_c = {1./ttd:.3e}, '
          f'low edge gma_m(tt_dyn) = {gamma_synCooled(ttd, GM0):.3e}')
  plot_integrated(show=show)


if __name__ == '__main__':
  main()
