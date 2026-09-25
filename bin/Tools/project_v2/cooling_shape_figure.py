# -*- coding: utf-8 -*-
# @Author: acharlet

'''
How a power-law electron distribution is reshaped by synchrotron cooling.

Base module of the four that share figures/cooling_distributions (it defines OUTDIR for
all of them): {cooling_shape_figure, cooling_integrated_figure} x {synchrotron only,
+ adiabatic}, the adiabatic halves being cooling_shape_adiabatic and
cooling_integrated_adiabatic. This one is the instantaneous, synchrotron-only corner.

Purely analytic figure -- no simulation data. It draws the exact solution that the
whole cooling pipeline is built on, using the very functions the pipeline calls
(cooling_distribution.gamma_synCooled / .distrib_plaw_cooled), so the picture is a
statement about the implemented model, not a redrawing of it.

The model. An electron injected with Lorentz factor gma0 and cooling by synchrotron
alone follows gma(tt) = gma0/(1 + gma0*tt), with the NORMALIZED TIME

    tt = tilde{t} = int dt'/t_{c,1}

(t_{c,1} = comoving cooling time of a gma = 1 electron). Number conservation,
N(gma,tt) dgma = N0(gma0) dgma0, then turns the injected power law
N0 = K0 gma0^-p on [gma_m0, gma_M0] into

    N(gma,tt) = K0 gma^-p (1 - gma*tt)^(p-2)   on  [gma_m(tt), gma_M(tt)]

with both edges cooling by the same law. The two panels split the story in time:

(a) The edges and the break versus tt, i.e. the time axis panel (b) samples. The two
    knees are at tilde{t}_M = 1/gma_M0 (the top edge starts burning) and
    tilde{t}_m = 1/gma_m0 (the bottom edge follows, the power law is gone) -- each the
    cooling time of the edge it takes down.

(b) The support burns down from the top. gma_M(tt) -> 1/tt for tt >> 1/gma_M0, so the
    cooling break 1/tt is an ASYMPTOTE the top edge slides down along, never a place
    where the distribution bends: the injected edge stays a sharp edge, only curled
    just below it by the factor (1 - gma*tt)^(p-2). Below the edge the gma^-p segment
    is untouched until tt reaches 1/gma_m0. Past that the bottom edge follows the top
    one down and the width collapses --

        gma_M/gma_m = (gma_M0/gma_m0) (1 + gma_m0 tt)/(1 + gma_M0 tt)  ->  1,

    so the distribution ends up MONO-ENERGETIC at gma ~ 1/tt, whatever it was injected
    as. That is the last curves of the figure: on the fiducial bounds (gma_m0 = 1e3,
    gma_M0 = 1e8) five injected decades are down to a 10%-wide spike at gma = 100 by
    tt = 1e-2, and 0.1% wide by tt = 1.

Number is conserved exactly at every tt (checked by check_number_conservation) -- the
distribution loses energy, not electrons.

Validity. gma_synCooled is the ultra-relativistic solution and drives gma -> 0; below
gma = 1 (marked in red in both panels) it is no longer the physical trajectory, which
is why the emission code truncates its gamma integral there. The TRACKS are still drawn
below it, as the integrated figures draw their curves into the same region -- the red
line is a caveat on the curve, not a reason to hide where the model points
(radiation_cooling.get_epnu:
number is conserved, so the clipped bound is the whole correction). The last
sampled time, tt = 1, lands exactly on that line: gma_M = 1/(1 + 1/gma_M0) -> 1, so on
these bounds tt = 1 IS the end of the model's validity, not a time it can be pushed
past.

Run:  python cooling_shape_figure.py
'''

import os
import numpy as np
import matplotlib.colors as mcolors

from environment import GAMMA_dir
from cooling_distribution import gamma_synCooled, norm_plaw_distrib, distrib_plaw_cooled

# --- defaults -------------------------------------------------------------------------
P_SYN = 2.5                 # phys_input.ini psyn
GM0, GMA_M0 = 1e3, 1e8      # injected bounds gma_m0, gma_M0 (fiducial case)
# sampled normalized times as log10(tt): from the top edge just starting to burn
# (tt = 1/gma_M0 = 1e-8) through the mono-energetic limit (tt >> 1/gma_m0 = 1e-3) and
# on to tt = 1, where the population has cooled all the way onto gma = 1
LOGTT_SAMPLES = (-8., -7., -6., -5., -4., -3., -2., -1., 0.)
NG = 800                    # points per curve
# ONE folder for the whole analytic-distribution family: the instantaneous shape and the
# time-integrated distribution, each with and without adiabatic cooling. Defined here, in
# the base module, and imported by the other three so the four cannot drift apart.
OUTDIR = os.path.join(GAMMA_dir, 'bin', 'Tools', 'figures', 'cooling_distributions')

MFC_FAC = 3.                 # MFC is taken as t_m/MFC_FAC .. t_m*MFC_FAC
BAND_ALPHA = .13            # tint of the regime bands

# recessive ink for every non-data mark (text never wears a series colour)
INK, MUTED = '0.25', '0.55'

# sized for ONE column of a two-column article: panels stacked, ~3.4 in wide, so
# every mark and every font is set for that final printed size, not rescaled after
FIGSIZE = (3.4, 5.0)
FS_LAB, FS_TICK, FS_ANN, FS_LEG = 9., 8., 8., 7.


def _band_bg(col, alpha=BAND_ALPHA):
  'Opaque colour a band tint blends to over white, so a label box can match its band.'
  return tuple(alpha*c + (1. - alpha) for c in mcolors.to_rgb(col))


# --- the distribution -----------------------------------------------------------------
def cooled_distrib(tt, p=P_SYN, gm0=GM0, gM0=GMA_M0, Ng=NG):
  '''
  (gma, N(gma,tt)) sampled over the SUPPORT [gma_m(tt), gma_M(tt)] only.
  distrib_plaw_cooled stays positive all the way up to 1/tt, but the electrons
  between gma_M(tt) and 1/tt do not exist -- nothing was injected above gma_M0.
  '''
  K0 = norm_plaw_distrib(gm0, gM0, p)
  gm, gM = gamma_synCooled(tt, gm0), gamma_synCooled(tt, gM0)
  gma = np.geomspace(gm, gM, Ng)
  return gma, K0*distrib_plaw_cooled(gma, p, tt)


def check_number_conservation(tt_arr, p=P_SYN, gm0=GM0, gM0=GMA_M0, Ng=20000):
  '''
  int N dgma must stay 1 for every tt: cooling moves electrons, it does not remove
  them. Returns the largest relative deviation over tt_arr.
  '''
  dev = 0.
  for tt in tt_arr:
    gma, N = cooled_distrib(tt, p, gm0, gM0, Ng)
    dev = max(dev, abs(np.trapezoid(N, gma) - 1.))
  return dev


def main():
  tt_arr = np.r_[0., 10.**np.asarray(LOGTT_SAMPLES)]
  print(f'number conservation: max |int N dgma - 1| = '
        f'{check_number_conservation(tt_arr):.2e}')
  gm, gM = gamma_synCooled(tt_arr[-1], GM0), gamma_synCooled(tt_arr[-1], GMA_M0)
  print(f'last sample tt=10^{LOGTT_SAMPLES[-1]:g}: '
        f'gma_m={gm:.3f}, gma_M={gM:.3f}, width gma_M/gma_m={gM/gm:.4f}')


if __name__ == '__main__':
  main()
