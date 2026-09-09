# -*- coding: utf-8 -*-
# @Author: acharlet

'''
The Ravasio et al. (2018) 2SBPL against the Granot & Sari shape, on the rarcut sweep.

WHAT THE 2SBPL IS. Ravasio et al. (2018, A&A 613, A16) fit GRB 160625B, the third-brightest
Fermi burst, and find that no single-break model works: the residuals of a Band/SBPL fit carry
a systematic excess near 60 keV. Their fix is a THIRD power-law segment below the peak, joined
by a second smooth break -- eq. (3), the "double smoothly broken power law". The photon indices
they recover, <alpha1> = -0.63 and <alpha2> = -1.48, are the synchrotron fast-cooling values
-2/3 and -3/2, and the break sits at ~100 keV with E_peak/E_break ~ 5-35, i.e. marginally fast
cooling. It is the same physical spectrum this project computes, fitted with a different
algebraic form -- which is the whole reason to test it here.

HOW IT DIFFERS FROM granot_sari_syn (the derivation is in phys_functions.two_sbpl):
  SAME  three segments, two smooth breaks, the same smoothing convention (larger = sharper),
        the same asymptotes once alpha1 = -2/3 and beta = -p/2 - 1 are held.
  SAME, ALGEBRAICALLY, IN THE SEPARATED LIMIT. Writing the lower SBPL as L and the high power
        law anchored on it as P, eq. (3) is [L**-n2 + P**-n2]**(-1/n2), the smooth MINIMUM of
        the two -- a NESTED broken power law. granot_sari_syn writes the upper break as a
        multiplicative correction to L instead. The two coincide wherever L has reached its
        mid asymptote by the upper break, and differ only through the lower break's residual
        curvature there: 0.026 dex maximum at 1 decade of separation, 0.002 at 2, 0.0002 at 3
        (s1 = 1.3, s2 = 2.0; smaller still for a sharper lower break). See shape_difference.
  DIFFERENT  the PARAMETERISATION. GS02 is written on the two crossing frequencies; the 2SBPL
        is written on E_break (a crossing) and E_peak (the nuFnu turnover), converted by its
        eq. (4). That is a bijection at fixed (alpha2, beta, n2), so it cannot change the
        fitted shape -- only what gets quoted. E_peak/E_break is the ratio the GRB literature
        reports and is NOT b_hi/b_lo: at the synchrotron values E_j = 0.709 E_peak.
  DIFFERENT  the mid slope is a free photon index there, whereas granot_sari_syn pins it from
        the ORDERING of nu_c and nu_m unless beta_mid is passed.
  DIFFERENT  the SMOOTHING PRESCRIPTION. Ravasio hold n2 = 2.69 (the GBM catalogue's SBPL
        curvature, Lambda = 0.3) and n1 = 5.38 (their own free-fit mean), so the low break is
        SHARPER than the peak. This project holds s1 = 1.3, s2 = 2.0 -- the opposite ordering,
        and both far smoother. That contrast is what the sweep test below is really about.
  DIFFERENT  no high-frequency cut-off. The 2SBPL runs as a power law above the peak (Ravasio
        add an exponential only to accommodate LAT data). Our spectra cut off at the burnoff
        frequency nu_M inside the band, so both shapes are fitted here to the SAME
        cut-off-flattened spectrum (spectral_breaks.measure_cutoff_nuM with the smeared shape,
        then everything below nu_M/CUT_FAC) -- which removes the roll-off from the comparison
        entirely and leaves only the three-segment body.

WHAT IS COMPARED. Both shapes are fitted to the same cut-off-flattened spectra of the rarcut
sweep (method 'data_rarcut', both shells), at LOGT epochs of each point, with the two OUTER
slopes held at the asymptotes this project has verified (4/3 and 1 - p/2 in nuFnu). Three
matched configurations, described in full on compare_spectrum:
    'anchored'  breaks and mid slope HELD at the segment route's measurements, so smoothing is
            the only shape freedom left. THE comparison -- and the GS02 side of it is literally
            spectral_breaks.fit_smoothing_held, i.e. the shipped route, not a re-implementation.
    'presc'     the same with each shape's published smoothing pair frozen: what a tabulated
            (1.3, 2.0) or (5.38, 2.69) costs.
    'free'      six free parameters each, no anchors. Its rms is the best either family can do
            with equal freedom; its s and a_mid are NOT measurements (the break/smoothing
            degeneracy documented in the spectral_breaks header), and are labelled so.

Example use in command line:
  python -c "import ravasio_2sbpl as R; R.main()"
  python -c "import ravasio_2sbpl as R; R.shape_difference()"
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

import cell_pool
import sweep_gammacm as swp
import spectral_breaks as sb
from phys_functions import (granot_sari_syn, two_sbpl, two_sbpl_Ej, two_sbpl_Epeak,
    RAVASIO_N1, RAVASIO_N2)
from plotting_functions import COL_RS, COL_FS

GAMMA_dir = swp.GAMMA_dir
OUTDIR = os.path.join(GAMMA_dir, 'bin', 'Tools', 'figures', 'ravasio_2sbpl')
KEY = 'cooling_g100'
METHOD = 'data_rarcut'          # THE RARCUT SWEEP: cells truncated at the modelled R_rar
Z_RS, Z_FS = 4, 1

# Epochs fitted per sweep point, in log10(bar{T}/bar{T}_f) -- the same clock as
# sweep_gammacm.SPEC_LOGT (-3..2), at twice its resolution. -3 is deep on the rise, 0 the
# crossing, +2 far into the high-latitude tail.
LOGT = np.arange(-3., 2. + 1e-9, 0.5)

# GS02's published smoothing for this project vs Ravasio's for GRB 160625B. Same convention
# (larger = sharper), so the pairs are directly comparable -- and they are opposite in
# ORDERING as well as ~3x apart on the low break.
S_GS02 = (swp.GS02_S1, swp.GS02_S2)
S_2SBPL = (RAVASIO_N1, RAVASIO_N2)

S_BOUNDS = sb.S_FIT_BOUNDS      # (0.15, 10.); both shapes' smoothings live in the same one
BMID_MARGIN = sb.BMID_MARGIN    # a free mid slope stays strictly between the outer asymptotes
BMID_DEP = sb.BMID_DEP          # ... or, bmid_physical, inside the one-zone interval widened
                                # by the departure shell integration actually produces
CUT_FAC = sb.CUT_FAC            # fit below nu_M/CUT_FAC even after flattening
FIT_DEC = sb.FIT_DEC            # ... and over the top this many decades

# Leave the laptop usable: resolve_nproc would take 7 of 8 cores here. Points are
# independent and each holds one sweep point's arrays (~8 MB), so 3 workers is ~0.2 GB.
NPROC = 3

SHELL = {4: 'RS', 1: 'FS'}
SHELL_COL = {4: COL_RS, 1: COL_FS}
CONFIGS = ('anchored', 'presc', 'free')
SHAPES = ('gs02', '2sbpl')


# ---------------------------------------------------------------------------
# how far apart the two shapes actually are, at matched parameters
# ---------------------------------------------------------------------------
def shape_difference(seps=(0.5, 1., 1.5, 2., 3., 4.), s_pairs=(S_GS02, S_2SBPL),
    psyn=2.5, fast=True, verbose=True):
  '''
  Maximum |log10| difference between granot_sari_syn and two_sbpl with EVERY parameter
  matched -- b_lo = E_break, b_hi = E_j, s1 = n1, s2 = n2, the same three slopes -- as a
  function of the break separation. This is the whole functional difference between the two
  forms, isolated from any fit: it is the lower break's curvature still present at the upper
  crossing, so it dies as the breaks separate and is worse for a SMOOTHER lower break.
  fast: mid slope at the fast-cooling asymptote (-1/2 in F_nu); False takes the slow one.
  Returns a DataFrame (sep, s1, s2, dmax).
  '''
  a1, b = -2/3., -psyn/2. - 1.
  a2 = (-1.5 if fast else -(psyn - 1.)/2. - 1.)
  E = np.logspace(-6., 8., 60001)
  rows = []
  for sep in seps:
    for s1, s2 in s_pairs:
      b_hi = 10**sep
      g = granot_sari_syn(E, b_hi, 1., psyn, s1=s1, s2=s2, nuM=None, F_ext=1., nuFnu=True,
                          beta_mid=a2 + 1.)
      t = two_sbpl(E, 1., None, a1, a2, b, n1=s1, n2=s2, nuFnu=True, E_j=b_hi)
      with np.errstate(divide='ignore', invalid='ignore'):
        d = np.log10(t/g)
      d = d - d[np.argmin(np.abs(E - 1e-4))]      # both are normalised well below b_lo
      m = (E > 1e-4) & (E < b_hi*1e4)
      rows.append(dict(sep=sep, s1=s1, s2=s2, dmax=float(np.nanmax(np.abs(d[m])))))
  df = pd.DataFrame(rows)
  if verbose:
    print(f'granot_sari_syn vs two_sbpl at matched parameters, p = {psyn}, '
          f'mid slope {"fast" if fast else "slow"}')
    print(df.pivot(index='sep', columns=['s1', 's2'], values='dmax').to_string(
        float_format=lambda v: f'{v:.5f}'))
  return df


# ---------------------------------------------------------------------------
# one spectrum: prepare it once, fit both shapes on it
# ---------------------------------------------------------------------------
def prepare_spectrum(x, sp, psyn, cutfac=CUT_FAC, fit_dec=FIT_DEC):
  '''
  The cut-off-flattened body of one nuFnu spectrum, shared by both fits so neither can win on
  a different sample. nu_M and its dex spread come from the smeared roll-off scan
  (spectral_breaks.measure_cutoff_nuM(smear=True)), the flattened spectrum is cut at
  nu_M/cutfac -- the roll-off is divided out, but the last factor of a few is still where the
  division is least trustworthy -- and only the top fit_dec decades are kept.
  Returns dict(x, ly, y0, nuM, sigma, ok); ly is log10(flux) with its own maximum removed, so
  the scale is a pure fit parameter and rms values are comparable across epochs.
  '''
  x = np.asarray(x, float); sp = np.asarray(sp, float)
  out = dict(x=None, ly=None, y0=np.nan, nuM=np.nan, sigma=np.nan, ok=False)
  g = np.isfinite(sp) & (sp > 0.) & np.isfinite(x) & (x > 0.)
  if g.sum() < 12:
    return out
  cut = sb.measure_cutoff_nuM(x[g], sp[g], psyn, flatten=True, smear=True)
  if not cut['ok']:
    return out
  out['nuM'], out['sigma'] = cut['nuM'], cut['sigma']
  xg, flat = x[g], cut['sp_flat']
  m = np.isfinite(flat) & (flat > 0.) & (xg < cut['nuM']/cutfac)
  if m.sum() < 12:
    return out
  xf, yf = xg[m], np.log10(flat[m])
  keep = yf > yf.max() - fit_dec
  xf, yf = xf[keep], yf[keep]
  if len(xf) < 12:
    return out
  y0 = float(yf.max())
  out.update(x=xf, ly=yf - y0, y0=y0, ok=True)
  return out


def _blank_fit(shape):
  return dict(shape=shape, rms=np.nan, s1=np.nan, s2=np.nan, b_lo=np.nan, b_hi=np.nan,
              sep=np.nan, a_mid=np.nan, x_pk=np.nan, E_peak=np.nan, npts=0,
              at_bound=True, ok=False)


def _bmid_bounds(psyn, margin=BMID_MARGIN):
  '''Bounds on the mid slope in F_nu index: strictly between the two outer asymptotes
  (1/3 and -p/2 in F_nu), with a margin so the fit cannot collapse to a single break.
  The same interval spectral_breaks.fit_smoothing_held uses, so the two mid slopes are
  measured against the same limits.'''
  return -psyn/2. + margin, 1/3. - margin


def _report(shape, xf, model, rms, s1, s2, b_lo, b_hi, bmid, at_bound, psyn):
  '''Common output row for either shape. x_pk is the model's own nuFnu peak on the fitted
  grid, i.e. the one quantity the two parameterisations name differently (GS02 has no
  E_peak parameter, the 2SBPL has no b_hi one) put on a common footing.'''
  d = dict(_blank_fit(shape))
  lm = np.log10(np.maximum(model, 1e-300))
  d.update(rms=float(rms), s1=float(s1), s2=float(s2), b_lo=float(b_lo), b_hi=float(b_hi),
           sep=float(np.log10(b_hi/b_lo)), a_mid=float(bmid) + 1., npts=int(len(xf)),
           x_pk=float(xf[int(np.argmax(lm))]), at_bound=bool(at_bound), ok=not bool(at_bound))
  d['E_peak'] = two_sbpl_Epeak(b_hi, bmid - 1., -psyn/2. - 1., s2)
  return d


def fit_gs02_flat(prep, psyn, s_hold=None, s_bounds=S_BOUNDS):
  '''
  granot_sari_syn fitted to a prepared (flattened) spectrum, nuM=None -- the roll-off is
  already divided out. Free: b_lo, b_hi (as b_lo x 10**dsep, dsep >= 0, so the ordering
  cannot invert), the mid F_nu slope, the scale, and s1, s2 unless s_hold freezes them.
  The two outer slopes are HELD at 1/3 and -p/2 in F_nu (4/3 and 1 - p/2 in nuFnu).
  '''
  if not prep['ok']:
    return _blank_fit('gs02')
  xf, yf = prep['x'], prep['ly']
  lo_b, hi_b = np.log10(xf.min()) - 0.5, np.log10(xf.max()) + 0.5
  bm_lo, bm_hi = _bmid_bounds(psyn)
  lsb = (np.log10(s_bounds[0]), np.log10(s_bounds[1]))
  free_s = s_hold is None

  def unpack(q):
    b_lo = 10**q[0]; b_hi = b_lo*10**q[1]
    s1, s2 = (10**q[4], 10**q[5]) if free_s else s_hold
    return b_lo, b_hi, 10**q[2], q[3], s1, s2

  def resid(q):
    b_lo, b_hi, A, bmid, s1, s2 = unpack(q)
    m = granot_sari_syn(xf, b_hi, b_lo, psyn, s1=s1, s2=s2, nuM=None, F_ext=A,
                        nuFnu=True, beta_mid=bmid)
    return np.log10(np.maximum(m, 1e-300)) - yf

  x_pk = xf[int(np.argmax(yf))]
  q0 = [np.log10(x_pk) - 1.5, 1.5, 0., 0.5*(bm_lo + bm_hi)]
  blo = [lo_b, 0., -8., bm_lo]
  bhi = [hi_b, hi_b - lo_b, 8., bm_hi]
  if free_s:
    q0 += [np.log10(S_GS02[0]), np.log10(S_GS02[1])]
    blo += [lsb[0], lsb[0]]; bhi += [lsb[1], lsb[1]]
  try:
    r = least_squares(resid, q0, bounds=(blo, bhi))
  except ValueError:
    return _blank_fit('gs02')
  b_lo, b_hi, A, bmid, s1, s2 = unpack(r.x)
  ab = _at_bound(r.x, blo, bhi, free_s)
  mod = granot_sari_syn(xf, b_hi, b_lo, psyn, s1=s1, s2=s2, nuM=None, F_ext=A,
                        nuFnu=True, beta_mid=bmid)
  return _report('gs02', xf, mod, np.sqrt(np.mean(r.fun**2)), s1, s2, b_lo, b_hi, bmid,
                 ab, psyn)


def fit_2sbpl_flat(prep, psyn, s_hold=None, s_bounds=S_BOUNDS):
  '''
  two_sbpl fitted to the SAME prepared spectrum with the SAME freedoms, so the rms difference
  is the functional form and nothing else. Free: E_break, E_j (as E_break x 10**dsep), the
  mid photon index alpha2, the scale, and n1, n2 unless s_hold freezes them. alpha1 and beta
  are held at -2/3 and -p/2 - 1, the photon indices of the nuFnu asymptotes 4/3 and 1 - p/2.

  PARAMETERISED ON THE CROSSING E_j, not on E_peak. Ravasio's eq. (4) is a bijection between
  the two at fixed (alpha2, beta, n2), so this reaches the identical optimum while sharing
  fit_gs02_flat's parameter vector element for element -- which is what makes the two rms
  values comparable. The paper's E_peak is recovered from the fitted E_j (two_sbpl_Epeak)
  and reported alongside.
  '''
  if not prep['ok']:
    return _blank_fit('2sbpl')
  xf, yf = prep['x'], prep['ly']
  lo_b, hi_b = np.log10(xf.min()) - 0.5, np.log10(xf.max()) + 0.5
  bm_lo, bm_hi = _bmid_bounds(psyn)     # in F_nu index; alpha2 = bmid - 1
  lsb = (np.log10(s_bounds[0]), np.log10(s_bounds[1]))
  free_s = s_hold is None
  a1, beta = -2/3., -psyn/2. - 1.

  def unpack(q):
    E_br = 10**q[0]; E_j = E_br*10**q[1]
    n1, n2 = (10**q[4], 10**q[5]) if free_s else s_hold
    return E_br, E_j, 10**q[2], q[3], n1, n2

  def resid(q):
    E_br, E_j, A, bmid, n1, n2 = unpack(q)
    m = two_sbpl(xf, E_br, None, a1, bmid - 1., beta, n1=n1, n2=n2, A=A, nuFnu=True,
                 E_j=E_j)
    return np.log10(np.maximum(m, 1e-300)) - yf

  x_pk = xf[int(np.argmax(yf))]
  q0 = [np.log10(x_pk) - 1.5, 1.5, 0., 0.5*(bm_lo + bm_hi)]
  blo = [lo_b, 0., -8., bm_lo]
  bhi = [hi_b, hi_b - lo_b, 8., bm_hi]
  if free_s:
    q0 += [np.log10(S_2SBPL[0]), np.log10(S_2SBPL[1])]
    blo += [lsb[0], lsb[0]]; bhi += [lsb[1], lsb[1]]
  try:
    r = least_squares(resid, q0, bounds=(blo, bhi))
  except ValueError:
    return _blank_fit('2sbpl')
  E_br, E_j, A, bmid, n1, n2 = unpack(r.x)
  ab = _at_bound(r.x, blo, bhi, free_s)
  mod = two_sbpl(xf, E_br, None, a1, bmid - 1., beta, n1=n1, n2=n2, A=A, nuFnu=True,
                 E_j=E_j)
  return _report('2sbpl', xf, mod, np.sqrt(np.mean(r.fun**2)), n1, n2, E_br, E_j, bmid,
                 ab, psyn)


def _at_bound(q, blo, bhi, free_s, tol=1e-3):
  '''A fit that has run into a bound on a SHAPE parameter -- a break outside the window, a
  mid slope pinned on an asymptote, a smoothing at the end of its range -- is not a
  measurement of that parameter. The scale (slot 2) is excluded: it is a normalisation and
  its bounds are never reached in practice.'''
  idx = [0, 1, 3] + ([4, 5] if free_s else [])
  return bool(any(min(abs(q[i] - blo[i]), abs(q[i] - bhi[i])) < tol for i in idx))


def flat_window(x, sp, nuM, sigma, cutfac=CUT_FAC, fit_dec=FIT_DEC):
  """
  The (x, log10 flux) sample the anchored fits actually see: the cut-off divided out at the
  given nuM/sigma, everything below nuM/cutfac, and the top fit_dec decades. A transcription
  of the prologue of spectral_breaks.fit_smoothing_held, so fit_2sbpl_held and plot_example
  cannot drift away from the window the shipped GS02 fit uses -- if they ever did, the
  residual panel of plot_example would show it immediately.
  Returns (x, ly - max(ly), y0) or (None, None, nan).
  """
  x = np.asarray(x, float); sp = np.asarray(sp, float)
  g = np.isfinite(sp) & (sp > 0.) & np.isfinite(x) & (x > 0.)
  if g.sum() < 12:
    return None, None, np.nan
  xg, spg = x[g], sp[g]
  flat = sb._flatten_cutoff(xg, spg, nuM, sigma)
  ok = np.isfinite(flat) & (flat > 0.) & (xg < nuM/cutfac)
  if ok.sum() < 12:
    return None, None, np.nan
  xf, yf = xg[ok], np.log10(flat[ok])
  keep = yf > yf.max() - fit_dec
  xf, yf = xf[keep], yf[keep]
  if len(xf) < 12:
    return None, None, np.nan
  y0 = float(yf.max())
  return xf, yf - y0, y0


def fit_2sbpl_held(x, sp, psyn, b_lo, b_hi, nuM, beta_mid, free_bhi=True, cutfac=CUT_FAC,
    fit_dec=FIT_DEC, bounds=S_BOUNDS, s_hold=None, sigma=None, free_bmid=False,
    bmid_margin=BMID_MARGIN, bmid_physical=False, bmid_dep=BMID_DEP):
  """
  The 2SBPL analogue of spectral_breaks.fit_smoothing_held, written to mirror it line for
  line: the SAME cut-off flattening (same nuM, same smeared sigma), the SAME window and
  fit_dec, the SAME held break positions and slopes, the SAME free vector -- (n1, n2, scale)
  plus E_j if free_bhi, plus the mid index if free_bmid -- and the same bounds. Only the
  three-segment function differs, which is the entire point: any rms difference between the
  two is the functional form and nothing else.

  b_lo -> E_break and b_hi -> E_j, both crossings, which is the mapping that makes the two
  parameterisations comparable (phys_functions.two_sbpl). beta_mid comes in as an F_nu index,
  as in fit_smoothing_held, and becomes the photon index alpha2 = beta_mid - 1.

  Returns dict(s1, s2, b_hi_fit, rms, npts, at_bound, ok, beta_mid, bmid_at_bound) -- the
  keys fit_smoothing_held returns, so the caller can treat the two identically.
  """
  x = np.asarray(x, float); sp = np.asarray(sp, float)
  out = dict(s1=np.nan, s2=np.nan, b_hi_fit=np.nan, b_lo_fit=b_lo, rms=np.nan, npts=0,
             at_bound=False, ok=False, F_ext=np.nan, y0=np.nan,
             beta_mid=(np.nan if free_bmid else beta_mid), bmid_at_bound=False)
  if not all(np.isfinite(v) for v in (b_lo, b_hi, nuM, beta_mid)) or nuM <= 0. or b_lo <= 0.:
    return out
  xf, yf, y0 = flat_window(x, sp, nuM, sigma, cutfac, fit_dec)
  if xf is None:
    return out
  lb, ub = np.log10(bounds[0]), np.log10(bounds[1])
  a1, beta = -2/3., -psyn/2. - 1.

  names = [] if s_hold is not None else ['s1', 's2']
  names.append('A')
  if free_bhi:
    names.append('b_hi')
  if free_bmid:
    names.append('bmid')
  slot = {n: i for i, n in enumerate(names)}

  def unpack(q):
    n1, n2 = s_hold if s_hold is not None else (10**q[slot['s1']], 10**q[slot['s2']])
    bh = 10**q[slot['b_hi']] if free_bhi else b_hi
    bm = q[slot['bmid']] if free_bmid else beta_mid
    return n1, n2, 10**q[slot['A']], bh, bm

  def resid(q):
    n1, n2, A, bh, bm = unpack(q)
    m = two_sbpl(xf, b_lo, None, a1, bm - 1., beta, n1=n1, n2=n2, A=A, nuFnu=True, E_j=bh)
    return np.log10(np.maximum(m, 1e-300)) - yf

  init = {'s1': (np.log10(S_2SBPL[0]), lb, ub), 's2': (np.log10(S_2SBPL[1]), lb, ub),
          'A': (0., -8., 8.)}
  if free_bhi:
    if not (b_lo < b_hi < nuM):
      return out
    init['b_hi'] = (np.log10(b_hi), np.log10(b_lo), np.log10(nuM))
  if free_bmid:
    # the same two choices fit_smoothing_held offers: the SHAPE limits (what a three-segment
    # spectrum can carry at all) or the PHYSICAL ones (what a fused fast- or slow-cooling
    # knee can produce), widened by the departure shell integration actually produces
    if bmid_physical:
      bm_lo, bm_hi = (3. - psyn)/2. - bmid_dep - 1., 0.5 + bmid_dep - 1.
    else:
      bm_lo, bm_hi = (1. - psyn/2.) + bmid_margin - 1., 4./3. - bmid_margin - 1.
    bm0 = float(np.clip(beta_mid if np.isfinite(beta_mid) else -0.5, bm_lo, bm_hi))
    init['bmid'] = (bm0, bm_lo, bm_hi)
  try:
    r = least_squares(resid, [init[n][0] for n in names],
                      bounds=([init[n][1] for n in names], [init[n][2] for n in names]))
  except ValueError:
    return out
  n1, n2, A, bh, bm = unpack(r.x)
  out.update(F_ext=float(A), y0=y0, s1=float(n1), s2=float(n2), b_hi_fit=float(bh),
             beta_mid=float(bm) if np.isfinite(bm) else np.nan,
             rms=float(np.sqrt(np.mean(r.fun**2))), npts=int(len(xf)))
  if free_bmid:
    e = init['bmid']
    out['bmid_at_bound'] = bool(min(abs(bm - e[1]), abs(bm - e[2])) < 1e-3)
  ed = [abs(r.x[slot[n]] - e) for n in ('s1', 's2') if n in slot for e in (lb, ub)]
  out['at_bound'] = bool(ed and min(ed) < 1e-3)
  out['ok'] = not out['at_bound']
  return out


# The identified shape classes that carry BOTH breaks, i.e. the ones on which a three-segment
# form has three segments to describe. The others (a single break, no nu^(4/3) segment in
# band, or a merged knee with no mid segment) are counted and reported, not fitted: neither
# shape has anything to say there that the other does not, and the 2SBPL's own reason to
# exist -- the third segment -- is not in the data.
TWO_BRK = ('2brk', '2brk_tangent', '2brk_free')


# The anchored configurations, as kwargs to whichever of the two held fitters is running.
# `s_hold` is a NAME, not a value: 'published' resolves per shape to (1.3, 2.0) / (5.38, 2.69),
# 'fitted' to whatever pair the caller passes in (the prescription measured HERE), None frees
# the smoothing. Everything else about the fit -- the anchors, the window, the flattening --
# is identical across rows, so any column can be read as a function of the freedoms alone.
ANCHORED_CONFIGS = {
    # s free, mid slope held except on a marginal spectrum: exactly what
    # smoothing_from_identified does. THE reference row.
    'anchored':     dict(free_bmid='mc', bmid_physical=False, s_hold=None),
    # ... and the same with the mid slope freed on EVERY class, inside the shape limits
    'freemid':      dict(free_bmid=True, bmid_physical=False, s_hold=None),
    # ... freed inside the PHYSICAL limits instead (BMID_DEP): the one-zone interval widened
    # by the departure shell integration produces. A fit that pins here wants a mid slope no
    # fused cooling knee can make, which is a result rather than a nuisance.
    'freemid_phys': dict(free_bmid=True, bmid_physical=True, s_hold=None),
    # each shape's published pair frozen
    'presc':        dict(free_bmid='mc', bmid_physical=False, s_hold='published'),
    # the pair measured on THIS sweep frozen -- filled in by prescription_check
    'presc_fit':    dict(free_bmid='mc', bmid_physical=False, s_hold='fitted'),
}

DEFAULT_CONFIGS = ('anchored', 'freemid', 'freemid_phys', 'presc', 'free')

# The identified shape classes that carry BOTH breaks, i.e. the ones on which a three-segment
# form has three segments to describe. The others (a single break, no nu^(4/3) segment in
# band, or a merged knee with no mid segment) are counted and reported, not fitted: neither
# shape has anything to say there that the other does not, and the 2SBPL's own reason to
# exist -- the third segment -- is not in the data.
TWO_BRK = ('2brk', '2brk_tangent', '2brk_free')

PUBLISHED_S = {'gs02': S_GS02, '2sbpl': S_2SBPL}


def compare_spectrum(x, sp, psyn, configs=DEFAULT_CONFIGS, s_fit_hold=None):
  '''
  Both shapes on one spectrum, in matched configurations. Returns a list of row dicts (2
  shapes x however many configurations the spectrum supports), each carrying the shape class
  the segment route read off it.

    'anchored'      break positions and the mid slope HELD at what the segment route measured
                (spectral_breaks.breaks_from_identified), the cut-off divided out with that
                same measurement's smeared nu_M -- so the only freedom left is the smoothing
                (plus b_hi and, on a marginal spectrum, the mid slope, exactly as
                smoothing_from_identified frees them). THE comparison: the fitted (s1, s2) vs
                (n1, n2) are then real measurements in one convention, not a degenerate ridge.
                The GS02 side is literally spectral_breaks.fit_smoothing_held, so it IS the
                shipped route rather than a re-implementation of it.
    'freemid'       the same with the mid slope freed on EVERY class rather than only the
                marginal ones. Freeing it costs the smoothing some of its constraint -- the
                two trade, which fit_smoothing_held's docstring measures on MC bins -- so
                read the pair (a_mid, s) from this row together, never one from here and the
                other from 'anchored'.
    'freemid_phys'  ... with the mid slope bounded by what a fused cooling knee can physically
                produce (BMID_DEP) instead of by what a three-segment shape can carry
                (BMID_MARGIN). The difference between the two rows is the whole question of
                whether a measured "departure" is physics or the two-break form imitating one
                broad knee.
    'presc'         each shape's PUBLISHED smoothing pair frozen -- (1.3, 2.0) for GS02,
                Ravasio's (5.38, 2.69) for the 2SBPL. What a tabulated pair costs.
    'presc_fit'     the pair measured on THIS sweep frozen (s_fit_hold, per shape). Only
                available when the caller supplies it: see prescription_check.
    'free'          no anchors at all: six free parameters each (two breaks, mid slope, two
                smoothings, scale) on the flattened spectrum. This is the best either family
                can do with equal freedom, and the ONLY configuration that survives on
                one-break spectra. Its rms is a fair comparison; its s and a_mid are NOT
                measurements -- freeing both breaks against both smoothings is the documented
                degeneracy (see the spectral_breaks header), and they are flagged 'free' so
                nothing downstream quotes them by accident.
  '''
  br = sb.breaks_from_identified(x, sp, psyn)
  meta = dict(regime=br.get('regime'), cls_shape=br.get('shape'),
              n_breaks=int(br.get('n_breaks', 0) or 0), nuM=br.get('nuM', np.nan),
              sigma=br.get('sigma', np.nan), two_brk=False,
              a_mid_seed=br.get('a_mid', np.nan))
  rows = []
  anchored = [c for c in configs if c in ANCHORED_CONFIGS]
  if anchored and br['ok'] and br['shape'] in TWO_BRK:
    meta['two_brk'] = True
    for cfg in anchored:
      spec = ANCHORED_CONFIGS[cfg]
      # 'mc' means: free the mid slope only where no mid segment was identified
      fb = (br['regime'] == 'MC') if spec['free_bmid'] == 'mc' else bool(spec['free_bmid'])
      for shape, fn in (('gs02', sb.fit_smoothing_held), ('2sbpl', fit_2sbpl_held)):
        if spec['s_hold'] == 'published':
          hold = PUBLISHED_S[shape]
        elif spec['s_hold'] == 'fitted':
          hold = (s_fit_hold or {}).get(shape)
          if hold is None:
            continue          # no measured pair supplied: the row simply does not exist
        else:
          hold = None
        f = fn(x, sp, psyn, br['b_lo'], br['b_hi'], br['nuM'], br['a_mid'] - 1.,
               sigma=br['sigma'], free_bmid=fb, s_hold=hold,
               bmid_physical=spec['bmid_physical'])
        b_hi = f['b_hi_fit'] if np.isfinite(f['b_hi_fit']) else br['b_hi']
        rows.append(dict(shape=shape, config=cfg, rms=f['rms'], s1=f['s1'], s2=f['s2'],
            b_lo=br['b_lo'], b_hi=b_hi, sep=np.log10(b_hi/br['b_lo']),
            a_mid=f['beta_mid'] + 1., npts=f['npts'], mid_fitted=bool(fb),
            at_bound=bool(f['at_bound'] or f['bmid_at_bound']),
            bmid_at_bound=bool(f['bmid_at_bound']),
            ok=bool(f['ok'] and not f['bmid_at_bound']), x_pk=np.nan,
            E_peak=two_sbpl_Epeak(b_hi, f['beta_mid'] - 1., -psyn/2. - 1., f['s2'])))

  if 'free' in configs:
    prep = prepare_spectrum(x, sp, psyn)
    for shape, fn in (('gs02', fit_gs02_flat), ('2sbpl', fit_2sbpl_flat)):
      d = fn(prep, psyn)
      d.update(config='free', mid_fitted=True, bmid_at_bound=False)
      rows.append(d)
  for d in rows:
    d.update(meta)
  return rows


# ---------------------------------------------------------------------------
# how well is THEIR alpha2 measured, at THEIR break ratios?
# ---------------------------------------------------------------------------
# GRB 160625B as Ravasio et al. report it: the GBM band the time-resolved fits use, a break
# near 100 keV, and E_peak/E_break running ~35 early to ~5 late (their Fig. 7, bottom).
GRB_BAND = (8., 4.0e4)        # keV; NaI 8 keV to BGO 40 MeV
GRB_NCHAN = 128               # CSPEC's logarithmically spaced channels
GRB_EBREAK = 100.             # keV
GRB_RATIOS = (5., 10., 20., 35.)
GRB_A1, GRB_A2, GRB_BETA = -0.63, -1.48, -2.5     # their mean photon indices


def mid_segment_width(ratios=GRB_RATIOS, a1=GRB_A1, a2=GRB_A2, beta=GRB_BETA,
    n1=RAVASIO_N1, n2=RAVASIO_N2, tols=(0.05, 0.10), verbose=True):
  '''
  How much STRAIGHT mid segment a 2SBPL actually has at Ravasio's own parameters -- the decades
  over which its local slope sits within `tol` of alpha2.

  This is the precondition for reading a mid index off a spectrum, and at their smaller break
  ratios it is barely met: with n1 = 5.38 and n2 = 2.69 the two transitions are ~0.22 and
  ~0.37 dex wide on their own, so at E_peak/E_break = 5 they overlap almost completely.
  Measured here (|slope - alpha2| < 0.05):

      E_pk/E_br     5     10     20     35    100
      width [dex]  0.11   0.29   0.57   0.81   1.26

  The minimum |slope - alpha2| is 0.000 at every ratio: the slope always passes THROUGH alpha2,
  it just does not linger there. So alpha2 at small ratio is a TANGENT, inferred from the
  curvature of the whole shape under an assumed smoothing, not a plateau read off the data --
  which is exactly the situation ravasio_recovery then prices.
  '''
  E = np.logspace(-2., 8., 200001)
  rows = []
  for ratio in ratios:
    N = two_sbpl(E, 1., ratio, a1, a2, beta, n1=n1, n2=n2)
    s = np.gradient(np.log10(N), np.log10(E))
    Ej = two_sbpl_Ej(ratio, a2, beta, n2)
    m = (E > 1./3.) & (E < Ej*3.)          # between the breaks, with a little room
    d = np.abs(s - a2)
    r = dict(ratio=ratio, sep_dex=float(np.log10(Ej)), dmin=float(d[m].min()))
    for tol in tols:
      k = m & (d < tol)
      r[f'width_{tol:g}'] = (float(np.log10(E[k].max()/E[k].min())) if k.any() else 0.)
    rows.append(r)
  out = pd.DataFrame(rows)
  if verbose:
    print(f'\nstraight mid segment at n1={n1}, n2={n2} (alpha2={a2})')
    print(out.to_string(index=False, float_format=lambda v: f'{v:.2f}'))
  return out


def _fit_2sbpl_photon(E, N, n1, n2, band=GRB_BAND, seed=(GRB_A1, GRB_A2, GRB_BETA)):
  '''
  Ravasio's own fit, on a photon spectrum: A, alpha1, E_break, alpha2, E_peak and beta free,
  n1 and n2 HELD (their Sect. 2.4 -- they fixed both because a free n1 "is not always
  constrained"). Residual in log10 flux, uniformly weighted over logarithmic channels, which
  is the shape-fitting proxy for their channel-space chi2.
  '''
  ly = np.log10(N)

  def unpack(q):
    return 10**q[0], 10**q[0]*10**q[1], 10**q[2], q[3], q[4], q[5]

  def resid(q):
    Eb, Ep, A, b1, b2, bb = unpack(q)
    m = two_sbpl(E, Eb, Ep, b1, b2, bb, n1=n1, n2=n2, A=A)
    return np.log10(np.maximum(m, 1e-300)) - ly

  # beta < -2 < alpha2 is required for eq. (4) to have a solution at all, hence the bounds
  q0 = [np.log10(GRB_EBREAK), 1., 0., seed[0], seed[1], seed[2]]
  blo = [np.log10(band[0]), 0., -20., -1.6, -2.15, -5.0]
  bhi = [np.log10(band[1]), 3.5, 20., -0.1, -0.85, -2.05]
  r = least_squares(resid, q0, bounds=(blo, bhi))
  Eb, Ep, A, b1, b2, bb = unpack(r.x)
  return dict(E_break=Eb, E_peak=Ep, a1=b1, a2=b2, beta=bb, ratio=Ep/Eb,
              rms=float(np.sqrt(np.mean(r.fun**2))))


def ravasio_recovery(ratios=GRB_RATIOS, n1_true=(1.12, 2.0, 2.69, 5.38, 10.0),
    a2_true=(GRB_A2,), band=GRB_BAND, nchan=GRB_NCHAN, n1_fit=RAVASIO_N1,
    n2_fit=RAVASIO_N2, outdir=OUTDIR, verbose=True):
  '''
  HOW WELL IS alpha2 MEASURED when the breaks are less than a decade apart? Synthetic 2SBPL
  spectra with a KNOWN mid index, on GRB 160625B's band and channel count, refitted the way
  the paper fits -- everything free except n1, n2, which are held at (5.38, 2.69).

  Two results, and they say opposite things:

  SELF-RECOVERY IS EXACT. Generated and fitted at the same (n1, n2), alpha2 comes back with
  bias 0.000 and rms ~1e-14 at every ratio down to 5. The estimator is unbiased and the break
  ratio is recovered exactly, so the paper's small quoted errors (+-0.02 to +-0.13) are real
  STATISTICAL errors. Nothing is wrong with the fit as a fit.

  THE SMOOTHING IS THE SYSTEMATIC, and it is largest exactly where the ratio is smallest.
  Generating with a smoother true break and still holding n1 = 5.38, alpha2 comes back HARDER
  than the truth (bias in alpha2, i.e. fitted minus true):

      n1_true      ratio 5   ratio 10   ratio 20   ratio 35
      1.12          +0.258     +0.208     +0.167     +0.140
      2.00          +0.172     +0.120     +0.085     +0.066
      2.69          +0.114     +0.074     +0.050     +0.037
      10.0          -0.047     -0.026     -0.016     -0.011

  and the misfit it leaves is only 0.003-0.008 dex, i.e. the wrong-smoothing fit is very nearly
  as good. A smoother break with a softer mid slope and a sharper break with a harder one are
  the same spectrum to within a few thousandths of a dex.

  READ THE SIGN CAREFULLY: a bias of +0.26 means that to OBSERVE alpha2 = -1.48 under a true
  n1 = 1.12 the truth must be alpha2 ~ -1.74 -- the SLOW-cooling value -(p-1)/2 - 1 at
  p = 2.5. Passing a2_true=(-1.75,) runs that case directly: at ratio 5 it fits back as -1.455
  at rms 0.0069, indistinguishable from their reported -1.48.

  WHAT PROTECTS THEM is alpha1, not alpha2. The same mis-specified fits want alpha1 ~ -0.78 to
  -0.83, against their measured -0.63 +- 0.08; only the milder n1 ~ 2 case (alpha1 -0.70,
  alpha2 -1.58) sits inside their error bar. So the low-energy index limits how far the
  smoothing can be wrong, and with it how far alpha2 can move -- but not to better than ~0.1
  at their smallest ratios.

  NB n1 = 5.38 is not an arbitrary choice on their part: it is the mean of their own free-n1
  fits. The caveat is their own -- those fits were "not always constrained" -- and the scatter
  behind that mean is not published, so its width cannot be propagated here.

  Returns a DataFrame; writes ravasio_recovery.csv.
  '''
  E = np.logspace(np.log10(band[0]), np.log10(band[1]), nchan)
  rows = []
  for a2t in a2_true:
    for ratio in ratios:
      for n1t in n1_true:
        N = two_sbpl(E, GRB_EBREAK, ratio*GRB_EBREAK, GRB_A1, a2t, GRB_BETA, n1=n1t,
                     n2=n2_fit)
        f = _fit_2sbpl_photon(E, N, n1_fit, n2_fit, band=band)
        rows.append(dict(a2_true=a2t, ratio_true=ratio, n1_true=n1t, a2_fit=f['a2'],
                         a2_bias=f['a2'] - a2t, a1_fit=f['a1'], beta_fit=f['beta'],
                         ratio_fit=f['ratio'], ratio_bias=f['ratio']/ratio, rms=f['rms']))
  out = pd.DataFrame(rows)
  os.makedirs(outdir, exist_ok=True)
  out.to_csv(os.path.join(outdir, 'ravasio_recovery.csv'), index=False)
  if verbose:
    print(f'\n=== alpha2 recovery with n1 HELD at {n1_fit} (as the paper does), '
          f'n2 = {n2_fit} ===')
    print(f"{'a2 true':>8} {'ratio':>6} {'n1 true':>8} {'a2 fit':>8} {'bias':>8} "
          f"{'a1 fit':>8} {'ratio fit':>10} {'rms [dex]':>10}")
    for _, r in out.iterrows():
      print(f"{r['a2_true']:>8.2f} {r['ratio_true']:>6.0f} {r['n1_true']:>8.2f} "
            f"{r['a2_fit']:>8.3f} {r['a2_bias']:>+8.3f} {r['a1_fit']:>8.3f} "
            f"{r['ratio_fit']:>10.2f} {r['rms']:>10.4f}")
  return out


# ---------------------------------------------------------------------------
# the sweep
# ---------------------------------------------------------------------------
def _run_point(args):
  '''One sweep point in a worker: load its cache, fit both shapes at every LOGT epoch.'''
  key, method, z, logr, logt, configs, s_fit_hold = args
  res = swp.load_sweep(swp.method_outdir(method, key, z))
  r = [q for q in res if abs(q['log10ratio'] - logr) < 1e-9][0]
  del res                       # the other seven points' arrays go straight back
  barT_f = swp.exit_onset_barT(key, z=z)
  x = swp.nu_over_num(r)
  psyn = r['env'].psyn
  barT = np.asarray(r['Tb'], float) - 1.
  rows = []
  for k, lt, i in swp._spectra_series(r, barT_f, logt=logt):
    for d in compare_spectrum(x, r['nuFnu'][i, :], psyn, configs=configs,
                              s_fit_hold=s_fit_hold):
      d.update(z=z, logr=logr, logt=lt, barT=float(barT[i]), i_bin=int(i), psyn=psyn,
               nu_M_nom=swp.nu_M_over_num(r))
      rows.append(d)
  print(f'  z={z} log10ratio={logr:+.1f}: {len(rows)} rows', flush=True)
  return rows


def compare_sweep(key=KEY, method=METHOD, zlist=(Z_RS, Z_FS), logt=LOGT, nproc=NPROC,
    outdir=OUTDIR, cache=True, configs=DEFAULT_CONFIGS, s_fit_hold=None,
    csv_name='fits.csv'):
  '''
  Both shapes on every cached point of the rarcut sweep, both shells. CACHE-ONLY: this never
  triggers a sweep, it only reads the point caches, so it is safe to run alongside anything
  else. Writes csv_name into outdir and returns the DataFrame.

  configs / s_fit_hold go straight to compare_spectrum. A restricted `configs` is how the
  second, prescription pass avoids recomputing the whole table -- it still pays for
  breaks_from_identified, which is most of the cost, but not for the fits it already has.
  csv_name must then differ, or that pass would overwrite the first one's cache.
  '''
  os.makedirs(outdir, exist_ok=True)
  csv = os.path.join(outdir, csv_name)
  if cache and os.path.isfile(csv):
    df = pd.read_csv(csv)
    print(f'loaded {len(df)} cached fit rows from {csv}')
    return df
  jobs = []
  for z in zlist:
    res = swp.load_sweep(swp.method_outdir(method, key, z))
    if not res:
      raise FileNotFoundError(f'no cached {method} sweep for z={z} -- run '
                              f'sweep_rarcut.main(z={z}) first')
    jobs += [(key, method, z, float(r['log10ratio']), logt, tuple(configs),
              s_fit_hold) for r in res]
    del res
  npr = cell_pool.resolve_nproc(nproc, cap=len(jobs))
  print(f'{len(jobs)} sweep points x {len(logt)} epochs x 2 shapes x '
        f'{len(configs)} configs on {npr} workers')
  if npr > 1:
    ctx = cell_pool.pool_context()
    with ctx.Pool(npr) as pool:
      out = pool.map(_run_point, jobs)
  else:
    out = [_run_point(j) for j in jobs]
  df = pd.DataFrame([row for rows in out for row in rows])
  df.to_csv(csv, index=False)
  print(f'{len(df)} fit rows written to {csv}')
  return df


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
def _q(v):
  '''median [q16-q84] of a finite sample, as a formatted string.'''
  v = np.asarray(v, float); v = v[np.isfinite(v)]
  if not v.size:
    return 'n/a'
  a, b, c = np.percentile(v, [16., 50., 84.])
  return f'{b:.3f} [{a:.3f}-{c:.3f}]'


def _qe(v, fmt='{:+.1e}'):
  '''median [q16-q84] in scientific notation. The anchored rms DIFFERENCE lands around
  1e-5 dex, which _q's three decimals render as a row of zeros -- true but unreadable.'''
  v = np.asarray(v, float); v = v[np.isfinite(v)]
  if not v.size:
    return 'n/a'
  a, b, c = np.percentile(v, [16., 50., 84.])
  return f'{fmt.format(b)} [{fmt.format(a)} {fmt.format(c)}]'


def separation_table(df, config='anchored', edges=(0., 0.75, 1.25, 2., 3., 9.),
    outdir=OUTDIR, verbose=True):
  '''
  THE TEST OF THE ANALYTIC PREDICTION. two_sbpl and granot_sari_syn are the same function
  once the breaks are far enough apart, and differ only through the lower break's curvature
  at the upper crossing -- so the paired rms difference must be a function of the SEPARATION
  and must vanish as it grows. This bins the fitted epochs by log10(b_hi/b_lo) and reports
  the paired difference in each bin, against shape_difference's prediction at the same
  separation.

  A positive d(rms) means the 2SBPL fits WORSE. Both are |differences of the same fit on the
  same points|, so the comparison is paired bin by bin and the sign is meaningful even when
  the magnitude is far below either shape's own rms.
  '''
  g = df[df.config == config]
  piv = g.pivot_table(index=['z', 'logr', 'logt'], columns='shape',
                      values=['rms', 'sep', 's1', 's2', 'ok'])
  both = piv[('ok', 'gs02')].astype(bool) & piv[('ok', '2sbpl')].astype(bool)
  piv = piv[both]
  sep = piv[('sep', 'gs02')].values
  dr = (piv[('rms', '2sbpl')] - piv[('rms', 'gs02')]).values
  rms0 = piv[('rms', 'gs02')].values
  rows = []
  for lo, hi in zip(edges[:-1], edges[1:]):
    m = (sep >= lo) & (sep < hi)
    if not m.sum():
      continue
    rows.append(dict(sep_lo=lo, sep_hi=hi, n=int(m.sum()),
                     sep_med=float(np.median(sep[m])), rms_gs02=_q(rms0[m]),
                     drms=_qe(dr[m]), drms_over_rms=_qe(dr[m]/rms0[m]),
                     frac_2sbpl_worse=float(np.mean(dr[m] > 0.))))
  out = pd.DataFrame(rows)
  os.makedirs(outdir, exist_ok=True)
  out.to_csv(os.path.join(outdir, f'by_separation_{config}.csv'), index=False)
  if verbose:
    print(f'\n=== {config} fits binned by break separation (both shells, all regimes) ===')
    print(f"{'sep [dex]':>12} {'n':>4} {'rms gs02 [dex]':>22} {'d(rms) = 2sbpl - gs02':>34} "
          f"{'2sbpl worse':>12}")
    for _, r in out.iterrows():
      print(f"{r['sep_lo']:>5.2f}-{r['sep_hi']:<6.2f} {r['n']:>4} {r['rms_gs02']:>22} "
            f"{r['drms']:>34} {100*r['frac_2sbpl_worse']:>11.0f}%")
  return out


def summary_table(df, outdir=OUTDIR, verbose=True,
    show=('anchored', 'freemid', 'presc', 'free')):
  '''
  Per shell and configuration: the fitted sharpness, the mid slope and the rms of each shape,
  over the bins where BOTH fits are usable (neither at a bound), so every comparison is
  paired. Writes summary.csv.
  '''
  rows = []
  for (z, cfg), g in df.groupby(['z', 'config']):
    piv = g.pivot_table(index=['logr', 'logt'], columns='shape',
                        values=['rms', 's1', 's2', 'a_mid', 'sep', 'b_lo', 'ok'])
    both = piv[('ok', 'gs02')].astype(bool) & piv[('ok', '2sbpl')].astype(bool)
    n = int(both.sum())
    if not n:
      continue
    q = piv[both]
    d = dict(shell=SHELL.get(z, z), config=cfg, n=n, n_all=len(piv))
    for sh in SHAPES:
      d[f's1_{sh}'] = _q(q[('s1', sh)]); d[f's2_{sh}'] = _q(q[('s2', sh)])
      d[f'a_mid_{sh}'] = _q(q[('a_mid', sh)]); d[f'rms_{sh}'] = _q(q[('rms', sh)])
      d[f'sep_{sh}'] = _q(q[('sep', sh)])
    dr = q[('rms', '2sbpl')] - q[('rms', 'gs02')]
    d['drms_2sbpl_minus_gs02'] = _qe(dr)
    d['frac_2sbpl_better'] = float(np.mean(dr < 0.))
    d['b_lo_ratio'] = _q(q[('b_lo', '2sbpl')]/q[('b_lo', 'gs02')])
    rows.append(d)
  out = pd.DataFrame(rows)
  os.makedirs(outdir, exist_ok=True)
  out.to_csv(os.path.join(outdir, 'summary.csv'), index=False)
  if verbose:
    for _, r in out.iterrows():
      if show is not None and r['config'] not in show:
        continue                       # in the csv, just not on the terminal
      print(f"\n=== {r['shell']}  config={r['config']}   "
            f"{r['n']} of {r['n_all']} epochs with both fits unpinned ===")
      print(f"{'':>10} {'s1 (low break)':>26} {'s2 (upper)':>26} "
            f"{'a_mid (nuFnu)':>26} {'rms [dex]':>26}")
      for sh in SHAPES:
        print(f"{sh:>10} {r[f's1_{sh}']:>26} {r[f's2_{sh}']:>26} "
              f"{r[f'a_mid_{sh}']:>26} {r[f'rms_{sh}']:>26}")
      print(f"  rms(2sbpl) - rms(gs02) = {r['drms_2sbpl_minus_gs02']} dex, "
            f"2sbpl better in {100*r['frac_2sbpl_better']:.0f}% of epochs")
      print(f"  break separation dex: gs02 {r['sep_gs02']}   2sbpl {r['sep_2sbpl']}")
  return out


def epoch_table(df, outdir=OUTDIR, verbose=True):
  '''
  The same comparison resolved in TIME rather than pooled, on the ANCHORED fits: the paired
  rms difference, the two smoothings and the mid slope per epoch, both shells together. The
  rise, the crossing and the tail are different spectra -- the tail softens the mid segment
  and pulls the breaks together -- so a pooled median can hide a form that only fails in one
  of them.
  '''
  g = df[df.config == 'anchored']
  piv = g.pivot_table(index=['z', 'logr', 'logt'], columns='shape',
                      values=['rms', 'a_mid', 'sep', 's1', 's2', 'ok'])
  both = piv[('ok', 'gs02')].astype(bool) & piv[('ok', '2sbpl')].astype(bool)
  piv = piv[both].reset_index()
  rows = []
  for lt, q in piv.groupby('logt'):
    rows.append(dict(logt=lt, n=len(q),
                     rms_gs02=_q(q[('rms', 'gs02')]), rms_2sbpl=_q(q[('rms', '2sbpl')]),
                     drms=_qe(q[('rms', '2sbpl')] - q[('rms', 'gs02')]),
                     a_mid_gs02=_q(q[('a_mid', 'gs02')]), a_mid_2sbpl=_q(q[('a_mid', '2sbpl')]),
                     s1_gs02=_q(q[('s1', 'gs02')]), s1_2sbpl=_q(q[('s1', '2sbpl')]),
                     s2_gs02=_q(q[('s2', 'gs02')]), s2_2sbpl=_q(q[('s2', '2sbpl')]),
                     sep=_q(q[('sep', 'gs02')])))
  out = pd.DataFrame(rows)
  os.makedirs(outdir, exist_ok=True)
  out.to_csv(os.path.join(outdir, 'per_epoch.csv'), index=False)
  if verbose:
    print('\n=== anchored fits per epoch (both shells pooled) ===')
    print(f"{'logt':>6} {'n':>4} {'rms gs02':>22} {'rms 2sbpl':>22} {'d(rms)':>34}")
    for _, r in out.iterrows():
      print(f"{r['logt']:>6.1f} {r['n']:>4} {r['rms_gs02']:>22} {r['rms_2sbpl']:>22} "
            f"{r['drms']:>34}")
    print(f"\n{'logt':>6} {'a_mid gs02':>22} {'a_mid 2sbpl':>22} {'s1 gs02':>22} "
          f"{'s1 2sbpl':>22}")
    for _, r in out.iterrows():
      print(f"{r['logt']:>6.1f} {r['a_mid_gs02']:>22} {r['a_mid_2sbpl']:>22} "
            f"{r['s1_gs02']:>22} {r['s1_2sbpl']:>22}")
  return out


def _pair(v1, v2):
  '''median [q16-q84] of a pair of samples, formatted as one "(a, b)" cell.'''
  def m(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return np.percentile(v, [16., 50., 84.]) if v.size else (np.nan,)*3
  a1, b1, c1 = m(v1); a2, b2, c2 = m(v2)
  return f'({b1:.2f} [{a1:.2f}-{c1:.2f}], {b2:.2f} [{a2:.2f}-{c2:.2f}])'


def smoothing_prescription(df, config='anchored', outdir=OUTDIR, verbose=True):
  '''
  THE BEST SMOOTHING PAIR EACH SHAPE WANTS on these spectra, from the anchored fits where it
  is the only shape freedom left. Reported pooled and split by shape class, because a single
  pair is only defensible if the classes agree on it -- and they do not, which is the point of
  the split (fit_smoothing_held's docstring documents the same thing for MC: a freed mid slope
  drags s2 down by ~2x).

  The pooled MEDIAN is what gets frozen in the 'presc_fit' row; prescription_check measures
  what freezing it costs. Quoted with q16-q84, never as a bare number: the spread IS the
  statement about how tabulatable the pair is.
  '''
  g = df[(df.config == config) & df.ok.astype(bool)]
  rows = []
  for shape in SHAPES:
    q = g[g['shape'] == shape]
    for cls in ('all',) + tuple(sorted(set(q.regime.dropna()))):
      qq = q if cls == 'all' else q[q.regime == cls]
      if len(qq) < 3:
        continue
      rows.append(dict(shape=shape, cls=cls, n=len(qq),
                       s1=_q(qq.s1), s2=_q(qq.s2),
                       s1_med=float(np.nanmedian(qq.s1)), s2_med=float(np.nanmedian(qq.s2)),
                       published=str(PUBLISHED_S[shape])))
  out = pd.DataFrame(rows)
  os.makedirs(outdir, exist_ok=True)
  out.to_csv(os.path.join(outdir, 'smoothing_prescription.csv'), index=False)
  best = {sh: (float(out[(out['shape'] == sh) & (out.cls == 'all')].s1_med.iloc[0]),
               float(out[(out['shape'] == sh) & (out.cls == 'all')].s2_med.iloc[0]))
          for sh in SHAPES if ((out['shape'] == sh) & (out.cls == 'all')).any()}
  if verbose:
    print(f'\n=== best smoothing pair, {config} fits (s is the only shape freedom there) ===')
    print(f"{'shape':>7} {'class':>6} {'n':>4} {'s1 / n1 (lower break)':>24} "
          f"{'s2 / n2 (upper)':>24}  published")
    for _, r in out.iterrows():
      print(f"{r['shape']:>7} {r['cls']:>6} {r['n']:>4} {r['s1']:>24} {r['s2']:>24}"
            f"  {r['published'] if r['cls'] == 'all' else ''}")
    for sh, (a, b) in best.items():
      print(f'  -> {sh}: measured pair ({a:.2f}, {b:.2f}) vs published '
            f'({PUBLISHED_S[sh][0]:.2f}, {PUBLISHED_S[sh][1]:.2f})')
  return out, best


def mid_slope_table(df, outdir=OUTDIR, verbose=True):
  '''
  What a FREE mid slope does, per shape class, and whether it is a departure from theory or
  the two-break form imitating a single knee.

  Four things per class, both shapes:
    a_mid held   the segment route's measured mid line, i.e. what 'anchored' holds
    a_mid free   refitted inside the SHAPE limits ('freemid')
    a_mid phys   ... inside the PHYSICAL ones ('freemid_phys'), plus the fraction that PINS
                 there -- a fit at that bound wants a mid slope no fused cooling knee makes
    d(rms)       what freeing it buys. A large gain bought by a slope outside the physical
                 window is the signature of the form absorbing curvature, not of physics.

  The departure column is against the ONE-ZONE asymptote of the class -- 1/2 in fast cooling,
  (3-p)/2 in slow, in nuFnu index. MC has no asymptote to depart from (its two knees have
  fused, so there is no mid segment in the data at all) and is reported without one.

  READ THIS ALONGSIDE the calibrated line-estimator result, which is the project's actual
  measurement of the mid-slope departure: on-axis it is consistent with ZERO in both branches
  once the estimator's own tilt is removed, and only the slow-cooling post-crossing softening
  survives. This table is a DIFFERENT estimator (a template parameter, not a fitted line) and
  its bias has not been calibrated, so it cannot on its own promote a departure to a
  measurement. What it can do -- and is here for -- is say whether the two functional forms
  agree, which they must if a departure is in the data rather than in the algebra.
  '''
  def asymptote(cls, p):
    return {'FC': 0.5, 'SC': (3. - p)/2.}.get(cls, np.nan)

  p = float(df.psyn.iloc[0])
  key = ['z', 'logr', 'logt', 'shape']
  cfgs = ('anchored', 'freemid', 'freemid_phys')
  g = df[df.config.isin(cfgs)]
  piv = g.pivot_table(index=key, columns='config',
                      values=['a_mid', 'rms', 'bmid_at_bound', 'ok'])
  # the class is a per-SPECTRUM label, not a fitted value, so it comes back by reindexing on
  # the pivot's own index -- joining it in would collide with the MultiIndex columns
  reg = df.drop_duplicates(key).set_index(key)['regime'].reindex(piv.index)
  keep = piv[('ok', 'anchored')].astype(bool)
  piv, reg = piv[keep], reg[keep]
  # ON-AXIS vs POST-CROSSING, the split the calibrated line-estimator result is quoted on:
  # logt is log10(bar{T}/bar{T}_f), so logt < 0 is emission while the shock is still crossing
  # and logt > 0 is the high-latitude tail. The two are different spectra and the tail is
  # where the surviving slow-cooling softening lives, so a pooled median mixes them.
  ep = np.where(piv.index.get_level_values('logt') < 0., 'on-axis', 'post-crossing')
  rows = []
  groups = list(piv.groupby([piv.index.get_level_values('shape'), reg.values]).groups.items())
  groups += [((sh, f'{c}/{e}'), ix) for (sh, c, e), ix in
             piv.groupby([piv.index.get_level_values('shape'), reg.values, ep]).groups.items()]
  for (shape, cls), idx in groups:
    q = piv.loc[idx]
    if len(q) < 3:
      continue
    a0 = asymptote(cls.split('/')[0], p)
    rows.append(dict(shape=shape, cls=cls, n=len(q), asymptote=a0,
        a_mid_held=_q(q[('a_mid', 'anchored')]),
        a_mid_free=_q(q[('a_mid', 'freemid')]),
        a_mid_phys=_q(q[('a_mid', 'freemid_phys')]),
        dep_free=_q(q[('a_mid', 'freemid')] - a0),
        dep_phys=_q(q[('a_mid', 'freemid_phys')] - a0),
        frac_free_pinned=float(np.mean(q[('bmid_at_bound', 'freemid')] > 0.5)),
        frac_phys_pinned=float(np.mean(q[('bmid_at_bound', 'freemid_phys')] > 0.5)),
        drms_free=_qe(q[('rms', 'freemid')] - q[('rms', 'anchored')]),
        drms_phys=_qe(q[('rms', 'freemid_phys')] - q[('rms', 'anchored')])))
  out = pd.DataFrame(rows)
  os.makedirs(outdir, exist_ok=True)
  out.to_csv(os.path.join(outdir, 'mid_slope.csv'), index=False)

  # DO THE TWO SHAPES AGREE on the freed mid slope? If a departure is in the data it cannot
  # depend on which three-segment algebra read it off; if the two disagree, it is algebra.
  w = piv[('a_mid', 'freemid')].unstack('shape')
  agree = (w['2sbpl'] - w['gs02']).dropna() if {'gs02', '2sbpl'} <= set(w.columns) \
          else pd.Series(dtype=float)

  if verbose:
    print(f'\n=== free mid slope by shape class (nuFnu index; asymptotes 1/2 fast, '
          f'{(3.-p)/2:.2f} slow) ===')
    print(f"{'shape':>7} {'cls':>18} {'n':>4} {'asym':>6} {'a_mid held':>21} "
          f"{'a_mid free':>21} {'departure (free)':>21} {'pinned':>7} "
          f"{'d(rms) from freeing':>32}".replace("{'cls':>4}", "{'cls':>18}"))
    for _, r in out[~out.cls.str.contains('/')].iterrows():
      a = f"{r['asymptote']:.2f}" if np.isfinite(r['asymptote']) else '  --'
      print(f"{r['shape']:>7} {r['cls']:>18} {r['n']:>4} {a:>6} {r['a_mid_held']:>21} "
            f"{r['a_mid_free']:>21} {r['dep_free']:>21} "
            f"{100*r['frac_free_pinned']:>6.0f}% {r['drms_free']:>32}")
    print('\n  the same split on-axis (still crossing) vs post-crossing (high-latitude '
          'tail), which is\n  how the calibrated line-estimator departures are quoted:')
    for _, r in out[out.cls.str.contains('/')].iterrows():
      a = f"{r['asymptote']:.2f}" if np.isfinite(r['asymptote']) else '  --'
      print(f"{r['shape']:>7} {r['cls']:>18} {r['n']:>4} {a:>6} {r['a_mid_held']:>21} "
            f"{r['a_mid_free']:>21} {r['dep_free']:>21} "
            f"{100*r['frac_free_pinned']:>6.0f}% {r['drms_free']:>32}")
    print(f"\n  refitted inside the PHYSICAL window [{(3.-p)/2-BMID_DEP:.2f}, "
          f"{0.5+BMID_DEP:.2f}] (BMID_DEP) instead of the shape one "
          f"[{1.-p/2.+BMID_MARGIN:.2f}, {4./3.-BMID_MARGIN:.2f}]:")
    print(f"{'shape':>7} {'cls':>18} {'a_mid phys':>21} {'departure':>21} {'pinned':>7} "
          f"{'d(rms) from freeing':>32}")
    for _, r in out[~out.cls.str.contains('/')].iterrows():
      print(f"{r['shape']:>7} {r['cls']:>18} {r['a_mid_phys']:>21} {r['dep_phys']:>21} "
            f"{100*r['frac_phys_pinned']:>6.0f}% {r['drms_phys']:>32}")
    if len(agree):
      print(f'\n  the two shapes agree on the freed mid slope to '
            f'{_qe(agree.values, "{:+.4f}")} (n = {len(agree)}) -- a departure that '
            'survives this is in the data, not in the algebra')
  return out


def prescription_check(key=KEY, method=METHOD, zlist=(Z_RS, Z_FS), logt=LOGT, nproc=NPROC,
    outdir=OUTDIR, df=None, verbose=True):
  '''
  Freeze each shape at the pair smoothing_prescription measured on this sweep, refit, and
  report what the freeze costs against the per-spectrum free fit -- the same question
  slope_validation asks of the GS02 pair, now asked of both shapes on the same spectra.

  This is a SECOND pass over the sweep (the pair is not known until the first one is done),
  computing only the 'presc_fit' rows. It re-runs breaks_from_identified, which is most of the
  cost; the two frozen fits themselves are nearly free.
  '''
  df = compare_sweep(key, method, zlist, logt, nproc, outdir) if df is None else df
  _, best = smoothing_prescription(df, outdir=outdir, verbose=False)
  df2 = compare_sweep(key, method, zlist, logt, nproc, outdir, cache=True,
                      configs=('presc_fit',), s_fit_hold=best, csv_name='fits_presc_fit.csv')
  both = pd.concat([df, df2], ignore_index=True)
  piv = both.pivot_table(index=['z', 'logr', 'logt', 'shape'], columns='config', values='rms')
  ok = both[(both.config == 'anchored')].set_index(['z', 'logr', 'logt', 'shape']).ok
  piv = piv[ok.reindex(piv.index).fillna(False).astype(bool)]
  rows = []
  for shape in SHAPES:
    q = piv[piv.index.get_level_values('shape') == shape]
    if not len(q):
      continue
    rows.append(dict(shape=shape, n=len(q),
                     pair_fitted=f'({best[shape][0]:.2f}, {best[shape][1]:.2f})',
                     pair_published=str(PUBLISHED_S[shape]),
                     rms_free=_q(q['anchored']), rms_presc_fit=_q(q['presc_fit']),
                     rms_presc_pub=_q(q['presc']),
                     cost_fitted=_qe(q['presc_fit'] - q['anchored'], '{:+.4f}'),
                     cost_published=_qe(q['presc'] - q['anchored'], '{:+.4f}')))
  out = pd.DataFrame(rows)
  out.to_csv(os.path.join(outdir, 'prescription_check.csv'), index=False)
  if verbose:
    print('\n=== what freezing the smoothing costs (anchored fits, both shells) ===')
    for _, r in out.iterrows():
      print(f"\n  {r['shape']}  (n = {r['n']})")
      print(f"    s free                      rms {r['rms_free']}")
      print(f"    frozen at {r['pair_fitted']:>14} (measured here)  rms {r['rms_presc_fit']}"
            f"   cost {r['cost_fitted']}")
      print(f"    frozen at {r['pair_published']:>14} (published)      rms "
            f"{r['rms_presc_pub']}   cost {r['cost_published']}")
  return out, best


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def coverage_table(df, outdir=OUTDIR, verbose=True):
  '''
  How often the 2SBPL has three segments to describe at all. Its reason to exist in Ravasio
  et al. is a THIRD power-law segment below the peak; on these spectra that segment is in
  band only where the segment route identifies both breaks (TWO_BRK). Everything else is a
  single break -- nu_c below the window (VFC/FC*) or the two knees merged (MC) -- where the
  extra segment describes nothing and the anchored comparison is not run.
  '''
  # df['shape'] by name, NOT df.shape -- that attribute is the DataFrame's dimensions
  g = df[(df.config == 'free') & (df['shape'] == 'gs02')]
  rows = []
  for (z, logr), q in g.groupby(['z', 'logr']):
    d = dict(shell=SHELL.get(z, z), logr=logr, n_epochs=len(q),
             frac_two_break=float(np.mean(q.two_brk.astype(bool))))
    for c in ('VFC', 'FC*', 'FC', 'MC', 'SC', 'VSC'):
      d[c] = int((q.regime == c).sum())
    rows.append(d)
  out = pd.DataFrame(rows)
  os.makedirs(outdir, exist_ok=True)
  out.to_csv(os.path.join(outdir, 'coverage.csv'), index=False)
  if verbose:
    print('\n=== shape class of each epoch (segment route), i.e. where a THIRD segment '
          'exists ===')
    print(out.to_string(index=False, float_format=lambda v: f'{v:.2f}'))
  return out


def plot_compare(df, outdir=OUTDIR):
  '''
  Four panels against the cooling regime: the paired rms of the two shapes (anchored, and
  with each shape's published smoothing frozen), the fitted sharpnesses and the fitted mid
  slope. Markers by shell, colour by shape. Everything but the rms panel is from the ANCHORED
  fits -- the free ones are degenerate in exactly these quantities.
  '''
  os.makedirs(outdir, exist_ok=True)
  fig, axs = plt.subplots(2, 2, figsize=(11., 8.), sharex=True)
  col = {'gs02': 'C0', '2sbpl': 'C1'}
  mk = {4: 'o', 1: 's'}
  free = df[df.config == 'anchored']
  held = df[df.config == 'presc']

  def scat(ax, d, val, ls='-', label=True):
    for (z, sh), g in d.groupby(['z', 'shape']):
      g = g[g.ok.astype(bool)]
      m = g.groupby('logr')[val].median()
      # only the solid pass is labelled: the dashed one draws the same four curves and
      # would double every entry in the legend
      ax.plot(m.index, m.values, ls, marker=mk[z], ms=4, color=col[sh], lw=1.3,
              label=(f'{sh} {SHELL[z]}' if label else None))

  scat(axs[0, 0], free, 'rms')
  scat(axs[0, 0], held, 'rms', ls='--', label=False)
  axs[0, 0].set(ylabel='fit rms [dex]', yscale='log',
                title='goodness of fit, anchored breaks (solid: s fitted,\n'
                      "dashed: s frozen at each shape's published pair)")
  scat(axs[0, 1], free, 'a_mid')
  p = float(df.psyn.iloc[0])
  for y, lab in ((0.5, 'fast 1/2'), ((3. - p)/2., f'slow (3-p)/2')):
    axs[0, 1].axhline(y, color='0.6', lw=.8, ls=':')
    axs[0, 1].text(0.02, y, lab, transform=axs[0, 1].get_yaxis_transform(), fontsize=8,
                   va='bottom', color='0.4')
  axs[0, 1].set(ylabel=r'mid slope, $\nu F_\nu$ index', title='mid slope (anchored)')
  scat(axs[1, 0], free, 's1')
  for y, c, lab in ((S_GS02[0], col['gs02'], 'GS02 1.3'),
                    (S_2SBPL[0], col['2sbpl'], 'Ravasio 5.38')):
    axs[1, 0].axhline(y, color=c, lw=.8, ls=':')
    axs[1, 0].text(0.02, y, lab, transform=axs[1, 0].get_yaxis_transform(), fontsize=8,
                   va='bottom', color=c)
  axs[1, 0].set(ylabel=r'$s_1$ / $n_1$  (lower break)', yscale='log',
                xlabel=r'$\log_{10}\mathcal{C}$', title='sharpness, lower break')
  scat(axs[1, 1], free, 's2')
  for y, c, lab in ((S_GS02[1], col['gs02'], 'GS02 2.0'),
                    (S_2SBPL[1], col['2sbpl'], 'Ravasio 2.69')):
    axs[1, 1].axhline(y, color=c, lw=.8, ls=':')
    axs[1, 1].text(0.02, y, lab, transform=axs[1, 1].get_yaxis_transform(), fontsize=8,
                   va='bottom', color=c)
  axs[1, 1].set(ylabel=r'$s_2$ / $n_2$  (upper break)', yscale='log',
                xlabel=r'$\log_{10}\mathcal{C}$', title='sharpness, upper break')
  axs[0, 0].legend(fontsize=7, ncol=2)
  for ax in axs.ravel():
    ax.grid(alpha=.25)
  fig.suptitle('Ravasio+2018 2SBPL vs Granot & Sari, rarcut sweep '
               '(medians over epochs per point)')
  fig.tight_layout()
  f = os.path.join(outdir, 'shape_compare.png')
  fig.savefig(f, dpi=140); plt.close(fig)
  print(f'saved {f}')


def plot_by_separation(df, config='anchored', outdir=OUTDIR):
  '''
  The paired rms difference against the break separation, with the analytic prediction on the
  same axis: the whole functional difference between the two shapes is the lower break's
  curvature at the upper crossing, so both must die together as the breaks separate.
  '''
  os.makedirs(outdir, exist_ok=True)
  g = df[df.config == config]
  piv = g.pivot_table(index=['z', 'logr', 'logt'], columns='shape',
                      values=['rms', 'sep', 'ok'])
  both = piv[('ok', 'gs02')].astype(bool) & piv[('ok', '2sbpl')].astype(bool)
  piv = piv[both]
  sep = piv[('sep', 'gs02')].values
  dr = (piv[('rms', '2sbpl')] - piv[('rms', 'gs02')]).values

  fig, ax = plt.subplots(figsize=(6.6, 4.6))
  pos, neg = dr > 0., dr <= 0.
  ax.plot(sep[pos], dr[pos], 'o', ms=4, color='C3', label='2SBPL worse')
  ax.plot(sep[neg], -dr[neg], 'o', ms=4, mfc='none', color='C0', label='2SBPL better')
  # over the same span the data covers; the prediction underflows to zero eventually and
  # those points simply cannot be drawn on a log axis
  sd = shape_difference(seps=np.arange(0.4, float(np.nanmax(sep)) + 0.05, 0.1),
                        s_pairs=(S_GS02,), verbose=False)
  sd = sd[sd.dmax > 0.]
  ax.plot(sd.sep, sd.dmax, 'k-', lw=1.2,
          label=r'analytic max$|\Delta\log_{10}|$ at matched parameters')
  ax.set(xscale='linear', yscale='log', xlabel=r'break separation $\log_{10}(b_{hi}/b_{lo})$'
         ' [dex]', ylabel=r'$|\,$rms(2SBPL) $-$ rms(GS02)$\,|$ [dex]',
         title=f'the two shapes differ only where the breaks are close ({config} fits)')
  ax.axhline(float(np.median(piv[('rms', 'gs02')])), color='0.6', lw=.9, ls=':')
  ax.text(0.98, float(np.median(piv[('rms', 'gs02')])), 'median fit rms', ha='right',
          va='bottom', fontsize=8, color='0.4', transform=ax.get_yaxis_transform())
  ax.legend(fontsize=8, loc='lower left'); ax.grid(alpha=.25)
  fig.tight_layout()
  f = os.path.join(outdir, f'rms_vs_separation_{config}.png')
  fig.savefig(f, dpi=140); plt.close(fig)
  print(f'saved {f}')


def plot_mid_slope(df, outdir=OUTDIR):
  '''
  The freed mid slope against the cooling regime, both shapes, with the two one-zone
  asymptotes and the physical window (BMID_DEP) drawn on. The held values are the segment
  route's measured mid line, so the vertical distance between the two curves is exactly what
  the extra freedom bought.
  '''
  os.makedirs(outdir, exist_ok=True)
  p = float(df.psyn.iloc[0])
  fig, axs = plt.subplots(1, 2, figsize=(11., 4.4), sharey=True)
  col = {'gs02': 'C0', '2sbpl': 'C1'}
  mk = {4: 'o', 1: 's'}
  for ax, cfg, ttl in ((axs[0], 'anchored', 'mid slope HELD at the segment route'),
                       (axs[1], 'freemid', 'mid slope FREE (shape limits)')):
    g = df[(df.config == cfg) & df.ok.astype(bool)]
    for (z, sh), q in g.groupby(['z', 'shape']):
      m = q.groupby('logr').a_mid.median()
      ax.plot(m.index, m.values, '-', marker=mk[z], ms=4, color=col[sh], lw=1.3,
              label=f'{sh} {SHELL[z]}')
    ax.axhspan((3. - p)/2. - BMID_DEP, 0.5 + BMID_DEP, color='0.9', zorder=0,
               label='physical window' if cfg == 'anchored' else None)
    for y, lab in ((0.5, 'fast  1/2'), ((3. - p)/2., 'slow  (3-p)/2')):
      ax.axhline(y, color='0.5', lw=.9, ls=':')
      ax.text(0.02, y, lab, transform=ax.get_yaxis_transform(), fontsize=8, va='bottom',
              color='0.35')
    ax.set(xlabel=r'$\log_{10}\mathcal{C}$', title=ttl)
    ax.grid(alpha=.25)
  axs[0].set(ylabel=r'mid slope, $\nu F_\nu$ index')
  axs[0].legend(fontsize=7, ncol=2)
  fig.suptitle('what freeing the mid slope does, per shape (medians over epochs per point)')
  fig.tight_layout()
  f = os.path.join(outdir, 'mid_slope.png')
  fig.savefig(f, dpi=140); plt.close(fig)
  print(f'saved {f}')


def plot_example(key=KEY, method=METHOD, z=Z_RS, logr=0., logt=0., config='anchored',
    outdir=OUTDIR):
  '''
  One cut-off-flattened spectrum with both fitted shapes and their residuals -- the visual
  behind the rms columns. config='anchored' (the default) plots the fits the comparison is
  quoted from, breaks and mid slope held at the segment route's values; 'free' plots the
  six-parameter ones instead, which is worth looking at once to see how differently the two
  forms can be parameterised at the same rms.
  '''
  os.makedirs(outdir, exist_ok=True)
  res = swp.load_sweep(swp.method_outdir(method, key, z))
  r = [q for q in res if abs(q['log10ratio'] - logr) < 1e-9][0]
  del res
  barT_f = swp.exit_onset_barT(key, z=z)
  x = swp.nu_over_num(r); psyn = r['env'].psyn
  ser = swp._spectra_series(r, barT_f, logt=np.array([logt]))
  if not ser:
    print(f'no epoch at logt={logt}')
    return
  sp = r['nuFnu'][ser[0][2], :]
  a1, beta = -2/3., -psyn/2. - 1.

  if config == 'anchored':
    br = sb.breaks_from_identified(x, sp, psyn)
    if not br['ok'] or br['shape'] not in TWO_BRK:
      print(f'z={z} logr={logr} logt={logt}: class {br.get("regime")}/{br.get("shape")} '
            'carries no second break -- nothing to compare, plotting the free fits instead')
      return plot_example(key, method, z, logr, logt, config='free', outdir=outdir)
    fb = (br['regime'] == 'MC')
    xf, yf, _ = flat_window(x, sp, br['nuM'], br['sigma'])
    fits, mods = {}, {}
    for sh, fn in (('gs02', sb.fit_smoothing_held), ('2sbpl', fit_2sbpl_held)):
      f = fn(x, sp, psyn, br['b_lo'], br['b_hi'], br['nuM'], br['a_mid'] - 1.,
             sigma=br['sigma'], free_bmid=fb)
      bh = f['b_hi_fit'] if np.isfinite(f['b_hi_fit']) else br['b_hi']
      if sh == 'gs02':
        m = granot_sari_syn(xf, bh, br['b_lo'], psyn, s1=f['s1'], s2=f['s2'], nuM=None,
                            F_ext=f['F_ext'], nuFnu=True, beta_mid=f['beta_mid'])
      else:
        m = two_sbpl(xf, br['b_lo'], None, a1, f['beta_mid'] - 1., beta, n1=f['s1'],
                     n2=f['s2'], A=f['F_ext'], nuFnu=True, E_j=bh)
      fits[sh], mods[sh] = dict(f, a_mid=f['beta_mid'] + 1.), m
    sub = (f"class {br['regime']}, breaks held at "
           f"$b_{{lo}}={br['b_lo']:.3g}$, sep {np.log10(br['b_hi']/br['b_lo']):.2f} dex")
  else:
    prep = prepare_spectrum(x, sp, psyn)
    if not prep['ok']:
      print('spectrum not usable')
      return
    xf, yf = prep['x'], prep['ly']
    fits = {'gs02': fit_gs02_flat(prep, psyn), '2sbpl': fit_2sbpl_flat(prep, psyn)}
    mods = {}
    for sh, f in fits.items():
      if sh == 'gs02':
        m = granot_sari_syn(xf, f['b_hi'], f['b_lo'], psyn, s1=f['s1'], s2=f['s2'], nuM=None,
                            F_ext=1., nuFnu=True, beta_mid=f['a_mid'] - 1.)
      else:
        m = two_sbpl(xf, f['b_lo'], None, a1, f['a_mid'] - 2., beta, n1=f['s1'], n2=f['s2'],
                     A=1., nuFnu=True, E_j=f['b_hi'])
      mods[sh] = m*10**np.median(yf - np.log10(m))    # the scale is a fit parameter
    sub = 'all six parameters free (degenerate: see compare_spectrum)'

  fig, axs = plt.subplots(2, 1, figsize=(7., 7.5), sharex=True,
                          gridspec_kw=dict(height_ratios=[2.4, 1.]))
  axs[0].plot(xf, 10**yf, 'k.', ms=2.5, label='computed (cut-off flattened)')
  col = {'gs02': 'C0', '2sbpl': 'C1'}
  for sh, f in fits.items():
    axs[0].plot(xf, mods[sh], color=col[sh], lw=1.4,
                label=f"{sh}: rms {f['rms']:.4f}, s=({f['s1']:.2f}, {f['s2']:.2f}), "
                      f"a_mid {f['a_mid']:.3f}")
    axs[1].plot(xf, np.log10(mods[sh]) - yf, color=col[sh], lw=1.2)
  axs[1].axhline(0., color='k', lw=.7)
  axs[0].set(xscale='log', yscale='log', ylabel=r'$\nu F_\nu$ [arb.]',
             title=(f'{SHELL[z]}  ' + r'$\log_{10}\mathcal{C}=$' + f'{logr:+.0f}, '
                    + r'$\log_{10}(\bar T/\bar T_f)=$' + f'{logt:+.1f}  --  {config}\n{sub}'))
  axs[1].set(xlabel=r'$\nu/\nu_{\mathrm{m},0}$', ylabel='model - data [dex]')
  axs[0].legend(fontsize=8); axs[0].grid(alpha=.25); axs[1].grid(alpha=.25)
  fig.tight_layout()
  f = os.path.join(outdir, f'example_{config}_z={z}_logr={logr:+.0f}_logt={logt:+.1f}.png')
  fig.savefig(f, dpi=140); plt.close(fig)
  print(f'saved {f}')
  return fits


def main(key=KEY, method=METHOD, zlist=(Z_RS, Z_FS), logt=LOGT, nproc=NPROC,
    outdir=OUTDIR, cache=True):
  '''
  The whole comparison: the analytic shape difference, the fits over the rarcut sweep, the
  pooled and per-epoch tables, and the figures.
  '''
  os.makedirs(outdir, exist_ok=True)
  sd = shape_difference()
  sd.to_csv(os.path.join(outdir, 'shape_difference.csv'), index=False)
  df = compare_sweep(key, method, zlist, logt, nproc, outdir, cache=cache)
  cv = coverage_table(df, outdir)
  sep = separation_table(df, outdir=outdir)
  st = summary_table(df, outdir)
  et = epoch_table(df, outdir)
  ps, best = smoothing_prescription(df, outdir=outdir)
  ms = mid_slope_table(df, outdir=outdir)
  pc, _ = prescription_check(key, method, zlist, logt, nproc, outdir, df=df)
  plot_compare(df, outdir)
  plot_by_separation(df, outdir=outdir)
  plot_mid_slope(df, outdir)
  # the marginal point at the crossing is where the two forms are furthest apart
  # (separation_table); +2 is the well-separated control, and the free fit of the marginal
  # one shows what the degeneracy does to the SAME spectrum at the same rms
  for logr, cfg in ((0., 'anchored'), (2., 'anchored'), (0., 'free')):
    plot_example(key=key, method=method, z=Z_RS, logr=logr, config=cfg, outdir=outdir)
  print(f'\nfigures and tables in {outdir}')
  return dict(fits=df, summary=st, epoch=et, coverage=cv, separation=sep,
              prescription=ps, mid_slope=ms, presc_check=pc, best_s=best)


if __name__ == '__main__':
  # REQUIRED under forkserver: a worker re-imports this module as __mp_main__, and without
  # the guard the driver's top level would re-run inside the forkserver and stall silently
  # (see cell_pool.pool_context).
  main()
