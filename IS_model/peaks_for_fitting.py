# -*- coding: utf-8 -*-
# @Author: acharlet

'''
Public model API for fitting an observed GRB binned peak data (single front).
Fitting strategy:
  1. fit the nu_pk - (nu F_nu)_pk correlation with correlation_track while
    - setting au to a constant value (very low effect of au on the correlation shape)
    - assuming the data points are on the track rather than computing them over bins
    This allows to fit the correlation track for (tau, nu_0, nFmax)
    (nFmax = max peak flux on the track, the physical F0 is recovered after
     the time-data fit via F0_from_nFmax, with au properly determined after step 2.)
  2. Fit nu_pk(t_bins) and (nu F_nu)_pk(t_bins) with model_peaks_binned, 
    using the tau, nu0, nFmax determined at step 1, to determine (au, T0, t_start)
  3. - Convert nFmax to F0 (exact operation once au and tau are known)
     - (TBA) Consistency check against rarefaction wave and other physical checks

Fitting functions:
  - correlation_track:        returns (nu F_nu)_pk in observed units
      as a function of nu_pk in observed units and parameters 
  - model_peaks_binned:       returns [nu_pk, (nu F_nu)_pk) in observed units
      as a function of observed time bins edges and parameters
Useful functions:
  - priors_from_data can be used to define data-informed priors and bounds
  - F0_from_nFmax to obtain physical F0 from fitting parameters 
'''

import numpy as np
from scipy.integrate import cumulative_trapezoid

# time binning functions
def S(x, alpha, beta):
  '''
  Band-function spectral shape nu F_nu / (nu F_nu)_pk, peaking at x = nu/nu_pk = 1
    alpha: slope (in F_nu) below the peak, beta: slope above
  '''
  b  = alpha-beta
  xb = b/(1+alpha)
  def S_inf(x):
    return np.exp(1+alpha) * x**alpha * np.exp(-x*(1+alpha))
  def S_sup(x):
    return np.exp(1+alpha) * x**beta * xb**b * np.exp(-b)
  return np.where(x<=xb, S_inf(x), S_sup(x))

def _bin_grid(au, tau, Tarr, peak_model, reverse=True, N=1000, renormFlux=True):
  '''
  Shared preamble for the per-bin routines: evaluate peak_model once on a single
  N-point grid spanning the full time range, with the bin edges forced in as
  nodes so each bin is an exact sub-slice / integration interval.
  Tarr holds the bin edges (normalized, T=1 at pulse start); need not be sorted.
  Returns (edges, T_grid, nu, nF, idx): edges sorted, and idx the positions of
    edges in T_grid, so idx[k]:idx[k+1] indexes the k-th bin.
  '''
  edges = np.sort(np.asarray(Tarr, dtype=float))
  T_grid = np.union1d(np.geomspace(edges[0], edges[-1], N), edges)
  nu, nF = peak_model(T_grid, au, tau, reverse, renormFlux)
  idx = np.searchsorted(T_grid, edges)
  return edges, T_grid, nu, nF, idx

def correlation_binned(au, tau, Tarr, peak_model, reverse=True,
  N=1000, renormFlux=True):
  '''
  Computes couples (nu_pk/nu0 - nuFnu_pk/nuFnu_max) over the given time bins
  Tarr contains the edges of the time bins, normalized (T=1 at pulse start), sorted.
  In each bin, nu_pk and nuFnu_pk are estimated as means over this bin (flux-weighted for nu)
  Returns (ave_nu, ave_nF), each of length len(Tarr)-1.
    au, tau:     model parameters (proper velocity contrast, t_on/t_off)
    peak_model:  the chosen peak function, peak_modeling_C25/R24.peak_model
    N:           total number of points to evaluate peak_model once over the full range
  '''

  edges, T_grid, nu, nF, idx = _bin_grid(au, tau, Tarr, peak_model, reverse, N, renormFlux)
  F = nF/nu
  # per-bin integrals as differences of the cumulative integral at the edges
  cum_nF = cumulative_trapezoid(nF, T_grid, initial=0.)
  cum_F  = cumulative_trapezoid(F,  T_grid, initial=0.)
  int_nF = cum_nF[idx[1:]] - cum_nF[idx[:-1]]
  int_F  = cum_F[idx[1:]]  - cum_F[idx[:-1]]
  DT = edges[1:] - edges[:-1]
  ave_nF = int_nF/DT       # mean nu F_nu over the bin
  ave_nu = int_nF/int_F    # flux-weighted mean peak frequency
  return ave_nu, ave_nF

def correlation_binned_full(au, tau, Tarr, peak_model, reverse=True,
  N=1000, Nnu=200, alpha=-0.5, beta=-1.25, renormFlux=True):
  '''
  Full-spectrum equivalent of correlation_binned: in each bin the per-instant
    spectra (Band shape S around each (nu_pk, nuFnu_pk)) are superposed and
    integrated in time, then (nu_pk, nuFnu_pk) is read off as the peak of the
    resulting bin-integrated nu F_nu spectrum.
  Tarr contains the edges of the time bins, normalized (T=1 at pulse start), sorted.
  Returns (ave_nu, ave_nF), each of length len(Tarr)-1.
    au, tau:      model parameters (proper velocity contrast, t_on/t_off)
    peak_model:   the chosen peak function, peak_modeling_C25/R24.peak_model
    N:            total number of points to evaluate peak_model once over the full range
    Nnu:          number of frequency points for the spectral grid
    alpha, beta:  Band-function slopes below/above the peak
  '''

  edges, T_grid, nu, nF, idx = _bin_grid(au, tau, Tarr, peak_model, reverse, N, renormFlux)
  # each bin spans its own frequency range, so the spectral grid is per-bin
  Nbin = len(edges) - 1
  ave_nu, ave_nF = np.empty(Nbin), np.empty(Nbin)
  for k in range(Nbin):
    sl = slice(idx[k], idx[k+1] + 1)
    T_bins, nu_bins, nF_bins = T_grid[sl], nu[sl], nF[sl]
    nub = np.logspace(np.log10(nu_bins.min()) - 0.5, np.log10(nu_bins.max()) + 0.5, Nnu)
    x = nub[None, :] / nu_bins[:, None]                  # (Nsub, Nnu)
    nF_full = nF_bins[:, None] * x * S(x, alpha, beta)   # (Nsub, Nnu)
    # nF_int is integrated in time; divide by the bin width to get the
    # time-averaged peak flux, consistent with correlation_binned (int_nF/DT)
    nF_int = np.trapezoid(nF_full, T_bins, axis=0) / (edges[k+1] - edges[k])
    imax = np.argmax(nF_int)
    ave_nu[k], ave_nF[k] = nub[imax], nF_int[imax]
  return ave_nu, ave_nF

def fluence_halftime(au, tau, Tarr, peak_model, reverse=True, N=1000, renormFlux=True):
  '''
  For each bin defined by the edges in Tarr, returns T_half, the time at which half of
    the bin's fluence (time-integrated peak flux nuFnu_pk) has been emitted:
      int_{T_k}^{T_half} nuFnu_pk dT = 0.5 * int_{T_k}^{T_{k+1}} nuFnu_pk dT
  Returns (T_half, nu_half, nF_half), each of length len(Tarr)-1:
    T_half           : half-fluence (median) time in each bin
    nu_half, nF_half : peak_model on the track at T_half (for plotting on the curve)
  Note: T_half is invariant under flux renormalization (cancels in the ratio).
  '''
  edges, T_grid, nu, nF, idx = _bin_grid(au, tau, Tarr, peak_model, reverse, N, renormFlux)
  # cumulative fluence; half-target per bin, inverted by interpolation
  cum = cumulative_trapezoid(nF, T_grid, initial=0.)
  half = 0.5 * (cum[idx[:-1]] + cum[idx[1:]])
  T_half = np.interp(half, cum, T_grid)
  # values on the curve at T_half, from the same fine grid (single peak_model call)
  nu_half = np.interp(T_half, T_grid, nu)
  nF_half = np.interp(T_half, T_grid, nF)
  return T_half, nu_half, nF_half

def photon_halftime(au, tau, Tarr, peak_model, reverse=True, N=1000, renormFlux=True):
  '''
  Photon-flux analogue of fluence_halftime: for each bin in Tarr, returns T_half, the time
    at which half the bin's PHOTON fluence has been emitted. Band-free proxy: the photon rate
    is taken proportional to nuFnu_pk / nu_pk (energy flux per characteristic photon energy):
      int_{T_k}^{T_half} (nuFnu_pk/nu_pk) dT = 0.5 * int_{T_k}^{T_{k+1}} (nuFnu_pk/nu_pk) dT
  Returns (T_half, nu_half, nF_half), each of length len(Tarr)-1 (peak_model on the track at
    T_half, for plotting on the nu_pk / nuFnu_pk curves). T_half is invariant under renormFlux.
  '''
  edges, T_grid, nu, nF, idx = _bin_grid(au, tau, Tarr, peak_model, reverse, N, renormFlux)
  # cumulative photon fluence; half-target per bin, inverted by interpolation
  Nph = nF/nu                                  # photon-rate proxy (= F_nu at the peak)
  cum = cumulative_trapezoid(Nph, T_grid, initial=0.)
  half = 0.5 * (cum[idx[:-1]] + cum[idx[1:]])
  T_half = np.interp(half, cum, T_grid)
  # values on the curve at T_half, from the same fine grid
  nu_half = np.interp(T_half, T_grid, nu)
  nF_half = np.interp(T_half, T_grid, nF)
  return T_half, nu_half, nF_half


# a_u range spanned by the C25 fit tables (fullsweep_au_{RS,FS}.csv):
# au = 1 + 10**log_aum, log_aum in [-1.0, 1.5]. R24 is analytic (no table); pass
# a wider au_range if fitting R24 outside this window.
AU_RANGE = (1.1, 1. + 10**1.5)   # ~ (1.1, 32.62)

def correlation_track(nu_pk, au, tau, nu0, nFmax, peak_model, reverse=True,
                      Tmax=30., N=1000, au_range=AU_RANGE):
  '''
  Correlation model (nu F_nu)_pk as a function of nu_pk, in OBSERVED units, for
  the time-independent fit of (tau, nu0, nFmax) with au held fixed.

  Builds the parametric track (nu_pk, nuFnu_pk) from peak_model over T in
  [1, 1+Tmax], scales it to observed units (nu0 on the frequency, nFmax on the
  flux), and interpolates nuFnu_pk at the requested nu_pk. The track is
  normalized to the maximal observed value nFmax to avoid introducing a
  systematic by fixing au (nFmax = f(au, tau, nu0 F0)).

  nu_pk:               observed peak frequencies at which to evaluate the model.
  au, tau, nu0, nFmax:  au fixed during this step; nu0, nFmax are the (x, y) scales.
  peak_model:          peak_modeling_C25/R24.peak_model.
  Tmax, N:             the parametric track spans T in [1, 1+Tmax] on N points.

  Returns nuFnu_pk at each nu_pk in observed units, np.nan outside the track
  range or on any invalid parameter / model failure (never raises), so the
  caller's log-likelihood can return -inf.
  '''

  nu_pk = np.asarray(nu_pk, dtype=float)
  bad = np.full(nu_pk.shape, np.nan)
  if not (np.isfinite([au, tau, nu0, nFmax]).all()
          and au_range[0] <= au <= au_range[1]
          and tau > 0. and nu0 > 0. and nFmax > 0.):
    return bad

  T = np.geomspace(1., 1. + Tmax, N)
  try:
    nu_track, nF_track = peak_model(T, au, tau, reverse, renormFlux=True)
  except ValueError:
    return bad
  if not (np.isfinite(nu_track).all() and np.isfinite(nF_track).all()):
    return bad

  # parametric track in observed units
  nu_obs = nu0 * nu_track
  nF_obs = nFmax * nF_track

  # nu_pk is monotonic along the track over the whole (au, tau) range, so the
  # nu_pk -> nuFnu_pk mapping is single-valued; argsort orients it increasing
  # (nu_pk decreases with T) as np.interp requires.
  order = np.argsort(nu_obs)
  return np.interp(nu_pk, nu_obs[order], nF_obs[order], left=np.nan, right=np.nan)

def model_peaks_binned(t_edges, au, tau, T0, nu0, nFmax, peak_model,
                       t_start=0., reverse=True, mode='fast',
                       alpha=-0.5, beta=-1.25,
                       N=1000, Nnu=200, au_range=AU_RANGE):
  '''
  Bin-integrated peak frequency and flux in OBSERVED units, one value per bin,
  ready to feed an MCMC log-likelihood against the binned data.

    t_edges:     edges of the observed-time bins in ABSOLUTE detector time,
                   length Nbin+1 (need not be pre-sorted).
    au, tau:     proper velocity contrast, t_on/t_off.
    T0:          angular time scale (same units as t_edges).
    nu0:         peak frequency normalisation (nu_pk = nu0 * nu~).
    nFmax:       maximal peak flux along the track. The physical normalization F0
                  is recovered at the end of the procedure
    peak_model:  peak_modeling_C25/R24.peak_model.
    t_start:     pulse onset in absolute detector time.
    reverse:     True for the reverse shock, False for the forward shock.
    mode:        'full' -> peak of the bin-integrated Band spectrum
                   (matches how the data points are produced; uses alpha,beta),
                 'fast' -> flux-weighted mean over the bin
                   (~% accurate for narrow bins, much cheaper).
    alpha,beta:  Band slopes below/above the peak, used by mode='full'; pass the
                   values assumed in the collaborators' spectral fits.
    N, Nnu:      grid resolutions forwarded to the binning routines.
    au_range:    (min, max) allowed au; outside -> all-nan (rejected step).

  Returns (nu_pk, nuFnu_pk), each of length len(t_edges)-1, in observed units.
  On any invalid parameter, model failure, or non-finite result, returns arrays
  of np.nan (never raises), so the caller's log-likelihood can return -inf.
  '''

  t_edges = np.sort(np.asarray(t_edges, dtype=float))
  nbin = max(t_edges.size - 1, 0)
  bad = (np.full(nbin, np.nan), np.full(nbin, np.nan))

  # parameter validity: reject rather than crash the chain. The onset may lie
  # anywhere before the last edge (incl. inside the first bin); require only that
  # some emission falls in the binned range.
  if not (np.isfinite([au, tau, T0, nu0, nFmax, t_start]).all()
          and au_range[0] <= au <= au_range[1]
          and tau > 0. and T0 > 0. and nu0 > 0. and nFmax > 0.
          and nbin >= 1 and t_start < t_edges.max()):
    return bad

  # absolute detector time -> normalized time
  # no emission before onset (T < 1)
  T_edges = 1. + (t_edges - t_start) / T0
  T_emit  = np.maximum(T_edges, 1.)
  DT_full = np.diff(T_edges)      # full normalized bin widths
  DT_emit = np.diff(T_emit)       # emitting (T>=1) widths; < DT_full for a straddling bin
  emit = DT_emit > 0.             # bin has some emission (its upper edge is at T > 1)

  with np.errstate(invalid='ignore', divide='ignore'):
    try:
      if mode == 'full':
        ave_nu, ave_nF = correlation_binned_full(
          au, tau, T_emit, peak_model, reverse=reverse,
          N=N, Nnu=Nnu, alpha=alpha, beta=beta, renormFlux=True)
      elif mode == 'fast':
        ave_nu, ave_nF = correlation_binned(
          au, tau, T_emit, peak_model, reverse=reverse,
          N=N, renormFlux=True)
      else:
        raise ValueError(f"mode must be 'full' or 'fast', got {mode!r}")
    except ValueError:
      # e.g. brentq bracket without a sign change at extreme (au, tau)
      return bad

    # rescale ave_nF to full bin width (rather than emitting bin width) to match data
    # Fully dark bins (emit=False) -> nan.
    frac = np.where(emit, DT_emit / DT_full, np.nan)   # emitting fraction of the bin
  ave_nF = ave_nF * frac
  ave_nu = np.where(emit, ave_nu, np.nan)

  # scale to observed units: nu_pk = nu0 * nu~, nuFnu = nFmax * phi_hat
  nu_pk = nu0 * ave_nu
  nF_pk = nFmax * ave_nF
  if not (np.isfinite(nu_pk).all() and np.isfinite(nF_pk).all()):
    return bad
  return nu_pk, nF_pk

#### Assisting functions
def phi_max(au, tau, peak_model, reverse=True, Tmax=30., N=4000, au_range=AU_RANGE):
  '''
  max(phi~), the peak of the raw normalized flux track (renormFlux=False). This
  is the au,tau-dependent factor relating the two flux normalisations:
      nFmax = nu0 * F0 * phi_max      <->      F0 = nFmax / (nu0 * phi_max)
  Returns np.nan on invalid parameters / a break at the scan boundary.
  '''
  if not (np.isfinite([au, tau]).all()
          and au_range[0] <= au <= au_range[1] and tau > 0.):
    return np.nan
  T = np.geomspace(1., 1. + Tmax, N)
  try:
    _, nF_track = peak_model(T, au, tau, reverse, renormFlux=False)
  except ValueError:
    return np.nan
  if not np.isfinite(nF_track).all():
    return np.nan
  imax = int(np.nanargmax(nF_track))
  if imax == N - 1:
    return np.nan
  return float(nF_track[imax])


def F0_from_nFmax(au, tau, nu0, nFmax, peak_model, reverse=True, **kw):
  '''
  Physical normalisation F0 from the fitted nFmax: F0 = nFmax / (nu0 * max(phi~)).
  Apply AFTER the fit, with the (au, tau) posterior, so the au,tau-dependent
  max(phi~) factor is evaluated at the best-determined shape and its uncertainty
  propagates into F0 (rather than being frozen at a guessed au in step 1).
  Accepts arrays for au, tau, nu0, nFmax (elementwise). Returns nan where phi_max
  is undefined. Extra kwargs (Tmax, N, au_range) forward to phi_max.
  '''
  au, tau, nu0, nFmax = np.broadcast_arrays(
    *(np.asarray(v, dtype=float) for v in (au, tau, nu0, nFmax)))
  pm = np.array([phi_max(a, t, peak_model, reverse=reverse, **kw)
                 for a, t in zip(au.ravel(), tau.ravel())]).reshape(au.shape)
  F0 = nFmax / (nu0 * pm)
  return F0 if F0.ndim else float(F0)


def nFmax_from_F0(au, tau, nu0, F0, peak_model, reverse=True, **kw):
  '''
  Inverse of F0_from_nFmax: nFmax = nu0 * F0 * max(phi~). See F0_from_nFmax.
  '''
  au, tau, nu0, F0 = np.broadcast_arrays(
    *(np.asarray(v, dtype=float) for v in (au, tau, nu0, F0)))
  pm = np.array([phi_max(a, t, peak_model, reverse=reverse, **kw)
                 for a, t in zip(au.ravel(), tau.ravel())]).reshape(au.shape)
  nFmax = nu0 * F0 * pm
  return nFmax if nFmax.ndim else float(nFmax)


def priors_from_data(t_edges, nu_data, nF_data, peak_model, reverse=True,
                     au_range=AU_RANGE, tau_range=(0.1, 10.), pad=2.):
  '''
  Data-driven priors for the 6 fit parameters (au, tau, T0, nu0, nFmax, t_start):
  it inverts the forward model, turning the measured peaks into parameter
  ranges + starting points for the MCMC.

  Robust data features used (nu_pk decreases monotonically, flux peaks at the
  break):
    nu_hi  = max(nu_data)                      -> onset frequency ~ nu0
    nF_hi  = max(nF_data) at bin i_pk          -> peak flux ~ nFmax
    t_pk   = centre time of the flux-peak bin  -> break time ~ t_start + Tr
  nFmax is bracketed DIRECTLY by nF_hi (no max(phi~) factor) -- that is the whole
  point of using nFmax over F0: the flux prior needs no shape assumption. The
  au,tau-dependent shape factors (Tf, nu_pk(Tf)/nu0) are still bracketed over a
  coarse (au,tau) grid (see _shape_ranges) for the T0 and nu0 inversions.

    t_edges:    absolute detector-time bin edges, length Nbin+1.
    nu_data,
    nF_data:    measured peak frequency and flux per bin, length Nbin (in the
                  SAME observed units the model returns: nu_pk and nuFnu_pk).
    peak_model: peak_modeling_C25/R24.peak_model .
    au_range,
    tau_range:  hard bounds for au and tau (physical / fit-table limits).
    pad:        multiplicative widening applied to the scale-parameter bounds
                  (nu0, nFmax, T0) to stay conservative; >= 1.

  Returns a dict keyed by parameter name, each mapping to
    {'low', 'high', 'init', 'sample'}
  where 'sample' suggests the MCMC parametrisation ('log10', 'log10(x-1)',
  'linear'). Also includes a 'features' entry echoing the extracted data
  features and shape ranges for transparency.
  '''

  edges = np.sort(np.asarray(t_edges, dtype=float))
  nu_data = np.asarray(nu_data, dtype=float)
  nF_data = np.asarray(nF_data, dtype=float)
  centres = 0.5 * (edges[:-1] + edges[1:])
  w0 = edges[1] - edges[0]

  # data features (robust to nan bins)
  if not (np.isfinite(nu_data).any() and np.isfinite(nF_data).any()):
    raise ValueError("nu_data / nF_data have no finite values to build priors from")
  nu_hi = float(np.nanmax(nu_data))
  i_pk = int(np.nanargmax(nF_data))
  nF_hi = float(nF_data[i_pk])
  t_pk = float(centres[i_pk])

  # au,tau-dependent shape factor ranges
  (dTf_lo, dTf_hi), (Rnu_lo, Rnu_hi), (phi_lo, phi_hi) = _shape_ranges(
    peak_model, reverse, au_range, tau_range)

  # --- t_start: onset expected within (or just before) the first bin ---
  ts_lo, ts_hi = edges[0] - w0, edges[1]
  ts_init = 0.5 * (edges[0] + edges[1])

  # --- nu0: nu_pk = nu0 * nu~, nu~ in [Rnu, 1] on the rise, so
  #     nu0 in [nu_hi, nu_hi/Rnu_lo] ---
  nu0_lo, nu0_hi = nu_hi, nu_hi / max(Rnu_lo, 1e-3)
  nu0_init = nu_hi * 1.2

  # --- T0: Tr = t_pk - t_start = T0*(Tf-1) -> T0 = Tr/(Tf-1) ---
  Tr = max(t_pk - ts_init, w0)          # keep positive if the flux peaks early
  T0_lo, T0_hi = Tr / dTf_hi / pad, Tr / dTf_lo * pad
  T0_init = Tr / np.sqrt(dTf_lo * dTf_hi)

  # --- nFmax: the peak flux is the top of the light curve / correlation hook.
  #     Binning dilutes the observed peak below the true one, so nF_hi is a lower
  #     bound; no max(phi~) factor is needed (that is the advantage over F0). ---
  nFmax_lo, nFmax_hi = nF_hi, nF_hi * pad
  nFmax_init = nF_hi * 1.2

  return {
    'au':      {'low': au_range[0], 'high': au_range[1],
                'init': 1. + np.sqrt((au_range[0]-1.)*(au_range[1]-1.)),
                'sample': 'log10(x-1)'},
    'tau':     {'low': tau_range[0], 'high': tau_range[1],
                'init': float(np.sqrt(tau_range[0]*tau_range[1])),
                'sample': 'log10'},
    'T0':      {'low': T0_lo, 'high': T0_hi, 'init': T0_init, 'sample': 'log10'},
    'nu0':     {'low': nu0_lo, 'high': nu0_hi, 'init': nu0_init, 'sample': 'log10'},
    'nFmax':    {'low': nFmax_lo, 'high': nFmax_hi, 'init': nFmax_init, 'sample': 'log10'},
    't_start': {'low': ts_lo, 'high': ts_hi, 'init': ts_init, 'sample': 'linear'},
    'features': {
      'nu_hi': nu_hi, 'nF_hi': nF_hi, 't_pk': t_pk, 'i_pk': i_pk, 'w0': w0,
      'dTf_range': (dTf_lo, dTf_hi), 'Rnu_range': (Rnu_lo, Rnu_hi),
      'phi_range': (phi_lo, phi_hi),
    },
  }


def _shape_ranges(peak_model, reverse, au_range, tau_range, nau=6, ntau=6,
                  Tmax=30., N=4000):
  '''
  Ranges of the au,tau-dependent SHAPE factors over a coarse grid. For each
  (au, tau) the break Tf is located model-agnostically as the argmax of the
  un-normalised peak flux along the track (valid for both C25 and R24), and the
  normalized-track factors are read off there. Returns (min,max) tuples for:
    dTf   = Tf - 1               (break duration in T0 units)
    Rnu   = nu_pk(Tf)/nu0        (break frequency / onset frequency)
    phi   = max(phi~)            (peak of the normalized flux track)
  '''
  aus  = 1. + np.geomspace(au_range[0] - 1., au_range[1] - 1., nau)
  taus = np.geomspace(tau_range[0], tau_range[1], ntau)
  T = np.geomspace(1., 1. + Tmax, N)
  dTf, Rnu, phi = [], [], []
  for au in aus:
    for tau in taus:
      try:
        nu_track, nF_track = peak_model(T, au, tau, reverse, renormFlux=False)
      except ValueError:
        continue
      if not np.isfinite(nF_track).all():
        continue
      imax = int(np.nanargmax(nF_track))
      if imax == N - 1:            # break not captured within [1, 1+Tmax]
        continue
      dTf.append(T[imax] - 1.)     # nu_track is already nu_pk/nu0
      Rnu.append(float(nu_track[imax]))
      phi.append(float(nF_track[imax]))
  rng = lambda v: (float(np.min(v)), float(np.max(v)))
  return rng(dTf), rng(Rnu), rng(phi)


