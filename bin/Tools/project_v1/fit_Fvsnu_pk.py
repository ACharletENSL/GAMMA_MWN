# -*- coding: utf-8 -*-
# Author: acharlet (corrected version following Charlet et al. 2025)
# Date: 11/11/2025

"""
fit_Fvsnu_pk.py (CORRECTED VERSION)

Fit (nu_pk, Fnu_pk) data with parametric model following Charlet et al. 2025,
using smooth broken power-laws for hydrodynamic quantities that naturally
produce the observed spectral shape.

Key physics:
- Peak frequency ν_0 occurs at collision radius R_0
- Spectral decay arises from evolution of Γ_d(R), ν'_m(R), L'_ν'_m(R)
- HLE tail (∝ ν³) dominates at low frequencies
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import least_squares, differential_evolution
import matplotlib.pyplot as plt
import argparse

# ---------------------------
# Model building blocks
# ---------------------------
def smooth_BPL(R, X_pl, X_sph, alpha, beta, s=1.0):
    """
    Smooth broken power-law (Eq. 13 from Charlet et al. 2025):
    X(R) = [(X_pl * R^alpha)^s + (X_sph * R^beta)^s]^(1/s)

    At R << R_transition: X ≈ X_pl * R^alpha
    At R >> R_transition: X ≈ X_sph * R^beta
    """
    R_ratio = R  # R is already normalized to R_0
    term_pl = np.abs(X_pl * (R_ratio**alpha))**s
    term_sp = np.abs(X_sph * (R_ratio**beta))**s
    return np.sign(X_pl) * (term_pl + term_sp)**(1.0/s)

def model_curve(Rgrid, theta):
    """
    Compute nu_pk(R) and nuFnu_pk(R) following the hydrodynamic model.

    theta layout (length 15):
      [G_pl, G_sph, m_gamma,      # Γ_d params (downstream Lorentz factor)
       nu_pl, nu_sph, d_nu,        # ν'_m params (comoving peak frequency)
       L_pl, L_sph, a_L,           # L'_ν' params (comoving luminosity)
       k_xi, xi_sat,               # Effective angle params
       k_nu, k_F, g, s_blend]      # Normalization and blending sharpness

    Returns:
        nu_pk: observed peak frequency at each R
        flux_pk: observed νF_ν at peak at each R
    """
    (G_pl, G_sph, m_gamma,
     nu_pl, nu_sph, d_nu,
     L_pl, L_sph, a_L,
     k_xi, xi_sat,
     k_nu, k_F, g, s_blend) = theta

    # Γ_d(R): downstream Lorentz factor (Eq. 14 in paper)
    alpha_G = -m_gamma/2.0  # Small-R behavior
    beta_G = 0.0            # Asymptotic constant at large R
    Gamma_d = smooth_BPL(Rgrid, G_pl, G_sph, alpha_G, beta_G, s=s_blend)

    # ν'_m(R): comoving peak frequency (Eq. 15)
    alpha_nu = d_nu         # Small-R power law
    beta_nu = -1.0          # Large-R decay (spherical effects)
    nu_prime = smooth_BPL(Rgrid, nu_pl, nu_sph, alpha_nu, beta_nu, s=s_blend)

    # L'_ν'(R): comoving luminosity (Eq. 16)
    alpha_L = a_L           # Small-R power law
    beta_L = 1.0            # Large-R scaling
    L_prime = smooth_BPL(Rgrid, L_pl, L_sph, alpha_L, beta_L, s=s_blend)

    # Effective angle ξ_eff (Section 3.4, related to EATS geometry)
    # ξ = (Γθ)² where θ is angle from line of sight
    # At time T, maximal ξ ≈ g²(T-1) where T is normalized time ≈ R/R_0
    Ttilde = np.maximum(Rgrid, 1.0)
    xi_max = (g**2) * (Ttilde - 1.0)

    # Effective ξ with saturation (Eq. 19 style)
    xi_eff = k_xi * ((1.0/np.maximum(xi_max, 1e-12)) + (1.0/xi_sat))**(-1.0)

    # Observed peak frequency (Eqs. 17 from paper)
    # ν_pk = k_ν * 2Γ_d * ν'_m / (1 + ξ)
    nu_pk = k_nu * (2.0 * Gamma_d * nu_prime) / (1.0 + xi_eff)

    # Observed peak flux (Eq. 18 style)
    # F_pk ∝ g² * (Γ/(1+ξ))³ * L' / Γ²
    flux_pk = k_F * (g**2) * ((Gamma_d/(1.0 + xi_eff))**3) * L_prime / (Gamma_d**2 + 1e-30)

    return nu_pk, flux_pk

# ---------------------------
# HLE blending (corrected)
# ---------------------------
def build_Fnu_with_HLE(nu_mod_grid, F_mod_grid, nu_obs, nu_bk, s_hle=2.0):
    """
    Combine intrinsic model with HLE tail.

    HLE (high-latitude emission): νF_ν ∝ ν³ at LOW frequencies (ν < ν_bk)
    This arises from geometric effects when viewing emission from angles > 1/Γ

    Blending ensures:
    - ν << ν_bk: HLE dominates (∝ ν³ rise)
    - ν ≈ ν_bk: smooth transition
    - ν >> ν_bk: intrinsic model dominates (natural decay)
    """
    # Sort and remove duplicates
    order = np.argsort(nu_mod_grid)
    nu_sorted = nu_mod_grid[order]
    F_sorted = F_mod_grid[order]

    nu_u, idx_first = np.unique(nu_sorted, return_index=True)
    F_u = F_sorted[idx_first]

    if len(nu_u) < 2:
        raise ValueError("Model must generate at least 2 unique frequency points")

    # Interpolate intrinsic model in log space
    interp_logF = interp1d(np.log(nu_u), np.log(F_u),
                          bounds_error=False, fill_value='extrapolate')

    # Intrinsic model at observed frequencies
    F_intrinsic = np.exp(interp_logF(np.log(nu_obs)))

    # HLE tail anchored at ν_bk
    F_at_bk = float(np.exp(interp_logF(np.log(nu_bk))))
    F_hle = F_at_bk * (nu_obs / nu_bk)**3

    # Smooth blending with weight function
    # At ν << ν_bk: w_hle → 1, w_intrinsic → 0
    # At ν >> ν_bk: w_hle → 0, w_intrinsic → 1
    nu_ratio = nu_obs / nu_bk
    w_intrinsic = nu_ratio**s_hle / (1.0 + nu_ratio**s_hle)
    w_hle = 1.0 - w_intrinsic

    F_final = w_intrinsic * F_intrinsic + w_hle * F_hle

    return F_final

# ---------------------------
# Fitting residuals
# ---------------------------
def residuals_with_HLE(params_ext, Rgrid, nu_obs, F_obs):
    """
    Residuals for least-squares fitting.
    params_ext = [theta (len 15), nu_bk, s_hle]
    """
    theta = params_ext[:-2]
    nu_bk = params_ext[-2]
    s_hle = params_ext[-1]

    try:
        nu_mod_grid, F_mod_grid = model_curve(Rgrid, theta)
        F_pred = build_Fnu_with_HLE(nu_mod_grid, F_mod_grid, nu_obs, nu_bk, s_hle=s_hle)
    except Exception as e:
        return np.ones_like(nu_obs) * 1e6

    # Log-space residuals (handles wide dynamic range)
    F_pred = np.maximum(F_pred, 1e-50)
    F_obs_safe = np.maximum(F_obs, 1e-50)

    res = np.log(F_pred) - np.log(F_obs_safe)
    return res

# ---------------------------
# Initial guess and bounds
# ---------------------------
def default_initial_and_bounds(nu_obs, F_obs):
    """
    Physically motivated initial guess and bounds based on observed data.
    """
    i_pk = np.argmax(F_obs)
    peak_nu = nu_obs[i_pk]
    peak_F = F_obs[i_pk]
    min_nu = np.min(nu_obs)
    max_nu = np.max(nu_obs)

    # Initial guess for 15 theta parameters + 2 HLE parameters
    p0_theta = [
        1.0, 1.0, 0.0,                      # Γ_d: G_pl, G_sph, m_gamma
        peak_nu*0.5, peak_nu*0.3, -2.5,     # ν': nu_pl, nu_sph, d_nu
        peak_F*0.8, peak_F*1.2, 1.8,        # L': L_pl, L_sph, a_L
        0.3, 5.0,                           # ξ: k_xi, xi_sat
        1.0, 1.0, 1.5,                      # k_nu, k_F, g
        1.0                                 # s_blend
    ]

    # Bounds (allowing reasonable physical variation)
    lb_theta = [
        0.5, 0.5, -3.0,                     # Γ_d bounds
        min_nu*0.01, min_nu*0.001, -8.0,    # ν' bounds
        peak_F*0.001, peak_F*0.001, 0.5,    # L' bounds
        0.01, 0.5,                          # ξ bounds
        0.1, 0.1, 0.8,                      # normalization bounds
        0.5                                 # s_blend bounds
    ]
    ub_theta = [
        5.0, 5.0, 3.0,
        max_nu*10, max_nu*5, -0.5,
        peak_F*100, peak_F*100, 4.0,
        10.0, 50.0,
        10.0, 10.0, 5.0,
        3.0
    ]

    # HLE parameters
    p0_hle = [peak_nu * 0.7, 2.0]           # nu_bk, s_hle
    lb_hle = [min_nu * 0.3, 0.5]
    ub_hle = [peak_nu * 1.5, 8.0]

    p0 = np.hstack([p0_theta, p0_hle])
    lb = np.hstack([lb_theta, lb_hle])
    ub = np.hstack([ub_theta, ub_hle])

    return p0, lb, ub

# ---------------------------
# Main fitting routine
# ---------------------------
def run_fit(nu_obs, F_obs, out_prefix='fit_nuFnu', do_plot=True, use_global=False):
    """
    Fit GRB spectrum data.

    Args:
        nu_obs: observed frequencies
        F_obs: observed νF_ν fluxes
        use_global: if True, use differential_evolution for global optimization
    """
    # R grid: from collision to well beyond doubling radius
    Rgrid = np.logspace(0, 2.5, 2000)

    p0, lb, ub = default_initial_and_bounds(nu_obs, F_obs)

    print("Fitting GRB spectrum with hydrodynamic model...")
    print(f"Data: {len(nu_obs)} points")
    print(f"Frequency range: {nu_obs.min():.2f} - {nu_obs.max():.2f} keV")
    print(f"Flux range: {F_obs.min():.3g} - {F_obs.max():.3g}")

    if use_global:
        print("\nUsing global optimization (differential_evolution)...")
        result = differential_evolution(
            lambda p: np.sum(residuals_with_HLE(p, Rgrid, nu_obs, F_obs)**2),
            bounds=list(zip(lb, ub)),
            maxiter=500,
            popsize=15,
            disp=True,
            workers=1,
            seed=42
        )
        popt = result.x
        print(f"Global optimization finished. Success: {result.success}")
    else:
        print("\nUsing local optimization (least_squares)...")
        res = least_squares(
            residuals_with_HLE, p0,
            args=(Rgrid, nu_obs, F_obs),
            bounds=(lb, ub),
            xtol=1e-9, ftol=1e-9,
            max_nfev=10000,
            verbose=1
        )
        popt = res.x
        print(f"\nLeast-squares finished. Success: {res.success}")

    # Print results
    param_names = ['G_pl', 'G_sph', 'm_gamma',
                   'nu_pl', 'nu_sph', 'd_nu',
                   'L_pl', 'L_sph', 'a_L',
                   'k_xi', 'xi_sat',
                   'k_nu', 'k_F', 'g', 's_blend',
                   'nu_bk', 's_hle']
    print("\nBest-fit parameters:")
    for name, val in zip(param_names, popt):
        print(f"  {name:10s} = {val:.6g}")

    # Construct best-fit model
    theta_best = popt[:-2]
    nu_bk_best = popt[-2]
    s_hle_best = popt[-1]

    nu_mod_grid, F_mod_grid = model_curve(Rgrid, theta_best)

    # Dense sampling for smooth plot
    nu_plot = np.logspace(
        np.log10(np.min(nu_obs)*0.3),
        np.log10(np.max(nu_obs)*3.0),
        500
    )
    F_plot = build_Fnu_with_HLE(nu_mod_grid, F_mod_grid, nu_plot,
                                nu_bk_best, s_hle=s_hle_best)

    if do_plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Left: fit with data
        ax1.loglog(nu_plot, F_plot, '-', lw=2.5, label='Best-fit model',
                   color='C0', zorder=10)
        ax1.scatter(nu_obs, F_obs, c='red', s=100, marker='o',
                   edgecolors='darkred', linewidths=1.5,
                   label='Data', zorder=15, alpha=0.8)

        # Show components
        interp_logF = interp1d(np.log(nu_mod_grid), np.log(F_mod_grid),
                              bounds_error=False, fill_value='extrapolate')
        F_intrinsic = np.exp(interp_logF(np.log(nu_plot)))
        F_at_bk = float(np.exp(interp_logF(np.log(nu_bk_best))))
        F_hle = F_at_bk * (nu_plot / nu_bk_best)**3

        ax1.loglog(nu_plot, F_intrinsic, '--', alpha=0.6,
                  label='Intrinsic model', color='C2')
        ax1.loglog(nu_plot, F_hle, ':', alpha=0.6,
                  label='HLE tail (∝ν³)', color='C3')
        ax1.axvline(nu_bk_best, color='gray', ls='--', alpha=0.5,
                   label=f'ν_bk = {nu_bk_best:.2f} keV')

        ax1.set_xlabel(r'$\nu$ (keV)', fontsize=12)
        ax1.set_ylabel(r'$\nu F_{\nu}$ (erg cm$^{-2}$ s$^{-1}$)', fontsize=12)
        ax1.legend(fontsize=9)
        ax1.grid(True, which='both', ls=':', alpha=0.3)
        ax1.set_title('Spectrum Fit', fontsize=12)

        # Right: residuals
        F_data_interp = np.exp(interp1d(
            np.log(nu_obs), np.log(F_obs),
            bounds_error=False, fill_value='extrapolate'
        )(np.log(nu_plot)))
        F_pred_at_data = build_Fnu_with_HLE(nu_mod_grid, F_mod_grid, nu_obs,
                                            nu_bk_best, s_hle=s_hle_best)
        residuals = (F_pred_at_data - F_obs) / F_obs

        ax2.semilogx(nu_obs, residuals * 100, 'o', markersize=8,
                     color='C1', markeredgecolor='darkred')
        ax2.axhline(0, color='k', ls='--', alpha=0.5)
        ax2.set_xlabel(r'$\nu$ (keV)', fontsize=12)
        ax2.set_ylabel('Residuals (%)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_title('Fit Residuals', fontsize=12)

        plt.tight_layout()
        fname = out_prefix + '.png'
        plt.savefig(fname, dpi=200)
        print(f"\nPlot saved to {fname}")
        plt.show()

    return popt

# ---------------------------
# Data loading
# ---------------------------
def load_csv_data(fname):
    """Load CSV data with nu, nuFnu columns."""
    with open(fname, 'r') as f:
        first_line = f.readline().strip()

    has_header = not all(c.isdigit() or c in '.,e-+ ' for c in first_line)

    data = np.genfromtxt(fname, delimiter=',',
                        skip_header=1 if has_header else 0)

    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"File must contain two columns: nu, nuFnu")

    nu = data[:, 0].astype(float)
    nuFnu = data[:, 1].astype(float)

    return nu, nuFnu

def synthetic_demo():
    """Generate synthetic data from known parameters."""
    Rgrid = np.logspace(0, 2.0, 1000)

    # True parameters (similar to paper's Table 4)
    true_theta = [
        2.0, 2.5, 0.5,              # Γ_d
        70.0, 40.0, -2.5,           # ν'
        1.5e-7, 2.0e-7, 1.8,        # L'
        0.3, 6.0,                   # ξ
        1.0, 1.0, 1.5,              # normalizations
        1.0                         # s_blend
    ]

    nu_mod, F_mod = model_curve(Rgrid, true_theta)

    # Sample points (more at peak region)
    idx = np.concatenate([
        np.linspace(100, 600, 20, dtype=int),
        np.linspace(600, 900, 10, dtype=int)
    ])

    nu_obs = nu_mod[idx] * (1.0 + 0.05*np.random.randn(len(idx)))
    F_obs = F_mod[idx] * np.exp(0.15*np.random.randn(len(idx)))

    return nu_obs, np.abs(F_obs)

def main():
    parser = argparse.ArgumentParser(
        description="Fit GRB spectrum following Charlet et al. 2025")
    parser.add_argument('--data', type=str, default=None,
                       help='CSV file: nu, nuFnu')
    parser.add_argument('--global', dest='use_global', action='store_true',
                       help='Use global optimization')
    args = parser.parse_args()

    if args.data is None:
        print("Running synthetic demo...")
        nu_obs, F_obs = synthetic_demo()
    else:
        print(f"Loading {args.data}...")
        nu_obs, F_obs = load_csv_data(args.data)

    # Sort data
    sort_idx = np.argsort(nu_obs)
    nu_obs = nu_obs[sort_idx]
    F_obs = F_obs[sort_idx]

    popt = run_fit(nu_obs, F_obs, out_prefix='fit_nuFnu',
                   do_plot=True, use_global=args.use_global)

if __name__ == '__main__':
    main()
