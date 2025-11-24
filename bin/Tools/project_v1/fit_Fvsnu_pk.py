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
    
    Args:
        R: Normalized radius R/R_0 (MUST be ≥ 1.0)
        X_pl: Amplitude for planar (small-R) regime
        X_sph: Amplitude for spherical (large-R) regime  
        alpha: Power-law index at small R (R → 1)
        beta: Power-law index at large R (R → ∞)
        s: Blending sharpness (smaller = smoother transition)
    
    At R = R_0 = 1: X(1) = [X_pl^s + X_sph^s]^(1/s)
    At R << R_transition: X ≈ X_pl * R^alpha
    At R >> R_transition: X ≈ X_sph * R^beta
    """
    R_ratio = R  # R is already normalized to R_0
    s *= np.sign(beta-alpha)
    term_pl = np.abs(X_pl * (R_ratio**alpha))**s
    term_sp = np.abs(X_sph * (R_ratio**beta))**s
    return np.sign(X_pl) * (term_pl + term_sp)**(1.0/s)

def model_curve(Rgrid, theta):
    """
    Compute nu_pk(R) and F_nu_pk(R) following the hydrodynamic model.
    
    IMPORTANT: Returns F_ν (not νF_ν) at the peak frequency.
    
    Following Charlet et al. 2025 Equations 17-18:
    - Hydrodynamic functions (Γ_d, ν'_m, L') evaluated at R_eff (effective radius)
    - R_eff = y_eff * R_L where y_eff depends on ξ_eff
    - For this implementation, Rgrid represents R_L (line of sight radius)
    
    NORMALIZED PARAMETERIZATION:
    - Γ_d, ν'_m, L'_ν' are normalized to have value ≈ 1 at R_0 (Rgrid[0] = 1)
    - k_nu and k_F absorb the physical normalizations
    
    Args:
        Rgrid: R_L/R_0 - Line of sight radii normalized to R_0. 
               MUST have Rgrid[0] = 1.0 (collision radius)
               These represent the maximum radius on the EATS at each observed time.
        theta: parameter vector (length 15)
    
    theta layout (length 15):
      [G_pl, G_sph, m_gamma,      # Γ_d params: Γ_d(R_0) = G_pl * G_sph^(1/s) ≈ 1
       nu_pl, nu_sph, d_nu,        # ν'_m params: ν'_m(R_0) = nu_pl * nu_sph^(1/s) ≈ 1  
       L_pl, L_sph, a_L,           # L'_ν' params: L'(R_0) = L_pl * L_sph^(1/s) ≈ 1
       k_xi, xi_sat,               # Effective angle params
       k_nu, k_F, g, s_blend]      # k_nu ~ max_nu, k_F ~ peak_Fnu (absorb normalizations)
    
    Returns:
        nu_pk: observed peak frequency at each R_L [keV]
        F_nu_pk: observed F_ν at peak at each R_L [erg/cm²/s/keV]
    """
    # Validate input
    if np.any(Rgrid < 1.0):
        raise ValueError(f"Rgrid contains values < 1.0 (R/R_0 < 1 is unphysical). "
                        f"Min value: {np.min(Rgrid):.3f}")
    
    (G_pl, G_sph, m_gamma,
     nu_pl, nu_sph, d_nu,
     L_pl, L_sph, a_L,
     k_xi, xi_sat,
     k_nu, k_F, g, s_blend) = theta

    # NOTE: Rgrid represents R_L (line of sight radius on EATS)
    # Normalized time T̃ ≈ R_L/R_0 (approximately, for constant Γ_sh)
    Ttilde = np.maximum(Rgrid, 1.0)
    
    # Effective angle ξ_eff (Section 3.4, Eq. 19)
    # ξ_max = g²(T̃ - 1) represents maximum angle on EATS
    xi_max = (g**2) * (Ttilde - 1.0)
    
    # Effective ξ with saturation
    xi_eff = k_xi * (np.maximum(xi_max, 1e-12)**(-s_blend) + xi_sat**(-s_blend))**(-1.0/s_blend)
    
    # Effective radius factor y_eff (from Eq. 19 context)
    # y_eff = R_eff/R_L = [1 + g^(-2)(m+1)ξ]^(-1/(m+1))
    # For simplicity, using approximate form with m = m_gamma
    m_for_y = m_gamma if abs(m_gamma) > 1e-8 else 0.0
    y_eff = (1.0 + (m_for_y + 1.0) * (g**(-2)) * xi_eff)**(-1.0/(m_for_y + 1.0))
    
    # CRITICAL: R_eff = y_eff * R_L
    # Hydrodynamic functions evaluated at R_eff, not R_L!
    R_eff = y_eff * Rgrid
    
    # Γ_d(R_eff): downstream Lorentz factor evaluated at effective radius
    alpha_G = -m_gamma/2.0
    beta_G = 0.0
    Gamma_d_norm = smooth_BPL(R_eff, G_pl, G_sph, alpha_G, beta_G, s=s_blend)

    # ν'_m(R_eff): comoving peak frequency evaluated at effective radius
    alpha_nu = d_nu
    beta_nu = -1.0
    nu_prime_norm = smooth_BPL(R_eff, nu_pl, nu_sph, alpha_nu, beta_nu, s=s_blend)

    # L'_ν'(R_eff): comoving luminosity evaluated at effective radius
    alpha_L = a_L
    beta_L = 1.0
    L_prime_norm = smooth_BPL(R_eff, L_pl, L_sph, alpha_L, beta_L, s=s_blend)

    # Observed peak frequency (Eq. 17)
    # ν_pk = k_ν * 2Γ_d(R_eff) * ν'_m(R_eff) / (1 + ξ_eff)
    nu_pk = k_nu * (2.0 * Gamma_d_norm * nu_prime_norm) / (1.0 + xi_eff)

    # Observed F_ν at peak (Eq. 18)
    # F_pk ∝ ξ_max * g² * (Γ(R_eff)/(1+ξ))³ * L'(R_eff) / Γ(R_L)²
    # The ξ_max factor ensures F = 0 at R_0 (where ξ_max = 0)
    # This represents the solid angle of the EATS contributing to emission
    # Note: The Γ²(R_L) in denominator represents the beaming factor
    #       evaluated at the line-of-sight radius
    Gamma_d_at_RL = smooth_BPL(Rgrid, G_pl, G_sph, alpha_G, beta_G, s=s_blend)
    
    F_nu_pk = k_F * xi_max * (g**2) * ((Gamma_d_norm/(1.0 + xi_eff))**3) * L_prime_norm / (Gamma_d_at_RL**2 + 1e-30)

    return nu_pk, F_nu_pk

# ---------------------------
# HLE blending (corrected)
# ---------------------------
def build_Fnu_with_HLE(nu_mod_grid, Fnu_mod_grid, nu_obs, nu_bk, s_hle=2.0):
    """
    Combine intrinsic model with HLE tail.
    
    IMPORTANT: Works with F_ν (not νF_ν)
    
    HLE (high-latitude emission): F_ν ∝ ν² at LOW frequencies (ν < ν_bk)
    This corresponds to νF_ν ∝ ν³ in the commonly plotted form.
    
    Physical origin: When viewing emission from angles > 1/Γ, geometric effects
    cause the observed flux to follow this power law.
    
    Blending ensures:
    - ν << ν_bk: HLE dominates (F_ν ∝ ν² rise)
    - ν ≈ ν_bk: smooth transition
    - ν >> ν_bk: intrinsic model dominates (natural decay)
    
    Args:
        nu_mod_grid: model frequencies (from model_curve)
        Fnu_mod_grid: model F_ν values (NOT νF_ν)
        nu_obs: observed frequencies where we want F_ν
        nu_bk: transition frequency
        s_hle: blending sharpness
    
    Returns:
        Fnu_final: F_ν at nu_obs with HLE blending
    """
    # Sort and remove duplicates
    order = np.argsort(nu_mod_grid)
    nu_sorted = nu_mod_grid[order]
    Fnu_sorted = Fnu_mod_grid[order]
    
    nu_u, idx_first = np.unique(nu_sorted, return_index=True)
    Fnu_u = Fnu_sorted[idx_first]
    
    if len(nu_u) < 2:
        raise ValueError("Model must generate at least 2 unique frequency points")
    
    # Interpolate intrinsic model in log space
    # Working with F_ν here, not νF_ν
    interp_logFnu = interp1d(np.log(nu_u), np.log(Fnu_u), 
                            bounds_error=False, fill_value='extrapolate')
    
    # Intrinsic model F_ν at observed frequencies
    Fnu_intrinsic = np.exp(interp_logFnu(np.log(nu_obs)))
    
    # HLE tail anchored at ν_bk
    # F_ν(ν_bk) from intrinsic model
    Fnu_at_bk = float(np.exp(interp_logFnu(np.log(nu_bk))))
    
    # HLE tail: F_ν ∝ ν² (so νF_ν ∝ ν³)
    Fnu_hle = Fnu_at_bk * (nu_obs / nu_bk)**2
    
    # Smooth blending with weight function
    # At ν << ν_bk: w_hle → 1, w_intrinsic → 0
    # At ν >> ν_bk: w_hle → 0, w_intrinsic → 1
    nu_ratio = nu_obs / nu_bk
    w_intrinsic = nu_ratio**s_hle / (1.0 + nu_ratio**s_hle)
    w_hle = 1.0 - w_intrinsic
    
    Fnu_final = w_intrinsic * Fnu_intrinsic + w_hle * Fnu_hle
    
    return Fnu_final

# ---------------------------
# Fitting residuals
# ---------------------------
def residuals_with_HLE(params_ext, Rgrid, nu_obs, Fnu_obs):
    """
    Residuals for least-squares fitting with physical constraints.
    
    IMPORTANT: Works with F_ν (not νF_ν)
    
    params_ext = [theta (len 15), nu_bk, s_hle]
    
    Physical constraint: ν_pk(R_0) must be ≥ max(ν_obs)
    This ensures the peak is not below the observed frequency range,
    as emission originates at R_0 where ν_pk is highest.
    
    Args:
        params_ext: parameter vector
        Rgrid: radii grid (normalized to R_0)
        nu_obs: observed frequencies [keV]
        Fnu_obs: observed F_ν [erg/cm²/s/keV] (NOT νF_ν)
    """
    theta = params_ext[:-2]
    nu_bk = params_ext[-2]
    s_hle = params_ext[-1]
    
    try:
        nu_mod_grid, Fnu_mod_grid = model_curve(Rgrid, theta)
        
        # Physical constraint: ν_pk at R_0 (Rgrid[0] = 1) must be ≥ max(ν_obs)
        nu_pk_at_R0 = nu_mod_grid[0]  # Peak frequency at collision radius
        nu_obs_max = np.max(nu_obs)
        
        if nu_pk_at_R0 < nu_obs_max:
            # Constraint violated: add large penalty
            # Penalty scales with how badly constraint is violated
            violation_factor = nu_obs_max / nu_pk_at_R0
            penalty = np.ones_like(nu_obs) * 10.0 * np.log(violation_factor)
            return penalty
        
        Fnu_pred = build_Fnu_with_HLE(nu_mod_grid, Fnu_mod_grid, nu_obs, nu_bk, s_hle=s_hle)
    except Exception as e:
        return np.ones_like(nu_obs) * 1e6
    
    # Log-space residuals (handles wide dynamic range)
    Fnu_pred = np.maximum(Fnu_pred, 1e-50)
    Fnu_obs_safe = np.maximum(Fnu_obs, 1e-50)
    
    res = np.log(Fnu_pred) - np.log(Fnu_obs_safe)
    return res

# ---------------------------
# Initial guess and bounds
# ---------------------------
def default_initial_and_bounds(nu_obs, Fnu_obs):
    """
    Physically motivated initial guess and bounds based on observed data.
    
    IMPORTANT: Input is F_ν (not νF_ν)
    
    NEW PARAMETERIZATION:
    - Hydrodynamic quantities (Γ_d, ν', L') normalized to ≈ 1 at R_0
    - k_nu and k_F absorb physical scales from data
    
    Key physical constraint: ν_pk(R_0) ≥ max(ν_obs)
    With normalized params: k_nu * 2 * Γ_d(R_0) * ν'(R_0) ≥ max(ν_obs)
    
    Note: F_ν(R_0) = 0 due to ξ_max = 0, so k_F scaled from flux at R > R_0
    """
    i_pk = np.argmax(Fnu_obs)
    peak_nu = nu_obs[i_pk]
    peak_Fnu = Fnu_obs[i_pk]
    min_nu = np.min(nu_obs)
    max_nu = np.max(nu_obs)
    
    # For normalized quantities at R_0 = 1:
    # X(R_0) = [X_pl^s + X_sph^s]^(1/s)
    # To get X(R_0) ≈ 1, we can use X_pl ≈ X_sph ≈ 1
    
    # Initial guess for 15 theta parameters + 2 HLE parameters
    # Params for Γ_d, ν', L' are chosen such that f(R_0) ≈ 1.
    p0_theta = [
        1.0, 1.1, 0.5,                      # Γ_d: G_pl, G_sph, m_gamma
        0.8, 1.2, -2.5,                     # ν': nu_pl, nu_sph, d_nu
        1.0, 1.2, 1.8,                      # L': L_pl, L_sph, a_L
        0.3, 5.0,                           # ξ: k_xi, xi_sat
        max_nu * 1.5,                       # k_nu: set to max_nu * safety factor
        peak_Fnu * 0.1,                     # k_F: F_ν scal
                                             # Set lower since ξ_max multiplies this
        1.5,                                # g: Γ/Γ_sh ratio
        1.0                                 # s_blend
    ]
    
    # Bounds: normalized quantities should stay O(1), k_nu/k_F absorb scales
    lb_theta = [
        0.5, 0.5, 0.01,                     # Γ_d bounds: keep normalized near 1
        0.3, 0.3, -8.0,                     # ν' bounds: keep normalized near 1
        0.5, 0.5, 0.5,                      # L' bounds: keep normalized near 1
        0.01, 0.5,                          # ξ bounds
        max_nu * 0.8,                       # k_nu: must be large enough for constraint
        peak_Fnu * 0.001,                   # k_F: lower bound (ξ_max multiplies this)
        0.8,                                # g bounds
        0.5                                 # s_blend bounds
    ]
    ub_theta = [
        3.0, 3.0, 3.0,                      # Γ_d: allow modest variation from 1
        3.0, 3.0, -0.5,                     # ν': allow modest variation from 1
        3.0, 3.0, 4.0,                      # L': allow modest variation from 1
        10.0, 50.0,                         # ξ bounds
        max_nu * 10.0,                      # k_nu: physical frequency scale
        peak_Fnu * 10.0,                    # k_F: higher upper bound (ξ_max < 1 typically)
        5.0,                                # g upper bound
        3.0                                 # s_blend upper bound
    ]
    
    # HLE parameters
    p0_hle = [peak_nu * 0.7, 2.0]           # nu_bk, s_hle
    lb_hle = [min_nu * 0.3, 0.5]
    ub_hle = [max_nu * 1.5, 8.0]            # Allow nu_bk up to max_nu
    
    p0 = np.hstack([p0_theta, p0_hle])
    lb = np.hstack([lb_theta, lb_hle])
    ub = np.hstack([ub_theta, ub_hle])
    
    return p0, lb, ub

def extract_physical_parameters(theta, reference_values=None):
    """
    Extract physical parameters from fitted normalized parameters.
    
    Args:
        theta: fitted parameter vector (length 15)
        reference_values: dict with 'L', 't_off', 'z', 'd_L', 'u_1', 'a_u'
                         If None, returns normalized quantities only
    
    Returns:
        dict with physical and normalized parameters
    """
    G_pl, G_sph, m_gamma = theta[0:3]
    nu_pl, nu_sph, d_nu = theta[3:6]
    L_pl, L_sph, a_L = theta[6:9]
    k_xi, xi_sat = theta[9:11]
    k_nu, k_F, g, s_blend = theta[11:15]
    
    # Normalized values at R_0 (should be ~1)
    Gamma_d_R0 = (G_pl**s_blend + G_sph**s_blend)**(1.0/s_blend)
    nu_prime_R0 = (nu_pl**s_blend + nu_sph**s_blend)**(1.0/s_blend)
    L_prime_R0 = (L_pl**s_blend + L_sph**s_blend)**(1.0/s_blend)
    
    result = {
        'normalized': {
            'Gamma_d_R0': Gamma_d_R0,
            'nu_prime_R0': nu_prime_R0,
            'L_prime_R0': L_prime_R0,
            'g': g,
        },
        'scaling_indices': {
            'm_gamma': m_gamma,
            'd_nu': d_nu,
            'a_L': a_L,
        },
        'angle_params': {
            'k_xi': k_xi,
            'xi_sat': xi_sat,
        },
        'physical_scales': {
            'k_nu': k_nu,  # keV
            'k_F': k_F,    # erg/cm²/s
        }
    }
    
    if reference_values is not None:
        # Extract reference values
        L = reference_values.get('L', 1e53)  # erg/s
        t_off = reference_values.get('t_off', 0.1)  # s
        z = reference_values.get('z', 1.0)
        d_L = reference_values.get('d_L', 2e28)  # cm
        u_1 = reference_values.get('u_1', 100)
        a_u = reference_values.get('a_u', 2.0)
        
        # Use equations from Appendix G to extract microphysics
        # From k_nu and Gamma_d_R0, nu_prime_R0, we can infer scales
        # This requires assumed values for epsilon_e, epsilon_B, xi_e, p
        
        # Placeholder: would need full inversion here
        result['inferred_source'] = {
            'note': 'Requires assumed microphysical parameters for full inversion',
            'k_nu_implies': f'ν_o ~ {k_nu:.2f} keV (depends on ϵ_e, ϵ_B, ξ_e)',
            'k_F_implies': f'F_o ~ {k_F:.2e} erg/cm²/s (depends on ϵ_rad, L)',
        }
    
    return result

# ---------------------------
# Main fitting routine
# ---------------------------
def run_fit(nu_obs, Fnu_obs, out_prefix='fit_nuFnu', do_plot=True, use_global=False):
    """
    Fit GRB spectrum data.
    
    IMPORTANT: Input should be F_ν (not νF_ν), but plotting will show νF_ν
    
    Args:
        nu_obs: observed frequencies [keV]
        Fnu_obs: observed F_ν [erg/cm²/s/keV] (NOT νF_ν!)
        use_global: if True, use differential_evolution for global optimization
    """
    # R grid: normalized radius R/R_0, starting at collision radius (R_0 = 1)
    # Physical constraint: R/R_0 ≥ 1 (cannot be before collision!)
    # Extend to ~300 R_0 to capture shock propagation well beyond doubling radius
    Rgrid = np.logspace(0, 2.5, 2000)  # 10^0 = 1.0 to 10^2.5 ≈ 316
    
    p0, lb, ub = default_initial_and_bounds(nu_obs, Fnu_obs)
    
    print("Fitting GRB spectrum with hydrodynamic model...")
    print(f"Data: {len(nu_obs)} points")
    print(f"Frequency range: {nu_obs.min():.2f} - {nu_obs.max():.2f} keV")
    print(f"F_ν range: {Fnu_obs.min():.3e} - {Fnu_obs.max():.3e} erg/cm²/s/keV")
    print(f"Rgrid: R/R_0 from {Rgrid[0]:.1f} to {Rgrid[-1]:.1f} ({len(Rgrid)} points)")
    print(f"  (Collision at R_0, extending to {Rgrid[-1]:.0f}× collision radius)")
    
    if use_global:
        print("\nUsing global optimization (differential_evolution)...")
        result = differential_evolution(
            lambda p: np.sum(residuals_with_HLE(p, Rgrid, nu_obs, Fnu_obs)**2),
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
            args=(Rgrid, nu_obs, Fnu_obs),
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
    
    # Check physical constraint
    theta_best = popt[:-2]
    nu_mod_grid, F_mod_grid = model_curve(Rgrid, theta_best)
    nu_pk_at_R0 = nu_mod_grid[0]
    nu_obs_max = np.max(nu_obs)
    
    # Extract normalized values at R_0 for diagnostics
    G_pl, G_sph, m_gamma = theta_best[0:3]
    nu_pl, nu_sph, d_nu = theta_best[3:6]
    L_pl, L_sph, a_L = theta_best[6:9]
    k_xi, xi_sat = theta_best[9:11]
    k_nu, k_F, g, s_blend = theta_best[11:15]
    
    # Normalized values at R_0 (should be ~1)
    Gamma_d_R0 = (G_pl**s_blend + G_sph**s_blend)**(1.0/s_blend)
    nu_prime_R0 = (nu_pl**s_blend + nu_sph**s_blend)**(1.0/s_blend)
    L_prime_R0 = (L_pl**s_blend + L_sph**s_blend)**(1.0/s_blend)
    
    print(f"\nPhysical constraint check:")
    print(f"  ν_pk(R_0) = {nu_pk_at_R0:.3f} keV")
    print(f"  max(ν_obs) = {nu_obs_max:.3f} keV")
    print(f"  Ratio: ν_pk(R_0)/max(ν_obs) = {nu_pk_at_R0/nu_obs_max:.3f}")
    
    print(f"\nNormalized quantities at R_0 (should be ~1):")
    print(f"  Γ_d(R_0) = {Gamma_d_R0:.3f}")
    print(f"  ν'_m(R_0) = {nu_prime_R0:.3f}")
    print(f"  L'(R_0) = {L_prime_R0:.3f}")
    
    print(f"\nPhysical scale parameters:")
    print(f"  k_ν = {k_nu:.3f} keV  (compare to max_nu = {nu_obs_max:.3f})")
    print(f"  k_F = {k_F:.3e} erg/cm²/s/keV  (F_ν scale)")
    print(f"  Compare to peak_Fnu_obs = {np.max(Fnu_obs):.3e} erg/cm²/s/keV")
    
    # Extract and display physical interpretation
    phys_params = extract_physical_parameters(theta_best)
    
    print(f"\nScaling indices (power-law behavior):")
    for key, val in phys_params['scaling_indices'].items():
        print(f"  {key:10s} = {val:.3f}")
    
    print(f"\nEffective angle parameters:")
    for key, val in phys_params['angle_params'].items():
        print(f"  {key:10s} = {val:.3f}")
    
    if nu_pk_at_R0 < nu_obs_max:
        print("\n  ⚠ WARNING: Physical constraint violated!")
        print("  The peak frequency at R_0 is below the observed maximum.")
        print("  This is unphysical - consider adjusting parameter bounds.")
    else:
        print("\n  ✓ Constraint satisfied")
    
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
        ax1.scatter(nu_obs, Fnu_obs, c='red', s=100, marker='o',
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
            np.log(nu_obs), np.log(Fnu_obs),
            bounds_error=False, fill_value='extrapolate'
        )(np.log(nu_plot)))
        F_pred_at_data = build_Fnu_with_HLE(nu_mod_grid, F_mod_grid, nu_obs,
                                            nu_bk_best, s_hle=s_hle_best)
        residuals = (F_pred_at_data - Fnu_obs) / Fnu_obs
        
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
    """
    Load CSV data with nu, nuFnu columns.
    
    IMPORTANT: The CSV contains (ν, νF_ν) but we need to work with F_ν internally.
    """
    with open(fname, 'r') as f:
        first_line = f.readline().strip()
    
    has_header = not all(c.isdigit() or c in '.,e-+ ' for c in first_line)
    
    data = np.genfromtxt(fname, delimiter=',', 
                        skip_header=1 if has_header else 0)
    
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"File must contain two columns: nu, nuFnu")
    
    nu = data[:, 0].astype(float)
    nuFnu = data[:, 1].astype(float)
    
    # Convert νF_ν to F_ν for internal use
    # F_ν = (νF_ν) / ν
    Fnu = nuFnu / nu
    
    return nu, Fnu

def synthetic_demo():
    """
    Generate synthetic data from known parameters using normalized formulation.
    Returns F_ν (not νF_ν).
    """
    # R/R_0 grid: starts at 1 (collision radius), extends to ~100 R_0
    Rgrid = np.logspace(0, 2.0, 1000)  # 1.0 to 100.0
    
    # True parameters with NORMALIZED hydrodynamic quantities
    # These should produce Γ_d, ν', L' ≈ 1 at R_0
    true_theta = [
        1.0, 1.08, 0.5,             # Γ_d: values near 1
        0.72, 1.2, -2.5,            # ν': values near 1
        1.0, 1.3, 1.8,              # L': values near 1
        0.3, 6.0,                   # ξ
        100.0,                      # k_nu: physical frequency scale (keV)
        1.5e-9,                     # k_F: physical F_ν scale (erg/cm²/s/keV)
        1.5,                        # g
        1.0                         # s_blend
    ]
    
    nu_mod, Fnu_mod = model_curve(Rgrid, true_theta)
    
    # Check that ν_pk(R_0) is reasonable
    print(f"Synthetic data: ν_pk(R_0) = {nu_mod[0]:.2f} keV (at R/R_0 = {Rgrid[0]:.1f})")
    print(f"Synthetic data: F_ν at peak(R_0) = {Fnu_mod[0]:.3e} erg/cm²/s/keV")
    
    # Sample points (more at peak region)
    idx = np.concatenate([
        np.linspace(100, 600, 20, dtype=int),
        np.linspace(600, 900, 10, dtype=int)
    ])
    
    nu_obs = nu_mod[idx] * (1.0 + 0.05*np.random.randn(len(idx)))
    Fnu_obs = Fnu_mod[idx] * np.exp(0.15*np.random.randn(len(idx)))
    
    return nu_obs, np.abs(Fnu_obs)

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
        nu_obs, Fnu_obs = synthetic_demo()
    else:
        print(f"Loading {args.data}...")
        nu_obs, Fnu_obs = load_csv_data(args.data)
    
    # Sort data
    sort_idx = np.argsort(nu_obs)
    nu_obs = nu_obs[sort_idx]
    Fnu_obs = Fnu_obs[sort_idx]
    
    popt = run_fit(nu_obs, Fnu_obs, out_prefix='fit_nuFnu', 
                   do_plot=True, use_global=args.use_global)

if __name__ == '__main__':
    main()