#!/usr/bin/env python3
"""
Visualization script to verify the model curve behavior:
- Monotonically decreasing nu_pk with R after the break
- Increasing F_pk with R after the break
This matches the spherical (solid line) behavior in the reference figure.
"""

import numpy as np
import matplotlib.pyplot as plt
from phys_radiation import approx_bhv_hybrid, tT_eff2
import sys

def plot_model_behavior(gi=1.0, tTfi=2.0, p2_values=[0.5], d2=1):
    """
    Plot (nu*F_nu)_pk vs nu_pk to verify monotonic behavior
    
    Parameters:
    -----------
    gi : float
        Initial Lorentz factor parameter
    tTfi : float
        Break time (R_f/R_0)
    p2_values : list
        List of p2 exponent values to test for (nu*F_nu)_pk after break
    d2 : float
        Exponent for nu_pk after break (default=1)
    """
    
    # Create time array spanning before and after break
    tTi = np.logspace(-1, 1, 500)
    
    # Setup plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ax_main, ax_nu, ax_nuFnu, ax_Fpk = axes.flatten()
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(p2_values)))
    
    for p2, color in zip(p2_values, colors):
        # Get model predictions
        nu, nuFnu = approx_bhv_hybrid(tTi, tTfi, gi, d1=1, d2=d2, p1=3, p2=p2)
        
        # Calculate F_pk = (nu*F_nu)_pk / nu_pk
        Fpk = nuFnu / nu
        
        # Identify before/after break
        before_break = tTi <= tTfi
        after_break = tTi > tTfi
        
        # Main plot: (nu*F_nu)_pk vs nu_pk (like the figure)
        label = f'p2={p2:.2f}'
        if p2 == 0.5:
            label += ' (new default)'
        ax_main.loglog(nu[before_break], nuFnu[before_break], 
                      c=color, ls='-', label=label, alpha=0.7)
        ax_main.loglog(nu[after_break], nuFnu[after_break], 
                      c=color, ls='--', alpha=0.7)
        
        # Mark the break point
        idx_break = np.argmin(np.abs(tTi - tTfi))
        ax_main.plot(nu[idx_break], nuFnu[idx_break], 'o', c=color, ms=8)
        
        # Individual quantities vs time
        ax_nu.loglog(tTi, nu, c=color, label=label)
        ax_nuFnu.loglog(tTi, nuFnu, c=color, label=label)
        ax_Fpk.loglog(tTi, Fpk, c=color, label=label)
        
        # Mark break point
        for ax in [ax_nu, ax_nuFnu, ax_Fpk]:
            ax.axvline(tTfi, c='gray', ls=':', alpha=0.5)
    
    # Format main plot
    ax_main.set_xlabel(r'$\nu_{pk}/\nu_0$', fontsize=12)
    ax_main.set_ylabel(r'$(\nu F_\nu)_{pk}/\nu_0 F_0$', fontsize=12)
    ax_main.set_title('Model Curve: Should match spherical (solid) line after break', fontsize=11)
    ax_main.legend()
    ax_main.grid(True, alpha=0.3)
    
    # Format individual plots
    ax_nu.set_ylabel(r'$\nu_{pk}/\nu_0$', fontsize=11)
    ax_nu.set_xlabel(r'$\tilde{T}$', fontsize=11)
    ax_nu.set_title(r'Peak frequency (should decrease after break)', fontsize=10)
    ax_nu.legend(fontsize=9)
    ax_nu.grid(True, alpha=0.3)
    
    ax_nuFnu.set_ylabel(r'$(\nu F_\nu)_{pk}/\nu_0 F_0$', fontsize=11)
    ax_nuFnu.set_xlabel(r'$\tilde{T}$', fontsize=11)
    ax_nuFnu.set_title(r'Peak $\nu F_\nu$ (should decrease after break)', fontsize=10)
    ax_nuFnu.legend(fontsize=9)
    ax_nuFnu.grid(True, alpha=0.3)
    
    ax_Fpk.set_ylabel(r'$F_{pk}$ [arbitrary units]', fontsize=11)
    ax_Fpk.set_xlabel(r'$\tilde{T}$', fontsize=11)
    ax_Fpk.set_title(r'Peak flux (should INCREASE after break)', fontsize=10)
    ax_Fpk.legend(fontsize=9)
    ax_Fpk.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Print behavior analysis
    print("\n" + "="*70)
    print("BEHAVIOR ANALYSIS AFTER BREAK")
    print("="*70)
    
    for p2 in p2_values:
        nu, nuFnu = approx_bhv_hybrid(tTi, tTfi, gi, d1=1, d2=d2, p1=3, p2=p2)
        Fpk = nuFnu / nu
        
        # Get values just after break
        idx_start = np.argmin(np.abs(tTi - tTfi*1.1))
        idx_end = np.argmin(np.abs(tTi - tTfi*2.0))
        
        nu_change = (nu[idx_end] - nu[idx_start]) / nu[idx_start]
        nuFnu_change = (nuFnu[idx_end] - nuFnu[idx_start]) / nuFnu[idx_start]
        Fpk_change = (Fpk[idx_end] - Fpk[idx_start]) / Fpk[idx_start]
        
        print(f"\np2 = {p2:.2f}:")
        print(f"  ν_pk change:      {nu_change:+.2%} {'✓ DECREASING' if nu_change < 0 else '✗ SHOULD DECREASE'}")
        print(f"  (νF_ν)_pk change: {nuFnu_change:+.2%} {'✓ DECREASING' if nuFnu_change < 0 else '✗ SHOULD DECREASE'}")
        print(f"  F_pk change:      {Fpk_change:+.2%} {'✓ INCREASING' if Fpk_change > 0 else '✗ SHOULD INCREASE'}")
        
        # Calculate effective power-law indices
        tTeff2_start = tT_eff2(tTi[idx_start], tTfi, gi)
        tTeff2_end = tT_eff2(tTi[idx_end], tTfi, gi)
        
        d_eff = np.log(nu[idx_end]/nu[idx_start]) / np.log(tTeff2_end/tTeff2_start)
        p_eff = np.log(nuFnu[idx_end]/nuFnu[idx_start]) / np.log(tTeff2_end/tTeff2_start)
        f_eff = np.log(Fpk[idx_end]/Fpk[idx_start]) / np.log(tTeff2_end/tTeff2_start)
        
        print(f"  Effective exponents:")
        print(f"    ν_pk ∝ tT_eff2^{d_eff:.2f} (expected: -{d2:.2f})")
        print(f"    (νF_ν)_pk ∝ tT_eff2^{p_eff:.2f} (expected: -{p2:.2f})")
        print(f"    F_pk ∝ tT_eff2^{f_eff:.2f} (expected: {-p2+d2:.2f})")
    
    print("\n" + "="*70 + "\n")
    
    return fig


if __name__ == '__main__':
    # Test different p2 values
    # p2 = 0.5: F_pk ∝ R^0.5 (increasing) - NEW DEFAULT
    # p2 = 1.0: F_pk ∝ R^0 (constant)
    # p2 = 3.0: F_pk ∝ R^-2 (decreasing) - OLD BEHAVIOR
    
    if len(sys.argv) > 1:
        p2_values = [float(x) for x in sys.argv[1:]]
    else:
        # Compare old vs new behavior
        p2_values = [0.5, 1.0, 3.0]
    
    print(f"Testing p2 values: {p2_values}")
    print("Note: p2=0.5 is the new default (increasing F_pk)")
    print("      p2=3.0 was the old value (decreasing F_pk)")
    
    fig = plot_model_behavior(gi=1.0, tTfi=0.8, p2_values=p2_values)
    plt.savefig('/workspace/bin/Tools/project_v1/model_verification.png', dpi=150, bbox_inches='tight')
    print("Plot saved to: bin/Tools/project_v1/model_verification.png")
    
    plt.show()
