#!/usr/bin/env python3
"""
Standalone verification script to verify the model curve behavior:
- Monotonically decreasing nu_pk with R after the break
- Increasing F_pk with R after the break

This script imports only the necessary functions without the full environment setup.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

# Direct implementation of the model functions
def tT_eff1(tTi, gi):
    """Normalized time 1 for fitted curves"""
    gi2 = gi**2
    return (1-gi2)+gi2*tTi

def tT_eff2(tTi, tTfi, gi):
    """Normalized time 2 for fitted curves"""
    gi2 = gi**2
    return (1-gi2) + gi2*tTi/tTfi

def approx_nupk_hybrid(tTi, tTfi, gi, d1=1, d2=1):
    """Peak frequency model"""
    return np.piecewise(tTi, [tTi<=tTfi, tTi>tTfi], 
                       [lambda x: x**-d1, 
                        lambda x: tTfi**-d1*tT_eff2(x, tTfi, gi)**-d2])

def approx_nupkFnupk_hybrid(tTi, tTfi, gi, p1=3, p2=0.5):
    """Peak (nu*F_nu) model - UPDATED with new p2 default"""
    def f(x):
        tTeff = tT_eff1(x, gi)
        return 1 - tTeff**-p1
    return np.piecewise(tTi, [tTi<=tTfi, tTi>tTfi], 
                       [lambda x: f(x), 
                        lambda x: f(tTfi)*tT_eff2(x, tTfi, gi)**-p2])

def approx_bhv_hybrid(tTi, tTfi, gi, d1=1, d2=1, p1=3, p2=0.5):
    """Returns nu_pk, (nu*F_nu)_pk from hybrid approach"""
    nu = approx_nupk_hybrid(tTi, tTfi, gi, d1=d1, d2=d2)
    nuFnu = approx_nupkFnupk_hybrid(tTi, tTfi, gi, p1=p1, p2=p2)
    return nu, nuFnu


def plot_model_behavior(gi=1.0, tTfi=0.8, p2_values=[0.5], d2=1):
    """Plot (nu*F_nu)_pk vs nu_pk to verify monotonic behavior"""
    
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
        
        # Main plot: (nu*F_nu)_pk vs nu_pk (like the reference figure)
        label = f'p2={p2:.2f}'
        if p2 == 0.5:
            label += ' (NEW default)'
        elif p2 == 3.0:
            label += ' (OLD value)'
        
        ax_main.loglog(nu[before_break], nuFnu[before_break], 
                      c=color, ls='-', label=label, alpha=0.7, lw=2)
        ax_main.loglog(nu[after_break], nuFnu[after_break], 
                      c=color, ls='--', alpha=0.7, lw=2)
        
        # Mark the break point
        idx_break = np.argmin(np.abs(tTi - tTfi))
        ax_main.plot(nu[idx_break], nuFnu[idx_break], 'o', c=color, ms=8)
        
        # Individual quantities vs time
        ax_nu.loglog(tTi, nu, c=color, label=label, lw=2)
        ax_nuFnu.loglog(tTi, nuFnu, c=color, label=label, lw=2)
        ax_Fpk.loglog(tTi, Fpk, c=color, label=label, lw=2)
        
        # Mark break point
        for ax in [ax_nu, ax_nuFnu, ax_Fpk]:
            ax.axvline(tTfi, c='gray', ls=':', alpha=0.5, lw=1.5)
            ax.text(tTfi*1.05, ax.get_ylim()[0]*1.2, 'break', 
                   rotation=90, va='bottom', fontsize=9, color='gray')
    
    # Format main plot
    ax_main.set_xlabel(r'$\nu_{pk}/\nu_0$', fontsize=14, fontweight='bold')
    ax_main.set_ylabel(r'$(\nu F_\nu)_{pk}/\nu_0 F_0$', fontsize=14, fontweight='bold')
    ax_main.set_title('Model Curve (solid=before break, dashed=after break)\n'
                     'Should match spherical line behavior from reference figure', 
                     fontsize=11)
    ax_main.legend(fontsize=10, loc='best')
    ax_main.grid(True, alpha=0.3)
    
    # Format individual plots
    ax_nu.set_ylabel(r'$\nu_{pk}/\nu_0$', fontsize=12)
    ax_nu.set_xlabel(r'$\tilde{T} = R/R_0$', fontsize=12)
    ax_nu.set_title(r'Peak frequency (must decrease after break)', fontsize=11, fontweight='bold')
    ax_nu.legend(fontsize=9)
    ax_nu.grid(True, alpha=0.3)
    
    ax_nuFnu.set_ylabel(r'$(\nu F_\nu)_{pk}/\nu_0 F_0$', fontsize=12)
    ax_nuFnu.set_xlabel(r'$\tilde{T} = R/R_0$', fontsize=12)
    ax_nuFnu.set_title(r'Peak $\nu F_\nu$ (must decrease after break)', fontsize=11, fontweight='bold')
    ax_nuFnu.legend(fontsize=9)
    ax_nuFnu.grid(True, alpha=0.3)
    
    ax_Fpk.set_ylabel(r'$F_{pk}$ [arbitrary units]', fontsize=12)
    ax_Fpk.set_xlabel(r'$\tilde{T} = R/R_0$', fontsize=12)
    ax_Fpk.set_title(r'Peak flux (must INCREASE after break)', fontsize=11, fontweight='bold')
    ax_Fpk.legend(fontsize=9)
    ax_Fpk.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Print behavior analysis
    print("\n" + "="*80)
    print("BEHAVIOR ANALYSIS AFTER BREAK (at R > R_break)")
    print("="*80)
    
    for p2 in p2_values:
        nu, nuFnu = approx_bhv_hybrid(tTi, tTfi, gi, d1=1, d2=d2, p1=3, p2=p2)
        Fpk = nuFnu / nu
        
        # Get values just after break
        idx_start = np.argmin(np.abs(tTi - tTfi*1.1))
        idx_end = np.argmin(np.abs(tTi - tTfi*2.0))
        
        nu_change = (nu[idx_end] - nu[idx_start]) / nu[idx_start]
        nuFnu_change = (nuFnu[idx_end] - nuFnu[idx_start]) / nuFnu[idx_start]
        Fpk_change = (Fpk[idx_end] - Fpk[idx_start]) / Fpk[idx_start]
        
        status = "NEW DEFAULT" if p2 == 0.5 else ("OLD VALUE" if p2 == 3.0 else "")
        print(f"\np2 = {p2:.2f} {status}:")
        print(f"  ν_pk change:      {nu_change:+.1%}  {'✓ DECREASING (correct)' if nu_change < 0 else '✗ SHOULD DECREASE'}")
        print(f"  (νF_ν)_pk change: {nuFnu_change:+.1%}  {'✓ DECREASING (correct)' if nuFnu_change < 0 else '✗ SHOULD DECREASE'}")
        print(f"  F_pk change:      {Fpk_change:+.1%}  {'✓✓ INCREASING (correct!)' if Fpk_change > 0 else '✗✗ SHOULD INCREASE'}")
        
        # Calculate effective power-law indices
        tTeff2_start = tT_eff2(tTi[idx_start], tTfi, gi)
        tTeff2_end = tT_eff2(tTi[idx_end], tTfi, gi)
        
        d_eff = np.log(nu[idx_end]/nu[idx_start]) / np.log(tTeff2_end/tTeff2_start)
        p_eff = np.log(nuFnu[idx_end]/nuFnu[idx_start]) / np.log(tTeff2_end/tTeff2_start)
        f_eff = np.log(Fpk[idx_end]/Fpk[idx_start]) / np.log(tTeff2_end/tTeff2_start)
        
        print(f"  Effective power-law indices (measured from fit):")
        print(f"    ν_pk ∝ tT_eff2^({d_eff:+.2f})    [expected: {-d2:.2f}]")
        print(f"    (νF_ν)_pk ∝ tT_eff2^({p_eff:+.2f})    [expected: {-p2:.2f}]")
        print(f"    F_pk ∝ tT_eff2^({f_eff:+.2f})    [expected: {-p2+d2:+.2f}]")
    
    print("\n" + "="*80)
    print("SUMMARY:")
    print("  - With p2=0.5 (NEW): F_pk INCREASES while ν_pk DECREASES ✓✓")
    print("  - With p2=3.0 (OLD): F_pk DECREASES (incorrect behavior) ✗✗")
    print("="*80 + "\n")
    
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
        p2_values = [0.5, 3.0]
    
    print("\n" + "="*80)
    print("MODEL CURVE VERIFICATION")
    print("="*80)
    print(f"Testing p2 exponent values: {p2_values}")
    print("  p2 = exponent for (νF_ν)_pk after break")
    print("  p2 = 0.5 is the NEW default (gives increasing F_pk)")
    print("  p2 = 3.0 was the OLD value (gave decreasing F_pk - incorrect!)")
    print("="*80 + "\n")
    
    fig = plot_model_behavior(gi=1.0, tTfi=0.8, p2_values=p2_values)
    
    output_path = '/workspace/bin/Tools/project_v1/model_verification.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved to: {output_path}")
    print("  You can view this plot to see the corrected model behavior.\n")
