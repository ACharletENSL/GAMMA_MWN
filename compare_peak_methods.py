#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compare peak detection methods between project_v1 and project_v2

This script properly compares the two methods by:
1. Converting to the same units
2. Using the same frequency grids
3. Documenting all differences
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add paths
sys.path.insert(0, '/workspace/bin/Tools/project_v1')
sys.path.insert(0, '/workspace/bin/Tools/project_v2')

from thinshell_analysis import get_pksnunFnu
from radiation import get_peaks_from_data
from environment import MyEnv

plt.ion()


def convert_to_same_units(nu_pk, nF_pk, env, front='RS', to_cgs=True):
    """
    Convert between normalized and CGS units
    
    Parameters:
    - nu_pk: peak frequencies (array)
    - nF_pk: peak fluxes (array)  
    - env: environment object
    - front: 'RS' or 'FS'
    - to_cgs: If True, convert normalized to CGS. If False, convert CGS to normalized
    
    Returns:
    - nu_pk, nF_pk in target units
    """
    if front == 'RS':
        nu0 = env.nu0
        F0 = env.zdl * env.L0 / 3
    else:
        nu0 = env.nu0FS
        F0 = env.zdl * env.L0FS / 3
    
    if to_cgs:
        # Convert normalized to CGS
        nu_pk_out = nu_pk * nu0
        nF_pk_out = nF_pk * (nu0 * F0)
    else:
        # Convert CGS to normalized
        nu_pk_out = nu_pk / nu0
        nF_pk_out = nF_pk / (nu0 * F0)
    
    return nu_pk_out, nF_pk_out


def compare_peak_methods(key, front='RS', plot=True, save_fig=True):
    """
    Compare peak detection methods with proper unit conversion
    
    Parameters:
    - key: simulation identifier
    - front: 'RS' or 'FS'
    - plot: whether to create comparison plots
    - save_fig: whether to save the figure
    
    Returns:
    - dict with comparison statistics
    """
    print(f"\n{'='*60}")
    print(f"Comparing peak detection methods for {key}, front={front}")
    print(f"{'='*60}\n")
    
    env = MyEnv(key)
    
    # Method 1 (project_v1) - returns data in CGS units
    print("Running Method 1 (project_v1/thinshell_analysis.py)...")
    try:
        nupks_v1, nuFnupks_v1 = get_pksnunFnu(key, func='Band', force=False)
        idx = 0 if front=='RS' else 1
        nu_pk_v1 = nupks_v1[idx]
        nF_pk_v1 = nuFnupks_v1[idx]
        print(f"  ✓ Got {len(nu_pk_v1)} time steps")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return None
    
    # Method 2 (project_v2) - returns normalized data
    print("Running Method 2 (project_v2/radiation.py)...")
    try:
        nu_pk_v2, nF_pk_v2 = get_peaks_from_data(key, front=front)
        print(f"  ✓ Got {len(nu_pk_v2)} time steps")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return None
    
    # Convert v2 to CGS for comparison
    print("\nConverting Method 2 results to CGS units...")
    nu_pk_v2_cgs, nF_pk_v2_cgs = convert_to_same_units(
        nu_pk_v2, nF_pk_v2, env, front=front, to_cgs=True
    )
    
    # Make sure arrays are same length (interpolate if needed)
    if len(nu_pk_v1) != len(nu_pk_v2_cgs):
        print(f"\n⚠ Warning: Different array lengths!")
        print(f"  Method 1: {len(nu_pk_v1)} points")
        print(f"  Method 2: {len(nu_pk_v2_cgs)} points")
        min_len = min(len(nu_pk_v1), len(nu_pk_v2_cgs))
        nu_pk_v1 = nu_pk_v1[:min_len]
        nF_pk_v1 = nF_pk_v1[:min_len]
        nu_pk_v2_cgs = nu_pk_v2_cgs[:min_len]
        nF_pk_v2_cgs = nF_pk_v2_cgs[:min_len]
        print(f"  Using first {min_len} points for comparison")
    
    # Filter out zeros/invalid values
    valid = (nu_pk_v1 > 0) & (nF_pk_v1 > 0) & (nu_pk_v2_cgs > 0) & (nF_pk_v2_cgs > 0)
    nu_pk_v1 = nu_pk_v1[valid]
    nF_pk_v1 = nF_pk_v1[valid]
    nu_pk_v2_cgs = nu_pk_v2_cgs[valid]
    nF_pk_v2_cgs = nF_pk_v2_cgs[valid]
    
    # Calculate ratios
    nu_ratio = nu_pk_v2_cgs / nu_pk_v1
    nF_ratio = nF_pk_v2_cgs / nF_pk_v1
    
    # Statistics
    results = {
        'key': key,
        'front': front,
        'n_points': len(nu_ratio),
        'nu_ratio': {
            'mean': np.mean(nu_ratio),
            'median': np.median(nu_ratio),
            'std': np.std(nu_ratio),
            'min': np.min(nu_ratio),
            'max': np.max(nu_ratio),
        },
        'nF_ratio': {
            'mean': np.mean(nF_ratio),
            'median': np.median(nF_ratio),
            'std': np.std(nF_ratio),
            'min': np.min(nF_ratio),
            'max': np.max(nF_ratio),
        },
    }
    
    # Print summary
    print(f"\n{'='*60}")
    print("COMPARISON RESULTS (Method 2 / Method 1)")
    print(f"{'='*60}")
    print(f"\nPeak Frequency Ratios:")
    print(f"  Mean:   {results['nu_ratio']['mean']:.3f}")
    print(f"  Median: {results['nu_ratio']['median']:.3f}")
    print(f"  Std:    {results['nu_ratio']['std']:.3f}")
    print(f"  Range:  [{results['nu_ratio']['min']:.3f}, {results['nu_ratio']['max']:.3f}]")
    
    print(f"\nPeak Flux Ratios:")
    print(f"  Mean:   {results['nF_ratio']['mean']:.3f}")
    print(f"  Median: {results['nF_ratio']['median']:.3f}")
    print(f"  Std:    {results['nF_ratio']['std']:.3f}")
    print(f"  Range:  [{results['nF_ratio']['min']:.3f}, {results['nF_ratio']['max']:.3f}]")
    
    # Interpretation
    print(f"\n{'='*60}")
    print("INTERPRETATION")
    print(f"{'='*60}")
    
    if abs(results['nu_ratio']['mean'] - 1.0) < 0.1:
        print("✓ Peak frequencies agree well (within 10%)")
    elif abs(results['nu_ratio']['mean'] - 1.0) < 0.3:
        print("⚠ Peak frequencies show moderate difference (10-30%)")
    else:
        print("✗ Peak frequencies show large difference (>30%)")
    
    if abs(results['nF_ratio']['mean'] - 1.0) < 0.1:
        print("✓ Peak fluxes agree well (within 10%)")
    elif abs(results['nF_ratio']['mean'] - 1.0) < 0.3:
        print("⚠ Peak fluxes show moderate difference (10-30%)")
    else:
        print("✗ Peak fluxes show large difference (>30%)")
    
    print("\nPossible reasons for remaining differences:")
    print("  1. Different frequency variables (nu_m vs nu_m2)")
    print("  2. Different frequency sampling grids")
    print("  3. Different data preprocessing (extrapolation/smoothing)")
    print("  4. Different cell selection (all vs first occurrence)")
    print("  5. Hybrid geometry corrections (only in method 1)")
    print(f"\nSee fix_peak_methods_comparison.md for detailed explanation.")
    
    # Create plots
    if plot:
        print(f"\nCreating comparison plots...")
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # Plot 1: Frequency comparison (log-log)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.loglog(nu_pk_v1, nu_pk_v2_cgs, 'o', alpha=0.5, markersize=4, label='Data')
        lim = [min(nu_pk_v1.min(), nu_pk_v2_cgs.min()), 
               max(nu_pk_v1.max(), nu_pk_v2_cgs.max())]
        ax1.loglog(lim, lim, 'k--', linewidth=2, label='1:1 line')
        ax1.set_xlabel('ν_pk Method 1 (Hz)', fontsize=11)
        ax1.set_ylabel('ν_pk Method 2 (Hz)', fontsize=11)
        ax1.set_title('Peak Frequency Comparison', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3, which='both')
        
        # Plot 2: Flux comparison (log-log)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.loglog(nF_pk_v1, nF_pk_v2_cgs, 'o', alpha=0.5, markersize=4, label='Data')
        lim = [min(nF_pk_v1.min(), nF_pk_v2_cgs.min()), 
               max(nF_pk_v1.max(), nF_pk_v2_cgs.max())]
        ax2.loglog(lim, lim, 'k--', linewidth=2, label='1:1 line')
        ax2.set_xlabel('νF_ν,pk Method 1 (erg/s/cm²)', fontsize=11)
        ax2.set_ylabel('νF_ν,pk Method 2 (erg/s/cm²)', fontsize=11)
        ax2.set_title('Peak Flux Comparison', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3, which='both')
        
        # Plot 3: Frequency ratio vs time
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(nu_ratio, 'o-', alpha=0.7, markersize=3)
        ax3.axhline(1.0, color='k', linestyle='--', linewidth=2)
        ax3.axhline(results['nu_ratio']['mean'], color='r', linestyle=':', 
                    linewidth=2, label=f"Mean = {results['nu_ratio']['mean']:.3f}")
        ax3.set_xlabel('Time Index', fontsize=11)
        ax3.set_ylabel('ν_pk Ratio (Method 2 / Method 1)', fontsize=11)
        ax3.set_title('Frequency Ratio Evolution', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Flux ratio vs time
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(nF_ratio, 'o-', alpha=0.7, markersize=3)
        ax4.axhline(1.0, color='k', linestyle='--', linewidth=2)
        ax4.axhline(results['nF_ratio']['mean'], color='r', linestyle=':', 
                    linewidth=2, label=f"Mean = {results['nF_ratio']['mean']:.3f}")
        ax4.set_xlabel('Time Index', fontsize=11)
        ax4.set_ylabel('νF_ν,pk Ratio (Method 2 / Method 1)', fontsize=11)
        ax4.set_title('Flux Ratio Evolution', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Frequency ratio histogram
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.hist(nu_ratio, bins=30, alpha=0.7, edgecolor='black')
        ax5.axvline(1.0, color='k', linestyle='--', linewidth=2)
        ax5.axvline(results['nu_ratio']['mean'], color='r', linestyle=':', linewidth=2)
        ax5.set_xlabel('ν_pk Ratio (Method 2 / Method 1)', fontsize=11)
        ax5.set_ylabel('Count', fontsize=11)
        ax5.set_title('Frequency Ratio Distribution', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y')
        
        # Plot 6: Flux ratio histogram
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.hist(nF_ratio, bins=30, alpha=0.7, edgecolor='black')
        ax6.axvline(1.0, color='k', linestyle='--', linewidth=2)
        ax6.axvline(results['nF_ratio']['mean'], color='r', linestyle=':', linewidth=2)
        ax6.set_xlabel('νF_ν,pk Ratio (Method 2 / Method 1)', fontsize=11)
        ax6.set_ylabel('Count', fontsize=11)
        ax6.set_title('Flux Ratio Distribution', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='y')
        
        fig.suptitle(f'Peak Detection Method Comparison: {key} ({front})', 
                    fontsize=14, fontweight='bold', y=0.995)
        
        if save_fig:
            figname = f'peak_comparison_{key}_{front}.png'
            plt.savefig(figname, dpi=150, bbox_inches='tight')
            print(f"  ✓ Saved figure: {figname}")
        
        plt.show()
    
    return results


def main():
    """Main function to run comparisons"""
    
    # Example usage - modify key and front as needed
    key = 'sph_fid'  # Change to your simulation key
    
    print("\n" + "="*70)
    print("PEAK DETECTION METHOD COMPARISON TOOL")
    print("="*70)
    print("\nThis script compares two methods:")
    print("  1. get_pksnunFnu (project_v1/thinshell_analysis.py)")
    print("  2. get_peaks_from_data (project_v2/radiation.py)")
    print("\nAfter fixing the normalization bug, both methods should give")
    print("similar results when converted to the same units.")
    
    # Check if key exists
    try:
        env = MyEnv(key)
        print(f"\n✓ Found simulation: {key}")
        print(f"  Geometry: {env.geometry}")
        print(f"  ν₀ (RS): {env.nu0:.3e} Hz")
        print(f"  ν₀ (FS): {env.nu0FS:.3e} Hz")
    except Exception as e:
        print(f"\n✗ Error loading simulation '{key}': {e}")
        print("\nPlease edit this script and set the correct simulation key.")
        return
    
    # Compare both fronts
    for front in ['RS', 'FS']:
        try:
            results = compare_peak_methods(key, front=front, plot=True, save_fig=True)
        except Exception as e:
            print(f"\n✗ Error comparing {front}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("DONE")
    print("="*70)
    print("\nFor more information on the differences, see:")
    print("  - fix_peak_methods_comparison.md")
    print("  - peak_method_comparison.md")
    print("  - remaining_differences_analysis.md")
    print()


if __name__ == '__main__':
    main()
