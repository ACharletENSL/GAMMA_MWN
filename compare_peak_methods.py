#!/usr/bin/env python3
"""
Diagnostic script to compare peak values from both methods
and identify the source of differences.
"""

import sys
import numpy as np

# Add paths
sys.path.insert(0, '/workspace/bin/Tools/project_v1')
sys.path.insert(0, '/workspace/bin/Tools/project_v2')

from project_v1.thinshell_analysis import get_pksnunFnu
from project_v1.environment import MyEnv as MyEnv_v1
from project_v2.radiation import get_peaks_from_data
from project_v2.IO import MyEnv as MyEnv_v2

def compare_peaks(key, front='RS'):
    """
    Compare peak values from both methods and diagnose differences.
    
    Parameters:
    -----------
    key : str
        Run identifier
    front : str
        'RS' or 'FS'
    """
    
    print(f"\n{'='*60}")
    print(f"Comparing peak calculation methods for {key}, {front} front")
    print(f"{'='*60}\n")
    
    # Get peaks from project_v1
    print("Method 1 (project_v1/get_pksnunFnu):")
    print("-" * 40)
    try:
        nupks_v1, nuFnupks_v1 = get_pksnunFnu(key, func='Band', force=False)
        idx = 0 if front == 'RS' else 1
        nu_pk_v1 = nupks_v1[idx]
        nF_pk_v1 = nuFnupks_v1[idx]
        
        # Get environment for normalization
        env_v1 = MyEnv_v1(key)
        nu0_v1 = env_v1.nu0 if front == 'RS' else env_v1.nu0FS
        F0_v1 = env_v1.F0 if front == 'RS' else env_v1.F0FS
        nu0F0_v1 = nu0_v1 * F0_v1
        
        print(f"  Number of time steps: {len(nu_pk_v1)}")
        print(f"  Peak frequency range: {nu_pk_v1.min():.3e} - {nu_pk_v1.max():.3e} Hz")
        print(f"  Peak flux range: {nF_pk_v1.min():.3e} - {nF_pk_v1.max():.3e} erg/s/cm²")
        print(f"  Units: CGS (physical)")
        print(f"  Normalization factors:")
        print(f"    nu0 = {nu0_v1:.3e} Hz")
        print(f"    F0 = {F0_v1:.3e} erg/s/cm²")
        print(f"    nu0*F0 = {nu0F0_v1:.3e}")
        
        # Compute normalized values
        nu_pk_v1_norm = nu_pk_v1 / nu0_v1
        nF_pk_v1_norm = nF_pk_v1 / nu0F0_v1
        print(f"  Normalized peak flux range: {nF_pk_v1_norm.min():.3e} - {nF_pk_v1_norm.max():.3e}")
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return
    
    # Get peaks from project_v2
    print(f"\nMethod 2 (project_v2/get_peaks_from_data):")
    print("-" * 40)
    try:
        nu_pk_v2, nF_pk_v2 = get_peaks_from_data(key, front=front)
        
        # Get environment
        env_v2 = MyEnv_v2(key)
        nu0_v2 = env_v2.nu0 if front == 'RS' else env_v2.nu0FS
        F0_v2 = env_v2.F0 if front == 'RS' else env_v2.F0FS
        nu0F0_v2 = nu0_v2 * F0_v2
        
        print(f"  Number of time steps: {len(nu_pk_v2)}")
        print(f"  Peak frequency range: {nu_pk_v2.min():.3e} - {nu_pk_v2.max():.3e} (normalized)")
        print(f"  Peak flux range: {nF_pk_v2.min():.3e} - {nF_pk_v2.max():.3e} (normalized)")
        print(f"  Units: Normalized (dimensionless)")
        print(f"  Normalization factors:")
        print(f"    nu0 = {nu0_v2:.3e} Hz")
        print(f"    F0 = {F0_v2:.3e} erg/s/cm²")
        print(f"    nu0*F0 = {nu0F0_v2:.3e}")
        
        # Convert to CGS
        nu_pk_v2_cgs = nu_pk_v2 * nu0_v2
        nF_pk_v2_cgs = nF_pk_v2 * nu0F0_v2
        print(f"  CGS peak flux range: {nF_pk_v2_cgs.min():.3e} - {nF_pk_v2_cgs.max():.3e} erg/s/cm²")
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return
    
    # Compare
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}\n")
    
    # Check if same number of points
    if len(nu_pk_v1) != len(nu_pk_v2):
        print(f"WARNING: Different number of time steps!")
        print(f"  project_v1: {len(nu_pk_v1)} points")
        print(f"  project_v2: {len(nu_pk_v2)} points")
        print(f"  Will compare only first {min(len(nu_pk_v1), len(nu_pk_v2))} points")
        N = min(len(nu_pk_v1), len(nu_pk_v2))
    else:
        N = len(nu_pk_v1)
    
    # Compare frequencies (normalized)
    freq_ratio = nu_pk_v1_norm[:N] / nu_pk_v2[:N]
    print(f"\nFrequency comparison (normalized):")
    print(f"  Mean ratio: {freq_ratio.mean():.4f}")
    print(f"  Std ratio: {freq_ratio.std():.4f}")
    print(f"  Min ratio: {freq_ratio.min():.4f}")
    print(f"  Max ratio: {freq_ratio.max():.4f}")
    if np.abs(freq_ratio.mean() - 1.0) > 0.01:
        print(f"  ⚠️  Frequencies differ by ~{100*np.abs(freq_ratio.mean()-1):.1f}%")
        print(f"     This is likely due to using different frequency variables:")
        print(f"     project_v1 uses 'nu_m' (from simulation gmin)")
        print(f"     project_v2 uses 'nu_m2' (from theoretical gmin)")
    
    # Compare fluxes (normalized)
    flux_ratio = nF_pk_v1_norm[:N] / nF_pk_v2[:N]
    print(f"\nFlux comparison (normalized):")
    print(f"  Mean ratio: {flux_ratio.mean():.4f}")
    print(f"  Std ratio: {flux_ratio.std():.4f}")
    print(f"  Min ratio: {flux_ratio.min():.4f}")
    print(f"  Max ratio: {flux_ratio.max():.4f}")
    if np.abs(flux_ratio.mean() - 1.0) > 0.01:
        print(f"  ⚠️  Fluxes differ by ~{100*np.abs(flux_ratio.mean()-1):.1f}%")
        print(f"     Possible causes:")
        print(f"     1. Different data sources (preprocessed vs raw)")
        print(f"     2. Different cell selection (all vs first occurrence)")
        print(f"     3. Different frequency grids (~600 vs 500 points)")
        print(f"     4. Hybrid corrections in project_v1")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"\nThe two methods give different results because:")
    print(f"1. Different frequency variables (nu_m vs nu_m2)")
    print(f"2. Different data preprocessing")
    print(f"3. Different frequency sampling grids")
    print(f"4. Potentially different hybrid corrections")
    print(f"\nSee 'remaining_differences_analysis.md' for details.")
    print(f"\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compare_peak_methods.py <key> [front]")
        print("  key: run identifier (e.g., 'sph_fid')")
        print("  front: 'RS' or 'FS' (default: 'RS')")
        sys.exit(1)
    
    key = sys.argv[1]
    front = sys.argv[2] if len(sys.argv) > 2 else 'RS'
    
    compare_peaks(key, front)
