#!/usr/bin/env python3
"""
Diagnostic script to verify the normalization bug in project_v2/radiation.py

The bug: Line 118 divides by (3*L0) when it should divide by (L0/3),
causing a factor of 9 error in the normalized flux.
"""

import sys
import numpy as np

sys.path.insert(0, '/workspace/bin/Tools/project_v1')
sys.path.insert(0, '/workspace/bin/Tools/project_v2')

from project_v1.thinshell_analysis import get_pksnunFnu
from project_v1.environment import MyEnv as MyEnv_v1
from project_v2.radiation import get_peaks_from_data
from project_v2.IO import MyEnv as MyEnv_v2

def diagnose_normalization(key, front='RS'):
    """
    Diagnose the normalization difference between v1 and v2.
    """
    
    print(f"\n{'='*70}")
    print(f"Normalization Diagnosis for {key}, {front} front")
    print(f"{'='*70}\n")
    
    # Get peaks from both methods
    try:
        nupks_v1, nuFnupks_v1 = get_pksnunFnu(key, func='Band', force=False)
        idx = 0 if front == 'RS' else 1
        nu_pk_v1 = nupks_v1[idx]
        nF_pk_v1 = nuFnupks_v1[idx]
        
        env_v1 = MyEnv_v1(key)
        nu0_v1 = env_v1.nu0 if front == 'RS' else env_v1.nu0FS
        F0_v1 = env_v1.F0 if front == 'RS' else env_v1.F0FS
        L0_v1 = env_v1.L0 if front == 'RS' else env_v1.L0FS
        
        # Normalize v1
        nu_pk_v1_norm = nu_pk_v1 / nu0_v1
        nF_pk_v1_norm = nF_pk_v1 / (nu0_v1 * F0_v1)
        
        print("Method 1 (project_v1):")
        print(f"  L0 = {L0_v1:.4e} erg/(s*Hz)")
        print(f"  F0 = zdl * L0 / 3 = {F0_v1:.4e} erg/(s*cm²*Hz)")
        print(f"  Formula: nuFnu_norm = (nu/nu0) * (zdl*Lth*S/T²) / (zdl*L0/3)")
        print(f"                      = (nu/nu0) * (Lth/L0) * 3 * S/T²")
        print(f"  Peak flux range (normalized): {nF_pk_v1_norm.min():.4e} - {nF_pk_v1_norm.max():.4e}")
        
    except Exception as e:
        print(f"Error in v1: {e}")
        return
    
    try:
        nu_pk_v2, nF_pk_v2 = get_peaks_from_data(key, front=front)
        
        env_v2 = MyEnv_v2(key)
        nu0_v2 = env_v2.nu0 if front == 'RS' else env_v2.nu0FS
        F0_v2 = env_v2.F0 if front == 'RS' else env_v2.F0FS
        L0_v2 = env_v2.L0 if front == 'RS' else env_v2.L0FS
        
        print(f"\nMethod 2 (project_v2):")
        print(f"  L0 = {L0_v2:.4e} erg/(s*Hz)")
        print(f"  F0 = zdl * L0 / 3 = {F0_v2:.4e} erg/(s*cm²*Hz)")
        print(f"  Formula: nF = (nu/nu0) * (Lth/(3*L0)) * S/T²")
        print(f"  Peak flux range (normalized): {nF_pk_v2.min():.4e} - {nF_pk_v2.max():.4e}")
        
    except Exception as e:
        print(f"Error in v2: {e}")
        return
    
    # Compare
    print(f"\n{'='*70}")
    print("ANALYSIS")
    print(f"{'='*70}\n")
    
    N = min(len(nF_pk_v1_norm), len(nF_pk_v2))
    ratio = nF_pk_v1_norm[:N] / nF_pk_v2[:N]
    
    print("Flux ratio (v1_normalized / v2):")
    print(f"  Mean:   {ratio.mean():.4f}")
    print(f"  Median: {np.median(ratio):.4f}")
    print(f"  Std:    {ratio.std():.4f}")
    print(f"  Min:    {ratio.min():.4f}")
    print(f"  Max:    {ratio.max():.4f}")
    
    print(f"\n{'='*70}")
    print("CONCLUSION")
    print(f"{'='*70}\n")
    
    expected_ratio = 9.0
    measured_ratio = np.median(ratio)
    
    if abs(measured_ratio - expected_ratio) < 0.5:
        print(f"✓ The ratio is approximately {expected_ratio:.1f}, as predicted!")
        print(f"\nThe bug is in radiation.py, line 118:")
        print(f"  CURRENT:  F /= 3 * (env.L0 if (cell.trac < 1.5) else env.L0FS)")
        print(f"  SHOULD BE: F /= (env.L0 if (cell.trac < 1.5) else env.L0FS) / 3")
        print(f"\nOR equivalently:")
        print(f"  F /= (env.L0 if (cell.trac < 1.5) else env.L0FS)")
        print(f"  F *= 3")
        print(f"\nThis causes project_v2 to underestimate the flux by a factor of 9.")
    else:
        print(f"⚠ The ratio is {measured_ratio:.2f}, not {expected_ratio:.1f}")
        print(f"There may be additional differences beyond the normalization bug.")
    
    print()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnose_normalization.py <key> [front]")
        print("  key: run identifier (e.g., 'sph_fid')")
        print("  front: 'RS' or 'FS' (default: 'RS')")
        sys.exit(1)
    
    key = sys.argv[1]
    front = sys.argv[2] if len(sys.argv) > 2 else 'RS'
    
    diagnose_normalization(key, front)
