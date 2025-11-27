#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick check script to verify if there's a discrepancy between v1 and v2 peak results
"""

import sys
import os
from pathlib import Path
import numpy as np

# Add both project directories to path
project_v1_path = Path(__file__).parent / 'project_v1'
project_v2_path = Path(__file__).parent / 'project_v2'
sys.path.insert(0, str(project_v1_path))
sys.path.insert(0, str(project_v2_path))

def quick_comparison(key='HPC_cart', shock='RS'):
    """
    Quick comparison of peak values from v1 and v2
    """
    print(f"\nQuick Check: Comparing {key}, shock {shock}")
    print("="*70)
    
    # Try v1
    try:
        from project_v1.environment import MyEnv as MyEnv_v1
        from project_v1.thinshell_analysis import get_pksnunFnu
        
        env_v1 = MyEnv_v1(key)
        nupks_v1, nuFnupks_v1 = get_pksnunFnu(key, func='Band', force=False)
        
        idx = 0 if shock == 'RS' else 1
        nu0 = env_v1.nu0 if shock == 'RS' else env_v1.nu0FS
        
        # Get peak of peaks
        i_max = np.argmax(nuFnupks_v1[idx])
        nu_pk_v1 = nupks_v1[idx][i_max] / nu0
        nF_pk_v1 = nuFnupks_v1[idx][i_max] / env_v1.nu0F0
        
        print(f"\n✓ V1 loaded successfully")
        print(f"  Geometry: {env_v1.geometry}")
        print(f"  Peak frequency (normalized): {nu_pk_v1:.6e}")
        print(f"  Peak flux (normalized):      {nF_pk_v1:.6e}")
        
        v1_success = True
    except Exception as e:
        print(f"\n✗ V1 failed: {e}")
        v1_success = False
    
    # Try v2
    try:
        from project_v2.environment import MyEnv as MyEnv_v2
        from project_v2.run_analysis import get_peaks_from_data
        
        env_v2 = MyEnv_v2(key)
        front = 'RS' if shock == 'RS' else 'FS'
        nu_pk_v2, nF_pk_v2 = get_peaks_from_data(key, front=front)
        
        # Get peak of peaks
        i_max = np.argmax(nF_pk_v2)
        nu_pk_v2_val = nu_pk_v2[i_max]
        nF_pk_v2_val = nF_pk_v2[i_max]
        
        print(f"\n✓ V2 loaded successfully")
        print(f"  Geometry: {env_v2.geometry}")
        print(f"  Peak frequency (normalized): {nu_pk_v2_val:.6e}")
        print(f"  Peak flux (normalized):      {nF_pk_v2_val:.6e}")
        
        v2_success = True
    except Exception as e:
        print(f"\n✗ V2 failed: {e}")
        v2_success = False
    
    # Compare
    if v1_success and v2_success:
        print(f"\n" + "="*70)
        print("COMPARISON:")
        print("="*70)
        
        freq_diff = 100 * (nu_pk_v1 - nu_pk_v2_val) / nu_pk_v1
        flux_diff = 100 * (nF_pk_v1 - nF_pk_v2_val) / nF_pk_v1
        
        print(f"\nPeak Frequency:")
        print(f"  V1: {nu_pk_v1:.6e}")
        print(f"  V2: {nu_pk_v2_val:.6e}")
        print(f"  Difference: {freq_diff:+.2f}%")
        
        print(f"\nPeak Flux:")
        print(f"  V1: {nF_pk_v1:.6e}")
        print(f"  V2: {nF_pk_v2_val:.6e}")
        print(f"  Difference: {flux_diff:+.2f}%")
        
        # Verdict
        print(f"\n" + "="*70)
        threshold = 1.0  # 1% threshold
        if abs(freq_diff) < threshold and abs(flux_diff) < threshold:
            print("✓ VERDICT: Results agree within 1% - differences likely numerical")
        elif abs(freq_diff) < 5.0 and abs(flux_diff) < 5.0:
            print("⚠ VERDICT: Results differ by 1-5% - investigate with diagnostic script")
        else:
            print("✗ VERDICT: Significant discrepancy detected (>5%)")
            print("  → Run full diagnostic: python diagnostic_hybrid_comparison.py")
        print("="*70)
        
    elif not v1_success and not v2_success:
        print("\n✗ Both versions failed - check file paths and data availability")
    else:
        failed = "V1" if not v1_success else "V2"
        print(f"\n✗ {failed} failed - cannot compare")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick check for hybrid analysis discrepancies')
    parser.add_argument('--key', type=str, default='HPC_cart',
                       help='Run key (default: HPC_cart)')
    parser.add_argument('--shock', type=str, default='RS', choices=['RS', 'FS'],
                       help='Shock front (default: RS)')
    
    args = parser.parse_args()
    
    try:
        quick_comparison(args.key, args.shock)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
    
    print()
