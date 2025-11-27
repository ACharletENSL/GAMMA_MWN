#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostic script to compare hybrid analysis results between project_v1 and project_v2

This script helps identify why peak frequency and flux differ between the two implementations
for the 'hybrid' (cartesian) geometry case.
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Add both project directories to path
project_v1_path = Path(__file__).parent / 'project_v1'
project_v2_path = Path(__file__).parent / 'project_v2'
sys.path.insert(0, str(project_v1_path))
sys.path.insert(0, str(project_v2_path))

def compare_stored_vs_calculated_variables(key, shock='RS', n_samples=10):
    """
    Compare the nu_m and Lth values stored in v1's CSV vs freshly calculated
    
    Parameters:
    -----------
    key : str
        Run key (e.g., 'HPC_cart', 'plan_fid')
    shock : str
        'RS' or 'FS'
    n_samples : int
        Number of sample points to check
    """
    print(f"\n{'='*80}")
    print(f"Comparing stored vs calculated variables for {key}, shock {shock}")
    print(f"{'='*80}\n")
    
    # Import from v1
    from project_v1.environment import MyEnv as MyEnv_v1
    from project_v1.thinshell_analysis import get_contribspath
    from project_v1.data_IO import openData
    from project_v1.df_funcs import get_variable
    
    env = MyEnv_v1(key)
    
    # Read stored contributions from v1
    contrib_path = get_contribspath(key) + f'{shock}.csv'
    
    if not os.path.exists(contrib_path):
        print(f"ERROR: Contribution file not found: {contrib_path}")
        print("Please run analysis_hydro_thinshell() and extract_contribs() first")
        return None
        
    df_stored = pd.read_csv(contrib_path, index_col='it')
    
    # Sample n_samples iterations evenly
    its = df_stored.index.values
    sample_indices = np.linspace(0, len(its)-1, min(n_samples, len(its)), dtype=int)
    sample_its = its[sample_indices]
    
    results = {
        'iteration': [],
        'r_stored': [],
        'nu_m_stored': [],
        'Lth_stored': [],
        'nu_m_calculated': [],
        'Lth_calculated': [],
        'nu_m_diff_%': [],
        'Lth_diff_%': [],
    }
    
    print(f"{'Iter':<8} {'r/R0':<10} {'nu_m_stored':<15} {'nu_m_calc':<15} {'Δ%':<10} {'Lth_stored':<15} {'Lth_calc':<15} {'Δ%':<10}")
    print("-"*110)
    
    for it in sample_its:
        # Get stored values
        row = df_stored.loc[it]
        r_stored = row[f'r_{{{shock}}}']
        nu_m_stored = row[f'nu_m_{{{shock}}}']
        Lth_stored = row[f'Lth_{{{shock}}}']
        
        # Load raw data file and calculate fresh
        try:
            df_raw = openData(key, it)
            # Get the cell behind the shock
            from project_v1.data_IO import df_get_cellsBehindShock
            RS, FS = df_get_cellsBehindShock(df_raw)
            cell = RS if shock == 'RS' else FS
            
            if not cell.empty:
                nu_m_calc = get_variable(cell, 'nu_m2', env)
                Lth_calc = get_variable(cell, 'Lth', env)
                
                # Calculate differences
                nu_m_diff = 100 * (nu_m_calc - nu_m_stored) / nu_m_stored if nu_m_stored != 0 else np.nan
                Lth_diff = 100 * (Lth_calc - Lth_stored) / Lth_stored if Lth_stored != 0 else np.nan
                
                # Store results
                results['iteration'].append(it)
                results['r_stored'].append(r_stored)
                results['nu_m_stored'].append(nu_m_stored)
                results['Lth_stored'].append(Lth_stored)
                results['nu_m_calculated'].append(nu_m_calc)
                results['Lth_calculated'].append(Lth_calc)
                results['nu_m_diff_%'].append(nu_m_diff)
                results['Lth_diff_%'].append(Lth_diff)
                
                print(f"{it:<8} {r_stored*3e10/env.R0:<10.4f} {nu_m_stored:<15.4e} {nu_m_calc:<15.4e} {nu_m_diff:<10.3f} "
                      f"{Lth_stored:<15.4e} {Lth_calc:<15.4e} {Lth_diff:<10.3f}")
        except Exception as e:
            print(f"Error processing iteration {it}: {e}")
            continue
    
    # Summary statistics
    if results['nu_m_diff_%']:
        print("\n" + "="*110)
        print("SUMMARY STATISTICS:")
        print(f"  nu_m differences: mean = {np.nanmean(results['nu_m_diff_%']):.3f}%, "
              f"std = {np.nanstd(results['nu_m_diff_%']):.3f}%, "
              f"max = {np.nanmax(np.abs(results['nu_m_diff_%'])):.3f}%")
        print(f"  Lth differences:  mean = {np.nanmean(results['Lth_diff_%']):.3f}%, "
              f"std = {np.nanstd(results['Lth_diff_%']):.3f}%, "
              f"max = {np.nanmax(np.abs(results['Lth_diff_%'])):.3f}%")
        print("="*110)
    
    return pd.DataFrame(results)


def compare_hybrid_corrections(key, shock='RS', n_samples=5):
    """
    Compare hybrid corrections applied in v1 vs v2
    """
    print(f"\n{'='*80}")
    print(f"Comparing hybrid corrections for {key}, shock {shock}")
    print(f"{'='*80}\n")
    
    from project_v1.environment import MyEnv as MyEnv_v1
    from project_v1.thinshell_analysis import get_contribspath
    from project_v1.phys_constants import c_
    
    env = MyEnv_v1(key)
    
    if env.geometry != 'cartesian':
        print(f"Warning: Geometry is '{env.geometry}', not 'cartesian'. Hybrid corrections won't be applied.")
    
    # Read contributions
    contrib_path = get_contribspath(key) + f'{shock}.csv'
    if not os.path.exists(contrib_path):
        print(f"ERROR: Contribution file not found: {contrib_path}")
        return None
        
    df = pd.read_csv(contrib_path, index_col='it')
    
    # Sample iterations
    its = df.index.values
    sample_indices = np.linspace(0, len(its)-1, min(n_samples, len(its)), dtype=int)
    sample_its = its[sample_indices]
    
    print(f"{'Iter':<8} {'r/R0':<10} {'nu_m (raw)':<15} {'nu_m (corr)':<15} {'L (raw)':<15} {'L (corr)':<15}")
    print("-"*90)
    
    for it in sample_its:
        row = df.loc[it]
        r = row[f'r_{{{shock}}}']
        nu_m_raw = row[f'nu_m_{{{shock}}}']
        L_raw = row[f'Lth_{{{shock}}}']
        
        # Apply hybrid corrections as in v1
        rfac = r * c_ / env.R0
        
        if env.geometry == 'cartesian':
            nu_m_corr = nu_m_raw / rfac  # d=-1: nu_m *= r^(-1)
            L_corr = L_raw * rfac         # a=1:  L *= r^1
        else:
            nu_m_corr = nu_m_raw
            L_corr = L_raw
        
        print(f"{it:<8} {rfac:<10.4f} {nu_m_raw:<15.4e} {nu_m_corr:<15.4e} {L_raw:<15.4e} {L_corr:<15.4e}")
    
    print("\nNote: v1 applies nu_m /= (r/R0) and L *= (r/R0)")
    print("      v2 applies the same corrections")


def compare_peak_values(key, shock='RS'):
    """
    Compare final peak values from v1 and v2
    """
    print(f"\n{'='*80}")
    print(f"Comparing peak values between v1 and v2 for {key}, shock {shock}")
    print(f"{'='*80}\n")
    
    try:
        # Import from v1
        from project_v1.environment import MyEnv as MyEnv_v1
        from project_v1.thinshell_analysis import get_pksnunFnu
        
        env_v1 = MyEnv_v1(key)
        nupks_v1, nuFnupks_v1 = get_pksnunFnu(key, func='Band', force=False)
        
        idx = 0 if shock == 'RS' else 1
        nu_pk_v1 = nupks_v1[idx]
        nF_pk_v1 = nuFnupks_v1[idx]
        
        # Normalize
        nu0 = env_v1.nu0 if shock == 'RS' else env_v1.nu0FS
        nu0F0 = env_v1.nu0F0
        
        nu_pk_v1_norm = nu_pk_v1 / nu0
        nF_pk_v1_norm = nF_pk_v1 / nu0F0
        
        print(f"V1 Results ({shock}):")
        print(f"  First 5 nu_pk/nu0: {nu_pk_v1_norm[:5]}")
        print(f"  First 5 nuFnu_pk/nu0F0: {nF_pk_v1_norm[:5]}")
        print(f"  Peak value at max: nu_pk/nu0 = {nu_pk_v1_norm[np.argmax(nF_pk_v1_norm)]:.4e}, "
              f"nuFnu_pk/nu0F0 = {np.max(nF_pk_v1_norm):.4e}")
        
    except Exception as e:
        print(f"Error loading v1 results: {e}")
        return
    
    try:
        # Import from v2
        from project_v2.environment import MyEnv as MyEnv_v2
        from project_v2.run_analysis import get_peaks_from_data
        
        env_v2 = MyEnv_v2(key)
        front = 'RS' if shock == 'RS' else 'FS'
        nu_pk_v2_norm, nF_pk_v2_norm = get_peaks_from_data(key, front=front)
        
        print(f"\nV2 Results ({shock}):")
        print(f"  First 5 nu_pk/nu0: {nu_pk_v2_norm[:5]}")
        print(f"  First 5 nuFnu_pk/nu0F0: {nF_pk_v2_norm[:5]}")
        print(f"  Peak value at max: nu_pk/nu0 = {nu_pk_v2_norm[np.argmax(nF_pk_v2_norm)]:.4e}, "
              f"nuFnu_pk/nu0F0 = {np.max(nF_pk_v2_norm):.4e}")
        
        # Compare lengths
        if len(nu_pk_v1) != len(nu_pk_v2_norm):
            print(f"\nWARNING: Different array lengths! v1: {len(nu_pk_v1)}, v2: {len(nu_pk_v2_norm)}")
        else:
            # Calculate differences
            min_len = min(len(nu_pk_v1_norm), len(nu_pk_v2_norm))
            nu_diff = nu_pk_v1_norm[:min_len] - nu_pk_v2_norm[:min_len]
            nF_diff = nF_pk_v1_norm[:min_len] - nF_pk_v2_norm[:min_len]
            
            nu_rel_diff = 100 * nu_diff / nu_pk_v1_norm[:min_len]
            nF_rel_diff = 100 * nF_diff / nF_pk_v1_norm[:min_len]
            
            print(f"\nDifferences (v1 - v2):")
            print(f"  nu_pk: mean = {np.nanmean(nu_rel_diff):.3f}%, std = {np.nanstd(nu_rel_diff):.3f}%, "
                  f"max = {np.nanmax(np.abs(nu_rel_diff)):.3f}%")
            print(f"  nuFnu_pk: mean = {np.nanmean(nF_rel_diff):.3f}%, std = {np.nanstd(nF_rel_diff):.3f}%, "
                  f"max = {np.nanmax(np.abs(nF_rel_diff)):.3f}%")
            
    except Exception as e:
        print(f"Error loading v2 results: {e}")


def main():
    """
    Main diagnostic routine
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Diagnose hybrid analysis discrepancies')
    parser.add_argument('key', type=str, help='Run key (e.g., HPC_cart, plan_fid)')
    parser.add_argument('--shock', type=str, default='RS', choices=['RS', 'FS'],
                       help='Shock front to analyze')
    parser.add_argument('--test', type=str, default='all',
                       choices=['all', 'stored', 'corrections', 'peaks'],
                       help='Which test to run')
    parser.add_argument('--samples', type=int, default=10,
                       help='Number of sample points to check')
    
    args = parser.parse_args()
    
    if args.test in ['all', 'stored']:
        print("\n" + "="*80)
        print("TEST 1: Comparing stored vs calculated variables")
        print("="*80)
        df_results = compare_stored_vs_calculated_variables(args.key, args.shock, args.samples)
    
    if args.test in ['all', 'corrections']:
        print("\n" + "="*80)
        print("TEST 2: Examining hybrid corrections")
        print("="*80)
        compare_hybrid_corrections(args.key, args.shock, min(args.samples, 5))
    
    if args.test in ['all', 'peaks']:
        print("\n" + "="*80)
        print("TEST 3: Comparing final peak values")
        print("="*80)
        compare_peak_values(args.key, args.shock)
    
    print("\n" + "="*80)
    print("Diagnostic complete!")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
