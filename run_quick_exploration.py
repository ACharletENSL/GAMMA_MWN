#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Parameter Space Exploration Test
========================================
This script runs a minimal parameter space exploration for testing purposes.
Uses a small 2x2 parameter grid to verify the system is working correctly.
"""

import numpy as np
from parameter_space_exploration import ParameterSpaceExplorer

def main():
    """
    Run a quick test exploration with minimal parameter ranges
    """
    print("="*60)
    print("Quick Parameter Space Exploration Test")
    print("="*60)
    print("\nThis will run 4 simulations to test the system:")
    print("  - u1: [100, 200]")
    print("  - u4/u1 ratios: [1.5, 2.0]")
    print("\nExpected runtime: ~30-60 minutes (depending on your system)")
    print("="*60)
    
    # Small parameter ranges for testing
    u1_values = np.array([100, 200])
    u4_u1_ratios = np.array([1.5, 2.0])
    
    # Initialize explorer
    explorer = ParameterSpaceExplorer(
        base_ini_path='phys_input.ini',
        n_nodes=1  # Use 1 node for testing
    )
    
    # Run exploration
    explorer.explore_parameter_space(
        u1_values=u1_values,
        u4_u1_ratios=u4_u1_ratios,
        cells=[1, 4]  # Analyze FS and RS
    )
    
    # Save results
    df = explorer.save_results('quick_test_results.csv')
    
    # Print results summary
    if df is not None:
        print("\n" + "="*60)
        print("Quick Test Results:")
        print("="*60)
        print(f"\nTotal simulations: {len(df)}")
        print(f"Successful: {len(df[df['status'] == 'success'])}")
        print(f"Failed: {len(df[df['status'] != 'success'])}")
        
        if len(df[df['status'] == 'success']) > 0:
            print("\n✓ Test completed successfully!")
            print("  You can now run the full parameter space exploration.")
            print("\nSuccessful parameter combinations:")
            success_df = df[df['status'] == 'success'][['u1', 'u4', 'u4_u1_ratio']]
            print(success_df.to_string(index=False))
        else:
            print("\n✗ All simulations failed.")
            print("  Please check the error messages above and verify:")
            print("  1. GAMMA is compiled (./bin/GAMMA exists)")
            print("  2. results/Last directory exists")
            print("  3. MPI is working correctly")


if __name__ == '__main__':
    main()
