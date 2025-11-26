#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parameter Space Exploration Script
===================================
This script explores the parameter space of u1 and u4/u1 ratio by:
1. Running simulations for each parameter combination
2. Analyzing results using get_hydrofits_shell
3. Saving all results to a comprehensive output file
"""

import os
import sys
import numpy as np
import pandas as pd
import subprocess
import shutil
from datetime import datetime
from pathlib import Path

# Add project_v2 to path
GAMMA_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(GAMMA_dir / 'bin' / 'Tools' / 'project_v2'))

from run_analysis import analyze_run, extract_plaws
from IO import open_rundata, get_physfile
from hydro_fits import get_hydrofits_shell

class ParameterSpaceExplorer:
    """
    Class to manage parameter space exploration for GAMMA simulations
    """
    
    def __init__(self, base_ini_path='phys_input.ini', n_nodes=1):
        """
        Initialize the explorer
        
        Parameters:
        -----------
        base_ini_path : str
            Path to the base phys_input.ini file
        n_nodes : int
            Number of MPI nodes to use for simulations
        """
        self.base_ini_path = base_ini_path
        self.n_nodes = n_nodes
        self.gamma_dir = GAMMA_dir
        self.results_base = self.gamma_dir / 'results'
        self.results_base.mkdir(exist_ok=True)
        
        # Storage for results
        self.results_list = []
        
    def modify_ini_file(self, u1, u4, output_path):
        """
        Modify phys_input.ini with new u1 and u4 values
        
        Parameters:
        -----------
        u1 : float
            Value for u1 parameter
        u4 : float
            Value for u4 parameter
        output_path : str
            Path to write modified ini file
        """
        with open(self.base_ini_path, 'r') as f:
            lines = f.readlines()
        
        modified_lines = []
        for line in lines:
            if line.strip().startswith('u1'):
                # Preserve the spacing/format
                parts = line.split()
                if len(parts) >= 2:
                    modified_lines.append(f"u1          {u1}\n")
                else:
                    modified_lines.append(line)
            elif line.strip().startswith('u4'):
                parts = line.split()
                if len(parts) >= 2:
                    modified_lines.append(f"u4          {u4}\n")
                else:
                    modified_lines.append(line)
            else:
                modified_lines.append(line)
        
        with open(output_path, 'w') as f:
            f.writelines(modified_lines)
    
    def run_simulation(self, key):
        """
        Run GAMMA simulation
        
        Parameters:
        -----------
        key : str
            Simulation key/name (subfolder in results/)
            
        Returns:
        --------
        success : bool
            True if simulation completed successfully
        """
        # Ensure results directory exists
        results_dir = self.results_base / key
        results_dir.mkdir(exist_ok=True)
        
        # Run simulation
        cmd = f"mpirun -n {self.n_nodes} {self.gamma_dir}/bin/GAMMA -w"
        
        print(f"Running simulation: {cmd}")
        print(f"Results will be saved to: {results_dir}")
        
        try:
            result = subprocess.run(
                cmd, 
                shell=True, 
                cwd=self.gamma_dir,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode == 0:
                print(f"Simulation completed successfully for {key}")
                # Move results from Last to key folder
                last_dir = self.results_base / 'Last'
                if last_dir.exists():
                    # Copy all output files to the key directory
                    for file in last_dir.glob('*.out'):
                        shutil.copy2(file, results_dir)
                return True
            else:
                print(f"Simulation failed for {key}")
                print(f"Error: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"Simulation timed out for {key}")
            return False
        except Exception as e:
            print(f"Error running simulation for {key}: {e}")
            return False
    
    def analyze_results(self, key, cells=[1, 4]):
        """
        Analyze simulation results and extract hydro fits
        
        Parameters:
        -----------
        key : str
            Simulation key/name
        cells : list
            List of cell zones to analyze (1=FS, 4=RS)
            
        Returns:
        --------
        results : dict
            Dictionary containing fitting parameters for each cell
        """
        results = {}
        
        try:
            # First run analyze_run to generate the run_data CSV files
            print(f"Analyzing run {key}...")
            analyze_run(key, cells=cells, savefile=True)
            
            # Extract hydro fits for each cell
            for z in cells:
                print(f"Extracting hydro fits for cell {z}...")
                data = open_rundata(key, z)
                
                if isinstance(data, bool) and not data:
                    print(f"No data available for cell {z}")
                    continue
                
                # Get fitting parameters
                popt_lfac2, popt_ShSt = get_hydrofits_shell(data)
                
                # Also get power law parameters
                plaw_lf, plaw_sh = extract_plaws(key, z)
                
                # Store results
                cell_name = 'FS' if z == 1 else 'RS'
                results[cell_name] = {
                    'lfac2_fit': {
                        'X_sph': popt_lfac2[0],
                        'alpha': popt_lfac2[1],
                        's': popt_lfac2[2]
                    },
                    'ShSt_fit': {
                        'X_sph': popt_ShSt[0],
                        'alpha': popt_ShSt[1],
                        's': popt_ShSt[2]
                    },
                    'power_laws': {
                        'lfac_m': plaw_lf[0],
                        'lfac_x': plaw_lf[1],
                        'ShSt_n': plaw_sh[0],
                        'ShSt_x': plaw_sh[1]
                    }
                }
                
            return results
            
        except Exception as e:
            print(f"Error analyzing results for {key}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def explore_parameter_space(self, u1_values, u4_u1_ratios, cells=[1, 4]):
        """
        Explore parameter space by running simulations for all combinations
        
        Parameters:
        -----------
        u1_values : array-like
            Array of u1 values to explore
        u4_u1_ratios : array-like
            Array of u4/u1 ratio values to explore
        cells : list
            List of cell zones to analyze (1=FS, 4=RS)
        """
        total_sims = len(u1_values) * len(u4_u1_ratios)
        sim_count = 0
        
        print(f"Starting parameter space exploration")
        print(f"Total simulations: {total_sims}")
        print(f"u1 values: {u1_values}")
        print(f"u4/u1 ratios: {u4_u1_ratios}")
        print("="*60)
        
        for u1 in u1_values:
            for ratio in u4_u1_ratios:
                sim_count += 1
                u4 = u1 * ratio
                
                # Create unique key for this simulation
                key = f"u1_{u1:.1f}_ratio_{ratio:.2f}"
                
                print(f"\n[{sim_count}/{total_sims}] Running simulation: {key}")
                print(f"  u1 = {u1}, u4 = {u4}, u4/u1 = {ratio}")
                
                # Modify ini file in results directory
                results_dir = self.results_base / key
                results_dir.mkdir(exist_ok=True)
                ini_path = results_dir / 'phys_input.ini'
                self.modify_ini_file(u1, u4, ini_path)
                
                # Also modify the main ini file for the simulation
                self.modify_ini_file(u1, u4, self.base_ini_path)
                
                # Run simulation
                success = self.run_simulation(key)
                
                if not success:
                    print(f"  Skipping analysis due to simulation failure")
                    self.results_list.append({
                        'u1': u1,
                        'u4': u4,
                        'u4_u1_ratio': ratio,
                        'key': key,
                        'status': 'failed',
                        'timestamp': datetime.now().isoformat()
                    })
                    continue
                
                # Analyze results
                analysis_results = self.analyze_results(key, cells)
                
                if analysis_results is None:
                    print(f"  Analysis failed for {key}")
                    self.results_list.append({
                        'u1': u1,
                        'u4': u4,
                        'u4_u1_ratio': ratio,
                        'key': key,
                        'status': 'analysis_failed',
                        'timestamp': datetime.now().isoformat()
                    })
                    continue
                
                # Store results
                result_entry = {
                    'u1': u1,
                    'u4': u4,
                    'u4_u1_ratio': ratio,
                    'key': key,
                    'status': 'success',
                    'timestamp': datetime.now().isoformat()
                }
                
                # Flatten the nested dictionary for easier CSV export
                for cell_name, cell_data in analysis_results.items():
                    for fit_type, fit_params in cell_data.items():
                        for param_name, param_value in fit_params.items():
                            col_name = f"{cell_name}_{fit_type}_{param_name}"
                            result_entry[col_name] = param_value
                
                self.results_list.append(result_entry)
                print(f"  Analysis completed successfully")
        
        print("\n" + "="*60)
        print("Parameter space exploration completed!")
    
    def save_results(self, output_file='parameter_space_results.csv'):
        """
        Save all results to a CSV file
        
        Parameters:
        -----------
        output_file : str
            Path to output CSV file
        """
        if not self.results_list:
            print("No results to save!")
            return
        
        df = pd.DataFrame(self.results_list)
        
        # Save to CSV
        output_path = self.gamma_dir / output_file
        df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
        print(f"Total entries: {len(df)}")
        print(f"Successful simulations: {len(df[df['status'] == 'success'])}")
        
        # Also save a summary
        summary_file = output_path.with_name(output_path.stem + '_summary.txt')
        with open(summary_file, 'w') as f:
            f.write("Parameter Space Exploration Summary\n")
            f.write("="*60 + "\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total simulations: {len(df)}\n")
            f.write(f"Successful: {len(df[df['status'] == 'success'])}\n")
            f.write(f"Failed: {len(df[df['status'] != 'success'])}\n\n")
            f.write("Results file: " + str(output_path) + "\n")
        
        print(f"Summary saved to: {summary_file}")
        
        return df


def main():
    """
    Main function with example usage
    """
    # Define parameter ranges to explore
    # Adjust these according to your needs
    u1_values = np.array([50, 100, 200])  # Example: 3 values of u1
    u4_u1_ratios = np.array([1.0, 1.5, 2.0, 3.0])  # Example: 4 different ratios
    
    # For testing, you might want to use a smaller range:
    # u1_values = np.array([100])
    # u4_u1_ratios = np.array([2.0])
    
    # Initialize explorer
    explorer = ParameterSpaceExplorer(
        base_ini_path='phys_input.ini',
        n_nodes=1  # Adjust based on your system
    )
    
    # Run exploration
    explorer.explore_parameter_space(
        u1_values=u1_values,
        u4_u1_ratios=u4_u1_ratios,
        cells=[1, 4]  # Analyze both FS and RS
    )
    
    # Save results
    df = explorer.save_results('parameter_space_results.csv')
    
    # Print preview of results
    if df is not None:
        print("\n" + "="*60)
        print("Results Preview:")
        print(df.head())


if __name__ == '__main__':
    main()
