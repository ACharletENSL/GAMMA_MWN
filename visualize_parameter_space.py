#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parameter Space Visualization
==============================
This script provides visualization tools for the parameter space exploration results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

class ParameterSpaceVisualizer:
    """
    Visualize results from parameter space exploration
    """
    
    def __init__(self, results_file='parameter_space_results.csv'):
        """
        Load results from CSV file
        
        Parameters:
        -----------
        results_file : str
            Path to the results CSV file
        """
        self.results_file = results_file
        self.df = pd.read_csv(results_file)
        self.successful = self.df[self.df['status'] == 'success'].copy()
        
        print(f"Loaded {len(self.df)} results")
        print(f"  Successful: {len(self.successful)}")
        print(f"  Failed: {len(self.df) - len(self.successful)}")
    
    def plot_parameter_dependencies(self, variable, front='FS', figsize=(12, 4)):
        """
        Plot how a variable depends on u1 and u4/u1 ratio
        
        Parameters:
        -----------
        variable : str
            Variable to plot (e.g., 'lfac2_fit_alpha', 'power_laws_lfac_m')
        front : str
            'FS' or 'RS'
        """
        if len(self.successful) == 0:
            print("No successful results to plot!")
            return
        
        col_name = f"{front}_{variable}"
        if col_name not in self.successful.columns:
            print(f"Column {col_name} not found!")
            print(f"Available columns: {[c for c in self.successful.columns if front in c]}")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot vs u1
        ax = axes[0]
        for ratio in sorted(self.successful['u4_u1_ratio'].unique()):
            subset = self.successful[self.successful['u4_u1_ratio'] == ratio]
            ax.plot(subset['u1'], subset[col_name], 'o-', label=f'u4/u1={ratio:.2f}')
        ax.set_xlabel('u1')
        ax.set_ylabel(variable)
        ax.set_title(f'{front} {variable} vs u1')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot vs ratio
        ax = axes[1]
        for u1 in sorted(self.successful['u1'].unique()):
            subset = self.successful[self.successful['u1'] == u1]
            ax.plot(subset['u4_u1_ratio'], subset[col_name], 'o-', label=f'u1={u1:.1f}')
        ax.set_xlabel('u4/u1 ratio')
        ax.set_ylabel(variable)
        ax.set_title(f'{front} {variable} vs u4/u1 ratio')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_heatmap(self, variable, front='FS', figsize=(8, 6)):
        """
        Create a 2D heatmap of a variable across the parameter space
        
        Parameters:
        -----------
        variable : str
            Variable to plot
        front : str
            'FS' or 'RS'
        """
        if len(self.successful) == 0:
            print("No successful results to plot!")
            return
        
        col_name = f"{front}_{variable}"
        if col_name not in self.successful.columns:
            print(f"Column {col_name} not found!")
            return
        
        # Pivot data for heatmap
        pivot = self.successful.pivot_table(
            values=col_name,
            index='u1',
            columns='u4_u1_ratio'
        )
        
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(pivot, aspect='auto', origin='lower', cmap='viridis')
        
        # Set ticks
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f'{r:.2f}' for r in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f'{u:.1f}' for u in pivot.index])
        
        ax.set_xlabel('u4/u1 ratio')
        ax.set_ylabel('u1')
        ax.set_title(f'{front} {variable}')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(variable)
        
        # Add values as text
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                if not np.isnan(pivot.iloc[i, j]):
                    text = ax.text(j, i, f'{pivot.iloc[i, j]:.3f}',
                                 ha="center", va="center", color="w", fontsize=8)
        
        plt.tight_layout()
        return fig
    
    def plot_all_fits(self, front='FS', figsize=(15, 10)):
        """
        Create a comprehensive plot of all fitting parameters
        
        Parameters:
        -----------
        front : str
            'FS' or 'RS'
        """
        variables = ['lfac2_fit_X_sph', 'lfac2_fit_alpha', 'lfac2_fit_s',
                    'ShSt_fit_X_sph', 'ShSt_fit_alpha', 'ShSt_fit_s']
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        for i, var in enumerate(variables):
            col_name = f"{front}_{var}"
            if col_name not in self.successful.columns:
                continue
            
            ax = axes[i]
            
            # Create scatter plot colored by ratio
            scatter = ax.scatter(self.successful['u1'], 
                               self.successful[col_name],
                               c=self.successful['u4_u1_ratio'],
                               cmap='viridis', s=100, alpha=0.6)
            
            ax.set_xlabel('u1')
            ax.set_ylabel(var)
            ax.set_title(f'{front} {var}')
            ax.grid(True, alpha=0.3)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('u4/u1 ratio')
        
        plt.tight_layout()
        return fig
    
    def compare_fronts(self, variable='lfac2_fit_alpha', figsize=(12, 5)):
        """
        Compare a variable between FS and RS
        
        Parameters:
        -----------
        variable : str
            Variable to compare (without front prefix)
        """
        fs_col = f"FS_{variable}"
        rs_col = f"RS_{variable}"
        
        if fs_col not in self.successful.columns or rs_col not in self.successful.columns:
            print(f"Columns not found!")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Scatter plot FS vs RS
        ax = axes[0]
        scatter = ax.scatter(self.successful[fs_col], 
                           self.successful[rs_col],
                           c=self.successful['u4_u1_ratio'],
                           cmap='viridis', s=100, alpha=0.6)
        ax.plot([self.successful[fs_col].min(), self.successful[fs_col].max()],
               [self.successful[fs_col].min(), self.successful[fs_col].max()],
               'k--', alpha=0.3, label='FS = RS')
        ax.set_xlabel(f'FS {variable}')
        ax.set_ylabel(f'RS {variable}')
        ax.set_title('FS vs RS comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('u4/u1 ratio')
        
        # Ratio plot
        ax = axes[1]
        ratio = self.successful[rs_col] / self.successful[fs_col]
        scatter = ax.scatter(self.successful['u1'], ratio,
                           c=self.successful['u4_u1_ratio'],
                           cmap='viridis', s=100, alpha=0.6)
        ax.axhline(y=1, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('u1')
        ax.set_ylabel(f'RS/FS ratio')
        ax.set_title(f'RS/FS ratio for {variable}')
        ax.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('u4/u1 ratio')
        
        plt.tight_layout()
        return fig
    
    def save_all_plots(self, output_dir='parameter_space_plots'):
        """
        Generate and save all plots
        
        Parameters:
        -----------
        output_dir : str
            Directory to save plots
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"Generating plots in {output_path}/...")
        
        # Plot key variables for both fronts
        variables = [
            ('lfac2_fit_alpha', 'Lorentz factor power-law index'),
            ('lfac2_fit_X_sph', 'Lorentz factor transition radius'),
            ('ShSt_fit_alpha', 'Shock strength power-law index'),
            ('power_laws_lfac_m', 'Lorentz factor slope'),
        ]
        
        for front in ['FS', 'RS']:
            for var, description in variables:
                # Dependency plots
                fig = self.plot_parameter_dependencies(var, front)
                if fig:
                    filename = output_path / f'{front}_{var}_dependencies.png'
                    fig.savefig(filename, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    print(f"  Saved: {filename}")
                
                # Heatmap
                fig = self.plot_heatmap(var, front)
                if fig:
                    filename = output_path / f'{front}_{var}_heatmap.png'
                    fig.savefig(filename, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    print(f"  Saved: {filename}")
            
            # All fits overview
            fig = self.plot_all_fits(front)
            if fig:
                filename = output_path / f'{front}_all_fits.png'
                fig.savefig(filename, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"  Saved: {filename}")
        
        # Comparison plots
        for var, description in variables:
            fig = self.compare_fronts(var)
            if fig:
                filename = output_path / f'comparison_{var}.png'
                fig.savefig(filename, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"  Saved: {filename}")
        
        print(f"\nAll plots saved to {output_path}/")


def main():
    """
    Example usage
    """
    import sys
    
    # Check if results file is provided
    if len(sys.argv) > 1:
        results_file = sys.argv[1]
    else:
        results_file = 'parameter_space_results.csv'
    
    # Check if file exists
    if not Path(results_file).exists():
        print(f"Error: Results file '{results_file}' not found!")
        print("\nUsage: python visualize_parameter_space.py [results_file.csv]")
        return
    
    # Load and visualize
    viz = ParameterSpaceVisualizer(results_file)
    
    if len(viz.successful) == 0:
        print("No successful results to visualize!")
        return
    
    # Generate all plots
    viz.save_all_plots()
    
    # Show an example interactive plot
    print("\nGenerating interactive plot example...")
    fig = viz.plot_all_fits('FS')
    if fig:
        print("Close the plot window to continue...")
        plt.show()


if __name__ == '__main__':
    main()
