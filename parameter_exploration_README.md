# Parameter Space Exploration for GAMMA

This toolkit allows you to systematically explore the parameter space of `u1` and `u4/u1` ratio by running multiple GAMMA simulations and collecting the hydro fit results.

## Overview

The exploration script:
1. **Modifies** `phys_input.ini` with different values of `u1` and `u4` (based on the ratio)
2. **Runs** GAMMA simulations for each parameter combination
3. **Analyzes** results using `get_hydrofits_shell` from `project_v2/run_analysis`
4. **Saves** all results in a comprehensive CSV file

## Files

- **`parameter_space_exploration.py`**: Main exploration script
- **`run_quick_exploration.py`**: Quick test script with small parameter range
- **`parameter_exploration_README.md`**: This documentation file

## Requirements

- GAMMA compiled and available at `./bin/GAMMA`
- Python 3 with required packages:
  - numpy
  - pandas
  - scipy
  - more_itertools
- MPI installation (for running GAMMA)
- Sufficient disk space for multiple simulation results

## Usage

### Basic Usage

1. **Edit the parameter ranges** in the script:

```python
# In parameter_space_exploration.py, modify the main() function:
u1_values = np.array([50, 100, 200])  # Values of u1 to explore
u4_u1_ratios = np.array([1.0, 1.5, 2.0, 3.0])  # u4/u1 ratios to explore
```

2. **Run the exploration**:

```bash
python parameter_space_exploration.py
```

### Quick Test

To test with a small parameter range first:

```bash
python run_quick_exploration.py
```

This runs a minimal exploration (2 u1 values × 2 ratios = 4 simulations) to verify everything works.

### Custom Exploration

You can also use the `ParameterSpaceExplorer` class in your own scripts:

```python
from parameter_space_exploration import ParameterSpaceExplorer
import numpy as np

# Initialize
explorer = ParameterSpaceExplorer(
    base_ini_path='phys_input.ini',
    n_nodes=4  # Use 4 MPI nodes
)

# Define parameters
u1_values = np.linspace(50, 200, 10)  # 10 values from 50 to 200
u4_u1_ratios = np.linspace(1.0, 3.0, 10)  # 10 ratios from 1.0 to 3.0

# Run exploration
explorer.explore_parameter_space(u1_values, u4_u1_ratios)

# Save results
explorer.save_results('my_results.csv')
```

## Output Files

### Main Results File: `parameter_space_results.csv`

Contains one row per simulation with columns:

**Input Parameters:**
- `u1`: Value of u1 parameter
- `u4`: Value of u4 parameter
- `u4_u1_ratio`: The ratio u4/u1
- `key`: Unique identifier for this simulation
- `status`: 'success', 'failed', or 'analysis_failed'
- `timestamp`: When the simulation was run

**Fitting Parameters (for each front: FS and RS):**

For the **Lorentz factor fit** (lfac2_fit):
- `{FS/RS}_lfac2_fit_X_sph`: Spherical transition radius
- `{FS/RS}_lfac2_fit_alpha`: Power-law index
- `{FS/RS}_lfac2_fit_s`: Smoothing parameter

For the **shock strength fit** (ShSt_fit):
- `{FS/RS}_ShSt_fit_X_sph`: Spherical transition radius
- `{FS/RS}_ShSt_fit_alpha`: Power-law index
- `{FS/RS}_ShSt_fit_s`: Smoothing parameter

For the **power-law parameters**:
- `{FS/RS}_power_laws_lfac_m`: Lorentz factor power-law slope
- `{FS/RS}_power_laws_lfac_x`: Lorentz factor transition point
- `{FS/RS}_power_laws_ShSt_n`: Shock strength power-law slope
- `{FS/RS}_power_laws_ShSt_x`: Shock strength transition point

### Summary File: `parameter_space_results_summary.txt`

Contains overview statistics:
- Total number of simulations
- Number of successful/failed runs
- Timestamp
- Output file location

### Individual Simulation Results

Each simulation creates a subdirectory in `results/` named:
```
results/u1_{value}_ratio_{ratio}/
```

This directory contains:
- All simulation output files (`phys*.out`)
- Modified `phys_input.ini` used for this run
- Analysis files: `run_data_1.csv`, `run_data_4.csv`

## Understanding the Results

### Lorentz Factor Fit (lfac2_fit)

The Lorentz factor squared follows a smooth broken power law:
```
Γ²(R) = Γ₀² × smooth_bpl(R/R₀, X_sph, alpha, 0, s)
```

where:
- `X_sph`: Transition radius (in units of R₀) where behavior changes
- `alpha`: Power-law index (typically negative, indicating decay)
- `s`: Smoothness of the transition

### Shock Strength Fit (ShSt_fit)

The shock strength (Γ_upstream/Γ_downstream - 1) also follows a smooth broken power law with similar parameters.

### Typical Analysis Workflow

1. **Load the results**:
```python
import pandas as pd
df = pd.read_csv('parameter_space_results.csv')
successful = df[df['status'] == 'success']
```

2. **Plot parameter dependencies**:
```python
import matplotlib.pyplot as plt

# Plot how FS power-law index varies with u1
plt.scatter(successful['u1'], successful['FS_power_laws_lfac_m'])
plt.xlabel('u1')
plt.ylabel('FS Lorentz factor power-law index')
plt.show()

# Plot how RS transition radius varies with ratio
plt.scatter(successful['u4_u1_ratio'], successful['RS_lfac2_fit_X_sph'])
plt.xlabel('u4/u1 ratio')
plt.ylabel('RS transition radius')
plt.show()
```

3. **Create parameter space heatmaps**:
```python
import numpy as np
import matplotlib.pyplot as plt

# Pivot data for heatmap
pivot = successful.pivot_table(
    values='FS_lfac2_fit_alpha',
    index='u1',
    columns='u4_u1_ratio'
)

plt.imshow(pivot, aspect='auto', origin='lower')
plt.colorbar(label='FS alpha parameter')
plt.xlabel('u4/u1 ratio')
plt.ylabel('u1')
plt.title('Parameter Space: FS alpha')
plt.show()
```

## Configuration

### Adjusting Simulation Settings

Modify `phys_input.ini` to set:
- Grid resolution (`Nsh1`, `Next`)
- Maximum iterations (`itmax`)
- Output frequency (`itdump`)
- Other physical parameters (keep constant during exploration)

### Adjusting MPI Nodes

In the script initialization:
```python
explorer = ParameterSpaceExplorer(n_nodes=8)  # Use 8 MPI nodes
```

### Changing Analyzed Cells

By default, both FS (cell 1) and RS (cell 4) are analyzed. To analyze different cells:
```python
explorer.explore_parameter_space(
    u1_values=u1_values,
    u4_u1_ratios=u4_u1_ratios,
    cells=[1, 2, 3, 4]  # Analyze all 4 cells
)
```

Cell mapping:
- 1: Downstream of Forward Shock (FS)
- 2: Contact Discontinuity in Shell 2
- 3: Contact Discontinuity in Shell 3
- 4: Downstream of Reverse Shock (RS)

## Troubleshooting

### Simulation Failures

If simulations fail:
1. Check that GAMMA is compiled: `ls -l bin/GAMMA`
2. Verify `results/Last` directory exists: `mkdir -p results/Last`
3. Test a single simulation manually: `mpirun -n 1 ./bin/GAMMA -w`
4. Check the console output for error messages

### Analysis Failures

If analysis fails but simulation succeeds:
1. Verify output files exist in `results/{key}/`
2. Check that simulation ran long enough (check `itmax`)
3. Ensure shocked regions are present in the data
4. Check Python dependencies are installed

### Memory Issues

For large parameter spaces:
- Run in smaller batches
- Increase system swap space
- Use fewer MPI nodes per simulation
- Delete old simulation outputs: `rm -rf results/u1_*`

## Performance Considerations

### Estimated Runtime

For a simulation with:
- `itmax = 40000`
- `Nsh1 = 450`
- Running on 1 MPI node

Expect ~10-30 minutes per simulation.

For a full exploration of 10×10 = 100 parameter combinations, expect ~20-50 hours.

### Parallelization

The script runs simulations **sequentially**. To parallelize:

1. Split parameter space into chunks
2. Run multiple instances of the script on different machines/nodes
3. Manually specify different parameter ranges in each instance
4. Combine results afterward

Example parallel execution:
```bash
# Machine 1
python parameter_space_exploration.py --u1-range 50 100
# Machine 2  
python parameter_space_exploration.py --u1-range 100 150
# Machine 3
python parameter_space_exploration.py --u1-range 150 200
```

(Note: You would need to modify the script to accept command-line arguments for this)

## Citation

If you use this tool in your research, please cite the GAMMA code paper:
- GAMMA: a new method for modeling relativistic hydrodynamics and non-thermal emission on a moving mesh ([arXiv:2104.09397](https://arxiv.org/abs/2104.09397))

## Contact

For issues or questions about this exploration toolkit, refer to the main GAMMA documentation or contact the GAMMA developers.
