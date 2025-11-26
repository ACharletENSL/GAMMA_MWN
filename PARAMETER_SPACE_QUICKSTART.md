# Parameter Space Exploration - Quick Start Guide

This guide will get you up and running with parameter space exploration in 5 minutes.

## What You'll Need

✓ GAMMA compiled (`./bin/GAMMA` exists)  
✓ Python 3 with numpy, pandas, scipy  
✓ Directory `results/Last` created  
✓ Basic familiarity with `phys_input.ini`

## Quick Start (3 Steps)

### 1. Verify Your Setup

```bash
# Check GAMMA is compiled
ls -l bin/GAMMA

# Create results directory if needed
mkdir -p results/Last

# Check Python packages
python -c "import numpy, pandas, scipy; print('✓ All packages installed')"
```

### 2. Run a Quick Test (4 simulations)

```bash
python run_quick_exploration.py
```

This will:
- Run 4 simulations with u1=[100, 200] and ratios=[1.5, 2.0]
- Take ~30-60 minutes depending on your system
- Create `quick_test_results.csv`

**Expected output:**
```
[1/4] Running simulation: u1_100.0_ratio_1.50
  u1 = 100.0, u4 = 150.0, u4/u1 = 1.5
  Simulation completed successfully
  Analysis completed successfully
[2/4] Running simulation: u1_100.0_ratio_2.00
  ...
```

### 3. Run Full Exploration

Edit `parameter_space_exploration.py` to set your parameter ranges:

```python
# Around line 320
u1_values = np.array([50, 100, 150, 200])  # Your u1 values
u4_u1_ratios = np.array([1.0, 1.5, 2.0, 2.5, 3.0])  # Your ratios
```

Then run:

```bash
python parameter_space_exploration.py
```

## Understanding the Output

### Results File: `parameter_space_results.csv`

| u1  | u4  | u4_u1_ratio | FS_lfac2_fit_alpha | FS_ShSt_fit_alpha | RS_lfac2_fit_alpha | ... |
|-----|-----|-------------|-------------------|------------------|-------------------|-----|
| 100 | 150 | 1.5         | -0.234            | -0.567           | -0.189            | ... |
| 100 | 200 | 2.0         | -0.221            | -0.543           | -0.198            | ... |

### Key Columns

**Input parameters:**
- `u1`, `u4`, `u4_u1_ratio` - Your input parameters
- `status` - 'success', 'failed', or 'analysis_failed'

**Forward Shock (FS) results:**
- `FS_lfac2_fit_alpha` - Lorentz factor power-law index
- `FS_lfac2_fit_X_sph` - Transition radius
- `FS_ShSt_fit_alpha` - Shock strength power-law index

**Reverse Shock (RS) results:**
- Same structure with `RS_` prefix

## Visualizing Results

### Quick Visualization

```bash
python visualize_parameter_space.py parameter_space_results.csv
```

This creates a `parameter_space_plots/` directory with all plots.

### Custom Analysis in Python

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv('parameter_space_results.csv')
successful = df[df['status'] == 'success']

# Plot FS alpha vs u1
plt.figure(figsize=(8, 6))
plt.scatter(successful['u1'], successful['FS_lfac2_fit_alpha'], 
           c=successful['u4_u1_ratio'], cmap='viridis', s=100)
plt.colorbar(label='u4/u1 ratio')
plt.xlabel('u1')
plt.ylabel('FS Lorentz factor power-law index')
plt.title('Parameter Space: FS α')
plt.grid(True, alpha=0.3)
plt.show()
```

## Common Issues & Solutions

### ❌ "Simulation failed"
**Solution:** Check that:
```bash
make -B  # Recompile GAMMA
mkdir -p results/Last  # Ensure directory exists
mpirun -n 1 ./bin/GAMMA -w  # Test single run manually
```

### ❌ "Analysis failed"
**Solution:** Simulation may not have run long enough. In `phys_input.ini`:
```ini
itmax       80000    # Increase from 40000
itdump      100      # Keep or adjust
```

### ❌ "Import error: No module named 'project_v2'"
**Solution:** Run from the GAMMA root directory:
```bash
cd /path/to/GAMMA
python parameter_space_exploration.py
```

### ❌ Scripts run too slow
**Solution:** Use more MPI nodes. Edit the script:
```python
explorer = ParameterSpaceExplorer(n_nodes=4)  # Use 4 nodes instead of 1
```

## File Structure

After running, you'll have:

```
GAMMA/
├── parameter_space_exploration.py      # Main script
├── run_quick_exploration.py           # Test script
├── visualize_parameter_space.py       # Plotting script
├── parameter_space_results.csv        # Results
├── parameter_space_results_summary.txt
├── parameter_space_plots/             # Generated plots
│   ├── FS_lfac2_fit_alpha_dependencies.png
│   ├── FS_lfac2_fit_alpha_heatmap.png
│   └── ...
├── results/
│   ├── Last/                          # Temporary simulation files
│   ├── u1_100.0_ratio_1.50/          # Individual run results
│   │   ├── phys*.out
│   │   ├── run_data_1.csv
│   │   ├── run_data_4.csv
│   │   └── phys_input.ini
│   └── u1_100.0_ratio_2.00/
│       └── ...
```

## Next Steps

1. **Explore specific parameters**: Edit the ranges in the main script
2. **Analyze trends**: Use the visualization script or load CSV in your favorite tool
3. **Compare with theory**: Cross-reference with analytical predictions
4. **Publication**: Use the generated plots directly or as starting points

## Advanced Usage

### Parallel Execution on Multiple Machines

Split your parameter space:

**Machine 1:**
```python
u1_values = np.array([50, 100, 150])  # First half
```

**Machine 2:**
```python
u1_values = np.array([200, 250, 300])  # Second half
```

Then merge CSVs:
```python
df1 = pd.read_csv('machine1_results.csv')
df2 = pd.read_csv('machine2_results.csv')
combined = pd.concat([df1, df2])
combined.to_csv('full_results.csv', index=False)
```

### Custom Analysis Functions

Add to the explorer class:
```python
def custom_metric(self, key):
    """Calculate a custom metric from simulation results"""
    data_fs = open_rundata(key, 1)
    data_rs = open_rundata(key, 4)
    # Your analysis here
    return custom_value
```

### Automated Post-Processing

Create a pipeline:
```bash
# 1. Run exploration
python parameter_space_exploration.py

# 2. Generate plots
python visualize_parameter_space.py

# 3. Generate report
python generate_report.py parameter_space_results.csv
```

## Support

For detailed documentation, see:
- `parameter_exploration_README.md` - Full documentation
- GAMMA README.md - General GAMMA usage
- `bin/Tools/project_v2/` - Analysis tools

## Citation

If you use this in research, cite the GAMMA paper:
```bibtex
@article{ayache2021gamma,
  title={GAMMA: a new method for modeling relativistic hydrodynamics and 
         non-thermal emission on a moving mesh},
  author={Ayache, E. et al.},
  journal={arXiv preprint arXiv:2104.09397},
  year={2021}
}
```

---

**Questions?** Check the full README or contact the GAMMA developers.

**Ready to explore?** `python run_quick_exploration.py` 🚀
