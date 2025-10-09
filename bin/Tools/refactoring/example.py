# Example usage
from pathlib import Path

# Simple analysis
run = Run.load('sph_fid')
run.plot('lfac', xscale='r', theory=True).savefig('lfac.png')

# Full pipeline
pipeline = AnalysisPipeline('sph_fid')
pipeline.extract_cells() \
    .fit_evolution(['lfac', 'nu_m', 'Lth']) \
    .compute_radiation(nu_obs, T_obs) \
    .save(Path('./results')) \
    .plot_all(Path('./plots'))

# Access fits
rs_lfac_fit = pipeline.results['fits']['RS']['lfac']
print(f"Power-law index: {rs_lfac_fit.params['m']:.2f}")

# Custom plotting
from plotting import TimeSeriesPlotter
plotter = TimeSeriesPlotter(VARIABLE_REGISTRY['nu_m'])
fig, ax = plt.subplots()
for shock in ['RS', 'FS']:
    plotter.plot(run, shock=shock, ax=ax)
plt.legend()