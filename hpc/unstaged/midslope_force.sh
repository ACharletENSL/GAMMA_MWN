#!/bin/bash
#SBATCH --job-name="midslope"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --output=slurm-%j.out
# Force a RE-MEASURE of mid_slopes.csv: table_is_current only stamps the table against the
# POINT CACHE, so a code change (here the FC* 2brk_flo shape) leaves a stale table looking
# current and the figures get redrawn from it.
export MPLBACKEND=Agg
cd "$SLURM_SUBMIT_DIR/bin/Tools/project_v2"
python3 -u -c "
import mid_slope_evolution as M
M.main(key='cooling_g100_hires', use_cache=False)
"
