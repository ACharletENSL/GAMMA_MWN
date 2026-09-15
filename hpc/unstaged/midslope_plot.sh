#!/bin/bash
#SBATCH --job-name="msplot"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --output=slurm-%j.out
# Replot only: the tables are current, what was missing is the BIAS GRIDS, which live under
# bin/Tools/figures/ and are therefore git-ignored -- so they never reached this machine and
# every a_mid figure drawn here came out fully uncorrected.
export MPLBACKEND=Agg
cd "$SLURM_SUBMIT_DIR/bin/Tools/project_v2"
python3 -u -c "
import mid_slope_evolution as M
for k in ('cooling_g100', 'cooling_g100_hires'):
    print('--- '+k, flush=True); M.main(key=k, use_cache=True)
"
