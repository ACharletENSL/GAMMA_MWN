#!/bin/bash
#SBATCH --job-name="regen_val"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=150G
#SBATCH --no-requeue
#SBATCH --output=slurm-%j.out
# Only the two steps that died with NameError in job 987740; every other fiducial figure
# from that job is valid and is not redone.
export MPLBACKEND=Agg
export GAMMACM_NPROC=${SLURM_CPUS_PER_TASK:-1}
export REGEN_NPROC=${SLURM_CPUS_PER_TASK:-1}
export REGEN_SKIP="sweep_gammacm,lightcurve_shape,mid_slope,sweep_shells,sweep_rarcut,sweep_compare,segment_route,sweep_efficiency"
export REGEN_KEY="${REGEN_KEY:?set REGEN_KEY}"
cd "$SLURM_SUBMIT_DIR/bin/Tools/project_v2"
python3 -u regen_all.py
