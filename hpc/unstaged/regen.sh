#!/bin/bash
#SBATCH --job-name="regen"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=150G
#SBATCH --no-requeue
#SBATCH --output=slurm-%j.out
# Full article-figure regeneration for one REGEN_KEY, on the CORRECTED (_fc2) sweeps.
export MPLBACKEND=Agg
export GAMMACM_NPROC=${SLURM_CPUS_PER_TASK:-1}
export REGEN_NPROC=${SLURM_CPUS_PER_TASK:-1}
export REGEN_SKIP="sweep_efficiency,segment_route"
export REGEN_KEY="${REGEN_KEY:?set REGEN_KEY}"
cd "$SLURM_SUBMIT_DIR/bin/Tools/project_v2"
echo "=== regenerating key=$REGEN_KEY on ${SLURM_CPUS_PER_TASK} cpus ==="
python3 -u regen_all.py
