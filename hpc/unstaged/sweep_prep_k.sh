#!/bin/bash
#SBATCH --job-name="swp_prep"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=150G
#SBATCH --no-requeue
#SBATCH --output=slurm-%j.out
# One sweep point (logr = -3) for shell $ZSH of run $RUNKEY, method $METHOD: it builds the
# shared prologue (cellscan, and the rarefaction head for the cut methods) that every point
# writes on the same path, so the array must not be what creates them.
export MPLBACKEND=Agg
export GAMMACM_NPROC=${SLURM_CPUS_PER_TASK:-1}
cd "$SLURM_SUBMIT_DIR/bin/Tools/project_v2"
python3 -u -c "
import os, sweep_gammacm as S
S.run_sweep(os.environ[\"RUNKEY\"], [-3.], z=int(os.environ[\"ZSH\"]),
            method=os.environ[\"METHOD\"], nproc=int(os.environ[\"GAMMACM_NPROC\"]))
print(\"prep done:\", os.environ[\"RUNKEY\"], \"z =\", os.environ[\"ZSH\"])
"
