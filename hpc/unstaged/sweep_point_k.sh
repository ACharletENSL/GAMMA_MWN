#!/bin/bash
#SBATCH --job-name="swp_pt"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=150G
#SBATCH --no-requeue
#SBATCH --output=slurm-%A_%a.out
export MPLBACKEND=Agg
export GAMMACM_NPROC=${SLURM_CPUS_PER_TASK:-1}
LOGRS=(-5 -4 -2 -1 0 1 2 3)
export LOGR=${LOGRS[$SLURM_ARRAY_TASK_ID]}
cd "$SLURM_SUBMIT_DIR/bin/Tools/project_v2"
python3 -u -c "
import os, sweep_gammacm as S
S.run_sweep(os.environ[\"RUNKEY\"], [float(os.environ[\"LOGR\"])], z=int(os.environ[\"ZSH\"]),
            method=os.environ[\"METHOD\"], nproc=int(os.environ[\"GAMMACM_NPROC\"]))
print(\"point done:\", os.environ[\"RUNKEY\"], \"z =\", os.environ[\"ZSH\"], \"logr =\", os.environ[\"LOGR\"])
"
