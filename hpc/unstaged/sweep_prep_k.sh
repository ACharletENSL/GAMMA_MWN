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
# FORCE=1 recomputes points that already have a cache. WITHOUT IT run_sweep
# returns in ~5 s per point having skipped everything (skip_cached defaults to
# True), which looks exactly like a successful recompute in sacct.
export MPLBACKEND=Agg
export GAMMACM_NPROC=${SLURM_CPUS_PER_TASK:-1}
cd "$SLURM_SUBMIT_DIR/bin/Tools/project_v2"
python3 -u -c "
import os, sweep_gammacm as S
S.run_sweep(os.environ[\"RUNKEY\"], [-3.], z=int(os.environ[\"ZSH\"]),
            method=os.environ[\"METHOD\"], nproc=int(os.environ[\"GAMMACM_NPROC\"]),
            skip_cached=(os.environ.get(\"FORCE\", \"0\") != \"1\"))
print(\"prep done:\", os.environ[\"RUNKEY\"], \"z =\", os.environ[\"ZSH\"])
"
