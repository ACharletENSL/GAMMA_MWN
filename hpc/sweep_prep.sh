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
#
# Staged version of sweep_prep_k.sh: the cells are copied to the node's /tmp first and
# whatever this writes is copied back at the end (hpc/README.md). STAGE_SHELL=$ZSH stages
# only the shell being computed, which is what makes the hi-res run fit at all.
set -u
export MPLBACKEND=Agg
export GAMMACM_NPROC=${SLURM_CPUS_PER_TASK:-1}

STAGE_WORK="${SLURM_SUBMIT_DIR:-$PWD}"
. "$STAGE_WORK/hpc/stage.sh"
export STAGE_SHELL="${ZSH}"
stage_open "prep.$RUNKEY.z$ZSH"
stage_analysis_in "$RUNKEY" || exit 2
stage_analysis_cd || exit 1

python3 -u -c "
import os, sweep_gammacm as S
S.run_sweep(os.environ[\"RUNKEY\"], [-3.], z=int(os.environ[\"ZSH\"]),
            method=os.environ[\"METHOD\"], nproc=int(os.environ[\"GAMMACM_NPROC\"]))
print(\"prep done:\", os.environ[\"RUNKEY\"], \"z =\", os.environ[\"ZSH\"])
"
rc=$?
stage_analysis_out "$RUNKEY"
exit $rc
