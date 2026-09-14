#!/bin/bash
#SBATCH --job-name="swp_pt"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=150G
#SBATCH --no-requeue
#SBATCH --output=slurm-%A_%a.out
# Staged version of sweep_point_k.sh. Every task of this array reads EVERY cell of the run
# once -- eight tasks x 20 000 opens at ~2.3 s of NFS latency each is the load the file
# server went down under -- so each task copies the shell it needs to its own node's /tmp
# and works from there. Two tasks landing on the same node will not both fit (one shell is
# 165 GiB against ~208 GiB of /tmp); the second one notices and reads from work instead.
set -u
export MPLBACKEND=Agg
export GAMMACM_NPROC=${SLURM_CPUS_PER_TASK:-1}
LOGRS=(-5 -4 -2 -1 0 1 2 3)
export LOGR=${LOGRS[$SLURM_ARRAY_TASK_ID]}

STAGE_WORK="${SLURM_SUBMIT_DIR:-$PWD}"
. "$STAGE_WORK/hpc/stage.sh"
export STAGE_SHELL="${ZSH}"
stage_open "pt.$RUNKEY.z$ZSH.$SLURM_ARRAY_TASK_ID"
stage_analysis_in "$RUNKEY" || exit 2
stage_analysis_cd || exit 1

python3 -u -c "
import os, sweep_gammacm as S
S.run_sweep(os.environ[\"RUNKEY\"], [float(os.environ[\"LOGR\"])], z=int(os.environ[\"ZSH\"]),
            method=os.environ[\"METHOD\"], nproc=int(os.environ[\"GAMMACM_NPROC\"]))
print(\"point done:\", os.environ[\"RUNKEY\"], \"z =\", os.environ[\"ZSH\"], \"logr =\", os.environ[\"LOGR\"])
"
rc=$?
stage_analysis_out "$RUNKEY"
exit $rc
