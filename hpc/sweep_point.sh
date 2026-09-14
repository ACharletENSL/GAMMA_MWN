#!/bin/bash
#SBATCH --job-name="swp_pt"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=150G
#SBATCH --no-requeue
#SBATCH --output=slurm-%A_%a.out
#
# THE SWEEP POINTS OF ONE (RUNKEY, ZSH), STAGED ONCE AND COMPUTED TOGETHER.
#
# What costs the file server is the NUMBER of operations, and a sweep point opens every
# cell of the shell exactly once. So the old eight-task array cost eight full passes over
# the cells -- 8 x 20 000 opens -- and STAGING THAT ARRAY WOULD NOT HAVE HELPED: copying
# a shell in is itself one pass, so a task that reads it once only moves the opens from
# the analysis to rsync. Staging only pays when the copy is read more than once.
#
# Hence the default here: ONE task, all nine points, one staged copy read nine times --
# 1/9 of the passes, and not 9x the wall clock either, since the cold cost of a point is
# dominated by the ~2.3 s cell open and only the first pass pays it. Split it with
# POINTS_PER_TASK (and a matching --array) when the clock matters more than the server
# does: each task is then its own pass, so the cost is one pass per TASK, not per point.
set -u
export MPLBACKEND=Agg
export GAMMACM_NPROC=${SLURM_CPUS_PER_TASK:-1}

LOGRS=(${LOGR_LIST:--5 -4 -3 -2 -1 0 1 2 3})
PER=${POINTS_PER_TASK:-${#LOGRS[@]}}
T=${SLURM_ARRAY_TASK_ID:-0}
CHUNK=("${LOGRS[@]:$((T * PER)):PER}")
if [ ${#CHUNK[@]} -eq 0 ]; then
  echo "task $T: no points in this chunk, nothing to do"
  exit 0
fi
export LOGR_CHUNK="${CHUNK[*]}"

STAGE_WORK="${SLURM_SUBMIT_DIR:-$PWD}"
. "$STAGE_WORK/hpc/stage.sh"
export STAGE_SHELL="${ZSH}"          # one shell's cells: 165 GiB fits, both do not
stage_open "pt.$RUNKEY.z$ZSH.$T"
stage_analysis_in "$RUNKEY" || exit 2
stage_analysis_cd || exit 1

echo "=== $RUNKEY z=$ZSH method=$METHOD points: $LOGR_CHUNK"
python3 -u -c "
import os, sweep_gammacm as S
logrs = [float(x) for x in os.environ['LOGR_CHUNK'].split()]
S.run_sweep(os.environ['RUNKEY'], logrs, z=int(os.environ['ZSH']),
            method=os.environ['METHOD'], nproc=int(os.environ['GAMMACM_NPROC']))
print('points done:', os.environ['RUNKEY'], 'z =', os.environ['ZSH'], logrs)
"
rc=$?
stage_analysis_out "$RUNKEY"
exit $rc
