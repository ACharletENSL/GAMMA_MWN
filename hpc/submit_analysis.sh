#!/bin/bash
#SBATCH --job-name="gamma-post"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --no-requeue
#SBATCH --output=slurm-%j.out
#
# Post-processing on a compute node, run out of the node's LOCAL disk as the site asks
# (see hpc/stage.sh for the policy and the measured numbers). Everything the job reads
# or writes is mirrored into /tmp first; what will not fit is symlinked back to ~/work
# and read over NFS, so a job degrades in speed and never in correctness.
#
#   sbatch [slurm opts] hpc/submit_analysis.sh '<shell command, run in project_v2>'
#
# The command runs with its working directory inside the STAGED tree, which is what
# redirects the analysis: IO.py derives the project root from the cwd, so every
# results/, figures/ and extracted_data/ path resolves to the local copy.
#
# Environment (sbatch exports the submitting shell's environment by default):
#   KEY=<run>        the run to stage. Required.
#   STAGE_CELLS=in   what to do with results/$KEY/cells:
#                      in   copy it in     (sweeps, anything that READS cell histories)
#                      out  start empty    (extraction: written locally, drained back)
#                      link read over NFS  (no local copy)
#   STAGE_DUMPS=0    1 copies the phys*.out snapshots in as well. 502 GiB at hi-res, so
#                    it is off by default; stage_in falls back to a symlink if it does
#                    not fit, which is what happens on any big run.
#   CELLS_FILTER     extra rsync filters for the cells copy. The hi-res both-shell set is
#                    331 GiB and does not fit, one shell (165 GiB) does:
#                      CELLS_FILTER="--include=1[0-9][0-9][0-9][0-9].* \
#                                    --include=20[0-3][0-9][0-9].* --exclude=*"
#   DRAIN_SECS=600   how often to push freshly written cells back (STAGE_CELLS=out only)
#   GAMMA_STAGE=0    bypass all of this and run straight out of ~/work
#
# Examples
#   cell extraction (writes 331 GiB, drained back as it goes):
#     KEY=cooling_g100_hires STAGE_CELLS=out STAGE_DUMPS=1 \
#     sbatch --cpus-per-task=32 --mem=180G hpc/submit_analysis.sh \
#       'python -c "import analysis_hydro as a; a.extract_data_cells(\"$KEY\", None, nproc=32, cell_block=5000)"'
#
#   a sweep (reads every cell once per point -- the case staging is FOR):
#     KEY=cooling_g100_hires STAGE_CELLS=in \
#     sbatch --cpus-per-task=32 --mem=120G hpc/submit_analysis.sh \
#       'python sweep_gammacm.py'

set -u

CMD="${1:-}"
if [ -z "$CMD" ]; then
  echo "usage: sbatch hpc/submit_analysis.sh '<command>'   (KEY=<run> must be set)" >&2
  exit 2
fi
KEY="${KEY:-}"
if [ -z "$KEY" ]; then
  echo "KEY is not set: which run should be staged?" >&2
  exit 2
fi

STAGE_WORK="${SLURM_SUBMIT_DIR:-$PWD}"
. "$STAGE_WORK/hpc/stage.sh"

STAGE_CELLS="${STAGE_CELLS:-in}"
STAGE_DUMPS="${STAGE_DUMPS:-0}"
CELLS_FILTER="${CELLS_FILTER:-}"
DRAIN_SECS="${DRAIN_SECS:-600}"
DRAIN_QUIET_MIN="${DRAIN_QUIET_MIN:-2}"   # a file must be this many minutes untouched to move

# The analysis stack is threaded through numpy/BLAS as well as through its own pools;
# without this a "1 worker" run quietly takes the whole node.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

echo "=== $(date '+%F %T')  job ${SLURM_JOB_ID:-none} on $(hostname -s) ==="
echo "key=$KEY  cells=$STAGE_CELLS  dumps=$STAGE_DUMPS  cpus=${SLURM_CPUS_PER_TASK:-?}"

stage_open "post.$KEY"
stage_analysis_in "$KEY" || exit 2

# --- drain (STAGE_CELLS=out) -----------------------------------------------------------
# An extraction writes more than /tmp holds, so finished cells go back as the job runs.
# Only files untouched for DRAIN_QUIET_MIN minutes move, so a cell still being written is
# left alone. The subshell clears the EXIT trap: the drainer must never run stage_close.
DRAIN_PID=""
if [ "$STAGE_CELLS" = "out" ] && [ "$STAGE_ACTIVE" = 1 ]; then
  ( trap - EXIT
    while true; do
      sleep "$DRAIN_SECS"
      stage_drain "results/$KEY/cells" "$DRAIN_QUIET_MIN"
    done ) &
  DRAIN_PID=$!
  echo "[stage] draining results/$KEY/cells every ${DRAIN_SECS}s (pid $DRAIN_PID)"
fi

# --- run -------------------------------------------------------------------------------
stage_analysis_cd || exit 1
echo "[stage] python=$(command -v python || echo MISSING)"
echo "=== command: $CMD"
eval "$CMD"
rc=$?
echo "=== command exited $rc at $(date '+%F %T')"

if [ -n "$DRAIN_PID" ]; then
  # children first: a drain rsync outliving its parent would race the final sync below,
  # and it is the one holding --remove-source-files
  pkill -P "$DRAIN_PID" 2>/dev/null
  kill "$DRAIN_PID" 2>/dev/null
  wait "$DRAIN_PID" 2>/dev/null
fi

stage_analysis_out "$KEY"
echo "=== staged out at $(date '+%F %T')"

exit "$rc"
