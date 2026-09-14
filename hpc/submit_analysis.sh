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

# --- stage in -------------------------------------------------------------------------
# The code travels with the job: it is 7 MB, it pins the analysis to one commit for the
# job's lifetime, and the cwd is what makes IO.py resolve to the staged tree at all.
# A filtered stage_in never falls back to a symlink on its own (it would mask what it
# had already copied), so the two filtered calls say what they want instead. The code
# must exist in the staged tree one way or the other -- an empty project_v2 is a job
# that cannot even import.
stage_in "bin/Tools/project_v2" --exclude='__pycache__' --exclude='results' \
  || stage_link "bin/Tools/project_v2"
stage_in "phys_input.ini"
stage_in "bin/Tools/figures"        # sweep point caches live here: without them a sweep
                                    # recomputes every point instead of reloading it
stage_in "extracted_data"

mkdir -p "$STAGE_DIR/results/$KEY"

# The run directory's small files -- phys_input.ini, field_correction.json, the fit and
# cellscan caches -- are read constantly and always worth having locally.
# phys_input.ini and field_correction.json decide what the physics IS: if they are
# missing from the staged run directory the analysis silently falls back to the repo
# root's .ini and reports a different run. Link them rather than lose them.
stage_in "results/$KEY" --exclude='cells/' --exclude='phys[0-9]*.out' \
  || find "$STAGE_WORK/results/$KEY" -maxdepth 1 -mindepth 1 \
          ! -name 'phys[0-9]*.out' ! -name cells \
          -exec ln -sf -t "$STAGE_DIR/results/$KEY/" {} +

# The snapshots are handled by hand rather than through stage_in, because the fallback
# has to apply to THEM and not to the directory that now holds the files just staged.
# 502 GiB at hi-res against ~200 GiB of /tmp, so the symlink branch is the usual one.
# Guarded on STAGE_ACTIVE: these are raw commands, not stage_* calls, so with staging
# off they would otherwise link every snapshot onto itself.
DUMP_SRC="$STAGE_WORK/results/$KEY"
if [ "$STAGE_ACTIVE" = 1 ]; then
  if [ "$STAGE_DUMPS" = "1" ] && \
     [ "$(stage_estimate_kb "$DUMP_SRC")" -lt "$(stage_free_kb)" ]; then
    echo "[stage] copying the snapshots in"
    rsync -a --include='phys[0-9]*.out' --exclude='*' "$DUMP_SRC/" "$STAGE_DIR/results/$KEY/"
  else
    [ "$STAGE_DUMPS" = "1" ] && echo "[stage] snapshots do not fit -- symlinking them"
    # one find with a batched exec, NOT a shell loop: there are 153 224 snapshots in a
    # hi-res run and a loop would fork ln that many times
    find "$DUMP_SRC" -maxdepth 1 -name 'phys[0-9]*.out' \
         -exec ln -sf -t "$STAGE_DIR/results/$KEY/" {} +
  fi
fi

case "$STAGE_CELLS" in
  in)   stage_in "results/$KEY/cells" ${CELLS_FILTER} ;;
  out)  mkdir -p "$STAGE_DIR/results/$KEY/cells" ;;
  link) stage_link "results/$KEY/cells" ;;
  *)    echo "unknown STAGE_CELLS='$STAGE_CELLS'" >&2; exit 2 ;;
esac

# --- drain (STAGE_CELLS=out) -----------------------------------------------------------
# An extraction writes more than /tmp holds, so finished cells go back as the job runs.
# Only files untouched for 2 minutes move, so a cell still being written is left alone.
DRAIN_PID=""
if [ "$STAGE_CELLS" = "out" ] && [ "$STAGE_ACTIVE" = 1 ]; then
  ( trap - EXIT        # never let the drainer's exit run stage_close on the live tree
    while true; do
      sleep "$DRAIN_SECS"
      stage_drain "results/$KEY/cells" "$DRAIN_QUIET_MIN"
    done ) &
  DRAIN_PID=$!
  echo "[stage] draining results/$KEY/cells every ${DRAIN_SECS}s (pid $DRAIN_PID)"
fi

# --- run -------------------------------------------------------------------------------
# Name the staged tree explicitly. Without this the analysis stack infers its root from
# the cwd by the first path element containing 'GAMMA', which is right here but is not
# something to rely on for a job that writes 300 GiB into it.
export GAMMA_DIR="$STAGE_DIR"
cd "$STAGE_DIR/bin/Tools/project_v2" || exit 1

# Verify it before spending a day on it: if the stack resolves anywhere but the staged
# tree, drop back to ~/work rather than analysing the wrong directory. Slow is a nuisance,
# wrong is a lost job.
ROOT_SEEN="$(python -c 'import IO; print(IO.GAMMA_dir)' 2>/dev/null)"
if [ "$STAGE_ACTIVE" = 1 ] && [ "$ROOT_SEEN" != "$STAGE_DIR" ]; then
  echo "[stage] ERROR: the analysis stack resolves to '$ROOT_SEEN', not '$STAGE_DIR'." >&2
  echo "[stage]        falling back to running in $STAGE_WORK (no staging)." >&2
  stage_close
  STAGE_ACTIVE=0
  STAGE_DIR="$STAGE_WORK"
  export GAMMA_DIR="$STAGE_WORK"
  cd "$STAGE_WORK/bin/Tools/project_v2" || exit 1
fi

echo "[stage] cwd=$PWD"
echo "[stage] root=$GAMMA_DIR"
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

# --- stage out -------------------------------------------------------------------------
# Unconditionally, even on failure: a job that died after twenty hours of extraction
# still wrote cells worth keeping. No --delete anywhere -- the work copy holds results
# this job never produced.
cd "$STAGE_WORK" || exit 1
stage_out "results/$KEY" --exclude='phys[0-9]*.out' --no-links
stage_out "bin/Tools/figures"
stage_out "extracted_data"
echo "=== staged out at $(date '+%F %T')"

exit "$rc"
