#!/bin/bash
#SBATCH --job-name="au_cold"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --no-requeue
#SBATCH --output=slurm-%A_%a.out
#
# THE COLD a_u SWEEP: one array task per point, log10(a_u - 1) = (ID - 20)/10, so
#   sbatch --array=0-40%10 hpc/sweep_au_cold.sh
# covers -2.0 .. 2.0 in steps of 0.1. Each point is written to results/$KEY with
# KEY=sweep_Th5e-7_log_au=<x>; the original sweep_log_au=* runs are never touched, and
# a task refuses to overwrite an existing key.
#
# Inputs: hpc/phys_input_sweep_au_cold.ini, i.e. the original sweep with Theta0 = 5e-7
# and a lab-time stop (see the template for why). The stop time is
#   tstop = 1.5 x max(1.55 tRS, 1.25 tFS)
# with tRS, tFS the analytic crossing times: the original sweep crossed at 1.2-1.5 tRS
# and 1.0-1.2 tFS, so this ends ~1.5x past the crossing, keeping the start of the
# rarefaction phase. The snapshots are KEPT: the sweep is meant for reuse.
#
# Everything runs on the node's /tmp and is copied back once at the end (hpc/README.md):
# each run writes a snapshot every fraction of a second, which 10 at a time on ~/work is
# the load the file server went down under. GAMMA writes into the tree its binary sits
# in, so each task builds a private copy (src, Makefile, setup.py; bin/Tools and
# extracted_data are symlinked, read-only). The fits then run on the local copy with
# SHOCK_FINDER=pjump, since GAMMA's Sd flag does not fire at low a_u.
set -u
WORK="${SLURM_SUBMIT_DIR:-$PWD}"
T=${SLURM_ARRAY_TASK_ID:?run as an array job}
LA=$(awk -v t="$T" 'BEGIN{printf "%.1f", (t-20)/10}')
# THETA0 overrides the template's 5e-7 (e.g. 5e-5 for warm controls); it names the key
THETA0=${THETA0:-5e-7}
KEY="sweep_Th${THETA0}_log_au=$LA"
DEST="$WORK/results/$KEY"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OMP_PROC_BIND=close OMP_PLACES=cores
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg

if [ -e "$DEST" ]; then
  echo "$DEST exists -- refusing to overwrite it; move it aside to rerun this point"
  exit 3
fi

# lower-case parent: IO.py finds the root as the first path element containing 'GAMMA'
LOC="${TMPDIR:-/tmp}/stage.${USER:-$(id -un)}.${SLURM_ARRAY_JOB_ID:-$$}.$T/GAMMA_MWN"
cleanup(){ case "$LOC" in */stage.*/GAMMA_MWN) rm -rf "$(dirname "$LOC")" ;; esac; }
trap cleanup EXIT
mkdir -p "$LOC/bin" "$LOC/results/Last" || exit 1
cp -r "$WORK/src" "$WORK/Makefile" "$WORK/setup.py" "$LOC/" || exit 1
ln -s "$WORK/bin/Tools" "$LOC/bin/Tools"
ln -s "$WORK/extracted_data" "$LOC/extracted_data"
cd "$LOC" || exit 1
export GAMMA_DIR="$LOC"

U4=$(awk -v la="$LA" 'BEGIN{printf "%.10g", 10*(1+10^la)}')
sed -e "s/@U4@/$U4/" -e "s/@TSTOP@/0/" -e "s/^Theta0 .*/Theta0      $THETA0/" "$WORK/hpc/phys_input_sweep_au_cold.ini" > phys_input.ini
TSTOP=$(python3 -c "
import sys; sys.path.insert(1, 'bin/Tools/project_v2')
from environment import MyEnv
e = MyEnv('./phys_input.ini')
print(f'{1.5*max(1.55*e.tRS, 1.25*e.tFS):.6g}')
" 2>/dev/null | tail -1)
[ -n "$TSTOP" ] || { echo "could not compute tstop"; exit 1; }
sed -i "s/^tstop .*/tstop       $TSTOP/" phys_input.ini
echo "=== $KEY: a_u = 1 + 10^$LA, u4 = $U4, tstop = $TSTOP, $OMP_NUM_THREADS threads on $(hostname -s)"

python3 setup.py --src ./phys_input.ini || exit 1
make clean > /dev/null && make -B > make.log 2>&1 || { tail -20 make.log; exit 1; }
mpirun -n 1 --bind-to none ./bin/GAMMA -w > gamma.log 2>&1
rc=$?
grep -E "Stopping|rror|nan" gamma.log | tail -5
mv results/Last "results/$KEY" && cp gamma.log "results/$KEY/"

if [ $rc -eq 0 ]; then
  # from $LOC, not from inside the symlinked bin/Tools: the cwd would resolve into ~/work
  SHOCK_FINDER=pjump OMP_NUM_THREADS=1 python3 -u -c "
import sys; sys.path.insert(1, 'bin/Tools/project_v2')
from analysis_thinshell import extract_fittingData
extract_fittingData('$KEY', nSH=3)
" > "results/$KEY/fits.log" 2>&1 || echo "fits FAILED, see results/$KEY/fits.log"
fi

# the one pass over ~/work: everything, on failure too
rsync -a "results/$KEY/" "$DEST/" || { echo "copy back FAILED"; trap - EXIT; echo "kept $LOC"; exit 4; }
echo "=== $KEY done (GAMMA rc=$rc), $(ls "$DEST" | wc -l) files in $DEST"
exit $rc
