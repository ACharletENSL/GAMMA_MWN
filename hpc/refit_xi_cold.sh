#!/bin/bash
#SBATCH --job-name="xi_refit"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --no-requeue
#SBATCH --output=slurm-%A_%a.out
#
# Refit the cold a_u sweep's hydro + xi fits with consistent_vx_u=True
# (fits_hydro.cellsBehindShock_fromFit), one array task per point, same index map as
# hpc/sweep_au_cold.sh:
#   sbatch --array=0-40%10 hpc/refit_xi_cold.sh
# Reads only results/$KEY/run_data_{1,4}.csv, never the snapshots. The previous fits are
# kept once as fit_{RS,FS}_betau.out (cp -n: an existing backup is never overwritten).
set -u
T=${SLURM_ARRAY_TASK_ID:?run as an array job}
LA=$(awk -v t="$T" 'BEGIN{printf "%.1f", (t-20)/10}')
THETA0=${THETA0:-5e-7}
KEY="sweep_Th${THETA0}_log_au=$LA"
D="${SLURM_SUBMIT_DIR:-$PWD}/results/$KEY"
[ -f "$D/run_data_4.csv" ] || { echo "no run_data in $D"; exit 2; }
cp -n "$D/fit_RS.out" "$D/fit_RS_betau.out"
cp -n "$D/fit_FS.out" "$D/fit_FS_betau.out"

cd "${SLURM_SUBMIT_DIR:-$PWD}/bin/Tools/project_v2" || exit 1
export PYTHONDONTWRITEBYTECODE=1 SHOCK_FINDER=pjump OMP_NUM_THREADS=1 MPLBACKEND=Agg
python3 -u -c "
from analysis_thinshell import extract_fits
extract_fits('$KEY', $LA, noPrint=True, consistent_vx_u=True)
print('refit done: $KEY')
"
