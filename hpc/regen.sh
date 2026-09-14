#!/bin/bash
#SBATCH --job-name="regen"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=150G
#SBATCH --no-requeue
#SBATCH --output=slurm-%j.out
# Full article-figure regeneration for one REGEN_KEY, on the CORRECTED (_fc2) sweeps.
#
# Staged version of regen.sh. No STAGE_SHELL here: regen_all runs sweep_shells, which
# needs BOTH shells, so this stages all the cells there are. The fiducial's are 790 MB and
# fit easily; the hi-res run's are 331 GiB and do not, so there they are symlinked and read
# over NFS exactly as before -- staging cannot help a job that needs every cell at once.
set -u
export MPLBACKEND=Agg
export GAMMACM_NPROC=${SLURM_CPUS_PER_TASK:-1}
export REGEN_NPROC=${SLURM_CPUS_PER_TASK:-1}
# nuc_validation is PARKED (see the banner in nuc_validation.py): the estimator is out
# of the paper, and its harvest is hours at hi-res AND inconsistent with the spectra it
# validates against. This default must stay in step with regen_all.py's, since setting
# REGEN_SKIP here overrides it entirely.
export REGEN_SKIP="${REGEN_SKIP:-sweep_efficiency,segment_route,nuc_validation}"
export REGEN_KEY="${REGEN_KEY:?set REGEN_KEY}"

STAGE_WORK="${SLURM_SUBMIT_DIR:-$PWD}"
. "$STAGE_WORK/hpc/stage.sh"
stage_open "regen.$REGEN_KEY"
stage_analysis_in "$REGEN_KEY" || exit 2
stage_analysis_cd || exit 1

echo "=== regenerating key=$REGEN_KEY on ${SLURM_CPUS_PER_TASK} cpus ==="
python3 -u regen_all.py
rc=$?
stage_analysis_out "$REGEN_KEY"
exit $rc
