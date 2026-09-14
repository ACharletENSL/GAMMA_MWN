#!/bin/bash
# Full recompute of the _fc2 sweeps, then the figure regeneration, for both runs.
# data+rarcut computes the reference and the modelled cut in one pass, filling both
# method caches.
#
# Same chain as the original submit_rerun.sh, pointed at the staged launchers in hpc/:
# every task now works off its own node's /tmp (hpc/README.md). Run it from the repo root.
set -e
M=${METHOD:-data+rarcut}
for KEY in ${KEYS:-cooling_g100 cooling_g100_hires}; do
  DEPS=""
  for Z in 4 1; do
    P=$(sbatch --parsable --export=ALL,RUNKEY=$KEY,ZSH=$Z,METHOD=$M hpc/sweep_prep.sh)
    A=$(sbatch --parsable --dependency=afterok:$P --export=ALL,RUNKEY=$KEY,ZSH=$Z,METHOD=$M \
               --array=0-7 hpc/sweep_point.sh)
    echo "$KEY z=$Z : prep $P -> array $A"
    DEPS="$DEPS:$A"
  done
  R=$(sbatch --parsable --dependency=afterok${DEPS} --export=ALL,REGEN_KEY=$KEY hpc/regen.sh)
  echo "$KEY regen : $R  (after${DEPS})"
done
