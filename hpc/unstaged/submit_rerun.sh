#!/bin/bash
# Full recompute of the _fc2 sweeps under the leading-edge injection event (e743d8c),
# then the figure regeneration, for both runs. data+rarcut computes the reference and the
# modelled cut in one pass, filling both method caches.
set -e
M=data+rarcut
declare -A REGDEP
for KEY in cooling_g100 cooling_g100_hires; do
  DEPS=""
  for Z in 4 1; do
    P=$(sbatch --parsable --export=ALL,RUNKEY=$KEY,ZSH=$Z,METHOD=$M sweep_prep_k.sh)
    A=$(sbatch --parsable --dependency=afterok:$P --export=ALL,RUNKEY=$KEY,ZSH=$Z,METHOD=$M --array=0-7 sweep_point_k.sh)
    echo "$KEY z=$Z : prep $P -> array $A"
    DEPS="$DEPS:$A"
  done
  R=$(sbatch --parsable --dependency=afterok${DEPS} --export=ALL,REGEN_KEY=$KEY regen.sh)
  echo "$KEY regen : $R  (after${DEPS})"
done
