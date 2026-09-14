#!/bin/bash
# Full recompute of the _fc2 sweeps, then the figure regeneration, for both runs.
# data+rarcut computes the reference and the modelled cut in one pass, filling both
# method caches. Run it from the repo root.
#
# DEFAULT: one job per (run, shell), all nine points on one staged copy of the cells.
# That is one pass over the cells per shell instead of one per point -- the file server
# is limited by the NUMBER of operations, and a point opens every cell of the shell once.
#
# POINTS_PER_TASK=n splits each shell into ceil(9/n) array tasks for wall clock, at the
# price of one cell pass per task. In that mode the prologue (cellscan, rarefaction head)
# has to be built before the tasks race for it, so sweep_prep.sh goes first and the array
# covers the other eight points.
set -e
M=${METHOD:-data+rarcut}
PER=${POINTS_PER_TASK:-0}

for KEY in ${KEYS:-cooling_g100 cooling_g100_hires}; do
  DEPS=""
  for Z in 4 1; do
    if [ "$PER" -eq 0 ]; then
      J=$(sbatch --parsable --export=ALL,RUNKEY=$KEY,ZSH=$Z,METHOD=$M hpc/sweep_point.sh)
      echo "$KEY z=$Z : all nine points in job $J"
    else
      P=$(sbatch --parsable --export=ALL,RUNKEY=$KEY,ZSH=$Z,METHOD=$M hpc/sweep_prep.sh)
      NTASK=$(( (8 + PER - 1) / PER ))
      J=$(sbatch --parsable --dependency=afterok:$P --array=0-$((NTASK - 1)) \
                 --export=ALL,RUNKEY=$KEY,ZSH=$Z,METHOD=$M,POINTS_PER_TASK=$PER,"LOGR_LIST=-5 -4 -2 -1 0 1 2 3" \
                 hpc/sweep_point.sh)
      echo "$KEY z=$Z : prep $P -> array $J ($NTASK tasks of $PER points)"
    fi
    DEPS="$DEPS:$J"
  done
  R=$(sbatch --parsable --dependency=afterok${DEPS} --export=ALL,REGEN_KEY=$KEY hpc/regen.sh)
  echo "$KEY regen : $R  (after${DEPS})"
done
