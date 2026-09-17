#!/bin/bash
#SBATCH --job-name="bandoff"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --output=slurm-%j.out
# The offset table alone, measured from deep-band sweeps ALREADY on disk -- no emission is
# recomputed. Split out from calib_deepband.sh because the two have nothing to do with each
# other in cost: the sweep is hours on 56 cores, this is a serial pass over cached spectra
# whose cost does not depend on the run's resolution (it reads nuFnu arrays, not cells).
#
# Use it when the sweep was run one shell at a time (which is how hi-res must be run), or
# after ANY change to the bias grid -- though for that, band_offset.reindex() is cheaper
# still: it re-reads only the two columns that are grid lookups.
export MPLBACKEND=Agg
export KEY="${KEY:-}"
cd "$SLURM_SUBMIT_DIR/bin/Tools/project_v2"
python3 -u -c "
import os, band_offset as B
key = os.environ.get('KEY') or None
print('building the offset table for', key or '<default>', flush=True)
B.main(step=3, key=key)
"
