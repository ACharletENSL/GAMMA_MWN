#!/bin/bash
#SBATCH --job-name="deepband"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=150G
#SBATCH --no-requeue
#SBATCH --output=slurm-%j.out
# CALIBRATION SET: the sweep recomputed with the tnu<1 cut LIFTED and the band taken well
# below nu_B, so FC*/VFC spectra acquire the nu^(4/3) window that breaks_from_identified
# needs to re-centre their mid slope. Measuring the same spectrum on the extended band and
# on the production sub-range gives the estimator offset from the REAL spectra instead of
# from GS02 synthetics -- see the memory note below-nub-calibration.
#
# THIS IS NOT A PRODUCTION SWEEP AND MUST NEVER OVERWRITE ONE:
#   - SYN_NO_LOWCUT=1 changes every spectrum and every energy budget
#     (step_radiated_energy's analytic frequency integral assumes the cut)
#   - so it writes to its OWN directory, passed as outdir, and touches nothing else.
# ZSH selects the shell; one job per shell, points run SERIALLY so the first builds the
# shared prologue (cellscan, rarefaction head) and the rest reuse it -- the reason the
# production driver splits prep from array does not apply when nothing runs concurrently.
export MPLBACKEND=Agg
export GAMMACM_NPROC=${SLURM_CPUS_PER_TASK:-1}
export SYN_NO_LOWCUT=1
export ZSH="${ZSH:?set ZSH}"
export DEEP_LOGNU="${DEEP_LOGNU:--10.0}"
# KEY selects the RUN. The table is measured on one run's spectra and is not transferable:
# the fiducial and hi-res raw FC*/VFC slopes differ by 0.005-0.015 at the fast-cooling end,
# which is the size of the offset itself. Every run whose a_mid figure is drawn corrected
# needs its own pass here.
export KEY="${KEY:-}"
cd "$SLURM_SUBMIT_DIR/bin/Tools/project_v2"
echo "=== deep-band calibration: key=${KEY:-<default>}, z=$ZSH, lognu_min=$DEEP_LOGNU, SYN_NO_LOWCUT=$SYN_NO_LOWCUT ==="
python3 -u -c "
import os, numpy as np, sweep_gammacm as S
import radiation_cooling as rc
assert not rc.SYN_LOWCUT, 'SYN_NO_LOWCUT did not reach the process'
z = int(os.environ['ZSH'])
key = os.environ.get('KEY') or S.DEFAULT_KEY
out = S.method_outdir(S.DEFAULT_METHOD, key, z) + '_deepband'
print('key', key, '-> writing to', out, flush=True)
S.run_sweep(key, list(S.LOG10RATIO_ARR), z=z,
            lognu_min=float(os.environ['DEEP_LOGNU']), outdir=out,
            nproc=int(os.environ['GAMMACM_NPROC']), skip_cached=False)
print('deep-band sweep done: key =', key, 'z =', z)
"
# The table needs BOTH shells, so only the job that finds the other one already on disk
# builds it -- otherwise the first to finish writes a half table and the two race for the
# same csv. It is measured from the caches, so re-running it costs nothing but reading.
python3 -u -c "
import os, sweep_gammacm as S, band_offset as B
key = os.environ.get('KEY') or S.DEFAULT_KEY
have = [os.path.isdir(S.method_outdir(S.DEFAULT_METHOD, key, z) + '_deepband') for z in (4, 1)]
if all(have):
    B.main(step=3, key=key)
else:
    print('other shell not on disk yet -- run band_offset.main(key=...) when it is')
"
