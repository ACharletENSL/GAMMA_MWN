# Optional args are forwarded to setup.py, e.g.:
#   ./HPC_launch.sh --alpha 3 --zeta 5 [--src results/<key>/phys_input.ini]
# for a Granot (2012) hydro unit-rescaled run. No args -> fiducial run.
# Unlike local_launch.sh this does NOT wipe results/Last -- a long run may be living
# there. Move the previous run aside by hand before launching.
mkdir -p results/Last   # setup.py's closing mv lands here; it fails silently without it
python setup.py "$@"      # writes results/Last/phys_input.ini (rescaled inputs)
make clean && make -B     # clean is required: setup.py rewrites Shells.cpp/environment.h
                          # constants and the Makefile tracks no header dependencies
sbatch submit.sh
