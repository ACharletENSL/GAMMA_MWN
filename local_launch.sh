#!/bin/sh
# Optional args are forwarded to setup.py, e.g.:
#   ./local_launch.sh --alpha 3 --zeta 5 [--src results/<key>/phys_input.ini]
# for a Granot (2012) hydro unit-rescaled run. No args -> fiducial run.

rm -rf results/Last
mkdir -p results/Last
python setup.py "$@"      # writes results/Last/phys_input.ini (rescaled inputs)
make -B
mpirun -n 1 ./bin/GAMMA -w
