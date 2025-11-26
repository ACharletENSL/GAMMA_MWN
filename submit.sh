#!/bin/sh
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --job-name="big_sph"

srun mpirun -n 1  ./bin/GAMMA -w