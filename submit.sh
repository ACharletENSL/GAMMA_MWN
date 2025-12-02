#!/bin/sh
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --job-name="param_sweep"

srun mpirun -n 1  ./bin/GAMMA -w