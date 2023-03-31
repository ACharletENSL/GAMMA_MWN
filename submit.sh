#!/bin/sh
#SBATCH --nodes=1
##SBATCH --nodelist=cn04
#SBATCH --job-name="test"
#SBATCH --output=test.out
#SBATCH --mail-user arthur.charlet@ens-lyon.fr
#SBATCH --mail-type BEGIN
#SBATCH --mail-type END
#SBATCH --mail-type FAIL

srun mpirun -n 1 ./bin/GAMMA -w

## load spack.io
#source /prefix/tool/spack/share/spack/setup-env.sh

## load modules
#spack load mpich gsl cmake

## not required !!! after modules are loaded
#CPATH=/prefix/el8/software/gcc-8.5.0/gsl-2.7.1-g2z66xwa/include
#export LD_LIBRARY_PATH="/prefix/el8/software/gcc-8.5.0/gsl-2.7.1-g2z66xwa/lib"

# add -c np to set number of CPUs
#srun --mpi=pmix -n 1 ./bin/GAMMA -w
