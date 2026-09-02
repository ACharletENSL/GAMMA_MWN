#!/bin/bash
#SBATCH --job-name="gamma1d"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --no-requeue
#SBATCH --output=slurm-%j.out

# The 1D build is one MPI rank (checkEnvironment throws if worldsize != 1) with OpenMP
# across the node's cores, so cpus-per-task is the only parallelism knob that does
# anything. Measured on a gold6130 node at 20800 cells (2000 iterations, every thread
# count bit-identical):
#   nt      1     2     4     8    16    32    64
#   speedup 1.00  1.96  3.82  7.31 13.3  21.2  30.7
# 32 is the physical core count; the SMT siblings still buy a real 1.45x on top, so take
# all 64. Drop to 32 if the node needs to be shared.
# Note: this cluster refuses --exclusive and --mem=0 ("please request specific
# resources"), so memory is asked for explicitly. The run itself is small (~20800 cells).
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# NOT 'srun mpirun': that nests a second MPI launcher inside Slurm's. And not a bare
# 'srun' either -- this OpenMPI (4.1.5, PMIx 4.2.4) cannot be direct-launched by this
# Slurm, whose only pmix plugin is v3: MPI_Init aborts with "OPAL ERROR: Unreachable ...
# ext3x_client.c". So mpirun, with --bind-to none, because OpenMPI binds a 1-rank job to
# a single core by default -- which would pin every OpenMP thread to that one core and
# silently flatten the scaling.
mpirun -n 1 --bind-to none ./bin/GAMMA -w
