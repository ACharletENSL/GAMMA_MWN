# -*- coding: utf-8 -*-
# @Author: acharlet

'''
Process-pool plumbing shared by every parallel driver in the project.

There are two levels of parallelism over the same work:
  - POINT level: one worker per sweep point (sweep_gammacm.run_sweep,
    sweep_efficiency.run_sweep). Capped at the number of points -- 8 for the
    gamma_c/gamma_m sweep -- so extra cores do nothing beyond that.
  - CELL level: one worker per chunk of emitting cells, INSIDE one point
    (working_cooling_data.get_shell_nuFnu_fromData, ncell_proc). Capped only by
    the cell count (~1e4 at hi-res), which is what makes an HPC core count
    usable.

They are alternatives, never nested: a pool whose workers each build their own
pool multiplies the forkserver cost and the peak memory by the outer width, and
is nearly impossible to debug when it stalls. The drivers pick one.

This module owns the start method, the worker count, the thread pinning and the
numba warm-up, so the two levels cannot drift apart on any of them.
'''

import os
import contextlib


def pool_context():
  '''
  Start method for the process pools: 'forkserver', explicitly, on every Python
  version. Both alternatives are broken here, in ways that cost hours because neither
  raises -- the pool just stops.

  NOT 'fork'. The parent is multi-threaded by the time the pool is built: numpy's BLAS
  spins up one thread per core on the serial warm-up point. A forked child gets a copy
  of the parent's memory but only the forking thread, so any mutex a BLAS thread held at
  fork time is locked forever in the child. Measured here: sweep 1 finished, sweep 2's
  four workers each sat in futex_wait_queue at 25 MB RSS -- a bare interpreter, blocked
  before it could load a single cell -- for six hours with no error. Python 3.14
  deprecated fork in multi-threaded processes for exactly this, and made forkserver the
  Linux default; do not "restore" fork to get the old behaviour back.

  NOT the platform default either, which is still fork on <=3.13 and would silently
  reintroduce the above wherever this runs.

  The forkserver cost is that workers do NOT inherit the in-process state the parent
  builds (_RAR_HEAD_MEM, _DATA_END_MEM, loaded cell fits, numba's JIT) and re-read
  it from disk instead. That is a slowdown, not a correctness problem -- every one of
  those caches is backed by a file, which is what the parent's serial prologue is really
  for: it populates them serially so parallel workers only ever READ those shared paths.

  Callers must be import-safe: a forkserver child re-imports the main module (as
  __mp_main__), so a driver SCRIPT needs its work behind `if __name__ == '__main__':`.
  Without it the driver's top level re-runs inside the forkserver, which then sits in a
  second sweep and never serves a worker -- the same silent stall, different cause.
  Driving via `python -c` has no main path and is unaffected.

  Falls back to the default context where forkserver is unavailable (Windows: spawn).
  '''
  import multiprocessing as mp
  try:
    return mp.get_context('forkserver')
  except ValueError:
    return mp.get_context()


def resolve_nproc(nproc=None, cap=None, env_var='GAMMACM_NPROC'):
  '''
  Resolve a worker count: explicit arg > env GAMMACM_NPROC > the number of cores this
  process may actually run on, minus one (leave one core free). Clamped to [1, cap];
  cap=None leaves it UNCAPPED.

  The cap is the caller's, not this function's: a point-level pool caps at the number of
  points, a cell-level pool at the number of chunks. Resolving the budget and clamping it
  used to be one step, which is precisely what limited the sweep to 8 workers -- the cap
  was baked in at the only place that knew the budget.

  NOT os.cpu_count(): that is the machine's core count and ignores the allocation. Under
  Slurm a job granted 8 of a node's 64 CPUs would otherwise start 63 workers on 8 cores.
  SLURM_CPUS_PER_TASK first, then the affinity mask (which the cgroup pins for us), then
  cpu_count as the last resort.
  '''
  if nproc is None:
    env_np = os.environ.get(env_var)
    if env_np:
      nproc = int(env_np)
    else:
      slurm_np = os.environ.get('SLURM_CPUS_PER_TASK')
      if slurm_np:
        navail = int(slurm_np)
      elif hasattr(os, 'sched_getaffinity'):
        navail = len(os.sched_getaffinity(0))
      else:
        navail = os.cpu_count() or 1
      nproc = max(1, navail - 1)
  nproc = max(1, int(nproc))
  return nproc if cap is None else max(1, min(nproc, int(cap)))


_THREAD_VARS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')

def set_thread_env(n=1):
  '''
  Pin every BLAS/threading backend to n threads in os.environ, so pool workers do not
  each spin up one thread per core. MUST be called BEFORE the pool is created: a
  forkserver child inherits the environment as it stood when the forkserver was started,
  and numpy reads these at import, i.e. once, in the child.

  The radiation kernels are numba + elementwise numpy, so there is no BLAS work to lose
  here -- the threads are pure oversubscription, and multi-threaded BLAS in the parent is
  already the documented cause of one six-hour stall (see pool_context).

  Only sets variables the user has not set: an explicit OMP_NUM_THREADS in the job script
  wins.
  '''
  for v in _THREAD_VARS:
    os.environ.setdefault(v, str(n))


def warm_numba():
  '''
  Compile the @njit(cache=True) radiation kernels once, in this process, so workers find
  a populated on-disk cache instead of all racing to compile the same functions.

  This is what the sweep's serial warm-up point used to provide for free. Cell-level
  parallelism has no serial first point to hide behind, and N fresh workers each
  compiling _outer_loglog_blend is both wasted time and N concurrent writers to one
  numba cache directory.

  Argument dtypes/layouts match the real call sites exactly (all C-contiguous float64
  1-D), so the cached specialization is the one the workers will look up.
  '''
  import numpy as np
  from phys_functions import func_R
  from radiation_cooling import _outer_loglog_blend

  func_R(np.array([1.0]))                        # -> _func_R_kernel
  n = 4
  _outer_loglog_blend(np.full(n, -800.),         # logL_grid  (radiation_cooling:382)
                      np.ascontiguousarray(np.linspace(0., n - 2., 3)),   # u
                      np.ascontiguousarray(np.linspace(0., 1., 2)),       # s
                      np.ones(2))                                          # w


@contextlib.contextmanager
def cell_executor(nproc, initializer=None, initargs=()):
  '''
  ProcessPoolExecutor on the forkserver context, with the threading environment pinned
  and the numba cache warmed before any worker starts.

  initializer/initargs ship the read-only per-point context (grids, env, variants) to
  each worker ONCE, so the per-task payload stays down to a chunk spec.
  '''
  import concurrent.futures as cf
  set_thread_env()
  warm_numba()
  with cf.ProcessPoolExecutor(max_workers=int(nproc), mp_context=pool_context(),
                              initializer=initializer, initargs=initargs) as ex:
    yield ex
