# Full downstream regeneration for the fiducial under the CORRECTED gamma_c definition.
# Every step is guarded: one failure must not cost the rest of the set.
# __main__ guard matters -- several of these build forkserver pools.
import time, traceback


def main():
  K, NP = 'cooling_g100', 7
  import sweep_gammacm as swp
  import lightcurve_shape as lcs
  import sweep_shells, sweep_rarcut, sweep_compare
  import nuc_validation, slope_validation, segment_route, sweep_efficiency

  steps = [
    ('sweep_gammacm  data_rarcut z=4', lambda: swp.main(key=K, method='data_rarcut', z=4, nproc=NP)),
    ('sweep_gammacm  data_rarcut z=1', lambda: swp.main(key=K, method='data_rarcut', z=1, nproc=NP)),
    ('sweep_gammacm  data        z=4', lambda: swp.main(key=K, method='data', z=4, nproc=NP)),
    ('sweep_gammacm  data        z=1', lambda: swp.main(key=K, method='data', z=1, nproc=NP)),
    ('lightcurve_shape z=4',           lambda: lcs.main(key=K, z=4, nproc=NP)),
    ('lightcurve_shape z=1',           lambda: lcs.main(key=K, z=1, nproc=NP)),
    ('sweep_shells',                   lambda: sweep_shells.main(key=K, nproc=NP)),
    ('sweep_rarcut',                   lambda: sweep_rarcut.main(key=K, nproc=NP)),
    ('sweep_compare',                  lambda: sweep_compare.main(key=K, nproc=NP)),
    ('nuc_validation',                 lambda: nuc_validation.main(key=K)),
    ('slope_validation',               lambda: slope_validation.main(key=K)),
    ('segment_route',                  lambda: segment_route.main(key=K)),
    # last: it computes its OWN points, including data_norar_prerar, and is the long pole
    ('sweep_efficiency',               lambda: sweep_efficiency.main(key=K, nproc=NP)),
  ]
  results = []
  for name, fn in steps:
    t0 = time.time()
    print(f'\n########## {name} ##########', flush=True)
    try:
      fn()
      results.append((name, 'OK', time.time()-t0))
    except Exception as e:
      print(f'!!! {name} FAILED: {type(e).__name__}: {e}', flush=True)
      traceback.print_exc()
      results.append((name, f'FAILED {type(e).__name__}', time.time()-t0))
    print(f'########## {name}: {results[-1][1]} in {results[-1][2]/60:.1f} min', flush=True)
  print('\n===== REGENERATION SUMMARY =====', flush=True)
  for name, st, dt in results:
    print(f'  {name:34s} {st:24s} {dt/60:7.1f} min', flush=True)


if __name__ == '__main__':
  main()
