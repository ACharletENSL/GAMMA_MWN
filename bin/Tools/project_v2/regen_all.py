# Full downstream regeneration for the fiducial under the CORRECTED gamma_c definition.
# Every step is guarded: one failure must not cost the rest of the set.
# __main__ guard matters -- several of these build forkserver pools.
import os, time, traceback


def main():
  # REGEN_KEY selects the run (default the fiducial); REGEN_SKIP is a comma-separated list
  # of substrings, so a step whose name contains one is skipped. sweep_efficiency is the
  # one worth skipping on a hi-res key: it computes its OWN points at 10 per decade rather
  # than replotting the sweep's, which is cheap at 500 cells and is not at 10000.
  K = os.environ.get('REGEN_KEY', 'cooling_g100')
  NP = int(os.environ.get('REGEN_NPROC', '7'))
  skip = [t for t in os.environ.get('REGEN_SKIP', '').split(',') if t]
  print(f'regenerating for key={K!r}, nproc={NP}, skipping={skip}', flush=True)
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
  # COMPLETENESS GUARD. sweep_gammacm.main(use_cache=True) calls load_sweep first and only
  # falls through to run_sweep when that returns NOTHING -- so a cache holding ONE point
  # counts as "cached" and every figure below is drawn from that one point, silently. That
  # is exactly what happened to the fiducial z=4 set on 2026-09-08: a killed run left one
  # banked point, the relaunch replotted it, and the whole downstream set ran on it.
  import glob
  from sweep_gammacm import method_outdir, LOG10RATIO_ARR
  want = len(LOG10RATIO_ARR)
  bad = []
  for m in ('data_rarcut', 'data'):
    for zz in (4, 1):
      d = method_outdir(m, K, zz)
      n = len(glob.glob(d + '/cache/point_logr=*.npz'))
      if n != want:
        bad.append(f'{m} z={zz}: {n}/{want} points ({d})')
  if bad:
    print('INCOMPLETE SWEEP CACHES -- downstream figures would be drawn from a subset:',
          flush=True)
    for b in bad:
      print('   ' + b, flush=True)
    raise SystemExit('refusing to regenerate from an incomplete sweep; run '
                     'sweep_gammacm.main(..., use_cache=False) for those (method, z) first')
  print(f'all four sweep caches complete at {want} points', flush=True)

  results = []
  for name, fn in steps:
    if any(t in name for t in skip):
      print(f'\n########## {name}: SKIPPED (REGEN_SKIP) ##########', flush=True)
      results.append((name, 'skipped', 0.))
      continue
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
