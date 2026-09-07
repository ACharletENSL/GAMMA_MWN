'''
Extract hydro data from simulation, analysis
Contains:
  - get_critical_radii
  - get_critical_times
  - get_plaws_hydro
  - get_Rcrossing
  - extract_data_thinshell
  - extract_data_cell
  - smooth_dump_density (de-jitter the moving-mesh partition, see below)
'''

import json
import numpy as np
from scipy.ndimage import median_filter

from IO import *
from phys_constants import *
from phys_functions import prim2cons
from fits_hydro import get_hydrofits_shell
import cell_pool


##### Moving-mesh density de-jittering
# Long runs develop a cell-to-cell jitter in rho that reaches ~50% by the end, while p
# stays smooth (~6%). It is a MESH-PARTITION artifact, not a density fluctuation: the
# cell mass rho*x^2*dx stays smooth to 1e-4 throughout and corr(dln rho, dln dx) = -1.000
# exactly at every late snapshot. The moving mesh has partitioned a smooth mass
# distribution into unevenly wide cells and rho = mass/V inherits that inversely.
#
# Most of the pipeline is immune, because it consumes rho only through rho*V' = mass
# (cell_radiated_energy's Pmax*V3p, cell_injected_energy) or through p (syn/t_c1, since
# e' = p/(gma_ad-1)). What is NOT immune is anything reading rho on its own: the adiabatic
# factor (rho_j+1/rho_j)^(1/3) in evolve_gma_bounds_edges (+-40% in rho -> +-12% in gamma)
# and the total-variation budget of _refine_tt_on_rho_data (13x wasted sub-steps).
#
# CRITICAL: rho must never be smoothed alone. rho and dx being anti-correlated to -1.000,
# smoothing rho by itself would break rho*dx = mass -- the very thing that makes the flux
# and the energy budget exact. dx is therefore rescaled by rho_raw/rho_smooth below, which
# preserves the product to machine epsilon (measured 2.2e-16) and, since dx_new =
# mass/rho_smooth, comes out smooth as well.
RHO_SMOOTH_WINDOW = 9      # cells in the running median; None disables the filter
RHO_SMOOTH_ROUGH  = 0.05   # gate: median |dln rho| between neighbours, per tracer group


def smooth_dump_density(df, window=RHO_SMOOTH_WINDOW, rough_min=RHO_SMOOTH_ROUGH):
  '''
  De-jitter the moving-mesh partition of ONE snapshot, in place-safe fashion (returns a
  copy when it acts). Running median of ln(rho) over `window` cells, with dx rescaled so
  the cell mass rho*dx is preserved exactly (see module notes).

  Filtered separately per tracer group (external medium / shell 4 / shell 1), ordered in
  radius, so the contact discontinuity and the shell/ambient boundaries are never averaged
  across; each group is gated independently.

  GATE. A median filter preserves sharp steps but erodes ramps of order the window width,
  so it must stay off while the shell still has real radial structure. Roughness is a poor
  predictor of that erosion -- measured on cooling_g100, at it=1e5 the roughness is only
  0.0035 yet filtering cuts the sharpest gradient by 25% (a ramp), while at it=3e5-4.5e5
  the roughness is higher and the gradient larger but the loss is exactly 0% (a step). So
  the gate's only job is to keep the filter out of the structured phase entirely. That
  phase peaks at roughness 0.0169 (it=4.5e5); the jitter knee, where the loss jumps
  11.5% -> 64.7%, sits at 0.05-0.06. rough_min = 0.05 therefore clears the structure by 3x
  and lands on the knee. Firing late costs <=5% in rho (<=1.6% in gamma); firing early
  costs a 25% erosion of a real feature, so the asymmetry is deliberate.

  Window 9 is the smallest that cleans fully: once jitter dominates the loss is
  window-independent, while on a real ramp it grows monotonically with width (17.3% at
  w=5 to 27.5% at w=25). w=9 leaves residual roughness 0.0054 at it=8.5e5 against 0.0053
  for w=15 and 0.0311 for w=5.
  '''
  if window is None or 'rho' not in df or 'dx' not in df:
    return df

  rho = df['rho'].to_numpy(dtype=float)
  trac = df['trac'].to_numpy(dtype=float) if 'trac' in df else np.zeros(len(df))
  x = df['x'].to_numpy(dtype=float)

  rho_new = rho.copy()
  acted = False
  # Group by tracer value: 1 = shell 4, 2 = shell 1. The external medium (trac = 0) is
  # deliberately EXCLUDED -- it is a numerical buffer that extract_data_cells never puts
  # in a cell history (it selects trac > 0), and it is the one group whose roughness
  # exceeds the gate while the shells are still structured (0.0665 at shock crossing,
  # against maxima of 0.0043 for shell 4 and 0.0059 for shell 1 over the same span), so
  # filtering it would smear real structure for no benefit.
  for tval in np.unique(np.round(trac[trac > 0.5])):
    g = np.flatnonzero(np.round(trac) == tval)
    if g.size <= window:
      continue
    g = g[np.argsort(x[g])]                     # radial order within the group
    lr = np.log(rho[g])
    if np.median(np.abs(np.diff(lr))) <= rough_min:
      continue                                  # still structured (or already smooth)
    rho_new[g] = np.exp(median_filter(lr, size=window, mode='nearest'))
    acted = True

  if not acted:
    return df
  out = df.copy()
  out['rho'] = rho_new
  # rho*dx (the cell mass, x unchanged) preserved exactly -- see module notes
  out['dx'] = df['dx'].to_numpy(dtype=float) * (rho / rho_new)
  # The conserved DENSITIES must follow the new rho, or they silently contradict the new
  # dx: D, sx and tau are per unit volume, so leaving them stale while dx moves breaks
  # sum(sx*V) by up to 24% even though the primitives themselves conserve momentum to
  # ~1e-7 (that error is O(Theta) = O(p/rho c^2) ~ 1e-5 per cell and cancels in the sum).
  # Recomputed from the smoothed primitives with the code's own Taub-Matthews EoS, which
  # reproduces the dumped columns to 1e-11.
  if all(c in df for c in ('D', 'sx', 'tau', 'vx')):
    v = df['vx'].to_numpy(dtype=float)
    D, s, tau = prim2cons(rho_new, v/np.sqrt(1. - v**2), df['p'].to_numpy(dtype=float))
    out['D'], out['sx'], out['tau'] = D, s, tau
  return out



# Extracting hydro data
#### Thinshell approx: only cells downstream are interesting
def get_critical_radii(key, z):
  '''
    Rc:     power-law behavior to constant transition (Tsph)
    Rf:     crossing time
  '''
  m, x_sph = get_plaws_hydro(key, z)[0]
  Rc = x_sph**(-1/m)
  Rf = get_Rcrossing(key, z)
  return Rc, Rf


def get_critical_obstimes(key, z, k=0.5):
  '''
  Important normalized observed times for peak modelization:
    Tc:     power-law behavior to constant transition
    Tf:     crossing time
    Tsat:   saturation of effective angle contribution from beaming
  '''
  Rc, Rf = get_critical_radii(key, z)
  Tf = Rf**(m+1)
  Tc = Rc**(m+1)
  env = MyEnv(key)
  g2 = (env.gRS if z == 4 else env.gFS)**2
  Tsat = 1 + (m+1)/(g2*k)
  return Tc, Tf, Tsat
  

def get_plaws_hydro(key, z):
  '''
  Returns the fitting parameters for downstream LF and shock strength
  '''

  data = open_rundata(key, z)
  popt_lfac, popt_ShSt, _, _, _ = get_hydrofits_shell(data)
  x_lf, m, s_lf = popt_lfac
  x_sh, n, o, s_sh = popt_ShSt
  return [-m, x_lf], [-n, x_sh]

def get_Rcrossing(key, z):
  '''
  Get final radius of the shock front at crossing, in units R0
  '''
  data = open_rundata(key, z)
  env = MyEnv(key)
  Rf = data.iloc[-1].x * c_ / env.R0
  return Rf

def get_tcrossing(key, z):
  '''
  Get crossing time of the shock front, in units t0
  '''
  data = open_rundata(key, z)
  env = MyEnv(key)
  tc = data.iloc[-1].t  / env.t0
  return tc

### Extract hydro data
def _thinshell_dump(args):
  '''
  One dump's front states, for extract_data_thinshell's pool. Module-level so it is
  picklable; returns (j, [values, one array per interface]) laid out exactly as the
  serial path builds them, so the two are bit-identical.
  '''
  key, it, j, cells, nCD, nSH, varlist, Nk = args
  df, t = openData_withtime(key, it)
  out = []
  for z in cells:
    cell = df_get_frontsnCD(df, z, nCD=nCD, nSH=nSH)
    if cell.empty:
      values = np.zeros(Nk)
      values[0:2] = it, t
    else:
      cell_vals = cell.reindex(varlist[3:]).to_numpy()
      values = np.concatenate([[it, t, 0.], cell_vals])
    out.append(values)
  return j, out


def extract_data_thinshell(key, itmin=0, itmax=None,
    cells=[1, 4], nCD=1, nSH=5, savefile=True, noOut=False, noPrint=False, nproc=1):
  '''
  Analyze a run, returning pandas dataframes one for each interface
  1: downstream FS, 2: CD in S2, 3: CD in S3, 4: downstream RS
  if savefile, writes it in a corresponding .csv file
  !!! if itmin != 0, starts at first iteration AFTER itmin

  nproc > 1 spreads the per-dump work over a forkserver pool (cell_pool). The dumps are
  independent and results are placed by index, so any worker count gives the SAME table --
  verified bit-identical against the serial path. nproc=1 (the default) keeps the original
  serial loop untouched. This is not a micro-optimisation at high resolution: one 20800-cell
  dump costs 0.23 s to read and front-detect, so a 153k-dump run is ~9.8 h on one core.
  '''

  fpaths = [get_runfile(key, z)[0] for z in cells]
  df0 = openData(key, it=0)
  varlist = df0.keys().to_list()
  varlist.insert(1, 'dt')
  varlist.insert(0, 'it')
  varlist.append('vx_u')

  its = np.array(dataList(key, itmin, itmax))
  if itmin:
    its = [it for it in its if i>itmin]
  Nc = len(cells)
  Nj = len(its)
  Nk = len(varlist)
  datas = np.zeros((Nc, Nj, Nk))
  dics = [{} for i in range(Nc)]
  dfs = []
  nproc = cell_pool.resolve_nproc(nproc, cap=Nj) if nproc != 1 else 1
  if nproc > 1:
    tasks = [(key, int(it), j, list(cells), nCD, nSH, varlist, Nk)
             for j, it in enumerate(its)]
    # chunk so each worker gets ~40 dumps at a time: the payload is tiny and the per-task
    # cost is ~0.2 s, so this keeps dispatch overhead negligible without hurting balance
    chunk = max(1, min(64, Nj // (nproc*8) or 1))
    done = 0
    with cell_pool.cell_executor(nproc) as ex:
      for j, out in ex.map(_thinshell_dump, tasks, chunksize=chunk):
        for i in range(Nc):
          datas[i, j] += out[i]
        done += 1
        if not noPrint and done % 5000 == 0:
          print(f"{done}/{Nj} dumps analysed", flush=True)
  else:
   for j, it in enumerate(its):
    if not noPrint:
      if it % 1000 == 0:
        print(f"Analyzing file of it = {it}")
    df, t = openData_withtime(key, it)
    for i, z in enumerate(cells):
      cell = df_get_frontsnCD(df, z, nCD=nCD, nSH=nSH)
      if cell.empty:
        values = np.zeros(Nk)
        values[0:2] = it, t
      else:
        # values = cell.to_numpy(copy=True)
        # # leave room for dt
        # values = np.insert(values, 1, 0.)
        # values = np.insert(values, 0, it)
        # cell_vals = cell.reindex(varlist[2:]).to_numpy()
        # values = np.concatenate([[it, 0.], cell_vals])
        cell_vals = cell.reindex(varlist[3:]).to_numpy()   # skip 'dt', start from 'nact'
        values = np.concatenate([[it, t, 0.], cell_vals])  # values[1]=t, values[2]=0 (dt placeholder)
      datas[i, j] += values

  # add dt
  for i, z in enumerate(cells):
    t = datas[i,:,1]
    dt = np.gradient(t)
    datas[i,:,2] = dt
    dics[i] = {varlist[k]:datas[i,:,k] for k in range(Nk)}
    df = pd.DataFrame.from_dict(dics[i]).set_index('it')
    dfs.append(df)
    if savefile:
      df.to_csv(fpaths[i])
  if not noOut:
    return dfs



_CELLS_CTX = {}


def _cells_init(key, klist, cols, window, rough):
  """Per-worker context for the cell extraction: read-only, shipped once."""
  _CELLS_CTX.update(key=key, klist=np.asarray(klist), cols=list(cols),
                    window=window, rough=rough)


def _cells_chunk(task):
  """
  One contiguous run of dumps, for every cell of the current block. Returns
  (j0, array of shape (nk, len(its), ncol)). Module-level so it is picklable.

  The gather is ONE `df.loc[klist, cols]` per dump. The loop this replaces did
  `df.at[k, var]` per (cell, column) -- 280 000 scalar pandas lookups per dump at
  20 000 cells, 4e10 over a hi-res run, which is why the old extractor could not be
  pointed at one.
  """
  j0, its = task
  c = _CELLS_CTX
  kl, cols = c['klist'], c['cols']
  out = np.empty((len(kl), len(its), len(cols)))
  for j, it in enumerate(its):
    df = openData(c['key'], it)
    df = smooth_dump_density(df, c['window'], c['rough'])
    out[:, j, :] = df.loc[kl, cols].to_numpy()
  return j0, out


def extract_data_cells(key, klist, itmin=0, itmax=None, itstep=None,
    savefile=True, noOut=False,
    rho_smooth_window=RHO_SMOOTH_WINDOW, rho_smooth_rough=RHO_SMOOTH_ROUGH,
    nproc=1, cell_block=None, dump_chunk=64):
  '''
  Extracts hydro data by reorganizing into cells history
    klist: list of cell indices to extract
      if klist = None, extracts all cells
  rho_smooth_window / rho_smooth_rough: passed to smooth_dump_density, applied to each
    snapshot before the per-cell pull. This is the ONLY place the moving-mesh density
    jitter is corrected -- everything downstream (open_celldata -> generate_cell_fromData
    -> the emission kernels) then reads clean rho with no change of its own.
    window=None disables it and reproduces the raw extraction exactly.

  cell_block: how many cells to hold at once (None = all). THE MEMORY KNOB. One block
    is a dense (nk, n_dumps, ncol) float64 array = nk * n_dumps * 112 bytes; 20 000
    cells over the 153 224-dump hi-res run is 368 GB, so it MUST be blocked there
    (8155 cells fits in 150 GB). Each block is one more full pass over the dumps, so
    prefer fewer, larger blocks.
  nproc: workers for the dump loop inside a block (forkserver pool, cell_pool). Dumps
    are independent and every chunk is placed by its own index, so the result does not
    depend on the worker count -- verified bit-identical, as is nproc=1 against the
    pre-rewrite extractor.
  dump_chunk: dumps per task; each task returns nk * dump_chunk * 112 bytes.
  '''

  dirpath = get_dirpath(key)
  # itstep: keep only every itstep-th iteration (dataList already supports it). Use when a
  # run's dump cadence is far denser than the runs it will be compared with -- e.g. a
  # near-vacuum ambient keeps the shock detector busy, so 'crossed' fires late, the
  # cadence never drops to ITDUMP_LATE_, and the run lands ~6x more dumps than its
  # counterparts for the same physical span. Safe for anything measured on a common
  # radial grid (see boundary_comparison.crash_radius_uniform, verified cadence-
  # independent); NOT safe for anything differentiating over a fixed number of rows.
  its = dataList(key, itmin, itmax, itstep)[0:]
  # create subfolder to save cells
  Path(dirpath+'cells').mkdir(parents=True, exist_ok=True)

  d0 = openData(key, 0)
  cols = list(d0.keys())
  if not klist:
    klist = d0.loc[d0['trac']>0]['i'].to_list()
  klist = list(klist)
  its = np.asarray(its)
  nb = len(klist) if cell_block is None else int(cell_block)
  out_arr = [] if not noOut else None

  for b0 in range(0, len(klist), nb):
    kblock = klist[b0:b0+nb]
    arr = np.zeros((len(kblock), len(its), len(cols)))
    tasks = [(j0, tuple(int(x) for x in its[j0:j0+dump_chunk]))
             for j0 in range(0, len(its), dump_chunk)]
    npb = cell_pool.resolve_nproc(nproc, cap=len(tasks)) if nproc != 1 else 1
    tag = f'cells {kblock[0]}-{kblock[-1]}'
    if npb > 1:
      # BOUNDED submission window, not ex.map. map() submits every task at once and holds
      # each completed result until the ones before it have been yielded, so the peak is
      # the WHOLE run's chunks: 2395 x 5000 x 64 x 14 x 8 = 86 GB on top of the 86 GB
      # block array. The first hi-res extraction (slurm 931793) peaked at 188.7 GB against
      # a 180 GiB request -- it finished, but with nothing to spare. Results are placed by
      # their own j0, so completion order never mattered; only map's ordering guarantee
      # forced the buffering. 2*nproc in flight keeps it to a few GB.
      import concurrent.futures as cf
      import itertools
      with cell_pool.cell_executor(npb, initializer=_cells_init,
            initargs=(key, kblock, cols, rho_smooth_window, rho_smooth_rough)) as ex:
        pending, queue, n = set(), iter(tasks), 0
        for t in itertools.islice(queue, 2*npb):
          pending.add(ex.submit(_cells_chunk, t))
        while pending:
          done, pending = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
          for fut in done:
            j0, chunk = fut.result()
            arr[:, j0:j0+chunk.shape[1], :] = chunk
            n += 1
            if not (n % 200):
              print(f'{tag}: {min(n*dump_chunk, len(its))}/{len(its)} dumps', flush=True)
            nxt = next(queue, None)
            if nxt is not None:
              pending.add(ex.submit(_cells_chunk, nxt))
    else:
      _cells_init(key, kblock, cols, rho_smooth_window, rho_smooth_rough)
      for n, task in enumerate(tasks):
        j0, chunk = _cells_chunk(task)
        arr[:, j0:j0+chunk.shape[1], :] = chunk
        if not (n % 200):
          print(f'{tag}: {min((n+1)*dump_chunk, len(its))}/{len(its)} dumps', flush=True)

    for i, k in enumerate(kblock):
      dic = {'it': its.astype(int)}
      dic.update({var: arr[i, :, c] for c, var in enumerate(cols)})
      out_k = pd.DataFrame.from_dict(dic).set_index('it')
      if savefile:
        # IO.CELL_FMT, not to_csv: at hi-res the sweep re-reads every cell once per point
        # and the CSV tokenizer dominates. Existing CSV runs stay readable (get_cellfile).
        save_celldata(key, k, out_k)
      if not noOut:
        out_arr.append(out_k)
    del arr

  if savefile:
    # provenance: which de-jittering these cells were built with
    with open(dirpath + 'cells/_extraction.json', 'w') as f:
      json.dump({'rho_smooth_window': rho_smooth_window,
                 'rho_smooth_rough': rho_smooth_rough,
                 'itmin': itmin, 'itmax': itmax}, f)

  if not noOut:
    return out_arr
