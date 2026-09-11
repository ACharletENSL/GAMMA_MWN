'''
Reading and writing data
'''

# Imports
# --------------------------------------------------------------------------------------------------
import os
import glob
import functools
from pathlib import Path
import numpy as np
import pandas as pd
import more_itertools as mit

from environment import MyEnv
from variables import var2func

# get local GAMMA directory
cwd = os.getcwd().split('/')
iG = [i for i, s in enumerate(cwd) if 'GAMMA' in s][0]
GAMMA_dir = '/'.join(cwd[:iG+1])

def get_dirpath(key):
  return GAMMA_dir + '/results/%s/' % (key)

def list_keys():
  res_dir = GAMMA_dir + '/results/'
  key_list = glob.glob(res_dir+'*')
  keys = [key.replace(res_dir, '') for key in key_list]
  return keys

# opening snapshots
##### opening file
def get_physfile(key):
  '''
  Returns path of phys_input file of the corresponding results folder
  '''
  
  dir_path = get_dirpath(key)
  file_path = dir_path + "phys_input.ini"
  if os.path.isfile(file_path):
    return file_path
  else:
    return GAMMA_dir + "/phys_input.ini"

@functools.lru_cache(maxsize=None)
def get_runatts(key):
  '''
  Returns attributes of the selected run

  Memoized: open_celldata calls this once per CELL, so a shell pass re-opened and
  re-parsed phys_input.ini 500 times at the fiducial resolution and 10000 times at
  hi-res. A run's inputs are frozen once it has been run, so caching them for the life
  of the process is safe; call get_runatts.cache_clear() if a .ini is edited in place
  from a live session.
  '''

  physpath  = get_physfile(key)
  attrs = ('mode', 'runname', 'rhoNorm', 'geometry')
  with open(physpath, 'r') as f:
    lines = f.read().splitlines()
    lines = filter(lambda x: x.startswith(attrs) , lines)
    for line in lines:
      name, value = line.split()[:2]
      if name == 'mode':
        mode = value
      elif name == 'runname':
        runname = value
      elif name == 'rhoNorm':
        rhoNorm = value
      elif name == 'geometry':
        geometry = value
  return [mode, runname, rhoNorm, geometry]

def openData(key, it=None, sequence=True):
  if sequence:
    filename = GAMMA_dir + '/results/%s/phys%010d.out' % (key, it)
  elif it is None:
    filename = GAMMA_dir + '/results/%s' % (key)
  else:
    filename = GAMMA_dir + '/results/%s%d.out' % (key, it)
  data = pd.read_csv(filename, sep=" ")
  mode, runname, rhoNorm, geometry = get_runatts(key)
  data.attrs['mode'] = mode
  data.attrs['runname'] = runname
  data.attrs['rhoNorm'] = rhoNorm
  data.attrs['geometry']= geometry
  data.attrs['key'] = key
  data.attrs['it'] = it
  env = MyEnv(key)
  data.attrs['rhoscale'] = env.rhoscale
  return(data)

def openData_withtime(key, it):
  '''Returns dataframe and the time counted from collision'''
  data = openData(key, 0, True)
  t0 = data['t'][0]
  data = openData(key, it)
  t = data['t'][0] - t0
  return data, t

##### returning arrays of desired vars
##### note: also works on time series dataframes
def get_vars(df, varlist, env):
  '''
  Return array of variables
  '''
  if type(df) == pd.core.series.Series:
    out = np.zeros((len(varlist)))
  else:
    out = np.zeros((len(varlist), len(df)))
  for i, var in enumerate(varlist):
    out[i] = get_variable(df, var, env)
  return out

def get_variable_cells(df_arr, var, env):
  '''
  Returns chosen variable for several cells
  '''
  out = []
  for df in df_arr:
    out.append(get_variable(df, var, env))
  return out

def get_variable(df, var, env):
  '''
  Returns chosen variable
  '''
  if var in df.keys():
    try:
      res = df[var].to_numpy(copy=True)
    except AttributeError:
      res = df[var]
  else:
    func, inlist, envlist = var2func[var]
    res = df_get_var(df, func, inlist, envlist, env)
  return res

def df_get_var(df, func, inlist, envlist, env):
  params, envParams = get_params(df, inlist, envlist, env)
  res = func(*params, *envParams)
  return res

def get_params(df, inlist, envlist, env):
  if (type(df) == pd.core.series.Series):
    params = np.empty(len(inlist))
  else:
    params = np.empty((len(inlist), len(df)))
  for i, var in enumerate(inlist):
    if var in df.attrs:
      params[i] = df.attrs[var].copy()
    else:
      try:
        params[i] = df[var].to_numpy(copy=True)
      except AttributeError:
        params[i] = df[var]
  envParams = []
  if len(envlist) > 0.:
    envParams = [getattr(env, name) for name in envlist]
  return params, envParams

##### detecting shocks
def df_get_frontsnCD(df, z, nCD=1, nSH=5):
  if z == 1:
    return df_get_cellBehindShock(df, 'FS', nSH)
  elif (z == 2) or (z == 3):
    return df_get_CDcell(df, z, nCD)
  elif z == 4:
    return df_get_cellBehindShock(df, 'RS', nSH)

def df_get_CD_cells(df, n=1):
  '''
  Returns the cells on both sides of the CD
  '''
  CDm, CDp = [df_get_CDcell(df, z, n) for z in [3,2]]

def df_get_CDcell(df, z, n=1):
  '''
  Return the cell at CD for given zone id (2 or 3)
  '''
  trac = df['trac']
  if z==2:
    shell = (trac > 1.5)
    sign = 1
    func = np.min
  else:
    shell = (trac < 1.5)
    sign = -1
    func = np.max
  S = df.loc[shell]
  i = func(S.index)
  CD = df.iloc[i+sign*n]
  return CD

def df_get_cellsBehindShock(df, n=5):
  return [df_get_cellBehindShock(df, sh, n) for sh in ['RS', 'FS']]

def df_get_cellBehindShock(df, shFront, n=5, m=1, up=3):
  '''
  Return a cell object with radiative values & position taken at shock front
  and hydro values taken downstream (n cells after shock)
  '''
  if shFront not in ['RS', 'FS']:
    print("'shFront' input must be 'RS' or 'FS'")
    out = pd.DataFrame({key:[] for key in df.keys()})
    for key in df.attrs.keys():
      out.attrs[key] = df.attrs[key]
    return out

  hdvars = ['dx', 'rho', 'vx', 'p', 'D', 'sx', 'tau']
  iCD = df.loc[(df['trac'] > 0.99) & (df['trac'] < 1.01)].index.max()
  RS, FS = df_to_shocks(df)
  if shFront == 'RS':
    sh = RS
    up = -up
  elif shFront == 'FS':
    sh = FS
  for key in df.attrs.keys():
    sh.attrs[key] = df.attrs[key]
  if sh.empty:
    return sh
  else:
    i_min, i_max = sh.index.min(), sh.index.max()
    if shFront == 'RS':
      i = i_min
      i_d = min(i_max+n,iCD-m)
    elif shFront == 'FS':
      i = i_max
      i_d = max(i_min-n,iCD+1+m)

    # downstream values
    front = df.iloc[i].copy()
    down = df.iloc[i_d]
    for key in df.attrs.keys():
      front.attrs[key] = df.attrs[key]
      down.attrs[key] = df.attrs[key]
    front[hdvars] = down[hdvars]

    # save upstream velocity for shock strength.
    # The upstream sample sits `up` cells beyond the front, which can fall outside the
    # grid once the shock comes within `up` cells of a domain edge. That never happened
    # while every run carried Next external-medium cells on each side, but with Next = 0
    # the shell edge IS the domain edge: the FS overruns the top (IndexError) and -- worse
    # -- the RS goes negative, where .iloc wraps silently to the far end of the array and
    # returns the WRONG cell with no error. Clamp to the outermost cell, which with
    # zero-gradient ghosts is exactly the upstream state anyway.
    i_up = min(max(i + up, 0), len(df) - 1)
    vx_u = df.iloc[i_up]['vx']
    front['vx_u'] = vx_u
    return front

   
def df_to_shocks(df):
  '''
  Extract the shocked part of data
  Cleans 'false' shock at wave onset when resolution is not high enough
  '''

  out = []
  S4 = df.loc[(df['trac'] > 0.99) & (df['trac'] < 1.01)]
  S1 = df.loc[(df['trac'] > 1.99) & (df['trac'] < 2.01)]
  i4b = S4.index.min()
  icd = S4.index.max()
  i1f = S1.index.max()

  # separate into blocks of consecutive indices
  RSsh = df.loc[(df['Sd']==1.) & (df.index <= icd)]
  iterable = RSsh.index.to_list()
  RSilist = [list(group) for group in mit.consecutive_groups(iterable)]
  RSlist = [df.iloc[iarr] for iarr in RSilist]

  FSsh = df.loc[(df['Sd']==1.) & (df.index > icd)]
  iterable = FSsh.index.to_list()
  FSilist = [list(group) for group in mit.consecutive_groups(iterable)]
  FSlist = [df.iloc[iarr] for iarr in FSilist]

  for (front, ilims, shlist) in zip(['RS', 'FS'], [(i4b, icd), (icd, i1f)], [RSlist, FSlist]):
    iL, iR = ilims
    crossedSh = [sh for sh in shlist if ((sh.index.max()<iL) | (sh.index.min()>iR))]
    if (len(crossedSh) > 0) or (len(shlist)== 0):
        out.append(pd.DataFrame(columns=df.keys()))
    elif len(shlist) == 1:
      out.append(shlist[0])
    else:
      fronts = [sh for sh in shlist if ((sh.index.min()>=iL) & (sh.index.max()<=iR))]
      if front == 'RS':
        def pjump(sh):
          pL = df.iloc[sh.index.min() - 1].p
          pR = df.iloc[sh.index.max() + 1].p
          return pR/pL
      else:
        def pjump(sh):
          pL = df.iloc[sh.index.min() - 1].p
          pR = df.iloc[sh.index.max() + 1].p
          return pL/pR
      pjumps = [pjump(sh) for sh in fronts]
      i_max = pjumps.index(max(pjumps))
      out.append(fronts[i_max])
  return out

# opening sims
#### time series
def dataList(key, itmin=0, itmax=None, itstep=None):
  '''
  For a given key, return the list of iterations in the folder
  '''

  its = []
  dir_path = get_dirpath(key)
  for path in os.scandir(dir_path):
    if path.is_file:
      fname = path.name
      if fname.endswith('.out'):
        it = int(fname[-14:-4])
        its.append(it)
  its = sorted(its)
  ittemp = its
  if itmax:
    imax = its.index(itmax)
    its = its[:imax]
  if itmin:
    imin = its.index(itmin)
    its = its[imin:]
  if itstep:
    its = [it for it in its if (it%itstep == 0)]
  return its

# Storage format for NEW cell extractions. 'npz' is an uncompressed np.savez of one array
# per column: same bytes as the in-memory float64, no text round-trip, and np.load reads
# it with a header parse + memcpy instead of a CSV tokenizer. On a hi-res run a cell
# history is ~90k rows x 15 columns, the sweep re-reads every cell once per point, and
# read_csv is ~100x slower than np.load on that shape -- at 1e4 cells that is the
# difference between minutes and hours of pure parsing, per point.
# 'csv' remains readable forever (see get_cellfile): existing runs are NOT re-extracted,
# so no published number moves.
CELL_FMT = 'npz'
CELL_EXTS = ('npz', 'csv')      # read preference order

# reserved np.savez keys for what a DataFrame carries besides its columns. Prefixed so
# they cannot collide with a hydro column name.
_CELL_IDX, _CELL_IDXNAME, _CELL_COLS = '__index__', '__index_name__', '__columns__'

def get_cellfile(key, k, fmt=None):
  '''
  Returns path of file with data of cell k and boolean for its existence.
  fmt: None (default) resolves for READING -- the first extension in CELL_EXTS that
    exists, so a run extracted to either format just works, and falls back to the
    CELL_FMT path (with False) when the cell is absent. An explicit 'npz'/'csv' returns
    that exact path, which is what writing wants.
  '''
  dir_path = get_dirpath(key)
  stem = dir_path + f'cells/{k:04d}'
  if fmt is not None:
    file_path = f'{stem}.{fmt}'
    return file_path, os.path.isfile(file_path)
  for ext in CELL_EXTS:
    file_path = f'{stem}.{ext}'
    if os.path.isfile(file_path):
      return file_path, True
  return f'{stem}.{CELL_FMT}', False

def _read_cell_npz(path):
  '''Rebuild the extracted-cell DataFrame written by _write_cell_npz (columns, order,
  dtypes and index all preserved exactly -- unlike the CSV path, which re-infers them).'''
  with np.load(path, allow_pickle=False) as d:
    cols = [str(c) for c in d[_CELL_COLS]]
    df = pd.DataFrame({c: d[c] for c in cols}, columns=cols)
    name = str(d[_CELL_IDXNAME])
    df.index = pd.Index(d[_CELL_IDX], name=(name if name else None))
  return df

def _write_cell_npz(path, df):
  '''One array per column + the index, uncompressed (savez, not savez_compressed: the
  point is load speed, and these are dense float64 with nothing to gain from deflate).'''
  name = df.index.name
  np.savez(path, **{c: df[c].to_numpy() for c in df.columns},
           **{_CELL_COLS: np.array([str(c) for c in df.columns]),
              _CELL_IDX: df.index.to_numpy(),
              _CELL_IDXNAME: np.array(name if name is not None else '')})

def open_cellframe(key, k, cols):
  """
  open_celldata restricted to `cols` (the stored index and attrs are kept), or False if
  the cell is not extracted.

  For callers that need a DataFrame -- because what they pass it to is pandas-shaped --
  but only a few of the fourteen columns. Reading a cooling_g100_hires cell cold costs
  4.85 s for all fourteen against 3.46 s for the nine the emission path uses: the file
  open dominates (~2.3 s of it is fixed, whatever the column count), so this is a 1.40x
  saving, not the order of magnitude a column count would suggest. Use open_cellcolumns
  where plain arrays will do.
  """
  dfile_path, dfile_bool = get_cellfile(key, k)
  if not dfile_bool:
    return dfile_bool
  if dfile_path.endswith('.npz'):
    with np.load(dfile_path, allow_pickle=False) as d:
      keep = [c for c in (str(c) for c in d[_CELL_COLS]) if c in cols]
      if not keep:
        return False
      df = pd.DataFrame({c: d[c] for c in keep}, columns=keep)
      name = str(d[_CELL_IDXNAME])
      df.index = pd.Index(d[_CELL_IDX], name=(name if name else None))
  else:
    df = open_celldata(key, k)
    if df is False:
      return False
    df = df[[c for c in df.columns if c in cols]]
  mode, runname, rhoNorm, geometry = get_runatts(key)
  df.attrs['key'] = key
  df.attrs['mode'] = mode
  df.attrs['runname'] = runname
  df.attrs['rhoNorm'] = rhoNorm
  df.attrs['geometry'] = geometry
  return df


def open_cellcolumns(key, k, cols):
  """
  Named columns of cell k as plain float arrays, or None if the cell is not extracted.

  open_celldata builds the whole DataFrame -- 14 columns plus the index, dtypes, column
  order -- and that construction, NOT the I/O, is what a per-cell scan pays for. Measured
  on a cooling_g100_hires cell (153224 rows): open_celldata 3.637 s, the five columns
  measured_injection_event needs 0.017 s, a 213x difference. Over a 10000-cell shell that
  is 10 hours against three minutes, which is why a shell pass must not reach for a
  DataFrame it will immediately unwrap.

  _write_cell_npz stores one array per column with savez (ZIP_STORED), so a member is read
  on its own without touching the rest. The CSV path has no such addressing and falls back
  to the full read.
  """
  dfile_path, dfile_bool = get_cellfile(key, k)
  if not dfile_bool:
    return None
  if dfile_path.endswith('.npz'):
    with np.load(dfile_path, allow_pickle=False) as d:
      if not all(c in d.files for c in cols):
        return None
      return {c: np.asarray(d[c], dtype=float) for c in cols}
  df = open_celldata(key, k)
  if df is False or not len(df):
    return None
  return {c: df[c].to_numpy(dtype=float) for c in cols}


def open_celldata(key, k):
  '''
  Returns a pandas dataframe with the extracted data from cell k
  (either storage format -- see CELL_FMT), or False if the cell is not extracted.
  '''
  dfile_path, dfile_bool = get_cellfile(key, k)
  if not dfile_bool:
    return dfile_bool
  mode, runname, rhoNorm, geometry = get_runatts(key)
  if dfile_path.endswith('.npz'):
    df = _read_cell_npz(dfile_path)
  else:
    df = pd.read_csv(dfile_path, index_col=0)
  df.attrs['key'] = key
  df.attrs['mode']    = mode
  df.attrs['runname'] = runname
  df.attrs['rhoNorm'] = rhoNorm
  df.attrs['geometry'] = geometry
  return df

def save_celldata(key, k, df, fmt=None):
  '''
  Write one extracted cell history in CELL_FMT (or an explicit fmt). Counterpart of
  open_celldata: the two formats must stay round-trip identical, which is what the
  npz path buys by storing dtypes instead of re-inferring them from text.
  '''
  fmt = CELL_FMT if fmt is None else fmt
  path, _ = get_cellfile(key, k, fmt=fmt)
  if fmt == 'npz':
    _write_cell_npz(path, df)
  elif fmt == 'csv':
    df.to_csv(path, index=True)
  else:
    raise ValueError(f"unknown cell format {fmt!r} (expected one of {CELL_EXTS})")
  return path

def get_fitsfile(key, z):
  # path = GAMMA_dir+'/extracted_data/'
  # fname = path + key + '_'
  path = get_dirpath(key) 
  front = 'FS' if z == 1 else 'RS'
  fname = path + 'fit_' + front+'.out'
  return fname

def open_fits(key, z, l=4):
  '''
  l the typical length of popt
  '''
  fname = get_fitsfile(key, z)
  logau, Tf, t_max, *rest = np.loadtxt(fname)
  # params per variable
  groups = [4, 4, 4, 4, 3]
  popts = []
  i = 0
  for n in groups:
    popts.append(np.array(rest[i:i+n]))
    i += n
  popt_lfac, popt_ShSt, popt_nu, popt_L = popts
  return Tf, t_max, popt_lfac, popt_ShSt, popt_nu, popt_L, popt_xi

def join_extracted(keys, noPks=False):
  '''
  Join the extracted data from the sweep in one table
  '''
  prefix = ['log_aum']
  varnames_1 = [] if noPks else ['Tf', 't_max']
  varnames_2 = [name+'_'+s for name in ['lfac', 'ShSt', 'nu', 'L']
      for s in ['A', 'x_b', 'alpha', 's']]
  varnames_3 = ['xi_' + s for s in ['k', 'sat', 's']]
  varnames = prefix + varnames_1 + varnames_2 + varnames_3
  header = "\t".join(varnames)
  N = len(varnames)
  folder = GAMMA_dir + '/extracted_data/'
  # for front in ['FS', 'RS']:
  #   search_exp = folder + 'sweep*' + front + '.out'
  #   files = glob.glob(search_exp)
  for front, z in zip(['RS', 'FS'], [4, 1]):
    files = [get_fitsfile(key, z) for key in keys]
    arr0 = np.loadtxt(files[0])
    if len(arr0) != N:
      print(f'Header length {N} != array length {len(arr0)}')
      return 0.
    arrays = [np.loadtxt(f) for f in files]
    table = np.stack(arrays)
    table = table[table[:, 0].argsort()]
    fmt = '%2f, ' + ','.join(['%10f']*(N-1)) 
    np.savetxt(folder + "fullsweep_au_"+front+".csv",
      table, delimiter='\t', header=header, comments='')

def open_sweep(front):
  path = GAMMA_dir+'/extracted_data/'
  fname = path + 'fullsweep_au_' + front +'.csv'
  df = pd.read_csv(fname, sep='\t')
  return df

def logau_to_key(log_au):
  key = 'sweep_' + f"log_au={log_au:.1f}"
  return key

def check_done_logau():
  folders = glob.glob(GAMMA_dir + "/results/sweep_log_au=*")
  done_logau = np.array(sorted([float(f.split("=")[1]) for f in folders]))
  return done_logau

def get_runfile(key, z):
  '''
  Returns path of file with analyzed datas over the run and boolean for its existence
  '''

  dir_path = get_dirpath(key)
  file_path = dir_path + f'run_data_{z}.csv'
  file_bool = os.path.isfile(file_path)
  return file_path, file_bool

def open_rundata(key, z):
  '''
  Returns a pandas dataframe with the extracted data from zone z
  '''

  dir_path = get_dirpath(key)
  fold_exist = os.path.isdir(dir_path)
  dfile_path, dfile_bool = get_runfile(key, z)
  mode, runname, rhoNorm, geometry = get_runatts(key)
  if dfile_bool:
    df = pd.read_csv(dfile_path, index_col=0)
    df.attrs['key'] = key
    df.attrs['mode']    = mode
    df.attrs['runname'] = runname 
    df.attrs['rhoNorm'] = rhoNorm
    return df[df.x > 0.]
  else:
    return dfile_bool

#### Files related to radiation
def get_radfile_thinshell(key, front):
  '''
  Returns path of file containing radiation derived in thinshell regime
  front: string  'RS' or 'FS'
  '''
  
  dir_path = get_dirpath(key)
  file_path = dir_path + f"radiation_vFC_{front}.npz"
  return file_path

def get_radfile_activity(key, front):
  '''
  Returns path of file containing frequency ratio and corresponding
    crossed radii and activity time
  front: string  'RS' or 'FS'
  '''

  dir_path = get_dirpath(key)
  file_path = dir_path + f"source_activity_{front}.npz"
  return file_path