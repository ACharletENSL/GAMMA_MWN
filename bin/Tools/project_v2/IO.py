'''
Reading and writing data
'''

# Imports
# --------------------------------------------------------------------------------------------------
import os
import glob
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

def get_runatts(key):
  '''
  Returns attributes of the selected run
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

    # save upstream velocity for shock strength
    vx_u = df.iloc[i+up]['vx']
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

def get_cellfile(key, k):
  '''
  Returns path of file with data of cell k and boolean for its existence
  '''
  dir_path = get_dirpath(key)
  file_path = dir_path + f'cells/{k:04d}.csv'
  file_bool = os.path.isfile(file_path)
  return file_path, file_bool

def open_celldata(key, k):
  '''
  Returns a pandas dataframe with the extracted data from cell k
  '''
  dfile_path, dfile_bool = get_cellfile(key, k)
  mode, runname, rhoNorm, geometry = get_runatts(key)
  if dfile_bool:
    df = pd.read_csv(dfile_path, index_col=0)
    df.attrs['key'] = key
    df.attrs['mode']    = mode
    df.attrs['runname'] = runname 
    df.attrs['rhoNorm'] = rhoNorm
    df.attrs['geometry'] = geometry
    return df
  else:
    return dfile_bool

def get_fitsfile(key, z):
  path = GAMMA_dir+'/extracted_data/'
  fname = path + key + '_'
  front = 'FS' if z == 1 else 'RS'
  path = fname+front+'.out'
  return path

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

def join_extracted(noPks=False):
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
  for front in ['FS', 'RS']:
    search_exp = folder + 'sweep*' + front + '.out'
    files = glob.glob(search_exp)
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