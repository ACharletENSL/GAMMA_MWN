'''
Extract hydro data from simulation, analysis
Contains:
  - get_critical_radii
  - get_critical_times
  - get_plaws_hydro
  - get_Rcrossing
  - extract_data_thinshell
  - extract_data_cell
'''

from IO import *
from phys_constants import *
from fits_hydro import get_hydrofits_shell


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
  popt_lfac2, popt_ShSt = get_hydrofits_shell(data)
  x_lf, m, s_lf = popt_lfac2
  x_sh, n, s_sh = popt_ShSt
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
def extract_data_thinshell(key, itmin=0, itmax=None,
    cells=[1, 2, 3, 4], savefile=True, noOut=False):
  '''
  Analyze a run, returning pandas dataframes one for each interface
  1: downstream FS, 2: CD in S2, 3: CD in S3, 4: downstream RS
  if savefile, writes it in a corresponding .csv file
  !!! if itmin != 0, starts at first iteration AFTER itmin
  '''

  fpaths = [get_runfile(key, z)[0] for z in cells]
  df0 = openData(key, it=0)
  varlist = df0.keys().to_list()
  varlist.insert(1, 'dt')
  varlist.insert(0, 'it')
  its = np.array(dataList(key, itmin, itmax))
  if itmin:
    its = [it for it in its if i>itmin]
  Nc = len(cells)
  Nj = len(its)
  Nk = len(varlist)
  datas = np.zeros((Nc, Nj, Nk))
  dics = [{} for i in range(Nc)]
  dfs = []
  for j, it in enumerate(its):
    if it % 1000 == 0:
      print(f"Analyzing file of it = {it}")
    df, t = openData_withtime(key, it)
    for i, z in enumerate(cells):
      cell = df_get_frontsnCD(df, z)
      if cell.empty:
        values = np.zeros(Nk)
        values[0:2] = it, t
      else:
        values = cell.to_numpy(copy=True)
        # leave room for dt
        values = np.insert(values, 1, 0.)
        values = np.insert(values, 0, it)
      datas[i, j] += values

  # add dt
  for i, z in enumerate(cells):
    t = datas[i,:,1]
    dt = np.gradient(t)
    datas[i,:,2] += dt
    dics[i] = {varlist[k]:datas[i,:,k] for k in range(Nk)}
    df = pd.DataFrame.from_dict(dics[i]).set_index('it')
    dfs.append(df)
    if savefile:
      df.to_csv(fpaths[i])
  if not noOut:
    return dfs



def extract_data_cells(key, klist, itmin=0, itmax=None,
    savefile=True, noOut=False):
  '''
  Extracts hydro data by reorganizing into cells history
    klist: list of cell indices to extract
      if klist = None, extracts all cells
  '''

  dirpath = get_dirpath(key)
  its = dataList(key, itmin, itmax)[0:]
  # create subfolder to save cells
  Path(dirpath+'cells').mkdir(parents=True, exist_ok=True) 

  # create dictionaries with arrays to fill
  d0 = openData(key, 0)
  varlist = d0.keys()
  varlist = varlist.insert(0, 'it')
  if not klist:
    klist = d0.loc[d0['trac']>0]['i'].to_list()
  dics_arr = [{var:np.zeros(len(its)) for var in varlist} for k in klist]
  
  for j, it in enumerate(its):
    if not (it%100):
      print(f'Opening file it {it}')
    df = openData(key, it)
    for i, k in enumerate(klist):
      dics_arr[i]['it'][j] += it
      for var in varlist[1:]:
        dics_arr[i][var][j] += df.at[k, var]
  
  for dic in dics_arr:
    dic['it'] = dic['it'].astype(int)
  out_arr = [pd.DataFrame.from_dict(dic).set_index('it') for dic in dics_arr]

  
  if savefile:
    for k, out_k in zip(klist, out_arr):
      cellfile, _ = get_cellfile(key, k)
      out_k.to_csv(cellfile, index=it)
  
  if not noOut:
    return out_arr