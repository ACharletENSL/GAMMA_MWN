'''
Analyzing a run (extracting hydro, producing radiation..)
'''

from IO import *
from phys_constants import *
from hydro_fits import get_hydrofits_shell

def thinshell_radiation(key, z, nuobs, Tobs):
  '''
  Derives observed flux from shock front (z=4: RS, z=1: FS)
    in thinshell approximation
  '''
  return 0.


# Extracting hydro data
#### Thinshell approx: only cells downstream are interesting
def critical_radii(key, z):
  '''
    Rc:     power-law behavior to constant transition (Tsph)
    Rf:     crossing time
  '''
  m, x_sph = extract_plaws(key, z)[0]
  Rc = x_sph**(-1/m)
  Rf = get_Rcrossing(key, z)
  return Rc, Rf


def critical_times(key, z, k=0.5):
  '''
  Important normalized observed times for peak modelization:
    Tc:     power-law behavior to constant transition
    Tf:     crossing time
    Tsat:   saturation of effective angle contribution from beaming
  '''
  Rc, Rf = critical_radii(key, z)
  Tf = Rf**(m+1)
  Tc = Rc**(m+1)
  env = MyEnv(key)
  g2 = (env.gRS if z == 4 else env.gFS)**2
  Tsat = 1 + (m+1)/(g2*k)
  return Tc, Tf, Tsat
  

def extract_plaws(key, z):
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

def analyze_run(key, itmin=0, itmax=None,
    cells=[1, 2, 3, 4], savefile=True):
  '''
  Analyze a run, returning pandas dataframes one for each cell
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
    if it % 100 == 0:
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
  return dfs
