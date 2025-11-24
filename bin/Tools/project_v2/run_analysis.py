'''
Analyzing a run (extracting hydro, producing radiation..)
'''

from IO import *

def thinshell_radiation(key, z, nuobs, Tobs):
  '''
  Derives observed flux from shock front (z=4: RS, z=1: FS)
    in thinshell approximation
  '''
  return 0.


# Extracting hydro data
#### Thinshell approx: only cells downstream are interesting
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
