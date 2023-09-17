# -*- coding: utf-8 -*-
# @Author: acharlet

'''
This file analyze opened data
'''

# Imports
# --------------------------------------------------------------------------------------------------
import numpy as np
from data_IO import *
from environment import MyEnv

# Analysis functions
# --------------------------------------------------------------------------------------------------
def get_timeseries(var, key):

  '''
  Returns dataframe with selected variables only
  '''
  run_data = open_clean_rundata(key)
  mode = run_data.attrs['mode']
  Nz = get_Nzones(key)
  extr_vars = ['Nc', 'R', 'u', 'ShSt', 'V', 'Emass', 'Ekin', 'Eint']
  calc_vars = {
    "Rcd":run_get_Rcd,
    "Rct":run_get_Rct,
    "v":run_get_v,
    "vcd":run_get_vcd,
    "epsth":run_get_epsth
  }

  if var in extr_vars:
    varlist = get_varlist(var, Nz, mode)
    varlist.insert(0, 'time')
    out = run_data[varlist].copy(deep=True)
  elif var in calc_vars:
    out = calc_vars[var](run_data)
  else:
    print("Implement function for requested variable")
    out = run_data
  
  return out



def open_clean_rundata(key):

  '''
  Returns pandas dataframe containing data from time series
  '''  
  # open file as pandas dataframe (if doesn't exists, returns False)
  dfile_path, dfile_bool = get_runfile(key)
  
  run_data = open_rundata(key)
  its = np.array(dataList(key))

  # check if .csv exists
  if not dfile_bool:
    analyze_run(key)
    run_data = open_rundata(key)
  
  last_it = run_data.index[-1]
  if last_it <= its[-1]:
    # check if last data row fits with current analysis
    df = openData_withZone(key, last_it)
    last_data = df_get_all(df)
    lastrow = run_data.iloc[-1]
    #check diffs
    same = True
    if same:
      if last_it < its[-1]:
        # continue after last_it
        analyze_run(key)
      else:
        # everything's good
        pass
    else:
      # delete file and do whole analysis
      os.remove(get_runfile(key)[0])
      analyze_run(key)
  else: # last_it > its[-1]
    # delete file and do whole analysis
    os.remove(get_runfile(key)[0])
    analyze_run(key)
  
  run_data = open_rundata(key)
  return run_data

def compare_crosstimes(key):
  '''
  Compare crossing times found numerically with the analytical values
  '''

  env = MyEnv(get_physfile(key))
  tRS_num, tFS_num = get_crosstimes(key)
  reldiff_tRS = reldiff(tRS_num, env.tRS)
  reldiff_tFS = reldiff(tFS_num, env.tFS)
  return reldiff_tRS, reldiff_tFS

def get_crosstimes(key):
  '''
  Outputs crossing time for a given run
  Find them by selecting when corresponding Ncells = 0
  '''
  run_data = open_clean_rundata(key)
  StRS = run_data['ShSt_{rs}'].to_numpy()[1:]
  StFS = run_data['ShSt_{fs}'].to_numpy()[1:]
  t   = run_data['time'].to_numpy()[1:]
  tRS = t[StRS==0.].min()
  tFS = t[StFS==0.].min()
  
  return tRS, tFS


def analyze_run(key, itmin=0, itmax=None):

  '''
  Analyze a run, filling a pandas dataframe
  writes it in a corresponding .csv file
  /!\ if itmin != 0, starts at first iteration AFTER itmin
  '''
  dfile_path, dfile_bool = get_runfile(key)
  varlist = vars_header(key)
  varlist.insert(0, 'time')
  data = pd.DataFrame(columns=varlist)
  its = np.array(dataList(key, itmin, itmax))
  if itmin:
    its = [it for it in its if i>itmin]
  for i, it in enumerate(its):
    print(f"Analyzing file of it = {it}")
    df, t, dt = openData_withtime(key, it)
    tup = df_get_all(df)
    results = [item for arr in tup for item in arr]
    results.insert(0, dt)
    dict = {var:res for var, res in zip(varlist, results)}
    data.loc[it] = pd.Series(dict)
    #entry = pd.DataFrame.from_dict(dict)
    #data = pd.concat([data, entry], ignore_index=True)
  
  data.index = its
  data.to_csv(dfile_path)


def vars_header(key, dfile_path=None):
  if not dfile_path:
    dfile_path, dfile_bool = get_runfile(key)

  allvars = ['Nc', 'R', 'u', 'ShSt', 'V', 'Emass', 'Ekin', 'Eint']
  Nz  = get_Nzones(key)
  mode = get_runatts(key)[0]
  varlist = []
  for var in allvars:
    varlist.extend(get_varlist(var, Nz, mode))
  
  return varlist

# functions for derived variables
def run_get_Rcd(run_data):
  '''
  Returns dataframe with distance of shock to interface
  (shocked shells width)
  '''
  time = run_data['time'].to_numpy()
  Rrs  = run_data['R_{rs}'].to_numpy()
  Rcd  = run_data['R_{cd}'].to_numpy()
  Rfs  = run_data['R_{fs}'].to_numpy()
  D3   = Rcd - Rrs
  D2   = Rfs - Rcd
  out  = pd.DataFrame(np.array([time, D3, D2]).transpose(), columns=['time', 'D3', 'D2'], index=run_data.index)
  return out

def run_get_Rct(run_data):
  '''
  Returns dataframe with distance of interfaces - c*t
  '''
  time = run_data['time'].to_numpy()
  Rrs  = run_data['R_{rs}'].to_numpy() - c_*time
  Rcd  = run_data['R_{cd}'].to_numpy() - c_*time
  Rfs  = run_data['R_{fs}'].to_numpy() - c_*time
  out  = pd.DataFrame(np.array([time, Rrs, Rcd, Rfs]).transpose(),
    columns=['time', 'R_{rs}', 'R_{cd}', 'R_{fs}'], index=run_data.index)
  return out

def run_get_v(run_data):
  '''
  Returns dataframe of interface velocities
  '''
  time = run_data['time'].to_numpy()
  Rrs  = run_data['R_{rs}'].to_numpy()
  Rcd  = run_data['R_{cd}'].to_numpy()
  Rfs  = run_data['R_{fs}'].to_numpy()
  outarr = [time]
  for r in [Rrs, Rcd, Rfs]:
    v = np.gradient(r, time)/c_
    outarr.append(v)
  out = pd.DataFrame(np.array(outarr).transpose(),
    columns=['time', 'v_{rs}', 'v_{cd}', 'v_{fs}'], index=run_data.index)
  return out

def run_get_vcd(run_data):
  '''
  Returns dataframe of interfaces velocities compared to the CD
  (shocked shells growth rate)
  '''
  shellw = run_get_Rcd(run_data)
  time   = shellw['time'].to_numpy()
  D3     = shellw['D3'].to_numpy()
  D2     = shellw['D2'].to_numpy()
  outarr = [time]
  for w in [D3, D2]:
    dw = np.gradient(w, time)
    outarr.append(dw)
  out = pd.DataFrame(np.array(outarr).transpose(),
    columns=['time', 'dD3', 'dD2'], index=run_data.index)
  return out

def run_get_epsth(run_data):
  '''
  Returns dataframe of thermal efficiencies in the shells
  (energy converted from kinetic to thermal)
  '''
  time = run_data['time'].to_numpy()
  eth3 = run_data['Eint_3'].to_numpy()/run_data['Ekin_4'][0]
  eth2 = run_data['Eint_2'].to_numpy()/run_data['Ekin_1'][0]
  out  = pd.DataFrame(np.array(time, eth3, eth2).transpose(),
    columns=['time', 'eth3', 'eth2'], index=run_data.index())
  return out