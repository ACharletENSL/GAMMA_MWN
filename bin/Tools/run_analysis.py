# -*- coding: utf-8 -*-
# @Author: acharlet

'''
This file analyze opened data
'''

# Imports
# --------------------------------------------------------------------------------------------------
import numpy as np
from data_IO import *


# Analysis functions
# --------------------------------------------------------------------------------------------------
def get_timeseries(var, key):

  '''
  Return asked variable as a function of time and generate legend handle accordingly
  return time, var, var_legends
  '''
  run_data = open_clean_rundata(key)

  Nz = get_Nzones(key)
  if var == 'v':
    varlist = get_varlist('R', Nz)
    time = run_data['time'].to_numpy()
    out = run_data['time'].copy(deep=True)
    vlist = get_varlist('v', Nz)
    for rad, vel in zip(varlist, vlist):
      r = run_data[rad].to_numpy()
      v = np.gradient(r, time)
      vcol = pd.DataFrame({vel: v}, index=out.index)
      out = pd.concat([out, vcol], axis='columns')
  else:
    varlist = get_varlist(var, Nz)
    varlist.insert(0, 'time')
    out = run_data[varlist].copy(deep=True)
  
  return out

    
  # get variable(s) asked
  # create plot legend and return everything

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

def analyze_run(key, itmin=0, itmax=None):

  '''
  Analyze a run, filling a pandas dataframe
  writes it in a corresponding .csv file
  /!\ if itmin != 0, starts at first iteration AFTER itmin
  '''
  dfile_path, dfile_bool = get_runfile(key)
  varlist = prep_header(key)
  varlist.insert(0, 'time')
  data = pd.DataFrame(columns=varlist)
  its = np.array(dataList(key, itmin, itmax))
  if itmin:
    its = [it for it in its if i>itmin]
  for i, it in enumerate(its):
    df, t, dt = openData_withtime(key, it)
    tup = df_get_all(df)
    results = [item for arr in tup for item in arr]
    results.insert(0, dt)
    dict = {var:res for var, res in zip(varlist, results)}
    data.loc[it] = pd.Series(dict)
    #entry = pd.DataFrame.from_dict(dict)
    #data = pd.concat([data, entry], ignore_index=True)
  
  data.index
  data.to_csv(dfile_path)

# olds

def get_radii_new(key='Last', itmin=0, itmax=None):

  '''
  Returns various R(t) of z given results folder, writes them in a file
  Rewritten for a pandas dataframe
  Add checking for similarity with existing data and write missing ones
  '''
  df = open_rundata(key)
  pass


def prep_fileheader(key, dfile_path):

  '''
  Create header for results file
  '''
  its = np.array(dataList(key, 0, None))
  df  = openData_withZone(key, its[-1])
  Nz  = int(df['zone'].max())
  radList  = ['R_' + sub for sub in intList[:Nz]]
  zoneList = zsubList[:Nz]
  varList  = [var + '_' + sub for var in zvarsList for sub in zoneList]
    
  header = "it\ttime"
  for rad in radList:
    header += "\t" + rad 
  for var in varList:
    header += "\t" + var
  header += "\n"

  with open(dfile_path, 'w') as f:
    f.write(header)

def resultsFile_lastit(dfile_path):

  '''
  Get the last iteration written in the results file
  '''
  with open(dfile_path, 'rb') as f:
    try:    # catch OSError in case of one line file
      f.seek(-2, os.SEEK_END)
      while f.read(1) != b'\n':
        f.seek(-2, os.SEEK_CUR)
      last_line = f.readline().decode()
      last_it = int(last_line.split('\t')[0])
    except OSError:
      f.seek(0)
      last_it = -1.
  
  return last_it


def get_timeseries_old(var, key, slope=False, positive=False, itmin=0, itmax=None):

  '''
  Return the asked variable as function of time
  '''
  plot_legend = []
  intVars  = ['R', 'vel']
  if var in intVars:
    varname = {'R':'R', 'vel':'\\beta'}
    time, res = get_radii(key, itmin, itmax)
    Nr = res.shape[0]
    if var != 'R':
      res = np.gradient(res, time, axis=1)
      if positive:
        time, vars = return_posvalues(time, res)
        res = np.log10(vars)
    plot_legend = ['$' + varname[var] + '_' + sub + '$' for sub in intList[:Nr]]
  else:
    time, res = get_zoneIntegrated(var, key, itmin, itmax)
    Nr = res.shape[0]
    plot_legend = ['$' + var + '_' + sub + '$' for sub in zsubList[:Nr]]
  
  if slope:
    logt = np.log10(time)
    logvars = np.log10(res)
    res = np.gradient(logvars, logt, axis=1)
    new_legends = ["d$\\log(" + var_leg.replace('$', '') + ")/$d$\\log r$" for var_leg in var_legends]
    plot_legend = new_legends

  return time, res, plot_legend


def get_radii(key='Last', itmin=0, itmax=None):

  '''
  Returns various R(t) of z given results folder, writes them in a file
  3 cases: no file, file but incomplete, full file
  TBD: change incomplete case for real data comparison later
  '''
  dfile_path, dfile_bool = get_runfile(key)
  its = np.array(dataList(key, itmin, itmax))
  df  = openData_withZone(key, its[-1])
  zone = df['zone']
  Nz = int(zone.max())
  fullData = True

  if dfile_bool:  # if file exists, compare nb. of data lines vs data files
    with open(dfile_path, 'r') as f:
      nd = len(f.readlines()) - 1
      if nd < len(its):
        fullData = False
  else:           # if not, create it and write the header
    prep_fileheader(key, dfile_path)
    dfile_bool = True
    fullData = False

  if not fullData:
    last_it = resultsFile_lastit(dfile_path)
    its_miss = its[its>last_it]
    readwrite_datas(key, its_miss, dfile_path)
    fullData = True

  if dfile_bool and fullData:
    data = np.loadtxt(dfile_path, skiprows=1).transpose()
    time = data[1]
    imin = its.tolist().index(itmin)
    if itmax:
      imax = its.tolist().index(itmax)
    else:
      imax = len(its)
    run_radii = data[2:Nz+2][imin:imax]
  
  return time, run_radii

def get_zoneIntegrated(var, key='Last', itmin=0, itmax=None):

  '''
  Returns various var as function of timegiven results folder, writes them in a file
  3 cases: no file, file but incomplete, full file
  TBD: change incomplete case for real data comparison later
  '''

  dfile_path, dfile_bool = get_runfile(key)
  its = np.array(dataList(key, 0, None))
  df  = openData_withZone(key, its[-1])
  zone = df['zone']
  Nz = int(zone.max())
  its = np.array(dataList(key, itmin, itmax))
  fullData = True

  if dfile_bool:  # if file exists, compare nb. of data lines vs data files
    with open(dfile_path, 'r') as f:
      nd = len(f.readlines()) - 1
      if nd < len(its):
        fullData = False
  else:           # if not, create it and write the header
    prep_fileheader(key,dfile_path)
    dfile_bool = True
    fullData = False

  if not fullData:
    last_it = resultsFile_lastit(dfile_path)
    its_miss = its[its>last_it]
    readwrite_datas(key, its_miss, dfile_path)
    fullData = True

  if dfile_bool and fullData:
    data = np.loadtxt(dfile_path, skiprows=1).transpose()
    time = data[1]
    imin = its.tolist().index(itmin)
    if itmax:
      imax = its.tolist().index(itmax)
    else:
      imax = len(its)
    try:
      Nvar = Nz*(zvarsList.index(var) +1) + 2
    except ValueError:
      print("Requested var doesn't exist.")
      pass

    run_var = data[Nvar:Nvar+Nz][imin:imax]
  
  return time, run_var

def readwrite_datas(key, its, filename):
  '''
  Read datafiles in key, its and write results in filename, without header
  '''

  for i, it in enumerate(its):
    df, t, dt = openData_withtime(key, it)
    line = f"{it}\t{dt}\t"
    radii  = df_get_radii(df)
    Nz = len(radii)
    Nvar = len(zvarsList)
    zoneData = np.zeros(Nz*Nvar)
    for n, var in enumerate(zvarsList):
      res = df_get_zoneIntegrated(df, var)
      zoneData[Nz*n:Nz*(n+1)] = res
    for r_n in radii:
      line += f"{r_n}\t"
    for val in zoneData:
      line += f"{val}\t"
    line += "\n"
    with open(filename, 'a') as f:
      f.write(line)

def return_posvalues(time, vars):
  '''
  Return the time and variables where all are > 0
  Useful for interface velocity plots
  '''
  l = time.shape[0]
  Nv = vars.shape[0]
  i = 100
  for n in range(Nv): # find which array is the most stringent on v>0 condition
    v = vars[n][vars[n]>0.]
    if l >= len(v):
      i = n
      l = len(v)
    
  time = time[vars[i]>0.]
  res = np.zeros((Nv, time.shape[0]))
  for n in range(Nv):
    v = vars[n][vars[i]>0.]
    res[n] = v
  
  return time, res