# -*- coding: utf-8 -*-
# @Author: acharlet

'''
This file opens and writes GAMMA data files, based on plotting_scripts.py
in original GAMMA folder
to add:
  writing arrays in a GAMMA-compatible file to create initializations as we see fit
'''

# Imports
# --------------------------------------------------------------------------------------------------
import os
import numpy as np
import pandas as pd
from phys_functions import *

# Read data
# --------------------------------------------------------------------------------------------------
cwd = os.getcwd().split('/')
iG = [i for i, s in enumerate(cwd) if 'GAMMA' in s][0]
GAMMA_dir = '/'.join(cwd[:iG+1])

def openData(key, it=None, sequence=True):
  if sequence:
    filename = GAMMA_dir + '/results/%s/phys%010d.out' % (key, it)
  elif it is None:
    filename = GAMMA_dir + '/results/%s' % (key)
  else:
    filename = GAMMA_dir + '/results/%s%d.out' % (key, it)
  data = pd.read_csv(filename, sep=" ")
  return(data)

def openData_withtime(key, it):
  data = openData(key, 0, True)
  t0 = data['t'][0]
  data = openData_withZone(key, it)
  t = data['t'][0] # beware more complicated sims with adaptive time step ?
  dt = t-t0
  return data, t, dt

def openData_withZone(key, it=None, sequence=True):
  data = openData(key, it, sequence)
  data_z = zoneID(data)
  return data_z

def open_rundata(key):

  '''
  returns a pandas dataframe with the extracted data
  '''
  dfile_path, dfile_bool = get_runfile(key)
  if dfile_bool:
    df = pd.read_csv(dfile_path, index_col=0)
    return df
  else:
    return dfile_bool

def pivot(data, key):
  return(data.pivot(index="j", columns="i", values=key).to_numpy())

def dataList(key, itmin, itmax):

  '''
  For a given key, return the list of iterations
  '''

  its = []
  dir_path = GAMMA_dir + '/results/%s/' % (key)
  for path in os.scandir(dir_path):
    if path.is_file:
      fname = path.name
      if fname.endswith('.out'):
        it = int(fname[-14:-4])
        its.append(it)
  its = sorted(its)
  ittemp = its
  if itmin and not(itmax):
    ittemp = its[itmin:]
  elif not(itmin) and itmax:
    ittemp = its[:itmax]
  elif itmin and itmax:
    ittemp = its[itmin:itmax]
  
  its = ittemp
  return its

def get_physfile(key):
  '''
  Returns path of phys_input file of the corresponding results folder
  '''
  dir_path = GAMMA_dir + '/results/%s/' % (key)
  file_path = dir_path + "phys_input.MWN"
  if os.path.isfile(file_path):
    return file_path
  else:
    return GAMMA_dir + "/phys_input.MWN"

def get_runfile(key):

  '''
  Returns path of file with analyzed datas over the run and boolean for its existence
  '''

  dir_path = GAMMA_dir + '/results/%s/' % (key)
  #file_path = dir_path + "extracted_data.txt"
  file_path = dir_path + 'run_data.csv'
  file_bool = os.path.isfile(file_path)
  return file_path, file_bool

# Functions on one data file
# --------------------------------------------------------------------------------------------------
def df_get_radii(df):

  '''
  Get the various radii from a given data file
  '''

  Nz = int(df['zone'].max())
  radii = np.zeros(Nz)
  for n in range(Nz):
    r_n = get_radius_zone(df, n+1)
    radii[n] = r_n
  
  return radii

def get_radius_zone(df, n=0):

  '''
  Get the radius of the zone of chosen index
  (n=1 wind TS, n=2 nebula radius/CD, etc.)
  '''
  if n==0:
    print("Asked n=0, returning rmin0")
  if 'zone' not in df.columns:
    df = zoneID(df)

  r = df['x']
  z = df['zone']
  rz = r[z == n].min()
  while(np.isnan(rz)):      # if no cell of the zone, take max of previous one
    n -= 1
    rz = r[z==n].max()
  return rz

def get_variable(df, var):

  '''
  Returns coordinate and chosen variable, in code units
  '''

  # List of vars requiring calculation, as dict with associated function
  calc_vars = {
    "T":df_get_T,
    "h":df_get_h,
    "lfac":df_get_lfac,
    "u":df_get_u,
    "Ekin":df_get_Ekin,
    "Eint":df_get_Eint,
    "Emass":df_get_Emass,
    "dt":df_get_dt
  }

  r = df["x"]
  if var in calc_vars:
    return r, calc_vars[var](df)
  elif var in df.columns:
    return r, df[var]
  else:
    print("Required variable doesn't exist.")


# Zone identification
# --------------------------------------------------------------------------------------------------
def zoneID(df):

  '''
  Takes a panda dataframe as input and adds a zone marker column to it
  External interacting layer and CSM id to be added
  0: unshocked wind
  1: shocked wind, jump in p at RS
  2: shocked shell, jump in rho at CD
  3: SNR, fall in p at FS
  4: shocked SNR, rho jump by 4
  5: shocked CSM, d rho / rho changes sign at CD
  6: unshocked CSM
  '''

  if 'zone' in df.columns:
    print("Zone column already exists in data")
    return(df)
  else:
    rho  = df['rho'] 
    p    = df['p']
    u    = df_get_u(df)
    trac = df['trac'].to_numpy()
    zone = np.zeros(trac.shape)
    i_w  = np.argwhere(trac >= 1.5).max()
    i_ej = np.argwhere((trac < 1.5) & (trac >= 0.5)).max()
    
    i_ts  = get_step(-u)
    i_sh = max(get_step(-p), get_step(rho))
    #print(i_w, i_ej, i_ts, i_sh)

    zone[:i_ts] = 0
    zone[i_ts:i_w+1]  = 1
    zone[i_w:i_sh+1]  = 2
    zone[i_sh:i_ej+1] = 3

    df2 = df.assign(zone=zone)
    return df2

def get_step(arr):

  '''
  Get step up in data by convoluting with step function
  '''

  d = arr - np.average(arr)
  step = np.hstack((np.ones(len(d)), -1*np.ones(len(d))))
  d_step = np.convolve(d, step, mode='valid')
  step_id = np.argmax(d_step)

  return step_id


# Physical functions from dataframe
def df_get_T(df):
  rho = df["rho"]
  p = df["p"]
  return derive_temperature(rho, p)

def df_get_h(df):
  rho = df["rho"]
  p = df["p"]
  return derive_enthalpy(rho, p)

def df_get_Eint(df):
  rho = df["rho"]
  p = df["p"]
  return derive_Eint(rho, p)

def df_get_lfac(df):
  v = df["vx"]
  return derive_Lorentz(v)

def df_get_u(df):
  v = df["vx"]
  lfac = derive_Lorentz(v)
  return lfac * v

def df_get_Ekin(df):
  rho = df["rho"]
  lfac = df_get_lfac(df)
  return (lfac-1)*rho

def df_get_Emass(df):
  rho = df["rho"]
  return rho

# code time step for each cell (before code takes the min of them)
def df_get_dt(df):
  '''
  Get time step for each cell from the Riemannian waves
  '''
  
  # create copies of dataframe with one ghost cell on left and right respectively
  # dfL : add cell with wind on the left
  dfL = df.copy(deep=True)
  row0 = dfL.iloc[0]
  dfL.loc[-1] = row0  # let's test with a simple copy
  dfL.index = dfL.index + 1
  dfL = dfL.sort_index()

  #dfR : add cell of ejecta on the right
  dfR = df.copy(deep=True)
  rowLast = pd.DataFrame(dfR.iloc[-1]).transpose() # let's test with a simple copy
  rowLast.index = rowLast.index + 1
  dfR = pd.concat([dfR, rowLast])

  # apply function
  N = len(df)+1
  dt = np.zeros(N-1)
  waves = np.zeros((N, 3))
  for i in range(N):
    SL = dfL.loc[i]
    SR = dfR.loc[i]
    lL, lR, lS = waveSpeedEstimates(SL, SR)
    waves[i] = [lL, lR, lS]
  
  for i in range(N-1):
    a = df.iloc[i]['dx']

    # wave from left side
    l = waves[i][1]
    v = waves[i][2]
    dt_candL = a / (l - v) if (l > v) else 1e15

    # wave from right side
    l = waves[i+1][0]
    v = waves[i+1][2]
    dt_candR = a / (v - l) if (l < v) else 1e15

    dt_cand = min(dt_candL, dt_candR)
    dt[i] = dt_cand

  return dt


  