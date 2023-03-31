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
  file_path = dir_path + "extracted_data.txt"
  file_bool = os.path.isfile(file_path)
  return file_path, file_bool

# Functions on one data file
# --------------------------------------------------------------------------------------------------
def get_radius_zone(df, n=0):

  '''
  Get the radius of the zone of chosen index
  (n=1 wind TS, n=2 nebula radius, etc.)
  '''
  if n==0:
    print("Asked n=0, returning rmin0")
  if 'zone' not in df.columns:
    df = zoneID(df)

  r = df['x']
  z = df['zone']
  rz = r[z == n].min()
  if np.isnan(rz):      # if no cell of the zone, take max of previous one
    rz = r[z==n-1].max()
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
    "Emass":df_get_Emass
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
    p   = df['p']
    trac= df['trac'].to_numpy()
    zone = np.zeros(trac.shape)
    i_w  = np.argwhere(trac >= 1.5).max()
    i_ej = np.argwhere((trac < 1.5) & (trac >= 0.5)).max()
    
    i_ts  = get_step(p)
    i_sh = get_step(-p)
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
