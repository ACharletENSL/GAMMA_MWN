# -*- coding: utf-8 -*-
# @Author: acharlet

'''
This file opens and writes GAMMA data files, based on plotting_scripts.py
in original GAMMA folder
to add: 
  data analysis at opening, identifying zones and adding marker
  writing arrays in a GAMMA-compatible file to create initializations as we see fit
'''

# Imports
# --------------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd

# Read data
# --------------------------------------------------------------------------------------------------
def readData(key, it=None, sequence=True):
  if sequence:
    filename = '../../results/%s/phys%010d.out' % (key, it)
  elif it is None:
    filename = '../../results/%s' % (key)
  else:
    filename = '../../results/%s%d.out' % (key, it)
  data = pd.read_csv(filename, sep=" ")
  return(data)

def readData_withzone(key, it=None, sequence=True):
  data = readData(key, it, sequence)
  data_z = zoneID(data)
  return data_z

def pivot(data, key):
  return(data.pivot(index="j", columns="i", values=key).to_numpy())

def dataList(key, itmin, itmax):

  '''
  For a given key, return the list of iterations
  '''

  its = []
  dir_path = '../../results/%s/' % (key)
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

# Functions on one data file
# --------------------------------------------------------------------------------------------------
def get_radius_zone(df, n=0):

  '''
  Get the radius of the zone of chosen index
  (n=1 wind TS, n=2 nebula radius, etc.)
  '''
  if 'zone' not in df.columns:
    df = zoneID(df)

  r = df['x']
  z = df['zone']
  return r[zone == n].min()


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
    rho = df['rho']
    vx  = df['vx']
    p   = df['p']
    D   = df['D']
    zone = np.zeros(rho.shape)
    
    i_b  = get_step(p)
    i_s  = get_step(rho)
    i_ej = get_step(-p)

    zone[i_b:]  += 1
    zone[i_s:]  += 1
    zone[i_ej:] += 1

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