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
def analyze_run(key):

  '''
  Analyze a run, creating a pandas dataframe and corresponding .csv file
  including interfaces run_radii, number of cells
  '''

def get_radii_new(key='Last', itmin=0, itmax=None):

  '''
  Returns various R(t) of z given results folder, writes them in a file
  Rewritten for a pandas dataframe
  Add checking for similarity with existing data and write missing ones
  '''
  df = open_rundata(key)
  

def get_radii(key='Last', itmin=0, itmax=None):

  '''
  Returns various R(t) of z given results folder, writes them in a file
  3 cases: no file, file but incomplete, full file
  TBD: change incomplete case for real data comparison later
  '''
  dfile_path, dfile_bool = get_runfile(key)
  its = np.array(dataList(key, itmin, itmax))
  fullData = True

  if dfile_bool:  # if file exists, compare nb. of data lines vs data files
    with open(dfile_path, 'r') as f:
      nd = len(f.readlines()) - 1
      if nd < len(its):
        fullData = False
  else:           # if not, create it and write the header
    df  = openData_withZone(key, its[-1])
    Nz  = int(df['zone'].max())
    radlist = ['R_ts', 'R_b', 'R_sh', 'R_rs', 'R_cd', 'R_fs']
    header = "it\ttime"
    for rad in radlist[:Nz]:
      header += "\t" + rad 

    header += "\n"
    with open(dfile_path, 'w') as f:
      f.write(header)
    dfile_bool = True
    fullData = False

  if not fullData:
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

    its_miss = its[its>last_it]
    readwrite_radii(key, its_miss, dfile_path)
    fullData = True

  if dfile_bool and fullData:
    data = np.loadtxt(dfile_path, skiprows=1).transpose()
    time = data[1]
    imin = its.tolist().index(itmin)
    if itmax:
      imax = its.tolist().index(itmax)
    else:
      imax = len(its)
    run_radii = data[2:][imin:imax]
  
  return time, run_radii

def readwrite_radii(key, its, filename):
  '''
  Read datafiles in key, its and write results in filename, without header
  '''
  for i, it in enumerate(its):
    df, t, dt = openData_withtime(key, it)
    line = f"{it}\t{dt}\t"
    radii = df_get_radii(df)
    for r_n in radii:
      line += f"{r_n}\t"
    line += "\n"
    with open(filename, 'a') as f:
      f.write(line)


