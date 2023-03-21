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
def get_radii(key='Last', itmin=0, itmax=None):

  '''
  Returns various R(t) of z given results folder
  Add checking for similarity with existing data and write missing ones
  '''
  dfile_path, dfile_bool = get_runfile(key)
  all_its = dataList(key, 0, None)
  its = dataList(key, itmin, itmax)

  if dfile_bool:
    with open(dfile_path, 'r') as f:
      header = f.readline().strip('\n')
      vars = header.split('\t')
      Nr = sum([True for var in vars if var.startswith('R_')])
    
    data = np.loadtxt(dfile_path, skiprows=1).transpose()
    time = data[0]
    imin = its.index(itmin)
    if itmax:
      imax = its.index(itmax)
    else:
      imax = len(its)
    radii = data[1:Nr+1][imin:imax]

  else:
    Nit = len(its)
    time= np.zeros(Nit)
    df  = openData_withZone(key, its[-1])
    Nz  = int(df['zone'].max())
    radlist = ['R_ts', 'R_b', 'R_sh', 'R_rs', 'R_cd', 'R_fs']
    header = "time\t"
    for rad in radlist[:Nz]:
      header += rad + "\t"
    header += "\n"
    with open(dfile_path, 'w') as f:
      f.write(header)

    radii = np.zeros((Nz, Nit))

    for i, it in enumerate(its):
      df, t, dt = openData_withtime(key, it)
      time[i] = dt
      line = f"{dt}\t"
      for n in range(Nz):
        r_n = get_radius_zone(df, n+1)
        radii[n,i] = r_n
        line += f"{r_n}\t"
      line += "\n"
      with open(dfile_path, 'a') as f:
        f.write(line)   
  
  return time, radii
