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
  '''
  dfile_path, dfile_bool = get_runfile(key)
  all_its = dataList(key, 0, None)
  its = dataList(key, itmin, itmax)

  if dfile_bool:
    with open(dfile_path, 'r') as f:
      header = f.readline().strip('\n')
      vars = header.split()
      Nr = sum([True for var in vars if var.startswith('R_')])
    
    data = np.loadtxt(dfile_path, skiprows=1).transpose()
    time = data[0]
    imin = its.index(itmin)
    if itmax:
      imax = its.index(itmax)
    else:
      imax = len(its)
    radii = data[1:Nr][imin:imax]

  else:
    Nit = len(its)
    time= np.zeros(Nit)
    df  = readData_withZone(key, its[-1])
    Nz  = int(df['zone'].max())
    radlist = ['R_ts', 'R_b', 'R_sh', 'R_rs', 'R_cd', 'R_fs']
    header = "time\t"
    for rad in radlist[:Nz]:
      header += rad + "\t"
    radii = np.zeros((Nz, Nit))

    for i, it in enumerate(its):
      df, t, dt = openData_withtime(key, it)
      time[i] = dt
      for n in range(Nz):
        r_n = get_radius_zone(df, n+1)
        radii[n,i] = r_n

    with open(dfile_path, 'w') as f:
      f.write(header)
      for i in len(its):
        line = f"{time[i]}\t"
        for n in range(Nz):
          line += f"{radii[n, i]}\t"
        f.write(line)
  
  return time, radii
