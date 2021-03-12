# -*- coding: utf-8 -*-
# @Author: Eliot Ayache
# @Date:   2021-01-30 17:59:32
# @Last Modified by:   Eliot Ayache
# @Last Modified time: 2021-03-12 20:30:34

import pandas as pd
import numpy as np
import h5py

def readData(key, it=None, sequence=False):
  if sequence:
    filename = '../../results/%s/phys%010d.out'  %(key,it)
  elif it==None:
    filename = '../../results/%s'  %(key)
  else:
    filename = '../../results/%s%d.out'  %(key,it)
  data = pd.read_csv(filename, sep=" ")
  return(data)




def extract(data, key):
  return(data[key].to_numpy())

def pivot(data, key):
  return(data.pivot(index="j", columns="i", values=key).to_numpy())


def pandas2double(data):

  T = data["t"][0] # time

  rho = extract(data, "rho")
  p = extract(data, "p")
  vx = extract(data, "vx")
  vy = extract(data, "vy")
  x = extract(data, "x")
  dset = np.column_stack((rho, p, vx, vy, x))  # cell values

  ntot = data.shape[0]
  Nr = pivot(data, "nact")[:,0] # number of cells in each track
  Nr = np.expand_dims(Nr,1)
  index = np.arange(ntot)
  data["index"] = index
  Index = pivot(data, "index")[:,0]
  Index = np.expand_dims(Index,1)
    # Index of first cell of track in memory
  th = pivot(data, "y")[:,0]
  dth = pivot(data, "dy")[:,0]
  t_jph = th - dth/2.
  t_jph = np.append(t_jph, t_jph[-1]+dth[-1]) # theta interface positions

  p_kph = np.array([0,2*np.pi]) # phi interface posisions

  return(dset, Index, Nr, T, p_kph, t_jph)





def toH5(key, it, sequence):
  pandata = readData(key, it, sequence)

  filename = '../../results/%s/phys%010d.h5'  %(key,it)
  f = h5py.File(filename, 'a')

  values, Index, Nr, T, p_kph, t_jph = pandas2double(pandata)

  # fluid values
  data = f.create_group("Data")
  dset = f.create_dataset("Data/Cells", data=values)

  # Geometry
  grid = f.create_group("Grid")
  dset = f.create_dataset("Grid/Index", data=Index)
  dset = f.create_dataset("Grid/Nr", data=Nr)
  dset = f.create_dataset("Grid/T", data=T)
  dset = f.create_dataset("Grid/p_kph", data=p_kph)
  dset = f.create_dataset("Grid/t_jph", data=t_jph)



