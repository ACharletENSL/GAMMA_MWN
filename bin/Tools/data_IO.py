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
import scipy.signal as sps
import scipy.ndimage as spi
from phys_functions import *
from phys_constants import c_, pi_
# matplotlib for tests
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

# subscripts
interfaces_names = {
  'PWN':['{ts}', 'b', '{sh}', '{rs}', '{cd}', '{fs}'],
  'shells':['{rs}', '{cd}', '{fs}']
}
zone_names = {
  'PWN':['w', 'b', '{sh}', '{ej}', '{sh.ej}', '{sh.csm}', '{csm}'],
  'shells':['4', '3', '2', '1']
}

# Useful for testing
def df_get_primvar(df):
  rho = df['rho'].to_numpy()
  u   = df_get_u(df)
  p   = df['p'].to_numpy()
  return rho, u, p

def df_get_primsteps(df, h=0.02):
  rho, u, p = df_get_primvar(df)
  steps_rho = get_steps(rho, h)
  steps_u   = get_steps(u, h)
  steps_p   = get_steps(p, h)
  return steps_rho, steps_u, steps_p

def df_plot_primvar(df):
  rho, u, p = df_get_primvar(df)
  plt.plot(rho/rho.max(), label='$\\rho/\\rho_{max}$')
  plt.plot(u/u.max(), label='$u/u_{max}$')
  plt.plot(p/p.max(), label='$p/p_{max}$')
  plt.legend()
  plt.show()

def df_test_ilist(df):
  steps_rho, steps_u, steps_p = df_get_primsteps(df)
  steps = np.sort(np.concatenate((steps_rho, steps_u, steps_p)))
  ilist = []
  shlist = []
  wlist = []
  isteps = iter(steps)
  i = 0
  while i in range(len(steps)):
    s = steps[i]
    try:
      if ((s==steps[i+1]) and (s==steps[i+2])):
        shlist.append(s)
        ilist.append(s)
        i += 2
      elif (steps[i+2]<=s+10):
        wlist.append(s)
        ilist.append(s)
        i += 2
      else:
        ilist.append(s)
    except IndexError:
      ilist.append(s) 
    i += 1

  # identify end of wave
  dGp = step_gaussfilter(df['p'].to_numpy())
  for iw in wlist:
    if wlist.index(iw) > len(ilist)/2.:
      arr = - np.flip(dGp[iw:])
      print('flipped')
    else:
      arr = dGp[iw:]
    iint  = np.where(arr<=0)[0].min() + iw - 1  
    print(f"Wave at {iw}, shell at {iint}")
    ilist = [iint if i==iw else i for i in ilist]
    #wlist[wlist.index(iw)] = (iw, iint)
 
  return ilist, shlist, wlist


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
  mode, external = get_runatts(key)
  data.attrs['mode'] = mode
  data.attrs['external'] = external
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

def get_runatts(key):
  physpath  = get_physfile(key)
  attrs = ('mode', 'external')
  with open(physpath, 'r') as f:
    lines = f.read().splitlines()
    lines = filter(lambda x: x.startswith(attrs) , lines)
    for line in lines:
      name, value = line.split()[:2]
      if name == 'mode':
        mode = value
      elif name == 'external':
        external = value
  return mode, external

def open_rundata(key):

  '''
  returns a pandas dataframe with the extracted data
  '''
  dfile_path, dfile_bool = get_runfile(key)
  mode, external = get_runatts(key)
  if dfile_bool:
    df = pd.read_csv(dfile_path, index_col=0)
    df.attrs['mode'] = mode
    df.attrs['external'] = external
    return df
  else:
    return dfile_bool

def pivot(data, key):
  return(data.pivot(index="j", columns="i", values=key).to_numpy())

def dataList(key, itmin=0, itmax=None):

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
  file_path = dir_path + "phys_input.ini"
  if os.path.isfile(file_path):
    return file_path
  else:
    return GAMMA_dir + "/phys_input.ini"

def get_runfile(key):

  '''
  Returns path of file with analyzed datas over the run and boolean for its existence
  '''

  dir_path = GAMMA_dir + '/results/%s/' % (key)
  #file_path = dir_path + "extracted_data.txt"
  file_path = dir_path + 'run_data.csv'
  file_bool = os.path.isfile(file_path)
  return file_path, file_bool


def get_varlist(var, Nz, mode):
  '''
  Creates list of variables names with zone / interface subscripts
  ''' 
  varlist = []
  zvarsList = ['V', 'Emass', 'Ekin', 'Eint']
  ivarsList = ['R', 'v']
  zsubList  = zone_names[mode]
  intList   = interfaces_names[mode]

  # construct shocks list
  shList = [z for z in intList[:Nz-1] if z not in ['b', '{cd}']]

  if var == 'Nc':
    varlist = [var + '_' + sub for sub in zsubList[:Nz]]
  elif var in zvarsList:
    if mode == 'PWN':
      varlist = [var + '_' + sub for sub in zsubList[1:Nz-1]]
    elif mode =='shells':
      varlist = [var + '_' + sub for sub in zsubList[:Nz]]
  elif var == 'ShSt':
    varlist = [var + '_' + sub for sub in shList]
  elif var in ivarsList:
    varlist = [var + '_' + sub for sub in intList[:Nz-1]]
  else:
    print("Asked variable does not exist")
  
  return varlist

def prep_header(key):

  '''
  List of variables in the file
  '''
  vars = ['Nc', 'R', 'V', 'ShSt', 'Emass', 'Ekin', 'Eint']
  Nz  = get_Nzones(key)
  mode = get_runatts(key)[0]
  l = [get_varlist(var, Nz, mode) for var in vars]
  varlist = [item for sublist in l for item in sublist]

  return varlist

def get_Nzones(key):

  '''
  Return number of zones in the sim to consider for a run analysis
  '''

  dfile_path, dfile_bool = get_runfile(key)
  its = np.array(dataList(key))
  df  = openData_withZone(key, its[-1])
  zone = df['zone']
  Nz = int(zone.max())+1

  return Nz

# Functions on one data file
# --------------------------------------------------------------------------------------------------
def df_get_all(df):

  '''
  Extract all required datas:
  All interfaces position, cells number in each zone,
  volume and energy content of zones not touching box limits (in r)
  '''
  zone = df['zone']
  Nz = int(zone.max())
  Nint = Nz-1
  mode = df.attrs['mode']
  external = df.attrs['external']
  noext = True    # if simulation has zones at the edges we don't want to analyze
  if mode == 'MWN' or (mode == 'shells' and external):
    noext = False
  if noext:
    Nint = Nz+1
  res_rad = np.zeros(Nz)
  res_Nc  = np.zeros(Nz+1)
  res_V   = np.zeros(Nint)
  res_Em  = np.zeros(Nint)
  res_Ek  = np.zeros(Nint)
  res_Ei  = np.zeros(Nint)
  
  res_shocks = get_shocksStrength(df)

  for n in range(Nz):
    res_rad[n] = get_radius_zone(df, n+1)*c_
  for n in range(Nz+1):
    res_Nc[n]  = zone[zone==n].count()
  if noext:
    for n in range(Nz+1):
      res_V[n-1]   = zone_get_zoneIntegrated(df, 'V', n)
      res_Em[n-1]  = zone_get_zoneIntegrated(df, 'Emass', n)
      res_Ek[n-1]  = zone_get_zoneIntegrated(df, 'Ekin', n)
      res_Ei[n-1]  = zone_get_zoneIntegrated(df, 'Eint', n)
  else:
    for n in range(1, Nz):
      res_V[n-1]   = zone_get_zoneIntegrated(df, 'V', n)
      res_Em[n-1]  = zone_get_zoneIntegrated(df, 'Emass', n)
      res_Ek[n-1]  = zone_get_zoneIntegrated(df, 'Ekin', n)
      res_Ei[n-1]  = zone_get_zoneIntegrated(df, 'Eint', n)
  
  return res_Nc, res_rad, res_V, res_shocks, res_Em, res_Ek, res_Ei

def get_shocksStrength(df):
  '''
  Derivates shock strength at shock interfaces
  '''
  
  # expected number of shocks
  zone = df['zone'].to_numpy()
  Nz = int(zone.max())+1
  mode = df.attrs['mode']
  external = df.attrs['external']
  noext = True    # if simulation has zones at the edges we don't want to analyze
  if mode == 'MWN' or (mode == 'shells' and external):
    noext = False
  shlist = get_varlist('ShSt', Nz, mode)
  Ns = len(shlist)
  shstr_list = np.array([])
  Sd_list = []
  if df['Sd'].sum() == 0.:
    if len(np.unique(zone)) < Ns + 2:
      return np.zeros(Ns)
    if noext:
      try:
        Sd_list = [np.argwhere(zone==z).min() for z in [1., 3.]]
      except ValueError:
        return np.zeros(Ns)
    elif mode == 'shells':
      try:
        Sd_list = [np.argwhere(zone==z).min() for z in [2., 4.]]
      except ValueError:
        return np.zeros(Ns)
    elif mode == 'MWN':
      Sd_list = [np.argwhere(zone==z).min() for z in [1., 3., 4., 6.]]
    for i in Sd_list:
      i_u = i - 1
      i_d = i + 1
      shstr = df_get_shockStrength(df, i_u, i_d)
      shstr_list = np.append(shstr_list, shstr)

  else:
    Sd_list = df.index[df['Sd']==1.0].tolist()
    while len(Sd_list):
      i_u = Sd_list[0] - 1
      i_d = Sd_list[0] + 1
      if len(Sd_list) > 1:
        if Sd_list[1] - Sd_list[0] < 10: # maybe not best criterion?
          i_d = Sd_list[1] + 1
          del Sd_list[1]
      del Sd_list[0]
      shstr = df_get_shockStrength(df, i_u, i_d)
      shstr_list = np.append(shstr_list, shstr)
    while len(shstr_list) < Ns:
      shstr_list = np.append(shstr_list, 0.)

  return np.array(shstr_list)

def df_get_zoneIntegrated(df, var):

  '''
  Get the zone-integrated variable for a given data file
  '''

  zone = df['zone']
  Nz = int(zone.max())
  res = np.zeros(Nz-1)
  for n in range(1, Nz):
    res[n-1] = zone_get_zoneIntegrated(df, var, n)
  
  return res

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
    "dt":df_get_dt,
    "res":df_get_res
  }

  r = df["x"]
  #if smoothing:

  if var in calc_vars:
    out = calc_vars[var](df)
    return r, out
  elif var in df.columns:
    out = df[var]
    return r, out
  else:
    print("Required variable doesn't exist.")


# Zone identification
# --------------------------------------------------------------------------------------------------
def zoneID(df):
  p      = df['p'].to_numpy()
  zone   = np.zeros(p.shape)
  shocks = np.zeros(p.shape)
  steps_rho, steps_u, steps_p = df_get_primsteps(df)
  steps = np.sort(np.concatenate((steps_rho, steps_u, steps_p)))
  ilist = []
  shlist = []
  wlist = []
  isteps = iter(steps)
  i = 0
  while i in range(len(steps)):
    s = steps[i]
    try:
      if ((s==steps[i+1]) and (s==steps[i+2])):
        shlist.append(s)
        ilist.append(s)
        i += 2
      elif (steps[i+2]<=s+10):
        wlist.append(s)
        ilist.append(s)
        i += 2
      else:
        ilist.append(s)
    except IndexError:
      ilist.append(s) 
    i += 1

  # identify end of wave
  dGp = step_gaussfilter(p)
  for iw in wlist:
    if wlist.index(iw) > len(ilist)/2.:
      arr = - np.flip(dGp[iw:])
      print('flipped')
    else:
      arr = dGp[iw:]
    iint  = np.where(arr<=0)[0].min() + iw - 1  
    ilist = [iint if i==iw else i for i in ilist]
  
  shocks[shlist] = 1.
  l = 0
  for k in range(len(zone)):
    if k in ilist: l+=1
    zone[k] = l

  df2 = df.assign(zone=zone)
  if df['Sd'].sum() == 0.:
    df2 = df2.assign(Sd=shocks)
  return df2

def zoneID_old(df):

  '''
  Takes a panda dataframe as input and adds a zone marker column to it
  External interacting layer and CSM id to be added
  For PWN/MWN:
    0: unshocked wind
    1: shocked wind, jump in p at RS
    2: shocked shell, jump in rho at CD
    3: SNR, fall in p at FS
    4: shocked SNR, rho jump by 4
    5: shocked CSM, d rho / rho changes sign at CD
    6: unshocked CSM
  For shells:
    0 to 5 -> external medium, shells 4 to 1, external medium
  '''

  if 'zone' in df.columns:
    print("Zone column already exists in data")
    return(df)
  else:
    rho  = df['rho'] 
    p    = df['p']
    u    = df_get_u(df)
    zone = np.zeros(rho.shape)

    i_list = get_steps(rho)
    l=0
    for k in range(len(zone)):
      if k in i_list: l+=1
      zone[k] = l
    if df['t'].mean() == 0.: # ad hoc, need to correct this
      zone = np.where(zone>=2., zone+2., zone)
    df2 = df.assign(zone=zone)
    return df2

def get_steps(arr, h=0.02):

  '''
  Get step up in data by convoluting with step function
  '''

  d = step_gaussfilter(arr)
  
  # get peaks
  steps_up   = sps.find_peaks(d, height=h)[0]
  steps_down = sps.find_peaks(-d, height=h)[0]
  steps = np.concatenate((steps_up, steps_down))

  return np.sort(steps)

def step_gaussfilter(arr, sigma=1):
  '''
  Gaussian filtering part of the steps detection
  '''
  arr /= arr.max()
  smth_arr = spi.gaussian_filter1d(arr, sigma, order=0)
  d = np.gradient(smth_arr)
  return d

def step_convolve(arr):
  '''
  Convolution part of the steps detection
  '''
  d = arr - np.average(arr)
  step = np.hstack((np.ones(len(d)), -1*np.ones(len(d))))
  d_step = np.convolve(d, step, mode='valid')
  return d_step


def df_denoise_zonedata(df):
  '''
  Replace data with smoothed data, using zone conditions
  '''
  keylist = ['rho', 'vx', 'p', 'D', 'sx', 'tau']
  try:
    zone = df['zone']
  except KeyError:
    df = zoneID(df)
    zone = df['zone']
  for key in keylist:
    arr = df[key]
    arr_shw = arr[zone==1]
    arr_sm = denoise_data(arr_shw)
    out = np.where(zone==1, arr_sm, arr)
  pass


def denoise_data(arr, method='savgol', ker_size=30, interpol_deg=3, sigma=1):

  '''
  Denoising data for better gradient & slopes calculation
  '''
  out = np.zeros(arr.shape)
  if method == 'roll_ave':
    ker = np.ones(ker_size)/ker_size
    out = np.convolve(arr, ker, mode='same')
  elif method == 'savgol':
    if ker_size%2==0:
      ker_size +=1
    out = sps.savgol_filter(arr, ker_size, interpol_deg)
  elif method == 'gaussian':
    out = spi.gaussian_filter1d(arr, sigma=sigma)
  else:
    print('No method corresponding to this keyword.')
    pass
  
  return out

def zone_get_zoneIntegrated(df, var, n):

  '''
  Zone integrated var for zone of index n
  '''
  zone = df['zone']
  r = df['x']*c_
  dr = df['dx']*c_
  dV = 4.*pi_*r**2*dr
  res = 0.
  if var == 'Nc':
    res = zone[zone==n].count()
  else:
    if var=='V':
      data = dV
    else:
      r, data = get_variable(df, var)
      data *= dV
    res = data[zone==n].sum()
  
  return res


# Physical functions from dataframe
def df_get_T(df):
  rho = df["rho"].to_numpy(copy=True)
  p = df["p"].to_numpy(copy=True)
  return derive_temperature(rho, p)

def df_get_h(df):
  rho = df["rho"].to_numpy(copy=True)
  p = df["p"].to_numpy(copy=True)
  return derive_enthalpy(rho, p)

def df_get_Eint(df):
  rho = df["rho"].to_numpy(copy=True)
  p = df["p"].to_numpy(copy=True)
  return derive_Eint(rho, p)

def df_get_lfac(df):
  v = df["vx"].to_numpy(copy=True)
  return derive_Lorentz(v)

def df_get_u(df):
  v = df["vx"].to_numpy(copy=True)
  lfac = derive_Lorentz(v)
  return lfac * v

def df_get_Ekin(df):
  rho = df["rho"].to_numpy(copy=True)
  v = df["vx"].to_numpy(copy=True)
  return derive_Ekin(rho, v)

def df_get_Emass(df):
  rho = df["rho"].to_numpy(copy=True)
  return rho

def df_get_res(df):
  r = df["x"].to_numpy(copy=True)
  dr = df["dx"].to_numpy(copy=True)
  return dr/r

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

def df_detect_shocks(df):
  '''
  Detects shocks in data and adds a dataframe column 'Sd' if it doesn't exist
  '''
  if df['Sd'].sum() > 0.:
    # if shocks were already detected by GAMMA and written in data
    print('Shock detection already done by GAMMA')
    return df
  else:
    df['Sd'] = 0.
    df['Sd'].iloc[:-1] = df.apply(
                              lambda row: detect_shocks(row, df.iloc[row.name+1]) ,
                              axis=1).iloc[:-2]
    return df

def df_get_shockStrength(df, i_u, i_d):
  '''
  Derives the shock strength by comparing hydrodynamical properties
  upstream (at i_u) and downstream (at i_d)
  '''
  b_u = df['vx'].loc[i_u]
  g_u = derive_Lorentz(b_u)
  b_d = df['vx'].loc[i_d]
  g_d = derive_Lorentz(b_d)
  relg = g_u*g_d* (1. - b_u*b_d)
  return relg - 1.