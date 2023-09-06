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

plt.ion()

# Useful for testing
def test_interfaces(it, key ='Last'):
  df = openData(key, it)
  df_test_interfaces(df)

def test_primvar(it, key='Last'):
  df = openData(key, it)
  df_plot_primvar(df)
  df_plot_filt(df)

def df_get_primvar(df):
  rho = df['rho'].to_numpy()
  u   = df_get_u(df)
  p   = df['p'].to_numpy()
  return rho, u, p

def df_get_gaussfprim(df, sigma=3):
  rho, u, p = df_get_primvar(df)
  dGrho = step_gaussfilter(rho, sigma)
  dGu   = step_gaussfilter(u, sigma)
  dGp   = step_gaussfilter(p, sigma)
  return dGrho, dGu, dGp

def df_get_primsteps(df, sigma=1, h=0.02):
  rho, u, p = df_get_primvar(df)
  steps_rho = get_steps(rho, sigma, h)
  steps_u   = get_steps(u, sigma, h, False)
  steps_p   = get_steps(p, sigma, h)
  return steps_rho, steps_u, steps_p

def df_allprimsteps(df, h=0.02):
  steps_rho, steps_u, steps_p = df_get_primsteps(df, h)
  steps = np.sort(np.concatenate((steps_rho, steps_u, steps_p)))
  return steps

def df_plot_primvar(df):
  rho, u, p = df_get_primvar(df)
  plt.figure()
  plt.plot(rho/rho.max(), label='$\\rho/\\rho_{max}$')
  plt.plot(u/u.max(), label='$u/u_{max}$')
  plt.plot(p/p.max(), label='$p/p_{max}$')
  plt.legend()

def df_plot_filt(df, sigma=3):
  dGrho, dGu, dGp = df_get_gaussfprim(df, sigma)
  plt.plot(dGrho/np.abs(dGrho).max(), label='$\\nabla \\tilde{\\rho}$')
  plt.plot(dGu/np.abs(dGu).max(), label='$\\nabla \\tilde{u}$')
  plt.plot(dGp/np.abs(dGp).max(), label='$\\nabla \\tilde{p}$')
  plt.legend()

def df_test_ilist_v2(df):
  steps = df_allprimsteps(df)
  iList = []
  isteps = iter(steps)
  i = 0
  while i in range(len(steps)):
    s = steps[i]
    try:
      s2 = steps[i+2]
      s1 = steps[i+1]
      if (s2<=s+10):
        interface = (s, 3, s2-s)
        iList.append(interface)
        i += 2
      elif (s1<=s+10):
        interface = (s, 2, s1-s)
        iList.append(interface)
        i += 1
      else:
        interface = (s, 1, 0)
        iList.append(interface)
    except IndexError:
      try:
        s1 = steps[i+1]
        if (s1<=s+10):
          interface = (s, 2, s1-s)
          iList.append(interface)
          i += 1
        else:
          interface = (s, 1, 0)
          iList.append(interface)
      except IndexError:
        interface = (s, 1, 0)
        iList.append(interface)
    i += 1
  return iList

def check_wave(df, interface):
  '''
  Input needs to be tuple (index, multiplicity, size)
  '''
  i, m, l = interface
  dGrho, dGu, dGp = df_get_gaussfprim(df, sigma=1)
  iLrho, iRrho = find_close0s(dGrho, i)
  iLu, iRu     = find_close0s(dGu, i)
  iLp, iRp     = find_close0s(dGp, i)
  len_int = min(iRrho-iLrho, iRu-iLu, iRp-iLp)
  if len_int < 40:
    if m == 3:
      return ("shock", i)
    else:
      return ("CD", i)
  else:
    iw = iRrho if (iRrho-i > i-iLrho) else iLrho
    return ("wave", iw)

def df_test_interfaces(df):
  ilist = df_test_ilist_v2(df)
  for interface in ilist:
    print(check_wave(df, interface))

def df_test_ilist(df):
  steps = df_allprimsteps(df)
  ilist = []
  shlist = []
  wlist = []
  isteps = iter(steps)
  i = 0
  while i in range(len(steps)):
    s = steps[i]
    try:
      if (steps[i+2]<=s+2):
        shlist.append(s)
        ilist.append(s)
        i += 2
      elif (steps[i+2]<=s+10):
        wlist.append(s)
        ilist.append(s)
        i += 2
      elif (steps[i+1]<=s+10):
        wlist.append(s)
        ilist.append(s)
        i += 1
      else:
        ilist.append(s)
    except IndexError:
      ilist.append(s) 
    i += 1

  # identify end of wave
  dGp = step_gaussfilter(df['p'].to_numpy())
  for iw in wlist:
    if wlist.index(iw) > len(ilist)/2.:
      arr = - np.flip(dGp[:iw])
      print('flipped')
    else:
      arr = dGp[iw:]
    iint  = np.where(arr<=0)[0].min() + iw - 1  
    print(f"Wave at {iw}, shell at {iint}")
    ilist = [iint if i==iw else i for i in ilist]
    #wlist[wlist.index(iw)] = (iw, iint)
 
  return ilist, shlist, wlist

def find_next0(arr, i, reverse=False, tol=1e-4):
  if arr[i] < 0:
    arr = -arr
  if reverse:
    arr = np.flip(arr[:i])
    i0  = np.where(arr<=tol)[0].min()
    ii  = i - i0
  else:
    arr = arr[i:]
    i0  = np.where(arr<=tol)[0].min()
    ii  = i + i0
  return ii

def find_close0s(arr, i):
  iL = find_next0(arr, i, True)
  iR = find_next0(arr, i, False)
  return iL, iR

def find_0to0(arr, i):
  iL, iR = find_close0s(arr, i)
  return iR - iL


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
    varlist = [var + '_' + str(n) for n in reversed(range(Nz))]
  elif var in zvarsList:
    if mode == 'PWN':
      varlist = [var + '_' + sub for sub in zsubList[1:Nz-1]]
    elif mode =='shells':
      varlist = [var + '_' + sub for sub in zsubList[:Nz]]
  elif var == 'ShSt':
    varlist = [var + '_' + sub for sub in shList]
  elif var in ivarsList:
    varlist = [var + '_' + sub for sub in intList[:Nz-1]]
  elif var == 'u':
    varlist = [var]
  else:
    print("Asked variable does not exist")
  
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
  Nz = int(zone.max()) + 1
  mode = df.attrs['mode']
  allvars = ['Nc', 'R', 'u', 'ShSt', 'V', 'Emass', 'Ekin', 'Eint']
  allres  = []
  for var in allvars:
    Nvar = len(get_varlist(var, Nz, mode))
    res = []
    if var == 'Nc':
      res = [zone[zone==n].count() for n in reversed(range(Nz))]
    elif var == 'R':
      zones = [3., 2., 1.] if mode == 'shells' else [n+1 for n in range(Nz-1)]
      res = [get_radius_zone(df, n)*c_ for n in zones]
    elif var == 'u':
      res = [get_shocked_u(df)]
    elif var == 'ShSt':
      res = get_shocksStrength(df)
    else:
      zones = [4., 3., 2., 1.] if mode == 'shells' else [n for n in range(1, Nvar+1)]
      res = [zone_get_zoneIntegrated(df, var, n) for n in zones]
    allres.append(res)
  
  return allres


def df_get_all_old(df):
  zone = df['zone']
  Nz = int(zone.max()) + 1
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
  if df.attrs['mode'] == 'shells':
    RS = df.loc[(df['Sd'] == 1.0) & (df['trac'] > 0.99) & (df['trac'] < 1.01)]
    try:
      iu_RS = min(RS.index) - 1
      id_RS = max(RS.index) + 1
      shst_RS = df_get_shockStrength(df, iu_RS, id_RS)
    except ValueError:
      shst_RS = 0.
    FS = df.loc[(df['Sd'] == 1.0) & (df['trac'] > 1.99) & (df['trac'] < 2.01)]
    try:
      iu_FS = max(FS.index) + 1
      id_FS = min(FS.index) - 1
      shst_FS = df_get_shockStrength(df, iu_FS, id_FS)
    except ValueError:
      shst_FS = 0.
    return [shst_RS, shst_FS]
  else:
    return get_shocksStrength_v2(df)
  

def get_shocksStrength_v2(df):
  zone = df['zone'].to_numpy()
  Nz = int(zone.max())+1
  mode = df.attrs['mode']
  external = df.attrs['external']
  shstr = []
  if mode == 'shells':
    try:
      Sd_list = [np.argwhere(zone==n).min() for n in [3., 1.]]
      for i in Sd_list:
        i_u = i - 5
        i_d = i + 5
        shstr.append(df_get_shockStrength(df, i_u, i_d))
      return shstr
    except ValueError:
      return np.zeros(2)
  else:
    return get_shocksStrength_old(df)

def get_shocked_u(df):
  zone = df['zone'].to_numpy()
  u    = df_get_u(df)
  u2   = u[zone==2.]
  u3   = u[zone==3.]
  u_sh = np.concatenate((u3[3:], u2[:-3]))
  if u_sh.size > 0:
    res = u_sh.mean()
  else:
    res = 0.
  return res


def get_shocksStrength_old(df):
  '''
  Derivates shock strength at shock interfaces
  '''
  
  # expected number of shocks
  zone = df['zone']
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

def get_radius_zone(df, n):

  '''
  Get the radius of the zone of chosen index
  (n=1 wind TS, n=2 nebula radius/CD, etc.)
  '''
  if 'zone' not in df.columns:
    df = zoneID(df)

  r = df['x']
  z = df['zone']
  rz = r[z == n].min()
  down = True
  if (n<3 and df.attrs['mode'] == 'shells'):
    down = False
  while(np.isnan(rz)):      # if no cell of the zone, take max of previous one
    if down:
      n -= 1
      rz = r[z==n].max()
    else:
      n -= 1
      rz = r[z==n].min()
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
def id_interfaces(df):
  '''
  Returns list of interfaces index, also one for shocks
  '''
  p = df['p']
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
      if (steps[i+2]<=s+2):
        shlist.append(s)
        ilist.append(s)
        i += 2
      elif (steps[i+2]<=s+10):
        wlist.append(s)
        ilist.append(s)
        i += 2
      elif (steps[i+1]<=s+10):
        wlist.append(s)
        ilist.append(s)
        i += 1
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
  
  return ilist, shlist

def id_interfaces_v2(df):
  interfaces = df_test_ilist_v2(df)
  ilist  = []
  shlist = []
  for interface in interfaces:
    int_type = check_wave(df, interface)
    if int_type[0] == "wave":
      ilist.append(int_type[1])
    else:
      i = interface[0]
      ilist.append(i)
      if int_type[0] == "shock":
        shlist.append(i)
  
  return ilist, shlist

def get_ilist(df):
  steps = df_allprimsteps(df)
  iList = []
  isteps = iter(steps)
  i = 0
  while i in range(len(steps)):
    s = steps[i]
    try:
      s2 = steps[i+2]
      s1 = steps[i+1]
      if (s2<=s+10):
        interface = (s, 3, s2-s)
        iList.append(interface)
        i += 2
      elif (s1<=s+10):
        interface = (s, 2, s1-s)
        iList.append(interface)
        i += 1
      else:
        interface = (s, 1, 0)
        iList.append(interface)
    except IndexError:
      try:
        s1 = steps[i+1]
        if (s1<=s+10):
          interface = (s, 2, s1-s)
          iList.append(interface)
          i += 1
        else:
          interface = (s, 1, 0)
          iList.append(interface)
      except IndexError:
        interface = (s, 1, 0)
        iList.append(interface)
    i += 1
  return iList

def get_ilist_v2(df):
  steps_rho, steps_u, steps_p = df_get_primsteps(df)

def get_lenInterface_old(dGrho, dGu, dGp, i):
  iLrho, iRrho = find_close0s(dGrho, i)
  iLu, iRu     = find_close0s(dGu, i)
  iLp, iRp     = find_close0s(dGp, i)
  iL = max(iLrho, iLu, iLp)
  iR = min(iRrho, iRu, iRp)
  lenL = i - min(iLrho, iLu, iLp)
  lenR = max(iRrho, iRu, iRp) - i
  return lenL, lenR

def get_limPeak(arr, i):
  lims = sps.peak_widths(arr, [i], rel_height=.75)
  iL = round(lims[2][0])
  iR = round(lims[3][0])
  return iL, iR

def get_lenInterface(dGrho, dGu, dGp, i):
  iLrho, iRrho = get_limPeak(np.abs(dGrho), i)
  iLu, iRu     = get_limPeak(np.abs(dGu), i)
  iLp, iRp     = get_limPeak(np.abs(dGp), i)
  iL = max(iLrho, iLu, iLp)
  iR = min(iRrho, iRu, iRp)
  lenL = i - min(iLrho, iLu, iLp)
  lenR = max(iRrho, iRu, iRp) - i
  return lenL, lenR

def test_forwave(interface, dGrho, dGu, dGp, asym=2.5):
  i, m, l = interface
  lenL, lenR = get_lenInterface(dGrho, dGu, dGp, i)


def id_interfaces_sps_cleaned(df):
  
  out = id_interfaces_sps(df)

  # for shell case, delete the shock propagating in external medium after crossing time
  if df.attrs['mode'] == 'shells':
    typelist = [item[0] for item in out]
    if ['waveOn', 'waveOff'] == typelist[1:3]:
      del out[0]
    if ['waveOff', 'waveOn'] == typelist[-3:-1]:
      del out[-1]

  return out

def id_interfaces_sps(df):
  ilist = get_fused_interf(df)
  out   = []

  # check interface closest to grid lim 
  for item in ilist:
    i, iL, iR, m = item
    type = ''
    # special case of the most external interfaces for colliding shells
    # can only be the shell interface with external medium or a shock
    if (item==ilist[0] or item==ilist[-1]) and df.attrs['mode'] == 'shells':
      if m == 1:
        type = 'shell'
      else:
        type = 'shock'
    else:
      lenL = i - iL
      lenR = iR - i
      # test for asymmetry in peak size
      # may want to correct criteria if not good enough (test absolute diff?)
      if np.abs(lenR - lenL) < 5:
      # by construction, m = 1 means jump in density only
        if m == 1:
          type = 'CD'
        else:
          type = 'shock'
      else:
        type = 'waveOn'
        iw   = iL if lenL > lenR else iR
        out.append(('waveOff', iw))
    
    out.append((type, i))
  
  # sort in case wave onset and end are not ordered
  out.sort(key=lambda x: x[1])
  return out

def get_fused_interf(df):
  int_rho, int_u, int_p = get_allinterf(df)
  ilist = []

  # loop on rho interfaces because they are more numerous
  for item in int_rho:
    # check if the interface is present in the other variables
    m = 1
    i, iL, iR = item
    check_u = []
    check_p = []
    for itemu in int_u:
      iu, iLu, iRu = itemu
      if (iu >= iL) and (iu <= iR):
        check_u.append((i, iLu, iRu))
    if bool(check_u):
      m +=1
      iLu = min([itemu[1] for itemu in check_u])
      iRu = max([itemu[2] for itemu in check_u])
      iL = min(iL, iLu)
      iR = max(iR, iRu)
    for itemp in int_p:
      ip, iLp, iRp = itemp
      if (ip >= iL) and (ip <= iR):
        check_p.append((i, iLp, iRp))
    if bool(check_p):
      m +=1
      iLp = min([itemp[1] for itemp in check_p])
      iRp = max([itemp[2] for itemp in check_p])
      iL = min(iL, iLp)
      iR = max(iR, iRp)
    ilist.append((i, iL, iR, m))
  
  return ilist

def get_allinterf(df):
  rho, u, p = df_get_primvar(df)
  # filtering rho because data can be noisy
  rho       = spi.gaussian_filter1d(rho, sigma=1, order=0)
  int_rho   = get_varinterf(rho)
  int_u     = get_varinterf(u, False)
  int_p     = get_varinterf(p)
  return int_rho, int_u, int_p

def get_varinterf(arr, takelog=True, h=.05, prom=.05, rel_h=.99):
  if takelog:
    arr = np.log10(arr)
  arr   = np.gradient(arr)
  interfaces = get_peaksid(arr, h, prom, rel_h)
  return interfaces

def get_peaksid(arr, h=.05, prom=.05, rel_h=.99):
  d   = np.abs(arr/arr.max())
  pks = sps.find_peaks(d, height=h, prominence=prom)[0]
  wid = sps.peak_widths(d, pks, rel_height=rel_h)
  res = [(pks[i], round(wid[2][i]), round(wid[3][i])) for i in range(len(pks))]
  return res

def id_interfaces_raw(df):
  # get steps in data
  interfaces = get_ilist(df)
  dGrho, dGu, dGp = df_get_gaussfprim(df, sigma=1)
  ilist  = []

  # id interfaces 
  for item in interfaces:
    i, m, l = item
    if m == 1:
      if item == interfaces[0] or item == interfaces[-1]:
        # if we're near the box limits, it's the shell interface
        interface = ("shell", i, 0, 0)
      else:
        # either a CD or the onset of a wave
        lenL, lenR = get_lenInterface(dGrho, dGu, dGp, i)


    lenL, lenR = get_lenInterface(dGrho, dGu, dGp, i)
    if ((lenL <= 2.5*lenR) and (lenR <= 2.5*lenL)):
      if m == 3:
        interface = ("shock", i, lenL, lenR)
      else:
        interface = ("CD", i, lenL, lenR)
    else:
      interface = ("waveOn", i, lenL, lenR)
    ilist.append(interface)
  

  # avoid mistakes when interfaces are too close together
  #k = 0
  #print(len(ilist))
  #while k < len(ilist)-1:
  #  type, i, iL, iR = ilist[k]
  #  inext = ilist[k+1][1]
  #  if i + iR > inext:
  #    ilist[k] = (type, i, iL, inext-i)
  #  k += 1
  #while k > 0:
  #  type, i, iL, iR = ilist[k]
  #  iprev = ilist[k-1][1]
  #  if i - iL < inext:
  #    ilist[k] = (type, i, i-inext, iR)
  #  k -= 1

  return ilist

def correct_waves(ilist):
  # correct for waves
  k = 0
  while k<=len(ilist):
    type, i, lenL, lenR = ilist[k]
    if type == 'waveOn':
      if lenL > lenR: # wave goes to the left
        del ilist[k+1]
        ilist.insert(k-1, ('waveS', i-lenL))
      elif lenR > lenL: # wave goes to the right
        ilist.insert(k+1, ('waveS', i+lenR))
        del ilist[k-1]
        k += 1 # no need to check interface just created
    k += 1
  
  return ilist

def id_interfaces_v4(df):
  ilist = id_interfaces_raw(df)
  ilist = correct_waves(ilist)
  return ilist

def zoneID(df):
  ilist = id_interfaces_sps_cleaned(df)
  zone  = np.zeros(df['p'].to_numpy().shape)
  ids   = [interface[1] for interface in ilist]
  l     = len(ilist)   

  if df.attrs['mode'] == 'shells':
    if l == 3:
      # only case: t = t0
      zvals = [5., 4., 1., 0.]
    elif l == 4:
      # when a shock reaches shell edge
      if [inter[0] for inter in ilist[:2]] == ['shock', 'CD']:
        # RS crossing
        zvals = [5., 3., 2., 1., 0.]
      else:
        # FS crossing
        zvals = [5., 4., 3., 2., 0.]
    else:
      zvals = [5., 4., 3., 2., 1., 0.]
  
  zones = np.split(zone, ids)
  for z, zval in zip(zones, zvals):
    z += zval
  zone = np.concatenate(zones)
  df2 = df.assign(zone=zone)

  return df2
    

def zoneID_old2(df):
  p      = df['p'].to_numpy()
  zone   = np.zeros(p.shape)
  shocks = np.zeros(p.shape)
  ilist, shlist = id_interfaces_v2(df)
  
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

def get_steps(arr, sigma=1, h=0.02, takelog=True):

  '''
  Get step up in data by convoluting with step function
  '''
  if takelog:
    # takes log for easier jump detection on low values
    # makes sure only positive values before applying filter
    arr = np.log10(arr)
    arr = arr - min(0., arr.min()) + 0.1
  d = step_gaussfilter(arr, sigma)
  
  # get peaks
  steps_up   = sps.find_peaks(d, height=h)[0]
  steps_down = sps.find_peaks(-d, height=h)[0]
  steps = np.concatenate((steps_up, steps_down))

  return np.sort(steps)

def step_gaussfilter(arr, sigma=3):
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
  zone = df['zone'].to_numpy()
  if np.argwhere(zone==n).any():
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

  else:
    res = 0.
  
  return res


# Physical functions from dataframe
def df_get_T(df):
  rho  = df["rho"].to_numpy(copy=True)
  p    = df["p"].to_numpy(copy=True)
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