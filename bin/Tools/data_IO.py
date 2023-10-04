# -*- coding: utf-8 -*-
# @Author: acharlet

'''
This file contains IO functions:
find informations on the run, find a specific data file
open data file, get relevant data
'''

# Imports
# --------------------------------------------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import scipy.signal as sps
from scipy.ndimage import gaussian_filter1d
from phys_functions import *
from phys_constants import c_, pi_
from environment import MyEnv

cwd = os.getcwd().split('/')
iG = [i for i, s in enumerate(cwd) if 'GAMMA' in s][0]
GAMMA_dir = '/'.join(cwd[:iG+1])

# open data and run informations
# --------------------------------------------------------------------------------------------------
def openData(key, it=None, sequence=True):
  if sequence:
    filename = GAMMA_dir + '/results/%s/phys%010d.out' % (key, it)
  elif it is None:
    filename = GAMMA_dir + '/results/%s' % (key)
  else:
    filename = GAMMA_dir + '/results/%s%d.out' % (key, it)
  data = pd.read_csv(filename, sep=" ")
  mode, runname, rhoNorm, geometry = get_runatts(key)
  data.attrs['mode'] = mode
  data.attrs['runname'] = runname
  data.attrs['rhoNorm'] = rhoNorm
  data.attrs['geometry']= geometry
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

  '''
  Returns attributes of the selected run
  '''

  physpath  = get_physfile(key)
  attrs = ('mode', 'runname', 'rhoNorm', 'geometry')
  with open(physpath, 'r') as f:
    lines = f.read().splitlines()
    lines = filter(lambda x: x.startswith(attrs) , lines)
    for line in lines:
      name, value = line.split()[:2]
      if name == 'mode':
        mode = value
      elif name == 'runname':
        runname = value
      elif name == 'rhoNorm':
        rhoNorm = value
      elif name == 'geometry':
        geometry = value
  return mode, runname, rhoNorm, geometry

def open_rundata(key):

  '''
  returns a pandas dataframe with the extracted data
  '''
  dfile_path, dfile_bool = get_runfile(key)
  mode, runname, rhoNorm, geometry = get_runatts(key)
  if dfile_bool:
    df = pd.read_csv(dfile_path, index_col=0)
    df.attrs['mode']    = mode
    df.attrs['runname'] = runname 
    df.attrs['rhoNorm'] = rhoNorm
    return df
  else:
    return dfile_bool

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

def get_varlist(var, Nz, mode):
  '''
  Creates list of variables names with zone / interface subscripts
  ''' 
  interfaces_names = {
    'PWN':['{ts}', 'b', '{sh}', '{rs}', '{cd}', '{fs}'],
    'shells':['{4b}', '{rs}', '{cd}', '{fs}', '{1f}']
    }
  zone_names = {
    'PWN':['w', 'b', '{sh}', '{ej}', '{sh.ej}', '{sh.csm}', '{csm}'],
    'shells':['4', '3', '2', '1']
    }

  varlist = []
  zvarsList = ['V', 'M', 'Ek', 'Ei']
  ivarsList = ['R', 'v']
  zsubList  = zone_names[mode]
  intList   = interfaces_names[mode]


  # construct shocks list
  if mode == 'shells':
    shList = ['{rs}', '{fs}']
  else:
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
    if mode == 'shells':
      if var == 'v':
        varlist = [var + '_' + sub for sub in intList[1:Nz-1]]
      elif var == 'R':
        varlist = [var + '_' + sub for sub in intList]
    else:
      varlist = [var + '_' + sub for sub in intList[:Nz-1]]
  elif var == 'u':
    varlist = [var]
  elif var == 'rho':
    varlist = ['rho_3', 'rho_2']
  elif var == 'pdV':
    varlist = ['pdV_4', 'pdV_1']
  else:
    print("Asked variable does not exist")
  
  return varlist

# analyze dataframe
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
  allvars = ['Nc', 'R', 'u', 'rho', 'ShSt', 'V', 'M', 'Ek', 'Ei', 'pdV']
  allres  = []
  for var in allvars:
    Nvar = len(get_varlist(var, Nz, mode))
    res = []
    if var == 'Nc':
      res = [zone[zone==n].count() for n in reversed(range(Nz))]
    elif var == 'R':
      zones = [4, 3, 2, 1, 0] if mode == 'shells' else [n+1 for n in range(Nz-1)]
      res = [get_radius_zone(df, n)*c_ for n in zones]
    elif var == 'u':
      res = [df_get_shocked_u(df)]
    elif var == 'rho':
      res = df_get_rhoCD(df)
    elif var == 'ShSt':
      res = get_shocksStrength(df)
    elif var == 'pdV':
      res = [df_get_pdV(df, 4, False), df_get_pdV(df, 1, True)]
    else:
      zones = [4., 3., 2., 1.] if mode == 'shells' else [n for n in range(1, Nvar+1)]
      res = [zone_get_zoneIntegrated(df, var, n) for n in zones]
    allres.append(res)
  
  return allres

def get_shocksStrength(df):
  if df.attrs['mode'] == 'shells':
    RS, FS = df_to_shocks(df)
    try:
      iu_RS = min(RS.index) - 1
      id_RS = max(RS.index) + 1
      shst_RS = df_get_shockStrength(df, iu_RS, id_RS)
    except ValueError:
      shst_RS = 0.
    try:
      iu_FS = max(FS.index) + 1
      id_FS = min(FS.index) - 1
      shst_FS = df_get_shockStrength(df, iu_FS, id_FS)
    except ValueError:
      shst_FS = 0.
    return [shst_RS, shst_FS]
  else:
    print("Implement relative Mach number")
    return [0., 0.]

def df_get_shockStrength(df, i_u, i_d, n=5):
  '''
  Derives the shock strength by comparing hydrodynamical properties
  upstream (at i_u) and downstream (at i_d), sampling of n points on each side
  '''
  iL = min(i_u, i_d)
  iR = max(i_u, i_d)
  bL = df['vx'].iloc[iL-n:iL].mean()
  gL = derive_Lorentz(bL)
  bR = df['vx'].loc[iR:iR+n].mean()
  gR = derive_Lorentz(bR)
  relg = gL*gR* (1. - bL*bR)
  return relg - 1.

def df_get_shocked_u(df):
  '''
  Get the proper velocity in the shocked regions
  '''

  zone = df['zone'].to_numpy()
  u    = df_get_u(df)
  u2   = u[zone==2.]
  u3   = u[zone==3.]
  if df.attrs['geometry'] == 'spherical':
    # get value at CD in spherical case
    u_sh = np.concatenate((u3[-3:], u2[:3]))
  else:
    u_sh = np.concatenate((u3[3:], u2[:-3]))
  if u_sh.size > 0:
    res = u_sh.mean()
  else:
    res = 0.
  return res

def df_get_rhoCD(df, n=3, m=5):
  '''
  Get the proper density ratio at CD
  mean over a sample size n, separated by m cells of the interface
  '''
  zone = df['zone'].to_numpy()
  rho  = df['rho'].to_numpy()
  
  # choose cells near CD
  rho3 = rho[zone==3.][-(n+m):-m]
  rho2 = rho[zone==2.][m:m+n]

  #check if exists
  if rho2.size > 0 and rho3.size > 0:
    return rho3.mean(), rho2.mean()
  else:
    rho4 = rho[zone==4.][-(n+m):-m]
    rho1 = rho[zone==1.][m:m+n]
    return rho4.mean(), rho1.mean()

def df_get_shocked_p(df):
  '''
  Get the average pressure in the shocked regions
  '''

  zone = df['zone'].to_numpy()
  p    = df['p'].to_numpy()
  p2   = p[zone==2.]
  p3   = p[zone==3.]
  p_sh = np.concatenate((p3[3:], p2[:-3]))
  if p_sh.size > 0:
    res = p_sh.mean()
  else:
    res = 0.
  return res

def df_get_pdV(df, n, front=True):
  '''
  Returns pdV work rate across shell n external surface, divided by surface area
  '''

  r, rho, v, p = df_get_var_ext(df, n, front)
  A = 1.
  if df.attrs['geometry'] == 'spherical':
    A = r**2
  return pdV_rate(v, p, A)


def df_get_var_ext(df, n, front=True, varlist=['rho', 'vx', 'p'], ns=5):
  '''
  Return value of hydro variables right outside (front or back) a given shell
  '''
  S = df.loc[df['zone']==n]
  res = np.zeros(len(varlist)+1)
  if front:
    iF = S.index.max() + 1
    res[0] = df['x'].iloc[iF]
    for k, var in enumerate(varlist):
      res[k+1] = df[var].iloc[iF+ns].mean()
    return res
  else:
    iB = S.index.min() - 1
    res[0] = df['x'].iloc[iB]
    for k, var in enumerate(varlist):
      res[k+1] = df[var].iloc[iB-ns:iB].mean()
    return res


def df_get_mean(df, var, n):
  '''
  Get the mean value of the selected variable in a chosen region
  '''

  res = df.loc[df['zone']==n][var][4:-4].mean()
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
  '''

  if 'zone' not in df.columns:
    df = zoneID(df)

  mode = df.attrs['mode']
  r = df['x']
  z = df['zone']
  if mode=="shells" and n==0:
    rz = r[z==1].max()
  else:
    rz = r[z == n].min()
  while(np.isnan(rz)):
    # if no cell of the zone, take max/min of previous one
    if mode == 'shells':
      n -= 1
      rz = r[z==n].min()
    else:
      n += 1
      rz = r[z==n].min()
  return rz

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

def zone_get_zoneIntegrated(df, var, n):
  '''
  Zone integrated var for zone of index n
  Possible vars: volume V, mass M, kinetic and internal energies Ekin, Eint
  '''

  zone = df['zone'].to_numpy()
  if np.argwhere(zone==n).any():
    r = df['x'].to_numpy()*c_
    dr = df['dx'].to_numpy()*c_
    if df.attrs['geometry'] == 'spherical':
      dV = 4.*pi_*r**2*dr
    else:
      dV = dr
    res = 0.
    if var == 'V':
      data = dV
    elif var == 'M':
      data = df['D'].to_numpy() * dV
    else:
      r, data = get_variable(df, var)
      data *= dV
    res = data[zone==n].sum()
  else:
    res = 0.
  
  return res


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
    "Ek":df_get_Ekin,
    "Ei":df_get_Eint,
    "dt":df_get_dt,
    "res":df_get_res
  }

  r = df["x"].to_numpy()
  if var in calc_vars:
    out = calc_vars[var](df)
    return r, out
  elif var in df.columns:
    out = df[var].to_numpy()
    return r, out
  else:
    print("Required variable doesn't exist.")

# get variables 
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
  v = df["vx"].to_numpy(copy=True)
  p = df["p"].to_numpy(copy=True)
  return derive_Eint(rho, v, p)

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

def intersect(a, b):
  '''
  check if segments [aL, aR] and [bL, bR] intersect
  '''
  aL, aR = a
  bL, bR = b
  return bool(aL in range(bL, bR+1) or aR in range(bL, bR+1))

def df_to_shocklist(df):
  '''
  Returns indices of shock-identified regions in data
  '''
  ilist = get_fused_interf(df)
  shocks = df.loc[(df['Sd'] == 1.0)]
  shlist = np.split(shocks, np.flatnonzero(np.diff(shocks.index) != 1) + 1)
  clnsh  = [sh for inter in ilist for sh in shlist 
      if intersect([sh.index.min(), sh.index.max()], [inter[1], inter[2]])]
      # need to clean better to account for overlaps in peaks
  idlist = [[sh.index.min(), sh.index.max()] for sh in clnsh]
  return idlist


def df_to_shocks(df):
  '''
  Extract the shocked part of data
  Cleans 'false' shock at wave onset when resolution is not high enough
  '''

  ilist = get_fused_interf(df)
  shocks = df.loc[(df['Sd'] == 1.0)]
  shlist = np.split(shocks, np.flatnonzero(np.diff(shocks.index) != 1) + 1)
  clnsh  = [sh for inter in ilist for sh in shlist 
      if intersect([sh.index.min(), sh.index.max()], [inter[1], inter[2]])]

  if len(clnsh) == 2:
    RS = clnsh[0]
    FS = clnsh[1]
    RS = RS.loc[(df['trac'] > 0.99) & (df['trac'] < 1.01)]
    FS = FS.loc[(df['trac'] > 1.99) & (df['trac'] < 2.01)]
    return RS, FS
  elif len(clnsh) > 2:
    S4 = df.loc[(df['trac'] > 0.99) & (df['trac'] < 1.01)]
    S1 = df.loc[(df['trac'] > 1.99) & (df['trac'] < 2.01)]
    icd = S1.index.min()
    left = [sh for sh in clnsh if sh.index.max() < icd]
    right = [sh for sh in clnsh if sh.index.min() > icd]
    RS = left[0]
    FS = right[-1]
    RS = RS.loc[(df['trac'] > 0.99) & (df['trac'] < 1.01)]
    FS = FS.loc[(df['trac'] > 1.99) & (df['trac'] < 2.01)]
    return RS, FS
  else:
    RS = shocks.loc[(df['trac'] > 0.99) & (df['trac'] < 1.01)]
    FS = shocks.loc[(df['trac'] > 1.99) & (df['trac'] < 2.01)]
    return RS, FS

def zoneID(df):
  '''
  Detect zones from passive tracer and shocks
  /!\ MUST BE ON DATA PRODUCED WITH SHOCK DETECTION /!\
  '''
  r    = df['x'].to_numpy()
  trac = df['trac'].to_numpy()
  Sd   = df['Sd'].to_numpy()
  zone = np.zeros(trac.shape)
  RScr = False
  FScr = False

  # use tracer to identify shells 1 and 4
  i_4  = np.argwhere((trac > 0.99) & (trac < 1.01))[:,0]
  i_1  = np.argwhere((trac > 1.99) & (trac < 2.01))[:,0]
  zone[i_4] = 4.
  zone[i_1] = 1.
  #zone[:i_4.min()] = 6.

  # use GAMMA-detected shocks
  shocks = df.loc[(df['Sd'] == 1.0)]
  if shocks.empty:
    pass
  else:
    RS, FS = df_to_shocks(df) 
    try:
      id_RS  = max(RS.index)
      rRS    = r[id_RS]
      zone[(zone == 4.) & (r > rRS)] = 3.
    except ValueError: # no shock inside shell 4
      if min(shocks.index) < i_4.min():
        # shock exists and has crossed
        RScr = True
        zone[zone == 4.] = 3.

    try:
      id_FS  = min(FS.index)
      rFS    = r[id_FS]
      zone[(zone == 1.) & (r < rFS)] = 2.
    except ValueError: # no shock inside shell 1
      if max(shocks.index) > i_1.max():
        # shock exists and has crossed
        FScr = True
        zone[zone == 1.] = 2. 
    
    # case where no shocks propagating in the shells, use own algorithm
    if RScr or FScr:
      ilist  = get_fused_interf(df)
      if RScr:
        i4   = i_4.min()
        wave = [item for item in ilist if i4<item[2]][0]
        iw   = wave[2]
        zone[i_4.min():iw+1] = 4.
      if FScr:
        i1   = i_1.max()
        wave = [item for item in ilist if i1>item[1]][-1]
        iw   = wave[1]
        zone[iw:i_1.max()+1] = 1.
  
  df2 = df.assign(zone=zone)
  return df2

def get_fused_interf(df, h=.05, prom=.05, rel_h=.999):
  int_rho, int_u, int_p = get_allinterf(df, h, prom, rel_h)
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

def df_id_shocks(df):
  '''
  Returns indices of shock fronts
  '''
  idlist = []
  shocks = df.loc[(df['Sd'] == 1.0)]
  shlist = np.split(shocks, np.flatnonzero(np.diff(shocks.index) != 1) + 1)
  for sh in shlist:
    idlist.append([sh.index.min(), sh.index.max()])
  return idlist



def get_allinterf(df, h=.05, prom=.05, rel_h=.99):
  rho, u, p = df_get_primvar(df)
  # filtering rho because data can be noisy
  rho       = gaussian_filter1d(rho, sigma=1, order=0)
  int_rho   = get_varinterf(rho, True, h, prom, rel_h)
  int_u     = get_varinterf(u, False, h, prom, rel_h)
  int_p     = get_varinterf(p, True, h, prom, rel_h)
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

def df_get_primvar(df):
  rho = df['rho'].to_numpy()
  u   = df_get_u(df)
  p   = df['p'].to_numpy()
  return rho, u, p
