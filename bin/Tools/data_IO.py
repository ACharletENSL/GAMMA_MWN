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
from phys_radiation import *
from phys_constants import *
from phys_edistrib import *
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
  data.attrs['key'] = key
  env = MyEnv(key)
  data.attrs['rhoscale'] = env.rhoscale
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

def openData_withDistrib(key, it):
  if it == 0.:
    print("Open later data")
    df = openData_withZone(key, it)
    df = df.assign(gm=0., gb=0., gM=0., fc=False, inj=False, dt=0.)
  else:
    its = dataList(key)
    i_it= its.index(it)
    it0 = its[i_it-1]
    df0 = openData_withZone(key, it0)
    df  = openData_withZone(key, it)
    shells = (df['trac']>0.5)
    varlist = ['gmin', 'gmax', 'trac2']
    gmin, gmax, trac = df.loc[shells][varlist].to_numpy().transpose()
    gmin0, gmax0, trac0 = df0.loc[shells][varlist].to_numpy().transpose()

    fc  = (gmax < gmin0) & (gmax > 1.1)
    inj = ((trac - trac0 > 0.5))
    #sc  = (gmax < gmax0) & (gmax > gmin0)
    #inj = ((gmax/gmax0 > 10.) | (gmin/gmin0 < 10.))

    # first add cooling rules
    gm = gmin
    gb = np.where(fc, gmin0, gmax)
    gM = gmax0
    dt = df['t'] - df0['t']

    # injection really happens a bit earlier
    # we need to consider theoretical gmin, gmax at shock
    # the corresponding dt will not be the full time elapsed between 
    if inj.any():
      id_inj = np.argwhere(inj)[:, 0]
      env = MyEnv(df.attrs['key'])
      ish = shells[shells==True].index.min()
      for i in id_inj:
        k = i+ish
        cell = df.iloc[k]
        gmin_th, gmax_th = cell_derive_edistrib(cell, env)
        gm[i] = gmin_th
        gb[i] = gmin_th
        gM[i] = gmax_th
        dt[k] /= 2. 
        # better: check on the rate of trac2 decay and determine dt since trac2=1
    
    # add to dataframe
    #for arr in [gm, gb, gM, fc, inj]:
    #  arr = resize_shell2sim(df, arr)
    gm = resize_shell2sim(df, gm)
    gb = resize_shell2sim(df, gb)
    gM = resize_shell2sim(df, gM)
    fc = resize_shell2sim(df, fc)
    inj= resize_shell2sim(df, inj)
    df = df.assign(gm=gm, gb=gb, gM=gM, fc=fc, inj=inj, dt=dt)

  return df


def resize_shell2sim(df, arr):
  '''
  Resize 'shell-sized' array into a full sim one
  '''
  trac = df['trac'].to_numpy()
  out = np.zeros(trac.shape)
  out[trac>0.5] = arr
  return out

def open_cell(df, i):
  '''
  returns the i-th cell starting from the min index of shell 4
  '''
  trac = df['trac'].to_numpy()
  i4  = np.argwhere((trac > 0.99) & (trac < 1.01))[:,0].min()
  return df.iloc[i4+i]

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

def open_raddata(key):

  '''
  returns a pandas dataframe with the extracted spectral data
  '''
  rfile_path, rfile_bool = get_radfile(key)
  if rfile_bool:
    df = pd.read_csv(rfile_path, index_col=0)
    return df
  else:
    return rfile_bool

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
  imin = 0
  imax = -1
  if itmin:
    imin = its.index(itmin)
  if itmax:
    imax = its.index(itmax)
  
  return its[imin:imax]

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

def get_radfile(key):

  '''
  Returns path of file with  spectras over the run and boolean for its existence
  '''

  dir_path = GAMMA_dir + '/results/%s/' % (key)
  #file_path = dir_path + "extracted_data.txt"
  file_path = dir_path + 'run_observed.csv'
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
def df_get_rads(df, nuobs, Tobs, dTobs, env, func='analytic'):
  '''
  Get contributions to observed flux, separating the (former) two shells
  '''

  rad_RS = np.zeros((len(Tobs), len(nuobs)))
  rad_FS = np.zeros((len(Tobs), len(nuobs)))
  if func == 'analytic':
    RS, FS = df_get_cellsBehindShock(df)
    if not RS.empty:
      rad_RS += cell_flux(RS, nuobs, Tobs, dTobs, env, func)
    if not FS.empty:
      rad_FS += cell_flux(FS, nuobs, Tobs, dTobs, env, func)
  else:
    rad_RS = df_get_rad_contribs(df, nuobs, Tobs, dTobs, env, 'RS', func)
    rad_FS = df_get_rad_contribs(df, nuobs, Tobs, dTobs, env, 'FS', func)
  return rad_RS, rad_FS

def df_get_rad_contribs(df, nuobs, Tobs, dTobs, env, contrib='all', func='analytic'):
  '''
  Get contributions from a data file to observed flux, selecting a specific shell if needed
  df need to come from openData_withDistrib
  '''

  out = np.zeros((len(Tobs), len(nuobs)))
  if contrib in ['RS', 'FS']:
    sub = '_' + contrib
  else:
    sub = ''
  if func == 'analytic':
    RS, FS = df_get_cellsBehindShock(df)
    if contrib == 'RS':
      if not RS.empty:
        out += cell_flux(RS, nuobs, Tobs, dTobs, env, func)
    elif contrib == 'FS':
      if not FS.empty:
        out += cell_flux(FS, nuobs, Tobs, dTobs, env, func)
    else:
      for cell in [RS, FS]:
        out += cell_flux(cell, nuobs, Tobs, dTobs, env, func)
  else:
    sh = df.loc[(df['trac'] > 0.) & (df['gmax'] > 1.)]
    if contrib == 'RS':
      sh = df.loc[(df['trac'] > 0.99) & (df['trac'] < 1.01) & (df['gmax'] > 1.)]
    elif contrib == 'FS':
      sh = df.loc[(df['trac'] > 1.99) & (df['trac'] < 2.01) & (df['gmax'] > 1.)]
    for i in sh.index:
      cell = df.iloc[i]
      out += cell_flux(cell, nuobs, Tobs, dTobs, env, func)
  return out
      

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

def df_get_crosspdV(df, n, front=True):
  '''
  Returns total pdV work rate across shell n external surface, scaled
  '''
  r, v1, p1, v2, p2 = df_get_var_interface(df, n, ['vx', 'p'], front)
  A = 1.
  if df.attrs['geometry'] == 'spherical':
    A = r**2
  rate_in = pdV_rate(v2, p2, A) - pdV_rate(v1, p1, A)
  return rate_in


def df_get_var_interface(df, n, varlist=['rho', 'vx', 'p'], front=True, ns=3):
  '''
  Return value of hydro variables right inside and outside (front or back) a given shell n
  time, vars inside, vars outside
  '''
  S = df.loc[df['zone']==n]
  l = len(varlist)
  res = np.zeros(1+2*l)

  if front:
    iF = S.index.max() + 1
    res[0] = df['x'].iloc[iF]
    for k, var in enumerate(varlist):
      res[k+1]   = df[var].iloc[iF-ns:iF].mean()
      res[k+1+l] = df[var].iloc[iF:iF+ns].mean()
    return res
  else:
    iB = S.index.min()
    res[0] = df['x'].iloc[iB-1]
    for k, var in enumerate(varlist):
      res[k+1] = df[var].iloc[iB-ns:iB].mean()
      res[k+1+l] = df[var].iloc[iB:iB+ns].mean()
    return res


def df_get_pdV(df, n, front=True):
  '''
  Returns pdV work rate given to shell n external surface, scaled
  '''

  r, v, p = df_get_var_ext(df, n, front, ["vx", "p"])
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

def df_get_cellsBehindShock(df, n=2):
  '''
  Return a cell object with radiative values & position taken at shock front
  and hydro values taken downstream (n cells after shock)
  '''
  env = MyEnv(df.attrs['key'])
  hdvars = ['dx', 'rho', 'vx', 'p', 'D', 'sx', 'tau']
  anvars = ['rho', 'vx', 'p', 'gb']
  anvalsRS = [env.rho3/env.rhoscale, env.beta, env.p_sh/(env.rhoscale*c_**2), env.gma_m]
  anvalsFS = [env.rho2/env.rhoscale, env.beta, env.p_sh/(env.rhoscale*c_**2), env.gma_mFS]
  iCD = df.loc[(df['trac'] > 0.99) & (df['trac'] < 1.01)].index.max()
  RS, FS = df_to_shocks(df)
  lRS, lFS = len(RS), len(FS)
  if RS.empty:
    RSfront = RS
  else:
    iRS = RS.index.min()
    iRSd = min(iRS+lRS+n, iCD)
    RSfront = df.iloc[iRS].copy()
    RSdown = df.iloc[iRSd]
    RSfront[hdvars] = RSdown[hdvars]
    RSfront[anvars] = anvalsRS
  if FS.empty:
    FSfront = FS
  else:
    iFS = FS.index.min()
    iFSd = max(iFS-lRS-n, iCD+1)
    FSfront = df.iloc[iFS].copy()
    FSdown = df.iloc[iFSd]
    FSfront[hdvars] = FSdown[hdvars]
    FSfront[anvars] = anvalsFS
  return RSfront, FSfront

# get variables
cool_vars = ["V4", "dt", "inj", "fc", "gm", "gb", "gM",
  "num", "nub", "nuM", "Lp", "L", "V4r", "tc"]
var2func = {
  "T":(derive_temperature, ['rho', 'p'], []),
  "h":(derive_enthalpy, ['rho', 'p'], []),
  "lfac":(derive_Lorentz, ['vx'], []),
  "u":(derive_proper, ['vx'], []),
  "Ek":(derive_Ekin, ['rho', 'vx'], []),
  "Ei":(derive_Eint, ['rho', 'p'], []),
  "ei":(derive_Eint_comoving, ['rho', 'p'], ['rhoscale']),
  "B":(derive_B_comoving, ['rho', 'p'], ['rhoscale', 'eps_B']),
  "res":(derive_resolution, ['x', 'dx'], []),
  "numax":(derive_nu, ['rho', 'p', 'gmax'], ['rhoscale', 'eps_B']),
  "numin":(derive_nu, ['rho', 'p', 'gmin'], ['rhoscale', 'eps_B']),
  "num":(derive_nu, ['rho', 'p', 'gm'], ['rhoscale', 'eps_B']),
  "nub":(derive_nu, ['rho', 'p', 'gb'], ['rhoscale', 'eps_B']),
  "nuM":(derive_nu, ['rho', 'p', 'gM'], ['rhoscale', 'eps_B']),
  "Pmax":(derive_max_emissivity, ['rho', 'p', 'gmin', 'gmax'], ['rhoscale', 'eps_B', 'psyn', 'xi_e']),
  "Tth":(derive_normTime, ['x', 'vx'], []),
  "Lp":(derive_localLum, ['rho', 'vx', 'p', 'x', 'dx', 'dt', 'gmin', 'gmax', 'gb'],
    ['rhoscale', 'eps_B', 'psyn', 'xi_e', 'R0', 'geometry']),
  "L":(derive_obsLum, ['rho', 'vx', 'p', 'x', 'dx', 'dt', 'gmin', 'gmax', 'gb'], 
    ['rhoscale', 'eps_B', 'psyn', 'xi_e', 'R0', 'geometry']),
  "n":(derive_n, ['rho'], ['rhoscale']),
  "Pfac":(derive_emiss_prefac, [], ['psyn']),
  "obsT":(derive_obsTimes, ['t', 'x', 'vx'], ['t0', 'z']),
  "T0":(derive_obsNormTime, ['t', 'x', 'vx'], ['t0', 'z']),
  "Tej":(derive_obsEjTime, ['t', 'x', 'vx'], ['t0', 'z']),
  "Tpk":(derive_obsPkTime, ['t', 'x', 'vx'], ['t0', 'z']),
  "V3":(derive_3volume, ['x', 'dx'], ['R0', 'geometry']),
  "V4":(derive_4volume, ['x', 'dx', 'dt'], ['R0', 'geometry']),
  "V4r":(derive_rad4volume, ['x', 'dx', 'rho', 'vx', 'p', 'dt', 'gb'],
    ['rhoscale', 'eps_B', 'R0', 'geometry']),
  "tc":(derive_tcool, ['rho', 'vx', 'p', 'gb'], ['rhoscale', 'eps_B'])
}

def get_vars(df, varlist):
  '''
  Return array of variables
  '''
  if type(df) == pd.core.series.Series:
    out = np.zeros((len(varlist)))
  else:
    out = np.zeros((len(varlist), len(df)))
  for i, var in enumerate(varlist):
    out[i] = get_variable(df, var)
  return out

def get_variable(df, var):
  '''
  Returns chosen variable
  '''
  if var in df.keys():
    try:
      res = df[var].to_numpy(copy=True)
    except AttributeError:
      res = df[var]
  else:
    func, inlist, envlist = var2func[var]
    res = df_get_var(df, func, inlist, envlist)
  return res

def df_get_var(df, func, inlist, envlist):
  params, envParams = get_params(df, inlist, envlist)
  res = func(*params, *envParams)
  return res

def get_params(df, inlist, envlist):
  if (type(df) == pd.core.series.Series):
    params = np.empty(len(inlist))
  else:
    params = np.empty((len(inlist), len(df)))
  for i, var in enumerate(inlist):
    if var in df.attrs:
      params[i] = df.attrs[var].copy()
    else:
      try:
        params[i] = df[var].to_numpy(copy=True)
      except AttributeError:
        params[i] = df[var]
  envParams = []
  if len(envlist) > 0.:
    env = MyEnv(df.attrs['key'])
    envParams = [getattr(env, name) for name in envlist]
  return params, envParams

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

  #ilist = get_fused_interf(df)
  # RS = df.loc[(df['Sd'] == 1.0) & (df['trac'] > 0.99) & (df['trac'] < 1.01)]
  # FS = df.loc[(df['Sd'] == 1.0) & (df['trac'] > 1.99) & (df['trac'] < 2.01)]
  # RSlist = np.split(RS, np.flatnonzero(np.diff(RS.index) != 1) + 1)
  # FSlist = np.split(FS, np.flatnonzero(np.diff(FS.index) != 1) + 1)
  # clnRS = []
  # for sh in RSlist:
  #   for inter in ilist:
  #     if intersect([sh.index.min(), sh.index.max()], [inter[1], inter[2]]):
  #       clnRS.append(sh)
  #       break
  # RSlist = clnRS
  # clnFS = []
  # for sh in FSlist:
  #   for inter in ilist:
  #     if intersect([sh.index.min(), sh.index.max()], [inter[1], inter[2]]):
  #       clnFS.append(sh)
  #       break
  # FSlist = clnFS
  # shocks = pd.concat([RS, FS], axis=0)
  # shlist = RSlist + FSlist
  
  out = []
  S4 = df.loc[(df['trac'] > 0.99) & (df['trac'] < 1.01)]
  S1 = df.loc[(df['trac'] > 1.99) & (df['trac'] < 2.01)]
  i4b = S4.index.min()
  icd = S4.index.max()
  i1f = S1.index.max()
  RSsh = S4.loc[df['Sd']==1.]
  RSlist = np.split(RSsh, np.flatnonzero(np.diff(RSsh.index) != 1) + 1)
  FSsh = S1.loc[df['Sd']==1.]
  FSlist = np.split(FSsh, np.flatnonzero(np.diff(FSsh.index) != 1) + 1)
  for (front, ilims, shlist) in zip(['RS', 'FS'], [(i4b, icd), (icd, i1f)], [RSlist, FSlist]):
    iL, iR = ilims
    crossedSh = [sh for sh in shlist if ((sh.index.max()<iL) | (sh.index.min()>iR))]
    if (len(crossedSh) > 0) or (len(shlist)== 0):
        out.append(pd.DataFrame(columns=df.keys()))
    elif len(shlist) == 1:
      out.append(shlist[0])
    else:
      fronts = [sh for sh in shlist if ((sh.index.min()>=iL) & (sh.index.max()<=iR))]
      if front == RS:
        out.append(shlist[0])
      else:
        out.append(shlist[-1])
  return out



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
    RScr = True
    FScr = True
  else:
    RS, FS = df_to_shocks(df) 
    try:
      id_RS  = min(RS.index)
      rRS    = r[id_RS]
      zone[(zone == 4.) & (r >= rRS)] = 3.
    except ValueError: # no shock inside shell 4
      if min(shocks.index) < i_4.min():
        # shock exists and has crossed
        RScr = True
        zone[zone == 4.] = 3.

    try:
      id_FS  = max(FS.index)
      rFS    = r[id_FS]
      zone[(zone == 1.) & (r <= rFS)] = 2.
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
      zone[i4:iw] = 4.
      zone[iw:i_4.max()] = 3.
    if FScr:
      i1   = i_1.max()
      wave = [item for item in ilist if i1>item[1]][-1]
      iw   = wave[1]
      zone[iw:i1] = 1.
      zone[i_1.min():iw] = 2.
  
  df2 = df.assign(zone=zone)
  return df2

def get_fused_interf(df, h=.05, prom=.05, rel_h=.999):
  int_rho, int_u, int_p = get_allinterf(df, h, prom, rel_h)
  ilist = []

  # clean potential duplicates
  n = 0
  while n < len(int_rho)-1:
    i, iL, iR = int_rho[n]
    j, jL, jR = int_rho[n+1]
    if (i>=jL and i<=jR) or (j>=iL and j<=iR):
      k = int(np.ceil((i+j)/2.))
      kL = min(iL, jL)
      kR = max(iR, jR)
      del int_rho[n+1]
      del int_rho[n]
      int_rho.insert(n, (k, kL, kR))
    n += 1

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
  #rho       = gaussian_filter1d(rho, sigma=1, order=0)
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
  rho, u, p = get_vars(df, ['rho', 'u', 'p'])
  return rho, u, p


