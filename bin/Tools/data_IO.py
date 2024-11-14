# -*- coding: utf-8 -*-
# @Author: acharlet

'''
This file contains IO functions:
find informations on the run, find a specific data file
open data file, get relevant data
Functions that are pure transformation of dataframes were moved to df_funcs
'''

# Imports
# --------------------------------------------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import scipy.signal as sps
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import make_interp_spline
from df_funcs import *
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
  data.attrs['it'] = it
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
  its = dataList(key)
  if it == its[0]:
    it_p = its[1]
    df  = openData_withZone(key, it)
    dfp = openData_withZone(key, itp)
    dt  = dfp['t'] - df['t'] 
    df  = df.assign(gm=0., gb=0., gM=0., fc=False, inj=False, dt=0.)
  else:
    i_it= its.index(it)
    itm = its[i_it-1]
    dfm = openData_withZone(key, itm)
    df  = openData_withZone(key, it)
    shells = (df['trac']>0.5)
    varlist = ['gmin', 'gmax', 'trac2']
    gmin, gmax, trac = df.loc[shells][varlist].to_numpy().transpose()
    gminm, gmaxm, tracm = dfm.loc[shells][varlist].to_numpy().transpose()

    fc  = (gmax < gminm) & (gmax > 1.1)
    inj = ((trac - tracm > 0.5))
    #sc  = (gmax < gmaxm) & (gmax > gminm)
    #inj = ((gmax/gmaxm > 10.) | (gmin/gminm < 10.))

    # first add cooling rules
    gm = gmin
    gb = np.where(fc, gminm, gmax)
    gM = gmaxm
    if it == its[-1]:
      dt = df['t'] - dfm['t']
    else:
      itp = its[i_it+1]
      dfp = openData_withZone(key, itp)
      dt = (dfp['t'] - dfm['t'])/2.

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
        gmin_th, gmax_th = get_variable(cell, 'gmM')
        gm[i] = gmin_th
        gb[i] = gmin_th
        gM[i] = gmax_th
        #dt[k] /= 2. 
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

def open_raddata(key, func='Band'):

  '''
  returns a pandas dataframe with the extracted spectral data
  '''
  rfile_path, rfile_bool = get_radfile(key, func)
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
  if itmax:
    imax = its.index(itmax)
    its = its[:imax]
  if itmin:
    imin = its.index(itmin)
    its = its[imin:]
  
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

def get_radfile(key, func='Band'):

  '''
  Returns path of file with  spectras over the run and boolean for its existence
  '''

  dir_path = GAMMA_dir + '/results/%s/' % (key)
  #file_path = dir_path + "extracted_data.txt"
  file_path = dir_path + 'run_observed_' + func + '.csv'
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
  elif var == 'ish':
    varlist = ['ish_{rs}', 'ish_{fs}']
  else:
    print("Asked variable does not exist")
  
  return varlist

# analyze dataframe
# --------------------------------------------------------------------------------------------------
def df_get_rads(df, nuobs, Tobs, env, 
      method='analytic', func='Band', shOnly=True):
  '''
  nuFnu contribution of the datafile df, separarating RS and FS
  '''

  return [df_shock_nuFnu(df, shFront, nuobs, Tobs, env, method, func, shOnly)
    for shFront in ['RS', 'FS']]

def df_shock_nuFnu(df, shFront, nuobs, Tobs, env,
      method='analytic', func='Band', shOnly=True):
  '''
  Choose shock front and extract its nuFnu contribution
  If shOnly = True, we only consider the cell right behind the shock
  '''

  out = np.zeros((len(Tobs), len(nuobs)))
  if shFront not in ['RS', 'FS']:
    print("'shFront' input must be 'RS' of 'FS'")
    return out
  if shOnly:
    cell = df_get_cellBehindShock(df, shFront, theory=True)
    if cell.empty:
      return out
    if method == 'analytic':
      out = cell_nuFnu(cell, nuobs, Tobs, env, method, func)
    else:
      print("Numeric method to be coming")
    #out = cell_flux_old(cell, nuobs, Tobs, dTobs, env, func='analytic')
    #out = nuobs*cell_flux(cell, nuobs, Tobs, dTobs, env, method, func)
  else:
    n = 3. if shFront == 'RS' else 2.
    sh = df.loc[(df['zone'] == n) & (df['gmax'] > 1.)]
    if len(sh) == 0:
      return out
    for i in sh.index:
      cell = df.iloc[i]
      out += nuobs*cell_flux(cell, nuobs, Tobs, dTobs, env, method, func)
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
  allvars = ['Nc', 'R', 'ish', 'u', 'rho', 'ShSt', 'V', 'M', 'Ek', 'Ei', 'pdV']
  allres  = []
  for var in allvars:
    Nvar = len(get_varlist(var, Nz, mode))
    res = []
    if var == 'Nc':
      res = [zone[zone==n].count() for n in reversed(range(Nz))]
    elif var == 'R':
      zones = [4, 3, 2, 1, 0] if mode == 'shells' else [n+1 for n in range(Nz-1)]
      res = [get_radius_zone(df, n)*c_ for n in zones]
    elif var == 'ish':
      res = get_shocksId(df)
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

def get_shocksId(df, n=0):
  '''
  Cells id of shock fronts
  '''
  RS, FS = df_to_shocks(df)
  out = []
  for i, sh in enumerate([RS, FS]):
    if sh.empty:
      iL, iR = 0., 0.
    else:
      iL = min(sh.index) - n
      iR = max(sh.index) + n
    out.append(iR) if i==0 else out.append(iL)
  return out


def get_shocksStrength_old(df):
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
  u    = get_variable(df, 'u')
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
    rho4 = rho4.mean() if rho4.size >0 else 0.
    rho1 = rho1.mean() if rho1.size >0 else 0.
    return rho4, rho1

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
  env = MyEnv(df.attrs['key'])
  zone = df['zone'].to_numpy()
  if np.argwhere(zone==n).any():
    r = df['x'].to_numpy()*c_
    dr = df['dx'].to_numpy()*c_
    if df.attrs['geometry'] == 'spherical':
      dV = 4.*pi_*r**2*dr
    else:
      dV = 4.*pi_*env.R0**2*dr
    res = 0.
    if var == 'V':
      data = dV
    elif var == 'M':
      data = df['D'].to_numpy() * dV * env.rhoscale
    else:
      data = get_variable(df, var)
      data *= dV
    res = data[zone==n].sum()
  else:
    res = 0.
  return res


def get_shocksStrength(df, n=3):
  '''
  Derives shocks strength by calculating relative Lorentz factor,
  with a a distance of n cells to the identified shock on each side
  '''
  RS, FS = df_to_shocks(df)
  iCD = df.loc[(df['trac'] > 0.99) & (df['trac'] < 1.01)].index.max()
  out = []
  for i, sh in enumerate([RS, FS]):
    if sh.empty:
      shst = 0.
    elif 0. in sh['trac'].to_list():
      shst = 0.
    else:
      iL = min(sh.index) - n
      iR = max(sh.index) + n
      if i: # FS
        iL = max(iL, iCD+1)
      else: # RS
        iR = min(iR, iCD)
      vL = df.iloc[iL]['vx']
      vR = df.iloc[iR]['vx']
      lfacL = derive_Lorentz(vL)
      lfacR = derive_Lorentz(vR)
      shst = lfacL*lfacR*(1-vL*vR) - 1.
    out.append(shst)
  return out

def get_shocksLfac(df, n=3):
  '''
  Derives shock Lorentz factor from upstream and downstream values
  cf R24a eqn. C1
  '''
  RS, FS = df_to_shocks(df)
  iCD = df.loc[(df['trac'] > 0.99) & (df['trac'] < 1.01)].index.max()
  out = []
  for i, sh in enumerate([RS, FS]):
    if sh.empty:
      lfacsh = 0.
    elif 0. in sh['trac'].to_list():
      lfacsh = 0.
    else:
      iL = min(sh.index) - n
      iR = max(sh.index) + n
      if i: # FS
        iL = max(iL, iCD+1)
        iu = iR
        id = iL
      else: # RS
        iR = min(iR, iCD)
        iu = iL
        id = iR
      vu, Du = df.iloc[iu][['vx', 'D']].to_list()
      vd, Dd = df.iloc[id][['vx', 'D']].to_list()
      Drat = Du/Dd
      vsh = (Drat*vu - vd)/(Drat - 1)
      lfacsh = derive_Lorentz(vsh)
    out.append(lfacsh)
  return out


def df_get_shellCell0(df):
  '''
  Returns the index in dataframe of the first cell in the shells
  '''
  return df.loc[df['trac']>0.5].index.min()

def df_get_CDcells(df, n=0):
  '''
  Returns the cells on both sides of the CD
  '''
  trac = df['trac']
  S1 = df.loc[trac > 1.5]
  S4 = df.loc[trac < 1.5]
  iL = S4.index.max()
  iR = S1.index.min()
  CDm = df.iloc[iL-n]
  CDp = df.iloc[iR+n]
  return CDm, CDp

def df_get_CDvals(df, n=3, m=2):
  '''
  Returns hydro values at CD, averaged over n cells on each side
  density is taken m cells away to avoid numerical oscillations
  rho both sides, lfac, p
  '''
  trac = df['trac']
  iCD = df.loc[trac > 1.5].index.min()

  rdr = df.iloc[iCD-1:iCD+1][['x', 'dx']].to_numpy()
  rCD = 0.5*((rdr[0,0]+0.5*rdr[0,1]) + (rdr[1,0]-0.5*rdr[0,1]))
  CD = df.iloc[iCD-n:iCD+n]

  rho2 = df.iloc[iCD+m:iCD+m+n]['rho'].mean()
  rho3 = df.iloc[iCD-m-n:iCD-m]['rho'].mean()
  vCD = CD['vx'].mean()
  pCD = CD['p'].mean()
  return rCD, rho2, rho3, vCD, pCD

def df_get_valsAtShocks(sh):
  '''
  Returns hydro values behind a shock front
  '''
  if sh.empty:
    i, r, dr, rho, v, p = [0. for i in range(6)]
  else:
    i, r, dr, rho, v, p = sh[['i', 'x', 'dx', 'rho', 'vx', 'p']].to_list()
    i = int(i)

  return i, r, dr, rho, v, p


def df_get_frontsPosition(df, n=3, tol=1e-4):
  '''
  Returns the position of the shock fronts by fitting shock criterion
  '''

  RS, FS = df_to_shocks(df)
  out = []
  for sh in [RS, FS]:
    if sh.empty:
      r = 0.
    else:
      r = df_get_shockfrontPosition(sh, df, n, tol)
    out.append(r)
  return out
  

def df_get_shockfrontPosition(sh, df, n=3, tol=1e-4):
  '''
  Returns the position of the shock front by fitting shock criterion
  '''
  i_sh = sh.index.to_list()
  sign = -1.
  # if sh['trac'].mean() > 1.5:
  #   # FS
  #   sign = 1.
  if sh.empty:
    rsh = 0.
  else:
    k = 0
    while k < n:
      k += 1
      iL = i_sh[0]-1
      iR = i_sh[-1]+1
      #if df['zone'].iloc[iL] > 0.:
      i_sh.insert(0, iL)
      #if df['zone'].iloc[iR] > 0.:
      i_sh += [iR]
    int_hi = np.array([i+0.5 for i in i_sh[:-1]])
    r_arr, dr_arr, v_arr = df[['x', 'dx', 'vx']].iloc[i_sh].to_numpy().transpose()
    int_r = np.array([r+0.5*dr for r, dr in zip(r_arr[:-1], dr_arr[:-1])])
    try:
      int_relv = np.array([(v_arr[i]-v_arr[i+1])/(1-v_arr[i]*v_arr[i+1]) for i in range(len(v_arr)-1)])
      int_2nd = make_interp_spline(int_hi[int_relv>tol], np.log10(int_relv[int_relv>tol]))
    except IndexError:
      v_arr *= -1
      int_relv = np.array([(v_arr[i]-v_arr[i+1])/(1-v_arr[i]*v_arr[i+1]) for i in range(len(v_arr)-1)])
      int_2nd = make_interp_spline(int_hi[int_relv>tol], np.log10(int_relv[int_relv>tol]))
    int_ri = make_interp_spline(int_hi, int_r)
    xi = np.linspace(int_hi[0], int_hi[-1])
    ri = int_ri(xi)
    vi = 10**int_2nd(xi)
    rsh = ri[np.argmax(vi)]
  return rsh

def df_get_shFronts(df):
  '''
  Return cells at the front of each shock
  '''
  
  RS, FS = df_to_shocks(df)
  if RS.empty:
    RS = df.loc[df['zone']==3.]
  RSf = RS.iloc[0]
  if FS.empty:
    FS = df.loc[df['zone']==2.]
  FSf = FS.iloc[-1]
  return RSf, FSf

def df_get_shDown(df, n=3, m=2):
  '''
  Return cells downstream of shock
  '''
  zone = df['zone'].to_numpy()
  i_3  = np.argwhere(zone==3.)[:,0]
  i_2  = np.argwhere(zone==2.)[:,0]
  iCD = df.loc[(df['trac'] > 0.99) & (df['trac'] < 1.01)].index.max()
  RS, FS = df_to_shocks(df)
  RSf, FSf = df_get_shFronts(df)
  if RS.empty:
    RS = df.loc[df['zone']==3.]
  if FS.empty:
    FS = df.loc[df['zone']==2.]
    
  iRS = RS.index.max()
  iFS = FS.index.min()
  iRSd = min(iRS+n, iCD-m)
  #iRSd = max(iRSd, i_3.min())
  iFSd = max(iFS-n, iCD+1+m)
  #iFSd = min(iFSd), i_2.max())
  
  RSd = df.iloc[iRSd].copy()
  FSd = df.iloc[iFSd].copy()
  for cell in [RSd, FSd]:
    cell['gmin'] = get_variable(cell, 'gma_m')
  return RSd, FSd

def df_get_cellsBehindShock(df, n=5):
  return [df_get_cellBehindShock(df, sh, n) for sh in ['RS', 'FS']]

def df_get_cellBehindShock(df, shFront, n=5, m=1):
  '''
  Return a cell object with radiative values & position taken at shock front
  and hydro values taken downstream (n cells after shock)
  If theory == True, the analytical values from shock jump conditions are inputted
  '''
  if shFront not in ['RS', 'FS']:
    print("'shFront' input must be 'RS' or 'FS'")
    out = pd.DataFrame({key:[] for key in df.keys()})
    for key in df.attrs.keys():
      out.attrs[key] = df.attrs[key]
    return out

  #env = MyEnv(df.attrs['key'])
  hdvars = ['dx', 'rho', 'vx', 'p', 'D', 'sx', 'tau']
  anvars = ['rho', 'vx', 'p', 'gmin', 'gmax', 'gm', 'gb', 'gM']
  anvals = []
  sign = 1
  iCD = df.loc[(df['trac'] > 0.99) & (df['trac'] < 1.01)].index.max()
  # if shFront == 'RS':
  #   anvals = [env.rho3/env.rhoscale, env.beta, env.p_sh/(env.rhoscale*c_**2),
  #     env.gma_m, env.gma_Max, env.gma_m, env.gma_m, env.gma_Max] 
  # elif shFront == 'FS':
  #   anvals = [env.rho2/env.rhoscale, env.beta, env.p_sh/(env.rhoscale*c_**2),
  #     env.gma_mFS, env.gma_Max, env.gma_mFS, env.gma_mFS, env.gma_Max]
  RS, FS = df_to_shocks(df)
  if shFront == 'RS':
    sh = RS
  elif shFront == 'FS':
    sh = FS
    #sign = -1
  l = len(sh)
  for key in df.attrs.keys():
    sh.attrs[key] = df.attrs[key]
  if sh.empty:
    return sh
  else:
    i_min, i_max = sh.index.min(), sh.index.max()
    if shFront == 'RS':
      i = i_min
      i_d = min(i_max+n,iCD-m)
    elif shFront == 'FS':
      i = i_max
      i_d = max(i_min-n,iCD+1+m)
    
    front = df.iloc[i].copy()
    down = df.iloc[i_d]
    for key in df.attrs.keys():
      front.attrs[key] = df.attrs[key]
      down.attrs[key] = df.attrs[key]
    front[hdvars] = down[hdvars]
    front['gmin'] = get_variable(front, "gma_m")

    # if theory:
    #   front[anvars] = anvals
    return front


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
  RSsh = df.loc[(df['Sd']==1.) & (df.index <= icd)]
  ilist = (RSsh.index.diff() > 1).cumsum()
  if len(ilist) <= 1:
    RSilist = np.split(ilist, np.flatnonzero(np.diff(ilist)!=1))
  else:
    RSilist = np.split(ilist, np.flatnonzero(np.diff(ilist)!=1))[1:]
  RSlist = [RSsh.iloc[iarr] for iarr in RSilist]
  FSsh = df.loc[(df['Sd']==1.) & (df.index > icd)]
  ilist = (FSsh.index.diff() > 1).cumsum()
  if len(ilist) <= 1:
    FSilist = np.split(ilist, np.flatnonzero(np.diff(ilist)!=1))
  else:
    FSilist = np.split(ilist, np.flatnonzero(np.diff(ilist)!=1))[1:]
  FSlist = [FSsh.iloc[iarr] for iarr in FSilist]
  # RSsh = S4.loc[df['Sd']==1.]
  # RSlist = np.split(RSsh, np.flatnonzero(np.diff(RSsh.index) != 1) + 1)
  # FSsh = S1.loc[df['Sd']==1.]
  # FSlist = np.split(FSsh, np.flatnonzero(np.diff(FSsh.index) != 1) + 1)
  for (front, ilims, shlist) in zip(['RS', 'FS'], [(i4b, icd), (icd, i1f)], [RSlist, FSlist]):
    iL, iR = ilims
    crossedSh = [sh for sh in shlist if ((sh.index.max()<iL) | (sh.index.min()>iR))]
    if (len(crossedSh) > 0) or (len(shlist)== 0):
        out.append(pd.DataFrame(columns=df.keys()))
    elif len(shlist) == 1:
      out.append(shlist[0])
    else:
      fronts = [sh for sh in shlist if ((sh.index.min()>=iL) & (sh.index.max()<=iR))]
      if front == 'RS':
        out.append(shlist[0])
      else:
        out.append(shlist[-1])
  return out

def df_check_crossed(df, tol=50):
  '''
  Check which shocks have crossed their respective shells
  '''
  RScr, FScr = False, False
  shocks = df.loc[(df['Sd'] == 1.0)]
  t = df['t'].mean()

  if not shocks.empty: # shocks.empty == True: setup OR both crossed and are out of box
    RS, FS = df_to_shocks(df)
    
    if (t > tol):
      if RS.empty: # no shock inside shell 4
        # shock has crossed
        RScr = True
      if FS.empty: # no shock inside shell 1
        # shock has crossed
        FScr = True
  else:
    if (t > tol): # t != 0.
      RScr, FScr = True, True
  return RScr, FScr
      
def zoneID(df, Ttol=.02):
  '''
  Detect zones from passive tracer and shocks
  !!! MUST BE ON DATA PRODUCED WITH SHOCK DETECTION
  '''
  r    = df['x'].to_numpy()
  trac = df['trac'].to_numpy()
  Sd   = df['Sd'].to_numpy()
  zone = np.zeros(trac.shape)

  # use tracer to identify shells 1 and 4
  S4 = (trac > 0.99) & (trac < 1.01)
  S1 = (trac > 1.99) & (trac < 2.01)
  i_4  = np.argwhere(S4)[:,0]
  i_1  = np.argwhere(S1)[:,0]
  zone[i_4] = 4.
  zone[i_1] = 1.
  RS, FS = df_to_shocks(df)
  RScr, FScr = df_check_crossed(df)
  # case where no shocks propagating in the shells, use own algorithm
  ilist  = get_fused_interf(df)
  if RScr:
    zone[S4] = 3.
    i4   = i_4.min()
    wave = [item for item in ilist if i4<item[2]][0]
    iw   = min(wave[2], i_4.max())
    #zone[i4:iw] = 4.
      #zone[iw:i_4.max()] = 3.
  else:
    if RS.empty: # too early for shocks
      T = get_variable(df, 'T')
      rho, p = df[['rho', 'p']].to_numpy().transpose()
      T4 = (1+Ttol)*p[S4].min()/rho[S4].min()
      sh = (T>T4) & S4
      Sd[sh] = 1.
      zone[sh] = 3.
    else:
      id_RS  = min(RS.index) #shocked region begins at shock
      zone[id_RS:i_4.max()+1] = 3.
  if FScr:
    zone[S1] = 2.
    i1   = i_1.max()
    wave = [item for item in ilist if i1>item[1]][-1]
    iw   = max(wave[1], i_1.min())
    zone[iw:i1+1] = 1.
      #zone[i_1.min():iw] = 2.
  else:
    if FS.empty: # too early for shocks
      T = get_variable(df, 'T')
      rho, p = df[['rho', 'p']].to_numpy().transpose()
      T1 = (1+Ttol)*p[S1].min()/rho[S1].min()
      sh = (T>T1) & S1
      Sd[sh] = 1.
      zone[sh] = 2.
    else:
      id_FS  = max(FS.index)
      zone[i_1.min():id_FS+1] = 2.
  
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
  x = df['x'].to_numpy()
  rho, u, p = df_get_primvar(df)
  # filtering rho because data can be noisy
  #rho       = gaussian_filter1d(rho, sigma=1, order=0)
  int_rho   = get_varinterf(rho, x, True, h, prom, rel_h)
  int_u     = get_varinterf(u, x, False, h, prom, rel_h)
  int_p     = get_varinterf(p, x, True, h, prom, rel_h)
  return int_rho, int_u, int_p

def get_varinterf(arr, x, takelog=True, h=.05, prom=.05, rel_h=.99):
  if takelog:
    arr = np.log10(arr)
  arr   = np.gradient(arr, x)
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


