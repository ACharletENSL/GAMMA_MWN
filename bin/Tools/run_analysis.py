# -*- coding: utf-8 -*-
# @Author: acharlet

'''
This file analyze opened data
'''

# Imports
# --------------------------------------------------------------------------------------------------
import numpy as np
import scipy.integrate as spi
from data_IO import *
from phys_radiation import thinshell_spectra
from environment import MyEnv

# Analysis functions
# --------------------------------------------------------------------------------------------------
def get_timeseries(var, key):

  '''
  Returns dataframe with selected variables only
  '''
  run_data = open_clean_rundata(key)
  mode = run_data.attrs['mode']
  Nz = get_Nzones(key)
  extr_vars = ['Nc', 'R', 'u', 'rho', 'ShSt', 'V', 'M', 'Ek', 'Ei', 'pdV']
  calc_vars = {
    "D":run_get_shellswidth,
    "Rct":run_get_Rct,
    "v":run_get_v,
    "u_i":run_get_u,
    "f":run_get_compressionratio,
    "vcd":run_get_vcd,
    "epsth":run_get_epsth,
    "Msh":run_get_Mshell,
    "Nsh":run_get_Nshell,
    "E":run_get_E,
    "Etot":run_get_Etot,
    "Esh":run_get_Eshell,
    "Econs":run_get_Econs,
    "W":run_get_W,
    "Wtot":run_get_Wtot,
    "ShSt ratio":run_get_ShStratio
  }

  if var in extr_vars:
    varlist = get_varlist(var, Nz, mode)
    varlist.insert(0, 'time')
    out = run_data[varlist].copy(deep=True)
  elif var in calc_vars:
    out = calc_vars[var](run_data)
  else:
    print("Implement function for requested variable")
    out = run_data
  
  return out

def open_clean_rundata(key):

  '''
  Returns pandas dataframe containing data from time series
  '''  
  # open file as pandas dataframe (if doesn't exists, returns False)
  dfile_path, dfile_bool = get_runfile(key)
  run_data = open_rundata(key)
  its = np.array(dataList(key))

  # check if .csv exists
  if not dfile_bool:
    analyze_run(key)
    run_data = open_rundata(key)
  
  last_it = run_data.index[-1]
  if last_it <= its[-1]:
    # check if last data row fits with current analysis
    df = openData_withZone(key, last_it)
    last_data = df_get_all(df)
    lastrow = run_data.iloc[-1]
    #check diffs
    same = True
    if same:
      if last_it < its[-1]:
        # continue after last_it
        analyze_run(key)
      else:
        # everything's good
        pass
    else:
      # delete file and do whole analysis
      os.remove(get_runfile(key)[0])
      analyze_run(key)
  else: # last_it > its[-1]
    # delete file and do whole analysis
    os.remove(get_runfile(key)[0])
    analyze_run(key)
  
  run_data = open_rundata(key)
  return run_data

def compare_crosstimes(key):
  '''
  Compare crossing times found numerically with the analytical values
  '''

  env = MyEnv(get_physfile(key))
  tRS_num, tFS_num = get_crosstimes(key)
  reldiff_tRS = reldiff(tRS_num, env.tRS)
  reldiff_tFS = reldiff(tFS_num, env.tFS)
  return reldiff_tRS, reldiff_tFS

def get_crosstimes(key):
  '''
  Outputs crossing time for a given run
  Find them by selecting when corresponding Ncells = 0
  '''
  run_data = open_clean_rundata(key)
  StRS = run_data['ShSt_{rs}'].to_numpy()[1:]
  StFS = run_data['ShSt_{fs}'].to_numpy()[1:]
  t    = run_data['time'].to_numpy()[1:]
  try:
    tRS = t[StRS==0.].min()
  except ValueError:
    print('Run didnt reach RS crossing time')
    tRS = t[-1]
  try:
    tFS = t[StFS==0.].min()
  except ValueError:
    print('Run didnt reach FS crossing time')
    tFS = t[-1]
  
  return tRS, tFS

def get_spectrum_run(key, mode='r'):
  '''
  Read data files from a run, returns data file with time and freq bins
  '''

  rfile_path, rfile_bool = get_radfile(key)
  if (not rfile_bool) or (mode == 'w'): 
    derive_spectrum_run(key)
  
  out = pd.read_csv(rfile_path, index_col=0)
  return out

def derive_spectrum_run(key, NTbins=50, Tmax=5., nubins=np.logspace(-3, 1, 100)):
  '''
  Creates data file for spectrum
  algorithm (v0, simple Band function):
  - open data file
  - compute spectras from each shock with corresponding observer times
  - add those to corresponding observer time bins
  - write a file
  '''

  its = np.array(dataList(key))
  env = MyEnv(get_physfile(key))
  nubins *= env.nu0
  spec_tot = np.zeros((NTbins-1, len(nubins)))
  dT = Tmax/NTbins
  Tbins = np.arange(NTbins)*dT
  cTbins = 0.5 * (Tbins[1:] + Tbins[:-1])
  
  tRS, tFS = get_crosstimes(key)
  rads = get_timeseries('R', key)
  dtime = np.diff(rads['time'].to_numpy())
  radsRS = rads['R_{rs}'].to_numpy()
  radsFS = rads['R_{fs}'].to_numpy()

  for it, rRS, rFS, dt in zip(its, radsRS, radsFS, dtime):
    print(f"Analyzing file of it = {it}")
    df, t, tsim = openData_withtime(key, it)
    # emissivity can vary

    if tsim <= tRS:
      Ton = env.t0 + t - rRS/c_
      Tth = rRS/(2*env.lfacRS**2*c_)
      tTbin = 1. + (cTbins - Ton)/Tth
      spec_tot += (dt/dT) * thinshell_spectra(nubins, tTbin, rRS, env, True)
    if tsim <= tFS:
      Ton = env.t0 + t - rFS/c_
      Tth = rFS/(2*env.lfacFS**2*c_)
      tTbin = 1. + (cTbins - Ton)/Tth
      spec_tot += (dt/dT) *  thinshell_spectra(nubins, tTbin, rFS, env, False)

  # time as column or index?
  out = pd.DataFrame(spec_tot, columns=np.arange(len(nubins)), index=cTbins)
  out.to_csv(rfile_path)

def analyze_run(key, itmin=0, itmax=None):

  '''
  Analyze a run, filling a pandas dataframe
  writes it in a corresponding .csv file
  /!\ if itmin != 0, starts at first iteration AFTER itmin
  '''
  dfile_path, dfile_bool = get_runfile(key)
  varlist = vars_header(key)
  varlist.insert(0, 'time')
  data = pd.DataFrame(columns=varlist)
  its = np.array(dataList(key, itmin, itmax))
  if itmin:
    its = [it for it in its if i>itmin]
  for it in its:
    print(f"Analyzing file of it = {it}")
    df, t, dt = openData_withtime(key, it)
    tup = df_get_all(df)
    results = [item for arr in tup for item in arr]
    results.insert(0, dt)
    dict = {var:res for var, res in zip(varlist, results)}
    data.loc[it] = pd.Series(dict)
    #entry = pd.DataFrame.from_dict(dict)
    #data = pd.concat([data, entry], ignore_index=True)
  
  data.index = its
  #scale properly
  env      = MyEnv(get_physfile(key))
  rhoscale = getattr(env, env.rhoNorm)
  Escale   = rhoscale*c_**2
  Sscale   = 4.*pi_*c_**2
  Vscale   = 1.
  if env.geometry != 'spherical':
    Vscale = 4.*pi_*env.R0**2
    Sscale = Vscale
  for var in [f'V_{i}' for i in range(1,5)]:
    data[var] = data[var].multiply(Vscale)
  for var in ['rho_3', 'rho_2']:
    data[var] = data[var].multiply(rhoscale)
  for var in ['pdV_4', 'pdV_1']:
    data[var] = data[var].multiply(Escale*c_*Sscale)
  for var in [f'M_{i}' for i in range(1,5)]:
    data[var] = data[var].multiply(rhoscale*Vscale)
  for var in [f'Ek_{i}' for i in range(1,5)]:
    data[var] = data[var].multiply(Escale*Vscale)
  for var in [f'Ei_{i}' for i in range(1,5)]:
    data[var] = data[var].multiply(Escale*Vscale)
  data.to_csv(dfile_path)


def vars_header(key, dfile_path=None):
  if not dfile_path:
    dfile_path, dfile_bool = get_runfile(key)

  allvars = ['Nc', 'R', 'u', 'rho', 'ShSt', 'V', 'M', 'Ek', 'Ei', 'pdV']
  Nz  = get_Nzones(key)
  mode = get_runatts(key)[0]
  varlist = []
  for var in allvars:
    varlist.extend(get_varlist(var, Nz, mode))
  
  return varlist

# functions for derived variables
def run_get_shellswidth(run_data):
  '''
  Returns dataframe with distance of shock to interface
  (shocked shells width)
  '''
  time = run_data['time'].to_numpy()
  R4b  = run_data['R_{4b}'].to_numpy()
  Rrs  = run_data['R_{rs}'].to_numpy()
  Rcd  = run_data['R_{cd}'].to_numpy()
  Rfs  = run_data['R_{fs}'].to_numpy()
  R1f  = run_data['R_{1f}'].to_numpy()
  D4   = Rrs - R4b
  D3   = Rcd - Rrs
  D2   = Rfs - Rcd
  D1   = R1f - Rfs
  out  = pd.DataFrame(np.array([time, D4, D3, D2, D1]).transpose(), columns=['time', 'D_4', 'D_3', 'D_2', 'D_1'], index=run_data.index)
  return out

def run_get_Rct(run_data):
  '''
  Returns dataframe with distance of interfaces - c*t
  '''
  time = run_data['time'].to_numpy()
  Rrs  = run_data['R_{rs}'].to_numpy() - c_*time
  Rcd  = run_data['R_{cd}'].to_numpy() - c_*time
  Rfs  = run_data['R_{fs}'].to_numpy() - c_*time
  out  = pd.DataFrame(np.array([time, Rrs, Rcd, Rfs]).transpose(),
    columns=['time', 'R_{rs}', 'R_{cd}', 'R_{fs}'], index=run_data.index)
  return out

def run_get_v(run_data):
  '''
  Returns dataframe of interface velocities
  '''
  time = run_data['time'].to_numpy()
  Rrs  = run_data['R_{rs}'].to_numpy()
  Rcd  = run_data['R_{cd}'].to_numpy()
  Rfs  = run_data['R_{fs}'].to_numpy()
  outarr = [time]
  for r in [Rrs, Rcd, Rfs]:
    r = gaussian_filter1d(r, sigma=3, order=0)
    v = np.gradient(r, time)/c_
    outarr.append(v)
  out = pd.DataFrame(np.array(outarr).transpose(),
    columns=['time', 'v_{rs}', 'v_{cd}', 'v_{fs}'], index=run_data.index)
  return out

def run_get_u(run_data):
  '''
  Returns dataframe of interface velocities
  '''
  time = run_data['time'].to_numpy()
  Rrs  = run_data['R_{rs}'].to_numpy()
  Rcd  = run_data['R_{cd}'].to_numpy()
  Rfs  = run_data['R_{fs}'].to_numpy()
  outarr = [time]
  for r in [Rrs, Rcd, Rfs]:
    r = gaussian_filter1d(r, sigma=3, order=0)
    v = np.gradient(r, time)/c_
    u = derive_proper(v)
    outarr.append(u)
  out = pd.DataFrame(np.array(outarr).transpose(),
    columns=['time', 'u_{rs}', 'u_{cd}', 'u_{fs}'], index=run_data.index)
  return out

def run_get_vcd(run_data):
  '''
  Returns dataframe of interfaces velocities compared to the CD
  (shocked shells growth rate)
  '''
  shellw = run_get_Rcd(run_data)
  time   = shellw['time'].to_numpy()
  D3     = shellw['D3'].to_numpy()
  D2     = shellw['D2'].to_numpy()
  outarr = [time]
  for w in [D3, D2]:
    w = gaussian_filter1d(r, sigma=3, order=0)
    dw = np.gradient(w, time)
    outarr.append(dw)
  out = pd.DataFrame(np.array(outarr).transpose(),
    columns=['time', 'dD3', 'dD2'], index=run_data.index)
  return out

def run_get_epsth(run_data):
  '''
  Returns dataframe of thermal efficiencies in the shells
  (energy converted from kinetic to thermal)
  '''
  time = run_data['time'].to_numpy()
  eth3 = run_data['Ei_3'].to_numpy()/run_data['Ek_4'][0]
  eth2 = run_data['Ei_2'].to_numpy()/run_data['Ek_1'][0]
  out  = pd.DataFrame(np.array(time, eth3, eth2).transpose(),
    columns=['time', 'eth3', 'eth2'], index=run_data.index)
  return out

def run_get_E(run_data):
  '''
  Returns dataframe of kinetic + internal energy for each zone
  '''
  time = run_data['time'].to_numpy()
  E4   = run_data['Ei_4'].to_numpy() + run_data['Ek_4'].to_numpy()
  E3   = run_data['Ei_3'].to_numpy() + run_data['Ek_3'].to_numpy()
  E2   = run_data['Ei_2'].to_numpy() + run_data['Ek_2'].to_numpy()
  E1   = run_data['Ei_1'].to_numpy() + run_data['Ek_1'].to_numpy()
  out  = pd.DataFrame(np.array([time, E4, E3, E2, E1]).transpose(),
    columns=['time', 'E_4', 'E_3', 'E_2', 'E_1'], index=run_data.index)
  return out

def run_get_Etot(run_data):
  '''
  Return dataframe of summed energy to check for energy conservation
  '''
  time = run_data['time'].to_numpy()
  E4   = run_data['Ei_4'].to_numpy() + run_data['Ek_4'].to_numpy()
  E3   = run_data['Ei_3'].to_numpy() + run_data['Ek_3'].to_numpy()
  E2   = run_data['Ei_2'].to_numpy() + run_data['Ek_2'].to_numpy()
  E1   = run_data['Ei_1'].to_numpy() + run_data['Ek_1'].to_numpy()
  #W4   = spi.cumulative_trapezoid(run_data['pdV_4'].to_numpy(), x=time, initial=0.)
  #W1   = spi.cumulative_trapezoid(run_data['pdV_1'].to_numpy(), x=time, initial=0.)
  Etot = E1 + E2 + E3 + E4 #+ W1 + W4
  out = pd.DataFrame(np.array([time, Etot]).transpose(),
    columns=['time', 'E_{tot}'], index=run_data.index)
  return out

def run_get_Nshell(run_data):
  '''
  Return number of cells per shell
  '''
  time = run_data['time'].to_numpy()
  Nsh1 = run_data['Nc_1'].to_numpy() + run_data['Nc_2'].to_numpy()
  Nsh4 = run_data['Nc_4'].to_numpy() + run_data['Nc_3'].to_numpy()
  out = pd.DataFrame(np.array([time, Nsh4, Nsh1]).transpose(),
    columns=['time', 'N_4+N_3', 'N_1+N_2'], index=run_data.index)
  return out

def run_get_Mshell(run_data):
  '''
  Return dataframe of summed mass to check for energy conservation
  '''
  time = run_data['time'].to_numpy()
  Msh4 = run_data['M_4'].to_numpy() + run_data['M_3'].to_numpy()
  Msh1 = run_data['M_1'].to_numpy() + run_data['M_2'].to_numpy()
  out = pd.DataFrame(np.array([time, Msh4, Msh1]).transpose(),
    columns=['time', 'M_4+M_3', 'M_1+M_2'], index=run_data.index)
  return out

def run_get_Eshell(run_data):
  '''
  Return dataframe of summed energy per shell 
  '''
  time = run_data['time'].to_numpy()
  E4   = run_data['Ei_4'].to_numpy() + run_data['Ek_4'].to_numpy()
  E3   = run_data['Ei_3'].to_numpy() + run_data['Ek_3'].to_numpy()
  E2   = run_data['Ei_2'].to_numpy() + run_data['Ek_2'].to_numpy()
  E1   = run_data['Ei_1'].to_numpy() + run_data['Ek_1'].to_numpy()
  #W4   = spi.cumulative_trapezoid(run_data['pdV_4'].to_numpy(), x=time, initial=0.)
  #W1   = spi.cumulative_trapezoid(run_data['pdV_1'].to_numpy(), x=time, initial=0.)
  Esh4 = E3 + E4 #+ W4
  Esh1 = E1 + E2 #+ W1
  out = pd.DataFrame(np.array([time, Esh4, Esh1]).transpose(),
    columns=['time', 'E_4+E_3', 'E_1+E_2'], index=run_data.index)
  return out

def run_get_W(run_data):
  '''
  Return dataframe of work performed on outside interfaces
  '''
  time = run_data['time'].to_numpy()
  W4   = spi.cumulative_trapezoid(run_data['pdV_4'].to_numpy(), x=time, initial=0.)
  W1   = spi.cumulative_trapezoid(run_data['pdV_1'].to_numpy(), x=time, initial=0.)
  out  = pd.DataFrame(np.array([time, W4, W1]).transpose(),
    columns=['time', 'W_4', 'W_1'], index=run_data.index)
  return out

def run_get_Wtot(run_data):
  '''
  Return dataframe of work performed on outside interfaces
  '''
  time = run_data['time'].to_numpy()
  W4   = spi.cumulative_trapezoid(run_data['pdV_4'].to_numpy(), x=time, initial=0.)
  W1   = spi.cumulative_trapezoid(run_data['pdV_1'].to_numpy(), x=time, initial=0.)
  W    = W4 - W1
  out  = pd.DataFrame(np.array([time, W]).transpose(),
    columns=['time', 'W'], index=run_data.index)
  return out

def run_get_Econs(run_data):
  '''
  Return dataframe of summed energy to check for energy conservation
  '''
  time = run_data['time'].to_numpy()
  E4   = run_data['Ei_4'].to_numpy() + run_data['Ek_4'].to_numpy()
  E3   = run_data['Ei_3'].to_numpy() + run_data['Ek_3'].to_numpy()
  E2   = run_data['Ei_2'].to_numpy() + run_data['Ek_2'].to_numpy()
  E1   = run_data['Ei_1'].to_numpy() + run_data['Ek_1'].to_numpy()
  W4   = spi.cumulative_trapezoid(run_data['pdV_4'].to_numpy(), x=time, initial=0.)
  W1   = spi.cumulative_trapezoid(run_data['pdV_1'].to_numpy(), x=time, initial=0.)
  Etot = E1 + E2 + E3 + E4 + W1 - W4
  out = pd.DataFrame(np.array([time, Etot]).transpose(),
    columns=['time', 'E_{tot}'], index=run_data.index)
  return out

def run_get_ShStratio(run_data):
  '''
  Return dataframe of shocks strength ratio 
  '''
  time = run_data['time'].to_numpy()
  shst_RS = run_data['ShSt_{rs}'].to_numpy()
  shst_FS = run_data['ShSt_{fs}'].to_numpy()
  ratio = shst_RS/shst_FS
  out = pd.DataFrame(np.array([time, ratio]).transpose(),
    columns=['time', 'ShSt_{rs}/ShSt_{fs}'], index=run_data.index)
  return out

def run_get_compressionratio(run_data):
  '''
  Return dataframe of compression ratio near CD
  '''
  time = run_data['time'].to_numpy()
  n3   = run_data['rho_3'].to_numpy()
  n2   = run_data['rho_2'].to_numpy()
  ratio = n3/n2
  out = pd.DataFrame(np.array([time, ratio]).transpose(),
    columns=['time', 'f'], index=run_data.index)
  return out