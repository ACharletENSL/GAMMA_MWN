# -*- coding: utf-8 -*-
# @Author: acharlet

'''
This file analyze opened data
'''

# Imports
# --------------------------------------------------------------------------------------------------
import numpy as np
import scipy.integrate as spi
from scipy.signal import savgol_filter
from data_IO import *
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
  extr_vars = ['Nc', 'R', 'ish', 'u', 'rho', 'ShSt', 'V', 'M', 'Ek', 'Ei', 'pdV']
  calc_vars = {
    "D":run_get_shellswidth,
    "Rct":run_get_Rct,
    "v":run_get_v,
    "u_i":run_get_u,
    "u_sh":run_get_ush,
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

def get_runTimes(key):
  '''
  Open (creates if needed) files containing times and Delta t of the run
  '''
  dir_path = GAMMA_dir + '/results/%s/' % (key)
  file_path = dir_path + "run_times.txt"
  if os.path.isfile(file_path):
    out = np.loadtxt(file_path, delimiter=',')
  else:
    print("Creating times file")
    its = dataList(key)
    times = []
    for it in its:
      df = openData(key, it)
      times.append(df.iloc[0]['t'])
    dts = [(times[j+1]-times[j-1])/2. for j in range(1,len(times)-1)]
    dts.insert(0, (3*times[0]-times[1])/2)
    dts.append((3*times[-1]-times[-2])/2)
    out = np.asarray([its, times, dts])
    np.savetxt(file_path, out, delimiter=',')
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
  run_data.attrs['key'] = key
  return run_data

def get_cell_timevalues(key, var, i, its=[]):
  if type(var) == list:
    cool = any(item in var for item in cool_vars)

  else:
    cool = (var in cool_vars)

  if len(its) == 0.:
    its = np.array(dataList(key))[1:]
  
  time = np.zeros(len(its))

  if type(var)==list:
    data = np.zeros((len(var), len(its)))
  else:
    data = np.zeros(len(its))
  for k, it in enumerate(its):
    print(f'Reading file {it}')
    df, t, dt = openData_withtime(key, it)
    if cool:
      df = openData_withDistrib(key, it)
    trac = df['trac'].to_numpy()
    i4 = np.argwhere((trac > 0.99) & (trac < 1.01))[:,0].min()
    cell = df.iloc[i+i4]
    time[k] = dt
    if type(var)==list:
      for j, name in enumerate(var):
        data[j][k] = get_variable(cell, name)
    else:
      data[k] = get_variable(cell, var)

  return time, data

def get_behindShock_vals(key, varlist, itmin=0, itmax=None, thcorr=True, n=2, m=1):
  '''
  Return arrays of values behind shocks
  if contribsOnly=True, returns arrays with only the values that contribues to emission
  '''
  
  env = MyEnv(key)
  anvars = ['rho', 'vx', 'p']
  anvalsRS = [env.rho3/env.rhoscale, env.beta, env.p_sh/(env.rhoscale*c_**2)]
  anvalsFS = [env.rho2/env.rhoscale, env.beta, env.p_sh/(env.rhoscale*c_**2)]
  
  RS_ids, FS_ids = [], []
  RS_out, FS_out = [], []
  its = np.array(dataList(key, itmin=itmin, itmax=itmax))[1:]
  # coolvar = (len([var for var in varlist if var in cool_vars])>0)
  # if coolvar:
  #   its = its[1:]
  
  for it in its:
    df = openData_withDistrib(key, it)
    t  = df['t'].mean()
    #RS, FS = df_get_cellsBehindShock(df)

    RScr, FScr = df_check_crossed(df)
    bothCrossed = (RScr and FScr)
    if bothCrossed: break

    RS, FS = df_get_shDown(df, n=n, m=m)
    ish = df.loc[df['trac']>0.5].index.min()

    if not RS.empty:
      iRS = int(RS['i']) - ish
      if iRS not in RS_ids:
        print(f'Reading file {it} for RS')
        if thcorr and (len(RS_ids) == 0):
          print('Using analytical values for 1st RS contribution')
          RS[anvars] = anvalsRS
          RS['gmin'] = get_variable(RS, 'gma_m')
        RSvals = get_vars(RS, varlist)
        RSvals = np.concatenate((np.array([it, t]), RS['x'], RSvals))
        RS_ids.append(iRS)
        RS_out.append(RSvals)
        
    if not FS.empty:
      iFS = int(FS['i']) - ish
      if iFS not in FS_ids:
        print(f'Reading file {it} for FS')
        if thcorr and (len(FS_ids) == 0):
          print('Using analytical values for 1st FS contribution')
          FS[anvars] = anvalsFS
          FS['gmin'] = get_variable(FS, 'gma_m')
        FSvals = get_vars(FS, varlist)
        FSvals = np.concatenate((np.array([it, t]), FS['x'], FSvals))
        FS_ids.append(iFS)
        FS_out.append(FSvals)

  RS_out = np.array(RS_out).transpose()
  FS_out = np.array(FS_out).transpose()
  return RS_out, FS_out

def get_behindShock_value(key, var, its=[]):
  if len(its) == 0.:
    its = np.array(dataList(key))[1:]
  if var in cool_vars:
    its = its[1:]
  rs = np.zeros((2, len(its)))
  data = np.zeros((2, len(its)))
  time = np.zeros(len(its))
  for k, it in enumerate(its):
    print(f'Reading file {it}')
    df, t, dt = openData_withtime(key, it)
    if var in cool_vars:
      df = openData_withDistrib(key, it)
    #RS, FS = df_get_cellsBehindShock(df)
    RS, FS = df_get_shDown(df)
    time[k] = dt
    if var == 'i':
      data[0,k] = RS.name if len(RS) > 0 else 0
      data[1,k] = FS.name if len(FS) > 0 else 0
    else:
      if not RS.empty:
        rs[0,k] = RS['x']
        data[0,k] = get_variable(RS, var)
      else:
        rs[0,k] = 0.
        data[0,k] = 0.
      if not FS.empty:
        rs[1,k] = FS['x']
        data[1,k] = get_variable(FS, var)
      else:
        rs[1,k]
        data[1,k] = 0.
  return its, time, rs[0], rs[1], data[0], data[1]

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
  '''
  its = dataList(key)[0:]
  Nf = len(its)
  i = 0
  RScr, FScr, bthCr = False, False, False
  t_old = 0.
  while ((i < Nf-1) or (not bthCr)):
    it = its[i]
    print(f'Opening file it {it}')
    df, t, tsim = openData_withtime(key, it)
    RS, FS = df_check_crossed(df)
    if RS != RScr:
      tRS = (t+t_old)/2.
    if FS != FScr:
      tFS = (t+t_old)/2.
    RScr, FScr = RS, FS
    bthCr = RScr and FScr
    t_old = t
    i += 1
  return tRS, tFS
  

def get_crosstimes_old(key):
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

def get_radEnv(key):
  '''
  Returns necessary variables for spectrum calculation from a run
  Nnu : number of frequency bins
  Tbmax : max value of \bar{T} = (Tobs - Ts)/T0
  nT : number of bins per unit Tnorm = betaRS*T0
  '''
  env = MyEnv(key)
  if 'big' in key:
    Tbmax = 10
    nT = 200
    lognumin = -5
    nnu_low = 400
  else:
    Tbmax=5
    nT= 200
    lognumin = -3
    nnu_low = 200
  nuobs = np.concatenate((np.logspace(lognumin, -1, nnu_low), np.logspace(-1, 1.5, 300)), axis=None)*env.nu0
  Tobs = env.Ts + np.linspace(0, Tbmax, nT*Tbmax)*env.T0
  return nuobs, Tobs, env

def run_lightcurve_theoric(key, lognu):
  '''
  Return theoretical flux using equations 8-9-10 from Rahaman et al. 2023
  '''

  nuobs, Tobs, env = get_radEnv(key)
  nu = 10**lognu * env.nu0
  F0, g, nupk, Tej, T0, Tf = env.F0, env.gRS, env.nu0, env.TeffejRS, env.T0, env.TRS
  tT = (Tobs-Tej)/T0
  tnu = nu/nupk
  tA = tnu * tT
  def func(x):
    return x**-3 * (tA**-1*x - g**2)**2 * Band_func(x) 
  ymin = np.where(Tobs <= Tej+T0, 1, tT**-1)
  ymax = np.where(Tobs <= Tej+Tf, 1, (tT*T0/Tf)**-1)
  xmin = tA*(1+g**2*(ymin**-1 - 1))
  xmax = tA*(1+g**2*(ymax**-1 - 1))
  flux_RS = nu*F0*(3*g**2/(1-g**2)**3)* tnu**2 * tT**3 * np.array([spi.quad_vec(func, xm, xM)[0] for xm, xM in zip(xmin, xmax)])

  F0, g, nupk, Tej, T0, Tf = env.F0FS, env.gFS, env.nu0FS, env.TeffejFS, env.T0FS, env.TFS
  tT = (Tobs-Tej)/T0
  tnu = nu/nupk
  tA = tnu * tT
  def func(x):
    return x**-3 * (tA**-1*x - g**2)**2 * Band_func(x) 
  ymin = np.where(Tobs <= Tej+T0, 1, tT**-1)
  ymax = np.where(Tobs <= Tej+Tf, 1, (tT*T0/Tf)**-1)
  xmin = tA*ymin*(1+g**2*(ymin**-1 - 1))
  xmax = tA*ymax*(1+g**2*(ymax**-1 - 1))
  flux_FS = nu*F0*(3*g**2/(1-g**2)**3)* tnu**2 * tT**3 * np.array([spi.quad_vec(func, xm, xM)[0] for xm, xM in zip(xmin, xmax)])

  # tT = (Tobs-env.TeffejFS)/env.T0FS
  # tTeff1 = (1-env.gFS**2) + env.gFS**2*tT
  # tTeff1_f = (1-env.gFS**2) + env.gFS**2*env.tTfFS
  # tTeff2 = (1-env.gFS**2) + env.gFS**2*(tT/env.tTfFS)
  # nupFS = env.nu0FS * np.where(tT<=env.tTfFS, tT**-1, (env.tTfFS*tTeff2)**-1)
  # nuFnupFS = env.nu0F0FS * np.where(tT<=env.tTfFS, 1-tTeff1**-3, (1-tTeff1_f**-3)*tTeff2**-3)
  # x = nuobs[np.newaxis, :]/nupFS[:, np.newaxis]
  # flux_FS = nuFnupFS[:, np.newaxis] * Band_func(x)
  return flux_RS, flux_FS


def get_run_radiation(key, func='Band', method='analytic', Nnu=200, n=2, m=1, itmin=0, itmax=None, thcorr=True):

  '''
  Get radiation from a run, filling a pandas dataframe
  writes it in a corresponding .csv file
  !!! if itmin != 0, starts at first iteration AFTER itmin
  '''

  dfile_path, dfile_bool = get_radfile(key, func)
  nuobs, Tobs, env = get_radEnv(key, Nnu=Nnu)
  # analytical values for 1st timestep
  anvars = ['rho', 'vx', 'p']
  anvalsRS = [env.rho3/env.rhoscale, env.beta, env.p_sh/(env.rhoscale*c_**2)]
  anvalsFS = [env.rho2/env.rhoscale, env.beta, env.p_sh/(env.rhoscale*c_**2)]
  RS_ids, FS_ids = [], []
  its = np.array(dataList(key, itmin, itmax))[2:]
  RS_rad, FS_rad = np.zeros((2, len(Tobs), len(nuobs)))
  hybrid = True if key == 'cart_fid' else False
  bothCrossed = False
  for it in its:
    RS_rad_i, FS_rad_i = np.zeros((2, len(Tobs), len(nuobs)))
    df = openData_withDistrib(key, it)
    #RS, FS = df_get_cellsBehindShock(df)
    RS, FS = df_get_shDown(df, n=n, m=m)
    RScr, FScr = df_check_crossed(df)
    bothCrossed = (RScr and FScr)
    if bothCrossed: break
    ish = df.loc[df['trac']>0.5].index.min()
    if not RS.empty:
      iRS = int(RS['i']) - ish
      if iRS not in RS_ids:
        print(f'Generating RS flux from file {it}')
        if thcorr and (len(RS_ids) <= 1):
          print('Using analytical values for 1st RS contribution')
          RS[anvars] = anvalsRS
          RS['gmin'] = get_variable(RS, 'gma_m')
        RS_rad_i = cell_nuFnu(RS, nuobs, Tobs, env, func, hybrid, method)
        #RS_rad_i = df_shock_nuFnu(df, 'RS', nuobs, Tobs, dTobs, env, method, func)
        RS_ids.append(iRS)
    if not FS.empty:
      iFS = int(FS['i']) - ish
      if iFS not in FS_ids:
        print(f'Generating FS flux from file {it}')
        if thcorr and (len(FS_ids) <= 1):
          print('Using analytical values for 1st FS contribution')
          FS[anvars] = anvalsFS
          FS['gmin'] = get_variable(FS, 'gma_m')
        FS_rad_i = cell_nuFnu(FS, nuobs, Tobs, env, func, hybrid, method)
        #FS_rad_i = df_shock_nuFnu(df, 'FS', nuobs, Tobs, dTobs, env, method, func)
        FS_ids.append(iFS)
    RS_rad += RS_rad_i
    FS_rad += FS_rad_i
  
  nulist = [f'nuobs_{i}_'+sub for sub in ['RS', 'FS'] for i in range(len(nuobs))]
  rads = np.concatenate((RS_rad, FS_rad), axis=1)
  data = pd.DataFrame(rads, columns=nulist)
  data.insert(0, "Tobs", Tobs)
  data.to_csv(dfile_path)
  

def get_run_contribCells(key, varlist=['x', 'nup_m', 'Ton', 'EpnuFC'], n=1, itmin=0, itmax=None):
  dfile_path, dfile_bool = get_radfile(key)
  nuobs, Tobs, env = get_radEnv(key)
  its = np.array(dataList(key, itmin, itmax))[2:]
  RS_out = np.zeros((len(varlist)+2, env.Nsh4))
  FS_out = np.zeros((len(varlist)+2, env.Nsh1+2))
  kR, kF = 0, 0
  for it in its:
    print(f'Opening file {it}')
    df = openData_withDistrib(key, it)
    RS, FS = df_get_cellsBehindShock(df, n=n, theory=False)
    ShStRS, ShStFS = 0., 0.
    if 'ShSt' in varlist:
      ShStRS, ShStFS = get_shocksStrength(df)
    ish = df.loc[df['trac']>0.5].index.min()
    if not RS.empty:
      iRS = int(RS['i']) - ish
      if iRS not in RS_out[0]:
        RS_out[0, kR] = iRS
        RS_out[1, kR] = it
        for k, var in enumerate(varlist):
          RS_out[2+k, kR] = get_variable(RS, var) if var != 'ShSt' else ShStRS
        kR += 1
    if not FS.empty:
      iFS = int(FS['i']) - ish
      if iFS not in FS_out[0]:
        FS_out[0, kF] = iFS
        FS_out[1, kF] = it
        for k, var in enumerate(varlist):
          FS_out[2+k, kF] = get_variable(FS, var) if var != 'ShSt' else ShStFS
        kF += 1
  return RS_out, FS_out

def derive_spectrum_run_2shocks(key, NTbins=50, Tmax=5., nubins=np.logspace(-3, 1, 100)):
  '''
  Creates data file for spectrum
  algorithm (v0, simple Band function):
  - open data file
  - compute spectras from each shock with corresponding observer times
  - add those to corresponding observer time bins
  - write a file
  '''
  # add normalization for timebins depending on pulse carac size T0 (Tth with R0, etc.)
  # Tobs max = ~ 1000 Tbins, Tbins ~ 1/100 T0
  its = np.array(dataList(key))
  env = MyEnv(get_physfile(key))
  rfile_path, rfile_bool = get_radfile(key)
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
  !!! if itmin != 0, starts at first iteration AFTER itmin
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
  
  times = data['time'].to_numpy()
  dts = [(times[j+1]-times[j-1])/2. for j in range(1,len(times)-1)]
  dts.insert(0, (3*times[0]-times[1])/2)
  dts.append((3*times[-1]-times[-2])/2)
  data.insert(1, "delta_t", dts)
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
  # for var in [f'M_{i}' for i in range(1,5)]:
  #   data[var] = data[var].multiply(rhoscale*Vscale)
  # for var in [f'Ek_{i}' for i in range(1,5)]:
  #   data[var] = data[var].multiply(Escale*Vscale)
  # for var in [f'Ei_{i}' for i in range(1,5)]:
  #   data[var] = data[var].multiply(Escale*Vscale)
  data.to_csv(dfile_path)


def vars_header(key, dfile_path=None):
  if not dfile_path:
    dfile_path, dfile_bool = get_runfile(key)

  allvars = ['Nc', 'R', 'ish', 'u', 'rho', 'ShSt', 'V', 'M', 'Ek', 'Ei', 'pdV']
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
  env = MyEnv(run_data.attrs['key'])
  time = run_data['time'].to_numpy()
  t = time + env.t0
  Rrs  = run_data['R_{rs}'].to_numpy() - c_*t
  Rcd  = run_data['R_{cd}'].to_numpy() - c_*t
  Rfs  = run_data['R_{fs}'].to_numpy() - c_*t
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
  l, d = 3, 1
  l2, d2 = 3, 1
  for r in [Rrs, Rcd, Rfs]:
    #r = savgol_filter(r, l, d)
    v = np.gradient(r, time)/c_
    #v = savgol_filter(v, l2, d2)
    outarr.append(v)
  out = pd.DataFrame(np.array(outarr).transpose(),
    columns=['time', 'v_{rs}', 'v_{cd}', 'v_{fs}'], index=run_data.index)
  return out

def run_get_ush(run_data):
  '''
  Returns dataframe of shocks proper velocity
  '''
  time = run_data['time'].to_numpy()
  dt  = run_data['delta_t'].to_numpy()
  Rrs = run_data['R_{rs}'].to_numpy()/c_
  irs = run_data['ish_{rs}'].to_numpy()
  Rfs = run_data['R_{fs}'].to_numpy()/c_
  ifs = run_data['ish_{fs}'].to_numpy()
  dtlist_rs = np.split(dt, np.flatnonzero(np.diff(irs))+1)
  rlist_rs  = np.split(Rrs, np.flatnonzero(np.diff(irs))+1)
  vlist_rs  = np.split(np.zeros(len(time)), np.flatnonzero(np.diff(irs))+1)
  dtlist_fs = np.split(dt, np.flatnonzero(np.diff(ifs))+1)
  rlist_fs  = np.split(Rfs, np.flatnonzero(np.diff(ifs))+1)
  vlist_fs  = np.split(np.zeros(len(time)), np.flatnonzero(np.diff(ifs))+1)
  for i, dt, r in zip(range(len(dtlist_rs)), dtlist_rs, rlist_rs):
    l = len(dt)
    dt = dt[0] if l == 0 else dt.sum()
    vlist_rs[i] = ((r[-1]-r[0])/dt)*np.ones(l)
  for i, dt, r in zip(range(len(dtlist_fs)), dtlist_fs, rlist_fs):
    l = len(dt)
    dt = dt[0] if l == 0 else dt.sum()
    vlist_fs[i] = ((r[-1]-r[0])/dt)*np.ones(l)
  vrs = np.concatenate(vlist_rs).ravel()
  urs = derive_proper(vrs)
  vfs = np.concatenate(vlist_fs).ravel()
  ufs = derive_proper(vfs)
  outarr = [time, urs, ufs]
  out = pd.DataFrame(np.array(outarr).transpose(),
    columns=['time', 'u_{rs}', 'u_{fs}'], index=run_data.index)
  return out

def run_get_u(run_data):
  '''
  Returns dataframe of interface velocities
  '''
  l, d = 3, 1
  l2, d2 = 3, 1
  l3, d3 = 7, 1
  print(f"r_shock double filtered with savgol ({l}, {d}) then ({l2}, {d2}) then ({l3}, {d3})")
  time = run_data['time'].to_numpy()
  Rrs  = run_data['R_{rs}'].to_numpy()
  Rcd  = run_data['R_{cd}'].to_numpy()
  Rfs  = run_data['R_{fs}'].to_numpy()
  outarr = [time]
  for r in [Rrs, Rcd, Rfs]:
    #r = savgol_filter(r, l, d)
    v = np.gradient(r, time)/c_
    #v = savgol_filter(v, l2, d2)
    u = derive_proper(v)
    #u = savgol_filter(u, l3, d3)
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