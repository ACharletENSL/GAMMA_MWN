# -*- coding: utf-8 -*-
# @Author: acharlet

'''
This file contains a rewrite of what is performed in
thinshell_analysis, but simpler:
  - only extracts shock front values
  - data compatible with get_variable procedure (implies easier plotting)
'''

import matplotlib.transforms as transforms
from plotting_scripts_new import *
import scipy.integrate as spi
from scipy.special import gamma, betainc, hyp2f1
from collections import OrderedDict
plt.ion()

def radEnv(key, lowfreqs=False, scalefac=0):
  '''
  observed frequency, time, and environment variables
  '''
  env = MyEnv(key)
  scaling = 10**scalefac
  env.R0 *= scaling
  env.t0 *= scaling
  env.nu0 /= scaling
  env.T0 *= scaling
  env.Ts *= scaling
  env.F0 *= scaling
  Tbmax = 5
  nT = 100
  numin = -7 if lowfreqs else -3 
  nuobs = env.nu0*np.logspace(numin, 1.5, 600)
  Tobs = env.Ts + np.linspace(0, Tbmax, nT*Tbmax)*env.T0
  return nuobs, Tobs, env

def get_gmasdata_cell(n, key='Last'):
  '''
  Returns part of extracted data of cell n 
  '''
  env = MyEnv(key)
  sh, Nsh, col = ('RS', env.Nsh4, 'r') if n<0 else ('FS', env.Nsh1, 'b')
  path = GAMMA_dir + f'/results/{key}/gmas_{sh}.csv'
  data = pd.read_csv(path)
  Nc = env.Next+Nsh
  k = Nc+n
  dk = data.loc[data['i']==k]
  return dk

def func_cooling_synThin(t, y, rho, drhodt, B2):
  '''
  Differential equation to solve for cooling
  Optically thin synchrotron cooling + adiabatic
  f'(y) = - (sigT B**2 / (6 pi me c)) * y**2 + (d rho/d t) / (3 rho) * y
  '''

  syn = - alpha_ * B2 * y**2
  adiab = (drhodt/(3*rho)) * y
  # (y**2 - 1) instead of y**2, leave until tests vs GAMMA are good
  return syn + adiab

def func_cooling_synThin_noadiab(t, y, rho, drhodt, B2):
  syn = - alpha_ * B2 * y**2
  return syn


def test_constantHydro(n, key='Last'):
  '''
  For testing purpose:
    generates a false dataset where hydro values are constant, then evolves gmas
  '''

  hdvars = ['dx', 'rho', 'vx', 'p', 'D', 'sx', 'tau']
  func_cool = func_cooling_synThin

  # get normal datas
  env = MyEnv(key)
  sh, Nsh, col = ('RS', env.Nsh4, 'r') if n<0 else ('FS', env.Nsh1, 'b')
  path = GAMMA_dir + f'/results/{key}/gmas_{sh}.csv'
  data = pd.read_csv(path)
  Nc = env.Next+Nsh
  k = Nc+n
  dk = data.loc[data['i']==k]

  # create test data with constant hydro
  out = dk.copy()
  out.drop(columns=['gmin_', 'gmax_'])
  values = out.iloc[0][hdvars].to_numpy()
  out[hdvars] = values
  its = out.index.to_list()

  # inject and evolve gmas
  out['dt'] = np.gradient(out['t'].to_numpy())
  out['drho'] = np.gradient(out['rho'].to_numpy())
  out[['gmin_', 'gmax_']] =  0., 0.
  gmin, gmax = get_vars(out.iloc[0], ['gma_m', 'gma_M'], env)
  out.loc[its[0], ['gmin_', 'gmax_']] = np.array([gmin, gmax])
  for j, it_next in enumerate(its[1:]):
    cell = out.iloc[j]
    cell_new = out.iloc[j+1]
    gmas_new = evolve_gmas(cell, cell_new, func_cool, env)
    out.loc[it_next, ['gmin_', 'gmax_']] = gmas_new

  return out


############### plotting functions
def plot_compared_vals(var, logvs, key='Last',
  integrated=False, mFC=None, **kwargs):
  '''
  plot_emission with array of vals
  '''

  lstlist = ['-', '--', '-.', ':']
  varlabel = ''
  ylabel = '$\\nu F_\\nu/\\nu_0 F_0$'
  if var == 'lc':
    xlabel = '$\\bar{T}$'
    varlabel = '$\\log_{10}\\tilde{\\nu} = '
  elif var == 'sp':
    xlabel = '$\\tilde{\\nu}$'
    varlabel = '$\\log_{10}\\bar{T} = '
  dummy_lst = []
  dummy_names = []

  fig, ax = plt.subplots()

  for logv, l in zip(logvs, lstlist):
    dummy_lst.append(plt.plot([],[],c='k',ls=l)[0])
    varstr = ''
    try:
      varstr = f'{logv:d}'
    except ValueError:
      varstr = f'{logv:.1f}'
    dummy_names.append(varlabel+varstr+'$')
    plot_emission(var, logv, key=key, integrated=integrated, mFC=mFC, ax_in=ax, ls=l, **kwargs)

  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  ax.legend(dummy_lst, dummy_names)
  fig.tight_layout()


def compare_vFCmFC(key, var='sp', logv=0):
  '''
  Figure comparing flux for both shocks in vFC to FS in mFC and RS in vFC
  '''
  fig, ax = plt.subplots()
  plot_emission(var, logv, key, mFC=None, ax_in=ax, ls='-')
  plt.autoscale(False)
  plot_emission(var, logv, key, mFC='FS', ax_in=ax, ls='-.')
  ax.set_xlabel("$\\tilde{\\nu}$")
  ax.set_ylabel('$\\nu F_{\\nu}/\\nu_0F_0$')

  dummy_col = [plt.plot([], [], c=c)[0] for c in ['r', 'b', 'k']]
  leg_col = ax.legend(dummy_col, ['RS', 'FS', 'RS+FS'], loc='upper right')
  dummy_lst = [plt.plot([], [], c='k', ls=l)[0] for l in ['-', '-.']]
  ax.legend(dummy_lst, ['fast cooling FS', 'marg. fast cooling FS'], loc='upper left')
  ax.add_artist(leg_col)
  plt.tight_layout()

def plot_emission(var, logv, key='Last', integrated=False, mFC=None, ax_in=None, **kwargs):
  '''
  Plots lightcurve ('lc') or spectrum ('sp')
  integrated = True will perform integration over freq for lightcurve, time for spectrum 
  mFC_arr in order [RS, FS]
  '''

  if not ax_in:
    fig, ax = plt.subplots()
  else:
    ax = ax_in
  
  nuobs, Tobs, env = radEnv(key)
  path = get_path_thinshellObs(key)
  tnu = nuobs/env.nu0
  Tb = (Tobs-env.Ts)/env.T0
  xlabel, ylabel = '', '$\\nu F_{\\nu}/\\nu_0F_0$'
  logx, logy = False, False
  mFC_arr = [False, False]
  if mFC == 'RS': mFC_arr[0] = True
  if mFC == 'FS': mFC_arr[1] = True
  if mFC == 'both': mFC_arr = [True, True]
  nuFnu_RS, nuFnu_FS = open_thinshellObs_data(key, mFC_arr)

  if var == 'lc':
    i = find_closest(tnu, 10**logv)
    x = Tb
    data_RS, data_FS = nuFnu_RS[:, i], nuFnu_FS[:, i]
    data_tot = data_RS + data_FS
    xlabel=("$\\bar{T}$")
  elif var == 'sp':
    i = find_closest(Tb, 10**logv)
    x = tnu
    data_RS, data_FS = nuFnu_RS[i, :], nuFnu_FS[i, :]
    data_tot = data_RS + data_FS
    xlabel=("$\\tilde{\\nu}$")
    logx, logy = True, True
    ipk_RS, ipk_FS = np.argmax(data_RS), np.argmax(data_FS)
    #alpha1 = logslope(tnu[0], data_tot[])

  ax.plot(x, data_RS, c='r', **kwargs)
  ax.plot(x, data_FS, c='b', **kwargs)
  ax.plot(x, data_tot, c='k', **kwargs)

  if logx: ax.set_xscale('log')
  if logy: ax.set_yscale('log')
  if not ax_in:
    dummy_col = [plt.plot([], [], c=col)[0] for col in ['r', 'b', 'k']]
    ax.legend(dummy_col, ['RS', 'FS', 'RS+FS'])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()

def plot_spectralSlopes(key, logT, mFC=None, integrated=False, ax_in=None, **kwargs):
  '''
  Plot slopes of instantaneous spectra at log \bar{T} or of integrated spectra
  '''

  if not ax_in:
    fig, ax = plt.subplots()
  else:
    ax = ax_in

  nuobs, Tobs, env = radEnv(key)
  tnu = nuobs/env.nu0
  Tb = (Tobs-env.Ts)/env.T0
  i = find_closest(Tb, 10**logT)
  path = get_path_thinshellObs(key)
  logx = np.log10(tnu)

  xlabel, ylabel = '$\\tilde{\\nu}$', '${\\rm d}\\log(\\nu F_{\\nu})/{\\rm d}\\log\\nu$'
  mFC_RS, mFC_FS = False, False
  if mFC == 'RS': mFC_RS = True
  if mFC == 'FS': mFC_FS = True
  if mFC == 'both': mFC_RS, mFC_FS = True, True

  # nuFnu is of shape (len(T), len(nu))
  nuFnu_RS = np.load(path+'RS'+('mFC' if mFC_RS else '')+'.npy')
  nuFnu_FS = np.load(path+'FS'+('mFC' if mFC_FS else '')+'.npy')
  data_RS, data_FS = nuFnu_RS[i, :], nuFnu_FS[i, :]
  data_tot = data_RS + data_FS
  data_arr = [data_RS, data_FS, data_tot]
  for data, col in zip(data_arr, ['r', 'b', 'k']):
    logy = np.log10(data)
    sl = np.gradient(logy, logx)
    ax.semilogx(tnu, sl, c=col, **kwargs)
  
  if not ax_in:
    dummy_col = [plt.plot([], [], c=col)[0] for col in ['r', 'b', 'k']]
    ax.legend(dummy_col, ['RS', 'FS', 'RS+FS'])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()


def plot_any(var, key='Last', xscale='r', ax_in=None, logx=True, logy=True, **kwargs):
  '''
  Plots any variable that can be obtained by the get_var() func
  '''

  env = MyEnv(key)
  dir_path = GAMMA_dir + f'/results/{key}/'
  path = dir_path + 'thinshell_'

  if not ax_in:
    fix, ax = plt.subplots()
  else:
    ax = ax_in

  if xscale == 'r':
    xlabel = '$r/R_0$'
  elif xscale == 't':
    xlabel = '$t/t_0$'
  elif xscale == 'T':
    xlabel = '$\\bar{T}$'
  elif xscale == 'it':
    xlabel = 'it'
  
  for sh, col in zip(['RS', 'FS'], ['r', 'b']):
    data = pd.read_csv(path+sh+'.csv')
    data.attrs['key'] = key
    it, t, r = data[['it', 't', 'x']].to_numpy().transpose()
    y = get_variable(data, var, env)
    if xscale == 'r':
      x = r*c_/env.R0
    elif xscale == 't':
      x = t/env.t0 + 1.
    elif xscale == 'T':
      T = (1+env.z)*(t + env.t0 - r)
      x = (T-T[0])/env.T0
    elif xscale == 'it':
      x = it
    
    ax.plot(x, y, c=col, **kwargs)

  if logx:
    ax.set_xscale('log')
  if logy:
    ax.set_yscale('log')
  ax.set_xlabel(xlabel)
  ax.set_ylabel(var_exp[var])


############## IO functions
#### hydro
def get_path_hydro(key):
  dir_path = GAMMA_dir + f'/results/{key}/'
  path = dir_path + 'thinshellHydro_'
  return path

def open_hydro_data(key):
  '''
  Returns two dataframes corresponding to data behind shocks
  '''
  path = get_path_hydro(key)
  dataRS = pd.read_csv(path+'RS.csv')
  dataFS = pd.read_csv(path+'FS.csv')
  return dataRS, dataFS

def analyze_run_hydro(key, itmin=0, itmax=None):
  '''
  Extract hydro data after crossing for each front, writes in two separate files
  '''

  env = MyEnv(key)
  path = get_path_hydro(key)
  d0 = openData(key, 0)
  varlist = d0.keys()
  its = dataList(key, itmin, itmax)[0:]
  if itmin == 0:
    its = its[1:]
  Nf = len(its)
  varl = varlist.insert(0, 'it')
  varl = varl.insert(2, 'dt')
  dic_RS = {var:np.zeros(env.Nsh4) for var in varl}
  dic_FS = {var:np.zeros(env.Nsh1) for var in varl}

  istop = -1
  skips = 0
  calc_its = []
  bothCrossed = False

  # fill dicts with data behind shock fronts when new cell crossed
  jRS, jFS = 0, 0
  for it in its:
    df = openData(key, it)
    RScr, FScr = df_check_crossed(df)
    bothCrossed = (RScr and FScr)
    if bothCrossed:
      skips +=1
      if skips > 10 and (jRS>0 or jFS>0):
        break
      continue
    if not (it%1000):
      print(f'Opening file it {it}')
    RS, FS = df_get_cellsBehindShock(df)
    if not RS.empty:
      iRS = RS['i']
      if (iRS not in dic_RS['i']):
        dic_RS['it'][jRS] = it
        for var in varlist:
          dic_RS[var][jRS] = RS[var]
        jRS += 1
      else:
        continue
    if not FS.empty:
      iFS = FS['i']
      if iFS not in dic_FS['i']:
        dic_FS['it'][jFS] = it
        for var in varlist:
          dic_FS[var][jFS] = FS[var]
        jFS += 1
      else:
        continue
  
  # add corresponding dt
  dic_RS['dt'] = np.gradient(dic_RS['t'])
  dic_FS['dt'] = np.gradient(dic_FS['t'])

  RSout = pd.DataFrame.from_dict(dic_RS)
  RSout.to_csv(path + 'RS.csv', index=False)
  FSout = pd.DataFrame.from_dict(dic_FS)
  FSout.to_csv(path + 'FS.csv', index=False)

######## IO
def get_path_thinshellObs(key):
  dir_path = GAMMA_dir + f'/results/{key}/'
  path = dir_path + 'thinshellObs_'
  return path

def get_path_obs(key):
  dir_path = GAMMA_dir + f'/results/{key}/'
  path = dir_path + 'obs_'
  return path

def get_path_gmas(key):
  dir_path = GAMMA_dir + f'/results/{key}/'
  path = dir_path + 'gmas_'
  return path


def open_obs_data(key):
  '''
  Returns two dataframes corresponding to observed data behind shocks
  '''
  path = get_path_obs(key)
  dataRS = pd.read_csv(path+'RS.csv')
  dataFS = pd.read_csv(path+'FS.csv')
  return dataRS, dataFS

def get_paths_shThinshellObs(key, mFC_arr=[False, False]):
  path = get_path_thinshellObs(key)
  RSpath = path+'RS'+('mFC' if mFC_arr[0] else '')+'.npy'
  FSpath = path+'FS'+('mFC' if mFC_arr[1] else '')+'.npy'
  return RSpath, FSpath

def open_thinshellObs_data(key, mFC_arr=[False, False]):
  '''
  Returns two dataframes corresponding to observed data behind shocks in thinshell approx
  '''
  RSpath, FSpath = get_paths_shThinshellObs(key, mFC_arr)
  dataRS = np.load(RSpath)
  dataFS = np.load(FSpath)
  return dataRS, dataFS


def df_replaceShocked_withDownHydro(df, n=6, m=2):
  '''
  Returns dataframe giving shocked cells the hydro values downstream of them
  '''

  hdvars = ['dx', 'rho', 'vx', 'p', 'D', 'sx', 'tau']
  iCD = df.loc[df['trac']>1.5].index.min() - 1

  # identify shocked cells in data and group them by consecutive indices
  shockedCells = df.loc[df['Sd']==1.]
  indices = shockedCells.index.to_list()
  ilist = [list(group) for group in mit.consecutive_groups(indices)]
  shocks = [df.iloc[iarr] for iarr in ilist]

  # for each shock, identify downstream cell and replace values with it
  for sh in shocks:
    iL, iR = sh.index.min(), sh.index.max()
    rhoL, rhoR = df.at[iL-1, 'rho'], df.at[iR+1, 'rho']
    if rhoL < rhoR: # RS type shock
      i_f = iL
      i_d = min(iR + len(sh) + n, iCD - m) + 1
    else:  # FS type shock
      i_f = iR
      i_d = max(iL - len(sh) - n, iCD + m + 1)
    iL, iR = min(i_f, i_d), max(i_f, i_d)
    down_values = df.iloc[i_d][hdvars].to_numpy()
    indices = [i for i in range(iL, iR+1)]
    df.loc[indices, hdvars] = down_values
  return df

def analyze_hydro2rad(key, itmin=0, itmax=None, preinject=False):
  '''
  Analyze a run, evolving electron distribution and creating flux concurrently
  '''
  # idea: we need information on gma_min/max during a time step to
  # properly calculate emission. Since we do solve the evolution of
  # gmas during time step with odeint, let's directly use the 'sol'
  # output for gma_min/max as borns for the integral

  nuobs, Tobs, env = radEnv(key)
  Lnu_out = np.zeros((len(Tobs), len(nuobs)))
  path = get_path_gmas(key)
  its = dataList(key, itmin, itmax, itstep)

  dfm = openData(key, its[0])
  dfm['gmin_'] = 0.
  dfm['gmax_'] = 0.
  varlist = dfm.keys().to_list()
  varlist.insert(1, 'dt')
  varlist.insert(varlist.index('rho'), 'drho')
  varlist_full = ['it'] + varlist
  dic_RS, dic_FS = [{var:[] for var in varlist_full} for i in range(2)]

  for j in range(Nit-1):
    it, itp = its[j], its[j+1]
    df = openData(key, it)

    # hydro values in shocked cells are those downstream
    df = df_replaceShocked_withDownHydro(df)

    df['it'] = it
    dfp = openData(key, itp)
    RScr, FScr = df_check_crossed(df)
    bothCrossed = (RScr and FScr)
    if bothCrossed:
      skips +=1
      if skips > 10 and ((len(dic_RS['t'])>0 or len(dic_FS['t']))):
        print(f'Ending on file {it}')
        break
      continue
    if not it%1000:
      print(f'Opening file it {it}')

    # adding delta rho and delta t to reconstruct derivative from central difference
    df['drho'] = df['rho'].to_numpy() - dfm['rho'].to_numpy()
    df['dt'] = df['t'].to_numpy() - dfm['t'].to_numpy()
    
    if preinject:
    # inject e- in dfm to make them evolve properly between t_inj and t_j
    # open next file, use trac2 rate of decay to determine proper t_inj
      dfp = openData(key, its[j+1])
      RS, FS = df_get_cellsBehindShock(df)
      for cell, dic in zip([RS, FS], [dic_RS, dic_FS]):
        if not cell.empty:
          k = int(cell['i'])
          if dfm.iloc[k]['trac2'] <= 0.1: # only inject in cells shocked for the 1st time
            if (df.attrs['it'] - dfm.attrs['it']) > 1:
              cellp = dfp.iloc[k]
              dtracdt = (cellp['trac2'] - cell['trac2'])/(cellp['t'] - cell['t'])
              dt = (cell['trac2']-1)/dtracdt
              tinj = cell['t'] - dt
              dfm.at[k, 't'] = tinj
              dfm.at[k, 'x'] = dfm.at[k, 'x'] + dfm.at[k, 'vx']*(tinj - dfm.at[k, 't'])
              dfm.at[k, 'dt'] = dt
            inj_gmin, inj_gmax = get_vars(cell, ['gma_m', 'gma_M'], env)
            dfm.at[k, 'gmin_'] = inj_gmin
            dfm.at[k, 'gmax_'] = inj_gmax

            # add values to dic
            #dic['it'].append(it)
            for var, arr in dic.items():
              dic[var].append(dfm.at[k, var])

    # evolve electron distribution where it exists
    ev_cells = dfm.loc[dfm['gmax_'] > 1.]
    gmin, gmax = np.zeros((2, len(dfm.index)))
    if not ev_cells.empty:
      for k, cell in ev_cells.iterrows():
        # cool local electron distribution
        cell_new = df.iloc[k]
        cell_Lnu, gmas_new = step2rad_integ(cell, cell_new, nuobs, Tobs, env, finalLF=True)
        Lnu_out += cell_Lnu
        gmin[k] += gmas_new[0]
        gmax[k] += gmas_new[1]

    # inject electron ditribution in shocked cells
    sh_cells = df.loc[(df['Sd']==1.)]
    # below is good function
    if not preinject:
      if GAMMAtest:
        for k, cell in sh_cells.iterrows():
          inj_gmin, inj_gmax = get_vars(cell, ['gma_m', 'gma_M'], env)
          gmin[k] += inj_gmin
          gmax[k] += inj_gmax
      else:
        RS, FS = df_get_cellsBehindShock(df)
        for cell in [RS, FS]:
          if not cell.empty:
            k = int(cell['i'])
            if dfm.iloc[k]['trac2'] <= 0.1: # only inject in cells shocked for the 1st time
              inj_gmin, inj_gmax = get_vars(cell, ['gma_m', 'gma_M'], env)
              gmin[k] += inj_gmin
              gmax[k] += inj_gmax

    # add that to the dataframe
    df['gmin_'] = gmin
    df['gmax_'] = gmax

    # extract data
    sh_RS = sh_cells.loc[sh_cells['trac']<1.5]
    iRS = env.Next if sh_RS.empty else sh_RS['i'].min()
    RScells = df.iloc[iRS:].loc[(df['gmax']>1.01) | (df['gmax_']>1.01)].loc[(df['trac']<1.5)]
    for k, cell in RScells.iterrows():
      dic_RS['it'].append(it)
      for var in varlist:
        dic_RS[var].append(cell[var])

    sh_FS = sh_cells.loc[sh_cells['trac']>1.5]
    iFS = env.Ntot-env.Next if sh_FS.empty else sh_FS['i'].max()
    FScells = df.iloc[:iFS+1].loc[(df['gmax']>1.01) | (df['gmax_']>1.01)].loc[(df['trac']>1.5)]
    for k, cell in FScells.iterrows():
      dic_FS['it'].append(it)
      for var in varlist:
        dic_FS[var].append(cell[var])
    
    dfm = df.copy()
      
  RSout = pd.DataFrame.from_dict(dic_RS)
  path_RS = path+'RS.csv'
  RSout.to_csv(path_RS, index=False)
  FSout = pd.DataFrame.from_dict(dic_FS)
  path_FS = path+'FS.csv'
  FSout.to_csv(path_FS, index=False)


def analyze_run_obsthinshell(key, mFC_arr=[False, False], func='Band', itmin=0, itmax=None):
  '''
  Analyzes a run and creates the observed radiation in the thin shell approximation
  mFC_arr array of bool for applying marg. FC to RS or FS respectively
  '''

  # thinshell means we can extract the hydro first before analyzing
  path_hydro = get_path_hydro(key)
  RS_exists = os.path.exists(path_hydro+'RS.csv')
  FS_exists = os.path.exists(path_hydro+'FS.csv')
  if not (RS_exists and FS_exists):
    analyze_run_hydro(key, itmin, itmax)
  RS_hydro, FS_hydro = open_hydro_data(key)

  RSpath, FSpath = get_paths_shThinshellObs(key, mFC_arr)
  nuobs, Tobs, env = radEnv(key)
  tnu = nuobs/env.nu0

  for sh, data, path, mFC_bool in zip(
        ['RS', 'FS'], [RS_hydro, FS_hydro], [RSpath, FSpath], mFC_arr):
    #arr = sh_hydro.to_numpy()
    print('Generating contribution from ' + sh +
      (' with marginally fast cooling' if mFC_bool else ''))
    nuFnu = np.zeros((len(Tobs), len(nuobs)))
    #for cell in np.nditer(arr):
    for k, cell in data.iterrows():
      nuFnu += tnu * get_Fnu_thinshell(cell, nuobs, Tobs, env, func=func, mFC=mFC_bool)/env.F0
    np.save(path, nuFnu, allow_pickle=False)

def analyze_extract_distrib(key, itmin=0, itmax=None, itstep=None,
    preinject=False, GAMMAtest=False, noadiab=False):
  '''
  Read run, add electron distribution and extract relevant cells for emission
  '''

  env = MyEnv(key)
  path = get_path_gmas(key)
  its = dataList(key, itmin, itmax, itstep)
  both = False
  if GAMMAtest: preinject=False
  func_cool = func_cooling_synThin
  if noadiab:
    print('Adiabatic cooling deactivated')
    func_cool = func_cooling_synThin_noadiab
    pureSyn = True

  istop = -1
  skips = 0
  Nit = len(its[1:])
  bothCrossed = False

  dfm = openData(key, its[0])
  dfm['gmin_'] = 0.
  dfm['gmax_'] = 0.
  varlist = dfm.keys().to_list()
  varlist.insert(1, 'dt')
  varlist.insert(varlist.index('rho')+1, 'drho')
  varlist_full = ['it'] + varlist
  dic_RS, dic_FS = [{var:[] for var in varlist_full} for i in range(2)]

  for j in range(Nit-1):
    it, itp = its[j], its[j+1]
    df = openData(key, it)

    # hydro values in shocked cells are those downstream
    df = df_replaceShocked_withDownHydro(df)

    df['it'] = it
    dfp = openData(key, itp)
    RScr, FScr = df_check_crossed(df)
    bothCrossed = (RScr and FScr)
    if bothCrossed:
      skips +=1
      if skips > 10 and ((len(dic_RS['t'])>0 or len(dic_FS['t']))):
        print(f'Ending on file {it}')
        break
      continue
    if not it%1000:
      print(f'Opening file it {it}')

    # adding delta rho and delta t to reconstruct derivative from central difference
    df['drho'] = df['rho'].to_numpy() - dfm['rho'].to_numpy()
    df['dt'] = df['t'].to_numpy() - dfm['t'].to_numpy()
    
    if preinject:
    # inject e- in dfm to make them evolve properly between t_inj and t_j
    # open next file, use trac2 rate of decay to determine proper t_inj
      dfp = openData(key, its[j+1])
      RS, FS = df_get_cellsBehindShock(df)
      for cell, dic in zip([RS, FS], [dic_RS, dic_FS]):
        if not cell.empty:
          k = int(cell['i'])
          if dfm.iloc[k]['trac2'] <= 0.1: # only inject in cells shocked for the 1st time
            if (df.attrs['it'] - dfm.attrs['it']) > 1:
              cellp = dfp.iloc[k]
              dtracdt = (cellp['trac2'] - cell['trac2'])/(cellp['t'] - cell['t'])
              dt = (cell['trac2']-1)/dtracdt
              tinj = cell['t'] - dt
              dfm.at[k, 't'] = tinj
              dfm.at[k, 'x'] = dfm.at[k, 'x'] + dfm.at[k, 'vx']*(tinj - dfm.at[k, 't'])
              dfm.at[k, 'dt'] = dt
            inj_gmin, inj_gmax = get_vars(cell, ['gma_m', 'gma_M'], env)
            dfm.at[k, 'gmin_'] = inj_gmin
            dfm.at[k, 'gmax_'] = inj_gmax

            # add values to dic
            #dic['it'].append(it)
            for var, arr in dic.items():
              dic[var].append(dfm.at[k, var])

    # evolve electron distribution where it exists
    ev_cells = dfm.loc[dfm['gmax_'] > 1.]
    gmin, gmax = np.zeros((2, len(dfm.index)))
    if not ev_cells.empty:
      for k, cell in ev_cells.iterrows():
        # cool local electron distribution
        cell_new = df.iloc[k]
        gmas_new = evolve_gmas(cell, cell_new, func_cool, env, pureSyn)
        gmin[k] += gmas_new[0]
        gmax[k] += gmas_new[1]


    # inject electron ditribution in shocked cells
    sh_cells = df.loc[(df['Sd']==1.) & (df['trac']>0.5)]
    # below is good function
    if not preinject:
      if GAMMAtest:
        for k, cell in sh_cells.iterrows():
          inj_gmin, inj_gmax = get_vars(cell, ['gma_m', 'gma_M'], env)
          gmin[k] += inj_gmin
          gmax[k] += inj_gmax
      else:
        RS, FS = df_get_cellsBehindShock(df)
        for cell in [RS, FS]:
          if not cell.empty:
            k = int(cell['i'])
            if dfm.iloc[k]['trac2'] <= 0.1: # only inject in cells shocked for the 1st time
              inj_gmin, inj_gmax = get_vars(cell, ['gma_m', 'gma_M'], env)
              gmin[k] += inj_gmin
              gmax[k] += inj_gmax

    # add that to the dataframe
    df['gmin_'] = gmin
    df['gmax_'] = gmax

    # extract data
    sh_RS = sh_cells.loc[sh_cells['trac']<1.5]
    iRS = env.Next if sh_RS.empty else sh_RS['i'].min()
    RScells = df.iloc[iRS:].loc[(df['gmax']>1.01) | (df['gmax_']>1.01)].loc[(df['trac']<1.5)]
    for k, cell in RScells.iterrows():
      dic_RS['it'].append(it)
      for var in varlist:
        dic_RS[var].append(cell[var])

    sh_FS = sh_cells.loc[sh_cells['trac']>1.5]
    iFS = env.Ntot-env.Next if sh_FS.empty else sh_FS['i'].max()
    FScells = df.iloc[:iFS+1].loc[(df['gmax']>1.01) | (df['gmax_']>1.01)].loc[(df['trac']>1.5)]
    for k, cell in FScells.iterrows():
      dic_FS['it'].append(it)
      for var in varlist:
        dic_FS[var].append(cell[var])
    
    dfm = df.copy()
      
  RSout = pd.DataFrame.from_dict(dic_RS)
  path_RS = path+'RS.csv'
  RSout.to_csv(path_RS, index=False)
  FSout = pd.DataFrame.from_dict(dic_FS)
  path_FS = path+'FS.csv'
  FSout.to_csv(path_FS, index=False)

def plot_test_methods_cell(n=-10, var='sp', logv=0, key='Last'):

  modes = ['cool_int', 'vFC']
  xlabel, ylabel = '', '$\\nu F_{\\nu}/\\nu_0F_0$'
  logx, logy = False, False

  nuobs, Tobs, env = radEnv(key)
  tnu = nuobs/env.nu0
  Tb = (Tobs-env.Ts)/env.T0
  dk = get_gmasdata_cell(n, key)
  
  if var == 'lc':
    i = find_closest(tnu, 10**logv)
    nuobs = np.array([nuobs[i]])
    x = Tb
    xlabel=("$\\bar{T}$")
  elif var == 'sp':
    i = find_closest(Tb, 10**logv)
    Tobs = np.array([Tobs[i]])
    x = tnu
    xlabel=("$\\tilde{\\nu}$")
    logx, logy = True, True



def distrib2rad(key, mode='cool_int', output=False):
  '''
  Read the file of extracted electron distribution 
    to generate the observed emission
    mode = 'cool', 'plaw', 'cool_int', 'vFC'
  '''
  nuobs, Tobs, env = radEnv(key)
  tnu = nuobs/env.nu0
  nuFnus = np.zeros((2, len(Tobs), len(nuobs)))


  for i, sh in enumerate(['RS', 'FS']):
    path = GAMMA_dir + f'/results/{key}/gmas_{sh}.csv'
    outpath = GAMMA_dir + f'/results/{key}/coolingObs{'vFC' if forcevFC else ''}_{sh}'
    data = pd.read_csv(path)
    k_list = list(dict.fromkeys(data['i'].astype(int).to_list()))
    for k in k_list:
      k = int(k)
      dk = data.loc[data['i']==k]
      if mode == 'vFC':
        nuFnu_k = cell2rad(dk, nuobs, Tobs, env, mode=mode)
      elif mode == 'cool':
        nuFnu_k = cell2rad(dk, nuobs, Tobs, env, mode=mode)
      elif mode == 'plaw':
        nuFnu_k = cell2rad(dk, nuobs, Tobs, env, mode=mode)
      else:
        nuFnu_k = cell2rad_integ(dk, nuobs, Tobs, env, allsteps=False)
      nuFnus[i] += nuFnu_k

    np.save(outpath, nuFnus[i], allow_pickle=False)

  if output:
    return nuFnus

def cell2rad_new(dk, nuobs, Tobs, env, withsteps=False):
  '''
  Contribution to observed flux from 1 cell, can return array of contributions with regimes
  '''
  tnu = nuobs/env.nu0
  if (type(Tobs)==np.ndarray) and (type(nuobs)==np.ndarray):
    out = np.zeros((len(Tobs), len(nuobs)))
  elif (type(Tobs)==np.ndarray):
    out = np.zeros(Tobs.shape)
  elif (type(nuobs)==np.ndarray):
    out = np.zeros(nuobs.shape)

  Nj = len(dk)
  cell_0 = dk.iloc[0]
  nF_arr, reg_arr = [], []
  for j in range(Nj-1):
    cell = dk.iloc[j]
    cell_next = dk.iloc[j+1]
    if cell_next['gmax_'] == 0.:
      break
    Fnu, reg = get_Fnu_cooling(cell, cell_next, cell_0, nuobs, Tobs, env,
      refLF='mid', onlyInt=False)
    nF = tnu * Fnu /env.F0
    nF_arr.append(nF)
    reg_arr.append(reg)
    out += nF
  
  nF_arr = np.array(nF_arr)
  if withsteps:
    return nF_arr, reg_arr
  else:
    return nF_arr.sum(axis=0)
  

def cell2rad(dk, nuobs, Tobs, env, mode='cool', refLF='mid', onlyInt=False, withsteps=False):
  '''
  Contribution to observed flux from 1 cell 
  '''
  tnu = nuobs/env.nu0
  if (type(Tobs)==np.ndarray) and (type(nuobs)==np.ndarray):
    out = np.zeros((len(Tobs), len(nuobs)))
  elif (type(Tobs)==np.ndarray):
    out = np.zeros(Tobs.shape)
  elif (type(nuobs)==np.ndarray):
    out = np.zeros(nuobs.shape)

  Nj = len(dk)
  cell_0 = dk.iloc[0]
  nF_arr, reg_arr = [], []
  for j in range(Nj-1):
    cell = dk.iloc[j]
    cell_next = dk.iloc[j+1]
    if cell_next['gmax_'] == 0.:
      break
    if mode == 'vFC':
      Fnu = get_Fnu_cooling_forcevFC(cell, nuobs, Tobs, env)
      reg = ''
    elif mode == 'cool':
      Fnu, reg = get_Fnu_cooling(cell, cell_next, cell_0, nuobs, Tobs, env, refLF=refLF, onlyInt=onlyInt)
    elif mode == 'plaw':
      Fnu, reg = get_Fnu_cooling_pureplaws(cell, cell_next, nuobs, Tobs, env)
    
    if (len(Tobs) == 1) or (len(nuobs) == 1):
      out = out.flatten()
      Fnu = Fnu.flatten()
    nF = tnu * Fnu/env.F0
    out += nF
    nF_arr.append(nF)
    reg_arr.append(reg)


  if withsteps:
    return out, np.array(nF_arr), reg_arr
  else:
    return out

def test_Fnu_coolsteps(key='test'):
  n, var, logv, istep = -10, 'sp', 0, 0
  nuobs, Tobs, env = radEnv(key)
  tnu = nuobs/env.nu0
  Tb = (Tobs-env.Ts)/env.T0
  i = find_closest(Tb, 10**logv)
  Tobs = np.array([Tobs[i]])
  dk = get_gmasdata_cell(n, key)

  tot, steps = cell2rad_integ(dk, nuobs, Tobs, env, allsteps=True)
  ts = tnu * get_Fnu_thinshell(dk.iloc[0], nuobs, Tobs, env, func='plaw')/env.F0
  x = tnu
  y_arr = np.array([ts.flatten(), tot.flatten()])

  fig, ax = plt.subplots()
  for y, ls in zip(y_arr, ['--', '-']):
    ax.loglog(x, y, c='r', ls=ls, zorder=2)
  for step in steps:
    ax.loglog(x, step.flatten(), c='magenta', lw=.9, zorder=1)

  ax.set_xlabel('$\\tilde{\\nu}$')
  ax.set_ylabel('$\\nu F_{\\nu}/\\nu_0 F_0$')



def cell2rad_integ(dk, nuobs, Tobs, env, allsteps=False):
  '''
  Radiation from 1 cell when performing the integral over the LF distrib during the time step
  allsteps = True outputs an array of the time steps contributions 
  '''

  out = np.zeros((len(Tobs), len(nuobs)))
  tnu = nuobs/env.nu0
  outsteps = []
  Nj = len(dk)
  for j in range(Nj-1):
    cell, cell_new = dk.iloc[j], dk.iloc[j+1]
    if cell_new['gmax_'] == 0.:
      break
    nuLnu = step2rad_integ(cell, cell_new, nuobs, Tobs, env)
    nuLnu *= tnu
    outsteps.append(nuLnu)
  outsteps = np.array(outsteps)
  if allsteps:
    return outsteps.sum(axis=0), outsteps
  else:
    return outsteps.sum(axis=0)


def step2rad_integ(cell, cell_new, nuobs, Tobs, env, finalLF=False,
    cooling='syn_th'):
  '''
  Luminosity from a given cell from its initial state to cell_new, normalized
  cooling = 'syn_th' for analytical, 
  '''
  Lnu = np.zeros((len(Tobs), len(nuobs)))
  gm, gM = cell[['gmin_', 'gmax_']]
  if cooling == 'syn_th':
    tmin = min(cell['t'], cell_new['t'])
    tmax = max(cell['t'], cell_new['t'])
    tc = 1/get_variable(cell, 'syn', env)
    lfac = get_variable(cell, 'lfac', env)
    lfac_new = get_variable(cell_new, 'lfac', env)
    lfac_n = .5*(lfac+lfac_new)
    tmin /= lfac_n
    tmax /= lfac_n
    # tmin /= lfac
    # tmax /= lfac_new
    t_span = [tmin, tmax]
    def func_gmin(t):
      tt = (t-tmin)/tc
      return gm/(1+gm*tt)
    def func_gmax(t):
      tt = (t-tmin)/tc
      return gM/(1+gM*tt)
    gm_new = func_gmin(tmax)
    gM_new = func_gmax(tmax)

  else:
    # solve cooling, withSol to get the dense output necessary for integration 
    sol_gmin, sol_gmax = solve_cooling_gmas(cell, cell_new, func_cool, env, withSol=True)
    func_gmin, func_gmax = sol_gmin.sol, sol_gmax.sol
    t_span = (sol_gmax.t[0], sol_gmax.t[-1])
    gm_next = sol_gmin.y[0][-1]
    gM_next = sol_gmax.y[0][-1]

  # values for syn flux
  p = env.psyn
  nup_B = get_variable(cell, "nup_B", env)
  Pe = get_variable(cell, 'Pmax', env)
  V3 = get_variable(cell, 'V3p', env)
  tc = 1/get_variable(cell, 'syn', env)
  Ton, Tth, Tej = get_variable(cell, 'obsT', env)
  Dop = get_variable(cell, 'Dop', env)
  tT = ((Tobs - Tej)/Tth)
  tnu = nuobs/(Dop*nup_B) # normalize to syn frequency
  tT = tT[:, np.newaxis]
  tnu = tnu[np.newaxis, :]
  Lnu_m = 1.

  tTtnu = tT*tnu
  # syn flux calculation
  if (gM/gm<=2):
    # monoE
    xB = gM_new**-2
    xM = (gM_new/gM)**2
    tnu *= gM**-2 # normalize to nu'_M
    Lnu_m *= (3/5.) * Pe * tc * V3 * (1+env.z) / (gM * Tth * env.L0)
    Lnu += mixed_plaw(tTtnu, xB, xM)
    Lnu *= tT**-2 
    Lnu *= Lnu_m

  else:
    # cooling from a plaw
    K = (p-1)*gm**(p-1)
    Lnu_m *= K * Pe * tc  *V3 * (1+env.z) / (Tth * env.L0) # Pe already contains ne
    # Lnu += syn_rad_fromCooling(tnu, t_span, func_gmin, func_gmax, tc, p)
    
    def func_integ(gma, t):
      tt = (t-tmin)/tc
      return gma**(-p-2/3.) * (1-gma*tt)**(p-2)

    
    sqtnu_arr = np.sqrt(tTtnu)
    rows, cols = Lnu.shape
    for i in range(0, rows):
      for j in range(0, cols):
        sqtnu = sqtnu_arr[i,j]
        def gma_min(t):
          return max(sqtnu, func_gmin(t))
        Lnu[i,j] += spi.dblquad(func_integ, tmin, tmax, gma_min, func_gmax)[0]
    Lnu *= tTtnu**(1/3)

    Lnu *= tT**-2 
    Lnu *= Lnu_m
  
  if finalLF:
    return Lnu, [gm_new, gM_new]
  else:
    return Lnu

def get_base_gmacgmam(key, n):
  env = MyEnv(key)
  dk = get_gmasdata_cell(n, key)
  cell_0 = dk.iloc[0]
  lfac = get_variable(cell_0, 'lfac', env)
  gmam = dk.iloc[0]['gmin_']
  gmac = env.R0/(lfac*c_)
  gmaM = dk.iloc[0]['gmax_']
  return np.log10(gmac/gmam)

def data_rescaling(logRatio, jmax=150, key='test', n=-10, nt=5):
  '''
  Rescaling part of test_coolingregimes
  time resolution is nt steps per cooling time
  !! add rescaling to env !
  '''
  env = MyEnv(key)
  dk = get_gmasdata_cell(n, key)

  if not logRatio:
    return dk, 0.

  varlist = ['it', 't', 'dt', 'x', 'dx', 'rho', 'vx', 'p']
  cell_0 = dk.iloc[0][varlist].copy()
  
  # find necessary rescaling value
  lfac = get_variable(cell_0, 'lfac', env)
  tc = 1/get_variable(cell_0, 'syn', env)
  tdyn = env.R0/(lfac*c_)
  gmam = dk.iloc[0]['gmin_']
  gmac = tc/tdyn
  #gmac = nt  # if local dyn. time is t_c/nt, gma_c becomes our time resolution
  gmaM = dk.iloc[0]['gmax_']
  scalefac = logRatio - np.log10(gmac/gmam)
  scaling = 10**scalefac
  print(f'rescaling by factor {scaling:.1e}')

  cell_0.loc['dt'] = lfac*(tc/nt) 
  # rescale with geometrical scalings
  cell_0.loc['t'] *= scaling
  cell_0.loc['dt'] *= scaling
  cell_0.loc['x'] *= scaling
  cell_0.loc['dx'] *= scaling
  cell_0.loc['rho'] /= scaling**2
  cell_0.loc['p'] /= scaling**2
  gmaM *= np.sqrt(scaling)

  # evolve cell with constant velocity
  def evolve_cell_hydro(cell, env):
    cell_out = cell.copy()
    cell_out[0] += 1  # it +=1
    cell_out[1] += cell[2]  # t += dt
    cell_out[3] += cell[2]*cell[6]  # r += v*dt
    rscale = cell_out[3]/cell[3]
    cell_out[5] *= rscale**-2   # rho *= (r/r_i)**-2
    cell_out[7] *= rscale**-2   # p *= (r/r_i)**-2
    return cell_out

  init = cell_0[varlist].to_numpy()
  cell = init.copy()
  out = [cell]
  j = 0
  while (cell[-1]>1.) or (j<jmax):
    cell_new = evolve_cell_hydro(cell, env)
    out.append(cell_new)
    cell = cell_new
    j += 1
  Nj = len(out)

  data = pd.DataFrame(out, columns=varlist)
  rho = data['rho'].to_numpy()
  drho = np.gradient(rho)
  data.insert(5, 'drho', drho)

  # evolve gmas
  gmin, gmax = np.zeros((2, Nj))
  gmin[0] += gmam
  gmax[0] += gmaM
  data.insert(7, 'gmin_', gmin)
  data.insert(8, 'gmax_', gmax)
  cell = data.iloc[0]
  for j in range(1, Nj):
    cell, cell_new = data.iloc[j-1], data.iloc[j] 
    gmin_new, gmax_new = evolve_gmas(cell, cell_new, func_cooling_synThin, env, pureSyn=True)
    data.at[j, 'gmin_'] = gmin_new
    data.at[j, 'gmax_'] = gmax_new

  return data, scalefac

def testplot_instantPnu(logRatio, jmax=1000, key='test',
    n=-10, var='sp', logv=0, Np=20):

  data, scalefac = data_rescaling(logRatio, jmax=jmax, key=key, n=n)
  nuobs0, Tobs0, env0 = radEnv(key, lowfreqs=True)
  nuobs, Tobs, env = radEnv(key, lowfreqs=True, scalefac=scalefac)
  nuobs = nuobs0
  Tobs = Tobs0
  tnu = nuobs0/env0.nu0
  Tb = (Tobs0-env0.Ts)/env0.T0

  xlabel, ylabel = '', "$P'_{\\nu'}/P_0$"
  logx, logy = False, False
  col = 'r' if n < 0 else 'b'

  if var == 'lc':
    i = find_closest(tnu, 10**logv)
    nuobs = np.array([nuobs[i]])
    x = Tb
    xlabel=("$\\bar{T}$")
  elif var == 'sp':
    i = find_closest(Tb, 10**logv)
    Tobs = np.array([Tobs[i]])
    x = tnu
    xlabel=("$\\tilde{\\nu}$")
    logx, logy = True, True
  
  fig, ax = plt.subplots()

  Nj = len(data)
  P0 = get_variable(data.iloc[0], 'Pmax', env)
  for j in range(0, Nj, int(Nj/Np)):
    cell = data.iloc[j]
    Dop = get_variable(cell, 'Dop', env)
    nup = nuobs0/Dop
    Pnu = instant_Pnu_bplaw(nup, cell, env)/P0
    ax.plot(nup/env.nu0p, Pnu, c=col, lw=.8, alpha=.9)
  
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  if logx: ax.set_xscale('log')
  if logy: ax.set_yscale('log')

 

def instant_Pnu_bplaw(nup, cell, env):
  '''
  Instantaneous spectrum of a power-law distrib of e- over comoving frequency
  '''

  p = env.psyn
  def shape(gmin, gmax, p):
    return (gmin**(1/3 - p) - gmax**(1/3 - p))/(gmin**(1 - p) - gmax**(1 - p))

  nusyn = get_variable(cell, 'nup_B', env)
  tnu = nup/nusyn
  Pe = get_variable(cell, 'Pmax', env)
  gmin, gmax = cell[['gmin_', 'gmax_']]

  num = gmin**2
  nuM = gmax**2
  seg1 = (tnu <= num)
  seg2 = (tnu > num) & (tnu <= nuM)
  S = np.zeros(tnu.shape)
  S[seg1] = shape(gmin, gmax, p)
  S[seg2] = shape(np.sqrt(tnu)[seg2], gmax, p)
  S *= tnu**(1/3)

  return (3*(p-1)/(3*p-1)) * Pe * S

def test_coolingregime(logRatio, jmax=1000, key='test', n=-10, var='sp', logv=0,
  withsteps=False, refLF='mid', onlyInt=False, ax_in=None, **kwargs):

  '''
  Test radiation method by rescaling to a given gma_c/gma_m = 10**logRatio
  rescaling is done analytically, assuming pure geometric effects & constant v
  electrons are then evolved along these rescaled hydro,
    stopping when gmax = 1 or after jmax steps
  '''
  data, scalefac = data_rescaling(logRatio, jmax=jmax, key=key, n=n)
  Nj = len(data)

  nuobs0, Tobs0, env0 = radEnv(key, lowfreqs=True)
  nuobs, Tobs, env = radEnv(key, lowfreqs=True, scalefac=scalefac)

  nuobs = nuobs0
  Tobs = Tobs0
  tnu = nuobs0/env0.nu0
  Tb = (Tobs0-env0.Ts)/env0.T0

  xlabel, ylabel = '', '$\\nu F_{\\nu}/\\nu_0F_0$'
  logx, logy = False, False
  col = 'r' if n < 0 else 'b'
  
  if var == 'lc':
    i = find_closest(tnu, 10**logv)
    nuobs = np.array([nuobs[i]])
    x = Tb
    xlabel=("$\\bar{T}$")
  elif var == 'sp':
    i = find_closest(Tb, 10**logv)
    Tobs = np.array([Tobs[i]])
    x = tnu
    xlabel=("$\\tilde{\\nu}$")
    logx, logy = True, True


  # produce emission

  # cell_0 = data.iloc[0]
  # nuFnu_arr, reg_arr = [], []
  # for j in range(1, Nj):
  #   cell, cell_new = data.iloc[j-1], data.iloc[j] 
  #   Fnu, reg = get_Fnu_cooling(cell, cell_new, cell_0, nuobs, Tobs, env,
  #       refLF=refLF, onlyInt=onlyInt)
  #   nuFnu = tnu*Fnu.flatten()/env.F0 # flatten because of shape (1,N) or (N,1)
  #   nuFnu_arr.append(nuFnu)
  #   reg_arr.append(reg)
  # nuFnu_arr = np.array(nuFnu_arr)
  # nuFnu_tot = np.sum(nuFnu_arr, axis=0)
  # rad = cell2rad_new(data, nuobs, Tobs, env, withsteps=withsteps)
  #rad2 = cell2rad(data, nuobs, Tobs, env0, mode='cool', refLF=refLF, onlyInt=False)
  #rad2 = rad2.flatten()
  
  nuFnu_tot, nuFnu_arr, reg_arr = cell2rad(data, nuobs, Tobs, env,
    mode='cool', refLF=refLF, onlyInt=False, withsteps=True)


  # plot
  if not ax_in:
    fig, ax = plt.subplots()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
  else:
    fig = plt.gcf()
    ax = ax_in
    # deactivate individual timesteps when comparing

  line, = ax.plot(x, nuFnu_tot, zorder=-1, **kwargs)
  #ax.plot(x, rad2, ls=':')
  if not ax_in:
    line.set_color(col)
  if logx: ax.set_xscale('log')
  if logy: ax.set_yscale('log')
  if withsteps:
    regimes = []
    for reg in reg_arr:
      if (reg and (reg not in regimes)): regimes.append(reg)
    regcol = {'fast':'m', 'slow':'c', 'monoE':'g'}
    dummy_col = [ax.plot([],[],c=regcol[reg])[0] for reg in regimes]
    if 'monoE' in regimes:
      dummy_col.append(ax.plot([],[],c='darkgreen')[0])
      ax.legend(dummy_col, regimes+['$\\Sigma$monoE'], loc='lower right')
    sum_mono = np.zeros(nuFnu_tot.shape)
    for nuFnu, reg in zip(nuFnu_arr, reg_arr):
      nuFnu = nuFnu
      if reg == 'monoE':
        sum_mono += nuFnu
        plt.autoscale(enable=False)
      ax.plot(x, nuFnu, lw=.9, c=regcol[reg], **kwargs)
    ax.plot(x, sum_mono, c='darkgreen', **kwargs)
  fig.tight_layout()

def compare_coolregimes(logs, key='test', n=-10, var='sp', logv=0, refLF='mid', onlyInt=True):

  fig, ax = plt.subplots()
  base = get_base_gmacgmam(key, n)
  ncol = len(logs)
  colors = plt.cm.jet(np.linspace(0,1,ncol))
  names = [f'{logRatio:.1f}' for logRatio in [base if v is None else v for v in logs]]
  dummy_col = []
  xlabel, ylabel = '', '$\\nu F_{\\nu}/\\nu_0F_0$'
  
  if var == 'lc':
    xlabel=("$\\bar{T}$")
  elif var == 'sp':
    xlabel=("$\\tilde{\\nu}$")

  for i, logRatio in enumerate(logs):
    test_coolingregime(logRatio, key=key, n=n, var=var, logv=logv,
      withsteps=False, onlyInt=onlyInt, ax_in=ax, ls='-', c=colors[i])
    dummy_col.append(plt.plot([],[],c=colors[i],ls='-')[0])
  
  ax.legend(dummy_col, names,
    title='$\\log_{10}\\gamma_{\\rm c}/\\gamma_{\\rm m}$')
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)

  fig.tight_layout()

########## intermediate calculations

def stop_condition(t, y, rho, drhodt, B2):
  return y[0]-1.

stop_condition.terminal = True

def solve_cooling_gmas(cell, cell_new, func_cool, env, withSol):
  tmin = min(cell['t'], cell_new['t']) + env.t0
  tmax = max(cell['t'], cell_new['t']) + env.t0
  gmin0, gmax0 = cell[['gmin_', 'gmax_']]
  B2 = env.eps_B * 8*pi_ * get_variable(cell, 'ei', env)
  lfac = get_variable(cell, 'lfac', env)
  lfac_new = get_variable(cell_new, 'lfac', env)
  lfac_n = .5*(lfac+lfac_new)
  # t_span = (tmin/lfac_n, tmax/lfac_n)
  t_span = (tmin/lfac, tmax/lfac_new)
  dtp = .5*(cell['dt']/lfac_n + cell_new['dt']/lfac_n)
  drhodt = (cell['drho']+cell_new['drho'])/dtp
  # if cell['i'] == 450: print((drhodt/(3*cell['rho'])))
  sol_gmin = spi.solve_ivp(func_cool, t_span, [gmin0], method='RK23',
      dense_output=withSol, events=stop_condition, args=(cell['rho'], drhodt, B2))
  sol_gmax = spi.solve_ivp(func_cool, t_span, [gmax0], method='RK23',
      dense_output=withSol, events=stop_condition, args=(cell['rho'], drhodt, B2))
  return sol_gmin, sol_gmax

def evolve_gmas(cell, cell_new, func_cool, env, pureSyn=False):
  '''
  Evolve the electron distribution between two hydro time steps
  cell already contains gma_min and gma_max
  Coolings: optically thin synchrotron and adiabatic
  '''
  if pureSyn:
    gm, gM = cell[['gmin_', 'gmax_']]
    tc = 1/get_variable(cell, 'syn', env)
    lfac = get_variable(cell, 'lfac', env)
    lfac_new = get_variable(cell_new, 'lfac', env)
    lfac_n = .5*(lfac+lfac_new)
    tt = (cell_new['t']-cell['t'])/(lfac_n*tc)
    gm_new = gm/(1+gm*tt)
    gM_new = gM/(1+gM*tt)
  else:
    sol_gmin, sol_gmax = solve_cooling_gmas(cell, cell_new, func_cool, env, withSol=False)
    gm_new, gM_new = sol_gmin.y[0][-1], sol_gmax.y[0][-1]
  return gm_new, gM_new

def syn_rad_fromCooling(tnu_arr, t_span, func_gmin, func_gmax, tc, p):
  '''
  Calculate synchrotron flux from cooling plaw of electrons, tnu_arr normalized to syn freq
  Replaces DeltaEnup, using the outputs of solve_ivp insteap of analytical form
  '''

  out = np.zeros(tnu_arr.shape)
  rows, cols = tnu_arr.shape
  gm, gm_end = func_gmin(t_span[0]), func_gmin(t_span[1])
  gM, gM_end = func_gmax(t_span[0]), func_gmax(t_span[1])

  sqtnu_arr = np.sqrt(tnu_arr)

  def func_gmaf(gma, t):
    tt = (t-t_span[0])/tc
    return gma**(-p-2/3.) * (1-gma*tt)**(p-2)
  
  def gma_max(t):
    return func_gmax(t)

  for i in range(0, rows):
    for j in range(0, cols):
      sqtnu = sqtnu_arr[i,j]
      def gma_min(t):
        return max(sqtnu, func_gmin(t))

      if (gm_end-gm!=0.) and (gM_end-gM!=0.):
        print(t_span[0], t_span[1], sqtnu, gm, gm_end, gM, gM_end)
        out[i,j] += spi.dblquad(func_gmaf, t_span[0], t_span[1], gma_min, gma_max)[0]
  
  out *= tnu_arr**(1/3.)
  return out

# def syn_rad_fromCooling_delta(gmax, gmax_next, tnu, tT, env):
#   '''
#   Calculate synchrotron flux from cooling delta distrib of electrons.
#   tnu is normalized to initial characteristic frequency, tT is normalized time
#   '''


def get_efficiency_hydroThinshell(key, mFC_arr=[False, False]):
  '''
  Calculate radiative efficiency: total emitted energy over initial kinetic energy
  '''
  
  env = MyEnv(key)
  dataRS, dataFS = open_hydro_data(key)
  # get radiated energy by summing over contributing cell
  Ei_RS, Ei_FS = 0., 0.
  for k, cell in dataRS.iterrows():
    if cell['rho'] == 0.: continue
    Ei = get_variable(cell, 'Ei', env)
    Ei_RS += Ei
  if mFC_arr[0]:  Ei_RS *= 0.5

  for k, cell in dataFS.iterrows():
    if cell['rho'] == 0.: continue
    Ei = get_variable(cell, 'Ei', env)
    Ei_FS += Ei
  if mFC_arr[1]:  Ei_FS *= 0.5

  # get total internal energy at shock crossing
  it_crRS, it_crFS = int(dataRS['it'].iloc[-1]), int(dataFS['it'].iloc[-1])
  df = openData(key, it_crRS)
  trac = df['trac'].to_numpy()
  Ei = get_variable(df, 'Ei', env)
  Ei_RSf = Ei[(trac > 0.) & (trac < 1.5)].sum()

  df = openData(key, it_crFS)
  Ei = get_variable(df, 'Ei', env)
  Ei_FSf = Ei[(trac > 1.5)].sum()
  #print(f'From final state: {Ei_RSf:.2e}, {Ei_FSf:.2e}')
  #print(f'From summing: {Ei_RS:.2e}, {Ei_FS:.2e}')

  effth_gl = (Ei_RSf + Ei_FSf)/(env.Ek4 + env.Ek1)
  effth_sum = (Ei_RS + Ei_FS)/(env.Ek4 + env.Ek1)
  return effth_gl, effth_sum


def get_Fnu_thinshell(data, nuobs, Tobs, env, func='Band', mFC=False):
  '''
  Calculates Fnu from a given cell in the thinshell approximation (deep in fast cooling)
  func is the spectral shape in comoving frame ('Band' or 'plaw')
  mFC = True calculates flux from marginally fast cooling
  '''
  a, d = 1, -1
  Fnu = np.zeros((len(Tobs), len(nuobs)))

  eps_r, b1 = (.5, 1/3.) if mFC else (1, -0.5)
  b2 = -env.psyn/2
  if func == 'Band':
    def S(x, b1, b2):
      return Band_func(x, b1, b2)
  elif func == 'plaw':
    def S(x, b1, b2):
      return broken_plaw(x, b1, b2)
  else:
    print("func must be 'Band' of 'plaw'!")
    return Fnu
  
  r = data['x']
  Ton, Tth, Tej = get_variable(data, 'obsT', env)
  L = get_variable(data, 'Lth', env)
  # nu_m = get_variable(data, 'nu_m', env) if 'gmin_' in data.keys() else get_variable(data, 'nu_m2', env)
  nu_m = get_variable(data, 'nu_m2', env)

  if r == 0.:
    return Fnu
  if env.geometry == 'cartesian':
    nu_m *= (r*c_/env.R0)**d
    L *= (r*c_/env.R0)**a

  tnu = (nuobs/nu_m)
  tT = ((Tobs - Tej)/Tth)
  Fnu[tT>=1.] = env.zdl * eps_r * derive_Lnu(tnu, tT[tT>=1.], L, S, b1, b2)
  return Fnu

def derive_Lnu(tnu, tT, L, S, b1, b2):
  if (type(tT)==np.ndarray) and (type(tnu)==np.ndarray):
    out = np.zeros((len(tT), len(tnu)))
    tnu = tnu[np.newaxis, :]
    tT  = tT[:, np.newaxis]
  elif (type(tT)==np.ndarray):
    out = np.zeros(tT.shape)
  elif (type(tnu)==np.ndarray):
    out = np.zeros(tnu.shape)
  out += L * tT**-2 * S(tnu*tT, b1, b2)
  return out

def get_Fnu_cooling_forcevFC(cell, nuobs, Tobs, env):
  '''
  Derive the radiation emitted by cell in force very fast cooling
  '''
  # identify regime
  out = np.zeros((len(Tobs), len(nuobs)))
  gmin, gmax = cell[['gmin_', 'gmax_']]
  if not gmin:
    return out, ''
  nup_B = get_variable(cell, "nup_B", env)
  Ton, Tth, Tej = get_variable(cell, 'obsT', env)
  tT = ((Tobs - Tej)/Tth)
  width = gmax/gmin
  Dop = get_variable(cell, 'Dop', env)
  nup = nuobs/Dop
  p = env.psyn

  nup_m = nup_B * gmin**2
  x_B = gmin**-2
  x_M = gmax**2 * x_B
  Epnu = get_variable(cell, 'Ep_th', env)
  breaks = [x_B, 1, x_M]
  slopes = [-0.5, -0.5*p]

  nu_m = Dop*nup_m
  tnu = nuobs/nu_m

  if (type(tT)==np.ndarray) and (type(tnu)==np.ndarray):
    out = np.zeros((len(tT), len(tnu)))
    tnu = tnu[np.newaxis, :]
    tT  = tT[:, np.newaxis]
  elif (type(tT)==np.ndarray):
    out = np.zeros(tT.shape)
  elif (type(tnu)==np.ndarray):
    out = np.zeros(tnu.shape)

  S = broken_plaw_withstops(tnu*tT, breaks, slopes)
  out += env.zdl * ((1+env.z)*Epnu/Tth) * tT**-2 * S
  return out

def test_Fnu_cooling(refLF='mid', onlyInt=False):
  key, n, var, logv, istep = 'test', -10, 'sp', 0, 0
  nuobs, Tobs, env = radEnv(key, lowfreqs=True)
  tnu = nuobs/env.nu0
  Tb = (Tobs-env.Ts)/env.T0
  i = find_closest(Tb, 10**logv)
  Tobs = np.array([Tobs[i]])
  dk = get_gmasdata_cell(n, key)
  cell_0 = dk.iloc[0]
  cell = dk.iloc[istep]
  cell_next = dk.iloc[istep+1]

  Fnu_1, freqvals = get_Fnu_cooling(cell, cell_next, cell_0, nuobs, Tobs, env,
    testing=True, refLF=refLF, onlyInt=onlyInt)
  Fnu_2, _ = get_Fnu_cooling_pureplaws(cell, cell_next, nuobs, Tobs, env)
  Fnus = [Fnu_1, Fnu_2]
  x = tnu
  y_arr = [tnu*Fnu.flatten()/env.F0 for Fnu in Fnus]
  
  fig, ax = plt.subplots()
  for y, ls in zip(y_arr, ['-', '--']):
    ax.loglog(x, y, c='r', ls=ls)
  ax.set_xlabel('$\\tilde{\\nu}$')
  ax.set_ylabel('$\\nu F_{\\nu}/\\nu_0 F_0$')

  freqnames = ['$\\nu_{\\rm m,j}$', '$\\nu_{\\rm M,j}$',
      '$\\nu_{\\rm m,j+1}$', '$\\nu_{\\rm M,j+1}$']
  trans = transforms.blended_transform_factory(
      ax.transData, ax.transAxes)
  for val, name in zip(freqvals, freqnames):
    ax.axvline(val, c='k', ls=':', lw=.8)
    #ax.text(val, -0.1, name, transform=trans)


def get_Fnu_cooling(cell, cell_next, cell_0, nuobs, Tobs, env,
  testing=False, refLF='mid', onlyInt=True):
  '''
  Derive the radiation emitted between states cell and cell_next
  comparison between (gmin, gmax) gives the cooling regime
  normalization is derived using hydro state cell
  '''
  # identify regime

  gmin, gmax = cell[['gmin_', 'gmax_']]
  gmin_nxt, gmax_nxt = cell_next[['gmin_', 'gmax_']]
  if not gmin:
    return out, ''
  nup_B = get_variable(cell, "nup_B", env)
  Ton, Tth, Tej = get_variable(cell, 'obsT', env)
  width = gmax/gmin
  Dop = get_variable(cell, 'Dop', env)
  p = env.psyn

  Pe = get_variable(cell, 'Pmax', env)
  V3 = get_variable(cell, 'V3p', env)
  tc = 1/get_variable(cell, 'syn', env)

  if (type(Tobs)==np.ndarray) and (type(nuobs)==np.ndarray):
    out = np.zeros((len(Tobs), len(nuobs)))
    nuobs = nuobs[np.newaxis, :]
    Tobs  = Tobs[:, np.newaxis]
  elif (type(Tobs)==np.ndarray):
    out = np.zeros(Tobs.shape)
  elif (type(nuobs)==np.ndarray):
    out = np.zeros(nuobs.shape)

  tT = ((Tobs - Tej)/Tth)
  nup = nuobs/Dop

  regime = ''
  norm_index = 1
  if (width <= 2.) and (not onlyInt): # "mono-E" regime
    regime = 'monoE'
    # either gmax & gmax_next or gmin and gmin_next
    # or the mean
    gm, gm_nxt = np.sqrt(gmin*gmax), np.sqrt(gmin_nxt*gmax_nxt)
    if refLF == 'min':
      gm, gm_nxt = gmin, gmin_nxt
    elif refLF == 'max':
      gm, gm_nxt = gmax, gmax_nxt
    elif refLF == 'mean':
      gm, gm_nxt = .5*(gmin+gmax), .5*(gmin_nxt+gmax_nxt)
    nup_m = nup_B * gm**2
    x_B = gm_nxt**-2
    x_m = (gm_nxt/gm)**2
    Epnu_m = (3/5.) * Pe * tc * V3 / gm
    # normalized with nup_m
    def spec(tnu):
      return mixed_plaw(tnu, x_B, x_m)
    tnu = nup/nup_m
    Epnu = Epnu_m * spec(tT*tnu)

  else:   # 'exact' cooling by integrating evolution of gmas min/max
    regime = 'fast' if gmin >= gmax_nxt else 'slow'
    tnu = nup/nup_B
    K = (p-1)/(gmin**(1-p)-gmax**(1-p))
    Epnu_m = K*Pe*tc*V3 # Pe already contains ne
    Epnu = DeltaEnup_v2(tT*tnu, cell_0, cell, cell_next, env)
    Epnu *= Epnu_m

  out += env.zdl * ((1+env.z)/Tth) * Epnu * tT**-2
  if testing:
    gm0, gM0 = cell_0[['gmin_', 'gmax_']]
    gm, gM = cell[['gmin_', 'gmax_']]
    gm_next, gM_next = cell_next[['gmin_', 'gmax_']]
    critfreqs = (nup_B/((1+env.z)*env.nu0p))*np.array([gm**2, gM**2, gm_next**2, gM_next**2])
    return out, critfreqs
  return out, regime

def DeltaEnup(tnu_arr, cell_0, cell, cell_next, env):
  rows, cols = tnu_arr.shape
  out = np.zeros((rows, cols))
  sqtnu_arr = np.sqrt(tnu_arr)
  p = env.psyn

  gm0, gM0 = cell_0[['gmin_', 'gmax_']]
  gm, gM = cell[['gmin_', 'gmax_']]
  gm_next, gM_next = cell_next[['gmin_', 'gmax_']]

  lfac = get_variable(cell, 'lfac', env)
  tc = get_variable(cell, 'syn', env)**-1
  tt_i = (cell['t'] - cell_0['t'])/(tc*lfac)
  tt_f = (cell_next['t'] - cell_0['t'])/(tc*lfac)

  def gma_m(t):
    return gm0/(1+gm0*t)
  def gma_M(t):
    return gM0/(1+gM0*t)

  def func_gmai(gma, t):
    return gma**(-p-2/3.) * (1+gma*t)**(2/3.)
  def func_gmaf(gma, t):
    return gma**(-p-2/3.) * (1-gma*t)**(p-2)

  for i in range(0, rows):
    for j in range(0, cols):
      sqtnu = sqtnu_arr[i,j]
      def gma_low(t):
        return max(sqtnu, gma_m(t))
      tt_nuM = 1/sqtnu - 1/gM0
      tt_max = min(tt_nuM, tt_f)
      out[i,j] += spi.dblquad(func_gmaf, tt_i, tt_max, gma_low, gma_M)[0]
  
  out *= tnu_arr**(1/3.)
  return out
  
def DeltaEnup_v2(tnu_arr, cell_0, cell, cell_next, env):
  '''
  Delta E'_{\nu'} spectrum in fast cooling case between cell and cell_next
  in units K P'_{e, max} t'_c
  mix and match between integral forms and integrated
  '''

  rows, cols = tnu_arr.shape
  out = np.zeros((rows, cols))
  sqtnu_arr = np.sqrt(tnu_arr)
  p = env.psyn
  sq_nuB = np.sqrt(get_variable(cell, 'nup_B', env))

  gm0, gM0 = cell_0[['gmin_', 'gmax_']]
  gm, gM = cell[['gmin_', 'gmax_']]
  gm_next, gM_next = cell_next[['gmin_', 'gmax_']]

  lfac = get_variable(cell, 'lfac', env)
  tc = get_variable(cell, 'syn', env)**-1
  tt_i = (cell['t'] - cell_0['t'])/(tc*lfac)
  tt_f = (cell_next['t'] - cell_0['t'])/(tc*lfac)
  def func_gmaf(gma, t):
    return gma**(-p-2/3.) * (1-gma*t)**(p-2)
  def gma_M(tt):
    return gM0/(1+gM0*tt)

  
  g1 = min(gm, gM_next)
  g2 = max(gm, gM_next)
  printcount = 0
  for i in range(0, rows):
    for j in range(0, cols):
      sqtnu = sqtnu_arr[i,j]
      def gma_nu(tt):
        return sqtnu
      if sqtnu <= 1.:
        # tnu already normalized to cyclotron freq
        continue
      elif sqtnu <= gm_next:
        out[i,j] += DeltaE_func(tt_i, tt_f, gm0, gM0, p)
      elif sqtnu <= g1:
        tt_num = 1/sqtnu - 1/gm0
        term1 = DeltaE_func(tt_i, tt_num, gm0, gM0, p)
        term2 = spi.dblquad(func_gmaf, tt_num, tt_f, gma_nu, gma_M)[0]
        out[i,j] += term1 + term2
      elif sqtnu <= g2:
        if gm>=gM_next: # fast cooling
          tt_num = 1/sqtnu - 1/gm0
          tt_nuM = 1/sqtnu - 1/gM0
          term1 = DeltaE_func(tt_i, tt_num, gm0, gM0, p)
          term2 = spi.dblquad(func_gmaf, tt_num, tt_nuM, gma_nu, gma_M)[0]
          out[i,j] += term1 + term2
        else: # slow cooling
          out[i,j] += spi.dblquad(func_gmaf, tt_i, tt_f, gma_nu, gma_M)[0]
      elif sqtnu <= gM:
        tt_nuM = 1/sqtnu - 1/gM0
        out[i,j] += spi.dblquad(func_gmaf, tt_i, tt_nuM, gma_nu, gma_M)[0]

  out *= tnu_arr**(1/3.)
  return out

def DeltaEnup_v1(tnu_arr, cell_0, cell, cell_next, env):
  '''
  Delta E'_{\nu'} spectrum in fast cooling case between cell and cell_next
  in units K P'_{e, max} t'_c
  '''

  rows, cols = tnu_arr.shape
  out = np.zeros((rows, cols))
  sqtnu_arr = np.sqrt(tnu_arr)
  p = env.psyn

  gm0, gM0 = cell_0[['gmin_', 'gmax_']]
  gm, gM = cell[['gmin_', 'gmax_']]
  gm_next, gM_next = cell_next[['gmin_', 'gmax_']]

  lfac = get_variable(cell, 'lfac', env)
  tc = get_variable(cell, 'syn', env)**-1
  tt_i = (cell['t'] - cell_0['t'])/(tc*lfac)
  tt_f = (cell_next['t'] - cell_0['t'])/(tc*lfac)
  
  g1 = min(gm, gM_next)
  g2 = max(gm, gM_next)
  printcount = 0
  for i in range(0, rows):
    for j in range(0, cols):
      sqtnu = sqtnu_arr[i,j]
      if sqtnu <= gm_next:
        out[i,j] += DeltaE_func(tt_i, tt_f, gm0, gM0, p)
        #out[i,j] += DeltaE_integ_gmai_csts(tt_i, tt_f, gm0, gM0, p)
      elif sqtnu <= g1:
        tt_num = 1/sqtnu - 1/gm0
        tt_hf = .5*(tt_num+tt_f)
        #tt_hf = min(tt_num, tt_f)
        gma_nu = sqtnu/(1-tt_hf*sqtnu)
        out[i,j] += DeltaE_func(tt_i, tt_num, gm0, gM0, p) + DeltaE_func(tt_num, tt_f, gma_nu, gM0, p)
        # out[i,j] += DeltaE_func(tt_i, tt_num, gm0, gM0, p) + \
        #     DeltaE_integ_gmai(tt_num, tt_f, sqtnu, gM0, p)
      elif sqtnu <= g2:
        if gm>=gM_next: # fast cooling
          tt_num = 1/sqtnu - 1/gm0
          tt_nuM = 1/sqtnu - 1/gM0
          tt_hf = .5*(tt_num+tt_nuM)
          gma_nu = sqtnu/(1-tt_hf*sqtnu)
          out[i,j] += DeltaE_func(tt_i, tt_num, gm0, gM0, p) + DeltaE_func(tt_num, tt_nuM, gma_nu, gM0, p)
          # out[i,j] += DeltaE_func(tt_i, tt_num, gm0, gM0, p) + \
          #     DeltaE_integ_gmai(tt_num, tt_nuM, sqtnu, gM0, p)
        else: # slow cooling
          tt_hf = .5*(tt_i+tt_f)
          gma_nu = sqtnu/(1-tt_hf*sqtnu)
          out[i,j] += DeltaE_func(tt_i, tt_f, gma_nu, gM0, p)
          #out[i,j] += DeltaE_integ_gmai(tt_i, tt_f, sqtnu, gM0, p)
      elif sqtnu <= gM:
        tt_nuM = 1/sqtnu - 1/gM0
        tt_hf = .5*(tt_i+min(tt_f, tt_nuM))
        gma_nu = sqtnu/(1-tt_hf*sqtnu)
        out[i,j] += DeltaE_func(tt_i, tt_nuM, gma_nu, gM0, p)
        #out[i,j] += DeltaE_integ_gmai(tt_i, tt_nuM, sqtnu, gM0, p)

  out *= tnu_arr**(1/3.)
  return out

def DeltaE_integ_gmai_csts(t_l, t_u, gm0, gM0, p):
  '''
  Integral form of E'_{\nu'}\tilde{\nu}^{-1/3} in units K P'_{e,max}
    the integral on the LF is done on variable gma_i over whole contribs
  '''
  # lower bound of integration
  def gma_low(t):
    return gm0
  
  # upper bound of integration, constant
  def gma_up(t):
    return gM0
  
  # integrand
  def func(y, t):
    return y**(-p-2/3.)*(1+y*t)**(2/3.)
  
  out = spi.dblquad(func, t_l, t_u, gma_low, gma_up)[0]
  return out

def DeltaE_integ_gmai(t_l, t_u, sqtnu, gM0, p):
  '''
  Integral form of E'_{\nu'}\tilde{\nu}^{-1/3} in units K P'_{e,max}
    the integral on the LF is done on variable gma_i
  '''

  # lower bound of integration
  def gma_low(t):
    return sqtnu/(1-sqtnu*t)
  
  # upper bound of integration, constant
  def gma_up(t):
    return gM0
  
  # integrand
  def func(y, t):
    return y**(-p-2/3.)*(1+y*t)**(2/3.)
  
  out = 0.
  if (gma_low(t_l)-gma_low(t_u)!=0.) and (gma_up(t_l)-gma_up(t_u)!=0.):
    out += spi.dblquad(func, t_l, t_u, gma_low, gma_up)[0]
  return out

def DeltaE_integ_gmaf(t_l, t_u, sqtnu, gM0, p):
  '''
  Integral form of E'_{\nu'}\tilde{\nu}^{-1/3} in units K P'_{e,max}
    the integral on the LF is done on variable gma_f
  '''
  # lower bound of integration
  def gma_low(t):
    return sqtnu
  
  # upper bound of integration, constant
  def gma_up(t):
    return gM0/(1+gM0*t)
  
  # integrand
  def func(y, t):
    return y**(-p-2/3.)*(1-y*t)**(p-2)
  
  out = 0.
  if (gma_low(t_l)-gma_low(t_u)!=0.) and (gma_up(t_l)-gma_up(t_u)!=0.):
    # nonzero
    out += spi.dblquad(func, t_l, t_u, gma_low, gma_up)[0]
  return out


def DeltaE_func(t_l, t_u, gma_l, gma_u, p):
  '''
  Functional form of E'_{\nu'}\tilde{\nu}^{-1/3} in units K P'_{e,max}
  the inputs depend on frequency, the times are normalized
  '''

  El = E_func(t_l, gma_l, gma_u, p) if t_l else 0.
  Eu = E_func(t_u, gma_l, gma_u, p)
  return Eu - El

def E_func(dt, gma_l, gma_u, p):
  '''
  Function \mathcal{E}_p in my document
  '''
  Fl = Fp_func(gma_l, dt, p)
  Fu = Fp_func(gma_u, dt, p)

  return (3/5.) * (Fu - Fl)/((p-1)*(3*p+2))

def Fp_func(gma, t, p):
  '''
  Expression, noted \mathcal{F}_p in my document, that appears in the time integral
  gma is a Lorentz factor (typically gma_m or gma_M), t is the normalized time interval
  '''

  gt = gma*t
  gtm = -1/gt
  term1 = 2*(1+gt)*hyp2f1(1/3,p-1,p,gtm)
  term2 = (3*p-1+(3*p+4)*gt)*hyp2f1(-2/3,p-1,p,gtm)
  diff = gt**(2/3.) * (term1-term2)
  return gma**(-p-2/3.) * (3*(p-1) + diff)


def get_coolingvals_fromstep(j, dk, env):
  '''
  Returns values to calculate cooling emission between steps j and j+1 
  '''

  cell = dk.iloc[j]
  cell_next = dk.iloc[j+1]
  gmin0, gmax0 = cell[['gmin_', 'gmax_']]
  #gmin_nxt, gmax_nxt = cell_next[['gmin_', 'gmax_']]

  p = env.psyn
  tc = 1/get_variable(cell, 'syn', env)
  dt = (cell_next['t']-cell['t'])/get_variable(cell, 'lfac', env)
  dT = dt/tc

  return gmin0, gmax0, dT


def mixed_plaw(x, x_B, x_m):
  '''
  Spectral shape for monoE regime from integrating cooling during a timestep
  x**-1/2 - x**1/3 between x_B and 1
  '''
  y = np.zeros(x.shape)
  segment1 = (x>=x_B) & (x<=x_m)
  segment2 = (x>=x_m) & (x<=1.)
  y[segment1] = x[segment1]**(1/3.) * (x_m**-(5/6.) - 1)
  # y[segment2] = x[segment2]**(1/3.)*(x[segment2]**(-5/6.) - 1)
  y[segment2] = x[segment2]**(-.5) - x[segment2]**(1/3.)

  return y


def get_Fnu_cooling_pureplaws(cell, cell_next, nuobs, Tobs, env):
  '''
  Derive the radiation emitted between states cell and cell_next
  comparison between (gmin, gmax) gives the cooling regime
  normalization is derived using hydro state cell
  '''
  # identify regime
  gmin, gmax = cell[['gmin_', 'gmax_']]
  gmin_nxt, gmax_nxt = cell_next[['gmin_', 'gmax_']]
  if not gmin:
    return out, ''
  nup_B = get_variable(cell, "nup_B", env)
  Ton, Tth, Tej = get_variable(cell, 'obsT', env)
  tT = ((Tobs - Tej)/Tth)
  width = gmax/gmin
  Dop = get_variable(cell, 'Dop', env)
  nup = nuobs/Dop
  p = env.psyn

  regime = ''
  norm_index = 1
  if width <= 2.: # "mono-E" regime
    regime = 'monoE'
    nup_m = nup_B * gmax**2
    x_B = gmax_nxt**-2
    x_m = (gmax_nxt/gmax)**2
    Ne = get_variable(cell, 'Ne', env)*env.xi_e
    dgma = gmax - gmax_nxt
    Epnu = 0.8*Ne*dgma*me_*c_**2 / nup_m
    #Pe = get_variable(cell, 'Pmax', env)
    #V3 = get_variable(cell, 'V3p', env)
    #tc = 1/get_variable(cell, 'syn', env)
    #Epnu = (3/5.) * Pe * tc * V3 * (x_m**(1/3)) * (x_m**-(5/6.) - 1) / gmax
    breaks = [x_B, x_m, 1.]
    slopes = [1/3., -.5]
    norm_index = 1

  else:
    width_nxt = gmax_nxt/gmin_nxt
    if width_nxt > 5: # slow cooling
      regime = 'slow'
      nup_m = nup_B * gmin_nxt**2
      x_B = gmin_nxt**-2
      x_c = gmax_nxt**2 * x_B
      x_M = gmax**2 * x_B
      breaks = [x_B, 1., x_c, x_M]
      slopes = [1/3, 0.5*(1-p), -0.5*p]
      Epnu = get_variable(cell, 'EpnuSC', env)
    else: # fast cooling
      regime = 'fast'
      nup_m = nup_B * gmin**2
      x_B = gmin**-2
      x_M = gmax**2 * x_B
      Epnu = get_variable(cell, 'Ep_th', env)
      if gmax_nxt == 1.: # very fast cooling
        breaks = [x_B, 1, x_M]
        slopes = [-0.5, -0.5*p]
      else: 
        x_c = gmax_nxt**2 * x_B
        breaks = [x_B, x_c, 1., x_M]
        slopes = [1/3., -0.5, -0.5*p]
        norm_index = 2
        Wp = 2*(p-1)/(p-2)
        Xp = Wp - 1.25*np.sqrt(x_c)
        Epnu *= Wp/Xp 
  def spec(tnu):
    return broken_plaw_withstops(tnu, breaks, slopes, norm_index)

  nu_m = Dop*nup_m
  tnu = nuobs/nu_m

  if (type(tT)==np.ndarray) and (type(tnu)==np.ndarray):
    out = np.zeros((len(tT), len(tnu)))
    tnu = tnu[np.newaxis, :]
    tT  = tT[:, np.newaxis]
  elif (type(tT)==np.ndarray):
    out = np.zeros(tT.shape)
  elif (type(tnu)==np.ndarray):
    out = np.zeros(tnu.shape)

  S = spec(tT*tnu)
  #S = broken_plaw_withstops(tnu*tT, breaks, slopes, norm_index)
  out += env.zdl * ((1+env.z)*Epnu/Tth) * tT**-2 * S
  return out, regime


def broken_plaw_withstops(x, breaks, indices, norm_index=1):
  '''
  Broken p-law returning 0 outside of first and last breaks
  '''
  y = np.zeros(x.shape)
  if len(indices) != len(breaks) - 1:
    print("You must give exactly 1 less index than there are breaks!")
    return y

  y0 = 1.
  yn = 1.
  breaks = np.array(breaks)
  for i, p in enumerate(indices):
    if i == norm_index:
      yn = y0
    xmin, xmax = breaks[[i, i+1]]
    segment = (x>=xmin) & (x<=xmax)
    y[segment] = y0*(x[segment]/xmin)**p
    y0 = y0*(xmax/xmin)**p
  y /= yn
  return y

######## overcharging functions because it's easier than to rewrite data_IO
def get_vars(df, varlist, env):
  '''
  Return array of variables
  '''
  if type(df) == pd.core.series.Series:
    out = np.zeros((len(varlist)))
  else:
    out = np.zeros((len(varlist), len(df)))
  for i, var in enumerate(varlist):
    out[i] = get_variable(df, var, env)
  return out

def get_variable(df, var, env):
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
    res = df_get_var(df, func, inlist, envlist, env)
  return res

def df_get_var(df, func, inlist, envlist, env):
  params, envParams = get_params(df, inlist, envlist, env)
  res = func(*params, *envParams)
  return res

def get_params(df, inlist, envlist, env):
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
    envParams = [getattr(env, name) for name in envlist]
  return params, envParams