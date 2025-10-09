# -*- coding: utf-8 -*-
# @Author: acharlet

'''
Extracts data from hydro sims from a cell perspective,
produces radiation
'''

import sys
from pathlib import Path
from plotting_scripts_new import *
import scipy.integrate as spi
from scipy.special import gamma, hyp2f1
from scipy.optimize import brentq
from betainc import betai
plt.ion()
np.seterr(all='raise')


########################  Hydro  ############################
# data extraction
def get_dirPath(key):
  return GAMMA_dir + f'/results/{key}/'

def get_cellPath(key, k):
  return get_dirPath(key) + f'extractedCells/{k:04d}.csv'

def get_cellData(key, k, fromSh=True, fit=True):
  '''
  Returns dataframe with data history from a cell,
  starting from the instant cell is shocked if fromSh = True
  fromSh also triggers replacing hydro values by those once shock crossed
  '''

  path = get_cellPath(key, k)
  attrslist = ['key', 'mode', 'runname', 'rhoNorm', 'geometry']
  attrs = get_runatts(key)
  attrs.insert(0, key)
  df = pd.read_csv(path)
  for attr, val in zip(attrslist, attrs):
    df.attrs[attr] = val

  if fromSh:
    out = df[(df.Sd == 1.).idxmax():]
    if fit:
      out = mean_replace_data(out)
    return out
  else:
    return df

def fit_replace_data(df):
  '''
  Fit hydro data and return dataframe with the fit
  -> cleans numerical oscillation
  '''
  out = df.copy()
  return out

def mean_replace_data(df):
  '''
  Replace early data with simple mean + geom. scaling of established downstream values
  '''

  varlist = ['rho', 'p', 'D', 'sx', 'tau']
  # use data length of shock establishing as scale for the mean
  cr_len = len(df[df.Sd==1])
  index = df.index
  imin, imax = int(cr_len*2), int(cr_len*3)
  dic = df.to_dict(orient='list')

  # constant vx
  vx = np.array(dic['vx'])
  lfac = derive_Lorentz(vx)
  lfac_mean = lfac[imin:imax].mean()
  lfac[:imin] = lfac_mean
  dic['vx'] = derive_velocity(lfac).tolist()

  for var in varlist:
    val = np.array(dic[var])
    if df.attrs['geometry'] == 'cartesian':
      r_weight = np.ones(cr_len)
      r_fac = np.ones(imin)
    else:
      r = np.array(dic['x'])
      r_weight = (r[imin:imax]/r[imin])**2
      r_fac = (r[:imin]/r[imin])**-2
    mean = (val[imin:imax]*r_weight).mean()
    val[:imin] = mean*r_fac
    dic[var] = val

  df = pd.DataFrame.from_dict(dic)
  return df

def get_cellData_rescaled(key, k, logRatio=None, fromSh=True):
  '''
  Returns cell data rescaled accordingly
  if None, returns writeen data and env
  '''

  env0 = MyEnv(key)
  df0 = get_cellData(key, k, fromSh=fromSh)
  if logRatio is not None:
    scalefac = get_scalefac(logRatio, df0, env0)
    df = rescale_data(df0, scalefac)
    env = MyEnv(key, scalefac)
    return df, env
  else:
    return df0, env0

def get_cellData_withDistrib(key, k, logRatio=None, fromSh=True, Ngma=50):
  '''
  Returns dataframe with history of cell, rescaled, and with electrons
  '''
  df, env = get_cellData_rescaled(key, k, logRatio=logRatio, fromSh=fromSh)
  df = generate_cellDistrib(df, env, Ngma=Ngma)
  return df, env

def rescale_data(df, scalefac):
  '''
  Rescale data, keeping vx (lfac) constant
  Only lengths and time are rescaled, the rest is handled by the environment
  '''

  out = df.copy()
  sc = 10**scalefac
  print(f'rescaling by factor {sc:.1e}')
  out.t *= sc
  out.x *= sc
  out.dx *= sc
  out.dlx *= sc

  return out

def get_scalefac(logRatio, df, env):
  '''
  Returns (log) scale factor of radius to obtain desired gma_c/gma_m (log) ratio
  '''
  current = check_coolRegime(df, env)
  scalefac = logRatio - current
  return scalefac

def check_coolRegime(df, env):
  '''
  Check gma_c/gma_m, using doubling radius for t_dyn
  '''
  cell = df.iloc[0]
  gma_m = get_variable(cell, 'gma_m', env)
  tdyn = get_tdyn(df, env)
  gma_c = tdyn/get_variable(cell, 'syn', env)
  return np.log10(gma_c/gma_m)

def get_tdyn(df, env):
  '''
  Comoving dynamical time 
  '''

  tstart = df.iloc[0].t
  tcr = np.nan # for now, will need implementation of crossing time
  tend = min(df.iloc[-1].t, tcr)
  tdyn1 = (tend-tstart)/env.lfac
  tdyn2 = env.R0/(env.lfac*c_)
  tdyn = min(tdyn1, tdyn2)
  return tdyn

def extract_hydro(key, itmin=0, itmax=None):
  '''
  Extracts hydro data by reorganizing into cells history
  '''

  dirpath = get_dirPath(key)
  its = dataList(key, itmin, itmax)[0:]
  # create subfolder to save cells
  Path(dirpath+'extractedCells').mkdir(parents=True, exist_ok=True) 

  # create dictionaries with arrays to fill
  d0 = openData(key, 0)
  varlist = d0.keys()
  klist = d0.loc[d0['trac']>0]['i'].to_list()
  dics_arr = [{var:np.zeros(len(its)) for var in varlist} for k in klist]
  
  for j, it in enumerate(its):
    if not (it%100):
      print(f'Opening file it {it}')
    df = openData(key, it)
    for i, k in enumerate(klist):
      for var in varlist:
        dics_arr[i][var][j] += df.at[k, var]
  
  for k, dic in zip(klist, dics_arr):
    cellpath = get_cellPath(key, k)
    dic['it'] = its
    out_k = pd.DataFrame.from_dict(dic)
    out_k.to_csv(cellpath, index=False)

# extracted data plotting 
def plot_any(var, df, xscale='r', ax_in=None, logx=False, logy=False, **kwargs):
  '''
  Plots any variable that can be obtained by the get_var() func
  '''

  env = MyEnv(df.attrs['key'])

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
  elif xscale == 'tc':
    xlabel="$t'/t'_{\\rm c}$"

  if var in var_exp.keys():
    ylabel = var_exp[var]
  elif 'gma' in var:
    ylabel = '$\\gamma$'
    logy = True
  else:
    ylabel = var

  
  it, t, r = df[['it', 't', 'x']].to_numpy().transpose()
  y = get_variable(df, var, env)
  if xscale == 'r':
    x = r*c_/env.R0
  elif xscale == 't':
    x = t/env.t0 + 1.
  elif xscale == 'T':
    T = (1+env.z)*(t + env.t0 - r)
    x = (T-T[0])/env.T0
  elif xscale == 'it':
    x = it
  elif xscale == 'tc':
    x = (t-t[0])/(env.lfac*env.tc)
    
  ax.plot(x, y, **kwargs)

  if logx:
    ax.set_xscale('log')
  if logy:
    ax.set_yscale('log')
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)


########################  Radiation  ############################
def radEnv(env, largeFreqs=False):
  '''
  Arrays of observed frequency and time of main emission space
  '''

  Tbmax = 5
  nT = 100
  numin, numax = (-6, 8) if largeFreqs else (-3, 1.5)
  nuobs = env.nu0*np.logspace(numin, numax, 600)
  Tobs = env.Ts + np.linspace(0, Tbmax, nT*Tbmax)*env.T0
  return nuobs, Tobs

def gma_cooledExp(tt, gma0):
    c0 = (gma0-1)/(gma0+1)
    return (np.exp(2*tt)+c0)/(np.exp(2*tt)-c0)

def testplot_step(j, k=450, key='test', logRatio=None):
  df, env = get_cellData_withDistrib(key, k, logRatio)
  nuobs, Tobs = radEnv(env, largeFreqs=True)
  p=env.psyn
  tnu = np.logspace(0, 20, 10000)
  tT = 2.26
  tnu = tT*tnu
  cell_0, cell, cell_next = df.iloc[0], df.iloc[j], df.iloc[j+1]
  gm0, gM0 = cell_0[['gmin', 'gmax']]
  #cell = cell_0
  lfac = get_variable(cell_0, 'lfac', env)
  tc = 1/get_variable(cell_0, 'syn', env)
  tt_i = (cell['t'] - cell_0['t'])/(tc*lfac)
  tt_f = (cell_next['t'] - cell_0['t'])/(tc*lfac)
  gm, gM, gm_nxt, gM_nxt = *cell[['gmin', 'gmax']], *cell_next[['gmin', 'gmax']]
  if gm>=gM_nxt:
    print("local 'fast cooling'")
  else:
    print("local 'slow cooling'")
  dtt = tt_f-tt_i
  gm_th, gM_th = gma(tt_i, gm0), gma(tt_i, gM0)
  gm_nxth, gM_nxth = gma(tt_f, gm0), gma(tt_f, gM0)
  x_B = get_variable(cell_0, 'nu_B', env)/env.nu0
  x_arr = np.array([gm**2, gM**2, gm_nxt**2, gM_nxt**2])
  x_names = ['m,j', 'M,j', 'm,j+1', 'M,j+1']
  x_arr2 = np.array([gm_th**2, gM_th**2, gm_nxth**2, gM_nxth**2])
  x_names2 = ['m,th,j', 'M,th,j', 'm,th,j+1', 'M,th,j+1']
  # Epnu1 = tnu*DeltaEnup_func(tnu, gm0, gM0, tt_i, gm, gM, tt_f, gm_nxt, gM_nxt, p)
  # y1 = Epnu1.flatten()
  # try:
  #   Epnu2 = tnu*DeltaEnup_mixed(tnu, gm0, gM0, tt_i, gm, gM, tt_f, gm_nxt, gM_nxt, p)
  #   y2 = Epnu2.flatten()
  # except TypeError:
  #   y2 = np.zeros(y1.shape)

  # Epnu3 = tnu*DeltaEnup_integ(tnu, gm0, gM0, tt_i, gm, gM, tt_f, gm_nxt, gM_nxt, p)
  # y3 = Epnu3.flatten()
  Epnu4 = tnu*DeltaEnup_new(tnu, tt_i, tt_f, cell, cell_next, cell_0, p)
  y4 = Epnu4.flatten()


  fig, ax = plt.subplots()
  # ax.loglog(tnu, y1, label='func')
  # ax.loglog(tnu, y2, ls='--', label='func and $\\int$')
  # ax.loglog(tnu, y3, ls=':', label='$\\int$ only')
  ax.loglog(tnu, y4)
  ax.set_xlabel("$x=\\nu'/\\nu'_{\\rm B}$")
  ax.set_ylabel("$\\Delta E'_{\\nu'}$")
  transy = transforms.blended_transform_factory(
    ax.transData, ax.transAxes)
  for val, name in zip(x_arr, x_names):
    ax.axvline(val, ls=':', c='k')
    ax.text(val, 1.01, '$x_{\\rm '+name+'}$', transform=transy)
  # for val, name in zip(x_arr2, x_names2):
  #   ax.axvline(val, ls=':', c='b')
  #   ax.text(val, .97, '$x_{\\rm '+name+'}$', c='b', transform=transy)
  ax.legend()

def plot_cellEmission(var, logv, k=450, key='test', logRatio=None,
  largeFreqs=True, integrated=False, withsteps=True, theory=True, spec='int',
  ax_in=None, **kwargs):
  '''
  Plots lightcurve ('lc') or spectrum ('sp')
  integrated = True will perform integration over freq for lightcurve, time for spectrum 
  '''

  if not ax_in:
    fig, ax = plt.subplots()
  else:
    ax = ax_in
  xlabel, ylabel = '', '$\\nu F_{\\nu}/\\nu_{\\rm m} F_{\\nu_{\\rm m}}$'
  logx, logy = False, False
  

  df, env = get_cellData_withDistrib(key, k, logRatio)

  nuobs, Tobs = radEnv(env, largeFreqs=True)
  tnu = nuobs/env.nu0
  Tb = (Tobs-env.Ts)/env.T0
  
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

  if withsteps:
    nuFnu_tot, nuFnu_arr, reg_arr = cell2rad(df, nuobs, Tobs, env, 
        withsteps=withsteps, spec=spec)
  else:
    nuFnu_tot = cell2rad(df, nuobs, Tobs, env, 
        withsteps=withsteps, spec=spec)
  y = nuFnu_tot.flatten()

  line, = ax.plot(x, y, zorder=-1, **kwargs)
  if 'c' in kwargs:
    col = kwargs['c']
  else:
    col = 'r' if k < env.Next+env.Nsh4 else 'b'
    line.set_color(col)
  
  if theory and (var=='sp'):
    # add broken power-law for reference
    cell0 = df.iloc[0]
    cell1 = df.iloc[1]
    
    Ton, Tth, Tej = get_variable(cell0, 'obsT', env)
    tT = (Tobs-Tej)/Tth
    
    x_B = get_variable(cell0, 'nu_B', env)/env.nu0
    # x_m = get_variable(cell0, 'nu_m', env)/env.nu0
    # x_M = get_variable(cell0, 'nu_M', env)/env.nu0
    x_m = x_B * (cell0.gmin**2)
    x_M = x_B * (cell0.gmax**2)
    tc = 1/get_variable(cell0, 'syn', env)
    tdyn = get_tdyn(df, env)
    x_c = x_B * (tc/tdyn)**2
    y_th = x * spectrum_bplaw(tT*x, x_B, x_c, x_m, x_M, env.psyn)

    # normalize to local flux max
    #i = np.nanargmax(y)
    i = np.searchsorted(tT*x, x_m)
    y_th *= y[i]/y_th[i]
    ax.plot(x, y_th, c=col, ls=':')

  if not ax_in:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

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
    sum_mono = np.zeros(y.shape)
    for nuFnu, reg in zip(nuFnu_arr, reg_arr):
      nuFnu = nuFnu
      if reg == 'monoE':
        sum_mono += nuFnu
        plt.autoscale(enable=False)
      ax.plot(x, nuFnu, lw=.9, c=regcol[reg], **kwargs)
    ax.plot(x, sum_mono, c='darkgreen', **kwargs)
  if (logRatio is not None) and (ax_in is None):
    plt.title("$\\log(\\gamma_{\\rm c}/\\gamma_{\\rm m}) = " + f'{logRatio:.1f}$')
  plt.tight_layout()

### cooling regimes
def compare_coolregimes(logs, key='test', k=450, var='sp', logv=0, theory=False):

  df, env = get_cellData_rescaled(key, k, None)
  base = check_coolRegime(df, env)
  fig, ax = plt.subplots()
  ncol = len(logs)
  colors = plt.cm.jet(np.linspace(0,1,ncol))
  names = [f'{logRatio:.1f}' for logRatio in [base if v is None else v for v in logs]]
  dummy_col = []
  xlabel, ylabel = '', '$\\nu F_{\\nu}/\\nu_{\\rm m}F_{\\nu_{\\rm m}}$'

  if var == 'lc':
    xlabel=("$\\bar{T}$")
  elif var == 'sp':
    xlabel=("$\\tilde{\\nu}$")

  for i, logRatio in enumerate(logs):
    plot_cellEmission(var, logv, k, key, logRatio,
      withsteps=False, theory=theory, ax_in=ax, ls='-', c=colors[i])
    dummy_col.append(plt.plot([],[],c=colors[i],ls='-')[0])
  
  ax.legend(dummy_col, names,
    title='$\\log_{10}\\gamma_{\\rm c}/\\gamma_{\\rm m}$')
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)

  fig.tight_layout()

def compare_gmas_coolregimes(logRatio_arr, key='test', k=450):
  '''
  Compare gmas accross logRatios
  '''

def plot_gmas(key, k, logRatio=None, xscale='tc', theory=True,
    ax_in=None, **kwargs):
  '''
  Plots (gmin, gmax) for a given logRatio
  '''

  df, env = get_cellData_withDistrib(key, k, logRatio)
  if ax_in is None:
    fig, ax = plt.subplots()
  else:
    ax = ax_in
  plot_any('gmin', df, logy=True, xscale=xscale, ax_in=ax, **kwargs)
  plot_any('gmax', df, logy=True, xscale=xscale, ax_in=ax, **kwargs)
  ax.set_ylabel('$\\gamma$')
  line0 = ax.get_lines()[0]
  cm = line0.get_color()
  line1 = ax.get_lines()[1]
  cM = line1.get_color()
  dummy_col = [plt.plot([], [], c=col)[0] for col in [cm, cM]]
  ax.legend(dummy_col, ['$\\gamma_\\min$', '$\\gamma_\\max$'], fontsize=16)

  if theory:
    cell0 = df.iloc[0]
    def gma(tt, gma0):
      c0 = (gma0-1)/(gma0+1)
      return (np.exp(2*tt)+c0)/(np.exp(2*tt)-c0)
    gm0, gM0 = cell0[['gmin', 'gmax']]
    lfac = get_variable(cell0, 'lfac', env)
    tc = 1/get_variable(cell0, 'syn', env)
    tt = (df.t - cell0.t)/(lfac*tc)
    gm = gma(tt, gm0)
    gM = gma(tt, gM0)

    xd = line1.get_xdata()
    ax.plot(xd, gm, c='k', ls=':')
    ax.plot(xd, gM, c='k', ls=':')
  plt.tight_layout()


def spectrum_bplaw(x, x_B, x_c, x_m, x_M, p):
  '''
  Broken power-law spectrum (theory
  see Rahaman 2025 (Cooling Regimes) for ref
  x is frequency normalized to nu0 (x_m ~ 1)
  '''

  y = np.zeros(x.shape)
  if x_c <= x_B: # very fast cooling
    reg = 'VFC'
    segment1 = (x >= x_B) & (x < x_m)
    segment2 = (x >= x_m) & (x <= x_M)
    y[segment1] = (x[segment1]/x_B)**(-.5)
    y[segment2] = (x_m/x_B)**(-.5) * (x[segment2]/x_m)**(-p/2)
  elif x_c <= x_m: # fast cooling
    reg = 'FC'
    segment1 = (x >= x_B) & (x <= x_c)
    segment2 = (x > x_c) & (x <= x_m)
    segment3 = (x > x_m) & (x <= x_M)
    y[segment1] = (x[segment1]/x_c)**(1./3)
    y[segment2] = (x[segment2]/x_c)**(-.5)
    y[segment3] = (x_m/x_c)**(-.5) * (x[segment3]/x_m)**(-p/2)
  elif x_c <= x_M: # slow cooling
    reg = 'SC'
    segment1 = (x >= x_B) & (x <= x_m)
    segment2 = (x > x_m) & (x <= x_c)
    segment3 = (x > x_c) & (x <= x_M)
    y[segment1] = (x[segment1]/x_m)**(1./3)
    y[segment2] = (x[segment2]/x_m)**((1-p)/2)
    y[segment3] = (x_c/x_m)**((1-p)/2) * (x[segment3]/x_c)**(-p/2)
  else: # very slow cooling
    reg = 'VSC'
    segment1 = (x >= x_B) & (x <= x_m)
    segment2 = (x > x_m) & (x <= x_M)
    y[segment1] = (x[segment1]/x_m)**(1./3)
    y[segment2] = (x[segment2]/x_m)**((1-p)/2)
  
  print(reg)
  return y


### electron distribution
def synThin_cooling(t, y, adiab, B2):
  '''
  Differential equation to solve for cooling
  Optically thin synchrotron cooling + adiabatic
  f'(y) = - (sigT B**2 / (6 pi me c)) * (y**2-1) + (d rho/d t) / (3 rho) * y
  '''

  # technically y**2 - 1
  return adiab*y - alpha_*B2*(y**2 -1)

def stop_condition(t, y, adiab, B2):
  return y[0]-1.

def evolve_gmas(cell, cell_new, env, Nout=None, func_cool=synThin_cooling):
  '''
  Evolve the electron distribution between two hydro time steps
  cell already contains gma_min and gma_max
  Coolings: optically thin synchrotron and adiabatic
  '''
  if cell.gmax == 1.:
    gm_new, gM_new = 1., 1.
  else:
    tmin = min(cell['t'], cell_new['t']) + env.t0
    tmax = max(cell['t'], cell_new['t']) + env.t0
    lfac = get_variable(cell, 'lfac', env)
    lfac_new = get_variable(cell_new, 'lfac', env)
    lfac_n = .5*(lfac+lfac_new)
    t_span = (tmin/lfac, tmax/lfac_new)
    B2 = get_variable(cell, 'B2', env)
    dtp = .5*(cell['dt']/lfac_n + cell_new['dt']/lfac_n)
    drhodt = (cell['drho']+cell_new['drho'])/dtp
    adiab = drhodt/(3*cell.rho)
    t_eval = None
    if Nout:
      # t_eval = np.linspace(t_span[0], t_span[-1], Nout)
      t_eval = t_span[0]*np.logspace(0, np.log10(t_span[-1]/t_span[0]), Nout)
    if cell.gmin == 1.:
      gm_new = 1.
    else:
      sol_gmin = spi.solve_ivp(func_cool, t_span, [cell.gmin], method='RK23',
        t_eval=t_eval, events=stop_condition, args=(adiab, B2))
      gm_new = sol_gmin.y[0][-1]
    sol_gmax = spi.solve_ivp(func_cool, t_span, [cell.gmax], method='RK23',
      t_eval=t_eval, events=stop_condition, args=(adiab, B2))
    gM_new = sol_gmax.y[0][-1]

  if Nout:
    return sol_gmin, sol_gmax
  else:
    return gm_new, gM_new

def evolve_gmas_full(cell, cell_new, env, func_cool=synThin_cooling):
  '''
  Evolve a distrib of Ngma values
  '''
  varlist = ['gmin'] + [var for var in cell.keys() if 'gma' in var]
  Ngma = len(varlist)

  if cell.gmax == 1.:
    gmas_new = np.ones(Ngma)
  else:
    tmin = min(cell['t'], cell_new['t']) + env.t0
    tmax = max(cell['t'], cell_new['t']) + env.t0
    lfac = get_variable(cell, 'lfac', env)
    lfac_new = get_variable(cell_new, 'lfac', env)
    lfac_n = .5*(lfac+lfac_new)
    t_span = (tmin/lfac, tmax/lfac_new)
    B2 = get_variable(cell, 'B2', env)
    dtp = .5*(cell['dt']/lfac_n + cell_new['dt']/lfac_n)
    drhodt = (cell['drho']+cell_new['drho'])/dtp
    adiab = drhodt/(3*cell.rho)
    gmas = cell[varlist].to_numpy()
    sol = spi.solve_ivp(func_cool, t_span, gmas, method='RK23', args=(adiab, B2))
    gmas_new = sol.y[:,-1]
  
  return gmas_new
    

def generate_cellDistrib_v1(df, env):
  '''
  Generate and evolves the electron distribution, returning dataframe with gmas
  df needs to start at shock! (fromSh = True in get_cellData)
  '''
  out = df.copy()
  its = out.index.to_list()

  out['dt'] = np.gradient(df.t.to_numpy())
  out['drho'] = np.gradient(df.rho.to_numpy())
  out[['gmin', 'gmax']] =  0., 0.
  gmin, gmax = get_vars(out.iloc[0], ['gma_m', 'gma_M'], env)
  out.loc[its[0], varlist] = np.array([gmin, gmax])
  for j, it_next in enumerate(its[1:]):
    cell = out.iloc[j]
    cell_new = out.iloc[j+1]
    gmas_new = evolve_gmas(cell, cell_new, env)
    gmas_new = [max(gma, 1) for gma in gmas_new]
    out.loc[it_next, ['gmin', 'gmax']] = gmas_new
  return out

def generate_cellDistrib(df, env, Ngma=50):
  '''
  Generate and evolves the electron distribution, returning dataframe with gmas
  df needs to start at shock! (fromSh = True in get_cellData)
  '''
  out = df.copy()
  its = out.index.to_list()

  out['dt'] = np.gradient(df.t.to_numpy())
  out['drho'] = np.gradient(df.rho.to_numpy())
  varlist = ['gmin'] + [f'gma_e{i}' for i in range(1,Ngma-1)] + ['gmax']
  out[varlist] =  [0. for i in range(Ngma)]
  gmin, gmax = get_vars(out.iloc[0], ['gma_m', 'gma_M'], env)
  out.loc[its[0], varlist] = np.logspace(np.log10(gmin), np.log10(gmax), Ngma)
  for j, it_next in enumerate(its[1:]):
    cell = out.iloc[j]
    cell_new = out.iloc[j+1]
    gmas_new = evolve_gmas_full(cell, cell_new, env)
    gmas_new = [max(gma, 1) for gma in gmas_new]
    out.loc[it_next, varlist] = gmas_new
  return out

# def generate_fluxNdistrib(df, env, var=None, logv=None, largeFreqs=True):
#   '''
#   Evolves the electron distribution and create flux at the same time
#   '''

#   nuobs, Tobs = radEnv(env, largeFreqs)
#   tnu = nuobs/env.nu0
#   Tb = (Tobs-env.Ts)/env.T0
#   if (var is not None) and (logv is not None):
#     plotting = True
#     if var == 'lc':
#       i = find_closest(tnu, 10**logv)
#       nuobs = np.array([nuobs[i]])
#       x = Tb
#       xlabel=("$\\bar{T}$")
#     elif var == 'sp':
#       i = find_closest(Tb, 10**logv)
#       Tobs = np.array([Tobs[i]])
#       x = tnu
#       xlabel=("$\\tilde{\\nu}$")
#       logx, logy = True, True

#   if plotting:
#     nuFnu_tot, nuFnu_arr, reg_arr = cell2rad(df, nuobs, Tobs, env, withsteps=True, spec='BPL')
#   else:
#     nuFnu_tot = cell2rad(df, nuobs, Tobs, env, spec='BPL')


def cell2rad(dk, nuobs, Tobs, env, withsteps=True, spec='int'):
  '''
  Contribution to luminosity in obs frame from 1 cell 
  '''

  # proper shaping of output
  if (type(Tobs)==np.ndarray) and (type(nuobs)==np.ndarray):
    out = np.zeros((len(Tobs), len(nuobs)))
  elif (type(nuobs)==float) or (len(nuobs) == 1):
    out = np.zeros(Tobs.shape)
    if type(nuobs) == np.ndarray: nuobs = nuobs[0]
  elif (type(Tobs) == float) or (len(Tobs) == 1):
    out = np.zeros(nuobs.shape)
    if type(Tobs) == np.ndarray: Tobs = Tobs[0]
  
  tnu = nuobs/env.nu0

  Nj = len(dk)
  cell_0 = dk.iloc[0]
  nL_arr, reg_arr = [], []
  for j in range(Nj-1):
    cell = dk.iloc[j]
    cell_next = dk.iloc[j+1]
    if cell['gmax'] == 1.:
      break
    Lnu, reg = get_Lnu_cooling(cell, cell_next, cell_0, nuobs, Tobs, env)
    nL = tnu * Lnu/env.L0
    out += nL
    nL_arr.append(nL)
    reg_arr.append(reg)

  if withsteps:
    return out, np.array(nL_arr), reg_arr
  else:
    return out

def get_Lnu_cooling(cell, cell_next, cell_0, nuobs, Tobs, env):
  '''
  Derive the radiation emitted between states cell and cell_next
  comparison between (gmin, gmax) gives the 'local cooling regime'
  normalization is derived using hydro state cell
  '''
  if (type(Tobs)==np.ndarray) and (type(nuobs)==np.ndarray):
    out = np.zeros((len(Tobs), len(nuobs)))
    nuobs = nuobs[np.newaxis, :]
    Tobs  = Tobs[:, np.newaxis]
  elif (type(nuobs)==float) or (len(nuobs) == 1):
    out = np.zeros(Tobs.shape)
    if type(nuobs) == np.ndarray: nuobs = nuobs[0]
  elif (type(Tobs) == float) or (len(Tobs) == 1):
    out = np.zeros(nuobs.shape)
    if type(Tobs) == np.ndarray: Tobs = Tobs[0]

  gm0, gM0 = cell_0[['gmin', 'gmax']]
  gm, gM, gm_nxt, gM_nxt = *cell[['gmin', 'gmax']], *cell_next[['gmin', 'gmax']]
  g1 = min(gm, gM_nxt)
  g2 = max(gm, gM_nxt)
  reg = ''
  if not gm:
    return out, reg

  Ton, Tth, Tej = get_variable(cell, 'obsT', env)
  tT = ((Tobs - Tej)/Tth)
  lfac0 = get_variable(cell_0, 'lfac', env)
  # lfac = get_variable(cell, 'lfac', env)
  # lfac_nxt = get_variable(cell_next, 'lfac', env)
  lfac, lfac_nxt = lfac0, lfac0
  tc = 1/get_variable(cell_0, 'syn', env)
  # tt_i = (cell['t'] - cell_0['t'])/(tc*lfac)
  # tt_f = (cell_next['t'] - cell_0['t'])/(tc*lfac)
  tt_i = (cell['t']/lfac - cell_0['t']/lfac0)/tc
  tt_f = (cell_next['t']/lfac_nxt - cell_0['t']/lfac0)/tc

  cell = cell_0
  nup_B = get_variable(cell, "nup_B", env)
  Dop = get_variable(cell, 'Dop', env)
  p = env.psyn
  nup = nuobs/Dop
  Pe = get_variable(cell, 'Pmax', env)
  V3 = get_variable(cell, 'V3p', env)

  reg = 'fast' if gm >= gM_nxt else 'slow'
  tnu = nup/nup_B
  Epnu = np.zeros(out.shape)
  K = (p-1)/(gm0**(1-p)-gM0**(1-p))
  Epnu_m = K*Pe*tc*V3 # Pe already contains ne
  #Epnu = DeltaEnup_mixed(tT*tnu, gm0, gM0, tt_i, gm, gM, tt_f, gm_nxt, gM_nxt, p)
  if tt_i > tt_f:
    print(tt_i, tt_f, tt_f-tt_i)
    Epnu = 0.
    print('negative time difference!')
  else:
    Epnu = DeltaEnup_new(tT*tnu, tt_i, tt_f, cell, cell_next, cell_0, p)
  
  Epnu *= Epnu_m

  out += ((1+env.z)/Tth) * Epnu * tT**-2
  if (len(nuobs) == 1) or (len(Tobs)==1):
    out = out.flatten()
  return out, reg

def DeltaEnup_new(tnu_arr, tt_i, tt_f, cell, cell_next, cell_0, p):
  
  if len(tnu_arr.shape) == 1:
    tnu_arr = np.array([tnu_arr])
  rows, cols = tnu_arr.shape
  out = np.zeros((rows, cols))
  sqtnu_arr = np.sqrt(tnu_arr)

  gm0, gM0 = cell_0[['gmin', 'gmax']]
  gm, gM, gm_nxt, gM_nxt = *cell[['gmin', 'gmax']], *cell_next[['gmin', 'gmax']]
  g1 = min(gm, gM_nxt)
  g2 = max(gm, gM_nxt)
  # local integration on gma_f
  def func_gmin(tt):
    return gm/(1+gm*tt)
  def func_gmax(tt):
    return gM/(1+gM*tt)

  if len(tnu_arr.shape) == 1:
    tnu_arr = np.array([tnu_arr])
  rows, cols = tnu_arr.shape
  for i in range(0, rows):
    for j in range(0, cols):
      sqtnu = sqtnu_arr[i,j]
      def func_gmanu(tt):
        return sqtnu
      tt_num = min(max(1/sqtnu - 1/gm, 0) + tt_i, tt_f)
      tt_nuM = min(max(1/sqtnu - 1/gM, 0) + tt_i, tt_f)
      if sqtnu <= 1.:
        # tnu already normalized to cyclotron freq
        continue
      elif sqtnu <= gm_nxt:
        out[i,j] += DeltaE_func(tt_i, tt_f, gm0, gM0, p)
      elif sqtnu <= g1:
        term1 = dEfunc_semiinteg_gmaf(tt_i, tt_num, func_gmin, func_gmax, p)
        term2 = dEfunc_semiinteg_gmaf(tt_num, tt_f, func_gmanu, func_gmax, p)
        out[i,j] += term1 + term2
      elif sqtnu <= g2:
        if gm>=gM_nxt: # fast cooling
          term1 = dEfunc_semiinteg_gmaf(tt_i, tt_num, func_gmin, func_gmax, p)
          term2 = dEfunc_semiinteg_gmaf(tt_num, tt_nuM, func_gmanu, func_gmax, p)
          out[i,j] += term1 + term2
        else: # slow cooling
          out[i,j] += dEfunc_semiinteg_gmaf(tt_i, tt_f, func_gmanu, func_gmax, p)
      elif sqtnu <= gM:
        out[i,j] += dEfunc_semiinteg_gmaf(tt_i, tt_nuM, func_gmanu, func_gmax, p)
  out *= tnu_arr**(1/3.)
  return out

def DeltaE_sumgmanu(sqtnu, t_i, t_f, cell, cell_next, cell_0, p):
  '''
  Contribution to (normalized) obs frequency tnu = sqtnu**2 > gma_m**2
  '''

  out = 0
  varlist = ['gmin'] + [var for var in cell.keys() if 'gma' in var]
  gmas_0 = cell_0[varlist].to_numpy()
  gM0 = gmas_0[-1]
  gmas = cell[varlist].to_numpy()
  gmas_next = cell_next[varlist].to_numpy()
  i1 = np.searchsorted(gmas, sqtnu)
  i2 = np.searchsorted(gmas_next, sqtnu)
  N = i2 - i1
  ilist = [i for i in range(i1, i2)]
  gmas0list = gmas_0[ilist]
  tt_range = np.linspace(t_i, t_f, N+1)
  for i in range(N):
    gmanu0 = gmas0list[i]
    t1, t2 = tt_range[i], tt_range[i+1]
    out += DeltaE_func(t1, t2, gmanu0, gM0, p)
  return out

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

def spectrum_bplaw_mono(x, x_B, x_m, x_M):
  y = np.zeros(x.shape)
  segment1 = (x>=x_B) & (x<=x_m)
  segment2 = (x>=x_m) & (x<=x_M)
  x /= x_m
  y[segment1] = x[segment1]**(1/3.)
  y[segment2] = x[segment2]**(-.5)
  return y

def get_stepenv(j, key='test', k=450, logRatio=None):
  '''
  Returns main variables for radiation calculation
  '''
  df, env = get_cellData_withDistrib(key, k, logRatio)
  cell_0, cell, cell_next = df.iloc[0], df.iloc[j], df.iloc[j+1]
  gm0, gM0, gm, gM, gm_nxt, gM_nxt = *cell_0[['gmin', 'gmax']], *cell[['gmin', 'gmax']], *cell_next[['gmin', 'gmax']]
  # Ton, Tth, Tej = get_variable(cell, 'obsT', env)
  # tT = ((Tobs - Tej)/Tth)
  lfac0 = get_variable(cell_0, 'lfac', env)
  # lfac = get_variable(cell, 'lfac', env)
  # lfac_nxt = get_variable(cell_next, 'lfac', env)
  lfac, lfac_nxt = lfac0, lfac0
  tc = 1/get_variable(cell_0, 'syn', env)
  # tt_i = (cell['t'] - cell_0['t'])/(tc*lfac)
  # tt_f = (cell_next['t'] - cell_0['t'])/(tc*lfac)
  tt_i = (cell['t']/lfac - cell_0['t']/lfac0)/tc
  tt_f = (cell_next['t']/lfac_nxt - cell_0['t']/lfac0)/tc
  return tt_i, tt_f, gm0, gM0, gm, gM, gm_nxt, gM_nxt, env


def plot_dE_step(j, key='test', k=450, logRatio=None, distrib='f', withsteps=True, ax_in=None):
  '''

  '''
  if withsteps:
    ax_in = None
  if ax_in is None:
    fig, ax = plt.subplots()
  else:
    ax = ax_in

  tnu_arr = np.logspace(0, 12, 5000)
  dE = get_DeltaE_step(j, key, k, logRatio, distrib)
  ax.loglog(tnu_arr, dE)
  if withsteps:
    plot_instantPnus_step(j, key=key, k=k, logRatio=logRatio, distrib=distrib, ax_in=ax)

def plot_instantPnus_step(j, Nj=5, key='test', k=450, logRatio=None, distrib='i', ax_in=None):
  '''
  Plots Nj instantaneous spectra sampled between steps j and j+1 of data
  '''

  if ax_in is None:
    fig, ax = plt.subplots()
  else:
    ax = ax_in

  if distrib == 'f':
    func = instant_Pnu_fromgmaf_num
  elif distrib == 'i':
    func = instant_Pnu_fromgmai_num
  else:
    print("'distrib' argument needs to be 'i' or 'f'")
    return 0.
  tt_i, tt_f, gm0, gM0, gm, gM, gm_nxt, gM_nxt, env = get_stepenv(j, key, k, logRatio)
  p = env.psyn
  t_arr = np.linspace(tt_i, tt_f, Nj)
  tnu = np.logspace(0, 12, 5000)
  sqtnu = np.sqrt(tnu)
  colors = plt.cm.jet(np.linspace(0,1,Nj))
  for i, tt in enumerate(t_arr):
    dtt = (tt-tt_i)
    gmin = gm/(1+dtt*gm)
    gmax = gM/(1+dtt*gM)
    Pnu = func(dtt, tnu, gm, gM)
    # Pnu2 = integrated_instantPnu_fromgmai(dtt, tnu, gm, gM)
    ax.loglog(tnu, Pnu, c=colors[i], label='$\\tilde{t} = $'+f'{dtt/(tt_f-tt_i):.2f}'+'$\\tilde{t}_f$')
    # ax.loglog(tnu, Pnu2, c=colors[i], ls='--')
    ax.axvline(gmin**2, c=colors[i], ls=':')
    ax.axvline(gmax**2, c=colors[i], ls=':')
  ax.legend()

def plot_dE_cell(k=450, key='test', logRatio=None, jmax=None):
  tnu = np.logspace(0, 12, 5000)
  dE = get_DeltaE_cell(k, key, logRatio, jmax)
  fig, ax = plt.subplots()
  ax.loglog(tnu, dE)


def get_DeltaE_cell(k=450, key='test', logRatio=None, jmax=None, distrib='f'):
  '''
  '''
  tnu_arr = np.logspace(0, 12, 5000)
  dE = np.zeros(tnu_arr.shape)
  df, env = get_cellData_withDistrib(key, k, logRatio)
  Nj = jmax if jmax else len(df)
  for j in range(Nj-1):
    cell = df.iloc[j]
    if cell['gmax'] == 1.:
      break
    dE += get_DeltaE_step(j, key, k, logRatio, distrib)
  return dE


def get_DeltaE_step(j, key='test', k=450, logRatio=None, distrib='f'):
  '''
  '''
  if distrib == 'f':
    func = instant_Pnu_numfunc_gmaf
  elif distrib == 'i':
    func = instant_Pnu_numfunc_gmai
  else:
    print("'distrib' argument needs to be 'i' or 'f'")
    return 0.
  tnu_arr = np.logspace(0, 12, 5000)
  tt_i, tt_f, gm0, gM0, gm, gM, gm_nxt, gM_nxt, env = get_stepenv(j, key, k, logRatio)
  p = env.psyn
  K = (p-1)/(gm**(1-p)-gM**(1-p))
  dE = np.zeros(tnu_arr.shape)
  for i, tnu in enumerate(tnu_arr):
    dE[i] += spi.quad(func, 0., tt_f-tt_i, args=(tnu, gm, gM, p))[0]
  dE *= K
  return dE

def gma_cooled_ex(tt, gma0):
  '''
  gamma at tt = (t'-t'0)/t'c
  '''
  c0 = (gma0-1)/(gma0+1)
  gma = (np.exp(2*tt)+c0)/(np.exp(2*tt)-c0)
  return gma

def gma_heated_ex(tt, gm):
  '''
  Inverse of gma_cooled
  '''
  e = np.exp(2*tt)
  em = 1-e
  ep = 1+e
  gma = (em + gm*ep)/(ep - gm*em)
  return gma

def gma_cooled(tt, gm0):
  return gm0/(1+gm0*tt)
def gma_heated(tt, gm):
  return gm/(1-gm*tt)


def instant_Pnu_fromgmaf_anl(tt, tnu_arr, gm, gM, p=2.5):
  '''
  "instantaneous" spectra from analytical integration over gma_f
  gmin, gmax = gm, gM at tt = 0
  '''
  
  Pnu = [instant_Pnu_anlfunc_gmaf(tt, tnu_arr[i], gm, gM, p) for i in range(len(tnu_arr))]
  return np.array(Pnu)

def instant_Pnu_anlfunc_gmaf(tt, tnu, gm, gM, p=2.5):
  '''
  Integrand for the instantaneous spectra, from the instantaneous electron distrib
  analytical integration
  '''

  sqtnu = np.sqrt(tnu)
  gmin = gma_cooled(tt, gm)
  gmax = gma_cooled(tt, gM)
  Pnu = 0
  if sqtnu >= 1.:
    if sqtnu <= gmin:
      Pnu = Pfunc_integ_gmaf(tt, gmin, gmax, p)
    elif sqtnu <= gmax:
      Pnu = Pfunc_integ_gmaf(tt, sqtnu, gmax, p)
  Pnu *= tnu**(1/3)
  return Pnu

def Pfunc_integ_gmaf(tt, gmin, gmax, p):
  '''
  Analytical integration of dn/d\gma P(\gma) over \gma_f
  '''
  return tt**(p-1/3.) * (Bfunc_gmaf(gmax, tt, p) - Bfunc_gmaf(gmin, tt, p))

def Bfunc_gmaf(gma, tt, p):
  '''
  Incomplete beta function that appears when integrating over gma_f
  '''
  return betai(1/3.-p, p-1, gma*tt)

def func_gmaf(gma, tt, p):
  '''
  Integrand when written as a function of the instantaneous LF
  '''
  return gma**(-p-2/3)*(1-gma*tt)**(p-2)

def Enu_fromgmaf_num_dblquad(dtt, tnu_arr, gm, gM, p=2.5):
  '''
  Emitted energy per frequency over normalize dtime interval dtt
  dblquad
  '''

  Enu = np.zeros(tnu_arr.shape)
  sqtnu_arr = np.sqrt(tnu_arr)
  # define integral bounds
  def gmin(t):
    return gma_cooled(t, gm)
  def gmax(t):
    return gma_cooled(t, gM)

  g0 = gmin(dtt)
  g1 = min(gm, gmax(dtt))
  g2 = gM
  for i, sqtnu in enumerate(sqtnu_arr):
    def gnu(t):
      return sqtnu
    if sqtnu < 1:
      continue
    elif sqtnu <= g0:
      Enu[i] = spi.dblquad(func_gmaf, 0, dtt, gmin, gmax, args=([p]))[0]
    elif sqtnu <= g1:
      def func(t):
        return gmin(t) - sqtnu
      tt_num = brentq(func, 0, dtt)
      tt_max = dtt
      if (gmax(dtt)<sqtnu):
        def func(t):
          return gmax(t) - sqtnu
        tt_max = brentq(func, 0, dtt)
      int1 = spi.dblquad(func_gmaf, 0, tt_num, gmin, gmax, args=([p]))[0]
      int2 = spi.dblquad(func_gmaf, tt_num, tt_max, gnu, gmax, args=([p]))[0]
      Enu[i] = int1 + int2
    elif sqtnu <= g2:
      tt_max = dtt
      if (gmax(dtt)<sqtnu):
        def func(t):
          return gmax(t) - sqtnu
        tt_max = brentq(func, 0, dtt)
      Enu[i] = spi.dblquad(func_gmaf, 0, tt_max, gnu, gmax, args=([p]))[0]

  Enu *= tnu_arr**(1/3)
  return Enu


def Enu_fromgmaf_num_quadquad(dtt, tnu_arr, gm, gM, p=2.5):
  '''
  Emitted energy per frequency over normalize dtime interval dtt
  two consecutive integrals
  '''
  Enu = [spi.quad(instant_Pnu_numfunc_gmaf, 0., dtt, args=(tnu, gm, gM, p))[0] for tnu in tnu_arr]
  return Enu

def instant_Pnu_fromgmaf_num(tt, tnu_arr, gm, gM, p=2.5):
  '''
  "instantaneous" spectra from numerical integration over gma_f
  gmin, gmax = gm, gM at tt = 0
  '''
  
  Pnu = [instant_Pnu_numfunc_gmaf(tt, tnu_arr[i], gm, gM, p) for i in range(len(tnu_arr))]
  return np.array(Pnu)

def instant_Pnu_numfunc_gmaf(tt, tnu, gm, gM, p=2.5):
  '''
  Integrand for the instantaneous spectra, from the instantaneous electron distrib
  numerical integration
  '''

  sqtnu = np.sqrt(tnu)
  Pnu = 0
  gmin = gma_cooled(tt, gm)
  gmax = gma_cooled(tt, gM)
  points = [1/tt] if tt else None
  if sqtnu >= 1:
    if sqtnu <= gmin:
      Pnu = spi.quad(func_gmaf, gmin, gmax, args=(tt, p), points=points)[0]
    elif sqtnu <= gmax:
      Pnu = spi.quad(func_gmaf, sqtnu, gmax, args=(tt, p), points=points)[0]
    else:
      Pnu = 0.
  Pnu *= tnu**(1/3)
  return Pnu


def instant_Pnu_fromgmai_anl(tt, tnu_arr, gm, gM, p=2.5):
  '''
  "instantaneous" spectra from analytical integration over gma_i
  gmin, gmax = gm, gM at tt = 0
  '''
  if tt != 0:
    def func(tnu):
      return instant_Pnu_anlfunc_gmai(tt, tnu, gm, gM, p)
  else:
    def func(tnu):
      return instant_Pnu_anlfunc_gmai_t0(tnu, gm, gM, p)
  
  Pnu = [func(tnu_arr[i]) for i in range(len(tnu_arr))]
  return np.array(Pnu)

def instant_Pnu_fromgmai_num(tt, tnu_arr, gm, gM, p=2.5):
  '''
  "instantaneous" spectra from numerical integration over gma_i
  gmin, gmax = gm, gM at tt = 0
  '''
  
  Pnu = [instant_Pnu_numfunc_gmai(tt, tnu_arr[i], gm, gM, p) for i in range(len(tnu_arr))]
  return np.array(Pnu)


def instant_Pnu_anlfunc_gmai_t0(tnu, gm, gM, p=2.5):
  '''
  Analytical integration of dn/d\gma P(\gma) over \gma_i at t0
  (no cooling has occured)
  '''
  a = 1/3 - p
  sqtnu = np.sqrt(tnu)
  Pnu = 0
  if sqtnu >= 1.:
    if sqtnu <= gm:
      Pnu = (gm**a - gM**a)/(3*p-1)
    elif sqtnu <= gM:
      Pnu = (sqtnu**a - gM**a)/(3*p-1)
    else:
      Pnu = 0.
  Pnu *= tnu**(1/3)
  return Pnu

def instant_Pnu_anlfunc_gmai(tt, tnu, gm, gM, p=2.5):
  '''
  Integrand for the instantaneous spectra, from the initial electron distrib
  analytical integration
  '''

  sqtnu = np.sqrt(tnu)
  gmin = gma_cooled(tt, gm)
  gmax = gma_cooled(tt, gM)
  Pnu = 0
  if sqtnu >= 1.:
    if sqtnu <= gmin:
      Pnu = Pfunc_integ_gmai(tt, gmin, gmax, p)
    elif sqtnu <= gmax:
      gmnu = gma_heated(tt, sqtnu)
      gmnu = max(gmnu, gm)
      gmnu = min(gmnu, gM)
      Pnu = Pfunc_integ_gmai(tt, gmnu, gmax, p)
    else:
      Pnu = 0.
  Pnu *= tnu**(1/3)
  return Pnu

def Pfunc_integ_gmai(tt, gma_l, gma_u, p):
  '''
  Analytical integration of dn/d\gma P(\gma) over \gma_i
  '''
  return (-1)**(-p) * tt**(p-1/3) * (Bfunc_gmai(gma_u, tt, p) - Bfunc_gmai(gma_l, tt, p))
  
def Bfunc_gmai(gma, tt, p):
  '''
  Incomplete beta function that appears when integrating over gma_i
  '''
  return betai(p-1, 5./3., -1./(gma*tt))

def func_gmai(gma, tt, p):
  '''
  Integrand when written as a function of the initial LF
  '''
  return gma**(-p-2/3)*(1+gma*tt)**(2/3)


def instant_Pnu_numfunc_gmai(tt, tnu, gm, gM, p=2.5):
  '''
  Integrand for the instantaneous spectra, from the initial electron distrib
  numerical integration
  '''

  sqtnu = np.sqrt(tnu)
  Pnu = 0
  gmin = gma_cooled(tt, gm)
  gmax = gma_cooled(tt, gM)
  if sqtnu > 1:
    if sqtnu <= gmin:
      Pnu = spi.quad(func_gmai, gm, gM, args=(tt, p))[0]
    elif sqtnu <= gmax:
      gmnu = gma_heated(tt, sqtnu)
      gmnu = max(gmnu, gm)
      gmnu = min(gmnu, gM)
      Pnu = spi.quad(func_gmai, gmnu, gM, args=(tt, p))[0]
    else:
      Pnu = 0.
  Pnu *= tnu**(1/3)
  return Pnu

def integrated_instantPnu_fromgmai(tt, tnu_arr, gm, gM, p=2.5):
  '''
  Integrated form of P'nu from the form in gma_i
  '''
  sqtnu = np.sqrt(tnu_arr)
  int = 0
  gmin = gm/(1+gm*tt)
  gmax = gM/(1+gM*tt)
  gmnu = sqtnu/(1-sqtnu*tt)
  gmnu[gmnu<gm] = gm
  Pnu = np.zeros(tnu_arr.shape)
  segment1 = (sqtnu>=1) & (sqtnu<=gmin)
  segment2 = (sqtnu>gmin)&(sqtnu<=gmax)
  if tt == 0:
    Pnu[segment1] = Pfunc_integ_gmai_t0(gm, gM, p)
    Pnu[segment2] = Pfunc_integ_gmai_t0(gmnu[segment2], gM, p)
  else:
    print(Bfunc_gmai(gm, tt, p), Bfunc_gmai(gM, tt, p))
    val0 = Pfunc_integ_gmai(tt, gm, gM, p)
    Pnu[segment1] = val0
    Pnu[segment2] = Pfunc_integ_gmai(tt, gmnu[segment2], gM, p)
  Pnu *= tnu_arr**(1/3)
  return Pnu



def plot_instant_dndgma(tt_arr, gm0, gM0, p=2.5):
  '''
  dn/dgma_f at normalized time tt
  '''

  fig, ax = plt.subplots()
  if type(tt_arr) == float:
    tt_arr = [tt_arr]
  
  x = np.logspace(0, int(np.log10(gM0))+1, 200)
  y0 = np.zeros(len(x))
  K0 = 1/(gm0**(1-p)-gM0**(1-p))
  seg1 = (x>=gm0) & (x<=gM0)
  y0[seg1] = K0 * x[seg1]**(-p)
  ax.loglog(x, y0, label='$\\tilde{t}=0$')

  for tt in tt_arr:
    gm, gM = gm0/(1+gm0*tt), gM0/(1+gM0*tt)
    seg2 = (x>=gm) & (x<=gM)
    y = np.zeros(len(x))
    y[seg2] = K0 * (x[seg2]**(-p)) * (1-(tt*x[seg2]))**(p-2)
    ax.loglog(x, y, label='$\\log_{10}\\tilde{t}='+f'{int(np.log10(tt))}$')

  #ax.axvline(tt**-1, ls=':', c='k')
  ax.set_ylabel('$dn/d\\gamma$')
  ax.set_xlabel('$\\gamma$')
  ax.legend()

def plot_instantPnu(tt, gm0=1e3, gM0=1e7, p=2.5, slopes=False, ax_in=None):
  '''
  P'_nu' at normalized time tt
  '''

  x = np.logspace(0, 2*int(np.log10(gM0))+1, 600)
  xm0, xM0 = gm0**2, gM0**2
  xm, xM = (gm0/(1+gm0*tt))**2, (gM0/(1+gM0*tt))**2
  xc = tt**-2
  y = Pnu_localCool(x, tt, gm0, gM0, p)

  if ax_in is None:
    fig, ax = plt.subplots()
  else:
    ax= ax_in

  ax.set_xscale('log')
  if slopes:
    logx = np.log10(x)
    logy = np.log10(y[y>0])
    z = np.zeros(y.shape)
    z[y>0] = np.gradient(logy, logx[y>0])
    y = z
    vals = [1/3., -.5, (1-p)/2, -p/2]
    names = ['$\\frac{1}{3}$', '$\\frac{-1}{2}$',
        '$\\frac{-(p-1)}{2}$', '$\\frac{-p}{2}$']
    transx = transforms.blended_transform_factory(
      ax.transAxes, ax.transData)
    for val, name in zip(vals, names):
      ax.axhline(val, ls='--', c='b')
      ax.text(1.01, val, name, c='b', transform=transx)

  else:
    ax.set_yscale('log')
    # some testing
    FC = Pnu_lFC(x, tt, gm0, gM0, p)
    SC = Pnu_lSC(x, tt, gm0, gM0, p)
    ax.plot(x, FC, ls='--', c='r')
    ax.plot(x, SC, ls='--', c='b')
    ax.plot(x, SC+FC, ls='--', c='k')

    
  ax.plot(x, y, label='$\\tilde{t}=10^{'+f'{np.log10(tt):.1f}'+'}$')
  ax.set_xlabel("$x=\\nu'/\\nu'_{\\rm B}$")
  transy = transforms.blended_transform_factory(
      ax.transData, ax.transAxes)
  for val, name in zip([xm, xM, xc, xm0, xM0], ['m', 'M', 'c', 'm0', 'M0']):
    ax.axvline(val, ls=':', c='k')
    ax.text(val, 1.02, '$x_{\\rm '+name+'}$', transform=transy)



def Pnu_localCool(x, tt, gm0, gM0, p):
  '''
  Instantaneous spectrum of a cooling electron distribution
    using asymptotical forms func_Pnu_lSC and func_Pnu_lFC
    x = nu'/nu'_B ; Pnu in units K P'e
  '''
  out = np.zeros(x.shape)

  gmac = tt**-1
  sqx = 1/np.sqrt(x)
  x_m = (gm0/(1+gm0*tt))**2
  x_M = (gM0/(1+gM0*tt))**2
  gma_nu = 1/(sqx-tt)
  segment1 = (x >= 1) & (x < x_m)
  segment2 = (x >= x_m) & (x <= x_M)
  if gmac < gm0:
    print('local fast cooling')
    out[segment1] = (2/3) * tt**(2/3)
    out[segment2] = func_Pnu_lFC(tt, gma_nu[segment2], gM0, p)
  elif gmac > gM0:
    print('local slow cooling')
    out[segment1] = func_Pnu_lSC(tt, gm0, gM0, p)
    out[segment2] = func_Pnu_lSC(tt, gma_nu[segment2], gM0, p)
  else:
    print('intermediate regime')
    x_c = gmac**2
    segment2 = (x >= x_m) & (x < x_M)
    #segment3 = (x >= x_c) & (x <= x_M)
    out[segment1] = func_Pnu_lSC(tt, gm0, gM0, p) #+ func_Pnu_lFC(tt, gmac, gM0, p)
    out[segment2] = func_Pnu_lSC(tt, gma_nu[segment2], gM0, p) #+ func_Pnu_lFC(tt, gmac, gM0, p)
    #out[segment3] = func_Pnu_lFC(tt, gma_nu[segment3], gM0, p)
  
  out *= x**(1/3)
  return out

def Pnu_lSC(x, tt, gm0, gM0, p):

  x_m = (gm0/(1+gm0*tt))**2
  x_M = (gM0/(1+gM0*tt))**2
  gmac = tt**-1
  sqx = 1/np.sqrt(x)
  gma_nu = 1/(sqx-tt)

  out = np.zeros(x.shape)
  segment1 = (x >= 1) & (x < x_m)
  segment2 = (x >= x_m) & (x <= x_M)
  out[segment1] = func_Pnu_lSC(tt, gm0, gM0, p)
  out[segment2] = func_Pnu_lSC(tt, gma_nu[segment2], gM0, p)
  out *= x**(1/3)
  return out

def Pnu_lFC(x, tt, gm0, gM0, p):

  x_m = (gm0/(1+gm0*tt))**2
  x_M = (gM0/(1+gM0*tt))**2
  gmac = tt**-1
  sqx = 1/np.sqrt(x)
  gma_nu = 1/(sqx-tt)

  out = np.zeros(x.shape)
  segment1 = (x >= 1) & (x < x_m)
  segment2 = (x >= x_m) & (x <= x_M)
  out[segment1] = (2/3) * tt**(2/3)
  out[segment2] = func_Pnu_lFC(tt, gma_nu[segment2], gM0, p)
  out *= x**(1/3)
  return out


def func_Pnu_lSC(tt, gma_l, gma_u, p):
  '''
  Instantaneous spectrum in the 'locally slow cooling' regime,
    in units K P'_e,max tnu**1/3
  '''
  a = 1/3 - p
  out = (3/(3*p-1)) * (gma_l**a - gma_u**a)
  return out

def func_Pnu_lFC(tt, gma_l, gma_u, p):
  '''
  Instantaneous spectrum in the 'locally fast cooling' regime,
    in units K P'_e,max tnu**1/3
  '''
  a = 1-p
  out = (2/3)*(p-1) * tt**(2/3) * (gma_l**a - gma_u**a)
  return out

def DeltaEnup_integ(tnu_arr, gm0, gM0, tt_i, gm, gM, tt_f, gm_next, gM_next, p):
  '''
  Delta E'_{\nu'} spectrum between cell and cell_next in units K P'_{e, max} t'_c
  only integral form
  '''
  if len(tnu_arr.shape) == 1:
    tnu_arr = np.array([tnu_arr])
  rows, cols = tnu_arr.shape
  out = np.zeros((rows, cols))
  sqtnu_arr = np.sqrt(tnu_arr)

  def func_gmaf(gma, t):
    return gma**(-p-2/3.) * (1-gma*t)**(p-2)
  def gma_m(tt):
    return gm0/(1+gm0*tt)
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
        out[i,j] += DeltaE_integ_f(tt_i, tt_f, gm, gM, p)
        #out[i,j] += spi.dblquad(func_gmaf, tt_i, tt_f, gma_m, gma_M)[0]
      elif sqtnu <= g1:
        tt_num = min(1/sqtnu - 1/gm0, tt_f)
        term1 = DeltaE_integ_cstbnd(tt_i, tt_num, gm0, gM0, p)
        term2 = DeltaE_integ(tt_num, tt_f, sqtnu, gM0, p)
        # term1 = spi.dblquad(func_gmaf, tt_i, tt_num, gma_m, gma_M)[0]
        # term2 = spi.dblquad(func_gmaf, tt_num, tt_f, gma_nu, gma_M)[0]
        out[i,j] += term1 + term2
      elif sqtnu <= g2:
        if gm>=gM_next: # fast cooling
          tt_num = max(1/sqtnu - 1/gm0, tt_i)
          tt_nuM = min(1/sqtnu - 1/gM0, tt_f)
          term1 = DeltaE_integ_cstbnd(tt_i, tt_num, gm0, gM0, p)
          term1 = DeltaE_integ(tt_num, tt_nuM, sqtnu, gM0, p)
          # term1 = spi.dblquad(func_gmaf, tt_i, tt_num, gma_m, gma_M)[0]
          # term2 = spi.dblquad(func_gmaf, tt_num, tt_nuM, gma_nu, gma_M)[0]
          out[i,j] += term1 + term2
        else: # slow cooling
          out[i,j] += DeltaE_integ(tt_i, tt_f, sqtnu, gM0, p)
          # out[i,j] += spi.dblquad(func_gmaf, tt_i, tt_f, gma_nu, gma_M)[0]
      elif sqtnu <= gM:
        tt_nuM = min(1/sqtnu - 1/gM0, tt_f)
        out[i,j] += DeltaE_integ(tt_i, tt_nuM, sqtnu, gM0, p)
        # out[i,j] += spi.dblquad(func_gmaf, tt_i, tt_nuM, gma_nu, gma_M)[0]

  out *= tnu_arr**(1/3.)
  return out

def DeltaEnup_mixed(tnu_arr, gm0, gM0, tt_i, gm, gM, tt_f, gm_next, gM_next, p):
  '''
  Delta E'_{\nu'} spectrum between cell and cell_next in units K P'_{e, max} t'_c
  mix and match between integral forms and integrated
  '''
  if len(tnu_arr.shape) == 1:
    tnu_arr = np.array([tnu_arr])
  rows, cols = tnu_arr.shape
  out = np.zeros((rows, cols))
  sqtnu_arr = np.sqrt(tnu_arr)

  def func_gmaf(gma, t):
    return gma**(-p-2/3.) * (1-gma*t)**(p-2)
  def gma_M(tt):
    return gM/(1+gM*tt)

  
  g1 = min(gm, gM_next)
  g2 = max(gm, gM_next)
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
        tt_nuM = min(1/sqtnu - 1/gM0, tt_f)
        out[i,j] += spi.dblquad(func_gmaf, tt_i, tt_nuM, gma_nu, gma_M)[0]

  out *= tnu_arr**(1/3.)
  return out

def DeltaEnup_func(tnu_arr, gm0, gM0, tt_i, gm, gM, tt_f, gm_next, gM_next, p):
  '''
  Delta E'_{\nu'} spectrum in fast cooling case between cell and cell_next
  in units K P'_{e, max} t'_c
  '''
  if len(tnu_arr.shape) == 1:
    tnu_arr = np.array([tnu_arr])
  rows, cols = tnu_arr.shape
  out = np.zeros((rows, cols))
  sqtnu_arr = np.sqrt(tnu_arr)

  
  g1 = min(gm, gM_next)
  g2 = max(gm, gM_next)
  printcount = 0
  for i in range(0, rows):
    for j in range(0, cols):
      sqtnu = sqtnu_arr[i,j]
      if sqtnu <= 1.:
        # tnu already normalized to cyclotron freq
        continue
      elif sqtnu <= gm_next:
        out[i,j] += DeltaE_func(tt_i, tt_f, gm0, gM0, p)
      elif sqtnu <= g1:
        tt_num = 1/sqtnu - 1/gm0
        term1 = DeltaE_func(tt_i, tt_num, gm0, gM0, p)
        tt_hf = .5*(tt_num + tt_f)
        prod = tt_hf*sqtnu
        if prod < 1:
          gma_nu = min(sqtnu/(1-prod),gM0)
        else:
          gma_nu = gM0
        # term2 = DeltaE_func(tt_num, tt_f, gma_nu, gM0, p)
        term2 = DeltaE_integ(tt_num, tt_f, sqtnu, gM, p)
        out[i,j] += term1 + term2
      elif sqtnu <= g2:
        if gm>=gM_next: # fast cooling
          tt_num = 1/sqtnu - 1/gm0
          term1 = DeltaE_func(tt_i, tt_num, gm0, gM0, p)
          tt_nuM = 1/sqtnu - 1/gM0
          # tt_hf = .5*(tt_num + tt_nuM)
          # gma_nu = sqtnu/(1-tt_hf*sqtnu)
          # term2 = DeltaE_func(tt_num, tt_nuM, gma_nu, gM0, p)
          term2 =  DeltaE_integ(tt_num, tt_nuM, sqtnu, gM, p)
          out[i,j] += term1 + term2
        else: # slow cooling
          # tt_hf = .5*(tt_i + tt_f)
          # prod = tt_hf*sqtnu
          # if prod < 1:
          #   gma_nu = min(sqtnu/(1-prod),gM0)
          # else:
          #   gma_nu = gM0
          # term1 = DeltaE_func(tt_i, tt_f, gma_nu, gM0, p)
          term1 = DeltaE_integ(tt_i, tt_f, sqtnu, gM, p)
          out[i,j] += term1
      elif sqtnu <= gM:
        tt_nuM = 1/sqtnu - 1/gM0
        # tt_hf = .5*(tt_i+min(tt_nuM, tt_f))
        # prod = tt_hf*sqtnu
        # if prod < 1:
        #   gma_nu = min(sqtnu/(1-prod),gM0)
        # else:
        #   gma_nu = gM0
        #term1 = DeltaE_func(tt_i, tt_nuM, gma_nu, gM0, p)
        term1 = DeltaE_integ(tt_i, tt_nuM, sqtnu, gM, p)
        out[i,j] += term1

  out *= tnu_arr**(1/3.)
  return out

def DeltaE_integ_cstbnd(t_l, t_u, gm0, gM0, p):
  '''
  Integral form, over gma_i, with constant bounds on both sides
  '''
  def func_gmai(gma, tt):
    return gma**(-p-2/3)*(1+gma*tt)**(2/3)
  def func_gm0(tt):
    return gm0
  def func_gM0(tt):
    return gM0
  int = spi.dblquad(func_gmai, t_l, t_u, func_gm0, func_gM0)[0]
  return int

def DeltaE_integ(t_l, t_u, sqtnu, gM, p):
  '''
  Integral form of E'_{\nu'}\tilde{\nu}^{-1/3} in units K P'_{e,max},
  using the func in gma_f
  '''

  def func_gmanu(tt):
    return sqtnu
  
  def func_gM(tt):
    return gM/(1+gM*tt)
  
  def func_gmaf(gma, tt):
    return gma**(-p-2/3)*(1-gma*tt)**(p-2)


  int = spi.dblquad(func_gmaf, 0, t_u-t_l, func_gmanu, func_gM)[0]
  return int

def DeltaE_integ_i(t_l, t_u, sqtnu, gM0, p):
  '''
  Integral form of E'_{\nu'}\tilde{\nu}^{-1/3} in units K P'_{e,max}
  '''

  def func_gmanu(tt):
    c0 = np.exp(2*tt)*((sqtnu-1)/(sqtnu+1))
    return (1+c0)/(1-c0)
    #return sqtnu/(1-sqtnu*tt)
  
  def func_gM0(tt):
    return gM0
  
  def func_gmai(gma, tt):
    return gma**(-p-2/3)*(1+gma*tt)**(2/3)
  
  int = spi.dblquad(func_gmai, t_l, t_u, func_gmanu, func_gM0)[0]
  return int

def DeltaE_integ_f(t_l, t_u, gm, gM, p):
  def func_gm(tt):
    return gma_cooled(tt, gm)
  
  def func_gM(tt):
    return gma_cooled(tt, gM)
  
  def func_gmaf(gma, tt):
    return gma**(-p-2/3)*(1-gma*tt)**(p-2)

  int = spi.dblquad(func_gmaf, 0, t_u-t_l, func_gm, func_gM)[0]
  return int

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

def get_variable_cells(df_arr, var, env):
  '''
  Returns chosen variable for several cells
  '''
  out = []
  for df in df_arr:
    out.append(get_variable(df, var, env))
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