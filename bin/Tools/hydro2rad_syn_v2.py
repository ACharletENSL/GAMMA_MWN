# -*- coding: utf-8 -*-
# @Author: acharlet

'''
Extracts data from hydro sims from a cell perspective,
produces radiation
'''

from pathlib import Path
import matplotlib.transforms as transforms
from plotting_scripts_new import *
import scipy.integrate as spi
from scipy.special import gamma, betainc, hyp2f1
plt.ion()



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
    if fit and (df.attrs['geometry'] == 'cartesian'):
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

  out = df.copy()
  cr_len = len(df[df.Sd==1])
  imin, imax = int(cr_len*2.5), int(cr_len*3.5)
  varlist = ['rho', 'p', 'D', 'sx', 'tau']
  varlocs = [df.columns.get_loc(var) for var in varlist]
  indices = [i for i in range(imin)]
  refval = out.iloc[[imin, imax]]['vx'].mean()
  out.iloc[indices, df.columns.get_loc('vx')] = refval
  if df.attrs['geometry'] == 'cartesian':
    refvals = out.iloc[[imin, imax]][varlist].mean(axis=0)
  else:
    r_fac = out.iloc[imin:imax]['x'].to_numpy()
    r_fac /= r_fac[0]
    r_fac = r_fac**2
    r_fac = r_fac[:, np.newaxis] 
    refvals = r_fac * out.iloc[imin:imax][varlist].to_numpy()
    refvals = refvals.mean(axis=0)
    out.iloc[indices, varlocs] = refvals
  return out

def get_cellData_rescaled(key, k, logRatio=None, fromSh=True):
  '''
  Returns cell data rescaled accordingly
  if None, returns writeen data and env
  '''

  env0 = MyEnv(key)
  df0 = get_cellData(key, k, fromSh)
  if logRatio is not None:
    scalefac = get_scalefac(logRatio, df0, env0)
    df = rescale_data(df0, scalefac)
    env = MyEnv(key, scalefac)
    return df, env
  else:
    return df0, env0

def get_cellData_withDistrib(key, k, logRatio=None, fromSh=True):
  '''
  Returns dataframe with history of cell, rescaled, and with electrons
  '''
  df, env = get_cellData_rescaled(key, k, logRatio, fromSh)
  df = generate_cellDistrib(df, env)
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
    
  ax.plot(x, y, **kwargs)

  if logx:
    ax.set_xscale('log')
  if logy:
    ax.set_yscale('log')
  ax.set_xlabel(xlabel)
  ax.set_ylabel(var_exp[var])


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

def testplot_step(j, k=450, key='test', logRatio=None):
  df, env = get_cellData_withDistrib(key, k, logRatio)
  nuobs, Tobs = radEnv(env, largeFreqs=True)
  p=env.psyn
  tnu = np.logspace(0, 20, 1000)
  cell_0, cell, cell_next = df.iloc[0], df.iloc[j], df.iloc[j+1]
  gm0, gM0 = cell_0[['gmin', 'gmax']]
  #cell = cell_0
  lfac = get_variable(cell_0, 'lfac', env)
  tc = 1/get_variable(cell_0, 'syn', env)
  tt_i = (cell['t'] - cell_0['t'])/(tc*lfac)
  tt_f = (cell_next['t'] - cell_0['t'])/(tc*lfac)
  gm, gM, gm_nxt, gM_nxt = *cell[['gmin', 'gmax']], *cell_next[['gmin', 'gmax']]
  x_B = get_variable(cell_0, 'nu_B', env)/env.nu0
  x_arr = np.array([gm**2, gM**2, gm_nxt**2, gM_nxt**2])
  x_names = ['m,j', 'M,j', 'm,j+1', 'M,j+1']
  Epnu1 = tnu*DeltaEnup_v1(tnu, gm0, gM0, tt_i, gm, gM, tt_f, gm_nxt, gM_nxt, p)
  Epnu2 = tnu*DeltaEnup_v2(tnu, gm0, gM0, tt_i, gm, gM, tt_f, gm_nxt, gM_nxt, p)
  y1 = Epnu1.flatten()
  y2 = Epnu2.flatten()

  fig, ax = plt.subplots()
  ax.loglog(tnu, y1, label='func')
  ax.loglog(tnu, y2, ls='--', label='func and $\\int$')
  ax.set_xlabel("$x=\\nu'/\\nu'_{\\rm B}$")
  ax.set_ylabel("$\\Delta E'_{\\nu'}$")
  transy = transforms.blended_transform_factory(
    ax.transData, ax.transAxes)
  for val, name in zip(x_arr, x_names):
    ax.axvline(val, ls=':', c='k')
    ax.text(val, 1.02, '$x_{\\rm '+name+'}$', transform=transy)
  ax.legend()

def plot_stepEmission(j, var='sp', logv=0, k=450, key='test', logRatio=None,
  largeFreqs=True, withsteps=True, theory=True, spec='int',
  ax_in=None, **kwargs):
  '''
  Plots contribution of step j to lightcurve ('lc') or spectrum ('sp')
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

  cell_0 = df.iloc[0]
  cell = df.iloc[j]
  cell_next = df.iloc[j+1]
  Lnu, reg = get_Lnu_cooling(cell, cell_next, cell_0, nuobs, Tobs, env)
  nL = tnu * Lnu/env.L0
  y = nL.flatten()

  line, = ax.plot(x, y, zorder=-1, **kwargs)
  if 'c' in kwargs:
    col = kwargs['c']
  else:
    col = 'r' if k < env.Next+env.Nsh4 else 'b'
    line.set_color(col)
  
  Ton, Tth, Tej = get_variable(cell_0, 'obsT', env)
  tT = (Tobs-Tej)/Tth
  gm, gM, gm_nxt, gM_nxt = *cell[['gmin', 'gmax']], *cell_next[['gmin', 'gmax']]
  x_B = get_variable(cell_0, 'nu_B', env)/env.nu0
  x_arr = tT*x_B*np.array([gm**2, gM**2, gm_nxt**2, gM_nxt**2])
  x_names = ['m1', 'M1', 'm2', 'M2']
  transy = transforms.blended_transform_factory(
      ax.transData, ax.transAxes)
  for val, name in zip(x_arr, x_names):
    ax.axvline(val, ls=':', c='k')
    ax.text(val, 1.02, '$x_{\\rm '+name+'}$', transform=transy)
  
  if not ax_in:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

  if logx: ax.set_xscale('log')
  if logy: ax.set_yscale('log')
  plt.tight_layout()

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
    
    Ton, Tth, Tej = get_variable(cell0, 'obsT', env)
    tT = (Tobs-Tej)/Tth
    
    x_B = get_variable(cell0, 'nu_B', env)/env.nu0
    # x_m = get_variable(cell0, 'nu_m', env)/env.nu0
    # x_M = get_variable(cell0, 'nu_M', env)/env.nu0
    x_m = x_B * (cell0.gmin**2)
    x_M = x_B * (cell0.gmax**2)
    tc = 1/get_variable(cell0, 'syn', env)
    tdyn = get_tdyn(df, env)
    #print(tdyn1, tdyn2)
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

def plot_gmas(key, k, logRatio=None):
  '''
  Plots (gmin, gmax) for a given logRatio
  '''

  df, env = get_cellData_withDistrib(key, k, logRatio)
  fig, ax = plt.subplots()
  plot_any('gmin', df, logy=True, ax_in=ax)
  plot_any('gmax', df, logy=True, ax_in=ax)



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
  f'(y) = - (sigT B**2 / (6 pi me c)) * y**2 + (d rho/d t) / (3 rho) * y
  '''

  # technically y**2 - 1
  return adiab*y - alpha_*B2*y**2 

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

def generate_cellDistrib(df, env):
  '''
  Generate and evolves the electron distribution, returning dataframe with gmas
  df needs to start at shock! (fromSh = True in get_cellData)
  '''
  out = df.copy()
  its = out.index.to_list()

  out['dt'] = np.gradient(df.t.to_numpy())
  out['drho'] = np.gradient(df.rho.to_numpy())
  out[['gmin', 'gmax']] =  0., 0.
  gmin, gmax = get_varlist(out.iloc[0], ['gma_m', 'gma_M'], env)
  out.loc[its[0], ['gmin', 'gmax']] = np.array([gmin, gmax])
  for j, it_next in enumerate(its[1:]):
    cell = out.iloc[j]
    cell_new = out.iloc[j+1]
    gmas_new = evolve_gmas(cell, cell_new, env)
    gmas_new = [max(gma, 1) for gma in gmas_new]
    out.loc[it_next, ['gmin', 'gmax']] = gmas_new
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
    if spec == 'int':
      Lnu, reg = get_Lnu_cooling(cell, cell_next, cell_0, nuobs, Tobs, env)
    elif spec == 'BPL':
      Lnu, reg = get_Lnu_integBPL(cell, cell_next, cell_0, nuobs, Tobs, env)
    nL = tnu * Lnu/env.L0
    out += nL
    nL_arr.append(nL)
    reg_arr.append(reg)

  if withsteps:
    return out, np.array(nL_arr), reg_arr
  else:
    return out

# def get_Lnu_integBPL(cell, cell_next, cell_0, nuobs, Tobs, env):
#   '''
#   Derive the radiation emitted between states cell and cell_next
#     subdivizing the time step into cooling steps to get 'locally slow cooling'
#     electrons, giving 'instantaneous' spectrum, then performing simple integral
#     nout controls number of outputs of gammas during a time step
#   '''


#   if (type(Tobs)==np.ndarray) and (type(nuobs)==np.ndarray):
#     if (len(Tobs) > 1) and (len(nuobs) > 1):
#       nuobs = nuobs[np.newaxis, :]
#       Tobs  = Tobs[:, np.newaxis]
#       out = np.zeros((len(Tobs), len(nuobs)))
#   if (type(nuobs)==float) or (len(nuobs)==1):
#     out = np.zeros(Tobs.shape)
#   elif (type(Tobs)==float) or (len(Tobs)==1):
#     out = np.zeros(nuobs.shape)

#   gmin, gmax, gmin_nxt, gmax_nxt = *cell[['gmin', 'gmax']], *cell_next[['gmin', 'gmax']]
#   reg = ''
#   if not gmin:
#     return out, reg
  
#   # if local cooling too fast, apply method:
#   tc = 1/get_variable(cell, 'syn', env)
#   x = nuobs/env.nu0
#   Ton, Tth, Tej = get_variable(cell, 'obsT', env)
#   tT = ((Tobs - Tej)/Tth)
#   p = env.psyn
#   tc = 1/get_variable(cell, 'syn', env)
#   Pe = get_variable(cell, 'Pmax', env)
#   V3 = get_variable(cell, 'V3p', env)
#   nup_B = get_variable(cell, 'nup_B', env)
#   x_B = nup_B/env.nu0p

#   width = gmax/gmin
#   if width <= 2:
#     reg = 'monoE'
#     # take geometric mean as the value where all the energy lies
#     gm, gm_nxt = np.sqrt(gmin*gmax), np.sqrt(gmin_nxt*gmax_nxt)
#     ne = get_variable(cell, "n", env)*env.xi_e
#     dgma = gm - gm_nxt
#     nupm = nup_B * gm_nxt**2
#     Epnum = 0.8 * V3 * ne * dgma * me_*c_**2 / nupm
#     x_m = (gm_nxt/env.gma_m)**2
#     x_M = (gm/env.gma_m)**2
#     Epnu = spectrum_bplaw_mono(x*tT, x_B, x_m, x_M)
#     Epnu *= Epnum
#   else:   
#     Nout = 10*(int(np.log10(gmax/gmax_nxt)) + 1)
#     flux = np.zeros((Nout-1, *out.shape))
#     sol_gmin, sol_gmax = evolve_gmas(cell, cell_next, env, Nout=Nout)
#     t_arr, gmin_arr, gmax_arr = sol_gmax.t, sol_gmin.y[0], sol_gmax.y[0]
#     t_arr = .5*(t_arr[1:]+t_arr[:-1])
#     reg = 'fast' if (gmax_arr[-1] < gmin_arr[0]) else 'slow'
#     Epnum = ((p-1)/(3*p-1))*Pe*V3
#     # Epnum = get_variable(cell, "EpnuSC", env)
#     # Epnum = get_variable(cell, "Ep_th", env)
#     for i in range(Nout-1):
#       gm, gM, gc = gmin_arr[i], gmax_arr[i], gmax_arr[i+1]
#       x_m = (gm/env.gma_m)**2
#       x_M = (gM/env.gma_m)**2
#       x_c = (gc/env.gma_m)**2
#       spec = spectrum_bplaw(x*tT, x_B, x_c, x_m, x_M, env.psyn)
#       flux[i] += spec
#     Epnu = np.trapezoid(flux, t_arr, axis=0)
#     Epnu *= Epnum

#   out += ((1+env.z)/Tth) * Epnu * tT**-2
#   if (len(nuobs) == 1) or (len(Tobs)==1):
#     out = out.flatten()
#   return out, reg

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
  reg = ''
  if not gm:
    return out, reg

  Ton, Tth, Tej = get_variable(cell, 'obsT', env)
  tT = ((Tobs - Tej)/Tth)
  lfac = get_variable(cell, 'lfac', env)
  tc = 1/get_variable(cell, 'syn', env)
  tt_i = (cell['t'] - cell_0['t'])/(tc*lfac)
  tt_f = (cell_next['t'] - cell_0['t'])/(tc*lfac)

  #cell = cell_0
  nup_B = get_variable(cell, "nup_B", env)
  Dop = get_variable(cell, 'Dop', env)
  p = env.psyn
  nup = nuobs/Dop
  Pe = get_variable(cell, 'Pmax', env)
  V3 = get_variable(cell, 'V3p', env)


  # width = gmax/gmin
  # if width <= 2:
  #   reg = 'monoE'
  #   # take geometric mean as the value where all the energy lies
  #   gm, gm_nxt = np.sqrt(gmin*gmax), np.sqrt(gmin_nxt*gmax_nxt)
  #   nup_m = nup_B * gm**2
  #   x_B = gm_nxt**-2
  #   x_m = (gm_nxt/gm)**2
  #   Epnu_m = (3/5.) * Pe * tc * V3 / gm
  #   # normalized with nup_m
  #   def spec(tnu):
  #     return mixed_plaw(tnu, x_B, x_m)
  #   tnu = nup/nup_m
  #   Epnu = Epnu_m * spec(tT*tnu)
  # else:   # 'exact' cooling by integrating evolution of gmas min/max

  reg = 'fast' if gm >= gM_nxt else 'slow'
  tnu = nup/nup_B
  #K = (p-1)/(gm**(1-p)-gM**(1-p))
  K = (p-1)/(gm0**(1-p)-gM0**(1-p))
  Epnu_m = K*Pe*tc*V3 # Pe already contains ne
  Epnu = DeltaEnup(tT*tnu, gm0, gM0, tt_i, gm, gM, tt_f, gm_nxt, gM_nxt, p)
  Epnu *= Epnu_m

  out += ((1+env.z)/Tth) * Epnu * tT**-2
  if (len(nuobs) == 1) or (len(Tobs)==1):
    out = out.flatten()
  return out, reg

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

def plot_instant_dndgma(tt, gm0, gM0, p=2.5):
  '''
  dn/dgma_f at normalized time tt
  '''

  fig, ax = plt.subplots()
  gm, gM = gm0/(1+gm0*tt), gM0/(1+gM0*tt)
  x = np.logspace(int(np.log10(gm)), int(np.log10(gM0))+1, 200)
  y0, y = np.zeros((2, len(x)))

  seg1 = (x>=gm0) & (x<=gM0)
  y0[seg1] = (x[seg1]/gm0)**(-p)
  seg2 = (x>=gm) & (x<=gM)
  y[seg2] = (x[seg2]/gm)**(-p) * (1-(x[seg2]/gm)*tt)**(p-2)

  ax.loglog(x, y0, label='$\\tilde{t}=0$')
  ax.loglog(x, y, label='$\\tilde{t}='+f'{tt:.1e}$')
  ax.axvline(tt**-1, ls=':', c='k')
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

def DeltaEnup(tnu_arr, gm0, gM0, tt_i, gm, gM, tt_f, gm_next, gM_next, p):
  '''
  Delta E'_{\nu'} spectrum between cell and cell_next in units K P'_{e, max} t'_c
  mix and match between integral forms and integrated
  '''

  rows, cols = tnu_arr.shape
  out = np.zeros((rows, cols))
  sqtnu_arr = np.sqrt(tnu_arr)
  # p = env.psyn
  # sq_nuB = np.sqrt(get_variable(cell, 'nup_B', env))

  # gm0, gM0 = cell_0[['gmin', 'gmax']]
  # gm, gM = cell[['gmin', 'gmax']]
  # gm_next, gM_next = cell_next[['gmin', 'gmax']]

  # lfac = get_variable(cell, 'lfac', env)
  # tc = get_variable(cell, 'syn', env)**-1
  # tt_i = (cell['t'] - cell_0['t'])/(tc*lfac)
  # tt_f = (cell_next['t'] - cell_0['t'])/(tc*lfac)

  dgm = (gm_next**(-p-2/3)-gm**(-p-2/3))
  dgM = (gM_next**(-p-2/3)-gM**(-p-2/3))
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
        out[i,j] += (3/(3*p+2))*(dgm-dgM)
      elif sqtnu <= g1:
        tt_num = 1/sqtnu - 1/gm0
        dgm_nu = (sqtnu**(-p-2/3)-gm**(-p-2/3))
        out +=  (tt_f-tt_i)*sqtnu**(-p/2+1/3) + dgm_nu-dgM
      elif sqtnu <= g2:
        if gm>=gM_next: # ''fast'' cooling
          tt_num = 1/sqtnu - 1/gm0
          tt_nuM = 1/sqtnu - 1/gM0

def DeltaEnup_v2(tnu_arr, gm0, gM0, tt_i, gm, gM, tt_f, gm_next, gM_next, p):
  '''
  Delta E'_{\nu'} spectrum between cell and cell_next in units K P'_{e, max} t'_c
  mix and match between integral forms and integrated
  '''
  if len(tnu_arr.shape) == 1:
    tnu_arr = np.array([tnu_arr])
  rows, cols = tnu_arr.shape
  out = np.zeros((rows, cols))
  sqtnu_arr = np.sqrt(tnu_arr)
  # p = env.psyn
  # sq_nuB = np.sqrt(get_variable(cell, 'nup_B', env))

  # gm0, gM0 = cell_0[['gmin', 'gmax']]
  # gm, gM = cell[['gmin', 'gmax']]
  # gm_next, gM_next = cell_next[['gmin', 'gmax']]

  # lfac = get_variable(cell, 'lfac', env)
  # tc = get_variable(cell, 'syn', env)**-1
  # tt_i = (cell['t'] - cell_0['t'])/(tc*lfac)
  # tt_f = (cell_next['t'] - cell_0['t'])/(tc*lfac)
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
      elif sqtnu < g2:
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

def DeltaEnup_v1(tnu_arr, gm0, gM0, tt_i, gm, gM, tt_f, gm_next, gM_next, p):
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
        #out[i,j] += DeltaE_integ_gmai_csts(tt_i, tt_f, gm0, gM0, p)
      elif sqtnu <= g1:
        tt_num = 1/sqtnu - 1/gm0
        tt_hf = .5*(tt_num+tt_f)
        #tt_hf = min(tt_num, tt_f)
        gma_nu = sqtnu/(1-tt_hf*sqtnu)
        out[i,j] += DeltaE_func(tt_i, tt_num, gm0, gM0, p) + DeltaE_func(tt_num, tt_f, gma_nu, gM0, p)
      elif sqtnu <= g2:
        if gm>=gM_next: # fast cooling
          tt_num = 1/sqtnu - 1/gm0
          tt_nuM = 1/sqtnu - 1/gM0
          tt_hf = .5*(tt_num+tt_nuM)
          gma_nu = sqtnu/(1-tt_hf*sqtnu)
          out[i,j] += DeltaE_func(tt_i, tt_num, gm0, gM0, p) + DeltaE_func(tt_num, tt_nuM, gma_nu, gM0, p)
        else: # slow cooling
          tt_hf = .5*(tt_i+tt_f)
          gma_nu = sqtnu/(1-tt_hf*sqtnu)
          out[i,j] += DeltaE_func(tt_i, tt_f, gma_nu, gM0, p)
      elif sqtnu <= gM:
        tt_nuM = 1/sqtnu - 1/gM0
        tt_hf = .5*(tt_i+min(tt_f, tt_nuM))
        gma_nu = sqtnu/(1-tt_hf*sqtnu)
        out[i,j] += DeltaE_func(tt_i, tt_nuM, gma_nu, gM0, p)

  out *= tnu_arr**(1/3.)
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

  # if t<0:
  #   t = -t
  gt = gma*t
  gtm = -1/gt
  term1 = 2*(1+gt)*hyp2f1(1/3,p-1,p,gtm)
  term2 = (3*p-1+(3*p+4)*gt)*hyp2f1(-2/3,p-1,p,gtm)
  diff = gt**(2/3.) * (term1-term2)
  return gma**(-p-2/3.) * (3*(p-1) + diff)


######## overcharging functions because it's easier than to rewrite data_IO
def get_varlist(df, varlist, env):
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