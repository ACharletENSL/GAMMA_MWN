# -*- coding: utf-8 -*-
# @Author: acharlet

'''
Adding electron distribution to data for cooling effects
Contains:
  - generate_cellDistrib
  - gamma_synCooled
  - norm_plaw_distrib
  - distrib_plaw_cooled
  - evolve_gmas_analytic
  - coolingFunc_ODE
  - evolve_gmas_num
  - split_hydrostep
'''

import numpy as np
import scipy.integrate as spi
from IO import *
from phys_constants import *
from fits_hydro import extrapolate_early


####### With analytical expression for synchrotron cooling
def gamma_synCooled(tt, gma0):
  '''
  gamma(t) from solving cooling equation (only synchrotron, no adiabatic)
  '''
  gma = gma0/(1+gma0*tt)
  return gma

def evolve_gmas_analytic(cell, cell_new, env,
  func_cooling=gamma_synCooled):
  '''
  Evolve a distrib (of Ngma bins) using the analytical expression
  '''

  gmas_names = [name for name in cell.keys() if ('gm' in name)]
  gmas_next = np.array([func_cooling(cell.dtt, cell[name]) for name in gmas_names])
  return gmas_next

# other relevant functions
def norm_plaw_distrib(gmin, gmax, p):
  '''
  Normalization ('K') of a power-law distribution of electrons
  with index p between gmin and gmax
  '''
  K = (p-1)/(gmin**(1-p)-gmax**(1-p))
  return K
  
def distrib_plaw_cooled(gma, p, tt):
  '''
  Power law distribution after time tt = t'/t_c1
  normalization K = (p-1)/(gmin0**(1-p)-gmax0**(1-p))
  '''

  return gma**(-p)*(1-gma*tt)**(p-2)

####### By solving the ODE for cooling
def coolingFunc_ODE(t, y, adiab, B2):
  '''
  Differential equation to solve for cooling
  Optically thin synchrotron cooling + adiabatic
  f'(y) = - (sigT B**2 / (6 pi me c)) * (y**2-1) + (d rho/d t) / (3 rho) * y
  '''

  # technically y**2 - 1
  return adiab*y - alpha_*B2*(y**2 -1)

def stop_condition(t, y, adiab, B2):
  return y[0]-1.


def evolve_gmas_num(cell, cell_new, env, ODE=coolingFunc_ODE):
  '''
  Evolve a distrib of Ngma bins by solving the ODE
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
    sol = spi.solve_ivp(ODE, t_span, gmas, method='RK23', args=(adiab, B2))
    gmas_new = sol.y[:,-1]
  
  return gmas_new





#### functions from v1 that will be reused
def get_scalefac(logRatio, data, env):
  '''
  Returns (log) scale factor of radius to obtain desired log(gma_c/gma_m)
  '''
  current = check_coolRegime_cell(data, env)
  scalefac = logRatio - current
  return scalefac

def check_coolRegime_cell(cell, env):
  '''
  log10(gma_c/gma_m)
  cell is data starting at shock
  '''
  cell0 = cell.iloc[0]
  gma_m = get_variable(cell0, 'gma_m', env)
  tc1  = get_variable(cell0, 'tc1', env)
  tbl = get_tdbl_cell(cell, env)
  tcell=get_tdyn_cell(cell)
  tdyn = max(tbl, tcell)
  gma_c = tc1/tdyn
  return np.log10(gma_c/gma_m)

def get_tdyn_cell(cell):
  '''
  Comoving dynamical time seen by a given fluid element
  cell is data starting at shock 
  '''
  tp = cell.tp.to_numpy()
  tdyn = tp[-1] - tp[0] 
  return tdyn

def get_tdbl_global(env):
  '''
  Comoving radius doubling time for shocked material at collision
  '''
  tdb = env.R0/(env.lfac0*c_)
  return tdb

def get_tdbl_cell(cell, env):
  '''
  Comoving radius doubling time for a given fluid element
  '''
  cell0 = cell.iloc[0]
  lfac0 = get_variable(cell0, 'lfac', env)
  tdb = cell0.x / lfac0
  return tdb

def get_tcooled_cell(cell, env):
  '''
  Time 
  '''

def get_tdyn(cell, env):
  '''
  '''
  cell0 = cell.iloc[0]
  z = 4 if (cell0.trac < 1.5) else 1
  tstart = cell0.t
  tend = cell.iloc[-1].t
  lfac0 = get_variable(cell0, 'lfac', env)
  tdyn1 = (tend-tstart)/lfac0     # comoving time range
  tdyn2 = cell0.x / lfac0         # comoving doubling time
  return min(tdyn1, tdyn2)

##### Time binning for cooling 
def time_binning_cooling(cell, env, end_val, end_cond='tt',
  r_ref=1.2, Nmin=2, Nmax=None, func_cooling=gamma_synCooled):
  '''
  Generate a time array from logarithmic time binning in tilde_t according to cooling time
    of the initial state 'cell' gamma_max, with a controled ratio r_ref of gma_max between steps
  '''

  tcp = 1/get_variable(cell, 'syn', env)
  lfac = get_variable(cell, 'lfac', env)
  t0, tp0, tt0 = cell.t, 0., 0.
  try:
    # when splitting a hydro step in cooling steps, gmax already exists
    gmax0 = cell.gmax
    tt0 = cell.tt
    tp0 = cell.tp
  except AttributeError:
    # when splitting a complete cell history according to cooling
    gmax0 = get_variable(cell, "gma_M", env)
  if end_cond == 'tt':
    gmax_end = func_cooling(tt_max, gmax0)
  elif end_cond == 'gmax':
    gmax_end = end_val
  else:
    print("'end_cond' must be 'tt' or 'gmax'")
    return 0.
  ratio = gmax0/gmax_end
  N = int(np.log10(ratio)/np.log10(r_ref))
  N = max(N, Nmin)
  if Nmax is not None:
    N = min(N, Nmax)
  
  # separate bins of expected gmas then bin time steps accordingly
  gmax_bins = np.geomspace(gmax0, gmax_end, N+1, endpoint=True)
  tt_arr = gmax_bins**-1 - gmax0**-1  

  dtt_arr = np.diff(tt_arr)
  tt_arr = tt_arr[:-1]
  tt_arr += tt0
  dtp_arr = dtt_arr * tcp
  dt_arr = dtp_arr * lfac
  tp_arr = np.insert(np.cumsum(dtp_arr[:-1]), 0, 0.) + tp0
  t_arr = np.insert(np.cumsum(dt_arr[:-1]), 0, 0.) + t0

  return t_arr, dt_arr, tp_arr, dtp_arr, tt_arr, dtt_arr


# Creating and evolving the electron distribution
def generate_cellDistrib(data, scalefac=0, Ngma=0,
    adiab=False, extrapolate=True, analytic=False):
  '''
  Generate and evolves the electron distribution, returning dataframe with gmas
  Boolean 'adiab' adds adiabatic term
  Divides the initial distribution into Ngma bins
    default: 0 (binning only for ease of calculation, not for physical reasons)
  extrapolate boolean controls the extrapolation of early data
  '''

  # select data starting from the shock
  out = data.loc[(data.Sd == 1.).idxmax():].copy()
  its = out.index.to_list()
  Nj = len(out)

  env = MyEnv(data.attrs['key'], scalefac)
  sc = 10**scalefac
  out['t'] = sc * out['t']
  out['x'] = sc * out['x']

  if analytic:    # completely replace the data with analytical values
    extrapolate = False
    rho0 = env.rho3 if (data.iloc[0].trac < 1.5) else env.rho2
    rho0 /= env.rhoscale
    v0, p0 = env.beta, env.p_sh
    p0 /= env.rhoscale*c_*c_

    if data.attrs['geometry'] == 'cartesian':
      out['rho'] = rho0
      out['vx']  = v0
      out['p']   = p0

    # Create uniform time array for analytic case to avoid discretization artifacts
    # Use same start/end but with uniform spacing
    t_start, t_end = out.t.iloc[0], out.t.iloc[-1]
    out['t'] = np.linspace(t_start, t_end, len(out))

    # Also make x uniform if cartesian (constant velocity)
    if data.attrs['geometry'] == 'cartesian':
      x_start, x_end = out.x.iloc[0], out.x.iloc[-1]
      out['x'] = np.linspace(x_start, x_end, len(out))

    # need to implement proper handling of spherical case


  if extrapolate:
    out = extrapolate_early(out, env, it_s=15, dfindex=False)

  # Compute time intervals and proper time
  lfac = get_variable(out, 'lfac', env)
  t_arr = out.t.to_numpy()
  dt = np.gradient(t_arr)
  dtp = dt/lfac

  # For extrapolated region, smooth dt to remove irregular timestep artifacts
  # The simulation may have irregular output cadence that creates artificial
  # features in the integrated spectrum
  if extrapolate:
    it_s = 15
    # Smooth dt over the entire extrapolated region, not just the boundary
    # This removes discretization artifacts from irregular simulation outputs
    dt_smoothed = dt.copy()
    for i in range(it_s):
      # 3-point moving average
      i_start = max(0, i - 1)
      i_end = min(len(dt), i + 2)
      dt_smoothed[i] = np.mean(dt[i_start:i_end])
    dt = dt_smoothed
    dtp = dt/lfac

  tp = np.cumsum(dtp)
  tc = 1./get_variable(out, "syn", env)
  dtt = dtp/tc
  tt = np.cumsum(dtt) - dtt[0]

  out['dt'] = dt
  out['tp'] = tp
  out['dtp'] = dtp
  out['tt'] = tt
  out['dtt'] = dtt

  if adiab:
    rho = out.rho.to_numpy(copy=True)
    adiab = (rho/rho[0])**(1/3)
  else:
    adiab = np.ones(Nj)

  gmin, gmax = get_vars(out.iloc[0], ['gma_m', 'gma_M'], env)
  if Ngma > 0:
    varlist = ['gmin'] + [f'gma_e{i}' for i in range(1,Ngma-1)] + ['gmax']
  else:
    varlist = ['gmin', 'gmax']
  Nv = len(varlist)
  out[varlist] =  [0. for i in range(Nv)]
  out.loc[its[0], varlist] = np.logspace(np.log10(gmin), np.log10(gmax), Nv, endpoint=True)
  
  for j, it_next in enumerate(its[1:]):
    if (j%500 == 0.):
      print(f'Evolving e- distribution, it {it_next}')
    cell, cell_new = out.iloc[j], out.iloc[j+1]
    gmas_new = evolve_gmas_analytic(cell, cell_new, env)
    # took the max(gma, 1) out, we want to track # e- that cooled to NR
    gmas_new = np.array([(adiab[j] * gma) for gma in gmas_new])
    out.loc[it_next, varlist] = gmas_new
  return out, env


def generate_cellDistrib_interpolated(data, env,
  scalefac=0, adiab=False, r_ref=1.2):
  '''
  Generate a dataframe with interpolated hydrodynamics
    then adds the electron distribution
  '''

  # select data starting from the shock
  out = data.loc[(data.Sd == 1.).idxmax():].copy()
  its = out.index.to_list()
  Nj = len(out)
  cell0 = out.iloc[0]

  # rescale x and t
  env = MyEnv(data.attrs['key'], scalefac)
  sc = 10**scalefac
  out['t'] = sc * out['t']
  out['x'] = sc * out['x']

  # normalized time
  delta_t = out.t.iloc[-1] - out.t.iloc[0]
  lfac = get_variable(cell0, 'lfac', env)
  tc1 = get_variable(cell0, 'tc1', env)
  delta_tt = delta_t/(lfac*tc1)

  # number of time bins depending on the accepted 
  #   ratio of gma_max at each step, r_ref
  N = int(np.log(cell0.gmax)/np.log(r_ref))
  tt_arr = np.geomspace(1, 1+delta_tt, N+1, endpoint=True) - 1.
  dtt_arr = np.diff(tt_arr)
  tt_arr = tt_arr[:-1]
  dtp_arr = dtt_arr * tc1
  dt_arr = dtp_arr * lfac
  tp_arr = np.insert(np.cumsum(dtp_arr[:-1]), 0, 0.)
  t_arr = np.insert(np.cumsum(dt_arr[:-1]), 0, 0.)

  # interpolate hydro
  #


def split_hydrostep_old(cell, cell_next, env, r_ref=1.2, Nmin=2, Nmax=20,
  func_cooling=gamma_synCooled):
  '''
  Splits a hydro time step according to cooling
    r_ref: reference ratio of gma_max between two time bins
  '''

  tcp = 1/get_variable(cell, 'syn', env)
  lfac = get_variable(cell, 'lfac', env)
  ratio = cell.gmax/cell_next.gmax
  N = int(np.log10(ratio)/np.log10(r_ref))
  N = max(N, Nmin)
  N = min(N, Nmax)

  # separate bins of expected gmas then bin time steps accordingly
  gmax_bins = np.geomspace(cell.gmax, cell_next.gmax, N+1, endpoint=True)
  tt_arr = gmax_bins**-1 - cell.gmax**-1  

  dtt_arr = np.diff(tt_arr)
  tt_arr = tt_arr[:-1]

  gma_keys = [key for key in cell.keys() if 'gm' in key]
  gmas = cell[gma_keys].to_numpy()
  gmas_next = func_cooling(tt_arr[np.newaxis, :], gmas[:, np.newaxis])

  tt_arr += cell.tt
  dtp_arr = dtt_arr * tcp
  dt_arr = dtp_arr * lfac
  tp_arr = np.insert(np.cumsum(dtp_arr[:-1]), 0, 0.) + cell['tp']
  t_arr = np.insert(np.cumsum(dt_arr[:-1]), 0, 0.) + cell.t

  # no interpolate hydro (for now?)
  dic = {key:np.ones(N)*cell[key] for key in cell.keys()}
  t_vars = ['dtt', 'dtp', 'dt', 'tt', 'tp']
  t_vals = [dtt_arr, dtp_arr, dt_arr, tt_arr, tp_arr]
  dic_t = {k:v for k, v in zip(t_vars, t_vals)}
  dic.update(dic_t)
  dic_g = {k:v for k, v in zip(gma_keys, gmas_next)}
  dic.update(dic_g)
  out = pd.DataFrame.from_dict(dic)
  return out

def split_hydrostep(cell, env, r_ref=1.2, Nmin=2, Nmax=20,
  func_cooling=gamma_synCooled):
  '''
  Splits a hydro time step according to cooling
    r_ref: reference ratio of gma_max between two time bins
  
  '''

  tt_max = cell.dtt
  time_arrays = time_binning_cooling(cell, tt_max, env, r_ref, Nmin, Nmax, func_cooling)
  t_arr, dt_arr, tp_arr, dtp_arr, tt_arr, dtt_arr = time_arrays
  N = len(t_arr)

  gma_keys = [key for key in cell.keys() if 'gm' in key]
  gmas = cell[gma_keys].to_numpy()
  gmas_next = func_cooling(tt_arr[np.newaxis, :] - cell.tt, gmas[:, np.newaxis])

  # no interpolate hydro (for now?)
  dic = {key:np.ones(N)*cell[key] for key in cell.keys()}
  t_vars = ['dtt', 'dtp', 'dt', 'tt', 'tp']
  t_vals = [dtt_arr, dtp_arr, dt_arr, tt_arr, tp_arr]
  dic_t = {k:v for k, v in zip(t_vars, t_vals)}
  dic.update(dic_t)
  dic_g = {k:v for k, v in zip(gma_keys, gmas_next)}
  dic.update(dic_g)
  out = pd.DataFrame.from_dict(dic)
  return out
