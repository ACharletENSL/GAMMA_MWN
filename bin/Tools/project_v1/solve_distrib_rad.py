# -*- coding: utf-8 -*-
# @Author: acharlet

'''
Cool electrons over the course of one hydro time step (t_j to t_j+1),
producing radiation on the go
General algorithm:
- open data
- generate the normalized time array (rescale here if desired)
- generate the e- distribution (using the defined cooling function)
- split hydro timestep logarithmically in cooling time steps (dt'_0 ~ t'_c/2)
- interpolate hydro states across the time 
    (can be taken constant as first approx but r, t cannot)
- we now have an array of 'time steps' over which we can apply our usual method
- produce radiation
'''

# contains:
# some test plot functions to demonstrate the steps
# splitting hydro timestep: hydro_step_splitting
# notes:
#   ask how to get t' in an accelerating flow
#   ask for marg. fast & slow cooling

from hydro2rad_syn import *


def Tobs_to_tildeT(Tobs, cell, env):
  '''
  Normalizes observed time
  (Why did I never write this before?)
  '''
  Ton, Tth, Tej = get_variable(cell, 'obsT', env)
  tT = ((Tobs - Tej)/Tth)
  return tT

def nuobs_to_nucomov_normalized(nuobs, cell, env, norm='syn'):
  '''
  From observer to comoving frame
  norm = syn normalizes to cyclotron freq, other to nu'_0 in env
  '''

  D = get_variable(cell, 'Dop', env)
  nup = nuobs/D
  if norm == 'syn':
    nuB = get_variable(cell, 'nup_B', env)
    nu = nup/nuB
  else:
    nu = nup/env.nu0p
  return nu

def step_bplaw_theoric_norm(df_step, nup, tT, env):
  cell_0 = df_step.iloc[0]
  cell_f = df_step.iloc[-1]
  nup_B = get_variable(cell_0, 'nup_B', env)
  nup_m  = nup_B*cell_0.gmin**2
  x_B = nup_B/nup_m
  x_M = x_B*cell_0.gmax**2
  x_c = x_B*cell_f.gmax**2
  x = nup/nup_m
  y = tT**-2 * spectrum_bplaw(tT*x, x_B, x_c, 1, x_M, env.psyn)
  return y


def step_bplaw_theoric(df_step, nup, tT, env):
  cell_0 = df_step.iloc[0]
  cell_f = df_step.iloc[-1]
  nup_B = get_variable(cell_0, 'nup_B', env)
  nup_m  = nup_B*cell_0.gmin**2
  x_B = nup_B/nup_m
  x_M = x_B*cell_0.gmax**2
  x_c = x_B*cell_f.gmax**2
  Lth = get_variable(cell_0, 'Lth', env)
  x = nup/nup_m
  y = Lth * tT**-2 * spectrum_bplaw(tT*x, x_B, x_c, 1, x_M, env.psyn)
  return y

def gamma_synCooled(tt, gma0):
  '''
  gamma(t) from solving cooling equation (only synchrotron, no adiabatic)
  '''
  gma = gma0/(1+gma0*tt)
  return gma


def gamma_synCooled_exp(tt, gma0):
  c0 = (gma0-1)/(gma0+1)
  return (np.exp(2*tt)+c0)/(np.exp(2*tt)-c0)

def norm_plaw_distrib(gmin, gmax, p):
  '''
  Normalization ('K') of a power-law distribution of electrons
  with index p between gmin and gmax
  '''
  K = (p-1)/(gmin**(1-p)-gmax**(1-p))
  return K

def distrib_plaw_init(gma, p):
  '''
  Initial power law distribution
  normalization K = (p-1)/(gmin**(1-p)-gmax**(1-p))
  '''

  return gma**(-p)

def distrib_plaw_cooled(gma, p, tt):
  '''
  Power law distribution after time tt = t'/t_c1
  normalization K = (p-1)/(gmin0**(1-p)-gmax0**(1-p))
  '''

  return gma**(-p)*(1-gma*tt)**(p-2)

def syn_emiss_simple(gma, tnu):
  '''
  Simplest synchrotron emission function:
  power law in nu' until nu'_B*gma**2 then 0
  '''
  if tnu <= gma**2:
    return tnu**(1./3.)*gma**(-2./3.)
  else:
    return 0.

# coeffs from table 1 in Finke, Dermer & Böttcher 08
coeffs_bel1 = [-0.35775237, -0.83695385, -1.1449608,
    -0.68137283, -0.22754737, -0.031967334]
coeffs_abv1 = [-0.35842494, -0.79652041, -1.6113032,
    0.26055213, -1.6979017, 0.032955035]

def syn_emiss_exact(gma, tnu):
  '''
  "exact" synchrotron emission function from Crusius & Schlikeiser (1986)
  for an electron with LF gma at tnu = nu'/nu'_B, nu'_B=e*B/(2*pi_*me_*c_)
  
  '''
  # underflows in np.exp will be treated as 0.
  np.seterr(under='ignore')
  if tnu < 1.:
    return 0.
  else:
    x = (2*tnu)/(3*gma**2)
    return func_R(x)

def func_R(x):
  '''
  R function see eqns (18) to (20) in Finke, Dermer & Böttcher 2008
  '''
  y = np.log10(x)
  if x < 0.01:
    R = 1.80842*x**(1./3.)
    return R
  elif (x>=0.01) and (x<1):
    A0, A1, A2, A3, A4, A5 = coeffs_bel1
    logR = A0 + A1*y + A2*y**2 + A3*y**3 + A4*y**4 + A5*y**5
    return 10**(logR)
  elif (x>=1) and (x<10):
    A0, A1, A2, A3, A4, A5 = coeffs_abv1
    logR = A0 + A1*y + A2*y**2 + A3*y**3 + A4*y**4 + A5*y**5
    return 10**(logR)
  else:
    R = .5*pi_*(1-99/(162*x))*np.exp(-x)
    return R


def get_cellData_withDistrib_analytic(key, k, Ng=20,
      scalefac=0, adiab=False, func_cooling=gamma_synCooled):
  '''
  Adds the electron distribution to the dataframe, cooling according to local conditions
  also adds normalized time (normalized to initial conditions)
  Separates (gmin, gmax) into Ng logarithmic bins
  '''
  df = get_cellData(key, k, fromSh=True)
  df = rescale_data(df, scalefac)
  env = MyEnv(key, scalefac)
  Nj = len(df)
  cell0 = df.iloc[0]
  dt_arr, tp_arr, dtp_arr, tt_arr, dtt_arr = np.zeros((5,Nj))
  gmas_arr = np.zeros((2+Ng,Nj))
  gmin0, gmax0 = get_vars(cell0, ['gma_m', 'gma_M'], env)
  gmas0 = np.geomspace(gmin0, gmax0, num=Ng+2, endpoint=True) # array [gmin0, gmax0] for Ng=0
  gmas_arr[:,0] = gmas0

  t = df.t.to_numpy(copy=True)
  dt = np.diff(t)
  dt = np.insert(dt, -1, 0.)
  tc0 = get_variable(cell0, "lfac", env)/get_variable(cell0, "syn", env)
  
  # "piecewise" cooling
  for j in range(Nj-1):
    cell, cell_next = df.iloc[j], df.iloc[j+1]
    lfac = get_variable(cell, "lfac", env)
    tc = 1/get_variable(cell, "syn", env)
    dt = cell_next.t - cell.t
    dtp = dt/lfac
    dtt = dtp/tc
    dt_arr[j] = dt
    dtp_arr[j] = dtp
    dtt_arr[j] = dtt
    tp_arr[j+1] = tp_arr[j] + dtp
    tt_arr[j+1] = tt_arr[j] + dtt
    gmas_next = func_cooling(dtt, gmas_arr[:,j])
    if adiab:
      adb = (cell_next.rho/cell.rho)**(1/3)
      gmas_next *= adb
    gmas_arr[:,j+1] = gmas_next
  gmas_arr[gmas_arr < 1.] = 1.

  # Create a dictionary of all new columns
  new_cols = {
      'dt': dt_arr,
      'tp': tp_arr,
      'dtp': dtp_arr,
      'tt': tt_arr,
      'dtt': dtt_arr,
      'gmin': gmas_arr[0]
  }

  # Add all gma columns
  if Ng > 0:
      for i in range(Ng):
          new_cols[f'gma_{i}'] = gmas_arr[i+1]

  new_cols['gmax'] = gmas_arr[-1]

  # Convert to DataFrame and concatenate
  new_df = pd.DataFrame(new_cols, index=df.index)
  out = pd.concat([df, new_df], axis=1)
  return out, env

def split_hydrostep(cell, cell_next, env, r_ref=1.5, Nmin=2, Nmax=20,
  func_cooling=gamma_synCooled):
  '''
  Splits based on gma_max ratio
  '''

  func_cooling=gamma_synCooled
  tcp = 1/get_variable(cell, 'syn', env)
  lfac = get_variable(cell, 'lfac', env)
  ratio = cell.gmax/cell_next.gmax
  N = int(np.log10(ratio)/np.log10(r_ref))
  N = max(N, Nmin)
  N = min(N, Nmax)

  # 1st version: count bins of gmas then separate t
  # if cell.tt == 0:
  #   dtt = 1/(cell.gmax*r_ref)
  #   tt_arr = np.geomspace(.1*dtt, cell_next.tt, num=N+1, endpoint=True)
  #   tt_arr = np.insert(tt_arr[1:], 0, 0.0)
  # else:
  #   tt_arr = np.geomspace(cell.tt, cell_next.tt, num=N+1, endpoint=True)
  
  # 2nd version: separate bins of expected gmas then bin time steps accordingly
  gmax_bins = np.geomspace(cell.gmax, cell_next.gmax, N+1, endpoint=True)
  tt_arr = gmax_bins**-1 - cell.gmax**-1  

  dtt_arr = np.diff(tt_arr)
  tt_arr = tt_arr[:-1]

  gma_keys = [key for key in cell.keys() if 'gm' in key]
  gmas = cell[gma_keys].to_numpy()
  gmas_next = func_cooling(tt_arr[np.newaxis, :], gmas[:, np.newaxis])

  # gmin_arr = func_cooling(tt_arr, cell.gmin)
  # gmax_arr = func_cooling(tt_arr, cell.gmax)
  tt_arr += cell.tt
  dtp_arr = dtt_arr * tcp
  dt_arr = dtp_arr * lfac
  tp_arr = np.insert(np.cumsum(dtp_arr[:-1]), 0, 0.) + cell['tp']
  t_arr = np.insert(np.cumsum(dt_arr[:-1]), 0, 0.) + cell.t

  # no interpolate hydro for now
  dic = {key:np.ones(N)*cell[key] for key in cell.keys()}
  t_vars = ['dtt', 'dtp', 'dt', 'tt', 'tp']
  t_vals = [dtt_arr, dtp_arr, dt_arr, tt_arr, tp_arr]
  dic_t = {k:v for k, v in zip(t_vars, t_vals)}
  dic.update(dic_t)
  dic_g = {k:v for k, v in zip(gma_keys, gmas_next)}
  dic.update(dic_g)
  out = pd.DataFrame.from_dict(dic)
  return out


def get_Fnu_cell(nuobs, Tobs, data, env,
  norm=True, width_tol=1.1, r_ref=1.2, Nmin=2, Nmax=20):
  '''
  F_nu(T) from a cell (dataframe with the cell history)
  array of size nuobs array
  '''

  Fnu_out = np.zeros(nuobs.shape)

  cell_0 = data.iloc[0]
  K0 = norm_plaw_distrib(cell_0.gmin, cell_0.gmax, env.psyn)
  nu_B0 = get_variable(cell_0, 'nu_B', env)

  # no need to calculate contributions if nu_M outside of observed freqs 
  gmax_cut = max(1., np.sqrt(nuobs.min()/nu_B0))
  gmax_arr = data.gmax.to_numpy()
  # np.searchsorted works in ascending order
  jcut = gmax_arr.size - np.searchsorted(gmax_arr[::-1], 5, side = "right")
  print(f'Calculating over {jcut} sim iterations')
  for j in range(jcut-1):
    hydro = data.iloc[j]
    hydro_next = data.iloc[j+1]
    Fnu = get_Fnu_hydrostep(nuobs, Tobs, hydro, hydro_next, K0, env, norm, width_tol, r_ref, Nmin, Nmax)
    Fnu_out += Fnu
  return Fnu_out


def get_Fnu_hydrostep(nuobs, Tobs, cell, cell_next, K0, env,
  norm=True, width_tol=1.1, r_ref=1.5, Nmin=2, Nmax=20):
  '''
  F_\nu (T) from a hydro step
  '''

  Fnu_out = np.zeros(nuobs.shape)
  steps = split_hydrostep(cell, cell_next, env, r_ref, Nmin, Nmax)
  Nj = len(steps)
  for j in range(Nj):
    step = steps.iloc[j]
    Fnu = get_Fnu_step(nuobs, Tobs, step, K0, env, norm, width_tol)
    Fnu_out += Fnu
  return Fnu_out



def get_Fnu_step(nuobs, Tobs, step, K0, env, norm=True, width_tol=1.1):
  '''
  F_\nu (T) from a (cooling) step
  '''
  
  D = get_variable(step, "Dop", env)
  nup = nuobs/D
  lfac = get_variable(step, 'lfac', env)
  tT = Tobs_to_tildeT(Tobs, step, env)
  Fnu = get_Lnu_comov(nup, step, K0, env, norm, width_tol)
  Fnu *= 2*lfac * tT**-2
  if norm:
    Fnu /= 2*env.lfac
  else:
    Fnu *= env.zdl
  return Fnu

def get_Lnu_comov(nup, step, K0, env, norm=True, width_tol=1.1):
  '''
  L'_\nu' from a (cooling) step
  '''

  nup_B = get_variable(step, 'nup_B', env)
  tnu = nup/nup_B

  Ttz = get_variable(step, "Tth", env)/(1+env.z)
  V3p = get_variable(step, "V3p", env)
  Lnu = get_epnu(tnu, step, K0, env, width_tol)
  Lnu *= V3p/Ttz
  if norm:
    Lnu /= env.L0p_Rshape
  return Lnu

def get_epnu(tnu_arr, step, K0, env, width_tol=1.1,
    func_cooling=gamma_synCooled, func_distrib=distrib_plaw_cooled, func_emiss=syn_emiss_exact):
  '''
  Delta e'_\nu' of a step (assuming constant P'_\nu' in the bin)
  K0 as variable because it depends on initial gma_min/max of the distrib
  '''
  p   = env.psyn
  gmin, gmax = step.gmin, step.gmax
  gma_keys = [key for key in step.keys() if 'gm' in key]
  Pmax = get_variable(step, "Pmax", env)
  K = K0
  if gmax/gmin <= width_tol:
    K = 1
    gma_mn = np.sqrt(gmin*gmax)
    #gma_mn = gmin
    enu = np.array([func_emiss(gma_mn, tnu) for tnu in tnu_arr])
  else:
    if len(gma_keys) > 2:
      gmas_arr = step[gma_keys].to_numpy(copy=True)
      enu = Pnu_instant_bins(tnu_arr, gmas_arr, env, func_distrib, func_emiss)
    else:
      enu = Pnu_instant(tnu_arr, gmin, gmax, env, func_distrib, func_emiss)
  enu *= K*step['dtp']*Pmax
  return enu


def Pnu_instant_bins(tnu_arr, gmas_arr, env,
    func_distrib=distrib_plaw_cooled, func_emiss=syn_emiss_exact):
  '''
  Energy per unit freq per unit volume and time at normalized time tt
  in units P'_e,max
  from an array of logarithmic bins
  ''' 
  p = env.psyn
  gmax = gmas_arr[-1]
  def func(gma, tnu):
    return func_distrib(gma, p, 1/gmax)*func_emiss(gma, tnu)
  
  Pnu = np.zeros(tnu_arr.shape)
  # center values in bins for calculation
  gmas_ctr = np.sqrt(gmas_arr[:-1] * gmas_arr[1:])
  dgmas = np.diff(gmas_arr)
  for i, tnu in enumerate(tnu_arr):
    for gma, dgma in zip(gmas_ctr, dgmas):
      Pnu[i] += func(gma, tnu)*dgma
  return Pnu

def Pnu_instant(tnu_arr, gmin, gmax, env,
    func_distrib=distrib_plaw_cooled, func_emiss=syn_emiss_exact):
  '''
  Energy per unit freq per unit volume and time at normalized time tt
  in units P'_e,max
  '''
  p = env.psyn
  def func(gma, tnu):
    return func_distrib(gma, p, 1/gmax)*func_emiss(gma, tnu)

  Pnu = np.zeros(tnu_arr.shape)
  #Pnu = np.array([compute_rad_instant(tnu, gmin, gmax, func_distrib, func_emiss, p) for tnu in tnu_arr])
  for i, tnu in enumerate(tnu_arr):
    int, err = spi.quad(func, gmin, gmax, args=(tnu))
    Pnu[i] += int
  return Pnu


## double quad t and gma
# def compute_emission(tnu, ti, tf, gmin0, gmax0, func_cooling, func_distrib, func_emiss, p):
#   '''
#   Computes the normalized emitted energy at normalized comoving frequency tnu
#   during a time span t_arr, with electron distribution gmas_arr0 at initial time
#   cooling following the function func_cooling and distribution func_distrib
#   '''

#   def gmin(tt):
#     return func_cooling(tt, gmin0)
#   def gmax(tt):
#     return func_cooling(tt, gmax0)
#   def func(tt, tnu):
#     return compute_rad_instant(tnu, gmin(tt), gmax(tt), func_distrib, func_emiss, p)

#   int, err = spi.quad(func, ti, tf, args=(tnu))
#   return int

