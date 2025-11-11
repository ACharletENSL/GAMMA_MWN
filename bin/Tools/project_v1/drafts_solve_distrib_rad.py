# -*- coding: utf-8 -*-
# @Author: acharlet

'''
This file contains functions that were written during the evolution of solve_distrib_rad
'''

#########################################################################
########################## Plotting functions ###########################


def testPlot_Lnu_cell(key='test', k=450):
  df, env = get_cellData_withDistrib_analytic(key, k)
  cell0 = df.iloc[0]
  nub = np.logspace(-7, 5, 100)
  nuobs = env.nu0 * nub
  Tobs = env.Ts

  # Lnu_arr1 = cell_Lnu_obs_steps(nuobs, Tobs, key, k)
  # Lnu_arr1 /= env.L0
  # Lnu1 = Lnu_arr1.sum(axis=0)
  # y1 = nub*Lnu1

  Lnu_arr2 = cell_run_emission_withsteps(key, k, nuobs, Tobs)
  Lnu_arr2 /= env.L0
  Lnu2 = Lnu_arr2.sum(axis=0)
  y2 = nub*Lnu2

  x_B = get_variable(cell0, 'nu_B', env)/env.nu0
  x_m, x_M = x_B*cell0.gmin**2, x_B*cell0.gmax**2
  tc = get_variable(cell0, "lfac", env)/get_variable(cell0, "syn", env)
  #tdyn = get_tdyn(df, env)
  tdyn = df.iloc[-1].t
  x_c = x_B * (tc/tdyn)**2
  #x_c = x_B*cell_next.gmax**2
  tT = Tobs_to_tildeT(Tobs, cell0, env)
  Lth = get_variable(cell0, 'Lth', env)
  x = nub
  y = (Lth/env.L0) * x * spectrum_bplaw(tT*x, x_B, x_c, x_m, x_M, env.psyn)
  # renormalize because it's not a plaw (1.075 is int R(x) dx)
  y *= 3*(p-1)/(p-2)*1.0751412667993323
  names = ['cooling', 'thinshell']
  norm = y2.max()/y.max()
  y2 /= norm
  Lnu_arr2 /= norm


  fig, ax = plt.subplots()
  ax.loglog(nub, y, c='orange', ls='--')
  ax.loglog(nub, y2, c='steelblue')
  y2max = y2.max()
  ymin, ymax = y2max*1e-4, y2max*3
  ax.set_ylim((ymin, ymax))
  # for Lnu in Lnu_arr1:
  #   ax.loglog(nub, nub*Lnu, c='saddlebrown', ls='--', lw=.85, zorder=-1)
  for Lnu in Lnu_arr2:
    ax.loglog(nub, nub*Lnu, c='darkslateblue', lw=.85, zorder=-1)

  ax.set_xlabel("$\\nu/\\nu_0$")
  ax.set_ylabel("$\\nu L_\\nu / \\nu_0 L_0$")
  # create_legend_styles(ax, 
  #   ["$\\Sigma P'_{\\nu'}\\delta t'$", "$\\int P'_{\\nu'}{\\rm d}t'$"],
  #   ['-', '--'])

  fig.tight_layout()

def testPlot_coolregimes(logs, key='test', k=450):
  # normalizes peaks to peak of line 0
  df, env = get_cellData_withDistrib_analytic(key, k)
  nub = np.logspace(-5, 5, 100)
  nuobs = env.nu0 * nub
  Tobs = env.Ts
  base = check_coolRegime(df, env)

  ncol = len(logs)
  colors = plt.cm.jet(np.linspace(0,1,ncol))
  xlabel, ylabel = '$\\nu/\\nu_0$', '$\\nu L_{\\nu}/\\nu_0 L_0$'

  fig, ax = plt.subplots()
  cols = []
  names = []
  refmax = 1
  for i, logRatio in enumerate(logs):
    if logRatio is not None:
      scalefac = logRatio - base
      name = f'{logRatio:.1f}'
    else:
      scalefac = 0
      name = f'{base:.1f}'
    Lnu = cell_run_emission(key, k, nuobs, Tobs, scalefac=scalefac)
    Lnu /= env.L0
    if i == 0:
      refmax = Lnu.max()
    else:
      Lnu *= refmax/Lnu.max()
    col = colors[i]
    cols.append(col)
    names.append(name)
    ax.loglog(nub, nub*Lnu, c=col)
  
  create_legend_colors(ax, names, cols, title='$\\log_{10}\\gamma_{\\rm c}/\\gamma_{\\rm m}$')
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  fig.tight_layout()



def testPlot_Lnu_step(j, jstep=1, key='test', k=450, Ni=5, Nt=20):
  df, env = get_cellData_withDistrib_analytic(key, k)
  nub = np.logspace(-4, 5, 150)
  nuobs = env.nu0 * nub
  Tobs = env.Ts
  func_cooling = gamma_synCooled
  func_distrib = distrib_plaw_cooled
  func_emiss = syn_emiss_exact

  p = env.psyn
  Nj = len(df)
  jnext = min(j+jstep, Nj)
  cell0, cell, cell_next = df.iloc[0], df.iloc[j], df.iloc[jnext]
  gmin0, gmax0 = cell0.gmin, cell0.gmax
  K = norm_plaw_distrib(gmin0, gmax0, p)
  Pe = get_variable(cell, 'Pmax', env)
  lfac = get_variable(cell, "lfac", env)
  tcp = 1/get_variable(cell, "syn", env)
  nuB = get_variable(cell0, 'nu_B', env)
  tnu_arr = nuobs/nuB

  df_step = hydro_step_splitting_interpolate(cell, cell_next, lfac, tcp, Nt,
    func_cooling=func_cooling)
  # Lnu_arr2 = []
  # for i in range(len(df_step)):
  #   step = df_step.iloc[i]
  #   enu = coolingCell_epnu(tnu_arr, step, env, K)
  #   Ttz = get_variable(step, "Tth", env)/(1+env.z)
  #   V3p = get_variable(step, "V3p", env)
  #   Lnu = enu * V3p/Ttz
  #   Lnu_arr2.append(Lnu)


  # dt_arr = np.array(dt_arr)
  # gmin_arr = np.array(gmin_arr)
  # gmax_arr = np.array(gmax_arr)
  # de_arr = np.array(de_arr)
  # t_arr = np.cumsum(dt_arr)
  # Lnu_arr = np.array(Lnu_arr)
  # Lnu_step_tr = np.trapezoid(Lnu_arr, t_arr, axis=0)
  # Lnu_arr = dt_arr[:,np.newaxis]*Lnu_arr
  # Lnu_step = Lnu_arr.sum(axis=0)
  # y = nub*Lnu_step
  # y_tr = nub*Lnu_step_tr
  dLnu_arr, dtp_arr = hydroStep_Lnus(nuobs, Tobs, j, df, env, Nt=Nt)
  dLnu_arr /= env.L0
  Lnu_sum, Lnu_trapz = np.zeros(dLnu_arr[0].shape), np.zeros(dLnu_arr[0].shape)
  for dLnu, dt in zip(dLnu_arr, dtp_arr):
    Lnu_sum += dLnu*dt
  tp_arr = np.cumsum(dtp_arr)
  Lnu_trapz += np.trapezoid(dLnu_arr, tp_arr, axis=0)
  y = nub*Lnu_sum
  y_tr = nub*Lnu_trapz
  
  # gmin_arr, gmax_arr = df_step.gmin, df_step.gmax
  # fig2, ax2 = plt.subplots()
  # ax2.loglog(t_arr, gmin_arr)
  # ax2.scatter(t_arr, gmin_arr, marker='x', c='k')
  # ax2.loglog(t_arr, gmax_arr)
  # ax2.scatter(t_arr, gmax_arr, marker='x', c='k')
  # ax2.set_ylabel("$\\gamma$")
  # ax2.set_xlabel('$t$')

  #fig3, ax3 = plt.subplots()
  # fig4, ax4 = plt.subplots()
  # isteps = [0, 5, 10, 15, 16, 17, 18, 19]
  # for step in [df_step.iloc[i]for i in isteps]:
  #   dt = step['dt']
  #   gmin, gmax = step.gmin, step.gmax
  #   gmas = np.geomspace(gmin, gmax)
  #   Pnu = Pnu_instant(tnu_arr, gmin, gmax, env)
  #   #ax3.loglog(gmas, func_distrib(gmas, p, 1/step.gmax))
  #   ax4.loglog(tnu_arr, tnu_arr*Pnu*dt)
  

  # Lnu2 = Lnu_obs(nuobs, Tobs, j, df, env)
  # y2 = nub*Lnu2/env.L0
  names = ["$\\Sigma P'_{\\nu'}\\delta t'$", "$\\int P'_{\\nu'}{\\rm d}t'$"]

  cell_f = df_step.iloc[-1]
  x_B = get_variable(cell0, 'nu_B', env)/env.nu0
  x_m, x_M = x_B*cell.gmin**2, x_B*cell.gmax**2
  # tdyn = get_tdyn(df, env)
  # x_c = x_B * (tc/tdyn)**2
  x_c = x_B*cell_f.gmax**2
  tT = Tobs_to_tildeT(Tobs, cell, env)
  Lth = get_variable(cell, 'Lth', env)
  x = nub
  y2 = (Lth/env.L0) * tT**-2 * x * spectrum_bplaw(tT*x, x_B, x_c, x_m, x_M, env.psyn)
  # renormalize because it's not a plaw (1.075 is int R(x) dx)
  y2 *= 3*(p-1)/(p-2)*1.0751412667993323
  #names = ['cooling', 'thinshell']
  norm = y_tr.max()/y2.max()
  print('num/theoric ratio: ', norm)
  #norm = 1.
  y2 *= norm


  fig, ax = plt.subplots()
  #fig.suptitle(f"Cell {k} step {j}")
  ax.loglog(nub, y, zorder=-1)
  ax.loglog(nub, y_tr, ls='-.')
  ax.loglog(nub, y2, ls='--')
  x_B = get_variable(cell, 'nu_B', env)/env.nu0
  xmax = min(x_B*cell.gmax**2, nub.max())*10
  yref = max(y.max(), y2.max())
  ymin, ymax = yref*1e-4, yref*3
  ymin /= norm
  ax.set_xlim((nub.min()*0.3, xmax))
  ax.set_ylim((ymin, ymax))
  for dLnu, dtp in zip(dLnu_arr, dtp_arr):
    ax.loglog(nub, nub*dLnu*dtp, c='cyan', lw=.85)
  
  ## compare between methods ot time splitting
  # df_step_test = OLD_hydro_step_splitting_interpolate(cell, cell_next, tc, Nt,
  #   func_cooling=func_cooling)
  # Lnu_arr = []
  # for i in range(len(df_step_test)):
  #   step = df_step_test.iloc[i]
  #   Lnu = coolingCell_Lnu(nuobs, Tobs, step, env, K, 1.5, func_cooling, func_distrib, func_emiss)
  #   Lnu_arr.append(Lnu)
  # Lnu_arr = np.array(Lnu_arr)
  # Lnu_arr /= env.L0
  # Lnu_step = Lnu_arr.sum(axis=0)
  # y = nub*Lnu_step
  # ax.loglog(nub, y, c='k', ls='--')
  # for Lnu in Lnu_arr:
  #   ax.loglog(nub, nub*Lnu, c='k',ls=':', lw=.85)
  # fig2, ax2 = plt.subplots()
  # ax2.semilogy(df_step.dt/tc)
  # ax2.semilogy(df_step_test.dt/tc, ls='--', c='k')
  # ax2.axhline(.5/cell.gmax, ls=':', c='k')
  
  ax.set_xlabel("$\\nu/\\nu_0$")
  ax.set_ylabel("$\\nu L_\\nu / \\nu_0 L_0$")
  names=["$\\Sigma \\delta t L_\\nu$", "$\\int {\\rm d}t L_\\nu$ (trapz)", f"bpl x {norm:.1f}"]
  create_legend_styles(ax, names, ['-', '-.', '--'])
  trans = transy(ax)
  gmin_i, gmax_i = cell.gmin, cell.gmax
  gmin_f, gmax_f = cell_f.gmin, cell_f.gmax
  gmas, subs = [], []
  if gmax_i/gmin_i < 2.:
    gmean_i = np.sqrt(gmin_i*gmax_i)
    gmas.append(gmean_i)
    subs.append('mean,i')
  else:
    gmas += [gmin_i, gmax_i]
    subs += ['m,i', 'M,i']
  if gmax_f/gmin_f < 2.:
    gmean_f = np.sqrt(gmin_f*gmax_f)
    gmas.append(gmean_f)
    subs.append('mean,f')
  else:
    gmas += [gmin_f, gmax_f]
    subs += ['m,f', 'M,f']
  names =['$\\gamma_{\\rm ' + s + '}^2$' for s in subs]
  for gma, name in zip(gmas, names):
    ax.axvline(x_B*gma**2, c='k', ls=':')
    ax.text(x_B*gma**2, 1.02, name, transform=trans, horizontalalignment='center')
  fig.tight_layout()



def testPlot_comparePow(tt=0):
  fig, ax = plt.subplots()
  tnu_arr = np.logspace(1, 15, 300)
  gmin0, gmax0 = 1e3, 1e7
  p = 2.5
  gmin = gmin0/(1+gmin0*tt)
  gmax = gmax0/(1+gmax0*tt)
  title = f"cooled p. law distrib ($\\tilde{{t}}="+sci_notation(tt)[1:-1]+")$"
  Pnu = np.array([compute_rad_instant(tnu, gmin, gmax, distrib_plaw_cooled, syn_emiss_simple, p) for tnu in tnu_arr])
  Pnu2 = np.array([compute_rad_instant(tnu, gmin, gmax, distrib_plaw_cooled, syn_emiss_exact, p) for tnu in tnu_arr])
  y1 = tnu_arr*Pnu
  y2 = tnu_arr*Pnu2
  ymin = y1[0]*1e-4
  ymax = y1.max()*10**0.4
  ax.loglog(tnu_arr, y1, ls='--', label='$x^{1/3}$')
  ax.loglog(tnu_arr, y2, ls='-', label='$R(x)$')
  ax.set_ylim((ymin, ymax))
  ax.set_ylabel("$\\nu' P'_{\\nu'}$")
  ax.set_xlabel("$\\nu'/\\nu'_{\\rm B}$")
  ax.set_title(title, fontsize=14)
  ax.legend(title="$P'_{\\nu'}(x)\\propto$")
  fig.tight_layout()

def testPlot_R(gma):
  fig, ax = plt.subplots()
  tnu_arr = np.logspace(1, 12, 500)
  x = (2*tnu_arr)/(3*gma**2)
  R = [syn_emiss_exact(gma, tnu) for tnu in tnu_arr]
  y = np.zeros(x.shape)
  y[x<=1.5] = x[x<=1.5]**(1/3)
  ax.loglog(x, y, ls='--', label='$x^{1/3}$')
  ax.loglog(x, R, label='$R(x)$')
  ax.set_title(f'$\\gamma =$'+sci_notation(gma))
  ax.set_xlabel("$x=\\frac{2}{3}\\frac{\\tilde{\\nu}}{\\gamma^2}$")
  ax.set_ylabel("$R(x)$")
  ax.legend()
  fig.tight_layout()


def testPlot_gmas(key='test', k=450, func_cooling=gamma_synCooled):
  '''
  Plots gamma_min, max
  '''
  df, env = get_cellData_withDistrib_analytic(key, k, func_cooling=func_cooling)
  cell0 = df.iloc[0]
  tc0 = get_variable(cell0, "lfac", env)/get_variable(cell0, "syn", env)
  x = (df.t - cell0.t)/tc0

  fig, ax = plt.subplots()
  for var, col in zip(['gmin', 'gmax'], ['orange', 'r']):
    val = getattr(df, var).to_numpy()
    ax.semilogy(x, val, c=col)
    x_th = func_cooling(x, val[0])
    x_th[x_th<1.] = 1.
    ax.semilogy(x, x_th, ls=':', c='k')
  ax.set_xlabel("$t/t_{\\rm c,0}$")
  ax.set_ylabel("$\\gamma_{\\rm e}$")
  fig.tight_layout()

#########################################################################
############################### Third version ###########################

def split_hydrostep(hydro, tt_f, env, Nt=20, func_cooling=gamma_synCooled):
  '''
  Like the name suggests, tt_f is the normalized time of next hydro step,
  counted from hydro state
  TO DO: change fixed NT by something more adaptive?
  '''

  lfac = get_variable(hydro, 'lfac', env)
  tcp = 1/get_variable(hydro, 'syn', env)

  # define normalized time array
  gmin, gmax = hydro.gmin, hydro.gmax
  dtt = 1/gmax
  if dtt > tt_f:
    # to add: split intro 3 linear steps
    cell_out = hydro.copy()
    dic = cell_out.to_dict()
    dic = {key:[dic[key]] for key in dic.keys()} # convert dic of scalars into dic of arrays
    df_out = pd.DataFrame.from_dict(dic)
    return df_out
  tt_arr = np.geomspace(.1*dtt, tt_f, num=Nt+1, endpoint=True)
  tt_arr = np.insert(tt_arr[1:], 0, 0.0)
  dtt_arr = np.diff(tt_arr)
  tt_arr = tt_arr[:-1]

  gmin_arr = func_cooling(tt_arr, gmin)
  gmax_arr = func_cooling(tt_arr, gmax)

  dtp_arr = dtt_arr * tcp
  dt_arr = dtp_arr * lfac
  tp_arr = np.insert(np.cumsum(dtp_arr[:-1]), 0, 0.) + hydro['tp']
  t_arr = np.insert(np.cumsum(dt_arr[:-1]), 0, 0.) + hydro.t
  
  # no interpolate hydro for now
  dic = {key:np.ones(Nt)*hydro[key] for key in hydro.keys()}
  #dic['x'] = hydro.x + dt_arr * hydro.vx
  dic['gmin'] = gmin_arr
  dic['gmax'] = gmax_arr
  dic['dtt'] = dtt_arr
  dic['dtp'] = dtp_arr
  dic['dt']  = dt_arr
  dic['tt'] = tt_arr
  dic['tp'] = tp_arr
  #dic['t'] = t_arr
  df_out = pd.DataFrame.from_dict(dic)

  return df_out

def get_Fnu_hydrostep(nuobs, Tobs, hydro, tt_f, K0, env, width_tol=1.1, norm=True):
  '''
  F_\nu (T) from a hydro step, tt_f = hydro_next.tt - hydro.tt
  '''

  Fnu_out = np.zeros(nuobs.shape)
  steps = split_hydrostep(hydro, tt_f, env)
  Nj = len(steps)
  for j in range(Nj):
    step = steps.iloc[j]
    Fnu = get_Fnu_step(nuobs, Tobs, step, K0, env, width_tol, norm)
    Fnu_out += Fnu
  return Fnu_out

def get_Fnu_cell(nuobs, Tobs, cell, env, width_tol=1.1, norm=True):
  '''
  F_nu(T) from a cell (dataframe with the cell history)
  array of size nuobs array
  '''

  Fnu_out = np.zeros(nuobs.shape)

  cell_0 = cell.iloc[0]
  K0 = norm_plaw_distrib(cell_0.gmin, cell_0.gmax, env.psyn)
  nu_B0 = get_variable(cell_0, 'nu_B', env)

  # no need to calculate contributions if nu_M outside of observed freqs 
  gmax_cut = max(1., np.sqrt(nuobs.min()/nu_B0))
  gmax_arr = cell.gmax.to_numpy()
  jcut = np.searchsorted(gmax_arr, gmax_cut)
  for j in range(jcut-1):
    hydro = cell.iloc[j]
    tt_f = cell.iloc[j+1].tt - hydro.tt
    Fnu = get_Fnu_hydrostep(nuobs, Tobs, hydro, tt_f, K0, env, width_tol, norm)
    Fnu_out += Fnu
  return Fnu_out

#########################################################################
################## Second version (with t splitting) ####################

def run_emission(key, nuobs, Tobs, jstep=1, scalefac=0,
    func_cooling=gamma_synCooled, func_distrib=distrib_plaw_cooled, func_emiss=syn_emiss_exact):
  '''
  Observed flux from whole sim
  '''
  d0 = openData(key, 0)
  klist = d0.loc[d0['trac']>0]['i'].to_list()

  # out shape
  if (type(Tobs)==np.ndarray) and (type(nuobs)==np.ndarray):
    if len(nuobs) == 1:
      out = np.zeros(len(Tobs))
      nuobs = nuobs[0]
    elif len(Tobs) == 1:
      out = np.zeros(len(nuobs))
      Tobs = Tobs[0]
    else:
      out = np.zeros((len(Tobs), len(nuobs)))
  elif (type(Tobs)==np.ndarray):
    out = np.zeros(Tobs.shape)
  elif (type(nuobs)==np.ndarray):
    out = np.zeros(nuobs.shape)
  
  for k in klist:
    Lnu = cell_run_emission(key, k, nuobs, Tobs, jstep, scalefac,
      func_cooling, func_distrib, func_emiss)
    out += Lnu
  

def cell_run_emission_withsteps(key, k, nuobs, Tobs, jstep=1, Nt=20, scalefac=0,
    func_cooling=gamma_synCooled, func_distrib=distrib_plaw_cooled, func_emiss=syn_emiss_exact):
  '''
  Emitted luminosity from a cell across its cooling
  scalefac rescales the cooling time by a factor 10**scalefac
  '''
  
  df, env = get_cellData_withDistrib_analytic(key, k, scalefac, func_cooling)
  Nj = len(df)
  Lnu_arr = []
  nuB_0 = get_variable(df.iloc[0], "nu_B", env)
  gma_max_obs = np.sqrt(nuobs.min()/nuB_0)
  for j in range(0, Nj, jstep):
    gmax = df.iloc[j].gmax
    if gmax < .1*gma_max_obs:
      print(f"Cell {k} cooled below relevant obs frequency in {j} steps, stopping calculation.")
      break
    if gmax == 1.:
      print(f"Cell {k} cooled in {j} steps")
      break
    Lnu = hydroStep_Lnu(nuobs, Tobs, j, df, env, jstep, Nt)
    Lnu_arr.append(Lnu)
  
  return np.array(Lnu_arr)


def cell_run_emission(key, k, nuobs, Tobs, jstep=1, Nt=20, scalefac=0, width_tol=1.1,
    func_cooling=gamma_synCooled, func_distrib=distrib_plaw_cooled, func_emiss=syn_emiss_exact):
  '''
  Emitted luminosity from a cell across its cooling
  scalefac rescales the cooling time by a factor 10**scalefac
  '''

  # out shape
  if (type(Tobs)==np.ndarray) and (type(nuobs)==np.ndarray):
    if len(nuobs) == 1:
      out = np.zeros(len(Tobs))
      nuobs = nuobs[0]
    elif len(Tobs) == 1:
      out = np.zeros(len(nuobs))
      Tobs = Tobs[0]
    else:
      out = np.zeros((len(Tobs), len(nuobs)))
  elif (type(Tobs)==np.ndarray):
    out = np.zeros(Tobs.shape)
  elif (type(nuobs)==np.ndarray):
    out = np.zeros(nuobs.shape)
  
  df, env = get_cellData_withDistrib_analytic(key, k, scalefac, func_cooling)
  Nj = len(df)
  nuB_0 = get_variable(df.iloc[0], "nu_B", env)
  gma_max_obs = np.sqrt(nuobs.min()/nuB_0)
  for j in range(0, Nj-jstep, jstep):
    gmax = df.iloc[j].gmax
    if gmax < .5*gma_max_obs:
      print(f"Cell {k} cooled below relevant obs frequency in {j} steps,.")
      break
    if gmax == 1.:
      print(f"Cell {k} cooled in {j} steps")
      break
    Lnu = hydroStep_Lnu(nuobs, Tobs, j, df, env, jstep, Nt, width_tol)
    out += Lnu
  
  return out

def hydroStep_Lnu(nuobs, Tobs, j, df, env, jstep=1, Nt=20, width_tol=1.1, method='sum',
    func_cooling=gamma_synCooled, func_distrib=distrib_plaw_cooled, func_emiss=syn_emiss_exact):

  # out shape
  if (type(Tobs)==np.ndarray) and (type(nuobs)==np.ndarray):
    if len(nuobs) == 1:
      out = np.zeros(len(Tobs))
      nuobs = nuobs[0]
    elif len(Tobs) == 1:
      out = np.zeros(len(nuobs))
      Tobs = Tobs[0]
    else:
      out = np.zeros((len(Tobs), len(nuobs)))
  elif (type(Tobs)==np.ndarray):
    out = np.zeros(Tobs.shape)
  elif (type(nuobs)==np.ndarray):
    out = np.zeros(nuobs.shape)
  
  Lnu_arr, dt_arr = hydroStep_Lnus(nuobs, Tobs, j, df, env, jstep, Nt, width_tol,
      func_cooling, func_distrib, func_emiss)
  if method == 'sum':
    for Lnu, dt in zip(Lnu_arr, dt_arr):
      out += Lnu*dt
  elif method == 'trapz':
    t = np.cumsum(dt_arr)
    out += np.trapezoid(Lnu_arr, t_arr, axis=0)
  return out


def hydroStep_Lnus(nuobs, Tobs, j, df, env, jstep=1, Nt=20, width_tol=1.1,
    func_cooling=gamma_synCooled, func_distrib=distrib_plaw_cooled, func_emiss=syn_emiss_exact):
  '''
  Derives emitted luminosity during a given hydro time step
  '''

  # separate hydro step
  p = env.psyn
  Nj = len(df)
  jnext = min(j+jstep, Nj)
  cell0, cell, cell_next = df.iloc[0], df.iloc[j], df.iloc[jnext]
  gmin0, gmax0 = cell0.gmin, cell0.gmax
  K = norm_plaw_distrib(gmin0, gmax0, p)
  Pe  = get_variable(cell, 'Pmax', env)
  tc = get_variable(cell, "lfac", env)/get_variable(cell, "syn", env)
  df_step = hydro_step_splitting_interpolate(cell, cell_next, tc, Nt,
    func_cooling=func_cooling)

  # sub steps emission
  dLnu_arr, dtp_arr = [], [] 
  for i in range(len(df_step)):
    step = df_step.iloc[i]
    tT = Tobs_to_tildeT(Tobs, step, env)
    tnu = nuobs_to_nucomov_normalized(nuobs, step, env, 'syn')
    dtp = step['dtp']
    dtp_arr.append(dtp)
    enu = coolingCell_epnu(tT*tnu, step, env, K)
    Ttz = get_variable(step, "Tth", env)/(1+env.z)
    V3p = get_variable(step, "V3p", env)
    dLnu = (enu/dtp) * V3p/Ttz
    dLnu_arr.append(dLnu)

  return np.array(dLnu_arr), np.array(dtp_arr)


def hydro_step_splitting_interpolate(cell, cell_next, lfac, tcp, Nt=20,
    func_cooling=gamma_synCooled):
  '''
  Splits the hydro time step between states cell and cell_next
  depending on the "cooling" timescale tcp, interpolates hydro in between
  values at half step
  '''

  # define normalized time array
  tt_f = (cell_next.tt - cell.tt) 
  dtt = 1/cell.gmax
  if dtt > tt_f:
    # to add: split intro 3 linear steps
    cell_out = cell.copy()
    dic = cell_out.to_dict()
    dic = {key:[dic[key]] for key in dic.keys()} # convert dic of scalars into dic of arrays
    df_out = pd.DataFrame.from_dict(dic)
    return df_out
  tt_arr = np.geomspace(.1*dtt, tt_f, num=Nt+1, endpoint=True)
  tt_arr = np.insert(tt_arr[1:], 0, 0.0)
  dtt_arr = np.diff(tt_arr)
  tt_arr = tt_arr[:-1]

  gmin_arr = func_cooling(tt_arr, cell.gmin)
  gmax_arr = func_cooling(tt_arr, cell.gmax)

  dtp_arr = dtt_arr * tcp
  dt_arr = dtp_arr * lfac
  tp_arr = np.insert(np.cumsum(dtp_arr[:-1]), 0, 0.) + cell['tp']
  t_arr = np.insert(np.cumsum(dt_arr[:-1]), 0, 0.) + cell.t
  # interpolate hydro
  # we keep actual hydro values constant for now
  # x, t, gmin, gmax interpolated
  dic = {key:np.ones(Nt)*cell[key] for key in cell.keys()}
  dic['x'] = cell.x + dt_arr * cell.vx
  dic['gmin'] = gmin_arr
  dic['gmax'] = gmax_arr
  dic['dtt'] = dtt_arr
  dic['dtp'] = dtp_arr
  dic['dt']  = dt_arr
  dic['tt'] = tt_arr
  dic['tp'] = tp_arr
  dic['t'] = t_arr
  df_out = pd.DataFrame.from_dict(dic)

  return df_out

def get_splittedCell(j, key='test', k=450, Nt=20, scalefac=0, func_cooling=gamma_synCooled):
  '''
  Returns splitted hydro time step
  '''
  df, env = get_cellData_withDistrib_analytic(key, k)
  cell, cell_next = df.iloc[j], df.iloc[j+1]
  lfac = get_variable(cell, "lfac", env)
  tcp = 1/get_variable(cell, "syn", env)
  df_step = hydro_step_splitting_interpolate(cell, cell_next, lfac, tcp, Nt, func_cooling)
  return df_step


def TESTING_hydro_step_splitting_interpolate(cell, cell_next, tc, r=2,
    func_cooling=gamma_synCooled):
  '''
  Splits the hydro time step between states cell and cell_next
  depending on the "cooling" timescale tc, interpolates hydro in between
  fix factor r between step size 
    then generate time array starting from dt0 = tc/r
  '''

  # define normalized time array
  tt_f = (cell_next.t - cell.t)/tc 
  dtt = 1/cell.gmax
  if dtt > tt_f:
    # to add: split intro 3 linear steps
    cell_out = cell.copy()
    cell_out['dt'] = cell_next.t - cell.t
    dic = cell_out.to_dict()
    dic = {key:[dic[key]] for key in dic.keys()} # convert dic of scalars into dic of arrays
    df_out = pd.DataFrame.from_dict(dic)
    return df_out
  tt_arr = np.geomspace(.01*dtt, tt_f, num=Nt+1, endpoint=True)
  tt_arr = np.insert(tt_arr[1:], 0, 0.0)
  dtt_arr = np.diff(tt_arr)
  tt_arr = tt_arr[:-1]

  gmin_arr = func_cooling(tt_arr, cell.gmin)
  gmax_arr = func_cooling(tt_arr, cell.gmax)

  dt_arr = dtt_arr * tc   # in lab frame, for interpolating
  # interpolate hydro
  # we keep actual hydro values constant for now
  # x, t, tt, gmin, gmax interpolated
  dic = {key:np.ones(Nt)*cell[key] for key in cell.keys()}
  dic['x'] = cell.x + dt_arr * cell.vx
  dic['t'] = cell.t + dt_arr
  #dic['dt'] = np.diff(np.append(dic['t'],cell_next.t))
  dic['dt'] = dt_arr
  dic['gmin'] = gmin_arr
  dic['gmax'] = gmax_arr
  dic['tt'] = tt_arr
  df_out = pd.DataFrame.from_dict(dic)

  return df_out


# def OLD_hydro_step_splitting_interpolate(cell, cell_next, tc, Nt=20,
#     func_cooling=gamma_synCooled):
#   '''
#   Splits the hydro time step between states cell and cell_next
#   depending on the "cooling" timescale tc, interpolates hydro in between
#   values at half step
#   '''

#   # define normalized time array
#   tt_f = (cell_next.t - cell.t)/tc 
#   dtt = 1/(2*cell.gmax)
#   if dtt > tt_f:
#     cell_out = cell.copy()
#     cell_out['dt'] = cell_next.t - cell.t
#     dic = cell_out.to_dict()
#     dic = {key:[dic[key]] for key in dic.keys()} # convert dic of scalars into dic of arrays
#     df_out = pd.DataFrame.from_dict(dic)
#     return df_out
#   tt_arr = np.geomspace(dtt, tt_f, num=Nt, endpoint=True)
#   tt_arr = np.insert(tt_arr[:-1], 0, 0.0)
#   gmin_arr = func_cooling(tt_arr, cell.gmin)
#   gmax_arr = func_cooling(tt_arr, cell.gmax)

#   dt_arr = tt_arr * tc   # in lab frame, for interpolating
#   # interpolate hydro
#   # we keep actual hydro values constant for now
#   # x, t, tt, gmin, gmax interpolated
#   dic = {key:np.ones(Nt)*cell[key] for key in cell.keys()}
#   dic['x'] = cell.x + dt_arr * cell.vx
#   dic['t'] = cell.t + dt_arr
#   dic['tt'] = tt_arr
#   dic['dt'] = np.diff(np.append(dic['t'],cell_next.t))
#   dic['gmin'] = gmin_arr
#   dic['gmax'] = gmax_arr
#   df_out = pd.DataFrame.from_dict(dic)
#   return df_out

#########################################################################
#################### Some modifs on 1st version #########################

def coolingCell_Lnu(nuobs, Tobs, cell, env, K, width_tol=1.1,
    func_cooling=gamma_synCooled, func_distrib=distrib_plaw_cooled, func_emiss=syn_emiss_exact):
  '''
  Derives luminosity from a "subcell" of the cooling time stepping
  '''
  p   = env.psyn
  gmin, gmax = cell.gmin, cell.gmax
  tT  = Tobs_to_tildeT(Tobs, cell, env)
  tnu = nuobs_to_nucomov_normalized(nuobs, cell, env, norm='syn')
  tnu_arr = tT*tnu
  if (gmin==0) or (gmax==0):
    return np.zeros(tnu_arr.shape)

  Ttz = get_variable(cell, "Tth", env)/(1+env.z)
  V3p  = get_variable(cell, "V3p", env)
  if gmax/gmin <= width_tol:
    gma_mn = np.sqrt(gmin*gmax)
    K = 1.
    Lnu = np.array([func_emiss(gma_mn, tnu) for tnu in tnu_arr])
  else:
    Lnu = Pnu_instant(tnu_arr, gmin, gmax, env, func_distrib, func_emiss)
  Lnu *= (K*V3p/Ttz) * tT**-2
  # Lnu = coolingCell_epnu(tnu_arr, cell, env, K, width_tol,
  #     func_cooling, func_distrib, func_emiss)
  # Lnu *= V3p / (tT**2 * Ttz)
  return Lnu

def coolingCell_epnu(tnu_arr, cell, env, K, width_tol=1.1,
    func_cooling=gamma_synCooled, func_distrib=distrib_plaw_cooled, func_emiss=syn_emiss_exact):
  '''
  E'_\nu'/\Delta V'^(3) of a cell (assuming constant P'_\nu' in the bin)
  '''
  p   = env.psyn
  gmin, gmax = cell.gmin, cell.gmax
  Pmax = get_variable(cell, "Pmax", env)
  if gmax/gmin <= width_tol:
    gma_mn = np.sqrt(gmin*gmax)
    K = 1.
    enu = np.array([func_emiss(gma_mn, tnu) for tnu in tnu_arr])
  else:
    enu = Pnu_instant(tnu_arr, gmin, gmax, env, func_distrib, func_emiss)
  enu *= K*cell['dtp']*Pmax
  return enu

#########################################################################
############################### First version ###########################

def Epnu_step(tnu_arr, j, df, env, step=1,
    func_cooling=gamma_synCooled, func_distrib=distrib_plaw_cooled, func_emiss=syn_emiss_exact):
  '''
  Energy per unit freq emitted by synchrotron during step j in comoving frame
  '''

  p = env.psyn
  Nj = len(df)
  jnext = min(j+step, Nj)
  cell0, cell, cell_next = df.iloc[0], df.iloc[j], df.iloc[jnext]
  gmin0, gmax0 = cell0.gmin, cell0.gmax
  gmin, gmax = cell.gmin, cell.gmax
  if (gmin==0) or (gmax==0):
    return np.zeros(tnu_arr.shape)

  ti, tf = cell.tt, cell_next.tt
  Enu = np.array([compute_emission(tnu, ti, tf, gmin0, gmax0, func_cooling, func_distrib, func_emiss, p) for tnu in tnu_arr])
  K = norm_plaw_distrib(gmin0, gmax0, p)
  Pe = get_variable(cell, 'Pmax', env)
  V3 = get_variable(cell, 'V3p', env)
  tc = 1/get_variable(cell0, 'syn', env)
  norm = K*Pe*tc*V3
  Enu *= norm
  return Enu

def Lnu_obs(nuobs, Tobs, j, df, env, step=1,
    func_cooling=gamma_synCooled, func_distrib=distrib_plaw_cooled, func_emiss=syn_emiss_exact):
  '''
  Energy per unit freq emitted by synchrotron during step j in observed frame
  '''
  if (type(nuobs)==np.ndarray) and (type(Tobs)==np.ndarray):
    nuobs = nuobs[np.newaxis, :]
    Tobs  = Tobs[:, np.newaxis]

  cell = df.iloc[j]
  Tth = get_variable(cell, 'Tth', env)
  tT = Tobs_to_tildeT(Tobs, cell, env)
  tnu = nuobs_to_nucomov_normalized(nuobs, cell, env, norm='syn')
  Lnu = Epnu_step(tnu*tT, j, df, env, step,
      func_cooling, func_distrib, func_emiss)
  Lnu *= ((1+env.z)/Tth) * tT**-2

  # cell0 = df.iloc[0]
  # cell_next = df.iloc[j+1]
  # ti, tf = cell.tt, cell_next.tt
  # V3p = get_variable(cell, 'V3p', env)
  # tc = 1/get_variable(cell0, 'syn', env)
  # V4 = (tf-ti)*tc*V3p
  # print("Lnu_obs,  Tthz: ", Tth/(1+env.z))
  # print("Lnu_obs, V4: ", V4)
  # print("Lnu_obs, Lnu.max()/V4: ", Lnu.max()/V4)
  return Lnu

def cell_Lnu_obs(nuobs, Tobs, key, k, itstep=1,
    func_cooling=gamma_synCooled, func_distrib=distrib_plaw_cooled, func_emiss=syn_emiss_exact):
  '''
  Flux emitted by a cell through synchrotron cooling in observed frame
  '''
  if (type(Tobs)==np.ndarray) and (type(nuobs)==np.ndarray):
    if len(nuobs) == 1:
      out = np.zeros(len(Tobs))
      nuobs = nuobs[0]
    elif len(Tobs) == 1:
      out = np.zeros(len(nuobs))
      Tobs = Tobs[0]
    else:
      out = np.zeros((len(Tobs), len(nuobs)))
  elif (type(Tobs)==np.ndarray):
    out = np.zeros(Tobs.shape)
  elif (type(nuobs)==np.ndarray):
    out = np.zeros(nuobs.shape)

  df, env = get_cellData_withDistrib_analytic(key, k, func_cooling=func_cooling)
  Nj = len(df)
  nuB_0 = get_variable(df.iloc[0], "nu_B", env)
  gma_max_obs = np.sqrt(nuobs.min()/nuB_0)
  for j in range(0, Nj, itstep):
    gmax = df.iloc[j].gmax
    if gmax < .5*gma_max_obs:
      print(f"(old method) Cell {k} cooled below relevant obs frequency after {j} steps.")
      break
    if gmax == 1.:
      print(f"(old method) Cooling of cell {k} finished after {j} iterations.")
      break
    # derive emission
    Lnu = Lnu_obs(nuobs, Tobs, j, df, env, itstep, func_cooling, func_distrib, func_emiss)
    out += Lnu

  return out

def cell_Lnu_obs_steps(nuobs, Tobs, key, k, itstep=1,
    func_cooling=gamma_synCooled, func_distrib=distrib_plaw_cooled, func_emiss=syn_emiss_exact):
  '''
  Flux emitted by a cell through synchrotron cooling in observed frame
  outputs array of the steps
  '''

  df, env = get_cellData_withDistrib_analytic(key, k, func_cooling=func_cooling)
  Nj = len(df)
  its = df.index.to_list()
  Lnu_arr = []
  nuB_0 = get_variable(df.iloc[0], "nu_B", env)
  gma_max_obs = np.sqrt(nuobs.min()/nuB_0)
  for j in range(0, Nj, itstep):
    gmax = df.iloc[j].gmax
    if gmax < .5*gma_max_obs:
      print(f"(old method) cell {k} cooled below relevant obs frequency, stopping calculation.")
      break
    if gmax == 1.:
      print(f"(old method) cooling of cell {k} finished after {j} iterations.")
      break
    # derive emission
    Lnu = Lnu_obs(nuobs, Tobs, j, df, env, itstep, func_cooling, func_distrib, func_emiss)
    Lnu_arr.append(Lnu)

  Lnu_arr = np.array(Lnu_arr)
  return Lnu_arr
