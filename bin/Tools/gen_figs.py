'''
Script to generate figures for documents
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.transforms as transforms
import matplotlib.colors as colors
from plotting_scripts_new import *
from hydro2rad_syn import *

# List of figures to do
## Article
### Hydro
#### planar
# snapshots vs theory: initial conditions, intermediate (0.8*tRS?), post crossing (1.2*tRS?)
# Minu hydro fig 5: E(t), M(t), Delta(t) (vs theory?) for zones
# (x-env.x)/env.x (t), for u and shock strength
#### spherical
# snapshots (no theory), same relative times
# u(t) behind RS, at CD, behind FS; shock strength(t) for both shocks


regcol = {'fast':'m', 'slow':'c', 'monoE':'g'}
cols_dic = {'RS':'r', 'FS':'b', 'RS+FS':'k', 'CD':'purple',
  'CD-':'mediumvioletred', 'CD+':'mediumpurple', '3':'darkred', '2':'darkblue'}

# Review article
# --------------------------------------------------------------------------------------------------
class localEnv:
  def __init__(self, params, parlist):
    for key, val in zip(parlist, params):
      setattr(self, key, val)

def get_twinax():
  fig = plt.gcf()
  ax1 = plt.gca()
  ax2 = ax1.twinx()
  return ax2

def plot_Pnu_celltimestep(n, istep=0, key='Last'):
  '''
  Plots instantaneous P'_\nu' at start half, and end of a given time step
  '''

  fig, ax = plt.subplots()

  nuobs, Tobs, env = radEnv(key, True)
  dk = get_gmasdata_cell(n, key)
  p =env.psyn

  cell = dk.iloc[istep]
  cell_next = dk.iloc[istep+1]

  gmin0 = cell['gmin_']
  tt_max = (cell_next['t'] - cell['t'])/get_variable(cell, 'lfac', env)
  nup = nuobs/get_variable(cell, 'Dop', env)
  tnu = nup/get_variable(cell, 'nup_B', env)

  Pnu_0 = Pnu_cooling(tnu, 0, gmin0, p)
  Pnu_h = Pnu_cooling(tnu, .5*tt_max, gmin0, p)
  Pnu_f = Pnu_cooling(tnu, tt_max, gmin0, p)
  Pnus = [Pnu_0, Pnu_h, Pnu_f]

  x = nuobs/env.nu0
  for y, name in zip(Pnus, ["$t'_{\\rm j}$", "$(t'_{\\rm j}+t'_{\\rm j})/2$", "$t'_{\\rm j+1}$"]):
    ax.loglog(x, y, label=name)
  
  ax.legend()
  ax.set_xlabel("$\\nu/\\nu_0")
  ax.set_ylabel("$P'_{\\nu'}/n'P_{\\rm e,max}$")
  plt.tight_layout()


def plot_shockedCell(var, n, key='Last', xscale='t', logx=False, logy=True, ax_in=None, **kwargs):
  '''
  Plots value of variable var in cell n (counted from the CD) while there is accelerated electrons
  '''

  env = MyEnv(key)
  dk = get_gmasdata_cell(n, key)
  col = 'r' if n < 0 else 'b'

  if not ax_in:
    fig, ax = plt.subplots()
  else:
    ax = ax_in

  it, t, r = dk[['it', 't', 'x']].to_numpy().transpose()
  val = get_variable(dk, var, env)
  x = it
  xlabel = 'it'
  if xscale == 'r':
    xlabel = '$r/R_0$'
    x = r*c_/env.R0
  elif xscale == 't':
    xlabel = '$t/t_0$'
    x = t/env.t0 + 1.
  elif xscale == 'T':
    xlabel = '$\\bar{T}$'
    T = (1+env.z)*(t + env.t0 - r)
    x = (T-T[0])/env.T0
  
  l1, = ax.plot(x, val, **kwargs)

  if logx: ax.set_xscale('log')
  if logy: ax.set_yscale('log')

  if not ax_in:
    l1.set_color(col)
    ax.set_xlabel(xlabel)
  
  ax.set_ylabel(var_exp[var])

def plot_coolrad_steps(imin=0, imax=None, n=-10, var='sp', logv=0, key='Last',
  func='plaw', ax_in=None, refLF='mid', onlyInt=False, **kwargs):
  '''
  Plots the radiation from the cooling steps of a given cell between imin and imax
  '''

  if not ax_in:
    fig, ax = plt.subplots()
  else:
    ax = ax_in
  
  xlabel, ylabel = '', '$\\nu F_{\\nu}/\\nu_0F_0$'
  logx, logy = False, False

  nuobs, Tobs, env = radEnv(key, True)
  tnu = nuobs/env.nu0
  Tb = (Tobs-env.Ts)/env.T0
  dk = get_gmasdata_cell(n, key)
  
  dk = dk.loc[dk['gmax_']>0.]
  Ni = len(dk)
  steps = [i for i in range(imin, Ni-1)]
  if imax and (imax < Ni):
    steps = steps[:imax]
  regimes = []

  if var == 'sp':
    nuFnu_mono = np.zeros(nuobs.shape)
    x = tnu
  elif var == 'lc':
    nuFnu_mono = np.zeros(Tobs.shape)
    x = Tb

  for j in steps:
    nuFnu_step, reg = plot_coolrad_step(j, n, var, logv, key=key,
      func=func, ax_in=ax, refLF=refLF, onlyInt=onlyInt, lw=.9, **kwargs)
    if reg and (reg not in regimes):
      regimes.append(reg)
    if reg == 'monoE':
      plt.autoscale(enable=False)
      nuFnu_mono += nuFnu_step

  ax.plot(x, nuFnu_mono, c='darkgreen', **kwargs)
  
  if logx: ax.set_xscale('log')
  if logy: ax.set_yscale('log')

  if not ax_in:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    dummy_col = [ax.plot([],[],c=regcol[reg])[0] for reg in regimes]
    ax.legend(dummy_col, regimes, loc='lower right')
    fig.tight_layout()
  else:
    return regimes


def plot_coolrad_step(istep, n=-10, var='sp', logv=0, key='Last',
  func='', critfreq=False, ax_in=None, refLF='mid', onlyInt=False, **kwargs):
  '''
  Plots the radiation from one cooling step of a given cell
  '''

  if not ax_in:
    fig, ax = plt.subplots()
  else:
    ax = ax_in
  
  xlabel, ylabel = '', '$\\nu F_{\\nu}/\\nu_0F_0$'
  logx, logy = False, False

  nuobs, Tobs, env = radEnv(key, True)
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

  if istep >= len(dk):
    print(f"There are only {len(dk)} cooling steps, pick a lower value")
    return ''
  
  cell_0 = dk.iloc[0]
  cell = dk.iloc[istep]
  cell_next = dk.iloc[istep+1]
  if not cell_next['gmax_']:
    return ''
  if func == 'plaw':
    Fnu, reg = get_Fnu_cooling_pureplaws(cell, cell_next, nuobs, Tobs, env)
  else:
    Fnu, reg = get_Fnu_cooling(cell, cell_next, cell_0, nuobs, Tobs, env, refLF=refLF, onlyInt=onlyInt)
  nuFnu = tnu * Fnu/env.F0

  y = nuFnu.flatten()
  ax.plot(x, y, c=regcol[reg], label=istep, **kwargs)

  if logx: ax.set_xscale('log')
  if logy: ax.set_yscale('log')

  if not ax_in:
    ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
  else:
    return y, reg

def compare_cool2vFC(var, logv, key='Last'):
  '''
  Compares observed flux between very fast cooling approx and general case for a cell 
  added forced vFC on cooling method for comparison
  '''
  if var not in ['sp', 'lc']:
    print("First arument must be 'lc' or 'sp'")
    return 0
    
  fig, ax = plt.subplots()
  xlabel, ylabel = '', '$\\nu F_{\\nu}/\\nu_0F_0$'
  logx, logy = False, False
  lstlist = ['-', '-.', ':']
  methods = ['cooling', 'cool+vFC', 'thin shell']
  dummy_lst = [ax.plot([], [], c='k', ls=l)[0] for l in lstlist]
  clist = ['r', 'b', 'k']
  dummy_col = [ax.plot([], [], ls='-', c=col)[0] for col in clist]

  nuobs, Tobs, env = radEnv(key, True)
  tnu = nuobs/env.nu0
  Tb = (Tobs-env.Ts)/env.T0

  path_cl = GAMMA_dir + f'/results/{key}/coolingObs_'
  path_fc = GAMMA_dir + f'/results/{key}/coolingObsvFC_'
  path_ts = GAMMA_dir + f'/results/{key}/thinshellObs_'
  paths = [path_cl, path_fc, path_ts]

  for path, l in zip(paths, lstlist):
    nuFnu_RS, nuFnu_FS = np.load(path+'RS.npy'), np.load(path+'FS.npy')
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

    ax.plot(x, data_RS, c='r', ls=l)
    ax.plot(x, data_FS, c='b', ls=l)
    ax.plot(x, data_tot, c='k', ls=l)
  
  if logx: ax.set_xscale('log')
  if logy: ax.set_yscale('log')
  lstleg = ax.legend(dummy_lst, methods, loc='lower right')
  colleg = ax.legend(dummy_col, ['RS', 'FS', 'RS+FS'], loc='upper left')
  ax.add_artist(lstleg)
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  fig.tight_layout()



def compare_cool2vFC_cell(n, var, logv, key='Last',
  withsteps=False, testing=False, refLF='mid'):
  '''
  Compares observed flux between very fast cooling approx and general case for a cell
  '''

  fig, ax = plt.subplots()
  xlabel, ylabel = '', '$\\nu F_{\\nu}/\\nu_0F_0$'
  logx, logy = False, False

  lstlist = ['-.', '-', '--']
  #lstnames = ['thinshell', 'cooling', 'p.law']
  lstnames = ['thinshell', 'cool + monoE', 'cool. only']

  dummy_lst = [ax.plot([], [], c='k', ls=l)[0] for l in lstlist]
  col = 'r' if n < 0 else 'b'

  nuobs, Tobs, env = radEnv(key, True)
  tnu = nuobs/env.nu0
  Tb = (Tobs-env.Ts)/env.T0

  dk = get_gmasdata_cell(n, key)
  if testing: dk = test_constantHydro(n, key)

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
  
  cell_0 = dk.iloc[0]

  nuFnu_ts = tnu * get_Fnu_thinshell(cell_0, nuobs, Tobs, env, func='plaw')/env.F0
  #nuFnu_cool = cell2rad_integ(dk, nuobs, Tobs, env)
  nuFnu_cool = cell2rad(dk, nuobs, Tobs, env, mode='cool', refLF=refLF, onlyInt=False)
  nuFnu_integ = cell2rad(dk, nuobs, Tobs, env, mode='cool', refLF=refLF, onlyInt=True)
  y_arr = np.array([nuFnu_ts.flatten(), nuFnu_cool.flatten(), nuFnu_integ.flatten()])#, nuFnu_plaw.flatten()
  
  for y, ls in zip(y_arr, lstlist):
    ax.plot(x, y, ls=ls, c=col)

  if logx: ax.set_xscale('log')
  if logy: ax.set_yscale('log')
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  lstleg = ax.legend(dummy_lst, lstnames)

  if withsteps:
    #plot_coolrad_steps(0, None, n, var, logv, ax_in=ax, alpha=.7, ls='-.')
    regimes = plot_coolrad_steps(0, None, n, var, logv, key=key, func='', ax_in=ax, alpha=.7, onlyInt=False)
    plot_coolrad_steps(0, None, n, var, logv, key=key, func='', ax_in=ax, alpha=.7, onlyInt=True)
    dummy_col = [ax.plot([],[],c=regcol[reg])[0] for reg in regimes]
    dummy_col.append(ax.plot([],[],c='darkgreen')[0])
    ax.legend(dummy_col, regimes+['$\\Sigma$monoE'], loc='lower right')
    ax.add_artist(lstleg)

  fig.tight_layout()

def compare_gmasmethod(n, key='Last', xscale='t', logx=False, logy=True, **kwargs):
  '''
  Compares the gma min and gma max of cell n
  plot_shockedCell but for gmin and gmax
  '''

  env = MyEnv(key)
  sh, Nsh, clist = ('RS', env.Nsh4, ['r', 'darkred']) if n<0 else ('FS', env.Nsh1, ['b', 'darkblue'])
  path = GAMMA_dir + f'/results/{key}/gmas_{sh}.csv'
  data = pd.read_csv(path)
  Nc = env.Next+Nsh
  k = Nc+n
  dk = data.loc[data['i']==k]
  fig, ax = plt.subplots()
  lstlist = ['-', '-.', ':']
  dummy_lst = [ax.plot([], [], c='k', ls=l)[0] for l in lstlist]
  dummy_col = [ax.plot([], [], c=c, ls='-')[0] for c in clist]
  cool1 = dk.loc[dk['Sd']<1.].iloc[0]
  gmin_err = (cool1['gmin_']-cool1['gmin'])/cool1['gmin']
  gmax_err = (cool1['gmax_']-cool1['gmax'])/cool1['gmax']
  print(f"Err on gmin = {gmin_err:.3f}, err on gmax = {gmax_err:.3f}")
  
  for s, l in zip(['_', ''], lstlist):
    gm_th, gM_th = (env.gma_m, env.gma_max) if n<0 else (env.gma_mFS, env.gma_maxFS)

    it, t, r, gm, gM = dk[['it', 't', 'x', 'gmin'+s, 'gmax'+s]].to_numpy().transpose()
    x = it
    xlabel = 'it'
    if xscale == 'r':
      xlabel = '$r/R_0$'
      x = r*c_/env.R0
    elif xscale == 't':
      xlabel = '$t/t_0$'
      x = t/env.t0 + 1.
    elif xscale == 'T':
      xlabel = '$\\bar{T}$'
      T = (1+env.z)*(t + env.t0 - r)
      x = (T-T[0])/env.T0
    for val, valth, c in zip([gm, gM], [gm_th, gM_th], clist):
      ax.plot(x, val, c=c, ls=l)
      #ax.scatter(x, val, c='k', marker='x')
      ax.axhline(valth, c=c, ls=':')
  
  if logx:
    ax.set_xscale('log')
  if logy:
    ax.set_yscale('log')
  ax.set_xlabel(xlabel)
  ax.set_ylabel("$\\gamma_{\\rm e}$")
  names = ['us', 'GAMMA', 'R24b']
  varnames = [f'$\\gamma_{{\\rm min,{sh}}}$', f'$\\gamma_{{\\rm max,{sh}}}$']
  ax.legend(dummy_lst+dummy_col, names+varnames)
  plt.tight_layout()

def compare_vFCmFC(key, var='sp', logv=0):
  '''
  Figure comparing flux for both shocks in vFC to FS in mFC and RS in vFC
  '''
  fig, ax = plt.subplots()
  for mfc, l in zip([None, 'FS'], ['-', '-.']):
    plot_emission(var, logv, key, mFC=mfc, ax_in=ax, ls=l)
  ax.set_xlabel("$\\tilde{\\nu}$")
  ax.set_ylabel('$\\nu F_{\\nu}/\\nu_0F_0$')

def fig_facnu(ax_in=None):
  '''
  Figure of nu_0RS/nu_0FS across the parameter space
  '''
  if ax_in:
    ax=ax_in
  else:
    fig, ax = plt.subplots(layout='compressed')
  
  au_arr = np.linspace(1,5,50)[1:]
  chi_arr = np.linspace(0.5,2,150)
  X, Y = np.meshgrid(au_arr, chi_arr)
  facnu_arr = np.zeros((len(chi_arr), len(au_arr)))
  for j, au in enumerate(au_arr):
    for i, chi in enumerate(chi_arr):
      loc = get_localEnv(au, chi)
      facnu_arr[i][j] += 1/loc.fac_nu
  Z = facnu_arr
  CS = ax.contourf(X, Y, Z,
    origin='lower', locator=ticker.LogLocator(), cmap='plasma')
  CB = plt.colorbar(CS)
  CB.ax.set_title('$\\tilde{\\nu}^{-1}$', fontsize=12)
  ax.grid(True, which='both')

  if not ax_in:
    ax.set_ylabel('$\\chi$', rotation='horizontal')
    ax.set_xlabel('$a_{\\rm u}$')

def fig_midslope(ax_in=None, FSmFC=False):
  '''
  Figure of expected nuFnu slope above the low energy spectral break
  '''

  if ax_in:
    ax = ax_in
  else:
    fig, ax = plt.subplots()

  au_arr = np.linspace(1,5,50)[1:]
  chi_arr = np.linspace(0.5,2,150)
  X, Y = np.meshgrid(au_arr, chi_arr)
  slope_arr = np.zeros((len(chi_arr), len(au_arr)))
  title = ''
  for j, au in enumerate(au_arr):
    for i, chi in enumerate(chi_arr):
      if FSmFC:
        title = 'FS in marg. FC'
        slope = get_midslope(au, chi, 1, [1, 0.5])
      else:
        slope = get_midslope(au, chi, 1)
      slope_arr[i][j] += slope

  Z = slope_arr
  levels = np.linspace(-0.08,0.48,8)
  CS = ax.contourf(X, Y, Z, levels=levels, origin='lower', cmap='plasma')
  CB = plt.colorbar(CS, ax=ax)
  CB.ax.set_title('$1+b_{\\rm mid}$', fontsize=12)
  ax.set_title(title, fontsize=16)
  ax.grid(True, which='both')

  if not ax_in:
    ax.ylabel('$\\chi$', rotation='horizontal')
    ax.set_xlabel('$a_{\\rm u}$')

def fig_diversity():
  '''
  fig_facnu and fig_midslope
  '''

  fig, axs = plt.subplots(2, 1, layout='compressed')
  axs[-1].set_xlabel('$a_{\\rm u}$')
  fig.supylabel('$\\chi$', rotation='horizontal')
  fig_midslope(ax_in=axs[0])
  fig_facnu(ax_in=axs[1])

def get_localEnv(au, chi, u1=100, t4=0.1, toff=0.1):
  '''
  Params must be in same order as parlist
  '''
  z = 1
  params = [u1, au*u1, chi*t4, t4, toff]
  parlist = ['u1', 'u4', 't1', 't4', 'toff']
  loc = localEnv(params, parlist)
  loc.au = au
  loc.chi = chi
  loc.f = (chi/au)**2
  sqf = chi/au
  au2 = au*au

  loc.G1 = derive_Lorentz_from_proper(u1)
  loc.v1 = derive_velocity_from_proper(loc.u1)
  loc.G4 = derive_Lorentz_from_proper(loc.u4)
  loc.v4 = derive_velocity_from_proper(loc.u4)
  loc.R0c = loc.v1*loc.v4*toff/(loc.v4-loc.v1)

  G_denom = 2*np.sqrt((au+sqf)*(sqf*au2+au))
  loc.G21 = (2*au+sqf*(1+au2))/G_denom
  loc.G34 = (au2+2*sqf*au+1)/G_denom
  loc.v21 = derive_velocity(loc.G21)
  loc.v34 = derive_velocity(loc.G34)
  loc.u0 = loc.G21*loc.G1*(loc.v21+loc.v1)
  loc.G0 = derive_Lorentz_from_proper(loc.u0)
  loc.v0 = derive_velocity_from_proper(loc.u0)

  loc.vRS = (loc.v4-4*loc.G34*(loc.u0/loc.G4))/(1-4*loc.G34*(loc.G0/loc.G4))
  loc.tRS = loc.v4*loc.t4/(loc.v4-loc.vRS)
  loc.GRS = derive_Lorentz(loc.vRS)
  loc.gRS = loc.G0/loc.GRS
  loc.vFS = ((loc.u1/loc.G0)-4*loc.G21*loc.v0)/((loc.G1/loc.G0)-4*loc.G21)
  loc.tFS = loc.v1*loc.t1/(loc.vFS-loc.v1)
  loc.GFS = derive_Lorentz(loc.vFS)
  loc.gFS = loc.G0/loc.GFS

  loc.T0 = (1+z)*loc.R0c*(1-loc.vRS)/loc.vRS
  loc.RfRS0 = 1 + loc.tRS*loc.vRS/loc.R0c
  loc.T0FS = (1+z)*loc.R0c*(1-loc.vFS)/loc.vFS
  loc.RfFS0 = 1 + loc.tFS*loc.vFS/loc.R0c

  ratio_p = (loc.G34*(loc.G21+1))/(loc.G21*(loc.G34+1))
  ratio_m = ((loc.G34-1)/(loc.G21-1))
  loc.fac_nu = np.sqrt(ratio_p) * ratio_m**2
  loc.fac_F = (loc.v34/loc.v21) * np.sqrt(ratio_p) * ratio_m**-2
  loc.fac_nuF = loc.fac_nu*loc.fac_F
  return loc

def get_pksTheory(T, env, eps_rads=[1., 1.]):
  '''
  Gives nu and nuFnu pks for RS and FS at observed time T (T=0 at Ts)
  in unit nu0 and nu0F0 (normalized at RS)
  '''
  nupks, nuFnupks = [], []
  for sh, eps in zip(['RS', 'FS'], eps_rads):
    tT, tTf, tTeff1, tTeff1f, tTeff2 = get_normsT(T, env, sh)
    if tT <= tTf:
      nupk = 1/tT
      nuFnupk = eps*(1-tTeff1**-3)
    else:
      nupk = 1/(tTf*tTeff2)
      nuFnupk = eps*(1-tTeff1f**-3)*tTeff2**-3
    nupks.append(nupk)
    nuFnupks.append(nuFnupk)

  nupks = [nupks[0], nupks[1]/env.fac_nu]
  nuFnupks = [nuFnupks[0], nuFnupks[1]/(env.fac_nuF)]
  
  return nupks, nuFnupks

def get_pkstotTheory(T, env, eps_rads=[1., 1.], b1RS=-0.5, b2FS=-1.25):
  '''
  peaks of total nuFnu
  '''
  nupks, nuFnupks = get_pksTheory(T, env, eps_rads)
  x = nupks[1]/nupks[0]
  nuFnupks_tot = [nuFnupks[0] + nuFnupks[1]*Band_func_old(1/x, b2=b2FS)/x,
    nuFnupks[1]+nuFnupks[0]*x*Band_func_old(x, b1=b1RS)]
  
  return nupks, nuFnupks_tot

def get_midslope(au, chi, T_inT0, eps_rads=[1., 1.], b1RS=-0.5, b2FS=-1.25):
  '''
  Derives the slope of the middle part of nuFnu spectrum at T (in unit T0) in hybrid case
  '''

  loc = get_localEnv(au, chi)
  T = T_inT0*loc.T0
  nupks, nuFnupks_tot = get_pkstotTheory(T, loc, eps_rads, b1RS, b2FS)
  
  slope = logslope(nupks[1], nuFnupks_tot[1], nupks[0], nuFnupks_tot[0])
  return slope

def get_normsT(T, env, sh='RS'):
  '''
  Normalized times appearing in eqs 8-10 of R24b
  '''
  g2, T0, Rf = (env.gRS**2, env.T0, env.RfRS0) if sh == 'RS' else (env.gFS**2, env.T0FS, env.RfFS0)
  tTf = Rf
  tT = 1+T/T0
  tTeff1 = 1 - g2 + g2*tT
  tTeff1f = 1 - g2 + g2*tTf
  tTeff2 = 1 - g2 + tT/tTf
  return tT, tTf, tTeff1, tTeff1f, tTeff2

# Presentation
# --------------------------------------------------------------------------------------------------
def plot_R24lightcurves(nulist, key='Last'):
  lslist = ['-.', '--', ':'][:len(nulist)]
  fig, ax = plt.subplots()
  names, colors = ['RS', 'FS', 'RS+FS'], ['r', 'b', 'k']
  dummy_col = [plt.plot([],[], c=c, ls='-')[0] for c in colors]
  fig.legend(dummy_col, names)
  ylabel = '$\\nu F_\\nu/\\nu_0 F_0$'
  xlabel = '$\\bar{T}$'
  dummy_lst = []
  for lognu, ls in zip(nulist, lslist):
    plot_R24lightcurve(lognu, key, ax, ls=ls)
    dummy_lst.append(plt.plot([],[], ls=ls, color='k')[0])
  lstlegend = ax.legend(dummy_lst, [f'{lognu:d}' for lognu in nulist],
    title='$\\log_{10}(\\nu/\\nu_0)$', loc='center right')
  ax.add_artist(lstlegend)
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)

def plot_R24spectras(Tlist, key='Last'):
  lslist = ['-.', '--', ':'][:len(Tlist)]
  fig, ax = plt.subplots()
  names, colors = ['RS', 'FS', 'RS+FS'], ['r', 'b', 'k']
  dummy_col = [plt.plot([],[], c=c, ls='-')[0] for c in colors]
  #fig.legend(dummy_col, names)
  ylabel = '$\\nu F_\\nu/\\nu_0 F_0$'
  xlabel = '$\\nu/\\nu_0$'
  dummy_lst = []
  for logT, ls in zip(Tlist, lslist):
    plot_R24spectrum(logT, key, ax, ls=ls)
    dummy_lst.append(plt.plot([],[], ls=ls, color='k')[0])
  lstlegend = ax.legend(dummy_lst, [f'{logT:d}' for logT in Tlist],
    title='$\\log_{10}(\\bar{T})$', loc='upper left')
  ax.add_artist(lstlegend)
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)

  
def plot_R24spectrum(logT, key='Last', ax_in=None, **kwargs):
  '''
  Plots instantaneous spectrum at \bar{T} = 10**logT from Minu
  '''
  if not ax_in:
    plt.figure()
    ax = plt.gca()
  else:
    ax = ax_in
  names= ['RS', 'FS', 'RS+FS']
  nuobs, Tobs, env = get_radEnv(key)
  nub = nuobs/env.nu0
  T = np.array([env.T0*10**logT + env.Ts])
  nuFnus = nuFnu_theoric(nuobs, T, env, geometry=env.geometry, func='Band')[:,:,0]
  nuFnus = np.append(nuFnus, [nuFnus.sum(axis=0)], axis=0)

  for sp, name in zip(nuFnus, names):
    ax.loglog(nub, sp, c=cols_dic[name], label=name, **kwargs)
  
  if not ax_in:
    plt.legend()
    plt.xlabel("$\\nu/\\nu_0$")
    plt.ylabel("$\\nu F_{\\nu}/\\nu_0 F_0$")
    plt.title(key+', $\\log\\bar{T}=$'+f'{logT:d}')

def plot_R24lightcurve(lognu, key='Last', ax_in=None, **kwargs):
  '''
  Plot theoretical lc from Minu at chosen power of nu/nu_0 ratio
  '''
  if not ax_in:
    plt.figure()
    ax = plt.gca()
  else:
    ax = ax_in
  names= ['RS', 'FS', 'RS+FS']
  nuobs, Tobs, env = get_radEnv(key)
  nuobs = np.array([10**lognu*env.nu0])
  Tb = (Tobs-env.Ts)/env.T0
  nuFnus = nuFnu_theoric(nuobs, Tobs, env, geometry=env.geometry, func='Band')[:,0,:]
  nuFnus = np.append(nuFnus, [nuFnus.sum(axis=0)], axis=0)

  for lc, name in zip(nuFnus, names):
    ax.plot(Tb, lc, c=cols_dic[name], label=name, **kwargs)
  
  if not ax_in:
    plt.legend()
    plt.xlabel("$\\bar{T}$")
    plt.ylabel("$\\nu F_{\\nu}/\\nu_0 F_0$")
    plt.title(key+', $\\log\\tilde{\\nu}=$'+f'{lognu:d}')

# Article
# --------------------------------------------------------------------------------------------------
def timeSeries(key, varlist=['E', 'M', 'D', 'u', 'ShSt'], theory=True):
  env = MyEnv(key)
  if func(varlist)==str: varlist = [varlist]
  for var in varlist:
    ax_timeseries(var)
    plt.title('')
    plt.savefig('./figures/series_' + env.runname + '_' + var + '.png')
    plt.close()

#timeSeries('Last')

def snapSeries(key, theory=False):
  '''
  Creates series of snapshots of run key around a reference time
  '''
  env = MyEnv(key)
  lastmult = 1.6 if env.geometry=='spherical' else 1.5
  t_snaps = [env.t0, lastmult*env.t0]
  names = ['0', '1', '2']
  its, times, _ = get_runTimes(key)
  it_snaps = [0]
  for t in t_snaps:
    i = find_closest(times, t)
    it_snaps.append(its[i])
  for it, name, leg in zip(it_snaps, names, (False, True, False)):
    prim_snapshot(it, key, theory, xscaling='Rcd', itoff=True, legend=leg)
    figname = './figures/snap_' + env.runname + '_' + name + '.png' 
    plt.savefig(figname)
    plt.close()

# Work document
# --------------------------------------------------------------------------------------------------
def broken_power_law(break_positions, plaw_indices, norm_index=0, logbreaks=False):
  '''
  Broken power-law with given breaks positions and power-law indices
  set logbreaks=True if you give the break_positions as powers of 10
  Normalized (x=1, y=1) at norm_index

  Parameters:
      break_positions (list): List of break positions on the x-axis.
      power_law_indices (list): List of indices indicating the power laws between breaks. 
  '''

  if len(plaw_indices) != len(break_positions) - 1:
    print("You must give exactly 1 less index than there are breaks!")
    return 0.

  break_positions = np.array(break_positions)
  if not logbreaks:
    break_positions = np.log10(break_positions)
  #break_positions -= break_positions[norm_index]
  x  = np.logspace(break_positions[0], break_positions[-1], 20*len(break_positions))
  y  = np.zeros(x.shape) 
  y0 = 1.
  yn = 1.
  for i, p in enumerate(plaw_indices):
    if i == norm_index:
      yn = y0
    xmin, xmax = 10**break_positions[[i, i+1]]
    segment = (x>=xmin) & (x<=xmax)
    y[segment] = y0*(x[segment]/xmin)**p
    y0 = y0*(xmax/xmin)**p
  y /= yn
  return x, y

def plot_broken_power_law(break_positions, power_law_indices, break_names, plaw_names, norm_name='X',
  xvar='x', yvar='y', title='', norm_index=0, logbreaks=False):
  """
  Plots broken power laws with given break positions, names, and power law indices.
    
  Parameters:
    break_positions (list): List of break positions on the x-axis.
    break_names (list): List of names corresponding to each break position.
    power_law_indices (list): List of indices indicating the power laws between breaks.
  """
  # Create figure and axis
  fig, ax = plt.subplots()

  # Create dataset
  x, y = broken_power_law(break_positions, power_law_indices, norm_index, logbreaks)
  
  # Plot
  ax.loglog(x, y)

  # Decorators
  c = ax.lines[-1].get_color()
  plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
  plt.tick_params(
    axis='y',          # changes apply to the y-axis
    which='both',      # both major and minor ticks are affected
    left=False,        # ticks along the left edge are off
    right=False,       # ticks along the right edge are off
    labelleft=False)   # labels along the left edge are off
  ax.set_xlabel(xvar)
  ax.xaxis.set_label_coords(1.02, -0.02)
  ax.set_ylabel(yvar, rotation='horizontal')
  ax.yaxis.set_label_coords(-0.02, 1.02)
  
  ax.hlines(y=1, xmin=0., xmax=break_positions[norm_index], colors='b', ls='--', lw=.8)
  ax.text(x=-0.01, y=1., s=norm_name, transform=ax.get_yaxis_transform(), c='b', va='center', ha='right')
  y_brs = [y[np.searchsorted(x, x_br)] for x_br in break_positions]
  ax.vlines(break_positions, ymin=0., ymax=y_brs, colors='k', ls='--', lw=.8)
  for i, name in enumerate(break_names):
    if i != len(break_names) - 1:
      p = power_law_indices[i]
      pstr = plaw_names[i]
      xmid = 10**(0.5*(np.log10(break_positions[i])+np.log10(break_positions[i+1])))
      ymid = 10**(0.5*(np.log10(y_brs[i])+np.log10(y_brs[i+1])))
      ha = 'right' if p > 0. else 'left'
      ax.text(x=xmid, y=ymid*1.05, s=pstr, c=c, ha=ha, va='bottom') 
    ax.text(x=break_positions[i], y=-0.05, s=name, c=c, transform=ax.get_xaxis_transform(), ha='center')
  plt.title(title)
  plt.tight_layout()

def create_fig_coolregimes(regime):
  '''
  Create the fig corresponding to the wanted emission/cooling regime
  '''
  p = 2.5
  if regime not in ['SC', 'FC', 'vFC', 'CD']:
    print("Input needs to be an implemented regime")
    return 0.
  
  figtitle = './figures/plaw_' + regime + '.png'
  
  if regime == 'SC':
    break_positions = [1, 10, 100, 1000]
    break_names = ["$\\nu'_{\\rm B}$", "$\\nu'_{\\rm m}$",
      "$\\nu'_{\\rm c}$", "$\\nu'_{\\rm M}$"]
    power_law_indices = [1./3, (1-p)/2, -p/2]
    plaw_names = ["1/3", "(1-p)/2", "-p/2"]
    title = 'Slow cooling'

  elif regime == 'FC':
    break_positions = [1, 10, 100, 1000]
    break_names = ["$\\nu'_{\\rm B}$", "$\\nu'{\\rm c}$",
      "$\\nu'_{\\rm m}$", "$\\nu'_{\\rm M}$"]
    power_law_indices = [1./3, -1/2, -p/2]
    plaw_names = ['1/3', '-1/2', '-p/2']
    title = 'Fast cooling'
  
  elif regime == 'vFC':
    break_positions = [1, 100, 1000]
    break_names = ["$\\nu'_{\\rm B}$", "$\\nu'_{\\rm m}$", "$\\nu'_{\\rm M}$"]
    power_law_indices = [-1/2, -p/2]
    plaw_names = ['-1/2', '-p/2']
    title = 'Very fast cooling'

  elif regime == 'CD':
    break_positions = [1, 10, 15]
    break_names = ["$\\nu'_{\\rm B}$", "$\\nu'_{\\rm m}$", "$\\nu'_{\\rm M}$"]
    power_law_indices = [1/3, -1/2]
    plaw_names = ['1/3', '-1/2']
    title = 'Mono energetic'

  norm_index = break_names.index("$\\nu'_{\\rm m}$")
  plot_broken_power_law(break_positions, power_law_indices, break_names, plaw_names,
      norm_name="$\\Delta E'_{\\nu'_m}$", xvar="$\\nu'$", yvar="$\\Delta E'_{\\nu'}$", title=title, norm_index=norm_index) 
  plt.grid(visible=False)
  plt.savefig(figtitle)
  plt.show()