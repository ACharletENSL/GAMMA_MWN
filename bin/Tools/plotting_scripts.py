# -*- coding: utf-8 -*-
# @Author: acharlet

'''
This files contains plotting functions to plot GAMMA outputs and analyzed datas
'''

# Imports
# --------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from matplotlib import ticker
from environment import *
from run_analysis import *
from data_IO import *
from phys_functions_shells import *

# matplotlib options
# --------------------------------------------------------------------------------------------------
plt.rc('font', family='serif', size=12)
plt.rc('xtick', labelsize=12) 
plt.rc('ytick', labelsize=12) 
plt.rc('legend', fontsize=12) 
plt.rcParams['savefig.dpi'] = 200

formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) 
formatter.set_powerlimits((-1,1)) 

nolog_vars = ['trac', 'Sd', 'gmin', 'gmax', 'zone']
var_label = {'R':'$r$ (cm)', 'v':'$\\beta$', 'u':'$\\gamma\\beta$',
  'V':'$V$ (cm$^3$)', 'Nc':'$N_{cells}$', 'ShSt':'$\\Gamma_{ud}-1$',
  'M':'$M$', 'Ek':'$E_k$', 'Ei':'E_{int}',
  'Rct':'$(r - ct)/R_0$ (cm)', 'Rcd':'$r - r_{cd}$ (cm)',
  'vcd':"$\\beta - \\beta_{cd}$", 'epsth':'$\\epsilon_{th}$'}

# Compare theory with models
def compare_snapwiththeory(it, key='Last'):
  '''
  Returns the relative difference of the variables in the shocked regions
  '''
  physpath  = get_physfile(key)
  env = MyEnv(physpath)
  df, t, dt = openData_withtime(key, it)
  rho2, rho3, u_sh, p_sh, lfac21, lfac34 = shells_shockedvars(env.Ek1, env.u1, env.D01, env.Ek4, env.u4, env.D04, env.R0)
  rho0      = getattr(env, env.rhoNorm)
  rho2     /= rho0
  rho3     /= rho0
  p_sh     /= rho0*c_**2
  rho2_num  = df_get_mean(df, 'rho', 2)
  rho3_num  = df_get_mean(df, 'rho', 3)
  u_num     = df_get_shocked_u(df)
  p_num     = df_get_shocked_p(df)

  lfac_num  = derive_Lorentz_from_proper(u_num)
  lfac1     = derive_Lorentz_from_proper(env.u1)
  lfac4     = derive_Lorentz_from_proper(env.u4)
  lfac21_num= lfac1*lfac_num - env.u1*u_num
  lfac34_num= lfac4*lfac_num - env.u4*u_num

  diffrho2 = reldiff(rho2_num, rho2)
  diffrho3 = reldiff(rho3_num, rho3)
  diffu    = reldiff(u_num, u_sh)
  diffp    = reldiff(p_num, p_sh)
  diffL21  = reldiff(lfac21_num, lfac21)
  diffL34  = reldiff(lfac34_num, lfac34)

  return diffrho2, diffrho3, diffu, diffp, diffL21, diffL34

# Create figure 
# --------------------------------------------------------------------------------------------------
def snapshot_withtheory(it, key='Last'):
  '''
  Plots the primitive variables (with lfac*v instead of v)
  at chosen iteration in a single figure, compared to the theoretical values
  '''
  physpath = get_physfile(key)
  env = MyEnv(physpath)
  df, t, dt = openData_withtime(key, it)
  varlist = ['rho', 'u', 'p']
  f, axes = plt.subplots(3, 1, sharex=True, figsize=(6,6), layout='constrained')

  for var, k, ax in zip(varlist, range(3), axes):
    title, legend = plot_snapshot_1D(var, it, key, 'CGS', 'CGS', ax)
    if k!=2: #only have xlabel on plot at bottom
      ax.set_xlabel('')

  r = axes[-1].get_lines()[-1].get_xdata()
  for var, ax in zip(varlist, axes):
    z = get_analytical(var, t, r, env)
    ax.plot(r, z, 'k')

  f.suptitle(title)
  f.add_artist(legend)
  plt.tight_layout()


def plot_mono(var, it, key='Last', xscale='CGS', yscale='code', slope=False, fig=None):
  '''
  Plots a single var in a single figure with title, labels, etc.
  '''

  if fig:
    ax = fig.gca()
  else:
    fig = plt.figure()
    ax = plt.gca()
  title, legend = plot_snapshot_1D(var, it, key, xscale, yscale, ax, col='Zone', slope=None)
  fig.suptitle(title)
  fig.legend(legend)
  fig.tight_layout()

# Several lines in one figure
def plot_primvar(it, key='Last', xscale='CGS', yscale='code'):
  '''
  Plots the primitive variables - with lfac*v instead of v - at chosen iteration in a single figure
  '''

  plot_multi(["rho", "u", "p"], it, key, yscale)

def plot_consvar(it, key='Last', xscale='CGS', yscale='code'):
  '''
  Plots the conservative variables at chosen iteration in a single figure
  '''
  plot_multi(["D", "sx", "tau"], it, key, xscale, yscale)

def plot_energy(it, key='Last'):
  '''
  Plot energy content
  '''
  varlist = ["Ei", "Ek"]
  plot_comparison(varlist, it, key, col=None)

def plot_comparison(varlist, it, key='Last', xscale='CGS', yscale='code', col=None):
  '''
  Plots variables among a list in a single figure
  '''

  plt.figure()
  ax = plt.gca()

  for var in varlist:
    title, legend = plot_snapshot_1D(var, it, key, xscale, yscale, ax, col, slope=None)
  
  plt.suptitle(title)
  plt.legend(ax.scatter.legend_elements())
  plt.tight_layout()


def plot_multi(varlist, it, key='Last', xscale='CGS', yscale='code'):
  '''
  Plots variables among a list in a single figure with subplots
  '''

  Nk = len(varlist)
  f, axes = plt.subplots(Nk, 1, sharex=True, figsize=(6,2*Nk))

  for var, k, ax in zip(varlist, range(Nk), axes):
    title, legend = plot_snapshot_1D(var, it, key, xscale, yscale, ax)
    if k!=Nk-1: #only have xlabel on plot at bottom
      ax.set_xlabel('')
  
  f.suptitle(title)
  f.add_artist(legend)
  plt.tight_layout()


def plot_timeseries(var_keys, key='Last', tscaling=None, slope=False, fig=False):
  '''
  Create plot of time series data
  '''

  if fig:
    ax = fig.gca()
  else:
    fig = plt.figure()
    ax = plt.gca()
  
  if var_keys == 'E':
    energies = ['Ek', 'Ei']
    linestyles = ['--', '-.']
    for i, var in enumerate(energies):
      title = plot_time(var, key, tscaling, slope, ax, ls=linestyles[i])
  else:
    title = plot_time(var_keys, key, tscaling, slope, ax)

  plt.legend()
  fig.suptitle(title)

def plot_itseries(var, key='Last', fig=False):
  if fig:
    ax = fig.gca()
  else:
    fig = plt.figure()
    ax = plt.gca()

  title = plot_it(var, key, ax)
  fig.suptitle(title)

# Create plot in ax to insert in figure and generate title
# --------------------------------------------------------------------------------------------------
def plot_snapshot_1D(var, it, key, xscale='CGS', yscale='code', ax=None, col='Zone', slope=None, line=True, **kwargs):
  '''
  Creates ax object to be insered in plot. Scales data
  '''
  physpath = get_physfile(key)
  env = MyEnv(physpath)
  df, t, dt = openData_withtime(key, it)
  n = df["zone"]
  x, z, xlabel, zlabel = get_scaledvar_snapshot(var, df, env, xscale, yscale, slope)
  
  if env.mode == 'shells':
    rc0 = env.R0
    r_scale_str = 'R_0'
    ncd = 4
    t_scale = env.t0
    t_scale_str = 't_0'
  elif env.mode == 'PWN':
    rc0 = env.R_b
    ncd = 2
    t_scale = env.t_0
    t_scale_str = 't_0'
  
  rc   = (get_radius_zone(df, ncd)*c_ - rc0)
  tsim = t/t_scale
  t_str = reformat_scientific(f"{dt:.2e}")
  rc_str= reformat_scientific(f"{rc:.2e}")
  title = f"it {it}, dt = {dt:.3f} s" #$\\Delta t/{t_scale_str} = {t_str}$" #, $\\Delta R_{{CD}}/{r_scale_str} = {rc_str}$"
  
  if ax is None:
    plt.figure()
    ax = plt.gca()
  
  ax.set_xscale('log')
  ax.set_yscale('log')
  if slope:
    ax.set_yscale('linear')
    ax.set_ylim(-5, 2)
  else:
    if var in nolog_vars:
      ax.set_yscale('linear')

  ax.set_ylabel(zlabel)
  ax.set_xlabel(xlabel)

  if line:
    ax.plot(x, z, 'k', zorder=1)
  if col == 'Zone' or var == 'zone':
    if env.mode == 'PWN':
      n += 1
    scatter = ax.scatter(x, z, c=n, lw=1, zorder=2, cmap='Paired', **kwargs)
  else:
    scatter = ax.scatter(x, z, lw=1, zorder=2, **kwargs)
  legend1 = ax.legend(*scatter.legend_elements())
  ax.legend('', frameon=False)

  return title, legend1

def plot_it(var, key, ax=None, **kwargs):
  '''
  Plot according to run iterations
  '''

  if ax is None:
    plt.figure()
    ax = plt.gca()

  physpath = get_physfile(key)
  env = MyEnv(physpath)
  title = ""

  it, vars, xlabel, ylabel, var_legends = get_scaledvar_it(var, key, env)
  Nv = vars.shape[0]
  for n in range(Nv):
    ax.plot(it, vars[n], c=plt.cm.Paired(n), label=var_legends[n], **kwargs)
  if var not in ['Nc', 'v', 'R', 'Rcd', 'Rct', 'u']:
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(formatter)

  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)

  return title

def plot_time(var, key, tscaling='t0', slope=False, ax=None, crosstimes=True, **kwargs):
  '''
  Create time series plot to be inserted into a figure
  '''
  
  if ax is None:
    plt.figure()
    ax = plt.gca()

  physpath = get_physfile(key)
  env = MyEnv(physpath)
  title = ""

  time, vars, tlabel, ylabel, tscale, var_legends = get_scaledvar_timeseries(var, key, env, tscaling, slope)
  if crosstimes:
    tscaleRS = env.tRS/tscale
    ax.axvline(tscaleRS, color='r')
    ax.text(tscaleRS, 1.05, '$t_{RS}$', color='r', transform=ax.get_xaxis_transform(),
      ha='center', va='top')
    tscaleFS = env.tFS/tscale
    ax.axvline(tscaleFS, color='r')
    ax.text(tscaleFS, 1.05, '$t_{FS}$', color='r', transform=ax.get_xaxis_transform(),
      ha='center', va='top')

  # add theoretical values
  if var == 'u':
    ax.axhline(env.u, color='k', ls='--', lw=.6)
    ax.text(1.02, env.u, '$u_{th}$', color='k', transform=ax.get_yaxis_transform(),
      ha='left', va='center')
  elif var == 'ShSt':
    ax.axhline(env.lfac34-1., color='k', ls='--', lw=.6)
    ax.axhline(env.lfac21-1., color='k', ls='--', lw=.6)
    ax.text(1.02, env.lfac34-1., '$\\Gamma_{34}-1$', color='k', transform=ax.get_yaxis_transform(),
      ha='left', va='center')
    ax.text(1.02, env.lfac21-1., '$\\Gamma_{21}-1$', color='k', transform=ax.get_yaxis_transform(),
      ha='left', va='center')
  elif var == 'v':
    ax.axhline(env.betaRS, color='k', ls='--', lw=.6)
    ax.axhline(env.betaFS, color='k', ls='--', lw=.6)
    ax.text(1.02, env.betaRS, '$\\beta_{RS}$', color='k', transform=ax.get_yaxis_transform(),
      ha='left', va='center')
    ax.text(1.02, env.betaFS, '$\\beta_{FS}$', color='k', transform=ax.get_yaxis_transform(),
      ha='left', va='center')
  elif var == 'vcd':
    betacd = derive_velocity_from_proper(env.u)
    ax.axhline(betacd - env.betaRS, color='k', ls='--', lw=.6)
    ax.axhline(env.betaFS - betacd, color='k', ls='--', lw=.6)
    ax.text(1.02, betacd - env.betaRS, '$\\beta_{CD}-\\beta_{RS}$', color='k', transform=ax.get_yaxis_transform(),
      ha='left', va='center')
    ax.text(1.02, env.betaFS - betacd, '$\\beta_{FS}-\\beta_{CD}$', color='k', transform=ax.get_yaxis_transform(),
      ha='left', va='center')


  Nv = vars.shape[0]
  for n in range(Nv):
    if slope:
      ax.plot(time[1:], vars[n][1:], c=plt.cm.Paired(n), label=var_legends[n], **kwargs)
    else:
      ax.plot(time[1:], vars[n][1:], c=plt.cm.Paired(n), label=var_legends[n], **kwargs)
  if var not in ['Nc', 'R', 'Rcd', 'Rct', 'u']:
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(formatter)

  ax.set_xlabel(tlabel)
  ax.set_ylabel(ylabel)

  return title
  

# Get values with scaling and labels for axes
# --------------------------------------------------------------------------------------------------
def get_scaledvar_snapshot(var, df, env, xscale='CGS', yscale='code', slope=None):
  '''
  Return x, y values and relevant labels according to chosen yscale
  Includes: code, CGS, shells
  to add: MWN_sd, MWN_SNR
  '''
  # Legends and labels
  var_exp = {
  "x":"$r$", "dx":"$dr$", "rho":"$\\rho$", "vx":"$\\beta$", "p":"$p$",
  "D":"$\\gamma\\rho$", "sx":"$\\gamma^2\\rho h$", "tau":"$\\tau$",
  "trac":"tracer", "Sd":"", "gmin":"$\\gamma_{min}$", "gmax":"$\\gamma_{max}$", "zone":"",
  "T":"$\\Theta$", "h":"$h$", "lfac":"$\\gamma$", "u":"$\\gamma\\beta$",
  "Ei":"$e$", "Ek":"$e_k$", "dt":"dt", "res":"dr/r"
  }
  units_CGS  = {
  "x":" (cm)", "dx":" (cm)", "rho":" (g cm$^{-3}$)", "vx":"", "p":" (Ba)",
  "D":"", "sx":"", "tau":"", "trac":"", "Sd":"", "gmin":"", "gmax":"",
  "T":"", "h":"", "lfac":"", "u":"", "zone":"", "dt":" (s)", "res":"",
  "Ei":" (erg cm$^{-3}$)", "Ek":" (erg cm$^{-3}$)", "Emass":" (erg cm$^{-3}$)"
  }
  def get_normunits(xnormstr, rhonormsub):
    rhonormsub  = '{' + rhonormsub +'}'
    CGS2norm = {" (s)":" (s)",
    " (cm)":" (ls)" if xnormstr == 'c' else "$/"+xnormstr+"$", " (g cm$^{-3}$)":f"$/\\rho_{rhonormsub}$",
    " (Ba)":f"$/\\rho_{rhonormsub}c^2$", " (erg cm$^{-3}$)":f"$/\\rho_{rhonormsub}c^2$"}
    return {key:(CGS2norm[value] if value else "") for (key,value) in units_CGS.items()}

  def get_varscaling(var, rhoNorm):
    pNorm = rhoNorm*c_**2
    unit  = units_CGS[var]
    var_scale = 1.
    if unit == " (g cm$^{-3}$)":
      var_scale = rhoNorm
    elif unit == " (Ba)" or unit == " (erg cm$^{-3}$)":
      var_scale = pNorm
    else:
      var_scale = 1.
    return var_scale

  units = {}
  if xscale == 'code':
    xlabel = '$r$ (ls)'
    r_scale = 1.
  elif xscale == 'CGS':
    xlabel = '$r$ (cm)'
    r_scale = c_
  elif xscale == 'R0':
    xlabel = '$r/R_0$'
    r_scale = c_/env.R0
  elif xscale == 'Rcd':
    xlabel = '$r/R_{cd}$'
    r_scale = 1./get_radius_zone(df, 2)
  else:
    xlabel = '$r$'
    r_scale = 1.
  if yscale == 'code':
    units = get_normunits('c', 'Norm')
    rho_scale = 1.
  elif yscale == 'CGS':
    units = units_CGS
    rho_scale = getattr(env, env.rhoNorm)
  else:
    if env.mode == 'shells':
      sub = env.rhoNorm[3:]
      units = get_normunits('R_0', sub)
      r_scale, rho_scale = c_/env.R0, getattr(env, env.rhoNorm)
    elif env.mode == 'PWN':
      print("Implement PWN cases")

  x, z  = get_variable(df, var)
  x = x*r_scale
  z = z*get_varscaling(var, rho_scale)
  #xlabel = var_exp['x'] + units['x']
  zlabel = (var_exp[var] + units[var]) if units[var] else var_exp[var]
  if slope:
    logx = np.log10(x)
    logz = np.log10(denoise_data(z))
    z = np.gradient(logz, logx)
    zlabel = "d$\\log(" + zlabel.replace('$', '') + ")/$d$\\log r$"

  return x, z, xlabel, zlabel



def get_scaledvar_timeseries(var, key, env, tscaling='t0', slope=False):
  '''
  Return time and values for chosen variable, as well as their labels
  '''

  if tscaling == 't0':
    tscale = env.t0
    tscaling = '/t_0'
  else:
    tscale = 1.
    tscaling = ''
  df_res = get_timeseries(var, key)
  time = df_res['time'].to_numpy()/tscale
  Nz = get_Nzones(key)
  tlabel = '$t' + tscaling + '$' if tscaling else '$t$ (s)'
  ylabel = var_label[var]
  if var in ['Rct', 'Rcd']:
    varlist = get_varlist('R', Nz, env.mode)
  elif var == 'vcd':
    varlist = get_varlist('v', Nz, env.mode)
  else:
    varlist = get_varlist(var, Nz, env.mode)
  if var == 'Rcd':
    varlist.remove('R_{cd}')
  elif var == 'vcd':
    varlist.remove('v_{cd}')
  elif var == 'epsth':
    varlist = ['$\\epsilon_{th,3}$', '$\\epsilon_{th,2}$']
    df_res[:,1] /= env.Ek4
    df_res[:,2] /= env.Ek2
  
    
  var_legends = ['$' + var_leg + '$' for var_leg in varlist]
  vars = df_res[varlist].to_numpy().transpose()
  if var == 'v':
    vars /= c_
  Nv = vars.shape[0]
  if slope:
    logt = np.log10(time)
    logvars = np.log10(vars)
    vars = np.gradient(logvars, logt, axis=1)
    new_legends = ["d$\\log(" + var_leg.replace('$', '') + ")/$d$\\log r$" for var_leg in var_legends]
    var_legends = new_legends

  return time, vars, tlabel, ylabel, tscale, var_legends

def get_scaledvar_it(var, key, env):
  

  df_res = get_timeseries(var, key)
  it = df_res.index
  Nz = get_Nzones(key)
  xlabel = 'it'
  ylabel = var_label[var]
  if var in ['Rct', 'Rcd']:
    varlist = get_varlist('R', Nz, env.mode)
  else:
    varlist = get_varlist(var, Nz, env.mode)
  if var == 'Rcd':
    varlist.remove('R_{cd}')
  elif var == 'vcd':
    varlist.remove('v_{cd}')
  elif var == 'epsth':
    varlist = ['$\\epsilon_{th,3}$', '$\\epsilon_{th,2}$']
    df_res[:,1] /= env.Ek4
    df_res[:,2] /= env.Ek2
  
    
  var_legends = ['$' + var_leg + '$' for var_leg in varlist]
  vars = df_res[varlist].to_numpy().transpose()
  Nv = vars.shape[0]

  return it, vars, xlabel, ylabel, var_legends

# useful formatting functions
# --------------------------------------------------------------------------------------------------
def reformat_scientific(string):
  '''
  Reformat python scientific notation into LaTeX \times10^{}
  '''
  base, exponent = string.split("e")
  string = f"{base} \\times 10^{{{int(exponent)}}}"
  return string

def reformat_pow10(string):
  '''
  Reformat in 10^{int}
  '''
  base, exponent = string.split("e")
  string = f"10^{{{int(exponent)}}}"
  return string


def main():
  plt.ion()

if __name__ == "__main__":
    main()

