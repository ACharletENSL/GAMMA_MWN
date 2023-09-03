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

# Create figure 
# --------------------------------------------------------------------------------------------------
def plot_mono(var, it, key='Last', scaletype='default', slope=False, fig=None):
  '''
  Plots a single var in a single figure with title, labels, etc.
  '''
  
  if fig:
    ax = fig.gca()
  else:
    fig = plt.figure()
    ax = plt.gca()
  title, legend = plot_snapshot_1D(var, it, key, scaletype, slope=None, ax=None, col='Zone')
  fig.suptitle(title)
  fig.legend(legend)
  fig.tight_layout()

# Several lines in one figure
def plot_primvar(it, key='Last', scaletype=None):

  '''
  Plots the primitive variables - with lfac*v instead of v - at chosen iteration in a single figure
  '''
  plot_multi(["rho", "u", "p"], it, key, scaletype)

def plot_consvar(it, key='Last', scaletype=None):
  '''
  Plots the conservative variables at chosen iteration in a single figure
  '''
  plot_multi(["D", "sx", "tau"], it, key, scaletype)

def plot_energy(it, key='Last'):
  '''
  Plot energy content
  '''
  varlist = ["Eint", "Ekin", "Emass"]
  plot_comparison(varlist, it, key, col=None)

def plot_comparison(varlist, it, key='Last', scaletype=None, col=None):
  '''
  Plots variables among a list in a single figure
  '''

  plt.figure()
  ax = plt.gca()

  for var in varlist:
    title, legend = plot_snapshot_1D(var, it, key, scaletype, ax, col, slope=None)
  
  plt.suptitle(title)
  plt.legend(ax.scatter.legend_elements())
  plt.tight_layout()


def plot_multi(varlist, it, key='Last', scaletype=None):
  '''
  Plots variables among a list in a single figure with subplots
  '''

  Nk = len(varlist)
  f, axes = plt.subplots(Nk, 1, sharex=True, figsize=(6,2*Nk))

  for var, k, ax in zip(varlist, range(Nk), axes):
    title, legend = plot_snapshot_1D(var, it, key, scaletype, ax)
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
    energies = ['Emass', 'Ekin', 'Eint']
    linestyles = [':', '--', '-.']
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
def plot_snapshot_1D(var, it, key, scaletype='default', ax=None, col='Zone', slope=None, line=True, **kwargs):
  '''
  Creates ax object to be insered in plot. Scales data
  '''
  physpath = get_physfile(key)
  env = MyEnv(physpath)
  df, t, dt = openData_withtime(key, it)
  n = df["zone"]
  x, z, xlabel, zlabel = get_scaledvar_snapshot(var, df, env, scaletype, slope)
  
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
  if var == 'u':
    ax.axhline(env.u, color='k', ls='--', lw=.6)
    ax.text(1.02, env.u, '$u_{th}$', color='k', transform=ax.get_yaxis_transform(),
      ha='left', va='center')
  Nv = vars.shape[0]
  for n in range(Nv):
    if slope:
      ax.plot(time, vars[n], c=plt.cm.Paired(n), label=var_legends[n], **kwargs)
    else:
      ax.plot(time, vars[n], c=plt.cm.Paired(n), label=var_legends[n], **kwargs)
  if var not in ['Nc', 'v', 'R', 'Rcd', 'Rct', 'u']:
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(formatter)

  ax.set_xlabel(tlabel)
  ax.set_ylabel(ylabel)

  return title
  

# Get values with scaling and labels for axes
# --------------------------------------------------------------------------------------------------
def get_scaledvar_snapshot(var, df, env, scaletype='default', slope=None):
  '''
  Return x, y values and relevant labels according to chosen scaletype
  Includes: code, CGS, shells
  to add: MWN_sd, MWN_SNR
  '''
  # Legends and labels
  var_exp = {
  "x":"$r$", "dx":"$dr$", "rho":"$\\rho$", "vx":"$\\beta$", "p":"$p$",
  "D":"$\\gamma\\rho$", "sx":"$\\gamma^2\\rho h$", "tau":"$\\tau$",
  "trac":"", "Sd":"", "gmin":"$\\gamma_{min}$", "gmax":"$\\gamma_{max}$", "zone":"",
  "T":"$\\Theta$", "h":"$h$", "lfac":"$\\gamma$", "u":"$\\gamma\\beta$",
  "Eint":"$e$", "Ekin":"$e_k$", "Emass":"$\\rho c^2$", "dt":"dt", "res":"dr/r"
  }
  units_CGS  = {
  "x":" (cm)", "dx":" (cm)", "rho":" (g cm$^{-3}$)", "vx":"", "p":" (Ba)",
  "D":"", "sx":"", "tau":"", "trac":"", "Sd":"", "gmin":"", "gmax":"",
  "T":"", "h":"", "lfac":"", "u":"", "zone":"", "dt":" (s)", "res":"",
  "Eint":" (erg cm$^{-3}$)", "Ekin":" (erg cm$^{-3}$)", "Emass":" (erg cm$^{-3}$)"
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
  if scaletype == 'code':
    units = get_normunits('c', 'Norm')
    r_scale, rho_scale = 1., 1.
  elif scaletype == 'CGS':
    units = units_CGS
    r_scale, rho_scale = c_, getattr(env, env.rhoNorm)
  else:
    if env.mode == 'shells':
      sub = env.rhoNorm[3:]
      units = get_normunits('R_0', sub)
      r_scale, rho_scale = c_/env.R0, getattr(env, env.rhoNorm)
    elif env.mode == 'PWN':
      print("Implement PWN cases")

  x, z   = get_variable(df, var)
  x = x.to_numpy()*r_scale
  z = z*get_varscaling(var, rho_scale)
  xlabel = var_exp['x'][:-1] + units['x'][1:]
  zlabel = (var_exp[var][:-1] + units[var][1:]) if units[var] else var_exp[var]
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
  var_label = {'R':'$r$ (cm)', 'v':'$\\dot{R}$', 'u':'$\\gamma\\beta$',
    'V':'$V$ (cm$^3$)', 'Nc':'$N_{cells}$', 'ShSt':'$\\Gamma_{ud}-1$',
    'Emass':'$E_{r.m.}$', 'Ekin':'$E_k$', 'Eint':'E_{int}',
    'Rct':'$(r - ct)/R_0$ (cm)', 'Rcd':'$r - r_{cd}$ (cm)'}

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
  else:
    varlist = get_varlist(var, Nz, env.mode)
  if var == 'Rcd':
    varlist.remove('R_{cd}')
    
  var_legends = ['$' + var_leg + '$' for var_leg in varlist]
  vars = df_res[varlist].to_numpy().transpose()
  Nv = vars.shape[0]
  if slope:
    logt = np.log10(time)
    logvars = np.log10(vars)
    vars = np.gradient(logvars, logt, axis=1)
    new_legends = ["d$\\log(" + var_leg.replace('$', '') + ")/$d$\\log r$" for var_leg in var_legends]
    var_legends = new_legends

  return time, vars, tlabel, ylabel, tscale, var_legends

def get_scaledvar_it(var, key, env):
  var_label = {'R':'$r$ (cm)', 'v':'$\\dot{R}$', 'u':'$\\gamma\\beta$',
    'V':'$V$ (cm$^3$)', 'Nc':'$N_{cells}$', 'ShSt':'$\\Gamma_{ud}-1$',
    'Emass':'$E_{r.m.}$', 'Ekin':'$E_k$', 'Eint':'E_{int}',
    'Rct':'$(r - ct)/R_0$ (cm)', 'Rcd':'$r - r_{cd}$ (cm)'}

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

