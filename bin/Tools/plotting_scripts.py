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
from run_analysis import get_timeseries
from data_IO import *
from init_conditions_MWN import env_init

# matplotlib options
# --------------------------------------------------------------------------------------------------
plt.rc('font', family='serif', size=12)
plt.rc('xtick', labelsize=12) 
plt.rc('ytick', labelsize=12) 
plt.rc('legend', fontsize=12) 
plt.rcParams['savefig.dpi'] = 200
# add color coding of zones from 0 to 6, with 0-1, 2-4, 5-6 color grouped
# if 'Paired' cmap is not what we want

# Scaling
scale = "sim"   # choice of t and r scaling: None, sim, sd (spindown)
rNorm = c_

def get_scalings(rhoNorm):
  pNorm = rhoNorm*c_**2
  var_scale = {}
  for var, unit in var_units.items():
    if unit == " (cm)": var_scale[var]=rNorm
    elif var == "rho" or var == "D": var_scale[var]=rhoNorm
    elif var == "p" or unit == " (erg cm$^{-3}$)": var_scale[var]=pNorm
    else: var_scale[var]=1.
  return var_scale


# Legends and labels
var_exp = {
  "x":"$r$", "dx":"$dr$", "rho":"$\\rho$", "vx":"$\\beta$", "p":"$p$",
  "D":"$\\gamma\\rho$", "sx":"$\\gamma^2\\rho h$", "tau":"$\\tau$",
  "trac":"", "Sd":"", "gmin":"$\\gamma_{min}$", "gmax":"$\\gamma_{max}$", "zone":"",
  "T":"$\\Theta$", "h":"$h$", "lfac":"$\\gamma$", "u":"$\\gamma\\beta$",
  "Eint":"$e$", "Ekin":"$e_k$", "Emass":"$\\rho c^2$", "dt":"dt"
}
var_units = {
  "x":" (cm)", "dx":" (cm)", "rho":" (g cm$^{-3}$)", "vx":"", "p":" (Ba)",
  "D":"", "sx":"", "tau":"", "trac":"", "Sd":"", "gmin":"", "gmax":"",
  "T":"", "h":"", "lfac":"", "u":"", "zone":"", "dt":" (s)",
  "Eint":" (erg cm$^{-3}$)", "Ekin":" (erg cm$^{-3}$)", "Emass":" (erg cm$^{-3}$)"
}
nolog_vars = ['trac', 'Sd', 'gmin', 'gmax', 'zone']

# One-file plotting functions
# --------------------------------------------------------------------------------------------------
def plot_mono(var, it, key='Last', slope=False, scaletype='sim', code_units=False, fig=None):

  '''
  Plots a var in a single figure with title, labels, etc.
  '''
  
  if fig:
    ax = fig.gca()
  else:
    fig = plt.figure()
    ax = plt.gca()
  title = plot1D(var, slope, it, key, scaletype, ax, code_units)
  fig.suptitle(title)
  fig.tight_layout()

def plot_primvar(it, key='Last', scaletype='sim'):
  
  '''
  Plots the primitive variables - with lfac*v instead of v - at chosen iteration in a single figure
  '''
  plot_multi(["rho", "u", "p"], it, key, scaletype)

def plot_consvar(it, key='Last', scaletype='sim'):

  '''
  Plots the conservative variables at chosen iteration in a single figure
  '''
  plot_multi(["D", "sx", "tau"], it, key, scaletype)

def plot_multi(varlist, it, key='Last', scaletype='sim'):

  '''
  Plots variables among a list in a single figure with subplots
  '''

  Nk = len(varlist)
  f, axes = plt.subplots(Nk, 1, sharex=True, figsize=(6,2*Nk))

  for var, k, ax in zip(varlist, range(Nk), axes):
    title = plot1D(var, False, it, key, scaletype, ax)
    if k!=Nk-1: #only have xlabel on plot at bottom
      ax.set_xlabel('')
  
  f.suptitle(title)
  plt.tight_layout()

def plot_energy(it, key='Last'):

  '''
  Plot energy content
  '''
  varlist = ["Eint", "Ekin", "Emass"]
  plot_comparison(varlist, it, key, colors=None)


def plot_comparison(varlist, it, key='Last', scaletype='sim', colors='Zone'):

  '''
  Plots variables among a list in a single figure
  '''

  plt.figure()
  ax = plt.gca()

  for var in varlist:
    varlabel = var_exp[var]+var_units[var]
    title = plot1D(var, False, it, key, scaletype, ax, colors=colors, label=varlabel)
  
  plt.suptitle(title)
  plt.legend()
  plt.tight_layout()

def plot1D(var, slope, it, key, scaletype, ax=None, code_units=False, line=True, colors='Zone', **kwargs):

  '''
  Creates ax object to be insered in plot. Scales data
  '''

  physpath = get_physfile(key)
  env = MyEnv()
  env_init(env, physpath)
  scalesDict = {
    'sim':(env.t_start, env.R_b),
    'sd':(env.t_0, 1.) # add characteristic scale depending on E_0/E_SN
  }
  scalesNames = {
    'sim':('t_i', 'R_{CD,i}'),
    'sd':('t_0', 'R_0')
  }

  df, t, dt = openData_withtime(key, it)
  
  if scaletype:
    t_scale, r_scale = scalesDict[scaletype]
    t_scale_str, r_scale_str = scalesNames[scaletype]
    xlabel = f"$r/{r_scale_str}$"
    rc = (get_radius_zone(df, n=2)*c_ - env.R_b) / r_scale
    if rc < 0.: rc = 0.
    dt /= t_scale
    t_str = reformat_scientific(f"{dt:.2e}")
    rc_str= reformat_scientific(f"{rc:.2e}")
    title = f"it {it}, $\\Delta t/{t_scale_str} = {t_str}$, $\\Delta R_{{CD}}/{r_scale_str} = {rc_str}$"
  else:
    t_str = reformat_scientific(f"{t:.2e}")
    xlabel = var_exp["x"]+var_units["x"]
    title = f"it {it}, $t = {t_str}$ s"

  var_scale = 1. if code_units else get_scalings(env.rho_w)[var]
  varlabel = var_exp[var]
  if slope:
    varlabel = "d$\\log(" + varlabel.replace('$', '') + ")/$d$\\log r$"
  else:
    if code_units:
      varlabel += ' (code units)'
    else:
      varlabel += var_units[var]
  
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

  ax.set_ylabel(varlabel)
  ax.set_xlabel(xlabel)
  
  n = df["zone"] + 1
  #if slope: smoothing=slope
  x, z = get_variable(df, var)
  r = x*rNorm/r_scale
  if slope:
    logx = np.log10(x)
    logz = np.log10(denoise_data(z))
    z = np.gradient(logz, logx)
  else:
    z *= var_scale

  if line:
    ax.plot(r, z, 'k', zorder=1)
  if colors == 'Zone':
    scatter = ax.scatter(r, z, c=n, lw=1, zorder=2, cmap='Paired', **kwargs)
  else:
    scatter = ax.scatter(r, z, lw=1, zorder=2, **kwargs)
  ax.legend(*scatter.legend_elements())
  return title


# Multi file plotting functions
# --------------------------------------------------------------------------------------------------
def plot_timeseries(var_keys, key, titletype='E', slope=False, fig=False):
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
      title = plotT(var, slope, key, titletype, ax, ls=linestyles[i])
  else:
    title = plotT(var_keys, slope, key, titletype, ax)

  plt.legend()
  fig.suptitle(title)

def plotT(var, slope, key, titletype, ax=None, **kwargs):

  '''
  Create plot to be inserted into a figure
  '''

  var_label = {'R':'$r$ (cm)', 'vel':'$\\beta$', 'posvel':'$\\beta$',
    'V':'$V$ (cm$^3$)', 'Nc':'$N_{cells}$',
    'Emass':'$E_{r.m.}$', 'Ekin':'$E_k$', 'Eint':'E_{int}'}
  physpath = get_physfile(key)
  env = MyEnv()
  env_init(env, physpath)
  if ax is None:
    plt.figure()
    ax = plt.gca()

  if titletype == 'Lt':
    L0_str = reformat_scientific(f'{env.L_0:.1e}')
    t0_str = reformat_scientific(f'{env.t_0:.2e}')
    lf_str = reformat_pow10(f'{env.lfacwind:.0e}') 
    title = f'$L_0={L0_str}$ erg s$^{{-1}}$, $t_0={t0_str}$ s, $\\gamma_w={lf_str}$'
  elif titletype == 'E':
    lf_str = reformat_pow10(f'{env.lfacwind:.0e}') 
    title = f'$E_0/E_{{sn}} = {env.E_0_frac}$, $\\gamma_w={lf_str}$'
  
  positive = True if var == 'posvel' else False
  time, vars, var_legends = get_timeseries(var, key, slope, positive)

  Nv = vars.shape[0]
  for n in range(Nv):
    if slope:
      ax.plot(time/env.t_0, vars[n], c=plt.cm.Paired(n), label=var_legends[n], **kwargs)
    else:
      ax.loglog(time/env.t_0, vars[n], c=plt.cm.Paired(n), label=var_legends[n], **kwargs)
  if var=='Nc':
    ax.set_yscale('linear')
  elif var=='vel':
    ax.set_yscale('symlog')
  if slope:
    ax.set_ylim(-2, 3)
    if var == 'R': ax.set_ylim(-.5, 1.5)
  ax.set_xlabel('$t/t_0$ (s)')
  ax.set_ylabel(var_label[var])
  ymin, ymax = ax.get_ylim()
  if np.log10(ymax) - np.log10(ymin) < 2.5:
    ax.grid(color='k', alpha=0.5, linestyle='dashed', linewidth=0.5, which='minor')
  
  return title

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

