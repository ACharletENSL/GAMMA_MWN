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
from environment import *
from run_analysis import get_radii
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
t_scale_str = "t_0"   # choice of t scaling. Dictionnary defined in plot1D
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
def plot_mono(var, it, key='Last', slope=False, code_units=False, fig=None):

  '''
  Plots a var in a single figure with title, labels, etc.
  '''
  
  if fig:
    ax = fig.gca()
  else:
    fig = plt.figure()
    ax = plt.gca()
  title = plot1D(var, slope, it, key, ax, code_units)
  ax.set_xlabel(var_exp["x"]+var_units["x"])
  varlabel = var_exp[var]
  if slope:
    varlabel = "d$\\log(" + varlabel.replace('$', '') + ")/$d$\\log r$"
  else:
    if code_units:
      varlabel += ' (code units)'
    else:
      varlabel += var_units[var]
  ax.set_ylabel(varlabel)
  fig.suptitle(title)
  fig.tight_layout()

def plot_primvar(it, key='Last'):
  
  '''
  Plots the primitive variables - with lfac*v instead of v - at chosen iteration in a single figure
  '''
  plot_multi(["rho", "u", "p"], it, key)

def plot_consvar(it, key='Last'):

  '''
  Plots the conservative variables at chosen iteration in a single figure
  '''
  plot_multi(["D", "sx", "tau"], it, key)

def plot_multi(varlist, it, key='Last'):

  '''
  Plots variables among a list in a single figure with subplots
  '''

  Nk = len(varlist)
  f, axes = plt.subplots(Nk, 1, sharex=True, figsize=(6,2*Nk))

  for var, k, ax in zip(varlist, range(Nk), axes):
    ax.set_ylabel(var_exp[var]+var_units[var])
    title = plot1D(var, False, it, key, ax)
  
  axes[-1].set_xlabel(var_exp["x"]+var_units["x"])
  f.suptitle(title)
  plt.tight_layout()

def plot_energy(it, key='Last'):

  '''
  Plot energy content
  '''
  varlist = ["Eint", "Ekin", "Emass"]
  plot_comparison(varlist, it, key, colors=None)


def plot_comparison(varlist, it, key='Last', colors='Zone'):

  '''
  Plots variables among a list in a single figure
  '''

  plt.figure()
  ax = plt.gca()

  for var in varlist:
    varlabel = var_exp[var]+var_units[var]
    title = plot1D(var, False, it, key, ax, colors=colors, label=varlabel)
  
  ax.set_xlabel(var_exp["x"]+var_units["x"])
  plt.suptitle(title)
  plt.legend()
  plt.tight_layout()

def plot1D(var, slope, it, key, ax=None, code_units=False, line=True, colors='Zone', **kwargs):

  '''
  Creates ax object to be insered in plot. Scales data
  '''
  physpath = get_physfile(key)
  env = MyEnv()
  env_init(env, physpath)
  # add other time scaling
  tscales = {
    '1':1.,
    't_0':env.t_0
  }
  var_scale = get_scalings(env.rho_w)
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
  
  df, t, dt = openData_withtime(key, it)
  rc = (get_radius_zone(df, n=2)*c_ - env.R_b) / env.R_b
  dt /= tscales[t_scale_str]
  t_str = f"{dt:.2e}"
  base, exponent = t_str.split("e")
  t_str = f"{base} \\times 10^{{{int(exponent)}}}"
  rc_str= f"{rc:.2e}"
  base, exponent = rc_str.split("e")
  rc_str=f"{base} \\times 10^{{{int(exponent)}}}"
  title = f"it {it}, $t/{t_scale_str} = {t_str}$, $\\Delta R_{{CD}}/R_{{CD,0}} = {rc_str}$"
  
  n = df["zone"] + 1
  x, z = get_variable(df, var)
  r = x*rNorm
  if code_units:
    zNorm=1.
  else:
    zNorm = var_scale[var]
  if slope:
    logx = np.log10(x)
    logz = np.log10(z)
    z = np.gradient(logz, logx)
  else:
    z *= zNorm

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
def plot_radii(key='Last'):
  '''
  Plot the various shock/CD radius of the simulation vs time
  '''
  rad_legend = ['$R_{TS}$', '$R_{b}$', '$R_{shell}$', '$R_{RS}$', '$R_{CD}$', '$R_{FS}$']
  time, radii = get_radii(key)
  Nr = radii.shape[0]
  physpath = get_physfile(key)
  env = MyEnv()
  env_init(env, physpath)
  # add other time scaling
  tscales = {
    '1':1.,
    't_0':env.t_0
  }
  for n in range(Nr):
    plt.loglog(time/env.t_0, radii[n], c=plt.cm.Paired(n), label=rad_legend[n])
  plt.legend()
