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
r_scale = c_
z_scale = 1.

# Legends and labels
var_exp = {
  "x":"$r$", "dx":"$dr$", "rho":"$\\rho$", "vx":"$\\beta$", "p":"$p$",
  "D":"$\\gamma\\rho$", "sx":"\\gamma^2\\rho h$", "tau":"$\\tau$",
  "T":"$\\Theta$", "h":"$h$", "lfac":"$\\gamma$", "u":"$\\gamma\\beta$",
  "Eint":"$e$", "Ekin":"$e_k$"
}
var_units = {
  "x":" (cm)", "dx":" (cm)", "rho":" (g cm$^{-3}$)", "vx":"", "p":" (Ba)",
  "D":"", "sx":"", "tau":"",
  "T":"", "h":"", "lfac":"", "u":"",
  "Eint":" (erg cm$^{-3}$)", "Ekin":" (erg cm$^{-3}$)"
}

# One-file plotting functions
# --------------------------------------------------------------------------------------------------
def plot_mono(var, it, key='Last', slope=False):

  '''
  Plots a var in a single figure with title, labels, etc.
  '''
  

  plt.figure()
  ax = plt.gca()
  t = plot1D(var, slope, it, key, ax)
  ax.set_xlabel(var_exp["x"]+var_units["x"])
  varlabel = var_exp[var]
  if slope:
    varlabel = "d$\\log(" + varlabel.replace('$', '') + ")/$d$\\log r$"
  else:
    varlabel += var_units[var]
  ax.set_ylabel(varlabel)
  t_str = f"{t:.2e}"
  base, exponent = t_str.split("e")
  t_str = f"{base} \\times 10^{{{int(exponent)}}}"
  title = f"it {it}, $t/{t_scale_str} = {t_str}$"
  plt.title(title)
  plt.tight_layout()

def plot_primvar(it, key='Last'):
  
  '''
  Plots the primitive variables - with lfac*v instead of v - at chosen iteration in a single figure
  '''
  plot_multi(["rho", "u", "p"], it, key)

def plot_consvar(it, key='Last'):

  '''
  Plots the conservative variables at chosen iteration in a signle figure
  '''
  plot_multi(["D", "sx", "tau"], it, key)

def plot_multi(varlist, it, key='Last'):

  '''
  Plots variables among a list in a single figure
  '''

  Nk = len(varlist)
  f, axes = plt.subplots(Nk, 1, sharex=True, figsize=(6,2*Nk))

  for var, k, ax in zip(varlist, range(Nk), axes):
    ax.set_ylabel(var_exp[var]+var_units[var])
    t = plot1D(var, False, it, key, ax)
  
  t_str = f"{t:.2e}"
  base, exponent = t_str.split("e")
  t_str = f"{base} \\times 10^{{{int(exponent)}}}"
  title = f"it {it}, $t/{t_scale_str} = {t_str}$"

  axes[-1].set_xlabel(var_exp["x"]+var_units["x"])
  f.suptitle(title)
  plt.tight_layout()

def plot1D(var, slope, it, key, ax=None, z_norm=1., line=True, **kwargs):

  '''
  Creates ax object to be insered in plot. Scales data
  '''
  physpath = get_physfile(key)
  env = MyEnv()
  env.setupEnv(physpath)
  # add other time scaling
  tscales = {
    '1':1.,
    't_0':env.t_0
  }
  if ax is None:
    plt.figure()
    ax = plt.gca()
  
  ax.set_xscale('log')
  if slope:
    ax.set_ylim(-5, 2)
  else:
    ax.set_yscale('log')
  
  df, t, dt = openData_withtime(key, it)
  dt /= tscales[t_scale_str]
  n = df["zone"] + 1
  x, z = get_variable(df, var)
  r = x*c_
  if slope:
    logx = np.log10(x)
    logz = np.log10(z)
    z = np.gradient(logz, logx)
  else:
    z *= z_norm

  if line:
    ax.plot(r, z, 'k', zorder=1)
  scatter = ax.scatter(r, z, c=n, lw=1, zorder=2, cmap='Paired')
  ax.legend(*scatter.legend_elements())
  return dt


# Multi file plotting functions
# --------------------------------------------------------------------------------------------------
def plot_radii(key='Last'):
  '''
  Plot the various shock/CD radius of the simulation vs time
  '''
  rad_legend = ['$R_{TS}$', '$R_{b}$', '$R_{shell}$', '$R_{RS}$', '$R_{CD}$', '$R_{FS}$']
  time, radii = get_radii(key)
  Nr = radii.shape[0]
  for n in range(Nr):
    plt.loglog(time, radii[n], c=plt.cm.Paired(n), label=rad_legend[n])
  plt.legend()
