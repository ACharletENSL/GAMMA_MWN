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
from data_IO import openData_withtime, readData_withZone, get_variable

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
# add scaling from environment as dictionnary entry
t_scale_str = "1"
t_scale = 1.
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
def plot_mono(var, it, key='Last'):

  '''
  Plots a var in a single figure with title, labels, etc.
  '''

  df, t, tsim = openData_withtime(key, it)
  # add t scaling (t_0, t_c, etc.)
  title = f"it {it}, $t/{t_scale_str} = {tsim/t_scale:.3e}$ s"

  plt.figure()
  ax = plt.gca()
  plot1D(var, it, key, ax)
  ax.set_xlabel(var_exp["x"]+var_units["x"])
  ax.set_ylabel(var_exp[var]+var_units[var])
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

  df, t, tsim = openData_withtime(key, it)
  # add t scaling (t_0, t_c, etc.)
  title = f"it {it}, $t/{t_scale_str} = {tsim/t_scale:.3e}$ s"

  Nk = len(varlist)
  f, axes = plt.subplots(Nk, 1, sharex=True, figsize=(6,2*Nk))

  for var, k, ax in zip(varlist, range(Nk), axes):
    ax.set_ylabel(var_exp[var]+var_units[var])
    plot1D(var, it, key, ax)
  
  axes[-1].set_xlabel(var_exp["x"]+var_units["x"])
  f.suptitle(title)
  plt.tight_layout()

def plot1D(var, it, key, ax=None, z_norm=1., line=True, **kwargs):

  '''
  Creates ax object to be insered in plot. Scales data
  '''

  if ax is None:
    plt.figure()
    ax = plt.gca()
  
  ax.set_xscale('log')
  ax.set_yscale('log')
  
  df = readData_withZone(key, it)
  n = df["zone"] - 1
  x, z = get_variable(df, var)
  r = x*c_
  z *= z_norm

  if line:
    ax.plot(r, z, 'k', zorder=1)
  ax.scatter(r, z, c=n, lw=1, zorder=2, cmap='Paired')


# Multi file plotting functions
# --------------------------------------------------------------------------------------------------
# write extracted datas in a separated file to avoid redo analysis every time

