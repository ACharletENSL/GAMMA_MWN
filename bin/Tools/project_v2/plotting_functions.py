# -*- coding: utf-8 -*-
# @Author: acharlet

'''
Contains many useful functions and parameters for plotting
'''

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from IO import openData, get_variable
from environment import MyEnv
from math import floor, log10

def sci_notation(num, decimal_digits=1, precision=None, exponent=None):
  """
  Returns a string representation of the scientific
  notation of the given number formatted for use with
  LaTeX or Mathtext, with specified number of significant
  decimal digits and precision (number of decimal digits
  to show). The exponent to be used can also be specified
  explicitly.
  """
  if num == 0.:
    return "$0$"
  if exponent is None:
      exponent = int(floor(log10(abs(num))))
  coeff = round(num / float(10**exponent), decimal_digits)
  if precision is None:
      precision = decimal_digits
  sci = r"${0:.{2}f}\cdot10^{{{1:d}}}$".format(coeff, exponent, precision)
  sci = sci.replace(f"1.{'0'*decimal_digits}\\cdot", "")
  return sci

### axes label
T_label  = "$\\tilde{T}$"
nu_label = "$\\nu/\\nu_0$"
nupk_label = "$\\nu_{\\rm pk}/\\nu_0$"
nF_label = "$\\nu F_\\nu / \\nu_0 F_0$"
nFpk_label = "$(\\nu F_\\nu)_{\\rm pk} / \\nu_0 F_0$"

scale_exps = {
  'log_aum':'log$_{10}(a_u -1)$',
  'chi':'$\\chi$' 
  }

def slope_label(label):
  slabel = '$\\frac{\\rm d}{\\rm d\\log}' + label[1:]
  return slabel

### create legends label
def create_legend_colors(ax, names, colors, **kwargs):
  dummy_col = [ax.plot([], [], c=col, ls='-')[0] for col in colors]
  legend = ax.legend(dummy_col, names, **kwargs)
  return legend

def create_legend_styles(ax, names, styles, **kwargs):
  dummy_lst = [ax.plot([], [], c='k', ls=l)[0] for l in styles]
  legend = ax.legend(dummy_lst, names, **kwargs)
  return legend

### transform axs to put text (or anything)
def transx(ax):
  return transforms.blended_transform_factory(ax.transAxes, ax.transData)
def transy(ax):
  return transforms.blended_transform_factory(ax.transData, ax.transAxes)

### generate colors
#colors = plt.cm.jet(np.linspace(0,1,N))

### Basic plotting
def plot_prim_snapshot(it, key='Last'):
  names = ['rho', 'u', 'p']
  plot_snapshot_multi(names, it, key)

def plot_cons_snapshot(it, key='Last'):
  names = ['D', 'sx', 'tau']
  plot_snapshot_multi(names, it, key)

def plot_snapshot_multi(names, it, key='Last'):
  N = len(names)
  fig, axs = plt.subplots(N, 1, sharex=True)
  for name, ax in zip(names, axs):
    plot_snapshot(name, it, key, ax_in=ax)
  fig.tight_layout()

def plot_its_snapshots(name, its, key='Last'):
  fig, ax = plt.subplots()
  for it in its:
    plot_snapshot(name, it, key=key, ax_in=ax, label=f'it {it}')
  if len(its) < 5:
    ax.legend()
  fig.tight_layout()

def plot_snapshot(name, it, key='Last', ax_in=None, **kwargs):
  df = openData(key, it)
  env = MyEnv(key)
  val = get_variable(df, name, env)

  if ax_in is None:
    fig, ax = plt.subplots()
    fig.tight_layout()
  else:
    ax = ax_in
  
  ax.plot(val, **kwargs)