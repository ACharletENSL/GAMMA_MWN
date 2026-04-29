# -*- coding: utf-8 -*-
# @Author: acharlet

'''
Contains many useful functions and parameters for plotting
'''

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from IO import openData, get_variable
from environment import MyEnv

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
def plot_snapshot(name, it, key='Last', ax_in=None, **kwargs):
  df = openData(key, it)
  env = MyEnv(key)
  val = get_variable(df, name, env)

  if ax_in is None:
    fig, ax = plt.subplots()
  else:
    ax = ax_in
  
  ax.plot(val, **kwargs)