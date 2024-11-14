'''
Script to generate figures for documents
'''

import numpy as np
import matplotlib.pyplot as plt
from plotting_scripts_new import *

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


cols_dic = {'RS':'r', 'FS':'b', 'RS+FS':'k', 'CD':'purple',
  'CD-':'mediumvioletred', 'CD+':'mediumpurple', '3':'darkred', '2':'darkblue'}


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
  if type(varlist)==str: varlist = [varlist]
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
  Broken power-law woth given breaks positions and power-law indices
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
    break_names = ["$\\nu'_B$", "$\\nu'_m$", "$\\nu'_c$", "$\\nu'_M$"]
    power_law_indices = [1./3, (1-p)/2, -p/2]
    plaw_names = ["$\\nu'^{1/3}$", "$\\nu'^{\\frac{1-p}{2}}$", "$\\nu'^{-p/2}$"]
    title = 'Slow cooling'

  elif regime == 'FC':
    break_positions = [1, 10, 100, 1000]
    break_names = ["$\\nu'_B$", "$\\nu'_c$", "$\\nu'_m$", "$\\nu'_M$"]
    power_law_indices = [1./3, -1/2, -p/2]
    plaw_names = ['1/3', '-1/2', '-p/2']
    title = 'Fast cooling'
  
  elif regime == 'vFC':
    break_positions = [1, 100, 1000]
    break_names = ["$\\nu'_B$", "$\\nu'_m$", "$\\nu'_M$"]
    power_law_indices = [-1/2, -p/2]
    plaw_names = ['-1/2', '-p/2']
    title = 'Very fast cooling'

  elif regime == 'CD':
    break_positions = [1, 100, 150]
    break_names = ["$\\nu'_B$", "$\\nu'_m$", "$\\nu'_M$"]
    power_law_indices = [1/3, -1/2]
    plaw_names = ['1/3', '-1/2']
    title = 'Cooled state'

  norm_index = break_names.index("$\\nu'_m$")
  plot_broken_power_law(break_positions, power_law_indices, break_names, plaw_names,
      norm_name="$P'_{\\nu'_m}$", xvar="$\\nu'$", yvar="$P'_{\\nu'}$", title=title, norm_index=norm_index) 
  plt.savefig(figtitle)
  plt.show()