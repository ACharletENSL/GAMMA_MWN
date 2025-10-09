# -*- coding: utf-8 -*-
# @Author: eliotayache
# @Date:   2020-05-14 16:24:48
# @Last Modified by:   Arthur Charlet
# @Last Modified time: 

'''
This file contains functions used to print GAMMA outputs. These functions 
should be run from the ./bin/Tools directory.
This can be run from a jupyter or iPython notebook:
$run plotting_scripts.py

acharlet: Added z normalisation, separated data opening and data plotting
Modifications for wrapper script additional_plotting.py
Global rewrite to be done to fully separate the two
'''

# Imports
# --------------------------------------------------------------------------------------------------
import numpy as np
from scipy.signal import savgol_filter
import pandas as pd
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


# MPL options
# --------------------------------------------------------------------------------------------------
plt.rc('font', family='serif', size=12)
plt.rc('xtick', labelsize=12) 
plt.rc('ytick', labelsize=12) 
plt.rc('legend', fontsize=12) 
plt.rcParams['savefig.dpi'] = 200


# IO functions
# --------------------------------------------------------------------------------------------------
def readData(key, it=None, sequence=False):
  if sequence:
    filename = '../../results/%s/phys%010d.out' % (key, it)
  elif it is None:
    filename = '../../results/%s' % (key)
  else:
    filename = '../../results/%s%d.out' % (key, it)
  data = pd.read_csv(filename, sep=" ")
  return(data)


def pivot(data, key):
  return(data.pivot(index="j", columns="i", values=key).to_numpy())

# Plotting functions
# --------------------------------------------------------------------------------------------------
def plot_primvars(key, it):
  data = readData(key, it, sequence=True)
  keys = ['rho', 'u', 'p']
  plotMulti(data, keys, logx=False, logz=['rho', 'p'])

def plot_energies(key, it):
  data = readData(key, it, sequence=True)
  keys = ['erest', 'ekin', 'eint']
  plotMulti(data, keys, logx=False, logz=['erest', 'ekin', 'eint'])

def plot_vars(key, it, varlist, loglist=None):
  data = readData(key, it, sequence=True)
  if not loglist:
    loglist = []
  plotMulti(data, varlist, logx=False, logz=loglist)

def plotMulti(data, keys, jtrack=None, logx=True, logz=[], znorms={}, labels={}, **kwargs):

  '''
  Plots multiple variables for a single 1D track in the same figure.

  Args:
  -----
  data: pandas dataframe. Output data.
  keys: list of string. Variables to plot.

  kwargs:
  -------
  log: list of strings. keys of variables to be plotted in logspace.

  Returns:
  --------
  f: pyplot figure.
  axes: list of axes contained in the figure.

  Example usage:
  --------------
  f, axes = plotMulti(data, ["rho","p","lfac"], 
    tracer=False,
    line=False,
    labels={"rho":"$\\rho/\\rho_0$", "p":"$p/p_0$","lfac":"$\\gamma$"}, 
    x_norm=RShock)
  '''

  Nk = len(keys)
  f, axes = plt.subplots(Nk, 1, sharex=True, figsize=(6,2*Nk))

  for key, k, ax in zip(keys, range(Nk), axes):

    logkey = False
    logzkey = False
    label = None
    z_norm = None
    if key in logz:
      logzkey = True
    if key in labels:
      label = labels[key]
    if key in znorms:
      z_norm = znorms[key]

    plot1D(data, key, ax, z_norm=z_norm, jtrack=jtrack, logx=logx, logz=logzkey, label=label, **kwargs)

  #plt.tight_layout()

  return(f, axes)


def plot1D(data, key, ax=None, mov="x", logx=False, logz=True, v1min=None, tracer=False,
           line=True, r2=False, x_norm=None, z_norm=None, jtrack=None, label=None, 
           **kwargs):

  '''
  Plots 1D outputs from GAMMA.
  Works on 1D AND 2D outputs. In the 2D case, specify the jtrack to plot.

  Args:
  -----
  data: pandas dataframe. Output data.
  key: string. Variable to plot.

  Example usage:
  --------------
  data = readData('Last/phys0000000000.out')
  plot1D(data, "rho", log=True, jtrack=0)
  '''

  x, z, tracvals = getData1D(data, key, r2, x_norm, z_norm, jtrack)

  if ax is None:
    plt.figure()
    ax = plt.gca()

  if label is not(None):
    ax.set_ylabel(label)
  else:
    ax.set_ylabel(key)

  if logx:
    ax.set_xscale('log')
  
  if logz:
    ax.set_yscale('log')

  if line:
    ax.plot(x, z, 'k',zorder=1)    
  ax.scatter(x, z, c='None', edgecolors='k', lw=2, zorder=2, label="numerical")
  if tracer:
    ax.scatter(x, z, c=tracvals, edgecolors='None', zorder=3, cmap='cividis')

def plotSlope1D(data, key, ax=None, mov="x", v1min=None, tracer=False,
           line=True, r2=False, x_norm=None, jtrack=None, label=None, 
           **kwargs):
  '''
  Plots the logarithmic slope of the chosen data. Follows the rules of plot1D()
  Without options logx, logz and znorm of plot1D
  '''

  x, z, tracvals = getData1D(data, key, r2, x_norm, jtrack)

  if ax is None:
    plt.figure()
    ax = plt.gca()

  if label is not(None):
    ax.set_ylabel(label)
  else:
    ax.set_ylabel(key+' slope')
  
  logx = np.log10(x)
  logz = np.log10(z)
  grad = np.gradient(logz, logx)
  s = savgol_filter(grad, 9, 3)

  if line:
    ax.plot(x, s, 'k',zorder=1)    
  ax.scatter(x, s, c='None', edgecolors='k', lw=2, zorder=2, label="numerical")
  if tracer:
    ax.scatter(x, s, c=tracvals, edgecolors='None', zorder=3, cmap='cividis')
  
  ax.set_xscale("log")
  ax.set_ylim((-5, 5))


def getData1D(data, key, r2=False, x_norm=None, z_norm=None, jtrack=None):
  '''
  Returns x and z data according to chosen key
  '''
  def gamma(rho, p):
    g = 5./3.
    a = p/(rho * (g-1.))
    e_ratio = a + np.sqrt(1 + a*a)
    g_eff = g - (g-1.)/2. * (1.-1./(e_ratio*e_ratio))
    return g_eff

  if key == "lfac":
    var = "vx"
  elif key == "erest":
    var = "rho"
  elif key == "u":
    var = "vx"
  else:
    var = key

  if jtrack is not(None):
    z = pivot(data, var)[jtrack, :]
    x = pivot(data, "x")[jtrack, :]
    tracvals = pivot(data, "trac")[jtrack, :]
  else:
    if key == "T":
      rho = data["rho"].to_numpy()
      p = data["p"].to_numpy()
      z = p / rho
    elif key == "h":
      rho = data["rho"].to_numpy()
      p = data["p"].to_numpy()
      gma = gamma(rho, p)
      z   = 1.+p*gma/(gma-1.)/rho
    elif key == "u":
      v = data["vx"].to_numpy()
      lfac = 1./np.sqrt(1 - v**2)
      z = v*lfac
    elif key == "ref":
      D = data["D"].to_numpy()
      z = refCriterion(D)
    elif key == "ekin":
      rho = data["rho"].to_numpy()
      vx = data["vx"].to_numpy()
      lfac = 1./np.sqrt(1 - vx**2)
      z = (lfac-1)*rho
    elif key == "eint":
      rho = data["rho"].to_numpy()
      p = data["p"].to_numpy()
      gma = gamma(rho, p)
      z = p/(gma-1)
    else:
      z = data[var].to_numpy()
    x = np.copy(data["x"].to_numpy())
    tracvals = data["trac"].to_numpy()

  if x_norm is not(None):
    x *= x_norm
  
  if z_norm is not(None):
    z *= z_norm

  if key == "lfac":
    z = 1./np.sqrt(1 - z**2)

  if r2:
    z *= x**2
  
  return x, z, tracvals


def plot2D(data, key, z_override=None, mov="x", log=False, v1min=None, 
           geometry="cartesian", quiver=False, color=None,  edges='None', 
           invert=False, r2=False, cmap='magma', tlayout=False, colorbar=True, 
           slick=False, phi=0., fig=None, label=None, axis=None,  thetaobs=0., 
           nuobs=1.e17, shrink=0.6, expand=False):

  '''
  Plots 2D outputs from GAMMA.

  Args:
  -----
  data: pandas dataframe. Output data.
  key: string. Variable to plot.

  Returns:
  --------
  xmin: double. Minimum coordinate x in data.
  xmax: double. Maximum coordinate x in data.
  thetamax: double. Highest track angle in polar geometry.
  im: pyplot.image. 2D map of the requested variable.

  Example usage:
  --------------
  data = readData('Last/phys0000000000.out')

  # On specific axes
  f = plt.figure()
  ax = plt.axes(projection='polar')
  plot2D(data, "rho", fig=f, axis=ax, **kwargs)

  # On axies of its own
  plot2D(data, "rho", geometry='polar', **kwargs)
  '''

  if z_override is not None:
    z = z_override
  if key == "lfac":
    vx = data.pivot(index='j', columns='i', values="vx").to_numpy()
    vy = data.pivot(index='j', columns='i', values="vy").to_numpy()
    z = 1./np.sqrt(1 - (vx**2+vy**2))    
  else:
    z = data.pivot(index='j', columns='i', values=key).to_numpy()
  x = data.pivot(index='j', columns='i', values='x').to_numpy()
  dx = data.pivot(index='j', columns='i', values='dx').to_numpy()
  y = data.pivot(index='j', columns='i', values='y').to_numpy()
  dy = data.pivot(index='j', columns='i', values='dy').to_numpy()

  # duplicating last row for plotting
  z = np.append(z, np.expand_dims(z[-1, :], axis=0), axis=0)
  x = np.append(x, np.expand_dims(x[-1, :], axis=0), axis=0)
  dx = np.append(dx, np.expand_dims(dx[-1, :], axis=0), axis=0)
  y = np.append(y, np.expand_dims(y[-1, :], axis=0), axis=0)
  dy = np.append(dy, np.expand_dims(dy[-1, :], axis=0), axis=0)

  # duplicating first column for plotting
  z = np.append(z, np.expand_dims(z[:, -1], axis=1), axis=1)
  x = np.append(x, np.expand_dims(x[:, -1], axis=1), axis=1)
  dx = np.append(dx, np.expand_dims(dx[:, -1], axis=1), axis=1)
  y = np.append(y, np.expand_dims(y[:, -1], axis=1), axis=1)
  dy = np.append(dy, np.expand_dims(dy[:, -1], axis=1), axis=1)

  nact = np.array([np.count_nonzero(~np.isnan(xj)) for xj in x])

  if (quiver):
    vx = data.pivot(index='j', columns='i', values='vx').to_numpy()
    vx = np.append(vx, np.expand_dims(vx[-1, :], axis=0), axis=0)
    vx = np.ma.masked_array(vx, np.isnan(vx))

  if r2:
    z *= x**2

  xmin = np.nanmin(x)
  xmax = np.nanmax(x)
  ymin = np.nanmin(y)
  ymax = np.nanmax(y)

  vmax = np.nanmax(z[4:, :])
  vmin = np.nanmin(z)
  if log:
    vmin = np.nanmin(z[z > 0])
  if v1min:
    vmin = v1min

  if geometry == "polar":
    projection = "polar"
  else:
    projection = None

  if axis is None:
    f = plt.figure()
    ax = plt.axes(projection=projection)
  else:
    f = fig
    ax = axis

  if geometry == "polar" or axis is not None:
    ax.set_thetamax(ymax*180./np.pi)
    ax.set_thetamin(ymin*180./np.pi)
    if invert:
      ax.set_thetamin(-ymax*180./np.pi)

  if slick:
    ax.axis("off")

  for j in range(z.shape[0]-1):
    xj = x - dx/2.
    yj = y - dy/2.
    dyj = dy
    xj[j, nact[j]-1] += dx[j, nact[j]-1]

    if mov == 'y':
      tmp = np.copy(xj)
      xj = yj
      yj = np.copy(tmp)
      dyj = dx

    xj[j+1, :] = xj[j, :]   
    yj[j+1, :] = yj[j, :]+dyj[j, :]   

    xj = xj[j:j+2, :]
    yj = yj[j:j+2, :]
    zj = z[j:j+2, :]

    if invert:
      yj *= -1

    if log:
      im = ax.pcolor(yj, xj, zj, 
                     norm=LogNorm(vmin=vmin, vmax=vmax), 
                     edgecolors=edges,
                     cmap=cmap,
                     facecolor=color)
    else:
      im = ax.pcolor(yj, xj, zj, 
                     vmin=vmin, vmax=vmax, 
                     edgecolors=edges, 
                     cmap=cmap,
                     facecolor=color)

  if geometry != "polar":
    ax.set_aspect('equal')
  if geometry == "polar" or axis is not None:
    ax.set_rorigin(0)
    ax.set_rmin(xmin)
    ax.set_rticks([xmin, xmax])

  if colorbar:
    cb = f.colorbar(im, ax=ax, orientation='vertical', shrink=shrink, pad=0.1)
    if label is None:
      label = key  
    cb.set_label(label, fontsize=14)

  if tlayout:
    f.tight_layout()

  thetamax = ymax*180./np.pi
  return xmin, xmax, thetamax, im
