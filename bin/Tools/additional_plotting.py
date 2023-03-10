# -*- coding: utf-8 -*-
# @Author: acharlet

'''
This file contains several functions to complete those
in plotting_scripts for plots more suited to our work.
'''

# Imports
# --------------------------------------------------------------------------------------------------
import os
from environment import *
from plotting_scripts_old import *
import matplotlib.animation as mpla

t_start = 100 # need to rewrite a few things to update this correctly
# maybe norms as a class? Plotting as functions of a data class?
r_t = v_t * t_start
rho_norm = D * t_start**3
p_norm = rho_norm * c_**2
#rho_norm = 1*mp   # normalize to external ambient density, read params if you want to change aprameter later
x_norm = c_

znorms = {#scale all variables
  "rho":rho_norm,
  "D":rho_norm,
  "vx":c_,
  "p":p_norm,
  "dx":c_,
  "h":1.,
  "T":1.,
  "dx":x_norm,
  "ref":1.,
  "u": 1,
  "ekin":1,
  "eint":1.,
  "erest":1.}
legends = {
  "rho":"$\\rho$",
  "D":"$\\gamma\\rho$",
  "vx":"$v$",
  "p":"$p$",
  "u":"$\\gamma \\beta$",
  "T":"$\\Theta=p/\\rho c^2$",
  "dx":"$dr$",
  "ref":"ref. crit.",
  "ekin":"$e_k$",
  "eint":"$e$",
  "erest":"$\\rho c^2$"}
labels = {
  "dx":"$dr$ (cm)",
  "ref":"ref. crit.",
  "rho":"$\\rho$ (g.cm$^{-3}$)",
  "D":"\\gamma\\rho",
  "vx":"$v$ (cm.s$^{-1}$)",
  "u":"$\\gamma \\beta$",
  "p":"$p$ (Ba)",
  "h":"$h$ ($c^2$)",
  "T":"$\\Theta = p/\\rho c^2$"}

# IO
def openData_withtime(it, key='Last'):
  data = readData(key, 0, True)
  t0 = data['t'][0]
  data = readData(key, it, True)
  t = data['t'][0] # beware more complicated sims with adaptive time step ?
  dt = t-t0
  return data, t, dt

def read_physInput(key='Last'):
  dir_path = f'../../results/{key}/phys_input.MWN'
  setupEnv(dir_path)

# error criterion for AMR
def calc_LoehnerError(sigma, eps=1e-2):
  '''Get refinement criterion for a given sigma(U)
  Boundaries may nopt be calculated properly'''
  sig0   = sigma[0]
  sigN   = sigma[-1]
  ext_sig = np.concatenate((np.array([sigma[0]]), sigma, np.array([sigma[-1]])))
  dSig   = np.diff(ext_sig)
  dSigL  = dSig[:-1]
  dSigR  = dSig[1:]
  sigRef = np.array([np.abs(ext_sig[:-1][i-1]) + 2.*np.abs(sigma[i]) + np.abs(ext_sig[1:][i+1]) for i in range(len(sigma))])
  chi = np.sqrt( (dSigR - dSigL)**2 / (np.abs(dSigR)+np.abs(dSigL)+eps*sigRef)**2)
  return chi

# Animation
def plotAnimated(var, itmin=0, itmax=None, filename='out.gif', key='Last'):
  '''
  Create animated plot of variable var over a whole simulation run
  func must create a plot
  '''
  fig, ax = plt.subplots()
  line1 = [None]
  #title = ax.text(0.5,1., "", transform=ax.transAxes, ha="center")
  ax.set_ylabel(labels[var])

  its = []
  dir_path = '../../results/%s/' % (key)
  for path in os.scandir(dir_path):
    if path.is_file:
      fname = path.name
      if fname.endswith('.out'):
        it = int(fname[-14:-4])
        its.append(it)
  its = sorted(its)
  ittemp = its
  if itmin and not(itmax):
    ittemp = its[itmin:]
  elif not(itmin) and itmax:
    ittemp = its[:itmax]
  elif itmin and itmax:
    ittemp = its[itmin:itmax]
  
  its = ittemp
  
  def init():
    data, t, dt = openData_withtime(itmin, key)
    x, z, tracvals = getData1D(data, var, x_norm=x_norm, z_norm=znorms[var])
    ax.set_title("it {},  ".format(it) + r"$t_{sim} = $" +"{:.3e} s".format(dt))
    ax.loglog(x, z, 'rx')
  
  def update(it):
    ax.lines.pop(0)
    data, t, dt = openData_withtime(it, key)
    x, z, tracvals = getData1D(data, var, x_norm=x_norm, z_norm=znorms[var])
    ax.loglog(x, z, 'rx')
    ax.relim()
    ax.autoscale_view()
    #ax.set_ylim(1.05*min(z), max(z))
    ax.set_title("it {},  ".format(it) + r"$t_{sim} = $" +"{:.3e} s".format(dt))

  ani = mpla.FuncAnimation(fig, update, init_func=init, frames=its, interval=400, blit=False)
  ani.save(filename)
  plt.close()

def plotAnimated_default(itmin=0, itmax=None, filename='out.gif', key='Last'):
  '''
  Create animated plot of similar to plotSim_default
  over a whole simulation run
  '''
  varlist = ["rho", "u", "p"]
  fig, axes = plt.subplots(3, 1, sharex=True)
  line1 = [None]
  line2 = [None]
  line3 = [None]
  for i, var in enumerate(varlist):
    axes[i].set_ylabel(labels[var])
    if var == "p":
      axes[i].set_autoscale_on(False)

  its = []
  dir_path = '../../results/%s/' % (key)
  for path in os.scandir(dir_path):
    if path.is_file:
      fname = path.name
      if fname.endswith('.out'):
        it = int(fname[-14:-4])
        its.append(it)
  its = sorted(its)
  ittemp = its
  if itmin and not(itmax):
    ittemp = its[itmin:]
  elif not(itmin) and itmax:
    ittemp = its[:itmax]
  elif itmin and itmax:
    ittemp = its[itmin:itmax]
  
  its = ittemp

  def init():
    data, t, dt = openData_withtime(itmin, key)
    axes[0].set_title("it {},  ".format(it) + r"$t_{sim} = $" +"{:.3e} s".format(dt))
    for i, var in enumerate(varlist):
      x, z, tracvals = getData1D(data, var, x_norm=x_norm, z_norm=znorms[var])
      axes[i].loglog(x, z, 'rx')
    
  def update(it):
    data, t, dt = openData_withtime(it, key)
    axes[0].set_title("it {},  ".format(it) + r"$t_{sim} = $" +"{:.3e} s".format(dt))
    for i, var in enumerate(varlist):
      axes[i].lines.pop(0)
      x, z, tracvals = getData1D(data, var, x_norm=x_norm, z_norm=znorms[var])
      axes[i].loglog(x, z, 'rx')
  
  ani = mpla.FuncAnimation(fig, update, init_func=init, frames=its, interval=400, blit=False)
  ani.save(filename)
  plt.close()

# Default plot
def plotSim_default(it, key='Last'):
  '''
  Plots density, velocity and pressure in loglog scale
  By default will look for phys*it*.out in the Last/ folder
  '''
  data, t, dt = openData_withtime(it, key)
  read_physInput(key)
  varlist = ["rho", "u", "p"]
  
  title = f"it {it},  " + r"$t_{sim}/t_{c} = $" + f"{dt/t_c}"

  f, axes = plotMulti(data, varlist, logz=varlist, znorms=znorms, line=True, labels=labels, x_norm=x_norm)
  axes[-1].set_xlabel("$r$ (cm)")
  f.suptitle(title)
  plt.tight_layout()
  plt.show()

def plot_energy(it, key='Last'):
  '''
  Plots energy densities: kinetic, internal (p), rest-mass (rho) in units c**2
  '''
  data, t, dt = openData_withtime(it, key) 
  x_norm = c
  varlist = ["ekin", "eint", "erest"]
  title = "it {},  ".format(it) + r"$t_{sim} = $" +"{:.3e} s".format(dt)
  plt.figure()
  ax = plt.gca()
  for var in varlist:
    z_norm = znorms[var]
    label = labels[var]
    x, z, tracvals = getData1D(data, var, x_norm=c, z_norm=znorms[var])
    plt.loglog(x, z, label=label)
    plt.xlabel("$r$ (cm)")
    plt.ylabel("$\\rho_0 c^2$")
  plt.title(title)
  plt.legend()
  plt.tight_layout()
  plt.show()

# plot1D wrapper
def plot_wrapped(it, var, key='Last'):
  '''
  Wraps plot1D function with scaling and legend for easy 1 var plotting
  '''
  data, t, dt = openData_withtime(it, key)
  title = "it {},  ".format(it) + r"$t_{sim} = $" +"{:.3e} s".format(dt)

  x_norm = c
  if var == "chi":
    z_norm = 1.
    label = "$\chi$"
    x, z, tracvals = getData1D(data, "D")
    chi = calc_LoehnerError(z)
    plt.figure()
    ax = plt.gca()
    ax.set_ylabel(label)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.plot(x, chi, 'k',zorder=1)
  else:  
    z_norm = znorms[var]
    label = labels[var]
    plot1D(data, var, logx=True, logz=True, z_norm=z_norm, line=True, label=label, x_norm=x_norm)


  plt.xlabel("$r$ (cm)")
  plt.title(title)
  plt.tight_layout()
  plt.show()


def plot_slope(it, var, key='Last'):
  '''
  Plots the log slope of the chosen variable, following the same rules as plot_wrapped
  '''
  data, t, dt = openData_withtime(it, key)
  x_norm = c
  z_norm = znorms[var]
  varname = legends[var]
  label = "d$\\log$" + varname + "/d$\\log r$"

  title = "it {},  ".format(it) + r"$t_{sim} = $" +"{:.3e} s".format(dt)
  plotSlope1D(data, var, line=True, label=label, x_norm=x_norm)
  plt.xlabel("$r$ (cm)")
  plt.title(title)
  plt.tight_layout()
  plt.show()


# Time-evolution graphs
# --------------------------------------------------------------------------------------------------
def timeplot_resolution(key='Last'):
  '''
  Plots the evolution of resolution with time
  '''
  folder = '../../results/%s' % (key)
  allfiles = os.listdir(folder)
  phys_files = sorted(filter(lambda x: x.startswith('phys'), allfiles))
  
  time = []
  res_avg = []
  res_min = []
  res_max = []
  for file in phys_files:
    fname = key + '/' + file
    data = readData(fname)
    t = data['t'][0] # beware more complicated sims with adaptive time step ?
    x = data['x']
    dx = data['dx']
    xmin = x.min()
    xmax = x.max()
    N = data.shape[0]
    res = (xmin-xmax)*c/N
    maxres = dx.max()
    minres = dx.min()

    time.append(t)
    res_avg.append(res)
    res_min.append(minres)
    res_max.append(maxres)
  
  plt.semilogy(time, res_avg, label='average')
  plt.semilogy(time, res_min, label='min')
  plt.semilogy(time, res_max, label='max')
  plt.xlabel('t (s)')
  plt.ylabel('resolution (cm)')
  plt.legend()
  plt.tight_layout()
  plt.show()
