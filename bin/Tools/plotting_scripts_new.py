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
from phys_functions_shells import *

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
var_label = {'R':'$r$ (cm)', 'v':'$\\beta$', 'u':'$\\gamma\\beta$',
  'V':'$V$ (cm$^3$)', 'Nc':'$N_{cells}$', 'ShSt':'$\\Gamma_{ud}-1$',
  'Emass':'$E_{r.m.}$', 'Ekin':'$E_k$', 'Eint':'E_{int}',
  'Rct':'$(r - ct)/R_0$ (cm)', 'Rcd':'$r - r_{cd}$ (cm)',
  'vcd':"$\\beta - \\beta_{cd}$", 'epsth':'$\\epsilon_{th}$'}
#var_legends = {}

# Time plots
# --------------------------------------------------------------------------------------------------
def plot_timeseries(var, key='Last',
  tscaling='t0', analytics=True, crosstimes=True,
  logt=False, logy=True, **kwargs):
  '''
  Creates time series plot. time scaling options: None (s), t0, it
  No theoretical values in it plotting mode
  '''
  env  = MyEnv(get_physfile(key))
  theoretical = {
    "u":[[env.u],['$u_{th}$']],
    "ShSt":[[env.lfac34-1., env.lfac21-1.],
      ['$\\Gamma_{34}-1$', '$\\Gamma_{21}-1$']],
    "v":[[env.betaRS, env.betaFS],
      ['$\\beta_{RS}$', '$\\beta_{FS}$']],
    "vcd":[[env.beta-env.betaRS, env.betaFS-env.beta],
      ['$\\beta_{CD}-\\beta_{RS}$', '$\\beta_{FS}-\\beta_{CD}$']]
  }

  
  data = get_timeseries(var, key)
  title = env.runname
  fig = plt.figure()
  ax = plt.gca()

  if tscaling == 'it':
    t = data.index
    tlabel = 'it'
    crosstimes = False
    analytics  = False
  elif tscaling == 't0':
    t = data['time'].to_numpy()/env.t0 + 1.
    tlabel = '$t/t_0$'
    tRS = env.tRS/env.t0 + 1.
    tFS = env.tFS/env.t0 + 1.
  else:
    t = data['time'].to_numpy()
    tlabel = '$t_{sim}$ (s)'
    tRS = env.tRS
    tFS = env.tFS

  ax.set_xlabel(tlabel)
  ylabel = var_label[var]
  ax.set_ylabel(ylabel)
  varlist = data.keys()[1:]
  for n, varname in enumerate(varlist):
    legend = '$' + varname + '$'
    ax.plot(t[1:], data[varname][1:], c=plt.cm.Paired(n), label=legend, **kwargs)
    if analytics:
      if var in []:
        exp_val = get_analytical(varname, t, env)
        ax.plot(t[1:], exp_val[1:], c=plt.cm.Paired(n), ls=':', lw=.8)
      elif var in theoretical:
        expec = theoretical[var]
        for val, name in zip(expec[0], expec[1]):
          ax.axhline(val, color='k', ls='--', lw=.6)
          ax.text(1.02, val, name, color='k', transform=ax.get_yaxis_transform(),
            ha='left', va='center')

  if logt:
    ax.set_xscaling('log')
  if logy:
    ax.set_yscale('log')

  if crosstimes:
    ax.axvline(tRS, color='r')
    ax.text(tRS, 1.05, '$t_{RS}$', color='r', transform=ax.get_xaxis_transform(),
      ha='center', va='top')
    ax.axvline(tFS, color='r')
    ax.text(tFS, 1.05, '$t_{FS}$', color='r', transform=ax.get_xaxis_transform(),
      ha='center', va='top')
  
  plt.legend()
  fig.suptitle(title)
  
  


# Snapshot plots
# --------------------------------------------------------------------------------------------------
def default_comparison(it, key='Last'):
  f, axes = plt.subplots(3, 1, sharex=True, figsize=(6,6), layout='constrained')
  varlist = ['rho', 'u', 'p']
  for var, k, ax in zip(varlist, range(3), axes):
    title, scatter = ax_snapshot(var, it, key, theory=True, ax=ax)
    if k != 2: ax.set_xlabel('')
  
  f.suptitle(title)
  plt.legend(*scatter.legend_elements(), bbox_to_anchor=(1.02, 0), loc='lower left', borderaxespad=0.)


def ax_snapshot(var, it, key='Last', theory=False, ax=None,
    logx=False, logy=True, xscaling='R0', yscaling='code', **kwargs):
  '''
  Create line plot on given ax of a matplotlib figure
  /!\ find an updated way to normalize all variables properly with units
  '''
  if ax is None:
    plt.figure()
    ax = plt.gca()
  
  # Legends and labels
  var_exp = {
  "x":"$r$", "dx":"$dr$", "rho":"$\\rho$", "vx":"$\\beta$", "p":"$p$",
  "D":"$\\gamma\\rho$", "sx":"$\\gamma^2\\rho h$", "tau":"$\\tau$",
  "trac":"tracer", "Sd":"", "gmin":"$\\gamma_{min}$", "gmax":"$\\gamma_{max}$", "zone":"",
  "T":"$\\Theta$", "h":"$h$", "lfac":"$\\gamma$", "u":"$\\gamma\\beta$",
  "Eint":"$e$", "Ekin":"$e_k$", "Emass":"$\\rho c^2$", "dt":"dt", "res":"dr/r"
  }
  units_CGS  = {
  "x":" (cm)", "dx":" (cm)", "rho":" (g cm$^{-3}$)", "vx":"", "p":" (Ba)",
  "D":"", "sx":"", "tau":"", "trac":"", "Sd":"", "gmin":"", "gmax":"",
  "T":"", "h":"", "lfac":"", "u":"", "zone":"", "dt":" (s)", "res":"",
  "Eint":" (erg cm$^{-3}$)", "Ekin":" (erg cm$^{-3}$)", "Emass":" (erg cm$^{-3}$)"
  }
  auth_theory = ["rho", "vx", "u", "p", "D", "sx", "tau", "T", "h", "lfac", "Ekin", "Eint", "Emass"]
  nonlog_var = ['u', 'trac', 'Sd', 'gmin', 'gmax', 'zone']
  if var not in auth_theory:
    theory = False
  if var in nonlog_var:
    logy = False
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
  
  physpath = get_physfile(key)
  env = MyEnv(physpath)

  units = {}
  xscale = 1.
  if xscaling == 'code':
    xlabel = '$r$ (ls)'
  elif xscaling == 'CGS':
    xlabel = '$r$ (cm)'
    xscale = c_
  elif xscaling == 'R0':
    xlabel = '$r/R_0$'
    xscale = c_/env.R0
  elif xscaling == 'Rcd':
    xlabel = '$r/R_{cd}$'
    xscale = 1./get_radius_zone(df, 2)
  else:
    xlabel = '$r$'
  if yscaling == 'code':
    units  = get_normunits('c', 'Norm')
    yscale = 1.
  elif yscaling == 'CGS':
    units = units_CGS
    yscale = get_varscaling(var, getattr(env, env.rhoNorm))
  ylabel = (var_exp[var] + units[var]) if units[var] else var_exp[var]
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  
  df, t, dt = openData_withtime(key, it)
  n = df["zone"]
  Rcd = get_radius_zone(df, 2)*c_
  dRcd = (Rcd - env.R0)/env.R0
  rc_str= reformat_scientific(f"{dRcd:.3e}")
  title = f"it {it}, dt = {dt:.3f} s, $(R_{{cd}}-R_0)/R_0= {rc_str}$"

  x, y = get_variable(df, var)
  if theory:
    r = x*c_
    y_th = get_analytical(var, t, r, env)
    y_th *= yscale
    ax.plot(x*xscale, y_th, 'k--', zorder=3)

  x *= xscale
  y *= yscale
  ax.plot(x, y, 'k', zorder=1)
  scatter = ax.scatter(x, y, c=n, lw=1, zorder=2, cmap='Paired', **kwargs)
  
  if logx:
    ax.set_xscale('log')
  if logy:
    ax.set_yscale('log')

  return title, scatter


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

