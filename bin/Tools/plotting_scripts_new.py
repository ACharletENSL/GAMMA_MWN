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
from scipy import integrate

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

# Legends and labels
var_exp = {
  "x":"$r$", "dx":"$dr$", "rho":"$\\rho$", "vx":"$\\beta$", "p":"$p$",
  "D":"$\\gamma\\rho$", "sx":"$\\gamma^2\\rho h$", "tau":"$\\tau$",
  "trac":"tracer", "trac2":"wasSh", "Sd":"shock id", "gmin":"$\\gamma_{min}$", "gmax":"$\\gamma_{max}$", "zone":"",
  "T":"$\\Theta$", "h":"$h$", "lfac":"$\\gamma$", "u":"$\\gamma\\beta$",
  "Ei":"$e_{int}$", "Ekin":"$e_k$", "Emass":"$\\rho c^2$", "dt":"dt", "res":"dr/r", "Tth":"$T_\\theta$",
  "numax":"$\\nu'_{max}$", "numin":"$\\nu'_{min}$", "B":"$B'$", "Pmax":"$P'_{max}$",
  "gm":"$\\gamma_0$", "gb":"$\\gamma_1$", "gM":"$\\gamma_2$", "Lp":"$L'$", "L":"$L$",
  "num":"$\\nu'_m$", "nub":"$\\nu'_b$", "nuM":"$\\nu'_M$", "inj":"inj", "fc":"fc",
  "T0":"$T_{0,k}$", "Tej":"$T_{ej,k}$", "Tpk":"$T_{pk}$", "tc":"$t_c$",
  "V4":"$\\Delta V^{(4)}$", "V3":"$\\Delta V^{(3)}$", "V4r":"$\\Delta V^{(4)}_c$",
  }
units_CGS  = {
  "x":" (cm)", "dx":" (cm)", "rho":" (g cm$^{-3}$)", "vx":"", "p":" (Ba)",
  "D":"", "sx":"", "tau":"", "trac":"", "trac2":"", "Sd":"", "gmin":"", "gmax":"",
  "T":"", "h":"", "lfac":"", "u":"", "zone":"", "dt":" (s)", "res":"", "Tth":" (s)",
  "Ei":" (erg cm$^{-3}$)", "Ek":" (erg cm$^{-3}$)", "M":" (g)", "Pmax":" (erg s$^{-1}$cm$^{-3}$)", "Lp":" (erg s$^{-1}$)",
  "numax":" (Hz)", "numin":" (Hz)", "B":" (G)", "gm":"", "gb":"", "gM":"", "L":" (erg s$^{-1}$) ",
  "num":" (Hz)", "nub":" (Hz)", "nuM":" (Hz)", "inj":"", "fc":"", "tc":" (s).",
  "T0":" (s)", "Tej": " (s)", "Tpk":" (s)", "V4":" (cm$^3$s)", "V3":" (cm$^3$)", "V4r":" (cm$^3$s)"
  }
var_label = {'R':'$r$ (cm)', 'v':'$\\beta$', 'u':'$\\gamma\\beta$',
  'f':"$n'_3/n'_2$", 'rho':"$n'$", 'rho3':"$n'_3$", 'rho2':"$n'_2$", 'Nsh':'$N_{cells}$',
  'V':'$V$ (cm$^3$)', 'Nc':'$N_{cells}$', 'ShSt':'$\\Gamma_{ud}-1$', "Econs":"E",
  'ShSt ratio':'$(\\Gamma_{34}-1)/(\\Gamma_{21}-1)$', "Wtot":"$W_4 - W_1$",
  'M':'$M$ (g)', 'Msh':'$M$ (g)', 'Ek':'$E_k$ (erg)', 'Ei':'$E_{int}$ (erg)',
  'E':'$E$ (erg)', 'Etot':'$E$ (erg)', 'Esh':'$E$ (erg)', 'W':'$W_{pdV}$ (erg)',
  'Rct':'$(r - ct)/R_0$ (cm)', 'D':'$\\Delta$ (cm)', 'u_i':'$\\gamma\\beta$',
  'vcd':"$\\beta - \\beta_{cd}$", 'epsth':'$\\epsilon_{th}$', 'pdV':'$pdV$'
  }
cool_vars = ["inj", "fc", "gm", "gb", "gM", "num", "nub", "nuM"]
nonlog_var = ['u', 'trac', 'Sd', 'zone', 'trac2']
def get_normunits(xnormstr, rhonormsub):
  rhonormsub  = '{' + rhonormsub +'}'
  CGS2norm = {" (s)":"$/T_0$",  " (s).":"$/t_{c,RS}$", " (g)":" (g)", " (Hz)":f"$/\\nu'_0$", " (eV)":" (eV)", " (G)":"$/B'_3$",
  " (cm)":" (ls)" if xnormstr == 'c' else "$/"+xnormstr+"$", " (g cm$^{-3}$)":f"$/\\rho_{rhonormsub}$",
  " (Ba)":f"$/\\rho_{rhonormsub}c^2$", " (erg cm$^{-3}$)":f"$/\\rho_{rhonormsub}c^2$",
  " (erg s$^{-1}$)":"$/L'_0$"," (erg s$^{-1}$) ":"$/L_0$", " (cm$^3$s)":"$/V_0t_0$", " (cm$^3$)":"$/V_0$",
  " (cm$^3$)":" (cm$^3$)", " (erg s$^{-1}$cm$^{-3}$)":"$/P'_0$"}
  return {key:(CGS2norm[value] if value else "") for (key,value) in units_CGS.items()}
def get_varscaling(var, env):
  rhoNorm = env.rhoscale
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


# Cells timelapse plots
# --------------------------------------------------------------------------------------------------
#

def plotduo_behindShocks(varlist, key='Last', itmax=None, tscaling='it', yscalings=['code', 'code'], logt=False, logys=[False, False],
    names = ['RS', 'FS'], lslist = ['-', ':'], colors = ['r', 'b'], **kwargs):
  '''
  Plots 2 chosen variables as measured in the cell behind each shock front
  '''

  its = np.array(dataList(key, itmax=itmax))[1:]
  its, time, yRS, yFS = get_behindShock_vals(key, varlist, its)
  y = np.array([yRS, yFS])
  y = y.transpose((1,0,2)) # reshape to order var -> front -> it

  env = MyEnv(key)
  fig, ax1 = plt.subplots()
  ax2 = ax1.twinx()
  axes = [ax1, ax2]
  varexps = [var_exp[var] for var in varlist]
  dummy_lst = [plt.plot([],[], ls=ls, color='k')[0] for ls in lslist]
  dummy_col = [plt.plot([],[],ls='-',color=c)[0] for c in colors]
  legendlst = plt.legend(dummy_lst, varexps, bbox_to_anchor=(1.0, 1.02),
    loc='lower right', borderaxespad=0.)
  plt.legend(dummy_col, names, bbox_to_anchor=(0., 1.02),
    loc='lower left', borderaxespad=0.)
  fig.add_artist(legendlst)

  if tscaling == 'it':
    t = its
    tlabel = 'it'
  elif tscaling == 't0':
    t = time/env.t0 + 1.
    tlabel = '$t/t_0$'
  else:
    t = time
    tlabel = '$t_{sim}$ (s)'
  ax1.set_xlabel(tlabel)

  for var, yvar, ax, yscaling, logy, varexp, ls in zip(varlist, y, axes, yscalings, logys, varexps, lslist):
    if yscaling == 'code':
      units  = get_normunits('c', 'Norm')
      yscale = 1.
      if var.startswith('nu'):
        yscale = 1./env.nu0p
      elif var == 'Lp':
        yscale = 1./env.L0p
      elif var == 'L':
        yscale = 1/env.L0
      elif var == 'B':
        yscale = 1./env.Bp
      elif var.startswith('T'):
        yscale = 1./env.T0
      elif "V4" in var:
        yscale = 1/(env.V0*env.t0)
      elif var == "V3":
        yscale = 1/env.V0
      elif var == "Pmax":
        yscale = 1/env.Pmax0
      elif var == "tc":
        yscale = 1/env.tc
    elif yscaling == 'CGS':
      units = units_CGS
      yscale = get_varscaling(var, env)
    if var == 'i':
      ylabel = 'i'
    else:
      ylabel = (varexp + units[var]) if units[var] else varexp

    yvar *= yscale
    if var.startswith('T'):
      ylabel = '$\\bar{T}' + var_exp[var][2:]
      yvar -= env.Ts/env.T0
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='y')

    for val, color in zip(yvar, colors):
      ax.plot(t, val, ls=ls, color=color, **kwargs)

    if logt:
      ax.set_xscale('log')
    if logy:
      ax.set_yscale('log')




def ax_behindShocks_timeseries(var, key='Last', itmax=None, tscaling='it', yscalings='code', ax_in=None, logt=False, logy=False, **kwargs):
  '''
  Plots a time series of var in cells behind the shocks
  '''
  if var == 'i':
    yscaling = 'code'
  env  = MyEnv(key)

  if ax_in is None:
    plt.figure()
    ax = plt.gca()
  else:
    ax = ax_in
  
  # create dataset
  its = np.array(dataList(key, itmax=itmax))[1:]
  its, time, yRS, yFS = get_behindShock_value(key, var, its)
  y = np.array([yRS, yFS])

  # plot 
  if tscaling == 'it':
    t = its
    tlabel = 'it'
  elif tscaling == 't0':
    t = time/env.t0 + 1.
    tlabel = '$t/t_0$'
  else:
    t = time
    tlabel = '$t_{sim}$ (s)'
  ax.set_xlabel(tlabel)

  if yscaling == 'code':
    units  = get_normunits('c', 'Norm')
    yscale = 1.
    if var.startswith('nu'):
      yscale = 1./env.nu0p
    elif var == 'Lp':
      yscale = 1./env.L0p
    elif var == 'B':
      yscale = 1./env.Bp
    elif var.startswith('T'):
      yscale = 1./env.T0
  elif yscaling == 'CGS':
    units = units_CGS
    yscale = get_varscaling(var, env)
  if var == 'i':
    ylabel = 'i'
  else:
    ylabel = (var_exp[var] + units[var]) if units[var] else var_exp[var]
  y *= yscale
  if var.startswith('T'):
      ylabel = '$\\bar{T}' + var_exp[var][2:]
      yvar -= env.Ts/env.T0
  ax.set_ylabel(ylabel)

  ax.plot(t, y[0], label='RS', **kwargs)
  ax.plot(t, y[1], label='FS', **kwargs)
  if logt:
    ax.set_xscale('log')
  if logy:
    ax.set_yscale('log')
  
  if ax_in is None:
    plt.legend()

def plot_injection_timeseries(i, key='Last', tscaling='it', **kwargs):
  plt.figure()
  ax = plt.gca()
  varlist = ['trac2', 'fc', 'inj']
  its = np.array(dataList(key))[1:]
  t, y = get_cell_timevalues(key, varlist, i, its)
  # plot 
  if tscaling == 'it':
    t = its
    tlabel = 'it'
  elif tscaling == 't0':
    t = time/env.t0 + 1.
    tlabel = '$t/t_0$'
  else:
    t = time
    tlabel = '$t_{sim}$ (s)'
  ax.set_xlabel(tlabel)
  
  for k, var in enumerate(varlist):
    ax.plot(t, y[k], label=var, **kwargs)
  plt.legend()

def plot_edistrib_timeseries(i, key='Last', tscaling='it', **kwargs):
  plt.figure()
  ax = plt.gca()
  varlist = ['gmin', 'gmax', 'gm', 'gb', 'gM']
  its = np.array(dataList(key))[1:]
  t, y = get_cell_timevalues(key, varlist, i, its)
  # plot 
  if tscaling == 'it':
    t = its
    tlabel = 'it'
  elif tscaling == 't0':
    t = time/env.t0 + 1.
    tlabel = '$t/t_0$'
  else:
    t = time
    tlabel = '$t_{sim}$ (s)'
  ax.set_xlabel(tlabel)
  
  for k, var in enumerate(varlist):
    ls = '-'
    if var in ['gm', 'gb', 'gM']:
      ls = ':'
    ax.semilogy(t, y[k], label=var_exp[var], linestyle=ls, **kwargs)
  plt.legend()


def ax_cell_timeseries(var, i, key='Last', tscaling='it', yscaling='code', ax_in=None, logt=False, logy=False, **kwargs):
  '''
  Plots a time series of var in cell i
  Cells are counted in the shells only
  '''
  
  env  = MyEnv(key)
  Nshtot = env.Nsh1+env.Nsh4
  if i > Nshtot:
    print(f"No corresponding cell in shells, i must be lower than {Nshtot}")

  if ax_in is None:
    plt.figure()
    ax = plt.gca()
  else:
    ax = ax_in
  
  # create dataset
  its = np.array(dataList(key))
  if var in cool_vars:
    its = its[1:]
  data = np.zeros(its.shape)
  time, y = get_cell_timevalues(key, var, i, its)

  # plot 
  if tscaling == 'it':
    t = its
    tlabel = 'it'
  elif tscaling == 't0':
    t = time/env.t0 + 1.
    tlabel = '$t/t_0$'
  else:
    t = time
    tlabel = '$t_{sim}$ (s)'
  ax.set_xlabel(tlabel)

  if yscaling == 'code':
    units  = get_normunits('c', 'Norm')
    yscale = 1.
    if var.startswith('nu'):
      yscale = 1./env.nu0p
    elif var == 'Lp':
      yscale = 1./env.L0p
    elif var == 'B':
      yscale = 1./env.Bp
  elif yscaling == 'CGS':
    units = units_CGS
    yscale = get_varscaling(var, env)
  
  y *= yscale

  ax.plot(t, y, label=var_exp[var], **kwargs)
  if logt:
    ax.set_xscale('log')
  if logy:
    ax.set_yscale('log')
  
  if ax_in == None:
    ylabel = (var_exp[var] + units[var]) if units[var] else var_exp[var]
    ax.set_ylabel(ylabel)


# Time plots
# --------------------------------------------------------------------------------------------------
# to add a new variable, modify
# this file: var_label, theoretical
# run_analysis: get_timeseries
# phys_function_shells: time_analytical


lslist  = ['-', '--', '-.', ':']

def compare_runs(var, keylist, tscaling='t0',
    logt=False, logy=False, **kwargs):
  '''
  Compare time plots for different runs
  '''
  names   = []
  leglist = []
  dummy_lst = []
  fig = plt.figure()
  ax = plt.gca()
  
  for k, key in enumerate(keylist):
    name, legend = ax_timeseries(var, key, ax, tscaling=tscaling, 
      yscaling=False, theory=False, crosstimes=False,
      logt=logt, logy=logy, ls=lslist[k]) 
    names.append(name)
    leglist.append(legend)
    dummy_lst.append(plt.plot([],[],c='k',ls=lslist[k])[0])
  leg = leglist[0]
  legend1 = plt.legend(leg[0], leg[1])
  plt.legend(dummy_lst, names, bbox_to_anchor=(1.0, 1.02),
    loc='lower right', borderaxespad=0., ncol=len(names))
  ax.add_artist(legend1)
  plt.tight_layout()

def plot_conservation(var, key='Last'):
  plt.figure()
  ax = plt.gca()
  if var == 'M':
    varlist = ['M', 'Msh']
  elif var == 'E':
    varlist = ['E', 'Etot']
  for name in varlist:
    title, legend = ax_timeseries(name, key, ax_in=ax)
  plt.title(title)
  plt.legend(legend[0], legend[1])
  plt.tight_layout()

def ax_timeseries(var, key='Last', ax_in=None,
  tscaling='t0', yscaling=False,
  theory=True, crosstimes=True, reldiff=False,
  logt=False, logy=False, **kwargs):
  '''
  Creates time series plot. time scaling options: None (s), t0, it
  No theoretical values in it plotting mode
  Add scaling to y values for better runs comparison -> which scalings?
  '''
  
  if ax_in is None:
    plt.figure()
    ax = plt.gca()
  else:
    ax = ax_in
  
  env  = MyEnv(key)
  if env.geometry != 'cartesian':
    crosstimes = False
  theoretical = {
    "u":[[env.u],['$u_{th}$']],
    "ShSt":[[env.lfac34-1., env.lfac21-1.],
      ['$\\Gamma_{34}-1$', '$\\Gamma_{21}-1$']],
    "ShSt ratio":[[(env.lfac34-1.)/(env.lfac21-1.)], [""]],
    "v":[[env.betaRS, env.betaFS],
      ['$\\beta_{RS}$', '$\\beta_{FS}$']],
    "vcd":[[env.beta-env.betaRS, env.betaFS-env.beta],
      ['$\\beta_{CD}-\\beta_{RS}$', '$\\beta_{FS}-\\beta_{CD}$']],
    "rho":[[env.rho3, env.rho2], ["$n'_3$", "$n'_2$"]],
    "rho3":[[env.rho3], ["$n'_3$"]],
    "rho2":[[env.rho2], ["$n'_2$"]],
    "f":[[env.rho3/env.rho2], [""]]
  }
  correct_varlabels = {
    "D":"\\Delta",
    "rho":"\\rho"
  }
  if var not in theoretical:
    reldiff = False
  if reldiff:
    theory = False
    #logy = True
  if var.startswith("N"):
    theory = False
  
  data = get_timeseries(var, key)
  if var == 'Nc':
    theory = False
  if theory and (var not in theoretical):
    expec_vals = time_analytical(var, data['time'].to_numpy(), env)
    dummy_lst = [plt.plot([],[],c='k',ls='-')[0], plt.plot([],[],c='k',ls=':')[0]]
  title = env.runname

  if tscaling == 'it':
    t = data.index
    tlabel = 'it'
    crosstimes = False
    theory  = False
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
  varlist = data.keys()[1:]
  yscale = 1.
  if var.startswith('E') or var.startswith('W'):
    yscaling = True
    yscale = (env.Ek4 + env.Ek1)/2.
    ylabel = '$W/E_0$' if var.startswith("W") else "$E/E_0$"
  if reldiff:
    ylabel = '$(X - X_{th})/X_{th}$'
  ax.set_ylabel(ylabel)
  for n, varname in enumerate(varlist):
    y = data[varname].multiply(1./yscale)
    vsplit = varname.split('_')
    if vsplit[0] in correct_varlabels:
      varname = varname.replace(vsplit[0], correct_varlabels[vsplit[0]])
    varlabel = "$" + varname + "$"
    
    if reldiff:
      expected = theoretical[var][0][n]
      y = derive_reldiff(y, expected)
    ax.plot(t, y, c=plt.cm.Paired(n), label=varlabel, **kwargs)
    if theory:
      if var in theoretical:
        expec = theoretical[var]
        for val, name in zip(expec[0], expec[1]):
          ax.axhline(val, color='k', ls='--', lw=.6)
          ax.text(1.02, val, name, color='k', transform=ax.get_yaxis_transform(),
            ha='left', va='center')
      else:
        ax.plot(t, expec_vals[n]/yscale, c='k', ls=':', lw=.8)
        th_legend = ax.legend(dummy_lst, ['data', 'analytical'], fontsize=11)

  if logt:
    ax.set_xscaling('log')
  if logy:
    if reldiff:
      ax.set_yscale('symlog')
    else:
      ax.set_yscale('log')

  if crosstimes:
    ax.axvline(tRS, color='r')
    ax.text(tRS, 1.05, '$t_{RS}$', color='r', transform=ax.get_xaxis_transform(),
      ha='center', va='top')
    ax.axvline(tFS, color='r')
    ax.text(tFS, 1.05, '$t_{FS}$', color='r', transform=ax.get_xaxis_transform(),
      ha='center', va='top')
  
  if ax_in is None:
    if len(varlist)>1:
      ax.legend(bbox_to_anchor=(1.02, 1.0), loc='upper left', borderaxespad=0.)
    elif reldiff:
      ax.legend(bbox_to_anchor=(0., 1.01), loc='lower left', borderaxespad=0.)
    if theory and (var not in theoretical):
      ax.add_artist(th_legend)
    plt.title(title)
    plt.tight_layout()
  else:
    ax.legend()
    legend = ax.get_legend_handles_labels()
    ax.legend('', frameon=False)
    return title, legend
  
  


# Snapshot plots
# --------------------------------------------------------------------------------------------------
# to add a new variable, modify
# this file: var_exp, units_CGS, (auth_theory, nonlog_var, get_varscaling if necessary)
# data_IO: get_variable
# phys_functions_shells: get_analytical

def article_series(name, its, key='cart_fid'):
  '''
  Series of snapshot constructed around 'name' prefix
  '''

def article_snap(name, it, key='cart_fid'):
  prim_snapshot(it, key, theory=True, xscaling='Rcd')
  figname = './figures/' + name + '_' + str(it) + 'png' 
  plt.savefig(figname)
  plt.close()

def prim_snapshot(it, key='Last', theory=False, xscaling='R0'):
  f, axes = plt.subplots(3, 1, sharex=True, figsize=(6,6), layout='constrained')
  varlist = ['rho', 'u', 'p']
  #scatlist= []
  for var, k, ax in zip(varlist, range(3), axes):
    title, scatter = ax_snapshot(var, it, key, theory, ax_in=ax, xscaling=xscaling)
    #scatlist.append(scatter)
    if k != 2: ax.set_xlabel('')
  
  f.suptitle(title)
  #scatter = scatlist[1]
  plt.legend(*scatter.legend_elements(), bbox_to_anchor=(1.02, 0), loc='lower left', borderaxespad=0.)

def rad_snapshot(it, key='Last', xscaling='n', distr='gamma'):
  Nk = 3
  f, axes = plt.subplots(Nk, 1, sharex=True, figsize=(6, 2*Nk), layout='constrained')
  varlist = ['Sd', 'gmax', 'gmin']
  yscalings = ['CGS', 'CGS', 'CGS']
  if distr == 'nu':
    varlist = ['Sd', 'numax', 'numin']
    yscalings = ['CGS', 'code', 'code']
  logs = [False, True, True]
  for var, k, ax in zip(varlist, range(Nk), axes):
    title, scatter = ax_snapshot(var, it, key, theory=False, ax_in=ax, xscaling=xscaling, yscaling=yscalings[k], logy=logs[k])
    if k != 2: ax.set_xlabel('')
  f.suptitle(title)
  plt.legend(*scatter.legend_elements(), bbox_to_anchor=(1.02, 0), loc='lower left', borderaxespad=0.)

def accel_snapshot(it, key='Last', xscaling='n'):
  Nk = 3
  f, axes = plt.subplots(Nk, 1, sharex=True, figsize=(6, 2*Nk), layout='constrained')
  varlist = ['Sd', 'trac2', 'gmax']
  logs = [False, False, True]
  yscalings = ['code', 'code', 'code']
  for var, k, ax in zip(varlist, range(Nk), axes):
    title, scatter = ax_snapshot(var, it, key, theory=False, ax_in=ax, xscaling=xscaling, yscaling=yscalings[k], logy=logs[k])
    if k != 3: ax.set_xlabel('')
  f.suptitle(title)
  plt.legend(*scatter.legend_elements(), bbox_to_anchor=(1.02, 0), loc='lower left', borderaxespad=0.)


def emission_snapshot(it, key='Last', xscaling='n'):
  Nk = 3
  f, axes = plt.subplots(Nk, 1, sharex=True, figsize=(6, 2*Nk), layout='constrained')
  varlist = ['nub', 'L', 'Tpk']
  logs = [True, True, False]
  #yscalings = ['CGS', 'code', 'code']
  for var, k, ax in zip(varlist, range(Nk), axes):
    title, scatter = ax_snapshot(var, it, key, theory=False, ax_in=ax, xscaling=xscaling, yscaling='code', logy=logs[k])
    if k != 3: ax.set_xlabel('')
  f.suptitle(title)
  plt.legend(*scatter.legend_elements(), bbox_to_anchor=(1.02, 0), loc='lower left', borderaxespad=0.)

def obs_snapshot(it, key='Last', xscaling='n'):
  Nk = 3
  f, axes = plt.subplots(Nk, 1, sharex=True, figsize=(6, 2*Nk), layout='constrained')
  varlist = ['B', 'nub', 'Lp']
  logs = [True, True, True]
  yscalings = ['CGS', 'code', 'code']
  for var, k, ax in zip(varlist, range(Nk), axes):
    title, scatter = ax_snapshot(var, it, key, theory=False, ax_in=ax, xscaling=xscaling, yscaling=yscalings[k], logy=logs[k])
    if k != 3: ax.set_xlabel('')
  f.suptitle(title)
  plt.legend(*scatter.legend_elements(), bbox_to_anchor=(1.02, 0), loc='lower left', borderaxespad=0.)

def snap_multi(varlist, it, key='Last', xscaling='n'):
  Nk = len(varlist)
  f, axes = plt.subplots(Nk, 1, sharex=True, figsize=(6, 2*Nk), layout='constrained')
  for var, k, ax in zip(varlist, range(Nk), axes):
    title, scatter = ax_snapshot(var, it, key, theory=False, ax_in=ax, xscaling=xscaling, yscaling='code', logy=True)
    if k != 3: ax.set_xlabel('')
  f.suptitle(title)
  plt.legend(*scatter.legend_elements(), bbox_to_anchor=(1.02, 0), loc='lower left', borderaxespad=0.)

def ax_snapshot(var, it, key='Last', theory=False, ax_in=None,
    logx=False, logy=True, xscaling='n', yscaling='code', **kwargs):
  '''
  Create line plot on given ax of a matplotlib figure
  /!\ find an updated way to normalize all variables properly with units
  '''

  if ax_in is None:
    plt.figure()
    ax = plt.gca()
    plt.suptitle(f'it {it}')
  else:
    ax=ax_in
  
  auth_theory = ["rho", "vx", "u", "p", "D", "sx", "tau",
    "T", "h", "lfac", "Ek", "Ei", "M", "B"]
  if var not in auth_theory:
    theory = False
  if var in nonlog_var:
    logy = False
  
  env = MyEnv(key)
  
  df, t, dt = openData_withtime(key, it)
  if it != 0:
    df = openData_withDistrib(key, it)
  n = df["zone"]
  Rcd = get_radius_zone(df, 2)*c_
  dRcd = (Rcd - env.R0)/env.R0

  rc_str= reformat_scientific(f"{dRcd:.3e}")
  title = env.runname + f", it {it}, $(R_{{cd}}-R_0)/R_0= {rc_str}$"
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
  elif xscaling == 'n':
    xlabel = 'n'
  else:
    xlabel = '$r$'
  if yscaling == 'code':
    units  = get_normunits('c', 'Norm')
    yscale = 1.
    if var.startswith('nu'):
      yscale = 1./env.nu0p
    elif var == 'Lp':
      yscale = 1./env.L0p
    elif var == "L":
      yscale = 1/env.L0
    elif var == 'B':
      yscale = 1./env.Bp
    elif var.startswith('T'):
      yscale = 1/env.T0
  elif yscaling == 'CGS':
    units = units_CGS
    yscale = get_varscaling(var, env)
  ylabel = (var_exp[var] + units[var]) if units[var] else var_exp[var]


  x = df['x']
  if xscaling == 'n':
    trac = df['trac'].to_numpy()
    i4  = np.argwhere((trac > 0.99) & (trac < 1.01))[:,0].min()
    x = df['i'] - i4
  y = get_variable(df, var)
  if theory:
    r = x*c_
    y_th = get_analytical(var, t, r, env)
    y_th *= yscale
    ax.plot(x*xscale, y_th, 'k--', zorder=3)

  x *= xscale
  y *= yscale
  if var.startswith('T'):
      ylabel = '$\\bar{T}' + var_exp[var][2:]
      y -= env.Ts/env.T0
  
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  ax.plot(x, y, 'k', zorder=1)
  scatter = ax.scatter(x, y, c=n, lw=1, zorder=2, cmap='Paired', **kwargs)
  
  if logx:
    ax.set_xscale('log')
  if logy:
    ax.set_yscale('log')
  
  return title, scatter


# spectras and lightcurves
# --------------------------------------------------------------------------------------------------
def construct_run_lightcurve(key, lognu, logx=False, logy=False, crosstimes=True, fig_in=None):
  '''
  Constructs the lightcurve at chosen frequency, displays iteration after iteration
  '''

  if fig_in:
    fig = fig_in
  else:
    fig = plt.figure()
  ax = fig.add_subplot(111)
  nuobs, Tobs, dTobs, env = get_radEnv(key)
  nuobs = np.array([env.nu0 * 10**lognu])
  Ton = env.Ts
  Tth = env.T0
  Tb = (Tobs-Ton)/Tth
  RS_rad, FS_rad = np.zeros((2, len(Tobs), len(nuobs)))

  plt.title(env.runname + f", log$_{{10}}(\\nu/\\nu_0) = {lognu}$")
  plt.xlabel("$\\bar{{T}}$")
  plt.ylabel("$\\nu F_\\nu / \\nu_0 F_0$")
  dummy_col = [plt.plot([],[],ls='-',color=c)[0] for c in ['r', 'b', 'k']]
  plt.legend(dummy_col, ['RS', 'FS', 'RS + FS'], bbox_to_anchor=(0.98, 0.98),
    loc='upper right', borderaxespad=0.)
  if logx:
    ax.set_xscale('log')
  if logy:
    ax.set_yscale('log')
  if crosstimes:
    TbRS = (env.TRS - Ton)/Tth
    TbFS = (env.TFS - Ton)/Tth
    ax.axvline(TbRS, color='r', ls='--', lw=.7)
    ax.text(TbRS, -.07, '$\\bar{T}_{RS}$', color='r', transform=ax.get_xaxis_transform(),
      ha='center', va='top')
    ax.axvline(TbFS, color='b', ls='--', lw=.7)
    ax.text(TbFS, -.07, '$\\bar{T}_{FS}$', color='b', transform=ax.get_xaxis_transform(),
      ha='center', va='top')
  plt.tight_layout()
  lineRS, = ax.plot(Tb, RS_rad, color='r')
  lineFS, = ax.plot(Tb, FS_rad, color='b')
  linesum,= ax.plot(Tb, (RS_rad+FS_rad), color='k')
  ax.text(0.95, 0.48, '',
      verticalalignment='bottom', horizontalalignment='right',
      transform=ax.transAxes, fontsize=15)
  ax.set_ylim([-0.2, 5.])
  plt.draw()
  its = np.array(dataList(key))[2:]
  for it in its:
    print(f'Generating flux from file {it}')
    ax.texts[-1].set_text(f'it {it}')
    df = openData_withDistrib(key, it)
    RS, FS = df_get_cellsBehindShock(df)
    if not RS.empty:
      RS_rad_i = cell_flux(RS, nuobs, Tobs, dTobs, env, func='analytic')
      RS_rad += RS_rad_i
    if not FS.empty:
      FS_rad_i = cell_flux(FS, nuobs, Tobs, dTobs, env, func='analytic')
      FS_rad += FS_rad_i
    sum_rad = RS_rad + FS_rad
    lineRS.set_ydata(RS_rad)
    lineFS.set_ydata(FS_rad)
    linesum.set_ydata(sum_rad)
    ax.texts[-1].set_text('it '+str(it))
    fig.canvas.draw()

def plot_run_lightcurve(key, lognu, logx=False, logy=False, crosstimes=True):
  '''
  Plots lightcurve at chosen frequency, generated from whole run
  '''

  plt.figure()
  ax = plt.gca()
  nuobs, Tobs, dTobs, env = get_radEnv(key)
  Ton = env.Ts
  Tth = env.T0
  Tb = (Tobs-Ton)/Tth
  its = np.array(dataList(key))[2:]
  rads = open_raddata(key)
  nuRS = [s for s in rads.keys() if 'RS' in s]
  nuFS = [s for s in rads.keys() if 'FS' in s]
  RS_rad = rads[nuRS].to_numpy()
  FS_rad = rads[nuFS].to_numpy()
  i_nu = find_closest(np.log10(nuobs/env.nu0), lognu)
  nu = nuobs[i_nu]
  light_RS = RS_rad[:, i_nu]
  light_FS = FS_rad[:, i_nu]
  
  ax.plot(Tb, light_RS/env.nu0F0, color='r', label='RS')
  ax.plot(Tb, light_FS/env.nu0F0, color='b', label='FS')
  ax.plot(Tb, (light_FS+light_RS)/env.nu0F0, color='k', label='RS + FS')
  if logx:
    ax.set_xscale('log')
  if logy:
    ax.set_yscale('log')
  if crosstimes:
    TbRS = (env.TRS - Ton)/Tth
    TbFS = (env.TFS - Ton)/Tth
    ax.axvline(TbRS, color='r', ls='--', lw=.7)
    ax.text(TbRS, -.07, '$\\bar{T}_{RS}$', color='r', transform=ax.get_xaxis_transform(),
      ha='center', va='top')
    ax.axvline(TbFS, color='b', ls='--', lw=.7)
    ax.text(TbFS, -.07, '$\\bar{T}_{FS}$', color='b', transform=ax.get_xaxis_transform(),
      ha='center', va='top')

  plt.title(env.runname + f", log$_{{10}}(\\nu/\\nu_0) = {np.log10(nu/env.nu0):.1f}$")
  plt.xlabel("$\\bar{{T}}$")
  plt.ylabel("$\\nu F_\\nu / \\nu_0 F_0$")
  plt.legend()
  plt.tight_layout()


def plot_run_spectrum(key, T):
  '''
  Plots instantaneous spectrum, generated from whole run
  '''
  plt.figure()
  nuobs, Tobs, dTobs, env = get_radEnv(key)
  its = np.array(dataList(key))[2:]
  rads = open_raddata(key)
  nuRS = [s for s in rads.keys() if 'RS' in s]
  nuFS = [s for s in rads.keys() if 'FS' in s]
  RS_rad = rads[nuRS].to_numpy()
  FS_rad = rads[nuFS].to_numpy()
  iTobs = find_closest(Tobs, T)
  spec_RS = RS_rad[iTobs]
  spec_FS = FS_rad[iTobs]
  
  plt.loglog(nuobs/env.nu0, spec_RS/env.nu0F0, color='r', label='RS')
  plt.loglog(nuobs/env.nu0, spec_FS/env.nu0F0, color='b', label='FS')
  plt.loglog(nuobs/env.nu0, (spec_FS+spec_RS)/env.nu0F0, color='k', label='RS + FS')
  plt.title(env.runname + f", $\\bar{{T}} = {T/env.T0:.2f}$")
  plt.xlabel("$\\nu/\\nu_0$")
  plt.ylabel("$\\nu F_\\nu / \\nu_0 F_0$")
  plt.legend()
  plt.tight_layout()

def plot_df_spectrum(key, it, T, func='analytic'):
  '''
  Plot instantaneous spectrum of iteration it of run key,
  observed at observer time T
  '''
  plt.figure()
  df = openData_withDistrib(key, it)
  nuobs, Tobs, dTobs, env = get_radEnv(key)
  RS_rad, FS_rad = df_get_rads(df, nuobs, Tobs, dTobs, env, func)
  iTobs = find_closest(Tobs, T)
  spec_RS = RS_rad[iTobs]
  spec_FS = FS_rad[iTobs]
  Rcd = get_radius_zone(df, 2)*c_
  dRcd = (Rcd - env.R0)/env.R0
  rc_str= reformat_scientific(f"{dRcd:.3e}")
  title = env.runname + f", it {it}, $\\Delta R_{{cd}}/R_0= {rc_str}$, $\\bar{{T}} = {(T-env.Ts)/env.T0:.2f}$"
  
  plt.loglog(nuobs/env.nu0, spec_RS/env.nu0F0, color='r', label='RS')
  plt.loglog(nuobs/env.nu0, spec_FS/env.nu0F0, color='b', label='FS')
  plt.loglog(nuobs/env.nu0, (spec_FS+spec_RS)/env.nu0F0, color='k', label='RS + FS')
  plt.title(title)
  plt.xlabel("$\\nu/\\nu_0$")
  plt.ylabel("$\\nu F_\\nu / \\nu_0 F_0$")
  plt.legend()
  plt.tight_layout()

def plot_df_lightcurve(key, it, lognu, func='analytic', fig_in=None):
  '''
  Plot lightcurve contribution of iteration it of run key,
  observed at frequency 10**lognu * nu_0
  '''
  if fig_in==None:
    fig, ax = plt.subplots()
  else:
    fig = fig_in
    ax = plt.gca()
  df = openData_withDistrib(key, it)
  nuobs, Tobs, dTobs, env = get_radEnv(key)
  RS_rad, FS_rad = df_get_rads(df, nuobs, Tobs, dTobs, env, func)
  i_nu = find_closest(np.log10(nuobs/env.nu0), lognu)
  lc_RS = RS_rad[:, i_nu]
  lc_FS = FS_rad[:, i_nu]
  Tb = (Tobs - env.Ts)/env.T0
  Rcd = get_radius_zone(df, 2)*c_
  dRcd = (Rcd - env.R0)/env.R0
  rc_str= reformat_scientific(f"{dRcd:.3e}")
  title = env.runname + f", it {it}, $\\Delta R_{{cd}}/R_0= {rc_str}$, $\\log_{{10}}\\nu/\\nu_0={lognu}$"
  
  ax.plot(Tb, lc_RS/env.nu0F0, color='r', label='RS')
  ax.plot(Tb, lc_FS/env.nu0F0, color='b', label='FS')
  ax.plot(Tb, (lc_FS+lc_RS)/env.nu0F0, color='k', alpha=.5, label='RS + FS')
  if fig_in == None:
    plt.title(title)
    ax.legend()
  ax.set_xlabel("$\\bar{T}$")
  ax.set_ylabel("$\\nu F_\\nu / \\nu_0 F_0$")
  plt.tight_layout()

def get_cell_env(key, it, i, Nnu=100, nT=50):
  '''
  Returns necessary variables for spectrum calculation
  Mainly useful for testing
  '''
  env  = MyEnv(key)
  df   = openData_withDistrib(key, it)
  cell = open_cell(df, i)
  lfac = cell['D']/cell['rho']
  Tth, Ton, Tej = cell_derive_obsTimes(cell, env)
  dT = Tth/nT
  Tobs = Ton + (np.arange(5*nT)+0.5)*dT
  B = cell_derive_B(cell, env)
  nus = 2*lfac*lfac2nu(cell['gb'], B)/(1+env.z)
  nuobs = np.logspace(-5, 2, Nnu) * nus

  return cell, nuobs, Tobs, dT, env


def plot_cell_compare_fluences(key, it, i):
  '''
  Plot time-integrated spectrum of cell i of iteration it of run key
  compare analytical and numerical results
  '''
  plt.figure()
  cell, nuobs, Tobs, dT, env = get_cell_env(key, it, i)
  res1 = nuobs * cell_spectra_analytic(cell, nuobs, Tobs, dT, env)
  spec1 = np.trapz(res1, x=Tobs, axis=0)
  res2 = nuobs * cell_spectra_numeric(cell, nuobs, Tobs, dT, env)
  spec2 = np.trapz(res2, x=Tobs, axis=0)
  ratio = spec2/spec1
  print(ratio.mean())
  
  plt.loglog(nuobs/env.nu0, spec2/(env.nu0F0*env.T0), label='numerical')
  plt.loglog(nuobs/env.nu0, spec1/(env.nu0F0*env.T0), label='analytical')
  plt.title(env.runname + f", cell {i}, it {it}")
  plt.xlabel("$\\nu/\\nu_0$")
  plt.ylabel("$f_\\nu / \\nu_0 F_0 T_0$")
  plt.legend()
  plt.tight_layout()

def plot_cell_lightcurve(key, it, i, lognu):
  '''
  Compare lightcurve emitted by a spherical cell between numerical and analytical,
  at frequency nu = 10**lognu nu_0
  '''

  #plt.figure()
  cell, nuobs, Tobs, dT, env = get_cell_env(key, it, i)
  nuobs, Tobs, dTobs, env = get_radEnv(key)
  i_nu = find_closest(np.log10(nuobs/env.nu0), lognu)
  #r = cell['x']*c_
  #lfac = cell['D']/cell['rho']
  #Tth, Ton, Tej = cell_derive_obsTimes(cell, env)
  Tb = (Tobs - env.Ts)/env.T0

  flux = cell_flux(cell, nuobs, Tobs, dT, env, 'analytic')

  title=f'$\\log_{{10}}(\\nu/\\nu_0) = {lognu}$'
  plt.semilogy(Tb, flux[:,i_nu]/env.nu0F0)
  plt.title(env.runname + f", cell {i}, it {it}, "+title)
  #plt.xlabel('$T_{obs}$')
  plt.xlabel('$\\bar{T}$')
  plt.ylabel('$\\frac{{\\nu F_\\nu}}{{\\nu_0 F_0}}$', rotation='horizontal')
  plt.tight_layout()

def plot_cell_timeintegspec_analytic(key, it, i):
  '''
  Plot time-integrated spectrum of cell i of iteration it of run key
  '''
  plt.figure()
  cell, nuobs, Tobs, dT, env = get_cell_env(key, it, i)
  res = nuobs * cell_spectra_analytic(cell, nuobs, Tobs, dT, env)
  spec = np.trapz(res, x=Tobs, axis=0)
  
  plt.loglog(nuobs/env.nu0, spec/(env.nu0F0*env.T0))
  plt.title(env.runname + f", cell {i}, it {it}, analytical")
  plt.xlabel("$\\nu/\\nu_0$")
  plt.ylabel("$f_\\nu / \\nu_0 F_{{\\nu_0}}T_0$")
  plt.tight_layout()

def plot_cell_timeintegspec_numeric(key, it, i, type='Band'):
  '''
  Plot time-integrated spectrum of cell i of iteration it of run key
  '''
  plt.figure()
  cell, nuobs, Tobs, dT, env = get_cell_env(key, it, i)
  res = nuobs * cell_spectra_numeric(cell, nuobs, Tobs, dT, env, type)
  spec = np.trapz(res, x=Tobs, axis=0)
  
  plt.loglog(nuobs/env.nu0, spec/(env.nu0F0*env.T0))
  plt.title(env.runname + f", cell {i}, it {it}, numerical")
  plt.xlabel("$\\nu/\\nu_0$")
  plt.ylabel("$f_\\nu / \\nu_0 F_{{\\nu_0}}T_0$")
  plt.tight_layout()

def plot_cell_lightcurve_analytic(key, it, i, lognu):
  '''
  Lightcurve at log10(nu/nu0) of cell i of iteration it of run key
  '''

  plt.figure()
  cell, nuobs, Tobs, dT, env = get_cell_env(key, it, i)
  res = cell_flux(cell, nuobs, Tobs, dT, env, 'analytic')
  i_nu = find_closest(np.log10(nuobs/env.nu0), lognu)
  r = cell['x']*c_
  lfac = cell['D']/cell['rho']
  Ton = env.t0 + cell['t'] - r/c_
  Tth = r/(2*lfac**2*c_)
  Tb = (Tobs - Ton)/Tth

  title=f'$\\log_{{10}}(\\nu/\\nu_0) = {lognu}$'
  plt.semilogy(Tb, res[:,i_nu]/env.nu0F0)
  plt.title(env.runname + f", cell {i}, it {it}, "+title)
  plt.xlabel('$\\bar{T}$')
  plt.ylabel('$\\frac{{\\nu F_\\nu}}{{\\nu_0 F_0}}$', rotation='horizontal')
  plt.tight_layout()

def plot_cell_lightcurve_numeric(key, it, i, lognu, type='Band'):
  '''
  Lightcurve at log10(nu/nu0) of cell i of iteration it of run key
  '''

  plt.figure()
  cell, nuobs, Tobs, dT, env = get_cell_env(key, it, i)
  res = nuobs * cell_spectra_numeric(cell, nuobs, Tobs, dT, env, type)
  i_nu = find_closest(np.log10(nuobs/env.nu0), lognu)
  r = cell['x']*c_
  lfac = cell['D']/cell['rho']
  Ton = env.t0 + cell['t'] - r/c_
  Tth = r/(2*lfac**2*c_)
  Tb = (Tobs - Ton)/Tth

  title=f'$\\log_{{10}}(\\nu/\\nu_0) = {lognu}$'
  plt.semilogy(Tb, res[:,i_nu]/env.nu0F0)
  plt.title(env.runname + f", cell {i}, it {it}, "+title)
  plt.xlabel('$\\bar{T}$')
  plt.ylabel('$\\frac{{\\nu F_\\nu}}{{\\nu_0 F_0}}$', rotation='horizontal')
  plt.tight_layout()

def plot_cell_spectrum_analytic(key, it, i, T):
  '''
  Plot instantaneous spectrum of cell i of iteration it of run key,
  observed at observer time T
  '''
  plt.figure()
  cell, nuobs, Tobs, dT, env = get_cell_env(key, it, i)
  res = cell_flux(cell, nuobs, Tobs, dT, env, 'analytic')
  iTobs = find_closest(Tobs, T)
  spec = res[iTobs]
  
  plt.loglog(nuobs/env.nu0, spec/env.nu0F0)
  plt.title(env.runname + f", cell {i}, it {it}, $T_{{obs}} = {T}$")
  plt.xlabel("$\\nu/\\nu_0$")
  plt.ylabel("$\\nu F_\\nu / \\nu_0 F_{{\\nu_0}}$")
  plt.tight_layout()


def plot_cell_spectrum_numeric(key, it, i, T, type='Band'):
  '''
  Plot instantaneous spectrum of cell i of iteration it of run key,
  observed at observer time T
  '''

  env  = MyEnv(key)
  df   = openData_withDistrib(key, it)
  cell = df.iloc[i] 
  r = cell['x']*c_
  lfac = cell['D']/cell['rho']
  Tth = r/(2*lfac**2*c_)
  dT = Tth/100
  Tobs = np.arange(1000)*dT
  Tobs = 0.5*(Tobs[1:]+Tobs[:-1])
  iTobs = find_closest(Tobs, T)
  B = np.sqrt(env.eps_B * derive_Eint_comoving(cell['rho']*env.rhoscale*c_**2, cell['p']*env.rhoscale*c_**2))
  nus = 2*lfac*lfac2nu(cell['gm'], B)/(1+env.z)
  nuobs = np.logspace(-3, 2) * nus
  res = nuobs * cell_spectra_numeric(cell, nuobs, Tobs, dT, env, type)
  spec = res[iTobs]
  
  plt.loglog(nuobs/env.nu0, spec/env.nu0F0)
  plt.title(env.runname + f", cell {i}, it {it}, $T_{{obs}} = {T}$")
  plt.xlabel("$\\nu/\\nu_0$")
  plt.ylabel("$\\nu F_\\nu / \\nu_0 F_{{\\nu_0}}$")
  plt.tight_layout()


def time_integrated_flux(key):
  '''
  Plots time-integrated flux
  '''
  env = MyEnv(key)
  rad = open_raddata(key)
  # be careful of integration, check fig 4 Minu : plot nu * integrate(Fnu(T)) / nu0 Fnu0 T0
  spec = rad.iloc[:, 0:].apply(lambda x: integrate.trapz(x, rad.index))
  x = np.logspace(-3, 1, 100)

  plt.loglog(x, spec)
  plt.xlabel('$\\nu/\\nu_0$')
  plt.ylabel('$\\nu F_\\nu / \\nu_0 F_0$')
  plt.title(env.runname)

# basic functions
# --------------------------------------------------------------------------------------------------
def update_line(hl, new_data):
  '''
  Function to update plot
  '''
  #hl.set_xdata(numpy.append(hl.get_xdata(), new_data))
  hl.set_ydata(numpy.append(hl.get_ydata(), new_data))
  plt.draw()

# basic testing functions
# --------------------------------------------------------------------------------------------------
def test_primvar(it, key='Last'):
  df = openData(key, it)
  df_plot_primvar(df)
  df_plot_filt(df)

def test_interf(it, key='Last'):
  df = openData(key, it)
  df_plot_primvar(df)
  plt.plot(df['Sd'], c='k', label='Sd')
  plt.plot(df['trac']/2., c='r', label='tracer')
  plt.legend()

def df_plot_primvar(df):
  rho, u, p = df_get_primvar(df)
  plt.figure()
  plt.plot(rho/rho.max(), label='$\\rho/\\rho_{max}$')
  plt.plot(u/u.max(), label='$u/u_{max}$')
  plt.plot(p/p.max(), label='$p/p_{max}$')
  plt.legend()

def df_plot_filt(df, sigma=3):
  dGrho, dGu, dGp = df_get_gaussfprim(df, sigma)
  plt.plot(dGrho/np.abs(dGrho).max(), label='$\\nabla \\tilde{\\rho}$')
  plt.plot(dGu/np.abs(dGu).max(), label='$\\nabla \\tilde{u}$')
  plt.plot(dGp/np.abs(dGp).max(), label='$\\nabla \\tilde{p}$')
  plt.legend()

def df_get_primvar(df):
  rho = df['rho'].to_numpy()
  u   = df_get_u(df)
  p   = df['p'].to_numpy()
  return rho, u, p

def df_get_gaussfprim(df, sigma=3):
  rho, u, p = df_get_primvar(df)
  dGrho = step_gaussfilter(rho, sigma)
  dGu   = step_gaussfilter(u, sigma)
  dGp   = step_gaussfilter(p, sigma)
  return dGrho, dGu, dGp

def step_gaussfilter(arr, sigma=3):
  '''
  Gaussian filtering part of the steps detection
  '''
  arr /= arr.max()
  smth_arr = gaussian_filter1d(arr, sigma, order=0)
  d = np.gradient(smth_arr)
  return d

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

