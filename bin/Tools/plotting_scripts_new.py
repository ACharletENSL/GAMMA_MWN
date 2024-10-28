# -*- coding: utf-8 -*-
# @Author: acharlet

'''
This files contains plotting functions to plot GAMMA outputs and analyzed datas
'''

# Imports
# --------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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
plt.rcParams['axes.grid'] = True
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) 
formatter.set_powerlimits((-1,1)) 

cm = plt.get_cmap('tab20')

# Legends and labels
var_exp = {
  "x":"$r$", "dx":"$dr$", "rho":"$\\rho$", "vx":"$\\beta$", "p":"$p$", "v":"$\\beta$",
  "D":"$\\gamma\\rho$", "sx":"$\\gamma^2\\rho h$", "tau":"$\\tau$", "gma_m":"$\\gamma_m$",
  "trac":"tracer", "trac2":"wasSh", "Sd":"shock id", "gmin":"$\\gamma_{min}$", "gmax":"$\\gamma_{max}$", "zone":"",
  "T":"$\\Theta$", "h":"$h$", "lfac":"$\\Gamma$", "u":"$\\Gamma\\beta$", "u_i":"$\\Gamma\\beta$", "u_sh":"$\\gamma\\beta$",
  "Ei":"$e_{int}$", "Ekin":"$e_k$", "Emass":"$\\rho c^2$", "dt":"dt", "res":"dr/r", "Tth":"$T_\\theta$",
  "numax":"$\\nu'_{max}$", "numin":"$\\nu'_{min}$", "B":"$B'$", "Pmax":"$P'_{max}$",
  "gm":"$\\gamma_0$", "gb":"$\\gamma_1$", "gM":"$\\gamma_2$", "Lp":"$L'$", "L":"$L$",
  "Lth":"$\\tilde{L}_{\\nu_m}$", "Lum":"$\\tilde{L}_{\\nu_m}$",
  "num":"$\\nu'_m$", "nub":"$\\nu'_b$", "nuM":"$\\nu'_M$", "inj":"inj", "fc":"fc", "nu_m":"$\\nu_m$","nu_m2":"$\\nu_m$",
  "Ton":"$T_{on}$", "Tth":"$T_{\\theta,k}$", "Tej":"$T_{ej,k}$", "tc":"$t_c$", 
  "V4":"$\\Delta V^{(4)}$", "V3":"$\\Delta V^{(3)}$", "V4r":"$\\Delta V^{(4)}_c$",
  "i":"i", 'i_sh':'i$_{sh}$', 'i_d':'i$_d$'
  }
units_CGS  = {
  "x":" (cm)", "dx":" (cm)", "rho":" (g cm$^{-3}$)", "vx":"", "p":" (Ba)",
  "D":"", "sx":"", "tau":"", "trac":"", "trac2":"", "Sd":"", "gmin":"", "gmax":"", "u_sh":"",
  "T":"", "h":"", "lfac":"", "u":"", "u_i":"", "zone":"", "dt":" (s)", "res":"", "Tth":" (s)",
  "Ei":" (erg cm$^{-3}$)", "Ek":" (erg cm$^{-3}$)", "M":" (g)", "Pmax":" (erg s$^{-1}$cm$^{-3}$)", "Lp":" (erg s$^{-1}$)",
  "numax":" (Hz)", "numin":" (Hz)", "B":" (G)", "gm":"", "gb":"", "gM":"", "L":" (erg s$^{-1}$) ", "Lum":" (erg s$^{-1}$) ",
  "num":" (Hz)", "nub":" (Hz)", "nuM":" (Hz)", "inj":"", "fc":"", "tc":" (s).", "Ton":" (s)",
  "Tth":" (s)", "Tej": " (s)", "V4":" (cm$^3$s)", "V3":" (cm$^3$)", "V4r":" (cm$^3$s)", "i":"", "nu_m":" (Hz)", "nu_m2":" (Hz)",
  }
var_label = {'R':'$r$ (cm)', 'v':'$\\beta$', 'u':'$\\gamma\\beta$', 'u_i':'$\\gamma\\beta$', 'u_sh':'$\\gamma\\beta$',
  'f':"$n'_3/n'_2$", 'rho':"$n'$", 'rho3':"$n'_3$", 'rho2':"$n'_2$", 'Nsh':'$N_{cells}$',
  'V':'$V$ (cm$^3$)', 'Nc':'$N_{cells}$', 'ShSt':'$\\Gamma_{ud}-1$', "Econs":"E",
  'ShSt ratio':'$(\\Gamma_{34}-1)/(\\Gamma_{21}-1)$', "Wtot":"$W_4 - W_1$",
  'M':'$M$ (g)', 'Msh':'$M$ (g)', 'Ek':'$E_k$ (erg)', 'Ei':'$E_{int}$ (erg)',
  'E':'$E$ (erg)', 'Etot':'$E$ (erg)', 'Esh':'$E$ (erg)', 'W':'$W_{pdV}$ (erg)',
  'Rct':'$(r - ct)/R_0$ (cm)', 'D':'$\\Delta$ (cm)', 'u_i':'$\\gamma\\beta$',
  'vcd':"$\\beta - \\beta_{cd}$", 'epsth':'$\\epsilon_{th}$', 'pdV':'$pdV$'
  }
cool_vars = ["inj", "fc", "gmin", "gmax", "trac2", "gm", "gb", "gM", "num", "nub", "nuM"]
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

def plotduo_behindShocks(varlist, key='Last', itmin=0, itmax=None,
    xscaling='it', yscalings=['code', 'code'], logt=False, logys=[False, False],
    names = ['RS', 'FS'], lslist = ['-', ':'], colors = ['r', 'b'], **kwargs):
  '''
  Plots 2 chosen variables as measured in the cell behind each shock front
  '''
  
  env = MyEnv(key)
  fig, ax1 = plt.subplots()
  ax2 = ax1.twinx()
  axes = [ax1, ax2]
  varexps = [var_exp[var] for var in varlist]
  dummy_lst = [plt.plot([],[], ls=ls, color='k')[0] for ls in lslist]
  dummy_col = [plt.plot([],[],ls='-',color=c)[0] for c in colors]
  legendlst = plt.legend(dummy_lst, varexps, bbox_to_anchor=(0.8, 1.02),
    loc='lower right', borderaxespad=0.)
  plt.legend(dummy_col, names, bbox_to_anchor=(0.2, 1.02),
    loc='lower left', borderaxespad=0.)
  fig.add_artist(legendlst)
  RS_vals, FS_vals = get_behindShock_vals(key, varlist, itmin, itmax)
  # each is series of [it, t, val1, val2]

  for vals, col in zip([RS_vals, FS_vals], colors):
    if xscaling == 'it':
      Nt = 0
      tscale, toffs = 1., 0.
      xlabel = 'it'
    elif xscaling == 't0':
      Nt = 1
      tscale, toffs = 1/env.t0, 1.
      xlabel = '$t/t_0$'
    else:
      Nt = 1
      tscale, toffs = 1., 0.
      xlabel = '$t_{sim}$ (s)'
    ax1.set_xlabel(xlabel)
    t = vals[Nt]*tscale + toffs
    y = [vals[-2], vals[-1]]

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

    ax.plot(t, yvar, ls=ls, color=col, **kwargs)
    if logt:
      ax.set_xscale('log')
    if logy:
      ax.set_yscale('log')

def ax_behindShocks_timeseries(var, key='Last', itmin=0, itmax=None, xscaling='it', yscaling='code', ax_in=None, logt=False, logy=False, **kwargs):
  '''
  Plots a time series of var in cells behind the shocks
  '''
  if var == 'i':
    yscaling = 'code'
  env  = MyEnv(key)

  if ax_in is None:
    plt.figure()
    ax = plt.gca()
    plt.title(f"run {env.runname}, {var_exp[var]} behind shock")
  else:
    ax = ax_in
  
  # create dataset
  its = np.array(dataList(key, itmin=itmin, itmax=itmax))[1:]
  its, time, rRS, rFS, yRS, yFS = get_behindShock_value(key, var, its)
  y = np.array([yRS, yFS])

  # plot 
  if xscaling == 'it':
    xRS, xFS = its, its
    xlabel = 'it'
  elif xscaling == 't0':
    t = time/env.t0 + 1.
    xRS, xFS = t, t
    xlabel = '$t/t_0$'
  elif xscaling == 'r':
    xRS, xFS = rRS*c_/env.R0, rFS*c_/env.R0
    xlabel = '$r/R_0$'
  elif xscaling == 'T':
    t = time + env.t0
    xRS = ((t - rRS) - env.Ts)/env.T0
    xFS = ((t - rFS) - env.Ts)/env.T0
    xlabel = "$\\bar{T}$"
  else:
    xRS, xFS = time, time
    xlabel = '$t_{sim}$ (s)'
  ax.set_xlabel(xlabel)

  if yscaling == 'code':
    units  = get_normunits('c', 'Norm')
    yscale = 1.
    if var.startswith('nu'):
      yscale = 1./env.nu0p
    elif var == 'L':
      yscale = 1./env.L0
    elif var == 'Lp':
      yscale = 1./env.L0p
    elif var == 'Pmax':
      yscale = 1./env.Pmax0
    elif var == 'B':
      yscale = 1./env.Bp
    elif var.startswith('T'):
      yscale = 1./env.T0
    elif "V4" in var:
      yscale = 1/(env.V0*env.t0)
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
      y -= env.Ts/env.T0
  ax.set_ylabel(ylabel)

  ax.plot(xRS, y[0], label='RS', c='r', **kwargs)
  ax.plot(xFS, y[1], label='FS', c='b', **kwargs)
  if logt:
    ax.set_xscale('log')
  if logy:
    ax.set_yscale('log')
  
  if ax_in is None:
    plt.legend()

def plot_injection_timeseries(i, key='Last', xscaling='it', **kwargs):
  plt.figure()
  ax = plt.gca()
  varlist = ['trac2', 'fc', 'inj']
  its = np.array(dataList(key))[1:]
  t, y = get_cell_timevalues(key, varlist, i, its)
  # plot 
  if xscaling == 'it':
    t = its
    xlabel = 'it'
  elif xscaling == 't0':
    t = time/env.t0 + 1.
    xlabel = '$t/t_0$'
  else:
    t = time
    xlabel = '$t_{sim}$ (s)'
  ax.set_xlabel(xlabel)
  
  for k, var in enumerate(varlist):
    ax.plot(t, y[k], label=var, **kwargs)
  plt.legend()

def plot_edistrib_timeseries(i, key='Last', xscaling='it', **kwargs):
  plt.figure()
  ax = plt.gca()
  varlist = ['gmin', 'gmax', 'gm', 'gb', 'gM']
  its = np.array(dataList(key))[1:]
  t, y = get_cell_timevalues(key, varlist, i, its)
  # plot 
  if xscaling == 'it':
    t = its
    xlabel = 'it'
  elif xscaling == 't0':
    t = time/env.t0 + 1.
    xlabel = '$t/t_0$'
  else:
    t = time
    xlabel = '$t_{sim}$ (s)'
  ax.set_xlabel(xlabel)
  
  for k, var in enumerate(varlist):
    ls = '-'
    if var in ['gm', 'gb', 'gM']:
      ls = ':'
    ax.semilogy(t, y[k], label=var_exp[var], linestyle=ls, **kwargs)
  plt.legend()

def ax_cell_timeseries(var, i, key='Last', xscaling='it', yscaling='code', ax_in=None, logt=False, logy=False, **kwargs):
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
  if xscaling == 'it':
    t = its
    xlabel = 'it'
  elif xscaling == 't0':
    t = time/env.t0 + 1.
    xlabel = '$t/t_0$'
  else:
    t = time
    xlabel = '$t_{sim}$ (s)'
  ax.set_xlabel(xlabel)

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

def plot_behindShocks(key, varlist, itmin=0, itmax=None, thcorr=False, n=2, m=1):
  '''
  Creates plots for chose variables
  '''

  RS_vals, FS_vals = get_behindShock_vals(key, varlist, itmin, itmax, thcorr, n, m)
  Nvar = len(varlist)
  fig, axs = plt.subplots(Nvar, 1, sharex='all')
  for vals, col in zip([RS_vals, FS_vals], ['r', 'b']):
    for y, ax in zip(vals[2:], axs):
      ax.semilogy(vals[0], y, c=col)
      ax.scatter(vals[0], y, marker='x', c=col)

# Time plots
# --------------------------------------------------------------------------------------------------
# to add a new variable, modify
# this file: var_label, theoretical
# run_analysis: get_timeseries
# phys_function_shells: time_analytical


lslist  = ['-', '--', '-.', ':']

def compare_runs(var, keylist, xscaling='t0',
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
    name, legend = ax_timeseries(var, key, ax, xscaling=xscaling, 
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
  xscaling='t0', yscaling=False, stopatcross=True,
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

  if xscaling == 'it':
    t = data.index
    xlabel = 'it'
    crosstimes = False
    theory  = False
  elif xscaling == 't0':
    t = data['time'].to_numpy()/env.t0 + 1.
    xlabel = '$t/t_0$'
    tRS = env.tRS/env.t0 + 1.
    tFS = env.tFS/env.t0 + 1.
  else:
    t = data['time'].to_numpy()
    xlabel = '$t_{sim}$ (s)'
    tRS = env.tRS
    tFS = env.tFS
  ax.set_xlabel(xlabel)

  ylabel = var_label[var]
  varlist = data.keys()[1:]
  yscale = 1.
  if var.startswith('E') or var.startswith('W'):
    yscaling = True
    yscale = (env.Ek4 + env.Ek1)/2.
    ylabel = '$W/E_{k,0}$' if var.startswith("W") else "$E/E_{k,0}$"
  elif var.startswith('M'):
    yscaling = True
    yscale = (env.M4 + env.M1)/2.
    ylabel ="$M/M_0$"
  elif var == 'u':
    yscaling = True
    yscale = env.u
    ylabel = "$\\gamma\\beta/u_{th}$"
  elif var == 'D':
    yscaling = True
    yscale = env.D01
    ylabel = "$\\Delta/\\Delta_0$"
  if reldiff:
    ylabel = '$(X - X_{th})/X_{th}$'
  ax.set_ylabel(ylabel)
  for n, varname in enumerate(varlist):
    y = data[varname].multiply(1./yscale)
    vsplit = varname.split('_')
    if vsplit[0] in correct_varlabels:
      varname = varname.replace(vsplit[0], correct_varlabels[vsplit[0]])
    varlabel = "$" + varname + "$"
    if varname.startswith('ShSt'):
      if 'rs' in varname: varlabel = 'RS'
      elif 'fs' in varname: varlabel = 'FS'
    
    if reldiff:
      expected = theoretical[var][0][n]
      y = derive_reldiff(y*yscale, expected)
    ax.plot(t, y, c=plt.cm.Paired(n), label=varlabel, **kwargs)
    if theory:
      if var in theoretical:
        expec = theoretical[var]
        for val, name in zip(expec[0], expec[1]):
          ax.axhline(val/yscale, color='k', ls='--', lw=.6)
          ax.text(1.02, val, name, color='k', transform=ax.get_yaxis_transform(),
            ha='left', va='center')
      else:
        ax.plot(t, expec_vals[n]/yscale, c='k', ls=':', lw=.8)
        th_legend = ax.legend(dummy_lst, ['data', 'analytical'], fontsize=11)

  if logt:
    ax.set_xscale('log')
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
    if len(varlist)>2:
      ax.legend(bbox_to_anchor=(1.02, 1.0), loc='upper left', borderaxespad=0.)
    elif reldiff:
      ax.legend(bbox_to_anchor=(0., 1.01), loc='lower left', borderaxespad=0.)
    else:
      ax.legend()
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

def article_snap(name, it, key='cart_fid'):
  prim_snapshot(it, key, theory=True, xscaling='Rcd')
  figname = './figures/' + name + '_' + str(it) + 'png' 
  plt.savefig(figname)
  plt.close()

def prim_snapshot(it, key='Last', theory=False, xscaling='R0', itoff='False', suptitle=True):
  f, axes = plt.subplots(3, 1, sharex=True, figsize=(6,6), layout='constrained')
  varlist = ['rho', 'u', 'p']
  #scatlist= []
  for var, k, ax in zip(varlist, range(3), axes):
    title, scatter = ax_snapshot(var, it, key, theory, ax_in=ax, xscaling=xscaling, itoff=itoff)
    #scatlist.append(scatter)
    if k != 2: ax.set_xlabel('')
  
  if suptitle:
    f.suptitle(title)
  #scatter = scatlist[1]
  plt.legend(*scatter.legend_elements(), bbox_to_anchor=(1.02, 0), loc='lower left', borderaxespad=0.)

def cons_snapshot(it, key='Last', theory=False, xscaling='R0', itoff='False'):
  f, axes = plt.subplots(3, 1, sharex=True, figsize=(6,6), layout='constrained')
  varlist = ['D', 'sx', 'tau']
  #scatlist= []
  for var, k, ax in zip(varlist, range(3), axes):
    title, scatter = ax_snapshot(var, it, key, theory, ax_in=ax, xscaling=xscaling, itoff=itoff)
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

def ax_snapshot(var, it, key='Last', theory=False, ax_in=None, itoff=False, tformat='t0',
    logx=False, logy=True, xscaling='R0', yscaling='norm', **kwargs):
    
  '''
  Create line plot on given ax of a matplotlib figure
  !!! find an updated way to normalize all variables properly with units
  tformat: 'Rcd', 't0'
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
  if (it != 0) and (var in cool_vars):
    df = openData_withDistrib(key, it)
  n = df["zone"].to_numpy()

  if tformat == 'Rcd':
    Rcd = get_radius_zone(df, 2)*c_
    dRcd = (Rcd - env.R0)/env.R0
    rc_str= reformat_scientific(f"{dRcd:.3e}")
    tstr = f", $(R_{{cd}}-R_0)/R_0= {rc_str}$"
  else:
    tstr = f", $t/t_0={1.+t/env.t0:.2f}$"

  it_str = "" if itoff else f", it {it}"
  title = env.runname + it_str + tstr
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
  if yscaling in ['code', 'norm']:
    units  = get_normunits('c', 'Norm')
    yscale = 1.
  elif yscaling=='norm':
    if var.startswith('nu'):
      yscale = 1./env.nu0p
    elif var == 'Lp':
      yscale = 1./env.L0p
    elif var == "L":
      yscale = 1./env.L0
    elif var == 'B':
      yscale = 1./env.Bp
    elif var in ["Tth", "Ton", "Tej"]:
      yscale = 1/env.T0
    elif var == 'u':
      yscale = 1./env.u
  elif yscaling == 'CGS':
    units = units_CGS
    yscale = get_varscaling(var, env)
  ylabel = (var_exp[var] + units[var]) if units[var] else var_exp[var]


  x = df['x'].to_numpy()
  if xscaling == 'n':
    trac = df['trac'].to_numpy()
    i4  = np.argwhere((trac > 0.99) & (trac < 1.01))[:,0].min()
    x = df['i'].to_numpy() - i4
    xscale = 1
  y = get_variable(df, var)
  if theory:
    r = df['x'].to_numpy()*c_
    y_th = get_analytical(var, t, r, env)
    y_th *= yscale
    ax.plot(x*xscale, y_th, 'k--', zorder=3)

  x *= xscale
  y *= yscale
  if var in ["Tth", "Ton", "Tej"]:
      ylabel = '$\\bar{T}' + var_exp[var][2:]
      y -= env.Ts/env.T0
  
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  ax.plot(x, y, 'k', zorder=1)
  scatter = ax.scatter(x, y, c=n, lw=1, zorder=2, cmap=cm, **kwargs)
  
  if logx:
    ax.set_xscale('log')
  if logy:
    ax.set_yscale('log')
  
  return title, scatter


# spectras and lightcurves
# --------------------------------------------------------------------------------------------------
def construct_run_lightcurve(lognu, key='Last', itmin=0, itmax=None, logx=False, logy=False, crosstimes=True, fig_in=None):
  '''
  Constructs the lightcurve at chosen frequency, displays iteration after iteration
  '''

  if fig_in:
    fig = fig_in
  else:
    fig = plt.figure()
  ax = fig.add_subplot(111)
  nuobs, Tobs, env = get_radEnv(key)
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
  
  its = np.array(dataList(key, itmin=itmin, itmax=itmax))[2:]
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
    for line in ax.get_lines():
      line.remove()
    ax.plot(Tb, RS_rad/env.nu0F0, color='r')
    ax.plot(Tb, FS_rad/env.nu0F0, color='b')
    ax.plot(Tb, (RS_rad+FS_rad)/env.nu0F0, color='k')
    ax.texts[-1].set_text('it '+str(it))
    plt.savefig(f'./figures/serie/lightcurve_{it:05d}.png')

def plot_compare_lightcurves(key, lognu, logT=False, logF=False, crosstimes=True):
  '''
  Plots the 3 lightcurves (analytic, semi-analytic, numeric) at one frequency 
  '''
  plt.figure()
  ax = plt.gca()
  nuobs, Tobs, env = get_radEnv(key)
  Ts = env.Ts
  T0 = env.T0
  Tb = (Tobs-Ts)/T0
  radfile_path, _ = get_radfile(key)
  i_nu = find_closest(np.log10(nuobs/env.nu0), lognu)
  nu = nuobs[i_nu]
  title = env.runname + f", log$_{{10}}(\\nu/\\nu_0) = {np.log10(nu/env.nu0):.1f}$"

  lstlist = ['-', '-.', ':']
  lstnames = ['semi-analytic', 'numeric', 'analytic']
  dummy_lst = [plt.plot([],[], ls=ls, color='k')[0] for ls in lstlist]
  lstx, lsty = (0.7, 0.15) if (logT and logF) else (0.98, 0.7)
  lstlegend = ax.legend(dummy_lst, lstnames,
      fontsize=11, bbox_to_anchor=(lstx, lsty), loc='upper right', borderaxespad=0.)
  clist = ['r', 'b', 'k']
  cnames = ['RS', 'FS', 'RS+FS']
  dummy_col = [plt.plot([],[], c=c, ls='-')[0] for c in clist]

  
  for f, ls in zip(['th', 'num'], lstlist[:-1]):
    path = radfile_path[:-4] + '_' + f + radfile_path[-4:]
    rads = pd.read_csv(path, index_col=0)
    nuRS = [s for s in rads.keys() if 'RS' in s]
    nuFS = [s for s in rads.keys() if 'FS' in s]
    light_RS = RS_rad[:, i_nu]/env.nu0F0
    light_FS = FS_rad[:, i_nu]/env.nu0F0
    light_tot = light_RS + light_FS
    ax.plot(Tb, light_RS, c='r', ls=ls)
    ax.plot(Tb, light_FS, c='b', ls=ls)
    ax.plot(Tb, light_tot, c='k', ls=ls)


def plot_run_lightcurve(key, lognu, func='Band', hybrid=False,
   logx=False, logy=False, crosstimes=True, theory=True, g1=None, norm=None, Nnu=200):
  '''
  Plots lightcurve at chosen frequency, generated from whole run
  '''

  plt.figure()
  ax = plt.gca()
  nuobs, Tobs, env = get_radEnv(key, Nnu)
  hybrid = True if env.geometry == 'cartesian' else False
  Ts = env.Ts
  T0 = env.T0
  Tb = (Tobs-Ts)/T0
  rads = open_raddata(key, func)
  nuRS = [s for s in rads.keys() if 'RS' in s]
  nuFS = [s for s in rads.keys() if 'FS' in s]
  RS_rad = rads[nuRS].to_numpy()
  FS_rad = rads[nuFS].to_numpy()
  i_nu = find_closest(np.log10(nuobs/env.nu0), lognu)
  nu = nuobs[i_nu]
  light_RS = RS_rad[:, i_nu]/env.nu0F0
  light_FS = FS_rad[:, i_nu]/env.nu0F0
  light_tot = light_RS + light_FS
  title = env.runname + f", log$_{{10}}(\\nu/\\nu_0) = {np.log10(nu/env.nu0):.1f}$"
  
  ax.plot(Tb, light_RS, color='r', label='RS')
  ax.plot(Tb, light_FS, color='b', label='FS')
  ax.plot(Tb, light_tot, color='k', label='RS + FS')

  if crosstimes:
    TbRS = (env.TRS - Ts)/T0
    TbFS = (env.TFS - Ts)/T0
    ax.axvline(TbRS, color='r', ls='--', lw=.7)
    ax.text(TbRS, -.07, '$\\bar{T}_{RS}$', color='r', transform=ax.get_xaxis_transform(),
      ha='center', va='top')
    ax.axvline(TbFS, color='b', ls='--', lw=.7)
    ax.text(TbFS, -.07, '$\\bar{T}_{FS}$', color='b', transform=ax.get_xaxis_transform(),
      ha='center', va='top')
  if theory:
    #plt.autoscale(False)
    nuobs = np.array([10.**lognu*env.nu0])
    lc_RS, lc_FS = nuFnu_theoric(nuobs, Tobs, env, g1=g1, hybrid=hybrid, func=func)
    LRS = lc_RS[0]
    LFS = lc_FS[0]
    Ltot = LFS + LRS
    print(f'Ratio at peaks = {light_tot.max()/Ltot.max()}')

    dummy_lst = [plt.plot([],[], ls=ls, color='k')[0] for ls in ['-', '-.']]
    lstlegend = ax.legend(dummy_lst, ['numerical', 'analytical'],
      fontsize=11, bbox_to_anchor=(0.98, 0.7), loc='upper right', borderaxespad=0.)
    
    normRS, normFS = 1., 1.
    TbRS = (env.TRS - Ts)/T0
    TbFS = (env.TFS - Ts)/T0
    ipkRS = find_closest(Tb, TbRS)
    ipkFS = find_closest(Tb, TbFS)
    if norm == 'RS':
      #normRS = light_RS.max()/LRS.max()
      normRS = light_RS[ipkRS]/LRS[ipkRS]
      normFS = normRS
      print(f"Normalization factor {normRS}")
    elif norm == 'FS':
      normFS = light_FS[ipkFS]/LFS[ipkFS]
      normRS = normFS
      print(f"Normalization factor {normFS}")
    elif norm == 'both':
      normRS = light_RS[ipkRS]/LRS[ipkRS]
      normFS = light_FS[ipkFS]/LFS[ipkFS]
      print(f"Normalization factors: RS = {normRS}, FS = {normFS}")
    elif type(norm) == float:
      normfac = norm
      title = title + f', peak $/{norm}$'
    if norm in ['RS', 'FS']:
      print(f'Normalized to {norm} peak, with a ratio of {normfac:.3f}')
      title = title + f', normalized to {norm}'
    elif norm == 'both':
      title = title + f', both normalized'

    ax.plot(Tb, LRS*normRS, 'r-.')
    ax.plot(Tb, LFS*normFS, 'b-.')
    ax.plot(Tb, LRS*normRS+LFS*normFS, 'k-.')

    # m, a, d = 0., 0., 0.
    # if env.geometry == 'spherical':
    #   m, a, d = 0., 1., -1.
    # lc_RS2, lc_FS2 = nuFnu_general(nuobs, Tobs, env, m, a, d)
    # lctot2 = lc_RS2 + lc_FS2
    # ax.plot(Tb, lc_RS2, 'r:')
    # ax.plot(Tb, lc_FS2, 'b:')
    # ax.plot(Tb, lctot2, 'k:')

  if logx:
    ax.set_xscale('log')
  if logy:
    ax.set_yscale('log')

  plt.title(title)
  plt.xlabel("$\\bar{{T}}$")
  plt.ylabel("$\\nu F_\\nu / \\nu_0 F_0$")
  plt.legend()
  if theory:
    ax.add_artist(lstlegend)
  plt.tight_layout()

def plot_run_spectrum(key, logTb, theory=True, geometry=None, Nnu=200):
  '''
  Plots instantaneous spectrum at T_bar = 10**logTb, generated from whole run
  '''
  plt.figure()
  ax = plt.gca()
  nuobs, Tobs, env = get_radEnv(key, Nnu)
  nub = nuobs/env.nu0
  its = np.array(dataList(key))[2:]
  rads = open_raddata(key)
  nuRS = [s for s in rads.keys() if 'RS' in s]
  nuFS = [s for s in rads.keys() if 'FS' in s]
  RS_rad = rads[nuRS].to_numpy()
  FS_rad = rads[nuFS].to_numpy()
  Ton = env.Ts
  Tth = env.T0
  T = Tth*(10**logTb) + Ton
  iTobs = find_closest(Tobs, T)
  spec_RS = RS_rad[iTobs]/env.nu0F0
  spec_FS = FS_rad[iTobs]/env.nu0F0
  spec_tot= spec_FS+spec_RS
  
  ax.loglog(nub, spec_RS, color='r', label='RS')
  ax.loglog(nub, spec_FS, color='b', label='FS')
  ax.loglog(nub, spec_tot, color='k', label='RS + FS')

  if theory:
    #plt.autoscale(False)
    hybrid = True if env.geometry == 'cartesian' else False
    dummy_lst = [plt.plot([],[], ls=ls, color='k')[0] for ls in ['-', '-.']]
    lstlegend = ax.legend(dummy_lst, ['numerical', 'analytical'],
      fontsize=11, bbox_to_anchor=(0.5, 0.02), loc='lower center', borderaxespad=0.)
    sp_RS, sp_FS = nuFnu_theoric(nuobs, T, env, hybrid=hybrid)
    sp_tot = sp_RS + sp_FS
    print(f'Ratio at peaks = {spec_tot.max()/sp_tot.max()}')
    ax.loglog(nub, sp_RS, 'r-.')
    ax.loglog(nub, sp_FS, 'b-.')
    ax.loglog(nub, sp_tot, 'k-.')

  plt.title(env.runname + ", $\\log_{10}(\\bar{T}) = " + f"{logTb:.1f}$")
  plt.xlabel("$\\nu/\\nu_0$")
  plt.ylabel("$\\nu F_\\nu / \\nu_0 F_0$")
  plt.legend()
  if theory:
    ax.add_artist(lstlegend)
  plt.tight_layout()

def plot_df_spectrum(key, it, T, method='analytic', func='Band'):
  '''
  Plot instantaneous spectrum of iteration it of run key,
  observed at observer time T
  '''
  plt.figure()
  df = openData_withDistrib(key, it)
  nuobs, Tobs, env = get_radEnv(key)
  RS_rad, FS_rad = df_get_rads(df, nuobs, Tobs, env, method, func)
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

def plot_df_lightcurve_old(key, it, lognu, method='analytic', func='Band',
    theory=False, g1=False, dR=1e-3,
    fig_in=None, logT=False, logy=False):
  '''
  Plot lightcurve contribution of iteration it of run key,
  observed at frequency 10**lognu * nu_0
  '''
  if fig_in==None:
    fig, ax = plt.subplots()
    alpha = 1.
  else:
    fig = fig_in
    ax = plt.gca()
    alpha = 0.5
  df = openData_withDistrib(key, it)
  nuobs, Tobs, env = get_radEnv(key)
  #RS_rad, FS_rad = df_get_rads(df, nuobs, Tobs, dTobs, env, method, func)
  RS, FS = df_get_cellsBehindShock(df)
  nuobs = 10**lognu * env.nu0

  TbRS, lc_RS = cell_lightcurve(RS, nuobs, Tobs, env, method, func)
  TbFS, lc_FS = cell_lightcurve(FS, nuobs, Tobs, env, method, func)
  #i_nu = find_closest(np.log10(nuobs/env.nu0), lognu)
  #lc_RS = RS_rad[:, i_nu]/env.nu0F0
  #lc_FS = FS_rad[:, i_nu]/env.nu0F0
  Tb = (Tobs - env.Ts)/env.T0
  title = env.runname + f", it {it}, $\\log_{{10}}\\nu/\\nu_0={lognu}$"
  
  ax.plot(Tb, lc_RS, color='r', label='RS')
  ax.plot(Tb, lc_FS, color='b', label='FS')
  ax.plot(Tb, (lc_FS+lc_RS), color='k', alpha=alpha, label='RS + FS')
  if theory:
    lc_RSth, lc_FSth = np.zeros((2, len(Tobs)))
    ipkRS = lc_RS.argmax()
    #TbRS = Tb[ipkRS:]-Tb[ipkRS]
    tTRS = 1+TbRS
    gTRS = 1 + env.gRS*TbRS
    ipkFS = lc_FS.argmax()
    #TbFS = Tb[ipkFS:]-Tb[ipkFS]
    tTFS = 1+TbFS
    gTFS = 1 + env.gFS*TbFS
    RSmax = lc_RS.max()
    FSmax = lc_FS.max()
    lc_RSth[TbRS>=0] = lightcurve_shockfront(lognu, TbRS[TbRS>=0], env, sh='RS', g1=g1, dR=dR)
    lc_FSth[TbFS>=0] = lightcurve_shockfront(lognu, TbFS[TbFS>=0], env, sh='FS', g1=g1, dR=dR)
    BandRS = tTRS * gTRS**-3 * Band_func(10**lognu*gTRS)
    BandFS = tTFS * gTFS**-3 * Band_func(10**lognu*gTFS)
    TbRS = Tb[ipkRS-1:-1]
    TbFS = Tb[ipkFS-1:-1]
    ax.plot(Tb, lc_RSth*RSmax/lc_RSth.max(), 'r:')
    ax.plot(Tb, lc_FSth*FSmax/lc_FSth.max(), 'b:')
    ax.plot(Tb, BandRS*RSmax/BandRS.max(), 'r-.')
    ax.plot(Tb, BandFS*FSmax/BandFS.max(), 'b-.')
    #ax.plot(Tb, (lc_FSth+lc_RSth), 'k:', alpha=.7)
    dummy_lst = [plt.plot([],[], ls=ls, color='k')[0] for ls in ['-', ':', '-.']]
    lstlegend = ax.legend(dummy_lst, ['data', f'$\\Delta R = 10^{{{np.log10(dR):.0f}}}$', '$g_T^{-3}\\tilde{T}S(g_T\\frac{\\nu}{\\nu_0})$'],
      fontsize=11, bbox_to_anchor=(0.98, 0.6), loc='upper right', borderaxespad=0.)
  if fig_in == None:
    plt.title(title)
    ax.legend()
    if theory:
      ax.add_artist(lstlegend)
  if logT==True:
    ax.set_xscale('log')
    ionRS = lc_RS.argmax()
    ionFS = lc_FS.argmax()
    ax.set_xlim(xmin=Tb[min(ionRS, ionFS)-10])
  if logy==True:
    ax.set_yscale('log')
    ax.set_ylim(ymin=(min(lc_RS.max(),lc_FS.max())/10.))

  ax.set_xlabel("$\\bar{T}$")
  ax.set_ylabel("$\\nu F_\\nu / \\nu_0 F_0$")
  plt.tight_layout()

def plot_df_lightcurve(key, it, lognu, method='analytic', func='Band',
  theory=False, logT=False, logF=False, xscaling='RS', fig_in=None, **kwargs):
  '''
  Plots the lightcurve of a given datafile
  '''
  if fig_in==None:
    fig, ax = plt.subplots()
  else:
    fig = fig_in
    ax = plt.gca()
  
  df = openData_withDistrib(key, it)
  if method == 'analytic':
    plot_cell_lightcurve(key, it, 'RS', lognu, method, func, theory, logT, logF, xscaling, fig, color='r')
    line_RS = ax.get_lines()[-1]
    T = line_RS.get_xdata()
    nuFnu_RS = line_RS.get_ydata()
    plot_cell_lightcurve(key, it, 'FS', lognu, method, func, theory, logT, logF, xscaling, fig, color='b')
    nuFnu_FS = ax.get_lines()[-1].get_ydata()
    nuFnu_tot = nuFnu_RS + nuFnu_FS
    plt.plot(T, nuFnu_tot, color='k')
  
  if xscaling == 'RS':
    ax.set_xlabel("$\\bar{T}_{RS}$")
  elif xscaling == 'FS':
    T = (Tobs - env.Ts)/env.T0FS
    ax.set_xlabel("$\\bar{T}_{FS}$")
  else:
    ax.set_xlabel("$T_{obs}$")
  
  dummy_col = [plt.plot([],[],ls='-',color=c)[0] for c in ['r', 'b', 'k']]
  plt.legend(dummy_col, ['RS', 'FS', 'RS + FS'], bbox_to_anchor=(0.98, 0.98),
    loc='upper right', borderaxespad=0.)


def plot_cell_lightcurve(key, it, i, lognu, func='Band',
  theory=False, logT=False, logF=False, xscaling='local', logslope=False,
  fig_in=None, dR=None, **kwargs):
  '''
  Plots the lightcurve of a given cell. i can also be the name of a shock front
  dR is the value for Delta R/R to for the analytical expression
  '''
  if fig_in==None:
    fig, ax = plt.subplots()
  else:
    fig = fig_in
    ax = plt.gca()

  nuobs, Tobs, env = get_radEnv(key)
  nuobs = 10**lognu * env.nu0
  df = openData_withDistrib(key, it)
  if type(i)==str:
    title_app = ", at " + i
    if i not in ['RS', 'FS']:
      print("Choose a valid cell index or shock front for input i")
      return 0
    cell = df_get_cellBehindShock(df, i, theory=False)
  else:
    title_app = f", at cell {i}"
    cell = open_cell(df, i)
  
  Tb, nuFnua = cell_lightcurve(cell, nuobs, Tobs, env, 'analytic', func)
  nuFnua /= env.nu0F0
  Tb, nuFnun = cell_lightcurve(cell, nuobs, Tobs, env, 'numeric', func)
  nuFnun /= env.nu0F0
  print(f"Analytic/numeric peak ratio = {nuFnua.max()/nuFnun.max()}")

  T = Tobs
  if xscaling == 'local':
    T = Tb
    ax.set_xlabel("$\\bar{T}$")
    #ax.set_xlim(xmin=0.)
  elif xscaling == 'RS':
    T = (Tobs - env.Ts)/env.T0
    ax.set_xlabel("$\\bar{T}_{RS}$")
  elif xscaling == 'FS':
    T = (Tobs - env.Ts)/env.T0FS
    ax.set_xlabel("$\\bar{T}_{FS}$")
  else:
    ax.set_xlabel("$T_{obs}$")
  if fig_in:
    ax.set_xlabel("")
  title = f"it {it}, $R/R_0 = {cell['x']*c_/env.R0:.2e}$, $\\nu=10^{{{lognu}}}\\nu_0$" + title_app
  
  ax.set_ylabel('$\\nu F_\\nu/\\nu_0 F_0$')
  ax.plot(T, nuFnua, label='semi-analytic', c='r', ls='-', **kwargs)
  ax.plot(T, nuFnun, label='numeric', c='g', ls='-.', **kwargs)
  if theory:
    c='k'
    if fig_in:
      c = ax.lines[-1].get_color()
    isIn1 = cell['trac']>1.5
    tT = 1.+ Tb
    gT = 1 + (env.gFS if isIn1 else env.gRS)**2 * Tb
    fac_nu = env.fac_nu if isIn1 else 1.
    if not dR:
      dR = cell['dx']/cell['x']

    Fnu_th = np.zeros(nuFnua.shape)
    Fnu_th[Tb>=0] = flux_shockfront(lognu, Tb[Tb>=0], env, ('FS' if isIn1 else 'RS'), dR=dR, g1=1.00001)
    nuFnu_th = (10**lognu / fac_nu) * Fnu_th
    norm = nuFnua.max()/nuFnu_th.max()
    ax.plot(T, nuFnu_th*norm, c=c, ls=':', label=f'$\\Delta R/R_0 = 10^{{{np.log10(dR):.1f}}}$')
    print(f"Normalization factor: {norm:.2e}")

  if not fig_in:
    plt.legend()
    plt.title(title)

  if logslope:
    logT, logF = False, False
  if logT:
    ax.set_xscale('log')
    ax.set_xlim(xmin=Tb[Tb>0][0])
  if logF:
    ax.set_yscale('log')
    ax.set_ylim(ymin=1e-2*nuFnua.max())


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