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

nolog_vars = ['trac', 'trac2', 'Sd', 'gmin', 'gmax', 'zone']
var_label = {'R':'$r$ (cm)', 'v':'$\\beta$', 'u':'$\\gamma\\beta$',
  'f':"$n'_3/n'_2$", 'rho':"$n'$", 'rho3':"$n'_3$", 'rho2':"$n'_2$", 'Nsh':'$N_{cells}$',
  'V':'$V$ (cm$^3$)', 'Nc':'$N_{cells}$', 'ShSt':'$\\Gamma_{ud}-1$', "Econs":"E",
  'ShSt ratio':'$(\\Gamma_{34}-1)/(\\Gamma_{21}-1)$', "Wtot":"$W_4 - W_1$",
  'M':'$M$ (g)', 'Msh':'$M$ (g)', 'Ek':'$E_k$ (erg)', 'Ei':'$E_{int}$ (erg)',
  'E':'$E$ (erg)', 'Etot':'$E$ (erg)', 'Esh':'$E$ (erg)', 'W':'$W_{pdV}$ (erg)',
  'Rct':'$(r - ct)/R_0$ (cm)', 'D':'$\\Delta$ (cm)', 'u_i':'$\\gamma\\beta$',
  'vcd':"$\\beta - \\beta_{cd}$", 'epsth':'$\\epsilon_{th}$', 'pdV':'$pdV$'}
#var_legends = {}

# Time plots
# --------------------------------------------------------------------------------------------------
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
  
  env  = MyEnv(get_physfile(key))
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

def rad_snapshot(it, key='Last', xscaling='R0', distr='gamma'):
  Nk = 3
  f, axes = plt.subplots(Nk, 1, sharex=True, figsize=(6, 2*Nk), layout='constrained')
  varlist = ['Sd', 'gmax', 'gmin']
  if distr == 'nu':
    varlist = ['Sd', 'numax', 'numin']
  logs = [False, True, True]
  for var, k, ax in zip(varlist, range(Nk), axes):
    title, scatter = ax_snapshot(var, it, key, theory=False, ax_in=ax, xscaling=xscaling, yscaling='CGS', logy=logs[k])
    if k != 2: ax.set_xlabel('')
  f.suptitle(title)
  plt.legend(*scatter.legend_elements(), bbox_to_anchor=(1.02, 0), loc='lower left', borderaxespad=0.)


def ax_snapshot(var, it, key='Last', theory=False, ax_in=None,
    logx=False, logy=True, xscaling='R0', yscaling='code', **kwargs):
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
  
  # Legends and labels
  var_exp = {
  "x":"$r$", "dx":"$dr$", "rho":"$\\rho$", "vx":"$\\beta$", "p":"$p$",
  "D":"$\\gamma\\rho$", "sx":"$\\gamma^2\\rho h$", "tau":"$\\tau$",
  "trac":"tracer", "trac2":"wasSh", "Sd":"shock id", "gmin":"$\\gamma_{min}$", "gmax":"$\\gamma_{max}$", "zone":"",
  "T":"$\\Theta$", "h":"$h$", "lfac":"$\\gamma$", "u":"$\\gamma\\beta$",
  "Ei":"$e_{int}$", "Ekin":"$e_k$", "Emass":"$\\rho c^2$", "dt":"dt", "res":"dr/r",
  "numax":"$\\nu_{max}$", "numin":"$\\nu_{min}$", "B":"$B"
  }
  units_CGS  = {
  "x":" (cm)", "dx":" (cm)", "rho":" (g cm$^{-3}$)", "vx":"", "p":" (Ba)",
  "D":"", "sx":"", "tau":"", "trac":"", "trac2":"", "Sd":"", "gmin":"", "gmax":"",
  "T":"", "h":"", "lfac":"", "u":"", "zone":"", "dt":" (s)", "res":"",
  "Ei":" (erg cm$^{-3}$)", "Ek":" (erg cm$^{-3}$)", "M":" (g)",
  "numax":" (eV)", "numin":" (eV)", "B":" (G)"
  }
  auth_theory = ["rho", "vx", "u", "p", "D", "sx", "tau", "T", "h", "lfac", "Ek", "Ei", "M"]
  nonlog_var = ['u', 'trac', 'Sd', 'zone']
  if var not in auth_theory:
    theory = False
  if var in nonlog_var:
    logy = False
  def get_normunits(xnormstr, rhonormsub):
    rhonormsub  = '{' + rhonormsub +'}'
    CGS2norm = {" (s)":" (s)", " (g)":" (g)", " (Hz)":" (Hz)", " (eV)":" (eV)", " (G)":" (G)",
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
    elif unit == " (G)":
      var_scale = np.sqrt(pNorm)
    else:
      var_scale = 1.
    return var_scale
  
  physpath = get_physfile(key)
  env = MyEnv(physpath)
  
  df, t, dt = openData_withtime(key, it)
  n = df["zone"]
  Rcd = get_radius_zone(df, 2)*c_
  dRcd = (Rcd - env.R0)/env.R0

  rc_str= reformat_scientific(f"{dRcd:.3e}")
  title = f"it {it}, $t_{{sim}}/t_0 = {dt/env.t0:.3f}$, $(R_{{cd}}-R_0)/R_0= {rc_str}$"
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

