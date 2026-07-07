# -*- coding: utf-8 -*-
# @Author: acharlet

'''
File to write new functions and test them in a notebook
(easier to modify/debug in file than in notebook)
'''

from analysis_thinshell import *
from plotting_functions import *
from peak_modeling import *
from flux_modeling import *

import matplotlib
matplotlib.rcParams["savefig.directory"] = GAMMA_dir + "/bin/Tools/figures"
import matplotlib.pyplot as plt
plt.ion()

### General functions
def savgol_smooth(array, window=7, polyorder=3):
  window = max(window, polyorder + 2)  # savgol constraint
  if window % 2 == 0:
    window += 1  # savgol requires odd window
  out = savgol_filter(array, window, polyorder)
  return out

def get_initslope(x, y, N=10):
  return logslope(x[0], y[0], x[N], y[N])

def key_to_logau(key):
  if 'sweep_log_au' in key:
    log_au_str = key.split('=')[-1]
    log_au = float(log_au_str)
  else:
    env = MyEnv(key)
    log_au = np.log10(env.a_u - 1)
  return log_au

### Visualizing raw data
def plot_earlydata(name, normName='', key='Last', z=4, xvar='x',
    imax=None, ax_in=None, cleanData=False, **kwargs):
  '''
  Function for raw data plotting
  '''
  if ax_in is None:
    fig, ax = plt.subplots()
  else:
    ax = ax_in
  env = MyEnv(key)
  data = open_rundata(key, z)
  if type(data) == bool:
    data = extract_data_thinshell(key, cells=[z], noOut=False)[0]
  if xvar == 'x':
    x = data.x.to_numpy() * c_ /env.R0
    xlabel = '$R/R_0$'
  elif xvar == 'it':
    x = data.index
    xlabel = 'it'
  elif xvar == 't':
    x = data.t.to_numpy() / env.t0
    xlabel = '$t/t_0$'
  if imax is not None:
    index = data.index
    xmax = x[index<=imax][-1]
  else:
    xmax = x[-1]
  val = get_variable(data, name, env)
  if normName:
    norm = getattr(env, normName)
    val /= norm
    #ax.axhline(1., c='k', ls=':', lw=.9)
  if cleanData:
    i0, i1 = find_fitting_boundaries(x, val, cr_offset=2)
    x = x[i0:i1]
    val = val[i0:i1]
  ax.plot(x[x<=xmax], val[x<=xmax], **kwargs)
  ax.set_xlabel(xlabel)
  if ax_in is None:
    ax.set_title(name)

def plot_early_both(name, normName='', key='Last', xvar='x', cleanData=False):
  fig, ax = plt.subplots()
  colors = ['r', 'b']
  labels = ['RS', 'FS']
  for z, col, label in zip([4, 1], colors, labels):
    plot_earlydata(name, normName=normName, key=key,
      z=z, xvar=xvar, ax_in=ax, cleanData=cleanData, c=col, label=label)
  ax.legend()

def compare_early_both(logau_arr, varName, normName='', linestyles=None):
  fig, ax = plt.subplots()
  keys = [logau_to_key(logau) for logau in logau_arr]
  if linestyles is None:
    linestyles = ['-', '--', '-.', ':']
    linestyles = linestyles[:len(keys)]
  colors = ['r', 'b']
  fronts = ['RS', 'FS']
  for key, ls in zip(keys, linestyles):
    for z, col in zip([4, 1], colors):
      plot_earlydata(varName, normName=normName, key=key,
        z=z, ax_in=ax, cleanData=True, c=col, ls=ls)
  legcol = create_legend_colors(ax, fronts, colors)
  leglst = create_legend_styles(ax, logau_arr, linestyles,
    title='log$_{10}(a_u-1)$', bbox_to_anchor=(1, 0.85))
  ax.add_artist(legcol)

def compare_num_reconst(key, z):
  '''
  Compares raw extracted data with reconstructed
  '''

  env = MyEnv(key)
  data = open_rundata(key, z)
  xd = data.x.to_numpy() * c_ / env.R0
  rec  = cellsBehindShock_fromData(data)
  xr = rec.x.to_numpy() * c_ / env.R0

  fig, (ax0, ax1, ax2) = plt.subplots(3, 1, 
    sharex=True, gridspec_kw = {'wspace':0, 'hspace':0})
  ax0.plot(xd, data.rho)
  ax0.plot(xr, rec.rho, ls='--', c='k')
  ax1.plot(xd, get_variable(data, 'u', env))
  ax1.plot(xr, get_variable(rec, 'u', env), ls='--', c='k')
  ax2.plot(xd, data.p)
  ax2.plot(xr, p.rho, ls='--', c='k')


def plot_rundata(key, z=4, ax_in=None, **kwargs):
  '''
  Plots the 3 basic hydro variables
  '''
  if ax_in is None:
    fig, ax = plt.subplots()
  else:
    ax = ax_in
  
  env = MyEnv(key)
  plot_earlydata('rho', 'rho3_sc', key=key, ax_in=ax, label='$\\rho/\\rho_3$', **kwargs)
  plot_earlydata('lfac', 'lfac0', key=key, ax_in=ax, label='$\\Gamma/\\Gamma_0$', **kwargs)
  plot_earlydata('p', 'p_shsc', key=key, ax_in=ax, label='$p/p_0$', **kwargs)
  if ax_in is None:
    ax.set_title(key)
    ax.legend()

### Fitting the hydro 
dfvars = {'lfac': 'lfac', 'ShSt': 'ShSt', 'nu': 'nup_m2', 'L': 'Lbol'}
def get_norms(env, z):
  z_norms = {
    1: {'rho':env.rho2_sc, 'p':env.p_shsc,
      'lfac': env.lfac0, 'ShSt': env.lfac21-1, 'nu': env.nu0pFS, 'L': env.LbolpFS},
    4: {'rho':env.rho3_sc, 'p':env.p_shsc,
      'lfac': env.lfac0, 'ShSt': env.lfac34-1, 'nu': env.nu0p,   'L': env.Lbolp  },
    }
  return z_norms[z]

def plot_fit_hydro(key, z, varname, ax_in=None,
    fullrescale=False, xvar='x', **kwargs):
  '''
  Plot the fit vs data for one of the hydro var
  '''
  func = smooth_bpl0_apy

  if ax_in is not None:
    ax = ax_in
  else:
    fig, ax = plt.subplots()

  env = MyEnv(key)
  data = open_rundata(key, z)
  if type(data) == bool:
    data = extract_data_thinshell(key, cells=[z], noOut=False)[0]
  x_d = data.x.to_numpy() * c_ / env.R0
  norms = get_norms(env, z)
  if varname not in norms:
    print("Variable 'varname' must be in", list(norms))
    return 0.
  mult = x_d if varname == 'nu' else 1.
  y = get_variable(data, dfvars[varname], env) * mult / norms[varname]

  # fitfuncs = (smooth_bpl_apy, smooth_bpl0_apy)
  # noBeta = True # if varname=='lfac' else False
  # popt = get_fitting_smoothBPL_v2(x_d, y,
  #   noBeta=noBeta, fitfuncs=fitfuncs, initBreak=True)
  # if noBeta:
  #   func = fitfuncs[1]
  # else:
  #   func = fitfuncs[0]
  # fit = func(x_d, *popt)

  popt = get_fitting_smoothBPL_new(x_d, y)
  fit = func(x_d, *popt)
  
  x = data.index.astype(int) if xvar == 'it' else x_d
  fitcolor = 'k'
  if (ax_in is not None) and ('c' in kwargs):
    fitcolor = kwargs.get('c')

  # smooth data to see a bit better what's going on
  # same smoothing as in the fitting method
  y = savgol_filter(y, 7, 3)
  if fullrescale:
    scale = func(1., *popt)
    # force y(0)=1
    y /= scale
    fit /= scale
  line, = ax.loglog(x, y, **kwargs)
  ax.loglog(x, fit, ls='--', c=fitcolor, lw=.9)

  # if y[-1] >= 1:
  #   ax.set_ylim((0.98, 1.05*y[-1]))
  # else:
  #   ax.set_ylim((0.9*y[-1], 1.1))

  if ax_in is None:
    ax.set_title(varname)
    create_legend_colors(ax, ['data', 'fit'], ['C0', 'k'])

def plot_fit_hydro_both(key, varname):
  fig, ax = plt.subplots()
  colors = ['r', 'b']
  labels = ['RS', 'FS']
  for z, col, label in zip([4, 1], colors, labels):
    plot_fit_hydro(key, z, varname, ax_in=ax,
      c=col, label=label)
  ax.set_title('run '+ key)
  ax.legend()

def plot_hydro_fitted(key, z):
  '''
  Plot hydro and fitted for basic hydro vars
  '''

  fig, ax = plt.subplots()
  vars = ['lfac', 'ShSt', 'nu', 'L']
  exps = {
    'lfac':'$\\Gamma$', 'ShSt':'$\\Gamma_{ud}-1$',
    'nu':"$\\nu'\\times R$", 'L':"$L'_{\\rm bol}$"
  }
  colors = ['C0', 'C1', 'C2', 'C3']
  legnames = []
  for varname, col in zip(vars, colors):
    plot_fit_hydro(key, z, varname, ax_in=ax, c=col)
    legnames.append(exps[varname])
  
  create_legend_colors(ax, legnames, colors)

def reconstruct_fits(au, popt_lfac, z=4, upstream='r-2', geometry='spherical'):
  '''
  Return callable shapes of (Gamma_ud-1), nu'_m (x R/R_0) and L'_bol as functions
  of x = R/R_0, reconstructed analytically from the Gamma_d(x)/Gamma_d,0 fit
  (popt_lfac) and the proper-velocity ratio a_u alone. All shapes normalized to
  1 at x=1.

  Parameters:
    au        : proper velocities ratio
    popt_lfac : smooth_bpl0_apy parameters of f_Gamma(x) = Gamma_d(x)/Gamma_d,0
    z         : 4 for RS (upstream = shell 4), 1 for FS (upstream = shell 1)
    upstream  : 'uniform' (rho_u = const) or 'r-2' (rho_u ~ R^-2)
    geometry  : 'spherical' (A ~ R^2) or 'cartesian' (A = const), affects L'_bol only

  Returns:
    func_ShSt, func_nu, func_L : callables x -> shape of (Gamma_ud-1),
      nu'_m * R/R_0 and L'_bol respectively
  '''
  K_u      = np.sqrt(2./(1. + au*au))
  # ratio of Gamma_upstream / Gamma_downstream at x=1
  ratio0   = (1./K_u) if z == 4 else (1./(au*K_u))
  Gud0     = 0.5*(K_u + 1./K_u) if z == 4 else 0.5*(au/K_u + K_u/au)
  Gud0_m1  = Gud0 - 1.
  beta_ud0 = np.tanh(np.abs(np.log(ratio0)))

  # x-exponent that combines rho_u(x) and area scaling
  if upstream == 'uniform':
    n_nu, n_L = 1, 2
  elif upstream == 'r-2':
    n_nu, n_L = 0, 0
  else:
    raise ValueError("upstream must be 'uniform' or 'r-2'")
  if geometry == 'cartesian':
    n_L -= 2

  def _Gud_at(x):
    # in the UR limit, Gamma_ud depends only on the ratio r(x) = Gamma_u/Gamma_d(x)
    fG = smooth_bpl0_apy(x, *popt_lfac)
    d_eta   = np.abs(np.log(ratio0/fG))
    Gud_m1  = 2.*np.sinh(d_eta/2.)**2
    Gud     = 1. + Gud_m1
    beta_ud = np.tanh(d_eta)
    return Gud, Gud_m1, beta_ud

  def func_ShSt(x):
    _, Gud_m1, _ = _Gud_at(x)
    return Gud_m1/Gud0_m1

  def func_nu(x):
    Gud, Gud_m1, _ = _Gud_at(x)
    return (Gud_m1/Gud0_m1)**2.5 * np.sqrt(Gud/Gud0) * x**n_nu

  def func_L(x):
    Gud, Gud_m1, beta_ud = _Gud_at(x)
    return (Gud_m1*Gud*beta_ud)/(Gud0_m1*Gud0*beta_ud0) * x**n_L

  return func_ShSt, func_nu, func_L


def compare_lfac_derived(key, z, ax_in=None, upstream='r-2'):
  '''
  For ShSt = Gamma_ud-1, nu'_m * R/R0, and L'_bol, compare:
    - raw values from the downstream cell
    - smooth-bpl fit of each quantity (current method)
    - expected evolution derived analytically from the Gamma_d(x) fit alone,
      assuming Gamma_u = const and an analytic upstream density profile.

  upstream : 'uniform' (rho_u = const, shock inside cold shell) or
             'r-2'    (rho_u ~ R^-2, post-crossing external medium).
  '''
  func = smooth_bpl0_apy

  env = MyEnv(key)
  data = open_rundata(key, z)
  x = data.x.to_numpy() * c_ / env.R0
  norms = get_norms(env, z)

  # --- Fit Gamma_d(x) (normalized by env.lfac0)
  y_lfac = get_variable(data, dfvars['lfac'], env) / norms['lfac']
  popt_lfac = get_fitting_smoothBPL_new(x, y_lfac)
  y_lfac_fit = func(x, *popt_lfac)

  # # --- Expected (normalized) shapes derived from popt_lfac and a_u only.
  # geometry = getattr(env, 'geometry', 'spherical')
  # fn_ShSt, fn_nu, fn_L = reconstruct_fits(env.a_u, popt_lfac, z=z,
  #     upstream=upstream, geometry=geometry)
  # expected = {'ShSt': fn_ShSt(x), 'nu': fn_nu(x), 'L': fn_L(x)}

  lfac_d_fit = y_lfac_fit * env.lfac0
  beta_d_fit = derive_velocity(lfac_d_fit)

  # --- Reconstruct Gamma_ud(x) from Gamma_d(x) and Gamma_u
  # Use rapidity to avoid catastrophic cancellation when both beta's are ~1:
  #   Gamma_ud = cosh(d_eta),  Gamma_ud - 1 = 2 sinh^2(d_eta/2) >= 0,
  #   beta_ud  = tanh(d_eta).
  lfac_u   = env.lfac1 if z == 1 else env.lfac4
  lfac_ud0 = env.lfac21 if z == 1 else env.lfac34
  beta_ud0 = derive_velocity(lfac_ud0)

  u_u = np.sqrt(lfac_u**2 - 1.)
  u_d = np.sqrt(np.maximum(lfac_d_fit**2 - 1., 0.))
  d_eta   = np.abs(np.arcsinh(u_u) - np.arcsinh(u_d))
  Gud_m1  = 2.*np.sinh(d_eta/2.)**2
  Gud_fit = 1. + Gud_m1
  beta_ud = np.tanh(d_eta)
  Gud_m1_0 = lfac_ud0 - 1.

  # --- Expected (normalized) curves from the Gamma_d fit only.
  # Microphysics is identical in both cases:
  #   gamma_m   ~ (Gud - 1)
  #   B'        ~ sqrt(e'_int) ~ sqrt(Gud (Gud-1) rho_u)
  #   nu'_m     ~ B' gamma_m^2 ~ (Gud-1)^(5/2) sqrt(Gud) sqrt(rho_u)
  #   L'_bol    ~ e'_int beta_ud R^2 ~ Gud (Gud-1) beta_ud rho_u R^2
  # The upstream density profile only changes the rho_u(x) factor:
  #   'uniform' : rho_u = rho_u0           -> L ~ Gud(Gud-1)beta_ud * x^2
  #                                           nu*x ~ (Gud-1)^(5/2) sqrt(Gud) * x
  #   'r-2'     : rho_u = rho_u0 / x^2     -> L ~ Gud(Gud-1)beta_ud   (x cancels)
  #                                           nu*x ~ (Gud-1)^(5/2) sqrt(Gud) (x cancels)
  ShSt_exp = Gud_m1 / Gud_m1_0
  shape_nu = (Gud_m1 / Gud_m1_0)**2.5 * np.sqrt(Gud_fit / lfac_ud0)
  shape_L  = (Gud_m1 * Gud_fit * beta_ud) / (Gud_m1_0 * lfac_ud0 * beta_ud0)
  if upstream == 'uniform':
    nu_exp = shape_nu * x
    L_exp  = shape_L  * x**2
  elif upstream == 'r-2':
    nu_exp = shape_nu
    L_exp  = shape_L
  else:
    raise ValueError("upstream must be 'uniform' or 'r-2'")
  if getattr(env, 'geometry', 'spherical') == 'cartesian':
    # cartesian: r = R0 (constant), so the x^2 from spherical area is absent.
    L_exp = L_exp / x**2

  expected = {'ShSt': ShSt_exp, 'nu': nu_exp, 'L': L_exp}

  # --- Plot raw / direct fit / from-lfac-fit for each of ShSt, nu, L
  vars   = ['ShSt', 'nu', 'L']
  labels = {'ShSt': '$\\Gamma_{ud}-1$',
            'nu':   "$\\nu'_m\\, R/R_0$",
            'L':    "$L'_{\\rm bol}$"}
  colors = {'ShSt': 'C0', 'nu': 'C1', 'L': 'C2'}

  if ax_in is None:
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(6, 10))
  else:
    axes = ax_in
    fig  = axes[0].figure

  y_lfac_smooth = savgol_filter(y_lfac, 7, 3)
  axes[0].loglog(x, y_lfac_smooth, lw=1.2)
  axes[0].loglog(x, y_lfac_fit, ls='--', c='k', lw=0.9)
  axes[0].set_ylabel('$\\Gamma$')
  axes[0].grid(True, which='both', alpha=.3)
  for ax, vname in zip(axes[1:], vars):
    mult = x if vname == 'nu' else 1.
    y_raw = get_variable(data, dfvars[vname], env) * mult / norms[vname]
    popt  = get_fitting_smoothBPL_new(x, y_raw)
    y_smooth = savgol_filter(y_raw, 7, 3)
    y_fit    = func(x, *popt)
    y_exp = expected[vname]
    y_exp *= y_fit[0]/y_exp[0] # we care about shape, sometimes norms are !=

    ax.loglog(x, y_smooth, c=colors[vname], lw=1.2, label='data')
    ax.loglog(x, y_fit, ls='--', c='k',          lw=0.9, label='direct fit')
    ax.loglog(x, y_exp, ls=':',  c='r',          lw=1.1, label="from $\\Gamma_d$ fit")
    ax.set_ylabel(labels[vname])
    ax.grid(True, which='both', alpha=.3)

  axes[1].legend(loc='best')
  axes[-1].set_xlabel('$R/R_0$')
  if ax_in is None:
    axes[0].set_title(f'{key}  (z={z})')
    fig.tight_layout()

def compare_fits_hydro(keys, z, varname,
    fullrescale=True, scalename='log_aum'):
  '''
  Plot the fit vs hydro of one var for several runs
  '''
  if scalename in scale_exps.keys():
    scale_exp = scale_exps[scalename]
  else:
    scale_exp = '$'+ scalename[:-1] + '_' + scalename[-1:] + '$'
  fig, ax = plt.subplots()
  N = len(keys)
  scale_arr = [getattr(MyEnv(key), scalename) for key in keys]
  cmap = plt.cm.jet
  norm = plt.Normalize(vmin=min(scale_arr), vmax=max(scale_arr))
  colors = cmap(norm(scale_arr))
  names = []
  sorted_scales, sorted_keys = zip(*sorted(zip(scale_arr, keys)))
  for i, (scale, key) in enumerate(zip(sorted_scales, sorted_keys)):
    plot_fit_hydro(key, z, varname, ax_in=ax, fullrescale=fullrescale, c=colors[i])
    names.append(f'{scale:.1f}')

  leg_st = create_legend_styles(ax, ['data', 'fit'], ['-', '--'], loc='lower right')
  if N <= 5:
    create_legend_colors(ax, names, colors, title=scale_exp, loc='upper right')
    ax.add_artist(leg_st)
  else:
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, ax=ax, label=scale_exp)
  

def get_effective_asympt(x_inf, alpha, s):
  '''
  Slope of smooth_bpl0 at x=1
  '''
  s_eff = np.sign(-alpha)*np.abs(s)
  a_eff = alpha * (1 - x_inf**s_eff)
  x_break = x_inf ** (1. / a_eff)
  return a_eff, x_break

def a_from_popt(popt, norm=None):
  '''
  Slope of smooth_bpl0_apy at x=1
  '''
  A, x_b, alpha, s = popt
  a = - alpha * (1 + x_b**(-1/s))
  return a

def yinf_from_popt(popt, norm=None):
  '''
  Value at x >> x_b of smooth_bpl0_apy
  '''
  A, x_b, alpha, s = popt
  yinf = A * 2**(-alpha*s)
  if norm is not None:
    yinf /= norm
  return yinf

def get_fit(varname, log_au=0, z=4):
  '''
  Get fitting parameters of varname with smooth_bpl0_apy
  '''
  varnames = ['lfac', 'ShSt', 'nu', 'L']
  key = 'sweep_' + f"log_au={log_au:.1f}"
  env = MyEnv(key)
  data = open_rundata(key, z)
  norms = get_norms(env, z)
  x = data.x.to_numpy() * c_ / env.R0
  mult = x if varname == 'nu' else 1.
  if varname not in norms:
    print("Variable 'varname' must be in", list(norms))
    return 0.
  y = get_variable(data, dfvars[varname], env) * mult / norms[varname]
  popt = get_fitting_smoothBPL_new(x, y)
  return popt, env

def get_fits_values(varname, param, log_au=0, z=4):
  '''Return fitting parameter param of variable varname'''
  varnames = ['lfac', 'ShSt', 'nu', 'L']
  params = ['A', 'x_b', 'alpha', 'beta', 's']
  derived = ['a', 'y_inf']
  der_funcs = [a_from_popt, yinf_from_popt]
  popt, env = get_fit(varname, log_au, z)
  i = 0
  noBeta = True # if varname=='lfac' else False
  if noBeta:
    del params[params.index('beta')]
  if param in params:
    i = params.index(param)
    out = popt[i]
  elif param in derived:
    i = derived.index(param)
    out = der_funcs[i](popt)
  else:
    print("Variable 'param' must be in ", params)
    return 0.
  return out

def plot_fits_values(logau_arr, varname, param, z=4, ax_in=None, **kwargs):
  '''
  Plot a fitting parameters vs log(a_u-1)
  '''
  if ax_in is not None:
    ax = ax_in
  else:
    fig, ax = plt.subplots()
  vals = []
  for log_au in logau_arr:
    val = get_fits_values(varname, param, log_au, z)
    vals.append(val)
  ax.scatter(logau_arr, vals, marker='x', **kwargs)
  if ax_in is None:
    prefix = '\\' if param in ['alpha', 'beta'] else ''
    title = f'${prefix+param}$ from fitting {varname}'
    ax.set_xlabel('log$_{10}(a_u-1)$')
    ax.set_.title(title)

def compare_fits(logau_arr, varnames, param, z=4):
  '''
  Plot several fitting parameters vs log(a_u-1)
  '''
  fig, ax = plt.subplots()
  #colors = plt.cm.jet(np.linspace(0,1,len(varnames)))
  colors = ['C0', 'C1', 'C2', 'C3']
  for i, name in enumerate(varnames):
    plot_fits_values(logau_arr, name, param, z=z, ax_in=ax)#, color=colors[i])
  
  create_legend_colors(ax, varnames, colors)
  ax.set_xlabel('log$_{10}(a_u - 1)$')
  name = 'y_\\infty' if param == 'y_inf' else param
  prefix = '\\' if name in ['alpha', 'beta'] else ''
  params = ['A', 'x_b', 'alpha', 'beta', 's']
  head = 'Fitting parameter ' if name in params else 'Physical quantity '
  ax.set_title(head + f'${prefix+name}$')

def plot_fittedvals(var, names=['lfac', 'ShSt', 'nu', 'L'], fronts=['RS', 'FS']):
  '''
  Same as compare_fits but from the extracted values
    instead of fitting each dataset
  '''
  vars = ['A', 'x_b', 'alpha', 's']
  varsexps = {
    key:f'${"\\" if key=='alpha' else ""}{key}$' for key in vars
  }
  ylabel = varsexps[var]
  if var not in vars:
    print("Variable var must be in: ", vars)
    return 0.

  fig, ax = plt.subplots()
  ax.set_xlabel('log$_{10}(a_u-1)$')
  ax.set_ylabel(ylabel, rotation='horizontal')
  exps = {
    'lfac':'$\\Gamma$', 'ShSt':'$\\Gamma_{ud}-1$',
    'nu':"$\\nu'\\times R$", 'L':"$L'_{\\rm bol}$"
  }
  styles = ['-', '--']
  colors = ['C0', 'C1', 'C2', 'C3', 'C4']
  colors = colors[:len(names)]
  for front, ls in zip(fronts, styles):
    df = open_sweep(front)
    #df = df.loc[np.abs(df.log_aum) <= 0.5]
    keys = df.keys()
    x = df.log_aum
    for name, col in zip(names, colors):
      y = df[name + '_' + var]
      if var == 'alpha':
        y = -y
      ax.plot(x, y, c=col, ls=ls)
      ax.scatter(x, y, c='k', marker='x')

  legcol = create_legend_colors(ax, [exps[name] for name in names], colors, loc='lower left')
  #create_legend_styles(ax, fronts, styles)
  #ax.add_artist(legcol)

  fig.tight_layout()

### Fitting the peaks
def plot_fit_angle(name, z=4):
  '''
  Plot a fitting parameter of xi_eff
  '''
  key = 'xi_' + name
  subs = ['k', 'A', 'x_b', 's']
  names = ['xi_' + s for s in subs]
  A0 = 6*2**(-2)
  bounds = {'k':(1e-4, 1.), 'A':(A0*0.01, A0*100.), 
    'x_b':(1e-3, 100), 's':(1e-3, 3)}
  if name not in subs:
    print("Variable 'name' must be in ", subs)
    return 0.
  front = 'FS' if z==1 else 'RS'
  df = open_sweep(front)
  x = df.log_aum.to_numpy()
  y = df[key].to_numpy()
  
  plt.figure()
  plt.plot(x, y)
  for b in bounds[name]:
    plt.axhline(b, ls=':', c='k')

def plot_xieff(key, z, ax_in=None, **kwargs):
  '''
  xi_max (EATS size) as function of t = g^2 bar{T}
  '''
  if ax_in is None:
    fig, ax = plt.subplots()
  else:
    ax = ax_in
  
  front = 'FS' if z==1 else 'RS'
  df = open_sweep(front)
  env = MyEnv(key)
  logau_key = np.round(np.log10(env.a_u-1), 1)
  logau_arr = df.log_aum.to_numpy()
  i_row = find_closest(logau_arr, logau_key)
  xi_keys = ['xi_' + s for s in ['k', 'sat', 's']]
  k, xi_sat, s_eats = df.iloc[i_row][xi_keys].to_numpy()
  t = np.linspace(0, 20)
  y = k * (t**(-s_eats) + xi_sat**(-s_eats))**(-1/s_eats)
  ax.plot(t, y)
  ax.set_title(f'k = {k:.2e}, xi_sat = {xi_sat:.1f}, s = {s_eats:.1f}')
  ax.set_xlabel('$g^2\\bar{T}$')
  ax.set_ylabel('$\\xi_{\\rm eff}$')

def compare_xis(front):
  '''
  Plot the extracted fitting parameters of xi_eff vs log(a_y-1)
  '''
  fig, ax = plt.subplots()
  df = open_sweep(front)
  logau_arr = df['log_aum'].to_numpy()
  N = len(logau_arr)
  cmap = plt.cm.jet
  norm = plt.Normalize(vmin=logau_arr[0], vmax=logau_arr[-1])
  colors = cmap(norm(logau_arr))
  sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
  fig.colorbar(sm, ax=ax, label='log$_{10}(a_u - 1)$')
  params_arr = df[['xi_A', 'xi_x_b', 'xi_s']].to_numpy()
  T = np.linspace(0, 20, 500)
  for log_au, params, col in zip(au_arr, params_arr, colors):
    au = 1 + 10**log_au
    g2 = g2_from_au(au)[0 if front == 'FS' else 1]
    t = g2*T  # T is bar{T} = tilde{T}-1 here
    plot_ximax(t, params, ax_in=ax, c=col)

def get_peaks_data(key, z, withT=False, Tf=None, Tmax=20):
  '''
  Compute flux and extract peaks from a sim
  Tf: float or None, artificial cutoff time forcing high-latitude emission
    (passed through to run_nuFnu_vFC)
  '''
  env = MyEnv(key)
  N = getattr(env, f'Nsh{z}')
  nuobs, Tobs, env = obs_arrays_peakcentred(key, Tmax=Tmax, NT=2*N)
  tT = 1 + (Tobs - env.Ts)/(env.T0FS if z==1 else env.T0)
  data = open_rundata(key, z)
  if type(data) == bool:
    print('Cannot find data for key ', key)
    tT, nupk_data, nFpk_data = 0., 0., 0.
  else:
    d_fit = cellsBehindShock_fromData(data)
    nF = run_nuFnu_vFC(nuobs, Tobs, d_fit, env, Tf=Tf)
    tnu = normalize_freq(nuobs, z, env)
    nupk_data, nFpk_data = extract_peaks(tnu, nF)
  if withT:
    return tT, nupk_data, nFpk_data
  else:
    return nupk_data, nFpk_data

def compare_peaks(keys, z, scalename='log_aum', Tf=None, Tmax_in=20):
  '''
  Compare peaks of sims in keys array with varying values of scalename
  Tf: float or None, artificial cutoff time \tilde{T}_f forcing the transition
    to high-latitude emission (passed through to get_peaks_data/run_nuFnu_vFC)
  '''

  front = 'FS' if z==1 else 'RS'
  title = 'Peak of $(\\nu F_\\nu)_{\\rm ' + front + '}$'
  env = MyEnv(keys[0])
  title_logaum = const_title = ', ' + scale_exps['log_aum'] + f'= {np.log10(env.a_u-1):.1f}'
  title_chi =  scale_exps['chi'] + f'= {env.chi:.1f}'
  if scalename in scale_exps.keys():
    scale_exp = scale_exps[scalename]
    if scalename == 'chi':
      title = title + title_logaum
    else:
      title = title + title_chi
  else:
    scale_exp = '$'+ scalename[:-1] + '_' + scalename[-1:] + '$'
    title = title + title_logaum + title_chi

  # colormap
  N = len(keys)
  scale_arr = [getattr(MyEnv(key), scalename) for key in keys]
  sorted_scales, sorted_keys = zip(*sorted(zip(scale_arr, keys)))
  scale_arr = sorted_scales
  keys = sorted_keys
  names = [f'{scale:.1f}' for scale in scale_arr]
  cmap = plt.cm.Set1
  norm = plt.Normalize(vmin=scale_arr[0], vmax=scale_arr[-1])
  colors = cmap(norm(scale_arr))
  
  # values
  if Tf is not None:
    Tmax = min(2*Tf+1, Tmax_in)
  else:
    Tmax = Tmax_in
  tT_arr, nupks_arr, nFpks_arr = [], [], []
  for key in keys:
    tT, nupk_data, nFpk_data = get_peaks_data(key, z, withT=True, Tf=Tf, Tmax=Tmax)
    tT_arr.append(tT)
    nupks_arr.append(nupk_data)
    nFpks_arr.append(nFpk_data)
  
  # plot
  fig, axs = plt.subplots(2, 1, sharex=True)
  for i, (tT, nupks, nFpks) in enumerate(zip(tT_arr, nupks_arr, nFpks_arr)):
    axs[0].semilogy(tT, nupks, c=colors[i])
    axs[1].plot(tT, nFpks, c=colors[i])
  axs[0].set_ylabel(nupk_label)
  axs[1].set_ylabel(nFpk_label)
  axs[1].set_xlabel(T_label)

  ax = axs[0]
  ax.set_title(title)
  if N <= 5:
    create_legend_colors(ax, names, colors, title=scale_exp, loc='upper right')
  else:
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, ax=ax, label=scale_exp)
  fig.tight_layout()

def get_model_params(key, z):
  '''
  Get parameters (a_u, tau=t_on/t_off, reverse) for C25_peak_model from a sim
  '''
  env = MyEnv(key)
  au = env.a_u
  reverse = False if z==1 else True
  tau = getattr(env, f't{z}')/env.toff
  return au, tau, reverse

def get_paramsfit_sim(key):
  '''
  Returns a_u, tau4 and tau1 (tau=t_on/t_off) from a sim
  '''
  env = MyEnv(key)
  params = [env.a_u]
  for z in [4, 1]:
    tau = getattr(env, f't{z}')/env.toff
    params.append(tau)
  return params

def get_Tf_fromfit(key, z):
  env = MyEnv(key)
  data = open_rundata(key, z)
  au = env.a_u
  tau = getattr(env, f't{z}')/env.toff
  x = data.x.to_numpy() * c_ / env.R0
  y = get_variable(data, 'lfac', env)/env.lfac0
  popt_lfac = get_fitting_smoothBPL_new(x, y)
  Tf = compute_Tf(tau, au, popt_lfac, reverse=(z==4))[0]
  return Tf


def construct_peaks_model(key, z, T, Tmax=20, withlos=False):
  '''
  Get peaks by fitting a sim and putting that in the model 
  '''
  env = MyEnv(key)
  N = getattr(env, f'Nsh{z}')
  data = open_rundata(key, z)
  g2 = env.gFS**2 if z==1 else env.gRS**2
  Tf, t_max, popt_lfac, popt_ShSt, popt_nu, popt_L, popt_xi = get_anglefits(data, Tmax=Tmax, returnAll=True)
  nupk, nFpk = build_peaks_functions(Tf, g2, [popt_lfac, popt_nu, popt_L, popt_xi])(T)
  if nupk[0] < nupk[1]:
    print('Wrong freq initial behavior ')
    print(popt_lfac, popt_nu, popt_xi)
  if withlos:
    popt_xi_los = [0, *popt_xi[1:]]
    nu_los, nF_los  = build_peaks_functions(Tf, g2, [popt_lfac, popt_nu, popt_L, popt_xi_los])(T)
    return nupk, nFpk, nu_los, nF_los
  else:
    return nupk, nFpk


def compare_model_data(key, z, Tmax=20, direct=True, withlos=False):
  '''
  Compare data with the emission model using the parameters from data
  '''
  env = MyEnv(key)
  N = getattr(env, f'Nsh{z}')
  col = 'b' if z==1 else 'r'
  nuobs, Tobs, env = obs_arrays_peakcentred(key, Tmax=Tmax, NT=2*N)
  T = normalize_time(Tobs, z, env)
  au, tau, reverse = get_model_params(key, z)
  if direct:
    if withlos:
      nu_mod, nF_mod, nu_los, nF_los = construct_peaks_model(key, z, T, Tmax, withlos=True)
    else:
      nu_mod, nF_mod  = construct_peaks_model(key, z, T, Tmax, withlos=False)
  else:
    nu_mod, nF_mod = C25_peak_model(T, au, tau, reverse)
  nu_dat, nF_dat = get_peaks_data(key, z, Tmax=Tmax)
  # nu_dat = savgol_smooth(nu_dat)
  # nF_dat = savgol_smooth(nF_dat)

  # Tf in construct_peaks_models directly from sim
  # compare with the one from method

  Tf_method = get_Tf_fromfit(key, z)
  fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True)
  ax0.semilogy(T, nu_dat, c=col, label='simulation')
  ax0.semilogy(T, nu_mod, ls='--', c='k', label='model')
  ax1.plot(T, nF_dat, c=col)
  ax1.plot(T, nF_mod, ls='--', c='k')
  if withlos:
    ax0.semilogy(T, nu_los, ls=':', c='k', label='l.o.s.')
    ax1.plot(T, nF_los, ls=':', c='k')
  #ax0.axvline(Tf_method, ls=':', c='k', lw=.9)
  #ax1.axvline(Tf_method, ls=':', c='k', lw=.9)
  #ax0.set_title(f'$a_u = {au:.1f}$, $\\Delta R/R_0 = {dR:.2f}$')
  ax1.set_xlabel(T_label)
  ax0.set_ylabel(nu_label)
  ax1.set_ylabel(nF_label)
  ax0.legend()
  fig.tight_layout()

def fit_peaks_for_au(T, nupk_data, nFpk_data, reverse=True):
  '''
  Take peak data and try to find corresponding a_u and Delta R/R0
  edit: switch dRR0 to tau 
  '''

  # params : a_u, DeltaR/R0
  def residuals(params, T, nupk_data, nFpk_data):
    #nupk_model_fit, nFpk_model_fit = peaks_model_C25(T, *params, reverse=reverse)
    nupk_model_fit, nFpk_model_fit = C25_peak_model(T, *params, reverse=reverse)
    r1 = nupk_model_fit - nupk_data
    r2 = nFpk_model_fit - nFpk_data  
    return np.concatenate([r1, r2])
  # priors
  # -> prior and bounds on dR could be obtained by peak finding
  # p0 = [3., 1.5]
  # bounds = ([1.3, 0.1], [4.2, 10.])
  p0 = [2., 1.]
  bounds = ([1.1, 0.1], [30, 10.])
  res = least_squares(residuals, p0, bounds=bounds,
    args=(T, nupk_data, nFpk_data))
  popt = res.x
  return popt

def test_fitting(key, z, fit=True, Tmax=10):
  '''
  Compare extracted peaks, the model with params from the sim
  if fit)True, fit the peaks to the model and plot the best-fit parameters
  '''
  dfile_path, dfile_bool = get_runfile(key, z)
  if not dfile_bool:
    extract_data_thinshell(key, cells=[1, 4], noOut=True)
  env = MyEnv(key)
  au = env.a_u
  tau = getattr(env, f't{z}')/env.toff
  N = getattr(env, f'Nsh{z}')
  nuobs, Tobs, env = obs_arrays_peakcentred(key, Tmax=Tmax, NT=2*N)
  T = normalize_time(Tobs, z, env)
  nupk_data, nFpk_data = get_peaks_data(key, z, Tmax=Tmax)
  reverse = False if z==1 else True
  col = 'b' if z==1 else 'r'
  data = open_rundata(key, z)
  data_fit = cellsBehindShock_fromData(data)
  #dR = (data_fit.iloc[-1].x * c_ / env.R0) - 1.
  #nupk_model_sim, nFpk_model_sim = peaks_model_C25(T, env.a_u, dR, reverse=reverse)
  nupk_model_sim, nFpk_model_sim = C25_peak_model(T, au, tau, reverse=reverse)
  fig, (ax0, ax1) = plt.subplots(2, 1)
  ax0.semilogy(T, nupk_data, c=col, label=f'simulation')
  #mod_label = f'sim: ({env.a_u:.2f}, {dR:.2f})' if fit else 'model'
  mod_label = f'sim: ({au:.2f}, {tau:.2f})' if fit else 'model'
  ax0.semilogy(T, nupk_model_sim, ls='--', c='k', label=mod_label)
  ax1.plot(T, nFpk_data, c=col)
  ax1.plot(T, nFpk_model_sim, ls='--', c='k')
  legtitle = ''
  if fit:
    popt = fit_peaks_for_au(T, nupk_data, nFpk_data, reverse)
    nupk_model_fit, nFpk_model_fit = C25_peak_model(T, *popt, reverse=reverse)
    print(f'Initial params: $a_u = {env.a_u:.2f}$, t_{{\\rm on}}/t_{{\\rm off}} = {tau:.2f}')
    print(f'Found from fit: au = {popt[0]:.2f}, tau = {popt[1]:.2f}')
    ax0.semilogy(T, nupk_model_fit, ls=':', c='k', label=f'best fit: ({popt[0]:.2f}, {popt[1]:.2f})')
    ax1.plot(T, nFpk_model_fit, ls=':', c='k')
    legtitle = '$(a_u, t_{{\\rm on}}/t_{{\\rm off}})$'

  ax0.legend(title=legtitle)
  ax1.set_xlabel(T_label)
  ax0.set_ylabel(nu_label)
  ax1.set_ylabel(nF_label)
  fig.tight_layout()

### Fitting the full flux
def get_flux_separated(Tobs, nuobs, key):
  '''
  Get the observed flux (normalized to F0,RS) of the 2 fronts
  '''
  env = MyEnv(key)
  nF_arr = np.zeros((2, len(Tobs), len(nuobs)))
  for i, z in enumerate([4, 1]):
    data = open_rundata(key, z)
    if type(data) == bool:
      extract_data_thinshell(key, cells=[z], savefile=True, noOut=True)
      data = open_rundata(key, z)
    fit = cellsBehindShock_fromData(data)
    nF = run_nuFnu_vFC(nuobs, Tobs, fit, env, norm=False)
    nF_arr[i] += nF
  return nF_arr

def get_flux_data(Tobs, nuobs, key):
  '''
  Get the observed flux (normalized to F0,RS)
  '''
  nF_arr = get_flux_separated(Tobs, nuobs, key)
  nF_tot = nF_arr.sum(axis=0)
  return nF_tot

def fit_flux(T, nu, nF_data):
  '''
  Fit flux as contribution from two fronts
  all inputs are normalized wrt the RS
  Carful, T is bar{T}, not tilde{T}!
  '''

  # params : a_u, DeltaR_RS/R0, DeltaR_FS/R0
  def residuals(params, T, nu, nF_data):
    nF_model = nu * flux_model_C25_normed(T, nu, *params)
    r = (nF_model - nF_data).ravel()
    return r
    
  # priors
  # -> prior and bounds on dR could be obtained by peak finding
  p0 = [2., 1., 1.]
  # bounds on a_u = our currently explored param space
  bounds = ([1.3, 0.1, 0.1], [4.2, 10., 10.])
  res = least_squares(residuals, p0, bounds=bounds,
    args=(T, nu, nF_data))
  
  popt = res.x
  return popt

def get_fit_fullflux(key, 
    Tmax=5, NT=500, lognu_min=-3, lognu_max=2, Nnu=300):
  '''
  Fit the flux (RS + FS) of a sim to find best fit parameters
  '''
  # get flux from simulation
  nuobs, Tobs, env = obs_arrays(key, 
      Tmax=Tmax, NT=NT, lognu_min=lognu_min, lognu_max=lognu_max, Nnu=Nnu)
  nF = get_flux_data(Tobs, nuobs, key)

  # normalize
  # # now from env, but define method to determine norms from data
  nu = nuobs/env.nu0
  T = (Tobs - env.Ts)/env.T0
  nF /= env.nu0F0

  # fit 
  popt = fit_flux(T, nu, nF)
  return popt

### Time-integrated spectra
def compute_timeIntegrated_sp_fromTf(log_au, z, nub, Tf=2.5, Tmax=20, N=500, p=2.5):
  '''
  Time-integrated spectra from model parameters
  '''
  Tb_1 = np.linspace(0, Tf-1, 2*N)
  Tb_2 = np.geomspace(Tf-1, Tmax, N)
  Tb = np.concatenate((Tb_1, Tb_2))
  T = Tb + 1
  au = 10**log_au + 1
  reverse = False if z==1 else True
  # per-time nu F_nu spectrum from a single C25 shock front (flux_modeling)
  nuFnu = flux_model_1shock_C25_fromTf(T, nub, au, Tf, reverse, slopes=(-0.5, -0.5*p))
  f = np.trapezoid(nuFnu, Tb, axis=0)
  return f

def compute_timeIntegrated_sp_fromtau(au, tau, nub, reverse=True, Tmax=10, N=250, p=2.5):
  '''
  Time-integrated spectra from model parameters a_u, tau
  '''
  popts = fitparams_from_au(au, reverse)
  Tf, xf = compute_Tf(tau, au, popts[0], reverse)
  Tb_1 = np.linspace(0, Tf-1, 2*N)
  Tb_2 = np.geomspace(Tf-1, Tmax, N)
  Tb = np.concatenate((Tb_1, Tb_2))
  T = Tb + 1
  nuFnu = flux_model_1shock_C25_fromTf(T, nub, au, Tf, reverse, slopes=(-0.5, -0.5*p))
  f = np.trapezoid(nuFnu, Tb, axis=0)
  return f


def compute_timeIntegrated_withau(logau_arr, z, lognu_min=-3, lognu_max=1, Nnu=200, Tf=2.5, Tmax=20):
  '''
  Time-integrated spectras for an array of log(a_u-1) values
  '''
  nub = np.logspace(lognu_min, lognu_max, Nnu)
  spectras = []
  for log_au in logau_arr:
    f = compute_timeIntegrated_sp_fromTf(log_au, z, nub, Tf=Tf, Tmax=Tmax)
    spectras.append(f)
  return spectras

def compute_timeIntegrated_withTf(Tf_arr, z, lognu_min=-3, lognu_max=1, Nnu=200, log_au=0, Tmax=20):
  '''
  Time-integrated spectras for an array of tilde{T}_f values
  '''
  nub = np.logspace(lognu_min, lognu_max, Nnu)
  spectras = []
  for Tf in Tf_arr:
    f = compute_timeIntegrated_sp_fromTf(log_au, z, nub, Tf=Tf, Tmax=Tmax)
    spectras.append(f)
  return spectras

def plot_timeIntegrated(key, z, lognu_min=-3, lognu_max=1, Nnu=200, Tmax=20):
  '''
  Compute and plot a time-integrated spectra for a given simulation
  '''
  nuobs, Tobs, env = obs_arrays(key, Tmax=Tmax, lognu_min=lognu_min, lognu_max=lognu_max, Nnu=Nnu)
  nu0 = env.nu0FS if z==1 else env.nu0
  nub = nuobs/nu0
  f = compute_timeIntegrated_sp_fromTf(key, z, nub, Tmax)
  
  plt.figure()
  plt.loglog(nub, f)

def compare_timeIntegrated_withau(logau_arr, z, lognu_min=-3, lognu_max=1, Nnu=200, Tf=1.5, Tmax=20):
  '''
  Plot time-integrated spectra for an array of log(a_u-1) values
  '''
  nub = np.logspace(lognu_min, lognu_max, Nnu)
  spectras = compute_timeIntegrated_withau(logau_arr, z,
      lognu_min=lognu_min, lognu_max=lognu_max, Nnu=Nnu, Tf=Tf, Tmax=Tmax)
  N = len(logau_arr)
  fig, ax = plt.subplots()
  cmap = plt.cm.plasma
  norm = plt.Normalize(vmin=logau_arr[0], vmax=logau_arr[-1])
  colors = cmap(norm(logau_arr))
  sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
  plt.colorbar(sm, ax=ax, label='log$_{10}(a_u-1)$')
  for i, (log_au, f) in enumerate(zip(logau_arr, spectras)):
    plt.loglog(nub, f, color=colors[i], label=f'{log_au:.1f}')
  plt.title('Time-integrated spectra with log$_{10}(a_u-1)$, $\\tilde{T}_f='+f'{Tf:.1f}$')
  plt.xlabel(nu_label)
  plt.ylabel(nF_label)

def compare_timeIntegrated_withTf(Tf_arr, z, lognu_min=-3, lognu_max=1, Nnu=200, log_au=0., Tmax=20):
  '''
  Plot time-integrated spectra for an array of tilde{T}_f values
  '''
  nub = np.logspace(lognu_min, lognu_max, Nnu)
  spectras = compute_timeIntegrated_withTf(Tf_arr, z,
      lognu_min=lognu_min, lognu_max=lognu_max, Nnu=Nnu, log_au=log_au, Tmax=Tmax)
  N = len(Tf_arr)
  fig, ax = plt.subplots()
  cmap = plt.cm.plasma
  norm = plt.Normalize(vmin=Tf_arr[0], vmax=Tf_arr[-1])
  colors = cmap(norm(Tf_arr))
  sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
  plt.colorbar(sm, ax=ax, label='$\\tilde{T}_f$')
  for i, (Tf, f) in enumerate(zip(Tf_arr, spectras)):
    ax.loglog(nub, f, color=colors[i])
  plt.title('Time-integrated spectra with $\\tilde{T}_f$, log$_{10}(a_u-1)='+f'{log_au:.1f}$')
  plt.xlabel(nu_label)
  plt.ylabel(nF_label)

def compute_pks_tintegrated(logau_arr, z, lognu_min=-2, lognu_max=0, Nnu=500,
    Tf_max=50, NT=100):
  '''
  Compute pks of nu f_nu with tilde{T}_f for chosen values of log(a_u-1)
  '''

  nub = np.logspace(lognu_min, lognu_max, Nnu)
  Tf_arr = np.geomspace(1+1e-4, Tf_max, NT)
  Tmax = Tf_max*1.5 # make sure all spectra are computed of full pulse
  nupks_arr = []
  nfpks_arr = []
  for i, log_au in enumerate(logau_arr):
    spectras = compute_timeIntegrated_withTf(Tf_arr, z,
            lognu_min=lognu_min, lognu_max=lognu_max, Nnu=Nnu, log_au=log_au, Tmax=Tmax)
    pks = np.array([[nub[np.argmax(f)], f.max()] for f in spectras])
    nupks_arr.append(pks[:,0])
    nfpks_arr.append(pks[:,1])
  return Tf_arr, np.array(nupks_arr), np.array(nfpks_arr)

def compare_pks_tintegrated(logau_arr, z, lognu_min=-2, lognu_max=0, Nnu=300,
    Tf_max=50, NT=100):

  front = 'RS' if z==4 else 'FS'
  title = f'Peak of $(\\nu f_\\nu)_{{{front}}}$ with $\\tilde{{T}}_f$'
  Tf_arr, pks_arr, nfpks_arr = compute_pks_tintegrated(logau_arr, z, 
    lognu_min=lognu_min, lognu_max=lognu_max, Nnu=Nnu,
    Tf_max=Tf_max, NT=NT)

  Tmax = Tf_max*1.5 # make sure all spectra are computed of full pulse
  theoric = 1/(0.805+0.706*Tf_arr)  # R24b eqn (13); Rf/R0 = tilde{T} for cst LF
  cmap = plt.cm.jet
  norm = plt.Normalize(vmin=logau_arr[0], vmax=logau_arr[-1])
  colors = cmap(norm(logau_arr))
  sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

  fig, ax = plt.subplots()
  for i, pks in enumerate(pks_arr):
    ax.loglog(Tf_arr, pks, c=colors[i])
  ylims = ax.get_ylim()
  ax.set_ylim(ylims)
  ax.loglog(Tf_arr, theoric, ls='--', c='k')
  ax.set_ylabel(nupk_label)
  ax.set_xlabel('$\\tilde{T}_f$')
  ax.set_title(title)
  plt.colorbar(sm, ax=ax, label='log $(a_u-1)$')
  create_legend_styles(ax, ['C25', 'R24'], ['-', '--'])
  fig.tight_layout()

def compute_tint_and_atpkflux(nub, au, tau, reverse=True, Tmax=10, N=250, p=2.5):
  popts = fitparams_from_au(au, reverse)
  Tf, xf = compute_Tf(tau, au, popts[0], reverse)
  Tb_1 = np.linspace(0, Tf-1, 2*N)
  Tb_2 = np.geomspace(Tf-1, Tmax, N)
  Tb = np.concatenate((Tb_1, Tb_2))
  T = Tb + 1
  nuFnu = flux_model_1shock_C25_fromTf(T, nub, au, Tf, reverse, slopes=(-0.5, -0.5*p))
  i_f = np.searchsorted(T, Tf)
  nF_atpk = nuFnu[i_f]
  nf = np.trapezoid(nuFnu, Tb, axis=0)
  return nF_atpk, nf
