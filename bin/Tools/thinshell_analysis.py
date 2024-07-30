# -*- coding: utf-8 -*-
# @Author: acharlet

'''
Analysis in the thin shell approximation
NEW IDEA FOR SHOCK POSITION (to implement):
select all shocked cells +1 each side
derive v12 (shock detection cf Ayache+ 22)
polynomial fit, get peak position 
'''

from plotting_scripts_new import *
plt.ion()

cols_dic = {'RS':'r', 'FS':'b', 'RS+FS':'k', 'CD':'purple',
  'CD-':'mediumvioletred', 'CD+':'mediumpurple', '3':'darkred', '2':'darkblue'}
spsh_cols = {'Band':'g', 'plaw':'teal'}
# create plots
# plotlist in 1st page of draft
# hydro plots:
#   plot_rhop(key): rho'(r)
#   plot_lfac(key): Gamma(r) (shock fronts, dowstream of shocks, CD)
#   plot_prs(key): p(r)
#   plot_ShSt: ShSt(r)
# rad plots: lightcurve & spectrum, cartesian/spherical - Band/synchrotron,
#   plot_spectrum(key, logT, func): spectral shape at Tb = 10**logT, spectral shape func
#   plot_lightcurve(key, lognu, func): lightcurve at nu/nu0 = 10**lognu, spectral shape func
#   plot_shBehav(keylist): nuFnu_pk(nu_pk) for different runs
# code time-integrated spectrum, integrate up to convergence (code with variable Tbmax I guess)

### plots

  

def plot_contribs(key, varlist=['nu_m', 'Lth'], itmin=0, itmax=None):
  sh_names = ['RS', 'FS']
  var_in = varlist.copy()
  contribs = flux_contribs_run(key, varlist, itmin, itmax)

  var_in.insert(0, 'i_d')

  fig, axs = plt.subplots(len(var_in),1,sharex='all')
  dummy_col = [plt.plot([],[], c=cols_dic[name], ls='-')[0] for name in sh_names]

  for sh, cont in zip(sh_names, contribs):
    for i, var in enumerate(cont[1:]):
      axs[i].semilogy(cont[0][1:], var[1:], c=cols_dic[sh])
      axs[i].scatter(cont[0][1:], var[1:], s=4., c='k', marker='x')
  axs[0].set_yscale('linear')
  fig.legend(dummy_col, sh_names)
  
  for i, var, ax in zip(range(len(axs)), var_in, axs):
    label = var_exp[var] #if i else '$i_{sh}$'
    ax.set_ylabel(label) 
    #ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    #ax.yaxis.set_minor_formatter(ticker.ScalarFormatter())
  axs[-1].set_xlabel('it')

def plot_hydroPanels(key):
  '''
  Plots hydro variable in a column panel
  '''
  fig, axs = plt.subplots(4, 1, figsize=(6,12), layout='constrained', sharex='all')
  #fig.set_figheight(8)
  axs[-1].set_xlabel('$r/R_0$')
  plotfuncs = [plot_rhop, plot_lfac, plot_prs, plot_ShSt]

  for plotfunc, ax in zip(plotfuncs, axs):
    plotfunc(key, ax_in=ax)
  fig.set_figheight(6)
  names = ['RS', '3', 'CD-', 'CD', 'CD+', '2', 'FS']
  legnames = ['RS', '3', 'CD$_3$', 'CD', 'CD$_2$', '2', 'FS']
  dummy_col = [plt.plot([],[], c=cols_dic[name], ls='-')[0] for name in names]
  fig.legend(dummy_col, legnames, handlelength=1.5, loc='outside center right')
  #plt.tight_layout()

def plot_hydroPanels_reduced(key):
  '''
  Plots hydro variable in a column panel
  '''
  fig, axs = plt.subplots(2, 1, layout='constrained', sharex='all')
  #fig.set_figheight(8)
  axs[-1].set_xlabel('$r/R_0$')
  plotfuncs = [plot_lfac, plot_ShSt]

  for plotfunc, ax in zip(plotfuncs, axs):
    plotfunc(key, ax_in=ax)
  fig.set_figheight(6)
  names = ['RS', '3', 'CD', '2', 'FS']
  dummy_col = [plt.plot([],[], c=cols_dic[name], ls='-')[0] for name in names]
  fig.legend(dummy_col, names, handlelength=1.5, loc='outside center right')
  #plt.tight_layout()

def plot_radPanels_t(keylist=['sph_fid', 'cart_fid'], func='Band'):
  '''
  Same as radPanel_t but to compare two or more runs
  '''
  fig = plt.figure()
  plotfuncs = [plot_nupk, plot_nupkFnupk]
  gs = fig.add_gridspec(len(plotfuncs), 1, wspace=0)
  axs = gs.subplots(sharex='all')
  names, colors = ['RS', 'FS'], ['r', 'b']
  lstyles = ['-', '-.']
  runnames = ['sph', 'hybrid'] if keylist == ['sph_fid', 'cart_fid'] else keylist
  dummy_col = [plt.plot([],[], c=col, ls='-')[0] for name, col in zip(names, colors)]
  dummy_lst = [plt.plot([],[], ls=ls, color='k')[0] for ls in lstyles]
  ylabels = ['$\\nu_{pk}/\\nu_0$', "$\\nu_{pk}F_{\\nu_{pk}}/\\nu_0F_0$"]

  for label, plotfunc, ax in zip(ylabels, plotfuncs, axs):
    ax.set_ylabel(label)
    for key, ls in zip(keylist, lstyles):
      plotfunc(key, func=func, ax_in=ax, ls=ls)
    ax.set_xscale('linear')
  axs[-1].set_yscale('linear')
  axs[-1].set_xlim((0,5))
  axs[-1].set_xlabel("$\\bar{T}$")
  axs[0].set_ylim(ymin=5e-3)
  axs[0].legend(dummy_col, names, handlelength=1.5)
  axs[1].legend(dummy_lst, runnames)
  plt.tight_layout()


def plot_radPanel_t(key, func='Band'):
  '''
  Plots nu_pk, L_pk, and their product with time in a single figure
  '''
  fig = plt.figure()
  plotfuncs = [plot_nupk, plot_nupkFnupk]
  gs = fig.add_gridspec(len(plotfuncs), 1, wspace=0)
  axs = gs.subplots(sharex='all')
  names, colors = ['RS', 'FS'], ['r', 'b']
  dummy_col = [plt.plot([],[], c=col, ls='-')[0] for name, col in zip(names, colors)]

  ylabels = ['$\\nu_{pk}/\\nu_0$', "$\\nu_{pk}F_{\\nu_{pk}}/\\nu_0F_0$"]
  fig.supxlabel("$\\bar{T}$")

  for label, plotfunc, ax in zip(ylabels, plotfuncs, axs):
    ax.set_ylabel(label)
    plotfunc(key, func=func, ax_in=ax)
    ax.set_xscale('linear')
  axs[-1].set_yscale('linear')
  axs[-1].set_xlim((0,5))
  axs[0].set_ylim(ymin=9e-3)
  fig.legend(dummy_col, names, handlelength=1.5)


def plot_radPanels(key):
  '''
  Plots hydro variable in a column panel
  '''
  fig, axs = plt.subplots(3, 1, figsize=(6,12), layout='constrained', sharex='all')
  #fig.set_figheight(8)
  axs[-1].set_xlabel('$r/R_0$')
  plotfuncs = [plot_num, plot_Lnum, plot_numLnum]
  ylabels = ['$\\nu_m/\\nu_0$', "$L_{\\nu_m}/L_0$",
    "$\\nu_m L_{\\nu_m}/\\nu_0L_0$"]

  for plotfunc, ax, ylabel in zip(plotfuncs, axs, ylabels):
    plotfunc(key, ax_in=ax)
    ax.set_ylabel(ylabel)
  fig.set_figheight(6)
  names = ['RS', '3', 'CD-', 'CD', 'CD+', '2', 'FS']
  dummy_col = [plt.plot([],[], c=cols_dic[name], ls='-')[0] for name in names]
  fig.legend(dummy_col, names, handlelength=1.5, loc='outside center right')
  #plt.tight_layout()

def plot_compareAll(var, lognu=-1, logT=0, Nnu=200, theory=False,
  pres_mode=False, art_mode=False):
  '''
  Plots 2x1 of the selected var comparing between runs
  'lc' lightcurves - 'sp' (instantaneous) spectra - 
  'tsp' time-integrated spectra - 'bhv' spectral behavior
  pres_mode: presentation mode (only RS for bhv, colored spec shape)
  art_mode: article mode (adds hybrid Band onto syn-BPL panel)
  '''
  runs = ['sph_fid', 'cart_fid']
  shapes = ['Band', 'plaw']

  Nr, Ns = len(runs), len(shapes)
  fig = plt.figure()
  if pres_mode and var=='bhv':
    fig.set_size_inches((5,7.2))
  gs = fig.add_gridspec(Nr, 1, hspace=0.1, wspace=0)
  axs = gs.subplots(sharex='all', sharey='all')
  title = ''
  ylabel = '$\\nu F_\\nu/\\nu_0 F_0$'
  xlabel = '$\\bar{T}$' if var=='lc' else '$\\tilde{\\nu}$'
  names = ['RS', 'FS', 'RS+FS']
  lstyles = ['-', '-.', ':']


  if var == 'lc':
    title = 'lightcurves at $\\log_{10}\\tilde{\\nu}=$' + f'{lognu:d}'
    def plotfunc(key, func, ax_in, **kwargs):
      plot_lightcurve(key, lognu, func, theory, totonly=False, ax_in=ax_in, Nnu=Nnu, **kwargs)
    def plot_hybtotline(ax_in):
      plot_lightcurve('cart_fid', lognu, 'Band', theory=False, totonly=True, ax_in=ax_in, Nnu=Nnu, ls=lstyles[-1])
  elif var == 'sp':
    title = 'spectras at $\\bar{T}=$'+f'{10**logT:.1f}'
    def plotfunc(key, func, ax_in, **kwargs):
      plot_spectrum(key, logT, func, theory, totonly=False, ax_in=ax_in, Nnu=Nnu, **kwargs)
    def plot_hybtotline(ax_in):
      plot_spectrum('cart_fid', logT, 'Band', theory=False, totonly=True, ax_in=ax_in, Nnu=Nnu, ls=lstyles[-1])
  elif var == 'tsp':
    title = 'time-integrated spectras'
    ylabel = '$\\nu f_\\nu/\\nu_0 F_0 T_0$'
    def plotfunc(key, func, ax_in, **kwargs):
      plot_integSpec(key, func, theory=theory, ax_in=ax_in, Nnu=Nnu, **kwargs)
    def plot_hybtotline(ax_in):
      plot_integSpec('cart_fid', 'Band', theory=False, totonly=True, ax_in=ax_in, Nnu=Nnu, ls=lstyles[-1])
  elif var == 'bhv':
    title = 'spectral behavior'
    def plotfunc(key, func, ax_in, **kwargs):
      plot_radBehav(key, func, False, ax_in, Nnu=Nnu,
          pres_mode=pres_mode, **kwargs)
  
  fig.supylabel(ylabel)
  axs[-1].set_xlabel(xlabel)
  axs[0].set_title(title)
  dummy_lst = [plt.plot([],[], ls=ls, color='k')[0] for ls in ['-', '-.']]
  if not pres_mode:
    loc = 'upper left' if var == 'bhv' else 'upper right'
    dummy_col = [plt.plot([],[], c=cols_dic[name], ls='-')[0] for name in names]
    fig.legend(dummy_col, names, loc=loc)


  lstlegend = axs[-1].legend(dummy_lst, ['sph.', 'hybrid'])

  fig.add_artist(lstlegend)

  for func, ax in zip(shapes, axs):
    ax.yaxis.set_label_position("right")
    ylabel = 'syn-BPL' if func == 'plaw' else func
    col = spsh_cols[func] if pres_mode else 'k'
    ax.set_ylabel(ylabel, c=col)
    for key, ls in zip(runs, lstyles):
      plotfunc(key, func, ax, ls=ls)
  if art_mode and var != 'bhv':
    plot_hybtotline(axs[-1])



def plot_comparedRad(key, lognu=-1, logT=0, Nnu=200, theory=True):
  '''
  Creates combined plot (2 lightcurves, 2 spectras) to compare Band & plaw
  at given lognu and logT
  '''
  fig = plt.figure()
  env = MyEnv(key)
  fig.suptitle(env.runname)
  gs = fig.add_gridspec(2, 2, hspace=0, wspace=0)
  axs = gs.subplots(sharex='col', sharey='col')
  ax_lcb, ax_lcp, ax_spb, ax_spp = axs[0][0], axs[1][0], axs[0][1], axs[1][1]
  
  ax_spb.yaxis.tick_right()
  ax_spb.yaxis.set_label_position("right")
  ax_spp.yaxis.tick_right()
  ax_spp.yaxis.set_label_position("right")
  ax_lcb.set_title('$\\log\\tilde{\\nu}=$'+f'{lognu:d}')
  ax_spb.set_title('$\\log\\bar{T}=$'+f'{logT:d}')
  ax_spb.set_ylabel('Band',c=spsh_cols['Band'])
  ax_spp.set_ylabel('syn-BPL', c=spsh_cols['plaw'])
  fig.supylabel('$\\nu F_\\nu/\\nu_0 F_0$')
  ax_lcp.set_xlabel('$\\bar{T}$')
  ax_spp.set_xlabel('$\\tilde{\\nu}$')

  plot_lightcurve(key, lognu, 'Band', theory=theory, ax_in=ax_lcb, Nnu=Nnu)
  plot_lightcurve(key, lognu, 'plaw', theory=theory, ax_in=ax_lcp, Nnu=Nnu)
  plot_spectrum(key, logT, 'Band', theory=theory, ax_in=ax_spb, Nnu=Nnu)
  plot_spectrum(key, logT, 'plaw', theory=theory, ax_in=ax_spp, Nnu=Nnu)

def plot_spectrum(key, logT, func='Band',
  theory=False, hybrid=True, totonly=False, newmethod=True,
  ax_in=None, Nnu=200, **kwargs):
  '''
  Plots instantaneous spectrum at \bar{T} = 10**logT
  '''
  nuobs, Tobs, env = get_radEnv(key, Nnu)
  Tb = (Tobs-env.Ts)/env.T0
  T = np.array([env.T0*10**logT + env.Ts])
  nub = nuobs/env.nu0
  nuFnus = nuFnu_thinshell_run(key, nuobs, T, env, func, Nnu)[:,0,:] if newmethod \
    else nuFnu_thinshell_run_v1(key, nuobs, T, env, func, Nnu)[:,0,:]
  nuFnus = np.append(nuFnus, [nuFnus.sum(axis=0)], axis=0)
  nuFnus /= env.nu0F0
  names = ['RS', 'FS', 'RS+FS']
  if totonly: theory = False
  if env.geometry != 'cartesian': hybrid = False

  if not ax_in:
    plt.figure()
    ax = plt.gca()
  else:
    ax = ax_in
  if theory:
    if not ax_in:
      dummy_lst = [plt.plot([],[], ls=ls, color='k')[0] for ls in ['-', '-.']]
      lstlegend = ax.legend(dummy_lst, ['numerical', 'analytical'],
        fontsize=11, bbox_to_anchor=(0.98, 0.7), loc='upper right', borderaxespad=0.)
      ax.add_artist(lstlegend)
    spRS, spFS = nuFnu_theoric(nuobs, T, env, hybrid=hybrid, func=func)
    spTot = spRS + spFS
    specs = np.array([spRS, spFS, spTot])
    for sp, name in zip(specs, names):
      ax.loglog(nub, sp, c=cols_dic[name], ls='-.')
  if totonly:
    ax.loglog(nub, nuFnus[-1], c='k', **kwargs)
  else:
    for val, name in zip(nuFnus, names):
      ax.loglog(nub, val, c=cols_dic[name], label=name, **kwargs)

  if not ax_in:
    plt.legend()
    plt.xlabel("$\\nu/\\nu_0$")
    plt.ylabel("$\\nu F_{\\nu}/\\nu_0 F_0$")
    funcname = 'syn-BPL' if func == 'plaw' else func
    plt.title(key+', $\\bar{T}=$'+f'{10**logT:.1f}'+', '+funcname)

def plot_lightcurve(key, lognu, func='Band',
  theory=False, hybrid=True, totonly=False, newmethod=True,
  ax_in=None, Nnu=200, **kwargs):
  '''
  Plots lightcurve at nu = 10**lognu nu_O
  '''
  nuobs, Tobs, env = get_radEnv(key, Nnu)
  nuobs = np.array([10**lognu*env.nu0])
  Tb = (Tobs-env.Ts)/env.T0
  nuFnus = nuFnu_thinshell_run(key, nuobs, Tobs, env, func, Nnu)[:,:,0] if newmethod \
    else nuFnu_thinshell_run_v1(key, nuobs, Tobs, env, func, Nnu)[:,:,0]
  nuFnus = np.append(nuFnus, [nuFnus.sum(axis=0)], axis=0)
  nuFnus /= env.nu0F0
  names = ['RS', 'FS', 'RS+FS']
  if totonly: theory = False
  if env.geometry != 'cartesian': hybrid = False

  if not ax_in:
    plt.figure()
    ax = plt.gca()
  else:
    ax = ax_in
  if theory:
    if not ax_in:
      dummy_lst = [plt.plot([],[], ls=ls, color='k')[0] for ls in ['-', '-.']]
      lstlegend = ax.legend(dummy_lst, ['numerical', 'analytical'],
        fontsize=11, bbox_to_anchor=(0.98, 0.7), loc='upper right', borderaxespad=0.)
      ax.add_artist(lstlegend)
    lcRS, lcFS = nuFnu_theoric(nuobs, Tobs, env, hybrid=hybrid, func=func)[:,0,:]
    lcTot = lcRS + lcFS
    nuFnu_th = [lcRS, lcFS, lcTot]
    for lc, name in zip(nuFnu_th, names):
      ax.plot(Tb, lc, c=cols_dic[name], ls='-.')
  if totonly:
    ax.plot(Tb, nuFnus[-1], c='k', **kwargs)
  else:
    for val, name in zip(nuFnus, names):
      ax.plot(Tb, val, c=cols_dic[name], label=name, **kwargs)

  if not ax_in:
    plt.legend()
    plt.xlabel("$\\bar{T}$")
    plt.ylabel("$\\nu F_{\\nu}/\\nu_0 F_0$")
    plt.title(key+', $\\log\\tilde{\\nu}=$'+f'{lognu:d}'+', '+func)

def plot_integSpec(key, func='Band',
  theory=False, hybrid=True, totonly=False, ax_in=None, Nnu=200, **kwargs):
  '''
  Plots time integrated spectrum
  '''
  nuobs, Tobs, env = get_radEnv(key, Nnu)
  nub = nuobs/env.nu0
  nuFnus = nuFnu_thinshell_run(key, nuobs, Tobs, env, func, Nnu)
  fnus = np.trapz(nuFnus, x=Tobs, axis=1)
  fnus = np.append(fnus, [fnus.sum(axis=0)], axis=0)
  fnus /= env.T0*env.nu0F0
  names = ['RS', 'FS', 'RS+FS']
  if totonly: theory=False
  if env.geometry != 'cartesian': hybrid = False

  if not ax_in:
    plt.figure()
    ax = plt.gca()
  else:
    ax = ax_in
  if theory:
    if not ax_in:
      dummy_lst = [plt.plot([],[], ls=ls, color='k')[0] for ls in ['-', '-.']]
      lstlegend = ax.legend(dummy_lst, ['numerical', 'analytical'],
        fontsize=11, loc='upper right', borderaxespad=0.) #bbox_to_anchor=(0.98, 0.7)
      ax.add_artist(lstlegend)
    nuFnus_th = nuFnu_theoric(nuobs, Tobs, env, hybrid=hybrid, func=func)
    fnus_th = np.trapz(nuFnus_th, x=Tobs, axis=2)/env.T0
    fnus_th = np.append(fnus_th, [fnus_th.sum(axis=0)], axis=0)
    for f, name in zip(fnus_th, names):
      ax.plot(nub, f, c=cols_dic[name], ls='-.')
  if totonly:
    ax.loglog(nub, fnus[-1], c='k', **kwargs)
  else:
    for val, name in zip(fnus, names):
      ax.loglog(nub, val, c=cols_dic[name], label=name, **kwargs)
  if not ax_in:
    plt.legend()
    plt.xlabel("$\\nu/\\nu_0$")
    plt.ylabel("$\\nu f_{\\nu}/\\nu_0 F_0 T_0$")
    plt.title(key+', '+func)

def plot_radBehav(key, func='Band', theory=False, ax_in=None, Nnu=200, 
    pres_mode=False, **kwargs):
  '''
  Plots nuFnu_pk(nu_pk) of the observed flux
  '''

  if not ax_in:
    plt.figure()
    ax = plt.gca()
  else:
    ax = ax_in
  
  names, colors = ['RS', 'FS'], ['r', 'b']
  nuobs, Tobs, env = get_radEnv(key, Nnu)
  nuFnus = nuFnu_thinshell_run(key, nuobs, Tobs, env, func, Nnu)
  Nit = 1 if pres_mode else nuFnus.shape[0]
  NT = nuFnus.shape[1]

  if theory:
    # nuFnus_th = nuFnu_theoric(nuobs, Tobs, env, geometry=env.geometry, func=func)
    # for k, nuFnu, name, col in zip(range(Nit), nuFnus_th, names, colors):
    #   nupks, nuFnupks = np.zeros((2,NT))
    #   for i in range(NT):
    #     ipk = np.argmax(nuFnu[i])
    #     nupks[i] = nuobs[ipk]
    #     nuFnupks[i] = nuFnu[i,ipk]
    # ax.loglog(nupks/env.nu0, nuFnupks, c=col, ls='-.')
    for k, name in zip(range(Nit), names):
      plot_bhv_hybrid(sh=name, ax_in=ax, ls=':')
    if not ax_in:
      dummy_lst = [plt.plot([],[], ls=ls, color='k')[0] for ls in ['-', ':']]
      loc = 'lower center' if pres_mode else 'upper left'
      lstlegend = ax.legend(dummy_lst, ['numerical', 'analytical'], loc=loc) #bbox_to_anchor=(0.98, 0.7)
      ax.add_artist(lstlegend)

  for k, nuFnu, name, col in zip(range(Nit), nuFnus, names, colors):
    nupks, nuFnupks = np.zeros((2,NT))
    for i in range(NT):
      ipk = np.argmax(nuFnu[i])
      nupks[i] = nuobs[ipk]
      nuFnupks[i] = nuFnu[i,ipk]
    ax.loglog(nupks[1:]/env.nu0, nuFnupks[1:]/env.nu0F0, c=col, label=name, **kwargs)
  
  xmin = 5e-3
  if not pres_mode:
    ax.set_xlim(xmin=xmin)
  ax.set_ylim(ymin=1e-2)

  if not ax_in:
    if not pres_mode:
      plt.legend()
    if pres_mode:
      plt.title(env.geometry)
    plt.xlabel("$\\nu_{pk}/\\nu_0$")
    plt.ylabel("$\\nu_{pk}F_{\\nu_{pk}}/\\nu_0F_0$")

def plot_bhv_hybrid(dRoR=None, sh='RS', ax_in=None, Nnu=200, **kwargs):
  '''
  Plots spectral behavior of chosen front, from fits on hybrid approach
  '''
  if not ax_in:
    plt.figure()
    ax = plt.gca()
  else:
    ax = ax_in

  nuobs, Tobs, env = get_radEnv('cart_fid', Nnu)
  g, Tth, dRR, fac_nu, fac_F, col = (env.gFS, env.T0FS, env.dRFS, env.fac_nu, env.fac_F, 'b') if sh == 'FS' else (env.gRS, env.T0, env.dRRS, 1., 1., 'r')
  tT = 1. + (Tobs - env.Ts)/Tth
  dR = dRoR if dRoR else dRR
  tTf = 1. + dR
  nu, nuFnu = approx_bhv_hybrid(tT, tTf, g)
  nu /= fac_nu
  nuFnu *= fac_F
  ax.loglog(nu, nuFnu, c=col, **kwargs)

  if not ax_in:
    plt.xlabel(f"$\\nu/\\nu_{{{'0,'+sh}}}$")
    plt.ylabel("$\\nu F_{\\nu}/\\nu_0 F_0$")
    plt.tight_layout()

def plot_nupk(key, func='Band', ax_in=None, Nnu=500, **kwargs):
  '''
  Plots nu_pk (\bar{T}) of the observed flux
  '''

  if not ax_in:
    plt.figure()
    ax = plt.gca()
  else:
    ax = ax_in
  
  names, colors = ['RS', 'FS'], ['r', 'b']
  nuobs, Tobs, env = get_radEnv(key, Nnu)
  Tb = (Tobs-env.Ts)/env.T0
  nuFnus = nuFnu_thinshell_run(key, nuobs, Tobs, env, func, Nnu)


  for nuFnu, name, col in zip(nuFnus, names, colors):
    nupks, nuFnupks = np.zeros((2,nuFnu.shape[0]))
    for i in range(nuFnu.shape[0]):
      ipk = np.argmax(nuFnu[i])
      nupks[i] = nuobs[ipk]
    ax.loglog(Tb[1:], nupks[1:]/env.nu0, c=col, label=name, **kwargs)

    

  if not ax_in:
    plt.legend()
    plt.xlabel("$\\bar{T}$")
    plt.ylabel("$\\nu_{pk}/\\nu_0$")

def plot_nupkFnupk(key, func='Band', ax_in=None, Nnu=500, **kwargs):
  '''
  Plots nu_pk F_{nu_pk} (\bar{T}) of the observed flux
  '''

  if not ax_in:
    plt.figure()
    ax = plt.gca()
  else:
    ax = ax_in
  
  names, colors = ['RS', 'FS'], ['r', 'b']
  nuobs, Tobs, env = get_radEnv(key, Nnu)
  Tb = (Tobs-env.Ts)/env.T0
  nuFnus = nuFnu_thinshell_run(key, nuobs, Tobs, env, func, Nnu)

  # if theory:
  #   nuFnus_th = nuFnu_theoric(nuobs, Tobs, env, geometry=env.geometry, func=func)
  #   for nuFnu, name, col in zip(nuFnus_th, names, colors):
  #     nupks, nuFnupks = np.zeros((2,nuFnu.shape[0]))
  #     for i in range(nuFnu.shape[0]):
  #       ipk = np.argmax(nuFnu[i])
  #       nupks[i] = nuobs[ipk]
  #       nuFnupks[i] = nuFnu[i,ipk]
  #   ax.loglog(nupks/env.nu0, nuFnupks, c=col, ls='-.')
  #   if not ax_in:
  #     dummy_lst = [plt.plot([],[], ls=ls, color='k')[0] for ls in ['-', '-.']]
  #     lstlegend = ax.legend(dummy_lst, ['numerical', 'analytical'],
  #       fontsize=11, loc='upper right', borderaxespad=0.) #bbox_to_anchor=(0.98, 0.7)
  #     ax.add_artist(lstlegend)
  for nuFnu, name, col in zip(nuFnus, names, colors):
    nupks, nuFnupks = np.zeros((2,nuFnu.shape[0]))
    for i in range(nuFnu.shape[0]):
      ipk = np.argmax(nuFnu[i])
      nuFnupks[i] = nuFnu[i,ipk]
    ax.loglog(Tb[1:], nuFnupks[1:]/env.nu0F0, c=col, label=name, **kwargs)

  if not ax_in:
    plt.legend()
    plt.xlabel("$\\bar{T}$")
    plt.ylabel("$\\nu_{pk}F_{\\nu_{pk}}/\\nu_0F_0$")

def plot_numLnum(key, ax_in=None, xscale='r'):
  '''
  Plots peak bolometric luminosity with radius
  '''
  env = MyEnv(key)
  file_path = get_fpath_thinshell(key)
  run_data = pd.read_csv(file_path, index_col=0)
  x = run_data['time'].to_numpy()/env.t0
  xlabel = "$t/t_0$" if xscale == 't' else "$r/R_0$"

  names = ['RS', 'FS']
  rlist = run_data[[f'r_{{{name}}}' for name in names]].to_numpy().transpose()
  nulist = run_data[[f'nu_m_{{{name}}}' for name in names]].to_numpy().transpose()
  Llist = run_data[[f'Lth_{{{name}}}' for name in names]].to_numpy().transpose()
  nu0L0 = env.nu0 * env.L0
  nu0L0FS = env.nu0FS * env.L0FS
  thvals = [nu0L0, nu0L0FS]
  if not ax_in:
    plt.figure()
    ax = plt.gca()
  else:
    ax = ax_in
  for r, nu, L, name, th in zip(rlist, nulist, Llist, names, thvals):
    col = cols_dic[name]
    if xscale != 't':
      x = r*c_/env.R0
      ax.loglog(x[r>0.], nu[r>0.]*L[r>0.]/nu0L0, label=f'{name}', c=col)
    else:
      ax.semilogy(x[r>0.], nu[r>0.]*L[r>0.]/nu0L0, label=f'{name}')
    ax.axhline(th/nu0L0, c=col, ls='--', lw=.8)
  ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
  ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
  
  if not ax_in:
    plt.xlabel(xlabel)
    plt.ylabel("$\\nu_mL_{\\nu_m}/\\nu_0 L_0$")
    plt.legend()

def plot_num(key, ax_in=None, xscale='r'):
  '''
  Plots peak observed frequency with radius
  '''

  env = MyEnv(key)
  file_path = get_fpath_thinshell(key)
  run_data = pd.read_csv(file_path, index_col=0)

  names = ['RS', 'FS']
  rlist = run_data[[f'r_{{{name}}}' for name in names]].to_numpy().transpose()
  vallist = run_data[[f'nu_m_{{{name}}}' for name in names]].to_numpy().transpose()
  x = run_data['time'].to_numpy()/env.t0
  xlabel = "$t/t_0$" if xscale == 't' else "$r/R_0$"

  if not ax_in:
    plt.figure()
    ax = plt.gca()
  else:
    ax = ax_in
  for r, val, name, th in zip(rlist, vallist, names, [env.nu0, env.nu0FS]):
    col = cols_dic[name]
    if xscale != 't':
      x = r*c_/env.R0
      ax.loglog(x[r>0.], val[r>0.]/env.nu0, c=col, label=f'{name}')
    else:
      ax.semilogy(x[r>0.], val[r>0.]/env.nu0, c=col, label=f'{name}')
    ax.axhline(th/env.nu0, c=col, ls='--', lw=.8)
  ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
  ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
  
  if not ax_in:
    plt.xlabel(xlabel)
    plt.ylabel("$\\nu_m/\\nu_0$")
    plt.legend()

def plot_Lnum(key, ax_in=None, xscale='r'):
  '''
  Plots peak observed luminosity
  '''

  env = MyEnv(key)
  file_path = get_fpath_thinshell(key)
  run_data = pd.read_csv(file_path, index_col=0)
  x = run_data['time'].to_numpy()/env.t0
  xlabel = "$t/t_0$" if xscale == 't' else "$r/R_0$"

  names = ['RS', 'FS']
  rlist = run_data[[f'r_{{{name}}}' for name in names]].to_numpy().transpose()
  vallist = run_data[[f'Lth_{{{name}}}' for name in names]].to_numpy().transpose()
  thlist = [env.L0, env.L0FS]

  if not ax_in:
    plt.figure()
    ax = plt.gca()
  else:
    ax = ax_in
  for r, val, name, th in zip(rlist, vallist, names, thlist):
    col = cols_dic[name]
    if xscale != 't':
      x = r*c_/env.R0
      ax.loglog(x[r>0.], val[r>0.]/env.L0, c=col, label=f'{name}')
    else:
      ax.semilogy(x[r>0.], val[r>0.]/env.L0, c=col, label=f'{name}')
    ax.axhline(th/env.L0, c=col, ls='--', lw=.8)
  ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
  ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
  
  if not ax_in:
    plt.xlabel(xlabel)
    plt.ylabel("$L_{\\nu_m}/L_0$")
    plt.legend()

def plot_rhop(key, rscale=2., ax_in=None, smoothing=True, kernel_size=50):
  '''
  Plots comoving densities downstream of shocks and on both sides of the CD
  '''
  env = MyEnv(key)
  file_path = get_fpath_thinshell(key)
  run_data = pd.read_csv(file_path, index_col=0)
  if env.geometry == 'cartesian':
    rscale = 0.

  name_int = ['RS', 'CD', 'CD', 'FS']
  rlist = run_data[[f'r_{{{name}}}' for name in name_int]].to_numpy().transpose()
  name_int = ['RS', 'CD-', 'CD+', 'FS']
  rholist = run_data[[f'rho_{{{name}}}' for name in name_int]].to_numpy().transpose()
  name_cols = ['3', 'CD-', 'CD+', '2']


  if not ax_in:
    plt.figure()
    ax = plt.gca()
  else:
    ax = ax_in
  for r, rho, name in zip(rlist, rholist, name_cols):
    col = cols_dic[name]
    r_sc = r[r>0.]*c_/env.R0
    rho_sc = rho[r>0.]*env.rhoscale/env.rho3
      # smooth data
    if smoothing:
      print('Smoothing data')
      kernel = np.ones(kernel_size) / kernel_size
      rho_sc = np.convolve(rho_sc, kernel, mode='same')
    ax.loglog(r_sc, rho_sc*r_sc**rscale, label=name, c=col)
  
  ax.set_ylim(ymin=.75)
  ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
  ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
  ax.axhline(1., c='r', ls='--', lw=.8)
  ax.axhline(env.rho2/env.rho3, c='b', ls='--', lw=.8)
  rscale_str = f"$(r/R_0)^{{{rscale:{".0f" if rscale==int(rscale) else ".1f"}}}}$" if rscale else ""
  ax.set_ylabel("$\\tilde{\\rho'}$" + rscale_str)

  if not ax_in:
    plt.xlabel("$r/R_0$")
    plt.legend()
    plt.tight_layout()

def plot_gfac(key, ax_in=None):
  '''
  Plots the g = Gamma/Gamma_sh factor for each shock
  '''
  env = MyEnv(key)
  file_path = get_fpath_thinshell(key)
  run_data = pd.read_csv(file_path, index_col=0)

  name_int = ['RS', 'FS']
  rlist = run_data[[f'r_{{{name}}}' for name in name_int]].to_numpy().transpose()
  lfaclist = run_data[[f'lfac_{{{name}}}' for name in name_int]].to_numpy().transpose()
  gmaRS, gmaFS = run_get_Gammash(key)
  vallist = lfaclist/np.array([gmaRS, gmaFS])

  if not ax_in:
    plt.figure()
    ax = plt.gca()
  else:
    ax = ax_in
  for r, g, name in zip(rlist, vallist, name_int):
    col = cols_dic[name]
    ax.loglog(r[r>0.]*c_/env.R0, g[r>0.], label=f'$g_{{{name}}}$', c=col)
  ax.axhline(env.gRS, c='r', ls='--', lw=.8)
  ax.axhline(env.gFS, c='b', ls='--', lw=.8)
  ax.set_ylim(ymax=1.24)
  ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
  ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
  ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
  ax.yaxis.set_minor_formatter(ticker.ScalarFormatter())
  ax.yaxis.set_minor_locator(ticker.FixedLocator([0.9, 1., 1.1, 1.2])) 

  if not ax_in:
    plt.xlabel("$r/R_0$")
    plt.legend()
    plt.tight_layout()

def plot_lfac(key, ax_in=None, smoothing=True, kernel_size=10):
  '''
  Plots relevant Lorentz factors (in lab frame):
  Gamma_4, _d,RS, _CD, _d,FS, _1
  '''
  env = MyEnv(key)
  file_path = get_fpath_thinshell(key)
  run_data = pd.read_csv(file_path, index_col=0)

  name_int = ['RS', 'CD', 'FS']
  rlist = run_data[[f'r_{{{name}}}' for name in name_int]].to_numpy().transpose()
  vallist = run_data[[f'lfac_{{{name}}}' for name in name_int]].to_numpy().transpose()
  gmaRS, gmaFS = run_get_Gammash(key, smoothing, kernel_size) 
  name_int = ['RS', 'FS', '3', 'CD', '2']
  rlist = np.stack((rlist[0], rlist[-1], *rlist))
  vallist = np.stack((gmaRS, gmaFS, *vallist))

  if not ax_in:
    plt.figure()
    ax = plt.gca()
  else:
    ax = ax_in

  for r, lfac, name in zip(rlist, vallist, name_int):
    col = cols_dic[name]
    ax.loglog(r[r>0.]*c_/env.R0, lfac[r>0.], c=col, label=f'$\\Gamma_{{{name}}}$')
  ax.axhline(env.lfacRS, c='r', ls='--', lw=.8)
  ax.axhline(env.lfacFS, c='b', ls='--', lw=.8)
  ax.set_ylabel(f"$\\Gamma$")
  ax.set_ylim((105., 145.))
  ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
  ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
  ax.yaxis.set_minor_formatter(ticker.ScalarFormatter())
  ax.yaxis.set_minor_locator(ticker.FixedLocator([110, 120, 130, 140])) 

  if not ax_in:
    plt.xlabel("$r/R_0$")
    plt.legend()
    plt.tight_layout()

def plot_prs(key, rscale=2., ax_in=None, smoothing=True, kernel_size=100):
  '''
  Plots pressures
  '''
  env = MyEnv(key)
  file_path = get_fpath_thinshell(key)
  run_data = pd.read_csv(file_path, index_col=0)

  name_int = ['RS', 'CD', 'FS']
  name_cols = ['3', 'CD', '2']
  rlist = run_data[[f'r_{{{name}}}' for name in name_int]].to_numpy().transpose()
  vallist = run_data[[f'p_{{{name}}}' for name in name_int]].to_numpy().transpose()
  
  # smooth data
  if smoothing:
    kernel = np.ones(kernel_size) / kernel_size
    for p in vallist:
      p = np.convolve(p, kernel, mode='same')

  if not ax_in:
    plt.figure()
    ax = plt.gca()
  else:
    ax = ax_in
  if env.geometry == 'cartesian':
    rscale = 0.

  pscale = env.rhoscale*c_**2/env.p_sh
  for r, p, name in zip(rlist, vallist, name_cols):
    col = cols_dic[name]
    r_sc = r[r>0.]*c_/env.R0
    ax.loglog(r_sc, (p[r>0.]*pscale)*r_sc**rscale, c=col, label=name)

  ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
  ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
  ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
  ax.yaxis.set_minor_formatter(ticker.ScalarFormatter())
  ax.set_ylim((0.69, 1.2))
  rscale_str = f"$(r/R_0)^{{{rscale:{".0f" if rscale==int(rscale) else ".1f"}}}}$" if rscale else ""
  ax.set_ylabel("$\\tilde{p}$" + rscale_str)

  if not ax_in:
    plt.xlabel("$r/R_0$")
    plt.legend()
    plt.tight_layout()

def plot_lfacsh(key, ax_in=None):
  '''
  Plots shock Lorentz factors (in lab frame)
  '''

  env = MyEnv(key)
  file_path = get_fpath_thinshell(key)
  run_data = pd.read_csv(file_path, index_col=0)

  names = ['RS', 'FS']
  rlist = run_data[[f'r_{{{name}}}' for name in names]].to_numpy().transpose()
  lfacsh = run_get_Gammash(key)

  if not ax_in:
    plt.figure()
    ax = plt.gca()
  else:
    ax = ax_in
  for r, lfac, name, valth in zip(rlist, lfacsh, names, [env.lfacRS, env.lfacFS]):
    col = cols_dic[name]
    ax.loglog(r[r>0.]*c_/env.R0, lfac[r>0.], c=col, label=f'$\\Gamma_{{{name}}}$')
    ax.axhline(valth, color='k', ls='--', lw=.6)
    ax.text(1.02, valth, f"$\\Gamma_{{{name+',th'}}}$", color='k', transform=ax.get_yaxis_transform(),
            ha='left', va='center')

  ax.set_ylabel(f"$\\Gamma$")

  if not ax_in:
    plt.xlabel("$r/R_0$")
    plt.legend()
    plt.tight_layout()

def plot_ShSt(key, ax_in=None):
  '''
  Plots shock Lorentz factors (in lab frame)
  '''

  env = MyEnv(key)
  file_path = get_fpath_thinshell(key)
  run_data = pd.read_csv(file_path, index_col=0)

  names = ['RS', 'FS']
  rlist = run_data[[f'r_{{{name}}}' for name in names]].to_numpy().transpose()
  vallist = run_data[[f'ShSt_{{{name}}}' for name in names]].to_numpy().transpose()

  if not ax_in:
    plt.figure()
    ax = plt.gca()
  else:
    ax = ax_in
  for r, ShSt, name in zip(rlist, vallist, names):
    col = cols_dic[name]
    ax.loglog(r[r>0.]*c_/env.R0, ShSt[r>0.], c=col, label=f'{name}')
  ax.axhline(env.lfac34-1., c='r', ls='--')
  ax.axhline(env.lfac21-1., c='b', ls='--')
  ax.set_ylabel("$\\Gamma_{ud}-1$")
  ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
  ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
  ax.set_ylim(ymin=.95e-2)
  #ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
  #ax.yaxis.set_minor_formatter(ticker.ScalarFormatter())

  if not ax_in:
    plt.xlabel("$r/R_0$")
    plt.legend()
    plt.tight_layout()

def plot_relLF(key, ax_in=None):
  '''
  Plots shock Lorentz factors (in lab frame)
  '''

  env = MyEnv(key)
  file_path = get_fpath_thinshell(key)
  run_data = pd.read_csv(file_path, index_col=0)

  names = ['RS', 'FS']
  rlist = run_data[[f'r_{{{name}}}' for name in names]].to_numpy().transpose()
  vallist = run_data[[f'ShSt_{{{name}}}' for name in names]].to_numpy().transpose()

  if not ax_in:
    plt.figure()
    ax = plt.gca()
  else:
    ax = ax_in
  for r, ShSt, name in zip(rlist, vallist, names):
    col = cols_dic[name]
    ax.loglog(r[r>0.]*c_/env.R0, 1.+ShSt[r>0.], c=col, label=f'{name}')
  ax.axhline(env.lfac34, c='r', ls='--')
  ax.axhline(env.lfac21, c='b', ls='--')
  ax.set_ylabel("$\\Gamma_{ud}$")
  ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
  ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
  #ax.set_ylim(ymin=1.2)
  ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
  ax.yaxis.set_minor_formatter(ticker.ScalarFormatter())

  if not ax_in:
    plt.xlabel("$r/R_0$")
    plt.legend()
    plt.tight_layout()


### calculate radiation
def analysis_rad_thinshell(key, func='Band'):
  '''
  Calculates observed nuFnu and writes to a file
  '''
  nuobs, Tobs, env = get_radEnv(key)
  rads = nuFnu_thinshell_run(key, nuobs, Tobs, env, func)
  rads = np.concatenate((rads[0], rads[1]), axis=1)
  nulist = [f'nuobs_{i}_'+sub for sub in ['RS', 'FS'] for i in range(len(nuobs))]
  data = pd.DataFrame(rads, columns=nulist)
  data.insert(0, "Tobs", Tobs)
  # create csv file
  file_path = get_rpath_thinshell(key)
  data.to_csv(file_path)

def nuFnu_thinshell_run(key, nu_in, T_in, env, func, hybrid=True, Nnu=200):
  '''
  Returns nuFnu for 2 fronts at chosen obs freq and time
  '''
  nuobs, Tobs, env = get_radEnv(key, Nnu)
  rads = open_raddata(key, func)
  nuRS = [s for s in rads.keys() if 'RS' in s]
  nuFS = [s for s in rads.keys() if 'FS' in s]
  RS_rad = rads[nuRS].to_numpy()
  FS_rad = rads[nuFS].to_numpy()
  if len(nu_in) < len(nuobs):
    i_nu = np.array([find_closest(nuobs, nu) for nu in nu_in])
    # if len(i_nu) == 1:
    #   i_nu = i_nu[0]
    RS_rad = RS_rad[:, i_nu]
    FS_rad = FS_rad[:, i_nu]
  if len(T_in) < len(Tobs):
    i_T = np.array([find_closest(Tobs, T) for T in T_in])
    # if len(i_T) == 1:
    #   i_T = i_T[0]
    RS_rad = RS_rad[i_T]
    FS_rad = FS_rad[i_T]
  nuFnus = np.array([RS_rad, FS_rad])
  return nuFnus

def nuFnu_thinshell_run_v1(key, nuobs, Tobs, env, func, hybrid=True):
  '''
  Calculates nuFnu for both shock fronts of whole run
  func is the spectral shape in comoving frame ('Band' or 'plaw')
  '''
  nuFnus = np.zeros((2, len(Tobs), len(nuobs)))
  file_path = get_fpath_thinshell(key)
  run_data = pd.read_csv(file_path, index_col=0)
  its = run_data.index.to_list()
  iIn = 3

  sh_names = ['RS', 'FS']
  for i, sh in enumerate(sh_names):
    ids = run_data[f'ish_{{{sh}}}'].to_numpy()
    its_split = np.split(its, np.flatnonzero(np.diff(ids))+1)
    df_dataIn = run_data.loc[its_split[iIn][0]]
    #nu, Lnu = df_dataIn[[f'nu_m_{{{sh}}}', f'Lth_{{{sh}}}']]
    for j, it_group in enumerate(its_split[1:]): 
      # only calculate contribution from first it where cell is shocked
      it = it_group[0]
      #print(f'Contribution from {sh} at it {it}')
      df_data = run_data.loc[it].copy()
      #if j<iIn:
      #  df_data[[f'nu_m_{{{sh}}}', f'Lth_{{{sh}}}']] = nu, Lnu # artificially set values at start to 2 it after
      nuFnus[i] += nuobs * df_get_Fnu_thinshell(df_data, sh, nuobs, Tobs, env, func, hybrid)
  
  return nuFnus

def flux_contribs_run(key, varlist=['nu_m', 'Lth'], itmin=0, itmax=None):
  '''
  Returns only the values used as contribution
  '''
  out = []
  file_path = get_fpath_thinshell(key)
  run_data = pd.read_csv(file_path, index_col=0)
  its = run_data.index.to_list()
  sh_names = ['RS', 'FS']
  for i, sh in enumerate(sh_names):
    ids = run_data[f'ish_{{{sh}}}'].to_numpy()
    ids_split = np.split(ids, np.flatnonzero(np.diff(ids))+1)
    its_split = np.split(its, np.flatnonzero(np.diff(ids))+1)
    out_sh = np.zeros((len(its_split),len(varlist)+2))
    for j, it_group, ids_group in zip(range(len(its_split)), its_split, ids_split):
      # only calculate contribution from first it where cell is shocked
      it = it_group[0]
      ish = int(ids_group[0])
      df_data = run_data.loc[it].copy()
      var_str = [f'{var}_{{{sh}}}' for var in varlist]
      data = df_data[var_str].to_numpy()
      out_sh[j] += np.insert(data, 0, [it, ish])
    out.append(out_sh.transpose())
  return out

def df_get_Fnu_thinshell(df_data, sh, nuobs, Tobs, env, func, hybrid=True):
  '''
  Calculates Fnu for chosen shock front sh ('RS' or 'FS')
  func is the spectral shape in comoving frame ('Band' or 'plaw')
  '''
  a, d = 1, -1
  #a, d = 1.52, -1.4
  Fnu = np.zeros((len(Tobs), len(nuobs)))
  b1, b2 = -0.5, -env.psyn/2.
  if func == 'Band':
    def S(x):
      return Band_func(x, b1=b1, b2=b2)
  elif func == 'plaw':
    def S(x):
      return broken_plaw_simple(x, b1=b1, b2=b2)
  else:
    print("func must be 'Band' of 'plaw'!")
    return nuFnus
  
  var_sh = ['Tej', 'Tth', 'r', 'nu_m', 'Lth']
  varlist = [f'{var}_{{{sh}}}' for var in var_sh]
  Tej, Tth, r, nu_m, L = df_data[varlist].to_list()
  if r == 0.:
    return Fnu
  if hybrid and env.geometry == 'cartesian':
    nu_m *= (r*c_/env.R0)**d
    L *= (r*c_/env.R0)**a

  nub = (nuobs/nu_m)
  tT = ((Tobs - Tej)/Tth)
  Fnu[tT>=1.] = env.zdl * derive_Lnu(nub, tT[tT>=1.], L, S)
  return Fnu

def derive_Lnu(nub, tT, L, S):
  if (type(tT)==np.ndarray) and (type(nub)==np.ndarray):
    out = np.zeros((len(tT), len(nub)))
    nub = nub[np.newaxis, :]
    tT  = tT[:, np.newaxis]
  elif (type(tT)==np.ndarray):
    out = np.zeros(tT.shape)
  elif (type(nub)==np.ndarray):
    out = np.zeros(nub.shape)
  out += L * tT**-2 * S(nub*tT)
  return out
  

### get datas from run
def run_get_Gammash(key, smoothing=True, kernel_size=10):
  '''
  Derives Lorentz factor of the shock fronts
  Derive the (assumed constant) velocity during crossing of each cell
  '''

  file_path = get_fpath_thinshell(key)
  run_data = pd.read_csv(file_path, index_col=0)
  t = run_data['time'].to_numpy()
  dt  = run_data['dt'].to_numpy()
  lfac_sh = np.zeros((2, len(t)))
  sh_names = ['RS', 'FS']
  for i, sh in enumerate(sh_names):
    ish = run_data[f'ish_{{{sh}}}'].to_numpy()
    r   = run_data[f'r_{{{sh}}}'].to_numpy()
    dr  = run_data[f'dr_{{{sh}}}'].to_numpy()
    v   = run_data[f'v_{{{sh}}}'].to_numpy()
    # group radii and dt by same shocked cell id

    tlist, dtlist, rlist, drlist, vlist, lfaclist = \
      [np.split(arr, np.flatnonzero(np.diff(ish))+1) for arr in [t, dt, r, dr, v, np.zeros(len(t))]]
    Npos = len(tlist)
    int_list = np.zeros((2, Npos))
    for j in range(Npos-1):
      rm, drm, vm, tm, dtm = rlist[j][-1], drlist[j][-1], vlist[j][-1], tlist[j][-1], dtlist[j][-1]
      rp, drp, vp, tp, dtp = rlist[j+1][0], drlist[j+1][0], vlist[j+1][0], tlist[j+1][0], dtlist[j+1][0]
      if sh == 'FS':
        ri_m = rm + 0.5*(drm + vm*dtm)
        ri_p = rp - 0.5*(drp + vp*dtp)
      else:
        ri_m = rm - 0.5*drm + 0.5*vm*dtm
        ri_p = rp + 0.5*drp - 0.5*vp*dtp
      ri = 0.5*(ri_m + ri_p)
      ti = 0.5*(tm + tp)
      int_list[0,j] = ri
      int_list[1,j] = ti
    int_v = np.gradient(int_list[0], int_list[1])
    int_lfac = derive_Lorentz(int_v)

    # smooth data
    if smoothing:
      kernel = np.ones(kernel_size) / kernel_size
      int_lfac = np.convolve(int_lfac, kernel, mode='same')

    # resize 
    lfaclist = [int_lfac[k]*np.ones(len(tlist[k])) for k in range(len(tlist))]
    lfac_sh[i] += np.concatenate(lfaclist).ravel()

  return lfac_sh

def analysis_hydro_thinshell(key, itmin=0, itmax=None):
  '''
  Extract all datas from all files
  scaling and post-processing will be done separately
  '''

  # create varlist
  name_int = ['RS', 'CD', 'FS']
  name_sh = ['RS', 'FS']
  var_int = ['r', 'v', 'lfac', 'p']
  var_sh  = ['id', 'ish', 'dr', 'ShSt', 'Tej', 'Tth', 'nu_m', 'Lth']
  varlist_rho = ['rho_{RS}', 'rho_{CD-}', 'rho_{CD+}', 'rho_{FS}']
  varlist_int = [f'{s1}_{{{s2}}}' for s1 in var_int for s2 in name_int]
  varlist_sh  = [f'{s1}_{{{s2}}}' for s1 in var_sh for s2 in name_sh]
  varlist = varlist_int + varlist_rho + varlist_sh
  varlist.insert(0, 'time')
  data = pd.DataFrame(columns=varlist)

  # analyze datafiles
  its = dataList(key, itmin, itmax)[0:]
  if itmin == 0:
    its = its[1:]
  Nf = len(its)
  istop = -1
  bothCrossed = False
  for it in its:
    print(f'Opening file it {it}')
    df, t, tsim = openData_withtime(key, it)
    RScr, FScr = df_check_crossed(df)
    bothCrossed = (RScr and FScr)
    if bothCrossed:
      istop = its.index(it)
      break
    tup = df_get_all_thinshell_v1(df)
    results = [item for arr in tup for item in arr]
    results.insert(0, tsim)
    dict = {var:res for var, res in zip(varlist, results)}
    data.loc[it] = pd.Series(dict)
  
  # add delta t
  times = data['time'].to_numpy()
  dts = [(times[j+1]-times[j-1])/2. for j in range(1,len(times)-1)]
  dts.insert(0, (3*times[0]-times[1])/2)
  dts.append((3*times[-1]-times[-2])/2)
  data.insert(1, "dt", dts)
  data.index = its[:istop] if istop>0 else its

  # create csv file
  file_path = get_fpath_thinshell(key)
  data.to_csv(file_path)

def get_fpath_thinshell(key):
  dir_path = GAMMA_dir + f'/results/{key}/'
  file_path = dir_path + 'thinshell_hydro.csv'
  return file_path

def get_rpath_thinshell(key):
  dir_path = GAMMA_dir + f'/results/{key}/'
  file_path = dir_path + 'thinshell_rad.csv'
  return file_path

def df_get_all_thinshell(df):
  '''
  Returns all quantities needed for the thin shell paper
  '''
  frontvars = ['i', 'x', 'dx', 'vx']
  downvars = ['i', 'rho', 'D', 'p']
  RSf, FSf = df_get_shFronts(df)
  RSd, FSd = df_get_shDown(df)
  #RSd, FSd = df_get_cellsBehindShock(df, n=5)
  for cell in [RSd, FSd]:
    cell['gmin'] = get_variable(cell, 'gma_m')

  RScr, FScr = df_check_crossed(df)

  iRS, rRS, drRS, vRS = RSf[frontvars] if not RScr else (0., 0., 0., 0.)
  iFS, rFS, drFS, vFS = FSf[frontvars] if not FScr else (0., 0., 0., 0.)
  iRSd, rhoRS, DRS, pRS = RSd[downvars] if not RScr else (0., 0., 0., 0.)
  iFSd, rhoFS, DFS, pFS = FSd[downvars] if not FScr else (0., 0., 0., 0.)
  lfacRS = DRS/rhoRS if rhoRS else 0.
  lfacFS = DFS/rhoFS if rhoFS else 0.
  TonRS, TthRS, TejRS = (0., 0., 0.) if RSf.empty else get_variable(RSf, 'obsT')
  TonFS, TthFS, TejFS = (0., 0., 0.) if FSf.empty else get_variable(FSf, 'obsT')
  rCD, rho2, rho3, vCD, pCD = df_get_CDvals(df)

  radList  = [rRS, rCD, rFS]
  idDList  = [iRSd, iFSd]
  rhoList  = [rhoRS, rho3, rho2, rhoFS]
  vList    = [vRS, vCD, vFS]
  lfaclist = [lfacRS, derive_Lorentz(vCD), lfacFS]
  prsList  = [pRS, pCD, pFS]
  idShList = [iRS, iFS]
  drList   = [drRS, drFS]
  ShStList = get_shocksStrength(df)
  TejList  = [TejRS, TejFS]
  TthList  = [TthRS, TthFS]
  numList  = [0. if cell.empty else get_variable(cell, 'nu_m') for cell in [RSd, FSd]]
  LthList  = [0. if cell.empty else get_variable(cell, 'Lth') for cell in [RSd, FSd]]

  return radList, vList, lfaclist, prsList, rhoList, idDList, \
    idShList, drList, ShStList, TejList, TthList, numList, LthList


def df_get_all_thinshell_v1(df, theory=False):
  '''
  Returns all quantities needed for the thin shell paper
  '''

  RS, FS = df_get_cellsBehindShock(df)
  rCD, rho2, rho3, vCD, pCD = df_get_CDvals(df)
  iRS, rRS, drRS, rhoRS, vRS, pRS = df_get_valsAtShocks(RS)
  iFS, rFS, drFS, rhoFS, vFS, pFS = df_get_valsAtShocks(FS)
  TonRS, TthRS, TejRS = (0., 0., 0.) if RS.empty else get_variable(RS, 'obsT')
  TonFS, TthFS, TejFS = (0., 0., 0.) if FS.empty else get_variable(FS, 'obsT')

  radList  = [rRS, rCD, rFS]
  rhoList  = [rhoRS, rho3, rho2, rhoFS]
  vList    = [vRS, vCD, vFS]
  lfacList = [derive_Lorentz(v) for v in vList]
  prsList  = [pRS, pCD, pFS]
  idShList = [iRS, iFS]
  drList   = [drRS, drFS]
  ShStList = get_shocksStrength(df)
  TejList  = [TejRS, TejFS]
  TthList  = [TthRS, TthFS]
  numList  = [0. if cell.empty else get_variable(cell, 'nu_m') for cell in [RS, FS]]
  LthList  = [0. if cell.empty else get_variable(cell, 'Lth') for cell in [RS, FS]]

  return radList, vList, lfacList, prsList, rhoList, idShList, \
    idShList, drList, ShStList, TejList, TthList, numList, LthList

