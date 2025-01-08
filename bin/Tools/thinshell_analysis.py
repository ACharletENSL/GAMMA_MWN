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
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
plt.ion()

cols_dic = {'RS':'r', 'FS':'b', 'RS+FS':'k', 'CD':'purple',
  'CD-':'mediumvioletred', 'CD+':'mediumpurple', '3':'darkred', '2':'darkblue'}
spsh_cols = {'Band':'g', 'plaw':'teal'}
kernel_size = 10
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

def run_fullAnalyzis(key, smoothing=False):
  analysis_hydro_thinshell(key)
  extract_contribs(key, smoothing)
  write_pks(key, force=True)

def get_Rf_th(key, m, sh='RS'):
  '''
  Derives theoretical crossing radius assuming power-law for shock front 
  '''
  env = MyEnv(key)

def get_crossradii(key):
  '''
  Get crossing radii
  '''
  path = get_contribspath(key)
  for i, sh in enumerate(['RS', 'FS']):
    df = pd.read_csv(path+sh+'.csv')

def get_pksnunFnu(key, func='Band', force=False, detailed=True):
  '''
  Returns nupk and nupkFnupk
  '''
  path = get_contribspath(key) + 'pks.csv'
  f_exists = os.path.exists(path)
  if f_exists and not force:
    df = pd.read_csv(path)
    print('Reading file ' + path)
    nuRS, nuFS, nuFnuRS, nuFnuFS = df.to_numpy().transpose()
    nupks = np.array([nuRS, nuFS])
    nuFnupks = np.array([nuFnuRS, nuFnuFS])

  else:
    print('Writing file ' + path)
    forpks = detailed
    nuobs, Tobs, env = get_radEnv(key, forpks=True)
    nuFnus = nuFnu_thinshell_run(key, nuobs, Tobs, env, func, printing=True)
    NT = nuFnus.shape[1]
    nupks, nuFnupks = np.zeros((2,2,NT))
    for k, nuFnu in enumerate(nuFnus):
      for i in range(NT):
        ipk = np.argmax(nuFnu[i])
        nupks[k][i] = nuobs[ipk]
        nuFnupks[k][i] = nuFnu[i,ipk]
    dic = {'nu_RS':nupks[0], 'nu_FS':nupks[1], 'nuFnu_RS':nuFnupks[0], 'nuFnu_FS':nuFnupks[1]}
    df = pd.DataFrame.from_dict(dic)
    df.to_csv(path, index=False)

  return nupks, nuFnupks

def write_pks(key, func='Band', force=True, detailed=True):
  nupks, nuFnupks = get_pksnunFnu(key, func, force, detailed)
  path = get_contribspath(key) + 'pks'
  dic = {'nu_RS':nupks[0], 'nu_FS':nupks[1], 'nuFnu_RS':nuFnupks[0], 'nuFnu_FS':nuFnupks[1]}
  df = pd.DataFrame.from_dict(dic)
  df.to_csv(path+'.csv', index=False)

### plots


def plot_nuRatios_compared(keylist = ['sph_fid', 'cart_fid'], xscale='dR'):
  '''
  Compare two Delta R/ R0 plots
  '''
  fig, ax = plt.subplots()
  lstyles = ['-', '-.']
  dummy_lst = [ax.plot([],  [], c='k', ls=l)[0] for l in lstyles]
  ax.legend(dummy_lst, ['sph.', 'hybrid'])
  for key, l in zip(keylist, lstyles):
    plot_nubrkRatio(key, ax, xscale=xscale, ls=l)
  xlabel = '$\\Delta R/R_0$'
  if xscale == 'ton':
    xlabel = '$t_\\mathrm{on}/t_\\mathrm{off}$'
  ax.set_xlabel(xlabel)
  ax.set_ylabel('$\\nu_\\mathrm{bk}/\\nu_{1/2}$')
  ax.set_xscale('log')
  ax.set_yscale('log')
  ax.grid(True, which='both')
  plt.tight_layout()


def plot_nubrkRatio(key, ax_in=None, ratio=.5, dRRmin=1e-2, xscale='dR', **kwargs):
  '''
  Plot ratio between frequency at break nu_bk and at nuFnu(nu_bk)/2 as a function of Delta R/R0
  '''

  env = MyEnv(key)
  dRR, t_ons, nubks, nuhfs = get_nubk_nuhf(key, ratio)
  x = dRR
  xlabel = '$\\Delta R/R_0$'
  if xscale == 'ton':
    x = t_ons
    xlabel = '$t_\\mathrm{on}/t_\\mathrm{off}$'
  i_start = find_closest(dRR, dRRmin)
  nu_ratio =  nubks[i_start:]/nuhfs[i_start:]

  if not ax_in:
    plt.figure()
    ax = plt.gca()
  else:
    ax = ax_in
  
  ax.plot(x[i_start:-1], nu_ratio[:-1], c='r', **kwargs)

  if not ax_in: 
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$\\nu_{bk}/\\nu_{1/2}$')



def plot_nubrks_compared(keylist = ['sph_fid', 'cart_fid']):
  '''
  Compare two Delta R/ R0 plots
  '''
  fig, ax = plt.subplots()
  lstyles = ['-', '-.']
  dummy_lst = [ax.plot([],  [], c='k', ls=l)[0] for l in lstyles]
  ax.legend(dummy_lst, ['sph.', 'hybrid'])
  for key, l in zip(keylist, lstyles):
    plot_nubrk(key, ax, ls=l)
  ax.set_xlabel('$\\Delta R/R_0$')
  ax.set_ylabel('$\\nu_{bk}/\\nu_0$')
  ax.set_ylim(ymin = 8e-2)


def plot_nubknhf(key, ax_in=None, ratio=0.5, dRRmin=1e-2, xscale='dR', **kwargs):
  '''
  Plot frequency at break and freq at half flux 
  Can be plotted vs Delta R/R0 or ton/toff
  '''
  env = MyEnv(key)
  dRR, t_ons, nubks, nuhfs = get_nubk_nuhf(key, ratio) 
  i_start = find_closest(dRR, dRRmin)
  x = dRR
  xlabel = '$\\Delta R/R_0$'
  if xscale == 'ton':
    x = t_ons
    xlabel = '$t_\\mathrm{on}/t_\\mathrm{off}$'

  if not ax_in:
    plt.figure()
    ax = plt.gca()
  else:
    ax = ax_in
  
  colors = ['r', 'darkorange']
  dummy_col = [plt.plot([], [], c=c)[0] for c in colors]
  for val, c in zip([nubks, nuhfs], colors):
    ax.loglog(x[i_start:-1], val[i_start:-1], c=c, **kwargs)
    
  if not ax_in: 
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$\\nu/\\nu_0$')
    ax.legend(dummy_col, ['$\\nu_\\mathrm{bk}$', '$\\nu_{1/2}$'])

def plot_comovContribs(key, xscale='t'):
  '''
  plots Gamma, nu'_m, L'
  vs it, t, T, or r
  '''
  sh_names = ['RS', 'FS']
  var_in = ['i_d', 'lfac', 'nu_m', 'Lth']
  
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

def plot_compareContribs(keylist, varlist=['Ton', 'nu_m', 'Lth'], xscaling='r'):
  '''
  Same as plot_contribs but comparing runs
  '''
  

  sh_names = ['RS', 'FS']
  lstyles = ['-', '-.', ':']
  xlabel = ''
  if xscaling == 'it':
    xlabel = 'it'
  elif xscaling == 'r':
    xlabel = '$r/R_0$'
  elif xscaling == 't':
    xlabel = '$t/t_0$'
  elif xscaling == 'T':
    xlabel = '$\\bar{T}$'

  fig, axs = plt.subplots(len(varlist)+1,1,sharex='all')
  dummy_ls  = [plt.plot([],[], c='k', ls=l)[0] for l in lstyles[:len(keylist)]]
  dummy_col = [plt.plot([],[], c=cols_dic[name], ls='-')[0] for name in sh_names]


  for key, ls in zip(keylist, lstyles):
    contribs = flux_contribs_run(key, varlist)
    env = MyEnv(key)

    for sh, cont in zip(sh_names, contribs):
      it, t, r = cont[0:3]
      #vals = cont[3:]
      if xscaling == 'it':
        x = it
      elif xscaling == 'r':
        x = r*c_/env.R0
      elif xscaling == 't':
        x = t/env.t0 + 1.
      elif xscaling == 'T':
        T = (1+env.z)*(t + env.t0 - cont[2])
        x = (T-T[0])/env.T0
      
      for ax, var, name in zip(axs, cont[3:], ['i_sh']+varlist):
        label = var_exp[name]
        scale = 'log'
        if name == 'i_sh':
          label = '$i_\\mathrm{sh}-i_\\mathrm{CD}$'
          var -= env.Next+env.Nsh4
          scale = 'linear'
        elif name == 'nu_m':
          label = '$\\nu_m/\\nu_0$'
          var /= env.nu0
        elif name.startswith('T'):
          label = '$\\bar{T}_\\mathrm{' + name[1:] + '}$'
          var = (var - env.Ts)/env.T0
          scale = 'linear'
        ax.set_ylabel(label)
        ax.plot(x[1:], var[1:], c=cols_dic[sh], ls=ls)
        ax.set_yscale(scale)
        ax.scatter(x[1:], var[1:], s=4., c='k', marker='x')
  axs[0].set_yscale('linear')
  axs[0].legend(dummy_col, sh_names)
  axs[-1].legend(dummy_ls, keylist)
  axs[-1].set_xlabel(xlabel)

def plot_contribs(key, varlist=['Ton', 'nu_m', 'Lth'], xscaling='it'):
  sh_names = ['RS', 'FS']
  contribs = flux_contribs_run(key, varlist)
  env = MyEnv(key)
  # careful, contribs is a list of 2 arrays with different lengths
  # order: it, time, r, i_sh, vars
  
  xlabel = ''
  if xscaling == 'it':
    xlabel = 'it'
  elif xscaling == 'r':
    xlabel = '$r/R_0$'
  elif xscaling == 't':
    xlabel = '$t/t_0$'
  elif xscaling == 'T':
    xlabel = '$\\bar{T}$'

  fig, axs = plt.subplots(len(varlist)+1,1,sharex='all')
  axs[0].set_title(key)
  dummy_col = [plt.plot([],[], c=cols_dic[name], ls='-')[0] for name in sh_names]

  for sh, cont in zip(sh_names, contribs):
    it, t, r = cont[0:3]
    #vals = cont[3:]
    if xscaling == 'it':
      x = it
    elif xscaling == 'r':
      x = r*c_/env.R0
    elif xscaling == 't':
      x = t/env.t0 + 1.
    elif xscaling == 'T':
      T = (1+env.z)*(t + env.t0 - cont[2])
      x = (T-T[0])/env.T0
    
    for ax, var, name in zip(axs, cont[3:], ['i_sh']+varlist):
      label = var_exp[name]
      scale = 'log'
      if name == 'i_sh':
        label = '$i_\\mathrm{sh}-i_\\mathrm{CD}$'
        var -= env.Next+env.Nsh4
        scale = 'linear'
      elif name == 'nu_m':
        label = '$\\nu_m/\\nu_0$'
        var /= env.nu0
      elif name.startswith('T'):
        label = '$\\bar{T}_\\mathrm{' + name[1:] + '}$'
        var = (var - env.Ts)/env.T0
        scale = 'linear'
      ax.set_ylabel(label)
      ax.plot(x[1:], var[1:], c=cols_dic[sh])
      ax.set_yscale(scale)
      ax.scatter(x[1:], var[1:], s=4., c='k', marker='x')
  axs[0].set_yscale('linear')
  fig.legend(dummy_col, sh_names)

  axs[-1].set_xlabel(xlabel)

def plot_hydroPanels(key, smoothing=True, art_mode=True):
  '''
  Plots hydro variable in a column panel
  '''
  fig = plt.figure(figsize=(7,9), layout='constrained')
  gs = fig.add_gridspec(4, 1, hspace=0.05, wspace=0)
  axs = gs.subplots(sharex='all')
  #fig, axs = plt.subplots(4, 1, figsize=(6,12), layout='constrained', sharex='all')
  axs[-1].set_xlabel('$r/R_0$')
  plotfuncs = [plot_rhop, plot_lfac, plot_prs, plot_ShSt]


  for plotfunc, ax in zip(plotfuncs, axs):
    plotfunc(key, ax_in=ax, smoothing=smoothing)
  #fig.set_figheight(6)
  # names = ['RS', '3', 'CD-', 'CD', 'CD+', '2', 'FS']
  # legnames = ['RS', '3', 'CD$_3$', 'CD', 'CD$_2$', '2', 'FS']
  # dummy_col = [plt.plot([],[], c=cols_dic[name], ls='-')[0] for name in names]
  # fig.legend(dummy_col, legnames, handlelength=1.5, loc='outside center right')
  #fig.set_figsize((9,7))
  if art_mode:
    rhonames = ['3', 'CD-', 'CD+', '2']
    dummy_col = [plt.plot([],[], c=cols_dic[name], ls='-')[0] for name in rhonames]
    axs[0].legend(dummy_col, ['d,3', 'CD,3', 'CD,2', 'd,2'], loc='lower center', ncol=len(dummy_col))
    shnames = ['RS', 'FS']
    dummy_col = [plt.plot([],[], c=cols_dic[name], ls='-')[0] for name in shnames]
    axs[3].legend(dummy_col, shnames, loc='center', ncol=len(dummy_col))
    axs[0].set_ylim(ymin=0.4)
    axs[2].set_ylim(ymax=1.1)
    axs[3].set_ylim(ymin=1.9e-2)
    axs[0].set_xlim(xmin=1.)

def plot_compareHydro(keylist):
  lstyles = ['-', '-.', ':']
  fig, axs = plt.subplots(2, 1, layout='constrained', sharex='all')
  axs[-1].set_xlabel('$r/R_0$')
  plotfuncs = [plot_lfac, plot_ShSt]
  for key, ls in zip(keylist, lstyles):
    for plotfunc, ax in zip(plotfuncs, axs):
      plotfunc(key, ax_in=ax, ls=ls)


def plot_hydroPanels_reduced(key, smoothing=False):
  '''
  Plots hydro variable in a column panel
  '''
  fig, axs = plt.subplots(2, 1, layout='constrained', sharex='all')
  #fig.set_figheight(8)
  axs[-1].set_xlabel('$r/R_0$')
  plotfuncs = [plot_lfac, plot_ShSt]

  for plotfunc, ax in zip(plotfuncs, axs):
    plotfunc(key, ax_in=ax, smoothing=smoothing)
  fig.set_figheight(6)
  # names = ['RS', '3', 'CD', '2', 'FS']
  # dummy_col = [plt.plot([],[], c=cols_dic[name], ls='-')[0] for name in names]
  # fig.legend(dummy_col, names, handlelength=1.5, loc='upper center', ncol=len(dummy_col))
  #plt.tight_layout()

def plot_radPanels_t(keylist=['sph_fid', 'cart_fid'], func='Band', logT=False):
  '''
  Same as radPanel_t but to compare two or more runs
  '''
  fig = plt.figure()
  plotfuncs = [plot_nupk, plot_nupkFnupk]
  crTb = 1.
  gs = fig.add_gridspec(len(plotfuncs), 1, wspace=0)
  axs = gs.subplots(sharex='all')
  names, colors = ['RS', 'FS'], ['r', 'b']
  lstyles = ['-', '-.']
  runnames = ['sph', 'hybrid'] #if keylist == ['sph_fid', 'cart_fid'] else keylist
  dummy_col = [plt.plot([],[], c=col, ls='-')[0] for name, col in zip(names, colors)]
  dummy_lst = [plt.plot([],[], ls=ls, color='k')[0] for ls in lstyles]
  ylabels = ['$\\nu_\\mathrm{pk}/\\nu_0$', "$(\\nu F_\\nu)_\\mathrm{pk}/\\nu_0F_0$"]

  for label, plotfunc, ax in zip(ylabels, plotfuncs, axs):
    ax.set_ylabel(label)
    for key, ls in zip(keylist, lstyles):
      #theory = True if key == 'cart_fid' else False
      plotfunc(key, func=func, ax_in=ax, logT=logT, theory=False, ls=ls)
  #   ax.set_xscale('linear')
  # axs[-1].set_yscale('linear')
  xmin = 1 if logT else 0
  xlabel = "$\\tilde{T}$" if logT else "$\\bar{T}$"
  axs[-1].set_xlim((xmin,5))
  axs[-1].set_xlabel(xlabel)
  axs[0].set_ylim(ymin=5e-3)
  axs[0].legend(dummy_col, names, handlelength=1.5)
  axs[1].legend(dummy_lst, runnames)
  axs[0].grid(True, which='both')
  axs[1].grid(True, which='both')
  plt.tight_layout()


def plot_radPanel_t(key, func='Band', theory=False, logT=False, smallWin=False):
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
    plotfunc(key, func=func, ax_in=ax, logT=logT, theory=theory)
    ax.set_xscale('log') if logT else ax.set_xscale('linear')

  axs[-1].set_yscale('linear')
  if smallWin:
    axs[-1].set_xlim((0,5))
    axs[0].set_ylim(ymin=5e-3)
  fig.legend(dummy_col, names, handlelength=1.5)
  axs[0].set_title(key)


def plot_radPanels(key, comov=True, xscale='r'):
  '''
  Plots hydro variable in a column panel
  '''
  plotfuncs = [plot_num, plot_Lnum]
  fig, axs = plt.subplots(len(plotfuncs), 1, figsize=(6,12), layout='constrained', sharex='all')
  #fig.set_figheight(8)
  xlabel = ''
  if xscale == 'r':
    xlabel = '$r/R_0$'
  elif xscale == 't':
    xlabel = '$t/t_0$'
  elif xscale == 'T':
    xlabel = '$\\bar{T}$'
  ylabels = ["$\\nu'_m/\\nu'_0$", "$L'_{\\nu_m}/L'_0$"] if comov else ["$\\nu_m/\\nu_0$", "$L_{\\nu_m}/L_0$"]

  for plotfunc, ax, ylabel in zip(plotfuncs, axs, ylabels):
    plotfunc(key, ax_in=ax, comov=comov, xscale=xscale)
    ax.set_ylabel(ylabel)
  axs[-1].set_xlabel(xlabel)
  fig.set_figheight(6)
  names = ['RS', 'FS']
  dummy_col = [plt.plot([],[], c=cols_dic[name], ls='-')[0] for name in names]
  axs[0].legend(dummy_col, names, handlelength=1.5, loc='lower right')
  #plt.tight_layout()

def plot_compareAll(var, lognu=-1, logT=0, theory=False,
  pres_mode=False, art_mode=False, HPC=False):
  '''
  Plots 2x1 of the selected var comparing between runs
  'lc' lightcurves - 'sp' (instantaneous) spectra - 
  'tsp' time-integrated spectra - 'bhv' spectral behavior
  pres_mode: presentation mode (only RS for bhv, colored spec shape)
  art_mode: article mode (adds hybrid Band onto syn-BPL panel)
  '''
  runs = ['sph_fid', 'cart_fid']
  if HPC:
    runs = ['HPC/'+run for run in runs]
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
      plot_lightcurve(key, lognu, func, theory, totonly=False, ax_in=ax_in, **kwargs)
    def plot_hybtotline(ax_in):
      plot_lightcurve(runs[1], lognu, 'Band', theory=False, totonly=True, ax_in=ax_in, ls=lstyles[-1])
  elif var == 'sp':
    title = 'spectras at $\\bar{T}=$'+f'{10**logT:.1f}'
    def plotfunc(key, func, ax_in, **kwargs):
      plot_spectrum(key, logT, func, theory, totonly=False, ax_in=ax_in, **kwargs)
    def plot_hybtotline(ax_in):
      plot_spectrum(runs[1], logT, 'Band', theory=False, totonly=True, ax_in=ax_in, ls=lstyles[-1])
  elif var == 'tsp':
    title = 'time-integrated spectras'
    ylabel = '$\\nu f_\\nu/\\nu_0 F_0 T_0$'
    def plotfunc(key, func, ax_in, **kwargs):
      plot_integSpec(key, func, theory=theory, ax_in=ax_in, **kwargs)
    def plot_hybtotline(ax_in):
      plot_integSpec(runs[1], 'Band', theory=False, totonly=True, ax_in=ax_in, ls=lstyles[-1])
  elif var == 'bhv':
    title = 'spectral behavior'
    def plotfunc(key, func, ax_in, **kwargs):
      plot_radBehav(key, func, False, ax_in,
          pres_mode=pres_mode, **kwargs)
  
  fig.supylabel(ylabel)
  axs[-1].set_xlabel(xlabel)
  axs[0].set_title(title)
  dummy_lst = [plt.plot([],[], ls=ls, color='k')[0] for ls in ['-', '-.', ':']]
  if not pres_mode:
    loc = 'upper right'
    if var == 'bhv':
      loc = 'upper left'
    elif ('sp' in var):
      loc = 'lower center'
    dummy_col = [plt.plot([],[], c=cols_dic[name], ls='-')[0] for name in names]
    axs[0].legend(dummy_col, names, loc=loc)


  lstlegend = axs[-1].legend(dummy_lst, ['sph.', 'hybrid', 'hyb. Band'])

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

  


def plot_comparedRad(key, lognu=-1, logT=0, theory=True):
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

  plot_lightcurve(key, lognu, 'Band', theory=theory, ax_in=ax_lcb)
  plot_lightcurve(key, lognu, 'plaw', theory=theory, ax_in=ax_lcp)
  plot_spectrum(key, logT, 'Band', theory=theory, ax_in=ax_spb)
  plot_spectrum(key, logT, 'plaw', theory=theory, ax_in=ax_spp)

def plot_spectrum(key, logT, func='Band',
  theory=False, hybrid=True, totonly=False, newmethod=True,
  ax_in=None, **kwargs):
  '''
  Plots instantaneous spectrum at \bar{T} = 10**logT
  '''
  nuobs, Tobs, env = get_radEnv(key)
  Tb = (Tobs-env.Ts)/env.T0
  iT = find_closest(Tb, 10**logT)
  T = np.array([Tobs[iT]])
  nub = nuobs/env.nu0
  nuFnus = nuFnu_thinshell_run(key, nuobs, T, env, func)[:,0,:] #if newmethod \
    #else nuFnu_thinshell_run_v1(key, nuobs, T, env, func)[:,0,:]
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

  # nupks, Fnupks = get_pksnunFnu(key)
  # cols = ['r', 'b']
  # for nupk, Fnupk, col in zip(nupks, Fnupks, cols):
  #   ax.scatter(nupk[iT]/env.nu0, Fnupk[iT]/env.nu0F0, c=col)

  ax.grid(visible=True, which='both')
  if not ax_in:
    plt.legend()
    plt.xlabel("$\\nu/\\nu_0$")
    plt.ylabel("$\\nu F_{\\nu}/\\nu_0 F_0$")
    funcname = 'syn-BPL' if func == 'plaw' else func
    plt.title(key+', $\\bar{T}=$'+f'{10**logT:.1f}'+', '+funcname)

def plot_lightcurve(key, lognu, func='Band',
  theory=False, hybrid=True, totonly=False, newmethod=True,
  ax_in=None, **kwargs):
  '''
  Plots lightcurve at nu = 10**lognu nu_O
  '''
  nuobs, Tobs, env = get_radEnv(key)
  nuobs = np.array([10**lognu*env.nu0])
  Tb = (Tobs-env.Ts)/env.T0
  nuFnus = nuFnu_thinshell_run(key, nuobs, Tobs, env, func)[:,:,0] #if newmethod \
    #else nuFnu_thinshell_run_v1(key, nuobs, Tobs, env, func)[:,:,0]
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
  ax.grid(visible=True, which='both')
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
  theory=False, hybrid=True, totonly=False, ax_in=None, **kwargs):
  '''
  Plots time integrated spectrum
  '''
  nuobs, Tobs, env = get_radEnv(key)
  nub = nuobs/env.nu0
  nuFnus = nuFnu_thinshell_run(key, nuobs, Tobs, env, func)
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

def plot_bhvCompared(runs=['sph_fid', 'cart_fid'], func='Band'):
  fig, ax = plt.subplots()
  for run, ls in zip(runs, ['-', '-.']):
    plot_radBehav(run, func=func, theory=False, ax_in=ax, pres_mode=True, ls=ls)
  dummy_lst = [plt.plot([],[], ls=ls, color='k')[0] for ls in ['-', '-.']]
  ax.legend(dummy_lst, ['spherical', 'hybrid'])
  plt.xlabel("$\\nu_\\mathrm{pk}/\\nu_0$")
  plt.ylabel("$(\\nu F_\\nu)_\\mathrm{pk}/\\nu_0F_0$")
  #plt.title('spectral behavior')
  ax.grid(visible=True, which='both')

def plot_radBehav(key, func='Band', theory=False, ax_in=None, 
    pres_mode=False, **kwargs):
  '''
  Plots nuFnu_pk(nu_pk) of the observed flux
  '''

  if not ax_in:
    plt.figure()
    ax = plt.gca()
  else:
    ax = ax_in
  
  ax.grid(True, which='both')
  names, colors = ['RS', 'FS'], ['r', 'b']
  nupks, nuFnupks = get_pksnunFnu(key)
  nuobs, Tobs, env = get_radEnv(key)
  #nuFnus = nuFnu_thinshell_run(key, nuobs, Tobs, env, func)
  Nit = 1 if pres_mode else nuFnupks.shape[0]
  NT = nuFnupks.shape[1]

  if theory:
    for k, name in zip(range(Nit), names):
      if env.geometry == 'spherical':
        dRoR = 1.52 if name == 'RS' else 1.67
      else:
        dRoR = None
      plot_bhv_hybrid(dRoR=dRoR, sh=name, ax_in=ax, ls=':')
    if not ax_in:
      dummy_lst = [plt.plot([],[], ls=ls, color='k')[0] for ls in ['-', ':']]
      loc = 'upper center'
      lstlegend = ax.legend(dummy_lst, ['numerical', 'analytical'], loc=loc) #bbox_to_anchor=(0.98, 0.7)
      ax.add_artist(lstlegend)

  
  for k, nupk, nuFnupk, name, col in zip(range(Nit), nupks, nuFnupks, names, colors):
    ax.loglog(nupk[1:]/env.nu0, nuFnupk[1:]/env.nu0F0, c=col, label=name, **kwargs)
  
  ax.grid(visible=True, which='both')
  # xmin = 5e-3
  # if not pres_mode:
  #   ax.set_xlim(xmin=xmin)
  # ax.set_ylim(ymin=1e-2)
  if pres_mode:
    ax.set_xlim(xmin=4e-2)

  if not ax_in:
    if not pres_mode:
      plt.legend()
    if pres_mode:
      plt.title(env.geometry)
    else:
      plt.title(key)
    plt.xlabel("$\\nu_{pk}/\\nu_0$")
    plt.ylabel("$\\nu_{pk}F_{\\nu_{pk}}/\\nu_0F_0$")

def plot_bhv_hybrid(dRoR=None, sh='RS', ax_in=None, **kwargs):
  '''
  Plots spectral behavior of chosen front, from fits on hybrid approach
  '''
  if not ax_in:
    plt.figure()
    ax = plt.gca()
  else:
    ax = ax_in

  nuobs, Tobs, env = get_radEnv('cart_fid')
  g, Tth, dRR, fac_nu, fac_F, col = (env.gFS, env.T0FS, env.dRFS, env.fac_nu, env.fac_F, 'b') if sh == 'FS' else (env.gRS, env.T0, env.dRRS, 1., 1., 'r')
  tT = 1. + (Tobs - env.Ts)/Tth
  dR = dRoR if dRoR else dRR/env.R0
  tTf = 1. + dR
  nu, nuFnu = approx_bhv_hybrid(tT, tTf, g)
  nu /= fac_nu
  nuFnu /= (fac_F*fac_nu)
  ax.loglog(nu, nuFnu, c=col, **kwargs)
  ax.grid(visible=True, which='both')

  if not ax_in:
    plt.xlabel(f"$\\nu/\\nu_{{{'0,'+sh}}}$")
    plt.ylabel("$\\nu F_{\\nu}/\\nu_0 F_0$")
    plt.tight_layout()

def plot_nupk(key, func='Band', ax_in=None, logT=False, slope=False, theory=False, **kwargs):
  '''
  Plots nu_pk (\bar{T}) of the observed flux
  '''

  if not ax_in:
    plt.figure()
    ax = plt.gca()
  else:
    ax = ax_in
  
  names, colors = ['RS', 'FS'], ['r', 'b']
  nupks, nuFnupks = get_pksnunFnu(key)
  nuobs, Tobs, env = get_radEnv(key, forpks=True) 
  Tb = (Tobs - env.Ts)/env.T0
  x = Tb
  xlabel = "$\\bar{T}$"
  if logT:
    x += 1
    xlabel = "\\tilde{T}"
  xscale = 'log' if logT else 'linear'



  for nupk, name, col in zip(nupks, names, colors):
    y = nupk/env.nu0
    if slope:
      logx = np.log10(x)
      #smty = savgol_filter(y, 9, 3)
      intp = interp1d(x, y, kind='linear')
      logy = np.log10(y)
      grad = np.gradient(logy, logx)
      y = grad

    ax.semilogy(x[1:], y[1:], c=col, label=name, **kwargs)
    ax.set_xscale(xscale)
    if theory:
      g, Tth, dRR, fac_nu, fac_F = (env.gFS, env.T0FS, env.dRFS, env.fac_nu, env.fac_F) \
        if name == 'FS' else (env.gRS, env.T0, env.dRRS, 1., 1.)
      tT = 1. + (Tobs - env.Ts)/Tth
      tTf = 1. + dRR/env.R0
      if env.geometry == 'cartesian':
        nu, nuFnu = approx_bhv_hybrid(tT, tTf, g)
        nu /= fac_nu
        ax.semilogy(Tb, nu, c=col, ls=':', **kwargs)
      else:
        def fitfunc(x, d1, d2):
          return approx_nupk_hybrid(x, tTfi=tTf, gi=g, d1=d1, d2=d2)
        curve_fit(fitfunc, tT[2:], nupks[2:])

  if not ax_in:
    if theory:
      dummy_lst = [plt.plot([],[], c='k', ls=l)[0] for l in ['-', ':']]
      lstlegend = ax.legend(dummy_lst, ['numerical', 'analytical'], loc='lower right')
      ax.add_artist(lstlegend)
    plt.legend()
    plt.title(key)
    plt.xlabel(xlabel)
    plt.ylabel("$\\nu_{pk}/\\nu_0$")


def plot_nupkFnupk(key, func='Band', ax_in=None, logT=False, logF=False, theory=False, **kwargs):
  '''
  Plots nu_pk F_{nu_pk} (\bar{T}) of the observed flux
  '''

  if not ax_in:
    plt.figure()
    ax = plt.gca()
  else:
    ax = ax_in
  
  names, colors = ['RS', 'FS'], ['r', 'b']
  nupks, nuFnupks = get_pksnunFnu(key) 
  nuobs, Tobs, env = get_radEnv(key, forpks=True)
  Tb = (Tobs-env.Ts)/env.T0
  x = Tb
  xlabel = "$\\bar{T}$"
  if logT:
    x += 1
    xlabel = "\\tilde{T}"
  xscale = 'log' if logT else 'linear'
  yscale = 'log' if logF else 'linear'

  for nuFnupk, name, col in zip(nuFnupks, names, colors):
    ax.plot(x[1:], nuFnupk[1:]/env.nu0F0, c=col, label=name, **kwargs)
    if theory:
      g, Tth, dRR, fac_nu, fac_F, col = (env.gFS, env.T0FS, env.dRFS, env.fac_nu, env.fac_F, 'b') \
        if name == 'FS' else (env.gRS, env.T0, env.dRRS, 1., 1., 'r')
      tT = 1. + (Tobs - env.Ts)/Tth
      tTf = 1. + dRR/env.R0
      nu, nuFnu = approx_bhv_hybrid(tT, tTf, g)
      nuFnu /= (fac_F*fac_nu)
      ax.plot(x, nuFnu, c=col, ls=':', **kwargs)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)

  if not ax_in:
    if theory:
      dummy_lst = [plt.plot([],[], c='k', ls=l)[0] for l in ['-', ':']]
      lstlegend = ax.legend(dummy_lst, ['numerical', 'analytical'], loc='lower right')
      ax.add_artist(lstlegend)
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
  t = run_data['time'].to_numpy()/env.t0
  xlabel = ""
  if xscale == 't':
    xlabel = "$t/t_0$"
  elif xscale == 'r': 
    xlabel = "$r/R_0$"
  elif xscale == 'T':
    xlabel = "$\\bar{T}$"

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
    if xscale == 'r':
      x = r*c_/env.R0
      ax.loglog(x[r>0.], nu[r>0.]*L[r>0.]/nu0L0, label=f'{name}', c=col)
    else:
      if xscale == 'T':
        T = (1+env.z)*(t*env.t0 - r)
        x = (T-env.Ts)/env.T0
      ax.semilogy(x[r>0.], nu[r>0.]*L[r>0.]/nu0L0, label=f'{name}')
    ax.axhline(th/nu0L0, c=col, ls='--', lw=.8)
  ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
  ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
  
  if not ax_in:
    plt.xlabel(xlabel)
    plt.ylabel("$\\nu_mL_{\\nu_m}/\\nu_0 L_0$")
    plt.legend()


def plot_num(key, ax_in=None, xscale='r', comov=True, rscaling=False):
  '''
  Plots peak observed frequency with radius
  '''

  env = MyEnv(key)
  file_path = get_fpath_thinshell(key)
  run_data = pd.read_csv(file_path, index_col=0)

  names = ['RS', 'FS']
  rlist = run_data[[f'r_{{{name}}}' for name in names]].to_numpy().transpose()
  vallist = run_data[[f'nu_m_{{{name}}}' for name in names]].to_numpy().transpose()
  if comov:
    lfaclist = run_data[[f'lfac_{{{name}}}' for name in names]].to_numpy().transpose()
    vallist /= 2*lfaclist/(1+env.z)
  t = run_data['time'].to_numpy()/env.t0 + 1
  x = t
  xlabel = ''
  if xscale == 'r':
    xlabel = '$r/R_0$'
  elif xscale == 't':
    xlabel = '$t/t_0$'
  elif xscale == 'T':
    xlabel = '$\\bar{T}$'
  ylabel, scale = ("$\\nu'_m/\\nu'_0$", env.nu0p) if comov else ("$\\nu_m/\\nu_0$", env.nu0)
  if rscaling:
    ylabel = "$r\\nu'_m/R_0\\nu'_0$" if comov else "$r\\nu_m/R_0\\nu_0$"
  thvals = [env.nu0p, env.nu0pFS] if comov else [env.nu0, env.nu0FS]

  if not ax_in:
    plt.figure()
    ax = plt.gca()
  else:
    ax = ax_in
  for r, val, name, th in zip(rlist, vallist, names, thvals):
    if rscaling:
      val *= r*c_/env.R0
    col = cols_dic[name]
    if xscale == 'r':
      x = r*c_/env.R0
      ax.loglog(x[r>0.], val[r>0.]/scale, c=col, label=f'{name}')
    else:
      if xscale == 'T':
        T = (1+env.z)*(t*env.t0 - r)
        x = (T-env.Ts)/env.T0
      ax.semilogy(x[r>0.], val[r>0.]/scale, c=col, label=f'{name}')
    ax.axhline(th/scale, c=col, ls='--', lw=.8)
  ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
  ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
  
  if not ax_in:
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()

def plot_Lnum(key, ax_in=None, xscale='r', comov=True):
  '''
  Plots peak observed luminosity
  '''

  env = MyEnv(key)
  file_path = get_fpath_thinshell(key)
  run_data = pd.read_csv(file_path, index_col=0)
  t = run_data['time'].to_numpy()/env.t0 + 1
  x = t
  xlabel = ''
  if xscale == 'r':
    xlabel = '$r/R_0$'
  elif xscale == 't':
    xlabel = '$t/t_0$'
  elif xscale == 'T':
    xlabel = '$\\bar{T}$'
  ylabel = "$L'_{\\nu'_m}/L'_0$" if comov else "$L_{\\nu_m}/L_0$"
  #scale = env.tL0/(2*env.lfac) if comov else env.tL0
  scale = env.L0p if comov else env.L0

  names = ['RS', 'FS']
  rlist = run_data[[f'r_{{{name}}}' for name in names]].to_numpy().transpose()
  vallist = run_data[[f'Lth_{{{name}}}' for name in names]].to_numpy().transpose()
  if comov:
    lfaclist = run_data[[f'lfac_{{{name}}}' for name in names]].to_numpy().transpose()
    vallist /= 2*lfaclist/(1+env.z)

  if not ax_in:
    plt.figure()
    ax = plt.gca()
  else:
    ax = ax_in
  for r, val, name in zip(rlist, vallist, names):
    col = cols_dic[name]
    if xscale == 'r':
      x = r*c_/env.R0
      ax.loglog(x[r>0.], val[r>0.]/scale, c=col, label=f'{name}')
    else:
      if xscale == 'T':
        T = (1+env.z)*(t*env.t0 - r)
        x = (T-env.Ts)/env.T0
      ax.semilogy(x[r>0.], val[r>0.]/scale, c=col, label=f'{name}')
    #ax.axhline(th/env.L0, c=col, ls='--', lw=.8)
  ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
  ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
  
  if not ax_in:
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()

def plot_Lp(key, ax_in=None, xscale='r'):
  '''
  Plots peak observed luminosity
  '''

  env = MyEnv(key)
  file_path = get_fpath_thinshell(key)
  run_data = pd.read_csv(file_path, index_col=0)
  t = run_data['time'].to_numpy()/env.t0 + 1
  x = t
  xlabel = ''
  if xscale == 'r':
    xlabel = '$r/R_0$'
  elif xscale == 't':
    xlabel = '$t/t_0$'
  elif xscale == 'T':
    xlabel = '$\\bar{T}$'
  ylabel = "$L'_{\\nu'_m}/L'_0$"
  #scale = env.tL0/(2*env.lfac) if comov else env.tL0
  scale = env.L0p 

  names = ['RS', 'FS']
  rlist = run_data[[f'r_{{{name}}}' for name in names]].to_numpy().transpose()
  vallist = run_data[[f'Lp_{{{name}}}' for name in names]].to_numpy().transpose()

  if not ax_in:
    plt.figure()
    ax = plt.gca()
  else:
    ax = ax_in
  for r, val, name in zip(rlist, vallist, names):
    col = cols_dic[name]
    if xscale == 'r':
      x = r*c_/env.R0
      ax.loglog(x[r>0.], val[r>0.]/scale, c=col, label=f'{name}')
    else:
      if xscale == 'T':
        T = (1+env.z)*(t*env.t0 - r)
        x = (T-env.Ts)/env.T0
      ax.semilogy(x[r>0.], val[r>0.]/scale, c=col, label=f'{name}')
    #ax.axhline(th/env.L0, c=col, ls='--', lw=.8)
  ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
  ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
  
  if not ax_in:
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()

def plot_nm(key):
  '''
  Combine plot_ShSt and plot_lfac, with fits
  '''
  fig, axs = plt.subplots(2, 1, sharex='all')
  plot_ShSt(key, ax_in=axs[0], fit=True)
  plot_lfac(key, ax_in=axs[1], fit=True)
  axs[0].set_title(key)
  axs[-1].set_xlabel('$r/R_0$')

def plot_rhop(key, rscale=2., ax_in=None, xscale='r', smoothing=True, N=kernel_size):
  '''
  Plots comoving densities downstream of shocks and on both sides of the CD
  '''
  env = MyEnv(key)
  file_path = get_fpath_thinshell(key)
  run_data = pd.read_csv(file_path, index_col=0)
  # if env.geometry == 'cartesian':
  #   rscale = 0.

  name_int = ['RS', 'CD', 'CD', 'FS']
  t = run_data['time'].to_numpy() + env.t0
  rlist = run_data[[f'r_{{{name}}}' for name in name_int]].to_numpy().transpose()
  name_int = ['RS', 'CD-', 'CD+', 'FS']
  rholist = run_data[[f'rho_{{{name}}}' for name in name_int]].to_numpy().transpose()
  name_cols = ['3', 'CD-', 'CD+', '2']
  x = t
  xlabel = ''
  if xscale == 'r':
    xlabel = '$r/R_0$'
  elif xscale == 't':
    xlabel = '$t/t_0$'
  elif xscale == 'T':
    xlabel = '$\\bar{T}$'

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
      rho_sc = smoothing_rollave(rho_sc, N)
    if xscale == 't':
      x = t/env.t0
    elif xscale == 'r':
      x = r*c_/env.R0
    elif xscale == 'T':
      T = (1+env.z)*(t - r)
      x = (T-env.Ts)/env.T0
    ax.loglog(x[r>0], rho_sc*r_sc**rscale, label=name, c=col)
  if xscale == 'T':
    ax.set_xscale('linear')
  #ax.set_ylim(ymin=.75)
  ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
  ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
  ax.axhline(1., c='r', ls='--', lw=.8)
  ax.axhline(env.rho2/env.rho3, c='b', ls='--', lw=.8)
  rscale_str = f"$(r/R_0)^{{{rscale:{'.0f' if rscale==int(rscale) else '.1f'}}}}$" if rscale else ""
  ax.set_ylabel("$\\tilde{\\rho'}$" + rscale_str)
  ax.grid(visible=True, which='both')

  if not ax_in:
    plt.xlabel(xlabel)
    plt.legend()
    plt.tight_layout()

def plot_gfac(key, ax_in=None, xscale='r'):
  '''
  Plots the g = Gamma/Gamma_sh factor for each shock
  '''
  env = MyEnv(key)
  file_path = get_fpath_thinshell(key)
  run_data = pd.read_csv(file_path, index_col=0)

  name_int = ['RS', 'FS']
  t = run_data['time'].to_numpy() + env.t0
  rlist = run_data[[f'r_{{{name}}}' for name in name_int]].to_numpy().transpose()
  lfaclist = run_data[[f'lfac_{{{name}}}' for name in name_int]].to_numpy().transpose()
  gmaRS, gmaFS = run_get_Gammash(key)
  vallist = lfaclist/np.array([gmaRS, gmaFS])
  x = t
  xlabel = ''
  if xscale == 'r':
    xlabel = '$r/R_0$'
  elif xscale == 't':
    xlabel = '$t/t_0$'
  elif xscale == 'T':
    xlabel = '$\\bar{T}$'

  if not ax_in:
    plt.figure()
    ax = plt.gca()
  else:
    ax = ax_in
  for r, g, name in zip(rlist, vallist, name_int):
    col = cols_dic[name]
    if xscale == 't':
      x = t/env.t0
    elif xscale == 'r':
      x = r*c_/env.R0
    elif xscale == 'T':
      T = (1+env.z)*(t - r)
      x = (T-env.Ts)/env.T0
    ax.loglog(x[r>0], g[r>0.], label=f'$g_{{{name}}}$', c=col)
  ax.axhline(env.gRS, c='r', ls='--', lw=.8)
  ax.axhline(env.gFS, c='b', ls='--', lw=.8)
  ax.set_ylim(ymax=1.24)
  ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
  ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
  ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
  ax.yaxis.set_minor_formatter(ticker.ScalarFormatter())
  ax.yaxis.set_minor_locator(ticker.FixedLocator([0.9, 1., 1.1, 1.2])) 
  ax.grid(visible=True, which='both')

  if not ax_in:
    plt.xlabel(xlabel)
    plt.legend()
    plt.tight_layout()

def plot_lfac(key, ax_in=None, smoothing=True, N=kernel_size, fit=False, ilim=30, **kwargs):
  '''
  Plots relevant Lorentz factors (in lab frame):
  Gamma_4, _d,RS, _CD, _d,FS, _1
  '''
 # last data points (the time the shock crosses entirely) are dodgy
  env = MyEnv(key)
  file_path = get_fpath_thinshell(key)
  run_data = pd.read_csv(file_path, index_col=0)
  if not ax_in:
    plt.figure()
    ax = plt.gca()
  else:
    ax = ax_in
  name_int = ['RS', 'CD', 'FS']
  rlist = run_data[[f'r_{{{name}}}' for name in name_int]].to_numpy().transpose()
  gma3, gmaCD, gma2 = run_data[[f'lfac_{{{name}}}' for name in name_int]].to_numpy().transpose()
  gmaRS, gmaFS = run_data[[f'lfacsh_{{{name}}}' for name in ['RS', 'FS']]].to_numpy().transpose()

  if fit:
    planvals = [env.lfacRS, env.lfacFS, env.lfac, env.lfac, env.lfac]
    sphvals = [120.4, 133.6, 136.3, 128.3, 124.4]
    #for rad, gma, Xplan, Xsph, name in zip(rlist, vallist, planvals, sphvals, name_int):
    for rad, gma, Xplan, name in zip(rlist, vallist, planvals, name_int):
      r = rad[rad>0]*c_/env.R0
      lfac = gma[rad>0]
      col = cols_dic[name]
      i_s = max(find_closest(r, 1.05), kernel_size)
      i_smp = max(find_closest(r, 1.1), kernel_size+30)
      m = logslope(r[i_s], lfac[i_s], r[i_smp], lfac[i_smp])
      def fitfunc(r, Xsph, s):
        if m>0:
          return ((Xplan * r**m)**-s + Xsph**-s)**(-1/s)
        elif m<0:
          return ((Xplan * r**m)**s + Xsph**s)**(1/s)
        else:
          return Xsph * np.ones(r.shape)
      sigma = np.ones(r.shape)
      sigma[i_s] = 0.01
      popt, pcov = curve_fit(fitfunc, r, lfac, bounds=([100, 1], [200, np.inf]))
      Xsph, s = popt
      fitted = fitfunc(r, *popt)
      ax.plot(r, fitted, c='k', ls=':')
      print(f'{name}: gma_sph = {Xsph:.1f}, m={-2*m:.2f}, s={s:.2f}')

  if smoothing:
    for arr in [gmaRS, gmaFS, gma3, gma2]:
      arr[:] = gaussian_filter1d(arr, sigma=2, order=0)
      arr[:] = smoothing_rollave(arr, N)
      # arr[:] = smoothing_rollave(arr, N)

  name_int = ['RS', 'FS', '3', 'CD', '2']
  rlist = np.stack((rlist[0], rlist[-1], *rlist))
  vallist = np.stack((gmaRS, gmaFS, gma3, gmaCD, gma2))

  for r, lfac, name in zip(rlist, vallist, name_int):
    col = cols_dic[name]
    lfac = lfac[r>0.]
    r = r[r>0.]*c_/env.R0
    ax.loglog(r[:-ilim], lfac[:-ilim], c=col, label=f'$\\Gamma_{{{name}}}$', **kwargs)

  ax.hlines([env.lfacRS, env.lfacFS], 0, 0.5, transform=ax.get_yaxis_transform(), colors=['r', 'b'], ls='--')
  ax.hlines([120.3, 133.6], 0.5, 1, transform=ax.get_yaxis_transform(), colors=['r', 'b'], ls=':')


  ax.set_ylabel(f"$\\Gamma$")
  ax.set_ylim((105., 145.))
  ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
  ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
  ax.yaxis.set_minor_formatter(ticker.ScalarFormatter())
  ax.yaxis.set_minor_locator(ticker.FixedLocator([110, 120, 130, 140])) 
  ax.grid(visible=True, which='both')
  
  # names = ['RS', '3', 'CD', '2', 'FS']
  # dummy_col = [plt.plot([],[], c=cols_dic[name], ls='-')[0] for name in names]
  # ax.legend(dummy_col, names, handlelength=1.5, loc='lower center',
  #   ncol=len(dummy_col), framealpha=1.)

  if not ax_in:
    plt.xlabel("$r/R_0$")
    #plt.legend()
    plt.tight_layout()

def plot_prs(key, rscale=2., ax_in=None, smoothing=True, N=kernel_size):
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
    for p in vallist:
      p[:] = smoothing_rollave(p, N)

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
  ax.grid(visible=True, which='both')

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
  lfaclist = run_data[[f'lfac_{{{name}}}' for name in names]].to_numpy().transpose()
  lfacsh = run_get_Gammash(key)

  if not ax_in:
    plt.figure()
    ax = plt.gca()
  else:
    ax = ax_in
  for r, lfac1, lfac2, name, valth in zip(rlist, lfacsh, lfaclist, names, [env.lfacRS, env.lfacFS]):
    col = cols_dic[name]
    ax.loglog(r[r>0.]*c_/env.R0, lfac1[r>0.], c=col, label=f'$\\Gamma_{{{name}}}$')
    ax.loglog(r[r>0.]*c_/env.R0, lfac2[r>0.], c='k', ls=':')
    ax.axhline(valth, color=col, ls='--', lw=.6)
    ax.text(1.02, valth, f"$\\Gamma_{{{name+',th'}}}$", color='k', transform=ax.get_yaxis_transform(),
            ha='left', va='center')

  ax.set_ylabel(f"$\\Gamma$")
  ax.grid(visible=True, which='both')

  if not ax_in:
    plt.xlabel("$r/R_0$")
    plt.legend()
    plt.tight_layout()

def plot_ShSt(key, ax_in=None, xscale='r', fit=False, smoothing=True, N=kernel_size, **kwargs):
  '''
  Plots shock Lorentz factors (in lab frame)
  '''

  env = MyEnv(key)
  file_path = get_fpath_thinshell(key)
  run_data = pd.read_csv(file_path, index_col=0)

  if fit: xscale = 'r'
  names = ['RS', 'FS']
  t = run_data['time'].to_numpy() + env.t0
  rlist = run_data[[f'r_{{{name}}}' for name in names]].to_numpy().transpose()
  vallist = run_data[[f'ShSt_{{{name}}}' for name in names]].to_numpy().transpose()

  xlabel = ''
  if xscale == 'r':
    xlabel = '$r/R_0$'
  elif xscale == 't':
    xlabel = '$t/t_0$'
  elif xscale == 'T':
    xlabel = '$\\bar{T}$'

  if not ax_in:
    plt.figure()
    ax = plt.gca()
  else:
    ax = ax_in

  planvals = [env.lfac34 - 1, env.lfac21 - 1]
  sphvals = [7.35e-2, 2.36e-2]
  for rad, ShSt, name, Xplan in zip(rlist, vallist, names, planvals):
    col = cols_dic[name]
    r = rad[rad>0.]*c_/env.R0
    x = r
    ShSt = ShSt[rad>0.]

    if fit:
      i_smp=20
      n0 = logslope(r[0], ShSt[0], r[i_smp], ShSt[i_smp])
      nl = min(n0*0.8, n0*1.2)
      nu = max(n0*0.8, n0*1.2)
      def fitfunc(r, Xpl, Xsph, n, s):
        if n>0:
          return ((Xpl * r**n)**-s + Xsph**-s)**(-1/s)
        elif n<0:
          return ((Xpl * r**n)**s + Xsph**s)**(1/s)
        else:
          return Xsph * np.ones(r.shape)
      popt, pcov = curve_fit(fitfunc, r, ShSt, p0=[Xplan, 0.8*Xplan, n0, 2],
        bounds=([0.8*Xplan, 0, nl, 1], [1.2*Xplan, Xplan, nu, np.inf]))
      Xpl, Xsph, n, s = popt
      fitted = fitfunc(r, *popt)
      ax.plot(r, fitted, c='k', ls=':')
      print(f'{name}: starts at {Xpl:.2e} ({100*(Xpl-Xplan)/Xplan:.2f}% diff with theory), converges at {Xsph:.2e}, n={-n:.2f}, s={s:.2f}')

    if smoothing:
      ShSt = smoothing_rollave(ShSt, N)
    if xscale == 't':
      x = t/env.t0
    elif xscale == 'T':
      T = (1+env.z)*(t - rad)
      x = (T-env.Ts)/env.T0
    ax.loglog(x[r>0], ShSt, c=col, label=f'{name}', **kwargs)

  ax.hlines(planvals, 0, 0.5, transform=ax.get_yaxis_transform(), colors=['r', 'b'], ls='--')
  ax.hlines(sphvals, 0.5, 1, transform=ax.get_yaxis_transform(), colors=['r', 'b'], ls=':')


  if xscale=='T':
    ax.set_xscale('linear')

  ax.set_ylabel("$\\Gamma_{ud}-1$")
  ax.grid(visible=True, which='both')
  ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
  ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
  #ax.set_ylim(ymin=.95e-2)
  #ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
  #ax.yaxis.set_minor_formatter(ticker.ScalarFormatter())

  if not ax_in:
    plt.xlabel(xlabel)
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
  ax.grid(visible=True, which='both')

  if not ax_in:
    plt.xlabel("$r/R_0$")
    plt.legend()
    plt.tight_layout()


### calculate radiation
# def analysis_rad_thinshell(key, func='Band'):
#   '''
#   Calculates observed nuFnu and writes to a file
#   '''
#   nuobs, Tobs, env = get_radEnv(key)
#   rads = nuFnu_thinshell_run(key, nuobs, Tobs, env, func)
#   rads = np.concatenate((rads[0], rads[1]), axis=1)
#   nulist = [f'nuobs_{i}_'+sub for sub in ['RS', 'FS'] for i in range(len(nuobs))]
#   data = pd.DataFrame(rads, columns=nulist)
#   data.insert(0, "Tobs", Tobs)
#   # create csv file
#   file_path = get_rpath_thinshell(key)
#   data.to_csv(file_path)

# def nuFnu_thinshell_run_v0(key, nu_in, T_in, env, func, hybrid=True):
#   '''
#   Returns nuFnu for 2 fronts at chosen obs freq and time
#   '''
#   nuobs, Tobs, env = get_radEnv(key)
#   rads = open_raddata(key, func)
#   nuRS = [s for s in rads.keys() if 'RS' in s]
#   nuFS = [s for s in rads.keys() if 'FS' in s]
#   RS_rad = rads[nuRS].to_numpy()
#   FS_rad = rads[nuFS].to_numpy()
#   if len(nu_in) < len(nuobs):
#     i_nu = np.array([find_closest(nuobs, nu) for nu in nu_in])
#     # if len(i_nu) == 1:
#     #   i_nu = i_nu[0]
#     RS_rad = RS_rad[:, i_nu]
#     FS_rad = FS_rad[:, i_nu]
#   if len(T_in) < len(Tobs):
#     i_T = np.array([find_closest(Tobs, T) for T in T_in])
#     # if len(i_T) == 1:
#     #   i_T = i_T[0]
#     RS_rad = RS_rad[i_T]
#     FS_rad = FS_rad[i_T]
#   nuFnus = np.array([RS_rad, FS_rad])
#   return nuFnus

# def nuFnu_thinshell_run_v1(key, nuobs, Tobs, env, func, hybrid=True):
#   '''
#   Calculates nuFnu for both shock fronts of whole run
#   func is the spectral shape in comoving frame ('Band' or 'plaw')
#   '''
#   nuFnus = np.zeros((2, len(Tobs), len(nuobs)))
#   file_path = get_fpath_thinshell(key)
#   run_data = pd.read_csv(file_path, index_col=0)
#   its = run_data.index.to_list()
#   iIn = 3

#   sh_names = ['RS', 'FS']
#   for i, sh in enumerate(sh_names):
#     ids = run_data[f'ish_{{{sh}}}'].to_numpy()
#     its_split = np.split(its, np.flatnonzero(np.diff(ids))+1)
#     #df_dataIn = run_data.loc[its_split[iIn][0]]
#     #nu, Lnu = df_dataIn[[f'nu_m_{{{sh}}}', f'Lth_{{{sh}}}']]
#     for j, it_group in enumerate(its_split[1:]): 
#       # only calculate contribution from first it where cell is shocked
#       it = it_group[0]
#       #print(f'Contribution from {sh} at it {it}')
#       df_data = run_data.loc[it].copy()
#       #if j<iIn:
#       #  df_data[[f'nu_m_{{{sh}}}', f'Lth_{{{sh}}}']] = nu, Lnu # artificially set values at start to 2 it after
#       nuFnus[i] += nuobs * df_get_Fnu_thinshell(df_data, sh, nuobs, Tobs, env, func, hybrid)
  
#   return nuFnus

def get_nubk_nuhf(key, ratio=0.5, func='Band', testing=False):

  nuobs, Tobs, env = get_radEnv(key, forpks=True)
  path = get_contribspath(key)
  df = pd.read_csv(path+'RS.csv')
  nupks, nuFnupks = get_pksnunFnu(key, func) # we're looking only at RS
  nupks = nupks[0]/env.nu0
  nuFnupks = nuFnupks[0]/env.nu0F0
  if testing:
    fig, ax = plt.subplots()
    ax.loglog(nupks, nuFnupks)
    cmap = mpl.colormaps['rainbow']
    Nl = 5
    colors = cmap(np.linspace(0, 1, Nl+1))
    k = 0
    dummy_col = []
    dummy_leg = []

  Nsh = len(df.index)
  dRR, tons, nubks, nuhfs = np.zeros((4, Nsh))
  for i in range(Nsh):
    row = df.iloc[i]
    dRR[i] = row['r_{RS}']*c_/env.R0 - 1.
    ish = int(row['ish_{RS}']) - env.Next
    dr0 = (env.Nsh4 - ish)*env.D04/env.Nsh4 # R0 - R_cell at t0
    ton = dr0/(c_*env.beta4)
    tons[i] = ton/env.toff
    # Tej, Tth = row[['Tej_{RS}', 'Tth_{RS}']]
    # Ton = Tej + Tth
    Ton = row['Ton_{RS}']
    ion = find_closest(Tobs, Ton)
    Tb = (Ton - env.Ts)/env.T0
    nubk = nupks[ion]
    nuFnubk = nuFnupks[ion]
    ihf = find_closest(nuFnupks[nupks>=nubk], ratio*nuFnubk)
    nubks[i] = nubk
    nuhf = nupks[nupks>=nubk][ihf]
    nuhfs[i] = nuhf
    if testing and (not (i % int(Nsh/Nl))) and i:
      col = colors[k]
      dummy_col.append(plt.plot([], [], c=col)[0])
      dummy_leg.append(f'$\\bar{{T}}={Tb:.1f}$')
      ax.axvline(nubk, c=col)
      ax.axvline(nuhf, c=col, ls=':')
      ax.axhline(nuFnubk, c=col)
      ax.axhline(ratio*nuFnubk, c=col, ls=':')
      k += 1
  
  if testing:
    ax.set_xlim(xmin=7e-2)
    ax.set_ylim(ymin=1e-2)
    ax.legend(dummy_col, dummy_leg)
  
  return dRR, tons, nubks, nuhfs

def get_breakfreqsRS_run(key, func='Band', ratio=0.5):
  '''
  Get the break frequency and the freq at half flux at each increment of the flux coming from the RS
  Allows to explore the range of Delta R/R_0 of the full shell
  '''
  nuobs, Tobs, env = get_radEnv(key)
  path = get_contribspath(key)
  df = pd.read_csv(path+'RS.csv')
  nu = nuobs/env.nu0
  nuFnu = np.zeros((len(Tobs), len(nuobs)))
  Nsh = len(df.index)
  ipks = []
  dRR, nubks, nuhfs, nuFnubks = np.zeros((4, Nsh))
  for i in range(Nsh):
    row = df.iloc[i]
    dRR[i] = row['r_{RS}']*c_/env.R0 - 1.
    nuFnu += (nuobs * df_get_Fnu_thinshell(row, 'RS', nuobs, Tobs, env, func))/env.nu0F0
    # Tej, Tth = row[['Tej_{RS}', 'Tth_{RS}']]
    # Ton = Tej + Tth
    Ton = row['Ton_{RS}']
    ion = find_closest(Tobs, Ton)
    ibk = np.argmax(nuFnu[ion]) 
    nu_bk = nu[ibk]
    nubks[i] = nu_bk
    nuFnu_bk = nuFnu[ion][ibk]
    nuFnubks[i] = nuFnu_bk
    ihf = find_closest(nuFnubks, ratio*nuFnu_bk)
    nuhfs[i] = nubks[ihf]

  return dRR, nubks, nuhfs

def get_freqs_brknratio(key, ratio=.5, dRRmin=0.2, func='Band', max=False):
  '''
  Returns nu_bk and nu at ratio*nuFnu(nu_bk)
  '''
  colors = ['r', 'b', 'g', 'c', 'm']
  plt.figure()
  nuobs, Tobs, env = get_radEnv(key)
  path = get_contribspath(key)
  df = pd.read_csv(path+'RS.csv')
  nu = nuobs/env.nu0
  nuFnu = np.zeros((len(Tobs), len(nuobs)))
  Nsh = len(df.index)
  dRR, nu_bks, nu_atr = np.zeros((3, Nsh))
  l = 0
  for i in range(Nsh):
    row = df.iloc[i]
    nuFnu += (nuobs * df_get_Fnu_thinshell(row, 'RS', nuobs, Tobs, env, func))/env.nu0F0
    dRR[i] = row['r_{RS}']*c_/env.R0 - 1.
    if dRR[i] >= dRRmin:
      #Tej, Tth = row[['Tej_{RS}', 'Tth_{RS}']]
      Ton = row['Ton_{RS}']
      ion = np.searchsorted(Tobs, Ton)

      # for i in range(NT):
      #   ipk = np.argmax(nuFnu[i])
      #   nupks[k][i] = nuobs[ipk]
      #   nuFnupks[k][i] = nuFnu[i,ipk]
      # construct the nupkFnupk(nupk) curve
      nupks, nuFnupks = get_pksnunFnu(key) 
      nupks = nupks[0]
      nuFnupks = nuFnupks[0]

      # peak frequency at crossing time is our "break frequency" 
      ibk = np.argmax(nuFnu[ion])
      nu_bk = nuobs[ibk]
      if max: # break and max are not the same
        ibk = np.argmax(nuFnupks)
        nu_bk = nupks[ibk]
      # find_sorted & np.searchsorted need sorted array, nuFnupks may decrease
      nFn_atbrk = nuFnupks[-1]
      i_h = find_closest(nuFnupks[nuFnupks<=nuFnupks.max()], ratio*nFn_atbrk)
      nu_h = nupks[i_h]
      
      nu_bks[i] = nu_bk
      nu_atr[i] = nu_h
      if dRR[i] % 0.2 <= 0.002:
        col = colors[l]
        plt.loglog(nupks/env.nu0, nuFnupks/env.nu0F0, label=f'{dRR[i]:.2f}', alpha=.7, c=col)
        plt.axvline(nu_bk/env.nu0, c=col)
        plt.axvline(nu_h/env.nu0, c=col, ls=':')
        plt.axhline(nFn_atbrk/env.nu0F0, c=col)
        plt.axhline(ratio*nFn_atbrk/env.nu0F0, c=col, ls=':')
        l += 1

  plt.legend()
  return dRR[dRR >= dRRmin], nu_bks[dRR >= dRRmin], nu_atr[dRR >= dRRmin]

def get_breakRatioRS_run(key, ratio=.5, dRRmin=.2, func='Band', max=False):
  '''
  Returns nu_{bk}/nu_{ratio}, ratio typically is 1/2 or 1/3 
  nu_{1/2} is the frequency at which the peak flux (rising) is at half its value at break
  '''

  dRR, nu_bks, nu_atr = get_freqs_brknratio(key, ratio=ratio, dRRmin=dRRmin, func=func, max=max)
  return dRR, nu_bks/nu_atr
  


def nuFnu_thinshell_run(key, nuobs, Tobs, env, func, hybrid=True, printing=False):
  '''
  Calculates nuFnu for both shock fronts of whole run
  func is the spectral shape in comoving frame ('Band' or 'plaw')
  '''
  nuFnus = np.zeros((2, len(Tobs), len(nuobs)))
  path = get_contribspath(key)
  for i, sh in enumerate(['RS', 'FS']):
    if printing:
      print(f'Calculating contributions from {sh}')
    df = pd.read_csv(path+sh+'.csv')
    N = len(df.index)
    for j in range(N):
      row = df.iloc[j]
      if printing and (not (j % int(N/100))):
        print(f'from it {row['it']}')
      nuFnus[i] += nuobs * df_get_Fnu_thinshell(row, sh, nuobs, Tobs, env, func, hybrid)
  return nuFnus

def flux_contribs_run(key, varlist=['nu_m', 'Lth']):
  '''
  Returns only the values used as contribution, + t, r, and downstream cell id
  out is a list of two np.array, one per front
  '''
  out = []
  var_in = ['time', 'r', 'ish'] + varlist
  env = MyEnv(key)
  path = get_contribspath(key)
  for sh in ['RS', 'FS']:
    df = pd.read_csv(path+sh+'.csv', index_col='it')
    its = df.index.to_numpy()
    vars = [f'{var}_{{{sh}}}' if var != 'time' else var for var in var_in]
    outsh = np.insert(df[vars].to_numpy().transpose(), 0, its, axis=0)
    out.append(outsh)
  return out

def flux_contribs_run_old(key, varlist=['nu_m', 'Lth'], itmin=0, itmax=None):
  '''
  Returns only the values used as contribution
  '''
  varlist[0:0] = ['time', 'r', 'ish']
  env = MyEnv(key)
  file_path = get_fpath_thinshell(key)
  run_data = pd.read_csv(file_path, index_col=0)
  its = run_data.index.to_list()
  sh_names = ['RS', 'FS']
  out = []
  for sh in sh_names:
    ids = run_data[f'ish_{{{sh}}}'].to_numpy()
    ids_split = np.split(ids, np.flatnonzero(np.diff(ids))+1)
    its_split = np.split(its, np.flatnonzero(np.diff(ids))+1)
    out_sh = np.zeros((len(its_split),len(varlist)+1))
    for j, it_group, ids_group in zip(range(len(its_split)), its_split, ids_split):
      # only calculate contribution from first it where cell is shocked
      it = it_group[0]
      ish = int(ids_group[0])
      df_data = run_data.loc[it].copy()
      var_str = [f'{var}_{{{sh}}}' if var != 'time' else var for var in varlist]
      data = df_data[var_str].to_numpy()
      data = np.concatenate((np.array([it]), data))
      out_sh[j] += data
    out_sh = out_sh.transpose()
    out.append(out_sh)
  return out

def df_get_Fnu_thinshell(df_data, sh, nuobs, Tobs, env, func, hybrid=True):
  '''
  Calculates Fnu for chosen shock front sh ('RS' or 'FS')
  func is the spectral shape in comoving frame ('Band' or 'plaw')
  '''
  a, d = 1, -1
  Fnu = np.zeros((len(Tobs), len(nuobs)))
  b1, b2 = -0.5, -env.psyn/2.
  if func == 'Band':
    def S(x, b1, b2):
      return Band_func(x, b1, b2)
  elif func == 'plaw':
    def S(x, b1, b2):
      return broken_plaw_simple(x, b1, b2)
  else:
    print("func must be 'Band' of 'plaw'!")
    return Fnu
  
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
  Fnu[tT>=1.] = env.zdl * derive_Lnu(nub, tT[tT>=1.], L, S, b1, b2)
  return Fnu

def derive_Lnu(nub, tT, L, S, b1, b2):
  if (type(tT)==np.ndarray) and (type(nub)==np.ndarray):
    out = np.zeros((len(tT), len(nub)))
    nub = nub[np.newaxis, :]
    tT  = tT[:, np.newaxis]
  elif (type(tT)==np.ndarray):
    out = np.zeros(tT.shape)
  elif (type(nub)==np.ndarray):
    out = np.zeros(nub.shape)
  out += L * tT**-2 * S(nub*tT, b1, b2)
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

# def extrapolate_early_old(df, env, nsamp=10, i_s=15):
#   '''
#   Do the extrapolation of contributions at early times before shock settled properly
#   Performs extrapolation by sampling over nsamp datapoints starting from the i_s'th
#   '''
#   sample = df.iloc[i_s:i_s+nsamp].copy()
#   dt = np.diff(sample['time'].to_numpy()).mean()
#   ids = df.index
#   sh = df.keys()[1][-3:-1]
#   v  = sample[f'v_{{{sh}}}'].mean()
#   dr = dt*v
#   a_g = np.diff(np.log10(sample[f'lfac_{{{sh}}}'].to_numpy())).mean()
#   a_nu = np.diff(np.log10(sample[f'nu_m_{{{sh}}}'].to_numpy())).mean()
#   a_L = np.diff(np.log10(sample[f'Lth_{{{sh}}}'].to_numpy())).mean()
#   df = df.drop(ids[:i_s])
#   t_i, r_i, gma_i, nu_i, L_i = df.iloc[0][['time', f'r_{{{sh}}}', f'lfac_{{{sh}}}', f'nu_m_{{{sh}}}', f'Lth_{{{sh}}}']]
#   Nit = 1+int(min(np.floor(t_i/dt), np.floor(r_i/dr)))
#   df_in =  pd.DataFrame(columns=df.keys())
#   for n in range(1, Nit):
#     i = i_s - n
#     t, r, gma, nu, L = t_i - n*dt, r_i - n*dr, gma_i*10**(-n*a_g), nu_i*10**(-n*a_nu), L_i*10**(-n*a_L)
#     Ton, Tth, Tej = derive_obsTimes(t, r, v, env.t0, env.z)
#     vals = [t, r, v, Tej, Tth, gma, nu, L]
#     dic = {key:val for key, val in zip(df.keys(), vals)}
#     row = pd.DataFrame(dic, index=[i])
#     df_in = pd.concat([row, df_in], ignore_index=True)
#   df = pd.concat([df_in, df])
#   return df

def extrapolate_early(df, env, nsamp=10, i_s=15):
  '''
  New version
  '''
  sh = df.keys()[1][-3:-1]
  varlist = ['lfac', 'p', 'rho', 'ShSt', 'nu_m', 'Lth', 'Lp']
  sample = df.iloc[i_s:i_s+nsamp].copy()
  rs = sample[f'r_{{{sh}}}'].to_numpy()

  for var in varlist:
    name = f'{var}_{{{sh}}}'
    svals = sample[name].to_numpy()
    # get plaw with r
    a = logslope(rs[0], svals[0], rs[-1], svals[-1])
    val_s = df.iloc[i_s][name]
    # replace values assuming this plaw
    for i, row in df.head(i_s).iterrows():
      dr = row[f'r_{{{sh}}}']/rs[0]
      df.loc[i,name] = val_s*dr**a
  return df

def smooth_contribs_roll(df, N=25):
  '''
  Smooth contribs with a rolling averaged window of size N
  '''

  dic = {}
  kernel = np.ones((N,))/N
  for var in df.keys():
    val = df[var].to_numpy()
    if not var.startswith(('time', 'r', 'ish', 'T')):
      arr_padded = np.pad(var, (N//2, N-1-N//2), mode='edge')
      val = np.convolve(arr_padded, kernel, mode='valid')
    dic[var] = val
  df_in = pd.DataFrame(dic)

def smooth_contribs(df, env):
  '''
  Replace freq and luminosity by a fit to clean noise
  '''
  sh = df.keys()[2][-3:-1]
  rname = f'r_{{{sh}}}'
  r = df[rname].to_numpy(copy=True)
  r *= c_/env.R0
  rf = r[-1]

  # fit lfac, nu_m and Lth to function
  lfacname = f'lfac_{{{sh}}}'
  lfac = df[lfacname].to_numpy(copy=True)
  lfac /= env.lfac
  m0 = logslope(r[0], lfac[0], r[10], lfac[10])
  def fitfunc_lfac(r, lfacpl, lfacsph, m, s):
    if m<0:
      return ((lfacpl * r**m)**s + lfacsph**s)**(1/s)
    else:
      return ((lfacpl * r**m)**-s + lfacsph**-s)**(-1/s)
  popt, pcov = curve_fit(fitfunc_lfac, r[r>0.], lfac[r>0.],
      p0=[1, 1, m0, 1])
  lfacfit = env.lfac*fitfunc_lfac(r[r>0], *popt)
  df.loc[df[rname]>0., lfacname] = lfacfit
  vfit = derive_velocity(lfacfit)
  df.loc[r>0., f'v_{{{sh}}}'] = vfit
  Tonfit, Tthfit, Tejfit = derive_obsTimes(
    df.loc[r>0.]['time'], df.loc[r>0.][rname], vfit, env.t0, env.z)  
  df.loc[r>0., f'Ton_{{{sh}}}'] = Tonfit
  df.loc[r>0., f'Tth_{{{sh}}}'] = Tthfit
  df.loc[r>0., f'Tej_{{{sh}}}'] = Tejfit

  nuname = f'nu_m_{{{sh}}}'
  nu = df[nuname].to_numpy(copy=True)
  nu0 = getattr(env, 'nu0' + f'{sh if sh=='FS' else ''}')
  nu /= nu0
  d0 = logslope(r[0], nu[0], r[10], nu[10])
  def fitfunc_rnu(r, nupl, nusph, d, s):
    return ((nupl * r**d)**s + (nusph)**s)**(1/s)
  popt, pcov = curve_fit(fitfunc_rnu, r[r>0.], r[r>0.]*nu[r>0.],
      p0=[1, 1, d0, 1])
  def nu_fitted(r):
    return fitfunc_rnu(r, *popt)/r
  
  Lname = f'Lth_{{{sh}}}'
  L = df[Lname].to_numpy(copy=True)
  tL0 = getattr(env, 'tL0' + f'{sh if sh=='FS' else ''}')
  L /= tL0
  a0 = logslope(r[0], L[0], r[10], L[10])
  def fitfunc_L(r, Lpl, Lsph, a, s):
    return ((Lpl * r**a)**-s + (Lsph*r)**-s)**(-1/s)
  popt, pcov = curve_fit(fitfunc_L, r[r>0.], L[r>0.],
      p0=[1, 1, a0, 1])
  def L_fitted(r):
    return fitfunc_L(r, *popt)

  df.loc[r>0., nuname] = nu0*nu_fitted(df.loc[r>0.][rname]*c_/env.R0)
  df.loc[r>0., Lname] = tL0*L_fitted(df.loc[r>0.][rname]*c_/env.R0)
  return df

def extract_contribs(key, extrapolate=True, smoothing=False):
  '''
  Extracts data per shock
  '''

  file_path = get_fpath_thinshell(key)
  outpath = get_contribspath(key)
  env = MyEnv(key)
  run_data = pd.read_csv(file_path, index_col=0)
  its = run_data.index.to_list()
  varlist = ['r', 'ish', 'v', 'rho', 'p', 'ShSt', 'Ton', 'Tej', 'Tth', 'lfac', 'nu_m', 'Lth', 'Lp']
  for sh in ['RS', 'FS']:
    keylist = ['time'] + [var + f'_{{{sh}}}' for var in varlist]
    ids = run_data[f'ish_{{{sh}}}'].to_numpy()
    its_split = np.split(its, np.flatnonzero(np.diff(ids))+1)
    its_contribs = []
    for it_group in (its_split[1:] if sh=='RS' else its_split): 
      its_contribs.append(it_group[0])
    df = run_data.loc[its_contribs][keylist].copy(deep=True)
    df['it']=its_contribs
    if extrapolate:
      df = extrapolate_early(df, env)
    # clean potential holes
    r = df[f'r_{{{sh}}}']
    holes = np.where(r == 0)[0]
    #print('holes at it', holes)
    for i in holes:
      if i == len(df)-1:
        df.iloc[i] = df.iloc[i-1]
      else:
        df.iloc[i] = .5*(df.iloc[i-1] + df.iloc[i+1])
    if smoothing:
      df = smooth_contribs(df, env)
    df.to_csv(outpath+sh+'.csv', index_label='it')

def analysis_hydro_thinshell(key, itmin=0, itmax=None):
  '''
  Extract all datas from all files
  scaling and post-processing will be done separately
  '''

  # create varlist
  name_int = ['RS', 'CD', 'FS']
  name_sh = ['RS', 'FS']
  var_int = ['r', 'v', 'lfac', 'p']
  var_sh  = ['id', 'ish', 'dr', 'ShSt', 'lfacsh', 'Ton', 'Tej', 'Tth', 'nu_m', 'Lth', 'Lp']
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
  skips = 0
  calc_its = []
  bothCrossed = False
  for it in its:
    df, t, tsim = openData_withtime(key, it)
    RScr, FScr = df_check_crossed(df)
    bothCrossed = (RScr and FScr)
    if bothCrossed:
      skips +=1
      if skips > 10 and len(data)>0:
        break
      continue
    print(f'Opening file it {it}')
    calc_its.append(it)
    tup = df_get_all_thinshell_v1(df)
    results = [item for arr in tup for item in arr]
    results.insert(0, tsim)
    dict = {var:res for var, res in zip(varlist, results)}
    data.loc[it] = pd.Series(dict)
    istop = its.index(it)
    skips = 0
  
  # add delta t
  if istop >=0:
    times = data['time'].to_numpy()
    dts = [(times[j+1]-times[j-1])/2. for j in range(1,len(times)-1)]
    dts.insert(0, (3*times[0]-times[1])/2)
    dts.append((3*times[-1]-times[-2])/2)
    data.insert(1, "dt", dts)
    data.index = calc_its 

  # create csv file
  file_path = get_fpath_thinshell(key)
  data.to_csv(file_path)

def get_contribspath(key):
  dir_path = GAMMA_dir + f'/results/{key}/'
  file_path = dir_path + 'thinshell_'
  return file_path

def get_fpath_thinshell(key):
  dir_path = GAMMA_dir + f'/results/{key}/'
  file_path = dir_path + 'thinshell_hydro.csv'
  return file_path

def get_rpath_thinshell(key):
  dir_path = GAMMA_dir + f'/results/{key}/'
  file_path = dir_path + 'thinshell_rad.csv'
  return file_path

# def df_get_all_thinshell(df):
#   '''
#   Returns all quantities needed for the thin shell paper
#   '''
#   frontvars = ['i', 'x', 'dx', 'vx']
#   downvars = ['i', 'rho', 'D', 'p']
#   RSf, FSf = df_get_shFronts(df)
#   RSd, FSd = df_get_shDown(df)
#   #RSd, FSd = df_get_cellsBehindShock(df, n=5)
#   for cell in [RSd, FSd]:
#     cell['gmin'] = get_variable(cell, 'gma_m')

#   RScr, FScr = df_check_crossed(df)

#   iRS, rRS, drRS, vRS = RSf[frontvars] if not RScr else (0., 0., 0., 0.)
#   iFS, rFS, drFS, vFS = FSf[frontvars] if not FScr else (0., 0., 0., 0.)
#   iRSd, rhoRS, DRS, pRS = RSd[downvars] if not RScr else (0., 0., 0., 0.)
#   iFSd, rhoFS, DFS, pFS = FSd[downvars] if not FScr else (0., 0., 0., 0.)
#   lfacRS = DRS/rhoRS if rhoRS else 0.
#   lfacFS = DFS/rhoFS if rhoFS else 0.
#   TonRS, TthRS, TejRS = (0., 0., 0.) if RSf.empty else get_variable(RSf, 'obsT')
#   TonFS, TthFS, TejFS = (0., 0., 0.) if FSf.empty else get_variable(FSf, 'obsT')
#   rCD, rho2, rho3, vCD, pCD = df_get_CDvals(df)

#   radList  = [rRS, rCD, rFS]
#   idDList  = [iRSd, iFSd]
#   rhoList  = [rhoRS, rho3, rho2, rhoFS]
#   vList    = [vRS, vCD, vFS]
#   lfaclist = [lfacRS, derive_Lorentz(vCD), lfacFS]
#   prsList  = [pRS, pCD, pFS]
#   idShList = [iRS, iFS]
#   drList   = [drRS, drFS]
#   ShStList = get_shocksStrength(df)
#   lfacShList = get_shocksLfac(df)
#   TejList  = [TejRS, TejFS]
#   TthList  = [TthRS, TthFS]
#   numList  = [0. if cell.empty else get_variable(cell, 'nu_m') for cell in [RSd, FSd]]
#   LthList  = [0. if cell.empty else get_variable(cell, 'Lth') for cell in [RSd, FSd]]

#   return radList, vList, lfaclist, prsList, rhoList, idDList, \
#     idShList, drList, ShStList, lfacShList, TejList, TthList, numList, LthList


def df_get_all_thinshell_v1(df):
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
  lfacShList = get_shocksLfac(df)
  TonList  = [TonRS, TonFS]
  TejList  = [TejRS, TejFS]
  TthList  = [TthRS, TthFS]
  numList  = [0. if cell.empty else get_variable(cell, 'nu_m') for cell in [RS, FS]]
  LthList  = [0. if cell.empty else get_variable(cell, 'Lth') for cell in [RS, FS]]
  LpList   = [0. if cell.empty else get_variable(cell, 'Lp') for cell in [RS, FS]]

  return radList, vList, lfacList, prsList, rhoList, idShList, \
    idShList, drList, ShStList, lfacShList, TonList, TejList, TthList, numList, LthList, LpList

# plot any possible variable obtainable from the basic hydro variables
def plot_multi_fronts(key, varlist=['lfac', 'gma_m', 'B'], xscale='r', plaw=False):
  fig = plt.figure()
  gs = fig.add_gridspec(len(varlist), 1, hspace=0)
  axs = gs.subplots(sharex='all')

  xlabel = ''
  if xscale == 'r':
    xlabel = '$r/R_0$'
  elif xscale == 't':
    xlabel = '$t/t_0$'
  elif xscale == 'T':
    xlabel = '$\\bar{T}$'
  axs[-1].set_xlabel(xlabel)

  for var, ax in zip(varlist, axs):
    plot_atfronts(key, var, xscale=xscale, plaw=plaw, ax_in=ax)
  

def plot_atfronts(key, var, xscale='r', plaw=False, ax_in=None):
  '''plots variable var behind shock fronts'''
  kernel_size = 50
  sigma = 5
  if not ax_in:
    plt.figure()
    ax = plt.gca()
  else:
    ax = ax_in

  xlabel = ''
  if xscale == 'r':
    xlabel = '$r/R_0$'
  elif xscale == 't':
    xlabel = '$t/t_0$'
  elif xscale == 'T':
    xlabel = '$\\bar{T}$'
  if not ax_in: ax.set_xlabel(xlabel)
  ylabel = var_exp[var]
  if plaw:
    ylabel = 'd$\\ln ' + ylabel[1:] + '/d$\\ln ' + xlabel[1:]
  ax.set_ylabel(ylabel)

  env = MyEnv(key)
  file_path = get_fpath_thinshell(key)
  run_data = pd.read_csv(file_path, index_col=0)
  run_data.attrs['key'] = key
  t = run_data['time'].to_numpy() + env.t0

  for sh in ['RS', 'FS']:
    col = cols_dic[sh]
    r = run_data[f'r_{{{sh}}}'].to_numpy()
    val = get_variable_sh(run_data, var, sh)
    val = val[r>0]
    t = t[r>0]
    r = r[r>0]
    if xscale == 't':
      x = t/env.t0
    elif xscale == 'r':
      x = r*c_/env.R0
    elif xscale == 'T':
      T = (1+env.z)*(t - r)
      x = (T-env.Ts)/env.T0
    
    if plaw:
      # plot d ln(val)/d ln(x)
      # smooth first
      arr = val/val.max()
      val = gaussian_filter1d(arr, sigma, order=0)
      # kernel = np.ones(kernel_size) / kernel_size
      # val = np.convolve(val, kernel, mode='same')
      if xscale == 'T': x += 1.
      lnx = np.log(x)
      lnval = np.log(val)
      val = np.gradient(lnval, lnx)
  
    ax.loglog(x, val, c=col)


# smoothing procedure
def smoothing_rollave(arr, N):
  '''
  Smooth array with a rolling average of size N
  '''
  kernel = np.ones((N,))/N
  arr_padded = np.pad(arr, (N//2, N-1-N//2), mode='edge')
  return np.convolve(arr_padded, kernel, mode='valid')


# overwrite corresponding functions from data_io to include shock name
def get_vars_sh(df, varlist, sh):
  '''
  Return array of variables
  '''
  if type(df) == pd.core.series.Series:
    out = np.zeros((len(varlist)))
  else:
    out = np.zeros((len(varlist), len(df)))
  for i, var in enumerate(varlist):
    out[i] = get_variable_sh(df, var, sh)
  return out

def get_variable_sh(df, var, sh):
  '''
  Returns chosen variable for the corresponding shock front
  '''
  varsh = f'{var}_{{{sh}}}'
  if varsh in df.keys():
    try:
      res = df[varsh].to_numpy(copy=True)
    except AttributeError:
      res = df[varsh]
  else:
    func, inlist, envlist = var2func[var]
    res = df_get_var(df, func, inlist, envlist, sh)
  return res

def df_get_var_sh(df, func, inlist, envlist, sh):
  params, envParams = get_params_sh(df, inlist, envlist, sh)
  res = func(*params, *envParams)
  return res

def get_params_sh(df, inlist, envlist, sh):
  if (type(df) == pd.core.series.Series):
    params = np.empty(len(inlist))
  else:
    params = np.empty((len(inlist), len(df)))
  for i, var in enumerate(inlist):
    var = f'{var}_{{{sh}}}'
    if var in df.attrs:
      params[i] = df.attrs[var].copy()
    else:
      try:
        params[i] = df[var].to_numpy(copy=True)
      except AttributeError:
        params[i] = df[var]
  envParams = []
  if len(envlist) > 0.:
    env = MyEnv(df.attrs['key'])
    envParams = [getattr(env, name) for name in envlist]
  return params, envParams