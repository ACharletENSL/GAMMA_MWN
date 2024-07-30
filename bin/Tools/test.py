# -*- coding: utf-8 -*-
# @Author: acharlet

'''
Run this file to perform the current test
(instead of run plotting_scripts, def environment, etc.)
'''

from plotting_scripts_new import *
plt.ion()

'''
Plot the flux integrand at fixed nu and fixed T to get why it doesn't converge
'''

key = 'Last'
nuobs, Tobs, env = get_radEnv(key)
lognu, logT = -1, 0
nub = nuobs/env.nu0
i_nu = find_closest(nub, 10**lognu)
Tb = (Tobs-env.Ts)/env.T0
i_T = find_closest(Tb, 10**logT)
b1, b2 = -0.5, -1.25
colors = ['r', 'b', 'g', 'y', 'c', 'm', 'teal', 'darkorange']

def plot_fluxes(key):
  fig = plt.figure()
  gs = fig.add_gridspec(1, 2, hspace=0, wspace=0)
  axs = gs.subplots(sharey=True)
  axs[0].set_ylabel("$\\nu F_\\nu/\\nu_0F_0$")
  axs[0].set_xlabel("$\\bar{T}$")
  axs[1].set_xlabel("$\\tilde{nu}$")

  funcs = [flux_shockfront_hybrid, flux_shockfront_general]
  fnames = ['hybrid', 'general']
  lstyles = [':', '--', '-.']
  dummy_lst = [plt.plot([],[], ls=ls, color='k')[0] for ls in lstyles[:len(funcs)]]
  lstlegend = fig.legend(dummy_lst, fnames)
  for func, ls in zip(funcs, lstyles):
    nuFnus = get_nuFnu(nuobs, Tobs, env, func)
    lctot = np.zeros(nuFnus.shape[2])
    sptot = np.zeros(nuFnus.shape[1])
    for nuFnu, col in zip(nuFnus, colors):
      lc = nuFnu[i_nu, :]
      lctot += lc
      sp = nuFnu[:, i_T]
      sptot += sp
      axs[0].plot(Tb, lc, c=col, ls=ls)
      axs[1].loglog(nub, sp, c=col, ls=ls)
    axs[0].plot(Tb, lctot, c='k', ls=ls)
    axs[1].loglog(nub, sptot, c='k', ls=ls)

def plot_integrandsHyb(nu_bar, T_tilde, func='Band'):
  g, dR = env.gRS, env.dRRS/env.R0
  plt.figure()
  i=0
  for tT in T_tilde:
    for nub in nu_bar:
      ymin = min(tT**-1, 1.)
      ymax = min((1+dR)/tT, 1.)
      y = np.linspace(ymin, ymax)
      f1 = integrand_general(y, nub, tT, g, 1., -1., 0., b1, b2, func)
      f2 = integrand_hybrid_y(y, nub, tT, g, b1, b2, func)
      label = '$\\tilde{T}='+f'{tT:.1f}'+'$, $\\tilde{\\nu}=' + f'{nub:.1e}$'
      plt.semilogy(y, f1, ls='-.', c=colors[i], label=label)
      plt.semilogy(y, f2, ls='-', c='k', label=label)
      i += 1
      if i > len(colors):
        i = 0

def plot_integrands0(nu_bar, T_tilde, func='Band'):
  #nu0, g, dR = env.nu0FS, env.gFS, env.dRFS/env.R0 if sh == 'FS' else env.nu0, env.gRS, env.dRRS/env.R0
  g, dR = env.gRS, env.dRRS/env.R0
  plt.figure()
  i=0
  for tT in T_tilde:
    for nub in nu_bar:
      ymin = min(tT**-1, 1.)
      ymax = min((1+dR)/tT, 1.)
      y = np.linspace(ymin, ymax)
      f1 = integrand_general(y, nub, tT, g, 0., 0., 0., b1, b2, func)
      gy = 1 + g**2*(y**-1-1.)
      f2 = y**-2 *  gy**-3 * Band_func(gy*nub)
      label = '$\\tilde{T}='+f'{tT:.1f}'+'$, $\\tilde{\\nu}=' + f'{nub:.1e}$'
      plt.semilogy(y, f1, ls='-', c=colors[i], label=label)
      plt.semilogy(y, f2, ls='-.', c='k', label=label)
      i += 1
      if i > len(colors):
        i = 0
  #plt.legend()


