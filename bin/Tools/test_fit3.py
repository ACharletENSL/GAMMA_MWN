from thinshell_analysis import *
from functools import partial
import sys

key = sys.argv[1] if len(sys.argv) > 1 else 'sph_big'
func = 'Band'
plot_1stfit = False
plot_xi = True
nuobs, Tobs0, env = get_radEnv(key, forpks=True)
path = get_contribspath(key)
nupks, nuFnupks = get_pksnunFnu(key)
Tb = (Tobs0-env.Ts)/env.T0

shocks = ['RS', 'FS']
cols = ['r', 'b']
gval = True
g_vals = [env.gRS, env.gFS] if gval else [1, 1]
nu_facs = [1., 1./env.fac_nu]
F_facs = [1., 1./env.fac_F]
norms_T = [env.T0, env.T0FS]
norms_nu = [env.nu0, env.nu0FS]
norms_Lp = [env.L0p, env.L0pFS]
norms_F = [env.F0, env.F0FS]


fig = plt.figure()
gs = fig.add_gridspec(2, 1, hspace=0)
axs = gs.subplots(sharex='all')
ax_nu, ax_nF = axs
ax_nu.set_ylabel('$\\nu_{pk}/\\nu_0$')
ax_nF.set_ylabel('$\\nu_{pk}F_{\\nu_{pk}}/\\nu_0F_0$')
ax_nF.set_xlabel('$\\bar{T}_{sh}$')
title = ''

if plot_xi:
  fig2, axs2 = plt.subplots(2, 1, sharex='all')
  ax_xi, ax_err = axs2
  #ax_xi.set_title(f'log fitting {'ON' if withlog else 'OFF'}, effective final radius {'ON' if withReff else 'OFF'}')
  ax_xi.set_ylabel('$\\xi_{eff}$')
  #ax_y.set_ylabel('$y_{eff}$')
  ax_err.set_ylabel('err.')
  ax_err.set_xlabel('$\\bar{T}_{sh}$')
  dummy_lst = [plt.plot([], [], ls=l, c='k')[0] for l in ['-.', '--']]
  ax_err.legend(dummy_lst, ['$\\nu_{pk}$', '$\\nu_{pk} F_{\\nu_{pk}}$'])

if plot_1stfit:
  # figr, axr = plt.subplots()
  # axr.set_xlabel('$\\tilde{T}$')
  # axr.set_ylabel('$R_L$')
  fig3, axs3 = plt.subplots(3, 1, sharex='all')
  ax1, ax2, ax3 = axs3
  ax1.set_ylabel('$\\Gamma$')
  ax2.set_ylabel("$\\nu'_m/\\nu'_0$")
  ax3.set_ylabel("$L'/L'_0$")
  axs3[-1].set_xlabel('$r/R_0$')
i_smp = 20 # sampling to estimate initial plaw index
iff = -2
i_s = 15 # flawed data from numerics at beginning

inputs = [shocks, cols, nupks, nuFnupks, g_vals, norms_T, norms_nu, norms_Lp, norms_F, nu_facs, F_facs, ['--', '-.']]
for sh, col, nupk, nuFnupk, g, T0, nu0, Lp0, F0, fac_nu, fac_F, l in zip(*inputs):

  Tobs = Tobs0[nuFnupk>0]
  nu = nupk[nuFnupk>0]/nu0
  nuFnu = nuFnupk[nuFnupk>0]/(nu0*F0)
  nup0 = (1+env.z)*nu0/(2*env.lfac)
  #Lp0 = 3*F0/(2*env.lfac*env.zdl)
  fac_nuFnu = fac_nu*fac_F # difference in normalization
  nuNorm = 1.
  FnuNorm = 1.

  df = pd.read_csv(path+sh+'.csv')
  rsh, lfac, num, Lp = df[[f'r_{{{sh}}}', f'lfac_{{{sh}}}', f'nu_m_{{{sh}}}', f'Lp_{{{sh}}}']].to_numpy().transpose()
  rsh, lfac, num, Lp = rsh[rsh>0.][i_s:], lfac[rsh>0.][i_s:], num[rsh>0.][i_s:], Lp[rsh>0.][i_s:]
  rsh *= c_/env.R0
  nup = (1+env.z)*num/(2*lfac)
  #nup /= env.nu0p
  nup /= nup0
  #Lp = Lnum/(2*lfac)
  #Lp /= env.L0p
  Lp /= Lp0
  lfac /= env.lfac
  tT = 1 + (Tobs - env.Ts)/T0
  tT = tT
  
  def tT_to_tTth(tT, Tth):
    '''
    Go from normalized by T_r(R0) to normalized by T_theta
    '''
    return 1 + (tT - 1) * T0/Tth
  
  row = df.iloc[-2]
  tf, rf, Tej, Tthf = row[['time', f'r_{{{sh}}}', f'Tej_{{{sh}}}', f'Tth_{{{sh}}}']]
  Tonf = Tej + Tthf
  tTf = 1 + (Tonf - env.Ts)/T0
  i_f = find_closest(tT, tTf)


  # fit lfac
  m0 = logslope(rsh[0], lfac[0], rsh[i_smp], lfac[i_smp])
  ml = min(m0*0.9, m0*1.1)
  mu = max(m0*0.9, m0*1.1)
  def fitfunc_lfac(r, Xpl, Xsph, m, s):
    if m<0:
      return ((Xpl * r**m)**s + Xsph**s)**(1/s)
    elif m>0:
      return ((Xpl * r**m)**-s + Xsph**-s)**(-1/s)
    else:
      return Xsph*np.ones(r.shape)
  popt_lfac, pcov = curve_fit(fitfunc_lfac, rsh, lfac, p0=[lfac[0], lfac[-1], m0, 2],
    bounds=([0.75, 0.5, ml, 1], [1.25, 2, mu, np.inf]))
  lfacpl, lfacsph, m, s_lfac = popt_lfac
  print(f'lfac {sh}: m = {m:.2f}, starts at {lfacpl*env.lfac:.1f}, saturates at {lfacsph*env.lfac:.1f}, s = {s_lfac:.1f}')


  # fit nu'
  d0 = logslope(rsh[0], nup[0], rsh[i_smp], nup[i_smp]) - 1
  dl = min(d0*0.9, d0*1.1)
  du = max(d0*0.9, d0*1.1)
  def fitfunc_nup(r, Xpl, Xsph, d, s):  
    return ((Xpl * r**d)**s + (Xsph*r**-1)**s)**(1/s)
  popt_nup, pcov = curve_fit(fitfunc_nup, rsh, nup, p0=[nup[0], nup[-1], d0, 2],
      bounds=([0.1, 0.1, dl, 1], [10, 10, du, np.inf]))
  nuppl, nupsph, d, s_nup = popt_nup
  print(f"nup {sh}: d = {d:.2f}, starts at {nuppl:.1e} nu'_0, saturates at {nupsph:.1e} nu'_0, s = {s_nup:.1f}")


  # fit Lp
  a0 = logslope(rsh[0], Lp[0], rsh[i_smp], Lp[i_smp])
  al = min(a0*0.9, a0*1.1)
  au = max(a0*0.9, a0*1.1)
  def fitfunc_Lp(r, Xpl, Xsph, a, s):
    return ((Xpl * r**a)**-s + (Xsph*r)**-s)**(-1/s)
  popt_Lp, pcov = curve_fit(fitfunc_Lp, rsh, Lp, #p0=[Lp[0], Lp[-1], a0, 2],
    bounds=([0.1, 0.1, al, 1], [10, 10, au, np.inf]))
  Lppl, Lpsph, a, s_Lp = popt_Lp
  print(f"Lp {sh}: a = {a:.2f}, starts at {Lppl:.1e} L', saturates at {Lpsph:.1e} L'0, s = {s_Lp:.1f}")

  def lfac_fitted(r):
    #return fitfunc_lfac(r, *popt_lfac)
    return fitfunc_lfac(r, 1., lfacsph/lfacpl, m, s_lfac)
  def nup_fitted(r):
    #return fitfunc_nup(r, *popt_nup)
    return fitfunc_nup(r, 1., nupsph/nuppl, d, s_nup)
  def Lp_fitted(r):
    #return fitfunc_Lp(r, *popt_Lp)
    return fitfunc_Lp(r, 1., Lpsph/Lppl, a, s_Lp)
  if plot_1stfit:
    ax1.semilogx(rsh, lfac*env.lfac, c=col)
    ax1.semilogx(rsh, lfac_fitted(rsh)*lfacpl*env.lfac, c='lime', ls=':', lw=1.1)
    ax1.grid(True, which='both')
    ax2.loglog(rsh, nup, c=col)
    ax2.loglog(rsh, nup_fitted(rsh)*nuppl, c='lime', ls=':', lw=1.1)
    ax2.grid(True, which='both')
    ax3.loglog(rsh, Lp, c=col)
    ax3.loglog(rsh, Lp_fitted(rsh)*Lppl, c='lime', ls=':', lw=1.1)
    ax3.grid(True, which='both')


  def tT_to_ximax(tT, m=0):
    return (g**2/(m+1))*(tT-1)

  def xi_eff(tT, xi_sat, k, s, m=0):
    ximax = tT_to_ximax(tT, m)
    return k*(ximax**-s + xi_sat**-s)**(-1/s)

  def xi_to_y(xi, m=0):
    return (1 + (m+1)*g**-2*xi)**(-1/(m+1))

  nu_pl = lfacpl*nuppl
  F_pl = lfacpl*Lppl
  nuFnu_pl = nu_pl*F_pl

  def fit_nupk_rise(tT, knu, xi_sat, k, s, m=0):
    xi = xi_eff(tT, xi_sat, k, s, m)
    y = xi_to_y(xi, m)
    r = tT # R_L(T) = R_0 \tilde{T}
    reff = y*r
    return  knu * nu_pl * lfac_fitted(reff)*nup_fitted(reff)/(1+xi) 

  def fit_nuFnupk_rise(tT, knu, kF, xi_sat, k, s, m=0):
    ximax = tT_to_ximax(tT, m)
    xi = xi_eff(tT, xi_sat, k, s, m)
    y = xi_to_y(xi, m)
    r = tT # R_L(T) = R_0 \tilde{T}
    reff = r*y
    nu = fit_nupk_rise(tT, knu, xi_sat, k, s, m)
    Fnu = kF * F_pl * ximax*(1+xi)**-3 * lfac_fitted(reff)**3*lfac_fitted(r)**-2*Lp_fitted(reff)
    return nu*Fnu

  # to fit both functions with the same underlying physical parameters
  def fit_both_rise(tT, knu, kF, xi_sat, k, s, m):
    nufit = fit_nupk_rise(tT, knu, xi_sat, k, s, m)
    nuFnufit = fit_nuFnupk_rise(tT, knu, kF, xi_sat, k, s, m)
    fit = np.hstack((nufit, nuFnufit))
    return fit

  tT_rise, tT_HLE = np.split(tT, [i_f])
  istart = np.searchsorted(tT_rise-1, 1e-2)
  tT_rise = tT_rise[istart:]
  nu_rise, nu_HLE = np.split(nu, [i_f])
  nu_rise = nu_rise[istart:]
  nuFnu_rise, nuFnu_HLE = np.split(nuFnu, [i_f])
  nuFnu_rise = nuFnu_rise[istart:]
  ydata_rise = np.hstack((nu_rise, nuFnu_rise))
  ydata_HLE = np.hstack((nu_HLE, nuFnu_HLE))

  sigma = np.ones(tT_rise.shape)
  sigma[-1] = 0.01

  sigma = np.hstack((sigma, sigma))
  # params order: knu, kF, xi_sat, k, s
  popt1, pcov1 = curve_fit(partial(fit_both_rise, m=0), tT_rise, ydata_rise,
      p0=[1., 1., 5, 0.3, 3], bounds=([0.9, 0.9, 1, 0.1, 1.], [1.1, 1.1, 10, 1, np.inf]), sigma=sigma)
  knu, kF, xi_sat, k, s = popt1
  title += f'{sh}: $\\xi_{{sat}}$ = {xi_sat:.2f}, k = {k:.2f}, s = {s:.1f}, $k_\\nu$ = {knu:.2f}, $k_F$ = {kF:.2f}'

  fit_xi = xi_eff(tT[:i_f], xi_sat, k, s, m)
  fit_nu_rise = fit_nupk_rise(tT[:i_f], *popt1)
  fit_nuFnu_rise = fit_nuFnupk_rise(tT[:i_f], *popt1)

  # fit HLE
  Tth = Tthf
  yf = (1 + (m+1)*g**-2 *fit_xi[-1])**(-1/(m+1))
  Tth = Tthf * yf**(m+1)
  def fit_nupk_HLE(tT, dnu):
    nupkf = fit_nupk_rise(tTf, *popt1)
    tTth = tT_to_tTth(tT, Tth)
    return nupkf * (tTth/tTth[0])**dnu
  def fit_nuFnupk_HLE(tT, dnu, dF):
    nuFnupkf = fit_nuFnupk_rise(tTf, *popt1)
    tTth = tT_to_tTth(tT, Tth)
    d = dnu+dF
    return nuFnupkf * (tTth/tTth[0])**d
  def fit_both_HLE(tT, dnu, dF):
    nufit = fit_nupk_HLE(tT, dnu)
    nuFnufit = fit_nuFnupk_HLE(tT, dnu, dF)
    fit = np.hstack((nufit, nuFnufit))
    return fit

  sigma = np.ones(tT_HLE.shape)
  sigma[0] = 0.01
  sigma = np.hstack((sigma, sigma))
  popt2, pcov2 = curve_fit(fit_both_HLE, tT_HLE, ydata_HLE, p0=[-1, -2], sigma=sigma)
  dnu, dF = popt2
  fit_nu_HLE = fit_nupk_HLE(tT_HLE, dnu) 
  fit_nuFnu_HLE = fit_nuFnupk_HLE(tT_HLE, dnu, dF)
  title += f', d$_\\nu$ = {dnu:.2f}, d$_F$ = {dF:.2f}\n'


  fit_nu = np.concatenate((fit_nu_rise, fit_nu_HLE), axis=None)
  fit_nuFnu = np.concatenate((fit_nuFnu_rise, fit_nuFnu_HLE), axis=None)
  nu_err_f = np.abs(fit_nu[istart:] - nu[istart:])/nu[istart:]
  nu_err = nu_err_f.max()
  nuFnu_err_f = np.abs(fit_nuFnu[istart:] - nuFnu[istart:])/nuFnu[istart:]
  nuFnu_err = nuFnu_err_f.max()

  Tb = tT - 1
  if plot_xi:
    ax_xi.loglog(Tb[:i_f], fit_xi, c=col)
    ax_xi.grid(True, which='both')
    ax_err.semilogx(Tb[istart:], nu_err_f, ls='-.', c=col)
    ax_err.semilogx(Tb[istart:], nuFnu_err_f, ls='--', c=col)
    #ax_y.loglog(Tb, xi_to_y(fit_xi), c=col)
    ax_err.grid(True, which='both')
    
  ax_nu.loglog(Tb, nu*fac_nu, c=col)
  ax_nu.loglog(Tb, fit_nu*fac_nu, ls='--', c='k')
  ax_nu.grid(True, which='both')
  ax_nF.loglog(Tb, nuFnu*fac_nuFnu, c=col)
  ax_nF.semilogx(Tb, fit_nuFnu*fac_nuFnu, ls='--', c='k')
  ax_nF.grid(True, which='both')
  # ax_nu.axvline(Tb[i_f], c=col, ls='--', lw=.8)
  # ax_nF.axvline(Tb[i_f], c=col, ls='--', lw=.8)

fig.suptitle(title[:-1])
plt.show()