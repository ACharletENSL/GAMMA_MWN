from thinshell_analysis import *
import sys

key = sys.argv[1] if len(sys.argv) > 1 else 'HPC_sph'
func = 'Band'
plot_1stfit = True
nuobs, Tobs, env = get_radEnv(key)
path = get_contribspath(key)
nupks, nuFnupks = get_pksnunFnu(key)
Tb = (Tobs-env.Ts)/env.T0

shocks = ['RS', 'FS']
cols = ['r', 'b']
gval, mval, nval = True, False, False
g_vals = [env.gRS, env.gFS] if gval else [1, 1]
m_vals = [-0.26, 0.06] if mval else [0, 0]
n_vals = [0.53, 0.2] if nval else [0, 0]
norms_nu = [env.nu0, env.nu0FS]
norms_nuF = [env.nu0F0, env.nu0F0FS]
norms_T = [env.T0, env.T0FS]

fig = plt.figure()
gs = fig.add_gridspec(2, 1, hspace=0)
axs = gs.subplots(sharex='all')
ax_nu, ax_nF = axs
ax_nu.set_ylabel('$\\nu_{pk}/\\nu_0$')
ax_nF.set_ylabel('$\\nu_{pk}F_{\\nu_{pk}}/\\nu_0F_0$')
ax_nF.set_xlabel('$\\bar{T}_{sh}$')
title = ''


fig2, axs2 = plt.subplots(2, 1, sharex='all')
ax_xi, ax_y = axs2
ax_xi.set_ylabel('$\\xi_{eff}$')
ax_y.set_ylabel('$R_{eff}$')
ax_y.set_xlabel('$\\bar{T}$')


if plot_1stfit:
  figr, axr = plt.subplots()
  axr.set_xlabel('$\\bar{T}$')
  axr.set_ylabel('$R_L$')
  fig3, axs3 = plt.subplots(3, 1, sharex='all')
  ax1, ax2, ax3 = axs3
  axs3[-1].set_xlabel('$r/R_0$')
i_smp = 20 # sampling to estimate initial plaw index
iff = -1
i_s = 5 # flawed data from numerics at beginning

inputs = [shocks, cols, nupks, nuFnupks, g_vals, norms_T, norms_nu, norms_nuF, ['--', '-.']]
for sh, col, nupk, nuFnupk, g, T0, norm_nu, norm_nuF, l in zip(*inputs):

  nu = nupk/env.nu0
  nuFnu = nuFnupk/env.nu0F0

  df = pd.read_csv(path+sh+'.csv')
  rsh, lfac, num, Lnum = df[[f'r_{{{sh}}}', f'lfac_{{{sh}}}', f'nu_m_{{{sh}}}', f'Lth_{{{sh}}}']].to_numpy().transpose()
  rsh, lfac, num, Lnum = rsh[rsh>0.], lfac[rsh>0.], num[rsh>0.], Lnum[rsh>0.]
  rsh *= c_/env.R0
  nup = (1+env.z)*num/(2*lfac)
  nup /= env.nu0p
  Lp = Lnum/(2*lfac)
  Lp /= (env.dEp0/env.T0)
  lfac /= env.lfac
  tT = 1 + (Tobs - env.Ts)/T0
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
  popt_lfac, pcov = curve_fit(fitfunc_lfac, rsh, lfac, bounds=([0.75, 0.5, ml, 1], [1.25, 2, mu, np.inf]))
  lfacpl, lfacsph, m, s_lfac = popt_lfac
  lfacpl  *= env.lfac
  lfacsph *= env.lfac
  print(f'lfac {sh}: m = {-2*m:.2f}, starts at {lfacpl:.1f}, saturates at {lfacsph:.1f}, s = {s_lfac:.1f}')

  # fit r
  # def fitfunc_rl(tT, k, s):
  #   return (tT**s + tT**(k*s))**(1/s)

  # T0pl  = (1+env.z)*env.R0/(2*(m+1)*lfacpl**2*c_)
  # T0sph = (1+env.z)*env.R0/(2*lfacsph**2*c_)
  # def fitfunc_rl(tT, s):
  #   T = T0*tT - 1
  #   tT_pl  = 1 + T/T0pl
  #   tT_sph = 1 + T/T0sph
  #   Rpl = tT_pl**(1/(m+1))
  #   Rsph = tT_sph
  #   return (Rpl**s + Rsph**s)**(1/s)
  # popt_r, pcov = curve_fit(fitfunc_rl, tT[:i_f], rsh[:i_f])

  # fit nup * r
  d0 = logslope(rsh[0], rsh[0]*nup[0], rsh[i_smp], rsh[i_smp]*nup[i_smp])
  def fitfunc_rnup(r, Xpl, Xsph, d, s):  
    return ((Xpl * r**d)**s + Xsph**s)**(1/s)
  popt_nup, pcov = curve_fit(fitfunc_rnup, rsh, rsh*nup,
      bounds=([1e-3, 1e-4, 1.1*d0, 1], [1., 1., 0.9*d0, np.inf]))
  rnuppl, rnupsph, d, s_nup = popt_nup
  print(f"nup {sh}: d = {d-1:.2f}, starts at {rnuppl:.1e} nu'_0, saturates at {rnupsph:.1e} nu'_0, s = {s_nup:.1f}")

  # fit Lp
  a0 = logslope(rsh[0], Lp[0], rsh[i_smp], Lp[i_smp])
  def fitfunc_Lp(r, Xpl, Xsph, a, s):
    return ((Xpl * r**a)**-s + Xsph**-s)**(-1/s)
  popt_Lp, pcov = curve_fit(fitfunc_Lp, rsh, Lp, bounds=([1e-4, 1e-4, 0.9*a0, 1], [1, 1, 1.1*a0, np.inf]))
  Lppl, Lpsph, a, s_Lp = popt_Lp
  print(f"Lp {sh}: a = {a:.2f}, starts at {Lppl:.1e} L', saturates at {Lpsph:.1e} L'0, s = {s_Lp:.1f}")

  def rl_fitted(tT):
    return fitfunc_rl(tT, *popt_r)
  def lfac_fitted(r):
    return fitfunc_lfac(r, *popt_lfac)
  def nup_fitted(r):
    return fitfunc_rnup(r, *popt_nup)/r
  def Lp_fitted(r):
    return fitfunc_Lp(r, *popt_Lp)
  if plot_1stfit:
    # axr.loglog(tT[:i_f], rsh[:i_f], c=col)
    # axr.loglog(tT[:i_f], rl_fitted(tT[:i_f]), ls='--', c='k')
    ax1.semilogx(rsh, lfac, c=col)
    ax1.semilogx(rsh, lfac_fitted(rsh), c='lime', ls=':', lw=1.1)
    ax1.set_ylabel('$\\Gamma$')
    ax1.grid(True, which='both')
    ax2.loglog(rsh, nup, c=col)
    ax2.loglog(rsh, nup_fitted(rsh), c='lime', ls=':', lw=1.1)
    ax2.set_ylabel("$\\nu'/\\nu'_0$")
    ax2.grid(True, which='both')
    ax3.loglog(rsh, Lp, c=col)
    ax3.loglog(rsh, Lp_fitted(rsh), c='lime', ls=':', lw=1.1)
    ax3.set_ylabel("$L'/L'_0$")
    ax3.grid(True, which='both')


  def tT_to_ximax(tT, m=0):
    return (g**2/(m+1))*(tT-1)

  def xi_eff(tT, k, l, s):
    ximax = tT_to_ximax(tT)
    return k*(ximax**-s + (ximax**l)**-s)**(-1/s)
  
  def xi_to_y(xi, m=0):
    return (1 + (m+1)*g**-2*xi)**(-1/(m+1))

  
  def fit_nupk_rise(tT, k):#, l, s):
    #xi = xi_eff(tT, k, l, s)
    xi = k*tT_to_ximax(tT)
    y = xi_to_y(xi)
    r = tT # R_L(T) = R_0 \tilde{T}
    #r = rl_fitted(tT)
    reff = y*r
    return lfac_fitted(reff)*nup_fitted(reff)/(1+xi)
  
  def fit_nuFnupk_rise(tT, k):#, l, s):
    #xi = xi_eff(tT, k, l, s)
    ximax = tT_to_ximax(tT)
    xi = k*ximax
    y = xi_to_y(xi)
    r = tT # R_L(T) = R_0 \tilde{T}
    #r = rl_fitted(tT)
    reff = r*y
    fittedfuncs = lfac_fitted(reff)**4 * lfac_fitted(r)**-2 * nup_fitted(reff) * Lp_fitted(reff)
    # factor of 3 for the normalization, Fs = 3 F0
    return 3 * (ximax/(g**2+ximax)) * y**-2 * (1+xi)**-4 * fittedfuncs

  # to fit both functions with the same underlying physical parameters
  def fit_both_rise(tT, k):#, l, s):
    nufit = fit_nupk_rise(tT, k)#, l, s)
    nuFnufit = fit_nuFnupk_rise(tT, k)#, l, s)
    fit = np.hstack((nufit, nuFnufit))
    return fit

  tT_rise, tT_HLE = np.split(tT, [i_f])
  tT_rise = tT_rise[i_s:]
  nu_rise, nu_HLE = np.split(nu, [i_f])
  nu_rise = nu_rise[i_s:]
  nuFnu_rise, nuFnu_HLE = np.split(nuFnu, [i_f])
  nuFnu_rise = nuFnu_rise[i_s:]
  ydata_rise = np.hstack((nu_rise, nuFnu_rise))
  ydata_HLE = np.hstack((nu_HLE, nuFnu_HLE))


  sigma = np.ones(tT_rise.shape)
  sigma[iff] = 0.01

  sigma = np.hstack((sigma, sigma))
  # params order: k, l, s
  popt1, pcov1 = curve_fit(fit_both_rise, tT_rise, ydata_rise,
      p0=[0.3], bounds=([0.1], [1]), sigma=sigma)
      #p0=[0.3, 1.3, 3], bounds=([0.1, 0.8, 1], [1, 2, np.inf]), sigma=sigma)
  #k, l, s = popt1
  k = popt1[0]
  title += f'{sh}: d = {d:.2f}, k = {k:.2f}'#, l={l:.2f}, s = {s:.1f}'
  title += '\n'
  fit_xi = k * tT_to_ximax(tT[:i_f])
  #fit_xi = xi_eff(tT[:i_f], k, l, s)
  fit_nu_rise = fit_nupk_rise(tT[:i_f], *popt1)
  fit_nuFnu_rise = fit_nuFnupk_rise(tT[:i_f], *popt1)

  fit_nu = fit_nu_rise
  fit_nuFnu = fit_nuFnu_rise
  Tb = tT[:i_f] - 1
  reff = tT[:i_f]*xi_to_y(fit_xi)
  ax_xi.loglog(Tb, fit_xi, c=col)
  ax_xi.grid(True, which='both')
  ax_y.loglog(Tb, reff, c=col)
  ax_y.grid(True, which='both')
  ax_nu.loglog(Tb, nu[:i_f], c=col)
  ax_nu.loglog(Tb, fit_nu, ls=':', c='k')
  ax_nu.grid(True, which='both')
  ax_nF.loglog(Tb, nuFnu[:i_f], c=col)
  ax_nF.semilogx(Tb, fit_nuFnu, ls=':', c='k')
  ax_nF.grid(True, which='both')
  # ax_nu.axvline(Tb[i_f], c=col, ls='--', lw=.8)
  # ax_nF.axvline(Tb[i_f], c=col, ls='--', lw=.8)

ax1.set_ylabel('$\\Gamma/\\Gamma_{pl}$')
ax2.set_ylabel("$\\nu'_m/\\nu'_0$")
ax3.set_ylabel("$L'/L'_0$")
fig.suptitle(title[:-1])
plt.show()