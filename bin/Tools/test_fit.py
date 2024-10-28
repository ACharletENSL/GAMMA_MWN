from thinshell_analysis import *

key = 'sph_fid'
func = 'Band'
nuobs, Tobs, env = get_radEnv(key)
path = get_contribspath(key)
nupks, nuFnupks = get_pksnunFnu(key)
Tb = (Tobs-env.Ts)/env.T0

shocks = ['RS', 'FS']
cols = ['r', 'b']
g_vals = [env.gRS, env.gFS]
m_vals = [-0.26, 0.02]
norms_nu = [env.nu0, env.nu0FS]
norms_nuF = [env.nu0F0, env.nu0F0FS]
norms_T = [env.T0, env.T0FS]

fig = plt.figure()
gs = fig.add_gridspec(2, 1, hspace=0)
axs = gs.subplots(sharex='all')
ax_nu, ax_nF = axs
ax_nu.set_ylabel('$\\nu_{pk}/\\nu_0$')
ax_nF.set_ylabel('$\\nu_{pk}F_{\\nu_{pk}}/\\nu_0F_0$')
ax_nF.set_xlabel('$\\bar{T}_{RS}$')

for sh, nupk, nuFnupk, g, m, T0, norm_nu, norm_nuF, col in zip(shocks, nupks, nuFnupks, g_vals, m_vals, norms_T, norms_nu, norms_nuF, cols):
  df = pd.read_csv(path+sh+'.csv')
  row = df.iloc[-2]
  tf, rf, Tej, Tth = row[['time', f'r_{{{sh}}}', f'Tej_{{{sh}}}', f'Tth_{{{sh}}}']]
  Tonf = Tej + Tth
  tT = 1 + (Tobs - env.Ts)/T0
  tTf = 1 + (Tonf - env.Ts)/T0
  i_f = find_closest(tT, tTf)
  nu = nupk/norm_nu
  nuFnu = nuFnupk/norm_nuF

  def nupk_ofxi(xi, tT, d):
    return (1+xi)**-1 * (1+g**-2*xi)**-d * tT**d
  def fit_nupk_rise1(tT, d, k):
    xi = k * (g**2/(m+1)) * (tT-1)
    return nupk_ofxi(xi, tT, d)
  def fit_nupk_rise2(tT, d, k):
    xi = k 
    return nupk_ofxi(xi, tT, d)
  def fit_nupk_rise_2d(tT, d1, d2, k):
    xi = k * (g**2/(m+1)) * (tT-1)
    return np.where(xi<=k, fit_nupk_rise1(tT, d1, k), fit_nupk_rise2(tT, d2, k))
  popt, pcov = curve_fit(fit_nupk_rise_2d, tT[2:i_f], nu[2:i_f], p0=[-2, -1.9, 0.05], bounds=([-5, -5, 0], [0, 0, 1]))
  d1, d2, k1 = popt
  perr_nu1 = np.sqrt(np.diag(pcov))
  nu_rise = fit_nupk_rise_2d(tT[:i_f], *popt)

  def get_tTeff2(tT):
    return 1. + np.abs((m+1+g**2)/(g**2*(g**2-1)))  * (rf*c_/env.R0)**-(m+1) * (tT-1.)
  def fit_nupk_HLE(tT, d):
    tTeff2 = get_tTeff2(tT)
    nuf = fit_nupk_rise_2d(tTf, d1, d2, k1)
    return nuf * (tTeff2/tTeff2[0])**d
  popt, pcov = curve_fit(fit_nupk_HLE, tT[i_f:], nu[i_f:], p0=[-1])
  d3 = popt[0]
  perr_nu2 = np.sqrt(np.diag(pcov))
  nu_HLE = fit_nupk_HLE(tT[i_f:], *popt)
  nu_fit = np.concatenate((nu_rise, nu_HLE), axis=None)

  def integ(x):
    return x**-4 * Band_func(x)
  def fit_nuFnupk_rise1(tT, k):
    xi_m = (g**2/(m+1)) * (tT - 1)
    return xi_m * integ(1+k*xi_m)
  def fit_nuFnupk_rise2(tT, k):
    quant = spi.quad(integ, 1, 1+k)[0]
    return (quant/k) * np.ones(tT.shape)
  def fit_nuFnupk_rise_2k(tT, k1, k2, s):
    nuFnu1 = fit_nuFnupk_rise1(tT, k1)
    nuFnu2 = fit_nuFnupk_rise2(tT, k2)
    return (nuFnu1**s + nuFnu2**s)**(1/s)
  def fit_nuFnupk_rise(tT, k):
    xi = k * (g**2/(m+1)) * (tT-1)
    return np.where(xi<=k, fit_nuFnupk_rise1(tT, k), fit_nuFnupk_rise2(tT, k))
  def fit_nuFnupk_rise_1k(tT, k, s):
    nuFnu1 = fit_nuFnupk_rise1(tT, k)
    nuFnu2 = fit_nuFnupk_rise2(tT, k)
    return (nuFnu1**s + nuFnu2**s)**(1/s)
  # popt, pcov = curve_fit(fit_nuFnupk_rise_2k, tT[:i_f], nuFnu[:i_f], p0=[0.1, 0.1, 3], bounds=([0, 0, -np.inf], [1, 1, np.inf]))
  # k2, k3, s = popt

  # nuFnu_rise = fit_nuFnupk_rise_2k(tT[:i_f], *popt)
  # popt, pcov = curve_fit(fit_nuFnupk_rise, tT[:i_f], nuFnu[:i_f], p0=[0.1], bounds=(0, 1))
  # k2 = popt[0]
  # nuFnu_rise = fit_nuFnupk_rise(tT[:i_f], *popt)
  popt, pcov = curve_fit(fit_nuFnupk_rise_1k, tT[:i_f], nuFnu[:i_f], p0=[0.1, -2], bounds=([0, -np.inf], [1, np.inf]))
  k2, s = popt
  nuFnu_rise = fit_nuFnupk_rise_1k(tT[:i_f], *popt)
  # perr_nF1 = np.sqrt(np.diag(pcov))

  def get_tTeff2(tT, k):
    return 1. + np.abs(k*(m+1+g**2)/(g**2*(g**2-1)))  * (rf*c_/env.R0)**-(m+1) * (tT-1.)
  def fit_nuFnupk_HLE(tT, d, k):
    #nuFnupkf = fit_nuFnupk_rise_2k(tTf, k2, k3, s)
    #nuFnupkf = fit_nuFnupk_rise(tTf, k2)
    nuFnupkf = fit_nuFnupk_rise_1k(tTf, k2, s)
    tTeff2 = get_tTeff2(tT, k)
    return nuFnupkf * (tTeff2/tTeff2[0])**d
  popt, pcov = curve_fit(fit_nuFnupk_HLE, tT[i_f:], nuFnu[i_f:], p0=[-3, 1], bounds=([-5, -1], [0,np.inf]))
  d4, k4 = popt
  perr_nF2 = np.sqrt(np.diag(pcov))
  nuFnu_HLE = fit_nuFnupk_HLE(tT[i_f:], *popt)
  nuFnu_fit = np.concatenate((nuFnu_rise, nuFnu_HLE), axis=None)

  print(f'nu_{sh}: d1, d2, d3 = {d1:.2f}, {d2:.2f}, {d3:.2f}; k1 = {k1:.2f}')
  #print(f'Error in nu_{sh} fit: rise {perr_nu1:.3f}, HLE {perr_nu2:.3f}')
  #print(f'nuFnu_{sh}: k2, k3 = {k2:.2f}, {k3:.2f}; s = {s:.2f}; d4 = {d4:.2f}, k4 = {k4:.2f}')
  #print(f'nuFnu_{sh}: k2 = {k2:.2f}; d4 = {d4:.2f}, k3 = {k4:.2f}')
  print(f'nuFnu_{sh}: k2 = {k2:.2f}, s = {s:.2f}; d4 = {d4:.2f}, k3 = {k4:.2f}')
  #print(f'Error in nuFnu_{sh} fit: rise {perr_nF1:.3f}, HLE {perr_nF2:.3f}')

  i_s = 2 if sh=='RS' else 0
  fac_nu = 1/env.fac_nu if sh=='FS' else 1.
  fac_nF = 1./(env.fac_F*env.fac_nu) if sh=='FS' else 1.
  ax_nu.semilogy(Tb[i_s:], nupk[i_s:]/env.nu0, c=col)
  ax_nu.semilogy(Tb, nu_fit*fac_nu, ls=':', c='k')
  ax_nF.semilogy(Tb, nuFnupk/env.nu0F0, c=col)
  ax_nF.semilogy(Tb, nuFnu_fit*fac_nF, ls=':', c='k')

  # ax_nu.loglog(tT[i_s:], nupk[i_s:]/env.nu0, c=col)
  # ax_nu.loglog(tT, nu_fit*fac_nu, ls=':', c='k')
  # ax_nF.loglog(tT, nuFnupk/env.nu0F0, c=col)
  # ax_nF.loglog(tT, nuFnu_fit*fac_nF, ls=':', c='k')

plt.show()