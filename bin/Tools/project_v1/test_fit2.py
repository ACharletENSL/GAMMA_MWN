from thinshell_analysis import *
import sys

key = sys.argv[1] if len(sys.argv) > 1 else 'sph_fid'
func = 'Band'
withlog, withReff = True, True
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

fig2, axs = plt.subplots(2, 1, sharex='all')
ax_xi, ax_err = axs
ax_xi.set_title(f'log fitting {'ON' if withlog else 'OFF'}, effective final radius {'ON' if withReff else 'OFF'}')
ax_xi.set_ylabel('$\\xi_{eff}$')
ax_err.set_ylabel('err.')
ax_err.set_xlabel('$\\bar{T}_{sh}$')
dummy_lst = [plt.plot([], [], ls=l, c='k')[0] for l in ['-.', '--']]
ax_err.legend(dummy_lst, ['$\\nu_{pk}$', '$\\nu_{pk} F_{\\nu_{pk}}$'])

inputs = [shocks, nupks, nuFnupks, g_vals, m_vals, n_vals, norms_T, norms_nu, norms_nuF, cols]
for sh, nupk, nuFnupk, g, m, n, T0, norm_nu, norm_nuF, col in zip(*inputs):
  
  ad = -(n+3.*m/2) #a+d
  df = pd.read_csv(path+sh+'.csv')
  row = df.iloc[-2]
  tf, rf, Tej, Tthf = row[['time', f'r_{{{sh}}}', f'Tej_{{{sh}}}', f'Tth_{{{sh}}}']]
  Tonf = Tej + Tthf
  tT = 1 + (Tobs - env.Ts)/T0
  def tT_to_tTth(tT, Tth):
    '''
    Go from normalized by T_r(R0) to normalized by T_theta
    '''
    return 1 + (tT - 1) * T0/Tth


  tTf = 1 + (Tonf - env.Ts)/T0
  i_f = find_closest(tT, tTf)
  fac_nu = 1/env.fac_nu if sh=='FS' else 1.
  fac_nF = 1./(env.fac_F*env.fac_nu) if sh=='FS' else 1.
  #iff = -211 if ((sh=='FS') and (key=='sph_big')) else -1
  iff = -1
  i_s = 5 # flawed data from numerics at beginning
  nu = nupk/norm_nu
  nuFnu = nuFnupk/norm_nuF

  def tT_to_ximax(tT):
    return (g**2/(m+1))*(tT-1)

  def xi_eff(tT, xi_sat, k, s):
    ximax = tT_to_ximax(tT)
    return k*(ximax**-s + xi_sat**-s)**(-1/s)
  
  def fit_nupk_rise(tT, d, xi_sat, k, s, knu):
    xi = xi_eff(tT, xi_sat, k, s)
    return knu * tT**d * (1+xi*(m+1)/g**2)**(-d) / (1+xi)
  
  def fit_lognupk_rise(tT, d, xi_sat, k, s, knu):
    xi = xi_eff(tT, xi_sat, k, s)
    return np.log10(knu) + d*np.log10(tT / (1+xi*(m+1)/g**2)) - np.log10(1+xi)
  
  def fit_nuFnupk_rise(tT, xi_sat, k, s, kF):
    ximax = tT_to_ximax(tT)
    xi = xi_eff(tT, xi_sat, k, s)
    return kF * ximax * (1+xi)**-4 * tT**ad * (1+xi*(m+1)/g**2)**((m-ad)/(m+1))
  
  def fit_lognuFnupk_rise(tT, xi_sat, k, s, kF):
    ximax = tT_to_ximax(tT)
    xi = xi_eff(tT, xi_sat, k, s)
    return np.log10(kF) + np.log10(ximax) - 4*np.log10(1+xi) + ad*np.log10(tT) + ((m-ad)/(m+1))*np.log10(1+xi*(m+1)/g**2)
  
  # to fit both functions with the same underlying physical parameters
  def fit_both_rise(tT, d, xi_sat, k, s, knu, kF):
    nufit = fit_nupk_rise(tT, d, xi_sat, k, s, knu)
    nuFnufit = fit_nuFnupk_rise(tT, xi_sat, k, s, kF)
    fit = np.hstack((nufit, nuFnufit))
    return fit
  
  def fit_logboth_rise(tT, d, xi_sat, k, s, knu, kF):
    nufit = fit_lognupk_rise(tT, d, xi_sat, k, s, knu)
    nuFnufit = fit_lognuFnupk_rise(tT, xi_sat, k, s, kF)
    fit = np.hstack((nufit, nuFnufit))
    return fit
  

  if withlog:
    nu = np.log10(nu)
    nuFnu = np.log10(nuFnu)
    def fit_nupk_rise(tT, d, xi_sat, k, s, knu):
      return fit_lognupk_rise(tT, d, xi_sat, k, s, knu)
    def fit_nuFnupk_rise(tT, xi_sat, k, s, kF):
      return fit_lognuFnupk_rise(tT, xi_sat, k, s, kF)
    def fit_both_rise(tT, d, xi_sat, k, s, knu, kF):
      return fit_logboth_rise(tT, d, xi_sat, k, s, knu, kF)


  tT_rise, tT_HLE = np.split(tT, [i_f])
  tT_rise = tT_rise[i_s:]
  nu_rise, nu_HLE = np.split(nu, [i_f])
  nu_rise = nu_rise[i_s:]
  nuFnu_rise, nuFnu_HLE = np.split(nuFnu, [i_f])
  nuFnu_rise = nuFnu_rise[i_s:]
  ydata_rise = np.hstack((nu_rise, nuFnu_rise))
  ydata_HLE = np.hstack((nu_HLE, nuFnu_HLE))

  sigma = np.ones(tT_rise.shape)
  #i_hl = int(i_f/10.)
  sigma[iff] = 0.01
  sigma = np.hstack((sigma, sigma))
  # params order: d, xi_sat, k, s, knu, kF
  popt1, pcov1 = curve_fit(fit_both_rise, tT_rise, ydata_rise,
      p0=[-2, 1, 0.3, 3, 1, 1], bounds=([-5, 0.5, 0, 1, 0.1, 0.1], [0, 10, 1, 10, 10, 10]), sigma=sigma)
  d, xi_sat, k, s, knu, kF = popt1
  title += f'{sh}: d = {d:.2f}, $\\xi_{{sat}}$ = {xi_sat:.2f}, k = {k:.2f}, s = {s:.2f}, $k_\\nu$ = {knu:.2f}, $k_F$ = {kF:.2f}'
  fit_xi =  xi_eff(tT[:i_f], xi_sat, k, s)
  fit_nu_rise = fit_nupk_rise(tT[:i_f], d, xi_sat, k, s, knu)
  fit_nuFnu_rise = fit_nuFnupk_rise(tT[:i_f], xi_sat, k, s, kF)

  Tb_r = tT[:i_f] - 1
  ax_xi.loglog(Tb_r, fit_xi, c=col)

  # fit the HLE now
  Tth = Tthf
  if withReff:
    yf = (1 + (m+1)*g**-2 *fit_xi[-1])**(-1/(m+1))
    Tth = Tthf * yf**(m+1)
  def fit_nupk_HLE(tT, dnu):
    nupkf = fit_nupk_rise(tTf, d, xi_sat, k, s, knu)
    tTth = tT_to_tTth(tT, Tth)
    return nupkf * (tTth/tTth[0])**dnu
  def fit_nuFnupk_HLE(tT, dnu, dF):
    nuFnupkf = fit_nuFnupk_rise(tTf, xi_sat, k, s, kF)
    tTth = tT_to_tTth(tT, Tth)
    d = dnu+dF
    return nuFnupkf * (tTth/tTth[0])**d
  def fit_both_HLE(tT, dnu, dF):
    nufit = fit_nupk_HLE(tT, dnu)
    nuFnufit = fit_nuFnupk_HLE(tT, dnu, dF)
    fit = np.hstack((nufit, nuFnufit))
    return fit
  
  def fit_lognupk_HLE(tT, dnu):
    nupkf = fit_lognupk_rise(tTf, d, xi_sat, k, s, knu)
    tTth = np.log10(tT_to_tTth(tT, Tth))
    return nupkf + (tTth - tTth[0])*dnu
  def fit_lognuFnupk_HLE(tT, dnu, dF):
    nuFnupkf = fit_lognuFnupk_rise(tTf, xi_sat, k, s, kF)
    tTth = np.log10(tT_to_tTth(tT, Tth))
    d = dnu+dF
    return nuFnupkf + (tTth - tTth[0])*d
  def fit_logboth_HLE(tT, dnu, dF):
    nufit = fit_lognupk_HLE(tT, dnu)
    nuFnufit = fit_lognuFnupk_HLE(tT, dnu, dF)
    fit = np.hstack((nufit, nuFnufit))
    return fit
  if withlog:
    def fit_nupk_HLE(tT, dnu):
      return fit_lognupk_HLE(tT, dnu)
    def fit_nuFnupk_HLE(tT, dnu, dF):
      return fit_lognuFnupk_HLE(tT, dnu, dF)
    def fit_both_HLE(tT, dnu, dF):
      return fit_logboth_HLE(tT, dnu, dF)

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
  nu_err_f = np.abs(fit_nu[i_s:] - nu[i_s:])/nu[i_s:]
  nu_err = nu_err_f.max()
  nuFnu_err_f = np.abs(fit_nuFnu[i_s:] - nuFnu[i_s:])/nuFnu[i_s:]
  nuFnu_err = nuFnu_err_f.max()
  if withlog:
    nu_err_f = np.abs(10**fit_nu[i_s:] - 10**nu[i_s:])/10**nu[i_s:]
    nu_err = nu_err_f.max()
    nuFnu_err_f = np.abs(10**fit_nuFnu[i_s:] - 10**nuFnu[i_s:])/10**nuFnu[i_s:]
    nuFnu_err = nuFnu_err_f.max()
  print(f'{sh} rise: d = {d:.2f}, xi_sat = {xi_sat:.2f}, k = {k:.2f}, s = {s:.2f}, k_nu={knu:.2f}, k_F = {kF:.2f}')
  print(f'{sh} HLE: d_nu = {dnu:.2f}, d_F = {dF:.2f}')
  print(f'{sh} maximum error on nu {nu_err*100:.2f} and nuFnu {nuFnu_err*100:.2f}')

  Tb = tT - 1
  ax_nu.semilogx(Tb, nu, c=col)
  ax_nu.semilogx(Tb, fit_nu, ls=':', c='k')
  ax_nF.semilogx(Tb, nuFnu, c=col)
  ax_nF.semilogx(Tb, fit_nuFnu, ls=':', c='k')
  ax_nu.axvline(Tb[i_f], c=col, ls='--', lw=.8)
  ax_nF.axvline(Tb[i_f], c=col, ls='--', lw=.8)
  if not withlog:
    ax_nu.set_yscale('log')
    ax_nF.set_yscale('log')

  ax_err.semilogx(Tb[i_s:], nu_err_f, ls='-.', c=col)
  ax_err.semilogx(Tb[i_s:], nuFnu_err_f, ls='--', c=col)

fig.suptitle(title[:-2])
plt.show()