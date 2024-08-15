from thinshell_analysis import *

key = 'sph_fid'
func = 'Band'
nuobs, Tobs, env = get_radEnv(key)
path = get_contribspath(key)
nupks, nuFnupks = get_pksnunFnu(key)

# objectif : fit courbes nupk(T), nupkFnupk(T)
# step 1: trouver le break entre les deux parties
# step 2: faire fit
#mRS = -2*0.09103
mRS = -2*0.13176423165558485
mFS = 0.02
fig = plt.figure()
gs = fig.add_gridspec(2, 1, hspace=0)
axs = gs.subplots(sharex='all')
for i, sh, istart, T0, m, g, fac_nu, fac_F, col in zip(range(2), ['RS', 'FS'], [2, 0],
    [env.T0, env.T0FS], [mRS, mFS], [env.gRS, env.gFS], [1., 1./env.fac_nu], [1., env.fac_F], ['r', 'b']):
  df = pd.read_csv(path+sh+'.csv')
  row = df.iloc[-2]
  t, r, Tej, Tth = row[['time', f'r_{{{sh}}}', f'Tej_{{{sh}}}', f'Tth_{{{sh}}}']]
  Tonf = Tej + Tth
  tT = 1 + (Tobs - env.Ts)/T0
  tTf = 1 + (Tonf - env.Ts)/T0
  i_f = find_closest(tT, tTf)

  # additional points to force fit
  ifits = []
  # for r in [8, 6, 4]:
  #   tTfit = 1 + (tTf-1)/r
  #   ifits.append(find_closest(tT, tTfit))
  ifits += [i_f-1, i_f, -1]
  
  def get_tTeff1(tT, k):
    return 1. + np.abs(k*(m+1+g**2)/(g**2*(g**2-1))) * (tT - 1.)
  def get_tTeff2(tT, k):
    return 1. + np.abs(k*(m+1+g**2)/(g**2*(g**2-1)))  * (r*c_/env.R0)**-(m+1) * (tT-1.)

  sigma = np.ones(tT.shape)
  sigma[ifits] = 0.01
  # fit nupk
  def fitfunc_nu_inf(tT, d, k):
    tTeff1 = get_tTeff1(tT, k)
    return fac_nu * tTeff1**-d
  popt, pcov = curve_fit(fitfunc_nu_inf, tT[istart:i_f], nupks[i][istart:i_f]/env.nu0, p0=[1., 1.], sigma=sigma[istart:i_f])
  n1, k1 = popt
  nupks_fit_inf = fitfunc_nu_inf(tT[:i_f], n1, k1)

  def fitfunc_nu_sup(tT, d, k):
    tTeff2 = get_tTeff2(tT, k)
    return fitfunc_nu_inf(tTf, n1, k1) * (tTeff2/tTeff2[0])**-d
  popt, pcov = curve_fit(fitfunc_nu_sup, tT[i_f:], nupks[i][i_f:]/env.nu0, p0=[1., 1.], sigma=sigma[i_f:])
  n2, k2 = popt
  nupks_fit_sup = fitfunc_nu_sup(tT[i_f:], n2, k2)

  print(f'nu_{sh} rise: {k1:.2f}, {n1:.2f}, HLE: {k2:.2f}, {n2:.2f}')
  nupks_fit = np.concatenate((nupks_fit_inf, nupks_fit_sup), axis=None)

  # fit nupk Fnupk
  def fitfunc_nuFnupk_inf(tT, d, k):
    tTeff1 = get_tTeff1(tT, k)
    return 1 - tTeff1**-d
  popt, pcov = curve_fit(fitfunc_nuFnupk_inf, tT[:i_f], nuFnupks[i][:i_f]/env.nu0F0, p0=[0.5, 1.], sigma=sigma[:i_f])
  p1, k1 = popt
  nuFnupks_fit_inf = fitfunc_nuFnupk_inf(tT[:i_f], p1, k1)

  def fitfunc_nuFnupk_sup(tT, d, k):
    tTeff1f = get_tTeff1(tTf, k1)
    nuFnupkf = (1 - tTeff1f**-p1)
    tTeff2 = get_tTeff2(tT, k)
    return nuFnupkf * (tTeff2/tTeff2[0])**-d
  popt, pcov = curve_fit(fitfunc_nuFnupk_sup, tT[i_f:], nuFnupks[i][i_f:]/env.nu0F0, p0=[3., 1.], sigma=sigma[i_f:])
  p2, k2 = popt
  nuFnupks_fit_sup = fitfunc_nuFnupk_sup(tT[i_f:], p2, k2)

  print(f'nuFnu_{sh} rise: {k1:.2f}, {p1:.2f}, HLE: {k2:.2f}, {p2:.2f}')
  nuFnupks_fit = np.concatenate((nuFnupks_fit_inf, nuFnupks_fit_sup), axis=None)

  axs[0].loglog(tT[istart:], nupks[i][istart:]/env.nu0, c=col)
  axs[0].loglog(tT, nupks_fit, c='k', ls=':')
  axs[0].axvline(tTf, c=col, ls=':')
  axs[1].loglog(tT, nuFnupks[i]/env.nu0F0, c=col)
  axs[1].loglog(tT, nuFnupks_fit, c='k', ls=':')
  axs[1].axvline(tTf, c=col, ls=':')

axs[0].set_ylabel('$\\nu_{pk}/\\nu_0$')
axs[1].set_xlabel('$\\tilde{T}_{sh}$')
axs[1].set_ylabel('$\\nu_{pk}F_{\\nu_{pk}}/\\nu_0F_0$')
plt.tight_layout()

