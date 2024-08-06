from thinshell_analysis import *

key = 'sph_fid'
func, name = 'Band', 'RS'
nuobs, Tobs, env = get_radEnv(key)
Tb = (Tobs-env.Ts)/env.T0
nuFnus = nuFnu_thinshell_run(key, nuobs, Tobs, env, func)

nuFnu = nuFnus[0]
nupks = np.zeros(nuFnu.shape[0])
for i in range(nuFnu.shape[0]):
  ipk = np.argmax(nuFnu[i])
  nupks[i] = nuobs[ipk]
g, Tth, dRR = env.gRS, env.T0, env.dRRS
tT = 1. + Tb
tTf = 2.52
def fitfunc(x, d1, d2):
  return approx_nupk_hybrid(x, tTfi=tTf, gi=g, d1=d1, d2=d2)

popt, pcov = curve_fit(fitfunc, tT[2:], nupks[2:], p0=[1.34, 1.34])
print(popt)

fig, ax = plt.subplots()
ax.semilogy(Tb, nupks/env.nu0, c='r', ls='-')
ax.semilogy(Tb, fitfunc(tT, *popt), c='k', ls=':')