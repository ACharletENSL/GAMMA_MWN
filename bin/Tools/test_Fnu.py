# -*- coding: utf-8 -*-
# @Author: acharlet


from plotting_scripts_new import *
from thinshell_analysis import *
plt.ion()

key, func = 'cart_fid', 'Band'
lognu, logT = -1, 0
nuobs, Tobs, env = get_radEnv(key)
Tb = (Tobs-env.Ts)/env.T0
nub = nuobs/env.nu0
TbFS = env.fac_T * Tb
nubFS = nub * env.fac_nu
gRS, gFS = env.gRS, env.gFS
nu_ = np.array([10**lognu])
nu_FS = nu_ #* env.fac_nu
#i_nu = find_closest(nuobs, nu)
T_ = np.array([10**logT])
T_FS = env.fac_T * T_
#i_T = find_closest(Tobs, T)


lc_RS1 = nu_[:, np.newaxis] * flux_shockfront_planar(nu_, Tb, gRS, env.dRRS/env.R0, func=func)
lc_FS1 = nu_FS[:, np.newaxis] * env.fac_F**-1 * flux_shockfront_planar(nu_FS, TbFS, gFS, env.dRFS/env.R0, func=func)
lc_RS2 = nu_[:, np.newaxis] * flux_shockfront_planar_yInt(nu_, Tb, gRS, env.dRRS/env.R0, func=func)
lc_FS2 = nu_FS[:, np.newaxis] * env.fac_F**-1 * flux_shockfront_planar_yInt(nu_FS, TbFS, gFS, env.dRFS/env.R0, func=func)
lc_RS3 = nu_[:, np.newaxis] * flux_shockfront_general(nu_, Tb, gRS, env.dRRS/env.R0, a=0, d=0, m=0, func=func)
lc_FS3 = nu_FS[:, np.newaxis] * env.fac_F**-1 * flux_shockfront_general(nu_FS, TbFS, gFS, env.dRFS/env.R0, a=0, d=0, m=0, func=func)
lc_arr = [[lc_RS1, lc_RS2, lc_RS3], [lc_FS1, lc_FS2, lc_FS3]]

sp_RS1 = nub[:, np.newaxis] * flux_shockfront_planar(nub, T_, gRS, env.dRRS/env.R0, func=func)
sp_FS1 = nubFS[:, np.newaxis] * env.fac_F**-1 * flux_shockfront_planar(nubFS, T_FS, gFS, env.dRFS/env.R0, func=func)
sp_RS2 = nub[:, np.newaxis] * flux_shockfront_planar_yInt(nub, T_, gRS, env.dRRS/env.R0, func=func)
sp_FS2 = nubFS[:, np.newaxis] * env.fac_F**-1 * flux_shockfront_planar_yInt(nubFS, T_FS, gFS, env.dRFS/env.R0, func=func)
sp_RS3 = nub[:, np.newaxis] * flux_shockfront_general(nub, T_, gRS, env.dRRS/env.R0, a=0, d=0, m=0, func=func)
sp_FS3 = nubFS[:, np.newaxis] * env.fac_F**-1 * flux_shockfront_general(nubFS, T_FS, gFS, env.dRFS/env.R0, a=0, d=0, m=0, func=func)

sp_arr = [[sp_RS1, sp_RS2, sp_RS3], [sp_FS1, sp_FS2, sp_FS3]]

fig = plt.figure()
gs = fig.add_gridspec(1, 2, wspace=0)
axs = gs.subplots()
styles = [[['-', 'r'], [':','k'], ['-.','k']], [['-', 'b'], [':','k'], ['-.','k']]]
for lc_sh, sp_sh, style in zip(lc_arr, sp_arr, styles):
  for lc, sp, sty in zip(lc_sh, sp_sh, style):
    lc = lc[0]
    sp = sp[:,0]
    lst, col = sty
    axs[0].plot(Tb, lc, c=col, ls=lst)
    axs[1].loglog(nub, sp, c=col, ls=lst)
axs[0].set_ylabel('$\\nu F_{\\nu}/\\nu_0F_0$')
axs[0].set_xlabel('$\\bar{T}$')
axs[0].set_title(f'$\\log_{{10}}\\tilde{{\\nu}}={lognu:.0f}$')
axs[1].set_xlabel('$\\tilde{\\nu}$')
axs[1].yaxis.tick_right()
axs[1].set_title(f'$\\bar{{T}}={10**logT:.1f}$')
fig.suptitle('planar')

lc_RS1 = nu_[:, np.newaxis] * flux_shockfront_hybrid(nu_, Tb, gRS, env.dRRS/env.R0, func=func)
lc_FS1 = nu_FS[:, np.newaxis] * env.fac_F**-1 * flux_shockfront_hybrid(nu_FS, TbFS, gFS, env.dRFS/env.R0, func=func)
lc_RS3 = nu_[:, np.newaxis] * flux_shockfront_general(nu_, Tb, gRS, env.dRRS/env.R0, a=1, d=-1, m=0, func=func)
lc_FS3 = nu_FS[:, np.newaxis] * env.fac_F**-1 * flux_shockfront_general(nu_FS, TbFS, gFS, env.dRFS/env.R0, a=1, d=-1, m=0, func=func)
lc_arr = [[lc_RS1, lc_RS3], [lc_FS1, lc_FS3]]

sp_RS1 = nub[:, np.newaxis] * flux_shockfront_hybrid(nub, T_, gRS, env.dRRS/env.R0, func=func)
sp_FS1 = nubFS[:, np.newaxis] * env.fac_F**-1 * flux_shockfront_hybrid(nubFS, T_FS, gFS, env.dRFS/env.R0, func=func)
sp_RS3 = nub[:, np.newaxis] * flux_shockfront_general(nub, T_, gRS, env.dRRS/env.R0, a=1, d=-1, m=0, func=func)
sp_FS3 = nubFS[:, np.newaxis] * env.fac_F**-1 * flux_shockfront_general(nubFS, T_FS, gFS, env.dRFS/env.R0, a=1, d=-1, m=0, func=func)
sp_arr = [[sp_RS1, sp_RS3], [sp_FS1, sp_FS3]]

fig = plt.figure()
gs = fig.add_gridspec(1, 2, wspace=0)
axs = gs.subplots()
styles = [[['-', 'r'], ['-.','k']], [['-', 'b'], ['-.','k']]]
for lc_sh, sp_sh, style in zip(lc_arr, sp_arr, styles):
  for lc, sp, sty in zip(lc_sh, sp_sh, style):
    lc = lc[0]
    sp = sp[:,0]
    lst, col = sty
    axs[0].plot(Tb, lc, c=col, ls=lst)
    axs[1].loglog(nub, sp, c=col, ls=lst)
axs[0].set_ylabel('$\\nu F_{\\nu}/\\nu_0F_0$')
axs[0].set_xlabel('$\\bar{T}$')
axs[0].set_title(f'$\\log_{{10}}\\tilde{{\\nu}}={lognu:.0f}$')
axs[1].set_xlabel('$\\tilde{\\nu}$')
axs[1].yaxis.tick_right()
axs[1].set_title(f'$\\bar{{T}}={10**logT:.1f}$')
fig.suptitle('hybrid')