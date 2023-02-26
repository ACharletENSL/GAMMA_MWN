# -*- coding: utf-8 -*-
# @Author: acharlet

'''
This file generates initial conditions for a nebula expanding in a supernova ejecta
'''

# Imports
# --------------------------------------------------------------------------------------------------
from models_MWN import *
from solve_shocked_layers import get_shockedLayer

# Parameters
# to add: consistent reading of parameters 
# --------------------------------------------------------------------------------------------------
lfacwind = 100                  # wind Lorentz factor
theta = 1.e-3                   # wind relativistic temperature at injection
wc = 0.8                        # 0<wc<1, limit between internal and external shells
r_0 = 1e18                      # scaling length for CSM in case k != 0
k = 0                           # CSM density index
t = 1e3                         # SN age at sim start

# grid limits. Will be obtained from grid constructor later
rmin0 = 1e9
rmax0 = 1e12

# intermediate values
# --------------------------------------------------------------------------------------------------


# Init condition generator
# --------------------------------------------------------------------------------------------------
# r is in cm, (rho, v, p) in CGS units
def init_conds_noShocks(r, t):
    R_b = R_1st(t)                  # nebula radius
    R_c = vt*t                      # ejecta core radius
    R_e = R_c/wc                    # ejecta envelope radius

    r_w = r[r<=R_b]
    rho_w, v_w, p_w = get_wind_vars(r_w, t)
    r_ej = r[(r>R_b) & (r<=R_e)]
    rho_ej, v_ej, p_ej = get_ejecta_vars(r_ej, t)
    try:
        p_ej *= min((p_w[-1]/p_ej[0]), 1.) # put ejecta pressure at the wind pressure
    except IndexError:
        p_ej *= 1.
    r_csm = r[r>R_e]
    rho_csm, v_csm, p_csm = get_CSM_vars(r_csm, t)

    rho = np.concatenate((rho_w, rho_ej, rho_csm))
    v = np.concatenate((v_w, v_ej, v_ext))
    p = np.concatenate((p_w, p_ej, p_ext))

    return rho, v, p

def init_conds_withShocks(r, t):
    R_b = R_1st(t)                  # nebula radius
    R_c = vt*t                      # ejecta core radius
    print(f"Initial conditions:\nNebula radius {R_b:.3e} cm, ejecta core radius {R_c:.3e} cm.\n")
    gma = 5./3.

    # wind
    r_w = r[r<R_b]
    rho_w, v_w, p_w = get_wind_vars(r_w, t)

    # ejecta shell in front of the PWN
    sol_thinshell = get_shockedLayer('MWN_out')
    R_s = R_b/sol_thinshell.t[-1]
    print(f"Shell width {(R_s-R_b):.3e} cm")
    r_sh = r[(r>=R_b) & (r<=R_s)]
    l_sh = len(r_sh)
    rho_sh = np.array([])
    v_sh = np.array([])
    p_sh = np.array([])
    if l_sh:
        print(f"Shell initialized on {l_sh} cells\n")
        eta = r_sh/R_b
        O, U, C = sol_thinshell.sol(eta)
        rho_sh = D * vt**delta * r_sh**(-delta) * t**(delta-3) * O
        v_sh = (r_sh/t) * U
        cs_sh =  (r_sh/t) * C
        p_sh = np.sqrt(rho_sh/gma)*cs_sh
    else:
        print("No shell initialized\n")

    # solve profile at ejecta - CSM interaction to get RS position
    # get ratio of self-sim pressures to obtain CD scaling
    sol_inner = get_shockedLayer('ext_in')
    rscd = 1./sol_inner.t[-1]           # R_rs/R_cd 
    O, U, C = sol_inner.y
    Pr_in = (O[-1] * C[-1]**2) / (O[0] * C[0]**2) 

    sol_outer = get_shockedLayer('ext_out')
    fscd = 1./sol_outer.t[-1]      # R_fs/R_cd
    O, U, C = sol_outer.y
    Pr_out = (O[-1] * C[-1]**2)/(O[0] * C[0]**2)

    A = (Pr_in/Pr_out) * ((3-k)/(omega-3))**2
    a = (omega - 3)/(omega - k)
    R_e = (A * D *vt**omega / (mp_ * r_0**k))**(1/(omega-k)) * t**a
    R_rs = rscd * R_e
    R_fs = fscd * R_e

    print(f"Solving shock profile finds ejecta boundary at {R_e:.3e} cm")
    print(f"The reverse shock is at {R_rs:.3e} cm and the forward shock at {R_fs:.3e}")    
    
    # SNR ejecta
    r_ej = r[(r>R_s) & (r<R_rs)]
    rho_ej, v_ej, p_ej = get_ejecta_vars(r_ej, t)
    try:
        p_ej *= min((p_w[-1]/p_ej[0]), 1.) # put ejecta pressure at the wind pressure
    except IndexError:
        p_ej *= 1.
    # CSM
    r_csm = r[r>R_fs]
    rho_csm, v_csm, p_csm = get_CSM_vars(r_csm, t)

    # shocked ejecta at CSM interaction
    rho_in = np.array([])
    v_in = np.array([])
    p_in = np.array([])
    rho_ext = np.array([])
    v_ext = np.array([])
    p_ext = np.array([])

    r_in = r[(r>=R_rs) & (r<=R_e)]
    if len(r_in):
        eta = r_in/R_e
        O, U, C = sol_inner.sol(eta)
        rho_in = D * vt**omega * r_in**(-omega) * t**(omega-3) * O
        v_in = (r_in/t) * U
        cs_in = (r_in/t) * C
        p_in = np.sqrt(rho_in/gma)*cs_in

        # shocked CSM
        r_ext = r[(r>=R_e) & (r<=R_fs)]
        if len(r_ext):
            eta = r_ext/R_e
            O, U, C = sol_outer.sol(eta)
            rho_ext = rho_csm[0] * r_ext**-k * O
            v_ext = (r_ext/t) * U
            cs_ext = (r_ext/t) * C
            p_ext = np.sqrt(rho_ext/gma)*cs_ext

    rho = np.concatenate((rho_w, rho_sh, rho_ej, rho_in, rho_ext, rho_csm))
    v = np.concatenate((v_w, v_sh, v_ej, v_in, v_ext, v_csm))
    p = np.concatenate((p_w, p_sh, p_ej, p_in, p_ext, p_csm))

    return rho, v, p

# generator functions for each zone
# CSM
def get_CSM_vars(r, t):
    ''' Generates density, velocity and pressure profile of the CSM'''
    try:
        size = len(r)
        rho = np.ones(size)
        v = np.zeros(size)
        p = np.ones(size)
    except TypeError:
        rho = 1.
        v = 0.
        p = 1.

    rho *= mp_ * (r_0/r)**k
    p *= rho * c_**2 * 1e-6

    return rho, v, p

# ejecta
def get_ejecta_vars(r, t):
    ''' Generates density, velocity and pressure profile of the ejecta'''
    R_c = vt*t                      # ejecta core radius
    R_e = R_c/wc                    # ejecta envelope radius
    try:
        size = len(r)
        rho = np.ones(size)
        v = np.ones(size)
        p = np.ones(size)
        rho = np.where(r <= R_c, D * (vt/r)**delta / t**(3-delta), D * (vt/r)**omega / t**(3-omega))
    except TypeError:
        rho = 1.
        v = 1.
        p = 1.
        if r <= R_c:
            rho *= D * (vt/r)**delta / t**(3-delta)
        else:
            rho *= D * (vt/r)**omega / t**(3-omega)
    
    v *= r/t
    p *= rho * c_**2 * 1e-6

    return rho, v, p

# wind
def get_wind_vars(r, t, theta=1e-6):
    ''' Generates density, velocity and pressure profile of the (relat. cold) wind
    Profile for any relativistic temperature to be added'''
    gma = 5./3.

    try:
        size = len(r)
        rho = np.ones(size)
        v = np.ones(size)
        p = np.ones(size)
    except TypeError:
        rho = 1.
        v = 1.
        p = 1.

    rho = L0/(4. * pi_ * r**2 * lfacwind**2 * c_**3)
    v *= c_
    p *= rho * theta * c_**2 * (rmin0/r)**(2*(gma-1.))

    return rho, v, p



    