# -*- coding: utf-8 -*-
# @Author: acharlet

'''
This file solves for the shocked layer profiles
'''

# Imports
# --------------------------------------------------------------------------------------------------
import numpy as np
from scipy.integrate import solve_ivp
from environment import delta, omega, k

# Get shocked profile in CGS units
# --------------------------------------------------------------------------------------------------
def get_shockedLayer(layer):
    '''
    Return the shocked profile in physical variables
    D is the density scaling
    layer argument must be "MWN_out", "ext_in", "ext_out" 
    for the shell in front of nebula, shocked SNR and CSM at their interaction respectively
    '''

    if layer == "MWN_out":
        a = (6-delta)/(5-delta)
        sol = selfSimilar_shell(a, delta)
    
    elif layer == "ext_in":
        a = (omega - 3)/(omega - k)
        sol = selfSimilar(a, omega, 3-omega, True, True)
    
    elif layer == "ext_out":
        a = (omega - 3)/(omega - k)
        sol = selfSimilar(a, k, 0, False, False)
    
    else:
        print("Wrong 'layer' argument. Please enter 'MWN_out', 'ext_in', or 'ext_out'")
        exit(1)

    return sol

# Solve self-similar profile in dimensionless units
# --------------------------------------------------------------------------------------------------
def selfSimilar(a, alpha, beta, vout:bool, etac_is_gtr:bool):
    '''
    Solve self-similar profile for a shocked, spherically symmetric, non-relativistic flow
    a defined for R_cd in t^a, alpha and beta defined for rho = D r^-alpha t^-beta
    vout = True when the upstream flow has a nonzero velocity (= is the SNR ejecta)
    etac_is_gtr = True if etac = R_cd/R_s > 1 (if we solve from RS to CD)
    '''
    gma = 5./3.
    def selfSim_ODE(eta:float, y: np.ndarray) -> np.ndarray:
        """
        Returns the ode value at (eta, y)
        """
        O, U, C = y
        repeat = eta * (U/a - 1.)
        lhs_matrix = np.zeros((3,3))

        lhs_matrix[0,0] = repeat/O
        lhs_matrix[0,1] = eta / a
        lhs_matrix[1,0] = eta * C**2 / (a * gma * O)
        lhs_matrix[1,1] = repeat
        lhs_matrix[1,2] = 2 * eta * C / (a * gma)
        lhs_matrix[2,0] = repeat * (1-gma) / O
        lhs_matrix[2,2] = 2 * repeat / C

        rhs_matrix = np.array([(alpha-3)*U + beta,
            (alpha-2)*C**2/gma + U*(1-U),
            2*(1-U) + (1-gma)*(alpha*U+beta)])
        
        return np.linalg.solve(lhs_matrix, rhs_matrix)
    
    def jumpcond(vout):
        O = 4.
        U = (3.*a + 1)/4. if vout else 0.75*a
        C = np.sqrt(5) * np.abs(a-1.) / 4. if vout else np.sqrt(5) * a
        return np.array([O, U, C])
    
    y0 = jumpcond(vout)
    
    def u_boundary(eta, y):
        return y[1] - a
    u_boundary.terminal = True
    
    eta_max = 10. if etac_is_gtr else 0.5

    sol = solve_ivp(selfSim_ODE, (1, eta_max), y0, dense_output=True, events=u_boundary, atol=1e-3)
    return sol

def selfSimilar_shell(a, delta):
    '''
    Solve self-similar profile for a shocked, spherically symmetric, non-relativistic flow
    describes the shocked shell of SNR ejecta around the nebula
    a defined for R_cd in t^a, delta the density index rho = D r^-delta t^(delta-3)
    '''
    gma = 5./3.
    beta = a*delta - delta + 3        # -'delta' in Suzuki & Maeda 17
    def selfSim_ODE(eta:float, y: np.ndarray) -> np.ndarray:
        """
        Returns the ode value at (eta, y)
        """
        O, U, C = y
        repeat = eta * a * (U - 1.)
        lhs_matrix = np.zeros((3,3))

        lhs_matrix[0,0] = repeat/O
        lhs_matrix[0,1] = eta * a
        lhs_matrix[1,0] = eta * C**2 * a / (gma * O)
        lhs_matrix[1,1] = repeat
        lhs_matrix[1,2] = 2 * eta * C * a / gma
        lhs_matrix[2,0] = repeat * (1-gma) / O
        lhs_matrix[2,2] = 2 * repeat / C

        rhs_matrix = np.array([beta - 3.*a*U,
            (1.-a*U)*U - 2.*a*C**2/gma,
            2*(1-a*U) + (1-gma)*beta])
        
        return np.linalg.solve(lhs_matrix, rhs_matrix)
    
    y0 = np.array([
        4.,
        (3.*a + 1)/4.,
        np.sqrt(5) * (a-1.) / 4.
    ])
    
    def u_boundary(eta, y):
        return y[1] - a
    u_boundary.terminal = True

    # solve from FS to CD
    sol = solve_ivp(selfSim_ODE, (1, 0.5), y0, dense_output=True, events=u_boundary, atol=1e-3)
    return sol

