# -*- coding: utf-8 -*-
# @Author: acharlet

'''
This file contains physical constants in CGS units, read parameters and derives useful quantities
'''
from pathlib import Path
import os
from numpy import pi

# Constants
c_ = 2.99792458e10
mp_ = 1.67262192e-24
Msun_ = 1.9891e33
pi_ = pi

# Standard values
R_ns = 1.e6                 # NS radius (cm)
I = 1.e45                   # NS inertia moment (g cm^2)
B_0 = 1e14                  # Initial surface magnetic field (G)
theta_B = 0.                # inclination angle between rotation and magnetic dipole (radn 1.0472=pi/3)
sp_law = 'force-free'       # spin-down law. 'force-free' or 'vacuum'
n = 3                       # braking index
m = (n+1)/(n-1)             # spindown luminosity index
M_ej = 3. * Msun_           # ejecta mass (g)
E_sn = 1.e51                # SNR energy (erg)
delta = 0                   # ejecta core density index
omega = 10                  # ejecta envelope density index
wc = 0.8                    # ratio SNR core / SNR envelope (for non-shock initializations)
rho_0 = 1. * mp_            # external density
R_0 = 1.e18                 # external density scaling length (for non-constant CSM)        

# parameters, to be initialized by reading phys_input.MWN
P_0 = None                  # initial rotation period (s)
E_0 = None                  # total energy injected in the nebula (erg)
lfacwind = None             # wind Lorentz factor

P_0, E_0, lfacwind, t_start = readParams()

# intermediate values
f = get_spindown_f(theta_B, sp_law)
t_0 = I * c_**3 * P_0**2 / (8. * pi**2 * f * R_ns**6 * B_0**2)
#L_0 = f * B_0**2 * R_ns**6 * (2.*pi)**4 / (c_**3 * P_0**4)
L_0 = 2. * E_0 / ((n-1.)*t_0)

fac1 = (5-delta)/(11-2*delta)
v_t = np.sqrt(2*(5-delta)*(omega-5) / ((3-delta)*(omega-3))) * np.sqrt(E_sn/M_ej)
D = (5-delta)*(omega-5)/(2*pi*(omega-delta)) * E_sn / v_t**5

# crossing time and radius from Blondin et al 2001 (see Granot 2017 appendix B)
t_c = t_0 * 2.64 * ((omega - 5)/omega) * ((n-1)/2)
R_c = v_t * t_c

params_path = Path().absolute().parents[1] / 'phys_input.MWN'

def readParams(path=params_path):
    '''Reads GAMMA/phys_input.MWN file, returns P_0'''
    vars = ['P_0', 'E_0', 'lfacwind', 't_start']
    P_0 = 0.
    E_0 = 0.
    lfacwind = 0.
    if path.exists():
        with path.open('r') as f:
            lines = f.read().splitlines()
            for line in lines:
                l = line.split()
                if l[0] == vars[0]:
                    P_0 = float(l[2]) * 1e-3
                elif l[0] == vars[1]:
                    E_0 = float(l[2]) * E_sn
                elif l[0] == vars[2]:
                    lfacwind = float(l[2])
                elif l[0] == vars[3]:
                    t_start = float(l[3]) * t_0
    else:
        print('Parameter file not found')
        exit(2)
    
    return P_0, E_0, lfacwind
    

def get_spindown_f(theta_B, sp_law):
    '''Derive spindown parameter f'''
    if sp_law=='vacuum':
        f = (2./3.)*np.sin(theta_B)**2
    elif sp_law=='force-free':
        f = 1. + np.sin(theta_B)**2
    else:
        print('Wrong spindown law input, must be vacuum or force-free\nExiting...')
        exit(1)
    return f
