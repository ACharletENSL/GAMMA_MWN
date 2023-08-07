# -*- coding: utf-8 -*-
# @Author: acharlet

'''
This file contains functions to derive all relevant quantities
and default parameters in the pulsar/magnetar wind nebula situation
'''

# Imports
from phys_constants import *

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
v_min = 0.                  # min SNR ejecta velocity
v_max = 0.7*c_              # max SNR ejecta velocity
delta = 0                   # ejecta core density index, must be < 3
omega = 10                  # ejecta envelope density index, must be > 5
k = 0                       # CSM density index
beta_ej = 0.7               # max ejecta velocity (units of c)
n_0=  1.                    # number density of the CSM
rho_csm = n_0 * mp_         # external density
R_0 = 1.e18                 # external density scaling length (for non-constant CSM)        


# functions for relevant physical quantities
def f_l(l, x, y):
  '''
  Extension of f_l function from Suzuki&Maeda 2017 with nonzero v_min
  Obtained by integrating density profile and equating to M_ej
  '''
  return (omega-l)*(l-delta)/(omega-delta-(omega-l)*x**(l-delta)-(l-delta)*y**(l-omega))

def vt_from_SNRparams(M_ej, E_sn, delta, omega, vmin, vmax):
  '''
  Solves for v_t (velocity in ejecta at break between density power laws)
  '''
  
  def F_vt(v):
    '''
    Function to solve for v_t
    Obtained by integrating kinetic energy and equating to E_sn
    '''
    return (2*E_sn/M_ej) * (f_l(5, vmin/v, vmax/v)/f_l(3, vmin/v, vmax/v)) - v**2

  vt_guess = np.sqrt(2*E_sn / M_ej)
  root = fsolve(F_vt, vt_guess)
  return root[0]

def get_spindown_f(theta_B, sp_law='force free'):
  '''Derive spindown parameter f'''
  if sp_law=='vacuum':
    f = (2./3.)*np.sin(theta_B)**2
  elif sp_law=='force-free':
    f = 1. + np.sin(theta_B)**2
  else:
    print('Wrong spindown law input, must be vacuum or force-free\nExiting...')
    exit(1)
  return f

def derive_SNR_densityscale(M_ej, v_min, v_max, v_t):
  '''
  Derives the density scaling for the unshocked SNR ejecta
  '''
  return f_l(3, v_min/v_t, v_max/v_t) * M_ej / (4.*pi_ * v_t**3)

def derive_spindown_fromP(P_0):
  '''
  Derives spindown time, power, and injected energy from the initial rotation period
  '''
  t_0 = I * c_**3 * P_0**2 / (8. * pi_**2 * f * R_ns**6 * B_0**2)
  L_0 = (f * B_0**2 * R_ns**6 * (2.*pi_/P_0)**4 / c_**3)
  E_0 = (n-1) * I * (2 * pi_ / P_0)**2 / 4.
  return t_0, L_0, E_0

def derive_spindown_fromE(E_0):
  '''
  Derives spindown time, power, and rotation period from the injected energy
  '''
  t_0 = (n-1) * I**2 * c_**3 / (8. * f * R_ns**6 * B_0**2 * E_0)
  L_0 = 2. * E_0 / ((n-1.)*t_0)
  P_0 = np.sqrt(E_0 / ((n-1) * pi_**2* I))
  return t_0, L_0, P_0

