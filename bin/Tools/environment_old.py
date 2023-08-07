# -*- coding: utf-8 -*-
# @Author: acharlet

'''
This file read parameters and derives useful parameter-dependant quantities
to generate setup et scale plots
'''

mode = 'shells'            # 'shells', 'PWN'. Determines the plotting/variable names

# Imports
# --------------------------------------------------------------------------------------------------
from pathlib import Path
import os
import numpy as np
from scipy.optimize import fsolve
from phys_constants import *

#params_path = Path().absolute().parents[1] / 'phys_input.MWN'




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

# functions to determine v_t
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

v_t = vt_from_SNRparams(M_ej, E_sn, delta, omega, v_min, v_max)
D = f_l(3, v_min/v_t, v_max/v_t) * M_ej / (4.*pi_ * v_t**3)


# stored in the MyEnv class:
#P_0                  # initial rotation period (s)
#E_0                  # total energy injected in the nebula (erg)
#lfacwind             # wind Lorentz factor
#t_start              # SNR age at simulation beginning (s)
#rmin0                # grid rmin (cm)
#rmax0                # grid rmax (cm)
#t_0                  # spindown time (s)
#L_0                  # spindown luminosity (erg/s)
#t_c                  # crossing time for the nebula (Blondin+ 01, see Granot+ 17)
#R_c                  # crossing radius

# added in the MyEnv class in MWN_setup.py and init_conditions_MWN.py
#R_b                  # nebula radius (cm)
#R_c                  # ejecta core radius (cm)
#R_e                  # ejecta envelope radius (cm)
#beta_w               # wind velocity (~1 in units c)
#rho_w                # wind density at injection and sim start (g/cm3)
#rho_ej               # ejecta core density at sim start (g/cm3)
#rhoNorm, pNorm       # normalization density and pressure, pNorm = rhoNorm*c**2

class MyEnv:
  def setupEnv(self, path):
    self.P_0 = None
    self.E_0 = None
    self.read_input(path)
    self.update_params()

  def read_input(self, path):
    '''Reads GAMMA/phys_input.MWN file, updates environment values'''
    with open(path, 'r') as f:
      lines = f.read().splitlines()
      for line in lines:
        if line.startswith('P_0'):
          l = line.split()
          self.P_0 = float(l[2]) * 1e-3
        elif line.startswith('E_0'):
          l = line.split()
          self.E_0_frac = float(l[2])
          self.E_0 = float(l[2]) * E_sn
        elif line.startswith('lfacwind'):
          l = line.split()
          self.lfacwind = float(l[2])
        elif line.startswith('t_start'):
          l = line.split()
          self.t_start = float(l[2])
        elif line.startswith('rmin0'):
          l = line.split()
          self.rmin0 = float(l[2])
        elif line.startswith('rmax0'):
          l = line.split()
          self.rmax0 = float(l[2])
      
  def update_params(self):
    '''Physical values derived from input parameters'''
    f = get_spindown_f(theta_B, sp_law)
    if self.P_0 and self.E_0:
      print("Comment either P_0 or E_0 in phys_inputs file")
      pass
    if (not self.P_0) and (not self.E_0):
      print("Uncomment either P_0 or E_0 in phys_inputs file")
      pass
    if self.P_0:
      self.t_0 = I * c_**3 * self.P_0**2 / (8. * pi_**2 * f * R_ns**6 * B_0**2)
      self.L_0 = (f * B_0**2 * R_ns**6 * (2.*pi_/self.P_0)**4 / c_**3 )
      self.E_0 = (n-1) * I * (2 * pi_ / self.P_0)**2 / 4.
    if self.E_0:
      self.t_0 = (n-1) * I**2 * c_**3 / (8. * f * R_ns**6 * B_0**2 * self.E_0)
      self.L_0 = 2. * self.E_0 / ((n-1.)*self.t_0)
      self.P_0 = np.sqrt(self.E_0 / ((n-1) * pi_**2* I))
    self.t_c = self.t_0 * 2.64 * ((omega - 5)/omega) * ((n-1)/2)
    self.R_c = v_t * self.t_c

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
