# -*- coding: utf-8 -*-
# @Author: acharlet

'''
This file contains physical constants in CGS units, read parameters and derives useful quantities
'''
from pathlib import Path
import os
import numpy as np

#params_path = Path().absolute().parents[1] / 'phys_input.MWN'

# Constants
c_ = 2.99792458e10
mp_ = 1.67262192e-24
Msun_ = 1.9891e33
pi_ = np.pi

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


# ejecta core expansion velocity (cm/s) at the core/envelope boundary
v_t = np.sqrt(2*(5-delta)*(omega-5) / ((3-delta)*(omega-3))) * np.sqrt(E_sn/M_ej)
# density scaling of the ejecta (g cm^-3 s^3), rho = D * (v_t * t / r)^(index) / t^3
D = (5-delta)*(omega-5)/(2*pi_*(omega-delta)) * E_sn / v_t**5

# stored in the MyEnv class:
#P_0                  # initial rotation period (s)
#E_0                  # total energy injected in the nebula (erg)
#lfacwind             # wind Lorentz factor
#t_start              # SNR age at simulation beginning
#rmin0                # grid rmin (cm)
#rmax0                # grid rmax (cm)
#t_0                  # spindown time (s)
#L_0                  # spindown luminosity (erg/s)
#t_c                  # crossing time for the nebula (Blondin+ 01, see Granot+ 17)
#R_c                  # crossing radius

class MyEnv:
  def setupEnv(self, path):
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
    self.t_0 = I * c_**3 * self.P_0**2 / (8. * pi_**2 * f * R_ns**6 * B_0**2)
    self.t_start *= self.t_0
    self.L_0 = 2. * self.E_0 / ((n-1.)*self.t_0)
    self.t_c = self.t_0 * 2.64 * ((omega - 5)/omega) * ((n-1)/2)
    self.R_c = v_t * self.t_c

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
