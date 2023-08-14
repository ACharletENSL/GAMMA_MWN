# -*- coding: utf-8 -*-
# @Author: acharlet

'''
This file contains functions to derive all relevant quantities
in the colliding shells situation
'''

# Imports
from phys_functions import *
from phys_constants import *

# basic parameters (will be modified later/read from file/function inputs)
# start from phys params : (L, u, ton) pairs + toff OR num params (rho, u, D0) pairs + R0

def shells_complete_setup(env):
  '''
  Completes environment class with missing information
  '''
  vars2add  = []
  names2add = []

  # add physical set if setup contains numerical set or vice-versa
  if 'L1' in env.__dict__.keys():
    vars2add  = shells_phys2num(env.L1, env.u1, env.t1, env.L4, env.u4, env.t4, env.toff)
    names2add = ['rho1', 'beta1', 'D01', 'rho4', 'beta4', 'D04', 't0']
  elif 'rho1' in env.__dict__.keys():
    vars2add  = shells_num2phys(env.rho1, env.u1, env.D01, env.rho4, env.u4, env.D04, env.R0)
    names2add = ['L1', 'beta1', 't1', 'L4', 'beta4', 't4', 'toff']
  for name, value in zip(names2add, vars2add):
    setattr(env, name, value)
  
  # complete with pressure
  p0 = min(env.rho1, env.rho4)*env.Theta0*c_**2
  env.p1 = p0
  env.p4 = p0

  # add t0 if R0 and vice-versa
  if 't0' in env.__dict__.keys():
    env.R0 = env.t0*c_
  elif 'R0' in env.__dict__.keys():
    env.t0 = env.R0/c_

def shells_add_analytics(env):
  '''
  Add analytical estimates from data in the env class
  '''
  vars2add  = shells_phys2analytics(env.L1, env.u1, env.t1, env.D01, env.L4, env.u4, env.t4, env.D04, env.toff)
  names2add = ['u', 'betaFS', 'tFS', 'Df2', 'Eint2', 'Ek2', 'betaRS', 'tRS', 'Df3', 'Eint3', 'Ek3']
  for name, value in zip(names2add, vars2add):
    setattr(env, name, value)


def shells_num2phys(rho1, u1, D01, rho4, u4, D04, R0):
  '''
  Derives physical quantities from a numerical setup
  '''
  beta1   = derive_velocity_from_proper(u1)
  beta4   = derive_velocity_from_proper(u4)
  t0   = R0/c_
  toff = (beta4-beta1)*t0/beta1
  t1   = D01/beta1
  t4   = D04/beta4
  L1   = rho1 * (4.*pi_*R0**2*D01*c_**2) / t1
  L4   = rho4 * (4.*pi_*R0**2*D04*c_**2) / t4
  return L1, beta1, t1, L4, beta4, t4, toff

def shells_phys2num(L1, u1, t1, L4, u4, t4, toff):
  '''
  Derives quantities for the numerical setup from the physical inputs
  '''
  beta1 = derive_velocity_from_proper(u1)
  beta4 = derive_velocity_from_proper(u4)
  t0    = (beta1*toff)/(beta4-beta1)
  D01   = beta1*t1
  D04   = beta4*t4
  rho1  = L1*t1 / (4.*pi_*R0**2*D01*c_**2)
  rho4  = L4*t4 / (4.*pi_*R0**2*D04*c_**2)
  return rho1, beta1, D01, rho4, beta4, D04, t0

def shells_phys2analytics(L1, u1, t1, D1, L4, u4, t4, D4, toff):
  '''
  Derives all relevant analytical estimates to compare to sim results
  Returns velocity of FS and RS, respective crossing times, and
  final state of the shells (2 and 3) at respective crossing times
  '''

  # intermediate values
  f      = derive_proper_density_ratio(L1, L4, t1, t4, u1, u4)
  u21    = derive_u_in1(u1, u4, f)
  u34    = u21/np.sqrt(f)
  lfac21 = derive_Lorentz_from_proper(u21)
  lfac34 = derive_Lorentz_from_proper(u34)
  lfac1  = derive_Lorentz_from_proper(u1)
  lfac4  = derive_Lorentz_from_proper(u4)
  beta1  = u1/lfac1
  beta4  = u4/lfac4
  M1     = L1*t1/((lfac1-1)*c_**2)
  M4     = L4*t4/((lfac4-1)*c_**2)

  # Analytical estimates
  u      = derive_u_lab(u1, u21)
  lfac   = derive_Lorentz_from_proper(u)

  betaFS = derive_betaFS(u1, u21, u)
  tFS    = derive_FS_crosstime(beta1, D1, betaFS)
  D2     = derive_shellwidth_crosstime(D1, lfac1, lfac, lfac21)
  Eint2  = derive_Eint_crosstime(M1, u, lfac21)
  Ek2    = derive_Ek_crosstime(M1, lfac)

  betaRS = derive_betaRS(u4, u34, u)
  tRS    = derive_RS_crosstime(beta4, D4, betaRS)
  D3     = derive_shellwidth_crosstime(D4, lfac4, lfac, lfac34)
  Eint3  = derive_Eint_crosstime(M4, u, lfac34)
  Ek3    = derive_Ek_crosstime(M4, lfac)

  return u, betaFS, tFS, D2, Eint2, Ek2, betaRS, tRS, D3, Eint3, Ek3

# Individual functions
# -----------------------------------------------------
def derive_R0(beta1, beta4, toff):
  '''
  Derives distance from source at which shells collides
  Time t0 at which this happens = R0/c
  '''
  return (beta1*toff*c_)/(beta4-beta1)

def derive_shell_density(L, ton, D0, R0):
  '''
  Derives shell density in lab frame when shells collide
  '''
  return L*t / (4.*pi_*R0**2*D0*c_**2)

# from hydro quantities to analytical estimates
def derive_proper_density_ratio(L1, L4, t1, t4, u1, u4):
  '''
  Derives proper density ratio between shells from the parameters
  '''
  lfac1 = derive_Lorentz_from_proper(u1)
  beta1 = u1/lfac1
  lfac4 = derive_Lorentz_from_proper(u4)
  beta4 = u4/lfac4
  chi   = (beta1*t1)/(beta4*t4)
  Ek1   = L1*t1
  Ek4   = L4*t4
  return chi * (Ek4/Ek1) * ((lfac1*(lfac1-1))/(lfac4*(lfac4-1)))

def derive_u_in1(u1, u4, f):
  '''
  Derives proper speed of the shocked fluid relative to frame 1
  '''
  lfac1  = derive_Lorentz_from_proper(u1)
  beta1  = u1/lfac1
  lfac4  = derive_Lorentz_from_proper(u4)
  beta4  = u4/lfac4
  lfac41 = lfac1*lfac4 - u1*u4
  u41    = derive_proper_from_Lorentz(lfac41) 
  return u41 * np.sqrt( (2*f**(3/2)*lfac41 - f*(1+f)) / (2*f*(u41**2+lfac41**2) - (1+f**2)))

def derive_u_lab(u1, u21):
  '''
  Derives proper speed of the shocked fluid in the lab frame
  u21 = derive_u_in1(u1, u4, f)
  '''
  lfac21 = derive_Lorentz_from_proper(u21)
  lfac1  = derive_Lorentz_from_proper(u1)
  beta21 = u21/lfac21
  beta1  = u1/lfac1
  return lfac21*lfac1*(beta21+beta1)

def derive_betaFS(u1, u21, u):
  '''
  Derives velocity of forward shock in the lab frame
  '''
  lfac21 = derive_Lorentz_from_proper(u21)
  lfac   = derive_Lorentz_from_proper(u)
  lfac1  = derive_Lorentz_from_proper(u1)
  beta   = u/lfac
  num    = (0.25/lfac21)*(u1/lfac) - beta
  denom  = (0.25/lfac21)*(lfac1/lfac) - 1.
  return num/denom 

def derive_betaRS(u4, u34, u):
  '''
  Derives velocity of reverse shock in the lab frame
  u34 = u21/np.sqrt(f), u21 = derive_u_in1(u1, u4, f)
  '''
  lfac34 = derive_Lorentz_from_proper(u34)
  lfac4  = derive_Lorentz_from_proper(u4)
  beta4  = u4/lfac4
  lfac   = derive_Lorentz_from_proper(u)
  num    = beta4 - 4*lfac34*(u/lfac4)
  denom  = 1 - 4*lfac34*(lfac/lfac4)
  return num/denom

def derive_FS_crosstime(beta1, D1, betaFS):
  '''
  Derives time taken by the FS to cross the whole shell 1
  '''
  return D1 / (c_*(betaFS-beta1))

def derive_RS_crosstime(beta4, D4, betaRS):
  '''
  Derives time taken by the FS to cross the whole shell 1
  '''
  return D4 / (c_*(beta4-betaRS))

def derive_Eint_crosstime(M0, u, lfacrel):
  '''
  Derives internal energy in shell 2/3 at FS/RS crossing time
  M0, lfacrel = M1, lfac21 for shell 2 ; M4, lfac34 for shell 3
  '''
  lfac = derive_Lorentz_from_proper(u)
  beta = u/lfac
  return lfac*M0*c_**2*(1+beta**2*((lfacrel+1)/(3*lfacrel)))*(lfacrel-1)

def derive_Ek_crosstime(M0, lfac):
  '''
  Derives internal energy in shell 2/3 at FS/RS crossing time
  M0 = M1 for shell 2, M4 for shell 3. lfac Lorentz factor of shocked region
  '''
  return (lfac-1)*M0*c_**2

def derive_shellwidth_crosstime(D0, lfac0, lfac, lfacrel):
  '''
  Derives width of shell with initial width D0 and Lorentz lfac0
   at shock crossing time
  For lfacrel = lfac21 for shell 2, lfac34 for shell 3
  '''
  return (D0/(4*lfacrel))*(lfac0/lfac)

# a_u = u4/u1 = 2                         # proper velocity contrast
# eta = t_on1/t_on4 = 1                   # activity timescale ratio
# chi = D1_0/D4_0 = eta*beta1/beta4       # initial widths ratio
# f = np4/np1 = u1*(G1-1) / (u4*(G4-1))   # proper density contrast
# G41 = G1*G4-u1*u4                       # relative Lorentz factor between 1 and 4
# u21 = u31 = u41 * np.sqrt( (2*f**(3/2)*G41 - f*(1+f)) / (2*f*(u41**2+G41**2) - (1+f**2)))
## proper speed of the shocked fluid relative to frame 1
# u = G21*G1*(beta1+beta21)               # proper speed of shocked fluid in the lab frame
# u34 = f**(-1/2) * u21                   # proper speed of shocked fluid relative to frame 4
# betaFS = ( ((1/(4.*G21))*(u1/G) - beta) / ( (1/(4*G21))*(G1/G) - 1)) 
# betaRS = ( (beta4 - 4*G34*(u/G4)) / (1-4*G34*(G/G4)) )    
# tFS = D1_0 / (c_*(betaFS-beta1))        # time for the FS to cross shell 1
# tRS = D4_0 / (c_*(beta4-betaRS))        # time for the RS to cross shell 4
# E2_int = G*M1_0*c_**2*(1+beta**2*((G21+1)/(3*G21)))*(G21-1)
## internal energy accumulated in zone 2 at FS crossing time
# E3_int = G*M4_0*c_**2*(1+beta**2*((G34+1)/(3*G34)))*(G34-1)
## internal energy accumulated in zone 3 at RS crossing time
# E2_k = (G-1)*M1_0*c_**2                 # max bulk kinetic energy at FS crossing time
# E3_k = (G-1)*M4_0*c_**2                 # max bulk kinetic energy at RS crossing time
# D2_f = D1_0*(1/(4*G21))*(G1/G)          # final radial width of region 2 at FS crossing time
# D3_f = D4_0*(1/(4*G34))*(G4/G)          # final radial width of region 3 at RS crossing time
# W_RS = E4_k0 * (4/3) * ((G34**2-1)/(G4*(G4-1)) * (beta/(beta4-beta)) * (1-(1/(4*G34))*(G4/G)))
## pdV work done by region 3 on region 2 at RS crossing time
