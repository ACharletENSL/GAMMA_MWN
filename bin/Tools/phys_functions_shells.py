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
  def Ain_keys_butnotB(A, B):
    return A in env.__dict__.keys() and not B in env.__dict__.keys()

  # add D0 if ton and vice-versa
  if Ain_keys_butnotB('D01', 't1'):
    env.beta1 = derive_velocity_from_proper(env.u1)
    env.t1    = env.D01/(env.beta1*c_)
  elif Ain_keys_butnotB('t1', 'D01'):
    env.beta1 = derive_velocity_from_proper(env.u1)
    env.D01   = env.t1*env.beta1*c_
  if Ain_keys_butnotB('D04', 't4'):
    env.beta4 = derive_velocity_from_proper(env.u4)
    env.t4    = env.D04/(env.beta4*c_)
  elif Ain_keys_butnotB('t4', 'D04'):
    env.beta4 = derive_velocity_from_proper(env.u1)
    env.D04   = env.t4*env.beta4*c_

  # t0 and R0
  if 'toff' in env.__dict__.keys():
    env.t0 = env.beta1*env.toff/(env.beta4 - env.beta1)
  if Ain_keys_butnotB('t0', 'R0'):
    env.R0 = env.beta4*c_*env.t0
  elif Ain_keys_butnotB('R0', 't0'):
    env.t0 = env.R0/(env.beta4*c_)

  # add rho if Ek and vice-versa
  if Ain_keys_butnotB('Ek1', 'rho1'):
    lfac1    = derive_Lorentz_from_proper(env.u1)
    V1c2     = 4.*pi_*env.R0**2*env.D01*c_**2
    env.rho1 = (env.Ek1/(lfac1-1))/(V1c2*lfac1)
  elif Ain_keys_butnotB('rho1', 'Ek1'):
    lfac1    = derive_Lorentz_from_proper(env.u1)
    V1c2     = 4.*pi_*env.R0**2*env.D01*c_**2
    env.Ek1  = (lfac1-1)*env.rho1*V1c2*lfac1
  if Ain_keys_butnotB('Ek4', 'rho4'):
    lfac4    = derive_Lorentz_from_proper(env.u4)
    V4c2     = 4.*pi_*env.R0**2*env.D04*c_**2
    env.rho4 = (env.Ek4/(lfac4-1))/(V4c2*lfac4)
  elif Ain_keys_butnotB('rho4', 'Ek4'):
    lfac4    = derive_Lorentz_from_proper(env.u4)
    V4c2     = 4.*pi_*env.R0**2*env.D04*c_**2
    env.Ek4  = (lfac4-1)*env.rho4*V4c2*lfac4
  
  # complete with pressure
  p0 = min(env.rho1, env.rho4)*env.Theta0*c_**2
  env.p1 = p0
  env.p4 = p0

def shells_add_analytics(env):
  '''
  Completes environment class with analytical values
  '''

  env.f      = env.rho4/env.rho1
  u21        = derive_u_in1(env.u1, env.u4, env.f)
  u34        = u21/np.sqrt(env.f)
  env.lfac21 = derive_Lorentz_from_proper(u21)
  env.lfac34 = derive_Lorentz_from_proper(u34)
  lfac1  = derive_Lorentz_from_proper(u1)
  lfac4  = derive_Lorentz_from_proper(u4)
  beta1  = u1/lfac1
  beta4  = u4/lfac4
  M1     = Ek1/((lfac1-1)*c_**2)
  M4     = Ek4/((lfac4-1)*c_**2)

  # Analytical estimates
  env.u_sh = derive_u_lab(env.u1, u21)
  lfac     = derive_Lorentz_from_proper(env.u)
  env.beta = env.u/lfac
  env.rho2 = 4.*env.lfac21*env.rho1
  env.p_sh = (4./3.) * u21**2 * env.rho1 * c_**2
  env.rho3 = 4.*env.lfac34*env.rho4

  # shocks
  env.betaFS = derive_betaFS(u1, u21, env.u)
  env.tFS    = derive_FS_crosstime(beta1, env.D01, env.betaFS)
  env.D2f    = derive_shellwidth_crosstime(env.D01, lfac1, lfac, env.lfac21)
  env.Ei2f   = derive_Eint_crosstime(M1, env.u, env.lfac21)
  env.Ek2f   = derive_Ek_crosstime(M1, lfac)

  env.betaRS = derive_betaRS(env.u4, u34, env.u)
  env.tRS    = derive_RS_crosstime(beta4, env.D04, env.betaRS)
  env.D3f    = derive_shellwidth_crosstime(env.D04, lfac4, lfac, env.lfac34)
  env.Ei3f   = derive_Eint_crosstime(M4, env.u, env.lfac34)
  env.Ek3f   = derive_Ek_crosstime(M4, lfac)

  # rarefaction waves
  T2     = env.p/(env.rho2*c_**2)
  num    = (3*T2**2 + 5.*T2*np.sqrt(T2**2+(4./9.)))
  denom  = (2. + 12*T2**2 + 12*T2*np.sqrt(T2**2+(4./9.)))
  betas2 = np.sqrt(num/denom)
  T3     = env.p/(env.rho3*c_**2)
  num    = (3*T3**2 + 5.*T3*np.sqrt(T3**2+(4./9.)))
  denom  = (2. + 12*T3**2 + 12*T3*np.sqrt(T3**2+(4./9.)))
  betas3 = np.sqrt(num/denom)
  env.betaRFm2 = (env.beta-betas2)/(1-env.beta*betas2)
  env.betaRFp2 = (env.beta+betas2)/(1+env.beta*betas2)
  env.betaRFm3 = (env.beta-betas3)/(1-env.beta*betas3)
  env.betaRFp3 = (env.beta+betas3)/(1+env.beta*betas3)

  env.tRFp3 = env.D3f / (c_*(env.betaRFp3-env.beta))
  env.tRFp2 = (env.betaFS-env.beta)*(env.tRS+env.tRFp3)/(env.betaRFp2-env.betaFS)
  env.tRFm2 = env.D2f / (c_*(env.beta-env.betaRFm2))
  env.tRFm3 = (env.beta-env.betaRS)*(env.tFS+env.tRFm2)/(env.betaRS-env.betaRFm3)

def shells_snapshot_fromenv(env, r, t):
  '''
  Given a array of radii and a time, returns values of rho, u and p
  to plot a theoretical snapshot
  '''
  rho = np.zeros(r.shape)
  u   = np.zeros(r.shape)
  p   = np.zeros(r.shape)

  beta1 = derive_velocity_from_proper(env.u1)
  beta4 = derive_velocity_from_proper(env.u4)
  Rcd = env.R0 + env.beta*c_*t

  if t<env.tRS:
    # RS propagates through the unshocked shell 4
    Rrs = env.R0 + env.betaRS*c_*t
    R4b = env.R0 - env.D04 + beta4*c_*t
    i_4 = np.argwhere((r>=R4b) & (r<Rrs))[:,0]
    i_3 = np.argwhere((r>=Rrs) & (r<=Rcd))[:,0]
    rho[i_4] = env.rho4
    u[i_4]   = env.u4
    rho[i_3] = env.rho3
    u[i_3]   = env.u_sh
    p[i_3]   = env.p_sh
  elif (t>=env.tRS) and (t<env.tRFp3):
    # rarefaction wave + propagates in the shocked shell 3
    print("RS has crossed the shell, implement rarefaction wave")
    Rrfp = Rcd - env.D3f + env.betaRFp3*c_*(t-env.tRS)
    i_3  = np.argwhere((r>=Rrfp) & (r<=Rcd))[:,0]
    rho[i_3] = env.rho3
    u[i_3]   = env.u_sh
    p[i_3]   = env.p_sh
  elif (t>=env.tRFp3):
    print("Rarefaction wave + has crossed the CD")


  R1f = env.R0 + env.D01 + beta1*c_*t


def shells_shockedvars(Ek1, u1, D01, Ek4, u4, D04, R0):
  '''
  Returns comoving density, proper velocity and pressure in shocked regions
  '''
  # unshocked shells
  lfac1 = derive_Lorentz_from_proper(u1)
  beta1 = u1/lfac1
  lfac4 = derive_Lorentz_from_proper(u4)
  beta4 = u4/lfac4
  V1c2  = 4.*pi_*R0**2*D01*c_**2
  rho1  = (Ek1/(lfac1-1))/(V1c2*lfac1)
  V4c2  = 4.*pi_*R0**2*D04*c_**2
  rho4  = (Ek4/(lfac4-1))/(V4c2*lfac4)

  # shocked shells
  f      = derive_proper_density_ratio(Ek1, Ek4, D01, D04, u1, u4)
  u21    = derive_u_in1(u1, u4, f)
  lfac21 = derive_Lorentz_from_proper(u21)
  u      = derive_u_lab(u1, u21)
  lfac   = derive_Lorentz_from_proper(u)
  rho2   = 4.*lfac21*rho1
  p      = (4./3.) * u21**2 * rho1 * c_**2
  u34    = u21/np.sqrt(f)
  lfac34 = derive_Lorentz_from_proper(u34)
  rho3   = 4.*lfac34*rho4

  return rho2, rho3, u, p, lfac21, lfac34
  
def shells_snapshot(Ek1, u1, D01, Ek4, u4, D04, R0, t, r):
  '''
  Creates maps of rho, u, p from shell params and t over radii r
  '''

  # unshocked shells
  lfac1 = derive_Lorentz_from_proper(u1)
  beta1 = u1/lfac1
  lfac4 = derive_Lorentz_from_proper(u4)
  beta4 = u4/lfac4
  V1c2  = 4.*pi_*R0**2*D01*c_**2
  rho1  = (Ek1/(lfac1-1))/(V1c2*lfac1)
  V4c2  = 4.*pi_*R0**2*D04*c_**2
  rho4  = (Ek4/(lfac4-1))/(V4c2*lfac4)

  # shocked shells
  f      = derive_proper_density_ratio(Ek1, Ek4, D01, D04, u1, u4)
  u21    = derive_u_in1(u1, u4, f)
  lfac21 = derive_Lorentz_from_proper(u21)
  u2     = derive_u_lab(u1, u21)
  lfac   = derive_Lorentz_from_proper(u2)
  rho2   = 4.*lfac21*rho1
  p2     = (4./3.) * u21**2 * rho1 * c_**2
  u34    = u21/np.sqrt(f)
  lfac34 = derive_Lorentz_from_proper(u34)
  rho3   = 4.*lfac34*rho4
  
  # radii
  beta   = derive_velocity_from_proper(u2)
  betaFS = derive_betaFS(u1, u21, u2)
  betaRS = derive_betaRS(u4, u34, u2)
  Rcd  = R0 + beta*c_*t
  R4   = R0 - D04 + beta4*c_*t
  R1   = R0 + D01 + beta1*c_*t
  Rfs  = R0 + betaFS*c_*t
  Rrs  = R0 + betaRS*c_*t

  rho = np.zeros(r.shape)
  u   = np.zeros(r.shape)
  p   = np.zeros(r.shape)

  #print(f"{R4}, {Rrs}, {Rcd}, {Rfs}, {R1}")
  if R4 < Rrs:
    i_4 = np.argwhere((r>=R4) & (r<Rrs))[:,0]
    rho[i_4] = rho4
    u[i_4]   = u4
    i_3 = np.argwhere((r>=Rrs) & (r<=Rcd))[:,0]
    rho[i_3] = rho3
    u[i_3]   = u2
    p[i_3]   = p2
  else:
    D3f = derive_shellwidth_crosstime(D04, lfac4, lfac, lfac34)
    Rrs = Rcd - D3f
    i_3 = np.argwhere((r>=Rrs) & (r<=Rcd))[:,0]
    rho[i_3] = rho3
    u[i_3]   = u2
    p[i_3]   = p2
    print("RS crossing time passed, need to implement rarefaction wave")
  
  if R1 > Rfs:
    i_1 = np.argwhere((r>Rfs) & (r<=R1))[:,0]
    rho[i_1] = rho1
    u[i_1]   = u1
    i_2 = np.argwhere((r>Rcd) & (r<=Rfs))[:,0]
    rho[i_2] = rho2
    u[i_2]   = u2
    p[i_2]  = p2
  else:
    D2f = derive_shellwidth_crosstime(D01, lfac1, lfac, lfac21)
    Rfs = Rcd + D2f
    i_2 = np.argwhere((r>Rcd) & (r<=Rfs))[:,0]
    rho[i_2] = rho2
    u[i_2]   = u2
    p[i_2]  = p2
    print("FS crossing time passed, need to implement rarefaction wave")
  
  return rho, u, p

def shells_add_analytics_old(env):
  '''
  Add analytical estimates from data in the env class
  '''
  vars2add  = shells_phys2analytics(env.Ek1, env.u1, env.D01, env.Ek4, env.u4, env.D04)
  names2add = ['u', 'f', 'lfac21', 'betaFS', 'tFS', 'Df2', 'Eintf2', 'Ekf2',
    'lfac34', 'betaRS', 'tRS', 'Df3', 'Eintf3', 'Ekf3']
  for name, value in zip(names2add, vars2add):
    setattr(env, name, value)

def shells_phys2analytics(Ek1, u1, D1, Ek4, u4, D4):
  '''
  Derives all relevant analytical estimates to compare to sim results
  Returns velocity of FS and RS, respective crossing times, and
  final state of the shells (2 and 3) at respective crossing times
  '''

  # intermediate values
  f      = derive_proper_density_ratio(Ek1, Ek4, D1, D4, u1, u4)
  u21    = derive_u_in1(u1, u4, f)
  u34    = u21/np.sqrt(f)
  lfac21 = derive_Lorentz_from_proper(u21)
  lfac34 = derive_Lorentz_from_proper(u34)
  lfac1  = derive_Lorentz_from_proper(u1)
  lfac4  = derive_Lorentz_from_proper(u4)
  beta1  = u1/lfac1
  beta4  = u4/lfac4
  M1     = Ek1/((lfac1-1)*c_**2)
  M4     = Ek4/((lfac4-1)*c_**2)

  # Analytical estimates
  u      = derive_u_lab(u1, u21)
  lfac   = derive_Lorentz_from_proper(u)

  betaFS = derive_betaFS(u1, u21, u)
  tFS    = derive_FS_crosstime(beta1, D1, betaFS)
  D2     = derive_shellwidth_crosstime(D1, lfac1, lfac, lfac21)
  Eint2  = derive_Eint_crosstime(M1, u, lfac21)
  Ekf2   = derive_Ek_crosstime(M1, lfac)

  betaRS = derive_betaRS(u4, u34, u)
  tRS    = derive_RS_crosstime(beta4, D4, betaRS)
  D3     = derive_shellwidth_crosstime(D4, lfac4, lfac, lfac34)
  Eint3  = derive_Eint_crosstime(M4, u, lfac34)
  Ekf3   = derive_Ek_crosstime(M4, lfac)

  return u, f, lfac21, betaFS, tFS, D2, Eint2, Ekf2, lfac34, betaRS, tRS, D3, Eint3, Ekf3

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
def derive_proper_density_ratio(Ek1, Ek4, D01, D04, u1, u4):
  '''
  Derives proper density ratio between shells from the parameters
  '''
  lfac1 = derive_Lorentz_from_proper(u1)
  beta1 = u1/lfac1
  lfac4 = derive_Lorentz_from_proper(u4)
  beta4 = u4/lfac4
  chi   = D01/D04
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
  fac    = np.sqrt((2*f**(3/2)*lfac41 - f*(1+f))/(2*f*(u41**2+lfac41**2) - (1+f**2)))
  return u41 * fac

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
