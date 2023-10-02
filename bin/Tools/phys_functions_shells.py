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
  env.beta1 = derive_velocity_from_proper(env.u1)
  env.beta4 = derive_velocity_from_proper(env.u4)

  if Ain_keys_butnotB('D01', 't1'):
    env.t1    = env.D01/(env.beta1*c_)
  elif Ain_keys_butnotB('t1', 'D01'):
    env.D01   = env.t1*env.beta1*c_
  if Ain_keys_butnotB('D04', 't4'):
    env.t4    = env.D04/(env.beta4*c_)
  elif Ain_keys_butnotB('t4', 'D04'):
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
  env.lfac1  = derive_Lorentz_from_proper(env.u1)
  env.lfac4  = derive_Lorentz_from_proper(env.u4)
  env.beta1  = env.u1/env.lfac1
  env.beta4  = env.u4/env.lfac4
  env.M1     = env.Ek1/((env.lfac1-1)*c_**2)
  env.M4     = env.Ek4/((env.lfac4-1)*c_**2)

  # Analytical estimates
  env.u    = derive_u_lab(env.u1, u21)
  env.lfac = derive_Lorentz_from_proper(env.u)
  env.beta = env.u/env.lfac
  env.rho2 = 4.*env.lfac21*env.rho1
  env.p    = (4./3.) * u21**2 * env.rho1 * c_**2
  env.rho3 = 4.*env.lfac34*env.rho4

  # shocks
  env.betaFS = derive_betaFS(env.u1, u21, env.u)
  env.tFS    = derive_FS_crosstime(env.beta1, env.D01, env.betaFS)
  env.D2f    = derive_shellwidth_crosstime(env.D01, env.lfac1, env.lfac, env.lfac21)
  env.Ei2f   = derive_Eint_crosstime(env.M1, env.u, env.lfac21)
  env.Ek2f   = derive_Ek_crosstime(env.M1, env.lfac)

  env.betaRS = derive_betaRS(env.u4, u34, env.u)
  env.tRS    = derive_RS_crosstime(env.beta4, env.D04, env.betaRS)
  env.D3f    = derive_shellwidth_crosstime(env.D04, env.lfac4, env.lfac, env.lfac34)
  env.Ei3f   = derive_Eint_crosstime(env.M4, env.u, env.lfac34)
  env.Ek3f   = derive_Ek_crosstime(env.M4, env.lfac)


  # rarefaction waves
  T2     = env.p/(env.rho2*c_**2)
  betas2 = derive_cs_fromT(T2)
  T3     = env.p/(env.rho3*c_**2)
  betas3 = derive_cs_fromT(T3)
  env.betaRFm2 = (env.beta-betas2)/(1-env.beta*betas2)
  env.betaRFp2 = (env.beta+betas2)/(1+env.beta*betas2)
  env.betaRFm3 = (env.beta-betas3)/(1-env.beta*betas3)
  env.betaRFp3 = (env.beta+betas3)/(1+env.beta*betas3)

  env.tRFp3 = env.D3f / (c_*(env.betaRFp3-env.beta))
  env.tRFp2 = (env.betaFS-env.beta)*(env.tRS+env.tRFp3)/(env.betaRFp2-env.betaFS)
  env.tRFm2 = env.D2f / (c_*(env.beta-env.betaRFm2))
  env.tRFm3 = (env.beta-env.betaRS)*(env.tFS+env.tRFm2)/(env.betaRS-env.betaRFm3)

  # thermal efficiencies
  add_weightfactors(env)
 

def shells_snapshot_fromenv(env, r, t):
  '''
  Given a array of radii and a time, returns values of rho, u and p
  to plot a theoretical snapshot
  Vars in code units (units of c and chosen rho), r in cgs
  '''
  rho = np.zeros(r.shape)
  vel = np.zeros(r.shape)
  prs = np.zeros(r.shape)

  rho0 = getattr(env, env.rhoNorm)
  p0   = rho0*c_**2

  R4, Rrs, Rcd, Rfs, R1 = derive_radii_withtime(env, t)

  if t < env.tRS:
    i_4 = np.argwhere((r>=R4) & (r<Rrs))[:,0]
    rho[i_4] = env.rho4
    vel[i_4] = env.u4
    i_3 = np.argwhere((r>=Rrs) & (r<=Rcd))[:,0]
    rho[i_3] = env.rho3
    vel[i_3] = env.u
    prs[i_3] = env.p
  else:
    i_3 = np.argwhere((r>=Rrs) & (r<=Rcd))[:,0]
    rho[i_3] = env.rho3
    vel[i_3] = env.u
    prs[i_3] = env.p
    print("RS crossing time passed, need to implement rarefaction wave")
  
  if t < env.tFS:
    i_1 = np.argwhere((r>Rfs) & (r<=R1))[:,0]
    rho[i_1] = env.rho1
    vel[i_1] = env.u1
    i_2 = np.argwhere((r>Rcd) & (r<=Rfs))[:,0]
    rho[i_2] = env.rho2
    vel[i_2] = env.u
    prs[i_2] = env.p
  else:
    i_2 = np.argwhere((r>Rcd) & (r<=Rfs))[:,0]
    rho[i_2] = env.rho2
    vel[i_2] = env.u
    prs[i_2] = env.p
    print("FS crossing time passed, need to implement rarefaction wave")
  
  # renormalize to code units
  rho /= rho0
  prs /= p0

  if env.geometry == 'spherical':
    rho *= (r/env.R0)**-2
    prs *= (r/env.R0)**-2
  return rho, vel, prs

def add_weightfactors(env):
  '''
  Add the weighting factors (defined as the final mass ratios 
  of shocked shell vs initial shell) depending on the case
  for rarefaction wave propagation
  '''
  case = get_rfscenario(env)
  env.alpha2 = 1.
  env.alpha3 = 1.
  if case == 1:
    env.alpha2 = (env.tRS+env.tRFp3+env.tRFp2)/env.tFS
  elif case == 6:
    env.alpha3 = (env.tFS + env.tRFm2 + env.tRFm3)/env.tRS


def get_rfscenario(env):
  '''
  Returns the expected case for the rarefaction wave propagation
  '''
  # derive the various velocity differences and lfac ratios
  FSms1   = env.betaFS - env.beta1
  FSmsh   = env.betaFS - env.beta
  RFp2mFS = env.betaRFp2 - env.betaFS
  s4mRS   = env.beta4 - env.betaRS
  RFp3msh = env.betaRFp3 - env.beta
  shmRS   = env.beta - env.betaRS
  shmRFm2 = env.beta - env.betaRFm2
  RSmRFm3 = env.betaRS - env.betaRFm3
  lfacratio34 = (env.lfac4/env.lfac)/(4.*env.lfac34)
  lfacratio21 = (env.lfac1/env.lfac)/(4.*env.lfac21)

  # derive the critical lines
  chi1 = FSms1*(1.+ FSmsh/RFp2mFS)*((1./s4mRS)+lfacratio34/RFp3msh)
  chi2 = FSms1*((1./s4mRS)+lfacratio34/RFp3msh)
  chi3 = FSms1/s4mRS
  chi4 = 1./(s4mRS*((1./FSms1) + lfacratio21/shmRFm2))
  chi5 = 1./(s4mRS*(1. + shmRS/RSmRFm3)*((1./FSms1) + lfacratio21/shmRFm2))

  # which case?
  chi = env.D01 / env.D04
  critlines = [chi1, chi2, chi3, chi4, chi5]
  case = np.searchsorted(critlines, chi) + 1

  return case

# time estimates
def derive_masses_withtime(env, t):
  '''
  Return the masses for each shell with time
  '''
  M1 = np.where(t<=env.tFS, env.M1 * (1. - t/env.tFS),
    env.M1 * (t-env.tFS)/env.tRFm2)
  M2 = np.where(t<=env.tFS, env.M1 * (t/env.tFS),
    env.M1 * (1. - (t-env.tFS)/env.tRFm2))
  M3 = np.where(t<=env.tRS, env.M4 * (t/env.tRS),
    env.M4 * (1. - (t-env.tRS)/env.tRFp3))
  M4 = np.where(t<=env.tRS, env.M4 * (1. - t/env.tRS),
    env.M4 *(t-env.tRS)/env.tRFp3)
  return M4, M3, M2, M1

def derive_Eint_withtime(env, t):
  '''
  Return internal energy for each shell with time
  '''
  E1 = np.where(t<=env.tFS, 0., 0.)
    #env.Ei2f * (t-env.tFS)/env.tRFm2)
  E2 = np.where(t<=env.tFS, env.Ei2f * (t/env.tFS),
    env.Ei2f * (1. - (t-env.tFS)/env.tRFm2))
  E3 = np.where(t<=env.tRS, env.Ei3f * (t/env.tRS),
    env.Ei3f * (1. - (t-env.tRS)/env.tRFp3))
  E4 = np.where(t<=env.tRS, 0., 0.)
    #env.Ei3f * (t-env.tRS)/env.tRFp3)
  return E4, E3, E2, E1

def derive_Ekin_withtime(env, t):
  '''
  Return kinetic energy for each shell with time
  '''
  E1 = np.where(t<=env.tFS, env.Ek1 * (1. - t/env.tFS), 0.)
    #env.Ek2f * (t-env.tFS)/env.tRFm2)
  E2 = np.where(t<=env.tFS, env.Ek2f * (t/env.tFS),
    env.Ek2f * (1. - (t-env.tFS)/env.tRFm2))
  E3 = np.where(t<=env.tRS, env.Ek3f * (t/env.tRS),
    env.Ek3f * (1. - (t-env.tRS)/env.tRFp3))
  E4 = np.where(t<=env.tRS, env.Ek4 * (1. - t/env.tRS), 0.)
    #env.Ek3f * (t-env.tRS)/env.tRFp3)
  return E4, E3, E2, E1

def derive_shellwidth_withtime(env, t):
  '''
  Return shell width for each shell with time
  '''
  D1 = np.where(t<=env.tFS, env.D01 * (1. - t/env.tFS), 0.)
  D2 = np.where(t<=env.tFS, env.D2f * (t/env.tFS),
    env.D2f*(1. - (t-env.tFS)/env.tRFm2))
  D3 = np.where(t<=env.tRS, env.D3f * (t/env.tRS),
    env.D3f*(1. - (t-env.tRS)/env.tRFp3))
  D4 = np.where(t<=env.tRS, env.D04 * (1. - t/env.tRS), 0.)
  return D4, D3, D2, D1

def derive_radii_withtime(env, t):
  '''
  Returns expected radii for each interface with time
  need to determine min velocity of the RF when interacting with ext medium
  '''

  Rb0 = np.array([-env.D04, 0.]) + env.R0
  Rf0 = np.array([0., env.D01]) + env.R0
  vb0 = np.array([env.beta4, env.betaRS])*c_
  vf0 = np.array([env.betaFS, env.beta1])*c_
  vbc = np.array([0., env.betaRFp3])*c_
  vfc = np.array([env.betaRFm2, 0.])*c_

  R4  = np.where(t<env.tRS, ) 
  Rcd = env.R0 + env.beta*c_*t

  Rb  = [np.where(t<env.tRS, Rb0[n]+vb0[n]*t, Rb0[n]+vb0[n]*env.tRS+vbc[n]*(t-env.tRS))
    for n in [0, 1]]
  Rf  = [np.where(t<env.tFS, Rf0[n]+vf0[n]*t, Rf0[n]+vf0[n]*env.tFS+vbc[n]*(t-env.tFS))
    for n in [0, 1]]
  R4, Rrs = Rb
  Rfs, R1 = Rf
  return R4, Rrs, Rcd, Rfs, R1


# Theoretical expectations with time
# -----------------------------------------------------
def time_analytical(varkey, t, env):
  '''
  Returns time series of asked variable
  follows same rules as run_analysis:get_timeseries
  '''
  if varkey == 'R':
    return derive_radii_withtime(env, t)
  elif varkey == 'Rct':
    return np.array(derive_radii_withtime(env, t))-c_*t
  elif varkey == 'D':
    return derive_shellwidth_withtime(env, t)
  elif varkey == 'v':
    radii = derive_radii_withtime(env, t)
    vel = np.gradient(radii, t, axis=1)
    return vel
  elif varkey == 'u_i':
    radii = derive_radii_withtime(env, t)
    vel = np.gradient(radii, t, axis=1)
    u = derive_proper(vel)
    return u
  elif varkey == 'V':
    if env.geometry == 'spherical':
      R4, Rrs, Rcd, Rfs, R1 = derive_radii_withtime(env, t)
      V4 = (4./3.)*pi_*(Rrs**3 - R4**3)
      V3 = (4./3.)*pi_*(Rcd**3 - Rrs**3)
      V2 = (4./3.)*pi_*(Rfs**3 - Rcd**3)
      V1 = (4./3.)*pi_*(R1**3 - Rfs**3)
      return V4, V3, V2, V1
    elif env.geometry == 'cartesian':
      D4, D3, D2, D1 = np.array(derive_shellwidth_withtime(env, t))*(4.*pi_*env.R0**2)
      return V4, V3, V2, V1
  elif varkey == 'M':
    return derive_masses_withtime(env, t)
  elif varkey == 'Msh':
    M4, M3, M2, M1 = derive_masses_withtime(env, t)
    return M4+M3, M1+M2
  elif varkey == 'Ek':
    return derive_Ekin_withtime(env, t)
  elif varkey == 'Ei':
    return derive_Eint_withtime(env, t)
  elif varkey == 'E':
    E4, E3, E2, E1 = np.array(derive_Ekin_withtime(env, t)) + np.array(derive_Eint_withtime(env, t))
    return E4, E3, E2, E1
  elif varkey == 'Esh':
    E4, E3, E2, E1 = np.array(derive_Ekin_withtime(env, t)) + np.array(derive_Eint_withtime(env, t))
    return E4+E3, E1+E2
  elif varkey == 'Etot':
    E4, E3, E2, E1 = np.array(derive_Ekin_withtime(env, t)) + np.array(derive_Eint_withtime(env, t))
    Etot = E4+E3+E2+E1
    return [Etot]
  elif varkey == 'epsth':
    Ei4, Ei3, Ei2, Ei1 = derive_Eint_withtime(env, t)
    return Ei3/env.Ek4, Ei2/env.Ek1


def get_analytical(var, t, r, env):
  '''
  Returns array of theoretical values
  '''
  rho, u, p = shells_snapshot_fromenv(env, r, t)
  if var in ["rho", "u", "p"]:
    if var == "rho": return rho
    elif var == "u": return u
    elif var == "p": return p
  elif var == "vx":
    v = derive_velocity_from_proper(u)
    return v
  elif var == "lfac":
    lfac = derive_Lorentz_from_proper(u)
    return lfac
  elif var in ["D", "sx", "tau"]:
    D, s, tau = prim2cons(rho, u, p)
    if var == "D": return D
    elif var == "sx": return s
    elif var == "tau": return tau
  elif var == "T":
    T = derive_temperature(rho, p)
    return T
  elif var == "h":
    h = derive_enthalpy(rho, p)
    return h
  elif var == "Eint":
    eint = derive_Eint(rho, p)
    return eint
  elif var == "Ekin":
    ek = derive_Ekin(rho, u)
    return ek
  elif var == "Emass":
    return rho

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

# W_RS = E4_k0 * (4/3) * ((G34**2-1)/(G4*(G4-1)) * (beta/(beta4-beta)) * (1-(1/(4*G34))*(G4/G)))
## pdV work done by region 3 on region 2 at RS crossing time
