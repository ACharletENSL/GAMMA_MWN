# -*- coding: utf-8 -*-
# @Author: acharlet

'''
This file contains functions to derive all relevant quantities
in the colliding shells situation
'''

# Imports
from phys_functions import *
from phys_constants import *
from scipy.special import gamma

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

  env.beta1 = derive_velocity_from_proper(env.u1)
  env.lfac1 = derive_Lorentz_from_proper(env.u1)
  env.beta4 = derive_velocity_from_proper(env.u4)
  env.lfac4 = derive_Lorentz_from_proper(env.u4)

  # t0 and R0
  if 'toff' in env.__dict__.keys():
    env.t0 = env.beta1*env.toff/(env.beta4 - env.beta1)
  if Ain_keys_butnotB('t0', 'R0'):
    env.R0 = env.beta4*c_*env.t0
  elif Ain_keys_butnotB('R0', 't0'):
    env.t0 = env.R0/(env.beta4*c_)

  # add D0 if ton and vice-versa
  if Ain_keys_butnotB('D01', 't1'):
    #env.t1  = (env.D01/env.R0)*env.toff
    env.t1 = env.D01/(env.beta1*c_)
  elif Ain_keys_butnotB('t1', 'D01'):
    #env.D01 = (env.t1/env.toff)*env.R0
    env.D01 = env.t1*env.beta1*c_#*(1+env.R0/env.lfac1**2)
  if Ain_keys_butnotB('D04', 't4'):
    #env.t4  = (env.D04/env.R0)*env.toff
    env.t4  = env.D04/(env.beta4*c_)
  elif Ain_keys_butnotB('t4', 'D04'):
    #env.D04 = (env.t4/env.toff)*env.R0
    env.D04 = env.t4*env.beta4*c_#*(1+env.R0/env.lfac4**2)
  
  env.Nsh4 = int(np.floor(env.Nsh1*env.D04/env.D01))
  env.Ntot = 2*env.Next + env.Nsh1 + env.Nsh4
  env.V0 = (4/3.)*pi_*((env.R0+env.D01)**3 - (env.R0-env.D04)**3)
  env.dr4 = env.D04/env.Nsh4
  env.dr1 = env.D01/env.Nsh1

  # add rho if Ek and vice-versa
  if Ain_keys_butnotB('Ek1', 'rho1'):
    V1c2     = 4.*pi_*env.R0**2*env.D01*c_**2
    env.rho1 = (env.Ek1/(env.lfac1-1))/(V1c2*env.lfac1)
  elif Ain_keys_butnotB('rho1', 'Ek1'):
    V1c2     = 4.*pi_*env.R0**2*env.D01*c_**2
    env.Ek1  = (env.lfac1-1)*env.rho1*V1c2*env.lfac1
  if Ain_keys_butnotB('Ek4', 'rho4'):
    V4c2     = 4.*pi_*env.R0**2*env.D04*c_**2
    env.rho4 = (env.Ek4/(env.lfac4-1))/(V4c2*env.lfac4)
  elif Ain_keys_butnotB('rho4', 'Ek4'):
    V4c2     = 4.*pi_*env.R0**2*env.D04*c_**2
    env.Ek4  = (env.lfac4-1)*env.rho4*V4c2*env.lfac4
  
  # complete with pressure
  p0 = min(env.rho1, env.rho4)*env.Theta0*c_**2
  env.p1 = p0
  env.p4 = p0

  env.rhoscale = getattr(env, env.rhoNorm)
  env.Theta1 = env.p1/(env.rho1*c_**2)
  env.Theta4 = env.p4/(env.rho4*c_**2)

def shells_add_analytics(env):
  '''
  Completes environment class with analytical values
  '''

  env.f      = env.rho4/env.rho1
  env.a_u    = env.u4/env.u1
  env.chi    = env.D01 / env.D04
  env.u21    = derive_u_in1(env.u1, env.u4, env.f)
  env.u34    = env.u21/np.sqrt(env.f)
  env.lfac21 = derive_Lorentz_from_proper(env.u21)
  env.lfac34 = derive_Lorentz_from_proper(env.u34)
  env.beta1  = env.u1/env.lfac1
  env.beta4  = env.u4/env.lfac4
  env.M1     = env.Ek1/((env.lfac1-1)*c_**2)
  env.M4     = env.Ek4/((env.lfac4-1)*c_**2)

  # Analytical estimates
  env.u    = derive_u_lab(env.u1, env.u21)
  env.lfac = derive_Lorentz_from_proper(env.u)
  env.beta = env.u/env.lfac
  env.rho2 = 4.*env.lfac21*env.rho1
  env.p_sh = (4./3.) * env.u21**2 * env.rho1 * c_**2
  env.rho3 = 4.*env.lfac34*env.rho4

  # shocks
  env.betaFS = derive_betaFS(env.u1, env.u21, env.u)
  env.lfacFS = derive_Lorentz(env.betaFS)
  env.gFS    = env.lfac/env.lfacFS
  env.tFS    = derive_FS_crosstime(env.beta1, env.D01, env.betaFS)
  env.D2f    = derive_shellwidth_crosstime(env.D01, env.lfac1, env.lfac, env.lfac21)
  env.Ei2f   = derive_Eint_crosstime(env.M1, env.u, env.lfac21)
  env.Ek2f   = derive_Ek_crosstime(env.M1, env.lfac)
  env.eff_1  = env.Ei2f/env.Ek1

  env.betaRS = derive_betaRS(env.u4, env.u34, env.u)
  env.lfacRS = derive_Lorentz(env.betaRS)
  env.gRS    = env.lfac/env.lfacRS
  env.tRS    = derive_RS_crosstime(env.beta4, env.D04, env.betaRS)
  env.D3f    = derive_shellwidth_crosstime(env.D04, env.lfac4, env.lfac, env.lfac34)
  env.Ei3f   = derive_Eint_crosstime(env.M4, env.u, env.lfac34)
  env.Ek3f   = derive_Ek_crosstime(env.M4, env.lfac)
  env.eff_4  = env.Ei3f/env.Ek4

  # rarefaction waves
  T2     = env.p_sh/(env.rho2*c_**2)
  betas2 = derive_cs_fromT(T2)
  T3     = env.p_sh/(env.rho3*c_**2)
  betas3 = derive_cs_fromT(T3)
  env.betaRFm2 = (env.beta-betas2)/(1-env.beta*betas2)
  env.betaRFp2 = (env.beta+betas2)/(1+env.beta*betas2)
  env.betaRFm3 = (env.beta-betas3)/(1-env.beta*betas3)
  env.betaRFp3 = (env.beta+betas3)/(1+env.beta*betas3)

  env.tRFp3 = env.D3f / (c_*(env.betaRFp3-env.beta))
  env.tRFp2 = (env.betaFS-env.beta)*(env.tRS+env.tRFp3)/(env.betaRFp2-env.betaFS)
  env.tRFm2 = env.D2f / (c_*(env.beta-env.betaRFm2))
  env.tRFm3 = (env.beta-env.betaRS)*(env.tFS+env.tRFm2)/(env.betaRS-env.betaRFm3)

  # global thermal efficiency
  env.eff_th = env.eff_1 * env.t1/(env.t1+env.t4) + env. eff_4 * env.t4/(env.t1+env.t4)
  #add_weightfactors(env)

def shells_add_radNorm(env, z=1., dL=2e28):
  '''
  Add the normalization values for emission curves
  Minu paper equations G11, 15, 17
  ''' 
  env.z = z
  #env.u34 = derive_proper_from_Lorentz(env.lfac34)
  env.beta34 = derive_velocity(env.lfac34)
  env.u21 = derive_proper_from_Lorentz(env.lfac21)
  env.beta21 = derive_velocity(env.lfac21)
  Hau = (env.lfac34-1) * env.u34 / (1+env.a_u**2)
  L = ((env.Ek1/env.t1)+(env.Ek4/env.t4))/2
  Gp = (env.psyn-2)/(env.psyn-1)
  Wp = 2./Gp
  env.eint2p = 4.*env.lfac21*(env.lfac21-1)*env.rho1*c_**2
  env.eint3p = 4.*env.lfac34*(env.lfac34-1)*env.rho4*c_**2
  env.ad_gma3 = (4*env.lfac34+1)/(3*env.lfac34)
  env.ad_gma2 = (4*env.lfac21+1)/(3*env.lfac21)
  env.Bp = np.sqrt(8.*pi_*env.eps_B*env.eint3p)
  env.BpFS = np.sqrt(8.*pi_*env.eps_B*env.eint2p)
  env.gma_m = (mp_/me_)*Gp*(env.eps_e/env.xi_e)*(env.lfac34-1)
  env.gma_mFS = (mp_/me_)*Gp*(env.eps_e/env.xi_e)*(env.lfac21-1)
  env.gma_max = np.sqrt(3*e_/(sigT_*env.Bp))
  env.gma_maxFS = np.sqrt(3*e_/(sigT_*env.BpFS))
  env.nuBp = e_*env.Bp/(2.*pi_*me_*c_)
  env.nuBpFS = e_*env.BpFS/(2.*pi_*me_*c_)
  env.nu0p = env.gma_m**2 * env.nuBp
  env.nu0pFS = env.gma_mFS**2 * env.nuBpFS
  env.eps_rad = 1
  env.eps_radFS = 0.5
  env.Lbolp = (4/3.) * env.eps_e * env.eps_rad * (4*pi_*env.R0**2) * c_**3 * \
     (env.lfac34-1)*env.u34 * env.rho4 
  env.L0p = env.Lbolp / (Wp*env.nu0p)
  env.L0 = 2*env.lfac*env.L0p

  env.nu0 = 2.*env.lfac*env.nu0p/(1+z)
  env.T0 = (1+z) * (env.R0/c_) * ((1-env.betaRS)/env.betaRS)
  env.zdl = (1+z) / (4*pi_*dL**2)
  env.Fs = env.zdl * env.L0
  env.F0 = env.Fs/3.
  env.nu0F0 = env.nu0*env.F0
  env.tL0 = (3*env.lfac*env.dr4/(c_*env.T0))*env.L0p

  Pfac = (4./3.) * (4*(env.psyn-1)/(3*env.psyn-1)) * (16*me_*c_**2*sigT_/(18*pi_*e_))
  env.Pmax0 = Pfac * env.Bp * env.xi_e * (env.rho3/mp_)
  env.Pmax0FS = Pfac * env.BpFS * env.xi_e * (env.rho2/mp_)
  env.tc = 3*me_*c_*env.lfac/(4*sigT_*env.eint3p*env.eps_B*env.gma_m)
  env.tcFS = 3*me_*c_*env.lfac/(4*sigT_*env.eint2p*env.eps_B*env.gma_mFS)

  # for the FS
  # ratio_1 = (env.lfac34+1)/(env.lfac21+1)
  # ratio_2 = env.lfac34/env.lfac21
  # ratio_3 = (env.lfac34-1)/(env.lfac21-1)
  # ratio_freq = (ratio_1**(-0.5))*(ratio_2**(0.5))*(ratio_3**2)
  # ratio_bol = (ratio_1**(-1))*(ratio_2)*ratio_vel
  # ratio_lum = (ratio_freq**(-1))*(ratio_bol)
  env.fac_nu = ((env.lfac34+1)/(env.lfac21+1))**-0.5 * (env.lfac34/env.lfac21)**0.5 * ((env.lfac34-1)/(env.lfac21-1))**2
  env.fac_F  = ((env.lfac34+1)/(env.lfac21+1))**-0.5 * (env.lfac34/env.lfac21)**0.5 * ((env.lfac34-1)/(env.lfac21-1))**-2 * \
      (env.beta34/env.beta21) #* (env.eps_rad/env.eps_radFS)
  env.slope_mid = np.log10(env.fac_F*env.fac_nu)/np.log10(env.fac_nu)
  env.fac_Lp = env.fac_F
  env.fac_T  = ((env.lfacFS/env.lfacRS)**2)*(env.betaFS/env.betaRS)*((1+env.betaFS)/(1+env.betaRS))
  env.nu0pFS = env.nu0p / env.fac_nu
  env.nu0FS = env.nu0 / env.fac_nu
  env.L0pFS = env.L0p / env.fac_F
  env.L0FS  = env.L0 / env.fac_F
  env.F0FS  = env.F0 / env.fac_F
  env.nu0F0FS = env.nu0FS*env.F0FS
  env.T0FS  = (1+z) * (env.R0/c_) * ((1-env.betaFS)/env.betaFS)
  env.tL0FS = (3*env.lfac*env.dr1/(c_*env.T0FS))*env.L0pFS
  env.Ts   = (1+z)*(env.t0 - env.R0/c_)

  # final radii and times
  env.dRRS = env.tRS*env.betaRS*c_
  env.RfRS = env.R0 + env.dRRS
  env.RfRS0 = env.RfRS/env.R0
  env.TRS  = (1+env.z)*(env.t0+env.tRS-env.RfRS/c_)
  env.tTfRS = env.RfRS/env.R0

  env.dRFS = env.tFS*env.betaFS*c_
  env.RfFS = env.R0 + env.dRFS
  env.RfFS0 = env.RfFS/env.R0
  env.TFS  = (1+env.z)*(env.t0+env.tFS-(env.R0+env.betaFS*env.tFS*c_)/c_)
  env.tTfFS = env.RfFS/env.R0
  
  # normalized times (eqn F38)
  env.nTfRS = env.betaRS * c_ * env.tRS / env.R0
  env.nTfFS = ((env.betaRS*(1+env.betaRS))/(env.betaFS*(1+env.betaFS))) * (env.lfacRS/env.lfacFS)**2 * env.betaFS * c_ * env.tFS / env.R0

  # effective ejection timescales (eqn E3, E4)
  env.teffejRS = (1+z)*(env.t0 - env.R0/(env.betaRS*c_))
  env.TeffejRS = (1+z)*env.teffejRS
  env.teffejFS = (1+z)*(env.t0 - env.R0/(env.betaFS*c_))
  env.TeffejFS = (1+z)*env.teffejFS

  # equivalent luminosity
  env.DV0 = pi_*env.R0**2*env.D3f/(env.Nsh4*env.lfac34)
  env.DV0FS = pi_*env.R0**2*env.D2f/(env.Nsh1*env.lfac21)
  env.tL0 = ((1+z)/env.T0)*env.lfac*env.DV0*env.eps_e*env.eint3p/(Wp*env.nu0p)
  env.tL0FS = ((1+z)/env.T0FS)*env.lfac*env.DV0FS*env.eps_e*env.eint2p/(Wp*env.nu0pFS)
  #env.tL0 = (env.eps_e*env.eint3p/(Wp*env.nu0p)) * env.lfac*c_ * pi_*env.R0*env.D04/(env.lfac34*env.Nsh4)
  env.dV04 = 4*pi_*env.R0**2*(env.D04/env.Nsh4)
  env.dV01 = 4*pi_*env.R0**2*(env.D01/env.Nsh1)
  env.dEp0 = env.eps_e*(env.lfac34-1)*c_**2*env.lfac*env.dV04*env.rho4/(Wp*env.nu0p)
  env.dEp0FS = env.eps_e*(env.lfac21-1)*c_**2*env.lfac*env.dV01*env.rho1/(Wp*env.nu0pFS)

  # some more useful values for rad calculations
  env.Gammap = - gamma(1/3.-env.psyn)*gamma(env.psyn)/gamma(1/3.)

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
    prs[i_3] = env.p_sh
  else:
    i_3 = np.argwhere((r>=Rrs) & (r<=Rcd))[:,0]
    rho[i_3] = env.rho3
    vel[i_3] = env.u
    prs[i_3] = env.p_sh
    print("RS crossing time passed, need to implement rarefaction wave")
  
  if t < env.tFS:
    i_1 = np.argwhere((r>Rfs) & (r<=R1))[:,0]
    rho[i_1] = env.rho1
    vel[i_1] = env.u1
    i_2 = np.argwhere((r>Rcd) & (r<=Rfs))[:,0]
    rho[i_2] = env.rho2
    vel[i_2] = env.u
    prs[i_2] = env.p_sh
  else:
    i_2 = np.argwhere((r>Rcd) & (r<=Rfs))[:,0]
    rho[i_2] = env.rho2
    vel[i_2] = env.u
    prs[i_2] = env.p_sh
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
  critlines = [chi1, chi2, chi3, chi4, chi5]
  case = np.searchsorted(critlines, env.chi) + 1

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

  Rcd = np.array(env.R0 + env.beta*c_*t)
  Rb  = [np.where(t<env.tRS, Rb0[n]+vb0[n]*t, Rb0[n]+vb0[n]*env.tRS+vbc[n]*(t-env.tRS))
    for n in [0, 1]]
  Rf  = [np.where(t<env.tFS, Rf0[n]+vf0[n]*t, Rf0[n]+vf0[n]*env.tFS+vbc[n]*(t-env.tFS))
    for n in [0, 1]]
  R4, Rrs = Rb
  Rfs, R1 = Rf
  return np.array([R4, Rrs, Rcd, Rfs, R1])


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
  elif varkey in ['W', 'pdV']:
    p4, p1 = 0., 0.
    v4, v1 = 0., 0.
    A4, A1 = [4.*pi_*env.R0**2 for i in [0,1]]
    if env.geometry == 'spherical':
      R4, Rrs, Rcd, Rfs, R1 = derive_radii_withtime(env, t)
      A4 = 4.*pi_*R4**2
      A1 = 4.*pi_*R1**2
    pdV4 = pdV_rate(v4, p4, A4)*c_*np.ones(len(t))
    pdV1 = pdV_rate(v1, p1, A1)*c_*np.ones(len(t))
    if varkey == 'pdV':
      return pdV4, pdV1
    elif varkey == 'W':
      W4 = pdV4*t
      W1 = pdV1*t
      return W4, W1
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
    #E4, E3, E2, E1 = np.array(derive_Ekin_withtime(env, t)) + np.array(derive_Eint_withtime(env, t))
    Etot = (env.Ek1+env.Ek4)*np.ones(t.shape) #E4+E3+E2+E1
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
    eint = derive_Eint_comoving(rho*env.rhoscale*c_**2, p*env.rhoscale*c_**2)
    return eint
  elif var == "Ekin":
    ek = derive_Ekin(rho, u)
    return ek
  elif var == "Emass":
    return rho
  elif var == "B":
    eint = derive_Eint_comoving(rho*env.rhoscale*c_**2, p*env.rhoscale*c_**2)
  return np.sqrt(env.eps_B * eint * 8 * pi_)

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
  lfac1  = derive_Lorentz_from_proper(u1)
  lfac   = derive_Lorentz_from_proper(u)
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
  lfac   = derive_Lorentz_from_proper(u)
  beta4  = u4/lfac4
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
  (M0, lfacrel) = (M1, lfac21) for shell 2 ; (M4, lfac34) for shell 3
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
