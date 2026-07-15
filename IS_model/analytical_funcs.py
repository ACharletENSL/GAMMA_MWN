# -*- coding: utf-8 -*-
# @Author: acharlet


'''
Contains analytical functions used in the peaks models
  - relLfac_from_au
  - g2_from_au
  - tau_to_dRR0
  - ratios_RSvFS_from_au
  - sound_speed_TM
  - chis_crit
'''

import numpy as np

def relLfac_from_au(au):
  '''
  Relative Lorentz factors at each shock front
  '''

  K = np.sqrt(2 / (1 + au*au))
  G21 = .5 * (K*au + 1/(K*au))
  G34 = .5 * (K + 1/K)
  return G34, G21

def g2_from_au(au):
  '''
  Ratios of downstream L.F. to shock front L.F., squared
  '''

  K = np.sqrt(2 / (au**2+1))
  G34, G21 = relLfac_from_au(au)
  gFS2 = (K*au - 4*G21)/(1/(K*au) - 4*G21)
  gRS2 = (4*G34 - K)/(4*G34 - 1/K)
  return gRS2, gFS2

def tau_to_dRR0(au):
  '''
  Proportionality factor between the crossed radius by the shock in units R0
    and the ratio of activity time over off time for the source,
    for constant velocity shock fronts
  '''

  au2 = au*au
  gRS2, gFS2 = g2_from_au(au)
  KFS = 2 * (au2 - 1) / (2*au2 - (au2 + 1)*gFS2)
  KRS = 2 * (au2 - 1) / ((au2 + 1)*gRS2 - 2)
  return KRS, KFS

def ratios_RSvFS_from_au(au):
  '''
  Ratios between normalizations of RS and FS
  '''
  G34, G21 = relLfac_from_au(au)
  b21 = np.sqrt(1. - G21**-2)
  b34 = np.sqrt(1. - G34**-2)
  gRS2, gFS2 = g2_from_au(au)

  ratio_T = (gRS2/gFS2)
  fac1 = np.sqrt((G34/G21)*((G21+1)/(G34+1)))
  fac2 = ((G34-1)/(G21-1))**2
  ratio_nu = fac1 * fac2
  ratio_F = fac1 * (b34/b21) / fac2
  return ratio_T, ratio_nu, ratio_F

def sound_speed_TM(Gij):
  '''
  Comoving sound speed in a region shocked at relative L.F. Gij,
    TM EoS (Mathews 71), cf. R24a eqs. (H1)-(H3)
  '''
  Th = (Gij*Gij - 1.) / (3.*Gij)
  rt = np.sqrt(Th*Th + 4./9.)
  bs2 = (5.*Th*rt + 3.*Th*Th) / (12.*Th*rt + 12.*Th*Th + 2.)
  return np.sqrt(bs2)

def chis_crit(au):
  '''
  Critical initial width ratios chi_cX = Delta_{1,0}/Delta_{4,0} = ton1/ton4
    = tau1/tau4, X = 1..5, from R24a Table 7 (planar geometry),
    for constant engine power (f = au^-2) and ultrarelativistic shells.
  All velocity differences are scaled by 2 Gamma^2, which cancels in the
    chi_cX and avoids catastrophic cancellation at high Lorentz factor;
    the result then depends on au only.
  Returns (chi_c1, chi_c2, chi_c3, chi_c4, chi_c5), in decreasing order:
    FS fully crosses S1 iff chi <= chi_c1,
    RS fully crosses S4 iff chi >= chi_c5
  '''
  K = np.sqrt(2./(1. + au*au))
  G34, G21 = relLfac_from_au(au)
  # compression ratios Delta_jf/Delta_j0, eqs. (14a)-(14b); Gma/Gma1 = K au, Gma/Gma4 = K
  xFS = 1./(4.*G21*K*au)
  xRS = 1./(4.*G34*K)
  # d_* = 2 Gamma^2 (beta_a - beta_b)
  d_b1  = K*K*au*au - 1.    # beta - beta1
  d_4b  = 1. - K*K          # beta4 - beta
  d_FS1 = d_b1/(1. - xFS)   # betaFS - beta1, eq. (11a)
  d_4RS = d_4b/(1. - xRS)   # beta4 - betaRS, eq. (11b)
  d_bRS = d_4RS - d_4b      # beta - betaRS
  d_FSb = d_FS1 - d_b1      # betaFS - beta
  # rarefaction head speeds from relativistic velocity addition of beta_s
  bs2 = sound_speed_TM(G21)
  bs3 = sound_speed_TM(G34)
  d_2m = 2.*bs2/(1. - bs2)  # beta - beta_2rf-
  d_3m = 2.*bs3/(1. - bs3)  # beta - beta_3rf-
  d_2p = 2.*bs2/(1. + bs2)  # beta_2rf+ - beta
  d_3p = 2.*bs3/(1. + bs3)  # beta_3rf+ - beta

  chi_c3 = d_FS1/d_4RS
  chi_c2 = d_FS1 * (1./d_4RS + xRS/d_3p)
  chi_c1 = chi_c2 * (1. + d_FSb/(d_2p - d_FSb))
  chi_c4 = 1./(d_4RS * (1./d_FS1 + xFS/d_2m))
  chi_c5 = chi_c4 / (1. + d_bRS/(d_3m - d_bRS))
  return chi_c1, chi_c2, chi_c3, chi_c4, chi_c5