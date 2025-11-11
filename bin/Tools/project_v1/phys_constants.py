# -*- coding: utf-8 -*-
# @Author: acharlet

'''
This file contains physical constants in CGS units
'''
import numpy as np

# Constants
c_    = 2.99792458e10           # cm/s: light speed 
mp_   = 1.67262192e-24          # g: proton mass
Msun_ = 1.9891e33               # g: solar mass
pi_   = np.pi
e_    = 4.80320425e-10          # Fr (cm^3/2 g^1/2 s^-1): charge unit
me_   = 9.10938291e-28          # g: electron mass
rme_  = e_*c_**2                # electron rest-mass
h_    = 6.62607004e-27          # cm2.g.s-1: Planck constant 
h_eV_ = 4.135667696e-15         # conversion factor Hz -> eV
kB_   = 1.380649e-16            # erg.K-1
sigT_ = 6.652457318e-25         # cm2: Thomson cross-section
alpha_= 1.29251816e-09          # sigmaT_/6 pi_me_c_ (Van Eerten+2010)

### Normalization of spectral shape R(x) 
# cf Finke et al. 2009 eqns 18 - 20
norm_R_ = 1.0751412667993323