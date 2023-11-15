# -*- coding: utf-8 -*-
# @Author: acharlet

'''
This file contains the functions necessary to compute the electron distribution
'''

import numpy as np
from phys_constants import *
from scipy.signal import unit_impulse

def edistrib_plaw_vec(gmin0, gmax0, gmin1, gmax1):
  #fc = (gmin0 >= gmax1)
  g1 = np.where(gmin0 >= gmax1, gmax1, gmin1)
  g2 = np.where(gmin0 >= gmax1, gmin0, gmax1)
  g3 = np.where(gmin0 >= gmax1, gmax0, gmax0)
  return g1, g2, g3

def edistrib_plaw(gmin0, gmax0, gmin1, gmax1):
  if gmin0 >= gmax1:
    return gmax1, gmin0, gmax0
  else:
    return gmin1, gmax1, gmax0

def cell_edistrib(gmin0, gmax0, gmin1, gmax1, p):
  '''
  Returns electron distribution at time t depending on gamma_min and max
  of times t and t-dt
  '''
  def distrib(gammae):
    return 0.

  if gmin0 >= gmax1:
    # fast cooling
    # gmax1 >> -2 >> gmin0 >> -p-1 >> gmax0
    def distrib(gammae):
      return broken_plaw(gammae, gmax1, gmin0, gmax0, 2, p+1)
    
  else:
    # slow cooling 
    # gmin1 >> -p >> gmax1 >> -p-1 >> gmax0
    def distrib(gammae):
      return broken_plaw(gammae, gmin1, gmax1, gmax0, p, p+1)
  
  return distrib


def broken_plaw(x, g1, g2, g3, a1, a2):
  '''
  Powerlaw with index a1 between g1 and g2, a2 between g2 and g3
  '''
  try:
    n = np.where(x < g1, 0., 
      np.where(x < g2, (x/g1)**(-a1),
        np.where(x < g3, ((g2/g1)**(-a1))*(x/g2)**(-a2), 0.)))
    return n
  except TypeError:
    if x < g1: return 0.
    elif x < g2: return (x/g1)**(-a1)
    elif x < g3: return ((g2/g1)**(-a1))*(x/g2)**(-a2)
    else: return 0.
