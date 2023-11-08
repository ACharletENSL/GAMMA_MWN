# -*- coding: utf-8 -*-
# @Author: acharlet

'''
This file contains the functions necessary to compute the electron distribution
'''

import numpy as np
from phys_constants import *
from scipy.signal import unit_impulse

def edistrib_plaw_vec(gmin0, gmax0, gmin1, gmax1):
  g1 = np.where(gmin0 >= gmax1, gmax1, gmin1)
  g2 = np.where(gmin0 >= gmax1, 0., gmax1)
  g3 = np.where(gmin0 >= gmax1, 0., gmax0)
  return g1, g2, g3

def edistrib_plaw(gmin0, gmax0, gmin1, gmax1):
  if gmin0 >= gmax1:
    return gmax1, 0., 0.
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
    # very strong cooling, resultant distrib is delta function
    def distrib(gammae):
      try:
        n = len(gammae)
        i = np.searchsorted(gammae, gmax1)
        return unit_impulse(n, i)
      except TypeError:  # it's a float
        return float(np.isclose(gammae, gmax1))
  
  else:
    # cooling adds another power-law of index -p-1 between gmax1 and gmax0
    def distrib(gammae):
      try:
        n = np.where(gammae < gmin1, 0., 
          np.where(gammae < gmax1, gammae**(-p),
          np.where(gammae < gmax0, gammae**(-p-1), 0.)))
        return n
      except TypeError:
        if gammae < gmin1: return 0.
        elif gammae < gmax1: return gammae**(-p)
        elif gammae < gmax0: return gammae**(-p-1)
        else: return 0.
  
  return distrib


