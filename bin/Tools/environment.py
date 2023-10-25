# -*- coding: utf-8 -*-
# @Author: acharlet

'''
This file read parameters and stores useful parameter-dependant quantities
to generate setup et scale plots
'''

# Imports
# --------------------------------------------------------------------------------------------------
from pathlib import Path
import os
import numpy as np
from scipy.optimize import fsolve
from phys_constants import *
from phys_functions_shells import shells_complete_setup, shells_add_analytics, shells_add_radNorm

default_path = str(Path().absolute().parents[1] / 'phys_input.ini')
Initial_path = str(Path().absolute().parents[1] / 'src/Initial/')

class MyEnv:
  def __init__(self, path):
    self.read_input(path)
    self.create_setup()
  
  def read_input(self, path):
    '''
    Reads GAMMA/phys_input.ini file and puts value in the MyEnv class
    '''
    strinputs = ['mode', 'runname', 'rhoNorm', 'geometry']
    intinputs = ['Ncells', 'itmax']
    with open(path, 'r') as f:
      lines = f.read().splitlines()
      lines = filter(lambda x: not x.startswith('#') and len(x)>0, lines)
      physvars = []
      for line in lines:
        name, value = line.split()[:2]
        if name in intinputs:
          value = int(value)
        elif name not in strinputs:
          value = eval(value)
        setattr(self, name, value)
  
  def create_setup(self):
    '''
    Derive values for the numerical setup from the physical inputs
    '''
    varlist = []
    vallist = []
    if self.mode == 'shells':
      shells_complete_setup(self)
      shells_add_analytics(self)
      #shells_add_radNorm(self)
    elif self.mode == 'MWN':
      # varlist, vallist = MWN_phys2num()
      pass
    for name, value in zip(varlist, vallist):
      setattr(self, name, value)
  
  def get_scalings(self, type):
    '''
    Returns t and R scaling and corresponding variable names
    '''
    t_scale, r_scale, rho_scale = (1., 1., 1.)
    t_scale_str, r_scale_str = ('', '')

    if type:
      if self.mode == 'shells':
        t_scale, r_scale, rho_scale = self.t0, self.R0, self.rho1
        t_scale_str, r_scale_str = ('t_0', 'R_0')
      elif self.mode == 'PWN':
        pass
    
    return t_scale, r_scale, rho_scale, t_scale_str, r_scale_str