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
from phys_functions_shells import shells_phys2num

default_path = str(Path().absolute().parents[1] / 'phys_input.ini')
Initial_path = str(Path().absolute().parents[1] / 'src/Initial/')

class MyEnv:
  def setupEnv(self, path):
    self.read_input(path)
    self.create_setup()
  
  #def setup_fromCode(self, mode):
  #  if mode == 'MWN':
  #    path = Initial_path + 'MWN/MWN1D_tests.cpp'
  #  elif mode == 'shells':
  #    path = Initial_path + 'Shells/Shells_v1.cpp'
    
    #with open(path, 'r') as f:
    #  lines = f.read().splitlines()
    #  lines = filter(lambda x: x.startswith('static double') and len(x)>0, lines)

  
  def read_input(self, path):
    '''
    Reads GAMMA/phys_input.ini file and puts value in the MyEnv class
    '''
    with open(path, 'r') as f:
      lines = f.read().splitlines()
      lines = filter(lambda x: not x.startswith('#') and len(x)>0, lines)
      physvars = []
      for line in lines:
        name, value = line.split()[:2]
        if name not in ['mode', 'runname', 'boundaries']:
          value = float(value)
        setattr(self, name, value)
  
  def create_setup(self):
    '''
    Derive values for the numerical setup from the physical inputs
    '''
    varlist = []
    vallist = []
    if self.mode == 'shells':
      if 'L1' in self.__dict__.keys():
        varlist, vallist = shells_phys2num(self.L1, self.t1, self.u1,
          self.L4, self.t4, self.u4, self.toff, self.Theta0)
      elif 'rho1' in self.__dict__.keys():
        p0 = min(self.rho1, self.rho4)*self.Theta0*c_**2
        self.p1 = p0
        self.p4 = p0
        self.rmin0 = self.R_0 - self.D04
        self.rmax0 = self.R_0 + self.D01
        self.t_0   = self.R_0/c_
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
        t_scale, r_scale, rho_scale = self.t_0, self.R_0, self.rho1
        t_scale_str, r_scale_str = ('t_0', 'R_0')
      elif self.mode == 'PWN':
        pass
    
    return t_scale, r_scale, rho_scale, t_scale_str, r_scale_str