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
import copy
import numpy as np
from scipy.optimize import fsolve
from phys_constants import *
from phys_functions_shells import shells_complete_setup, shells_add_analytics, shells_add_radNorm

default_path = str(Path().absolute().parents[1] / 'phys_input.ini')
Initial_path = str(Path().absolute().parents[1] / 'src/Initial/')
cwd = os.getcwd().split('/')
iG = [i for i, s in enumerate(cwd) if 'GAMMA' in s][0]
GAMMA_dir = '/'.join(cwd[:iG+1])


##### Updating the environment and fetching variables
# changing one input variable
def update_env(name, value=None, env=None):
  '''
  Update environment after changing one (or more) of the 'input' variables
    (variables defined in phys_input.ini)
    - single variable:  update_env('u1', 12., env)
    - several at once:   update_env(['u1', 'u4'], [12., 24.], env)
    - several at once:   update_env({'u1': 12., 'u4': 24.}, env=env)
  '''
  if isinstance(name, dict):
    updates = name
  else:
    name_isarr = ((type(name) == list) or (type(name) == np.ndarray))
    value_isarr = ((type(value) == list) or (type(value) == np.ndarray))
    if (name_isarr and value_isarr):
      updates = dict(zip(name, value))
    else:
      updates = {name: value}
  for n, v in updates.items():
    setattr(env, n, v)
  # drop derived geometry so shells_complete_setup re-derives it from the
  # primary inputs: R0 from t0 (itself from toff), D0i from ti. Otherwise the
  # 'if A and not B' guards keep the stale values when u1/u4 change.
  derived = {'R0': 'toff', 'D01': 't1', 'D04': 't4'}
  for name, primary in derived.items():
    if hasattr(env, primary) and hasattr(env, name):
      delattr(env, name)
  shells_complete_setup(env, 0.)
  shells_add_analytics(env)
  shells_add_radNorm(env)
  env.add_code_norms()   # rhoscale changed -> code-unit constants change

def rescale_proper_velocities(factor, env):
  '''
  Rescale the proper velocities u1, u4 by 'factor', keeping their ratio
    a_u = u4/u1 (and everything else) constant.
  Returns a new environment object, leaving the input env untouched.
  '''
  new_env = copy.deepcopy(env)
  update_env({'u1': new_env.u1 * factor, 'u4': new_env.u4 * factor}, env=new_env)
  return new_env

def rescale_hydro(alpha, zeta, env):
  '''
  Granot (2012) hydrodynamic unit-rescaling of a fixed physical solution:
    lengths & times x alpha, energy & mass x zeta  (=> rho, p x zeta*alpha^-3,
    velocities/Lorentz factors invariant). Returns a new env; input untouched.
  Composes with rescale_proper_velocities (disjoint inputs: u1,u4 vs toff,t1,t4,L).
  '''
  new_env = copy.deepcopy(env)
  update_env({'toff': new_env.toff * alpha,
              't1':   new_env.t1   * alpha,
              't4':   new_env.t4   * alpha,
              'L':    new_env.L    * zeta / alpha}, env=new_env)
  return new_env

def fetch_value_updated(name, name_var, value_var, env):
  '''
  Get the new value of quantity 'name' after updating the env
  '''
  update_env(name_var, value_var, env)
  value = getattr(env, name)
  return value

# rescaling
def get_physfile(key):
  '''
  Returns path of phys_input file of the corresponding results folder
  '''
  # allow passing a literal existing file path (e.g. a rescaled temp input)
  if os.path.isfile(key):
    return key
  dir_path = GAMMA_dir + '/results/%s/' % (key)
  file_path = dir_path + "phys_input.ini"
  if os.path.isfile(file_path):
    return file_path
  else:
    return GAMMA_dir + "/phys_input.ini"

##### Environment class containing all analytics
class MyEnv:
  def __init__(self, key, scalefac=0.):
    path = get_physfile(key)
    self.read_input(path)
    self.create_setup(scalefac)
  
  def read_input(self, path):
    '''
    Reads GAMMA/phys_input.ini file and puts value in the MyEnv class
    '''
    strinputs = ['mode', 'runname', 'rhoNorm', 'geometry', 'stop']
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

  def create_setup(self, scalefac):
    '''
    Derive values for the numerical setup from the physical inputs
    '''
    varlist = []
    vallist = []
    if self.mode == 'shells':
      shells_complete_setup(self, scalefac)
      shells_add_analytics(self)
      shells_add_radNorm(self)
    elif self.mode == 'MWN':
      # varlist, vallist = MWN_phys2num()
      pass
    for name, value in zip(varlist, vallist):
      setattr(self, name, value)
    self.add_code_norms()

  def add_code_norms(self):
    '''
    Scaling constants from GAMMA code units; must be refreshed
    whenever rhoscale changes (e.g. after update_env on u1/u4)
    '''
    rhoNorm = self.rhoscale
    vNorm = c_
    lNorm = vNorm
    self.Nalpha_ = alpha_ / (1. / (rhoNorm * vNorm * lNorm))
    self.Nmp_ = mp_ / (rhoNorm * lNorm**3)
    self.Nme_ = me_ / (rhoNorm * lNorm**3)
    self.Nqe_ = e_ / (np.sqrt(rhoNorm) * lNorm**2. * vNorm)
    self.NsigmaT_ =  sigT_ / lNorm**2.
  
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