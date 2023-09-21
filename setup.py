# -*- coding: utf-8 -*-
# @Author: acharlet

'''
Update the setup in the relevant Initial/ file by reading the phys_inputs file
'''

# Imports
# --------------------------------------------------------------------------------------------------
import subprocess
from pathlib import Path
import os
import numpy as np
import sys
sys.path.insert(1, 'bin/Tools/')
from environment import MyEnv

Initial_path = str(Path().absolute() / 'src/Initial/')

def main():
  # update .cpp file and copies phys_input.ini in the results folder
  # to come: add file check and automatic moving results in new folder
  env = MyEnv('./phys_input.ini')
  
  if env.mode == 'shells':
    simfile = Initial_path + '/Shells/Shells.cpp'
  elif mode == 'MWN':
    simfile = Initial_path + '/MWN/MWN1D_tests.cpp'
  update_simFile(simfile, env)
  update_Makefile(env)
  update_envFile(env)
  subprocess.call("cp -f ./phys_input.ini ./results/Last/", shell=True)
  #run_name = get_runName("./phys_input.ini")

def update_envFile(env):
  '''
  Update environment file to go with geometry
  we found shock strength threshold was to be changed between cartesian and spherical
  '''
  filepath = os.getcwd() + '/src/environment.h'
  ch_sh = 0.05
  if env.geometry == 'cartesian':
    chi_sh = 0.1
  elif env.geometry == 'spherical':
    chi_sh = 0.05
  
  out_lines = []
  with open(filepath, 'r') as f:
    inFile = f.read().splitlines()
    # copy file line by line with updated values where needed
    for line in inFile:
      if line.startswith('#define'):
        l = line.split()
        if l[1] == 'DETECT_SHOCK_THRESHOLD_':
          line = line.replace(l[2], str(chi_sh))
      out_lines.append(line)
  
  with open(filepath, 'w') as outf:
    outf.writelines(f'{s}\n' for s in out_lines)

def update_Makefile(env):
  '''
  Update Makefile with given parameters (especially geometry)
  '''
  filepath = './Makefile'
  geometry = env.geometry
  mode = env.mode
  compile_cart = {
    'GEOMETRY':'cartesian', 'HYDRO':'rel_cart', 'RADIATION':'radiation_cart'
  }
  compile_sph = {
    'GEOMETRY':'spherical1D', 'HYDRO':'rel_sph', 'RADIATION':'radiation_sph'
  }
  compile_options = {'cartesian':compile_cart, 'spherical':compile_sph}

  out_lines = []
  with open(filepath, 'r') as inf:
    inFile = inf.read().splitlines()
    # copy file line by line with updated values where needed
    comp_geom = compile_options[geometry]
    for line in inFile:
      if line:
        l = line.split()
        if l[0] == 'INITIAL':
          name = 'Shells/Shells' if mode == 'shells' else 'MWN/MWN1D_tests'
          line = line.replace(l[2], name)
        elif l[0] in comp_geom.keys():
          line = line.replace(l[2], comp_geom[l[0]])
      out_lines.append(line)
  
  with open(filepath, 'w') as outf:
    outf.writelines(f'{s}\n' for s in out_lines)

def update_simFile(filepath, env):
  gridvars = {
    'Ncells':env.Ncells, 'rhoNorm':env.rhoNorm,
    'itmax':env.itmax, 'geometry':env.geometry
  }
  physvars = {}
  if env.mode == 'MWN':
    physvars = {
      'R_b':env.R_b, 'R_c':env.R_c, 'R_e':env.R_e, 'Theta0':env.Theta0,
      'rho_w':env.rho_w, 'lfacwind':env.lfacwind, 'rho_ej':env.rho_ej, 'rho_csm':rho_csm,
      't_sd':env.t_0, 't_start':env.t_start, 'm_fade':m, 'rmin0':env.rmin0, 'rmax0':env.rmax0
    }
  elif env.mode =='shells':
    physvars = {
      'R_0':env.R0, 't_start':env.t0, #'rho0':env.rho0
      'rho1':env.rho1, 'u1':env.u1, 'p1':env.p1, 'D01':env.D01,
      'rho4':env.rho4, 'u4':env.u4, 'p4':env.p4, 'D04':env.D04,
      'rmin0':(env.R0 - 1.05*env.D04),
      'rmax0':(env.R0 + 1.1*env.D01)
    }
  vars2update = {**gridvars, **physvars}
  out_lines = []
  with open(filepath, 'r') as inf:
    inFile = inf.read().splitlines()
    # copy file line by line with updated values where needed
    for line in inFile:
      if line.startswith('static'):
        l = line.split()
        var = l[2]
        if var == 'GEOMETRY_':
          line = line.replace(l[4], '1' if env.geometry=='spherical' else '0')
        if var in vars2update:
          if var in ['lfacwind', 'u1', 'u4']:
            line = line.replace(l[4], f'{vars2update[l[2]]:.0e}')
          elif var == 'm_fade':
            line = line.replace(l[4], f'{vars2update[l[2]]:.1f}')
          elif var == 'Ncells':
            line = line.replace(l[4], f'{int(vars2update[l[2]]):d}')
          elif var == 'rhoNorm':
            line = line.replace(l[4], env.rhoNorm)
          else:
            line = line.replace(l[4], f'{vars2update[l[2]]}')
      if 'if ( it >' in line:
        l = line.split()
        line = line.replace(l[4], f'{int(env.itmax):d}')
      out_lines.append(line)
  
  with open(filepath, 'w') as outf:
    outf.writelines(f'{s}\n' for s in out_lines)

def get_runName(filename):
  with open(filename, 'r') as inf:
    inFile = inf.read().splitlines()
    for line in inFile:
      if line.startswith('run name'):
        l = line.split()
        runName = l[2]
  
  return runName

if __name__ == "__main__":
    main()
