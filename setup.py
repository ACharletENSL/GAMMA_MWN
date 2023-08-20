# -*- coding: utf-8 -*-
# @Author: acharlet

'''
Update the setup in the relevant Initial/ file by reading the phys_inputs file
'''

# Imports
# --------------------------------------------------------------------------------------------------
import subprocess
from pathlib import Path
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
    simFile = Initial_path + '/Shells/Shells_v1.cpp'
  elif mode == 'MWN':
    simFile = Initial_path + '/MWN/MWN1D_tests.cpp'
  update_simFile(simFile, env)
  subprocess.call("cp -f ./phys_input.ini ./results/Last/", shell=True)
  #run_name = get_runName("./phys_input.ini")

def update_simFile(filepath, env):
  gridvars = {
    'Ncells':env.Ncells, 'rhoNorm':env.rhoNorm, 'itmax':env.itmax
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
      'R0':env.R0, 't_start':env.t0, #'rho0':env.rho0
      'rho1':env.rho1, 'u1':env.u1, 'p1':env.p1, 'D01':env.D01,
      'rho4':env.rho4, 'u4':env.u4, 'p4':env.p4, 'D04':env.D04,
      'rmin0':(env.R0 - 1.1*env.D04) if env.external else (env.R0 - env.D04),
      'rmax0':(env.R0 + 1.1*env.D01) if env.external else (env.R0 + env.D01)
    }
  vars2update = {**gridvars, **physvars}
  out_lines = []
  with open(filepath, 'r') as inf:
    inFile = inf.read().splitlines()
    # copy file line by line with updated values where needed
    for line in inFile:
      if line.startswith('static double'):
        l = line.split()
        var = l[2]
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
            line = line.replace(l[4], f'{vars2update[l[2]]:.4e}')
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
