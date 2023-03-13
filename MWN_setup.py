# -*- coding: utf-8 -*-
# @Author: acharlet

'''
Update the setup in the relevant Initial/ file by reading the phys_inputs file
'''

# Imports
# --------------------------------------------------------------------------------------------------
import subprocess
import pathlib
import numpy as np
import sys
sys.path.insert(1, 'bin/Tools/')
from environment import *
from models_MWN import R_1st

simFile = './src/Initial/MWN/MWN1D_tests.cpp'

def main():
  # update MWN1D.cpp and copies phys_input.MWN in the results folder
  # to come: add file check and automatic moving results in new folder
  update_simFile(simFile)
  subprocess.call("cp -f ./phys_input.MWN ./results/Last/", shell=True)
  #run_name = get_runName("./phys_input.MWN")
  

def update_simFile(filepath):
  env = MyEnv()
  env.setupEnv('./phys_input.MWN')

  R_b = R_1st(env.t_start, env.L_0, env.t_0)                  # nebula radius
  R_c = v_t*env.t_start                     # ejecta core radius
  R_e = R_c/wc                              # ejecta envelope radius
  beta_w = np.sqrt(1. - env.lfacwind**(-2))
  rho_w  = env.L_0/(4. * pi_*env.rmin0**2 * env.lfacwind**2 * c_**3 * beta_w)
  rho_ej = D / env.t_start**3

  vars2update = {
    'rmin0':env.rmin0, 'rmax0':env.rmax0, 'R_b':R_b, 'R_c':R_c, 'R_e':R_e,
    'rho_w':rho_w, 'beta_w':beta_w, 'rho_ej':rho_ej, 'rho_csm':rho_csm
  }
  out_lines = []
  with open(filepath, 'r') as inf:
    inFile = inf.read().splitlines()
    # copy file line by line with updated values where needed
    for line in inFile:
      if line.startswith('static double'):
        l = line.split()
        if l[2] in vars2update:
          line = line.replace(l[4], f'{vars2update[l[2]]:.4e}')
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
