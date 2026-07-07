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
sys.path.insert(1, 'bin/Tools/project_v2')
from environment import MyEnv

Initial_path = str(Path().absolute() / 'src/Initial/')

def main(alpha=1., zeta=1.):
  # update .cpp file and copies phys_input.ini in the results folder
  # to come: add file check and automatic moving results in new folder
  # alpha, zeta: Granot (2012) hydro unit-rescaling of the inputs before setup,
  # so the launched sim reproduces the fiducial rescaled a posteriori with
  # environment.rescale_hydro(alpha, zeta, env) (see rescale_input below)
  # always write the (possibly identity) rescaled input to a temp file, so the
  # setup below and the copy into results/Last both use the same, consistent
  # inputs (identity when alpha=zeta=1)
  rescale_input(alpha, zeta, dst='./phys_input_temp.ini')
  env = MyEnv('./phys_input_temp.ini')

  if env.mode == 'shells':
    simfile = Initial_path + '/Shells/Shells.cpp'
  elif mode == 'MWN':
    simfile = Initial_path + '/MWN/MWN1D_tests.cpp'
  update_simFile(simfile, env)
  update_Makefile(env)
  update_envFile(env)
  subprocess.call("mv ./phys_input_temp.ini ./results/Last/phys_input.ini", shell=True)
  #run_name = get_runName("./phys_input.ini")

def rescale_input(alpha, zeta, src='./phys_input.ini', dst='./phys_input.ini'):
  '''
  Granot (2012) hydrodynamic unit-rescaling of the phys_input.ini parameters,
  so a sim launched from the result equals the original rescaled a posteriori
  with environment.rescale_hydro(alpha, zeta, env). Scales dimensional inputs:
    length/time (toff,t0,t1,t4,R0,D01,D04) x alpha
    energy      (Ek1,Ek4)                  x zeta
    power       (L)                        x zeta/alpha
    density/pressure (rho1,rho4,p1,p4)     x zeta*alpha**-3
  Velocities (u1,u4) and dimensionless params (Theta0, rhoContr, eps_*, ...)
  are left untouched. Comments/layout preserved; only active lines edited.
  In place by default; pass a pristine src (e.g. results/<key>/phys_input.ini)
  to avoid compounding rescalings.
  '''
  factors = {
    'toff':alpha, 't0':alpha, 't1':alpha, 't4':alpha, 'R0':alpha, 'D01':alpha, 'D04':alpha,
    'Ek1':zeta, 'Ek4':zeta,
    'L':zeta/alpha,
    'rho1':zeta*alpha**-3, 'rho4':zeta*alpha**-3, 'p1':zeta*alpha**-3, 'p4':zeta*alpha**-3,
  }
  out_lines = []
  with open(src, 'r') as f:
    inFile = f.read().splitlines()
    for line in inFile:
      if line and not line.startswith('#'):
        head, sep, tail = line.partition('#')
        l = head.split()
        if len(l) >= 2 and l[0] in factors:
          newval = eval(l[1]) * factors[l[0]]
          newhead = head.replace(l[1], '%.12g' % newval, 1)
          line = newhead + (sep + tail if sep else '')
      out_lines.append(line)
  with open(dst, 'w') as outf:
    outf.writelines(f'{s}\n' for s in out_lines)
  print(f'rescale_input: alpha={alpha:g}, zeta={zeta:g}  ({src} -> {dst})')

def update_envFile(env):
  '''
  Update environment file to go with geometry
  we found shock strength threshold was to be changed between cartesian and spherical
  '''
  filepaths = [os.getcwd() + '/src/environment.h', os.getcwd() + '/src/constants.h']
  
  envVars = {
    'DETECT_SHOCK_THRESHOLD_':env.chi_sh,
    'eps_e_':env.eps_e,
    'eps_B_':env.eps_B,
    'zeta_':env.xi_e,
    'p_':env.psyn,
    }
    # 'acc_eff_':env.alpha
  
  for filepath in filepaths:
    with open(filepath, 'r') as f:
      out_lines = []
      inFile = f.read().splitlines()
      # copy file line by line with updated values where needed
      for line in inFile:
        if line.startswith('#define'):
          l = line.split()
          if l[1] in envVars:
            line = line.replace(l[2], str(envVars[l[1]]))
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
    'GEOMETRY':'cartesian', 'HYDRO':'rel_cart', 'RADIATION':'radiation_sph'
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
    'rhoNorm':env.rhoNorm, 'Nsh1':env.Nsh1, 'Ntot1':env.Nsh1+env.Next,
    'Nsh4':env.Nsh4, 'Ntot4':env.Nsh4+env.Next, 'stop':env.stop,
    'itmax':env.itmax, 'EXTRA_TIME':0.1*env.tmax, 'geometry':env.geometry
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
      'R_0':env.R0, 't_start':env.t0, 'Theta0':env.Theta0, 'cont':env.rhoContr,
      'rho1':env.rho1, 'u1':env.u1, 'p1':env.p1, 'D01':env.D01,
      'rho4':env.rho4, 'u4':env.u4, 'p4':env.p4, 'D04':env.D04
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
            val = vars2update[l[2]]
            line = line.replace(l[4], f'{val:{"f" if np.log10(val)<6 else ".2e"}}')
          elif var == 'm_fade':
            line = line.replace(l[4], f'{vars2update[l[2]]:.1f}')
          elif var.startswith('N'):
            line = line.replace(l[4], f'{int(vars2update[l[2]]):d}')
          elif var == 'rhoNorm':
            line = line.replace(l[4], env.rhoNorm)
          else:
            line = line.replace(l[4], f'{vars2update[l[2]]}')
      if 'if ( it >' in line:
        if (env.stop != 'it'):
          # make sure stopping with it is commented
          if ('//' not in line):
            line = line[:2] + '//  ' + line[2:]
        else:
          # make sure stopping with it is uncommented
          if ('//' in line):
            line = line.replace('// ', '')
          l = line.split()
          line = line.replace(l[4], f'{int(env.itmax):d}')
      if 'if ( t >' in line:
        if (env.stop != 'time'):
          # make sure stopping with time is commented
          if ('//' not in line):
            line = line[:2] + '//  ' + line[2:]
        else:
          # make sure stopping with time is uncommented
          if ('//' in line):
            line = line.replace('// ', '')
          l = line.split()
          line = line.replace(l[4], f'{int(3*env.tmax):d}')
      elif 'grid.printCols' in line:
        l = line.split()
        line = line.replace(l[3], f'{int(env.itdump):d}')
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
    import argparse
    parser = argparse.ArgumentParser(
      description='Prepare a GAMMA run from phys_input.ini, optionally applying a '
                  'Granot (2012) hydro unit-rescaling (alpha: length-time, zeta: mass-energy) '
                  'to the inputs first.')
    parser.add_argument('--alpha', type=float, default=1., help='length-time rescaling factor (default 1)')
    parser.add_argument('--zeta',  type=float, default=1., help='mass-energy rescaling factor (default 1)')
    parser.add_argument('--src', default=None,
      help='pristine phys_input.ini to rescale FROM (avoids compounding); '
           'defaults to in-place ./phys_input.ini')
    args = parser.parse_args()
    if (args.alpha != 1. or args.zeta != 1.) and args.src:
      rescale_input(args.alpha, args.zeta, src=args.src)
      main()  # inputs already rescaled from pristine src
    else:
      main(alpha=args.alpha, zeta=args.zeta)
