# -*- coding: utf-8 -*-
# @Author: acharlet

'''
Launches simulations to perform a parameter space sweep
'''



from setup import *

# values of a_u - 1 to perform sweep
#au_arr = np.logspace(-1, 1.5)
au_arr = [1, 4]
def_u1 = 100

def main():
  subprocess.call("./HPC_setup.sh", shell=True)
  for aum in au_arr:
    update_input(aum)
    subprocess.call("./HPC_launch.sh", shell=True)




def update_input(aum):
  '''
  Updates the phys_input.ini file with new value of u4
  '''

  file = 'phys_input.ini'
  au = 1+aum
  u1 = def_u1
  # find value for u1
  with open(file, 'r') as f:
    inFile = f.read().splitlines()
    for line in inFile:
      if line:
        l = line.split()
        if l[0] == 'u1': u1 = float(l[1])
  
  u4 = au * u1
  out_lines = []
  # replace value of u4
  with open(file, 'r') as f:
    inFile = f.read().splitlines()
    for line in inFile:
      if line:
        l = line.split()
        if l[0] == 'u4':
          line = line.replace(l[1], f'{u4:.0f}')
      out_lines.append(line)

  with open(file, 'w') as outf:
    outf.writelines(f'{s}\n' for s in out_lines)

def check_simFinished():
  '''
  Check if job is finished by checking the queue
  '''

  # HPC Python version is 3.6
  # check the jobs in queue and stores the shell output without overhead
  result = subprocess.run(f'squeue -u arthurc -h',
    shell=True,stdout=subprocess.PIPE, universal_newlines=True)

  if not result.stdout.strip():
    # if empty output, no jobs are currently running
    return True
  else:
    return False

def move_results(name):
  '''
  Moves simulated data from the results/last folder into results/<name>
  '''

  subprocess.call(f"mv results/Last results/{name}", shell=True)
  subprocess.call("mkdir -p results/Last", shell=True)