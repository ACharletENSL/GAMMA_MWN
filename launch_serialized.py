# -*- coding: utf-8 -*-
# @Author: acharlet

'''
Launches simulations to perform a parameter space sweep
'''


import time
from setup import *

# values of a_u - 1 to perform sweep
#au_arr = np.logspace(-1, 1.5)
au_arr = [1, 4]
def_u1 = 100
waittime = 60 # default wait time between checks

def main():
  # load modules
  subprocess.call("source ~/.bashrc", shell=True)
  for aum in au_arr:
    au = aum+1
    print(f'Running simulation with a_u = {au:.2f}')
    name = f"au={au:.2f}"
    update_input(au)
    subprocess.call("./HPC_launch.sh", shell=True)
    #subprocess.call("./local_launch.sh", shell=True)
    while(check_simRunning()):
      time.sleep(60)
    print('Run finished, moving in results/sweep/' + name)
    move_results(name)


def update_input(au):
  '''
  Updates the phys_input.ini file with new value of u4
  '''

  file = 'phys_input.ini'
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

def check_simRunning():
  '''
  Check if job is still running by checking the queue
  '''

  # HPC Python version is 3.6
  # check the jobs in queue and stores the shell output without overhead
  result = subprocess.run(f'squeue -u arthurc -h',
    shell=True,stdout=subprocess.PIPE, universal_newlines=True)

  if not result.stdout.strip():
    # if empty output, no jobs are currently running
    return False
  else:
    return True

def move_results(name):
  '''
  Moves simulated data from the results/last folder into results/<name>
  '''
  name = name.strip("'\"") 
  path = os.path.join('results/sweep/', name)

  subprocess.run(['mv', 'results/Last', path])
  subprocess.call("mkdir -p results/Last", shell=True)


if __name__ == "__main__":
    main()
