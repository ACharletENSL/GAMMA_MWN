# -*- coding: utf-8 -*-
# @Author: acharlet

'''
Launches simulations to perform a parameter space sweep
'''


import time
import glob
from setup import *
from analysis_thinshell import extract_fittingData
from IO import join_extracted, check_done_logau, logau_to_key

# values of a_u - 1 to perform sweep
#logau_arr = np.arange(-0.6, 0.6, 0.1)
#logau_arr[np.abs(logau_arr)<1e-2] = 0
#logau_arr = np.arange(0.6, 1.6, 0.1)
logau_arr = np.arange(-1, -0.5, 0.1)
#logau_arr = [0, np.log10(4)]
def_u1 = 100
waittime = 60 # default wait time between checksx
ONHPC = ('arthurc' in os.environ['HOME'])
delete = True

def main():
  clean = False
  if delete and (not ONHPC):
    clean = True
    print("Data will be deleted after each run")
  if ONHPC:
    # load modules
    subprocess.call("source ~/.bashrc", shell=True)
  for log_au in logau_arr:
    key = logau_to_key(log_au)
    au = 1 + 10**log_au
    print(f'Running simulation with log a_u - 1 = {log_au:.1f} (a_u = {au:.2f})')
    run_sim(au)
    print('Run finished, moving in results/' + key)
    move_results(name)
    extract_fittingData(key, log_au)
    if clean:
      print("Deleting data")
      os.popen('rm -f results/' + key + '/phys*.out')
  join_extracted()

def analyze_all():
  logau_done = check_done_logau()
  for log_au in logau_done:
    key = logau_to_key(log_au)
    extract_fittingData(key, log_au)
  join_extracted()

def run_sim(au):
  update_input(au)
  if ONHPC:
    subprocess.call("./HPC_launch.sh", shell=True)
    while(check_simRunning()):
      time.sleep(60)
  else:
    subprocess.run("./local_launch.sh", shell=True)

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
  path = 'results/sweep_' + name
  if os.path.isdir(path):
    subprocess.call(['rm', '-rf', path])

  subprocess.run(['mv', 'results/Last', path])
  subprocess.call("mkdir -p results/Last", shell=True)


if __name__ == "__main__":
    main()
