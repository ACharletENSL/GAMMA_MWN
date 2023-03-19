# -*- coding: utf-8 -*-
# @Author: acharlet

'''
Check the setup by relaunching simulation and showing initial conditions and 
'''

# Imports
# --------------------------------------------------------------------------------------------------
import subprocess
import sys
sys.path.insert(1, 'bin/Tools/')
from plotting_scripts import *

def main():
  subprocess.call("./relaunch_test.sh", shell=True)
  #f1 = plt.figure(1)
  #f2 = plt.figure(2)
  #plot_mono('p', 0, fig=f1)
  #plot_mono('p', 1, fig=f2)
  plot_primvar(100)
  plt.show()

if __name__ == "__main__":
    main()
