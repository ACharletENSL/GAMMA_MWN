# -*- coding: utf-8 -*-
'''
Path resolution for the convergence tests, shared by every script here.

These scripts used to live outside the repository and each hard-coded
`/home/acharlet/Science/ARCO/MWN/GAMMA_MWN` (or, for the two that run on the
cluster, `~/work/GAMMA_MWN`). Now that they sit inside the tree, the root is
four levels up from this file, so the same script runs unchanged on the laptop
and on the HPC. GAMMA_DIR in the environment still wins, for the odd case of
pointing the tests at a second checkout.

Import it first, before anything from project_v2:

    from _paths import GAMMA_DIR, FIG      # also puts project_v2 on sys.path
'''
import os
import sys

# .../GAMMA_MWN/bin/Tools/project_v2/convergence_tests/_paths.py -> .../GAMMA_MWN
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir, os.pardir, os.pardir))

GAMMA_DIR = os.environ.setdefault('GAMMA_DIR', _ROOT)
PROJECT_V2 = os.path.join(GAMMA_DIR, 'bin', 'Tools', 'project_v2')
FIG = os.path.join(GAMMA_DIR, 'bin', 'Tools', 'figures')
RESULTS = os.path.join(_HERE, 'results')

# project_v2 first (the modules under test), then this directory (so the scripts
# can import each other), matching the order the standalone versions used.
for _p in (PROJECT_V2, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if not os.path.isdir(FIG):
    print(f'WARNING: no figures directory at {FIG} -- is GAMMA_DIR right?',
          file=sys.stderr)
