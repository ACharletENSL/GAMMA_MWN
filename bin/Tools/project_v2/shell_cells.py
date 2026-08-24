# -*- coding: utf-8 -*-
# @Author: acharlet

'''
Shell/cell bookkeeping shared by the per-cell hydro diagnostics.

These were factored out of sweep_norar.py when the free-coasting no-rarefaction laws
were retired (see working_cooling_data._norar_rows history): they are generic, they
have nothing to do with any extension law, and prerar_model / prerar_cell_evolution
depend on them.
'''

import numpy as np

from environment import MyEnv
from IO import open_celldata
from working_cooling_data import (select_postshock_rows, load_shockfront_states,
    _prepend_shocked_row)
from sweep_gammacm import Z_SHELL, EARLY_ANA

KEY = 'cooling_g100'
N_PROFILE = 42                      # cells in the per-cell profiles
SHELL_NAME = {4: 'RS', 1: 'FS'}


# ---------------------------------------------------------------------------
# cell bookkeeping
# ---------------------------------------------------------------------------
def shell_cell_range(key=KEY, z=Z_SHELL):
  '''(first, last, CD-side) cell indices of shell z; grid is [Next | Nsh4 | Nsh1 | Next].'''
  env = MyEnv(key)
  k4 = int(env.Next)
  kCD = k4 + int(env.Nsh4)
  k1 = kCD + int(env.Nsh1)
  return (k4, kCD - 1, kCD - 1) if z == 4 else (kCD, k1 - 1, kCD)


def profile_cells(key=KEY, z=Z_SHELL, n=N_PROFILE):
  '''n cell indices spread over shell z (endpoints included).'''
  kmin, kmax, _ = shell_cell_range(key, z)
  return np.unique(np.linspace(kmin, kmax, n).astype(int))


_SH_MEM = {}

def _history(key, k, n_settle=1, z=Z_SHELL, early_ana=EARLY_ANA, early_frac=0.):
  '''
  Post-shock history EXACTLY as get_shell_nuFnu_fromData's first pass builds it, shockfit
  prepend included.

  The prepend is not optional bookkeeping: it inserts the shock-front state at a smaller
  radius and a much higher pressure than the first measured row, and the crash detector
  reads slopes. Diagnostics that skip it validate a history production never sees -- which
  is how a detector artifact (a single-row jump read as a crash, dragging 30+ cells from
  R_h/R_inj ~ 2.7 to ~1.0) survived a full round of per-cell checks here. Keep this the
  one place histories are built for the diagnostics.
  '''
  d = open_celldata(key, k)
  if d is False:
    return None
  s = select_postshock_rows(d, n_settle)
  if len(s) < 2:
    return None
  s = s.copy()
  s['t'] = s['t'] - (d.t.iloc[0] if d.index[0] == 0 else 0.)
  if early_ana is not None:
    if (key, z, early_ana) not in _SH_MEM:
      _SH_MEM[(key, z, early_ana)] = load_shockfront_states(key, z, MyEnv(key),
                                                            source=early_ana)
    sh = _SH_MEM[(key, z, early_ana)]
    sel = sh.loc[sh.i == k]
    if len(sel):
      s = _prepend_shocked_row(s, sel.iloc[0], MyEnv(key), early_frac)
  return s
