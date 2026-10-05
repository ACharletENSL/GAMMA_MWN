# -*- coding: utf-8 -*-
# @Author: acharlet

'''
Dissipation efficiency eps_diss(a_u) from the a_u sweep, tabulated for peak_modeling.

eps_diss is the shell's internal energy produced by its shock over the crossing, in the lab
frame, divided by the shell's initial kinetic energy: the analytic chain's env.eff_4 (RS,
Ei3f/Ek4) and env.eff_1 (FS, Ei2f/Ek1), phys_functions_shells.derive_Eint_crosstime. The
analytic value gives every cell the planar, constant-velocity jump (Gamma_0, Gamma_34). The
sweep value gives each cell the jump it actually gets, from the table's lfac(x) and ShSt(x)
fits at the radius x_k where the shock reaches it, mass-weighted and summed over the crossing:
  E_diss = sum_k dm_k c^2 Gamma_d(x_k) (1 + beta_d^2 (G_ud+1)/(3 G_ud)) (G_ud(x_k) - 1)
This is energy dissipated as the shock passes, before any adiabatic loss, i.e. what is
available to radiate in fast cooling.

The sweep runs at u1 = 10, where the analytic chain is 1-9% off its ultra-relativistic limit,
so the table stores the u1-independent HYDRO correction
  ratio = E_diss / Ei_analytic(same env)
and eps_C25(a_u) = peak_modeling.R24_eff_diss(a_u) * ratio, on the same UR, equal-energy
footing as R24_eff_diss.

Output: extracted_data/<table>_eff.csv, columns log_aum, eff_RS, eff_FS, ratio_RS, ratio_FS.
'''

import os
import numpy as np
import pandas as pd

from IO import GAMMA_dir, SWEEP_TABLE
from environment import MyEnv
from phys_constants import c_
from fits_hydro import reconstruct_data, smooth_bpl0_apy
from peak_modeling import R24_eff_diss

TEMPLATE = GAMMA_dir + '/hpc/phys_input_sweep_au_cold.ini'
LFAC_COLS = ['lfac_A', 'lfac_x_b', 'lfac_alpha', 'lfac_s']
SHST_COLS = ['ShSt_A', 'ShSt_x_b', 'ShSt_alpha', 'ShSt_s']


def sweep_env(log_aum, template=TEMPLATE, tmp=None):
  '''
  The environment of one sweep point: the sweep's own input at u4 = u1 a_u.
  The stop time is irrelevant here and set to 1.
  '''
  au = 1. + 10**log_aum
  txt = open(template).read()
  u1 = float([l.split()[1] for l in txt.splitlines() if l.startswith('u1 ')][0])
  txt = txt.replace('@U4@', f'{u1*au:.10g}').replace('@TSTOP@', '1')
  tmp = tmp or os.path.join(GAMMA_dir, 'extracted_data', '.eff_tmp_phys_input.ini')
  with open(tmp, 'w') as f:
    f.write(txt)
  try:
    env = MyEnv(tmp)
  finally:
    os.remove(tmp)
  return env


def dissipated_ratio(env, row, reverse=True):
  '''
  E_diss/Ei_analytic for one shell, from one sweep-table row (pandas Series).
  '''
  fast = reverse
  popt_lfac = row[LFAC_COLS].to_numpy(float)
  popt_ShSt = row[SHST_COLS].to_numpy(float)
  t_hit, cells_i, R, dx, rho, vx, lfac, p, trac = reconstruct_data(
    row['t_max'], env, fast, popt_lfac, popt_ShSt)
  x = R*c_/env.R0
  ShSt0 = (env.lfac34 if fast else env.lfac21) - 1.
  Gud = 1. + ShSt0*smooth_bpl0_apy(x, *popt_ShSt)
  beta2 = 1. - 1./lfac**2
  # cell rest masses: uniform initial density over the shell, so dm ~ r_init^2
  D0, N = (-env.D04, env.Nsh4) if fast else (env.D01, env.Nsh1)
  r_init = 1. + np.arange(N)*D0/(N*env.R0)
  w = r_init**2/np.sum(r_init**2)
  M, Ei_ana = (env.M4, env.Ei3f) if fast else (env.M1, env.Ei2f)
  E = np.sum(w*M*c_**2*lfac*(1. + beta2*(Gud+1.)/(3.*Gud))*(Gud-1.))
  return E/Ei_ana, t_hit.max()/row['t_max']


def build(table=SWEEP_TABLE, write=True):
  '''
  Tabulate eps_diss for every row of extracted_data/<table>_{RS,FS}.csv.
  '''
  folder = GAMMA_dir + '/extracted_data/'
  rows = {fr: pd.read_csv(folder + f'{table}_{fr}.csv', sep='\t') for fr in ['RS', 'FS']}
  out = []
  for (_, rRS), (_, rFS) in zip(rows['RS'].iterrows(), rows['FS'].iterrows()):
    la = round(rRS['log_aum'], 1)
    assert np.isclose(la, rFS['log_aum'])
    env = sweep_env(la)
    au = 1. + 10**la
    ratio_RS, hitRS = dissipated_ratio(env, rRS, reverse=True)
    ratio_FS, hitFS = dissipated_ratio(env, rFS, reverse=False)
    out.append([la, R24_eff_diss(au, True)*ratio_RS, R24_eff_diss(au, False)*ratio_FS,
                ratio_RS, ratio_FS, hitRS, hitFS])
  df = pd.DataFrame(out, columns=['log_aum', 'eff_RS', 'eff_FS', 'ratio_RS', 'ratio_FS',
                                  'lasthit_RS', 'lasthit_FS'])
  if write:
    df.drop(columns=['lasthit_RS', 'lasthit_FS']).to_csv(
      folder + f'{table}_eff.csv', sep='\t', index=False)
  return df


if __name__ == '__main__':
  print(build().to_string(index=False))
