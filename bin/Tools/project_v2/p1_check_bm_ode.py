# -*- coding: utf-8 -*-
'''
Single Blandford-McKee fluid element (k=0): direct ODE integration in R of synchrotron
+ adiabatic cooling against the closed form gma = gma_i A/(1 + gma_i I(R)).
Output: agreement to all 6 printed digits for gma_i = 1e2, 1e4, 1e7 at R/R_i = 1.1, 2, 4.

Paper I derivation check, recovered 2026-10-16 from a session scratchpad (was ode.py).
Its results are quoted in the article notes, notes/derivation_rework.md. Standalone:
imports nothing from project_v2.
'''

import numpy as np
from scipy.integrate import solve_ivp
# BM element, k=0, in units R_i=1, c=1, Gamma_i=100; tc_i chosen so gbar_ci = tc_i/tdyn_i = 50
k=0.; g=(7-2*k)/2; ln=(13-2*k)/2; lB=(13-2*k)/3; Gi=100.; tdyn_i=1/Gi; gbci=50.; tci=gbci*tdyn_i
Gam=lambda R: Gi*R**-g; tc=lambda R: tci*R**(2*lB)
def rhs(R,y):  # d gamma/dR = (dt'/dR) * [ -g^2/tc - g/t_ad ],  dt'/dR = 1/Gamma, t_ad^-1 = -(1/3) dln n'/dt'
    gm=y[0]; dtp=1/Gam(R); return [dtp*(-gm**2/tc(R)) - gm*ln/(3*R)]
Lam=ln/3+2*lB-1-g
for gi in (1e2,1e4,1e7):
    sol=solve_ivp(rhs,[1,4],[gi],rtol=1e-10,atol=1e-12,dense_output=True,method='LSODA')
    for R in (1.1,2.,4.):
        A=R**(-ln/3); I=(1-R**-Lam)/(Lam*gbci); an=gi*A/(1+gi*I)
        print(f"gi={gi:.0e} R={R}: num={sol.sol(R)[0]:.6g} analytic={an:.6g}")
