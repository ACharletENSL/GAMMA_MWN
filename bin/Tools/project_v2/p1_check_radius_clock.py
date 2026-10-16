# -*- coding: utf-8 -*-
'''
Radius-clock algebra: the I(R) integrand exponent, the Jacobian
A^(p-1) gma^-p (1 - gma I/A)^(p-2) (printed unsimplified; it is zero), and direct numerical
shell integrals for the coasting homogeneous shell.
Output: frozen FC slope -0.66666 (expect -2/3); smeared slope +0.564 (expect +1/2: the fit
window reaches the bend from the R_i = R_0 lower limit, so this value is biased; the notes'
+0.5002 is NOT what this script prints); R_tr = 7.197; W = 8.07.

Paper I derivation check, recovered 2026-10-16 from a session scratchpad (was checkR.py).
Its results are quoted in the article notes, notes/derivation_rework.md. Standalone:
imports nothing from project_v2.
'''

import sympy as sp, numpy as np
from scipy.integrate import quad
x,r,Ri,ln,lB,g,G,tc,c,gam,A,I,p=sp.symbols('x r R_i lambda_n lambda_B g Gamma_i tc c gamma A I p',positive=True)
Lam=ln/3+2*lB-1-g
# integrand exponent and I
expo=sp.simplify(-ln/3+g-2*lB); print("integrand exponent + (Lam+1):",sp.simplify(expo+Lam+1))
print("int_1^r x^(-Lam-1):",sp.simplify(sp.integrate(x**(-Lam-1),(x,1,r)).subs({ln:2,lB:sp.Rational(5,3),g:0})))
# Jacobian
gi=gam/(A-gam*I); print("Jacobian:",sp.simplify(gi**(-p)*sp.diff(gi,gam)-A**(p-1)*gam**(-p)*(1-gam*I/A)**(p-2)))
# numeric shell integrals, reference case: g=0, ln=2, lB=5/3, Lam=3, lc_sh=7/3, lam_N=0 (const rate per dR)
Lm=3.; d=2/3.; lc=7/3.
def gam_inf(R,Rinj,gbc0): return (R/Rinj)**(-d)*Lm*gbc0*(Rinj)**lc  # R0=1
# frozen FC: count elements per gamma bin, Ndot=const in R_i in [1,2]
Ri_s=np.linspace(1,2,200001); gs=gam_inf(10.,Ri_s,1e-2*1e3/3.43)
h,e=np.histogram(np.log(gs),bins=30); dn=h/np.diff(np.exp(e)); cen=np.exp(0.5*(e[1:]+e[:-1]))
print("frozen FC slope (expect -2/3):",np.polyfit(np.log(cen[3:-3]),np.log(dn[3:-3]),1)[0])
# smeared: SC elements, gm(R;Ri)=gm0*(R/Ri)^-d, f=(p-1)gm^(p-1) g^-p for g>gm
pp=2.5; R=100.
def dN(gv):
    Rstar=min(R, R*(gv/1.0)**(1/d))  # gm,i=1 constant
    return quad(lambda Rii:(pp-1)*((R/Rii)**(-d))**(pp-1)*gv**(-pp),1,Rstar)[0]
gg=np.logspace(np.log10((1/R)**d)+0.3,-0.3,10); vals=[dN(v) for v in gg]
print("smeared slope (expect +1/2):",np.polyfit(np.log(gg),np.log(vals),1)[0])
print("R_tr:",100**(3/7), " W:",8*(1+3*(1-1/8)*0.01/3))
