# -*- coding: utf-8 -*-
'''
When do fresh electrons dominate? Fixed-R snapshot, BM forward shock (Part C3 of the
notes): exponent alpha of dL_nu/dlnR_i per spectral segment, observed (x Gamma_f) and comoving.
Output: SC nu<nu_m (52-17k)/9, SC nu_m<nu<nu_c [(37-5k)p+57-21k]/12, FC nu<nu_c (40-17k)/9;
thresholds 52/17 and 40/17 (observed).

Paper I derivation check, recovered 2026-10-16 from a session scratchpad (was fresh.py).
Its results are quoted in the article notes, notes/derivation_rework.md. Standalone:
imports nothing from project_v2.
'''

import sympy as sp
k,p=sp.symbols('k p')
# BM forward shock: element (Lagrangian) and injection indices
g=(7-2*k)/2; gsh=(3-k)/2; ln=(13-2*k)/2; lB=(13-2*k)/3
lc=k+gsh-1; lm=gsh; lN=2-k
aG=g-gsh                      # Gamma_f at fixed R ~ R_i^aG
aB=lB-(1+gsh+lc)/2            # B' at fixed R ~ R_i^aB
zm=ln/3-lm; eta=ln/3+lc       # gamma_m, gamma_inf ~ R_i^zm, R_i^eta
f=lambda e: sp.factor(sp.simplify(e))
print("aG",f(aG),"aB",f(aB),"zeta",f(zm),"eta",f(eta))
for dop,name in ((1,"observed (x Gamma_f)"),(0,"comoving")):
    base=1+lN+dop*aG+aB
    em=dop*aG+aB+2*zm; einf=dop*aG+aB+2*eta
    a_low=f(base-em/3); a_mid=f(base+(p-1)*em/2); a_fc=f(base-einf/3)
    print(name)
    print("  SC nu<nu_m :",a_low, [a_low.subs(k,v) for v in (0,2)], " zero at k=",sp.solve(a_low,k))
    print("  SC nu_m<nu<nu_c:",a_mid, [sp.N(a_mid.subs({k:v,p:2.5}),3) for v in (0,2)])
    print("  FC nu<nu_c :",a_fc, [a_fc.subs(k,v) for v in (0,2)], " zero at k=",sp.solve(a_fc,k))
print("fraction 1-2^-a for FC observed k=2:",sp.N(1-2**(-sp.Rational(2,3)),3))
