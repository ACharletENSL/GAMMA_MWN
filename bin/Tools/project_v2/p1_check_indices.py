# -*- coding: utf-8 -*-
'''
Freezing index Lambda for every row of Table tab:indices, the FS shock-front indices
and the Sect. 3.2 segment indices for the BM forward shock (sympy).
Output: Lambda = 3, 7/3, 4, (19-2k)/3, (23+3g)/7; FS frozen FC (2-7k)/(16+k) -> 1/8 (k=0),
-2/3 (k=2); adiabatic smear (14-7k)/(4+k) -> 7/2, 0; I(2R_i)/I_inf = 7/8, gbar_c/gbar_ci = 3.43.

Paper I derivation check, recovered 2026-10-16 from a session scratchpad (was check.py).
Its results are quoted in the article notes, notes/derivation_rework.md. Standalone:
imports nothing from project_v2.
'''

import sympy as sp
k,g,ln,lB,gam,p=sp.symbols('k g lambda_n lambda_B gammahat p')
Lam=lambda ln,lB,g: sp.simplify(ln/3+2*lB-1-g)
print("coasting cold, const eps_B:",Lam(2,sp.Rational(5,3),0))
print("coasting hot :",Lam(2,sp.Rational(4,3),0), " spreading hot:",Lam(3,2,0))
print("coasting frozen flux:",Lam(2,1,0))
# BM element (GS02 A9): g_f=(7-2k)/2, ln=(13-2k)/2, lB=(13-2k)/3
gf=(7-2*k)/2; lnb=(13-2*k)/2; lBb=(13-2*k)/3
print("BM Lambda:",sp.factor(Lam(lnb,lBb,gf)), " check e'~n'^(4/3):",sp.simplify(2*lBb-sp.Rational(4,3)*lnb))
# thin-shell RS after crossing (KS00): n'~R^-6(3+g)/7, e'~R^-8(3+g)/7
lnr=6*(3+g)/7; print("thin RS eps_B:",sp.factor(Lam(lnr,4*(3+g)/7,g))," frozen flux:",sp.factor(Lam(lnr,lnr-1,g)))
# FS injection (Eulerian): Gamma_sh index (3-k)/2, B'_i^2 ~ Gamma^2 n_ext
gs=(3-k)/2; lc_inj=sp.simplify(2*(gs+k/2)-1-gs); print("FS gbar_c,i index:",lc_inj)
aN=3-k; mu=gs
fc=sp.simplify(aN/(lnb/3+lc_inj)-1); sm=sp.simplify(aN/(lnb/3-mu)-1)
print("FS frozen FC idx:",sp.factor(fc),[fc.subs(k,v) for v in (0,2)])
print("FS adiab-smeared idx:",sp.factor(sm),[sm.subs(k,v) for v in (0,2)])
# observer time: gbar_c,i in t_obs (t_obs ~ R^{4-k})
print("gamma_c ~ t_obs^",sp.simplify(lc_inj/(4-k)))
# old results recovered: coasting homogeneous, inj per dlnR ~ R^{1+kold}
ko=sp.symbols('k_old'); print("old FC:",sp.simplify((1+ko)/3-1), " old smear:",sp.simplify((1+ko)/sp.Rational(2,3)-1))
print("I(2Ri)/Iinf for Lambda=3:",1-sp.Rational(1,8), " gbar_c/gbar_ci:",sp.N(3/(1-sp.Rational(1,8))))
