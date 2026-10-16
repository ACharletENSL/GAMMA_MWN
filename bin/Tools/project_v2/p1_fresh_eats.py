# -*- coding: utf-8 -*-
'''
When do fresh electrons dominate? With equal-arrival-time integration over the BM
forward shock, GS02 Eq. (A14) in (y, chi) (Part C5 of the notes). Reproduces GS02's threshold
k = 23/13 for PLS E (frozen fast-cooling electrons).
Output: alpha = (23-13k)/9, (35-13k)/9, [(4k+1)p+93-36k]/24; zeros at k = 23/13, 35/13, 191/52
(p=2.5). The numerical small-y slope with the full GS02 functions matches the analytic one to
4 digits at k = 0, 1.5, 23/13, 2.

Paper I derivation check, recovered 2026-10-16 from a session scratchpad (was eats.py).
Its results are quoted in the article notes, notes/derivation_rework.md. Standalone:
imports nothing from project_v2.
'''

import sympy as sp, numpy as np
from scipy.integrate import quad
k,p=sp.symbols('k p')
F=lambda e: sp.factor(sp.simplify(e))
# integrand of F_nu(t_obs) in (y, chi):  y^(10-2k) chi D^-2 P'_nu'   (lab dV, j=delta^2 j', mu-integral by arrival-time delta)
# hydro (GS02 A4-A9,A12), powers of y and chi (chi>>1):
n_=( -k-(3-k)/2 , -(10-3*k)/(2*(4-k)) )
B_=( sp.Rational(-3,2), -(17-4*k)/(6*(4-k)) )
nup=( -(5-k)/2, sp.Rational(-1,2) )                 # nu' = nu/delta (times D)
gmax=( (1+k)/2, -(16+k)/(6*(4-k)) )
gm  =( -(3-k)/2, -(4+k)/(6*(4-k)) )
base=( 10-2*k, 1 )
def seg(gX,sig):   # P' ~ n' B' (nu'/(B' gX^2))^sig ; returns (A,B,Cextra)
    A=base[0]+n_[0]+B_[0]+sig*(nup[0]-B_[0]-2*gX[0]); Bc=base[1]+n_[1]+B_[1]+sig*(nup[1]-B_[1]-2*gX[1]); return F(A),F(Bc)
def alpha_q(A,Bc):
    up=F((A+1-(4-k)*(1+Bc))/2); low=F(A+1); cond=F(A+(4-k)*(1+Bc)+1); return up,low,cond
for name,gX,sig in (("FC nu<nu_c (PLS E)",gmax,sp.Rational(1,3)),("SC nu<nu_m",gm,sp.Rational(1,3)),("SC nu_m<nu<nu_c",gm,-(p-1)/2)):
    A,Bc=seg(gX,sig); up,low,cond=alpha_q(A,Bc)
    print(name,"| y-exp A=",A," chi-exp B=",Bc," | dF/dlnR0 ~ R0^",up,"(if",cond,">0)  zero at k=",sp.solve(up.subs(p,sp.Rational(5,2)),k))
    print("     y-integrand after chi-integral ~ y^",F(A-(4-k)*(1+Bc)) if True else "")
    print("     k=0,2 (p=2.5):",[up.subs({k:v,p:2.5}) for v in (0,2)],"cond:",[cond.subs({k:v,p:2.5}) for v in (0,2)])
# numerical check of PLS E with full (non-asymptotic) GS02 functions
def run(kv):
    nn=lambda y,c: y**(-kv-(3-kv)/2)*c**(-(10-3*kv)/(2*(4-kv)))
    BB=lambda y,c: y**-1.5*c**(-(17-4*kv)/(6*(4-kv)))
    def gmx(y,c):
        R0=y*c**(-1/(4-kv)); pre=R0**((1+kv)/2)
        return pre*c**((25-2*kv)/(6*(4-kv)))/(c**((19-2*kv)/(3*(4-kv)))-1)
    D=lambda y,c: 1+(7-2*kv)*c*y**(4-kv)
    nu=lambda y,c: y**(-(5-kv)/2)*c**-0.5*D(y,c)
    f=lambda y,c: y**(10-2*kv)*c*D(y,c)**-2*nn(y,c)*BB(y,c)**(2/3)*gmx(y,c)**(-2/3)*nu(y,c)**(1/3)
    ys=np.logspace(-6,-3,7); I=[]
    for y in ys:
        lo=1.5; hi=y**(kv-4)   # start just above chi=1 (element not yet frozen there; asymptotic regime)
        v=quad(lambda lc: f(y,np.exp(lc))*np.exp(lc), np.log(lo), np.log(hi), limit=400)[0]; I.append(v)
    s=np.polyfit(np.log(ys),np.log(I),1)[0]
    print(f"k={kv}: numerical small-y slope of chi-integrated integrand = {s:.4f}, analytic {(37-26*kv)/9:.4f}; y-integral converges iff slope>-1")
for kv in (0.,1.5,23/13,2.):
    run(kv)
