/*
* @Author: Eliot Ayache
* @Date:   2020-10-25 10:19:37
* @Last Modified by:   Eliot Ayache
* @Last Modified time: 2021-01-26 16:05:20
*/


#include "../environment.h"
#include "../interface.h"
#include "../cell.h"
#include "../grid.h"
#include "../constants.h"


#if SHOCK_DETECTION_ == ENABLED_

  static double compute_cs(double rho, double p){
    return(sqrt(GAMMA_*p / (rho + p*GAMMA_/(GAMMA_-1.))));
  }

  static double compute_Sd(FluidState S1, FluidState S2, int dim, 
                           bool reverse=false){

    int uux = UU1+dim;
    int uut = UU2-dim;

    // 1S1R Rezzolla&Zanotti2013 eq. 4.211 (p238)
    // ------------------------------------------
    // this equation involves numerically solving an integral over pressure
    int n_evals = 10; // number of points in the integral
    double chi = 0.2; // param for strength of shocks to detect (0:weak, 1:strong)

    double p1 = S1.prim[PPP];
    double p2 = S2.prim[PPP];

    if (p1==p2) return(-1); // no shock possible

    double delta_p = p2 - p1;
    double dp = delta_p / (double) n_evals;

    double rho1 = S1.prim[RHO];
    double rho2 = S2.prim[RHO];
    double h1 = 1 + p1*GAMMA_/(GAMMA_-1.)/rho1;
    double h2 = 1 + p2*GAMMA_/(GAMMA_-1.)/rho2;

    // projecting the velocity field in a local tetrad (R&Z13 p244)
    // in cartesian coordinates, gamma = diag(1,1,1) => M = diag(1,1,1) => no projection
    double lfac1 = S1.lfac();
    double lfac2 = S2.lfac();
    double vx1 = S1.prim[uux] / lfac1;
    double vx2 = S2.prim[uux] / lfac2;
    if (reverse){
      vx1 *= -1;
      vx2 *= -1;
    }
    double ut1 = S1.prim[uut];
    double A1 = h1*ut1;

    double I = 0;
    // if (p1!=p2) { // we get I = 0 if p1=p2, so no need for integration:
      for (int ip = 0; ip < n_evals; ++ip){
        double p = p1 + (ip+.5)*dp;

        // h is computed from s (entropy). Since rarefaction waves are isentropic,
        // we can set s = s1, so rho = rho1(p/p1)^(1/GAMMA_) (Laplace)
        double rho = rho1  * pow(p/p1, 1./GAMMA_);    
        double h = 1 + p*GAMMA_/(GAMMA_-1.)/rho;
        double cs = compute_cs(rho,p);

        double dI = sqrt(h*h + A1*A1 * (1.-cs*cs)) / ((h*h + A1*A1) * rho * cs) * dp;
        I += dI;
      }
    // }

    double vSR = tanh(I);

    // 2S Rezzolla&Zanotti2013 eq. 4.206 (p238)
    // ----------------------------------------
    double rho22 = rho2*rho2;
    double lfac22 = lfac2*lfac2;
    double vx22 = vx2*vx2;
    double g = GAMMA_;
    double gm1 = (g-1.);
    double gm12 = gm1*gm1;

    double Da = 4. * g * p1 * ((g-1)*p2 + p1) / (gm12 * (p1-p2) * (p1-p2));
    double Db = h2 * (p2-p1) / rho2 - h2*h2;
    double D = 1. - Da*Db;

    double h3 = (sqrt(D)-1) * gm1 * (p1-p2) / (2 * (gm1*p2 + p1));

    double J232a = - g/gm1 * (p1-p2);
    double J232b = h3*(h3-1)/p1 - h2*(h2-1)/p2;
    double J232 = J232a/J232b;
    double J23 = sqrt(J232);

    double Vsa = rho22*lfac22*vx2 + fabs(J23) * sqrt(J232 + rho22*lfac22*(1.-vx22));
    double Vsb = rho22*lfac22 + J232;
    double Vs = Vsa / Vsb;

    double v2Sa = (p1 - p2) * (1. - vx2*Vs);
    double v2Sb = (Vs - vx2) * (h2*rho2*lfac22*(1.-vx22) + p1 - p2);
    double v2S = v2Sa / v2Sb;
    
    // Shock detection threshold:
    double vlim = vSR + chi*(v2S - vSR);
    double v12 = (vx1 - vx2) / (1 - vx1*vx2);
    double Sd = v12 - vlim;

    return(Sd);
  }

  void Interface :: measureShock(Cell *cL, Cell *cR){

    double Sd;
    // Forward shocks
    Sd = compute_Sd(SL, SR, dim);
    cL->Sd = fmax(cL->Sd, Sd);
    // Reverse shocks
    Sd = compute_Sd(SR, SL, dim, true);
    cR->Sd = fmax(cR->Sd, Sd);

  }  

  void Cell :: detectShock(){

    if (Sd > DETECT_SHOCK_THRESHOLD_) isShocked = true;

  }

  void Cell :: resetShock(){

    Sd = 0;
    isShocked = false;

  }

#endif


#if LOCAL_SYNCHROTRON_ == ENABLED_

  double radiation_gammae2trac(double gme, FluidState S){

    double lfac = S.lfac();
    double rho = S.prim[RHO];

    if (gme != 0.) return(lfac * pow(rho, 4./3.) / gme);
    else return(0);

  }

  double radiation_trac2gammae(double gme_trac, FluidState S){
    // same function as above, but avoids confusion

    double lfac = S.lfac();
    double rho = S.prim[RHO];

    if (gme_trac != 0.) return(lfac * pow(rho, 4./3.) / gme_trac);
    else return(0);

  }

  void Cell :: radiation_apply_trac2gammae(){
    // To apply only on Cdump cells in output!

    double lfac = S.lfac();
    double rho = S.prim[RHO];
    double gmin = S.prim[GMN] * (lfac*rho);
    double gmax = S.prim[GMX] * (lfac*rho);
    S.prim[GMN] = radiation_trac2gammae(gmin, S);
    S.prim[GMX] = radiation_trac2gammae(gmax, S);

  }

  void Cell :: radiation_initGammas(){

    double lfac = S.lfac();
    double rho = S.prim[RHO];
    double lim = lfac * pow(rho, 4./3.);  // equiv. gammae = 1.
    S.prim[GMN] = lim / (lfac*rho);
    S.prim[GMX] = lim / (lfac*rho);
    S.cons[GMN] = lim;
    S.cons[GMX] = lim;

  }

  void Cell :: radiation_injectParticles(){

    double lfac = S.lfac();
    double rho = S.prim[RHO];
    double lim = lfac * pow(rho, 4./3.) / (lfac*rho);  // equiv. gammae = 1.
    double *gmax = &S.prim[GMX];
    double *gmin = &S.prim[GMN];

    if (isShocked){
      *gmax = radiation_gammae2trac(GAMMA_MAX_INIT_, S) / (lfac*rho);
      *gmin = radiation_gammae2trac(1., S) / (lfac*rho);
    }
    if (*gmax > lim or *gmax == 0.) *gmax = lim;
    if (*gmin > lim or *gmin == 0.) *gmin = lim;

  }

  void Cell :: radiativeSourceTerms(double dt){

    double rho = S.prim[RHO];
    double p = S.prim[PPP];
    double h = 1 + p*GAMMA_/(GAMMA_-1.)/rho;
    double eps = rho * (h-1.) / GAMMA_;
    double eB = eps_B_ * eps;
    double B = sqrt(8.*PI*eB);

    double dgmax = Nalpha_ * pow(rho, 4./3.) * B*B * G.dV * dt;
    // printf("%le\n", dgmax);
    S.prim[GMX] += dgmax;

  }

#endif













