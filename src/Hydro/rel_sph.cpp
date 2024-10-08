/*
* @Author: eliotayache
* @Date:   2020-06-10 11:18:13
* @Last Modified by:   Eliot Ayache
* @Last Modified time: 2021-02-17 14:37:48
*/

#include "../fluid.h"
#include "../environment.h"
#include "../err.h"
#include "../constants.h"
#include "../interface.h"
#include "../cell.h"
#include <iostream>
#include <gsl/gsl_roots.h>   // root finding algorithms
#include <gsl/gsl_math.h>
#include <gsl/gsl_errno.h>

FluidState::FluidState(){

  for (int i = 0; i < NUM_Q; ++i) prim[i]=0.;
  prim[RHO]=1.;
  prim[PPP]=1.;

  for (int i = 0; i < NUM_Q; ++i) cons[i]=0.;
  prim[DEN]=1.;

  for (int n = 0; n < NUM_D; ++n){
    for (int i = 0; i < NUM_Q; ++i) flux[n][i]=0.;
  }

}

FluidState::~FluidState(){}


double FluidState::gamma(){

  #if EOS_ == IDEAL_EOS_
    return(GAMMA_);

  #elif EOS_ == SYNGE_EOS_
    double rho = prim[RHO];
    double p = prim[PPP];
    double a = p/rho / (GAMMA_-1.);
    double e_ratio = a + sqrt(a*a+1.);
    double gamma_eff = GAMMA_ - (GAMMA_-1.)/2. * (1.-1./(e_ratio*e_ratio));
    return(gamma_eff);

  #elif EOS_ == RYU_EOS_
    double rho = prim[RHO];
    double p = prim[PPP];
    double T = p/rho;
    // double h = 2. * (6. * T * T + 4. * T + 1.) / (3. * T + 2.);
    // double gamma_eff = (h - 1.) / (h - 1. - T);
    double y = 3. * T + 1.;
    double gamma_eff = (4. * y + 1.) / (3. * y);
    return(gamma_eff);
  
  #elif EOS_ == TAUB_EOS_
    double rho = prim[RHO];
    double p = prim[PPP];
    double T = p/rho;
    //double h = 2.5 * T + sqrt(2.25 * T * T + 1.);
    double gamma_eff = (1./6.)*(8. - 3.*T + sqrt(4. + 9.*T*T));
    return(gamma_eff);
    
  #endif

}


void FluidState::prim2cons(double r){

  double rho = prim[RHO];
  double p   = prim[PPP];
  double u   = 0;
  double uu[NUM_D];
  for (int i = 0; i < NUM_D; ++i){
    uu[i] = prim[UU1+i];
    u += uu[i]*uu[i];
  }
  u = sqrt(u);
  double lfac = sqrt(1+u*u);
  double gma = gamma();
  double h   = 1.+p*gma/(gma-1.)/rho; // ideal gas EOS (TBC)

  double D   = lfac*rho;
  double tau = D*h*lfac-p-D;
  double ss[NUM_D];
  for (int i = 0; i < NUM_D; ++i) ss[i] = D*h*uu[i];

  cons[DEN] = D;
  cons[TAU] = tau;
  for (int i = 0; i < NUM_D; ++i) cons[SS1+i] = ss[i];
  cons[SS1+t_] *= r;

  for (int t = 0; t < NUM_T; ++t) cons[TR1+t] = prim[TR1+t]*D; 

}


void FluidState::state2flux(double r){

  double p = prim[PPP];
  double u = 0;
  double uu[NUM_D];
  for (int i = 0; i < NUM_D; ++i)
  {
    uu[i] = prim[UU1+i];
    u += uu[i]*uu[i];
  }
  u = sqrt(u);
  double lfac = sqrt(1+u*u);
  double D = cons[DEN];
  double ss[NUM_D];
  for (int i = 0; i < NUM_D; ++i) ss[i] = cons[SS1+i];
  ss[t_] /= r;

  for (int n = 0; n < NUM_D; ++n){
    flux[n][DEN] = D*uu[n]/lfac;
    for (int i = 0; i < NUM_D; ++i){
      if (i==n) flux[n][SS1+i] = ss[i]*uu[n]/lfac + p;
      else flux[n][SS1+i] = ss[i]*uu[n]/lfac;
    }
    flux[n][SS1+t_] *= r; // spherical coords
    flux[n][TAU] = ss[n] - D*uu[n]/lfac;
    for (int t = 0; t < NUM_T; ++t) flux[n][TR1+t] = cons[TR1+t]*uu[n]/lfac;
  }

}

/**
 *
 * PRIMITIVE VARIABLES RECONSTRUCTION:
 *
 */

// // This is the function we need to set to zero:
// static double f_eval(double p, double *f, double *dfdp, double D, double S2, double E){

//   double g = GAMMA_;
//   double Ep2 = (E+p)*(E+p);
//   double v2 = S2/Ep2;
//   double rhoe = E*(1.-v2) - D*sqrt(fabs(1.-v2)) - p*v2;
//   double pNew = (g-1.)*rhoe;

//   double oe = 1./(sqrt(fabs(1.-v2))*E/D - 1. - p/D*v2/sqrt(fabs(1.-v2)) );
//   double c2 = (g-1.)/(1.+oe/g);

//   *f = pNew - p;
//   *dfdp = v2*c2 - 1.;

// }

// void FluidState::cons2prim(double r, double pin){

//   UNUSED(pin);

//   double D = cons[DEN];
//   double tau = cons[TAU];
//   double E = D+tau;
//   double ss[NUM_D];
//   double S2 = 0.;

//   for (int d = 0; d < NUM_D; ++d) ss[d] = cons[SS1+d];
//   ss[t_] /= r;
//   for (int d = 0; d < NUM_D; ++d) S2 += ss[d]*ss[d];

//   double p = prim[PPP];
//   double f, dfdp;
//   f_eval(p, &f, &dfdp, D, S2, E);
//   int iter = 0;
//   while (fabs(f/p) > 1.e-13 and iter < 100){
//     p -= f/dfdp;
//     f_eval(p, &f, &dfdp, D, S2, E);
//     iter++;
//   }

//   double Ep2 = (E+p)*(E+p);
//   double v2 = S2/Ep2;
//   double lfac = 1./sqrt(fabs(1.-v2));
//   double rho = D/lfac;

//   double uu[NUM_D];
//   for (int d = 0; d < NUM_D; ++d){ 
//     double v = (ss[d]/(E+p));
//     uu[d] = lfac*v;
//   }

//   cons2prim_user(&rho, &p, uu);

//   prim[RHO] = rho;
//   prim[PPP] = p;
//   for (int d = 0; d < NUM_D; ++d){ 
//     prim[UU1+d] = uu[d];
//   }

//   prim2cons(r); 
//   state2flux(r);

// }

struct f_params{
  double D,S,E;
};

// This is the function we need to set to zero:
static double f(double p, void *params){

  // parsing parameters
  struct f_params *par = (struct f_params *) params;
  double  D = par->D;
  double  S = par->S;
  double  E = par->E;
  double  lfac = 1. / sqrt(fabs(1. - (S*S) / ((E+p) * (E+p))));

  FluidState state;
  state.prim[RHO] = D/lfac;
  state.prim[PPP] = p;
  double gma = state.gamma();
    // Meliani+(2008) eq. A.9
    // v doesn't come into play as these are comoving quantities.

  return( (E + p - D * lfac - (gma * p * lfac * lfac) / (gma - 1.)) );
    // Mignone (2006) eq. 5

}

/*
bool cons2primNormal(double D, double mm, double E,
  double gma, double pgas_old,
  double *rho, double *uu, double *p){

  // Parameters
  int max_iterations = 15;
  double tol = 1.0e-12;
  double pgas_uniform_min = 1.0e-12;
  double a_min = 1.0e-12;
  double v_sq_max = 1.0 - 1.0e-12;
  double rr_max = 1.0 - 1.0e-12;

  // Extract conserved values
  // double D = cons[DEN];         // dd in athena 
  // double tau = cons[TAU];
  // double E = D+tau;             // ee in athena
  // double mm = cons[SS1];
  double mm_sq = mm*mm;
  // double gma = 4./3.;

  // Calculate functions of conserved quantities
  double pgas_min = -E;
  pgas_min = std::max(pgas_min, pgas_uniform_min);

  // Iterate until convergence
  double pgas[3];
  //double pgas_old;
  //pgas_old = *p;
  pgas[0] = std::max(pgas_old, pgas_min);
  int n;
  for (n = 0; n < max_iterations; ++n) {
    // Step 1: Calculate cubic coefficients
    double a;
    if (n%3 != 2) {
      a = E + pgas[n%3];      // (NH 5.7)
      a = std::max(a, a_min);
    }

    // Step 2: Calculate correct root of cubic equation
    double v_sq;
    if (n%3 != 2) {
      v_sq = mm_sq / (a*a);                                     // (NH 5.2)
      v_sq = std::min(std::max(v_sq, 0.), v_sq_max);
      double lfac2 = 1.0/(1.0-v_sq);                            // (NH 3.1)
      double lfac = std::sqrt(lfac2);                           // (NH 3.1)
      double wgas = a/lfac2;                                    // (NH 5.1)
      double rho = D/lfac;                                      // (NH 4.5)
      pgas[(n+1)%3] = (gma-1.0)/gma * (wgas - rho);             // (NH 4.1)
      pgas[(n+1)%3] = std::max(pgas[(n+1)%3], pgas_min);
    }

    // Step 3: Check for convergence
    if (n%3 != 2) {
      if (pgas[(n+1)%3] > pgas_min && std::abs(pgas[(n+1)%3]-pgas[n%3]) < tol) {
        break;
      }
    }

    // Step 4: Calculate Aitken accelerant and check for convergence
    if (n%3 == 2) {
      double rr = (pgas[2] - pgas[1]) / (pgas[1] - pgas[0]);    // (NH 7.1)
      if (!std::isfinite(rr) || std::abs(rr) > rr_max) {
        continue;
      }
      pgas[0] = pgas[1] + (pgas[2] - pgas[1]) / (1.0 - rr);     // (NH 7.2)
      pgas[0] = std::max(pgas[0], pgas_min);
      if (pgas[0] > pgas_min && std::abs(pgas[0]-pgas[2]) < tol) {
        break;
      }
    }
    
  }

  // Step 5: Set primitives
  if (n == max_iterations) {
    return false;
  }
  double prs = pgas[(n+1)%3];
  *p = prs;
  if (!std::isfinite(prs)) {
    return false;
  }
  double a = E + prs;                                 // (NH 5.7)
  a = std::max(a, a_min);
  double v_sq = mm_sq / (a*a);                        // (NH 5.2)
  v_sq = std::min(std::max(v_sq, 0.), v_sq_max);
  double lfac2 = 1.0/(1.0-v_sq);                      // (NH 3.1)
  double lfac = std::sqrt(lfac2);                     // (NH 3.1)
  double dty = D/lfac;
  *rho = dty;                                         // (NH 4.5)
  if (!std::isfinite(dty)) {
    return false;
  }
  double v = mm / a;                                  // (NH 4.6)
  *uu = lfac*v;                                       // (NH 3.3)
  double u1 = *uu;
  if (!std::isfinite(u1)) {
    return false;
    }
  //*nf_lfac = lfac;
  return true;
}


void FluidState::cons2prim(double r, double pin){
  if (C2P_ == NH14_){
    // conservative to primitive conversion from Newman & Hamlin (2014)


    // Extract conserved values
    double D = cons[DEN];         // dd in athena 
    double tau = cons[TAU] + D;
    double E = D+tau;             // ee in athena
    double mm = cons[SS1];
    double mm_sq = mm*mm;
    double gma = gamma();
    double rho;
    double uu;
    double p;

    bool success = cons2primNormal(D, mm, E,
      gma, pgas_old, rho, uu, p);

    // Handle failures
    if (!success) {
      rho = rho_old;
      p = pgas_old;
      uu = u1_old;
      //fixed = true;
    }
  }

  else if (C2P_ == M06_){
    // conservative to primitive conversion from Mignone et al. 2006

    int     status;
    int     iter = 0, max_iter = 1000;
    double  res;
    struct  f_params                params;
    const   gsl_root_fsolver_type   *T;
    gsl_root_fsolver                *s;
    gsl_function                    F;

    F.function = &f;
    F.params = &params;

    // setting initial parameters
    double D = cons[DEN];
    double tau = cons[TAU];
    double E = D+tau;
    double ss[NUM_D];
    double S2 = 0.;

    for (int d = 0; d < NUM_D; ++d) ss[d] = cons[SS1+d];
    ss[t_] /= r;
    for (int d = 0; d < NUM_D; ++d) S2 += ss[d]*ss[d];

    double S = sqrt(S2);

    params.D = D;
    params.S = S;
    params.E = E;

    double pCand;
    if (pin != 0) pCand = pin;
    else pCand = fmax(1.e-13,prim[PPP]);

    // Looking for pressure only if current one doesn't work;
    double f_init = f(pCand, &params);
    if (fabs(f_init) > 0.) {

      double p_lo, p_hi;
      T = gsl_root_fsolver_brent;
      s = gsl_root_fsolver_alloc (T);

      // setting boundaries
      // f(p_lo) has to be positive. No solution otherwise
      // hence, f(p_hi) has to be negative
      p_lo = fmax(0., (1. + 1e-13) * S - E);
        // the endpoints not straddling can come from here (when p_lo is set to zero)
      bool find_p = true;
      if (f(p_lo, &params)<0) { find_p = false; } // no solution

      if (f_init < 0){
        p_hi = 1.*pCand;
      }
      else {
        p_hi = pCand;
        while (f(p_hi, &params) > 0) {p_hi *= 10;}
      }

      if (p_hi<p_lo) {find_p = false; } // no solution

      if (find_p){
        gsl_root_fsolver_set (s, &F, p_lo, p_hi);
        do {
          iter++;
          status = gsl_root_fsolver_iterate (s);
          res = gsl_root_fsolver_root (s);
          p_lo = gsl_root_fsolver_x_lower (s);
          p_hi = gsl_root_fsolver_x_upper (s);
          status = gsl_root_test_interval (p_lo, p_hi, 0., 1.e-13);
        } while (status == GSL_CONTINUE && iter < max_iter);
        gsl_root_fsolver_free (s);
        pCand = res;
      }
    }

    double p = pCand;
    double Ep2 = (E+p)*(E+p);
    double v2 = S2/Ep2;
    double lfac = 1./sqrt(fabs(1.-v2)); // fabs in case of non-physical state
    double rho = D/lfac;

    double uu[NUM_D];
    if (fabs(S) == 0.) {
      for (int d = 0; d < NUM_D; ++d) uu[d] = 0;
    }
    else {
      double v = sqrt(1.- 1./(lfac*lfac));
      for (int d = 0; d < NUM_D; ++d) uu[d] = lfac*v*(ss[d]/S);
        // Mignone (2006) eq. 3
    }
  } 

  cons2prim_user(&rho, &p, uu);

  // no matter what we did with the rest, we can't allow the pressure to drop to zero:
  p = fmax(p,P_FLOOR_);

  prim[PPP] = p;
  prim[RHO] = rho;
  for (int d = 0; d < NUM_D; ++d){ 
    prim[UU1+d] = uu[d];
  }
  for (int t = 0; t < NUM_T; ++t){
    prim[TR1+t] = cons[TR1+t] / D;
  }

  prim2cons(r); // for consistency
 
}
*/


void FluidState::cons2prim(double r, double pin){

  int     status;
  int     iter = 0, max_iter = 1000;
  double  res;
  struct  f_params                params;
  const   gsl_root_fsolver_type   *T;
  gsl_root_fsolver                *s;
  gsl_function                    F;

  F.function = &f;
  F.params = &params;

  // setting initial parameters
  double D = cons[DEN];
  double tau = cons[TAU];
  double E = D+tau;
  double ss[NUM_D];
  double S2 = 0.;

  for (int d = 0; d < NUM_D; ++d) ss[d] = cons[SS1+d];
  ss[t_] /= r;
  for (int d = 0; d < NUM_D; ++d) S2 += ss[d]*ss[d];

  double S = sqrt(S2);

  params.D = D;
  params.S = S;
  params.E = E;

  double pCand;
  if (pin != 0) pCand = pin;
  else pCand = fmax(1.e-13,prim[PPP]);

  // Looking for pressure only if current one doesn't work;
  double f_init = f(pCand, &params);
  if (fabs(f_init) > 0.) {

    double p_lo, p_hi;
    T = gsl_root_fsolver_brent;
    s = gsl_root_fsolver_alloc (T);

    // setting boundaries
    // f(p_lo) has to be positive. No solution otherwise
    // hence, f(p_hi) has to be negative
    p_lo = fmax(0., (1. + 1e-13) * S - E);
      // the endpoints not straddling can come from here (when p_lo is set to zero)
    bool find_p = true;
    if (f(p_lo, &params)<0) { find_p = false; } // no solution

    if (f_init < 0){
      p_hi = 1.*pCand;
    }
    else {
      p_hi = pCand;
      while (f(p_hi, &params) > 0) {p_hi *= 10;}
    }

    if (p_hi<p_lo) {find_p = false; } // no solution

    if (find_p){
      gsl_root_fsolver_set (s, &F, p_lo, p_hi);
      do {
        iter++;
        status = gsl_root_fsolver_iterate (s);
        res = gsl_root_fsolver_root (s);
        p_lo = gsl_root_fsolver_x_lower (s);
        p_hi = gsl_root_fsolver_x_upper (s);
        status = gsl_root_test_interval (p_lo, p_hi, 0., 1.e-13);
      } while (status == GSL_CONTINUE && iter < max_iter);
      gsl_root_fsolver_free (s);
      pCand = res;
    }
  }

  double p = pCand;
  double Ep2 = (E+p)*(E+p);
  double v2 = S2/Ep2;
  double lfac = 1./sqrt(fabs(1.-v2)); // fabs in case of non-physical state
  double rho = D/lfac;

  double uu[NUM_D];
  if (fabs(S) == 0.) {
    for (int d = 0; d < NUM_D; ++d) uu[d] = 0;
  }
  else {
    double v = sqrt(1.- 1./(lfac*lfac));
    for (int d = 0; d < NUM_D; ++d) uu[d] = lfac*v*(ss[d]/S);
      // Mignone (2006) eq. 3
  }

  cons2prim_user(&rho, &p, uu);

  // no matter what we did with the rest, we can't allow the pressure to drop to zero:
  p = fmax(p,P_FLOOR_);

  prim[PPP] = p;
  prim[RHO] = rho;
  for (int d = 0; d < NUM_D; ++d){ 
    prim[UU1+d] = uu[d];
  }
  for (int t = 0; t < NUM_T; ++t){
    prim[TR1+t] = cons[TR1+t] / D;
  }

  prim2cons(r); // for consistency
  
}


void Interface::wavespeedEstimates(){

  int    u = UU1+dim;
  double lfacL = SL.lfac();
  double lfacR = SR.lfac();
  double vL = SL.prim[u]/lfacL;
  double vR = SR.prim[u]/lfacR;

  // Computing speed of sound parameter sigmaS:
  double cSL = sqrt(GAMMA_ * SL.prim[PPP] 
    / (SL.prim[RHO] + SL.prim[PPP]*GAMMA_/(GAMMA_-1.)));
  double cSR = sqrt(GAMMA_ * SR.prim[PPP] 
    / (SR.prim[RHO] + SR.prim[PPP]*GAMMA_/(GAMMA_-1.)));
    // Mignone (2006) eq. 4
  double sgSL = cSL * cSL / (lfacL * lfacL * (1. - cSL * cSL));
  double sgSR = cSR * cSR / (lfacR * lfacR * (1. - cSR * cSR));
    // Mignone (2006) eq. 22-23s

  // Retreiving L and R speeds:
  double  l1, l2;       // 2 intermediates
  l1 = (vR - sqrt(sgSR * (1. - vR * vR + sgSR))) / (1. + sgSR);
  l2 = (vL - sqrt(sgSL * (1. - vL * vL + sgSL))) / (1. + sgSL);
  lL = fmin(l1,l2);

  l1 = (vR + sqrt(sgSR * (1. - vR * vR + sgSR))) / (1. + sgSR);
  l2 = (vL + sqrt(sgSL * (1. - vL * vL + sgSL))) / (1. + sgSL);
  lR = fmax(l1,l2);

}

void Interface::computeLambda(){

  int    u = UU1+dim;
  int    s = SS1+dim;
  double d;
  if (dim == t_) d = x[r_];
  else           d = 1.;

  wavespeedEstimates();
  
  double lfacL = SL.lfac();
  double lfacR = SR.lfac();
  double vL = SL.prim[u]/lfacL;
  double vR = SR.prim[u]/lfacR;
  double mmL = SL.cons[s]/d;
  double mmR = SR.cons[s]/d;
  double pL = SL.prim[PPP];
  double pR = SR.prim[PPP];
  double EL = SL.cons[TAU]+SL.cons[DEN];
  double ER = SR.cons[TAU]+SR.cons[DEN];

  // Setting up temporary variables:
  double AL = lL * EL - mmL;
  double AR = lR * ER - mmR;
  double BL = mmL * (lL - vL) - pL;
  double BR = mmR * (lR - vR) - pR;
    // Mignone (2006) eq. 17

  // Coefficients for lambdaStar equation:
  double FhllE = (lL * AR - lR * AL) / (lR - lL);
  double Ehll  = (AR - AL) / (lR - lL);
  double Fhllm = (lL * BR - lR * BL) / (lR - lL);
  double mhll  = (BR - BL) / (lR - lL);

  if (fabs(FhllE) == 0){
    lS = mhll / (Ehll + Fhllm);
    return;
  }

  double delta = pow(Ehll + Fhllm,2.) - 4. * FhllE * mhll;
  lS = ((Ehll + Fhllm) - sqrt(delta)) / (2. * FhllE);

}


FluidState Interface::starState(FluidState Sin, double lbda){

  int u,s, uu[2], ss[2];
  if (dim == x_) {
    u = UU1 + x_;
    s = SS1 + x_;
    uu[0] = UU1 + y_; uu[1] = UU1 + z_;
    ss[0] = SS1 + y_; ss[1] = SS1 + z_;
  }
  else if (dim == y_) {
    u = UU1 + y_;
    s = SS1 + y_;
    uu[0] = UU1 + x_; uu[1] = UU1 + z_;
    ss[0] = SS1 + x_; ss[1] = SS1 + z_;
  }
  else {
    u = UU1 + z_;
    s = SS1 + z_;
    uu[0] = UU1 + x_; uu[1] = UU1 + y_;
    ss[0] = SS1 + x_; ss[1] = SS1 + y_;
  }

  double r = x[r_];
  double mm[NUM_D];
  for (int d = 0; d < NUM_D; ++d) mm[d] = Sin.cons[SS1+d];
  mm[t_] /= r;

  double lfac = Sin.lfac();
  double v = Sin.prim[u]/lfac;
  double m = mm[dim];
  double p = Sin.prim[PPP];
  double D = Sin.cons[DEN];
  double E = Sin.cons[TAU]+Sin.cons[DEN];
  double tr[NUM_T];
  for (int t = 0; t < NUM_T; ++t) tr[t] = Sin.cons[TR1+t];

  double A = lbda * E - m;
  double B = m * (lbda-v) - p;
  FluidState Sout;
  double pS = (A*lS - B) / (1. - lbda*lS);
  double k = (lbda-v) / (lbda - lS);
  Sout.cons[DEN] = D * k;
  Sout.cons[s]   = (m * (lbda-v) + pS - p) / (lbda-lS); 
  for (int n = 0; n < NUM_D-1; ++n){ Sout.cons[ss[n]] = mm[ss[n]-SS1] * k; }
  Sout.cons[SS1+t_] *= r;
  Sout.cons[TAU] = (E * (lbda-v) + pS * lS - p * v) / (lbda - lS) - D*k;
  for (int t = 0; t < NUM_T; ++t) Sout.cons[TR1+t] = tr[t] * k;
    // Mignone (2006) eq. 16 (-D)
    // Mignone (2006) eq. 17

  return Sout;
  // no need to compute the flux, will be done in the solver function

}

#if NUM_D == 1
  void Cell::sourceTerms(double dt){

    double p   = S.prim[PPP];
    double r   = G.x[r_];
    double dr  = G.dx[r_];
    double dV  = G.dV;
    double r1  = r-dr/2.;
    double r2  = r+dr/2.;
    double rsq = (r2*r2+r1*r1+r2*r1)/3.;

    S.cons[SS1] += (2.*p) *(r/rsq) * dV * dt;

  }
#endif

#if NUM_D == 2
  void Cell::sourceTerms(double dt){

    double rho = S.prim[RHO];
    double ut  = S.prim[UU1+t_];
    double p   = S.prim[PPP];
    double gma = S.gamma();
    double h   = 1.+p*gma/(gma-1.)/rho; // look at this expression
    double r   = G.x[r_];
    double dr  = G.dx[r_];
    double th  = G.x[t_];
    double dth = G.dx[t_];
    double dV  = G.dV;
    double r1  = r-dr/2.;
    double r2  = r+dr/2.;
    double th1 = th-dth/2.;
    double th2 = th+dth/2.;
    double rsq = (r2*r2+r1*r1+r2*r1)/3.;

    S.cons[SS1] += (rho*h*ut*ut + 2.*p) *(r/rsq) * dV * dt;
    // S.cons[SS1] += (rho*ut*ut + 2.*p) * (r/rsq)* dV * dt;
    // S.cons[SS1] += (rho*h*ut*ut + 2.*p) / r * dV * dt;
    // S.cons[SS1] += (rho*ut*ut + 2.*p) / r * dV * dt;
    S.cons[SS2] += p*cos(th)/fabs(sin(th)) * dV * dt;
      // fabs to cope with negative thetas (ghost cells)
    // S.cons[SS2] += 2./3.* PI * p * (r2*r2*r2 - r1*r1*r1) * (sin(th2) - sin(th1)) * dt;

  }
#endif













