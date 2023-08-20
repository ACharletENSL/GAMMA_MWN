/*
  Setup for 1D simulations of colliding shells
  following Minhajur Sk Rahaman's analytical work:
  a "front" shell (subscripts 1) is caught up by a faster, lighter, "back" shell (subscripts 4)
  Author A. Charlet
*/


#include "../../environment.h"
#include "../../grid.h"
#include "../../constants.h"
#include "../../simu.h"

// set CBM parameters
static double n0      = 1.;           // cm-3:    CBM number density
static double rho0    = n0*mp_;       // g.cm-3:  comoving CBM mass density
static double Theta0  = 1.0000e-4 ;  //          Theta0 = p/(rho*c^2)
static double p0      = Theta0*rho0*c_*c_;

// set shells parameters
static double rho1 = 1.0000e-10 ;     // lab frame density of front shell
static double u1   = 1e+02 ;          // proper velocity (gamma*beta) of front shell
static double p1   = 2.2469e+05 ;
static double D01  = 1.0000e+07 ;     // spatial extension of front shell
static double rho4 = 2.5000e-11 ;     // lab frame density of back shell
static double u4   = 2e+02 ;          // proper velocity of back shell
static double p4   = 2.2469e+05 ;
static double D04  = 1.0000e+07 ;        // spatial extension of back shell
static double beta1= u1/sqrt(1+u1*u1);
static double beta4= u4/sqrt(1+u4*u4);

// box size
static double R_0     = 1.0000e+10 ;
static double rmin0   = 9.9890e+09 ; 
static double rmax0   = 1.0011e+10 ;
static double Ncells  = 1000 ;

// normalisation constants:
static double rhoNorm = rho1 ;                // density normalised to CBM density
static double lNorm = c_;                     // distance normalised to c
static double vNorm = c_;                     // velocity normalised to c
static double pNorm = rhoNorm*vNorm*vNorm;    // pressure normalised to rho_CMB/c^2


void loadParams(s_par *par){

  par->tini      = 0.;
  par->ncell[x_] = Ncells;
  par->nmax      = Ncells+500;    // max number of cells in MV direction
  par->ngst      = 2;

}

int Grid::initialGeometry(){

  double x = (rmax0 - rmin0)/lNorm;
  for (int i = 0; i < ncell[x_]; ++i){
    Cell *c = &Cinit[i];
    c->G.x[x_]  = (double) x*(i+0.5)/ncell[x_] + rmin0/lNorm;
    c->G.dx[x_] =          x/ncell[x_];
    c->computeAllGeom();
  }
  return 0;

}

int Grid::initialValues(){

  for (int i = 0; i < ncell[MV]; ++i){
    Cell *c = &Cinit[i];
    double x = c->G.x[x_];
    double r = x*lNorm;

    if (r <= R_0-D01){
      c->S.prim[RHO] = 1e-3*rho4/rhoNorm;
      c->S.prim[VV1] = beta4;
      c->S.prim[PPP] = p4/pNorm;
      c->S.cons[TR1] = 0.;
    }
    if ((r >= R_0-D04) and (r <= R_0)){
      c->S.prim[RHO] = rho4/rhoNorm;
      c->S.prim[VV1] = beta4;
      c->S.prim[PPP] = p4/pNorm;
      c->S.cons[TR1] = 1.;
    }
    if ((r > R_0) and (r <= R_0+D01)){
      c->S.prim[RHO] = rho1/rhoNorm;
      c->S.prim[VV1] = beta1;
      c->S.prim[PPP] = p1/pNorm;
      c->S.cons[TR1] = 2.;
    }
    if (r > R_0+D01){
      c->S.prim[RHO] = 1e-3*rho1/rhoNorm;
      c->S.prim[VV1] = beta1;
      c->S.prim[PPP] = p1/pNorm;
      c->S.cons[TR1] = 0.;
    }
  }

  return 0;

}

void Grid::userKinematics(int it, double t){

  UNUSED(it);
  UNUSED(t);

  // DEFAULT VALUES
  //double vIn  = 1.;
  //double vOut = 1.;
  //double vb = vOut;

  // check if rarefaction wave(s) too close to boundary
  int iout = iRbnd-1;
  int iin  = iLbnd+1;
  double rout = Ctot[iout].G.x[r_];
  double rin  = Ctot[iin].G.x[r_];
  double rlimL = rin + 0.1 * (rout-rin);
  double rlimR = rin + 0.9 * (rout-rin);

  // left boundary
  int ia = iin;
  double rcand = rin;
  Cell c = Ctot[ia];
  double pcand = c.S.prim[PPP];
  double ptemp;
  while (rcand < rlimL){
    ia++;
    c = Ctot[ia];
    rcand = c.G.x[r_];
    ptemp = std::max(c.S.prim[PPP], pcand);
    pcand = ptemp;
  }
  if (pcand >= 10.*p4/pNorm){
    for (int n = 0; n < ngst; ++n){
      int    iL = n;
      Itot[iL].v = 0.9;
  }
  }
  
  // right boundary
  ia = iout;
  rcand = rout;
  c = Ctot[ia];
  pcand = c.S.prim[PPP];
  while (rcand > rlimR){
    ia--;
    c = Ctot[ia];
    rcand = c.G.x[r_];
    ptemp = std::max(c.S.prim[PPP], pcand);
    pcand = ptemp;
  }
  if (pcand >= 10.*p1/pNorm){
    //std::cout << "Shell edge too close to sim edge";
    for (int n = 0; n < ngst; ++n){
      int    iR = ntrack-2-n;
      Itot[iR].v = 1.1;
    }
  }
}

void Cell::userSourceTerms(double dt){

  UNUSED(dt);

}

void Grid::userBoundaries(int it, double t){

  UNUSED(it);
  UNUSED(t);

}

int Grid::checkCellForRegrid(int j, int i){

  UNUSED(j);
  // UNUSED(i);

  int iin  = iLbnd+1;
  int iout = iRbnd-1;
  Cell c  = Ctot[i];
  double r   = c.G.x[r_];
  double dr  = c.G.dx[r_]*lNorm;
  double dr0 = (rmax0-rmin0)/Ncells;
  double ar  = dr/dr0;

  if ((i <= iin + 2) or (i >= iout - 10)){
    if (ar > 2.){return(split_);}
    
  }

  return(skip_);
  
}

void Cell::user_regridVal(double *res){
  
  UNUSED(*res);

}

void FluidState::cons2prim_user(double *rho, double *p, double *uu){

  UNUSED(*rho);
  UNUSED(uu);
  UNUSED(*p);

  return;
}

void Simu::dataDump(){

  // if (it%1 == 0){ grid.printCols(it, t); }
  if (it%10 == 0){ grid.printCols(it, t); }

}

void Simu::runInfo(){

  // if ((worldrank == 0) and (it%1 == 0)){ printf("it: %ld time: %le\n", it, t);}
  if ((worldrank == 0) and (it%100 == 0)){ printf("it: %ld time: %le\n", it, t);}

}

void Simu::evalEnd(){

  if ( it > 25000 ){ stop = true; }
  //if (t > 3.33e8){ stop = true; } // 3.33e8 BOXFIT simu

}
