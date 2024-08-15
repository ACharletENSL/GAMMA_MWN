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

// geometry: cartesian or spherical
static int GEOMETRY_  = 0 ;           // 0 for cartesian, 0 for spherical

// set CBM parameters
static double n0      = 1.;           // cm-3:    CBM number density
static double rho0    = n0*mp_;       // g.cm-3:  comoving CBM mass density
static double Theta0  = 0.0001 ;   //          Theta0 = p/(rho*c^2)
static double p0      = Theta0*rho0*c_*c_;

// set shells parameters
static double rho1 = 4.667291079080062e-12 ;     // comoving density of front shell
static double u1   = 1e+02 ;          // proper velocity (gamma*beta) of front shell
static double p1   = 104345.78059068859 ;
static double D01  = 2997774695.012281 ;     // spatial extension of front shell
static double rho4 = 1.1610033862318822e-12 ;     // comoving density of back shell
static double u4   = 2e+02 ;          // proper velocity of back shell
static double p4   = 104345.78059068859 ;
static double D04  = 2997887106.645374 ;     // spatial extension of back shell
static double beta1= u1/sqrt(1+u1*u1);
static double beta4= u4/sqrt(1+u4*u4);
static double cont = 0.05 ;           // density contrast between shell and ext medium

// box size
static double R_0     = 79947153684166.38 ;
static int Nsh1   = 450 ;
static int Ntot1  = 500 ;
static int Nsh4   = 450 ;
static int Ntot4  = 500 ;
static int Ncells = Ntot4 + Ntot1;

// normalisation constants:
static double rhoNorm = rho1 ;                // density normalised to CBM density
static double lNorm = c_;                     // distance normalised to c
static double vNorm = c_;                     // velocity normalised to c
static double pNorm = rhoNorm*vNorm*vNorm;    // pressure normalised to rho_CMB/c^2


void loadParams(s_par *par){

  par->tini      = 0.;
  par->ncell[x_] = Ncells;
  par->nmax      = 2*Ncells;    // max number of cells in MV direction
  par->ngst      = 2;

  normalizeConstants(rhoNorm, vNorm, lNorm);

}

/*
We want shell edge to coincide perfectly with cell edge,
but also keep resolution as constant as possible across the zones
-> do this linear grid over 4+external medium then 1+external medium
Nsh4 = ceil(Nsh1*D04/D01)
don't forget to set ext medium in a way it won't have to create cells immediately
*/
/*
int Grid::initialGeometry(){

  double x = (rmax0 - rmin0)/lNorm;
  for (int i = 0; i < ncell[x_]; ++i){
    Cell *c = &Cinit[i];
    c->G.x[x_]  = (double) x*(i+0.5)/ncell[x_] + rmin0/lNorm;
    c->G.dx[x_] =          x/ncell[x_];
    c->computeAllGeom();
  }
  return 0;

}*/

int Grid::initialGeometry(){

  double x1 = D01/lNorm;
  double x4 = D04/lNorm;
  for (int i = 0; i < Ncells; ++i){
    Cell *c = &Cinit[i];
    if (i <= Ntot4){
      c->G.x[x_]  = (double) x4*(i-Ntot4+0.5)/Nsh4 + R_0/lNorm;
      c->G.dx[x_] =          x4/Nsh4;
    }
    else {
      c->G.x[x_]  = (double) x1*(i-Ntot4+0.5)/Nsh1 + R_0/lNorm;
      c->G.dx[x_] =          x1/Nsh1;
    }
    c->computeAllGeom();
  }
  
  return 0;

}

int Grid::initialValues(){

  for (int i = 0; i < ncell[MV]; ++i){
    Cell *c = &Cinit[i];
    double x = c->G.x[x_];
    double r = x*lNorm;

    if (GEOMETRY_ == 0){ // cartesian geometry
      // std::cout << "Cartesian geometry";
      if (r <= R_0-D04){
        c->S.prim[RHO] = cont*rho4/rhoNorm;
        c->S.prim[VV1] = beta4;
        c->S.prim[PPP] = p4/pNorm;
        c->S.prim[TR1] = 0.;
      }
      if ((r >= R_0-D04) and (r <= R_0)){
        c->S.prim[RHO] = rho4/rhoNorm;
        c->S.prim[VV1] = beta4;
        c->S.prim[PPP] = p4/pNorm;
        c->S.prim[TR1] = 1.;
      }
      if ((r > R_0) and (r <= R_0+D01)){
        c->S.prim[RHO] = rho1/rhoNorm;
        c->S.prim[VV1] = beta1;
        c->S.prim[PPP] = p1/pNorm;
        c->S.prim[TR1] = 2.;
      }
      if (r > R_0+D01){
        c->S.prim[RHO] = cont*rho1/rhoNorm;
        c->S.prim[VV1] = beta1;
        c->S.prim[PPP] = p1/pNorm;
        c->S.prim[TR1] = 0.;
      }
    }
    else if (GEOMETRY_ == 1){ // spherical geometry
      // std::cout << "Spherical geometry";
      double R4 = R_0 - D04;
      double R1 = R_0 + D01;
      double gma = 5./3.;

      if (r <= R4){
        double rho = cont*rho4*pow(r/R4, -2.);
        //double p   = p4*pow(r/R4, -2*gma);
        c->S.prim[RHO] = rho/rhoNorm;
        c->S.prim[VV1] = beta4;
        c->S.prim[PPP] = p4/pNorm;
        c->S.prim[TR1] = 0.;
      }
      if ((r >= R4) and (r <= R_0)){
        //double rho = rho4*pow(r/R_0, -2.);
        c->S.prim[RHO] = rho4/rhoNorm;
        c->S.prim[VV1] = beta4;
        c->S.prim[PPP] = p4/pNorm;
        c->S.prim[TR1] = 1.;
      }
      if ((r > R_0) and (r <= R1)){
        //double rho = rho1*pow(r/R_0, -2.);
        c->S.prim[RHO] = rho1/rhoNorm;
        c->S.prim[VV1] = beta1;
        c->S.prim[PPP] = p1/pNorm;
        c->S.prim[TR1] = 2.;
      }
      if (r > R1){
        double rho = cont*rho1*pow(r/R1, -2.);
        //double p   = p1*pow(r/R1, -2*gma);
        c->S.prim[RHO] = rho/rhoNorm;
        c->S.prim[VV1] = beta1;
        c->S.prim[PPP] = p1/pNorm;
        c->S.prim[TR1] = 0.;
      }
    }
    c->S.prim[TR1+1] = 0.;
  }

  return 0;

}

void Grid::userKinematics(int it, double t){

  UNUSED(it);
  UNUSED(t);

  // // DEFAULT VALUES
  // //double vIn  = 1.;
  // //double vOut = 1.;
  // //double vb = vOut;

  // // check if rarefaction wave(s) too close to boundary
  // int iout = iRbnd-1;
  // int iin  = iLbnd+1;
  // double rout = Ctot[iout].G.x[r_];
  // double rin  = Ctot[iin].G.x[r_];
  // double rlimL = rin + 0.02 * (rout-rin);
  // double rlimR = rin + 0.98 * (rout-rin);

  // // left boundary
  // int ia = iin;
  // double rcand = rin;
  // Cell c = Ctot[ia];
  // double pcand = c.S.prim[PPP];
  // double ptemp;
  // while (rcand < rlimL){
  //   ia++;
  //   c = Ctot[ia];
  //   rcand = c.G.x[r_];
  //   ptemp = std::max(c.S.prim[PPP], pcand);
  //   pcand = ptemp;
  // }
  // if (pcand >= 10.*p4/pNorm){
  //   for (int n = 0; n < ngst; ++n){
  //     int    iL = n;
  //     Itot[iL].v = 0.9;
  // }
  // }
  
  // // right boundary
  // ia = iout;
  // rcand = rout;
  // c = Ctot[ia];
  // pcand = c.S.prim[PPP];
  // while (rcand > rlimR){
  //   ia--;
  //   c = Ctot[ia];
  //   rcand = c.G.x[r_];
  //   ptemp = std::max(c.S.prim[PPP], pcand);
  //   pcand = ptemp;
  // }
  // if (pcand >= 10.*p1/pNorm){
  //   //std::cout << "Shell edge too close to sim edge";
  //   for (int n = 0; n < ngst; ++n){
  //     int    iR = ntrack-2-n;
  //     Itot[iR].v = 1.1;
  //   }
  // }
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
  double dr0 = (D01+D04)/(Nsh1+Nsh4);
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

  if (it % 5 == 0){ grid.printCols(it, t); }

}

void Simu::runInfo(){

  if ((worldrank == 0) and (it%100 == 0)){ printf("it: %ld time: %le\n", it, t);}

}

void Simu::evalEnd(){

  if ( it > 12500 ){ stop = true; }
  //if (t > 3.33e8){ stop = true; } // 3.33e8 BOXFIT simu

}
