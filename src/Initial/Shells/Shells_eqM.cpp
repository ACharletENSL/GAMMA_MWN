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

// phys parameters
static double tw = 10 ;               // (s), wind injection duration
static double R0 = 3e+6 ;             // (cm) normalization radius
static double L = 2e+51 ;             // (erg/s) injection power
static double theta0 = 1e-3 ;         // relat. temperature p/rho c²
static double lfacf = 400 ;           // LF at the end of shell

static double betaf = sqrt(1 - pow(lfacf, -2)) ;
static double Rmin = 400 * R0 ;       // back end of the shell
static double Rmax = Rmin + tw * c_ ; // front end of the shell
static int Ncells = 1000 ;
static double gma1 = GAMMA_/(GAMMA_-1) ;
static double corrf = betaf * c_ * (1 + theta0*(gma1 - pow(lfacf, -2)));
static double rhof = L/(4*M_PI*Rmin*Rmin*lfacf*lfacf*c_*c_*corrf);


static void calcRho(double r, double lfac, double *rho){
  // derives density from LF
  double lfac2 = lfac*lfac ;
  double beta = sqrt(1 - 1/lfac2) ;
  double corr = beta * c_ * (1 + theta0*(gma1 - 1/lfac2));
  *rho = L/(4*M_PI*r*r*lfac2*c_*c_*corr);
}

// normalisation constants:
static double rhoNorm = rhof ;                // density normalised to CBM density
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


int Grid::initialGeometry(){
  double rmin = Rmin/lNorm;
  double dR = ((Rmax-Rmin)/Ncells)/lNorm ;
  for (int i = 0; i < Ncells; ++i){
    Cell *c = &Cinit[i];
    c->G.x[x_]  = (double) rmin + (i + 0.5)*dR ;
    c->G.dx[x_] = dR ;
    c->computeAllGeom();
  }
  
  return 0;

}

int Grid::initialValues(){

  for (int i = 0; i < ncell[MV]; ++i){
    Cell *c = &Cinit[i];
    double x = c->G.x[x_];
    double r = x*lNorm;
    double t = (Rmax - r)/c_ ;

    if (t >= 0.4*tw){
      double rho ;
      calcRho(r, lfacf, &rho);
      c->S.prim[RHO] = rho/rhoNorm ;
      c->S.prim[VV1] = betaf ;
      c->S.prim[PPP] = rho*theta0/rhoNorm ;
      c->S.prim[TR1] = 1.;
    }
    else{
      double lfac = 250 - 150*cos(M_PI*t/(0.4*tw)) ;
      double beta = sqrt(1 - pow(lfac, -2)) ;
      double rho ;
      calcRho(r, lfac, &rho);
      c->S.prim[RHO] = rho/rhoNorm ;
      c->S.prim[VV1] = beta ;
      c->S.prim[PPP] = rho*theta0/rhoNorm ;
      c->S.prim[TR1] = 2.;
    }
    
    c->S.prim[TR1+1] = 0.;
  }

  return 0;

}

void Grid::userKinematics(int it, double t){

  UNUSED(it);
  UNUSED(t);

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
  UNUSED(i);

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

  if (it % 1 == 0){ grid.printCols(it, t); }

}

void Simu::runInfo(){

  if ((worldrank == 0) and (it%100 == 0)){ printf("it: %ld time: %le\n", it, t);}

}

void Simu::evalEnd(){

  if ( it > 10 ){ stop = true; }
  //if (t > 3.33e8){ stop = true; } // 3.33e8 BOXFIT simu

}
