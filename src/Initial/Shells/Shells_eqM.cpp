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
static double lfacb = 250 ;           // LF at the back of shell
static double lfacf = 95 ;           // LF at the front of the shell
static double tfrac = 0.4 ;           // fraction of tw of the fast shell
static double betab = sqrt(1 - pow(lfacb, -2)) ;
static double Rmin = 4e4 * R0 ;       // back end of the shell
static double Rint = Rmin + tfrac*tw*c_ ;
static double Rmax = Rmin + tw * c_ ; // front end of the shell
static int Ncells = 1000 ;
static int Next = 0 ;
static double gma1 = GAMMA_/(GAMMA_-1) ;
static double corrf = betab * c_ * (1 + theta0*(gma1 - pow(lfacb, -2)));
static double rhof = L/(4*M_PI*Rmin*Rmin*lfacb*lfacb*c_*c_*corrf);
// static double rhoint = L/(4*M_PI*Rint*Rint*lfacb*lfacb*c_*c_*corrf);
// static double pint = theta0*rhoint ;

static void calcRho(double r, double lfac, double *rho){
  // derives density from LF
  double lfac2 = lfac*lfac ;
  double beta = sqrt(1 - 1/lfac2) ;
  double corr = beta * c_ * (1 + theta0*(gma1 - 1/lfac2));
  *rho = L/(4*M_PI*r*r*lfac2*c_*c_*corr);
}

// normalisation constants:
static double rhoNorm = 1e-6*rhof ;                // density normalised to CBM density
static double lNorm = c_;                     // distance normalised to c
static double vNorm = c_;                     // velocity normalised to c
static double pNorm = rhoNorm*vNorm*vNorm;    // pressure normalised to rho_CMB/c^2

void loadParams(s_par *par){

  par->tini      = 0.;
  par->ncell[x_] = Ncells+Next;
  par->nmax      = 2*Ncells;    // max number of cells in MV direction
  par->ngst      = 2;

  normalizeConstants(rhoNorm, vNorm, lNorm);

}


int Grid::initialGeometry(){
  double dR = ((Rmax-Rmin)/Ncells)/lNorm ;
  double rmin = Rmin/lNorm - Next*dR ;
  for (int i = 0; i < Ncells+Next; ++i){
    Cell *c = &Cinit[i];
    c->G.x[x_]  = (double) rmin + (i + 0.5)*dR ;
    c->G.dx[x_] = dR ;
    c->computeAllGeom();
  }
  
  return 0;

}

int Grid::initialValues(){
  cout << "Density at back of shell = " << rhof << " g cm^-3\n" ;
  double rhoint, p_int ;
  calcRho(Rint, lfacb, &rhoint) ;
  p_int = theta0*rhoint ;

  for (int i = 0; i < ncell[MV]; ++i){
    Cell *c = &Cinit[i];
    double x = c->G.x[x_];
    double r = x*lNorm;
    double t = (Rmax - r)/c_ ;

    if (t > tw){
      //double rho ;
      //calcRho(Rmin, lfacb, &rho);
      c->S.prim[RHO] = 1e-3*rhof/rhoNorm ;
      c->S.prim[VV1] = betab ;
      c->S.prim[PPP] = 1e-3*p_int/rhoNorm ; //1e-3*rhof*theta0/rhoNorm ;
      c->S.prim[TR1] = 1.;
    }
    if (t >= tfrac*tw){
      double rho ;
      calcRho(r, lfacb, &rho);
      c->S.prim[RHO] = rho/rhoNorm ;
      c->S.prim[VV1] = betab;//0.9*betab ;
      c->S.prim[PPP] = rho*theta0/rhoNorm ;
      c->S.prim[TR1] = 1.;
    }
    else{
      double lfac = (lfacb-lfacf) - lfacf*cos(M_PI*t/(tfrac*tw)) ;
      double beta = sqrt(1 - pow(lfac, -2)) ;
      double rho ;
      calcRho(r, lfac, &rho);
      c->S.prim[RHO] = rho/rhoNorm ;
      c->S.prim[VV1] = beta;//0.9*beta ;
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

  // for (int i = 0; i <= iLbnd; ++i){
  //   Cell *c = &Ctot[i];
  //   double rho, p;
  //   double r = c->G.x[r_]*lNorm;

  //   calcRho(r, lfacb, &rho);
  //   //rho = 1e-3*rhof ; //*pow(r/Rmin, -2);
  //   p = rho*theta0/rhoNorm ;
  //   c->S.prim[RHO] = rho/rhoNorm;
  //   c->S.prim[UU1] = 0.8*betab ;
  //   c->S.prim[PPP] = p;
  //   c->S.prim[TR1] = 0.;
  // }

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

  if (it % 100 == 0){ grid.printCols(it, t); }

}

void Simu::runInfo(){

  if ((worldrank == 0) and (it%100 == 0)){ printf("it: %ld time: %le\n", it, t);}

}

void Simu::evalEnd(){

  // if ( it > 100 ){ stop = true; }
  if (t > 2e6){ stop = true; } // 3.33e8 BOXFIT simu

}
