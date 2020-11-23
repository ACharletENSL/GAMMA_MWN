/*
* @Author: eliotayache
* @Date:   2020-05-05 10:31:06
* @Last Modified by:   Eliot Ayache
* @Last Modified time: 2020-10-28 12:14:28
*/

#include "../environment.h"
#include "../grid.h"
#include "../constants.h"

static double Eiso  = 1.e53;   // erg
static double t90   = 100;     // s
static double eta   = 1.e-3;    
static double n0    = 1.e0;    // cm-3
static double lfac0 = 100;
static double theta0= 0.25;     // rad: jet opening angle
static double rmin  = 2.e12;  // cm : begining of the box at startup
static double rmax  = 10.e12;   // cm : end of the box at startup
static double r0    = 3.e12;  // cm : back of shell
static double r1    = 6.e12;   // cm : head of shell
static double dtheta= 0.25;   // rad; grid opening angle

static double rho0  = n0*mp_;
static double p0    = rho0*c_*c_*eta;

static double rhoNorm = rho0;
static double lNorm = c_;
static double vNorm = c_;
static double pNorm = rhoNorm*vNorm*vNorm;

static double ejecta_regrid_ratio = 0.1;
static double csm_regrid_ratio    = 1;

void loadParams(s_par *par){

  par->tini      = 0.;
  par->ncell[x_] = 300;
  par->ncell[y_] = 1;
  par->nmax      = 1000;    // max number of cells in MV direction
  par->ngst      = 2;

}

// ---------------------------------------------------------------------------------------
// GRB-specific functions




// ---------------------------------------------------------------------------------------
// Simulation functions

int Grid::initialGeometry(){
  // log grid
  double logrmin = log(rmin);
  double logrmax = log(rmax);
  double dlogr = (logrmax - logrmin);

  for (int j = 0; j < ncell[y_]; ++j){
    for (int i = 0; i < ncell[x_]; ++i){
      Cell *c = &Cinit[j][i];
      double rlog  = (double) dlogr*(i+0.5)/ncell[x_] + logrmin;
      double rlogL = (double) dlogr*(i  )/ncell[x_] + logrmin;
      double rlogR = (double) dlogr*(i+1)/ncell[x_] + logrmin;
      double r   = exp(rlog);
      double dr  = exp(rlogR) - exp(rlogL);
      double th  = (double) dtheta*(j+0.5)/ncell[y_];
      double dth = dtheta/ncell[y_];

      c->G.x[x_]  = r / lNorm;
      c->G.dx[x_] = dr / lNorm;
      c->G.x[y_]  = th;
      c->G.dx[y_] = dth;
      c->computeAllGeom();
    }
  }
  return 0;

}

int Grid::initialValues(){

  double lfac02 = lfac0*lfac0;
  double vr1  = sqrt(1-1./(lfac02)) * c_;
  double Edot = Eiso/t90;
  double k = GAMMA_/(GAMMA_-1);
  double c2 = c_*c_;

  for (int j = 0; j < ncell[F1]; ++j){
    for (int i = 0; i < ncell[MV]; ++i){
      Cell *c = &Cinit[j][i];

      double r    = c->G.x[r_]*lNorm; // deNormalise for calculation
      double th   = c->G.x[t_];
      double r2   = r*r;
      double rho1 = Edot / (4*PI*r2*vr1*lfac02*c2*(1 + eta*(k - 1./lfac02)) );
      double p1   = eta*rho1*c2;

      // normalisation
      double rho0_norm = rho0/rhoNorm;
      double p0_norm = p0/(pNorm);
      double rho1_norm = rho1/rhoNorm;
      double p1_norm = p1/(pNorm);
      double vr1_norm = vr1/vNorm;

      if (fabs(th) < theta0 and r0 < r and r < r1){
        c->S.prim[RHO] = rho1_norm;
        c->S.prim[VV1] = vr1_norm;
        c->S.prim[VV2] = 0.;
        c->S.prim[PPP] = p1_norm;
        c->S.prim[TR1] = 1.;
      }
      else{
        c->S.prim[RHO] = rho0_norm;
        c->S.prim[VV1] = 0.0;
        c->S.prim[VV2] = 0.0;
        c->S.prim[PPP] = p0_norm;
        c->S.prim[TR1] = 2.;            
      }
    }
  }

  return 0;

}


void Grid::userKinematics(){

  // setting lower and higher i boundary interface velocities to zero
  // int check_dist = 90;
  double vIn     = 0.8;     // can't be lower than 1 for algo to work
  double vOut    = 1.2;     
  // double rhoInlim = 1.e3;     // threshold to detect back of the ejecta
  // double vOutlim = 0.1;

  // int moveInner = 1.;
  // int moveOuter = 1.;
  // for (int j = 0; j < nde_nax[F1]; ++j){

  //   Cell      Cin  = Ctot[j][iLbnd[j]+check_dist];
  //   Interface Iout = Itot[j][iRbnd[j]-check_dist];

  //   if (Cin.S.prim[RHO] > rhoInlim)  moveInner = 0;
  //   if (Iout.v > vOutlim) moveOuter = 1;

  // }

  // communicating to all nodes
  double allmoveInner = 1;
  double allmoveOuter = 1;
  // double allmoveInner;
  // double allmoveOuter;
  // MPI_Allreduce(&moveInner, &allmoveInner, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  // MPI_Allreduce(&moveOuter, &allmoveOuter, 2, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

  for (int j = 0; j < nde_nax[F1]; ++j){
    for (int n = 0; n <= ngst; ++n){
      int    iL = n;
      int    iR = ntrack[j]-2-n;
      if (allmoveInner) Itot[j][iL].v = vIn;
      if (allmoveOuter) Itot[j][iR].v = vOut;
    }
  }
  // for (int j = 0; j < nde_nax[F1]; ++j){
  //   for (int n = 0; n <= ngst; ++n){
  //     int    iL = n;
  //     int    iR = ntrack[j]-2-n;
  //     Itot[j][iL].v = 0;
  //     Itot[j][iR].v = 0;
  //   }
  // }

}


void Grid::userBoundaries(int it, double t){

  UNUSED(it);
  UNUSED(t);

}


void Cell::userSourceTerms(double dt){

  // // sources have to be expressed in terms of conserved variables
  // double y   = G.x[y_]; 
  // double dV  = G.dV;
  // double rho = S.prim[RHO];
  // double fg = -rho*g*dV*dt;

  // S.cons[SS1] += fg;

}


int Grid::checkCellForRegrid(int j, int i){

  // We always want ncells[x_] cells in the y direction
  // We allow dx in [0.1, 10] around target uniform resolution

  Cell c = Ctot[j][i];

  double trac = c.S.prim[TR1];
  double split_ratio = 0.1;     // relative to target aspect ratio
  double merge_ratio = 0.01;    // relative to target aspect ratio
  double target_ar = 1;
  double dr  = c.G.dx[r_];
  double dth = c.G.dx[t_];
  double r   = c.G.x[r_];
  double ar  = dr / (r*dth);  // aspect ratio

  if (fabs(trac-1) < 0.2){
    split_ratio *= ejecta_regrid_ratio;   // smaller cells in ejecta
    merge_ratio *= ejecta_regrid_ratio;
  }

  if (fabs(trac-2) < 0.5){
    split_ratio *= csm_regrid_ratio;   // bigger cells in csm
    merge_ratio *= csm_regrid_ratio;
  }

  if (ar > split_ratio * target_ar) {
    // printf("%d %d\n", j, i);
    return(split_);
  }
  if (ar < merge_ratio * target_ar) {
    return(merge_);
  }
  return(skip_);

}


void Cell::user_regridVal(double *res){
  // user function to find regrid victims
  // adapts the search to special target resolution requirements
  // depending on the tracer value

  double trac = S.prim[TR1];
  if (fabs(trac-1 < 0.2)){
    *res /= ejecta_regrid_ratio;
  }
  if (fabs(trac-2 < 0.5)){
    *res /= csm_regrid_ratio;
  }

}



void FluidState::cons2prim_user(double *rho, double *p, double *uu){

  UNUSED(uu);
  double ratio = 1.e-5;

  *p = fmax(*p, ratio*p0/pNorm);
  *rho = fmax(*rho, ratio*rho0/rhoNorm);

  return;

}

