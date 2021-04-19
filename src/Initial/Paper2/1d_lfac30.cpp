#include "../../environment.h"
#include "../../grid.h"
#include "../../constants.h"
#include "../../simu.h"

// initial Fireball (FB) setup

static double lfac    = 30.;                 // --:      energy per unit mass ~ coasting Lorentz factor
static double Eiso    = 1.e52;                // erg:     isotropic equivalent energy
static double R0      = 100.;                  // ls:      initial shell width
static double n0      = 1.;                   // cm-3:    CBM proton number density

// simulation settings
static double th_simu = PI/32./200;        // rad:     simulation angle
static double t_ini   = 0.;                   // s:     start time of dynamic simulation
static double rmin    = 0.01;                  // light-seconds, box inner boundary at t_ini
static double rmax    = 150.;                 // light-seconds, box outer boundary at t_ini

static double R03     = R0*R0*R0;
static double c2      = c_*c_;
static double c3      = c_*c_*c_;

// set external medium density-pressure relation
static double eta     = 1.e-5;                // p0 = rho0*c2, since cold external medium

// calculate CBM parameters
static double rho0    = n0*mp_;               // g.cm-3:  CBM lab frame mass density
static double p0      = rho0*eta*c2;

// calculate FB parameters
static double Miso    = Eiso/(lfac*c2);       // g:         FB initial isotropic equivalent mass
static double Viso    = (4./3.)*PI*R03*c3;    // cm3:       FB initial isotropic equivalent volume
static double E_f     = Eiso/Viso;            // erg.cm-3:  FB total energy density: E = tau + D
static double D_f     = Miso/Viso;            // g.cm-3:    FB mass density (lab frame, but initial velocity is 0, so same in comoving frame)
static double E_D_f   = D_f * c2;             // erg.cm-3:  FB rest-mass energy density
static double tau_f   = E_f - E_D_f;          // erg.cm-3:  FB internal energy density

// normalisation constants:
static double rhoNorm   = rho0;               // density normalised to CBM density
static double lNorm   = c_;                   // distance normalised to c
static double vNorm   = c_;                   // distance normalised to c
static double pNorm   = rhoNorm*vNorm*vNorm;  // pressure normalised to rho_0*c2
static double ENorm   = rhoNorm*lNorm*lNorm;  // energy density normalised based on E/V = E/L3 = (M/L3)*L2/T2 = D*L2/T2, T = time, no normalisation

// regridding parameters
static double fin_AR      = 1.;               //      final target AR
static double R1          = 1.e4;             // ls:  radius to use for target_AR calculation: target_AR = fmax(pow(r_AR_1/r, 0.2), 1.)*fin_AR;
static double R2          = 2.e6;
static double split_AR    = 5.;               //      set upper bound on AR as ratio of target_AR
static double merge_AR    = 0.2;              //      set upper bound on AR as ratio of target_AR
static double tMove       = 3.e6;             // s:   time at which inner boundary starts to move


void loadParams(s_par *par){

  par->tini      = t_ini;             // initial time
  par->ncell[x_] = 600;               // number of cells in r direction
  par->nmax      = 4000;              // max number of cells in MV direction
  par->ngst      = 2;                 // number of ghost cells

  normalizeConstants(rhoNorm, vNorm, lNorm);

}

int Grid::initialGeometry(){                              // loop across grid, calculate cell positions and grid spacing, add info to cell struct "c"

// grid defined in length units of light-seconds

  double logrmin  = log(rmin);
  double logrmax  = log(rmax);
  double dlogr    = (logrmax - logrmin);
  
  for (int i = 0; i < ncell[x_]; ++i){
    Cell *c = &Cinit[i];
    
    double rlog   = (double) dlogr*(i+0.5)/ncell[x_] + logrmin;
    double rlogL  = (double) dlogr*(i  )/ncell[x_] + logrmin;
    double rlogR  = (double) dlogr*(i+1)/ncell[x_] + logrmin;
    double r_ls   = exp(rlog);
    double dr_ls  = exp(rlogR) - exp(rlogL);
    
    c->G.x[x_]    = r_ls;
    c->G.dx[x_]   = dr_ls;
    
    c->computeAllGeom();
  }
  return 0;
}

int Grid::initialValues(){

  if (GAMMA_ != 4./3.){
    printf("WARNING - Set GAMMA_ to 4./3.\n");
  }
  if (VI != 1.){
    printf("WARNING - Set VI to 1.\n");
  }

  // initialise grid
  for (int i = 0; i < ncell[MV]; ++i){      // loop through cells along r
    Cell *c = &Cinit[i];

    double r = c->G.x[x_];                  // ls:  radial coordinate
    // double th = c->G.x[y_];                 // rad: theta angular coordinate

    if (r <= R0){
      c->S.cons[DEN] = D_f/rhoNorm;
      c->S.cons[TAU] = tau_f/ENorm;
      c->S.cons[SS1] = 0.;
      c->S.cons[SS2] = 0.;
      c->S.cons[TR1] = 1.;
      c->S.cons2prim(r);
      // printf("rho = %e, p = %e\n",c->S.prim[RHO],c->S.prim[PPP]);
      // printf("rho = %e, p = %e\n",(c->S.prim[RHO])*rhoNorm,(c->S.prim[PPP])*pNorm);
      // double gam = (c->S.cons[TAU]+c->S.cons[DEN])/c->S.cons[DEN];
      // printf("Lorentz Factor = %e\n",gam);
    }
    else{
      c->S.prim[RHO] = rho0/rhoNorm;
      c->S.prim[PPP] = p0/pNorm;
      c->S.prim[VV1] = 0.;
      c->S.prim[VV2] = 0.;
      c->S.prim[TR1] = 2.;
      // printf("rho = %e, p = %e\n",c->S.prim[RHO],c->S.prim[PPP]);
    }
  }

  return 0;
}


void Grid::userKinematics(int it, double t){

  // setting lower and higher i boundary interface velocities

  // set by boundary velocity:
  double vIn;
  if (t < tMove){
    vIn = 0.;
  }
  else{
    vIn = 0.55;
  }

  // double vIn = 0.;
  double vOut = 1.05;

  for (int n = 0; n < ngst; ++n){
    int    iL = n;
    int    iR = ntrack-2-n;
    Itot[iL].v = vIn;
    Itot[iR].v = vOut;
  }

}

void Cell::userSourceTerms(double dt){

  UNUSED(dt);

}

void Grid::userBoundaries(int it, double t){

  // Making sure that we don't got to negative radii
  // States inside ghosts cells are already copied at this stage
  // just need to edit the geometry
  // double xL = Ctot[iLbnd+1].G.x[MV];
  // double dxL = Ctot[iLbnd+1].G.dx[MV];
  // double xbnd = xL - dxL;
  // double dxg = fmin(dxL, xL/4.);
  // for (int i = 0; i <= iLbnd; ++i){
  //   Ctot[i].G.dx[MV] = dxg;
  //   Ctot[i].G.x[MV]  = xbnd + dxg/2. - (iLbnd-i+1) * dxg;
  //   Ctot[i].computeAllGeom();
  //   int ind[] = {i};
  //   assignId(ind);
  // }
  // for (int i = 0; i <= iLbnd-1; ++i){
  //   Itot[i] = Itot[iLbnd+1];
  //   Itot[i].x[MV] -= (iLbnd-i+1) * dxg;
  //   Itot[i].computedA();
  // }

  // reflective inner boundary
  if (t < tMove) {
    for (int i = 0; i < ngst; ++i){
      Ctot[i].S.prim[UU1] -= 1;
    }
  }

  UNUSED(it);
  UNUSED(t);

}


int Grid::checkCellForRegrid(int j, int i){

  Cell c = Ctot[i];                      // get current cell (track j, cell i)
  // double trac = c.S.prim[TR1];           // get tracer value
  double r   = c.G.x[r_];                   // get cell radial coordinate
  double dr  = c.G.dx[r_];                  // get cell radial spacing
  double dth = th_simu;                  // get cell angular spacing
  
  Cell cOut = Ctot[iRbnd];            // get outermost cell in track j
  double Rout = cOut.G.x[r_];               // get radial coordinate of outermost cell

  // determine which radius to use for AR calculations
  double r_AR;
  if (Rout < R2){                           // if furthest cell in track is within R2
    r_AR = Rout;
  }
  else if (r < R2){                         // if furthest cell in track is greater than R2 but cell is within R2
    r_AR = R2;
  }
  else{                                     // if cell is beyond R2
    r_AR = r;
  }

  // double AR = dr / (r * dth);               // calculate aspect ratio (only true AR if r_AR = r)
  double AR = dr / (r_AR * dth);            // calculate aspect ratio (only true AR if r_AR = r)
  
  // calculate target aspect ratio
  // double target_AR  = fmax(pow(R1/r, 0.2), 1.)*fin_AR;
  double target_AR  = fmax(pow(R1/r_AR, 0.2), 1.)*fin_AR;

  if (AR > split_AR * target_AR) {          // if cell is too long for its width
      return(split_);                       // split
  }
  if (AR < merge_AR * target_AR) {          // if cell is too short for its width
      return(merge_);                       // merge
  }
  return(skip_);
}


void Cell::user_regridVal(double *res){
  // user function to find regrid victims
  // adapts the search to special target resolution requirements
  // depending on the tracer value
  
  UNUSED(*res);

}


void FluidState::cons2prim_user(double *rho, double *p, double *uu){

  UNUSED(rho);
  UNUSED(p);
  UNUSED(uu);
  
  return;

}

void Simu::dataDump(){

  if (it%500 == 0){ grid.printCols(it, t); }

  double t_C = 3.e3;
  double t_S = 9.e4;
  double t_g = 6.498e6;
  double t_M = 2.019e7;

  if ((t>=t_C) and ((t-dt)<t_C)){grid.printCols(it,t);}
  if ((t>=t_S) and ((t-dt)<t_S)){grid.printCols(it,t);}
  if ((t>=t_g) and ((t-dt)<t_g)){grid.printCols(it,t);}
  if ((t>=t_M) and ((t-dt)<t_M)){grid.printCols(it,t);}

}

void Simu::runInfo(){

  if ((worldrank == 0) and (it%100 == 0)){ printf("it: %ld time: %le\n", it, t);}
  // if ((worldrank == 0) and (it%1 == 0)){ printf("it: %ld time: %le\n", it, t);}

}

void Simu::evalEnd(){

  if (t > 5.e7){ stop = true; }
  if (it > 10000000){ stop = true; }

}












