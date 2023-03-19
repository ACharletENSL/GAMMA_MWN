/*
  Setup for 1D spherical simulations of a magnetar wind nebulae.
  Values for parameters are set from python script in GAMMA main folder
  Author A. Charlet
*/

#include "../../environment.h"
#include "../../grid.h"
#include "../../constants.h"
#include "../../simu.h"

// SNR age at simulation starting time (s)
static double t_start = 1.e2;

// Grid parameters
static double dth     = M_PI/2./1000.;
static double Ncells  = 2000;               // Initial cell numbers in r direction
static double Nmax    = 5000;               // Max number of cells, must be > Ncells
static double rmin0   = 1.0000e+09 ;        // min r coordinate of grid (cm)
static double rmax0   = 1.0000e+11 ;        // max r coordinate of grid (cm)

// Physical radii
static double R_b = 5.1175e+10 ;            // nebula bubble radius (cm)
static double R_c = 2.1556e+11 ;            // ejecta core radius (cm)
static double R_e = 2.1556e+12 ;            // ejecta envelope radius (cm)
static double R_0 = 1.e18 ;                 // CSM scaling radius (for non-constant CSM)

// Flow variables
static double rho_w    = 8.6552e-10 ;       // wind density at grid inner radius (g cm^-3)
static double lfacwind = 1.e2       ;       // wind Lorentz factor
static double beta_w = sqrt(1. - pow(lfacwind, -2));
static double Theta    = 1.e-4 ;            // ejecta relativistic temperature (p/rho c^2), easier set this than wind
// change Theta def when including hot bubble/young SN ?
static double rho_ej   = 9.9566e-02 ;       // ejecta core density (g cm^-3)
static double rho_csm  = 1.6726e-24 ;       // CSM density (g cm^-3) 
static double delta = 0 ;                   // ejecta core density gradient
static double omega = 10 ;                  // ejecta envelope density gradient
static double k = 0 ;                       // CSM density gradient

// Refinement parameters
static double ar0 = (rmax0-rmin0) / (Ncells*rmax0*dth);        // initial cell aspect ratio
static double target_ar  = ar0;             // target aspect ratio
static double split_AR   = 1.8;             // set upper bound as ratio of target_AR
static double merge_AR   = 0.1;             // set lower bound as ratio of target_AR
static double split_chi  = 0.3;             // set upper bound for gradient-based refinement criterion (split cells at interface)
static double merge_chi  = 0.1;             // set lower bound for gradient-based refinement criterion (merge cells outside of interfaces)

// normalisation constants:
static double rhoNorm = rho_w;        // density normalisation
static double lNorm = c_;                     // distance normalised to c
static double vNorm = c_;                     // velocity normalised to c
static double pNorm = rhoNorm*vNorm*vNorm;    // pressure normalised to rhoNorm c^2

void loadParams(s_par *par){

  par->tini      = t_start;           // initial time
  par->ncell[x_] = Ncells;            // number of cells in r direction
  par->nmax      = Nmax;              // max number of cells in MV direction
  par->ngst      = 2;                 // number of ghost cells (?); probably don't change

  normalizeConstants(rhoNorm, vNorm, lNorm);

}

double calcGamma_wind(double r_denorm, double theta0){
  // using calcGamma and p = p0 (r/r0)**-2 gamma gives the correct 5/3 value
  // for some reason deriving Theta and then p = rho*theta gives 5/2

  if (EOS_ == IDEAL_EOS_) {
    return(GAMMA_);
  }
  else if (EOS_ == SYNGE_EOS_) {
    // self consistent gamma in wind has not been implemented yet
    // using gamma = 5./3. as first order approximation (cold wind)
    return(GAMMA_);
  }
  else if (EOS_ == RYU_EOS_) {
    double xi = r_denorm / rmin0;
    double theta = theta0 * (sqrt(3.* pow(xi, -4./3.) + 1.) - 1.);
    double a = 3.*theta + 1.;
    double gamma_eff = (4. * a + 1.)/(3. * a);
    return(gamma_eff);
  }
}

int Grid::initialGeometry(){                              
  // Creates initial grid
  // linear grid in wind, logarithmic grid for SNR ejecta
  double rmin = rmin0;
  double rmax = rmax0;
  rmin /= lNorm;
  rmax /= lNorm;

  double dr = (rmax-rmin)/ncell[x_];
  // double step = log10((rmax + abs(rmin)-rmin)/abs(rmin))/ncell[x_];
  // double dr = (r + abs(rmin) - rmin)*(pow(10, step) - 1);
  double r = rmin;
  for (int i = 0; i < ncell[x_]; ++i){
    Cell *c = &Cinit[i];
    double r      = rmin + (i+0.5)*dr;
    c->G.x[x_]    = r;
    c->G.dx[x_]   = dr;
    c->computeAllGeom();
  }
  return 0;
}

int Grid::initialValues(){
  // Initialises grid with physical values

  double gma = 4./3.;
  // set pressure at rmin0 such as ejecta is at P_FLOOR_ at minimum
  // double p_ej  = Theta*rho_ej*c_*c_;                // ejecta pressure, Theta in code units
  // double p_inj = p_ej * pow(R_b/rmin0, 2*gma);      // pressure at injection set such as p(R_b)=p_ej
  double p_inj = Theta * rho_w * c_* c_;
  double p_ram = p_inj;

  for (int i = 0; i < ncell[MV]; ++i){              // loop through cells along r
    Cell *c = &Cinit[i];
    double r = c->G.x[x_];
    double r_denorm = r*lNorm;
    
    if (r_denorm <= R_b){                           // unshocked wind
      double rho = rho_w * pow(r_denorm/rmin0, -2);
      double p = p_inj * pow(r_denorm/rmin0, -2*gma);
      p_ram = std::min(p, p_ram);

      c->S.prim[RHO] = rho / rhoNorm;
      c->S.prim[VV1] = beta_w;
      c->S.prim[PPP] = p / pNorm;
    }
    else if ((r_denorm > R_b) && (r_denorm <= R_e)){  // SNR ejecta
      if (r_denorm <= R_c){     // ejecta core
        double rho = rho_ej * pow(r_denorm/R_c, -delta);
        c->S.prim[RHO] = rho / rhoNorm;
      }
      else{                     // ejecta envelope
        double rho = rho_ej * pow(r_denorm/R_c, -omega);
        c->S.prim[RHO] = rho / rhoNorm;
      }
      double v = r_denorm / t_start;
      c->S.prim[VV1] = v / vNorm;
      c->S.prim[PPP] = p_ram / pNorm;
    }
    else{                                             // CSM
      double rho = rho_csm * pow(r_denorm/R_0, k);
      c->S.prim[RHO] = rho / rhoNorm;
      c->S.prim[VV1] = 0.;
      c->S.prim[PPP] = p_ram / pNorm;
    }
  }
  return 0;
}

void Grid::userKinematics(int it, double t){
  // Sets the interface velocities
  // Overrides Grid::updateKinematic in 1d/2d/3d.cpp
  // Defaut behavior moves interfaces at middle wave velocity
  // see e.g. Ling, Duan, Tang 2019 for description and proof that it is PCP

  UNUSED(it);
  UNUSED(t);
  
  for (int n = 0; n <= ngst; ++n){
    // fixes ghost cells of left boundary with zero velocity
    int    iL = n;
    Itot[iL].v = 0;
  }
}

void Cell::userSourceTerms(double dt){
  // For sources inside the grid

  UNUSED(dt);

}

void Grid::userBoundaries(int it, double t){
  // Overrides Grid::updateGhosts in 1d/2d/3d.cpp

  double gma = 4./3.;
  double rmin = rmin0;
  double rmax = rmax0;
  rmin /= lNorm;
  rmax /= lNorm;
  double u = beta_w * lfacwind;
  // double step = log10((rmax + abs(rmin)-rmin)/abs(rmin))/ncell[x_];
  double dr = (rmax-rmin)/ncell[x_];

  for (int i = 0; i <+ iLbnd+1; ++i){
    Cell *c = &Ctot[i];
    double r = rmin - (iLbnd-i+0.5)*dr;
    // double r_denorm = c->G.x[x_]*lNorm;
    // double r = rmin - (iLbnd-i+1)*dr;
    double r_denorm = r*lNorm;
    
    double rho = rho_w * pow(r_denorm/rmin0, -2);
    //double p = std::max(Theta*rho_w*c_*c_/pNorm, P_FLOOR_*pow(R_b/rmin0, 2*gma))
    double p = Theta * rho_w * pow(r_denorm/rmin0, -2*gma) * c_ *c_ ;
    /*if (i == iLbnd){
      cout << "r = " << r_denorm << "\n";
      cout << "rho = " << rho << "\n";
      cout << "p = " << p << "\n";
    }*/

    // c->G.x[x_]     = r;
    // c->computeAllGeom();
    c->S.prim[RHO] = rho / rhoNorm;
    c->S.prim[VV1] = u;
    c->S.prim[VV2] = 0;
    c->S.prim[PPP] = p / pNorm;
    c->S.prim[TR1] = 1.;

    // double dr_n = c->G.dx[x_];

    // cout << "calc dr = " << dr*c_ << ", cell init dr = " << dr_i*c_ << ", new dr = " << dr_n*c_ << "\n";
  }

  UNUSED(it);
  UNUSED(t);

}

int Grid::checkCellForRegrid(int j, int i){
  // check cells to apply split / merge method to

  UNUSED(j);
  
  Cell c  = Ctot[i];
  double dr  = c.G.dx[r_];                  // get cell radial spacing
  double rOut = Ctot[iRbnd-1].G.x[x_];
  double ar  = dr / (rOut*dth);             // calculate cell aspect ratio
  double g = 1.;

  // note: split_AR = 3, merge_AR = 0.1
  
  if (ar > split_AR * target_ar / g) { // if cell is too long for its width
    return(split_);                       // split
  }
  if (ar < merge_AR * target_ar / g) { // if cell is too short for its width
    return(merge_);                       // merge
  }

  return(skip_);

}

void Cell::user_regridVal(double *res){
  // user function to find regrid victims
  // adapts the search to special target resolution requirements
  // depending on the tracer value
  
  // double r = G.x[r_];
  // double dr = G.dx[r_];
  // double lfac = S.lfac();
  // *res = dr / (r*dth) * pow(lfac, 3./2.);
  UNUSED(*res);

}

void FluidState::cons2prim_user(double *rho, double *p, double *uu){

  UNUSED(uu);
  UNUSED(*p);

  return;

}


void Simu::dataDump(){
  // if (it%5 == 0){ grid.printCols(it, t); }
  if (it%10 == 0){ grid.printCols(it, t); }

}

void Simu::runInfo(){

  // if ((worldrank == 0) and (it%1 == 0)){ printf("it: %ld time: %le\n", it, t);}
  if ((worldrank == 0) and (it%100 == 0)){ printf("it: %ld time: %le\n", it, t);}

}

void Simu::evalEnd(){

  if (it > 100){ stop = true; }
  // if (t > 1.02e3){stop = true; } // 3.33e8 BOXFIT simu

}
