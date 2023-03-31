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
static int GRID_TYPE_ = 0;                  // for testing. 0 = lin grid, 1 = log grid
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
static double mmax = 1;                     // maximum "derefinement" level
static double lmax = 5;                     // maximum "refinement level"

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

static void calcWind(double r_denorm, double t, double *rho, double *u, double *p){
  // returns normalised primitive variables for relativistic wind at position r (denormalised).
  double p_inj = Theta * rho_w * c_* c_;
  double gma = 5./3.;

  *rho = rho_w * pow(r_denorm/rmin0, -2) / rhoNorm;
  *u = c_*sqrt(1.-1./(lfacwind*lfacwind))*lfacwind / vNorm;
  *p = p_inj * pow(r_denorm/rmin0, -2*gma) / pNorm;

}

double calcCellSize(double r, int Nc, double rmin, double rmax){
  // returns cell size at given position, depending on grid geometry choice
  // for grid initialisation and cell size check at regrid step
  double dr;
  
  //dr = (rmax-rmin)/Nc;
  if (GRID_TYPE_ == 0){ // linear grid, if it works add 'GRID_TYPE_', 'LIN_' and 'LOG_' to def.h
    dr = (rmax-rmin)/Nc;
  }
  else if (GRID_TYPE_ == 1){ // log grid}
    double step = log10((rmax + abs(rmin)-rmin)/abs(rmin))/Nc;
    dr  = (r + abs(rmin) - rmin)*(pow(10, step) - 1);
  }
  return(dr);
}

int Grid::initialGeometry(){                              
  // Creates initial grid
  // linear grid in wind, logarithmic grid for SNR ejecta
  double rmin = rmin0/lNorm;
  double rmax = rmax0/lNorm;

  double r = rmin;
  for (int i = 0; i < ncell[x_]; ++i){
    Cell *c = &Cinit[i];

    double dr = calcCellSize(r, ncell[x_], rmin, rmax);
    r = r + dr;
    c->G.x[x_]    = r;
    c->G.dx[x_]   = dr;
    c->computeAllGeom();
  }
  return 0;
}

int Grid::initialValues(){
  // Initialises grid with physical values

  double p_ram = Theta * rho_w * c_* c_;

  for (int i = 0; i < ncell[MV]; ++i){              // loop through cells along r
    Cell *c = &Cinit[i];
    double r = c->G.x[x_];
    double r_denorm = r*lNorm;
    
    if (r_denorm <= R_b){                           // unshocked wind
      double rho, u, p;
      calcWind(r_denorm, t_start, &rho, &u, &p);
      p_ram = std::min(p, p_ram);

      c->S.prim[RHO] = rho;
      c->S.prim[VV1] = beta_w;
      c->S.prim[PPP] = p;
      c->S.prim[TR1] = 2.;
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
      c->S.prim[PPP] = p_ram;   // p already normalized
      c->S.prim[TR1] = 1.;
    }
    else{                                             // CSM
      double rho = rho_csm * pow(r_denorm/R_0, k);
      c->S.prim[RHO] = rho / rhoNorm;
      c->S.prim[VV1] = 0.;
      c->S.prim[PPP] = p_ram;
      c->S.prim[TR1] = 0.;
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
  
  for (int n = 0; n <= ngst+1; ++n){
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

  for (int i = 0; i <= iLbnd+1; ++i){
    Cell *c = &Ctot[i];
    double rho, u, p;
    double r = c->G.x[r_]*lNorm;

    calcWind(r, t, &rho, &u, &p);
    
    c->S.prim[RHO] = rho;
    c->S.prim[UU1] = u;
    c->S.prim[UU2] = 0;
    c->S.prim[PPP] = p;
    c->S.prim[TR1] = 2.;
    }

  UNUSED(it);
  UNUSED(t);

}

int Grid::checkCellForRegrid(int j, int i){
  // check cells to apply split / merge method to

  UNUSED(j);
  
  Cell c  = Ctot[i];
  double r   = c.G.x[r_];                   // get cell position
  double dr  = c.G.dx[r_];                  // get cell radial spacing
  double rmin = Ctot[iLbnd+1].G.x[x_];
  double rmax = Ctot[iRbnd-1].G.x[x_];
  double dr_i= calcCellSize(r, ncell[x_], rmin, rmax);  // get original grid size at same position
  double ar = dr/dr_i;
  double dr0 = calcCellSize(rmin, Ncells, rmin, rmax); // initial resolution at grid min radius


  if (i==iLbnd+1){
    if (dr>dr0){
      return(split_);
    }
  }
  if (ar > pow(2., mmax)){    // if cell too big compared to expected size
    return(split_);
  }
  if (ar < pow(2., -lmax)){   // if cell is more than 2^lmax smaller than original grid
    return(merge_);
  }

  /*
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
  */
  return(skip_);

}

void Cell::user_regridVal(double *res){
  // user function to find regrid victims for circular regridding
  // called in 1d.cpp/targetRegridVictims etc.
  // adapts the search to special target resolution requirements
  // depending on the tracer value
  
  // double r = G.x[r_];
  // double dr = G.dx[r_];
  // double lfac = S.lfac();
  // *res = dr / (r*dth) * pow(lfac, 3./2.);
  UNUSED(*res);

}

void FluidState::cons2prim_user(double *rho, double *p, double *uu){
  // user defined function for cons2prim, called in the hydro upodate function
  // implementation of Newman & Hanlin (NH) 2014 from athena code (Stone+ 2020)
  // In 1D for now
  
  UNUSED(uu);
  UNUSED(*p);
  /*
  // Parameters
  int max_iterations = 15;
  double tol = 1.0e-12;
  double pgas_uniform_min = 1.0e-12;
  double a_min = 1.0e-12;
  double v_sq_max = 1.0 - 1.0e-12;
  double rr_max = 1.0 - 1.0e-12;

  // Extract conserved values
  double D = cons[DEN];         // dd in athena 
  double tau = cons[TAU] + D;
  double E = D+tau;             // ee in athena
  double mm = cons[SS1];
  double mm_sq = mm*mm;
  double gma = gamma();

  // Calculate functions of conserved quantities
  double pgas_min = -E;
  pgas_min = std::max(pgas_min, pgas_uniform_min);

  // Iterate until convergence
  double pgas[3];
  double pgas_old;
  pgas_old = *p;
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
    return;
  }
  double prs = pgas[(n+1)%3];
  *p = prs;
  //if (!std::isfinite(p)) {
  //  return;
  //}
  double a = E + prs;                                 // (NH 5.7)
  a = std::max(a, a_min);
  double v_sq = mm_sq / (a*a);                        // (NH 5.2)
  v_sq = std::min(std::max(v_sq, 0.), v_sq_max);
  double lfac2 = 1.0/(1.0-v_sq);                      // (NH 3.1)
  double lfac = std::sqrt(lfac2);                     // (NH 3.1)
  double dty = D/lfac;
  *rho = dty;                                         // (NH 4.5)
  //if (!std::isfinite(rho)) {
  //  return;
  //}
  double v = mm / a;                                  // (NH 4.6)
  *uu = lfac*v;                                       // (NH 3.3)
  //if (!std::isfinite(uu)
  //    || !std::isfinite(prim(IVY,k,j,i))
  //    || !std::isfinite(prim(IVZ,k,j,i)) ) {
  //  return;
  double wgas = a/lfac2;
  //prs = (gma-1.0)/gma * (wgas - dty);
  //*p = prs;
  */
  return;
  }

void Simu::dataDump(){
  // if (it%5 == 0){ grid.printCols(it, t); }
  if (it%1000 == 0){ grid.printCols(it, t); }

}

void Simu::runInfo(){

  // if ((worldrank == 0) and (it%1 == 0)){ printf("it: %ld time: %le\n", it, t);}
  if ((worldrank == 0) and (it%100 == 0)){ printf("it: %ld time: %le\n", it, t);}

}

void Simu::evalEnd(){

  if (it > 10000){ stop = true; }
  // if (t > 1.02e3){stop = true; } // 3.33e8 BOXFIT simu

}
