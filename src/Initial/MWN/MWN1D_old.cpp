/*
  Setup for 1D spherical simulations of a magnetar wind nebulae.
  Author A. Charlet
*/

#include "../../environment.h"
#include "../../grid.h"
#include "../../constants.h"
#include "../../simu.h"

bool AMR = false;

double dth     = M_PI/2./1000.;
double epsilon = 1.e-2;               // Noise parameter for AMR
double Ncells  = 2000;                // Initial cell numbers in r direction
double Nmax    = 5000;                // Max number of cells, must be > Ncells
double lmax    = 5;                   // max refinement level

// set CBM parameters
static double n0      = 1.;           // cm-3:    CBM number density
static double rho0    = n0*mp_;       // g.cm-3:  comoving CBM mass density

// simulation starting time and size
static double tinit = 1.e3;           // s: starting time (physics)
// static double rmin0 = 1.e11;        // cm: grid rmin
// static double rmax0 = 1.e12;        // cm: grid rmax
// BE CAREFUL ABOUT STARTING TIME, Rcore grows slower than Rwind

// SNR parameters
static double Esn = 1.e51;          // erg: SNR energy
static double Mej = 3. * 1.9891e33; // g: ejecta mass
static double n = 10.;              // external shell density profile
static double m = 0.;               // internal shell density profile
static double wc = 0.8;             // 0<wc<1, limit between internal and external shells

// wind parameters
static double L0 = 5.e45;       // erg.s-1: wind luminosity
static double lfacwind = 100;   // wind lfac
static double theta = 1.e-3;    // theta = p/(rho*c^2) at wind injection

// intermediate quantities
static double f3 = (n-3.)*(3.-m)/(n-m-(3.-m)*pow(wc,n-3.));               // numerical function
static double f5 = (n-5.)*(5.-m)/(n-m-(5.-m)*pow(wc,n-5.));               // numerical function
static double vt = sqrt(2.*f5*Esn/(f3*Mej));                              // ejecta velocity at at limit detemrined by wc
static double Rcore = vt * tinit;                                        // transition radius between internal and external shells
static double Rshell = Rcore/wc;                                           // end of the external shell
static double Rw = 1.50 * pow(n/(n-5.),.2) * sqrt((n-5.)/(n-3.)) * pow(Esn,.3) * pow(L0,.2) * pow(tinit,1.2) / sqrt(Mej);
// wind FS position according to Blondin+01
// std::cout << "Rw = " << Rw << ", Rcore = " << Rcore << ", Rshell = " << Rshell << "\n";

// grid size depending on shocked bubble position
static double rmin0 = 1.05*Rw;                   // min r coordinate of grid
static double rmax0 = 2*Rw;                   // max r coordinate of (initial) grid
static double dr0   = (rmax0-rmin0)/Ncells;     // inital cell size (for linear grid)
static double ar0   = dr0 / (rmax0*dth);        // inital cell aspect ratio

// intermediate quantities (cont.)
static double rhoej = f3*Mej/(4.*PI*pow(Rcore,3));                        // g.cm-3: prefactor for ejecta density profile 
static double rhow = L0/(4.*PI*pow(rmin0,2)*pow(lfacwind,2)*pow(c_,3));   // g.cm-3: wind density
static double betaw = sqrt(1.-pow(lfacwind,-2));      
static double uw = betaw * lfacwind;                                      // wind velocity in units c. Close to unity
static double p_inj = rhow * c_ * c_ * theta;                   // Ba:    pressure at wind injection
static double pfloor = std::min(rhow * uw * uw, rhoej) * c_ * c_ * theta;             // floor for pressure

// determine shock position then implement wind profile for next iteration

// normalisation constants:
static double rhoNorm = rho0;                 // density normalised to CBM density
// static double rhoNorm = rhow;               // density normalised to wind density at rmin
static double lNorm = c_;                     // distance normalised to c
static double vNorm = c_;                     // velocity normalised to c
static double pNorm = rhoNorm*vNorm*vNorm;    // pressure normalised to rho_CMB/c^2

// ARM parameters
static double target_ar  = ar0;              // target aspect ratio
static double split_AR   = 1.8;              // set upper bound as ratio of target_AR
static double merge_AR   = pow(2., -lmax);   // set lower bound as ratio of target_AR
static double split_chi  = 0.3;              // set upper bound for gradient-based refinement criterion (split cells at interface)
static double merge_chi  = 0.1;              // set lower bound for gradient-based refinement criterion (merge cells outside of interfaces)

// Numerical viscosity
static double eta_vnr = 4.;                   // numerical von Neumann-Richtmyer viscosity (see vN&R 1950)

void loadParams(s_par *par){

  par->tini      = tinit;             // initial time
  par->ncell[x_] = Ncells;            // number of cells in r direction
  par->nmax      = Nmax;              // max number of cells in MV direction
  par->ngst      = 2;                 // number of ghost cells (?); probably don't change

  normalizeConstants(rhoNorm, vNorm, lNorm);

}

double H_func(float sum){
  if (sum>0.){return 1.;}
  else {return 0.;}
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

double calc_LoehnerError(double sigL, double sig, double sigR){
  // derives 2nd order error criterion for refinement, see Löhner 1987 or Mignone 2012 
  double dSigL  = sig - sigL;
  double dSigR  = sigR - sig;
  double sigRef = abs(sigR) + 2.*abs(sig) + abs(sigL);
  double dSig_diff = abs(dSigR - dSigL);
  double dSig_sum  = abs(dSigR) + abs(dSigL) + epsilon*sigRef;
  double chi = sqrt((dSig_diff*dSig_diff) / (dSig_sum*dSig_sum));
  return(chi);
}

int Grid::initialGeometry(){                              

  // logarithmic grid for SNR ejecta
  double rmin = rmin0;
  double rmax = rmax0;
  rmin /= lNorm;
  rmax /= lNorm;

  double dr = (rmax-rmin)/ncell[x_];
  double step = log10((rmax + abs(rmin)-rmin)/abs(rmin))/ncell[x_];
  double r = rmin;
  for (int i = 0; i < ncell[x_]; ++i){
    Cell *c = &Cinit[i];
    double r      = rmin + (i+0.5)*dr;

    if (r*lNorm > Rw){
      double dr = (r + abs(rmin) - rmin)*(pow(10, step) - 1);
      r = r + dr;
    }
    c->G.x[x_]    = r;
    c->G.dx[x_]   = dr;
    c->computeAllGeom();
  }
  return 0;
}

int Grid::initialValues(){
  // add EoS here
  // Careful, geometrical quantities are already normalised when using this function.
    if ((EOS_ == IDEAL_EOS_) && (GAMMA_ != 4./3.)){// no warning if synge or taub
        printf("WARNING - Set GAMMA_ to 4./3.\n");
    }
    else if (EOS_ == SYNGE_EOS_){
        printf("Self-consistent wind not implemented for Synge-like EoS, using 5./3.");
    }
    if (VI != 1.){
        printf("WARNING - Set VI to 1.\n");
    }
    if (Rw > Rcore){
        printf("WARNING - Rwind > Rcore, set lower tinit");
    }
    double gv = uw * c_;                      // Lorentz factor * v
    double p_ram = p_inj;
    cout << "With t_start = " << tinit << " s, wind - ejecta interface at r = " << Rw << "cm, ejecta core ends at r = " << Rcore << " cm and shell at r = " << Rshell <<" cm.\n";
    cout << "Grid set between " << rmin0 << "cm and " << rmax0 << "cm.\n";
    // initialise grid
    for (int i = 0; i < ncell[MV]; ++i){      // loop through cells along r
        Cell *c = &Cinit[i];
        double r = c->G.x[x_];                  // ls:  radial coordinate
        //double dr = c->G.dx[x_];              // ls:  radial coordinate
        double r_denorm = r*lNorm;

        if (r_denorm <= Rw ){ // wind bubble
          double rho = rhow * pow(r_denorm/rmin0, -2);
          double gma = calcGamma_wind(r_denorm, theta);
          double p = p_inj * pow(r_denorm/rmin0, -2*gma);
          p_ram = std::min(p_ram, p);

          c->S.prim[RHO] = rho / rhoNorm;
          c->S.prim[VV1] = betaw;
          // c->S.prim[PPP] = std::max(p, pfloor) / pNorm;
          c->S.prim[PPP] = p / pNorm;
        }
        else if ( (r_denorm > Rw) && (r_denorm <= Rshell) ){ // ejecta
            double vsnr = r_denorm/tinit;
            if (r_denorm <= Rcore){ // ejecta core
              double rho = rhoej*pow(r_denorm/Rcore, -m);
              c->S.prim[RHO] = rho / rhoNorm;
            }
            else{ // ejecta tail
              double rho = rhoej*pow(r_denorm/Rcore, -n);
              c->S.prim[RHO] = rho / rhoNorm;
            }
            // c->S.prim[PPP] = std::max(p_ram, pfloor)  / pNorm;
            c->S.prim[PPP] = p_ram / pNorm;
            c->S.prim[VV1] = vsnr/vNorm;
            c->S.prim[TR1] = 1.;
        }
        else{ // CSM
            c->S.prim[RHO] = rho0 / rhoNorm;
            c->S.prim[VV1] = 0.;
            c->S.prim[PPP] = std::max(theta*rho0*c_*c_, p_ram)/pNorm;
            c->S.prim[TR1] = 1.;
        }

    }
    return 0;

}

void Grid::userKinematics(int it, double t){
  // Overrides the interface velocities
  // Defaut behavior moves interfaces at middle wave velocity
  // see e.g. Ling, Duan, Tang 2019 for description and proof that it is PCP

  UNUSED(it);
  UNUSED(t);
  /*
  for (int n = 0; n < ngst; ++n){
    // fixes ghost cells of left boundary with zero velocity
    int    iL = n;
    Itot[iL].v = 0;
  }*/
}

void Cell::userSourceTerms(double dt){

  UNUSED(dt);

}

void Grid::userBoundaries(int it, double t){
  /*double rmin = rmin0;
  double rmax = rmax0;
  rmin /= lNorm;
  rmax /= lNorm;
  // double step = log10((rmax + abs(rmin)-rmin)/abs(rmin))/ncell[x_];
  double dr = (rmax-rmin)/ncell[x_];
 
  for (int i = 0; i <+ iLbnd+1; ++i){
    Cell *c = &Ctot[i];
    double r = rmin - (iLbnd-i+1)*dr;
    double r_denorm = r*lNorm;
    // double dr = (r + abs(rmin) - rmin)*(pow(10, step) - 1);
    double rho = rhow * pow(r_denorm/rmin0, -2);
    double gma = calcGamma_wind(r_denorm, theta);
    double p = p_inj * pow(r_denorm/rmin0, -2*gma);

    c->G.x[x_]     = r;
    c->computeAllGeom();
    c->S.prim[RHO] = rho/rhoNorm;
    c->S.prim[VV1] = uw;
    c->S.prim[VV2] = 0;
    c->S.prim[PPP] = p / pNorm;
    c->S.prim[TR1] = 1.;

    // double dr_n = c->G.dx[x_];

    // cout << "calc dr = " << dr*c_ << ", cell init dr = " << dr_i*c_ << ", new dr = " << dr_n*c_ << "\n";
  }*/

  UNUSED(it);
  UNUSED(t);

}

int Grid::checkCellForRegrid(int j, int i){
  // refinement criterion based on second derivative error norm
  // as implemented in PLUTO (Mignone 2012) following Löhner 1987
  /*double sigma(Cell *c){// function of conservative variables to be used as refinement criterion
    double D = c->S.cons[DEN];
    // double S = c->S.cons[SS1];
    // double tau = c->S.cons[TAU];
    return(D);
  }*/

  UNUSED(j);
  
  Cell c  = Ctot[i];
  // double trac = c.S.prim[TR1];           // get tracer value
  // double r   = c.G.x[r_];                // get cell radial coordinate
  
  double dr  = c.G.dx[r_];                  // get cell radial spacing
  double rOut = Ctot[iRbnd-1].G.x[x_];
  double ar  = dr / (rOut*dth);             // calculate cell aspect ratio
  double g = 1.;
  // note: split_AR = 3, merge_AR = 0.1, target_ar = 1
  
  if (ar > split_AR * target_ar / g) { // if cell is too long for its width
    return(split_);                       // split
  }

  #if AMR
    Cell cL = Ctot[i-1];
    Cell cR = Ctot[i+1];
    double sig  = c.S.cons[DEN];
    double sigL = cL.S.cons[DEN];
    double sigR = cR.S.cons[DEN];

    if (ar > merge_AR * target_ar){ // only check gradient criterion if cell big enough
      double chi = calc_LoehnerError(sigL, sig, sigR); 
      if (chi > split_chi){ // if gradient too strong
        return(split_);
      }
    }

    if (dr < dr0/2.){ // check merging criterion if cells were refined once at least
      double chi = calc_LoehnerError(sigL, sig, sigR);
      if (chi < merge_chi){ // merge cells when refinement is not needed
        return(merge_);
      }
    }
  #endif

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
  if (it%100 == 0){ grid.printCols(it, t); }

}

void Simu::runInfo(){

  // if ((worldrank == 0) and (it%1 == 0)){ printf("it: %ld time: %le\n", it, t);}
  if ((worldrank == 0) and (it%100 == 0)){ printf("it: %ld time: %le\n", it, t);}

}

void Simu::evalEnd(){

  if (it > 5000){ stop = true; }
  // if (t > 1.02e3){stop = true; } // 3.33e8 BOXFIT simu

}