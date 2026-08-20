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
static int GEOMETRY_  = 1 ;           // 0 for cartesian, 1 for spherical

// set CBM parameters
static double n0      = 1.;           // cm-3:    CBM number density
static double rho0    = n0*mp_;       // g.cm-3:  comoving CBM mass density
static double Theta0  = 5e-05 ;   //          Theta0 = p/(rho*c^2)
static double p0      = Theta0*rho0*c_*c_;

// set shells parameters
static double rho1 = 4.667291079080063e-15 ;     // comoving density of front shell
static double u1   = 100.000000 ;          // proper velocity (gamma*beta) of front shell
static double p1   = 52.17289029534429 ;
static double D01  = 29977746950.122807 ;     // spatial extension of front shell
static double rho4 = 1.1610033862318821e-15 ;     // comoving density of back shell
static double u4   = 200.000000 ;          // proper velocity of back shell
static double p4   = 52.17289029534429 ;
static double D04  = 29978871066.45374 ;     // spatial extension of back shell
static double beta1= u1/sqrt(1+u1*u1);
static double beta4= u4/sqrt(1+u4*u4);
static double cont = 0.05 ;           // density contrast between shell and ext medium
// Ambient pressure treatment (phys_input 'pmatch'; 0 = historical, keeps every existing
// run reproducible):
//   0 -> p_ext = cont*p_shell. Ambient keeps the shells' Theta but is under-pressured by
//        1/cont, so the shell edge blasts into it at t=0. Harmless at cont=5e-2 (a factor
//        20), crippling as cont shrinks: at 5e-4 the inner buffer cell was compressed 100x
//        in dx and 40x in rho, and that thin dense cell throttled the CFL timestep to
//        ~0.037 s/it -- 12 million iterations to reach t=5e5.
//   1 -> p_ext = p_shell. Ambient in pressure equilibrium, no t=0 blast, timestep stays
//        healthy at any cont. Isolates ambient DENSITY from the initial pressure jump,
//        which is what an ambient-dependence test actually wants to vary. Note the
//        ambient is then relativistically hot (Theta_ext = Theta0/cont), by construction:
//        matching p while cutting rho has no other outcome.
// The cartesian branch has always used the unscaled p (see below), i.e. it was already
// pmatch=1 while the spherical branch was pmatch=0.
static int PMATCH_ = 0 ;

// box size
static double R_0     = 799471536841663.8 ;
static int Nsh1   = 500 ;
static int Ntot1  = 520 ;
static int Nsh4   = 500 ;
static int Ntot4  = 520 ;
static int Ncells = Ntot4 + Ntot1;

// --- stopping criterion (set by setup.py from the phys_input 'stop' keyword) ---
// STOP_: 0 = it          -> stop at iteration ITMAX_
//        1 = shock       -> stop EXTRA_TIME after both shocks finish crossing
//        2 = rarefaction -> stop EXTRA_TIME after both rarefaction waves have
//                           swept the shocked layer (the pressure plateau vanishes)
//        3 = tstop       -> stop at lab time TSTOP_ (ITMAX_ still a backstop)
static int    STOP_      = 0 ;
static long   ITMAX_     = 850000 ;
// mode 3 target, in lab seconds (phys_input 'tstop'; scaled by alpha in rescale_input).
// Iteration count is NOT a physical clock and two setups do not share it: the timestep
// is CFL-limited by the steepest feature in the box, so a run whose shell edges are
// outflow boundaries (Next = 0, no rarefaction) advances much further in t per
// iteration than the fiducial -- measured 1.56x by it=50k and 2.51x by it=100k, still
// widening. Stopping both at the same t is what makes paired runs comparable; stopping
// both at the same ITMAX_ does not.
static double TSTOP_     = 0.0 ;
// post-condition buffer, 5% of the theoretical shock-crossing time
static double EXTRA_TIME = 2156.4307108738403 ;
// rarefaction-convergence detection, self-normalised to the current shell p_max:
// a 'plateau' cell has p > PLATEAU_PFRAC * p_max; both rarefactions have swept the
// whole layer once the plateau fraction drops below PLATEAU_NFRAC (its minimum, when
// the two rarefaction heads meet). Calibrated on a spherical cooling_fid run: the
// fraction falls from ~0.6 at crossing to a ~0.085 minimum at t~1.33 t_cross, then a
// smooth decaying peak keeps a ~0.06-0.08 floor; 0.10 fires at the convergence
// minimum, cleanly above that floor. Tunable; the backstop (ITMAX_) guards runaways.
static double PLATEAU_PFRAC = 0.95 ;
static double PLATEAU_NFRAC = 0.10 ;
// ... and the same counter on the way UP: during crossing regions 2 & 3 share one
// pressure across the CD, so the plateau fraction IS the swept fraction of the shell.
// It crossing PLATEAU_EFRAC marks 'the shock is established over enough cells' and ends
// the dense early dump phase. Guarded by p_max > 2*p_max(t=0) so the uniform initial
// shells (where every cell of the higher-p shell trivially passes PLATEAU_PFRAC) cannot
// fire it before any shock exists -- p_shocked/p_4 ~ 2e3 here, so the guard is loose.
static double PLATEAU_EFRAC = 0.20 ;

// --- dump cadence (set by setup.py from the phys_input itdump* keywords) ---
// Early data carries the onset ladder that sets the early lightcurve, while the late
// run is logarithmic in t (dt ~ t), so a uniform cadence both under-samples the
// crossing and floods the late phase. Three phases, keyed on the run phase below:
//   before 'established'  -> ITDUMP_EARLY_   (dense, resolves individual cell shockings)
//   until  'converged'    -> ITDUMP_MID_     (the historical itdump)
//   after                 -> ITDUMP_LATE_    (sparse; still ~log-uniform since dt ~ t)
// Setting all three equal reproduces the former single-cadence behaviour exactly.
static long   ITDUMP_EARLY_ = 5 ;
static long   ITDUMP_MID_   = 50 ;
static long   ITDUMP_LATE_  = 500 ;

// normalisation constants:
static double rhoNorm = rho4 ;                // density normalised t
static double lNorm = c_;                     // distance normalised to c
static double vNorm = c_;                     // velocity normalised to c
static double pNorm = rhoNorm*vNorm*vNorm;    // pressure normalised to rho/c^2


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
        //double rho = cont*rho4*pow(r/R4, -2.);
        //double p   = p4*pow(r/R4, -2*gma);
        c->S.prim[RHO] = cont*rho4/rhoNorm;
        c->S.prim[VV1] = beta4;
        c->S.prim[PPP] = (PMATCH_ ? p4 : cont*p4)/pNorm;
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
        //double rho = cont*rho1*pow(r/R1, -2.);
        //double p   = p1*pow(r/R1, -2*gma);
        c->S.prim[RHO] = cont*rho1/rhoNorm;
        c->S.prim[VV1] = beta1;
        c->S.prim[PPP] = (PMATCH_ ? p1 : cont*p1)/pNorm;
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

  // Edge splits target the external-medium buffer. With Next = 0 (Ntot4 == Nsh4) the
  // grid boundaries sit on the shell edges, and a split there would shift every cell
  // index to its right -- the per-cell histories are keyed on that index, so it would
  // scramble them. Skip regridding entirely in that case.
  if ((Ntot4 > Nsh4) and ((i <= iin + 2) or (i >= iout - 10))){
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

  /*
  // Added a density floor to avoid numerical errors
  double rho_floor = cont * rho4 / rhoNorm * 1.e-6;
  double p_floor   = Theta0 * rho_floor * c_ * c_ / pNorm;

  if (*rho < rho_floor) *rho = rho_floor;
  if (*p   < p_floor)   *p   = p_floor;
  */
  return;
}

// --- run phase, shared by dataDump (cadence) and evalEnd (stopping) ---------------
// Both key off the same three-stage picture of the run, so the detection is done ONCE
// per iteration here rather than inside evalEnd. It has to live outside evalEnd: mode
// STOP_ == 0 returns before any detection runs, and the dump cadence still needs the
// phases in that mode.
static bool   established = false;   // shock has swept PLATEAU_EFRAC of the shell
static bool   crossed     = false;   // both shocks have finished crossing
static double t_cross     = -1.;
static bool   converged   = false;   // both rarefactions have swept the shocked layer
static double t_conv      = -1.;

static void updateRunPhase(Grid &grid, long it, double t){

  #if SHOCK_DETECTION_ == ENABLED_
    static int noShockCount = 0;
    static const int NO_SHOCK_REQUIRED = 100;
    // unshocked shell pressure of the initial condition. Taken from the setup constants
    // rather than sampled at runtime: the first call lands at it=1, by which point the
    // interface cells are already shocked, so a sampled reference would be the SHOCKED
    // pressure and the guard below could never be passed.
    const double p_unshocked = std::max(p1, p4)/pNorm;

    // --- shock-crossing completion: no shocked shell cell for NO_SHOCK_REQUIRED its,
    // and, in the same sweep, the current max shell pressure
    bool anyShockedTracer = false;
    double pmax = 0.;
    for (int i = grid.iLbnd+1; i <= grid.iRbnd-1; ++i){
      Cell *c = &grid.Ctot[i];
      if (c->S.prim[TR1] > 0.){
        if (c->isShocked){ anyShockedTracer = true; }
        if (c->S.prim[PPP] > pmax){ pmax = c->S.prim[PPP]; }
      }
    }

    if (anyShockedTracer){
      noShockCount = 0;
    } else if (it > 100){
      noShockCount++;
    }

    if (noShockCount >= NO_SHOCK_REQUIRED && !crossed){
      crossed = true;
      t_cross = t;
      if (worldrank == 0){
        printf("Shocks crossed at it=%ld, t=%le.\n", it, t);
      }
    }

    // --- plateau fraction: rises to ~0.6 while the shock sweeps the shell, then is
    // eaten away from both sides by the rarefactions. Read on the way up (established)
    // and on the way down (converged); pointless once converged.
    if (!converged && pmax > 2.*p_unshocked){
      int nplat = 0, nshell = 0;
      for (int i = grid.iLbnd+1; i <= grid.iRbnd-1; ++i){
        Cell *c = &grid.Ctot[i];
        if (c->S.prim[TR1] > 0.){
          nshell++;
          if (c->S.prim[PPP] > PLATEAU_PFRAC * pmax){ nplat++; }
        }
      }
      if (nshell > 0){
        if (!established && nplat >= PLATEAU_EFRAC * nshell){
          established = true;
          if (worldrank == 0){
            printf("Shock established at it=%ld, t=%le (plateau %d/%d). Dump cadence -> %ld\n",
                   it, t, nplat, nshell, ITDUMP_MID_);
          }
        }
        if (crossed && nplat < PLATEAU_NFRAC * nshell){
          converged = true;
          t_conv = t;
          if (worldrank == 0){
            printf("Rarefactions converged at it=%ld, t=%le (plateau %d/%d). Dump cadence -> %ld\n",
                   it, t, nplat, nshell, ITDUMP_LATE_);
          }
        }
      }
    }
  #else
    UNUSED(grid); UNUSED(it); UNUSED(t);
    established = true;      // no detection available: single cadence, ITDUMP_MID_
  #endif

}

void Simu::dataDump(){
  updateRunPhase(grid, it, t);
  long dump = (!established) ? ITDUMP_EARLY_ : (converged ? ITDUMP_LATE_ : ITDUMP_MID_);
  if (it % dump == 0){ grid.printCols(it, t); }

}

void Simu::runInfo(){

  if ((worldrank == 0) and (it%100 == 0)){ printf("it: %ld time: %le\n", it, t);}

}

void Simu::evalEnd(){

  // The phase flags (crossed/converged and their times) are maintained by
  // updateRunPhase, called once per iteration from dataDump -- which Simu::run invokes
  // just before this. evalEnd only decides when to stop.

  // mode 0: fixed iteration cap
  if (STOP_ == 0){
    if (it > ITMAX_){ stop = true; }
    return;
  }

  // mode 3: fixed lab time. Needs no shock detection, so it sits outside the block
  // below, next to mode 0. ITMAX_ stays a backstop: if TSTOP_ is set beyond what the
  // run can reach, this must not spin forever.
  if (STOP_ == 3){
    if (t >= TSTOP_){
      if (worldrank == 0){ printf("Stopping (tstop): it=%ld, t=%le >= %le\n", it, t, TSTOP_); }
      stop = true;
      return;
    }
    if (it > ITMAX_){
      if (worldrank == 0){
        printf("Stopping (tstop backstop): it=%ld > ITMAX_=%ld at t=%le, before TSTOP_=%le\n",
               it, ITMAX_, t, TSTOP_);
      }
      stop = true;
    }
    return;
  }

  #if SHOCK_DETECTION_ == ENABLED_
    // mode 1: shock -- stop EXTRA_TIME after crossing completes
    if (STOP_ == 1){
      if (crossed && (t - t_cross >= EXTRA_TIME)){
        if (worldrank == 0){ printf("Stopping (shock): it=%ld, t=%le\n", it, t); }
        stop = true;
      }
      return;
    }

    // mode 2: rarefaction -- after crossing, the shocked layer is one high-pressure
    // plateau (regions 2 & 3, equal p across the CD) bounded by the two rarefaction
    // heads. Both heads have swept the layer once the near-maximal flat top has been
    // eaten away from both sides (updateRunPhase). Detection is self-normalised to the
    // current shell p_max, so the overall spherical pressure decline does not trigger
    // it, and the meeting point need not sit at the CD (handles asymmetric shells).
    if (STOP_ == 2){
      // safety backstop: never run past ITMAX_ even if convergence is not detected
      if (it > ITMAX_){
        if (worldrank == 0){
          printf("Stopping (rarefaction backstop): it=%ld > ITMAX_=%ld before convergence\n", it, ITMAX_);
        }
        stop = true;
        return;
      }
      if (converged && (t - t_conv >= EXTRA_TIME)){
        if (worldrank == 0){ printf("Stopping (rarefaction): it=%ld, t=%le\n", it, t); }
        stop = true;
      }
      return;
    }
  #endif

}
