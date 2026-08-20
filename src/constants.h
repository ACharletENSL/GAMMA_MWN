#ifndef CONSTANTS_H_
#define CONSTANTS_H_ 

#include "environment.h"

// Pressure flooring, applied as p = fmax(p, P_FLOOR_) in Hydro/rel_{sph,cart}.cpp.
// This is the binding constraint on how far a shell run can be followed, NOT itmax:
// the shocked p_inj is only 0.03-0.30 in code units, so 1.e-10 left barely 9 decades of
// headroom and clamped every cell of cooling_g100 while rho/rho_inj was still ~5e-6
// (an adiabatic gamma drop of only x53-81). Past the clamp the adiabat is broken:
// a_rho = -dln rho/dln R flattens from ~2.4 to ~1, c_s stops falling so dt stops
// growing ~ t, and p is overestimated by up to 45x (which inflates syn ~ e' ~ p in the
// post-processed cooling). Verified directly: cooling_g100_probe (1.e-14) reproduces
// cooling_g100_pf1e-10 bit-for-bit while p is free and diverges exactly where the old
// run clamps -- over R/R_inj = 110-160 in cell 519, a_rho is 2.69 unclamped vs -0.31
// (density RISING) clamped.
// 1.e-14 is not enough for a run that reaches rho/rho_inj = 1e-6: the CD-adjacent cells
// fall fastest (a_rho ~ 3.0 vs ~2.25 outer) and land at p ~ 3e-15. 1.e-16 leaves them
// 30x of margin, and the outer cells >1e4x. NB p/(rho c^2) ~ 1e-6 at Gamma ~ 130 there,
// which is the cold, high-Lorentz-factor corner where relativistic cons2prim is hardest
// -- watch for NaN if pushing further.
#define P_FLOOR_ 1.e-16         // pressure flooring

#define PI      M_PI            // Pi
#define mp_     1.6726219e-24   // Proton mass (g)
#define me_     9.1093835e-28   // electron mass (g)
#define qe_     4.80320425e-10  // electron charge (statC=cm3/2.g1/2.s-1) (rho01/2.l02.c)
#define c_      2.99792458e10   // speed of light (cm.s-1)
#define sigmaT_ 6.65345871e-25  // Thomson Cross section (cm2)
#define h_      6.62607004e-27  // Planck constant (cm2.g.s-1)

#define Msun_   1.98855e33      // Solar mass (g)

#define min_p_      1.e-10      // minimum value allowed for p
#define min_rho_    1.e-10      // minimum value allowed for rho

// Radiation related:
#define alpha_      1.29251816e-09  // sigmaT_/6 pi_me_c_ (Van Eerten+2010)

// emissivity parameters: (values from Rahaman et al. 2023)
#define p_          2.5         // slope of electron population
#define eps_e_      0.3333333333333333    // constribution to electron acceleration
#define eps_B_      0.105413    // contribution to magnetic field
#define zeta_       0.01        // fraction of accelerated electrons
#define acc_eff_    1.          // acceleration efficiency
#define theta_      M_PI/2.     // electron pitch angle 

// Normalized constants:
extern double Nmp_, Nme_, Nqe_, NsigmaT_, Nalpha_;

void normalizeConstants(double rhoNorm, double vNorm, double lNorm);

#endif
