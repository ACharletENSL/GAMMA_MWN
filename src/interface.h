#ifndef FLUID_CELL_INTERFACE_H_
#define FLUID_CELL_INTERFACE_H_

//////////////////////////////////////////////////////////////////////////////////////////
#include "environment.h"
#include "fluid.h"

class Cell;     // forward declaration

class Interface
{
public:
  Interface(int d=MV);
  ~Interface();

  // MEMBERS
  int  status; 
  int  memNumber;
  int  dim;               // orientation (orthogonal vector direction)

  double x[NUM_D], x0[NUM_D];        // position (in a single direction)
  double v, v0;           // velocity (only in MV dimension) (lab frame)
  double lfac;            // Lorentz factor (only in MV dimension) (lab frame)
  double dx[NUM_D-1];     // spatial extent
                          // for 3D order is either (y,z) - (x,z) - (x,y)
                          // for 3D order is either (t,p) - (r,p) - (r,t)
  double dA;              // surface area

  FluidState S,SL,SR;     
  double lL,lS,lR;        // wavespeeds
  double flux[NUM_Q];

  void wavespeedEstimates();
  void computeLambda();
  void computeFlux();
  FluidState starState(FluidState Sin, double lbda);

  void move(double dt);
  void computedA();

  #if SHOCK_DETECTION_ == ENABLED_
    void measureShock(Cell *cL, Cell *cR);

    // Thread-safe variant, used by the 1D path. Interfaces i-1 and i share cell i, so
    // the two-argument form above cannot be called from a parallel loop over interfaces
    // without racing on that cell. This one computes the same two contributions and
    // parks them on the interface; Grid::computeFluxes folds them into the cells in a
    // separate loop, with the same reduction and in the same order.
    void measureShock();
    double Sd_fwd, pspec_fwd;   // forward shock  -> the LEFT  cell (i)
    double Sd_rev, pspec_rev;   // reverse shock  -> the RIGHT cell (i+1)
  #endif

};

#endif
