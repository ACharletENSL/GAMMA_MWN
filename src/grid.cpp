/*
* @Author: eliotayache
* @Date:   2020-05-06 09:26:35
* @Last Modified by:   Eliot Ayache
* @Last Modified time: 2020-11-28 11:36:47
*/

#include "grid.h"

Grid::Grid()
{
}

Grid::~Grid()
{

}


void Grid::prepForRun(){

  // converting velocities to 4-velocities:
  for (int j = 0; j < nde_nax[F1]; ++j){
    for (int i = 0; i < ntrack[j]; ++i){
      double v   = 0;
      double vv[NUM_D];
      for (int d = 0; d < NUM_D; ++d){
        vv[d] = Ctot[j][i].S.prim[VV1+d];
        v += vv[d]*vv[d];
      }
      v = sqrt(v);
      double lfac = 1./sqrt(1.-v*v);
      for (int d = 0; d < NUM_D; ++d){
        Ctot[j][i].S.prim[UU1+d] *= lfac;
      }
    }
  }

  interfaceGeomFromCellPos();

}

void Grid::prepForUpdate(int it, double t){

  prim2cons();
  state2flux();
  computeNeighbors();
  movDir_ComputeLambda();   // has to be done before kinematics update (and flux calc)
  updateKinematics(it, t);
  computeFluxes();
  
}

