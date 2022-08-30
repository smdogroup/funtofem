#ifndef MELD_THERMAL_H
#define MELD_THERMAL_H

#include "TransferScheme.h"
#include "mpi.h"

/*
  MELD (Matching-based Extrapolation of Loads and Displacments) is scalable
  scheme for transferring loads and displacements between large non-matching
  aerodynamic and structural meshes. It connects each aerodynamic node to a
  specified number of nearest structural nodes, and extrapolates its motion
  from the connected structural nodes through the solution of a shape-matching
  problem. The aerodynamic loads are extrapolated to the structural mesh in a
  consistent and conservative manner, derived from the principle of virtual
  work.

  Users must specify symmetry in the constructor
  isymm = -1 for no symmetry
        =  0 for symmetry across x = 0
        =  1 for symmetry across y = 0
        =  2 for symmetry across z = 0

  Users must also specify number of nearest nodes in initialize(num_nearest)
*/

class MELDThermal : public ThermalTransfer {
 public:
  // Constructor
  MELDThermal(MPI_Comm global_comm, MPI_Comm struct_comm, int struct_root,
              MPI_Comm aero_comm, int aero_root, int isymm, int num_nearest,
              F2FScalar beta);

  // Destructor
  ~MELDThermal();

  // Initialization
  virtual void initialize();

  // Set the aerodynamic and structural node locations
  void setStructNodes(const F2FScalar *struct_X, int struct_nnodes);
  void setAeroNodes(const F2FScalar *aero_X, int aero_nnodes);

  // Temperature and flux transfers
  void transferTemp(const F2FScalar *struct_temp, F2FScalar *aero_temp);
  void transferFlux(const F2FScalar *aero_flux, F2FScalar *struct_flux);

  // Action of transpose Jacobians needed for solving adjoint system
  void applydTdtS(const F2FScalar *vecs, F2FScalar *prods);
  void applydTdtSTrans(const F2FScalar *vecs, F2FScalar *prods);
  void applydQdqA(const F2FScalar *vecs, F2FScalar *prods);
  void applydQdqATrans(const F2FScalar *vecs, F2FScalar *prods);

 protected:
  // Symmetry specifier
  int isymm;

  // Data for aerostructural connectivity and weighting
  int nn;                 // number of nearest nodes
  F2FScalar global_beta;  // weighting decay parameter
  int *global_conn;       // connectivity

  F2FScalar *global_W;  // The global weights
};

#endif  // MELD_THERMAL_H
