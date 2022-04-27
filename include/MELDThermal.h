#ifndef MELDTHERMAL_H
#define MELDTHERMAL_H

#include "mpi.h"
#include "TransferScheme.h"

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

class F2F_API MELDThermal : public TransferScheme {
 public:
  // Constructor
  MELDThermal( MPI_Comm all,
	       MPI_Comm structure, int _struct_root,
	       MPI_Comm aero, int _aero_root,
	       int _isymm, int num_nearest,
	       F2FScalar beta );

  // Destructor
  ~MELDThermal();

  // Initialization
  virtual void initialize();

  // Set the aerodynamic and structural node locations
  void setStructNodes(const F2FScalar *struct_X, int struct_nnodes);
  void setAeroNodes(const F2FScalar *aero_X, int aero_nnodes);

  // Temperature and flux transfers
  void transferTemp(const F2FScalar *struct_temp,
                     F2FScalar *aero_temp);
  void transferFlux(const F2FScalar *aero_flux,
                    F2FScalar *struct_flux);

  // Action of transpose Jacobians needed for solving adjoint system
  void applydTdtS(const F2FScalar *vecs, F2FScalar *prods);
  void applydTdtSTrans(const F2FScalar *vecs, F2FScalar *prods);
  void applydQdqA(const F2FScalar *vecs, F2FScalar *prods);
  void applydQdqATrans(const F2FScalar *vecs, F2FScalar *prods);

  // Test Functions
  void testFluxTransfer(const F2FScalar *struct_temps,
                        const F2FScalar *aero_flux,
                        const F2FScalar *pert,
                        const F2FScalar h);
  void testTempJacVecProducts(const F2FScalar *struct_temps,
                              const F2FScalar *test_vec_a,
                              const F2FScalar *test_vec_s,
                              const F2FScalar h);
  void testFluxJacVecProducts(const F2FScalar *struct_temps,
                              const F2FScalar *aero_flux,
                              const F2FScalar *test_vec_s1,
                              const F2FScalar *test_vec_s2,
                              const F2FScalar h);

  // Inherited Displacement and Load Transfer functions (NOT implemented for Thermal MELD scheme)
  void transferDisps(const F2FScalar*, F2FScalar*){}
  void transferLoads(const F2FScalar*, F2FScalar*){}
  void applydDduS(const F2FScalar*, F2FScalar*){}
  void applydDduSTrans(const F2FScalar*, F2FScalar*){}
  void applydLduS(const F2FScalar*, F2FScalar*){}
  void applydLduSTrans(const F2FScalar*, F2FScalar*){}
  void applydDdxA0(const F2FScalar*, F2FScalar*){}
  void applydDdxS0(const F2FScalar*, F2FScalar*){}
  void applydLdxA0(const F2FScalar*, F2FScalar*){}
  void applydLdxS0(const F2FScalar*, F2FScalar*){}

 protected:
  // Local structural data
  int ns_local;
  F2FScalar* Xs_local;

  // Symmetry specifier
  int isymm;

  // Mesh update indicator
  int mesh_update;

  // Data for the connectivity and weighting
  int nn; // number of nearest nodes
  F2FScalar global_beta; // weighting decay parameter
  int *global_conn; // Connectivity for each thermal node

  // Data for thermal transfer
  F2FScalar *global_W;
  F2FScalar *global_xs0bar;
  F2FScalar *global_R;
  F2FScalar *global_S;

  // Parallel movement of structural vectors
  void distributeStructuralMesh();
  void collectStructuralVector(const F2FScalar *local, F2FScalar *global, int vars_per_node=3);
  void distributeStructuralVector(F2FScalar *global, F2FScalar *local, int vars_per_node=3);

  // Auxiliary functions for creating connectivity and weighting
  void setAeroStructConn(int *aerostruct_conn);
  void computeWeights(F2FScalar *W);
};

#endif //MELDTHERMAL_H
