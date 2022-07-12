#ifndef MELD_H
#define MELD_H

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
class F2F_API MELD : public TransferScheme {
 public:
  // Constructor
  MELD(MPI_Comm all,
       MPI_Comm structure, int _struct_root,
       MPI_Comm aero, int _aero_root,
       int _isymm, int num_nearest,
       F2FScalar beta);

  // Destructor
  ~MELD();

  // Initialization
  virtual void initialize();

  void setStructNodes(const F2FScalar *struct_X, int struct_nnodes);

  // Load and displacement transfers
  virtual void transferDisps(const F2FScalar *struct_disps,
                             F2FScalar *aero_disps);
  virtual void transferLoads(const F2FScalar *aero_loads,
                             F2FScalar *struct_loads);

  // Action of transpose Jacobians needed for solving adjoint system
  virtual void applydDduS(const F2FScalar *vecs, F2FScalar *prods);
  virtual void applydDduSTrans(const F2FScalar *vecs, F2FScalar *prods);
  virtual void applydLduS(const F2FScalar *vecs, F2FScalar *prods);
  virtual void applydLduSTrans(const F2FScalar *vecs, F2FScalar *prods);

  // Action of Jacobians needed for assembling gradient from adjoint variables
  virtual void applydDdxA0(const F2FScalar *vecs, F2FScalar *prods);
  virtual void applydDdxS0(const F2FScalar *vecs, F2FScalar *prods);
  virtual void applydLdxA0(const F2FScalar *vecs, F2FScalar *prods);
  virtual void applydLdxS0(const F2FScalar *vecs, F2FScalar *prods);

 protected:
  // Local structural data
  int ns_local;
  F2FScalar* Xs_local;

  // Symmetry specifier
  int isymm;

  // Mesh update indicator
  int mesh_update;

  // Data for aerostructural connectivity and weighting
  int nn; // number of nearest nodes
  F2FScalar global_beta; // weighting decay parameter
  int *global_conn;

  // Data for load and displacement transfers
  F2FScalar *global_W;
  F2FScalar *global_xs0bar;
  F2FScalar *global_R;
  F2FScalar *global_S;

  // Data for Jacobian-vector products
  F2FScalar *global_M1;
  int *global_ipiv;

  // Parallel movement of structural vectors
  void distributeStructuralMesh();
  void collectStructuralVector(const F2FScalar *local, F2FScalar *global);
  void distributeStructuralVector(F2FScalar *global, F2FScalar *local);

  // Auxiliary functions for creating connectivity and weighting
  void setAeroStructConn(int *aerostruct_conn);
  void computeWeights(F2FScalar *W);

  // Auxiliary functions for displacement transfer
  void computeCentroid(const int *local_conn, const F2FScalar *W, 
                       const F2FScalar *X, 
                       F2FScalar *xsbar );
  void computeCovariance(const F2FScalar *X, const F2FScalar *Xd, 
                         const int *local_conn, const F2FScalar *W, 
                         const F2FScalar *xs0bar, const F2FScalar *xsbar,
                         F2FScalar *H);

  // Auxiliary functions for Jacobian-vector products
  void assembleM3( const F2FScalar *R, const F2FScalar *S,
                   F2FScalar *A );
};

#endif //MELD_H
