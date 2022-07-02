#ifndef FUNTOFEM_BEAM_TRANSFER_H
#define FUNTOFEM_BEAM_TRANSFER_H

#include "TransferScheme.h"


class BeamTransfer : public TransferScheme {
 public:
  BeamTransfer( MPI_Comm comm,
                MPI_Comm comm_struct, int struct_root,
                MPI_Comm comm_aero, int aero_root,
                const int *_conn, int _nelems, int _order,
                int _dof_per_node );
  ~BeamTransfer();

  // Initialization
  void setStructNodes(const F2FScalar *struct_X, int struct_nnodes);

  void initialize();

  // Load and displacement transfers
  void transferDisps( const F2FScalar *struct_disps,
                      F2FScalar *aero_disps );
  void transferLoads( const F2FScalar *aero_loads,
                      F2FScalar *struct_loads );

  // Action of transpose Jacobians needed for solving adjoint system
  void applydDduS( const F2FScalar *vecs, F2FScalar *prods );
  void applydDduSTrans( const F2FScalar *vecs, F2FScalar *prods );
  void applydLduS( const F2FScalar *vecs, F2FScalar *prods );
  void applydLduSTrans( const F2FScalar *vecs, F2FScalar *prods );

  // Action of Jacobians needed for assembling gradient from adjoint variables
  void applydDdxA0( const F2FScalar *vecs, F2FScalar *prods );
  void applydDdxS0( const F2FScalar *vecs, F2FScalar *prods );
  void applydLdxA0( const F2FScalar *vecs, F2FScalar *prods );
  void applydLdxS0( const F2FScalar *vecs, F2FScalar *prods );

 private:
  F2FScalar findParametricPoint( const F2FScalar X1[],
                                 const F2FScalar X2[],
                                 const F2FScalar Xa[],
                                 double *xi );
  void computeRotation( const F2FScalar *q,
                        const F2FScalar *d,
                        F2FScalar r[] );
  void computeRotationTranspose( const F2FScalar *q,
                                 const F2FScalar *d,
                                 F2FScalar r[] );
  void computeRotationDerivProduct( const F2FScalar *v,
                                    const F2FScalar *q,
                                    const F2FScalar *d,
                                    F2FScalar r[] );
  void addTransposeRotationDeriv( const double s,
                                  const F2FScalar *q,
                                  const F2FScalar *d,
                                  const F2FScalar *fa,
                                  F2FScalar fs[] );
  void addTransposeRotationDerivAdjoint( const double scale,
                                         const F2FScalar *q,
                                         const F2FScalar *fa,
                                         const F2FScalar *fs,
                                         F2FScalar psi[] );

  // Store the element connectivity
  int *conn;

  // The number of elements in the beam
  int nelems;

  // The order of the beam = 2 or 3
  int order;

  // Store the number of degrees of freedom per node
  int dof_per_node;

  // Additional data required by the transfer routines
  int *aero_pt_to_elem;
  double *aero_pt_to_param;
};

#endif // FUNTOFEM_BEAM_TRANSFER_H
