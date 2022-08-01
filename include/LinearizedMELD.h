#ifndef LINEARIZEDMELD_H
#define LINEARIZEDMELD_H

#include "MELD.h"

/*
  Linearized MELD is a transfer scheme developed from the MELD transfer scheme
  assuming displacements tend to zero.
*/
class LinearizedMELD : public MELD {
 public:
  // Constructor
  LinearizedMELD(MPI_Comm all, MPI_Comm structure, int struct_root,
                 MPI_Comm aero, int aero_root, int num_nearest, F2FScalar beta);

  // Destructor
  ~LinearizedMELD();

  // Initialization
  void initialize();

  // Load and displacement transfers
  void transferDisps(const F2FScalar *struct_disps, F2FScalar *aero_disps);
  void transferLoads(const F2FScalar *aero_loads, F2FScalar *struct_loads);

  // Action of transpose Jacobians needed for solving adjoint system
  void applydDduS(const F2FScalar *vecs, F2FScalar *prods);
  void applydDduSTrans(const F2FScalar *vecs, F2FScalar *prods);
  void applydLduS(const F2FScalar *vecs, F2FScalar *prods);
  void applydLduSTrans(const F2FScalar *vecs, F2FScalar *prods);

  // Action of Jacobians needed for assembling gradient from adjoint variables
  void applydDdxA0(const F2FScalar *vecs, F2FScalar *prods);
  void applydDdxS0(const F2FScalar *vecs, F2FScalar *prods);
  void applydLdxA0(const F2FScalar *vecs, F2FScalar *prods);
  void applydLdxS0(const F2FScalar *vecs, F2FScalar *prods);

 private:
  // Data for the transfers
  F2FScalar *global_H;

  // Auxiliary functions for linearized load and displacement transfer
  void computePointInertiaInverse(const F2FScalar *H, F2FScalar *Hinv);
  void computeDispContribution(const F2FScalar w, const F2FScalar *r,
                               const F2FScalar *Hinv, const F2FScalar *q,
                               const F2FScalar *us, F2FScalar *ua);
  void computeLoadContribution(const F2FScalar w, const F2FScalar *q,
                               const F2FScalar *Hinv, const F2FScalar *r,
                               const F2FScalar *fa, F2FScalar *fj);
};
#endif  // LINEARIZEDMELD_H
