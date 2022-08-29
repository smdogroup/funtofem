#ifndef RBF_H
#define RBF_H

#include "TransferScheme.h"
#include "mpi.h"

/*
  Interpolation of loads and displacements using radial basis functions (RBFs)

  The basic algorithm and notation (names of variables) were taken from
  "Unified fluid–structure interpolation and mesh motion using radial basis
  functions" by T. C. S. Rendall and C. B. Allen.
*/
class RBF : public LDTransferScheme {
 public:
  // RBF type
  enum RbfType {
    GAUSSIAN,
    MULTIQUADRIC,
    INVERSE_MULTIQUADRIC,
    THIN_PLATE_SPLINE
  };

  // Constructor
  RBF(MPI_Comm global_comm, MPI_Comm struct_comm, int struct_root,
      MPI_Comm aero_comm, int aero_root, RbfType rbf_type, int sampling_ratio);

  // Destructor
  ~RBF();

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
  // Interpolation matrix
  F2FScalar *interp_mat;

  // Function to build interpolation matrix
  void buildInterpolationMatrix();

  // Pointer to radial basis function
  F2FScalar (*phi)(F2FScalar *x, F2FScalar *y);

  // Sampling data
  int denominator;  // one point sampled for every denominator points
  int nsub;         // number of structural points sampled
  int *sample_ids;  // IDs of the sampled points

  // Functions defining types of radial basis functions
  static F2FScalar gaussian(F2FScalar *x, F2FScalar *y);
  static F2FScalar multiquadric(F2FScalar *x, F2FScalar *y);
  static F2FScalar invMultiquadric(F2FScalar *x, F2FScalar *y);
  static F2FScalar thinPlateSpline(F2FScalar *x, F2FScalar *y);

  // Function to write out point clouds for Tecplot visualization
  void writeCloudsToTecplot();
};

#endif  // RBF_H
