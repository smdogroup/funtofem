#ifndef TRANSFERSCHEME_H
#define TRANSFERSCHEME_H

#include <complex>

#include "mpi.h"

/*
  Use the cplx type for F2FComplex
*/
typedef double F2FReal;
typedef std::complex<double> F2FComplex;

/*
  Define the basic scalar type F2FScalar
*/
#ifdef FUNTOFEM_USE_COMPLEX
#define F2F_MPI_TYPE MPI_DOUBLE_COMPLEX
typedef F2FComplex F2FScalar;
#else
#define F2F_MPI_TYPE MPI_DOUBLE
typedef F2FReal F2FScalar;
#endif

// Define the real part function for the complex data type
inline double F2FRealPart(const F2FComplex &c) { return real(c); }

// Define the imaginary part function for the complex data type
inline double F2FImagPart(const F2FComplex &c) { return imag(c); }

// Dummy function for real part
inline double F2FRealPart(const double &r) { return r; }

// Compute the absolute value
inline F2FReal F2Ffabs(const F2FReal &c) {
  if (c < 0.0) {
    return -c;
  }
  return c;
}

// Compute the absolute value
inline F2FComplex F2Ffabs(const F2FComplex &c) {
  if (real(c) < 0.0) {
    return -c;
  }
  return c;
}

class TransferScheme {
 public:
  // Constructor
  TransferScheme(int dof_per_node = 3) : dof_per_node(dof_per_node) {}

  // Destructor
  virtual ~TransferScheme();

  // Mesh loading
  virtual void setAeroNodes(const F2FScalar *aero_X, int aero_nnodes);
  virtual void setStructNodes(const F2FScalar *struct_X, int struct_nnodes);

  // Initialization
  virtual void initialize() = 0;

  // Get information from the transfer object
  int getStructDofPerNode() { return dof_per_node; }
  int getNumAeroNodes() { return na; }
  int getNumStructNodes() { return ns; }
  int getStructArrayLen() { return dof_per_node * ns; }
  int getAeroArrayLen() { return 3 * na; }

  // Load and displacement transfers
  virtual void transferDisps(const F2FScalar *struct_disps,
                             F2FScalar *aero_disps) = 0;
  virtual void transferLoads(const F2FScalar *aero_loads,
                             F2FScalar *struct_loads) = 0;

  // Action of transpose Jacobians needed for solving adjoint system
  virtual void applydDduS(const F2FScalar *vecs, F2FScalar *prods) = 0;
  virtual void applydDduSTrans(const F2FScalar *vecs, F2FScalar *prods) = 0;
  virtual void applydLduS(const F2FScalar *vecs, F2FScalar *prods) = 0;
  virtual void applydLduSTrans(const F2FScalar *vecs, F2FScalar *prods) = 0;

  // Action of Jacobians. These are valid when the transfer scheme is derived
  // using the method of virtual work. If not, they must be implemented
  // directly.
  virtual void applydLdfA(const F2FScalar *vecs, F2FScalar *prods) {
    applydDduSTrans(vecs, prods);
  }
  virtual void applydLdfATrans(const F2FScalar *vecs, F2FScalar *prods) {
    applydDduS(vecs, prods);
  }

  // Action of Jacobians needed for assembling gradient from adjoint
  // variables
  virtual void applydDdxA0(const F2FScalar *vecs, F2FScalar *prods) = 0;
  virtual void applydDdxS0(const F2FScalar *vecs, F2FScalar *prods) = 0;
  virtual void applydLdxA0(const F2FScalar *vecs, F2FScalar *prods) = 0;
  virtual void applydLdxS0(const F2FScalar *vecs, F2FScalar *prods) = 0;

  // Convert aero displacements into equivalent rigid + elastic deformation
  void transformEquivRigidMotion(const F2FScalar *aero_disps, F2FScalar *R,
                                 F2FScalar *t, F2FScalar *u);
  void applydRduATrans(const F2FScalar *vecs, F2FScalar *prods);
  void applydRdxA0Trans(const F2FScalar *aero_disps, const F2FScalar *vecs,
                        F2FScalar *prods);

  // Routines to test necessary functionality of transfer scheme
  int testAllDerivatives(const F2FScalar *struct_disps,
                         const F2FScalar *aero_loads, const F2FScalar h,
                         const double rtol, const double atol);
  int testLoadTransfer(const F2FScalar *struct_disps,
                       const F2FScalar *aero_loads, const F2FScalar *pert,
                       const F2FScalar h, const double rtol, const double atol);
  int testDispJacVecProducts(const F2FScalar *struct_disps,
                             const F2FScalar *test_vec_a,
                             const F2FScalar *test_vec_s, const F2FScalar h,
                             const double rtol, const double atol);
  int testLoadJacVecProducts(const F2FScalar *struct_disps,
                             const F2FScalar *aero_loads,
                             const F2FScalar *test_vec_s1,
                             const F2FScalar *test_vec_s2, const F2FScalar h,
                             const double rtol, const double atol);
  int testdDdxA0Products(const F2FScalar *struct_disps,
                         const F2FScalar *test_vec_a1,
                         const F2FScalar *test_vec_a2, const F2FScalar h,
                         const double rtol, const double atol);
  int testdDdxS0Products(const F2FScalar *struct_disps,
                         const F2FScalar *test_vec_a,
                         const F2FScalar *test_vec_s, const F2FScalar h,
                         const double rtol, const double atol);
  int testdLdxA0Products(const F2FScalar *struct_disps,
                         const F2FScalar *aero_loads,
                         const F2FScalar *test_vec_a,
                         const F2FScalar *test_vec_s, const F2FScalar h,
                         const double rtol, const double atol);
  int testdLdxS0Products(const F2FScalar *struct_disps,
                         const F2FScalar *aero_loads,
                         const F2FScalar *test_vec_s1,
                         const F2FScalar *test_vec_s2, const F2FScalar h,
                         const double rtol, const double atol);

 protected:
  // Transfer scheme object counter and ID
  static int object_count;
  int object_id;

  // Degrees of freedom per node for the structural solution and load vector.
  // Note that there are always 3 displacements for aerodynamic nodes and 3
  // components for the aerodynamic force vector. For the structures, there may
  // be displacements + rotations and forces + moments. If the structures uses
  // just the u, v, w components for displacements then dof_per_node = 3,
  // however, if the structures uses u, v, w, theta_x, theta_y, theta_z, then
  // dof_per_node = 6. Handling this case is up to the specific transfer
  // scheme implementation.
  int dof_per_node;

  // Communicators
  MPI_Comm global_comm;
  MPI_Comm struct_comm;
  MPI_Comm aero_comm;
  int struct_root;
  int aero_root;

  // Aerodynamic data
  F2FScalar *Xa;
  F2FScalar *Fa;
  int na;
  int na_global;

  // Structural data
  F2FScalar *Xs;
  F2FScalar *Us;
  int ns;

  // Rigid transformation data
  F2FScalar Raero[9];
  F2FScalar Saero[9];
  F2FScalar xa0bar[3];
  F2FScalar xabar[3];

  // Parallel movement of aerodynamic vectors
  void collectAerodynamicVector(const F2FScalar *local, F2FScalar *global);
  void distributeAerodynamicVector(F2FScalar *global, F2FScalar *local);

  // Auxiliary function for computing rotation from covariance matrix
  void computeRotation(const F2FScalar *H, F2FScalar *R, F2FScalar *S);

  // Auxiliary functions for load transfer (needed in complex compute rotation)
  void assembleM1(const F2FScalar *R, const F2FScalar *S, F2FScalar *A);
};

// Functions for vector and matrix math
void vec_add(const F2FScalar *x, const F2FScalar *y, F2FScalar *xy);
void vec_diff(const F2FScalar *x, const F2FScalar *y, F2FScalar *xy);
void vec_scal_mult(const F2FScalar a, const F2FScalar *x, F2FScalar *ax);
void vec_cross(const F2FScalar *x, const F2FScalar *y, F2FScalar *xy);
F2FScalar vec_mag(const F2FScalar *x);
F2FScalar vec_dot(const F2FScalar *x, const F2FScalar *y);
F2FScalar det(const F2FScalar *A);

#endif  // TRANSFERSCHEME_H
