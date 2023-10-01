#ifndef TRANSFER_SCHEME_H
#define TRANSFER_SCHEME_H

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
  TransferScheme(MPI_Comm global_comm, MPI_Comm struct_comm, int struct_root,
                 MPI_Comm aero_comm, int aero_root, int struct_node_dof = 3,
                 int aero_node_dof = 3)
      : global_comm(global_comm),
        struct_comm(struct_comm),
        struct_root(struct_root),
        aero_comm(aero_comm),
        aero_root(aero_root),
        struct_node_dof(struct_node_dof),
        aero_node_dof(aero_node_dof) {
    na = 0;
    na_global = 0;
    ns = 0;
    ns_local = 0;
    mesh_update = 0;

    Xa = NULL;        // Local array of aerodynamic nodes
    Xs = NULL;        // Global array of structural nodes
    Xs_local = NULL;  // Local array of structural nodes

    object_id = object_count;
    object_count++;
  }

  // Destructor
  virtual ~TransferScheme();

  // Mesh loading
  void setAeroNodes(const F2FScalar *aero_X, int aero_nnodes);
  void setStructNodes(const F2FScalar *struct_X, int struct_nnodes);

  // Initialization
  virtual void initialize() = 0;

  // Get information from the transfer object about the lengths of the expected
  // arrays
  int getStructNodeDof() { return struct_node_dof; }
  int getAeroNodeDof() { return aero_node_dof; }
  int getNumLocalAeroNodes() { return na; }
  int getNumLocalStructNodes() { return ns_local; }
  int getLocalAeroArrayLen() { return aero_node_dof * na; }
  int getLocalStructArrayLen() { return struct_node_dof * ns_local; }

 protected:
  // Distribute the structural mesh if mesh_update is true on one of the
  // processors.
  void distributeStructuralMesh();

  // Add structural values from all processors and scatter them to the
  // structures
  void structAddScatter(int global_len, F2FScalar *global_data, int local_len,
                        F2FScalar *local_data);

  // Gather local structural values and broadcast them to all processors
  void structGatherBcast(int local_len, const F2FScalar *local_data,
                         int global_len, F2FScalar *global_data);

  // Takes aero values from all processors and scatter them to the
  // aero processors
  void aeroScatter(int global_len, F2FScalar *global_data, int local_len,
                   F2FScalar *local_data);

  // Gather local aero values and broadcast them to all processors
  void aeroGatherBcast(int local_len, const F2FScalar *local_data,
                       int global_len, F2FScalar *global_data);

  // Build an aerostructural connectivity through LocatePoint search, linking
  // each aerodynamic node with a specified number of nearest structural nodes
  void computeAeroStructConn(int isymm, int nn, int *conn, double tol = 1e-7);

  // Computes weights of structural nodes based on an exponential decay
  void computeWeights(double beta, int isymm, int nn, const int *conn,
                      F2FScalar *W, double tol = 1e-7);

  // Communicators
  MPI_Comm global_comm;  // Global communicator
  MPI_Comm struct_comm;  // Communicator for the structures
  int struct_root;       // Structural rank-0 proc on global_comm
  MPI_Comm aero_comm;    // Communicator for the aerodynamics
  int aero_root;         // Aerodynamic rank-0 proc on global_comm
  int struct_node_dof;   // Degrees of freedom per structural node
  int aero_node_dof;     // Degrees of freedom per aerodynamic node

  // Keep track if the mesh has been updated
  int mesh_update;

  // Aerodynamic data
  F2FScalar *Xa;  // Aerodynamics node locations (x, y, z) at each node
  int na;         // Number of local aerodynamic nodes
  int na_global;  // Number of global aerodynamic nodes (on all aero procs)

  // Structural data
  // Degrees of freedom per node for the structural solution and load vector.
  // Note that there are always 3 displacements for aerodynamic nodes and 3
  // components for the aerodynamic force vector. For the structures, there
  // may be displacements + rotations and forces + moments. If the structures
  // uses just the u, v, w components for displacements then dof_per_node = 3,
  // however, if the structures uses u, v, w, theta_x, theta_y, theta_z, then
  // dof_per_node = 6. Handling this case is up to the specific transfer
  // scheme implementation.
  F2FScalar *Xs;        // Global array of (x, y,z) locations for structures
  F2FScalar *Xs_local;  // Local array of (x, y, z) locations for structures
  int ns;        // Number of global structural nodes across all struct procs
  int ns_local;  // Number of local structural nodes on this processor

  // Transfer scheme object counter and ID
  static int object_count;
  int object_id;
};

class LDTransferScheme : public TransferScheme {
 public:
  LDTransferScheme(MPI_Comm global_comm, MPI_Comm struct_comm, int struct_root,
                   MPI_Comm aero_comm, int aero_root, int struct_node_dof = 3)
      : TransferScheme(global_comm, struct_comm, struct_root, aero_comm,
                       aero_root, struct_node_dof, 3) {
    Us = NULL;
    Fa = NULL;
  }
  virtual ~LDTransferScheme() {
    if (Us) {
      delete[] Us;
    }
    if (Fa) {
      delete[] Fa;
    }
  }

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
  // Aerodynamic load data
  F2FScalar *Fa;

  // Structural data
  F2FScalar *Us;

  // Rigid transformation data
  F2FScalar Raero[9];
  F2FScalar Saero[9];
  F2FScalar xa0bar[3];
  F2FScalar xabar[3];

  // Auxiliary function for computing rotation from covariance matrix
  void computeRotation(const F2FScalar *H, F2FScalar *R, F2FScalar *S);

  // Auxiliary functions for load transfer (needed in complex compute
  // rotation)
  void assembleM1(const F2FScalar *R, const F2FScalar *S, F2FScalar *A);
  F2FScalar printDetM1(const F2FScalar *A);
};

class ThermalTransfer : public TransferScheme {
 public:
  ThermalTransfer(MPI_Comm global_comm, MPI_Comm struct_comm, int struct_root,
                  MPI_Comm aero_comm, int aero_root)
      : TransferScheme(global_comm, struct_comm, struct_root, aero_comm,
                       aero_root, 1, 1) {
    Ha = NULL;
    Ts = NULL;
  }
  virtual ~ThermalTransfer() {
    if (Ha) {
      delete[] Ha;
    }
    if (Ts) {
      delete[] Ts;
    }
  }

  // Temperature and flux transfers
  virtual void transferTemp(const F2FScalar *struct_temp,
                            F2FScalar *aero_temp) = 0;
  virtual void transferFlux(const F2FScalar *aero_flux,
                            F2FScalar *struct_flux) = 0;

  // Action of transpose Jacobians needed for solving adjoint system
  virtual void applydTdtS(const F2FScalar *vecs, F2FScalar *prods) = 0;
  virtual void applydTdtSTrans(const F2FScalar *vecs, F2FScalar *prods) = 0;
  virtual void applydQdqA(const F2FScalar *vecs, F2FScalar *prods) = 0;
  virtual void applydQdqATrans(const F2FScalar *vecs, F2FScalar *prods) = 0;

  // Test Functions
  int testAllDerivatives(const F2FScalar *struct_temps,
                         const F2FScalar *aero_flux, const F2FScalar h,
                         const double rtol, const double atol);
  int testFluxTransfer(const F2FScalar *struct_temps,
                       const F2FScalar *aero_flux, const F2FScalar *pert,
                       const F2FScalar h, const double rtol, const double atol);
  int testTempJacVecProducts(const F2FScalar *struct_temps,
                             const F2FScalar *test_vec_a,
                             const F2FScalar *test_vec_s, const F2FScalar h,
                             const double rtol, const double atol);
  int testFluxJacVecProducts(const F2FScalar *struct_temps,
                             const F2FScalar *aero_flux,
                             const F2FScalar *test_vec_s1,
                             const F2FScalar *test_vec_s2, const F2FScalar h,
                             const double rtol, const double atol);

 protected:
  // Aerodynamic load data
  F2FScalar *Ha;

  // Structural data
  F2FScalar *Ts;
};

// Functions for vector and matrix math
void vec_add(const F2FScalar *x, const F2FScalar *y, F2FScalar *xy);
void vec_diff(const F2FScalar *x, const F2FScalar *y, F2FScalar *xy);
void vec_scal_mult(const F2FScalar a, const F2FScalar *x, F2FScalar *ax);
void vec_cross(const F2FScalar *x, const F2FScalar *y, F2FScalar *xy);
F2FScalar vec_mag(const F2FScalar *x);
F2FScalar vec_dot(const F2FScalar *x, const F2FScalar *y);
F2FScalar det(const F2FScalar *A);

#endif  // TRANSFER_SCHEME_H
