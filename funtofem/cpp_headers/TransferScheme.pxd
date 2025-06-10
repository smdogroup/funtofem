# For the use of MPI
from mpi4py.libmpi cimport *
cimport mpi4py.MPI as MPI

# Typdefs required for either real or complex mode
include "FuntofemTypedefs.pxi"

cdef extern from "TransferScheme.h":
  cppclass LDTransferScheme:
    # Mesh loading
    void setAeroNodes(const F2FScalar *aero_X, int aero_nnodes)
    void setStructNodes(const F2FScalar *struct_X, int struct_nnodes)

    # Initialization
    void initialize()

    # Load and displacement transfers
    void transferDisps(const F2FScalar *struct_disps,
                       F2FScalar *aero_disps)
    void transferLoads(const F2FScalar *aero_loads,
                       F2FScalar *struct_loads)

    # Get the dimensions
    int getStructNodeDof()
    int getAeroNodeDof()
    int getNumLocalAeroNodes()
    int getNumLocalStructNodes()
    int getLocalAeroArrayLen()
    int getLocalStructArrayLen()

    # Action of transpose Jacobians needed for solving adjoint system
    void applydDduS(const F2FScalar *vecs, F2FScalar *prods)
    void applydDduSTrans(const F2FScalar *vecs, F2FScalar *prods)
    void applydLduS(const F2FScalar *vecs, F2FScalar *prods)
    void applydLduSTrans(const F2FScalar *vecs, F2FScalar *prods)
    void applydLdfA(const F2FScalar *vecs, F2FScalar *prods)
    void applydLdfATrans(const F2FScalar *vecs, F2FScalar *prods)

    # Action of Jacobians needed for assembling gradient from adjoint variables
    void applydDdxA0(const F2FScalar *vecs, F2FScalar *prods)
    void applydDdxS0(const F2FScalar *vecs, F2FScalar *prods)
    void applydLdxA0(const F2FScalar *vecs, F2FScalar *prods)
    void applydLdxS0(const F2FScalar *vecs, F2FScalar *prods)

    # Convert aero displacements into equivalent rigid + elastic deformation
    void transformEquivRigidMotion(const F2FScalar *aero_disps,
                                   F2FScalar *R, F2FScalar *t, F2FScalar *u)
    void applydRduATrans(const F2FScalar *vecs, F2FScalar *prods)
    void applydRdxA0Trans(const F2FScalar *aero_disps, const F2FScalar *vecs,
                          F2FScalar *prods)

    # Routines to test necessary functionality of transfer scheme
    int testAllDerivatives(const F2FScalar *struct_disps,
                           const F2FScalar *aero_loads,
                           const F2FScalar h, const double rtol,
                           const double atol)
    int testLoadTransfer(const F2FScalar *struct_disps,
                         const F2FScalar *aero_loads,
                         const F2FScalar *pert,
                         const F2FScalar h, const double rtol,
                         const double atol)
    int testDispJacVecProducts(const F2FScalar *struct_disps,
                               const F2FScalar *test_vec_a,
                               const F2FScalar *test_vec_s,
                               const F2FScalar h, const double rtol,
                               const double atol)
    int testLoadJacVecProducts(const F2FScalar *struct_disps,
                               const F2FScalar *aero_loads,
                               const F2FScalar *test_vec_s1,
                               const F2FScalar *test_vec_s2,
                               const F2FScalar h, const double rtol,
                               const double atol)
    int testdDdxA0Products(const F2FScalar *struct_disps,
                           const F2FScalar *test_vec_a1,
                           const F2FScalar *test_vec_a2,
                           const F2FScalar h, const double rtol,
                           const double atol)
    int testdDdxS0Products(const F2FScalar *struct_disps,
                           const F2FScalar *test_vec_a,
                           const F2FScalar *test_vec_s,
                           const F2FScalar h, const double rtol,
                           const double atol)
    int testdLdxA0Products(const F2FScalar *struct_disps,
                           const F2FScalar *aero_loads,
                           const F2FScalar *test_vec_a,
                           const F2FScalar *test_vec_s,
                           const F2FScalar h, const double rtol,
                           const double atol)
    int testdLdxS0Products(const F2FScalar *struct_disps,
                           const F2FScalar *aero_loads,
                           const F2FScalar *test_vec_s1,
                           const F2FScalar *test_vec_s2,
                           const F2FScalar h, const double rtol,
                           const double atol)

  cppclass ThermalTransfer:
    # Mesh loading
    void setAeroNodes(const F2FScalar *aero_X, int aero_nnodes)
    void setStructNodes(const F2FScalar *struct_X, int struct_nnodes)

    # Initialization
    void initialize()

    # Transfer temperatures and heat fluxes
    void transferTemp(const F2FScalar *struct_temp,
                      F2FScalar *aero_temp)
    void transferFlux(const F2FScalar *aero_flux,
                      F2FScalar *struct_flux)

    # Get the dimensions
    int getStructNodeDof()
    int getAeroNodeDof()
    int getNumLocalAeroNodes()
    int getNumLocalStructNodes()
    int getLocalAeroArrayLen()
    int getLocalStructArrayLen()

    # Action of transpose Jacobians needed for solving adjoint system
    void applydTdtS(const F2FScalar *vecs, F2FScalar *prods)
    void applydTdtSTrans(const F2FScalar *vecs, F2FScalar *prods)
    void applydQdqA(const F2FScalar *vecs, F2FScalar *prods)
    void applydQdqATrans(const F2FScalar *vecs, F2FScalar *prods)

    # Routines to test necessary functionality of transfer scheme
    int testAllDerivatives(const F2FScalar *struct_temps,
                           const F2FScalar *aero_flux, const F2FScalar h,
                           const double rtol, const double atol)

cdef extern from "MELD.h":
  cppclass MELD(LDTransferScheme):
    # Constructor
    MELD(MPI_Comm all,
         MPI_Comm structure, int struct_root,
         MPI_Comm aero, int aero_root,
         int symmetry, int num_nearest, F2FScalar beta)

cdef extern from "MELDThermal.h":
  cppclass MELDThermal(ThermalTransfer):
    # Constructor
    MELDThermal(MPI_Comm all,
                MPI_Comm structure, int struct_root,
                MPI_Comm aero, int aero_root,
                int symmetry, int num_nearest, F2FScalar beta)

cdef extern from "LinearizedMELD.h":
  cppclass LinearizedMELD(MELD):
    # Constructor
    LinearizedMELD(MPI_Comm all,
                   MPI_Comm structure, int struct_root,
                   MPI_Comm aero, int aero_root,
                   int symmetry, int num_nearest, F2FScalar beta)

cdef extern from "RBF.h":
  enum RbfType "RBF::RbfType":
    GAUSSIAN "RBF::GAUSSIAN"
    MULTIQUADRIC "RBF::MULTIQUADRIC"
    INVERSE_MULTIQUADRIC "RBF::INVERSE_MULTIQUADRIC"
    THIN_PLATE_SPLINE "RBF::THIN_PLATE_SPLINE"

  cppclass RBF(LDTransferScheme):
    # Constructor
    RBF(MPI_Comm all,
        MPI_Comm structure, int struct_root,
        MPI_Comm aero, int aero_root,
        RbfType rbf_type, int sampling_ratio)

cdef extern from "BeamTransfer.h":
  cppclass BeamTransfer(LDTransferScheme):
    # Constructor
    BeamTransfer(MPI_Comm all,
                 MPI_Comm structure, int struct_root,
                 MPI_Comm aero, int aero_root,
                 const int *_conn, int _nelems, int _order,
                 int _dof_per_node)