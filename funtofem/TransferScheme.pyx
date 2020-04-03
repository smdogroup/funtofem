#!/bin/python
#distuils: language = c++
# For the use of MPI
from mpi4py.libmpi cimport *
cimport mpi4py.MPI as MPI

cdef extern from "mpi-compat.h":
       pass

# Import the declarations required from the pxd file
from TransferScheme cimport *

# Import numpy
import numpy as np
cimport numpy as np

# For the use of the numpy C API
np.import_array()

# Include the definitions
include "FuntofemDefs.pxi"

# Wrap the transfer scheme class and its functions
cdef class pyTransferScheme:
    """
    Abstract class that defines the transfer scheme interface

    Notes
    -----
    C++ extension must be compiled in complex mode in order to use complex
    step approximation in test functions

    """
    cdef TransferScheme *ptr

    def setAeroNodes(self, np.ndarray[F2FScalar, ndim=1, mode='c'] X):
        """
        Set and store the aerodynamic surface node locations in memory

        Parameters
        ----------
        X: ndarray
            One-dimensional array of aerodynamic surface node locations

        """
        cdef int nnodes = int(len(X)/3)
        self.ptr.setAeroNodes(<F2FScalar*>X.data, nnodes)

        return

    def setStructNodes(self, np.ndarray[F2FScalar, ndim=1, mode='c'] X):
        """
        Set and store the structural node locations in memory

        Parameters
        ----------
        X: ndarray
            One-dimensional array of structural node locations

        """
        cdef int nnodes = int(len(X)/3)
        self.ptr.setStructNodes(<F2FScalar*>X.data, nnodes)

        return

    def initialize(self):
        """
        Run routines (e.g. building connectivity through search, assembling
        interpolation matrix, etc.) necessary to prepare transfer scheme to
        perform load and displacement transfer

        """
        self.ptr.initialize()
        
        return

    def transferDisps(self, 
            np.ndarray[F2FScalar, ndim=1, mode='c'] struct_disps,
            np.ndarray[F2FScalar, ndim=1, mode='c'] aero_disps):
        """
        Convert the input structural node displacements into aerodynamic
        surface node displacements and store in empty input array

        Parameters
        ----------
        struct_disps: ndarray
            One-dimensional array of structural displacements
        aero_disps: ndarray
            One-dimensional empty array of size of aerodynamic displacements

        """
        self.ptr.transferDisps(<F2FScalar*>struct_disps.data,
                               <F2FScalar*>aero_disps.data)
        return

    def transferLoads(self, 
            np.ndarray[F2FScalar, ndim=1, mode='c'] aero_loads,
            np.ndarray[F2FScalar, ndim=1, mode='c'] struct_loads):
        """
        Convert the input aerodynamic surface loads into structural loads and
        store in empty input array

        Parameters
        ----------
        aero_loads: ndarray
            One-dimensional array of aerodynamic surface loads
        struct_loads: ndarray
            One-dimensional empty array of size of structural loads

        """
        self.ptr.transferLoads(<F2FScalar*>aero_loads.data,
                               <F2FScalar*>struct_loads.data)
        return

    def applydDduS(self, np.ndarray[F2FScalar, ndim=1, mode='c'] v,
                   np.ndarray[F2FScalar, ndim=1, mode='c'] p):
        """
        Apply the action of the Jacobian containing the derivatives of the
        displacement transfer residuals with respect to the structural
        displacements to an input vector and stores the products in empty
        input array

        Parameters
        ----------
        v: ndarray
            One-dimensional array of size of structural displacements
        p: ndarray
            One-dimensional empty array of size of aerodynamic displacements

        """
        self.ptr.applydDduS(<F2FScalar*>v.data, <F2FScalar*>p.data)
        return

    def applydDduSTrans(self, np.ndarray[F2FScalar, ndim=1, mode='c'] v,
                        np.ndarray[F2FScalar, ndim=1, mode='c'] p):
        """
        Apply the action of the transpose of the Jacobian containing the
        derivatives of the displacement transfer residuals with respect to the
        structural displacements to an input vector and store the products in
        empty input array

        Parameters
        ----------
        v: ndarray
            One-dimensional array of size of aerodynamic displacements
        p: ndarray
            One-dimensional empty array of size of structural displacements

        """
        self.ptr.applydDduSTrans(<F2FScalar*>v.data, <F2FScalar*>p.data)
        return

    def applydLduS(self, np.ndarray[F2FScalar, ndim=1, mode='c'] v,
                   np.ndarray[F2FScalar, ndim=1, mode='c'] p):
        """
        Apply the action of the Jacobian containing the derivatives of the load
        transfer residuals with respect to the structural displacements to an
        input vector and store the products in empty input array

        Parameters
        ----------
        v: ndarray
            One-dimensional array of size of structural displacements
        p: ndarray
            One-dimensional empty array of size of structural loads

        """
        self.ptr.applydLduS(<F2FScalar*>v.data, <F2FScalar*>p.data)
        return

    def applydLduSTrans(self, np.ndarray[F2FScalar, ndim=1, mode='c'] v,
                        np.ndarray[F2FScalar, ndim=1, mode='c'] p):
        """
        Apply the action of the transpose of the Jacobian containing the
        derivatives of the load transfer residuals with respect to the
        structural displacements to an input vector and store the products in
        empty input array

        Parameters
        ----------
        v: ndarray
            One-dimensional array of size of structural loads
        p: ndarray
            One-dimensional empty array of size of structural displacements

        """
        self.ptr.applydLduSTrans(<F2FScalar*>v.data, <F2FScalar*>p.data)
        return

    def applydDdxA0(self, np.ndarray[F2FScalar, ndim=1, mode='c'] v,
                    np.ndarray[F2FScalar, ndim=1, mode='c'] p):
        """
        Apply the action of the Jacobian containing the derivatives of the
        displacement transfer residuals with respect to the initial aerodynamic
        surface node locations to an input vector and store the products in
        empty input array

        Parameters
        ----------
        v: ndarray
            One-dimensional array of size of aerodynamic surface node locations
        p: ndarray
            One-dimensional empty array of size of aerodynamic displacements

        """
        self.ptr.applydDdxA0(<F2FScalar*>v.data, <F2FScalar*>p.data)
        return

    def applydDdxS0(self, np.ndarray[F2FScalar, ndim=1, mode='c'] v,
                    np.ndarray[F2FScalar, ndim=1, mode='c'] p):
        """
        Apply the action of the Jacobian containing the derivatives of the
        displacement transfer residuals with respect to the initial structural
        node locations to an input vector and store the products in empty input
        array

        Parameters
        ----------
        v: ndarray
            One-dimensional array of size of structural node locations
        p: ndarray
            One-dimensional empty array of size of aerodynamic displacements

        """
        self.ptr.applydDdxS0(<F2FScalar*>v.data, <F2FScalar*>p.data)
        return

    def applydLdxA0(self, np.ndarray[F2FScalar, ndim=1, mode='c'] v,
                    np.ndarray[F2FScalar, ndim=1, mode='c'] p):
        """
        Apply the action of the Jacobian containing the derivatives of the
        displacement transfer residuals with respect to the initial structural
        node locations to an input vector and store the products in empty input
        array

        Parameters
        ----------
        v: ndarray
            One-dimensional array of size of structural node locations
        p: ndarray
            One-dimensional empty array of size of structural loads

        """
        self.ptr.applydLdxA0(<F2FScalar*>v.data, <F2FScalar*>p.data)
        return

    def applydLdxS0(self, np.ndarray[F2FScalar, ndim=1, mode='c'] v,
                    np.ndarray[F2FScalar, ndim=1, mode='c'] p):
        """
        Apply the action of the Jacobian containing the derivatives of the load
        transfer residuals with respect to the initial structural node locations
        to an input vector and store the products in empty input array

        Parameters
        ----------
        v: ndarray
            One-dimensional array of size of structural node locations
        p: ndarray
            One-dimensional empty array of size of structural loads

        """
        self.ptr.applydLdxS0(<F2FScalar*>v.data, <F2FScalar*>p.data)
        return

    def testLoadTransfer(self, 
            np.ndarray[F2FScalar, ndim=1, mode='c'] struct_disps,
            np.ndarray[F2FScalar, ndim=1, mode='c'] aero_loads,
            np.ndarray[F2FScalar, ndim=1, mode='c'] test_vec_s,
            F2FScalar h):
        """
        Test the output of :meth:`transferLoads` by comparison with results from
        finite difference approximation or complex step approximation

        Parameters
        ----------
        struct_disps: ndarray
            One-dimensional array of structural displacements
        aero_loads: ndarray
            One-dimensional array of aerodynamic loads
        test_vec_s: ndarray
            One-dimensional array of perturbations of size of structural
            displacements
        h: float
            Step size (for finite difference or complex step)

        """
        self.ptr.testLoadTransfer(<F2FScalar*>struct_disps.data,
                                  <F2FScalar*>aero_loads.data,
                                  <F2FScalar*>test_vec_s.data, h)

        return

    def testDispJacVecProducts(self, 
            np.ndarray[F2FScalar, ndim=1, mode='c'] struct_disps,
            np.ndarray[F2FScalar, ndim=1, mode='c'] test_vec_a,
            np.ndarray[F2FScalar, ndim=1, mode='c'] test_vec_s,
            F2FScalar h):
        """
        Test output of :meth:`applydDduS` and :meth:`applydDduSTrans` by
        comparison with results from finite difference approximation or
        complex step approximation

        Parameters
        ----------
        struct_disps: ndarray
            One-dimensional array of structural displacements
        test_vec_a: ndarray
            One-dimensional array of perturbations of size of displacement
            transfer residuals
        test_vec_s: ndarray
            One-dimensional array of perturbations of size of structural
            displacements
        h: float
            Step size (for finite difference or complex step)

        """
        self.ptr.testDispJacVecProducts(<F2FScalar*>struct_disps.data,
                                        <F2FScalar*>test_vec_a.data,
                                        <F2FScalar*>test_vec_s.data, h)

        return

    def testLoadJacVecProducts(self, 
            np.ndarray[F2FScalar, ndim=1, mode='c'] struct_disps,
            np.ndarray[F2FScalar, ndim=1, mode='c'] aero_loads,
            np.ndarray[F2FScalar, ndim=1, mode='c'] test_vec_s1,
            np.ndarray[F2FScalar, ndim=1, mode='c'] test_vec_s2,
            F2FScalar h):
        """
        Test output of :meth:`applydLduS` and :meth:`applydLduSTrans` by
        comparison with results from finite difference approximation or
        complex step approximation

        Parameters
        ----------
        struct_disps: ndarray
            One-dimensional array of structural displacements
        aero_loads: ndarray
            One-dimensional array of aerodynamic loads
        test_vec_s1: ndarray
            One-dimensional array of perturbations of size of load transfer
            residuals
        test_vec_s2: ndarray
            One-dimensional array of perturbations of size of structural
            displacements
        h: float
            Step size (for finite difference or complex step)

        """
        self.ptr.testLoadJacVecProducts(<F2FScalar*>struct_disps.data,
                                        <F2FScalar*>aero_loads.data,
                                        <F2FScalar*>test_vec_s1.data,
                                        <F2FScalar*>test_vec_s2.data, h)

        return

    def testdDdxA0Products(self, 
            np.ndarray[F2FScalar, ndim=1, mode='c'] struct_disps,
            np.ndarray[F2FScalar, ndim=1, mode='c'] test_vec_a1,
            np.ndarray[F2FScalar, ndim=1, mode='c'] test_vec_a2,
            F2FScalar h):
        """
        Test output of :meth:`applydDdxA0` by comparison with results from
        finite difference approximation or complex step approximation

        Parameters
        ----------
        struct_disps: ndarray
            One-dimensional array of structural displacements
        test_vec_a1: ndarray
            One-dimensional array of perturbations of size of displacement
            transfer residuals
        test_vec_a2: ndarray
            One-dimensional array of perturbations of size of aerodynamic
            displacements
        h: float
            Step size (for finite difference or complex step)

        """
        self.ptr.testdDdxA0Products(<F2FScalar*>struct_disps.data,
                                    <F2FScalar*>test_vec_a1.data,
                                    <F2FScalar*>test_vec_a2.data, h)

        return

    def testdDdxS0Products(self, 
            np.ndarray[F2FScalar, ndim=1, mode='c'] struct_disps,
            np.ndarray[F2FScalar, ndim=1, mode='c'] test_vec_a,
            np.ndarray[F2FScalar, ndim=1, mode='c'] test_vec_s,
            F2FScalar h):
        """
        Test output of :meth:`applydDdxS0` by comparison with results from
        finite difference approximation or complex step approximation

        Parameters
        ----------
        struct_disps: ndarray
            One-dimensional array of structural displacements
        test_vec_a: ndarray
            One-dimensional array of perturbations of size of displacement
            transfer residuals
        test_vec_s: ndarray
            One-dimensional array of perturbations of size of structural
            displacements
        h: float
            Step size (for finite difference or complex step)

        """
        self.ptr.testdDdxS0Products(<F2FScalar*>struct_disps.data,
                                    <F2FScalar*>test_vec_a.data,
                                    <F2FScalar*>test_vec_s.data, h)

        return

    def testdLdxA0Products(self, 
            np.ndarray[F2FScalar, ndim=1, mode='c'] struct_disps,
            np.ndarray[F2FScalar, ndim=1, mode='c'] aero_loads,
            np.ndarray[F2FScalar, ndim=1, mode='c'] test_vec_a,
            np.ndarray[F2FScalar, ndim=1, mode='c'] test_vec_s,
            F2FScalar h):
        """
        """
        self.ptr.testdDdxS0Products(<F2FScalar*>struct_disps.data,
                                    <F2FScalar*>test_vec_a.data,
                                    <F2FScalar*>test_vec_s.data, h)

        return

    def testdLdxA0Products(self, 
            np.ndarray[F2FScalar, ndim=1, mode='c'] struct_disps,
            np.ndarray[F2FScalar, ndim=1, mode='c'] aero_loads,
            np.ndarray[F2FScalar, ndim=1, mode='c'] test_vec_a,
            np.ndarray[F2FScalar, ndim=1, mode='c'] test_vec_s,
            F2FScalar h):
        """
        Test output of :meth:`applydLdxA0` by comparison with results from
        finite difference approximation or complex step approximation

        Parameters
        ----------
        struct_disps: ndarray
            One-dimensional array of structural displacements
        aero_loads: ndarray
            One-dimensional array of aerodynamic loads
        test_vec_a: ndarray
            One-dimensional array of perturbations of size of displacement
            transfer residuals
        test_vec_s: ndarray
            One-dimensional array of perturbations of size of structural
            displacements
        h: float
            Step size (for finite difference or complex step)

        """
        self.ptr.testdLdxA0Products(<F2FScalar*>struct_disps.data,
                                    <F2FScalar*>aero_loads.data,
                                    <F2FScalar*>test_vec_a.data,
                                    <F2FScalar*>test_vec_s.data, h)

        return

    def testdLdxS0Products(self, 
            np.ndarray[F2FScalar, ndim=1, mode='c'] struct_disps,
            np.ndarray[F2FScalar, ndim=1, mode='c'] aero_loads,
            np.ndarray[F2FScalar, ndim=1, mode='c'] test_vec_s1,
            np.ndarray[F2FScalar, ndim=1, mode='c'] test_vec_s2,
            F2FScalar h):
        """
        Test output of :meth:`applydLdxS0` by comparison with results from
        finite difference approximation or complex step approximation

        Parameters
        ----------
        struct_disps: ndarray
            One-dimensional array of structural displacements
        aero_loads: ndarray
            One-dimensional array of aerodynamic loads
        test_vec_s1: ndarray
            One-dimensional array of perturbations of size of load transfer
            residuals
        test_vec_s2: ndarray
            One-dimensional array of perturbations of size of structural
            displacements
        h: float
            Step size (for finite difference or complex step)

        """
        self.ptr.testdLdxS0Products(<F2FScalar*>struct_disps.data,
                                    <F2FScalar*>aero_loads.data,
                                    <F2FScalar*>test_vec_s1.data,
                                    <F2FScalar*>test_vec_s2.data, h)

        return

    def transformEquivRigidMotion(self, 
        np.ndarray[F2FScalar, ndim=1, mode='c'] aero_disps,
        np.ndarray[F2FScalar, ndim=1, mode='c'] R,
        np.ndarray[F2FScalar, ndim=1, mode='c'] t,
        np.ndarray[F2FScalar, ndim=1, mode='c'] u):
        """
        Compute least-squares fit of rigid rotation and translation to set of
        aerodynamic surface displacements.
        Use rotation and translation to compute difference between given
        displacements and the displacements due to rigid motion alone,
        i.e. the elastic deformation.
        Store the computed rotation, translation, and elastic deformation in
        empty input arrays

        Parameters
        ----------
        aero_disps: ndarray
            One-dimensional array of aerodynamic displacements
        R: (3,3) ndarray
            Empty (3,3) array for output rotation matrix
        t: (,3) ndarray
            Empty (3,1) array for output translatio
        u: ndarray
            One-dimensional array of elastic deformations of size of
            aerodynamic displacements

        """
        self.ptr.transformEquivRigidMotion(<F2FScalar*>aero_disps.data,
                                           <F2FScalar*>R.data,
                                           <F2FScalar*>t.data,
                                           <F2FScalar*>u.data)

        return

    def applydRduATrans(self, 
        np.ndarray[F2FScalar, ndim=1, mode='c'] vecs,
        np.ndarray[F2FScalar, ndim=1, mode='c'] prods):
        """
        Compute rigid transform Jacobian-vector products for solving adjoint
        equations

        Parameters
        ----------
        vecs: ndarray
            array of rigid transform adjoint variables
        prods: ndarray
            array of Jacobian-vector products

        """
        self.ptr.applydRduATrans(<F2FScalar*>vecs.data,
                                 <F2FScalar*>prods.data)

        return

    def applydRdxA0Trans(self,
        np.ndarray[F2FScalar, ndim=1, mode='c'] aero_disps,
        np.ndarray[F2FScalar, ndim=1, mode='c'] vecs,
        np.ndarray[F2FScalar, ndim=1, mode='c'] prods):
        """
        Compute rigid transform Jacobian-vector products for gradient assembly

        Parameters
        ----------
        aero_disps: ndarray
            One-dimensional array of aerodynamic displacements
        vecs: ndarray
            array of rigid transform adjoint variables
        prods: ndarray
            array of Jacobian-vector products

        """
        self.ptr.applydRdxA0Trans(<F2FScalar*>aero_disps.data,
                             <F2FScalar*>vecs.data,
                             <F2FScalar*>prods.data)

        return

# Wrap the MELD class
cdef class pyMELD(pyTransferScheme):
    """
    MELD (Matching-based Extrapolation of Loads and Displacments) is scalable
    scheme for transferring loads and displacements between large non-matching
    aerodynamic and structural meshes. It connects each aerodynamic node to a
    specified number of nearest structural nodes, and extrapolates its motion
    from the connected structural nodes through the solution of a shape-matching
    problem. The aerodynamic loads are extrapolated to the structural mesh in a
    consistent and conservative manner, derived from the principle of virtual
    work

    Parameters
    ----------
    comm: MPI.comm
        MPI communicator for all processes
    struct: MPI.comm
        MPI communicator for the structural root process
    struct_root: int
        id of the structural root process
    aero: MPI.comm
        MPI communicator for the aerodynamic root process
    aero_root: int
        id of the aerodynamic root process
    symmetry: int
        symmetry specifier (-1 for none, 0 for x-plane, 1 for y-plane,
        2 for z-plane)
    num_nearest: int
        number of structural nodes linked to each aerodynamic node
    beta: float
        weighting decay parameter

    """
    def __cinit__(self, MPI.Comm comm,
                  MPI.Comm struct, int struct_root,
                  MPI.Comm aero, int aero_root,
                  int symmetry, int num_nearest,
                  F2FScalar beta):
        cdef MPI_Comm c_comm = comm.ob_mpi
        cdef MPI_Comm struct_comm = struct.ob_mpi
        cdef MPI_Comm aero_comm = aero.ob_mpi

        # Allocate the underlying class
        self.ptr = new MELD(c_comm, struct_comm, struct_root, 
                            aero_comm, aero_root, symmetry,
                            num_nearest, beta)

        return

    def __dealloc__(self):
        del self.ptr


# Wrap the MELD class
cdef class pyMELDThermal(pyTransferScheme):
    """
    MELD (Matching-based Extrapolation of Loads and Displacments) is scalable
    scheme for transferring loads and displacements between large non-matching
    aerodynamic and structural meshes. It connects each aerodynamic node to a
    specified number of nearest structural nodes, and extrapolates its motion
    from the connected structural nodes through the solution of a shape-matching
    problem. The aerodynamic loads are extrapolated to the structural mesh in a
    consistent and conservative manner, derived from the principle of virtual
    work

    Version modified to transfer temperature and flux rather than load and displacement

    Parameters
    ----------
    comm: MPI.comm
        MPI communicator for all processes
    struct: MPI.comm
        MPI communicator for the structural root process
    struct_root: int
        id of the structural root process
    aero: MPI.comm
        MPI communicator for the aerodynamic root process
    aero_root: int
        id of the aerodynamic root process
    symmetry: int
        symmetry specifier (-1 for none, 0 for x-plane, 1 for y-plane,
        2 for z-plane)
    num_nearest: int
        number of structural nodes linked to each aerodynamic node
    beta: float
        weighting decay parameter

    """
    def __cinit__(self, MPI.Comm comm,
                  MPI.Comm struct, int struct_root,
                  MPI.Comm aero, int aero_root,
                  int symmetry, int num_nearest,
                  F2FScalar beta):
        cdef MPI_Comm c_comm = comm.ob_mpi
        cdef MPI_Comm struct_comm = struct.ob_mpi
        cdef MPI_Comm aero_comm = aero.ob_mpi

        # Allocate the underlying class
        self.ptr = new MELDThermal(c_comm, struct_comm, struct_root, 
                                   aero_comm, aero_root, symmetry,
                                   num_nearest, beta)

        return

    def __dealloc__(self):
        del self.ptr

    def transferTemp(self, 
            np.ndarray[F2FScalar, ndim=1, mode='c'] struct_temp,
                     np.ndarray[F2FScalar, ndim=1, mode='c'] aero_temp):
        """
        Convert the input structural node displacements into aerodynamic
        surface node displacements and store in empty input array

        Parameters
        ----------
        struct_disps: ndarray
            One-dimensional array of structural temperatures
        aero_disps: ndarray
            One-dimensional empty array of size of aerodynamic temperatures

        """
        cdef MELDThermal *mt = <MELDThermal*> self.ptr
        mt.transferTemp(<F2FScalar*>struct_temp.data, <F2FScalar*>aero_temp.data)         
        return

    def transferFlux(self, 
            np.ndarray[F2FScalar, ndim=1, mode='c'] aero_flux,
                     np.ndarray[F2FScalar, ndim=1, mode='c'] struct_flux):
        """
        Convert the input aerodynamic surface loads into structural loads and
        store in empty input array

        Parameters
        ----------
        aero_loads: ndarray
            One-dimensional array of aerodynamic surface flux
        struct_loads: ndarray
            One-dimensional empty array of size of structural flux

        """
        cdef MELDThermal *mt = <MELDThermal*> self.ptr
        mt.transferFlux(<F2FScalar*>aero_flux.data, <F2FScalar*>struct_flux.data)
        return


# Wrap the MELD class
cdef class pyLinearizedMELD(pyTransferScheme):
    """
    Linearized MELD is a transfer scheme developed from the MELD transfer scheme
    assuming displacements tend to zero

    Parameters
    ----------
    comm: MPI.comm
        MPI communicator for all processes
    struct: MPI.comm
        MPI communicator for the structural root process
    struct_root: int
        id of the structural root process
    aero: MPI.comm
        MPI communicator for the aerodynamic root process
    aero_root: int
        id of the aerodynamic root process
    num_nearest: int
        number of structural nodes linked to each aerodynamic node
    beta: float
        weighting decay parameter

    """
    def __cinit__(self, MPI.Comm comm,
                  MPI.Comm struct, int struct_root,
                  MPI.Comm aero, int aero_root,
                  int num_nearest, F2FScalar beta):
        cdef MPI_Comm c_comm = comm.ob_mpi
        cdef MPI_Comm struct_comm = struct.ob_mpi
        cdef MPI_Comm aero_comm = aero.ob_mpi

        # Allocate the underlying class
        self.ptr = new LinearizedMELD(c_comm, struct_comm, struct_root, 
                                      aero_comm, aero_root, 
                                      num_nearest, beta)

        return

    def __dealloc__(self):
        del self.ptr

# Wrap the MELD class
PY_GAUSSIAN = GAUSSIAN
PY_MULTIQUADRIC = MULTIQUADRIC
PY_INVERSE_MULTIQUADRIC = INVERSE_MULTIQUADRIC
PY_THIN_PLATE_SPLINE = THIN_PLATE_SPLINE

cdef class pyRBF(pyTransferScheme):
    """
    Interpolation of loads and displacements using radial basis functions (RBFs)

    Parameters
    ----------
    comm: MPI.comm
        MPI communicator for all processes
    struct: MPI.comm
        MPI communicator for the structural root process
    struct_root: int
        id of the structural root process
    aero: MPI.comm
        MPI communicator for the aerodynamic root process
    aero_root: int
        id of the aerodynamic root process
    rbf_type: C++ enum
        type of radial basis function to use (PY_GAUSSIAN, PY_MULTIQUADRIC,
        PY_INVERSE_MULTIQUADRIC, PY_THIN_PLATE_SPLINE)
    sampling_ratio: int
        minimum number of points in leaf node of octree (one point sampled
        from each node)

    """
    def __cinit__(self, MPI.Comm comm,
                  MPI.Comm struct, int struct_root,
                  MPI.Comm aero, int aero_root,
                  RbfType rbf_type, int sampling_ratio):
        """

        Parameters
        ----------
        comm
        struct
        struct_root
        aero
        aero_root
        rbf_type
        sampling_ratio

        Returns
        -------

        """
        cdef MPI_Comm c_comm = comm.ob_mpi
        cdef MPI_Comm struct_comm = struct.ob_mpi
        cdef MPI_Comm aero_comm = aero.ob_mpi

        # Allocate the underlying class
        self.ptr = new RBF(c_comm, struct_comm, struct_root, 
                           aero_comm, aero_root, 
                           rbf_type, sampling_ratio)

        return

    def __dealloc__(self):
        del self.ptr
