#!/bin/python
# distutils: language=c++
# For the use of MPI

__all__ = ["pyTransferScheme", "pyThermalTransfer", "pyMELD", "pyMELDThermal", "pyRBF", "pyLinearizedMELD", "pyBeamTransfer"]

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
    cdef LDTransferScheme *ptr

    def setAeroNodes(self, np.ndarray[F2FScalar, ndim=1, mode='c'] X):
        """
        Set and store the aerodynamic surface node locations in memory

        Parameters
        ----------
        X: ndarray
            One-dimensional array of aerodynamic surface node locations
        """
        cdef int nnodes = 0
        cdef F2FScalar *array = NULL
        if X is not None:
            nnodes = int(len(X)//3)
            array = <F2FScalar*>X.data

        self.ptr.setAeroNodes(array, nnodes)

        return

    def setStructNodes(self, np.ndarray[F2FScalar, ndim=1, mode='c'] X):
        """
        Set and store the structural node locations in memory

        Parameters
        ----------
        X: ndarray
            One-dimensional array of structural node locations
        """
        cdef int nnodes = 0
        cdef F2FScalar *array = NULL
        if X is not None:
            nnodes = int(len(X)//3)
            array = <F2FScalar*>X.data

        self.ptr.setStructNodes(array, nnodes)

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
        cdef F2FScalar *struct_array = NULL
        cdef int struct_len = 0
        cdef F2FScalar *aero_array = NULL
        cdef int aero_len = 0
        if struct_disps is not None:
            struct_len = len(struct_disps)
            struct_array = <F2FScalar*>struct_disps.data
        if aero_disps is not None:
            aero_len = len(aero_disps)
            aero_array = <F2FScalar*>aero_disps.data

        if struct_len != self.ptr.getLocalStructArrayLen():
            raise ValueError("Structural array incorrect length")
        if aero_len != self.ptr.getLocalAeroArrayLen():
            raise ValueError("Aerodynamic array incorrect length")

        self.ptr.transferDisps(struct_array, aero_array)

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
        cdef F2FScalar *struct_array = NULL
        cdef int struct_len = 0
        cdef F2FScalar *aero_array = NULL
        cdef int aero_len = 0
        if struct_loads is not None:
            struct_len = len(struct_loads)
            struct_array = <F2FScalar*>struct_loads.data
        if aero_loads is not None:
            aero_len = len(aero_loads)
            aero_array = <F2FScalar*>aero_loads.data

        if struct_len != self.ptr.getLocalStructArrayLen():
            raise ValueError("Structural array incorrect length")
        if aero_len != self.ptr.getLocalAeroArrayLen():
            raise ValueError("Aerodynamic array incorrect length")

        self.ptr.transferLoads(aero_array, struct_array)

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
        cdef F2FScalar *struct_array = NULL
        cdef int struct_len = 0
        cdef F2FScalar *aero_array = NULL
        cdef int aero_len = 0
        if v is not None:
            struct_len = len(v)
            struct_array = <F2FScalar*>v.data
        if p is not None:
            aero_len = len(p)
            aero_array = <F2FScalar*>p.data

        if struct_len != self.ptr.getLocalStructArrayLen():
            raise ValueError("Structural array incorrect length")
        if aero_len != self.ptr.getLocalAeroArrayLen():
            raise ValueError("Aerodynamic array incorrect length")

        self.ptr.applydDduS(struct_array, aero_array)
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
        cdef F2FScalar *struct_array = NULL
        cdef int struct_len = 0
        cdef F2FScalar *aero_array = NULL
        cdef int aero_len = 0
        if p is not None:
            struct_len = len(p)
            struct_array = <F2FScalar*>p.data
        if v is not None:
            aero_len = len(v)
            aero_array = <F2FScalar*>v.data

        if struct_len != self.ptr.getLocalStructArrayLen():
            raise ValueError("Structural array incorrect length")
        if aero_len != self.ptr.getLocalAeroArrayLen():
            raise ValueError("Aerodynamic array incorrect length")

        self.ptr.applydDduSTrans(aero_array, struct_array)
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
        cdef F2FScalar *struct_array1 = NULL
        cdef int struct_len1 = 0
        cdef F2FScalar *struct_array2 = NULL
        cdef int struct_len2 = 0
        if v is not None:
            struct_len1 = len(v)
            struct_array1 = <F2FScalar*>v.data
        if p is not None:
            struct_len2 = len(p)
            struct_array2 = <F2FScalar*>p.data

        if struct_len1 != self.ptr.getLocalStructArrayLen():
            raise ValueError("Input structural array incorrect length")
        if struct_len2 != self.ptr.getLocalStructArrayLen():
            raise ValueError("Output structural array incorrect length")

        self.ptr.applydLduS(struct_array1, struct_array2)
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

        cdef F2FScalar *struct_array1 = NULL
        cdef int struct_len1 = 0
        cdef F2FScalar *struct_array2 = NULL
        cdef int struct_len2 = 0
        if v is not None:
            struct_len1 = len(v)
            struct_array1 = <F2FScalar*>v.data
        if p is not None:
            struct_len2 = len(p)
            struct_array2 = <F2FScalar*>p.data

        if struct_len1 != self.ptr.getLocalStructArrayLen():
            raise ValueError("Input structural array incorrect length")
        if struct_len2 != self.ptr.getLocalStructArrayLen():
            raise ValueError("Output structural array incorrect length")

        self.ptr.applydLduSTrans(struct_array1, struct_array2)
        return

    def applydLdfA(self, np.ndarray[F2FScalar, ndim=1, mode='c'] v,
                   np.ndarray[F2FScalar, ndim=1, mode='c'] p):
        """
        Apply the action of the Jacobian containing the derivatives of the load
        transfer residuals with respect to the aerodynamic forces to an
        input vector and store the products in empty input array

        Parameters
        ----------
        v: ndarray
            One-dimensional array of size of aerodynamic forces
        p: ndarray
            One-dimensional empty array of size of structural loads
        """
        cdef F2FScalar *struct_array = NULL
        cdef int struct_len = 0
        cdef F2FScalar *aero_array = NULL
        cdef int aero_len = 0
        if p is not None:
            struct_len = len(p)
            struct_array = <F2FScalar*>p.data
        if v is not None:
            aero_len = len(v)
            aero_array = <F2FScalar*>v.data

        if struct_len != self.ptr.getLocalStructArrayLen():
            raise ValueError("Structural array incorrect length")
        if aero_len != self.ptr.getLocalAeroArrayLen():
            raise ValueError("Aerodynamic array incorrect length")

        self.ptr.applydLdfA(aero_array, struct_array)
        return

    def applydLdfATrans(self, np.ndarray[F2FScalar, ndim=1, mode='c'] v,
                        np.ndarray[F2FScalar, ndim=1, mode='c'] p):
        """
        Apply the action of the transpose of the Jacobian containing the
        derivatives of the load transfer residuals with respect to the
        aerodynamic forces to an input vector and store the products in
        empty input array

        Parameters
        ----------
        v: ndarray
            One-dimensional array of size of structural loads
        p: ndarray
            One-dimensional empty array of size of aerodynamic forces
        """
        cdef F2FScalar *struct_array = NULL
        cdef int struct_len = 0
        cdef F2FScalar *aero_array = NULL
        cdef int aero_len = 0
        if v is not None:
            struct_len = len(v)
            struct_array = <F2FScalar*>v.data
        if p is not None:
            aero_len = len(p)
            aero_array = <F2FScalar*>p.data

        if struct_len != self.ptr.getLocalStructArrayLen():
            raise ValueError("Structural array incorrect length")
        if aero_len != self.ptr.getLocalAeroArrayLen():
            raise ValueError("Aerodynamic array incorrect length")

        self.ptr.applydLdfATrans(struct_array, aero_array)
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
            One-dimensional empty array of size of aerodynamic nodes
        """
        cdef F2FScalar *aero_array = NULL
        cdef int aero_len = 0
        cdef F2FScalar *xA0_array = NULL
        cdef int xA0_len = 0
        if v is not None:
            aero_len = len(v)
            aero_array = <F2FScalar*>v.data
        if p is not None:
            xA0_len = len(p)
            xA0_array = <F2FScalar*>p.data

        if aero_len != self.ptr.getLocalAeroArrayLen():
            raise ValueError("Aerodynamic array incorrect length")
        if xA0_len != 3 * self.ptr.getNumLocalAeroNodes():
            raise ValueError("Aerodynamic node array incorrect length")

        self.ptr.applydDdxA0(aero_array, xA0_array)
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
            One-dimensional array of size of aerodynamic surface node locations
        p: ndarray
            One-dimensional empty array of size of structural nodes
        """
        cdef F2FScalar *aero_array = NULL
        cdef int aero_len = 0
        cdef F2FScalar *xA0_array = NULL
        cdef int xA0_len = 0
        if v is not None:
            aero_len = len(v)
            aero_array = <F2FScalar*>v.data
        if p is not None:
            xS0_len = len(p)
            xS0_array = <F2FScalar*>p.data

        if aero_len != self.ptr.getLocalAeroArrayLen():
            raise ValueError("Aerodynamic array incorrect length")
        if xS0_len != 3 * self.ptr.getNumLocalStructNodes():
            raise ValueError("Structural node array incorrect length")

        self.ptr.applydDdxS0(aero_array, xS0_array)
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
        cdef F2FScalar *struct_array = NULL
        cdef int struct_len = 0
        cdef F2FScalar *xA0_array = NULL
        cdef int xA0_len = 0
        if v is not None:
            struct_len = len(v)
            struct_array = <F2FScalar*>v.data
        if p is not None:
            xA0_len = len(p)
            xA0_array = <F2FScalar*>p.data

        if struct_len != self.ptr.getLocalStructArrayLen():
            raise ValueError("Structural array incorrect length")
        if xA0_len != 3 * self.ptr.getNumLocalAeroNodes():
            raise ValueError("Aerodynamic node array incorrect length")

        self.ptr.applydLdxA0(struct_array, xA0_array)
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
        cdef F2FScalar *struct_array = NULL
        cdef int struct_len = 0
        cdef F2FScalar *xA0_array = NULL
        cdef int xA0_len = 0
        if v is not None:
            struct_len = len(v)
            struct_array = <F2FScalar*>v.data
        if p is not None:
            xS0_len = len(p)
            xS0_array = <F2FScalar*>p.data

        if struct_len != self.ptr.getLocalStructArrayLen():
            raise ValueError("Structural array incorrect length")
        if xS0_len != 3 * self.ptr.getNumLocalStructNodes():
            raise ValueError("Structural node array incorrect length")

        self.ptr.applydLdxS0(struct_array, xS0_array)
        return

    def testAllDerivatives(self,
            np.ndarray[F2FScalar, ndim=1, mode='c'] struct_disps,
            np.ndarray[F2FScalar, ndim=1, mode='c'] aero_loads,
            F2FScalar h, double rtol=1e-6, double atol=1e-30):
        """
        Test the output of :meth:`transferLoads` by comparison with results from
        finite difference approximation or complex step approximation

        Parameters
        ----------
        struct_disps: ndarray
            One-dimensional array of structural displacements
        aero_loads: ndarray
            One-dimensional array of aerodynamic loads
        h: float
            Step size (for finite difference or complex step)
        rtol: float
            Relative error tolerance used in the test
        atol: float
            Absolute error tolerance used in the test
        """
        cdef F2FScalar *struct_array = NULL
        cdef int struct_len = 0
        cdef F2FScalar *aero_array = NULL
        cdef int aero_len = 0
        if struct_disps is not None:
            struct_len = len(struct_disps)
            struct_array = <F2FScalar*>struct_disps.data
        if aero_loads is not None:
            aero_len = len(aero_loads)
            aero_array = <F2FScalar*>aero_loads.data

        return self.ptr.testAllDerivatives(struct_array, aero_array, h, rtol, atol)

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
        cdef F2FScalar *aero_array = NULL
        cdef int aero_len = 0
        if aero_disps is not None:
            aero_len = len(aero_disps)
            aero_array = <F2FScalar*>aero_disps.data

        if aero_len != self.ptr.getLocalAeroArrayLen():
            raise ValueError("Aerodynamic array incorrect length")

        self.ptr.transformEquivRigidMotion(aero_array,
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

# Generic thermal transfer scheme
cdef class pyThermalTransfer:
    """
    Abstract class that defines the transfer scheme interface

    Notes
    -----
    C++ extension must be compiled in complex mode in order to use complex
    step approximation in test functions
    """
    cdef ThermalTransfer *ptr

    def setAeroNodes(self, np.ndarray[F2FScalar, ndim=1, mode='c'] X):
        """
        Set and store the aerodynamic surface node locations in memory

        Parameters
        ----------
        X: ndarray
            One-dimensional array of aerodynamic surface node locations
        """
        cdef int nnodes = 0
        cdef F2FScalar *array = NULL
        if X is not None:
            nnodes = int(len(X)//3)
            array = <F2FScalar*>X.data

        self.ptr.setAeroNodes(array, nnodes)

        return

    def setStructNodes(self, np.ndarray[F2FScalar, ndim=1, mode='c'] X):
        """
        Set and store the structural node locations in memory

        Parameters
        ----------
        X: ndarray
            One-dimensional array of structural node locations
        """
        cdef int nnodes = 0
        cdef F2FScalar *array = NULL
        if X is not None:
            nnodes = int(len(X)//3)
            array = <F2FScalar*>X.data

        self.ptr.setStructNodes(array, nnodes)

        return

    def initialize(self):
        """
        Run routines (e.g. building connectivity through search, assembling
        interpolation matrix, etc.) necessary to prepare transfer scheme to
        perform load and displacement transfer
        """
        self.ptr.initialize()

        return

    def transferTemp(self,
                     np.ndarray[F2FScalar, ndim=1, mode='c'] struct_temps,
                     np.ndarray[F2FScalar, ndim=1, mode='c'] aero_temps):
        """
        Convert the input structural node displacements into aerodynamic
        surface node displacements and store in empty input array

        Parameters
        ----------
        struct_temps: ndarray
            One-dimensional array of structural temperatures
        aero_temps: ndarray
            One-dimensional empty array of size of aerodynamic temperatures
        """
        cdef F2FScalar *struct_array = NULL
        cdef int struct_len = 0
        cdef F2FScalar *aero_array = NULL
        cdef int aero_len = 0
        if struct_temps is not None:
            struct_len = len(struct_temps)
            struct_array = <F2FScalar*>struct_temps.data
        if aero_temps is not None:
            aero_len = len(aero_temps)
            aero_array = <F2FScalar*>aero_temps.data

        if struct_len != self.ptr.getLocalStructArrayLen():
            raise ValueError("Structural array incorrect length")
        if aero_len != self.ptr.getLocalAeroArrayLen():
            raise ValueError("Aerodynamic array incorrect length")

        self.ptr.transferTemp(struct_array, aero_array)
        return

    def transferFlux(self,
            np.ndarray[F2FScalar, ndim=1, mode='c'] aero_flux,
            np.ndarray[F2FScalar, ndim=1, mode='c'] struct_flux):
        """
        Convert the input aerodynamic surface loads into structural loads and
        store in empty input array

        Parameters
        ----------
        aero_flux: ndarray
            One-dimensional array of aerodynamic surface heat flux
        struct_flux: ndarray
            One-dimensional empty array of size of structural heat flux
        """
        cdef F2FScalar *struct_array = NULL
        cdef int struct_len = 0
        cdef F2FScalar *aero_array = NULL
        cdef int aero_len = 0
        if struct_flux is not None:
            struct_len = len(struct_flux)
            struct_array = <F2FScalar*>struct_flux.data
        if aero_flux is not None:
            aero_len = len(aero_flux)
            aero_array = <F2FScalar*>aero_flux.data

        if struct_len != self.ptr.getLocalStructArrayLen():
            raise ValueError("Structural array incorrect length")
        if aero_len != self.ptr.getLocalAeroArrayLen():
            raise ValueError("Aerodynamic array incorrect length")

        self.ptr.transferFlux(aero_array, struct_array)
        return

    def applydTdtS(self, np.ndarray[F2FScalar, ndim=1, mode='c'] v,
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

        self.ptr.applydTdtS(<F2FScalar*>v.data, <F2FScalar*>p.data)
        return

    def applydTdtSTrans(self, np.ndarray[F2FScalar, ndim=1, mode='c'] v,
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
        self.ptr.applydTdtSTrans(<F2FScalar*>v.data, <F2FScalar*>p.data)
        return

    def applydQdqA(self, np.ndarray[F2FScalar, ndim=1, mode='c'] v,
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
        self.ptr.applydQdqA(<F2FScalar*>v.data, <F2FScalar*>p.data)
        return

    def applydQdqATrans(self, np.ndarray[F2FScalar, ndim=1, mode='c'] v,
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
        self.ptr.applydQdqATrans(<F2FScalar*>v.data, <F2FScalar*>p.data)
        return

    def testAllDerivatives(self,
            np.ndarray[F2FScalar, ndim=1, mode='c'] struct_disps,
            np.ndarray[F2FScalar, ndim=1, mode='c'] aero_loads,
            F2FScalar h, double rtol=1e-6, double atol=1e-30):
        """
        Test the output of :meth:`transferLoads` by comparison with results from
        finite difference approximation or complex step approximation

        Parameters
        ----------
        struct_disps: ndarray
            One-dimensional array of structural displacements
        aero_loads: ndarray
            One-dimensional array of aerodynamic loads
        h: float
            Step size (for finite difference or complex step)
        rtol: float
            Relative error tolerance used in the test
        atol: float
            Absolute error tolerance used in the test
        """
        cdef F2FScalar *struct_array = NULL
        cdef int struct_len = 0
        cdef F2FScalar *aero_array = NULL
        cdef int aero_len = 0
        if struct_disps is not None:
            struct_len = len(struct_disps)
            struct_array = <F2FScalar*>struct_disps.data
        if aero_loads is not None:
            aero_len = len(aero_loads)
            aero_array = <F2FScalar*>aero_loads.data

        return self.ptr.testAllDerivatives(struct_array, aero_array, h, rtol, atol)


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

cdef class pyMELDThermal(pyThermalTransfer):
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
                  int symmetry, int num_nearest, F2FScalar beta):
        cdef MPI_Comm c_comm = comm.ob_mpi
        cdef MPI_Comm struct_comm = struct.ob_mpi
        cdef MPI_Comm aero_comm = aero.ob_mpi

        # Allocate the underlying class
        self.ptr = new LinearizedMELD(c_comm, struct_comm, struct_root,
                                      aero_comm, aero_root,
                                      symmetry, num_nearest, beta)

        return

    def __dealloc__(self):
        del self.ptr

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

cdef class pyBeamTransfer(pyTransferScheme):
    """
    Interpolation of loads and displacements for beam elements
    """
    def __cinit__(self, MPI.Comm comm,
                  MPI.Comm struct, int struct_root,
                  MPI.Comm aero, int aero_root,
                  np.ndarray[int, ndim=2, mode='c'] conn,
                  int dof_per_node=6):
        cdef MPI_Comm c_comm = comm.ob_mpi
        cdef MPI_Comm struct_comm = struct.ob_mpi
        cdef MPI_Comm aero_comm = aero.ob_mpi
        cdef int *conn_data = NULL
        cdef int nelems = 0
        cdef int order = 2

        if conn is not None:
            conn_data = <int*>conn.data
            nelems = conn.shape[0]
            order = conn.shape[1]

        self.ptr = new BeamTransfer(c_comm, struct_comm, struct_root,
                                    aero_comm, aero_root,
                                    conn_data, nelems, order,
                                    dof_per_node)

        return