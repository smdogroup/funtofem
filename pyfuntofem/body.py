#!/usr/bin/env python
"""
This file is part of the package FUNtoFEM for coupled aeroelastic simulation
and design optimization.

Copyright (C) 2015 Georgia Tech Research Corporation.
Additional copyright (C) 2015 Kevin Jacobson, Jan Kiviaho and Graeme Kennedy.
All rights reserved.

FUNtoFEM is licensed under the Apache License, Version 2.0 (the "License");
you may not use this software except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
from .base import Base
from mpi4py import MPI
from funtofem import TransferScheme

try:
    from .hermes_transfer import HermesTransfer
except:
    pass


class Body(Base):
    """
    Defines a body base class for FUNtoFEM. Can be used as is or as a
    parent class for bodies with shape parameterization.
    """

    def __init__(
        self,
        name,
        analysis_type,
        id=0,
        group=None,
        boundary=0,
        fun3d=True,
        motion_type="deform",
    ):
        """

        Parameters
        ----------
        name: str
            name of the body
        id: int
            ID number of the body in the list of bodies in the model
        group: int
            group number for the body. Coupled variables defined in the body will be coupled with
            bodies in the same group
        boundary: int
            FUN3D boundary number associated with the body
        fun3d: bool
            whether or not you are using FUN3D. If true, the body class will auto-populate 'rigid_motion' required by FUN3D
        motion_type: str
            the type of motion the body is undergoing. Possible options: 'deform','rigid','deform+rigid','rigid+deform'

        See Also
        --------
        :mod:`base` : Body inherits from Base
        :mod:`massoud_body` : example of class inheriting Body and adding shape parameterization

        """

        from .variable import Variable as dv

        super(Body, self).__init__(name, id, group)

        self.name = name
        self.analysis_type = analysis_type
        self.id = id
        self.group = group
        self.group_root = False
        self.boundary = boundary
        self.motion_type = motion_type

        self.variables = {}
        self.derivatives = {}

        self.parent = None
        self.children = []

        if fun3d:
            self.add_variable("rigid_motion", dv("RotRate", active=False, id=1))
            self.add_variable("rigid_motion", dv("RotFreq", active=False, id=2))
            self.add_variable("rigid_motion", dv("RotAmpl", active=False, id=3))
            self.add_variable("rigid_motion", dv("RotOrgx", active=False, id=4))
            self.add_variable("rigid_motion", dv("RotOrgy", active=False, id=5))
            self.add_variable("rigid_motion", dv("RotOrgz", active=False, id=6))
            self.add_variable("rigid_motion", dv("RotVecx", active=False, id=7))
            self.add_variable("rigid_motion", dv("RotVecy", active=False, id=8))
            self.add_variable("rigid_motion", dv("RotVecz", active=False, id=9))
            self.add_variable("rigid_motion", dv("TrnRate", active=False, id=10))
            self.add_variable("rigid_motion", dv("TrnFreq", active=False, id=11))
            self.add_variable("rigid_motion", dv("TrnAmpl", active=False, id=12))
            self.add_variable("rigid_motion", dv("TrnVecx", active=False, id=13))
            self.add_variable("rigid_motion", dv("TrnVecy", active=False, id=14))
            self.add_variable("rigid_motion", dv("TrnVecz", active=False, id=15))

        # Set the data type to use
        self.dtype = TransferScheme.dtype

        # shape parameterization
        self.shape = None
        self.parameterization = 1

        # load and displacement transfer
        self.transfer = None

        # heat flux and temperature transfer
        self.thermal_transfer = None

        # Number of nodes
        self.struct_nnodes = {}
        self.aero_nnodes = {}

        # Number of degrees of freedom on the thermal structural side of the transfer
        self.therm_xfer_ndof = 1
        self.thermal_index = (
            3  # 0,1,2 for xyz and 3 for temp (mod 4) see tacs_interface.py
        )
        self.T_ref = 300.0  # reference temperature in Kelvin

        # Node locations u
        self.struct_X = None
        self.aero_X = None

        # ID number of nodes
        self.struct_id = None
        self.aero_id = None

        # Aitken acceleration settings
        self.theta_init = 0.125
        self.theta_therm_init = 0.125
        self.theta_min = 0.01
        self.theta_max = 1.0

        self.aitken_init = None
        self.aitken_vec = None
        self.aitken_therm_vec = None
        self.up_prev = None
        self.therm_up_prev = None

        # forward variables
        self.struct_disps = {}
        self.struct_loads = {}
        self.struct_temps = {}
        self.struct_heat_flux = {}
        self.struct_shape_term = {}

        self.rigid_transform = {}
        self.aero_disps = {}
        self.aero_loads = {}
        self.aero_shape_term = {}

        self.aero_temps = {}
        self.aero_heat_flux = {}

        return

    def initialize_struct_mesh(self, struct_X, struct_id=None):
        """
        Initialize the structural mesh on any processors that have an instance
        of the structural solver.

        This function only needs to be called from processors that own an
        instance of the structural solver. This function is called before
        the transfer scheme is initialized.

        Parameters
        ----------
        struct_X: np.ndarray
            The structural node locations
        struct_id:
            The nodal ids of each structural node
        """

        self.struct_X = np.array(struct_X).astype(self.dtype)
        self.struct_nnodes = len(struct_X) // 3
        self.struct_id = struct_id

        return

    def set_struct_nodes(self, struct_X):
        """
        Set the structural node locations without changing the number of nodes or structural connectivity

        Parameters
        ----------
        struct_X: np.ndarray
            The structural node locations
        """

        self.struct_X[:] = struct_X[:]
        return

    def initialize_aero_mesh(self, aero_X, aero_id=None):
        """
        Initialize the aerodynamic surface mesh on any processors that have an
        instance of the aerodynamic solver

        This function only needs to be called from processors that own an
        instance of the aerodynamic solver. This function is called before
        the transfer scheme is initialized.

        Parameters
        ----------
        aero_X: np.ndarray
            The aerodynamic node locations
        aero_id:
            The nodal ids of each aerodynamic node
        """

        self.aero_X = np.array(aero_X).astype(self.dtype)
        self.aero_nnodes = len(aero_X) // 3
        self.aero_id = aero_id

        return

    def set_aero_nodes(self, aero_X):
        """
        Set the aerodynamic node locations without changing the number of nodes or aerodynamic mesh connectivity

        Parameters
        ----------
        aero_X: np.ndarray
            The aerodynamic node locations
        """

        self.aero_X[:] = aero_X[:]
        return

    def initialize_transfer(
        self,
        comm,
        struct_comm,
        struct_root,
        aero_comm,
        aero_root,
        transfer_options=None,
    ):
        """
        Initialize the transfer scheme

        Parameters
        ----------
        comm: MPI.comm
            MPI communicator
        transfer_options: dictionary or list of dictionaries
            options for the load and displacement transfer scheme for the bodies
        """

        # If the user did not specify a transfer scheme default to MELD
        if transfer_options is None:
            transfer_options = {"scheme": "meld", "isym": -1, "beta": 0.5, "npts": 200}

        # Initialize the transfer and thermal transfer objects to None
        self.transfer = None
        self.thermal_transfer = None

        body_analysis_type = self.analysis_type
        if "analysis_type" in transfer_options:
            body_analysis_type = transfer_options["analysis_type"].lower()

        # Set up the transfer schemes based on the type of analysis set for this body
        if (
            body_analysis_type == "aeroelastic"
            or body_analysis_type == "aerothermoelastic"
        ):

            # Set up the load and displacement transfer schemes
            if transfer_options["scheme"].lower() == "hermes":
                self.transfer = HermesTransfer(
                    self.comm, self.struct_comm, self.aero_comm
                )

            elif transfer_options["scheme"].lower() == "rbf":
                basis = TransferScheme.PY_THIN_PLATE_SPLINE

                if "basis function" in transfer_options:
                    if (
                        transfer_options["basis function"].lower()
                        == "thin plate spline"
                    ):
                        basis = TransferScheme.PY_THIN_PLATE_SPLINE
                    elif transfer_options["basis function"].lower() == "gaussian":
                        basis = TransferScheme.PY_GAUSSIAN
                    elif transfer_options["basis function"].lower() == "multiquadric":
                        basis = TransferScheme.PY_MULTIQUADRIC
                    elif (
                        transfer_options["basis function"].lower()
                        == "inverse multiquadric"
                    ):
                        basis = TransferScheme.PY_INVERSE_MULTIQUADRIC
                    else:
                        print("Unknown RBF basis function for body number")
                        quit()

                self.transfer = TransferScheme.pyRBF(
                    comm, struct_comm, struct_root, aero_comm, aero_root, basis, 1
                )

            elif transfer_options["scheme"].lower() == "meld":
                # defaults
                isym = -1  # No symmetry
                beta = 0.5  # Decay factor
                num_nearest = 200  # Number of nearest neighbours

                if "isym" in transfer_options:
                    isym = transfer_options["isym"]
                if "beta" in transfer_options:
                    beta = transfer_options["beta"]
                if "npts" in transfer_options:
                    num_nearest = transfer_options["npts"]

                self.transfer = TransferScheme.pyMELD(
                    comm,
                    struct_comm,
                    struct_root,
                    aero_comm,
                    aero_root,
                    isym,
                    num_nearest,
                    beta,
                )

            elif transfer_options["scheme"].lower() == "linearized meld":
                # defaults
                isym = -1
                beta = 0.5
                num_nearest = 200

                if "isym" in transfer_options:
                    isym = transfer_options["isym"]
                if "beta" in transfer_options:
                    beta = transfer_options["beta"]
                if "npts" in transfer_options:
                    num_nearest = transfer_options["npts"]

                self.transfer = TransferScheme.pyLinearizedMELD(
                    comm,
                    struct_comm,
                    struct_root,
                    aero_comm,
                    aero_root,
                    isym,
                    num_nearest,
                    beta,
                )

            elif transfer_options["scheme"].lower() == "beam":
                conn = transfer_options["conn"]
                nelems = transfer_options["nelems"]
                order = transfer_options["order"]
                ndof = transfer_options["ndof"]

                self.xfer_ndof = ndof
                self.transfer = TransferScheme.pyBeamTransfer(
                    comm,
                    struct_comm,
                    struct_root,
                    aero_comm,
                    aero_root,
                    conn,
                    nelems,
                    order,
                    ndof,
                )
            else:
                print("Error: Unknown transfer scheme for body")
                quit()

        # Set up the transfer schemes based on the type of analysis set for this body
        if (
            body_analysis_type == "aerothermal"
            or body_analysis_type == "aerothermoelastic"
        ):
            # Set up the load and displacement transfer schemes

            if transfer_options["thermal_scheme"].lower() == "meld":
                # defaults
                isym = -1
                beta = 0.5
                num_nearest = 200

                if "isym" in transfer_options:
                    isym = transfer_options["isym"]
                if "beta" in transfer_options:
                    beta = transfer_options["beta"]
                if "npts" in transfer_options:
                    num_nearest = transfer_options["npts"]

                self.thermal_transfer = TransferScheme.pyMELDThermal(
                    comm,
                    struct_comm,
                    struct_root,
                    aero_comm,
                    aero_root,
                    isym,
                    num_nearest,
                    beta,
                )
            else:
                print("Error: Unknown thermal transfer scheme for body")
                quit()

        # Set the node locations
        self.update_transfer()

        # Initialize the load/displacement transfer
        if self.transfer is not None:
            self.transfer.initialize()

        # Initialize the thermal transfer
        if self.thermal_transfer is not None:
            self.thermal_transfer.initialize()

        return

    def update_transfer(self):
        """
        Update the positions of the nodes in transfer schemes
        """

        if self.transfer is not None:
            self.transfer.setStructNodes(self.struct_X)
            self.transfer.setAeroNodes(self.aero_X)

        if self.thermal_transfer is not None:
            self.thermal_transfer.setStructNodes(self.struct_X)
            self.thermal_transfer.setAeroNodes(self.aero_X)

        return

    def set_id(self, id):
        """
        **[model call]**
        Update the id number of the body or scenario

        Parameters
        ----------
        id: int
           id number of the scenario
        """
        self.id = id

        for vartype in self.variables:
            for var in self.variables[vartype]:
                var.body = self.id

    def add_variable(self, vartype, var):
        """
        Add a new variable to the body's variable dictionary.

        Parameters
        ----------
        vartype: str
            type of variable
        var: Variable object
            variable to be added
        """
        var.body = self.id

        super(Body, self).add_variable(vartype, var)

    def initialize_variables(self, scenario):
        """
        Initialie the variables each time we run an analysis.

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        """

        # We re-initialize aitken acceleration every time
        self.aitken_is_initialized = False

        if self.transfer is not None:
            ns = 3 * self.struct_nnodes
            na = 3 * self.aero_nnodes

            if scenario.steady:
                self.struct_loads[scenario.id] = np.zeros(ns, dtype=self.dtype)
                self.aero_loads[scenario.id] = np.zeros(na, dtype=self.dtype)
                self.struct_disps[scenario.id] = np.zeros(ns, dtype=self.dtype)
                self.aero_disps[scenario.id] = np.zeros(na, dtype=self.dtype)
            else:
                id = scenario.id
                self.struct_loads[id] = []
                self.aero_loads[id] = []
                self.struct_disps[id] = []
                self.aero_disps[id] = []

                for time_index in range(scenario.steps + 1):
                    self.struct_loads[id].append(np.zeros(ns, dtype=self.dtype))
                    self.aero_loads[id].append(np.zeros(na, dtype=self.dtype))
                    self.struct_disps[id].append(np.zeros(ns, dtype=self.dtype))
                    self.aero_disps[id].append(np.zeros(na, dtype=self.dtype))

        if self.thermal_transfer is not None:
            ns = self.struct_nnodes
            na = self.aero_nnodes

            if scenario.steady:
                self.struct_heat_flux[scenario.id] = np.zeros(ns, dtype=self.dtype)
                self.aero_heat_flux[scenario.id] = np.zeros(na, dtype=self.dtype)
                self.struct_temps[scenario.id] = np.zeros(ns, dtype=self.dtype)
                self.aero_temps[scenario.id] = np.zeros(na, dtype=self.dtype)
            else:
                id = scenario.id
                self.struct_heat_flux[id] = []
                self.aero_heat_flux[id] = []
                self.struct_temps[id] = []
                self.aero_temps[id] = []

                for time_index in range(scenario.steps + 1):
                    self.struct_heat_flux[id].append(np.zeros(ns, dtype=self.dtype))
                    self.aero_heat_flux[id].append(np.zeros(na, dtype=self.dtype))
                    self.struct_temps[id].append(np.zeros(ns, dtype=self.dtype))
                    self.aero_temps[id].append(np.zeros(na, dtype=self.dtype))

        return

    def initialize_adjoint_variables(self, scenario):
        """
        Initialize the adjoint variables for the body.

        The adjoint variables in the body are not indexed by scenario.
        The variables are initialized once for each scenario and used until the
        adjoint computation is completed. For each new scenario a new set of
        adjoint variables are initialized. You cannot solve multiple adjoints for different
        scenarios at the same time.

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        """

        # Count up the number of functions for this scenario
        nfunctions = scenario.count_adjoint_functions()

        # Allocate the adjoint variables and internal body variables required
        ns = 3 * self.struct_nnodes
        na = 3 * self.aero_nnodes
        self.aero_shape_term = np.zeros((na, nfunctions), dtype=self.dtype)
        self.struct_shape_term = np.zeros((ns, nfunctions), dtype=self.dtype)

        if self.transfer is not None:
            ns = 3 * self.struct_nnodes
            na = 3 * self.aero_nnodes

            # Load transfer adjoint and right-hand-side
            self.psi_L = np.zeros((ns, nfunctions), dtype=self.dtype)
            self.adjL_rhs = np.zeros((ns, nfunctions), dtype=self.dtype)

            # Aero forces adjoint right-hand-side
            self.psi_F = np.zeros((na, nfunctions), dtype=self.dtype)
            self.adjF_rhs = np.zeros((na, nfunctions), dtype=self.dtype)

            # Structural adjoint right-hand-side
            self.adjS_rhs = np.zeros((ns, nfunctions), dtype=self.dtype)

            # Displacement transfer adjoint and right-hand-side
            self.psi_D = np.zeros((na, nfunctions), dtype=self.dtype)
            self.adjD_rhs = np.zeros((na, nfunctions), dtype=self.dtype)

        if self.thermal_transfer is not None:
            ns = self.struct_nnodes
            na = self.aero_nnodes

            psi_H = np.zeros((na, nfunctions), dtype=self.dtype)

        # if body.transfer is not None:
        #     body.psi_L = np.zeros(
        #         (body.struct_nnodes * body.xfer_ndof, nfunctions),
        #         dtype=TransferScheme.dtype,
        #     )
        #     body.psi_S = np.zeros(
        #         (body.struct_nnodes * body.xfer_ndof, nfunctions),
        #         dtype=TransferScheme.dtype,
        #     )
        #     body.struct_rhs = np.zeros(
        #         (body.struct_nnodes * body.xfer_ndof, nfunctions),
        #         dtype=TransferScheme.dtype,
        #     )

        #     body.dLdfa = np.zeros(
        #         (body.aero_nnodes * 3, nfunctions), dtype=TransferScheme.dtype
        #     )
        #     body.dGdua = np.zeros(
        #         (body.aero_nnodes * 3, nfunctions), dtype=TransferScheme.dtype
        #     )
        #     body.psi_D = np.zeros(
        #         (body.aero_nnodes * 3, nfunctions), dtype=TransferScheme.dtype
        #     )

        # if body.thermal_transfer is not None:
        #     # Thermal terms
        #     body.psi_Q = np.zeros(
        #         (body.struct_nnodes * body.therm_xfer_ndof, nfunctions),
        #         dtype=TransferScheme.dtype,
        #     )
        #     body.psi_T_S = np.zeros(
        #         (body.struct_nnodes * body.therm_xfer_ndof, nfunctions),
        #         dtype=TransferScheme.dtype,
        #     )
        #     body.struct_rhs_T = np.zeros(
        #         (body.struct_nnodes * body.therm_xfer_ndof, nfunctions),
        #         dtype=TransferScheme.dtype,
        #     )

        #     body.dQdfta = np.zeros(
        #         (body.aero_nnodes, nfunctions), dtype=TransferScheme.dtype
        #     )
        #     body.dQfluxdfta = np.zeros(
        #         (body.aero_nnodes * 3, nfunctions), dtype=TransferScheme.dtype
        #     )
        #     body.dAdta = np.zeros(
        #         (body.aero_nnodes, nfunctions), dtype=TransferScheme.dtype
        #     )
        #     body.psi_T = np.zeros(
        #         (body.aero_nnodes, nfunctions), dtype=TransferScheme.dtype
        #     )

        return

    def get_aero_disps(self, scenario, time_index=0):
        """
        Get the displacements on the aerodynamic surface for the given scenario.

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        time_index: int
            The time-index for time-dependent problems
        """
        if self.transfer is not None:
            if scenario.steady:
                return self.aero_disps[scenario.id]
            else:
                return self.aero_disps[scenario.id][time_index]
        else:
            return None

    def get_struct_disps(self, scenario, time_index=0):
        """
        Get the displacements on the structure for the given scenario.

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        time_index: int
            The time-index for time-dependent problems
        """
        if self.transfer is not None:
            if scenario.steady:
                return self.struct_disps[scenario.id]
            else:
                return self.struct_disps[scenario.id][time_index]
        else:
            return None

    def get_aero_loads(self, scenario, time_index=0):
        """
        Get the loads on the aerodynamic surface for the given scenario.

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        time_index: int
            The time-index for time-dependent problems
        """
        if self.transfer is not None:
            if scenario.steady:
                return self.aero_loads[scenario.id]
            else:
                return self.aero_loads[scenario.id][time_index]
        else:
            return None

    def get_struct_loads(self, scenario, time_index=0):
        """
        Get the loads on the structure for the given scenario.

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        time_index: int
            The time-index for time-dependent problems
        """
        if self.transfer is not None:
            if scenario.steady:
                return self.struct_loads[scenario.id]
            else:
                return self.struct_loads[scenario.id][time_index]
        else:
            return None

    def get_aero_temps(self, scenario, time_index=0):
        """
        Get the temperatures on the aerodynamic surface for the given scenario.

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        time_index: int
            The time-index for time-dependent problems
        """
        if self.thermal_transfer is not None:
            if scenario.steady:
                return self.aero_temps[scenario.id]
            else:
                return self.aero_temps[scenario.id][time_index]
        else:
            return None

    def get_aero_heat_flux(self, scenario, time_index=0):
        """
        Get the heat flux on the aerodynamic surface for the given scenario.

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        time_index: int
            The time-index for time-dependent problems
        """
        if self.thermal_transfer is not None:
            if scenario.steady:
                return self.aero_heat_flux[scenario.id]
            else:
                return self.aero_heat_flux[scenario.id][time_index]
        else:
            return None

    def get_struct_temps(self, scenario, time_index=0):
        """
        Get the temperatures on the structure for the given scenario.

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        time_index: int
            The time-index for time-dependent problems
        """
        if self.thermal_transfer is not None:
            if scenario.steady:
                return self.struct_temps[scenario.id]
            else:
                return self.struct_temps[scenario.id][time_index]
        else:
            return None

    def get_struct_heat_flux(self, scenario, time_index=0):
        """
        Get the heat flux on the structure for the given scenario.

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        time_index: int
            The time-index for time-dependent problems
        """
        if self.thermal_transfer is not None:
            if scenario.steady:
                return self.struct_heat_flux[scenario.id]
            else:
                return self.struct_heat_flux[scenario.id][time_index]
        else:
            return None

    def transfer_disps(self, scenario, time_index=0):
        """
        Transfer the displacements on the structural mesh to the aerodynamic mesh
        for the given scenario.

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        time_index: int
            The time-index for time-dependent problems
        """
        if self.transfer is not None:
            if scenario.steady:
                aero_disps = self.aero_disps[scenario.id]
                struct_disps = self.struct_disps[scenario.id]
            else:
                aero_disps = self.aero_disps[scenario.id][time_index]
                struct_disps = self.struct_disps[scenario.id][time_index]
            self.transfer.transferDisps(struct_disps, aero_disps)

        return

    def transfer_loads(self, scenario, time_index=0):
        """
        Transfer the aerodynamic loads on the aero surface mesh to loads on the
        structural mesh for the given scenario.

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        time_index: int
            The time-index for time-dependent problems
        """
        if self.transfer is not None:
            if scenario.steady:
                aero_loads = self.aero_loads[scenario.id]
                struct_loads = self.struct_loads[scenario.id]
            else:
                aero_loads = self.aero_loads[scenario.id][time_index]
                struct_loads = self.struct_loads[scenario.id][time_index]
            self.transfer.transferLoads(aero_loads, struct_loads)

        return

    def transfer_temps(self, scenario, time_index=0):
        """
        Transfer the temperatures on the structural mesh to the aerodynamic mesh
        for the given scenario.

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        time_index: int
            The time-index for time-dependent problems
        """
        if self.thermal_transfer is not None:
            if scenario.steady:
                struct_temps = self.struct_temps[scenario.id]
                aero_temps = self.aero_temps[scenario.id]
            else:
                struct_temps = self.struct_temps[scenario.id][time_index]
                aero_temps = self.aero_temps[scenario.id][time_index]
            self.thermal_transfer.transferTemp(struct_temps, aero_temps)

    def transfer_heat_flux(self, scenario, time_index=0):
        """
        Transfer the aerodynamic heat flux on the aero surface mesh to the heat flux on the
        structural mesh for the given scenario.

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        time_index: int
            The time-index for time-dependent problems
        """
        if self.thermal_transfer is not None:
            if scenario.steady:
                aero_flux = self.aero_heat_flux[scenario.id]
                struct_flux = self.struct_heat_flux[scenario.id]
            else:
                aero_flux = self.aero_heat_flux[scenario.id][time_index]
                struct_flux = self.struct_heat_flux[scenario.id][time_index]
            self.thermal_transfer.transferFlux(aero_flux, struct_flux)

        return

    def get_aero_loads_adjoint(self, scenario):
        """
        Get the aerodynamic load adjoint psi_F

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        """

        if self.transfer:
            return self.psi_F

        return None

    def get_aero_coordinate_sensitivity(self, scenario):
        if self.transfer:
            return self.adjD_rhs

        return None

    def get_aero_temp_sensitivity(self, scenario):
        if self.thermal_transfer is not None:
            return something

        return None

    def get_disp_transfer_adjoint_rhs(self, scenario):

        if self.transfer:
            return self.adjD_rhs

        return None

    def get_load_adjoint_rhs(self, scenario):

        if self.transfer:
            return self.adjL_rhs

        return None

    def get_struct_adjoint_rhs(self, scenario):

        if self.transfer:
            return self.adjS_rhs

        return None

    def get_struct_flux_adjoint_rhs(self, scenario):
        return None

    def transfer_loads_adjoint(self, scenario, time_index=0):
        nfunctions = scenario.count_adjoint_functions()

        if self.transfer is not None:
            # Solve for psi_L - Note that dL/dfs is the identity matrix
            self.psi_L[:] = self.adjL_rhs[:]
            self.adjL_rhs[:] = 0.0

            # Contribute to the force integration and structural adjoint right-hand-sides
            # from the load transfer adjoint
            temp_fa = np.zeros(3 * self.aero_nnodes, dtype=self.dtype)
            temp_us = np.zeros(3 * self.struct_nnodes, dtype=self.dtype)
            for k in range(nfunctions):
                # Copy the values into contiguous memory
                psi_Lk = self.psi_L[:, k].copy()

                # Contribute adjF_rhs -= dL/dfa^{T} * psi_L
                self.transfer.applydLdfATrans(psi_Lk, temp_fa)
                self.adjF_rhs[:, k] -= temp_fa

                # Contribute adjS_rhs -= dL/dus^{T} * psi_L
                self.transfer.applydLduSTrans(psi_Lk, temp_us)
                self.adjS_rhs[:, k] -= temp_us

            # Solve for the aerodynamic force adjoint
            self.psi_F[:] = self.adjF_rhs[:]

        return

    def transfer_disps_adjoint(self, scenario, step):
        nfunctions = scenario.count_adjoint_functions()

        if self.transfer is not None:
            # Solve for psi_D - Note that dD/dua is the identity matrix
            self.psi_D[:] = self.adjD_rhs[:]
            self.adjD_rhs[:] = 0.0

            temp_us = np.zeros(3 * self.struct_nnodes, dtype=self.dtype)
            for k in range(nfunctions):
                # Copy the values into contiguous memory
                psi_Dk = self.psi_D[:, k].copy()

                # Contribute adjS_rhs -= dD/dus^{T} * psi_D
                self.transfer.applydDduSTrans(psi_Dk, temp_us)
                self.adjS_rhs[:, k] -= temp_us

        return

    def transfer_heat_flux_adjoint(self, scenario, step):
        nfunctions = scenario.count_adjoint_functions()

        #        if self.thermal_transfer is not None:

        # for func in range(nfunctions):
        #     if body.thermal_transfer is not None:
        #         # Transform heat flux transfer adjoint variables using transpose Jacobian from
        #         # funtofem: dQdftA^T * psi_Q = dTdts * psi_Q
        #         psi_Q_r = np.zeros(body.aero_nnodes, dtype=TransferScheme.dtype)
        #         body.thermal_transfer.applydQdqATrans(
        #             body.psi_T_S[:, func].copy(order="C"), psi_Q_r
        #         )

        #         # Only set heat flux magnitude component of thermal adjoint in FUN3D
        #         # Can either use surface normal magnitude OR x,y,z components, not both
        #         body.dQdfta[:, func] = psi_Q_r

        return

    def transfer_temps_adjoint(self, scenario, step):

        # for func in range(nfunctions):
        #     if body.thermal_transfer is not None:
        #         # calculate dTdt_s^T * psi_T
        #         psi_T_product = np.zeros(
        #             body.struct_nnodes * body.therm_xfer_ndof,
        #             dtype=TransferScheme.dtype,
        #         )
        #         body.psi_T = body.dAdta
        #         body.thermal_transfer.applydTdtSTrans(
        #             body.psi_T[:, func].copy(order="C"), psi_T_product
        #         )
        #         body.struct_rhs_T[:, func] = psi_T_product
        return

    def add_aero_coordinate_derivative(self, scenario, step, dfdx):
        self.aero_shape_term[:] += dfdx[:]

    def add_struct_coordinate_derivative(self, scenario, step, dfdx):
        self.struct_shape_term[:] += dfdx[:]

    def add_coordinate_derivative(self, scenario, step):
        """
        Add the coordinate derivatives for each function of interest to the aerodynamic
        and structural surface nodes - stored in aero_shape_term and struct_shape_term, respectively.
        """

        if self.transfer is not None:
            nfunctions = scenario.count_adjoint_functions()

            # Aerodynamic coordinate derivatives
            temp_xa = np.zeros(3 * self.aero_nnodes, dtype=self.dtype)
            temp_xs = np.zeros(3 * self.struct_nnodes, dtype=self.dtype)
            for k in range(nfunctions):
                # Load transfer terms
                psi_Lk = self.psi_L[:, k].copy()
                self.transfer.applydLdxA0(psi_Lk, temp_xa)
                self.aero_shape_term[:, k] += temp_xa

                self.transfer.applydLdxS0(psi_Lk, temp_xs)
                self.struct_shape_term[:, k] += temp_xs

                # Displacement transfer terms
                psi_Dk = self.psi_D[:, k].copy()
                self.transfer.applydDdxA0(psi_Dk, temp_xa)
                self.aero_shape_term[:, k] += temp_xa

                self.transfer.applydDdxS0(psi_Dk, temp_xs)
                self.struct_shape_term[:, k] += temp_xs

        return

    def aitken_relax(self, scenario, tol=1e-13):
        """
        Perform Aitken relaxation for the displacements set in the
        """

        if not self.aitken_is_initialized:
            self.theta = self.theta_init
            self.prev_update = np.zeros(3 * self.struct_nnodes, dtype=self.dtype)
            self.aitken_vec = np.zeros(3 * self.struct_nnodes, dtype=self.dtype)
            self.aitken_is_initialized = True

        if self.transfer is not None:
            struct_disps = self.get_struct_disps(scenario)
            up = struct_disps - self.aitken_vec
            norm2 = np.linalg.norm(up - self.prev_update) ** 2.0

            # Only update theta if the displacements changed
            if norm2 > tol:
                # Compute the tentative theta value
                self.theta *= 1.0 - (up - self.prev_update).dot(up) / norm2

                self.theta = np.max(
                    (np.min((self.theta, self.theta_max)), self.theta_min)
                )

            # handle the min/max for complex step
            if type(self.theta) == np.complex128 or type(self.theta) == complex:
                self.theta = self.theta.real + 0.0j

            self.aitken_vec += self.theta * up
            self.prev_update[:] = up[:]
            struct_disps[:] = self.aitken_vec

        return

    def aitken_adjoint_relax(self, scenario, tol=1e-13):
        return

    def collect_coordinate_derivatives(self, comm, discipline, root=0):
        """
        Write the sensitivity files for the aerodynamic and structural meshes on
        the root processor.

        This code collects the sensitivities from each processor and collects the
        result to the root node.
        """

        if discipline == "aero":
            all_aero_ids = comm.gather(self.aero_id, root=root)
            all_aero_shape = comm.gather(self.aero_shape_term, root=root)

            aero_ids = []
            aero_shape = []

            if comm.rank == root:
                # Discard any entries that are None
                aero_ids = []
                for d in all_aero_ids:
                    if d is not None:
                        aero_ids.append(d)

                aero_shape = []
                for d in all_aero_shape:
                    if d is not None:
                        aero_shape.append(d)

                if len(aero_shape) > 0:
                    aero_shape = np.concatenate(aero_shape)
                else:
                    aero_shape = np.zeros((3, 1))

                if len(aero_ids) == 0:
                    aero_ids = np.arange(aero_shape.shape[0] // 3, dtype=int)
                else:
                    aero_ids = np.concatenate(aero_ids)

            return aero_ids, aero_shape

        elif discipline == "struct":
            all_struct_ids = comm.gather(self.struct_id, root=root)
            all_struct_shape = comm.gather(self.struct_shape_term, root=root)

            struct_ids = []
            struct_shape = []

            if comm.rank == root:
                # Discard any entries that are None
                struct_ids = []
                for d in all_struct_ids:
                    if d is not None:
                        struct_ids.append(d)

                struct_shape = []
                for d in all_struct_shape:
                    if d is not None:
                        struct_shape.append(d)

                if len(struct_shape) > 0:
                    struct_shape = np.concatenate(struct_shape)
                else:
                    struct_shape = np.zeros((3, 1))

                if len(struct_ids) == 0:
                    struct_ids = np.arange(struct_shape.shape[0] // 3, dtype=int)
                else:
                    struct_ids = np.concatenate(struct_ids)

            return struct_ids, struct_shape

        return

    def initialize_shape_parameterization(self):
        """
        **[driver call]**
        Create the map of structural node locations since TACS doesn't give us global id numbers

        """

        return

    def update_shape(self, complex_run=False):
        """
        **[driver call]**
        perturbs the shape of the body based on the body's shape variables and update the meshes

        Parameters
        ----------
        complex_run: bool
            whether or not the run is complex mode

        """

        return

    def shape_derivative(self, scenario, offset):
        """
        **[driver call]**
        Calculates the shape derivative given the coordinate derivatives

        Parameters
        ----------
        scenario: scenario object
            The current scenario
        offset: int
            function offset number
        """

        return

    # def _aitken_relax(self):
    #     """
    #     Solves the aitken relaxation

    #     """

    #     if self.aitken_init:
    #         self.aitken_init = False

    #         # initialize the 'previous update' to zero
    #         self.up_prev = []
    #         self.aitken_vec = []
    #         self.theta = []

    #         self.therm_up_prev = []
    #         self.aitken_therm_vec = []
    #         self.theta = []

    #         for ind, body in enumerate(self.model.bodies):
    #             if body.transfer is not None:
    #                 self.up_prev.append(
    #                     np.zeros(
    #                         body.struct_nnodes * body.xfer_ndof,
    #                         dtype=TransferScheme.dtype,
    #                     )
    #                 )
    #                 self.aitken_vec.append(
    #                     np.zeros(
    #                         body.struct_nnodes * body.xfer_ndof,
    #                         dtype=TransferScheme.dtype,
    #                     )
    #                 )
    #                 self.theta.append(self.theta_init)

    #             if body.thermal_transfer is not None:
    #                 self.therm_up_prev.append(
    #                     np.zeros(
    #                         body.struct_nnodes * body.therm_xfer_ndof,
    #                         dtype=TransferScheme.dtype,
    #                     )
    #                 )
    #                 self.aitken_therm_vec.append(
    #                     np.zeros(
    #                         body.struct_nnodes * body.therm_xfer_ndof,
    #                         dtype=TransferScheme.dtype,
    #                     )
    #                 )
    #                 self.theta.append(self.theta_init)

    #     # do the Aitken update
    #     for ibody, body in enumerate(self.model.bodies):

    #         if body.transfer is not None:
    #             if body.struct_nnodes > 0:
    #                 up = body.struct_disps - self.aitken_vec[ibody]
    #                 norm2 = np.linalg.norm(up - self.up_prev[ibody]) ** 2.0

    #                 # Only update theta if the displacements changed
    #                 if norm2 > 1e-13:
    #                     self.theta[ibody] *= (
    #                         1.0 - (up - self.up_prev[ibody]).dot(up) / norm2
    #                     )
    #                     self.theta[ibody] = np.max(
    #                         (
    #                             np.min((self.theta[ibody], self.theta_max)),
    #                             self.theta_min,
    #                         )
    #                     )

    #                 # handle the min/max for complex step
    #                 if (
    #                     type(self.theta[ibody]) == np.complex128
    #                     or type(self.theta[ibody]) == complex
    #                 ):
    #                     self.theta[ibody] = self.theta[ibody].real + 0.0j

    #                 self.aitken_vec[ibody] += self.theta[ibody] * up
    #                 self.up_prev[ibody] = up[:]
    #                 body.struct_disps = self.aitken_vec[ibody]

    #         if body.thermal_transfer is not None:
    #             if body.struct_nnodes > 0:
    #                 up = body.struct_temps - self.aitken_therm_vec[ibody]
    #                 norm2 = np.linalg.norm(up - self.therm_up_prev[ibody]) ** 2.0

    #                 # Only update theta if the displacements changed
    #                 if norm2 > 1e-13:
    #                     self.theta[ibody] *= (
    #                         1.0 - (up - self.therm_up_prev[ibody]).dot(up) / norm2
    #                     )
    #                     self.theta[ibody] = np.max(
    #                         (
    #                             np.min((self.theta[ibody], self.theta_max)),
    #                             self.theta_min,
    #                         )
    #                     )

    #                 # handle the min/max for complex step
    #                 if (
    #                     type(self.theta[ibody]) == np.complex128
    #                     or type(self.theta[ibody]) == complex
    #                 ):
    #                     self.theta[ibody] = self.theta[ibody].real + 0.0j

    #                 self.aitken_therm_vec[ibody] += self.theta[ibody] * up
    #                 self.therm_up_prev[ibody] = up[:]
    #                 body.struct_temps = self.aitken_therm_vec[ibody]

    #     return

    # def _aitken_adjoint_relax(self, scenario):
    #     nfunctions = scenario.count_adjoint_functions()
    #     if self.aitken_init:
    #         self.aitken_init = False

    #         # initialize the 'previous update' to zero
    #         self.up_prev = []
    #         self.aitken_vec = []
    #         self.theta = []

    #         # initialize the 'previous update' to zero
    #         self.therm_up_prev = []
    #         self.aitken_therm_vec = []
    #         self.theta_therm = []

    #         for ibody, body in enumerate(self.model.bodies):
    #             if body.transfer is not None:
    #                 up_prev_body = []
    #                 aitken_vec_body = []
    #                 theta_body = []
    #                 for func in range(nfunctions):
    #                     up_prev_body.append(
    #                         np.zeros(
    #                             body.struct_nnodes * body.xfer_ndof,
    #                             dtype=TransferScheme.dtype,
    #                         )
    #                     )
    #                     aitken_vec_body.append(
    #                         np.zeros(
    #                             body.struct_nnodes * body.xfer_ndof,
    #                             dtype=TransferScheme.dtype,
    #                         )
    #                     )
    #                     theta_body.append(self.theta_init)
    #                 self.up_prev.append(up_prev_body)
    #                 self.aitken_vec.append(aitken_vec_body)
    #                 self.theta.append(theta_body)

    #             if body.thermal_transfer is not None:
    #                 up_prev_body = []
    #                 aitken_therm_vec_body = []
    #                 theta_body = []
    #                 for func in range(nfunctions):
    #                     up_prev_body.append(
    #                         body.T_ref
    #                         * np.ones(
    #                             body.struct_nnodes * body.therm_xfer_ndof,
    #                             dtype=TransferScheme.dtype,
    #                         )
    #                     )
    #                     aitken_therm_vec_body.append(
    #                         body.T_ref
    #                         * np.ones(
    #                             body.struct_nnodes * body.therm_xfer_ndof,
    #                             dtype=TransferScheme.dtype,
    #                         )
    #                     )
    #                     theta_body.append(self.theta_therm_init)
    #                 self.therm_up_prev.append(up_prev_body)
    #                 self.aitken_therm_vec.append(aitken_therm_vec_body)
    #                 self.theta_therm.append(theta_body)

    #     # do the Aitken update
    #     for ibody, body in enumerate(self.model.bodies):
    #         if body.struct_nnodes > 0:
    #             if body.transfer is not None:
    #                 for func in range(nfunctions):
    #                     up = body.psi_S[:, func] - self.aitken_vec[ibody][func]
    #                     norm2 = np.linalg.norm(up - self.up_prev[ibody][func]) ** 2.0

    #                     # Only update theta if the vector changed
    #                     if norm2 > 1e-13:
    #                         self.theta[ibody][func] *= (
    #                             1.0
    #                             - (up - self.up_prev[ibody][func]).dot(up)
    #                             / np.linalg.norm(up - self.up_prev[ibody][func]) ** 2.0
    #                         )
    #                         self.theta[ibody][func] = np.max(
    #                             (
    #                                 np.min((self.theta[ibody][func], self.theta_max)),
    #                                 self.theta_min,
    #                             )
    #                         )
    #                     self.aitken_vec[ibody][func] += self.theta[ibody][func] * up
    #                     self.up_prev[ibody][func] = up[:]
    #                     body.psi_S[:, func] = self.aitken_vec[ibody][func][:]

    #             if body.thermal_transfer is not None:
    #                 for func in range(nfunctions):
    #                     up = body.psi_T_S[:, func] - self.aitken_therm_vec[ibody][func]
    #                     norm2 = (
    #                         np.linalg.norm(up - self.therm_up_prev[ibody][func]) ** 2.0
    #                     )

    #                     # Only update theta if the vector changed
    #                     if norm2 > 1e-13:
    #                         self.theta_therm[ibody][func] *= (
    #                             1.0
    #                             - (up - self.therm_up_prev[ibody][func]).dot(up)
    #                             / np.linalg.norm(up - self.therm_up_prev[ibody][func])
    #                             ** 2.0
    #                         )
    #                         self.theta_therm[ibody][func] = np.max(
    #                             (
    #                                 np.min(
    #                                     (self.theta_therm[ibody][func], self.theta_max)
    #                                 ),
    #                                 self.theta_min,
    #                             )
    #                         )
    #                     self.aitken_therm_vec[ibody][func] += (
    #                         self.theta_therm[ibody][func] * up
    #                     )
    #                     self.therm_up_prev[ibody][func] = up[:]
    #                     body.psi_T_S[:, func] = self.aitken_therm_vec[ibody][func][:]

    #     return
