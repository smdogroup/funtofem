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
from ._base import Base
from mpi4py import MPI
from funtofem import TransferScheme
from ..driver.transfer_settings import TransferSettings

try:
    from .hermes_transfer import HermesTransfer
except:
    pass


class AitkenRelaxation:
    """
    Class to define aitken relaxation settings
    """

    def __init__(
        self,
        theta_init=0.125,
        theta_therm_init=0.125,
        theta_min=0.01,
        theta_max=1.0,
    ):
        """
        Construct an aitken relaxation setting object
        Parmeters
        ----------
        theta_init : float
            initial learning rate for displacement update
        theta_therm_init : float
            initial learning rate for temperature update
        theta_min : float
            minimum learning rate
        theta_max : float
            maximum learning rate
        """
        self.theta_init = theta_init
        self.theta_therm_init = theta_therm_init
        self.theta_min = theta_min
        self.theta_max = theta_max


class SimpleRelaxation:
    """
    Class to define simple exponential decay relaxation scheme
    """

    def __init__(
        self,
        theta_init: float = 1.0,
        theta_therm_init: float = 1.0,
        tenth_steps: int = 50,
        min_learning_rate: float = 1.0e-4,
        min_learning_rate_t: float = 1.0e-4,
    ):
        self.theta = theta_init
        self.theta_t = theta_therm_init
        self.decay_rate = np.power(0.1, 1.0 / tenth_steps)
        self.min_learning_rate = min_learning_rate
        self.min_learning_rate_t = min_learning_rate_t

    def relax_displacement(self):
        """
        Perform the update to the learning rate for displacement transfer
        """
        self.theta *= self.decay_rate
        self.theta = np.max((self.theta, self.min_learning_rate))

    def relax_thermal(self):
        """
        Perform the update to the learning rate for thermal transfer
        """
        self.theta_t *= self.decay_rate
        self.theta_t = np.max((self.theta_t, self.min_learning_rate_t))


class Body(Base):
    """
    Defines a body base class for FUNtoFEM. Can be used as is or as a
    parent class for bodies with shape parameterization.
    """

    ANALYSIS_TYPES = ["aerothermal", "aeroelastic", "aerothermoelastic"]

    def __init__(
        self,
        name,
        analysis_type,
        id=0,
        group=None,
        boundary=0,
        fun3d=True,
        motion_type="deform",
        relaxation_scheme=None,
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
        use_aitken_accel: bool
            whether or not to use Aitken acceleration. Defaults to true.

        See Also
        --------
        :mod:`base` : Body inherits from Base
        :mod:`massoud_body` : example of class inheriting Body and adding shape parameterization

        """

        from .variable import Variable as dv

        assert analysis_type in Body.ANALYSIS_TYPES

        super(Body, self).__init__(name, id, group)

        self.name = name
        self.analysis_type = analysis_type
        self.id = id
        self.group = group
        self.group_root = False
        self.boundary = boundary
        self.motion_type = motion_type

        self.variables = {}

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

        self.rigid_transform = np.identity(4, dtype=TransferScheme.dtype)

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
        self.struct_nnodes = 0
        self.aero_nnodes = 0

        # Number of degrees of freedom on the thermal structural side of the transfer
        self.therm_xfer_ndof = 1
        self.thermal_index = (
            3  # 0,1,2 for xyz and 3 for temp (mod 4) see tacs_interface.py
        )

        # Node locations u
        self.struct_X = None
        self.aero_X = None

        # ID number of nodes
        self.struct_id = None
        self.aero_id = None

        # relaxation scheme object
        self.relaxation_scheme = relaxation_scheme
        self.use_aitken_accel = isinstance(relaxation_scheme, AitkenRelaxation)
        self.use_simple_accel = isinstance(relaxation_scheme, SimpleRelaxation)

        # Aitken acceleration settings
        self.theta_init = 0.125
        self.theta_therm_init = 0.125
        self.theta_min = 0.01
        self.theta_max = 1.0

        self.aitken_init = False
        self._aero_X_orig = None

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

    @classmethod
    def aeroelastic(
        cls,
        name: str,
        boundary: int = 0,
        use_fun3d: bool = True,
        motion_type: str = "deform",
        relaxation_scheme=None,
    ):
        """
        Class method to create a body object for aeroelastic coupling.
        Recommendation: set name, boundary here and use method cascades for motion_type, relaxation_scheme
        """
        return cls(
            name=name,
            analysis_type="aeroelastic",
            boundary=boundary,
            fun3d=use_fun3d,
            motion_type=motion_type,
            relaxation_scheme=relaxation_scheme,
        )

    @classmethod
    def aerothermal(
        cls,
        name: str,
        boundary: int = 0,
        use_fun3d: bool = True,
        motion_type: str = "deform",
        relaxation_scheme=None,
    ):
        """
        Class method to create a body object for aerothermal coupling.
        Recommendation: set name, boundary here and use method cascades for motion_type, relaxation_scheme
        """
        return cls(
            name=name,
            analysis_type="aerothermal",
            boundary=boundary,
            fun3d=use_fun3d,
            motion_type=motion_type,
            relaxation_scheme=relaxation_scheme,
        )

    @classmethod
    def aerothermoelastic(
        cls,
        name: str,
        boundary: int = 0,
        use_fun3d: bool = True,
        motion_type: str = "deform",
        relaxation_scheme=None,
    ):
        """
        Class method to create a body object for aerothermoelastic coupling.
        Recommendation: set name, boundary here and use method cascades for motion_type, relaxation_scheme
        """
        return cls(
            name=name,
            analysis_type="aerothermoelastic",
            boundary=boundary,
            fun3d=use_fun3d,
            motion_type=motion_type,
            relaxation_scheme=relaxation_scheme,
        )

    def motion_type(self, new_motion_type):
        """
        set the motion type in a method cascade
        """
        self.motion_type = new_motion_type
        return self

    def relaxation(self, new_relaxation_scheme):
        """
        set the relaxation scheme in a method cascade
        """
        self.relaxation_scheme = new_relaxation_scheme
        self.use_aitken_accel = isinstance(new_relaxation_scheme, AitkenRelaxation)
        self.use_simple_accel = isinstance(new_relaxation_scheme, SimpleRelaxation)
        return self

    def register_to(self, funtofem_model):
        """
        add this body to the funtofem model in a method cascade
        """
        funtofem_model.add_body(self)
        return self

    def initialize_struct_nodes(self, struct_X, struct_id=None):
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

    def get_struct_nodes(self):
        """
        Get the structural node locations associated with this body
        """

        return self.struct_X

    def get_num_struct_nodes(self):
        """
        Get the number of aerodynamic surface nodes
        """

        return self.struct_nnodes

    def get_struct_node_ids(self):
        """
        Get the structural node ids
        """

        return self.struct_id

    def initialize_aero_nodes(self, aero_X, aero_id=None):
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

        self._aero_X_orig = self.aero_X.copy()

        return

    def get_aero_nodes(self):
        """
        Get the aerodynamic node locations associated with this body
        """

        return self.aero_X

    def get_num_aero_nodes(self):
        """
        Get the number of aerodynamic surface nodes
        """

        return self.aero_nnodes

    def get_aero_node_ids(self):
        """
        Get the aerodynamic node ids
        """

        return self.aero_id

    def verify_analysis_type(self, analysis_type):
        """ "
        Input verification for analysis type when initializing a body.

        Parameters
        ----------
        analysis_type: str
            type of analysis
        """

        if not analysis_type in Body.ANALYSIS_TYPES:
            raise ValueError("analysis_type specified is not recognized as valid")

        return

    def initialize_transfer(
        self,
        comm,
        struct_comm,
        struct_root,
        aero_comm,
        aero_root,
        transfer_settings=None,
    ):
        """
        Initialize the load and displacement and/or thermal transfer scheme for this body

        Parameters
        ----------
        comm: MPI.comm
            MPI communicator
        transfer_settings: TransferSettings
            options for the load and displacement transfer scheme for the bodies
        """

        # If the user did not specify a transfer scheme default to MELD
        if transfer_settings is None:
            transfer_settings = TransferSettings()

        # Initialize the transfer and thermal transfer objects to None
        self.transfer = None
        self.thermal_transfer = None

        # Verify analysis type is valid
        self.verify_analysis_type(self.analysis_type)

        elastic_analyses = [_ for _ in Body.ANALYSIS_TYPES if "elastic" in _]
        thermal_analyses = [_ for _ in Body.ANALYSIS_TYPES if "therm" in _]

        # Set up the transfer schemes based on the type of analysis set for this body
        if self.analysis_type in elastic_analyses:
            # Set up the load and displacement transfer schemes
            if transfer_settings.elastic_scheme == "hermes":
                self.transfer = HermesTransfer(
                    self.comm, self.struct_comm, self.aero_comm
                )

            elif transfer_settings.elastic_scheme == "rbf":
                basis = TransferScheme.PY_THIN_PLATE_SPLINE

                if "basis function" in transfer_settings.options:
                    if (
                        transfer_settings.options["basis function"].lower()
                        == "thin plate spline"
                    ):
                        basis = TransferScheme.PY_THIN_PLATE_SPLINE
                    elif (
                        transfer_settings.options["basis function"].lower()
                        == "gaussian"
                    ):
                        basis = TransferScheme.PY_GAUSSIAN
                    elif (
                        transfer_settings.options["basis function"].lower()
                        == "multiquadric"
                    ):
                        basis = TransferScheme.PY_MULTIQUADRIC
                    elif (
                        transfer_settings.options["basis function"].lower()
                        == "inverse multiquadric"
                    ):
                        basis = TransferScheme.PY_INVERSE_MULTIQUADRIC
                    else:
                        print("Unknown RBF basis function for body number")
                        quit()

                self.transfer = TransferScheme.pyRBF(
                    comm, struct_comm, struct_root, aero_comm, aero_root, basis, 1
                )

            elif transfer_settings.elastic_scheme == "meld":
                self.transfer = TransferScheme.pyMELD(
                    comm,
                    struct_comm,
                    struct_root,
                    aero_comm,
                    aero_root,
                    transfer_settings.isym,
                    transfer_settings.npts,
                    transfer_settings.beta,
                )

            elif transfer_settings.elastic_scheme == "linearized meld":
                self.transfer = TransferScheme.pyLinearizedMELD(
                    comm,
                    struct_comm,
                    struct_root,
                    aero_comm,
                    aero_root,
                    transfer_settings.isym,
                    transfer_settings.npts,
                    transfer_settings.beta,
                )

            elif transfer_settings.elastic_scheme == "beam":
                self.xfer_ndof = transfer_settings.options["ndof"]
                self.transfer = TransferScheme.pyBeamTransfer(
                    comm,
                    struct_comm,
                    struct_root,
                    aero_comm,
                    aero_root,
                    transfer_settings.options["conn"],
                    transfer_settings.options["nelems"],
                    transfer_settings.options["order"],
                    transfer_settings.options["ndof"],
                )
            else:
                print("Error: Unknown transfer scheme for body")
                quit()

        # Set up the transfer schemes based on the type of analysis set for this body
        if self.analysis_type in thermal_analyses:
            # Set up the thermal transfer schemes

            if transfer_settings.thermal_scheme == "meld":
                self.thermal_transfer = TransferScheme.pyMELDThermal(
                    comm,
                    struct_comm,
                    struct_root,
                    aero_comm,
                    aero_root,
                    transfer_settings.isym,
                    transfer_settings.thermal_npts,
                    transfer_settings.thermal_beta,
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
        Initialize the variables each time we run an analysis.

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        """

        # We re-initialize aitken acceleration every time
        self.aitken_is_initialized = False
        self.aitken_adj_is_initialized = False

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
                self.struct_temps[scenario.id] = (
                    np.ones(ns, dtype=self.dtype) * scenario.T_ref
                )
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
                    self.struct_temps[id].append(
                        np.ones(ns, dtype=self.dtype) * scenario.T_ref
                    )
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
        nf = scenario.count_adjoint_functions()
        self.aitken_init = True

        # Allocate the adjoint variables and internal body variables required
        ns = 3 * self.struct_nnodes
        na = 3 * self.aero_nnodes
        self.aero_shape_term[scenario.id] = np.zeros((na, nf), dtype=self.dtype)
        self.struct_shape_term[scenario.id] = np.zeros((ns, nf), dtype=self.dtype)

        if self.transfer is not None:
            ns = 3 * self.struct_nnodes
            na = 3 * self.aero_nnodes

            self.struct_loads_ajp = np.zeros((ns, nf), dtype=self.dtype)
            self.aero_loads_ajp = np.zeros((na, nf), dtype=self.dtype)
            self.struct_disps_ajp = np.zeros((ns, nf), dtype=self.dtype)
            self.aero_disps_ajp = np.zeros((na, nf), dtype=self.dtype)

            # There are two contributions to the term struct_disps_ajp. We compute and
            # store them separately and then add the contributions. These two contributions
            # are from the displacement and load transfer, respectively and are computed in
            # the transfer_disps_adjoint and transfer_loads_adjoint. Since they are computed
            # at different times and we don't impose a calling order/requirement we have to
            # store them separately.
            self.struct_disps_ajp_disps = np.zeros((ns, nf), dtype=self.dtype)
            self.struct_disps_ajp_loads = np.zeros((ns, nf), dtype=self.dtype)

        if self.thermal_transfer is not None:
            ns = self.struct_nnodes
            na = self.aero_nnodes

            self.struct_flux_ajp = np.zeros((ns, nf), dtype=self.dtype)
            self.aero_flux_ajp = np.zeros((na, nf), dtype=self.dtype)
            self.struct_temps_ajp = np.zeros((ns, nf), dtype=self.dtype)
            self.aero_temps_ajp = np.zeros((na, nf), dtype=self.dtype)

        return

    def get_aero_disps(self, scenario, time_index=0, add_dxa0=False):
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
                aero_disps = self.aero_disps[scenario.id]
            else:
                aero_disps = self.aero_disps[scenario.id][time_index]
            if add_dxa0:
                aero_disps += self.aero_X - self._aero_X_orig
                ddisps = self.aero_X - self._aero_X_orig
            return aero_disps
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

    def transfer_disps(self, scenario, time_index=0, jump=False):
        """
        Transfer the displacements on the structural mesh to the aerodynamic mesh
        for the given scenario.

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        time_index: int
            The time-index for time-dependent problems
        jump: bool
            Whether to transfer from current to next time index
        """
        if self.transfer is not None:
            if scenario.steady:
                aero_disps = self.aero_disps[scenario.id]
                struct_disps = self.struct_disps[scenario.id]
            else:
                aero_time_index = time_index + 1 if jump else time_index
                aero_disps = self.aero_disps[scenario.id][aero_time_index]
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

    def transfer_temps(self, scenario, time_index=0, jump=False):
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
                aero_time_index = time_index + 1 if jump else time_index
                struct_temps = self.struct_temps[scenario.id][time_index]
                aero_temps = self.aero_temps[scenario.id][aero_time_index]
            self.thermal_transfer.transferTemp(struct_temps, aero_temps)

        return

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

    def get_aero_loads_ajp(self, scenario):
        """
        Get the product of the load transfer adjoint variables with the derivative
        of the load transfer residuals with respect to the aerodynamic forces

        aero_loads_ajp = dL/dfA^{T} * psi_L

        The value of this product is computed

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        """

        if self.transfer is not None:
            return self.aero_loads_ajp

        return None

    def get_aero_disps_ajp(self, scenario):
        """
        Get the product of the adjoint variables with the grid residuals with the
        derivative of the displacement at the aerodynamic surface nodes

        aero_disps_ajp = dG/duA^{T} * psi_G

        Note that this term is typically just the shape sensitivity

        dG/duA^{T} * psi_G = dG/dxA0 * psi_G

        This term must be set by the aerodynamic analysis.

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        """

        if self.transfer is not None:
            return self.aero_disps_ajp

        return None

    def get_struct_loads_ajp(self, scenario):
        """
        Get the product of the structural adjoint variables with the derivative of the
        structural residuals with respect to the structural loads.

        struct_loads_ajp = dS/dfS^{T} * psi_S

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        """

        if self.transfer is not None:
            return self.struct_loads_ajp

        return None

    def get_struct_disps_ajp(self, scenario):
        """
        Get the product of the displacement transfer adjoint variables with the derivative
        of the displacement transfer residuals w.r.t. the structural displacements plus the
        product of the load transfer adjoint variables with the derivative of the load
        transfer residuals w.r.t. the structural displacements.

        struct_disps_ajp = dD/duS^{T} * psi_D + dL/duS^{T} * psi_L

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        """

        if self.transfer is not None:
            return self.struct_disps_ajp

        return None

    def get_aero_heat_flux_ajp(self, scenario):
        """

        aero_flux_ajp = dQ/dhA^{T} * psi_Q

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        """

        if self.thermal_transfer is not None:
            return self.aero_flux_ajp

        return None

    def get_struct_heat_flux_ajp(self, scenario):
        """

        struct_flux_ajp = dS/dhS^{T} * psi_S

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        """

        if self.thermal_transfer is not None:
            return self.struct_flux_ajp

        return None

    def get_aero_temps_ajp(self, scenario):
        """
        Get the


        aero_temps_ajp = dA/dtA^{T} * psi_A

        This variable must be set in an aerodynamic analysis.

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        """

        if self.thermal_transfer is not None:
            return self.aero_temps_ajp

        return None

    def get_struct_temps_ajp(self, scenario):
        """

        struct_temps_ajp = dT/dtS^{T} * psi_T

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        """

        if self.thermal_transfer is not None:
            return self.struct_temps_ajp

        return None

    def transfer_loads_adjoint(self, scenario, time_index=0):
        """
        Perform the adjoint computation for the load transfer operation. This code
        computes and stores the required Jacobian-adjoint products.

        Find the load transfer adjoint variables, psi_L, by solving the equation

        dL/dfS^{T} * psi_L = - dS/dfS^{T} * psi_S

        The right-hand-side term is stored as

        struct_loads_ajp = dS/dfS^{T} * psi_S

        and the Jacobian dL/dfS = I. After computing psi_L, the code computes the
        Jacobian-adjoint products

        aero_loads_ajp = dL/dfA^{T} * psi_L
        struct_disps_ajp_loads = dL/duS^{T} * psi_L

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        """
        nfunctions = scenario.count_adjoint_functions()

        if self.transfer is not None:
            # Contribute to the force integration and structural adjoint right-hand-sides
            # from the load transfer adjoint
            temp_fa = np.zeros(3 * self.aero_nnodes, dtype=self.dtype)
            temp_us = np.zeros(3 * self.struct_nnodes, dtype=self.dtype)
            for k in range(nfunctions):
                # Solve for psi_L - Note that dL/dfS is the identity matrix. Ensure that
                # the vector is in contiguous memory.
                psi_L = -self.struct_loads_ajp[:, k].copy()

                # Compute aero_loads_ajp = dL/dfa^{T} * psi_L
                self.transfer.applydLdfATrans(psi_L, temp_fa)
                self.aero_loads_ajp[:, k] = temp_fa

                # Compute struct_disps_ajp_loads = dL/dus^{T} * psi_L
                self.transfer.applydLduSTrans(psi_L, temp_us)
                self.struct_disps_ajp_loads[:, k] = temp_us

            # Compute the update to the struct_disps_ajp
            self.struct_disps_ajp[:] = (
                self.struct_disps_ajp_disps[:] + self.struct_disps_ajp_loads[:]
            )

        return

    def transfer_disps_adjoint(self, scenario, time_index=0):
        """
        Perform the adjoint computation for the displacement transfer operations. This code
        computes and stores required Jacobian-adjoint products.

        Find the displacement transfer adjoint variables, psi_D, by solving the equation:

        dD/duA^{T} * psi_D = - dG/duA^{T} * psi_G

        Here the right-hand-side term is stored as

        aero_disps_ajp = dG/duA^{T} * psi_G

        The Jacobian dD/duA = I. After computing psi_D, the code computes the Jacobian-adjoint
        products

        struct_disps_ajp_disps = dD/duS^{T} * psi_D

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        """
        nfunctions = scenario.count_adjoint_functions()

        if self.transfer is not None:
            temp_us = np.zeros(3 * self.struct_nnodes, dtype=self.dtype)

            for k in range(nfunctions):
                # Solve for psi_D - Note that dD/dua is the identity matrix
                # Copy the values into contiguous memory.
                psi_D = -self.aero_disps_ajp[:, k].copy()

                # Set the dD/duS^{T} * psi_D product
                self.transfer.applydDduSTrans(psi_D, temp_us)
                self.struct_disps_ajp_disps[:, k] = temp_us

            # Compute the update to the struct_disps_ajp
            self.struct_disps_ajp[:] = (
                self.struct_disps_ajp_disps[:] + self.struct_disps_ajp_loads[:]
            )

        return

    def transfer_heat_flux_adjoint(self, scenario, time_index=0):
        """
        Perform the adjoint computation for the heat flux transfer operations.

        This function first solves for the heat flux transfer adjoint, psi_Q by solving
        the equation

        dQ/dhS^{T} * psi_Q = - dS/dhS^{T} * psi_S

        Here the right-hand-side is stored as

        struct_flux_ajp = dS/dhS^{T} * psi_S

        Since dQ/dhS = I, the adjoint variables are psi_Q = - struct_flux_ajp.

        The code then computes

        aero_flux_ajp = dQ/dhA^{T} * psi_Q

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        """

        nfunctions = scenario.count_adjoint_functions()

        if self.thermal_transfer is not None:
            temp = np.zeros(self.aero_nnodes, dtype=self.dtype)

            for k in range(nfunctions):
                psi_Q = -self.struct_flux_ajp[:, k].copy()

                # In the MELD interface code qA = hA so applydQdqATrans could/should?
                # be called applydQdhATrans
                self.thermal_transfer.applydQdqATrans(psi_Q, temp)
                self.aero_flux_ajp[:, k] = temp

        return

    def transfer_temps_adjoint(self, scenario, time_index=0):
        """
        Perform the adjoint computations for the thermal transfer operation.

        This function first solves for the temperature transfer adjoint psi_T, by solving
        the equation

        dT/dtA^{T} * psi_T = - dA/dtA^{T} * psi_A

        Here the right-hand-side is stored as

        aero_temps_ajp = dA/dtA^{T} * psi_A

        Since dT/dtA = I, then psi_T = - aero_temps_ajp.

        The code then computes

        struct_temps_ajp = dT/dtS^{T} * psi_T

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        """

        nfunctions = scenario.count_adjoint_functions()

        if self.thermal_transfer is not None:
            temp = np.zeros(self.struct_nnodes, dtype=self.dtype)

            for k in range(nfunctions):
                psi_T = -self.aero_temps_ajp[:, k].copy()

                self.thermal_transfer.applydTdtSTrans(psi_T, temp)
                self.struct_temps_ajp[:, k] = temp

        return

    def get_aero_coordinate_derivatives(self, scenario):
        """
        Get the coordinate derivatives for the functions with respect to the
        aerodynamic surface nodes

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        """
        return self.aero_shape_term[scenario.id]

    def get_struct_coordinate_derivatives(self, scenario):
        """
        Get the coordinate derivatives for the functions with respect to the
        structural nodes

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        """
        return self.struct_shape_term[scenario.id]

    def add_coordinate_derivative(self, scenario, step):
        """
        Add the coordinate derivatives for each function of interest to the aerodynamic
        and structural surface nodes - stored in aero_shape_term and struct_shape_term, respectively

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        """

        if self.transfer is not None:
            nfunctions = scenario.count_adjoint_functions()

            # Aerodynamic + Structural coordinate derivatives from transfer scheme
            temp_xa = np.zeros(3 * self.aero_nnodes, dtype=self.dtype)
            temp_xs = np.zeros(3 * self.struct_nnodes, dtype=self.dtype)
            aero_shape_term = self.get_aero_coordinate_derivatives(scenario)
            struct_shape_term = self.get_struct_coordinate_derivatives(scenario)
            for k in range(nfunctions):
                # Solve for psi_L - Note that dL/dfS is the identity matrix. Ensure that
                # the vector is in contiguous memory.
                psi_L = -self.struct_loads_ajp[:, k].copy()
                self.transfer.applydLdxA0(psi_L, temp_xa)
                aero_shape_term[:, k] += temp_xa

                self.transfer.applydLdxS0(psi_L, temp_xs)
                struct_shape_term[:, k] += temp_xs

                # Solve for psi_D - Note that dD/dua is the identity matrix
                # Copy the values into contiguous memory.
                psi_D = -self.aero_disps_ajp[:, k].copy()
                self.transfer.applydDdxA0(psi_D, temp_xa)
                aero_shape_term[:, k] += temp_xa

                self.transfer.applydDdxS0(psi_D, temp_xs)
                struct_shape_term[:, k] += temp_xs

        # TODO : what about thermal transfer coordinate derivatives?

        return

    def aitken_relax(self, comm, scenario, tol=1e-13):
        """
        Perform Aitken relaxation for the displacements set in the
        """

        # If Aitken relaxation is turned off, skip this
        if not (self.use_aitken_accel) and not (self.use_simple_accel):
            return

        if not self.aitken_is_initialized:
            # Aitken data for the displacements
            self.theta = self.theta_init
            self.prev_update = np.zeros(3 * self.struct_nnodes, dtype=self.dtype)
            self.aitken_vec = np.zeros(3 * self.struct_nnodes, dtype=self.dtype)

            # Aitken data for the temperatures
            self.theta_t = self.theta_init
            self.prev_update_t = np.zeros(self.struct_nnodes, dtype=self.dtype)
            self.aitken_vec_t = (
                np.ones(self.struct_nnodes, dtype=self.dtype) * scenario.T_ref
            )

            self.aitken_is_initialized = True

        if self.transfer is not None:
            struct_disps = self.get_struct_disps(scenario)
            up = struct_disps - self.aitken_vec

            if self.use_aitken_accel:
                norm2 = np.linalg.norm(up - self.prev_update) ** 2.0
                norm2 = comm.allreduce(norm2)

                # Only update theta if the displacements changed
                if norm2 > tol:
                    # Compute the tentative theta value
                    value = (up - self.prev_update).dot(up)
                    value = comm.allreduce(value)
                    self.theta *= 1.0 - value / norm2

                    self.theta = np.max(
                        (np.min((self.theta, self.theta_max)), self.theta_min)
                    )

                # handle the min/max for complex step
                if type(self.theta) == np.complex128 or type(self.theta) == complex:
                    self.theta = self.theta.real + 0.0j

            if self.use_simple_accel:
                # overwrite theta learning rate from the relaxation scheme
                self.theta = self.relaxation_scheme.theta

                # call the scheme to drop the learning rate for the next iteration
                self.relaxation_scheme.relax_displacement()

            # perform the aitken update for displacement transfer
            self.aitken_vec += self.theta * up
            self.prev_update[:] = up[:]
            struct_disps[:] = self.aitken_vec

        if self.thermal_transfer is not None:
            struct_temps = self.get_struct_temps(scenario)
            up = struct_temps - self.aitken_vec_t

            if self.use_aitken_accel:
                norm2 = np.linalg.norm(up - self.prev_update_t) ** 2.0
                norm2 = comm.allreduce(norm2)

                # Only update theta if the temperatures changed
                if norm2 > tol:
                    # Compute the tentative theta value
                    value = (up - self.prev_update_t).dot(up)
                    value = comm.allreduce(value.real)
                    self.theta_t *= 1.0 - value / norm2

                    self.theta_t = np.max(
                        (np.min((self.theta_t, self.theta_max)), self.theta_min)
                    )

                # handle the min/max for complex step
                if type(self.theta_t) == np.complex128 or type(self.theta_t) == complex:
                    self.theta_t = self.theta_t.real + 0.0j

            if self.use_simple_accel:
                # overwrite theta learning rate from the relaxation scheme
                self.theta_t = self.relaxation_scheme.theta_t

                # call the scheme to drop the learning rate for the next iteration
                self.relaxation_scheme.relax_thermal()

            self.aitken_vec_t += self.theta_t * up
            self.prev_update_t[:] = up[:]
            struct_temps[:] = self.aitken_vec_t

        return

    def aitken_adjoint_relax(self, comm, scenario, tol=1e-16):
        # If Aitken relaxation is turned off, skip this
        if not (self.use_aitken_accel) and not (self.use_simple_accel):
            return

        # exit early to not perform aitken adjoint relax
        return

        # number of nodes and functions
        ns = self.struct_nnodes
        nf = scenario.count_adjoint_functions()

        if not self.aitken_adj_is_initialized:
            # Aitken data for the displacements
            # self.theta_adj = self.theta_init
            # self.prev_update = np.zeros(3 * self.struct_nnodes, dtype=self.dtype)
            # self.aitken_vec = np.zeros(3 * self.struct_nnodes, dtype=self.dtype)

            # Aitken data for the temperatures
            self.theta_adj_t = np.ones((nf), dtype=self.dtype) * self.theta_init

            self.prev_adj_update_t = np.zeros((ns, nf), dtype=self.dtype)
            self.aitken_adj_vec_t = np.zeros((ns, nf), dtype=self.dtype)

            self.aitken_adj_is_initialized = True

        struct_flux_ajp = self.get_struct_heat_flux_ajp(scenario)
        if self.thermal_transfer is not None:
            for ifunc in range(nf):
                # if comm.rank == 0: print(f"Aitken adjoint loop, func {ifunc}",flush=True)
                up = struct_flux_ajp[:, ifunc] - self.aitken_adj_vec_t[:, ifunc]

                if self.use_aitken_accel:
                    norm2 = np.linalg.norm(up - self.prev_adj_update_t[:, ifunc]) ** 2.0
                    norm2 = comm.allreduce(norm2)

                    # Only update theta if the temperatures changed
                    if norm2 > tol:
                        # Compute the tentative theta value
                        value = (up - self.prev_adj_update_t[:, ifunc]).dot(up)
                        value = comm.allreduce(value.real)
                        self.theta_adj_t[ifunc] *= 1.0 - value / norm2

                        self.theta_adj_t[ifunc] = np.max(
                            (
                                np.min((self.theta_adj_t[ifunc], self.theta_max)),
                                self.theta_min,
                            )
                        )

                    # handle the min/max for complex step
                    if (
                        type(self.theta_adj_t[ifunc]) == np.complex128
                        or type(self.theta_adj_t[ifunc]) == complex
                    ):
                        self.theta_adj_t[ifunc] = self.theta_adj_t[ifunc].real + 0.0j

                # if self.use_simple_accel:
                #     # overwrite theta learning rate from the relaxation scheme
                #     self.theta_t = self.relaxation_scheme.theta_t

                #     # call the scheme to drop the learning rate for the next iteration
                #     self.relaxation_scheme.relax_thermal()

                self.aitken_adj_vec_t[:, ifunc] += self.theta_adj_t[ifunc] * up
                self.prev_adj_update_t[:, ifunc] = up[:]
                struct_flux_ajp[:, ifunc] = self.aitken_adj_vec_t[:, ifunc]

                # barrier before next iteration
                # comm.Barrier()

        # elastic adjoint transfer
        # if self.transfer is not None:
        #     struct_disps = self.get_struct_disps(scenario)
        #     up = struct_disps - self.aitken_vec

        #     if self.use_aitken_accel:
        #         norm2 = np.linalg.norm(up - self.prev_update) ** 2.0
        #         norm2 = comm.allreduce(norm2)

        #         # Only update theta if the displacements changed
        #         if norm2 > tol:
        #             # Compute the tentative theta value
        #             value = (up - self.prev_update).dot(up)
        #             value = comm.allreduce(value)
        #             self.theta *= 1.0 - value / norm2

        #       :q      self.theta = np.max(
        #                 (np.min((self.theta, self.theta_max)), self.theta_min)
        #             )

        #         # handle the min/max for complex step
        #         if type(self.theta) == np.complex128 or type(self.theta) == complex:
        #             self.theta = self.theta.real + 0.0j

        #     if self.use_simple_accel:
        #         # overwrite theta learning rate from the relaxation scheme
        #         self.theta = self.relaxation_scheme.theta

        #         # call the scheme to drop the learning rate for the next iteration
        #         self.relaxation_scheme.relax_displacement()

        #     # perform the aitken update for displacement transfer
        #     self.aitken_vec += self.theta * up
        #     self.prev_update[:] = up[:]
        #     struct_disps[:] = self.aitken_vec

        return

    def _distribute_aero_loads(self, data):
        """
        distribute the aero loads and heat flux from a loads file
        """
        for scenario_id in data:
            scenario_data = data[scenario_id]
            for entry in scenario_data:
                for ind, aero_id in enumerate(self.aero_id):
                    if entry["aeroID"] == aero_id and entry["bodyName"] == self.name:
                        if self.transfer is not None:
                            self.aero_loads[scenario_id][3 * ind : 3 * ind + 3] = entry[
                                "load"
                            ]
                        if self.thermal_transfer is not None:
                            self.aero_heat_flux[scenario_id][ind] = entry["hflux"]
                        break

    def _collect_aero_mesh(self, comm, root=0):
        """
        gather the aero ids, and aero mesh coordinates or aero_X onto the root processor
        these will later be written to the loads file
        """
        all_aero_ids = comm.gather(self.aero_id, root=root)
        all_aero_X = comm.gather(self.aero_X, root=root)

        aero_ids = []
        aero_X = []

        if comm.rank == root:
            aero_ids = []
            for d in all_aero_ids:
                if d is not None:
                    aero_ids.append(d)

            aero_X = []
            for d in all_aero_X:
                if d is not None:
                    aero_X.append(d)

            if len(aero_ids) == 0:
                aero_ids = np.arange(aero_X.shape[0] // 3, dtype=int)
            else:
                aero_ids = np.concatenate(aero_ids)

            if len(aero_X) == 0:
                aero_X = np.zeros((3 * len(aero_ids), 1))
            else:
                aero_X = np.concatenate(aero_X)

        return aero_ids, aero_X

    def _collect_aero_loads(self, comm, scenario, root=0):
        """
        gather the aerodynamic load and heat flux from each MPI processor onto the root
        Then return the global aero ids, heat fluxes, and loads, which are later written to a file
        """
        all_aero_ids = comm.gather(self.aero_id, root=root)
        if self.transfer is not None:
            all_aero_loads = comm.gather(self.aero_loads[scenario.id], root=root)
        else:
            all_aero_loads = []
        if self.thermal_transfer is not None:
            all_aero_hflux = comm.gather(self.aero_heat_flux[scenario.id], root=root)
        else:
            all_aero_hflux = []

        aero_ids = []
        aero_loads = []
        aero_hflux = []

        if comm.rank == root:
            aero_ids = []
            for d in all_aero_ids:
                if d is not None:
                    aero_ids.append(d)

            aero_loads = []
            for d in all_aero_loads:
                if d is not None:
                    aero_loads.append(d)

            aero_hflux = []
            for d in all_aero_hflux:
                if d is not None:
                    aero_hflux.append(d)

            if len(aero_ids) == 0:
                aero_ids = np.arange(aero_loads.shape[0] // 3, dtype=int)
            else:
                aero_ids = np.concatenate(aero_ids)

            if len(aero_loads) > 0:
                aero_loads = np.concatenate(aero_loads)
            else:
                aero_loads = np.zeros((3 * len(aero_ids), 1))

            if len(aero_hflux) > 0:
                aero_hflux = np.concatenate(aero_hflux)
            else:
                aero_hflux = np.zeros((1 * len(aero_ids), 1))

        return aero_ids, aero_hflux, aero_loads

    def _collect_struct_loads(self, comm, scenario, root=0):
        """
        gather the structural load and heat flux from each MPI processor onto the root
        Then return the global structural ids, heat fluxes, and loads, which are later written to a file
        """
        all_struct_ids = comm.gather(self.struct_id, root=root)
        if self.transfer is not None:
            all_struct_loads = comm.gather(self.struct_loads[scenario.id], root=root)
        else:
            all_struct_loads = []
        if self.thermal_transfer is not None:
            all_struct_hflux = comm.gather(
                self.struct_heat_flux[scenario.id], root=root
            )
        else:
            all_struct_hflux = []

        struct_ids = []
        struct_loads = []
        struct_hflux = []

        if comm.rank == root:
            struct_ids = []
            for d in all_struct_ids:
                if d is not None:
                    struct_ids.append(d)

            struct_loads = []
            for d in all_struct_loads:
                if d is not None:
                    struct_loads.append(d)

            struct_hflux = []
            for d in all_struct_hflux:
                if d is not None:
                    struct_hflux.append(d)

            if len(struct_ids) == 0:
                struct_ids = np.arange(struct_loads.shape[0] // 3, dtype=int)
            else:
                struct_ids = np.concatenate(struct_ids)

            if len(struct_loads) > 0:
                struct_loads = np.concatenate(struct_loads)
            else:
                struct_loads = np.zeros((3 * len(struct_ids), 1))

            if len(struct_hflux) > 0:
                struct_hflux = np.concatenate(struct_hflux)
            else:
                struct_hflux = np.zeros((1 * len(struct_ids), 1))

        return struct_ids, struct_hflux, struct_loads

    def collect_coordinate_derivatives(self, comm, discipline, scenarios, root=0):
        """
        Write the sensitivity files for the aerodynamic and structural meshes on
        the root processor.

        This code collects the sensitivities from each processor and collects the
        result to the root node.

        NOTE : we do not need to compute composite function coordinate derivatives
        If we are able to compute shape derivatives of analysis functions, then
        the composite function shape derivatives can be automatically computed without
        writing extra coordinate derivatives to ESP/CAPS
        """

        if discipline == "aerodynamic" or discipline == "flow" or discipline == "aero":
            all_aero_ids = comm.gather(self.aero_id, root=root)

            # append struct shapes for each scenario
            full_aero_shape_term = []
            for scenario in scenarios:
                full_aero_shape_term.append(self.aero_shape_term[scenario.id])
            full_aero_shape_term = np.concatenate(full_aero_shape_term, axis=1)

            all_aero_shape = comm.gather(full_aero_shape_term, root=root)

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

        elif (
            discipline == "structures"
            or discipline == "structural"
            or discipline == "struct"
        ):
            all_struct_ids = comm.gather(self.struct_id, root=root)

            # append struct shapes for each scenario
            full_struct_shape_term = []
            for scenario in scenarios:
                full_struct_shape_term.append(self.struct_shape_term[scenario.id])
            full_struct_shape_term = np.concatenate(full_struct_shape_term, axis=1)

            # gather the full struct shape terms across processors
            all_struct_shape = comm.gather(full_struct_shape_term, root=root)

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
                    # add 1 to struct_ids to make one-based for nastran
                    struct_ids += 1
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
    #                         scenario.T_ref
    #                         * np.ones(
    #                             body.struct_nnodes * body.therm_xfer_ndof,
    #                             dtype=TransferScheme.dtype,
    #                         )
    #                     )
    #                     aitken_therm_vec_body.append(
    #                         scenario.T_ref
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
