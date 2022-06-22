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

class Body(Base):
    """
    Defines a body base class for FUNtoFEM. Can be used as is or as a
    parent class for bodies with shape parameterization.
    """
    def __init__(self, name, analysis_type, id=0, group=None,
                 boundary=0, fun3d=True, motion_type='deform'):
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
            self.add_variable('rigid_motion', dv('RotRate', active=False, id=1))
            self.add_variable('rigid_motion', dv('RotFreq', active=False, id=2))
            self.add_variable('rigid_motion', dv('RotAmpl', active=False, id=3))
            self.add_variable('rigid_motion', dv('RotOrgx', active=False, id=4))
            self.add_variable('rigid_motion', dv('RotOrgy', active=False, id=5))
            self.add_variable('rigid_motion', dv('RotOrgz', active=False, id=6))
            self.add_variable('rigid_motion', dv('RotVecx', active=False, id=7))
            self.add_variable('rigid_motion', dv('RotVecy', active=False, id=8))
            self.add_variable('rigid_motion', dv('RotVecz', active=False, id=9))
            self.add_variable('rigid_motion', dv('TrnRate', active=False, id=10))
            self.add_variable('rigid_motion', dv('TrnFreq', active=False, id=11))
            self.add_variable('rigid_motion', dv('TrnAmpl', active=False, id=12))
            self.add_variable('rigid_motion', dv('TrnVecx', active=False, id=13))
            self.add_variable('rigid_motion', dv('TrnVecy', active=False, id=14))
            self.add_variable('rigid_motion', dv('TrnVecz', active=False, id=15))

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
        self.thermal_index = 3 #0,1,2 for xyz and 3 for temp (mod 4) see tacs_interface.py
        self.T_ref = 300.0 # reference temperature in Kelvin

        # Node locations u
        self.struct_X = None
        self.aero_X = None

        # ID number of nodes
        self.struct_id = None
        self.aero_id = None

        # forward variables
        self.struct_disps = {}
        self.struct_loads = {}
        self.struct_temps  = {}
        self.struct_heat_flux = {}
        self.struct_shape_term = {}

        self.rigid_transform = {}
        self.aero_disps = {}
        self.aero_loads = {}
        self.aero_shape_term = {}

        self.aero_temps  = {}
        self.aero_heat_flux = {}

        return

    def initialize_struct_mesh(self, scenario):

        self.struct_nnodes[scenario]

        pass

    def initialize_aero_mesh(self, scenario):
        pass

    def initialize_variables(self, scenario):

        if self.transfer is not None:
            if scenario.steady:
                self.struct_loads[scenario.id] = np.zeros(3 * self.struct_nnodes[scenario.id], dtype=self.dtype)
                self.aero_loads[scenario.id] = np.zeros(3 * self.aero_nnodes[scenario.id], dtype=self.dtype)

                self.struct_disps[scenario.id] = np.zeros(3 * self.struct_nnodes[scenario.id], dtype=self.dtype)
                self.aero_disps[scenario.id] = np.zeros(3 * self.aero_nnodes[scenario.id], dtype=self.dtype)
            else:
                for time_index in range(scenario.steps):
                    self.struct_loads[scenario.id] = []
                    self.aero_loads[scenario.id] = []

                    np.zeros(3 * self.struct_nnodes[scenario.id], dtype=self.dtype)
                    np.zeros(3 * self.aero_nnodes[scenario.id], dtype=self.dtype)

                    self.struct_disps[scenario.id] = np.zeros(3 * self.struct_nnodes[scenario.id], dtype=self.dtype)
                    self.aero_disps[scenario.id] = np.zeros(3 * self.aero_nnodes[scenario.id], dtype=self.dtype)



        if self.thermal_transfer is not None:
            if scenario.steady:
                self.struct_temps[scenario.id] = np.zeros(self.struct_nnodes[scenario.id], dtype=self.dtype)
                self.aero_temps[scenario.id] = np.zeros(self.aero_nnodes[scenario.id], dtype=self.dtype)


                    body.struct_temps = np.zeros(
                        body.struct_nnodes * body.therm_xfer_ndof, dtype=TACS.dtype
                    )
                    body.struct_temps = np.zeros(
                        body.struct_nnodes * body.therm_xfer_ndof, dtype=TACS.dtype
                    )
            body.struct_disps = np.zeros(
                        body.struct_nnodes * body.xfer_ndof, dtype=TACS.dtype
                    )

    def _initialize_transfer(self, transfer_options):
        """
        Initialize the transfer scheme

        Parameters
        ----------
        transfer_options: dictionary or list of dictionaries
            options for the load and displacement transfer scheme for the bodies
        """

        # If the user did not specify a transfer scheme default to MELD
        if transfer_options is None:
            transfer_options = []
            for body in self.model.bodies:
                transfer_options.append({'scheme': 'meld', 'isym': -1, 'beta': 0.5, 'npts': 200})

        # if the user gave a dictionary instead of a list of
        # dictionaries, assume all bodies use the same settings
        if type(transfer_options) is dict:
            transfer_options = len(self.model.bodies) * [ transfer_options ]

        for ibody, body in enumerate(self.model.bodies):
            body.transfer = None
            body.thermal_transfer = None

            body_analysis_type = 'aeroelastic'
            if 'analysis_type' in transfer_options[ibody]:
                body_analysis_type = transfer_options[ibody]['analysis_type'].lower()

            # Set up the transfer schemes based on the type of analysis set for this body
            if body_analysis_type == 'aeroelastic' or body_analysis_type == 'aerothermoelastic':

                # Set up the load and displacement transfer schemes
                if transfer_options[ibody]['scheme'].lower() == 'hermes':

                    body.transfer = HermesTransfer(self.comm, self.struct_comm, self.aero_comm)

                elif transfer_options[ibody]['scheme'].lower() == 'rbf':
                    basis = TransferScheme.PY_THIN_PLATE_SPLINE

                    if 'basis function' in transfer_options[ibody]:
                        if transfer_options[ibody]['basis function'].lower() == 'thin plate spline':
                            basis = TransferScheme.PY_THIN_PLATE_SPLINE
                        elif transfer_options[ibody]['basis function'].lower() == 'gaussian':
                            basis = TransferScheme.PY_GAUSSIAN
                        elif transfer_options[ibody]['basis function'].lower() == 'multiquadric':
                            basis = TransferScheme.PY_MULTIQUADRIC
                        elif transfer_options[ibody]['basis function'].lower() == 'inverse multiquadric':
                            basis = TransferScheme.PY_INVERSE_MULTIQUADRIC
                        else:
                            print('Unknown RBF basis function for body number', ibody)
                            quit()

                    body.transfer = TransferScheme.pyRBF(self.comm, self.struct_comm,
                                                         self.struct_root, self.aero_comm,
                                                         self.aero_root, basis, 1)

                elif transfer_options[ibody]['scheme'].lower() == 'meld':
                    # defaults
                    isym = -1 # No symmetry
                    beta = 0.5 # Decay factor
                    num_nearest = 200 # Number of nearest neighbours

                    if 'isym' in transfer_options[ibody]:
                        isym = transfer_options[ibody]['isym']
                    if 'beta' in transfer_options[ibody]:
                        beta = transfer_options[ibody]['beta']
                    if 'npts' in transfer_options[ibody]:
                        num_nearest = transfer_options[ibody]['npts']

                    body.transfer = TransferScheme.pyMELD(self.comm, self.struct_comm,
                                                          self.struct_root, self.aero_comm,
                                                          self.aero_root,
                                                          isym, num_nearest, beta)

                elif transfer_options[ibody]['scheme'].lower() == 'linearized meld':
                    # defaults
                    isym = -1
                    beta = 0.5
                    num_nearest = 200

                    if 'isym' in transfer_options[ibody]:
                        isym = transfer_options[ibody]['isym']
                    if 'beta' in transfer_options[ibody]:
                        beta = transfer_options[ibody]['beta']
                    if 'npts' in transfer_options[ibody]:
                        num_nearest = transfer_options[ibody]['npts']


                    body.transfer = TransferScheme.pyLinearizedMELD(self.comm, self.struct_comm,
                                                                    self.struct_root, self.aero_comm,
                                                                    self.aero_root,
                                                                    isym, num_nearest, beta)

                elif transfer_options[ibody]['scheme'].lower()== 'beam':
                    conn = transfer_options[ibody]['conn']
                    nelems = transfer_options[ibody]['nelems']
                    order = transfer_options[ibody]['order']
                    ndof = transfer_options[ibody]['ndof']

                    body.xfer_ndof = ndof
                    body.transfer = TransferScheme.pyBeamTransfer(self.comm, self.struct_comm,
                                                                  self.struct_root, self.aero_comm,
                                                                  self.aero_root, conn, nelems,
                                                                  order, ndof)
                else:
                    print("Error: Unknown transfer scheme for body", ibody)
                    quit()

            # Set up the transfer schemes based on the type of analysis set for this body
            if body_analysis_type == 'aerothermal' or body_analysis_type == 'aerothermoelastic':
                # Set up the load and displacement transfer schemes

                if transfer_options[ibody]['thermal_scheme'].lower() == 'meld':
                    # defaults
                    isym = -1
                    beta = 0.5
                    num_nearest = 200

                    if 'isym' in transfer_options[ibody]:
                        isym = transfer_options[ibody]['isym']
                    if 'beta' in transfer_options[ibody]:
                        beta = transfer_options[ibody]['beta']
                    if 'npts' in transfer_options[ibody]:
                        num_nearest = transfer_options[ibody]['npts']

                    body.thermal_transfer = TransferScheme.pyMELDThermal(self.comm, self.struct_comm,
                                                                         self.struct_root, self.aero_comm,
                                                                         self.aero_root,
                                                                         isym, num_nearest, beta)
                else:
                    print("Error: Unknown thermal transfer scheme for body", ibody)
                    quit()

            # Load structural and aerodynamic meshes into FUNtoFEM
            # Only want real part for the initialization
            if body.transfer is not None:
                if TransferScheme.dtype == np.complex128 or TransferScheme.dtype == complex:
                    if self.struct_comm != MPI.COMM_NULL:
                        body.transfer.setStructNodes(body.struct_X.real + 0.0j)
                    else:
                        body.struct_nnodes = 0

                    if self.aero_comm != MPI.COMM_NULL:
                        body.transfer.setAeroNodes(body.aero_X.real + 0.0j)
                    else:
                        body.aero_nnodes = 0
                else:
                    if self.struct_comm != MPI.COMM_NULL:
                        body.transfer.setStructNodes(body.struct_X)
                    else:
                        body.struct_nnodes = 0

                    if self.aero_comm != MPI.COMM_NULL:
                        body.transfer.setAeroNodes(body.aero_X)
                    else:
                        body.aero_nnodes = 0

                # Initialize FUNtoFEM
                body.transfer.initialize()

                # Load structural and aerodynamic meshes into FUNtoFEM
                if TransferScheme.dtype == np.complex128 or TransferScheme.dtype == complex:
                    if self.struct_comm != MPI.COMM_NULL:
                        body.transfer.setStructNodes(body.struct_X)
                    else:
                        body.struct_nnodes = 0
                    if self.aero_comm != MPI.COMM_NULL:
                        body.transfer.setAeroNodes(body.aero_X)
                    else:
                        body.aero_nnodes = 0

            # Initialize the thermal problem
            if body.thermal_transfer is not None:
                if TransferScheme.dtype == np.complex128 or TransferScheme.dtype == complex:
                    if self.struct_comm != MPI.COMM_NULL:
                        body.thermal_transfer.setStructNodes(body.struct_X.real + 0.0j)
                    else:
                        body.struct_nnodes = 0

                    if self.aero_comm != MPI.COMM_NULL:
                        body.thermal_transfer.setAeroNodes(body.aero_X.real + 0.0j)
                    else:
                        body.aero_nnodes = 0
                else:
                    if self.struct_comm != MPI.COMM_NULL:
                        body.thermal_transfer.setStructNodes(body.struct_X)
                    else:
                        body.struct_nnodes = 0

                    if self.aero_comm != MPI.COMM_NULL:
                        body.thermal_transfer.setAeroNodes(body.aero_X)
                    else:
                        body.aero_nnodes = 0

                # Initialize FUNtoFEM
                body.thermal_transfer.initialize()

                # Load structural and aerodynamic meshes into FUNtoFEM
                if TransferScheme.dtype == np.complex128 or TransferScheme.dtype == complex:
                    if self.struct_comm != MPI.COMM_NULL:
                        body.thermal_transfer.setStructNodes(body.struct_X)
                    else:
                        body.struct_nnodes = 0
                    if self.aero_comm != MPI.COMM_NULL:
                        body.thermal_transfer.setAeroNodes(body.aero_X)
                    else:
                        body.aero_nnodes = 0

        return

    def _update_transfer(self):
        """
        Update the positions of the nodes in transfer schemes
        """
        self.struct_disps = []
        self.struct_temps = []
        for body in self.model.bodies:
            if body.transfer is not None:
                if self.struct_comm != MPI.COMM_NULL:
                    body.transfer.setStructNodes(body.struct_X)
                else:
                    body.struct_nnodes = 0
                if self.aero_comm != MPI.COMM_NULL:
                    body.transfer.setAeroNodes(body.aero_X)
                else:
                    body.aero_nnodes = 0

            if body.thermal_transfer is not None:
                if self.struct_comm != MPI.COMM_NULL:
                    body.thermal_transfer.setStructNodes(body.struct_X)
                else:
                    body.struct_nnodes = 0
                if self.aero_comm != MPI.COMM_NULL:
                    body.thermal_transfer.setAeroNodes(body.aero_X)
                else:
                    body.aero_nnodes = 0

        return

    def update_id(self, id):
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
        Add a new variable to the body's variable dictionary

        Parameters
        ----------
        vartype: str
            type of variable
        var: Variable object
            variable to be added
        """
        var.body = self.id

        super(Body, self).add_variable(vartype, var)


    def initalize_adjoint_variables(self, scenario):
        """
        Initialize the adjoint variables for the body
        """

        if body.transfer is not None:
            body.psi_L = np.zeros((body.struct_nnodes*body.xfer_ndof, nfunctions),
                                    dtype=TransferScheme.dtype)
            body.psi_S = np.zeros((body.struct_nnodes*body.xfer_ndof, nfunctions),
                                    dtype=TransferScheme.dtype)
            body.struct_rhs = np.zeros((body.struct_nnodes*body.xfer_ndof, nfunctions),
                                        dtype=TransferScheme.dtype)

            body.dLdfa = np.zeros((body.aero_nnodes*3, nfunctions),
                                    dtype=TransferScheme.dtype)
            body.dGdua = np.zeros((body.aero_nnodes*3, nfunctions),
                                    dtype=TransferScheme.dtype)
            body.psi_D = np.zeros((body.aero_nnodes*3, nfunctions),
                                    dtype=TransferScheme.dtype)

        if body.thermal_transfer is not None:
            # Thermal terms
            body.psi_Q = np.zeros((body.struct_nnodes*body.therm_xfer_ndof, nfunctions),
                                    dtype=TransferScheme.dtype)
            body.psi_T_S = np.zeros((body.struct_nnodes*body.therm_xfer_ndof, nfunctions),
                                    dtype=TransferScheme.dtype)
            body.struct_rhs_T = np.zeros((body.struct_nnodes*body.therm_xfer_ndof, nfunctions),
                                            dtype=TransferScheme.dtype)

            body.dQdfta = np.zeros((body.aero_nnodes, nfunctions),
                                    dtype=TransferScheme.dtype)
            body.dQfluxdfta = np.zeros((body.aero_nnodes*3, nfunctions),
                                        dtype=TransferScheme.dtype)
            body.dAdta = np.zeros((body.aero_nnodes, nfunctions),
                                    dtype=TransferScheme.dtype)
            body.psi_T = np.zeros((body.aero_nnodes, nfunctions),
                                    dtype=TransferScheme.dtype)

        body.aero_shape_term = np.zeros((body.aero_nnodes*3, nfunctions_total),
                                        dtype=TransferScheme.dtype)
        body.struct_shape_term = np.zeros((body.struct_nnodes*body.xfer_ndof, nfunctions_total),
                                            dtype=TransferScheme.dtype)


    def get_aero_disps(self, scenario, time_index=0):
        """
        Get the displacements on the aerodynamic surface for the given scenario
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
        Get the displacements on the aerodynamic surface for the given scenario
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
        Get the displacements on the aerodynamic surface for the given scenario
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
        Get the displacements on the aerodynamic surface for the given scenario
        """
        if self.transfer is not None:
            if scenario.steady:
                return self.struct_loads[scenario.id]
            else:
                return self.struct_loads[scenario.id][time_index]
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

    def transfer_temp(self, scenario, time_index=0):
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

    def transfer_loads_adjoint(self, scenario, time_index=0):
        nfunctions = scenario.get_num_functions()

        if self.transfer is not None:
            # Solve for psi_L - the load transfer adjoint
            adjL_rhs = self.adjL_rhs[scenario.id]
            psi_L = self.psi_L[scenario.id]
            adjF_rhs = self.adjF_rhs[scenario.id]
            adjS_rhs = self.adjS_rhs[scenario.id]

            # Solve for psi_L
            psi_L[:] = adjL_rhs[:]

            # Contribute to the force integration and structural adjoint right-hand-sides
            # from the load transfer adjoint
            temp_fa = np.zeros(3 * self.aero_nnodes, dtype=self.dtype)
            temp_us = np.zeros(3 * self.struct_nnodes, dtype=self.dtype)
            for k in range(nfunctions):
                # Copy the values into contiguous memory
                psi_Lk = psi_L[:, k].copy()

                # Contribute adjF_rhs -= dL/dfa^{T} * psi_L
                self.transfer.applydLdfaTrans(psi_Lk, temp_fa)
                adjF_rhs[:, k] -= temp_fa

                # Contribute adjS_rhs -= dL/dus^{T} * psi_L
                self.transfer.applydLdusTrans(psi_Lk, temp_us)
                adjS_rhs[:, k] -= temp_us

        return

    def transfer_disps_adjoint(self, scenario, step):
        nfunctions = scenario.get_num_functions()

        if self.transfer is not None:
            # Solve for psi_D - the displacement transfer adjoint
            psi_D = self.psi_D[scenario.id]
            adjD_rhs = self.adjD_rhs[scenario.id]
            adjS_rhs = self.adjS_rhs[scenario.id]

            # Solve for psi_D
            psi_D[:] = adjD_rhs[:]

            temp = np.zeros(3 * self.struct_nnodes, dtype=self.dtype)
            for k in range(nfunctions):
                # Copy the values into contiguous memory
                psi_Dk = psi_D[:, k].copy()

                # Contribute adjS_rhs -= dD/dus^{T} * psi_D
                self.transfer.applydDdusTrans(psi_Dk, temp)
                adjS_rhs[:, k] -= temp

        return

    def transfer_heat_flux_adjoint(self, scenario, step):
        nfunctions = scenario.get_num_functions()

        if self.thermal_transfer is not None:
            self.psi_T = np.zeros(body.aero_nnodes, dtype=TransferScheme.dtype)

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

    def add_coord_derivative(self, scenario, step):

        if self.transfer is not None:
            nfunctions = scenario.count_adjoint_functions()

            psi_L = self.psi_L[scenario.id]
            psi_D = self.psi_D[scenario.id]

            # Aerodynamic coordinate derivatives
            temp_xa = np.zeros(3 * self..aero_nnodes, dtype=self.dtype)
            temp_xs = np.zeros(3 * self.struct_nnodes, dtype=self.dtype)
            for k in range(nfunctions):
                # Load transfer term
                psi_Lk = psi_L[:, k].copy()
                self.transfer.applydLdxA0(psi_Lk, temp_xa)
                self.aero_shape_term[:, k] += temp_xa

                self.transfer.applydLdxS0(psi_Lk, temp_xs)
                self.struct_shape_term[:, k] += temp_xs

                # Displacement transfer term
                psi_Dk = psi_D[:, k].copy()
                self.transfer.applydDdxA0(psi_Dk, temp_xa)
                body.aero_shape_term[:, k] += temp_xa

                self.transfer.applydDdxS0(psi_Dk, temp_xs)
                self.struct_shape_term[:, k] += temp_xs

        return


    def collect_coordinate_derivatives(self, comm, discipline, root=0):
        """
        Write the sensitivity files for the aerodynamic and structural meshes on
        the root processor.

        This code collects the sensitivities from each processor and collects the
        result to the root node.
        """

        if discipline == 'aero':
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
                    aero_ids = np.arange(aero_shape.shape[0]//3, dtype=int)
                else:
                    aero_ids = np.concatenate(aero_ids)

            return aero_ids, aero_shape

        elif discipline == 'struct':
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
                    struct_ids = np.arange(struct_shape.shape[0]//3, dtype=int)
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
