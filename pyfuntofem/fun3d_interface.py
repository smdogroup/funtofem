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
import os
from fun3d.solvers import Flow, Adjoint
from fun3d import interface
from funtofem import TransferScheme
from .solver_interface import SolverInterface


class Fun3dInterface(SolverInterface):
    """
    FUNtoFEM interface class for FUN3D. Works for both steady and unsteady analysis.
    Requires the FUN3D directory structure.
    During the forward analysis, the FUN3D interface will operate in the scenario.name/Flow directory and
    scenario.name/Adjoint directory for the adjoint.

    FUN3D's FUNtoFEM coupling interface requires no additional configure flags to compile.
    To tell FUN3D that a body's motion should be driven by FUNtoFEM, set *motion_driver(i)='funtofem'*.
    """

    def __init__(
        self,
        comm,
        model,
        flow_dt=1.0,
        qinf=1.0,
        thermal_scale=1.0,
        fun3d_dir=None,
        forward_options=None,
        adjoint_options=None,
    ):
        """
        The instantiation of the FUN3D interface class will populate the model with the aerodynamic surface
        mesh, body.aero_X and body.aero_nnodes.
        The surface mesh on each processor only holds it's owned nodes. Transfer of that data to other processors
        is handled inside the FORTRAN side of FUN3D's FUNtoFEM interface.

        Parameters
        ----------
        comm: MPI.comm
            MPI communicator
        model: :class:`FUNtoFEMmodel`
            FUNtoFEM model. This instantiatio
        flow_dt: float
            flow solver time step size. Used to scale the adjoint term coming into and out of FUN3D since
            FUN3D currently uses a different adjoint formulation than FUNtoFEM.
        """

        self.comm = comm

        #  Instantiate FUN3D
        self.fun3d_flow = Flow()
        self.fun3d_adjoint = Adjoint()

        # Root and FUN3D directories
        self.root_dir = os.getcwd()
        if fun3d_dir is None:
            self.fun3d_dir = self.root_dir
        else:
            self.fun3d_dir = fun3d_dir

        # command line options
        self.forward_options = forward_options
        self.adjoint_options = adjoint_options

        # Temporary measure until FUN3D adjoint is reformulated
        self.flow_dt = flow_dt

        # dynamic pressure
        self.qinf = qinf
        self.dFdqinf = []

        # heat flux
        self.thermal_scale = thermal_scale  # = 1/2 * rho_inf * (V_inf)^3
        self.dHdq = []

        # Initialize the nodes associated with the bodies
        self._initialize_body_nodes(model.scenarios[0], model.bodies)

        return

    def _initialize_body_nodes(self, scenario, bodies):

        # Change directories to the flow directory
        flow_dir = os.path.join(self.fun3d_dir, scenario.name, "Flow")
        os.chdir(flow_dir)

        # Do the steps to initialize FUN3D
        self.fun3d_flow.initialize_project(comm=self.comm)
        if self.forward_options is None:
            options = {}
        else:
            options = self.forward_options
        self.fun3d_flow.setOptions(kwargs=options)
        self.fun3d_flow.initialize_data()
        interface.design_initialize()
        self.fun3d_flow.initialize_grid()

        # Initialize the flow solution
        bcont = self.fun3d_flow.initialize_solution()
        if bcont == 0:
            if self.comm.Get_rank() == 0:
                print("Negative volume returning fail")
            return 1

        # Go through the bodies and initialize the node locations
        for ibody, body in enumerate(bodies, 1):
            aero_nnodes = self.fun3d_flow.extract_surface_num(body=ibody)
            aero_X = np.zeros(3 * aero_nnodes, dtype=TransferScheme.dtype)

            if aero_nnodes > 0:
                x, y, z = self.fun3d_flow.extract_surface(aero_nnodes, body=ibody)
                aero_id = self.fun3d_flow.extract_surface_id(aero_nnodes, body=ibody)

                aero_X[0::3] = x[:]
                aero_X[1::3] = y[:]
                aero_X[2::3] = z[:]
            else:
                aero_id = np.zeros(aero_nnodes, dtype=int)

            # Initialize the aerodynamic node locations
            body.initialize_aero_nodes(aero_X, aero_id=aero_id)

        # Change directory back to the root directory
        self.fun3d_flow.post()
        os.chdir(self.root_dir)

        return

    def initialize(self, scenario, bodies):
        """
        Changes the directory to ./`scenario.name`/Flow, then
        initializes the FUN3D flow (forward) solver.

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The scenario that needs to be initialized
        bodies: :class:`~body.Body`
            list of FUNtoFEM bodies to either get new surface meshes from or to set the original mesh in

        Returns
        -------
        fail: int
            If the grid deformation failed, the intiialization will return 1
        """

        # Change directory to the directory associated with the scenario
        flow_dir = os.path.join(self.fun3d_dir, scenario.name, "Flow")
        os.chdir(flow_dir)

        # Do the steps to initialize FUN3D
        self.fun3d_flow.initialize_project(comm=self.comm)
        if self.forward_options is None:
            options = {}
        else:
            options = self.forward_options
        self.fun3d_flow.setOptions(kwargs=options)
        self.fun3d_flow.initialize_data()
        interface.design_initialize()
        self.fun3d_flow.initialize_grid()

        # Set the node locations based
        for ibody, body in enumerate(bodies, 1):
            aero_X = body.get_aero_nodes()
            aero_id = body.get_aero_node_ids()
            aero_nnodes = body.get_num_aero_nodes()

            if aero_nnodes > 0:
                fun3d_aero_X = np.reshape(aero_X, (3, -1), order="F")
                interface.design_push_body_mesh(ibody, fun3d_aero_X, aero_id)
                interface.design_push_body_name(ibody, body.name)
            else:
                interface.design_push_body_mesh(ibody, [], [])
                interface.design_push_body_name(ibody, body.name)

        bcont = self.fun3d_flow.initialize_solution()
        if bcont == 0:
            if self.comm.Get_rank() == 0:
                print("Negative volume returning fail")
            return 1

        return 0

    def set_functions(self, scenario, bodies):
        """
        Set the function definitions into FUN3D using the design interface.
        Since FUNtoFEM only allows single discipline functions, the FUN3D composite
        function is the same as the component.

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The scenario. Contains the function list
        bodies: :class:`~body.Body`
            list of FUNtoFEM bodies
        """

        for function in scenario.functions:
            if function.adjoint:
                start = 1 if function.stop == -1 else function.start
                stop = 1 if function.stop == -1 else function.stop
                ftype = -1 if function.averaging else 1

                interface.design_push_composite_func(
                    function.id,
                    1,
                    start,
                    stop,
                    1.0,
                    0.0,
                    1.0,
                    function.value,
                    ftype,
                    100.0,
                    -100.0,
                )

                if function.body == -1:
                    boundary = 0
                else:
                    boundary = bodies[function.body].boundary

                # The funtofem function in FUN3D acts as any adjoint function
                # that isn't dependent on FUN3D variables
                name = (
                    function.name
                    if function.analysis_type == "aerodynamic"
                    else "funtofem"
                )

                interface.design_push_component_func(
                    function.id, 1, boundary, name, function.value, 1.0, 0.0, 1.0
                )

        return

    def set_variables(self, scenario, bodies):
        """
        Set the aerodynamic variable definitions into FUN3D using the design interface.
        FUN3D expects 6 global variables (Mach number, AOA, yaw, etc.) that are stored in the scenario.
        It also expects a set of rigid motion variables for each body that are stored in the body.
        If the body has been specific as *motion_driver(i)='funtofem'*, the rigid motion variables will
        not affect the body's movement but must be passed regardless.

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The scenario. Contains the global aerodynamic list
        bodies: :class:`~body.Body`
            list of FUNtoFEM bodies. Bodies contains unused but necessary rigid motion variables
        """

        interface.design_set_design(len(bodies), scenario.count_adjoint_functions())

        # push the global aerodynamic variables to fun3d
        for var in scenario.variables["aerodynamic"]:
            if var.id <= 6:
                interface.design_push_global_var(
                    var.id, var.active, var.value, var.lower, var.upper
                )
            elif "dynamic pressure" == var.name.lower():
                self.qinf = var.value
            elif "thermal scale" == var.name.lower():
                self.thermal_scale = var.value

        # push the push the shape and rigid motion variables
        for ibody, body in enumerate(bodies, 1):
            num = len(body.variables["shape"]) if "shape" in body.variables else 1
            interface.design_set_body(ibody, body.parameterization, num)
            if "shape" in body.variables:
                for var in body.variables["shape"]:
                    interface.design_push_body_shape_var(
                        ibody, var.id, var.active, var.value, var.lower, var.upper
                    )

            for var in body.variables["rigid_motion"]:
                interface.design_push_body_rigid_var(
                    ibody, var.id, var.name, var.active, var.value, var.lower, var.upper
                )

        return

    def get_functions(self, scenario, bodies):
        """
        Populate the scenario with the aerodynamic function values.

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The scenario
        bodies: :class:`~body.Body`
            list of FUNtoFEM bodies
        """
        for function in scenario.functions:
            if function.analysis_type == "aerodynamic":
                # the [6] index returns the value
                val = 0.0
                if self.comm.Get_rank() == 0:
                    val = interface.design_pull_composite_func(function.id)[6]
                function.value = self.comm.bcast(val, root=0)

        return

    def eval_function_gradients(self, scenario, bodies):
        """
        Populates the FUNtoFEM model with derivatives w.r.t. aerodynamic variables

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The scenario. Contains the global aerodynamic list
        bodies: :class:`~body.Body`
            list of FUNtoFEM bodies. Bodies contains unused but necessary rigid motion variables
        """

        for func, function in enumerate(scenario.functions):
            if function.adjoint:
                for var in scenario.get_active_variables():
                    if var.id <= 6:
                        deriv = interface.design_pull_global_derivative(
                            function.id, var.id
                        )
                    elif var.name.lower() == "dynamic pressure":
                        deriv = self.comm.reduce(self.dFdqinf[func])

                    function.set_gradient_component(var, deriv)

        return scenario, bodies

    def get_coordinate_derivatives(self, scenario, bodies, step):
        """
        Adds FUN3D's contribution to the aerodynamic surface coordinate derivatives.
        This is just the grid adjoint variable, $\lambda_G$.

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The scenario.
        bodies: :class:`~body.Body`
            list of FUNtoFEM bodies.
        step: int
            the time step number
        """

        nfunctions = scenario.count_adjoint_functions()
        for ibody, body in enumerate(bodies, 1):
            aero_nnodes = body.get_num_aero_nodes()

            if aero_nnodes > 0:
                # Aero solver contribution = dGdxa0^T psi_G
                # dx, dy, dz are the x, y, and z components of dG/dxA0
                dx, dy, dz = self.fun3d_adjoint.extract_grid_adjoint_product(
                    aero_nnodes, nfunctions, body=ibody
                )
                body.aero_shape_term[0::3, :nfunctions] += dx[:, :] * self.flow_dt
                body.aero_shape_term[1::3, :nfunctions] += dy[:, :] * self.flow_dt
                body.aero_shape_term[2::3, :nfunctions] += dz[:, :] * self.flow_dt

        return

    def iterate(self, scenario, bodies, step):
        """
        Forward iteration of FUN3D.
        For the aeroelastic cases, these steps are:

        #. Get the mesh movement - the bodies' surface displacements and rigid rotations.
        #. Step forward in the grid deformationa and flow solver.
        #. Set the aerodynamic forces into the body data types

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The scenario
        bodies: :class:`~body.Body`
            list of FUNtoFEM bodies.
        step: int
            the time step number
        """

        # Deform aerodynamic mesh
        for ibody, body in enumerate(bodies, 1):
            aero_disps = body.get_aero_disps(scenario)
            if "deform" in body.motion_type and aero_disps is not None:
                dx = np.asfortranarray(aero_disps[0::3])
                dy = np.asfortranarray(aero_disps[1::3])
                dz = np.asfortranarray(aero_disps[2::3])
                self.fun3d_flow.input_deformation(dx, dy, dz, body=ibody)

            # if "rigid" in body.motion_type and body.transfer is not None:
            #     transform = np.asfortranarray(body.rigid_transform)
            #     self.fun3d_flow.input_rigid_transform(transform, body=ibody)

            aero_temps = body.get_aero_temps(scenario)
            if aero_temps is not None:
                temps = np.asfortranarray(aero_temps[:]) / body.T_ref
                self.fun3d_flow.input_wall_temperature(temps, body=ibody)

        # Take a step in FUN3D
        self.comm.Barrier()
        bcont = self.fun3d_flow.iterate()
        if bcont == 0:
            if self.comm.Get_rank() == 0:
                print("Negative volume returning fail")
            fail = 1
            os.chdir(self.root_dir)
            return fail

        for ibody, body in enumerate(bodies, 1):
            # Compute the aerodynamic nodes on the body
            aero_loads = body.get_aero_loads(scenario)
            aero_nnodes = body.get_num_aero_nodes()
            if aero_loads is not None and aero_nnodes > 0:
                aero_nnodes = body.get_num_aero_nodes()
                fx, fy, fz = self.fun3d_flow.extract_forces(aero_nnodes, body=ibody)

                # Set the dimensional values of the forces
                aero_loads[0::3] = self.qinf * fx[:]
                aero_loads[1::3] = self.qinf * fy[:]
                aero_loads[2::3] = self.qinf * fz[:]

            # Compute the heat flux on the body
            heat_flux = body.get_aero_heat_flux(scenario)
            if heat_flux is not None and aero_nnodes > 0:
                # Extract the components of the heat flux and magnitude (along the unit norm)
                cqx, cqy, cqz, cq_mag = self.fun3d_flow.extract_heat_flux(
                    aero_nnodes, body=ibody
                )

                # Set the dimensional values of the normal component of the heat flux
                heat_flux[:] = self.thermal_scale * cq_mag[:]

        return 0

    def post(self, scenario, bodies):
        """
        Calls FUN3D post to save history files, deallocate memory etc.
        Then moves back to the problem's root directory

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The scenario
        bodies: :class:`~body.Body`
            list of FUNtoFEM bodies.
        first_pass: bool
            Set to true during instantiation
        """

        self.fun3d_flow.post()
        os.chdir(self.root_dir)

        return

    def initialize_adjoint(self, scenario, bodies):
        """
        Changes the directory to ./`scenario.name`/Adjoint, then
        initializes the FUN3D adjoint solver.

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The scenario that needs to be initialized
        bodies: :class:`~body.Body`
            list of FUNtoFEM bodies to either get surface meshes from

        Returns
        -------
        fail: int
            If the grid deformation failed, the intiialization will return 1
        """

        adjoint_dir = os.path.join(self.fun3d_dir, scenario.name, "Adjoint")
        os.chdir(adjoint_dir)

        if scenario.steady:
            # Initialize FUN3D adjoint - special order for static adjoint
            if self.adjoint_options is None:
                options = {"getgrad": True}
            else:
                options = self.adjoint_options

            self.fun3d_adjoint.initialize_project(comm=self.comm)
            self.fun3d_adjoint.setOptions(kwargs=options)
            self.fun3d_adjoint.initialize_data()
            interface.design_initialize()
            for ibody, body in enumerate(bodies, 1):
                aero_X = body.get_aero_nodes()
                aero_id = body.get_aero_node_ids()
                aero_nnodes = body.get_num_aero_nodes()
                if aero_nnodes > 0:
                    fun3d_aero_X = np.reshape(aero_X, (3, -1), order="F")
                    interface.design_push_body_mesh(ibody, fun3d_aero_X, aero_id)
                    interface.design_push_body_name(ibody, body.name)
                else:
                    interface.design_push_body_mesh(ibody, [], [])
                    interface.design_push_body_name(ibody, body.name)
            self.fun3d_adjoint.initialize_grid()
            self.fun3d_adjoint.set_up_moving_body()
            self.fun3d_adjoint.initialize_funtofem_adjoint()

            # Deform the aero mesh before finishing FUN3D initialization
            for ibody, body in enumerate(bodies, 1):
                aero_disps = body.get_aero_disps(scenario)
                if aero_disps is not None:
                    dx = np.asfortranarray(aero_disps[0::3])
                    dy = np.asfortranarray(aero_disps[1::3])
                    dz = np.asfortranarray(aero_disps[2::3])
                    self.fun3d_adjoint.input_deformation(dx, dy, dz, body=ibody)

                aero_temps = body.get_aero_temps(scenario)
                if body.thermal_transfer is not None:
                    temps = np.asfortranarray(aero_temps[:]) / body.T_ref
                    self.fun3d_adjoint.input_wall_temperature(temps, body=ibody)

            self.fun3d_adjoint.initialize_solution()
        else:
            if self.adjoint_options is None:
                options = {"timedep_adj_frozen": True}
            else:
                options = self.adjoint_options

            self.fun3d_adjoint.initialize_project(comm=self.comm)
            self.fun3d_adjoint.setOptions(kwargs=options)
            self.fun3d_adjoint.initialize_data()
            interface.design_initialize()
            for ibody, body in enumerate(bodies, 1):
                aero_X = body.get_aero_nodes()
                aero_id = body.get_aero_node_ids()
                aero_nnodes = body.get_num_aero_nodes()
                if aero_nnodes > 0:
                    fun3d_aero_X = np.reshape(aero_X, (3, -1), order="F")
                    interface.design_push_body_mesh(ibody, fun3d_aero_X, aero_id)
                    interface.design_push_body_name(ibody, body.name)
                else:
                    interface.design_push_body_mesh(ibody, [], [])
                    interface.design_push_body_name(ibody, body.name)
            self.fun3d_adjoint.initialize_grid()
            self.fun3d_adjoint.initialize_solution()

        self.dFdqinf = np.zeros(len(scenario.functions), dtype=TransferScheme.dtype)
        self.dHdq = np.zeros(len(scenario.functions), dtype=TransferScheme.dtype)

        return 0

    def iterate_adjoint(self, scenario, bodies, step):
        """
        Adjoint iteration of FUN3D.
        For the aeroelastic cases, these steps are:

        #. Get the force adjoint from the body data structures
        #. Step in the flow and grid adjoint solvers
        #. Set the grid adjoint into the body data structures

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The scenario
        bodies: :class:`~body.Body`
            list of FUNtoFEM bodies.
        step: int
            the forward time step number
        """

        fail = 0
        rstep = scenario.steps - step + 1
        if scenario.steady:
            rstep = step

        nfuncs = scenario.count_adjoint_functions()
        for ibody, body in enumerate(bodies, 1):
            # Get the adjoint Jacobian product for the aerodynamic loads
            aero_loads_ajp = body.get_aero_loads_ajp(scenario)
            if aero_loads_ajp is not None:
                aero_nnodes = body.get_num_aero_nodes()
                psi_F = -aero_loads_ajp

                dtype = TransferScheme.dtype
                lam_x = np.zeros((aero_nnodes, nfuncs), dtype=dtype)
                lam_y = np.zeros((aero_nnodes, nfuncs), dtype=dtype)
                lam_z = np.zeros((aero_nnodes, nfuncs), dtype=dtype)

                for func in range(nfuncs):
                    lam_x[:, func] = self.qinf * psi_F[0::3, func] / self.flow_dt
                    lam_y[:, func] = self.qinf * psi_F[1::3, func] / self.flow_dt
                    lam_z[:, func] = self.qinf * psi_F[2::3, func] / self.flow_dt

                self.fun3d_adjoint.input_force_adjoint(lam_x, lam_y, lam_z, body=ibody)

                # Get the aero loads
                aero_loads = body.get_aero_loads(scenario)

                # Add the contributions to the derivative of the dynamic pressure
                for func in range(nfuncs):
                    # get contribution to dynamic pressure derivative
                    if scenario.steady and ibody == 1:
                        self.dFdqinf[func] = 0.0
                    if step > 0:
                        self.dFdqinf[func] -= (
                            np.dot(aero_loads, psi_F[:, func]) / self.qinf
                        )

            # Get the adjoint Jacobian products for the aero heat flux
            aero_flux_ajp = body.get_aero_heat_flux_ajp(scenario)
            if aero_flux_ajp is not None:
                # Solve the aero heat flux integration adjoint
                # dH/dhA^{T} * psi_H = - dQ/dhA^{T} * psi_Q = - aero_flux_ajp
                psi_H = -aero_flux_ajp

                dtype = TransferScheme.dtype
                lam_x = np.zeros((aero_nnodes, nfuncs), dtype=dtype)
                lam_y = np.zeros((aero_nnodes, nfuncs), dtype=dtype)
                lam_z = np.zeros((aero_nnodes, nfuncs), dtype=dtype)
                lam = np.zeros((aero_nnodes, nfuncs), dtype=dtype)

                for func in range(nfuncs):
                    lam[:, func] = self.thermal_scale * psi_H[:, func] / self.flow_dt

                self.fun3d_adjoint.input_heat_flux_adjoint(
                    lam_x, lam_y, lam_z, lam, body=ibody
                )

                for func in range(nfuncs):
                    if scenario.steady and ibody == 1:
                        self.dHdq[func] = 0.0
                    if step > 0:
                        self.dHdq[func] -= (
                            np.dot(body.aero_heat_flux_mag, psi_H[:, func])
                            / self.thermal_scale
                        )

            if "rigid" in body.motion_type:
                self.fun3d_adjoint.input_rigid_transform(
                    body.rigid_transform, body=ibody
                )

        # Update the aerodynamic and grid adjoint variables (Note: step starts at 1
        # in FUN3D)
        self.fun3d_adjoint.iterate(rstep)

        for ibody, body in enumerate(bodies, 1):
            # Extract aero_disps_ajp = dG/du_A^T psi_G from FUN3D
            aero_disps_ajp = body.get_aero_disps_ajp(scenario)
            aero_nnodes = body.get_num_aero_nodes()
            if aero_disps_ajp is not None:
                lam_x, lam_y, lam_z = self.fun3d_adjoint.extract_grid_adjoint_product(
                    aero_nnodes, nfuncs, body=ibody
                )

                for func in range(nfuncs):
                    aero_disps_ajp[0::3, func] = lam_x[:, func] * self.flow_dt
                    aero_disps_ajp[1::3, func] = lam_y[:, func] * self.flow_dt
                    aero_disps_ajp[2::3, func] = lam_z[:, func] * self.flow_dt

            # Extract aero_temps_ajp = dA/dt_A^{T} * psi_A from FUN3D
            aero_temps_ajp = body.get_aero_temps_ajp(scenario)
            if aero_temps_ajp is not None:
                lam_t = self.fun3d_adjoint.extract_thermal_adjoint_product(
                    aero_nnodes, nfuncs, body=ibody
                )

                scale = self.flow_dt / body.T_ref
                for func in range(nfuncs):
                    aero_temps_ajp[:, func] = scale * lam_t[:, func]

            if "rigid" in body.motion_type:
                body.dGdT = (
                    self.fun3d_adjoint.extract_rigid_adjoint_product(nfuncs)
                    * self.flow_dt
                )

        return fail

    def post_adjoint(self, scenario, bodies):
        """
        Calls post fo the adjoint solver in FUN3D.
        Then moves back to the problem's root directory

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The scenario
        bodies: :class:`~body.Body`
            list of FUNtoFEM bodies.
        """

        # solve the initial condition adjoint
        self.fun3d_adjoint.post()
        os.chdir(self.root_dir)

    def set_states(self, scenario, bodies, step):
        """
        Loads the saved aerodynamic forces for the time dependent adjoint

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The scenario
        bodies: :class:`~body.Body`
            list of FUNtoFEM bodies.
        step: int
            the time step number
        """

        return

    def step_pre(self, scenario, bodies, step):
        self.fun3d_flow.step_pre(step)
        return 0

    def step_solver(self, scenario, bodies, step, fsi_subiter):
        """
        Forward iteration of FUN3D.
        For the aeroelastic cases, these steps are:

        #. Get the mesh movement - the bodies' surface displacements and rigid rotations.
        #. Step forward in the grid deformationa and flow solver.
        #. Set the aerodynamic forces into the body data types

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The scenario
        bodies: :class:`~body.Body`
            list of FUNtoFEM bodies.
        step: int
            the time step number
        """

        # Deform aerodynamic mesh
        for ibody, body in enumerate(bodies, 1):
            if (
                "deform" in body.motion_type
                and body.aero_nnodes > 0
                and body.transfer is not None
            ):
                dx = np.asfortranarray(body.aero_disps[scenario.id][0::3])
                dy = np.asfortranarray(body.aero_disps[scenario.id][1::3])
                dz = np.asfortranarray(body.aero_disps[scenario.id][2::3])
                self.fun3d_flow.input_deformation(dx, dy, dz, body=ibody)
            if "rigid" in body.motion_type and body.transfer is not None:
                self.fun3d_flow.input_rigid_transform(body.rigid_transform, body=ibody)
            if body.thermal_transfer is not None:
                temps = np.asfortranarray(body.aero_temps[:]) / body.T_ref
                self.fun3d_flow.input_wall_temperature(temps, body=ibody)

        # Take a step in FUN3D
        self.comm.Barrier()
        bcont = self.fun3d_flow.step_solver()
        if bcont == 0:
            if self.comm.Get_rank() == 0:
                print("Negative volume returning fail")
            fail = 1
            os.chdir(self.root_dir)
            return fail

        # Pull out the forces from FUN3D
        for ibody, body in enumerate(bodies, 1):
            if body.aero_nnodes > 0:
                if body.transfer is not None:
                    fx, fy, fz = self.fun3d_flow.extract_forces(
                        body.aero_nnodes, body=ibody
                    )
                    body.aero_loads = np.zeros(
                        3 * body.aero_nnodes, dtype=TransferScheme.dtype
                    )
                    body.aero_loads[scenario.id][0::3] = fx[:]
                    body.aero_loads[scenario.id][1::3] = fy[:]
                    body.aero_loads[scenario.id][2::3] = fz[:]

                if body.thermal_transfer is not None:
                    cqx, cqy, cqz, cq_mag = self.fun3d_flow.extract_heat_flux(
                        body.aero_nnodes, body=ibody
                    )
                    body.aero_heat_flux = np.zeros(
                        3 * body.aero_nnodes, dtype=TransferScheme.dtype
                    )
                    body.aero_heat_flux_mag = np.zeros(
                        body.aero_nnodes, dtype=TransferScheme.dtype
                    )
                    body.aero_heat_flux[0::3] = self.thermal_scale * cqx[:]
                    body.aero_heat_flux[1::3] = self.thermal_scale * cqy[:]
                    body.aero_heat_flux[2::3] = self.thermal_scale * cqz[:]
                    body.aero_heat_flux_mag[:] = self.thermal_scale * cq_mag[:]
        return 0

    def step_post(self, scenario, bodies, step):
        self.fun3d_flow.step_post(step)

        # save this steps forces for the adjoint
        self.force_hist[scenario.id][step] = {}
        for ibody, body in enumerate(bodies, 1):
            if body.transfer is not None:
                self.force_hist[scenario.id][step][ibody] = body.aero_loads.copy()

            if body.thermal_transfer is not None:
                self.heat_flux_hist[scenario.id][step][
                    ibody
                ] = body.aero_heat_flux.copy()
                self.heat_flux_mag_hist[scenario.id][step][
                    ibody
                ] = body.aero_heat_flux_mag.copy()
                self.aero_temps_hist[scenario.id][step][ibody] = body.aero_temps.copy()

        return 0
