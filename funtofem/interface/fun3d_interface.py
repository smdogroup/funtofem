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

__all__ = ["Fun3dInterface"]

import numpy as np
import os, sys, importlib, shutil
from fun3d.solvers import Flow, Adjoint
from fun3d import interface
from funtofem import TransferScheme
from ._solver_interface import SolverInterface
from .utils.general_utils import real_norm, imag_norm


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
        fun3d_dir=None,
        forward_options=None,
        adjoint_options=None,
        auto_coords=True,
        coord_test_override=False,
        debug=False,
        forward_tolerance=1e-9,
        adjoint_tolerance=1e-9,
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
            FUNtoFEM model. This is used to loop over different scenarios and bodies for the flow analysis.
        fun3d_dir: path
            location of the FUN3D directory which holds sub-dirs of scenarios (super-directory of each scenario)
        forward_options: dict
            list of forward options passed into FUN3D f2py objects - see unsteady FUN3D-TACS examples
        adjoint_options
            list of adjoint options passed into FUN3D f2py objects - see unsteady FUN3D-TACS examples
        auto_coords: bool
            whether the aerodynamic coordinates of FUN3D are pulled into the FUNtoFEM body class upon instantiation or not.
            if not, then the _initialize_body_nodes() is called later (after the new aero mesh is built)
        coord_test_override: bool
            override the aero displacements in F2F to add fixed displacements for mesh morphing coordinate derivative tests
        debug: bool
            whether to print debug statements or not such as the real/imag norms of state vectors in FUN3D
        """

        self.comm = comm
        self.model = model

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

        # dynamic pressure derivative term
        self.dFdqinf = []

        # heat flux
        self.thermal_scale = 1.0  # = 1/2 * rho_inf * (V_inf)^3
        self.dHdq = []

        # fun3d residual data
        self._forward_done = False
        self._forward_resid = None
        self._adjoint_done = False
        self._adjoint_resid = None

        self.forward_tolerance = forward_tolerance
        self.adjoint_tolerance = adjoint_tolerance

        # coordinate derivative testing option
        self._coord_test_override = coord_test_override
        self._aero_X_orig = None

        # set debug flag
        self._debug = debug
        if self.comm.rank != 0:
            self._debug = False

        # Initialize the nodes associated with the bodies
        self.auto_coords = auto_coords
        if auto_coords:
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

        # store the initial aero coordinates
        self._aeroX_orig = body.get_aero_nodes().copy()

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

        if self._debug:
            print(
                f"Comm {self.comm.Get_rank()} check at the start of fun3d_interface:initialize."
            )

        # copy the *_body1.dat file for fun3d mesh morphing from the Fun3dAim folder to the scenario folder
        # if mesh morphing is online
        if self.model.flow is not None:
            morph_flag = self.model.flow.mesh_morph
            if morph_flag and self.comm.rank == 0:
                src = self.model.flow.mesh_morph_filepath
                dest = os.path.join(
                    self.root_dir, flow_dir, self.model.flow.mesh_morph_filename
                )
                shutil.copy2(src, dest)

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

        if self._debug:
            print(
                f"Comm {self.comm.Get_rank()} check after initialize_grid in fun3d_interface:initialize."
            )

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

        # turn on mesh morphing with Fun3dAim if the Fun3dModel has it on
        if self.model.flow is not None:
            self.fun3d_flow.set_mesh_morph(self.model.flow.mesh_morph)

        bcont = self.fun3d_flow.initialize_solution()
        if bcont == 0:
            if self.comm.Get_rank() == 0:
                print("Negative volume returning fail")
            return 1

        # update FUNtoFEM xA0 coords from FUN3D if doing mesh morphing
        if self.model.flow is not None:
            if self.model.flow.mesh_morph:
                for ibody, body in enumerate(bodies, 1):
                    aero_X = body.get_aero_nodes()
                    aero_nnodes = body.get_num_aero_nodes()

                    if aero_nnodes > 0:
                        x, y, z = interface.extract_surface(aero_nnodes, body=ibody)

                        aero_X[0::3] = x[:]
                        aero_X[1::3] = y[:]
                        aero_X[2::3] = z[:]

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
                unsteady = not (scenario.steady)
                if function.analysis_type != "aerodynamic":
                    start = 1
                    stop = 1
                else:
                    start = 1 if function.start is None else function.start
                    if unsteady:
                        # default aero function to include all time steps for the unsteady case
                        stop = (
                            scenario.steps if function.stop is None else function.stop
                        )
                    else:
                        stop = 1 if function.stop is None else function.stop

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
                scenario.qinf = var.value
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

    def get_function_gradients(self, scenario, bodies):
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

                    # function.set_gradient_component(var, deriv)
                    function.add_gradient_component(var, deriv)

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

            if aero_nnodes > 0 and step > 0:
                # Aero solver contribution = dGdxa0^T psi_G
                # dx, dy, dz are the x, y, and z components of dG/dxA0
                (
                    dGdxa0_x,
                    dGdxa0_y,
                    dGdxa0_z,
                ) = self.fun3d_adjoint.extract_grid_coord_adjoint_product(
                    aero_nnodes, nfunctions, body=ibody
                )
                aero_shape_term = body.get_aero_coordinate_derivatives(scenario)
                for ifunc in range(nfunctions):
                    aero_shape_term[0::3, ifunc] += (
                        dGdxa0_x[:, ifunc] * scenario.flow_dt
                    )
                    aero_shape_term[1::3, ifunc] += (
                        dGdxa0_y[:, ifunc] * scenario.flow_dt
                    )
                    aero_shape_term[2::3, ifunc] += (
                        dGdxa0_z[:, ifunc] * scenario.flow_dt
                    )

        return

    def conditioner_iterate(self, scenario, bodies, step):
        """
        flow solver preconditioner iterations for aerothermal and aerothermoelastic analysis
        to solve temperature profiles to stagnation first
        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The scenario
        bodies: :class:`~body.Body`
            list of FUNtoFEM bodies.
        step: int
            the time step number
        """

        # Take a step in FUN3D
        self.comm.Barrier()
        bcont = self.fun3d_flow.iterate()
        if bcont == 0:
            if self.comm.Get_rank() == 0:
                print("Negative volume returning fail")
            fail = 1
            os.chdir(self.root_dir)
            return fail

        return 0

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
            aero_disps = body.get_aero_disps(
                scenario, time_index=step, add_dxa0=self._coord_test_override
            )
            aero_nnodes = body.get_num_aero_nodes()
            deform = "deform" in body.motion_type
            if deform and aero_disps is not None and aero_nnodes > 0:
                dx = np.asfortranarray(aero_disps[0::3])
                dy = np.asfortranarray(aero_disps[1::3])
                dz = np.asfortranarray(aero_disps[2::3])
                self.fun3d_flow.input_deformation(dx, dy, dz, body=ibody)

            # if "rigid" in body.motion_type and body.transfer is not None:
            #     transform = np.asfortranarray(body.rigid_transform)
            #     self.fun3d_flow.input_rigid_transform(transform, body=ibody)

            aero_temps = body.get_aero_temps(scenario, time_index=step)
            if aero_temps is not None and aero_nnodes > 0:
                # Nondimensionalize by freestream temperature
                temps = np.asfortranarray(aero_temps[:]) / scenario.T_inf
                self.fun3d_flow.input_wall_temperature(temps, body=ibody)

            if self._debug:
                struct_disps = body.get_struct_disps(scenario, time_index=step - 1)
                struct_loads = body.get_struct_loads(scenario, time_index=step - 1)
                print(f"========================================")
                print(f"Inside fun3d_interface:iterate, step: {step}")
                if struct_loads is not None:
                    print(f"norm of real struct_loads: {real_norm(struct_loads)}")
                    print(f"norm of imag struct_loads: {imag_norm(struct_loads)}")
                if struct_disps is not None:
                    print(f"norm of real struct_disps: {real_norm(struct_disps)}")
                    print(f"norm of imag struct_disps: {imag_norm(struct_disps)}")
                if aero_disps is not None:
                    print(f"norm of real aero_disps: {real_norm(aero_disps)}")
                    print(f"norm of imaginary aero_disps: {imag_norm(aero_disps)}")
                print(f"========================================\n", flush=True)

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
            aero_loads = body.get_aero_loads(scenario, time_index=step)
            aero_nnodes = body.get_num_aero_nodes()
            if aero_loads is not None and aero_nnodes > 0:
                fx, fy, fz = self.fun3d_flow.extract_forces(aero_nnodes, body=ibody)

                # Set the dimensional values of the forces
                aero_loads[0::3] = scenario.qinf * fx[:]
                aero_loads[1::3] = scenario.qinf * fy[:]
                aero_loads[2::3] = scenario.qinf * fz[:]

            if self._debug:
                print(f"========================================")
                print(f"Inside fun3d_interface:iterate, after iterate, step: {step}")
                if aero_loads is not None:
                    print(f"norm of real aero_loads: {real_norm(aero_loads)}")
                    print(f"norm of imaginary aero_loads: {imag_norm(aero_loads)}")
                print(f"========================================\n", flush=True)

            # Compute the heat flux on the body
            # FUN3D is nondimensional, it doesn't output a heat flux (which can't be scaled linearly).
            # Instead, FUN3D can directly output a temperature gradient at the wall. We then compute
            # the heat flux manually by calculating viscosity based on aero temps to get thermal conductivity,
            # and then take the product of thermal conductivity and area-weighted temperature gradient.
            heat_flux = body.get_aero_heat_flux(scenario, time_index=step)

            if heat_flux is not None and aero_nnodes > 0:
                # Extract the area-weighted temperature gradient normal to the wall (along the unit norm)
                dTdn = self.fun3d_flow.extract_cqa(aero_nnodes, body=ibody)

                dTdn_dim = dTdn * scenario.T_inf

                aero_temps = body.get_aero_temps(scenario)
                k_dim = scenario.get_thermal_conduct(aero_temps)

                # actually a heating rate integral(heat_flux) over the area
                heat_flux[:] = dTdn_dim[:] * k_dim[:]

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

        # report warning if flow residual too large
        resid = self.get_forward_residual(step=scenario.steps, all=True)  # step=scenario.steps
        if self.comm.rank == 0:
            print(f"Forward residuals = {resid}")
        self._forward_done = True
        self._forward_resid = resid
        if abs(np.linalg.norm(resid).real) > self.forward_tolerance:
            if self.comm.rank == 0:
                print(
                    f"\tWarning: fun3d forward flow residual = {resid} > {self.forward_tolerance:.2e}, is rather large..."
                )

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

        # copy the *_body1.dat file for fun3d mesh morphing from the Fun3dAim folder to the scenario folder
        # if mesh morphing is online
        if self.model.flow is not None:
            morph_flag = self.model.flow.mesh_morph
            if morph_flag and self.comm.rank == 0:
                src = self.model.flow.mesh_morph_filepath
                dest = os.path.join(
                    self.root_dir, adjoint_dir, self.model.flow.mesh_morph_filename
                )
                shutil.copy2(src, dest)

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

            # turn on mesh morphing with Fun3dAim if the Fun3dModel has it on
            if self.model.flow is not None:
                self.fun3d_adjoint.set_mesh_morph(self.model.flow.mesh_morph)

            # Deform the aero mesh before finishing FUN3D initialization
            for ibody, body in enumerate(bodies, 1):
                aero_disps = body.get_aero_disps(
                    scenario, add_dxa0=self._coord_test_override
                )
                aero_nnodes = body.get_num_aero_nodes()
                if aero_disps is not None and aero_nnodes > 0:
                    dx = np.asfortranarray(aero_disps[0::3])
                    dy = np.asfortranarray(aero_disps[1::3])
                    dz = np.asfortranarray(aero_disps[2::3])
                    self.fun3d_adjoint.input_deformation(dx, dy, dz, body=ibody)

                aero_temps = body.get_aero_temps(scenario)
                if body.thermal_transfer is not None:
                    # Nondimensionalize by freestream temperature
                    temps = np.asfortranarray(aero_temps[:]) / scenario.T_inf
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

            # turn on mesh morphing with Fun3dAim if the Fun3dModel has it on
            if self.model.flow is not None:
                self.fun3d_adjoint.set_mesh_morph(self.model.flow.mesh_morph)

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
            aero_nnodes = body.get_num_aero_nodes()
            if aero_loads_ajp is not None and aero_nnodes > 0:
                aero_nnodes = body.get_num_aero_nodes()
                psi_F = -aero_loads_ajp

                dtype = TransferScheme.dtype
                lam_x = np.zeros((aero_nnodes, nfuncs), dtype=dtype)
                lam_y = np.zeros((aero_nnodes, nfuncs), dtype=dtype)
                lam_z = np.zeros((aero_nnodes, nfuncs), dtype=dtype)

                for func in range(nfuncs):
                    lam_x[:, func] = (
                        scenario.qinf * psi_F[0::3, func] / scenario.flow_dt
                    )
                    lam_y[:, func] = (
                        scenario.qinf * psi_F[1::3, func] / scenario.flow_dt
                    )
                    lam_z[:, func] = (
                        scenario.qinf * psi_F[2::3, func] / scenario.flow_dt
                    )

                    if self._debug:
                        print(f"========================================")
                        print(f"Inside fun3d_interface:iterate_adjoint, step: {step}")
                        print(f"func: {func}")
                        print(
                            f"norm of real psi_F: {real_norm(lam_x[:, func])}, {real_norm(lam_y[:, func])}, {real_norm(lam_z[:, func])}"
                        )
                        print(
                            f"norm of imaginary psi_F: {imag_norm(lam_x[:, func])}, {imag_norm(lam_y[:, func])}, {imag_norm(lam_z[:, func])}"
                        )
                        print(f"========================================\n", flush=True)

                self.fun3d_adjoint.input_force_adjoint(lam_x, lam_y, lam_z, body=ibody)

                # Get the aero loads
                aero_loads = body.get_aero_loads(scenario, time_index=step)

                # Add the contributions to the derivative of the dynamic pressure
                for func in range(nfuncs):
                    # get contribution to dynamic pressure derivative
                    if scenario.steady and ibody == 1:
                        self.dFdqinf[func] = 0.0
                    if step > 0:
                        self.dFdqinf[func] -= (
                            np.dot(aero_loads, psi_F[:, func]) / scenario.qinf
                        )
                    if self._debug:
                        print(f"========================================")
                        print(
                            f"Inside fun3d_interface:iterate_adjoint after dFdqinf contribution, step: {step}"
                        )
                        print(f"func: {func}")
                        if self.dFdqinf[func] is not None:
                            print(
                                f"norm of real dFdqinf: {real_norm(self.dFdqinf[func])}"
                            )
                            print(
                                f"norm of imaginary dFdqinf: {imag_norm(self.dFdqinf[func])}"
                            )
                        else:
                            print(f"dFdqinf[func] is NoneType")
                        print(f"========================================\n", flush=True)

            # Get the adjoint Jacobian products for the aero heat flux
            aero_flux_ajp = body.get_aero_heat_flux_ajp(scenario)
            aero_nnodes = body.get_num_aero_nodes()
            aero_flux = body.get_aero_heat_flux(scenario, time_index=step)
            aero_temps = body.get_aero_temps(scenario, time_index=step)

            if aero_flux_ajp is not None and aero_nnodes > 0:
                # Solve the aero heat flux integration adjoint
                # dH/dhA^{T} * psi_H = - dQ/dhA^{T} * psi_Q = - aero_flux_ajp
                psi_H = -aero_flux_ajp

                # new viscosity law effect
                k_dim = scenario.get_thermal_conduct(aero_temps)

                dtype = TransferScheme.dtype
                lam = np.zeros((aero_nnodes, nfuncs), dtype=dtype)

                scale = scenario.T_inf / scenario.flow_dt

                for func in range(nfuncs):
                    lam[:, func] = scale * psi_H[:, func] * k_dim[:]

                self.fun3d_adjoint.input_cqa_adjoint(lam, body=ibody)

                for func in range(nfuncs):
                    if scenario.steady and ibody == 1:
                        self.dHdq[func] = 0.0
                    if step > 0:
                        self.dHdq[func] -= (
                            np.dot(aero_flux, psi_H[:, func]) / self.thermal_scale
                        )

            # if "rigid" in body.motion_type:
            #     self.fun3d_adjoint.input_rigid_transform(
            #         body.rigid_transform, body=ibody
            #     )

        # Update the aerodynamic and grid adjoint variables (Note: step starts at 1
        # in FUN3D)
        self.fun3d_adjoint.iterate(rstep)

        for ibody, body in enumerate(bodies, 1):
            # Extract aero_disps_ajp = dG/du_A^T psi_G from FUN3D
            aero_disps_ajp = body.get_aero_disps_ajp(scenario)
            aero_nnodes = body.get_num_aero_nodes()
            if aero_disps_ajp is not None and aero_nnodes > 0:
                lam_x, lam_y, lam_z = self.fun3d_adjoint.extract_grid_adjoint_product(
                    aero_nnodes, nfuncs, body=ibody
                )

                for func in range(nfuncs):
                    aero_disps_ajp[0::3, func] = lam_x[:, func] * scenario.flow_dt
                    aero_disps_ajp[1::3, func] = lam_y[:, func] * scenario.flow_dt
                    aero_disps_ajp[2::3, func] = lam_z[:, func] * scenario.flow_dt

                    if self._debug:
                        print(f"========================================")
                        print(f"Inside fun3d_interface:iterate_adjoint, step: {step}")
                        print(f"func: {func}")
                        print(
                            f"norm of real psi_D: {real_norm(lam_x[:, func])}, {real_norm(lam_y[:, func])}, {real_norm(lam_z[:, func])}"
                        )
                        print(
                            f"norm of imaginary psi_D: {imag_norm(lam_x[:, func])}, {imag_norm(lam_y[:, func])}, {imag_norm(lam_z[:, func])}"
                        )
                        print(f"========================================\n", flush=True)

            # Extract aero_temps_ajp = dA/dt_A^{T} * psi_A from FUN3D
            aero_temps_ajp = body.get_aero_temps_ajp(scenario)
            aero_nnodes = body.get_num_aero_nodes()

            if aero_temps_ajp is not None and aero_nnodes > 0:
                # additional terms
                dkdtA = scenario.get_thermal_conduct_deriv(aero_temps)

                lam_t = self.fun3d_adjoint.extract_thermal_adjoint_product(
                    aero_nnodes, nfuncs, body=ibody
                )

                scale = scenario.flow_dt / scenario.T_inf
                for func in range(nfuncs):
                    aero_temps_ajp[:, func] = scale * lam_t[:, func]

                    # contribution from viscosity in adjoint path
                    aero_temps_ajp[:, func] += (
                        scenario.flow_dt
                        * aero_flux_ajp[:, func]
                        * (aero_flux[:] / k_dim[:])
                        * dkdtA[:]
                    )

            # if "rigid" in body.motion_type:
            #     body.dGdT = (
            #         self.fun3d_adjoint.extract_rigid_adjoint_product(nfuncs)
            #         * scenario.flow_dt
            #     )

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

        # report warning if flow residual too large
        resid = self.get_adjoint_residual(step=scenario.steps, all=True)
        if self.comm.rank == 0:
            print(f"Adjoint residuals = {resid}")
        self._adjoint_done = True
        self._adjoint_resid = resid
        if abs(np.linalg.norm(resid).real) > self.adjoint_tolerance:
            if self.comm.rank == 0:
                print(
                    f"\tWarning fun3d adjoint residual = {resid} > {self.adjoint_tolerance:.2e}, is rather large..."
                )

        # solve the initial condition adjoint
        self.fun3d_adjoint.post()
        os.chdir(self.root_dir)
        return

    def get_forward_residual(self, step=0, all=False):
        """
        Returns L2 norm of scalar residual norms for each flow state
        L2norm([R1,...,R6])

        Parameters
        ----------
        step: int
            the time step number
        all: bool
            whether to return a list of all residuals or just a scalar
        """
        if not self._forward_done:
            residuals = self.fun3d_flow.get_flow_rms_residual(step)
        else:
            residuals = self._forward_resid
        
        if all:
            return residuals
        else:
            return np.linalg.norm(residuals)

    def get_adjoint_residual(self, step=0, all=False):
        """
        Returns L2 norm of list of scalar adjoint residuals L2norm([R1,...,R6])

        Parameters
        ----------
        step: int
            the time step number
        all: bool
            whether to return a list of all residuals or a scalar
        """
        if not self._adjoint_done:
            residuals = self.fun3d_adjoint.get_flow_rms_residual(step)
            return np.linalg.norm(residuals)
        else:
            residuals = self._adjoint_resid

        if all:
            return residuals
        else:
            return np.linalg.norm(residuals)

    def set_states(self, scenario, bodies, step):
        """
        Loads the saved aerodynamic displacements and temperatures
        for the time dependent adjoint.

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
            aero_disps = body.get_aero_disps(
                scenario, time_index=step, add_dxa0=self._coord_test_override
            )
            aero_nnodes = body.get_num_aero_nodes()
            deform = "deform" in body.motion_type
            if deform and aero_disps is not None and aero_nnodes > 0:
                dx = np.asfortranarray(aero_disps[0::3])
                dy = np.asfortranarray(aero_disps[1::3])
                dz = np.asfortranarray(aero_disps[2::3])
                self.fun3d_flow.input_deformation(dx, dy, dz, body=ibody)

            # if "rigid" in body.motion_type and body.transfer is not None:
            #     transform = np.asfortranarray(body.rigid_transform)
            #     self.fun3d_flow.input_rigid_transform(transform, body=ibody)

            aero_temps = body.get_aero_temps(scenario, time_index=step)
            if aero_temps is not None and aero_nnodes > 0:
                # Nondimensionalize by freestream temperature
                temps = np.asfortranarray(aero_temps[:]) / scenario.T_inf
                self.fun3d_flow.input_wall_temperature(temps, body=ibody)

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
                temps = np.asfortranarray(body.aero_temps[:]) / scenario.T_inf
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
                    cqx, cqy, cqz, cq_mag = self.fun3d_flow.extract_cqa(
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

    @classmethod
    def copy_real_interface(cls, fun3d_interface):
        """
        copy used for derivative testing
        drivers.solvers.make_real_flow()
        """

        # unload and reload fun3d Flow, Adjoint as real versions
        os.environ["CMPLX_MODE"] = ""
        importlib.reload(sys.modules["fun3d.interface"])

        return cls(
            comm=fun3d_interface.comm,
            model=fun3d_interface.model,
            fun3d_dir=fun3d_interface.fun3d_dir,
            auto_coords=fun3d_interface.auto_coords,
            coord_test_override=fun3d_interface._coord_test_override,
        )

    @classmethod
    def copy_complex_interface(cls, fun3d_interface):
        """
        copy used for derivative testing
        driver.solvers.make_complex_flow()
        """

        # unload and reload fun3d Flow, Adjoint as complex versions
        os.environ["CMPLX_MODE"] = "1"
        importlib.reload(sys.modules["fun3d.interface"])

        return cls(
            comm=fun3d_interface.comm,
            model=fun3d_interface.model,
            fun3d_dir=fun3d_interface.fun3d_dir,
            auto_coords=fun3d_interface.auto_coords,
            coord_test_override=fun3d_interface._coord_test_override,
        )
