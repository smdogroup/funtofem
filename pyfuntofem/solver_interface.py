#!/usr/bin/env python

# This file is part of the package FUNtoFEM for coupled aeroelastic simulation
# and design optimization.

# Copyright (C) 2015 Georgia Tech Research Corporation.
# Additional copyright (C) 2015 Kevin Jacobson, Jan Kiviaho and Graeme Kennedy.
# All rights reserved.

# FUNtoFEM is licensed under the Apache License, Version 2.0 (the "License");
# you may not use this software except in compliance with the License.
# You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np


class SolverInterface(object):
    """
    A base class to define what functions solver interfaces in FUNtoFEM need
    """

    def __init__(self, *args, **kwargs):
        """
        The solver constructor is required to set discipline node locations (either :math:`x_a` or :math:`x_s`)
        in the funtofem body class. These values are passed via a call to either
        ``body.initialize_struct_nodes(struct_X)`` or ``body.initialize_aero_nodes(aero_X)``.
        The constructor can be used flexibly for other discipline solver specific activities (e.g. solver
        instantiation, reading mesh, allocating solver data).

        Examples
        --------
        Notional aerodynamic solver implementation ``solver``:

        .. code-block:: python

            # Set aerodynamic surface meshes
            for ibody, body in enumerate(bodies):
                aero_X = solver.get_mesh(ibody)
                body.initialize_aero_nodes(aero_X)

        Notional structural solver implementation ``solver``:

        .. code-block:: python

            # Set structural meshes
            for ibody, body in enumerate(bodies):
                struct_X = solver.get_mesh(ibody)
                body.initialize_struct_nodes(struct_X)
        """
        pass

    """
    A base class to define what functions solver interfaces in FUNtoFEM need
    """

    def set_variables(self, scenario, bodies):
        """
        Set the design variables into the solver.
        The scenario and bodies objects have dictionaries of :class:`~variable.Variable` objects.
        The interface class should pick out which type of variables it needs and pass them into the solver

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        bodies: list of :class:`~body.Body` objects
            The bodies in the model

        Examples
        --------
        Structural Solver:

        .. code-block:: python

            for ibody, body in enumerate(bodies):
                for i, var in enumerate(self.variables):
                    solver.set_variable(i, var.value)

        Aerodynamic Solver:

        .. code-block:: python

            for ibody, body in enumerate(bodies):
                for i, var in enumerate(self.variables):
                    solver.set_variable(i, var.value)
        """
        pass

    def set_functions(self, scenario, bodies):
        """
        Set the function definitions into the solver.
        The scenario has a list of function objects.

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        bodies: list of :class:`~body.Body` objects
            The bodies in the model

        Examples
        --------
        Structural Solver:

        .. code-block:: python

            for func in scenario.functions:

                # Set the structural functions in
                if function.adjoint and function.analysis_type == "structural":
                    solver.set_adjoint_function(function.name, function.start, function,stop)

                # Tell the solver that an adjoint is needed, but the function is
                # not explicitly dependent on structural states
                elif function.adjoint and function.analysis_type != "structural":
                    solver.set_dummy_function()

                # Functions such as structural mass do not need and adjoint
                elif not function.adjoint and function.analysis_type == "structural":
                    solver.set_nonadjoint_function(function.name)
        """
        pass

    def get_functions(self, scenario, bodies):
        """
        Put the function values from the solver in the value attribute of the scneario's functions.
        The scenario has the list of function objects where the functions owned by this solver will be set.
        You can evaluate the functions based on the name or based on the functions set during
        :func:`~solver_interface.SolverInterface.set_functions`.
        The solver is only responsible for returning the values of functions it owns.

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        bodies: list of :class:`~body.Body` objects
            The bodies in the model

        Examples
        --------
        Structural Solver:

        .. code-block:: python

            for func in scenario.functions:

                # Set the structural functions in
                if func.analysis_type == "structural":
                    if func.name == "mass":
                        func.value = solver.evaluate_mass()
                    elif func.name == "ksfailure":
                        func.value = solver.get_ksfailure()
                    else:
                        print("Unknown structural function in get_functions")
        """
        pass

    def get_function_gradients(self, scenario, bodies):
        """
        Get the derivatives of all the functions with respect to design variables associated with this solver.

        Each solver sets the function gradients for its own variables into the function objects using either
        ``function.set_gradient(var, value)`` or ``function.add_gradient(var, vaule)``. Note that before
        this function is called, all gradient components are zeroed.

        The derivatives are stored in a dictionary in each function class. As a result, the gradients are
        stored in an unordered format. The gradients returned by ``model.get_function_gradients()`` are
        flattened into a list of lists whose order is determined by the variable list stored in the model
        class.

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        bodies: list of :class:`~body.Body` objects
            The bodies in the model

        Examples
        --------
        Aerodynamic Solver:

        .. code-block:: python

            # Make sure the gradient is consistent across all processors before setting
            # the values
            grad = self.comm.allreduce(grad)

            for i, func in enumerate(scenario.functions):
                for j, var in enumerate(self.variables):
                    func.set_derivatives(var, grad[i, j])
        """
        pass

    def get_coordinate_derivatives(self, scenario, bodies, step):
        """
        Add the solver's contributions to the coordinate derivatives for this time step or the total value
        for the steady case. The coordinate derivatives are stored in the body objects in the aero_shape_term
        and struct_shape_term attributes.

        For time dependent problems, this is called at the end of every time step during reverse marching.

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        bodies: list of :class:`~body.Body` objects
            The bodies in the model
        step: int
            The time step number

        Examples
        --------
        Aerodynamic Solver:

        .. code-block:: python

            nfunctions = scenario.count_adjoint_functions()
            for ibody, body in enumerate(bodies):
                lam_x, lam_y, lam_Z = solver.extract_coordinate_derivatives(ibody, step)

                aero_shape_term = body.get_aero_coordinate_derivatives(scenario)
                aero_shape_term[ ::3,:nfunctions] += lam_x[:,:]
                aero_shape_term[1::3,:nfunctions] += lam_y[:,:]
                aero_shape_term[2::3,:nfunctions] += lam_z[:,:]

        Structural Solver:

        .. code-block:: python

            for ibody, body in enumerate(bodies):
                # Add the derivatives to the body
                lam_x, lam_y, lam_Z = solver.extract_coordinate_derivatives(ibody, step)

                struct_shape_term = body.get_struct_coordinate_derivatives(scenario)
                struct_shape_term[ ::3,:nfunctions] += lam_x[:,:]
                struct_shape_term[1::3,:nfunctions] += lam_y[:,:]
                struct_shape_term[2::3,:nfunctions] += lam_z[:,:]
        """
        pass

    def initialize(self, scenario, bodies):
        """
        Set up anything that is necessary for a specific scenario prior to calls to ``iterate``.
        A requirement is that discipline solvers update their mesh representations to be consistent
        with the coordinates in the bodies. These coordinate locations can be obtained by
        calling ``body.get_aero_nodes`` (:math:`x_a`) or ``body.get_struct_nodes`` (:math:`x_s`),
        respectively. These coordinates may be updated as a result of design changes.

        Note, it is possible that a discipline solver mesh representation is updated by reading a mesh file
        instead of accepting the funtofem body representation. Ultimately, the requirement is that the funtofem
        body mesh representations ``body.aero_X`` and ``body.struct_X`` and their corresponding discipline solver
        mesh representations are consistent.

        The initialize step may accomodate other useful activities such as data allocation,
        setting initial conditions, etc.

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        bodies: list of :class:`~body.Body` objects
            The bodies in the model

        Examples
        --------
        Aerodynamic Solver:

        .. code-block:: python

            # Set in the new aerodynamic surface meshes
            for ibody, body in enumerate(bodies):
                aero_X = body.get_aero_nodes()
                solver.set_surface_node_locations(ibody, aero_X)

           # Initialize internal solver data in preparation for a flow solver iterations
           solver.initialize()

        Structural Solver:

        .. code-block:: python

            # Set the new structural surface mesh
            for ibody, body in enumerate(bodies):
                struct_X = body.get_struct_nodes()
                solver.set_structural_node_locations(ibody, struct_X)
        """
        return 0

    def iterate(self, scenario, bodies, step):
        """
        Iterate on the primal residual equation for the present discipline solver. Called in NLBGS solver.

        For an aerodynamic solver, this might include:

        #. Obtain displacements at aerodynamic surface nodes :math:`u_a` from funtofem body objects
        #. Deform the meshes
        #. Solve for the new aerodynamic state :math:`q`
        #. Integrate and localize aerodynamic surface forces :math:`f_a` at aerodynamic surface node locations

        For a structural solver:

        #. Obtain the forces at structural nodes :math:`f_s`
        #. Solve for new displacement state :math:`u_s`
        #. Set new structural displacements :math:`u_s`

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        bodies: list of :class:`~body.Body` objects
            The bodies in the model

        Examples
        --------
        Aerodynamic Solver:

        .. code-block:: python

            # Get the surface node locations and temperatures
            for ibody, body in enumerate(bodies):
                aero_disps = body.get_aero_disps(scenario) # May return None if not aeroelastic
                if aero_disps is not None:
                    solver.input_surface_deformation(ibody, aero_disps)

                aero_temps = body.get_aero_temps(scenario) # May return None if not aerothermal
                if aero_temps is not None:
                    solver.input_surface_temperatures(ibody, aero_temps)

            # Advance the solver
            solver.iterate()

            # Extract the aerodynamic forces at the nodes and the corresponding
            # area-weighted normal component of the heat flux
            for ibody, body in enumerate(bodies):
                aero_loads = body.get_aero_loads(scenario)
                if aero_loads is not None:
                    aero_loads[:] = solver.get_surface_forces(ibody)

                aero_flux = body.get_aero_heat_flux(scenario)
                if aero_flux is not None:
                    aero_flux[:] = solver.get_surface_heat_flux(ibody)

        Structural Solver:

        .. code-block:: python

            # Input the forces and/or heat flux to the structure
            for ibody, body in enumerate(bodies):
                struct_loads = body.get_struct_loads(scenario)
                if struct_loads is not None:
                    solver.input_forces(ibody, struct_loads)

                struct_flux = body.get_struct_heat_flux(scenario)
                if struct_flux is not None:
                    solver.input_heat_flux(ibody, struct_flux)

            # Solve the problem with the forces/heat loads
            solver.iterate()

            # Extract the displacemets
                struct_disps = body.get_struct_disps(scenario)
                if struct_disps is not None:
                    struct_disps[:] = solver.get_displacements(ibody)

                struct_temps = body.get_struct_temps(scenario)
                if struct_temps is not None:
                    struct_disps[:] = solver.get_temperatures(ibody)
        """
        return 0

    def post(self, scenario, bodies):
        """
        Perform any tasks the solver needs to do after the forward steps are complete, e.g., evaluate functions,
        post-process, deallocate unneeded memory.

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        bodies: list of :class:`~body.Body` objects
            The bodies in the model
        """
        pass

    def initialize_adjoint(self, scenario, bodies):
        """
        Perform any tasks the solver needs to do before taking adjoint steps

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        bodies: list of :class:`~body.Body` objects
            The bodies in the model
        """
        return 0

    def iterate_adjoint(self, scenario, bodies, step):
        """
        Adjoint iteration for the solver.

        For an aerodynamic solver this will include:

        #. Obtain the adjoint-Jacobian product of the surface forces from the body classes
        #. Obtain the adjoint-Jacobian product of the surface heat fluxes from the body classes
        #. Set these values into the adjoint solver
        #. Solve for the aerodynamic adjoint
        #. Solve the grid adjoint
        #. Compute the output adjoint-Jacobian products
        #. Set the adjoint-Jacobian product of the aerodynamic displacements into the body classes
        #. Set the adjoint-Jacobian product of the aerodynamic surface temperatures into the body classes

        For a structural solver this will include:

        #. Get the displacement adjoint-Jacobian product from the body classes
        #. Get the temperature adjoint-Jacobian product from the body classes
        #. Set the adjoint-Jacobian products into the right-hand-side of the structural solver
        #. Solve the structural or thermomechanical adjoint equations
        #. Compute the output adjoint-Jacobian products
        #. Set the adjoint-Jacobian product of the structural forces into the body classes
        #. Set the adjoint-Jacobian product of the structural heat flux into the body classes

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        bodies: list of :class:`~body.Body` objects
            The bodies in the model
        step: int
            The step number. Will start at the final step and march backwards to 0, the initial condition.

        Examples
        --------
        Aerodynamic Solver:

        .. code-block:: python

            # Set the inputs to the aerodynamic adjoint
            for ibody, body in enumerate(bodies):
                # Get the adjoint-Jacobian product for the aero_loads. This may be None
                # if it is not defined
                aero_loads_ajp = body.get_aero_loads_ajp(scenario)
                if aero_loads_ajp is not None:
                    # Note: This part may or may not be in the solver itself. This
                    # hypothetical code is like FUN3D where the integration adjoint
                    # terms are solved externally and then set into the solver.

                    # Solve the aero force integration adjoint equation
                    # dF/dfA^{T} * psi_{F} = - dL/dfA^{T} * psi_L = - aero_loads_ajp.
                    # Note that in this case dF/dfA = I.
                    psi_F = -aero_loads_ajp

                    for ifunc, func in enumerate(scenario.functions):
                        solver.set_aero_forces_adjoint(ibody, psi_F[:, ifunc])

                # Get the adjoint-Jacobian product for the aero_flux. This may be
                # None if it is not defined
                aero_flux_ajp = body.get_aero_heat_flux_ajp(scenario)
                if aero_flux_ajp is not None:
                    # Solve the aero heat flux integration
                    # dH/dhA^{T} * psi_H = - dQ/dhA^{T} * psi_Q = - aero_flux_ajp
                    psi_H = -aero_flux_ajp

                    for ifunc, func in enumerate(scenario.functions):
                        solver.set_aero_flux_adjoint(ibody, psi_H[:, ifunc])

            # Iterate on the adjoint equations
            solver.iterate_adjoint()

            # Extract the output adjoint-Jacobian products from the solver
            for ibody, body in enumerate(bodies):
                aero_disps_ajp = body.get_aero_disps_ajp(scenario)
                if aero_disps_ajp is not None:
                    for ifunc, func in enumerate(scenario.functions):
                        dfdxA = solver.get_disp_adjoint_product(ibody, ifunc)
                        aero_disps_ajp[:, ifunc] = dfdxA

                aero_temps_ajp = body.get_aero_temps_ajp(scenario)
                if aero_temps_ajp is not None:
                    for ifunc, func in enumerate(scenario.functions):
                        dfdtA = solver.get_temp_adjoint_product(ibody, ifunc)
                        aero_temps_ajp[:, ifunc] = dfdtA

        Structural Solver:

        .. code-block:: python

            # Outer loop over the number of functions of interest
            for ifunc, func in enumerate(scenario.functions):
                rhs[:] = -dfdu[ifunc] # Set the right-hand-side

                # Set the inputs to the structural adjoint
                for ibody, body in enumerate(bodies):
                    struct_disps_ajp = body.get_struct_disps_ajp(scenario)
                    if struct_disps_ajp is not None:
                        # Add the contributions to the adjoint right-hand-side
                        rhs[load_locations] -= struct_disps_ajp

                    struct_temps_ajp = body.get_struct_temps_ajp(scenario)
                    if struct_temps_ajp is not None:
                        rhs[temp_locations] -= struct_temps_ajp

                # Solve the adjoint
                solver.solve_adjoint(rhs)

                # Extract the load and heat-flux ajdoint-vector products
                psi_S = solver.get_adjoint()

                # Here the required adjoint vector products are
                # struct_loads_ajp = psi_S^{T} * dS/dfS
                # struct_flux_ajp = psi_S^{T} * dS/fhS
                # We use the residual S = r(u) - fS - hS
                for ibody, body in enumerate(bodies):
                    struct_loads_ajp = body.get_struct_loads_ajp(scenario)
                    if struct_loads_ajp is not None:
                        struct_loads_ajp[:] = -psi_S[load_locations]

                    struct_flux_ajp = body.get_struct_heat_flux_ajp(scenario)
                    if struct_flux_ajp is not None:
                        struct_flux_ajp[:] = -psi_S[temp_locations]
        """
        return 0

    def post_adjoint(self, scenario, bodies):
        """
        Any actions that need to be performed after completing the adjoint solve, e.g., evaluating gradients, deallocating memory, etc.
        """
        pass

    def set_states(self, scenario, bodies, step):
        """
        Load the states (aero_loads, struct_disps) associated with this step either from memory or the disk for
        the transfer scheme to linearize about. This function is called at the beginning of each adjoint
        step in time dependent problems.

        **Note: in the NLBGS algorithm the transfer scheme uses the structural displacements from the prior step.
        set_states will request the states from the previous step but then ask the structural solver to linearize
        about the current step in iterate_adjoint**

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        bodies: list of :class:`~body.Body` objects
            The bodies in the model
        step: int
            The time step number that the driver wants the states from

        Examples
        --------
        Structural Solver:

        .. code-block:: python

            # Find the step's values on the disk
            disps_hist = solver.load_state(step)

            # Set the structural displacements for the transfer scheme residuals to linearize about
            for ibody,body in enumerate(bodies):
                body.struct_disps = disps_hist[ibody]
        """
        pass

    def step_pre(self, scenario, bodies, step):
        """
        Operations before at a step in an FSI subiteration case. Called in NLBGS with FSI subiterations.

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        bodies: list of :class:`~body.Body` objects
            The bodies in the model
        step: int
            The step number. Will start at the final step and march backwards to 0, the initial condition.
        """
        return 0

    def step_solver(self, scenario, bodies, step, fsi_subiter):
        """
        Step in an FSI subiteration case. Called in NLBGS with FSI subiterations.

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        bodies: list of :class:`~body.Body` objects
            The bodies in the model
        step: int
            The step number. Will start at the final step and march backwards to 0, the initial condition.
        fsi_subiter: int
            The FSI subiteration number
        """
        return 0

    def step_post(self, scenario, bodies, step):
        """
        Operations after at a step in an FSI subiteration case. Called in NLBGS with FSI subiterations.

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        bodies: list of :class:`~body.Body` objects
            The bodies in the model
        step: int
            The step number. Will start at the final step and march backwards to 0, the initial condition.
        """
        return 0

    def test_adjoint(
        self,
        solver_type,
        scenario,
        bodies,
        step=0,
        epsilon=1e-6,
        complex_step=False,
        rtol=1e-6,
    ):
        """
        Test the adjoint implementation depending on the solver type.

        This test evaluates the consistency between the forward and adjoint modes.
        """

        if solver_type == "flow" or solver_type == "aerodynamic":
            return self._test_flow_adjoint(
                scenario,
                bodies,
                step=step,
                epsilon=epsilon,
                complex_step=complex_step,
                rtol=rtol,
            )
        elif solver_type == "structural":
            return self._test_struct_adjoint(
                scenario,
                bodies,
                step=step,
                epsilon=epsilon,
                complex_step=complex_step,
                rtol=rtol,
            )
        else:
            print("Unrecognized solver type in test_adjoint")

        # Return true if no test has been performed
        return True

    def _test_flow_adjoint(
        self, scenario, bodies, step=0, epsilon=1e-6, complex_step=False, rtol=1e-6
    ):
        """
        Test to see if the adjoint methods are implemented correctly
        """

        self.set_functions(scenario, bodies)
        self.set_variables(scenario, bodies)

        for body in bodies:
            body.initialize_variables(scenario)

        # Comptue one step of the forward solution
        self.initialize(scenario, bodies)
        self.iterate(scenario, bodies, step)
        self.post(scenario, bodies)

        # Evaluate and store the functions of interest
        fvals_init = []
        self.get_functions(scenario, bodies)
        for func in scenario.functions:
            fvals_init.append(func.value)

        # Save the output forces and heat fluxes
        aero_loads_list = []
        aero_flux_list = []
        for body in bodies:
            aero_loads = body.get_aero_loads(scenario)
            if aero_loads is not None:
                aero_loads_list.append(aero_loads.copy())

            aero_flux = body.get_aero_heat_flux(scenario)
            if aero_flux is not None:
                aero_flux_list.append(aero_flux.copy())

        # Initialize the bodies for the adjoint computation
        for body in bodies:
            body.initialize_adjoint_variables(scenario)

            aero_loads_ajp = body.get_aero_loads_ajp(scenario)
            if aero_loads_ajp is not None:
                shape = aero_loads_ajp.shape
                aero_loads_ajp[:] = np.random.uniform(size=shape).astype(body.dtype)

            aero_flux_ajp = body.get_aero_heat_flux_ajp(scenario)
            if aero_flux_ajp is not None:
                shape = aero_flux_ajp.shape
                aero_flux_ajp[:] = np.random.uniform(size=shape).astype(body.dtype)

        # Compute one step of the adjoint
        self.initialize_adjoint(scenario, bodies)
        self.iterate_adjoint(scenario, bodies, step)
        self.post_adjoint(scenario, bodies)

        # Perturb the displacements and surface temperatures
        adjoint_product = 0.0
        disp_pert_list = []
        temp_pert_list = []
        for body in bodies:
            body.initialize_variables(scenario)

            aero_disps = body.get_aero_disps(scenario)
            if aero_disps is not None:
                pert = np.random.uniform(size=aero_disps.shape)
                if complex_step:
                    aero_disps[:] += 1j * epsilon * pert
                else:
                    aero_disps[:] += epsilon * pert
                disp_pert_list.append(pert)

            aero_temps = body.get_aero_temps(scenario)
            if aero_temps is not None:
                pert = np.random.uniform(size=aero_temps.shape)
                if complex_step:
                    aero_temps[:] += 1j * epsilon * pert
                else:
                    aero_temps[:] += epsilon * pert
                temp_pert_list.append(pert)

            # Take the dot-product with the exact adjoint computation
            aero_disps_ajp = body.get_aero_disps_ajp(scenario)
            if aero_disps_ajp is not None:
                adjoint_product += np.dot(aero_disps_ajp[:, 0], disp_pert_list[-1])

            aero_temps_ajp = body.get_aero_temps_ajp(scenario)
            if aero_temps_ajp is not None:
                adjoint_product += np.dot(aero_temps_ajp[:, 0], temp_pert_list[-1])

        # Sum up the result across all processors
        adjoint_product = self.comm.allreduce(adjoint_product)

        # Run the perturbed aerodynamic simulation
        self.initialize(scenario, bodies)
        self.iterate(scenario, bodies, step)
        self.post(scenario, bodies)

        # Evaluate and store the functions of interest
        fvals = []
        self.get_functions(scenario, bodies)
        for func in scenario.functions:
            fvals.append(func.value)

        # Compute the finite-difference approximation
        if complex_step:
            fd_product = fvals[0].imag / epsilon
        else:
            fd_product = (fvals[0] - fvals_init[0]) / epsilon
        for body in bodies:
            aero_loads = body.get_aero_loads(scenario)
            aero_loads_ajp = body.get_aero_loads_ajp(scenario)
            if aero_loads is not None and aero_loads_ajp is not None:
                aero_loads_copy = aero_loads_list.pop(0)
                if complex_step:
                    fd = aero_loads.imag / epsilon
                else:
                    fd = (aero_loads - aero_loads_copy) / epsilon
                fd_product += np.dot(fd, aero_loads_ajp[:, 0])

            aero_flux = body.get_aero_heat_flux(scenario)
            aero_flux_ajp = body.get_aero_heat_flux_ajp(scenario)
            if aero_flux is not None and aero_flux_ajp is not None:
                aero_flux_copy = aero_flux_list.pop(0)
                if complex_step:
                    fd = aero_flux.imag / epsilon
                else:
                    fd = (aero_flux - aero_flux_copy) / epsilon
                fd_product += np.dot(fd, aero_flux_ajp[:, 0])

        # Compute the finite-differenc approximation
        fd_product = self.comm.allreduce(fd_product)

        fail = False
        if self.comm.rank == 0:
            rel_err = (adjoint_product - fd_product) / fd_product
            fail = abs(rel_err) >= rtol
            print("Flow solver adjoint result:           ", adjoint_product)
            print("Flow solver finite-difference result: ", fd_product)
            print("Flow solver relative error:           ", rel_err)
            print("Flow solver fail flag:                ", fail)

        fail = self.comm.bcast(fail, root=0)

        return fail

    def _test_struct_adjoint(
        self, scenario, bodies, step=0, epsilon=1e-6, complex_step=False, rtol=1e-6
    ):
        """
        Test to see if the adjoint methods are implemented correctly
        """

        self.set_functions(scenario, bodies)
        self.set_variables(scenario, bodies)

        for body in bodies:
            body.initialize_variables(scenario)

        # Set random loads and heat fluxes
        for body in bodies:
            struct_loads = body.get_struct_loads(scenario)
            if struct_loads is not None:
                shape = struct_loads.shape
                body.struct_loads_saved = np.random.uniform(size=shape)
                struct_loads[:] = body.struct_loads_saved

            struct_flux = body.get_struct_heat_flux(scenario)
            if struct_flux is not None:
                shape = struct_flux.shape
                body.struct_flux_saved = np.random.uniform(size=shape)
                struct_flux[:] = body.struct_flux_saved

        # Comptue one step of the forward solution
        self.initialize(scenario, bodies)
        self.iterate(scenario, bodies, step)
        self.post(scenario, bodies)

        # Evaluate and store the functions of interest
        fvals_init = []
        self.get_functions(scenario, bodies)
        for func in scenario.functions:
            fvals_init.append(func.value)

        # Save the output forces and heat fluxes
        struct_disps_list = []
        struct_temps_list = []
        for body in bodies:
            struct_disps = body.get_struct_disps(scenario)
            if struct_disps is not None:
                struct_disps_list.append(struct_disps.copy())

            struct_temps = body.get_struct_temps(scenario)
            if struct_temps is not None:
                struct_temps_list.append(struct_temps.copy())

        # Initialize the bodies for the adjoint computation
        for body in bodies:
            body.initialize_adjoint_variables(scenario)

            struct_disps_ajp = body.get_struct_disps_ajp(scenario)
            if struct_disps_ajp is not None:
                shape = struct_disps_ajp.shape
                struct_disps_ajp[:] = np.random.uniform(size=shape).astype(body.dtype)

            struct_temps_ajp = body.get_struct_temps_ajp(scenario)
            if struct_temps_ajp is not None:
                shape = struct_temps_ajp.shape
                struct_temps_ajp[:] = np.random.uniform(size=shape).astype(body.dtype)

        # Compute one step of the adjoint
        self.initialize_adjoint(scenario, bodies)
        self.iterate_adjoint(scenario, bodies, step)
        self.post_adjoint(scenario, bodies)

        # Perturb the displacements and surface temperatures
        adjoint_product = 0.0
        load_pert_list = []
        flux_pert_list = []
        for body in bodies:
            body.initialize_variables(scenario)

            # Reset the structural loads and heat fluxes
            struct_loads = body.get_struct_loads(scenario)
            if struct_loads is not None:
                struct_loads[:] = body.struct_loads_saved

            struct_flux = body.get_struct_heat_flux(scenario)
            if struct_flux is not None:
                struct_flux[:] = body.struct_flux_saved

            # Perturb the structural loads and heat fluxes
            struct_loads = body.get_struct_loads(scenario)
            if struct_loads is not None:
                pert = np.random.uniform(size=struct_loads.shape)
                if complex_step:
                    struct_loads[:] += 1j * epsilon * pert
                else:
                    struct_loads[:] += epsilon * pert
                load_pert_list.append(pert)

            struct_flux = body.get_struct_heat_flux(scenario)
            if struct_flux is not None:
                pert = np.random.uniform(size=struct_flux.shape)
                if complex_step:
                    struct_flux[:] += 1j * epsilon * pert
                else:
                    struct_flux[:] += epsilon * pert
                flux_pert_list.append(pert)

            # Take the dot-product with the exact adjoint computation
            struct_loads_ajp = body.get_struct_loads_ajp(scenario)
            if struct_loads_ajp is not None:
                adjoint_product += np.dot(struct_loads_ajp[:, 0], load_pert_list[-1])

            struct_flux_ajp = body.get_struct_heat_flux_ajp(scenario)
            if struct_flux_ajp is not None:
                adjoint_product += np.dot(struct_flux_ajp[:, 0], flux_pert_list[-1])

        # Sum up the result across all processors
        adjoint_product = self.comm.allreduce(adjoint_product)

        # Run the perturbed aerodynamic simulation
        self.initialize(scenario, bodies)
        self.iterate(scenario, bodies, step)
        self.post(scenario, bodies)

        # Evaluate and store the functions of interest
        fvals = []
        self.get_functions(scenario, bodies)
        for func in scenario.functions:
            fvals.append(func.value)

        # Compute the finite-difference approximation
        if complex_step:
            fd_product = fvals[0].imag / epsilon
        else:
            fd_product = (fvals[0] - fvals_init[0]) / epsilon
        for body in bodies:
            struct_disps = body.get_struct_disps(scenario)
            struct_disps_ajp = body.get_struct_disps_ajp(scenario)
            if struct_disps is not None and struct_disps_ajp is not None:
                struct_disps_copy = struct_disps_list.pop(0)
                if complex_step:
                    fd = struct_disps.imag / epsilon
                else:
                    fd = (struct_disps - struct_disps_copy) / epsilon
                fd_product += np.dot(fd, struct_disps_ajp[:, 0])

            struct_temps = body.get_struct_temps(scenario)
            struct_temps_ajp = body.get_struct_temps_ajp(scenario)
            if struct_temps is not None and struct_temps_ajp is not None:
                struct_temps_copy = struct_temps_list.pop(0)
                if complex_step:
                    fd = struct_temps.imag / epsilon
                else:
                    fd = (struct_temps - struct_temps_copy) / epsilon
                fd_product += np.dot(fd, struct_temps_ajp[:, 0])

        # Compute the finite-differenc approximation
        fd_product = self.comm.allreduce(fd_product)

        fail = False
        if self.comm.rank == 0:
            rel_err = (adjoint_product - fd_product) / fd_product
            fail = abs(rel_err) >= rtol
            print("Structural solver adjoint result:           ", adjoint_product)
            print("Structural solver finite-difference result: ", fd_product)
            print("Structural solver relative error:           ", rel_err)
            print("Structural solver fail flag:                ", fail)

        fail = self.comm.bcast(fail, root=0)

        return fail
