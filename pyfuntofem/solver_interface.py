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

class SolverInterface(object):
    """
    A base class to define what functions solver interfaces in FUNtoFEM need
    """
    def set_variables(self,scenario,bodies):
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

           for ibody,body in enumerate(bodies):
               if 'structural' in body.variables:
                   for var in body.variables['structural']:
                       solver.set_body_variable(ibody, var.value)

        Aerodynamic Solver:

        .. code-block:: python

           if 'aerodynamic' in scenario.variables:
               for var in scenario.variables['aerodynamic']:
                   if var.active:
                       solver.set_body_variable(var.name, var.value)
        """
        pass

    def set_functions(self,scenario,bodies):
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
               if function.adjoint and function.analysis_type=='structural':
                   solver.set_adjoint_function(function.name, function.start, function,stop)

               # Tell the solver that an adjoint is needed, but the function is not explicitly dependent on structural states
               elif function.adjoint and function.analysis_type !='structural':
                   solver.set_dummy_function()

               # Functions such as structural mass do not need and adjoint
               elif not function.adjoint and function.analysis_type=='structural':
                   solver.set_nonadjoint_function(function.name)


        """
        pass

    def get_functions(self,scenario,bodies):
        """
        Put the function values from the solver in the value attribute of the scneario's functions.
        The scenario has the list of function objects where the function's owned by this solver will be set.
        You can evaluate the functions based on the name or based on the functions set during :func:`~solver_interface.SolverInterface.set_functions`.
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
               if function.analysis_type=='structural':
                   if function.name == 'mass':
                       function.value = solver.evaluate_mass()
                   elif function.name == 'ksfailure':
                       function.value = solver.get_ksfailure()
                   elif function.name == 'ksfailure':
                       function.value = solver.get_ksfailure()
                   else function.name == 'ksfailure':
                       print("Unknown structural function in get_functions")

        """
        pass

    def get_function_gradients(self,scenario,bodies,offset):
        """
        Get the derivatives of all the functions with respect to design variables associated with this solver.
        The derivatives in the scenario and body objects are a Python dictionary where the keys are the type of variable.
        Each entry in the dictionary a list where each entry is associated with a function in the model. Finally, there is a list where each index is associated with a particular design variable.


        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        bodies: list of :class:`~body.Body` objects
            The bodies in the model
        offset: int
            offset of the scenario's function index w.r.t the full list of functions in the model.

        Examples
        --------
        Aerodynamic Solver:

        .. code-block:: python

            for func, function in enumerate(scenario.functions):
                for vartype in scenario.variables:
                    if vartype == 'aerodynamic':
                        for i, var in enumerate(scenario.variables[vartype]):
                            if var.active:
                                scenario.derivatives[vartype][offset+func][i] = solver.get_derivative(function.id,var.id)

        Structural Solver:

        .. code-block:: python

            for func, function in enumerate(scenario.functions):
                for ibody, body in enumerate(bodies):
                    for vartype in body.variables:
                        if vartype == 'structural':
                            for i, var in enumerate(body.variables[vartype]):
                                if var.active:
                                    body.derivatives[vartype][offset+func][i] = solver.get_derivative(func,ibody,i)
        """
        pass

    def get_coordinate_derivatives(self,scenario,bodies,step):
        """
        Add the solver's contributions to the coordinate derivatives for this time step or the total value for the steady case.
        The coordinate derivatives are stored in the body objects in the aero_shape_term and struct_shape_term attributes.

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
                if body.shape and body.aero_nnodes > 0:

                    lam_x, lam_y, lam_Z = solver.extract_coordinate_derivatives(step)

                    # Add (don't overwrite) the solver's contribution
                    body.aero_shape_term[ ::3,:nfunctions] += lam_x[:,:]
                    body.aero_shape_term[1::3,:nfunctions] += lam_y[:,:]
                    body.aero_shape_term[2::3,:nfunctions] += lam_z[:,:]

        Structural Solver:

        .. code-block:: python

            for ibody, body in enumerate(bodies):
                if body.shape:

                    lam_x, lam_y, lam_Z = solver.extract_coordinate_derivatives(step)

                    # Add (don't overwrite) the solver's contribution
                    body.struct_shape_term[ ::3,:nfunctions] += lam_x[:,:]
                    body.struct_shape_term[1::3,:nfunctions] += lam_y[:,:]
                    body.struct_shape_term[2::3,:nfunctions] += lam_z[:,:]


        """
        pass

    def initialize(self,scenario,bodies):
        """
        This function allows the solver to set up anything that is necessary before the scenario is simulated (forward analysis), e.g., load in the mesh which has been updated by the shape parameterization, allocate arrays, set initial conditions, etc.


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
                solver.set_mesh(ibody, body.aero_X)

           # Initialize the flow field
           solver.initialize()
        """
        return 0

    def iterate(self,scenario,bodies,step):
        """
        Advance the solver's residual(s).
        Called in NLBGS solver.

        For an aerodynamic solver, this might include:
        #. Deforming the meshes
        #. Solving for the new flow state
        #. Integrating to get new aerodynamic surface forces and putting them in the body objects

        For a structural solver, the structural forces should be applied and new structural displacements should be put into the body objects

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

            # Input the surface deflections
            for ibody, body in enumerate(bodies):
                if body.aero_nnodes > 0:
                    solver.input_surface_deformation(ibody, body.aero_disps)

            # Advance the solver
            solver.iterate()

            # Extract the surface forces
            for ibody, body in enumerate(bodies):
                if body.aero_nnodes > 0:
                    body.aero_loads = solver.extract_forces(ibody)

        Structural Solver:

        .. code-block:: python

            # Input the forces on the structure
            for ibody, body in enumerate(bodies):
                if body.aero_nnodes > 0:
                    solver.input_forces(ibody, body.struct_loads)

            # Advance the solver
            solver.iterate()

            # Extract the displacemets
            for ibody, body in enumerate(bodies):
                if body.aero_nnodes > 0:
                    body.struct_disps = solver.extract_displacements(ibody)
        """
        return 0

    def post(self,scenario,bodies):
        """
        Perform any tasks the solver needs to do after the forward steps are complete, e.g., evaluate functions, post-process, deallocate unneeded memory.

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        bodies: list of :class:`~body.Body` objects
            The bodies in the model
        """
        pass

    def initialize_adjoint(self,scenario,bodies):
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

    def set_states(self,scenario,bodies,step):
        """
        Load the states (aero_loads, struct_disps) associated with this step either from memory or the disk for the transfer scheme to linearize about.
        This function is called at the beginning of each adjoint step in time dependent problems.


        **Note: in the NLBGS algorithm the transfer scheme uses the structural displacements from the prior step. set_states will request the states from the previous step but then ask the structural solver to linearize about the current step in iterate_adjoint**

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

    def iterate_adjoint(self,scenario,bodies,step):
        """
        Adjoint iteration for the solver. Typical involves the solver reading in a RHS term then returning an adjoint or adjoint-product.
        Called in NLBGS solver.

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

            # Read in the force adjoint
            for ibody, body in enumerate(bodies):
                solver.set_force_adjoint(body.psi_F)

                # take a reverse step in the adjoint solver
                solver.iterate_adjoint(step)

                # pull out the grid adjoint value
                for ibody, body in enumerate(bodies):
                    psi_G = solver.get_grid_adjoint(ibody)

                    # 'solve' for the displacement adjoint
                    body.psi_D = -psi_G


        Structural Solver:

        .. code-block:: python

            # put the body RHS's into the solver bvec
            for ibody, body in enumerate(bodies):
                solver.add_body_rhs_term(ibody, body.struct_rhs)

            # take a reverse step in the adjoint solver
            solver.iterate_adjoint(step)

            # pull out the structural adjoint value
            for ibody, body in enumerate(bodies):
                body.psi_S = solver.get_struct_adjoint(ibody)
        """
        return 0

    def post_adjoint(self,scenario,bodies):
        """
        Any actions that need to be performed after completing the adjoint solve, e.g., evaluating gradients, deallocating memory, etc.
        """
        pass

    def step_pre(self,scenario,bodies,step):
        """
        Operations before at a step in an FSI subiteration case.
        Called in NLBGS with FSI subiterations.

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

    def step_solver(self,scenario,bodies,step,fsi_subiter):
        """
        Step in an FSI subiteration case.
        Called in NLBGS with FSI subiterations.

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
    def step_post(self,scenario,bodies,step):
        """
        Operations after at a step in an FSI subiteration case.
        Called in NLBGS with FSI subiterations.

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
