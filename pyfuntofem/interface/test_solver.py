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

__all__ = ["TestAerodynamicSolver", "TestStructuralSolver"]

import numpy as np
from funtofem import TransferScheme
from ._solver_interface import SolverInterface


class TestAerodynamicSolver(SolverInterface):
    def __init__(self, comm, model):
        """
        This class provides the functionality that FUNtoFEM expects from
        an aerodynamic solver.

        Forward analysis
        ----------------

        Any aerodynamic solver must provide the aerodynamic forces and area-weighted
        heat flux at the aerodynmic surface node locations as a function of the displacements
        and wall temperatures at those same aerodynamic surface nodes. For
        the aerodynamic forces, this relationship is

        fA = fA(uA, tA, xA0, x)

        where uA are the displacements at the aerodynamic nodes, tA are the temperatures at
        the aerodynamic nodes, xA0 are the original node locations and x are the design variable
        values.

        Written in terms of the variable names used within the code, this expression is

        aero_forces = fA(aero_disps, aero_temps, aero_X, aero_dvs).

        Note that there are 3 components of the force for each aerodynamic surface node.

        For the area-weighted normal component of the heat flux, this relationship is

        hA = hA(uA, tA, xA0, x)

        where hA is the area-weighted normal component of the heat flux. In terms of the
        variable names used within the code, this expression is

        aero_flux = hA(aero_disps, aero_temps, aero_X, aero_dvs).

        For this test solver, we substitute artifical relationships that are randomly
        generated. These relationships have no physical significance, but reflect the
        dependence between variables in a true solver. For the aerodynamic forces we set

        aero_forces = Jac1 * aero_disps + b1 * aero_X + c1 * aero_dvs

        and for the aerodynamic heat flux we set

        aero_flux = Jac2 * aero_temps + b2 * aero_X + c2 * aero_dvs.

        Adjoint analysis
        ----------------

        For adjoint analysis, the aerodynamic analysis must take in the adjoint contributions
        from the aero forces and heat flux and compute the corresponding contributions to
        the displacement and temperature adjoint terms.

        The terms passed between solvers are always the product of the adjoint variables with
        the derivative of their associated residuals with respect to a certain state variable.
        These terms are adjoint-Jacobian products (AJPs). Our naming convention is to use the
        name of the derivative variable as the name of variable in the code. The size of the
        adjoint-Jacobian product corresponds to the size of the named forward variable, except
        we solve "nfunctions" adjoints at the same time.

        For an aerodynamic analysis, the input is the contributions are the adjoint-Jacobian products
        from the other coupling analyses. These variables within the code are the contribution from
        the aerodynamic loads

        aero_loads_ajp = dL/dfA^{T} * psi_L

        and the contribution from the aerodynamic heat flux

        aero_flux_ajp = dQ/dhA^{T} * psi_Q

        For a fully-coupled CFD solver with mesh deformation, these inputs would be used in the
        following manner

        1. Solve for the force integration adjoint, psi_F
        dF/dfA^{T} * psi_F = - aero_loads_ajp

        2. Solve for the heat flux integration adjoint, psi_H
        dH/dhA^{T} * psi_H = - aero_flux_ajp

        3. Solve for the aerodynamic adjoint variables
        dA/dq^{T} * psi_A = -df/dq^{T} - dF/dq^{T} * psi_F - dH/dq * psi_H

        4. Solve for the grid adjoint variables
        dG/dxG^{T} * psi_G = -df/dxG^{T} - dA/dxG^{T} * psi_A - dF/dxG^{T} * psi_F - dH/dxG^{T} * psi_H

        5. Compute the adjoint-Jacobian products required for the output

        aero_disps_ajp = dG/duA^{T} * psi_G
        aero_temps_ajp = dA/dtA^{T} * psi_A

        Parameters
        ----------
        comm: MPI.comm
            MPI communicator
        model: :class:`~funtofem_model.FUNtoFEMmodel`
            The model containing the design data
        """

        self.comm = comm
        self.npts = 10
        np.random.seed(0)

        # Get the list of active design variables
        self.variables = model.get_variables()

        # Count the number of aerodynamic design variables (if any)
        self.aero_variables = []  # List of the variable objects

        # List of the variable values - converted into an numpy array
        self.aero_dvs = []

        for var in self.variables:
            if var.analysis_type == "aerodynamic":
                self.aero_variables.append(var)
                self.aero_dvs.append(var.value)

        # Allocate space for the aero dvs
        self.aero_dvs = np.array(self.aero_dvs, dtype=TransferScheme.dtype)

        # Aerodynaimic forces = Jac1 * aero_disps + b1 * aero_X + c1 * aero_dvs
        self.Jac1 = 0.01 * (np.random.rand(3 * self.npts, 3 * self.npts) - 0.5)
        self.b1 = 0.01 * (np.random.rand(3 * self.npts, 3 * self.npts) - 0.5)
        self.c1 = 0.01 * (np.random.rand(3 * self.npts, len(self.aero_dvs)) - 0.5)

        # Aero heat flux = Jac2 * aero_temps + b2 * aero_X + c2 * aero_dvs
        self.Jac2 = 0.05 * (np.random.rand(self.npts, self.npts) - 0.5)
        self.b2 = 0.1 * (np.random.rand(self.npts, 3 * self.npts) - 0.5)
        self.c2 = 0.01 * (np.random.rand(self.npts, len(self.aero_dvs)) - 0.5)

        # Set random initial node locations
        self.aero_X = np.random.rand(3 * self.npts).astype(TransferScheme.dtype)

        # Data for generating functional output values
        self.func_coefs1 = np.random.rand(3 * self.npts)
        self.func_coefs2 = np.random.rand(self.npts)

        # Initialize the coordinates of the aerodynamic or structural mesh
        for body in model.bodies:
            body.initialize_aero_nodes(self.aero_X)

        return

    def set_variables(self, scenario, bodies):
        """Set the design variables for the solver"""

        # Set the aerodynamic design variables
        index = 0
        for var in self.variables:
            if var.analysis_type == "aerodynamic":
                self.aero_dvs[index] = var.value
                index += 1

        return

    def set_functions(self, scenario, bodies):
        """
        Set the functions to be used for the given scenario.

        In this function, each discipline should initialize the data needed to evaluate
        the given set of functions set in each scenario.
        """

        return

    def get_functions(self, scenario, bodies):
        """
        Evaluate the functions of interest and set the function values into
        the scenario.functions objects
        """

        if scenario.steady:
            for index, func in enumerate(scenario.functions):
                if func.analysis_type == "aerodynamic":
                    value = 0.0
                    for body in bodies:
                        aero_disps = body.get_aero_disps(scenario)
                        if aero_disps is not None:
                            value += np.dot(self.func_coefs1, aero_disps)
                        aero_temps = body.get_aero_temps(scenario)
                        if aero_temps is not None:
                            value += np.dot(self.func_coefs2, aero_temps)
                    func.value = self.comm.allreduce(value)
        else:
            # Set the time index to the final time step
            time_index = scenario.steps

            for index, func in enumerate(scenario.functions):
                if func.analysis_type == "aerodynamic":
                    value = 0.0
                    for body in bodies:
                        aero_disps = body.get_aero_disps(scenario, time_index)
                        if aero_disps is not None:
                            value += np.dot(self.func_coefs1, aero_disps)
                        aero_temps = body.get_aero_temps(scenario, time_index)
                        if aero_temps is not None:
                            value += np.dot(self.func_coefs2, aero_temps)
                    func.value = self.comm.allreduce(value)

        return

    def get_function_gradients(self, scenario, bodies):
        """
        Return the function gradients
        """

        # Set the derivatives of the functions for the given scenario
        for findex, func in enumerate(scenario.functions):
            for vindex, var in enumerate(self.aero_variables):

                for body in bodies:
                    aero_loads_ajp = body.get_aero_loads_ajp(scenario)
                    if aero_loads_ajp is not None:
                        value = np.dot(aero_loads_ajp[:, findex], self.c1[:, vindex])
                        func.add_gradient_component(var, value)

                    aero_flux_ajp = body.get_aero_heat_flux_ajp(scenario)
                    if aero_flux_ajp is not None:
                        value = np.dot(aero_flux_ajp[:, findex], self.c2[:, vindex])
                        func.add_gradient_component(var, value)

        return

    def get_coordinate_derivatives(self, scenario, bodies, step):
        """
        Add the contributions to the gradient w.r.t. the aerodynamic coordinates
        """

        for findex, func in enumerate(scenario.functions):
            for body in bodies:
                aero_shape_term = body.get_aero_coordinate_derivatives(scenario)
                aero_loads_ajp = body.get_aero_loads_ajp(scenario)
                if aero_loads_ajp is not None:
                    aero_shape_term[:, findex] += np.dot(
                        aero_loads_ajp[:, findex], self.b1
                    )

                aero_flux_ajp = body.get_aero_heat_flux_ajp(scenario)
                if aero_flux_ajp is not None:
                    aero_shape_term[:, findex] += np.dot(
                        aero_flux_ajp[:, findex], self.b2
                    )

        return

    def initialize(self, scenario, bodies):
        """Note that this function must return a fail flag of zero on success"""

        for body in bodies:
            self.aero_X[:] = body.get_aero_nodes()

        return 0

    def iterate(self, scenario, bodies, step):
        """
        Iterate for the aerodynamic or structural solver

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        bodies: :class:`~body.Body`
            list of FUNtoFEM bodies
        step: integer
            Step number for the steady-state solution method
        """

        for body in bodies:
            # Perform the "analysis"
            # Note that the body class expects that you will write the new
            # aerodynamic loads into the aero_loads array.
            aero_disps = body.get_aero_disps(scenario)
            aero_loads = body.get_aero_loads(scenario)
            if aero_disps is not None:
                aero_loads[:] = np.dot(self.Jac1, aero_disps)
                aero_loads[:] += np.dot(self.b1, self.aero_X)
                aero_loads[:] += np.dot(self.c1, self.aero_dvs)

            # Perform the heat transfer "analysis"
            aero_temps = body.get_aero_temps(scenario)
            aero_flux = body.get_aero_heat_flux(scenario)
            if aero_temps is not None:
                aero_flux[:] = np.dot(self.Jac2, aero_temps)
                aero_flux[:] += np.dot(self.b2, self.aero_X)
                aero_flux[:] += np.dot(self.c2, self.aero_dvs)

        # This analysis is always successful so return fail = 0
        fail = 0
        return fail

    def post(self, scenario, bodies):
        pass

    def initialize_adjoint(self, scenario, bodies):
        """Note that this function must return a fail flag of zero on success"""
        return 0

    def iterate_adjoint(self, scenario, bodies, step):
        """
        Iterate for the aerodynamic or structural adjoint

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        bodies: :class:`~body.Body`
            list of FUNtoFEM bodies
        step: integer
            Step number for the steady-state solution method
        """

        for body in bodies:
            aero_loads_ajp = body.get_aero_loads_ajp(scenario)
            aero_disps_ajp = body.get_aero_disps_ajp(scenario)
            if aero_loads_ajp is not None:
                for k, func in enumerate(scenario.functions):
                    aero_disps_ajp[:, k] = np.dot(self.Jac1.T, aero_loads_ajp[:, k])
                    if func.analysis_type == "aerodynamic":
                        aero_disps_ajp[:, k] += self.func_coefs1

            aero_flux_ajp = body.get_aero_heat_flux_ajp(scenario)
            aero_temps_ajp = body.get_aero_temps_ajp(scenario)
            if aero_flux_ajp is not None:
                for k, func in enumerate(scenario.functions):
                    aero_temps_ajp[:, k] = np.dot(self.Jac2.T, aero_flux_ajp[:, k])
                    if func.analysis_type == "aerodynamic":
                        aero_temps_ajp[:, k] += self.func_coefs2

        fail = 0
        return fail

    def post_adjoint(self, scenario, bodies):
        pass


class TestStructuralSolver(SolverInterface):
    def __init__(self, comm, model, elastic_k=1.0, thermal_k=1.0):
        """
        A test solver that provides the functionality that FUNtoFEM expects from
        a structural solver.

        Forward analysis
        ----------------


        Adjoint analysis
        ----------------


        Parameters
        ----------
        comm: MPI.comm
            MPI communicator
        model: :class:`~funtofem_model.FUNtoFEMmodel`
            The model containing the design data
        """

        self.comm = comm
        self.npts = 25
        np.random.seed(54321)

        # Get the list of active design variables
        self.variables = model.get_variables()

        # Count the number of structural design variables (if any)
        self.struct_variables = []  # List of the variable objects

        # List of the variable values - converted into an numpy array
        self.struct_dvs = []

        for var in self.variables:
            if var.analysis_type == "structural":
                self.struct_variables.append(var)
                self.struct_dvs.append(var.value)

        # Allocate space for the aero dvs
        self.struct_dvs = np.array(self.struct_dvs, dtype=TransferScheme.dtype)

        # elastic and thermal scales 1/stiffness
        elastic_scale = 1.0 / elastic_k
        thermal_scale = 1.0 / thermal_k

        # Struct disps = Jac1 * struct_forces + b1 * struct_X + c1 * struct_dvs
        self.Jac1 = (
            0.01 * elastic_scale * (np.random.rand(3 * self.npts, 3 * self.npts) - 0.5)
        )
        self.b1 = (
            0.01 * elastic_scale * (np.random.rand(3 * self.npts, 3 * self.npts) - 0.5)
        )
        self.c1 = (
            0.01
            * elastic_scale
            * (np.random.rand(3 * self.npts, len(self.struct_dvs)) - 0.5)
        )

        # Struct temps = Jac2 * struct_flux + b2 * struct_X + c2 * struct_dvs
        self.Jac2 = 0.05 * thermal_scale * (np.random.rand(self.npts, self.npts) - 0.5)
        self.b2 = 0.1 * thermal_scale * (np.random.rand(self.npts, 3 * self.npts) - 0.5)
        self.c2 = (
            0.01
            * thermal_scale
            * (np.random.rand(self.npts, len(self.struct_dvs)) - 0.5)
        )

        # Set random initial node locations
        self.struct_X = np.random.rand(3 * self.npts).astype(TransferScheme.dtype)

        # Data for output functional values
        self.func_coefs1 = np.random.rand(3 * self.npts)
        self.func_coefs2 = np.random.rand(self.npts)

        # Initialize the coordinates of the structural mesh
        for body in model.bodies:
            body.initialize_struct_nodes(self.struct_X)

        return

    def set_variables(self, scenario, bodies):
        """Set the design variables for the solver"""

        # Set the aerodynamic design variables
        index = 0
        for var in self.variables:
            if var.analysis_type == "structural":
                self.struct_dvs[index] = var.value
                index += 1

        return

    def set_functions(self, scenario, bodies):
        """
        Set the functions to be used for the given scenario.

        In this function, each discipline should initialize the data needed to evaluate
        the given set of functions set in each scenario.
        """

        return

    def get_functions(self, scenario, bodies):
        """
        Evaluate the functions of interest and set the function values into
        the scenario.functions objects
        """

        if scenario.steady:
            for index, func in enumerate(scenario.functions):
                if func.analysis_type == "structural":
                    value = 0.0
                    for body in bodies:
                        struct_loads = body.get_struct_loads(scenario)
                        if struct_loads is not None:
                            value += np.dot(self.func_coefs1, struct_loads)
                        struct_flux = body.get_struct_heat_flux(scenario)
                        if struct_flux is not None:
                            value += np.dot(self.func_coefs2, struct_flux)
                    func.value = self.comm.allreduce(value)
        else:
            # Set the time index to the final time step
            time_index = scenario.steps

            for index, func in enumerate(scenario.functions):
                if func.analysis_type == "structural":
                    value = 0.0
                    for body in bodies:
                        struct_loads = body.get_struct_loads(scenario, time_index)
                        if struct_loads is not None:
                            value += np.dot(self.func_coefs1, struct_loads)
                        struct_flux = body.get_struct_heat_flux(scenario, time_index)
                        if struct_flux is not None:
                            value += np.dot(self.func_coefs2, struct_flux)
                    func.value = self.comm.allreduce(value)

        return

    def get_function_gradients(self, scenario, bodies):
        """
        Evaluate the function gradients and set them into the function classes.

        Note: The function gradients can be evaluated elsewhere (for instance in
        post_adjoint(). This function must get these values and place them into the
        associated function.)
        """

        # Set the derivatives of the functions for the given scenario
        for findex, func in enumerate(scenario.functions):
            for vindex, var in enumerate(self.struct_variables):

                for body in bodies:
                    struct_disps_ajp = body.get_struct_disps_ajp(scenario)
                    if struct_disps_ajp is not None:
                        value = np.dot(struct_disps_ajp[:, findex], self.c1[:, vindex])
                        func.add_gradient_component(var, value)

                    struct_temps_ajp = body.get_struct_temps_ajp(scenario)
                    if struct_temps_ajp is not None:
                        value = np.dot(struct_temps_ajp[:, findex], self.c2[:, vindex])
                        func.add_gradient_component(var, value)

        return

    def get_coordinate_derivatives(self, scenario, bodies, step):
        """
        Add the contributions to the gradient w.r.t. the structural coordinates
        """

        for findex, func in enumerate(scenario.functions):
            for body in bodies:
                struct_shape_term = body.get_struct_coordinate_derivatives(scenario)
                struct_disps_ajp = body.get_struct_disps_ajp(scenario)
                if struct_disps_ajp is not None:
                    struct_shape_term[:, findex] += np.dot(
                        struct_disps_ajp[:, findex], self.b1
                    )

                struct_temps_ajp = body.get_struct_temps_ajp(scenario)
                if struct_temps_ajp is not None:
                    struct_shape_term[:, findex] += np.dot(
                        struct_temps_ajp[:, findex], self.b2
                    )

        return

    def initialize(self, scenario, bodies):
        """Note that this function must return a fail flag of zero on success"""

        # Initialize the coordinates of the aerodynamic or structural mesh
        for body in bodies:
            self.struct_X[:] = body.get_struct_nodes()

        return 0

    def iterate(self, scenario, bodies, step):
        """
        Iterate for the aerodynamic or structural solver

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        bodies: :class:`~body.Body`
            list of FUNtoFEM bodies
        step: integer
            Step number for the steady-state solution method
        """

        for body in bodies:
            # Perform the "analysis"
            struct_loads = body.get_struct_loads(scenario)
            struct_disps = body.get_struct_disps(scenario)
            if struct_loads is not None:
                struct_disps[:] = np.dot(self.Jac1, struct_loads)
                struct_disps[:] += np.dot(self.b1, self.struct_X)
                struct_disps[:] += np.dot(self.c1, self.struct_dvs)

            # Perform the heat transfer "analysis"
            struct_flux = body.get_struct_heat_flux(scenario)
            struct_temps = body.get_struct_temps(scenario)
            if struct_flux is not None:
                struct_temps[:] = np.dot(self.Jac2, struct_flux)
                struct_temps[:] += np.dot(self.b2, self.struct_X)
                struct_temps[:] += np.dot(self.c2, self.struct_dvs)

        # This analysis is always successful so return fail = 0
        fail = 0
        return fail

    def post(self, scenario, bodies):
        pass

    def initialize_adjoint(self, scenario, bodies):
        """Note that this function must return a fail flag of zero on success"""
        return 0

    def iterate_adjoint(self, scenario, bodies, step):
        """
        Iterate for the aerodynamic or structural adjoint

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        bodies: :class:`~body.Body`
            list of FUNtoFEM bodies
        step: integer
            Step number for the steady-state solution method
        """

        for body in bodies:
            struct_disps_ajp = body.get_struct_disps_ajp(scenario)
            struct_loads_ajp = body.get_struct_loads_ajp(scenario)
            if struct_disps_ajp is not None:
                for k, func in enumerate(scenario.functions):
                    struct_loads_ajp[:, k] = np.dot(self.Jac1.T, struct_disps_ajp[:, k])
                    if func.analysis_type == "structural":
                        struct_loads_ajp[:, k] += self.func_coefs1

            struct_temps_ajp = body.get_struct_temps_ajp(scenario)
            struct_flux_ajp = body.get_struct_heat_flux_ajp(scenario)
            if struct_temps_ajp is not None:
                for k, func in enumerate(scenario.functions):
                    struct_flux_ajp[:, k] = np.dot(self.Jac2.T, struct_temps_ajp[:, k])
                    if func.analysis_type == "structural":
                        struct_flux_ajp[:, k] += self.func_coefs2

        fail = 0
        return fail

    def post_adjoint(self, scenario, bodies):
        return
