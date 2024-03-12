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

__all__ = [
    "TestAerodynamicSolver",
    "TestStructuralSolver",
    "NullAerodynamicSolver",
    "TestResult",
    "CoordinateDerivativeTester",
]

import numpy as np
from funtofem import TransferScheme
from ._solver_interface import SolverInterface


class TestAerodynamicSolver(SolverInterface):
    def __init__(self, comm, model, copy_struct_mesh=False):
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
        self.model = model
        self.npts = 10
        np.random.seed(0)

        # setup forward and adjoint tolerances
        super().__init__()

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

        # Set random initial node locations
        if copy_struct_mesh:
            self._copy_struct_mesh()
        else:
            self.aero_X = np.random.rand(3 * self.npts).astype(TransferScheme.dtype)

        # define data owned by each scenario
        class ScenarioData:
            def __init__(self, npts, dvs):
                self.npts = npts
                self.aero_dvs = dvs

                # choose random aero time step
                self.dt = 0.01

                # Aerodynamic forces = Jac1 * aero_disps + b1 * aero_X + c1 * aero_dvs + omega1 * (dt * step)
                self.Jac1 = 0.01 * (np.random.rand(3 * self.npts, 3 * self.npts))
                self.b1 = 0.01 * (np.random.rand(3 * self.npts, 3 * self.npts) - 0.5)
                self.c1 = 0.01 * (
                    np.random.rand(3 * self.npts, len(self.aero_dvs)) - 0.5
                )

                # Aero heat flux = Jac2 * aero_temps + b2 * aero_X + c2 * aero_dvs + omega2 * dt
                self.Jac2 = 0.1 * (np.random.rand(self.npts, self.npts) - 0.5)
                self.b2 = 0.1 * (np.random.rand(self.npts, 3 * self.npts) - 0.5)
                self.c2 = 0.01 * (np.random.rand(self.npts, len(self.aero_dvs)) - 0.5)

                # Data for generating functional output values
                self.func_coefs1 = np.random.rand(3 * self.npts)
                self.func_coefs2 = np.random.rand(self.npts)

                # omega values
                rate = 0.001
                self.omega1 = rate * (np.random.rand(3 * self.npts) - 0.5)
                self.omega2 = rate * (np.random.rand(self.npts) - 0.5)

        # make scenario data classes for each available scenario
        self.scenario_data = {}
        for scenario in model.scenarios:
            self.scenario_data[scenario.id] = ScenarioData(self.npts, self.aero_dvs)

        # Initialize the coordinates of the aerodynamic or structural mesh
        aero_id = np.arange(1, self.npts + 1)
        for body in model.bodies:
            body.initialize_aero_nodes(self.aero_X, aero_id)

        return

    def _copy_struct_mesh(self):
        """copy aero mesh from the structures side"""
        body = self.model.bodies[0]
        self.npts = body.get_num_struct_nodes()
        aero_id = np.arange(1, body.get_num_struct_nodes() + 1)
        body.initialize_aero_nodes(body.struct_X, aero_id)
        self.aero_X = body.aero_X
        return self

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

        time_index = 0 if scenario.steady else scenario.steps

        for func in scenario.functions:
            if func.analysis_type == "aerodynamic":
                value = 0.0
                for body in bodies:
                    aero_disps = body.get_aero_disps(scenario, time_index)
                    if aero_disps is not None:
                        value += np.dot(
                            self.scenario_data[scenario.id].func_coefs1, aero_disps
                        )
                    aero_temps = body.get_aero_temps(scenario, time_index)
                    if aero_temps is not None:
                        value += np.dot(
                            self.scenario_data[scenario.id].func_coefs2, aero_temps
                        )
                func.value = self.comm.allreduce(value)

        return

    def get_function_gradients(self, scenario, bodies):
        """
        Return the function gradients
        """

        # Set the derivatives of the functions for the given scenario
        for findex, func in enumerate(scenario.adjoint_functions):
            for vindex, var in enumerate(self.aero_variables):
                for body in bodies:
                    aero_loads_ajp = body.get_aero_loads_ajp(scenario)
                    if aero_loads_ajp is not None:
                        value = np.dot(
                            aero_loads_ajp[:, findex],
                            self.scenario_data[scenario.id].c1[:, vindex],
                        )
                        func.add_gradient_component(var, value)

                    aero_flux_ajp = body.get_aero_heat_flux_ajp(scenario)
                    if aero_flux_ajp is not None:
                        value = np.dot(
                            aero_flux_ajp[:, findex],
                            self.scenario_data[scenario.id].c2[:, vindex],
                        )
                        func.add_gradient_component(var, value)

        return

    def get_coordinate_derivatives(self, scenario, bodies, step):
        """
        Add the contributions to the gradient w.r.t. the aerodynamic coordinates
        """

        if step == 0:
            return

        for ifunc, func in enumerate(scenario.adjoint_functions):
            ifull = scenario.adjoint_map[ifunc]
            for body in bodies:
                aero_shape_term = body.get_aero_coordinate_derivatives(scenario)
                aero_loads_ajp = body.get_aero_loads_ajp(scenario)
                if aero_loads_ajp is not None:
                    aero_shape_term[:, ifull] += np.dot(
                        aero_loads_ajp[:, ifunc], self.scenario_data[scenario.id].b1
                    )

                aero_flux_ajp = body.get_aero_heat_flux_ajp(scenario)
                if aero_flux_ajp is not None:
                    aero_shape_term[:, ifull] += np.dot(
                        aero_flux_ajp[:, ifunc], self.scenario_data[scenario.id].b2
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

        aero_time = step * self.scenario_data[scenario.id].dt

        for body in bodies:
            # Perform the "analysis"
            # Note that the body class expects that you will write the new
            # aerodynamic loads into the aero_loads array.
            aero_disps = body.get_aero_disps(scenario, step)
            aero_loads = body.get_aero_loads(scenario, step)
            if aero_disps is not None:
                aero_loads[:] = np.dot(self.scenario_data[scenario.id].Jac1, aero_disps)
                aero_loads[:] += np.dot(self.scenario_data[scenario.id].b1, self.aero_X)
                aero_loads[:] += np.dot(
                    self.scenario_data[scenario.id].c1, self.aero_dvs
                )
                if not scenario.steady:
                    aero_loads[:] += self.scenario_data[scenario.id].omega1 * aero_time

            # Perform the heat transfer "analysis"
            aero_temps = body.get_aero_temps(scenario, step)
            aero_flux = body.get_aero_heat_flux(scenario, step)
            if aero_temps is not None:
                aero_flux[:] = np.dot(self.scenario_data[scenario.id].Jac2, aero_temps)
                aero_flux[:] += np.dot(self.scenario_data[scenario.id].b2, self.aero_X)
                aero_flux[:] += np.dot(
                    self.scenario_data[scenario.id].c2, self.aero_dvs
                )
                if not scenario.steady:  # omega * dt term here
                    aero_flux[:] += self.scenario_data[scenario.id].omega2 * aero_time

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

        # If the scenario is unsteady only add the rhs for the final state
        include_rhs = True
        if not scenario.steady and step != scenario.steps:
            include_rhs = False

        for body in bodies:
            aero_loads_ajp = body.get_aero_loads_ajp(scenario)
            aero_disps_ajp = body.get_aero_disps_ajp(scenario)
            if aero_loads_ajp is not None:
                for k, func in enumerate(scenario.adjoint_functions):
                    aero_disps_ajp[:, k] = np.dot(
                        self.scenario_data[scenario.id].Jac1.T, aero_loads_ajp[:, k]
                    )
                    if include_rhs and func.analysis_type == "aerodynamic":
                        aero_disps_ajp[:, k] += self.scenario_data[
                            scenario.id
                        ].func_coefs1

            aero_flux_ajp = body.get_aero_heat_flux_ajp(scenario)
            aero_temps_ajp = body.get_aero_temps_ajp(scenario)
            if aero_flux_ajp is not None:
                for k, func in enumerate(scenario.adjoint_functions):
                    aero_temps_ajp[:, k] = np.dot(
                        self.scenario_data[scenario.id].Jac2.T, aero_flux_ajp[:, k]
                    )
                    if include_rhs and func.analysis_type == "aerodynamic":
                        aero_temps_ajp[:, k] += self.scenario_data[
                            scenario.id
                        ].func_coefs2

        if not scenario.steady:
            if step > 0:
                self.get_function_gradients(scenario, bodies)
            else:  # step == 0
                # want to zero out adjoints used for derivatives, since no analysis done on step 0
                for body in bodies:
                    aero_loads_ajp = body.get_aero_loads_ajp(scenario)
                    if aero_loads_ajp is not None:
                        aero_loads_ajp *= 0.0

                    aero_flux_ajp = body.get_aero_heat_flux_ajp(scenario)
                    if aero_flux_ajp is not None:
                        aero_flux_ajp *= 0.0

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

        # setup forward and adjoint tolerances
        super().__init__()

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

        # scenario data for the multi scenario case
        class ScenarioData:
            def __init__(self, npts, struct_dvs):
                self.npts = npts
                self.struct_dvs = struct_dvs

                # choose time step
                self.dt = 0.01

                # Struct disps = Jac1 * struct_forces + b1 * struct_X + c1 * struct_dvs + omega1 * time
                self.Jac1 = (
                    0.01
                    * elastic_scale
                    * (np.random.rand(3 * self.npts, 3 * self.npts) - 0.5)
                )
                self.b1 = (
                    0.01
                    * elastic_scale
                    * (np.random.rand(3 * self.npts, 3 * self.npts) - 0.5)
                )
                self.c1 = (
                    0.01
                    * elastic_scale
                    * (np.random.rand(3 * self.npts, len(self.struct_dvs)) - 0.5)
                )

                # Struct temps = Jac2 * struct_flux + b2 * struct_X + c2 * struct_dvs + omega2 * time
                self.Jac2 = (
                    0.05 * thermal_scale * (np.random.rand(self.npts, self.npts) - 0.5)
                )
                self.b2 = (
                    0.1
                    * thermal_scale
                    * (np.random.rand(self.npts, 3 * self.npts) - 0.5)
                )
                self.c2 = (
                    0.01
                    * thermal_scale
                    * (np.random.rand(self.npts, len(self.struct_dvs)) - 0.5)
                )

                # Data for output functional values
                self.func_coefs1 = np.random.rand(3 * self.npts)
                self.func_coefs2 = np.random.rand(self.npts)

                # unsteady state variable drift
                rate = 0.001
                self.omega1 = rate * (np.random.rand(3 * self.npts) - 0.5)
                self.omega2 = rate * (np.random.rand(self.npts) - 0.5)

        # create scenario data for each scenario
        self.scenario_data = {}
        for scenario in model.scenarios:
            self.scenario_data[scenario.id] = ScenarioData(self.npts, self.struct_dvs)

        # Set random initial node locations
        self.struct_X = np.random.rand(3 * self.npts).astype(TransferScheme.dtype)

        # Initialize the coordinates of the structural mesh
        struct_id = np.arange(1, self.npts + 1)
        for body in model.bodies:
            body.initialize_struct_nodes(self.struct_X, struct_id)

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

        time_index = 0 if scenario.steady else scenario.steps

        for func in scenario.functions:
            if func.analysis_type == "structural":
                value = 0.0
                for body in bodies:
                    struct_loads = body.get_struct_loads(scenario, time_index)
                    if struct_loads is not None:
                        value += np.dot(
                            self.scenario_data[scenario.id].func_coefs1, struct_loads
                        )
                    struct_flux = body.get_struct_heat_flux(scenario, time_index)
                    if struct_flux is not None:
                        value += np.dot(
                            self.scenario_data[scenario.id].func_coefs2, struct_flux
                        )
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
        for findex, func in enumerate(scenario.adjoint_functions):
            for vindex, var in enumerate(self.struct_variables):
                for body in bodies:
                    struct_disps_ajp = body.get_struct_disps_ajp(scenario)
                    if struct_disps_ajp is not None:
                        value = np.dot(
                            struct_disps_ajp[:, findex],
                            self.scenario_data[scenario.id].c1[:, vindex],
                        )
                        func.add_gradient_component(var, value)

                    struct_temps_ajp = body.get_struct_temps_ajp(scenario)
                    if struct_temps_ajp is not None:
                        value = np.dot(
                            struct_temps_ajp[:, findex],
                            self.scenario_data[scenario.id].c2[:, vindex],
                        )
                        func.add_gradient_component(var, value)

        return

    def get_coordinate_derivatives(self, scenario, bodies, step):
        """
        Add the contributions to the gradient w.r.t. the structural coordinates
        """

        if step == 0:
            return

        adjoint_map = scenario.adjoint_map
        for ifunc, func in enumerate(scenario.adjoint_functions):
            ifull = adjoint_map[ifunc]
            for body in bodies:
                struct_shape_term = body.get_struct_coordinate_derivatives(scenario)
                struct_disps_ajp = body.get_struct_disps_ajp(scenario)
                if struct_disps_ajp is not None:
                    struct_shape_term[:, ifull] += np.dot(
                        struct_disps_ajp[:, ifunc], self.scenario_data[scenario.id].b1
                    )

                struct_temps_ajp = body.get_struct_temps_ajp(scenario)
                if struct_temps_ajp is not None:
                    struct_shape_term[:, ifull] += np.dot(
                        struct_temps_ajp[:, ifunc], self.scenario_data[scenario.id].b2
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

        struct_time = step * self.scenario_data[scenario.id].dt

        for body in bodies:
            # Perform the "analysis"
            struct_loads = body.get_struct_loads(scenario, step)
            struct_disps = body.get_struct_disps(scenario, step)
            if struct_loads is not None:
                struct_disps[:] = np.dot(
                    self.scenario_data[scenario.id].Jac1, struct_loads
                )
                struct_disps[:] += np.dot(
                    self.scenario_data[scenario.id].b1, self.struct_X
                )
                struct_disps[:] += np.dot(
                    self.scenario_data[scenario.id].c1, self.struct_dvs
                )
                if not scenario.steady:
                    struct_disps[:] += (
                        self.scenario_data[scenario.id].omega1 * struct_time
                    )

            # Perform the heat transfer "analysis"
            struct_flux = body.get_struct_heat_flux(scenario, step)
            struct_temps = body.get_struct_temps(scenario, step)
            if struct_flux is not None:
                struct_temps[:] = np.dot(
                    self.scenario_data[scenario.id].Jac2, struct_flux
                )
                struct_temps[:] += np.dot(
                    self.scenario_data[scenario.id].b2, self.struct_X
                )
                struct_temps[:] += np.dot(
                    self.scenario_data[scenario.id].c2, self.struct_dvs
                )
                if not scenario.steady:
                    struct_temps[:] += (
                        self.scenario_data[scenario.id].omega2 * struct_time
                    )

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

        # If the scenario is unsteady only add the rhs for the final state
        include_rhs = True
        if not scenario.steady and step != scenario.steps:
            include_rhs = False

        for body in bodies:
            struct_disps_ajp = body.get_struct_disps_ajp(scenario)
            struct_loads_ajp = body.get_struct_loads_ajp(scenario)
            if struct_disps_ajp is not None:
                for k, func in enumerate(scenario.adjoint_functions):
                    struct_loads_ajp[:, k] = np.dot(
                        self.scenario_data[scenario.id].Jac1.T, struct_disps_ajp[:, k]
                    )
                    if include_rhs and func.analysis_type == "structural":
                        struct_loads_ajp[:, k] += self.scenario_data[
                            scenario.id
                        ].func_coefs1

            struct_temps_ajp = body.get_struct_temps_ajp(scenario)
            struct_flux_ajp = body.get_struct_heat_flux_ajp(scenario)

            if struct_temps_ajp is not None:
                for k, func in enumerate(scenario.adjoint_functions):
                    struct_flux_ajp[:, k] = np.dot(
                        self.scenario_data[scenario.id].Jac2.T, struct_temps_ajp[:, k]
                    )
                    if include_rhs and func.analysis_type == "structural":
                        struct_flux_ajp[:, k] += self.scenario_data[
                            scenario.id
                        ].func_coefs2

        # add derivative values
        if not scenario.steady:
            if step > 0:
                self.get_function_gradients(scenario, bodies)
            else:  # step == 0
                # want to zero out adjoints used for derivatives, since no analysis done on step 0
                for body in bodies:
                    struct_disps_ajp = body.get_struct_disps_ajp(scenario)
                    if struct_disps_ajp is not None:
                        struct_disps_ajp *= 0.0

                    struct_temps_ajp = body.get_struct_temps_ajp(scenario)
                    if struct_temps_ajp is not None:
                        struct_temps_ajp *= 0.0
        fail = 0
        return fail

    def post_adjoint(self, scenario, bodies):
        return


class NullAerodynamicSolver(SolverInterface):
    def __init__(self, comm, model, auto_copy=False):
        """
        This class provides the functionality that FUNtoFEM expects from
        an aerodynamic solver, except no aerodynamics is performed here.

        All solver interface routines we just do nothing and those methods
        come from the super class SolverInterface

        Parameters
        ----------
        comm: MPI.comm
            MPI communicator
        model: :class:`~funtofem_model.FUNtoFEMmodel`
            The model containing the design data
        auto_copy: bool
            whether to copy the aero mesh from the structures mesh
        """

        self.comm = comm
        self.model = model
        self.auto_copy = auto_copy

        if auto_copy:
            self.copy_aero_mesh()
        return

    def copy_struct_mesh(self):
        """copy aero mesh from the structures side"""
        for body in self.model.bodies:
            aero_id = np.arange(1, body.get_num_struct_nodes() + 1)
            body.initialize_aero_nodes(body.struct_X, aero_id)
        return self


class TestResult:
    def __init__(
        self,
        name,
        func_names,
        complex_TD,
        adjoint_TD,
        rel_error=None,
        comm=None,
        method="complex_step",
        i_funcs=None,
        m_funcs=None,
        f_funcs=None,
        var_names=None,
        epsilon=None,
    ):
        """
        Class to store test results from complex step method
        """
        self.name = name
        self.func_names = func_names  # list of function names
        self.var_names = var_names
        self.complex_TD = complex_TD
        self.adjoint_TD = adjoint_TD
        self.method = method
        self.i_funcs = i_funcs
        self.m_funcs = m_funcs
        self.f_funcs = f_funcs
        self.epsilon = epsilon
        if rel_error is None:
            rel_error = []
            for i, _ in enumerate(self.complex_TD):
                rel_error.append(
                    TestResult.relative_error(complex_TD[i], adjoint_TD[i])
                )
        self.rel_error = rel_error
        self.comm = comm

        self.nfuncs = len(func_names)

    def set_name(self, new_name):
        self.name = new_name
        return self

    @property
    def root_proc(self) -> bool:
        return self.comm is None or self.comm.rank == 0

    def write(self, file_hdl):
        """
        write the test result out to a file handle
        """
        if self.root_proc:
            file_hdl.write(f"Test: {self.name}\n")
            if self.epsilon is not None:
                file_hdl.write(f"\tStep size: {self.epsilon}\n")
            if self.var_names is not None:
                file_hdl.write(f"\tVariables = {self.var_names}\n")
            if isinstance(self.func_names, list):
                for ifunc in range(self.nfuncs):
                    file_hdl.write(f"\tFunction {self.func_names[ifunc]}\n")
                    if self.i_funcs is not None:
                        if self.f_funcs is not None:  # if both defined write this
                            file_hdl.write(
                                f"\t\tinitial value = {self.i_funcs[ifunc]}\n"
                            )
                            if self.m_funcs is not None:
                                file_hdl.write(
                                    f"\t\tmid value = {self.m_funcs[ifunc]}\n"
                                )
                            file_hdl.write(f"\t\tfinal value = {self.f_funcs[ifunc]}\n")
                        else:
                            file_hdl.write(f"\t\tvalue = {self.i_funcs[ifunc]}\n")
                    file_hdl.write(f"\t\t{self.method} TD = {self.complex_TD[ifunc]}\n")
                    file_hdl.write(f"\t\tAdjoint TD = {self.adjoint_TD[ifunc]}\n")
                    file_hdl.write(f"\t\tRelative error = {self.rel_error[ifunc]}\n")
                file_hdl.flush()
            else:
                file_hdl.write(f"\tFunction {self.func_names}")
                if self.i_funcs is not None:
                    if self.f_funcs is not None:  # if both defined write this
                        file_hdl.write(f"\t\tinitial value = {self.i_funcs[ifunc]}\n")
                        file_hdl.write(f"\t\tfinal value = {self.f_funcs[ifunc]}\n")
                    else:
                        file_hdl.write(f"\t\tvalue = {self.i_funcs[ifunc]}\n")
                file_hdl.write(f"\t{self.method} TD = {self.complex_TD}\n")
                file_hdl.write(f"\tAdjoint TD = {self.adjoint_TD}\n")
                file_hdl.write(f"\tRelative error = {self.rel_error}\n")
                file_hdl.flush()
            file_hdl.close()
        return self

    def report(self):
        if self.root_proc:
            print(f"Test Result - {self.name}")
            print("\tFunctions = ", self.func_names)
            print(f"\t{self.method}  = ", self.complex_TD)
            print("\tAdjoint TD      = ", self.adjoint_TD)
            print("\tRelative error        = ", self.rel_error)
        return self

    @classmethod
    def relative_error(cls, truth, pred):
        if truth == 0.0 and pred == 0.0:
            print("Warning the derivative test is indeterminate!")
            return 0.0
        elif truth == 0.0 and pred != 0.0:
            return 1.0  # arbitrary 100% error provided to fail test avoiding /0
        elif abs(truth) <= 1e-8 and abs(pred) < 1e-8:
            print("Warning the derivative test has very small derivatives!")
            return pred - truth  # use absolute error if too small a derivative
        else:
            return (pred - truth) / truth

    @classmethod
    def complex_step(cls, test_name, model, driver, status_file, epsilon=1e-30):
        """
        perform complex step test on a model and driver for multiple functions & variables
        used for fun3d+tacs coupled derivative tests only...
        """

        # determine the number of functions and variables
        nfunctions = len(model.get_functions(all=True))
        nvariables = len(model.get_variables())
        func_names = [func.full_name for func in model.get_functions(all=True)]

        # generate random contravariant tensor, an input space curve tangent dx/ds for design vars
        dxds = np.random.rand(nvariables)

        # solve the adjoint
        if driver.solvers.uses_fun3d:
            driver.solvers.make_flow_real()
        driver.solve_forward()
        driver.solve_adjoint()
        model.evaluate_composite_functions()
        gradients = model.get_function_gradients(all=True)

        # compute the adjoint total derivative df/ds = df/dx * dx/ds
        adjoint_TD = np.zeros((nfunctions))
        for ifunc in range(nfunctions):
            for ivar in range(nvariables):
                adjoint_TD[ifunc] += gradients[ifunc][ivar].real * dxds[ivar]

        # perform complex step method
        print(f"uses fun3d = {driver.solvers.uses_fun3d}", flush=True)
        if driver.solvers.uses_fun3d:
            print(f"make flow complex call", flush=True)
            driver.solvers.make_flow_complex()
        variables = model.get_variables()

        # perturb the design vars by x_pert = x + 1j * h * dx/ds
        for ivar in range(nvariables):
            variables[ivar].value += 1j * epsilon * dxds[ivar]

        # run the complex step method
        driver.solve_forward()
        model.evaluate_composite_functions()
        functions = model.get_functions(all=True)

        # compute the complex step total derivative df/ds = Im{f(x+ih * dx/ds)}/h for each func
        complex_TD = np.zeros((nfunctions))
        for ifunc in range(nfunctions):
            complex_TD[ifunc] += functions[ifunc].value.imag / epsilon

        # compute rel error between adjoint & complex step for each function
        rel_error = [
            TestResult.relative_error(
                truth=complex_TD[ifunc], pred=adjoint_TD[ifunc]
            ).real
            for ifunc in range(nfunctions)
        ]

        # make test results object and write it to file
        file_hdl = open(status_file, "a") if driver.comm.rank == 0 else None
        cls(
            test_name,
            func_names,
            complex_TD,
            adjoint_TD,
            rel_error,
            comm=driver.comm,
            var_names=[var.name for var in model.get_variables()],
            i_funcs=[func.value.real for func in functions],
            f_funcs=None,
            epsilon=epsilon,
            method="complex_step",
        ).write(file_hdl).report()

        abs_rel_error = [abs(_) for _ in rel_error]
        max_rel_error = max(np.array(abs_rel_error))

        return max_rel_error

    @classmethod
    def finite_difference(
        cls,
        test_name,
        model,
        driver,
        status_file,
        epsilon=1e-5,
        central_diff=True,
        both_adjoint=False,  # have to call adjoint in both times for certain drivers
    ):
        """
        perform finite difference test on a model and driver for multiple functions & variables
        """
        nfunctions = len(model.get_functions(all=True))
        nvariables = len(model.get_variables())
        func_names = [func.full_name for func in model.get_functions(all=True)]

        # generate random contravariant tensor in input space x(s)
        if nvariables > 1:
            dxds = np.random.rand(nvariables)
        else:
            dxds = np.array([1.0])

        # central difference approximation
        variables = model.get_variables()
        # compute forward analysise f(x) and df/dx with adjoint
        driver.solve_forward()
        driver.solve_adjoint()
        gradients = model.get_function_gradients(all=True)
        m_functions = [func.value.real for func in model.get_functions(all=True)]

        # compute adjoint total derivative df/dx
        adjoint_TD = np.zeros((nfunctions))
        for ifunc in range(nfunctions):
            for ivar in range(nvariables):
                adjoint_TD[ifunc] += gradients[ifunc][ivar].real * dxds[ivar]

        # compute f(x-h)
        if central_diff:
            for ivar in range(nvariables):
                variables[ivar].value -= epsilon * dxds[ivar]
            driver.solve_forward()
            if both_adjoint:
                driver.solve_adjoint()
            i_functions = [func.value.real for func in model.get_functions(all=True)]
        else:
            i_functions = [None for func in model.get_functions()]

        # compute f(x+h)
        alpha = 2 if central_diff else 1
        for ivar in range(nvariables):
            variables[ivar].value += alpha * epsilon * dxds[ivar]
        driver.solve_forward()
        if both_adjoint:
            driver.solve_adjoint()
        f_functions = [func.value.real for func in model.get_functions(all=True)]

        finite_diff_TD = [
            (
                (f_functions[ifunc] - i_functions[ifunc]) / 2 / epsilon
                if central_diff
                else (f_functions[ifunc] - m_functions[ifunc]) / epsilon
            )
            for ifunc in range(nfunctions)
        ]

        # compute relative error
        rel_error = [
            TestResult.relative_error(
                truth=finite_diff_TD[ifunc], pred=adjoint_TD[ifunc]
            ).real
            for ifunc in range(nfunctions)
        ]

        # make test results object and write to file
        file_hdl = open(status_file, "a") if driver.comm.rank == 0 else None
        cls(
            test_name,
            func_names,
            finite_diff_TD,
            adjoint_TD,
            rel_error,
            comm=driver.comm,
            var_names=[var.name for var in model.get_variables()],
            i_funcs=i_functions,
            m_funcs=m_functions,
            f_funcs=f_functions,
            epsilon=epsilon,
            method="central_diff" if central_diff else "finite_diff",
        ).write(file_hdl).report()
        abs_rel_error = [abs(_) for _ in rel_error]
        max_rel_error = max(np.array(abs_rel_error))
        return max_rel_error

    @classmethod
    def derivative_test(
        cls, test_name, model, driver, status_file, complex_mode=True, epsilon=None
    ):
        """
        call either finite diff or complex step test depending on real mode of funtofem + TACS
        """
        if complex_mode:
            if epsilon is None:
                epsilon = 1e-30
            return cls.complex_step(
                test_name,
                model,
                driver,
                status_file,
                epsilon=epsilon,
            )
        else:
            if epsilon is None:
                epsilon = 1e-5
            return cls.finite_difference(
                test_name,
                model,
                driver,
                status_file,
                epsilon=epsilon,
            )


class CoordinateDerivativeTester:
    """
    Perform a complex step test over the coordinate derivatives of a driver
    """

    def __init__(self, driver):
        self.driver = driver
        self.comm = self.driver.comm

    @property
    def flow_solver(self):
        return self.driver.solvers.flow

    @property
    def aero_X(self):
        """aero coordinate derivatives in FUN3D"""
        return self.flow_solver.aero_X

    @property
    def struct_solver(self):
        return self.driver.solvers.structural

    @property
    def struct_X(self):
        """structure coordinates in TACS"""
        return self.struct_solver.struct_X.getArray()

    @property
    def model(self):
        return self.driver.model

    def test_struct_coordinates(
        self,
        test_name,
        status_file,
        body=None,
        scenario=None,
        epsilon=1e-30,
        complex_mode=True,
    ):
        """test the structure coordinate derivatives struct_X with complex step"""
        # assumes only one body
        if body is None:
            body = self.model.bodies[0]
        if scenario is None:  # test doesn't work for multiscenario yet
            scenario = self.model.scenarios[0]

        # random covariant tensor to aggregate derivative error among one or more functions
        # compile full struct shape term
        nf = len(self.model.get_functions())
        func_names = [func.full_name for func in self.model.get_functions()]

        # random contravariant tensor d(struct_X)/ds for testing struct shape
        dstructX_ds = np.random.rand(body.struct_X.shape[0])
        dstructX_ds_row = np.expand_dims(dstructX_ds, axis=0)

        if self.driver.solvers.uses_fun3d:
            self.driver.solvers.make_flow_real()

        """Adjoint method to compute coordinate derivatives and TD"""
        self.driver.solve_forward()
        self.driver.solve_adjoint()

        # add coordinate derivatives among scenarios
        dfdxS0 = body.get_struct_coordinate_derivatives(scenario)
        dfdx_adjoint = dstructX_ds_row @ dfdxS0
        dfdx_adjoint = list(np.reshape(dfdx_adjoint, newshape=(nf)))
        adjoint_derivs = [dfdx_adjoint[i].real for i in range(nf)]

        if complex_mode:
            """Complex step to compute coordinate total derivatives"""
            # perturb the coordinate derivatives
            if self.driver.solvers.uses_fun3d:
                self.driver.solvers.make_flow_complex()
            body.struct_X += 1j * epsilon * dstructX_ds
            self.driver.solve_forward()

            truth_derivs = np.array(
                [func.value.imag / epsilon for func in self.model.get_functions()]
            )

        else:  # central finite difference
            # f(x;xA-h)
            body.struct_X -= epsilon * dstructX_ds
            self.driver.solve_forward()
            i_functions = [func.value.real for func in self.model.get_functions()]

            # f(x;xA+h)
            body.struct_X += 2 * epsilon * dstructX_ds
            self.driver.solve_forward()
            f_functions = [func.value.real for func in self.model.get_functions()]

            truth_derivs = np.array(
                [
                    (f_functions[i] - i_functions[i]) / 2 / epsilon
                    for i in range(len(self.model.get_functions()))
                ]
            )

        rel_error = [
            TestResult.relative_error(truth_derivs[i], adjoint_derivs[i])
            for i in range(nf)
        ]

        # make test results object and write it to file
        file_hdl = open(status_file, "a") if self.comm.rank == 0 else None
        TestResult(
            test_name,
            func_names,
            truth_derivs,
            adjoint_derivs,
            rel_error,
            method="complex_step" if complex_mode else "finite_diff",
            epsilon=epsilon,
        ).write(file_hdl).report()

        abs_rel_error = [abs(_) for _ in rel_error]
        max_rel_error = max(np.array(abs_rel_error))

        return max_rel_error

    def test_aero_coordinates(
        self,
        test_name,
        status_file,
        scenario=None,
        body=None,
        epsilon=1e-30,
        complex_mode=True,
    ):
        """test the structure coordinate derivatives struct_X with complex step"""
        # assumes only one body
        if body is None:
            body = self.model.bodies[0]
        if scenario is None:  # test doesn't work for multiscenario yet
            scenario = self.model.scenarios[0]

        # random covariant tensor to aggregate derivative error among one or more functions
        # compile full struct shape term
        nf = len(self.model.get_functions())
        func_names = [func.full_name for func in self.model.get_functions()]

        # random contravariant tensor d(aero_X)/ds for testing aero shape
        daeroX_ds = np.random.rand(body.aero_X.shape[0])
        daeroX_ds_row = np.expand_dims(daeroX_ds, axis=0)

        if self.driver.solvers.uses_fun3d:
            self.driver.solvers.make_flow_real()

        """Adjoint method to compute coordinate derivatives and TD"""
        self.driver.solve_forward()
        self.driver.solve_adjoint()

        # add coordinate derivatives among scenarios
        dfdxA0 = body.get_aero_coordinate_derivatives(scenario)
        dfdx_adjoint = daeroX_ds_row @ dfdxA0
        dfdx_adjoint = list(np.reshape(dfdx_adjoint, newshape=(nf)))
        adjoint_derivs = [dfdx_adjoint[i].real for i in range(nf)]

        if complex_mode:
            """Complex step to compute coordinate total derivatives"""
            # perturb the coordinate derivatives
            if self.driver.solvers.uses_fun3d:
                self.driver.solvers.make_flow_complex()
            body.aero_X += 1j * epsilon * daeroX_ds
            self.driver.solve_forward()

            truth_derivs = np.array(
                [func.value.imag / epsilon for func in self.model.get_functions()]
            )

        else:  # central finite difference
            # f(x;xA-h)
            body.aero_X -= epsilon * daeroX_ds
            self.driver.solve_forward()
            i_functions = [func.value.real for func in self.model.get_functions()]

            # f(x;xA+h)
            body.aero_X += 2 * epsilon * daeroX_ds
            self.driver.solve_forward()
            f_functions = [func.value.real for func in self.model.get_functions()]

            truth_derivs = np.array(
                [
                    (f_functions[i] - i_functions[i]) / 2 / epsilon
                    for i in range(len(self.model.get_functions()))
                ]
            )

        rel_error = [
            TestResult.relative_error(truth_derivs[i], adjoint_derivs[i])
            for i in range(nf)
        ]

        # make test results object and write it to file
        file_hdl = open(status_file, "a") if self.comm.rank == 0 else None
        TestResult(
            test_name,
            func_names,
            truth_derivs,
            adjoint_derivs,
            rel_error,
            comm=self.comm,
            method="complex_step" if complex_mode else "finite_diff",
            epsilon=epsilon,
        ).write(file_hdl).report()

        abs_rel_error = [abs(_) for _ in rel_error]
        max_rel_error = max(np.array(abs_rel_error))

        return max_rel_error
