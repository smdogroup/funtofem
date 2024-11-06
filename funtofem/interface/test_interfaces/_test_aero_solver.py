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
    "NullAerodynamicSolver",
]

import numpy as np
from funtofem import TransferScheme
from .._solver_interface import SolverInterface
import os


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
