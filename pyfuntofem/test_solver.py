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
from funtofem import TransferScheme
from pyfuntofem.solver_interface import SolverInterface


class TestSolver(SolverInterface):
    def __init__(self, comm, model, solver="flow"):
        """
        A test solver that provides the functionality that FUNtoFEM expects from
        either a structrual solver or a flow solver.

        Parameters
        ----------
        comm: MPI.comm
            MPI communicator
        model: :class:`~funtofem_model.FUNtoFEMmodel`
            The model containing the design data
        solver: str
            String indicating whether this is a "flow" or "structural" solver
        """

        self.comm = comm
        self.solver = solver  # Solver type

        self.npts = 10
        np.random.seed(0)

        self.b1 = 0.01 * (np.random.rand(3 * self.npts) - 0.5)
        self.Jac1 = 0.01 * (np.random.rand(3 * self.npts, 3 * self.npts) - 0.5)

        self.b2 = 0.1 * (np.random.rand(self.npts) - 0.5)
        self.Jac2 = 0.05 * (np.random.rand(self.npts, self.npts) - 0.5)

        # Data for output functional values
        self.func_coefs = []

        # Initialize the coordinates of the aerodynamic or structural mesh
        for body in model.bodies:
            if self.solver == "flow":
                X = np.random.rand(3 * self.npts)
                body.initialize_aero_mesh(X)
            else:
                X = np.random.rand(3 * self.npts)
                body.initialize_struct_mesh(X)

        return

    def initialize(self, scenario, bodies):
        """Note that this function must return a fail flag of zero on success"""
        return 0

    def initialize_adjoint(self, scenario, bodies):
        """Note that this function must return a fail flag of zero on success"""
        return 0

    def set_variables(self, scenario, bodies):
        pass

    def set_functions(self, scenario, bodies):
        """
        Set the functions to be used for the given scenario.

        In this function, each discipline should initialize the data needed to evaluate
        the given set of functions set in each scenario.
        """

        for func in scenario.functions:
            if self.solver == "flow" and func.analysis_type == "aerodynamic":
                self.func_coefs.append(np.random.rand(self.npts))
            elif self.solver == "structural" and func.analysis_type == "structural":
                self.func_coefs.append(np.random.rand(self.npts))
            else:
                self.func_coefs.append(None)

        return

    def get_functions(self, scenario, bodies):
        """
        Evaluate the functions of interest and set the function values into
        the scenario.functions objects
        """

        for index, func in enumerate(scenario.functions):
            if self.solver == "flow" and func.analysis_type == "aerodynamic":
                value = 0.0
                for body in bodies:
                    aero_disps = body.get_aero_disps(scenario)
                    value += np.dot(self.func_coefs[index], aero_disps)
                func.value = self.comm.allreduce(value)
            elif self.solver == "structural" and func.analysis_type == "structural":
                value = 0.0
                for body in bodies:
                    struct_loads = body.get_struct_loads(scenario)
                    value += np.dot(self.func_coefs[index], struct_loads)
                func.value = self.comm.allreduce(value)

        return

    def get_function_gradients(self, scenario, bodies, offset):
        for index, function in enumerate(scenario.functions):
            # Do the scenario variables first
            for vartype in scenario.variables:
                if vartype == "aerodynamic":
                    for i, var in enumerate(scenario.variables[vartype]):
                        if var.active:
                            if function.adjoint:
                                if var.id <= 6:
                                    scenario.derivatives[vartype][offset + func][
                                        i
                                    ] = interface.design_pull_global_derivative(
                                        function.id, var.id
                                    )
                                elif var.name.lower() == "dynamic pressure":
                                    scenario.derivatives[vartype][offset + func][
                                        i
                                    ] = self.comm.reduce(self.dFdqinf[func])
                                    scenario.derivatives[vartype][offset + func][
                                        i
                                    ] = self.comm.reduce(self.dHdq[func])
                            else:
                                scenario.derivatives[vartype][offset + func][i] = 0.0
                            scenario.derivatives[vartype][offset + func][
                                i
                            ] = self.comm.bcast(
                                scenario.derivatives[vartype][offset + func][i], root=0
                            )

            for body in bodies:
                for vartype in body.variables:
                    if vartype == "rigid_motion":
                        for i, var in enumerate(body.variables[vartype]):
                            if var.active:
                                # rigid motion variables are not active in funtofem path
                                body.derivatives[vartype][offset + func][i] = 0.0

        return scenario, bodies

    def get_coordinate_derivatives(self, scenario, bodies, step):
        pass

    def iterate(self, scenario, bodies, step):
        """
        Iterate for the aerodynamic or structural solver
        """

        if self.solver == "flow":
            for body in bodies:
                # Perform the "analysis"
                # Note that the body class expects that you will write the new
                # aerodynamic loads into the aero_loads array.
                aero_disps = body.get_aero_disps(scenario)
                aero_loads = body.get_aero_loads(scenario)
                if aero_disps is not None:
                    aero_loads[:] = np.dot(self.Jac1, aero_disps) + self.b1

                # Perform the heat transfer "analysis"
                aero_temps = body.get_aero_temps(scenario)
                aero_flux = body.get_aero_heat_flux(scenario)
                if aero_temps is not None:
                    aero_flux[:] = np.dot(self.Jac2, aero_temps) + self.b2
        else:  # The solver is a structural solver
            for body in bodies:
                # Perform the "analysis"
                struct_loads = body.get_struct_loads(scenario)
                struct_disps = body.get_struct_disps(scenario)
                if struct_loads is not None:
                    struct_disps[:] = np.dot(self.Jac1, struct_loads) + self.b1

                # Perform the heat transfer "analysis"
                struct_flux = body.get_struct_heat_flux(scenario)
                struct_temps = body.get_struct_temps(scenario)
                if struct_flux is not None:
                    struct_temps[:] = np.dot(self.Jac2, struct_flux) + self.b2

        # This analysis is always successful so return fail = 0
        fail = 0
        return fail

    def post(self, scenario, bodies):
        pass

    def set_states(self, scenario, bodies, step):
        pass

    def iterate_adjoint(self, scenario, bodies, step):
        fail = 0

        # if self.solver == "flow":
        #     for body in bodies:
        #         # Solve for the "flow" adjoint
        #         adjR_rhs

        # else: # The solver is a structural solver
        #     for body in bodies:
        #         adjS_rhs = body.get

        # nfunctions = scenario.count_adjoint_functions()
        # for ibody, body in enumerate(bodies, 1):
        #     if body.aero_nnodes > 0:
        #         # Solve the force and heat flux adjoint equation
        #         if (
        #             body.analysis_type == "aeroelastic"
        #             or body.analysis_type == "aerothermoelastic"
        #         ):
        #             psi_F = -body.dLdfa

        #         if (
        #             body.analysis_type == "aerothermal"
        #             or body.analysis_type == "aerothermoelastic"
        #         ):
        #             psi_Q = -body.dQdfta

        # for ibody, body in enumerate(bodies, 1):
        #     if body.aero_nnodes > 0:

        #         if body.thermal_transfer is not None:
        #             for func in range(nfunctions):
        #                 if (
        #                     body.analysis_type == "aeroelastic"
        #                     or body.analysis_type == "aerothermoelastic"
        #                 ):
        #                     body.dGdua[:, func] = np.dot(self.Jac1.T, psi_F[:, func])

        #                 if (
        #                     body.analysis_type == "aerothermal"
        #                     or body.analysis_type == "aerothermoelastic"
        #                 ):
        #                     body.dAdta[:, func] = np.dot(self.Jac2.T, psi_Q[:, func])

        return fail

    def post_adjoint(self, scenario, bodies):
        pass

    def step_pre(self, scenario, bodies, step):
        return 0

    def step_post(self, scenario, bodies, step):
        return 0
