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

__all__ = ["FUNtoFEMDriver"]

import numpy as np
from mpi4py import MPI
from funtofem import TransferScheme

try:
    from .hermes_transfer import HermesTransfer
except:
    pass

np.set_printoptions(precision=15)


class FUNtoFEMDriver(object):
    """
    The FUNtoFEM driver base class has all of the driver except for the coupling algorithms
    """

    def __init__(
        self,
        solvers,
        comm_manager=None,
        transfer_settings=None,
        model=None,
    ):
        """
        Parameters
        ----------
        solvers: SolverManager
           the various disciplinary solvers
        comm_manager: CommManager
            manager of discipline communicators
        transfer_settings: TransferSettings
            options of the load and displacement transfer scheme
        model: :class:`~funtofem_model.FUNtoFEMmodel`
            The model containing the design data
        """

        # add the comm manger
        if comm_manager is not None:
            comm_manager = comm_manager
        else:
            # use default comm manager from solvers if not available
            comm_manager = solvers.comm_manager
        self.comm_manager = comm_manager

        # communicator
        self.comm = comm_manager.master_comm
        self.aero_comm = comm_manager.aero_comm
        self.aero_root = comm_manager.aero_root
        self.struct_comm = comm_manager.struct_comm
        self.struct_root = comm_manager.struct_root

        # use default transfer settings

        # SolverManager class
        self.solvers = solvers

        # Make a fake model if not given one
        if model is not None:
            self.fakemodel = False
        else:
            print("FUNtoFEM driver: generating a default model")
            from pyfuntofem.model import FUNtoFEMmodel, Body, Scenario, Function

            model = FUNtoFEMmodel("model")
            fakebody = Body("body")
            model.add_body(fakebody)

            fakescenario = Scenario("scenario")
            function = Function("cl", analysis_type="aerodynamic")
            fakescenario.add_function(function)
            model.add_scenario(fakescenario)

            self.fakemodel = True

        self.model = model

        # Initialize transfer scheme in each body class
        for body in self.model.bodies:
            body.initialize_transfer(
                self.comm,
                self.struct_comm,
                self.struct_root,
                self.aero_comm,
                self.aero_root,
                transfer_settings=transfer_settings,
            )

        # Initialize the shape parameterization
        for body in self.model.bodies:
            body.initialize_shape_parameterization()

        return

    def update_model(self, model):
        """
        Update the model object that the driver sees

        Parameters
        ----------
        model: FUNtoFEM model type
        """
        self.model = model

    def solve_forward(self, steps=None):
        """
        Solves the coupled forward problem

        Parameters
        ----------
        steps: int
            number of coupled solver steps. Only for use if a FUNtoFEM model is not defined
        """
        fail = 0

        complex_run = False
        if TransferScheme.dtype == np.complex128 or TransferScheme.dtype == complex:
            complex_run = True

        # update the shapes first
        for body in self.model.bodies:
            body.update_shape(complex_run)

        # loop over the forward problem for the different scenarios
        for scenario in self.model.scenarios:
            # tell the solvers what the variable values and functions are for this scenario
            if not self.fakemodel:
                self._distribute_variables(scenario, self.model.bodies)
                self._distribute_functions(scenario, self.model.bodies)

            # Set the new meshes Initialize the forward solvers
            fail = self._initialize_forward(scenario, self.model.bodies)
            if fail != 0:
                if self.comm.Get_rank() == 0:
                    print("Fail flag return during initialization")

            # Update transfer postions to the initial conditions
            for body in self.model.bodies:
                body.update_transfer()

            if scenario.steady:
                fail = self._solve_steady_forward(scenario, steps)
                if fail != 0:
                    if self.comm.Get_rank() == 0:
                        print("Fail flag return during forward solve")
            else:
                fail = self._solve_unsteady_forward(scenario, steps)
                if fail != 0:
                    if self.comm.Get_rank() == 0:
                        print("Fail flag return during forward solve")

            # Perform any operations after the forward solve
            self._post_forward(scenario, self.model.bodies)
            if fail == 0:
                self._get_functions(scenario, self.model.bodies)

        return fail

    def solve_adjoint(self):
        """
        Solves the coupled adjoint problem and computes gradients
        """

        fail = 0

        # Get the list of functions
        functions = self.model.get_functions()

        # Make sure we have functions defined before we start the adjoint
        if self.fakemodel:
            print("Aborting: attempting to run FUNtoFEM adjoint with no model defined")
            quit()
        elif len(functions) == 0:
            print(
                "Aborting: attempting to run FUNtoFEM adjoint with no functions defined"
            )
            quit()

        # Zero the derivative values stored in the function
        for func in functions:
            func.zero_derivatives()

        # Set the functions into the solvers
        for scenario in self.model.scenarios:
            # tell the solvers what the variable values and functions are for this scenario
            self._distribute_variables(scenario, self.model.bodies)
            self._distribute_functions(scenario, self.model.bodies)

            # Initialize the adjoint solvers
            self._initialize_adjoint_variables(scenario, self.model.bodies)
            self._initialize_adjoint(scenario, self.model.bodies)

            if scenario.steady:
                fail = self._solve_steady_adjoint(scenario)
                if fail != 0:
                    return fail
            else:
                fail = self._solve_unsteady_adjoint(scenario)
                if fail != 0:
                    return fail

            # Perform any operations after the adjoint solve
            self._post_adjoint(scenario, self.model.bodies)

            self._get_function_grads(scenario)

        return fail

    def _initialize_forward(self, scenario, bodies):
        """
        Initialize the variables and solver data for a forward analysis
        """
        for body in bodies:
            body.initialize_variables(scenario)

        for solver in self.solvers.solver_list:
            fail = solver.initialize(scenario, bodies)
            if fail != 0:
                return fail
        return 0

    def _initialize_adjoint(self, scenario, bodies):
        """
        Initialize the variables and solver data for an adjoint solve
        """
        for body in bodies:
            body.initialize_adjoint_variables(scenario)

        for solver in self.solvers.solver_list:
            fail = solver.initialize_adjoint(scenario, bodies)
            if fail != 0:
                return fail
        return 0

    def _post_forward(self, scenario, bodies):
        for solver in self.solvers.solver_list:
            solver.post(scenario, bodies)

    def _post_adjoint(self, scenario, bodies):
        for solver in self.solvers.solver_list:
            solver.post_adjoint(scenario, bodies)

    def _distribute_functions(self, scenario, bodies):
        for solver in self.solvers.solver_list:
            solver.set_functions(scenario, bodies)

    def _distribute_variables(self, scenario, bodies):
        for solver in self.solvers.solver_list:
            solver.set_variables(scenario, bodies)

    def _get_functions(self, scenario, bodies):
        for solver in self.solvers.solver_list:
            solver.get_functions(scenario, self.model.bodies)

    def _get_function_grads(self, scenario):
        # Set the function gradients into the scenario and body classes
        bodies = self.model.bodies
        for solver in self.solvers.solver_list:
            solver.get_function_gradients(scenario, bodies)

        offset = self._get_scenario_function_offset(scenario)
        for body in bodies:
            body.shape_derivative(scenario, offset)

    def _get_scenario_function_offset(self, scenario):
        """
        The offset tells each scenario what is first function's index is
        """
        offset = 0
        for i in range(scenario.id - 1):
            offset += self.model.scenarios[i].count_functions()

        return offset

    def _extract_coordinate_derivatives(self, scenario, bodies, step):
        nfunctions = scenario.count_adjoint_functions()

        # get the contributions from the solvers
        for solver in self.solvers.solver_list:
            solver.get_coordinate_derivatives(scenario, self.model.bodies, step)

        # transfer scheme contributions to the coordinates derivatives
        if step > 0:
            for body in self.model.bodies:
                body.add_coordinate_derivative(scenario, step)

        return

    def _solve_steady_forward(self, scenario, steps):
        return 1

    def _solve_unsteady_forward(self, scenario, steps):
        return 1

    def _solve_steady_adjoint(self, scenario):
        return 1

    def _solve_unsteady_adjoint(self, scenario):
        return 1
