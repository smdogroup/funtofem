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


from __future__ import print_function

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
        comm,
        struct_comm,
        struct_root,
        aero_comm,
        aero_root,
        transfer_options=None,
        model=None,
    ):
        """
        Parameters
        ----------
        solvers: dict
           the various disciplinary solvers
        comm: MPI.comm
            MPI communicator
        transfer_options: dict
            options of the load and displacement transfer scheme
        model: :class:`~funtofem_model.FUNtoFEMmodel`
            The model containing the design data
        """

        # communicator
        self.comm = comm
        self.aero_comm = aero_comm
        self.aero_root = aero_root
        self.struct_comm = struct_comm
        self.struct_root = struct_root

        # Solver classes
        self.solvers = solvers

        # Make a fake model if not given one
        if model is not None:
            self.fakemodel = False
        else:
            print("FUNtoFEM driver: generating fake model")
            from pyfuntofem.model import FUNtoFEMmodel, Body, Scenario, Function

            model = FUNtoFEMmodel("fakemodel")

            fakebody = Body("fakebody")
            model.add_body(fakebody)

            fakescenario = Scenario("fakescenario")
            function = Function("cl", analysis_type="aerodynamic")
            fakescenario.add_function(function)
            model.add_scenario(fakescenario)

            self.fakemodel = True

        self.model = model

        # Initialize transfer scheme in each body class
        for body in self.model.bodies:
            body.initialize_transfer(
                comm,
                struct_comm,
                struct_root,
                aero_comm,
                aero_root,
                transfer_options=transfer_options,
            )

        # Initialize the shape parameterization
        for body in self.model.bodies:
            body.initialize_shape_parameterization()

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
                self._eval_functions(scenario, self.model.bodies)

        return fail

    def solve_adjoint(self):
        """
        Solves the coupled adjoint problem and computes gradients
        """

        fail = 0
        # Make sure we have functions defined before we start the adjoint
        if self.fakemodel:
            print("Aborting: attempting to run FUNtoFEM adjoint with no model defined")
            quit()
        elif not self.model.get_functions():
            print(
                "Aborting: attempting to run FUNtoFEM adjoint with no functions defined"
            )
            quit()

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

            self._eval_function_grads(scenario)

        self.model.enforce_coupling_derivatives()
        return fail

    def _initialize_forward(self, scenario, bodies):
        for body in bodies:
            body.initialize_variables(scenario)

        for solver in self.solvers.keys():
            fail = self.solvers[solver].initialize(scenario, bodies)
            if fail != 0:
                return fail
        return 0

    def _initialize_adjoint(self, scenario, bodies):
        for body in bodies:
            body.initialize_adjoint_variables(scenario)

        for solver in self.solvers.keys():
            fail = self.solvers[solver].initialize_adjoint(scenario, bodies)
            if fail != 0:
                return fail
        return 0

    def _post_forward(self, scenario, bodies):
        for solver in self.solvers.keys():
            self.solvers[solver].post(scenario, bodies)

    def _post_adjoint(self, scenario, bodies):
        for solver in self.solvers.keys():
            self.solvers[solver].post_adjoint(scenario, bodies)

    def _distribute_functions(self, scenario, bodies):
        for solver in self.solvers.keys():
            self.solvers[solver].set_functions(scenario, bodies)

    def _distribute_variables(self, scenario, bodies):
        for solver in self.solvers.keys():
            self.solvers[solver].set_variables(scenario, bodies)

    def _eval_functions(self, scenario, bodies):
        for solver in self.solvers.keys():
            self.solvers[solver].get_functions(scenario, self.model.bodies)

    def _eval_function_grads(self, scenario):
        offset = self._get_scenario_function_offset(scenario)
        for solver in self.solvers.keys():
            self.solvers[solver].get_function_gradients(
                scenario, self.model.bodies, offset
            )

        for body in self.model.bodies:
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
        for solver in self.solvers.keys():
            self.solvers[solver].get_coordinate_derivatives(
                scenario, self.model.bodies, step
            )

        # transfer scheme contributions to the coordinates derivatives
        if step > 0:
            for body in self.model.bodies:
                if body.transfer:
                    # Aerodynamic coordinate derivatives
                    temp = np.zeros((3 * body.aero_nnodes), dtype=TransferScheme.dtype)
                    for func in range(nfunctions):
                        # Load transfer term
                        body.transfer.applydLdxA0(
                            body.psi_L[:, func].copy(order="C"), temp
                        )
                        body.aero_shape_term[:, func] += temp.copy()

                        # Displacement transfer term
                        body.transfer.applydDdxA0(
                            body.psi_D[:, func].copy(order="C"), temp
                        )
                        body.aero_shape_term[:, func] += temp.copy()

                    # Structural coordinate derivatives
                    temp = np.zeros(
                        body.struct_nnodes * body.xfer_ndof, dtype=TransferScheme.dtype
                    )
                    for func in range(nfunctions):
                        # Load transfer term
                        body.transfer.applydLdxS0(
                            body.psi_L[:, func].copy(order="C"), temp
                        )
                        body.struct_shape_term[:, func] += temp.copy()

                        # Displacement transfer term
                        body.transfer.applydDdxS0(
                            body.psi_D[:, func].copy(order="C"), temp
                        )
                        body.struct_shape_term[:, func] += temp.copy()

        return

    def _solve_steady_forward(self, scenario, steps):
        return 1

    def _solve_unsteady_forward(self, scenario, steps):
        return 1

    def _solve_steady_adjoint(self, scenario):
        return 1

    def _solve_unsteady_adjoint(self, scenario):
        return 1
