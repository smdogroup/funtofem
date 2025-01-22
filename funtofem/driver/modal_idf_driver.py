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

__all__ = ["FUNtoFEMmodal"]

import numpy as np
from mpi4py import MPI
from funtofem import TransferScheme
from ._funtofem_driver import FUNtoFEMDriver
from ..optimization.optimization_manager import OptimizationManager
from ..interface.utils.general_utils import real_norm, imag_norm

try:
    from .hermes_transfer import HermesTransfer
except:
    pass


class FUNtoFEMmodalDriver(FUNtoFEMDriver):
    def __init__(
        self,
        solvers,
        comm_manager=None,
        transfer_settings=None,
        model=None,
        debug=False,
        reload_funtofem_states=False,
    ):
        """
        Driver that is based on FUNtoFEMnlbgs, but uses an IDF
        implementation to solve the coupled problem and a modal
        reconstruction of the structural deformation.

        Parameters
        ----------
        solvers: SolverManager
           the various disciplinary solvers
        comm_manager: CommManager
            manager for various discipline communicators
        transfer_settings: TransferSettings
            options of the load and displacement transfer scheme
        model: :class:`~funtofem_model.FUNtoFEMmodel`
            The model containing the design data
        reload_funtofem_states: bool
            whether to save and reload funtofem states
        """

        super(FUNtoFEMmodalDriver, self).__init__(
            solvers,
            comm_manager=comm_manager,
            transfer_settings=transfer_settings,
            model=model,
            debug=debug,
            reload_funtofem_states=reload_funtofem_states,
        )

        return

    @property
    def manager(self, hot_start: bool = False, write_designs: bool = True):
        """
        Create an optimization manager object for this driver
        """
        return OptimizationManager(
            driver=self,
            write_designs=write_designs,
            hot_start=hot_start,
        )

    def _initialize_adjoint_variables(self, scenario, bodies):
        """
        Initialize the adjoint variables stored in the body.

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The scenario
        bodies: :class:`~body.Body`
            List of FUNtoFEM bodies.
        """

        for body in bodies:
            body.initialize_adjoint_variables(scenario)

        return

    def _solve_steady_forward(self, scenario, steps=None):
        """
        Evaluate the aerothermoelastic forward analysis. Does *not* solve
        the coupled problem.
        Evaluation path for e.g., aeroelastic is D->A->L->S.

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        steps: int
            Number of iterations if not set by the model
        """

        assert scenario.steady
        fail = 0

        # Transfer modal displacements and temperatures from structure to aerodynamic mesh
        for body in self.model.bodies:
            # Get the modal coordinates on the structure mesh and compute the product
            # to get effective disps.
            body.convert_modal_disps(scenario)
            body.convert_modal_temps(scenario)
            # At this stage, we've recreated u_s and T_s

            body.transfer_disps(scenario)
            body.transfer_temps(scenario)

        # Solve the flow problem
        for step in range(1, scenario.steps + 1):
            fail = self.solvers.flow.iterate(scenario, self.model.bodies, step)

            fail = self.comm.allreduce(fail)
            if fail != 0:
                if self.comm.Get_rank() == 0:
                    print("Flow solver returned fail flag.")
                return fail

        # Transfer forces and heat fluxes from aerodynamic to structure mesh
        for body in self.model.bodies:
            body.transfer_loads(scenario)
            body.transfer_heat_flux(scenario)

        # Solve the structure problem
        fail = self.solvers.structural.iterate(scenario, self.model.bodies, step)

        fail = self.comm.allreduce(fail)
        if fail != 0:
            if self.comm.Get_rank() == 0:
                print("Structural solver returned fail flag.")
            return fail

        # Additional computation to transpose modal coordinate matrix
        for body in self.model.bodies:
            body.convert_modal_disps_transpose(scenario)
            body.convert_modal_temps_transpose(scenario)

        return fail

    def _solve_steady_adjoint(self, scenario):
        """
        Evaluate the aerothermoelastic adjoint analysis. Does *not* solve
        the coupled problem.
        Evaluation path for e.g., aeroelastic is S^bar->L^bar->A^bar->D^bar.

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        """

        assert scenario.steady
        fail = 0

        # Load the current state
        for body in self.model.bodies:
            body.transfer_disps(scenario)
            body.transfer_loads(scenario)

            body.transfer_temps(scenario)
            body.transfer_heat_flux(scenario)

        # Initialize the adjoint variables
        self._initialize_adjoint_variables(scenario, self.model.bodies)

        # Take a step in the structural adjoint
        fail = self.solvers.structural.iterate_adjoint(
            scenario, self.model.bodies, step=0
        )

        # Solve the flow adjoint
        for step in range(1, scenario.adjoint_steps + 1):
            # Get force and heat flux terms for flow solver
            for body in self.model.bodies:
                body.transfer_loads_adjoint(scenario)
                body.transfer_heat_flux_adjoint(scenario)

            fail = self.solvers.flow.iterate_adjoint(scenario, self.model.bodies, step)

            fail = self.comm.allreduce(fail)
            if fail != 0:
                if self.comm.Get_rank() == 0:
                    print("Flow solver returned fail flag.")
                return fail

        ###

        # loop over the adjoint NLBGS solver in a loose coupling phase
        for i, nlbgs_steps in enumerate(
            [scenario.adjoint_steps, scenario.post_tight_adjoint_steps]
        ):
            if i == 0:  # loose coupling phase
                start = 1
            else:  # tight coupling phase
                self.solvers.flow.initialize_adjoint_tight_coupling(scenario)
                start = self.solvers.flow.get_last_adjoint_step()

            for step in range(start, nlbgs_steps + start):
                # Get force and heat flux terms for the flow solver
                for body in self.model.bodies:
                    body.transfer_loads_adjoint(scenario)
                    body.transfer_heat_flux_adjoint(scenario)

                # Iterate over the aerodynamic adjoint
                fail = self.solvers.flow.iterate_adjoint(
                    scenario, self.model.bodies, step
                )

                fail = self.comm.allreduce(fail)
                if fail != 0:
                    if self.comm.Get_rank() == 0:
                        print("Flow solver returned fail flag")
                    return fail

                # Get the structural adjoint rhs
                for body in self.model.bodies:
                    body.transfer_disps_adjoint(scenario)
                    body.transfer_temps_adjoint(scenario)

                # take a step in the structural adjoint
                fail = self.solvers.structural.iterate_adjoint(
                    scenario, self.model.bodies, step
                )

                fail = self.comm.allreduce(fail)
                if fail != 0:
                    if self.comm.Get_rank() == 0:
                        print("Structural solver returned fail flag")
                    return fail

                for body in self.model.bodies:
                    body.aitken_adjoint_relax(self.comm, scenario)

                # check for early stopping criterion, exit if meets criterion
                exit_early = False
                # only exit early in the loose coupling phase
                if (
                    scenario.early_stopping
                    and step > scenario.min_adjoint_steps
                    and i == 0
                ):
                    all_converged = True  # assume all converged until proven otherwise (then when one isn't exit for loop)
                    for isolver, solver in enumerate(self.solvers.solver_list):
                        adjoint_resid = abs(solver.get_adjoint_residual(step=step))
                        adjoint_tol = solver.adjoint_tolerance
                        if self.comm.rank == 0:
                            print(
                                f"f2f scenario {scenario.name}, adjoint resid = {adjoint_resid}",
                                flush=True,
                            )

                        if adjoint_resid > adjoint_tol:
                            all_converged = False

                    if all_converged:
                        exit_early = True
                        if exit_early and self.comm.rank == 0:
                            print(
                                f"F2F Steady Adjoint analysis of scenario {scenario.name}"
                            )
                            print(
                                f"\texited early at step {step} with tolerance {adjoint_resid} < {adjoint_tol}",
                                flush=True,
                            )

                if exit_early:
                    break

        # are we then extracting coordinate derivatives at correct time step (doesn't actually use time step for steady case here)
        steps = (
            scenario.adjoint_steps * scenario.adjoint_coupling_frequency
            + scenario.post_tight_adjoint_steps
        )
        self._extract_coordinate_derivatives(scenario, self.model.bodies, steps)
        return 0

    def _solve_unsteady_forward(self, scenario, steps=None):
        """
        This function solves the unsteady forward problem using NLBGS without FSI subiterations

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            the current scenario
        steps: int
            number of time steps if not using the value defined in the scenario

        Returns
        -------
        fail: int
            fail flag for the coupled solver

        """

        pass

    def _solve_unsteady_adjoint(self, scenario):
        """
        Solves the unsteady adjoint problem using LBGS without FSI subiterations

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            the current scenario
        steps: int
            number of time steps

        Returns
        -------
        fail: int
            fail flag

        """

        pass
