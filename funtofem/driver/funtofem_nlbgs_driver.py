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

__all__ = ["FUNtoFEMnlbgs"]

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


class FUNtoFEMnlbgs(FUNtoFEMDriver):
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
        The FUNtoFEM driver for the Nonlinear Block Gauss-Seidel
        solvers for steady and unsteady coupled adjoint.

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

        super(FUNtoFEMnlbgs, self).__init__(
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
        Initialize the adjoint variables

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
        Solve the aerothermoelastic forward analysis using the nonlinear block Gauss-Seidel algorithm.
        Aitken under-relaxation is used here for stabilty.

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        steps: int
            Number of iterations if not set by the model
        """

        assert scenario.steady
        fail = 0

        # # Determine if we're using the scenario's number of steps or the argument
        # if steps is None:
        #     steps = scenario.steps

        # flow uncoupled steps (mainly for aerothermal and aerothermoelastic analysis)
        for step in range(1, scenario.uncoupled_steps + 1):
            # Take a step in the flow solver for (just aerodynamic iteration)
            fail = self.solvers.flow.uncoupled_iterate(
                scenario, self.model.bodies, step
            )

            # exit with failure if the flow iteration failed
            fail = self.comm.allreduce(fail)
            if fail != 0:
                if self.comm.Get_rank() == 0:
                    print("Flow solver returned fail flag")
                return fail

        # Loop over the NLBGS steps in a loose coupling phase then tight coupling phase
        for i, nlbgs_steps in enumerate(
            [scenario.steps, scenario.post_tight_forward_steps]
        ):
            if i == 1:
                self.solvers.flow.initialize_forward_tight_coupling(scenario)

            for step in range(1, nlbgs_steps + 1):
                # Transfer displacements and temperatures
                for body in self.model.bodies:
                    body.transfer_disps(scenario)
                    body.transfer_temps(scenario)

                # Take a step in the flow solver
                fail = self.solvers.flow.iterate(scenario, self.model.bodies, step)

                fail = self.comm.allreduce(fail)
                if fail != 0:
                    if self.comm.Get_rank() == 0:
                        print("Flow solver returned fail flag")
                    return fail

                # Transfer the loads and heat flux
                for body in self.model.bodies:
                    body.transfer_loads(scenario)
                    body.transfer_heat_flux(scenario)

                    if self._debug:
                        struct_loads = body.get_struct_loads(scenario)
                        aero_loads = body.get_aero_loads(scenario)
                        print(f"========================================")
                        print(f"Inside nlbgs driver, step: {step}")
                        if struct_loads is not None:
                            print(
                                f"norm of real struct_loads: {real_norm(struct_loads)}"
                            )
                            print(
                                f"norm of imaginary struct_loads: {imag_norm(struct_loads)}"
                            )
                        print(f"aero_loads: {aero_loads}")
                        if aero_loads is not None:
                            print(f"norm of real aero_loads: {real_norm(aero_loads)}")
                            print(
                                f"norm of imaginary aero_loads: {imag_norm(aero_loads)}"
                            )
                        print(f"========================================\n", flush=True)

                # Take a step in the FEM model
                fail = self.solvers.structural.iterate(
                    scenario, self.model.bodies, step
                )

                fail = self.comm.allreduce(fail)
                if fail != 0:
                    if self.comm.Get_rank() == 0:
                        print("Structural solver returned fail flag")
                    return fail

                # Under-relaxation for solver stability
                for body in self.model.bodies:
                    body.aitken_relax(self.comm, scenario)

                # check for early stopping criterion, exit if meets criterion
                exit_early = False
                # only exit early in the loose coupling phase
                if (
                    scenario.early_stopping
                    and step > scenario.min_forward_steps
                    and i == 0
                ):
                    all_converged = True
                    for solver in self.solvers.solver_list:
                        forward_resid = abs(solver.get_forward_residual(step=step))
                        if forward_resid != 0.0:
                            if self.comm.rank == 0:
                                print(
                                    f"f2f scenario {scenario.name}, forward resid = {forward_resid}",
                                    flush=True,
                                )
                            forward_tol = solver.forward_tolerance
                            if forward_resid > forward_tol:
                                all_converged = False
                                break

                    if all_converged:
                        exit_early = True
                        if exit_early and self.comm.rank == 0:
                            print(
                                f"F2F Steady Forward analysis of scenario {scenario.name} exited early"
                            )
                            print(
                                f"\tat step {step} with tolerance {forward_resid} < {forward_tol}",
                                flush=True,
                            )
                if exit_early:
                    break

        return fail

    def _solve_steady_adjoint(self, scenario):
        """
        Solve the aeroelastic adjoint analysis using the linear block Gauss-Seidel algorithm.
        Aitken under-relaxation for stabilty.

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

        # Initialize the adjoint variables
        self._initialize_adjoint_variables(scenario, self.model.bodies)

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

        assert not scenario.steady
        fail = 0

        if not steps:
            if not self.fakemodel:
                steps = scenario.steps
            else:
                if self.comm.Get_rank() == 0:
                    print(
                        "No number of steps given for the coupled problem. Using default (1000)"
                    )
                steps = 1000

        for time_index in range(1, steps + 1):
            # Transfer displacements and temperatures
            for body in self.model.bodies:
                body.transfer_disps(scenario, time_index)
                body.transfer_temps(scenario, time_index)

            # Take a step in the flow solver
            fail = self.solvers.flow.iterate(scenario, self.model.bodies, time_index)

            fail = self.comm.allreduce(fail)
            if fail != 0:
                if self.comm.Get_rank() == 0:
                    print("Flow solver returned fail flag")
                return fail

            # Transfer the loads and heat flux
            for body in self.model.bodies:
                body.transfer_loads(scenario, time_index)
                body.transfer_heat_flux(scenario, time_index)

                if self._debug:
                    struct_loads = body.get_struct_loads(
                        scenario, time_index=time_index
                    )
                    aero_loads = body.get_aero_loads(scenario, time_index=time_index)
                    print(f"========================================")
                    print(f"Inside nlbgs driver, step: {time_index}")
                    if struct_loads is not None:
                        print(f"norm of real struct_loads: {real_norm(struct_loads)}")
                        print(
                            f"norm of imaginary struct_loads: {imag_norm(struct_loads)}"
                        )
                    if aero_loads is not None:
                        print(f"norm of real aero_loads: {real_norm(aero_loads)}")
                        print(f"norm of imaginary aero_loads: {imag_norm(aero_loads)}")
                    print(f"========================================\n", flush=True)

            # Take a step in the FEM model
            fail = self.solvers.structural.iterate(
                scenario, self.model.bodies, time_index
            )

            fail = self.comm.allreduce(fail)
            if fail != 0:
                if self.comm.Get_rank() == 0:
                    print("Structural solver returned fail flag")
                return fail

        return fail

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

        assert not scenario.steady
        fail = 0

        # how many steps to take
        steps = scenario.steps

        # Initialize the adjoint variables
        self._initialize_adjoint_variables(scenario, self.model.bodies)

        # Loop over each time step in the reverse order
        for rstep in range(1, steps + 1):
            step = steps - rstep + 1

            # load current state, affects MELD jacobians in the adjoint matrix (esp. load transfer)
            for body in self.model.bodies:
                body.transfer_disps(scenario, time_index=step)
                body.transfer_temps(scenario, time_index=step)

            self.solvers.flow.set_states(scenario, self.model.bodies, step)
            # Due to the staggering, we linearize the transfer about t_s^(n-1)
            self.solvers.structural.set_states(scenario, self.model.bodies, step - 1)

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
                body.transfer_loads_adjoint(scenario)
                body.transfer_heat_flux_adjoint(scenario)

            fail = self.solvers.flow.iterate_adjoint(scenario, self.model.bodies, step)

            fail = self.comm.allreduce(fail)
            if fail != 0:
                if self.comm.Get_rank() == 0:
                    print("Flow solver returned fail flag")
                return fail

            for body in self.model.bodies:
                body.transfer_disps_adjoint(scenario)
                body.transfer_temps_adjoint(scenario)

            # extract and accumulate coordinate derivative every step
            self._extract_coordinate_derivatives(scenario, self.model.bodies, step)

        # end of solve loop

        # evaluate the initial conditions
        fail = self.solvers.flow.iterate_adjoint(scenario, self.model.bodies, step=0)
        fail = self.comm.allreduce(fail)
        if fail != 0:
            if self.comm.Get_rank() == 0:
                print("Flow solver returned fail flag")
            return fail

        fail = self.solvers.structural.iterate_adjoint(
            scenario, self.model.bodies, step=0
        )
        fail = self.comm.allreduce(fail)
        if fail != 0:
            if self.comm.Get_rank() == 0:
                print("Structural solver returned fail flag")
            return fail

        # extract coordinate derivative term from initial condition
        self._extract_coordinate_derivatives(scenario, self.model.bodies, step=0)

        return 0
