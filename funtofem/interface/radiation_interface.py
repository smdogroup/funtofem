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

from __future__ import print_function, annotations

__all__ = ["RadiationInterface"]

import numpy as np
import os
from typing import TYPE_CHECKING
from funtofem import TransferScheme
from ._solver_interface import SolverInterface

if TYPE_CHECKING:
    from ..model.body import Body
    from ..model.scenario import Scenario


class RadiationInterface(SolverInterface):
    """
    FUNtoFEM interface class for radiative heating. Works for steady analysis only.
    Unlike other interfaces in FUNtoFEM, this interface also implements the physical model;
    no external radiative heating discipline solver is used here.

    """

    def __init__(self, comm, model, conv_hist=False, complex_mode=False, debug=False):
        """
        The instantiation of the thermal radiation interface class will populate the model
        with the aerodynamic surface mesh, body.aero_X and body.aero_nnodes.

        Parameters
        ----------
        comm: MPI.comm
            MPI communicator
        model : :class:`FUNtoFEMmodel`
            FUNtoFEM model
        conv_hist : bool
            output convergence history to file
        """
        self.comm = comm
        self.conv_hist = conv_hist
        self.complex_mode = complex_mode
        self.debug = debug

        self.sigma_sb = 5.670374419e-8

        # setup forward and adjoint tolerances
        super().__init__()

        # Store previous iteration's aerodynamic forces and displacements for
        # each body in order to compute RMS error at each iteration
        if self.conv_hist:
            self.temps_prev = {}
            self.heat_prev = {}
            self.conv_hist_file = "conv_hist.dat"

        # Get the initial aerodynamic surface meshes
        self.initialize(model.scenarios[0], model.bodies)

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
        """
        pass

    def initialize(self, scenario, bodies):
        """
        Initialize the thermal radiation interface and solver.

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The scenario that needs to be initialized
        bodies: :class:`~body.Body`
            list of FUNtoFEM bodies to either get new surface meshes from or to
            set the original mesh in

        Returns
        -------
        fail: int
            Returns zero for successful completion of initialization

        """

        # Touch a file to record the RMS error output for convergence study
        if self.conv_hist:
            with open(self.conv_hist_file, "w") as f:
                pass

        # Extract the relevant components' node locations to body object
        for ibody, body in enumerate(bodies, 1):
            aero_X = body.get_aero_nodes()
            aero_id = body.get_aero_node_ids()
            aero_nnodes = body.get_num_aero_nodes()

            # Initialize the state values used for convergence study as well
            if self.conv_hist:
                self.temps_prev[body.id] = np.zeros(
                    3 * aero_nnodes, dtype=TransferScheme.dtype
                )
                self.heat_prev[body.id] = np.zeros(
                    3 * aero_nnodes, dtype=TransferScheme.dtype
                )

        return 0

    def iterate(self, scenario: Scenario, bodies: list[Body], step):
        """
        Forward iteration of thermal radiation.

        Add heat flux contribution from thermal radiation to aero heat flux.

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The scenario
        bodies: :class:`~body.Body`
            list of FUNtoFEM bodies.
        step: int
            the time step number

        """

        # Write step number to file output
        if self.conv_hist:
            with open(self.conv_hist_file, "a") as f:
                f.write("{0:03d} ".format(step))

        for ibody, body in enumerate(bodies, 1):
            aero_nnodes = body.get_num_aero_nodes()

            aero_temps = body.get_aero_temps(scenario, time_index=step)
            heat_flux = body.get_aero_heat_flux(scenario, time_index=step)

            if aero_temps is not None and aero_nnodes > 0:
                heat_rad = self.calc_heat_flux(aero_temps, scenario)

                heat_flux += heat_rad

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

    def iterate_adjoint(self, scenario: Scenario, bodies: list[Body], step):
        """
        Adjoint iteration of thermal radiation.

        Add contribution to aerodynamic temperature adjoint.
        """

        nfuncs = scenario.count_adjoint_functions()
        for ibody, body in enumerate(bodies, 1):
            # Get the adjoint-Jacobian product for the aero temperature
            aero_flux_ajp = body.get_aero_heat_flux_ajp(scenario)
            aero_nnodes = body.get_num_aero_nodes()
            aero_temps = body.get_aero_temps(scenario, time_index=step)

            aero_temps_ajp = body.get_aero_temps_ajp(scenario)

            if aero_temps_ajp is not None and aero_nnodes > 0:
                # Add contribution to aero_temps_ajp from radiation
                # dR/dhR^{T} * psi_R = dA_dhA^{T} * psi_A = - dQ/dhA^{T} * psi_Q = - aero_flux_ajp
                psi_R = aero_flux_ajp

                rad_heat_deriv = self.calc_heat_flux_deriv(aero_temps, scenario)

                dtype = TransferScheme.dtype
                lam = np.zeros((aero_nnodes, nfuncs), dtype=dtype)

                for func in range(nfuncs):
                    lam[:, func] = psi_R[:, func] * rad_heat_deriv[:]

                for ifunc in range(nfuncs):
                    if self.debug and self.comm.rank == 0:
                        print(f"aero_temps_ajp: {aero_temps_ajp[:, func]}")
                        print(f"lam: {lam[:, func]}")
                    aero_temps_ajp[:, ifunc] += lam[:, ifunc]

        return

    def post_adjoint(self, scenario, bodies):
        """
        Any actions that need to be performed after completing the adjoint solve, e.g., evaluating gradients, deallocating memory, etc.
        """
        pass

    def calc_heat_flux(self, temps, scenario=None):
        """
        Implementation of thermal radiation from a surface to a low-temperature environment.
        Calculate the heat flux per area as a function of temperature.

        Parameters
        ---------
        temps:
            Array of temperatures.
        scenario:
            Used to set the model parameters.
        """
        if scenario is None:
            emis = 0.8
            F_v = 1.0
            T_v = 0.0
        else:
            emis = scenario.emis
            F_v = scenario.F_v
            T_v = scenario.T_v

        rad_heat = np.zeros_like(temps, dtype=TransferScheme.dtype)

        for indx, temp_i in enumerate(temps):
            rad_heat[indx] = -self.sigma_sb * emis * F_v * (temp_i**4 - T_v**4)

        return rad_heat

    def calc_heat_flux_deriv(self, temps, scenario=None):
        """
        Calculate the derivative of radiative heat flux residual dR/dtA

        This forms a diagonal matrix, since the heat flux depends only on the temperature
        at its corresponding node.
        """
        if scenario is None:
            emis = 0.8
            F_v = 1.0
        else:
            emis = scenario.emis
            F_v = scenario.F_v

        rad_heat_deriv = np.zeros_like(temps, dtype=TransferScheme.dtype)

        for indx, temp_i in enumerate(temps):
            rad_heat_deriv[indx] = -4 * self.sigma_sb * emis * F_v * temp_i**3

        return rad_heat_deriv
