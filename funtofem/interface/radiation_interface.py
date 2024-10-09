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

from __future__ import print_function

__all__ = ["RadiationInterface"]

import numpy as np
import os
from funtofem import TransferScheme
from ._solver_interface import SolverInterface


class RadiationInterface(SolverInterface):
    """
    FUNtoFEM interface class for radiative heating. Works for steady analysis only.
    Unlike other interfaces in FUNtoFEM, this interface also implements the physical model;
    no external radiative heating discipline solver is used here.

    """

    def __init__(self, comm, model, conv_hist=False):
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

    def get_functions(self, scenario, bodies):
        """
        Populate the scenario with the aerodynamic function values.

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The scenario
        bodies: :class:`~body.Body`
            list of FUNtoFEM bodies
        """
        pass
        # for function in scenario.functions:
        #    if function.analysis_type=='aerodynamic':
        #        # the [6] index returns the value
        #        if self.comm.Get_rank() == 0:
        #            function.value = interface.design_pull_composite_func(function.id)[6]
        #        function.value = self.comm.bcast(function.value,root=0)

    def iterate(self, scenario, bodies, step):
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
            aero_X = body.get_aero_nodes()
            aero_id = body.get_aero_node_ids()
            aero_nnodes = body.get_num_aero_nodes()

            aero_temps = body.get_aero_temps(scenario, time_index=step)
            heat_rad = self.calc_heat_flux(aero_temps, scenario)

            heat_flux = body.get_aero_heat_flux(scenario, time_index=step)
            heat_flux += heat_rad

        return 0

    def calc_heat_flux(self, temps, scenario=None):
        """
        Implementation of thermal radiation from a surface to a low-temperature environment.
        Calculate the heat flux per area as a function of temperature.

        Paramters
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

        sigma_sb = 5.670374419e-8
        rad_heat = np.zeros_like(temps, dtype=TransferScheme.dtype)

        for indx, temp_i in enumerate(temps):
            rad_heat[indx] = -sigma_sb * emis * F_v * (temp_i**4 - T_v**4)

        return rad_heat
