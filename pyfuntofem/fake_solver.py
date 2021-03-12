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

import numpy as np
import os
from funtofem           import TransferScheme
from pyfuntofem.solver_interface   import SolverInterface

class FakeSolver(SolverInterface):
    def __init__(self, comm, model):
        self.comm = comm

        #  Instantiate FUN3D
        self.naero = 10
        np.random.seed(0)
        self.b = (np.random.rand(self.naero) - 0.5)
        self.Jac = 0.1*(np.random.rand(self.naero, self.naero) - 0.5)

        # Get the initial aero surface meshes
        self.initialize(model.scenarios[0], model.bodies, first_pass=True)
        self.post(model.scenarios[0], model.bodies, first_pass=True)

        # dynamic pressure
        self.qinf = 1.0
        self.dFdqinf = None

        # heat flux
        self.dHdq = None

        # multiple steady scenarios
        self.force_save = {}
        self.disps_save = {}
        self.heat_flux_save = {}
        self.temps_save = {}

        # unsteady scenarios
        self.force_hist = {}
        self.heat_flux_hist = {}
        for scenario in model.scenarios:
            self.force_hist[scenario.id] = {}
            self.heat_flux_hist[scenario.id] = {}

    def initialize(self, scenario, bodies, first_pass=False):

        if first_pass:
            for ibody, body in enumerate(bodies, 1):
                body.aero_nnodes = self.naero
                body.aero_X = np.zeros(3*body.aero_nnodes, dtype=TransferScheme.dtype)
                body.aero_X[:] += np.linspace(1, self.naero, 3*body.aero_nnodes)
                body.aero_id = np.zeros(3*body.aero_nnodes,dtype=int)
                body.rigid_transform = np.identity(4, dtype=TransferScheme.dtype)
        return 0

    def initialize_adjoint(self,scenario,bodies):

        if scenario.steady:
            # load the heat flux and temperatures
            if scenario.steady:
                for ibody, body in enumerate(bodies):
                    if body.aero_nnodes > 0:
                        if body.thermal_transfer is not None:
                            body.aero_heat_flux = self.heat_flux_save[scenario.id][ibody]
                            body.aero_temps = self.temps_save[scenario.id][ibody]

        self.dFdqinf = np.zeros(len(scenario.functions))
        self.dHdq = np.zeros(len(scenario.functions))
        return 0

    def set_functions(self,scenario,bodies):
        pass

    def set_variables(self,scenario,bodies):
        pass

    def get_functions(self,scenario,bodies):
        pass

    def get_function_gradients(self,scenario,bodies,offset):
        for func, function in enumerate(scenario.functions):
            # Do the scenario variables first
            for vartype in scenario.variables:
                if vartype == 'aerodynamic':
                    for i, var in enumerate(scenario.variables[vartype]):
                        if var.active:
                            if function.adjoint:
                                if var.id <= 6:
                                    scenario.derivatives[vartype][offset+func][i] = interface.design_pull_global_derivative(function.id,var.id)
                                elif var.name.lower() == 'dynamic pressure':
                                    scenario.derivatives[vartype][offset+func][i] = self.comm.reduce(self.dFdqinf[func])
                                    scenario.derivatives[vartype][offset+func][i] = self.comm.reduce(self.dHdq[func])
                            else:
                                scenario.derivatives[vartype][offset+func][i] = 0.0
                            scenario.derivatives[vartype][offset+func][i] = self.comm.bcast(scenario.derivatives[vartype][offset+func][i],root=0)

            for body in bodies:
                for vartype in body.variables:
                    if vartype == 'rigid_motion':
                        for i, var in enumerate(body.variables[vartype]):
                            if var.active:
                                # rigid motion variables are not active in funtofem path
                                body.derivatives[vartype][offset+func][i] = 0.0

        return scenario, bodies

    def get_coordinate_derivatives(self,scenario,bodies,step):
        pass

    def iterate(self,scenario,bodies,step):
        # Compute the heat flux
        for ibody, body in enumerate(bodies,1):
            body.aero_heat_flux = np.zeros(4*body.aero_nnodes, dtype=TransferScheme.dtype)
            body.aero_heat_flux[3::4] = np.dot(self.Jac, body.aero_temps) + self.b

        if not scenario.steady:
            # save this steps forces for the adjoint
            self.force_hist[scenario.id][step] = {}
            self.heat_flux_hist[scenario.id][step] = {}
            for ibody, body in enumerate(bodies,1):
                if body.transfer is not None:
                    self.force_hist[scenario.id][step][ibody] = body.aero_loads.copy()
                if body.thermal_transfer is not None:
                    self.heat_flux_hist[scenario.id][step][ibody] = body.aero_heat_flux.copy()
        return 0

    def post(self,scenario,bodies,first_pass=False):
        # save the forces for multiple scenarios if steady
        if scenario.steady and not first_pass:
            self.force_save[scenario.id] = {}
            self.disps_save[scenario.id] = {}
            self.heat_flux_save[scenario.id] = {}
            self.temps_save[scenario.id] = {}
            for ibody, body in enumerate(bodies):
                if body.aero_nnodes > 0:
                    if body.transfer is not None:
                        self.force_save[scenario.id][ibody] = body.aero_loads
                        self.disps_save[scenario.id][ibody] = body.aero_disps
                    if body.thermal_transfer is not None:
                        self.heat_flux_save[scenario.id][ibody] = body.aero_heat_flux
                        self.temps_save[scenario.id][ibody] = body.aero_temps

    def set_states(self,scenario,bodies,step):
        for ibody, body in enumerate(bodies,1):
            if body.transfer is not None:
                body.aero_loads = self.force_hist[scenario.id][step][ibody]
            if body.thermal_transfer is not None:
                body.aero_heat_flux = self.heat_flux_hist[scenario.id][step][ibody]

    def iterate_adjoint(self,scenario,bodies,step):
        fail = 0

        nfunctions = scenario.count_adjoint_functions()
        for ibody, body in enumerate(bodies,1):
            if body.aero_nnodes > 0:
                # Solve the heat flux adjoint equation
                psi_Q = -body.dQdfta

        for ibody, body in enumerate(bodies,1):
            if body.aero_nnodes > 0:

                if body.thermal_transfer is not None:
                    for func in range(nfunctions):
                        body.dAdta[:,func] = np.dot(self.Jac.T, psi_Q[3::4,func])

        return fail

    def post_adjoint(self,scenario,bodies):
        pass

    def step_pre(self,scenario,bodies,step):
        return 0

    def step_post(self,scenario,bodies,step):
        # save this steps forces for the adjoint
        self.force_hist[scenario.id][step] = {}
        for ibody, body in enumerate(bodies,1):
            if body.transfer is not None:
                self.force_hist[scenario.id][step][ibody] = body.aero_loads.copy()
            if body.thermal_transfer is not None:
                self.heat_flux_hist[scenario.id][step][ibody] = body.aero_heat_flux.copy()
        return 0
