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

from pyfuntofem.model import *
from pyfuntofem.driver import *
from pyfuntofem.pistontheory_interface import PistonInterface
from pyfuntofem.tacs_interface import TacsSteadyInterface

from structural_model import OneraPlate
from mpi4py import MPI
import os


class AdjointTest(object):
    def __init__(self):
        # cruise conditions
        self.v_inf = 171.5  # freestream velocity [m/s]
        self.rho = 0.01841  # freestream density [kg/m^3]
        self.cruise_q = 12092.5527126  # dynamic pressure [N/m^2]
        self.grav = 9.81  # gravity acc. [m/s^2]
        self.thermal_scale = 0.5 * self.rho * (self.v_inf) ** 3

        # Set the communicator
        comm = MPI.COMM_WORLD
        self.comm = comm

        # Set the tacs_comm. This must be consistent with the number
        # of processors allocated for tacs.
        n_tacs_procs = 1
        world_rank = comm.Get_rank()
        if world_rank < n_tacs_procs:
            color = 55
            key = world_rank
        else:
            color = MPI.UNDEFINED
            key = world_rank
        self.tacs_comm = comm.Split(color, key)

        # Set up the FUNtoFEM model for the TOGW problem
        self._build_model()

        self.ndv = len(self.model.get_variables())

        # instantiate TACS on the master
        solvers = {}
        qinf = 101325.0  # freestream pressure Pa
        M = 1.2  # Mach number
        U_inf = 411  # Freestream velocity m/s
        x0 = np.array([0, 0, 0])
        alpha = 10  # Angle of attack (degrees)
        length_dir = np.array(
            [np.cos(alpha * np.pi / 180), 0, np.sin(alpha * np.pi / 180)]
        )  # Unit vec in length dir
        width_dir = np.array([0, 1, 0])
        L = 1.20  # Length (chord)
        nL = 10  # Num elems in xi dir
        w = 1.20  # Width (span)
        nw = 20  # Num elems in eta dir
        solvers["flow"] = PistonInterface(
            self.comm,
            self.model,
            qinf,
            M,
            U_inf,
            x0,
            length_dir,
            width_dir,
            L,
            w,
            nL,
            nw,
        )
        assembler = None
        if world_rank < n_tacs_procs:
            assembler = OneraPlate(self.tacs_comm)
        solvers["structural"] = TacsSteadyInterface(
            self.comm, self.model, assembler=assembler
        )

        # L&D transfer options
        transfer_options = {"scheme": "meld", "beta": 0.9, "npts": 10, "isym": 1}

        # instantiate the driver
        self.driver = FUNtoFEMnlbgs(
            solvers,
            self.comm,
            self.tacs_comm,
            0,
            self.comm,
            0,
            transfer_options,
            model=self.model,
            theta_init=0.3,
            theta_min=0.01,
        )

        # Set up some variables and constants related to the problem
        self.cruise_lift = None
        self.cruise_drag = None
        self.num_con = 1
        self.mass = None

        self.var_scale = np.ones(self.ndv, dtype=TransferScheme.dtype)
        self.struct_tacs = solvers["structural"].assembler

    def _build_model(self):

        AoA = 10.0

        # Build the model
        model = FUNtoFEMmodel("onera")
        wing = Body("wing", "aeroelastic", group=0, boundary=1)

        avar = Variable("AOA", value=AoA, lower=0.1, upper=11)
        wing.add_variable("aerodynamic", avar)
        model.add_body(wing)

        steady = Scenario("steady", group=0, steps=50)
        steady.set_variable(
            "aerodynamic", name="AOA", value=0.0, lower=-15.0, upper=15.0
        )

        mass = Function("mass", analysis_type="structural")
        steady.add_function(mass)

        # failure = Function('ksfailure', analysis_type='structural')
        failure = Function("cl", analysis_type="aerodynamic")
        steady.add_function(failure)

        model.add_scenario(steady)
        # model.print_summary()

        self.model = model

    def eval_objcon(self, x):
        fail = 0
        var = x * self.var_scale
        self.model.set_variables(var)

        # Simulate the maneuver condition
        fail = self.driver.solve_forward()
        if fail == 1:
            print("simulation failed")
            return 0.0, 0.0, fail

        functions = self.model.get_functions()

        # Set the temperature as the objective and constraint
        obj = functions[0].value
        con = []
        for func in functions[1:]:
            con.append(func.value)

        return obj, con, fail

    def eval_objcon_grad(self, x):
        var = x * self.var_scale
        self.model.set_variables(var)

        fail = self.driver.solve_adjoint()
        funcs = self.model.get_functions()
        print("funcs: ", funcs)
        grads = self.model.get_function_gradients()
        print("grads: ", grads)
        g = grads[0]
        A = []
        for i, func in enumerate(funcs[1:], 1):
            A.append(grads[i])

        return g, A, fail


h = 1e-30
# h = 1e-6
# x = np.array([0.015 +1j*h], dtype=TransferScheme.dtype)
x = np.array([1.5 + 1j * h], dtype=TransferScheme.dtype)
xreal = np.array([1.5], dtype=TransferScheme.dtype)

dp = AdjointTest()

obj1, con1, fail = dp.eval_objcon(xreal)
print("Objective value: ", obj1)
print("\n Constraint value: ", con1)
g, A, fail = dp.eval_objcon_grad(xreal)
print("Adjoint objective gradient ", g)
for ac in A:
    print("Adjoint constraint gradient ", ac)

obj, con, fail = dp.eval_objcon(x)
print("Forward objective gradient ", (obj - obj1) / h)
for c in con:
    print("Perturbed constraint value = ", c)
    print("Forward constraint gradient = ", (c - con1).imag / h)
