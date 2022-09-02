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

from pyfuntofem.model import *
from pyfuntofem.driver import *
from pyfuntofem.fun3d_interface import *

from tacs_model import wedgeTACS

from mpi4py import MPI
import os


class wedge_adjoint(object):
    """
    -------------------------------------------------------------------------------
    TOGW minimization
    -------------------------------------------------------------------------------
    """

    def __init__(self):
        print("start")

        # cruise conditions
        self.v_inf = 171.5  # freestream velocity [m/s]
        self.rho = 0.01841  # freestream density [kg/m^3]
        self.cruise_q = 12092.5527126  # dynamic pressure [N/m^2]
        self.grav = 9.81  # gravity acc. [m/s^2]
        self.thermal_scale = 0.5 * self.rho * (self.v_inf) ** 3

        # Set up the communicators
        n_tacs_procs = 1

        comm = MPI.COMM_WORLD
        self.comm = comm
        print("set comm")

        world_rank = comm.Get_rank()
        if world_rank < n_tacs_procs:
            color = 55
            key = world_rank
        else:
            color = MPI.UNDEFINED
            key = world_rank
        self.tacs_comm = comm.Split(color, key)
        print("comm misc")
        # Set up the FUNtoFEM model for the TOGW problem
        self._build_model()
        print("built model")
        self.ndv = len(self.model.get_variables())
        print("ndvs: ", self.ndv)
        # instantiate TACS on the master
        solvers = {}
        solvers["flow"] = Fun3dInterface(self.comm, self.model, flow_dt=1.0)
        solvers["structural"] = wedgeTACS(
            self.comm, self.tacs_comm, self.model, n_tacs_procs
        )

        # L&D transfer options
        transfer_options = {
            "analysis_type": "aerothermal",
            "scheme": "meld",
            "thermal_scheme": "meld",
        }

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
        )

        # Set up some variables and constants related to the problem
        self.cruise_lift = None
        self.cruise_drag = None
        self.num_con = 1
        self.mass = None

        self.var_scale = np.ones(self.ndv, dtype=TransferScheme.dtype)
        self.struct_tacs = solvers["structural"].assembler

    def _build_model(self):

        thickness = 0.015
        # Build the model
        model = FUNtoFEMmodel("wedge")
        plate = Body("plate", "aerothermal", group=0, boundary=1)
        plate.add_variable(
            "structural", Variable("thickness", value=thickness, lower=0.01, upper=0.1)
        )
        model.add_body(plate)

        steady = Scenario("steady", group=0, steps=100)
        steady.set_variable(
            "aerodynamic", name="AOA", value=0.0, lower=-15.0, upper=15.0
        )
        temp = Function("temperature", analysis_type="structural")  # temperature
        steady.add_function(temp)

        lift = Function("cl", analysis_type="aerodynamic")
        steady.add_function(lift)

        drag = Function("cd", analysis_type="aerodynamic")
        steady.add_function(drag)

        model.add_scenario(steady)

        self.model = model

    def eval_objcon(self, x):

        fail = 0
        var = x * self.var_scale

        self.model.set_variables(var)

        variables = self.model.get_variables()

        ########################### Simulations ################################
        # Simulate the maneuver condition
        fail = self.driver.solve_forward()  # steps = 1)
        if fail == 1:
            print("simulation failed")
            return 0.0, 0.0, fail

        functions = self.model.get_functions()

        ########################## Objective evaluation ########################

        temp = functions[0].value
        obj = temp

        ####################### Constraints evaluations ########################
        con = np.zeros(self.num_con, dtype=TransferScheme.dtype)

        print("variables:")
        for i in range(self.ndv):
            print(variables[i].name, variables[i].value)

        return obj, con, fail

    def eval_objcon_grad(self, x):

        var = x * self.var_scale
        self.model.set_variables(var)

        variables = self.model.get_variables()

        shape_dv = []
        for var in variables:
            if "shape" in var.name:
                shape_dv.append(var.value)

        thickness_dv = []
        for var in variables:
            if "thickness" in var.name:
                thickness_dv.append(var.value)

        fail = self.driver.solve_adjoint()
        grads = self.model.get_function_gradients()
        funcs = self.model.get_functions()

        if self.comm.Get_rank() == 0:
            for i, func in enumerate(funcs):
                print("Func ", func.name, " ", funcs[i].value)
                for j, var in enumerate(variables):
                    print(
                        "%s %s %s %s %s %.30E"
                        % ("Grad ", func.name, "Var: ", var.name, " ", grads[i][j])
                    )

        cruise_lift_grad = np.array(grads[1][:])  # * 2.0 * self.cruise_q

        cruise_drag_grad = np.array(grads[2][:])  # * 2.0 * self.cruise_q

        temp_grad = np.array(grads[0][:])

        ########################## Objective Gradient ##########################
        g = np.zeros((3, self.ndv), dtype=TransferScheme.dtype)
        g[0, :] = temp_grad
        g[1, :] = cruise_lift_grad
        g[2, :] = cruise_drag_grad
        A = np.zeros((self.num_con, self.ndv), dtype=TransferScheme.dtype)

        print("variables:")
        for i in range(self.ndv):
            print(variables[i].name, variables[i].value)
        return g, A, fail


################################################################################
x = np.array([0.0, 0.015])  # AOA and Thickness, first Aero then struct
dp = wedge_adjoint()
print("created object")
obj, con, fail = dp.eval_objcon(x)
print("objective = ", obj)
g, A, fail = dp.eval_objcon_grad(x)
print("grad = ", g)
print("FINISHED")
