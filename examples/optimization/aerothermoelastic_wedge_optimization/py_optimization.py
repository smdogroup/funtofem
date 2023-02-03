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
from pyfuntofem.fun3d_interface import *

# from pyfuntofem.massoud_body import *

from tacs_model import wedgeTACS
from pyoptsparse import Optimization, SLSQP
from mpi4py import MPI
import os, sys
import time


class wedge_adjoint(object):
    """
    -------------------------------------------------------------------------------
    TOGW minimization
    -------------------------------------------------------------------------------
    """

    def __init__(self):
        # steady conditions
        self.v_inf = (
            1962.44 / 6.6 * 0.5
        )  # 148.67 Mach 0.5   # freestream velocity [m/s]
        self.rho = 0.01037  # freestream density [kg/m^3]
        self.grav = 9.81  # gravity acc. [m/s^2]
        self.steady_q = 0.5 * self.rho * (self.v_inf) ** 2  # dynamic pressure [N/m^2]
        self.thermal_scale = (
            0.5 * self.rho * (self.v_inf) ** 3
        )  # heat flux * area [J/s]

        self.steps = 100

        self.maximum_mass = 40.0  # mass = 45 at t=1.0
        self.num_tacs_dvs = (
            10  # need for number of tacs dvs to add to model, before tacs created
        )
        # Set up the communicators
        n_tacs_procs = 1

        comm = MPI.COMM_WORLD
        self.comm = comm
        self.tacs_proc = False

        world_rank = comm.Get_rank()
        if world_rank < n_tacs_procs:
            color = 55
            key = world_rank
            self.tacs_proc = True
        else:
            color = MPI.UNDEFINED
            key = world_rank
        self.tacs_comm = comm.Split(color, key)

        # Set up the FUNtoFEM model for the TOGW problem
        self._build_model()

        self.ndv = len(self.model.get_variables())

        # FUN3D adjoint options (none is default)
        flow_adj_options = {"getgrad": True, "outer_loop_krylov": True}

        # instantiate TACS on the master
        solvers = {}
        solvers["flow"] = Fun3dInterface(
            self.comm, self.model, adjoint_options=flow_adj_options
        )
        solvers["structural"] = wedgeTACS(
            self.comm, self.tacs_comm, self.model, n_tacs_procs
        )

        # L&D transfer options
        transfer_options = {
            "analysis_type": "aerothermoelastic",
            "scheme": "meld",
            "thermal_scheme": "meld",
        }
        transfer_options["isym"] = -1
        transfer_options["beta"] = 10.0
        transfer_options["npts"] = 10

        # instantiate the driver
        self.driver = FUNtoFEMnlbgs_aerothermoelastic(
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
        self.temperature = None
        self.mass = None
        self.steady_lift = None
        self.steady_drag = None

        self.num_con = 1

        self.var_scale = np.ones(self.ndv, dtype=TransferScheme.dtype)
        self.assembler = solvers["structural"].assembler

    def _build_model(self):
        thickness = 0.5
        # Build the model
        model = FUNtoFEMmodel("wedge")
        plate = Body("plate", group=0, boundary=1)
        # plate.add_variable('structural',Variable('thickness',value=thickness,lower = 0.001, upper = 1.0))
        for i in range(self.num_tacs_dvs):
            plate.add_variable(
                "structural",
                Variable("thickness " + str(i), value=thickness, lower=1e-5, upper=1e4),
            )

        model.add_body(plate)

        steady = Scenario("steady", group=0, steps=self.steps)

        steady.add_variable(
            "aerodynamic",
            Variable(
                name="dynamic pressure", value=self.steady_q, lower=1.0, upper=150000.0
            ),
        )
        steady.add_variable(
            "aerodynamic",
            Variable(
                name="thermal scale", value=self.thermal_scale, lower=1.0, upper=1e10
            ),
        )

        temp = Function("temperature", analysis_type="structural")  # temperature
        steady.add_function(temp)

        mass = Function("mass", analysis_type="structural", adjoint=False)
        steady.add_function(mass)

        model.add_scenario(steady)

        self.model = model

    def eval_objcon(self, x):
        fail = 0
        if self.tacs_proc:
            # need to include aerodynamic design variables
            # the FUN3D force and heat flux scale factors are constant, so not returned by optimizer
            const_vars = np.array([dp.steady_q, dp.thermal_scale])

            x_des = self.assembler.createDesignVec()
            x_array = x_des.getArray()
            num_tacs_dvs = x_array.shape[0]

            # Use the quadratic bernstein basis. These functions are always positive
            # and perserve A
            A = np.zeros((num_tacs_dvs, 3))
            u = np.linspace(0, 1, num_tacs_dvs)

            A[:, 0] = (1.0 - u) ** 2
            A[:, 1] = 2.0 * u * (1.0 - u)
            A[:, 2] = u**2

            x_array[:] = np.dot(A, x)

            var = np.concatenate((const_vars, x_array))
            print(var)
            self.model.set_variables(var)

        variables = self.model.get_variables()

        ########################### Simulations ################################
        # Simulate the maneuver condition
        fail = self.driver.solve_forward()
        if fail == 1:
            print("simulation failed")
            return 0.0, 0.0, fail

        functions = self.model.get_functions()

        self.temperature = functions[0].value
        self.mass = functions[1].value

        ########################## Objective evaluation ########################
        obj = self.temperature

        ####################### Constraints evaluations ########################
        con = np.zeros(self.num_con, dtype=TransferScheme.dtype)
        con[0] = self.mass - self.maximum_mass
        if self.comm.Get_rank() == 0:
            for i, func in enumerate(functions):
                print("Func ", func.name, " ", functions[i].value)
            print("variables:")
            for i in range(self.ndv):
                print(variables[i].name, variables[i].value)

        return obj, con, fail

    def eval_objcon_grad(self, x, dummy_obj, dummy_con):
        if self.tacs_proc:
            # need to include aerodynamic design variables
            # the FUN3D force and heat flux scale factors are constant, so not returned by optimizer
            const_vars = np.array([dp.steady_q, dp.thermal_scale])

            x_des = self.assembler.createDesignVec()
            x_array = x_des.getArray()
            num_tacs_dvs = x_array.shape[0]

            # Use the quadratic bernstein basis. These functions are always positive
            # and perserve A
            A = np.zeros((num_tacs_dvs, 3))
            u = np.linspace(0, 1, num_tacs_dvs)

            A[:, 0] = (1.0 - u) ** 2
            A[:, 1] = 2.0 * u * (1.0 - u)
            A[:, 2] = u**2

            x_array[:] = np.dot(A, x)

            var = np.concatenate((const_vars, x_array))
            print(var)
            self.model.set_variables(var)

        #        var = x*self.var_scale
        #        self.model.set_variables(var)

        variables = self.model.get_variables()

        #        shape_dv = []
        #        for var in variables:
        #            if 'shape' in var.name:
        #                shape_dv.append(var.value)

        #        thickness_dv = []
        #        for var in variables:
        #            if 'thickness' in var.name:
        #                thickness_dv.append(var.value)

        fail = self.driver.solve_adjoint()
        grads = self.model.get_function_gradients()
        funcs = self.model.get_functions()

        if self.comm.Get_rank() == 0:
            for i, func in enumerate(funcs):
                print("Func ", func.name, " ", funcs[i].value)
                for j, var in enumerate(variables):
                    print("Grad ", func.name, "Var: ", var.name, " ", grads[i][j])

        temp_grad = np.array(grads[0][:])
        mass_grad = np.array(grads[1][:])

        ########################## Objective Gradient ##########################
        g = np.zeros((2, self.ndv - 2), dtype=TransferScheme.dtype)
        # g[0,:] = temp_grad
        g[0, :] = temp_grad[2:]  # exclude q_inf and therm_scale
        g[1, :] = mass_grad[2:]  # exclude q_inf and therm_scale
        ########################## Constraint Gradients ########################
        A = np.zeros((self.num_con, self.ndv - 2), dtype=TransferScheme.dtype)

        # mass constraint gradient
        A[0, :] = mass_grad[2:]
        # A[0,:] *= self.var_scale
        if self.comm.Get_rank() == 0:
            print("variables:")
            for i in range(self.ndv):
                print(variables[i].name, variables[i].value)
            print("g")
            print(g)
            print("A")
            print(A)

        return g, A, fail


################################################################################
if __name__ == "__main__":
    dp = wedge_adjoint()
    t0 = 0.5
    print("thickness = ", t0)
    # x = np.array([dp.steady_q,dp.thermal_scale,1.0])

    def unscale(x):
        y = x
        return y

    design_problem = PyOptOptimization(
        dp.comm,
        dp.eval_objcon,
        dp.eval_objcon_grad,
        number_of_steps=1,
        unscale_design_variables=unscale,
    )

    opt_prob = Optimization("Aerothermoelasticity", design_problem.eval_obj_con)
    opt_prob.addObj("Average Temperature")

    opt_prob.addVar("x1", type="c", value=t0, lower=0.1, upper=1.0)
    opt_prob.addVar("x2", type="c", value=t0, lower=0.1, upper=1.0)
    opt_prob.addVar("x3", type="c", value=t0, lower=0.1, upper=1.0)

    opt_prob.addCon("maximum mass", type="i", lower=-np.inf, upper=np.inf, equal=0.0)

    if "verify" in sys.argv:
        x0 = np.array([t], TransferScheme.dtype)
        f0, c0, fail = dp.eval_obj(x0)
        g, A, fail = dp.eval_obj_grad(x0, f0, c0)

        for dh in [1e-6, 5e-7, 1e-7, 5e-8, 1e-8]:
            x = x0 + dh
            f1, c1, fail = dp.eval_obj(x)
            fd = (c1[0] - c0[0]) / dh
            rel_err = (fd - A[0, 0]) / fd
            print("Finite-difference interval: %25.15e" % (dh))
            print("Finite-difference value:    %25.15e" % (fd))
            print("Gradient value:             %25.15e" % (A[0, 0]))
            print("Relative error:             %25.15e" % (rel_err))
    else:
        opt = SLSQP(pll_type="POA")
        opt.setOption("ACC", 1e-10)  # Set optimization tolerance to 1e-5 (default 1e-6)
        opt.setOption("MAXIT", 999)
        opt(opt_prob, sens_type=design_problem.eval_obj_con_grad, disp_opts=True)
        print("FINISHED")
