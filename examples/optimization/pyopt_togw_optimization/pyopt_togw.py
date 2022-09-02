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

from pyfuntofem.model import Function, FUNtoFEMmodel
from pyfuntofem.fun3d_interface import Fun3dInterface
from pyfuntofem.massoud_body import MassoudBody
from tacs_model import CRMtacs
from pyOpt import Optimization, SLSQP
from mpi4py import MPI


class crm_togw(object):
    """
    -------------------------------------------------------------------------------
    TOGW minimization
    -------------------------------------------------------------------------------
    """

    def __init__(self):

        # Set up the communicators
        n_tacs_procs = 3

        comm = MPI.COMM_WORLD
        self.comm = comm

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
        solvers["flow"] = Fun3dInterface(self.comm, self.model)
        solvers["structural"] = CRMtacs(
            self.comm, self.tacs_comm, self.model, n_tacs_procs
        )

        # L&D transfer options
        transfer_options = {"scheme": "meld", "isym": 1}

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
        self.W2 = None
        self.maneuver_lift = None
        self.maneuver_ks = None

        self.num_con = 1 + 2 + 1 + 2 * (187 - 3)  # ks + lift + area + smoothness

        # conversion factors
        #                   lbm2kg / (    h2s *   lbf2N )
        tsfc_conversion = 0.453227 / (3600.0 * 4.44822)
        nm2m = 1850.0

        # set up some parameters for the design problem
        self.grav = 9.81  # gravity acc. [m/s^2]
        self.range = 7725.0 * nm2m  # range [m]
        self.tsfc = 0.53 * tsfc_conversion  # tsfc [kg/Ns]
        self.safety_factor = 1.5  # KS failure safety factor
        self.delta = 0.001  # max panel variation [m]
        self.w_secondary = 8000.0 * self.grav  # secondary wing weight
        self.w_fixed = 100900.0 * self.grav  # fixed weight of aircraft
        self.w_reservefuel = 15000.0 * self.grav  # reserve fuel weight

        # cruise conditions
        self.v_inf = 252.129540873  # freestream velocity [m/s]
        self.cruise_q = 12092.5527126  # dynamic pressure [N/m^2]

        # maneuver conditions
        self.maneuver_q = 24128.1008682  # dynamic pressure [N/m^2]

        # scaling of functions
        self.c1 = 1.0e-4  # TOGW scaling
        self.c2 = 10.0  # KS failure scaling
        self.c3 = 1.0e-5  # trim lift scaling
        self.c4 = 1.0  # Area constraint scaling
        self.c5 = 1000.0  # smoothness constraint scaling

        # scaling of design variables
        self.thickness_scale = 0.001
        self.aoa_scale = 1.0
        self.shape_scale = np.loadtxt("shape_scale.dat")

        # vector of design variable scaling (for gradient calculation)
        self.var_scale = np.zeros(self.ndv, dtype=TransferScheme.dtype)
        self.var_scale[0] = self.aoa_scale
        self.var_scale[1] = self.aoa_scale
        self.var_scale[2:54] = self.shape_scale
        self.var_scale[54:] = self.thickness_scale

        # initial coordinates for area calculation
        self.x0 = [22.8000000000, -0.0010000000, 4.42000000000]
        self.x1 = [31.0000000000, 10.8800000000, 4.40000000000]
        self.x2 = [43.0675000000, 27.0000000000, 4.61000000000]
        self.x3 = [45.0000000000, 29.6000000000, 4.63000000000]
        self.x4 = [48.5000000000, 29.6000000000, 4.63000000000]
        self.x5 = [47.1800000000, 27.0000000000, 4.61000000000]
        self.x6 = [39.0000000000, 10.8800000000, 4.40000000000]
        self.x7 = [37.0000000000, -0.0010000000, 2.87000000000]

        self.area0 = self.eval_area(np.zeros(52))

    def _build_model(self):

        # Build the CRM model
        model = FUNtoFEMmodel("crm")

        # Add the CRM wing as a Massoud body
        aero_gp_file = "crm_mesh2_aero.gp"
        struct_gp_file = "crm_mesh2_struct.gp"
        dv_file = "design.1"
        usd_file = "design.usd.1"
        shape_ndv = 72
        wing = MassoudBody(
            "wing",
            group=0,
            boundary=3,
            ndv=72,
            aero_gp_file=aero_gp_file,
            struct_gp_file=struct_gp_file,
            dv_file=dv_file,
            usd_file=usd_file,
            comm=self.comm,
        )

        if TransferScheme.dtype == np.complex128:
            thickness = np.loadtxt("sizing_complex.dat", dtype=TransferScheme.dtype)
        else:
            thickness = np.loadtxt("sizing.dat", dtype=TransferScheme.dtype)

        for i in range(thickness.size):
            wing.add_variable(
                "structural",
                Variable(
                    "thickness " + str(i), value=thickness[i], lower=0.001, upper=0.1
                ),
            )

        is_active = range(3, 10)
        is_active.extend([11, 12])
        is_active.extend(range(14, 22))
        is_active.extend(range(25, 28))
        is_active.extend(range(36, 48))
        is_active.extend(range(52, 72))

        for i in range(shape_ndv):
            active = True if i in is_active else False
            wing.set_variable(
                "shape", index=[i], value=0.0, lower=-0.1, upper=0.1, active=active
            )

        model.add_body(wing)

        # cruise scenario
        cruise = Scenario("cruise", group=0, steps=3)
        cruise.set_variable(
            "aerodynamic", name="AOA", value=3.0, lower=-15.0, upper=15.0, coupled=False
        )

        drag = Function("cd", analysis_type="aerodynamic")
        cruise.add_function(drag)

        lift = Function("cl", analysis_type="aerodynamic")
        cruise.add_function(lift)

        mass = Function("mass", analysis_type="structural", adjoint=False)
        cruise.add_function(mass)

        model.add_scenario(cruise)

        # maneuver scenario
        maneuver = Scenario("maneuver", group=0, steps=3)

        maneuver.set_variable(
            "aerodynamic", name="AOA", value=4.0, lower=-15.0, upper=15.0, coupled=False
        )

        ks = Function("ksfailure", analysis_type="structural")
        maneuver.add_function(ks)

        lift = Function("cl", analysis_type="aerodynamic")
        maneuver.add_function(lift)

        model.add_scenario(maneuver)

        self.model = model

    def eval_objcon(self, x):

        fail = 0

        self.model.set_variables(x * self.var_scale)

        variables = self.model.get_variables()

        shape_dv = []
        for var in variables:
            if "shape" in var.name:
                shape_dv.append(var.value)

        thickness_dv = []
        for var in variables:
            if "thickness" in var.name:
                thickness_dv.append(var.value)

        ########################### Simulations ################################
        # Simulate the maneuver condition
        fail = self.driver.solve_forward()

        if fail != 0:
            print("simulation failed")
            return 0.0, 0.0, fail

        functions = self.model.get_functions()

        self.cruise_drag = functions[0].value * 2.0 * self.cruise_q
        self.cruise_lift = functions[1].value * 2.0 * self.cruise_q
        self.maneuver_ks = functions[3].value
        self.maneuver_lift = functions[4].value * 2.0 * self.maneuver_q

        ########################## Objective evaluation ########################

        # Get the wing weight from TACS
        mass = functions[2].value
        self.w_wing = mass * self.grav

        # total weight
        W2 = 2.0 * self.w_wing + self.w_fixed + self.w_reservefuel + self.w_secondary
        self.W2 = W2

        # form the final objective (TOGW) [W2 + fuel]
        TOGW = W2 * np.exp(
            self.range * self.tsfc / self.v_inf * (self.cruise_drag / self.cruise_lift)
        )
        obj = self.c1 * TOGW

        ####################### Constraints evaluations ########################
        con = np.zeros(self.num_con, dtype=TransferScheme.dtype)

        # lift trim constraints (num=2)
        cruise_weight = 0.5 * (TOGW + W2)
        con[0] = self.c3 * (cruise_weight - self.cruise_lift)

        maneuver_weight = 0.5 * (TOGW + W2)
        con[1] = self.c3 * (2.5 * maneuver_weight - self.maneuver_lift)

        # fixed area (num=1)
        area = self.eval_area(shape_dv)
        con[2] = self.c4 * (area - self.area0)

        # KS failure constraint (num=1)
        con[3] = self.c2 * (self.safety_factor * self.maneuver_ks - 1.0)

        # thickness smooth variation (num = 2*(187-3) )
        con[4:] = self.c5 * self.eval_smoothness(thickness_dv)

        return obj, con, fail

    def eval_objcon_grad(self, x, obj, con):

        self.model.set_variables(x * self.var_scale)

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

        funcs = self.model.get_functions()

        if self.comm.Get_rank() == 0:
            for i, func in enumerate(funcs):
                for j, var in enumerate(variables):
                    print("Grad ", func.name, "Var: ", var.name, " ", grads[i][j])

        cruise_drag_grad = np.array(grads[0][:]) * 2.0 * self.cruise_q

        cruise_lift_grad = np.array(grads[1][:]) * 2.0 * self.cruise_q

        mass_grad = np.array(grads[2][:])

        maneuver_ks_grad = np.array(grads[3][:])

        maneuver_lift_grad = np.array(grads[4][:]) * 2.0 * self.maneuver_q

        ########################## Objective Gradient ##########################
        g = np.zeros((1, self.ndv), dtype=TransferScheme.dtype)

        dW2dx = 2.0 * mass_grad * self.grav

        # TOGW gradient
        TOGW_grad = np.zeros(self.ndv, dtype=TransferScheme.dtype)

        expon = np.exp(
            self.range * self.tsfc / self.v_inf * self.cruise_drag / self.cruise_lift
        )

        TOGW_grad += dW2dx * expon

        TOGW_grad += (
            self.W2
            * expon
            * self.range
            * self.tsfc
            / self.v_inf
            * cruise_drag_grad
            / self.cruise_lift
        )

        TOGW_grad += (
            self.W2
            * expon
            * self.range
            * self.tsfc
            / self.v_inf
            * self.cruise_drag
            * -cruise_lift_grad
            / (self.cruise_lift**2.0)
        )

        g[0, :] = self.c1 * TOGW_grad
        g[0, :] *= self.var_scale

        ########################## Constraint Gradients ########################
        A = np.zeros((self.num_con, self.ndv), dtype=TransferScheme.dtype)

        # lift trim constraint gradients
        cruise_weight_grad = 0.5 * (TOGW_grad + dW2dx)
        A[0, :] = self.c3 * (cruise_weight_grad - cruise_lift_grad)
        A[0, :] *= self.var_scale

        maneuver_weight_grad = 0.5 * (TOGW_grad + dW2dx)
        A[1, :] = self.c3 * (2.5 * maneuver_weight_grad - maneuver_lift_grad)
        A[1, :] *= self.var_scale

        # fixed area gradient
        A[2, :] = self.c4 * self.area_grad(shape_dv)
        A[2, :] *= self.var_scale

        # KS failure constraint gradient
        A[3, :] = self.c2 * self.safety_factor * maneuver_ks_grad
        A[3, :] *= self.var_scale

        # smoothness gradients
        A[4:, :] = self.c5 * self.eval_smoothness_grad(thickness_dv)
        A[4:, :] *= self.var_scale

        return g, A, fail

    def eval_smoothness(self, x):

        con = np.zeros(2 * (187 - 3), dtype=TransferScheme.dtype)
        j = 0
        k = 0
        for i in range(187 - 3):
            # skip the end points
            if j == 47 or j == 91 or j == 139:
                j += 1
            con[k] = x[j] - x[j + 1] - self.delta
            con[k + 1] = x[j + 1] - x[j] - self.delta
            k += 2
            j += 1

        return con

    def eval_smoothness_grad(self, x):

        offset = 54
        grad = np.zeros([2 * (187 - 3), self.ndv], dtype=TransferScheme.dtype)

        j = 0
        k = 0
        for i in range(187 - 3):
            # skip the end points
            if j == 47 or j == 91 or j == 139:
                j += 1
            grad[k, j + offset] = 1.0
            grad[k, j + 1 + offset] = -1.0
            grad[k + 1, j + offset] = -1.0
            grad[k + 1, j + 1 + offset] = 1.0
            j += 1
            k += 2
        return grad

    def eval_area(self, dv):
        area = 0.0

        ################################ quad 1 ################################
        ax = self.x0[0]
        ay = self.x0[1]

        bx = self.x1[0] + dv[0]
        by = self.x1[1] + dv[1]

        cx = self.x6[0] + dv[13]
        cy = self.x6[1] + dv[14]

        dx = self.x7[0] + dv[16]
        dy = self.x7[1]

        area += self.quad_area(ax, ay, bx, by, cx, cy, dx, dy)
        area1 = area

        ############################### quad 2 #################################

        ax = self.x1[0] + dv[0]
        ay = self.x1[1] + dv[1]

        bx = self.x2[0] + dv[3]
        by = self.x2[1] + dv[4]

        cx = self.x5[0] + dv[10]
        cy = self.x5[1] + dv[11]

        dx = self.x6[0] + dv[13]
        dy = self.x6[1] + dv[14]

        area += self.quad_area(ax, ay, bx, by, cx, cy, dx, dy)
        area2 = area - area1

        ############################### quad 3 #################################

        ax = self.x2[0] + dv[3]
        ay = self.x2[1] + dv[4]

        bx = self.x3[0] + dv[6]
        by = self.x3[1]

        cx = self.x4[0] + dv[8]
        cy = self.x4[1]

        dx = self.x5[0] + dv[10]
        dy = self.x5[1] + dv[11]

        area += self.quad_area(ax, ay, bx, by, cx, cy, dx, dy)
        area3 = area - area1 - area2

        return area

    def quad_area(self, ax, ay, bx, by, cx, cy, dx, dy):

        # diagonal distance
        e = np.sqrt((cy - ay) ** 2.0 + (cx - ax) ** 2.0)

        # intersection of diagonal and perpendicular line through b
        m = (cy - ay) / (cx - ax)
        x_int = (m * (by - ay) + m**2.0 * ax + bx) / (m**2.0 + 1.0)
        y_int = m * (x_int - ax) + ay

        # height of triangle 1
        h1 = np.sqrt((x_int - bx) ** 2.0 + (y_int - by) ** 2.0)

        # intersection of diagonal and perpendicular line through d
        m = (cy - ay) / (cx - ax)
        x_int = (m * (dy - ay) + m**2.0 * ax + dx) / (m**2.0 + 1.0)
        y_int = m * (x_int - ax) + ay

        # height of triangle 2
        h2 = np.sqrt((x_int - dx) ** 2.0 + (y_int - dy) ** 2.0)

        # total area of the quad
        area = 0.5 * e * (h1 + h2)

        return area

    def area_grad(self, dv):
        area_grad = np.zeros(self.ndv, dtype=TransferScheme.dtype)
        offset = 2

        ################################ quad 1 ################################
        ax = self.x0[0]
        ay = self.x0[1]

        bx = self.x1[0] + dv[0]
        by = self.x1[1] + dv[1]

        cx = self.x6[0] + dv[13]
        cy = self.x6[1] + dv[14]

        dx = self.x7[0] + dv[16]
        dy = self.x7[1]

        area_grad[offset + 0] += self.quad_area_grad(
            ax, ay, bx, by, cx, cy, dx, dy, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0
        )
        area_grad[offset + 1] += self.quad_area_grad(
            ax, ay, bx, by, cx, cy, dx, dy, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0
        )
        area_grad[offset + 13] += self.quad_area_grad(
            ax, ay, bx, by, cx, cy, dx, dy, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0
        )
        area_grad[offset + 14] += self.quad_area_grad(
            ax, ay, bx, by, cx, cy, dx, dy, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0
        )
        area_grad[offset + 16] += self.quad_area_grad(
            ax, ay, bx, by, cx, cy, dx, dy, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0
        )

        ############################### quad 2 #################################

        ax = self.x1[0] + dv[0]
        ay = self.x1[1] + dv[1]

        bx = self.x2[0] + dv[3]
        by = self.x2[1] + dv[4]

        cx = self.x5[0] + dv[10]
        cy = self.x5[1] + dv[11]

        dx = self.x6[0] + dv[13]
        dy = self.x6[1] + dv[14]

        area_grad[offset + 0] += self.quad_area_grad(
            ax, ay, bx, by, cx, cy, dx, dy, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        )
        area_grad[offset + 1] += self.quad_area_grad(
            ax, ay, bx, by, cx, cy, dx, dy, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        )
        area_grad[offset + 3] += self.quad_area_grad(
            ax, ay, bx, by, cx, cy, dx, dy, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0
        )
        area_grad[offset + 4] += self.quad_area_grad(
            ax, ay, bx, by, cx, cy, dx, dy, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0
        )
        area_grad[offset + 10] += self.quad_area_grad(
            ax, ay, bx, by, cx, cy, dx, dy, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0
        )
        area_grad[offset + 11] += self.quad_area_grad(
            ax, ay, bx, by, cx, cy, dx, dy, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0
        )
        area_grad[offset + 13] += self.quad_area_grad(
            ax, ay, bx, by, cx, cy, dx, dy, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0
        )
        area_grad[offset + 14] += self.quad_area_grad(
            ax, ay, bx, by, cx, cy, dx, dy, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0
        )

        ############################### quad 3 #################################

        ax = self.x2[0] + dv[3]
        ay = self.x2[1] + dv[4]

        bx = self.x3[0] + dv[6]
        by = self.x3[1]

        cx = self.x4[0] + dv[8]
        cy = self.x4[1]

        dx = self.x5[0] + dv[10]
        dy = self.x5[1] + dv[11]

        area_grad[offset + 3] += self.quad_area_grad(
            ax, ay, bx, by, cx, cy, dx, dy, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        )
        area_grad[offset + 4] += self.quad_area_grad(
            ax, ay, bx, by, cx, cy, dx, dy, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        )
        area_grad[offset + 6] += self.quad_area_grad(
            ax, ay, bx, by, cx, cy, dx, dy, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0
        )
        area_grad[offset + 8] += self.quad_area_grad(
            ax, ay, bx, by, cx, cy, dx, dy, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0
        )
        area_grad[offset + 10] += self.quad_area_grad(
            ax, ay, bx, by, cx, cy, dx, dy, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0
        )
        area_grad[offset + 11] += self.quad_area_grad(
            ax, ay, bx, by, cx, cy, dx, dy, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0
        )

        return area_grad

    def quad_area_grad(
        self,
        ax,
        ay,
        bx,
        by,
        cx,
        cy,
        dx,
        dy,
        daxdx,
        daydx,
        dbxdx,
        dbydx,
        dcxdx,
        dcydx,
        ddxdx,
        ddydx,
    ):

        # diagonal distance
        e = np.sqrt((cy - ay) ** 2.0 + (cx - ax) ** 2.0)

        # intersection of diagonal and perpendicular line through b
        m = (cy - ay) / (cx - ax)
        x_int = (m * (by - ay) + m**2.0 * ax + bx) / (m**2.0 + 1.0)
        y_int = m * (x_int - ax) + ay

        # height of triangle 1
        h1 = np.sqrt((x_int - bx) ** 2.0 + (y_int - by) ** 2.0)

        # now for the derivatives
        dmdx = (dcydx - daydx) / (cx - ax) + (cy - ay) / (cx - ax) ** 2.0 * -(
            dcxdx - daxdx
        )

        dx_intdx = (
            dmdx * (by - ay)
            + m * (dbydx - daydx)
            + m**2.0 * daxdx
            + 2.0 * m * dmdx * ax
            - dbxdx
        ) / (m**2.0 + 1.0) + (m * (by - ay) + m**2.0 * ax - bx) / (
            (m**2.0 + 1.0) ** 2.0
        ) * -2.0 * m

        dy_intdx = dmdx * (x_int - ax) + m * (dx_intdx - daxdx) + daydx

        dh1dx = (
            0.5
            * ((x_int - bx) ** 2.0 + (y_int - by) ** 2.0) ** -0.5
            * (
                2.0 * (x_int - bx) * (dx_intdx - dbxdx)
                + 2.0 * (y_int - by) * (dy_intdx - dbydx)
            )
        )

        # intersection of diagonal and perpendicular line through d
        m = (cy - ay) / (cx - ax)
        x_int = (m * (dy - ay) + m**2.0 * ax + dx) / (m**2.0 + 1.0)
        y_int = m * (x_int - ax) + ay

        # height of triangle 2
        h2 = np.sqrt((x_int - dx) ** 2.0 + (y_int - dy) ** 2.0)

        # now for the derivatives
        dmdx = (dcydx - daydx) / (cx - ax) + (cy - ay) / (cx - ax) ** 2.0 * -(
            dcxdx - daxdx
        )

        dx_intdx = (
            dmdx * (by - ay)
            + m * (dbydx - daydx)
            + m**2.0 * daxdx
            + 2.0 * m * dmdx * ax
            - dbxdx
        ) / (m**2.0 + 1.0) + (m * (by - ay) + m**2.0 * ax - bx) / (
            (m**2.0 + 1.0) ** 2.0
        ) * -2.0 * m

        dy_intdx = dmdx * (x_int - ax) + m * (dx_intdx - daxdx) + daydx

        dh2dx = (
            0.5
            * ((x_int - dx) ** 2.0 + (y_int - dy) ** 2.0) ** -0.5
            * (
                2.0 * (x_int - dx) * (dx_intdx - ddxdx)
                + 2.0 * (y_int - dy) * (dy_intdx - ddydx)
            )
        )

        dedx = (
            0.5
            * ((cy - ay) ** 2.0 + (cx - ax) ** 2.0) ** -0.5
            * (2.0 * (cy - ay) * (dcydx - daydx) + 2.0 * (cx - ax) * (dcxdx - daxdx))
        )

        dareadx = 0.5 * (h1 + h2) * dedx + 0.5 * e * (dh1dx + dh2dx)

        return dareadx


################################################################################

dp = crm_togw()

design_problem = PyOptOptimization(
    dp.comm, dp.eval_objcon, dp.eval_objcon_grad, number_of_steps=3
)

opt_prob = Optimization("crm_togw", design_problem.eval_obj_con)

opt_prob.addObj("TOGW")
opt_prob.addCon("cruise_lift", type="e")
opt_prob.addCon("maneuver_lift", type="e")
opt_prob.addCon("area", type="e")
opt_prob.addCon("ksfailure", type="i")

for i in range(187 - 3):
    opt_prob.addCon("Smoothness %i a" % i, type="i")
    opt_prob.addCon("Smoothness %i b" % i, type="i")

variables = dp.model.get_variables()

for i, var in enumerate(variables):
    print("i", i)
    opt_prob.addVar(
        var.name,
        type="c",
        value=var.value / dp.var_scale[i],
        lower=var.lower / dp.var_scale[i],
        upper=var.upper / dp.var_scale[i],
    )

opt = SLSQP(pll_type="POA")
opt.setOption("MAXIT", 999)
opt(opt_prob, sens_type=design_problem.eval_obj_con_grad, disp_opts=True)
