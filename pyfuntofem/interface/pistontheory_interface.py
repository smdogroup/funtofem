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

__all__ = ["PistonTheoryGrid", "PistonTheoryFlow", "PistonInterface"]

import numpy as np, sys
from dataclasses import dataclass

from funtofem import TransferScheme
from ._solver_interface import SolverInterface


@dataclass
class PistonTheoryGrid:
    """
    Piston theory grid class specifies the dimensions of the aerodynamic grid used in piston theory
    """

    origin: np.ndarray  # [x0, y0, z0] origin of the x, y, z grid
    length_dir: np.ndarray  # [x, y, z] direction
    width_dir: np.ndarray  # [x, y, z] direction
    length: float  # length x width size of the rect grid
    width: float
    n_length: int  # num length x num width elements of the rectangular grid
    n_width: int


@dataclass
class PistonTheoryFlow:
    """
    Piston theory flow settings such as mach number, qinf, V or Uinf
    """

    qinf: float  # qinf = 0.5 * rho_inf * v_inf^2
    mach: float  # mach number of flow = V / a
    U_inf: float  # freestream
    flow_dt: float = 1.0  # aerodynamic time step


class PistonInterface(SolverInterface):
    """
    FUNtoFEM interface class for a third order piston theory aerodynamic solver. Works for steady analysis for now.

    This current analysis tool depends on a purely rectangular aerodynamic grid for ease of implementation.
    """

    def __init__(
        self,
        comm,
        model,
        piston_grid: PistonTheoryGrid,
        piston_flow: PistonTheoryFlow,
    ):
        """
        The instantiation of the Piston Theory interface class will populate the model with the aerodynamic surface mesh, body.aero_X and body.aero_nnodes.
        The surface mesh on each processor only holds it's owned nodes.

        Parameters
        ----------
        comm: MPI.comm
            MPI communicator
        model: :class:`FUNtoFEMmodel`
            FUNtoFEM model
        piston_grid : PistonTheoryGrid
            data class for piston theory grid settings
        piston_flow : PistonTheoryFlow
            data class for piston theory flow settings
        """

        self.comm = comm

        self.qinf = piston_flow.qinf  # dynamic pressure
        self.M = piston_flow.mach
        self.U_inf = piston_flow.U_inf
        self.gamma = 1.4
        self.flow_dt = piston_flow.flow_dt

        # retrieve aerodynamic grid values
        self.x0 = (
            piston_grid.origin
            if not (isinstance(piston_grid.origin, list))
            else np.array(piston_grid.origin)
        )
        self.length_dir = (
            piston_grid.length_dir
            if not (isinstance(piston_grid.length_dir, list))
            else np.array(piston_grid.length_dir)
        )
        self.width_dir = (
            piston_grid.width_dir
            if not (isinstance(piston_grid.width_dir, list))
            else np.array(piston_grid.width_dir)
        )
        self.alpha = []  # Actual value declared in initialize
        self.L = piston_grid.length
        self.width = piston_grid.width
        self.nL = piston_grid.n_length  # num elems in xi direction
        self.nw = piston_grid.n_width  # num elems in eta direction

        # Check direction to validate unit vectors (and orthogonality?)
        if not (0.99 <= np.linalg.norm(self.length_dir) <= 1.01):
            print(
                "Length direction not a unit vector \n Calculations may be inaccurate",
                file=sys.stderr,
            )
            exit(1)
        if not (0.99 <= np.linalg.norm(self.width_dir) <= 1.01):
            print(
                "Width direction not a unit vector \n Calculations may be inaccurate",
                file=sys.stderr,
            )
            exit(1)
        if not (-0.01 <= np.dot(self.length_dir, self.width_dir) <= 0.01):
            print(
                "Spanning vectors not orthogonal \n Calculations may be inaccurate",
                file=sys.stderr,
            )
            exit(1)

        self.n = np.cross(
            self.width_dir, self.length_dir
        )  # Setup vector normal to plane

        self.CD_mat = []
        self.nmat = []
        self.aero_nnodes = []
        self.psi_P = None

        self.variables = model.get_variables()
        self.aero_variables = []

        for var in self.variables:
            if var.analysis_type == "aerodynamic":
                self.aero_variables.append(var)

        # heat flux
        self.thermal_scale = 1.0  # = 1/2 * rho_inf * (V_inf)^3

        for ibody, body in enumerate(model.bodies, 1):
            aero_nnodes = (self.nL + 1) * (self.nw + 1)
            self.aero_nnodes = aero_nnodes
            aero_X = np.zeros(3 * aero_nnodes, dtype=TransferScheme.dtype)

            self.alpha = np.arccos(self.length_dir[0]) * 180 / np.pi

            self.CD_mat = np.zeros(
                (aero_nnodes, aero_nnodes)
            )  # Matrix for central difference
            diag_ones = np.ones(aero_nnodes - 1)
            diag_neg = -np.ones(aero_nnodes - 1)
            self.CD_mat += np.diag(diag_ones, 1)
            self.CD_mat += np.diag(diag_neg, -1)
            self.CD_mat[0][0] = -2.0
            self.CD_mat[0][1] = 2.0
            self.CD_mat[-1][-2] = -2.0
            self.CD_mat[-1][-1] = 2.0
            self.CD_mat *= 1.0 / (2.0 * self.L / self.nL)

            self.nmat = np.zeros((3 * aero_nnodes, aero_nnodes))
            self.n = np.array([0, 0, 1])
            for i in range(aero_nnodes):
                self.nmat[3 * i : 3 * i + 3, i] = self.n

            # Setting aero node locations
            struct_length_dir = np.array([1, 0, 0])
            for i in range(self.nL + 1):
                for j in range(self.nw + 1):
                    coord = (
                        self.x0
                        + i * self.L / self.nL * struct_length_dir
                        + j * self.width / self.nw * self.width_dir
                    )
                    aero_X[3 * (self.nw + 1) * i + j * 3] = coord[0]
                    aero_X[3 * (self.nw + 1) * i + j * 3 + 1] = coord[1]
                    aero_X[3 * (self.nw + 1) * i + j * 3 + 2] = coord[2]

            body.initialize_aero_nodes(aero_X)

            # Setting internal aero node locations with AoA
            self.piston_aero_X = np.zeros(3 * aero_nnodes, dtype=TransferScheme.dtype)
            for i in range(self.nL + 1):
                for j in range(self.nw + 1):
                    coord = (
                        self.x0
                        + i * self.L / self.nL * self.length_dir
                        + j * self.width / self.nw * self.width_dir
                    )
                    self.piston_aero_X[3 * (self.nw + 1) * i + j * 3] = coord[0]
                    self.piston_aero_X[3 * (self.nw + 1) * i + j * 3 + 1] = coord[1]
                    self.piston_aero_X[3 * (self.nw + 1) * i + j * 3 + 2] = coord[2]

        class ScenarioData:
            def __init__(self, bodies):
                """
                store unsteady state history for each body
                """
                self.w_hist = {}
                for body in bodies:
                    self.w_hist[body.id] = []

        # store state history for the unsteady adjoint any unsteady scenarios
        self._has_unsteady = any(
            [not (scenario.steady) for scenario in model.scenarios]
        )
        if self._has_unsteady:
            self.scenario_data = {}
            for scenario in model.scenarios:
                self.scenario_data[scenario.id] = ScenarioData(model.bodies)

    def initialize(self, scenario, bodies):
        """
        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The scenario that needs to be initialized
        bodies: :class:`~body.Body`
            list of FUNtoFEM bodies to either get new surface meshes from or to set the original mesh in

        Returns
        -------
        fail: int
            If the grid deformation failed, the intiialization will return 1
        """

        return 0

    def initialize_AoA(self, bodies):
        for ibody, body in enumerate(bodies, 1):
            for i in range(self.nL + 1):
                for j in range(self.nw + 1):
                    coord = (
                        self.x0
                        + i * self.L / self.nL * self.length_dir
                        + j * self.width / self.nw * self.width_dir
                    )
                    self.piston_aero_X[3 * (self.nw + 1) * i + j * 3] = coord[0]
                    self.piston_aero_X[3 * (self.nw + 1) * i + j * 3 + 1] = coord[1]
                    self.piston_aero_X[3 * (self.nw + 1) * i + j * 3 + 2] = coord[2]

        # body.initialize_aero_nodes(aero_X)

    def initialize_adjoint(self, scenario, bodies):
        """
        Changes the directory to ./`scenario.name`/Adjoint, then
        initializes the FUN3D adjoint solver.

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The scenario that needs to be initialized
        bodies: :class:`~body.Body`
            list of FUNtoFEM bodies to either get surface meshes from

        Returns
        -------
        fail: int
            If the grid deformation failed, the intiialization will return 1
        """

        return 0

    def set_functions(self, scenario, bodies):
        """
        Set the function definitions into FUN3D using the design interface.
        Since FUNtoFEM only allows single discipline functions, the FUN3D composite function is the same as the component.

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The scenario. Contains the function list
        bodies: :class:`~body.Body`
            list of FUNtoFEM bodies
        """

        return

    def set_variables(self, scenario, bodies):
        """
        Sets the aerodynamic variables (currently only angle of attack is supported)

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The scenario. Contains the global aerodynamic list
        bodies: :class:`~body.Body`
            list of FUNtoFEM bodies. Bodies contains unused but necessary rigid motion variables
        """

        for var in scenario.variables["aerodynamic"]:
            if var.name == "AOA":
                self.set_AoA(var.value, scenario, bodies)

        return

    def set_AoA(self, alpha, scenario, bodies):
        self.alpha = alpha
        self.length_dir = np.array(
            [np.cos(alpha * np.pi / 180), 0, np.sin(alpha * np.pi / 180)]
        )  # Unit vec in length dir
        self.initialize_AoA(bodies)  # Is there a better way to reset this variable???
        return

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

        for function in scenario.functions:
            if function.analysis_type == "aerodynamic":
                if self.comm.Get_rank() == 0:
                    if function.name == "cl":
                        function.value = self.compute_cl(scenario, bodies)
                function.value = self.comm.bcast(function.value, root=0)

        return

    def compute_cl(self, scenario, bodies):
        for ibody, body in enumerate(bodies, 1):
            aero_loads = body.get_aero_loads(scenario)
            lift = np.sum(aero_loads[2::3])
            cl = lift / (self.qinf * self.L * self.width)
        return cl

    def get_function_gradients(self, scenario, bodies):
        """
        Populates the FUNtoFEM model with derivatives w.r.t. aerodynamic variables

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The scenario. Contains the global aerodynamic list
        bodies: :class:`~body.Body`
            list of FUNtoFEM bodies. Bodies contains unused but necessary rigid motion variables
        """

        for findex, func in enumerate(scenario.functions):
            for vindex, var in enumerate(self.aero_variables):
                if var.name == "AOA":
                    if func.name == "cl":
                        value = self.compute_cl_deriv(scenario, bodies)
                        func.add_gradient_component(var, value[:])
                    elif func.name == "ksfailure":
                        value = self.compute_ks_deriv(scenario, bodies)
                        func.add_gradient_component(var, value[:])

        return

    def compute_dwdt(self, scenario, body, step):
        if scenario.steady or len(w_hist) <= 1:
            dw_dt = np.zeros(self.aero_nnodes)
        else:
            w_hist = self.scenario_data[scenario.id].w_hist[body.id]
            dw_dt = (w_hist[step] - w_hist[step - 1]) / self.flow_dt
        return dw_dt

    def compute_cl_deriv(self, scenario, bodies):
        for ibody, body in enumerate(bodies, 1):
            aero_disps = body.get_aero_disps(scenario)
            # w = body.aero_X[2::3] + self.nmat.T @ aero_disps
            w = self.piston_aero_X[2::3] + self.nmat.T @ aero_disps
            dw_dxi = self.CD_mat @ w

            # compute rate of change of w over time
            dwdt = self.compute_dwdt(scenario, body)
            areas = self.compute_Areas()

            dwdxi_deriv = self.compute_Pressure_deriv(dw_dxi, dw_dt)
            dAeroX_dAlpha = np.zeros(body.aero_nnodes * 3, dtype=TransferScheme.dtype)
            for i in range(body.aero_nnodes):
                r = np.sqrt(
                    self.piston_aero_X[3 * i] ** 2 + self.piston_aero_X[3 * i + 2] ** 2
                )
                dAeroX_dAlpha[3 * i : 3 * i + 3] = np.array(
                    [
                        -r * np.pi / 180 * np.sin(self.alpha * np.pi / 180),
                        0,
                        r * np.pi / 180 * np.cos(self.alpha * np.pi / 180),
                    ]
                )

            dP_dAlpha = (
                -self.nmat
                @ np.diag(areas)
                @ np.diag(dwdxi_deriv)
                @ self.CD_mat
                @ self.nmat.T
                @ dAeroX_dAlpha
            )

            # Computing dCL_dAlpha
            lift_mat = np.zeros((1, self.aero_nnodes * 3))
            lift_mat[:, 2::3] = 1.0
            dCLdfa = 1 / (self.qinf * self.L * self.width) * lift_mat
            dCL_dAlpha = (
                dCLdfa
                @ self.nmat
                @ np.diag(areas)
                @ np.diag(dwdxi_deriv)
                @ self.CD_mat
                @ self.nmat.T
                @ dAeroX_dAlpha
            )

            cl_grad = dCL_dAlpha + self.psi_P.T @ dP_dAlpha

        return cl_grad

    def compute_ks_deriv(self, scenario, bodies):
        for ibody, body in enumerate(bodies, 1):
            aero_disps = body.get_aero_disps(scenario)
            w = self.piston_aero_X[2::3] + self.nmat.T @ aero_disps
            dw_dxi = self.CD_mat @ w
            dw_dt = np.zeros(self.aero_nnodes)  # Set dw/dt = 0  for now (steady)
            areas = self.compute_Areas()

            dwdxi_deriv = self.compute_Pressure_deriv(dw_dxi, dw_dt)
            dAeroX_dAlpha = np.zeros(body.aero_nnodes * 3, dtype=TransferScheme.dtype)
            for i in range(body.aero_nnodes):
                r = np.sqrt(
                    self.piston_aero_X[3 * i] ** 2 + self.piston_aero_X[3 * i + 2] ** 2
                )
                dAeroX_dAlpha[3 * i : 3 * i + 3] = np.array(
                    [
                        -r * np.pi / 180 * np.sin(self.alpha * np.pi / 180),
                        0,
                        r * np.pi / 180 * np.cos(self.alpha * np.pi / 180),
                    ]
                )

            dP_dAlpha = (
                -self.nmat
                @ np.diag(areas)
                @ np.diag(dwdxi_deriv)
                @ self.CD_mat
                @ self.nmat.T
                @ dAeroX_dAlpha
            )

            ks_grad = self.psi_P.T @ dP_dAlpha
        return ks_grad

    def get_coordinate_derivatives(self, scenario, bodies, step):
        """
        NOT APPLICABLE TO PISTON THEORY SOLVER !!!

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The scenario.
        bodies: :class:`~body.Body`
            list of FUNtoFEM bodies.
        step: int
            the time step number
        """

        pass

    def iterate(self, scenario, bodies, step):
        """
        Forward iteration of Piston Theory.
        For the aeroelastic cases, these steps are:

        #. Get the mesh movement - the bodies' surface displacements and rigid rotations.
        #. Step forward in the piston theory flow solver.
        #. Set the aerodynamic forces into the body data types


        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The scenario
        bodies: :class:`~body.Body`
            list of FUNtoFEM bodies.
        step: int
            the time step number
        """

        # Calculate aero_loads from aero_disps
        for ibody, body in enumerate(bodies, 1):
            aero_disps = body.get_aero_disps(scenario)
            aero_loads = body.get_aero_loads(scenario)

            if aero_disps is not None:
                # store the displacements at each step for unsteady case

                # self.compute_forces(aero_disps, aero_loads, self.piston_aero_X, step)

                # Compute w for piston theory: [dx,dy,dz] DOT freestream normal
                w = self.piston_aero_X[2::3] + self.nmat.T @ aero_disps

                # store the history if unsteady
                if not (scenario.steady):
                    self.scenario_data[scenario.id].w_hist[body.id].append(w)

                ####### Compute body.aero_loads using Piston Theory ######
                # First compute dw/dxi
                dw_dxi = self.CD_mat @ w

                # Set dw/dt = 0  for now (steady)
                dw_dt = self.compute_dwdt(scenario, bodies, step)

                # Call function to compute pressure
                press_i = self.compute_Pressure(dw_dxi, dw_dt)

                # Compute forces from pressure
                areas = self.compute_Areas()
                aero_loads[:] = self.nmat @ np.diag(areas) @ press_i

                # Write Loads to File at the last step
                # if step == scenario.steps:
                #     file = open("NodalForces.txt", "w")
                #     np.savetxt(file, aero_loads)
                #     file.close()

        return 0

    def compute_Pressure_adjoint(self, dw_dxi, dw_dt, press_adj):
        d_press_dxi = self.compute_Pressure_deriv(dw_dxi, dw_dt)
        dw_dxi_adjoint = np.diag(d_press_dxi) @ press_adj
        return dw_dxi_adjoint, None

    def compute_Pressure(self, dw_dxi, dw_dt):
        """
        Returns 'pressure' values at each node location using piston theory
        governing equation
        """

        press = (
            2
            * self.qinf
            / self.M
            * (
                (1 / self.U_inf * dw_dt + dw_dxi)
                + (self.gamma + 1) / 4 * self.M * (1 / self.U_inf * dw_dt + dw_dxi) ** 2
                + (self.gamma + 1)
                / 12
                * self.M**2
                * (1 / self.U_inf * dw_dt + dw_dxi) ** 3
            )
        )

        # Simplified First Order Piston Theory
        # press = 2.0 * self.qinf / self.M * ((1.0 / self.U_inf * dw_dt + dw_dxi))

        return press

    def compute_Pressure_deriv(self, dw_dxi, dw_dt):
        """
        Returns partial derivatives 'pressure' values at each node location
        with respect to dw_dxi
        """
        d_press_dwdxi = (
            2
            * self.qinf
            / self.M
            * (
                1
                + (self.gamma + 1)
                / 4
                * self.M
                * 2
                * (1 / self.U_inf * dw_dt + dw_dxi)
                * (1)
                + (self.gamma + 1)
                / 12
                * self.M**2
                * 3
                * (1 / self.U_inf * dw_dt + dw_dxi) ** 2
                * (1)
            )
        )

        # Simplified First Order Piston Theory
        # ones = np.ones(dw_dxi.shape)
        # d_press_dwdxi = 2.0 * self.qinf / self.M * ones

        return d_press_dwdxi

    def compute_Areas(self):
        """
        Computes area corresponding to each node (calculations based on rectangular
        evenly spaced mesh grid)
        """
        area_array = (
            (self.L / self.nL) * (self.width / self.nw) * np.ones(self.aero_nnodes)
        )  # Array of area corresponding to each node
        area_array[0 : self.nw + 1] *= 0.5
        area_array[-1 : -self.nw - 2 : -1] *= 0.5
        area_array[0 :: self.nw + 1] *= 0.5
        area_array[self.nw :: self.nw + 1] *= 0.5

        return area_array

    def post(self, scenario, bodies, first_pass=False):
        """
        Calls FUN3D post to save history files, deallocate memory etc.
        Then moves back to the problem's root directory

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The scenario
        bodies: :class:`~body.Body`
            list of FUNtoFEM bodies.
        first_pass: bool
            Set to true during instantiation
        """
        # self.fun3d_flow.post()
        # os.chdir("../..")

    def set_states(self, scenario, bodies, step):
        """
        Loads the saved aerodynamic forces for the time dependent adjoint

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The scenario
        bodies: :class:`~body.Body`
            list of FUNtoFEM bodies.
        step: int
            the time step number
        """
        return

    def iterate_adjoint(self, scenario, bodies, step):
        """
        Adjoint iteration of Piston Theory.
        For the aeroelastic cases, these steps are:

        #. Get the force adjoint from the body data structures
        #. Step in the piston theory adjoint solvers
        #. Set the piston theory adjoint into the body data structures

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The scenario
        bodies: :class:`~body.Body`
            list of FUNtoFEM bodies.
        step: int
            the forward time step number
        """

        fail = 0
        rstep = scenario.steps - step + 1
        if scenario.steady:
            rstep = step

        nfunctions = scenario.count_adjoint_functions()

        for ibody, body in enumerate(bodies, 1):
            aero_loads_ajp = body.get_aero_loads_ajp(scenario)
            if aero_loads_ajp is not None:
                self.psi_P = -aero_loads_ajp

        for ibody, body in enumerate(bodies, 1):
            # Extract the equivalent of dG/du_a^T psi_G from Piston Theory (dP/du_a^T psi_P)
            aero_disps_ajp = body.get_aero_disps_ajp(scenario)
            aero_nnodes = body.get_num_aero_nodes()
            aero_disps = body.get_aero_disps(scenario)
            aero_loads = body.get_aero_loads(scenario)

            if aero_disps_ajp is not None:
                dPdua = np.zeros(
                    (aero_nnodes * 3, aero_nnodes * 3), dtype=TransferScheme.dtype
                )
                w = self.piston_aero_X[2::3] + self.nmat.T @ aero_disps
                dw_dxi = self.CD_mat @ w
                dw_dt = self.compute_dwdt(scenario, body, step)
                areas = self.compute_Areas()

                dwdxi_deriv = self.compute_Pressure_deriv(dw_dxi, dw_dt)
                dPdua[:] = (
                    self.nmat
                    @ np.diag(areas)
                    @ np.diag(dwdxi_deriv)
                    @ self.CD_mat
                    @ self.nmat.T
                )

                for k, func in enumerate(scenario.functions):
                    aero_disps_ajp[:, k] = -dPdua.T @ self.psi_P[:, k].flatten()

                    if func.name == "cl":
                        aero_disps_ajp[:, k] += self.compute_dCLdua(
                            aero_disps,
                            aero_loads,
                            self.piston_aero_X,
                            aero_nnodes,
                            dw_dt,
                        ).flatten()

        return fail

    def compute_dCLdua(self, aero_disps, aero_loads, aero_X, aero_nnodes, dw_dt):
        w = aero_X[2::3] + self.nmat.T @ aero_disps
        dw_dxi = self.CD_mat @ w
        dw_dt = np.zeros(self.aero_nnodes)  # Set dw/dt = 0  for now (steady)
        areas = self.compute_Areas()
        dwdxi_deriv = self.compute_Pressure_deriv(dw_dxi, dw_dt)
        df_dua = (
            self.nmat
            @ np.diag(areas)
            @ np.diag(dwdxi_deriv)
            @ self.CD_mat
            @ self.nmat.T
        )

        lift_mat = np.zeros((1, aero_nnodes * 3))
        lift_mat[:, 2::3] = 1.0
        dCLdfa = 1 / (self.qinf * self.L * self.width) * lift_mat
        return dCLdfa @ df_dua

    def post_adjoint(self, scenario, bodies):
        """
        Calls post fo the adjoint solver in FUN3D.
        Then moves back to the problem's root directory

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The scenario
        bodies: :class:`~body.Body`
            list of FUNtoFEM bodies.
        """

        # solve the initial condition adjoint
        # self.fun3d_adjoint.post()
        # os.chdir("../..")
        pass
