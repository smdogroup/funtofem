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

from .funtofem_driver import *


class FUNtoFEMnlbgsFSISubiters(FUNtoFEMDriver):
    def __init__(
        self,
        solvers,
        comm,
        struct_comm,
        struct_master,
        aero_comm,
        aero_master,
        transfer_options=None,
        model=None,
        theta_init=0.125,
        theta_min=0.01,
        fsi_subiters=1,
    ):
        """
        The FUNtoFEM driver for the Nonlinear Block Gauss-Seidel solvers for steady and unsteady coupled adjoint.

        Parameters
        ----------
        solvers: dict
           the various disciplinary solvers
        comm: MPI.comm
            MPI communicator
        transfer_options: dict
            options of the load and displacement transfer scheme
        model: :class:`~funtofem_model.FUNtoFEMmodel`
            The model containing the design data
        theta_init: float
            Initial value of theta for the Aitken under-relaxation
        theta_min: float
            Minimum value of theta for the Aitken under-relaxation
        """

        super(FUNtoFEMnlbgsFSISubiters, self).__init__(
            solvers,
            comm,
            struct_comm,
            struct_master,
            aero_comm,
            aero_master,
            transfer_options=transfer_options,
            model=model,
        )

        # Aitken acceleration settings
        self.theta_init = theta_init
        self.theta_min = theta_min
        self.fsi_subiters = fsi_subiters
        self.theta = []

        self.aitken_init = None
        self.aitken_vec = None
        self.up_prev = None

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
        nfunctions = scenario.count_adjoint_functions()
        nfunctions_total = len(scenario.functions)

        for body in bodies:
            body.psi_L = np.zeros(
                (body.struct_nnodes * body.xfer_ndof, nfunctions),
                dtype=TransferScheme.dtype,
            )
            body.psi_S = np.zeros(
                (body.struct_nnodes * body.xfer_ndof, nfunctions),
                dtype=TransferScheme.dtype,
            )
            body.struct_rhs = np.zeros(
                (body.struct_nnodes * body.xfer_ndof, nfunctions),
                dtype=TransferScheme.dtype,
            )

            if body.transfer:
                body.psi_F = np.zeros(
                    (body.aero_nnodes * 3, nfunctions), dtype=TransferScheme.dtype
                )
                body.psi_D = np.zeros(
                    (body.aero_nnodes * 3, nfunctions), dtype=TransferScheme.dtype
                )

            if body.shape:
                body.aero_shape_term = np.zeros(
                    (body.aero_nnodes * 3, nfunctions_total), dtype=TransferScheme.dtype
                )
                body.struct_shape_term = np.zeros(
                    (body.struct_nnodes * body.xfer_ndof, nfunctions_total),
                    dtype=TransferScheme.dtype,
                )

    def _solve_steady_forward(self, scenario, steps=None):
        """
        Solve the aeroelastic forward analysis using the nonlinear block Gauss-Seidel algorithm.
        Aitken under-relaxation for stabilty.

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        steps: int
            Number of iterations if not set by the model
        """

        self.aitken_init = True
        fail = 0

        # Determine if we're using the scenario's number of steps or the argument
        if steps is None:
            if self.model:
                steps = scenario.steps
            else:
                if self.comm.Get_rank() == 0:
                    print(
                        "No number of steps given for the coupled problem. Using default (1000)"
                    )
                steps = 1000

        # Loop over the NLBGS steps
        for step in range(1, steps + 1):

            fail = self.solvers["flow"].iterate(scenario, self.model.bodies, step)
            if fail != 0:
                return fail

            # Transfer the loads
            for body in self.model.bodies:
                body.struct_loads = np.zeros(
                    body.struct_nnodes * body.xfer_ndof, dtype=TransferScheme.dtype
                )
                if body.transfer:
                    body.transfer.transferLoads(body.aero_loads, body.struct_loads)

            # Take a step in the FEM model
            fail = self.solvers["structural"].iterate(scenario, self.model.bodies, step)
            if fail != 0:
                return fail

            # Under-relaxation for solver stability
            self._aitken_relax()

            # Transfer displacements
            for body in self.model.bodies:
                if body.transfer:
                    body.aero_disps = np.zeros(
                        body.aero_nnodes * 3, dtype=TransferScheme.dtype
                    )
                    body.transfer.transferDisps(body.struct_disps, body.aero_disps)

        # end solve loop
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
        fail = 0
        self.aitken_init = True

        # how many steps to take
        steps = scenario.steps

        # Initialize the adjoint variables
        self._initialize_adjoint_variables(scenario, self.model.bodies)

        # Load the current state
        for body in self.model.bodies:
            if body.transfer:
                aero_disps = np.zeros(body.aero_disps.size, dtype=TransferScheme.dtype)
                body.transfer.transferDisps(body.struct_disps, aero_disps)

                struct_loads = np.zeros(
                    body.struct_loads.size, dtype=TransferScheme.dtype
                )
                body.transfer.transferLoads(body.aero_loads, struct_loads)

        # Initialize the adjoint variables
        nfunctions = scenario.count_adjoint_functions()
        self._initialize_adjoint_variables(scenario, self.model.bodies)

        # loop over the adjoint NLBGS solver
        for step in range(1, steps + 1):
            # Get psi_F for the flow solver
            for body in self.model.bodies:
                for func in range(nfunctions):
                    # 'Solve' for load transfer adjoint variables
                    body.psi_L[:, func] = body.psi_S[:, func]

                    # Transform load transfer adjoint variables using transpose Jacobian from
                    # funtofem: psi_F = dLdfA^T * psi_L
                    if body.transfer:
                        psi_F_r = np.zeros(
                            body.aero_nnodes * 3, dtype=TransferScheme.dtype
                        )
                        body.transfer.applydDduS(
                            body.psi_L[:, func].copy(order="C"), psi_F_r
                        )
                        body.psi_F[:, func] = psi_F_r

            fail = self.solvers["flow"].iterate_adjoint(
                scenario, self.model.bodies, step
            )
            if fail != 0:
                return fail

            # Get the structural adjoint rhs
            for body in self.model.bodies:
                for func in range(nfunctions):

                    # calculate dDdu_s^T * psi_D
                    psi_D_product = np.zeros(
                        body.struct_nnodes * body.xfer_ndof, dtype=TransferScheme.dtype
                    )
                    if body.transfer:
                        body.transfer.applydDduSTrans(
                            body.psi_D[:, func].copy(order="C"), psi_D_product
                        )

                    # calculate dLdu_s^T * psi_L
                    psi_L_product = np.zeros(
                        body.struct_nnodes * body.xfer_ndof, dtype=TransferScheme.dtype
                    )
                    if body.transfer:
                        body.transfer.applydLduSTrans(
                            body.psi_L[:, func].copy(order="C"), psi_L_product
                        )

                    body.struct_rhs[:, func] = -psi_D_product - psi_L_product

            # take a step in the structural adjoint
            fail = self.solvers["structural"].iterate_adjoint(
                scenario, self.model.bodies, step
            )
            if fail != 0:
                return fail
            self._aitken_adjoint_relax(scenario)

        # end of solve loop

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
        self.struct_disps_hist = []
        self.aero_loads_hist = []

        for step in range(1, steps + 1):
            for solver in self.solvers:
                fail = self.solvers[solver].step_pre(scenario, self.model.bodies, step)
                if fail != 0:
                    return fail

            for fsi_subiter in range(1, self.fsi_subiters + 1):
                for body in self.model.bodies:

                    # Transfer structural displacements to aerodynamic surface
                    if body.transfer:
                        body.aero_disps = np.zeros(
                            body.aero_nnodes * 3, dtype=TransferScheme.dtype
                        )
                        body.transfer.transferDisps(body.struct_disps, body.aero_disps)

                    if "rigid" in body.motion_type and "deform" in body.motion_type:
                        # TODO parallel rigid motion extraction
                        rotation = np.zeros(9, dtype=TransferScheme.dtype)
                        translation = np.zeros(3, dtype=TransferScheme.dtype)
                        u = np.zeros(body.aero_nnodes * 3, dtype=TransferScheme.dtype)
                        body.rigid_transform = np.zeros(
                            (4, 4), dtype=TransferScheme.dtype
                        )

                        body.transfer.transformEquivRigidMotion(
                            body.aero_disps, rotation, translation, u
                        )

                        body.rigid_transform[:3, :3] = rotation.reshape(
                            (
                                3,
                                3,
                            ),
                            order="F",
                        )
                        body.rigid_transform[:3, 3] = translation
                        body.rigid_transform[-1, -1] = 1.0

                        body.aero_disps = u.copy()

                    elif "rigid" in body.motion_type:
                        transform = self.solvers["structural"].get_rigid_transform(body)

                fail = self.solvers["flow"].step_solver(
                    scenario, self.model.bodies, step, fsi_subiter
                )
                if fail != 0:
                    return fail

                # Transfer loads from fluid and get loads on structure
                for body in self.model.bodies:
                    body.struct_loads = np.zeros(
                        body.struct_nnodes * body.xfer_ndof, dtype=TransferScheme.dtype
                    )
                    if body.transfer:
                        body.transfer.transferLoads(body.aero_loads, body.struct_loads)

                # Take a step in the FEM model
                fail = self.solvers["structural"].step_solver(
                    scenario, self.model.bodies, step, fsi_subiter
                )
                if fail != 0:
                    return fail

            for solver in self.solvers:
                fail = self.solvers[solver].step_post(scenario, self.model.bodies, step)
                if fail != 0:
                    return fail

        # end solve loop
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
        print("Error: FSI subiteration unsteady adjoint not implemented yet")
        return 1

    def _aitken_relax(self):
        if self.aitken_init:
            self.aitken_init = False

            # initialize the 'previous update' to zero
            self.up_prev = []
            self.aitken_vec = []
            self.theta = []

            for ind, body in enumerate(self.model.bodies):
                self.up_prev.append(
                    np.zeros(
                        body.struct_nnodes * body.xfer_ndof, dtype=TransferScheme.dtype
                    )
                )
                self.aitken_vec.append(
                    np.zeros(
                        body.struct_nnodes * body.xfer_ndof, dtype=TransferScheme.dtype
                    )
                )
                self.theta.append(self.theta_init)

        # do the Aitken update
        for ibody, body in enumerate(self.model.bodies):
            up = body.struct_disps - self.aitken_vec[ibody]
            self.theta[ibody] *= (
                1.0
                - (up - self.up_prev[ibody]).dot(up)
                / np.linalg.norm(up - self.up_prev[ibody]) ** 2.0
            )
            self.theta[ibody] = np.max(
                (np.min((self.theta[ibody], 1.0)), self.theta_min)
            )

            # handle the min/max for complex step
            if (
                type(self.theta[ibody]) == np.complex128
                or type(self.theta[ibody]) == complex
            ):
                self.theta[ibody] = self.theta[ibody].real + 0.0j

            self.aitken_vec[ibody] += self.theta[ibody] * up
            self.up_prev[ibody] = up[:]
            body.struct_disps = self.aitken_vec[ibody]

        return

    def _aitken_adjoint_relax(self, scenario):
        nfunctions = scenario.count_adjoint_functions()
        if self.aitken_init:
            self.aitken_init = False

            # initialize the 'previous update' to zero
            self.up_prev = []
            self.aitken_vec = []
            self.theta = []

            for ibody, body in enumerate(self.model.bodies):
                up_prev_body = []
                aitken_vec_body = []
                theta_body = []
                for func in range(nfunctions):
                    up_prev_body.append(
                        np.zeros(
                            body.struct_nnodes * body.xfer_ndof,
                            dtype=TransferScheme.dtype,
                        )
                    )
                    aitken_vec_body.append(
                        np.zeros(
                            body.struct_nnodes * body.xfer_ndof,
                            dtype=TransferScheme.dtype,
                        )
                    )
                    theta_body.append(self.theta_init)
                self.up_prev.append(up_prev_body)
                self.aitken_vec.append(aitken_vec_body)
                self.theta.append(theta_body)

        # do the Aitken update
        for ibody, body in enumerate(self.model.bodies):
            for func in range(nfunctions):
                up = body.psi_S[:, func] - self.aitken_vec[ibody][func]
                self.theta[ibody][func] *= (
                    1.0
                    - (up - self.up_prev[ibody][func]).dot(up)
                    / np.linalg.norm(up - self.up_prev[ibody][func]) ** 2.0
                )
                self.theta[ibody][func] = np.max(
                    (np.min((self.theta[ibody][func], 1.0)), self.theta_min)
                )
                self.aitken_vec[ibody][func] += self.theta[ibody][func] * up
                self.up_prev[ibody][func] = up[:]
                body.psi_S[:, func] = self.aitken_vec[ibody][func][:]

        return self.aitken_vec
