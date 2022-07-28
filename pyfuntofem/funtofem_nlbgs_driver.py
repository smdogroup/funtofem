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
from mpi4py import MPI
from funtofem import TransferScheme
from .funtofem_driver import FUNtoFEMDriver

try:
    from .hermes_transfer import HermesTransfer
except:
    pass


class FUNtoFEMnlbgs(FUNtoFEMDriver):
    def __init__(
        self,
        solvers,
        comm,
        struct_comm,
        struct_root,
        aero_comm,
        aero_root,
        transfer_options=None,
        model=None,
        theta_init=0.125,
        theta_min=0.01,
        theta_max=1.0,
    ):
        """
        The FUNtoFEM driver for the Nonlinear Block Gauss-Seidel
        solvers for steady and unsteady coupled adjoint.

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

        super(FUNtoFEMnlbgs, self).__init__(
            solvers,
            comm,
            struct_comm,
            struct_root,
            aero_comm,
            aero_root,
            transfer_options=transfer_options,
            model=model,
        )

        return

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
            # Transfer displacements and temperatures
            for body in self.model.bodies:
                body.transfer_disps(scenario)
                body.transfer_temps(scenario)

            # Take a step in the flow solver
            fail = self.solvers["flow"].iterate(scenario, self.model.bodies, step)

            fail = self.comm.allreduce(fail)
            if fail != 0:
                if self.comm.Get_rank() == 0:
                    print("Flow solver returned fail flag")
                return fail

            # Transfer the loads and heat flux
            for body in self.model.bodies:
                body.transfer_loads(scenario)
                body.transfer_heat_flux(scenario)

            # Take a step in the FEM model
            fail = self.solvers["structural"].iterate(scenario, self.model.bodies, step)

            fail = self.comm.allreduce(fail)
            if fail != 0:
                if self.comm.Get_rank() == 0:
                    print("Structural solver returned fail flag")
                return fail

            # Under-relaxation for solver stability
            for body in self.model.bodies:
                body.aitken_relax(scenario)

        return fail

    def _solve_steady_adjoint(self, scenario):
        """
        Solve the aeroelastic adjoint analysis using the linear block Gauss-Seidel algorithm.
        Aitken under-relaxation for stabilty.

        Solve for the load-transfer adjoint
        (1) dL/dfs^{T} psi_L + dS/dfs^{T} * psi_S = 0

        Solve for the force-integration adjoint
        (2) dF/dfa^{T} psi_F + dL/dfa^{T} * psi_L = 0

        Contribute to the structural adjoint right-hand-side:
        (3) adjS_rhs -= dL/dus^{T} * psi_L

        (4) Aerodynamic adjoint takes in psi_F and computes the surface mesh sensitivity and contributes it
        to the term adjD_rhs. Note that the contribution is to the right-hand-side and so may be negative,
        depending on the conventions used in the aerodynamic adjoint.

        Solve for the displacement-transfer adjoint
        (5) psi_D = adjD_rhs

        For FUN3D adjD_rhs is computed: adjD_rhs = - dG/dua^{T} * psi_G

        Contribute to the structural adjoint right-hand-side:
        (6) adjS_rhs -= dD/dus^{T} * psi_D

        Solve for the structures adjoint
        dS/dus^{T} * psi_S = adjS_rhs - df/dus

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        """

        fail = 0
        self.aitken_init = True

        # how many steps to take for the nonlinear block Gauss Seidel
        steps = scenario.steps

        time_index = 0

        # Load the current state
        for body in self.model.bodies:
            body.transfer_disps(scenario, time_index)
            body.transfer_loads(scenario, time_index)

        # Initialize the adjoint variables
        self._initialize_adjoint_variables(scenario, self.model.bodies)

        # loop over the adjoint NLBGS solver
        for step in range(1, steps + 1):
            # Get force and heat flux terms for the flow solver
            for body in self.model.bodies:
                body.transfer_loads_adjoint(scenario, time_index)
                body.transfer_heat_flux_adjoint(scenario, time_index)

            # Iterate over the aerodynamic adjoint
            fail = self.solvers["flow"].iterate_adjoint(
                scenario, self.model.bodies, step
            )

            fail = self.comm.allreduce(fail)
            if fail != 0:
                if self.comm.Get_rank() == 0:
                    print("Flow solver returned fail flag")
                return fail

            # Get the structural adjoint rhs
            for body in self.model.bodies:
                body.transfer_disps_adjoint(scenario, time_index)
                body.transfer_temps_adjoint(scenario, time_index)

            # take a step in the structural adjoint
            fail = self.solvers["structural"].iterate_adjoint(
                scenario, self.model.bodies, step
            )

            fail = self.comm.allreduce(fail)
            if fail != 0:
                if self.comm.Get_rank() == 0:
                    print("Structural solver returned fail flag")
                return fail

            for body in self.model.bodies:
                body.aitken_adjoint_relax(scenario)

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

        for time_index in range(1, steps + 1):
            # Transfer displacements and temperatures
            for body in self.model.bodies:
                body.transfer_disps(scenario, time_index)
                body.transfer_temps(scenario, time_index)

            # Take a step in the flow solver
            fail = self.solvers["flow"].iterate(scenario, self.model.bodies, time_index)

            fail = self.comm.allreduce(fail)
            if fail != 0:
                if self.comm.Get_rank() == 0:
                    print("Flow solver returned fail flag")
                return fail

            # Transfer the loads and heat flux
            for body in self.model.bodies:
                body.transfer_loads(scenario, time_index)
                body.transfer_heat_flux(scenario, time_index)

            # Take a step in the FEM model
            fail = self.solvers["structural"].iterate(
                scenario, self.model.bodies, time_index
            )

            fail = self.comm.allreduce(fail)
            if fail != 0:
                if self.comm.Get_rank() == 0:
                    print("Structural solver returned fail flag")
                return fail

        return fail

        #     # Transfer structural displacements and temperatures to aerodynamic surface
        #     for body in self.model.bodies:

        #         if body.transfer is not None:
        #             body.aero_disps = np.zeros(
        #                 body.aero_nnodes * 3, dtype=TransferScheme.dtype
        #             )
        #             body.transfer.transferDisps(body.struct_disps, body.aero_disps)

        #         if body.thermal_transfer is not None:
        #             body.aero_temps = np.zeros(
        #                 body.aero_nnodes, dtype=TransferScheme.dtype
        #             )
        #             body.thermal_transfer.transferTemp(
        #                 body.struct_temps, body.aero_temps
        #             )

        #         if "rigid" in body.motion_type and "deform" in body.motion_type:
        #             rotation = np.zeros(9, dtype=TransferScheme.dtype)
        #             translation = np.zeros(3, dtype=TransferScheme.dtype)
        #             u = np.zeros(body.aero_nnodes * 3, dtype=TransferScheme.dtype)
        #             body.rigid_transform = np.zeros((4, 4), dtype=TransferScheme.dtype)

        #             body.transfer.transformEquivRigidMotion(
        #                 body.aero_disps, rotation, translation, u
        #             )

        #             body.rigid_transform[:3, :3] = rotation.reshape(
        #                 (
        #                     3,
        #                     3,
        #                 ),
        #                 order="F",
        #             )
        #             body.rigid_transform[:3, 3] = translation
        #             body.rigid_transform[-1, -1] = 1.0

        #             body.aero_disps = u.copy()

        #         elif "rigid" in body.motion_type:
        #             transform = self.solvers["structural"].get_rigid_transform(body)

        #     # Take a time step for the flow solver
        #     fail = self.solvers["flow"].iterate(scenario, self.model.bodies, step)

        #     fail = self.comm.allreduce(fail)
        #     if fail != 0:
        #         if self.comm.Get_rank() == 0:
        #             print("Flow solver returned fail flag")
        #         return fail

        #     # Transfer loads and heat flux from fluid and get loads and temps on structure
        #     for body in self.model.bodies:

        #         if body.transfer is not None:
        #             body.struct_loads = np.zeros(
        #                 body.struct_nnodes * body.xfer_ndof, dtype=TransferScheme.dtype
        #             )
        #             body.transfer.transferLoads(body.aero_loads, body.struct_loads)

        #         if body.thermal_transfer is not None:
        #             body.struct_heat_flux = np.zeros(
        #                 body.struct_nnodes, dtype=TransferScheme.dtype
        #             )
        #             heat_flux_magnitude = body.aero_heat_flux[3::4].copy(order="C")
        #             body.thermal_transfer.transferFlux(
        #                 heat_flux_magnitude, body.struct_heat_flux
        #             )

        #     # Take a step in the FEM model
        #     fail = self.solvers["structural"].iterate(scenario, self.model.bodies, step)

        #     fail = self.comm.allreduce(fail)
        #     if fail != 0:
        #         if self.comm.Get_rank() == 0:
        #             print("Structural solver returned fail flag")
        #         return fail

        # # end solve loop

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

        fail = 0

        # how many steps to take
        steps = scenario.steps

        # Initialize the adjoint variables
        nfunctions = scenario.count_adjoint_functions()
        self._initialize_adjoint_variables(scenario, self.model.bodies)

        # Loop over each time step in the reverse order
        for rstep in range(1, steps + 1):
            step = steps - rstep + 1

            self.solvers["flow"].set_states(scenario, self.model.bodies, step)
            # Due to the staggering, we linearize the transfer about t_s^(n-1)
            self.solvers["structural"].set_states(scenario, self.model.bodies, step - 1)

            for body in self.model.bodies:
                if body.transfer is not None:
                    body.aero_disps = np.zeros(
                        body.aero_nnodes * 3, dtype=TransferScheme.dtype
                    )
                    body.transfer.transferDisps(body.struct_disps, body.aero_disps)

                    struct_loads = np.zeros(
                        body.struct_nnodes * body.xfer_ndof, dtype=TransferScheme.dtype
                    )
                    body.transfer.transferLoads(body.aero_loads, struct_loads)

                if "rigid" in body.motion_type and "deform" in body.motion_type:
                    rotation = np.zeros(9, dtype=TransferScheme.dtype)
                    translation = np.zeros(3, dtype=TransferScheme.dtype)
                    u = np.zeros(body.aero_nnodes * 3, dtype=TransferScheme.dtype)

                    body.rigid_transform = np.zeros((4, 4), dtype=TransferScheme.dtype)

                    body.transfer.transformEquivRigidMotion(
                        body.aero_disps, rotation, translation, u
                    )

                    body.rigid_transform[:3, :3] = rotation.reshape((3, 3), order="F")
                    body.rigid_transform[:3, 3] = translation
                    body.rigid_transform[-1, -1] = 1.0

                    body.global_aero_disps = body.aero_disps[:]
                    body.aero_disps = u.copy()

            # take a step in the structural adjoint
            fail = self.solvers["structural"].iterate_adjoint(
                scenario, self.model.bodies, step
            )

            fail = self.comm.allreduce(fail)
            if fail != 0:
                if self.comm.Get_rank() == 0:
                    print("Structural solver returned fail flag")
                return fail

            # Get load and heat flux terms for the flow solver
            for body in self.model.bodies:
                for func in range(nfunctions):
                    if body.transfer is not None:
                        # Transform load transfer adjoint variables using transpose Jacobian from
                        # funtofem: dLdfA^T * psi_L
                        psi_L_r = np.zeros(
                            body.aero_nnodes * 3, dtype=TransferScheme.dtype
                        )
                        body.transfer.applydDduS(
                            body.psi_S[:, func].copy(order="C"), psi_L_r
                        )
                        body.dLdfa[:, func] = psi_L_r

                    if body.thermal_transfer is not None:
                        # Transform heat flux transfer adjoint variables using transpose Jacobian from
                        # funtofem: dQdftA^T * psi_Q = dTdts * psi_Q
                        psi_Q_r = np.zeros(body.aero_nnodes, dtype=TransferScheme.dtype)
                        body.thermal_transfer.applydQdqATrans(
                            body.psi_T_S[:, func].copy(order="C"), psi_Q_r
                        )
                        body.dQdfta[:, func] = psi_Q_r

            fail = self.solvers["flow"].iterate_adjoint(
                scenario, self.model.bodies, step
            )

            fail = self.comm.allreduce(fail)
            if fail != 0:
                if self.comm.Get_rank() == 0:
                    print("Flow solver returned fail flag")
                return fail

            # From the flow grid adjoint, get to the displacement adjoint
            for body in self.model.bodies:
                if body.transfer is not None:
                    for func in range(nfunctions):
                        if body.motion_type == "deform":
                            # displacement adjoint equation
                            body.psi_D[:, func] = -body.dGdua[:, func]
                        elif (
                            "rigid" in body.motion_type and "deform" in body.motion_type
                        ):
                            # solve the elastic deformation adjoint
                            psi_E = np.zeros(
                                body.aero_nnodes * 3, dtype=TransferScheme.dtype
                            )
                            tmt = np.linalg.inv(np.transpose(body.rigid_transform))
                            for node in range(body.aero_nnodes):
                                for i in range(3):
                                    psi_E[3 * node + i] = (
                                        tmt[i, 0] * body.dGdua[3 * node + 0, func]
                                        + tmt[i, 1] * body.dGdua[3 * node + 1, func]
                                        + tmt[i, 2] * body.dGdua[3 * node + 2, func]
                                        + tmt[i, 3]
                                    )

                            # get the product dE/dT^T psi_E
                            dEdTmat = np.zeros((3, 4), dtype=TransferScheme.dtype)

                            for n in range(body.aero_nnodes):
                                for i in range(3):
                                    for j in range(4):
                                        if j < 3:
                                            dEdTmat[i, j] += (
                                                -(
                                                    body.aero_X[3 * n + j]
                                                    + body.aero_disps[3 * n + j]
                                                )
                                                * psi_E[3 * n + i]
                                            )
                                        else:
                                            dEdTmat[i, j] += -psi_E[3 * n + i]

                            dEdT = dEdTmat.flatten(order="F")
                            dEdT = self.comm.allreduce(dEdT)

                            # solve the rigid transform adjoint
                            psi_R = np.zeros(12, dtype=TransferScheme.dtype)
                            dGdT_func = body.dGdT[:, :, func]
                            dGdT = dGdT_func[:3, :4].flatten(order="F")

                            psi_R = -dGdT - dEdT

                            # now solve the displacement adjoint
                            dRduA = np.zeros(
                                3 * body.aero_nnodes, dtype=TransferScheme.dtype
                            )
                            body.transfer.applydRduATrans(psi_R, dRduA)

                            body.psi_D[:, func] = -psi_E - dRduA

                # form the RHS for the structural adjoint equation on the next reverse step
                for func in range(nfunctions):

                    if body.transfer is not None:
                        # calculate dDdu_s^T * psi_D
                        psi_D_product = np.zeros(
                            body.struct_nnodes * body.xfer_ndof,
                            dtype=TransferScheme.dtype,
                        )
                        body.transfer.applydDduSTrans(
                            body.psi_D[:, func].copy(order="C"), psi_D_product
                        )

                        # calculate dLdu_s^T * psi_L
                        psi_L_product = np.zeros(
                            body.struct_nnodes * body.xfer_ndof,
                            dtype=TransferScheme.dtype,
                        )
                        body.transfer.applydLduSTrans(
                            body.psi_L[:, func].copy(order="C"), psi_L_product
                        )
                        body.struct_rhs[:, func] = -psi_D_product - psi_L_product

                    if body.thermal_transfer is not None:
                        # calculate dTdt_s^T * psi_T
                        psi_T_product = np.zeros(
                            body.struct_nnodes * body.therm_xfer_ndof,
                            dtype=TransferScheme.dtype,
                        )
                        body.psi_T = body.dAdta
                        body.thermal_transfer.applydTdtSTrans(
                            body.psi_T[:, func].copy(order="C"), psi_T_product
                        )
                        body.struct_rhs_T[:, func] = -psi_T_product

            # extract and accumulate coordinate derivative every step
            self._extract_coordinate_derivatives(scenario, self.model.bodies, step)

        # end of solve loop

        # evaluate the initial conditions
        fail = self.solvers["flow"].iterate_adjoint(scenario, self.model.bodies, step=0)
        fail = self.comm.allreduce(fail)
        if fail != 0:
            if self.comm.Get_rank() == 0:
                print("Flow solver returned fail flag")
            return fail

        fail = self.solvers["structural"].iterate_adjoint(
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
