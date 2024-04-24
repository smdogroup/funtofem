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

__all__ = ["Fun3dAeroelasticTestInterface"]

import numpy as np
from funtofem import TransferScheme
from ..fun3d_interface import Fun3dInterface
from funtofem.interface.test_solver import TestResult


class Fun3dAeroelasticTestInterface(Fun3dInterface):
    """
    FUN3D Thermal interface for unit testing of FUN3D aerothermal code

    FUN3D's FUNtoFEM coupling interface requires no additional configure flags to compile.
    To tell FUN3D that a body's motion should be driven by FUNtoFEM, set *motion_driver(i)='funtofem'*.

    Intended use: run one forward + adjoint analysis with a FUNtoFEM nlbgs driver
    separately also creating a Fun3d14Interface. Then keep those states saved in the body class
    then we can compute the finite difference here about those states.
    """

    def __init__(
        self,
        comm,
        model,
        fun3d_dir=None,
        forward_options=None,
        adjoint_options=None,
        complex_mode=False,
    ):
        """
        The instantiation of the FUN3D Grid interface class will populate the model with the aerodynamic surface
        mesh, body.aero_X and body.aero_nnodes.
        The surface mesh on each processor only holds it's owned nodes. Transfer of that data to other processors
        is handled inside the FORTRAN side of FUN3D's FUNtoFEM interface.

        Parameters
        ----------
        comm: MPI.comm
            MPI communicator
        model: :class:`FUNtoFEMmodel`
            FUNtoFEM model. This instantiatio
        fun3d_dir: path
            path to the Flow directory of the fun3d scenario
        forward_options: dict
            list of options for FUN3D forward analysis
        adjoint_options: dict
            list of options for FUN3D adjoint analysis
        """

        # construct the super class Fun3dInterface
        super(Fun3dAeroelasticTestInterface, self).__init__(
            comm=comm,
            model=model,
            fun3d_dir=fun3d_dir,
            forward_options=forward_options,
            adjoint_options=adjoint_options,
            complex_mode=complex_mode,
        )

        # state variables related to grid deformation
        # for body in self.model.bodies:
        #     # initialize transfer schemes for the body classes so the elastic variables will be there
        #     body.initialize_transfer(
        #         self.comm,
        #         self.comm,
        #         0,
        #         self.comm,
        #         0,
        #         transfer_settings=None,
        #     )

        #     for scenario in self.model.scenarios:
        #         body.initialize_variables(
        #             scenario
        #         )  # need to initialize variables so that we can write data for the tests (before solve_forward)
        #         assert scenario.steady
        return

    def solve_forward(self):
        """forward thermal analysis of FUN3D"""
        for scenario in self.model.scenarios:
            # pre analysis setup
            super(Fun3dAeroelasticTestInterface, self).set_variables(
                scenario, self.model.bodies
            )
            super(Fun3dAeroelasticTestInterface, self).set_functions(
                scenario, self.model.bodies
            )
            super(Fun3dAeroelasticTestInterface, self).initialize(scenario, self.model.bodies)

            """forward analysis starts here"""
            # first input the deformation on the surface
            for step in range(scenario.steps):
                for ibody, body in enumerate(self.model.bodies, 1):
                    aero_disps = body.get_aero_disps(
                        scenario, time_index=step, add_dxa0=False
                    )
                    aero_nnodes = body.get_aero_num_nodes()
                    deform = "deform" in body.motion_type
                    if deform and aero_disps is not None and aero_nnodes > 0:
                        dx = np.asfortranarray(aero_disps[0::3])
                        dy = np.asfortranarray(aero_disps[1::3])
                        dz = np.asfortranarray(aero_disps[2::3])
                        self.fun3d_flow.input_deformation(dx, dy, dz, body=ibody)

                # iterate which skips force and just does grid deformation (don't use thermal coupling here)
                self.comm.Barrier()
                self.fun3d_flow.iterate()

                for ibody, body in enumerate(self.model.bodies, 1):
                    # Compute the aerodynamic nodes on the body
                    aero_loads = body.get_aero_loads(scenario, time_index=step)
                    aero_nnodes = body.get_num_aero_nodes()
                    if aero_loads is not None and aero_nnodes > 0:
                        fx, fy, fz = self.fun3d_flow.extract_forces(aero_nnodes, body=ibody)

                        # Set the dimensional values of the forces
                        aero_loads[0::3] = scenario.qinf * fx[:]
                        aero_loads[1::3] = scenario.qinf * fy[:]
                        aero_loads[2::3] = scenario.qinf * fz[:]

            # post analysis in fun3d interface
            super(Fun3dAeroelasticTestInterface, self).post(scenario, self.model.bodies)
        return

    def solve_adjoint(self):
        """adjoint grid deformation analysis in FUN3D"""
        for scenario in self.model.scenarios:
            # pre analysis setup
            super(Fun3dAeroelasticTestInterface, self).set_variables(
                scenario, self.model.bodies
            )
            super(Fun3dAeroelasticTestInterface, self).set_functions(
                scenario, self.model.bodies
            )
            # for body in self.model.bodies:
            #     body.initialize_adjoint_variables(scenario)
            super(Fun3dAeroelasticTestInterface, self).initialize_adjoint(
                scenario, self.model.bodies
            )

            """adjoint analysis starts here"""
            # first input the grid volume adjoint variables lam_xG
            dtype = TransferScheme.dtype
            nf = scenario.count_adjoint_functions()

            for step in range(scenario.steps):
                nfuncs = scenario.count_adjoint_functions()
                for ibody, body in enumerate(self.model.bodies, 1):
                    # Get the adjoint Jacobian product for the aerodynamic loads
                    aero_loads_ajp = body.get_aero_loads_ajp(scenario)
                    aero_nnodes = body.get_num_aero_nodes()
                    if aero_loads_ajp is not None and aero_nnodes > 0:
                        aero_nnodes = body.get_num_aero_nodes()
                        psi_F = -aero_loads_ajp

                        dtype = TransferScheme.dtype
                        lam_x = np.zeros((aero_nnodes, nfuncs), dtype=dtype)
                        lam_y = np.zeros((aero_nnodes, nfuncs), dtype=dtype)
                        lam_z = np.zeros((aero_nnodes, nfuncs), dtype=dtype)

                        for func in range(nfuncs):
                            lam_x[:, func] = (
                                scenario.qinf * psi_F[0::3, func] / scenario.flow_dt
                            )
                            lam_y[:, func] = (
                                scenario.qinf * psi_F[1::3, func] / scenario.flow_dt
                            )
                            lam_z[:, func] = (
                                scenario.qinf * psi_F[2::3, func] / scenario.flow_dt
                            )

                        self.fun3d_adjoint.input_force_adjoint(lam_x, lam_y, lam_z, body=ibody)

                self.fun3d_adjoint.iterate(step)

                for ibody, body in enumerate(self.model.bodies, 1):
                    # Extract aero_disps_ajp = dG/du_A^T psi_G from FUN3D
                    aero_disps_ajp = body.get_aero_disps_ajp(scenario)
                    aero_nnodes = body.get_num_aero_nodes()
                    if aero_disps_ajp is not None and aero_nnodes > 0:
                        lam_x, lam_y, lam_z = self.fun3d_adjoint.extract_grid_adjoint_product(
                            aero_nnodes, nfuncs, body=ibody
                        )

                        for func in range(nfuncs):
                            aero_disps_ajp[0::3, func] = lam_x[:, func] * scenario.flow_dt
                            aero_disps_ajp[1::3, func] = lam_y[:, func] * scenario.flow_dt
                            aero_disps_ajp[2::3, func] = lam_z[:, func] * scenario.flow_dt

            # call post adjoint
            super(Fun3dAeroelasticTestInterface, self).post_adjoint(scenario, self.model.bodies)
        return

    @classmethod
    def complex_step_test(cls, fun3d_ae_interface, epsilon=1e-30, filename="fun3d_AE_adjoint.txt"):
        assert isinstance(fun3d_ae_interface, cls)
        model = fun3d_ae_interface.model
        body = model.bodies[0]
        scenario = model.scenarios[0]
        na = body.get_num_aero_nodes()
        # nf = scenario.count_adjoint_functions()

        ua0 = body.get_aero_disps(scenario) * 1.0
        # deform the whole mesh up by +0.01 in the z direction
        ua0[2::3] += 0.01
        duads = np.random.rand(3*na)
        aero_loads_ajp = body.get_aero_loads_ajp(scenario)
        lamL = -aero_loads_ajp

        # set lamL to a random value
        lamL[:] = np.random.rand(3*na)[:]

        dtype = TransferScheme.dtype
        adj_product = None
        cmplx_product = None

        # forward analysis loads(disps)
        fun3d_ae_interface.solve_forward()

        # adjoint analysis on loads(disps), input load adjoint
        fun3d_ae_interface.solve_adjoint()
        lamD = body.get_aero_disps_ajp(scenario)

        adj_product = np.dot(lamD[:, 0], duads)
        # then sum across all processes
        adj_product = fun3d_ae_interface.comm.allreduce(adj_product)


        # forward analysis loads(ua+dua/ds*h*1j)
        aero_disps = body.get_aero_disps(scenario)
        aero_disps[:] = ua0[:] + duads[:] * epsilon * 1j
        fun3d_ae_interface.solve_forward()
        f_loads = body.get_aero_loads(scenario)

        cmplx_product = np.dot(np.imag(f_loads) / epsilon, lamL[:, 0])
        cmplx_product = fun3d_ae_interface.comm.allreduce(cmplx_product)

        rel_error = (adj_product - cmplx_product) / cmplx_product

        print(f"Fun3d 14 Interface aerothermal ajp test")
        print(f"\tadj product = {adj_product}")
        print(f"\tcmplx product = {cmplx_product}")
        print(f"\trel error = {rel_error}")

        # run the complex step test
        func_name = model.get_functions()[0].name
        hdl = open(filename, "w")
        TestResult(
            name="fun3d_ae_test",
            func_names=[func_name],
            complex_TD=[cmplx_product],
            adjoint_TD=[adj_product],
            rel_error=[rel_error],
            comm=fun3d_ae_interface.comm,
            method="complex step",
        ).write(hdl)
        return rel_error
