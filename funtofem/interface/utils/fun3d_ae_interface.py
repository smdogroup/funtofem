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

__all__ = ["Fun3dAeroelasticTestInterface", "Fun3d14AeroelasticTestInterface"]

import numpy as np
from funtofem import TransferScheme
from ..fun3d_interface import Fun3dInterface
from ..fun3d_14_interface import Fun3d14Interface
from ..test_solver import TestResult
import importlib, os, sys


class Fun3dAeroelasticTestInterface(Fun3dInterface):
    """
    FUN3D AE interface for unit testing of FUN3D aeroelastic code

    FUN3D's FUNtoFEM coupling interface requires no additional configure flags to compile.
    To tell FUN3D that a body's motion should be driven by FUNtoFEM, set *motion_driver(i)='funtofem'*.

    Intended use: tests for aero_loads and flow_states.
    """

    def __init__(
        self,
        comm,
        model,
        fun3d_project_name=None,
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
            fun3d_project_name=fun3d_project_name,
            fun3d_dir=fun3d_dir,
            forward_options=forward_options,
            adjoint_options=adjoint_options,
            forward_tolerance=1e1,
            adjoint_tolerance=1e4,
        )

        self.comm.Barrier()
        self.complex_mode = complex_mode

        # state variables related to grid deformation
        for body in self.model.bodies:
            # initialize transfer schemes for the body classes so the elastic variables will be there
            body.initialize_transfer(
                self.comm,
                self.comm,
                0,
                self.comm,
                0,
                transfer_settings=None,
            )

            for scenario in self.model.scenarios:
                body.initialize_variables(
                    scenario
                )  # need to initialize variables so that we can write data for the tests (before solve_forward)
                assert scenario.steady

                body.initialize_adjoint_variables(scenario)
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
            super(Fun3dAeroelasticTestInterface, self).initialize(
                scenario, self.model.bodies
            )

            # self.fun3d_flow.set_coupling_frequency(scenario.forward_coupling_frequency)

            """forward analysis starts here"""
            for step in range(scenario.steps):
                # first input the deformation on the surface
                for ibody, body in enumerate(self.model.bodies, 1):
                    aero_disps = body.get_aero_disps(scenario, add_dxa0=False)
                    aero_nnodes = body.get_num_aero_nodes()
                    deform = "deform" in body.motion_type
                    if deform and aero_disps is not None and aero_nnodes > 0:
                        dx = np.asfortranarray(aero_disps[0::3])
                        dy = np.asfortranarray(aero_disps[1::3])
                        dz = np.asfortranarray(aero_disps[2::3])

                        dx = dx if self.complex_mode else dx.astype(np.double)
                        dy = dy if self.complex_mode else dy.astype(np.double)
                        dz = dz if self.complex_mode else dz.astype(np.double)
                        self.fun3d_flow.input_deformation(dx, dy, dz, body=ibody)

                # iterate which skips force and just does grid deformation (don't use thermal coupling here)
                self.comm.Barrier()
                self.fun3d_flow.iterate()
                self._last_forward_step = step + 1

                for ibody, body in enumerate(self.model.bodies, 1):
                    # Compute the aerodynamic nodes on the body
                    aero_loads = body.get_aero_loads(scenario)
                    aero_nnodes = body.get_num_aero_nodes()
                    if aero_loads is not None and aero_nnodes > 0:
                        fx, fy, fz = self.fun3d_flow.extract_forces(
                            aero_nnodes, body=ibody
                        )

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
            super(Fun3dAeroelasticTestInterface, self).initialize_adjoint(
                scenario, self.model.bodies
            )

            """adjoint analysis starts here"""
            # first input the grid volume adjoint variables lam_xG
            dtype = TransferScheme.dtype

            # self.fun3d_adjoint.set_coupling_frequency(scenario.adjoint_coupling_frequency)

            nfuncs = scenario.count_adjoint_functions()

            for step in range(scenario.adjoint_steps):
                for ibody, body in enumerate(self.model.bodies, 1):
                    # Get the adjoint Jacobian product for the aerodynamic loads
                    aero_loads_ajp = body.get_aero_loads_ajp(scenario)
                    aero_nnodes = body.get_num_aero_nodes()
                    if aero_loads_ajp is not None and aero_nnodes > 0:
                        # aero_nnodes = body.get_num_aero_nodes()
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

                        self.fun3d_adjoint.input_force_adjoint(
                            lam_x, lam_y, lam_z, body=ibody
                        )

                self.comm.Barrier()
                self.fun3d_adjoint.iterate(step + 1)

                self._last_adjoint_step = step + 1

                for ibody, body in enumerate(self.model.bodies, 1):
                    # Extract aero_disps_ajp = dG/du_A^T psi_G from FUN3D
                    aero_disps_ajp = body.get_aero_disps_ajp(scenario)
                    aero_nnodes = body.get_num_aero_nodes()

                    if aero_disps_ajp is not None and aero_nnodes > 0:
                        lam_x, lam_y, lam_z = (
                            self.fun3d_adjoint.extract_grid_adjoint_product(
                                aero_nnodes, nfuncs, body=ibody
                            )
                        )

                        for func in range(nfuncs):
                            aero_disps_ajp[0::3, func] = (
                                lam_x[:, func] * scenario.flow_dt
                            )
                            aero_disps_ajp[1::3, func] = (
                                lam_y[:, func] * scenario.flow_dt
                            )
                            aero_disps_ajp[2::3, func] = (
                                lam_z[:, func] * scenario.flow_dt
                            )

            # call post adjoint
            super(Fun3dAeroelasticTestInterface, self).post_adjoint(
                scenario, self.model.bodies
            )
        return

    @classmethod
    def complex_step_test_flow_states(
        cls,
        fun3d_ae_interface,
        epsilon=1e-30,
        filename="fun3d_AE_adjoint.txt",
    ):
        """test the vector function of aero loads: fA(uA) from the displacements and the associated adjoints"""

        assert isinstance(fun3d_ae_interface, cls)
        model = fun3d_ae_interface.model
        body = model.bodies[0]
        scenario = model.scenarios[0]
        na = body.get_num_aero_nodes()
        nvol = fun3d_ae_interface.nvol
        nf = scenario.count_adjoint_functions()

        dtype = TransferScheme.dtype

        duads = np.random.rand(3 * na).astype(dtype)
        flow_ajp = np.random.rand(5 * nvol, nf).astype(dtype)
        body._flow_ajp[scenario.id] = flow_ajp * 1.0

        if na != 0:
            ua = body.get_aero_disps(scenario)
            # deform the whole mesh up by +0.01 in the z direction
            ua[2::3] += 0.01
            ua0 = ua * 1.0

        adj_product = None
        fd_product = None

        # forward analysis loads(disps)
        fun3d_ae_interface.solve_forward()

        # adjoint analysis on loads(disps), input load adjoint
        fun3d_ae_interface.solve_adjoint()
        lamD = body.get_aero_disps_ajp(scenario)

        if na != 0:
            adj_product = np.dot(lamD[:, 0], duads)
        else:
            adj_product = 0.0

        # then sum across all processes
        adj_product = fun3d_ae_interface.comm.allreduce(adj_product)

        # start FD computation
        if na != 0:
            # forward analysis loads(ua+dua/ds*h)
            aero_disps = body.get_aero_disps(scenario)
            aero_disps[:] = ua0[:] + duads[:] * epsilon * 1j
        fun3d_ae_interface.solve_forward()
        f_flow = body._aero_flow_states[scenario.id] * 1.0

        if na != 0:
            cmplx_product = np.dot(np.imag(f_flow) / epsilon, flow_ajp[:, 0])
        else:
            cmplx_product = 0.0

        cmplx_product = fun3d_ae_interface.comm.allreduce(cmplx_product)

        rel_error = (adj_product - cmplx_product) / cmplx_product

        adj_product = adj_product.real
        cmplx_product = cmplx_product.real
        rel_error = rel_error.real

        if fun3d_ae_interface.comm.rank == 0:
            print(f"Fun3d 14 Interface AE ajp test on flow states")
            print(f"\tadj product = {adj_product}")
            print(f"\tcomplex step product = {cmplx_product}")
            print(f"\trel error = {rel_error}")

        # run the complex step test
        func_name = model.get_functions()[0].name
        hdl = open(filename, "w")
        TestResult(
            name="fun3d_flow_state_test",
            func_names=[func_name],
            complex_TD=[cmplx_product],
            adjoint_TD=[adj_product],
            rel_error=[rel_error],
            comm=fun3d_ae_interface.comm,
            method="finite diff",
        ).write(hdl)
        return abs(rel_error)

    @classmethod
    def complex_step_test_aero_loads(
        cls, fun3d_ae_interface, epsilon=1e-30, filename="fun3d_AE_adjoint.txt"
    ):
        assert isinstance(fun3d_ae_interface, cls)
        model = fun3d_ae_interface.model
        body = model.bodies[0]
        scenario = model.scenarios[0]
        na = body.get_num_aero_nodes()
        nf = scenario.count_adjoint_functions()
        comm = fun3d_ae_interface.comm

        # gather the list of na and sum of na
        global_na = comm.reduce(na, root=0)
        na_list = comm.gather(na, root=0)
        # randomize the vectors on root proc and scatter them
        # so serial and MPI are equivalent
        if comm.rank == 0:
            glob_duads = np.random.rand(3 * global_na)
            glob_lamL = np.random.rand(3 * global_na, nf)
            duads_proc_list = []
            lamL_proc_list = []
            start = 0
            for rank, _na in enumerate(na_list):
                duads_proc_list += [glob_duads[3 * start : 3 * (start + _na)]]
                lamL_proc_list += [glob_lamL[3 * start : 3 * (start + _na), :]]
                start += _na
        else:
            duads_proc_list = None
            lamL_proc_list = None
        duads = comm.scatter(duads_proc_list, root=0)
        lamL = comm.scatter(lamL_proc_list, root=0)

        aero_loads_ajp = body.get_aero_loads_ajp(scenario)
        print(na)
        if na != 0:
            ua = body.get_aero_disps(scenario)
            # deform the whole mesh up by +0.01 in the z direction
            ua[2::3] += 0.01

            ua0 = ua * 1.0
            aero_loads_ajp[:, :] = lamL[:, :] * 1.0

        dtype = TransferScheme.dtype
        adj_product = None
        cmplx_product = None

        # forward analysis loads(disps)
        fun3d_ae_interface.solve_forward()

        # adjoint analysis on loads(disps), input load adjoint
        fun3d_ae_interface.solve_adjoint()
        lamD = body.get_aero_disps_ajp(scenario)

        if na != 0:
            adj_product = np.dot(lamD[:, 0], duads)
        else:
            adj_product = 0.0

        # then sum across all processes
        adj_product = fun3d_ae_interface.comm.allreduce(adj_product)

        fun3d_ae_interface = cls.copy_complex_interface(fun3d_ae_interface)

        if na != 0:
            # forward analysis loads(ua+dua/ds*h*1j)
            aero_disps = body.get_aero_disps(scenario)
            aero_disps[:] = ua0[:] + duads[:] * epsilon * 1j
        fun3d_ae_interface.solve_forward()
        f_loads = body.get_aero_loads(scenario)

        if na != 0:
            cmplx_product = np.dot(np.imag(f_loads) / epsilon, lamL[:, 0])
        else:
            cmplx_product = 0.0

        cmplx_product = fun3d_ae_interface.comm.allreduce(cmplx_product)

        rel_error = (adj_product - cmplx_product) / cmplx_product

        adj_product = adj_product.real
        cmplx_product = cmplx_product.real
        rel_error = rel_error.real

        if fun3d_ae_interface.comm.rank == 0:
            print(f"Fun3d 13.6 Interface AE ajp test")
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
        return abs(rel_error)

    @classmethod
    def finite_diff_test(
        cls, fun3d_ae_interface, epsilon=1e-4, filename="fun3d_AE_adjoint.txt"
    ):
        assert isinstance(fun3d_ae_interface, cls)
        model = fun3d_ae_interface.model
        body = model.bodies[0]
        scenario = model.scenarios[0]
        na = body.get_num_aero_nodes()
        nf = scenario.count_adjoint_functions()

        duads = np.random.rand(3 * na)
        aero_loads_ajp = body.get_aero_loads_ajp(scenario)
        print(na)
        if na != 0:
            ua = body.get_aero_disps(scenario)
            # deform the whole mesh up by +0.01 in the z direction
            ua[2::3] += 0.01

            ua0 = ua * 1.0
            lamL = aero_loads_ajp

            # set lamL to a random value
            lamL[:, :] = np.random.rand((3 * na), nf)[:, :]

        dtype = TransferScheme.dtype
        adj_product = None
        fd_product = None

        # forward analysis loads(disps)
        fun3d_ae_interface.solve_forward()

        # adjoint analysis on loads(disps), input load adjoint
        fun3d_ae_interface.solve_adjoint()
        lamD = body.get_aero_disps_ajp(scenario)

        if na != 0:
            adj_product = np.dot(lamD[:, 0], duads)
        else:
            adj_product = 0.0

        # then sum across all processes
        adj_product = fun3d_ae_interface.comm.allreduce(adj_product)

        # start FD computation

        if na != 0:
            # forward analysis loads(ua+dua/ds*h)
            aero_disps = body.get_aero_disps(scenario)
            aero_disps[:] = ua0[:] + duads[:] * epsilon
        fun3d_ae_interface.solve_forward()
        f_loads = body.get_aero_loads(scenario) * 1.0

        if na != 0:
            # forward analysis loads(ua-dua/ds*h)
            aero_disps = body.get_aero_disps(scenario)
            aero_disps[:] = ua0[:] - duads[:] * epsilon
        fun3d_ae_interface.solve_forward()
        i_loads = body.get_aero_loads(scenario) * 1.0

        if na != 0:
            fd_product = np.dot((f_loads - i_loads) / 2.0 / epsilon, lamL[:, 0])
        else:
            fd_product = 0.0

        fd_product = fun3d_ae_interface.comm.allreduce(fd_product)

        rel_error = (adj_product - fd_product) / fd_product

        adj_product = adj_product.real
        fd_product = fd_product.real
        rel_error = rel_error.real

        if fun3d_ae_interface.comm.rank == 0:
            print(f"Fun3d 13 Interface AE ajp test")
            print(f"\tadj product = {adj_product}")
            print(f"\tcentral diff product = {fd_product}")
            print(f"\trel error = {rel_error}")

        # run the complex step test
        func_name = model.get_functions()[0].name
        hdl = open(filename, "w")
        TestResult(
            name="fun3d_ae_test",
            func_names=[func_name],
            complex_TD=[fd_product],
            adjoint_TD=[adj_product],
            rel_error=[rel_error],
            comm=fun3d_ae_interface.comm,
            method="complex step",
        ).write(hdl)
        return abs(rel_error)

    @classmethod
    def copy_complex_interface(cls, fun3d_interface):
        """
        copy used for derivative testing
        driver.solvers.make_complex_flow()
        """

        # unload and reload fun3d Flow, Adjoint as complex versions
        os.environ["CMPLX_MODE"] = "1"
        importlib.reload(sys.modules["fun3d.interface"])

        return cls(
            comm=fun3d_interface.comm,
            model=fun3d_interface.model,
            fun3d_dir=fun3d_interface.fun3d_dir,
            fun3d_project_name=fun3d_interface.fun3d_project_name,
            complex_mode=True,
        )


class Fun3d14AeroelasticTestInterface(Fun3d14Interface):
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
        test_flow_states=False,
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
        super(Fun3d14AeroelasticTestInterface, self).__init__(
            comm=comm,
            model=model,
            fun3d_dir=fun3d_dir,
            forward_options=forward_options,
            adjoint_options=adjoint_options,
            complex_mode=complex_mode,
        )

        self.comm.Barrier()
        # get the number of grid volume coordinates
        self.test_flow_states = test_flow_states
        if test_flow_states:
            self.nvol = self.fun3d_flow.extract_num_volume_nodes()
            self.nflow = 5  # if inviscid, else 6 if turb
        else:
            self.nvol = None
            self.nflow = None

        # state variables related to grid deformation
        for body in self.model.bodies:
            # initialize transfer schemes for the body classes so the elastic variables will be there
            body.initialize_transfer(
                self.comm,
                self.comm,
                0,
                self.comm,
                0,
                transfer_settings=None,
            )

            body._aero_flow_states = {}
            body._flow_ajp = {}

            for scenario in self.model.scenarios:
                body.initialize_variables(
                    scenario
                )  # need to initialize variables so that we can write data for the tests (before solve_forward)
                assert scenario.steady

                body.initialize_adjoint_variables(scenario)
                if self.test_flow_states:
                    body._aero_flow_states[scenario.id] = np.zeros(
                        (self.nflow * self.nvol), dtype=TransferScheme.dtype
                    )
                    nf = scenario.count_adjoint_functions()
                    body._flow_ajp[scenario.id] = np.zeros(
                        (self.nflow * self.nvol, nf), dtype=TransferScheme.dtype
                    )  # 1 func for now
        return

    def solve_forward(self):
        """forward thermal analysis of FUN3D"""
        for scenario in self.model.scenarios:
            # pre analysis setup
            super(Fun3d14AeroelasticTestInterface, self).set_variables(
                scenario, self.model.bodies
            )
            super(Fun3d14AeroelasticTestInterface, self).set_functions(
                scenario, self.model.bodies
            )
            super(Fun3d14AeroelasticTestInterface, self).initialize(
                scenario, self.model.bodies
            )

            self.fun3d_flow.set_coupling_frequency(scenario.forward_coupling_frequency)

            """forward analysis starts here"""
            for step in range(scenario.steps):
                # first input the deformation on the surface
                for ibody, body in enumerate(self.model.bodies, 1):
                    aero_disps = body.get_aero_disps(scenario, add_dxa0=False)
                    aero_nnodes = body.get_num_aero_nodes()
                    deform = "deform" in body.motion_type
                    if deform and aero_disps is not None and aero_nnodes > 0:
                        dx = np.asfortranarray(aero_disps[0::3])
                        dy = np.asfortranarray(aero_disps[1::3])
                        dz = np.asfortranarray(aero_disps[2::3])

                        dx = dx if self.complex_mode else dx.astype(np.double)
                        dy = dy if self.complex_mode else dy.astype(np.double)
                        dz = dz if self.complex_mode else dz.astype(np.double)

                        self.fun3d_flow.input_deformation(dx, dy, dz, body=ibody)

                self.comm.Barrier()
                # iterate which skips force and just does grid deformation (don't use thermal coupling here)
                for inner_step in range(scenario.forward_coupling_frequency):
                    self.fun3d_flow.iterate()

                self._last_forward_step = (
                    step * scenario.forward_coupling_frequency + inner_step + 1
                )

                for ibody, body in enumerate(self.model.bodies, 1):
                    # Compute the aerodynamic nodes on the body
                    aero_loads = body.get_aero_loads(scenario)
                    aero_nnodes = body.get_num_aero_nodes()
                    if aero_loads is not None and aero_nnodes > 0:
                        fx, fy, fz = self.fun3d_flow.extract_forces(
                            aero_nnodes, body=ibody
                        )

                        # Set the dimensional values of the forces
                        aero_loads[0::3] = scenario.qinf * fx[:]
                        aero_loads[1::3] = scenario.qinf * fy[:]
                        aero_loads[2::3] = scenario.qinf * fz[:]

                    # get the flow states
                    if self.test_flow_states:
                        # receive the deformation in the volume
                        flow1, flow2, flow3, flow4, flow5 = (
                            self.fun3d_flow.extract_flow_states()
                        )
                        flow_states = body._aero_flow_states[scenario.id]
                        flow_states[0::5] = flow1[:]
                        flow_states[1::5] = flow2[:]
                        flow_states[2::5] = flow3[:]
                        flow_states[3::5] = flow4[:]
                        flow_states[4::5] = flow5[:]

            # post analysis in fun3d interface
            super(Fun3d14AeroelasticTestInterface, self).post(
                scenario, self.model.bodies
            )
        return

    def solve_adjoint(self):
        """adjoint grid deformation analysis in FUN3D"""
        for scenario in self.model.scenarios:
            # pre analysis setup
            super(Fun3d14AeroelasticTestInterface, self).set_variables(
                scenario, self.model.bodies
            )
            super(Fun3d14AeroelasticTestInterface, self).set_functions(
                scenario, self.model.bodies
            )
            super(Fun3d14AeroelasticTestInterface, self).initialize_adjoint(
                scenario, self.model.bodies
            )

            """adjoint analysis starts here"""
            # first input the grid volume adjoint variables lam_xG
            dtype = TransferScheme.dtype
            nfuncs = scenario.count_adjoint_functions()

            self.fun3d_adjoint.set_coupling_frequency(
                scenario.adjoint_coupling_frequency
            )

            for step in range(scenario.adjoint_steps):
                for ibody, body in enumerate(self.model.bodies, 1):
                    # Get the adjoint Jacobian product for the aerodynamic loads
                    aero_loads_ajp = body.get_aero_loads_ajp(scenario)
                    aero_nnodes = body.get_num_aero_nodes()
                    if aero_loads_ajp is not None and aero_nnodes > 0:
                        # aero_nnodes = body.get_num_aero_nodes()
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

                        if not self.complex_mode:
                            lam_x = lam_x.astype(np.double)
                            lam_y = lam_y.astype(np.double)
                            lam_z = lam_z.astype(np.double)

                        lam_x = np.asfortranarray(lam_x)
                        lam_y = np.asfortranarray(lam_y)
                        lam_z = np.asfortranarray(lam_z)

                        self.fun3d_adjoint.input_force_adjoint(
                            lam_x, lam_y, lam_z, body=ibody
                        )

                    # set the flow state adjoint in
                    if self.test_flow_states:
                        dtype = TransferScheme.dtype
                        flow_ajp = body._flow_ajp[scenario.id] * 1.0
                        lamq1 = flow_ajp[0::5, :]
                        lamq2 = flow_ajp[1::5, :]
                        lamq3 = flow_ajp[2::5, :]
                        lamq4 = flow_ajp[3::5, :]
                        lamq5 = flow_ajp[4::5, :]

                        if not self.complex_mode:
                            lamq1 = lamq1.astype(np.double)
                            lamq2 = lamq2.astype(np.double)
                            lamq3 = lamq3.astype(np.double)
                            lamq4 = lamq4.astype(np.double)
                            lamq5 = lamq5.astype(np.double)

                        lamq1 = np.asfortranarray(lamq1)
                        lamq2 = np.asfortranarray(lamq2)
                        lamq3 = np.asfortranarray(lamq3)
                        lamq4 = np.asfortranarray(lamq4)
                        lamq5 = np.asfortranarray(lamq5)

                        self.fun3d_adjoint.input_flow_adjoint(
                            lamq1, lamq2, lamq3, lamq4, lamq5
                        )

                self.comm.Barrier()
                for inner_step in range(scenario.adjoint_coupling_frequency):
                    self._last_adjoint_step = (
                        step * scenario.adjoint_coupling_frequency + inner_step + 1
                    )
                    self.fun3d_adjoint.iterate(self._last_adjoint_step)

                for ibody, body in enumerate(self.model.bodies, 1):
                    # Extract aero_disps_ajp = dG/du_A^T psi_G from FUN3D
                    aero_disps_ajp = body.get_aero_disps_ajp(scenario)
                    aero_nnodes = body.get_num_aero_nodes()

                    if aero_disps_ajp is not None and aero_nnodes > 0:
                        lam_x, lam_y, lam_z = (
                            self.fun3d_adjoint.extract_grid_adjoint_product(
                                aero_nnodes, nfuncs, body=ibody
                            )
                        )

                        for func in range(nfuncs):
                            aero_disps_ajp[0::3, func] = (
                                lam_x[:, func] * scenario.flow_dt
                            )
                            aero_disps_ajp[1::3, func] = (
                                lam_y[:, func] * scenario.flow_dt
                            )
                            aero_disps_ajp[2::3, func] = (
                                lam_z[:, func] * scenario.flow_dt
                            )

            # call post adjoint
            super(Fun3d14AeroelasticTestInterface, self).post_adjoint(
                scenario, self.model.bodies
            )
        return

    @classmethod
    def finite_diff_test_flow_states(
        cls,
        fun3d_ae_interface,
        epsilon=1e-4,
        filename="fun3d_AE_adjoint.txt",
    ):
        """test the vector function of aero loads: fA(uA) from the displacements and the associated adjoints"""

        assert isinstance(fun3d_ae_interface, cls)
        model = fun3d_ae_interface.model
        body = model.bodies[0]
        scenario = model.scenarios[0]
        na = body.get_num_aero_nodes()
        nvol = fun3d_ae_interface.nvol
        nf = scenario.count_adjoint_functions()
        comm = fun3d_ae_interface.comm

        dtype = TransferScheme.dtype

        duads = np.random.rand(3 * na).astype(dtype)
        if comm.rank == 0:
            flow_ajp = np.random.rand(5 * nvol, nf).astype(dtype)
        else:
            flow_ajp = None
        flow_ajp = comm.bcast(flow_ajp, root=0)
        body._flow_ajp[scenario.id] = flow_ajp * 1.0

        if na != 0:
            ua = body.get_aero_disps(scenario)
            # deform the whole mesh up by +0.01 in the z direction
            ua[2::3] += 0.01
            ua0 = ua * 1.0

        adj_product = None
        fd_product = None

        # forward analysis loads(disps)
        fun3d_ae_interface.solve_forward()

        # adjoint analysis on loads(disps), input load adjoint
        fun3d_ae_interface.solve_adjoint()
        lamD = body.get_aero_disps_ajp(scenario)

        if na != 0:
            adj_product = np.dot(lamD[:, 0], duads)
        else:
            adj_product = 0.0

        # then sum across all processes
        adj_product = fun3d_ae_interface.comm.allreduce(adj_product)

        # start FD computation
        if na != 0:
            # forward analysis loads(ua+dua/ds*h)
            aero_disps = body.get_aero_disps(scenario)
            aero_disps[:] = ua0[:] + duads[:] * epsilon
        fun3d_ae_interface.solve_forward()
        f_flow = body._aero_flow_states[scenario.id] * 1.0

        if na != 0:
            # forward analysis loads(ua-dua/ds*h)
            aero_disps = body.get_aero_disps(scenario)
            aero_disps[:] = ua0[:] - duads[:] * epsilon
        fun3d_ae_interface.solve_forward()
        i_flow = body._aero_flow_states[scenario.id] * 1.0

        if na != 0:
            fd_product = np.dot((f_flow - i_flow) / 2.0 / epsilon, flow_ajp[:, 0])
        else:
            fd_product = 0.0

        fd_product = fun3d_ae_interface.comm.allreduce(fd_product)

        rel_error = (adj_product - fd_product) / fd_product

        adj_product = adj_product.real
        fd_product = fd_product.real
        rel_error = rel_error.real

        if fun3d_ae_interface.comm.rank == 0:
            print(f"Fun3d 14 Interface AE ajp test on flow states")
            print(f"\tadj product = {adj_product}")
            print(f"\tcentral diff product = {fd_product}")
            print(f"\trel error = {rel_error}")

        # run the complex step test
        func_name = model.get_functions()[0].name
        if fun3d_ae_interface.comm.rank == 0:
            hdl = open(filename, "w")
            TestResult(
                name="fun3d_ae_test_flow_states",
                func_names=[func_name],
                complex_TD=[fd_product],
                adjoint_TD=[adj_product],
                rel_error=[rel_error],
                comm=fun3d_ae_interface.comm,
                method="finite diff",
            ).write(hdl)
        return abs(rel_error)

    @classmethod
    def finite_diff_test_aero_loads_serial(
        cls,
        fun3d_ae_interface,
        epsilon=1e-4,
        ua_index=None,
        ua0_scale=1e-3,
        filename="fun3d_AE_adjoint.txt",
    ):
        """test the vector function of aero loads: fA(uA) from the displacements and the associated adjoints"""

        assert isinstance(fun3d_ae_interface, cls)
        model = fun3d_ae_interface.model
        body = model.bodies[0]
        scenario = model.scenarios[0]
        na = body.get_num_aero_nodes()
        nf = scenario.count_adjoint_functions()
        comm = fun3d_ae_interface.comm

        # gather the list of na and sum of na
        temp = np.random.rand(3 * na)
        duads = np.zeros((3 * na,))
        if ua_index is None:
            duads = temp * 1.0
        else:
            assert isinstance(ua_index, list)
            for _index in ua_index:
                duads[3 * _index + 2] = temp[3 * _index + 2]
        for ia in range(na):
            xyz = duads[3 * ia : 3 * ia + 3]
            print(f"duads ind {ia} = {xyz}")
        lamF = np.random.rand(3 * na, nf)

        aero_loads_ajp = body.get_aero_loads_ajp(scenario)
        print(na)
        ua = body.get_aero_disps(scenario)
        # deform the whole mesh up by +0.01 in the z direction
        ua0 = ua * 0.0
        # if ua_index is None:
        #    ua0[2::3] += ua0_scale
        # else:
        #    ua0[3*ua_index] += ua0_scale
        ua0[2::3] += ua0_scale
        aero_loads_ajp[:, :] = lamF[:, :] * 1.0

        dtype = TransferScheme.dtype
        adj_product = None
        fd_product = None

        # forward analysis loads(disps)
        fun3d_ae_interface.solve_forward()

        # adjoint analysis on loads(disps), input load adjoint
        fun3d_ae_interface.solve_adjoint()
        lamD = body.get_aero_disps_ajp(scenario)

        adj_product = np.dot(lamD[:, 0], duads)
        # start FD computation

        # forward analysis loads(ua+dua/ds*h)
        aero_disps = body.get_aero_disps(scenario)
        aero_disps[:] = ua0[:] + duads[:] * epsilon
        fun3d_ae_interface.solve_forward()
        f_loads = body.get_aero_loads(scenario) * 1.0

        # forward analysis loads(ua-dua/ds*h)
        aero_disps[:] = ua0[:] - duads[:] * epsilon
        fun3d_ae_interface.solve_forward()
        i_loads = body.get_aero_loads(scenario) * 1.0

        fd_product = np.dot((f_loads - i_loads) / 2.0 / epsilon, lamF[:, 0])
        rel_error = (adj_product - fd_product) / fd_product

        adj_product = adj_product.real
        fd_product = fd_product.real
        rel_error = rel_error.real

        if fun3d_ae_interface.comm.rank == 0:
            print(f"Fun3d 14 Interface AE ajp test")
            print(f"\tadj product = {adj_product}")
            print(f"\tcentral diff product = {fd_product}")
            print(f"\trel error = {rel_error}")

        # run the complex step test
        func_name = model.get_functions()[0].name
        if fun3d_ae_interface.comm.rank == 0:
            hdl = open(filename, "w")
            TestResult(
                name="fun3d_ae_test_aero_loads",
                func_names=[func_name],
                complex_TD=[fd_product],
                adjoint_TD=[adj_product],
                rel_error=[rel_error],
                comm=fun3d_ae_interface.comm,
                method="finite diff",
            ).write(hdl)
        return abs(rel_error)

    @classmethod
    def finite_diff_test_aero_loads(
        cls, fun3d_ae_interface, epsilon=1e-4, filename="fun3d_AE_adjoint.txt"
    ):
        """test the vector function of aero loads: fA(uA) from the displacements and the associated adjoints"""

        assert isinstance(fun3d_ae_interface, cls)
        model = fun3d_ae_interface.model
        body = model.bodies[0]
        scenario = model.scenarios[0]
        na = body.get_num_aero_nodes()
        nf = scenario.count_adjoint_functions()
        comm = fun3d_ae_interface.comm

        # gather the list of na and sum of na
        global_na = comm.reduce(na, root=0)
        na_list = comm.gather(na, root=0)
        # randomize the vectors on root proc and scatter them
        # so serial and MPI are equivalent
        if comm.rank == 0:
            glob_duads = np.random.rand(3 * global_na)
            glob_lamL = np.random.rand(3 * global_na, nf)
            duads_proc_list = []
            lamL_proc_list = []
            start = 0
            for rank, _na in enumerate(na_list):
                duads_proc_list += [glob_duads[3 * start : 3 * (start + _na)]]
                lamL_proc_list += [glob_lamL[3 * start : 3 * (start + _na), :]]
                start += _na
        else:
            duads_proc_list = None
            lamL_proc_list = None
        duads = comm.scatter(duads_proc_list, root=0)
        lamL = comm.scatter(lamL_proc_list, root=0)

        aero_loads_ajp = body.get_aero_loads_ajp(scenario)
        print(na)
        if na != 0:
            ua = body.get_aero_disps(scenario)
            # deform the whole mesh up by +0.01 in the z direction
            ua[2::3] += 0.01

            ua0 = ua * 1.0
            aero_loads_ajp[:, :] = lamL[:, :] * 1.0

        dtype = TransferScheme.dtype
        adj_product = None
        fd_product = None

        # forward analysis loads(disps)
        fun3d_ae_interface.solve_forward()

        # adjoint analysis on loads(disps), input load adjoint
        fun3d_ae_interface.solve_adjoint()
        lamD = body.get_aero_disps_ajp(scenario)

        if na != 0:
            adj_product = np.dot(lamD[:, 0], duads)
        else:
            adj_product = 0.0

        # then sum across all processes
        adj_product = fun3d_ae_interface.comm.allreduce(adj_product)

        # start FD computation

        if na != 0:
            # forward analysis loads(ua+dua/ds*h)
            aero_disps = body.get_aero_disps(scenario)
            aero_disps[:] = ua0[:] + duads[:] * epsilon
        fun3d_ae_interface.solve_forward()
        f_loads = body.get_aero_loads(scenario) * 1.0

        if na != 0:
            # forward analysis loads(ua-dua/ds*h)
            aero_disps = body.get_aero_disps(scenario)
            aero_disps[:] = ua0[:] - duads[:] * epsilon
        fun3d_ae_interface.solve_forward()
        i_loads = body.get_aero_loads(scenario) * 1.0

        if na != 0:
            fd_product = np.dot((f_loads - i_loads) / 2.0 / epsilon, lamL[:, 0])
        else:
            fd_product = 0.0

        fd_product = fun3d_ae_interface.comm.allreduce(fd_product)

        rel_error = (adj_product - fd_product) / fd_product

        adj_product = adj_product.real
        fd_product = fd_product.real
        rel_error = rel_error.real

        if fun3d_ae_interface.comm.rank == 0:
            print(f"Fun3d 14 Interface AE ajp test")
            print(f"\tadj product = {adj_product}")
            print(f"\tcentral diff product = {fd_product}")
            print(f"\trel error = {rel_error}")

        # run the complex step test
        func_name = model.get_functions()[0].name
        if fun3d_ae_interface.comm.rank == 0:
            hdl = open(filename, "a")
            TestResult(
                name="fun3d_ae_test_aero_loads",
                func_names=[func_name],
                complex_TD=[fd_product],
                adjoint_TD=[adj_product],
                rel_error=[rel_error],
                comm=fun3d_ae_interface.comm,
                method="finite diff",
            ).write(hdl)
        return abs(rel_error)

    @classmethod
    def copy_complex_interface(cls, fun3d_interface):
        """
        copy used for derivative testing
        driver.solvers.make_complex_flow()
        """

        # unload and reload fun3d Flow, Adjoint as complex versions
        os.environ["CMPLX_MODE"] = "1"
        importlib.reload(sys.modules["fun3d.interface"])

        return cls(
            comm=fun3d_interface.comm,
            model=fun3d_interface.model,
            complex_mode=True,
            fun3d_dir=fun3d_interface.fun3d_dir,
        )

    @classmethod
    def complex_step_test_flow_states(
        cls,
        fun3d_ae_interface,
        epsilon=1e-30,
        filename="fun3d_AE_adjoint.txt",
    ):
        """test the vector function of aero loads: fA(uA) from the displacements and the associated adjoints"""

        assert isinstance(fun3d_ae_interface, cls)
        model = fun3d_ae_interface.model
        body = model.bodies[0]
        scenario = model.scenarios[0]
        na = body.get_num_aero_nodes()
        nvol = fun3d_ae_interface.nvol
        nf = scenario.count_adjoint_functions()

        dtype = TransferScheme.dtype

        duads = np.random.rand(3 * na).astype(dtype)
        flow_ajp = np.random.rand(5 * nvol, nf).astype(dtype)
        body._flow_ajp[scenario.id] = flow_ajp * 1.0

        if na != 0:
            ua = body.get_aero_disps(scenario)
            # deform the whole mesh up by +0.01 in the z direction
            ua[2::3] += 0.01
            ua0 = ua * 1.0

        adj_product = None
        fd_product = None

        # forward analysis loads(disps)
        fun3d_ae_interface.solve_forward()

        # adjoint analysis on loads(disps), input load adjoint
        fun3d_ae_interface.solve_adjoint()
        lamD = body.get_aero_disps_ajp(scenario)

        if na != 0:
            adj_product = np.dot(lamD[:, 0], duads)
        else:
            adj_product = 0.0

        # then sum across all processes
        adj_product = fun3d_ae_interface.comm.allreduce(adj_product)

        # start FD computation
        if na != 0:
            # forward analysis loads(ua+dua/ds*h)
            aero_disps = body.get_aero_disps(scenario)
            aero_disps[:] = ua0[:] + duads[:] * epsilon * 1j
        fun3d_ae_interface.solve_forward()
        f_flow = body._aero_flow_states[scenario.id] * 1.0

        if na != 0:
            cmplx_product = np.dot(np.imag(f_flow) / epsilon, flow_ajp[:, 0])
        else:
            cmplx_product = 0.0

        cmplx_product = fun3d_ae_interface.comm.allreduce(cmplx_product)

        rel_error = (adj_product - cmplx_product) / cmplx_product

        adj_product = adj_product.real
        cmplx_product = cmplx_product.real
        rel_error = rel_error.real

        if fun3d_ae_interface.comm.rank == 0:
            print(f"Fun3d 14 Interface AE ajp test on flow states")
            print(f"\tadj product = {adj_product}")
            print(f"\tcomplex step product = {cmplx_product}")
            print(f"\trel error = {rel_error}")

        # run the complex step test
        func_name = model.get_functions()[0].name
        hdl = open(filename, "a")
        TestResult(
            name="fun3d_flow_state_test",
            func_names=[func_name],
            complex_TD=[cmplx_product],
            adjoint_TD=[adj_product],
            rel_error=[rel_error],
            comm=fun3d_ae_interface.comm,
            method="finite diff",
        ).write(hdl)
        return abs(rel_error)

    @classmethod
    def complex_step_test_aero_loads(
        cls, fun3d_ae_interface, epsilon=1e-30, filename="fun3d_AE_adjoint.txt"
    ):
        assert isinstance(fun3d_ae_interface, cls)
        model = fun3d_ae_interface.model
        body = model.bodies[0]
        scenario = model.scenarios[0]
        na = body.get_num_aero_nodes()
        nf = scenario.count_adjoint_functions()

        duads = np.random.rand(3 * na)
        aero_loads_ajp = body.get_aero_loads_ajp(scenario)
        print(na)
        if na != 0:
            ua = body.get_aero_disps(scenario)
            # deform the whole mesh up by +0.01 in the z direction
            ua[2::3] += 0.01

            ua0 = ua * 1.0
            lamL = -aero_loads_ajp

            # set lamL to a random value
            lamL[:, :] = np.random.rand((3 * na), nf)[:, :]

        dtype = TransferScheme.dtype
        adj_product = None
        cmplx_product = None

        # forward analysis loads(disps)
        fun3d_ae_interface.solve_forward()

        # adjoint analysis on loads(disps), input load adjoint
        fun3d_ae_interface.solve_adjoint()
        lamD = body.get_aero_disps_ajp(scenario)

        if na != 0:
            adj_product = np.dot(lamD[:, 0], duads)
        else:
            adj_product = 0.0

        # then sum across all processes
        adj_product = fun3d_ae_interface.comm.allreduce(adj_product)

        fun3d_ae_interface = cls.copy_complex_interface(fun3d_ae_interface)

        if na != 0:
            # forward analysis loads(ua+dua/ds*h*1j)
            aero_disps = body.get_aero_disps(scenario)
            aero_disps[:] = ua0[:] + duads[:] * epsilon * 1j
        fun3d_ae_interface.solve_forward()
        f_loads = body.get_aero_loads(scenario)

        if na != 0:
            cmplx_product = np.dot(np.imag(f_loads) / epsilon, lamL[:, 0])
        else:
            cmplx_product = 0.0

        cmplx_product = fun3d_ae_interface.comm.allreduce(cmplx_product)

        rel_error = (adj_product - cmplx_product) / cmplx_product

        adj_product = adj_product.real
        cmplx_product = cmplx_product.real
        rel_error = rel_error.real

        if fun3d_ae_interface.comm.rank == 0:
            print(f"Fun3d 14.0.2 Interface AE ajp test")
            print(f"\tadj product = {adj_product}")
            print(f"\tcmplx product = {cmplx_product}")
            print(f"\trel error = {rel_error}")

        # run the complex step test
        func_name = model.get_functions()[0].name
        hdl = open(filename, "a")
        TestResult(
            name="fun3d_ae_test",
            func_names=[func_name],
            complex_TD=[cmplx_product],
            adjoint_TD=[adj_product],
            rel_error=[rel_error],
            comm=fun3d_ae_interface.comm,
            method="complex step",
        ).write(hdl)
        return abs(rel_error)
