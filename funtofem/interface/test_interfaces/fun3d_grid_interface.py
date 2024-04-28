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

__all__ = ["Fun3dGridInterface", "Fun3d14GridInterface"]

import numpy as np
import os, sys, importlib
from funtofem import TransferScheme
from ..fun3d_interface import Fun3dInterface
from ..utils.test_result import TestResult


class Fun3dGridInterface(Fun3dInterface):
    """
    FUN3D Grid Deformation interface for unit testing of FUN3D grid deformation.
    Uses namelist argument funtofem_grid_test=.true. in the namelist under massoud
    &massoud_output
        funtofem_grid_test = .true.
    /

    FUN3D's FUNtoFEM coupling interface requires no additional configure flags to compile.
    To tell FUN3D that a body's motion should be driven by FUNtoFEM, set *motion_driver(i)='funtofem'*.
    """

    def __init__(
        self,
        comm,
        model,
        flow_dt=1.0,
        fun3d_dir=None,
        forward_options=None,
        adjoint_options=None,
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
        flow_dt: float
            flow solver time step size. Used to scale the adjoint term coming into and out of FUN3D since
            FUN3D currently uses a different adjoint formulation than FUNtoFEM.
        fun3d_dir: path
            path to the Flow directory of the fun3d scenario
        forward_options: dict
            list of options for FUN3D forward analysis
        adjoint_options: dict
            list of options for FUN3D adjoint analysis
        """

        # construct the super class Fun3dInterface
        super(Fun3dGridInterface, self).__init__(
            comm=comm,
            model=model,
            flow_dt=flow_dt,
            fun3d_dir=fun3d_dir,
            forward_options=forward_options,
            adjoint_options=adjoint_options,
        )

        # get the number of grid volume coordinates
        self.nvol = self.fun3d_flow.extract_num_volume_nodes()

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
                body._aero_volume_coords = {}
                body._grid_volume_ajp = {}
                body._aero_volume_coords[scenario.id] = np.zeros(
                    (3 * self.nvol), dtype=TransferScheme.dtype
                )
                nf = scenario.count_adjoint_functions()
                body._grid_volume_ajp[scenario.id] = np.zeros(
                    (3 * self.nvol, nf), dtype=TransferScheme.dtype
                )  # 1 func for now
        return

    def solve_forward(self):
        """forward grid deformation analysis of FUN3D"""
        for scenario in self.model.scenarios:
            # pre analysis setup
            super(Fun3dGridInterface, self).set_variables(scenario, self.model.bodies)
            super(Fun3dGridInterface, self).set_functions(scenario, self.model.bodies)
            super(Fun3dGridInterface, self).initialize(scenario, self.model.bodies)

            """forward analysis starts here"""
            # first input the deformation on the surface
            for ibody, body in enumerate(self.model.bodies, 1):
                aero_disps = body.get_aero_disps(scenario, time_index=0)
                aero_nnodes = body.get_num_aero_nodes()
                deform = "deform" in body.motion_type
                if deform and aero_disps is not None and aero_nnodes > 0:
                    dx = np.asfortranarray(aero_disps[0::3])
                    dy = np.asfortranarray(aero_disps[1::3])
                    dz = np.asfortranarray(aero_disps[2::3])
                    self.fun3d_flow.input_deformation(dx, dy, dz, body=ibody)

            # iterate which skips force and just does grid deformation (don't use thermal coupling here)
            self.fun3d_flow.iterate()

            # receive the deformation in the volume
            gridx, gridy, gridz = self.fun3d_flow.extract_grid_coordinates()
            grid_coords = body._aero_volume_coords[scenario.id]
            grid_coords[0::3] = gridx[:]
            grid_coords[1::3] = gridy[:]
            grid_coords[2::3] = gridz[:]

            # post analysis in fun3d interface
            super(Fun3dGridInterface, self).post(scenario, self.model.bodies)
        return

    def solve_adjoint(self):
        """adjoint grid deformation analysis in FUN3D"""
        for scenario in self.model.scenarios:
            # pre analysis setup
            super(Fun3dGridInterface, self).set_variables(scenario, self.model.bodies)
            super(Fun3dGridInterface, self).set_functions(scenario, self.model.bodies)
            for body in self.model.bodies:
                body.initialize_adjoint_variables(scenario)
            super(Fun3dGridInterface, self).initialize_adjoint(
                scenario, self.model.bodies
            )

            """adjoint analysis starts here"""
            # first input the grid volume adjoint variables lam_xG
            dtype = TransferScheme.dtype
            nf = scenario.count_adjoint_functions()
            for ibody, body in enumerate(self.model.bodies, 1):
                lam_x = np.zeros((self.nvol, nf), dtype=dtype)
                lam_y = np.zeros((self.nvol, nf), dtype=dtype)
                lam_z = np.zeros((self.nvol, nf), dtype=dtype)

                grid_volume_ajp = body._grid_volume_ajp[scenario.id]

                for func in range(nf):
                    lam_x[:, func] = grid_volume_ajp[0::3, func]
                    lam_y[:, func] = grid_volume_ajp[1::3, func]
                    lam_z[:, func] = grid_volume_ajp[2::3, func]

                self.fun3d_adjoint.input_grid_volume_adjoint(
                    lam_x, lam_y, lam_z, n=self.nvol, nfunctions=1
                )

            # run the adjoint analysis
            self.fun3d_adjoint.iterate(1)

            # extract the surface aero displacements adjoint
            for ibody, body in enumerate(self.model.bodies, 1):
                # Extract aero_disps_ajp = dG/du_A^T psi_G from FUN3D
                aero_disps_ajp = body.get_aero_disps_ajp(scenario)
                aero_nnodes = body.get_num_aero_nodes()
                if aero_disps_ajp is not None and aero_nnodes > 0:
                    (
                        lam_x,
                        lam_y,
                        lam_z,
                    ) = self.fun3d_adjoint.extract_grid_adjoint_product(
                        aero_nnodes, nf, body=ibody
                    )

                    for func in range(nf):
                        aero_disps_ajp[0::3, func] = lam_x[:, func]
                        aero_disps_ajp[1::3, func] = lam_y[:, func]
                        aero_disps_ajp[2::3, func] = lam_z[:, func]

            # call post adjoint
            super(Fun3dGridInterface, self).post_adjoint(scenario, self.model.bodies)
        return

    def input_aero_disps(self, array, body=None, scenario=None):
        """input aerodynamic displacements at the surface"""
        body = body if body is not None else self.model.bodies[0]
        scenario = scenario if scenario is not None else self.model.scenarios[0]
        aero_nodes = body.get_num_aero_nodes()
        aero_disps = body.get_aero_disps(scenario)
        if aero_nodes > 0 and aero_disps is not None:
            aero_disps[:] = array[:]
        return

    def extract_grid_coordinates(self, body=None, scenario=None):
        """extract the volume grid coordinates after deformation"""
        body = body if body is not None else self.model.bodies[0]
        scenario = scenario if scenario is not None else self.model.scenarios[0]
        return body._aero_volume_coords[scenario.id]

    def perturb_aero_disps(self, array, body=None, scenario=None):
        """input aerodynamic displacements at the surface"""
        body = body if body is not None else self.model.bodies[0]
        scenario = scenario if scenario is not None else self.model.scenarios[0]
        aero_nodes = body.get_num_aero_nodes()
        aero_disps = body.get_aero_disps(scenario)
        if aero_nodes > 0 and aero_disps is not None:
            aero_disps[:] += array[:]
        return

    def input_volume_grid_adjoint(self, array, ifunc=0, body=None, scenario=None):
        """assumes one scenario here"""
        body = body if body is not None else self.model.bodies[0]
        scenario = scenario if scenario is not None else self.model.scenarios[0]
        # extract the number of volume grid coordinates
        if self.nvol > 0:
            body._grid_volume_ajp[scenario.id][:, ifunc] = array[:]
        return

    def extract_surface_grid_adjoint(self, body=None, scenario=None):
        body = body if body is not None else self.model.bodies[0]
        scenario = scenario if scenario is not None else self.model.scenarios[0]
        aero_disps_ajp = body.get_aero_disps_ajp(scenario)
        return aero_disps_ajp

    @classmethod
    def complex_step_test(
        cls, fun3d_grid_interface, filename="fun3d_grid_test.txt", scale=0.001
    ):
        assert isinstance(fun3d_grid_interface, cls)
        # get the dimensions of surf and volume grid from first interface
        nsurf = fun3d_grid_interface.model.bodies[0].get_num_aero_nodes()
        nvol = fun3d_grid_interface.nvol

        # random real aero disps (not for perturbations)
        rand_disps = scale * np.random.rand(3 * nsurf)
        # random contravariant duA/ds test vector
        p = np.random.rand(3 * nsurf)
        # random covariant dL/dxG test vector
        q = scale * np.random.rand(3 * nvol)

        # build a real interface and do the adjoint method
        real_interface = cls.make_real_interface(fun3d_grid_interface)
        real_interface.input_aero_disps(array=rand_disps)
        real_interface.solve_forward()
        real_interface.input_volume_grid_adjoint(array=q)
        real_interface.solve_adjoint()
        surface_grid_ajp = real_interface.extract_surface_grid_adjoint()[:, 0]

        if nsurf > 0 and nvol > 0:
            local_adjoint_TD = np.zeros(1)
            local_adjoint_TD[0] = np.sum(p * surface_grid_ajp)
        else:
            local_adjoint_TD = np.zeros(1)
        # add across all procs
        comm = fun3d_grid_interface.comm
        adjoint_TD = np.zeros(1)
        comm.Reduce(local_adjoint_TD, adjoint_TD, root=0)
        adjoint_TD = comm.bcast(adjoint_TD, root=0)

        # build a complex interface and evaluate grid deformation in complex mode
        h = 1e-30
        complex_interface = cls.make_complex_interface(fun3d_grid_interface)
        pert_disps = rand_disps + p * h * 1j
        complex_interface.input_aero_disps(array=pert_disps)
        complex_interface.solve_forward()
        xG_output = complex_interface.extract_grid_coordinates()
        dxGds = np.imag(xG_output) / h
        if nsurf > 0 and nvol > 0:
            local_complex_step_TD = np.zeros(1)
            local_complex_step_TD[0] = np.sum(q * dxGds)
        else:
            local_complex_step_TD = np.zeros(1)
        complex_step_TD = np.zeros(1)

        # add across all procs
        comm.Reduce(local_complex_step_TD, complex_step_TD, root=0)
        complex_step_TD = comm.bcast(complex_step_TD, root=0)

        rel_error = (adjoint_TD[0] - complex_step_TD[0]) / complex_step_TD[0]

        # report test result
        if comm.rank == 0:
            print(f"Adjoint TD = {adjoint_TD}")
            print(f"Complex step TD = {complex_step_TD}")
            print(f"rel error = {rel_error}")

        # run the complex step test
        func_name = real_interface.model.get_functions()[0].name
        hdl = open(filename, "w")
        TestResult(
            name="fun3d_grid_deformation",
            func_names=[func_name],
            complex_TD=[complex_step_TD],
            adjoint_TD=[adjoint_TD],
            rel_error=[rel_error],
            comm=real_interface.comm,
        ).write(hdl)
        return rel_error

    @classmethod
    def make_real_interface(cls, fun3d_grid_interface):
        """
        copy used for derivative testing
        """

        # unload and reload fun3d Flow, Adjoint as real versions
        os.environ["CMPLX_MODE"] = ""
        importlib.reload(sys.modules["fun3d.interface"])

        return cls(
            comm=fun3d_grid_interface.comm,
            model=fun3d_grid_interface.model,
            fun3d_dir=fun3d_grid_interface.fun3d_dir,
        )

    @classmethod
    def make_complex_interface(cls, fun3d_grid_interface):
        """
        copy used for derivative testing
        """

        # unload and reload fun3d Flow, Adjoint as complex versions
        os.environ["CMPLX_MODE"] = "1"
        importlib.reload(sys.modules["fun3d.interface"])

        return cls(
            comm=fun3d_grid_interface.comm,
            model=fun3d_grid_interface.model,
            fun3d_dir=fun3d_grid_interface.fun3d_dir,
        )


class Fun3d14GridInterface(Fun3dInterface):
    """
    FUN3D Grid Deformation interface for unit testing of FUN3D grid deformation.
    Uses namelist argument funtofem_grid_test=.true. in the namelist under massoud
    &massoud_output
        funtofem_grid_test = .true.
    /

    FUN3D's FUNtoFEM coupling interface requires no additional configure flags to compile.
    To tell FUN3D that a body's motion should be driven by FUNtoFEM, set *motion_driver(i)='funtofem'*.
    """

    def __init__(
        self,
        comm,
        model,
        complex_mode=False,
        flow_dt=1.0,
        fun3d_dir=None,
        forward_options=None,
        adjoint_options=None,
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
        flow_dt: float
            flow solver time step size. Used to scale the adjoint term coming into and out of FUN3D since
            FUN3D currently uses a different adjoint formulation than FUNtoFEM.
        fun3d_dir: path
            path to the Flow directory of the fun3d scenario
        forward_options: dict
            list of options for FUN3D forward analysis
        adjoint_options: dict
            list of options for FUN3D adjoint analysis
        """

        # construct the super class Fun3dInterface
        super(Fun3d14GridInterface, self).__init__(
            comm=comm,
            model=model,
            complex_mode=complex_mode,
            flow_dt=flow_dt,
            fun3d_dir=fun3d_dir,
            forward_options=forward_options,
            adjoint_options=adjoint_options,
        )

        # get the number of grid volume coordinates
        self.nvol = self.fun3d_flow.extract_num_volume_nodes()

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
                body._aero_volume_coords = {}
                body._grid_volume_ajp = {}
                body._aero_volume_coords[scenario.id] = np.zeros(
                    (3 * self.nvol), dtype=TransferScheme.dtype
                )
                nf = scenario.count_adjoint_functions()
                body._grid_volume_ajp[scenario.id] = np.zeros(
                    (3 * self.nvol, nf), dtype=TransferScheme.dtype
                )  # 1 func for now
        return

    def solve_forward(self):
        """forward grid deformation analysis of FUN3D"""
        for scenario in self.model.scenarios:
            # pre analysis setup
            super(Fun3d14GridInterface, self).set_variables(scenario, self.model.bodies)
            super(Fun3d14GridInterface, self).set_functions(scenario, self.model.bodies)
            super(Fun3d14GridInterface, self).initialize(scenario, self.model.bodies)

            """forward analysis starts here"""
            # first input the deformation on the surface
            for ibody, body in enumerate(self.model.bodies, 1):
                aero_disps = body.get_aero_disps(scenario, time_index=0)
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
            self.fun3d_flow.iterate()

            # receive the deformation in the volume
            gridx, gridy, gridz = self.fun3d_flow.extract_grid_coordinates()
            grid_coords = body._aero_volume_coords[scenario.id]
            grid_coords[0::3] = gridx[:]
            grid_coords[1::3] = gridy[:]
            grid_coords[2::3] = gridz[:]

            # post analysis in fun3d interface
            super(Fun3d14GridInterface, self).post(scenario, self.model.bodies)
        return

    def solve_adjoint(self):
        """adjoint grid deformation analysis in FUN3D"""
        for scenario in self.model.scenarios:
            # pre analysis setup
            super(Fun3d14GridInterface, self).set_variables(scenario, self.model.bodies)
            super(Fun3d14GridInterface, self).set_functions(scenario, self.model.bodies)
            for body in self.model.bodies:
                body.initialize_adjoint_variables(scenario)
            super(Fun3d14GridInterface, self).initialize_adjoint(
                scenario, self.model.bodies
            )

            """adjoint analysis starts here"""
            # first input the grid volume adjoint variables lam_xG
            dtype = TransferScheme.dtype
            nf = scenario.count_adjoint_functions()
            for ibody, body in enumerate(self.model.bodies, 1):
                lam_x = np.zeros((self.nvol, nf), dtype=dtype)
                lam_y = np.zeros((self.nvol, nf), dtype=dtype)
                lam_z = np.zeros((self.nvol, nf), dtype=dtype)

                grid_volume_ajp = body._grid_volume_ajp[scenario.id]

                for func in range(nf):
                    lam_x[:, func] = grid_volume_ajp[0::3, func]
                    lam_y[:, func] = grid_volume_ajp[1::3, func]
                    lam_z[:, func] = grid_volume_ajp[2::3, func]

                if not self.complex_mode:
                    lam_x = lam_x.astype(np.double)
                    lam_y = lam_y.astype(np.double)
                    lam_z = lam_z.astype(np.double)

                lam_x = np.asfortranarray(lam_x)
                lam_y = np.asfortranarray(lam_y)
                lam_z = np.asfortranarray(lam_z)

                self.fun3d_adjoint.input_grid_volume_adjoint(
                    lam_x, lam_y, lam_z, n=self.nvol, nfunctions=1
                )

            # run the adjoint analysis
            self.fun3d_adjoint.iterate(1)

            # extract the surface aero displacements adjoint
            for ibody, body in enumerate(self.model.bodies, 1):
                # Extract aero_disps_ajp = dG/du_A^T psi_G from FUN3D
                aero_disps_ajp = body.get_aero_disps_ajp(scenario)
                aero_nnodes = body.get_num_aero_nodes()
                if aero_disps_ajp is not None and aero_nnodes > 0:
                    (
                        lam_x,
                        lam_y,
                        lam_z,
                    ) = self.fun3d_adjoint.extract_grid_adjoint_product(
                        aero_nnodes, nf, body=ibody
                    )

                    for func in range(nf):
                        aero_disps_ajp[0::3, func] = lam_x[:, func] * scenario.flow_dt
                        aero_disps_ajp[1::3, func] = lam_y[:, func] * scenario.flow_dt
                        aero_disps_ajp[2::3, func] = lam_z[:, func] * scenario.flow_dt

            # call post adjoint
            super(Fun3d14GridInterface, self).post_adjoint(scenario, self.model.bodies)
        return

    def input_aero_disps(self, array, body=None, scenario=None):
        """input aerodynamic displacements at the surface"""
        body = body if body is not None else self.model.bodies[0]
        scenario = scenario if scenario is not None else self.model.scenarios[0]
        aero_nodes = body.get_num_aero_nodes()
        aero_disps = body.get_aero_disps(scenario)
        if aero_nodes > 0 and aero_disps is not None:
            aero_disps[:] = array[:]
        return

    def extract_grid_coordinates(self, body=None, scenario=None):
        """extract the volume grid coordinates after deformation"""
        body = body if body is not None else self.model.bodies[0]
        scenario = scenario if scenario is not None else self.model.scenarios[0]
        return body._aero_volume_coords[scenario.id]

    def perturb_aero_disps(self, array, body=None, scenario=None):
        """input aerodynamic displacements at the surface"""
        body = body if body is not None else self.model.bodies[0]
        scenario = scenario if scenario is not None else self.model.scenarios[0]
        aero_nodes = body.get_num_aero_nodes()
        aero_disps = body.get_aero_disps(scenario)
        if aero_nodes > 0 and aero_disps is not None:
            aero_disps[:] += array[:]
        return

    def input_volume_grid_adjoint(self, array, ifunc=0, body=None, scenario=None):
        """assumes one scenario here"""
        body = body if body is not None else self.model.bodies[0]
        scenario = scenario if scenario is not None else self.model.scenarios[0]
        # extract the number of volume grid coordinates
        if self.nvol > 0:
            body._grid_volume_ajp[scenario.id][:, ifunc] = array[:]
        return

    def extract_surface_grid_adjoint(self, body=None, scenario=None):
        body = body if body is not None else self.model.bodies[0]
        scenario = scenario if scenario is not None else self.model.scenarios[0]
        aero_disps_ajp = body.get_aero_disps_ajp(scenario)
        return aero_disps_ajp

    @classmethod
    def finite_diff_test(
        cls,
        fun3d_grid_interface,
        filename="fun3d_14_grid_deformation.txt",
        scale=0.001,
        epsilon=1e-4,
    ):
        assert isinstance(fun3d_grid_interface, cls)
        # get the dimensions of surf and volume grid from first interface
        nsurf = fun3d_grid_interface.model.bodies[0].get_num_aero_nodes()
        nvol = fun3d_grid_interface.nvol

        # random real aero disps (not for perturbations)
        rand_disps = scale * np.random.rand(3 * nsurf)
        # random contravariant duA/ds test vector
        p = np.random.rand(3 * nsurf)
        # random covariant dL/dxG test vector
        q = scale * np.random.rand(3 * nvol)

        # build a real interface and do the adjoint method
        fun3d_grid_interface.input_aero_disps(array=rand_disps)
        fun3d_grid_interface.solve_forward()
        fun3d_grid_interface.input_volume_grid_adjoint(array=q)
        fun3d_grid_interface.solve_adjoint()
        surface_grid_ajp = fun3d_grid_interface.extract_surface_grid_adjoint()[:, 0]

        if nsurf > 0 and nvol > 0:
            local_adjoint_TD = np.zeros(1)
            local_adjoint_TD[0] = np.sum(p * surface_grid_ajp)
        else:
            local_adjoint_TD = np.zeros(1)
        # add across all procs
        comm = fun3d_grid_interface.comm
        adjoint_TD = np.zeros(1)
        comm.Reduce(local_adjoint_TD, adjoint_TD, root=0)
        adjoint_TD = comm.bcast(adjoint_TD, root=0)

        # compute f(x+ph)
        h = epsilon * 1.0
        pert_disps = rand_disps + p * h
        fun3d_grid_interface.input_aero_disps(array=pert_disps)
        fun3d_grid_interface.solve_forward()
        R_xG_output = fun3d_grid_interface.extract_grid_coordinates()

        # compute f(x-ph)
        pert_disps = rand_disps - p * h
        fun3d_grid_interface.input_aero_disps(array=pert_disps)
        fun3d_grid_interface.solve_forward()
        L_xG_output = fun3d_grid_interface.extract_grid_coordinates()

        dxGds = (R_xG_output - L_xG_output) / 2.0 / h
        if nsurf > 0 and nvol > 0:
            local_FD_TD = np.zeros(1)
            local_FD_TD[0] = np.sum(q * dxGds)
        else:
            local_FD_TD = np.zeros(1)
        FD_TD = np.zeros(1)

        # add across all procs
        comm.Reduce(local_FD_TD, FD_TD, root=0)
        FD_TD = comm.bcast(FD_TD, root=0)

        rel_error = (adjoint_TD[0] - FD_TD[0]) / FD_TD[0]

        # report test result
        if comm.rank == 0:
            print(f"Adjoint TD = {adjoint_TD}")
            print(f"Finite diff step TD = {FD_TD}")
            print(f"rel error = {rel_error}")

        # run the complex step test
        func_name = fun3d_grid_interface.model.get_functions()[0].name
        hdl = open(filename, "w")
        TestResult(
            name="fun3d_grid_deformation",
            func_names=[func_name],
            complex_TD=[FD_TD],
            adjoint_TD=[adjoint_TD],
            rel_error=[rel_error],
            comm=fun3d_grid_interface.comm,
        ).write(hdl)
        return rel_error
