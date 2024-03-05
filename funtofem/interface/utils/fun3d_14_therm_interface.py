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

__all__ = ["Fun3dThermalInterface"]

import numpy as np
from funtofem import TransferScheme
from ..fun3d_14_interface import Fun3d14Interface
from funtofem.interface.test_solver import TestResult


class Fun3dThermalInterface(Fun3d14Interface):
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
        flow_dt=1.0,
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
        super(Fun3dThermalInterface, self).__init__(
            comm=comm,
            model=model,
            flow_dt=flow_dt,
            fun3d_dir=fun3d_dir,
            forward_options=forward_options,
            adjoint_options=adjoint_options,
            complex_mode=complex_mode
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
            super(Fun3dThermalInterface, self).set_variables(scenario, self.model.bodies)
            super(Fun3dThermalInterface, self).set_functions(scenario, self.model.bodies)
            super(Fun3dThermalInterface, self).initialize(scenario, self.model.bodies)

            """forward analysis starts here"""
            # first input the deformation on the surface
            for step in range(scenario.steps):
                for ibody, body in enumerate(self.model.bodies, 1):
                    aero_nnodes = body.get_num_aero_nodes()
                    aero_temps = body.get_aero_temps(scenario, time_index=step)
                    if aero_temps is not None and aero_nnodes > 0:
                        # Nondimensionalize by freestream temperature
                        temps = np.asfortranarray(aero_temps[:]) / scenario.T_inf
                        temps = temps if self.complex_mode else temps.astype(np.double)
                        self.fun3d_flow.input_wall_temperature(temps, body=ibody)

                # iterate which skips force and just does grid deformation (don't use thermal coupling here)
                self.comm.Barrier()
                self.fun3d_flow.iterate()

                for ibody, body in enumerate(self.model.bodies, 1):
                    heat_flux = body.get_aero_heat_flux(scenario, time_index=step)

                    if heat_flux is not None and aero_nnodes > 0:
                        # Extract the area-weighted temperature gradient normal to the wall (along the unit norm)
                        cqa = self.fun3d_flow.extract_cqa(aero_nnodes, body=ibody)
                        heat_flux[:] = cqa[:]

            # post analysis in fun3d interface
            super(Fun3dThermalInterface, self).post(scenario, self.model.bodies)
        return

    def solve_adjoint(self):
        """adjoint grid deformation analysis in FUN3D"""
        for scenario in self.model.scenarios:
            # pre analysis setup
            super(Fun3dThermalInterface, self).set_variables(scenario, self.model.bodies)
            super(Fun3dThermalInterface, self).set_functions(scenario, self.model.bodies)
            # for body in self.model.bodies:
            #     body.initialize_adjoint_variables(scenario)
            super(Fun3dThermalInterface, self).initialize_adjoint(
                scenario, self.model.bodies
            )

            """adjoint analysis starts here"""
            # first input the grid volume adjoint variables lam_xG
            dtype = TransferScheme.dtype
            nf = scenario.count_adjoint_functions()

            for step in range(scenario.steps):

                for ibody, body in enumerate(self.model.bodies, 1):
                    # Get the adjoint Jacobian products for the aero heat flux
                    aero_flux_ajp = body.get_aero_heat_flux_ajp(scenario)
                    aero_nnodes = body.get_num_aero_nodes()
                    aero_flux = body.get_aero_heat_flux(scenario, time_index=step)
                    aero_temps = body.get_aero_temps(scenario, time_index=step)

                    if aero_flux_ajp is not None and aero_nnodes > 0:
                        # Solve the aero heat flux integration adjoint
                        lam = body._lamH

                        if not self.complex_mode:
                            lam = lam.astype(np.double)

                        lam = np.asfortranarray(lam)

                        self.fun3d_adjoint.input_cqa_adjoint(lam, body=ibody)
                # run the adjoint analysis
                self.fun3d_adjoint.iterate(1)

                # extract the surface aero displacements adjoint
                for ibody, body in enumerate(self.model.bodies, 1):
                    # Extract aero_temps_ajp = dA/dt_A^{T} * psi_A from FUN3D
                    aero_temps_ajp = body.get_aero_temps_ajp(scenario)
                    aero_nnodes = body.get_num_aero_nodes()

                    if aero_temps_ajp is not None and aero_nnodes > 0:
                        lam_t = self.fun3d_adjoint.extract_thermal_adjoint_product(
                            aero_nnodes, nf, body=ibody
                        )

                        scale = scenario.flow_dt / scenario.T_inf
                        for func in range(nf):
                            aero_temps_ajp[:, func] = scale * lam_t[:, func]


            # call post adjoint
            super(Fun3dThermalInterface, self).post_adjoint(scenario, self.model.bodies)
        return

    @classmethod
    def finite_diff_test(
        cls, fun3d_therm_interface, filename="fun3d_therm_test.txt"
    ):
        assert isinstance(fun3d_therm_interface, cls)
        model = fun3d_therm_interface.model
        body = model.bodies[0]
        scenario = model.scenarios[0]
        na = body.get_num_aero_nodes()
        nf = scenario.count_adjoint_functions()

        temp0 = body.get_aero_temps(scenario) * 1.0
        dTds = np.random.rand(na)
        aero_flux_ajp = body.get_aero_heat_flux_ajp(scenario)
        psi_H = -aero_flux_ajp

        # new viscosity law effect
        k_dim = scenario.get_thermal_conduct(temp0)

        dtype = TransferScheme.dtype
        lamH = np.zeros((na, nf), dtype=dtype)

        scale = scenario.T_inf / scenario.flow_dt

        for func in range(nf):
            lamH[:, func] = scale * psi_H[:, func] * k_dim[:]

        if not fun3d_therm_interface.complex_mode:
            lamH = lamH.astype(np.double)

        lamH = 0.1 * (np.ones(na) + 0.001 *np.random.rand(na))
        
        _lamH = np.reshape(lamH, newshape=(na,1))
        _lamH = np.asfortranarray(_lamH)
        body._lamH = _lamH

        adj_product = None
        fd_product = 0.0
        epsilon = 1e-4
        test_steps = scenario.steps

        # forward analysis h(T)
        fun3d_therm_interface.solve_forward()

        # adjoint analysis on h(T), input lam_H
        fun3d_therm_interface.solve_adjoint()
        lamT = body.get_aero_temps_ajp(scenario)

        adj_product = np.dot(lamT[:,0], dTds)

        # forward analysis h(T+dT/ds*eps)
        aero_temps = body.get_aero_temps(scenario)
        aero_temps[:] = temp0[:] + dTds[:]*epsilon
        fun3d_therm_interface.solve_forward()
        cqaR = body.get_aero_heat_flux(scenario)

        fd_product += np.dot(cqaR, lamH[:,0]) / epsilon

        # forward analysis h(T-dT/ds*eps)
        aero_temps = body.get_aero_temps(scenario)
        aero_temps[:] = temp0[:] - dTds[:]*epsilon
        fun3d_therm_interface.solve_forward()
        cqaL = body.get_aero_heat_flux(scenario)

        fd_product -= np.dot(cqaL, lamH[:,0]) / epsilon

        rel_error = (adj_product - fd_product) / fd_product

        print(f"Fun3d 14 Interface aerothermal ajp test")
        print(f"\tadj product = {adj_product}")
        print(f"\tfd product = {fd_product}")
        print(f"\trel error = {rel_error}")

        # run the complex step test
        func_name = model.get_functions()[0].name
        hdl = open(filename, "w")
        TestResult(
            name="fun3d_therm_test",
            func_names=[func_name],
            complex_TD=[fd_product],
            adjoint_TD=[adj_product],
            rel_error=[rel_error],
            comm=fun3d_therm_interface.comm,
            method="finite_diff"
        ).write(hdl)
        return rel_error
