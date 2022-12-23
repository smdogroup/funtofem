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

__all__ = ["Fun3dClient"]

import numpy as np
import os
import zmq
from mpi4py import MPI
from fun3d.mda.fsi.fun3d_aero import (
    FUN3DAero,
    FUN3DAeroException,
    SurfaceMesh,
    AeroLoads,
    AdjointProduct,
)
from funtofem import TransferScheme
from ._solver_interface import SolverInterface


class Fun3dClient(SolverInterface):
    """
    FUNtoFEM client class for FUN3D. Works for both steady and unsteady analysis.
    Requires the FUN3D directory structure.
    During the forward analysis, the FUN3D interface will operate in the scenario.name/Flow directory and scenario.name/Adjoint directory for the adjoint.

    FUN3D's FUNtoFEM coupling interface requires no additional configure flags to compile.
    To tell FUN3D that a body's motion should be driven by FUNtoFEM, set *motion_driver(i)='funtofem'*.
    """

    def __init__(self, comm, model, flow_dt=1.0, host="localhost", port_base=49200):
        """


        The instantiation of the FUN3D interface class will populate the model with the aerodynamic surface mesh, body.aero_X and body.aero_nnodes.
        The surface mesh on each processor only holds it's owned nodes. Transfer of that data to other processors is handled inside the FORTRAN side of FUN3D's FUNtoFEM interface.

        Parameters
        ----------
        comm: MPI.comm
            MPI communicator
        model: :class:`FUNtoFEMmodel`
            FUNtoFEM model. This instantiatio
        flow_dt: float
            flow solver time step size. Used to scale the adjoint term coming into and out of FUN3D since FUN3D currently uses a different adjoint formulation than FUNtoFEM.
        host: FUN3D Aero Server
        port_base: FUN3D Aero Server base port (port for rank 0)
        """

        self.comm = comm
        #  Instantiate FUN3D
        rank = MPI.COMM_WORLD.Get_rank()
        port = port_base + rank
        context = zmq.Context()
        self.fun3d_client = FUN3DAero.Client(
            context, "tcp://%s:%d" % (host, port), zmq.REQ
        )

        # Get the initial aero surface meshes
        self.initialize(model.scenarios[0], model.bodies, first_pass=True)
        self.post(model.scenarios[0], model.bodies, first_pass=True)

        # Temporary measure until FUN3D adjoint is reformulated
        self.flow_dt = flow_dt

        # multiple steady scenarios
        self.force_save = {}
        self.disps_save = {}

        # unsteady scenarios
        self.force_hist = {}
        for scenario in model.scenarios:
            self.force_hist[scenario.id] = {}

    def set_client_options(self, kwargs):
        """
        Sets client options from a dictionary of options.

        Parameters
        ----------
        kwargs: :dictionary:
            The dictionary of client options
        """
        for key in kwargs:
            if type(kwargs[key]) is int:
                self.fun3d_client.set_int32_option(key, kwargs[key])
            elif type(kwargs[key]) is float:
                self.fun3d_client.set_real64_option(key, kwargs[key])
            elif type(kwargs[key]) is bool:
                self.fun3d_client.set_bool_option(key, kwargs[key])
            else:
                print("Unknown option type")

    def initialize(self, scenario, bodies, first_pass=False):
        """
        Changes the directory to ./`scenario.name`/Flow, then
        initializes the FUN3D flow (forward) solver.

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The scenario that needs to be initialized
        bodies: :class:`~body.Body`
            list of FUNtoFEM bodies to either get new surface meshes from or to set the original mesh in
        first_pass: bool
            When extracting the mesh, first_pass is set to True. Otherwise, the new mesh will be set in for bodies with shape parameterizations

        Returns
        -------
        fail: int
            If the grid deformation failed, the intiialization will return 1
        """

        directory = scenario.name + "/Flow"
        try:
            self.fun3d_client.pushd(directory)
            # Do the steps to initialize FUN3D
            self.fun3d_client.initialize_flow_project()

            if not scenario.steady:
                options = {"timedep_adj_frozen": True}
                self.set_client_options(options)

            self.fun3d_client.initialize_flow_data()
            self.fun3d_client.initialize_design()
            self.fun3d_client.initialize_flow_grid()

            # During the first pass we don't have any meshes yet
            if not first_pass:
                for ibody, body in enumerate(bodies, 1):
                    if body.shape and body.aero_nnodes > 0:
                        aero_X = np.reshape(body.aero_X, (3, -1), order="F")
                        self.fun3d_client.set_design_surface(
                            ibody, aero_X, body.aero_id
                        )
                        self.fun3d_client.set_design_surface_name(ibody, body.name)
                    else:
                        self.fun3d_client.set_design_surface(ibody, [], [])
                        self.fun3d_client.set_design_surface_name(ibody, body.name)

            self.fun3d_client.initialize_flow_solution()

            if first_pass:
                for ibody, body in enumerate(bodies, 1):
                    body.aero_nnodes = self.fun3d_client.extract_surface_num(ibody)
                    body.aero_X = np.zeros(
                        3 * body.aero_nnodes, dtype=TransferScheme.dtype
                    )
                    if body.aero_nnodes > 0:
                        mesh = self.fun3d_client.extract_surface(
                            ibody, body.aero_nnodes
                        )
                        body.aero_id = self.fun3d_client.extract_surface_id(
                            ibody, body.aero_nnodes
                        )

                        body.aero_X[::3] = mesh.x[:]
                        body.aero_X[1::3] = mesh.y[:]
                        body.aero_X[2::3] = mesh.z[:]
                    else:
                        body.aero_id = np.zeros(3 * body.aero_nnodes, dtype=int)

                    body.rigid_transform = np.identity(4, dtype=TransferScheme.dtype)

            return 0

        except FUN3DAeroException as e:
            print("Error:", e.reason)
            try:
                self.fun3d_client.popd()
            except:
                pass
        return 1

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
        try:
            self.fun3d_client.pushd(scenario.name + "/Adjoint")
            if scenario.steady:
                # load the forces and displacements
                if scenario.steady:
                    for ibody, body in enumerate(bodies):
                        if body.aero_nnodes > 0:
                            body.aero_loads = self.force_save[scenario.id][ibody]
                            body.aero_disps = self.disps_save[scenario.id][ibody]

                # Initialize FUN3D adjoint - special order for static adjoint
                options = {"getgrad": True}
                self.fun3d_client.initialize_adjoint_project()
                self.set_client_options(options)
                self.fun3d_client.initialize_adjoint_data()
                self.fun3d_client.initialize_design()
                for ibody, body in enumerate(bodies, 1):
                    if body.shape and body.aero_nnodes > 0:
                        aero_X = np.reshape(body.aero_X, (3, -1), order="F")
                        self.fun3d_client.set_design_surface(
                            ibody, aero_X, body.aero_id
                        )
                        self.fun3d_client.set_design_surface_name(ibody, body.name)
                    else:
                        self.fun3d_client.set_design_surface(ibody, [], [])
                        self.fun3d_client.set_design_surface_name(ibody, body.name)
                self.fun3d_client.initialize_adjoint_grid()

                self.fun3d_client.set_up_moving_body()
                self.fun3d_client.initialize_funtofem_adjoint()

                # Deform the aero mesh before finishing FUN3D initialization
                if body.aero_nnodes > 0:
                    for ibody, body in enumerate(bodies, 1):
                        dx = body.aero_disps[0::3]
                        dy = body.aero_disps[1::3]
                        dz = body.aero_disps[2::3]
                        self.fun3d_client.input_deformation(ibody, dx, dy, dz)
                self.fun3d_client.initialize_adjoint_solution()
            else:
                options = {"timedep_adj_frozen": True}
                self.fun3d_client.initialize_adjoint_project()
                self.set_client_options(options)
                self.fun3d_client.initialize_adjoint_data()
                self.fun3d_client.initialize_design()
                for ibody, body in enumerate(bodies, 1):
                    if body.shape and body.aero_nnodes > 0:
                        aero_X = np.reshape(body.aero_X, (3, -1), order="F")
                        self.fun3d_client.set_design_surface(
                            ibody, aero_X, body.aero_id
                        )
                        self.fun3d_client.set_design_surface_name(ibody, body.name)
                    else:
                        self.fun3d_client.set_design_surface(ibody, [], [])
                        self.fun3d_client.set_design_surface_name(ibody, body.name)
                self.fun3d_client.initialize_adjoint_grid()
                self.fun3d_client.initialize_adjoint_solution()

            return 0

        except FUN3DAeroException as e:
            print("Error:", e.reason)
            try:
                self.fun3d_client.popd()
            except:
                pass
            return 1

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
        try:
            for function in scenario.functions:
                if function.adjoint:
                    start = 1 if function.stop == -1 else function.start
                    stop = 1 if function.stop == -1 else function.stop
                    ftype = -1 if function.averaging else 1
                    self.fun3d_client.set_design_composite_function(
                        function.id,
                        1,
                        start,
                        stop,
                        1.0,
                        0.0,
                        1.0,
                        function.value,
                        ftype,
                        100.0,
                        -100.0,
                    )

                    if function.body == -1:
                        boundary = 0
                    else:
                        boundary = bodies[function.body].boundary

                    # The funtofem function in FUN3D acts as any adjoint function
                    # that isn't dependent on FUN3D variables
                    name = (
                        function.name
                        if function.analysis_type == "aerodynamic"
                        else "funtofem"
                    )

                    self.fun3d_client.set_design_component_function(
                        function.id, 1, boundary, name, function.value, 1.0, 0.0, 1.0
                    )
        except FUN3DAeroException as e:
            print("Error:", e.reason)
            try:
                self.fun3d_client.popd()
            except:
                pass

    def set_variables(self, scenario, bodies):
        """
        Set the aerodynamic variable definitions into FUN3D using the design interface.
        FUN3D expects 6 global variables (Mach number, AOA, yaw, etc.) that are stored in the scenario.
        It also expects a set of rigid motion variables for each body that are stored in the body.
        If the body has been specific as *motion_driver(i)='funtofem'*, the rigid motion variables will not affect the body's movement but must be passed regardless.

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The scenario. Contains the global aerodynamic list
        bodies: :class:`~body.Body`
            list of FUNtoFEM bodies. Bodies contains unused but necessary rigid motion variables
        """

        try:
            self.fun3d_client.set_design(
                len(bodies), scenario.count_adjoint_functions()
            )

            # push the global aerodynamic variables to fun3d
            for var in scenario.variables["aerodynamic"]:
                self.fun3d_client.set_design_global_var(
                    var.id, var.active, var.value, var.lower, var.upper
                )

            # push the push the shape and rigid motion variables
            for ibody, body in enumerate(bodies, 1):
                num = len(body.variables["shape"]) if "shape" in body.variables else 1
                self.fun3d_client.set_body(ibody, body.parameterization, num)
                if "shape" in body.variables:
                    for var in body.variables["shape"]:
                        self.fun3d_client.set_design_shape_var(
                            ibody, var.id, var.active, var.value, var.lower, var.upper
                        )

                for var in body.variables["rigid_motion"]:
                    self.fun3d_client.set_design_rigid_var(
                        ibody,
                        var.id,
                        var.name,
                        var.active,
                        var.value,
                        var.lower,
                        var.upper,
                    )
        except FUN3DAeroException as e:
            print("Error:", e.reason)
            try:
                self.fun3d_client.popd()
            except:
                pass

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
        try:
            for function in scenario.functions:
                if function.analysis_type == "aerodynamic":
                    # the [6] index returns the value
                    if self.comm.Get_rank() == 0:
                        function.value = (
                            self.fun3d_client.get_design_composite_function(function.id)
                        )
                    function.value = self.comm.bcast(function.value, root=0)
        except FUN3DAeroException as e:
            print("Error:", e.reason)
            try:
                self.fun3d_client.popd()
            except:
                pass

    def get_function_gradients(self, scenario, bodies, offset):
        """
        Populates the FUNtoFEM model with derivatives w.r.t. aerodynamic variables

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The scenario. Contains the global aerodynamic list
        bodies: :class:`~body.Body`
            list of FUNtoFEM bodies. Bodies contains unused but necessary rigid motion variables
        offset: int
            offset of the scenario's function index w.r.t the full list of functions in the model.
        """

        try:
            for func, function in enumerate(scenario.functions):
                # Do the scenario variables first
                for vartype in scenario.variables:
                    if vartype == "aerodynamic":
                        for i, var in enumerate(scenario.variables[vartype]):
                            if var.active:
                                if function.adjoint:
                                    scenario.derivatives[vartype][offset + func][
                                        i
                                    ] = self.fun3d_client.get_design_global_derivative(
                                        function.id, var.id
                                    )
                                else:
                                    scenario.derivatives[vartype][offset + func][
                                        i
                                    ] = 0.0
                                scenario.derivatives[vartype][offset + func][
                                    i
                                ] = self.comm.bcast(
                                    scenario.derivatives[vartype][offset + func][i],
                                    root=0,
                                )

                for ibody, body in enumerate(bodies, 1):
                    for vartype in body.variables:
                        if vartype == "rigid_motion":
                            for i, var in enumerate(body.variables[vartype]):
                                if var.active:
                                    if function.adjoint:
                                        body.derivatives[vartype][offset + func][
                                            i
                                        ] = self.fun3d_client.get_design_rigid_derivative(
                                            ibody, function.id, var.id
                                        )
                                    else:
                                        body.derivatives[vartype][offset + func][
                                            i
                                        ] = 0.0
                                    scenario.derivatives[vartype][offset + func][
                                        i
                                    ] = self.comm.bcast(
                                        scenario.derivatives[vartype][offset + func][i],
                                        root=0,
                                    )

        except FUN3DAeroException as e:
            print("Error:", e.reason)
            try:
                self.fun3d_client.popd()
            except:
                pass

        return scenario, bodies

    def get_coordinate_derivatives(self, scenario, bodies, step):
        """
        Adds FUN3D's contribution to the aerodynamic surface coordinate derivatives.
        This is just the grid adjoint variable, $\lambda_G$.

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The scenario.
        bodies: :class:`~body.Body`
            list of FUNtoFEM bodies.
        step: int
            the time step number
        """

        nfunctions = scenario.count_adjoint_functions()
        try:
            for ibody, body in enumerate(bodies, 1):
                if body.shape and body.aero_nnodes > 0:
                    # Aero solver contribution = dGdxa0^T psi_G
                    body.aero_id = self.fun3d_client.extract_surface_id(
                        ibody, body.aero_nnodes
                    )

                    product = self.fun3d_client.extract_grid_adjoint_product(
                        ibody, body.aero_nnodes, nfunctions
                    )
                    product.lam_x = np.array(product.lam_x).reshape(
                        (body.aero_nnodes, -1)
                    )
                    product.lam_y = np.array(product.lam_y).reshape(
                        (body.aero_nnodes, -1)
                    )
                    product.lam_z = np.array(product.lam_z).reshape(
                        (body.aero_nnodes, -1)
                    )
                    body.aero_shape_term[::3, :nfunctions] += (
                        product.lam_x[:, :] * self.flow_dt
                    )
                    body.aero_shape_term[1::3, :nfunctions] += (
                        product.lam_y[:, :] * self.flow_dt
                    )
                    body.aero_shape_term[2::3, :nfunctions] += (
                        product.lam_z[:, :] * self.flow_dt
                    )
        except FUN3DAeroException as e:
            print("Error:", e.reason)
            try:
                self.fun3d_client.popd()
            except:
                pass

    def iterate(self, scenario, bodies, step):
        """
        Forward iteration of FUN3D.
        For the aeroelastic cases, these steps are:

        #. Get the mesh movement - the bodies' surface displacements and rigid rotations.
        #. Step forward in the grid deformationa and flow solver.
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

        # Deform aerodynamic mesh
        try:
            for ibody, body in enumerate(bodies, 1):
                if "deform" in body.motion_type and body.aero_nnodes > 0:
                    dx = np.asfortranarray(body.aero_disps[0::3])
                    dy = np.asfortranarray(body.aero_disps[1::3])
                    dz = np.asfortranarray(body.aero_disps[2::3])
                    self.fun3d_client.input_deformation(ibody, dx, dy, dz)
                if "rigid" in body.motion_type:
                    self.fun3d_client.input_rigid_transform(ibody, body.rigid_transform)

            # Take a step in FUN3D
            self.comm.Barrier()
            self.fun3d_client.iterate_flow(step)

            # Pull out the forces from FUN3D
            for ibody, body in enumerate(bodies, 1):
                if body.aero_nnodes > 0:
                    loads = self.fun3d_client.extract_forces(ibody, body.aero_nnodes)

                body.aero_loads = np.zeros(
                    3 * body.aero_nnodes, dtype=TransferScheme.dtype
                )

                if body.aero_nnodes > 0:
                    body.aero_loads[0::3] = loads.fx[:]
                    body.aero_loads[1::3] = loads.fy[:]
                    body.aero_loads[2::3] = loads.fz[:]

        except FUN3DAeroException as e:
            print("Error:", e.reason)
            try:
                self.fun3d_client.popd()
            except:
                pass
            return 1

        if not scenario.steady:
            # save this steps forces for the adjoint
            self.force_hist[scenario.id][step] = {}
            for ibody, body in enumerate(bodies, 1):
                self.force_hist[scenario.id][step][ibody] = body.aero_loads.copy()
        return 0

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
        try:
            self.fun3d_client.post_flow()
            self.fun3d_client.popd()
        except FUN3DAeroException as e:
            print("Error:", e.reason)
            try:
                self.fun3d_client.popd()
            except:
                pass

        # save the forces for multiple scenarios if steady
        if scenario.steady and not first_pass:
            self.force_save[scenario.id] = {}
            self.disps_save[scenario.id] = {}
            for ibody, body in enumerate(bodies):
                if body.aero_nnodes > 0:
                    self.force_save[scenario.id][ibody] = body.aero_loads
                    self.disps_save[scenario.id][ibody] = body.aero_disps

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
        for ibody, body in enumerate(bodies, 1):
            body.aero_loads = self.force_hist[scenario.id][step][ibody]

    def iterate_adjoint(self, scenario, bodies, step):
        """
        Adjoint iteration of FUN3D.
        For the aeroelastic cases, these steps are:

        #. Get the force adjoint from the body data structures
        #. Step in the flow and grid adjoint solvers
        #. Set the grid adjoint into the body data structures

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
        try:
            for ibody, body in enumerate(bodies, 1):
                if body.aero_nnodes > 0:

                    # Solve the force adjoint equation
                    psi_F = -body.dLdfa

                    lam_x = np.zeros((body.aero_nnodes, nfunctions))
                    lam_y = np.zeros((body.aero_nnodes, nfunctions))
                    lam_z = np.zeros((body.aero_nnodes, nfunctions))
                    for func in range(nfunctions):
                        # Set up the load integration adjoint variables for FUN3D
                        lam_x[:, func] = psi_F[0::3, func] / self.flow_dt
                        lam_y[:, func] = psi_F[1::3, func] / self.flow_dt
                        lam_z[:, func] = psi_F[2::3, func] / self.flow_dt

                    self.fun3d_client.input_force_adjoint(
                        ibody,
                        nfunctions,
                        lam_x.flatten(),
                        lam_y.flatten(),
                        lam_z.flatten(),
                    )
                    if "rigid" in body.motion_type:
                        self.fun3d_client.input_rigid_transform(
                            ibody, body.rigid_transform
                        )

            # Update the aerodynamic and grid adjoint variables (Note: step starts at 1
            # in FUN3D)
            try:
                self.fun3d_client.iterate_adjoint(rstep)
            except FUN3DAeroException as e:  # May be Terminal Condition
                if e.code != 0:
                    raise

            for ibody, body in enumerate(bodies, 1):
                # Extract dG/du_a^T psi_G from FUN3D
                if body.aero_nnodes > 0:
                    product = self.fun3d_client.extract_grid_adjoint_product(
                        ibody, body.aero_nnodes, nfunctions
                    )
                    product.lam_x = np.array(product.lam_x).reshape(
                        (body.aero_nnodes, -1)
                    )
                    product.lam_y = np.array(product.lam_y).reshape(
                        (body.aero_nnodes, -1)
                    )
                    product.lam_z = np.array(product.lam_z).reshape(
                        (body.aero_nnodes, -1)
                    )
                    for func in range(nfunctions):
                        lam_x_temp = product.lam_x[:, func] * self.flow_dt
                        lam_y_temp = product.lam_y[:, func] * self.flow_dt
                        lam_z_temp = product.lam_z[:, func] * self.flow_dt

                        lam_x_temp = lam_x_temp.reshape((-1, 1))
                        lam_y_temp = lam_y_temp.reshape((-1, 1))
                        lam_z_temp = lam_z_temp.reshape((-1, 1))
                        body.dGdua[:, func] = np.hstack(
                            (lam_x_temp, lam_y_temp, lam_z_temp)
                        ).flatten(order="c")

                if "rigid" in body.motion_type:
                    body.dGdT = (
                        self.fun3d_client.extract_rigid_adjoint_product(nfunctions)
                        * self.flow_dt
                    )

        except FUN3DAeroException as e:
            print("Error:", e.reason)
            try:
                self.fun3d_client.popd()
            except:
                pass
            fail = 1

        return fail

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
        try:
            self.fun3d_client.post_adjoint()
            self.fun3d_client.popd()
        except FUN3DAeroException as e:
            print("Error:", e.reason)
            try:
                self.fun3d_client.popd()
            except:
                pass

    def step_pre(self, scenario, bodies, step):
        try:
            self.fun3d_client.flow_step_pre(step)
        except FUN3DAeroException as e:
            print("Error:", e.reason)
            try:
                self.fun3d_client.popd()
            except:
                pass
            return 1
        return 0

    def step_solver(self, scenario, bodies, step, fsi_subiter):
        """
        Forward iteration of FUN3D.
        For the aeroelastic cases, these steps are:

        #. Get the mesh movement - the bodies' surface displacements and rigid rotations.
        #. Step forward in the grid deformationa and flow solver.
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

        try:
            # Deform aerodynamic mesh
            for ibody, body in enumerate(bodies, 1):
                if "deform" in body.motion_type and body.aero_nnodes > 0:
                    dx = body.aero_disps[0::3]
                    dy = body.aero_disps[1::3]
                    dz = body.aero_disps[2::3]
                    self.fun3d_client.input_deformation(ibody, dx, dy, dz)
                if "rigid" in body.motion_type:
                    self.fun3d_client.input_rigid_transform(ibody, body.rigid_transform)

            # Take a step in FUN3D
            self.comm.Barrier()
            self.fun3d_client.flow_step()

            # Pull out the forces from FUN3D
            for ibody, body in enumerate(bodies, 1):
                if body.aero_nnodes > 0:
                    forces.fx, forces.fy, forces.fz = self.fun3d_client.extract_forces(
                        ibody, body.aero_nnodes
                    )

                body.aero_loads = np.zeros(
                    3 * body.aero_nnodes, dtype=TransferScheme.dtype
                )

                if body.aero_nnodes > 0:
                    body.aero_loads[0::3] = forces.fx[:]
                    body.aero_loads[1::3] = forces.fy[:]
                    body.aero_loads[2::3] = forces.fz[:]

        except FUN3DAeroException as e:
            print("Error:", e.reason)
            try:
                self.fun3d_client.popd()
            except:
                pass
            return 1

        return 0

    def step_post(self, scenario, bodies, step):

        try:
            self.fun3d_client.flow_step_post(step)
        except FUN3DAeroException as e:
            print("Error:", e.reason)
            try:
                self.fun3d_client.popd()
            except:
                pass

        # save this steps forces for the adjoint
        self.force_hist[scenario.id][step] = {}
        for ibody, body in enumerate(bodies, 1):
            self.force_hist[scenario.id][step][ibody] = body.aero_loads.copy()
        return 0
