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
import os
from fun3d.solvers      import Flow, Adjoint
from fun3d              import interface
from funtofem           import TransferScheme
from .solver_interface   import SolverInterface

class Fun3dInterface(SolverInterface):
    """
    FUNtoFEM interface class for FUN3D. Works for both steady and unsteady analysis.
    Requires the FUN3D directory structure.
    During the forward analysis, the FUN3D interface will operate in the scenario.name/Flow directory and scenario.name/Adjoint directory for the adjoint.

    FUN3D's FUNtoFEM coupling interface requires no additional configure flags to compile.
    To tell FUN3D that a body's motion should be driven by FUNtoFEM, set *motion_driver(i)='funtofem'*.
    """
    def __init__(self, comm, model, flow_dt=1.0, qinf=1.0,thermal_scale=1.0,
                 fun3d_dir=None, forward_options=None, adjoint_options=None):
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
        """

        self.comm = comm

        #  Instantiate FUN3D
        self.fun3d_flow = Flow()
        self.fun3d_adjoint = Adjoint()

        # Root and FUN3D directories
        self.root_dir = os.getcwd()
        if (fun3d_dir is None):
            self.fun3d_dir = self.root_dir
        else:
            self.fun3d_dir = fun3d_dir

        # command line options
        self.forward_options = forward_options
        self.adjoint_options = adjoint_options

        # Get the initial aero surface meshes
        self.initialize(model.scenarios[0], model.bodies, first_pass=True)
        self.post(model.scenarios[0], model.bodies, first_pass=True)

        # Temporary measure until FUN3D adjoint is reformulated
        self.flow_dt = flow_dt

        # dynamic pressure
        self.qinf = qinf
        self.dFdqinf = []

        # heat flux
        self.thermal_scale = thermal_scale # = 1/2 * rho_inf * (V_inf)^3
        self.dHdq = []

        # multiple steady scenarios
        self.force_save = {}
        self.disps_save = {}
        self.heat_flux_save = {}
        self.heat_flux_mag_save = {}
        self.temps_save = {}

        # unsteady scenarios
        self.force_hist = {}
        self.heat_flux_hist = {}
        self.heat_flux_mag_hist = {}
        self.aero_temps_hist = {}
        for scenario in model.scenarios:
            self.force_hist[scenario.id] = {}
            self.heat_flux_hist[scenario.id] = {}
            self.heat_flux_mag_hist[scenario.id] = {}
            self.aero_temps_hist[scenario.id] = {}

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

        flow_dir = os.path.join(self.fun3d_dir, scenario.name, "Flow")
        os.chdir(flow_dir)

        # Do the steps to initialize FUN3D
        self.fun3d_flow.initialize_project(comm=self.comm)

        if self.forward_options is None:
            options = {}
        else:
            options = self.forward_options
        self.fun3d_flow.setOptions(kwargs=options)

        self.fun3d_flow.initialize_data()
        interface.design_initialize()
        self.fun3d_flow.initialize_grid()

        # During the first pass we don't have any meshes yet
        if not first_pass:
            for ibody, body in enumerate(bodies, 1):
                if body.aero_nnodes > 0:
                    aero_X = np.reshape(body.aero_X, (3,-1), order='F')
                    interface.design_push_body_mesh(ibody, aero_X, body.aero_id)
                    interface.design_push_body_name(ibody, body.name)
                else:
                    interface.design_push_body_mesh(ibody, [], [])
                    interface.design_push_body_name(ibody, body.name)

        bcont = self.fun3d_flow.initialize_solution()

        if bcont == 0:
            if self.comm.Get_rank()==0:
                print("Negative volume returning fail")
            return 1

        if first_pass:
            for ibody, body in enumerate(bodies, 1):
                body.aero_nnodes = self.fun3d_flow.extract_surface_num(body=ibody)
                body.aero_X = np.zeros(3*body.aero_nnodes, dtype=TransferScheme.dtype)
                if body.aero_nnodes > 0:
                    x, y, z = self.fun3d_flow.extract_surface(body.aero_nnodes, body=ibody)
                    body.aero_id = self.fun3d_flow.extract_surface_id(body.aero_nnodes, body=ibody)

                    body.aero_X[ ::3] = x[:]
                    body.aero_X[1::3] = y[:]
                    body.aero_X[2::3] = z[:]
                else:
                    body.aero_id = np.zeros(3*body.aero_nnodes, dtype=int)

                body.rigid_transform = np.identity(4, dtype=TransferScheme.dtype)

        return 0

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

        adjoint_dir = os.path.join(self.fun3d_dir, scenario.name, "Adjoint")
        os.chdir(adjoint_dir)
        
        if scenario.steady:
            # load the forces and displacements
            for ibody, body in enumerate(bodies, 1):
                if body.transfer is not None:
                    body.aero_loads = self.force_save[scenario.id][ibody]
                    body.aero_disps = self.disps_save[scenario.id][ibody]

                if body.thermal_transfer is not None:
                    body.aero_heat_flux = self.heat_flux_save[scenario.id][ibody]
                    body.aero_heat_flux_mag = self.heat_flux_mag_save[scenario.id][ibody]
                    body.aero_temps = self.temps_save[scenario.id][ibody]

            # Initialize FUN3D adjoint - special order for static adjoint
            if self.adjoint_options is None:
                options = {'getgrad': True}
            else:
                options = self.adjoint_options
            self.fun3d_adjoint.initialize_project(comm=self.comm)
            self.fun3d_adjoint.setOptions(kwargs=options)
            self.fun3d_adjoint.initialize_data()
            interface.design_initialize()
            for ibody, body in enumerate(bodies, 1):
                if body.aero_nnodes > 0:
                    aero_X = np.reshape(body.aero_X, (3,-1), order='F')
                    interface.design_push_body_mesh(ibody, aero_X, body.aero_id)
                    interface.design_push_body_name(ibody, body.name)
                else:
                    interface.design_push_body_mesh(ibody, [], [])
                    interface.design_push_body_name(ibody, body.name)
            self.fun3d_adjoint.initialize_grid()
            self.fun3d_adjoint.set_up_moving_body()
            self.fun3d_adjoint.initialize_funtofem_adjoint()

            # Deform the aero mesh before finishing FUN3D initialization
            if body.aero_nnodes > 0:
                for ibody, body in enumerate(bodies, 1):
                    if body.transfer is not None:
                        dx = np.asfortranarray(body.aero_disps[0::3])
                        dy = np.asfortranarray(body.aero_disps[1::3])
                        dz = np.asfortranarray(body.aero_disps[2::3])
                        self.fun3d_adjoint.input_deformation(dx, dy, dz, body=ibody)
                    if body.thermal_transfer is not None:
                        temps = np.asfortranarray(body.aero_temps[:])/body.T_ref
                        self.fun3d_adjoint.input_wall_temperature(temps, body=ibody)

            self.fun3d_adjoint.initialize_solution()
        else:
            if self.adjoint_options is None:
                options = {'timedep_adj_frozen': True}
            else:
                options = self.adjoint_options
            self.fun3d_adjoint.initialize_project(comm=self.comm)
            self.fun3d_adjoint.setOptions(kwargs=options)
            self.fun3d_adjoint.initialize_data()
            interface.design_initialize()
            for ibody, body in enumerate(bodies, 1):
                if body.aero_nnodes > 0:
                    aero_X = np.reshape(body.aero_X, (3,-1), order='F')
                    interface.design_push_body_mesh(ibody, aero_X, body.aero_id)
                    interface.design_push_body_name(ibody, body.name)
                else:
                    interface.design_push_body_mesh(ibody, [], [])
                    interface.design_push_body_name(ibody, body.name)
            self.fun3d_adjoint.initialize_grid()
            self.fun3d_adjoint.initialize_solution()

        self.dFdqinf = np.zeros(len(scenario.functions), dtype=TransferScheme.dtype)
        self.dHdq = np.zeros(len(scenario.functions), dtype=TransferScheme.dtype)

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
        for function in scenario.functions:
            if function.adjoint:
                start =  1 if function.stop==-1 else function.start
                stop  =  1 if function.stop==-1 else function.stop
                ftype = -1 if function.averaging else 1
                interface.design_push_composite_func(function.id,
                                                     1,
                                                     start,
                                                     stop,
                                                     1.0,
                                                     0.0,
                                                     1.0,
                                                     function.value,
                                                     ftype,
                                                     100.0,
                                                     -100.0)

                if function.body ==-1:
                    boundary = 0
                else:
                    boundary = bodies[function.body].boundary

                # The funtofem function in FUN3D acts as any adjoint function
                # that isn't dependent on FUN3D variables
                name = function.name if function.analysis_type == 'aerodynamic' else 'funtofem'

                interface.design_push_component_func(function.id,
                                                     1,
                                                     boundary,
                                                     name,
                                                     function.value,
                                                     1.0,
                                                     0.0,
                                                     1.0)

        return

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

        interface.design_set_design(len(bodies), scenario.count_adjoint_functions())

        # push the global aerodynamic variables to fun3d
        for var in scenario.variables['aerodynamic']:
            if var.id <= 6:
                interface.design_push_global_var(var.id, var.active, var.value,
                                                 var.lower, var.upper)
            elif 'dynamic pressure' == var.name.lower():
                self.qinf = var.value
            elif 'thermal scale' == var.name.lower():
                self.thermal_scale = var.value

        # push the push the shape and rigid motion variables
        for ibody, body in enumerate(bodies, 1):
            num = len(body.variables['shape']) if 'shape' in body.variables else 1
            interface.design_set_body(ibody, body.parameterization, num)
            if 'shape' in body.variables:
                for var in body.variables['shape']:
                    interface.design_push_body_shape_var(ibody, var.id, var.active,
                                                         var.value, var.lower, var.upper)

            for var in body.variables['rigid_motion']:
                interface.design_push_body_rigid_var(ibody, var.id, var.name, var.active,
                                                     var.value, var.lower, var.upper)

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
            if function.analysis_type == 'aerodynamic':
                # the [6] index returns the value
                if self.comm.Get_rank() == 0:
                    function.value = interface.design_pull_composite_func(function.id)[6]
                function.value = self.comm.bcast(function.value, root=0)

        return

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

        for func, function in enumerate(scenario.functions):
            # Do the scenario variables first
            for vartype in scenario.variables:
                if vartype == 'aerodynamic':
                    for i, var in enumerate(scenario.variables[vartype]):
                        if var.active:
                            if function.adjoint:
                                if var.id <= 6:
                                    scenario.derivatives[vartype][offset+func][i] = interface.design_pull_global_derivative(function.id,var.id)
                                elif var.name.lower() == 'dynamic pressure':
                                    scenario.derivatives[vartype][offset+func][i] = self.comm.reduce(self.dFdqinf[func])
                                    scenario.derivatives[vartype][offset+func][i] = self.comm.reduce(self.dHdq[func])
                            else:
                                scenario.derivatives[vartype][offset+func][i] = 0.0
                            scenario.derivatives[vartype][offset+func][i] = self.comm.bcast(scenario.derivatives[vartype][offset+func][i],root=0)


            for body in bodies:
                for vartype in body.variables:
                    if vartype == 'rigid_motion':
                        for i, var in enumerate(body.variables[vartype]):
                            if var.active:
                                # rigid motion variables are not active in funtofem path
                                body.derivatives[vartype][offset+func][i] = 0.0

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
        for ibody, body in enumerate(bodies, 1):
            if body.aero_nnodes > 0:
                # Aero solver contribution = dGdxa0^T psi_G
                body.aero_id = self.fun3d_adjoint.extract_surface_id(body.aero_nnodes, body=ibody)

                dGdxa0_x, dGdxa0_y, dGdxa0_z = self.fun3d_adjoint.extract_grid_adjoint_product(body.aero_nnodes,
                                                                                               nfunctions,body=ibody)

                body.aero_shape_term[ ::3,:nfunctions] += dGdxa0_x[:,:] * self.flow_dt
                body.aero_shape_term[1::3,:nfunctions] += dGdxa0_y[:,:] * self.flow_dt
                body.aero_shape_term[2::3,:nfunctions] += dGdxa0_z[:,:] * self.flow_dt

        return

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
        for ibody, body in enumerate(bodies, 1):
            if 'deform' in body.motion_type and body.aero_nnodes > 0 and body.transfer is not None:
                dx = np.asfortranarray(body.aero_disps[0::3])
                dy = np.asfortranarray(body.aero_disps[1::3])
                dz = np.asfortranarray(body.aero_disps[2::3])

                self.fun3d_flow.input_deformation(dx, dy, dz, body=ibody)
            if 'rigid' in body.motion_type and body.transfer is not None:
                transform = np.asfortranarray(body.rigid_transform)
                self.fun3d_flow.input_rigid_transform(transform,body=ibody)
            if body.thermal_transfer is not None and body.aero_nnodes > 0:
                temps = np.asfortranarray(body.aero_temps[:])/body.T_ref
                self.fun3d_flow.input_wall_temperature(temps, body=ibody)

        # Take a step in FUN3D
        self.comm.Barrier()
        bcont = self.fun3d_flow.iterate()
        if bcont == 0:
            if self.comm.Get_rank() == 0:
                print("Negative volume returning fail")
            fail = 1
            os.chdir(self.root_dir)
            return fail

        # Pull out the forces from FUN3D
        for ibody, body in enumerate(bodies, 1):
            if body.transfer is not None:
                body.aero_loads = np.zeros(3*body.aero_nnodes, dtype=TransferScheme.dtype)

            if body.thermal_transfer is not None:
                body.aero_heat_flux = np.zeros(3*body.aero_nnodes, dtype=TransferScheme.dtype)
                body.aero_heat_flux_mag = np.zeros(body.aero_nnodes, dtype=TransferScheme.dtype)

            if body.aero_nnodes > 0:
                if body.transfer is not None:
                    fx, fy, fz = self.fun3d_flow.extract_forces(body.aero_nnodes, body=ibody)

                    body.aero_loads[0::3] = self.qinf * fx[:]
                    body.aero_loads[1::3] = self.qinf * fy[:]
                    body.aero_loads[2::3] = self.qinf * fz[:]

                if body.thermal_transfer is not None:
                    cqx, cqy, cqz, cq_mag = self.fun3d_flow.extract_heat_flux(body.aero_nnodes,
                                                                              body=ibody)

                    body.aero_heat_flux[0::3] = self.thermal_scale * cqx[:]
                    body.aero_heat_flux[1::3] = self.thermal_scale * cqy[:]
                    body.aero_heat_flux[2::3] = self.thermal_scale * cqz[:]
                    body.aero_heat_flux_mag[:] = self.thermal_scale * cq_mag[:]

        if not scenario.steady:
            # save this steps forces for the adjoint
            self.force_hist[scenario.id][step] = {}
            self.heat_flux_hist[scenario.id][step] = {}
            self.heat_flux_mag_hist[scenario.id][step] = {}
            self.aero_temps_hist[scenario.id][step] = {}
            for ibody, body in enumerate(bodies, 1):
                if body.transfer is not None:
                    self.force_hist[scenario.id][step][ibody] = body.aero_loads.copy()
                if body.thermal_transfer is not None:
                    self.heat_flux_hist[scenario.id][step][ibody] = body.aero_heat_flux.copy()
                    self.heat_flux_mag_hist[scenario.id][step][ibody] = body.aero_heat_flux_mag.copy()
                    self.aero_temps_hist[scenario.id][step][ibody] = body.aero_temps.copy()
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

        self.fun3d_flow.post()
        os.chdir(self.root_dir)

        # save the forces for multiple scenarios if steady
        if scenario.steady and not first_pass:
            self.force_save[scenario.id] = {}
            self.disps_save[scenario.id] = {}
            self.heat_flux_save[scenario.id] = {}
            self.heat_flux_mag_save[scenario.id] = {}
            self.temps_save[scenario.id] = {}

            for ibody, body in enumerate(bodies, 1):
                if body.transfer is not None:
                    self.force_save[scenario.id][ibody] = body.aero_loads
                    self.disps_save[scenario.id][ibody] = body.aero_disps
                if body.thermal_transfer is not None:
                    self.heat_flux_save[scenario.id][ibody] = body.aero_heat_flux
                    self.heat_flux_mag_save[scenario.id][ibody] = body.aero_heat_flux_mag
                    self.temps_save[scenario.id][ibody] = body.aero_temps

        return

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
            if body.transfer is not None:
                body.aero_loads = self.force_hist[scenario.id][step][ibody]
            if body.thermal_transfer is not None:
                body.aero_heat_flux = self.heat_flux_hist[scenario.id][step][ibody]
                body.aero_heat_flux_mag = self.heat_flux_hist_mag[scenario.id][step][ibody]
                body.aero_temps = self.aero_temps_hist[scenario.id][step][ibody]

        return

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
        for ibody, body in enumerate(bodies, 1):
            if body.aero_nnodes > 0:
                # Solve the force adjoint equation
                if body.transfer is not None:
                    psi_F = - body.dLdfa

                    lam_x = np.zeros((body.aero_nnodes, nfunctions),
                                     dtype=TransferScheme.dtype)
                    lam_y = np.zeros((body.aero_nnodes, nfunctions),
                                     dtype=TransferScheme.dtype)
                    lam_z = np.zeros((body.aero_nnodes, nfunctions),
                                     dtype=TransferScheme.dtype)

                    for func in range(nfunctions):
                        lam_x[:, func] = self.qinf * psi_F[0::3, func]/self.flow_dt
                        lam_y[:, func] = self.qinf * psi_F[1::3, func]/self.flow_dt
                        lam_z[:, func] = self.qinf * psi_F[2::3, func]/self.flow_dt

                    self.fun3d_adjoint.input_force_adjoint(lam_x, lam_y, lam_z, body=ibody)

                    # Add the contributions to the derivative of the dynamic pressure
                    for func in range(nfunctions):
                        # get contribution to dynamic pressure derivative
                        if scenario.steady and ibody == 1:
                            self.dFdqinf[func] = 0.0
                        if step > 0:
                            self.dFdqinf[func] -= np.dot(body.aero_loads, psi_F[:, func])/self.qinf

                # Solve the heat flux adjoint equation
                if body.thermal_transfer is not None:
                    psi_Q = - body.dQdfta

                    lam_x_thermal = np.zeros((body.aero_nnodes, nfunctions),
                                             dtype=TransferScheme.dtype)
                    lam_y_thermal = np.zeros((body.aero_nnodes, nfunctions),
                                             dtype=TransferScheme.dtype)
                    lam_z_thermal = np.zeros((body.aero_nnodes, nfunctions),
                                             dtype=TransferScheme.dtype)
                    lam_mag_thermal = np.zeros((body.aero_nnodes, nfunctions),
                                               dtype=TransferScheme.dtype)

                    for func in range(nfunctions):
                        lam_mag_thermal[:, func] = self.thermal_scale * psi_Q[:, func]/self.flow_dt

                    self.fun3d_adjoint.input_heat_flux_adjoint(lam_x_thermal, lam_y_thermal, lam_z_thermal,
                                                               lam_mag_thermal, body=ibody)

                    for func in range(nfunctions):
                        if scenario.steady and ibody == 1:
                            self.dHdq[func] = 0.0
                        if step > 0:
                            self.dHdq[func] -= np.dot(body.aero_heat_flux_mag, psi_Q[:, func])/ self.thermal_scale

                if 'rigid' in body.motion_type:
                    self.fun3d_adjoint.input_rigid_transform(body.rigid_transform, body=ibody)

        # Update the aerodynamic and grid adjoint variables (Note: step starts at 1
        # in FUN3D)
        self.fun3d_adjoint.iterate(rstep)

        for ibody, body in enumerate(bodies, 1):
            # Extract dG/du_a^T psi_G from FUN3D
            if body.transfer is not None:
                lam_x, lam_y, lam_z = self.fun3d_adjoint.extract_grid_adjoint_product(body.aero_nnodes,
                                                                                      nfunctions, body=ibody)
                for func in range(nfunctions):
                    lam_x_temp = lam_x[:,func]*self.flow_dt
                    lam_y_temp = lam_y[:,func]*self.flow_dt
                    lam_z_temp = lam_z[:,func]*self.flow_dt

                    lam_x_temp = lam_x_temp.reshape((-1 ,1))
                    lam_y_temp = lam_y_temp.reshape((-1, 1))
                    lam_z_temp = lam_z_temp.reshape((-1, 1))
                    body.dGdua[:,func] = np.hstack((lam_x_temp, lam_y_temp, lam_z_temp)).flatten(order='c')

            if body.thermal_transfer is not None:
                lam_t = self.fun3d_adjoint.extract_thermal_adjoint_product(body.aero_nnodes,
                                                                           nfunctions, body=ibody)

                for func in range(nfunctions):
                    lam_t_temp = (lam_t[:, func] / body.T_ref) * self.flow_dt
                    body.dAdta[:, func] = lam_t_temp

            if 'rigid' in body.motion_type:
                body.dGdT = self.fun3d_adjoint.extract_rigid_adjoint_product(nfunctions) * self.flow_dt

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
        self.fun3d_adjoint.post()
        os.chdir(self.root_dir)

    def step_pre(self, scenario, bodies, step):
        self.fun3d_flow.step_pre(step)
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

        # Deform aerodynamic mesh
        for ibody, body in enumerate(bodies, 1):
            if 'deform' in body.motion_type and body.aero_nnodes > 0 and body.transfer is not None:
                dx = np.asfortranarray(body.aero_disps[0::3])
                dy = np.asfortranarray(body.aero_disps[1::3])
                dz = np.asfortranarray(body.aero_disps[2::3])
                self.fun3d_flow.input_deformation(dx, dy, dz, body=ibody)
            if 'rigid' in body.motion_type and body.transfer is not None:
                self.fun3d_flow.input_rigid_transform(body.rigid_transform, body=ibody)
            if body.thermal_transfer is not None:
                temps = np.asfortranarray(body.aero_temps[:])/body.T_ref
                self.fun3d_flow.input_wall_temperature(temps, body=ibody)

        # Take a step in FUN3D
        self.comm.Barrier()
        bcont = self.fun3d_flow.step_solver()
        if bcont == 0:
            if self.comm.Get_rank()==0:
                print("Negative volume returning fail")
            fail = 1
            os.chdir(self.root_dir)
            return fail

        # Pull out the forces from FUN3D
        for ibody, body in enumerate(bodies, 1):
            if body.aero_nnodes > 0:
                if body.transfer is not None:
                    fx, fy, fz = self.fun3d_flow.extract_forces(body.aero_nnodes, body=ibody)
                    body.aero_loads = np.zeros(3*body.aero_nnodes, dtype=TransferScheme.dtype)
                    body.aero_loads[0::3] = fx[:]
                    body.aero_loads[1::3] = fy[:]
                    body.aero_loads[2::3] = fz[:]

                if body.thermal_transfer is not None:
                    cqx, cqy, cqz, cq_mag = self.fun3d_flow.extract_heat_flux(body.aero_nnodes, body=ibody)
                    body.aero_heat_flux = np.zeros(3*body.aero_nnodes, dtype=TransferScheme.dtype)
                    body.aero_heat_flux_mag = np.zeros(body.aero_nnodes, dtype=TransferScheme.dtype)
                    body.aero_heat_flux[0::3] = self.thermal_scale * cqx[:]
                    body.aero_heat_flux[1::3] = self.thermal_scale * cqy[:]
                    body.aero_heat_flux[2::3] = self.thermal_scale * cqz[:]
                    body.aero_heat_flux_mag[:] = self.thermal_scale * cq_mag[:]
        return 0

    def step_post(self, scenario, bodies, step):
        self.fun3d_flow.step_post(step)

        # save this steps forces for the adjoint
        self.force_hist[scenario.id][step] = {}
        for ibody, body in enumerate(bodies, 1):
            if body.transfer is not None:
                self.force_hist[scenario.id][step][ibody] = body.aero_loads.copy()

            if body.thermal_transfer is not None:
                self.heat_flux_hist[scenario.id][step][ibody] = body.aero_heat_flux.copy()
                self.heat_flux_mag_hist[scenario.id][step][ibody] = body.aero_heat_flux_mag.copy()
                self.aero_temps_hist[scenario.id][step][ibody] = body.aero_temps.copy()
        return 0
