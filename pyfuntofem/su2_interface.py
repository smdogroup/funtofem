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

import numpy as np
import shutil
from funtofem import TransferScheme
from pyfuntofem.solver_interface import SolverInterface
from mpi4py import MPI

import pysu2
try:
    import pysu2ad
except:
    pass

class SU2Interface(SolverInterface):
    """
    FUNtoFEM interface class for SU2.
    """

    def __init__(self, comm, model, su2_config, su2ad_config=None, qinf=1.0,
                 restart_file='restart_flow.dat',
                 solution_file='solution_flow.dat',
                 forward_options=None, adjoint_options=None):
        """
        Initialize the SU2 interface.

        Parameters
        ----------
        comm: MPI.comm
            MPI communicator
        model: :class:`FUNtoFEMmodel`
            FUNtoFEM model
        su2_config: string
            Name of the main configuration file for SU2.
        su2ad_config: string
            Name of the configuration file for adjoint computation in SU2. The contents should match
            the main config file except for those settings specific to the adjoint.
        qinf: float
            Dynamic pressure of the freestream flow.
        restart_file: string
            Name of the restart file, including filetype. Name of the file containing the flow state
            from which SU2 restarts its run.
        solution_file: string
            Name of the solution file, including filetype. SU2 writes the flow state to the solution file.
            Can then be read from to restart a future run.
        forward_options:

        adjoint_options:
        
        """
        self.comm = comm
        self.qinf = qinf
        self.su2_config = su2_config
        self.su2ad_config = su2ad_config
        self.restart_file = restart_file
        self.solution_file = solution_file
        self.su2 = None
        self.su2ad = None
        self.func_values = {}

        # Get the initial aero surface meshes
        self.initialize(model.scenarios[0], model.bodies, first_pass=True)
        self.post(model.scenarios[0], model.bodies, first_pass=True)

        return

    def _initialize_mesh(self, su2, scenario, bodies):
        # Get the identifiers of the moving surface
        moving_marker_tags = su2.GetAllDeformMeshMarkersTag()
        if moving_marker_tags is None or len(moving_marker_tags) == 0:
            raise RuntimeError('No moving surface defined in the mesh')

        # Get the marker ids for each surface
        all_marker_ids = su2.GetAllBoundaryMarkers()

        # Get all the surfaces marked for moving
        self.moving_surface_ids = []
        for tag in moving_marker_tags:
            if tag in all_marker_ids:
                self.moving_surface_ids.append(all_marker_ids[tag])

        # The indices of the local nodes that are owned by this processor
        self.local_nodes = {}

        # Set the local surface numbering for all the nodes on the surface
        self.num_local_nodes = 0
        for surf_id in self.moving_surface_ids:
            nverts = su2.GetNumberVertices(surf_id)
            for vert in range(nverts):
                # Check if this is a halo node or not
                if not su2.IsAHaloNode(surf_id, vert):
                    # Get the global vertex number
                    global_index = su2.GetVertexGlobalIndex(surf_id, vert)

                    # This is not a halo node, but check if it is already stored
                    # in the local_nodes dictionary
                    if not global_index in self.local_nodes:
                        self.local_nodes[global_index] = self.num_local_nodes
                        self.num_local_nodes += 1

        # The number of unique owned nodes
        self.num_owned_nodes = len(self.local_nodes)

        # Get the coordinates associated with the surface nodes
        bodies[0].aero_nnodes = self.num_owned_nodes
        bodies[0].aero_X = np.zeros(3*self.num_owned_nodes,
                                    dtype=TransferScheme.dtype)
        bodies[0].aero_loads = np.zeros(3*self.num_owned_nodes,
                                        dtype=TransferScheme.dtype)

        for ibody, body in enumerate(bodies):
            for surf_id in self.moving_surface_ids:
                nverts = su2.GetNumberVertices(surf_id)

                for vert in range(nverts):
                    if not su2.IsAHaloNode(surf_id, vert):
                        global_index = su2.GetVertexGlobalIndex(surf_id, vert)
                        index = self.local_nodes[global_index]

                        x, y, z = su2.GetInitialMeshCoord(surf_id, vert)
                        body.aero_X[3*index] = x
                        body.aero_X[3*index+1] = y
                        body.aero_X[3*index+2] = z

        return

    def _initialize_halo_nodes(self, su2):
        # The halo nodes that belong to this processor
         self.halo_nodes = []

         # Set the local surface numbering for all the nodes on the surface
         self.num_local_nodes = 0
         for surf_id in self.moving_surface_ids:
             nverts = su2.GetNumberVertices(surf_id)
             for vert in range(nverts):
                 # Get the global vertex number
                 global_index = su2.GetVertexGlobalIndex(surf_id, vert)

                 # Check if this is a halo node or not
                 if su2.IsAHaloNode(surf_id, vert):
                     self.halo_nodes.append(global_index)

         # Make the list of halo nodes unique
         self.halo_nodes = np.unique(self.halo_nodes).tolist()

         # Gather the nodes to all processors
         halo_nodes = self.comm.allgather(self.halo_nodes)

         # Extend the lists
         self.all_halo_nodes = []
         for nodes in halo_nodes:
             self.all_halo_nodes.extend(nodes)
         self.all_halo_nodes = np.unique(self.all_halo_nodes)

         # Store the index into the global halo node array for all halo nodes contributed
         # from this processor
         self.owned_halo_nodes = {}

         # Store where my halo nodes are coming from
         self.my_halo_nodes = {}
         for node in self.halo_nodes:
             self.my_halo_nodes[node] = np.where(self.all_halo_nodes == node)[0][0]

         # Find the locally owned halo nodes
         for i, index in enumerate(self.all_halo_nodes):
             if index in self.local_nodes:
                 self.owned_halo_nodes[index] = i

         return

    def _distribute_values(self, su2, owned, nvals=1):
        """
        Given the values of the locally owned surface nodes, distribute the
        values to all processors
        """

        # Place the local halo values into the array
        send = np.zeros(nvals*len(self.all_halo_nodes))
        recv = np.zeros(nvals*len(self.all_halo_nodes))

        # Set the halo node values into the global halo node array
        for global_index in self.owned_halo_nodes:
            local_index = self.local_nodes[global_index]
            index = self.owned_halo_nodes[global_index]

            # Values to be sent to other processors
            send[nvals*index:nvals*(index+1)] = owned[nvals*local_index:nvals*(local_index+1)]

        self.comm.Allreduce(send, recv, op=MPI.SUM)

        local = []
        for surf_id in self.moving_surface_ids:
            nverts = su2.GetNumberVertices(surf_id)
            vals = np.zeros(nvals*nverts)
            for vert in range(nverts):
                global_index = su2.GetVertexGlobalIndex(surf_id, vert)

                if not su2.IsAHaloNode(surf_id, vert):
                    index = self.local_nodes[global_index]
                    vals[nvals*vert:nvals*(vert+1)] = owned[nvals*index:nvals*(index+1)]
                else:
                    index = self.my_halo_nodes[global_index]
                    vals[nvals*vert:nvals*(vert+1)] = recv[nvals*index:nvals*(index+1)]

            local.append(vals)

        return local

    def initialize(self, scenario, bodies, first_pass=False):
        # Instantiate the SU2 flow solver
        if first_pass or self.su2 is None:
            self.su2 = pysu2.CSinglezoneDriver(self.su2_config, 1, self.comm)
            self._initialize_mesh(self.su2, scenario, bodies)

        return 0

    def iterate(self, scenario, bodies, step):
        """
        Forward iteration of SU2.
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

        for body in bodies:
            for surf_id in self.moving_surface_ids:
                nverts = self.su2.GetNumberVertices(surf_id)

                if body.transfer is not None:
                    for vert in range(nverts):
                        if not self.su2.IsAHaloNode(surf_id, vert):
                            global_index = self.su2.GetVertexGlobalIndex(surf_id, vert)
                            local_index = self.local_nodes[global_index]

                            u = body.aero_disps[3*local_index]
                            v = body.aero_disps[3*local_index+1]
                            w = body.aero_disps[3*local_index+2]
                            self.su2.SetMeshDisplacement(surf_id, vert, u, v, w)

                if body.thermal_transfer is not None:
                    for vert in range(nverts):
                        if not self.su2.IsAHaloNode(surf_id, vert):
                            global_index = self.su2.GetVertexGlobalIndex(surf_id, vert)
                            local_index = self.local_nodes[global_index]
                            Twall = body.aero_temps[local_index]
                            self.su2.SetVertexTemperature(surf_id, vert, Twall)

        # If this is an unsteady computation than we will need this:
        self.su2.ResetConvergence()
        self.su2.Preprocess(0)
        self.su2.Run()
        self.su2.Postprocess()
        self.su2.Monitor(0)
        self.su2.Output(0)

        # Pull out the forces from SU2
        for ibody, body in enumerate(bodies):
            if body.transfer:
                body.aero_loads[:] = 0.0
            if body.thermal_transfer:
                body.aero_heat_flux[:] = 0.0
                body.aero_heat_flux_mag[:] = 0.0

        for body in bodies:
            for surf_id in self.moving_surface_ids:
                nverts = self.su2.GetNumberVertices(surf_id)

                if body.transfer is not None:
                    for vert in range(nverts):
                        if not self.su2.IsAHaloNode(surf_id, vert):
                            global_vertex = self.su2.GetVertexGlobalIndex(surf_id, vert)
                            local_index = self.local_nodes[global_vertex]

                            fx, fy, fz = self.su2.GetFlowLoad(surf_id, vert)
                            body.aero_loads[3*local_index] = self.qinf * fx
                            body.aero_loads[3*local_index+1] = self.qinf * fy
                            body.aero_loads[3*local_index+2] = self.qinf * fz

                if body.thermal_transfer is not None:
                    for vert in range(nverts):
                        if not self.su2.IsAHaloNode(surf_id, vert):
                            global_vertex = self.su2.GetVertexGlobalIndex(surf_id, vert)
                            local_index = self.local_nodes[global_vertex]

                            hmag = self.su2.GetVertexNormalHeatFlux(surf_id, vert)
                            body.aero_heat_flux_mag[local_index] = hmag

        return 0

    def post(self, scenario, bodies, first_pass=False):
        # If this isn't the first pass, delete the flow object
        if not first_pass:
            # Record the function values before deleting everything
            self.func_values['drag'] = self.su2.Get_Drag()
            self.func_values['lift'] = self.su2.Get_Lift()
            self.func_values['mx'] = self.su2.Get_Mx()
            self.func_values['my'] = self.su2.Get_My()
            self.func_values['mz'] = self.su2.Get_Mz()
            self.func_values['cd'] = self.su2.Get_DragCoeff()
            self.func_values['cl'] = self.su2.Get_LiftCoeff()

            self.su2.Postprocessing()
            del self.su2
            self.su2 = None

        # Copy the restart file to the solution file...
        self.comm.Barrier()
        if not first_pass and self.comm.rank == 0:
            shutil.move(self.restart_file, self.solution_file)
        self.comm.Barrier()

        return

    def initialize_adjoint(self, scenario, bodies):
        # Instantiate the SU2 flow solver
        if self.su2ad is None:
            # Delete the primal if it exists at this point...
            if self.su2 is not None:
                del self.su2
                self.su2 = None

            # Create the discrete adjoint version of SU2
            self.su2ad = pysu2ad.CDiscAdjSinglezoneDriver(self.su2ad_config, 1, self.comm)
            self._initialize_mesh(self.su2ad, scenario, bodies)

        return

    def iterate_adjoint(self, scenario, bodies, step):

        func = 0

        for body in bodies:
            for surf_id in self.moving_surface_ids:
                nverts = self.su2ad.GetNumberVertices(surf_id)

                if body.transfer is not None:
                    psi_F = body.dLdfa[:, func]

                    for vert in range(nverts):
                        if not self.su2ad.IsAHaloNode(surf_id, vert):
                            global_vertex = self.su2ad.GetVertexGlobalIndex(surf_id, vert)
                            local_index = self.local_nodes[global_vertex]

                            fx_adj = self.qinf * psi_F[3*local_index]
                            fy_adj = self.qinf * psi_F[3*local_index+1]
                            fz_adj = self.qinf * psi_F[3*local_index+2]
                            self.su2ad.SetFlowLoad_Adjoint(surf_id, vert, fx_adj, fy_adj, fz_adj)

                if body.thermal_transfer is not None:
                    psi_Q = body.dQdfta[:, func]

                    for vert in range(nverts):
                        if not self.su2ad.IsAHaloNode(surf_id, vert):
                            global_vertex = self.su2ad.GetVertexGlobalIndex(surf_id, vert)
                            local_index = self.local_nodes[global_vertex]

                            hmag_adj = psi_Q[local_index]
                            self.su2ad.SetVertexNormalHeatFlux_Adjoint(surf_id, vert, hmag_adj)

        self.su2ad.ResetConvergence()
        self.su2ad.Preprocess(0)
        self.su2ad.Run()
        self.su2ad.Postprocess()
        self.su2ad.Update()
        stop = self.su2ad.Monitor(0)
        self.su2ad.Output(0)

        for ibody, body in enumerate(bodies):
            if body.transfer is not None:
                body.dGdua[:, func] = 0.0
            if body.thermal_transfer:
                body.dAdta[:, func] = 0.0

            for surf_id in self.moving_surface_ids:
                nverts = self.su2ad.GetNumberVertices(surf_id)

                if body.transfer is not None:
                    for vert in range(nverts):
                        if not self.su2ad.IsAHaloNode(surf_id, vert):
                            global_index = self.su2ad.GetVertexGlobalIndex(surf_id, vert)
                            local_index = self.local_nodes[global_index]

                            u_adj, v_adj, w_adj = self.su2ad.GetMeshDisp_Sensitivity(surf_id, vert)

                            # Over-write entries in dGdua. If a node
                            # appears twice or more in the list of
                            # boundary nodes, we don't want to
                            # double-count its contribution.
                            body.dGdua[3*local_index, func] = u_adj
                            body.dGdua[3*local_index+1, func] = v_adj
                            body.dGdua[3*local_index+2, func] = w_adj

                if body.thermal_transfer is not None:
                    for vert in range(nverts):
                        if not self.su2ad.IsAHaloNode(surf_id, vert):
                            global_index = self.su2ad.GetVertexGlobalIndex(surf_id, vert)
                            local_index = self.local_nodes[global_index]

                            Twall_adj = self.su2ad.GetVertexTemperature_Adjoint(surf_id, vert)
                            body.dAdta[local_index, func] = Twall_adj

        return 0

    def post_adjoint(self, scenario, bodies):
        if self.su2ad is not None:
            self.su2ad.Postprocessing()
            del self.su2ad
            self.su2ad = None

        return

    def set_functions(self, scenario, bodies):
        # Not sure what to do here yet...
        return

    def get_functions(self, scenario, bodies):
        for function in scenario.functions:
            if function.analysis_type == 'aerodynamic':
                function.value = self.func_values[function.name.lower()]

        return

    def get_function_gradients(self, scenario, bodies, offset):
        pass

    def get_coordinate_derivatives(self, scenario, bodies, step):
        pass

    def set_variables(self, scenario, bodies):
        pass

    def set_states(self, scenario, bodies, step):
        pass

    def adjoint_test(self, scenario, bodies, step=0, epsilon=1e-6):
        """
        The input to the forward computation are the displacements on the
        aerodynamic mesh. The output are the forces at the aerodynamic
        surface.

        fA = fA(uA)

        The Jacobian of the forward code is

        J = d(fA)/d(uA).

        A finite-difference adjoint-vector product gives

        J*pA ~= (fA(uA + epsilon*pA) - fA(uA))/epsilon

        The adjoint code computes the product

        lam_uA = J^{T}*lam_fA

        As a result, we should have the identity:

        lam_uA^{T}*pA = lam_fA*J*pA ~ lam_fA^{T}*(fA(uA + epsilon*pA) - fA(uA))/epsilon
        """

        # Comptue one step of the forward solution
        self.initialize(scenario, bodies)
        self.iterate(scenario, bodies, step)
        self.post(scenario, bodies)

        # Store the output forces
        for ibody, body in enumerate(bodies):
            if body.transfer is not None:
                # Copy the aero loads from the initial run for
                # later use...
                body.aero_loads_copy = body.aero_loads.copy()

                # Set the the adjoint input for the load transfer
                # at the aerodynamic loads
                body.dLdfa = np.random.uniform(size=body.dLdfa.shape)

        # Compute one step of the adjoint
        self.initialize_adjoint(scenario, bodies)
        self.iterate_adjoint(scenario, bodies, step)
        self.post_adjoint(scenario, bodies)

        # Perturb the displacements
        adjoint_product = 0.0
        for ibody, body in enumerate(bodies):
            if body.transfer is not None:
                body.aero_disps_pert = np.random.uniform(size=body.aero_disps.shape)
                body.aero_disps += epsilon*body.aero_disps_pert

                # Compute the adjoint product. Note that the
                # negative sign is from convention due to the
                # presence of the negative sign in psi_F = -dLdfa
                adjoint_product += np.dot(body.dGdua[:, 0], body.aero_disps_pert)

        # Sum up the result across all processors
        adjoint_product = self.comm.allreduce(adjoint_product)

        # Run the perturbed aerodynamic simulation
        self.initialize(scenario, bodies)
        self.iterate(scenario, bodies, step)
        self.post(scenario, bodies)

        # Compute the finite-difference approximation
        fd_product = 0.0
        for ibody, body in enumerate(bodies):
            if body.transfer is not None:
                fd = (body.aero_loads - body.aero_loads_copy)/epsilon
                fd_product += np.dot(fd, body.dLdfa[:, 0])

        # Compute the finite-differenc approximation
        fd_product = self.comm.allreduce(fd_product)

        if self.comm.rank == 0:
            print('SU2 FUNtoFEM adjoint result:           ', adjoint_product)
            print('SU2 FUNtoFEM finite-difference result: ', fd_product)

        return
