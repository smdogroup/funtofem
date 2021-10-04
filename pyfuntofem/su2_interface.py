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
from funtofem import TransferScheme
from pyfuntofem.solver_interface import SolverInterface

class SU2Interface(SolverInterface):
    """FUNtoFEM interface class for SU2."""

    def __init__(self, comm, model, su2, qinf, forward_options=None, adjoint_options=None):
        """Initialize the SU2 interface"""
        self.comm = comm
        self.su2 = su2
        self.qinf = qinf

        self.surf_id = None
        self.num_local_surf_nodes = 0

        # Get the identifiers of the moving surface
        moving_marker_tags = self.su2.GetAllDeformMeshMarkersTag()
        if moving_marker_tags is None or len(moving_marker_tags) == 0:
            raise RuntimeError('No moving surface defined in the mesh')

        # Get the marker ids for each surface
        all_marker_ids = self.su2.GetAllBoundaryMarkers()
        self.surface_ids = []
        self.num_surf_nodes = []
        for tag in moving_marker_tags:
            if tag in all_marker_ids:
                self.surface_ids.append(all_marker_ids[tag])
                self.num_surf_nodes.append(self.su2.GetNumberVertices(self.surface_ids[-1]))

        # Keep track of the total number of surface nodes
        self.num_total_surf_nodes = np.sum(self.num_surf_nodes)

        # Get the initial aero surface meshes
        self.initialize(model.scenarios[0], model.bodies, first_pass=True)
        self.post(model.scenarios[0], model.bodies, first_pass=True)

        self.num_su2_iters = 5

        return

    def initialize(self, scenario, bodies, first_pass=False):

        # Reset the convergence tolerances
        self.su2.ResetConvergence()

        # Get the coordinates associated with the surface nodes
        bodies[0].aero_nnodes = self.num_total_surf_nodes
        bodies[0].aero_X = np.zeros(3*self.num_total_surf_nodes, dtype=TransferScheme.dtype)
        bodies[0].aero_loads = np.zeros(3*self.num_total_surf_nodes,
                                        dtype=TransferScheme.dtype)

        for index, surf_id in enumerate(self.surface_ids):
            offset = 3*sum(self.num_surf_nodes[:index])

            for vert in range(self.num_surf_nodes[index]):
                x, y, z = self.su2.GetInitialMeshCoord(surf_id, vert)
                idx = 3*vert + offset
                bodies[0].aero_X[idx] = x
                bodies[0].aero_X[idx+1] = y
                bodies[0].aero_X[idx+2] = z

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

        for index, surf_id in enumerate(self.surface_ids):
            offset = 3*sum(self.num_surf_nodes[:index])

            if bodies[0].aero_disps is not None:
                for vert in range(self.num_surf_nodes[index]):
                    idx = 3*vert + offset
                    u = bodies[0].aero_disps[idx]
                    v = bodies[0].aero_disps[idx+1]
                    w = bodies[0].aero_disps[idx+2]
                    self.su2.SetMeshDisplacement(surf_id, vert, u, v, w)

        # If this is an unsteady computation than we will need this:
        # self.su2.DynamicMeshUpdate(step)

        self.su2.ResetConvergence()
        self.su2.Preprocess(0)
        self.su2.Run()
        self.su2.Postprocess()
        self.su2.Monitor(0)
        self.su2.Output(0)

        # Pull out the forces from SU2
        bodies[0].aero_loads[:] = 0.0
        for index, surf_id in enumerate(self.surface_ids):
            offset = 3*sum(self.num_surf_nodes[:index])

            if bodies[0].aero_loads is not None:
                for vert in range(self.num_surf_nodes[index]):
                    fx, fy, fz = self.su2.GetFlowLoad(surf_id, vert)

                    if not self.su2.IsAHaloNode(surf_id, vert):
                        bodies[0].aero_loads[3*vert] = self.qinf * fx
                        bodies[0].aero_loads[3*vert+1] = self.qinf * fy
                        bodies[0].aero_loads[3*vert+2] = self.qinf * fz

        return 0

    def post(self,scenario, bodies, first_pass=False):
        if not first_pass:
            self.su2.Postprocessing()
        return

    def initialize_adjoint(self, scenario, bodies):
        pass

    def iterate_adjoint(self,scenario,bodies,step):
        pass

    def post_adjoint(self,scenario,bodies):
        pass

    def set_functions(self,scenario,bodies):
        pass

    def set_variables(self,scenario,bodies):
        pass

    def get_functions(self,scenario,bodies):
        pass

    def get_function_gradients(self,scenario,bodies,offset):
        pass

    def get_coordinate_derivatives(self,scenario,bodies,step):
        pass

    def set_states(self, scenario, bodies, step):
        pass
