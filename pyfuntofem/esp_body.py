#!/usr/bin env python
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

from body import Body
import numpy as np

class ESPBody(Body):
    """
    Body class with an ESP/CAPS parameterization
    """
    def __init__(self, name, id=0, group=None, boundary=0, fun3d=True, motion_type='deform',
                 csmFile=None, comm=None, outLevel=0):
        """

        Parameters
        ----------
        name: str
            name of the body
        id: int
            id number of the body
        group: int
            group number for coupling of variables
        boundary: int
            FUN3D boundary number associated with the body
        fun3d: bool
            whether or not you are using FUN3D. If true, the body class will auto-populate 'rigid_motion' required by FUN3D
        motion_type:
            the type of motion the body is undergoing. Possible options: 'deform', 'rigid', 'deform+rigid'
        comm: MPI comm
            the FUNtoFEM communicator

        See Also
        --------
        :mod:`body` : MassoudBody inherits from Body
        """

        super(ESPBody, self).__init__(name, id, group, boundary, fun3d, motion_type)

        # Initialize CAPS with the CSM file
        if csmFile is not None:
            import pyCAPS
            self.caps = pyCAPS.Problem(name, capsFile=csmFile, outLevel=outLevel)

            # Get the design parameter keys and their design variables
            self.design_keys = self.geom.despmtr.keys()
            self.ndv = len(self.design_keys)

            # Create the EGADS aim
            self.egads = self.caps.analysis.create(aim='egadsTessAIM')

            from variable import Variable
            for v in range(self.ndv):
                self.add_variable('shape', Variable('shape ' + str(v), active=False, id=v+1))

        # Set parameters for the underlying body implementation
        self.shape = True

        # Set the positions of the aerodynamic nodes
        self.aero_X      = None
        self.aero_id     = None
        self.aero_nnodes = None

        # Set the positions of the structural nodes
        self.struct_X      = None
        self.struct_id     = None
        self.struct_nnodes = None

        # Set the global communicator
        self.comm = comm

        return

    def initialize_shape_parameterization(self):
        """
        **[driver call]**
        Create the map of structural node locations since TACS doesn't give us global id numbers

        """

        # if not self.struct_id:

        return

    def update_shape(self, complex_run=False):
        """
        **[driver call]**
        perturbs the shape of the body based on the body's shape variables and update the meshes

        Parameters
        ----------
        complex_run: bool
            whether or not the run is complex mode
        """

        return

    def write_sens_files(self, aero_filename, struct_filename):
        """
        Write the sensitivity files for the aerodynamic and structural meshes on
        the root processor.

        This code collects the sensitivities from each processor and collects the
        result to the root node.
        """

        aero_comm = self.comm
        struct_comm = self.comm

        aero_proc = True
        struct_proc = True

        if aero_proc:
            all_aero_ids = aero_comm.gather(self.aero_id, root=0)
            all_aero_shape = aero_comm.gather(self.aero_shape_term, root=0)

            if aero_comm.rank == 0:
                # Discard any entries that are None
                aero_ids = []
                for d in all_aero_ids:
                    if d is not None:
                        aero_ids.append(d)

                aero_shape = []
                for d in all_aero_shape:
                    if d is not None:
                        aero_shape.append(d)

                aero_ids = np.concatenate(aero_ids)
                aero_shape = np.concatenate(aero_shape)

                with open(aero_filename, "w") as fp:
                    # the number of functions and number of nodes
                    num_funcs = aero_shape.shape[1]
                    num_nodes = aero_ids.shape[0]
                    fp.write("{} {}\n".format(num_funcs, num_nodes))

                    for func in range(num_funcs):
                        for i in range(num_nodes):
                            fp.write('{} {} {} {}\n'.format(
                                aero_ids[i],
                                aero_shape[3 * i, func],
                                aero_shape[3 * i + 1, func],
                                aero_shape[3 * i + 2, func]))

        if struct_proc:
            all_struct_ids = struct_comm.gather(self.struct_id, root=0)
            all_struct_shape = struct_comm.gather(self.struct_shape_term, root=0)

            if struct_comm.rank == 0:
                # Discard any entries that are None
                struct_ids = []
                for d in all_struct_ids:
                    if d is not None:
                        aero_ids.append(d)

                struct_shape = []
                for d in all_struct_shape:
                    if d is not None:
                        struct_shape.append(d)

                struct_ids = np.concatenate(struct_ids)
                struct_shape = np.concatenate(struct_shape)

                with open(struct_filename, "w") as fp:
                    # the number of functions and number of nodes
                    num_funcs = struct_shape.shape[1]
                    num_nodes = struct_ids.shape[0]
                    fp.write("{} {}\n".format(num_funcs, num_nodes))

                    for func in range(num_funcs):
                        for i in range(num_nodes):
                            fp.write('{} {} {} {}\n'.format(
                                struct_ids[i],
                                struct_shape[3 * i, func],
                                struct_shape[3 * i + 1, func],
                                struct_shape[3 * i + 2, func]))

        return

    def shape_derivative(self, scenario, offset):
        """
        **[driver call]**
        Calculates the shape derivative given the coordinate derivatives

        Parameters
        ----------
        scenario: scenario object
            The current scenario
        offset: int
            function offset number
        """

        # for func in range(len(scenario.functions)):
        #     for var in range(self.ndv):
        #         self.derivatives['shape'][offset+func][var] = 0.0

        # for pt in range(self.aero_nnodes):
        #     for func in range(len(scenario.functions)):
        #         for var in range(self.ndv):
        #             self.derivatives['shape'][offset+func][var] += self.aero_shape_term[3*pt  ,func] * self.aero_sd[pt*self.ndv*3+var*3  ]
        #             self.derivatives['shape'][offset+func][var] += self.aero_shape_term[3*pt+1,func] * self.aero_sd[pt*self.ndv*3+var*3+1]
        #             self.derivatives['shape'][offset+func][var] += self.aero_shape_term[3*pt+2,func] * self.aero_sd[pt*self.ndv*3+var*3+2]

        # for pt in range(self.struct_nnodes):
        #     for func in range(len(scenario.functions)):
        #         for var in range(self.ndv):
        #             self.derivatives['shape'][offset+func][var] += self.struct_shape_term[3*pt  ,func] * self.struct_sd[pt*self.ndv*3+var*3  ]
        #             self.derivatives['shape'][offset+func][var] += self.struct_shape_term[3*pt+1,func] * self.struct_sd[pt*self.ndv*3+var*3+1]
        #             self.derivatives['shape'][offset+func][var] += self.struct_shape_term[3*pt+2,func] * self.struct_sd[pt*self.ndv*3+var*3+2]

        # # Get the contributions from all the processors
        # for func in range(len(scenario.functions)):
        #     for var in range(self.ndv):
        #         deriv = np.array(self.derivatives['shape'][offset+func][var])
        #         self.derivatives['shape'][offset+func][var] = self.comm.allreduce(deriv)

        return
