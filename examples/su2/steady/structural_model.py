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
from tacs import TACS, elements, functions, constitutive
from pyfuntofem.tacs_interface import TacsSteadyInterface
import numpy as np

class OneraPlate(TacsSteadyInterface):
    """
    This is an extension of the TacsSteadyInterface class, which abstracts the
    structural iteration

    Here we just specify the structural model and instantiate the variables
    that solver will call in the solution of a steady aeroelastic problem

    """
    def __init__(self, comm, tacs_comm, model, n_tacs_procs):
        super(OneraPlate, self).__init__(comm, tacs_comm, model)

        assembler = None

        if comm.Get_rank() < n_tacs_procs:
            # Set the creator object
            ndof = 6
            creator = TACS.Creator(tacs_comm, ndof)

            if tacs_comm.rank == 0:
                # Create the elements
                nx = 10
                ny = 10

                # Set the nodes
                nnodes = (nx+1)*(ny+1)
                nelems = nx*ny
                nodes = np.arange(nnodes).reshape((nx+1, ny+1))

                conn = []
                for j in range(ny):
                    for i in range(nx):
                        # Append the node locations
                        conn.append([nodes[i, j],
                                     nodes[i+1, j],
                                     nodes[i, j+1],
                                     nodes[i+1, j+1]])

                # Set the node pointers
                conn = np.array(conn, dtype=np.intc).flatten()
                ptr = np.arange(0, 4*nelems+1, 4, dtype=np.intc)
                elem_ids = np.zeros(nelems, dtype=np.intc)
                creator.setGlobalConnectivity(nnodes, ptr, conn, elem_ids)

                # Set up the boundary conditions
                bcnodes = np.array(nodes[0,:], dtype=np.intc)

                # Set the boundary condition variables
                nbcs = ndof*bcnodes.shape[0]
                bcvars = np.zeros(nbcs, dtype=np.intc)
                for i in range(ndof):
                    bcvars[i:nbcs:ndof] = i

                # Set the boundary condition pointers
                bcptr = np.arange(0, nbcs+1, ndof, dtype=np.intc)
                creator.setBoundaryConditions(bcnodes, bcvars, bcptr)

                # Set the node locations
                Xpts = np.zeros(3*nnodes)
                x = np.linspace(0, 10, nx+1)
                y = np.linspace(0, 10, ny+1)
                for j in range(ny+1):
                    for i in range(nx+1):
                        Xpts[3*nodes[i,j]] = x[i]
                        Xpts[3*nodes[i,j]+1] = y[j]

                # Set the node locations
                creator.setNodes(Xpts)

            # Set the material properties
            props = constitutive.MaterialProperties(rho=2570.0, E=70e9, nu=0.3, ys=350e6)

            # Set constitutive properties
            t = 0.025
            tnum = 0
            maxt = 0.015
            mint = 0.015
            con = constitutive.IsoShellConstitutive(props, t=t, tNum=tnum)

            # Create a transformation object
            transform = elements.ShellNaturalTransform()

            # Create the element object
            element = elements.Quad4Shell(transform, con)

            # Set the elements
            elems = [ element ]
            creator.setElements(elems)

            # Create TACS Assembler object from the mesh loader
            assembler = creator.createTACS()

            # Create distributed matrix and vector objects
            mat = assembler.createSchurMat()  # stiffness matrix

        self._initialize_variables(assembler, mat)

        self.initialize(model.scenarios[0], model.bodies)

        return

    def post_export_f5(self):
        flag = (TACS.OUTPUT_CONNECTIVITY |
                TACS.OUTPUT_NODES |
                TACS.OUTPUT_DISPLACEMENTS |
                TACS.OUTPUT_EXTRAS)
        f5 = TACS.ToFH5(self.tacs, TACS.BEAM_OR_SHELL_ELEMENT, flag)
        file_out = "onera_struct_out.f5"
        f5.writeToFile(file_out)

        return
