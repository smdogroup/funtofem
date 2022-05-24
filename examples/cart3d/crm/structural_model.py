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

class TacsCRM(TacsSteadyInterface):
    """
    This is an extension of the TacsSteadyInterface class, which abstracts the
    structural iteration

    Here we just specify the structural model and instantiate the variables
    that solver will call in the solution of a steady aeroelastic problem

    """
    def __init__(self, comm, tacs_comm, model, n_tacs_procs):
        super(TacsCRM, self).__init__(comm, tacs_comm, model)

        self.tacs_proc = False

        if comm.Get_rank() < n_tacs_procs:
            self.tacs_proc = True

            # Load CRM structural wingbox mesh from BDF file
            struct_mesh = TACS.MeshLoader(tacs_comm)
            #struct_mesh.scanBDFFile("uCRM-9_wingbox_medium.bdf")
            struct_mesh.scanBDFFile("wingbox_jagged_spcs.bdf")
            #struct_mesh.scanBDFFile("wingbox_rib_spcs.bdf")

            # Set constitutive properties
            rho = 2500.0           # density, kg/m^3
            E = 70.0e9             # elastic modulus, Pa
            nu = 0.3               # poisson's ratio
            kcorr = 5.0/6.0        # shear correction factor
            ys = 350e6             # yield stress, Pa
            min_thickness = 0.001  
            max_thickness = 0.100  
            thickness = 0.04       
            spar_thick = 0.015     

            # Loop over components in wingbox mesh, creating constitutive and
            # element object for each component
            num_components = struct_mesh.getNumComponents()
            for i in range(num_components):
                # Get the element/component descriptions from the BDF file
                elem_descr = struct_mesh.getElementDescript(i)
                comp_descr = struct_mesh.getComponentDescript(i)
                
                # If the component description indicates that component is
                # along a spar, assign it the spar thickness
                if 'SPAR' in comp_descr:
                    t = spar_thick
                else:
                    t = thickness

                stiff = constitutive.isoFSDT(rho, E, nu, kcorr, ys, t, i,
                                             min_thickness, max_thickness)

                # If the element description indicates the element is a CQUAD,
                # create a shell element in TACS
                element = None
                if elem_descr in ["CQUAD", "CQUADR", "CQUAD4"]:
                    element = elements.MITCShell(2, stiff, component_num=i)
                struct_mesh.setElement(i, element)

            # Create TACS Assembler object from the mesh loader
            self.dof = 6  # six degrees of freedom in shell elements
            tacs = struct_mesh.createTACS(self.dof)

            # Create distributed matrix and vector objects
            mat = tacs.createFEMat()  # stiffness matrix
            ans = tacs.createVec()    # state vector
            res = tacs.createVec()    # load vector (effectively)

            # Create distributed node vector from TACS Assembler object and
            # extract the node locations
            self.struct_X_vec = tacs.createNodeVec()
            self.struct_X = []
            self.struct_nnodes = []

            tacs.getNodes(self.struct_X_vec)
            self.struct_X.append(self.struct_X_vec.getArray())
            self.struct_nnodes.append(len(self.struct_X)/3)

            # Assemble the stiffness matrix
            alpha = 1.0
            beta = 0.0
            gamma = 0.0
            tacs.assembleJacobian(alpha, beta, gamma, res, mat)

            # Create the preconditioner for the stiffness matrix and factor
            pc = TACS.Pc(mat)
            pc.factor()

            # Initialize member variables pertaining to TACS that will be used
            # by FUNtoFEM in solve
            self.tacs = tacs
            self.res = res
            self.ans = ans
            self.mat = mat
            self.pc = pc

        self.initialize(model.scenarios[0], model.bodies)

    def post_export_f5(self):
            flag = (TACS.ToFH5.NODES | TACS.ToFH5.DISPLACEMENTS | TACS.ToFH5.EXTRAS)
            f5 = TACS.ToFH5(self.tacs, TACS.PY_SHELL, flag)
            file_out = "crm.f5"
            f5.writeToFile(file_out)
