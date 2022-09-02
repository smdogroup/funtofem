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


class CRMtacs(TacsSteadyInterface):
    def __init__(self, comm, tacs_comm, model, n_tacs_procs):
        super(CRMtacs, self).__init__(comm, tacs_comm, model)

        self.tacs_proc = False
        if comm.Get_rank() < n_tacs_procs:
            self.tacs_proc = True
            struct_mesh = TACS.MeshLoader(tacs_comm)
            struct_mesh.scanBDFFile("CRM_box_2nd.bdf")

            # Set constitutive properties
            rho = 2500.0  # density, kg/m^3
            E = 70.0e9  # elastic modulus, Pa
            nu = 0.3  # poisson's ratio
            kcorr = 5.0 / 6.0  # shear correction factor
            ys = 350e6  # yield stress, Pa
            min_thickness = 0.001
            max_thickness = 0.100

            thickness = 0.015
            spar_thick = 0.015

            # Loop over components in mesh, creating stiffness and element
            # object for each
            map = np.zeros(240, dtype=int)
            num_components = struct_mesh.getNumComponents()
            for i in range(num_components):
                descript = struct_mesh.getElementDescript(i)
                comp = struct_mesh.getComponentDescript(i)
                if "SPAR" in comp:
                    t = spar_thick
                else:
                    t = thickness
                stiff = constitutive.isoFSDT(
                    rho, E, nu, kcorr, ys, t, i, min_thickness, max_thickness
                )
                element = None
                if descript in ["CQUAD", "CQUADR", "CQUAD4"]:
                    element = elements.MITCShell(2, stiff, component_num=i)
                struct_mesh.setElement(i, element)

                # Create map
                if "LE_SPAR" in comp:
                    segnum = int(comp[-2:])
                    map[i] = segnum
                if "TE_SPAR" in comp:
                    segnum = int(comp[-2:])
                    map[i] = segnum + 48
                if "IMPDISP" in comp:
                    map[i] = i
                elif "RIB" in comp:
                    segnum = int(comp[-9:-7]) - 1
                    if segnum > 3:
                        segnum -= 1
                    map[i] = segnum + 188
                if "U_SKIN" in comp:
                    segnum = int(comp[-9:-7]) - 1
                    map[i] = segnum + 92
                if "L_SKIN" in comp:
                    segnum = int(comp[-9:-7]) - 1
                    map[i] = segnum + 140

            self.dof = 6

            # Create tacs assembler object
            tacs = struct_mesh.createTACS(self.dof)
            res = tacs.createVec()
            ans = tacs.createVec()
            mat = tacs.createFEMat()

            # Create distributed node vector from TACS Assembler object and
            # extract the node locations
            nbodies = 1
            struct_X = []
            struct_nnodes = []
            for body in range(nbodies):
                self.struct_X_vec = tacs.createNodeVec()
                tacs.getNodes(self.struct_X_vec)
                struct_X.append(self.struct_X_vec.getArray())
                struct_nnodes.append(len(struct_X) / 3)

            tacs.setNodes(self.struct_X_vec)

            # Create the preconditioner for the corresponding matrix
            pc = TACS.Pc(mat)

            alpha = 1.0
            beta = 0.0
            gamma = 0.0
            tacs.assembleJacobian(alpha, beta, gamma, res, mat)
            pc.factor()

            # Create GMRES object for structural adjoint solves
            nrestart = 0  # number of restarts before giving up
            m = 30  # size of Krylov subspace (max # of iterations)
            gmres = TACS.KSM(mat, pc, m, nrestart)

            # Initialize member variables pertaining to TACS
            self.tacs = tacs
            self.res = res
            self.ans = ans
            self.mat = mat
            self.pc = pc
            self.struct_X = struct_X
            self.struct_nnodes = struct_nnodes
            self.gmres = gmres
            self.svsens = tacs.createVec()
            self.struct_rhs_vec = tacs.createVec()
            self.psi_S_vec = tacs.createVec()
            psi_S = self.psi_S_vec.getArray()
            self.psi_S = np.zeros((psi_S.size, self.nfunc), dtype=TACS.dtype)
            self.ans_array = []
            for scenario in range(len(model.scenarios)):
                self.ans_array.append(self.ans.getArray().copy())
        self.initialize(model.scenarios[0], model.bodies)

    def post_export_f5(self):
        flag = TACS.ToFH5.NODES | TACS.ToFH5.DISPLACEMENTS | TACS.ToFH5.EXTRAS
        f5 = TACS.ToFH5(self.tacs, TACS.PY_SHELL, flag)
        filename_struct_out = "crm" + ".f5"
        f5.writeToFile(filename_struct_out)
