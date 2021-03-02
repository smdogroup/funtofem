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

class wedgeTACS(TacsSteadyInterface):
    def __init__(self, comm, tacs_comm, model, n_tacs_procs):
        super(wedgeTACS,self).__init__(comm, tacs_comm, model)

        self.tacs_proc = False
        if comm.Get_rank() < n_tacs_procs:
            self.tacs_proc = True
            #mesh = TACS.MeshLoader(tacs_comm)
            #mesh.scanBDFFile("tacs_aero.bdf")

            # Set constitutive properties
            T_ref = 300.0
            rho = 4540.0  # density, kg/m^3
            E = 118e9 # elastic modulus, Pa
            nu = 0.325 # poisson's ratio
            ys = 1050e6  # yield stress, Pa
            kappa = 6.89
            specific_heat=463.0
            thickness = 0.015
            volume = 25 # need tacs volume for TACSAverageTemperature function

            # Create the constitutvie propertes and model
            props_plate = constitutive.MaterialProperties(rho=4540.0, specific_heat=463.0, kappa = 6.89, E=118e9, nu=0.325, ys=1050e6)
            #con_plate = constitutive.ShellConstitutive(props_plate,thickness,1,0.01,0.10)
            #model_plate = elements.ThermoelasticPlateModel(con_plate)
            con_plate = constitutive.PlaneStressConstitutive(props_plate, t=1.0, tNum=0)
            model_plate = elements.HeatConduction2D(con_plate)

            # Create the basis class
            quad_basis = elements.LinearQuadBasis()

            # Create the element
            #element_shield = elements.Element2D(model_shield, quad_basis)
            #element_insulation = elements.Element2D(model_insulation, quad_basis)
            element_plate = elements.Element2D(model_plate, quad_basis)
            varsPerNode = model_plate.getVarsPerNode()

            # Load in the mesh
            mesh = TACS.MeshLoader(tacs_comm)
            mesh.scanBDFFile('tacs_aero.bdf')

            # Set the element
            mesh.setElement(0, element_plate)

            # Create the assembler object
            #varsPerNode = heat.getVarsPerNode()
            assembler = mesh.createTACS(varsPerNode)
            res = assembler.createVec()
            ans = assembler.createVec()
            mat = assembler.createSchurMat()

            # Create distributed node vector from TACS Assembler object and
            # extract the node locations
            nbodies = 1
            struct_X = []
            struct_nnodes = []
            for body in range(nbodies):
                self.struct_X_vec = assembler.createNodeVec()
                assembler.getNodes(self.struct_X_vec)
                struct_X.append(self.struct_X_vec.getArray())
                struct_nnodes.append(len(struct_X) / 3)

            assembler.setNodes(self.struct_X_vec)

            # Create the preconditioner for the corresponding matrix
            pc = TACS.Pc(mat)

            alpha = 1.0
            beta = 0.0
            gamma = 0.0
            assembler.assembleJacobian(alpha,beta,gamma,res,mat)
            pc.factor()

            # Create GMRES object for structural adjoint solves
            nrestart = 0 # number of restarts before giving up
            m = 30 # size of Krylov subspace (max # of iterations)
            gmres = TACS.KSM(mat, pc, m, nrestart)

            # Initialize member variables pertaining to TACS
            self.T_ref = T_ref
            self.vol = volume
            self.assembler = assembler
            self.res = res
            self.ans = ans
            self.mat = mat
            self.pc = pc
            self.struct_X = struct_X
            self.struct_nnodes = struct_nnodes
            self.gmres = gmres
            self.svsens = assembler.createVec()
            self.struct_rhs_vec = assembler.createVec()
            self.psi_T_S_vec = assembler.createVec()
            psi_T_S = self.psi_T_S_vec.getArray()
            self.psi_T_S = np.zeros((psi_T_S.size,self.nfunc),dtype=TACS.dtype)
            self.ans_array = []
            self.svsenslist = []
            self.dvsenslist = []

            for func in range(self.nfunc):
                self.svsenslist.append(self.assembler.createVec())
                self.dvsenslist.append(self.assembler.createDesignVec())

            for scenario in range(len(model.scenarios)):
                self.ans_array.append(self.ans.getArray().copy())
        self.initialize(model.scenarios[0],model.bodies)

    def post_export_f5(self):
        flag = (TACS.OUTPUT_CONNECTIVITY |
                TACS.OUTPUT_NODES |
                TACS.OUTPUT_DISPLACEMENTS |
                TACS.OUTPUT_STRAINS)
        f5 = TACS.ToFH5(self.assembler, TACS.SCALAR_2D_ELEMENT, flag)
        filename_struct_out = "tets"  + ".f5"
        f5.writeToFile(filename_struct_out)
