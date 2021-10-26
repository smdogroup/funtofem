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

        assembler = None
        mat = None
        pc = None
        gmres = None

        self.T_ref = 300.0

        if comm.Get_rank() < n_tacs_procs:
            # Set constitutive properties
            rho = 4540.0  # density, kg/m^3
            E = 118e9 # elastic modulus, Pa
            nu = 0.325 # poisson's ratio
            ys = 1050e6  # yield stress, Pa
            kappa = 6.89
            specific_heat=463.0
            thickness = 0.015
            volume = 25 # need tacs volume for TACSAverageTemperature function

            # Create the constitutvie propertes and model
            props = constitutive.MaterialProperties(rho=4540.0, specific_heat=463.0,
                                                    kappa = 6.89, E=118e9, nu=0.325, ys=1050e6)
            con = constitutive.SolidConstitutive(props, t=1.0, tNum=0)

            # Set the model type = linear thermoelasticity
            elem_model = elements.LinearThermoelasticity3D(con)

            # Create the basis class
            basis = elements.LinearHexaBasis()

            # Create the element
            element = elements.Element3D(elem_model, basis)
            varsPerNode = elem_model.getVarsPerNode()

            # Load in the mesh
            mesh = TACS.MeshLoader(tacs_comm)
            mesh.scanBDFFile('tacs_aero.bdf')

            # Set the element
            mesh.setElement(0, element)

            # Create the assembler object
            assembler = mesh.createTACS(varsPerNode)

            # Create the preconditioner for the corresponding matrix
            mat = assembler.createSchurMat()
            pc = TACS.Pc(mat)

            # Create GMRES object for structural adjoint solves
            nrestart = 0 # number of restarts before giving up
            m = 30 # size of Krylov subspace (max # of iterations)
            gmres = TACS.KSM(mat, pc, m, nrestart)

        self._initialize_variables(assembler, mat, pc, gmres)

        self.initialize(model.scenarios[0], model.bodies)

        return

    def post_export_f5(self):
        flag = (TACS.OUTPUT_CONNECTIVITY |
                TACS.OUTPUT_NODES |
                TACS.OUTPUT_DISPLACEMENTS |
                TACS.OUTPUT_STRAINS)
        f5 = TACS.ToFH5(self.assembler, TACS.SOLID_ELEMENT, flag)
        filename_struct_out = "tets"  + ".f5"
        f5.writeToFile(filename_struct_out)
