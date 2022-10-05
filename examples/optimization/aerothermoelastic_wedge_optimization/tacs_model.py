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

from mpi4py import MPI
from tacs import TACS, elements, functions, constitutive
from pyfuntofem.tacs_interface import TacsSteadyInterface

# from pyfuntofem.tacs_aerothermal_interface import TacsSteadyAerothermalInterface
import numpy as np


def createAssembler(tacs_comm):
    # Set constitutive properties
    rho = 4540.0  # density, kg/m^3
    E = 118e9  # elastic modulus, Pa 118e9
    nu = 0.325  # poisson's ratio
    ys = 1050e6  # yield stress, Pa
    kappa = 6.89
    specific_heat = 463.0

    # Set the back-pressure for the traction load
    pb = 654.9  # 10.0 # A value for the surface pressure

    # Load in the mesh
    mesh = TACS.MeshLoader(tacs_comm)
    mesh.scanBDFFile("tacs_aero.bdf")

    # Get the number of components set in the mesh. Each component is
    num_components = mesh.getNumComponents()

    # Each domain consists of the union of two components. The component
    # corresponding to the surface traction and the component corresponding
    # to remaining plate elements. There is one design variable associated
    # with each domain that shares a common (x,z) coordinate
    num_domains = num_components // 2

    # Create the constitutvie propertes and model
    props_plate = constitutive.MaterialProperties(
        rho=rho, specific_heat=specific_heat, kappp=kappa, E=E, nu=nu, ys=ys
    )

    # Create the basis class
    basis = elements.LinearHexaBasis()

    element_list = []
    for k in range(num_domains):
        # Create the elements in an element list
        con = constitutive.SolidConstitutive(props_plate, t=1.0, tNum=k)
        phys_model = elements.LinearThermoelasticity3D(con)

        # Create the element
        element_list.append(elements.Element3D(phys_model, basis))
        varsPerNode = phys_model.getVarsPerNode()

    # Set the face index for the side of the element where the traction
    # will be applied. The face indices are as follows for the hexahedral
    # basis class:
    # -x: 0, +x: 1, -y: 2, +y: 3, -z: 4, +z: 5
    faceIndex = 4

    # Set the wedge angle - 5 degrees
    theta = np.radians(5.0)

    # Compute the outward normal components for the face
    nx = np.sin(theta)
    nz = -np.cos(theta)

    # Set the traction components
    tr = [-pb * nx, 0.0, -pb * nz, 0.0]

    # Create the traction class
    traction = elements.Traction3D(varsPerNode, faceIndex, basis, tr)

    # Set the elements corresponding to each component number
    num_components = mesh.getNumComponents()
    for k in range(num_components):
        mesh.setElement(k, element_list[k % num_domains])

    # Create the assembler object
    assembler = mesh.createTACS(varsPerNode)

    # Set the traction load into the assembler object
    aux = TACS.AuxElements(assembler)
    for k in range(num_domains, num_components):
        mesh.addAuxElement(aux, k, traction)
    assembler.setAuxElements(aux)

    return assembler, num_domains


# class wedgeTACS(TacsSteadyAerothermalInterface):
class wedgeTACS(TacsSteadyInterface):
    def __init__(self, comm, tacs_comm, model, n_tacs_procs):
        super(wedgeTACS, self).__init__(comm, tacs_comm, model, ndof=1)

        self.tacs_proc = False
        if comm.Get_rank() < n_tacs_procs:
            # set refrence values here
            T_ref = 300.0
            volume = 0.01  # need tacs volume for TACSAverageTemperature function

            self.tacs_proc = True

            assembler, num_domains = createAssembler(tacs_comm)

            res = assembler.createVec()
            ans = assembler.createVec()
            mat = assembler.createSchurMat(TACS.ND_ORDER)

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
            assembler.assembleJacobian(alpha, beta, gamma, res, mat)
            pc.factor()

            # Create GMRES object for structural adjoint solves
            nrestart = 0  # number of restarts before giving up
            m = 30  # size of Krylov subspace (max # of iterations)
            gmres = TACS.KSM(mat, pc, m, nrestart)

            # Initialize member variables pertaining to TACS
            self.T_ref = T_ref
            self.vol = volume
            self.assembler = assembler
            self.res = res
            self.ans = ans
            self.ext_force = assembler.createVec()
            self.update = assembler.createVec()
            self.mat = mat
            self.pc = pc
            self.struct_X = struct_X
            self.struct_nnodes = struct_nnodes
            self.gmres = gmres
            self.svsens = assembler.createVec()
            self.struct_rhs_vec = assembler.createVec()
            self.psi_S_vec = assembler.createVec()
            psi_S = self.psi_S_vec.getArray()
            self.psi_S = np.zeros((psi_S.size, self.nfunc), dtype=TACS.dtype)
            self.psi_T_S_vec = assembler.createVec()
            psi_T_S = self.psi_T_S_vec.getArray()
            self.psi_T_S = np.zeros((psi_T_S.size, self.nfunc), dtype=TACS.dtype)
            self.ans_array = []
            self.svsenslist = []
            self.dvsenslist = []

            for func in range(self.nfunc):
                self.svsenslist.append(self.assembler.createVec())
                self.dvsenslist.append(self.assembler.createDesignVec())

            for scenario in range(len(model.scenarios)):
                self.ans_array.append(self.ans.getArray().copy())
        self.initialize(model.scenarios[0], model.bodies)

    def post_export_f5(self):
        flag = (
            TACS.OUTPUT_CONNECTIVITY
            | TACS.OUTPUT_NODES
            | TACS.OUTPUT_DISPLACEMENTS
            | TACS.OUTPUT_STRAINS
            | TACS.OUTPUT_EXTRAS
        )
        f5 = TACS.ToFH5(self.assembler, TACS.SOLID_ELEMENT, flag)
        # f5 = TACS.ToFH5(self.assembler, TACS.SCALAR_2D_ELEMENT, flag)
        filename_struct_out = "wedge_tacs" + ".f5"
        f5.writeToFile(filename_struct_out)


# Test the model
if __name__ == "__main__":
    comm = MPI.COMM_WORLD

    assembler, num_domains = createAssembler(comm)

    res = assembler.createVec()
    ans = assembler.createVec()
    update = assembler.createVec()
    mat = assembler.createSchurMat(TACS.ND_ORDER)

    # Create the preconditioner for the corresponding matrix
    pc = TACS.Pc(mat)

    # Create GMRES object for structural adjoint solves
    nrestart = 0  # number of restarts before giving up
    m = 30  # size of Krylov subspace (max # of iterations)
    gmres = TACS.KSM(mat, pc, m, nrestart)

    # Set the variables into the assembler object with the correct
    # boundary condition information
    ans.zeroEntries()
    assembler.setBCs(ans)
    assembler.setVariables(ans)

    # Assemble the Jacobian matrix
    alpha = 1.0
    beta = 0.0
    gamma = 0.0
    assembler.assembleJacobian(alpha, beta, gamma, res, mat)
    pc.factor()

    # Solve the system of equations J*update = -res
    gmres.solve(res, update)

    # Apply the update to the variables
    ans.axpy(-1.0, update)

    # Set the updated state variables into the assembler object
    assembler.setBCs(ans)
    assembler.setVariables(ans)

    # Create the average temperature function
    volume = 1 * 2 * 0.005  # need tacs volume for TACSAverageTemperature function
    avg_temp_func = functions.AverageTemperature(assembler, volume)

    # Create the TACS design vector. The length of this vector will be different
    # on different processors.
    x = assembler.createDesignVec()
    x_array = x.getArray()
    num_tacs_dvs = x_array.shape[0]

    # In our case, we're going to use 3 design variables that represent
    A = np.zeros((num_tacs_dvs, 3))
    u = np.linspace(0, 1, num_tacs_dvs)

    # Use the quadratic bernstein basis. These functions are always positive
    # and perserve a
    A[:, 0] = (1.0 - u) ** 2
    A[:, 1] = 2.0 * u * (1.0 - u)
    A[:, 2] = u**2

    # Set the input design variables. These are the values that will be passed
    # from the optimizer.
    xin = np.array([1.0, 0.25, 0.25])
    xlb = 0.1 * np.ones(3)
    xub = np.ones(3)

    # Compute the values of the design variables in TACS. x = A * xin
    x_array[:] = np.dot(A, xin)
    assembler.setDesignVars(x)

    # Now that new design variables have been set, resolve the system of equations
    ans.zeroEntries()
    assembler.setBCs(ans)
    assembler.setVariables(ans)
    assembler.assembleJacobian(alpha, beta, gamma, res, mat)
    pc.factor()
    gmres.solve(res, update)
    ans.axpy(-1.0, update)
    assembler.setBCs(ans)
    assembler.setVariables(ans)

    # Compute the gradient of the objective
    g = assembler.createDesignVec()

    dfdu = assembler.createVec()
    adjoint = assembler.createVec()

    # Evaluate the function and the df/du termp
    assembler.evalFunctions([avg_temp_func])
    assembler.addSVSens([avg_temp_func], [dfdu])

    # Solve for the adjoint equations
    assembler.assembleJacobian(alpha, beta, gamma, res, mat, matOr=TACS.TRANSPOSE)
    pc.factor()
    gmres.solve(dfdu, adjoint)

    # Compute the total derivative
    assembler.addDVSens([avg_temp_func], [g])
    assembler.addAdjointResProducts([adjoint], [g], alpha=-1.0)

    # Compute the gradient
    grad = np.dot(A.T, g.getArray())

    if comm.rank == 0:
        print(grad)

    # Write out the solution
    flag = TACS.OUTPUT_CONNECTIVITY | TACS.OUTPUT_NODES | TACS.OUTPUT_DISPLACEMENTS
    f5 = TACS.ToFH5(assembler, TACS.SOLID_ELEMENT, flag)
    filename_struct_out = "test.f5"
    f5.writeToFile(filename_struct_out)
