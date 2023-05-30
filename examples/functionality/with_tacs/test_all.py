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

"""
--------------------------------------------------------------------------------
test_all
--------------------------------------------------------------------------------
The following script tests funtofems functionality in an example using TACS
"""
import numpy as np
from mpi4py import MPI
from tacs import TACS, elements, functions, constitutive
from funtofem import FUNtoFEM

"""
--------------------------------------------------------------------------------
Input aerodynamic surface mesh and forces
--------------------------------------------------------------------------------
"""
XF = np.loadtxt("funtofemforces.dat")
aero_X = XF[:, :3].flatten().astype(FUNtoFEM.dtype)
aero_loads = (251.8 * 251.8 / 2 * 0.3) * XF[:, 3:].flatten().astype(FUNtoFEM.dtype)
aero_nnodes = aero_X.shape[0] / 3

"""
--------------------------------------------------------------------------------
Initialize TACS
--------------------------------------------------------------------------------
"""
tacs_comm = MPI.COMM_WORLD

# Load structural mesh
struct_mesh = TACS.MeshLoader(tacs_comm)
struct_mesh.scanBDFFile("CRM_box_2nd.bdf")

# Set constitutive properties
rho = 2500.0  # density, kg/m^3
E = 70e9  # elastic modulus, Pa
nu = 0.3  # poisson's ratio
kcorr = 5.0 / 6.0  # shear correction factor
ys = 350e6  # yield stress, Pa
min_thickness = 0.001
max_thickness = 0.020

thickness = 0.015
spar_thick = 0.015

# Loop over components in mesh, creating stiffness and element object for each
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

# Create tacs assembler object
tacs = struct_mesh.createTACS(6)
res = tacs.createVec()
ans = tacs.createVec()
mat = tacs.createFEMat()

# Create distributed node vector from TACS Assembler object and extract the
# node locations
struct_X_vec = tacs.createNodeVec()
tacs.getNodes(struct_X_vec)
struct_X = struct_X_vec.getArray().astype(FUNtoFEM.dtype)
struct_nnodes = len(struct_X) / 3

# Create the preconditioner for the corresponding matrix
pc = TACS.Pc(mat)

# Assemble the Jacobian
alpha = 1.0
beta = 0.0
gamma = 0.0
tacs.assembleJacobian(alpha, beta, gamma, res, mat)
pc.factor()

"""
--------------------------------------------------------------------------------
Set up FUNtoFEM
--------------------------------------------------------------------------------
"""
# Creating funtofem
comm = MPI.COMM_SELF
scheme = FUNtoFEM.PY_NONLINEAR
isymm = 1
funtofem = FUNtoFEM.funtofem(comm, comm, 0, comm, 0, scheme, isymm)

# Set nodes into funtofem
funtofem.setStructNodes(struct_X)
funtofem.setAeroNodes(aero_X)

# Initialize funtofem
num_nearest = 20
funtofem.initialize(num_nearest)

# Transfer loads
funtofem.transferDisps(np.zeros(struct_X.shape, dtype=FUNtoFEM.dtype))
funtofem.transferLoads(aero_loads)

# Extract structural loads
struct_loads = funtofem.getStructLoads()

"""
--------------------------------------------------------------------------------
Test funtofem in loop using TACS and frozen forces
--------------------------------------------------------------------------------
"""
# Compute displacements using TACS and transfer them with funtofem
struct_disps = np.zeros(struct_loads.shape, dtype=FUNtoFEM.dtype)

# Set structural loads into TACS
# (Shell elements: three forces, then three momements)
res_array = res.getArray()
res_array[0::6] = struct_loads[0::3]
res_array[1::6] = struct_loads[1::3]
res_array[2::6] = struct_loads[2::3]
tacs.applyBCs(res)

# Solve for displacements using TACS
pc.applyFactor(res, ans)
tacs.setVariables(ans)

# Extract displacements from TACS
# (Shell elements: three displacements, then three rotations)
ans_array = ans.getArray()
struct_disps[0::3] = ans_array[0::6]
struct_disps[1::3] = ans_array[1::6]
struct_disps[2::3] = ans_array[2::6]

# Load computed displacements into funtofem
funtofem.transferDisps(struct_disps)

# With loads and displacements input, can now use testAll
funtofem.testAll(True, 1e-6)
