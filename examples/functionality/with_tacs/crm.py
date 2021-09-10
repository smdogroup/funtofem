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
CRM (common research model)
--------------------------------------------------------------------------------
The following script demonstrates a "half-loop" in aeroelastic analysis using
TransferScheme 
"""
import numpy as np
from mpi4py import MPI
from tacs import TACS, elements, constitutive
from funtofem import TransferScheme

tacs_comm = MPI.COMM_WORLD
"""
--------------------------------------------------------------------------------
Input aerodynamic surface mesh and forces
--------------------------------------------------------------------------------
"""
XF = np.loadtxt('funtofemforces.dat')
aero_X = XF[:,:3].flatten().astype(TransferScheme.dtype)
aero_loads = (251.8 * 251.8 / 2 * 0.3)*XF[:,3:].flatten().astype(TransferScheme.dtype) / float(tacs_comm.Get_size())
aero_nnodes = aero_X.shape[0]/3

"""
--------------------------------------------------------------------------------
Initialize TACS
--------------------------------------------------------------------------------
"""

# Load structural mesh from bdf file
struct_mesh = TACS.MeshLoader(tacs_comm)
struct_mesh.scanBDFFile("CRM_box_2nd.bdf")

# Set constitutive properties
rho = 2500.0 # density, kg/m^3
E = 70e9 # elastic modulus, Pa
nu = 0.3 # poisson's ratio
kcorr = 5.0 / 6.0 # shear correction factor
ys = 350e6 # yield stress, Pa
min_thickness = 0.001
max_thickness = 0.020
thickness = 0.005

# Loop over components, creating stiffness and element object for each
num_components = struct_mesh.getNumComponents()
for i in range(num_components):
    descriptor = struct_mesh.getElementDescript(i)
    stiff = constitutive.isoFSDT(rho, E, nu, kcorr, ys, thickness, i,
                                 min_thickness, max_thickness)
    element = None
    if descriptor in ["CQUAD", "CQUADR", "CQUAD4"]:
        element = elements.MITCShell(2, stiff, component_num=i)
    struct_mesh.setElement(i, element)

# Create tacs assembler object
tacs = struct_mesh.createTACS(6)
res = tacs.createVec()
ans = tacs.createVec()
mat = tacs.createFEMat()

# Create distributed node vector from TACS Assembler object and extract the
# nodes
struct_X_vec = tacs.createNodeVec()
tacs.getNodes(struct_X_vec)
struct_X = struct_X_vec.getArray()
struct_nnodes = len(struct_X)/3

# Create the preconditioner for the corresponding matrix
pc = TACS.Pc(mat)

# Set design variables (thicknesses of components)
x = np.loadtxt('sizing.dat', dtype=TACS.dtype)
tacs.setDesignVars(x)

# Assemble the Jacobian
alpha = 1.0
beta = 0.0
gamma = 0.0
tacs.assembleJacobian(alpha, beta, gamma, res, mat)
pc.factor()

"""
--------------------------------------------------------------------------------
Set up MELD and transfer loads
--------------------------------------------------------------------------------
"""
# Creating MELD
comm = MPI.COMM_WORLD
isymm = -1
num_nearest = 100
beta = 0.5
meld = TransferScheme.pyMELD(comm, comm, 0, comm, 0, isymm, num_nearest, beta)

# Set nodes into transfer scheme
meld.setStructNodes(struct_X) 
meld.setAeroNodes(aero_X) 

# Initialize transfer scheme
meld.initialize()

# Transfer loads
init_disps = np.zeros(struct_X.shape, dtype=TransferScheme.dtype)
aero_disps = np.zeros(3*aero_nnodes, dtype=TransferScheme.dtype)
struct_loads = np.zeros(3*struct_nnodes, dtype=TransferScheme.dtype)

meld.transferDisps(init_disps, aero_disps)
meld.transferLoads(aero_loads, struct_loads)
meld_load_mags = np.sqrt(np.sum(struct_loads.reshape((3,1,-1))**2, axis=0))

# collect on the master
total_sum = comm.reduce(np.sum(struct_loads))
nnodes = comm.reduce(struct_nnodes)

if comm.Get_rank()==0:
    print("MELD sum loads:  ", total_sum)
    print("MELD mean load:  ", total_sum / (nnodes*3))
   # print "MELD load stdev: ", np.std(global_load_mags)

"""
--------------------------------------------------------------------------------
Set up RBF scheme and transfer loads
--------------------------------------------------------------------------------
"""
# Creating RBF
rbf_type = TransferScheme.PY_THIN_PLATE_SPLINE
denom = 10
rbf = TransferScheme.pyRBF(comm, comm, 0, comm, 0, rbf_type, denom)

# Set nodes into transfer scheme
rbf.setStructNodes(struct_X) 
rbf.setAeroNodes(aero_X) 

# Initialize transfer scheme
rbf.initialize()

# Transfer loads
aero_disps_rbf = np.zeros(3*aero_nnodes, dtype=TransferScheme.dtype)
struct_loads_rbf = np.zeros(3*struct_nnodes, dtype=TransferScheme.dtype)

rbf.transferLoads(aero_loads, struct_loads_rbf)
rbf_load_mags = np.sqrt(np.sum(struct_loads_rbf.reshape((3,1,-1))**2, axis=0))
print("RBF sum loads:  ", np.sum(struct_loads_rbf))
print("RBF mean load:  ", np.mean(rbf_load_mags))
print("RBF load stdev: ", np.std(rbf_load_mags))

"""
--------------------------------------------------------------------------------
Compute displacements due to MELD's loads using TACS
--------------------------------------------------------------------------------
"""
# Set structural loads into residual array res. Since we're using shell
# elements, res specifies three force components and three moment components per
# node
res.zeroEntries()
res_array = res.getArray()
res_array[0::6] = struct_loads[0::3]
res_array[1::6] = struct_loads[1::3]
res_array[2::6] = struct_loads[2::3]
tacs.applyBCs(res)

# Solve Ku = F and set u into answer array
pc.applyFactor(res, ans) 
tacs.setVariables(ans)

"""
# Analogous to res, ans specifies three displacements and three rotations per
# node
ans_array = ans.getArray()
struct_disps = np.zeros(len(struct_X), dtype=TransferScheme.dtype)
struct_disps[0::3] = ans_array[0::6]
struct_disps[1::3] = ans_array[1::6]
struct_disps[2::3] = ans_array[2::6]

# We can use TACS built-in visualization capability to look at the loads that
# came out of the load transfer. We do so by overwriting the rotations in ans
# with those loads
ans_array[3::6] = struct_loads[0::3]
ans_array[4::6] = struct_loads[1::3]
ans_array[5::6] = struct_loads[2::3]
tacs.applyBCs(ans)
tacs.setVariables(ans)
"""

# Output information from TACS for visualization in Tecplot
flag = (TACS.ToFH5.NODES |
        TACS.ToFH5.DISPLACEMENTS |
        TACS.ToFH5.STRAINS |
        TACS.ToFH5.STRESSES |
        TACS.ToFH5.EXTRAS)
f5 = TACS.ToFH5(tacs, TACS.PY_SHELL, flag)
f5.writeToFile('crm_meld.f5')

"""
--------------------------------------------------------------------------------
Compute displacements due to the RBF scheme's loads using TACS
--------------------------------------------------------------------------------
"""
# Set structural loads into residual array res. Since we're using shell
# elements, res specifies three force components and three moment components per
# node
res.zeroEntries()
res_array = res.getArray()
res_array[0::6] = struct_loads_rbf[0::3]
res_array[1::6] = struct_loads_rbf[1::3]
res_array[2::6] = struct_loads_rbf[2::3]
tacs.applyBCs(res)

# Solve Ku = F and set u into answer array
pc.applyFactor(res, ans) 
tacs.setVariables(ans)

"""
# Analogous to res, ans specifies three displacements and three rotations per
# node
ans_array = ans.getArray()
struct_disps = np.zeros(len(struct_X), dtype=TransferScheme.dtype)
struct_disps[0::3] = ans_array[0::6]
struct_disps[1::3] = ans_array[1::6]
struct_disps[2::3] = ans_array[2::6]

# We can use TACS built-in visualization capability to look at the loads that
# came out of the load transfer. We do so by overwriting the rotations in ans
# with those loads
ans_array[3::6] = struct_loads[0::3]
ans_array[4::6] = struct_loads[1::3]
ans_array[5::6] = struct_loads[2::3]
tacs.applyBCs(ans)
tacs.setVariables(ans)
"""

# Output information from TACS for visualization in Tecplot
f5.writeToFile('crm_rbf.f5')

"""
--------------------------------------------------------------------------------
Transfer displacments using TransferScheme
--------------------------------------------------------------------------------
"""
"""
meld.transferDisps(struct_disps, aero_disps)
rbf.transferDisps(struct_disps, aero_disps_rbf)
"""
