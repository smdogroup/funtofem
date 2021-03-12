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

"""
--------------------------------------------------------------------------------
Cloud
--------------------------------------------------------------------------------
The following script demonstrates the load and displacement transfer in their
most basic form: transferring loads from a point to a point cloud and
transferring displacements from a point cloud to a node.
"""
import numpy as np
from mpi4py import MPI
from funtofem import TransferScheme
import random
from tecplot_output import writeOutputForTecplot

"""
--------------------------------------------------------------------------------
Creating meshes
--------------------------------------------------------------------------------
"""
# Creating random cloud of structural nodes 
N = 7 # number of points in cloud
R = 1 # maximum radius of cloud about origin (0,0,0)
random.seed(12345) # seeding the random number generator with my PIN number
points = []
for n in range(N):
    r = R * (1.0 - 2.0*random.random())
    theta = 2.0*np.pi*random.random()
    phi = np.pi*0.5 #random.random()
    x = r*np.cos(theta)*np.sin(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(phi)
    point = [x,y,z]
    points.append(point)
struct_X = np.array(points).flatten()
struct_X = np.array([[ 1.0,  1.0, -0.5],
                     [-1.0,  1.0,  0.5],
                     [-1.0, -1.0, -0.5],
                     [ 1.0, -1.0,  0.5]], dtype=TransferScheme.dtype).flatten()
struct_nnodes = struct_X.shape[0]/3

# Creating one aero node 
aero_X = np.array([[ 2.0,  2.0,  0.0],
                   [-2.0,  2.0,  0.0],
                   [-2.0, -2.0,  0.0],
                   [ 2.0, -2.0,  0.0]], dtype=TransferScheme.dtype).flatten()
aero_nnodes = aero_X.shape[0]/3

"""
--------------------------------------------------------------------------------
Defining loads
--------------------------------------------------------------------------------
"""
aero_loads = np.zeros(3*aero_nnodes, dtype=TransferScheme.dtype)
aero_loads[1] += 1.0

"""
--------------------------------------------------------------------------------
Defining displacements
--------------------------------------------------------------------------------
"""
psi = np.pi / 2.0
T = np.array([[np.cos(psi), -np.sin(psi), 0.0],
              [np.sin(psi), np.cos(psi), 0.0],
              [0.0, 0.0, 1.0]])

rotation = np.dot(T, struct_X.reshape((-1,3)).T).T
struct_disps = rotation.flatten() - struct_X

"""
--------------------------------------------------------------------------------
Running FUNtoFEM
--------------------------------------------------------------------------------
"""
# Creating transfer scheme
comm = MPI.COMM_SELF
isymm = -1
num_nearest = 10
beta = 0.5
meld = TransferScheme.pyMELD(comm, comm, 0, comm, 0, isymm, num_nearest, beta)
rbf_type = TransferScheme.PY_INVERSE_MULTIQUADRIC
rbf = TransferScheme.pyRBF(comm, comm, 0, comm, 0, rbf_type, 2)

# Set nodes into transfer scheme
meld.setStructNodes(struct_X) 
meld.setAeroNodes(aero_X) 
rbf.setStructNodes(struct_X) 
rbf.setAeroNodes(aero_X) 

# Initialize transfer scheme
meld.initialize()
rbf.initialize()

# Transfer loads
init_disps = np.zeros(3*struct_nnodes, dtype=TransferScheme.dtype)
aero_disps = np.zeros(3*aero_nnodes, dtype=TransferScheme.dtype)
meld.transferDisps(init_disps, aero_disps)
struct_loads = np.zeros(3*struct_nnodes, dtype=TransferScheme.dtype)
meld.transferLoads(aero_loads, struct_loads)
print("Structural loads: ")
print(struct_loads)

struct_loads2 = np.zeros(3*struct_nnodes, dtype=TransferScheme.dtype)
rbf.transferLoads(aero_loads, struct_loads2)
print(struct_loads2)

# Transfer displacements
meld.transferDisps(struct_disps, aero_disps)
print("Aerodynamic displacements: ")
print(aero_disps)

aero_disps2 = np.zeros(3*aero_nnodes, dtype=TransferScheme.dtype)
meld.transferDisps(struct_disps, aero_disps2)
print(aero_disps2)

# Write struct and aero meshes (before and after) to file
writeOutputForTecplot(struct_X, aero_X, 
                      struct_disps, aero_disps,
                      struct_loads=struct_loads, aero_loads=aero_loads)

writeOutputForTecplot(struct_X, aero_X, 
                      struct_disps, aero_disps2,
                      struct_loads=struct_loads2, aero_loads=aero_loads, 
                      filename="rbf_test.dat")
