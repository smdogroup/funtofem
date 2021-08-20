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
The Works
--------------------------------------------------------------------------------
The following script demonstrates the displacement transfer's capability to
transfer a variety of types of displacements from a relatively simple structural
mesh to a relatively simple aerodynamic surface mesh
""" 
import numpy as np
from mpi4py import MPI
from funtofem import TransferScheme
import sys
sys.path.append('../')
from tecplot_output import writeOutputForTecplot
import meshpy.triangle as triangle

"""
--------------------------------------------------------------------------------
Creating meshes
--------------------------------------------------------------------------------
"""
# Create boundary of high aspect ratio, tapered plate for structure
struct_bound = [(1.791204, 0.654601),
                (1.980463, 4.844049),
                (3.535093, 4.533113),
                (3.994722, 0.654601)]
def round_trip_connect(start, end):
    result = []
    for i in range(start, end):
      result.append((i, i+1))
    result.append((end, start))
    return result
struct_facets = round_trip_connect(0, len(struct_bound)-1)

# Mesh the plate using Triangle
struct_info = triangle.MeshInfo()
struct_info.set_points(struct_bound)
struct_info.set_facets(struct_facets)
struct_mesh = triangle.build(struct_info, max_volume=1e-1, min_angle=25)
# triangle.write_gnuplot_mesh("triangles.dat", struct_mesh)

# Extracting points and connectivity
z_offset = 0.0
struct_X = []
for point in struct_mesh.points:
    point += [z_offset]
    struct_X.append(point)
struct_X = np.array(struct_X).flatten()
struct_nnodes = len(struct_X)/3

struct_conn = []
for i, t in enumerate(struct_mesh.elements):
    struct_conn += t
struct_conn = np.array(struct_conn) + 1
struct_nelems = len(struct_mesh.elements)
struct_ptr = np.arange(0, 3*struct_nelems+1, 3, dtype='intc') 

# Create rectangular plate for aerodynamic surface
aero_bound = [(1.5, 0.0),
              (1.5, 6.0),
              (4.5, 6.0),
              (4.5, 0.0)]
def round_trip_connect(start, end):
    result = []
    for i in range(start, end):
      result.append((i, i+1))
    result.append((end, start))
    return result
aero_facets = round_trip_connect(0, len(aero_bound)-1)

# Mesh the plate using Triangle
aero_info = triangle.MeshInfo()
aero_info.set_points(aero_bound)
aero_info.set_facets(aero_facets)
aero_mesh = triangle.build(aero_info, max_volume=1e-3, min_angle=25)

# Extracting points and connectivity
z_offset = 1.0
aero_X = []
for point in aero_mesh.points:
    point += [z_offset]
    aero_X.append(point)
aero_X = np.array(aero_X).flatten()
aero_nnodes = len(aero_X)/3

aero_conn = []
for i, t in enumerate(aero_mesh.elements):
    aero_conn += t
aero_conn = np.array(aero_conn) + 1
aero_nelems = len(aero_mesh.elements)
aero_ptr = np.arange(0, 3*aero_nelems+1, 3, dtype='intc') 

"""
--------------------------------------------------------------------------------
Defining displacements
--------------------------------------------------------------------------------
"""
# STRETCH
st = 1.0 # stretch factor
stretch = np.array([[1.0, 0.0, 0.0],
                    [0.0, st, 0.0],
                    [0.0, 0.0, 1.0]])
stretched = np.dot(stretch, struct_X.reshape((-1,3)).T).T

# SHEAR
sh = 0.25 # 2. / b # shear factor
shear = np.array([[1.0, sh, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0]])
sheared = np.dot(shear, stretched.T).T

# TWIST
theta_tip = -90.0 * np.pi / 180.0 # degrees of twist at tip
twisted = np.zeros(sheared.shape)
y = struct_X[1::3]
b = y.max() - y.min()
for k in range(struct_nnodes):
    p = sheared[k,:]
    y = p[1]
    theta = theta_tip * y / b
    twist = np.array([[np.cos(theta), 0.0, np.sin(theta)],
                      [0.0, 1.0, 0.0],
                      [-np.sin(theta), 0.0, np.cos(theta)]])
    p = np.dot(twist, p)
    twisted[k,:] = p

# BEND
bent_z = 0.05*struct_X[1::3]**2
bent_z = bent_z.reshape((-1,1))
bend = np.concatenate((np.zeros((struct_nnodes, 1)),
                       np.zeros((struct_nnodes, 1)),
                       bent_z), axis=1)
bent = twisted + bend

# TRANSLATION
translation = np.concatenate((np.zeros((struct_nnodes, 1)),
                              np.zeros((struct_nnodes, 1)),
                              0.0 * np.ones((struct_nnodes, 1))), axis=1)
translated = bent + translation 

struct_disps = translated.flatten() - struct_X

"""
--------------------------------------------------------------------------------
Running TransferScheme
--------------------------------------------------------------------------------
"""
# Creating transfer scheme
comm = MPI.COMM_SELF
isymm = -1
num_nearest = 20
beta = 0.5
meld = TransferScheme.pyMELD(comm, comm, 0, comm, 0, scheme, isymm)

# Set nodes into transfer scheme
meld.setStructNodes(struct_X)
meld.setAeroNodes(aero_X)

# Initialize funtofem
meld.initialize()

# Transfer displacements
aero_disps = np.zeros(3*aero_nnodes, dtype=TransferScheme.dtype)
meld.transferDisps(struct_disps)

# Write meshes to file
struct_elem_type = 1
aero_elem_type = 1
writeOutputForTecplot(struct_X, aero_X, 
                      struct_disps, aero_disps,
                      struct_conn, aero_conn,
                      struct_ptr, aero_ptr,
                      struct_elem_type, aero_elem_type)
