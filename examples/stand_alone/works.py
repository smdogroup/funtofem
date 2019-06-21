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
The Works
--------------------------------------------------------------------------------
The following script demonstrates the displacement transfer's capability to
transfer a variety of types of displacements from a relatively simple structural
mesh to a relatively simple aerodynamic surface mesh
"""

import numpy as np
from mpi4py import MPI
from funtofem import TransferScheme
from tecplot_output import writeOutputForTecplot

"""
--------------------------------------------------------------------------------
Creating meshes
--------------------------------------------------------------------------------
"""
# Creating long, thin, tapered rectangular struct surface mesh
bpts = 50
cpts = 50
struct_nnodes = bpts*cpts
b = 25.0 # span
c = b / 5.0 # chord

ys = np.linspace(0.0, b, bpts)
taper_ratio = 0.35 # c_tip / c_root

Xs = np.zeros((bpts*cpts, 1))
Ys = np.zeros((bpts*cpts, 1))
Zs = np.linspace(0.0, 0.5, bpts*cpts).reshape((bpts*cpts,1))
Zs = np.zeros((bpts*cpts,1))

for j in range(bpts):
    y = ys[j]
    cy = c *(1.0 + (taper_ratio - 1.0)*(y/b))
    xs = np.linspace(-cy / 2.0, cy / 2.0, cpts)
    Xs[j*cpts:(j+1)*cpts]= xs.reshape((cpts,1))
    Ys[j*cpts:(j+1)*cpts]= y * np.ones((cpts,1))

struct_X = np.concatenate((Xs, Ys, Zs), axis=1).flatten().astype(TransferScheme.dtype)

# Generating connectivity
struct_conn = []
struct_nelems = (bpts - 1)*(cpts - 1)
for j in range(1,bpts):
    for i in range(1,cpts):
        n1 = cpts * (j - 1) + i # bottom-left
        n2 = cpts * (j - 1) + i + 1 # bottom-right
        n3 = cpts * j + i + 1 # upper-right
        n4 = cpts * j + i # upper-left
        struct_conn += [n1, n2, n3, n4]
struct_conn = np.array(struct_conn, dtype='intc') 
struct_ptr = np.arange(0, 4*struct_nelems+1, 4, dtype='intc') 

# Creating rectangular aero surface mesh
lptsa = 2*bpts
wptsa = 2*cpts
aero_nnodes = lptsa*wptsa
xa = np.linspace(-2.0*c/2.0, 5.0*c/2., wptsa)
ya = np.linspace(-0.2*b, 1.2*b, lptsa)
Xa, Ya = np.meshgrid(xa, ya)
Za = 1.0 * np.ones((wptsa, lptsa))

# Generating connectivity
aero_conn = []
aero_nelems = (lptsa - 1)*(wptsa - 1)
for j in range(1,lptsa):
    for i in range(1,wptsa):
        n1 = wptsa * (j - 1) + i # bottom-left
        n2 = wptsa * (j - 1) + i + 1 # bottom-right
        n3 = wptsa * j + i + 1 # upper-right
        n4 = wptsa * j + i # upper-left
        aero_conn += [n1, n2, n3, n4]
aero_conn = np.array(aero_conn, dtype='intc') 
aero_ptr = np.arange(0, 4*aero_nelems+1, 4, dtype='intc') 

# Loading aero mesh into pyTransferScheme object
xa = Xa.reshape((lptsa*wptsa, 1), order='F')
ya = Ya.reshape((lptsa*wptsa, 1), order='F')
za = Za.reshape((lptsa*wptsa, 1), order='F')
aero_X = np.concatenate((xa, ya, za), axis=1).flatten().astype(TransferScheme.dtype)

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
sh = 0.#0.25 # 2. / b # shear factor
shear = np.array([[1.0, sh, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0]])
sheared = np.dot(shear, stretched.T).T

# TWIST
theta_tip = 0.0 * np.pi / 180.0 # degrees of twist at tip
twisted = np.zeros(sheared.shape)
for k in range(bpts*cpts):
    p = sheared[k,:]
    y = p[1]
    theta = theta_tip * y / b
    twist = np.array([[np.cos(theta), 0.0, -np.sin(theta)],
                      [0.0, 1.0, 0.0],
                      [np.sin(theta), 0.0, np.cos(theta)]])
    p = np.dot(twist, p)
    twisted[k,:] = p

# BEND
bend = np.concatenate((np.zeros((bpts*cpts, 1)),
                       np.zeros((bpts*cpts, 1)),
                       0.02 * Ys**2), axis=1)
bent = twisted + bend

# TRANSLATION
translation = np.concatenate((np.zeros((bpts*cpts, 1)),
                              np.zeros((bpts*cpts, 1)),
                              1.0 * np.ones((bpts*cpts, 1))), axis=1)
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
num_nearest = 40
beta = 0.5
meld = TransferScheme.pyMELD(comm, comm, 0, comm, 0, isymm, num_nearest, beta)

linmeld = TransferScheme.pyLinearizedMELD(comm, comm, 0, comm, 0, num_nearest, beta)

rbf_type = TransferScheme.PY_THIN_PLATE_SPLINE
sampling_ratio = 2
rbf = TransferScheme.pyRBF(comm, comm, 0, comm, 0, rbf_type,
                           sampling_ratio)

# Set nodes into transfer scheme
meld.setStructNodes(struct_X)
meld.setAeroNodes(aero_X)

linmeld.setStructNodes(struct_X)
linmeld.setAeroNodes(aero_X)

rbf.setStructNodes(struct_X)
rbf.setAeroNodes(aero_X)

# Initialize transfer scheme
meld.initialize()
linmeld.initialize()
rbf.initialize()

# Transfer displacements
aero_disps = np.zeros(3*aero_nnodes, dtype=TransferScheme.dtype)
meld.transferDisps(struct_disps, aero_disps)

aero_disps2 = np.zeros(3*aero_nnodes, dtype=TransferScheme.dtype)
linmeld.transferDisps(struct_disps, aero_disps2)

aero_disps3 = np.zeros(3*aero_nnodes, dtype=TransferScheme.dtype)
rbf.transferDisps(struct_disps, aero_disps3)

# Get equivalent rigid motion
R = np.zeros(9, dtype=TransferScheme.dtype)
t = np.zeros(3, dtype=TransferScheme.dtype)
u = np.zeros(3*aero_nnodes, dtype=TransferScheme.dtype)
meld.transformEquivRigidMotion(aero_disps, R, t, u)
R = R.reshape((3,3), order='F')
aero_disps_reconstructed = np.dot(aero_X.reshape((-1,3)), R.T).flatten() + \
                           np.outer(np.ones(aero_nnodes), t).flatten() - \
                           aero_X

# Write meshes to file
struct_elem_type = 2
aero_elem_type = 2
writeOutputForTecplot(struct_X, aero_X, 
                      struct_disps, aero_disps,
                      struct_conn, aero_conn,
                      struct_ptr, aero_ptr,
                      struct_elem_type, aero_elem_type,
                      filename="meld_test.dat")

writeOutputForTecplot(struct_X, aero_X, 
                      struct_disps, aero_disps_reconstructed,
                      struct_conn, aero_conn,
                      struct_ptr, aero_ptr,
                      struct_elem_type, aero_elem_type,
                      filename="rigid_body_test.dat")

writeOutputForTecplot(struct_X, aero_X, 
                      struct_disps, aero_disps2,
                      struct_conn, aero_conn,
                      struct_ptr, aero_ptr,
                      struct_elem_type, aero_elem_type,
                      filename="linearized_test.dat")

writeOutputForTecplot(struct_X, aero_X, 
                      struct_disps, aero_disps3,
                      struct_conn, aero_conn,
                      struct_ptr, aero_ptr,
                      struct_elem_type, aero_elem_type,
                      filename="rbf_test.dat")

struct_X = struct_X.reshape((-1,3))
print(np.min(struct_X, axis=0))
print(np.max(struct_X, axis=0))
