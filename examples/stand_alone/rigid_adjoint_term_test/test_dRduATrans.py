"""
--------------------------------------------------------------------------------
Derivative test
--------------------------------------------------------------------------------
Test the ability to take derivatives of rigid transform
"""

from __future__ import print_function

import numpy as np
from funtofem import TransferScheme
from mpi4py import MPI
import matplotlib.pyplot as plt

# Create really simple aerodynamic mesh 
aero_X = np.array([[ 1.0,  2.0,  0.0],
                   [-1.0,  1.0,  0.0],
                   [-1.0, -1.0,  0.0],
                   [ 1.0, -1.0,  0.0]], dtype=TransferScheme.dtype)
aero_X = np.random.random((4,3)).astype(TransferScheme.dtype)

# Create aerodynamic diplacements (rotation + translation + noise)
#R = np.eye(3, dtype=TransferScheme.dtype)
#t = np.zeros(3, dtype=TransferScheme.dtype)
#e = np.zeros(aero_X.shape, dtype=TransferScheme.dtype)
psi = 0.5*np.pi
R = np.array([[np.cos(psi), -np.sin(psi), 0.0],
              [np.sin(psi),  np.cos(psi), 0.0],
              [0.0, 0.0, 1.0]], dtype=TransferScheme.dtype)
t = np.array([2.0, 0.0, 0.0], dtype=TransferScheme.dtype)
e = np.random.random(aero_X.shape).astype(TransferScheme.dtype)
#e = np.arange(aero_X.size).reshape(aero_X.shape).astype(TransferScheme.dtype)

# Add complex perturbation to displacements
#cpert = np.ones(aero_X.size).reshape(aero_X.shape)
cpert = np.random.random(aero_X.shape).astype(TransferScheme.dtype)
#cpert = np.arange(aero_X.size).reshape(aero_X.shape)
e += 1e-30j*cpert

def computeTransformAndDisps(R, t, e, X):
    """
    Assemble 4 x 4 rigid transformation matrix from rotation matrix and
    translation vector and perturbations and apply to given array of points 
    """
    T = np.empty((4,4), dtype=TransferScheme.dtype)
    T[:3,:3] = R
    T[:3,3] = t
    T[3,:3] = 0.0
    T[3,3] = 1.0

    X_hmg = np.hstack((X, np.ones((4,1), dtype=TransferScheme.dtype)))
    disps_hmg = X_hmg.dot(T.T)
    disps_hmg -= X_hmg
    disps = disps_hmg[:,:-1] + e

    return T, disps
  
_, aero_disps = computeTransformAndDisps(R, t, e, aero_X)


# Create TransferScheme
comm = MPI.COMM_WORLD
beta = 0.5
num_nearest = 4
isymm = -1
meld = TransferScheme.pyMELD(comm, comm, 0, comm, 0, isymm, num_nearest, beta)

# Load data into TransferScheme
aero_nnodes = aero_X.shape[0]
meld.setAeroNodes(aero_X.flatten(order='C'))

# Decompose displacements
R = np.zeros(9, dtype=TransferScheme.dtype)
t = np.zeros(3, dtype=TransferScheme.dtype)
e = np.zeros(3*aero_nnodes, dtype=TransferScheme.dtype)
meld.transformEquivRigidMotion(aero_disps.flatten(order='C'), R, t, e)

# Compare the original and decomposed displacements
R = R.reshape((3,3), order='F')
e = e.reshape((-1,3))
_, decomp_disps = computeTransformAndDisps(R, t, e, aero_X)

#plt.figure(figsize=(8,6))
#plt.scatter(aero_X[:,0], aero_X[:,1], color='g')
#plt.scatter(aero_X[:,0] + aero_disps[:,0], aero_X[:,1] + aero_disps[:,1], color='r')
#plt.scatter(aero_X[:,0] + decomp_disps[:,0], aero_X[:,1] + decomp_disps[:,1], 
#            alpha=0.5, color='b')
#plt.axis('equal')
#plt.show()

# Test derivatives with respect to displacements 
#psi_R = np.hstack((np.zeros(9), np.zeros(3))).astype(TransferScheme.dtype)
psi_R = np.random.random(12).astype(TransferScheme.dtype)
products = np.zeros(3*aero_nnodes, dtype=TransferScheme.dtype)
meld.applydRduATrans(psi_R, products)

directional_der = -np.hstack((R.imag.flatten(order='F'), t.imag))/1e-30
cs_approx = directional_der.dot(psi_R)
hand_coded = products.dot(cpert.flatten(order='C'))

print("Hand-coded:", hand_coded.real)
print("CS Approx.:", cs_approx.real)
