"""
--------------------------------------------------------------------------------
Derivative test
--------------------------------------------------------------------------------
Test the ability to take derivatives of rigid transform
"""

import numpy as np
from funtofem import TransferScheme
from mpi4py import MPI
import unittest

np.random.seed(1234567)


def computeTransformAndDisps(R, t, e, X):
    """
    Assemble 4 x 4 rigid transformation matrix from rotation matrix and
    translation vector and perturbations and apply to given array of points
    """

    # Assemble the 4 x 4 transformation and translation matrix
    T = np.empty((4, 4), dtype=TransferScheme.dtype)
    T[:3, :3] = R
    T[:3, 3] = t
    T[3, :3] = 0.0
    T[3, 3] = 1.0

    # Assemble X with the identity as the last entry
    if len(X.shape) == 2:
        X_hmg = np.hstack((X, np.ones((4, 1), dtype=TransferScheme.dtype)))
    else:
        X0 = X.reshape(-1, 3)
        X_hmg = np.hstack((X0, np.ones((4, 1), dtype=TransferScheme.dtype)))

    # Compute the rigid component of the displacements
    disps_hmg = X_hmg.dot(T.T) - X_hmg

    # Add the local deformation component of the displacement
    disps = disps_hmg[:, :-1].reshape(e.shape) + e

    return disps


class RigidTransformTest(unittest.TestCase):
    def test_rigid(self, plot=False):

        rtol = 1e-6
        dh = 1e-6
        complex_step = False
        if TransferScheme.dtype is complex:
            rtol = 1e-9
            dh = 1e-30
            complex_step = True

        # Create a simple aerodynamic mesh
        aero_nnodes = 4
        aero_X = np.random.random(3 * aero_nnodes).astype(TransferScheme.dtype)

        # Create aerodynamic diplacements (rotation + translation + noise)
        psi = 0.5 * np.pi
        R = np.array(
            [
                [np.cos(psi), -np.sin(psi), 0.0],
                [np.sin(psi), np.cos(psi), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=TransferScheme.dtype,
        )
        t = np.array([2.0, 0.0, 0.0], dtype=TransferScheme.dtype)

        # Specified initial random shape vector
        e = np.random.random(aero_X.shape).astype(TransferScheme.dtype)

        # Create the array of perturbation to displacements
        pert = np.random.random(aero_X.shape).astype(TransferScheme.dtype)

        # Perturn the displacements by a complex-step
        if complex_step:
            e += 1j * dh * pert

        # Create TransferScheme
        comm = MPI.COMM_WORLD
        beta = 0.5
        num_nearest = 4
        isymm = -1
        meld = TransferScheme.pyMELD(comm, comm, 0, comm, 0, isymm, num_nearest, beta)

        # Load data into TransferScheme
        meld.setAeroNodes(aero_X)

        # Compute the displacements
        aero_disps = computeTransformAndDisps(R, t, e, aero_X)

        # Decompose displacements
        R0 = np.zeros(9, dtype=TransferScheme.dtype)
        t0 = np.zeros(3, dtype=TransferScheme.dtype)
        e0 = np.zeros(3 * aero_nnodes, dtype=TransferScheme.dtype)
        meld.transformEquivRigidMotion(aero_disps, R0, t0, e0)

        # Compute the exact derivative
        psi_R = np.random.random(12).astype(TransferScheme.dtype)
        products = np.zeros(3 * aero_nnodes, dtype=TransferScheme.dtype)
        meld.applydRduATrans(psi_R, products)

        if complex_step:
            directional_der = -np.hstack((R0.imag.flatten(order="F"), t0.imag)) / dh
            fd_approx = directional_der.dot(psi_R).real
            hand_coded = products.dot(pert).real
        else:
            # Apply the finite-difference perturbation
            e += dh * pert

            # Compute the displacements
            aero_disps = computeTransformAndDisps(R, t, e, aero_X)

            # Decompose displacements again for the perturbed displacements
            R1 = np.zeros(9, dtype=TransferScheme.dtype)
            t1 = np.zeros(3, dtype=TransferScheme.dtype)
            e1 = np.zeros(3 * aero_nnodes, dtype=TransferScheme.dtype)
            meld.transformEquivRigidMotion(aero_disps, R1, t1, e1)

            directional_der = (
                -np.hstack((R1.flatten(order="F") - R0.flatten(order="F"), t1 - t0))
                / dh
            )
            fd_approx = directional_der.dot(psi_R)
            hand_coded = products.dot(pert)

        rel_error = (hand_coded - fd_approx) / fd_approx
        pass_ = abs(rel_error) < rtol
        print("Hand-coded:  ", hand_coded)
        print("FD Approx.:  ", fd_approx)
        print("Rel. error:  ", rel_error)
        print("Test status: ", pass_)

        assert pass_

        if plot:
            self.plot(aero_X, aero_disps, R, t, e)

        return

    def plot(self, aero_X, aero_disps, R, t, e):
        import matplotlib.pylab as plt

        # Convert e, aero_X and aero_disps to n x 3 2d arrays
        if len(e.shape) == 1:
            e = e.reshape(-1, 3)
        if len(aero_X.shape) == 1:
            aero_X = aero_X.reshape(-1, 3)
        if len(aero_disps.shape) == 1:
            aero_disps = aero_disps.reshape(-1, 3)

        # Compare the original and decomposed displacements
        R = R.reshape((3, 3), order="F")
        decomp_disps = computeTransformAndDisps(R, t, e, aero_X)

        plt.figure(figsize=(8, 6))
        plt.scatter(aero_X[:, 0], aero_X[:, 1], color="g")
        plt.scatter(
            aero_X[:, 0] + aero_disps[:, 0],
            aero_X[:, 1] + aero_disps[:, 1],
            alpha=0.5,
            color="r",
        )
        plt.scatter(
            aero_X[:, 0] + decomp_disps[:, 0],
            aero_X[:, 1] + decomp_disps[:, 1],
            alpha=0.5,
            color="b",
        )
        plt.axis("equal")
        plt.show()

        return


if __name__ == "__main__":
    test = RigidTransformTest()
    test.test_rigid(plot=True)
