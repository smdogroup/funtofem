import numpy as np
from funtofem import TransferScheme
from mpi4py import MPI
import unittest


class TransferSchemeTest(unittest.TestCase):
    def test_meld(self):
        comm = MPI.COMM_WORLD

        # Set typical parameter values
        isymm = 1  # Symmetry axis (0, 1, 2 or -1 for no symmetry)
        nn = 10  # Number of nearest neighbors to consider
        beta = 0.5  # Relative decay factor
        transfer = TransferScheme.pyMELD(comm, comm, 0, comm, 0, isymm, nn, beta)

        aero_nnodes = 33
        aero_X = np.random.random(3 * aero_nnodes).astype(TransferScheme.dtype)
        transfer.setAeroNodes(aero_X)

        struct_nnodes = 51
        struct_X = np.random.random(3 * struct_nnodes).astype(TransferScheme.dtype)
        transfer.setStructNodes(struct_X)

        transfer.initialize()

        # Set random forces
        uS = np.random.random(3 * struct_nnodes).astype(TransferScheme.dtype)
        fA = np.random.random(3 * aero_nnodes).astype(TransferScheme.dtype)

        dh = 1e-6
        rtol = 1e-5
        atol = 1e-30
        if TransferScheme.dtype == complex:
            dh = 1e-30
            rtol = 1e-9
            atol = 1e-30

        fail = transfer.testAllDerivatives(uS, fA, dh, rtol, atol)

        assert fail == 0

        return


if __name__ == "__main__":
    test = TransferSchemeTest()
    test.test_meld()
