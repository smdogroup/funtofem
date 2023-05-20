import numpy as np
from pyfuntofem import TransferScheme
from mpi4py import MPI
import unittest

np.random.seed(1234567)


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

    def test_meld_thermal(self):
        comm = MPI.COMM_WORLD

        # Set typical parameter values
        isymm = 1  # Symmetry axis (0, 1, 2 or -1 for no symmetry)
        nn = 10  # Number of nearest neighbors to consider
        beta = 0.5  # Relative decay factor
        transfer = TransferScheme.pyMELDThermal(comm, comm, 0, comm, 0, isymm, nn, beta)

        aero_nnodes = 33
        aero_X = np.random.random(3 * aero_nnodes).astype(TransferScheme.dtype)
        transfer.setAeroNodes(aero_X)

        struct_nnodes = 51
        struct_X = np.random.random(3 * struct_nnodes).astype(TransferScheme.dtype)
        transfer.setStructNodes(struct_X)

        transfer.initialize()

        # Set random forces
        tS = np.random.random(struct_nnodes).astype(TransferScheme.dtype)
        hA = np.random.random(aero_nnodes).astype(TransferScheme.dtype)

        dh = 1e-6
        rtol = 1e-5
        atol = 1e-30
        if TransferScheme.dtype == complex:
            dh = 1e-30
            rtol = 1e-9
            atol = 1e-30

        fail = transfer.testAllDerivatives(tS, hA, dh, rtol, atol)

        assert fail == 0

    def test_linear_meld(self):
        comm = MPI.COMM_WORLD

        # Set typical parameter values
        isymm = 1  # Symmetry axis (0, 1, 2 or -1 for no symmetry)
        nn = 10  # Number of nearest neighbors to consider
        beta = 0.5  # Relative decay factor
        transfer = TransferScheme.pyLinearizedMELD(
            comm, comm, 0, comm, 0, isymm, nn, beta
        )

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

    def test_rbf(self):
        comm = MPI.COMM_WORLD

        # Set typical parameter values
        rbf_type = TransferScheme.PY_MULTIQUADRIC
        sampling_ratio = 1
        transfer = TransferScheme.pyRBF(
            comm, comm, 0, comm, 0, rbf_type, sampling_ratio
        )

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

    def test_beam_transfer(self):
        comm = MPI.COMM_WORLD

        nelems = 10
        nnodes = nelems + 1
        nodes = np.arange(0, nelems + 1, dtype=np.intc)
        conn = np.zeros((nelems, 2), dtype=np.intc)
        conn[:, 0] = nodes[:-1]
        conn[:, 1] = nodes[1:]

        dof_per_node = 6

        # Set typical parameter values
        transfer = TransferScheme.pyBeamTransfer(
            comm, comm, 0, comm, 0, conn, dof_per_node
        )

        aero_nnodes = 33
        aero_X = np.random.random(3 * aero_nnodes).astype(TransferScheme.dtype)
        transfer.setAeroNodes(aero_X)

        struct_nnodes = nnodes
        struct_X = np.random.random(3 * struct_nnodes).astype(TransferScheme.dtype)
        transfer.setStructNodes(struct_X)

        transfer.initialize()

        # Set random forces
        uS = np.random.random(dof_per_node * struct_nnodes).astype(TransferScheme.dtype)
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

    def test_quaternion_beam_transfer(self):
        comm = MPI.COMM_WORLD

        nelems = 10
        nnodes = nelems + 1
        nodes = np.arange(0, nelems + 1, dtype=np.intc)
        conn = np.zeros((nelems, 2), dtype=np.intc)
        conn[:, 0] = nodes[:-1]
        conn[:, 1] = nodes[1:]

        dof_per_node = 8

        # Set typical parameter values
        transfer = TransferScheme.pyBeamTransfer(
            comm, comm, 0, comm, 0, conn, dof_per_node
        )

        aero_nnodes = 33
        aero_X = np.random.random(3 * aero_nnodes).astype(TransferScheme.dtype)
        transfer.setAeroNodes(aero_X)

        struct_nnodes = nnodes
        struct_X = np.random.random(3 * struct_nnodes).astype(TransferScheme.dtype)
        transfer.setStructNodes(struct_X)

        transfer.initialize()

        # Set random forces
        uS = np.random.random(dof_per_node * struct_nnodes).astype(TransferScheme.dtype)
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


if __name__ == "__main__":
    test = TransferSchemeTest()
    test.test_meld()
    test.test_meld_thermal()
    test.test_rbf()
    test.test_linear_meld()
    test.test_beam_transfer()
    test.test_quaternion_beam_transfer()
