import numpy as np
from funtofem import TransferScheme
from mpi4py import MPI
import unittest


class TransferSchemeTest:
    transfer_schemes = ["pyMELD"]

    transfer_options = {
        "pyMELD": {"symmetry": 1, "num_nearest": 10, "beta": 0.5},
        "pyLinearizedMELD": {"num_nearest": 10, "beta": 0.5},
    }

    def test_disp_transfer(self):

        dh = 1e-6
        if TransferScheme.dtype is complex:
            dh = 1e-30

        for name in self.transfer_options:
            kwargs = self.transfer_options[name]

            transfer, aero_nnodes, struct_nnodes = self._setup_transfer(name, kwargs)

            uS = np.random.random(3 * struct_nnodes).astype(TransferScheme.dtype)
            uA = np.random.random(3 * aero_nnodes).astype(TransferScheme.dtype)

            fS = np.random.random(3 * struct_nnodes).astype(TransferScheme.dtype)
            fA = np.random.random(3 * aero_nnodes).astype(TransferScheme.dtype)

            uS_pert = np.random.random(3 * struct_nnodes).astype(TransferScheme.dtype)

            transfer.testLoadTransfer(uS, fA, uS_pert, dh)
            # transfer.testDispJacVecProducts(US, test_vec_a1, test_vec_s1, dh)
            # transfer.testLoadJacVecProducts(US, FA, test_vec_s1, test_vec_s2, dh)
            # transfer.testdDdxA0Products(US, test_vec_a1, test_vec_a2, dh)
            # transfer.testdDdxS0Products(US, test_vec_a1, test_vec_s1, dh)
            # transfer.testdLdxA0Products(US, FA, test_vec_a1, test_vec_s1, dh)
            # transfer.testdLdxS0Products(US, FA, test_vec_s1, test_vec_s2, dh)

    def _setup_transfer(self, name, kwargs):
        comm = MPI.COMM_WORLD

        scheme = getattr(TransferScheme, name)
        transfer = scheme(comm, comm, 0, comm, 0, **kwargs)

        aero_nnodes = 33
        aero_X = np.random.random(3 * aero_nnodes).astype(TransferScheme.dtype)
        transfer.setAeroNodes(aero_X)

        struct_nnodes = 51
        struct_X = np.random.random(3 * struct_nnodes).astype(TransferScheme.dtype)
        transfer.setStructNodes(struct_X)

        transfer.initialize()

        return transfer, aero_nnodes, struct_nnodes


if __name__ == "__main__":
    test = TransferSchemeTest()
    test.test_disp_transfer()
