"""
Linearized MELD with a symmetry plane, checked against two references that do not use its
symmetry code:

- full MELD with the same symmetry plane, at a displacement small enough that it equals its
  own linearization (full MELD reflects the mirrored nodes' displacements and loads);
- linearized MELD without symmetry, on a structure mirrored by hand.

The derivative checks of test_transfer_schemes.py compare the transfer against itself, so
they cannot see a wrong symmetry treatment that is applied consistently.
"""

from __future__ import annotations

import unittest

import numpy as np
from mpi4py import MPI

from funtofem import TransferScheme

DTYPE = TransferScheme.dtype
NN = 8  # Number of nearest neighbors to consider
BETA = 0.5  # Relative decay factor
EPS = 1e-7  # Displacement scale at which full MELD equals its linearization
FULL_RTOL = 1e-5  # Full MELD at EPS still differs from its linearization by ~3e-7
MIRROR_RTOL = 1e-10  # The mirrored structure takes the same arithmetic, up to roundoff
PLANE_TOL = 1e-7  # Nodes closer than this to the plane are not mirrored by MELD


def symmetricCase(isymm: int) -> dict[str, np.ndarray]:
    """Structural nodes on one side of the plane normal to axis isymm, aero nodes close to it.

    A few structural nodes lie on the plane, so the case also covers the nodes that MELD
    does not mirror. Displacements and loads are random, with components normal to the plane.
    """
    rng = np.random.default_rng(1234)
    ns, na = 40, 25
    xs = rng.random((ns, 3))
    xs[:, isymm] *= 0.8
    xs[:4, isymm] = 0.0
    xa = rng.random((na, 3))
    xa[:, isymm] = 0.02 + 0.3 * rng.random(na)
    return {
        "xs": xs,
        "xa": xa,
        "us": rng.standard_normal((ns, 3)),
        "fa": rng.standard_normal((na, 3)),
    }


def reflection(isymm: int) -> np.ndarray:
    """Factors that reflect a vector about the plane normal to axis isymm."""
    factors = np.ones(3)
    factors[isymm] = -1.0
    return factors


def mirroredStructure(xs: np.ndarray, isymm: int) -> tuple[np.ndarray, np.ndarray]:
    """Structural nodes plus the reflections of those off the plane, and the off-plane mask."""
    off = np.abs(xs[:, isymm]) > PLANE_TOL
    return np.concatenate([xs, xs[off] * reflection(isymm)]), off


def createScheme(name: str, isymm: int, xs: np.ndarray, xa: np.ndarray):
    """Create and initialize a MELD-type transfer scheme on the given nodes."""
    comm = MPI.COMM_WORLD
    transfer = getattr(TransferScheme, name)(comm, comm, 0, comm, 0, isymm, NN, BETA)
    transfer.setStructNodes(xs.ravel().astype(DTYPE))
    transfer.setAeroNodes(xa.ravel().astype(DTYPE))
    transfer.initialize()
    return transfer


def transferDisps(transfer, us: np.ndarray, na: int) -> np.ndarray:
    """Aero displacements, shape (na, 3), for structural displacements us."""
    ua = np.zeros(3 * na, dtype=DTYPE)
    transfer.transferDisps(us.ravel().astype(DTYPE), ua)
    return ua.reshape(-1, 3)


def transferLoads(transfer, fa: np.ndarray, ns: int) -> np.ndarray:
    """Structural loads, shape (ns, 3), for aero loads fa, at zero displacement."""
    na = fa.shape[0]
    transfer.transferDisps(np.zeros(3 * ns, dtype=DTYPE), np.zeros(3 * na, dtype=DTYPE))
    fs = np.zeros(3 * ns, dtype=DTYPE)
    transfer.transferLoads(fa.ravel().astype(DTYPE), fs)
    return fs.reshape(-1, 3)


class LinearizedMELDSymmetryTest(unittest.TestCase):
    def assertTransferEqual(self, actual: np.ndarray, desired: np.ndarray, rtol: float):
        """Agreement to rtol of the largest reference value."""
        atol = rtol * np.abs(desired).max()
        np.testing.assert_allclose(actual, desired, rtol=0.0, atol=atol)

    def test_disps_match_full_meld(self):
        for isymm in (0, 1, 2):
            with self.subTest(isymm=isymm):
                case = symmetricCase(isymm)
                xs, xa, us = case["xs"], case["xa"], case["us"]
                full = createScheme("pyMELD", isymm, xs, xa)
                expected = transferDisps(full, EPS * us, len(xa)) / EPS
                linear = createScheme("pyLinearizedMELD", isymm, xs, xa)
                self.assertTransferEqual(
                    transferDisps(linear, us, len(xa)), expected, FULL_RTOL
                )

    def test_loads_match_full_meld(self):
        for isymm in (0, 1, 2):
            with self.subTest(isymm=isymm):
                case = symmetricCase(isymm)
                xs, xa, fa = case["xs"], case["xa"], case["fa"]
                expected = transferLoads(
                    createScheme("pyMELD", isymm, xs, xa), fa, len(xs)
                )
                linear = createScheme("pyLinearizedMELD", isymm, xs, xa)
                self.assertTransferEqual(
                    transferLoads(linear, fa, len(xs)), expected, FULL_RTOL
                )

    def test_disps_match_mirrored_structure(self):
        for isymm in (0, 1, 2):
            with self.subTest(isymm=isymm):
                case = symmetricCase(isymm)
                xs, xa, us = case["xs"], case["xa"], case["us"]
                xs_mirrored, off = mirroredStructure(xs, isymm)
                us_mirrored = np.concatenate([us, us[off] * reflection(isymm)])
                mirrored = createScheme("pyLinearizedMELD", -1, xs_mirrored, xa)
                expected = transferDisps(mirrored, us_mirrored, len(xa))
                linear = createScheme("pyLinearizedMELD", isymm, xs, xa)
                self.assertTransferEqual(
                    transferDisps(linear, us, len(xa)), expected, MIRROR_RTOL
                )

    def test_loads_match_mirrored_structure(self):
        for isymm in (0, 1, 2):
            with self.subTest(isymm=isymm):
                case = symmetricCase(isymm)
                xs, xa, fa = case["xs"], case["xa"], case["fa"]
                xs_mirrored, off = mirroredStructure(xs, isymm)
                mirrored = createScheme("pyLinearizedMELD", -1, xs_mirrored, xa)
                fs_mirrored = transferLoads(mirrored, fa, len(xs_mirrored))

                # Fold the loads on the reflected nodes back onto their originals
                expected = fs_mirrored[: len(xs)].copy()
                expected[off] += fs_mirrored[len(xs) :] * reflection(isymm)

                linear = createScheme("pyLinearizedMELD", isymm, xs, xa)
                self.assertTransferEqual(
                    transferLoads(linear, fa, len(xs)), expected, MIRROR_RTOL
                )

    def test_derivatives_with_symmetry(self):
        dh = 1e-6
        rtol = 1e-5
        atol = 1e-30
        if DTYPE == complex:
            dh = 1e-30
            rtol = 1e-9

        for isymm in (0, 1, 2):
            with self.subTest(isymm=isymm):
                case = symmetricCase(isymm)
                transfer = createScheme(
                    "pyLinearizedMELD", isymm, case["xs"], case["xa"]
                )
                us = case["us"].ravel().astype(DTYPE)
                fa = case["fa"].ravel().astype(DTYPE)
                fail = transfer.testAllDerivatives(us, fa, dh, rtol, atol)
                self.assertEqual(fail, 0)


if __name__ == "__main__":
    unittest.main()
