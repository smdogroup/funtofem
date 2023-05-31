import numpy as np
from funtofem import TransferScheme
from funtofem.model import Scenario
import unittest

np.random.seed(343)


class ThermalConductTest(unittest.TestCase):
    def test_thermal_conduct_deriv(self):
        """
        Perform a finite difference check to test the implementation of the thermal
        conductivity scaling and its derivative.
        f = v * k(tA)
        dfdk = v

        fd = v*(k(tA+h*p)-k(tA))/h

        Parameters
        ----------
        """

        myType = TransferScheme.dtype
        steady = Scenario("steady", group=0, steps=1)

        aero_temps = np.array(np.random.rand(100) * 400, dtype=myType)
        rtol = 1e-6

        p = np.ones(aero_temps.shape, dtype=TransferScheme.dtype)
        h = 1e-5

        v = np.random.randn(*aero_temps.shape)
        v = np.array(v, dtype=TransferScheme.dtype)

        k0 = steady.get_thermal_conduct(aero_temps)

        temps_pert = aero_temps + h * p
        k1 = steady.get_thermal_conduct(temps_pert)

        fd = v * (k1 - k0) / h
        dkdtA = steady.get_thermal_conduct_deriv(aero_temps)
        dfdk = v
        dfdtA = dfdk * dkdtA

        fd_scalar = np.dot(fd, p)
        dfdtA_scalar = np.dot(dfdtA, p)

        rel_err = (fd_scalar - dfdtA_scalar) / dfdtA_scalar

        print(
            f"Finite difference check for thermal conductivity derivative: {rel_err}",
            flush=True,
        )
        assert abs(rel_err) < rtol


if __name__ == "__main__":
    unittest.main()
