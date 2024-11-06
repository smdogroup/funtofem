import numpy as np
from mpi4py import MPI
from funtofem import TransferScheme
from funtofem.model import Scenario, FUNtoFEMmodel
from funtofem.interface.radiation_interface import RadiationInterface
import unittest

np.random.seed(343)

nprocs = 1
comm = MPI.COMM_WORLD


class RadiationFluxTest(unittest.TestCase):
    def test_radiative_flux_deriv(self):
        """
        Perform a finite difference check to test the implementation of the thermal radiation
        heat flux derivative.

        hR = q_rad(tA)
        dhR/dtA = dq_rad/dtA

        fd = (q_rad(tA+h*p)-q_rad(tA))/h
        """

        myType = TransferScheme.dtype
        model = FUNtoFEMmodel("radiation")
        steady = Scenario("steady", group=0, steps=1)
        steady.register_to(model)

        aero_temps = np.array(np.random.rand(100) * 400, dtype=myType)
        rtol = 1e-6

        p = np.ones(aero_temps.shape, dtype=TransferScheme.dtype)
        h = 1e-5

        temps_pert = aero_temps + h * p

        thermal_rad_solver = RadiationInterface(comm=comm, model=model)

        q_rad0 = thermal_rad_solver.calc_heat_flux(aero_temps)

        q_rad1 = thermal_rad_solver.calc_heat_flux(temps_pert)

        fd = (q_rad1 - q_rad0) / h

        dq_dtA = thermal_rad_solver.calc_heat_flux_deriv(aero_temps)

        fd_scalar = np.dot(fd, p)
        dq_dtA_scalar = np.dot(dq_dtA, p)

        rel_err = (fd_scalar - dq_dtA_scalar) / dq_dtA_scalar

        print(
            f"Finite difference check for thermal radiation heat flux derivative: {rel_err}",
            flush=True,
        )
        assert abs(rel_err) < rtol


if __name__ == "__main__":
    unittest.main()
