import numpy as np
from funtofem import TransferScheme
from funtofem.model import Scenario
import unittest
import warnings

np.random.seed(343)


class ThermalConductTest(unittest.TestCase):
    """
    Finite-difference checks for get_thermal_conduct / get_thermal_conduct_deriv
    across all three k_eval_strategy options: "wall", "eckert", and "fixed".

    For each strategy we verify:
        d/dtA [ v . k(tA) ]  ≈  v . (dk/dtA)(tA)
    using a forward-difference with h = 1e-5.
    """

    # Cylinder case freestream conditions (Wieting 1987) used as a representative
    # aerothermal test point.
    T_INF = 241.5   # K
    MACH  = 6.47

    def _fd_check(self, scenario, aero_temps, rtol=1e-6):
        """Run the FD check and return the relative error."""
        myType = TransferScheme.dtype
        p = np.ones(aero_temps.shape, dtype=myType)
        h = 1e-5
        v = np.array(np.random.randn(*aero_temps.shape), dtype=myType)

        k0    = scenario.get_thermal_conduct(aero_temps)
        k1    = scenario.get_thermal_conduct(aero_temps + h * p)
        dkdtA = scenario.get_thermal_conduct_deriv(aero_temps)

        fd_scalar    = np.dot(v * (k1 - k0) / h, p)
        exact_scalar = np.dot(v * dkdtA, p)
        return (fd_scalar - exact_scalar) / exact_scalar

    def test_wall_strategy(self):
        """Legacy "wall" strategy: k evaluated at T_wall."""
        myType = TransferScheme.dtype
        scenario = Scenario("wall_test", group=0, steps=1)
        # suppress the expected UserWarning about the wall strategy
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            aero_temps = np.array(np.random.rand(100) * 400 + 200, dtype=myType)
            rel_err = self._fd_check(scenario, aero_temps)
        print(f"wall   FD rel err: {rel_err:.3e}", flush=True)
        self.assertLess(abs(rel_err), 1e-6)

    def test_eckert_strategy_turbulent(self):
        """Eckert strategy with turbulent recovery factor (r = Pr^(1/3))."""
        myType = TransferScheme.dtype
        scenario = Scenario(
            "eckert_turb_test",
            group=0,
            steps=1,
            T_inf=self.T_INF,
            Mach_inf=self.MACH,
            turbulent=True,
        )
        aero_temps = np.array(np.random.rand(100) * 400 + 200, dtype=myType)
        rel_err = self._fd_check(scenario, aero_temps)
        print(f"eckert (turbulent) FD rel err: {rel_err:.3e}", flush=True)
        self.assertLess(abs(rel_err), 1e-6)

    def test_eckert_strategy_laminar(self):
        """Eckert strategy with laminar recovery factor (r = sqrt(Pr))."""
        myType = TransferScheme.dtype
        scenario = Scenario(
            "eckert_lam_test",
            group=0,
            steps=1,
            T_inf=self.T_INF,
            Mach_inf=self.MACH,
            turbulent=False,
        )
        aero_temps = np.array(np.random.rand(100) * 400 + 200, dtype=myType)
        rel_err = self._fd_check(scenario, aero_temps)
        print(f"eckert (laminar)  FD rel err: {rel_err:.3e}", flush=True)
        self.assertLess(abs(rel_err), 1e-6)

    def test_fixed_strategy(self):
        """Fixed-k strategy: derivative should be exactly zero."""
        myType = TransferScheme.dtype
        k_val = 0.05  # representative air conductivity ~822 K
        scenario = Scenario(
            "fixed_test",
            group=0,
            steps=1,
            k_fixed=k_val,
        )
        aero_temps = np.array(np.random.rand(100) * 400 + 200, dtype=myType)

        k = scenario.get_thermal_conduct(aero_temps)
        self.assertTrue(np.all(k == k_val), "fixed strategy should return constant k")

        dkdtA = scenario.get_thermal_conduct_deriv(aero_temps)
        self.assertTrue(np.all(dkdtA == 0.0), "fixed strategy derivative should be zero")

    def test_strategy_precedence(self):
        """k_fixed takes precedence over Mach_inf when both are supplied."""
        scenario = Scenario(
            "precedence_test",
            group=0,
            steps=1,
            Mach_inf=self.MACH,
            k_fixed=0.05,
        )
        self.assertEqual(scenario.k_eval_strategy, "fixed")


if __name__ == "__main__":
    unittest.main()
