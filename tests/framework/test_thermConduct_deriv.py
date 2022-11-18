import numpy as np
from funtofem import TransferScheme
from pyfuntofem.scenario import Scenario

myType = TransferScheme.dtype
steady = Scenario("steady", group=0, steps=1)

my_temps = np.array(np.random.rand(100) * 400, dtype=myType)
rtol = 1e-6


def test_thermal_conduct_deriv(aero_temps):
    """
    Perform a finite difference check to test the implementation of the thermal
    conductivity scaling and its derivative.
    f = v * k(tA)
    dfdk = v

    fd = v*(k(tA+h*p)-k(tA))/h

    Parameters
    ----------
    myType: :class:`~scenario.Scenario`
        Current scenario.
    aero_temps: np.ndarray
        Current aero surface temperatures.
    """

    p = np.ones(aero_temps.shape, dtype=TransferScheme.dtype)
    h = 1e-6

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

    return rel_err


rel_err = test_thermal_conduct_deriv(my_temps)

print(
    f"Finite difference check for thermal conductivity derivative: {rel_err}",
    flush=True,
)
assert abs(rel_err) < rtol
