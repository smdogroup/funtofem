import numpy as np
from mpi4py import MPI
from funtofem import TransferScheme

from funtofem.model import FUNtoFEMmodel, Variable, Scenario, Body, Function
from funtofem.interface import (
    TestAerodynamicSolver,
    TestStructuralSolver,
    RadiationInterface,
    SolverManager,
)
from funtofem.driver import FUNtoFEMnlbgs, TransferSettings

import unittest


class CoupledRadiationTest(unittest.TestCase):

    def _setup_model_and_driver(self):
        # Build the model
        model = FUNtoFEMmodel("model")
        plate = Body("plate", "aerothermal", group=0, boundary=1)

        # Create a structural variable
        for i in range(5):
            thickness = np.random.rand() * 200
            svar = Variable(
                "thickness %d" % (i), value=thickness, lower=0.01, upper=10.0
            )
            plate.add_variable("structural", svar)

        model.add_body(plate)

        # Create a scenario to run
        steady = Scenario("steady", group=0, steps=100)

        # Add the aerodynamic variables to the scenario
        for i in range(4):
            value = np.random.rand()
            avar = Variable("aero var %d" % (i), value=value, lower=-10.0, upper=10.0)
            steady.add_variable("aerodynamic", avar)

        # Add a function to the scenario
        temp = Function("temperature", analysis_type="structural")
        steady.add_function(temp)
        steady.set_temperature(T_ref=200, T_inf=300)
        steady.set_therm_rad_vals(emis=0.8)

        # Add the steady-state scenario
        model.add_scenario(steady)

        # Instantiate a test solver for the flow and structures
        comm = MPI.COMM_WORLD
        solvers = SolverManager(comm)
        solvers.flow = TestAerodynamicSolver(comm, model)
        solvers.structural = TestStructuralSolver(comm, model, hot_temps=True)
        solvers.thermal_rad = RadiationInterface(comm, model)

        # L&D transfer options
        transfer_settings = TransferSettings(npts=5)

        # instantiate the driver
        driver = FUNtoFEMnlbgs(
            solvers, transfer_settings=transfer_settings, model=model
        )

        model.print_summary()

        return model, driver

    def test_coupled_derivatives(self):
        model, driver = self._setup_model_and_driver()

        # Check whether to use the complex-step method or now
        complex_step = False
        epsilon = 1e-6
        rtol = 1e-5
        if TransferScheme.dtype == complex:
            complex_step = True
            epsilon = 1e-30
            rtol = 1e-9

        # Solve the forward analysis
        driver.solve_forward()
        # driver.solve_adjoint()

        # Get the functions
        functions = model.get_functions()
        variables = model.get_variables()

        # Store the function values
        fvals_init = []
        for func in functions:
            fvals_init.append(func.value)

        # Solve the adjoint and get the function gradients
        driver.solve_adjoint()
        grads = model.get_function_gradients()

        print(f"Gradients: {grads[0][:]}")

        # Set the new variable values
        if complex_step:
            variables[0].value = variables[0].value + 1j * epsilon
            model.set_variables(variables)
        else:
            variables[0].value = variables[0].value + epsilon
            model.set_variables(variables)

        driver.solve_forward()

        # Store the function values
        fvals = []
        for func in functions:
            fvals.append(func.value)

        if complex_step:
            deriv = fvals[0].imag / epsilon

            rel_error = (deriv - grads[0][0]) / deriv
            pass_ = False
            if driver.comm.rank == 0:
                pass_ = abs(rel_error) < rtol
                print("Approx. gradient (CS) = ", deriv.real)
                print("Adjoint gradient      = ", grads[0][0].real)
                print("Relative error        = ", rel_error.real)
                print("Pass flag             = ", pass_)

            pass_ = driver.comm.bcast(pass_, root=0)
            assert pass_
        else:
            deriv = (fvals[0] - fvals_init[0]) / epsilon

            rel_error = (deriv - grads[0][0]) / deriv
            pass_ = False
            if driver.comm.rank == 0:
                pass_ = abs(rel_error) < rtol
                print("Approx. gradient (FD) = ", deriv)
                print("Adjoint gradient      = ", grads[0][0])
                print("Relative error        = ", rel_error)
                print("Pass flag             = ", pass_)

            pass_ = driver.comm.bcast(pass_, root=0)
            assert pass_

        return


if __name__ == "__main__":
    unittest.main()
