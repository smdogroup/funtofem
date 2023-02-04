import os
from tacs import TACS
from mpi4py import MPI
import numpy as np
from funtofem import TransferScheme

from pyfuntofem.model import FUNtoFEMmodel, Variable, Scenario, Body, Function
from pyfuntofem.interface import (
    TestAerodynamicSolver,
    TacsSteadyInterface,
    SolverManager,
)
from pyfuntofem.driver import FUNtoFEMnlbgs, TransferSettings

from bdf_test_utils import elasticity_callback, thermoelasticity_callback
import unittest

np.random.seed(1234567)

base_dir = os.path.dirname(os.path.abspath(__file__))
bdf_filename = os.path.join(base_dir, "input_files", "test_bdf_file.bdf")


class TacsFrameworkTest(unittest.TestCase):
    N_PROCS = 2

    def _setup_model_and_driver(self, analysis_type):
        # Build the model
        model = FUNtoFEMmodel("wedge")
        plate = Body("plate", analysis_type, group=0, boundary=1)

        # Create a structural variable
        thickness = 1.0
        svar = Variable("thickness", value=thickness, lower=0.01, upper=0.1)
        plate.add_variable("structural", svar)
        model.add_body(plate)

        # Create a scenario to run
        steps = 150
        steady = Scenario("steady", group=0, steps=steps)

        # Add a function to the scenario
        ks = Function("ksfailure", analysis_type="structural")
        steady.add_function(ks)

        # Add a function to the scenario
        temp = Function("temperature", analysis_type="structural")
        steady.add_function(temp)

        model.add_scenario(steady)

        # Instantiate the solvers we'll use here
        solvers = {}

        # Build the TACS interface
        nprocs = 1
        comm = MPI.COMM_WORLD

        if "therm" in analysis_type:
            callback = thermoelasticity_callback
        else:
            callback = elasticity_callback

        solvers = SolverManager(comm)
        solvers.structural = TacsSteadyInterface.create_from_bdf(
            model, comm, nprocs, bdf_filename, callback=callback
        )
        solvers.flow = TestAerodynamicSolver(comm, model)

        # L&D transfer options
        transfer_settings = TransferSettings(npts=5)

        # instantiate the driver
        driver = FUNtoFEMnlbgs(
            solvers, transfer_settings=transfer_settings, model=model
        )

        return model, driver

    def test_aeroelastic(self):
        model, driver = self._setup_model_and_driver("aeroelastic")

        # Check whether to use the complex-step method or now
        complex_step = False
        epsilon = 1e-5
        rtol = 1e-4
        if TransferScheme.dtype == complex and TACS.dtype == complex:
            complex_step = True
            epsilon = 1e-30
            rtol = 1e-9

        # Solve the forward analysis
        driver.solve_forward()
        driver.solve_adjoint()

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

        if driver.comm.rank == 0:
            if complex_step:
                deriv = fvals[0].imag / epsilon

                rel_error = (deriv - grads[0][0]) / deriv
                print("Approximate gradient  = ", deriv.real)
                print("Adjoint gradient      = ", grads[0][0].real)
                print("Relative error        = ", rel_error.real)
                assert abs(rel_error) < rtol
            else:
                deriv = (fvals[0] - fvals_init[0]) / epsilon

                rel_error = (deriv - grads[0][0]) / deriv
                print("Approximate gradient  = ", deriv)
                print("Adjoint gradient      = ", grads[0][0])
                print("Relative error        = ", rel_error)
                assert abs(rel_error) < rtol

        return

    def test_aerothermal(self):
        model, driver = self._setup_model_and_driver("aerothermal")

        # Check whether to use the complex-step method or now
        complex_step = False
        epsilon = 1e-5
        rtol = 1e-4
        if TransferScheme.dtype == complex and TACS.dtype == complex:
            complex_step = True
            epsilon = 1e-30
            rtol = 1e-9

        # Solve the forward analysis
        driver.solve_forward()
        driver.solve_adjoint()

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

        if driver.comm.rank == 0:
            if complex_step:
                deriv = fvals[0].imag / epsilon

                rel_error = (deriv - grads[0][0]) / deriv
                print("Approximate gradient  = ", deriv.real)
                print("Adjoint gradient      = ", grads[0][0].real)
                print("Relative error        = ", rel_error.real)
                assert abs(rel_error) < rtol
            else:
                deriv = (fvals[0] - fvals_init[0]) / epsilon

                rel_error = (deriv - grads[0][0]) / deriv
                print("Approximate gradient  = ", deriv)
                print("Adjoint gradient      = ", grads[0][0])
                print("Relative error        = ", rel_error)
                assert abs(rel_error) < rtol

        return

    def test_aerothermoelastic(self):
        model, driver = self._setup_model_and_driver("aerothermoelastic")

        # Check whether to use the complex-step method or now
        complex_step = False
        epsilon = 1e-5
        rtol = 1e-5
        if TransferScheme.dtype == complex and TACS.dtype == complex:
            complex_step = True
            epsilon = 1e-30
            rtol = 1e-9

        # Solve the forward analysis
        driver.solve_forward()
        driver.solve_adjoint()

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

        if driver.comm.rank == 0:
            if complex_step:
                deriv = fvals[0].imag / epsilon

                rel_error = (deriv - grads[0][0]) / deriv
                print("Approximate gradient  = ", deriv.real)
                print("Adjoint gradient      = ", grads[0][0].real)
                print("Relative error        = ", rel_error.real)
                assert abs(rel_error) < rtol
            else:
                deriv = (fvals[0] - fvals_init[0]) / epsilon

                rel_error = (deriv - grads[0][0]) / deriv
                print("Approximate gradient  = ", deriv)
                print("Adjoint gradient      = ", grads[0][0])
                print("Relative error        = ", rel_error)
                assert abs(rel_error) < rtol

        return


if __name__ == "__main__":
    unittest.main()
