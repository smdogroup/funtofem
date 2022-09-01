import os
from tacs import TACS
from mpi4py import MPI
import numpy as np
from funtofem import TransferScheme
from pyfuntofem.funtofem_model import FUNtoFEMmodel
from pyfuntofem.variable import Variable
from pyfuntofem.scenario import Scenario
from pyfuntofem.body import Body
from pyfuntofem.function import Function
from pyfuntofem.test_solver import TestAerodynamicSolver
from pyfuntofem.funtofem_nlbgs_driver import FUNtoFEMnlbgs
from pyfuntofem.tacs_interface import createTacsInterfaceFromBDF
from bdf_test_utils import elasticity_callback
import unittest

np.random.seed(1234567)

base_dir = os.path.dirname(os.path.abspath(__file__))
bdf_filename = os.path.join(base_dir, "input_files", "test_bdf_file.bdf")


class TacsFrameworkTest(unittest.TestCase):
    def _setup_model_and_driver(self):

        # Build the model
        model = FUNtoFEMmodel("wedge")
        plate = Body("plate", "aeroelastic", group=0, boundary=1)

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

        solvers["structural"] = createTacsInterfaceFromBDF(
            model, comm, nprocs, bdf_filename, callback=elasticity_callback
        )
        solvers["flow"] = TestAerodynamicSolver(comm, model)

        tacs_comm = solvers["structural"].tacs_comm

        # L&D transfer options
        transfer_options = {
            "analysis_type": "aeroelastic",
            "scheme": "meld",
            "thermal_scheme": "meld",
            "npts": 5,
        }

        # instantiate the driver
        driver = FUNtoFEMnlbgs(
            solvers, comm, tacs_comm, 0, comm, 0, transfer_options, model=model
        )

        return model, driver

    def test_solver_coupling(self):
        model, driver = self._setup_model_and_driver()

        # Check whether to use the complex-step method or now
        complex_step = False
        epsilon = 1e-5
        rtol = 1e-5
        if TransferScheme.dtype == complex and TACS.dtype == complex:
            complex_step = True
            epsilon = 1e-30
            rtol = 1e-9

        # Manual test of the disciplinary solvers
        scenario = model.scenarios[0]
        bodies = model.bodies
        solvers = driver.solvers

        fail = solvers["flow"].test_adjoint(
            "flow",
            scenario,
            bodies,
            epsilon=epsilon,
            complex_step=complex_step,
            rtol=rtol,
        )
        assert fail == False

        fail = solvers["structural"].test_adjoint(
            "structural",
            scenario,
            bodies,
            epsilon=epsilon,
            complex_step=complex_step,
            rtol=rtol,
        )
        assert fail == False

        return

    def test_coupled_derivatives(self):

        model, driver = self._setup_model_and_driver()

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
    test = TacsFrameworkTest()
    test.test_solver_coupling()
    test.test_coupled_derivatives()
