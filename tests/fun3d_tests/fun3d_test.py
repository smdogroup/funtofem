import warnings
import os
from tacs import TACS
from mpi4py import MPI
from funtofem import TransferScheme
from pyfuntofem.funtofem_model import FUNtoFEMmodel
from pyfuntofem.variable import Variable
from pyfuntofem.scenario import Scenario
from pyfuntofem.body import Body
from pyfuntofem.function import Function
from pyfuntofem.funtofem_nlbgs_driver import FUNtoFEMnlbgs
from pyfuntofem.test_solver import TestAerodynamicSolver, TestStructuralSolver
import unittest
import numpy as np

try:
    from fun3d.solvers import Flow, Adjoint
    from fun3d import interface

    has_fun3d = True
except:
    has_fun3d = False
    warnings.warn("Could not import FUN3D - all tests will pass by default")

# Set a random seed so the results are the same on subsequent runs on the same
# number of processors
np.random.seed(1234567)

# base_dir = os.path.dirname(os.path.abspath(__file__))
# bdf_filename = os.path.join(base_dir, "input_files", "test_bdf_file.bdf")


def build_default_model(analysis_type="aeroelastic"):

    # Build the model
    model = FUNtoFEMmodel("model")
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

    return model


class Fun3dTest:
    class UncoupledFun3dTest(unittest.TestCase):
        """
        This is a base class for an uncoupled FUN3D test

        When inheriting from this class you must implement the member function:

        build_model()

        that instantiates and initializes a FUNtoFEMmodel class. If this function
        is not implemented, a default model is used in its place (which is probably
        not what you want.)
        """

        def _build_driver(self):

            try:
                self.model = self.build_model()
            except:
                self.model = build_default_model(analysis_type="aeroelastic")

            # Instantiate the solvers we'll use here
            solvers = {}

            comm = MPI.COMM_WORLD
            solvers["structural"] = TestStructuralSolver(comm, self.model)

            if has_fun3d:
                from pyfuntofem.fun3d_interface import Fun3dInterface
            else:
                return None

            solvers["flow"] = Fun3dInterface(comm, self.model)

            # L&D transfer options
            transfer_options = {
                "scheme": "meld",
                "thermal_scheme": "meld",
                "npts": 50,
            }

            # instantiate the driver
            driver = FUNtoFEMnlbgs(
                solvers, comm, comm, 0, comm, 0, transfer_options, model=self.model
            )

            return driver

        def test_solver_coupling(self):
            driver = self._build_driver()

            if driver is not None:
                # Check whether to use the complex-step method or now
                complex_step = False
                epsilon = 1e-5
                rtol = 1e-4
                if TransferScheme.dtype == complex and TACS.dtype == complex:
                    complex_step = True
                    epsilon = 1e-30
                    rtol = 1e-9

                # Manual test of the disciplinary solvers
                scenario = self.model.scenarios[0]
                bodies = self.model.bodies
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

            return

    # def test_coupled_derivatives(self):

    #     model, driver = self._setup_model_and_driver()

    #     # Check whether to use the complex-step method or now
    #     complex_step = False
    #     epsilon = 1e-5
    #     rtol = 1e-4
    #     if TransferScheme.dtype == complex and TACS.dtype == complex:
    #         complex_step = True
    #         epsilon = 1e-30
    #         rtol = 1e-9

    #     # Solve the forward analysis
    #     driver.solve_forward()
    #     driver.solve_adjoint()

    #     # Get the functions
    #     functions = model.get_functions()
    #     variables = model.get_variables()

    #     # Store the function values
    #     fvals_init = []
    #     for func in functions:
    #         fvals_init.append(func.value)

    #     # Solve the adjoint and get the function gradients
    #     driver.solve_adjoint()
    #     grads = model.get_function_gradients()

    #     # Set the new variable values
    #     if complex_step:
    #         variables[0].value = variables[0].value + 1j * epsilon
    #         model.set_variables(variables)
    #     else:
    #         variables[0].value = variables[0].value + epsilon
    #         model.set_variables(variables)

    #     driver.solve_forward()

    #     # Store the function values
    #     fvals = []
    #     for func in functions:
    #         fvals.append(func.value)

    #     if complex_step:
    #         deriv = fvals[0].imag / epsilon

    #         rel_error = (deriv - grads[0][0]) / deriv
    #         print("Approximate gradient  = ", deriv.real)
    #         print("Adjoint gradient      = ", grads[0][0].real)
    #         print("Relative error        = ", rel_error.real)
    #         assert abs(rel_error) < rtol
    #     else:
    #         deriv = (fvals[0] - fvals_init[0]) / epsilon

    #         rel_error = (deriv - grads[0][0]) / deriv
    #         print("Approximate gradient  = ", deriv)
    #         print("Adjoint gradient      = ", grads[0][0])
    #         print("Relative error        = ", rel_error)
    #         assert abs(rel_error) < rtol

    #     return
