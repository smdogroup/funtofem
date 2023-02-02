import numpy as np
from mpi4py import MPI
from funtofem import TransferScheme

from pyfuntofem.model import Variable, Scenario, Body, Function, FUNtoFEMmodel
from pyfuntofem.driver import FUNtoFEMnlbgs, TransferSettings
from pyfuntofem.interface import (
    TestAerodynamicSolver,
    TestStructuralSolver,
    SolverManager,
)

import unittest
import traceback


class SensitivityFileTest(unittest.TestCase):
    def _setup_model_and_driver(self):
        # Build the model
        model = FUNtoFEMmodel("model")
        plate = Body("plate", "aerothermal", group=0, boundary=1)

        # Create a structural variable
        for i in range(5):
            thickness = np.random.rand()
            svar = Variable(
                "thickness %d" % (i), value=thickness, lower=0.01, upper=0.1
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

        # Add the steady-state scenario
        model.add_scenario(steady)

        # Instantiate a test solver for the flow and structures
        comm = MPI.COMM_WORLD
        solvers = SolverManager(comm)
        solvers.flow = TestAerodynamicSolver(comm, model)
        solvers.structural = TestStructuralSolver(comm, model)

        # L&D transfer options
        transfer_settings = TransferSettings(npts=5)

        # instantiate the driver
        driver = FUNtoFEMnlbgs(
            solvers, transfer_settings=transfer_settings, model=model
        )

        return model, driver

    def test_sens_file(self):
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
        driver.solve_adjoint()

        comm = MPI.COMM_WORLD
        pass_ = True
        try:
            model.write_sensitivity_file(comm, "struct.sens", discipline="structural")
        except:
            print(traceback.format_exc())
            pass_ = False

        try:
            model.write_sensitivity_file(comm, "aero.sens", discipline="aerodynamic")
        except:
            print(traceback.format_exc())
            pass_ = False

        assert pass_

        return


if __name__ == "__main__":
    unittest.main()
