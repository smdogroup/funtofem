import numpy as np, os
from mpi4py import MPI
from funtofem import TransferScheme

from funtofem.model import Variable, Scenario, Body, Function, FUNtoFEMmodel
from funtofem.driver import FUNtoFEMnlbgs, TransferSettings
from funtofem.interface import (
    TestAerodynamicSolver,
    TestStructuralSolver,
    SolverManager,
    test_directories,
)

import unittest
import traceback

comm = MPI.COMM_WORLD
base_dir = os.path.dirname(os.path.abspath(__file__))
_, output_dir = test_directories(comm, base_dir)
aero_sens_file = os.path.join(output_dir, "aero.sens")
struct_sens_file = os.path.join(output_dir, "struct.sens")


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

        pass_ = True
        try:
            model.write_sensitivity_file(
                comm, struct_sens_file, discipline="structural"
            )
        except:
            print(traceback.format_exc())
            pass_ = False

        try:
            model.write_sensitivity_file(comm, aero_sens_file, discipline="aerodynamic")
        except:
            print(traceback.format_exc())
            pass_ = False

        assert pass_

        return


if __name__ == "__main__":
    unittest.main()
