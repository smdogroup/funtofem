import numpy as np, unittest, os
from mpi4py import MPI
from funtofem import TransferScheme

from pyfuntofem.model import FUNtoFEMmodel, Variable, Scenario, Body, Function
from pyfuntofem.interface import (
    TestAerodynamicSolver,
    TacsSteadyInterface,
    SolverManager,
    TestResult,
)
from pyfuntofem.driver import FUNtoFEMnlbgs, TransferSettings

from bdf_test_utils import elasticity_callback, thermoelasticity_callback

np.random.seed(123456)

base_dir = os.path.dirname(os.path.abspath(__file__))
bdf_filename = os.path.join(base_dir, "input_files", "test_bdf_file.bdf")
comm = MPI.COMM_WORLD
ntacs_procs = 1
complex_mode = TransferScheme.dtype == complex


class MultiScenarioTacsTest(unittest.TestCase):
    FILENAME = "testfake+tacs-multiscenario.txt"

    def test_aerothermoelastic_structDV(self):
        # build a funtofem model and body
        model = FUNtoFEMmodel("wedge")
        plate = Body.aerothermoelastic("plate")
        Variable.structural("thickness").set_bounds(
            lower=0.001, value=0.01, upper=2.0
        ).register_to(plate)
        plate.register_to(model)

        # make two scenarios, each with same functions (intend to have identical functions in general case)
        # make the first scenario
        test_scenario1 = Scenario.steady("test1", steps=20)
        test_scenario1.include(Function.ksfailure()).include(Function.mass())
        test_scenario1.register_to(model)

        # make the second scenario
        test_scenario2 = Scenario.steady("test2", steps=20)
        test_scenario2.include(Function.ksfailure()).include(Function.mass())
        test_scenario2.register_to(model)

        # build the solvers and driver
        solvers = SolverManager(comm)
        solvers.flow = TestAerodynamicSolver(comm, model)
        solvers.structural = TacsSteadyInterface.create_from_bdf(
            model, comm, ntacs_procs, bdf_filename, callback=thermoelasticity_callback
        )
        driver = FUNtoFEMnlbgs(
            solvers, transfer_settings=TransferSettings(npts=10), model=model
        )

        # build a test result to run a complex step test
        max_rel_error = TestResult.derivative_test(
            "testfake+tacs-multiscenarios-structDV",
            model,
            driver,
            MultiScenarioTacsTest.FILENAME,
            complex_mode=complex_mode,
        )
        rtol = 1e-7 if complex_mode else 1e-4
        self.assertTrue(max_rel_error < rtol)
        return

    def test_aerothermoelastic_aeroDV(self):
        # build a funtofem model and body
        model = FUNtoFEMmodel("wedge")
        plate = Body.aerothermoelastic("plate")
        Variable.structural("thickness").set_bounds(
            lower=0.001, value=0.01, upper=2.0
        ).register_to(plate)
        plate.register_to(model)

        # make two scenarios, each with same functions (intend to have identical functions in general case)
        # make the first scenario
        test_scenario1 = Scenario.steady("test1", steps=20)
        test_scenario1.include(Function.ksfailure()).include(Function.mass())
        test_scenario1.register_to(model)

        # make the second scenario
        test_scenario2 = Scenario.steady("test2", steps=20)
        test_scenario2.include(Function.ksfailure()).include(Function.mass())
        test_scenario2.register_to(model)

        # build the solvers and driver
        solvers = SolverManager(comm)
        solvers.flow = TestAerodynamicSolver(comm, model)
        solvers.structural = TacsSteadyInterface.create_from_bdf(
            model, comm, ntacs_procs, bdf_filename, callback=thermoelasticity_callback
        )

        driver = FUNtoFEMnlbgs(
            solvers, transfer_settings=TransferSettings(npts=5), model=model
        )

        # build a test result to run a complex step test
        max_rel_error = TestResult.derivative_test(
            "testfake+tacs-multiscenarios-aeroDV",
            model,
            driver,
            MultiScenarioTacsTest.FILENAME,
            complex_mode=complex_mode,
        )
        rtol = 1e-7 if complex_mode else 1e-4
        self.assertTrue(max_rel_error < rtol)
        return


if __name__ == "__main__":
    unittest.main()
