import numpy as np, unittest, os
from mpi4py import MPI
from funtofem import TransferScheme

from funtofem.model import FUNtoFEMmodel, Variable, Scenario, Body, Function
from funtofem.interface import (
    TestAerodynamicSolver,
    TestStructuralSolver,
    SolverManager,
    TestResult,
    make_test_directories,
)
from funtofem.driver import FUNtoFEMnlbgs, TransferSettings

base_dir = os.path.dirname(os.path.abspath(__file__))
comm = MPI.COMM_WORLD
complex_mode = TransferScheme.dtype == complex
results_folder, _ = make_test_directories(comm, base_dir)


class MultiScenarioFrameworkTest(unittest.TestCase):
    FILENAME = "testfake-multiscenario.txt"
    FILEPATH = os.path.join(results_folder, FILENAME)

    def test_structDV_with_driver(self):
        # build a funtofem model and body
        model = FUNtoFEMmodel("fake")
        fake_body = Body.aerothermoelastic("fake")
        Variable.structural("fake-thick").set_bounds(value=1.0).register_to(fake_body)
        fake_body.register_to(model)

        # make two scenarios, each with same functions (intend to have identical functions in general case)
        # make the first scenario
        test_scenario1 = Scenario.steady("test1", steps=20)
        Function.test_aero().register_to(test_scenario1)
        Function.test_struct().register_to(test_scenario1)
        test_scenario1.register_to(model)

        # make the second scenario
        test_scenario2 = Scenario.steady("test2", steps=20)
        Function.test_aero().register_to(test_scenario2)
        Function.test_struct().register_to(test_scenario2)
        test_scenario2.register_to(model)

        # build the solvers and driver
        solvers = SolverManager(comm)
        solvers.flow = TestAerodynamicSolver(comm, model)
        solvers.structural = TestStructuralSolver(comm, model)
        driver = FUNtoFEMnlbgs(
            solvers, transfer_settings=TransferSettings(npts=10), model=model
        )

        # build a test result to run a complex step test
        max_rel_error = TestResult.derivative_test(
            "testfake-multiscenarios-structDV",
            model,
            driver,
            self.FILEPATH,
            complex_mode=complex_mode,
        )
        rtol = 1e-7 if complex_mode else 1e-4
        self.assertTrue(max_rel_error < rtol)
        return

    def test_aeroDV_with_driver(self):
        # build a funtofem model and body
        model = FUNtoFEMmodel("fake")
        fake_body = Body.aerothermoelastic("fake")
        Variable.aerodynamic("fake-aoa").set_bounds(value=1.0).register_to(fake_body)
        fake_body.register_to(model)

        # make two scenarios, each with same functions (intend to have identical functions in general case)
        # make the first scenario
        test_scenario1 = Scenario.steady("test1", steps=20)
        Function.test_aero().register_to(test_scenario1)
        Function.test_struct().register_to(test_scenario1)
        test_scenario1.register_to(model)

        # make the second scenario
        test_scenario2 = Scenario.steady("test2", steps=20)
        Function.test_aero().register_to(test_scenario2)
        Function.test_struct().register_to(test_scenario2)
        test_scenario2.register_to(model)

        # build the solvers and driver
        solvers = SolverManager(comm)
        solvers.flow = TestAerodynamicSolver(comm, model)
        solvers.structural = TestStructuralSolver(comm, model)

        driver = FUNtoFEMnlbgs(
            solvers, transfer_settings=TransferSettings(npts=5), model=model
        )

        # build a test result to run a complex step test
        max_rel_error = TestResult.derivative_test(
            "testfake-multiscenarios-aeroDV",
            model,
            driver,
            self.FILEPATH,
            complex_mode=complex_mode,
        )
        rtol = 1e-7 if complex_mode else 1e-4
        self.assertTrue(max_rel_error < rtol)
        return


if __name__ == "__main__":
    if comm.rank == 0:
        open(MultiScenarioFrameworkTest.FILEPATH, "w").close()
    unittest.main()
