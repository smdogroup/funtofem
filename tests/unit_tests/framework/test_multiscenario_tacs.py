import numpy as np, unittest, os
from mpi4py import MPI
from funtofem import TransferScheme

from funtofem.model import FUNtoFEMmodel, Variable, Scenario, Body, Function
from funtofem.interface import (
    TestAerodynamicSolver,
    TacsInterface,
    SolverManager,
    TestResult,
    test_directories,
)
from funtofem.driver import FUNtoFEMnlbgs, TransferSettings
from _bdf_test_utils import thermoelasticity_callback

np.random.seed(12345)

base_dir = os.path.dirname(os.path.abspath(__file__))
bdf_filename = os.path.join(base_dir, "input_files", "test_bdf_file.bdf")
comm = MPI.COMM_WORLD
ntacs_procs = 1
complex_mode = TransferScheme.dtype == complex
results_folder, output_dir = test_directories(comm, base_dir)


class MultiScenarioTacsTest(unittest.TestCase):
    FILENAME = "testfake+tacs-multiscenario.txt"
    FILEPATH = os.path.join(results_folder, FILENAME)

    def test_aerothermoelastic_structDV(self):
        # build a funtofem model and body
        model = FUNtoFEMmodel("wedge")
        plate = Body.aerothermoelastic("plate")
        Variable.structural("thickness").set_bounds(
            lower=0.001, value=0.1, upper=2.0
        ).register_to(plate)
        plate.register_to(model)

        # make two scenarios, each with same functions (intend to have identical functions in general case)
        # make the first scenario
        test_scenario1 = Scenario.steady("test1", steps=20)
        Function.ksfailure().register_to(test_scenario1)
        Function.mass().register_to(test_scenario1)
        test_scenario1.register_to(model)

        # make the second scenario
        test_scenario2 = Scenario.steady("test2", steps=20)
        Function.ksfailure().register_to(test_scenario2)
        Function.mass().register_to(test_scenario2)
        test_scenario2.register_to(model)

        # build the solvers and driver
        solvers = SolverManager(comm)
        solvers.flow = TestAerodynamicSolver(comm, model)
        solvers.structural = TacsInterface.create_from_bdf(
            model,
            comm,
            ntacs_procs,
            bdf_filename,
            callback=thermoelasticity_callback,
            output_dir=output_dir,
        )
        driver = FUNtoFEMnlbgs(
            solvers, transfer_settings=TransferSettings(npts=10), model=model
        )

        # build a test result to run a complex step test
        max_rel_error = TestResult.derivative_test(
            "testfake+tacs-multiscenarios-structDV",
            model,
            driver,
            self.FILEPATH,
            complex_mode=complex_mode,
        )
        rtol = 1e-9 if complex_mode else 1e-3
        self.assertTrue(max_rel_error < rtol)
        return

    def test_aerothermoelastic_aeroDV(self):
        # build a funtofem model and body
        model = FUNtoFEMmodel("wedge")
        plate = Body.aerothermoelastic("plate")
        Variable.structural("thickness").set_bounds(
            lower=0.001, value=0.1, upper=2.0
        ).register_to(plate)
        plate.register_to(model)

        # make two scenarios, each with same functions (intend to have identical functions in general case)
        # make the first scenario
        test_scenario1 = Scenario.steady("test1", steps=20)
        Function.ksfailure().register_to(test_scenario1)
        Function.mass().register_to(test_scenario1)
        test_scenario1.register_to(model)

        # make the second scenario
        test_scenario2 = Scenario.steady("test2", steps=20)
        Function.ksfailure().register_to(test_scenario2)
        Function.mass().register_to(test_scenario2)
        test_scenario2.register_to(model)

        # build the solvers and driver
        solvers = SolverManager(comm)
        solvers.flow = TestAerodynamicSolver(comm, model)
        solvers.structural = TacsInterface.create_from_bdf(
            model,
            comm,
            ntacs_procs,
            bdf_filename,
            callback=thermoelasticity_callback,
            output_dir=output_dir,
        )

        driver = FUNtoFEMnlbgs(
            solvers, transfer_settings=TransferSettings(npts=5), model=model
        )

        # build a test result to run a complex step test
        max_rel_error = TestResult.derivative_test(
            "testfake+tacs-multiscenarios-aeroDV",
            model,
            driver,
            self.FILEPATH,
            complex_mode=complex_mode,
        )
        rtol = 1e-9 if complex_mode else 1e-3
        self.assertTrue(max_rel_error < rtol)
        return


if __name__ == "__main__":
    if comm.rank == 0:
        open(MultiScenarioTacsTest.FILEPATH, "w").close()
    unittest.main()
