import os, numpy as np, unittest
from tacs import TACS
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
ntacs_procs = 2
complex_mode = TransferScheme.dtype == complex and TACS.dtype == complex


@unittest.skipIf(
    not complex_mode,
    "parallel subtractive subtractive cancellation of FD is worse sometimes",
)
class TacsParallelFrameworkTest(unittest.TestCase):
    N_PROCS = 2
    FILENAME = "testAero-tacs.txt"

    def test_aeroelastic(self):
        model = FUNtoFEMmodel("wedge")
        plate = Body.aeroelastic("plate")
        Variable.structural("thickness").set_bounds(
            lower=0.01, value=1.0, upper=2.0
        ).register_to(plate)
        plate.register_to(model)
        test_scenario = Scenario.steady("test", steps=150).include(Function.ksfailure())
        test_scenario.register_to(model)

        solvers = SolverManager(comm)
        solvers.structural = TacsSteadyInterface.create_from_bdf(
            model, comm, ntacs_procs, bdf_filename, callback=elasticity_callback
        )
        solvers.flow = TestAerodynamicSolver(comm, model)

        driver = FUNtoFEMnlbgs(
            solvers, transfer_settings=TransferSettings(npts=5), model=model
        )

        max_rel_error = TestResult.derivative_test(
            "testaero+tacs-aeroelastic",
            model,
            driver,
            TacsParallelFrameworkTest.FILENAME,
            complex_mode=complex_mode,
        )
        rtol = 1e-7 if complex_mode else 1e-4
        self.assertTrue(max_rel_error < rtol)

        return

    def test_aerothermal(self):
        model = FUNtoFEMmodel("wedge")
        plate = Body.aerothermal("plate")
        Variable.structural("thickness").set_bounds(
            lower=0.01, value=1.0, upper=2.0
        ).register_to(plate)
        plate.register_to(model)
        test_scenario = Scenario.steady("test", steps=150).include(
            Function.temperature()
        )
        test_scenario.register_to(model)

        solvers = SolverManager(comm)
        solvers.structural = TacsSteadyInterface.create_from_bdf(
            model, comm, ntacs_procs, bdf_filename, callback=thermoelasticity_callback
        )
        solvers.flow = TestAerodynamicSolver(comm, model)

        driver = FUNtoFEMnlbgs(
            solvers, transfer_settings=TransferSettings(npts=5), model=model
        )

        max_rel_error = TestResult.derivative_test(
            "testaero+tacs-aerothermal",
            model,
            driver,
            TacsParallelFrameworkTest.FILENAME,
            complex_mode=complex_mode,
        )
        rtol = 1e-7 if complex_mode else 1e-4
        self.assertTrue(max_rel_error < rtol)

        return

    def test_aerothermoelastic(self):
        model = FUNtoFEMmodel("wedge")
        plate = Body.aerothermoelastic("plate")
        Variable.structural("thickness").set_bounds(
            lower=0.01, value=0.1, upper=2.0
        ).register_to(plate)
        plate.register_to(model)
        test_scenario = Scenario.steady("test", steps=150).include(Function.ksfailure())
        test_scenario.include(Function.temperature()).register_to(model)

        solvers = SolverManager(comm)
        solvers.structural = TacsSteadyInterface.create_from_bdf(
            model, comm, ntacs_procs, bdf_filename, callback=thermoelasticity_callback
        )
        solvers.flow = TestAerodynamicSolver(comm, model)

        driver = FUNtoFEMnlbgs(
            solvers, transfer_settings=TransferSettings(npts=10), model=model
        )

        max_rel_error = TestResult.derivative_test(
            "testaero+tacs-aerothermoelastic",
            model,
            driver,
            TacsParallelFrameworkTest.FILENAME,
            complex_mode=complex_mode,
        )
        rtol = 1e-7 if complex_mode else 1e-3
        self.assertTrue(max_rel_error < rtol)

        return


if __name__ == "__main__":
    unittest.main()
