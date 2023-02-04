import os, numpy as np, unittest
from tacs import TACS
from mpi4py import MPI
from funtofem import TransferScheme

from pyfuntofem.model import FUNtoFEMmodel, Variable, Scenario, Body, Function
from pyfuntofem.interface import (
    TestAerodynamicSolver,
    TacsUnsteadyInterface,
    IntegrationSettings,
    SolverManager,
    TestResult,
)
from pyfuntofem.driver import FUNtoFEMnlbgs, TransferSettings
from bdf_test_utils import elasticity_callback, thermoelasticity_callback

np.random.seed(1234567)

base_dir = os.path.dirname(os.path.abspath(__file__))
bdf_filename = os.path.join(base_dir, "input_files", "test_bdf_file.bdf")
tacs_folder = os.path.join(base_dir, "tacs")
if not os.path.exists(tacs_folder):
    os.mkdir(tacs_folder)

comm = MPI.COMM_WORLD
ntacs_procs = 1

class TacsUnsteadyFrameworkTest(unittest.TestCase):
    FILENAME = "testaero-tacs-unsteady.txt"

    def test_aeroelastic(self):
        # Build the model
        model = FUNtoFEMmodel("wedge")
        plate = Body.aeroelastic("plate")
        Variable.structural("thickness").set_bounds(
            lower=0.01, value=1.0, upper=2.0
        ).register_to(plate)
        plate.register_to(model)
        test_scenario = Scenario.steady("test", steps=150).include(Function.ksfailure())
        test_scenario.register_to(model)

        integration_settings = IntegrationSettings(dt=0.01, num_steps=150)

        solvers = SolverManager(comm)
        solvers.structural = TacsUnsteadyInterface.create_from_bdf(
            model,
            comm,
            ntacs_procs,
            bdf_filename,
            callback=elasticity_callback,
            integration_settings=integration_settings,
            output_dir=tacs_folder,
        )
        solvers.flow = TestAerodynamicSolver(comm, model)

        # instantiate the driver
        driver = FUNtoFEMnlbgs(
            solvers, transfer_settings=TransferSettings(npts=5), model=model
        )

        complex_mode = TransferScheme.dtype == complex and TACS.dtype == complex
        max_rel_error = TestResult.derivative_test(
            "testaero+tacs-aeroelastic-unsteady",
            model,
            driver,
            TacsUnsteadyFrameworkTest.FILENAME,
            complex_mode=complex_mode,
        )
        rtol = 1e-7 if complex_mode else 1e-4
        self.assertTrue(max_rel_error < rtol)
        return

    def test_aerothermal(self):
        # Build the model
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

        integration_settings = IntegrationSettings(dt=0.01, num_steps=150)

        solvers = SolverManager(comm)
        solvers.structural = TacsUnsteadyInterface.create_from_bdf(
            model,
            comm,
            ntacs_procs,
            bdf_filename,
            callback=thermoelasticity_callback,
            integration_settings=integration_settings,
            output_dir=tacs_folder,
        )
        solvers.flow = TestAerodynamicSolver(comm, model)

        # instantiate the driver
        driver = FUNtoFEMnlbgs(
            solvers, transfer_settings=TransferSettings(npts=5), model=model
        )

        complex_mode = TransferScheme.dtype == complex and TACS.dtype == complex
        max_rel_error = TestResult.derivative_test(
            "testaero+tacs-aerothermal-unsteady",
            model,
            driver,
            TacsUnsteadyFrameworkTest.FILENAME,
            complex_mode=complex_mode,
        )
        rtol = 1e-7 if complex_mode else 1e-4
        self.assertTrue(max_rel_error < rtol)
        return

    def test_aerothermoelastic(self):
        # Build the model
        model = FUNtoFEMmodel("wedge")
        plate = Body.aerothermoelastic("plate")
        Variable.structural("thickness").set_bounds(
            lower=0.01, value=1.0, upper=2.0
        ).register_to(plate)
        plate.register_to(model)
        test_scenario = Scenario.steady("test", steps=150).include(Function.ksfailure())
        test_scenario.include(Function.temperature()).register_to(model)

        integration_settings = IntegrationSettings(dt=0.01, num_steps=150)

        solvers = SolverManager(comm)
        solvers.structural = TacsUnsteadyInterface.create_from_bdf(
            model,
            comm,
            ntacs_procs,
            bdf_filename,
            callback=thermoelasticity_callback,
            integration_settings=integration_settings,
            output_dir=tacs_folder,
        )
        solvers.flow = TestAerodynamicSolver(comm, model)

        # instantiate the driver
        driver = FUNtoFEMnlbgs(
            solvers, transfer_settings=TransferSettings(npts=5), model=model
        )

        complex_mode = TransferScheme.dtype == complex and TACS.dtype == complex
        max_rel_error = TestResult.derivative_test(
            "testaero+tacs-aerothermoelastic-unsteady",
            model,
            driver,
            TacsUnsteadyFrameworkTest.FILENAME,
            complex_mode=complex_mode,
        )
        rtol = 1e-7 if complex_mode else 1e-4
        self.assertTrue(max_rel_error < rtol)
        return


if __name__ == "__main__":
    unittest.main()
