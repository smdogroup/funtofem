import os, numpy as np, unittest
from tacs import TACS
from mpi4py import MPI
from funtofem import TransferScheme

from funtofem.model import FUNtoFEMmodel, Variable, Scenario, Body, Function
from funtofem.interface import (
    TestAerodynamicSolver,
    TacsInterface,
    TacsIntegrationSettings,
    SolverManager,
    TestResult,
)
from funtofem.driver import FUNtoFEMnlbgs, TransferSettings
from _bdf_test_utils import elasticity_callback, thermoelasticity_callback

np.random.seed(123456)

base_dir = os.path.dirname(os.path.abspath(__file__))
bdf_filename = os.path.join(base_dir, "input_files", "test_bdf_file.bdf")
output_folder = os.path.join(base_dir, "output")
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

comm = MPI.COMM_WORLD

results_folder = os.path.join(base_dir, "results")
if comm.rank == 0:  # make the results folder if doesn't exist
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)

ntacs_procs = 1
complex_mode = TransferScheme.dtype == complex and TACS.dtype == complex
in_github_workflow = bool(os.getenv("GITHUB_ACTIONS"))

# user settings
steps = 10
dt = 1.0
thickness = 0.01
elastic_scheme = "meld"

print(f"complex mode = {complex_mode}")


@unittest.skipIf(not complex_mode, "finite diff test buggy")
class TestNoAdjoint(unittest.TestCase):
    FILENAME = "testaero-tacs-unsteady-adj-false.txt"
    FILEPATH = os.path.join(results_folder, FILENAME)

    def test_aeroelastic(self):
        # Build the model
        model = FUNtoFEMmodel("wedge")
        plate = Body.aeroelastic("plate")
        Variable.structural("thickness").set_bounds(
            lower=0.01, value=thickness, upper=2.0
        ).register_to(plate)
        plate.register_to(model)
        test_scenario = Scenario.unsteady("test", steps=steps)
        Function.ksfailure(ks_weight=10.0).register_to(test_scenario)
        Function.test_aero().register_to(test_scenario)
        Function.mass().register_to(test_scenario)
        TacsIntegrationSettings(dt=dt, num_steps=test_scenario.steps).register_to(
            test_scenario
        )
        test_scenario.register_to(model)

        solvers = SolverManager(comm)
        solvers.structural = TacsInterface.create_from_bdf(
            model,
            comm,
            ntacs_procs,
            bdf_filename,
            callback=elasticity_callback,
            output_dir=output_folder,
        )
        solvers.flow = TestAerodynamicSolver(comm, model, copy_struct_mesh=True)

        # instantiate the driver
        transfer_settings = TransferSettings(npts=10, elastic_scheme=elastic_scheme)
        driver = FUNtoFEMnlbgs(
            solvers, transfer_settings=transfer_settings, model=model
        )

        max_rel_error = TestResult.derivative_test(
            "testaero+tacs-aeroelastic-unsteady",
            model,
            driver,
            self.FILEPATH,
            complex_mode=complex_mode,
        )
        rtol = 1e-6 if complex_mode else 1e-4
        self.assertTrue(max_rel_error < rtol)
        return

    @unittest.skipIf(in_github_workflow, "still working on aerothermal")
    def test_aerothermal(self):
        # Build the model
        model = FUNtoFEMmodel("wedge")
        plate = Body.aerothermal("plate")
        Variable.structural("thickness").set_bounds(
            lower=0.01, value=thickness, upper=2.0
        ).register_to(plate)
        plate.register_to(model)
        test_scenario = Scenario.unsteady("test", steps=steps)
        Function.temperature().register_to(test_scenario)
        Function.test_aero().register_to(test_scenario)
        TacsIntegrationSettings(dt=dt, num_steps=test_scenario.steps).register_to(
            test_scenario
        )
        test_scenario.register_to(model)

        solvers = SolverManager(comm)
        solvers.structural = TacsInterface.create_from_bdf(
            model,
            comm,
            ntacs_procs,
            bdf_filename,
            callback=thermoelasticity_callback,
            output_dir=output_folder,
        )
        solvers.flow = TestAerodynamicSolver(comm, model, copy_struct_mesh=True)

        # instantiate the driver
        transfer_settings = TransferSettings(npts=10)
        driver = FUNtoFEMnlbgs(
            solvers, transfer_settings=transfer_settings, model=model
        )

        max_rel_error = TestResult.derivative_test(
            "testaero+tacs-aerothermal-unsteady",
            model,
            driver,
            self.FILEPATH,
            complex_mode=complex_mode,
        )
        rtol = 1e-6 if complex_mode else 1e-3
        self.assertTrue(max_rel_error < rtol)
        return

    @unittest.skipIf(in_github_workflow, "still working on aerothermal")
    def test_aerothermoelastic(self):
        # Build the model
        model = FUNtoFEMmodel("wedge")
        plate = Body.aerothermoelastic("plate")
        Variable.structural("thickness").set_bounds(
            lower=0.01, value=thickness, upper=2.0
        ).register_to(plate)
        plate.register_to(model)
        test_scenario = Scenario.unsteady("test", steps=steps)
        Function.ksfailure(ks_weight=10.0).register_to(test_scenario)
        Function.temperature().register_to(test_scenario)
        Function.test_aero().register_to(test_scenario)
        TacsIntegrationSettings(dt=dt, num_steps=test_scenario.steps).register_to(
            test_scenario
        )
        test_scenario.register_to(model)

        solvers = SolverManager(comm)
        solvers.structural = TacsInterface.create_from_bdf(
            model,
            comm,
            ntacs_procs,
            bdf_filename,
            callback=thermoelasticity_callback,
            output_dir=output_folder,
        )
        solvers.flow = TestAerodynamicSolver(comm, model, copy_struct_mesh=True)

        # instantiate the driver
        transfer_settings = TransferSettings(npts=10, elastic_scheme=elastic_scheme)
        driver = FUNtoFEMnlbgs(
            solvers, transfer_settings=transfer_settings, model=model
        )

        max_rel_error = TestResult.derivative_test(
            "testaero+tacs-aerothermoelastic-unsteady",
            model,
            driver,
            self.FILEPATH,
            complex_mode=complex_mode,
        )
        rtol = 1e-6 if complex_mode else 1e-3
        self.assertTrue(max_rel_error < rtol)
        return

    @unittest.skipIf(in_github_workflow, "still working on aerothermal")
    def test_multiscenario_aerothermoelastic(self):
        # Build the model
        model = FUNtoFEMmodel("wedge")
        plate = Body.aerothermoelastic("plate")
        Variable.structural("thickness").set_bounds(
            lower=0.01, value=thickness, upper=2.0
        ).register_to(plate)
        plate.register_to(model)

        # make the first scenario with ksfailure, temperature
        test_scenario1 = Scenario.unsteady("test1", steps=steps)
        Function.ksfailure(ks_weight=10.0).register_to(test_scenario1)
        Function.temperature().register_to(test_scenario1)
        Function.test_aero().register_to(test_scenario1)
        TacsIntegrationSettings(dt=dt, num_steps=test_scenario1.steps).register_to(
            test_scenario1
        )
        test_scenario1.register_to(model)

        # make the second scenario with ksfailure, temperature
        test_scenario2 = Scenario.unsteady("test2", steps=steps)
        Function.ksfailure(ks_weight=10.0).register_to(test_scenario2)
        Function.temperature().register_to(test_scenario2)
        Function.test_aero().register_to(test_scenario2)
        TacsIntegrationSettings(dt=dt, num_steps=test_scenario2.steps).register_to(
            test_scenario2
        )
        test_scenario2.register_to(model)

        solvers = SolverManager(comm)
        solvers.structural = TacsInterface.create_from_bdf(
            model,
            comm,
            ntacs_procs,
            bdf_filename,
            callback=thermoelasticity_callback,
            output_dir=output_folder,
        )
        solvers.flow = TestAerodynamicSolver(comm, model, copy_struct_mesh=True)

        # instantiate the driver
        transfer_settings = TransferSettings(npts=10, elastic_scheme=elastic_scheme)
        driver = FUNtoFEMnlbgs(
            solvers, transfer_settings=transfer_settings, model=model
        )

        max_rel_error = TestResult.derivative_test(
            "testaero+tacs-aerothermoelastic-unsteady-multiscenario",
            model,
            driver,
            self.FILEPATH,
            complex_mode=complex_mode,
        )
        rtol = 1e-6 if complex_mode else 1e-3
        self.assertTrue(max_rel_error < rtol)
        return


if __name__ == "__main__":
    if comm.rank == 0:
        open(TestNoAdjoint.FILEPATH, "w").close()  # clear file
    complex_mode = True
    unittest.main()
