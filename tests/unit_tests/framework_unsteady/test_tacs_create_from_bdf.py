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
    make_test_directories,
)
from funtofem.driver import FUNtoFEMnlbgs, TransferSettings

np.random.seed(123456)

base_dir = os.path.dirname(os.path.abspath(__file__))
bdf_filename = os.path.join(base_dir, "input_files", "stiffened_plate.dat")
comm = MPI.COMM_WORLD

results_folder, output_folder = make_test_directories(comm, base_dir)

ntacs_procs = 1
complex_mode = TransferScheme.dtype == complex and TACS.dtype == complex
in_github_workflow = bool(os.getenv("GITHUB_ACTIONS"))

steps = 2
dt = 1.0
thickness = 0.01


@unittest.skipIf(not complex_mode, "finite diff test buggy")
class TacsUnsteadyFrameworkTest(unittest.TestCase):
    FILENAME = "testaero-tacs-unsteady.txt"
    FILEPATH = os.path.join(results_folder, FILENAME)

    def test_aeroelastic(self):
        # Build the model
        model = FUNtoFEMmodel("wedge")
        plate = Body.aeroelastic("plate")
        Variable.structural("thick").set_bounds(
            lower=0.01, value=thickness, upper=2.0
        ).register_to(plate)
        plate.register_to(model)
        test_scenario = Scenario.unsteady("test", steps=steps)
        Function.ksfailure(ks_weight=10.0).register_to(test_scenario)
        Function.lift().register_to(test_scenario)
        integration_settings = TacsIntegrationSettings(
            dt=dt, num_steps=test_scenario.steps
        )
        test_scenario.include(integration_settings)
        test_scenario.register_to(model)

        solvers = SolverManager(comm)
        solvers.flow = TestAerodynamicSolver(comm, model)
        solvers.structural = TacsInterface.create_from_bdf(
            model,
            comm,
            ntacs_procs,
            bdf_filename,
            output_dir=output_folder,
        )
        # solvers.flow = TestAerodynamicSolver(comm, model, copy_struct_mesh=True)

        # instantiate the driver
        driver = FUNtoFEMnlbgs(
            solvers, transfer_settings=TransferSettings(npts=5), model=model
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


if __name__ == "__main__":
    if comm.rank == 0:
        open(TacsUnsteadyFrameworkTest.FILEPATH, "w").close()  # clear file
    unittest.main()
