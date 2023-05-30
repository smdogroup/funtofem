import os
from tacs import TACS
from mpi4py import MPI
import numpy as np
from funtofem import TransferScheme

from funtofem.model import FUNtoFEMmodel, Variable, Scenario, Body, Function
from funtofem.interface import (
    TestAerodynamicSolver,
    TacsSteadyInterface,
    SolverManager,
    TestResult,
)
from funtofem.driver import TacsOnewayDriver, TransferSettings, FUNtoFEMnlbgs
import unittest

np.random.seed(123456)

base_dir = os.path.dirname(os.path.abspath(__file__))
loaded_mesh = os.path.join(base_dir, "input_files", "loaded_plate.dat")

complex_mode = TransferScheme.dtype == complex and TACS.dtype == complex
nprocs = 1
comm = MPI.COMM_WORLD

results_folder = os.path.join(base_dir, "results")
if comm.rank == 0:  # make the results folder if doesn't exist
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)


@unittest.skipIf(not complex_mode, "Don't want to do real-mode finite difference here.")
class TacsConstLoadTest(unittest.TestCase):
    FILENAME = "tacs-const-load.txt"
    FILEPATH = os.path.join(results_folder, FILENAME)

    def test_gridforce_aeroelastic(self):
        """test a constant hanging load with aeroelastic coupling"""
        # Build the model
        model = FUNtoFEMmodel("wedge")
        plate = Body.aeroelastic("plate")
        svar = Variable.structural("face2", value=0.1).register_to(plate)
        plate.register_to(model)

        # Create a scenario to run
        steady = Scenario.steady("test", steps=10)
        ksfailure = Function.ksfailure()
        steady.include(ksfailure)
        steady.register_to(model)

        # Build the solver interfaces
        solvers = SolverManager(comm)
        solvers.structural = TacsSteadyInterface.create_from_bdf(
            model, comm, nprocs, loaded_mesh, callback=None
        )
        solvers.flow = TestAerodynamicSolver(comm, model)

        # random struct loads
        funtofem_driver = FUNtoFEMnlbgs(
            solvers, transfer_settings=TransferSettings(npts=10, beta=1.0), model=model
        )

        epsilon = 1e-30 if complex_mode else 1e-5
        rtol = 1e-9 if complex_mode else 1e-4
        max_rel_error = TestResult.derivative_test(
            "tacs+testaero-aeroelastic",
            model,
            funtofem_driver,
            TacsConstLoadTest.FILEPATH,
            complex_mode,
            epsilon,
        )
        self.assertTrue(max_rel_error < rtol)

        return


if __name__ == "__main__":
    if comm.rank == 0:
        open(TacsConstLoadTest.FILENAME, "w").close()  # clear file
    unittest.main()
