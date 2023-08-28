import numpy as np
from mpi4py import MPI
from funtofem import TransferScheme
from tacs import TACS

from funtofem.model import FUNtoFEMmodel, Variable, Scenario, Body, Function
from funtofem.interface import (
    TestAerodynamicSolver,
    TestStructuralSolver,
    SolverManager,
    TestResult,
)
from funtofem.driver import FUNtoFEMnlbgs, TransferSettings

import unittest

complex_mode = TransferScheme.dtype == complex and TACS.dtype == complex
comm = MPI.COMM_WORLD

steps = 10


class CoupledUnsteadyFrameworkTest(unittest.TestCase):
    FILENAME = "fake-solvers-drivers.txt"

    def test_structDV_with_driver(self):
        # build the funtofem model with an unsteady scenario
        model = FUNtoFEMmodel("test")
        plate = Body.aerothermoelastic("plate")
        for iS in range(5):
            Variable.structural(f"thick{iS}").set_bounds(value=0.1).register_to(plate)
        plate.register_to(model)
        test_scenario = Scenario.unsteady("test", steps=steps)
        test_scenario.include(Function.ksfailure()).include(Function.lift())
        test_scenario.register_to(model)

        # build a funtofem driver
        solvers = SolverManager(comm)
        solvers.flow = TestAerodynamicSolver(comm, model)
        solvers.structural = TestStructuralSolver(comm, model)
        driver = FUNtoFEMnlbgs(
            solvers, transfer_settings=TransferSettings(npts=10), model=model
        )

        rtol = 1e-7 if complex_mode else 1e-4
        max_rel_error = TestResult.derivative_test(
            "fake_solvers-structDV",
            model,
            driver,
            CoupledUnsteadyFrameworkTest.FILENAME,
            complex_mode,
        )
        self.assertTrue(max_rel_error < rtol)
        return

    def test_aeroDV_with_driver(self):
        # build the funtofem model with an unsteady scenario
        model = FUNtoFEMmodel("test")
        plate = Body.aerothermoelastic("plate")
        for iA in range(5):
            Variable.aerodynamic(f"aero{iA}").set_bounds(value=0.1).register_to(plate)
        plate.register_to(model)
        test_scenario = Scenario.unsteady("test", steps=steps)
        test_scenario.include(Function.ksfailure()).include(Function.lift())
        test_scenario.register_to(model)

        # build a funtofem driver
        solvers = SolverManager(comm)
        solvers.flow = TestAerodynamicSolver(comm, model)
        solvers.structural = TestStructuralSolver(comm, model)
        driver = FUNtoFEMnlbgs(
            solvers, transfer_settings=TransferSettings(npts=10), model=model
        )

        rtol = 1e-7 if complex_mode else 1e-4
        max_rel_error = TestResult.derivative_test(
            "fake_solvers-aeroDV",
            model,
            driver,
            CoupledUnsteadyFrameworkTest.FILENAME,
            complex_mode,
        )
        self.assertTrue(max_rel_error < rtol)
        return


if __name__ == "__main__":
    if comm.rank == 0:
        open(CoupledUnsteadyFrameworkTest.FILENAME, "w").close()  # clear file
    unittest.main()
