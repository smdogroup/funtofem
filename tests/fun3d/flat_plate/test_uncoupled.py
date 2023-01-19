import os, numpy as np, unittest
from mpi4py import MPI

# from funtofem import TransferScheme
from pyfuntofem.model import FUNtoFEMmodel, Variable, Scenario, Body, Function
from pyfuntofem.interface import (
    Fun3dInterface,
    TestStructuralSolver,
    SolverManager,
    TestResult,
)
from pyfuntofem.driver import FUNtoFEMnlbgs, TransferSettings

np.random.seed(1234567)


class TestFun3dUncoupled(unittest.TestCase):
    FILENAME = "fun3d-fake-laminar"

    def test_laminar_aeroelastic(self):
        # build the funtofem model with one body and scenario
        model = FUNtoFEMmodel("plate")
        plate = Body.aeroelastic("plate", boundary=6)
        Variable.structural("thickness").set_bounds(
            lower=0.001, value=0.01, upper=2.0
        ).register_to(plate)
        plate.register_to(model)
        test_scenario = Scenario.steady("laminar1", steps=200).set_temperature(
            T_ref=300.0, T_inf=300.0
        )
        # test_scenario.include(Function.ksfailure(ks_weight=50.0))
        test_scenario.include(Function.lift())
        test_scenario.register_to(model)

        # build the solvers and coupled driver
        comm = MPI.COMM_WORLD
        solvers = SolverManager(comm)
        solvers.flow = Fun3dInterface(comm, model).set_units(qinf=1.0e2)
        solvers.structural = TestStructuralSolver(comm, model, elastic_k=1000.0)
        transfer_settings = TransferSettings()
        driver = FUNtoFEMnlbgs(
            solvers, transfer_settings=transfer_settings, model=model
        )

        # run the complex step test on the model and driver
        max_rel_error = TestResult.complex_step(
            "fun3d+fake-laminar-aeroelastic", model, driver, TestFun3dUncoupled.FILENAME
        )
        self.assertTrue(max_rel_error < 1e-7)

    def _laminar_aerothermal(self):
        # build the funtofem model with one body and scenario
        model = FUNtoFEMmodel("plate")
        plate = Body.aerothermal("plate", boundary=6)
        Variable.structural("thickness").set_bounds(
            lower=0.001, value=0.01, upper=2.0
        ).register_to(plate)
        plate.register_to(model)
        test_scenario = Scenario.steady("laminar2", steps=500).set_temperature(
            T_ref=300.0, T_inf=300.0
        )
        test_scenario.include(Function.temperature())
        test_scenario.register_to(model)

        # build the solvers and coupled driver
        comm = MPI.COMM_WORLD
        solvers = SolverManager(comm)
        solvers.flow = Fun3dInterface(comm, model)
        solvers.structural = TestStructuralSolver(comm, model, thermal_k=1.0)
        transfer_settings = TransferSettings(npts=5)
        driver = FUNtoFEMnlbgs(
            solvers, transfer_settings=transfer_settings, model=model
        )

        # run the complex step test on the model and driver
        self._complex_step_check(model, driver)

    def _laminar_aerothermoelastic(self):
        # build the funtofem model with one body and scenario
        model = FUNtoFEMmodel("plate")
        plate = Body.aerothermoelastic("plate", boundary=1)
        Variable.structural("thickness").set_bounds(
            lower=0.001, value=0.01, upper=2.0
        ).register_to(plate)
        plate.register_to(model)
        test_scenario = Scenario.steady("laminar2", steps=500).set_temperature(
            T_ref=300.0, T_inf=300.0
        )
        test_scenario.include(Function.ksfailure())
        test_scenario.register_to(model)

        # build the solvers and coupled driver
        comm = MPI.COMM_WORLD
        solvers = SolverManager(comm)
        solvers.flow = Fun3dInterface(comm, model).set_units(qinf=1.0e2)
        solvers.structural = TestStructuralSolver(
            comm, model, elastic_k=1000.0, thermal_k=1.0
        )
        transfer_settings = TransferSettings(npts=5)
        driver = FUNtoFEMnlbgs(
            solvers, transfer_settings=transfer_settings, model=model
        )

        # run the complex step test on the model and driver
        self._complex_step_check(model, driver)


if __name__ == "__main__":
    unittest.main()
