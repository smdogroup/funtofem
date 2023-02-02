import os, unittest, numpy as np
from mpi4py import MPI
from tacs import TACS
from funtofem import TransferScheme
from pyfuntofem.model import FUNtoFEMmodel, Variable, Scenario, Body, Function
from pyfuntofem.interface import (
    TestAerodynamicSolver,
    TacsSteadyInterface,
    SolverManager,
    TestResult,
)
from pyfuntofem.driver import FUNtoFEMnlbgs, TransferSettings, TacsSteadyAnalysisDriver

from bdf_test_utils import thermoelasticity_callback, elasticity_callback

np.random.seed(1234567)

base_dir = os.path.dirname(os.path.abspath(__file__))
bdf_filename = os.path.join(base_dir, "input_files", "test_bdf_file.bdf")


class TestOnewayDriver(unittest.TestCase):

    """
    This class performs unit test on the oneway-coupled TacsSteadyAnalysisDriver
    which uses fixed aero loads
    TODO : in the case of an unsteady one, add methods for those too?
    """

    FILENAME = "oneway-driver.txt"

    def test_aeroelastic(self):
        # build the model and driver
        model = FUNtoFEMmodel("wedge")
        plate = Body.aeroelastic("plate", boundary=1)
        Variable.structural("thickness").set_bounds(
            lower=0.01, value=0.1, upper=1.0
        ).register_to(plate)
        plate.register_to(model)

        # build the scenario
        scenario = Scenario.steady("test", steps=200).include(Function.ksfailure())
        scenario.register_to(model)

        # build the tacs interface, coupled driver, and oneway driver
        comm = MPI.COMM_WORLD
        solvers = SolverManager(comm)
        solvers.flow = TestAerodynamicSolver(comm, model)
        solvers.structural = TacsSteadyInterface.create_from_bdf(
            model, comm, 1, bdf_filename, callback=elasticity_callback
        )
        transfer_settings = TransferSettings(npts=5)
        coupled_driver = FUNtoFEMnlbgs(
            solvers, transfer_settings=transfer_settings, model=model
        )
        oneway_driver = TacsSteadyAnalysisDriver(solvers.structural, model)

        # prime the oneway driver by running one forward analysis of coupled driver
        coupled_driver.solve_forward()

        complex_mode = False
        rtol = 1e-4
        if TransferScheme.dtype == complex and TACS.dtype == complex:
            complex_mode = True
            rtol = 1e-7

        # run teh oomplex step test
        max_rel_error = TestResult.derivative_test(
            "oneway-aeroelastic",
            model,
            oneway_driver,
            TestOnewayDriver.FILENAME,
            has_fun3d=False,
            complex_mode=complex_mode,
        )
        self.assertTrue(max_rel_error < rtol)

        return

    def test_aerothermal(self):
        # build the model and driver
        model = FUNtoFEMmodel("wedge")
        plate = Body.aerothermal("plate", boundary=1)
        Variable.structural("thickness").set_bounds(
            lower=0.01, value=0.1, upper=1.0
        ).register_to(plate)
        plate.register_to(model)

        # build the scenario
        scenario = (
            Scenario.steady("test", steps=150)
            .include(Function.ksfailure())
            .include(Function.temperature())
        )
        scenario.register_to(model)

        # build the tacs interface, coupled driver, and oneway driver
        comm = MPI.COMM_WORLD
        solvers = SolverManager(comm)
        solvers.flow = TestAerodynamicSolver(comm, model)
        solvers.structural = TacsSteadyInterface.create_from_bdf(
            model, comm, 1, bdf_filename, callback=thermoelasticity_callback
        )
        transfer_settings = TransferSettings(npts=5)
        coupled_driver = FUNtoFEMnlbgs(
            solvers, transfer_settings=transfer_settings, model=model
        )
        oneway_driver = TacsSteadyAnalysisDriver(solvers.structural, model)

        # prime the oneway driver by running one forward analysis of coupled driver
        # to obtain fixed aero loads
        coupled_driver.solve_forward()

        complex_mode = False
        rtol = 1e-3
        if TransferScheme.dtype == complex and TACS.dtype == complex:
            complex_mode = True
            rtol = 1e-7

        # run teh oomplex step test
        max_rel_error = TestResult.derivative_test(
            "oneway-aerothermal",
            model,
            oneway_driver,
            TestOnewayDriver.FILENAME,
            has_fun3d=False,
            complex_mode=complex_mode,
        )
        self.assertTrue(max_rel_error < rtol)

        return

    def test_aerothermoelastic(self):
        # build the model and driver
        model = FUNtoFEMmodel("wedge")
        plate = Body.aerothermoelastic("plate", boundary=1)
        Variable.structural("thickness").set_bounds(
            lower=0.01, value=0.1, upper=1.0
        ).register_to(plate)
        plate.register_to(model)

        # build the scenario
        scenario = (
            Scenario.steady("test", steps=150)
            .include(Function.ksfailure())
            .include(Function.temperature())
        )
        scenario.register_to(model)

        # build the tacs interface, coupled driver, and oneway driver
        comm = MPI.COMM_WORLD
        solvers = SolverManager(comm)
        solvers.flow = TestAerodynamicSolver(comm, model)
        solvers.structural = TacsSteadyInterface.create_from_bdf(
            model, comm, 1, bdf_filename, callback=thermoelasticity_callback
        )
        transfer_settings = TransferSettings(npts=5)
        coupled_driver = FUNtoFEMnlbgs(
            solvers, transfer_settings=transfer_settings, model=model
        )
        oneway_driver = TacsSteadyAnalysisDriver(solvers.structural, model)

        # prime the oneway driver by running one forward analysis of coupled driver
        # to obtain fixed aero loads
        coupled_driver.solve_forward()

        complex_mode = False
        rtol = 1e-3
        if TransferScheme.dtype == complex and TACS.dtype == complex:
            complex_mode = True
            rtol = 1e-7

        # run teh oomplex step test
        max_rel_error = TestResult.derivative_test(
            "oneway-aerothermoelastic",
            model,
            oneway_driver,
            TestOnewayDriver.FILENAME,
            has_fun3d=False,
            complex_mode=complex_mode,
        )
        self.assertTrue(max_rel_error < rtol)
        return


if __name__ == "__main__":
    open(TestOnewayDriver.FILENAME, "w").close()
    unittest.main()
