import os
from tacs import TACS
from mpi4py import MPI
import numpy as np
from funtofem import TransferScheme

from funtofem.model import FUNtoFEMmodel, Variable, Scenario, Body, Function
from funtofem.interface import (
    TestAerodynamicSolver,
    TacsInterface,
    SolverManager,
    TacsIntegrationSettings,
    CoordinateDerivativeTester,
)
from funtofem.driver import TransferSettings, FUNtoFEMnlbgs

from bdf_test_utils import elasticity_callback, thermoelasticity_callback
import unittest

np.random.seed(123456)

base_dir = os.path.dirname(os.path.abspath(__file__))
bdf_filename = os.path.join(base_dir, "input_files", "test_bdf_file.bdf")

complex_mode = TransferScheme.dtype == complex and TACS.dtype == complex
nprocs = 1
comm = MPI.COMM_WORLD

results_folder = os.path.join(base_dir, "results")
if comm.rank == 0:  # make the results folder if doesn't exist
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)

in_github_workflow = bool(os.getenv("GITHUB_ACTIONS"))


@unittest.skipIf(not complex_mode, "some finite differences near 1e-2 others great")
class TestFuntofemDriverStructCoordinate(unittest.TestCase):
    FILENAME = "f2f-steady-struct-coord.txt"
    FILEPATH = os.path.join(results_folder, FILENAME)

    def test_steady_struct_aeroelastic(self):
        # build the model and driver
        model = FUNtoFEMmodel("wedge")
        plate = Body.aeroelastic("plate", boundary=1)
        Variable.structural("thickness").set_bounds(
            lower=0.01, value=0.1, upper=1.0
        ).register_to(plate)
        plate.register_to(model)

        # build the scenario
        scenario = Scenario.steady("test", steps=200).include(Function.ksfailure())
        scenario.include(Function.drag()).include(Function.lift())
        scenario.register_to(model)

        # build the tacs interface, coupled driver, and oneway driver
        comm = MPI.COMM_WORLD
        solvers = SolverManager(comm)
        solvers.flow = TestAerodynamicSolver(comm, model)
        solvers.structural = TacsInterface.create_from_bdf(
            model, comm, 1, bdf_filename, callback=elasticity_callback
        )
        transfer_settings = TransferSettings(npts=5)
        coupled_driver = FUNtoFEMnlbgs(
            solvers, transfer_settings=transfer_settings, model=model
        )

        epsilon = 1e-30 if complex_mode else 1e-4
        rtol = 1e-6 if complex_mode else 1e-5

        """complex step test over coordinate derivatives"""
        tester = CoordinateDerivativeTester(coupled_driver)
        rel_error = tester.test_struct_coordinates(
            "funtofem_driver struct coordinate derivatives steady-aeroelastic",
            status_file=self.FILEPATH,
            epsilon=epsilon,
            complex_mode=complex_mode,
        )
        assert abs(rel_error < rtol)
        return

    def test_steady_struct_aerothermal(self):
        # build the model and driver
        model = FUNtoFEMmodel("wedge")
        plate = Body.aerothermal("plate", boundary=1)
        Variable.structural("thickness").set_bounds(
            lower=0.01, value=0.1, upper=1.0
        ).register_to(plate)
        plate.register_to(model)

        # build the scenario
        scenario = Scenario.steady("test", steps=200)
        scenario.include(Function.temperature())
        scenario.include(Function.drag()).include(Function.lift())
        scenario.register_to(model)

        # build the tacs interface, coupled driver, and oneway driver
        comm = MPI.COMM_WORLD
        solvers = SolverManager(comm)
        solvers.flow = TestAerodynamicSolver(comm, model)
        solvers.structural = TacsInterface.create_from_bdf(
            model, comm, 1, bdf_filename, callback=thermoelasticity_callback
        )
        transfer_settings = TransferSettings(npts=5)
        coupled_driver = FUNtoFEMnlbgs(
            solvers, transfer_settings=transfer_settings, model=model
        )

        epsilon = 1e-30 if complex_mode else 1e-4
        rtol = 1e-9 if complex_mode else 1e-5

        """complex step test over coordinate derivatives"""
        tester = CoordinateDerivativeTester(coupled_driver)
        rel_error = tester.test_struct_coordinates(
            "funtofem_driver struct coordinate derivatives steady-aerothermal",
            status_file=self.FILEPATH,
            epsilon=epsilon,
            complex_mode=complex_mode,
        )
        assert abs(rel_error) < rtol
        return

    def test_steady_struct_aerothermoelastic(self):
        # build the model and driver
        model = FUNtoFEMmodel("wedge")
        plate = Body.aerothermoelastic("plate", boundary=1)
        Variable.structural("thickness").set_bounds(
            lower=0.01, value=0.1, upper=1.0
        ).register_to(plate)
        plate.register_to(model)

        # build the scenario
        scenario = Scenario.steady("test", steps=200).include(Function.temperature())
        scenario.include(Function.ksfailure())
        scenario.include(Function.drag()).include(Function.lift())
        scenario.register_to(model)

        # build the tacs interface, coupled driver, and oneway driver
        comm = MPI.COMM_WORLD
        solvers = SolverManager(comm)
        solvers.flow = TestAerodynamicSolver(comm, model)
        solvers.structural = TacsInterface.create_from_bdf(
            model, comm, 1, bdf_filename, callback=thermoelasticity_callback
        )
        transfer_settings = TransferSettings(npts=5)
        coupled_driver = FUNtoFEMnlbgs(
            solvers, transfer_settings=transfer_settings, model=model
        )

        epsilon = 1e-30 if complex_mode else 1e-4
        rtol = 1e-9 if complex_mode else 1e-5

        """complex step test over coordinate derivatives"""
        tester = CoordinateDerivativeTester(coupled_driver)
        rel_error = tester.test_struct_coordinates(
            "funtofem_driver struct coordinate derivatives steady-aerothermoelastic",
            status_file=self.FILEPATH,
            epsilon=epsilon,
            complex_mode=complex_mode,
        )
        assert abs(rel_error) < rtol
        return

    @unittest.skip("need to fix multi-scenario coord test")
    def test_steady_struct_multiscenario_aerothermoelastic(self):
        # build the model and driver
        model = FUNtoFEMmodel("wedge")
        plate = Body.aerothermoelastic("plate", boundary=1)
        Variable.structural("thickness").set_bounds(
            lower=0.01, value=0.1, upper=1.0
        ).register_to(plate)
        plate.register_to(model)

        # build the scenario
        cruise = Scenario.steady("cruise", steps=200)
        Function.lift().register_to(cruise)
        Function.drag().register_to(cruise)
        Function.mass().register_to(cruise)
        cruise.register_to(model)

        climb = Scenario.steady("climb", steps=200)
        Function.ksfailure().register_to(climb)
        Function.temperature().register_to(climb)
        climb.register_to(model)

        # build the tacs interface, coupled driver, and oneway driver
        comm = MPI.COMM_WORLD
        solvers = SolverManager(comm)
        solvers.flow = TestAerodynamicSolver(comm, model)
        solvers.structural = TacsInterface.create_from_bdf(
            model, comm, 1, bdf_filename, callback=thermoelasticity_callback
        )
        transfer_settings = TransferSettings(npts=5)
        coupled_driver = FUNtoFEMnlbgs(
            solvers, transfer_settings=transfer_settings, model=model
        )

        epsilon = 1e-30 if complex_mode else 1e-4
        rtol = 1e-9 if complex_mode else 1e-5

        """complex step test over coordinate derivatives"""
        tester = CoordinateDerivativeTester(coupled_driver)
        rel_error = tester.test_struct_coordinates(
            "funtofem_driver struct coordinate derivatives multiscenario steady-aerothermoelastic",
            status_file=self.FILEPATH,
            epsilon=epsilon,
            complex_mode=complex_mode,
        )
        assert abs(rel_error) < rtol
        return


if __name__ == "__main__":
    open(TestFuntofemDriverStructCoordinate.FILEPATH, "w").close()
    complex_mode = False
    unittest.main()
