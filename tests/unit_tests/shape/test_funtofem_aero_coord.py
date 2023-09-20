import os
from tacs import TACS
from mpi4py import MPI
import numpy as np
from funtofem import TransferScheme

from funtofem.model import FUNtoFEMmodel, Scenario, Body, Function
from funtofem.interface import (
    TestAerodynamicSolver,
    TacsInterface,
    SolverManager,
    CoordinateDerivativeTester,
    test_directories,
)
from funtofem.driver import TransferSettings, FUNtoFEMnlbgs

from _bdf_test_utils import elasticity_callback, thermoelasticity_callback
import unittest

np.random.seed(123456)

base_dir = os.path.dirname(os.path.abspath(__file__))
bdf_filename = os.path.join(base_dir, "input_files", "test_bdf_file.bdf")

complex_mode = TransferScheme.dtype == complex and TACS.dtype == complex
nprocs = 1
comm = MPI.COMM_WORLD

results_folder, output_dir = test_directories(comm, base_dir)
in_github_workflow = bool(os.getenv("GITHUB_ACTIONS"))

elastic_scheme = "rbf"


@unittest.skipIf(
    not complex_mode, "only testing coordinate derivatives with complex step"
)
class TestFuntofemDriverAeroCoordinate(unittest.TestCase):
    FILENAME = "f2f-steady-aero-coord.txt"
    FILEPATH = os.path.join(results_folder, FILENAME)

    def test_steady_aero_aeroelastic(self):
        # build the model and driver
        model = FUNtoFEMmodel("wedge")
        plate = Body.aeroelastic("plate", boundary=1)
        plate.register_to(model)

        # build the scenario
        scenario = Scenario.steady("test", steps=200)
        Function.ksfailure().register_to(scenario)
        Function.lift().register_to(scenario)
        scenario.register_to(model)

        # build the tacs interface, coupled driver, and oneway driver
        comm = MPI.COMM_WORLD
        solvers = SolverManager(comm)
        solvers.flow = TestAerodynamicSolver(comm, model)
        solvers.structural = TacsInterface.create_from_bdf(
            model,
            comm,
            1,
            bdf_filename,
            callback=elasticity_callback,
            output_dir=output_dir,
        )
        transfer_settings = TransferSettings(npts=20, elastic_scheme=elastic_scheme)
        coupled_driver = FUNtoFEMnlbgs(
            solvers, transfer_settings=transfer_settings, model=model
        )

        epsilon = 1e-30 if complex_mode else 1e-4
        rtol = 1e-9 if complex_mode else 1e-5

        """complex step test over coordinate derivatives"""
        tester = CoordinateDerivativeTester(coupled_driver)
        rel_error = tester.test_aero_coordinates(
            "funtofem_driver aero coordinate derivatives steady-aeroelastic",
            status_file=self.FILEPATH,
            epsilon=epsilon,
            complex_mode=complex_mode,
        )
        assert abs(rel_error) < rtol
        return

    def test_steady_aero_aerothermal(self):
        # build the model and driver
        model = FUNtoFEMmodel("wedge")
        plate = Body.aerothermal("plate", boundary=1)
        plate.register_to(model)

        # build the scenario
        scenario = Scenario.steady("test", steps=200)
        Function.ksfailure().register_to(scenario)
        Function.lift().register_to(scenario)
        scenario.register_to(model)

        # build the tacs interface, coupled driver, and oneway driver
        comm = MPI.COMM_WORLD
        solvers = SolverManager(comm)
        solvers.flow = TestAerodynamicSolver(comm, model)
        solvers.structural = TacsInterface.create_from_bdf(
            model,
            comm,
            1,
            bdf_filename,
            callback=thermoelasticity_callback,
            output_dir=output_dir,
        )
        transfer_settings = TransferSettings(npts=20, elastic_scheme=elastic_scheme)
        coupled_driver = FUNtoFEMnlbgs(
            solvers, transfer_settings=transfer_settings, model=model
        )

        epsilon = 1e-30 if complex_mode else 1e-4
        rtol = 1e-9 if complex_mode else 1e-5

        """complex step test over coordinate derivatives"""
        tester = CoordinateDerivativeTester(coupled_driver)
        rel_error = tester.test_aero_coordinates(
            "funtofem_driver aero coordinate derivatives steady-aerothermal",
            status_file=self.FILEPATH,
            epsilon=epsilon,
            complex_mode=complex_mode,
        )
        assert abs(rel_error) < rtol
        return

    def test_steady_aero_aerothermoelastic(self):
        # build the model and driver
        model = FUNtoFEMmodel("wedge")
        plate = Body.aerothermoelastic("plate", boundary=1)
        plate.register_to(model)

        # build the scenario
        scenario = Scenario.steady("test", steps=200)
        Function.ksfailure().register_to(scenario)
        Function.temperature().register_to(scenario)
        Function.lift().register_to(scenario)
        scenario.register_to(model)

        # build the tacs interface, coupled driver, and oneway driver
        comm = MPI.COMM_WORLD
        solvers = SolverManager(comm)
        solvers.flow = TestAerodynamicSolver(comm, model)
        solvers.structural = TacsInterface.create_from_bdf(
            model,
            comm,
            1,
            bdf_filename,
            callback=thermoelasticity_callback,
            output_dir=output_dir,
        )
        transfer_settings = TransferSettings(npts=20, elastic_scheme=elastic_scheme)
        coupled_driver = FUNtoFEMnlbgs(
            solvers, transfer_settings=transfer_settings, model=model
        )

        epsilon = 1e-30 if complex_mode else 1e-4
        rtol = 1e-9 if complex_mode else 1e-5

        """complex step test over coordinate derivatives"""
        tester = CoordinateDerivativeTester(coupled_driver)
        rel_error = tester.test_aero_coordinates(
            "funtofem_driver aero coordinate derivatives steady-aerothermoelastic",
            status_file=self.FILEPATH,
            epsilon=epsilon,
            complex_mode=complex_mode,
        )
        assert abs(rel_error) < rtol
        return


if __name__ == "__main__":
    if comm.rank == 0:
        open(TestFuntofemDriverAeroCoordinate.FILEPATH, "w").close()
    complex_mode = False
    unittest.main()
