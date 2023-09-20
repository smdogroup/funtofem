import os, unittest, numpy as np
from tacs import TACS
from mpi4py import MPI
from funtofem import TransferScheme

from funtofem.model import FUNtoFEMmodel, Variable, Scenario, Body, Function
from funtofem.interface import (
    TestAerodynamicSolver,
    TestStructuralSolver,
    SolverManager,
    CoordinateDerivativeTester,
    test_directories,
)
from funtofem.driver import TransferSettings, FUNtoFEMnlbgs

np.random.seed(123456)

base_dir = os.path.dirname(os.path.abspath(__file__))
complex_mode = TransferScheme.dtype == complex and TACS.dtype == complex
nprocs = 1
comm = MPI.COMM_WORLD
results_folder, output_folder = test_directories(comm, base_dir)
in_github_workflow = bool(os.getenv("GITHUB_ACTIONS"))

# user-defined settings
steps = 10
elastic_scheme = "rbf"
dt = 0.001


@unittest.skipIf(
    not complex_mode, "only testing coordinate derivatives with complex step"
)
class TestFrameworkUnsteadyAeroCoordinate(unittest.TestCase):
    FILENAME = "framework-unsteady-aero-coord.txt"
    FILEPATH = os.path.join(results_folder, FILENAME)

    # @unittest.skip("under development")
    def test_unsteady_aero_aeroelastic(self):
        # build the model and driver
        model = FUNtoFEMmodel("wedge")
        plate = Body.aeroelastic("plate", boundary=1)
        plate.register_to(model)

        # build the scenario
        scenario = Scenario.unsteady("test", steps=steps)
        Function.test_struct().register_to(scenario)
        Function.test_aero().register_to(scenario)
        Variable.shape("rotation").register_to(scenario)
        scenario.register_to(model)

        # build the tacs interface, coupled driver, and oneway driver
        comm = MPI.COMM_WORLD
        solvers = SolverManager(comm)
        solvers.flow = TestAerodynamicSolver(comm, model)
        solvers.structural = TestStructuralSolver(comm, model)
        transfer_settings = TransferSettings(elastic_scheme=elastic_scheme, npts=10)
        coupled_driver = FUNtoFEMnlbgs(
            solvers, transfer_settings=transfer_settings, model=model
        )

        epsilon = 1e-30 if complex_mode else 1e-4
        rtol = 1e-7 if complex_mode else 1e-5

        """complex step test over coordinate derivatives"""
        tester = CoordinateDerivativeTester(coupled_driver)
        rel_error = tester.test_aero_coordinates(
            "funtofem_driver aero coordinate derivatives unsteady-aeroelastic",
            status_file=self.FILEPATH,
            epsilon=epsilon,
            complex_mode=complex_mode,
        )
        assert abs(rel_error) < rtol
        return

    # @unittest.skip("under development")
    def test_unsteady_aero_aerothermal(self):
        # build the model and driver
        model = FUNtoFEMmodel("wedge")
        plate = Body.aerothermal("plate", boundary=1)
        plate.register_to(model)

        # build the scenario
        scenario = Scenario.unsteady("test", steps=steps)
        Function.test_struct().register_to(scenario)
        Function.test_aero().register_to(scenario)
        Variable.shape("rotation").register_to(scenario)
        scenario.register_to(model)

        # build the tacs interface, coupled driver, and oneway driver
        comm = MPI.COMM_WORLD
        solvers = SolverManager(comm)
        solvers.flow = TestAerodynamicSolver(comm, model)
        solvers.structural = TestStructuralSolver(comm, model)
        transfer_settings = TransferSettings(elastic_scheme=elastic_scheme, npts=10)
        coupled_driver = FUNtoFEMnlbgs(
            solvers, transfer_settings=transfer_settings, model=model
        )

        epsilon = 1e-30 if complex_mode else 1e-4
        rtol = 1e-7 if complex_mode else 1e-5

        """complex step test over coordinate derivatives"""
        tester = CoordinateDerivativeTester(coupled_driver)
        rel_error = tester.test_aero_coordinates(
            "funtofem_driver aero coordinate derivatives unsteady-aerothermal",
            status_file=self.FILEPATH,
            epsilon=epsilon,
            complex_mode=complex_mode,
        )
        assert abs(rel_error) < rtol
        return

    # @unittest.skip("under development")
    def test_unsteady_aero_aerothermoelastic(self):
        # build the model and driver
        model = FUNtoFEMmodel("wedge")
        plate = Body.aerothermoelastic("plate", boundary=1)
        plate.register_to(model)

        # build the scenario
        scenario = Scenario.unsteady("test", steps=steps)
        Function.test_struct().register_to(scenario)
        Function.test_aero().register_to(scenario)
        Variable.shape("rotation").register_to(scenario)
        scenario.register_to(model)

        # build the tacs interface, coupled driver, and oneway driver
        comm = MPI.COMM_WORLD
        solvers = SolverManager(comm)
        solvers.flow = TestAerodynamicSolver(comm, model)
        solvers.structural = TestStructuralSolver(comm, model)
        transfer_settings = TransferSettings(elastic_scheme=elastic_scheme, npts=10)
        coupled_driver = FUNtoFEMnlbgs(
            solvers, transfer_settings=transfer_settings, model=model
        )

        epsilon = 1e-30 if complex_mode else 1e-4
        rtol = 1e-7 if complex_mode else 1e-5

        """complex step test over coordinate derivatives"""
        tester = CoordinateDerivativeTester(coupled_driver)
        rel_error = tester.test_aero_coordinates(
            "funtofem_driver aero coordinate derivatives unsteady-aerothermoelastic",
            status_file=self.FILEPATH,
            epsilon=epsilon,
            complex_mode=complex_mode,
        )
        assert abs(rel_error) < rtol
        return


if __name__ == "__main__":
    if comm.rank == 0:
        open(TestFrameworkUnsteadyAeroCoordinate.FILEPATH, "w").close()
    unittest.main()
