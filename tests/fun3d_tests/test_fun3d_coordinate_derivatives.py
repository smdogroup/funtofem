import os, importlib
from tacs import TACS
from mpi4py import MPI
import numpy as np
from funtofem import TransferScheme

from pyfuntofem.model import FUNtoFEMmodel, Variable, Scenario, Body, Function
from pyfuntofem.interface import (
    TacsInterface,
    SolverManager,
    TacsIntegrationSettings,
    CoordinateDerivativeTester,
    TestStructuralSolver,
)
from pyfuntofem.driver import TransferSettings, FUNtoFEMnlbgs

# check whether fun3d is available
fun3d_loader = importlib.util.find_spec("fun3d")
has_fun3d = fun3d_loader is not None

if has_fun3d:
    from pyfuntofem.interface import Fun3dInterface

# from bdf_test_utils import elasticity_callback, thermoelasticity_callback
import unittest

np.random.seed(123456)

base_dir = os.path.dirname(os.path.abspath(__file__))
# bdf_filename = os.path.join(base_dir, "input_files", "test_bdf_file.bdf")


complex_mode = TransferScheme.dtype == complex and TACS.dtype == complex
nprocs = 1
comm = MPI.COMM_WORLD

results_folder = os.path.join(base_dir, "results")
if comm.rank == 0:  # make the results folder if doesn't exist
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)


@unittest.skipIf(
    not complex_mode, "only testing coordinate derivatives with complex step"
)
class TestFun3dCoordinateDerivatives(unittest.TestCase):
    FILENAME = "f2fdriver-fun3daero-coordinates.txt"
    FILEPATH = os.path.join(results_folder, FILENAME)

    def test_steady_aero_aeroelastic(self):
        # build the model and driver
        model = FUNtoFEMmodel("wedge")
        plate = Body.aeroelastic("plate", boundary=1)
        Variable.structural("thickness").set_bounds(
            lower=0.01, value=1.0, upper=1.0
        ).register_to(plate)
        plate.register_to(model)

        # build the scenario
        scenario = Scenario.steady("turbulent", steps=7000).include(
            Function.ksfailure()
        )
        scenario.adjoint_steps = 2000
        scenario.register_to(model)

        # build the tacs interface, coupled driver, and oneway driver
        comm = MPI.COMM_WORLD
        solvers = SolverManager(comm)
        solvers.flow = Fun3dInterface(comm, model, fun3d_dir="meshes")
        solvers.structural = TestStructuralSolver(comm, model)
        transfer_settings = TransferSettings(npts=50)
        coupled_driver = FUNtoFEMnlbgs(
            solvers, transfer_settings=transfer_settings, model=model
        )

        rtol = 1e-9

        """complex step test over coordinate derivatives"""
        tester = CoordinateDerivativeTester(coupled_driver)
        rel_error = tester.test_aero_coordinates(
            "funtofem driver, steady-aeroelastic",
            TestFun3dCoordinateDerivatives.FILEPATH,
        )
        assert abs(rel_error) < rtol
        return

    @unittest.skip("temp")
    def test_steady_aero_aerothermal(self):
        # build the model and driver
        model = FUNtoFEMmodel("wedge")
        plate = Body.aerothermal("plate", boundary=1)
        Variable.structural("thickness").set_bounds(
            lower=0.01, value=0.1, upper=1.0
        ).register_to(plate)
        plate.register_to(model)

        # build the scenario
        scenario = Scenario.steady("turbulent", steps=1000).include(
            Function.temperature()
        )
        scenario.register_to(model)

        # build the tacs interface, coupled driver, and oneway driver
        comm = MPI.COMM_WORLD
        solvers = SolverManager(comm)
        solvers.flow = Fun3dInterface(comm, model, fun3d_dir="meshes")
        solvers.structural = TacsInterface.create_from_bdf(
            model, comm, 1, bdf_filename, callback=thermoelasticity_callback
        )
        transfer_settings = TestStructuralSolver(comm, model)
        coupled_driver = FUNtoFEMnlbgs(
            solvers, transfer_settings=transfer_settings, model=model
        )

        rtol = 1e-9

        """complex step test over coordinate derivatives"""
        tester = CoordinateDerivativeTester(coupled_driver)
        rel_error = tester.test_aero_coordinates(
            "funtofem_driver aero coordinate derivatives steady-aerothermal"
        )
        assert abs(rel_error) < rtol
        return

    @unittest.skip("temp")
    def test_steady_aero_aerothermoelastic(self):
        # build the model and driver
        model = FUNtoFEMmodel("wedge")
        plate = Body.aerothermoelastic("plate", boundary=1)
        Variable.structural("thickness").set_bounds(
            lower=0.01, value=0.1, upper=1.0
        ).register_to(plate)
        plate.register_to(model)

        # build the scenario
        scenario = Scenario.steady("turbulent", steps=1000).include(
            Function.temperature()
        )
        scenario.include(Function.drag()).include(Function.lift())
        scenario.register_to(model)

        # build the tacs interface, coupled driver, and oneway driver
        comm = MPI.COMM_WORLD
        solvers = SolverManager(comm)
        solvers.flow = Fun3dInterface(comm, model, fun3d_dir="meshes")
        solvers.structural = TacsInterface.create_from_bdf(
            model, comm, 1, bdf_filename, callback=thermoelasticity_callback
        )
        transfer_settings = TestStructuralSolver(comm, model)
        coupled_driver = FUNtoFEMnlbgs(
            solvers, transfer_settings=transfer_settings, model=model
        )

        rtol = 1e-9

        """complex step test over coordinate derivatives"""
        tester = CoordinateDerivativeTester(coupled_driver)
        rel_error = tester.test_aero_coordinates(
            "funtofem_driver aero coordinate derivatives steady-aerothermoelastic"
        )
        assert abs(rel_error) < rtol
        return

    @unittest.skip("temp")
    def test_steady_aero_multiscenario_aerothermoelastic(self):
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

        rtol = 1e-9

        """complex step test over coordinate derivatives"""
        tester = CoordinateDerivativeTester(coupled_driver)
        rel_error = tester.test_aero_coordinates(
            "funtofem_driver aero coordinate derivatives multiscenario, steady-aerothermoelastic"
        )
        assert abs(rel_error) < rtol
        return


if __name__ == "__main__":
    unittest.main()
