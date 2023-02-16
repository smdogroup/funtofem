import numpy as np, unittest, importlib, os
from mpi4py import MPI

# from funtofem import TransferScheme
from pyfuntofem.model import FUNtoFEMmodel, Variable, Scenario, Body, Function
from pyfuntofem.interface import (
    TestStructuralSolver,
    SolverManager,
    TestResult,
)
from pyfuntofem.driver import FUNtoFEMnlbgs, TransferSettings

# check whether fun3d is available
fun3d_loader = importlib.util.find_spec("fun3d")
has_fun3d = fun3d_loader is not None
if has_fun3d:
    from pyfuntofem.interface import Fun3dInterface

comm = MPI.COMM_WORLD
np.random.seed(1234567)
results_folder = os.path.join(os.getcwd(), "results")
if not os.path.exists(results_folder) and comm.rank == 0:
    os.mkdir(results_folder)


@unittest.skipIf(not has_fun3d, "skipping fun3d test without fun3d")
class TestFun3dUncoupled(unittest.TestCase):
    FILENAME = "fun3d-fake-laminar.txt"
    FILEPATH = os.path.join(results_folder, FILENAME)

    def test_laminar_aeroelastic(self):
        # build the funtofem model with one body and scenario
        model = FUNtoFEMmodel("plate")
        plate = Body.aeroelastic("plate", boundary=6)
        Variable.structural("thickness").set_bounds(
            lower=0.001, value=0.01, upper=2.0
        ).register_to(plate)
        plate.register_to(model)
        test_scenario = Scenario.steady("laminar", steps=500).set_temperature(
            T_ref=300.0, T_inf=300.0
        )
        test_scenario.include(Function.ksfailure(ks_weight=50.0)).include(
            Function.lift()
        ).include(Function.drag())
        test_scenario.register_to(model)

        # build the solvers and coupled driver
        comm = MPI.COMM_WORLD
        solvers = SolverManager(comm)
        solvers.flow = Fun3dInterface(comm, model, fun3d_dir="meshes").set_units(
            qinf=1.0
        )
        solvers.structural = TestStructuralSolver(comm, model, elastic_k=1.0e5)
        transfer_settings = TransferSettings()
        driver = FUNtoFEMnlbgs(
            solvers, transfer_settings=transfer_settings, model=model
        )

        # run the complex step test on the model and driver
        max_rel_error = TestResult.complex_step(
            "fun3d+fake-laminar-aeroelastic", model, driver, TestFun3dUncoupled.FILEPATH
        )
        self.assertTrue(max_rel_error < 1e-7)

    @unittest.skip("test aero solver thermal")
    def _laminar_aerothermal(self):
        # build the funtofem model with one body and scenario
        model = FUNtoFEMmodel("plate")
        plate = Body.aerothermal("plate", boundary=6)
        Variable.structural("thickness").set_bounds(
            lower=0.001, value=0.01, upper=2.0
        ).register_to(plate)
        plate.register_to(model)
        test_scenario = Scenario.steady("laminar", steps=500).set_temperature(
            T_ref=300.0, T_inf=300.0
        )
        test_scenario.include(Function.temperature()).include(Function.lift()).include(
            Function.drag()
        )
        test_scenario.register_to(model)

        # build the solvers and coupled driver
        comm = MPI.COMM_WORLD
        solvers = SolverManager(comm)
        solvers.flow = Fun3dInterface(comm, model, fun3d_dir="meshes")
        solvers.structural = TestStructuralSolver(comm, model, thermal_k=1.0)
        transfer_settings = TransferSettings()
        driver = FUNtoFEMnlbgs(
            solvers, transfer_settings=transfer_settings, model=model
        )

        # run the complex step test on the model and driver
        max_rel_error = TestResult.complex_step(
            "fun3d+fake-laminar-aerothermal", model, driver, TestFun3dUncoupled.FILEPATH
        )
        self.assertTrue(max_rel_error < 1e-7)

    @unittest.skip("test aero solver thermal")
    def test_laminar_aerothermoelastic(self):
        # build the funtofem model with one body and scenario
        model = FUNtoFEMmodel("plate")
        plate = Body.aerothermoelastic("plate", boundary=1)
        Variable.structural("thickness").set_bounds(
            lower=0.001, value=0.01, upper=2.0
        ).register_to(plate)
        plate.register_to(model)
        test_scenario = Scenario.steady("laminar", steps=500).set_temperature(
            T_ref=300.0, T_inf=300.0
        )
        test_scenario.include(Function.ksfailure()).include(
            Function.temperature()
        ).include(Function.lift()).include(Function.drag())
        test_scenario.register_to(model)

        # build the solvers and coupled driver
        comm = MPI.COMM_WORLD
        solvers = SolverManager(comm)
        solvers.flow = Fun3dInterface(comm, model, fun3d_dir="meshes").set_units(
            qinf=1.0
        )
        solvers.structural = TestStructuralSolver(
            comm, model, elastic_k=1000.0, thermal_k=1.0
        )
        transfer_settings = TransferSettings()
        driver = FUNtoFEMnlbgs(
            solvers, transfer_settings=transfer_settings, model=model
        )

        # run the complex step test on the model and driver
        max_rel_error = TestResult.complex_step(
            "fun3d+fake-laminar-aerothermoelastic",
            model,
            driver,
            TestFun3dUncoupled.FILEPATH,
        )
        self.assertTrue(max_rel_error < 1e-7)


if __name__ == "__main__":
    # open and close the file to reset it
    if comm.rank == 0:
        open(TestFun3dUncoupled.FILEPATH, "w").close()

    unittest.main()
