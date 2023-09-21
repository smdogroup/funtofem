import numpy as np, unittest, importlib, os
from mpi4py import MPI

# from funtofem import TransferScheme
from funtofem.model import (
    FUNtoFEMmodel,
    Scenario,
    Body,
    Function,
)
from funtofem.interface import SolverManager, TestResult, make_test_directories

# check whether fun3d is available
fun3d_loader = importlib.util.find_spec("fun3d")
has_fun3d = fun3d_loader is not None

if has_fun3d:
    from funtofem.interface import Fun3dInterface
    from funtofem.driver import Fun3dOnewayDriver

np.random.seed(1234567)

comm = MPI.COMM_WORLD
base_dir = os.path.dirname(os.path.abspath(__file__))
results_folder, _ = make_test_directories(comm, base_dir)


@unittest.skipIf(not has_fun3d, "skipping fun3d test without fun3d")
class TestFun3dOnewayAero(unittest.TestCase):
    """
    This class performs unit test on the oneway-coupled FUN3D driver
    which uses fixed struct disps or no struct disps
    TODO : in the case of an unsteady one, add methods for those too?
    """

    FILENAME = "fun3d-oneway-aero.txt"
    FILEPATH = os.path.join(results_folder, FILENAME)

    def test_nominal(self):
        """test no struct disps into FUN3D"""
        # build the funtofem model with one body and scenario
        model = FUNtoFEMmodel("wing")
        wing = Body.aeroelastic("wing", boundary=2)
        wing.register_to(model)
        test_scenario = Scenario.steady("turbulent", steps=2000).set_temperature(
            T_ref=300.0, T_inf=300.0
        )
        aoa = test_scenario.get_variable("AOA").set_bounds(value=2.0)
        test_scenario.include(Function.lift()).include(Function.drag())
        test_scenario.register_to(model)

        # build the solvers and coupled driver
        solvers = SolverManager(comm)
        solvers.flow = Fun3dInterface(comm, model, fun3d_dir="meshes")

        # comm_manager = CommManager(comm, tacs_comm, 0, comm, 0)
        driver = Fun3dOnewayDriver.analysis(solvers, model)

        # run the complex step test on the model and driver
        max_rel_error = TestResult.complex_step(
            "fun3d-oneway-turbulent-aeroelastic",
            model,
            driver,
            self.FILEPATH,
        )
        self.assertTrue(max_rel_error < 1e-7)

        return


if __name__ == "__main__":
    if comm.rank == 0:
        open(TestFun3dOnewayAero.FILEPATH, "w").close()
    unittest.main()
