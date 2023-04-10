import numpy as np, unittest, importlib, os
from mpi4py import MPI

# from funtofem import TransferScheme
from pyfuntofem.model import (
    FUNtoFEMmodel,
    Variable,
    Scenario,
    Body,
    Function,
    AitkenRelaxation,
)
from pyfuntofem.interface import (
    SolverManager,
    TestResult,
)

# check whether fun3d is available
fun3d_loader = importlib.util.find_spec("fun3d")
has_fun3d = fun3d_loader is not None

if has_fun3d:
    from pyfuntofem.interface import Fun3dInterface
    from pyfuntofem.driver import Fun3dOnewayDriver

np.random.seed(1234567)

comm = MPI.COMM_WORLD
base_dir = os.path.dirname(os.path.abspath(__file__))
bdf_file = os.path.join(base_dir, "meshes", "nastran_CAPS.dat")

results_folder = os.path.join(base_dir, "results")
if comm.rank == 0:  # make the results folder if doesn't exist
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)


class TestFun3dOnewayDriver(unittest.TestCase):
    """
    This class performs unit test on the oneway-coupled FUN3D driver
    which uses fixed struct disps or no struct disps
    TODO : in the case of an unsteady one, add methods for those too?
    """

    FILENAME = "fun3d-oneway-driver.txt"
    FILEPATH = os.path.join(results_folder, FILENAME)

    def test_nominal(self):
        """test no struct disps into FUN3D"""
        # build the funtofem model with one body and scenario
        model = FUNtoFEMmodel("wing")
        wing = Body.aeroelastic("wing", boundary=2)
        wing.register_to(model)
        test_scenario = Scenario.steady("turbulent", steps=1000).set_temperature(
            T_ref=300.0, T_inf=300.0
        )
        aoa = test_scenario.get_variable("AOA")
        test_scenario.include(Function.lift()).include(Function.drag())
        test_scenario.register_to(model)

        # build the solvers and coupled driver
        solvers = SolverManager(comm)
        solvers.flow = Fun3dInterface(comm, model, fun3d_dir="meshes")

        # comm_manager = CommManager(comm, tacs_comm, 0, comm, 0)
        driver = Fun3dOnewayDriver.nominal(solvers, model)

        # run the complex step test on the model and driver
        max_rel_error = TestResult.complex_step(
            "fun3d-oneway-turbulent-aeroelastic",
            model,
            driver,
            TestFun3dOnewayDriver.FILEPATH,
        )
        self.assertTrue(max_rel_error < 1e-7)

        return


if __name__ == "__main__":
    open(TestFun3dOnewayDriver.FILEPATH, "w").close()
    unittest.main()
