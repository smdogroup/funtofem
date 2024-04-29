import numpy as np, unittest, importlib
import os
from mpi4py import MPI

from funtofem.model import (
    FUNtoFEMmodel,
    Body,
    Scenario,
    Function,
    AitkenRelaxation,
    Variable,
)
from funtofem.interface import make_test_directories

np.random.seed(1234567)

# check whether fun3d is available
fun3d_loader = importlib.util.find_spec("fun3d")
has_fun3d = fun3d_loader is not None
if has_fun3d:
    from funtofem.interface import Fun3d14GridInterface

comm = MPI.COMM_WORLD
base_dir = os.path.dirname(os.path.abspath(__file__))
results_folder, _ = make_test_directories(comm, base_dir)


@unittest.skipIf(not has_fun3d, "skipping fun3d test without fun3d")
class TestFun3dGridDeformation(unittest.TestCase):
    FILENAME = "fun3d_grid_deform_test.txt"
    FILEPATH = os.path.join(results_folder, FILENAME)

    def test_fun3d_grid_deformation(self):
        # build a funtofem model with one body and scenario
        # build the funtofem model with one body and scenario
        model = FUNtoFEMmodel("plate")
        plate = Body.aeroelastic("plate", boundary=2) #.relaxation(AitkenRelaxation())
        Variable.structural("fake").register_to(plate)
        plate.register_to(model)
        test_scenario = Scenario.steady("grid", steps=2)
        test_scenario.include(Function.ksfailure())
        test_scenario.register_to(model)

        # build a FUN3D Grid deformation interface and perform the test on it from the class
        grid_interface = Fun3d14GridInterface(comm, model, fun3d_dir="meshes", complex_mode=False, forward_min_tolerance=1e10, adjoint_min_tolerance=1e10)
        rel_error = Fun3d14GridInterface.finite_diff_test(grid_interface, self.FILEPATH)
        rtol = 1e-9

        self.assertTrue(abs(rel_error) < rtol)
        return


if __name__ == "__main__":
    if comm.rank == 0:
        open(TestFun3dGridDeformation.FILEPATH, "w").close()
    unittest.main()
