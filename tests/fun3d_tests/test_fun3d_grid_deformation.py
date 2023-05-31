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

np.random.seed(1234567)

# check whether fun3d is available
fun3d_loader = importlib.util.find_spec("fun3d")
has_fun3d = fun3d_loader is not None
if has_fun3d:
    from funtofem.interface import Fun3dGridInterface

comm = MPI.COMM_WORLD
base_dir = os.path.dirname(os.path.abspath(__file__))
results_folder = os.path.join(base_dir, "results")
if not os.path.exists(results_folder) and comm.rank == 0:
    os.mkdir(results_folder)


@unittest.skipIf(not has_fun3d, "skipping fun3d test without fun3d")
class TestFun3dGridDeformation(unittest.TestCase):
    FILENAME = "fun3d_grid_deform_test.txt"
    FILEPATH = os.path.join(results_folder, FILENAME)

    def test_fun3d_grid_deformation(self):
        # build a funtofem model with one body and scenario
        # build the funtofem model with one body and scenario
        model = FUNtoFEMmodel("plate")
        plate = Body.aeroelastic("plate", boundary=6).relaxation(AitkenRelaxation())
        Variable.structural("fake").register_to(plate)
        plate.register_to(model)
        test_scenario = Scenario.steady("grid", steps=1)
        test_scenario.include(Function.ksfailure())
        test_scenario.register_to(model)

        # build a FUN3D Grid deformation interface and perform the test on it from the class
        grid_interface = Fun3dGridInterface(comm, model, fun3d_dir="meshes")
        rel_error = Fun3dGridInterface.complex_step_test(
            grid_interface, TestFun3dGridDeformation.FILEPATH
        )
        rtol = 1e-9

        self.assertTrue(abs(rel_error) < rtol)
        return


if __name__ == "__main__":
    unittest.main()
