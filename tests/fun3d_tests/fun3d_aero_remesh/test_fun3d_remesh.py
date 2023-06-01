import numpy as np, unittest, importlib, os
from mpi4py import MPI

# from funtofem import TransferScheme
from pyfuntofem.model import (
    FUNtoFEMmodel,
    Variable,
    Scenario,
    Body,
    Function,
)
from pyfuntofem.interface import SolverManager, TestResult, Fun3dBC, Fun3dModel

# check whether fun3d is available
fun3d_loader = importlib.util.find_spec("fun3d")
has_fun3d = fun3d_loader is not None

if has_fun3d:
    from pyfuntofem.driver import Fun3dOnewayDriver, Fun3dRemote

np.random.seed(1234567)

comm = MPI.COMM_WORLD
base_dir = os.path.dirname(os.path.abspath(__file__))
csm_path = os.path.join(base_dir, "naca_wing.csm")

analysis_file = os.path.join(base_dir, "run_fun3d_analysis.py")
fun3d_dir = os.path.join(base_dir, "meshes")

results_folder = os.path.join(base_dir, "results")
if comm.rank == 0:  # make the results folder if doesn't exist
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)


@unittest.skipIf(
    fun3d_loader is None,
    "required to have FUN3D",
)
class TestFun3dOnewayRemesh(unittest.TestCase):
    """
    This class performs unit test on the oneway-coupled FUN3D driver
    which uses fixed struct disps or no struct disps
    TODO : in the case of an unsteady one, add methods for those too?
    """

    FILENAME = "fun3d-oneway-shape.txt"
    FILEPATH = os.path.join(results_folder, FILENAME)

    def test_remote_driver(self):
        # build the funtofem model with one body and scenario
        model = FUNtoFEMmodel("wing")
        # design the shape
        fun3d_model = Fun3dModel.build(csm_file=csm_path, comm=comm)
        aflr_aim = fun3d_model.aflr_aim

        aflr_aim.set_surface_mesh(ff_growth=1.2, min_scale=0.01, max_scale=5.0)
        aflr_aim.set_boundary_layer(initial_spacing=0.001, thickness=0.01)
        Fun3dBC.inviscid(caps_group="wall", wall_spacing=0.001).register_to(fun3d_model)
        Fun3dBC.Farfield(caps_group="Farfield").register_to(fun3d_model)
        fun3d_model.setup()
        model.flow = fun3d_model

        wing = Body.aeroelastic("wing", boundary=2)
        Variable.shape(name="aoa").set_bounds(
            lower=-1.0, value=0.0, upper=1.0
        ).register_to(wing)
        wing.register_to(model)
        test_scenario = (
            Scenario.steady("euler", steps=10)
            .set_temperature(T_ref=300.0, T_inf=300.0)
            .fun3d_project("funtofem_CAPS")
        )
        # test_scenario.get_variable("AOA").set_bounds(value=2.0)
        test_scenario.include(Function.lift()).include(Function.drag())
        test_scenario.register_to(model)

        # build the solvers and coupled driver
        solvers = SolverManager(comm)
        fun3d_remote = Fun3dRemote(analysis_file, fun3d_dir, nprocs=48)
        driver = Fun3dOnewayDriver.remote(solvers, model, fun3d_remote)

        # run the complex step test on the model and driver
        max_rel_error = TestResult.finite_difference(
            "fun3d+oneway-remesh-euler",
            model,
            driver,
            TestFun3dOnewayRemesh.FILEPATH,
            both_adjoint=True,  # since the Fun3dOnewayDriver doesn't know function values until after solve_adjoint()
            epsilon=1e-1,
        )
        self.assertTrue(max_rel_error < 1e-4)

        return


if __name__ == "__main__":
    if comm.rank == 0:
        open(TestFun3dOnewayRemesh.FILEPATH, "w").close()
    unittest.main()
