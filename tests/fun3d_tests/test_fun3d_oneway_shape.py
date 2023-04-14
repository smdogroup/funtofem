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
csm_path = os.path.join(base_dir, "diamond_wedge.csm")

analysis_file = os.path.join(base_dir, "run_fun3d_analysis.py")
fun3d_dir = os.path.join(base_dir, "meshes")

results_folder = os.path.join(base_dir, "results")
if comm.rank == 0:  # make the results folder if doesn't exist
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)


@unittest.skipIf(not has_fun3d, "skipping fun3d test without fun3d")
class TestFun3dOnewayShape(unittest.TestCase):
    """
    This class performs unit test on the oneway-coupled FUN3D driver
    with aero shape derivatives from the FUN3D AIM.
    """

    FILENAME = "fun3d-oneway-shape.txt"
    FILEPATH = os.path.join(results_folder, FILENAME)

    def test_nominal(self):
        # build the funtofem model with one body and scenario
        model = FUNtoFEMmodel("wing")
        # design the shape
        fun3d_model = Fun3dModel.build(csm_file=csm_path, comm=comm)
        aflr_aim = fun3d_model.aflr_aim

        aflr_aim.set_surface_mesh(ff_growth=1.3, min_scale=0.1, max_scale=5.0)
        aflr_aim.set_boundary_layer(initial_spacing=0.01, thickness=0.1)
        Fun3dBC.viscous(caps_group="wall", wall_spacing=0.01).register_to(fun3d_model)
        Fun3dBC.Farfield(caps_group="Farfield").register_to(fun3d_model)
        fun3d_model.setup()
        model.flow = fun3d_model

        wing = Body.aeroelastic("wing", boundary=2)
        Variable.shape(name="sspan").set_bounds(
            lower=0.4, value=5.0, upper=9.6
        ).register_to(wing)
        wing.register_to(model)
        test_scenario = (
            Scenario.steady(
                "turbulent", steps=1
            )  # the steps aren't used here since this is in the remote script
            .set_temperature(T_ref=300.0, T_inf=300.0)
            .fun3d_project("funtofem_CAPS")
        )
        test_scenario.get_variable("AOA")
        test_scenario.include(Function.lift()).include(Function.drag())
        test_scenario.register_to(model)

        # build the solvers and coupled driver
        solvers = SolverManager(comm)
        fun3d_remote = Fun3dRemote(analysis_file, fun3d_dir, nprocs=48)
        driver = Fun3dOnewayDriver.remote(solvers, model, fun3d_remote)

        # run the complex step test on the model and driver
        max_rel_error = TestResult.finite_difference(
            "fun3d+oneway-shape-turbulent-aeroelastic",
            model,
            driver,
            TestFun3dOnewayShape.FILEPATH,
            both_adjoint=True,
        )
        self.assertTrue(max_rel_error < 1e-4)

        return


if __name__ == "__main__":
    if comm.rank == 0:
        open(TestFun3dOnewayShape.FILEPATH, "w").close()
    unittest.main()
