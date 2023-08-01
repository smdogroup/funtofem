import os, unittest, importlib
from pyfuntofem.interface import Fun3dModel, Fun3dBC
from mpi4py import MPI

base_dir = os.path.dirname(os.path.abspath(__file__))
comm = MPI.COMM_WORLD
csm_path = os.path.join(base_dir, "input_files", "flow_wing.csm")

fun3d_loader = importlib.util.find_spec("fun3d")
caps_loader = importlib.util.find_spec("pyCAPS")


# first just test the fun3d and aflr aim features
@unittest.skipIf(
    fun3d_loader is None and caps_loader is None,
    "need CAPS to run this job, FUN3D not technically required but skipping anyways.",
)
class TestFun3dAim(unittest.TestCase):
    def test_pre_analysis(self):
        """just check that it runs without error"""
        fun3d_model = Fun3dModel.build(
            csm_file=csm_path, comm=comm, project_name="wing_test"
        )
        aflr_aim = fun3d_model.aflr_aim
        fun3d_aim = fun3d_model.fun3d_aim

        aflr_aim.set_surface_mesh(ff_growth=1.3, min_scale=0.01, max_scale=5.0)
        aflr_aim.set_boundary_layer(initial_spacing=0.001, thickness=0.1)
        Fun3dBC.viscous(caps_group="wall", wall_spacing=0.0001).register_to(fun3d_model)
        Fun3dBC.Farfield(caps_group="Farfield").register_to(fun3d_model)
        fun3d_model.setup()
        fun3d_aim.pre_analysis()


if __name__ == "__main__":
    unittest.main()
