import os, unittest, importlib
from funtofem.interface import Fun3dModel, Fun3dBC, HandcraftedMeshMorph
from funtofem.driver import TransferSettings
from mpi4py import MPI

base_dir = os.path.dirname(os.path.abspath(__file__))
comm = MPI.COMM_WORLD
csm_path = os.path.join(base_dir, "flow_wing.csm")

fun3d_loader = importlib.util.find_spec("fun3d")
caps_loader = importlib.util.find_spec("pyCAPS")


# first just test the fun3d and aflr aim features
@unittest.skipIf(
    fun3d_loader is None and caps_loader is None,
    "need CAPS to run this job, FUN3D not technically required but skipping anyways.",
)
class TestFun3dAimHandcraftedMesh(unittest.TestCase):
    def test_forward_process(self):
        """just check that it runs without error"""
        fun3d_model = Fun3dModel.build(
            csm_file=csm_path,
            comm=comm,
            project_name="wing_test",
            root=0,
            mesh_morph=True,
        )
        mesh_aim = fun3d_model.mesh_aim
        fun3d_aim = fun3d_model.fun3d_aim

        # get coordinates from FUN3D (from Pointwise xyz and ids), can also maybe get these coordinates from FUN3D
        # or handcrafted input mesh file, in this case
        handcrafted_mesh_file = None

        handcrafted_mesh_morph = HandcraftedMeshMorph(
            transfer_settings=TransferSettings(npts=200, beta=0.5),
            handcrafted_mesh_file=handcrafted_mesh_file,
        )

        mesh_aim.surface_aim.set_surface_mesh(
            ff_growth=1.3, min_scale=0.01, max_scale=5.0
        )
        mesh_aim.volume_aim.set_boundary_layer(initial_spacing=0.001, thickness=0.1)
        Fun3dBC.viscous(caps_group="wall", wall_spacing=0.0001).register_to(fun3d_model)
        Fun3dBC.Farfield(caps_group="Farfield").register_to(fun3d_model)
        fun3d_model.setup()
        fun3d_aim.pre_analysis()

        # read in the initial dat file and store in the
        handcrafted_mesh_morph.read_morph_file(_caps_morph_file)

        # initial write of our own mesh morph file for the handcrafted mesh
        handcrafted_mesh_morph.transfer_disps()
        handcrafted_mesh_morph.write_morph_file(_handcrafted_morph_file)

        # run the Fun3dInterface forward + adjoint analysis on SSW
        # however in this case => just sum the surface coordinates

        # transfer coordinate derivatives


if __name__ == "__main__":
    unittest.main()
