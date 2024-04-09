import os, unittest, importlib
from funtofem.interface import Fun3dModel, Fun3dBC, HandcraftedMeshMorph
from funtofem.driver import TransferSettings
from mpi4py import MPI

# Imports from FUNtoFEM
from funtofem.model import (
    FUNtoFEMmodel,
    Variable,
    Scenario,
    Body,
    Function,
)
from funtofem.interface import (
    TacsSteadyInterface,
    SolverManager,
    TestResult,
    make_test_directories,
)
from funtofem.driver import FUNtoFEMnlbgs, TransferSettings

base_dir = os.path.dirname(os.path.abspath(__file__))
comm = MPI.COMM_WORLD
csm_path = os.path.join(base_dir, "meshes", "flow_wing.csm")

fun3d_loader = importlib.util.find_spec("fun3d")
caps_loader = importlib.util.find_spec("pyCAPS")

has_fun3d = fun3d_loader is not None
if has_fun3d:
    from funtofem.interface import Fun3d14Interface


# first just test the fun3d and aflr aim features
@unittest.skipIf(
    fun3d_loader is None and caps_loader is None,
    "need CAPS to run this job, FUN3D not technically required but skipping anyways.",
)
class TestFun3dAimHandcraftedMesh(unittest.TestCase):
    def test_forward_process(self):
        """just check that it runs without error"""

        # build the funtofem model with one body and scenario
        model = FUNtoFEMmodel("wing")
        # design the shape
        fun3d_model = Fun3dModel.build(
            csm_file=csm_path,
            comm=comm,
            project_name="wing_test",
            root=0,
            mesh_morph=True,
        )
        mesh_aim = fun3d_model.mesh_aim
        fun3d_aim = fun3d_model.fun3d_aim

        fun3d_model.handcrafted_mesh_morph = HandcraftedMeshMorph(
            transfer_settings=TransferSettings(npts=200, beta=0.5),
        )

        mesh_aim.surface_aim.set_surface_mesh(
            ff_growth=1.3, min_scale=0.01, max_scale=5.0
        )
        mesh_aim.volume_aim.set_boundary_layer(initial_spacing=0.001, thickness=0.1)
        Fun3dBC.viscous(caps_group="wall", wall_spacing=0.0001).register_to(fun3d_model)
        Fun3dBC.Farfield(caps_group="Farfield").register_to(fun3d_model)
        fun3d_model.setup()
        model.flow = fun3d_model

        wing = Body.aeroelastic("wing", boundary=2)
        Variable.shape(name="aoa").set_bounds(
            lower=-1.0, value=0.0, upper=1.0
        ).register_to(wing)
        wing.register_to(model)
        test_scenario = (
            Scenario.steady("euler", steps=5000)
            .set_temperature(T_ref=300.0, T_inf=300.0)
            .fun3d_project(fun3d_aim.project_name)
        )
        test_scenario.adjoint_steps = 4000
        # test_scenario.get_variable("AOA").set_bounds(value=2.0)

        test_scenario.include(Function.lift()).include(Function.drag())
        test_scenario.register_to(model)

        # build the solvers and coupled driver
        solvers = SolverManager(comm)
        solvers.flow = Fun3d14Interface(
            comm, model, fun3d_dir="meshes"
        )

        # copy the coordinates from the fun3d14interface to the handcrafted mesh object
        

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
