import os, unittest, importlib
from funtofem.interface import Fun3dModel, Fun3dBC, HandcraftedMeshMorph
from funtofem.driver import TransferSettings
from mpi4py import MPI
import matplotlib.pyplot as plt
import numpy as np
import niceplots

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
meshes_dir = os.path.join(base_dir, "meshes")
csm_path = os.path.join(meshes_dir, "flow_wing.csm")

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

        mesh_aim.surface_aim.set_surface_mesh(
            ff_growth=1.3, min_scale=0.01, max_scale=5.0
        )
        mesh_aim.volume_aim.set_boundary_layer(initial_spacing=0.001, thickness=0.1)
        Fun3dBC.viscous(caps_group="wall", wall_spacing=0.0001).register_to(fun3d_model)
        Fun3dBC.Farfield(caps_group="Farfield").register_to(fun3d_model)
        fun3d_model.setup()
        model.flow = fun3d_model

        wing = Body.aeroelastic("wing", boundary=2)
        twist = (
            Variable.shape(name="twist")
            .set_bounds(lower=-1.0, value=0.0, upper=1.0)
            .register_to(wing)
        )
        sweep = (
            Variable.shape(name="sweep")
            .set_bounds(lower=-5.0, value=0.0, upper=10.0)
            .register_to(wing)
        )
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
        # solvers = SolverManager(comm)
        # solvers.flow = Fun3d14Interface(
        #     comm, model, fun3d_dir="meshes"
        # )

        # create handcrafted mesh morph object after FUN3D is built
        handcrafted_mesh_morph = HandcraftedMeshMorph(
            comm=comm,
            model=model,
            transfer_settings=TransferSettings(npts=200, beta=0.5),
        )
        # fun3d_model.handcrafted_mesh_morph = handcrafted_mesh_morph

        # copy the coordinates from the fun3d14interface to the handcrafted mesh object
        # read in the initial dat file and store in the
        _hc_morph_file = os.path.join(meshes_dir, "hc_mesh.dat")
        handcrafted_mesh_morph.read_surface_file(_hc_morph_file, is_caps_mesh=False)

        # change the shape somehow
        fun3d_aim.set_design_sensitivity(False, include_file=False)
        fun3d_aim.pre_analysis()

        # copy the CAPS coordinates into the body to write a blank sens file
        _caps_morph_file = fun3d_aim.mesh_morph_filepath
        handcrafted_mesh_morph.read_surface_file(_caps_morph_file)

        wing.initialize_aero_nodes(
            handcrafted_mesh_morph.caps_aero_X, handcrafted_mesh_morph.caps_aero_id
        )
        wing.initialize_variables(test_scenario)
        wing.initialize_adjoint_variables(test_scenario)
        model.write_sensitivity_file(
            comm=comm,
            filename=fun3d_aim.sens_file_path,
            discipline="aerodynamic",
            root=fun3d_aim.root,
            write_dvs=False,
        )
        fun3d_aim.post_analysis(None)
        fun3d_aim.unlink()

        twist.value += 10.0
        sweep.value += 10.0
        fun3d_aim.pre_analysis()

        # read the new surface file
        _new_caps_morph_file = fun3d_aim.mesh_morph_filepath
        print(f"new caps morph file = {_new_caps_morph_file}")
        handcrafted_mesh_morph.read_surface_file(_new_caps_morph_file)

        # transfer shape displacements and then
        handcrafted_mesh_morph.transfer_shape_disps()

        _hc_morph_file_out = os.path.join(meshes_dir, "hc_mesh_out.dat")
        handcrafted_mesh_morph.write_surface_file(_hc_morph_file_out)

        # if this was not a unittest type of script next we would
        # run the Fun3dInterface forward + adjoint analysis on SSW
        # however in this case => just sum the surface coordinates

        # view the previous and displaced meshes using matplotlib
        plt.style.use(niceplots.get_style())

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

        xyz = np.real(handcrafted_mesh_morph.hc_aero_X)
        x0 = xyz[0::3]
        y0 = xyz[1::3]
        z0 = xyz[2::3]
        mask = np.logical_and(np.abs(x0) < 2, np.abs(y0) < 2)
        mask = np.logical_and(mask, np.abs(z0) < 2)
        ax.scatter(x0[mask], y0[mask], z0[mask], alpha=0.8, s=3)

        u_hc = np.real(handcrafted_mesh_morph.u_hc)
        xdef = x0 + u_hc[0::3]
        ydef = y0 + u_hc[1::3]
        zdef = z0 + u_hc[2::3]
        ax.scatter(xdef[mask], ydef[mask], zdef[mask], alpha=0.2, s=3)
        ax.set_aspect("equal")

        plt.title("Handcrafted mesh morph def")
        _plt_out = os.path.join(meshes_dir, "handcrafted_mesh_morph.png")
        plt.savefig(_plt_out, dpi=400)

        print(f"u_hc = {u_hc}")


if __name__ == "__main__":
    unittest.main()
