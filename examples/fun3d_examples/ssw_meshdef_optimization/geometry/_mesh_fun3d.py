from pathlib import Path
from funtofem.interface import Fun3dModel, Fun3dBC
from mpi4py import MPI

here = Path(__file__).parent
comm = MPI.COMM_WORLD

# Set whether to build an inviscid or viscous mesh
# ------------------------------------------------
# case = "inviscid"
case = "turbulent"
if case == "inviscid":
    project_name = "ssw-inviscid"
else:  # turbulent
    project_name = "ssw-turb"
# ------------------------------------------------

# Set up FUN3D model, AIMs, and turn on the flow view
# ------------------------------------------------
fun3d_model = Fun3dModel.build(csm_file="ssw.csm", comm=comm, project_name=project_name)
aflr_aim = fun3d_model.aflr_aim
fun3d_aim = fun3d_model.fun3d_aim
fun3d_aim.set_config_parameter("view:flow", 1)
fun3d_aim.set_config_parameter("view:struct", 0)
# ------------------------------------------------

global_max = 10
global_min = 0.1

aflr_aim.set_surface_mesh(
    ff_growth=1.2,
    mesh_length=1.0,
    min_scale=global_min,
    max_scale=global_max,
    use_quads=True,
)
# if comm.rank == 0:
#     aim = aflr_aim._aflr4_aim
#     aim.input.Mesh_Sizing = {"wingTip" : {"numEdgePoints" : 50}}
# aflr_aim.set_boundary_layer(initial_spacing=0.001, thickness=0.1)
aflr_aim._aflr4_aim.input.Mesh_Sizing = {
    "rootEdgeMesh": {"numEdgePoints": 150},
    "wingJointEdgeMesh": {"numEdgePoints": 150},
}
if case == "inviscid":
    Fun3dBC.inviscid(caps_group="wing").register_to(fun3d_model)
else:
    aflr_aim.set_boundary_layer(
        initial_spacing=0.001, max_layers=35, thickness=0.01, use_quads=True
    )
    Fun3dBC.viscous(caps_group="wing", wall_spacing=1).register_to(fun3d_model)

refinement = 1

FluidMeshOptions = {"aflr4AIM": {}, "aflr3AIM": {}}

FluidMeshOptions["aflr4AIM"]["Mesh_Sizing"] = {
    "leEdgeMesh": {"scaleFactor": 0.08, "edgeWeight": 1.0},
    "teEdgeMesh": {"scaleFactor": 0.2},
    "tipEdgeMesh": {"scaleFactor": 0.5},
    "rootEdgeMesh": {"scaleFactor": 0.5},
    "wingMesh": {"scaleFactor": 1.0, "AFLR4_quad_local": 1.0, "min_scale": global_min},
}

FluidMeshOptions["aflr4AIM"]["curv_factor"] = 0.001
FluidMeshOptions["aflr4AIM"]["ff_cdfr"] = 1.2
FluidMeshOptions["aflr4AIM"]["mer_all"] = 1

aflr_aim.saveDictOptions(FluidMeshOptions)

Fun3dBC.SymmetryY(caps_group="SymmetryY").register_to(fun3d_model)
Fun3dBC.Farfield(caps_group="Farfield").register_to(fun3d_model)

fun3d_model.setup()
fun3d_aim.pre_analysis()
