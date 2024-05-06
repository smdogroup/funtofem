from pathlib import Path
from funtofem.interface import Fun3dModel, Fun3dBC
from mpi4py import MPI

here = Path(__file__).parent
comm = MPI.COMM_WORLD

# Set whether to build an inviscid or viscous mesh
# ------------------------------------------------
# setting up for laminar flow with no BL (very coarse mesh for quick demo)
project_name = "ssw-turb"

# ------------------------------------------------

# Set up FUN3D model, AIMs, and turn on the flow view
# ------------------------------------------------
fun3d_model = Fun3dModel.build(
    csm_file="ssw.csm",
    comm=comm,
    project_name=project_name,
    problem_name="capsFluidEgads",
    volume_mesh="aflr3",
    surface_mesh="egads",
)
mesh_aim = fun3d_model.mesh_aim
fun3d_aim = fun3d_model.fun3d_aim
fun3d_aim.set_config_parameter("view:flow", 1)
fun3d_aim.set_config_parameter("view:struct", 0)
# ------------------------------------------------

mesh_aim.surface_aim.set_surface_mesh(
    edge_pt_min=15,
    edge_pt_max=20,
    mesh_elements="Mixed",
    global_mesh_size=0.5,
    max_surf_offset=0.01,
    max_dihedral_angle=15,
)

mesh_aim.volume_aim.set_boundary_layer(
    initial_spacing=0.001, max_layers=35, thickness=0.01, use_quads=True
)
Fun3dBC.viscous(caps_group="wing", wall_spacing=1).register_to(fun3d_model)

refinement = 1

FluidMeshOptions = {"egadsTessAIM": {}, "aflr3AIM": {}}

mesh_aim.saveDictOptions(FluidMeshOptions)

Fun3dBC.SymmetryY(caps_group="SymmetryY").register_to(fun3d_model)
Fun3dBC.Farfield(caps_group="Farfield").register_to(fun3d_model)

fun3d_model.setup()
fun3d_aim.pre_analysis()


