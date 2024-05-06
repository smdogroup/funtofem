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
    edge_pt_min=10,
    edge_pt_max=12,  # Farfield edges are being controlled by edge_pt_max for some reason
    mesh_elements="Mixed",
    global_mesh_size=1.0,
    max_surf_offset=0.01,
    max_dihedral_angle=15,
)

le_dx = 0.005
te_dx = 0.010
tip_dy = 0.5
num_pts_up = 40
num_pts_bot = 40
num_pts_y = 60
mesh_aim.surface_aim.aim.input.Mesh_Sizing = {
    "teEdgeMesh": {
        "numEdgePoints": num_pts_y,
        "edgeDistribution": "Tanh",
        "initialNodeSpacing": [0, tip_dy],
    },
    "leEdgeMesh": {
        "numEdgePoints": num_pts_y,
        "edgeDistribution": "Tanh",
        "initialNodeSpacing": [0, tip_dy],
    },
    "tipEdgeMesh": {
        "numEdgePoints": num_pts_bot,
        "edgeDistribution": "Tanh",
        "initialNodeSpacing": [le_dx, te_dx],
    },
    "rootEdgeMesh": {
        "numEdgePoints": num_pts_bot,
        "edgeDistribution": "Tanh",
        "initialNodeSpacing": [le_dx, te_dx],
    },
    "tipMesh": {"tessParams": [0.02, 0.01, 20.0]},
}

# Can coarsen up by changing tipMesh 1st tessParams a
# bit, or by changing num_pts_up/num_pts_bot/num_pts_y

if case == "inviscid":
    Fun3dBC.inviscid(caps_group="wing").register_to(fun3d_model)
else:
    mesh_aim.volume_aim.set_boundary_layer(
        initial_spacing=5e-5, max_layers=55, thickness=0.1, use_quads=True
    )
    Fun3dBC.viscous(caps_group="wing", wall_spacing=1).register_to(fun3d_model)

refinement = 1

FluidMeshOptions = {"egadsTessAIM": {}, "aflr3AIM": {}}

mesh_aim.saveDictOptions(FluidMeshOptions)

# Fun3dBC.SymmetryY(caps_group="SymmetryY").register_to(fun3d_model)
Fun3dBC.Farfield(caps_group="Farfield").register_to(fun3d_model)

fun3d_model.setup()
fun3d_aim.pre_analysis()

# if comm.rank == 0:
#     mesh_aim.surface_aim.aim.runAnalysis()
#     mesh_aim.surface_aim.aim.geometry.view()
# exit()

# fun3d_aim.pre_analysis()
