"""
Sean P. Engelstad, Georgia Tech 2023
"""

from pathlib import Path
from funtofem.interface import Fun3dModel, Fun3dBC
from mpi4py import MPI

here = Path(__file__).parent
comm = MPI.COMM_WORLD
csm_file = str(here.joinpath("geometry").joinpath("sst_v2.csm"))

# case = "inviscid"  # "turbulent"
case = "turbulent"
if case == "inviscid":
    project_name = "sst-inviscid"
else:  # turbulent
    project_name = "sst-turb"

fun3d_model = Fun3dModel.build(csm_file=csm_file, comm=comm, project_name=project_name)
aflr_aim = fun3d_model.aflr_aim
fun3d_aim = fun3d_model.fun3d_aim
fun3d_aim.set_config_parameter("mode:flow", 1)
fun3d_aim.set_config_parameter("mode:struct", 0)

if comm.rank == 0:
    aim = aflr_aim.surface_aim
    aim.input.Mesh_Sizing = {"botFuse": {"scaleFactor": 0.1}}

# min_scale has the greatest effect on mesh size (scale that to affect # mesh elements)
# min_scale = 0.005 => 3.5M cells
# min_scale = 0.01 => 1.5M cells
# min_scale = 0.02 => 600k cells
aflr_aim.set_surface_mesh(
    ff_growth=1.2, mesh_length=30.0, min_scale=0.005, max_scale=0.5, use_quads=True
)

if case == "inviscid":
    Fun3dBC.inviscid(caps_group="wall").register_to(fun3d_model)
    Fun3dBC.inviscid(caps_group="staticWall").register_to(fun3d_model)
else:
    Fun3dBC.viscous(caps_group="wall", wall_spacing=0.0001).register_to(fun3d_model)
    Fun3dBC.viscous(caps_group="staticWall", wall_spacing=0.0001).register_to(
        fun3d_model
    )

refinement = 1

FluidMeshOptions = {"aflr4AIM": {}, "aflr3AIM": {}}

FluidMeshOptions["aflr4AIM"]["Mesh_Sizing"] = {
    "wingUpMesh": {"edgeWeight": 1.0},
    "wingDownMesh": {"edgeWeight": 1.0},
    "fuselageDownMesh": {"edgeWeight": 1.0},
    "fuselageUpMesh": {"edgeWeight": 1.0},
    "centerlineUp": {"scaleFactor": 0.5 ** (refinement / 2)},
    "centerlineDown": {"scaleFactor": 0.1 ** (refinement / 2)},
    "centerlineCenter": {"scaleFactor": 0.5 ** (refinement / 2)},
    "wingFuseJunc": {"scaleFactor": 0.5 ** (refinement / 2)},
    "tipEdgeMesh": {"scaleFactor": 0.5 ** (refinement / 2)},
    "wingUpMesh": {"cdfr": 1.2},
}

aflr_aim.saveDictOptions(FluidMeshOptions)

Fun3dBC.SymmetryY(caps_group="SymmetryY").register_to(fun3d_model)
Fun3dBC.Farfield(caps_group="Farfield").register_to(fun3d_model)
fun3d_model.setup()
fun3d_aim.pre_analysis()
