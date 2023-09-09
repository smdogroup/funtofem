from pathlib import Path
from funtofem.interface import Fun3dModel, Fun3dBC
from mpi4py import MPI

here = Path(__file__).parent
comm = MPI.COMM_WORLD
csm_file = str(here.joinpath("meshes").joinpath("hsct.csm"))

fun3d_model = Fun3dModel.build(csm_file=csm_file, comm=comm, project_name="hsct")
aflr_aim = fun3d_model.aflr_aim
fun3d_aim = fun3d_model.fun3d_aim
fun3d_aim.set_config_parameter("mode:flow", 1)
fun3d_aim.set_config_parameter("mode:struct", 0)

aflr_aim.set_surface_mesh(ff_growth=1.4, mesh_length=5.0)
# aflr_aim.set_boundary_layer(initial_spacing=0.001, thickness=0.1)
Fun3dBC.viscous(caps_group="wall", wall_spacing=0.0001).register_to(fun3d_model)
Fun3dBC.viscous(caps_group="staticWall", wall_spacing=0.0001).register_to(fun3d_model)
Fun3dBC.SymmetryY(caps_group="SymmetryY").register_to(fun3d_model)
Fun3dBC.Farfield(caps_group="Farfield").register_to(fun3d_model)
fun3d_model.setup()
fun3d_aim.pre_analysis()
