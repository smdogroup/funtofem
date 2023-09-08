from pathlib import Path
from mpi4py import MPI

# from funtofem import TransferScheme
from funtofem.model import (
    FUNtoFEMmodel,
    Variable,
    Scenario,
    Body,
    Function,
)
from funtofem.interface import SolverManager, Fun3dBC, Fun3dModel
from funtofem.driver import FuntofemShapeDriver, Fun3dRemote
from tacs import caps2tacs

comm = MPI.COMM_WORLD
here = Path(__file__).parent
csm_file = here.joinpath("meshes", "hsct.csm")
analysis_file = here.joinpath("3_run_funtofem_analysis.py")
fun3d_dir = here.joinpath("meshes")

# build the funtofem model with one body and scenario
model = FUNtoFEMmodel("wing")

# design the FUN3D aero shape model
fun3d_model = Fun3dModel.build(csm_file=csm_file, comm=comm)
aflr_aim = fun3d_model.aflr_aim

fun3d_aim = fun3d_model.fun3d_aim
fun3d_aim.set_config_parameter("mode:flow", 1)
fun3d_aim.set_config_parameter("mode:struct", 0)

aflr_aim.set_surface_mesh(ff_growth=1.4, mesh_length=5.0)
Fun3dBC.inviscid(caps_group="wall", wall_spacing=0.001).register_to(fun3d_model)
Fun3dBC.inviscid(caps_group="staticWall", wall_spacing=0.001).register_to(fun3d_model)
Fun3dBC.Farfield(caps_group="Farfield").register_to(fun3d_model)
fun3d_model.setup()
model.flow = fun3d_model

# design the TACS struct shape model
tacs_model = caps2tacs.TacsModel.build(csm_file=csm_file, comm=comm)
tacs_model.mesh_aim.set_mesh(  # need a refined-enough mesh for the derivative test to pass
    edge_pt_min=15,
    edge_pt_max=20,
    global_mesh_size=0.01,
    max_surf_offset=0.01,
    max_dihedral_angle=5,
).register_to(
    tacs_model
)
tacs_aim = tacs_model.tacs_aim
tacs_aim.set_config_parameter("mode:flow", 0)
tacs_aim.set_config_parameter("mode:struct", 1)
model.structural = tacs_model

# setup the funtofem bodies
wing = Body.aeroelastic("wing", boundary=2)

# setup the material and shell properties
aluminum = caps2tacs.Isotropic.aluminum().register_to(tacs_model)

nribs = int(tacs_model.get_config_parameter("nribs"))
nspars = int(tacs_model.get_config_parameter("nspars"))
nOML = int(tacs_model.get_output_parameter("nOML"))

for irib in range(1, nribs + 1):
    caps2tacs.ShellProperty(
        caps_group=f"rib{irib}", material=aluminum, membrane_thickness=0.05
    ).register_to(tacs_model)
for ispar in range(1, nspars + 1):
    caps2tacs.ShellProperty(
        caps_group=f"spar{ispar}", material=aluminum, membrane_thickness=0.05
    ).register_to(tacs_model)
for iOML in range(1, nOML + 1):
    caps2tacs.ShellProperty(
        caps_group="OML", material=aluminum, membrane_thickness=0.03
    ).register_to(tacs_model)
for group in ["LEspar", "TEspar"]:
    caps2tacs.ShellProperty(
        caps_group=group, material=aluminum, membrane_thickness=0.03
    ).register_to(tacs_model)

# register the aero-struct shape variable
Variable.shape(name="aoa").set_bounds(lower=-5.0, value=0.0, upper=5.0).register_to(
    wing
)
wing.register_to(model)

# add remaining constraints to tacs model
caps2tacs.PinConstraint("root").register_to(tacs_model)

# define the funtofem scenarios
test_scenario = (
    Scenario.steady("euler", steps=5000)
    .set_temperature(T_ref=300.0, T_inf=300.0)
    .fun3d_project("funtofem_CAPS")
)
test_scenario.adjoint_steps = 4000
# test_scenario.get_variable("AOA")
test_scenario.include(Function.lift()).include(Function.drag())
test_scenario.include(Function.ksfailure(ks_weight=10.0))
test_scenario.register_to(model)

# build the solvers and coupled driver
solvers = SolverManager(comm)
fun3d_remote = Fun3dRemote(analysis_file, fun3d_dir, nprocs=48)
driver = FuntofemShapeDriver.aero_remesh(solvers, model, fun3d_remote)
