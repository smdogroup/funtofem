"""
3_geom_twist.py

Run a coupled optimization of the geometric twist at each station.
No thickness variables here.
TODO : this example isn't done yet
"""

from pyoptsparse import SNOPT, Optimization
from funtofem import *
from mpi4py import MPI
from tacs import caps2tacs
import os

comm = MPI.COMM_WORLD

base_dir = os.path.dirname(os.path.abspath(__file__))
csm_path = os.path.join(base_dir, "geometry", "ssw.csm")

# Optimization options
hot_start = False
store_history = True

nprocs_tacs = 8

# FUNTOFEM MODEL
# <----------------------------------------------------
# Freestream quantities -- see README
T_inf = 268.338  # Freestream temperature
q_inf = 1.21945e4  # Dynamic pressure

# Construct the FUNtoFEM model
f2f_model = FUNtoFEMmodel("ssw-twist")
tacs_model = caps2tacs.TacsModel.build(
    csm_file=csm_path,
    comm=comm,
    problem_name="capsStruct3",
    active_procs=[0],
    verbosity=0,
)
tacs_model.mesh_aim.set_mesh(
    edge_pt_min=2,
    edge_pt_max=20,
    global_mesh_size=0.3,
    max_surf_offset=0.2,
    max_dihedral_angle=15,
).register_to(tacs_model)
f2f_model.structural = tacs_model

tacs_aim = tacs_model.tacs_aim
tacs_aim.set_config_parameter("view:flow", 0)
tacs_aim.set_config_parameter("view:struct", 1)

for proc in tacs_aim.active_procs:
    if comm.rank == proc:
        aim = tacs_model.mesh_aim.aim
        aim.input.Mesh_Sizing = {
            "chord": {"numEdgePoints": 20},
            "span": {"numEdgePoints": 8},
            "vert": {"numEdgePoints": 4},
        }

# add tacs constraints in
caps2tacs.PinConstraint("root").register_to(tacs_model)

# FUN3D AIM Stuff
# Set up FUN3D model, AIMs, and turn on the flow view
# ------------------------------------------------
fun3d_model = Fun3dModel.build(
    csm_file=csm_path, comm=comm, project_name="ssw-turb", mesh_morph=True, verbosity=0
)
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

if comm.rank == 0:
    aflr_aim._aflr4_aim.input.Mesh_Sizing = {
        "rootEdgeMesh": {"numEdgePoints": 150},
        "wingJointEdgeMesh": {"numEdgePoints": 150},
    }
case = "inviscid"
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
f2f_model.flow = fun3d_model

# ---------------------------------------------------->

# BODIES AND STRUCT DVs
# <----------------------------------------------------

wing = Body.aeroelastic("wing", boundary=3)

# setup the material and shell properties
aluminum = caps2tacs.Isotropic.aluminum().register_to(tacs_model)

nribs = int(tacs_model.get_config_parameter("nribs"))
nspars = int(tacs_model.get_config_parameter("nspars"))
nOML = nribs - 1

for irib in range(1, nribs + 1):
    name = f"rib{irib}"
    prop = caps2tacs.ShellProperty(
        caps_group=name, material=aluminum, membrane_thickness=0.04
    ).register_to(tacs_model)
    Variable.structural(name, value=0.01).set_bounds(
        lower=0.001,
        upper=0.15,
        scale=100.0,
        active=False,
    ).register_to(wing)

for ispar in range(1, nspars + 1):
    name = f"spar{ispar}"
    prop = caps2tacs.ShellProperty(
        caps_group=name, material=aluminum, membrane_thickness=0.04
    ).register_to(tacs_model)
    Variable.structural(name, value=0.01).set_bounds(
        lower=0.001,
        upper=0.15,
        scale=100.0,
        active=False,
    ).register_to(wing)

for iOML in range(1, nOML + 1):
    name = f"OML{iOML}"
    prop = caps2tacs.ShellProperty(
        caps_group=name, material=aluminum, membrane_thickness=0.04
    ).register_to(tacs_model)
    Variable.structural(name, value=0.01).set_bounds(
        lower=0.001,
        upper=0.15,
        scale=100.0,
        active=False,
    ).register_to(wing)

for prefix in ["LE", "TE"]:
    name = f"{prefix}spar"
    prop = caps2tacs.ShellProperty(
        caps_group=name, material=aluminum, membrane_thickness=0.04
    ).register_to(tacs_model)
    Variable.structural(name, value=0.01).set_bounds(
        lower=0.001,
        upper=0.15,
        scale=100.0,
        active=False,
    ).register_to(wing)

for prefix in range(1, 4 + 1):
    Variable.shape(f"twist{prefix}", value=1.0).set_bounds(
        lower=-10.0,
        upper=10.0,
        active=True,
    ).register_to(wing)

# register the wing body to the model
wing.register_to(f2f_model)

# ---------------------------------------------------->

# INITIAL STRUCTURE MESH, SINCE NO STRUCT SHAPE VARS
# <----------------------------------------------------

tacs_aim.setup_aim()
tacs_aim.pre_analysis()

# ---------------------------------------------------->

# SCENARIOS
# <----------------------------------------------------

# make a funtofem scenario
cruise = Scenario.steady("cruise", steps=300, uncoupled_steps=0)
ksfailure = Function.ksfailure(ks_weight=10.0, safety_factor=1.5).optimize(
    scale=1.0, upper=1.0, objective=False, plot=True, plot_name="ks-cruise"
)
cl_cruise = Function.lift(body=0)
aoa_cruise = cruise.get_variable("AOA").set_bounds(
    lower=-4, value=2.0, upper=15, active=False
)
cruise.include(cl_cruise).include(ksfailure)
cruise.set_temperature(T_ref=T_inf, T_inf=T_inf)
cruise.set_flow_ref_vals(qinf=q_inf)
cruise.register_to(f2f_model)

# ---------------------------------------------------->

# COMPOSITE FUNCTIONS
# <----------------------------------------------------

# skin thickness adjacency constraints
# variables = f2f_model.get_variables()
# section_prefix = ["rib", "OML"]
# section_nums = [nribs, nOML]
# for isection, prefix in enumerate(section_prefix):
#     section_num = section_nums[isection]
#     for iconstr in range(1, section_num):
#         left_var = f2f_model.get_variables(names=f"{prefix}{iconstr}")
#         right_var = f2f_model.get_variables(names=f"{prefix}{iconstr+1}")
#         adj_constr = (left_var - right_var) / left_var
#         adj_ratio = 0.15
#         adj_constr.set_name(f"{prefix}{iconstr}-{iconstr+1}").optimize(
#             lower=-adj_ratio, upper=adj_ratio, scale=1.0, objective=False
#         ).register_to(f2f_model)

cl_target = 1.2

cruise_lift = (cl_cruise - cl_target) ** 2
cruise_lift.set_name(f"LiftObj").optimize(
    lower=-1e-2, upper=10, scale=1.0, objective=True, plot=True, plot_name="Lift-Obj"
).register_to(f2f_model)

# ---------------------------------------------------->

# DISCIPLINE INTERFACES AND DRIVERS
# <----------------------------------------------------

solvers = SolverManager(comm)
solvers.flow = Fun3dInterface(
    comm,
    f2f_model,
    fun3d_project_name="ssw-turb",
    fun3d_dir="cfd",
    forward_tolerance=1e-4,
    adjoint_tolerance=1e-1,
    auto_coords=False,
)

transfer_settings = TransferSettings(npts=200)

# Build the FUNtoFEM driver
f2f_driver = FuntofemShapeDriver.aero_morph(
    solvers=solvers,
    model=f2f_model,
    transfer_settings=transfer_settings,
    struct_nprocs=nprocs_tacs,
)

# ---------------------------------------------------->

# PYOPTSPARSE OPTMIZATION
# <----------------------------------------------------

# create an OptimizationManager object for the pyoptsparse optimization problem
# design_in_file = os.path.join(base_dir, "design", "design-2.txt")
design_out_file = os.path.join(base_dir, "design", "design-3.txt")


design_folder = os.path.join(base_dir, "design")
if comm.rank == 0:
    if not os.path.exists(design_folder):
        os.mkdir(design_folder)
history_file = os.path.join(design_folder, "design-3.hst")
store_history_file = history_file if store_history else None
hot_start_file = history_file if hot_start else None

# Reload the previous design
# f2f_model.read_design_variables_file(comm, design_in_file)

if comm.rank == 0:
    f2f_driver.print_summary()
    f2f_model.print_summary()

manager = OptimizationManager(
    f2f_driver,
    design_out_file=design_out_file,
    hot_start=hot_start,
    debug=True,
    hot_start_file=hot_start_file,
)

# create the pyoptsparse optimization problem
opt_problem = Optimization("sswOpt", manager.eval_functions)

# add funtofem model variables to pyoptsparse
manager.register_to_problem(opt_problem)

# run an SNOPT optimization
snoptimizer = SNOPT(options={"IPRINT": 1})

sol = snoptimizer(
    opt_problem,
    sens=manager.eval_gradients,
    storeHistory=store_history_file,
    hotStart=hot_start_file,
)

# print final solution
sol_xdict = sol.xStar
if comm.rank == 0:
    print(f"Final solution = {sol_xdict}", flush=True)

# ---------------------------------------------------->
