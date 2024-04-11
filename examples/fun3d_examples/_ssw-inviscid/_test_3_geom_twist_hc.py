"""
3_geom_twist.py

Run a coupled optimization of the geometric twist at each station.
No thickness variables here.
"""

from pyoptsparse import SNOPT, Optimization
from funtofem import *
from mpi4py import MPI
from tacs import caps2tacs
import os
import time

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
    csm_file=csm_path,
    comm=comm,
    project_name="ssw-inviscid",
    verbosity=0,
    mesh_morph=True,
    volume_mesh="aflr3",
    surface_mesh="egads",
)
mesh_aim = fun3d_model.mesh_aim
fun3d_aim = fun3d_model.fun3d_aim
fun3d_aim.set_config_parameter("view:flow", 1)
fun3d_aim.set_config_parameter("view:struct", 0)
# ------------------------------------------------

global_max = 10
global_min = 0.1

mesh_aim.surface_aim.set_surface_mesh(
    edge_pt_min=15,
    edge_pt_max=20,
    mesh_elements="Mixed",
    global_mesh_size=0.5,
    max_surf_offset=0.01,
    max_dihedral_angle=15,
)

case = "inviscid"
if case == "inviscid":
    Fun3dBC.inviscid(caps_group="wing").register_to(fun3d_model)
else:
    aflr_aim.set_boundary_layer(
        initial_spacing=0.001, max_layers=35, thickness=0.01, use_quads=True
    )
    Fun3dBC.viscous(caps_group="wing", wall_spacing=1).register_to(fun3d_model)

refinement = 1

FluidMeshOptions = {"egadsTessAIM": {}, "aflr3AIM": {}}

mesh_aim.saveDictOptions(FluidMeshOptions)

Fun3dBC.SymmetryY(caps_group="SymmetryY").register_to(fun3d_model)
Fun3dBC.Farfield(caps_group="Farfield").register_to(fun3d_model)

fun3d_model.setup()
f2f_model.flow = fun3d_model

# ---------------------------------------------------->

# BODIES AND STRUCT DVs
# <----------------------------------------------------

wing = Body.aeroelastic("wing", boundary=2)

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

N = 1 # 4, all four or just one of the twist variables
for prefix in range(1, N + 1):
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
cruise = Scenario.steady(
    "cruise_inviscid",
    steps=30,
    forward_coupling_frequency=10,  # 300 total fun3d steps
    adjoint_steps=100,
    adjoint_coupling_frequency=30,  # 3000 total adjoint steps
    uncoupled_steps=0,
)
cruise.set_stop_criterion(
    early_stopping=True, min_forward_steps=10, min_adjoint_steps=10
)
cruise.fun3d_project_name = "ssw-inviscid"
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
solvers.flow = Fun3d14Interface(
    comm,
    f2f_model,
    fun3d_dir="cfd",
    forward_stop_tolerance=1e-15,
    forward_min_tolerance=1e-11,
    adjoint_stop_tolerance=1e-14,
    adjoint_min_tolerance=1e-11,
)
# NOTE : the user needs to have put their handcrafted mesh in the Scenario/Flow directory already

# create handcrafted mesh morph object after FUN3D is built
handcrafted_mesh_morph = HandcraftedMeshMorph(
    comm=comm,
    model=f2f_model,
    transfer_settings=TransferSettings(npts=200, beta=0.5),
)
fun3d_model.handcrafted_mesh_morph = handcrafted_mesh_morph

transfer_settings = TransferSettings(npts=200)

# Build the FUNtoFEM driver
f2f_driver = FuntofemShapeDriver.aero_morph(
    solvers=solvers,
    model=f2f_model,
    transfer_settings=transfer_settings,
    struct_nprocs=nprocs_tacs,
    reload_funtofem_states=True,
)

#test_derivatives = True
#if test_derivatives:  # test using the finite difference test
    # load the previous design
    # design_in_file = os.path.join(base_dir, "design", "sizing-oneway.txt")
    # f2f_model.read_design_variables_file(comm, design_in_file)

start_time = time.time()

# run the finite difference test
max_rel_error = TestResult.derivative_test(
    "case3 handcrafted derivatives",
    model=f2f_model,
    driver=f2f_driver,
    status_file="3-hc-derivs.txt",
    complex_mode=False,
    epsilon=1e-5,
)

end_time = time.time()
dt = end_time - start_time
if comm.rank == 0:
    print(f"total time for ssw derivative test is {dt} seconds", flush=True)
    print(f"max rel error = {max_rel_error}", flush=True)