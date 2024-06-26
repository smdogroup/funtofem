"""
test_fun3d_interface.py

Run a coupled optimization of the panel thicknesses of the wing structure.
No shape variables are included in this optimization.
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

global_debug_flag = False

# Derivative test stuff
FILENAME = "complex-step-testStruct.txt"
FILEPATH = os.path.join(base_dir, FILENAME)

# FUNTOFEM MODEL
# <----------------------------------------------------
# Freestream quantities -- see README
T_inf = 268.338  # Freestream temperature
q_inf = 1.21945e4  # Dynamic pressure

# Construct the FUNtoFEM model
f2f_model = FUNtoFEMmodel("ssw-sizing1")
tacs_model = caps2tacs.TacsModel.build(
    csm_file=csm_path,
    comm=comm,
    problem_name="capsStruct1",
    active_procs=[0],
    verbosity=1,
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

# ---------------------------------------------------->

# BODIES AND STRUCT DVs
# <----------------------------------------------------

# wing = Body.aeroelastic("wing", boundary=3).relaxation(
#     AitkenRelaxation(theta_init=0.6, theta_max=0.95)
# )
wing = Body.aeroelastic("wing", boundary=3)

# setup the material and shell properties
aluminum = caps2tacs.Isotropic.aluminum().register_to(tacs_model)

# nribs = int(tacs_model.get_config_parameter("nribs"))
# nspars = int(tacs_model.get_config_parameter("nspars"))
nribs = 7
nspars = 4
nOML = nribs - 1

for irib in range(1, nribs + 1):
    name = f"rib{irib}"
    prop = caps2tacs.ShellProperty(
        caps_group=name, material=aluminum, membrane_thickness=0.04
    ).register_to(tacs_model)
    Variable.structural(name, value=0.01).set_bounds(
        lower=0.001, upper=0.15, scale=100.0
    ).register_to(wing)

for ispar in range(1, nspars + 1):
    name = f"spar{ispar}"
    prop = caps2tacs.ShellProperty(
        caps_group=name, material=aluminum, membrane_thickness=0.04
    ).register_to(tacs_model)
    Variable.structural(name, value=0.01).set_bounds(
        lower=0.001, upper=0.15, scale=100.0
    ).register_to(wing)

for iOML in range(1, nOML + 1):
    name = f"OML{iOML}"
    prop = caps2tacs.ShellProperty(
        caps_group=name, material=aluminum, membrane_thickness=0.04
    ).register_to(tacs_model)
    Variable.structural(name, value=0.01).set_bounds(
        lower=0.001, upper=0.15, scale=100.0
    ).register_to(wing)

for prefix in ["LE", "TE"]:
    name = f"{prefix}spar"
    prop = caps2tacs.ShellProperty(
        caps_group=name, material=aluminum, membrane_thickness=0.04
    ).register_to(tacs_model)
    Variable.structural(name, value=0.01).set_bounds(
        lower=0.001, upper=0.15, scale=100.0
    ).register_to(wing)

# register the wing body to the model
wing.register_to(f2f_model)

# ---------------------------------------------------->

# INITIAL STRUCTURE MESH, SINCE NO STRUCT SHAPE VARS
# <----------------------------------------------------

tacs_aim.setup_aim()
tacs_aim.pre_analysis()

comm.barrier()

# ---------------------------------------------------->

# SCENARIOS
# <----------------------------------------------------

# make a funtofem scenario
cruise = Scenario.steady("cruise", steps=100, uncoupled_steps=0)
cruise.adjoint_steps = 150
mass = Function.mass().optimize(
    scale=1.0e-4, objective=True, plot=True, plot_name="mass"
)
ksfailure = Function.ksfailure(ks_weight=150.0, safety_factor=1.5).optimize(
    scale=1.0, upper=1.0, objective=False, plot=True, plot_name="ks-cruise"
)
cruise.include(ksfailure).include(Function.lift()).include(mass)
cruise.set_temperature(T_ref=T_inf, T_inf=T_inf)
cruise.set_flow_ref_vals(qinf=q_inf)
cruise.register_to(f2f_model)

# ---------------------------------------------------->

# COMPOSITE FUNCTIONS
# <----------------------------------------------------

# skin thickness adjacency constraints
variables = f2f_model.get_variables()
section_prefix = ["rib", "OML"]
section_nums = [nribs, nOML]
for isection, prefix in enumerate(section_prefix):
    section_num = section_nums[isection]
    for iconstr in range(1, section_num):
        left_var = f2f_model.get_variables(names=f"{prefix}{iconstr}")
        right_var = f2f_model.get_variables(names=f"{prefix}{iconstr+1}")
        adj_constr = (left_var - right_var) / left_var
        adj_ratio = 0.15
        adj_constr.set_name(f"{prefix}{iconstr}-{iconstr+1}").optimize(
            lower=-adj_ratio, upper=adj_ratio, scale=1.0, objective=False
        ).register_to(f2f_model)

# ---------------------------------------------------->

# DISCIPLINE INTERFACES AND DRIVERS
# <----------------------------------------------------

solvers = SolverManager(comm)
solvers.flow = Fun3dInterface(
    comm,
    f2f_model,
    fun3d_project_name="ssw-pw1.2",
    fun3d_dir="cfd",
    forward_tol=1e-5,
    adjoint_tol=1e-4,
    debug=global_debug_flag,
)
solvers.structural = TacsSteadyInterface.create_from_bdf(
    model=f2f_model,
    comm=comm,
    nprocs=nprocs_tacs,
    bdf_file=tacs_aim.root_dat_file,
    prefix=tacs_aim.root_analysis_dir,
    debug=global_debug_flag,
)

transfer_settings = TransferSettings(npts=200)

# Build the FUNtoFEM driver
f2f_driver = FUNtoFEMnlbgs(
    solvers=solvers,
    transfer_settings=transfer_settings,
    model=f2f_model,
    debug=global_debug_flag,
)

# ---------------------------------------------------->

# Derivative test
# <----------------------------------------------------

if comm.rank == 0:
    f2f_driver.print_summary()
    f2f_model.print_summary()

max_rel_error = TestResult.derivative_test(
    "fun3d+tacs-ssw1",
    model=f2f_model,
    driver=f2f_driver,
    status_file=FILEPATH,
    complex_mode=True,
    epsilon=1e-30,
)

# ---------------------------------------------------->
