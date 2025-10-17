"""
_oneway_sizing.py

Brian J. Burke, Georgia Tech 2024

Oneway sizing optimization of the initial SSW structure.

Run a oneway-coupled optimization of the panel thicknesses of the wing structure. The
flow solver is run first to generate aerodynamic loads on the structure which are saved
to uncoupled_loads.txt.
A FUNtoFEM model is created with an aeroelastic body which only iterates through TACS
to solve the structural sizing optimization problem.
Local machine optimization for the panel thicknesses using nribs-1 OML panels and nribs-1 LE panels
"""

# from pyoptsparse import SLSQP, Optimization
from pyoptsparse import SNOPT, Optimization

# script inputs
hot_start = False
store_history = True
test_derivatives = False

# import openmdao.api as om
from funtofem import *
from mpi4py import MPI
from tacs import caps2tacs
import os

comm = MPI.COMM_WORLD

base_dir = os.path.dirname(os.path.abspath(__file__))
csm_path = os.path.join(base_dir, "geometry", "ssw.csm")

nprocs = 4

# Freestream quantities -- see README
T_inf = 268.338  # Freestream temperature
q_inf = 1.21945e4  # Dynamic pressure

# F2F MODEL and SHAPE MODELS
# ----------------------------------------

f2f_model = FUNtoFEMmodel("ssw-sizing-1way")
tacs_model = caps2tacs.TacsModel.build(
    csm_file=csm_path,
    comm=comm,
    problem_name="capsStruct0",
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

# BODIES AND STRUCT DVs
# -------------------------------------------------

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

# INITIAL STRUCTURE MESH, SINCE NO STRUCT SHAPE VARS
# --------------------------------------------------

tacs_aim.setup_aim()
tacs_aim.pre_analysis()

# SCENARIOS
# ----------------------------------------------------

# make a funtofem scenario
cruise = Scenario.steady("cruise", steps=300, uncoupled_steps=0)
mass = Function.mass().optimize(
    scale=1.0e-4, objective=True, plot=True, plot_name="mass"
)
ksfailure = Function.ksfailure(ks_weight=10.0, safety_factor=1.5).optimize(
    scale=1.0, upper=1.0, objective=False, plot=True, plot_name="ks-cruise"
)
cruise.include(mass).include(ksfailure)
cruise.set_temperature(T_ref=T_inf, T_inf=T_inf)
cruise.set_flow_ref_vals(qinf=q_inf)
cruise.register_to(f2f_model)

# COMPOSITE FUNCTIONS
# -------------------------------------------------------

# skin thickness adjacency constraints
if not test_derivatives:
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

# DISCIPLINE INTERFACES AND DRIVERS
# -----------------------------------------------------

solvers = SolverManager(comm)
solvers.structural = TacsSteadyInterface.create_from_bdf(
    model=f2f_model,
    comm=comm,
    nprocs=nprocs,
    bdf_file=tacs_aim.root_dat_file,
    prefix=tacs_aim.root_analysis_dir,
)

# read in aero loads
aero_loads_file = os.path.join(os.getcwd(), "cfd", "loads", "uncoupled_loads.txt")

transfer_settings = TransferSettings(npts=200)

# build the shape driver from the file
tacs_driver = OnewayStructDriver.prime_loads_from_file(
    filename=aero_loads_file,
    solvers=solvers,
    model=f2f_model,
    nprocs=nprocs,
    transfer_settings=transfer_settings,
)

if test_derivatives:  # test using the finite difference test
    # start_time = time.time()

    # run the finite difference test
    max_rel_error = TestResult.derivative_test(
        "fun3d+tacs-ssw1",
        model=f2f_model,
        driver=tacs_driver,
        status_file="1-derivs.txt",
        complex_mode=False,
        epsilon=1e-4,
    )
    exit()

# PYOPTSPARSE OPTMIZATION
# -------------------------------------------------------------

# create an OptimizationManager object for the pyoptsparse optimization problem
# design_in_file = os.path.join(base_dir, "design", "sizing.txt")
design_out_file = os.path.join(base_dir, "design", "sizing-oneway.txt")

design_folder = os.path.join(base_dir, "design")
if comm.rank == 0:
    if not os.path.exists(design_folder):
        os.mkdir(design_folder)
history_file = os.path.join(design_folder, "sizing-oneway.hst")
store_history_file = history_file if store_history else None
hot_start_file = history_file if hot_start else None

manager = OptimizationManager(
    tacs_driver,
    design_out_file=design_out_file,
    hot_start=hot_start,
    debug=True,
    hot_start_file=hot_start_file,
)

if comm.rank == 0:
    # f2f_driver.print_summary()
    f2f_model.print_summary()

# create the pyoptsparse optimization problem
opt_problem = Optimization("sswOpt", manager.eval_functions)

# add funtofem model variables to pyoptsparse
manager.register_to_problem(opt_problem)

# run an SNOPT optimization
snoptimizer = SNOPT()

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
