"""
Sean P. Engelstad, Georgia Tech 2023

Run a FUN3D analysis with the Fun3dOnewayDriver
then with those aero loads still in the F2F Body object
determine the optimal panel thicknesses using oneway-coupled
structural optimization in TACS.

NOTE: you need to run _mesh_fun3d.py first and move the .ugrid
FUN3D mesh into the meshes/turbulent folder first.
"""

from pyoptsparse import SNOPT, Optimization

# script inputs
hot_start = False
store_history = True

# import openmdao.api as om
from funtofem import *
from mpi4py import MPI
from tacs import caps2tacs
import os

comm = MPI.COMM_WORLD

base_dir = os.path.dirname(os.path.abspath(__file__))
csm_path = os.path.join(base_dir, "geometry", "sst_v2.csm")

# F2F MODEL and SHAPE MODELS
# ----------------------------------------

f2f_model = FUNtoFEMmodel("sst-sizing")
tacs_model = caps2tacs.TacsModel.build(
    csm_file=csm_path, comm=comm, problem_name="capsStruct1"
)
tacs_model.mesh_aim.set_mesh(  # need a refined-enough mesh for the derivative test to pass
    edge_pt_min=2,
    edge_pt_max=20,
    global_mesh_size=0.3,
    max_surf_offset=0.2,
    max_dihedral_angle=15,
).register_to(
    tacs_model
)
f2f_model.structural = tacs_model

tacs_aim = tacs_model.tacs_aim
tacs_aim.set_config_parameter("mode:flow", 0)
tacs_aim.set_config_parameter("mode:struct", 1)

if comm.rank == 0:
    aim = tacs_model.mesh_aim.aim
    aim.input.Mesh_Sizing = {
        "horizX": {"numEdgePoints": 20},
        "horizY": {"numEdgePoints": 8},
        "vert": {"numEdgePoints": 4},
    }

# add tacs constraints in
caps2tacs.PinConstraint("root").register_to(tacs_model)
caps2tacs.PinConstraint("wingFuse").register_to(tacs_model)
caps2tacs.TemperatureConstraint("midplane").register_to(tacs_model)

# BODIES AND STRUCT DVs
# -------------------------------------------------

wing = Body.aerothermoelastic("wing", boundary=4)

# setup the material and shell properties
titanium_alloy = caps2tacs.Isotropic.titanium_alloy().register_to(tacs_model)

nribs = int(tacs_model.get_config_parameter("wing:nribs"))
nspars = int(tacs_model.get_config_parameter("wing:nspars"))
nOML = int(tacs_aim.get_output_parameter("wing:nOML"))

for irib in range(1, nribs + 1):
    name = f"rib{irib}"
    prop = caps2tacs.ShellProperty(
        caps_group=name, material=titanium_alloy, membrane_thickness=0.04
    ).register_to(tacs_model)
    Variable.structural(name, value=0.01).set_bounds(
        lower=0.001, upper=0.15, scale=100.0
    ).register_to(wing)

for ispar in range(1, nspars + 1):
    name = f"spar{ispar}"
    prop = caps2tacs.ShellProperty(
        caps_group=name, material=titanium_alloy, membrane_thickness=0.04
    ).register_to(tacs_model)
    Variable.structural(name, value=0.01).set_bounds(
        lower=0.001, upper=0.15, scale=100.0
    ).register_to(wing)

for iOML in range(1, nOML + 1):
    name = f"OML{iOML}"
    prop = caps2tacs.ShellProperty(
        caps_group=name, material=titanium_alloy, membrane_thickness=0.04
    ).register_to(tacs_model)
    Variable.structural(name, value=0.01).set_bounds(
        lower=0.001, upper=0.15, scale=100.0
    ).register_to(wing)

for prefix in ["LE", "TE"]:
    name = f"{prefix}spar"
    prop = caps2tacs.ShellProperty(
        caps_group=name, material=titanium_alloy, membrane_thickness=0.04
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
climb = Scenario.steady("climb_inviscid", steps=350, uncoupled_steps=200)  # 2000
mass = Function.mass().optimize(
    scale=1.0e-4, objective=True, plot=True, plot_name="mass"
)
ksfailure = Function.ksfailure(ks_weight=10.0, safety_factor=1.5).optimize(
    scale=1.0, upper=1.0, objective=False, plot=True, plot_name="ks-climb"
)
climb.include(mass).include(ksfailure)
climb.set_temperature(T_ref=216, T_inf=216)
climb.set_flow_ref_vals(qinf=3.16e4)
climb.register_to(f2f_model)

# COMPOSITE FUNCTIONS
# -------------------------------------------------------

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

# DISCIPLINE INTERFACES AND DRIVERS
# -----------------------------------------------------

solvers = SolverManager(comm)
solvers.flow = Fun3dInterface(comm, f2f_model, fun3d_dir="cfd")
solvers.structural = TacsSteadyInterface.create_from_bdf(
    model=f2f_model,
    comm=comm,
    nprocs=20,
    bdf_file=tacs_aim.dat_file_path,
    prefix=tacs_aim.analysis_dir,
)

my_transfer_settings = TransferSettings(npts=200)
fun3d_driver = OnewayAeroDriver(
    solvers, f2f_model, transfer_settings=my_transfer_settings
)

# build the shape driver from the file
tacs_driver = OnewayStructDriver.prime_loads(fun3d_driver)

# PYOPTSPARSE OPTMIZATION
# -------------------------------------------------------------

# create an OptimizationManager object for the pyoptsparse optimization problem
design_out_file = os.path.join(base_dir, "design", "sizing.txt")

# reload previous design
f2f_model.read_design_variables_file(comm, design_out_file)

manager = OptimizationManager(
    tacs_driver, design_out_file=design_out_file, hot_start=hot_start
)

# create the pyoptsparse optimization problem
opt_problem = Optimization("hsctOpt", manager.eval_functions)

# add funtofem model variables to pyoptsparse
manager.register_to_problem(opt_problem)

# run an SNOPT optimization
snoptimizer = SNOPT(options={"IPRINT": 1})

design_folder = os.path.join(base_dir, "design")
if not os.path.exists(design_folder):
    os.mkdir(design_folder)
history_file = os.path.join(design_folder, "sizing.hst")
store_history_file = history_file if store_history else None
hot_start_file = history_file if hot_start else None


sol = snoptimizer(
    opt_problem,
    sens=manager.eval_gradients,
    storeHistory=store_history_file,
    hotStart=hot_start_file,
)

# print final solution
sol_xdict = sol.xStar
print(f"Final solution = {sol_xdict}", flush=True)
