"""
Run a FUN3D analysis with the Fun3dOnewayDriver
then with those aero loads still in the F2F Body object
determine the optimal panel thicknesses using oneway-coupled
structural optimization in TACS.

NOTE: you need to run _mesh_fun3d.py first and move the .ugrid
FUN3D mesh into the meshes/turbulent folder first.
"""

from pyoptsparse import SNOPT, Optimization

# import openmdao.api as om
from funtofem import *
from mpi4py import MPI
from tacs import caps2tacs
import os

comm = MPI.COMM_WORLD

base_dir = os.path.dirname(os.path.abspath(__file__))
csm_path = os.path.join(base_dir, "meshes", "hsct.csm")

# F2F MODEL and SHAPE MODELS
# ----------------------------------------

f2f_model = FUNtoFEMmodel("hsct_sizing")
tacs_model = caps2tacs.TacsModel.build(csm_file=csm_path, comm=comm)
tacs_model.mesh_aim.set_mesh(  # need a refined-enough mesh for the derivative test to pass
    edge_pt_min=20,
    edge_pt_max=30,
    global_mesh_size=0.01,
    max_surf_offset=0.01,
    max_dihedral_angle=5,
).register_to(
    tacs_model
)
f2f_model.structural = tacs_model

tacs_aim = tacs_model.tacs_aim
tacs_aim.set_config_parameter("mode:flow", 0)
tacs_aim.set_config_parameter("mode:struct", 1)

# mesh edge settings
interior_ct = 16
exterior_ct = 2 * interior_ct - 1  # +1 for small#, -1 for large #
nribs = int(tacs_aim.get_config_parameter("nribs"))
if comm.rank == 0:
    egads_aim = tacs_model.mesh_aim
    egads_aim.aim.input.Mesh_Sizing = {
        "rib1interior": {"numEdgePoints": interior_ct},
        "rib1exterior": {"numEdgePoints": exterior_ct},
        f"rib{nribs}interior": {"numEdgePoints": interior_ct},
        f"rib{nribs}exterior": {"numEdgePoints": exterior_ct},
    }

# add tacs constraints in
caps2tacs.PinConstraint("root").register_to(tacs_model)
caps2tacs.TemperatureConstraint("midplane").register_to(tacs_model)

# BODIES AND STRUCT DVs
# -------------------------------------------------

wing = Body.aerothermoelastic("wing", boundary=4)

# setup the material and shell properties
aluminum = caps2tacs.Isotropic.aluminum().register_to(tacs_model)

nribs = int(tacs_model.get_config_parameter("nribs"))
nspars = int(tacs_model.get_config_parameter("nspars"))
nOML = int(tacs_aim.get_output_parameter("nOML"))

for irib in range(1, nribs + 1):
    name = f"rib{irib}"
    prop = caps2tacs.ShellProperty(
        caps_group=name, material=aluminum, membrane_thickness=0.04
    ).register_to(tacs_model)
    Variable.structural(name, value=0.01).set_bounds(
        lower=0.001, upper=1.0, scale=100.0
    ).register_to(wing)

for ispar in range(1, nspars + 1):
    name = f"spar{ispar}"
    prop = caps2tacs.ShellProperty(
        caps_group=name, material=aluminum, membrane_thickness=0.04
    ).register_to(tacs_model)
    Variable.structural(name, value=0.01).set_bounds(
        lower=0.001, upper=1.0, scale=100.0
    ).register_to(wing)

for iOML in range(1, nOML + 1):
    name = f"OML{iOML}"
    prop = caps2tacs.ShellProperty(
        caps_group=name, material=aluminum, membrane_thickness=0.04
    ).register_to(tacs_model)
    Variable.structural(name, value=0.01).set_bounds(
        lower=0.001, upper=1.0, scale=100.0
    ).register_to(wing)

for prefix in ["LE", "TE"]:
    name = f"{prefix}spar"
    prop = caps2tacs.ShellProperty(
        caps_group=name, material=aluminum, membrane_thickness=0.04
    ).register_to(tacs_model)
    Variable.structural(name, value=0.01).set_bounds(
        lower=0.001, upper=1.0, scale=100.0
    ).register_to(wing)

# register the wing body to the model
wing.register_to(f2f_model)

# INITIAL STRUCTURE MESH, SINCE NO STRUCT SHAPE VARS
# --------------------------------------------------

tacs_aim.setup_aim()
tacs_aim.pre_analysis()

# SCENARIOS
# ----------------------------------------------------
safety_factor = 1.5
ks_max = 1 / safety_factor

# make a funtofem scenario
cruise = Scenario.steady("cruise", steps=500)  # 2000
mass = Function.mass().optimize(scale=1.0e-4, objective=True, plot=True)
ksfailure = Function.ksfailure(ks_weight=10.0).optimize(
    scale=30.0, upper=ks_max, objective=False, plot=True
)
cruise.include(mass).include(ksfailure)
cruise.set_temperature(T_ref=216, T_inf=216)
cruise.set_flow_units(qinf=3.16e4)
cruise.register_to(f2f_model)

# COMPOSITE FUNCTIONS
# -------------------------------------------------------

# skin thickness adjacency constraints
variables = f2f_model.get_variables()
section_prefix = ["rib", "spar", "OML"]
section_nums = [nribs, nspars, nOML]
for isection, prefix in enumerate(section_prefix):
    section_num = section_nums[isection]
    for iconstr in range(1, section_num):
        left_var = f2f_model.get_variables(names=f"{prefix}{iconstr}")
        right_var = f2f_model.get_variables(names=f"{prefix}{iconstr+1}")
        adj_constr = (left_var - right_var) / left_var
        if prefix in ["rib", "OML"]:
            adj_ratio = 0.5
        else:
            adj_ratio = 4.0
        adj_constr.set_name(f"{prefix}{iconstr}-{iconstr+1}").optimize(
            lower=-adj_ratio, upper=adj_ratio, scale=1.0, objective=False
        ).register_to(f2f_model)

# DISCIPLINE INTERFACES AND DRIVERS
# -----------------------------------------------------

solvers = SolverManager(comm)
solvers.flow = Fun3dInterface(comm, f2f_model, fun3d_dir="meshes")
solvers.structural = TacsSteadyInterface.create_from_bdf(
    model=f2f_model,
    comm=comm,
    nprocs=10,
    bdf_file=tacs_aim.dat_file_path,
    prefix=tacs_aim.analysis_dir,
)

my_transfer_settings = TransferSettings(npts=200)
fun3d_driver = Fun3dOnewayDriver(
    solvers, f2f_model, transfer_settings=my_transfer_settings
)

# build the shape driver from the file
tacs_driver = TacsOnewayDriver.prime_loads(fun3d_driver)

# PYOPTSPARSE OPTMIZATION
# -------------------------------------------------------------
hot_start = False
store_history = True

# create an OptimizationManager object for the pyoptsparse optimization problem
design_out_file = os.path.join(base_dir, "meshes", "sizing_design.txt")
manager = OptimizationManager(
    tacs_driver, design_out_file=design_out_file, hot_start=hot_start
)

# create the pyoptsparse optimization problem
opt_problem = Optimization("hsctOpt", manager.eval_functions)

# add funtofem model variables to pyoptsparse
manager.register_to_problem(opt_problem)

# run an SNOPT optimization
snoptimizer = SNOPT(options={"IPRINT": 1})

history_file = f"hsctSizing.hst"
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


# # OPENMDAO OPTIMIZATION
# # -------------------------------------------------------------

# design_out_file = os.path.join(base_dir, "meshes", "sizing_design.txt")

# # setup the OpenMDAO Problem object
# prob = om.Problem()

# # Create the OpenMDAO component using the built-in Funtofem component
# f2f_subsystem = FuntofemComponent(
#     driver=tacs_driver, write_dir=tacs_aim.analysis_dir, design_out_file=design_out_file
# )
# prob.model.add_subsystem("f2fSystem", f2f_subsystem)
# f2f_subsystem.register_to_model(prob.model, "f2fSystem")

# # setup the optimizer settings # COBYLA for auto-FDing
# optimizer = "pyoptsparse"
# if optimizer == "scipy":
#     prob.driver = om.ScipyOptimizeDriver(optimizer="SLSQP", tol=1.0e-9, disp=True)
# elif optimizer == "pyoptsparse":
#     prob.driver = om.pyOptSparseDriver(optimizer="SNOPT")
#     # prob.driver.opt_settings['Major feasibility tolerance'] = 1e-5 # lower tolerance to prevent tight oscillations

# # Start the optimization
# print("\n==> Starting optimization...")
# prob.setup()

# debug = False
# if debug:
#     print("Checking partials...", flush=True)
#     prob.check_partials(compact_print=True)

# else:
#     prob.run_driver()

#     # report the final optimal design
#     design_hdl = f2f_subsystem._design_hdl
#     for var in f2f_model.get_variables():
#         opt_value = prob.get_val(f"f2fSystem.{var.name}")
#         design_hdl.write(f"\t{var.name} = {opt_value}")

#     prob.cleanup()
#     # close the design hdl file
#     f2f_subsystem.cleanup()
