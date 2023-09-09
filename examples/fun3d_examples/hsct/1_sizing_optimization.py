"""
Run a FUN3D analysis with the Fun3dOnewayDriver
then with those aero loads still in the F2F Body object
determine the optimal panel thicknesses using oneway-coupled
structural optimization in TACS.

NOTE: you need to run _mesh_fun3d.py first and move the .ugrid
FUN3D mesh into the meshes/turbulent folder first.
"""

from pyoptsparse import SNOPT, Optimization
from funtofem import *
from mpi4py import MPI
from tacs import caps2tacs
import os

comm = MPI.COMM_WORLD

base_dir = os.path.dirname(os.path.abspath(__file__))
csm_path = os.path.join(base_dir, "meshes", "hsct.csm")

# make the funtofem and tacs model
f2f_model = FUNtoFEMmodel("hsct_sizing")
tacs_model = caps2tacs.TacsModel.build(csm_file=csm_path, comm=comm)
tacs_model.mesh_aim.set_mesh(  # need a refined-enough mesh for the derivative test to pass
    edge_pt_min=20,
    edge_pt_max=30,
    global_mesh_size=0.04,
    max_surf_offset=0.01,
    max_dihedral_angle=5,
).register_to(
    tacs_model
)
f2f_model.structural = tacs_model

# build a body which we will register variables to
wing = Body.aerothermoelastic("wing", boundary=4)

# setup the material and shell properties
aluminum = caps2tacs.Isotropic.aluminum().register_to(tacs_model)

tacs_aim = tacs_model.tacs_aim
tacs_aim.set_config_parameter("mode:flow", 0)
tacs_aim.set_config_parameter("mode:struct", 1)
nribs = int(tacs_model.get_config_parameter("nribs"))
nspars = int(tacs_model.get_config_parameter("nspars"))
nOML = int(tacs_aim.get_output_parameter("nOML"))

for irib in range(1, nribs + 1):
    name = f"rib{irib}"
    prop = caps2tacs.ShellProperty(
        caps_group=name, material=aluminum, membrane_thickness=0.04
    ).register_to(tacs_model)
    Variable.structural(name, value=0.1).set_bounds(
        lower=0.001, upper=0.04, scale=100.0
    ).register_to(wing)

for ispar in range(1, nspars + 1):
    name = f"spar{ispar}"
    prop = caps2tacs.ShellProperty(
        caps_group=name, material=aluminum, membrane_thickness=0.04
    ).register_to(tacs_model)
    Variable.structural(name, value=0.01).set_bounds(
        lower=0.001, upper=0.04, scale=100.0
    ).register_to(wing)

for iOML in range(1, nOML + 1):
    name = f"OML{iOML}"
    prop = caps2tacs.ShellProperty(
        caps_group=name, material=aluminum, membrane_thickness=0.04
    ).register_to(tacs_model)
    Variable.structural(name, value=0.01).set_bounds(
        lower=0.001, upper=0.04, scale=100.0
    ).register_to(wing)

for prefix in ["LE", "TE"]:
    name = f"{prefix}spar"
    prop = caps2tacs.ShellProperty(
        caps_group=name, material=aluminum, membrane_thickness=0.04
    ).register_to(tacs_model)
    Variable.structural(name, value=0.01).set_bounds(
        lower=0.001, upper=0.04, scale=100.0
    ).register_to(wing)

# register the wing body to the model
wing.register_to(f2f_model)

# add remaining information to tacs model
caps2tacs.PinConstraint("root").register_to(tacs_model)
caps2tacs.TemperatureConstraint("midplane").register_to(tacs_model)

# setup the tacs model
tacs_aim.setup_aim()
tacs_aim.pre_analysis()

# make a funtofem scenario
cruise = Scenario.steady("cruise", steps=10)  # 2000
mass = Function.mass().optimize(scale=1.0e-4, objective=True, plot=True)
ksfailure = Function.ksfailure().optimize(
    scale=30.0, upper=0.267, objective=False, plot=True
)
cruise.include(mass).include(ksfailure)
cruise.set_temperature(T_ref=216, T_inf=216)

# cruise_aoa = cruise.get_variable("AOA").set_bounds(value=2.0)
cruise.adjoint_steps = 2000
cruise.register_to(f2f_model)

# make the composite functions for adjacency constraints
variables = f2f_model.get_variables()
adj_ratio = 4.0
adj_scale = 10.0
for irib in range(
    1, nribs
):  # not (1, nribs+1) bc we want to do one less since we're doing nribs-1 pairs
    left_rib = f2f_model.get_variables(names=f"rib{irib}")
    right_rib = f2f_model.get_variables(names=f"rib{irib+1}")
    # make a composite function for relative diff in rib thicknesses
    adj_rib_constr = (left_rib - right_rib) / left_rib
    adj_rib_constr.set_name(f"rib{irib}-{irib+1}").optimize(
        lower=-adj_ratio, upper=adj_ratio, scale=1.0, objective=False
    ).register_to(f2f_model)

for ispar in range(1, nspars):
    left_spar = f2f_model.get_variables(names=f"spar{ispar}")
    right_spar = f2f_model.get_variables(names=f"spar{ispar+1}")
    # make a composite function for relative diff in spar thicknesses
    adj_spar_constr = (left_spar - right_spar) / left_spar
    adj_spar_constr.set_name(f"spar{ispar}-{ispar+1}").optimize(
        lower=-adj_ratio, upper=adj_ratio, scale=1.0, objective=False
    ).register_to(f2f_model)

for iOML in range(1, nOML):
    left_OML = f2f_model.get_variables(names=f"OML{iOML}")
    right_OML = f2f_model.get_variables(names=f"OML{iOML+1}")
    # make a composite function for relative diff in OML thicknesses
    adj_OML_constr = (left_OML - right_OML) / left_OML
    adj_OML_constr.set_name(f"OML{iOML}-{iOML+1}").optimize(
        lower=-adj_ratio, upper=adj_ratio, scale=1.0, objective=False
    ).register_to(f2f_model)


solvers = SolverManager(comm)
solvers.flow = Fun3dInterface(comm, f2f_model, fun3d_dir="meshes").set_units(
    qinf=3.16e4
)
solvers.structural = TacsSteadyInterface.create_from_bdf(
    model=f2f_model,
    comm=comm,
    nprocs=48,
    bdf_file=tacs_aim.dat_file_path,
    prefix=tacs_aim.analysis_dir,
)

my_transfer_settings = TransferSettings(npts=200)
fun3d_driver = Fun3dOnewayDriver(
    solvers, f2f_model, transfer_settings=my_transfer_settings
)

# build the shape driver from the file
tacs_driver = TacsOnewayDriver.prime_loads(fun3d_driver)

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

history_file = f"hsct.hst"
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
