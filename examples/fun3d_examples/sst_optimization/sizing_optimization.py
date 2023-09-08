from pyoptsparse import SNOPT, Optimization
from pyfuntofem import *
from mpi4py import MPI
from tacs import caps2tacs
import os

comm = MPI.COMM_WORLD

base_dir = os.path.dirname(os.path.abspath(__file__))
csm_path = os.path.join(base_dir, "input_files", "supersonic_transport_wing_v4.7.csm")
datfile_path = os.path.join(base_dir, "meshes", "nastran_CAPS.dat")

# make the funtofem and tacs model
f2f_model = FUNtoFEMmodel("wing")
tacs_model = caps2tacs.TacsModel.build(csm_file=csm_path, comm=comm)
tacs_model.egads_aim.set_mesh(  # need a refined-enough mesh for the derivative test to pass
    edge_pt_min=20,
    edge_pt_max=30,
    global_mesh_size=0.04,
    max_surf_offset=0.01,
    max_dihedral_angle=5,
).register_to(
    tacs_model
)
f2f_model.tacs_model = tacs_model

# build a body which we will register variables to
wing = Body.aeroelastic("wing")

# setup the material and shell properties
aluminum = caps2tacs.Isotropic.aluminum().register_to(tacs_model)

nribs = int(tacs_model.get_config_parameter("nribs"))
nspars = int(tacs_model.get_config_parameter("nspars"))
nOML = nribs - 1

for irib in range(1, nribs + 1):
    prop = caps2tacs.ShellProperty(
        caps_group=f"rib{irib}", material=aluminum, membrane_thickness=0.01
    ).register_to(tacs_model)
    Variable.from_caps(prop).set_bounds(
        lower=0.001, upper=0.04, scale=100.0
    ).register_to(wing)

for ispar in range(1, nspars + 1):
    prop = caps2tacs.ShellProperty(
        caps_group=f"spar{ispar}", material=aluminum, membrane_thickness=0.01
    ).register_to(tacs_model)
    Variable.from_caps(prop).set_bounds(
        lower=0.001, upper=0.04, scale=100.0
    ).register_to(wing)

for iOML in range(1, nOML + 1):
    prop = caps2tacs.ShellProperty(
        caps_group=f"OML{iOML}", material=aluminum, membrane_thickness=0.01
    ).register_to(tacs_model)
    Variable.from_caps(prop).set_bounds(
        lower=0.001, upper=0.04, scale=100.0
    ).register_to(wing)

for prefix in ["LE", "TE"]:
    prop = caps2tacs.ShellProperty(
        caps_group=f"{prefix}spar", material=aluminum, membrane_thickness=0.01
    ).register_to(tacs_model)
    Variable.from_caps(prop).set_bounds(
        lower=0.001, upper=0.04, scale=100.0
    ).register_to(wing)

# register the wing body to the model
wing.register_to(f2f_model)

# add remaining information to tacs model
caps2tacs.PinConstraint("root").register_to(tacs_model)

# setup the tacs model
tacs_aim = tacs_model.tacs_aim
tacs_aim.setup_aim()
tacs_aim.pre_analysis()

# make a funtofem scenario
test_scenario = (
    Scenario.steady("laminar", steps=300)
    .include(Function.mass())
    .include(Function.ksfailure())
)
test_scenario.register_to(f2f_model)

solvers = SolverManager(comm)
solvers.flow = Fun3dInterface(comm, f2f_model, fun3d_dir="meshes").set_units(qinf=1.0e4)
solvers.structural = TacsSteadyInterface.create_from_bdf(
    model=f2f_model, comm=comm, nprocs=48, bdf_file=tacs_aim.dat_file_path
)

my_transfer_settings = TransferSettings(npts=200)
coupled_driver = FUNtoFEMnlbgs(
    solvers, transfer_settings=my_transfer_settings, model=f2f_model
)

# build the shape driver from the file
tacs_driver = TacsOnewayDriver.prime_loads(coupled_driver)

# create an OptimizationManager object for the pyoptsparse optimization problem
manager = tacs_driver.manager

# create the pyoptsparse optimization problem
opt_problem = Optimization("SSTwingOpt", manager.eval_functions)

# add funtofem model variables to pyoptsparse
manager.add_sparse_variables(opt_problem)

# add objective and constraint
opt_problem.addObj("mass", scale=1.0e-4)
opt_problem.addCon("ksfailure", upper=0.267)

# run an SNOPT optimization
snoptimizer = SNOPT(options={"IPRINT": 1})

can_hot_start = False
can_store_history = True

history_file = f"sst_sizing.hst"
store_history = history_file if can_store_history else None
hot_start_file = history_file if can_hot_start else None

sol = snoptimizer(
    opt_problem,
    sens=manager.eval_gradients,
    storeHistory=store_history,
    hotStart=hot_start_file,
)

# print final solution
sol_xdict = sol.xStar
print(f"Final solution = {sol_xdict}", flush=True)
