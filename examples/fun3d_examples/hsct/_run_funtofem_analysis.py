import os
from mpi4py import MPI
from funtofem import *

comm = MPI.COMM_WORLD
base_dir = os.path.dirname(os.path.abspath(__file__))
fun3d_dir = os.path.join(base_dir, "meshes")
tacs_dir = os.path.join(fun3d_dir, "tacs")
if not os.path.exists(tacs_dir):
    os.mkdir(tacs_dir)
fun3d_remote = Fun3dRemote.paths(fun3d_dir)

# F2F MODEL. NO SHAPE MODELS IN ANALYSIS SCRIPT
# ------------------------------------------------------
hsct_model = FUNtoFEMmodel("hsct_MDO")

# BODIES AND STRUCT DVS. NO SHAPE DVS IN ANALYSIS SCRIPT
# ------------------------------------------------------

wing = Body.aerothermoelastic("hsct_wing", boundary=4)

# structural variables in the wing
# this part needs to be the same as in the CSM file
nribs = 25
nspars = 3
nOML = nribs - 1
for irib in range(1, nribs + 1):
    Variable.structural(name=f"rib{irib}").register_to(wing)
for ispar in range(1, nspars + 1):
    Variable.structural(name=f"spar{ispar}").register_to(wing)
for iOML in range(1, nOML + 1):
    Variable.structural(name=f"OML{iOML}").register_to(wing)
for prefix in ["LE", "TE"]:
    Variable.structural(name=f"{prefix}spar").register_to(wing)

wing.register_to(hsct_model)

# SCENARIOS AND AERO DVS. NO SHAPE DVS IN ANALYSIS SCRIPT
# ------------------------------------------------------------

climb = Scenario.steady("climb", steps=5000)
climb.set_temperature(T_ref=300.0, T_inf=300.0)  # modify this
climb.set_flow_units(qinf=1e4, flow_dt=1.0)
ksfailure_climb = Function.ksfailure().register_to(climb)
cl_climb = Function.lift().register_to(climb)
cd_climb = Function.drag().register_to(climb)
aoa_climb = climb.get_variable("AOA")
mach_climb = climb.get_variable("Mach")
climb.register_to(hsct_model)

cruise = Scenario.steady("cruise", steps=5000)
cruise.set_temperature(T_ref=216, T_inf=216)
cruise.set_flow_units(qinf=3.16e4, flow_dt=1.0)
ksfailure_cruise = Function.ksfailure().register_to(cruise)
cl_cruise = Function.lift().register_to(cruise)
cd_cruise = Function.drag().register_to(cruise)
moment = Function.moment().register_to(cruise)
mass = Function.mass().register_to(cruise)
aoa_cruise = cruise.get_variable("AOA")
mach_cruise = cruise.get_variable("Mach")
cruise.register_to(hsct_model)

# NO COMPOSITE FUNCTIONS IN ANALYSIS SCRIPT
# ----------------------------------------------------------------

# NOTE : don't need to include composite functions in analysis script
# as long as main analysis functions evaluated in analysis script
# the composite function derivatives in the driver script will be computed

# BUILD DISCIPLINE INTERFACES AND DRIVER
# -----------------------------------------------------

# build the solvers and coupled driver
solvers = SolverManager(comm)
solvers.flow = Fun3dInterface(comm, hsct_model, fun3d_dir="meshes")
solvers.structural = TacsSteadyInterface.create_from_bdf(
    model=hsct_model,
    comm=comm,
    nprocs=48,
    bdf_file=fun3d_remote.dat_file,
    prefix=fun3d_dir,
)
f2f_driver = FuntofemShapeDriver.analysis(solvers, hsct_model)

# RUN THE FORWARD + ADJOINT ANALYSIS
# ------------------------------------------------

# NOTE: this writes the design and sens file outputs
# to the driver script
f2f_driver.solve_forward()
f2f_driver.solve_adjoint()
