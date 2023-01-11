import os, time
import numpy as np
from mpi4py import MPI
from tacs import constitutive, elements

# import other pyfuntofem
from pyfuntofem.model import *
from pyfuntofem.driver import *
from pyfuntofem.interface import (
    Fun3dInterface,
    createTacsUnsteadyInterfaceFromBDF,
    IntegrationSettings,
)

# run settings
num_steps = 25
n_tacs_procs = 2
f2f_analysis_type = "aeroelastic"
flow_type = "laminar"

# number of tacs processors and setup MPI
comm = MPI.COMM_WORLD

# Build the model
model = FUNtoFEMmodel("simple_diamond")

wing_body = Body("simple_diamond", analysis_type=f2f_analysis_type, group=0, boundary=2)
wing_body.add_variable(
    "structural", Variable("thick", value=0.03, lower=0.0001, upper=1.0)
)
model.add_body(wing_body)

# make a new steady scenario that evaluates ksfailure and mass in funtofem
my_scenario = Scenario("fun3d", group=0, steady=False, steps=num_steps)

# functions
start = 0
stop = 9
ks_failure = Function(
    "ksfailure",
    start=start,
    stop=stop,
    analysis_type="structural",
    options={"ksweight": 1000.0},
)
mass = Function("mass", start=start, stop=stop, analysis_type="structural")
lift = Function("cl", start=start, stop=stop, analysis_type="aerodynamic")
drag = Function("cd", start=start, stop=stop, analysis_type="aerodynamic")

for function in [ks_failure, mass, lift, drag]:
    my_scenario.add_function(function)

model.add_scenario(my_scenario)

# select the integration settings for tacs
integration_settings = IntegrationSettings(
    integration_type="BDF",
    integration_order=2,
    L2_convergence=1e-12,
    L2_convergence_rel=1e-12,
    jac_assembly_freq=1,
    write_solution=True,
    number_solution_files=True,
    print_timing_info=False,
    print_level=2,
    start_time=0.0,
    dt=0.1,
    num_steps=num_steps,
)

# setup the tacs comm again for the driver
world_rank = comm.Get_rank()
if world_rank < n_tacs_procs:
    color = 55
    key = world_rank
else:
    color = MPI.UNDEFINED
    key = world_rank
tacs_comm = comm.Split(color, key)

# initialize the funtofem solvers
solvers = {}

# create the fun3d interface solver
solvers["flow"] = Fun3dInterface(
    comm=comm,
    model=model,
    flow_dt=0.01,
    qinf=1.0e4,
    thermal_scale=1.0e6,
    forward_options={"timedep_adj_frozen": True},
    adjoint_options={"timedep_adj_frozen": True},
)

cwd = os.getcwd()
tacs_folder = os.path.join(cwd, "tacs_output")

if not (os.path.exists(tacs_folder)) and comm.rank == 0:
    os.mkdir(tacs_folder)

# create tacs unsteady interface from the BDF / DAT file
solvers["structural"] = createTacsUnsteadyInterfaceFromBDF(
    model=model,
    comm=comm,
    nprocs=n_tacs_procs,
    bdf_file="nastran_CAPS.dat",
    integration_settings=integration_settings,
    output_dir=tacs_folder,
    callback=None,
    struct_options={},
)

# L&D transfer options
transfer_options = {
    "analysis_type": f2f_analysis_type,
    "scheme": "meld",
    "thermal_scheme": "meld",
}

# instantiate the driver
driver = FUNtoFEMnlbgs(
    solvers=solvers,
    comm=comm,
    struct_comm=tacs_comm,
    struct_root=0,
    aero_comm=comm,
    aero_root=0,
    transfer_options=transfer_options,
    model=model,
)
# solve the forward analysis
driver.solve_forward()
functions = model.get_functions()
variables = model.get_variables()
# end of running funtofem

# report function values from funtofem unsteady analysis
print("Finished running funtofem unsteady analysis...")
for ifunc, func in enumerate(functions):
    print(f"\tFunction {func.name} = {func.value.real}")

print("Running funtofem unsteady adjoint")
driver.solve_adjoint()
gradients = model.get_function_gradients()
