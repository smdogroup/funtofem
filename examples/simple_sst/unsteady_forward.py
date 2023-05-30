import os, time
import numpy as np
from mpi4py import MPI
from tacs import constitutive, elements

# import other funtofem
from funtofem import *

# number of tacs processors and setup MPI
comm = MPI.COMM_WORLD

# Build the model
model = FUNtoFEMmodel("simpleSST")

wing = Body.aeroelastic("simpleSST", boundary=2)
Variable.structural("thick").set_bounds(lower=0.001, value=3, upper=100).rescale(
    0.001
).register_to(wing)
wing.register_to(model)

# make a new steady scenario that evaluates ksfailure and mass in funtofem
laminar = Scenario.unsteady("laminar", 800)
laminar.include(Function.ksfailure(ks_weight=50.0, start=300, stop=350))
laminar.include(Function.mass(start=300, stop=350))
laminar.include(Function.lift(start=300, stop=350))
laminar.include(Function.drag(start=300, stop=350))
laminar.register_to(model)

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
    dt=0.05,
    num_steps=800,
)

# initialize the funtofem solvers
solvers = SolverManager(comm)
solvers.flow = Fun3dInterface(
    comm=comm,
    model=model,
    forward_options={"timedep_adj_frozen": True},
    adjoint_options={"timedep_adj_frozen": True},
)
solvers.flow.set_units(qinf=1.0e4, flow_dt=0.001)
tacs_folder = os.path.join(os.getcwd(), "tacs_output")
if not os.path.exists(tacs_folder) and comm.rank == 0:
    os.mkdir(tacs_folder)
solvers.structural = TacsUnsteadyInterface.create_from_bdf(
    comm=comm,
    model=model,
    nprocs=8,
    bdf_file="nastran_CAPS.dat",
    integration_settings=integration_settings,
    output_dir=tacs_folder,
)

# build the coupled driver
driver = FUNtoFEMnlbgs(solvers=solvers, model=model)

# solve the forward analysis
driver.solve_forward()
functions = model.get_functions()
variables = model.get_variables()
# end of running funtofem

# report function values from funtofem unsteady analysis
print("Finished running funtofem unsteady analysis...")
for ifunc, func in enumerate(functions):
    print(f"\tFunction {func.name} = {func.value.real}")

print("Running funtofem unsteady adjoint", flush=True)
driver.solve_adjoint()
gradients = model.get_function_gradients()
