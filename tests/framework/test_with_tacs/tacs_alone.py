
from __future__ import print_function

from pyfuntofem.funtofem_model import FUNtoFEMmodel
from pyfuntofem.variable import Variable
from pyfuntofem.scenario import Scenario
from pyfuntofem.body import Body
from pyfuntofem.function import Function
from pyfuntofem.funtofem_nlbgs_driver import FUNtoFEMnlbgs
from pyfuntofem.test_solver import TestSolver

from tacs_model import wedgeTACS
from mpi4py import MPI
import os, sys
import time

# Split the communicator
n_tacs_procs = 1
comm = MPI.COMM_WORLD

world_rank = comm.Get_rank()
if world_rank < n_tacs_procs:
    color = 55
    key = world_rank
else:
    color = MPI.UNDEFINED
    key = world_rank
tacs_comm = comm.Split(color, key)

# Build the model
model = FUNtoFEMmodel("wedge")
plate = Body("plate", "aerothermal", group=0, boundary=1)

# Create a structural variable
thickness = 0.01
svar = Variable("thickness", value=thickness, lower=0.01, upper=0.1)
plate.add_variable("structural", svar)
model.add_body(plate)

# Create a scenario to run
steps = 20
steady = Scenario("steady", group=0, steps=steps)

# Add a function to the scenario
temp = Function("temperature", analysis_type="structural")
steady.add_function(temp)

model.add_scenario(steady)

# Instantiate a test solver for the flow and structures
solvers = {}
solvers["flow"] = TestSolver(comm, model, solver="flow")
solvers["structural"] = wedgeTACS(comm, tacs_comm, model, n_tacs_procs)

# L&D transfer options
transfer_options = {
    "analysis_type": "aeroelastic",
    "scheme": "meld",
    "thermal_scheme": "meld",
    "npts": 5,
}

# instantiate the driver
driver = FUNtoFEMnlbgs(solvers, comm, tacs_comm, 0, comm, 0, transfer_options, model=model)

fail = driver.solve_forward()
if fail == 1:
    print("\nSimulation failed.\n")
