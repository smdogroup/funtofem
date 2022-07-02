from mpi4py import MPI
from pyfuntofem.funtofem_model import FUNtoFEMmodel
from pyfuntofem.variable import Variable
from pyfuntofem.scenario import Scenario
from pyfuntofem.body import Body
from pyfuntofem.function import Function
from pyfuntofem.test_solver import TestSolver
from pyfuntofem.funtofem_nlbgs_driver import FUNtoFEMnlbgs

# Build the model
model = FUNtoFEMmodel("wedge")
plate = Body("plate", "aerothermal", group=0, boundary=1)

# Create a structural variable
thickness = 0.01
svar = Variable("thickness", value=thickness, lower=0.01, upper=0.1)
plate.add_variable("structural", svar)
model.add_body(plate)

# Create a scenario to run
steady = Scenario("steady", group=0, steps=100)

# Add a function to the scenario
temp = Function("temperature", analysis_type="structural")
steady.add_function(temp)

temp = Function("temperature", analysis_type="structural")
steady.add_function(temp)

model.add_scenario(steady)

# Instantiate a test solver for the flow and structures
comm = MPI.COMM_WORLD
solvers = {}
solvers["flow"] = TestSolver(comm, model, solver="flow")
solvers["structural"] = TestSolver(comm, model, solver="structure")

# L&D transfer options
transfer_options = {
    "analysis_type": "aeroelastic",
    "scheme": "meld",
    "thermal_scheme": "meld",
    "npts": 5,
}

# instantiate the driver
driver = FUNtoFEMnlbgs(solvers, comm, comm, 0, comm, 0, transfer_options, model=model)

driver.solve_forward()
