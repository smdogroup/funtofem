from mpi4py import MPI
from pyfuntofem.funtofem_model import FUNtoFEMmodel
from pyfuntofem.variable import Variable
from pyfuntofem.scenario import Scenario
from pyfuntofem.body import Body
from pyfuntofem.function import Function
from pyfuntofem.test_solver import TestSolver
from pyfuntofem.funtofem_nlbgs_driver import FUNtoFEMnlbgs

# Build the model
model = FUNtoFEMmodel("model")
plate = Body("plate", "aeroelastic", group=0, boundary=1)

# Create a structural variable
thickness = 0.01
svar = Variable("thickness", value=thickness, lower=0.01, upper=0.1)
plate.add_variable("structural", svar)
model.add_body(plate)

# Create a scenario to run - this is an unsteady case
unsteady = Scenario("unsteady", group=0, steps=100, steady=False)

# Add a function to the scenario
temp = Function("temperature", analysis_type="structural")
unsteady.add_function(temp)

# Add the steady-state scenario
model.add_scenario(unsteady)

# Instantiate a test solver for the flow and structures
comm = MPI.COMM_WORLD
solvers = {}
solvers["flow"] = TestSolver(comm, model, solver="aerodynamic")
solvers["structural"] = TestSolver(comm, model, solver="structural")

# L&D transfer options
transfer_options = {
    "analysis_type": "aeroelastic",
    "scheme": "meld",
    "thermal_scheme": "meld",
    "npts": 5,
}

# instantiate the driver
driver = FUNtoFEMnlbgs(solvers, comm, comm, 0, comm, 0, transfer_options, model=model)

# Solve the forward analysis
driver.solve_forward()
