import numpy as np
from mpi4py import MPI
from pyfuntofem.funtofem_model import FUNtoFEMmodel
from pyfuntofem.variable import Variable
from pyfuntofem.scenario import Scenario
from pyfuntofem.body import Body
from pyfuntofem.function import Function
from pyfuntofem.test_solver import TestAerodynamicSolver, TestStructuralSolver
from pyfuntofem.funtofem_nlbgs_driver import FUNtoFEMnlbgs

# Build the model
model = FUNtoFEMmodel("model")
plate = Body("plate", "aerothermal", group=0, boundary=1)

# Create a structural variable
for i in range(5):
    thickness = np.random.rand()
    svar = Variable("thickness %d" % (i), value=thickness, lower=0.01, upper=0.1)
    plate.add_variable("structural", svar)

model.add_body(plate)

# Create a scenario to run
steady = Scenario("steady", group=0, steps=100)

# Add the aerodynamic variables to the scenario
for i in range(4):
    value = np.random.rand()
    avar = Variable("aero var %d" % (i), value=value, lower=-10.0, upper=10.0)
    steady.add_variable("aerodynamic", avar)

# Add a function to the scenario
temp = Function("temperature", analysis_type="structural")
steady.add_function(temp)

# Add the steady-state scenario
model.add_scenario(steady)

# Instantiate a test solver for the flow and structures
comm = MPI.COMM_WORLD
solvers = {}
solvers["flow"] = TestAerodynamicSolver(comm, model)
solvers["structural"] = TestStructuralSolver(comm, model)

# L&D transfer options
transfer_options = {
    "analysis_type": "aerothermal",
    "scheme": "meld",
    "thermal_scheme": "meld",
    "npts": 5,
}

# instantiate the driver
driver = FUNtoFEMnlbgs(solvers, comm, comm, 0, comm, 0, transfer_options, model=model)

# Manual test of the disciplinary solvers
scenario = model.scenarios[0]
bodies = model.bodies
solvers["flow"].test_iterate_adjoint(scenario, bodies)
solvers["structural"].test_iterate_adjoint(scenario, bodies)

# Solve the forward analysis
driver.solve_forward()
driver.solve_adjoint()

# Get the functions
functions = model.get_functions()
variables = model.get_variables()

driver.solve_adjoint()
grads = model.get_function_gradients()

# Set the new variable values
dh = 1e-30
variables[0].value = variables[0].value + 1j * dh
model.set_variables(variables)

driver.solve_forward()
deriv = functions[0].value.imag / dh

print("complex step = ", deriv)
print("adjoint      = ", grads[0][0])
