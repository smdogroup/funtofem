import numpy as np
from mpi4py import MPI
from funtofem import TransferScheme
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
    "analysis_type": "aerothermoelastic",
    "scheme": "meld",
    "thermal_scheme": "meld",
    "npts": 5,
}

# instantiate the driver
driver = FUNtoFEMnlbgs(solvers, comm, comm, 0, comm, 0, transfer_options, model=model)

# Check whether to use the complex-step method or now
complex_step = False
epsilon = 1e-6
if TransferScheme.dtype == complex:
    complex_step = True
    epsilon = 1e-30

# Manual test of the disciplinary solvers
scenario = model.scenarios[0]
bodies = model.bodies
solvers["flow"].test_adjoint(
    "flow", scenario, bodies, epsilon=epsilon, complex_step=complex_step
)
solvers["structural"].test_adjoint(
    "structural", scenario, bodies, epsilon=epsilon, complex_step=complex_step
)

# Solve the forward analysis
driver.solve_forward()
driver.solve_adjoint()

# Get the functions
functions = model.get_functions()
variables = model.get_variables()

# Store the function values
fvals_init = []
for func in functions:
    fvals_init.append(func.value)

# Solve the adjoint and get the function gradients
driver.solve_adjoint()
grads = model.get_function_gradients()

# Set the new variable values
if complex_step:
    variables[0].value = variables[0].value + 1j * epsilon
    model.set_variables(variables)
else:
    variables[0].value = variables[0].value + epsilon
    model.set_variables(variables)

driver.solve_forward()

# Store the function values
fvals = []
for func in functions:
    fvals.append(func.value)

if complex_step:
    deriv = fvals[0].imag / epsilon

    rel_error = (deriv - grads[0][0]) / deriv
    if comm.rank == 0:
        print("Approximate gradient  = ", deriv.real)
        print("Adjoint gradient      = ", grads[0][0].real)
        print("Relative error        = ", rel_error.real)
else:
    deriv = (fvals[0] - fvals_init[0]) / epsilon

    rel_error = (deriv - grads[0][0]) / deriv
    if comm.rank == 0:
        print("Approximate gradient  = ", deriv)
        print("Adjoint gradient      = ", grads[0][0])
        print("Relative error        = ", rel_error)
