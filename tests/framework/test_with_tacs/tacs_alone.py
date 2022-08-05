import os
from mpi4py import MPI
from funtofem import TransferScheme
from pyfuntofem.funtofem_model import FUNtoFEMmodel
from pyfuntofem.variable import Variable
from pyfuntofem.scenario import Scenario
from pyfuntofem.body import Body
from pyfuntofem.function import Function
from pyfuntofem.test_solver import TestAerodynamicSolver
from pyfuntofem.funtofem_nlbgs_driver import FUNtoFEMnlbgs
from pyfuntofem.tacs_interface import createTacsInterfaceFromBDF
from test_bdf_utils import generateBDF, thermoelasticity_callback

# Generate the BDF file if required
bdf_file = "test_bdf_file.bdf"
# if not os.path.exists(bdf_file):
generateBDF(bdf_file)

# Build the model
model = FUNtoFEMmodel("wedge")
plate = Body("plate", "aerothermoelastic", group=0, boundary=1)

# Create a structural variable
thickness = 1.0
svar = Variable("thickness", value=thickness, lower=0.01, upper=0.1)
plate.add_variable("structural", svar)
model.add_body(plate)

# Create a scenario to run
steps = 150
steady = Scenario("steady", group=0, steps=steps)

# Add a function to the scenario
temp = Function("ksfailure", analysis_type="structural")
steady.add_function(temp)

model.add_scenario(steady)

# Instantiate the solvers we'll use here
solvers = {}

# Build the TACS interface
nprocs = 1
comm = MPI.COMM_WORLD

solvers["structural"] = createTacsInterfaceFromBDF(
    model, comm, nprocs, bdf_file, callback=thermoelasticity_callback
)
solvers["flow"] = TestAerodynamicSolver(comm, model)

tacs_comm = solvers["structural"].tacs_comm

# L&D transfer options
transfer_options = {
    "analysis_type": "aerothermoelastic",
    "scheme": "meld",
    "thermal_scheme": "meld",
    "npts": 5,
}

# instantiate the driver
driver = FUNtoFEMnlbgs(
    solvers, comm, tacs_comm, 0, comm, 0, transfer_options, model=model
)

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
    print("Approximate gradient  = ", deriv.real)
    print("Adjoint gradient      = ", grads[0][0].real)
    print("Relative error        = ", rel_error.real)
else:
    deriv = (fvals[0] - fvals_init[0]) / epsilon

    rel_error = (deriv - grads[0][0]) / deriv
    print("Approximate gradient  = ", deriv)
    print("Adjoint gradient      = ", grads[0][0])
    print("Relative error        = ", rel_error)
