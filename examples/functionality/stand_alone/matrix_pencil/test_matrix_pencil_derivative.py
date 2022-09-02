from __future__ import print_function

#!/usr/bin/env python
import numpy as np

from funtofem import TransferScheme
from pyfuntofem.model import FUNtoFEMmodel, Variable, Body, Scenario, Function
from pyfuntofem.driver import FUNtoFEMnlbgs, PyOptOptimization

# Import the model functions
from structural_model_nbg import SpringStructure
from fake_aerodynamics import FakeAerodynamics

import argparse
from mpi4py import MPI

# The communicator
comm = MPI.COMM_WORLD

# Set parameters
minf = 0.85
flow_dt = 0.05
struct_time_step = 5e-4
alpha_init = 0.0
use_test_function = False

# Parse command line arguments for modifying the number of steps
p = argparse.ArgumentParser()
p.add_argument("--num_steps", type=int, default=1000)
p.add_argument("--use_test_function", action="store_true", default=False)
args = p.parse_args()

steps = args.num_steps
use_test_function = args.use_test_function
print("num_steps = ", steps)
print("use_test_function = ", use_test_function)

model = FUNtoFEMmodel("spring-mounted airfoil")

airfoil = Body("airfoil", "aeroelastic", group=0, boundary=1, motion_type="deform")
model.add_body(airfoil)

# Define scenario and design variables
scenario = Scenario("forward_flight", group=0, steps=steps, steady=False)

# Add the dynamic pressure variable as both aerodynamic and
# structural
qinf0 = 10000.0
lower = 5000.0
upper = 1.0e6

# Set the time step as the only structural design variables
struct_dt = Variable("struct_dt", value=1e-4, lower=0.0, upper=1.0, scaling=1.0)
scenario.add_variable("structural", struct_dt)

# Define the objective
objective = Function(
    "pitch damping estimate",
    analysis_type="structural",
    averaging=False,
    start=0,
    stop=-1,
)
scenario.add_function(objective)

model.add_scenario(scenario)

# Instantiate the flow solver
solvers = {}
forward_options = {"timedep_adj_frozen": True}
adjoint_options = {"timedep_adj_frozen": True}
solvers["flow"] = FakeAerodynamics(comm, model, flow_dt=flow_dt)

# Instantiate the structural solver
smodel = SpringStructure(
    comm,
    model,
    dtype=TransferScheme.dtype,
    use_test_function=use_test_function,
    aeroelastic_coupling=True,
)
solvers["structural"] = smodel

smodel.dt = struct_time_step
smodel.alpha0 = alpha_init

# Set the right data type
dtype = TransferScheme.dtype

# Set the mach number, flow_dt and dt values into the structural model
# so that they are consistent with what is set above
smodel.minf = minf
smodel.flow_dt = flow_dt

# Instantiate the driver
struct_comm = solvers["structural"].tacs_comm
struct_master = 0
aero_comm = comm
aero_master = 0
transfer_options = {"scheme": "meld"}
transfer_options["isym"] = -1
transfer_options["beta"] = 10.0
transfer_options["npts"] = 10
driver = FUNtoFEMnlbgs(
    solvers,
    comm,
    struct_comm,
    struct_master,
    aero_comm,
    aero_master,
    transfer_options,
    model,
)

# Create the design vector
x0 = np.zeros(1, dtype=dtype)
x0[0] = struct_time_step

# Set the design variables
model.set_variables(x0)
fail = driver.solve_forward()
funcs = model.get_functions()

fail = driver.solve_adjoint()
grads = model.get_function_gradients()

# Extract the function and gradient values
f0 = funcs[0].value
dfdx = grads[0][0].real

if TransferScheme.dtype == complex:
    # Perturb the design variables
    dh = 1e-30
    x0[0] = x0[0] + 1j * dh
    model.set_variables(x0)

    # Solve the forward problem again
    fail = driver.solve_forward()
    functions = model.get_functions()

    # Compute the CS approximation and relative error
    cs = functions[0].value.imag / dh
    rel_err = (dfdx - cs) / cs

    print("Complex step interval:      %25.15e" % (dh))
    print("Complex step value:         %25.15e" % (cs))
    print("Gradient value:             %25.15e" % (dfdx))
    print("Relative error:             %25.15e" % (rel_err))
else:
    # Try small step sizes since the relative size of the time step is
    # already quite small
    for exponent in np.linspace(-8, -11, 10):
        dh = 10**exponent
        x = x0 + dh
        model.set_variables(x)

        fail = driver.solve_forward()
        funcs = model.get_functions()

        fd = (funcs[0].value - f0) / dh
        rel_err = (dfdx - fd) / fd

        print("Finite-difference interval: %25.15e" % (dh))
        print("Finite-difference value:    %25.15e" % (fd))
        print("Gradient value:             %25.15e" % (dfdx))
        print("Relative error:             %25.15e" % (rel_err))
