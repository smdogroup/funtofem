#!/usr/bin/env python
"""
This file is part of the package FUNtoFEM for coupled aeroelastic simulation
and design optimization.

Copyright (C) 2015 Georgia Tech Research Corporation.
Additional copyright (C) 2015 Kevin Jacobson, Jan Kiviaho and Graeme Kennedy.
All rights reserved.

FUNtoFEM is licensed under the Apache License, Version 2.0 (the "License");
you may not use this software except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import sys
from funtofem.model import *
from funtofem.driver import *
from funtofem.su2_interface import SU2Interface
from structural_model import OneraPlate
from mpi4py import MPI

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

su2_config = "inv_ONERAM6.cfg"
su2_adj_config = "inv_ONERAM6_adjoint.cfg"

# Create model
onera = FUNtoFEMmodel("onera")

wing = Body("wing", analysis_type="aeroelastic", fun3d=False)

# Add a structural design variable to the wing
t = 0.025
svar = Variable("thickness", value=t, lower=1e-3, upper=1.0)
wing.add_variable("structural", svar)

steps = 20
if "test" in sys.argv:
    steps = 1

cruise = Scenario("cruise", steps=steps)
onera.add_scenario(cruise)

drag = Function("cd", analysis_type="aerodynamic")
cruise.add_function(drag)

# failure = Function('ksfailure', analysis_type='structural')
# cruise.add_function(drag)

# Add the body after the variables
onera.add_body(wing)

# Instatiate the flow and structural solvers
solvers = {}

qinf = 101325.0  # freestream pressure
solvers["flow"] = SU2Interface(
    comm, onera, su2_config, su2ad_config=su2_adj_config, qinf=1.0
)
solvers["structural"] = OneraPlate(comm, tacs_comm, onera, n_tacs_procs)

# Specify the transfer scheme options
options = {"scheme": "meld", "beta": 0.5, "npts": 50, "isym": 1}

# Instantiate the driver
struct_master = 0
aero_master = 0
driver = FUNtoFEMnlbgs(
    solvers,
    comm,
    tacs_comm,
    struct_master,
    comm,
    aero_master,
    model=onera,
    transfer_options=options,
    theta_init=0.5,
    theta_min=0.1,
)

if "test" in sys.argv:
    fail = driver.solve_forward()
    fail = driver.solve_adjoint()

    solvers["flow"].adjoint_test(cruise, onera.bodies, epsilon=1e-7)
    solvers["structural"].adjoint_test(cruise, onera.bodies, epsilon=1e-7)
else:
    # Perform a finite difference check
    dh = 1e-6
    x0 = np.array([0.025])

    # Get the function value
    onera.set_variables(x0)
    fail = driver.solve_forward()
    funcs0 = onera.get_functions()
    f0vals = []
    for func in funcs0:
        f0vals.append(func.value)
        if comm.rank == 0:
            print("Function value: ", func.value)

    # Evaluate the function gradient
    fail = driver.solve_adjoint()
    grads = onera.get_function_gradients()
    if comm.rank == 0:
        print("Adjoint gradient: ", grads)

    # Compute the function value at the perturbed point
    x = x0 + dh
    onera.set_variables(x)
    fail = driver.solve_forward()
    funcs1 = onera.get_functions()
    f1vals = []
    for func in funcs1:
        f1vals.append(func.value)

    # Compute the function value at the perturbed point
    x = x0 - dh
    onera.set_variables(x)
    fail = driver.solve_forward()
    funcs0 = onera.get_functions()
    f0vals = []
    for func in funcs0:
        f0vals.append(func.value)

    if comm.rank == 0:
        for k, funcs in enumerate(zip(f0vals, f1vals)):
            print("Function value: ", funcs[0], funcs[1])
            print("Adjoint gradient: ", grads)
            print("Finite-difference: ", 0.5 * (funcs[1] - funcs[0]) / dh)
