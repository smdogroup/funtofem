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

This example demonstrates functionality for performing aeroelastic analysis
with piston theory and a simple wing structure
"""

import sys
import numpy as np
from pyfuntofem.model import *
from pyfuntofem.driver import *
from pyfuntofem.pistontheory_interface import PistonInterface
from pyfuntofem.tacs_interface import TacsSteadyInterface
from structural_model import OneraPlate
from mpi4py import MPI


# No need to split the communicator? Maybe only for tacs, otherwise
# Fluid solve Serial Only
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

# Create model
onera = FUNtoFEMmodel("onera")

wing = Body("wing", analysis_type="aeroelastic", fun3d=False)

# Add a structural design variable to the wing
t = 0.025
# svar = Variable('thickness', value=t, lower=1e-3, upper=1.0)
# wing.add_variable('structural', svar)

# Aero design var
avar = Variable("AOA", value=5.0, lower=0.1, upper=11)
wing.add_variable("aerodynamic", avar)

steps = 50
if "test" in sys.argv:
    steps = 1

cruise = Scenario("cruise", steps=steps)
onera.add_scenario(cruise)

drag = Function("cl", analysis_type="aerodynamic")
cruise.add_function(drag)

# failure = Function('ksfailure', analysis_type='structural')
# cruise.add_function(drag)

# Add the body after the variables
onera.add_body(wing)

# Instatiate the flow and structural solvers
solvers = {}

qinf = 101325.0  # freestream pressure Pa
M = 1.2  # Mach number
U_inf = 411  # Freestream velocity m/s
x0 = np.array([0, 0, 0])
alpha = 10  # Angle of attack (degrees)
length_dir = np.array(
    [np.cos(alpha * np.pi / 180), 0, np.sin(alpha * np.pi / 180)]
)  # Unit vec in length dir
width_dir = np.array([0, 1, 0])  # Unit vec in width dir

# Check direction to validate unit vectors (and orthogonality?)
if not (0.99 <= np.linalg.norm(length_dir) <= 1.01):
    print(
        "Length direction not a unit vector \n Calculations may be inaccurate",
        file=sys.stderr,
    )
    exit(1)
if not (0.99 <= np.linalg.norm(width_dir) <= 1.01):
    print(
        "Width direction not a unit vector \n Calculations may be inaccurate",
        file=sys.stderr,
    )
    exit(1)
if not (-0.01 <= np.dot(length_dir, width_dir) <= 0.01):
    print(
        "Spanning vectors not orthogonal \n Calculations may be inaccurate",
        file=sys.stderr,
    )
    exit(1)

L = 1.20  # Length
nL = 30  # Num elems in xi dir
w = 1.20  # Width
nw = 50  # Num elems in eta dir
solvers["flow"] = PistonInterface(
    comm, onera, qinf, M, U_inf, x0, length_dir, width_dir, L, w, nL, nw
)
assembler = None
if world_rank < n_tacs_procs:
    assembler = OneraPlate(tacs_comm)
solvers["structural"] = TacsSteadyInterface(comm, onera)

# Specify the transfer scheme options
options = {"scheme": "meld", "beta": 0.9, "npts": 10, "isym": 1}

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
    theta_init=0.3,
    theta_min=0.01,
)

if "test" in sys.argv:
    fail = driver.solve_forward()
    exit(1)
    fail = driver.solve_adjoint()

    solvers["flow"].adjoint_test(cruise, onera.bodies, epsilon=1e-7)
    solvers["structural"].adjoint_test(cruise, onera.bodies, epsilon=1e-7)
else:
    # Perform a finite difference check
    dh = 1e-6
    x0 = np.array([1.5])

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
