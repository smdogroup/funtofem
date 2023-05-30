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

from __future__ import print_function

import numpy as np
from mpi4py import MPI

from funtofem.driver import *
from funtofem.fun3d_interface import Fun3dInterface

from build_model import *

from tacs_model import CRMtacs
from tacs import TACS

# split the communicator
n_tacs_procs = 2
comm = MPI.COMM_WORLD

world_rank = comm.Get_rank()
if world_rank < n_tacs_procs:
    color = 55
    key = world_rank
else:
    color = MPI.UNDEFINED
    key = world_rank
tacs_comm = comm.Split(color, key)


# build the model
crm = build_model()
steps = crm.scenarios[0].steps

# Set up the TACS integrator
options = {
    "integrator": "BDF",
    "start_time": 0.0,
    "step_size": 0.001,
    "steps": steps,
    "integration_order": 2,
    "solver_rel_tol": 1.0e-10,
    "solver_abs_tol": 1.0e-9,
    "max_newton_iters": 50,
    "femat": 1,
    "print_level": 1,
    "output_freq": 10,
    "ordering": TACS.PY_RCM_ORDER,
}

solvers = {}

# instantiate the fem_solver
solvers["structural"] = CRMtacs(comm, tacs_comm, options, crm, n_tacs_procs)

# instantiate the flow_solver
forward_options = {"timedep_adj_frozen": True}
adjoint_options = {"timedep_adj_frozen": True}
solvers["flow"] = Fun3dInterface(
    comm,
    crm,
    flow_dt=0.001,
    forward_options=forward_options,
    adjoint_options=adjoint_options,
)


transfer_options = {}
transfer_options["scheme"] = "MELD"
transfer_options["isym"] = -1
transfer_options["beta"] = 0.5
transfer_options["npts"] = 30

# instantiate the driver
driver = FUNtoFEMnlbgs(
    solvers, comm, tacs_comm, 0, comm, 0, transfer_options=transfer_options, model=crm
)

# run the forward analysis
fail = driver.solve_forward()

vrs = crm.get_variables()
funcs = crm.get_functions()
if comm.Get_rank() == 0:
    for func in funcs:
        print("FUNCTION: " + func.name + " = ", func.value)

# run the adjoint
if TransferScheme.dtype != complex:
    fail = driver.solve_adjoint()

    derivatives = crm.get_function_gradients()
    if comm.Get_rank() == 0:
        for i, func in enumerate(funcs):
            print("FUNCTION: " + funcs[i].name + " = ", funcs[i].value)
            for j, var in enumerate(vrs):
                print(" var " + var.name, derivatives[i][j])
