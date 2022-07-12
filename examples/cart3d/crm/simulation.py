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
from pyfuntofem.model import *
from pyfuntofem.driver import *
from pyfuntofem.cart3d_interface import *
from structural_model import TacsCRM
from mpi4py import MPI

def build_model():
    crm = FUNtoFEMmodel('crm')
    wing = Body('wing', id=2, fun3d=False)
    crm.add_body(wing)
    cruise = Scenario('cruise', steps=100)
    crm.add_scenario(cruise)

    return crm


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
crm = build_model()

# Instatiate the flow and structural solvers
solvers= {}

pinf = 101325  # freestream pressure
gamma = 1.4    # ratio of specific heats
with_conv_hist = True
adapt_growth = [9]
solvers['flow'] = Cart3DInterface(comm, crm, pinf, gamma, with_conv_hist, adapt_growth)
solvers['structural'] = TacsCRM(comm, tacs_comm, crm, n_tacs_procs)

# Specify the transfer scheme options
options = {'scheme': 'meld', 'beta': 0.5, 'npts': 500, 'isym': -1}

# Instantiate the driver
struct_master = 0
aero_master = 0
driver = FUNtoFEMnlbgs(solvers, comm, tacs_comm, struct_master,
                       comm, aero_master, model=crm, transfer_options=options)

# Run the forward analysis
fail = driver.solve_forward()
