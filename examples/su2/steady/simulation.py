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
from pyfuntofem.su2_interface import SU2Interface
from structural_model import OneraPlate
from mpi4py import MPI

# Import SU2
import pysu2

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

SU2_CFD_ConfigFile = 'inv_ONERAM6_new.cfg'
su2 = pysu2.CSinglezoneDriver(SU2_CFD_ConfigFile, 1, comm)

# Create model
onera = FUNtoFEMmodel('onera')
wing = Body('wing', analysis_type='aeroelastic', fun3d=False)
onera.add_body(wing)
cruise = Scenario('cruise', steps=50)
onera.add_scenario(cruise)

# Instatiate the flow and structural solvers
solvers = {}

qinf = 101325.0 # freestream pressure
solvers['flow'] = SU2Interface(comm, onera, su2, qinf)
solvers['structural'] = OneraPlate(comm, tacs_comm, onera, n_tacs_procs)

# Specify the transfer scheme options
options = {'scheme': 'meld', 'beta': 0.5, 'npts': 50, 'isym': -1}

# Instantiate the driver
struct_master = 0
aero_master = 0
driver = FUNtoFEMnlbgs(solvers, comm, tacs_comm, struct_master,
                       comm, aero_master, model=onera, transfer_options=options,
                       theta_init=0.5, theta_min=0.1)

# Run the forward analysis
fail = driver.solve_forward()
