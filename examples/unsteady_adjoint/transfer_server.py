#! /usr/bin/env python
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

import zmq
import numpy as np

from mpi4py        import MPI
from funtofem_server import Server

if __name__ == "__main__":

    n_struct_procs = 2
    comm = MPI.COMM_WORLD

    world_rank = comm.Get_rank()
    if world_rank < n_struct_procs:
        color = 55
        key = world_rank
    else:
        color = MPI.UNDEFINED
        key = world_rank
    struct_comm = comm.Split(color,key)

    transfer_options = {}
    transfer_options['scheme'] = 'MELD'
    transfer_options['isym']   =  -1
    transfer_options['beta']   =  0.5
    transfer_options['npts']   =  30

    context = zmq.Context()
    endpoint = 'tcp://*:' + str(43200+comm.Get_rank())
    server = Server(comm,struct_comm, context=context, endpoint=endpoint, type_=zmq.REP, transfer_options=transfer_options)
    server.serve()
    server.close()
    context.destroy()
