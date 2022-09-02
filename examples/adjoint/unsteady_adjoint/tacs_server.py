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
from mpi4py import MPI
from tacs_unsteady_server import Server
from tacs_builder import *
from tacs import TACS
from build_model import *


class CRMServer(Server):
    def __init__(self, comm, context, endpoint, type_, ndof, model, integrator_options):
        rho = 2500.0
        E = 70.0e9
        nu = 0.3
        kcorr = 5.0 / 6.0
        ys = 350.0e6
        thickness = 0.015
        tmin = 1.0e-4
        tmax = 1.0
        tdv = 0

        # Create an instance of TACS
        self.builder = TACSBuilder(comm)

        shellStiff = ShellStiffness(rho, E, nu, kcorr, ys, thickness, tmin, tmax)
        wing = self.builder.addMITCShellBody(
            "wing", "CRM_box_2nd.bdf", 0, shellStiff, isFixed=False
        )

        super(CRMServer, self).__init__(
            comm, context, endpoint, type_, ndof, model, integrator_options
        )


if __name__ == "__main__":
    # split the communicator
    comm = MPI.COMM_WORLD

    context = zmq.Context()
    endpoint = "tcp://*:" + str(44200 + comm.Get_rank())

    model = build_model()
    steps = model.scenarios[0].steps
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

    server = CRMServer(
        comm,
        context=context,
        endpoint=endpoint,
        type_=zmq.REP,
        ndof=6,
        model=model,
        integrator_options=options,
    )
    server.serve()
    server.close()
    context.destroy()
