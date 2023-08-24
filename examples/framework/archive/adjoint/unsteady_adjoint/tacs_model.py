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
from tacs import TACS, elements, functions, constitutive
from tacs_builder import TACSBuilder, ShellStiffness
from funtofem.tacs_interface_unsteady import TacsUnsteadyInterface
import numpy as np


class CRMtacs(TacsUnsteadyInterface):
    def __init__(self, comm, tacs_comm, options, model, n_tacs_procs):
        # Specify material properties
        self.comm = comm
        self.tacs_comm = tacs_comm
        self.tacs_proc = False
        if self.comm.Get_rank() < n_tacs_procs:
            self.tacs_proc = True
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
            self.builder = TACSBuilder(tacs_comm)

            shellStiff = ShellStiffness(rho, E, nu, kcorr, ys, thickness, tmin, tmax)
            wing = self.builder.addMITCShellBody(
                "wing", "CRM_box_2nd.bdf", 0, shellStiff, isFixed=False
            )

        super(CRMtacs, self).__init__(options, model, ndof=6)
