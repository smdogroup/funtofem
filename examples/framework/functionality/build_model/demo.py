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


"""
Demonstration of model set up for TOGW minimization
"""

import numpy as np
from funtofem.model import *

crm_model = FUNtoFEMmodel("crm wing")

# Set up the scenarios

# cruise
cruise = Scenario(name="cruise", group=0, steady=True, fun3d=True)

cruise.set_variable(
    "aerodynamic", "AOA", value=3.0, lower=-3.0, upper=10.0, active=True
)

lift = Function("cl", analysis_type="aerodynamic")
cruise.add_function(lift)

drag = Function("cd", analysis_type="aerodynamic")
cruise.add_function(drag)

mass = Function("mass", analysis_type="structural", adjoint=False)
cruise.add_function(mass)


crm_model.add_scenario(cruise)

# maneuver
maneuver = Scenario(name="maneuver", group=1, steady=False, fun3d=True, steps=400)

maneuver.set_variable(
    "aerodynamic", "AOA", value=5.0, lower=-3.0, upper=10.0, active=True
)

lift = Function("cl", analysis_type="aerodynamic", start=100, stop=400, averaging=True)
maneuver.add_function(lift)

options = {"ksweight": 50.0}
ks = Function("ksFailure", analysis_type="structural", options=options, averaging=False)
maneuver.add_function(ks)

crm_model.add_scenario(maneuver)

# Set up the body
wing = Body("wing", "aeroelastic", group=1, fun3d=True)

# Add the thickness variables
thicknesses = np.loadtxt("thicknesses.dat")
for i in range(thicknesses.size):
    thick = Variable(
        "thickness " + str(i),
        value=thicknesses[i],
        lower=0.003,
        upper=0.05,
        coupled=True if i % 2 == 0 else False,
    )
    wing.add_variable("structural", thick)

# Add the shape variables
shapes = np.loadtxt("shape_vars.dat")
for i in range(shapes.shape[0]):
    shpe = Variable(
        "shape " + str(i), value=shapes[i, 0], lower=shapes[i, 1], upper=shapes[i, 2]
    )
    wing.add_variable("shape", shpe)

crm_model.add_body(wing)

wing2 = Body("wing 2", "aeroelastic", group=1, fun3d=True)
for i in range(thicknesses.size):
    thick = Variable(
        "thickness " + str(i),
        value=thicknesses[i] * 4.0,
        lower=0.003,
        upper=0.05,
        coupled=True if i % 2 == 0 else False,
    )
    wing2.add_variable("structural", thick)

crm_model.add_body(wing2)

crm_model.print_summary()
