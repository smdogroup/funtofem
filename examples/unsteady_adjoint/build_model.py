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

def build_model():
    steps = 10
    crm = FUNtoFEMmodel('crm')

    wing = Body('wing',group=0,boundary=2)

    wing.add_variable('structural',Variable('thickness',value=0.2,lower = 0.001, upper = 0.1))
    #wing.add_variable('structural',Variable('thickness',value=0.02+1.0e-30j,lower = 0.001, upper = 0.1))

    crm.add_body(wing)

    cruise = Scenario('cruise',group=0,steps=steps,steady=False)
    cruise.set_variable('aerodynamic',name='AOA',value=3.0,lower=-15.0,upper=15.0,coupled=True)
    cruise.add_variable('aerodynamic',Variable('dynamic pressure',value=9510.486,lower = 0.001, upper = 100000.0))
    #cruise.set_variable('aerodynamic',name='AOA',value=3.0+1.0e-30j,lower=-15.0,upper=15.0)

    func = Function('cl',analysis_type='aerodynamic',start=steps,stop=steps,averaging=True)
    #func = Function('ksfailure',analysis_type='structural',start=3,stop=4)

    cruise.add_function(func)


    crm.add_scenario(cruise)

    return crm
