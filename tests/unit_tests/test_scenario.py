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
from pyfuntofem.model import Scenario, Function, Variable
import unittest

class ScenarioTest(unittest.TestCase):
    # Note: most of the functionality for Scenario is tested by test_body.py
    def build_scenario(self):
        cruise = Scenario(name='cruise', group=0,steady=False,fun3d=True,steps=10)

        drag = Function('cd',analysis_type='aerodynamic')
        cruise.add_function(drag)

        mass = Function('mass',analysis_type='structural',adjoint=False)
        cruise.add_function(mass)

        cruise.set_variable('aerodynamic','AOA', value = 5.0)

        return cruise

    def test_build_scenario(self):
        scenario = self.build_scenario()
        
        assert scenario.name == 'cruise'

        # Check the attributes
        assert scenario.steps == 10
        assert scenario.steady == False

        # Check the variables
        assert scenario.variables['aerodynamic'][0].name == 'Mach'
        aoa = scenario.variables['aerodynamic'][1]
        assert aoa.name == 'AOA'
        assert aoa.active == True
        assert aoa.value == 5.0
        assert scenario.variables['aerodynamic'][5].name == 'zrate'

        # now the functions
        assert len(scenario.functions) == 2
        assert scenario.functions[0].name == 'cd'
        assert scenario.functions[0].analysis_type == 'aerodynamic'
        assert scenario.functions[0].adjoint == True

        assert scenario.functions[1].name == 'mass'
        assert scenario.functions[1].analysis_type == 'structural'
        assert scenario.functions[1].adjoint == False
        
