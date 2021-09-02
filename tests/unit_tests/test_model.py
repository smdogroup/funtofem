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
import unittest

class ModelTest(unittest.TestCase):
    def build_model(self):
        rotor_model = FUNtoFEMmodel('rotorcraft')

        # Set up the scenarios

        # hover
        hover = Scenario(name='hover', group=0,steady=True,fun3d=True)

        hover.set_variable('aerodynamic','AOA',value=3.0,lower=-3.0,upper=10.0,active=True)

        thrust = Function('thrust',analysis_type='aerodynamic')
        hover.add_function(thrust)

        power = Function('power',analysis_type='aerodynamic')
        hover.add_function(power)

        mass = Function('mass',analysis_type='structural',adjoint=False)
        hover.add_function(mass)

        rotor_model.add_scenario(hover)

        # forward_flight
        forward_flight = Scenario(name='forward_flight', group=1,steady=False,fun3d=True, steps =1080)

        forward_flight.set_variable('aerodynamic','AOA',value=5.0,lower=-3.0,upper=10.0,active=True)

        thrust = Function('thrust',analysis_type='aerodynamic',start = 721, stop =1080, averaging = True)
        forward_flight.add_function(thrust)

        thrust = Function('power',analysis_type='aerodynamic',start = 721, stop =1080, averaging = True)
        forward_flight.add_function(thrust)

        roll = Function('roll moment',analysis_type='aerodynamic',start = 721, stop =1080, averaging = True)
        forward_flight.add_function(roll)

        pitch = Function('pitch moment',analysis_type='aerodynamic',start = 721, stop =1080, averaging = True)
        forward_flight.add_function(pitch)

        options = {'ksweight':50.0}
        ks = Function('ksFailure',analysis_type='structural',options=options,averaging = False)
        forward_flight.add_function(ks)

        rotor_model.add_scenario(forward_flight)

        # Set up the bodies
        for bl in range(3):
            blade = Body('blade %i' % bl,group=1,fun3d=True, analysis_type='aeroelastic'))

            # Add the coupled thickness variables
            thicknesses = np.array([ 1.0, 2.0 ,3.0, 4.0])
            for i in range(thicknesses.size):
                thick = Variable('thickness %i' % i, value=thicknesses[i], lower = 0.0, upper = 5.0,
                                 coupled = True)
                blade.add_variable('structural',thick)

            # Add the uncoupled variable
            thick = Variable('thickness %i' % 4, value = 5.0, lower = 0.003, upper = 5.0, coupled = False)
            blade.add_variable('structural',thick)

            # Add controls variables
            collective = Variable('collective', value = 0.2, lower = -0.1, upper = 1.0, coupled = True)
            blade.add_variable('controls',collective)

            cyclic_lat = Variable('cyclic_lat', value = 0.1, lower = -0.1, upper = 1.0, coupled = True)
            blade.add_variable('controls',cyclic_lat)

            cyclic_long = Variable('cyclic_long', value = 0.3, lower = -0.1, upper = 1.0, coupled = True)
            blade.add_variable('controls',cyclic_long)

            rotor_model.add_body(blade)
        return rotor_model

    def test_count_functions(self):
        model = self.build_model()
        assert model.count_functions() == 8

    def test_get_functions(self):
        model = self.build_model()
        functions = model.get_functions()

        assert functions[0].name == 'thrust'
        assert functions[0].scenario == 1
        assert functions[0].analysis_type == 'aerodynamic'

        assert functions[1].name == 'power'
        assert functions[1].scenario == 1

        assert functions[2].name == 'mass'
        assert functions[2].scenario == 1
        assert functions[2].analysis_type == 'structural'

        assert functions[3].name == 'thrust'
        assert functions[3].scenario == 2
        assert functions[3].analysis_type == 'aerodynamic'

        assert functions[4].name == 'power'
        assert functions[4].scenario == 2
        assert functions[4].analysis_type == 'aerodynamic'

        assert functions[5].name == 'roll moment'
        assert functions[5].scenario == 2
        assert functions[5].analysis_type == 'aerodynamic'

        assert functions[6].name == 'pitch moment'
        assert functions[6].scenario == 2
        assert functions[6].analysis_type == 'aerodynamic'

        assert functions[7].name == 'ksFailure'
        assert functions[7].scenario == 2
        assert functions[7].analysis_type == 'structural'

    def test_get_variables(self):
        model = self.build_model() 

        var_list = model.get_variables()

        assert len(var_list) == 2 + 4 + 3 + 3
        assert var_list[0].name == 'AOA'
        assert var_list[0].value == 3.0
        assert var_list[0].analysis_type == 'aerodynamic'
        assert var_list[0].scenario == 1

        assert var_list[1].name == 'AOA'
        assert var_list[1].value == 5.0
        assert var_list[1].analysis_type == 'aerodynamic'
        assert var_list[1].scenario == 2

        assert var_list[2].name == 'collective'
        assert var_list[2].value == 0.2
        assert var_list[2].analysis_type == 'controls'

        assert var_list[5].name == 'thickness 0'
        assert var_list[5].value == 1.0
        assert var_list[5].analysis_type == 'structural'

        for i in range(3):
            assert var_list[9+i].name == 'thickness 4'
            assert var_list[9+i].value == 5.0
            assert var_list[9+i].body  == i + 1
            assert var_list[9+i].analysis_type == 'structural'


    def test_set_variables(self):
       model = self.build_model() 

       var_list = model.get_variables()

       offset = 0.1
       for var in var_list:
           var.assign(value=var.value + offset)

       model.set_variables(var_list)
       assert model.scenarios[0].variables['aerodynamic'][1].value == 3.0 + offset
       assert model.scenarios[1].variables['aerodynamic'][1].value == 5.0 + offset

       for i in range(3):
           assert model.bodies[i].variables['structural'][0].value == 1.0 + offset
           assert model.bodies[i].variables['structural'][4].value == 5.0 + offset
           assert model.bodies[i].variables['controls'][0].value == 0.2 + offset

       offset2 = 0.2
       value_list = []
       for var in var_list:
           value_list.append(var.value + offset2)

       model.set_variables(value_list)
       assert model.scenarios[0].variables['aerodynamic'][1].value == 3.0 + offset + offset2
       assert model.scenarios[1].variables['aerodynamic'][1].value == 5.0 + offset + offset2

       for i in range(3):
           assert model.bodies[i].variables['structural'][0].value == 1.0 + offset + offset2
           assert model.bodies[i].variables['structural'][4].value == 5.0 + offset + offset2
           assert model.bodies[i].variables['controls'][0].value == 0.2 + offset + offset2

    def test_derivatives(self):
        model = self.build_model()

        # Put some dummy derivative values in 
        model.scenarios[0].derivatives['aerodynamic'][0][1] = 1.0
        model.scenarios[1].derivatives['aerodynamic'][3][1] = 2.0

        model.bodies[0].derivatives['structural'][2][0] = 3.0
        model.bodies[1].derivatives['structural'][2][0] = 2.0
        model.bodies[2].derivatives['structural'][2][0] = 1.0


        model.enforce_coupling_derivatives()
        derivs = model.get_function_gradients()

        assert derivs[0][0] == 1.0
        assert derivs[3][1] == 2.0
        assert derivs[2][5] == 6.0
