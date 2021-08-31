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

from pyfuntofem.model import Body
from pyfuntofem.model import Variable
import unittest

class BodyTest(unittest.TestCase):
    def test_body(self):
        body = Body(name='test body', id = 1, group = 2, boundary = 3, fun3d = False, motion_type='deform+rigid')

        assert body.name == 'test body'
        assert body.id == 1
        assert body.group == 2
        assert body.boundary == 3
        assert body.motion_type == 'deform+rigid'

    def create_body(self):
        body = Body(name='test body', id = 1, group = 2, boundary = 3, fun3d = False, motion_type='deform+rigid')
        body.update_id

        var = Variable(name='var 1',value = 0.0, active = True, coupled = False)
        body.add_variable('aerodynamic',var)

        var2 = Variable(name='var 2',value = 1.0, active = True, coupled = True)
        body.add_variable('aerodynamic',var2)

        var3 = Variable(name='var 3',value = 1.0,active = False, coupled = False)
        body.add_variable('structural',var3)

        return body

    def test_body_variable_functionality():
        body = create_body(self)

        assert body.variables['aerodynamic'][0].name == 'var 1'
        assert body.variables['aerodynamic'][0].body == body.id
        assert body.variables['aerodynamic'][0].value== 0.0

        assert body.variables['aerodynamic'][1].name == 'var 2'
        assert body.variables['aerodynamic'][1].body == body.id
        assert body.variables['aerodynamic'][1].value== 1.0

    def test_body_update_id(self):
        body = create_body()
        body.update_id(2)

        assert body.id == 2
        assert body.variables['aerodynamic'][0].body   == body.id
        assert body.variables['aerodynamic'][1].body   == body.id
        assert body.variables['structural'][0].body    == body.id

    def test_body_set_variable(self):
        body = create_body()

        # set by index
        body.set_variable('aerodynamic',index=1,value = 4.0,lower=-2.0,upper=6.0,scaling = 9.0, active = False, coupled = False)

        assert body.variables['aerodynamic'][1].value  == 4.0
        assert body.variables['aerodynamic'][1].lower  == -2.0
        assert body.variables['aerodynamic'][1].upper  == 6.0
        assert body.variables['aerodynamic'][1].scaling == 9.0
        assert body.variables['aerodynamic'][1].active  == False
        assert body.variables['aerodynamic'][1].coupled == False

        # set by name
        body.set_variable('structural',name='var 3',value = 3.0)

        assert body.variables['structural'][0].value  == 3.0

        # set by list
        body.set_variable('aerodynamic',index=[0,1],value = 5.0,active = True)

        assert body.variables['aerodynamic'][0].value  == 5.0
        assert body.variables['aerodynamic'][1].value  == 5.0

        assert body.variables['aerodynamic'][0].active  == True
        assert body.variables['aerodynamic'][1].active  == True

    def test_body_count_variables(self):
        body = create_body()
        active_vars_count = body.count_active_variables()

        assert  active_vars_count == 2

        body.set_variable('aerodynamic',index=[0,1],active = False)

        active_vars_count = body.count_active_variables()

        assert  active_vars_count == 0

        body.set_variable('aerodynamic',index=[0,1],active = True)
        body.set_variable('structural',index=0,active = True)
        var_count = body.count_uncoupled_variables()
        assert var_count == 2

    def test_body_active_variables(self):
        body = create_body()

        vars = body.active_variables()

        assert len(vars) == body.count_active_variables()
        assert vars[0].name == 'var 1'
        assert vars[1].name == 'var 2'

    def test_body_uncoupled_variables(self):
        body = create_body()

        vars = body.uncoupled_variables()

        assert len(vars) == body.count_uncoupled_variables()
        assert vars[0].name == 'var 1'
