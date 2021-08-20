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
from pyfuntofem.model import Variable

def test_variable():
    var = Variable(name='test', value=1.0, lower=-1.0, upper = 2.0, scaling = 3.0, active=False, coupled = True, id = 4)
    assert var.name=='test'
    assert var.value==1.0
    assert var.lower==-1.0
    assert var.upper==2.0
    assert var.scaling==3.0
    assert var.active==False
    assert var.coupled==True
    assert var.id==4

def test_variable_assign():
    var = Variable(name='test')
    var.assign(value=1.0, lower=-1.0, upper = 2.0, scaling = 3.0, active=False, coupled = True)
    assert var.value==1.0
    assert var.lower==-1.0
    assert var.upper==2.0
    assert var.scaling==3.0
    assert var.active==False
    assert var.coupled==True

