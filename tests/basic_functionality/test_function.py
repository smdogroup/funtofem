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
from pyfuntofem.model import Function
import unittest


class FunctionTest(unittest.TestCase):
    def test_function(self):
        func = Function(
            name="test",
            id=1,
            value=2.0,
            start=3,
            stop=4,
            analysis_type="aerodynamic",
            body=5,
            adjoint=False,
            options={"opt": 6},
            averaging=True,
        )
        assert func.name == "test"
        assert func.id == 1
        assert func.value == 2.0
        assert func.start == 3
        assert func.stop == 4
        assert func.analysis_type == "aerodynamic"
        assert func.body == 5
        assert func.adjoint == False
        assert func.options["opt"] == 6
        assert func.averaging == True
