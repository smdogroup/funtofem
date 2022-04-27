#!/usr/bin/env python

# This file is part of the package FUNtoFEM for coupled aeroelastic simulation
# and design optimization.

# Copyright (C) 2015 Georgia Tech Research Corporation.
# Additional copyright (C) 2015 Kevin Jacobson, Jan Kiviaho and Graeme Kennedy.
# All rights reserved.

# FUNtoFEM is licensed under the Apache License, Version 2.0 (the "License");
# you may not use this software except in compliance with the License.
# You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


class Function(object):
    """holds component function information in FUNtoFEM"""
    def __init__(self, name, id=0, value=0.0, start=0, stop=-1, analysis_type=None,
                 body=-1, adjoint=True, options=None, averaging=None):
        """

        Parameters
        ----------
        name: str
            name of the function
        id: int
            id number of function
        value: float
            value of the function
        start: int
            time step that the function window starts. If steady, then ignore
        stop: int
            time step that the function window ends. If steady, set to -1 (the default)
        analysis_type: str
            type of analysis this function is associated with: 'aerodynamic','structural'
        body: int
            body number that the function is associated with. The default is 0 which is all bodies
        adjoint: bool
            whether or not an adjoint is required for a function
        options: dict
            any options associated with the function and pass to the solvers
        averaging: bool
            whether the function is averaged or integrated or the function window. Ignored for steady functions

        Examples
        --------
        lift = Function('lift', analysis_type='aerodynamic')

        mass = Function('mass', analysis_type='structural', adjoint=False)

        ks = Function('ksFailure', analysis_type='structural', options={'ksweight':50.0})
        """
        self.name  = name
        self.id    = id
        self.start = start
        self.stop  = stop

        self.averaging = averaging

        self.value = value

        self.scenario = None

        self.analysis_type = analysis_type

        self.body = body

        # whether or not an adjoint is required
        self.adjoint = adjoint

        # any function options or parameters to pass to the solver
        self.options = options
