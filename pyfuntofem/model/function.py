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

__all__ = ["Function"]


class Function(object):
    """holds component function information in FUNtoFEM"""

    def __init__(
        self,
        name,
        id=0,
        value=0.0,
        start=0,
        stop=-1,
        analysis_type=None,
        body=-1,
        adjoint=True,
        options=None,
        averaging=None,
    ):
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
        self.name = name
        self.id = id
        self.start = start
        self.stop = stop
        self.averaging = averaging

        self.analysis_type = analysis_type
        self.scenario = None
        self.body = body

        # whether or not an adjoint is required
        self.adjoint = adjoint

        # any function options or parameters to pass to the solver
        self.options = options

        # Store the value of the function here
        self.value = value

        # Store the values of the derivatives w.r.t. this function
        self.derivatives = {}

        return

    def zero_derivatives(self):
        """
        Zero all the derivative values that are currently set
        """

        for var in self.derivatives:
            self.derivatives[var] = 0.0

        return

    def set_gradient_component(self, var, value):
        """
        Set the gradient value for the given variable into the dictionary of gradient values

        This call will overwrite the gradient value stored by the function.

        Parameter
        ---------
        var: Variable object
            Derivative of this function w.r.t. the given variable
        value: scalar value
            Value of the gradient
        """

        self.derivatives[var] = value

        return

    def add_gradient_component(self, var, value):
        """
        Add the gradient value for the given variable into the dictionary of gradient values

        Parameter
        ---------
        var: Variable object
            Derivative of this function w.r.t. the given variable
        value: scalar value
            Value of the gradient contribution
        """

        if var in self.derivatives:
            self.derivatives[var] += value
        else:
            self.derivatives[var] = value

        return

    def get_gradient_component(self, var):
        """
        Get the gradient value stored - return 0 if not defined

        Parameter
        ---------
        var: Variable object
            Derivative of this function w.r.t. the given variable
        """

        if var in self.derivatives:
            return self.derivatives[var]

        return 0.0

    @classmethod
    def ksfailure(
        cls, ks_weight: float = 50.0, start: int = 0, stop: int = -1, body: int = -1
    ):
        """
        Class constructor for the KS Failure function
        """
        return cls(
            name="ksfailure",
            analysis_type="structural",
            options={"ksWeight": ks_weight},
            start=start,
            stop=stop,
            body=body,
        )

    @classmethod
    def mass(cls, start: int = 0, stop: int = -1, body: int = -1):
        """
        Class constructor for the Mass function
        """
        return cls(
            name="mass", analysis_type="structural", start=start, stop=stop, body=body
        )

    @classmethod
    def lift(cls, start: int = 0, stop: int = -1, body: int = -1):
        """
        Class constructor for the Lift function
        """
        return cls(
            name="cl", analysis_type="aerodynamic", start=start, stop=stop, body=body
        )

    @classmethod
    def drag(cls, start: int = 0, stop: int = -1, body: int = -1):
        """
        Class constructor for the Drag function
        """
        return cls(
            name="cd", analysis_type="aerodynamic", start=start, stop=stop, body=body
        )

    @classmethod
    def temperature(cls, start: int = 0, stop: int = -1, body: int = -1):
        """
        Class constructor for the Temperature function
        """
        return cls(
            name="temperature",
            analysis_type="structural",
            start=start,
            stop=stop,
            body=body,
        )
