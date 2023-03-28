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

import numpy as np
from .composite_function import CompositeFunction


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

        self._scenario_name = None

        return

    @property
    def full_name(self) -> str:
        return f"{self._scenario_name}.{self.name}"

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

    def register_to(self, scenario):
        """register the function to the scenario"""
        scenario.include(self)
        return self

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

    @classmethod
    def xcom(cls, start: int = 0, stop: int = -1, body: int = -1):
        """Class constructor for the x center of mass TACS function"""
        return cls(
            name="xcom", analysis_type="structural", start=start, stop=stop, body=body
        )

    @classmethod
    def ycom(cls, start: int = 0, stop: int = -1, body: int = -1):
        """Class constructor for the y center of mass TACS function"""
        return cls(
            name="ycom", analysis_type="structural", start=start, stop=stop, body=body
        )

    @classmethod
    def zcom(cls, start: int = 0, stop: int = -1, body: int = -1):
        """Class constructor for the z center of mass TACS function"""
        return cls(
            name="zcom", analysis_type="structural", start=start, stop=stop, body=body
        )

    @classmethod
    def compliance(cls, start: int = 0, stop: int = -1, body: int = -1):
        """Class constructor for the compliance TACS function"""
        return cls(
            name="compliance",
            analysis_type="structural",
            start=start,
            stop=stop,
            body=body,
        )

    @classmethod
    def composite_name(cls, func1, func2):
        if isinstance(func1, Function) or isinstance(func1, CompositeFunction):
            left = func1.full_name
        elif (
            isinstance(func1, int)
            or isinstance(func1, float)
            or isinstance(func1, complex)
        ):
            left = "float"
        else:
            raise AssertionError(f"Func1 has unsupported type {type(func1)}")
        if isinstance(func2, Function) or isinstance(func2, CompositeFunction):
            right = func2.full_name
        elif (
            isinstance(func2, int)
            or isinstance(func2, float)
            or isinstance(func2, complex)
        ):
            right = "float"
        else:
            raise AssertionError(f"Func2 has unsupported type {type(func2)}")
        return f"{left}-{right}"

    def __add__(self, func):
        if isinstance(func, Function):

            def eval_hdl(funcs_dict):
                return funcs_dict[self.full_name] + funcs_dict[func.full_name]

            func_name = func.name
            functions = [self, func]
        elif isinstance(func, CompositeFunction):

            def eval_hdl(funcs_dict):
                return funcs_dict[self.full_name] + func.eval_hdl(funcs_dict)

            func_name = func.name
            functions = [self] + func.functions
        elif (
            isinstance(func, int)
            or isinstance(func, float)
            or isinstance(func, complex)
        ):

            def eval_hdl(funcs_dict):
                return funcs_dict[self.full_name] + func

            func_name = "float"
            functions = [self]
        else:
            raise AssertionError("Addition Overload failed for unsupported type.")
        return CompositeFunction(
            name=f"{self.name}+{func_name}", eval_hdl=eval_hdl, functions=functions
        )

    def __radd__(self, func):
        if isinstance(func, Function):

            def eval_hdl(funcs_dict):
                return funcs_dict[self.full_name] + funcs_dict[func.full_name]

            func_name = func.name
            functions = [self, func]
        elif isinstance(func, CompositeFunction):

            def eval_hdl(funcs_dict):
                return funcs_dict[self.full_name] + func.eval_hdl(funcs_dict)

            func_name = func.name
            functions = [self] + func.functions
        elif (
            isinstance(func, int)
            or isinstance(func, float)
            or isinstance(func, complex)
        ):

            def eval_hdl(funcs_dict):
                return funcs_dict[self.full_name] + func

            func_name = "float"
            functions = [self]
        else:
            raise AssertionError(
                "Reflected Addition Overload failed for unsupported type."
            )
        return CompositeFunction(
            name=f"{self.name}+{func_name}", eval_hdl=eval_hdl, functions=functions
        )

    def __sub__(self, func):
        if isinstance(func, Function):

            def eval_hdl(funcs_dict):
                return funcs_dict[self.full_name] - funcs_dict[func.full_name]

            func_name = func.name
            functions = [self, func]
        elif isinstance(func, CompositeFunction):

            def eval_hdl(funcs_dict):
                return funcs_dict[self.full_name] - func.eval_hdl(funcs_dict)

            func_name = func.name
            functions = [self] + func.functions
        elif (
            isinstance(func, int)
            or isinstance(func, float)
            or isinstance(func, complex)
        ):

            def eval_hdl(funcs_dict):
                return funcs_dict[self.full_name] - func

            func_name = "float"
            functions = [self]
        else:
            raise AssertionError("Subtraction Overload failed for unsupported type.")
        return CompositeFunction(
            name=f"{self.name}-{func_name}", eval_hdl=eval_hdl, functions=functions
        )

    def __rsub__(self, func):
        if isinstance(func, Function):

            def eval_hdl(funcs_dict):
                return funcs_dict[self.full_name] - funcs_dict[func.full_name]

            func_name = func.name
            functions = [self, func]
        elif isinstance(func, CompositeFunction):

            def eval_hdl(funcs_dict):
                return funcs_dict[self.full_name] - func.eval_hdl(funcs_dict)

            func_name = func.name
            functions = [self] + func.functions
        elif (
            isinstance(func, int)
            or isinstance(func, float)
            or isinstance(func, complex)
        ):

            def eval_hdl(funcs_dict):
                return funcs_dict[self.full_name] - func

            func_name = "float"
            functions = [self]
        else:
            raise AssertionError(
                "Reflected Subtraction Overload failed for unsupported type."
            )
        return CompositeFunction(
            name=f"{self.name}-{func_name}", eval_hdl=eval_hdl, functions=functions
        )

    def __mul__(self, func):
        if isinstance(func, Function):

            def eval_hdl(funcs_dict):
                return funcs_dict[self.full_name] * funcs_dict[func.full_name]

            func_name = func.name
            functions = [self, func]
        elif isinstance(func, CompositeFunction):

            def eval_hdl(funcs_dict):
                return funcs_dict[self.full_name] * func.eval_hdl(funcs_dict)

            func_name = func.name
            functions = [self] + func.functions
        elif (
            isinstance(func, int)
            or isinstance(func, float)
            or isinstance(func, complex)
        ):

            def eval_hdl(funcs_dict):
                return funcs_dict[self.full_name] * func

            func_name = "float"
            functions = [self]
        else:
            raise AssertionError("Multiple Overload failed for unsupported type.")
        return CompositeFunction(
            name=f"{self.name}*{func_name}", eval_hdl=eval_hdl, functions=functions
        )

    def __rmul__(self, func):
        if isinstance(func, Function):

            def eval_hdl(funcs_dict):
                return funcs_dict[self.full_name] * funcs_dict[func.full_name]

            func_name = func.name
            functions = [self, func]
        elif isinstance(func, CompositeFunction):

            def eval_hdl(funcs_dict):
                return funcs_dict[self.full_name] * func.eval_hdl(funcs_dict)

            func_name = func.name
            functions = [self] + func.functions
        elif (
            isinstance(func, int)
            or isinstance(func, float)
            or isinstance(func, complex)
        ):

            def eval_hdl(funcs_dict):
                return funcs_dict[self.full_name] * func

            func_name = "float"
            functions = [self]
        else:
            raise AssertionError(
                "Reflected Multiple Overload failed for unsupported type."
            )
        return CompositeFunction(
            name=f"{self.name}*{func_name}", eval_hdl=eval_hdl, functions=functions
        )

    def __truediv__(self, func):
        if isinstance(func, Function):

            def eval_hdl(funcs_dict):
                return funcs_dict[self.full_name] / funcs_dict[func.full_name]

            func_name = func.name
            functions = [self, func]
        elif isinstance(func, CompositeFunction):

            def eval_hdl(funcs_dict):
                return funcs_dict[self.full_name] / func.eval_hdl(funcs_dict)

            func_name = func.name
            functions = [self] + func.functions
        elif (
            isinstance(func, int)
            or isinstance(func, float)
            or isinstance(func, complex)
        ):

            def eval_hdl(funcs_dict):
                return funcs_dict[self.full_name] / func

            func_name = "float"
            functions = [self]
        else:
            raise AssertionError("Division Overload failed for unsupported type.")
        return CompositeFunction(
            name=f"{self.name}/{func_name}", eval_hdl=eval_hdl, functions=functions
        )

    def __rtruediv__(self, func):
        if isinstance(func, Function):

            def eval_hdl(funcs_dict):
                return funcs_dict[self.full_name] / funcs_dict[func.full_name]

            func_name = func.name
            functions = [self, func]
        elif isinstance(func, CompositeFunction):

            def eval_hdl(funcs_dict):
                return funcs_dict[self.full_name] / func.eval_hdl(funcs_dict)

            func_name = func.name
            functions = [self] + func.functions
        elif (
            isinstance(func, int)
            or isinstance(func, float)
            or isinstance(func, complex)
        ):

            def eval_hdl(funcs_dict):
                return funcs_dict[self.full_name] / func

            func_name = "float"
            functions = [self]
        else:
            raise AssertionError("Division Overload failed for unsupported type.")
        return CompositeFunction(
            name=f"{self.name}/{func_name}", eval_hdl=eval_hdl, functions=functions
        )

    def __pow__(self, func):
        if isinstance(func, Function) or isinstance(func, CompositeFunction):
            raise AssertionError("Don't raise a function to a function power.")
        elif (
            isinstance(func, int)
            or isinstance(func, float)
            or isinstance(func, complex)
        ):

            def eval_hdl(funcs_dict):
                return funcs_dict[self.full_name] ** func

            func_name = "float"
            functions = [self]
        else:
            raise AssertionError("Division Overload failed for unsupported type.")
        return CompositeFunction(
            name=f"{self.name}**{func_name}", eval_hdl=eval_hdl, functions=functions
        )
