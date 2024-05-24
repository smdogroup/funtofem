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

__all__ = ["Variable"]

from ._base import Base
import importlib
from .composite_function import CompositeFunction

# optional tacs import for caps2tacs
tacs_loader = importlib.util.find_spec("tacs")
caps_loader = importlib.util.find_spec("pyCAPS")
if tacs_loader is not None and caps_loader is not None:
    from tacs import caps2tacs


class Variable(object):
    """
    Design variable type for FUNtoFEM. Valid variable types are "structural", "aerodynamic", and "shape". For example, invoked by "Variable.structural('thickness', 0.1)"
    """

    ANALYSIS_TYPES = ["structural", "aerodynamic", "shape"]

    def __init__(
        self,
        name="unknown",
        analysis_type=None,
        value=0.0,
        lower=0.0,
        upper=1.0,
        scale=1.0,
        active=True,
        coupled=False,
        id=0,
        state=False,
    ):
        """
        Variable object for FUNtoFEM

        Parameters
        ----------
        name: str
            name of the variable
        value: float
            current value of the variable
        lower: float
            lower bound of the design variable
        upper: float
            upper bound of the design variable
        active: bool
            whether or not the design variable is active
        coupled: bool
            whether or not the design variable is coupled
        id: int
            id number of the design variable
        state: bool
            whether this variable is used as a state variable (for TACS only right now)

        Examples
        --------
        thickness = Variable(name='thickness 0', value=0.004, lower=0.001, upper=0.1)
        """

        self.name = name
        self.value = value
        self.lower = lower
        self.upper = upper
        self.active = active
        self.coupled = coupled
        self.state = state
        self.id = id
        self.scale = scale
        self.analysis_type = analysis_type

        self._scenario_name = None

    def set_bounds(
        self,
        lower=None,
        value=None,
        upper=None,
        scale=None,
        active=None,
        coupled=None,
        state=None,
    ):
        """
        Update the one or more of the attributes of the design variable

        Parameters
        ----------
        lower:float
            lower bound of the variable
        value: float
            new value of the variable
        upper: float
            upper bound of the design variable
        scale: float
            scale of the variable for the optimizer
        active: bool
            whether or not the design variable is active
        coupled: bool
            whether or not the design variable is coupled
        state: bool
            whether this variable is used as a state variable (for TACS only right now)
        """

        if value is not None:
            self.value = value
        if lower is not None:
            self.lower = lower
        if upper is not None:
            self.upper = upper
        if scale is not None:
            self.scale = scale
        if active is not None:
            self.active = active
        if coupled is not None:
            self.coupled = coupled
        if state is not None:
            self.state = state

        # return the object for method cascading
        return self

    @property
    def full_name(self) -> str:
        """full scenario name if scenario variable, otherwise just regular name"""
        if self._scenario_name is None:
            return self.name
        else:
            return f"{self._scenario_name}-{self.name}"

    @classmethod
    def structural(cls, name: str, value=0.0):
        """
        Create a structural analysis variable.
        (make sure to set optimal settings and then register it)
        """
        return cls(name=name, value=value, analysis_type="structural")

    @classmethod
    def aerodynamic(cls, name: str, value=0.0):
        """
        Create an aerodynamic analysis variable.
        (make sure to set optimal settings and then register it)
        """
        return cls(name=name, value=value, analysis_type="aerodynamic")

    @classmethod
    def shape(cls, name: str, value=0.0):
        """
        Create a shape analysis variable.
        (make sure to set optimal settings and then register it)
        """
        return cls(name=name, value=value, analysis_type="shape")

    def rescale(self, factor: float):
        """
        rescale the lb, value, ub of the variable
        """
        self.lower *= factor
        self.value *= factor
        self.upper *= factor

        # return the object for method cascading
        return self

    def register_to(self, base):
        """
        register a variable with previously defined analysis type to
        a body or scenario
        """
        assert self.analysis_type is not None
        assert isinstance(base, Base)

        # add variable to the base object - either scenario or body
        base.add_variable(vartype=self.analysis_type, var=self)
        return self

    @classmethod
    def from_caps(self, obj):
        """
        Create a funtofem variable from a caps2tacs ThicknessVariable, ShellProperty, or ShapeVariable.
        """
        if tacs_loader is None:
            raise AssertionError(
                "Can't build from caps2tacs object if tacs module is not available"
            )
        if caps_loader is None:
            raise AssertionError(
                "Can't use caps2tacs module if esp-caps is not available"
            )

        if isinstance(obj, caps2tacs.ThicknessVariable):
            return Variable.structural(name=obj.name, value=obj.value)
        elif isinstance(obj, caps2tacs.ShellProperty):
            return Variable.structural(
                name=obj.caps_group, value=obj.membrane_thickness
            )
        elif isinstance(obj, caps2tacs.ShapeVariable):
            return Variable.shape(name=obj.name, value=obj.value)
        else:
            raise AssertionError("Input caps2tacs object not appropriate type.")

    @property
    def composite_function(self):
        """turn this variable into a composite function"""
        return CompositeFunction.cast(self)

    # arithmetic with composite functions
    def __add__(self, func):
        return self.composite_function.__add__(func)

    def __radd__(self, func):
        return self.composite_function.__radd__(func)

    def __sub__(self, func):
        return self.composite_function.__sub__(func)

    def __rsub__(self, func):
        return self.composite_function.__rsub__(func)

    def __mul__(self, func):
        return self.composite_function.__mul__(func)

    def __rmul__(self, func):
        return self.composite_function.__rmul__(func)

    def __truediv__(self, func):
        return self.composite_function.__truediv__(func)

    def __rtruediv__(self, func):
        return self.composite_function.__rtruediv__(func)

    def __pow__(self, func):
        return self.composite_function.__pow__(func)
