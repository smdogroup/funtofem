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

from .base import Base


class Scenario(Base):
    """A class to hold scenario information for a design point in optimization"""

    def __init__(self, name, id=0, group=None, steady=True, fun3d=True, steps=1000):
        """
        Parameters
        ----------
        name: str
            name of the scenario
        id: int
            ID number of the body in the list of bodies in the model
        group: int
            group number for the scenario. Coupled variables defined in the scenario will be coupled with
            scenarios in the same group
        steady: bool
            whether the scenario's simulation is steady or unsteady
        fun3d: bool
            whether or not you are using FUN3D. If true, the scenario class will auto-populate 'aerodynamic' required by FUN3D
        steps: int
            the number of coupled time steps to run for the scenario

        See Also
        --------
        :mod:`base` : Scenario inherits from Base
        """

        from .variable import Variable as dv

        super(Scenario, self).__init__(name, id, group)

        self.name = name
        self.id = id
        self.group = group
        self.group_master = False
        self.variables = {}
        self.derivatives = {}

        self.functions = []
        self.steady = steady
        self.steps = steps

        if fun3d:
            self.add_variable("aerodynamic", dv("Mach", id=1, upper=5.0, active=False))
            self.add_variable(
                "aerodynamic", dv("AOA", id=2, lower=-15.0, upper=15.0, active=False)
            )
            self.add_variable(
                "aerodynamic", dv("Yaw", id=3, lower=-10.0, upper=10.0, active=False)
            )
            self.add_variable("aerodynamic", dv("xrate", id=4, upper=0.0, active=False))
            self.add_variable("aerodynamic", dv("yrate", id=5, upper=0.0, active=False))
            self.add_variable("aerodynamic", dv("zrate", id=6, upper=0.0, active=False))

    def add_function(self, function):
        """
        Add a new function to the scenario's function list

        Parameters
        ----------
        function: Function
            function object to be added to scenario
        """
        function.id = len(self.functions) + 1
        function.scenario = self.id

        if function.adjoint:
            for func in self.functions:
                if not func.adjoint:
                    print("Cannot add an adjoint function after a non-adjoint.")
                    print(
                        "Please reorder the functions so that all functions requiring an adjoint come first"
                    )
                    exit()

        self.functions.append(function)

        self.add_function_derivatives()

    def count_functions(self):
        """
        Returns the number of functions in this scenario

        Returns
        -------
        count: int
            number of functions in this scenario

        """
        return len(self.functions)

    def count_adjoint_functions(self):
        """
        Returns the number of functions that require an adjoint in this scenario

        Returns
        -------
        count: int
            number of adjoint-requiring functions in this scenario

        """
        is_adjoint = lambda func: func.adjoint
        return len(list(filter(is_adjoint, self.functions)))

    def add_variable(self, vartype, var):
        """
        Add a new variable to the scenario's variable dictionary

        Parameters
        ----------
        vartype: str
            type of variable
        var: Variable object
            variable to be added
        """
        var.scenario = self.id

        super(Scenario, self).add_variable(vartype, var)

        for index, _ in enumerate(self.functions):
            self.derivatives[vartype][index].append(0.0)

    def update_id(self, id):
        """
        **[model call]**
        Update the id number of the scenario

        Parameters
        ----------

        id: int
            id number of the scenario
        """
        self.id = id

        for vartype in self.variables:
            for var in self.variables[vartype]:
                var.scenario = id

        for func in self.functions:
            func.scenario = id
