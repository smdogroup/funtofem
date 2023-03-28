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

__all__ = ["Scenario"]

from ._base import Base
from .variable import Variable
from .function import Function, CompositeFunction


class Scenario(Base):
    """A class to hold scenario information for a design point in optimization"""

    def __init__(
        self,
        name,
        id=0,
        group=None,
        steady=True,
        fun3d=True,
        steps=1000,
        preconditioner_steps=0,
        T_ref=300,
        T_inf=300,
        suther1=1.458205e-6,
        suther2=110.3333,
        gamma=1.4,
        R_specific=287.058,
        Pr=0.72,
    ):
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
            the total number of fun3d time steps to run for the scenario
        preconditioner_steps: int
            the number of fun3d iterations ran before coupled iterations for preconditioning
        T_ref: double
            Structural reference temperature (i.e., unperturbed temperature of structure) in Kelvin.
        T_inf: double
            Fluid freestream reference temperature in Kelvin.
        suther1: double
            First constant in Sutherland's two-constant viscosity model. Units of kg/m-s-K^0.5
        suther2: double
            Second constant in Sutherland's two-constant viscosity model. Units of K
        gamma: double
            Ratio of specific heats.
        R_specific: double
            Specific gas constant of the working fluid (assumed air). Units of J/kg-K
        Pr: double
            Prandtl number.

        See Also
        --------
        :mod:`base` : Scenario inherits from Base
        """

        super(Scenario, self).__init__(name, id, group)

        self.name = name
        self.id = id
        self.group = group
        self.group_master = False
        self.variables = {}

        self.functions = []
        self.composite_functions = []
        self.steady = steady
        self.steps = steps
        self.preconditioner_steps = preconditioner_steps

        self.T_ref = T_ref
        self.T_inf = T_inf
        self.suther1 = suther1
        self.suther2 = suther2
        self.gamma = gamma
        self.R_specific = R_specific
        self.Pr = Pr

        # Heat capacity at constant pressure
        cp = R_specific * gamma / (gamma - 1)
        self.cp = cp

        if fun3d:
            mach = Variable("Mach", id=1, upper=5.0, active=False)
            aoa = Variable("AOA", id=2, lower=-15.0, upper=15.0, active=False)
            yaw = Variable("Yaw", id=3, lower=-10.0, upper=10.0, active=False)
            xrate = Variable("xrate", id=4, upper=0.0, active=False)
            yrate = Variable("yrate", id=5, upper=0.0, active=False)
            zrate = Variable("zrate", id=6, upper=0.0, active=False)

            self.add_variable("aerodynamic", mach)
            self.add_variable("aerodynamic", aoa)
            self.add_variable("aerodynamic", yaw)
            self.add_variable("aerodynamic", xrate)
            self.add_variable("aerodynamic", yrate)
            self.add_variable("aerodynamic", zrate)

    @classmethod
    def steady(cls, name: str, steps: int, preconditioner_steps: int = 0):
        return cls(
            name=name,
            steady=True,
            steps=steps,
            preconditioner_steps=preconditioner_steps,
        )

    @classmethod
    def unsteady(cls, name: str, steps: int, preconditioner_steps: int = 0):
        return cls(
            name=name,
            steady=False,
            steps=steps,
            preconditioner_steps=preconditioner_steps,
        )

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
        # return the object for method cascading
        return self

    def add_composite_function(self, comp_func):
        """
        add a new composite function (dependent on analysis functions) to this scenario
        """

        self.composite_functions.append(comp_func)
        return self

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

        # return object for method cascading
        return self

    def include(self, obj):
        """
        generic include method adds objects for readability
        """
        if isinstance(obj, Function):
            self.add_function(obj)
        elif isinstance(obj, CompositeFunction):
            self.add_composite_function(obj)
        elif isinstance(obj, Variable):
            assert obj.analysis_type is not None
            self.add_variable(vartype=obj.analysis_type, var=obj)
        else:
            raise ValueError(
                "Scenario include method does not currently support other methods"
            )

        # return the object for method cascading
        return self

    def register_to(self, funtofem_model):
        """
        add this scenario to the funtofem model at the end of a method cascade
        """
        funtofem_model.add_scenario(self)
        return self

    def set_temperature(self, T_ref: float = 300.0, T_inf: float = 300.0):
        """
        set structure temperature T_ref and freestream T_inf
        """
        self.T_ref = T_ref
        self.T_inf = T_inf
        return self

    def set_id(self, id):
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

    def get_thermal_conduct(self, aero_temps):
        """
        Calculate dimensional thermal conductivity at each aero surface node.
        First, use two-constant Sutherland's law to calculate viscosity for use in calculating aero heat flux.

        Parameters
        ----------
        aero_temps: np.ndarray
            Current aero surface temperatures.
        """

        # Gas constants
        s1 = self.suther1
        s2 = self.suther2
        cp = self.cp
        Pr = self.Pr

        # Compute viscosity at each aero surface node
        mu = s1 * aero_temps ** (3.0 / 2.0) / (aero_temps + s2)
        # Compute the dimensional thermal conductivity
        k = mu * cp / Pr

        return k

    def get_thermal_conduct_deriv(self, aero_temps):
        """
        Calculate derivative of thermal conductivity with respect to aero surface temperature.

        Parameters
        ----------
        aero_temps: np.ndarray
            Current aero surface temperatures.
        """

        # Gas constants
        s1 = self.suther1
        s2 = self.suther2
        cp = self.cp
        Pr = self.Pr

        # Compute viscosity at each aero surface node
        dmu_dtA = (
            s1
            * aero_temps ** (0.5)
            * (3 * s2 + aero_temps)
            / (2 * (s2 + aero_temps) ** 2)
        )
        # Compute the dimensional thermal conductivity
        dkdtA = dmu_dtA * cp / Pr

        return dkdtA
