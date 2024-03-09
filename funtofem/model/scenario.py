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
from .function import Function
import numpy as np
import importlib

tacs_loader = importlib.util.find_spec("tacs")
if tacs_loader is not None:
    from funtofem.interface import TacsIntegrationSettings


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
        uncoupled_steps=0,
        adjoint_steps=None,
        min_forward_steps=None,
        min_adjoint_steps=None,
        coupling_frequency=1,
        early_stopping=False,
        T_ref=300,
        T_inf=300,
        qinf=1.0,
        flow_dt=1.0,
        tacs_integration_settings=None,
        fun3d_project_name="funtofem_CAPS",
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
        uncoupled_steps: int
            the number of fun3d iterations ran before coupled iterations
        adjoint_steps: int
            optional number of adjoint steps when using FUN3D analysis, can have different
            number of forward and adjoint steps in steady-state
        early_stopping: bool
            whether to activate the early stopping criterion
        min_forward_steps: int
            (optional) minimum number of steps required before early stopping can happen. Note
            this is set to the # of uncoupled steps if not provided (hence you probably don't need to set this
            but you can in special circumstances)
        min_adjoint_steps: int
            (optional) minimum number of adjoint steps required before early stopping criterion is applied
        T_ref: double
            Structural reference temperature (i.e., unperturbed temperature of structure) in Kelvin.
        T_inf: double
            Fluid freestream reference temperature in Kelvin.
        qinf: float
            elastic load dimensionalization factor = 0.5 * rho_inf * v_inf^2
        flow_dt: float
            Equals the nondimensional time step in fun3d.nml (time_step_nondim)
        tacs_integration_settings: :class:`~interface.TacsUnsteadyInterface`
            Optional TacsIntegrator settings for the unsteady interface (required for unsteady)
        fun3d_project_name: filename
            name of project_rootname from fun3d.nml, ex: funtofem_CAPS would have a grid file funtofem_CAPS.lb8.ugrid

        Optional Parameters/Constants
        -----------------------------
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
        self._adjoint_steps = adjoint_steps
        self.variables = {}

        self.functions = []
        self.steady = steady
        self.steps = steps
        self.coupling_frequency = coupling_frequency
        self.uncoupled_steps = uncoupled_steps
        self.tacs_integration_settings = tacs_integration_settings
        self.fun3d_project_name = fun3d_project_name

        self.T_ref = T_ref
        self.T_inf = T_inf
        self.qinf = qinf
        self.flow_dt = flow_dt

        self.suther1 = suther1
        self.suther2 = suther2
        self.gamma = gamma
        self.R_specific = R_specific
        self.Pr = Pr

        # early stopping criterion
        self.min_forward_steps = (
            min_forward_steps if min_forward_steps is not None else uncoupled_steps
        )
        self.min_adjoint_steps = (
            min_adjoint_steps if min_adjoint_steps is not None else 0
        )
        self.early_stopping = early_stopping

        # Heat capacity at constant pressure
        cp = self.R_specific * self.gamma / (self.gamma - 1)
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
    def steady(cls, name: str, steps: int, coupling_frequency:int=1, uncoupled_steps: int = 0):
        return cls(
            name=name,
            steady=True,
            steps=steps,
            coupling_frequency=coupling_frequency,
            uncoupled_steps=uncoupled_steps,
        )

    @classmethod
    def unsteady(
        cls,
        name: str,
        steps: int,
        uncoupled_steps: int = 0,
        tacs_integration_settings=None,
    ):
        return cls(
            name=name,
            steady=False,
            steps=steps,
            tacs_integration_settings=tacs_integration_settings,
            uncoupled_steps=uncoupled_steps,
        )

    @property
    def adjoint_steps(self) -> int: 
        """
        in the steady case it's best to choose the 
        adjoint steps based on the funtofem coupling frequency
        """
        if self._adjoint_steps is not None and self.steady:
            return self._adjoint_steps
        elif not self.steady:
            return None # defaults to number of steps in unsteady case
        else: # choose it based on funtofem coupling frequency in steady case
            return int(np.ceil(self.steps/self.coupling_frequency))

    @adjoint_steps.setter
    def adjoint_steps(self, new_steps:int):
        assert self.steady
        self._adjoint_steps = new_steps

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
        function._scenario_name = self.name

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

    def add_variable(self, vartype, var: Variable):
        """
        Add a new variable to the scenario's variable dictionary

        Parameters
        ----------
        vartype: str
            type of variable
        var: Variable object
            variable to be added
        """
        # var.scenario = self.id
        var._scenario_name = self.name

        super(Scenario, self).add_variable(vartype, var)

        # return object for method cascading
        return self

    def include(self, obj):
        """
        generic include method adds objects for readability
        """
        if isinstance(obj, Function):
            self.add_function(obj)
        elif isinstance(obj, Variable):
            assert obj.analysis_type is not None
            self.add_variable(vartype=obj.analysis_type, var=obj)
        elif isinstance(obj, TacsIntegrationSettings):
            self.tacs_integration_settings = obj
        else:
            raise ValueError(
                "Scenario include method does not currently support other methods"
            )

        # return the object for method cascading
        return self

    def fun3d_project(self, new_proj_name):
        """set the fun3d project rootname from fun3d.nml for use in shape drivers"""
        self.fun3d_project_name = new_proj_name
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

    def set_stop_criterion(
        self,
        early_stopping: bool = True,
        min_forward_steps=None,
        min_adjoint_steps=None,
    ):
        """
        turn on the early stopping criterion, note you probably don't need
        to set the min steps (as it defaults to the # of uncoupled steps)
        The stopping tolerances are set in each discipline interface

        Parameters
        ----------
        early_stopping: bool
            whether to perform early stopping criterion
        min_forward_steps: int
            (optional) - the minimum number of steps for engaging the early stop criterion for forward analysis
        min_adjoint_steps: int
            (optional) - the minimum number of steps for engaging the early stopping criterion for adjoint analysis
        """
        self.early_stopping = early_stopping
        if min_forward_steps is not None:
            self.min_forward_steps = min_forward_steps
        if min_adjoint_steps is not None:
            self.min_adjoint_steps = min_adjoint_steps
        return self

    def set_flow_ref_vals(self, qinf: float = 1.0, flow_dt: float = 1.0):
        """
        Set flow reference values for FUN3D nondimensionalization.
        flow_dt should always be 1.0 for steady scenarios.

        Parameters
        ----------
        flow_dt: float
            Flow solver time step size. Used to scale the adjoint term coming into and out of
            FUN3D since FUN3D currently uses a different adjoint formulation than FUNtoFEM.
            Should be equal to non-dimensional time step in fun3d.nml (equals time_step_nondim)
        qinf: float
            Dynamic pressure of the freestream flow. Used to nondimensionalize force in FUN3D.
        """

        self.qinf = qinf
        self.flow_dt = flow_dt

        if self.steady is True and float(self.flow_dt) != 1.0:
            raise ValueError("For steady cases, flow_dt must be set to 1.")
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

    def __str__(self):
        line1 = f"Scenario (<ID> <Name>): {self.id} {self.name}"
        line2 = f"    Coupling Group: {self.group}"
        line3 = f"    Steps: {self.steps}"
        line4 = f"    Steady-state: {self.steady}"

        output = (line1, line2, line3, line4)

        return "\n".join(output)
