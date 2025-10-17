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

__all__ = ["Base"]

from funtofem import TransferScheme


class Base(object):
    """
    Base class for FUNtoFEM bodies and scenarios
    """

    VAR_TYPES = [
        "structural",
        "aerodynamic",
        "rigid_motion",
        "shape",
        "control",
        "thermal",
    ]

    def __init__(self, name, id=0, group=None):
        """

        Parameters
        ----------
        name: str
            name of the body or scenario
        id: int
            id number in list of bodies or scenarios in the model
        group: int
            group number for the body or scenario. Coupled variables defined in the body/scenario will be coupled with
            bodies/scenarios in the same group

        See Also
        --------
        :mod:`body`,:mod:`scenario` : subclass the Base class
        """

        self.name = name
        self.id = id
        if group:
            self.group = group
        else:
            self.group = -1
        self.group_root = False
        self.variables = {}

        return

    def register_to(self, funtofem_model):
        """
        required method for each subclass
        """
        pass

    def add_variable(self, vartype, var):
        """
        Add a new variable to the body's or scenario's variable dictionary

        Parameters
        ----------
        vartype: str
            type of variable
        var: Variable object
            variable to be added
        """

        # if it is a new vartype add it to the dictionaries
        if not vartype in self.variables:
            self.variables[vartype] = []

        # verify specified variable type is valid
        self.verify_vartype(vartype)

        # check if the variable is already defined in the list
        for v in self.variables[vartype]:
            if v.name == var.name:
                raise ValueError(
                    "Cannot add two variables with the same name and same vartype"
                )

        # assign identifying properties to the variable then to the list
        var.id = len(self.variables[vartype]) + 1
        var.analysis_type = vartype
        self.variables[vartype].append(var)

        return

    def set_variable(
        self,
        vartype,
        name=None,
        index=None,
        value=None,
        lower=None,
        upper=None,
        active=True,
        coupled=None,
    ):
        """
        Set one or more properties of a variable given the vartype and either the variable name or a list of id's

        Parameters
        ----------
        vartype: str
            type of variable
        name: str
            name of the variable
        index: int or list of ints
            list of id numbers for the variables to modify
        value: float or complex
            value of the variable
        lower: float
            lower bound for the variable
        upper: float
            upper bound for the variable
        active: bool
            whether or not the variable is active
        coupled: bool
            whether or not the variable is coupled

        Examples
        -------
        base.set_variable('aerodynamic', name='AOA', value=3.0)

        base.set_variable('structural', index=2, active=False)

        base.set_variable('structural', index=[0,1,2,3,4], active=False)

        """

        # if it is a new vartype add it to the dictionaries
        if not vartype in self.variables:
            self.variables[vartype] = []

        # verify specified variable type is valid
        self.verify_vartype(vartype)

        if name is not None:
            for variable in self.variables[vartype]:
                if variable.name == name:
                    variable.set_bounds(
                        value=value,
                        upper=upper,
                        lower=lower,
                        active=active,
                        coupled=coupled,
                    )
                    break
        elif index is not None:
            if type(index) == list:
                for ndx in index:
                    self.variables[vartype][ndx].set_bounds(
                        value=value,
                        upper=upper,
                        lower=lower,
                        active=active,
                        coupled=coupled,
                    )
            elif type(index) == int:
                self.variables[vartype][index].set_bounds(
                    value=value,
                    upper=upper,
                    lower=lower,
                    active=active,
                    coupled=coupled,
                )
            else:
                print("Warning unknown type for index. Variable not set")
        else:
            print("Warning no valid name or index given. Variable not set")

        return

    def verify_vartype(self, vartype):
        """
        Input verification for vartype when specifying a variable.

        Parameters
        ----------
        vartype: str
            type of variable
        """

        if not vartype in self.VAR_TYPES:
            print(
                f'Warning: vartype "{vartype}" is not a recognized variable type',
                flush=True,
            )

        return

    def count_active_variables(self):
        """
        Counts the number of active variables in this body or scenario

        Returns
        -------
        count:  int
            number of active variables in the variable dictionary
        """
        is_active = lambda var: var.active == True
        count = 0
        for vartype in self.variables:
            count += len(list(filter(is_active, self.variables[vartype])))
        return count

    def count_uncoupled_variables(self):
        """
        Counts the number of variables in this body or scenario that are both uncoupled and active
        This is the number of unique variables to this object

        Returns
        -------
        count:  int
            number of uncoupled, active variables in the variable dictionary
        """
        is_coupled = lambda var: var.active == True and not var.coupled
        count = 0
        for vartype in self.variables:
            count += len(list(filter(is_coupled, self.variables[vartype])))
        return count

    def get_active_variables(self):
        """
        Get the list of active variables in body or scenario

        Returns
        -------
        active_list: list of variables
            list of active variables
        """
        full_list = []
        is_active = lambda var: var.active == True

        for vartype in self.variables:
            full_list.extend(list(filter(is_active, self.variables[vartype])))

        return full_list

    def get_inactive_variables(self):
        """
        Get the list of active variables in body or scenario

        Returns
        -------
        active_list: list of variables
            list of active variables
        """
        full_list = []
        is_active = lambda var: var.active == False

        for vartype in self.variables:
            full_list.extend(list(filter(is_active, self.variables[vartype])))

        return full_list

    def get_uncoupled_variables(self):
        """
        Get the list of uncoupled, active variables in body or scenario

        Returns
        -------
        active_list: list of variables
            list of uncoupled, active variables
        """
        full_list = []
        is_coupled = lambda var: var.active == True and not var.coupled

        for vartype in self.variables:
            full_list.extend(list(filter(is_coupled, self.variables[vartype])))

        return full_list

    def get_variable(self, varname, set_active=True):
        """get the scenario variable with matching name, helpful for FUN3D automatic variables"""
        var = None
        for discipline in self.variables:
            discipline_vars = self.variables[discipline]
            for var in discipline_vars:
                if var.name == varname:
                    if set_active:
                        var.active = True
                    return var
        if var is None:
            raise AssertionError(f"Can't find variable from scenario {self.name}")

    def set_coupled_variables(self, base):
        """
        **[model call]**
        Set the coupled variables in the body or scenario based on the input's variables

        Parameters
        ----------
        base: body or scenario object
            body or scenario to copy coupled variables from
        """

        for vartype in base.variables:
            if vartype in self.variables:
                self.variables[vartype] = [
                    v1 if v1.coupled else v2
                    for v1, v2 in zip(base.variables[vartype], self.variables[vartype])
                ]

    def set_id(self, id):
        """
        **[model call]**
        Update the id number of the body or scenario

        Parameters
        ----------
        id: int
           id number of the scenario
        """
        self.id = id

    def _print_functions(self):
        print(
            "     --------------------------------------------------------------------------------------"
        )
        self._print_long("Function", width=18, indent_line=5)
        self._print_long("Analysis Type", width=15)
        self._print_long("Comp. Adjoint", width=15)
        self._print_long("Time Range", width=20)
        self._print_long("Averaging", end_line=True)

        print(
            "     --------------------------------------------------------------------------------------"
        )
        for func in self.functions:
            analysis_type = func.analysis_type
            adjoint = func.adjoint
            start = func.start
            stop = func.stop
            averaging = func.averaging
            _time_range = " ".join(("[", str(start), ",", str(stop), "]"))
            adjoint = str(adjoint)
            self._print_long(func.name, width=18, indent_line=5)
            self._print_long(analysis_type, width=15)
            self._print_long(adjoint, width=15)
            self._print_long(_time_range, width=20)
            self._print_long(averaging, end_line=True)

        print(
            "     --------------------------------------------------------------------------------------"
        )

        return

    def _print_variables(self, vartype):
        print(
            "     ----------------------------------------------------------------------------------------------"
        )
        self._print_long("Variable", width=20, indent_line=5)
        self._print_long("Var. ID", width=10)
        self._print_long("Value", width=16)
        self._print_long("Bounds", width=24)
        self._print_long("Active", width=8)
        self._print_long("Coupled", width=9, end_line=True)

        print(
            "     ----------------------------------------------------------------------------------------------"
        )
        for var in self.variables[vartype]:
            _name = "{:s}".format(var.name)
            _id = "{: d}".format(var.id)
            _value = "{:#.8g}".format(var.value)
            _lower = "{:#.3g}".format(var.lower)
            _upper = "{:#.3g}".format(var.upper)
            _active = str(var.active)
            _coupled = str(var.coupled)
            _bounds = " ".join(("[", _lower, ",", _upper, "]"))

            self._print_long(_name, width=20, indent_line=5)
            self._print_long(_id, width=10, align="<")
            self._print_long(_value, width=16)
            self._print_long(_bounds, width=24)
            self._print_long(_active, width=8)
            self._print_long(_coupled, width=9, end_line=True)

        print(
            "     ----------------------------------------------------------------------------------------------"
        )

        return

    def _print_long(self, value, width=12, indent_line=0, end_line=False, align="^"):
        if value is None:
            value = "None"
        if indent_line > 0:
            print("{val:{wid}}".format(wid=indent_line, val=""), end="")
        if not end_line:
            print("|{val:{ali}{wid}}".format(wid=width, ali=align, val=value), end="")
        else:
            print(
                "|{val:{ali}{wid}}|".format(wid=width, ali=align, val=value), end="\n"
            )
        return
