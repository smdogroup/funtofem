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

from funtofem import TransferScheme


class Base(object):
    """
    Base class for FUNtoFEM bodies and scenarios
    """

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
                    variable.assign(
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
                    self.variables[vartype][ndx].assign(
                        value=value,
                        upper=upper,
                        lower=lower,
                        active=active,
                        coupled=coupled,
                    )
            elif type(index) == int:
                self.variables[vartype][index].assign(
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

        if not vartype in [
            "structural",
            "aerodynamic",
            "rigid_motion",
            "shape",
            "controls",
        ]:
            print(
                "Warning: vartype specified is not a recognized variable type",
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
