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

__all__ = ["FUNtoFEMmodel"]

import numpy as np
from .variable import Variable


class FUNtoFEMmodel(object):
    """
    The FUNtoFEMmodel type holds all the data required for a FUNtoFEM coupled simulation.
    To create a model, instantiate it then add bodies and scenarios to it.
    There are functions in the model type that allow for extraction of the function and
    derivative values as well as setting the design variables

    See Also
    --------
    :mod:`body`, :mod:`scenario`
    """

    def __init__(self, name, id=0):
        """

        Parameters
        ----------
        name: str
            name of the model
        id: int
            id number of the model
        """

        self.name = name
        self.id = id

        self.scenarios = []
        self.bodies = []

    def add_body(self, body):
        """
        Add a body to the model. The body must be completely defined before adding to the model

        Parameters
        ----------
        body: body object
            the body to be added
        """
        if body.id == 0:
            body.set_id(len(self.bodies) + 1)
        else:
            body_ids = [b.id for b in self.bodies]
            if body.id in body_ids:
                print("Error: specified body id has already been assigned")
                print("Assigning a new body id")
                body.set_id(max(body_ids) + 1)

        body.group_root = True
        for by in self.bodies:
            if by.group == body.group:
                body.group_root = False
                break

        self.bodies.append(body)

    def add_scenario(self, scenario):
        """
        Add a scenario to model. The scenario must be completely defined before adding to the model

        Parameters
        ----------
        scenario: scenario object
            the scenario to be added
        """

        scenario.set_id(len(self.scenarios) + 1)

        scenario.group_root = True
        for scen in self.scenarios:
            if scen.group == scenario.group:
                scenario.group_root = False
                break

        self.scenarios.append(scenario)

    def print_summary(self, print_level=0):
        """
        Print out a summary of the assembled model for inspection

        Parameters
        ----------
        print_level: int
            how much detail to print in the summary. Print level < 0 does not print all the variables
        """

        print("==========================")
        print("= FUNtoFEM model summary =")
        print("==========================")
        print("Model name:", self.name)
        print("Number of bodies:", len(self.bodies))
        print("Number of scenarios:", len(self.scenarios))
        print(" ")
        print("------------------")
        print("| Bodies summary |")
        print("------------------")
        for body in self.bodies:
            print("Body:", body.id, body.name)
            print("    coupling group:", body.group)
            print("    transfer scheme:", type(body.transfer))
            print("    shape parameteration:", type(body.shape))
            for vartype in body.variables:
                print("    variable type:", vartype)
                print(
                    "      number of ",
                    vartype,
                    " variables:",
                    len(body.variables[vartype]),
                )
                if print_level >= 0:
                    for var in body.variables[vartype]:
                        print(
                            "        variable:",
                            var.name,
                            ", active?",
                            var.active,
                            ", coupled?",
                            var.coupled,
                        )
                        print(
                            "          value and bounds:",
                            var.value,
                            var.lower,
                            var.upper,
                        )

        print(" ")
        print("--------------------")
        print("| Scenario summary |")
        print("--------------------")
        for scenario in self.scenarios:
            print("scenario:", scenario.id, scenario.name)
            print("    coupling group:", scenario.group)
            print("    steps:", scenario.steps)
            print("    steady?:", scenario.steady)
            for func in scenario.functions:
                print(
                    "    function:", func.name, ", analysis_type:", func.analysis_type
                )
                print("      adjoint?", func.adjoint)
                if not scenario.steady:
                    print("      time range", func.start, ",", func.stop)
                    print("      averaging", func.averaging)

            for vartype in scenario.variables:
                print("    variable type:", vartype)
                print(
                    "      number of ",
                    vartype,
                    " variables:",
                    len(scenario.variables[vartype]),
                )
                if print_level >= 0:
                    for var in scenario.variables[vartype]:
                        print(
                            "      variable:",
                            var.id,
                            var.name,
                            ", active?",
                            var.active,
                            ", coupled?",
                            var.coupled,
                        )
                        print(
                            "        value and bounds:", var.value, var.lower, var.upper
                        )

        return

    def get_variables(self):
        """
        Get all the coupled and uncoupled variable objects for the entire model.
        Coupled variables only appear once.

        Returns
        -------
        var_list: list of variable objects
        """

        dv = []
        for scenario in self.scenarios:
            if scenario.group_root:
                dv.extend(scenario.get_active_variables())
            else:
                dv.extend(scenario.get_uncoupled_variables())

        for body in self.bodies:
            if body.group_root:
                dv.extend(body.get_active_variables())
            else:
                dv.extend(body.get_uncoupled_variables())

        return dv

    def set_variables(self, dv):
        """
        Set the variable values of the entire model given a list of values
        in the same order as get_variables

        Parameters
        ----------
        dv: list of float, complex, or Variable objects
            The variable values in the same order as :func:`get_variables` returns them
        """

        var_list = self.get_variables()

        for ivar, var in enumerate(var_list):
            if isinstance(dv[ivar], Variable):
                var.value = dv[ivar].value
            else:
                var.value = dv[ivar]

        return

    def count_functions(self):
        """
        Get the number of functions in the model

        Returns
        -------
        count: int
            The total number of functions in the model
        """

        return len(self.get_functions())

    def get_functions(self):
        """
        Get all the functions in the model

        Returns
        -------
        functions: list of function objects
            list of all the functions in the model ordered by the scenarios
        """

        functions = []
        for scenario in self.scenarios:
            functions.extend(scenario.functions)

        return functions

    def get_function_gradients(self):
        """
        Get the function gradients for all the functions in the model

        Returns
        -------
        gradients: list of list of floats
            derivative values
            1st index = function number in the same order as get_functions
            2st index = variable number in the same order as get_variables
        """

        functions = self.get_functions()
        variables = self.get_variables()

        gradients = []
        for func in functions:
            grad = []
            for var in variables:
                grad.append(func.get_gradient_component(var))
            gradients.append(grad)

        return gradients

    def write_sensitivity_file(self, comm, filename, discipline="aerodynamic", root=0):
        """
        Write the sensitivity file.

        This file contains the following information:

        Number of functionals

        Functional name
        Number of surface nodes
        for node in surface_nodes:
            node, dfdx, dfdy, dfdz

        Parameters
        ----------
        comm: MPI communicator
            Global communicator across all FUNtoFEM processors
        filename: str
            The name of the file to be generated
        discipline: str
            The name of the discipline sensitivity data to be written
        root: int
            The rank of the processor that will write the file
        """

        funcs = self.get_functions()

        count = 0
        ids = []
        derivs = []
        for body in self.bodies:
            id, deriv = body.collect_coordinate_derivatives(comm, discipline, root=root)
            count += len(id)
            ids.append(id)
            derivs.append(deriv)

        if comm.rank == root:
            variables = self.get_variables()
            discpline_vars = []
            for var in variables:
                # Write the variables whose analysis_type matches the discipline string.
                if discipline == var.analysis_type:
                    discpline_vars.append(var)

            # Write out the number of sets of discpline variables
            num_dvs = len(discpline_vars)

            # Write out the number of functionals and number of design variables
            data = "{} {}\n".format(len(funcs), num_dvs)

            for n, func in enumerate(funcs):
                # Print the function name
                data += "{}\n".format(func.name)

                # Print the function value
                data += "{}\n".format(func.value.real)

                # Print the number of coordinates
                data += "{}\n".format(count)

                for ibody in range(len(self.bodies)):
                    id = ids[ibody]
                    deriv = derivs[ibody]

                    for i in range(len(id)):
                        data += "{} {} {} {}\n".format(
                            int(id[i]),
                            deriv[3 * i, n].real,
                            deriv[3 * i + 1, n].real,
                            deriv[3 * i + 2, n].real,
                        )

                for var in discpline_vars:
                    deriv = func.get_gradient_component(var)
                    deriv = deriv.real

                    # Write the variable name and derivative value
                    data += var.name + "\n"
                    data += "1\n"
                    data += str(deriv) + "\n"

            with open(filename, "w") as fp:
                fp.write(data)

        return
