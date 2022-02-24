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
            body.update_id(len(self.bodies)+1)
        else:
            body_ids = [b.id for b in self.bodies]
            if body.id in body_ids:
                print("Error: specified body id has already been assigned")
                print("Assigning a new body id")
                body.update_id(max(body_ids)+1)

        body.group_root = True
        for by in self.bodies:
            if by.group == body.group:
                body.group_root = False
                break

        for scenario in self.scenarios:
            for func in scenario.functions:
                body.add_function_derivatives()

        self.bodies.append(body)

    def add_scenario(self, scenario):
        """
        Add a scenario to model. The scenario must be completely defined before adding to the model

        Parameters
        ----------
        scenario: scenario object
            the scenario to be added
        """

        scenario.update_id(len(self.scenarios)+1)

        scenario.group_root = True
        for scen in self.scenarios:
            if scen.group == scenario.group:
                scenario.group_root = False
                break

        for scen in self.scenarios:
            for _ in scen.functions:
                scenario.add_function_derivatives()

        for _ in scenario.functions:
            for body in self.bodies:
                body.add_function_derivatives()

            for scen in self.scenarios:
                scen.add_function_derivatives()

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
                print('    variable type:', vartype)
                print('      number of ', vartype, ' variables:', len(body.variables[vartype]))
                if print_level >= 0:
                    for var in body.variables[vartype]:
                      print('        variable:', var.name, ', active?', var.active,', coupled?', var.coupled)
                      print('          value and bounds:', var.value, var.lower, var.upper)

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
                print('    function:', func.name, ', analysis_type:', func.analysis_type)
                print('      adjoint?', func.adjoint)
                if not scenario.steady:
                    print('      time range', func.start, ',', func.stop)
                    print('      averaging', func.averaging)


            for vartype in scenario.variables:
                print('    variable type:', vartype)
                print('      number of ', vartype, ' variables:', len(scenario.variables[vartype]))
                if print_level >= 0:
                    for var in scenario.variables[vartype]:
                        print('      variable:', var.id, var.name, ', active?', var.active,', coupled?', var.coupled)
                        print('        value and bounds:', var.value, var.lower, var.upper)


    def _enforce_coupling(self):
        """
        Set the coupled variables in each group to the value in the group root
        """
        for body in self.bodies:
            if body.group_root:
                for body2 in self.bodies:
                    if body.group == body2.group and not body2.group_root:
                        body2.couple_variables(body)

        for scenario in self.scenarios:
            if scenario.group_root:
                for scenario2 in self.scenarios:
                    if scenario.group == scenario2.group and not scenario2.group_root:
                        scenario2.couple_variables(scenario)

    def get_variables(self):
        """
        Get the variable objects of the entire model. Coupled variables only appear one

        Returns
        -------
        var_list: list of variable objects
        """

        self._enforce_coupling()

        dv = []
        for scenario in self.scenarios:
            if scenario.group_root:
                dv.extend(scenario.active_variables())
            else:
                dv.extend(scenario.uncoupled_variables())

        for body in self.bodies:
            if body.group_root:
                dv.extend(body.active_variables())
            else:
                dv.extend(body.uncoupled_variables())

        return dv

    def set_variables(self, dv, scale=False):
        """
        Set the variable values of the entire model given a list of values
        in the same order as get_variables

        Parameters
        ----------
        dv: list of float, complex, or Variable objects
            The variable values in the same order as :func:`get_variables` returns them
        scale: bool
            If scale, then the model variables will be set to dv * scaling stored in the variable type
        """

        if type(dv) == np.ndarray:
            dv = dv.tolist()

        var_list = self.get_variables()

        for ivar, var in enumerate(var_list):
            if type(dv[0]) == Variable:
                var.value = dv[ivar].value * var.scaling if scale else dv[ivar].value
            else:
                value = dv.pop(0)
                var.value = value * var.scaling if scale else value

        # Make sure the coupled variables get set too
        self._enforce_coupling()

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
        funcs = self.get_functions()
        gradients = []

        for n, func in enumerate(funcs):
            func_list = []
            for scenario in self.scenarios:
                if scenario.group_root:
                    func_list.extend(scenario.active_derivatives(n))
                else:
                    func_list.extend(scenario.uncoupled_derivatives(n))

            for body in self.bodies:
                if body.group_root:
                    func_list.extend(body.active_derivatives(n))
                else:
                    func_list.extend(body.uncoupled_derivatives(n))
            gradients.append(func_list)

        return gradients

    def enforce_coupling_derivatives(self):
        """
        **[driver call]** Sum the coupled variable derivatives in each group.
        """
        for body in self.bodies:
            if body.group_root:
                # Sum and store in the group root
                for body2 in self.bodies:
                    if body.group == body2.group and not body2.group_root:
                        body.add_coupled_derivatives(body2)
                # now pass the totals back to the other bodies
                for body2 in self.bodies:
                    if body.group == body2.group and not body2.group_root:
                        body2.set_coupled_derivatives(body)

        for scenario in self.scenarios:
            if scenario.group_root:
                # Sum and store in the group root
                for scenario2 in self.scenarios:
                    if scenario.group == scenario2.group and not scenario2.group_root:
                        scenario.add_coupled_derivatives(scenario2)
                # now pass the totals back to the other bodies
                for scenario2 in self.scenarios:
                    if scenario.group == scenario2.group and not scenario2.group_root:
                        scenario2.set_coupled_derivatives(scenario)

    def write_sensitivity_file(self, comm, filename, discipline='aero', root=0):
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
            # Number of functionals
            data = '{}\n'.format(len(funcs))

            for n, func in enumerate(funcs):
                # Print the function name
                data += '{}\n'.format(func.name)

                # Print the function value
                data += '{}\n'.format(func.value)

                # Print the number of coordinates
                data += '{}\n'.format(count)

                for ibody in range(len(self.bodies)):
                    id = ids[ibody]
                    deriv = derivs[ibody]

                    for i in range(len(id)):
                        data += '{} {} {} {}\n'.format(
                            id[i],
                            deriv[3 * i, n],
                            deriv[3 * i + 1, n],
                            deriv[3 * i + 2, n])

            with open(filename, "w") as fp:
                fp.write(data)

        return

