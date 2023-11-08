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

import numpy as np, os, importlib
from .variable import Variable
from .function import CompositeFunction

# optional tacs import for caps2tacs
tacs_loader = importlib.util.find_spec("tacs")
caps_loader = importlib.util.find_spec("pyCAPS")
if caps_loader is not None:
    from funtofem.interface.caps2fun import Fun3dModel

if tacs_loader is not None and caps_loader is not None:
    from tacs import caps2tacs


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
        self.composite_functions = []

        self._struct_model = None
        self._flow_model = None

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

        # auto registration of variables to discipline models
        self._send_struct_variables(body)
        self._send_flow_variables(body)

        # end of tacs model auto registration of vars section

        self.bodies.append(body)

    def add_composite_function(self, composite_function):
        """
        Add a composite function to the existing list of composite functions in the model.
        Need all variables to be setup before making any composite functions...
        """
        composite_function.setup_derivative_dict(self.get_variables())
        self.composite_functions.append(composite_function)
        return

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

        # auto registration of variables to discipline models
        self._send_struct_variables(scenario)
        self._send_flow_variables(scenario)

        self.scenarios.append(scenario)

    def _send_struct_variables(self, base):
        """send variables to self.structural usually the TacsModel"""
        # if tacs loader and tacs model exist then create thickness variables and register to tacs model
        # in the case of defining shell properties
        if tacs_loader is None or caps_loader is None:
            return

        if isinstance(self.structural, caps2tacs.TacsModel):
            struct_variables = []
            shape_variables = []
            if "structural" in base.variables:
                struct_variables = base.variables["structural"]
            if "shape" in base.variables:
                shape_variables = base.variables["shape"]

            for var in struct_variables:
                # check if matching shell property exists
                matching_prop = False
                for prop in self.structural.tacs_aim._properties:
                    if prop.caps_group == var.name:
                        matching_prop = True
                        break

                matching_dv = False
                for dv in self.structural.thickness_variables:
                    if dv.name == var.name:
                        matching_dv = True
                        break

                if matching_prop and not (matching_dv):
                    caps2tacs.ThicknessVariable(
                        caps_group=var.name, value=var.value, name=var.name
                    ).register_to(self.structural)

            esp_caps_despmtrs = None
            comm = self.structural.comm
            if self.structural.root_proc:
                esp_caps_despmtrs = list(self.structural.geometry.despmtr.keys())
            esp_caps_despmtrs = comm.bcast(esp_caps_despmtrs, root=0)

            for var in shape_variables:
                matching_despmtr = False
                for despmtr in esp_caps_despmtrs:
                    if var.name == despmtr:
                        matching_despmtr = True
                        break

                matching_shape_dv = False
                for shape_var in self.structural.shape_variables:
                    if var.name == shape_var.name:
                        matching_shape_dv = True
                        break

                # create a matching shape variable in caps2tacs
                if matching_despmtr and not matching_shape_dv:
                    caps2tacs.ShapeVariable(name=var.name, value=var.value).register_to(
                        self.structural
                    )
        return

    def _send_flow_variables(self, base):
        """send variables to self.flow usually the Fun3dModel"""
        if caps_loader is None:
            return

        if isinstance(self.flow, Fun3dModel):
            shape_variables = []
            aero_variables = []
            if "shape" in base.variables:
                shape_variables = base.variables["shape"]
            if "aerodynamic" in base.variables:
                aero_variables = base.variables["aerodynamic"]

            esp_caps_despmtrs = None
            comm = self.flow.comm
            if self.flow.root_proc:
                esp_caps_despmtrs = list(self.flow.geometry.despmtr.keys())
            esp_caps_despmtrs = comm.bcast(esp_caps_despmtrs, root=0)

            active_shape_vars = []
            active_aero_vars = []

            # add shape variable names to varnames
            for var in shape_variables:
                for despmtr in esp_caps_despmtrs:
                    if var.name == despmtr:
                        active_shape_vars.append(var)
                        break

            # NOTE : we've decided to only use aero variables in FUN3D Fortran
            # not in Fun3dAim for now...

            # add aerodynamic variable names to varnames
            # for var in aero_variables:
            #     if var.active:
            #         active_aero_vars.append(var)

            # input the design parameters into the Fun3dModel and Fun3dAim
            self.flow.set_variables(active_shape_vars, active_aero_vars)
        return

    def get_variables(self, names=None):
        """
        Get all the coupled and uncoupled variable objects for the entire model.
        Coupled variables only appear once.

        Parameters
        ----------
        names: str or List[str]
            one variable name or a list of variable names

        Returns
        -------
        var_list: list of variable objects
        """

        if names is None:
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

        elif isinstance(names, str):
            # get the one variable associated with that name
            for var in self.get_variables():
                if var.name == names:
                    return var

        elif isinstance(names, list) and isinstance(names[0], str):
            varlist = []
            for var in self.get_variables():
                if var.name in names:
                    varlist.append(var)
            return varlist
        else:
            raise AssertionError(
                f"names input object to get_variables is wrong type {type(names)}"
            )
        # didn't find anything so return None, should error stuff out
        return None

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

    def get_functions(self, optim=False, all=False):
        """
        Get all the functions in the model

        Parameters
        ----------
        optim: bool
            get functions for optimization when True otherwise just analysis functions within drivers
        all: bool
            get all functions analysis or composite for unittests

        Returns
        -------
        functions: list of function objects
            list of all the functions in the model ordered by the scenarios
        """

        # add in analysis functions
        functions = []
        for scenario in self.scenarios:
            if optim:
                functions.extend([func for func in scenario.functions if func.optim])
            else:
                functions.extend(scenario.functions)

        if optim or all:
            # for optimization also include composite functions
            functions += self.composite_functions

            # filter out only functions with optim True flag, can be set with func.optimize()
            functions = [func for func in functions if func.optim or all]

        return functions

    def get_function_gradients(self, optim=False, all=False):
        """
        Get the function gradients for all the functions in the model

        Parameters
        ----------
        optim: bool
            get functions for optimization when True otherwise just analysis functions within drivers
        all: bool
            get all functions analysis or composite for unittests

        Returns
        -------
        gradients: list of list of floats
            derivative values
            1st index = function number in the same order as get_functions
            2st index = variable number in the same order as get_variables
        """

        functions = self.get_functions(optim=optim, all=all)
        variables = self.get_variables()

        gradients = []
        for func in functions:
            grad = []
            for var in variables:
                grad.append(func.get_gradient_component(var))
            gradients.append(grad)

        return gradients

    def evaluate_composite_functions(self, compute_grad=True):
        """
        compute the values and gradients of composite functions
        """
        # check that all have appropriate derivative dictionaries
        for composite_func in self.composite_functions:
            composite_func.check_derivative_dict(self.get_variables())

        # reset each composite function first
        for composite_func in self.composite_functions:
            composite_func.reset()

        # compute values and gradients of the composite functions
        for composite_func in self.composite_functions:
            composite_func.evaluate()
            if compute_grad:
                composite_func.evaluate_gradient()
        return

    def read_aero_loads(self, comm, filename, root=0):
        """
        Read the aerodynamic loads file for the OnewayStructDriver.

        This file contains the following information:

        # of Bodies, # of Scenarios

        # aero mesh section
        Body_mesh name
        for node in surface_nodes:
            node, xpos, ypos, zpos

        # aero loads section
        for each body and scenario:
            Scenario name
            Body name
            for node in surface_nodes:
                id hflux xload yload zload

        Parameters
        ----------
        comm: MPI communicator
            Global communicator across all FUNtoFEM processors
        filename: str
            The name of the file to be generated
        root: int
            The rank of the processor that will write the file
        """
        loads_data = None
        mesh_data = None

        if comm.rank == root:
            scenario_data = None
            loads_data = {}
            mesh_data = {}

            with open(filename, "r") as fp:
                for line in fp.readlines():
                    entries = line.strip().split(" ")
                    # print("==> entries: ", entries)
                    if len(entries) == 2:
                        assert int(entries[1]) == len(self.scenarios)
                        assert int(entries[0]) == len(self.bodies)

                    elif len(entries) == 3 and entries[0] == "scenario":
                        matching_scenario = False
                        for scenario in self.scenarios:
                            if str(scenario.name).strip() == str(entries[2]).strip():
                                matching_scenario = True
                                break
                        assert matching_scenario
                        if scenario_data is not None:
                            loads_data[scenario.id] = scenario_data
                        scenario_data = []
                    elif len(entries) == 4 and entries[0] == "body_mesh":
                        body_name = entries[2]
                        mesh_data[body_name] = {"aeroID": [], "aeroX": []}
                    elif len(entries) == 4 and entries[0] != "body":
                        mesh_data[body_name]["aeroID"] += [entries[0]]
                        mesh_data[body_name]["aeroX"] += entries[1:4]

                    elif len(entries) == 5:
                        entry = {
                            "bodyName": body_name,
                            "aeroID": entries[0],
                            "load": entries[1:4],
                            "hflux": entries[4],
                        }
                        scenario_data.append(entry)

            loads_data[scenario.id] = scenario_data

        loads_data = comm.bcast(loads_data, root=root)
        mesh_data = comm.bcast(mesh_data, root=root)

        # initialize the mesh data
        for body in self.bodies:
            global_aero_x = np.array(mesh_data[body.name]["aeroX"])
            global_aero_ids = np.array(mesh_data[body.name]["aeroID"])

            body_ind = np.array([_ for _ in range(len(global_aero_ids))])
            if comm.rank == root:
                split_body_ind = np.array_split(body_ind, comm.Get_size())
            else:
                split_body_ind = None

            local_body_ind = comm.scatter(split_body_ind, root=root)

            local_aero_ids = global_aero_ids[local_body_ind]

            aero_x_ind = (
                [3 * i for i in local_body_ind]
                + [3 * i + 1 for i in local_body_ind]
                + [3 * i + 2 for i in local_body_ind]
            )
            aero_x_ind = sorted(aero_x_ind)

            local_aero_x = list(global_aero_x[aero_x_ind])

            body.initialize_aero_nodes(local_aero_x, local_aero_ids)

        # return the loads data
        return loads_data

    def write_struct_loads(self, comm, filename, root=0):
        """
        Write the struct loads file for the OnewayStructDriver.

        This file contains the following information:

        # of Bodies, # of Scenarios

        # struct loads section
        for each body and scenario:
            Scenario name
            Body name
            for node in surface_nodes:
                id hflux xload yload zload

        Parameters
        ----------
        comm: MPI communicator
            Global communicator across all FUNtoFEM processors
        filename: str
            The name of the file to be generated
        root: int
            The rank of the processor that will write the file
        """
        if comm.rank == root:
            data = ""
            # Specify the number of scenarios in file
            data += f"{len(self.bodies)} {len(self.scenarios)} \n"

        if comm.rank == root:
            data += f"structloads \n"

        for scenario in self.scenarios:
            if comm.rank == root:
                data += f"scenario {scenario.id} {scenario.name} \n"

            for body in self.bodies:
                id, hflux, load = body._collect_struct_loads(comm, scenario, root=root)

                if comm.rank == root:
                    data += f"body {body.id} {body.name} {body.aero_nnodes} \n"
                    for i in range(len(id)):
                        data += "{} {} {} {} {} \n".format(
                            int(id[i]),
                            load[3 * i + 0].real,
                            load[3 * i + 1].real,
                            load[3 * i + 2].real,
                            float(hflux[i].real),
                        )

                    with open(filename, "w") as fp:
                        fp.write(data)
        return

    def write_aero_loads(self, comm, filename, root=0):
        """
        Write the aerodynamic loads file for the OnewayStructDriver.

        This file contains the following information:

        # of Bodies, # of Scenarios

        # aero mesh section
        Body_mesh name
        for node in surface_nodes:
            node, xpos, ypos, zpos

        # aero loads section
        for each body and scenario:
            Scenario name
            Body name
            for node in surface_nodes:
                id hflux xload yload zload

        Parameters
        ----------
        comm: MPI communicator
            Global communicator across all FUNtoFEM processors
        filename: str
            The name of the file to be generated
        root: int
            The rank of the processor that will write the file
        """
        if comm.rank == root:
            data = ""
            # Specify the number of scenarios in file
            data += f"{len(self.bodies)} {len(self.scenarios)} \n"
            data += "aeromesh" + "\n"

        for body in self.bodies:
            if comm.rank == root:
                data += f"body_mesh {body.id} {body.name} {body.aero_nnodes} \n"

            id, aeroX = body._collect_aero_mesh(comm, root=root)

            if comm.rank == root:
                for i in range(len(id)):
                    data += "{} {} {} {} \n".format(
                        int(id[i]),
                        aeroX[3 * i + 0].real,
                        aeroX[3 * i + 1].real,
                        aeroX[3 * i + 2].real,
                    )
        if comm.rank == root:
            data += f"aeroloads \n"

        for scenario in self.scenarios:
            if comm.rank == root:
                data += f"scenario {scenario.id} {scenario.name} \n"

            for body in self.bodies:
                id, hflux, load = body._collect_aero_loads(comm, scenario, root=root)

                if comm.rank == root:
                    data += f"body {body.id} {body.name} {body.aero_nnodes} \n"
                    for i in range(len(id)):
                        data += "{} {} {} {} {} \n".format(
                            int(id[i]),
                            load[3 * i + 0].real,
                            load[3 * i + 1].real,
                            load[3 * i + 2].real,
                            float(hflux[i].real),
                        )

                    with open(filename, "w") as fp:
                        fp.write(data)
        return

    def write_sensitivity_file(
        self, comm, filename, discipline="aerodynamic", root=0, write_dvs: bool = True
    ):
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
        write_dvs: bool
            whether to write the design variables for this discipline
        """

        funcs = self.get_functions()

        count = 0
        ids = []
        derivs = []
        for body in self.bodies:
            id, deriv = body.collect_coordinate_derivatives(
                comm, discipline, self.scenarios, root=root
            )
            count += len(id)
            ids.append(id)
            derivs.append(deriv)

        if comm.rank == root:
            variables = self.get_variables()
            discpline_vars = []
            if write_dvs:  # flag for registering dvs that will later get written out
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
                data += "{}\n".format(func.full_name)

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

    def read_design_variables_file(self, comm, filename, root=0):
        """
        Read the design variables file funtofem.in

        This file contains the following information:

        Discipline
        Var_name Var_value

        Parameters
        ----------
        comm: MPI communicator
            Global communicator across all FUNtoFEM processors
        filename: str
            The name of the file to be read in / filepath
        root: int
            The rank of the processor that will write the file
        """

        variables_dict = None  # this will be broadcast to other processors
        if comm.rank == root:  # read the file in on the root processor
            variables_dict = {}

            hdl = open(filename, "r")
            lines = hdl.readlines()
            hdl.close()

            for line in lines:
                chunks = line.split(" ")
                if len(chunks) == 3:
                    var_name = chunks[1]
                    var_value = chunks[2]

                    # only real numbers are read in from the file
                    variables_dict[var_name] = float(var_value)

        # broadcast the dictionary to the root processor
        variables_dict = comm.bcast(variables_dict, root=root)

        # update the variable values on each processor
        for var in self.get_variables():
            if var.full_name in variables_dict:
                var.value = variables_dict[var.full_name]

        return

    def write_design_variables_file(self, comm, filename, root=0):
        """
        Write the design variables file funtofem.in
        This file contains the following information:
        Discipline
        Var_name Var_value
        Parameters
        ----------
        comm: MPI communicator
            Global communicator across all FUNtoFEM processors
        filename: str
            The name of the file to be generated / filepath
        root: int
            The rank of the processor that will write the file
        """

        # get the variable values from the root processor
        if comm.rank == root:
            # get the disciplines and variables
            disciplines = []
            variables = {}
            scenarios_and_bodies = self.scenarios + self.bodies
            for base in scenarios_and_bodies:
                for discipline in base.variables:
                    if not (discipline in disciplines):
                        disciplines.append(discipline)
                        variables[discipline] = []
                    for var in base.variables[discipline]:
                        if var.active:
                            variables[discipline].append(var)

            # get the file hdl
            hdl = open(filename, "w")

            # write the variables file
            for discipline in variables:
                if (
                    len(variables[discipline]) == 0
                ):  # skip this discipline if empty, aka rigid_motion
                    continue
                hdl.write(f"Discipline {discipline}\n")

                # write each variable from that discipline
                for var in variables[discipline]:
                    # only real numbers written to this file
                    hdl.write(f"\tvar {var.full_name} {var.value.real}\n")

            hdl.close()

        return

    def read_functions_file(self, comm, filename, root=0):
        """
        Read the functions variables file funtofem.out

        This file contains the following information:

        nFunctions
        func_name, func_value

        Parameters
        ----------
        comm: MPI communicator
            Global communicator across all FUNtoFEM processors
        filename: str
            The name of the file to be read in / filepath
        root: int
            The rank of the processor that will write the file
        """

        functions_dict = None
        if comm.rank == root:  # read the file in on the root processor
            functions_dict = {}

            hdl = open(filename, "r")
            lines = hdl.readlines()
            hdl.close()

            for line in lines:
                chunks = line.split(" ")
                if len(chunks) == 2:
                    func_name = chunks[0]
                    func_value = chunks[1]

                    # only real numbers are read in from the file
                    functions_dict[func_name] = float(func_value)

        # broadcast the dictionary to the root processor
        functions_dict = comm.bcast(functions_dict, root=root)

        # update the variable values on each processor
        for func in self.get_functions():
            if func.name in functions_dict:
                func.value = functions_dict[func.name]

        return

    def write_functions_file(self, comm, filename, root=0):
        """
        Write the functions file funtofem.out

        This file contains the following information:

        Number of functionals

        Functional name, value

        Parameters
        ----------
        comm: MPI communicator
            Global communicator across all FUNtoFEM processors
        filename: str
            The name of the file to be generated
        root: int
            The rank of the processor that will write the file
        """

        funcs = self.get_functions()
        # also add composite functions at the end
        funcs += self.composite_functions

        if comm.rank == root:
            # Write out the number of functionals and number of design variables
            data = "{}\n".format(len(funcs))

            for n, func in enumerate(funcs):
                # Print the function name
                data += "{}\n".format(func.full_name)

                # Print the function value
                data += "{}\n".format(func.value.real)

            with open(filename, "w") as fp:
                fp.write(data)

    @property
    def structural(self):
        """structural discipline submodel such as TacsModel"""
        return self._struct_model

    @structural.setter
    def structural(self, structural_model):
        self._struct_model = structural_model

    @property
    def flow(self):
        """flow discipline submodel such as Fun3dModel"""
        return self._flow_model

    @flow.setter
    def flow(self, flow_model):
        self._flow_model = flow_model

    def print_summary(
        self, print_level=0, print_model_details=True, ignore_rigid=False
    ):
        """
        Print out a summary of the assembled model for inspection

        Parameters
        ----------
        print_level: int
            how much detail to print in the summary. Print level < 0 does not print all the variables
        """

        print("==========================================================")
        print("||                FUNtoFEM Model Summary                ||")
        print("==========================================================")
        print(self)

        if print_model_details:
            self._print_functions()
            self._print_variables()

        print("\n------------------")
        print("| Bodies Summary |")
        print("------------------")
        for body in self.bodies:
            print(body)
            for vartype in body.variables:
                print("\n    Variable type:", vartype)
                print(
                    "      Number of",
                    vartype,
                    "variables:",
                    len(body.variables[vartype]),
                )
                if (vartype == "rigid_motion") and ignore_rigid:
                    print("      Ignoring rigid_motion vartype list.")
                else:
                    if print_level >= 0:
                        body._print_variables(vartype)

        print(" ")
        print("--------------------")
        print("| Scenario Summary |")
        print("--------------------")
        for scenario in self.scenarios:
            print(scenario)
            scenario._print_functions()

            for vartype in scenario.variables:
                print("    Variable type:", vartype)
                print(
                    "      Number of",
                    vartype,
                    "variables:",
                    len(scenario.variables[vartype]),
                )
                if print_level >= 0:
                    scenario._print_variables(vartype)

        return

    def _print_functions(self):
        model_functions = self.get_functions(all=True)
        print(
            "     ------------------------------------------------------------------------------------"
        )
        print(
            "     | Function \t| Analysis Type\t| Comp. Adjoint\t| Time Range\t| Averaging\t|"
        )
        print(
            "     ------------------------------------------------------------------------------------"
        )
        for func in model_functions:
            if isinstance(func, CompositeFunction):
                analysis_type = func.analysis_type
                adjoint = "N/A"
                start = "N/A"
                stop = "N/A"
                averaging = "N/A"
            else:
                analysis_type = func.analysis_type
                adjoint = func.adjoint
                start = func.start
                stop = func.stop
                averaging = func.averaging
            if len(func.name) >= 8:
                print(
                    "     | ",
                    func.name,
                    "\t| ",
                    analysis_type,
                    "\t| ",
                    adjoint,
                    "\t| [",
                    start,
                    ",",
                    stop,
                    "] \t| ",
                    averaging,
                    "\t|",
                )
            else:
                print(
                    "     | ",
                    func.name,
                    "\t\t| ",
                    analysis_type,
                    "\t| ",
                    adjoint,
                    "\t| [",
                    start,
                    ",",
                    stop,
                    "] \t| ",
                    averaging,
                    "\t|",
                )
        print(
            "     ------------------------------------------------------------------------------------"
        )

        return

    def _print_variables(self):
        model_variables = self.get_variables()
        print(
            "     ------------------------------------------------------------------------------------------------------------"
        )
        print(
            "     | Variable\t\t| Var. ID\t| Value \t| Bounds\t\t| Active\t| Coupled\t|"
        )
        print(
            "     ------------------------------------------------------------------------------------------------------------"
        )
        for var in model_variables:
            print(
                "     | ",
                var.name,
                "\t\t|",
                var.id,
                "\t\t|",
                var.value,
                " \t| [",
                var.lower,
                ",",
                var.upper,
                "] \t|",
                var.active,
                " \t|",
                var.coupled,
                "\t|",
            )

        print(
            "     ------------------------------------------------------------------------------------------------------------"
        )

        return

    def __str__(self):
        line1 = f"Model (<Name>): {self.name}"
        line2 = f"  Number of bodies: {len(self.bodies)}"
        line3 = f"  Number of scenarios: {len(self.scenarios)}"

        output = (line1, line2, line3)

        return "\n".join(output)
