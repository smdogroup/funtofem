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
import sys

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
                chunks = var.name.split("-")
                chunk_thick_var = len(chunks) == 2 and chunks[1] == "T"
                for prop in self.structural.tacs_aim._properties:
                    if prop.caps_group == var.name:
                        matching_prop = True
                        break
                    elif chunk_thick_var and chunks[0] == prop.caps_group:
                        matching_prop = True
                        break

                matching_dv = False
                for dv in self.structural.thickness_variables:
                    if dv.name == var.name:
                        matching_dv = True
                        break
                    elif chunk_thick_var and chunks[0] == dv.name:
                        matching_dv = True
                        break

                if matching_prop and not (matching_dv):
                    if len(chunks) == 2:
                        var_name = chunks[0]
                    else:
                        var_name = var.name
                    caps2tacs.ThicknessVariable(
                        caps_group=var_name,
                        value=var.value,
                        name=var.name,
                        active=var.active,
                    ).register_to(self.structural)

            esp_caps_despmtrs = None
            comm = self.structural.comm
            if self.structural.root_proc:
                esp_caps_despmtrs = list(self.structural.geometry.despmtr.keys())
            esp_caps_despmtrs = comm.bcast(
                esp_caps_despmtrs, root=self.structural.root_proc_ind
            )

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
                    caps2tacs.ShapeVariable(
                        name=var.name, value=var.value, active=var.active
                    ).register_to(self.structural)

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
            esp_caps_despmtrs = comm.bcast(esp_caps_despmtrs, root=self.flow.root)

            active_shape_vars = []
            active_aero_vars = []

            # add shape variable names to varnames
            for var in shape_variables:
                if not (var.active):
                    continue
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

    def get_variables(self, names=None, all=False, optim=False):
        """
        Get all the coupled and uncoupled variable objects for the entire model.
        Coupled variables only appear once.

        Parameters
        ----------
        names: str or List[str]
            one variable name or a list of variable names
        all: bool
            Flag to include inactive variables.

        Returns
        -------
        var_list: list of variable objects
        """

        if optim:  # return all active and non-state variables
            return [var for var in self.get_variables() if not (var.state)]

        if names is None:
            dv = []
            for scenario in self.scenarios:
                if scenario.group_root:
                    dv.extend(scenario.get_active_variables())
                else:
                    dv.extend(scenario.get_uncoupled_variables())
                if all:
                    dv.extend(scenario.get_inactive_variables())

            for body in self.bodies:
                if body.group_root:
                    dv.extend(body.get_active_variables())
                else:
                    dv.extend(body.get_uncoupled_variables())
                if all:
                    dv.extend(body.get_inactive_variables())
            return dv

        elif isinstance(names, str):
            # get the one variable associated with that name
            for var in self.get_variables(all=all):
                if var.name == names:
                    return var

        elif isinstance(names, list) and isinstance(names[0], str):
            varlist = []
            for var in self.get_variables(all=all):
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

    def get_functions(self, optim=False, all=False, include_vars_only=True):
        """
        Get all the functions in the model

        Parameters
        ----------
        optim: bool
            get functions for optimization when True otherwise just analysis functions within drivers
        all: bool
            get all functions analysis or composite for unittests
        include_vars_only: bool
            whether to include composite functions that have variables only or not
            changed default to True so it doesn't mess up sparse gradient functionality for now..

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
            composite_functions = self.composite_functions
            if not include_vars_only:
                composite_functions = [
                    cfunc for cfunc in composite_functions if not (cfunc.vars_only)
                ]
            functions += composite_functions

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

    def clear_vars_only_composite_functions(self):
        """
        clears all vars only composite functions to save memory on large wing optimizations
        after registering them to the optimizer [their sparse gradients depend only on design variables and are not needed after this point.]
        In a large HSCT wing optimization case, there were 13k adjacency constraints which took up a considerable amount of space on memory..
        """
        import gc
        import sys

        # get vars only composite functions first to delete later
        vars_only_cfuncs = [
            cfunc for cfunc in self.composite_functions if cfunc.vars_only
        ]
        if len(vars_only_cfuncs) == 0:
            return
        nvars_only_cfuncs = [
            cfunc for cfunc in self.composite_functions if not (cfunc.vars_only)
        ]
        self.composite_functions = nvars_only_cfuncs
        first_vars_only_cfunc = vars_only_cfuncs[0]
        ref_count1 = sys.getrefcount(first_vars_only_cfunc)
        del vars_only_cfuncs
        gc.collect()  # garbage collection..
        # # check reference count on one of the vars only cfunc
        # # will only delete if no references available
        ref_count2 = sys.getrefcount(first_vars_only_cfunc)

        print(
            f"clear_vars_only_cfuncs: ref count1 {ref_count1} ref count2 {ref_count2}"
        )
        print(f"need to have 2 for the object to be truly deleted")
        return

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

    def _read_aero_loads(self, comm, filename, root=0):
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
                id xload yload zload hflux

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
                        mesh_data[body_name]["aeroID"] += [int(entries[0])]
                        mesh_data[body_name]["aeroX"] += entries[1:4]

                    elif len(entries) == 5:
                        entry = {
                            "bodyName": body_name,
                            "aeroID": int(entries[0]),
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

    def _read_struct_loads(self, comm, filename, root=0):
        """
        Read the structural loads file from file.

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
                id xload yload zload hflux

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

        if comm.rank == root:
            scenario_data = None
            loads_data = {}

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
                    elif len(entries) == 4 and entries[0] == "body":
                        body_name = entries[2]

                    elif len(entries) == 5:
                        entry = {
                            "bodyName": body_name,
                            "structID": int(entries[0]),
                            "load": entries[1:4],
                            "hflux": entries[4],
                        }
                        scenario_data.append(entry)

            loads_data[scenario.id] = scenario_data

        loads_data = comm.bcast(loads_data, root=root)

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
                id xload yload zload hflux

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
                    data += f"body {body.id} {body.name} {body.struct_nnodes} \n"
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

    @classmethod
    def _get_loads_filename(cls, prefix, itime: int, suffix=".txt", padding=3):
        """routine to get default padded 0 aero loads filenames"""
        return prefix + "_" + f"%0{padding}d" % itime + suffix

    def get_loads_files(self, prefix, suffix=".txt"):
        """get a list of the loads files for this unsteady scenario"""
        max_steps = max([scenario.steps for scenario in self.scenarios])
        return [
            FUNtoFEMmodel._get_loads_filename(prefix, itime=itime, suffix=suffix)
            for itime in range(max_steps)
        ]

    def write_aero_loads(self, comm, filename, itime=0, root=0):
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
                id xload yload zload hflux

        Parameters
        ----------
        comm: MPI communicator
            Global communicator across all FUNtoFEM processors
        filename: str
            The name of the file to be generated
        itime: int
            The time step of the loads to write  out (irrelevant if steady)
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
                id, hflux, load = body._collect_aero_loads(
                    comm, scenario, itime=itime, root=root
                )

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

    def write_unsteady_aero_loads(self, comm, prefix, suffix=".txt", root=0):
        """
        Write the aerodynamic loads files for unsteady scenarios for the OnewayStructDriver.

        Parameters
        ----------
        comm: MPI communicator
            Global communicator across all FUNtoFEM processors
        prefix: str
            file prefix
        root: int
            The rank of the processor that will write the file
        """
        loads_files = self.get_loads_files(prefix, suffix)
        for itime, load_file in enumerate(loads_files):
            self.write_aero_loads(comm, load_file, itime=itime, root=root)
        return loads_files

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
                    if discipline == var.analysis_type and var.active:
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

        IMPORTANT: To correctly set inactive variables, make sure to call (e.g.) tacs_aim.setup_aim(),
        then read_design_variables_file, then tacs_aim.pre_analysis.

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
        for var in self.get_variables(all=True):
            if var.full_name in variables_dict:
                var.value = variables_dict[var.full_name]

        if self.structural is not None:
            input_dict = {var.name: var.value for var in self.get_variables(all=True)}
            self.structural.update_design(input_dict)

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

    def print_memory_size(self, comm, root: int, starting_message=""):
        """print the memory of the funtofem model and its various constituents"""

        def print_list(m_list, name):
            number = len(m_list)
            size = sys.getsizeof(m_list) + sum([sys.getsizeof(item) for item in m_list])
            print(f"\t{number} {name} have size {size}")
            return size

        if comm.rank == root:
            print(f"\nFuntofem model size {starting_message}")

            # size of the whole funtofem model
            model_size = sys.getsizeof(self)
            print(f"\tmodel_size = {model_size}")

            # size of scenarios list
            scenario_size = print_list(
                m_list=self.scenarios,
                name="scenarios",
            )

            # size of bodies list
            body_size = print_list(
                m_list=self.bodies,
                name="bodies",
            )

            # size of regular functions
            functions = [
                func for scenario in self.scenarios for func in scenario.functions
            ]
            function_size = print_list(
                m_list=functions,
                name="functions",
            )

            # size of variables
            variables = [
                var for scenario in self.scenarios for var in scenario.variables
            ] + [var for body in self.bodies for var in body.variables]
            variable_size = print_list(
                m_list=variables,
                name="variables",
            )

            # size of composite funtions, vars only
            vars_only_composite_functions = [
                cfunc for cfunc in self.composite_functions if cfunc.vars_only
            ]
            vars_only_cfunc_size = print_list(
                m_list=vars_only_composite_functions,
                name="vars only composite functions",
            )

            # size of composite funtions, not vars only
            not_vars_only_composite_functions = [
                cfunc for cfunc in self.composite_functions if not (cfunc.vars_only)
            ]
            nvars_only_cfunc_size = print_list(
                m_list=not_vars_only_composite_functions,
                name="not vars only composite functions",
            )

            total_size = (
                model_size
                + scenario_size
                + body_size
                + function_size
                + variable_size
                + vars_only_cfunc_size
                + nvars_only_cfunc_size
            )
            print(f"\ttotal size = {total_size}")

            print("\n", flush=True)

        return

    def read_functions_file(self, comm, filename, root=0, **kwargs):
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
        gradients_dict = None
        if comm.rank == root:  # read the file in on the root processor
            functions_dict = {}
            gradients_dict = {}

            hdl = open(filename, "r")
            lines = hdl.readlines()
            hdl.close()

            for line in lines:
                chunks = line.split(" ")
                if len(chunks) == 3:  # func values
                    func_name = chunks[1]
                    func_value = chunks[2].strip()

                    # only real numbers are read in from the file
                    functions_dict[func_name] = float(func_value)
                    gradients_dict[func_name] = {}
                if len(chunks) == 2:  # derivative
                    var_name = chunks[0].strip()
                    derivative = chunks[1].strip()

                    # only real numbers are read in from the file
                    gradients_dict[func_name][var_name] = float(derivative)

        # broadcast the dictionary to the root processor
        functions_dict = comm.bcast(functions_dict, root=root)
        gradients_dict = comm.bcast(gradients_dict, root=root)

        # update the variable values on each processor
        # only updates the analysis functions
        for func in self.get_functions(kwargs):
            if func.full_name in functions_dict:
                func.value = functions_dict[func.full_name]
            for var in self.get_variables():
                c_gradients = gradients_dict[func.full_name]
                if var.full_name in c_gradients:
                    func.derivatives[var] = c_gradients[var.full_name]

        return

    def write_functions_file(
        self, comm, filename, root=0, full_precision=True, **kwargs
    ):
        """
        Write the functions file funtofem.out

        This file contains the following information:

        Number of functionals, number of variables

        Functional name, value
        d(func_name)/d(var_name), value

        Parameters
        ----------
        comm: MPI communicator
            Global communicator across all FUNtoFEM processors
        filename: str
            The name of the file to be generated
        root: int
            The rank of the processor that will write the file
        """

        funcs = self.get_functions(kwargs)
        variables = self.get_variables()

        if comm.rank == root:
            # Write out the number of functionals and number of design variables
            data = "{}\n".format(len(funcs))

            for n, func in enumerate(funcs):
                # Print the function name
                func_value = func.value.real if func.value is not None else None
                if full_precision:
                    data += f"func {func.full_name} {func_value}\n"
                else:
                    data += f"func {func.full_name} {func_value:.5e}\n"

                for var in variables:
                    derivative = float(func.derivatives[var])
                    if full_precision:
                        data += f"\t{var.full_name} {derivative}\n"
                    else:
                        data += f"\t{var.full_name} {derivative:.5e}\n"

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

    def save_forward_states(self, comm, scenario):
        """save all of the funtofem elastic/thermal states for AE, AT, or ATE coupling (to be reloaded)"""

        if not scenario.steady:
            if comm.rank == 0:
                print(
                    f"Funtofem : non-fatal warning, you tried to reload the funtofem states, but scenario {scenario.name} is unsteady."
                )
                return

        # make a workdirectory for saving the funtofem states
        work_dir = os.path.join(os.getcwd(), self.name)
        if comm.rank == 0 and not os.path.exists(work_dir):
            os.mkdir(work_dir)

        comm.Barrier()

        # for each scenario, for each body save the states for the coupling types of that body
        # assumes you have the same proc assignment as before (don't change this arrangement, needs to be deterministic)
        # i.e. you might have a different number of struct nodes on each proc..
        for iproc in range(comm.size):
            if comm.rank == iproc:  # write out separately for each processor
                for body in self.bodies:
                    # number of struct nodes on this processor
                    ns = body.get_num_struct_nodes()
                    if ns == 0:
                        continue

                    if body.analysis_type in ["aeroelastic", "aerothermoelastic"]:

                        # pickle the struct displacements
                        struct_disps_file = os.path.join(
                            work_dir,
                            f"{scenario.name}_body-{body.name}_{iproc}_uS.npy",
                        )
                        np.save(struct_disps_file, body.struct_disps[scenario.id])

                    if body.analysis_type in ["aerothermal", "aerothermoelastic"]:

                        # pickle the struct temperatures
                        struct_temps_file = os.path.join(
                            work_dir,
                            f"{scenario.name}_body-{body.name}_{iproc}_tS.npy",
                        )
                        np.save(struct_temps_file, body.struct_temps[scenario.id])

        comm.Barrier()

        return

    def load_forward_states(self, comm, scenario):
        """load all of the funtofem elastic/thermal states for AE, AT, or ATE coupling (to be reloaded)"""

        if not scenario.steady:
            if comm.rank == 0:
                print(
                    f"Funtofem : non-fatal warning, you tried to reload the funtofem states, but scenario {scenario.name} is unsteady."
                )
                return

        # make a workdirectory for saving the funtofem states
        work_dir = os.path.join(os.getcwd(), self.name)
        if not os.path.exists(work_dir) and comm.rank == 0:
            print(
                f"Funtofem - may be first iteration => no funtofem states to reload as workdir doesn't exist."
            )

        # for each scenario, for each body reload the states for the coupling types of that body
        # assumes you have the same proc assignment as before (don't change this arrangement, needs to be deterministic)
        # i.e. you might have a different number of struct nodes on each proc..
        for iproc in range(comm.size):
            if comm.rank == iproc:  # write out separately for each processor
                for body in self.bodies:
                    # number of struct nodes on this processor
                    ns = body.get_num_struct_nodes()
                    if ns == 0:
                        continue

                    if body.analysis_type in ["aeroelastic", "aerothermoelastic"]:

                        # pickle the struct displacements
                        struct_disps_file = os.path.join(
                            work_dir,
                            f"{scenario.name}_body-{body.name}_{iproc}_uS.npy",
                        )
                        if os.path.exists(struct_disps_file):
                            _struct_disps = np.load(struct_disps_file)
                            if 3 * ns == _struct_disps.shape[0]:
                                body.struct_disps[scenario.id] = _struct_disps * 1.0
                            else:
                                print(
                                    f"Funtofem - didn't reload struct disps on proc {iproc} as size has changed."
                                )
                                print(
                                    f"\tns prev = {_struct_disps.shape[0]}, ns new = {3*ns}"
                                )
                        else:
                            print(
                                f"Funtofem - warning, struct disps file '{scenario.name}_body-{body.name}_{iproc}_uS.npy' does not exist."
                            )

                    if body.analysis_type in ["aerothermal", "aerothermoelastic"]:

                        # pickle the struct temperatures
                        struct_temps_file = os.path.join(
                            work_dir,
                            f"{scenario.name}_body-{body.name}_{iproc}_tS.npy",
                        )
                        if os.path.exists(struct_temps_file):
                            _struct_temps = np.load(struct_temps_file)
                            if ns == _struct_temps.shape[0]:
                                body.struct_temps[scenario.id] = _struct_temps * 1.0
                            else:
                                print(
                                    f"Funtofem - didn't reload struct temps on proc {iproc} as size has changed."
                                )
                        else:
                            print(
                                f"Funtofem - warning, struct disps file '{scenario.name}_body-{body.name}_{iproc}_tS.npy' does not exist."
                            )

        comm.Barrier()

        return

    def save_adjoint_states(self, comm, scenario):
        """save all of the funtofem elastic/thermal states for AE, AT, or ATE coupling (to be reloaded)"""

        if not scenario.steady:
            if comm.rank == 0:
                print(
                    f"Funtofem : non-fatal warning, you tried to reload the funtofem adjoint states, but scenario {scenario.name} is unsteady."
                )
                return

        # make a workdirectory for saving the funtofem states
        work_dir = os.path.join(os.getcwd(), self.name)
        if not os.path.exists(work_dir) and comm.rank == 0:
            os.mkdir(work_dir)

        comm.Barrier()

        nfunc = scenario.count_adjoint_functions()

        # for each scenario, for each body save the states for the coupling types of that body
        # assumes you have the same proc assignment as before (don't change this arrangement, needs to be deterministic)
        # i.e. you might have a different number of struct nodes on each proc..
        for iproc in range(comm.size):
            if comm.rank == iproc:  # write out separately for each processor
                for body in self.bodies:
                    # number of struct nodes on this processor
                    ns = body.get_num_struct_nodes()
                    if ns == 0:
                        continue

                    if body.analysis_type in ["aeroelastic", "aerothermoelastic"]:

                        # pickle the struct displacements
                        struct_loads_ajp_file = os.path.join(
                            work_dir,
                            f"{scenario.name}_body-{body.name}_{iproc}_psiS.npy",
                        )
                        # note this is saving an array of size (3*ns, nf)
                        np.save(struct_loads_ajp_file, body.struct_loads_ajp)

                    if body.analysis_type in ["aerothermal", "aerothermoelastic"]:

                        # pickle the struct temperatures
                        struct_flux_ajp_file = os.path.join(
                            work_dir,
                            f"{scenario.name}_body-{body.name}_{iproc}_psiQ.npy",
                        )
                        # note this is saving an array of size (ns, nf)
                        np.save(struct_flux_ajp_file, body.struct_flux_ajp)

        comm.Barrier()

        return

    def load_adjoint_states(self, comm, scenario):
        """load all of the funtofem elastic/thermal states for AE, AT, or ATE coupling (to be reloaded)"""

        if not scenario.steady:
            if comm.rank == 0:
                print(
                    f"Funtofem : non-fatal warning, you tried to reload the funtofem states, but scenario {scenario.name} is unsteady."
                )
                return

        # make a workdirectory for saving the funtofem states
        work_dir = os.path.join(os.getcwd(), self.name)
        if not os.path.exists(work_dir) and comm.rank == 0:
            print(
                f"Funtofem - may be first iteration => no funtofem states to reload as workdir doesn't exist."
            )

        nfunc = scenario.count_adjoint_functions()

        # for each scenario, for each body reload the states for the coupling types of that body
        # assumes you have the same proc assignment as before (don't change this arrangement, needs to be deterministic)
        # i.e. you might have a different number of struct nodes on each proc..
        for iproc in range(comm.size):
            if comm.rank == iproc:  # write out separately for each processor
                for body in self.bodies:
                    # number of struct nodes on this processor
                    ns = body.get_num_struct_nodes()
                    if ns == 0:
                        continue

                    if body.analysis_type in ["aeroelastic", "aerothermoelastic"]:

                        # pickle the struct displacements
                        struct_loads_ajp_file = os.path.join(
                            work_dir,
                            f"{scenario.name}_body-{body.name}_{iproc}_psiS.npy",
                        )
                        if os.path.exists(struct_loads_ajp_file):
                            _struct_loads_ajp = np.load(struct_loads_ajp_file)
                            if (
                                3 * ns == _struct_loads_ajp.shape[0]
                                and nfunc == _struct_loads_ajp.shape[1]
                            ):
                                body.struct_loads_ajp[:, :] = _struct_loads_ajp * 1.0
                            else:
                                print(
                                    f"Funtofem - didn't reload struct loads ajp on proc {iproc} as size has changed."
                                )
                                print(
                                    f"\tprev shape = {_struct_loads_ajp.shape}, new shape = {(3*ns,nfunc)}"
                                )
                        else:
                            print(
                                f"Funtofem - warning, struct loads ajp file '{scenario.name}_body-{body.name}_{iproc}_psiS.npy' does not exist."
                            )

                    if body.analysis_type in ["aerothermal", "aerothermoelastic"]:

                        # pickle the struct temperatures
                        struct_flux_ajp_file = os.path.join(
                            work_dir,
                            f"{scenario.name}_body-{body.name}_{iproc}_psiQ.npy",
                        )
                        if os.path.exists(struct_flux_ajp_file):
                            _struct_flux_ajp = np.load(struct_flux_ajp_file)
                            if (
                                ns == _struct_flux_ajp.shape[0]
                                and nfunc == _struct_flux_ajp.shape[1]
                            ):
                                body.struct_flux_ajp[:, :] = _struct_flux_ajp * 1.0
                            else:
                                print(
                                    f"Funtofem - didn't reload struct flux ajp on proc {iproc} as size has changed."
                                )
                                print(
                                    f"\tprev shape = {_struct_flux_ajp.shape}, new shape = {(ns,nfunc)}"
                                )
                        else:
                            print(
                                f"Funtofem - warning, struct flux ajp file '{scenario.name}_body-{body.name}_{iproc}_psiQ.npy' does not exist."
                            )

        comm.Barrier()

        return

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
            "     --------------------------------------------------------------------------------"
        )
        self._print_long("Function", width=12, indent_line=5)
        self._print_long("Analysis Type", width=15)
        self._print_long("Comp. Adjoint", width=15)
        self._print_long("Time Range", width=20)
        self._print_long("Averaging", end_line=True)

        print(
            "     --------------------------------------------------------------------------------"
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
            _time_range = " ".join(("[", str(start), ",", str(stop), "]"))
            adjoint = str(adjoint)
            self._print_long(func.name, width=12, indent_line=5)
            self._print_long(analysis_type, width=15)
            self._print_long(adjoint, width=15)
            self._print_long(_time_range, width=20)
            self._print_long(averaging, end_line=True)
        print(
            "     --------------------------------------------------------------------------------"
        )

        return

    def _print_variables(self):
        model_variables = self.get_variables()
        print(
            "     --------------------------------------------------------------------------------------"
        )
        self._print_long("Variable", width=12, indent_line=5)
        self._print_long("Var. ID", width=10)
        self._print_long("Value", width=16)
        self._print_long("Bounds", width=24)
        self._print_long("Active", width=8)
        self._print_long("Coupled", width=9, end_line=True)

        print(
            "     --------------------------------------------------------------------------------------"
        )
        for var in model_variables:
            _name = "{:s}".format(var.name)
            _id = "{: d}".format(var.id)
            _value = "{:#.8g}".format(var.value)
            _lower = "{:#.3g}".format(var.lower)
            _upper = "{:#.3g}".format(var.upper)
            _active = str(var.active)
            _coupled = str(var.coupled)
            _bounds = " ".join(("[", _lower, ",", _upper, "]"))

            self._print_long(_name, width=12, indent_line=5)
            self._print_long(_id, width=10, align="<")
            self._print_long(_value, width=16)
            self._print_long(_bounds, width=24)
            self._print_long(_active, width=8)
            self._print_long(_coupled, width=9, end_line=True)

        print(
            "     --------------------------------------------------------------------------------------"
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

    def __str__(self):
        line1 = f"Model (<Name>): {self.name}"
        line2 = f"  Number of bodies: {len(self.bodies)}"
        line3 = f"  Number of scenarios: {len(self.scenarios)}"

        output = (line1, line2, line3)

        return "\n".join(output)
