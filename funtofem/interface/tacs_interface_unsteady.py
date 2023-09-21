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

from __future__ import print_function

__all__ = [
    "TacsIntegrationSettings",
    "TacsUnsteadyInterface",
]

from mpi4py import MPI
from tacs import TACS, pytacs, functions
from .utils import f2f_callback
from ._solver_interface import SolverInterface
from typing import TYPE_CHECKING
import os, numpy as np
from .utils.general_utils import real_norm, imag_norm


class TacsIntegrationSettings:
    INTEGRATION_TYPES = ["BDF", "DIRK"]

    def __init__(
        self,
        integration_type: str = "BDF",
        integration_order: int = 2,
        L2_convergence: float = 1e-12,
        L2_convergence_rel: float = 1e-12,
        jac_assembly_freq: int = 1,
        write_solution: bool = True,
        number_solution_files: bool = True,
        print_timing_info: bool = False,
        print_level: int = 0,
        start_time: float = 0.0,
        dt: float = 0.1,
        num_steps: int = 10,
    ):
        assert integration_type in TacsIntegrationSettings.INTEGRATION_TYPES

        self.integration_type = integration_type
        self.integration_order = integration_order
        self.L2_convergence = L2_convergence
        self.L2_convergence_rel = L2_convergence_rel
        self.jac_assembly_freq = jac_assembly_freq
        self.write_solution = write_solution
        self.number_solution_files = number_solution_files
        self.print_timing_info = print_timing_info
        self.print_level = print_level
        self.start_time = start_time
        self.end_time = start_time + dt * num_steps
        self.num_steps = num_steps

    @property
    def is_bdf(self) -> bool:
        return self.integration_type == "BDF"

    @property
    def is_dirk(self) -> bool:
        return self.integration_type == "DIRK"

    @property
    def num_stages(self) -> int:
        return self.integration_order - 1


class TacsOutputGeneratorUnsteady:
    def __init__(self, output_dir, name="tacs_output", f5=None):
        self.output_dir = output_dir if output_dir is not None else os.getcwd()
        self.name = name
        self.f5 = f5

    def __call__(self, step):
        if self.f5 is not None:
            filename = self.name + "_%3.3d" % step + ".f5"
            filepath = os.path.join(self.output_dir, filename)

            # is this how to do it?
            self.f5.writeToFile(filepath)
        return


class TacsUnsteadyInterface(SolverInterface):
    """
    A base class to do coupled unsteady simulations with TACS
    """

    def __init__(
        self,
        comm,
        model,
        assembler=None,
        gen_output: TacsOutputGeneratorUnsteady = None,
        thermal_index: int = 0,
        struct_id: int = None,
        integration_settings: TacsIntegrationSettings = None,
        tacs_comm=None,
        nprocs=None,
        debug=False,
    ):
        self.comm = comm
        self.tacs_comm = tacs_comm
        self.nprocs = nprocs

        # Debug flag
        self._debug = debug
        if self.comm.rank != 0:
            self._debug = False

        # get active design variables
        self.struct_variables = []
        for var in model.get_variables():
            if var.analysis_type == "structural":
                self.struct_variables.append(var)

        self.integration_settings = integration_settings
        self.gen_output = gen_output

        # override boolean to compute coordinate derivatives
        # when no shape variables present (needed for some unittests)
        self._coord_deriv_override = False

        # initialize variables
        self._initialize_variables(
            model, assembler, thermal_index=thermal_index, struct_id=struct_id
        )

        if self.assembler is not None:
            if self.tacs_comm is None:
                self.tacs_comm = self.assembler.getMPIComm()

            # Initialize the structural nodes in the bodies
            struct_X = self.struct_X.getArray()
            for body in model.bodies:
                body.initialize_struct_nodes(struct_X, struct_id=struct_id)

    # Allocate data for each scenario
    class ScenarioData:
        def __init__(self, assembler, func_list, func_tags):
            # Initialize the assembler objects
            self.assembler = assembler
            self.func_list = func_list
            self.func_tags = func_tags
            self.func_grad = []
            self.struct_rhs_vec = None

            self.u = None
            self.dfdx = []
            self.dfdXpts = []
            self.dfdu = []
            self.psi = []

            if self.assembler is not None:
                # Store the solution variables
                self.u = self.assembler.createVec()

                self.q = self.assembler.createVec()
                self.qdot = self.assembler.createVec()
                self.qddot = self.assembler.createVec()

                # Store information about the adjoint
                for func in self.func_list:
                    self.dfdx.append(self.assembler.createDesignVec())
                    self.dfdXpts.append(self.assembler.createNodeVec())
                    self.dfdu.append(self.assembler.createVec())
                    self.psi.append(self.assembler.createVec())

            return

    def _initialize_integrator(
        self,
        model,
    ):
        # setup the integrator looping over each of the scenarios
        self.integrator = {}
        self.F = {}
        self.auxElems = {}
        for scenario in model.scenarios:
            # Create the time integrator and allocate the load data structures
            if scenario.tacs_integration_settings.is_bdf:
                self.integrator[scenario.id] = TACS.BDFIntegrator(
                    self.assembler,
                    scenario.tacs_integration_settings.start_time,
                    scenario.tacs_integration_settings.end_time,
                    float(
                        scenario.tacs_integration_settings.num_steps
                    ),  # should this be -1 here?
                    scenario.tacs_integration_settings.integration_order,
                )

                self.integrator[scenario.id].setAbsTol(
                    scenario.tacs_integration_settings.L2_convergence
                )
                self.integrator[scenario.id].setRelTol(
                    scenario.tacs_integration_settings.L2_convergence_rel
                )

                self.integrator[scenario.id].setPrintLevel(
                    scenario.tacs_integration_settings.print_level
                )

                # Create a force vector for each time step
                self.F[scenario.id] = [
                    self.assembler.createVec()
                    for i in range(scenario.tacs_integration_settings.num_steps + 1)
                ]
                # Auxillary element object for applying tractions/pressure
                self.auxElems[scenario.id] = [
                    TACS.AuxElements()
                    for i in range(scenario.tacs_integration_settings.num_steps + 1)
                ]

            elif scenario.tacs_integration_settings.is_dirk:
                self.numStages = scenario.tacs_integration_settings.num_stages
                self.integrator[scenario.id] = TACS.DIRKIntegrator(
                    self.assembler,
                    self.tInit,
                    self.tFinal,
                    float(self.numSteps),
                    self.numStages,
                )
                # Create a force vector for each time stage
                self.F[scenario.id] = [
                    self.assembler.createVec()
                    for i in range((self.numSteps + 1) * self.numStages)
                ]
                # Auxiliary element object for applying tractions/pressure at each time stage
                self.auxElems[scenario.id] = [
                    TACS.AuxElements()
                    for i in range((self.numSteps + 1) * self.numStages)
                ]
        return

    def _initialize_variables(
        self,
        model,
        assembler=None,
        struct_id=None,
        thermal_index=0,
    ):
        self.thermal_index = thermal_index
        self.struct_id = struct_id

        # Boolean indicating whether TACSAssembler is on this processor
        # or not. If not, all variables are None.
        self.tacs_proc = False

        # Assembler object
        self.assembler = None

        # TACS vectors
        self.ans = None
        self.ext_force = None

        if assembler is not None:
            # Set the assembler
            self.assembler = assembler
            self.tacs_proc = True

            # Create the scenario-independent solution data
            self.ans = self.assembler.createVec()
            self.ext_force = self.assembler.createVec()

            # Allocate the nodal vector
            self.struct_X = assembler.createNodeVec()
            self.assembler.getNodes(self.struct_X)

            # required for AverageTemp function, not sure if needed on
            # body level
            self.vol = 1.0

            # Allocate the different solver pieces
        # Allocate the scenario data
        self.scenario_data = {}
        for scenario in model.scenarios:
            func_list, func_tags = self._allocate_functions(scenario)

            self.scenario_data[scenario.id] = self.ScenarioData(
                self.assembler, func_list, func_tags
            )

        if self.tacs_proc:
            self._initialize_integrator(model)

    def _allocate_functions(self, scenario):
        """
        Allocate the data required to store the function values and
        compute the gradient for a given scenario. This function should
        be called only from a processor where the assembler is defined.
        """

        func_list = []
        func_tag = []

        if self.tacs_proc:
            # Create the list of functions and their corresponding function tags
            for func in scenario.functions:
                if func.analysis_type != "structural":
                    func_list.append(None)
                    func_tag.append(0)

                elif func.name.lower() == "ksfailure":
                    ksweight = 50.0
                    if func.options is not None and "ksweight" in func.options:
                        ksweight = func.options["ksweight"]
                    func_list.append(
                        functions.KSFailure(self.assembler, ksWeight=ksweight)
                    )
                    func_tag.append(1)

                elif func.name.lower() == "compliance":
                    func_list.append(functions.Compliance(self.assembler))
                    func_tag.append(1)

                elif func.name.lower() == "temperature":
                    func_list.append(
                        functions.AverageTemperature(self.assembler, volume=self.vol)
                    )
                    func_tag.append(1)

                elif func.name.lower() == "heatflux":
                    func_list.append(functions.HeatFlux(self.assembler))
                    func_tag.append(1)

                # note center of mass x,y,z functions are not meant for unsteady analyses according to the github TACS repo
                # and are not included in this unsteady interface

                elif func.name == "mass":
                    func_list.append(functions.StructuralMass(self.assembler))
                    func_tag.append(-1)

                else:
                    print("WARNING: Unknown function being set into TACS set to mass")
                    func_list.append(functions.StructuralMass(self.assembler))
                    func_tag.append(-1)

        return func_list, func_tag

    def set_functions(self, scenario, bodies):
        """
        Set the functions into the TACS integrator, not for assembler.
        """
        if self.tacs_proc:
            func_list = self.scenario_data[scenario.id].func_list

            self.integrator[scenario.id].setFunctions(func_list)
        return

    def set_variables(self, scenario, bodies):
        """
        Set the design variable values into the structural solver.

        This takes the variables that are set in the list of :class:`~variable.Variable` objects
        and sets them into the TACSAssembler object.

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        bodies: list of :class:`~body.Body` objects
            The bodies in the model
        """

        if self.tacs_proc:
            # Set the design variable values on the processors that
            # have an instance of TACSAssembler.
            xvec = self.assembler.createDesignVec()
            self.assembler.getDesignVars(xvec)
            xarray = xvec.getArray()

            # This assumes that the TACS variables are not distributed and are set
            # only on the tacs_comm root processor.
            if self.tacs_comm.rank == 0:
                for i, var in enumerate(self.struct_variables):
                    xarray[i] = var.value

            self.assembler.setDesignVars(xvec)
        return

    def get_functions(self, scenario, bodies):
        """
        Evaluate the structural functions of interest.
        The functions are evaluated based on the values of the state variables set
        into the TACSAssembler and TACSIntegrator objects.
        These values are only available on the TACS processors,
        but are broadcast to all processors after evaluation.
        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        bodies: list of :class:`~body.Body` objects
            The bodies in the model
        """

        # Evaluate the list of functions of interest using TACS integrator
        feval = None
        if self.tacs_proc:
            feval = self.integrator[scenario.id].evalFunctions(
                self.scenario_data[scenario.id].func_list
            )

        # Broadcast the list across all procs including non-struct procs
        feval = self.comm.bcast(feval, root=0)

        # Set the function values on all procs
        for ifunc, func in enumerate(scenario.functions):
            if func.analysis_type == "structural":
                func.value += feval[ifunc]

        return

    def get_function_gradients(self, scenario, bodies):
        """
        Take the TACS gradients, computed in the post_adjoint() call
        and place them into the functions of interest. This function can only
        be called after solver.post_adjoint().

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        bodies: list of :class:`~body.Body` objects
            The bodies in the model
        """

        func_grad = self.scenario_data[scenario.id].func_grad

        for ifunc, func in enumerate(scenario.functions):
            for ivar, var in enumerate(self.struct_variables):
                func.add_gradient_component(var, func_grad[ifunc][ivar])

        return

    def _update_assembler_vars(self, scenario, bodies):
        """
        Update the internal data in the assembler before
        forward / adjoint analysis in TACS
        """
        # set the design variables into the assembler
        self.set_variables(scenario, bodies)

        if self.tacs_proc:
            for body in bodies:
                # get an in-place array of the structural nodes
                struct_X = self.struct_X.getArray()

                # set the structural node locations into the array
                struct_X[:] = body.get_struct_nodes()

                # set the structural nodes into the tacs assembler
                self.assembler.setNodes(self.struct_X)

            vars0 = self.scenario_data[scenario.id].q
            dvars0 = self.scenario_data[scenario.id].qdot
            ddvars0 = self.scenario_data[scenario.id].qddot

            # write in the initial conditions of the dynamics
            self.assembler.setInitConditions(vec=vars0, dvec=dvars0, ddvec=ddvars0)

        return

    def initialize(self, scenario, bodies):
        """
        Initialize the internal data for solving the FEM governing
        eqns. Set the nodes in the structural mesh to be consistent
        with the nodes stored in the body classes.

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        bodies: list of :class:`~body.Body` objects
            The bodies in the model
        """

        if self.tacs_proc:
            self._update_assembler_vars(scenario, bodies)
            self.ext_force.zeroEntries()
            self.integrator[scenario.id].iterate(0, self.ext_force)

        return 0

    def iterate(self, scenario, bodies, step):
        """
        unsteady TACS time integration

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        bodies: :class:`~body.Body`
            list of FUNtoFEM bodies
        step: integer
            Time step number in TACS
        """

        fail = 0

        if self.tacs_proc:
            # get the external force vector for the integrator & zero it
            self.ext_force.zeroEntries()
            ext_force_array = self.ext_force.getArray()

            # get ndof of the problem (3 for elastic, 4 for thermoelastic)
            ndof = self.assembler.getVarsPerNode()

            # Copy external loads and heat fluxes to the structure
            # Does this overwrite loads from multiple bodies on same elements?
            for body in bodies:
                # get and copy struct loads into ext_force_array
                struct_loads = body.get_struct_loads(scenario, time_index=step)
                if struct_loads is not None:
                    for i in range(3):
                        ext_force_array[i::ndof] += struct_loads[i::3].astype(
                            TACS.dtype
                        )

                # get and copy struct heat fluxes into ext_forces
                struct_flux = body.get_struct_heat_flux(scenario, time_index=step)
                if struct_flux is not None:
                    ext_force_array[self.thermal_index :: ndof] += struct_flux[
                        :
                    ].astype(TACS.dtype)

            # Iterate the TACS integrator
            self.integrator[scenario.id].iterate(step, self.ext_force)

            # extract the structural disps and temps from the integrator
            _, self.ans, _, _ = self.integrator[scenario.id].getStates(step)
            states = self.ans.getArray()

            for body in bodies:
                # copy struct_disps to the body
                struct_disps = body.get_struct_disps(scenario, time_index=step)
                if struct_disps is not None:
                    for i in range(3):
                        struct_disps[i::3] = states[i::ndof].astype(body.dtype)

                # copy struct temps to the body, converting from gauge to absolute temp with T_ref
                struct_temps = body.get_struct_temps(scenario, time_index=step)
                if struct_temps is not None:
                    struct_temps[:] = (
                        states[self.thermal_index :: ndof].astype(body.dtype)
                        + scenario.T_ref
                    )

        return fail

    def post(self, scenario, bodies):
        """
        This function is called after the analysis is completed

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        bodies: :class:`~body.Body`
            list of FUNtoFEM bodies
        """
        if self.tacs_proc:
            # Save the solution vector
            self.scenario_data[scenario.id].u.copyValues(self.ans)
            if self.gen_output is not None:
                vec = self.assembler.createVec()
                for time_step in range(scenario.steps):
                    # Extract states
                    time, q, _, _ = self.integrator[scenario.id].getStates(time_step)
                    vec.copyValues(q)
                    # Set states mode in assembler
                    self.assembler.setVariables(vec)
                    # Write output .f5
                    self.gen_output(time_step)

        return

    def initialize_adjoint(self, scenario, bodies):
        """
        Initialize the solver for adjoint computations

        Initializes the incoming load and heat flux ajp
        sensitivities for

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        bodies: :class:`~body.Body`
            list of FUNtoFEM bodies
        """

        self._update_assembler_vars(scenario, bodies)

        if self.tacs_proc:
            self.struct_rhs_vec = []
            func_list = self.scenario_data[scenario.id].func_list
            self.integrator[scenario.id].setFunctions(func_list)
            self.integrator[scenario.id].evalFunctions(func_list)

            for func in range(len(func_list)):
                self.struct_rhs_vec.append(self.assembler.createVec())

        return

    def set_states(self, scenario, bodies, step):
        """
        Load the states (struct_disps) associated with this step.

        **Note: this code is deprecated, it is included in the end of
        the iterate function.**

        **Note: in the NLBGS algorithm the transfer scheme uses the
        structural displacements from the prior step.
        set_states will request the states from the previous step
        but then ask the structural solver to linearize
        about the current step in iterate_adjoint**

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        bodies: list of :class:`~body.Body` objects
            The bodies in the model
        step: int
            The time step number that the driver wants the states from
        """

        pass

        # if self.tacs_proc:
        #     _, self.ans, _, _ = self.integrator[scenario.id].getStates(step)
        #     states = self.ans.getArray()
        #     ndof = self.assembler.getVarsPerNode()

        #     for body in bodies:
        #         struct_disps = body.get_struct_disps(scenario, time_index=step)
        #         if struct_disps is not None:
        #             for i in range(3):
        #                 struct_disps[i::3] = states[i::ndof].astype(body.dtype)

        #         struct_temps = body.get_struct_temps(scenario, time_index=step)
        #         if struct_temps is not None:
        #             struct_temps[:] = (
        #                 states[self.thermal_index :: ndof].astype(body.dtype)
        #                 + scenario.T_ref
        #             )
        # return

    def iterate_adjoint(self, scenario, bodies, step):
        """
        Iterate the adjoint sensitivities for TACS FEM solver, unsteady case

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        bodies: :class:`~body.Body`
            list of FUNtoFEM bodies
        """

        fail = 0
        # rev_step = scenario.steps - step + 1

        if self.tacs_proc:
            # extract the list of functions, dfdu, etc
            func_list = self.scenario_data[scenario.id].func_list
            func_tags = self.scenario_data[scenario.id].func_tags

            psi = self.scenario_data[scenario.id].psi

            # iterate over each function
            for ifunc in range(len(func_list)):
                # get the solution data for this function
                rhs_func = self.struct_rhs_vec[ifunc].getArray()
                rhs_func[:] = 0.0  # reset it to zero

                # if not an adjoint function, move onto next function
                if func_tags[ifunc] == -1:
                    continue

                ndof = self.assembler.getVarsPerNode()
                # add struct_disps, struct_flux ajps to the res_adjoint or
                # the residual of the TACS structural adjoint system
                for body in bodies:
                    struct_disps_ajp = body.get_struct_disps_ajp(scenario)
                    if struct_disps_ajp is not None:
                        for i in range(3):
                            rhs_func[i::ndof] -= struct_disps_ajp[i::3, ifunc].astype(
                                TACS.dtype
                            )

                    struct_temps_ajp = body.get_struct_temps_ajp(scenario)
                    if struct_temps_ajp is not None:
                        rhs_func[self.thermal_index :: ndof] -= struct_temps_ajp[
                            :, ifunc
                        ].astype(TACS.dtype)

            # iterate the integrator solver, outside of function loop
            self.integrator[scenario.id].initAdjoint(step)
            self.integrator[scenario.id].iterateAdjoint(step, self.struct_rhs_vec)
            self.integrator[scenario.id].postAdjoint(step)

            # function loop to extract struct load, heat flux adjoints for each func
            for ifunc in range(len(func_list)):
                # get the struct load, flux sensitivities out of integrator
                psi = self.integrator[scenario.id].getAdjoint(step, ifunc)
                psi_array = psi.getArray()

                # pass sensitivities back to each body for loads, heat flux
                for body in bodies:
                    # pass on struct loads adjoint product
                    struct_loads_ajp = body.get_struct_loads_ajp(scenario)
                    if struct_loads_ajp is not None:
                        for i in range(3):
                            struct_loads_ajp[i::3, ifunc] = -psi_array[i::ndof].astype(
                                body.dtype
                            )

                    # pass on struct flux adjoint product
                    struct_flux_ajp = body.get_struct_heat_flux_ajp(scenario)
                    if struct_flux_ajp is not None:
                        struct_flux_ajp[:, ifunc] = -psi_array[
                            self.thermal_index :: ndof
                        ].astype(body.dtype)

        return fail

    def post_adjoint(self, scenario, bodies):
        """
        This function is called after the adjoint variables have been computed.
        This function finalizes the total derivative of the function w.r.t. the design variables
        by compute the gradient
        gradient = df/dx + psi_S * dS/dx
        These values are only computed on the TACS processors.
        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        bodies: :class:`~body.Body`
            list of FUNtoFEM bodies
        """

        func_grad = None
        if self.tacs_proc:
            func_grad = []

            # self._final_iterate_adjoint(scenario, bodies)

            for ifunc, func in enumerate(scenario.functions):
                # ff = self.integrator[scenario.id].getStates(1)
                # print(ff, flush=True)
                dfdx = self.integrator[scenario.id].getGradient(ifunc)
                dfdx.beginSetValues()
                dfdx.endSetValues()

                func_grad.append(dfdx.getArray().copy())

        # Broadcast gradients to all processors
        self.scenario_data[scenario.id].func_grad = self.comm.bcast(func_grad, root=0)
        return

    def get_coordinate_derivatives(self, scenario, bodies, step):
        if self.tacs_proc:
            fXptSens_vec = self.assembler.createNodeVec()

            for ibody, body in enumerate(bodies):
                shape_variables = (
                    body.variables["shape"] if "shape" in body.variables else []
                ) + (
                    scenario.variables["shape"] if "shape" in scenario.variables else []
                )
                if len(shape_variables) > 0 or self._coord_deriv_override:
                    # TACS should accumulate the derivs internally, only evaluate at first timestep
                    if step == 0:
                        for nfunc, func in enumerate(scenario.functions):
                            if func.adjoint:
                                fXptSens_vec = self.integrator[
                                    scenario.id
                                ].getXptGradient(nfunc)
                            elif func.name == "mass":
                                tacsfunc = functions.StructuralMass(self.assembler)
                                self.assembler.evalXptSens(tacsfunc, fXptSens_vec)

                            fXptSens_vec.beginSetValues()
                            fXptSens_vec.endSetValues()

                            fxptSens = fXptSens_vec.getArray()
                            struct_shape_term = body.get_struct_coordinate_derivatives(
                                scenario
                            )

                            struct_shape_term[:, nfunc] += fxptSens.astype(body.dtype)

        pass

    def step_pre(self, scenario, bodies, step):
        pass

    def step_solver(self, scenario, bodies, step, fsi_subiter):
        pass

    def step_post(self, scenario, bodies, step):
        pass

    @classmethod
    def create_from_bdf(
        cls,
        model,
        comm,
        nprocs,
        bdf_file,
        output_dir=None,
        callback=None,
        struct_options={},
        thermal_index=-1,
        debug=False,
    ):
        """
        Create a TacsSteadyInterface instance using the pytacs BDF loader

        Parameters
        ----------
        model: :class:`FUNtoFEMmodel`
            The model class associated with the problem
        comm: MPI.comm
            MPI communicator (typically MPI_COMM_WORLD)
        bdf_file: str
            The BDF file name
        prefix: path
            Path to write f5 output
        callback: function
            The element callback function for pyTACS
        struct_options: dictionary
            The options passed to pyTACS
        """

        # Split the communicator
        world_rank = comm.Get_rank()
        if world_rank < nprocs:
            color = 1
        else:
            color = MPI.UNDEFINED
        tacs_comm = comm.Split(color, world_rank)

        assembler = None
        f5 = None
        if world_rank < nprocs:
            # Create the assembler class
            fea_assembler = pytacs.pyTACS(bdf_file, tacs_comm, options=struct_options)

            """
            Automatically adds structural variables from the BDF / DAT file into TACS
            as long as you have added them with the same name in the DAT file.
            Uses a custom funtofem callback to create thermoelastic shells which are unavailable
            in pytacs default callback. And creates the DVs in the correct order in TACS based on DVPREL cards.
            """

            # get dict of struct DVs from the bodies and structural variables
            # only supports thickness DVs for the structure currently
            structDV_dict = {}
            variables = model.get_variables()
            structDV_names = []

            # Get the structural variables from the global list of variables.
            struct_variables = []
            for var in variables:
                if var.analysis_type == "structural":
                    struct_variables.append(var)
                    structDV_dict[var.name.lower()] = var.value
                    structDV_names.append(var.name.lower())

            # use the default funtofem callback if none is provided
            if callback is None:
                include_thermal = any(
                    ["therm" in body.analysis_type for body in model.bodies]
                )
                callback = f2f_callback(
                    fea_assembler, structDV_names, structDV_dict, include_thermal
                )

            # Set up constitutive objects and elements
            fea_assembler.initialize(callback)

            # Set the assembler
            assembler = fea_assembler.assembler

            # Set the output file creator
            f5 = fea_assembler.outputViewer

        # Create the output generator
        gen_output = TacsOutputGeneratorUnsteady(output_dir, f5=f5)

        # get struct ids for coordinate derivatives and .sens file
        struct_id = None
        if assembler is not None:
            # get list of local node IDs with global size, with -1 for nodes not owned by this proc
            num_nodes = fea_assembler.meshLoader.bdfInfo.nnodes
            bdfNodes = range(num_nodes)
            local_tacs_ids = fea_assembler.meshLoader.getLocalNodeIDsFromGlobal(
                bdfNodes, nastranOrdering=False
            )

            """
            the local_tacs_ids list maps nastran nodes to tacs indices with:
                local_tacs_ids[nastran_node-1] = local_tacs_id
            Only a subset of all tacs ids are owned by each processor
                note: tacs_ids in [0, #local_tacs_ids], #local_tacs_ids <= nnodes
                for nastran nodes not on this processor, local_tacs_id[nastran_node-1] = -1

            The next lines of code invert this map to the list 'struct_id' with:
                struct_id[local_tacs_id] = nastran_node

            This is then later used by funtofem_model.write_sensitivity_file method to write
            ESP/CAPS nastran_CAPS.sens files for the tacsAIM to compute shape derivatives
            """

            # get number of non -1 tacs ids, total number of actual tacs_ids
            n_tacs_ids = len([tacs_id for tacs_id in local_tacs_ids if tacs_id != -1])

            # reverse the tacs id to nastran ids map since we want tacs_id => nastran_id - 1
            nastran_ids = np.zeros((n_tacs_ids), dtype=np.int64)
            for nastran_id_m1, tacs_id in enumerate(local_tacs_ids):
                if tacs_id != -1:
                    nastran_ids[tacs_id] = int(nastran_id_m1 + 1)

            # convert back to list of nastran_ids owned by this processor in order
            struct_id = list(nastran_ids)

        # We might need to clean up this code. This is making educated guesses
        # about what index the temperature is stored. This could be wrong if things
        # change later. May query from TACS directly?
        if assembler is not None and thermal_index == -1:
            varsPerNode = assembler.getVarsPerNode()

            # This is the likely index of the temperature variable
            thermal_index = varsPerNode - 1

        # Broad cast the thermal index to ensure it's the same on all procs
        thermal_index = comm.bcast(thermal_index, root=0)

        # Create the tacs interface
        return cls(
            comm,
            model,
            assembler,
            gen_output,
            thermal_index=thermal_index,
            struct_id=struct_id,
            tacs_comm=tacs_comm,
            debug=debug,
        )
