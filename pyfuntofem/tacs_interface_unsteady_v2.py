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
from mpi4py import MPI
from tacs import TACS, pytacs, functions
from .solver_interface import SolverInterface
from typing import TYPE_CHECKING
import os


class IntegrationSettings:
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
        # TODO : add comments for this
        """ """
        assert integration_type in IntegrationSettings.INTEGRATION_TYPES

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
        self.end_time = start_time + dt*num_steps
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
    def __init__(self, prefix, name="tacs_output", f5=None):
        self.count = 0
        self.prefix = prefix
        self.name = name
        self.f5 = f5
        # TODO : complete this class

    def __call__(self):
        # TODO : write f5 files for each time step, we don't know how to do this yet
        if self.f5 is not None:
            file = self.name + f"{self.count}03d.f5"
            filename = os.path.join(self.prefix, file)

            # is this how to do it?
            self.f5.writeToFile(filename)
        self.count += 1
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
        integration_settings: IntegrationSettings = None,
    ):

        self.comm = comm
        self.tacs_comm = None

        # get active design variables
        self.variables = model.get_variables()
        self.struct_variables = []
        for var in self.variables:
            if var.analysis_type == "structural":
                self.struct_variables.append(var)

        self.integration_settings = integration_settings
        self.gen_output = gen_output

        # initialize variables
        self._initialize_variables(
            model, assembler, thermal_index=thermal_index, struct_id=struct_id
        )

        if self.assembler is not None:
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

            self.u = None
            self.dfdx = []
            self.dfdXpts = []
            self.dfdu = []
            self.psi = []

            if self.assembler is not None:
                # Store the solution variables
                self.u = self.assembler.createVec()

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
            #self.integrator[scenario.id] = self.create

            # Create the time integrator and allocate the load data structures
            if self.integration_settings.is_bdf:
                self.integrator[scenario.id] = TACS.BDFIntegrator(
                    self.assembler,
                    self.integration_settings.start_time,
                    self.integration_settings.end_time,
                    float(self.integration_settings.num_steps),
                    self.integration_settings.integration_order,
                )

                self.integrator[scenario.id].setAbsTol(self.integration_settings.L2_convergence)
                self.integrator[scenario.id].setRelTol(self.integration_settings.L2_convergence_rel)

                # Create a force vector for each time step
                self.F[scenario.id] = [self.assembler.createVec() for i in range(self.integration_settings.num_steps + 1)]
                # Auxillary element object for applying tractions/pressure
                self.auxElems[scenario.id] = [TACS.AuxElements() for i in range(self.integration_settings.num_steps + 1)]

            elif self.integration_settings.is_dirk:
                self.numStages = self.integration_settings.num_stages
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
        mat=None,
        pc=None,
        gmres=None,
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
        self.res = None
        self.ans = None
        self.ext_force = None
        self.update = None

        # Matrix, preconditioner and solver method
        # TODO : do we need mat, pc, gmres for unsteady?
        # self.mat = None
        # self.pc = None
        # self.gmres = None

        if assembler is not None:
            # Set the assembler
            self.assembler = assembler
            self.tacs_proc = True

            # Create the scenario-independent solution data
            self.res = self.assembler.createVec()
            self.ans = self.assembler.createVec()
            self.ext_force = self.assembler.createVec()
            self.update = self.assembler.createVec()

            # Allocate the nodal vector
            self.struct_X = assembler.createNodeVec()
            self.assembler.getNodes(self.struct_X)

            # required for AverageTemp function, not sure if needed on
            # body level
            self.vol = 1.0

            # Allocate the different solver pieces - the
            # TODO : do we need gmres, pc, mat for unsteady case?
            # self.mat = mat
            # self.pc = pc
            # self.gmres = gmres

            # if mat is None:
            #     self.mat = assembler.createSchurMat()
            #     self.pc = TACS.Pc(self.mat)
            #     self.gmres = TACS.KSM(self.mat, self.pc, 30)
            # elif pc is None:
            #     self.mat = mat
            #     self.pc = TACS.Pc(self.mat)
            #     self.gmres = TACS.KSM(self.mat, self.pc, 30)
            # elif gmres is None:
            #     self.mat = mat
            #     self.pc = pc
            #     self.gmres = TACS.KSM(self.mat, self.pc, 30)

        # Allocate the scenario data
        self.scenario_data = {}
        for scenario in model.scenarios:
            func_list, func_tags = self._allocate_functions(scenario)
            
            # TODO : does scenario data need to be saved for each time step too?
            self.scenario_data[scenario] = self.ScenarioData(
                self.assembler, func_list, func_tags
            )

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
            func_list = self.scenario_data[scenario].func_list

            self.integrator[scenario.id].setFunctions(func_list)
            self.integrator[scenario.id].evalFunctions(func_list)
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
            feval = self.integrator[scenario.id].evalFunctions(self.scenario_data[scenario].funclist)

        # Broadcast the list across all procs including non-struct procs
        feval = self.comm.bcast(feval, root=0)

        # Set the function values on all procs
        for ifunc, func in enumerate(scenario.functions):
            if func.analysis_type == "structural":
                func.value = feval[ifunc]
        
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

        func_grad = self.scenario_data[scenario].func_grad

        for ifunc, func in enumerate(scenario.functions):
            for ivar, var in enumerate(self.struct_variables):
                func.set_gradient_component(var, func_grad[ifunc][ivar])

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
            for body in bodies:
                # get an in-place array of the structural nodes
                struct_X = self.struct_X

                # set the structural node locations into the array
                struct_X[:] = body.get_struct_nodes()

                # Reset the node locations in TACS (possibly distributing
                # the node locations across TACS processors
                self.assembler.setNodes(self.struct_X)

            # TODO : get_mesh() equivalent of previous unsteady here?

            # Set the solution to zero
            self.ans.zeroEntries()

            # Set the boundary conditions
            self.assembler.setBCs(self.ans)

            # Set the state variables into the assembler object
            # need to do this also for integrator?
            self.assembler.setVariables(self.ans)

            # TODO : assemble unsteady Jacobian
            # Assemble Jacobian matrix and pre-conditioner factor it
            #alpha = 1.0
            #beta = 0.0
            #gamma = 0.0
            #self.assembler.assembleJacobian(alpha, beta, gamma, self.res, self.mat)
            #self.pc.factor()

            # zeroth-iteration of integrator before full iteration loop
            self.integrator.iterate(0)


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

            # TODO : assemble residual of previous time step? 
            # self.res = M*qddot + C*qdot + K * q - force = 0

            # get the external force vector for the integrator & zero it
            self.ext_force.zeroEntries()
            ext_force_array = self.ext_force.getArray()

            # get ndof of the problem (3 for elastic, 4 for thermoelastic)
            ndof = self.assembler.getVarsPerNode()

            # Copy external loads and heat fluxes to the structure
            # Does this overwrite loads from multiple bodies on same elements?
            for body in bodies:
                
                # get and copy struct loads into ext_force_array
                struct_loads = body.get_struct_loads(scenario)
                if struct_loads is not None:
                    for i in range(3):
                        ext_force_array[i::ndof] += struct_loads[i::3].astype(TACS.dtype)

                # get and copy struct heat fluxes into ext_forces
                struct_flux = body.get_struct_flux(scenario)
                if struct_flux is not None:
                    ext_force_array[self.thermal_index::ndof] += struct_flux[:].astype(TACS.dtype)

            # TODO : See if need any of these steps - apply BCs, add ext forces, etc.
            #self.assembler.applyBCs(self.ext_force)

            # Iterate the TACS integrator
            self.integrator.iterate(step, self.ext_force)

            # TODO : anything with BCs here?
            #self.ans.axpy(-1.0, self.update)
            #self.assembler.setBCs(self.ans)

            # extract the structural disps, temps from the assembler
            self.assembler.setVariables(self.ans)
            ans_array = self.ans.getArray()

            # extract the disps, temps to the body
            for body in bodies:

                # copy struct_disps to the body
                struct_disps = body.get_struct_disps(scenario)
                if struct_disps is not None:
                    for i in range(3):
                        struct_disps[i::3] = ans_array[i::ndof].astype(body.dtype)

                # copy struct temps to the body, converting from gauge to abs temp with T_ref
                struct_temps = body.get_struct_temps(scenario)
                if struct_temps is not None:
                    struct_temps[:] = (
                        ans_array[self.thermal_index :: ndof].astype(body.dtype) + body.T_ref
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
            self.scenario_data[scenario].u.copyValues(self.ans)

            if self.gen_output is not None:
                self.gen_output()

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
        
        # TODO : finish initialize adjoint
        func_list = self.scenario_data[scenario].func_list
        self.integrator[scenario.id].evalFunctions(func_list)

        # TODO : do we need to initialize dfdu?
        # Zero the vectors in the sensitivity list
        # dfdu = self.scenario_data[scenario].dfdu
        # for vec in dfdu:
        #     vec.zeroEntries()

        # Compute the derivative of the function with respect to the
        # state variables
        #self.assembler.addSVSens(func_list, dfdu, 1.0, 0.0, 0.0)

            

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

        if self.tacs_proc:
            
            # extract the list of functions, dfdu, etc
            func_list = self.scenario_data[scenario].func_list
            func_tags = self.scenario_data[scenario].func_tags

            # TODO : need to save psi for each time step, each function?
            psi = self.scenario_data[scenario].psi

            # TODO : do we need a dfdu equivalent for integrator or built-in?
            #dfdu = self.scenario_data[scenario].dfdu
            

            # iterate over each function
            for ifunc in range(len(func_list)):

                # get the solution data for this function
                ext_force_adjoint = self.res.getArray()

                # if not an adjoint function, move onto next function
                if func_tags[ifunc] == -1:
                    continue

                # TODO : add dfdu sensivities here?

                ndof = self.assembler.getVarsPerNode()
                # add struct_disps, struct_flux ajps to the res_adjoint or 
                # the residual of the TACS structural adjoint system
                for body in bodies:

                    struct_disps_ajp = body.get_aero_disps(scenario)
                    if struct_disps_ajp is not None:
                        for i in range(3):
                            ext_force_adjoint[i::ndof] -= struct_disps_ajp[i::3, ifunc].astype(
                                TACS.dtype
                            )

                    struct_temps_ajp = body.get_struct_temps_ajp(scenario)
                    if struct_temps_ajp is not None:
                        ext_force_adjoint[self.thermal_index :: ndof] -= struct_temps_ajp[
                            :, ifunc
                        ].astype(TACS.dtype)

            # TODO : do I need to set functions for integrator again? Does this in TACS/problems/transient.py
            #self.integrator.setFunctions(func_list)

            # iterate the integrator solver, outside of function loop
            self.integrator[scenario.id].initAdjoint(step)
            self.integrator[scenario.id].iterateAdjoint(step, ext_force_adjoint)
            self.integrator[scenario.id].postAdjoint(step)

            # function loop to extract struct load, heat flux adjoints for each func
            for ifunc in range(len(func_list)):

                # TODO : maybe psi needs to be saved for each time step here, the adjoints
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

        func_grad = []

        # TODO : add sensitivity across all processors?
        # Add the values across processors - this is required to
        # collect the distributed design variable contributions
        # for vec in dfdx:
        #     vec.beginSetValues(TACS.ADD_VALUES)
        #     vec.endSetValues(TACS.ADD_VALUES)

        for ifunc, func in enumerate(scenario.functions):

            grad = self.integrator[scenario.id].getGradient(ifunc)
            
            # TODO : don't have to get DVsens and addAdjointResProducts?
            # TACS/problems/transient.py doesn't seem to do this

            func_grad.append(
                grad.getArray().copy()
            )

        # Broadcast gradients to all processors
        self.scenario_data[scenario.id].func_grad = self.comm.bcast(func_grad, root=0)

        
            



    def get_coordinate_derivatives(self, scenario, bodies, step):
        pass

    def step_pre(self, scenario, bodies, step):
        pass

    def step_solver(self, scenario, bodies, step, fsi_subiter):
        pass

    def step_post(self, scenario, bodies, step):
        pass



def createTacsUnsteadyInterfaceFromBDF(
    model,
    comm,
    nprocs,
    bdf_file,
    integration_settings:IntegrationSettings,
    t0=0.0,
    tf=1.0,
    prefix="",
    callback=None,
    struct_options={},
    thermal_index=-1,
):
    # TODO : determine if inputs should be t0,tf or nsteps, dt
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
    prefix:

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

        # Set up constitutive objects and elements
        fea_assembler.initialize(callback)

        # Set the assembler
        assembler = fea_assembler.assembler

        # Set the output file creator
        f5 = fea_assembler.outputViewer

    # Create the output generator
    gen_output = TacsOutputGeneratorUnsteady(prefix, f5=f5)

    # We might need to clean up this code. This is making educated guesses
    # about what index the temperature is stored. This could be wrong if things
    # change later. May query from TACS directly?
    if assembler is not None and thermal_index == -1:
        varsPerNode = assembler.getVarsPerNode()

        # This is the likely index of the temperature variable
        if varsPerNode == 1:  # Thermal only
            thermal_index = 0
        elif varsPerNode == 4:  # Solid + thermal
            thermal_index = 3
        elif varsPerNode >= 7:  # Shell or beam + thermal
            thermal_index = 3

    # Broad cast the thermal index to ensure it's the same on all procs
    thermal_index = comm.bcast(thermal_index, root=0)

    # Create the tacs interface
    interface = TacsUnsteadyInterface(
        comm, model, assembler, gen_output, thermal_index=thermal_index, integration_settings=integration_settings
    )

    return interface
