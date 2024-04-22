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

__all__ = ["TacsInterface", "TacsSteadyInterface"]

from mpi4py import MPI
from tacs import pytacs, TACS, functions
from .utils import f2f_callback, addLoadsFromBDF
from ._solver_interface import SolverInterface
import os, numpy as np
from .tacs_interface_unsteady import TacsUnsteadyInterface
from .utils.general_utils import real_norm, imag_norm
from .utils.relaxation_utils import AitkenRelaxationTacs


class TacsInterface:
    """
    Base Creator class for TacsSteadyInterface or TacsUnsteadyInterface
    """

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
        relaxation_scheme: AitkenRelaxationTacs = None,
    ):
        """
        Class method to create either a TacsSteadyInterface or TacsUnsteadyInterface instance using the pytacs BDF loader

        Parameters
        ----------
        model: :class:`FUNtoFEMmodel`
            The model class associated with the problem
        comm: MPI.comm
            MPI communicator (typically MPI_COMM_WORLD)
        bdf_file: str
            The BDF file name
        output_dir: filepath
            directory of output for .f5 files generated from TACS

        Optional Parameters usually not specified
        -----------------------------------------
        struct_DVs: List
            list of struct DV values for the built-in funtofem callback method
        callback: function
            The element callback function for pyTACS
        struct_options: dictionary
            The options passed to pyTACS
        thermal_index: int
            index for thermal index
        relaxation_scheme: Relaxation Scheme Object
            Object to store relaxation scheme settings. If None, then no relaxation is used.
        """

        # check whether each scenario is all steady or all unsteady
        steady_list = [scenario.steady for scenario in model.scenarios]
        unsteady_list = [not (scenario.steady) for scenario in model.scenarios]

        if all(steady_list):
            # create a TACS steady interface
            prefix = output_dir if output_dir is not None else ""
            return TacsSteadyInterface.create_from_bdf(
                model=model,
                comm=comm,
                nprocs=nprocs,
                bdf_file=bdf_file,
                prefix=prefix,
                callback=callback,
                struct_options=struct_options,
                thermal_index=thermal_index,
                debug=debug,
                relaxation_scheme=relaxation_scheme,
            )
        elif all(unsteady_list):
            # create a TACS steady interface
            return TacsUnsteadyInterface.create_from_bdf(
                model=model,
                comm=comm,
                nprocs=nprocs,
                bdf_file=bdf_file,
                output_dir=output_dir,
                callback=callback,
                struct_options=struct_options,
                thermal_index=thermal_index,
                debug=debug,
                relaxation_scheme=relaxation_scheme,
            )
        else:
            raise AssertionError(
                "Can't built a Tacs Interface if scenarios are a mixture of steady and unsteady..."
            )


class TacsSteadyInterface(SolverInterface):
    """
    A base class to do coupled steady simulations with TACS
    """

    PANEL_LENGTH_CONSTR = "length"

    def __init__(
        self,
        comm,
        model,
        assembler=None,
        gen_output=None,
        thermal_index=0,
        struct_id=None,
        tacs_comm=None,
        override_rotx=False,
        Fvec=None,
        nprocs=None,
        relaxation_scheme: AitkenRelaxationTacs = None,
        debug=False,
        panel_length_constraint=None,
    ):
        """
        Initialize the TACS implementation of the SolverInterface for the FUNtoFEM
        framework.

        Key assumptions
        ---------------
        1. There is only one body in the list of bodies

        2. The structural variables must be added to the FUNtoFEM framework in
        the same order that they are defined in the TACSAssembler object.

        3. The structural response is either linear or does not exhibit complex
        nonlinearity that would require a continuation-type solver.

        Parameters
        ----------
        comm: MPI.comm
            MPI communicator
        model: :class:`~funtofem_model.FUNtoFEMmodel`
            The model containing the design data
        assembler: The ``TACSAssembler`` object
            Can pass None if you want to override the class
        gen_output: Function
            Callback for generating output files for visualization
        thermal_index: int
            Index of the structural degree of freedom corresponding to the temperature
        struct_id: list or np.ndarray
            List of the unique global ids of all the structural nodes
        tacs_comm: MPI.comm
            MPI communicator with only n_tacs_procs active
        Fvec: python pointer to TACS load vector
            constant load vector such as for engine weight, if None then it is not used
        nprocs: int
            argument mainly for hidden use by drivers (matches tacs_comm)
        use_aitken: boolean
            Whether to use Aitken relaxation.
        """

        self.comm = comm
        self.tacs_comm = tacs_comm
        self.model = model
        self.nprocs = nprocs

        # Flag to output heat flux instead of rotx
        self.override_rotx = override_rotx

        # Set Aitken relaxation flag
        self.relaxation_scheme = relaxation_scheme
        self.use_aitken = isinstance(relaxation_scheme, AitkenRelaxationTacs)

        # const load in TACS, separate and added onto from coupled loading
        self.has_const_load = Fvec is not None
        self.const_force = None
        if self.has_const_load and assembler is not None:
            self.const_force = Fvec.getArray()

        # Get the list of active design variables from the FUNtoFEM model. This
        # returns the variables in the FUNtoFEM order. By scenario/body.
        self.variables = model.get_variables()

        # setup forward and adjoint tolerances
        super().__init__()

        # Get the structural variables from the global list of variables.
        self.struct_variables = []
        for var in self.variables:
            if var.analysis_type == "structural":
                self.struct_variables.append(var)

        # Set the assembler object - if it exists or not
        self._initialize_variables(
            model,
            assembler,
            thermal_index=thermal_index,
            struct_id=struct_id,
            relaxation_scheme=relaxation_scheme,
        )

        if self.assembler is not None:
            if self.tacs_comm is None:
                self.tacs_comm = self.assembler.getMPIComm()

            # Initialize the structural nodes in the bodies
            struct_X = self.struct_X.getArray()
            for body in model.bodies:
                body.initialize_struct_nodes(struct_X, struct_id=struct_id)

        # Generate output
        self.gen_output = gen_output

        # create panel length constraints
        self.panel_length_constraint = panel_length_constraint
        self.panel_length_name = "PanelLengthCon_PanelLength"

        self._eval_panel_length(forward=True, adjoint=True)

        # Debug flag
        self._debug = debug
        if self.comm.rank != 0:
            self._debug = False

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
        relaxation_scheme: AitkenRelaxationTacs = None,
    ):
        """
        Initialize the variables required for analysis and
        optimization using TACS. This initialization takes in optional
        TACSMat, TACSPc and TACSKsm objects for the solution
        procedure. The assembler object must be defined on the subset
        of structural processors. On all other processors it must be
        None.
        """

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

        # Aitken relaxation variables -- settings
        self.use_aitken = isinstance(relaxation_scheme, AitkenRelaxationTacs)
        self.aitken_min = 0.25
        self.aitken_max = 2.0
        self.theta_init = 1.0
        self.aitken_tol = 1e-13
        self.aitken_debug = False
        self.aitken_debug_more = False

        # Aitken relaxation variables -- primal
        self.theta = None
        self.prev_theta = None
        self.prev_update = None
        self.delta_update = None
        self.update_temp = None

        # Aitken relaxation variables -- adjoint
        self.theta_adj = None
        self.prev_theta_adj = None

        # Matrix, preconditioner and solver method
        self.mat = None
        self.pc = None
        self.gmres = None

        # TACS node locations
        self.struct_X = None

        self.vol = 1.0

        if assembler is not None:
            # Set the assembler
            self.assembler = assembler
            self.tacs_proc = True

            # Create the scenario-independent solution data
            self.res = self.assembler.createVec()
            self.ans = self.assembler.createVec()
            self.ext_force = self.assembler.createVec()
            self.update = self.assembler.createVec()

            # Create Aitken variables
            if self.use_aitken:
                # Update Aitken setting parameters from relaxation_scheme object
                self.aitken_min = relaxation_scheme.theta_min
                self.aitken_max = relaxation_scheme.theta_max
                self.theta_init = relaxation_scheme.theta_init
                self.aitken_tol = relaxation_scheme.aitken_tol
                self.aitken_debug = relaxation_scheme.aitken_debug
                self.aitken_debug_more = relaxation_scheme.aitken_debug_more

                self.theta = self.theta_init
                self.prev_theta = self.theta_init
                self.prev_update = self.assembler.createVec()
                self.delta_update = self.assembler.createVec()
                self.update_temp = self.assembler.createVec()

            # Allocate the nodal vector
            self.struct_X = assembler.createNodeVec()
            self.assembler.getNodes(self.struct_X)

            # required for AverageTemp function, not sure if needed on
            # body level
            self.vol = 1.0

            # Allocate the different solver pieces - the
            self.mat = mat
            self.pc = pc
            self.gmres = gmres

            if mat is None:
                self.mat = assembler.createSchurMat()
                self.pc = TACS.Pc(self.mat)
                self.gmres = TACS.KSM(self.mat, self.pc, 30)
            elif pc is None:
                self.mat = mat
                self.pc = TACS.Pc(self.mat)
                self.gmres = TACS.KSM(self.mat, self.pc, 30)
            elif gmres is None:
                self.mat = mat
                self.pc = pc
                self.gmres = TACS.KSM(self.mat, self.pc, 30)

        # Allocate the scenario data
        self.scenario_data = {}
        for scenario in model.scenarios:
            func_list, func_tags = self._allocate_functions(scenario)
            self.scenario_data[scenario] = self.ScenarioData(
                self.assembler, func_list, func_tags
            )

        return

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

            # Aitken previous psi
            self.prev_psi = []
            self.prev_update_adj = []
            self.update_adj = []
            self.delta_update_adj = []
            self.psi_temp = []

            if self.assembler is not None:
                # Store the solution variables
                self.u = self.assembler.createVec()

                # Store information about the adjoint
                for func in self.func_list:
                    self.dfdx.append(self.assembler.createDesignVec())
                    self.dfdXpts.append(self.assembler.createNodeVec())
                    self.dfdu.append(self.assembler.createVec())
                    self.psi.append(self.assembler.createVec())

                    # Aitken previous adjoint
                    self.prev_psi.append(self.assembler.createVec())
                    self.prev_update_adj.append(self.assembler.createVec())
                    self.update_adj.append(self.assembler.createVec())
                    self.delta_update_adj.append(self.assembler.createVec())
                    self.psi_temp.append(self.assembler.createVec())

            return

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
                    safetyFactor = 1.0
                    if func.options is not None and "safetyFactor" in func.options:
                        safetyFactor = func.options["safetyFactor"]
                    func_list.append(
                        functions.KSFailure(
                            self.assembler, ksWeight=ksweight, safetyFactor=safetyFactor
                        )
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

                elif func.name.lower() == "xcom":
                    func_list.append(
                        functions.CenterOfMass(self.assembler, direction=[1, 0, 0])
                    )
                    func_tag.append(1)

                elif func.name.lower() == "ycom":
                    func_list.append(
                        functions.CenterOfMass(self.assembler, direction=[0, 1, 0])
                    )
                    func_tag.append(1)

                elif func.name.lower() == "zcom":
                    func_list.append(
                        functions.CenterOfMass(self.assembler, direction=[0, 0, 1])
                    )
                    func_tag.append(1)

                elif func.name == "mass":
                    func_list.append(functions.StructuralMass(self.assembler))
                    func_tag.append(-1)

                else:
                    print("WARNING: Unknown function being set into TACS set to mass")
                    func_list.append(functions.StructuralMass(self.assembler))
                    func_tag.append(-1)

        return func_list, func_tag

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
            if self.comm.rank == 0:
                for i, var in enumerate(self.struct_variables):
                    xarray[i] = var.value

            self.assembler.setDesignVars(xvec)

        return

    def set_functions(self, scenario, bodies):
        """
        Set and initialize the types of functions that will be evaluated based
        on the list of functions stored in the scenario.

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        bodies: list of :class:`~body.Body` objects
            The bodies in the model
        """

        return

    def get_functions(self, scenario, bodies):
        """
        Evaluate the structural functions of interest.

        The functions are evaluated based on the values of the state variables set
        into the TACSAssembler object. These values are only available on the
        TACS processors, but these values are broadcast to all processors after the
        evaluation.

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        bodies: list of :class:`~body.Body` objects
            The bodies in the model
        """

        # Evaluate the list of the functions of interest
        feval = None
        if self.tacs_proc:
            feval = self.assembler.evalFunctions(self.scenario_data[scenario].func_list)

        # Broacast the list across all processors - not just structural procs
        feval = self.comm.bcast(feval, root=0)

        # Set the function values on all processors
        for i, func in enumerate(scenario.functions):
            if func.analysis_type == "structural":
                func.value = feval[i]

        return

    def get_function_gradients(self, scenario, bodies):
        """
        Take the gradients that were computed in the post_adjoint() call and
        place them into the functions of interest. This function can only be called
        after solver.post_adjoint(). This call order is guaranteed.

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        bodies: list of :class:`~body.Body` objects
            The bodies in the model
        """

        func_grad = self.scenario_data[scenario].func_grad

        for ifunc, func in enumerate(scenario.functions):
            for i, var in enumerate(self.struct_variables):
                # func.set_gradient_component(var, func_grad[ifunc][i])
                func.add_gradient_component(var, func_grad[ifunc][i])

        return

    def initialize(self, scenario, bodies):
        """
        Initialize the internal data here for solving the governing
        equations. Set the nodes in the structural mesh to be consistent
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
                # Get an in-place array of the structural nodes
                struct_X = self.struct_X.getArray()

                # Set the structural node locations into the array
                struct_X[:] = body.get_struct_nodes()

                # Reset the node locations in TACS (possibly distributing the
                # node locations across TACS processors)
                self.assembler.setNodes(self.struct_X)

            # Set the solution to zero
            self.ans.zeroEntries()

            # Set the boundary conditions
            self.assembler.setBCs(self.ans)

            # Set the variables into the assembler object
            self.assembler.setVariables(self.ans)

            # Assemble the Jacobian matrix and factor it
            alpha = 1.0
            beta = 0.0
            gamma = 0.0
            self.assembler.assembleJacobian(alpha, beta, gamma, self.res, self.mat)
            self.pc.factor()

        return 0

    def iterate(self, scenario, bodies, step):
        """
        This function performs an iteration of the structural solver

        The code performs an update of the governing equations

        S(u, fS, hS) = r(u) - fS - hS

        where fS are the structural loads and hS are the heat fluxes stored in the body
        classes.

        The code computes

        res = r(u) - fS - hS

        and then computes the update

        mat * update = -res

        applies the update

        u = u + update

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        bodies: :class:`~body.Body`
            list of FUNtoFEM bodies
        step: integer
            Step number for the steady-state solution method
        """
        fail = 0

        if self.tacs_proc:
            # Store previous update for Aitken relaxation
            if self.use_aitken and step > 0:
                self.prev_update.copyValues(self.update)
                aitken_min = self.aitken_min
                aitken_max = self.aitken_max

                theta = self.theta
                prev_theta = self.prev_theta

            # Compute the residual from tacs self.res = K*u - f_internal
            self.assembler.assembleRes(self.res)

            # Add the external forces into a TACS vector that will be added to
            # the residual
            self.ext_force.zeroEntries()
            ext_force_array = self.ext_force.getArray()

            # Add the external load and heat fluxes on the structure
            ndof = self.assembler.getVarsPerNode()
            for body in bodies:
                struct_loads = body.get_struct_loads(scenario, time_index=step)
                if self._debug:
                    print(f"========================================")
                    print(f"Inside tacs_interface, step: {step}")
                    print(f"norm of real struct_loads: {real_norm(struct_loads)}")
                    print(f"norm of imaginary struct_loads: {imag_norm(struct_loads)}")
                    print(f"========================================\n", flush=True)
                if struct_loads is not None:
                    for i in range(3):
                        ext_force_array[i::ndof] += struct_loads[i::3].astype(
                            TACS.dtype
                        )

                struct_flux = body.get_struct_heat_flux(scenario, time_index=step)
                if struct_flux is not None:
                    ext_force_array[self.thermal_index :: ndof] += struct_flux[
                        :
                    ].astype(TACS.dtype)

            # add in optional constant load
            if self.has_const_load:
                ext_force_array[:] += self.const_force[:]

            # Zero the contributions at the DOF associated with boundary
            # conditions so that it doesn't interfere with Dirichlet BCs
            self.assembler.applyBCs(self.ext_force)

            # Add the contribution to the residuals from the external forces
            self.res.axpy(-1.0, self.ext_force)

            # Solve for the update
            self.gmres.solve(self.res, self.update)

            if self.comm.rank == 0 and self.aitken_debug:
                print(f"TACS iterate step: {step}", flush=True)

            # Apply Aitken relaxation
            if self.use_aitken and step >= 2:
                # Store update into temp variable
                self.update_temp.copyValues(self.update)

                # Calculate change in the updates
                self.delta_update.copyValues(self.update_temp)
                self.delta_update.axpy(-1, self.prev_update)

                num = self.delta_update.dot(self.update_temp)
                den = self.delta_update.norm() ** 2.0

                # only update theta if vector has changed more than tolerance
                if np.real(den) > self.aitken_tol:
                    theta = prev_theta * (1 - num / den)
                    if self.comm.rank == 0 and self.aitken_debug:
                        print(f"Theta unbounded: {theta}", flush=True)
                else:
                    theta = self.theta_init
                    if self.comm.rank == 0 and self.aitken_debug:
                        print(
                            f"Aitken relaxation: update vector did not change enough to compute relaxation."
                        )

                theta = max(aitken_min, min(aitken_max, np.real(theta)))

                self.update.scale(theta)

            # Apply the update to the solution vector and reset the boundary condition
            # data so that it is precisely statisfied
            self.ans.axpy(-1.0, self.update)
            self.assembler.setBCs(self.ans)

            # Set the variables into the assembler object
            self.assembler.setVariables(self.ans)

            # Extract displacements and temperatures for each body
            ans_array = self.ans.getArray()
            for body in bodies:
                struct_disps = body.get_struct_disps(scenario, time_index=step)
                if struct_disps is not None:
                    for i in range(3):
                        struct_disps[i::3] = ans_array[i::ndof].astype(body.dtype)

                # Set the structural temperature
                struct_temps = body.get_struct_temps(scenario, time_index=step)
                if struct_temps is not None:
                    # absolute temperature in Kelvin of the structural surface
                    struct_temps[:] = (
                        ans_array[self.thermal_index :: ndof].astype(body.dtype)
                        + scenario.T_ref
                    )

        return fail

    def _eval_panel_length(self, forward=True, adjoint=True):
        # whether to do the panel length constraint
        _has_panel_length = None
        if self.comm.rank == 0:
            _has_panel_length = self.panel_length_constraint is not None
        _has_panel_length = self.comm.bcast(_has_panel_length, root=0)

        # print(f"has panel length rank {self.comm.rank} = {_has_panel_length}", flush=True)

        # compute the panel length constraint
        if _has_panel_length:
            if forward:
                funcs_dict = None
                # print(f"rank {self.comm.rank} enter forward", flush=True)
                if self.assembler is not None:
                    funcs = {}
                    funcs_dict = {}
                    ct = 0
                    self.panel_length_constraint.evalConstraints(funcs)
                    # print(f"inside rank 0 check post eval constraints", flush=True)
                    for func in self.model.composite_functions:
                        if self.PANEL_LENGTH_CONSTR in func.name:
                            # assume name of form f"{self.PANEL_LENGTH_CONSTR}-fnum"
                            func.value = funcs[self.panel_length_name][ct]
                            ct += 1
                            funcs_dict[func.full_name] = func.value
                    # print(f"inside rank 0 check : funcs dict = {funcs_dict}", flush=True)

                # broadcast the funcs dict to other processors
                funcs_dict = self.comm.bcast(funcs_dict, root=0)

                # print(f"rank {self.comm.rank} : forward funcs dict = {funcs_dict}", flush=True)

                for func in self.model.composite_functions:
                    if func.full_name in list(funcs_dict.keys()):
                        func.value = funcs_dict[func.full_name]

            # compute the panel length constraint
            if adjoint:
                grads_dict = None
                if self.assembler is not None:
                    funcSens = {}
                    if self.comm.rank == 0:
                        grads_dict = {}
                    ifunc = 0
                    self.panel_length_constraint.evalConstraintsSens(funcSens)
                    for func in self.model.composite_functions:
                        if (
                            self.PANEL_LENGTH_CONSTR in func.name
                            and self.comm.rank == 0
                        ):

                            grads_dict[func.full_name] = {}

                            # assume name of form f"{self.PANEL_LENGTH_CONSTR}-fnum"
                            for ivar, var in enumerate(self.struct_variables):
                                func.derivatives[var] = funcSens[
                                    self.panel_length_name
                                ]["struct"].toarray()[ifunc, ivar]
                                grads_dict[func.full_name][var.full_name] = (
                                    func.derivatives[var]
                                )

                            ifunc += 1

                # broadcast the funcs dict to other processors
                grads_dict = self.comm.bcast(grads_dict, root=0)

                # print(f"rank {self.comm.rank} : grads dict = {grads_dict}", flush=True)

                for func in self.model.composite_functions:
                    if func.full_name in list(grads_dict.keys()):
                        for ivar, var in enumerate(self.struct_variables):
                            func.derivatives[var] = grads_dict[func.full_name][
                                var.full_name
                            ]
                            # print(f"rank {self.comm.rank} : d{func.full_name}/d{var.full_name} = {func.derivatives[var]}", flush=True)

    def post(self, scenario, bodies):
        """
        This function is called after the analysis is completed

        The function is

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        bodies: :class:`~body.Body`
            list of FUNtoFEM bodies
        """

        self._eval_panel_length(adjoint=False)

        # update solution and dv1 state (like _updateAssemblerVars() in pytacs)
        self.set_variables(scenario, bodies)
        if self.tacs_proc:
            # Save the solution vector
            self.scenario_data[scenario].u.copyValues(self.ans)

            if self.override_rotx:
                # Override rotx to be heat flux
                ans_array = self.ans.getArray()
                ndof = self.assembler.getVarsPerNode()

                for body in bodies:
                    struct_flux = body.get_struct_heat_flux(scenario)
                    if struct_flux is not None:
                        ans_array[3::ndof] = struct_flux[:]

                self.assembler.setVariables(self.ans)

            if self.gen_output is not None:
                self.gen_output()

        return

    def initialize_adjoint(self, scenario, bodies):
        """
        Initialize the solver for adjoint computations.

        This code computes the transpose of the Jacobian matrix dS/du^{T}, and factorizes
        it. For structural problems, the Jacobian matrix is symmetric, however for coupled
        thermoelastic problems, the Jacobian matrix is non-symmetric.

        The code also computes the derivative of the structural functions of interest with
        respect to the state variables df/du. These derivatives are stored in the list
        func_list that stores svsens = -df/du. Note that the negative sign applied here.

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        bodies: :class:`~body.Body`
            list of FUNtoFEM bodies
        """

        if self.tacs_proc:
            # Initialize Aitken adjoint variables.
            # There is an Aitken relaxation parameter for each function.
            nf = len(self.scenario_data[scenario].func_list)
            self.theta_adj = np.ones((nf), dtype=TACS.dtype) * self.theta_init
            self.prev_theta_adj = np.ones((nf), dtype=TACS.dtype) * self.theta_init

            # Set the solution data for this scenario
            u = self.scenario_data[scenario].u
            self.assembler.setVariables(u)

            # Assemble the transpose of the Jacobian matrix for the adjoint
            # computations. Note that for thermoelastic computations, the Jacobian
            # matrix is non-symmetric due to the temperature-deformation coupling.
            # The transpose must be used here to get the right result.
            alpha = 1.0  # Jacobian coefficient for the state variables
            beta = 0.0  # Jacobian coeff. for the first time derivative of the state variables
            gamma = 0.0  # Coeff. for the second time derivative of the state variables
            self.assembler.assembleJacobian(
                alpha, beta, gamma, self.res, self.mat, matOr=TACS.TRANSPOSE
            )
            self.pc.factor()

            # Evaluate the functions in preparation for evaluating the derivative
            # of the functions w.r.t. the state variables. Some TACS functions
            # require their evaluation to store internal data before the sensitivities
            # can be computed.
            func_list = self.scenario_data[scenario].func_list
            self.assembler.evalFunctions(func_list)

            # Zero the vectors in the sensitivity list
            dfdu = self.scenario_data[scenario].dfdu
            for vec in dfdu:
                vec.zeroEntries()

            # Compute the derivative of the function with respect to the
            # state variables
            self.assembler.addSVSens(func_list, dfdu, 1.0, 0.0, 0.0)

            # Scale the vectors by -1
            for vec in dfdu:
                vec.scale(-1.0)

        return 0

    def iterate_adjoint(self, scenario, bodies, step):
        """
        This function solves the structural adjoint equations.

        The governing equations for the structures takes the form

        S(u, fS, hS) = r(u) - fS - hS = 0

        The function takes the following adjoint-Jacobian products stored in the bodies

        struct_disps_ajp = psi_D^{T} * dD/duS + psi_L^{T} * dL/dus
        struct_temps_ajp = psi_T^{T} * dT/dtS

        and computes the outputs that are stored in the same set of bodies

        struct_loads_ajp = psi_S^{T} * dS/dfS
        struct_flux_ajp = psi_S^{T} * dS/dhS

        Based on the governing equations, the outputs are computed based on the structural adjoint
        variables as

        struct_loads_ajp = - psi_S^{T}
        struct_flux_ajp = - psi_S^{T}

        To obtain these values, the code must solve the structural adjoint equation

        dS/duS^{T} * psi_S = - df/duS^{T} - dD/duS^{T} * psi_D - dL/duS^{T} * psi_L^{T} - dT/dtS^{T} * psi_T

        In the code, the right-hand-side for the

        dS/duS^{T} * psi_S = struct_rhs_array

        This right-hand-side is stored in the array struct_rhs_array, and computed based on the array

        struct_rhs_array = svsens - struct_disps_ajp - struct_flux_ajp

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        bodies: :class:`~body.Body`
            list of FUNtoFEM bodies
        step: integer
            Step number for the steady-state solution method
        """
        fail = 0

        if self.tacs_proc:
            # Extract the list of functions
            func_list = self.scenario_data[scenario].func_list
            func_tags = self.scenario_data[scenario].func_tags
            dfdu = self.scenario_data[scenario].dfdu
            psi = self.scenario_data[scenario].psi  # psi = psi_S the structual adjoint

            # Set up Aitken variables and parameters locally
            if self.use_aitken:
                aitken_max = self.aitken_max
                aitken_min = self.aitken_min
                prev_psi = self.scenario_data[scenario].prev_psi
                psi_temp = self.scenario_data[scenario].psi_temp
                prev_update_adj = self.scenario_data[scenario].prev_update_adj
                update_adj = self.scenario_data[scenario].update_adj
                delta_update_adj = self.scenario_data[scenario].delta_update_adj

            if self.comm.rank == 0 and self.aitken_debug:
                print(f"TACS adjoint iterate step: {step}", flush=True)

            for ifunc in range(len(func_list)):
                # Check if the function requires an adjoint computation or not
                if func_tags[ifunc] == -1:
                    continue

                # Copy values into the right-hand-side
                # res = - df/duS^{T}
                self.res.copyValues(dfdu[ifunc])

                # Extract the array in-place
                array = self.res.getArray()

                ndof = self.assembler.getVarsPerNode()
                for body in bodies:
                    # Form new right-hand side of structural adjoint equation using state
                    # variable sensitivites and the transformed temperature transfer
                    # adjoint variables. Here we use the adjoint-Jacobian products from the
                    # structural displacements and structural temperatures.
                    struct_disps_ajp = body.get_struct_disps_ajp(scenario)
                    if struct_disps_ajp is not None:
                        for i in range(3):
                            array[i::ndof] -= struct_disps_ajp[i::3, ifunc].astype(
                                TACS.dtype
                            )

                    struct_temps_ajp = body.get_struct_temps_ajp(scenario)
                    if struct_temps_ajp is not None:
                        array[self.thermal_index :: ndof] -= struct_temps_ajp[
                            :, ifunc
                        ].astype(TACS.dtype)

                # Zero the adjoint right-hand-side conditions at DOF locations
                # where the boundary conditions are applied. This is consistent with
                # the forward analysis where the forces/fluxes contributions are
                # zeroed at Dirichlet DOF locations.
                self.assembler.applyBCs(self.res)

                # Solve structural adjoint equation
                self.gmres.solve(self.res, psi[ifunc])

                # Aitken adjoint step
                if self.use_aitken:
                    psi_temp[ifunc].copyValues(psi[ifunc])
                    theta_adj = self.theta_adj
                    prev_theta_adj = self.prev_theta_adj

                    if step >= 2:
                        # Calculate adjoint update value
                        update_adj[ifunc].copyValues(psi_temp[ifunc])
                        update_adj[ifunc].axpy(-1, prev_psi[ifunc])

                    if step >= 3:
                        # Perform Aitken relaxation
                        delta_update_adj[ifunc].copyValues(update_adj[ifunc])
                        delta_update_adj[ifunc].axpy(-1, prev_update_adj[ifunc])

                        num = delta_update_adj[ifunc].dot(update_adj[ifunc])
                        den = delta_update_adj[ifunc].norm() ** 2.0

                        if self.comm.rank == 0 and self.aitken_debug_more:
                            print(
                                f"prev_theta_adj[ifunc]: {prev_theta_adj[ifunc]}",
                                flush=True,
                            )
                            print(f"num: {num}", flush=True)
                            print(f"den: {den}", flush=True)

                        # Only update theta if vector has changed more than tolerance
                        if np.real(den) > self.aitken_tol:
                            theta_adj[ifunc] = prev_theta_adj[ifunc] * (1.0 - num / den)
                            if self.comm.rank == 0 and self.aitken_debug:
                                print(
                                    f"Theta adjoint unbounded, ifunc {ifunc}: {theta_adj[ifunc]}",
                                    flush=True,
                                )
                        else:
                            theta_adj[ifunc] = self.theta_init

                        theta_adj[ifunc] = max(
                            aitken_min, min(aitken_max, np.real(theta_adj[ifunc]))
                        )

                        # Use psi_temp variable to store scaled update
                        psi_temp[ifunc].copyValues(update_adj[ifunc])
                        psi_temp[ifunc].scale(theta_adj[ifunc])

                        psi[ifunc].copyValues(prev_psi[ifunc])
                        psi[ifunc].axpy(1, psi_temp[ifunc])

                # Extract the structural adjoint array in-place
                psi_array = psi[ifunc].getArray()

                # Store psi and update_adj as previous for next iteration
                prev_psi[ifunc].copyValues(psi[ifunc])
                prev_update_adj[ifunc].copyValues(update_adj[ifunc])

                # Set the adjoint-Jacobian products for each body
                for body in bodies:
                    # Compute the structural loads adjoint-Jacobian product. Here
                    # S(u, fS, hS) = r(u) - fS - hS, so dS/dfS = -I and dS/dhS = -I
                    # struct_loads_ajp = psi_S^{T} * dS/dfS
                    struct_loads_ajp = body.get_struct_loads_ajp(scenario)
                    if struct_loads_ajp is not None:
                        for i in range(3):
                            struct_loads_ajp[i::3, ifunc] = -psi_array[i::ndof].astype(
                                body.dtype
                            )

                    # struct_flux_ajp = psi_S^{T} * dS/dfS
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

        self._eval_panel_length(adjoint=True)

        func_grad = []
        if self.tacs_proc:
            func_list = self.scenario_data[scenario].func_list
            dfdx = self.scenario_data[scenario].dfdx
            psi = self.scenario_data[scenario].psi

            for vec in dfdx:
                vec.zeroEntries()

            # get df/dx if the function is a structural function
            self.assembler.addDVSens(func_list, dfdx)
            self.assembler.addAdjointResProducts(psi, dfdx)

            # Add the values across processors - this is required to
            # collect the distributed design variable contributions
            for vec in dfdx:
                vec.beginSetValues(TACS.ADD_VALUES)
                vec.endSetValues(TACS.ADD_VALUES)

                # Set the gradients into the list of gradient objects
                func_grad.append(vec.getArray().copy())

        # Broadcast the gradients to all processors
        self.scenario_data[scenario].func_grad = self.comm.bcast(func_grad, root=0)

        return

    def get_coordinate_derivatives(self, scenario, bodies, step):
        """
        Evaluate gradients with respect to the structural node locations

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        bodies: list of :class:`~body.Body` objects
            The bodies in the model
        """

        if self.tacs_proc:
            func_list = self.scenario_data[scenario].func_list
            dfdXpts = self.scenario_data[scenario].dfdXpts
            psi = self.scenario_data[scenario].psi

            for vec in dfdXpts:
                vec.zeroEntries()

            # get df/dx if the function is a structural function
            self.assembler.addXptSens(func_list, dfdXpts)
            self.assembler.addAdjointResXptSensProducts(psi, dfdXpts)

            # Add the values across processors - this is required to collect
            # the distributed design variable contributions
            for vec in dfdXpts:
                vec.beginSetValues(TACS.ADD_VALUES)
                vec.endSetValues(TACS.ADD_VALUES)

            # Add the derivative to the body
            for body in bodies:
                struct_shape_term = body.get_struct_coordinate_derivatives(scenario)
                for ifunc, vec in enumerate(dfdXpts):
                    # Treat the sensitivity array as the same type as the body, even
                    # if there is a mismatch. This will allow TACS/FUNtoFEM to operate in
                    # mixed complex/real mode
                    array = vec.getArray()
                    struct_shape_term[:, ifunc] += array.astype(body.dtype)

        return

    @classmethod
    def create_from_bdf(
        cls,
        model,
        comm,
        nprocs,
        bdf_file,
        prefix="",
        callback=None,
        struct_options={},
        thermal_index=-1,
        override_rotx=False,
        debug=False,
        add_loads=True,  # whether it will try to add loads or not
        use_aitken=False,
    ):
        """
        Class method to create a TacsSteadyInterface instance using the pytacs BDF loader

        Parameters
        ----------
        model: :class:`FUNtoFEMmodel`
            The model class associated with the problem
        comm: MPI.comm
            MPI communicator (typically MPI_COMM_WORLD)
        bdf_file: str
            The BDF file name
        prefix: str
            Output prefix for .f5 files generated from TACS
        struct_DVs: List
            list of struct DV values for the built-in funtofem callback method
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
        Fvec = None
        panel_length_constraint = None
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

            # Set up constitutive objects and elements in pyTACS
            fea_assembler.initialize(callback)

            # get any constant loads for static case
            if add_loads:
                Fvec = addLoadsFromBDF(fea_assembler)
            # Fvec = None

            # make the panel length constraint object
            has_panel_length_funcs = any(
                [
                    cls.PANEL_LENGTH_CONSTR in comp_func.name
                    for comp_func in model.composite_functions
                ]
            )
            if has_panel_length_funcs:
                panel_length_constraint = fea_assembler.createPanelLengthConstraint(
                    "PanelLengthCon"
                )
                panel_length_constraint.addConstraint("PanelLength", dvIndex=0)
            else:
                panel_length_constraint = None

            # Retrieve the assembler from pyTACS fea_assembler object
            assembler = fea_assembler.assembler

            # Set the output file creator
            f5 = fea_assembler.outputViewer

        # Create the output generator
        gen_output = TacsOutputGenerator(prefix, f5=f5)

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
            if varsPerNode == 1:  # Thermal only
                thermal_index = 0
            elif varsPerNode == 4:  # Solid + thermal
                thermal_index = 3
            elif varsPerNode >= 7:  # Shell or beam + thermal
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
            override_rotx=override_rotx,
            Fvec=Fvec,
            debug=debug,
            panel_length_constraint=panel_length_constraint,
            use_aitken=use_aitken,
        )


class TacsOutputGenerator:
    def __init__(self, prefix, name="tacs_output_file", f5=None):
        """Store information about how to write TACS output files"""
        self.count = 0
        self.prefix = prefix
        self.name = name
        self.f5 = f5

    def __call__(self):
        """Generate the output from TACS"""

        if self.f5 is not None:
            file = self.name + "%03d.f5" % (self.count)
            filename = os.path.join(self.prefix, file)
            self.f5.writeToFile(filename)
        self.count += 1
        return

    def _deallocate(self):
        """free up memory before delete"""
        self.f5.__dealloc__()
