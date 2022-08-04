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

from tacs import TACS, functions
from .solver_interface import SolverInterface
import numpy as np


class TacsSteadyInterface(SolverInterface):
    """
    A base class to do coupled steady simulations with TACS
    """

    def __init__(self, comm, tacs_comm, model):

        self.comm = comm
        self.tacs_comm = tacs_comm

        # Get the list of active design variables
        self.variables = model.get_variables()

        # Get the structural variables
        self.struct_variables = []
        for var in self.variables:
            if var.analysis_type == "structural":
                self.struct_variables.append(var)

        self.nfunc = model.count_functions()
        self.funclist = None
        self.functag = None

        return

    def _initialize_variables(
        self,
        assembler=None,
        mat=None,
        pc=None,
        gmres=None,
        struct_id=None,
        thermal_index=0,
    ):
        """
        Initialize the variables required for analysis and
        optimization using TACS. This initialization takes in optional
        TACSMat, TACSPc and TACSKsm objects for the solution
        procedure. The assembler object must be defined on the subset
        of structural processors. On all other processors it must be
        None.
        """

        self.tacs_proc = False
        self.assembler = None
        self.res = None
        self.ans = None
        self.ext_force = None
        self.update = None
        self.mat = None
        self.pc = None
        self.gmres = None
        # self.thermal_index = thermal_index
        self.struct_id = struct_id

        self.struct_X_vec = None
        self.struct_nnodes = None
        self.struct_X = None
        self.dvsenslist = []
        self.svsenslist = []
        self.xptsenslist = []

        self.struct_rhs_vec = None
        self.psi_S = None

        self.ans_array = None
        self.func_grad = None
        self.vol = 1.0

        if assembler is not None:
            self.tacs_proc = True
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

            self.assembler = assembler
            self.res = assembler.createVec()
            self.ans = assembler.createVec()
            self.ext_force = assembler.createVec()
            self.update = assembler.createVec()

            # Get and set the structural node locations
            self.struct_X_vec = assembler.createNodeVec()
            assembler.getNodes(self.struct_X_vec)
            self.struct_nnodes = len(self.struct_X_vec.getArray()) // 3

            self.struct_X = np.zeros(3 * self.struct_nnodes, dtype=TACS.dtype)
            self.struct_X[:] = self.struct_X_vec.getArray()[:]
            self.dvsenslist = []
            self.svsenslist = []
            self.xptsenslist = []

            for i in range(self.nfunc):
                self.dvsenslist.append(assembler.createDesignVec())
                self.svsenslist.append(assembler.createVec())
                self.xptsenslist.append(assembler.createNodeVec())

            self.struct_rhs_vec = assembler.createVec()
            self.psi_S = []
            for ifunc in range(self.nfunc):
                self.psi_S.append(self.assembler.createVec())

            self.ans_array = assembler.createVec()
            self.func_grad = None

            # required for AverageTemp function, not sure if needed on
            # body level
            self.vol = 1.0

        return

    def set_variables(self, scenario, bodies):
        """
        Set the design variables into the TACSAssembler object.
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

    def set_functions(self, scenario, bodies):
        """
        Set and initialize the types of functions that will be evaluated based
        on the list of functions stored in the scenario.
        """

        if self.tacs_proc:
            self.funclist = []
            self.functag = []
            for func in scenario.functions:
                if func.analysis_type != "structural":
                    self.funclist.append(None)
                    self.functag.append(0)

                elif func.name.lower() == "ksfailure":
                    ksweight = 50.0
                    if func.options is not None and "ksweight" in func.options:
                        ksweight = func.options["ksweight"]
                    self.funclist.append(
                        functions.KSFailure(self.assembler, ksWeight=ksweight)
                    )
                    self.functag.append(1)

                elif func.name.lower() == "compliance":
                    self.funclist.append(functions.Compliance(self.assembler))
                    self.functag.append(1)

                elif func.name.lower() == "temperature":
                    self.funclist.append(
                        functions.AverageTemperature(self.assembler, volume=self.vol)
                    )
                    self.functag.append(1)

                elif func.name.lower() == "heatflux":
                    self.funclist.append(functions.HeatFlux(self.assembler))
                    self.functag.append(1)

                elif func.name == "mass":
                    self.funclist.append(functions.StructuralMass(self.assembler))
                    self.functag.append(-1)

                else:
                    print("WARNING: Unknown function being set into TACS set to mass")
                    self.funclist.append(functions.StructuralMass(self.assembler))
                    self.functag.append(-1)

        return

    def get_functions(self, scenario, bodies):
        """
        Evaluate the structural functions of interest.

        The functions are evaluated based on the values of the state variables set
        into the TACSAssembler object. These values are only available on the
        TACS processors, but these values are broadcast to all processors after the
        evaluation.
        """

        # Evaluate the list of the functions of interest
        feval = None
        if self.tacs_proc:
            feval = self.assembler.evalFunctions(self.funclist)

        # Broacast the list across all processors - not just structural procs
        feval = self.comm.bcast(feval, root=0)

        # Set the function values on all processors
        for i, func in enumerate(scenario.functions):
            if func.analysis_type == "structural":
                func.value = feval[i]

        return

    def eval_function_gradients(self, scenario, bodies, offset):
        """
        Get the gradients of the functions of interest
        """

        for ifunc, func in enumerate(scenario.functions):
            for i, var in enumerate(self.struct_variables):
                func.set_derivative(var, self.func_grad[ifunc][i])

        return

    def set_mesh(self, body):
        """
        Set the node locations for the structural model into the structural solver
        """

        if self.tacs_proc:
            # Set the node locations
            struct_X = self.struct_X_vec.getArray()
            struct_X[:] = body.struct_X[:]
            self.assembler.setNodes(self.struct_X_vec)

        return

    def get_mesh(self, body):
        """
        Get the node locations
        """
        if self.tacs_proc:
            body.struct_X = self.struct_X_vec.getArray().copy()
            body.struct_nnodes = body.struct_X.size // 3
            if self.struct_id is None:
                body.struct_id = None
            else:
                # body.struct_id = self.struct_id + 1
                body.struct_id = self.struct_id
        else:
            body.struct_nnodes = 0
            body.struct_X = np.array([], dtype=TACS.dtype)
            body.struct_id = np.array([], dtype=int)

        return

    def initialize_mesh(self, scenario, bodies):
        for body in bodies:
            body.initialize_struct_mesh(X)

    def initialize(self, scenario, bodies):
        """
        Initialize the solution


        """

        self.set_mesh(bodies)

        if self.tacs_proc:
            # Set the boundary conditions
            self.assembler.setBCs(self.ans)
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
            # Compute the residual from tacs self.res = K*u - f_internal
            self.assembler.assembleRes(self.res)

            # Add the external forces into a TACS vector that will be added to
            # the residual
            self.ext_force.zeroEntries()
            ext_force_array = self.ext_force.getArray()

            # Add the external load and heat fluxes on the structure
            ndof = self.assembler.getVarsPerNode()
            for body in bodies:
                struct_loads = body.get_struct_loads(scenario)
                if struct_loads is not None:
                    for i in range(3):
                        ext_force_array[i::ndof] += struct_loads[i::3]

                struct_flux = body.get_struct_heat_flux(scenario)
                if struct_flux is not None:
                    ext_force_array[self.thermal_index :: ndof] += struct_flux[:]

            # Zero the contributions at the DOF associated with boundary
            # conditions so that it doesn't interfere with Dirichlet BCs
            self.assembler.applyBCs(self.ext_force)

            # Add the contribution to the residuals from the external forces
            self.res.axpy(-1.0, self.ext_force)

            # Solve for the update
            self.gmres.solve(self.res, self.update)

            # Apply the update to the solution vector and reset the boundary condition
            # data so that it is precisely statisfied
            self.ans.axpy(-1.0, self.update)
            self.assembler.setBCs(self.ans)

            # Set the variables into the assembler object
            self.assembler.setVariables(self.ans)

            # Extract displacements and temperatures for each body
            ans_array = self.ans.getArray()
            for body in bodies:
                struct_disps = body.get_struct_disps(scenario)
                if struct_disps is not None:
                    for i in range(3):
                        struct_disps[i::3] = ans_array[i::ndof]

                # Set the structural temperature
                struct_temps = body.get_struct_temps(scenario)
                if struct_temps is not None:
                    struct_temps[:] = ans_array[self.thermal_index :: ndof] + body.T_ref

        return fail

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
        if self.tacs_proc:
            self.struct_vars_all[scenario.id] = self.ans.getArray().copy()

            # export the f5 file
            try:
                self.post_export_f5()
            except:
                print("No f5 export set up")

    def initialize_adjoint(self, scenario, bodies):
        """
        Initialize the solver for adjoint computations.

        This code computes the transpose of the Jacobian matrix dS/du^{T}, and factorizes
        it. For structural problems, the Jacobian matrix is symmetric, however for coupled
        thermoelastic problems, the Jacobian matrix is non-symmetric.

        The code also computes the derivative of the structural functions of interest with
        respect to the state variables df/du. These derivatives are stored in the list
        self.funclist that stores svsens = -df/du. Note that the negative sign applied here.

        Parameters
        ----------
        scenario: :class:`~scenario.Scenario`
            The current scenario
        bodies: :class:`~body.Body`
            list of FUNtoFEM bodies
        """

        nfunctions = scenario.count_adjoint_functions()
        if self.tacs_proc:
            ans_array = self.ans.getArray()
            ans_array[:] = self.struct_vars_all[scenario.id]

            self.assembler.setVariables(self.ans)
            self.assembler.evalFunctions(self.funclist)

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
            feval = self.assembler.evalFunctions(self.funclist)

            # Zero the vectors in the sensitivity list
            for svsens in self.svsenslist:
                svsens.zeroEntries()

            # Compute the derivative of the function with respect to the
            # state variables
            self.assembler.addSVSens(self.funclist, self.svsenslist, 1.0, 0.0, 0.0)

            # Evaluate state variable sensitivities and scale to get right-hand side
            for func in range(len(self.funclist)):
                if self.functag[func] == 1:
                    # Scale the derivative of the function w.r.t. the
                    # state variables by -1 since this will appear on
                    # the right-hand-side of the adjoint
                    self.svsenslist[func].scale(-1.0)
                else:
                    self.svsenslist[func].zeroEntries()

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
            # Evaluate state variable sensitivities and scale to get right-hand side
            for ifunc in range(len(self.funclist)):
                # Check if the function requires an adjoint computation or not
                if self.functag[ifunc] == -1:
                    continue

                # Copy values into the right-hand-side
                # struct_rhs_vec = - df/duS^{T}
                self.struct_rhs_vec.copyValues(self.svsenslist[ifunc])
                struct_rhs_array = self.struct_rhs_vec.getArray()

                ndof = self.assembler.getVarsPerNode()
                for body in bodies:
                    # Form new right-hand side of structural adjoint equation using state
                    # variable sensitivites and the transformed temperature transfer
                    # adjoint variables. Here we use the adjoint-Jacobian products from the
                    # structural displacements and structural temperatures.
                    struct_disps_ajp = body.get_struct_disps_ajp(scenario)
                    if struct_disps_ajp is not None:
                        for i in range(3):
                            struct_rhs_array[i::ndof] -= struct_disps_ajp[:, ifunc]

                    struct_temps_ajp = body.get_struct_temps_ajp(scenario)
                    if struct_temps_ajp is not None:
                        struct_rhs_array[
                            self.thermal_index :: ndof
                        ] -= struct_temps_ajp[:, ifunc]

                # Zero the adjoint right-hand-side conditions at DOF locations
                # where the boundary conditions are applied. This is consistent with
                # the forward analysis where the forces/fluxes contributiosn are
                # zeroed at Dirichlet DOF locations.
                self.assembler.applyBCs(self.struct_rhs_vec)

                # Solve structural adjoint equation
                self.gmres.solve(self.struct_rhs_vec, self.psi_S[ifunc])

                psi_S_array = self.psi_S[ifunc].getArray()

                # Set the adjoint-Jacobian products for each body
                for body in bodies:
                    # Compute the structural loads adjoint-Jacobian product. Here
                    # S(u, fS, hS) = r(u) - fS - hS, so dS/dfS = -I and dS/dhS = -I
                    # struct_loads_ajp = psi_S^{T} * dS/dfS
                    struct_loads_ajp = body.get_struct_loads_ajp(scenario)
                    if struct_loads_ajp is not None:
                        for i in range(3):
                            struct_loads_ajp[:, ifunc] = -psi_S_array[i::ndof]

                    # struct_flux_ajp = psi_S^{T} * dS/dfS
                    struct_flux_ajp = body.get_struct_flux_ajp(scenario)
                    if struct_flux_ajp is not None:
                        struct_flux_ajp[:, ifunc] = -psi_S_array[
                            self.thermal_index :: body.xfer_ndof
                        ]

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

        self.func_grad = []
        if self.tacs_proc:
            for func, dvsens in enumerate(self.dvsenslist):
                dvsens.zeroEntries()

            # get df/dx if the function is a structural function
            self.assembler.addDVSens(self.funclist, self.dvsenslist)
            self.assembler.addAdjointResProducts(self.psi_S, self.dvsenslist)

            # Add the values across processors - this is required to
            # collect the distributed design variable contributions
            for func, dvsens in enumerate(self.dvsenslist):
                dvsens.beginSetValues(TACS.ADD_VALUES)
                dvsens.endSetValues(TACS.ADD_VALUES)

                self.func_grad.append(dvsens.getArray().copy())

        # Broadcast the gradients to all processors
        self.func_grad = self.comm.bcast(self.func_grad, root=0)

        return

    def get_coordinate_derivatives(self, scenario, bodies, step):
        """Evaluate gradients with respect to structural design variables"""

        if self.tacs_proc:
            for func, xptsens in enumerate(self.xptsenslist):
                xptsens.zeroEntries()

            # get df/dx if the function is a structural function
            self.assembler.addXptSens(self.funclist, self.xptsenslist)
            self.assembler.addAdjointResXptSensProducts(self.psi_S, self.xptsenslist)

            # Add the values across processors - this is required to collect
            # the distributed design variable contributions
            for func, xptsens in enumerate(self.xptsenslist):
                xptsens.beginSetValues(TACS.ADD_VALUES)
                xptsens.endSetValues(TACS.ADD_VALUES)

            # Add the derivative to the body
            for body in bodies:
                struct_shape_term = body.get_struct_coordinate_derivatives()
                for ifunc, xptsens in enumerate(self.xptsenslist):
                    struct_shape_term[:, ifunc] += xptsens.getArray()

        return
