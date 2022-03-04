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
    def __init__(self, comm, tacs_comm, model=None):

        self.comm = comm
        self.tacs_comm = tacs_comm

        if model is not None:
            self.nfunc = model.count_functions()
        else:
            self.nfunc = 1

        # Set by the base class
        self.funclist = None
        self.functag = None

        self.struct_vars_all = {} # saved displacements and temps
        self.first_pass = True

        return

    def _initialize_variables(self, assembler=None, mat=None, pc=None, gmres=None,
                              struct_id=None, thermal_index=0):
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
        self.thermal_index = thermal_index
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
            self.struct_nnodes = len(self.struct_X_vec.getArray())//3

            self.struct_X = np.zeros(3*self.struct_nnodes, dtype=TACS.dtype)
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
        if self.tacs_proc:
            # Set the design variable values on the processors that
            # have an instance of TACSAssembler.
            xvec = self.assembler.createDesignVec()
            self.assembler.getDesignVars(xvec)
            xarray = xvec.getArray()

            if self.tacs_comm.rank == 0:
                for body in bodies:
                    if 'structural' in body.variables:
                        for i, var in enumerate(body.variables['structural']):
                            xarray[i] = var.value

            self.assembler.setDesignVars(xvec)

        return

    def set_functions(self, scenario, bodies):
        if self.tacs_proc:
            self.funclist = []
            self.functag = []
            for func in scenario.functions:
                if func.analysis_type != 'structural':
                    self.funclist.append(None)
                    self.functag.append(0)

                elif func.name.lower() == 'ksfailure':
                    if func.options:
                        ksweight = func.options['ksweight'] if 'ksweight' in func.options else 50.0
                    else:
                        ksweight = 50.0
                    self.funclist.append(functions.KSFailure(self.assembler, ksWeight=ksweight))
                    self.functag.append(1)

                elif func.name.lower() == 'compliance':
                    self.funclist.append(functions.Compliance(self.assembler))
                    self.functag.append(1)

                elif func.name.lower() == 'temperature':
                    self.funclist.append(functions.AverageTemperature(self.assembler, volume=self.vol))
                    self.functag.append(1)

                elif func.name.lower() == 'heatflux':
                    self.funclist.append(functions.HeatFlux(self.assembler))
                    self.functag.append(1)

                elif func.name == 'mass':
                    self.funclist.append(functions.StructuralMass(self.assembler))
                    self.functag.append(-1)

                else:
                    print('WARNING: Unknown function being set into TACS set to mass')
                    self.funclist.append(functions.StructuralMass(self.assembler))
                    self.functag.append(-1)

        return

    def get_functions(self, scenario, bodies):
        """
        Evaluate the structural functions of interest
        """
        if self.tacs_proc:
            feval = self.assembler.evalFunctions(self.funclist)

            for i, func in enumerate(scenario.functions):
                if func.analysis_type == 'structural':
                    func.value = feval[i]

        for func in scenario.functions:
            func.value = self.comm.bcast(func.value, root=0)

        return

    def get_function_gradients(self, scenario, bodies, offset):
        """
        Get the gradients of the functions of interest
        """
        for ifunc, func in enumerate(scenario.functions):
            for body in bodies:
                for vartype in body.variables:
                    if vartype == 'structural':
                        for i, var in enumerate(body.variables[vartype]):
                            if var.active:
                                if self.tacs_proc and self.tacs_comm.rank == 0:
                                    body.derivatives[vartype][offset + ifunc][i] = self.func_grad[ifunc][i]

                                # Do we have to broadcast for every single variable? This
                                # should be in an outer loop.
                                body.derivatives[vartype][offset + ifunc][i] = self.comm.bcast(body.derivatives[vartype][offset+ifunc][i], root=0)

        return

    def eval_gradients(self, scenario, bodies):
        """ Evaluate gradients with respect to structural design variables"""
        if self.tacs_proc:
            self.func_grad = []

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

        return

    def get_coordinate_derivatives(self, scenario, bodies, step):
        """ Evaluate gradients with respect to structural design variables"""

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

            for func, xptsens in enumerate(self.xptsenslist):
                for body in bodies:
                    if body.shape:
                        body.struct_shape_term[:, func] = xptsens.getArray()

        return

    def set_mesh(self, body):
        if self.tacs_proc:
            # Set the node locations
            struct_X = self.struct_X_vec.getArray()
            struct_X[:] = body.struct_X[:]
            self.assembler.setNodes(self.struct_X_vec)

        return

    def get_mesh(self, body):
        if self.tacs_proc:
            body.struct_X = self.struct_X_vec.getArray().copy()
            body.struct_nnodes = body.struct_X.size//3
            if self.struct_id is None:
                body.struct_id = None
            else:
                #body.struct_id = self.struct_id + 1
                body.struct_id = self.struct_id
        else:
            body.struct_nnodes = 0
            body.struct_X = np.array([], dtype=TACS.dtype)
            body.struct_id = np.array([], dtype=int)

        return

    def initialize(self, scenario, bodies):
        for body in bodies:
            if self.first_pass:
                self.get_mesh(body)

                # During the first pass, the transfer and thermal_transfer objects are
                # not defined, so used the flags
                if body.analysis_type == 'aeroelastic' or body.analysis_type == 'aerothermoelastic':
                    body.struct_disps = np.zeros(body.struct_nnodes*body.xfer_ndof, dtype=TACS.dtype)
                if body.analysis_type == 'aerothermal' or body.analysis_type == 'aerothermoelastic':
                    # FIXME need initial temperatures defined to pass to fluid solver
                    # currently initializing to the TACS reference temperature
                    body.struct_temps = np.ones(body.struct_nnodes*body.therm_xfer_ndof, dtype=TACS.dtype) * body.T_ref
                self.first_pass = False
            else:
                self.set_mesh(body)

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
                if body.transfer is not None:
                    for i in range(body.xfer_ndof):
                        ext_force_array[i::ndof] += body.struct_loads[i::body.xfer_ndof]
                if body.thermal_transfer is not None:
                    ext_force_array[self.thermal_index::ndof] += body.struct_heat_flux[:]

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
                if body.transfer is not None:
                    body.struct_disps = np.zeros(body.struct_nnodes*body.xfer_ndof,
                                                 dtype=TACS.dtype)
                    for i in range(body.xfer_ndof):
                        body.struct_disps[i::body.xfer_ndof] = ans_array[i::ndof]

                if body.thermal_transfer is not None:
                    body.struct_temps = np.zeros(body.struct_nnodes*body.therm_xfer_ndof,
                                                 dtype=TACS.dtype)
                    body.struct_temps[:] = ans_array[self.thermal_index::ndof] + body.T_ref
        else:
            for body in bodies:
                body.struct_disps = np.zeros(body.struct_nnodes*body.xfer_ndof, dtype=TACS.dtype)
                body.struct_temps = np.zeros(body.struct_nnodes*body.therm_xfer_ndof, dtype=TACS.dtype)

        return fail

    def post(self,scenario,bodies):
        if self.tacs_proc:
            self.struct_vars_all[scenario.id] = self.ans.getArray().copy()

            # export the f5 file
            try:
                self.post_export_f5()
            except:
                print("No f5 export set up")

    def initialize_adjoint(self,scenario,bodies):

        nfunctions = scenario.count_adjoint_functions()
        if self.tacs_proc:
            ans_array = self.ans.getArray()
            ans_array[:] = self.struct_vars_all[scenario.id]

            self.assembler.setVariables(self.ans)
            self.assembler.evalFunctions(self.funclist)

            # Extract the displacements and temperatures for each body
            ndof = self.assembler.getVarsPerNode()
            for body in bodies:
                if body.transfer is not None:
                    body.struct_disps = np.zeros(body.struct_nnodes*body.xfer_ndof, dtype=TACS.dtype)
                    for i in range(body.xfer_ndof):
                        body.struct_disps[i::body.xfer_ndof] = ans_array[i::ndof]

                if body.thermal_transfer is not None:
                    body.struct_temps = np.zeros(body.struct_nnodes*body.therm_xfer_ndof,
                                                 dtype=TACS.dtype)
                    body.struct_temps[:] = ans_array[self.thermal_index::ndof]

            # Assemble the transpose of the Jacobian matrix for the adjoint
            # computations. Note that for thermoelastic computations, the Jacobian
            # matrix is non-symmetric due to the temperature-deformation coupling.
            # The transpose must be used here to get the right result.
            alpha = 1.0 # Jacobian coefficient for the state variables
            beta = 0.0 # Jacobian coeff. for the first time derivative of the state variables
            gamma = 0.0 # Coeff. for the second time derivative of the state variables
            self.assembler.assembleJacobian(alpha, beta, gamma, self.res, self.mat,
                                            matOr=TACS.TRANSPOSE)
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
        else:
            for body in bodies:
                body.struct_disps = np.zeros(body.struct_nnodes*body.xfer_ndof, dtype=TACS.dtype)
                body.struct_temps = np.zeros(body.struct_nnodes*body.therm_xfer_ndof, dtype=TACS.dtype)

        return 0

    def iterate_adjoint(self, scenario, bodies, step):
        fail = 0

        for body in bodies:
            if body.transfer is not None:
                body.psi_S[:,:] = 0.0
            if body.thermal_transfer is not None:
                body.psi_T_S[:,:] = 0.0

        if self.tacs_proc:
            # Evaluate state variable sensitivities and scale to get right-hand side
            for func in range(len(self.funclist)):
                # Check if the function is a TACS function or not
                if self.functag[func] == -1:
                    continue

                # Copy values into the right-hand-side
                self.struct_rhs_vec.copyValues(self.svsenslist[func])
                struct_rhs_array = self.struct_rhs_vec.getArray()

                ndof = self.assembler.getVarsPerNode()
                for body in bodies:
                    # Form new right-hand side of structural adjoint equation using state
                    # variable sensitivites and the transformed temperature transfer
                    # adjoint variables
                    if body.transfer is not None:
                        for i in range(body.xfer_ndof):
                            struct_rhs_array[i::ndof] += body.struct_rhs[i::body.xfer_ndof, func]

                    if body.thermal_transfer is not None:
                        for i in range(body.therm_xfer_ndof):
                            struct_rhs_array[self.thermal_index::ndof] += \
                                body.struct_rhs_T[i::body.therm_xfer_ndof, func]

                # Zero the adjoint right-hand-side conditions at DOF locations
                # where the boundary conditions are applied. This is consistent with
                # the forward analysis where the forces/fluxes contributiosn are
                # zeroed at Dirichlet DOF locations.
                self.assembler.applyBCs(self.struct_rhs_vec)

                # Solve structural adjoint equation
                self.gmres.solve(self.struct_rhs_vec, self.psi_S[func])

                psi_S_array = self.psi_S[func].getArray()

                # Set the adjoint variables for each body
                for body in bodies:
                    if body.transfer is not None:
                        for i in range(body.xfer_ndof):
                            body.psi_S[i::body.xfer_ndof, func] = psi_S_array[i::ndof]

                    if body.thermal_transfer is not None:
                        body.psi_T_S[:, func] = psi_S_array[self.thermal_index::ndof]

        return fail

    def post_adjoint(self, scenario, bodies):
        if self.tacs_proc:
            self.eval_gradients(scenario, bodies)

    def adjoint_test(self, scenario, bodies, step=0, epsilon=1e-6):
        """
        For the structures problem, the input to the forward computation
        is the structural forces and the output is the displacements
        at the structural nodes.

        uS = uS(fS)

        The Jacobian of the forward code is

        J = d(uS)/d(fS).

        A finite-difference adjoint-vector product gives

        J*pS ~= (uS(fS + epsilon*pS) - uS(fS))/epsilon

        The adjoint code computes the product

        lam_fS = J^{T}*lam_uS

        As a result, we should have the identity:

        lam_fS^{T}*pS = lam_uS*J*pS ~ lam_uS^{T}*(uS(fS + epsilon*pS) - uS(fS))/epsilon
        """

        # Comptue one step of the forward solution
        self.initialize(scenario, bodies)
        self.iterate(scenario, bodies, step)
        self.post(scenario, bodies)

        # Store the output forces
        for ibody, body in enumerate(bodies):
            if body.transfer is not None:
                # Copy the structural displacements from the initial run
                # for later use
                body.struct_disps_copy = body.struct_disps.copy()

                # Set the the adjoint input to the structures
                body.struct_rhs = np.random.uniform(size=body.struct_rhs.shape)

        # Compute one step of the adjoint
        self.initialize_adjoint(scenario, bodies)
        self.iterate_adjoint(scenario, bodies, step)
        self.post_adjoint(scenario, bodies)

        # Perturb the structural loads
        adjoint_product = 0.0
        for ibody, body in enumerate(bodies):
            if body.transfer is not None:
                body.struct_loads_pert = np.random.uniform(size=body.struct_loads.shape)
                body.struct_loads += epsilon*body.struct_loads_pert

                # Compute the adjoint product. Note that the
                # negative sign is from convention due to the
                # presence of the negative sign in psi_F = -dLdfa
                adjoint_product += np.dot(body.psi_S[:, 0], body.struct_loads_pert)

        # Sum up the result across all processors
        adjoint_product = self.comm.allreduce(adjoint_product)

        # Run the perturbed aerodynamic simulation
        self.initialize(scenario, bodies)
        self.iterate(scenario, bodies, step)
        self.post(scenario, bodies)

        # Compute the finite-difference approximation
        fd_product = 0.0
        for ibody, body in enumerate(bodies):
            if body.transfer is not None:
                fd = (body.struct_disps - body.struct_disps_copy)/epsilon
                fd_product += np.dot(fd, body.struct_rhs[:, 0])

        # Compute the finite-differenc approximation
        fd_product = self.comm.allreduce(fd_product)

        if self.comm.rank == 0:
            print('TACS FUNtoFEM adjoint result:           ', adjoint_product)
            print('TACS FUNtoFEM finite-difference result: ', fd_product)

        return
