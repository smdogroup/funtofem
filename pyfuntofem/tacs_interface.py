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

from __future__ import print_function, division

from tacs             import TACS, functions
from .solver_interface import SolverInterface
import numpy as np


class TacsSteadyInterface(SolverInterface):
    """
    A base class to do coupled steady simulations with TACS
    """
    def __init__(self, comm, tacs_comm, model=None):

        # Needs to the set in the child class
        self.tacs = None
        self.res = None
        self.ans = None
        self.mat = None
        self.pc = None
        self.struct_X_vec = None
        self.struct_nnodes = None
        self.struct_X = None
        self.svsens = None
        self.struct_rhs_vec = None
        self.psi_S_vec = None
        self.psi_S = None
        self.ans_array = None
        self.comm = comm
        self.func_grad = None

        # Set by the base class
        self.gmres = None
        self.funclist = None
        self.functag = None
        self.num_components = None
        self.nfunc = model.count_functions() if model else 1

        self.dof = 6

        self.struct_disps_all = {}
        self.first_pass = True

    def set_mesh(self,body):

        if self.tacs_proc:
            struct_X = self.struct_X_vec.getArray()
            struct_X[:] = body.struct_X[:]

            self.tacs.setNodes(self.struct_X_vec)

            alpha = 1.0
            beta = 0.0
            gamma = 0.0
            self.tacs.assembleJacobian(alpha,beta,gamma,self.res,self.mat)
            self.pc.factor()

    def get_mesh(self,body):
        if self.tacs_proc:
            body.struct_X =  self.struct_X_vec.getArray().copy()
            body.struct_nnodes = int(body.struct_X.size/3)
        else:
            body.struct_nnodes = 0
            body.struct_X = np.array([],dtype=TACS.dtype)

    def set_variables(self,scenario,bodies):
        #FIXME only one body
        if self.tacs_proc:
            for body in bodies:
                if 'structural' in body.variables:

                    self.num_components = 0
                    x = np.zeros(len(body.variables['structural']),dtype=TACS.dtype)

                    for i,var in enumerate(body.variables['structural']):
                        x[i] = var.value
                        self.num_components += 1

                    self.tacs.setDesignVars(x)

                    alpha = 1.0
                    beta = 0.0
                    gamma = 0.0
                    self.tacs.assembleJacobian(alpha,beta,gamma,self.res,self.mat)
                    self.pc.factor()

    def set_functions(self,scenario,bodies):
        if self.tacs_proc:
            self.funclist = []
            self.functag = []
            for func in scenario.functions:
                if func.analysis_type != 'structural':
                    # use mass as a placeholder for nonstructural functions
                    self.funclist.append(functions.StructuralMass(self.tacs))
                    self.functag.append(0)

                elif func.name.lower() == 'ksfailure':
                    if func.options:
                        ksweight = func.options['ksweight'] if 'ksweight' in func.options else 50.0
                    else:
                        ksweight = 50.0
                    self.funclist.append(functions.KSFailure(self.tacs, ksweight))
                    self.functag.append(1)

                elif func.name.lower() == 'compliance':
                    self.funclist.append(functions.Compliance(self.tacs))
                    self.functag.append(1)

                elif func.name == 'mass':
                    self.funclist.append(functions.StructuralMass(self.tacs))
                    self.functag.append(-1)

                else:
                    print('WARNING: Unknown function being set into TACS set to mass')
                    self.funclist.append(functions.StructuralMass(self.tacs))
                    self.functag.append(-1)

    def get_functions(self,scenario,bodies):
        if self.tacs_proc:
            feval = self.tacs.evalFunctions(self.funclist)
            for i, func in enumerate(scenario.functions):
                if func.analysis_type == 'structural':
                    func.value = feval[i]
        for func in scenario.functions:
            func.value = self.comm.bcast(func.value,root=0)

    def get_function_gradients(self,scenario,bodies,offset):
        for ifunc, func in enumerate(scenario.functions):
            for body in bodies:
                for vartype in body.variables:
                    if vartype == 'structural':
                        for i, var in enumerate(body.variables[vartype]):
                            if var.active:
                                if self.tacs_proc:
                                    body.derivatives[vartype][offset+ifunc][i] = self.func_grad[ifunc][i]
                                body.derivatives[vartype][offset+ifunc][i] = self.comm.bcast(body.derivatives[vartype][offset+ifunc][i],root=0)

    def eval_gradients(self,scenario):
        """ Evaluate gradients with respect to structural design variables"""
        if self.tacs_proc:
            self.func_grad = []
            dvsens = np.zeros(self.num_components)
            for func in scenario.functions:

                # get df/dx if the function is a structural function
                self.tacs.evalDVSens(self.funclist[func], dvsens)
                if self.functag[func] == 0:
                    dvsens.zeroEntries()

                # get psi_S * dS/dx if a structural function that requires an adjoint
                adjResProduct = np.zeros(dvsens.size)
                if self.functag[func] > -1:
                    psi_S_array = self.psi_S_vec.getArray()
                    psi_S_array[:] = self.psi_S[:,func]
                    self.tacs.evalAdjointResProduct(self.psi_S_vec, adjResProduct)

                self.func_grad.append(dvsens[:] + adjResProduct[:])
        return

    def get_coordinate_derivatives(self,scenario,bodies,step):
        """ Evaluate gradients with respect to structural design variables"""
        #FIXME assuming only body
        if bodies[0].shape:
            if self.tacs_proc:

                fXptSens = self.tacs.createNodeVec()
                adjResProduct_vec = self.tacs.createNodeVec()

                for func,_ in enumerate(scenario.functions):

                    # get df/dx if the function is a structural function
                    self.tacs.evalXptSens(self.funclist[func], fXptSens)
                    df = fXptSens.getArray()
                    if self.functag[func] == 0:
                        df[:] = 0.0

                    # get psi_S * dS/dx if a structural function that requires an adjoint
                    if self.functag[func] > -1:
                        psi_S_array = self.psi_S_vec.getArray()
                        psi_S_array[:] = self.psi_S[:,func]
                        self.tacs.evalAdjointResXptSensProduct(self.psi_S_vec, adjResProduct_vec)
                        adjResProduct = adjResProduct_vec.getArray()
                    else:
                        adjResProduct = np.zeros(df.size,dtype=TACS.dtype)

                    bodies[0].struct_shape_term[:,func] += df[:] + adjResProduct[:]

    def initialize(self,scenario,bodies):
        if self.first_pass:
            for body in bodies:
                self.get_mesh(body)
            self.first_pass = False
        else:
            for body in bodies:
                self.set_mesh(body)
        return 0

    def iterate(self,scenario,bodies,step):
        fail = 0
        if self.tacs_proc:
            # Compute the residual from tacs self.res = K*u
            self.tacs.assembleRes(self.res)
            res_array = self.res.getArray()
            res_array[:] = 0.0

            for body in bodies:
                # Set loads on structure
                #FIXME: set into only body indices
                for i in range(body.xfer_ndof):
                    res_array[i::self.dof] += body.struct_loads[i::body.xfer_ndof]

            # Add the aerodynamic loads in the residual
            self.tacs.applyBCs(self.res)

            # Solve
            self.pc.applyFactor(self.res, self.ans)
            self.tacs.setVariables(self.ans)

            ans_array = self.ans.getArray()

            # Extract displacements
            for body in bodies:
                body.struct_disps = np.zeros(body.struct_nnodes*body.xfer_ndof,dtype=TACS.dtype)
                for i in range(body.xfer_ndof):
                    body.struct_disps[i::body.xfer_ndof] = ans_array[i::self.dof]
        else:
            for body in bodies:
                body.struct_disps = np.zeros(body.struct_nnodes*body.xfer_ndof,dtype=TACS.dtype)
        return fail

    def post(self,scenario,bodies):
        if self.tacs_proc:
            self.struct_disps_all[scenario.id]=self.ans.getArray().copy()

            # export the f5 file
            try:
                self.post_export_f5()
            except:
                print("No f5 export set up")

    def initialize_adjoint(self,scenario,bodies):
        if self.tacs_proc:
            ans_array = self.ans.getArray()
            ans_array[:] = self.struct_disps_all[scenario.id]

            self.tacs.setVariables(self.ans)
            self.tacs.evalFunctions(self.funclist)

            for body in bodies:
                struct_disps = np.zeros(body.struct_nnodes*body.xfer_ndof,dtype=TACS.dtype)
                for i in range(body.xfer_ndof):
                    struct_disps[i::body.xfer_ndof] = ans_array[i::self.dof]
        else:
            for body in bodies:
                struct_disps = np.zeros(body.struct_nnodes*body.xfer_ndof)

        return 0

    def iterate_adjoint(self,scenario,bodies,step):
        fail = 0
        for body in bodies:
            body.psi_S[:,:] = 0.0

        if self.tacs_proc:
            self.tacs.evalFunctions(self.funclist)
            for func,_ in enumerate(self.funclist):
                if self.functag[func] == -1:
                    break
                # Evaluate state variable sensitivities and scale to get right-hand side
                self.tacs.evalSVSens(self.funclist[func], self.svsens)
                if self.functag[func] == 1:

                    self.svsens.scale(-1.0)
                else:
                    self.svsens.scale(0.0)

                self.struct_rhs_vec.copyValues(self.svsens)
                struct_rhs_array = self.struct_rhs_vec.getArray()

                for body in bodies:
                    # Form new right-hand side of structural adjoint equation using state
                    # variable sensitivites and the transformed displacement transfer
                    # adjoint variables
                    #FIXME index slice for body
                    for i in range(body.xfer_ndof):
                        struct_rhs_array[i::self.dof] += body.struct_rhs[i::body.xfer_ndof,func]

                    # Solve structural adjoint equation
                    self.tacs.applyBCs(self.struct_rhs_vec)
                    self.gmres.solve(self.struct_rhs_vec, self.psi_S_vec)
                    psi_S_6dof = self.psi_S_vec.getArray()
                    self.psi_S[:,func] = psi_S_6dof[:]

                    # resize to the size of the structural force vector
                    for i in range(body.xfer_ndof):
                        body.psi_S[i::body.xfer_ndof,func] = self.psi_S[i::self.dof,func]

        return fail

    def post_adjoint(self,scenario,bodies):
        if self.tacs_proc:
            self.eval_gradients(scenario)

