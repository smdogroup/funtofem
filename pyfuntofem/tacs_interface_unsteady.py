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

from tacs             import TACS, functions
from tacs_builder     import TACSBodyType
from .solver_interface import SolverInterface

import numpy as np

class TacsUnsteadyInterface(SolverInterface):
    """
    A base class to do coupled unsteady simulations with TACS
    """

    def __init__(self, integrator_options, model=None,ndof=8):

        if self.tacs_proc:
            self.func_grad = None
            self.psi_S_vec = None
            self.struct_rhs_vec = None
            self.funclist = None
            self.functag = None
            self.ndvs = None
            self.nfunc = model.count_functions() if model else 1

            self.createTACS(integrator_options,model,ndof=ndof)

            self.ans = self.tacs.createVec()
            self.bvec_forces = self.tacs.createVec()

            self.tacs_flex_bodies = []

            for body in self.tacs_body_list:
                if body.btype == TACSBodyType.FLEXIBLE or body.btype == TACSBodyType.SOLID:
                    self.tacs_flex_bodies.append(body)
        for ibody,body in enumerate(model.bodies):
            self.get_mesh(ibody,body)

    def createTACS(self,options,model,ndof=8):
        """
        Set up TACS and TACSIntegator.
        """
        # grab the tacs instance from the builder
        self.tacs = self.builder.getTACS(options['ordering'], TACS.PY_DIRECT_SCHUR,ndof=ndof)


        # Things for configuring time marching
        self.integrator = {}
        for scenario in model.scenarios:
            self.integrator[scenario.id] = self.createIntegrator(self.tacs, options)

        # Control F5 output
            if self.builder.rigid_viz == 1:
                flag = (TACS.ToFH5.NODES|
                        TACS.ToFH5.DISPLACEMENTS|
                        TACS.ToFH5.EXTRAS)
                rigidf5 = TACS.ToFH5(self.tacs, TACS.PY_RIGID, flag)
                self.integrator[scenario.id].setRigidOutput(rigidf5)

            if self.builder.shell_viz == 1:
                flag = (TACS.ToFH5.NODES|
                        TACS.ToFH5.DISPLACEMENTS|
                        TACS.ToFH5.STRAINS|
                        TACS.ToFH5.STRESSES|
                        TACS.ToFH5.EXTRAS)
                shellf5 = TACS.ToFH5(self.tacs, TACS.PY_SHELL, flag)
                self.integrator[scenario.id].setShellOutput(shellf5)

            if self.builder.beam_viz == 1:
                flag = (TACS.ToFH5.NODES|
                        TACS.ToFH5.DISPLACEMENTS|
                        TACS.ToFH5.STRAINS|
                        TACS.ToFH5.STRESSES|
                        TACS.ToFH5.EXTRAS)
                beamf5 = TACS.ToFH5(self.tacs, TACS.PY_BEAM, flag)
                self.integrator[scenario.id].setBeamOutput(beamf5)

            if self.builder.solid_viz == 1:
                flag = (TACS.ToFH5.NODES|
                        TACS.ToFH5.DISPLACEMENTS|
                        TACS.ToFH5.STRAINS|
                        TACS.ToFH5.STRESSES|
                        TACS.ToFH5.EXTRAS)
                solidf5 = TACS.ToFH5(self.tacs, TACS.PY_SOLID, flag)
                self.integrator[scenario.id].setSolidOutput(solidf5)

        # store the reference to body list after initializations are complete
        self.tacs_body_list  = self.builder.body_list

        return

    def createIntegrator(self, tacs, options):
        """
        Create the Integrator (solver) and configure it
        """

        end_time = options['steps'] * options['step_size']

        # Create an integrator for TACS
        if options['integrator'] == 'BDF':
            integrator = TACS.BDFIntegrator(tacs,
                                            options['start_time'], end_time,
                                            options['steps'],
                                            options['integration_order'])

        # Set other parameters for integration
        integrator.setRelTol(options['solver_rel_tol'])
        integrator.setAbsTol(options['solver_abs_tol'])
        integrator.setMaxNewtonIters(options['max_newton_iters'])
        integrator.setUseFEMat(options['femat'],options['ordering'])
        #integrator.setPrintLevel(options['print_level'])
        integrator.setOutputFrequency(options['output_freq'])
        return integrator


    def get_mesh(self,ibody,body):
        if self.tacs_proc:
            X_vec = self.tacs.createNodeVec()
            self.tacs.getNodes(X_vec)
            X = X_vec.getArray()

            tacs_body = self.tacs_flex_bodies[ibody]
            nnodes = len(tacs_body.dist_nodes)
            struct_X = np.zeros(3*nnodes,dtype=TACS.dtype)
            for i,n in enumerate(tacs_body.dist_nodes):
                struct_X[3*i:3*(i+1)] = X[3*n:3*(n+1)]
        else:
            nnodes = np.array(0)
            struct_X = np.array([],dtype=TACS.dtype)

        body.struct_nnodes = nnodes
        body.struct_X = struct_X
        body.struct_disps = np.zeros(3*nnodes,dtype=TACS.dtype)

        return

    def set_mesh(self,ibody,body):
        if self.tacs_proc and body.shape:
            # Get the node vector from TACS
            Xpts = self.tacs.createNodeVec()
            self.tacs.getNodes(Xpts)
            X = Xpts.getArray()

            # Put the new coordinates in the node vector then set back into TACS
            for i,n in enumerate(self.tacs_flex_bodies[ibody].dist_nodes):
                X[3*n:3*(n+1)] = body.struct_X[3*i:3*(i+1)]

            self.tacs.setNodes(Xpts)

    def set_variables(self,scenario,bodies):
        self.ndvs = 0

        x = []
        for body in bodies:
            if 'structural' in body.variables:
                for var in body.variables['structural']:
                    x.append(var.value)
                    self.ndvs += 1

        if self.tacs_proc:
            self.tacs.setDesignVars(np.array(x,dtype=TACS.dtype))

    def set_functions(self,scenario,bodies):
        if self.tacs_proc:
            self.funclist = []
            self.functag = []
            for func in scenario.functions:
                if func.analysis_type != 'structural':
                    self.funclist.append(None)
                    self.functag.append(0)

                elif func.name.lower() == 'compliance':
                    self.funclist.append(functions.Compliance(self.tacs))
                    self.functag.append(1)

                elif func.name.lower() == 'ksfailure':
                    if func.options:
                        ksweight = func.options['ksweight'] if 'ksweight' in func.options else 50.0
                    else:
                        ksweight = 50.0
                    self.funclist.append(functions.KSFailure(self.tacs, ksweight))
                    self.functag.append(1)

                elif func.name == 'mass':
                    self.functag.append(-1)

                else:
                    print('WARNING: Unknown function being set into TACS set to mass')
                    self.functag.append(-1)

            func = scenario.functions[0]
            self.integrator[scenario.id].setFunctions(self.funclist,self.ndvs,func.start,func.stop)
            self.integrator[scenario.id].evalFunctions(self.funclist)

    def get_functions(self,scenario,bodies):
        if self.tacs_proc:
            feval = self.integrator[scenario.id].evalFunctions(self.funclist)
            for i, func in enumerate(scenario.functions):
                if func.analysis_type == 'structural' and func.adjoint:
                    func.value = feval[i]
                if func.analysis_type == 'structural' and func.name == 'mass':
                    funclist = [functions.StructuralMass(self.tacs)]
                    value = self.tacs.evalFunctions(funclist)
                    func.value = value[0]
        for func in scenario.functions:
            func.value = self.comm.bcast(func.value,root=0)
        return functions

    def get_function_gradients(self,scenario,bodies,offset):
        for ifunc, func in enumerate(scenario.functions):
            for body in bodies:
                for vartype in body.variables:
                    if vartype == 'structural':
                        for i, var in enumerate(body.variables[vartype]):
                            if var.active:
                                if self.comm.Get_rank()==0:
                                    body.derivatives[vartype][offset+ifunc][i] = self.func_grad[ifunc][i]
                                body.derivatives[vartype][offset+ifunc][i] = self.comm.bcast(body.derivatives[vartype][offset+ifunc][i],root=0)

    def eval_gradients(self,scenario):
        """ Evaluate gradients with respect to structural design variables"""
        if self.tacs_proc:
            dfdx = np.zeros(len(self.funclist)*self.ndvs,dtype=TACS.dtype)
            self.integrator[scenario.id].getGradient(dfdx)

            self.func_grad = []
            for func in range(len(self.funclist)):
                self.func_grad.append(dfdx[func*self.ndvs:(func+1)*self.ndvs].copy())

            for func in scenario.functions:
                if func.analysis_type == 'structural' and func.name == 'mass':
                    funclist = [functions.StructuralMass(self.tacs)]
                    dvsens = np.zeros(self.num_components)
                    self.tacs.evalDVSens(funclist[0], dvsens)

                    self.func_grad.append(dvsens.copy())

        return

    def get_coordinate_derivatives(self,scenario,bodies,step):
        """ Evaluate gradients with respect to structural design variables"""

        if self.tacs_proc:
            fXptSens_vec = self.tacs.createNodeVec()

            # get the list of indices for this body
            for ibody, body in enumerate(bodies):
                if body.shape:
                    nodes = self.tacs_flex_bodies[ibody].dist_nodes

                    # TACS does the summation over steps internally, so only get the values when evaluating initial conditions
                    if step == 0:

                        # find the coordinate derivatives for each function
                        for nfunc, func in enumerate(scenario.functions):
                            if func.adjoint:
                                fXptSens_vec = self.integrator[scenario.id].getXptGradient(nfunc)
                            elif func.name == 'mass':
                                tacsfunc = functions.StructuralMass(self.tacs)
                                self.tacs.evalXptSens(tacsfunc, fXptSens_vec)

                            # pick out the points for this body
                            fxptSens = fXptSens_vec.getArray()
                            for i,n in enumerate(nodes):
                                body.struct_shape_term[3*i:3*i+3,nfunc] += fxptSens[3*n:3*n+3]

        return

    def initialize(self,scenario,bodies):
        # Update the meshes
        if self.tacs_proc:
            for ibody, body in enumerate(bodies):
                self.set_mesh(ibody,body)
                self.get_mesh(ibody,body)
            self.integrator[scenario.id].iterate(0)

        return 0

    def iterate(self, scenario,bodies,step):
        fail = 0
        # Set loads into the force bvec
        if self.tacs_proc:
            load_vector = self.bvec_forces.getArray()
            for ibody, (tacs_body, body) in enumerate(zip(self.tacs_flex_bodies,bodies)):
                for i,n in enumerate(tacs_body.dist_nodes):
                    load_vector[tacs_body.dof*n:tacs_body.dof*n+body.xfer_ndof] = body.struct_loads[body.xfer_ndof*i:body.xfer_ndof*i+body.xfer_ndof]

            # take a step in the structural solver
            self.integrator[scenario.id].iterate(step, self.bvec_forces)

            # Extract the structural displacements
            self.tacs.getVariables(self.ans)
            disp_vector = self.ans.getArray()

            for ibody, (tacs_body, body) in enumerate(zip(self.tacs_flex_bodies,bodies)):
                for i,n in enumerate(tacs_body.dist_nodes):
                    body.struct_disps[body.xfer_ndof*i:body.xfer_ndof*i+body.xfer_ndof] = disp_vector[tacs_body.dof*n:tacs_body.dof*n+body.xfer_ndof]
        else:
            for body in bodies:
                body.struct_disps = np.zeros(body.struct_nnodes*body.xfer_ndof, dtype=TACS.dtype)

        return fail

    def initialize_adjoint(self,scenario,bodies):
        if self.tacs_proc:
            self.struct_rhs_vec = []
            for func in range(len(self.funclist)):
                self.struct_rhs_vec.append(self.tacs.createVec())
        return 0

    def post_adjoint(self,scenario,bodies):
        self.eval_gradients(scenario)

    def set_states(self,scenario,bodies,step):
        if self.tacs_proc:
            _, self.ans, _, _ = self.integrator[scenario.id].getStates(step)
            disps = self.ans.getArray()
            for ibody, (tacs_body, body) in enumerate(zip(self.tacs_flex_bodies,bodies)):
                for i,n in enumerate(tacs_body.dist_nodes):
                    body.struct_disps[body.xfer_ndof*i:body.xfer_ndof*i+body.xfer_ndof] = disps[tacs_body.dof*n:tacs_body.dof*n+body.xfer_ndof]

    def iterate_adjoint(self, scenario, bodies, step):
        fail = 0

        for body in bodies:
            body.psi_S[:,:] = 0.0

        # put the body rhs's into the TACS bvec
        if self.tacs_proc:
            for func in range(len(self.funclist)):
                rhs_func = self.struct_rhs_vec[func].getArray()
                for ibody, (tacs_body, body) in enumerate(zip(self.tacs_flex_bodies,bodies)):
                    for i,n in enumerate(tacs_body.dist_nodes):
                        rhs_func[tacs_body.dof*n:tacs_body.dof*n+body.xfer_ndof] = body.struct_rhs[body.xfer_ndof*i:body.xfer_ndof*i+body.xfer_ndof,func]

            # take a reverse step in the adjoint solver
            self.integrator[scenario.id].initAdjoint(step)
            self.integrator[scenario.id].iterateAdjoint(step, self.struct_rhs_vec)
            self.integrator[scenario.id].postAdjoint(step)

            # pull the adjoints out of the TACS bodies
            for func in range(len(self.funclist)):
                self.psi_S_vec = self.integrator[scenario.id].getAdjoint(step, func)
                psi_func = self.psi_S_vec.getArray()
                for ibody, (tacs_body, body) in enumerate(zip(self.tacs_flex_bodies,bodies)):
                    for i,n in enumerate(tacs_body.dist_nodes):
                        body.psi_S[body.xfer_ndof*i:body.xfer_ndof*i+body.xfer_ndof,func] = psi_func[tacs_body.dof*n:tacs_body.dof*n+body.xfer_ndof]
        return fail

    def step_pre(self, scenario,bodies,step):
        fail = 0
        return fail

    def step_solver(self, scenario,bodies,step,fsi_subiter):
        fail = 0
        # Set loads into the force bvec
        if self.tacs_proc:
            load_vector = self.bvec_forces.getArray()
            for ibody, (tacs_body, body) in enumerate(zip(self.tacs_flex_bodies,bodies)):
                for i,n in enumerate(tacs_body.dist_nodes):
                    load_vector[tacs_body.dof*n:tacs_body.dof*n+body.xfer_ndof] = body.struct_loads[body.xfer_ndof*i:body.xfer_ndof*i+body.xfer_ndof]

            # take a step in the structural solver
            self.integrator[scenario.id].iterate(step,self.bvec_forces)

            # Extract the structural displacements
            self.tacs.getVariables(self.ans)
            disp_vector = self.ans.getArray()

            for ibody, (tacs_body, body) in enumerate(zip(self.tacs_flex_bodies,bodies)):
                for i,n in enumerate(tacs_body.dist_nodes):
                    body.struct_disps[body.xfer_ndof*i:body.xfer_ndof*i+body.xfer_ndof] = disp_vector[tacs_body.dof*n:tacs_body.dof*n+body.xfer_ndof]
        else:
            for body in bodies:
                body.struct_disps = np.zeros(body.struct_nnodes*body.xfer_ndof)

        return fail

    def step_post(self, scenario,bodies,step):
        return 0
