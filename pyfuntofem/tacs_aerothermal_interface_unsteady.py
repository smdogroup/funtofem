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
from .solver_interface import SolverInterface

import numpy as np

class TacsUnsteadyAerothermalInterface(SolverInterface):
    """
    A base class to do coupled unsteady simulations with TACS
    """

    def __init__(self, integrator_options, comm, tacs_comm, model=None, ndof=1):
        if self.tacs_proc:
            self.func_grad = None
            self.psi_T_S_vec = None
            self.struct_rhs_vec = None
            self.funclist = None
            self.functag = None
            self.ndvs = None
            self.nfunc = model.count_functions() if model else 1
            self.ndof = ndof

            #FIXME from steady, might not be needed
            self.comm = comm
            self.vol = 1.0 # required for AverageTemp function, not sure if needed on body level
            self.thermal_index = ndof-1 # FIXME assume temperature/heat flux entry is last entry

            self.struct_temps_all = {}

        for ibody,body in enumerate(model.bodies):
            self.get_mesh(ibody,body)

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
        #integrator.setUseFEMat(options['femat'],options['ordering'])
        integrator.setPrintLevel(options['print_level'])
        integrator.setOutputFrequency(options['output_freq'])
        return integrator

    def get_mesh(self,ibody,body):
        if self.tacs_proc:
            body.struct_X =  self.struct_X_vec.getArray().copy()
            body.struct_nnodes = int(body.struct_X.size/3)
        else:
            body.struct_nnodes = 0
            body.struct_X = np.array([],dtype=TACS.dtype)

        body.struct_disps = np.zeros(3*body.struct_nnodes,dtype=TACS.dtype)
        body.struct_temps = np.zeros(body.struct_nnodes,dtype=TACS.dtype)

        return

    def set_mesh(self,ibody,body):
        if self.tacs_proc:
            struct_X = self.struct_X_vec.getArray()
            struct_X[:] = body.struct_X[:]
            self.assembler.setNodes(self.struct_X_vec)

    def set_variables(self,scenario,bodies):
        self.ndvs = 0

        x = []
        for body in bodies:
            if 'structural' in body.variables:
                for var in body.variables['structural']:
                    x.append(var.value)
                    self.ndvs += 1
        x_vec = self.assembler.createDesignVec()
        x_arr = x_vec.getArray()
        x_arr[:] = x[:]
        if self.tacs_proc:
            self.assembler.setDesignVars(x_vec)

    def set_functions(self,scenario,bodies):
        if self.tacs_proc:
            self.funclist = []
            self.functag = []
            for func in scenario.functions:
                if func.analysis_type != 'structural':
                    self.funclist.append(None)
                    self.functag.append(0)

                elif func.name.lower() == 'compliance':
                    self.funclist.append(functions.Compliance(self.assembler))
                    self.functag.append(1)

                elif func.name.lower() == 'ksfailure':
                    if func.options:
                        ksweight = func.options['ksweight'] if 'ksweight' in func.options else 50.0
                    else:
                        ksweight = 50.0
                    self.funclist.append(functions.KSFailure(self.assembler, ksweight))
                    self.functag.append(1)

                elif func.name == 'mass':
                    self.functag.append(-1)

                elif func.name.lower() == 'temperature':
                    self.funclist.append(functions.AverageTemperature(self.assembler, self.vol))
                    self.functag.append(1)

                elif func.name.lower() == 'heatflux':
                    self.funclist.append(functions.HeatFlux(self.assembler))
                    self.functag.append(1)

                else:
                    print('WARNING: Unknown function being set into TACS set to mass')
                    self.functag.append(-1)

            func = scenario.functions[0]
            self.integrator[scenario.id].setFunctions(self.funclist,func.start,func.stop)
            self.integrator[scenario.id].evalFunctions(self.funclist)

    def get_functions(self,scenario,bodies):
        if self.tacs_proc:
            feval = self.integrator[scenario.id].evalFunctions(self.funclist)
            for i, func in enumerate(scenario.functions):
                if func.analysis_type == 'structural' and func.adjoint:
                    func.value = feval[i]
                if func.analysis_type == 'structural' and func.name == 'mass':
                    funclist = [functions.StructuralMass(self.assembler)]
                    value = self.assembler.evalFunctions(funclist)
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
            self.func_grad = []

            for func in range(len(self.funclist)):
                dfdx = self.integrator[scenario.id].getGradient(func,)
                dfdx_arr = dfdx.getArray()
                self.func_grad.append(dfdx_arr.copy())
                #self.func_grad.append(dfdx_arr[func*self.ndvs:(func+1)*self.ndvs].copy())

            for func in scenario.functions:
                if func.analysis_type == 'structural' and func.name == 'mass':
                    funclist = [functions.StructuralMass(self.assembler)]
                    dvsens = np.zeros(self.num_components)
                    self.assembler.addDVSens([funclist[0]], [dvsens])
                    self.func_grad.append(dvsens.copy())

        return

    def get_coordinate_derivatives(self,scenario,bodies,step):
        """ Evaluate gradients with respect to structural design variables"""

        if self.tacs_proc:
            fXptSens_vec = self.assembler.createNodeVec()

            # get the list of indices for this body
            for ibody, body in enumerate(bodies):
                if body.shape:
                    nodes = body.struct_nnodes

                    # TACS does the summation over steps internally, so only get the values when evaluating initial conditions
                    if step == 0:

                        # find the coordinate derivatives for each function
                        for nfunc, func in enumerate(scenario.functions):
                            if func.adjoint:
                                fXptSens_vec = self.integrator[scenario.id].getXptGradient(nfunc)
                            elif func.name == 'mass':
                                tacsfunc = functions.StructuralMass(self.assembler)
                                self.assembler.evalXptSens(tacsfunc, fXptSens_vec)

                            # pick out the points for this body
                            fxptSens = fXptSens_vec.getArray()
                            for i,n in enumerate(nodes):
                                body.struct_shape_term[i:i+1,nfunc] += fxptSens[n:n+1]

        return

    def initialize(self,scenario,bodies):
        # Update the meshes
        if self.tacs_proc:
            for ibody, body in enumerate(bodies):
                self.set_mesh(ibody,body)
                self.get_mesh(ibody,body)
                body.struct_temps = np.ones(body.struct_nnodes*body.therm_xfer_ndof, dtype=TACS.dtype) * body.T_ref
            self.integrator[scenario.id].iterate(0)

        return 0

    def iterate(self, scenario, bodies, step):
        fail = 0
        # Set heat flux into the bvec
        if self.tacs_proc:
            heat_flux_array = self.bvec_heat_flux.getArray()
            heat_flux_array[:] = 0.0

            for body in bodies:
                # Set heat flux on structure
                heat_flux_array[self.thermal_index::self.ndof] += body.struct_heat_flux[:]

            # take a step in the structural solver
            self.integrator[scenario.id].iterate(step,self.bvec_heat_flux)

            # Extract the heat flux values
            self.assembler.getVariables(self.ans)
            temps_array = self.ans.getArray()

            # Extract displacements and temperatures
            for body in bodies:
                body.struct_temps = np.zeros(body.struct_nnodes*body.therm_xfer_ndof, dtype=TACS.dtype)
                body.struct_temps[:] = temps_array[self.thermal_index::self.ndof]
        else:
            for body in bodies:
                body.struct_temps = np.zeros(body.struct_nnodes*body.therm_xfer_ndof, dtype=TACS.dtype)

        return fail

    def post(self,scenario,bodies):
        if self.tacs_proc:
            for body in bodies:
                self.struct_temps_all[scenario.id] = body.struct_temps[:]

            # export the f5 file
            try:
                self.post_export_f5()
            except:
                print("No f5 export set up")

    def initialize_adjoint(self,scenario,bodies):
        if self.tacs_proc:
            self.struct_rhs_vec = []
            for func in range(len(self.funclist)):
                self.struct_rhs_vec.append(self.assembler.createVec())
            for body in bodies:
                body.struct_temps = np.zeros(body.struct_nnodes*body.therm_xfer_ndof, dtype=TACS.dtype)
                body.struct_temps[:] = self.struct_temps_all[scenario.id]

        else:
            for body in bodies:
                body.struct_temps = np.zeros(body.struct_nnodes*body.therm_xfer_ndof)
        return 0

    def post_adjoint(self,scenario,bodies):
        self.eval_gradients(scenario)

    def set_states(self,scenario,bodies,step):
        if self.tacs_proc:
            _, self.ans, _, _ = self.integrator[scenario.id].getStates(step)
            temps = self.ans.getArray()
            for ibody, body in enumerate(bodies):
                for i in range(body.struct_nnodes):
                    body.struct_temps[i::body.therm_xfer_ndof] = temps[i::self.ndof]

    def iterate_adjoint(self, scenario, bodies, step):
        fail = 0

        for body in bodies:
            body.psi_T_S[:,:] = 0.0

        # put the body rhs's into the TACS bvec
        if self.tacs_proc:
            for func in range(len(self.funclist)):
                rhs_func = self.struct_rhs_vec[func].getArray()
                rhs_func[:] = 0.0
                for ibody, body in enumerate(bodies):
                    for i in range(body.therm_xfer_ndof):
                        rhs_func[i::self.ndof] += body.struct_rhs_T[i::body.therm_xfer_ndof, func]

            # take a reverse step in the adjoint solver
            self.integrator[scenario.id].initAdjoint(step)
            self.integrator[scenario.id].iterateAdjoint(step, self.struct_rhs_vec)
            self.integrator[scenario.id].postAdjoint(step)

            # pull the adjoints out of the TACS bodies
            for func in range(len(self.funclist)):
                self.psi_T_S_vec = self.integrator[scenario.id].getAdjoint(step, func)
                psi_func = self.psi_T_S_vec.getArray()
                for ibody, body in enumerate(bodies):
                    for i in range(body.therm_xfer_ndof):
                        body.psi_T_S[i::body.therm_xfer_ndof, func] = psi_func[i::self.ndof]
        return fail

    def step_pre(self, scenario,bodies,step):
        fail = 0
        return fail

    def step_solver(self, scenario,bodies,step,fsi_subiter):
        fail = 0
        # Set heat flux into the bvec
        if self.tacs_proc:
            heat_flux_vector = self.bvec_heat_flux.getArray()
            for ibody, body in enumerate(bodies):
                for i,n in enumerate(body.struct_nnodes):
                    heat_flux_vector[body.dof*n:body.dof*n+body.therm_xfer_ndof] = body.struct_heat_flux[body.therm_xfer_ndof*i:body.therm_xfer_ndof*i+body.therm_xfer_ndof]

            # take a step in the structural solver
            self.integrator[scenario.id].iterate(step, self.bvec_heat_flux)

            # Extract the structural displacements
            self.assembler.getVariables(self.ans)
            temps_vector = self.ans.getArray()

            for ibody, body in enumerate(bodies):
                for i,n in enumerate(body.struct_nnodes):
                    body.struct_temps[body.therm_xfer_ndof*i:body.therm_xfer_ndof*i+body.therm_xfer_ndof] = temps_vector[body.dof*n:body.dof*n+body.therm_xfer_ndof]
        else:
            for body in bodies:
                body.struct_temps = np.zeros(body.struct_nnodes*body.therm_xfer_ndof)

        return fail

    def step_post(self, scenario,bodies,step):
        return 0
