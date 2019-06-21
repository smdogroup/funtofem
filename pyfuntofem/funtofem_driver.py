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

import numpy as np
from mpi4py import MPI
from funtofem import TransferScheme
try:
    from hermes_transfer import HermesTransfer
except:
    pass

np.set_printoptions(precision=15)

class FUNtoFEMDriver(object):
    """
    The FUNtoFEM driver base class has all of the driver except for the coupling algorithms
    """
    def __init__(self,solvers,comm,struct_comm,struct_master,aero_comm,aero_master,transfer_options=None,model=None):
        """
        Parameters
        ----------
        solvers: dict
           the various disciplinary solvers
        comm: MPI.comm
            MPI communicator
        transfer_options: dict
            options of the load and displacement transfer scheme
        model: :class:`~funtofem_model.FUNtoFEMmodel`
            The model containing the design data
        """

        # communicator
        self.comm = comm
        self.aero_comm = aero_comm
        self.aero_master = aero_master
        self.struct_comm = struct_comm
        self.struct_master = struct_master

        # Solver classes
        self.solvers  = solvers

        # Make a fake model if not given one
        if model is not None:
            self.fakemodel = False
        else:
            print("FUNtoFEM driver: generating fake model")
            from pyfuntofem.model import FUNtoFEMmodel,Body,Scenario,Function

            model = FUNtoFEMmodel('fakemodel')

            fakebody = Body('fakebody')
            model.add_body(fakebody)

            fakescenario = Scenario('fakescenario')
            function     = Function('cl',analysis_type='aerodynamic')
            fakescenario.add_function(function)
            model.add_scenario(fakescenario)

            self.fakemodel = True

        self.model = model

        # Initialize transfer scheme
        self._initialize_transfer(transfer_options)

        # Initialize the shape parameterization
        for body in self.model.bodies:
            if body.shape:
                body.initialize_shape_parameterization()

    def update_model(self,model):
        """
        Update the model object that the driver sees

        Parameters
        ----------
        model: FUNtoFEM model type
        """
        self.model = model

    def _initialize_transfer(self,transfer_options):
        """
        Initialize the transfer scheme

        Parameters
        ----------
        transfer_options: dictionary or list of dictionaries
            options for the load and displacement transfer scheme for the bodies
        """

        # If the user did not specify a transfer scheme default to MELD
        if transfer_options is None:
            transfer_options = []
            for body in self.model.bodies:
                transfer_options.append({'scheme': 'meld', 'isym': -1, 'beta': 0.5, 'npts': 200})

        # if the user gave a dictionary instead of a list of dictionaries, assume all bodies use the same settings
        if type(transfer_options) is dict:
            transfer_options = len(self.model.bodies) * [ transfer_options ]

        for ibody, body in enumerate(self.model.bodies):
            body.transfer = None

            if transfer_options[ibody]['scheme'].lower() == 'hermes':

                body.transfer = HermesTransfer(self.comm,self.struct_comm,self.aero_comm)

            elif transfer_options[ibody]['scheme'].lower() == 'rbf':

                basis=TransferScheme.PY_THIN_PLATE_SPLINE

                if 'basis function' in transfer_options[ibody]:
                    if transfer_options[ibody]['basis function'].lower()=='thin plate spline':
                        basis=TransferScheme.PY_THIN_PLATE_SPLINE
                    elif transfer_options[ibody]['basis function'].lower()=='gaussian':
                        basis=TransferScheme.PY_GAUSSIAN
                    elif transfer_options[ibody]['basis function'].lower()=='multiquadric':
                        basis=TransferScheme.PY_MULTIQUADRIC
                    elif transfer_options[ibody]['basis function'].lower()=='inverse multiquadric':
                        basis=TransferScheme.PY_INVERSE_MULTIQUADRIC
                    else:
                        print('Unknown RBF basis function for body number', ibody)
                        quit()

                body.transfer = TransferScheme.pyRBF(self.comm, self.comm, 0, self.comm, 0, basis, 1)

            elif transfer_options[ibody]['scheme'].lower()== 'meld':
                # defaults
                isym = -1
                beta = 0.5
                num_nearest = 200

                if 'isym' in transfer_options[ibody]:
                    isym = transfer_options[ibody]['isym']

                if 'beta' in transfer_options[ibody]:
                    beta = transfer_options[ibody]['beta']

                if 'npts' in transfer_options[ibody]:
                    num_nearest = transfer_options[ibody]['npts']

                body.transfer = TransferScheme.pyMELD(self.comm, self.struct_comm, self.struct_master, self.aero_comm, self.aero_master,
                                                      isym, num_nearest, beta)

            elif transfer_options[ibody]['scheme'].lower()== 'linearized meld':

                # defaults
                isym = -1
                beta = 0.5
                num_nearest = 200

                if 'beta' in transfer_options[ibody]:
                    beta = transfer_options[ibody]['beta']

                if 'npts' in transfer_options[ibody]:
                    num_nearest = transfer_options[ibody]['npts']


                body.transfer = TransferScheme.pyLinearizedMELD(self.comm, self.comm, 0, self.comm, 0,
                                                           num_nearest, beta)

            elif transfer_options[ibody]['scheme'].lower()== 'beam':
                conn = transfer_options[ibody]['conn']
                nelems = transfer_options[ibody]['nelems']
                order = transfer_options[ibody]['order']
                ndof = transfer_options[ibody]['ndof']

                body.xfer_ndof = ndof
                body.transfer = TransferScheme.pyBeamTransfer(self.comm, self.struct_comm, self.struct_master, self.aero_comm,
                                                                   self.aero_master, conn, nelems, order, ndof)
            else:
                print("Error: Unknown transfer scheme for body", ibody)
                quit()

            # Load structural and aerodynamic meshes into FUNtoFEM
            # Only want real part for the initialization
            if TransferScheme.dtype == np.complex128 or TransferScheme.dtype == complex:
                if self.struct_comm != MPI.COMM_NULL:
                    body.transfer.setStructNodes(body.struct_X.real + 0.0j)
                else:
                    body.struct_nnodes = 0

                if self.aero_comm != MPI.COMM_NULL:
                    body.transfer.setAeroNodes(body.aero_X.real + 0.0j)
                else:
                    body.aero_nnodes = 0
            else:
                if self.struct_comm != MPI.COMM_NULL:
                    body.transfer.setStructNodes(body.struct_X)
                    body.transfer.setStructNodes(body.struct_X)
                else:
                    body.struct_nnodes = 0

                if self.aero_comm != MPI.COMM_NULL:
                    body.transfer.setAeroNodes(body.aero_X)
                else:
                    body.aero_nnodes = 0

            # Initialize FUNtoFEM
            body.transfer.initialize()

            # Load structural and aerodynamic meshes into FUNtoFEM
            if TransferScheme.dtype == np.complex128 or TransferScheme.dtype == complex:
                if self.struct_comm != MPI.COMM_NULL:
                    body.transfer.setStructNodes(body.struct_X)
                else:
                    body.struct_nnodes = 0
                if self.aero_comm != MPI.COMM_NULL:
                    body.transfer.setAeroNodes(body.aero_X)
                else:
                    body.aero_nnodes = 0

    def _update_transfer(self):
        """
        Update the positions of the nodes in transfer schemes
        """
        self.struct_disps = []
        for body in self.model.bodies:
            if self.struct_comm != MPI.COMM_NULL:
                body.transfer.setStructNodes(body.struct_X)
            else:
                body.struct_nnodes = 0
            if self.aero_comm != MPI.COMM_NULL:
                body.transfer.setAeroNodes(body.aero_X)
            else:
                body.aero_nnodes = 0

    def solve_forward(self, steps=None):
        """
        Solves the coupled forward problem

        Parameters
        ----------
        steps: int
            number of coupled solver steps. Only for use if a FUNtoFEM model is not defined
        """
        fail = 0

        # update the shapes first
        for body in self.model.bodies:
            if body.shape:
                complex_run = True if (TransferScheme.dtype==np.complex128 or TransferScheme.dtype == complex) else False
                body.update_shape(complex_run)

        # loop over the forward problem for the different scenarios
        for scenario in self.model.scenarios:

            # tell the solvers what the variable values and functions are for this scenario
            if not self.fakemodel:
                self._distribute_variables(scenario,self.model.bodies)
                self._distribute_functions(scenario,self.model.bodies)

            # Set the new meshes Initialize the forward solvers
            fail = self._initialize_forward(scenario,self.model.bodies)
            if fail != 0:
                if self.comm.Get_rank() == 0:
                    print("Fail flag return during initialization")

            # Update transfer postions to the initial conditions
            self._update_transfer()

            if scenario.steady:
                fail = self._solve_steady_forward(scenario,steps)
                if fail != 0:
                    if self.comm.Get_rank() == 0:
                        print("Fail flag return during forward solve")
            else:
                fail = self._solve_unsteady_forward(scenario,steps)
                if fail != 0:
                    if self.comm.Get_rank() == 0:
                        print("Fail flag return during forward solve")

            # Perform any operations after the forward solve
            self._post_forward(scenario,self.model.bodies)
            if fail == 0:
                self._eval_functions(scenario,self.model.bodies)

        return fail

    def solve_adjoint(self):
        """
        Solves the coupled adjoint problem and computes gradients
        """

        fail = 0
        # Make sure we have functions defined before we start the adjoint
        if self.fakemodel:
            print("Aborting: attempting to run FUNtoFEM adjoint with no model defined")
            quit()
        elif not self.model.get_functions():
            print("Aborting: attempting to run FUNtoFEM adjoint with no functions defined")
            quit()

        # Set the functions into the solvers
        for scenario in self.model.scenarios:

            # tell the solvers what the variable values and functions are for this scenario
            self._distribute_variables(scenario,self.model.bodies)
            self._distribute_functions(scenario,self.model.bodies)

            # Initialize the adjoint solvers
            self._initialize_adjoint_variables(scenario,self.model.bodies)
            self._initialize_adjoint(scenario,self.model.bodies)

            if scenario.steady:
                fail = self._solve_steady_adjoint(scenario)
                if fail != 0:
                    return fail
            else:
                fail = self._solve_unsteady_adjoint(scenario)
                if fail != 0:
                    return fail

            # Perform any operations after the adjoint solve
            self._post_adjoint(scenario,self.model.bodies)

            self._eval_function_grads(scenario)

        self.model.enforce_coupling_derivatives()
        return fail

    def _initialize_forward(self,scenario,bodies):
        for solver in self.solvers:
            fail = self.solvers[solver].initialize(scenario,bodies)
            if fail!=0:
                return fail
        return 0

    def _initialize_adjoint(self,scenario,bodies):
        for solver in self.solvers:
            fail = self.solvers[solver].initialize_adjoint(scenario,bodies)
            if fail!=0:
                return fail
        return 0

    def _post_forward(self,scenario,bodies):
        for solver in self.solvers:
            self.solvers[solver].post(scenario,bodies)

    def _post_adjoint(self,scenario,bodies):
        for solver in self.solvers:
            self.solvers[solver].post_adjoint(scenario,bodies)

    def _distribute_functions(self,scenario,bodies):
        for solver in self.solvers:
            self.solvers[solver].set_functions(scenario,bodies)

    def _distribute_variables(self,scenario,bodies):
        for solver in self.solvers:
            self.solvers[solver].set_variables(scenario,bodies)

    def _eval_functions(self,scenario,bodies):
        for solver in self.solvers:
            self.solvers[solver].get_functions(scenario,self.model.bodies)

    def _eval_function_grads(self,scenario):

        offset = self._get_scenario_function_offset(scenario)

        for solver in self.solvers:
            self.solvers[solver].get_function_gradients(scenario,self.model.bodies,offset)

        for body in self.model.bodies:
            if body.shape:
                body.shape_derivative(scenario,offset)

    def _get_scenario_function_offset(self,scenario):
        """
        The offset tells each scenario what is first function's index is
        """
        offset = 0
        for i in range(scenario.id-1):
            offset += self.model.scenarios[i].count_functions()

        return offset

    def _extract_coordinate_derivatives(self,scenario,bodies,step):

        nfunctions = scenario.count_adjoint_functions()

        # get the contributions from the solvers
        for solver in self.solvers:
            self.solvers[solver].get_coordinate_derivatives(scenario,self.model.bodies,step)

        # transfer scheme contributions to the coordinates derivatives
        if step > 0:
            for body in self.model.bodies:
                if body.shape and body.transfer:

                    # Aerodynamic coordinate derivatives
                    temp = np.zeros((3*body.aero_nnodes),dtype=TransferScheme.dtype)
                    for func in range(nfunctions):
                        # Load transfer term
                        body.transfer.applydLdxA0(body.psi_L[:, func].copy(order='C'), temp)
                        body.aero_shape_term[:,func] += temp.copy()

                        # Displacement transfer term
                        body.transfer.applydDdxA0(body.psi_D[:, func].copy(order='C'), temp)
                        body.aero_shape_term[:,func] += temp.copy()

                    # Structural coordinate derivatives
                    temp = np.zeros(body.struct_nnodes*body.xfer_ndof,dtype=TransferScheme.dtype)
                    for func in range(nfunctions):
                        # Load transfer term
                        body.transfer.applydLdxS0(body.psi_L[:, func].copy(order='C'), temp)
                        body.struct_shape_term[:,func] += temp.copy()

                        # Displacement transfer term
                        body.transfer.applydDdxS0(body.psi_D[:, func].copy(order='C'), temp)
                        body.struct_shape_term[:,func] += temp.copy()

        return

    def _solve_steady_forward(self,scenario,steps):
        return 1

    def _solve_unsteady_forward(self,scenario,steps):
        return 1

    def _solve_steady_adjoint(self,scenario):
        return 1

    def _solve_unsteady_adjoint(self,scenario):
        return 1
