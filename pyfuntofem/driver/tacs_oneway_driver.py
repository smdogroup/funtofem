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


# TACS one-way coupled drivers that use fixed fun3d aero loads
__all__ = ["TacsOnewayDriver"]

from pyfuntofem.interface.tacs_interface import TacsSteadyInterface
from .funtofem_nlbgs_driver import FUNtoFEMnlbgs
from pyfuntofem.interface.solver_manager import SolverManager
from pyfuntofem.optimization.optimization_manager import OptimizationManager
from tacs import caps2tacs

from mpi4py import MPI
import numpy as np


class TacsOnewayDriver:
    def __init__(
        self,
        solvers,
        model,
        tacs_aim=None,
        nprocs=None,
    ):
        """
        build the tacs analysis driver for shape/no shape change, assumes you have already primed the loads (see class method to assist with that)

        Parameters
        ----------
        solvers: :class:`~interface.solver_manager.SolverManager`
            The various disciplinary solvers.
        model: :class:`~funtofem_model.FUNtoFEMmodel`
            The model containing the design data.
        tacs_aim: `caps2tacs.TacsAim`
            Interface object for TACS and ESP/CAPS, wraps the tacsAIM.
        nprocs: int
            Number of processes that TACS is running on.
        """
        self.solvers = solvers
        self.comm = solvers.comm
        self.model = model
        self.nprocs = nprocs
        self.tacs_interface = solvers.structural
        self.tacs_aim = tacs_aim

        # store the shape variables list
        self.shape_variables = [
            var for var in self.model.get_variables() if var.analysis_type == "shape"
        ]

        # check for unsteady problems
        self._unsteady = False
        for scenario in model.scenarios:
            if not scenario.steady:
                self._unsteady = True
                break

        # assertion check for unsteady
        # TODO : unsteady not available yet, can add this feature
        assert not self.unsteady

        # assertion checks for no shape change vs shape change
        if self.change_shape:
            assert tacs_aim is not None
            assert nprocs is not None
            assert isinstance(tacs_aim, caps2tacs.TacsAim)
        else:
            assert self.tacs_interface is not None
            assert isinstance(self.tacs_interface, TacsSteadyInterface)

        # reset struct mesh positions for no shape, just tacs analysis
        if not self.change_shape:
            for body in self.model.bodies:
                body.update_transfer()

    @property
    def change_shape(self) -> bool:
        return len(self.shape_variables) > 0

    @property
    def unsteady(self) -> bool:
        return self._unsteady

    @classmethod
    def prime_loads(cls, funtofem_driver):
        """
        Used to prime struct loads for optimization over tacs analysis with no shape variables

        Parameters
        ----------
        funtofem_driver: :class:`~funtofem_nlbgs_driver.FUNtoFEMnlbgs`
            the coupled funtofem NLBGS driver
        """
        funtofem_driver.solve_forward()
        return cls(funtofem_driver.solvers, funtofem_driver.model)

    @classmethod
    def prime_loads_shape(
        cls, flow_solver, tacs_aim, transfer_settings, nprocs, bdf_file=None
    ):
        """
        Used to prime aero loads for optimization over tacs analysis with shape change and tacs aim

        Parameters
        ----------
        flow_solver: [:class:`~interface.Fun3dInterface or :class`~interface.TestAerodynamicSolver`]
            Solver Interface for CFD such as Fun3dInterface, Su2Interface, or test aero solver
        tacs_aim: `caps2tacs.TacsAim`
            Interface object from TACS to ESP/CAPS, wraps the tacsAIM object.
        transfer_settings: :class:`~transfer_settings.TransferSettings`
            Settings for transfer of state variables across aero and struct meshes
        nprocs: int
            Number of processes that TACS is running on.
        bdf_file: str or `os.path`
            File or path to the bdf or dat file of structural mesh
        """

        # generate the geometry if necessary
        make_bdf = bdf_file is None
        if make_bdf:
            tacs_aim.pre_analysis()
            bdf_file = tacs_aim.dat_file_path

        # copy comm and model from other objects
        comm = tacs_aim.comm
        model = flow_solver.model

        # create the solver manager
        solvers = SolverManager(comm)
        solvers.flow = flow_solver
        solvers.structural = TacsSteadyInterface.create_from_bdf(
            model=model,  # copy model from flow solver
            comm=comm,
            nprocs=nprocs,
            bdf_file=bdf_file,
            prefix=tacs_aim.analysis_dir,
        )

        # build the funtofem driver and run a forward analysis to prime loads
        FUNtoFEMnlbgs(
            solvers, transfer_settings=transfer_settings, model=model
        ).solve_forward()

        if make_bdf:
            # initialize adjoint variables for each body to initialize struct_shape_term
            for scenario in model.scenarios:
                for body in model.bodies:
                    body.initialize_adjoint_variables(scenario)

            # write the model sensitivity file with zero derivatives
            model.write_sensitivity_file(
                comm=comm,
                filename=tacs_aim.sens_file_path,
                discipline="structural",
            )

            # run postAnalysis to prevent CAPS_DIRTY error
            tacs_aim.post_analysis()

        return cls(solvers, model, nprocs=nprocs, tacs_aim=tacs_aim)

    @classmethod
    def prime_loads_from_file(
        cls, filename, solvers, model, nprocs, transfer_settings, tacs_aim=None
    ):
        """
        Used to prime aero loads for optimization over tacs analysis with shape change and tacs aim
        Built from an aero loads file of a previous CFD analysis in FuntofemNlbgs driver (TODO : make uncoupled fluid solver features)
        The loads file is written from the FUNtoFEM model after a forward analysis of a flow solver

        Parameters
        ----------
        filename : str or path to aero loads file
            the filepath of the aerodynamic loads file from a previous CFD analysis (written from FUNtoFEM model)
        solvers: :class:`~interface.SolverManager`
            Solver Interface for CFD such as Fun3dInterface, Su2Interface, or test aero solver
        model : :class:`~model.FUNtoFEMmodel
        nprocs: int
            Number of processes that TACS is running on.
        transfer_settings: :class:`~transfer_settings.TransferSettings`
            Settings for transfer of state variables across aero and struct meshes
        tacs_aim: `caps2tacs.TacsAim`
            Interface object from TACS to ESP/CAPS, wraps the tacsAIM object.
        """
        comm = solvers.comm
        world_rank = comm.Get_rank()
        if world_rank < nprocs:
            color = 1
        else:
            color = MPI.UNDEFINED
        tacs_comm = comm.Split(color, world_rank)

        # initialize transfer settings
        comm_manager = solvers.comm_manager

        # read in the loads from the file
        loads_data, discipline = model.read_loads_file(comm, filename)
        print(f"loads data = {loads_data}")
        print(f"discipline = {discipline}")

        # initialize the transfer scheme since the load if statements depend on this
        for body in model.bodies:
            body.initialize_transfer(
                comm=comm,
                struct_comm=tacs_comm,
                struct_root=comm_manager.struct_root,
                aero_comm=comm_manager.aero_comm,
                aero_root=comm_manager.aero_root,
                transfer_settings=transfer_settings,
            )
            for scenario in model.scenarios:
                body.initialize_variables(scenario)

            # distribute the loads into each body
            body._distribute_loads(loads_data, discipline=discipline)

        # transfer aero loads to struct loads if aero loads were read in and we have no shape change
        if discipline == "aerodynamic" and tacs_aim is None:
            # transfer aero loads to struct loads if no shape change
            if tacs_aim is None:
                for body in model.bodies:
                    for scenario in model.scenarios:
                        # perform disps transfer first to prevent seg fault
                        body.transfer_disps(scenario)
                        body.transfer_temps(scenario)
                        body.transfer_loads(scenario)
                        body.transfer_heat_flux(scenario)

        else:  # struct discipline case, meaning struct loads were written
            # can't do shape change under fixed struct loads
            assert tacs_aim is None

        return cls(solvers, model, nprocs=nprocs, tacs_aim=tacs_aim)

    @property
    def manager(self, hot_start: bool = False):
        return OptimizationManager(self, hot_start=hot_start)

    @property
    def tacs_comm(self):
        world_rank = self.comm.rank
        if world_rank < self.n_tacs_procs:
            color = 1
            key = world_rank
        else:
            color = MPI.UNDEFINED
            key = world_rank
        return self.comm.Split(color, key)

    @property
    def root_proc(self) -> bool:
        return self.comm.rank == 0

    def _transfer_fixed_aero_loads(self):
        """
        transfer fixed aero loads over to the new
        """
        # loop over each body to copy and transfer loads for the new structure
        for body in self.model.bodies:
            # update the transfer schemes for the new mesh size
            body.update_transfer()

            ns = body.struct_nnodes
            dtype = body.dtype

            # zero the initial struct loads and struct flux for each scenario
            for scenario in self.model.scenarios:
                # initialize new struct shape term for new ns
                nf = scenario.count_adjoint_functions()
                # TODO : fix body.py struct_shape_term should be scenario dictionary for multiple scenarios
                body.struct_shape_term = np.zeros((3 * ns, nf), dtype=dtype)

                # initialize new elastic struct vectors
                if body.transfer is not None:
                    body.struct_loads[scenario.id] = np.zeros(3 * ns, dtype=dtype)
                    body.struct_disps[scenario.id] = np.zeros(3 * ns, dtype=dtype)

                # initialize new struct heat flux
                if body.thermal_transfer is not None:
                    body.struct_heat_flux[scenario.id] = np.zeros(ns, dtype=dtype)
                    body.struct_temps[scenario.id] = (
                        np.ones(ns, dtype=dtype) * scenario.T_ref
                    )

                # transfer the loads and heat flux from fixed aero loads to
                # the mesh for the new structural shape
                body.transfer_loads(scenario)
                body.transfer_heat_flux(scenario)

        return

    def solve_forward(self):
        """
        forward analysis for the given shape and functionals
        assumes shape variables have already been changed
        """

        if self.change_shape:
            # set the new shape variables into the model using update design to prevent CAPS_CLEAN errors
            input_dict = {var.name: var.value for var in self.model.get_variables()}
            self.model.tacs_model.update_design(input_dict)
            self.tacs_aim.setup_aim()

            # build the new structure geometry
            self.tacs_aim.pre_analysis()

            # make the new tacs interface of the structural geometry
            # TODO : need to make sure the InterfaceFromBDF method tells the struct_id properly
            self.tacs_interface = TacsSteadyInterface.create_from_bdf(
                model=self.model,
                comm=self.comm,
                nprocs=self.nprocs,
                bdf_file=self.tacs_aim.dat_file_path,
                prefix=self.tacs_aim.analysis_dir,
            )

            # make a solvers object to hold structural solver since flow is no longer used
            solvers = SolverManager(self.comm, use_flow=False)
            solvers.structural = self.tacs_interface

            # transfer the fixed aero loads
            self._transfer_fixed_aero_loads()
        # end of change shape section

        # run the tacs forward analysis for no shape
        self._solve_forward_no_shape()

    def solve_adjoint(self):
        """
        solve the adjoint analysis for the given shape
        assumes the forward analysis for this shape has already been performed
        """

        # run the adjoint structural analysis
        self._solve_adjoint_no_shape()

        if self.change_shape:
            # compute tacs coordinate derivatives
            for scenario in self.model.scenarios:
                self.tacs_interface.get_coordinate_derivatives(
                    scenario, self.model.bodies, step=0
                )

                # add transfer scheme contributions
                for body in self.model.bodies:
                    body.add_coordinate_derivative(scenario, step=0)

            # collect the coordinate derivatives for each body
            for body in self.model.bodies:
                body.collect_coordinate_derivatives(
                    comm=self.comm, discipline="structural"
                )

            # write the sensitivity file for the tacs AIM
            self.model.write_sensitivity_file(
                comm=self.comm,
                filename=self.tacs_aim.sens_file_path,
                discipline="structural",
            )

            # run the tacs aim postAnalysis to compute the chain rule product
            self.tacs_aim.post_analysis()

            # store the shape variables in the function gradients
            for scenario in self.model.scenarios:
                self._get_shape_derivatives(scenario)
        # end of change shape section

    def _get_shape_derivatives(self, scenario):
        """
        get shape derivatives together from tacs aim
        and store the data in the funtofem model
        """
        gradients = None

        # read shape gradients from tacs aim on root proc
        if self.root_proc:
            gradients = []
            direct_tacs_aim = self.tacs_aim.aim

            for ifunc, func in enumerate(scenario.functions):
                gradients.append([])
                for ivar, var in enumerate(self.shape_variables):
                    derivative = direct_tacs_aim.dynout[func.name].deriv(var.name)
                    gradients[ifunc].append(derivative)

        # broadcast shape gradients to all other processors
        gradients = self.comm.bcast(gradients, root=0)

        # store shape derivatives in funtofem model on all processors
        for ifunc, func in enumerate(scenario.functions):
            for ivar, var in enumerate(self.shape_variables):
                derivative = gradients[ifunc][ivar]
                func.set_gradient_component(var, derivative)

        return

    def _solve_forward_no_shape(self):
        """
        solve the forward analysis of TACS analysis with aerodynamic loads
        and heat fluxes from previous analysis still retained
        """

        fail = 0

        # zero all data to start fresh problem, u = 0, res = 0
        self._zero_tacs_data()

        for scenario in self.model.scenarios:
            # set functions and variables
            self.tacs_interface.set_variables(scenario, self.model.bodies)
            self.tacs_interface.set_functions(scenario, self.model.bodies)

            # run the forward analysis via iterate
            self.tacs_interface.initialize(scenario, self.model.bodies)
            self.tacs_interface.iterate(scenario, self.model.bodies, step=0)
            self.tacs_interface.post(scenario, self.model.bodies)

            # get functions to store the function values into the model
            self.tacs_interface.get_functions(scenario, self.model.bodies)

        return 0

    def _solve_adjoint_no_shape(self):
        """
        solve the adjoint analysis of TACS analysis with aerodynamic loads
        and heat fluxes from previous analysis still retained
        Similar to funtofem_driver
        """

        functions = self.model.get_functions()

        # Zero the derivative values stored in the function
        for func in functions:
            func.zero_derivatives()

        # zero adjoint data
        self._zero_adjoint_data()

        for scenario in self.model.scenarios:
            # set functions and variables
            self.tacs_interface.set_variables(scenario, self.model.bodies)
            self.tacs_interface.set_functions(scenario, self.model.bodies)

            # zero all coupled adjoint variables in the body
            for body in self.model.bodies:
                body.initialize_adjoint_variables(scenario)

            # initialize, run, and do post adjoint
            self.tacs_interface.initialize_adjoint(scenario, self.model.bodies)
            self.tacs_interface.iterate_adjoint(scenario, self.model.bodies, step=0)
            self.tacs_interface.post_adjoint(scenario, self.model.bodies)

            # transfer loads adjoint since fa -> fs has shape dependency
            if self.change_shape:
                for body in self.model.bodies:
                    body.transfer_loads_adjoint(scenario)

            # call get function gradients to store  the gradients from tacs
            self.tacs_interface.get_function_gradients(scenario, self.model.bodies)

    def _zero_tacs_data(self):
        """
        zero any TACS solution / adjoint data before running pure TACS
        """

        if self.tacs_interface.tacs_proc:
            # zero temporary solution data
            # others are zeroed out in the tacs_interface by default
            self.tacs_interface.res.zeroEntries()
            self.tacs_interface.ext_force.zeroEntries()
            self.tacs_interface.update.zeroEntries()

            # zero any scenario data
            for scenario in self.model.scenarios:
                # zero state data
                u = self.tacs_interface.scenario_data[scenario].u
                u.zeroEntries()
                self.tacs_interface.assembler.setVariables(u)

    def _zero_adjoint_data(self):
        if self.tacs_interface.tacs_proc:
            # zero adjoint variable
            for scenario in self.model.scenarios:
                psi = self.tacs_interface.scenario_data[scenario].psi
                for vec in psi:
                    vec.zeroEntries()
