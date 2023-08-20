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

from funtofem.interface.tacs_interface import (
    TacsInterface,
    TacsSteadyInterface,
    TacsUnsteadyInterface,
)
from funtofem.interface.solver_manager import SolverManager
from funtofem.optimization.optimization_manager import OptimizationManager

import importlib.util

caps_loader = importlib.util.find_spec("pyCAPS")
if caps_loader is not None:  # tacs loader not None check for this file anyways
    from tacs import caps2tacs

from mpi4py import MPI
import numpy as np


class TacsOnewayDriver:
    def __init__(
        self,
        solvers,
        model,
        transfer_settings=None,
        nprocs=None,
        external_shape=False,
    ):
        """
        build the tacs analysis driver for shape/no shape change, assumes you have already primed the loads (see class method to assist with that)

        Parameters
        ----------
        solvers: :class:`~interface.solver_manager.SolverManager`
            The various disciplinary solvers.
        model: :class:`~funtofem_model.FUNtoFEMmodel`
            The model containing the design data and the TACSmodel with tacsAIM wrapper inside (if using shape; otherwise can be None).
        transfer_settings: :class:`driver.TransferSettings`
        nprocs: int
            Number of processes that TACS is running on.
        external_shape: bool
            whether the tacs aim shape analysis is performed outside the class
        """
        self.solvers = solvers
        self.comm = solvers.comm
        self.model = model
        self.nprocs = nprocs
        self.transfer_settings = transfer_settings
        self.tacs_interface = solvers.structural
        if model.structural is None:
            tacs_aim = None
        else:
            tacs_aim = model.structural.tacs_aim
        self.tacs_aim = tacs_aim
        self.external_shape = external_shape

        self._shape_init_transfer = False

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

        # assertion checks for no shape change vs shape change
        if self.change_shape:
            assert tacs_aim is not None or external_shape
            if nprocs is None:  # will error if nprocs not defined anywhere
                nprocs = solvers.structural.nprocs
            if external_shape:
                # define properties needed for external shape paths
                self._dat_file_path = None
                self._analysis_dir = None
            else:
                if caps_loader is not None:
                    assert isinstance(tacs_aim, caps2tacs.TacsAim)
                else:
                    raise AssertionError(
                        "Need to have ESP/CAPS pyCAPS package to use shape change"
                    )
        else:  # not change shape
            assert self.tacs_interface is not None
            assert isinstance(self.tacs_interface, TacsSteadyInterface) or isinstance(
                self.tacs_interface, TacsUnsteadyInterface
            )

            # transfer to fixed structural loads in case the user got only aero loads from the Fun3dOnewayDriver
            for body in self.model.bodies:
                # initializing transfer schemes is the responsibility of drivers with aerodynamic analysis since they come first
                body.update_transfer()

                for scenario in self.model.scenarios:
                    # perform disps transfer first to prevent seg fault
                    body.transfer_disps(scenario)
                    body.transfer_temps(scenario)
                    # transfer aero to struct loads
                    body.transfer_loads(scenario)
                    body.transfer_heat_flux(scenario)

    @property
    def change_shape(self) -> bool:
        return len(self.shape_variables) > 0

    @property
    def steady(self) -> bool:
        return not (self._unsteady)

    @property
    def unsteady(self) -> bool:
        return self._unsteady

    @classmethod
    def prime_loads(
        cls, driver, transfer_settings=None, nprocs=None, external_shape=False
    ):
        """
        Used to prime struct/aero loads for optimization over TACS analysis.
        Can use the Fun3dOnewayDriver or FUNtoFEMnlbgs driver to prime the loads
        If structural solver exists, it will transfer to fixed structural loads in __init__ construction
        of the TacsOnewayDriver class.
        If shape variables exist in the FUNtoFEMmodel, you need to have model.structural be a TacsModel
        and the TacsModel contains a TacsAim wrapper class. If shape variables exist, shape derivatives
        and shape analysis will be performed.

        Parameters
        ----------
        driver: :class:`Fun3dOnewayDriver` or :class:`~funtofem_nlbgs_driver.FUNtoFEMnlbgs`
            the fun3d oneway driver or coupled funtofem NLBGS driver

        Optional Parameters
        -------------------
        nprocs: int
            number of procs for tacs analysis, only need to give this if doing shape optimization
        transfer_settings: :class:`driver.TransferSettings`
            used for transferring fixed aero loads to struct loads, need to give this or it uses default
        external_shape: bool
            whether to do shape analysis with ESP/CAPS inside this driver or outside of it
        """
        driver.solve_forward()
        if transfer_settings is None:
            try:
                transfer_settings = driver.transfer_settings
            except:
                transfer_settings = transfer_settings
        return cls(
            driver.solvers, driver.model, transfer_settings, nprocs, external_shape
        )

    @classmethod
    def prime_loads_from_file(
        cls,
        filename,
        solvers,
        model,
        nprocs,
        transfer_settings,
        external_shape=False,
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
        external_shape: bool
            whether the tacs aim shape analysis is performed outside this class
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
        loads_data = model.read_aero_loads(comm, filename)

        # initialize the transfer scheme then distribute aero loads
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
            body._distribute_aero_loads(loads_data)

        return cls(
            solvers,
            model,
            nprocs=nprocs,
            external_shape=external_shape,
        )

    @property
    def manager(self, hot_start: bool = False):
        return OptimizationManager(self, hot_start=hot_start)

    @property
    def tacs_comm(self):
        if self.change_shape:
            world_rank = self.comm.rank
            if world_rank < self.nprocs:
                color = 1
                key = world_rank
            else:
                color = MPI.UNDEFINED
                key = world_rank
            return self.comm.Split(color, key)
        else:
            return self.tacs_interface.tacs_comm

    @property
    def root_proc(self) -> bool:
        return self.comm.rank == 0

    @property
    def dat_file_path(self):
        if self.external_shape:
            return self._dat_file_path
        else:
            return self.tacs_aim.dat_file_path

    @property
    def analysis_dir(self):
        if self.external_shape:
            return self._analysis_dir
        else:
            return self.tacs_aim.analysis_dir

    def input_paths(self, dat_file_path, analysis_dir):
        """
        receive path information if the user is handling the shape optimization externally

        Parameters
        ----------
        dat_file_path: path object
            path to nastran_CAPS.dat file of the tacsAIM (which includes the bdf file)
        analysis_dir: path object
            path to write output f5 binary files from TACS analysis (which can be converted to vtk/tec files later)
        """
        assert self.external_shape
        self._dat_file_path = dat_file_path
        self._analysis_dir = analysis_dir
        return

    def _transfer_fixed_aero_loads(self):
        """
        transfer fixed aero loads over to the new
        """
        # loop over each body to copy and transfer loads for the new structure
        shape_init_transfer = self._shape_init_transfer
        for body in self.model.bodies:
            # update the transfer schemes for the new mesh size
            body.update_transfer()

            ns = body.struct_nnodes
            dtype = body.dtype

            if not shape_init_transfer:
                # need to initialize transfer schemes again with tacs_comm
                comm = self.comm
                comm_manager = self.solvers.comm_manager
                body.initialize_transfer(
                    comm=comm,
                    struct_comm=comm_manager.struct_comm,
                    struct_root=comm_manager.struct_root,
                    aero_comm=comm_manager.aero_comm,
                    aero_root=comm_manager.aero_root,
                    transfer_settings=self.transfer_settings,
                )

            # zero the initial struct loads and struct flux for each scenario
            for scenario in self.model.scenarios:
                # initialize new struct shape term for new ns
                nf = scenario.count_adjoint_functions()
                body.struct_shape_term[scenario.id] = np.zeros(
                    (3 * ns, nf), dtype=dtype
                )

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

                # transfer disps to prevent seg fault if coming from Fun3dOnewayDriver
                body.transfer_disps(scenario)
                body.transfer_temps(scenario)

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
            if not self.external_shape:
                # set the new shape variables into the model using update design to prevent CAPS_CLEAN errors
                input_dict = {var.name: var.value for var in self.model.get_variables()}
                self.model.structural.update_design(input_dict)
                self.tacs_aim.setup_aim()

                # build the new structure geometry
                self.tacs_aim.pre_analysis()

            # make the new tacs interface of the structural geometry
            self.tacs_interface = TacsInterface.create_from_bdf(
                model=self.model,
                comm=self.comm,
                nprocs=self.nprocs,
                bdf_file=self.dat_file_path,
                output_dir=self.analysis_dir,
            )

            # make a solvers object to hold structural solver since flow is no longer used
            solvers = SolverManager(self.comm, use_flow=False)
            solvers.structural = self.tacs_interface

            # transfer the fixed aero loads
            self._transfer_fixed_aero_loads()
        # end of change shape section

        if self.steady:
            for scenario in self.model.scenarios:
                self._solve_steady_forward(scenario, self.model.bodies)

        if self.unsteady:
            for scenario in self.model.scenarios:
                self._solve_unsteady_forward(scenario, self.model.bodies)

    def solve_adjoint(self):
        """
        solve the adjoint analysis for the given shape
        assumes the forward analysis for this shape has already been performed
        """

        # run the adjoint structural analysis

        functions = self.model.get_functions()

        # Zero the derivative values stored in the function
        for func in functions:
            func.zero_derivatives()

        if self.steady:
            for scenario in self.model.scenarios:
                self._solve_steady_adjoint(scenario, self.model.bodies)

        if self.unsteady:
            for scenario in self.model.scenarios:
                self._solve_unsteady_adjoint(scenario, self.model.bodies)

        # transfer loads adjoint since fa -> fs has shape dependency
        if self.change_shape:
            # TODO : for unsteady this part might have to be included before extract coordinate derivatives?
            for body in self.model.bodies:
                body.transfer_loads_adjoint(scenario)

        # call get function gradients to store  the gradients from tacs
        self.tacs_interface.get_function_gradients(scenario, self.model.bodies)

        if self.change_shape and not self.external_shape:
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
        return

    def _extract_coordinate_derivatives(self, scenario, bodies, step):
        """extract the coordinate derivatives at a given time step"""
        self.tacs_interface.get_coordinate_derivatives(
            scenario, self.model.bodies, step=step
        )

        # add transfer scheme contributions
        if step > 0:
            for body in bodies:
                body.add_coordinate_derivative(scenario, step=0)

        return

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
                    derivative = direct_tacs_aim.dynout[func.full_name].deriv(var.name)
                    gradients[ifunc].append(derivative)

        # broadcast shape gradients to all other processors
        gradients = self.comm.bcast(gradients, root=0)

        # store shape derivatives in funtofem model on all processors
        for ifunc, func in enumerate(scenario.functions):
            for ivar, var in enumerate(self.shape_variables):
                derivative = gradients[ifunc][ivar]
                func.add_gradient_component(var, derivative)

        return

    def _solve_steady_forward(self, scenario, bodies):
        """
        solve the forward analysis of TACS analysis with aerodynamic loads
        and heat fluxes from previous analysis still retained
        """

        fail = 0

        # run the tacs forward analysis for no shape
        # zero all data to start fresh problem, u = 0, res = 0
        self._zero_tacs_data()

        # set functions and variables
        self.tacs_interface.set_variables(scenario, bodies)
        self.tacs_interface.set_functions(scenario, bodies)

        # run the forward analysis via iterate
        self.tacs_interface.initialize(scenario, bodies)
        self.tacs_interface.iterate(scenario, bodies, step=0)
        self.tacs_interface.post(scenario, bodies)

        # get functions to store the function values into the model
        self.tacs_interface.get_functions(scenario, bodies)

        return 0

    def _solve_unsteady_forward(self, scenario, bodies):
        """
        solve the forward analysis of TACS analysis with aerodynamic loads
        and heat fluxes from previous analysis still retained
        """

        fail = 0

        # set functions and variables
        self.tacs_interface.set_variables(scenario, bodies)
        self.tacs_interface.set_functions(scenario, bodies)

        # run the forward analysis via iterate
        self.tacs_interface.initialize(scenario, bodies)
        for step in range(1, scenario.steps + 1):
            self.tacs_interface.iterate(scenario, bodies, step=step)
        self.tacs_interface.post(scenario, bodies)

        # get functions to store the function values into the model
        self.tacs_interface.get_functions(scenario, bodies)

        return 0

    def _solve_steady_adjoint(self, scenario, bodies):
        """
        solve the adjoint analysis of TACS analysis with aerodynamic loads
        and heat fluxes from previous analysis still retained
        Similar to funtofem_driver
        """

        # zero adjoint data
        self._zero_adjoint_data()

        # set functions and variables
        self.tacs_interface.set_variables(scenario, bodies)
        self.tacs_interface.set_functions(scenario, bodies)

        # zero all coupled adjoint variables in the body
        for body in bodies:
            body.initialize_adjoint_variables(scenario)

        # initialize, run, and do post adjoint
        self.tacs_interface.initialize_adjoint(scenario, bodies)
        self.tacs_interface.iterate_adjoint(scenario, bodies, step=0)
        self._extract_coordinate_derivatives(scenario, bodies, step=0)
        self.tacs_interface.post_adjoint(scenario, bodies)

        return

    def _solve_unsteady_adjoint(self, scenario, bodies):
        """
        solve the adjoint analysis of TACS analysis with aerodynamic loads
        and heat fluxes from previous analysis still retained
        Similar to funtofem_driver
        """

        # set functions and variables
        self.tacs_interface.set_variables(scenario, bodies)
        self.tacs_interface.set_functions(scenario, bodies)

        # zero all coupled adjoint variables in the body
        for body in bodies:
            body.initialize_adjoint_variables(scenario)

        # initialize, run, and do post adjoint
        self.tacs_interface.initialize_adjoint(scenario, bodies)
        for rstep in range(1, scenario.steps + 1):
            step = scenario.steps + 1 - rstep
            self.tacs_interface.iterate_adjoint(scenario, bodies, step=step)
        self.tacs_interface.iterate_adjoint(scenario, bodies, step=0)
        self._extract_coordinate_derivatives(scenario, bodies, step=0)
        self.tacs_interface.iterate_adjoint(scenario, bodies, step=step)
        self.tacs_interface.post_adjoint(scenario, bodies)

        return

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
