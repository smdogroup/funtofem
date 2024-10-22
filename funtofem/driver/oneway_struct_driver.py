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

"""
NOTE : Written by Sean Engelstad, Georgia Tech 2023

This class performs oneway-coupled structural analysis for structural and shape optimization of aircraft structures.
It relies on ESP/CAPS for shape derivatives currently.
"""

# TACS one-way coupled drivers that use fixed fun3d aero loads
__all__ = ["OnewayStructDriver"]

from funtofem.optimization.optimization_manager import OptimizationManager
from mpi4py import MPI
import numpy as np
import shutil, os
import time
import importlib.util
from funtofem.interface import Remote

caps_loader = importlib.util.find_spec("pyCAPS")

# 1) TACS
tacs_loader = importlib.util.find_spec("tacs")

if (
    caps_loader is not None and tacs_loader is not None
):  # tacs loader not None check for this file anyways
    from tacs import caps2tacs

if tacs_loader is not None:
    from funtofem.interface import (
        TacsInterface,
        TacsSteadyInterface,
        TacsUnsteadyInterface,
    )
# 2) TBD: Add additional structural solvers
# ----------------------------------------------------------


class OnewayStructDriver:
    def __init__(
        self,
        solvers,
        model,
        transfer_settings=None,
        nprocs=None,
        fun3d_dir=None,
        external_shape=False,
        timing_file=None,
    ):
        """
        build the analysis driver for shape/no shape change, assumes you have already primed the loads (see class method to assist with that)

        Parameters
        ----------
        solvers: :class:`~interface.solver_manager.SolverManager`
            The various disciplinary solvers.
        model: :class:`~funtofem_model.FUNtoFEMmodel`
            The model containing the design data and the TACSmodel with tacsAIM wrapper inside (if using shape; otherwise can be None).
        transfer_settings: :class:`driver.TransferSettings`
        nprocs: int
            Number of processes that TACS is running on.
        fun3d_dir: os.path object
            input with solvers.flow.fun3d_dir usually
            location of fun3d directory, used in an analysis file for FuntofemShapeDriver
        external_shape: bool
            whether the tacs aim shape analysis is performed outside the class
        timing_file: str or Path object
            file to write timing data to
        """
        self.solvers = solvers
        self.comm = solvers.comm
        self.model = model
        self.nprocs = nprocs
        self.fun3d_dir = fun3d_dir
        self.transfer_settings = transfer_settings
        self.external_shape = external_shape
        self._shape_init_transfer = False
        self.timing_file = timing_file

        self.struct_interface = solvers.structural
        self.struct_aim = None

        # figure out which discipline solver we are using
        self._struct_solver_type = None
        if model.structural is None:
            # TACS solver
            if tacs_loader is not None:
                if isinstance(solvers.structural, TacsSteadyInterface) or isinstance(
                    solvers.structural, TacsUnsteadyInterface
                ):
                    self._struct_solver_type = "tacs"
            # TBD more solvers
        # check for structural AIMs
        if caps_loader is not None and model.structural is not None:
            # TACS solver
            if tacs_loader is not None:
                if isinstance(model.structural, caps2tacs.TacsModel):
                    self._struct_solver_type = "tacs"
                    self.struct_aim = model.structural.tacs_aim
                    assert self.struct_aim.is_setup
            # TBD more solvers

        # store the shape variables list
        self.shape_variables = [
            var for var in self.model.get_variables() if var.analysis_type == "shape"
        ]
        self._unsteady = any([not scenario.steady for scenario in model.scenarios])

        # assertion checks for no shape change vs shape change
        if self.change_shape:
            assert self.struct_aim is not None or external_shape
            if nprocs is None:  # will error if nprocs not defined anywhere
                nprocs = solvers.structural.nprocs
            if external_shape:
                # define properties needed for external shape paths
                self._dat_file_path = None
                self._analysis_dir = None
        else:  # not change shape
            assert self.struct_interface is not None

            # transfer to fixed structural loads in case the user got only aero loads from the OnewayAeroDriver
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

        # make timing file if one was not provided
        if self.change_shape:
            self.timing_file = os.path.join(
                self.struct_aim.root_analysis_dir, "timing.txt"
            )

        # initialize timing file
        self._iteration = 0
        self._write_timing_data(
            "Funtofem OnewayStructDriver Timing data:\n", overwrite=True, root=0
        )

    def _write_timing_data(
        self, msg, overwrite=False, root: int = 0, barrier: bool = False
    ):
        """write to the funtofem timing file"""
        if not (self.timing_file):  # check whether we have a timing file or not
            return
        if self.comm.rank == root:
            hdl = open(self.timing_file, "w" if overwrite else "a")
            hdl.write(msg + "\n")
            hdl.flush()
            hdl.close()

        # MPI Barrier for other processors
        if barrier:
            self.comm.Barrier()
        return

    @property
    def change_shape(self) -> bool:
        return len(self.shape_variables) > 0

    @property
    def steady(self) -> bool:
        return not (self._unsteady)

    @property
    def unsteady(self) -> bool:
        return self._unsteady

    @property
    def uses_tacs(self) -> bool:
        return self._struct_solver_type == "tacs"

    @property
    def analysis_sens_file(self):
        """write location of sens file when used in FuntofemShapeDriver for double oneway drivers (analysis version)"""
        if self.fun3d_dir is None:
            return Remote.paths(self.comm, self.fun3d_dir).struct_sens_file
        else:
            return None

    @classmethod
    def prime_loads(
        cls,
        driver,
        transfer_settings=None,
        nprocs=None,
        external_shape=False,
        fun3d_dir=None,
        timing_file=None,
    ):
        """
        Used to prime struct/aero loads for optimization over TACS analysis.
        Can use the OnewayAeroDriver or FUNtoFEMnlbgs driver to prime the loads
        If structural solver exists, it will transfer to fixed structural loads in __init__ construction
        of the OnewayStructDriver class.
        If shape variables exist in the FUNtoFEMmodel, you need to have model.structural be a TacsModel
        and the TacsModel contains a TacsAim wrapper class. If shape variables exist, shape derivatives
        and shape analysis will be performed.

        Parameters
        ----------
        driver: :class:`OnewayAeroDriver` or :class:`~funtofem_nlbgs_driver.FUNtoFEMnlbgs`
            the fun3d oneway driver or coupled funtofem NLBGS driver

        Optional Parameters
        -------------------
        nprocs: int
            number of procs for tacs analysis, only need to give this if doing shape optimization
        transfer_settings: :class:`driver.TransferSettings`
            used for transferring fixed aero loads to struct loads, need to give this or it uses default
        external_shape: bool
            whether to do shape analysis with ESP/CAPS inside this driver or outside of it
        timing_file: str or path
            location of funtofem timing file
        """
        driver.solve_forward()
        if transfer_settings is None:
            try:
                transfer_settings = driver.transfer_settings
            except:
                transfer_settings = transfer_settings
        return cls(
            driver.solvers,
            model=driver.model,
            transfer_settings=transfer_settings,
            nprocs=nprocs,
            external_shape=external_shape,
            fun3d_dir=fun3d_dir,
            timing_file=timing_file,
        )

    @classmethod
    def prime_loads_from_unsteady_files(
        cls,
        files: list,
        solvers,
        model,
        nprocs,
        transfer_settings,
        external_shape=False,
        init_transfer=False,
        timing_file=None,
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
        struct_aim: `caps2tacs.TacsAim`
            Interface object from TACS to ESP/CAPS, wraps the tacsAIM object.
        external_shape: bool
            whether the tacs aim shape analysis is performed outside this class
        timing_file: str or path
            path to funtofem timing file statistics
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

        for itime, file in enumerate(files):
            # read in the loads from the file for each time step
            loads_data = model._read_aero_loads(comm, file)

            # initialize the transfer scheme then distribute aero loads
            for body in model.bodies:
                if (
                    itime == 0
                ):  # only initialize transfer and scenario data on the first time step after the aero mesh is loaded
                    body.initialize_transfer(
                        comm=comm,
                        struct_comm=tacs_comm,
                        struct_root=comm_manager.struct_root,
                        aero_comm=comm_manager.aero_comm,
                        aero_root=comm_manager.aero_root,
                        transfer_settings=transfer_settings,
                    )
                    for scenario in model.scenarios:
                        assert not scenario.steady
                        body.initialize_variables(scenario)

                body._distribute_aero_loads(loads_data, steady=False, itime=itime)

        tacs_driver = cls(
            solvers,
            model,
            nprocs=nprocs,
            external_shape=external_shape,
            timing_file=timing_file,
        )
        if init_transfer:
            tacs_driver._transfer_fixed_aero_loads()
        return tacs_driver

    @classmethod
    def prime_loads_from_file(
        cls,
        filename,
        solvers,
        model,
        nprocs,
        transfer_settings,
        external_shape=False,
        init_transfer=False,
        timing_file=None,
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
        struct_aim: `caps2tacs.TacsAim`
            Interface object from TACS to ESP/CAPS, wraps the tacsAIM object.
        external_shape: bool
            whether the tacs aim shape analysis is performed outside this class
        timing_file: str or path
            path to funtofem timing file statistics
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
        loads_data = model._read_aero_loads(comm, filename)

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
                assert scenario.steady
            body._distribute_aero_loads(loads_data, steady=True)

        tacs_driver = cls(
            solvers,
            model,
            nprocs=nprocs,
            external_shape=external_shape,
            timing_file=timing_file,
        )
        if init_transfer:
            tacs_driver._transfer_fixed_aero_loads()
        return tacs_driver

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
            return self.struct_interface.tacs_comm

    @property
    def root_proc(self) -> bool:
        return self.comm.rank == 0

    @property
    def dat_file_path(self):
        if self.external_shape:
            return self._dat_file_path
        else:
            return self.struct_aim.root_dat_file

    @property
    def analysis_dir(self):
        if self.external_shape:
            return self._analysis_dir
        else:
            return self.struct_aim.root_analysis_dir

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
                if scenario.steady:
                    if body.transfer is not None:
                        body.struct_loads[scenario.id] = np.zeros(3 * ns, dtype=dtype)
                        body.struct_disps[scenario.id] = np.zeros(3 * ns, dtype=dtype)

                    # initialize new struct heat flux
                    if body.thermal_transfer is not None:
                        body.struct_heat_flux[scenario.id] = np.zeros(ns, dtype=dtype)
                        body.struct_temps[scenario.id] = (
                            np.ones(ns, dtype=dtype) * scenario.T_ref
                        )

                else:  # unsteady
                    if body.transfer is not None:
                        body.struct_loads[scenario.id] = [
                            np.zeros(3 * ns, dtype=dtype) for _ in range(scenario.steps)
                        ]
                        body.struct_disps[scenario.id] = [
                            np.zeros(3 * ns, dtype=dtype) for _ in range(scenario.steps)
                        ]

                    # initialize new struct heat flux
                    if body.thermal_transfer is not None:
                        body.struct_heat_flux[scenario.id] = [
                            np.zeros(ns, dtype=dtype) for _ in range(scenario.steps)
                        ]
                        body.struct_temps[scenario.id] = [
                            (np.ones(ns, dtype=dtype) * scenario.T_ref)
                            for _ in range(scenario.steps)
                        ]

                # transfer disps to prevent seg fault if coming from OnewayAeroDriver
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

        self._write_timing_data(
            f"Starting iteration {self._iteration}", overwrite=False, root=0
        )
        self._starting_time = time.time()
        _start_meshing_time = self._starting_time * 1.0
        self._iteration += 1

        if self.change_shape:
            if not self.external_shape:
                input_dict = {var.name: var.value for var in self.model.get_variables()}
                self.model.structural.update_design(input_dict)
                self.struct_aim.setup_aim()
                self.struct_aim.pre_analysis()

            if self.uses_tacs:
                # make the new tacs interface of the structural geometry
                self.struct_interface = TacsInterface.create_from_bdf(
                    model=self.model,
                    comm=self.comm,
                    nprocs=self.nprocs,
                    bdf_file=self.dat_file_path,
                    output_dir=self.analysis_dir,
                )
            # TBD more structural solvers here

            # transfer the fixed aero loads
            self._transfer_fixed_aero_loads()
        # end of change shape section

        dt_meshing = (time.time() - _start_meshing_time) / 60.0
        self._write_timing_data(
            f"\tstruct mesh built in {dt_meshing} min",
            overwrite=False,
            barrier=True,
            root=0,
        )
        _start_forward_analysis = time.time()

        if self.steady:
            for scenario in self.model.scenarios:
                self._solve_steady_forward(scenario, self.model.bodies)

        if self.unsteady:
            for scenario in self.model.scenarios:
                self._solve_unsteady_forward(scenario, self.model.bodies)

        dt_forward = (time.time() - _start_forward_analysis) / 60.0
        self._write_timing_data(
            f"\tstruct forward analysis in {dt_forward} min",
            overwrite=False,
            barrier=True,
            root=0,
        )

    def solve_adjoint(self):
        """
        solve the adjoint analysis for the given shape
        assumes the forward analysis for this shape has already been performed
        """

        # timing data
        _start_adjoint = time.time()

        # run the adjoint structural analysis
        functions = self.model.get_functions()

        # Zero the derivative values stored in the function
        self._zero_derivatives()
        for func in functions:
            func.zero_derivatives()

        if self.steady:
            for scenario in self.model.scenarios:
                self._solve_steady_adjoint(scenario, self.model.bodies)

        if self.unsteady:
            for scenario in self.model.scenarios:
                self._solve_unsteady_adjoint(scenario, self.model.bodies)

        dt_adjoint = (time.time() - _start_adjoint) / 60.0
        self._write_timing_data(
            f"\tstruct adjoint analysis in {dt_adjoint} min",
            overwrite=False,
            barrier=True,
            root=0,
        )
        _start_derivatives = time.time()

        # transfer loads adjoint since fa -> fs has shape dependency
        if self.change_shape:
            # TODO : for unsteady this part might have to be included before extract coordinate derivatives?
            for body in self.model.bodies:
                body.transfer_loads_adjoint(scenario)

        # call get function gradients to store  the gradients from tacs
        self.struct_interface.get_function_gradients(scenario, self.model.bodies)

        if self.change_shape and not self.external_shape:
            # write the sensitivity file for the tacs AIM
            self.model.write_sensitivity_file(
                comm=self.comm,
                filename=(
                    self.struct_aim.root_sens_file
                    if not self.fun3d_dir
                    else self.analysis_sens_file
                ),
                discipline="structural",
            )

            self.comm.Barrier()

            # delete struct interface to free up memory in shape change
            # self.struct_interface._deallocate()
            # self.comm.Barrier()
            del self.struct_interface
            self.comm.Barrier()

            # copy struct sens files for parallel instances
            if self.uses_tacs:
                src = self.struct_aim.root_sens_file
                for proc in self.struct_aim.active_procs[1:]:
                    dest = self.struct_aim.sens_file_path(proc)
                    self.comm.Barrier()
                    if self.comm.rank == self.struct_aim.root_proc_ind:
                        shutil.copy(src, dest)

            # wait til the file is done writing on other procs
            self.comm.Barrier()

            # run the tacs aim postAnalysis to compute the chain rule product
            self.struct_aim.post_analysis()

            # wait til parallel tacsAIM instances run post_analysis before getting shape derivatives
            self.comm.Barrier()

            # store the shape variables in the function gradients
            for scenario in self.model.scenarios:
                self._get_shape_derivatives(scenario)

        dt_derivatives = (time.time() - _start_derivatives) / 60.0
        self._write_timing_data(
            f"\tderivative computation in {dt_derivatives} min",
            overwrite=False,
            barrier=True,
            root=0,
        )
        dt_iteration = (time.time() - self._starting_time) / 60.0
        self._write_timing_data(
            f"\titeration {self._iteration-1} took {dt_iteration} min",
            overwrite=False,
            barrier=True,
            root=0,
        )

        # end of change shape section
        return

    def _zero_derivatives(self):
        """zero all model derivatives"""
        for func in self.model.get_functions(all=True):
            for var in self.model.get_variables():
                func.derivatives[var] = 0.0
        return

    def _extract_coordinate_derivatives(self, scenario, bodies, step):
        """extract the coordinate derivatives at a given time step"""
        self.struct_interface.get_coordinate_derivatives(
            scenario, self.model.bodies, step=step
        )

        # add transfer scheme contributions
        if step > 0:
            for body in bodies:
                body.add_coordinate_derivative(scenario, step=0)

        return

    def _update_design(self):
        if self.comm.rank == 0:
            input_dict = {var.name: var.value for var in self.model.get_variables()}
            for key in input_dict:
                print(f"updating design for key = {key}", flush=True)
                if self.struct_aim.geometry.despmtr[key].value != input_dict[key]:
                    self.struct_aim.geometry.despmtr[key].value = input_dict[key]
        return

    def _get_shape_derivatives(self, scenario):
        """
        get shape derivatives together from tacs aim
        and store the data in the funtofem model
        """
        variables = self.model.get_variables()

        # read shape gradients from tacs aim among different processors
        # including sometimes parallel versions of the struct AIM
        gradients = []

        for ifunc, func in enumerate(scenario.functions):
            gradients.append([])
            for ivar, var in enumerate(variables):
                derivative = None
                if var.analysis_type == "shape":
                    # if tacs aim do this, make this more modular later
                    if self.uses_tacs:  # for parallel tacsAIMs
                        c_proc = self.struct_aim.get_proc_with_shape_var(var.name)
                        if self.comm.rank == c_proc:
                            derivative = self.struct_aim.aim.dynout[
                                func.full_name
                            ].deriv(var.name)
                        # then broadcast the derivative to other processors
                        derivative = self.comm.bcast(derivative, root=c_proc)
                    else:
                        if self.root_proc:
                            derivative = self.struct_aim.aim.dynout[
                                func.full_name
                            ].deriv(var.name)
                else:
                    derivative = 0.0

                # updat the derivatives list
                gradients[ifunc].append(derivative)

        # mpi comm barrier
        self.comm.Barrier()

        # store shape derivatives in funtofem model on all processors
        for ifunc, func in enumerate(scenario.functions):
            for ivar, var in enumerate(variables):
                derivative = gradients[ifunc][ivar]
                func.add_gradient_component(var, gradients[ifunc][ivar])

        return

    def _solve_steady_forward(self, scenario, bodies):
        """
        solve the forward analysis of TACS analysis with aerodynamic loads
        and heat fluxes from previous analysis still retained
        """

        fail = 0

        # run the tacs forward analysis for no shape
        # zero all data to start fresh problem, u = 0, res = 0
        if self.uses_tacs:
            self._zero_tacs_data()

        # set functions and variables
        self.struct_interface.set_variables(scenario, bodies)
        self.struct_interface.set_functions(scenario, bodies)

        # run the forward analysis via iterate
        self.struct_interface.initialize(scenario, bodies)
        self.struct_interface.iterate(scenario, bodies, step=0)
        self.struct_interface.post(scenario, bodies)

        # get functions to store the function values into the model
        self.struct_interface.get_functions(scenario, bodies)

        return 0

    def _solve_unsteady_forward(self, scenario, bodies):
        """
        solve the forward analysis of TACS analysis with aerodynamic loads
        and heat fluxes from previous analysis still retained
        """

        fail = 0

        # set functions and variables
        self.struct_interface.set_variables(scenario, bodies)
        self.struct_interface.set_functions(scenario, bodies)

        # run the forward analysis via iterate
        self.struct_interface.initialize(scenario, bodies)
        for step in range(1, scenario.steps + 1):
            self.struct_interface.iterate(scenario, bodies, step=step)
        self.struct_interface.post(scenario, bodies)

        # get functions to store the function values into the model
        self.struct_interface.get_functions(scenario, bodies)

        return 0

    def _solve_steady_adjoint(self, scenario, bodies):
        """
        solve the adjoint analysis of TACS analysis with aerodynamic loads
        and heat fluxes from previous analysis still retained
        Similar to funtofem_driver
        """

        # zero adjoint data
        if self.uses_tacs:
            self._zero_adjoint_data()

        # set functions and variables
        self.struct_interface.set_variables(scenario, bodies)
        self.struct_interface.set_functions(scenario, bodies)

        # zero all coupled adjoint variables in the body
        for body in bodies:
            body.initialize_adjoint_variables(scenario)

        # initialize, run, and do post adjoint
        self.struct_interface.initialize_adjoint(scenario, bodies)
        self.struct_interface.iterate_adjoint(scenario, bodies, step=0)
        self._extract_coordinate_derivatives(scenario, bodies, step=0)
        self.struct_interface.post_adjoint(scenario, bodies)

        return

    def _solve_unsteady_adjoint(self, scenario, bodies):
        """
        solve the adjoint analysis of TACS analysis with aerodynamic loads
        and heat fluxes from previous analysis still retained
        Similar to funtofem_driver
        """

        # set functions and variables
        self.struct_interface.set_variables(scenario, bodies)
        self.struct_interface.set_functions(scenario, bodies)

        # zero all coupled adjoint variables in the body
        for body in bodies:
            body.initialize_adjoint_variables(scenario)

        # initialize, run, and do post adjoint
        self.struct_interface.initialize_adjoint(scenario, bodies)
        for rstep in range(1, scenario.steps + 1):
            step = scenario.steps + 1 - rstep
            self.struct_interface.iterate_adjoint(scenario, bodies, step=step)
        self.struct_interface.iterate_adjoint(scenario, bodies, step=0)
        self._extract_coordinate_derivatives(scenario, bodies, step=0)
        self.struct_interface.iterate_adjoint(scenario, bodies, step=step)
        self.struct_interface.post_adjoint(scenario, bodies)

        return

    def _zero_tacs_data(self):
        """
        zero any TACS solution / adjoint data before running pure TACS
        """

        if self.struct_interface.tacs_proc:
            # zero temporary solution data
            # others are zeroed out in the struct_interface by default
            self.struct_interface.res.zeroEntries()
            self.struct_interface.ext_force.zeroEntries()
            self.struct_interface.update.zeroEntries()

            # zero any scenario data
            for scenario in self.model.scenarios:
                # zero state data
                u = self.struct_interface.scenario_data[scenario].u
                u.zeroEntries()
                self.struct_interface.assembler.setVariables(u)

    def _zero_adjoint_data(self):
        if self.struct_interface.tacs_proc:
            # zero adjoint variable
            for scenario in self.model.scenarios:
                psi = self.struct_interface.scenario_data[scenario].psi
                for vec in psi:
                    vec.zeroEntries()
