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


# FUN3D one-way coupled drivers that use fixed fun3d aero loads
__all__ = ["Fun3dOnewayDriver", "Fun3dRemote"]

import numpy as np

"""
Unfortunately, FUN3D has to be completely re-initialized for new aerodynamic meshes, so we have
to split our Fun3dOnewayDriver scripts in implementation into two files, a my_fun3d_driver.py and a my_fun3d_analyzer.py.
The file my_fun3d_driver.py is called from a run.pbs script and manages the optimization and AIMs; this file
also uses system calls to the file my_fun3d_analyzer.py which runs the FUN3D analysis for each mesh. There are two 
class methods Fun3dOnewayDriver.remote and Fun3dOnewayDriver.analysis which build the drivers for each of the two files.

NOTE : only aerodynamic functions can be evaluated from this driver. If you only need aerodynamic DVs, you should build
the driver with class method Fun3dOnewayDriver.analysis(). If you need shape derivatives through ESP/CAPS, you shoud
build the driver class using the class method Fun3dOnewayDriver.remote() and setup a separate script with a driver running the analysis.
More details on these two files are provided below. Do not construct a driver with analysis() class method and shape DVs,
it will error in FUN3D upon the 2nd analysis iteration.

my_fun3d_driver.py : main driver script which called from the run.pbs
    NOTE : similar to tests/fun3d_tests/test_fun3d_oneway_shape.py
    - Construct the FUNtoFEMmodel
    - Build the Fun3dModel and link with Fun3dAim + AflrAIM and mesh settings, then store in funtofem_model.flow = this
    - Construct bodies and scenarios
    - Register aerodynamic and shape DVs to the scenarios/bodies
    - Construct the SolverManager with comm, but leave flow and structural attributes empty
    - Construct the fun3d oneway driver with class method Fun3dOnewayDriver.remote to manage system calls to the other script.
    - Build the optimization manager / run the driver

my_fun3d_analyzer.py : fun3d analysis script, which is called indirectly from my_fun3d_driver.py
    NOTE : similar to tests/fun3d_tests/run_fun3d_analysis.py
    - Construct the FUNtoFEMmodel
    - Construct the bodies and scenarios
    - Register aerodynamic DVs to the scenarios/bodies (no shape variables added and no AIMs here)
    - Construct the Fun3dInterface
    - Construct the solvers (SolverManager), and set solvers.flow = my_fun3d_interface
    - Construct the a fun3d oneway driver with class method Fun3dOnewayDriver.analysis
    - Run solve_forward() and solve_adjoint() on the Fun3dOnewayAnalyzer

For an example implementation see tests/fun3d_tests/ folder with test_fun3d_oneway_aero.py for just aero DVs
and (test_fun3d_oneway_driver.py => run_fun3d_analysis.py) pair of files for the shape DVs using the Fun3dRemote and system calls.
"""

import os
from pyfuntofem.driver import TransferSettings
from pyfuntofem.optimization.optimization_manager import OptimizationManager

import importlib.util

fun3d_loader = importlib.util.find_spec("fun3d")
if fun3d_loader is not None:  # check whether we can import FUN3D
    from pyfuntofem.interface import Fun3dInterface


class Fun3dRemote:
    def __init__(
        self,
        analysis_file,
        fun3d_dir,
        output_name="f2f_analysis",
        nprocs=1,
        aero_name="fun3d",
        struct_name="tacs",
    ):
        """

        Manages remote analysis calls for a FUN3D / FUNtoFEM driver call

        Parameters
        ----------
        nprocs: int
            number of procs for the system call to the Fun3dOnewayAnalyzer
        analyzer_file: os filepath
            the location of the subprocess file for the Fun3dOnewayAnalyzer (my_fun3d_analyzer.py)
        fun3d_dir: filepath
            location of the fun3d directory for meshes, one level above the scenario folders
        output_file: filepath
            optional location to write an output file for the forward and adjoint analysis
        """
        self.analysis_file = analysis_file
        self.fun3d_dir = fun3d_dir
        self.nprocs = nprocs
        self.output_name = output_name
        self.aero_name = aero_name
        self.struct_name = struct_name

    @classmethod
    def paths(cls, fun3d_dir, aero_name="fun3d", struct_name="struct"):
        return cls(
            analysis_file=None,
            fun3d_dir=fun3d_dir,
            aero_name=aero_name,
            struct_name=struct_name,
        )

    @classmethod
    def fun3d_path(cls, fun3d_dir, filename):
        return os.path.join(fun3d_dir, filename)

    @property
    def struct_sens_file(self):
        return os.path.join(self.fun3d_dir, f"{self.struct_name}.sens")

    @property
    def aero_sens_file(self):
        return os.path.join(self.fun3d_dir, f"{self.aero_name}.sens")

    @property
    def output_file(self):
        return os.path.join(self.fun3d_dir, f"{self.output_name}.txt")

    @property
    def bdf_file(self):
        return os.path.join(self.fun3d_dir, f"{self.struct_name}.bdf")

    @property
    def dat_file(self):
        return os.path.join(self.fun3d_dir, f"{self.struct_name}.dat")

    @property
    def design_file(self):
        return os.path.join(self.fun3d_dir, "funtofem.in")


class Fun3dOnewayDriver:
    @classmethod
    def nominal(cls, solvers, model, transfer_settings=None):
        """
        build an Fun3dOnewayDriver with Fun3dAim shape variables and Fun3d analysis
        all in one, using the Fun3d mesh morphing
        """
        return cls(solvers, model, transfer_settings=transfer_settings, is_paired=False)

    @classmethod
    def remote(cls, solvers, model, fun3d_remote):
        """
        build a Fun3dOnewayDriver object for the my_fun3d_driver.py script:
            this object would be responsible for the fun3d, aflr AIMs and

        """
        return cls(solvers, model, fun3d_remote=fun3d_remote, is_paired=True)

    @classmethod
    def analysis(cls, solvers, model, transfer_settings=None):
        """
        build an Fun3dOnewayDriver object for the my_fun3d_analyzer.py script:
            this object would be responsible for running the FUN3D
            analysis and writing an aero.sens file to the fun3d directory
        If you are using the analysis driver by itself like for FUN3D mesh morphing turn is_paired off
        """
        return cls(solvers, model, transfer_settings=transfer_settings, is_paired=True)

    def __init__(
        self,
        solvers,
        model,
        transfer_settings=None,
        fun3d_remote=None,
        is_paired=False,
    ):
        """
        build the FUN3D analysis driver for shape/no shape change, can run another FUN3D analysis remotely or
        do it internally depending on how the object is constructed. The user is recommended to use the class methods
        to build the driver most of the time not the main constructor.

        Parameters
        ----------
        solvers: :class:`~pyfuntofem.interface.SolverManager'
            no need to add solvers.flow here just give it the comm so we can setup empty body class
        model: :class:`~funtofem_model.FUNtoFEMmodel`
            The model containing the design data.
        is_paired: bool
            Whether you need a pair of drivers to one remote and one analysis or just one analysis driver
        """
        self.solvers = solvers
        self.comm = solvers.comm
        self.model = model
        self.transfer_settings = transfer_settings
        self.fun3d_remote = fun3d_remote
        self.is_paired = is_paired

        # store the shape variables list
        self.shape_variables = [
            var for var in self.model.get_variables() if var.analysis_type == "shape"
        ]

        # make sure there is shape change otherwise they should just use Fun3dOnewayAnalyzer
        if not (self.change_shape) and self.is_remote:
            raise AssertionError(
                "Need shape variables to use the Fun3dOnewayDriver otherwise use the Fun3dOnewayAnalyzer which duals as the driver for no shape DVs."
            )

        if self.is_remote and self.model.flow is not None:
            if self.model.flow.mesh_morph:
                raise AssertionError(
                    "The mesh morphing does not require a remote driver! Make this driver regularly!"
                )

        if not self.is_remote:
            if self.model.flow is not None:
                if not self.is_paired and not self.model.flow.mesh_morph:
                    raise AssertionError(
                        "The nominal version of the driver only works for Fun3d mesh morphing not remeshing."
                    )

            assert isinstance(self.solvers.flow, Fun3dInterface)
            if self.change_shape and self.root_proc:
                print(
                    f"Warning!! You are trying to remesh without using remote system calls of FUN3D, this will likely cause a FUN3D bug."
                )

        # check for unsteady problems
        self._unsteady = False
        for scenario in model.scenarios:
            if not scenario.steady:
                self._unsteady = True
                break

        # get the fun3d aim for changing shape
        if model.flow is None:
            fun3d_aim = None
        else:
            fun3d_aim = model.flow.fun3d_aim
        self.fun3d_aim = fun3d_aim

        comm = solvers.comm
        comm_manager = solvers.comm_manager

        if transfer_settings is not None:
            transfer_settings = TransferSettings()  # default

        # initialize variables
        for body in self.model.bodies:
            # transfer to fixed structural loads in case the user got only aero loads from the Fun3dOnewayDriver
            body.initialize_transfer(
                comm=comm,
                struct_comm=comm_manager.struct_comm,
                struct_root=comm_manager.struct_root,
                aero_comm=comm_manager.aero_comm,
                aero_root=comm_manager.aero_root,
                transfer_settings=transfer_settings,  # using minimal settings since we don't use the state variables here (almost a dummy obj)
            )
            for scenario in model.scenarios:
                body.initialize_variables(scenario)

        # shape optimization
        if self.change_shape:
            assert fun3d_aim is not None
            assert self.fun3d_model.is_setup
            self._setup_grid_filepaths()
        else:
            # TODO :
            for body in self.model.bodies:
                body.update_transfer()
                # local_ns = body.struct_nnodes
                # global_ns = self.comm.Reduce(local_ns,root=0)
                # if body.struct_nnodes > 0:
                #     for scenario in self.model.scenarios:
                #        # perform disps transfer from fixed struct to aero disps
                #        body.transfer_disps(scenario)
                #        body.transfer_temps(scenario)
        # end of __init__ method

    def solve_forward(self):
        """
        forward analysis for the given shape and functionals
        assumes shape variables have already been changed
        """

        if self.change_shape:
            # run the pre analysis to generate a new mesh
            self.fun3d_aim.pre_analysis()

            # doing the mesh morph inside of Fortran now but have to move the surf.dat file into the Flow directory for each scenario
            # other way of mesh morph
            # if self.model.flow.mesh_morph:
            #    self.model.write_fun3d_surface_file(self.comm, os.path.join(self.fun3d_interface.fun3d_dir, "nominal_surf.dat"), root=0)
            #    self.model.read_fun3d_surface_file(self.comm, root=0)

        # system call FUN3D forward analysis
        if self.is_remote:
            # write the funtofem design input file
            self.model.write_design_variables_file(
                self.comm,
                filename=Fun3dRemote.paths(self.fun3d_remote.fun3d_dir).design_file,
                root=0,
            )

            # clear the output file
            if self.root_proc and os.path.exists(self.fun3d_remote.output_file):
                os.remove(self.fun3d_remote.output_file)

            # system call the analysis driver
            os.system(
                f"mpiexec_mpt -n {self.fun3d_remote.nprocs} python {self.fun3d_remote.analysis_file} 2>&1 > {self.fun3d_remote.output_file}"
            )

        else:  # non-remote call of FUN3D forward analysis
            # read in the funtofem design input file
            self.model.read_design_variables_file(
                self.comm,
                filename=Fun3dRemote.paths(self.fun3d_interface.fun3d_dir).design_file,
                root=0,
            )

            # run the FUN3D forward analysis with no shape change
            if self.steady:
                for scenario in self.model.scenarios:
                    self._solve_steady_forward(scenario, self.model.bodies)

            if self.unsteady:
                for scenario in self.model.scenarios:
                    self._solve_unsteady_forward(scenario, self.model.bodies)

        if self.change_shape and self.fun3d_aim.mesh_morph:
            self.fun3d_aim.unlink()
        return

    def solve_adjoint(self):
        """
        solve the adjoint analysis for the given shape
        assumes the forward analysis for this shape has already been performed
        """

        # run the adjoint aerodynamic analysis
        functions = self.model.get_functions()

        # Zero the derivative values stored in the function
        for func in functions:
            func.zero_derivatives()

        if not (self.is_remote):
            if self.steady:
                for scenario in self.model.scenarios:
                    self._solve_steady_adjoint(scenario, self.model.bodies)

            if self.unsteady:
                for scenario in self.model.scenarios:
                    self._solve_unsteady_adjoint(scenario, self.model.bodies)

            # write sens file for remote to read or if shape change all in one
            if not self.is_paired:
                filepath = self.model.flow.fun3d_aim.sens_file_path
            else:
                filepath = Fun3dRemote.paths(
                    self.fun3d_interface.fun3d_dir
                ).aero_sens_file

            # write the sensitivity file for the FUN3D AIM
            self.model.write_sensitivity_file(
                comm=self.comm,
                filename=filepath,
                discipline="aerodynamic",
            )

        # shape derivative section
        if self.change_shape:  # either remote or regular
            # src for movement of sens file or None if not moving it
            sens_file_src = self.fun3d_remote.aero_sens_file if self.is_paired else None

            # run the tacs aim postAnalysis to compute the chain rule product
            self.fun3d_aim.post_analysis(sens_file_src)

            # update function values, NOTE : function values are not available in the remote version of the driver
            # after solve_forward (if you just need one grid and solve_forward, you don't need a remote driver, build the analysis one)
            self._get_functions()

            # store the shape variables in the function gradients
            for scenario in self.model.scenarios:
                self._get_shape_derivatives(scenario)
        return

    @property
    def steady(self) -> bool:
        return not (self._unsteady)

    @property
    def unsteady(self) -> bool:
        return self._unsteady

    @property
    def manager(self, hot_start: bool = False):
        return OptimizationManager(self, hot_start=hot_start)

    @property
    def root_proc(self) -> bool:
        return self.comm.rank == 0

    def _transfer_fixed_struct_disps(self):
        """
        transfer fixed struct disps over to the new aero mesh for shape change
        """
        # TODO : set this up for shape change from struct to aero disps
        return

    def _solve_steady_forward(self, scenario, bodies):
        # set functions and variables
        self.fun3d_interface.set_variables(scenario, bodies)
        self.fun3d_interface.set_functions(scenario, bodies)

        # run the forward analysis via iterate
        self.fun3d_interface.initialize(scenario, bodies)
        for step in range(1, scenario.steps + 1):
            self.fun3d_interface.iterate(scenario, bodies, step=0)
        self.fun3d_interface.post(scenario, bodies)

        # get functions to store the function values into the model
        self.fun3d_interface.get_functions(scenario, bodies)
        return

    def _solve_unsteady_forward(self, scenario, bodies):
        # set functions and variables
        self.fun3d_interface.set_variables(scenario, bodies)
        self.fun3d_interface.set_functions(scenario, bodies)

        # run the forward analysis via iterate
        self.fun3d_interface.initialize(scenario, bodies)
        for step in range(1, scenario.steps + 1):
            self.fun3d_interface.iterate(scenario, bodies, step=step)
        self.fun3d_interface.post(scenario, bodies)

        # get functions to store the function values into the model
        self.fun3d_interface.get_functions(scenario, bodies)
        return

    def _solve_steady_adjoint(self, scenario, bodies):
        if scenario.adjoint_steps is None:
            steps = scenario.steps
        else:
            steps = scenario.adjoint_steps

        # set functions and variables
        self.fun3d_interface.set_variables(scenario, bodies)
        self.fun3d_interface.set_functions(scenario, bodies)

        # zero all coupled adjoint variables in the body
        for body in bodies:
            body.initialize_adjoint_variables(scenario)

        # initialize, run, and do post adjoint
        self.fun3d_interface.initialize_adjoint(scenario, bodies)
        for step in range(1, steps + 1):
            self.fun3d_interface.iterate_adjoint(scenario, bodies, step=step)
        self._extract_coordinate_derivatives(scenario, bodies, step=0)
        self.fun3d_interface.post_adjoint(scenario, bodies)

        # transfer disps adjoint since fa -> fs has shape dependency
        # if self.change_shape:
        #     for body in bodies:
        #         body.transfer_disps_adjoint(scenario)

        # call get function gradients to store the gradients w.r.t. aero DVs from FUN3D
        self.fun3d_interface.get_function_gradients(scenario, bodies)
        return

    def _solve_unsteady_adjoint(self, scenario, bodies):
        # set functions and variables
        self.fun3d_interface.set_variables(scenario, bodies)
        self.fun3d_interface.set_functions(scenario, bodies)

        # zero all coupled adjoint variables in the body
        for body in bodies:
            body.initialize_adjoint_variables(scenario)

        # initialize, run, and do post adjoint
        self.fun3d_interface.initialize_adjoint(scenario, bodies)
        for rstep in range(scenario.steps + 1):
            step = scenario.steps + 1 - rstep
            self.fun3d_interface.iterate_adjoint(scenario, bodies, step=step)
            self._extract_coordinate_derivatives(scenario, bodies, step=step)
        self.fun3d_interface.post_adjoint(scenario, bodies)
        self._extract_coordinate_derivatives(scenario, bodies, step=0)

        # transfer disps adjoint since fa -> fs has shape dependency
        # if self.change_shape:
        #     for body in bodies:
        #         body.transfer_disps_adjoint(scenario)

        # call get function gradients to store the gradients w.r.t. aero DVs from FUN3D
        self.fun3d_interface.get_function_gradients(scenario, bodies)
        return

    @property
    def fun3d_interface(self):
        return self.solvers.flow

    @property
    def is_remote(self) -> bool:
        """whether we are calling FUN3D in a remote manner"""
        return self.fun3d_remote is not None

    @property
    def fun3d_model(self):
        return self.model.flow

    @property
    def change_shape(self) -> bool:
        return len(self.shape_variables) > 0

    def _setup_grid_filepaths(self):
        """setup the filepaths for each fun3d grid file in scenarios"""
        if self.fun3d_interface is not None:
            fun3d_dir = self.fun3d_interface.fun3d_dir
        else:
            fun3d_dir = self.fun3d_remote.fun3d_dir
        grid_filepaths = []
        for scenario in self.model.scenarios:
            filepath = os.path.join(
                fun3d_dir,
                scenario.name,
                "Flow",
                f"{scenario.fun3d_project_name}.lb8.ugrid",
            )
            grid_filepaths.append(filepath)
        # set the grid filepaths into the fun3d aim
        self.fun3d_aim.grid_filepaths = grid_filepaths
        return

    @property
    def manager(self, hot_start: bool = False):
        return OptimizationManager(self, hot_start=hot_start)

    @property
    def root_proc(self) -> bool:
        return self.comm.rank == 0

    def _extract_coordinate_derivatives(self, scenario, bodies, step):
        """extract the coordinate derivatives at a given time step"""
        self.fun3d_interface.get_coordinate_derivatives(
            scenario, self.model.bodies, step=step
        )

        # add transfer scheme contributions
        if step > 0:
            for body in bodies:
                body.add_coordinate_derivative(scenario, step=0)

        return

    def _get_functions(self):
        """
        read function values from fun3dAIM when operating in the remote version of the driver
        """
        print(f"Entering get remote functions...", flush=True)
        functions = self.model.get_functions()
        nfunc = len(functions)
        remote_functions = None
        if self.fun3d_aim.root_proc:
            remote_functions = np.zeros((nfunc))
            direct_fun3d_aim = self.fun3d_aim.aim
            for ifunc, func in enumerate(functions):
                remote_functions[ifunc] = direct_fun3d_aim.dynout[func.full_name].value

        # broadcast the function values to other processors
        fun3d_aim_root = self.fun3d_aim.root
        remote_functions = self.comm.bcast(remote_functions, root=fun3d_aim_root)

        # update model function values in the remote version of the driver
        for ifunc, func in enumerate(functions):
            func.value = remote_functions[ifunc]
            if (
                self.comm.rank == 0
            ):  # debug print out for func values to check if read in properly
                print(f"function {func.name} = {func.value}")
        return

    def _get_shape_derivatives(self, scenario):
        """
        get shape derivatives together from FUN3D aim
        and store the data in the funtofem model
        """
        gradients = None

        # read shape gradients from tacs aim on root proc
        fun3d_aim_root = self.fun3d_aim.root
        if self.fun3d_aim.root_proc:
            gradients = []
            direct_fun3d_aim = self.fun3d_aim.aim

            for ifunc, func in enumerate(scenario.functions):
                gradients.append([])
                for ivar, var in enumerate(self.shape_variables):
                    derivative = direct_fun3d_aim.dynout[func.full_name].deriv(var.name)
                    gradients[ifunc].append(derivative)

        # broadcast shape gradients to all other processors
        gradients = self.comm.bcast(gradients, root=fun3d_aim_root)

        # store shape derivatives in funtofem model on all processors
        for ifunc, func in enumerate(scenario.functions):
            for ivar, var in enumerate(self.shape_variables):
                derivative = gradients[ifunc][ivar]
                func.add_gradient_component(var, derivative)

        return

    # @classmethod
    # def prime_disps(cls, funtofem_driver):
    #     """
    #     Used to prime aero disps for optimization over FUN3D analysis with no shape variables

    #     Parameters
    #     ----------
    #     funtofem_driver: :class:`~funtofem_nlbgs_driver.FUNtoFEMnlbgs`
    #         the coupled funtofem NLBGS driver
    #     """
    #     # TODO : this is currently not very usable since we use system calls to separate analyses
    #     # unless we made a disps read in file (but not high priority)
    #     funtofem_driver.solve_forward()
    #     return cls(funtofem_driver.solvers, funtofem_driver.model, nominal=False)
