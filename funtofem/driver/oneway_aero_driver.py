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
__all__ = ["OnewayAeroDriver"]

"""
Written by Sean Engelstad, Georgia Tech 2023

Unfortunately, FUN3D has to be completely re-initialized for new aerodynamic meshes, so we have
to split our OnewayAeroDriver scripts in implementation into two files, a my_fun3d_driver.py and a my_fun3d_analyzer.py.
The file my_fun3d_driver.py is called from a run.pbs script and manages the optimization and AIMs; this file
also uses system calls to the file my_fun3d_analyzer.py which runs the FUN3D analysis for each mesh. There are two 
class methods OnewayAeroDriver.remote and OnewayAeroDriver.analysis which build the drivers for each of the two files.

NOTE : only aerodynamic functions can be evaluated from this driver. If you only need aerodynamic DVs, you should build
the driver with class method OnewayAeroDriver.analysis(). If you need shape derivatives through ESP/CAPS, you shoud
build the driver class using the class method OnewayAeroDriver.remote() and setup a separate script with a driver running the analysis.
More details on these two files are provided below. Do not construct a driver with analysis() class method and shape DVs,
it will error in FUN3D upon the 2nd analysis iteration.

my_fun3d_driver.py : main driver script which called from the run.pbs
    NOTE : similar to tests/fun3d_tests/test_fun3d_oneway_shape.py
    - Construct the FUNtoFEMmodel
    - Build the Fun3dModel and link with Fun3dAim + AflrAIM and mesh settings, then store in funtofem_model.flow = this
    - Construct bodies and scenarios
    - Register aerodynamic and shape DVs to the scenarios/bodies
    - Construct the SolverManager with comm, but leave flow and structural attributes empty
    - Construct the fun3d oneway driver with class method OnewayAeroDriver.remote to manage system calls to the other script.
    - Build the optimization manager / run the driver

my_fun3d_analyzer.py : fun3d analysis script, which is called indirectly from my_fun3d_driver.py
    NOTE : similar to tests/fun3d_tests/run_fun3d_analysis.py
    - Construct the FUNtoFEMmodel
    - Construct the bodies and scenarios
    - Register aerodynamic DVs to the scenarios/bodies (no shape variables added and no AIMs here)
    - Construct the Fun3d14Interface
    - Construct the solvers (SolverManager), and set solvers.flow = my_fun3d_interface
    - Construct the a fun3d oneway driver with class method OnewayAeroDriver.analysis
    - Run solve_forward() and solve_adjoint() on the Fun3dOnewayAnalyzer

For an example implementation see tests/fun3d_tests/ folder with test_fun3d_oneway_aero.py for just aero DVs
and (test_fun3d_oneway_driver.py => run_fun3d_analysis.py) pair of files for the shape DVs using the Remote and system calls.
"""

import os, numpy as np
from funtofem.driver import TransferSettings
from funtofem.optimization.optimization_manager import OptimizationManager
from funtofem.interface.utils import Remote

import importlib.util

# Imports for each of the available flow/aero solvers
# ---------------------------------------------------
# 1) FUN3D
fun3d_loader = importlib.util.find_spec("fun3d")
if fun3d_loader is not None:  # check whether we can import FUN3D
    from funtofem.interface import Fun3d14Interface, Fun3dModel

# 2) TBD
# -----------------------------------------------------


class OnewayAeroDriver:
    @classmethod
    def aero_morph(cls, solvers, model, transfer_settings=None, external_shape=False):
        """
        Build a OnewayAeroDriver with fun3dAim shape variables and FUN3D analysis
        all in one, using the FUN3D mesh morphing.
        """
        return cls(
            solvers,
            model,
            transfer_settings=transfer_settings,
            is_paired=False,
            external_shape=external_shape,
        )

    @classmethod
    def aero_remesh(cls, solvers, model, remote, external_shape=False):
        """
        Build a OnewayAeroDriver object for the my_fun3d_driver.py script:
            this object would be responsible for the fun3d, aflr AIMs and

        """
        return cls(
            solvers, model, remote=remote, is_paired=True, external_shape=external_shape
        )

    @classmethod
    def analysis(
        cls,
        solvers,
        model,
        transfer_settings=None,
        external_shape=False,
        is_paired=True,
    ):
        """
        Build an OnewayAeroDriver object for the my_fun3d_analyzer.py script:
            this object would be responsible for running the FUN3D
            analysis and writing an aero.sens file to the FUN3D directory.
        If you are using the analysis driver by itself (e.g. for FUN3D mesh morphing) then turn "is_paired" off.
        """
        return cls(
            solvers,
            model,
            transfer_settings=transfer_settings,
            is_paired=is_paired,
            external_shape=external_shape,
        )

    def __init__(
        self,
        solvers,
        model,
        transfer_settings=None,
        remote=None,
        is_paired=False,
        external_shape=False,
    ):
        """
        Build the FUN3D analysis driver for shape/no shape change. Able to run another FUN3D analysis remotely or
        do it internally depending on how the object is constructed. The user is recommended to use the class methods
        to build the driver most of the time not the main constructor.

        NOTE : for using Fun3dOneway driver, put the moving_body.input file on "rigid" mesh_movement (makes it faster)

        Parameters
        ----------
        solvers: :class:`~funtofem.interface.SolverManager'
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
        self.remote = remote
        self.is_paired = is_paired
        self.external_shape = external_shape

        # store the shape variables list
        self.shape_variables = [
            var for var in self.model.get_variables() if var.analysis_type == "shape"
        ]

        # make sure there is shape change otherwise they should just use Fun3dOnewayAnalyzer
        if not (self.change_shape) and self.is_remote:
            raise AssertionError(
                "Need shape variables to use the OnewayAeroDriver otherwise use the Fun3dOnewayAnalyzer which duals as the driver for no shape DVs."
            )

        if self.is_remote and self.model.flow is not None:
            if self.model.flow.mesh_morph:
                raise AssertionError(
                    "The mesh morphing does not require a remote driver! Make this driver regularly!"
                )

        # check for unsteady problems
        self._unsteady = any([not scenario.steady for scenario in model.scenarios])

        # check which aero solver we were given
        self.flow_aim = None
        self._flow_solver_type = None
        if model.flow is None:
            if fun3d_loader is not None:
                if isinstance(solvers.flow, Fun3d14Interface):
                    self._flow_solver_type = "fun3d"
            # TBD on new types
        else:  # check with shape change
            if fun3d_loader is not None:
                if isinstance(model.flow, Fun3dModel):
                    self._flow_solver_type = "fun3d"
                    self.flow_aim = model.flow.fun3d_aim
            # TBD on new types

        if not self.is_remote:
            if self.model.flow is not None:
                if not self.is_paired and not self.model.flow.mesh_morph:
                    raise AssertionError(
                        "The nominal version of the driver only works for Fun3d mesh morphing not remeshing."
                    )

            if self.change_shape and self.root_proc:
                print(
                    f"Warning!! You are trying to remesh without using remote system calls of FUN3D, this will likely cause a FUN3D bug."
                )

        self.transfer_settings = (
            transfer_settings if transfer_settings is not None else TransferSettings()
        )
        if self.is_paired:  # if not mesh morphing initialize here
            self._initialize_funtofem()

        # rare use case for no shape, not paired
        if not self.is_paired and not self.change_shape:
            self._initialize_funtofem()

        self._first_forward = True

        # shape optimization
        if self.change_shape:
            assert self.flow_aim is not None
            assert self.model.flow.is_setup
            if self.uses_fun3d:
                self._setup_grid_filepaths()
        else:
            for body in self.model.bodies:
                body.update_transfer()

    def _initialize_funtofem(self):
        # initialize variables with newly defined aero size
        comm = self.solvers.comm
        comm_manager = self.solvers.comm_manager
        for body in self.model.bodies:
            # transfer to fixed structural loads in case the user got only aero loads from the OnewayAeroDriver
            body.initialize_transfer(
                comm=comm,
                struct_comm=comm_manager.struct_comm,
                struct_root=comm_manager.struct_root,
                aero_comm=comm_manager.aero_comm,
                aero_root=comm_manager.aero_root,
                transfer_settings=self.transfer_settings,  # using minimal settings since we don't use the state variables here (almost a dummy obj)
            )
            for scenario in self.model.scenarios:
                body.initialize_variables(scenario)
                body.initialize_adjoint_variables(
                    scenario
                )  # for writing sens files even in forward case

    @property
    def uses_fun3d(self) -> bool:
        return self._flow_solver_type == "fun3d"

    def solve_forward(self):
        """
        Forward analysis for the given shape and functionals.
        Assumes shape variables have already been changed.
        """

        if self.change_shape:
            if self.flow_aim.mesh_morph and self.root_proc:
                self.flow_aim.set_design_sensitivity(False, include_file=False)

            # run the pre analysis to generate a new mesh
            self.model.flow.pre_analysis()

            if not (self.is_paired):
                if (
                    self._first_forward and self.uses_fun3d
                ):  # FUN3D mesh morphing initialize body nodes
                    assert not (self.solvers.flow.auto_coords)
                    self.solvers.flow._initialize_body_nodes(
                        self.model.scenarios[0], self.model.bodies
                    )
                    # initialize funtofem transfer data with new aero_nnodes size
                    self._initialize_funtofem()
                    self._first_forward = False

        # system call FUN3D forward analysis and design variable inputs file
        if self.is_remote:  # currently remote only for FUN3D solver
            # write the funtofem design input file
            self.model.write_design_variables_file(
                self.comm,
                filename=Remote.paths(self.comm, self.remote.main_dir).design_file,
                root=0,
            )

            # clear the output file
            if self.root_proc and os.path.exists(self.remote.output_file):
                os.remove(self.remote.output_file)

            # system call the analysis driver
            os.system(
                f"mpiexec_mpt -n {self.remote.nprocs} python {self.remote.analysis_file} 2>&1 > {self.remote.output_file}"
            )

        else:  # non-remote call of FUN3D forward analysis
            # read in the funtofem design input file
            if self.is_paired:
                self.model.read_design_variables_file(
                    self.comm,
                    filename=Remote.paths(self.comm, self.flow_dir).design_file,
                    root=0,
                )

            # run the FUN3D forward analysis with no shape change
            if self.steady:
                for scenario in self.model.scenarios:
                    self._solve_steady_forward(scenario, self.model.bodies)

            if self.unsteady:
                for scenario in self.model.scenarios:
                    self._solve_unsteady_forward(scenario, self.model.bodies)

        # Write sens file for remote to read. Analysis functions/derivatives are being written to a file
        # to be read by the relevant AIM(s) which is in the remote driver.
        write_sens_file = self.is_paired or self.change_shape
        if not self.is_remote and write_sens_file:
            if not self.is_paired:
                filepath = self.flow_aim.sens_file_path
            else:
                filepath = Remote.paths(self.comm, self.flow_dir).aero_sens_file

            # write the sensitivity file for the FUN3D AIM
            self.model.write_sensitivity_file(
                comm=self.comm,
                filename=filepath,
                discipline="aerodynamic",
                write_dvs=False,
            )

        # post analysis for FUN3D mesh morphing
        if self.change_shape:  # either remote or regular
            # src for movement of sens file or None if not moving it
            sens_file_src = None
            if self.uses_fun3d and self.is_paired:
                sens_file_src = self.remote.aero_sens_file

            # run the tacs aim postAnalysis to compute the chain rule product
            self.flow_aim.post_analysis(sens_file_src)

            # get the analysis function values
            if self.flow_aim.mesh_morph:
                self.flow_aim.unlink()
            else:
                self._get_remote_functions()

        return

    def solve_adjoint(self):
        """
        Solve the adjoint analysis for the given shape and functionals.
        Assumes the forward analysis for this shape has already been performed.
        """

        # run the adjoint aerodynamic analysis
        functions = self.model.get_functions()

        # Zero the derivative values stored in the function
        self._zero_derivatives()
        for func in functions:
            func.zero_derivatives()

        if self.change_shape:
            if self.flow_aim.mesh_morph:
                self.flow_aim.set_design_sensitivity(True, include_file=False)

            # run the pre analysis to generate a new mesh
            self.flow_aim.pre_analysis()

        if not (self.is_remote):
            if self.steady:
                for scenario in self.model.scenarios:
                    self._solve_steady_adjoint(scenario, self.model.bodies)

            if self.unsteady:
                for scenario in self.model.scenarios:
                    self._solve_unsteady_adjoint(scenario, self.model.bodies)

            # write sens file for remote to read or if shape change all in one
            if not self.is_paired:
                filepath = self.flow_aim.sens_file_path
            else:
                filepath = Remote.paths(self.comm, self.flow_dir).aero_sens_file

            # write the sensitivity file for the FUN3D AIM
            self.model.write_sensitivity_file(
                comm=self.comm,
                filename=filepath,
                discipline="aerodynamic",
                write_dvs=False,
            )

        # shape derivative section
        if self.change_shape:  # either remote or regular
            # src for movement of sens file or None if not moving it
            sens_file_src = self.remote.aero_sens_file if self.is_paired else None

            # run the tacs aim postAnalysis to compute the chain rule product
            self.flow_aim.post_analysis(sens_file_src)

            # store the shape variables in the function gradients
            for scenario in self.model.scenarios:
                self._get_shape_derivatives(scenario)
        return

    def _zero_derivatives(self):
        """zero all model derivatives"""
        for func in self.model.get_functions(all=True):
            for var in self.model.get_variables():
                func.derivatives[var] = 0.0
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
        if self.flow_aim is not None:
            return self.comm.rank == self.flow_aim.root
        else:
            return self.comm.rank == 0

    @property
    def flow_dir(self):
        if self.uses_fun3d:
            return self.solvers.flow.fun3d_dir
        # TBD on other solvers

    def _transfer_fixed_struct_disps(self):
        """
        Transfer fixed structural displacements over to the new aero mesh for shape change.
        """
        # TODO : set this up for shape change from struct to aero disps
        return

    def _solve_steady_forward(self, scenario, bodies):
        # set functions and variables
        self.solvers.flow.set_variables(scenario, bodies)
        self.solvers.flow.set_functions(scenario, bodies)

        # run the forward analysis via iterate
        self.solvers.flow.initialize(scenario, bodies)
        for step in range(1, scenario.steps + 1):
            self.solvers.flow.iterate(scenario, bodies, step=step)
        self.solvers.flow.post(scenario, bodies, coupled_residuals=False)

        # get functions to store the function values into the model
        self.solvers.flow.get_functions(scenario, bodies)
        return

    def _solve_unsteady_forward(self, scenario, bodies):
        # set functions and variables
        self.solvers.flow.set_variables(scenario, bodies)
        self.solvers.flow.set_functions(scenario, bodies)

        # run the forward analysis via iterate
        self.solvers.flow.initialize(scenario, bodies)
        for step in range(1, scenario.steps + 1):
            self.solvers.flow.iterate(scenario, bodies, step=step)
        self.solvers.flow.post(scenario, bodies, coupled_residuals=False)

        # get functions to store the function values into the model
        self.solvers.flow.get_functions(scenario, bodies)
        return

    def _solve_steady_adjoint(self, scenario, bodies):
        if scenario.adjoint_steps is None:
            steps = scenario.steps
        else:
            steps = scenario.adjoint_steps

        # set functions and variables
        self.solvers.flow.set_variables(scenario, bodies)
        self.solvers.flow.set_functions(scenario, bodies)

        # zero all coupled adjoint variables in the body
        for body in bodies:
            body.initialize_adjoint_variables(scenario)

        # initialize, run, and do post adjoint
        self.solvers.flow.initialize_adjoint(scenario, bodies)
        # one extra call to match step 0 call (see fully coupled driver)
        for step in range(1, steps + 2):
            self.solvers.flow.iterate_adjoint(scenario, bodies, step=step)
        self._extract_coordinate_derivatives(scenario, bodies, step=0)
        self.solvers.flow.post_adjoint(scenario, bodies, coupled_residuals=False)

        # call get function gradients to store the gradients w.r.t. aero DVs from FUN3D
        self.solvers.flow.get_function_gradients(scenario, bodies)
        return

    def _solve_unsteady_adjoint(self, scenario, bodies):
        # set functions and variables
        self.solvers.flow.set_variables(scenario, bodies)
        self.solvers.flow.set_functions(scenario, bodies)

        # zero all coupled adjoint variables in the body
        for body in bodies:
            body.initialize_adjoint_variables(scenario)

        # initialize, run, and do post adjoint
        self.solvers.flow.initialize_adjoint(scenario, bodies)
        # one extra step here to include step 0 calls (see fully coupled driver)
        for rstep in range(1, scenario.steps + 2):
            step = scenario.steps + 1 - rstep
            self.solvers.flow.iterate_adjoint(scenario, bodies, step=step)
            self._extract_coordinate_derivatives(scenario, bodies, step=step)
        self.solvers.flow.post_adjoint(scenario, bodies, coupled_residuals=False)

        # call get function gradients to store the gradients w.r.t. aero DVs from FUN3D
        self.solvers.flow.get_function_gradients(scenario, bodies)
        return

    @property
    def is_remote(self) -> bool:
        """whether we are calling FUN3D in a remote manner"""
        return self.remote is not None

    @property
    def change_shape(self) -> bool:
        return len(self.shape_variables) > 0 and not self.external_shape

    def _setup_grid_filepaths(self):
        """setup the filepaths for each fun3d grid file in scenarios"""
        if self.solvers.flow is not None:
            fun3d_dir = self.solvers.flow.fun3d_dir
        else:
            fun3d_dir = self.remote.main_dir
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
        self.flow_aim.grid_filepaths = grid_filepaths

        # also setup the mapbc files
        mapbc_filepaths = []
        for scenario in self.model.scenarios:
            filepath = os.path.join(
                fun3d_dir,
                scenario.name,
                "Flow",
                f"{scenario.fun3d_project_name}.mapbc",
            )
            mapbc_filepaths.append(filepath)
        # set the mapbc filepaths into the fun3d aim
        self.flow_aim.mapbc_filepaths = mapbc_filepaths
        return

    @property
    def manager(self, hot_start: bool = False):
        return OptimizationManager(self, hot_start=hot_start)

    def _extract_coordinate_derivatives(self, scenario, bodies, step):
        """extract the coordinate derivatives at a given time step"""
        self.solvers.flow.get_coordinate_derivatives(
            scenario, self.model.bodies, step=step
        )

        # add transfer scheme contributions
        if step > 0:
            for body in bodies:
                body.add_coordinate_derivative(scenario, step=0)

        return

    def _get_remote_functions(self):
        """
        read function values from fun3dAIM when operating in the remote version of the driver
        """
        functions = self.model.get_functions()
        nfunc = len(functions)
        remote_functions = None
        if self.flow_aim.root_proc:
            remote_functions = np.zeros((nfunc))
            direct_flow_aim = self.flow_aim.aim
            for ifunc, func in enumerate(functions):
                remote_functions[ifunc] = direct_flow_aim.dynout[func.full_name].value

        # broadcast the function values to other processors
        flow_aim_root = self.flow_aim.root
        remote_functions = self.comm.bcast(remote_functions, root=flow_aim_root)

        # update model function values in the remote version of the driver
        for ifunc, func in enumerate(functions):
            func.value = remote_functions[ifunc]
        return

    def _get_shape_derivatives(self, scenario):
        """
        get shape derivatives together from FUN3D aim
        and store the data in the funtofem model
        """
        gradients = None

        # read shape gradients from tacs aim on root proc
        flow_aim_root = self.flow_aim.root
        if self.flow_aim.root_proc:
            gradients = []
            direct_flow_aim = self.flow_aim.aim

            for ifunc, func in enumerate(scenario.functions):
                gradients.append([])
                for ivar, var in enumerate(self.shape_variables):
                    derivative = direct_flow_aim.dynout[func.full_name].deriv(var.name)
                    gradients[ifunc].append(derivative)

        # broadcast shape gradients to all other processors
        gradients = self.comm.bcast(gradients, root=flow_aim_root)

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
