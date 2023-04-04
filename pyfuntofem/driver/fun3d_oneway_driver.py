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
__all__ = ["Fun3dOnewayDriver"]

import os

# from .funtofem_nlbgs_driver import FUNtoFEMnlbgs
# from pyfuntofem.interface.solver_manager import SolverManager
from pyfuntofem.optimization.optimization_manager import OptimizationManager

import importlib.util

fun3d_loader = importlib.util.find_spec("fun3d")
if fun3d_loader is not None:  # check whether we can import FUN3D
    from pyfuntofem.interface import Fun3dInterface


class Fun3dOnewayDriver:
    def __init__(
        self,
        solvers,
        model,
        transfer_settings=None,
        nominal=True,
        external_shape=False,
    ):
        """
        build the FUN3D analysis driver for shape/no shape change, assumes you have already primed the disps (see class method to assist with that)

        Parameters
        ----------
        solvers: :class:`~interface.solver_manager.SolverManager`
            The various disciplinary solvers.
        model: :class:`~funtofem_model.FUNtoFEMmodel`
            The model containing the design data.
        transfer_settings: :class:`~driver.TransferSettings`
            funtofem transfer settings from aero to structural meshes
        nominal: bool
            if True, then there will be no applied displacements/temperatures at the surface, pure fun3d
        external_shape: bool
            whether ESP/CAPS pre and postAnalysis are performed inside/outside this driver
        """
        self.solvers = solvers
        self.comm = solvers.comm
        self.model = model
        self._nominal = nominal
        self.external_shape = external_shape

        # get the fun3d interface out of solvers
        assert isinstance(self.solvers.flow, Fun3dInterface)
        self.fun3d_interface = self.solvers.flow

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

        # get the fun3d aim for changing shape
        if model.flow is None:
            fun3d_aim = None
        else:
            fun3d_aim = model.flow.fun3d_aim
        self.fun3d_aim = fun3d_aim

        # initialize transfer schemes
        comm = solvers.comm
        comm_manager = solvers.comm_manager

        # initialize variables
        for body in self.model.bodies:
            # transfer to fixed structural loads in case the user got only aero loads from the Fun3dOnewayDriver
            body.initialize_transfer(
                comm=comm,
                struct_comm=comm_manager.struct_comm,
                struct_root=comm_manager.struct_root,
                aero_comm=comm_manager.aero_comm,
                aero_root=comm_manager.aero_root,
                transfer_settings=transfer_settings,
            )
            for scenario in model.scenarios:
                body.initialize_variables(scenario)

        # shape optimization
        if self.change_shape:
            assert fun3d_aim is not None or external_shape
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

    @property
    def fun3d_model(self):
        return self.model.flow

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
    def nominal(cls, solvers, model, external_shape=False):
        """
        startup the fun3d oneway driver with no displacement just pure solvers with Fun3dInterface in it and model
        """
        return cls(solvers, model, nominal=True, external_shape=external_shape)

    @classmethod
    def prime_disps(cls, funtofem_driver):
        """
        Used to prime aero disps for optimization over FUN3D analysis with no shape variables

        Parameters
        ----------
        funtofem_driver: :class:`~funtofem_nlbgs_driver.FUNtoFEMnlbgs`
            the coupled funtofem NLBGS driver
        """
        funtofem_driver.solve_forward()
        return cls(funtofem_driver.solvers, funtofem_driver.model, nominal=False)

    def _setup_grid_filepaths(self):
        """setup the filepaths for each fun3d grid file in scenarios"""
        fun3d_dir = self.fun3d_interface.fun3d_dir
        grid_filepaths = []
        for scenario in self.model.scenarios:
            filepath = os.path.join(
                fun3d_dir, scenario.name, "Flow", "funtofem_CAPS.lb8.ugrid"
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

    def _transfer_fixed_struct_disps(self):
        """
        transfer fixed struct disps over to the new aero mesh for shape change
        """
        # TODO : set this up for shape change from struct to aero disps
        return

    def _load_new_mesh(self):
        """load the new aerodynamic mesh into the funtofem body class"""
        self.fun3d_interface._initialize_body_nodes(
            self.model.scenarios[0], self.model.bodies
        )

        # initialize transfer scheme again or no, prob no...
        for body in self.model.bodies:
            for scenario in self.model.scenarios:
                body.initialize_variables(scenario)
            body.update_transfer()
        return

    def solve_forward(self):
        """
        forward analysis for the given shape and functionals
        assumes shape variables have already been changed
        """

        # TODO : add an aero shape change section here
        if self.change_shape and not self.external_shape:
            # run the pre analysis to generate a new mesh
            self.fun3d_model.apply_shape_variables(self.shape_variables)
            self.fun3d_aim.pre_analysis()

            # move grid files to each scenario location
            self.fun3d_aim._move_grid_files()

            self._load_new_mesh()

        # run the FUN3D forward analysis with no shape change
        if self.steady:
            for scenario in self.model.scenarios:
                self._solve_steady_forward(scenario, self.model.bodies)

        if self.unsteady:
            for scenario in self.model.scenarios:
                self._solve_unsteady_forward(scenario, self.model.bodies)

        return

    def _solve_steady_forward(self, scenario, bodies):
        # set functions and variables
        self.fun3d_interface.set_variables(scenario, bodies)
        self.fun3d_interface.set_functions(scenario, bodies)

        # run the forward analysis via iterate
        self.fun3d_interface.initialize(scenario, bodies)
        if self._nominal:
            for step in range(1, scenario.steps + 1):
                self.comm.Barrier()
                bcont = self.fun3d_interface.fun3d_flow.iterate()
        else:
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
        if self._nominal:
            for step in range(1, scenario.steps + 1):
                self.comm.Barrier()
                bcont = self.fun3d_interface.fun3d_flow.iterate(step)
        else:
            for step in range(1, scenario.steps + 1):
                self.fun3d_interface.iterate(scenario, bodies, step=step)
        self.fun3d_interface.post(scenario, bodies)

        # get functions to store the function values into the model
        self.fun3d_interface.get_functions(scenario, bodies)
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

        if self.steady:
            for scenario in self.model.scenarios:
                self._solve_steady_adjoint(scenario, self.model.bodies)

        if self.unsteady:
            for scenario in self.model.scenarios:
                self._solve_unsteady_adjoint(scenario, self.model.bodies)

        # OPTIONAL : transfer disps adjoint here if we transfer disps in forward

        # shape derivative section
        if self.change_shape and not self.external_shape:
            # write the sensitivity file for the FUN3D AIM
            self.model.write_sensitivity_file(
                comm=self.comm,
                filename=self.fun3d_aim.sens_file_path,
                discipline="aerodynamic",
            )

            # run the tacs aim postAnalysis to compute the chain rule product
            self.fun3d_aim.post_analysis()

            # store the shape variables in the function gradients
            for scenario in self.model.scenarios:
                self._get_shape_derivatives(scenario)
        return

    def _solve_steady_adjoint(self, scenario, bodies):
        # set functions and variables
        self.fun3d_interface.set_variables(scenario, bodies)
        self.fun3d_interface.set_functions(scenario, bodies)

        # zero all coupled adjoint variables in the body
        for body in bodies:
            body.initialize_adjoint_variables(scenario)

        # initialize, run, and do post adjoint
        self.fun3d_interface.initialize_adjoint(scenario, bodies)
        if self._nominal:
            for step in range(1, scenario.steps + 1):
                # self.comm.Barrier()
                self.fun3d_interface.fun3d_adjoint.iterate(step)
        else:
            for step in range(1, scenario.steps + 1):
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
        if self._nominal:
            for step in range(1, scenario.steps + 1):
                # self.comm.Barrier()
                rstep = scenario.steps + 1 - step
                self.fun3d_interface.iterate(rstep)
                self._extract_coordinate_derivatives(scenario, bodies, step=step)
        else:
            for rstep in range(scenario.steps + 1):
                step = scenario.steps + 1 - rstep
                self.fun3d_interface.iterate_adjoint(scenario, bodies, step=step)
                self._extract_coordinate_derivatives(scenario, bodies, step=step)
        self.fun3d_interface.post_adjoint(scenario, bodies)

        # transfer disps adjoint since fa -> fs has shape dependency
        # if self.change_shape:
        #     for body in bodies:
        #         body.transfer_disps_adjoint(scenario)

        # call get function gradients to store the gradients w.r.t. aero DVs from FUN3D
        self.fun3d_interface.get_function_gradients(scenario, bodies)
        return

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
