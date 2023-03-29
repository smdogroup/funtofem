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

# from .funtofem_nlbgs_driver import FUNtoFEMnlbgs
# from pyfuntofem.interface.solver_manager import SolverManager
from pyfuntofem.optimization.optimization_manager import OptimizationManager

import importlib.util

fun3d_loader = importlib.util.find_spec("fun3d")
if fun3d_loader is not None:  # check whether we can import FUN3D
    from pyfuntofem.interface import Fun3dInterface

# from mpi4py import MPI
# import numpy as np


class Fun3dOnewayDriver:
    def __init__(
        self,
        solvers,
        model,
    ):
        """
        build the FUN3D analysis driver for shape/no shape change, assumes you have already primed the disps (see class method to assist with that)

        Parameters
        ----------
        solvers: :class:`~interface.solver_manager.SolverManager`
            The various disciplinary solvers.
        model: :class:`~funtofem_model.FUNtoFEMmodel`
            The model containing the design data.
        """
        self.solvers = solvers
        self.comm = solvers.comm
        self.model = model

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

        # assertion check for unsteady
        # TODO : unsteady not available yet, can add this feature
        assert not self.unsteady

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
    def nominal(cls, solvers, model):
        """
        startup the fun3d oneway driver with no displacement just pure solvers with Fun3dInterface in it and model
        """
        return cls(solvers, model)

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
        return cls(funtofem_driver.solvers, funtofem_driver.model)

    @classmethod
    def prime_disps_shape(
        cls, flow_solver, tacs_aim, transfer_settings, nprocs, bdf_file=None
    ):
        """
        Used to prime aero loads for optimization over tacs analysis with shape change and tacs aim

        Parameters
        ----------
        # TODO : include fun3d_aim and/or aflrAIM/pointwiseAIM here for shape change
        """

        return None  # not setup yet

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

    def solve_forward(self):
        """
        forward analysis for the given shape and functionals
        assumes shape variables have already been changed
        """

        # TODO : add an aero shape change section here

        # run the FUN3D forward analysis with no shape change
        for scenario in self.model.scenarios:
            # set functions and variables
            self.fun3d_interface.set_variables(scenario, self.model.bodies)
            self.fun3d_interface.set_functions(scenario, self.model.bodies)

            # run the forward analysis via iterate
            self.fun3d_interface.initialize(scenario, self.model.bodies)
            self.fun3d_interface.iterate(scenario, self.model.bodies, step=0)
            self.fun3d_interface.post(scenario, self.model.bodies)

            # get functions to store the function values into the model
            self.fun3d_interface.get_functions(scenario, self.model.bodies)

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

        for scenario in self.model.scenarios:
            # set functions and variables
            self.fun3d_interface.set_variables(scenario, self.model.bodies)
            self.fun3d_interface.set_functions(scenario, self.model.bodies)

            # zero all coupled adjoint variables in the body
            for body in self.model.bodies:
                body.initialize_adjoint_variables(scenario)

            # initialize, run, and do post adjoint
            self.fun3d_interface.initialize_adjoint(scenario, self.model.bodies)
            self.fun3d_interface.iterate_adjoint(scenario, self.model.bodies, step=0)
            self.fun3d_interface.post_adjoint(scenario, self.model.bodies)

            # transfer loads adjoint since fa -> fs has shape dependency
            if self.change_shape:
                for body in self.model.bodies:
                    body.transfer_loads_adjoint(scenario)

            # call get function gradients to store the gradients w.r.t. aero DVs from FUN3D
            self.fun3d_interface.get_function_gradients(scenario, self.model.bodies)

        # TODO : set this up for shape change
        return

    def _get_shape_derivatives(self, scenario):
        """
        get shape derivatives together from FUN3D aim
        and store the data in the funtofem model
        """
        # TODO : set this up for shape derivatives through fun3d aim
        return
