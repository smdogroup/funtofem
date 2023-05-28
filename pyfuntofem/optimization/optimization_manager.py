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

__all__ = ["OptimizationManager"]

import os


class OptimizationManager:
    """
    Manages the pyoptsparse opimization problems with funtofem drivers as well as oneway coupled tacs drivers
    Requires only a pre-built funtofem model and driver, coupled or oneway-coupled
    Performs a gatekeeper feature to prevent double-running the forward analysis
    """

    def __init__(self, driver, write_designs: bool = True, hot_start: bool = False):
        """
        Constructs the optimization manager class using a funtofem model and driver
        Parameters
        --------------
        driver : any driver coupled or oneway coupled, must have solve_forward and solve_adjoint methods

        """
        # main attributes of the manager
        self.comm = driver.comm
        self.model = driver.model
        self.driver = driver

        # optimization meta data
        self._iteration = 0
        self._x_dict = None
        self._funcs = None
        self._sens = None

        # initialize optimization object
        # self._initialize_optimization()

        # create a design folder to manage
        self.write_designs = write_designs
        if write_designs:
            self._design_folder = os.path.join(os.getcwd(), "design")
            if not (os.path.exists(self._design_folder)):
                os.mkdir(self._design_folder)

            # make the design file handle
            write_str = "a" if hot_start else "w"
            if self.comm.rank == 0:
                self._design_hdl = open(
                    os.path.join(self._design_folder, "design.txt"), write_str
                )

            # write an inital header for the design file
            if self.comm.rank == 0:
                if hot_start:
                    self._design_hdl.write(
                        "\nRestarting optimization with a hot start...\n"
                    )
                else:
                    self._design_hdl.write(
                        f"Starting new pyoptsparse optimization for the {self.model.name} model...\n"
                    )
                self._design_hdl.flush()

    def _gatekeeper(self, x_dict):
        """
        Gatekeeper function prevents double-running of the forward analysis during optimization
        Controls access to the complete forward and adjoint calls
        """

        # only if a new design run a complete analysis
        if not (x_dict == self._x_dict):
            # write the new design dict
            if self.comm.rank == 0 and self.write_designs:
                regular_dict = {key: float(x_dict[key]) for key in x_dict}
                self._design_hdl.write(f"New design = {regular_dict}\n")
                self._design_hdl.flush()

            # change the design
            self._x_dict = x_dict

            # run a complete analysis - both forward and adjoint
            self._run_complete_analysis()

            # increment the iteration number
            self._iteration += 1

            # write the new function values
            if self.comm.rank == 0 and self.write_designs:
                self._design_hdl.write(f"Functions = {self._funcs}\n")
                self._design_hdl.flush()

        return

    def _run_complete_analysis(self):
        """
        run a complete forward and adjoint analysis for the given FUNtoFEM driver
        """

        # update the model design variables
        for var in self.model.get_variables():
            for var_key in self._x_dict:
                if var.name == var_key:
                    # assumes here that only pyoptsparse single variables (no var groups are made)
                    var.value = float(self._x_dict[var_key])

        # run forward and adjoint analysis on the driver
        self.driver.solve_forward()
        self.driver.solve_adjoint()

        # evaluate any composite functions once the main analysis functions are computed
        self.model.evaluate_composite_functions()

        # store the functions and sensitivities from the complete analysis
        self._funcs = {}
        self._sens = {}
        # get only functions with optim=True, set with func.optimize() method (can method cascade it)
        for func in self.model.get_functions(optim=True):
            self._funcs[func.name] = func.value.real
            self._sens[func.name] = {}
            for var in self.model.get_variables():
                self._sens[func.name][var.name] = func.get_gradient_component(var).real

        return

    def add_sparse_variables(self, opt_problem):
        """
        add funtofem model variables to a pyoptsparse optimization problem
        """
        for var in self.model.get_variables():
            opt_problem.addVar(
                var.name,
                lower=var.lower,
                upper=var.upper,
                value=var.value,
                scale=var.scale,
            )

        return

    def eval_functions(self, x_dict):
        """
        obtain the functions dictionary for pyoptsparse
        """
        fail = False
        self._gatekeeper(x_dict)

        return self._funcs, fail

    def eval_gradients(self, x_dict, funcs):
        """
        obtain the sensitivity dictionary for pyoptsparse
        """
        fail = False
        self._gatekeeper(x_dict)

        return self._sens, fail
