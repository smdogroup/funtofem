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

import os, numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import shutil


class OptimizationManager:
    """
    Manages the pyoptsparse opimization problems with funtofem drivers as well as oneway coupled tacs drivers
    Requires only a pre-built funtofem model and driver, coupled or oneway-coupled
    Performs a gatekeeper feature to prevent double-running the forward analysis
    """

    def __init__(
        self,
        driver,
        write_designs: bool = True,
        hot_start: bool = False,
        design_out_file=None,
        hot_start_file=None,
    ):
        """
        Constructs the optimization manager class using a funtofem model and driver
        Parameters
        --------------
        driver : any driver object in funtofem
            coupled or oneway coupled driver, must have solve_forward and solve_adjoint methods
        write_designs: bool
            whether to write the design variable history at each iteration to a file
        hot_start: bool
            whether to reset the design history files in hot start (note this does not actually control the hot start itself, just an accompanying parameter)
        design_out_file: path or str
            path to the output file for writing design variable histories
        hot_start_file: path or str
            path to the hot start file which we copy and paste into the checkpoints folder from each design. The user can move this back to the main directory to restart from a previous iteration in case the optimization fails at some point.
        """
        # main attributes of the manager
        self.comm = driver.comm
        self.model = driver.model
        self.driver = driver
        self.design_out_file = design_out_file
        self.hot_start_file = hot_start_file

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
            if not (os.path.exists(self._design_folder)) and self.comm.rank == 0:
                os.mkdir(self._design_folder)
            self._checkpoints_folder = os.path.join(self._design_folder, "checkpoints")
            if not (os.path.exists(self._checkpoints_folder)) and self.comm.rank == 0:
                os.mkdir(self._checkpoints_folder)

            # make the design file handle
            write_str = "a" if hot_start else "w"
            if self.comm.rank == 0:
                self._design_hdl = open(
                    os.path.join(self._design_folder, f"{self.model.name}_design.txt"),
                    write_str,
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

            # setup function history for optimization plots and the history file path
            self._func_history = {
                func.plot_name: []
                for func in self.model.get_functions(optim=True)
                if func._plot
            }
            self._history_file = os.path.join(
                self._design_folder, f"{self.model.name}_history.png"
            )

            # add all variables (for off-scenario variables to derivatives dict for each function) to analysis functions
            for func in self.model.get_functions():
                for var in self.model.get_variables():
                    func.derivatives[var] = 0.0

            # initialize funcs, sens in case of failure on first design iteration of hot start
            self._funcs = {
                func.full_name: 0.0 for func in self.model.get_functions(optim=True)
            }
            self._sens = {}
            for func in self.model.get_functions(optim=True):
                self._sens[func.full_name] = {}
                for var in self.model.get_variables():
                    self._sens[func.full_name][var.full_name] = 0.0

    def _gatekeeper(self, x_dict):
        """
        Gatekeeper function prevents double-running of the forward analysis during optimization
        Controls access to the complete forward and adjoint calls
        """

        # only if a new design run a complete analysis
        fail = False
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

            # check for nans in any of the function values values
            for func_key in self._funcs:
                c_sens = self._sens[func_key]
                if np.isnan(self._funcs[func_key]):
                    if self.comm.rank == 0:
                        print(
                            f"Warning: func {func_key} = {self._funcs[var_key]} and has a nan"
                        )
                    fail = True
                for var_key in c_sens:
                    if np.isnan(c_sens[var_key]):
                        if self.comm.rank == 0:
                            print(
                                f"Warning: d{func_key}/d{var_key} = {c_sens[var_key]} and has a nan"
                            )
                        fail = True
                if fail:
                    break

            # write the new function values
            if self.comm.rank == 0 and self.write_designs:
                self._design_hdl.write(f"Functions = {self._funcs}\n")
                self._design_hdl.flush()

            # only update design file if analysis didn't fail and give nans
            if self.design_out_file is not None and not (fail):
                self.model.write_design_variables_file(
                    self.comm, self.design_out_file, root=0
                )
            if not (fail):
                # also write the design to the checkpoints folder
                dvs_file = os.path.join(
                    self._checkpoints_folder, f"funtofem{self._iteration}.in"
                )
                self.model.write_design_variables_file(self.comm, dvs_file, root=0)
                func_file = os.path.join(
                    self._checkpoints_folder, f"funtofem{self._iteration}.out"
                )
                self.model.write_functions_file(
                    self.comm, func_file, full_precision=False, optim=True
                )
                # copy the hotstart file to the checkpoints folder
                src = self.hot_start_file
                dest = os.path.join(
                    self._checkpoints_folder, f"hot_start{self._iteration}.hst"
                )
                shutil.copy(src, dest)
                self._iteration += 1

            # update and plot the current optimization history
            if self.write_designs and not (fail):
                for func in self.model.get_functions(optim=True):
                    if not func._plot:
                        continue
                    self._func_history[func.plot_name] += [func.value.real]
                self._plot_history()
        return fail

    def _run_complete_analysis(self):
        """
        run a complete forward and adjoint analysis for the given FUNtoFEM driver
        """

        # update the model design variables
        for var in self.model.get_variables():
            for var_key in self._x_dict:
                if var.full_name == var_key:
                    # assumes here that only pyoptsparse single variables (no var groups are made)
                    var.value = float(self._x_dict[var_key])

        # run forward and adjoint analysis on the driver
        self.driver.solve_forward()
        self.driver.solve_adjoint()

        # evaluate composite functions if not evaluated yet
        self.model.evaluate_composite_functions()

        # store the functions and sensitivities from the complete analysis
        self._funcs = {}
        self._sens = {}
        # get only functions with optim=True, set with func.optimize() method (can method cascade it)
        for func in self.model.get_functions(optim=True):
            self._funcs[func.full_name] = func.value.real
            self._sens[func.full_name] = {}
            for var in self.model.get_variables():
                self._sens[func.full_name][var.full_name] = func.get_gradient_component(
                    var
                ).real
        return

    def register_to_problem(self, opt_problem):
        """
        add funtofem model variables and functions to a pyoptsparse optimization problem
        """
        for var in self.model.get_variables():
            opt_problem.addVar(
                var.full_name,
                lower=var.lower,
                upper=var.upper,
                value=var.value,
                scale=var.scale,
            )

        for func in self.model.get_functions(optim=True):
            if func._objective:
                opt_problem.addObj(func.full_name, scale=func.scale)
            else:
                opt_problem.addCon(
                    func.full_name, lower=func.lower, upper=func.upper, scale=func.scale
                )

        return

    def eval_functions(self, x_dict):
        """
        obtain the functions dictionary for pyoptsparse
        """
        try:
            self._gatekeeper(x_dict)
            fail = False
        except:
            fail = True
            print("warning: eval functions failure..")
        return self._funcs, fail

    def eval_gradients(self, x_dict, funcs):
        """
        obtain the sensitivity dictionary for pyoptsparse
        """
        try:
            self._gatekeeper(x_dict)
            fail = False
        except:
            fail = True
        return self._sens, fail

    def _plot_history(self):
        driver = self.driver
        model = self.model

        keys = list(self._func_history.keys())
        nkeys = len(keys)
        colors = plt.cm.jet(np.linspace(0, 1, nkeys))
        if driver.comm.rank == 0:
            func_keys = list(self._func_history.keys())
            num_iterations = len(self._func_history[func_keys[0]])
            iterations = [_ for _ in range(num_iterations)]
            plt.figure()

            ax = plt.subplot(111)
            ind = 0
            for func in model.get_functions(optim=True):
                if func.plot_name in func_keys:
                    yvec = np.array(self._func_history[func.plot_name])
                    if func._objective:
                        yvec *= func.scale
                    else:  # constraint
                        constr_bndry = 1.0
                        # take relative errors against constraint boundaries, lower upper
                        yfinal = yvec[-1]
                        err_lower = 1e5
                        err_upper = 1e5
                        if func.lower is not None:
                            # use abs error since could have div 0
                            err_lower = abs(yfinal - func.lower)
                        if func.upper is not None:
                            # use abs error since could have div 0
                            err_upper = abs(yfinal - func.upper)
                        if err_lower < err_upper:
                            constr_bndry = func.lower
                        else:
                            constr_bndry = func.upper
                        if constr_bndry == 0.0:
                            yvec = np.abs(yvec * func.scale)
                        else:
                            yvec = np.abs((yvec - constr_bndry) / constr_bndry)
                    # plot the function
                    ax.plot(
                        iterations,
                        yvec,
                        color=colors[ind],
                        linewidth=2,
                        label=func.plot_name,
                    )
                    ind += 1
            # put axis on rhs of plot
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
            # set x-axis to integers only (since it represents iterations)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.xlabel("iterations")
            plt.ylabel("func values")
            plt.yscale("log")
            plt.savefig(self._history_file, dpi=300)
            plt.close("all")
