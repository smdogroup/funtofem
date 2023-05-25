from __future__ import print_function

__all__ = ["FuntofemComponent"]

import warnings

warnings.filterwarnings("ignore")

import os, matplotlib.pyplot as plt, numpy as np
from openmdao.api import ExplicitComponent


class FuntofemComponent(ExplicitComponent):
    """
    OpenMDAO component for funtofem design optimization
    """

    def register_to_model(self, openmdao_model, subsystem_name):
        """
        register the funtofem variables and functions
        to the openmdao problem (called in your run script)
        see examples/naca_wing/1_tacs_sizing_opt.py
        """
        driver = self.options["driver"]
        model = driver.model

        # add design variables to the model
        for var in model.get_variables():
            openmdao_model.add_design_var(
                f"{subsystem_name}.{var.name}",
                lower=var.lower,
                upper=var.upper,
                scaler=var.scale,
            )

        # add objectives & constraints to the model
        for func in model.get_functions(optim=True):
            if func._objective:
                openmdao_model.add_objective(
                    f"{subsystem_name}.{func.name}", scaler=func.scale
                )
            else:
                openmdao_model.add_constraint(
                    f"{subsystem_name}.{func.name}",
                    lower=func.lower,
                    upper=func.upper,
                    scaler=func.scale,
                )

    def initialize(self):
        self.options.declare("driver", types=object)
        self.options.declare("track_history", types=bool, default=True)
        self.options.declare(
            "write_dir", default=None
        )  # where to write design and opt plot files

    def setup(self):
        # self.set_check_partial_options(wrt='*',directional=True)
        driver = self.options["driver"]
        track_history = self.options["track_history"]
        write_dir = self.options["write_dir"]
        model = driver.model

        # get vars and funcs, only funcs with optim=True
        # which are for optimization
        variables = model.get_variables()
        assert len(variables) > 0

        # add f2f variables to openmdao
        for var in variables:
            self.add_input(var.name, val=var.value)

        # add f2f functions to openmdao
        functions = model.get_functions(optim=True)
        for func in functions:
            self.add_output(func.name)

        # store the variable dictionary of values
        # to prevent repeat analyses
        self._x_dict = {var.name: var.value for var in model.get_variables()}
        comm = driver.comm
        if write_dir is None:
            write_dir = os.getcwd()
        self._write_path = write_dir
        self._plot_filename = f"f2f-{model.name}_optimization.png"

        self._first_analysis = True
        self._first_opt = True

        # store function optimization history
        if track_history:
            self._func_history = {func.name: [] for func in functions if func._plot}

            if comm.rank == 0:
                self._design_hdl = open(
                    os.path.join(self._write_path, "design_hist.txt"), "w"
                )

    def setup_partials(self):
        driver = self.options["driver"]
        model = driver.model

        # declare any partial derivatives for optimization functions
        for func in model.get_functions(optim=True):
            for var in model.get_variables():
                self.declare_partials(func.name, var.name)

    def update_design(self, inputs, analysis=True):
        driver = self.options["driver"]
        model = driver.model
        changed_design = False

        for var in model.get_variables():
            if var.value != float(inputs[var.name]):
                changed_design = True
                var.value = float(inputs[var.name])

        if analysis and self._first_analysis:
            self._first_analysis = False
            changed_design = True
        elif not (analysis) and self._first_opt:
            self._first_opt = False
            changed_design = True

        if changed_design:
            self._design_report()
        return changed_design

    def compute(self, inputs, outputs):
        driver = self.options["driver"]
        track_history = self.options["track_history"]
        model = driver.model
        new_design = self.update_design(inputs, analysis=True)

        if new_design:
            driver.solve_forward()
            model.evaluate_composite_functions(compute_grad=False)

        if track_history:
            self._update_history()

        for func in model.get_functions(optim=True):
            outputs[func.name] = func.value.real
        return

    def compute_partials(self, inputs, partials):
        driver = self.options["driver"]
        model = driver.model
        new_design = self.update_design(inputs, analysis=False)

        if new_design:
            driver.solve_adjoint()
            model.evaluate_composite_functions(compute_grad=True)

        for func in model.get_functions(optim=True):
            for var in model.get_variables():
                partials[func.name, var.name] = func.get_gradient_component(var).real
        return

    def cleanup(self):
        """close the design handle file and any other cleanup"""
        track_history = self.options["track_history"]
        if track_history:
            self._design_hdl.close()

    # helper methods for writing history, plotting history, etc.
    def _update_history(self):
        driver = self.options["driver"]
        model = driver.model
        for func in model.get_functions(optim=True):
            if func.name in self._func_history:
                self._func_history[func.name].append(func.value.real)

        if driver.comm.rank == 0:
            self._plot_history()
            self._function_report()

    def _function_report(self):
        driver = self.options["driver"]

        if driver.comm.rank == 0:
            self._design_hdl.write("Analysis result:\n")
            for func_name in self._func_history:
                self._design_hdl.write(
                    f"\tfunc {func_name} = {self._func_history[func_name][-1]}\n"
                )
            self._design_hdl.write("\n")
            self._design_hdl.flush()

    def _plot_history(self):
        driver = self.options["driver"]

        if driver.comm.rank == 0:
            func_keys = list(self._func_history.keys())
            num_iterations = len(self._func_history[func_keys[0]])
            iterations = [_ for _ in range(num_iterations)]
            plt.figure()
            for func_name in func_keys:
                yvec = self._func_history[func_name]
                yvec /= max(np.array(yvec))
                plt.plot(iterations, yvec, linewidth=2, label=func_name)
            plt.legend()
            plt.xlabel("iterations")
            plt.ylabel("func values")
            plt.yscale("log")
            plot_filepath = os.path.join(self._write_path, self._plot_filename)
            plt.savefig(plot_filepath)
            plt.close("all")

    def _design_report(self):
        driver = self.options["driver"]
        model = driver.model

        if driver.comm.rank == 0:
            variables = model.get_variables()
            self._design_hdl.write("New Design...\n")
            self._design_hdl.write(f"\tf2f vars = {[_.name for _ in variables]}\n")
            real_xarray = [var.value for var in variables]
            self._design_hdl.write(f"\tvalues = {real_xarray}\n")
            self._design_hdl.flush()
