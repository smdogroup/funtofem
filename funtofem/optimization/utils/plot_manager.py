__all__ = ["PlotManager"]

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import niceplots


class PlotManager:
    @classmethod
    def from_hist_file(cls, hist_file, accepted_names=None, plot_names=None):
        """
        hist_file : a path to a design_hist.txt file
        get the history function dicts from the history file
        """
        hdl = open(hist_file, "r")
        lines = hdl.readlines()
        hdl.close()

        # make a map of new names if both accepted and plot names are given
        use_name_map = accepted_names and plot_names  # is not None is implied
        if use_name_map:
            name_map = {}
            for iname, accepted_name in enumerate(accepted_names):
                name_map[accepted_name] = plot_names[iname]

        hist_dict_list = []
        for line in lines:
            if "Functions" in line:
                after_lbrace = line.split("{")[1]
                before_rbrace = after_lbrace.split("}")[0]
                chunks = before_rbrace.split(",")
                pre_colons = [chunk.split(":")[0] for chunk in chunks]
                names = [_.split("'")[1] for _ in pre_colons]
                post_colons = [chunk.split(":")[1] for chunk in chunks]
                values = [float(_.strip()) for _ in post_colons]

                hist_dict = {}
                for i, name in enumerate(names):
                    if accepted_names is not None and not (name in accepted_names):
                        continue
                    if use_name_map:
                        plot_name = name_map[name]
                    else:
                        plot_name = name

                    value = values[i]
                    hist_dict[plot_name] = value
                hist_dict_list += [hist_dict]
        print(f"sample hist dict = {hist_dict_list[0]}")
        return cls(hist_dict_list=hist_dict_list)

    def __init__(self, hist_dict_list):
        """
        history array
        """
        self.hist_dict_list = hist_dict_list
        self.functions = []  # funtofem functions objects
        self.constr_dicts = []

    def include(self, func):
        self.functions += [func]

    def valid_function(self, func):
        return func.name in self.hist_dict_list[0]

    def get_hist_array(self, func):
        array = np.array([hist_dict[func.name] for hist_dict in self.hist_dict_list])
        return array * func.scale

    def get_constr_array(self, func):
        final_value = self.hist_dict_list[-1][func.name]
        constr_value = None
        if func.lower is not None and func.upper is not None:
            print(f"mode 1")
            lower_dist = abs(final_value - func.lower)
            upper_dist = abs(final_value - func.upper)
            if lower_dist < upper_dist:
                constr_value = func.lower * func.scale
            else:
                constr_value = func.upper * func.scale
        elif func.lower is not None:
            constr_value = func.lower * func.scale
        elif func.upper is not None:
            constr_value = func.upper * func.scale
        assert constr_value is not None
        return np.array([constr_value for _ in self.iterations])

    def add_constraint(self, name: str, value: float = 0.0):
        self.constr_dicts += [
            {
                "name": name,
                "value": value,
            }
        ]

    def get_r_string(self, omag):
        assert isinstance(omag, int)
        if omag == 0:
            return ""
        elif omag == 1:
            return r"*$10^1$"
        elif omag == 2:
            return r"*$10^2$"
        elif omag == 3:
            return r"*$10^3$"
        elif omag == 4:
            return r"*$10^4$"
        elif omag == 5:
            return r"*$10^5$"
        elif omag == 6:
            return r"*$10^6$"
        elif omag == -1:
            return r"*$10^{-1}$"
        elif omag == -2:
            return r"*$10^{-2}$"
        elif omag == -3:
            return r"*$10^{-3}$"
        elif omag == -4:
            return r"*$10^{-4}$"
        elif omag == -5:
            return r"*$10^{-5}$"
        elif omag == -6:
            return r"*$10^{-6}$"

    @property
    def iterations(self) -> list:
        return [_ for _ in range(len(self.hist_dict_list))]

    def __call__(self, plot_name=None, show_scales=False, legend_frac: float = 0.8):
        if plot_name is None:
            plot_name = "f2f-history"
        plt.style.use(niceplots.get_style())
        fig, ax = plt.subplots()
        for ifunc, func in enumerate(self.functions):
            if self.valid_function(func):
                if show_scales:
                    omag = int(np.floor(np.log(func.scale) / np.log(10)))
                    label = func.plot_name + self.get_r_string(omag)
                else:
                    label = func.plot_name
                ax.plot(
                    self.iterations,
                    self.get_hist_array(func),
                    linewidth=2,
                    label=label,
                )
                # if not (func._objective):  # plot constraint boundaries
                #     ax.plot(
                #         self.iterations,
                #         self.get_constr_array(func),
                #         "--",
                #         linewidth=2,
                #     )
                #     # color=colors[ifunc],
            else:
                print(f"Warning function {func.name} is not in function history list..")
        grey_colors = plt.cm.Greys(np.linspace(1.0, 0.5, len(self.constr_dicts)))
        for icon, constr_dict in enumerate(self.constr_dicts):
            name = constr_dict["name"]
            value = constr_dict["value"]
            ax.plot(
                self.iterations,
                np.array([value for _ in self.iterations]),
                linestyle="dotted",
                linewidth=2,
                color=grey_colors[icon],
                label=name,
            )
        # put axis on rhs of plot
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * legend_frac, box.height])
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        # set x-axis to integers only (since it represents iterations)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("iterations")
        ax.set_ylabel("func values")
        niceplots.adjust_spines(ax)
        niceplots.save_figs(
            fig,
            plot_name,
            ["png", "svg"],
            format_kwargs={"png": {"dpi": 400}},
            bbox_inches="tight",
        )
        return
