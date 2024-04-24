__all__ = ["PlotManager"]

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import niceplots


class PlotManager:
    @classmethod
    def from_hist_file(
        cls, hist_file, accepted_names=None, plot_names=None, ignore_other_names=False
    ):
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
            if "Design #" in line:
                hist_dict = {}
            if "Design #" in line or "Functions" in line:
                after_lbrace = line.split("{")[1]
                before_rbrace = after_lbrace.split("}")[0]
                chunks = before_rbrace.split(",")
                pre_colons = []
                post_colons = []
                for chunk in chunks:
                    if len(chunk.split(":")) == 2:
                        pre_colons += [chunk.split(":")[0]]
                        post_colons += [chunk.split(":")[1]]
                    elif len(chunk.split(":")) == 3:
                        temp = chunk.split(":")[:2]
                        temp1 = temp[0].strip()
                        temp2 = temp[1].split("'")[0].strip()
                        pre_colons += [temp1 + ":" + temp2]
                        post_colons += [chunk.split(":")[2]]
                names = [_.split("'")[1] for _ in pre_colons]
                values = [float(_.strip()) for _ in post_colons]

                for i, name in enumerate(names):
                    if (
                        accepted_names is not None
                        and not (name in accepted_names)
                        and ignore_other_names
                    ):
                        continue
                    if use_name_map:
                        plot_name = name_map[name]
                    else:
                        plot_name = name

                    value = values[i]
                    hist_dict[plot_name] = value
            if "Functions" in line:
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
        self.absolute_value = [] # absolute value functions

    def include(self, func):
        self.functions += [func]

    def valid_function(self, func):
        return func.name in self.hist_dict_list[0]

    def get_hist_array(self, func):
        array = np.array([hist_dict[func.name] for hist_dict in self.hist_dict_list])
        if func.name in self.absolute_value or func.plot_name in self.absolute_value:
            array = np.abs(array)
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

    def add_absolute_value(self, name):
        self.absolute_value = [name]

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

    def __call__(
        self,
        plot_name=None,
        niceplots_style="doumont-light",
        show_scales=False,
        legend_frac: float = 0.8,
        yaxis_name="func values",
        color_offset: int = 0,
        yscale_log=False,
        xmargin: int = 0.02,
        ymargin: int = 0.02,
    ):
        if plot_name is None:
            plot_name = "f2f-history"

        with plt.style.context(niceplots.get_style(niceplots_style)):
            fig, ax = plt.subplots()
            plt.margins(x=xmargin, y=ymargin)
            colors = niceplots.get_colors_list(niceplots_style)
            for ifunc, func in enumerate(self.functions):
                if self.valid_function(func):
                    if show_scales:
                        omag = int(np.floor(np.log10(func.scale)))
                        label = func.plot_name + self.get_r_string(omag)
                    else:
                        label = func.plot_name
                    ax.plot(
                        self.iterations,
                        self.get_hist_array(func),
                        linewidth=2,
                        color=colors[ifunc + color_offset],
                        label=label,
                    )
                else:
                    print(
                        f"Warning function {func.name} is not in function history list.."
                    )
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
            ax.set_ylabel(yaxis_name)
            if yscale_log:
                ax.set_yscale("log")
            niceplots.adjust_spines(ax)
            niceplots.save_figs(
                fig,
                plot_name,
                ["png", "svg"],
                format_kwargs={"png": {"dpi": 400}},
                bbox_inches="tight",
            )
        return
