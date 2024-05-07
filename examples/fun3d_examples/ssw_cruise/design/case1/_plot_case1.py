from funtofem import *
import matplotlib.pyplot as plt, niceplots

# 1 : plot 1_sizing_opt_local.py results
# ---------------------------------------------------------------------------
scenario_name = "cruise"
plotter = PlotManager.from_hist_file(
    "ssw-sizing1_design.txt",
    accepted_names=["cruise-ksfailure", "steady_flight", "togw"],
    plot_names=["ksfailure", "steady-flight", "togw"],
    ignore_other_names=True,
)

# MAKE EACH PLOT FOR A DIFFERENT MODE
Function.plot("togw").optimize(scale=0.2e-4, plot_name="TOGW").register_to(plotter)
Function.plot("ksfailure").optimize(scale=1.0, plot_name="ksfailure").register_to(
    plotter
)
plotter.add_constraint(value=1.0, name="ks-constr")
Function.plot("steady-flight").optimize(
    scale=1.0e-3, plot_name="steady-flight"
).register_to(plotter)
plotter.add_absolute_value("steady-flight")

# three color schemes from color scheme website https://coolors.co/palettes/popular/3%20colors
colors1 = ["#2b2d42", "#8d99ae", "#edf2f4"]
colors2 = ["#264653", "#2a9d8f", "#e9c46a"]
colors3 = ["#2e4057", "#048ba8", "#f18f01"]
colors4 = ["#c9cba3", "#ffe1a8", "#e26d5c"]
colors5 = ["#0b3954", "#bfd7ea", "#ff6663"]
colors6 = ["#0d3b66", "#faf0ca", "#f4d35e"]
colors7 = ["#000000", "#ff0000", "#ffe100"]
colors8 = ["#064789", "#427aa1", "#ebf2fa"]
colors9 = ["#0b132b", "#1c2541", "#3a506b"]
colors10 = ["#49beaa", "#456990", "#ef767a"]
colors11 = ["#1d2f6f", "#8390fa", "#fac748"]
six_colors = ["#264653", "#287271", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]

plt.figure("case1")
for ifunc, func in enumerate(plotter.functions):
    plt.plot(
        plotter.iterations,
        plotter.get_hist_array(func),
        linewidth=2,
        color=colors11[ifunc],
        label=func.plot_name,
    )
# plot the ksfailure constraint
grey_colors = plt.cm.Greys(np.linspace(1.0, 0.5, len(plotter.constr_dicts)))
for icon, constr_dict in enumerate(plotter.constr_dicts):
    name = constr_dict["name"]
    value = constr_dict["value"]
    plt.plot(
        plotter.iterations,
        np.array([value for _ in plotter.iterations]),
        linestyle="dashed",
        linewidth=2,
        color=grey_colors[icon],
        label=name,
    )
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("Function Values")

# read the SNOPT history file and plot it
hdl = open("ssw1-SNOPT.out", "r")
lines = hdl.readlines()
hdl.close()

feasibility = []
optimality = []
meritFunc = []
_start = False
_first_line = True
for line in lines:
    if _start:
        chunks = line.split(" ")
        nempty_chunks = [_ for _ in chunks if len(_) > 0]
        if len(nempty_chunks) < 5:  # skip blank lines
            continue
        else:
            try:
                float(nempty_chunks[0])
            except:
                continue
        _start_idx = 4 if _first_line else 5
        if _first_line:
            _first_line = False
        feas = nempty_chunks[_start_idx]
        if "(" in feas:
            feas = feas[1:-1]
        feasibility += [float(feas)]
        optim = nempty_chunks[_start_idx + 1]
        if "(" in optim:
            optim = optim[1:-1]
        optimality += [float(optim)]
        merit = nempty_chunks[_start_idx + 2]
        meritFunc += [float(merit)]
        # print(nempty_chunks)
    if "Itns" in line:
        _start = True
    if "SNOPTC EXIT" in line:
        break

iterations = [_ for _ in range(len(feasibility))]

plt.style.use(niceplots.get_style())
plt.figure("case1-pyoptsparse")
my_colors = colors11
plt.margins(x=0.05, y=0.05)
plt.plot(iterations, meritFunc, "o-", color=my_colors[0], label="meritFunc")
plt.plot(iterations, optimality, "o-", color=my_colors[1], label="optimality")
plt.plot(iterations, feasibility, "o-", color=my_colors[2], label="feasibility")

plt.legend()
plt.yscale("log")
plt.xlabel("Major Iterations")
plt.ylabel("Optimizer Metric")
plt.savefig("case1-SNOPT.png", dpi=400)
