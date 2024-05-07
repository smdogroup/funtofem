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
togw = Function.plot("togw").optimize(scale=1.0e-3).register_to(plotter)
ksfailure = Function.plot("ksfailure").optimize(scale=1.0).register_to(
    plotter
)
steady_flight = Function.plot("steady-flight").optimize(scale=1.0e-3).register_to(
    plotter
)
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
plt.style.use(niceplots.get_style())
fig, ax1 = plt.subplots(figsize=(8,6))
# my_colors = niceplots.get_colors_list() #colors11
my_colors = colors3 #colors3, colors5
grey_colors = plt.cm.Greys(np.linspace(1.0, 0.5, 2))
plt.margins(x=0.05, y=0.05)
ax1.set_xlabel('Iterations')
ax1.set_ylabel('TOGW (kN)', color=my_colors[0])
ax1.tick_params(axis='y', labelcolor=my_colors[0])
ax1.plot(plotter.iterations, plotter.get_hist_array(togw), "o-", color=my_colors[0], label="TOGW")

ax2 = ax1.twinx()

niter = len(plotter.iterations)
ax2.plot(plotter.iterations, np.ones((niter,)), color=grey_colors[0], linestyle="dashed")
ax2.plot(plotter.iterations, np.zeros((niter,)), color=grey_colors[1], linestyle="dashed")
ax2.plot(plotter.iterations, plotter.get_hist_array(ksfailure), "o-", color=my_colors[1], label="KSfailure")
ax2.plot(plotter.iterations, plotter.get_hist_array(steady_flight), "o-", color=my_colors[2], label="L=W")
ax2.set_ylabel("Constraint Values", color='k')
ax2.tick_params(axis='y', labelcolor='k')
ax2.set_yscale("log")
ax2.set_ylim(1e-6, 1e2)
plt.text(x=15, y=1.2, s="ks-constr", color=grey_colors[0])

# legend_frac = 0.8
# box = ax2.get_position()
# ax2.set_position([box.x0, box.y0, box.width * legend_frac, box.height])
# ax2.legend(loc="center left", bbox_to_anchor=(1, 0.5))

# plot the constraints
plt.margins(x=0.02, y=0.02)
plt.legend()
plt.savefig("case1-opt-history.png", dpi=400)
plt.close("case1")

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
        if len(nempty_chunks) < 5: # skip blank lines
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
        optim = nempty_chunks[_start_idx+1]
        if "(" in optim:
            optim = optim[1:-1]
        optimality += [float(optim)]
        merit = nempty_chunks[_start_idx+2]
        meritFunc += [float(merit)]
        #print(nempty_chunks)
    if "Itns" in line:
        _start = True
    if "SNOPTC EXIT" in line:
        break

iterations = [_ for _ in range(len(feasibility))]

plt.figure("case2")
plt.style.use(niceplots.get_style())
fig, ax1 = plt.subplots(figsize=(8,6))
my_colors = colors3
plt.margins(x=0.05, y=0.05)
ax1.set_xlabel('Major Iterations')
ax1.set_ylabel('Merit Function', color=my_colors[0])
ax1.plot(iterations, meritFunc, "o-", color=my_colors[0], label="meritFunc")
ax1.tick_params(axis='y', labelcolor=my_colors[0])

ax2 = ax1.twinx()

ax2.plot(iterations, optimality, "o-", color=my_colors[1], label="optimality")
ax2.plot(iterations, feasibility, "o-", color=my_colors[2], label="feasibility")
ax2.set_ylabel("Feasibility/Optimality", color=my_colors[1])
ax2.tick_params(axis='y', labelcolor=my_colors[1])
ax2.set_yscale("log")

plt.legend()
plt.savefig("case1-SNOPT.png", dpi=400)