from funtofem import *

# 1 : plot 1_panel_thickness.py results
# ---------------------------------------------------------------------------
plotter = PlotManager.from_hist_file(
    "ssw-sizing1_design.txt",
    accepted_names=["cruise_inviscid-mass", "cruise_inviscid-ksfailure"],
    plot_names=["mass", "ks-stress"],
    ignore_other_names=True,
)

# MAKE EACH PLOT FOR A DIFFERENT MODE
functions = []
Function.plot("ks-stress").optimize(scale=1.0, plot_name="ks-stress").register_to(
    plotter
)
plotter.add_constraint(value=1.0, name="ks-constr")
Function.plot("mass").optimize(scale=0.01, plot_name="mass").register_to(plotter)

plotter(
    plot_name="fc-sizing.png",
    legend_frac=0.9,
    yaxis_name="function values",
    color_offset=0,
    niceplots_style="doumont-light",
    yscale_log=True,
)

# 2 : plot 2_aero_aoa.py results
# ---------------------------------------------------------------------------
plotter = PlotManager.from_hist_file(
    "ssw-aoa_design.txt",
    accepted_names=["LiftObj", "cruise_inviscid-ksfailure"],
    plot_names=["dL^2", "ks-stress"],
    ignore_other_names=True,
)

# MAKE EACH PLOT FOR A DIFFERENT MODE
functions = []
Function.plot("ks-stress").optimize(scale=1.0, plot_name="ks-stress").register_to(
    plotter
)
plotter.add_constraint(value=1.0, name="ks-constr")
Function.plot("dL^2").optimize(scale=0.01, plot_name="dL^2").register_to(plotter)

plotter(
    plot_name="fc-aoa-design.png",
    legend_frac=0.9,
    yaxis_name="function values",
    color_offset=0,
    niceplots_style="doumont-light",
    yscale_log=True,
)

# 3 : plot 3_geom_twist.py results
# ---------------------------------------------------------------------------
plotter = PlotManager.from_hist_file(
    "ssw-twist_design.txt",
    accepted_names=["LiftObj", "cruise_inviscid-ksfailure"],
    plot_names=["dL^2", "ks-stress"],
    ignore_other_names=True,
)

# MAKE EACH PLOT FOR A DIFFERENT MODE
functions = []
Function.plot("ks-stress").optimize(scale=1.0, plot_name="ks-stress").register_to(
    plotter
)
plotter.add_constraint(value=1.0, name="ks-constr")
Function.plot("dL^2").optimize(scale=0.01, plot_name="dL^2").register_to(plotter)

plotter(
    plot_name="fc-twist-design.png",
    legend_frac=0.9,
    yaxis_name="function values",
    color_offset=0,
    niceplots_style="doumont-light",
    yscale_log=True,
)

# show the final twist distribution
