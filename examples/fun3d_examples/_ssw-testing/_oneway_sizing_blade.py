"""
Sean P. Engelstad, Georgia Tech 2023

Local machine optimization for the panel thicknesses using nribs-1 OML panels and nribs-1 LE panels
"""

import time
from pyoptsparse import SNOPT, Optimization
import numpy as np
import argparse

# script inputs
hot_start = False
store_history = True

# import openmdao.api as om
from funtofem import *
from mpi4py import MPI
from tacs import caps2tacs
import os

parent_parser = argparse.ArgumentParser(add_help=False)
parent_parser.add_argument("--case", type=int, default=1)
# case 1 - all vars
# case 2 - all vars except spar pitch
# case 3 - just panel thickness
args = parent_parser.parse_args()

from _blade_callback import *

assert args.case in [1, 2, 3]
if args.case == 1:
    callback = blade_elemCallBack
elif args.case == 2:
    callback = blade_elemCallBack_no_spitch
elif args.case == 3:
    callback = blade_elemCallback_justPanelThick


comm = MPI.COMM_WORLD

base_dir = os.path.dirname(os.path.abspath(__file__))
csm_path = os.path.join(base_dir, "geometry", "gbm.csm")

# F2F MODEL and SHAPE MODELS
# ----------------------------------------

f2f_model = FUNtoFEMmodel("ssw-sizing")
tacs_model = caps2tacs.TacsModel.build(
    csm_file=csm_path,
    comm=comm,
    problem_name="capsStruct1",
    active_procs=[0],
    verbosity=1,
)
tacs_model.mesh_aim.set_mesh(  # need a refined-enough mesh for the derivative test to pass
    edge_pt_min=2,
    edge_pt_max=20,
    global_mesh_size=0.3,  # 0.3
    max_surf_offset=0.2,
    max_dihedral_angle=15,
).register_to(
    tacs_model
)
f2f_model.structural = tacs_model

tacs_aim = tacs_model.tacs_aim
tacs_aim.set_config_parameter("view:flow", 0)
tacs_aim.set_config_parameter("view:struct", 1)

# add tacs constraints in
caps2tacs.PinConstraint("root").register_to(tacs_model)

# BODIES AND STRUCT DVs
# -------------------------------------------------

wing = Body.aeroelastic("wing", boundary=3)
# aerothermoelastic

# setup the material and shell properties
nribs = int(tacs_model.get_config_parameter("nribs"))
nspars = int(tacs_model.get_config_parameter("nspars"))
nOML = nribs - 1
null_material = caps2tacs.Orthotropic.null().register_to(tacs_model)

# create the design variables by components now
# since this mirrors the way TACS creates design variables
component_groups = [f"rib{irib}" for irib in range(1, nribs + 1)]
for prefix in ["spLE", "spTE", "uOML", "lOML"]:
    component_groups += [f"{prefix}{iOML}" for iOML in range(1, nOML + 1)]
component_groups = sorted(component_groups)

for icomp, comp in enumerate(component_groups):
    caps2tacs.CompositeProperty.null(comp, null_material).register_to(tacs_model)

    # NOTE : need to make the struct DVs in TACS in the same order as the blade callback
    # which is done by components and then a local order

    # panel length variable
    if "rib" in comp:
        panel_length = 0.38
    elif "sp" in comp:
        panel_length = 0.36
    elif "OML" in comp:
        panel_length = 0.65
    Variable.structural(f"{comp}-length", value=panel_length).set_bounds(
        lower=0.0, scale=1.0
    ).register_to(wing)

    # stiffener pitch variable
    if args.case == 1:
        Variable.structural(f"{comp}-spitch", value=0.20).set_bounds(
            lower=0.05, upper=0.5, scale=1.0
        ).register_to(wing)

    # panel thickness variable, shortened DV name for ESP/CAPS, nastran requirement here
    panel_thickness = 0.02
    Variable.structural(f"{comp}-T", value=panel_thickness).set_bounds(
        lower=0.002, upper=0.1, scale=100.0
    ).register_to(wing)

    if args.case in [1, 2]:
        # stiffener height
        Variable.structural(f"{comp}-sheight", value=0.05).set_bounds(
            lower=0.002, upper=0.1, scale=10.0
        ).register_to(wing)

        # stiffener thickness
        Variable.structural(f"{comp}-sthick", value=0.02).set_bounds(
            lower=0.002, upper=0.1, scale=100.0
        ).register_to(wing)

caps2tacs.PinConstraint("root", dof_constraint=246).register_to(tacs_model)
caps2tacs.PinConstraint("sob", dof_constraint=13).register_to(tacs_model)

# caps2tacs.PinConstraint("root", dof_constraint=123).register_to(tacs_model)

# register the wing body to the model
wing.register_to(f2f_model)

# INITIAL STRUCTURE MESH, SINCE NO STRUCT SHAPE VARS
# --------------------------------------------------

tacs_aim.setup_aim()
tacs_aim.pre_analysis()

# SCENARIOS
# ----------------------------------------------------

# make a funtofem scenario
cruise = Scenario.steady("climb_turb", steps=2)  # 2000
# increase ksfailure scale to make it stricter on infeasibility for that.
Function.ksfailure(ks_weight=10.0, safety_factor=1.5).optimize(
    scale=1.0, upper=1.0, objective=False, plot=True, plot_name="ks-cruise"
).register_to(cruise)
# does better with mass a lower value
Function.mass().optimize(
    scale=1.0e-3, objective=True, plot=True, plot_name="mass"
).register_to(cruise)
cruise.register_to(f2f_model)

# COMPOSITE FUNCTIONS
# -------------------------------------------------------

# skin thickness adjacency constraints
variables = f2f_model.get_variables()
adjacency_scale = 10.0
thick_adj = 2.5e-3

comp_groups = ["spLE", "spTE", "uOML", "lOML"]
comp_nums = [nOML for i in range(len(comp_groups))]
adj_types = ["T"]
if args.case in [1, 2]:
    adj_types += ["sthick", "sheight"]
adj_values = [thick_adj, thick_adj, 10e-3]
for igroup, comp_group in enumerate(comp_groups):
    comp_num = comp_nums[igroup]
    for icomp in range(1, comp_num):
        # no constraints across sob (higher stress there)
        # if icomp in [3,4]: continue
        for iadj, adj_type in enumerate(adj_types):
            adj_value = adj_values[iadj]
            name = f"{comp_group}{icomp}-{adj_type}"
            # print(f"name = {name}", flush=True)
            left_var = f2f_model.get_variables(f"{comp_group}{icomp}-{adj_type}")
            right_var = f2f_model.get_variables(f"{comp_group}{icomp+1}-{adj_type}")
            # print(f"left var = {left_var}, right var = {right_var}")
            adj_constr = left_var - right_var
            adj_constr.set_name(f"{comp_group}{icomp}-adj_{adj_type}").optimize(
                lower=-adj_value, upper=adj_value, scale=10.0, objective=False
            ).register_to(f2f_model)

    for icomp in range(1, comp_num + 1):
        skin_var = f2f_model.get_variables(f"{comp_group}{icomp}-T")
        sthick_var = f2f_model.get_variables(f"{comp_group}{icomp}-sthick")
        sheight_var = f2f_model.get_variables(f"{comp_group}{icomp}-sheight")
        spitch_var = f2f_model.get_variables(f"{comp_group}{icomp}-spitch")

        # stiffener - skin thickness adjacency here
        if args.case in [1, 2]:
            adj_value = thick_adj
            adj_constr = skin_var - sthick_var
            adj_constr.set_name(f"{comp_group}{icomp}-skin_stiff_T").optimize(
                lower=-adj_value, upper=adj_value, scale=10.0, objective=False
            ).register_to(f2f_model)

            # max stiffener aspect ratio constraint 10 * thickness - height >= 0
            max_AR_constr = 10 * sthick_var - sheight_var
            max_AR_constr.set_name(f"{comp_group}{icomp}-maxsAR").optimize(
                lower=0.0, scale=10.0, objective=False
            ).register_to(f2f_model)

            # min stiffener aspect ratio constraint 2 * thickness - height <= 0
            min_AR_constr = 2 * sthick_var - sheight_var
            min_AR_constr.set_name(f"{comp_group}{icomp}-minsAR").optimize(
                upper=0.0, scale=10.0, objective=False
            ).register_to(f2f_model)

        if args.case == 1:
            # minimum stiffener spacing pitch > 2 * height
            min_spacing_constr = spitch_var - 2 * sheight_var
            min_spacing_constr.set_name(f"{comp_group}{icomp}-sspacing").optimize(
                lower=0.0, scale=1.0, objective=False
            ).register_to(f2f_model)

for icomp, comp in enumerate(component_groups):
    CompositeFunction.external(
        f"{comp}-{TacsSteadyInterface.PANEL_LENGTH_CONSTR}"
    ).optimize(lower=0.0, upper=0.0, scale=1.0, objective=False).register_to(f2f_model)


# DISCIPLINE INTERFACES AND DRIVERS
# -----------------------------------------------------
solvers = SolverManager(comm)
# solvers.flow = Fun3dInterface(comm, f2f_model, fun3d_dir="meshes")
solvers.structural = TacsSteadyInterface.create_from_bdf(
    model=f2f_model,
    comm=comm,
    nprocs=4,
    bdf_file=tacs_aim.root_dat_file,
    prefix=tacs_aim.root_analysis_dir,
    callback=callback,
)

# after tacs steady interface evaluates panel length constraints, require again that the panel length constraints
# are lower=upper=constr+var where # is the evaluated constraint value
# this is temporary fix for structural optimization (will not work with shape)
# this is because pyoptsparse requires linear constraints to have their constants defined prior to optimization
# alternative might be to set up CompositeFunctions which are not included in the linear constraints
# (and have their sensitivities written out to ESP/CAPS ?)
# better way is to make the panel length variables an intermediate analysis state (less work in FUNtoFEM)
# print panel length constraints (NOTE : this should probably be improved)

npanel_func = 0
for ifunc, func in enumerate(f2f_model.get_functions(all=True)):
    if TacsSteadyInterface.PANEL_LENGTH_CONSTR in func.name:
        npanel_func += 1
        for ivar, var in enumerate(f2f_model.get_variables()):
            if (
                TacsSteadyInterface.PANEL_LENGTH_CONSTR in var.name
                and func.name.split("-")[0] == var.name.split("-")[0]
            ):
                true_panel_length = var.value + func.value
                func.lower = -true_panel_length
                func.upper = -true_panel_length
                func.value = 0.0
                var.value = true_panel_length
                # print(f"func {func.name} : lower {func.lower} upper {func.upper}")
                break

# remove these panel length composite functions from the model
# ncomp = len(f2f_model.composite_functions)
# nkeep = ncomp - npanel_func
# f2f_model.composite_functions = f2f_model.composite_functions[:nkeep]

# exit()

# read in aero loads
aero_loads_file = os.path.join(os.getcwd(), "cfd", "loads", "uncoupled_turb_loads.txt")

transfer_settings = TransferSettings(npts=200)

# build the shape driver from the file
tacs_driver = OnewayStructDriver.prime_loads_from_file(
    filename=aero_loads_file,
    solvers=solvers,
    model=f2f_model,
    nprocs=4,
    transfer_settings=transfer_settings,
)

test_derivatives = False
if test_derivatives:  # test using the finite difference test
    # load the previous design
    # design_in_file = os.path.join(base_dir, "design", "sizing-oneway.txt")
    # f2f_model.read_design_variables_file(comm, design_in_file)

    start_time = time.time()

    # run the finite difference test
    max_rel_error = TestResult.derivative_test(
        "fun3d+tacs-ssw1",
        model=f2f_model,
        driver=tacs_driver,
        status_file="1-derivs.txt",
        complex_mode=False,
        epsilon=1e-4,
    )

    end_time = time.time()
    dt = end_time - start_time
    if comm.rank == 0:
        print(f"total time for ssw derivative test is {dt} seconds", flush=True)
        print(f"max rel error = {max_rel_error}", flush=True)

    # exit before optimization
    exit()

# PYOPTSPARSE OPTMIZATION
# -------------------------------------------------------------

# create an OptimizationManager object for the pyoptsparse optimization problem
# design_in_file = os.path.join(base_dir, "design", "sizing.txt")
design_out_file = os.path.join(base_dir, "design", "sizing-blade.txt")

design_folder = os.path.join(base_dir, "design")
if not os.path.exists(design_folder) and comm.rank == 0:
    os.mkdir(design_folder)
history_file = os.path.join(design_folder, "sizing.hst")
store_history_file = history_file if store_history else None
hot_start_file = history_file if hot_start else None

# reload previous design
# not needed since we are hot starting
# f2f_model.read_design_variables_file(comm, design_out_file)

manager = OptimizationManager(
    tacs_driver,
    design_out_file=design_out_file,
    hot_start=hot_start,
    debug=True,
    hot_start_file=hot_start_file,
    sparse=True,
)

# create the pyoptsparse optimization problem
opt_problem = Optimization("ssw-sizing", manager.eval_functions)

# add funtofem model variables to pyoptsparse
manager.register_to_problem(opt_problem)

# run an SNOPT optimization
snoptimizer = SNOPT(
    # options={
    #     "Verify level": 0,
    #     "Function precision": 1e-6,
    #     "Major Optimality tol": 1e-4,
    # }
)

sol = snoptimizer(
    opt_problem,
    sens=manager.eval_gradients,
    storeHistory=store_history_file,
    hotStart=hot_start_file,
)

# print final solution
sol_xdict = sol.xStar
print(f"Final solution = {sol_xdict}", flush=True)
