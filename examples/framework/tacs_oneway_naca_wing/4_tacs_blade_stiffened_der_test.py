"""
Sean Engelstad, May 2023
GT SMDO Lab, Dr. Graeme Kennedy
OnewayStructDriver example
Source for callback: Alasdair Christison Gray
"""

from funtofem import *
from tacs import caps2tacs
import openmdao.api as om
from mpi4py import MPI

from _blade_callback import blade_elemCallBack

# --------------------------------------------------------------#
# Setup CAPS Problem and FUNtoFEM model - NOTE: not complete, needs stringer PR
# --------------------------------------------------------------#
comm = MPI.COMM_WORLD
f2f_model = FUNtoFEMmodel("tacs_wing")
wing = Body.aeroelastic(
    "wing"
)  # says aeroelastic but not coupled, may want to make new classmethods later...

# define the Tacs model
tacs_model = caps2tacs.TacsModel.build(csm_file="large_naca_wing.csm", comm=comm)
tacs_model.mesh_aim.set_mesh(  # need a refined-enough mesh for the derivative test to pass
    edge_pt_min=15,
    edge_pt_max=20,
    global_mesh_size=0.01,
    max_surf_offset=0.01,
    max_dihedral_angle=5,
).register_to(
    tacs_model
)
tacs_aim = tacs_model.tacs_aim

# setup the thickness design variables + automatic shell properties
# using Composite functions, this part has to go after all funtofem variables are defined...
nribs = int(tacs_model.get_config_parameter("nribs"))
nspars = int(tacs_model.get_config_parameter("nspars"))
nOML = int(tacs_aim.get_output_parameter("nOML"))

null_material = caps2tacs.Orthotropic.null().register_to(tacs_model)

# create the design variables by components now
# since this mirrors the way TACS creates design variables
component_groups = [f"rib{irib}" for irib in range(1, nribs + 1)]
component_groups += [f"spar{ispar}" for ispar in range(1, nspars + 1)]
component_groups += [f"OML{iOML}" for iOML in range(1, nOML + 1)]
component_groups = sorted(component_groups)

for icomp, comp in enumerate(component_groups):
    caps2tacs.CompositeProperty.null(comp, null_material).register_to(tacs_model)

    # NOTE : need to make the struct DVs in TACS in the same order as the blade callback
    # which is done by components and then a local order

    # panel length variable
    if "rib" in comp:
        panel_length = 0.38
    elif "spar" in comp:
        panel_length = 0.36
    elif "OML" in comp:
        panel_length = 0.65
    Variable.structural(f"{comp}-length", value=panel_length).set_bounds(
        lower=0.0, scale=1.0
    ).register_to(wing)

    # stiffener pitch variable
    Variable.structural(f"{comp}-spitch", value=0.2).set_bounds(
        lower=0.05, upper=0.5, scale=1.0
    ).register_to(wing)

    # panel thickness variable, shortened DV name for ESP/CAPS, nastran requirement here
    panel_thickness = 0.04 * (icomp + 1) / len(component_groups)
    Variable.structural(f"{comp}-T", value=panel_thickness).set_bounds(
        lower=0.01, upper=0.2, scale=100.0
    ).register_to(wing)

    # stiffener height
    Variable.structural(f"{comp}-sheight", value=0.05).set_bounds(
        lower=0.002, upper=0.1, scale=10.0
    ).register_to(wing)

    # stiffener thickness
    Variable.structural(f"{comp}-sthick", value=0.02).set_bounds(
        lower=0.002, upper=0.1, scale=100.0
    ).register_to(wing)

# add constraints and loads
caps2tacs.PinConstraint("root").register_to(tacs_model)
caps2tacs.GridForce("OML", direction=[0, 0, 1.0], magnitude=10).register_to(tacs_model)

# run the tacs model setup and register to the funtofem model
f2f_model.structural = tacs_model

# register the funtofem Body to the model
wing.register_to(f2f_model)

# make the scenario(s)
tacs_scenario = Scenario.steady("tacs", steps=100)
Function.ksfailure(ks_weight=10.0).optimize(
    scale=30.0, upper=0.267, objective=False, plot=True
).register_to(tacs_scenario)
Function.mass().optimize(scale=1.0e-2, objective=True, plot=True).register_to(
    tacs_scenario
)
tacs_scenario.register_to(f2f_model)

# make the composite functions for adjacency constraints
variables = f2f_model.get_variables()
adj_value = 2.5e-3
adjacency_scale = 10.0

comp_groups = ["spar", "OML"]
comp_nums = [nspars, nOML]
adj_types = ["T", "sthick", "sheight"]
for igroup, comp_group in enumerate(comp_groups):
    comp_num = comp_nums[igroup]
    for icomp in range(1, comp_num):
        for adj_type in adj_types:
            name = f"{comp_group}{icomp}-{adj_type}"
            # print(f"name = {name}", flush=True)
            left_var = f2f_model.get_variables(f"{comp_group}{icomp}-{adj_type}")
            right_var = f2f_model.get_variables(f"{comp_group}{icomp+1}-{adj_type}")
            # print(f"left var = {left_var}, right var = {right_var}")
            adjacency_rib_constr = left_var - right_var
            adjacency_rib_constr.set_name(f"{comp_group}adj{icomp}").optimize(
                lower=-adj_value, upper=adj_value, scale=1.0, objective=False
            ).register_to(f2f_model)

        # also add stiffener - panel adjacency here too

# add panel length composite functions
# doesn't work on this particular geometry as panels have more than one closed loop
# for icomp, comp in enumerate(component_groups):
#    CompositeFunction.external(f"{comp}-{TacsSteadyInterface.PANEL_LENGTH_CONSTR}").optimize(lower=0, upper=0, scale=1.0, objective=False).register_to(f2f_model)

# make the BDF and DAT file for TACS structural analysis
tacs_aim.setup_aim()
tacs_aim.pre_analysis()

structDV_names = [var.name for var in wing.variables["structural"]]
structDV_names = sorted(structDV_names)

# build the solver manager, no tacs interface since built for each new shape
# in the tacs driver
solvers = SolverManager(comm)
solvers.flow = NullAerodynamicSolver(comm=comm, model=f2f_model)
solvers.structural = TacsSteadyInterface.create_from_bdf(
    model=f2f_model,
    comm=comm,
    nprocs=1,
    bdf_file=tacs_aim.root_dat_file,
    prefix=tacs_aim.root_analysis_dir,
    callback=blade_elemCallBack,
    add_loads=True,
)
solvers.flow.copy_struct_mesh()  # this routine only works on single proc BTW
null_driver = NullDriver(solvers, model=f2f_model, transfer_settings=None)

# build the tacs oneway driver
tacs_driver = OnewayStructDriver.prime_loads(driver=null_driver)

# tacs_driver.solve_forward()
# tacs_driver.solve_adjoint()

# # get function values
# functions = f2f_model.get_functions(all=True)
# for ifunc,func in enumerate(functions):
#     print(f"func {func.full_name} = {func.value}")

max_rel_error = TestResult.finite_difference(
    "tacs-blade-structural",
    f2f_model,
    tacs_driver,
    "test_blade_tacs.txt",
)

print(f"max rel error = {max_rel_error}")
