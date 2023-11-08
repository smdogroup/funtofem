"""
Sean Engelstad, May 2023
GT SMDO Lab, Dr. Graeme Kennedy
OnewayStructDriver example
"""

from funtofem import *
from tacs import caps2tacs
import openmdao.api as om
from mpi4py import MPI

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

aluminum = caps2tacs.Isotropic.aluminum()
aluminum_stringer = caps2tacs.Orthotropic.smeared_stringer(
    aluminum, area_ratio=0.5
).register_to(tacs_model)

# setup the thickness design variables + automatic shell properties
# using Composite functions, this part has to go after all funtofem variables are defined...
nribs = int(tacs_model.get_config_parameter("nribs"))
nspars = int(tacs_model.get_config_parameter("nspars"))
nOML = int(tacs_aim.get_output_parameter("nOML"))

init_thickness = 0.08
for irib in range(1, nribs + 1):
    name = f"rib{irib}"
    caps2tacs.CompositeProperty.one_ply(
        caps_group=name,
        material=aluminum_stringer,
        thickness=init_thickness,
        ply_angle=0,
    ).register_to(tacs_model)
    Variable.structural(name, value=init_thickness).set_bounds(
        lower=0.01, upper=0.2, scale=100.0
    ).register_to(wing)

for ispar in range(1, nspars + 1):
    name = f"spar{ispar}"
    caps2tacs.CompositeProperty.one_ply(
        caps_group=name,
        material=aluminum_stringer,
        thickness=init_thickness,
        ply_angle=0,
    ).register_to(tacs_model)
    Variable.structural(name, value=init_thickness).set_bounds(
        lower=0.01, upper=0.2, scale=100.0
    ).register_to(wing)

for iOML in range(1, nOML + 1):
    name = f"OML{iOML}"
    caps2tacs.CompositeProperty.one_ply(
        caps_group=name,
        material=aluminum_stringer,
        thickness=init_thickness,
        ply_angle=0,
    ).register_to(tacs_model)
    Variable.structural(name, value=init_thickness).set_bounds(
        lower=0.01, upper=0.2, scale=100.0
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
Function.mass().optimize(scale=1.0e-2, objective=True, plot=True).register_to(
    tacs_scenario
)
Function.ksfailure(ks_weight=10.0).optimize(
    scale=30.0, upper=0.267, objective=False, plot=True
).register_to(tacs_scenario)
tacs_scenario.register_to(f2f_model)

# make the composite functions for adjacency constraints
variables = f2f_model.get_variables()
adj_ratio = 2.0
adj_scale = 10.0
for irib in range(
    1, nribs
):  # not (1, nribs+1) bc we want to do one less since we're doing nribs-1 pairs
    left_rib = f2f_model.get_variables(names=f"rib{irib}")
    right_rib = f2f_model.get_variables(names=f"rib{irib+1}")
    # make a composite function for relative diff in rib thicknesses
    adj_rib_constr = (left_rib - right_rib) / left_rib
    adj_rib_constr.set_name(f"rib{irib}-{irib+1}").optimize(
        lower=-adj_ratio, upper=adj_ratio, scale=1.0, objective=False
    ).register_to(f2f_model)

for ispar in range(1, nspars):
    left_spar = f2f_model.get_variables(names=f"spar{ispar}")
    right_spar = f2f_model.get_variables(names=f"spar{ispar+1}")
    # make a composite function for relative diff in spar thicknesses
    adj_spar_constr = (left_spar - right_spar) / left_spar
    adj_spar_constr.set_name(f"spar{ispar}-{ispar+1}").optimize(
        lower=-adj_ratio, upper=adj_ratio, scale=1.0, objective=False
    ).register_to(f2f_model)

for iOML in range(1, nOML):
    left_OML = f2f_model.get_variables(names=f"OML{iOML}")
    right_OML = f2f_model.get_variables(names=f"OML{iOML+1}")
    # make a composite function for relative diff in OML thicknesses
    adj_OML_constr = (left_OML - right_OML) / left_OML
    adj_OML_constr.set_name(f"OML{iOML}-{iOML+1}").optimize(
        lower=-adj_ratio, upper=adj_ratio, scale=1.0, objective=False
    ).register_to(f2f_model)

# make the BDF and DAT file for TACS structural analysis
tacs_aim.setup_aim()
tacs_aim.pre_analysis()

# build the solver manager, no tacs interface since built for each new shape
# in the tacs driver
solvers = SolverManager(comm)
solvers.flow = NullAerodynamicSolver(comm=comm, model=f2f_model)
solvers.structural = TacsSteadyInterface.create_from_bdf(
    model=f2f_model,
    comm=comm,
    nprocs=1,
    bdf_file=tacs_aim.dat_file_path,
    prefix=tacs_aim.analysis_dir,
)
solvers.flow.copy_struct_mesh()
null_driver = NullDriver(solvers, model=f2f_model, transfer_settings=None)

# build the tacs oneway driver
tacs_driver = OnewayStructDriver.prime_loads(driver=null_driver)

# --------------------------------------------------------------------------#
# Setup OpenMDAO Problem and Perform Sizing Optimization on the Wing
# --------------------------------------------------------------------------#

# setup the OpenMDAO Problem object
prob = om.Problem()

# Create the OpenMDAO component using the built-in Funtofem component
design_out_file = "design-test.txt"
f2f_subsystem = FuntofemComponent(
    driver=tacs_driver, write_dir=tacs_aim.analysis_dir, design_out_file=design_out_file
)
prob.model.add_subsystem("f2fSystem", f2f_subsystem)
f2f_subsystem.register_to_model(prob.model, "f2fSystem")

# setup the optimizer settings # COBYLA for auto-FDing
optimizer = "scipy"
if optimizer == "scipy":
    prob.driver = om.ScipyOptimizeDriver(optimizer="SLSQP", tol=1.0e-9, disp=True)
elif optimizer == "pyoptsparse":
    prob.driver = om.pyOptSparseDriver(optimizer="SNOPT")
    # prob.driver.opt_settings['Major feasibility tolerance'] = 1e-5 # lower tolerance to prevent tight oscillations

# Start the optimization
print("\n==> Starting optimization...")
prob.setup()

debug = False
if debug:
    print("Checking partials...", flush=True)
    prob.check_partials(compact_print=True)

else:
    prob.run_driver()

    # report the final optimal design
    design_hdl = f2f_subsystem._design_hdl
    for var in f2f_model.get_variables():
        opt_value = prob.get_val(f"f2fSystem.{var.name}")
        design_hdl.write(f"\t{var.name} = {opt_value}")

    prob.cleanup()
    # close the design hdl file
    f2f_subsystem.cleanup()
