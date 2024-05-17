"""
2 - aerothermoelastic two-way coupled sizing of a simple wing in laminar flow with thermal buckling constraints
Sean Engelstad, GT SMDO Lab
"""

from pyoptsparse import SNOPT, Optimization
from funtofem import *
from mpi4py import MPI
from tacs import caps2tacs
import os, time
import argparse

parent_parser = argparse.ArgumentParser(add_help=False)
parent_parser.add_argument(
    "--hotstart", default=False, action=argparse.BooleanOptionalAction
)
parent_parser.add_argument(
    "--testderiv", default=False, action=argparse.BooleanOptionalAction
)
args = parent_parser.parse_args()

# options
hot_start = args.hotstart
store_history = True
test_derivatives = args.testderiv

comm = MPI.COMM_WORLD

base_dir = os.path.dirname(os.path.abspath(__file__))
csm_path = os.path.join(base_dir, "geometry", "ssw.csm")

nprocs_tacs = 8

global_debug_flag = False

# Derivative test stuff
FILENAME = "complex-step.txt"
FILEPATH = os.path.join(base_dir, FILENAME)

aitken_file = os.path.join(base_dir, "aitken-hist.txt")

# FUNTOFEM MODEL
# <----------------------------------------------------
# Freestream quantities -- see README
T_inf = 550.0  # K, freestream temp
T_ref = 268.338  # struct ref temp
temp_BC = 300 - T_ref  # K, gauge temperature
# lower the dynamic pressure, since some inc due to temp
q_inf = 2.21945e4  # Dynamic pressure

# Construct the FUNtoFEM model
f2f_model = FUNtoFEMmodel("ssw-sizing3")
tacs_model = caps2tacs.TacsModel.build(
    csm_file=csm_path,
    comm=comm,
    problem_name="capsStruct2",
    active_procs=[0],
    verbosity=1,
)
tacs_model.mesh_aim.set_mesh(
    edge_pt_min=2,
    edge_pt_max=50,
    global_mesh_size=0.02,
    max_surf_offset=0.2,
    max_dihedral_angle=15,
).register_to(tacs_model)
f2f_model.structural = tacs_model

tacs_aim = tacs_model.tacs_aim
for proc in tacs_aim.active_procs:
    if comm.rank == proc:
        aim = tacs_model.mesh_aim.aim
        aim.input.Mesh_Sizing = {
            "chord": {"numEdgePoints": 20},
            "span": {"numEdgePoints": 10},
            # "vert": {"numEdgePoints": 4},
        }

tacs_aim.set_config_parameter("view:flow", 0)
tacs_aim.set_config_parameter("view:struct", 1)
tacs_aim.set_config_parameter("nspars", 1)  # need only one spar here
tacs_aim.set_config_parameter("thermal", 1)
rib_a1 = 0.6
spar_a1 = 0.6
tacs_aim.set_design_parameter("spar_a1", spar_a1)
tacs_aim.set_design_parameter("rib_a1", rib_a1)
tacs_aim.set_config_parameter("allOMLDVs", 1)

# add tacs constraints in
caps2tacs.PinConstraint("root").register_to(tacs_model)
caps2tacs.TemperatureConstraint("midplane", temperature=temp_BC).register_to(tacs_model)

# ---------------------------------------------------->

# BODIES AND STRUCT DVs
# <----------------------------------------------------
wing = Body.aerothermoelastic("wing", boundary=1)

# setup the material and shell properties
aluminum = caps2tacs.Isotropic.aluminum().register_to(tacs_model)

nribs = int(tacs_model.get_config_parameter("nribs"))
nspars = int(tacs_model.get_config_parameter("nspars"))
nOML = nribs - 1

for irib in range(1, nribs + 1):
    name = f"rib{irib}"
    prop = caps2tacs.ShellProperty(
        caps_group=name, material=aluminum, membrane_thickness=0.04
    ).register_to(tacs_model)
    Variable.structural(name, value=0.01).set_bounds(
        lower=0.001, upper=0.15, scale=100.0
    ).register_to(wing)

for ispar in range(1, nspars + 1):
    name = f"spar{ispar}"
    prop = caps2tacs.ShellProperty(
        caps_group=name, material=aluminum, membrane_thickness=0.04
    ).register_to(tacs_model)
    Variable.structural(name, value=0.01).set_bounds(
        lower=0.001, upper=0.15, scale=100.0
    ).register_to(wing)

for prefix in ["lOML", "rOML"]:
    for iOML in range(1, nOML + 1):
        name = prefix + str(iOML)
        prop = caps2tacs.ShellProperty(
            caps_group=name, material=aluminum, membrane_thickness=0.04
        ).register_to(tacs_model)
        Variable.structural(name, value=0.01).set_bounds(
            lower=0.001, upper=0.15, scale=100.0
        ).register_to(wing)

for prefix in ["LE", "TE"]:
    name = f"{prefix}spar"
    prop = caps2tacs.ShellProperty(
        caps_group=name, material=aluminum, membrane_thickness=0.04
    ).register_to(tacs_model)
    Variable.structural(name, value=0.01).set_bounds(
        lower=0.001, upper=0.15, scale=100.0
    ).register_to(wing)

# register the wing body to the model
wing.register_to(f2f_model)

# ---------------------------------------------------->

# INITIAL STRUCTURE MESH, SINCE NO STRUCT SHAPE VARS
# <----------------------------------------------------

tacs_aim.setup_aim()
tacs_aim.pre_analysis()

# ---------------------------------------------------->

# SCENARIOS
# <----------------------------------------------------

# make a funtofem scenario
cruise = Scenario.steady(
    "cruise_hot",
    steps=300,
    forward_coupling_frequency=10,  # 300 total fun3d steps
    adjoint_steps=100,
    adjoint_coupling_frequency=30,  # 3000 total adjoint steps
    uncoupled_steps=200,
)
cruise.set_stop_criterion(
    early_stopping=True, min_forward_steps=100, min_adjoint_steps=int(1000 / 30)
)

aoa = cruise.get_variable("AOA", set_active=True)
# aoa var not read in since scenario changes from cruise to cruise_hot so user-specified here
aoa.set_bounds(lower=0.0, value=2.0, upper=3.0, scale=10)

clift = Function.lift(body=0).register_to(cruise)
cdrag = Function.drag(body=0).register_to(cruise)
ksfailure = (
    Function.ksfailure(ks_weight=10.0, safety_factor=1.5)
    .optimize(scale=1.0, upper=1.0, objective=False, plot=True, plot_name="ks-cruise")
    .register_to(cruise)
)
temp_gauge_area = (
    Function.temperature()
    .optimize(scale=1e-2, objective=False, upper=1e10, plot=True, plot_name="temp")
    .register_to(cruise)
)
mass_wingbox = Function.mass().register_to(cruise)
cruise.set_temperature(T_ref=T_ref, T_inf=T_inf)
cruise.set_flow_ref_vals(qinf=q_inf)
cruise.register_to(f2f_model)

# ---------------------------------------------------->

# COMPOSITE FUNCTIONS
# <----------------------------------------------------

# steady flight constraint L=W
# at end of cruise (fix later)
# ----------------------------
# improved lift coeff with a better airfoil
# adjusted with a multiplier (will shape optimize for this later)
clift *= 4.5  # 0.095 => 1.33 approx
# flying wing
mass_payload = 100  # kg
mass_frame = 600  # kg, mostly flying wing
LGM = mass_payload + mass_frame + 2 * mass_wingbox
LGW = 9.81 * LGM  # kg => N
dim_lift = clift * 2 * q_inf
load_factor = dim_lift - 1.0 * LGW
load_factor.set_name("steady_flight").optimize(
    lower=0.0, upper=0.0, scale=1e-3, objective=False, plot=True
).register_to(f2f_model)

# TOGW
# ----------------------------
takeoff_WR = 0.97
climb_WR = 0.985
land_WR = 0.995
_range = 12800  # km
_range *= 1e3  # m
tsfc = 3.9e-5  # kg/N/s, Rolls Royce Olympus 593 engine
_mach_cruise = 0.5
_ainf_cruise = 295  # m/s, @ 60 kft
_vinf_cruise = _mach_cruise * _ainf_cruise
cruise_WR = CompositeFunction.exp(-_range * tsfc / _vinf_cruise * cdrag / clift)
fuel_WR = 1.06 * (1 - takeoff_WR * climb_WR * land_WR * cruise_WR)
togw = LGW / (1 - fuel_WR)
togw.set_name("togw").optimize(  # kg
    scale=1.0e-5, objective=True, plot=True, plot_name="togw"
).register_to(f2f_model)

# Buckling constraints
# --------------------

# define material properties here (assuming metal / isotropic)
E = aluminum._E1
Tref = T_ref
nu = aluminum._nu12
alpha = aluminum._alpha1  # CTE (coeff thermal expansion)

# plate dimensions
a = 5.0 / (nribs + 1)  # 1-direction length
# two internal panels
chord = 1.0
b1 = chord * 0.5 * spar_a1
b2 = chord - b1
blist = [b1, b2]

wing_area = 10
temp_gauge = temp_gauge_area / wing_area

rib_spaces = np.linspace(0.0, 1.0, nribs + 1)
rib_a2 = 0.0
rib_a3 = 1.0 - rib_a1 - rib_a2
mod_rib_spaces = rib_a1 * rib_spaces + rib_a2 * rib_spaces**2 + rib_a3 * rib_spaces**3

# for each skin panel set up a buckling constraint
# panel width of this panel
b = b2
for iOML in range(1, nOML + 1):
    # panel length computed using eta'(eta) panel spacing formula
    rib_space = mod_rib_spaces[iOML + 1] - mod_rib_spaces[iOML]
    a = rib_space * 5.0  # 5.0 is sspan

    # get the associated skin thickness variable
    thick = wing.get_variable(f"rOML{iOML}")

    # compute thermal stress and in-plane loads assuming plate is pinned
    # on all sides so no axial contraction (constrained)
    sigma_11 = (
        alpha * temp_gauge * E / (1 - nu)
    )  # compressive thermal stress here (+ is compressive)
    N11 = sigma_11 * thick

    # compute some laminate plate properties (despite metal and non-laminate)
    Q11 = E / (1 - nu**2)
    Q22 = Q11
    Q12 = nu * Q11
    G12 = E / 2.0 / (1 + nu)
    Q66 = G12
    I = thick**3 / 12.0
    D11 = Q11 * I
    D22 = Q22 * I
    D12 = Q12 * I
    D66 = G12 * I

    # compute important non-dimensional parameters
    rho_0 = a / b
    # xi normally defined with D but only one-ply metal so simplify with floats only (not CompositeFunctions)
    xi = (Q12 + 2 * Q66) / (Q11 * Q22) ** 0.5

    # then you can assume m1 close to rho_0 the plate aspect ratio
    m1 = np.max([int(rho_0), 1.0])

    # compute the critical in-plane load for an unstiffened panel in axial loading
    Dgeom_avg = D11  # would be sqrt(D11 * D22) but isotropic these are equal
    N11_cr = (
        np.pi**2 * Dgeom_avg / b**2 * (m1**2 / rho_0**2 + rho_0**2 / m1**2 + 2 * xi)
    )

    # compute the buckling failure criterion
    safety_factor = 3.0
    mu_thermal_buckle = N11 / N11_cr * safety_factor
    mu_thermal_buckle.set_name(f"therm_buckle_rOML{iOML}").optimize(
        upper=1.0, scale=1e0, objective=False, plot=True
    ).register_to(f2f_model)


# skin thickness adjacency constraints
# ------------------------------------
if not test_derivatives:
    variables = f2f_model.get_variables()
    section_prefix = ["rib", "lOML", "rOML"]
    section_nums = [nribs, nOML, nOML]
    for isection, prefix in enumerate(section_prefix):
        section_num = section_nums[isection]
        for iconstr in range(1, section_num):
            left_var = f2f_model.get_variables(names=f"{prefix}{iconstr}")
            right_var = f2f_model.get_variables(names=f"{prefix}{iconstr+1}")
            # adj_constr = (left_var - right_var) / left_var
            # adj_ratio = 0.15
            adj_constr = left_var - right_var
            adj_diff = 0.002
            adj_constr.set_name(f"{prefix}{iconstr}-{iconstr+1}").optimize(
                lower=-adj_diff, upper=adj_diff, scale=1.0, objective=False
            ).register_to(f2f_model)


# ---------------------------------------------------->

# DISCIPLINE INTERFACES AND DRIVERS
# <----------------------------------------------------

solvers = SolverManager(comm)
solvers.flow = Fun3d14Interface(
    comm,
    f2f_model,
    fun3d_dir="cfd",
    forward_stop_tolerance=1e-15,
    forward_min_tolerance=1e-12,
    adjoint_stop_tolerance=1e-13,
    adjoint_min_tolerance=1e-10,
    debug=global_debug_flag,
)
# fun3d_project_name = "ssw-pw1.2"
solvers.structural = TacsSteadyInterface.create_from_bdf(
    model=f2f_model,
    comm=comm,
    nprocs=nprocs_tacs,
    bdf_file=tacs_aim.root_dat_file,
    prefix=tacs_aim.root_analysis_dir,
    debug=global_debug_flag,
)

f2f_transfer_settings = TransferSettings(npts=200)

# Build the FUNtoFEM driver
f2f_driver = FUNtoFEMnlbgs(
    solvers=solvers,
    transfer_settings=f2f_transfer_settings,
    model=f2f_model,
    debug=global_debug_flag,
    reload_funtofem_states=False,
)

if test_derivatives:  # test using the finite difference test
    # load the previous design
    # design_in_file = os.path.join(base_dir, "design", "sizing-oneway.txt")
    # f2f_model.read_design_variables_file(comm, design_in_file)

    start_time = time.time()

    # run the finite difference test
    max_rel_error = TestResult.derivative_test(
        "fun3d+tacs-ssw3",
        model=f2f_model,
        driver=f2f_driver,
        status_file="3-derivs.txt",
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
# <----------------------------------------------------

# create an OptimizationManager object for the pyoptsparse optimization problem
design_out_file = os.path.join(base_dir, "design", "design-3.txt")

design_folder = os.path.join(base_dir, "design")
if comm.rank == 0:
    if not os.path.exists(design_folder):
        os.mkdir(design_folder)
history_file = os.path.join(design_folder, "design-3.hst")
store_history_file = history_file if store_history else None
hot_start_file = history_file if hot_start else None

# Reload the previous design
# design_in_file = os.path.join(base_dir, "design", "design-1.txt")
# f2f_model.read_design_variables_file(comm, design_in_file)

if comm.rank == 0:
    # f2f_driver.print_summary()
    f2f_model.print_summary()

manager = OptimizationManager(
    f2f_driver,
    design_out_file=design_out_file,
    hot_start=hot_start,
    hot_start_file=hot_start_file,
    debug=False,
)

# create the pyoptsparse optimization problem
opt_problem = Optimization("sswOpt", manager.eval_functions)

# add funtofem model variables to pyoptsparse
manager.register_to_problem(opt_problem)

# run an SNOPT optimization
snoptimizer = SNOPT(
    options={
        "Verify level": 0,  # -1 if hot_start else 0
        "Function precision": 1e-6,
        # "Major step limit": 1e-1,
        "Nonderivative linesearch": True,
        "Major Optimality tol": 1e-4,
    }
)

sol = snoptimizer(
    opt_problem,
    sens=manager.eval_gradients,
    storeHistory=store_history_file,
    hotStart=hot_start_file,
)

# print final solution
sol_xdict = sol.xStar

if comm.rank == 0:
    print(f"Final solution = {sol_xdict}", flush=True)

# ---------------------------------------------------->
