"""
1_panel_thickness.py

Run a coupled optimization of the panel thicknesses of the wing structure.
No shape variables are included in this optimization.
This example is finished and converged well in SNOPT
"""

from pyoptsparse import SNOPT, Optimization
from funtofem import *
from mpi4py import MPI
from tacs import caps2tacs
import os, time
import argparse

parent_parser = argparse.ArgumentParser(add_help=False)
parent_parser.add_argument("--hotstart", type=bool, default=False)
parent_parser.add_argument("--testderiv", type=bool, default=False)
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
T_ref = 268.338  # struct ref temp
T_inf = 500  # K
q_inf = 5.21945e4  # Dynamic pressure

# Construct the FUNtoFEM model
f2f_model = FUNtoFEMmodel("ssw-sizing3")
tacs_model = caps2tacs.TacsModel.build(
    csm_file=csm_path,
    comm=comm,
    problem_name="capsStruct3",
    active_procs=[0],
    verbosity=1,
)
tacs_model.mesh_aim.set_mesh(
    edge_pt_min=2,
    edge_pt_max=20,
    global_mesh_size=0.3,
    max_surf_offset=0.2,
    max_dihedral_angle=15,
).register_to(tacs_model)
f2f_model.structural = tacs_model

tacs_aim = tacs_model.tacs_aim
tacs_aim.set_config_parameter("view:flow", 0)
tacs_aim.set_config_parameter("view:struct", 1)

for proc in tacs_aim.active_procs:
    if comm.rank == proc:
        aim = tacs_model.mesh_aim.aim
        aim.input.Mesh_Sizing = {
            "chord": {"numEdgePoints": 20},
            "span": {"numEdgePoints": 8},
            "vert": {"numEdgePoints": 4},
        }

# add tacs constraints in
caps2tacs.PinConstraint("root").register_to(tacs_model)
caps2tacs.TemperatureConstraint("midplane").register_to(tacs_model)

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

for iOML in range(1, nOML + 1):
    name = f"OML{iOML}"
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
    early_stopping=True, min_forward_steps=100, min_adjoint_steps=20
)

aoa = cruise.get_variable("AOA", set_active=True)
aoa.set_bounds(lower=0.0, value=2.0, upper=4.0, scale=10)

clift = Function.lift(body=0).register_to(cruise)
cdrag = Function.drag(body=0).register_to(cruise)
ksfailure = (
    Function.ksfailure(ks_weight=10.0, safety_factor=1.5)
    .optimize(scale=1.0, upper=1.0, objective=False, plot=True, plot_name="ks-cruise")
    .register_to(cruise)
)
temperature = Function.temperature().register_to(cruise)
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
clift *= 2.5  # 0.095 => 1.33 approx
mass_wingbox = 308  # kg
q_inf = 1.21945e4
# flying wing, glider structure
mass_payload = 100  # kg
mass_frame = 0  # kg
mass_fuel_res = 2e3  # kg
LGM = mass_payload + mass_frame + mass_fuel_res + 2 * mass_wingbox
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
Tref = T_inf
nu = aluminum._nu12
alpha = aluminum._alpha1  # CTE (coeff thermal expansion)

# plate dimensions
a = 5.0 / (nribs + 1)  # 1-direction length
b = 1.0 / (nspars + 1)  # 2-direction width

# for each skin panel set up a buckling constraint
for iOML in range(1, nOML + 1):
    # get the associated skin thickness variable
    thick = wing.get_variable(f"OML{iOML}")

    # compute thermal stress and in-plane loads assuming plate is pinned
    # on all sides so no axial contraction (constrained)
    dT = temperature - Tref
    sigma_11 = (
        alpha * dT * E / (1 - nu)
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
    m1 = int(rho_0)

    # compute the critical in-plane load for an unstiffened panel in axial loading
    Dgeom_avg = D11  # would be sqrt(D11 * D22) but isotropic these are equal
    N11_cr = (
        np.pi**2 * Dgeom_avg / b**2 * (m1**2 / rho_0**2 + rho_0**2 / m1**2 + 2 * xi)
    )

    # compute the buckling failure criterion
    safety_factor = 1.5
    mu_thermal_buckle = N11 / N11_cr * safety_factor
    mu_thermal_buckle.set_name(f"therm_buckle_{iOML}").optimize(
        upper=1.0, scale=1e0, objective=False, plot=True
    ).register_to(f2f_model)

# skin thickness adjacency constraints
# ------------------------------------
if not test_derivatives:
    variables = f2f_model.get_variables()
    section_prefix = ["rib", "OML"]
    section_nums = [nribs, nOML]
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
design_in_file = os.path.join(base_dir, "design", "design-2.txt")
design_out_file = os.path.join(base_dir, "design", "design-3.txt")

design_folder = os.path.join(base_dir, "design")
if comm.rank == 0:
    if not os.path.exists(design_folder):
        os.mkdir(design_folder)
history_file = os.path.join(design_folder, "design-3.hst")
store_history_file = history_file if store_history else None
hot_start_file = history_file if hot_start else None

# Reload the previous design
f2f_model.read_design_variables_file(comm, design_in_file)

if comm.rank == 0:
    # f2f_driver.print_summary()
    f2f_model.print_summary()

manager = OptimizationManager(
    f2f_driver,
    design_out_file=design_out_file,
    hot_start=hot_start,
    hot_start_file=hot_start_file,
    debug=True,
)

# create the pyoptsparse optimization problem
opt_problem = Optimization("sswOpt", manager.eval_functions)

# add funtofem model variables to pyoptsparse
manager.register_to_problem(opt_problem)

# run an SNOPT optimization
snoptimizer = SNOPT(
    options={
        "Verify level": -1 if hot_start else 0,
        "Function precision": 1e-6,
        "Major step limit": 5e-2,
        "Nonderivative linesearch": None,
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
