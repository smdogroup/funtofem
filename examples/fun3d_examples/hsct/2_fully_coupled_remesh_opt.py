"""
This is the fully coupled aerothermoelastic optimization of the HSCT.
NOTE: You need to run the 1_sizing_optimization.py first and leave the
optimal panel thickness design variables in the meshes folder before running this.

NOTE : don't call this script with mpiexec_mpt, call it with python (otherwise system calls won't work)
"""
from pyoptsparse import SNOPT, Optimization
import os
from mpi4py import MPI
from funtofem import *
from tacs import caps2tacs

comm = MPI.COMM_WORLD
base_dir = os.path.dirname(os.path.abspath(__file__))
fun3d_dir = os.path.join(base_dir, "meshes")
csm_path = os.path.join(fun3d_dir, "hsct.csm")
analysis_file = os.path.join(base_dir, "_run_funtofem_analysis.py")

# FUNtoFEM and SHAPE MODELS
# ---------------------------------------------------------
hsct_model = FUNtoFEMmodel("hsct_MDO")

# design the FUN3D aero shape model
flow_model = Fun3dModel.build(csm_file=csm_path, comm=comm, project_name="hsct")
m_aflr_aim = flow_model.aflr_aim

flow_aim = flow_model.fun3d_aim
flow_aim.set_config_parameter("mode:flow", 1)
flow_aim.set_config_parameter("mode:struct", 0)

m_aflr_aim.set_surface_mesh(ff_growth=1.4, mesh_length=5.0, use_quads=True)
Fun3dBC.viscous(caps_group="wall", wall_spacing=1e-4).register_to(flow_model)
Fun3dBC.viscous(caps_group="staticWall", wall_spacing=1e-4).register_to(flow_model)
Fun3dBC.SymmetryY(caps_group="SymmetryY").register_to(flow_model)
Fun3dBC.Farfield(caps_group="Farfield").register_to(flow_model)
flow_model.setup()
hsct_model.flow = flow_model

# design the TACS struct shape model
tacs_model = caps2tacs.TacsModel.build(
    csm_file=csm_path, comm=comm, problem_name="capsStruct2"
)
tacs_model.mesh_aim.set_mesh(  # need a refined-enough mesh for the derivative test to pass
    edge_pt_min=20,
    edge_pt_max=30,
    global_mesh_size=0.01,
    max_surf_offset=0.01,
    max_dihedral_angle=5,
).register_to(
    tacs_model
)
tacs_aim = tacs_model.tacs_aim
tacs_aim.set_config_parameter("mode:flow", 0)
tacs_aim.set_config_parameter("mode:struct", 1)
hsct_model.structural = tacs_model

# tacs model constraints
caps2tacs.PinConstraint("root").register_to(tacs_model)
caps2tacs.TemperatureConstraint("midplane").register_to(tacs_model)

# BODIES, STRUCT DVs and SHAPE DVs
# ---------------------------------------------------
wing = Body.aerothermoelastic("wing", boundary=4)
wing.relaxation(AitkenRelaxation())

# setup the material and shell properties
aluminum = caps2tacs.Isotropic.aluminum().register_to(tacs_model)

nribs = int(tacs_model.get_config_parameter("nribs"))
nspars = int(tacs_model.get_config_parameter("nspars"))
nOML = int(tacs_aim.get_output_parameter("nOML"))

for irib in range(1, nribs + 1):
    name = f"rib{irib}"
    caps2tacs.ShellProperty(
        caps_group=name, material=aluminum, membrane_thickness=0.05
    ).register_to(tacs_model)
    Variable.structural(name, value=0.1).set_bounds(
        lower=0.001, upper=0.04, scale=100.0
    ).register_to(wing)
for ispar in range(1, nspars + 1):
    name = f"spar{ispar}"
    caps2tacs.ShellProperty(
        caps_group=name, material=aluminum, membrane_thickness=0.05
    ).register_to(tacs_model)
    Variable.structural(name, value=0.1).set_bounds(
        lower=0.001, upper=0.04, scale=100.0
    ).register_to(wing)
for iOML in range(1, nOML + 1):
    name = f"OML{iOML}"
    caps2tacs.ShellProperty(
        caps_group=name, material=aluminum, membrane_thickness=0.03
    ).register_to(tacs_model)
    Variable.structural(name, value=0.1).set_bounds(
        lower=0.001, upper=0.04, scale=100.0
    ).register_to(wing)
for name in ["LEspar", "TEspar"]:
    caps2tacs.ShellProperty(
        caps_group=name, material=aluminum, membrane_thickness=0.03
    ).register_to(tacs_model)
    Variable.structural(name, value=0.1).set_bounds(
        lower=0.001, upper=0.04, scale=100.0
    ).register_to(wing)

# structural shape variables
for prefix in ["rib", "spar"]:
    Variable.shape(f"{prefix}_a1", value=1.0).set_bounds(
        lower=0.6, upper=1.4
    ).register_to(wing)
    Variable.shape(f"{prefix}_a2", value=0.0).set_bounds(
        lower=-0.3, upper=0.3
    ).register_to(wing)

# wing size and shape variables
phi1_LE = Variable.shape(f"wing:phi1_LE", value=70).set_bounds(lower=60, upper=80)
phi2_LE = Variable.shape(f"wing:phi2_LE", value=50).set_bounds(lower=40, upper=60)
phi3_LE = Variable.shape(f"wing:phi3_LE", value=30).set_bounds(lower=20, upper=40)
phi_LE = [phi1_LE, phi2_LE, phi3_LE]

phi1_TE = Variable.shape(f"wing:phi1_TE", value=15).set_bounds(lower=10, upper=20)
phi2_TE = Variable.shape(f"wing:phi2_TE", value=10).set_bounds(lower=-15, upper=15)
phi3_TE = Variable.shape(f"wing:phi3_TE", value=8).set_bounds(lower=-5, upper=15)
phi_TE = [phi1_TE, phi2_TE, phi3_TE]

wing_area = Variable.shape("wing:area", value=700).set_bounds(lower=500, upper=800)
wing_aspect = Variable.shape("wing:aspect", value=7.0).set_bounds(lower=4.0, upper=8.0)
wing_span_fr1 = Variable.shape("wing:span_fr1", value=0.2).set_bounds(
    lower=0.1, upper=0.3
)
wing_span_fr2 = Variable.shape("wing:span_fr2", value=0.3).set_bounds(
    lower=0.2, upper=0.5
)
other_wing_vars = [wing_area, wing_aspect, wing_span_fr1, wing_span_fr2]

for var in phi_LE + phi_TE + other_wing_vars:
    var.register_to(wing)

# TODO : add wing airfoil shape variables
# TODO : add fuselage and tail shape variables

wing.register_to(hsct_model)

safety_factor = 1.5
ks_max = 1 / safety_factor

# SCENARIOS, AERO DVs, and remaining SHAPE VARS
# -----------------------------------------------

# NOTE: shape variables can be assigned to the body or scenario
# when using ESP/CAPS, it doesn't really matter

climb = Scenario.steady("climb", steps=500)
climb.fun3d_project(flow_aim.project_name)
climb.set_temperature(T_ref=300.0, T_inf=300.0)  # modify this
climb_qinf = 1e3  # TBD on this
climb.set_flow_units(qinf=climb_qinf, flow_dt=1.0)
ksfailure_climb = Function.ksfailure().optimize(
    scale=30.0, upper=ks_max, objective=False, plot=True
)
cl_climb = Function.lift(body=0)
cd_climb = Function.drag(body=0)
# aoa_climb = climb.get_variable("AOA").set_bounds(lower=3.0, value=4.0, upper=5.0)
# mach_climb = climb.get_variable("Mach").set_bounds(lower=0.5, value=0.7, upper=0.9)
for func in [ksfailure_climb, cl_climb, cd_climb]:
    func.register_to(climb)
climb.register_to(hsct_model)

# cruise flight condition
# altitude - 60 kft, ,
_mach_cruise = 2.5
_ainf_cruise = 295  # m/s
_rho_inf_cruise = 0.1165  # kg / m^3
# _mu_cruise = 1.42e-5 # kg/(m-s)
_aoa_cruise = 2.0
_Tinf_cruise = 216  # K
_vinf_cruise = _mach_cruise * _ainf_cruise
_qinf_cruise = 0.5 * _rho_inf_cruise * _vinf_cruise**2

cruise = Scenario.steady("cruise", steps=500)
cruise.fun3d_project(flow_aim.project_name)
cruise.set_temperature(T_ref=_Tinf_cruise, T_inf=_Tinf_cruise)
cruise.set_flow_units(qinf=_qinf_cruise, flow_dt=1.0)
ksfailure_cruise = Function.ksfailure().optimize(
    scale=30.0, upper=ks_max, objective=False, plot=True
)
cl_cruise = Function.lift(body=0)
cd_cruise = Function.drag(body=0)
moment = Function.moment(body=0).optimize(
    lower=0.0, upper=0.0, scale=1.0, objective=False, plot=True
)
wing_mass = Function.mass()
aoa_cruise = cruise.get_variable("AOA").set_bounds(
    lower=1.0, value=_aoa_cruise, upper=4.0
)
mach_cruise = cruise.get_variable("Mach").set_bounds(
    lower=2.3, value=_mach_cruise, upper=2.7
)
for func in [ksfailure_cruise, moment, wing_mass]:
    func.register_to(cruise)
cruise.register_to(hsct_model)

# COMPOSITE FUNCTIONS
# -----------------------------------------

# TOGW
g = 9.81  # m/s^2
lb_to_N = 4.448  # N/lb
tsfc = 3.9e-5  # kg/N/s, Rolls Royce Olympus 593 engine
fuselage_tail_weight = 6e5  # N
fuel_reserve_fraction = 0.06
num_passengers = 300
passenger_weight = 230 * num_passengers * lb_to_N
crew_weight = (450 + 5 * num_passengers) * lb_to_N
# descent_fuel = 6000 * lb_to_N  # N, fixed based on NASA report

wing_weight = 2 * wing_mass * g  # m/s^2 => N, doubled for sym
empty_weight = wing_weight + fuselage_tail_weight
boarded_weight = empty_weight + passenger_weight + crew_weight

cruise_lift = cl_cruise * _qinf_cruise * wing_area
cruise_drag = cd_cruise * _qinf_cruise * wing_area

takeoff_weight_ratio = 0.97
climb_weight_ratio = 0.985
land_weight_ratio = 0.995
rem_weight_ratios = takeoff_weight_ratio * climb_weight_ratio * land_weight_ratio

_range = 12800  # km
_range *= 1e3  # to m
cruise_LoverD = cruise_lift / cruise_drag
cruise_weight_ratio = CompositeFunction.exp(
    -_range * tsfc / _vinf_cruise / cruise_LoverD
)

mission_weight_ratio = rem_weight_ratios * cruise_weight_ratio
fuel_weight_ratio = 1.06 * (1 - mission_weight_ratio)  # 6% reserve fuel
togw = boarded_weight / (1 - fuel_weight_ratio)
togw.set_name("takeoff-gross-weight").optimize(  # kg
    lower=2e5, upper=3e5, scale=1.0, objective=True
).register_to(hsct_model)

# feasible wing span constraints => prevent negative sectional chord length
wing_sspan = 0.5 * (wing_area * wing_aspect) ** 0.5
wing_span_fr3 = 1 - wing_span_fr1 - wing_span_fr2
wing_sspans = [
    wing_sspan * wing_span_fr1,
    wing_sspan * wing_span_fr2,
    wing_sspan * wing_span_fr3,
]
phi_LE_rad = [_ * np.pi / 180 for _ in phi_LE]
phi_TE_rad = [_ * np.pi / 180 for _ in phi_TE]
CompositeFunction.tan(phi_LE_rad[0])
CompositeFunction.tan(phi_LE_rad[2])
CompositeFunction.tan(phi_TE_rad[0])
CompositeFunction.tan(phi_TE_rad[2])
chord_drops = [
    wing_sspans[i]
    * (CompositeFunction.tan(phi_LE_rad[i]) + CompositeFunction.tan(phi_TE_rad[i]))
    for i in range(3)
]
area_drops = [
    wing_sspans[0] * 0.5 * chord_drops[0],
    0.5 * wing_sspans[1] * (2 * chord_drops[0] + chord_drops[1]),
    0.5 * wing_sspans[2] * (2 * chord_drops[0] + 2 * chord_drops[1] + chord_drops[2]),
]
chords = [(wing_area + sum(area_drops)) / wing_sspan]
for i in range(3):
    chords += [chords[i] - chord_drops[i]]
# Composite function for each chord to make sure it is nonnegative
for i in range(4):
    chords[i].set_name(f"wing_chord{i}").optimize(
        lower=0.0, objective=False, scale=1.0
    ).register_to(hsct_model)

# adjacency skin thickness constraints
variables = hsct_model.get_variables()
adj_scale = 10.0
section_prefix = ["rib", "spar", "OML"]
section_nums = [nribs, nspars, nOML]
for isection, prefix in enumerate(section_prefix):
    section_num = section_nums[isection]
    for iconstr in range(1, section_num):
        left_var = hsct_model.get_variables(names=f"{prefix}{iconstr}")
        right_var = hsct_model.get_variables(names=f"{prefix}{iconstr+1}")
        adj_constr = (left_var - right_var) / left_var
        if prefix in ["rib", "OML"]:
            adj_ratio = 0.5
        else:
            adj_ratio = 4.0
        adj_constr.set_name(f"{prefix}{iconstr}-{iconstr+1}").optimize(
            lower=-adj_ratio, upper=adj_ratio, scale=1.0, objective=False
        ).register_to(hsct_model)

# BUILD THE DRIVER. NO DISCIPLINE INTERFACES IN DRIVER SCRIPT
# ------------------------------------------------------

# load in the previous design from the sizing optimization
# to overwrite the initial values
sizing_file = os.path.join(fun3d_dir, "sizing_design.txt")
hsct_model.read_design_variables_file(comm, sizing_file)

# build the solvers and coupled driver
solvers = SolverManager(comm)
fun3d_remote = Fun3dRemote(analysis_file, fun3d_dir, nprocs=48)
f2f_driver = FuntofemShapeDriver.aero_remesh(solvers, hsct_model, fun3d_remote)

# PYOPTSPARSE Optimization
# ------------------------------------------------------

hot_start = False
store_history = True

# create an OptimizationManager object for the pyoptsparse optimization problem
design_out_file = os.path.join(fun3d_dir, "full_design.txt")
manager = OptimizationManager(
    f2f_driver, design_out_file=design_out_file, hot_start=hot_start
)

# create the pyoptsparse optimization problem
opt_problem = Optimization("hsctMDO", manager.eval_functions)

# add funtofem model variables to pyoptsparse
manager.register_to_problem(opt_problem)

# run an SNOPT optimization
snoptimizer = SNOPT(options={"IPRINT": 1})

history_file = f"hsctMDO.hst"
store_history_file = history_file if store_history else None
hot_start_file = history_file if hot_start else None

sol = snoptimizer(
    opt_problem,
    sens=manager.eval_gradients,
    storeHistory=store_history_file,
    hotStart=hot_start_file,
)

# print final solution
sol_xdict = sol.xStar
print(f"Final solution = {sol_xdict}", flush=True)
