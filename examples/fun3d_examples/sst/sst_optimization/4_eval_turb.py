"""
Sean P. Engelstad, Georgia Tech 2023

This runs the fully coupled aerothermoelastic, turbulent forward analysis.
It prints out the aerostructural functionals for you to manually improve the design before full 
optimization.
"""
# TBD this one still needs to be made properly
import os
from mpi4py import MPI
from funtofem import *
from tacs import caps2tacs

comm = MPI.COMM_WORLD
base_dir = os.path.dirname(os.path.abspath(__file__))
fun3d_dir = os.path.join(base_dir, "cfd")
tacs_dir = os.path.join(base_dir, "struct")
csm_path = os.path.join(base_dir, "geometry", "sst_v2.csm")

# FUNtoFEM and SHAPE MODELS
# ---------------------------------------------------------
hsct_model = FUNtoFEMmodel("sst-turbulent")

# design the FUN3D aero shape model
flow_model = Fun3dModel.build(csm_file=csm_path, comm=comm, project_name="sst-turb")
m_aflr_aim = flow_model.aflr_aim

flow_aim = flow_model.fun3d_aim
flow_aim.set_config_parameter("mode:flow", 1)
flow_aim.set_config_parameter("mode:struct", 0)

# min_scale has the greatest effect on mesh size (scale that to affect # mesh elements)
aflr_aim.set_surface_mesh(
    ff_growth=1.4, mesh_length=8.0, min_scale=0.02, max_scale=0.5, use_quads=True
)
Fun3dBC.viscous(caps_group="wall", wall_spacing=1e-4).register_to(flow_model)
Fun3dBC.viscous(caps_group="staticWall", wall_spacing=1e-4).register_to(flow_model)
Fun3dBC.SymmetryY(caps_group="SymmetryY").register_to(flow_model)
Fun3dBC.Farfield(caps_group="Farfield").register_to(flow_model)
flow_model.setup()
hsct_model.flow = flow_model

# design the TACS struct shape model
tacs_model = caps2tacs.TacsModel.build(
    csm_file=csm_path, comm=comm, problem_name="capsStruct4"
)
tacs_model.mesh_aim.set_mesh(  # need a refined-enough mesh for the derivative test to pass
    edge_pt_min=2,
    edge_pt_max=20,
    global_mesh_size=0.3,
    max_surf_offset=0.2,
    max_dihedral_angle=15,
).register_to(
    tacs_model
)
tacs_aim = tacs_model.tacs_aim
tacs_aim.set_config_parameter("mode:flow", 0)
tacs_aim.set_config_parameter("mode:struct", 1)
for proc in tacs_aim.active_procs:
    if comm.rank == proc:
        aim = tacs_model.mesh_aim.aim
        aim.input.Mesh_Sizing = {
            "horizX": {"numEdgePoints": 20},
            "horizY": {"numEdgePoints": 8},
            "vert": {"numEdgePoints": 4},
        }
hsct_model.structural = tacs_model

# tacs model constraints
caps2tacs.PinConstraint("root").register_to(tacs_model)
caps2tacs.PinConstraint("wingFuse").register_to(tacs_model)
caps2tacs.TemperatureConstraint("midplane").register_to(tacs_model)

# BODIES, STRUCT DVs and SHAPE DVs
# ---------------------------------------------------
wing = Body.aerothermoelastic("wing", boundary=4)
wing.relaxation(AitkenRelaxation())

# setup the material and shell properties
titanium_alloy = caps2tacs.Isotropic.titanium_alloy().register_to(tacs_model)

nribs = int(tacs_model.get_config_parameter("wing:nribs"))
nspars = int(tacs_model.get_config_parameter("wing:nspars"))
nOML = int(tacs_aim.get_output_parameter("wing:nOML"))

for irib in range(1, nribs + 1):
    name = f"rib{irib}"
    caps2tacs.ShellProperty(
        caps_group=name, material=titanium_alloy, membrane_thickness=0.05
    ).register_to(tacs_model)
    Variable.structural(name, value=0.1).set_bounds(
        lower=0.001, upper=0.15, scale=100.0
    ).register_to(wing)
for ispar in range(1, nspars + 1):
    name = f"spar{ispar}"
    caps2tacs.ShellProperty(
        caps_group=name, material=titanium_alloy, membrane_thickness=0.05
    ).register_to(tacs_model)
    Variable.structural(name, value=0.1).set_bounds(
        lower=0.001, upper=0.15, scale=100.0
    ).register_to(wing)
for iOML in range(1, nOML + 1):
    name = f"OML{iOML}"
    caps2tacs.ShellProperty(
        caps_group=name, material=titanium_alloy, membrane_thickness=0.03
    ).register_to(tacs_model)
    Variable.structural(name, value=0.1).set_bounds(
        lower=0.001, upper=0.15, scale=100.0
    ).register_to(wing)
for name in ["LEspar", "TEspar"]:
    caps2tacs.ShellProperty(
        caps_group=name, material=titanium_alloy, membrane_thickness=0.03
    ).register_to(tacs_model)
    Variable.structural(name, value=0.1).set_bounds(
        lower=0.001, upper=0.15, scale=100.0
    ).register_to(wing)

# structural shape variables
for prefix in ["rib", "spar"]:
    Variable.shape(f"wing:{prefix}_a1", value=1.0).set_bounds(
        lower=0.6, upper=1.4
    ).register_to(wing)
    Variable.shape(f"wing:{prefix}_a2", value=0.0).set_bounds(
        lower=-0.3, upper=0.3
    ).register_to(wing)

# wing size and shape variables
cbar1 = Variable.shape("wing:cbar1", value=2.0).set_bounds(lower=1.5, upper=2.5)
cbar2 = Variable.shape("wing:cbar2", value=1.4).set_bounds(lower=1.0, upper=2.0)
cbar3 = Variable.shape("wing:cbar3", value=0.9).set_bounds(lower=0.5, upper=1.5)
cbars = [cbar1, cbar2, cbar3]

coffset1 = Variable.shape("wing:cbar_offset1", value=0.0).set_bounds(
    lower=-0.3, upper=0.3
)
coffset2 = Variable.shape("wing:cbar_offset2", value=0.0).set_bounds(
    lower=-0.3, upper=0.3
)
coffset3 = Variable.shape("wing:cbar_offset3", value=0.0).set_bounds(
    lower=-0.3, upper=0.3
)
coffsets = [coffset1, coffset2, coffset3]

dzhat1 = Variable.shape("wing:dzhat1", value=0.05).set_bounds(lower=0.0, upper=0.3)
dzhat2 = Variable.shape("wing:dzhat2", value=0.2).set_bounds(lower=0.0, upper=0.6)
dz_dihedral = Variable.shape("wing:dz_dihedral", value=-5.0).set_bounds(
    lower=-7.0, upper=0.0
)
dzs = [dzhat1, dzhat2, dz_dihedral]

wing_area = Variable.shape("wing:area", value=700).set_bounds(lower=500, upper=800)
wing_aspect = Variable.shape("wing:aspect", value=7.0).set_bounds(lower=4.0, upper=8.0)
wing_span_fr1 = Variable.shape("wing:span_fr1", value=0.2).set_bounds(
    lower=0.1, upper=0.3
)
wing_span_fr2 = Variable.shape("wing:span_fr2", value=0.3).set_bounds(
    lower=0.2, upper=0.5
)
other_wing_vars = [wing_area, wing_aspect, wing_span_fr1, wing_span_fr2]

for var in cbars + coffsets + dzs + other_wing_vars:
    var.register_to(wing)

# TODO : add wing airfoil shape variables
# TODO : add fuselage and tail shape variables

wing.register_to(hsct_model)

# SCENARIOS, AERO DVs, and remaining SHAPE VARS
# -----------------------------------------------

# NOTE: shape variables can be assigned to the body or scenario
# when using ESP/CAPS, it doesn't really matter

# climb flight condition
_climb_qinf = 3.28e4

climb = Scenario.steady("climb", steps=350, preconditioner_steps=200)
climb.fun3d_project(flow_aim.project_name)
climb.set_temperature(T_ref=300.0, T_inf=300.0)  # modify this
climb.set_flow_ref_vals(qinf=_climb_qinf, flow_dt=1.0)
ksfailure_climb = Function.ksfailure(ks_weight=10.0, safety_factor=1.5).optimize(
    scale=1.0, upper=1.0, objective=False, plot=True
)
cl_climb = Function.lift(body=0)
cd_climb = Function.drag(body=0)
aoa_climb = climb.get_variable("AOA").set_bounds(lower=0.0, value=3.0, upper=10.0)
mach_climb = climb.get_variable("Mach").set_bounds(lower=0.5, value=0.7, upper=0.9)
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

cruise = Scenario.steady("cruise", steps=350, preconditioner_steps=200)
cruise.fun3d_project(flow_aim.project_name)
cruise.set_temperature(T_ref=_Tinf_cruise, T_inf=_Tinf_cruise)
cruise.set_flow_ref_vals(qinf=_qinf_cruise, flow_dt=1.0)
ksfailure_cruise = Function.ksfailure(ks_weight=10.0, safety_factor=1.5).optimize(
    scale=1.0, upper=1.0, objective=False, plot=True
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
for func in [ksfailure_cruise, moment, wing_mass, cl_cruise, cd_cruise]:
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

cruise_lift = cl_cruise * _qinf_cruise
cruise_drag = cd_cruise * _qinf_cruise

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

# steady flight and climb conditions
cruise_weight = togw * 0.5 * (1 + cruise_weight_ratio)
steady_cruise = cruise_lift / cruise_weight
steady_cruise.set_name("steady_cruise").optimize(
    lower=1.0, upper=1.0, scale=1.0, objective=False
).register_to(hsct_model)
climb_lift = cl_climb * _climb_qinf
steady_climb = climb_lift / togw
steady_climb.set_name("steady_climb").optimize(
    lower=1.5, upper=2.5, scale=1.0, objective=False
).register_to(hsct_model)

# feasible wing chord constraints => prevent negative sectional chord length
wing_span_fr3 = 1 - wing_span_fr1 - wing_span_fr2
# normalized chord at tip = chord_tip / chord_mean
chord_tip_hat = (
    2
    - (cbar1 + cbar2) * wing_span_fr1
    - (cbar2 + cbar3) * wing_span_fr2
    - cbar3 * wing_span_fr3
) / wing_span_fr3
chord_tip_hat.set_name("chord_tip*").optimize(
    lower=0.0, objective=False, scale=1.0
).register_to(hsct_model)


# adjacency skin thickness constraints
variables = hsct_model.get_variables()
adj_scale = 10.0
section_prefix = ["rib", "OML"]
section_nums = [nribs, nOML]
for isection, prefix in enumerate(section_prefix):
    section_num = section_nums[isection]
    for iconstr in range(1, section_num):
        left_var = hsct_model.get_variables(names=f"{prefix}{iconstr}")
        right_var = hsct_model.get_variables(names=f"{prefix}{iconstr+1}")
        adj_constr = (left_var - right_var) / left_var
        adj_ratio = 0.15
        adj_constr.set_name(f"{prefix}{iconstr}-{iconstr+1}").optimize(
            lower=-adj_ratio, upper=adj_ratio, scale=1.0, objective=False
        ).register_to(hsct_model)

# measure the actual lift, drag of each scenario (properly normalized)
half_area = wing_area * 0.5
(cl_cruise / half_area).set_name("CL_cruise_ND").optimize(objective=False).register_to(
    hsct_model
)
(cd_cruise / half_area).set_name("CD_cruise_ND").optimize(objective=False).register_to(
    hsct_model
)
(cl_climb / half_area).set_name("CL_climb_ND").optimize(objective=False).register_to(
    hsct_model
)
(cd_climb / half_area).set_name("CD_climb_ND").optimize(objective=False).register_to(
    hsct_model
)

# measure dimensional lift, drag of each scenario

(cl_cruise * _qinf_cruise).set_name("CL_cruise_Dim").optimize(
    objective=False
).register_to(hsct_model)
(cd_cruise * _qinf_cruise).set_name("CD_cruise_Dim").optimize(
    objective=False
).register_to(hsct_model)
(cl_climb * _climb_qinf).set_name("CL_climb_Dim").optimize(objective=False).register_to(
    hsct_model
)
(cd_climb * _climb_qinf).set_name("CD_climb_Dim").optimize(objective=False).register_to(
    hsct_model
)

# BUILD THE DRIVER. NO DISCIPLINE INTERFACES IN DRIVER SCRIPT
# ------------------------------------------------------

# load in the previous design from the inviscid-AE design
sizing_file = os.path.join(base_dir, "design", "inviscid-ae.txt")
hsct_model.read_design_variables_file(comm, sizing_file)

# build the solvers and coupled driver
solvers = SolverManager(comm)
solvers.flow = Fun3dInterface(comm, hsct_model, fun3d_dir="cfd", auto_coords=False)
# solvers.structural will be built by the shape driver at runtime

# build the driver and run a forward analysis
# ----------------------------------------------------------------------------
transfer_settings = TransferSettings(
    elastic_scheme="meld", thermal_scheme="meld", isym=1, npts=200
)
f2f_driver = FuntofemShapeDriver.aero_morph(
    solvers, hsct_model, transfer_settings=transfer_settings, struct_nprocs=48
)
f2f_driver.solve_forward()

# eval composite functions and report all function values to the user
hsct_model.evaluate_composite_functions(compute_grad=False)

if comm.rank == 0:
    print(
        f"{hsct_model.name} Model, Turbulent Aerothermoelastic Forward analysis results...\n"
    )
    print("------------------------------------\n", flush=True)
    for func in hsct_model.get_functions(all=True):
        print(f"\t func {func.full_name} = {func.value.real}")
    print("------------------------------------\n", flush=True)
