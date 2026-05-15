"""
Sean P. Engelstad, Georgia Tech 2023

This is the fully coupled aeroelastic inviscid optimization of the HSCT.
NOTE : this is the analysis script corresponding to 5_fc_inviscid_ae_remesh.py.
    Now the mesh generation is performed HERE in the analysis script.
"""

import os
from mpi4py import MPI
from funtofem import *
from tacs import caps2tacs

# script inputs, NOTE : you need to run with restart=False first to make restart files
debug = False  # if True, CFD only runs one step each
restart = False  # use restart files for CFD
optimize_trim = False

comm = MPI.COMM_WORLD
base_dir = os.path.dirname(os.path.abspath(__file__))
fun3d_dir = os.path.join(base_dir, "cfd")
tacs_dir = os.path.join(base_dir, "struct")
csm_path = os.path.join(base_dir, "geometry", "sst_v2.csm")

# FUNtoFEM and SHAPE MODELS
# ---------------------------------------------------------
hsct_model = FUNtoFEMmodel("sst-inviscid")

# design the FUN3D aero shape model
flow_model = Fun3dModel.build(
    csm_file=csm_path, comm=comm, project_name="sst-inviscid", root=8
)
m_aflr_aim = flow_model.aflr_aim

flow_aim = flow_model.fun3d_aim
flow_aim.set_config_parameter("mode:flow", 1)
flow_aim.set_config_parameter("mode:struct", 0)

# min_scale has the greatest effect on mesh size (scale that to affect # mesh elements)
m_aflr_aim.set_surface_mesh(
    ff_growth=1.4, mesh_length=6.0, min_scale=0.006, max_scale=0.5, use_quads=True
)
Fun3dBC.inviscid(caps_group="wall").register_to(flow_model)
Fun3dBC.inviscid(caps_group="staticWall").register_to(flow_model)
Fun3dBC.SymmetryY(caps_group="SymmetryY").register_to(flow_model)
Fun3dBC.Farfield(caps_group="Farfield").register_to(flow_model)

# refine bottom of fuselage edge otherwise it's too coarse there and causes bad mesh quality + divergence
if comm.rank == flow_model.root:
    aim = m_aflr_aim.surface_aim
    aim.input.Mesh_Sizing = {"botFuse": {"scaleFactor": 0.1}}

flow_model.setup()
hsct_model.flow = flow_model

# design the TACS struct shape model
tacs_model = caps2tacs.TacsModel.build(
    csm_file=csm_path,
    comm=comm,
    problem_name="capsStruct5",
    active_procs=[_ for _ in range(12 if optimize_trim else 10)],
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
for proc in tacs_model.active_procs:
    if comm.rank == proc:
        mesh_aim = tacs_model.mesh_aim.aim
        mesh_aim.input.Mesh_Sizing = {
            "horizX": {"numEdgePoints": 20},
            "horizY": {"numEdgePoints": 8},
            "vert": {"numEdgePoints": 4},
        }

        # set optimal internal structure values
        # manually since don't want to compute
        # these derivatives here. (optimal values from internal structure optimization)
        shape_var_dict = {
            "wing:rib_a1": 0.65,
            "wing:rib_a2": -0.3,
            "wing:spar_a1": 1.163,
            "wing:spar_a2": -0.287,
        }
        for var_key in shape_var_dict:
            tacs_aim.geometry.despmtr[var_key].value = shape_var_dict[var_key]

hsct_model.structural = tacs_model

# tacs model constraints
caps2tacs.PinConstraint("root").register_to(tacs_model)
caps2tacs.PinConstraint("wingFuse").register_to(tacs_model)
caps2tacs.TemperatureConstraint("midplane").register_to(tacs_model)

# BODIES, STRUCT DVs and SHAPE DVs
# ---------------------------------------------------
wing = Body.aeroelastic("wing", boundary=4)
wing.relaxation(AitkenRelaxation(theta_init=0.01, theta_min=0.001, theta_max=0.3))

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

# don't want to optimize these => just fix them from optimal
# internal-structure design
# structural shape variables
# for prefix in ["rib", "spar"]:
#    Variable.shape(f"wing:{prefix}_a1", value=1.0).set_bounds(
#        lower=0.6, upper=1.4
#    ).register_to(wing)
#    Variable.shape(f"wing:{prefix}_a2", value=0.0).set_bounds(
#        lower=-0.3, upper=0.3
#    ).register_to(wing)

# wing size and shape variables
c1 = Variable.shape("wing:c1", value=41.2).set_bounds(lower=30.0, upper=50.0)
c2 = Variable.shape("wing:c2", value=28.862).set_bounds(lower=20.0, upper=40.0)
c3 = Variable.shape("wing:c3", value=18.554).set_bounds(lower=13.0, upper=25.0)
c4 = Variable.shape("wing:c4", value=7.422).set_bounds(lower=5.0, upper=15.0)
cbars = [c1, c2, c3, c4]

# coffset1 = Variable.shape("wing:cbar_offset1", value=0.0).set_bounds(
#    lower=-0.1, upper=0.1
# )
# coffset2 = Variable.shape("wing:cbar_offset2", value=0.0).set_bounds(
#    lower=-0.1, upper=0.1
# )
# coffset3 = Variable.shape("wing:cbar_offset3", value=0.0).set_bounds(
#    lower=-0.1, upper=0.1
# )
# coffsets = [coffset1, coffset2, coffset3]

# dzhat1 = Variable.shape("wing:dzhat1", value=0.05).set_bounds(lower=0.0, upper=0.1)
dzhat2 = Variable.shape("wing:dzhat2", value=0.2).set_bounds(lower=0.1, upper=0.3)
dz_dihedral = Variable.shape("wing:dz_dihedral", value=-5.0).set_bounds(
    lower=-5.1, upper=-2.0
)
dzs = [dzhat2, dz_dihedral]

# geometric AOA on wing
# geomAOA1 = Variable.shape("wing:geomAOA1", value=0.0).set_bounds(lower=-0.5, upper=0.5)
# geomAOA2 = Variable.shape("wing:geomAOA2", value=0.0).set_bounds(lower=-2.0, upper=2.0)
# geomAOA3 = Variable.shape("wing:geomAOA3", value=0.0).set_bounds(lower=-2.0, upper=2.0)
# geomAOA4 = Variable.shape("wing:geomAOA4", value=0.0).set_bounds(lower=-2.0, upper=2.0)
# geomAOA1, geomAOA2, geomAOA3, geomAOA4
for var in cbars + dzs:
    var.register_to(wing)

# wing thickness variables
wing_thick1 = Variable.shape("wing:thick1", value=1.237).set_bounds(
    lower=1.0, upper=1.4
)
wing_thick2 = Variable.shape("wing:thick2", value=1.154).set_bounds(
    lower=0.8, upper=1.3
)
wing_thick3 = Variable.shape("wing:thick3", value=0.742).set_bounds(
    lower=0.5, upper=0.9
)
wing_thick4 = Variable.shape("wing:thick4", value=0.296).set_bounds(
    lower=0.2, upper=0.5
)
for var in [wing_thick1, wing_thick2, wing_thick3, wing_thick4]:
    var.register_to(wing)

# canard shape variables
if optimize_trim:
    canard_area = Variable.shape("canard:area", value=70).set_bounds(
        lower=40, upper=200
    )
    canard_twist = Variable.shape("canard:geomAOA", value=5).set_bounds(
        lower=-5, upper=8
    )

    for canard_var in [canard_area, canard_twist]:
        canard_var.register_to(wing)

wing.register_to(hsct_model)
# end of wing section

# SCENARIOS, AERO DVs, and remaining SHAPE VARS
# -----------------------------------------------

# NOTE: shape variables can be assigned to the body or scenario
# when using ESP/CAPS, it doesn't really matter
_climb_qinf = 3.28e4

if not (debug):
    climb = Scenario.steady(
        "climb_inviscid",
        steps=650,
        uncoupled_steps=150,
    )
    climb.adjoint_steps = 300
else:
    climb = Scenario.steady("climb_inviscid", steps=2, uncoupled_steps=1)
    climb.adjoint_steps = 1
climb.fun3d_project(flow_aim.project_name)
climb.set_temperature(T_ref=300.0, T_inf=300.0)  # modify this
climb.set_flow_ref_vals(qinf=_climb_qinf, flow_dt=1.0)
ksfailure_climb = Function.ksfailure(ks_weight=10.0, safety_factor=1.5).optimize(
    scale=1.0, upper=1.0, objective=False, plot=True, plot_name="ks-climb"
)
cl_climb = Function.lift(body=0)
cd_climb = Function.drag(body=0)
aoa_climb = climb.get_variable("AOA").set_bounds(lower=0.0, value=4.0, upper=8.0)
mach_climb = climb.get_variable("Mach").set_bounds(lower=0.3, value=0.55, upper=0.9)
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

if not (debug):
    cruise = Scenario.steady(
        "cruise_inviscid",
        steps=400,
        uncoupled_steps=150,
    )
    cruise.adjoint_steps = 100
else:
    cruise = Scenario.steady("cruise_inviscid", steps=2, uncoupled_steps=1)
    cruise.adjoint_steps = 1
cruise.fun3d_project(flow_aim.project_name)
cruise.set_temperature(T_ref=_Tinf_cruise, T_inf=_Tinf_cruise)
cruise.set_flow_ref_vals(qinf=_qinf_cruise, flow_dt=1.0)
ksfailure_cruise = Function.ksfailure(ks_weight=10.0, safety_factor=1.5).optimize(
    scale=1.0, upper=1.0, objective=False, plot=True, plot_name="ks-cruise"
)
cl_cruise = Function.lift(body=0)
cd_cruise = Function.drag(body=0)
if optimize_trim:
    moment = Function.moment().optimize(
        lower=0.0,
        upper=0.0,
        scale=1.0e-3,
        objective=False,
        plot=True,
        plot_name="cm-cruise",
    )
wing_mass = Function.mass()
aoa_cruise = cruise.get_variable("AOA").set_bounds(
    lower=1.0, value=_aoa_cruise, upper=4.0
)
mach_cruise = cruise.get_variable("Mach").set_bounds(
    lower=2.3, value=_mach_cruise, upper=2.7
)
for func in [ksfailure_cruise, wing_mass, cl_cruise, cd_cruise]:
    func.register_to(cruise)
if optimize_trim:
    moment.register_to(cruise)
cruise.register_to(hsct_model)

# COMPOSITE FUNCTIONS
# -----------------------------------------

# TOGW
g = 9.81  # m/s^2
lb_to_N = 4.448  # N/lb
tsfc = 3.9e-5  # kg/N/s, Rolls Royce Olympus 593 engine
fuselage_tail_weight = 5.8e5  # N
fuel_reserve_fraction = 0.06
num_passengers = 250
passenger_weight = 230 * num_passengers * lb_to_N
crew_weight = (450 + 5 * num_passengers) * lb_to_N
# descent_fuel = 6000 * lb_to_N  # N, fixed based on NASA report

wing_weight = 2 * wing_mass * g  # m/s^2 => N, doubled for sym
empty_weight = wing_weight + fuselage_tail_weight
boarded_weight = empty_weight + passenger_weight + crew_weight

cruise_lift = 2 * cl_cruise * _qinf_cruise
cruise_drag = 2 * cd_cruise * _qinf_cruise

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
    scale=1.0e-5, objective=True, plot=True, plot_name="togw"
).register_to(hsct_model)

# steady flight and climb conditions

cruise_weight = togw * 0.5 * (1 + cruise_weight_ratio)
steady_cruise = cruise_lift / cruise_weight
steady_cruise.set_name("steady_cruise").optimize(
    lower=1.0,
    upper=1.0,
    scale=1.0,
    objective=False,
    plot=True,
    plot_name="steady-cruise",
).register_to(hsct_model)
climb_lift = 2 * cl_climb * _climb_qinf
steady_climb = climb_lift / togw
steady_climb.set_name("steady_climb").optimize(
    lower=1.5,
    upper=2.5,
    scale=1.0e-5,
    objective=False,
    plot=True,
    plot_name="steady-climb",
).register_to(hsct_model)

# adjacency skin thickness constraints (for structures discipline)
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

# BUILD THE DRIVER WHICH AUTO RUNS THE ANALYSIS
# ---------------------------------------------

# build the solvers and coupled driver
solvers = SolverManager(comm)
solvers.flow = Fun3dInterface(
    comm,
    hsct_model,
    fun3d_dir=fun3d_dir,
    auto_coords=False,
    forward_tolerance=1e-7,
    adjoint_tolerance=1e-7,
)

# select transfer settings in this case using MELD for steady-state aeroelastic transfer
# between structural and aerodynamic meshes
transfer_settings = TransferSettings(elastic_scheme="meld", npts=200, beta=0.5, isym=1)

f2f_driver = FuntofemShapeDriver.analysis(
    solvers,
    hsct_model,
    transfer_settings=transfer_settings,
    struct_nprocs=20,
    auto_run=True,
)
