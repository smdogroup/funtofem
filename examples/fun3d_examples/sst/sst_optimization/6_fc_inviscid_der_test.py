"""
Sean P. Engelstad, Georgia Tech 2023

This is the fully coupled aeroelastic, inviscid optimization of the HSCT.
NOTE: You need to run the 1_sizing_optimization.py and 2_sizing_shape.py or local version
first and leave the optimal panel thickness design variables in the meshes folder before running this.

We no longer perform the mesh generation in this script as then we can't use parallel tacsAIMs
NOTE : don't call this script with mpiexec_mpt, call it with python (otherwise system calls won't work)
"""
from pyoptsparse import SNOPT, Optimization
import os
import numpy as np
from mpi4py import MPI
from funtofem import *

# script inputs
hot_start = True
store_history = True
optimize_trim = False

comm = MPI.COMM_WORLD
base_dir = os.path.dirname(os.path.abspath(__file__))
fun3d_dir = os.path.join(base_dir, "cfd")
analysis_file = os.path.join(base_dir, "_run_inviscid_ae.py")

# FUNtoFEM and SHAPE MODELS
# ---------------------------------------------------------
hsct_model = FUNtoFEMmodel("sst-inviscid")

# BODIES, STRUCT DVs and SHAPE DVs
# ---------------------------------------------------
wing = Body.aeroelastic("wing", boundary=4)

nribs = 25
nspars = 3
nOML = nribs - 1
for irib in range(1, nribs + 1):
    Variable.structural(f"rib{irib}", value=0.1).set_bounds(
        lower=0.001, upper=0.15, scale=100.0
    ).register_to(wing)
for ispar in range(1, nspars + 1):
    Variable.structural(f"spar{ispar}", value=0.1).set_bounds(
        lower=0.001, upper=0.15, scale=100.0
    ).register_to(wing)
for iOML in range(1, nOML + 1):
    Variable.structural(f"OML{iOML}", value=0.1).set_bounds(
        lower=0.001, upper=0.15, scale=100.0
    ).register_to(wing)
for name in ["LEspar", "TEspar"]:
    Variable.structural(name, value=0.1).set_bounds(
        lower=0.001, upper=0.15, scale=100.0
    ).register_to(wing)

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

coffset1 = Variable.shape("wing:cbar_offset1", value=0.0).set_bounds(
    lower=-0.1, upper=0.1
)
coffset2 = Variable.shape("wing:cbar_offset2", value=0.0).set_bounds(
    lower=-0.1, upper=0.1
)
coffset3 = Variable.shape("wing:cbar_offset3", value=0.0).set_bounds(
    lower=-0.1, upper=0.1
)
coffsets = [coffset1, coffset2, coffset3]

dzhat1 = Variable.shape("wing:dzhat1", value=0.05).set_bounds(lower=0.0, upper=0.1)
dzhat2 = Variable.shape("wing:dzhat2", value=0.2).set_bounds(lower=0.1, upper=0.3)
dz_dihedral = Variable.shape("wing:dz_dihedral", value=-5.0).set_bounds(
    lower=-7.0, upper=-2.0
)
dzs = [dzhat1, dzhat2, dz_dihedral]

# geometric AOA on wing
# geomAOA1 = Variable.shape("wing:geomAOA1", value=0.0).set_bounds(lower=-0.5, upper=0.5)
# geomAOA2 = Variable.shape("wing:geomAOA2", value=0.0).set_bounds(lower=-2.0, upper=2.0)
# geomAOA3 = Variable.shape("wing:geomAOA3", value=0.0).set_bounds(lower=-2.0, upper=2.0)
# geomAOA4 = Variable.shape("wing:geomAOA4", value=0.0).set_bounds(lower=-2.0, upper=2.0)

for var in cbars + coffsets + dzs:
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

climb = Scenario.steady("climb_inviscid", steps=350, uncoupled_steps=200)
climb.set_temperature(T_ref=300.0, T_inf=300.0)  # modify this
climb.set_flow_ref_vals(qinf=_climb_qinf, flow_dt=1.0)
ksfailure_climb = Function.ksfailure(ks_weight=10.0, safety_factor=1.5).optimize(
    scale=1.0,
    upper=1.0,
    objective=False,
    plot=True,
    plot_name="ks-climb",
)
cl_climb = Function.lift(body=0)
cd_climb = Function.drag(body=0)
aoa_climb = climb.get_variable("AOA").set_bounds(lower=0.0, value=4.0, upper=8.0)
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

cruise = Scenario.steady("cruise_inviscid", steps=500)
cruise.set_temperature(T_ref=_Tinf_cruise, T_inf=_Tinf_cruise)
cruise.set_flow_ref_vals(qinf=_qinf_cruise, flow_dt=1.0)
ksfailure_cruise = Function.ksfailure(ks_weight=10.0, safety_factor=1.5).optimize(
    scale=1.0,
    upper=1.0,
    objective=False,
    plot=True,
    plot_name="ks-cruise",
)
cl_cruise = Function.lift(body=0)
cd_cruise = Function.drag(body=0)
if optimize_trim:
    moment = Function.moment(body=0).optimize(
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

# steady climb, prob should add some thrust contribution here
climb_factor = 2.0  # was 2.5 before
climb_lift = 2 * cl_climb * _climb_qinf
steady_climb = climb_lift - climb_factor * togw
steady_climb.set_name("steady_climb").optimize(
    lower=0.0,
    upper=0.0,
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

# BUILD THE DRIVER. NO DISCIPLINE INTERFACES IN DRIVER SCRIPT
# ------------------------------------------------------

# load in the previous design from the sizing optimization
#   to overwrite the initial values
# change this to "internal-struct.txt" and add internal struct
#   shape variables in once that is ready
design_in_file = os.path.join(base_dir, "design", "internal-struct.txt")  # sizing.txt
hsct_model.read_design_variables_file(comm, design_in_file)

# change structural design var lb and ubs to be closer to sizing design
# want to use improved internal struct design after this
margin = 0.3
for var in hsct_model.get_variables():
    if var.analysis_type == "structural":
        var.lower = np.max([(1 - margin) * var.value, var.lower])
        var.upper = np.min([(1 + margin) * var.value, var.upper])


# build the solvers and coupled driver
solvers = SolverManager(comm)
remote = Remote(comm, analysis_file, fun3d_dir, nprocs=192)
f2f_driver = FuntofemShapeDriver.aero_remesh(
    solvers, hsct_model, remote, forward_flow_post_analysis=True
)

# test the derivatives of analysis functions using FD
h = 1e-2
orig_values = [var.value for var in hsct_model.get_variables()]
pert_values = [
    np.random.rand() if var.analysis_type == "shape" else 0.0
    for var in hsct_model.get_variables()
]

f2f_driver.solve_forward()
f2f_driver.solve_adjoint()

m_funcs = [func.value.real for func in hsct_model.get_functions()]
adj_grads = [
    [func.derivatives[var] for var in hsct_model.get_variables()]
    for func in hsct_model.get_functions()
]
adj_dderivs = []
for ifunc, func in enumerate(hsct_model.get_functions()):
    adj_dderiv = 0.0
    for ivar, var in enumerate(hsct_model.get_variables()):
        adj_dderiv += adj_grads[ifunc][ivar] * pert_values[ivar]
    adj_dderivs += [adj_dderiv]

# run f(x-h) analysis
for ivar, var in enumerate(hsct_model.get_variables()):
    var.value -= pert_values[ivar] * h

f2f_driver.solve_forward()
i_funcs = [func.value.real for func in hsct_model.get_functions()]

# run f(x+h) analysis
for ivar, var in enumerate(hsct_model.get_variables()):
    var.value += 2 * pert_values[ivar] * h

f2f_driver.solve_forward()
f_funcs = [func.value.real for func in hsct_model.get_functions()]

# get the FD gradients
fd_derivs = [(f_funcs[ifunc] - i_funcs[ifunc]) / 2 / h for ifunc in range(len(i_funcs))]

# compare the gradients
rel_errors = [
    (fd_derivs[ifunc] - adj_dderivs[ifunc]) / fd_derivs[ifunc]
    for ifunc in range(len(fd_derivs))
]

if comm.rank == 0:
    print(f"Results of finite difference test:\n\n")
    for ifunc, func in enumerate(hsct_model.get_functions()):
        print(f"\n\nfunc {func.full_name}")
        print(f"\tx-h val = {i_funcs[ifunc]}")
        print(f"\tx val = {m_funcs[ifunc]}")
        print(f"\tx+h val = {f_funcs[ifunc]}")
        print(f"\tadj dderiv = {adj_dderivs[ifunc]}")
        print(f"\tfd dderiv = {fd_derivs[ifunc]}")
        print(f"\trel error = {rel_errors[ifunc]}")
    for ifunc, func in enumerate(hsct_model.get_functions()):
        for ivar, var in enumerate(hsct_model.get_variables()):
            print(
                f"\t\tadj d{func.full_name}/d{var.full_name} = {adj_grads[ifunc][ivar]}"
            )
