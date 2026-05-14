"""
wedge_optimization.py

Steady aerothermoelastic optimization of a supersonic diamond wedge panel.
Minimizes average structural temperature subject to a mass constraint,
using a fully coupled FUN3D + TACS analysis.

Run with:
    mpiexec -n <nprocs> python wedge_optimization.py

Prerequisites:
    1. Generate the TACS BDF mesh:   python gen_TACS_bdf_aero.py
       (writes tacs_aero.bdf into struct/)
    2. Place FUN3D mesh and input files in cfd/
"""

from pyoptsparse import SNOPT, Optimization
from funtofem import *
from mpi4py import MPI
import os

comm = MPI.COMM_WORLD
base_dir = os.path.dirname(os.path.abspath(__file__))

# Optimization flags
hot_start = False
store_history = True

# Freestream / reference quantities
v_inf = 1962.44 / 6.6 * 0.5  # ~148.67 m/s, Mach 0.5
rho = 0.01037  # kg/m^3
q_inf = 0.5 * rho * v_inf**2  # dynamic pressure, N/m^2
thermal_scale = 0.5 * rho * v_inf**3  # heat flux * area, J/s
T_ref = 300.0  # reference / freestream temperature, K

maximum_mass = 40.0  # mass constraint upper bound, kg

# FUNTOFEM MODEL
# -------------------------------------------------------

f2f_model = FUNtoFEMmodel("wedge")

plate = Body.aerothermoelastic("plate", boundary=1)

# 10 panel thickness design variables
for i in range(10):
    Variable.structural(f"thickness {i}", value=0.5).set_bounds(
        lower=1e-5, upper=1e4
    ).register_to(plate)

plate.register_to(f2f_model)

# SCENARIO
# -------------------------------------------------------

steady = Scenario.steady("steady", steps=100)
steady.set_temperature(T_ref=T_ref, T_inf=T_ref)
steady.set_flow_ref_vals(qinf=q_inf)

Function.temperature().optimize(
    scale=1.0, objective=True, plot=True, plot_name="temperature"
).register_to(steady)

Function.mass().optimize(
    scale=1.0, upper=maximum_mass, objective=False, plot=True, plot_name="mass"
).register_to(steady)

steady.register_to(f2f_model)

# DISCIPLINE INTERFACES AND DRIVER
# -------------------------------------------------------

bdf_file = os.path.join(base_dir, "struct", "tacs_aero.bdf")

solvers = SolverManager(comm)
solvers.flow = Fun3dInterface(
    comm,
    f2f_model,
    fun3d_dir="cfd",
    forward_tolerance=1e-6,
    adjoint_tolerance=1e-6,
)
solvers.structural = TacsSteadyInterface.create_from_bdf(
    model=f2f_model,
    comm=comm,
    nprocs=1,
    bdf_file=bdf_file,
    prefix=os.path.join(base_dir, "struct"),
)

transfer_settings = TransferSettings(
    elastic_scheme="meld",
    thermal_scheme="meld",
    npts=10,
    beta=10.0,
    isym=-1,
)

f2f_driver = FUNtoFEMnlbgs(
    solvers=solvers,
    transfer_settings=transfer_settings,
    model=f2f_model,
)

# PYOPTSPARSE OPTIMIZATION
# -------------------------------------------------------

design_out_file = os.path.join(base_dir, "design", "wedge.txt")

design_folder = os.path.join(base_dir, "design")
if comm.rank == 0 and not os.path.exists(design_folder):
    os.mkdir(design_folder)

history_file = os.path.join(design_folder, "wedge.hst")
store_history_file = history_file if store_history else None
hot_start_file = history_file if hot_start else None

manager = OptimizationManager(
    f2f_driver,
    design_out_file=design_out_file,
    hot_start=hot_start,
    hot_start_file=hot_start_file,
)

opt_problem = Optimization("wedgeOpt", manager.eval_functions)
manager.register_to_problem(opt_problem)

snoptimizer = SNOPT(options={"Major Optimality tol": 1e-6, "IPRINT": 1})

sol = snoptimizer(
    opt_problem,
    sens=manager.eval_gradients,
    storeHistory=store_history_file,
    hotStart=hot_start_file,
)

if comm.rank == 0:
    print(f"Final solution = {sol.xStar}", flush=True)
