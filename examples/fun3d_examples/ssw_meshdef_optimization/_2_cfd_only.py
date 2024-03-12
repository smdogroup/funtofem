"""
_2_cfd_only.py

Run an uncoupled analysis and adjoint of the angle of attack to minimize lift (L-L_*)^2.
No shape variables are included.
"""

from pyoptsparse import SLSQP, Optimization
from funtofem import *
from mpi4py import MPI
from tacs import caps2tacs
import os

comm = MPI.COMM_WORLD

base_dir = os.path.dirname(os.path.abspath(__file__))
csm_path = os.path.join(base_dir, "geometry", "ssw.csm")

# Optimization options
hot_start = False
store_history = True

# FUNTOFEM MODEL
# <----------------------------------------------------
# Freestream quantities -- see README
T_inf = 268.338  # Freestream temperature
q_inf = 1.21945e4  # Dynamic pressure

# Construct the FUNtoFEM model
f2f_model = FUNtoFEMmodel("ssw-aoa")

# ---------------------------------------------------->

# BODIES AND STRUCT DVs
# <----------------------------------------------------

wing = Body.aeroelastic("wing", boundary=2)

# register the wing body to the model
wing.register_to(f2f_model)

# ---------------------------------------------------->

# SCENARIOS
# <----------------------------------------------------

# make a funtofem scenario
cruise = Scenario.steady("cruise", steps=1500, uncoupled_steps=0)
cl_cruise = Function.lift(body=0)
aoa_cruise = cruise.get_variable("AOA").set_bounds(lower=-4, value=2.0, upper=15)
cruise.include(cl_cruise)
cruise.set_temperature(T_ref=T_inf, T_inf=T_inf)
cruise.set_flow_ref_vals(qinf=q_inf)
cruise.register_to(f2f_model)

# ---------------------------------------------------->

# COMPOSITE FUNCTIONS
# <----------------------------------------------------
cl_target = 1.2

cruise_lift = (cl_cruise - cl_target) ** 2
cruise_lift.set_name(f"LiftObj").optimize(
    lower=-1e-2, upper=10, scale=1.0, objective=True, plot=True, plot_name="Lift-Obj"
).register_to(f2f_model)

# ---------------------------------------------------->

# DISCIPLINE INTERFACES AND DRIVERS
# <----------------------------------------------------

solvers = SolverManager(comm)
solvers.flow = Fun3dInterface(
    comm,
    f2f_model,
    fun3d_project_name="ssw-pw1.2",
    fun3d_dir="cfd",
    forward_tolerance=1e-4,
    adjoint_tolerance=1e-4,
)

transfer_settings = TransferSettings(npts=200)

# Build the FUNtoFEM driver
f2f_driver = OnewayAeroDriver.analysis(
    solvers=solvers,
    model=f2f_model,
    transfer_settings=transfer_settings,
    is_paired=False,
)

# ---------------------------------------------------->
if comm.rank == 0:
    f2f_model.print_summary()

f2f_driver.solve_forward()
f2f_driver.solve_adjoint()

# ---------------------------------------------------->
