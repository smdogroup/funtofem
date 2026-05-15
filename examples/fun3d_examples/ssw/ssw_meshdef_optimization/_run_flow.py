"""
_run_flow.py

Run a FUN3D analysis using the OnewayAeroDriver.
The flow solver is run first to generate aerodynamic loads on the structure which are
saved to uncoupled_loads.txt.
A FUNtoFEM model is created with an aeroelastic body which only iterates through TACS
to solve the structural sizing optimization problem.
"""

from funtofem import *
from mpi4py import MPI
import os

comm = MPI.COMM_WORLD

# Freestream quantities -- see README
T_inf = 268.338  # Freestream temperature
q_inf = 1.21945e4  # Dynamic pressure

# Construct the FUNtoFEM model
f2f_model = FUNtoFEMmodel("ssw_flow")
wing = Body.aeroelastic("wing", boundary=2)
wing.register_to(f2f_model)

# Make a FUNtoFEM scenario
cruise = Scenario.steady("cruise", steps=1000, uncoupled_steps=1000)
cruise.set_stop_criterion(early_stopping=True, min_forward_steps=50)
Function.lift().register_to(cruise)
Function.drag().register_to(cruise)
cruise.set_temperature(T_ref=T_inf, T_inf=T_inf)
cruise.set_flow_ref_vals(qinf=q_inf)
cruise.register_to(f2f_model)


# DISCIPLINE INTERFACES AND DRIVERS
# -----------------------------------------------------

solvers = SolverManager(comm)
solvers.flow = Fun3d14Interface(
    comm,
    f2f_model,
    fun3d_dir="cfd",
    forward_stop_tolerance=1e-10,
    forward_min_tolerance=1e-6,
)
my_transfer_settings = TransferSettings(npts=200)
fun3d_driver = OnewayAeroDriver(
    solvers, f2f_model, transfer_settings=my_transfer_settings
)
fun3d_driver.solve_forward()

# write an aero loads file
aero_loads_file = os.path.join(os.getcwd(), "cfd", "loads", "uncoupled_loads.txt")
f2f_model.write_aero_loads(comm, aero_loads_file)
