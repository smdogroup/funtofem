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
wing = Body.aeroelastic("wing", boundary=3)
wing.register_to(f2f_model)

# Make a FUNtoFEM scenario
cruise = Scenario.steady("cruise", steps=1000)
Function.lift().register_to(cruise)
Function.drag().register_to(cruise)
cruise.set_temperature(T_ref=T_inf, T_inf=T_inf)
cruise.set_flow_ref_vals(qinf=q_inf)
cruise.register_to(f2f_model)


# DISCIPLINE INTERFACES AND DRIVERS
# -----------------------------------------------------

solvers = SolverManager(comm)
solvers.flow = Fun3dInterface(
    comm, f2f_model, fun3d_project_name="ssw-turb", fun3d_dir="cfd"
)
my_transfer_settings = TransferSettings(npts=200)
fun3d_driver = OnewayAeroDriver(
    solvers, f2f_model, transfer_settings=my_transfer_settings
)
fun3d_driver.solve_forward()

# write an aero loads file
aero_loads_file = os.path.join(os.getcwd(), "cfd", "loads", "uncoupled_loads.txt")
f2f_model.write_aero_loads(comm, aero_loads_file)
