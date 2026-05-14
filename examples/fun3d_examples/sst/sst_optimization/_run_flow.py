"""
Sean P. Engelstad, Georgia Tech 2023

Run a FUN3D analysis with the Fun3dOnewayDriver
"""

from funtofem import *
from mpi4py import MPI

comm = MPI.COMM_WORLD

f2f_model = FUNtoFEMmodel("hsct_flow")
wing = Body.aerothermoelastic("wing", boundary=4)
wing.register_to(f2f_model)

# make a funtofem scenario
cruise = Scenario.steady("cruise_inviscid", steps=500)  # 2000
cruise.set_stop_criterion(early_stopping=True)
Function.lift().register_to(cruise)
Function.drag().register_to(cruise)
cruise.set_temperature(T_ref=216, T_inf=216)
cruise.set_flow_ref_vals(qinf=3.16e4)
cruise.register_to(f2f_model)


# DISCIPLINE INTERFACES AND DRIVERS
# -----------------------------------------------------

solvers = SolverManager(comm)
solvers.flow = Fun3dInterface(
    comm, f2f_model, fun3d_dir="cfd", forward_tolerance=1e-6, adjoint_tolerance=1e-6
)
my_transfer_settings = TransferSettings(npts=200)
fun3d_driver = OnewayAeroDriver(
    solvers, f2f_model, transfer_settings=my_transfer_settings
)
fun3d_driver.solve_forward()
