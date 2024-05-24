"""
Test aero + struct functions and gradients for
case 1 - examples/fun3d_examples/_ssw-inviscid/1_panel_thickness.py
"""

from funtofem import *
from mpi4py import MPI
import os, time

# import from the base test
from _base_test import *

# ---------------------------------------------------->

# DISCIPLINE INTERFACES AND DRIVERS
# <----------------------------------------------------

solvers.flow = Fun3d14Interface(
    comm,
    f2f_model,
    fun3d_dir="cfd",
    forward_stop_tolerance=1e-15,
    forward_min_tolerance=1e-10,
    adjoint_stop_tolerance=1e-13,
    adjoint_min_tolerance=1e-8,
    debug=global_debug_flag,
)
solvers.flow = Fun3d14Interface.copy_complex_interface(solvers.flow)
transfer_settings = TransferSettings(npts=200)

# Build the FUNtoFEM driver
f2f_driver = FUNtoFEMnlbgs(
    solvers=solvers,
    transfer_settings=transfer_settings,
    model=f2f_model,
    debug=global_debug_flag,
    reload_funtofem_states=False,
)

start_time = time.time()

# run the finite difference test
max_rel_error = TestResult.derivative_test(
    "fun3d+tacs-ssw1",
    model=f2f_model,
    driver=f2f_driver,
    status_file="results/cmplx_1-derivs.txt",
    complex_mode=False,
    epsilon=1e-4,
)

end_time = time.time()
dt = end_time - start_time
if comm.rank == 0:
    print(f"total time for ssw derivative test is {dt} seconds", flush=True)
    print(f"max rel error = {max_rel_error}", flush=True)
