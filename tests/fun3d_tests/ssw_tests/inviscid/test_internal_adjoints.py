"""
Test the vector function fA(xG, q) and its associated adjoints, internally using 
a fun3d.nml flag that activates this test. Only run the forward analysis => since this test
can run internally on the fun3d_flow object only. NOTE : also need to run this test in serial as of right now. Don't need to run flow to completion.
"""

from funtofem import *
from mpi4py import MPI
import os, time

# import the base test interface
from ._base_test import *

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

transfer_settings = TransferSettings(npts=200)

# Build the FUNtoFEM driver
f2f_driver = FUNtoFEMnlbgs(
    solvers=solvers,
    transfer_settings=transfer_settings,
    model=f2f_model,
    debug=global_debug_flag,
    reload_funtofem_states=False,
)
f2f_driver.solve_forward()