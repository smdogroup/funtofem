"""
Test the vector function fA(xG, q) and its associated adjoints, internally using
a fun3d.nml flag that activates this test. Only run the forward analysis => since this test
can run internally on the fun3d_flow object only. NOTE : also need to run this test in serial as of right now. Don't need to run flow to completion.
"""

from funtofem import *
from mpi4py import MPI
import os, time
from _base_test import *

# SCENARIOS
#  <----------------------------------------------------

cruise.name = "cruise_inviscid_internal_cmplx"
cruise.uncoupled_steps = 0
cruise.steps = 3
cruise.forward_coupling_frequency = 1

# DISCIPLINE INTERFACES AND DRIVERS
# <----------------------------------------------------

solvers = SolverManager(comm)
solvers.structural = TacsSteadyInterface.create_from_bdf(
    model=f2f_model,
    comm=comm,
    nprocs=nprocs_tacs,
    bdf_file=tacs_aim.root_dat_file,
    prefix=tacs_aim.root_analysis_dir,
    debug=global_debug_flag,
)


# DISCIPLINE INTERFACES AND DRIVERS
# <----------------------------------------------------
solvers.flow = Fun3d14Interface(
    comm,
    f2f_model,
    fun3d_dir="cfd",
    forward_min_tolerance=1e10,
    debug=global_debug_flag,
)
solvers.flow = Fun3d14Interface.copy_complex_interface(solvers.flow)

# Build the FUNtoFEM driver
FUNtoFEMnlbgs(
    solvers=solvers,
    transfer_settings=TransferSettings(npts=200),
    model=f2f_model,
    debug=global_debug_flag,
    reload_funtofem_states=False,
).solve_forward()
