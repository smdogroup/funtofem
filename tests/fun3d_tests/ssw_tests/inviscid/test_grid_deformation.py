"""
Test the vector function xG(uA) and its associated adjoints, namely the grid deformation of volume nodes x_G
from the aero disps on a converged flow of the SSW inviscid mesh.
"""

from funtofem import *
from mpi4py import MPI
import os, time

# import the base test interface
from ._base_test import *

# ---------------------------------------------------->

# DISCIPLINE INTERFACES AND DRIVERS
# <----------------------------------------------------

solvers.flow = Fun3d14GridInterface(
    comm,
    f2f_model,
    fun3d_dir="cfd",
)

max_rel_error = Fun3d14GridInterface.finite_diff_test_flow_states(
    solvers.flow, 
    epsilon=1e-4,
    filename="results/grid_deformation.txt"
)
if comm.rank == 0:
    print(f"max rel error = {max_rel_error}")