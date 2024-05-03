"""
Test the vector function xG(uA) and its associated adjoints, namely the grid deformation of volume nodes x_G
from the aero disps on a converged flow of the SSW inviscid mesh.
NOTE : should probably just test this on one proc (since don't need converged flow, just mesh def for this.)
"""

from funtofem import *
from mpi4py import MPI
import os, time

# import the base test interface
from _base_test import *

# ---------------------------------------------------->

cruise.name = "cruise_inviscid_grid"
cruise.uncoupled_steps = 0
cruise.steps = 10
cruise.forward_coupling_frequency = 1

# don't want to use aerodynamic functions in this test since it adds extra function terms to the RHS, when only the aero loads
# not the function values itself are used.
cruise.functions = []
Function.ksfailure().register_to(cruise)

# DISCIPLINE INTERFACES AND DRIVERS
# <----------------------------------------------------

solvers.flow = Fun3d14GridInterface(
    comm,
    f2f_model,
    fun3d_dir="cfd",
)

max_rel_error = Fun3d14GridInterface.finite_diff_test(
    solvers.flow,
    epsilon=1e-4,
    filename="results/grid_deformation.txt",
    all_ones_forward=False,
    all_ones_adjoint=False,
    adjoint_scale=1e0,
    ua0_scale=1e-3,
    forward_scale=0,
)  # adjoint_scale = 1e3 works, 1e-3 bad error originally
if comm.rank == 0:
    print(f"max rel error = {max_rel_error}")
