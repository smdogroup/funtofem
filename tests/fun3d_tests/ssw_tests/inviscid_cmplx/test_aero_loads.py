"""
Test the vector function fA(uA) and its associated adjoints, namely the aero loads
from the aero disps on a converged flow of the SSW inviscid mesh.
"""

from funtofem import *
from mpi4py import MPI
import os, time

# import the base test interface
from _base_test import *

# ---------------------------------------------------->

# DISCIPLINE INTERFACES AND DRIVERS
# <----------------------------------------------------

solvers.flow = Fun3d14AeroelasticTestInterface(
    comm,
    f2f_model,
    fun3d_dir="cfd",
)

# don't want to use aerodynamic functions in this test since it adds extra function terms to the RHS, when only the aero loads
# not the function values itself are used.
cruise.functions = []
Function.ksfailure().register_to(cruise)

max_rel_error = Fun3d14AeroelasticTestInterface.complex_step_test_aero_loads(
    solvers.flow, epsilon=1e-30, filename="results/cmplx_aero_loads.txt"
)
if comm.rank == 0:
    print(f"max rel error = {max_rel_error}")
