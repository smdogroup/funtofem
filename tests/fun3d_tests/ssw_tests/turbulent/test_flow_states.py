"""
Test the vector function q(uA) and its associated adjoints, namely the flow states
from the aero disps on a converged flow of the SSW inviscid mesh.
"""

from funtofem import *
from mpi4py import MPI
import os, time

# import the base test interface
from _base_test import *

# ---------------------------------------------------->
cruise.name = "cruise_flow"
cruise.steps = 20
cruise.forward_coupling_frequency = 10
cruise.adjoint_coupling_frequency = 25
cruise.adjoint_steps = 20

# don't want to use aerodynamic functions in this test since it adds extra function terms to the RHS, when only the aero loads
# not the function values itself are used.
cruise.functions = []
ksfailure = Function.ksfailure().register_to(cruise)
# ksfailure.body = 0

# DISCIPLINE INTERFACES AND DRIVERS
# <----------------------------------------------------

solvers.flow = Fun3d14AeroelasticTestInterface(
    comm,
    f2f_model,
    fun3d_dir="cfd",
    test_flow_states=True,
)

max_rel_error = Fun3d14AeroelasticTestInterface.finite_diff_test_flow_states(
    solvers.flow, epsilon=1e-4, filename="results/flow_states.txt"
)
if comm.rank == 0:
    print(f"max rel error = {max_rel_error}")
