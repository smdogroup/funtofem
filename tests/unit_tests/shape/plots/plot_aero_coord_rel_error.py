import os, re, sys
from tacs import TACS
from mpi4py import MPI
import numpy as np, matplotlib.pyplot as plt
from funtofem import TransferScheme

from funtofem.model import FUNtoFEMmodel, Scenario, Body, Function
from funtofem.interface import (
    TestAerodynamicSolver,
    TacsInterface,
    SolverManager,
    TacsIntegrationSettings,
    CoordinateDerivativeTester,
)
from funtofem.driver import TransferSettings, FUNtoFEMnlbgs

sys.path.append("../")

from _bdf_test_utils import elasticity_callback, thermoelasticity_callback
import unittest

np.random.seed(123456)

base_dir = os.path.dirname(os.path.abspath(__file__))
bdf_filename = os.path.join(base_dir, "..", "input_files", "test_bdf_file.bdf")


complex_mode = TransferScheme.dtype == complex and TACS.dtype == complex

# build the model and driver
model = FUNtoFEMmodel("wedge")
plate = Body.aeroelastic("plate", boundary=1)
plate.register_to(model)

# build the scenario
scenario = Scenario.steady("test", steps=200).include(Function.ksfailure())
scenario.include(Function.drag()).include(Function.lift())
scenario.register_to(model)

# build the tacs interface, coupled driver, and oneway driver
comm = MPI.COMM_WORLD
solvers = SolverManager(comm)
solvers.flow = TestAerodynamicSolver(comm, model)
solvers.structural = TacsInterface.create_from_bdf(
    model, comm, 1, bdf_filename, callback=elasticity_callback
)
transfer_settings = TransferSettings(npts=5)
coupled_driver = FUNtoFEMnlbgs(
    solvers, transfer_settings=transfer_settings, model=model
)
p = np.random.rand(plate.aero_X.shape[0])
x_A0 = 1.0 * plate.aero_X

func_names = [func.full_name for func in model.get_functions()]
nf = len(func_names)


def aero_shape_func(x):
    """f_S(x_A0 + p*x)"""
    plate.aero_X = x_A0 + p * x
    coupled_driver.solve_forward()

    return np.array([func.value for func in model.get_functions()])


def adjoints():
    plate.aero_X = x_A0
    coupled_driver.solve_forward()
    coupled_driver.solve_adjoint()
    dfdxA0 = plate.get_aero_coordinate_derivatives(scenario)
    p_row_vec = np.expand_dims(p, axis=0)
    dfdx_adjoint = p_row_vec @ dfdxA0
    dfdx_adjoint = list(np.reshape(dfdx_adjoint, newshape=(nf)))
    return [dfdx_adjoint[i].real for i in range(nf)]


# get third derivatives
h1 = 1e-4
third_derivs = (
    0.5 * aero_shape_func(2 * h1)
    - aero_shape_func(h1)
    + aero_shape_func(-h1)
    - 0.5 * aero_shape_func(-2 * h1)
) / h1**3

h = 1e-2
# https://en.wikipedia.org/wiki/Finite_difference_coefficient
first_derivs2 = (aero_shape_func(h) - aero_shape_func(-h)) / 2 / h
first_derivs4 = (
    -aero_shape_func(2 * h) / 12
    + 2.0 / 3 * aero_shape_func(h)
    - 2.0 / 3 * aero_shape_func(-h)
    + 1.0 / 12 * aero_shape_func(-2 * h)
) / h

adjoints_vec = adjoints()
rel_errors2 = (first_derivs2 - adjoints_vec) / adjoints_vec
rel_errors4 = (first_derivs4 - adjoints_vec) / adjoints_vec

# check floating point error
function_vec = aero_shape_func(0.0)
df_vec = h * first_derivs2
df_ratio_vec = df_vec / function_vec

print(f"functions = {func_names}")
print(f"\tthird derivatives = {third_derivs}")
print(f"\tfirst derivs - 2nd order = {first_derivs2}")
print(f"\tfirst derivs - 4th order = {first_derivs4}")
print(f"\tadjoints = {adjoints_vec}")
print(f"\trel errors - 2nd order = {rel_errors2}")
print(f"\trel errors - 4th order = {rel_errors4}")
print(f"\tfunction vals = {function_vec}")
print(f"\tdf for fp error = {df_vec}")
print(f"\tdf/f for fp error = {df_ratio_vec}")


# plot struct functions, aero functions normalized on the plot to view their third order + nonlinear shapes
# xvec = np.linspace(-1e-4, 1e-4, 10)
# yvec = [aero_shape_func(x) for x in xvec]
# ksfailure_vec = np.array([_[0] for _ in yvec])
# lift_vec = np.array([_[2] for _ in yvec])
# ksfailure_vec -= np.mean(ksfailure_vec)
# ksfailure_vec /= np.max(ksfailure_vec)
# lift_vec -= np.mean(lift_vec)
# lift_vec /= np.max(lift_vec)

# plt.plot(xvec,ksfailure_vec, label="ksfailure")
# plt.plot(xvec, lift_vec, label="lift")
# plt.xlabel("aero shape variable h")
# plt.ylabel(r"[f(x_{A0}+p*h)-f(x_{A0})]/fmax")
# plt.legend()
# plt.savefig("functionals_nonlinearity.png")

# plot finite difference relative error convergence
exponents = np.linspace(0, -8, 50)
step_sizes = np.power(10, exponents)
first_derivs_step = [
    (aero_shape_func(h) - aero_shape_func(-h)) / 2 / h for h in step_sizes
]
rel_errors_FD = [
    np.abs((FDs - adjoints_vec) / adjoints_vec) for FDs in first_derivs_step
]
complex_step = [np.imag(aero_shape_func(h * 1j)) / h for h in step_sizes]
rel_errors_CS = [np.abs((CSs - adjoints_vec) / adjoints_vec) for CSs in complex_step]
step_sizes = list(step_sizes)

local_func_names = ["ksfailure", "lift", "drag"]
jet = plt.cm.jet
colors = jet(np.linspace(0, 1, 2 * len(local_func_names)))

fig = plt.figure()
ax = plt.subplot(111)
linewidth = 2

for ifunc, func_name in enumerate(local_func_names):
    rel_error_FD = [_[ifunc] for _ in rel_errors_FD]
    rel_error_CS = [_[ifunc] for _ in rel_errors_CS]

    if ifunc == 2:
        linestyle = "--"
    else:
        linestyle = "-"

    ax.plot(
        step_sizes,
        rel_error_FD,
        color=colors[2 * ifunc],
        label=f"{func_name}-FD",
        linestyle=linestyle,
        linewidth=2,
    )
    ax.plot(
        step_sizes,
        rel_error_CS,
        color=colors[2 * ifunc + 1],
        label=f"{func_name}-CS",
        linestyle=linestyle,
        linewidth=2,
    )

# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.xlabel("Step size")
plt.ylabel("Rel error")
plt.xscale("log")
plt.yscale("log")
plt.savefig("rel_errors_f2f-aero.png")
