import os
from tacs import TACS
from mpi4py import MPI
import numpy as np
from funtofem import TransferScheme

from funtofem.model import FUNtoFEMmodel, Variable, Scenario, Body, Function
from funtofem.interface import (
    TestAerodynamicSolver,
    TacsInterface,
    SolverManager,
    TacsIntegrationSettings,
    CoordinateDerivativeTester,
)
from funtofem.driver import TacsOnewayDriver, TransferSettings, FUNtoFEMnlbgs

from bdf_test_utils import elasticity_callback, thermoelasticity_callback
import unittest
import matplotlib.pyplot as plt

np.random.seed(123456)

base_dir = os.path.dirname(os.path.abspath(__file__))
bdf_filename = os.path.join(base_dir, "input_files", "test_bdf_file.bdf")


complex_mode = TransferScheme.dtype == complex and TACS.dtype == complex
nprocs = 1
comm = MPI.COMM_WORLD

results_folder = os.path.join(base_dir, "results")
if comm.rank == 0:  # make the results folder if doesn't exist
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)

step_sizes = [1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
rel_errors = []

for step_size in step_sizes:
    # build the model and driver
    model = FUNtoFEMmodel("wedge")
    plate = Body.aeroelastic("plate", boundary=1)
    plate.register_to(model)

    # build the scenario
    scenario = Scenario.steady("test", steps=200).include(Function.lift())
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

    rtol = 1e-7

    """complex step test over coordinate derivatives"""
    tester = CoordinateDerivativeTester(coupled_driver, epsilon=step_size)
    status_file = "f2f_aero_coord_test.txt"
    rel_error = tester.test_aero_coordinates(
        "funtofem_driver aero coordinate derivatives steady-aeroelastic",
        status_file=status_file,
        complex_mode=False,
    )

    rel_errors.append(abs(rel_error))

    print(f"eps={step_size}, rel error = {abs(rel_error)}")

plt.plot(step_sizes, rel_errors)
plt.xscale("log")
plt.yscale("log")
plt.savefig("f2f_aero_coords_FD.png")
