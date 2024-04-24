import unittest, os, numpy as np, sys
from _bdf_test_utils import elasticity_callback, thermoelasticity_callback
from mpi4py import MPI
from tacs import TACS

np.set_printoptions(threshold=sys.maxsize)

from funtofem import TransferScheme
from funtofem.model import FUNtoFEMmodel, Variable, Scenario, Body, Function
from funtofem.interface import (
    TestAerodynamicSolver,
    TacsInterface,
    SolverManager,
    TestResult,
    make_test_directories,
)
from funtofem.driver import FUNtoFEMnlbgs, TransferSettings, OnewayStructDriver

np.random.seed(1234567)

base_dir = os.path.dirname(os.path.abspath(__file__))
bdf_filename = os.path.join(base_dir, "input_files", "test_bdf_file.bdf")
comm = MPI.COMM_WORLD

complex_mode = TransferScheme.dtype == complex and TACS.dtype == complex

results_folder, output_dir = make_test_directories(comm, base_dir)
aero_loads_file = os.path.join(output_dir, "aero_loads.txt")
struct_loads_file = os.path.join(output_dir, "struct_loads.txt")


class TestStructLoadsFile(unittest.TestCase):
    N_PROCS = 2
    FILENAME = "test_struct_loads_file.txt"
    FILEPATH = os.path.join(results_folder, FILENAME)

    def test_struct_loads_file_aerothermoelastic(self):
        # ---------------------------
        # Write the loads file
        # ---------------------------
        # build the model and driver
        f2f_model = FUNtoFEMmodel("wedge")
        plate = Body.aerothermoelastic("plate", boundary=1)
        Variable.structural("thickness").set_bounds(
            lower=0.01, value=0.1, upper=1.0
        ).register_to(plate)
        plate.register_to(f2f_model)

        # build the scenario
        scenario = Scenario.steady("test", steps=150)
        Function.ksfailure().register_to(scenario)
        scenario.register_to(f2f_model)

        # make the solvers for a CFD analysis to store and write the loads file
        solvers = SolverManager(comm)
        solvers.flow = TestAerodynamicSolver(comm, f2f_model)
        solvers.structural = TacsInterface.create_from_bdf(
            f2f_model,
            comm,
            1,
            bdf_filename,
            callback=thermoelasticity_callback,
            output_dir=output_dir,
        )
        transfer_settings = TransferSettings(npts=5)
        FUNtoFEMnlbgs(
            solvers, transfer_settings=transfer_settings, model=f2f_model
        ).solve_forward()
        f2f_model.write_struct_loads(comm, struct_loads_file, root=0)
        return


if __name__ == "__main__":
    if comm.rank == 0:
        open(TestStructLoadsFile.FILEPATH, "w").close()
    unittest.main()
