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
    test_directories,
)
from funtofem.driver import FUNtoFEMnlbgs, TransferSettings, TacsOnewayDriver

np.random.seed(1234567)

base_dir = os.path.dirname(os.path.abspath(__file__))
bdf_filename = os.path.join(base_dir, "input_files", "test_bdf_file.bdf")
comm = MPI.COMM_WORLD

complex_mode = TransferScheme.dtype == complex and TACS.dtype == complex

results_folder, output_dir = test_directories(comm, base_dir)
aero_loads_file = os.path.join(output_dir, "aero_loads.txt")
struct_loads_file = os.path.join(output_dir, "struct_loads.txt")


class TestLoadsFile(unittest.TestCase):
    # N_PROCS = 2
    FILENAME = "test_oneway_loads_file.txt"
    FILEPATH = os.path.join(results_folder, FILENAME)

    def test_loads_file_aeroelastic(self):
        # ---------------------------
        # Write the loads file
        # ---------------------------
        # build the model and driver
        f2f_model = FUNtoFEMmodel("wedge")
        plate = Body.aeroelastic("plate", boundary=1)
        Variable.structural("thickness").set_bounds(
            lower=0.01, value=0.1, upper=1.0
        ).register_to(plate)
        plate.register_to(f2f_model)

        # build the scenario
        scenario = Scenario.steady("test", steps=150).include(Function.ksfailure())
        scenario.register_to(f2f_model)

        # make the solvers for a CFD analysis to store and write the loads file
        solvers = SolverManager(comm)
        solvers.flow = TestAerodynamicSolver(comm, f2f_model)
        solvers.structural = TacsInterface.create_from_bdf(
            f2f_model,
            comm,
            1,
            bdf_filename,
            callback=elasticity_callback,
            output_dir=output_dir,
        )
        transfer_settings = TransferSettings(npts=5)
        FUNtoFEMnlbgs(
            solvers, transfer_settings=transfer_settings, model=f2f_model
        ).solve_forward()
        f2f_model.write_aero_loads(comm, aero_loads_file, root=0)

        # -----------------------------------------------
        # Read the loads file and test the oneway driver
        # -----------------------------------------------
        solvers.flow = None
        oneway_driver = TacsOnewayDriver.prime_loads_from_file(
            aero_loads_file, solvers, f2f_model, 1, transfer_settings
        )

        max_rel_error = TestResult.derivative_test(
            "testaero=>tacs_loads-aeroelastic",
            f2f_model,
            oneway_driver,
            TestLoadsFile.FILEPATH,
            complex_mode=complex_mode,
        )
        rtol = 1e-7 if complex_mode else 1e-3
        self.assertTrue(max_rel_error < rtol)
        return

    def test_loads_file_aerothermoelastic(self):
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
        scenario = Scenario.steady("test", steps=150).include(Function.ksfailure())
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
        f2f_model.write_aero_loads(comm, aero_loads_file, root=0)

        # -----------------------------------------------
        # Read the loads file and test the oneway driver
        # -----------------------------------------------
        solvers.flow = None
        oneway_driver = TacsOnewayDriver.prime_loads_from_file(
            aero_loads_file, solvers, f2f_model, 1, transfer_settings
        )

        max_rel_error = TestResult.derivative_test(
            "testaero=>tacs_loads-aerothermoelastic",
            f2f_model,
            oneway_driver,
            TestLoadsFile.FILEPATH,
            complex_mode=complex_mode,
        )
        rtol = 1e-7 if complex_mode else 1e-3
        self.assertTrue(max_rel_error < rtol)
        return

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
        scenario = Scenario.steady("test", steps=150).include(Function.ksfailure())
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
    unittest.main()
