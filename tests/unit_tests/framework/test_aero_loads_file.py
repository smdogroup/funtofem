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


class TestAeroLoadsFile(unittest.TestCase):
    N_PROCS = 2
    FILENAME = "test_aero_loads_file.txt"
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
            callback=elasticity_callback,
            output_dir=output_dir,
        )
        transfer_settings = TransferSettings(npts=5)
        FUNtoFEMnlbgs(
            solvers, transfer_settings=transfer_settings, model=f2f_model
        ).solve_forward()

        # save initial loads in an array
        orig_aero_loads = plate.aero_loads[scenario.id] * 1.0
        f2f_model.write_aero_loads(comm, aero_loads_file, root=0)

        # zero the aero loads
        plate.aero_loads[scenario.id] *= 0.0

        # -----------------------------------------------
        # Read the loads file and test the oneway driver
        # -----------------------------------------------
        solvers.flow = None
        OnewayStructDriver.prime_loads_from_file(
            aero_loads_file, solvers, f2f_model, 1, transfer_settings
        )

        # verify the aero loads are the same
        new_aero_loads = plate.aero_loads[scenario.id]
        diff_aero_loads = new_aero_loads - orig_aero_loads
        orig_norm = np.max(np.abs(orig_aero_loads))
        abs_err_norm = np.max(np.abs(diff_aero_loads))
        rel_err_norm = abs_err_norm / orig_norm

        print("aeroelastic aero loads test:")
        print(f"\taero load norm = {orig_norm:.5f}")
        print(f"\tabs error = {abs_err_norm:.5f}")
        print(f"\trel error = {rel_err_norm:.5f}")
        rtol = 1e-7
        self.assertTrue(rel_err_norm < rtol)
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

        # save initial loads in an array
        orig_aero_loads = plate.aero_loads[scenario.id] * 1.0
        orig_aero_hflux = plate.aero_heat_flux[scenario.id] * 1.0
        f2f_model.write_aero_loads(comm, aero_loads_file, root=0)

        # zero the aero loads
        plate.aero_loads[scenario.id] *= 0.0

        # -----------------------------------------------
        # Read the loads file and test the oneway driver
        # -----------------------------------------------
        solvers.flow = None
        OnewayStructDriver.prime_loads_from_file(
            aero_loads_file, solvers, f2f_model, 1, transfer_settings
        )

        # verify the aero loads are the same
        new_aero_loads = plate.aero_loads[scenario.id]
        diff_aero_loads = new_aero_loads - orig_aero_loads
        abs_err_norm = np.max(np.abs(diff_aero_loads))
        rel_err_norm = abs_err_norm / np.max(np.abs(orig_aero_loads))
        print("aero loads + aero hflux test:")
        print("aero loads comparison")
        print(f"\tabs error = {abs_err_norm:.5f}")
        print(f"\trel error = {rel_err_norm:.5f}")

        # verify the aero heat flux is the same
        new_aero_hflux = plate.aero_heat_flux[scenario.id]
        diff_aero_hflux = new_aero_hflux - orig_aero_hflux
        abs_err_norm2 = np.max(np.abs(diff_aero_hflux))
        rel_err_norm2 = abs_err_norm / np.max(np.abs(orig_aero_hflux))
        print("aero hflux comparison")
        print(f"\tabs error = {abs_err_norm2:.5f}")
        print(f"\trel error = {rel_err_norm2:.5f}")
        rtol = 1e-7
        self.assertTrue(rel_err_norm < rtol)
        self.assertTrue(rel_err_norm2 < rtol)
        return


if __name__ == "__main__":
    if comm.rank == 0:
        open(TestAeroLoadsFile.FILEPATH, "w").close()
    unittest.main()
