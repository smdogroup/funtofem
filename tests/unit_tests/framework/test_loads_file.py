import unittest, os, numpy as np
from bdf_test_utils import elasticity_callback
from mpi4py import MPI

from funtofem import TransferScheme
from pyfuntofem.model import FUNtoFEMmodel, Variable, Scenario, Body, Function
from pyfuntofem.interface import (
    TestAerodynamicSolver,
    TacsSteadyInterface,
    SolverManager,
    TestResult,
)
from pyfuntofem.driver import FUNtoFEMnlbgs, TransferSettings, TacsOnewayDriver

np.random.seed(1234567)

base_dir = os.path.dirname(os.path.abspath(__file__))
bdf_filename = os.path.join(base_dir, "input_files", "test_bdf_file.bdf")
comm = MPI.COMM_WORLD

class TestLoadsFile(unittest.TestCase):
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
        scenario = Scenario.steady("test", steps=200).include(Function.ksfailure())
        scenario.register_to(f2f_model)

        # make the solvers for a CFD analysis to store and write the loads file
        solvers = SolverManager(comm)
        solvers.flow = TestAerodynamicSolver(comm, f2f_model)
        solvers.structural = TacsSteadyInterface.create_from_bdf(
            f2f_model, comm, 1, bdf_filename, callback=elasticity_callback
        )
        transfer_settings = TransferSettings(npts=5)
        FUNtoFEMnlbgs(
            solvers, transfer_settings=transfer_settings, model=f2f_model
        ).solve_forward()
        f2f_model.write_aero_loads(comm, "aero_loads.txt", root=0)

        # -----------------------------------------------
        # Read the loads file and test the oneway driver
        # -----------------------------------------------
        # build the model and driver
        f2f_model = FUNtoFEMmodel("wedge")
        plate = Body.aeroelastic("plate", boundary=1)
        Variable.structural("thickness").set_bounds(
            lower=0.01, value=0.1, upper=1.0
        ).register_to(plate)
        plate.register_to(f2f_model)

        # build the scenario
        scenario = Scenario.steady("test", steps=200).include(Function.ksfailure())
        scenario.register_to(f2f_model)

        # make the solvers for a CFD analysis to store and write the loads file
        solvers = SolverManager(comm)
        solvers.structural = TacsSteadyInterface.create_from_bdf(
            f2f_model, comm, 1, bdf_filename, callback=elasticity_callback
        )
        transfer_settings = TransferSettings(npts=5, beta=0.5)
        oneway_driver = TacsOnewayDriver.prime_loads_from_file("aero_loads.txt", solvers, f2f_model, 1, transfer_settings, tacs_aim=None)
        print(f"aero loads = {plate.aero_loads[1]}")
        print(f"aero X = {plate.aero_X}")
        print(f"struct X = {plate.struct_X}")
        print(f"struct disps = {plate.struct_disps[1]}")
        print(f"struct loads = {plate.struct_loads[1]}")
        return
    
if __name__ == "__main__":
    unittest.main()