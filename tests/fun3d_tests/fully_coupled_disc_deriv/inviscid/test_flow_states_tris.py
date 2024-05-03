"""
Unittest for FUN3D 14.0.2 finite-difference test
"""

import numpy as np, unittest, importlib, os
from mpi4py import MPI
import time

# Imports from FUNtoFEM
from funtofem.model import (
    FUNtoFEMmodel,
    Variable,
    Scenario,
    Body,
    Function,
)
from funtofem.interface import (
    TacsSteadyInterface,
    SolverManager,
    TestResult,
    make_test_directories,
)
from funtofem.driver import FUNtoFEMnlbgs, TransferSettings

# check whether fun3d is available
fun3d_loader = importlib.util.find_spec("fun3d")
has_fun3d = fun3d_loader is not None
if has_fun3d:
    from funtofem.interface import Fun3d14AeroelasticTestInterface

np.random.seed(1234567)
comm = MPI.COMM_WORLD
base_dir = os.path.dirname(os.path.abspath(__file__))
bdf_filename = os.path.join(base_dir, "meshes", "nastran_CAPS.dat")
results_folder, output_dir = make_test_directories(comm, base_dir)

# TEST SETTINGS
# get more accurate derivatives when early stopping is off and fully converges
early_stopping = True
forward_tol = 1e-15
adjoint_tol = 1e-15


class TestFun3dTacs(unittest.TestCase):
    FILENAME = "flow_states_tris.txt"
    FILEPATH = os.path.join(results_folder, FILENAME)

    def test_alpha_turbulent_aeroelastic_quads(self):
        # build the funtofem model with one body and scenario
        model = FUNtoFEMmodel("plate")
        plate = Body.aeroelastic("plate", boundary=2)
        plate.register_to(model)

        # build the scenario
        test_scenario = Scenario.steady(
            "plate_flow_tris",
            steps=25,
            forward_coupling_frequency=20,  # 500 total fun3d steps
            adjoint_steps=25,
            adjoint_coupling_frequency=20,
            uncoupled_steps=10,
        )
        test_scenario.set_stop_criterion(
            early_stopping=early_stopping, min_forward_steps=50
        )
        test_scenario.set_temperature(T_ref=300.0, T_inf=300.0)
        #Function.lift().register_to(test_scenario)
        Function.ksfailure().register_to(test_scenario)
        aoa = test_scenario.get_variable("AOA", set_active=True)
        aoa.set_bounds(lower=5.0, value=10.0, upper=15.0)
        test_scenario.set_flow_ref_vals(qinf=1.05e5)
        test_scenario.register_to(model)

        # build the solvers and coupled driver
        solvers = SolverManager(comm)
        solvers.flow = Fun3d14AeroelasticTestInterface(comm, model, test_flow_states=True, fun3d_dir="meshes")
        solvers.structural = TacsSteadyInterface.create_from_bdf(
            model, comm, nprocs=1, bdf_file=bdf_filename, prefix=output_dir
        )

        max_rel_error = Fun3d14AeroelasticTestInterface.finite_diff_test_flow_states(
            solvers.flow, epsilon=1e-4, filename=self.FILEPATH
        )
        self.assertTrue(max_rel_error < 1e-7)


if __name__ == "__main__":
    # open and close the file to reset it
    if comm.rank == 0:
        open(TestFun3dTacs.FILEPATH, "w").close()

    unittest.main()
