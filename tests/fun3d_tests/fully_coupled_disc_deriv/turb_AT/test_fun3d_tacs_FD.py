"""
Unittest for FUN3D 14.0.2 finite-difference test
"""
import numpy as np, unittest, importlib, os
from mpi4py import MPI

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
    from funtofem.interface import Fun3d14Interface

np.random.seed(1234567)
comm = MPI.COMM_WORLD
base_dir = os.path.dirname(os.path.abspath(__file__))
results_folder, output_dir = make_test_directories(comm, base_dir)
bdf_filename = os.path.join(base_dir, "meshes", "nastran_CAPS.dat")


class TestFun3dTacs(unittest.TestCase):
    FILENAME = "fun3d-tacs-driver.txt"
    FILEPATH = os.path.join(results_folder, FILENAME)

    def test_alpha_turbulent_aerothermal(self):
        # build the funtofem model with one body and scenario
        model = FUNtoFEMmodel("plate")
        plate = Body.aerothermal("plate", boundary=2)
        plate.register_to(model)

        # build the scenario
        test_scenario = Scenario.steady("turbulent", steps=500, uncoupled_steps=10)
        test_scenario.set_temperature(T_ref=300.0, T_inf=300.0)
        Function.ksfailure(ks_weight=10.0).register_to(test_scenario)
        Function.temperature().register_to(test_scenario)
        Function.lift().register_to(test_scenario)
        Function.drag().register_to(test_scenario)
        aoa = test_scenario.get_variable("AOA", set_active=True)
        aoa.set_bounds(lower=5.0, value=10.0, upper=15.0)
        test_scenario.set_flow_ref_vals(qinf=1.05e5)
        test_scenario.register_to(model)

        # build the solvers and coupled driver
        solvers = SolverManager(comm)
        solvers.flow = Fun3d14Interface(comm, model, debug=False, fun3d_dir="meshes")

        solvers.structural = TacsSteadyInterface.create_from_bdf(
            model, comm, nprocs=1, bdf_file=bdf_filename, prefix=output_dir
        )

        transfer_settings = TransferSettings(
            elastic_scheme="meld",
            npts=50,
        )
        driver = FUNtoFEMnlbgs(
            solvers,
            transfer_settings=transfer_settings,
            model=model,
        )

        # run the complex step test on the model and driver
        max_rel_error = TestResult.finite_difference(
            "fun3d+tacs-turbulent-aerothermal-flow",
            model,
            driver,
            TestFun3dTacs.FILEPATH,
            epsilon=1e-4,
        )
        self.assertTrue(max_rel_error < 1e-7)

    def test_thick_turbulent_aerothermal(self):
        # build the funtofem model with one body and scenario
        model = FUNtoFEMmodel("plate")
        plate = Body.aerothermal("plate", boundary=2)
        Variable.structural("thick").set_bounds(
            lower=0.001, value=0.1, upper=2.0
        ).register_to(plate)
        plate.register_to(model)

        # build the scenario
        test_scenario = Scenario.steady("turbulent", steps=500, uncoupled_steps=10)
        test_scenario.set_temperature(T_ref=300.0, T_inf=300.0)
        Function.ksfailure(ks_weight=10.0).register_to(test_scenario)
        Function.temperature().register_to(test_scenario)
        Function.lift().register_to(test_scenario)
        Function.drag().register_to(test_scenario)
        test_scenario.set_flow_ref_vals(qinf=1.05e5)
        test_scenario.register_to(model)

        # build the solvers and coupled driver
        solvers = SolverManager(comm)
        solvers.flow = Fun3d14Interface(comm, model, debug=True, fun3d_dir="meshes")

        solvers.structural = TacsSteadyInterface.create_from_bdf(
            model, comm, nprocs=1, bdf_file=bdf_filename, prefix=output_dir
        )

        transfer_settings = TransferSettings(
            elastic_scheme="meld",
            npts=50,
        )
        driver = FUNtoFEMnlbgs(
            solvers,
            transfer_settings=transfer_settings,
            model=model,
        )

        # run the complex step test on the model and driver
        max_rel_error = TestResult.finite_difference(
            "fun3d+tacs-turbulent-aerothermal-structural",
            model,
            driver,
            TestFun3dTacs.FILEPATH,
            epsilon=1e-4,
        )
        self.assertTrue(max_rel_error < 1e-7)


if __name__ == "__main__":
    # open and close the file to reset it
    if comm.rank == 0:
        open(TestFun3dTacs.FILEPATH, "w").close()

    unittest.main()
