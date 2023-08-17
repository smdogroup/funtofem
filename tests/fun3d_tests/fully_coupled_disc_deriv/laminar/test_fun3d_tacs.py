import numpy as np, unittest, importlib, os
from mpi4py import MPI

# Imports from FUNtoFEM
from funtofem.model import (
    FUNtoFEMmodel,
    Variable,
    Scenario,
    Body,
    Function,
    AitkenRelaxation,
)
from funtofem.interface import (
    TacsSteadyInterface,
    SolverManager,
    TestResult,
)
from funtofem.driver import FUNtoFEMnlbgs, TransferSettings

# check whether fun3d is available
fun3d_loader = importlib.util.find_spec("fun3d")
has_fun3d = fun3d_loader is not None
if has_fun3d:
    from funtofem.interface import Fun3dInterface

np.random.seed(1234567)
comm = MPI.COMM_WORLD

results_folder = os.path.join(os.getcwd(), "results")
if comm.rank == 0:
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)

base_dir = os.path.dirname(os.path.abspath(__file__))
bdf_filename = os.path.join(base_dir, "meshes", "nastran_CAPS.dat")


class TestFun3dTacs(unittest.TestCase):
    FILENAME = "fun3d-tacs-driver.txt"
    FILEPATH = os.path.join(results_folder, FILENAME)

    def test_thick_laminar_aerothermoelastic(self):
        # build the funtofem model with one body and scenario
        model = FUNtoFEMmodel("plate")
        plate = Body.aerothermoelastic("plate", boundary=2)
        Variable.structural("thick").set_bounds(
            lower=0.001, value=0.1, upper=2.0
        ).register_to(plate)
        plate.register_to(model)
        test_scenario = Scenario.steady(
            "laminar", steps=500, preconditioner_steps=10
        ).set_temperature(T_ref=300.0, T_inf=300.0)
        test_scenario.include(Function.ksfailure(ks_weight=50.0)).include(
            Function.lift()
        ).include(Function.drag())
        test_scenario.register_to(model)

        # build the solvers and coupled driver
        solvers = SolverManager(comm)
        solvers.flow = Fun3dInterface(comm, model, fun3d_dir="meshes").set_units(
            qinf=1.05e5
        )

        solvers.structural = TacsSteadyInterface.create_from_bdf(
            model, comm, nprocs=1, bdf_file=bdf_filename
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
        max_rel_error = TestResult.complex_step(
            "fun3d+tacs-laminar-aerothermoelastic-structural",
            model,
            driver,
            TestFun3dTacs.FILEPATH,
        )
        self.assertTrue(max_rel_error < 1e-7)

    def test_alpha_laminar_aerothermoelastic(self):
        # build the funtofem model with one body and scenario
        model = FUNtoFEMmodel("plate")
        plate = Body.aerothermoelastic("plate", boundary=2)
        plate.register_to(model)
        test_scenario = Scenario.steady(
            "laminar", steps=500, preconditioner_steps=10
        ).set_temperature(T_ref=300.0, T_inf=300.0)
        test_scenario.include(Function.ksfailure(ks_weight=50.0)).include(
            Function.lift()
        ).include(Function.drag())
        test_scenario.set_variable(
            "aerodynamic", "AOA", lower=5.0, value=10.0, upper=15.0
        )
        test_scenario.register_to(model)

        # build the solvers and coupled driver
        solvers = SolverManager(comm)
        solvers.flow = Fun3dInterface(comm, model, fun3d_dir="meshes").set_units(
            qinf=1.05e5
        )

        solvers.structural = TacsSteadyInterface.create_from_bdf(
            model, comm, nprocs=1, bdf_file=bdf_filename
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
        max_rel_error = TestResult.complex_step(
            "fun3d+tacs-laminar-aerothermoelastic-flow",
            model,
            driver,
            TestFun3dTacs.FILEPATH,
        )
        self.assertTrue(max_rel_error < 1e-7)

    def test_thick_laminar_aeroelastic(self):
        # build the funtofem model with one body and scenario
        model = FUNtoFEMmodel("plate")
        plate = Body.aeroelastic("plate", boundary=2)
        Variable.structural("thick").set_bounds(
            lower=0.001, value=0.1, upper=2.0
        ).register_to(plate)
        plate.register_to(model)
        test_scenario = Scenario.steady(
            "laminar", steps=500, preconditioner_steps=10
        ).set_temperature(T_ref=300.0, T_inf=300.0)
        test_scenario.include(Function.ksfailure(ks_weight=50.0)).include(
            Function.lift()
        ).include(Function.drag())
        test_scenario.register_to(model)

        # build the solvers and coupled driver
        solvers = SolverManager(comm)
        solvers.flow = Fun3dInterface(comm, model, fun3d_dir="meshes").set_units(
            qinf=1.05e5
        )

        solvers.structural = TacsSteadyInterface.create_from_bdf(
            model, comm, nprocs=1, bdf_file=bdf_filename
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
        max_rel_error = TestResult.complex_step(
            "fun3d+tacs-laminar-aeroelastic-structural",
            model,
            driver,
            TestFun3dTacs.FILEPATH,
        )
        self.assertTrue(max_rel_error < 1e-7)

    def test_alpha_laminar_aeroelastic(self):
        # build the funtofem model with one body and scenario
        model = FUNtoFEMmodel("plate")
        plate = Body.aeroelastic("plate", boundary=2)
        plate.register_to(model)
        test_scenario = Scenario.steady(
            "laminar", steps=500, preconditioner_steps=10
        ).set_temperature(T_ref=300.0, T_inf=300.0)
        test_scenario.include(Function.ksfailure(ks_weight=50.0)).include(
            Function.lift()
        ).include(Function.drag())
        test_scenario.set_variable(
            "aerodynamic", "AOA", lower=5.0, value=10.0, upper=15.0
        )
        test_scenario.register_to(model)

        # build the solvers and coupled driver
        solvers = SolverManager(comm)
        solvers.flow = Fun3dInterface(comm, model, fun3d_dir="meshes").set_units(
            qinf=1.05e5
        )

        solvers.structural = TacsSteadyInterface.create_from_bdf(
            model, comm, nprocs=1, bdf_file=bdf_filename
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
        max_rel_error = TestResult.complex_step(
            "fun3d+tacs-laminar-aeroelastic-flow",
            model,
            driver,
            TestFun3dTacs.FILEPATH,
        )
        self.assertTrue(max_rel_error < 1e-7)

    def test_thick_laminar_aerothermal(self):
        # build the funtofem model with one body and scenario
        model = FUNtoFEMmodel("plate")
        plate = Body.aerothermal("plate", boundary=2)
        Variable.structural("thick").set_bounds(
            lower=0.001, value=0.1, upper=2.0
        ).register_to(plate)
        plate.register_to(model)
        test_scenario = Scenario.steady(
            "laminar", steps=500, preconditioner_steps=10
        ).set_temperature(T_ref=300.0, T_inf=300.0)
        test_scenario.include(Function.ksfailure(ks_weight=50.0)).include(
            Function.lift()
        ).include(Function.drag())
        test_scenario.register_to(model)

        # build the solvers and coupled driver
        solvers = SolverManager(comm)
        solvers.flow = Fun3dInterface(comm, model, fun3d_dir="meshes").set_units(
            qinf=1.05e5
        )

        solvers.structural = TacsSteadyInterface.create_from_bdf(
            model, comm, nprocs=1, bdf_file=bdf_filename
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
        max_rel_error = TestResult.complex_step(
            "fun3d+tacs-laminar-aerothermal-structural",
            model,
            driver,
            TestFun3dTacs.FILEPATH,
        )
        self.assertTrue(max_rel_error < 1e-7)

    def test_alpha_laminar_aerothermal(self):
        # build the funtofem model with one body and scenario
        model = FUNtoFEMmodel("plate")
        plate = Body.aerothermal("plate", boundary=2)
        plate.register_to(model)
        test_scenario = Scenario.steady(
            "laminar", steps=500, preconditioner_steps=10
        ).set_temperature(T_ref=300.0, T_inf=300.0)
        test_scenario.include(Function.ksfailure(ks_weight=50.0)).include(
            Function.lift()
        ).include(Function.drag())
        test_scenario.set_variable(
            "aerodynamic", "AOA", lower=5.0, value=10.0, upper=15.0
        )
        test_scenario.register_to(model)

        # build the solvers and coupled driver
        solvers = SolverManager(comm)
        solvers.flow = Fun3dInterface(comm, model, fun3d_dir="meshes").set_units(
            qinf=1.05e5
        )

        solvers.structural = TacsSteadyInterface.create_from_bdf(
            model, comm, nprocs=1, bdf_file=bdf_filename
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
        max_rel_error = TestResult.complex_step(
            "fun3d+tacs-laminar-aerothermal-flow",
            model,
            driver,
            TestFun3dTacs.FILEPATH,
        )
        self.assertTrue(max_rel_error < 1e-7)


if __name__ == "__main__":
    # open and close the file to reset it
    if comm.rank == 0:
        open(TestFun3dTacs.FILEPATH, "w").close()

    unittest.main()
