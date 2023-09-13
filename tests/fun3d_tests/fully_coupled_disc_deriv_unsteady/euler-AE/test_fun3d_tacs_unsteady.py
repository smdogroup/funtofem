# This test uses the 'miniMesh' case.

import numpy as np, unittest, importlib, os
from mpi4py import MPI

from tacs import TACS, elements, constitutive
from funtofem.model import (
    FUNtoFEMmodel,
    Variable,
    Scenario,
    Body,
    Function,
    AitkenRelaxation,
)
from funtofem.interface import (
    TacsUnsteadyInterface,
    SolverManager,
    TestResult,
    TacsIntegrationSettings,
)
from funtofem.driver import FUNtoFEMnlbgs, TransferSettings

# check whether fun3d is available
fun3d_loader = importlib.util.find_spec("fun3d")
has_fun3d = fun3d_loader is not None

if has_fun3d:
    from funtofem.interface import Fun3dInterface

np.random.seed(314159)
comm = MPI.COMM_WORLD

results_folder = os.path.join(os.getcwd(), "results")
if not os.path.exists(results_folder) and comm.rank == 0:
    os.mkdir(results_folder)

base_dir = os.path.dirname(os.path.abspath(__file__))
bdf_filename = os.path.join(base_dir, "meshes", "nastran_CAPS.dat")

num_steps = 10
dim_dt = 0.001
a_inf = 347.224
qinf = 105493.815
flow_dt = 1 / a_inf


@unittest.skipIf(
    not has_fun3d, "Skipping FUN3D-TACS unsteady tests because FUN3D is unavailable."
)
class TestFun3dTacsUnsteady(unittest.TestCase):
    FILENAME = "fun3d-tacs-unsteady-driver.txt"
    FILEPATH = os.path.join(results_folder, FILENAME)

    def test_inviscid_aeroelastic_thick(self):
        # build the funtofem model with one body and scenario
        model = FUNtoFEMmodel("plate")
        plate = Body.aeroelastic("plate", boundary=2)

        Variable.structural("thick").set_bounds(
            lower=0.001, value=0.01, upper=2.0
        ).register_to(plate)

        plate.register_to(model)

        integration_settings = TacsIntegrationSettings(dt=dim_dt, num_steps=num_steps)

        test_scenario = Scenario.unsteady(
            "inviscid", steps=num_steps, tacs_integration_settings=integration_settings
        ).set_temperature(T_ref=300.0, T_inf=300.0)
        test_scenario.include(Function.ksfailure(ks_weight=50.0)).include(
            Function.lift()
        ).include(Function.drag()).include(
            Function.compliance()
        )  # .include(Function.mass())
        test_scenario.set_flow_ref_vals(qinf=qinf, flow_dt=flow_dt)

        test_scenario.register_to(model)

        ## Build the solvers and coupled driver
        solvers = SolverManager(comm)
        # Build the fluid solver
        solvers.flow = Fun3dInterface(
            comm,
            model,
            fun3d_dir="meshes",
            forward_options={"timedep_adj_frozen": True},
            adjoint_options={"timedep_adj_frozen": True},
        )

        # Build the structural solver
        solvers.structural = TacsUnsteadyInterface.create_from_bdf(
            model, comm, nprocs=1, bdf_file=bdf_filename, output_dir="tacs_output"
        )

        transfer_settings = TransferSettings(elastic_scheme="meld", npts=20)

        # Build the driver
        driver = FUNtoFEMnlbgs(
            solvers,
            transfer_settings=transfer_settings,
            model=model,
        )

        # run the complex step test on the model and driver
        max_rel_error = TestResult.complex_step(
            "fun3d+tacs-inviscid-aeroelastic-thick",
            model,
            driver,
            TestFun3dTacsUnsteady.FILEPATH,
        )
        self.assertTrue(max_rel_error < 1e-7)

    def test_inviscid_aeroelastic_alpha(self):
        # build the funtofem model with one body and scenario
        model = FUNtoFEMmodel("plate")
        plate = Body.aeroelastic("plate", boundary=2)
        plate.register_to(model)

        integration_settings = TacsIntegrationSettings(dt=dim_dt, num_steps=num_steps)

        test_scenario = Scenario.unsteady(
            "inviscid", steps=num_steps, tacs_integration_settings=integration_settings
        ).set_temperature(T_ref=300.0, T_inf=300.0)
        test_scenario.include(Function.ksfailure(ks_weight=50.0)).include(
            Function.lift()
        ).include(Function.drag()).include(Function.compliance())

        test_scenario.set_variable(
            "aerodynamic", "AOA", lower=5.0, value=10.0, upper=15.0
        )
        test_scenario.set_flow_ref_vals(qinf=qinf, flow_dt=flow_dt)
        test_scenario.register_to(model)

        ## Build the solvers and coupled driver
        solvers = SolverManager(comm)
        # Build the fluid solver
        solvers.flow = Fun3dInterface(
            comm,
            model,
            fun3d_dir="meshes",
            forward_options={"timedep_adj_frozen": True},
            adjoint_options={"timedep_adj_frozen": True},
        )

        # Build the structural solver
        solvers.structural = TacsUnsteadyInterface.create_from_bdf(
            model, comm, nprocs=1, bdf_file=bdf_filename, output_dir="tacs_output"
        )

        transfer_settings = TransferSettings(elastic_scheme="meld", npts=20)

        # Build the driver
        driver = FUNtoFEMnlbgs(
            solvers,
            transfer_settings=transfer_settings,
            model=model,
        )

        # run the complex step test on the model and driver
        max_rel_error = TestResult.complex_step(
            "fun3d+tacs-inviscid-aeroelastic-alpha",
            model,
            driver,
            TestFun3dTacsUnsteady.FILEPATH,
        )
        self.assertTrue(max_rel_error < 1e-7)

    def __del__(self):
        # close the file handle on deletion of the object
        try:
            if comm.rank == 0:
                self.file_hdl.close()
        except:
            pass


if __name__ == "__main__":
    # open and close the file to reset it
    if comm.rank == 0:
        open(TestFun3dTacsUnsteady.FILEPATH, "w").close()

    unittest.main()
