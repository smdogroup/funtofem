# This test uses the 'miniMesh' case.

import numpy as np, unittest, importlib, os
from mpi4py import MPI

from funtofem.model import (
    FUNtoFEMmodel,
    Variable,
    Scenario,
    Body,
    Function,
)
from funtofem.interface import (
    TacsUnsteadyInterface,
    SolverManager,
    TestResult,
    TacsIntegrationSettings,
    make_test_directories,
)
from funtofem.driver import FUNtoFEMnlbgs, TransferSettings

# check whether fun3d is available
fun3d_loader = importlib.util.find_spec("fun3d")
has_fun3d = fun3d_loader is not None

if has_fun3d:
    from funtofem.interface import Fun3dInterface

np.random.seed(314159)
comm = MPI.COMM_WORLD
base_dir = os.path.dirname(os.path.abspath(__file__))
bdf_filename = os.path.join(base_dir, "meshes", "nastran_CAPS.dat")
results_folder, output_dir = make_test_directories(comm, base_dir)

# settings
num_steps = 10
dim_dt = 0.001
a_inf = 347.224
qinf = 105493.815
# flow_dt = 1 / a_inf
flow_dt = 1.0
elastic_scheme = "rbf"  # "rbf" or "meld" (rbf better so far)


@unittest.skipIf(
    not has_fun3d, "Skipping FUN3D-TACS unsteady tests because FUN3D is unavailable."
)
class TestFun3dTacsUnsteady(unittest.TestCase):
    FILENAME = "fun3d-tacs-unsteady-driver.txt"
    FILEPATH = os.path.join(results_folder, FILENAME)

    def test_laminar_aeroelastic_thick(self):
        # build the funtofem model with one body and scenario
        model = FUNtoFEMmodel("plate")
        plate = Body.aeroelastic("plate", boundary=2)

        Variable.structural("thick").set_bounds(
            lower=0.001, value=0.01, upper=2.0
        ).register_to(plate)
        plate.register_to(model)

        # define the scenario
        test_scenario = Scenario.unsteady("laminar", steps=num_steps)
        TacsIntegrationSettings(dt=dim_dt, num_steps=num_steps).register_to(
            test_scenario
        )
        test_scenario.set_temperature(T_ref=300.0, T_inf=300.0)
        test_scenario.set_flow_ref_vals(qinf=qinf, flow_dt=flow_dt)
        Function.ksfailure(ks_weight=10.0).register_to(test_scenario)
        Function.compliance().register_to(test_scenario)
        Function.lift().set_timing(start=1, stop=num_steps, averaging=True).register_to(
            test_scenario
        )
        Function.drag().set_timing(start=1, stop=num_steps, averaging=True).register_to(
            test_scenario
        )
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
            model, comm, nprocs=1, bdf_file=bdf_filename, output_dir=output_dir
        )

        transfer_settings = TransferSettings(elastic_scheme=elastic_scheme, npts=20)

        # Build the driver
        driver = FUNtoFEMnlbgs(
            solvers,
            transfer_settings=transfer_settings,
            model=model,
        )

        # run the complex step test on the model and driver
        max_rel_error = TestResult.complex_step(
            "fun3d+tacs-laminar-aeroelastic-thick",
            model,
            driver,
            TestFun3dTacsUnsteady.FILEPATH,
        )
        self.assertTrue(max_rel_error < 1e-7)

    def test_laminar_aeroelastic_alpha(self):
        # build the funtofem model with one body and scenario
        model = FUNtoFEMmodel("plate")
        plate = Body.aeroelastic("plate", boundary=2)
        plate.register_to(model)

        # define the scenario
        test_scenario = Scenario.unsteady("laminar", steps=num_steps)
        TacsIntegrationSettings(dt=dim_dt, num_steps=num_steps).register_to(
            test_scenario
        )
        test_scenario.set_temperature(T_ref=300.0, T_inf=300.0)
        test_scenario.set_flow_ref_vals(qinf=qinf, flow_dt=flow_dt)
        aoa = test_scenario.get_variable("AOA", set_active=True)
        aoa.set_bounds(lower=5.0, value=10.0, upper=15.0)
        Function.ksfailure(ks_weight=10.0).register_to(test_scenario)
        Function.compliance().register_to(test_scenario)
        Function.lift().set_timing(start=1, stop=num_steps, averaging=True).register_to(
            test_scenario
        )
        Function.drag().set_timing(start=1, stop=num_steps, averaging=True).register_to(
            test_scenario
        )
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
            model, comm, nprocs=1, bdf_file=bdf_filename, output_dir=output_dir
        )

        transfer_settings = TransferSettings(elastic_scheme=elastic_scheme, npts=20)

        # Build the driver
        driver = FUNtoFEMnlbgs(
            solvers,
            transfer_settings=transfer_settings,
            model=model,
        )

        # run the complex step test on the model and driver
        max_rel_error = TestResult.complex_step(
            "fun3d+tacs-laminar-aeroelastic-alpha",
            model,
            driver,
            TestFun3dTacsUnsteady.FILEPATH,
        )
        self.assertTrue(max_rel_error < 1e-7)


if __name__ == "__main__":
    # open and close the file to reset it
    if comm.rank == 0:
        open(TestFun3dTacsUnsteady.FILEPATH, "w").close()
    unittest.main()
