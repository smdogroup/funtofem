import os, unittest, numpy as np, importlib
from mpi4py import MPI
from tacs import TACS, caps2tacs
from funtofem import TransferScheme
from pyfuntofem.model import FUNtoFEMmodel, Variable, Scenario, Body, Function
from pyfuntofem.interface import (
    TestAerodynamicSolver,
    TacsSteadyInterface,
    SolverManager,
    TestResult,
)
from pyfuntofem.driver import FUNtoFEMnlbgs, TransferSettings, TacsOnewayDriver

from bdf_test_utils import thermoelasticity_callback, elasticity_callback

np.random.seed(1234567)

comm = MPI.COMM_WORLD
base_dir = os.path.dirname(os.path.abspath(__file__))
bdf_filename = os.path.join(base_dir, "input_files", "test_bdf_file.bdf")

tacs_loader = importlib.util.find_spec("tacs")
caps_loader = importlib.util.find_spec("pyCAPS")

base_dir = os.path.dirname(os.path.abspath(__file__))
csm_path = os.path.join(base_dir, "input_files", "simple_naca_wing.csm")
dat_filepath = os.path.join(base_dir, "input_files", "simple_naca_wing.dat")

results_folder = os.path.join(base_dir, "results")
if comm.rank == 0:  # make the results folder if doesn't exist
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)


class TestTacsOnewayDriverParallel(unittest.TestCase):
    N_PROCS = 2
    """
    This class performs unit test on the oneway-coupled TacsSteadyAnalysisDriver
    which uses fixed aero loads
    TODO : in the case of an unsteady one, add methods for those too?
    """

    FILENAME = "oneway-driver.txt"
    FILEPATH = os.path.join(results_folder, FILENAME)

    @unittest.skip("temp")
    def test_aeroelastic(self):
        # build the model and driver
        model = FUNtoFEMmodel("wedge")
        plate = Body.aeroelastic("plate", boundary=1)
        Variable.structural("thickness").set_bounds(
            lower=0.01, value=0.1, upper=1.0
        ).register_to(plate)
        plate.register_to(model)

        # build the scenario
        scenario = Scenario.steady("test", steps=200).include(Function.ksfailure())
        scenario.register_to(model)

        # build the tacs interface, coupled driver, and oneway driver
        comm = MPI.COMM_WORLD
        solvers = SolverManager(comm)
        solvers.flow = TestAerodynamicSolver(comm, model)
        solvers.structural = TacsSteadyInterface.create_from_bdf(
            model, comm, 2, bdf_filename, callback=elasticity_callback
        )
        transfer_settings = TransferSettings(npts=5)
        coupled_driver = FUNtoFEMnlbgs(
            solvers, transfer_settings=transfer_settings, model=model
        )
        oneway_driver = TacsOnewayDriver.prime_loads(coupled_driver)

        complex_mode = False
        rtol = 1e-4
        if TransferScheme.dtype == complex and TACS.dtype == complex:
            complex_mode = True
            rtol = 1e-7

        # run teh oomplex step test
        max_rel_error = TestResult.derivative_test(
            "oneway-aeroelastic",
            model,
            oneway_driver,
            TestTacsOnewayDriverParallel.FILEPATH,
            complex_mode=complex_mode,
        )
        self.assertTrue(max_rel_error < rtol)

        return

    # @unittest.skip("temp")
    @unittest.skipIf(
        tacs_loader is None or caps_loader is None,
        "skipping test using caps2tacs if caps or tacs are unavailable",
    )
    def test_shape_aeroelastic(self):
        # make the funtofem and tacs model
        f2f_model = FUNtoFEMmodel("wing")
        tacs_model = caps2tacs.TacsModel.build(csm_file=csm_path, comm=comm)
        tacs_model.egads_aim.set_mesh(  # need a refined-enough mesh for the derivative test to pass
            edge_pt_min=15,
            edge_pt_max=20,
            global_mesh_size=0.1,
            max_surf_offset=0.01,
            max_dihedral_angle=5,
        ).register_to(
            tacs_model
        )
        f2f_model.tacs_model = tacs_model

        # build a body which we will register variables to
        wing = Body.aeroelastic("wing")

        # setup the material and shell properties
        aluminum = caps2tacs.Isotropic.aluminum().register_to(tacs_model)

        nribs = int(tacs_model.get_config_parameter("nribs"))
        nspars = int(tacs_model.get_config_parameter("nspars"))

        for irib in range(1, nribs + 1):
            caps2tacs.ShellProperty(
                caps_group=f"rib{irib}", material=aluminum, membrane_thickness=0.05
            ).register_to(tacs_model)
        for ispar in range(1, nspars + 1):
            caps2tacs.ShellProperty(
                caps_group=f"spar{ispar}", material=aluminum, membrane_thickness=0.05
            ).register_to(tacs_model)
        caps2tacs.ShellProperty(
            caps_group="OML", material=aluminum, membrane_thickness=0.03
        ).register_to(tacs_model)

        # register any shape variables to the wing which are auto-registered to tacs model
        Variable.shape(name="rib_a1").set_bounds(
            lower=0.4, value=1.0, upper=1.6
        ).register_to(wing)

        # register the wing body to the model
        wing.register_to(f2f_model)

        # add remaining information to tacs model
        caps2tacs.PinConstraint("root").register_to(tacs_model)

        # make a funtofem scenario
        test_scenario = Scenario.steady("test", steps=100).include(Function.mass())
        test_scenario.register_to(f2f_model)

        flow_solver = TestAerodynamicSolver(comm, f2f_model)
        transfer_settings = TransferSettings(npts=5)

        # setup the tacs model
        tacs_aim = tacs_model.tacs_aim
        tacs_aim.setup_aim()

        shape_driver = TacsOnewayDriver.prime_loads_shape(
            flow_solver,
            tacs_aim,
            transfer_settings,
            nprocs=2,
            bdf_file=dat_filepath,
        )

        max_rel_error = TestResult.derivative_test(
            "testaero=>tacs_shape-aeroelastic",
            f2f_model,
            shape_driver,
            TestTacsOnewayDriverParallel.FILEPATH,
            complex_mode=False,
        )
        rtol = 1e-4
        self.assertTrue(max_rel_error < rtol)


if __name__ == "__main__":
    if comm.rank == 0:
        open(TestTacsOnewayDriverParallel.FILEPATH, "w").close()
    unittest.main()
