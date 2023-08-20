import numpy as np, unittest, importlib, os
from mpi4py import MPI

# from funtofem import TransferScheme
from funtofem.model import (
    FUNtoFEMmodel,
    Variable,
    Scenario,
    Body,
    Function,
)
from funtofem.interface import SolverManager, TestResult, Fun3dBC, Fun3dModel

# check whether fun3d is available
fun3d_loader = importlib.util.find_spec("fun3d")
tacs_loader = importlib.util.find_spec("tacs")
caps_loader = importlib.util.find_spec("caps")
has_fun3d = fun3d_loader is not None

if has_fun3d:
    from funtofem.driver import FuntofemShapeDriver, Fun3dRemote

if tacs_loader is not None and caps_loader is not None:
    from tacs import caps2tacs

np.random.seed(1234567)

comm = MPI.COMM_WORLD
base_dir = os.path.dirname(os.path.abspath(__file__))
csm_path = os.path.join(base_dir, "naca_wing2.csm")

analysis_file = os.path.join(base_dir, "run_funtofem_analysis.py")
fun3d_dir = os.path.join(base_dir, "meshes")

results_folder = os.path.join(base_dir, "results")
if comm.rank == 0:  # make the results folder if doesn't exist
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)

active_cases = ["aero"]  # ["aero", "struct", "aero-struct"]


class TestFuntofemRemesh(unittest.TestCase):
    """
    This class performs unit test on the oneway-coupled FUN3D driver
    which uses fixed struct disps or no struct disps
    TODO : in the case of an unsteady one, add methods for those too?
    """

    FILENAME = "funtofem-shape-driver.txt"
    FILEPATH = os.path.join(results_folder, FILENAME)

    @unittest.skipIf(
        not ("aero" in active_cases),
        "struct shape case is toggled off by active_cases list",
    )
    def test_aero_shape(self):
        # build the funtofem model with one body and scenario
        model = FUNtoFEMmodel("wing")
        # design the shape
        fun3d_model = Fun3dModel.build(csm_file=csm_path, comm=comm)
        aflr_aim = fun3d_model.aflr_aim

        fun3d_aim = fun3d_model.fun3d_aim
        fun3d_aim.set_config_parameter("view:flow", 1)
        fun3d_aim.set_config_parameter("view:struct", 0)

        aflr_aim.set_surface_mesh(ff_growth=1.3, min_scale=0.01, max_scale=5.0)
        aflr_aim.set_boundary_layer(initial_spacing=0.001, thickness=0.01)
        Fun3dBC.inviscid(caps_group="wall", wall_spacing=0.001).register_to(fun3d_model)
        Fun3dBC.Farfield(caps_group="Farfield").register_to(fun3d_model)
        fun3d_model.setup()
        model.flow = fun3d_model

        wing = Body.aeroelastic("wing", boundary=2)
        Variable.shape(name="aoa").set_bounds(
            lower=-1.0, value=0.0, upper=1.0
        ).register_to(wing)
        wing.register_to(model)
        test_scenario = Scenario.steady("euler", steps=10).fun3d_project(
            "funtofem_CAPS"
        )
        # test_scenario.get_variable("AOA").set_bounds(value=2.0)
        test_scenario.include(Function.lift()).include(Function.drag())
        test_scenario.register_to(model)

        # build the solvers and coupled driver
        solvers = SolverManager(comm)
        fun3d_remote = Fun3dRemote(analysis_file, fun3d_dir, nprocs=48)
        driver = FuntofemShapeDriver.remote(solvers, model, fun3d_remote)

        # run the complex step test on the model and driver
        max_rel_error = TestResult.finite_difference(
            "funtofem-remesh-aero-shape-euler-aeroelastic",
            model,
            driver,
            TestFuntofemRemesh.FILEPATH,
            both_adjoint=False, 
            epsilon=1.0,
        )
        self.assertTrue(max_rel_error < 1e-4)

        return

    @unittest.skipIf(
        not ("aero-struct" in active_cases),
        "aero-struct shape case is toggled off by active_cases list",
    )
    def test_aero_struct_shape(self):
        # build the funtofem model with one body and scenario
        model = FUNtoFEMmodel("wing")

        # design the FUN3D aero shape model
        fun3d_model = Fun3dModel.build(csm_file=csm_path, comm=comm)
        aflr_aim = fun3d_model.aflr_aim

        fun3d_aim = fun3d_model.fun3d_aim
        fun3d_aim.set_config_parameter("view:flow", 1)
        fun3d_aim.set_config_parameter("view:struct", 0)

        aflr_aim.set_surface_mesh(ff_growth=1.3, min_scale=0.01, max_scale=5.0)
        aflr_aim.set_boundary_layer(initial_spacing=0.001, thickness=0.01)
        Fun3dBC.viscous(caps_group="wall", wall_spacing=0.001).register_to(fun3d_model)
        Fun3dBC.Farfield(caps_group="Farfield").register_to(fun3d_model)
        fun3d_model.setup()
        model.flow = fun3d_model

        # design the TACS struct shape model
        tacs_model = caps2tacs.TacsModel.build(csm_file=csm_path, comm=comm)
        tacs_model.egads_aim.set_mesh(  # need a refined-enough mesh for the derivative test to pass
            edge_pt_min=5,
            edge_pt_max=10,
            global_mesh_size=0.1,
            max_surf_offset=0.01,
            max_dihedral_angle=5,
        ).register_to(
            tacs_model
        )
        tacs_aim = tacs_model.tacs_aim
        tacs_aim.set_config_parameter("view:flow", 0)
        tacs_aim.set_config_parameter("view:struct", 1)
        model.structural = tacs_model

        # setup the funtofem bodies
        wing = Body.aeroelastic("wing", boundary=2)

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

        # register the internal structure shape variable
        Variable.shape(name="rib_a1").set_bounds(
            lower=0.4, value=1.0, upper=1.6
        ).register_to(wing)
        wing.register_to(model)
        # register the aero-struct shape variable
        Variable.shape(name="aoa").set_bounds(
            lower=-5.0, value=0.0, upper=5.0
        ).register_to(wing)

        # define the funtofem scenarios
        test_scenario = (
            Scenario.steady("euler", steps=5000)
            .set_temperature(T_ref=300.0, T_inf=300.0)
            .fun3d_project("funtofem_CAPS")
        )
        test_scenario.adjoint_steps = 2000
        # test_scenario.get_variable("AOA")
        test_scenario.include(Function.lift()).include(Function.drag())
        test_scenario.include(Function.ksfailure()).include(Function.mass())
        test_scenario.register_to(model)

        # build the solvers and coupled driver
        solvers = SolverManager(comm)
        fun3d_remote = Fun3dRemote(analysis_file, fun3d_dir, nprocs=48)
        driver = FuntofemShapeDriver.remote(solvers, model, fun3d_remote)

        # run the complex step test on the model and driver
        max_rel_error = TestResult.finite_difference(
            "funtofem-remesh-aero+struct-shape-euler-aeroelastic",
            model,
            driver,
            TestFuntofemRemesh.FILEPATH,
            epsilon=0.1,
        )
        self.assertTrue(max_rel_error < 1e-4)

        return


if __name__ == "__main__":
    if comm.rank == 0:
        open(TestFuntofemRemesh.FILEPATH, "w").close()
    unittest.main()
