import numpy as np, unittest, importlib, os
from mpi4py import MPI

# from funtofem import TransferScheme
from pyfuntofem.model import (
    FUNtoFEMmodel,
    Variable,
    Scenario,
    Body,
    Function,
)
from pyfuntofem.interface import SolverManager, TestResult, Fun3dBC, Fun3dModel

# check whether fun3d is available
tacs_loader = importlib.util.find_spec("tacs")
caps_loader = importlib.util.find_spec("pyCAPS")
fun3d_loader = importlib.util.find_spec("fun3d")
has_fun3d = fun3d_loader is not None
has_tacs_shape = tacs_loader is not None and caps_loader is not None
has_modules = has_tacs_shape and has_fun3d

if has_fun3d:
    from pyfuntofem.driver import FuntofemShapeDriver, Fun3dRemote

if has_tacs_shape:
    from tacs import caps2tacs

np.random.seed(1234567)

comm = MPI.COMM_WORLD
base_dir = os.path.dirname(os.path.abspath(__file__))
csm_path = os.path.join(base_dir, "diamond_wedge.csm")

analysis_file = os.path.join(base_dir, "run_funtofem_analysis.py")
fun3d_dir = os.path.join(base_dir, "meshes")

results_folder = os.path.join(base_dir, "results")
if comm.rank == 0:  # make the results folder if doesn't exist
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)

# select which cases are active, ["aero", "struct", "aero-struct"]
active_cases = ["aero"]


@unittest.skipIf(
    not has_modules, "skipping funtofem shape test if missing FUN3D, TACS or pyCAPS"
)
class TestFuntofemShape(unittest.TestCase):
    """
    This class performs unit test on the oneway-coupled FUN3D driver
    which uses fixed struct disps or no struct disps
    TODO : in the case of an unsteady one, add methods for those too?
    """

    FILENAME = "funtofem-CAPS-shape.txt"
    FILEPATH = os.path.join(results_folder, FILENAME)

    @unittest.skipIf(
        not ("aero" in active_cases),
        "aero shape case is toggled off by active_cases list",
    )
    def test_aero_shape(self):
        # build the funtofem model with one body and scenario
        model = FUNtoFEMmodel("wing")

        # design the FUN3D aero shape model
        fun3d_model = Fun3dModel.build(csm_file=csm_path, comm=comm)
        aflr_aim = fun3d_model.aflr_aim
        aflr_aim.set_surface_mesh(ff_growth=1.3, min_scale=0.1, max_scale=5.0)
        aflr_aim.set_boundary_layer(initial_spacing=0.01, thickness=0.1)
        Fun3dBC.viscous(caps_group="wall", wall_spacing=0.01).register_to(fun3d_model)
        Fun3dBC.Farfield(caps_group="Farfield").register_to(fun3d_model)
        fun3d_model.setup()
        model.flow = fun3d_model

        # setup the funtofem bodies
        wing = Body.aeroelastic("wing", boundary=2)
        Variable.shape(name="sspan").set_bounds(
            lower=0.4, value=5.0, upper=9.6
        ).register_to(wing)
        wing.register_to(model)

        # define the funtofem scenarios
        test_scenario = (
            Scenario.steady("turbulent", steps=1000)
            .set_temperature(T_ref=300.0, T_inf=300.0)
            .fun3d_project("funtofem_CAPS")
        )
        test_scenario.get_variable("AOA")
        test_scenario.include(Function.lift()).include(Function.drag())
        test_scenario.include(Function.ksfailure()).include(Function.mass())
        test_scenario.register_to(model)

        # build the solvers and coupled driver without solver interfaces
        # since this is the remote driver not the analysis driver
        solvers = SolverManager(comm)
        fun3d_remote = Fun3dRemote(analysis_file, fun3d_dir, nprocs=48)
        driver = FuntofemShapeDriver.remote(solvers, model, fun3d_remote)

        # run the complex step test on the model and driver
        max_rel_error = TestResult.finite_difference(
            "funtofem-aero-shape-turbulent-aeroelastic",
            model,
            driver,
            TestFuntofemShape.FILEPATH,
        )
        self.assertTrue(max_rel_error < 1e-4)

    @unittest.skipIf(
        not ("struct" in active_cases),
        "struct shape case is toggled off by active_cases list",
    )
    def test_struct_shape(self):
        # build the funtofem model with one body and scenario
        model = FUNtoFEMmodel("wing")

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
        Variable.shape(name="sspan").set_bounds(
            lower=0.4, value=5.0, upper=9.6
        ).register_to(wing)

        # define the funtofem scenarios
        test_scenario = (
            Scenario.steady("turbulent", steps=1000)
            .set_temperature(T_ref=300.0, T_inf=300.0)
            .fun3d_project("funtofem_CAPS")
        )
        test_scenario.get_variable("AOA")
        test_scenario.include(Function.lift()).include(Function.drag())
        test_scenario.include(Function.ksfailure()).include(Function.mass())
        test_scenario.register_to(model)

        # build the solvers and coupled driver
        # the one with pure structural shape is technically not required to
        # be ran with the remote driver, can be setup with analysis classmethod
        # but with the
        solvers = SolverManager(comm)
        fun3d_remote = Fun3dRemote(analysis_file, fun3d_dir, nprocs=48)
        driver = FuntofemShapeDriver.remote(solvers, model, fun3d_remote)

        # run the complex step test on the model and driver
        max_rel_error = TestResult.finite_difference(
            "funtofem-struct-shape-turbulent-aeroelastic",
            model,
            driver,
            TestFuntofemShape.FILEPATH,
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
        aflr_aim.set_surface_mesh(ff_growth=1.3, min_scale=0.1, max_scale=5.0)
        aflr_aim.set_boundary_layer(initial_spacing=0.01, thickness=0.1)
        Fun3dBC.viscous(caps_group="wall", wall_spacing=0.01).register_to(fun3d_model)
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
        Variable.shape(name="sspan").set_bounds(
            lower=0.4, value=5.0, upper=9.6
        ).register_to(wing)

        # define the funtofem scenarios
        test_scenario = (
            Scenario.steady("turbulent", steps=1000)
            .set_temperature(T_ref=300.0, T_inf=300.0)
            .fun3d_project("funtofem_CAPS")
        )
        test_scenario.get_variable("AOA")
        test_scenario.include(Function.lift()).include(Function.drag())
        test_scenario.include(Function.ksfailure()).include(Function.mass())
        test_scenario.register_to(model)

        # build the solvers and coupled driver
        solvers = SolverManager(comm)
        fun3d_remote = Fun3dRemote(analysis_file, fun3d_dir, nprocs=48)
        driver = FuntofemShapeDriver.remote(solvers, model, fun3d_remote)

        # run the complex step test on the model and driver
        max_rel_error = TestResult.finite_difference(
            "funtofem-aero+struct-shape-turbulent-aeroelastic",
            model,
            driver,
            TestFuntofemShape.FILEPATH,
        )
        self.assertTrue(max_rel_error < 1e-4)

        return


if __name__ == "__main__":
    if comm.rank == 0:
        open(TestFuntofemShape.FILEPATH, "w").close()
    unittest.main()
