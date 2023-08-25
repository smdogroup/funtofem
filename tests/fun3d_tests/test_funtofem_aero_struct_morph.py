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
caps_loader = importlib.util.find_spec("pyCAPS")
has_fun3d = fun3d_loader is not None

if has_fun3d:
    from funtofem.driver import FuntofemShapeDriver, TransferSettings
    from funtofem.interface import Fun3dInterface

if tacs_loader is not None and caps_loader is not None:
    from tacs import caps2tacs

np.random.seed(1234567)

comm = MPI.COMM_WORLD
base_dir = os.path.dirname(os.path.abspath(__file__))
csm_path = os.path.join(base_dir, "meshes", "naca_wing_multi-disc.csm")
nprocs = comm.Get_size()

results_folder = os.path.join(base_dir, "results")
if comm.rank == 0:  # make the results folder if doesn't exist
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)

# cases = ["euler", "turbulent"]
case = "euler"


class TestFuntofemAeroStructMorph(unittest.TestCase):
    """
    This class performs unit test on the oneway-coupled FUN3D driver
    which uses fixed struct disps or no struct disps
    Note that FUN3D mesh morphing as of 8/22/23 can only be done if
    the moving FSI changes shape but not if static aero surfaces change shape too
    """

    FILENAME = "funtofem-morph-shape-driver.txt"
    FILEPATH = os.path.join(results_folder, FILENAME)

    @unittest.skipIf(case != "euler", "choose which case to run above this class")
    def test_euler_aeroelastic(self):
        """test no struct disps into FUN3D"""
        # build the funtofem model with one body and scenario
        model = FUNtoFEMmodel("wing")
        # design the aero shape / flow model
        fun3d_model = Fun3dModel.build_morph(
            csm_file=csm_path, comm=comm, project_name="funtofem_CAPS"
        )
        aflr_aim = fun3d_model.aflr_aim
        fun3d_aim = fun3d_model.fun3d_aim

        fun3d_aim.set_config_parameter("view:flow", 1)
        fun3d_aim.set_config_parameter("view:struct", 0)

        # smaller mesh length is more refined
        aflr_aim.set_surface_mesh(ff_growth=1.4, mesh_length=5.0)
        Fun3dBC.inviscid(caps_group="wall").register_to(fun3d_model)

        farfield = Fun3dBC.Farfield(caps_group="Farfield").register_to(fun3d_model)
        aflr_aim.mesh_sizing(farfield)
        fun3d_model.setup()
        model.flow = fun3d_model

        # design the struct shape or tacs model
        tacs_model = caps2tacs.TacsModel.build(csm_file=csm_path, comm=comm)
        tacs_model.mesh_aim.set_mesh(
            edge_pt_min=15,
            edge_pt_max=20,
            global_mesh_size=0.01,
            max_surf_offset=0.01,
            max_dihedral_angle=5,
        ).register_to(tacs_model)
        tacs_aim = tacs_model.tacs_aim
        tacs_aim.set_config_parameter("view:flow", 0)
        tacs_aim.set_config_parameter("view:struct", 1)
        model.structural = tacs_model

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

        Variable.shape(name="aoa").set_bounds(
            lower=-1.0, value=0.0, upper=1.0
        ).register_to(wing)
        wing.register_to(model)

        # add remaining constraints to tacs model
        caps2tacs.PinConstraint("root").register_to(tacs_model)

        test_scenario = (
            Scenario.steady("euler", steps=5000)
            .set_temperature(T_ref=300.0, T_inf=300.0)
            .fun3d_project(fun3d_aim.project_name)
        )
        test_scenario.adjoint_steps = 4000

        test_scenario.include(Function.lift()).include(Function.drag())
        test_scenario.include(Function.ksfailure(ks_weight=10.0))
        test_scenario.register_to(model)

        # build the solvers and coupled driver
        solvers = SolverManager(comm)
        solvers.flow = Fun3dInterface(
            comm, model, fun3d_dir="meshes", auto_coords=False
        ).set_units(qinf=1e4)
        # solvers.structural will be built by the TACS model at runtime

        # analysis driver for mesh morphing
        driver = FuntofemShapeDriver.aero_morph(
            solvers,
            model,
            transfer_settings=TransferSettings(npts=200, elastic_scheme="meld"),
            struct_nprocs=nprocs,
        )

        # run the complex step test on the model and driver
        max_rel_error = TestResult.finite_difference(
            "funtofem-aero+struct-shape-euler-aeroelastic",
            model,
            driver,
            self.FILEPATH,
            epsilon=1e-1,  # larger step bc remeshing
            both_adjoint=False,
        )
        self.assertTrue(max_rel_error < 1e-4)

        return

    @unittest.skipIf(case != "turbulent", "choose which case to run above this class")
    def test_turbulent_aerothermoelastic(self):
        """test no struct disps into FUN3D"""
        # build the funtofem model with one body and scenario
        model = FUNtoFEMmodel("wing")
        # design the aero shape / flow model
        fun3d_model = Fun3dModel.build_morph(
            csm_file=csm_path, comm=comm, project_name="funtofem_CAPS"
        )
        aflr_aim = fun3d_model.aflr_aim
        fun3d_aim = fun3d_model.fun3d_aim

        fun3d_aim.set_config_parameter("view:flow", 1)
        fun3d_aim.set_config_parameter("view:struct", 0)

        # smaller mesh length is more refined
        aflr_aim.set_surface_mesh(ff_growth=1.4, mesh_length=5.0)
        aflr_aim.set_boundary_layer(initial_spacing=1e-3, thickness=0.1)
        Fun3dBC.viscous(caps_group="wall", wall_spacing=1e-3).register_to(fun3d_model)

        farfield = Fun3dBC.Farfield(caps_group="Farfield").register_to(fun3d_model)
        aflr_aim.mesh_sizing(farfield)
        fun3d_model.setup()
        model.flow = fun3d_model

        # design the struct shape or tacs model
        tacs_model = caps2tacs.TacsModel.build(csm_file=csm_path, comm=comm)
        tacs_model.mesh_aim.set_mesh(
            edge_pt_min=15,
            edge_pt_max=20,
            global_mesh_size=0.01,
            max_surf_offset=0.01,
            max_dihedral_angle=5,
        ).register_to(tacs_model)
        tacs_aim = tacs_model.tacs_aim
        tacs_aim.set_config_parameter("view:flow", 0)
        tacs_aim.set_config_parameter("view:struct", 1)
        model.structural = tacs_model

        wing = Body.aerothermoelastic("wing", boundary=2)

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

        Variable.shape(name="aoa").set_bounds(
            lower=-1.0, value=0.0, upper=1.0
        ).register_to(wing)
        wing.register_to(model)

        # add remaining constraints to tacs model
        caps2tacs.PinConstraint("root").register_to(tacs_model)
        # 0 means gauge temp in TACS, so matches 300 K ref temp from above
        caps2tacs.TemperatureConstraint("midplane", temperature=0).register_to(
            tacs_model
        )

        test_scenario = (
            Scenario.steady("turbulent", steps=5000)
            .set_temperature(T_ref=300.0, T_inf=300.0)
            .fun3d_project(fun3d_aim.project_name)
        )
        test_scenario.adjoint_steps = 4000

        test_scenario.include(Function.lift()).include(Function.drag())
        test_scenario.include(Function.ksfailure(ks_weight=10.0))
        test_scenario.register_to(model)

        # build the solvers and coupled driver
        solvers = SolverManager(comm)
        solvers.flow = Fun3dInterface(
            comm, model, fun3d_dir="meshes", auto_coords=False
        ).set_units(qinf=1e4)
        # solvers.structural will be built by the TACS model at runtime

        # analysis driver for mesh morphing
        driver = FuntofemShapeDriver.aero_morph(
            solvers,
            model,
            transfer_settings=TransferSettings(npts=200, elastic_scheme="meld"),
            struct_nprocs=nprocs,
        )

        # run the complex step test on the model and driver
        max_rel_error = TestResult.finite_difference(
            "funtofem-aero+struct-shape-turbulent-aerothermoelastic",
            model,
            driver,
            self.FILEPATH,
            epsilon=1e-1,  # larger step bc remeshing
            both_adjoint=False,
        )
        self.assertTrue(max_rel_error < 1e-4)

        return


if __name__ == "__main__":
    if comm.rank == 0:
        open(TestFuntofemAeroStructMorph.FILEPATH, "w").close()
    unittest.main()
