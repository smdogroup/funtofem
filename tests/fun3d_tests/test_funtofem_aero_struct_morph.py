import numpy as np, unittest, importlib, os
from mpi4py import MPI

# from funtofem import TransferScheme
from funtofem.model import (
    FUNtoFEMmodel,
    Variable,
    Scenario,
    Body,
    Function,
    AitkenRelaxation,
)
from funtofem.interface import SolverManager, TestResult, Fun3dBC, Fun3dModel

# check whether fun3d is available
fun3d_loader = importlib.util.find_spec("fun3d")
tacs_loader = importlib.util.find_spec("tacs")
caps_loader = importlib.util.find_spec("pyCAPS")
has_fun3d = fun3d_loader is not None

if has_fun3d:
    from funtofem.driver import FuntofemShapeDriver
    from funtofem.interface import Fun3dInterface

if tacs_loader is not None and caps_loader is not None:
    from tacs import caps2tacs

np.random.seed(1234567)

comm = MPI.COMM_WORLD
base_dir = os.path.dirname(os.path.abspath(__file__))
csm_path = os.path.join(base_dir, "naca_wing_MD.csm")
fun3d_dir = os.path.join(base_dir, "meshes")
nprocs = comm.Get_size()
bdf_file = os.path.join(base_dir, "meshes", "nastran_CAPS.dat")

results_folder = os.path.join(base_dir, "results")
if comm.rank == 0:  # make the results folder if doesn't exist
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)


class TestFuntofemMorph(unittest.TestCase):
    """
    This class performs unit test on the oneway-coupled FUN3D driver
    which uses fixed struct disps or no struct disps
    TODO : in the case of an unsteady one, add methods for those too?
    """

    FILENAME = "funtofem-morph-shape-driver.txt"
    FILEPATH = os.path.join(results_folder, FILENAME)

    def test_nominal(self):
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

        # smaller mesh length is more refined, original value = 5.0
        aflr_aim.set_surface_mesh(ff_growth=1.4, mesh_length=0.8)
        Fun3dBC.inviscid(caps_group="wall").register_to(fun3d_model)

        farfield = Fun3dBC.Farfield(caps_group="Farfield").register_to(fun3d_model)
        aflr_aim.mesh_sizing(farfield)
        fun3d_model.setup()
        model.flow = fun3d_model

        # design the struct shape or tacs model
        tacs_model = caps2tacs.TacsModel.build(csm_file=csm_path, comm=comm)
        tacs_model.egads_aim.set_mesh(
            edge_pt_min=5,
            edge_pt_max=10,
            global_mesh_size=0.1,
            max_surf_offset=0.01,
            max_dihedral=5,
        ).register_to(tacs_model)
        tacs_aim = tacs_model.tacs_aim
        tacs_aim.set_config_parameter("view:flow", 0)
        tacs_aim.set_config_parameter("view:struct", 1)
        model.structural = tacs_model

        wing = Body.aeroelastic("wing", boundary=2)
        Variable.shape(name="aoa").set_bounds(
            lower=-1.0, value=0.0, upper=1.0
        ).register_to(wing)
        wing.register_to(model)
        test_scenario = (
            Scenario.steady("turbulent", steps=10000)  # 5000
            .set_temperature(T_ref=300.0, T_inf=300.0)
            .fun3d_project(fun3d_aim.project_name)
        )
        test_scenario.adjoint_steps = 9000  # 2000
        # test_scenario.get_variable("AOA").set_bounds(value=2.0)

        test_scenario.include(Function.lift()).include(Function.drag())
        test_scenario.register_to(model)

        # build the solvers and coupled driver
        solvers = SolverManager(comm)
        solvers.flow = Fun3dInterface(
            comm, model, fun3d_dir="meshes", auto_coords=False
        ).set_units(qinf=1e4)
        solvers.structural = None  # will be built by the tacs model at runtime

        # analysis driver for mesh morphing
        driver = FuntofemShapeDriver.nominal(solvers, model)

        # run the complex step test on the model and driver
        max_rel_error = TestResult.finite_difference(
            "funtofem-aero+struct-shape-euler-aeroelastic",
            model,
            driver,
            TestFuntofemMorph.FILEPATH,
            epsilon=1e-4,
            both_adjoint=False,
        )
        self.assertTrue(max_rel_error < 1e-4)

        return


if __name__ == "__main__":
    if comm.rank == 0:
        open(TestFuntofemMorph.FILEPATH, "w").close()
    unittest.main()
