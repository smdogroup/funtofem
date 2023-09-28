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
from funtofem.interface import (
    SolverManager,
    TestResult,
    Fun3dBC,
    Fun3dModel,
    make_test_directories,
)
from funtofem.driver import FUNtoFEMnlbgs, TransferSettings

# check whether fun3d is available
fun3d_loader = importlib.util.find_spec("fun3d")
tacs_loader = importlib.util.find_spec("tacs")
has_fun3d = fun3d_loader is not None

if has_fun3d:
    from funtofem.interface import Fun3dInterface

if tacs_loader is not None:
    from funtofem.interface import TacsInterface, TacsIntegrationSettings

np.random.seed(1234567)

comm = MPI.COMM_WORLD
base_dir = os.path.dirname(os.path.abspath(__file__))
csm_path = os.path.join(base_dir, "meshes", "naca_wing.csm")
fun3d_dir = os.path.join(base_dir, "meshes")
nprocs = comm.Get_size()
bdf_file = os.path.join(base_dir, "meshes", "tacs_CAPS.dat")
results_folder, output_dir = make_test_directories(comm, base_dir)

num_steps = 10
dim_dt = 0.001
a_inf = 347.224
qinf = 105493.815
flow_dt = 0.1
elastic_scheme = "rbf"


class TestFuntofemUnsteady(unittest.TestCase):
    """
    This class performs unit test on the oneway-coupled FUN3D driver
    which uses fixed struct disps or no struct disps
    TODO : in the case of an unsteady one, add methods for those too?
    """

    FILENAME = "funtofem-naca-unsteady.txt"
    FILEPATH = os.path.join(results_folder, FILENAME)

    def test_euler_aeroelastic_thick(self):
        """test no struct disps into FUN3D"""
        # build the funtofem model with one body and scenario
        model = FUNtoFEMmodel("wing")
        # design the shape
        fun3d_model = Fun3dModel.build_morph(
            csm_file=csm_path, comm=comm, project_name="funtofem_CAPS"
        )
        aflr_aim = fun3d_model.aflr_aim
        fun3d_aim = fun3d_model.fun3d_aim

        # smaller mesh length is more refined, original value = 5.0
        aflr_aim.set_surface_mesh(ff_growth=1.4, mesh_length=5.0)
        Fun3dBC.inviscid(caps_group="wall").register_to(fun3d_model)

        farfield = Fun3dBC.Farfield(caps_group="Farfield").register_to(fun3d_model)
        aflr_aim.mesh_sizing(farfield)
        fun3d_model.setup()
        model.flow = fun3d_model

        # setup grid filepaths so the aero mesh ends up in the right place
        fun3d_aim.grid_filepaths = [
            os.path.join(
                fun3d_dir,
                "euler_unsteady",
                "Flow",
                f"{fun3d_aim.project_name}.lb8.ugrid",
            )
        ]

        # generate the mesh
        fun3d_model.pre_analysis()

        wing = Body.aeroelastic("wing", boundary=2)
        Variable.structural("thick").set_bounds(
            lower=0.001, value=0.01, upper=2.0
        ).register_to(wing)
        wing.register_to(model)

        test_scenario = Scenario.unsteady("euler_unsteady", steps=10)
        TacsIntegrationSettings(dt=dim_dt, num_steps=num_steps).register_to(
            test_scenario
        )
        test_scenario.set_temperature(T_ref=300.0, T_inf=300.0)
        test_scenario.set_flow_ref_vals(qinf=qinf, flow_dt=flow_dt)
        test_scenario.fun3d_project(fun3d_aim.project_name)
        Function.ksfailure(ks_weight=10.0).register_to(test_scenario)
        Function.compliance().register_to(test_scenario)
        Function.lift().set_timing(start=1, stop=num_steps, averaging=True).register_to(
            test_scenario
        )
        Function.drag().set_timing(start=1, stop=num_steps, averaging=True).register_to(
            test_scenario
        )
        test_scenario.register_to(model)

        # build the solvers and coupled driver
        solvers = SolverManager(comm)
        solvers.flow = Fun3dInterface(
            comm,
            model,
            fun3d_dir="meshes",
            auto_coords=False,
            forward_options={"timedep_adj_frozen": True},
            adjoint_options={"timedep_adj_frozen": True},
        )
        solvers.structural = TacsInterface.create_from_bdf(
            model, comm, nprocs=nprocs, bdf_file=bdf_file, output_dir=output_dir
        )

        # build the driver and transfer settings
        transfer_settings = TransferSettings(elastic_scheme=elastic_scheme, npts=200)
        driver = FUNtoFEMnlbgs(
            solvers, transfer_settings=transfer_settings, model=model
        )

        # run the complex step test on the model and driver
        max_rel_error = TestResult.complex_step(
            "funtofem-unsteady-laminar-aeroelastic",
            model,
            driver,
            self.FILEPATH,
        )
        self.assertTrue(max_rel_error < 1e-7)

        return


if __name__ == "__main__":
    if comm.rank == 0:
        open(TestFuntofemUnsteady.FILEPATH, "w").close()
    unittest.main()
