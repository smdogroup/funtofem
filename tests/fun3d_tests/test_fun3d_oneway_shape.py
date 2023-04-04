import numpy as np, unittest, importlib, os
from mpi4py import MPI

# from funtofem import TransferScheme
from pyfuntofem.model import (
    FUNtoFEMmodel,
    Variable,
    Scenario,
    Body,
    Function,
    AitkenRelaxation,
)
from pyfuntofem.interface import SolverManager, TestResult, Fun3dBC, Fun3dModel

# check whether fun3d is available
fun3d_loader = importlib.util.find_spec("fun3d")
has_fun3d = fun3d_loader is not None

if has_fun3d:
    from pyfuntofem.interface import Fun3dInterface, Fun3dAim, NamelistBlock
    from pyfuntofem.driver import Fun3dOnewayDriver

np.random.seed(1234567)

comm = MPI.COMM_WORLD
base_dir = os.path.dirname(os.path.abspath(__file__))
csm_path = os.path.join(base_dir, "input_files", "flow_wing.csm")

results_folder = os.path.join(base_dir, "results")
if comm.rank == 0:  # make the results folder if doesn't exist
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)


class TestFun3dOnewayShapeDriver(unittest.TestCase):
    """
    This class performs unit test on the oneway-coupled FUN3D driver
    which uses fixed struct disps or no struct disps
    TODO : in the case of an unsteady one, add methods for those too?
    """

    FILENAME = "fun3d-oneway-shape-driver.txt"
    FILEPATH = os.path.join(results_folder, FILENAME)

    def test_nominal(self):
        """test no struct disps into FUN3D"""
        # build the funtofem model with one body and scenario
        model = FUNtoFEMmodel("miniMesh")
        wing = Body.aeroelastic("plate", boundary=6).relaxation(AitkenRelaxation())
        Variable.shape(name="sspan").set_bounds(
            lower=0.4, value=1.0, upper=1.6
        ).register_to(wing)
        wing.register_to(model)
        test_scenario = Scenario.steady("laminar", steps=500).set_temperature(
            T_ref=300.0, T_inf=300.0
        )
        for var in test_scenario.variables["aerodynamic"]:
            if var.name == "AOA":
                var.active = True
        test_scenario.include(Function.ksfailure(ks_weight=50.0)).include(
            Function.lift()
        ).include(Function.drag())
        test_scenario.register_to(model)

        # design the shape
        fun3d_model = Fun3dModel.build(csm_file=csm_path, comm=comm, project_name="yes")
        aflr_aim = fun3d_model.aflr_aim

        aflr_aim.set_surface_mesh(ff_growth=1.3, min_scale=0.01, max_scale=5.0)
        aflr_aim.set_boundary_layer(initial_spacing=0.001, thickness=0.1)
        Fun3dBC.viscous(caps_group="wall", wall_spacing=0.0001).register_to(fun3d_model)
        Fun3dBC.Farfield(caps_group="Farfield").register_to(fun3d_model)
        fun3d_model.setup()

        # build the solvers and coupled driver
        solvers = SolverManager(comm)
        solvers.flow = Fun3dInterface(comm, model, fun3d_dir="meshes").set_units(
            qinf=1.0e4
        )

        driver = Fun3dOnewayDriver.nominal(solvers, model)

        # run the complex step test on the model and driver
        max_rel_error = TestResult.complex_step(
            "fun3d+tacs-laminar-aeroelastic",
            model,
            driver,
            TestFun3dOnewayShapeDriver.FILEPATH,
        )
        self.assertTrue(max_rel_error < 1e-7)

        return


if __name__ == "__main__":
    open(TestFun3dOnewayShapeDriver.FILEPATH, "w").close()
    unittest.main()
