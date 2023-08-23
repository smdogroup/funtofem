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
has_fun3d = fun3d_loader is not None

if has_fun3d:
    from funtofem.driver import Fun3dOnewayDriver
    from funtofem.interface import Fun3dInterface

np.random.seed(1234567)

comm = MPI.COMM_WORLD
base_dir = os.path.dirname(os.path.abspath(__file__))
csm_path = os.path.join(base_dir, "meshes", "naca_wing.csm")

analysis_file = os.path.join(base_dir, "run_fun3d_analysis.py")
fun3d_dir = os.path.join(base_dir, "meshes")

results_folder = os.path.join(base_dir, "results")
if comm.rank == 0:  # make the results folder if doesn't exist
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)


class TestFun3dOnewayMorph(unittest.TestCase):
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
        model = FUNtoFEMmodel("wing")
        # design the shape
        fun3d_model = Fun3dModel.build_morph(
            csm_file=csm_path, comm=comm, project_name="funtofem_CAPS"
        )
        aflr_aim = fun3d_model.aflr_aim
        fun3d_aim = fun3d_model.fun3d_aim

        aflr_aim.set_surface_mesh(ff_growth=1.4, mesh_length=5.0)
        Fun3dBC.inviscid(caps_group="wall").register_to(fun3d_model)
        farfield = Fun3dBC.Farfield(caps_group="Farfield").register_to(fun3d_model)
        aflr_aim.mesh_sizing(farfield)
        fun3d_model.setup()
        model.flow = fun3d_model

        wing = Body.aeroelastic("wing", boundary=2)
        Variable.shape(name="aoa").set_bounds(
            lower=-1.0, value=0.0, upper=1.0
        ).register_to(wing)
        wing.register_to(model)
        test_scenario = (
            Scenario.steady("turbulent", steps=5000)  # 5000
            .set_temperature(T_ref=300.0, T_inf=300.0)
            .fun3d_project(fun3d_aim.project_name)
        )
        test_scenario.adjoint_steps = 4000  # 4000
        # test_scenario.get_variable("AOA").set_bounds(value=2.0)

        test_scenario.include(Function.lift()).include(Function.drag())
        test_scenario.register_to(model)

        # build the solvers and coupled driver
        solvers = SolverManager(comm)
        solvers.flow = Fun3dInterface(
            comm, model, fun3d_dir="meshes", auto_coords=False
        )

        # analysis driver for mesh morphing
        driver = Fun3dOnewayDriver.aero_morph(solvers, model)

        # run the complex step test on the model and driver
        max_rel_error = TestResult.finite_difference(
            "fun3d+oneway-morph-turbulent-aeroelastic",
            model,
            driver,
            TestFun3dOnewayMorph.FILEPATH,
            epsilon=1e-4,
            both_adjoint=False,  # have to run adjoint twice to read funcVals from sensFile
        )
        self.assertTrue(max_rel_error < 1e-4)

        return


if __name__ == "__main__":
    if comm.rank == 0:
        open(TestFun3dOnewayMorph.FILEPATH, "w").close()
    unittest.main()
