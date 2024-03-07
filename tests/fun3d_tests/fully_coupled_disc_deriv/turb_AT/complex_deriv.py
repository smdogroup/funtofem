import numpy as np, unittest, importlib, os
from mpi4py import MPI

os.environ["CMPLX_MODE"] = "1"

# Imports from FUNtoFEM
from funtofem.model import (
    FUNtoFEMmodel,
    Variable,
    Scenario,
    Body,
    Function,
)
from funtofem.interface import (
    TacsSteadyInterface,
    SolverManager,
    TestResult,
    make_test_directories,
)
from funtofem.driver import FUNtoFEMnlbgs, TransferSettings

# import os

# check whether fun3d is available
fun3d_loader = importlib.util.find_spec("fun3d")
has_fun3d = fun3d_loader is not None
if has_fun3d:
    from funtofem.interface import Fun3d14Interface

np.random.seed(1234567)
comm = MPI.COMM_WORLD
base_dir = os.path.dirname(os.path.abspath(__file__))
bdf_filename = os.path.join(base_dir, "meshes", "nastran_CAPS.dat")
results_folder, output_dir = make_test_directories(comm, base_dir)


class TestFun3dTacs(unittest.TestCase):
    FILENAME = "fun3d-tacs-driver.txt"
    FILEPATH = os.path.join(results_folder, FILENAME)

    def test_alpha_turbulent_aeroelastic(self):
        # build the funtofem model with one body and scenario
        model = FUNtoFEMmodel("plate")
        plate = Body.aeroelastic("plate", boundary=2)
        plate.register_to(model)

        # build the scenario
        test_scenario = Scenario.steady("turbulent", steps=500, uncoupled_steps=10)
        test_scenario.set_temperature(T_ref=300.0, T_inf=300.0)
        Function.ksfailure(ks_weight=10.0).register_to(test_scenario)
        Function.lift().register_to(test_scenario)
        Function.drag().register_to(test_scenario)
        aoa = test_scenario.get_variable("AOA", set_active=True)
        aoa.set_bounds(lower=5.0, value=10.0, upper=15.0)
        test_scenario.set_flow_ref_vals(qinf=1.05e5)
        test_scenario.register_to(model)

        # build the solvers and coupled driver
        solvers = SolverManager(comm)
        solvers.flow = Fun3d14Interface(
            comm, model, complex_mode=True, debug=True, fun3d_dir="meshes"
        )

        solvers.structural = TacsSteadyInterface.create_from_bdf(
            model, comm, nprocs=1, bdf_file=bdf_filename, prefix=output_dir, debug=True
        )

        transfer_settings = TransferSettings(
            elastic_scheme="meld",
            npts=50,
        )
        driver = FUNtoFEMnlbgs(
            solvers,
            transfer_settings=transfer_settings,
            model=model,
        )

        # add complex perturbation to the DV
        h = 1e-30
        aoa.value += 1j * h

        # run the complex flow
        driver.solve_forward()

        print("\nComplex TD of turbulent, aeroelastic AOA deriv\n")

        functions = model.get_functions()
        for ifunc, func in enumerate(functions):
            TD = func.value.imag / h
            print(f"func {func.name} = {TD}")


if __name__ == "__main__":
    # open and close the file to reset it
    if comm.rank == 0:
        open(TestFun3dTacs.FILEPATH, "w").close()

    unittest.main()
