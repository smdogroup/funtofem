from contextlib import contextmanager
from os import devnull
import numpy as np, unittest, importlib
import os, sys
from mpi4py import MPI

from funtofem.model import (
    FUNtoFEMmodel,
    Body,
    Scenario,
    Function,
    AitkenRelaxation,
    Variable,
)
from funtofem.interface import (
    make_test_directories,
    SolverManager,
    TacsSteadyInterface
)
from funtofem.driver import TransferSettings, FUNtoFEMnlbgs

# check whether fun3d is available
fun3d_loader = importlib.util.find_spec("fun3d")
has_fun3d = fun3d_loader is not None
if has_fun3d:
    from funtofem.interface import Fun3d14Interface
    from funtofem.interface import Fun3dThermalInterface

comm = MPI.COMM_WORLD
base_dir = os.path.dirname(os.path.abspath(__file__))
results_folder, output_dir = make_test_directories(comm, base_dir)
bdf_filename = os.path.join(base_dir, "meshes", "nastran_CAPS.dat")

@unittest.skipIf(not has_fun3d, "skipping fun3d test without fun3d")
class TestTurbulentAerothermal(unittest.TestCase):
    FILENAME = "aerothermal_test.txt"
    FILEPATH = os.path.join(results_folder, FILENAME)

    def test_aerothermal_interface(self):
        # build a funtofem model with one body and scenario
        # build the funtofem model with one body and scenario
        model = FUNtoFEMmodel("plate")
        plate = Body.aerothermal("plate", boundary=6).relaxation(AitkenRelaxation())
        Variable.structural("thickness").set_bounds(
            lower=0.001, value=0.1, upper=2.0
        ).register_to(plate)
        plate.register_to(model)
        test_scenario = Scenario.steady("turbulent2", steps=500).set_temperature(
            T_ref=300.0, T_inf=300.0
        )
        test_scenario.include(Function.temperature())
        test_scenario.set_flow_ref_vals(qinf=1.0e4)
        test_scenario.register_to(model)

        # suppress stdout during each FUN3D analysis
        # build the solvers and coupled driver
        comm = MPI.COMM_WORLD
        solvers = SolverManager(comm)
        solvers.flow = Fun3d14Interface(comm, model, debug=False, fun3d_dir="meshes")
        solvers.structural = TacsSteadyInterface.create_from_bdf(
            model, comm, nprocs=1, bdf_file=bdf_filename, prefix=output_dir
        )

        transfer_settings = TransferSettings(
            elastic_scheme="meld",
            npts=50,
        )
        
        # perform one forward + adjoint analysis with NLBGS
        driver = FUNtoFEMnlbgs(
            solvers,
            transfer_settings=transfer_settings,
            model=model,
        )
        driver.solve_forward()
        driver.solve_adjoint()

        # keep the states in the body and reuse them for a directional
        # derivative test in the Fun3d14ThermalInterface

        fun3d_therm_interface = Fun3dThermalInterface(comm, model, fun3d_dir="meshes")
        rel_err = Fun3dThermalInterface.finite_diff_test(fun3d_therm_interface)

        assert rel_err < 1e-4
        return


if __name__ == "__main__":
    unittest.main()
