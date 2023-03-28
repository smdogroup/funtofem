import os, numpy as np, unittest
from tacs import TACS
from mpi4py import MPI
from funtofem import TransferScheme

from pyfuntofem.model import (
    FUNtoFEMmodel,
    Variable,
    Scenario,
    Body,
    Function,
    CompositeFunction,
)
from pyfuntofem.interface import (
    TestAerodynamicSolver,
    TacsSteadyInterface,
    SolverManager,
    TestResult,
)
from pyfuntofem.driver import FUNtoFEMnlbgs, TransferSettings
from bdf_test_utils import elasticity_callback

np.random.seed(123456)

base_dir = os.path.dirname(os.path.abspath(__file__))
bdf_filename = os.path.join(base_dir, "input_files", "test_bdf_file.bdf")

comm = MPI.COMM_WORLD
ntacs_procs = 1
complex_mode = TransferScheme.dtype == complex and TACS.dtype == complex


@unittest.skipIf(not complex_mode, "only available in complex step test")
class CompositeFunctionDriverTest(unittest.TestCase):
    FILENAME = "testaero-tacs-unsteady.txt"

    def test_aeroelastic(self):
        # Build the model
        model = FUNtoFEMmodel("wedge")
        plate = Body.aeroelastic("plate")
        svar = Variable.structural("thickness").set_bounds(value=0.1).register_to(plate)
        plate.register_to(model)
        test_scenario = Scenario.steady("test", steps=100)
        ksfailure = Function.ksfailure(ks_weight=10.0).register_to(test_scenario)
        lift = Function.lift().register_to(test_scenario)
        drag = Function.drag().register_to(test_scenario)
        # random_composite = ksfailure + lift/drag
        random_composite = ksfailure + 1.5 * lift / drag
        random_composite.register_to(model)
        print(f"random composite functions = {random_composite.function_names}")
        test_scenario.register_to(model)

        solvers = SolverManager(comm)
        solvers.structural = TacsSteadyInterface.create_from_bdf(
            model,
            comm,
            ntacs_procs,
            bdf_filename,
            callback=elasticity_callback,
        )
        solvers.flow = TestAerodynamicSolver(comm, model)

        # instantiate the driver
        driver = FUNtoFEMnlbgs(
            solvers, transfer_settings=TransferSettings(npts=5), model=model
        )

        """complex step test over composite function"""
        # adjoint for analysis funcs => composite function derivatives
        # adjoint evaluation of analysis function derivatives
        driver.solve_forward()
        driver.solve_adjoint()
        model.evaluate_composite_functions()
        adjoint_TD = random_composite.get_gradient_component(svar)

        # complex step directly to compute composite function values
        h = 1e-30
        svar.value += 1j * h
        driver.solve_forward()
        model.evaluate_composite_functions()
        complex_step_TD = random_composite.value.imag / h

        rel_error = (adjoint_TD - complex_step_TD) / complex_step_TD

        # report result
        print(f"adjoint TD = {adjoint_TD.real}")
        print(f"complex step TD = {complex_step_TD.real}")
        print(f"rel error = {rel_error.real}")

        rtol = 1e-9 if complex_mode else 1e-4
        self.assertTrue(abs(rel_error.real) < rtol)
        return


if __name__ == "__main__":
    unittest.main()
