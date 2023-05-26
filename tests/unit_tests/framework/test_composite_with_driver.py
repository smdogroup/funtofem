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

    def test_onescenario_aeroelastic(self):
        # Build the model
        model = FUNtoFEMmodel("wedge")
        plate = Body.aeroelastic("plate")
        svar = Variable.structural("thickness", value=0.1).register_to(plate)
        plate.register_to(model)
        test_scenario = Scenario.steady("test", steps=100)
        ksfailure = Function.ksfailure(ks_weight=10.0).register_to(test_scenario)
        lift = Function.lift().register_to(test_scenario)
        drag = Function.drag().register_to(test_scenario)
        random_composite = CompositeFunction.exp(ksfailure + 1.5 * lift / drag) + svar

        """
        optimize() here would set this to be included in optimization functions for 
        the optimization manager / openmdao component for funtofem
        have to call set_name() since these composite expressions have weird names
        (but only need to set the function name if it is used directly in optimization
        as the user needs to set the name for that, intermediate composites don't matter)
        """

        random_composite.set_name("my_composite").optimize().register_to(model)
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
        print("\none-scenario single-composite driver test")
        print(f"\tadjoint TD = {adjoint_TD.real}")
        print(f"\tcomplex step TD = {complex_step_TD.real}")
        print(f"\trel error = {rel_error.real}\n")

        rtol = 1e-9 if complex_mode else 1e-4
        self.assertTrue(abs(rel_error.real) < rtol)
        return

    def test_multiscenario_aeroelastic(self):
        """
        test composite functions over two scenarios with aerothermoelastic coupling and a funtofem driver
        """
        # Build the model
        model = FUNtoFEMmodel("wedge")
        plate = Body.aeroelastic("plate")
        svar = Variable.structural("thickness").set_bounds(value=0.1).register_to(plate)
        plate.register_to(model)

        # climb scenario with random test composite function 1
        cruise = Scenario.steady("cruise", steps=100)
        yaw = Variable.aerodynamic("yaw", value=10.0).register_to(cruise)
        ksfailure1 = Function.ksfailure(ks_weight=10.0).register_to(cruise)
        lift1 = Function.lift().register_to(cruise)
        drag1 = Function.drag().register_to(cruise)
        cruise.register_to(model)

        # cruise scenario with random test composite function 2
        climb = Scenario.steady("climb", steps=100)
        ksfailure2 = Function.ksfailure(ks_weight=10.0).register_to(climb)
        lift2 = Function.lift().register_to(climb)
        mach = Variable.aerodynamic("mach", value=1.45).register_to(climb)
        drag2 = Function.drag().register_to(climb)
        climb.register_to(model)

        # all composite functions must come after bodies and scenarios
        # otherwise they won't be linked to all potential design variables (which they may depend on)
        composite1 = (
            drag1
            * CompositeFunction.exp(ksfailure1 + 1.5 * lift1**2 / (1 + lift1))
            * yaw
        )
        composite1.optimize().set_name("composite1").register_to(model)
        composite2 = (
            drag2**3
            / (1.532 - ksfailure2)
            * CompositeFunction.log(1 + (lift2 / drag2) ** 2)
        ) - mach**2
        composite2.optimize().set_name("composite2").register_to(model)

        # composite function for minimum drag among two scenarios
        min_drag = CompositeFunction.boltz_min([drag1, drag2])
        min_drag.register_to(model)

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

        composite_functions = [composite1, composite2, min_drag]
        ncfunc = len(composite_functions)
        variables = model.get_variables()

        # random test vector for composite function derivatives
        dvar_ds = {var: np.random.rand() for var in model.get_variables()}
        # generate random contravariant tensor d(composite_i)/ds
        # for converting error among three composites to one scalar
        dcomposite_ds = np.random.rand(ncfunc)

        # adjoint for analysis funcs => composite function derivatives
        # adjoint evaluation of analysis function derivatives
        driver.solve_forward()
        driver.solve_adjoint()
        model.evaluate_composite_functions()

        adjoint_TD = 0.0
        for ifunc, cfunc in enumerate(composite_functions):
            adjoint_TD += cfunc.directional_derivative(dvar_ds) * dcomposite_ds[ifunc]

        # complex step directly to compute composite function values
        h = 1e-30
        for var in variables:
            var.value += 1j * h * dvar_ds[var]
        driver.solve_forward()
        model.evaluate_composite_functions()

        complex_step_TD = 0.0
        for ifunc, cfunc in enumerate(composite_functions):
            complex_step_TD += cfunc.value.imag / h * dcomposite_ds[ifunc]

        # compute the relative error between complex step + adjoint
        rel_error = (adjoint_TD - complex_step_TD) / complex_step_TD

        # report result
        print("\nmulti-scenario three-composite driver test")
        print(f"\tadjoint TD = {adjoint_TD.real}")
        print(f"\tcomplex step TD = {complex_step_TD.real}")
        print(f"\trel error = {rel_error.real}\n")

        rtol = 1e-9 if complex_mode else 1e-4
        self.assertTrue(abs(rel_error.real) < rtol)
        return


if __name__ == "__main__":
    unittest.main()
