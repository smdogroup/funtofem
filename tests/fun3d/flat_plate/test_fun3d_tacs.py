import numpy as np, unittest
from mpi4py import MPI

# from funtofem import TransferScheme
from pyfuntofem.model import FUNtoFEMmodel, Variable, Scenario, Body, Function
from pyfuntofem.interface import Fun3dInterface, TacsSteadyInterface, SolverManager
from pyfuntofem.driver import FUNtoFEMnlbgs, TransferSettings

np.random.seed(1234567)


class TestFun3dTacs(unittest.TestCase):
    def _complex_step_check(self, model, driver):

        # make sure the flow is real
        driver.solvers.make_flow_real()

        # solve the adjoint
        driver.solve_forward()
        driver.solve_adjoint()
        gradients = model.get_function_gradients()
        adjoint_TD = gradients[0][0].real

        # switch to complex flow
        driver.solvers.make_flow_complex()

        # perform complex step method
        epsilon = 1e-30
        rtol = 1e-9
        variables = model.get_variables()
        variables[0].value = variables[0].value + 1j * epsilon
        driver.solve_forward()
        functions = model.get_functions()
        complex_TD = functions[0].value.imag / epsilon

        # compute rel error between adjoint & complex step
        rel_error = (adjoint_TD - complex_TD) / complex_TD
        rel_error = rel_error.real

        print("Complex step TD  = ", complex_TD)
        print("Adjoint TD      = ", adjoint_TD)
        print("Relative error        = ", rel_error)

        self.assertTrue(abs(rel_error) < rtol)

    def test_laminar_aeroelastic(self):
        # build the funtofem model with one body and scenario
        model = FUNtoFEMmodel("plate")
        plate = Body.aeroelastic("plate", boundary=6)
        Variable.structural("thickness").set_bounds(
            lower=0.001, value=0.1, upper=2.0
        ).register_to(plate)
        plate.register_to(model)
        test_scenario = Scenario.steady("laminar", steps=100).set_temperature(
            T_ref=300.0, T_inf=300.0
        )
        test_scenario.include(Function.ksfailure(ks_weight=50.0))
        test_scenario.register_to(model)

        # build the solvers and coupled driver
        comm = MPI.COMM_WORLD
        solvers = SolverManager(comm)
        solvers.flow = Fun3dInterface(comm, model).set_units(qinf=1.0e2)
        solvers.structural = TacsSteadyInterface.create_from_bdf(
            model, comm, nprocs=1, bdf_file="nastran_CAPS.dat"
        )
        transfer_settings = TransferSettings(npts=5)
        driver = FUNtoFEMnlbgs(
            solvers, transfer_settings=transfer_settings, model=model
        )

        # run the complex step test on the model and driver
        self._complex_step_check(model, driver)

    def _laminar_aerothermal(self):
        # build the funtofem model with one body and scenario
        model = FUNtoFEMmodel("plate")
        plate = Body.aerothermal("plate", boundary=1)
        Variable.structural("thickness").set_bounds(
            lower=0.001, value=0.01, upper=2.0
        ).register_to(plate)
        plate.register_to(model)
        test_scenario = Scenario.steady("laminar", steps=100).set_temperature(
            T_ref=300.0, T_inf=300.0
        )
        test_scenario.include(Function.temperature())
        test_scenario.register_to(model)

        # build the solvers and coupled driver
        comm = MPI.COMM_WORLD
        solvers = SolverManager(comm)
        solvers.flow = Fun3dInterface(comm, model)
        solvers.structural = TacsSteadyInterface.create_from_bdf(
            model, comm, nprocs=1, bdf_file="nastran_CAPS.dat"
        )
        transfer_settings = TransferSettings(npts=5)
        driver = FUNtoFEMnlbgs(
            solvers, transfer_settings=transfer_settings, model=model
        )

        # run the complex step test on the model and driver
        self._complex_step_check(model, driver)

    def _laminar_aerothermoelastic(self):
        # build the funtofem model with one body and scenario
        model = FUNtoFEMmodel("plate")
        plate = Body.aerothermoelastic("plate", boundary=1)
        Variable.structural("thickness").set_bounds(
            lower=0.001, value=0.01, upper=2.0
        ).register_to(plate)
        plate.register_to(model)
        test_scenario = Scenario.steady("laminar", steps=100).set_temperature(
            T_ref=300.0, T_inf=300.0
        )
        test_scenario.include(Function.ksfailure())
        test_scenario.register_to(model)

        # build the solvers and coupled driver
        comm = MPI.COMM_WORLD
        solvers = SolverManager(comm)
        solvers.flow = Fun3dInterface(comm, model).set_units(qinf=1.0e2)
        solvers.structural = TacsSteadyInterface.create_from_bdf(
            model, comm, nprocs=1, bdf_file="nastran_CAPS.dat"
        )
        transfer_settings = TransferSettings(npts=5)
        driver = FUNtoFEMnlbgs(
            solvers, transfer_settings=transfer_settings, model=model
        )

        # run the complex step test on the model and driver
        self._complex_step_check(model, driver)


if __name__ == "__main__":
    unittest.main()
