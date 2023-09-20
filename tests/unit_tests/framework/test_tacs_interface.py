import os
from tacs import TACS
from mpi4py import MPI
import numpy as np
from funtofem import TransferScheme

from funtofem.model import FUNtoFEMmodel, Variable, Scenario, Body, Function
from funtofem.interface import (
    TestAerodynamicSolver,
    TacsSteadyInterface,
    TacsInterface,
    SolverManager,
    TestResult,
)
from funtofem.driver import FUNtoFEMnlbgs, TransferSettings

from _bdf_test_utils import elasticity_callback, thermoelasticity_callback
import unittest

np.random.seed(123456)

base_dir = os.path.dirname(os.path.abspath(__file__))
bdf_filename = os.path.join(base_dir, "input_files", "test_bdf_file.bdf")


complex_mode = TransferScheme.dtype == complex and TACS.dtype == complex
nprocs = 1
comm = MPI.COMM_WORLD

results_folder = os.path.join(base_dir, "results")
if comm.rank == 0:  # make the results folder if doesn't exist
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)


class TacsSteadyInterfaceTest(unittest.TestCase):
    FILENAME = "testaero-tacs-steady.txt"
    FILEPATH = os.path.join(results_folder, FILENAME)

    def test_driver_aeroelastic(self):
        # Build the model
        model = FUNtoFEMmodel("wedge")
        plate = Body.aeroelastic("plate")
        Variable.structural("thickness", value=1.0).register_to(plate)
        plate.register_to(model)

        # Create a scenario to run
        steady = Scenario.steady("test", steps=150).include(Function.ksfailure())
        steady.register_to(model)

        # Build the solver interfaces
        solvers = SolverManager(comm)
        solvers.structural = TacsSteadyInterface.create_from_bdf(
            model, comm, nprocs, bdf_filename, callback=elasticity_callback
        )
        solvers.flow = TestAerodynamicSolver(comm, model)

        # instantiate the driver
        driver = FUNtoFEMnlbgs(
            solvers, transfer_settings=TransferSettings(npts=5), model=model
        )

        epsilon = 1e-30 if complex_mode else 1e-5
        rtol = 1e-9 if complex_mode else 1e-4
        max_rel_error = TestResult.derivative_test(
            "tacs+testaero-aeroelastic",
            model,
            driver,
            TacsSteadyInterfaceTest.FILENAME,
            complex_mode,
            epsilon,
        )
        self.assertTrue(max_rel_error < rtol)

        return

    def test_driver_aerothermal(self):
        # Build the model
        model = FUNtoFEMmodel("wedge")
        plate = Body.aerothermal("plate")
        Variable.structural("thickness", value=1.0).register_to(plate)
        plate.register_to(model)

        # Create a scenario to run
        steady = Scenario.steady("test", steps=150)
        steady.include(Function.temperature()).register_to(model)

        # Build the solver interfaces
        solvers = SolverManager(comm)
        solvers.structural = TacsSteadyInterface.create_from_bdf(
            model, comm, nprocs, bdf_filename, callback=thermoelasticity_callback
        )
        solvers.flow = TestAerodynamicSolver(comm, model)

        # instantiate the driver
        driver = FUNtoFEMnlbgs(
            solvers, transfer_settings=TransferSettings(npts=5), model=model
        )

        epsilon = 1e-30 if complex_mode else 1e-4
        rtol = 1e-9 if complex_mode else 1e-3
        max_rel_error = TestResult.derivative_test(
            "tacs+testaero-aerothermal",
            model,
            driver,
            TacsSteadyInterfaceTest.FILENAME,
            complex_mode,
            epsilon,
        )
        self.assertTrue(max_rel_error < rtol)

        return

    def test_driver_aerothermoelastic(self):
        # Build the model
        model = FUNtoFEMmodel("wedge")
        plate = Body.aerothermoelastic("plate")
        Variable.structural("thickness", value=0.1).register_to(plate)
        plate.register_to(model)

        # Create a scenario to run
        steady = Scenario.steady("test", steps=150).include(Function.ksfailure())
        steady.include(Function.temperature()).register_to(model)

        # Build the solver interfaces
        solvers = SolverManager(comm)
        solvers.structural = TacsSteadyInterface.create_from_bdf(
            model, comm, nprocs, bdf_filename, callback=thermoelasticity_callback
        )
        solvers.flow = TestAerodynamicSolver(comm, model)

        # instantiate the driver
        driver = FUNtoFEMnlbgs(
            solvers, transfer_settings=TransferSettings(npts=5), model=model
        )

        epsilon = 1e-30 if complex_mode else 1e-5
        rtol = 1e-9 if complex_mode else 1e-3
        max_rel_error = TestResult.derivative_test(
            "tacs+testaero-aerothermoelastic",
            model,
            driver,
            TacsSteadyInterfaceTest.FILENAME,
            complex_mode,
            epsilon,
        )
        self.assertTrue(max_rel_error < rtol)

        return

    def test_base_interface(self):
        """test building tacs with base TacsInterface classmethod and run the driver"""
        # Build the model
        model = FUNtoFEMmodel("wedge")
        plate = Body.aeroelastic("plate")
        Variable.structural("thickness", value=0.1).register_to(plate)
        plate.register_to(model)

        # Create a scenario to run
        steady = Scenario.steady("test", steps=150).include(Function.ksfailure())
        steady.register_to(model)

        # Build the solver interfaces
        solvers = SolverManager(comm)
        solvers.structural = TacsInterface.create_from_bdf(
            model, comm, nprocs, bdf_filename, callback=elasticity_callback
        )
        solvers.flow = TestAerodynamicSolver(comm, model)

        # instantiate the driver
        driver = FUNtoFEMnlbgs(
            solvers, transfer_settings=TransferSettings(npts=5), model=model
        )

        epsilon = 1e-30 if complex_mode else 1e-5
        rtol = 1e-9 if complex_mode else 1e-4
        max_rel_error = TestResult.derivative_test(
            "tacs-base+testaero-aeroelastic",
            model,
            driver,
            TacsSteadyInterfaceTest.FILENAME,
            complex_mode,
            epsilon,
        )
        self.assertTrue(max_rel_error < rtol)

        return

    def test_backwards_compatible(self):
        """ensure the funtofem driver can be created in the old backwards compatible way"""
        # Build the model
        model = FUNtoFEMmodel("wedge")
        plate = Body("plate", "aeroelastic", group=0, boundary=1)

        # Create a structural variable
        thickness = 1.0
        svar = Variable("thickness", value=thickness, lower=0.01, upper=0.1)
        plate.add_variable("structural", svar)
        model.add_body(plate)

        # Create a scenario to run
        steps = 150
        steady = Scenario("steady", group=0, steps=steps)

        # Add a function to the scenario
        ks = Function("ksfailure", analysis_type="structural")
        steady.add_function(ks)

        # Add a function to the scenario
        temp = Function("temperature", analysis_type="structural")
        steady.add_function(temp)

        model.add_scenario(steady)

        # Instantiate the solvers we'll use here
        solvers = {}

        # Build the TACS interface
        nprocs = 1
        comm = MPI.COMM_WORLD

        solvers = SolverManager(comm)
        solvers.structural = TacsSteadyInterface.create_from_bdf(
            model, comm, nprocs, bdf_filename, callback=elasticity_callback
        )
        solvers.flow = TestAerodynamicSolver(comm, model)

        # L&D transfer options
        transfer_settings = TransferSettings(npts=5)

        # instantiate the driver
        driver = FUNtoFEMnlbgs(
            solvers, transfer_settings=transfer_settings, model=model
        )

        epsilon = 1e-30 if complex_mode else 1e-5
        rtol = 1e-9 if complex_mode else 1e-4
        max_rel_error = TestResult.derivative_test(
            "backwards-compatible-aeroelastic",
            model,
            driver,
            TacsSteadyInterfaceTest.FILENAME,
            complex_mode,
            epsilon,
        )
        self.assertTrue(max_rel_error < rtol)

        return


if __name__ == "__main__":
    if comm.rank == 0:
        open(TacsSteadyInterfaceTest.FILENAME, "w").close()  # clear file
    unittest.main()
