import os
import numpy as np
from mpi4py import MPI
from funtofem import TransferScheme

from funtofem.model import FUNtoFEMmodel, Variable, Scenario, Body, Function
from funtofem.driver import FUNtoFEMnlbgs, TransferSettings
from funtofem.interface import (
    TestAerodynamicSolver,
    TacsInterface,
    SolverManager,
    test_directories,
)

from _bdf_test_utils import generateBDF, thermoelasticity_callback
import unittest
import traceback

base_dir = os.path.dirname(os.path.abspath(__file__))
bdf_filename = os.path.join(base_dir, "input_files", "test_bdf_file.bdf")
comm = MPI.COMM_WORLD

_, output_dir = test_directories(comm, base_dir)
aero_sens_file = os.path.join(output_dir, "aero.sens")
struct_sens_file = os.path.join(output_dir, "struct.sens")


class SensitivityFileTest(unittest.TestCase):
    def _setup_model_and_driver(self):
        # Build the model
        model = FUNtoFEMmodel("wedge")
        plate = Body.aerothermoelastic("plate", boundary=1)

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

        # Build the TACS interface
        nprocs = 1

        solvers = SolverManager(comm)
        solvers.structural = TacsInterface.create_from_bdf(
            model,
            comm,
            nprocs,
            bdf_filename,
            callback=thermoelasticity_callback,
            output_dir=output_dir,
        )
        solvers.flow = TestAerodynamicSolver(comm, model)

        # L&D transfer options
        transfer_settings = TransferSettings(npts=5)

        # instantiate the driver
        driver = FUNtoFEMnlbgs(
            solvers, transfer_settings=transfer_settings, model=model
        )

        return model, driver

    def test_sens_file(self):
        model, driver = self._setup_model_and_driver()

        # Check whether to use the complex-step method or now
        complex_step = False
        epsilon = 1e-6
        rtol = 1e-5
        if TransferScheme.dtype == complex:
            complex_step = True
            epsilon = 1e-30
            rtol = 1e-9

        # Solve the forward analysis
        driver.solve_forward()
        driver.solve_adjoint()

        pass_ = True
        try:
            model.write_sensitivity_file(
                comm, struct_sens_file, discipline="structural"
            )
        except:
            print(traceback.format_exc())
            pass_ = False

        try:
            model.write_sensitivity_file(comm, aero_sens_file, discipline="aerodynamic")
        except:
            print(traceback.format_exc())
            pass_ = False

        assert pass_

        return


if __name__ == "__main__":
    unittest.main()
