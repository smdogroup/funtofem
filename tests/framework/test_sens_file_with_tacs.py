import numpy as np
from mpi4py import MPI
from funtofem import TransferScheme
from pyfuntofem.funtofem_model import FUNtoFEMmodel
from pyfuntofem.variable import Variable
from pyfuntofem.scenario import Scenario
from pyfuntofem.body import Body
from pyfuntofem.function import Function
from pyfuntofem.test_solver import TestAerodynamicSolver
from pyfuntofem.funtofem_nlbgs_driver import FUNtoFEMnlbgs
from pyfuntofem.tacs_interface import createTacsInterfaceFromBDF
from bdf_test_utils import generateBDF, thermoelasticity_callback
import unittest
import traceback


class SensitivityFileTest(unittest.TestCase):
    def _setup_model_and_driver(self):

        # Generate the BDF file if required
        bdf_file = "test_bdf_file.bdf"
        generateBDF(bdf_file)

        # Build the model
        model = FUNtoFEMmodel("wedge")
        plate = Body("plate", "aerothermoelastic", group=0, boundary=1)

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

        solvers["structural"] = createTacsInterfaceFromBDF(
            model, comm, nprocs, bdf_file, callback=thermoelasticity_callback
        )
        solvers["flow"] = TestAerodynamicSolver(comm, model)

        tacs_comm = solvers["structural"].tacs_comm

        # L&D transfer options
        transfer_options = {
            "analysis_type": "aerothermoelastic",
            "scheme": "meld",
            "thermal_scheme": "meld",
            "npts": 5,
        }

        # instantiate the driver
        driver = FUNtoFEMnlbgs(
            solvers, comm, tacs_comm, 0, comm, 0, transfer_options, model=model
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

        comm = MPI.COMM_WORLD
        pass_ = True
        try:
            model.write_sensitivity_file(comm, "struct.sens", discipline="structural")
        except:
            print(traceback.format_exc())
            pass_ = False

        try:
            model.write_sensitivity_file(comm, "aero.sens", discipline="aerodynamic")
        except:
            print(traceback.format_exc())
            pass_ = False

        assert pass_

        return


if __name__ == "__main__":
    test = SensitivityFileTest()
    test.test_sens_file()
