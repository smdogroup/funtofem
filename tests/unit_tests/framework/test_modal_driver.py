import os
from tacs import TACS
from mpi4py import MPI
import numpy as np
from funtofem import TransferScheme

from funtofem.model import FUNtoFEMmodel, Variable, Scenario, Body, Function
from funtofem.interface import (
    TestAerodynamicSolver,
    TacsInterface,
    TacsSteadyInterface,
    SolverManager,
    TestResult,
    make_test_directories,
)
from funtofem.driver import TransferSettings, FUNtoFEMmodalDriver

from _bdf_test_utils import elasticity_callback, thermoelasticity_callback
import unittest

np.random.seed(123456)

base_dir = os.path.dirname(os.path.abspath(__file__))
bdf_filename = os.path.join(base_dir, "input_files", "test_bdf_file.bdf")


complex_mode = TransferScheme.dtype == complex and TACS.dtype == complex
nprocs = 1
comm = MPI.COMM_WORLD
elastic_scheme = "rbf"

results_folder, output_dir = make_test_directories(comm, base_dir)


class TestModalIDFDriver(unittest.TestCase):
    FILENAME = "testmodal-idf-driver.txt"
    FILEPATH = os.path.join(results_folder, FILENAME)

    def test_modal_analysis(self):
        # Build the model
        model = FUNtoFEMmodel("wedge")
        plate = Body.aeroelastic("plate")
        Variable.structural("thickness", value=1.0).register_to(plate)
        plate.register_to(model)

        # Create a scenario to run
        Scenario.steady("test", steps=150).register_to(model)

        # Build the solver interfaces
        solvers = SolverManager(comm)
        solvers.structural = TacsInterface.create_from_bdf(
            model,
            comm,
            nprocs,
            bdf_filename,
            callback=elasticity_callback,
            output_dir=output_dir,
        )
        solvers.flow = TestAerodynamicSolver(comm, model)

        # run modal analysis now
        modal_problem = TacsSteadyInterface.create_modal_problem_from_bdf(
            model,
            comm,
            nprocs,
            bdf_filename,
            sigma=1.0,
            num_eigs=10,
            callback=elasticity_callback,
        )
        TacsSteadyInterface.make_modal_basis(
            modal_problem, num_modes=10, body=plate, output_dir=""
        )

    def test_modal_driver(self):
        # Build the model
        model = FUNtoFEMmodel("wedge")
        plate = Body.aeroelastic("plate")
        Variable.structural("thickness", value=1.0).register_to(plate)
        plate.register_to(model)

        # Create a scenario to run
        steady = Scenario.steady("test", steps=150)
        Function.ksfailure().register_to(steady)
        Function.test_aero().register_to(steady)
        steady.register_to(model)

        # Build the solver interfaces
        solvers = SolverManager(comm)
        solvers.structural = TacsInterface.create_from_bdf(
            model,
            comm,
            nprocs,
            bdf_filename,
            callback=elasticity_callback,
            output_dir=output_dir,
        )
        solvers.flow = TestAerodynamicSolver(comm, model)

        # run prelim modal analysis so that we set in the modal basis for each body
        # (in this case just one)
        modal_problem = TacsSteadyInterface.create_modal_problem_from_bdf(
            model,
            comm,
            nprocs,
            bdf_filename,
            sigma=1.0,
            num_eigs=10,
            callback=elasticity_callback,
        )
        TacsSteadyInterface.make_modal_basis(
            modal_problem, num_modes=10, body=plate, output_dir=""
        )

        # create the modal driver

        # this just tests whether it runs first..
        # TODO : test it with derivatives and functionals
        modal_driver = FUNtoFEMmodalDriver(
            solvers,
            transfer_settings=TransferSettings(npts=10, elastic_scheme=elastic_scheme),
            model=model,
        )
        modal_driver.solve_forward()
        modal_driver.solve_adjoint()

        # (
        #     solvers,
        #     transfer_settings=TransferSettings(npts=10, elastic_scheme=elastic_scheme),
        #     model=model,
        # )

        # epsilon = 1e-30 if complex_mode else 1e-5
        # rtol = 1e-9 if complex_mode else 1e-3
        # max_rel_error = TestResult.derivative_test(
        #     "tacs+testaero-aeroelastic",
        #     model,
        #     driver,
        #     self.FILEPATH,
        #     complex_mode,
        #     epsilon,
        # )
        # self.assertTrue(max_rel_error < rtol)

        return


if __name__ == "__main__":
    if comm.rank == 0:
        open(TestModalIDFDriver.FILEPATH, "w").close()  # clear file
    unittest.main()
