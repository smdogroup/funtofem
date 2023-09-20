import os
from tacs import TACS
from mpi4py import MPI
import numpy as np
from funtofem import TransferScheme

from funtofem.model import FUNtoFEMmodel, Variable, Scenario, Body, Function
from funtofem.interface import (
    TestAerodynamicSolver,
    TacsInterface,
    SolverManager,
    test_directories,
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

_, output_dir = test_directories(comm, base_dir)


class TacsInterfaceSolver(unittest.TestCase):
    def test_solvers_aeroelastic(self):
        # Build the model
        model = FUNtoFEMmodel("wedge")
        plate = Body.aeroelastic("plate")
        Variable.structural("thickness", value=1.0).register_to(plate)
        plate.register_to(model)

        # Create a scenario to run
        scenario = Scenario.steady("test", steps=150)
        Function.ksfailure().register_to(scenario)
        Function.test_aero().register_to(scenario)
        scenario.register_to(model)

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

        # instantiate the driver
        driver = FUNtoFEMnlbgs(
            solvers, transfer_settings=TransferSettings(npts=5), model=model
        )

        # Check whether to use the complex-step method or now
        complex_step = False
        epsilon = 1e-5
        rtol = 1e-4
        if TransferScheme.dtype == complex and TACS.dtype == complex:
            complex_step = True
            epsilon = 1e-30
            rtol = 1e-9

        # Manual test of the disciplinary solvers
        scenario = model.scenarios[0]
        bodies = model.bodies
        solvers = driver.solvers

        fail = solvers.flow.test_adjoint(
            "flow",
            scenario,
            bodies,
            epsilon=epsilon,
            complex_step=complex_step,
            rtol=rtol,
        )
        assert fail == False

        fail = solvers.structural.test_adjoint(
            "structural",
            scenario,
            bodies,
            epsilon=epsilon,
            complex_step=complex_step,
            rtol=rtol,
        )
        assert fail == False

        return

    def test_solvers_aerothermal(self):
        # Build the model
        model = FUNtoFEMmodel("wedge")
        plate = Body.aerothermal("plate")
        Variable.structural("thickness", value=1.0).register_to(plate)
        plate.register_to(model)

        # Create a scenario to run
        scenario = Scenario.steady("test", steps=150)
        Function.temperature().register_to(scenario)
        Function.test_aero().register_to(scenario)
        scenario.register_to(model)

        # Build the solver interfaces
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

        # instantiate the driver
        driver = FUNtoFEMnlbgs(
            solvers, transfer_settings=TransferSettings(npts=5), model=model
        )

        # Check whether to use the complex-step method or now
        complex_step = False
        epsilon = 1e-4
        rtol = 1e-4
        if TransferScheme.dtype == complex and TACS.dtype == complex:
            complex_step = True
            epsilon = 1e-30
            rtol = 1e-9

        # Manual test of the disciplinary solvers
        scenario = model.scenarios[0]
        bodies = model.bodies
        solvers = driver.solvers

        fail = solvers.flow.test_adjoint(
            "flow",
            scenario,
            bodies,
            epsilon=epsilon,
            complex_step=complex_step,
            rtol=rtol,
        )
        assert fail == False

        fail = solvers.structural.test_adjoint(
            "structural",
            scenario,
            bodies,
            epsilon=epsilon,
            complex_step=complex_step,
            rtol=rtol,
        )
        assert fail == False

        return

    def test_solvers_aerothermoelastic(self):
        # Build the model
        model = FUNtoFEMmodel("wedge")
        plate = Body.aerothermoelastic("plate")
        Variable.structural("thickness", value=1.0).register_to(plate)
        plate.register_to(model)

        # Create a scenario to run
        scenario = Scenario.steady("test", steps=150)
        Function.ksfailure().register_to(scenario)
        Function.temperature().register_to(scenario)
        Function.test_aero().register_to(scenario)
        scenario.register_to(model)

        # Build the solver interfaces
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

        # instantiate the driver
        driver = FUNtoFEMnlbgs(
            solvers, transfer_settings=TransferSettings(npts=5), model=model
        )

        # Check whether to use the complex-step method or now
        complex_step = False
        epsilon = 1e-5
        rtol = 1e-4
        if TransferScheme.dtype == complex and TACS.dtype == complex:
            complex_step = True
            epsilon = 1e-30
            rtol = 1e-9

        # Manual test of the disciplinary solvers
        scenario = model.scenarios[0]
        bodies = model.bodies
        solvers = driver.solvers

        fail = solvers.flow.test_adjoint(
            "flow",
            scenario,
            bodies,
            epsilon=epsilon,
            complex_step=complex_step,
            rtol=rtol,
        )
        assert fail == False

        fail = solvers.structural.test_adjoint(
            "structural",
            scenario,
            bodies,
            epsilon=epsilon,
            complex_step=complex_step,
            rtol=rtol,
        )
        assert fail == False

        return

    def test_functions(self):
        comm = MPI.COMM_WORLD
        model = FUNtoFEMmodel("wedge")
        plate = Body.aeroelastic("plate")
        Variable.structural("thickness").set_bounds(
            lower=0.01, value=1.0, upper=2.0
        ).register_to(plate)
        plate.register_to(model)
        test_scenario = Scenario.steady("test", steps=150).include(Function.xcom())
        test_scenario.include(Function.ycom()).include(Function.zcom())
        test_scenario.include(Function.ksfailure()).include(Function.mass())
        test_scenario.include(Function.compliance()).register_to(model)

        solvers = SolverManager(comm)
        solvers.structural = TacsInterface.create_from_bdf(
            model,
            comm,
            1,
            bdf_filename,
            callback=elasticity_callback,
            output_dir=output_dir,
        )
        solvers.flow = TestAerodynamicSolver(comm, model)

        driver = FUNtoFEMnlbgs(
            solvers, transfer_settings=TransferSettings(npts=5), model=model
        )

        driver.solve_forward()

        funcs = model.get_functions()
        print(f"x center of mass = {funcs[0].value}")
        print(f"y center of mass = {funcs[1].value}")
        print(f"z center of mass = {funcs[2].value}")
        print(f"ksfailure = {funcs[3].value}")
        print(f"structural mass = {funcs[4].value}")
        print(f"compliance = {funcs[5].value}")
        return


if __name__ == "__main__":
    unittest.main()
