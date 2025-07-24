import numpy as np
from mpi4py import MPI
from funtofem import TransferScheme

from funtofem.model import FUNtoFEMmodel, Variable, Scenario, Body, Function
from funtofem.interface import (
    TestAerodynamicSolver,
    TestStructuralSolver,
    SolverManager,
)
from funtofem.driver import FUNtoFEMnlbgs, TransferSettings

import unittest, os

comm = MPI.COMM_WORLD

# check if we're in github to run only online vs offline tests
in_github_workflow = bool(os.getenv("GITHUB_ACTIONS"))


@unittest.skipIf(in_github_workflow, "skipping test on GitHub")
class TestReloadFuntofemStates(unittest.TestCase):
    def test_reload_forward_states(self):
        # build the funtofem model with an unsteady scenario
        model = FUNtoFEMmodel("test")
        plate = Body.aerothermoelastic("plate")
        plate.register_to(model)
        test_scenario = Scenario.steady("test", steps=10)
        Function.test_struct().register_to(test_scenario)
        Function.test_aero().register_to(test_scenario)
        test_scenario.register_to(model)

        # build a funtofem driver
        solvers = SolverManager(comm)
        solvers.flow = TestAerodynamicSolver(comm, model)
        solvers.structural = TestStructuralSolver(comm, model)
        driver = FUNtoFEMnlbgs(
            solvers,
            transfer_settings=TransferSettings(npts=10),
            model=model,
            reload_funtofem_states=True,
        )

        # run coupled analysis
        driver.solve_forward()

        # print out the coupled states
        struct_disps_f = plate.struct_disps[test_scenario.id] * 1.0
        print(f"final struct disps = {struct_disps_f[:5]}")
        struct_temps_f = plate.struct_temps[test_scenario.id] * 1.0
        print(f"final struct temps = {struct_temps_f[:5]}")

        # clear the states
        plate.struct_disps[test_scenario.id] *= 0.0
        plate.struct_temps[test_scenario.id] *= 0.0
        print(f"cleared struct disps = {plate.struct_disps[test_scenario.id][:5]}")
        print(f"cleared struct temps = {plate.struct_temps[test_scenario.id][:5]}")

        # reload the states
        model.load_forward_states(comm, test_scenario)
        loaded_struct_disps = plate.struct_disps[test_scenario.id] * 1.0
        print(f"loaded struct disps = {loaded_struct_disps[:5]}")
        loaded_struct_temps = plate.struct_temps[test_scenario.id] * 1.0
        print(f"loaded struct temps = {loaded_struct_temps[:5]}")

        # compute the error in the final and reloaded states
        assert np.max(struct_disps_f) == np.max(loaded_struct_disps)
        assert np.max(struct_temps_f) == np.max(loaded_struct_temps)
        return

    def test_reload_adjoint_states(self):
        # build the funtofem model with an unsteady scenario
        model = FUNtoFEMmodel("test")
        plate = Body.aerothermoelastic("plate")
        plate.register_to(model)
        test_scenario = Scenario.steady("test", steps=10)
        Function.test_struct().register_to(test_scenario)
        Function.test_aero().register_to(test_scenario)
        test_scenario.register_to(model)

        # build a funtofem driver
        solvers = SolverManager(comm)
        solvers.flow = TestAerodynamicSolver(comm, model)
        solvers.structural = TestStructuralSolver(comm, model)
        driver = FUNtoFEMnlbgs(
            solvers,
            transfer_settings=TransferSettings(npts=10),
            model=model,
            reload_funtofem_states=True,
        )

        # run coupled analysis
        driver.solve_forward()
        driver.solve_adjoint()

        # print out the coupled states
        struct_loads_ajp_f = plate.struct_loads_ajp * 1.0
        print(f"final struct loads ajp = {struct_loads_ajp_f[:5,:]}")
        struct_flux_ajp_f = plate.struct_flux_ajp * 1.0
        print(f"final struct temps = {struct_flux_ajp_f[:5,:]}")

        # clear the states
        plate.struct_loads_ajp *= 0.0
        plate.struct_flux_ajp *= 0.0
        print(f"cleared struct loads ajp = {plate.struct_loads_ajp[:5,:]}")
        print(f"cleared struct flux ajp = {plate.struct_flux_ajp[:5,:]}")

        # reload the states
        model.load_adjoint_states(comm, test_scenario)
        loaded_struct_loads_ajp = plate.struct_loads_ajp * 1.0
        print(f"loaded struct loads ajp = {loaded_struct_loads_ajp[:5,:]}")
        loaded_struct_flux_ajp = plate.struct_flux_ajp * 1.0
        print(f"loaded struct flux ajp = {loaded_struct_flux_ajp[:5,:]}")

        # compute the error in the final and reloaded states
        assert np.max(struct_loads_ajp_f) == np.max(loaded_struct_loads_ajp)
        assert np.max(struct_flux_ajp_f) == np.max(loaded_struct_flux_ajp)
        return


if __name__ == "__main__":
    unittest.main()
