import numpy as np
from mpi4py import MPI
from funtofem import TransferScheme
from tacs import TACS

from funtofem.model import FUNtoFEMmodel, Variable, Scenario, Body, Function
from funtofem.interface import (
    TestAerodynamicSolver,
    TestStructuralSolver,
    SolverManager,
    TestResult,
)
from funtofem.driver import FUNtoFEMnlbgs, TransferSettings

import unittest

complex_mode = TransferScheme.dtype == complex and TACS.dtype == complex
comm = MPI.COMM_WORLD

steps = 2
# couplings = ["aeroelastic", "aerothermal", "aeorthermoelastic"]
coupling = "aeroelastic"
# DV_cases = ["structural", "aerodynamic"]
DV_case = "structural"


class TestFrameworkAdjointEqns(unittest.TestCase):
    FILENAME = "fake-solvers-drivers.txt"

    def test_adjoint_eqns(self):
        # build the funtofem model with an unsteady scenario
        model = FUNtoFEMmodel("test")
        plate = Body("plate", analysis_type=coupling)
        if DV_case == "structural":
            for iS in range(5):
                Variable.structural(f"thick{iS}").set_bounds(value=0.1).register_to(
                    plate
                )
        if DV_case == "aerodynamic":
            for iA in range(5):
                Variable.aerodynamic(f"aero{iA}").set_bounds(value=0.1).register_to(
                    plate
                )
        plate.register_to(model)
        test_scenario = Scenario.unsteady("test", steps=steps)
        # just do one function for now
        test_scenario.include(Function.ksfailure())
        # test_scenario.include(Function.lift())
        test_scenario.register_to(model)

        # build a funtofem driver
        solvers = SolverManager(comm)
        solvers.flow = TestAerodynamicSolver(comm, model)
        solvers.structural = TestStructuralSolver(comm, model)
        driver = FUNtoFEMnlbgs(
            solvers, transfer_settings=TransferSettings(npts=10), model=model
        )
        test_aero = solvers.flow
        test_struct = solvers.structural
        aero_data = test_aero.scenario_data[test_scenario.id]
        struct_data = test_struct.scenario_data[test_scenario.id]
        bodies = model.bodies
        scenario = test_scenario

        driver.solve_forward()
        plate.initialize_adjoint_variables(test_scenario)
        # solve the adjoint variables and store states at each step
        # to compare with adjoint equations
        step = 2
        plate.transfer_disps(test_scenario, time_index=step - 1, jump=True)

        test_struct.iterate_adjoint(scenario, bodies, step)
        psi_S2 = -1 * plate.get_struct_disps_ajp(scenario).copy()
        plate.transfer_loads_adjoint(scenario)
        test_aero.iterate_adjoint(scenario, bodies, step)
        plate.transfer_disps_adjoint(scenario)

        psi_L2 = -1 * plate.get_struct_loads_ajp(scenario).copy()
        psi_A2 = -1 * plate.get_aero_loads_ajp(scenario).copy()
        psi_D2 = -1 * plate.get_aero_disps_ajp(scenario).copy()

        step = 1

        # plate.transfer_disps(test_scenario, time_index=step - 1, jump=True)
        test_struct.iterate_adjoint(scenario, bodies, step)
        psi_S1 = -1 * plate.get_struct_disps_ajp(scenario).copy()
        plate.transfer_loads_adjoint(scenario)
        test_aero.iterate_adjoint(scenario, bodies, step)
        plate.transfer_disps_adjoint(scenario)

        psi_L1 = -1 * plate.get_struct_loads_ajp(scenario).copy()
        psi_A1 = -1 * plate.get_aero_loads_ajp(scenario).copy()
        psi_D1 = -1 * plate.get_aero_disps_ajp(scenario).copy()

        # check each of the equations
        resids = []
        # df/duA1 = 0
        df_duA1 = psi_D1 - aero_data.Jac1.T @ psi_A1
        resids += [np.linalg.norm(df_duA1)]

        # df/dfA1 = 0
        aero_out = np.zeros(3 * plate.aero_nnodes, dtype=plate.dtype)
        plate.transfer.applydLdfATrans(psi_L1[:, 0].copy(), aero_out)
        df_dfA1 = psi_A1 + aero_out
        resids += [np.linalg.norm(df_dfA1)]

        # df/dfS1 = 0
        df_dfS1 = psi_L1 - struct_data.Jac1.T @ psi_S1
        resids += [np.linalg.norm(df_dfS1)]

        # df/duS1 = 0
        D2_ajp = np.zeros(3 * plate.struct_nnodes, dtype=plate.dtype)
        plate.transfer.applydDduSTrans(psi_D2[:, 0].copy(), D2_ajp)
        L2_ajp = np.zeros(3 * plate.struct_nnodes, dtype=plate.dtype)
        plate.transfer.applydLduSTrans(psi_L2[:, 0].copy(), L2_ajp)
        df_duS1 = psi_S1 + D2_ajp + L2_ajp
        resids += [np.linalg.norm(df_duS1)]

        # df/duA2 = 0
        df_duA2 = psi_D2 - np.dot(aero_data.Jac1.T, psi_A2)
        resids += [np.linalg.norm(df_duA2)]

        # df/dfA2 = 0
        aero_out = np.zeros(3 * plate.aero_nnodes, dtype=plate.dtype)
        plate.transfer.applydLdfATrans(psi_L2[:, 0].copy(), aero_out)
        df_dfA2 = psi_A2 + np.expand_dims(aero_out, axis=-1)
        resids += [np.linalg.norm(df_dfA2)]

        # df/dfS2 = 0
        term2 = np.dot(struct_data.Jac1.T, psi_S2)
        partial_f_fS2 = np.expand_dims(struct_data.func_coefs1, axis=-1)
        df_dfS2 = psi_L2 - term2 + partial_f_fS2
        resids += [np.linalg.norm(df_dfS2)]

        # df/duS2 = 0
        df_duS2 = psi_S2
        resids += [np.linalg.norm(df_duS2)]

        adjoints = [
            df_duA1,
            df_dfA1,
            df_dfS1,
            df_duS1,
            df_duA2,
            df_dfA2,
            df_dfS2,
            df_duS2,
        ]
        adjoint_norms = [np.linalg.norm(_) for _ in adjoints]

        resids = [abs(_) for _ in resids]
        print(f"resids = {resids}")
        print(f"max resid = {max(resids)}")
        print(f"adjoint norms = {adjoint_norms}")
        return


if __name__ == "__main__":
    if comm.rank == 0:
        open(TestFrameworkAdjointEqns.FILENAME, "w").close()  # clear file
    unittest.main()
