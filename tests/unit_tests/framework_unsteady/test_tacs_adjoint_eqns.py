import numpy as np, os
from mpi4py import MPI
from funtofem import TransferScheme
from tacs import TACS

from funtofem.model import FUNtoFEMmodel, Variable, Scenario, Body, Function
from funtofem.interface import (
    TestAerodynamicSolver,
    TestStructuralSolver,
    SolverManager,
    TacsUnsteadyInterface,
    TacsIntegrationSettings,
)
from funtofem.driver import FUNtoFEMnlbgs, TransferSettings
from bdf_test_utils import elasticity_callback, thermoelasticity_callback
import unittest

complex_mode = TransferScheme.dtype == complex and TACS.dtype == complex
comm = MPI.COMM_WORLD
base_dir = os.path.dirname(os.path.abspath(__file__))
bdf_filename = os.path.join(base_dir, "input_files", "test_bdf_file.bdf")
output_folder = os.path.join(base_dir, "output")
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

results_folder = os.path.join(base_dir, "results")
if comm.rank == 0:  # make the results folder if doesn't exist
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)

steps = 10  # used for Nstep case
# couplings = ["aeroelastic", "aerothermal", "aerothermoelastic"]
coupling = "aerothermoelastic"
DV_cases = ["structural", "aerodynamic"]
ntacs_procs = 1
dt = 0.01


class TestFrameworkAdjointEqns(unittest.TestCase):
    FILENAME = "framework-adjoint-eqns.txt"
    FILEPATH = os.path.join(FILENAME, results_folder)

    def test_2step_adjoint_eqns(self):
        steps = 2

        # build the funtofem model with an unsteady scenario
        model = FUNtoFEMmodel("test")
        plate = Body("plate", analysis_type=coupling)
        if "structural" in DV_cases:
            for iS in range(5):
                Variable.structural(f"thick{iS}").set_bounds(value=0.1).register_to(
                    plate
                )
        if "aerodynamic" in DV_cases:
            for iA in range(5):
                Variable.aerodynamic(f"aero{iA}").set_bounds(value=0.1).register_to(
                    plate
                )
        plate.register_to(model)
        test_scenario = Scenario.unsteady("test", steps=steps)
        # just do one function for now
        test_scenario.include(Function.ksfailure())
        # test_scenario.include(Function.lift())
        integration_settings = TacsIntegrationSettings(
            dt=dt, num_steps=test_scenario.steps
        )
        test_scenario.include(integration_settings)
        test_scenario.register_to(model)

        # build a funtofem driver
        solvers = SolverManager(comm)
        solvers.flow = TestAerodynamicSolver(comm, model)
        solvers.structural = TacsUnsteadyInterface.create_from_bdf(
            model,
            comm,
            ntacs_procs,
            bdf_filename,
            callback=elasticity_callback
            if coupling == "aeroelastic"
            else thermoelasticity_callback,
            output_dir=output_folder,
        )
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
        df_dfA1 = psi_A1 + np.expand_dims(aero_out, axis=-1)
        resids += [np.linalg.norm(df_dfA1)]

        # df/dfS1 = 0
        df_dfS1 = psi_L1 - struct_data.Jac1.T @ psi_S1
        resids += [np.linalg.norm(df_dfS1)]

        # df/duS1 = 0
        D2_ajp = np.zeros(3 * plate.struct_nnodes, dtype=plate.dtype)
        plate.transfer.applydDduSTrans(psi_D2[:, 0].copy(), D2_ajp)
        L2_ajp = np.zeros(3 * plate.struct_nnodes, dtype=plate.dtype)
        plate.transfer.applydLduSTrans(psi_L2[:, 0].copy(), L2_ajp)
        df_duS1 = psi_S1 + np.expand_dims(D2_ajp + L2_ajp, axis=-1)
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
            psi_D1,
            psi_A1,
            psi_L1,
            psi_S1,
            psi_D2,
            psi_A2,
            psi_L2,
            psi_S2,
        ]
        adjoint_norms = [np.linalg.norm(_) for _ in adjoints]
        print(f"2 step adjoint eqns")
        print(f"\tadjoint norms = {adjoint_norms}")

        resids = [abs(_) for _ in resids]
        passing_resid = [_ < 1e-9 for _ in resids]
        npassing = sum(passing_resid)
        nresid = len(resids)
        print(f"\t{npassing}/{nresid} passing residuals")
        print(f"\tresids = {resids}")
        print(f"\tmax resid = {max(resids)}")

        assert max(resids) < 1e-9
        return

    def test_Nstep_adjoint_eqns(self):
        # build the funtofem model with an unsteady scenario
        model = FUNtoFEMmodel("test")
        plate = Body("plate", analysis_type=coupling)
        if "structural" in DV_cases:
            for iS in range(5):
                Variable.structural(f"thick{iS}").set_bounds(value=0.1).register_to(
                    plate
                )
        if "aerodynamic" in DV_cases:
            for iA in range(5):
                Variable.aerodynamic(f"aero{iA}").set_bounds(value=0.1).register_to(
                    plate
                )
        plate.register_to(model)
        test_scenario = Scenario.unsteady("test", steps=steps)
        # just do one function for now
        test_scenario.include(Function.ksfailure())
        # test_scenario.include(Function.lift())
        integration_settings = TacsIntegrationSettings(
            dt=dt, num_steps=test_scenario.steps
        )
        test_scenario.include(integration_settings)
        test_scenario.register_to(model)

        # build a funtofem driver
        solvers = SolverManager(comm)
        solvers.flow = TestAerodynamicSolver(comm, model)
        solvers.structural = TacsUnsteadyInterface.create_from_bdf(
            model,
            comm,
            ntacs_procs,
            bdf_filename,
            callback=elasticity_callback
            if coupling == "aeroelastic"
            else thermoelasticity_callback,
            output_dir=output_folder,
        )
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

        plate.transfer_disps(test_scenario, time_index=steps - 1, jump=True)

        psi_D = {}
        psi_A = {}
        psi_L = {}
        psi_S = {}

        for istep in range(1, steps + 1):
            step = steps - istep + 1  # reverse step

            test_struct.iterate_adjoint(scenario, bodies, step)
            psi_S[step] = -1 * plate.get_struct_disps_ajp(scenario).copy()
            plate.transfer_loads_adjoint(scenario)
            test_aero.iterate_adjoint(scenario, bodies, step)
            plate.transfer_disps_adjoint(scenario)

            psi_L[step] = -1 * plate.get_struct_loads_ajp(scenario).copy()
            psi_A[step] = -1 * plate.get_aero_loads_ajp(scenario).copy()
            psi_D[step] = -1 * plate.get_aero_disps_ajp(scenario).copy()

        # post-analyze the adjoint equations
        resids = []
        for istep in range(1, steps + 1):
            step = steps - istep + 1
            # check each of the equations
            # df/duAi = 0
            df_duA = psi_D[step] - aero_data.Jac1.T @ psi_A[step]
            resids += [np.linalg.norm(df_duA)]

            # df/dfAi = 0
            aero_out = np.zeros(3 * plate.aero_nnodes, dtype=plate.dtype)
            plate.transfer.applydLdfATrans(psi_L[step][:, 0].copy(), aero_out)
            df_dfA = psi_A[step] + np.expand_dims(aero_out, axis=-1)
            resids += [np.linalg.norm(df_dfA)]

            # df/dfSi = 0
            df_dfS = psi_L[step] - struct_data.Jac1.T @ psi_S[step]
            if step == steps:
                df_dfS += np.expand_dims(struct_data.func_coefs1, axis=-1)
            resids += [np.linalg.norm(df_dfS)]

            # df/duSi = 0
            if step < steps:
                D2_ajp = np.zeros(3 * plate.struct_nnodes, dtype=plate.dtype)
                plate.transfer.applydDduSTrans(psi_D[step + 1][:, 0].copy(), D2_ajp)
                L2_ajp = np.zeros(3 * plate.struct_nnodes, dtype=plate.dtype)
                plate.transfer.applydLduSTrans(psi_L[step + 1][:, 0].copy(), L2_ajp)
                df_duS = psi_S[step] + np.expand_dims(D2_ajp + L2_ajp, axis=-1)
            else:
                df_duS = psi_S[step]
            resids += [np.linalg.norm(df_duS)]

        print(f"{steps} step adjoint eqns")

        resids = [abs(_) for _ in resids]
        passing_resid = [_ < 1e-9 for _ in resids]
        npassing = sum(passing_resid)
        nresid = len(resids)
        print(f"\t{npassing}/{nresid} passing residuals")
        print(f"\tresids = {resids}")
        print(f"\tmax resid = {max(resids)}")

        assert max(resids) < 1e-9
        return


if __name__ == "__main__":
    if comm.rank == 0:
        open(TestFrameworkAdjointEqns.FILENAME, "w").close()  # clear file
    unittest.main()
